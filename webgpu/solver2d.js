"use strict";
// ===========================================================================
// solver2d.js -- the 2D RMHD solver: the CPU-side k grids (`makeGrid`), the
// 2D-specific WGSL kernels (`buildShaders`) and the `Solver` class that owns
// the device buffers, the pipelines, the LSRK33 step, the forcing, the
// diagnostics readbacks and the display chains.
//
// What deliberately does NOT live here: the shared equation + display kernel
// templates (physics.js), the equation-free machinery -- FFT template, generic
// reductions, device bring-up, cards, charts, controls, presets, the frame loop
// (common.js) -- and each app's own reference vectors, UI spec and defaults
// (its inline script). The 3D solver is not here either: rmhd3d.html keeps its
// own makeGrid / buildShaders / Solver, and does not load this file.
//
// Loaded as a plain classic <script src="solver2d.js"> AFTER common.js and
// physics.js and BEFORE the app's inline script, so it may use their top-level
// names at emit time and its own three top-level names are visible to the app
// (and must not be redeclared there). Works from file://, like the two files it
// sits on: no build step, no module syntax. Written to be loaded by more than one page.
// ===========================================================================

// ---------------------------------------------------------------------------
// grid arrays (CPU side; uploaded verbatim, so the self-test compares exactly
// what the kernels read)
//   gridA[m] = (kx, ky, ksq, inv_ksq)
//   gridB[m] = (dealias, lin_L, fmask, yfac)
// ---------------------------------------------------------------------------
function makeGrid(p) {
  const nx = p.nx, ny = p.ny, nkx = nx, nky = ny / 2 + 1, nm = nkx * nky;
  const gridA = new Float32Array(nm * 4), gridB = new Float32Array(nm * 4);
  const twoPiLx = 2 * Math.PI / p.Lx, twoPiLy = 2 * Math.PI / p.Ly;
  const kunit = Math.min(twoPiLx, twoPiLy);
  const [nmin, nmax] = p.fshell;
  const cutx = nx / 3.0, cuty = ny / 3.0;
  for (let i = 0; i < nkx; i++) {
    // numpy fftfreq(nx)*nx: [0,1,...,nx/2-1, -nx/2, ..., -1]  (index nx/2 is NEGATIVE)
    const ix = (i < nx / 2) ? i : i - nx;
    const kx = ix * twoPiLx;
    for (let j = 0; j < nky; j++) {
      const ky = j * twoPiLy;
      const m = i * nky + j;
      const ksq = kx * kx + ky * ky;
      gridA[4 * m] = kx; gridA[4 * m + 1] = ky; gridA[4 * m + 2] = ksq;
      gridA[4 * m + 3] = ksq > 0 ? 1 / ksq : 0;
      const dealias = ((ix / cutx) * (ix / cutx) + (j / cuty) * (j / cuty)) < 1.0 ? 1 : 0;
      const linL = -p.diss * Math.pow(ksq, p.hyper);
      const kn = Math.sqrt(ksq) / kunit;
      const fmask = (kn >= nmin && kn < nmax) ? 1 : 0;
      const yfac = (j === 0 || j === nky - 1) ? 1 : 2;
      gridB[4 * m] = dealias; gridB[4 * m + 1] = linL; gridB[4 * m + 2] = fmask; gridB[4 * m + 3] = yfac;
    }
  }
  // forcing shell mode list + the kx-mirror partner used by the reality symmetrization
  const shell = [];
  for (let i = 0; i < nkx; i++) for (let j = 0; j < nky; j++) {
    const m = i * nky + j;
    if (gridB[4 * m + 2] > 0.5) shell.push({ m, i, j, mir: -1 });
  }
  const pos = new Map(); shell.forEach((e, k) => pos.set(e.m, k));
  for (const e of shell) {
    if (e.j === 0 || e.j === nky - 1) {
      const mm = ((nkx - e.i) % nkx) * nky + e.j;
      e.mir = pos.has(mm) ? pos.get(mm) : -1;
    }
  }
  if (shell.length === 0) throw new Error("forcing shell contains no modes on this grid");
  return { nkx, nky, nm, gridA, gridB, shell, kunit };
}

// ---------------------------------------------------------------------------
// WGSL (2D-specific kernels; the shared physics + display templates come from
// physics.js, the generic reductions and the FFT template from common.js)
// (`struct Mode`, the colormap table and the colorize kernel are shared: physics.js)
// ---------------------------------------------------------------------------
function buildShaders(g) {
  const nx = g.nx, ny = g.ny, nky = g.nky, nm = g.nm, nr = nx * ny;
  const WGF = fftWG(ny), WGC = fftWG(nx);
  const invN2 = (1 / ((nx * ny) * (nx * ny))).toExponential(12);
  const pre = `
const NX: u32 = ${nx}u;
const NY: u32 = ${ny}u;
const NKY: u32 = ${nky}u;
const NM: u32 = ${nm}u;
const NR: u32 = ${nr}u;
const INVN2: f32 = ${invN2};
struct Cfg { dx: f32, dy: f32, cfl: f32, tau: f32, epsP: f32, epsM: f32, smax: f32, pad: f32 };
`;
  // the constants object physics.js templates over (see its header): no gridZ, a
  // flat mode index, real-space kernels at 64 threads, the display buffer is the
  // whole domain. `sigR` = this app's field list offers the residual energy sigma_r
  // (display mode 9), so its two display kernels know that mode; the 3D app does not.
  // `band` = the display k_perp filter's unit, the box k1 (ISO_PLAN D): present = prepDisp
  // emits the filter, absent = it emits the pre-filter text, byte for byte.
  // `shift` = this app offers the per-card DISPLAY OFFSET, so prepDisp emits the translation
  // phase and reads the two shift words (same gating pattern; the 3D app sets no `shift`,
  // because an offset is a picture of a plane and the 3D card's picture may be the box).
  const C = { pre, hasZ: false, wgReal: 64, nDisp: "NR", arrow: arrowDims(nx, ny),
              ns: g.shell.length, pm: g.pm, eqSrc: !!g.eqsrc, sigR: true, band: g.kunit,
              shift: true };

  const S = {};

  // ---- forward rows: real (batch,NX,NY) -> complex (batch,NX,NKY) ---------
  // the rfft row pair (common.js: both pages emit the same two kernels)
  const rows = fftRowPair(ny, nky);
  S.rowsR2C = rows.r2c;
  S.rowsC2R = rows.c2r;

  // ---- complex FFT along x (axis 0), stride NKY --------------------------
  const colsDecl = `@group(0) @binding(0) var<storage, read> cin: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> cout: array<vec2<f32>>;
const NX_: u32 = ${nx}u;
const NKY_: u32 = ${nky}u;`;
  const colsLoad = `  let b: u32 = line / NKY_;
  let jj: u32 = line % NKY_;
  let base: u32 = b * NX_ * NKY_ + jj;
  for (var idx: u32 = tid; idx < NX_; idx = idx + ${WGC}u) { buf[idx] = cin[base + idx * NKY_]; }`;
  S.colsFwd = fftKernel({
    N: nx, dir: -1, decl: colsDecl, load: colsLoad,
    store: `  for (var idx: u32 = tid; idx < NX_; idx = idx + ${WGC}u) { cout[base + idx * NKY_] = buf[src + idx]; }`
  });
  S.colsInv = fftKernel({
    N: nx, dir: +1, decl: colsDecl + `\nconst SCL: f32 = ${(1 / nx).toExponential(12)};`, load: colsLoad,
    store: `  for (var idx: u32 = tid; idx < NX_; idx = idx + ${WGC}u) { cout[base + idx * NKY_] = buf[src + idx] * SCL; }`
  });

  // ---- shared physics kernels (physics.js) --------------------------------
  //   prepGrads   perpendicular i*k gradients of phi, psi, vort, jpar -- one emission
  //               per chunk of this page's chunk list (physics.js GRAD_CHUNKS_2D)
  //   bracket     the two Poisson brackets in real space
  //   nlAssemble  the nonlinear RHS (dealias here and ONLY here)
  GRAD_CHUNKS_2D.forEach((ch, i) => {
    S["prepGrads" + gradChunkSuffix(GRAD_CHUNKS_2D, i)] =
      prepGradsWGSL(Object.assign({}, C, { gchunk: ch }));
  });
  S.bracket = bracketWGSL(C);
  S.nlAssemble = nlAssembleWGSL(C);

  // ---- add the (already normalized) Elsasser forcing ----------------------
  S.forcingAdd = pre + `
@group(0) @binding(0) var<storage, read> frc: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> sc: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs: array<vec2<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m: u32 = gid.x;
  if (m >= NM) { return; }
  let fp: vec2<f32> = frc[m] * sc[4];
  let fm: vec2<f32> = frc[NM + m] * sc[5];
  rhs[m]      = rhs[m] + 0.5 * (fp + fm);
  rhs[NM + m] = rhs[NM + m] + 0.5 * (fp - fm);
}`;

  // ---- maintain the equilibrium flux (REFINE_PLAN J2.3) -------------------
  // psi_eq,k, extracted ONCE per Reset from the k_y = 0 column of the uploaded IC --
  // which IS the equilibrium, every equilibrium seed here being y-INDEPENDENT and so
  // living entirely in that column --
  // and read by nlAssemble as the static source -lin_L*psi_eq,k. Taking it from the
  // solver's own fields means it is the dealiased, fp32 equilibrium the run actually
  // holds, not a CPU re-derivation of it. Emitted only when the preset asks.
  if (g.eqsrc) S.srcInit = pre + `
@group(0) @binding(0) var<storage, read> fields: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> eqk: array<vec2<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m: u32 = gid.x;
  if (m >= NM) { return; }
  eqk[m] = select(vec2<f32>(0.0, 0.0), fields[NM + m], (m % NKY) == 0u);
}`;

  // ---- LSRK33 stage update (integrating factor) ---------------------------
  // The 2D linear operator is DIAGONAL per field, so phi and psi may carry different
  // dissipation (REFINE_PLAN J.1 / J2.6): _dissLin substitutes Pm = nu/eta into the
  // exponent for the phi half of the stack (the diss slider being eta). At Pm 1 the
  // expression is the bare gridB[m].y this kernel has always had -- byte for byte.
  S.stage = pre + `
struct Stage { alpha: f32, beta: f32, gamma: f32, pad: f32 };
@group(0) @binding(0) var<storage, read_write> fields: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> delta: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> rhs: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> gridB: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> sc: array<f32>;
@group(0) @binding(5) var<uniform> st: Stage;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  if (idx >= 2u * NM) { return; }
  let m: u32 = select(idx, idx - NM, idx >= NM);
  let dt: f32 = sc[0];
  let E: f32 = exp((${_dissLin(C, "gridB[m].y")} * dt) * st.gamma);
  let d: vec2<f32> = st.alpha * delta[idx] + dt * rhs[idx];
  let dn: vec2<f32> = E * d;
  delta[idx] = dn;
  fields[idx] = E * fields[idx] + st.beta * dn;
}`;

  // ---- CFL max reduction (generic) ----------------------------------------
  const nPartR = Math.ceil(nr / 1024);
  S.cflPartial = cflPartialWGSL(pre);
  S.cflFinal = cflFinalWGSL(pre, nPartR);

  // ---- energy + dissipation rate -----------------------------------------
  const nPartM = Math.ceil(nm / 1024);
  S.energyPartial = energyPartialWGSL(C);
  S.energyFinal = energyFinalWGSL(pre, nPartM);

  // ---- shell-binned perpendicular energy spectra --------------------------
  // One workgroup per bin (dispatch NB), 256 threads striding all NM modes; the same
  // 0.5*ksq*|f|^2*yfac*INVN2 weight as energyPartial/energyFinal, so sum(bins) == E.
  // No atomics: every workgroup owns exactly one output bin and writes it once.
  //
  // THREE binned quantities (REFINE_PLAN H.1): [E_u | E_b | H_c]. The Elsasser spectra
  // are E+-(k) = E_u(k) + E_b(k) +- H_c(k) -- one extra accumulator lane in the SAME
  // kernel rather than a forked one, exactly as in energyPartial, and summing over k
  // reproduces sc[2]+sc[3]+-sc[8]. The accumulator is a vec4 (16-byte stride either
  // way) with the last lane unused.
  const nb = nbins(nx, ny, g.Lx, g.Ly);
  S.spectrum = pre + `
@group(0) @binding(0) var<storage, read> fields: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> gridA: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gridB: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> bins: array<f32>;
var<workgroup> sh: array<vec4<f32>, 256>;
const NB: u32 = ${nb}u;
const INVKU: f32 = ${(1 / g.kunit).toExponential(12)};
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid: u32 = lid.x;
  let bin: u32 = wgid.x;
  var acc: vec4<f32> = vec4<f32>(0.0);
  for (var m: u32 = tid; m < NM; m = m + 256u) {
    let A: vec4<f32> = gridA[m];
    let b: u32 = u32(round(sqrt(A.z) * INVKU));
    if (b == bin) {
      let w: f32 = 0.5 * A.z * gridB[m].w * INVN2;
      let phi: vec2<f32> = fields[m];
      let psi: vec2<f32> = fields[NM + m];
      acc = acc + vec4<f32>(w * dot(phi, phi), w * dot(psi, psi), 2.0 * w * dot(phi, psi), 0.0);
    }
  }
  sh[tid] = acc;
  workgroupBarrier();
  var stride: u32 = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { sh[tid] = sh[tid] + sh[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) { bins[bin] = sh[0].x; bins[NB + bin] = sh[0].y; bins[2u*NB + bin] = sh[0].z; }
}`;

  // ---- t += dt, and the time integral of the dissipation rate -------------
  S.tick = tickWGSL(pre);

  // ---- forcing: OU update on the shell modes, then the power normalization -
  S.ou = ouWGSL(C);
  S.scale = scaleWGSL(C);
  // ---- forcing, blob mode: the k-space transform of the placed gaussians --
  // emitted at every preset whether or not the page ever turns blob mode on, so the
  // kernel text is a fixed function of the grid; BLOB_FORCE_MAX is a compile-time
  // constant and never a UI number (physics.js)
  S.blobBuild = blobBuildWGSL(Object.assign({}, C, { nblob: BLOB_FORCE_MAX }));

  // ---- initial condition: dealias the transformed IC ----------------------
  S.icFinish = icFinishWGSL(C);

  // ---- display chain (physics.js): pick the field, collapse vector modes to
  // a magnitude, subsample for the arrows, gather the cut line, colorize ----
  S.prepDisp = prepDispWGSL(C);
  S.vecMag = vecMagWGSL(C);
  S.vecGather = vecGatherWGSL(C);
  S.colorize = colorizeWGSL(C);
  // the contour overlay's level table (physics.js): the potential plane itself rides the
  // display chain's own kernels, only its adapting level spacing needs one of its own
  S.contLevel = contLevelWGSL(C);
  // the cut chart's own prep (physics.js): (u_x, u_y, b_x, b_y) along x = Lx/2,
  // still in ky -- rowsC2R turns the four rows into real lines
  S.cutPrep = cutPrepWGSL(C);
  // the eigenfunction chart's own prep (physics.js, EIGF_PLAN): the k_y = j0 column of
  // (phi, psi) in kx, gathered into 2*NX complex numbers. 2D only -- the equilibria are,
  // and so is the card (`avail: cfg => !cfg.zslice`), so the 3D page emits nothing.
  S.eigfGather = eigfGatherWGSL(C);
  S.render = RENDER_WGSL;
  // sigma_c extras: |z+-|^2, the floor's max reduction, and the ratio
  S.vecMagSq = vecMagSqWGSL(C);
  S.maxSumPartial = maxSumPartialWGSL(C);
  S.sigmaCombine = sigmaCombineWGSL(C);
  // generic autoscale reduction (common.js)
  S.maxPartial = maxPartialWGSL(pre, "NR");
  S.maxFinal = maxFinalWGSL(pre, nPartR);

  return S;
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------
class Solver {
  constructor(device, p) {
    this.device = device;
    // `pm` is the magnetic Prandtl number nu/eta and `eqsrc` the maintained-equilibrium
    // source (REFINE_PLAN J.1 / J2.3, J2.6); pm = 1 with eqsrc off -- the default every
    // other caller gets, the self-test included -- is the historical scalar-dissipation
    // code path exactly, kernel text and all.
    this.p = Object.assign({
      nx: 256, ny: 256, Lx: 2 * Math.PI, Ly: 2 * Math.PI, diss: 1e-13, hyper: 4, pm: 1,
      eqsrc: false,
      epsP: 0.15, epsM: 0.15, tau: 1.0, fshell: [1, 3], cfl: 0.4, smax: 1e4, seed: 7
    }, p || {});
    const q = this.p;
    const grid = makeGrid(q);
    this.g = Object.assign({ nx: q.nx, ny: q.ny, Lx: q.Lx, Ly: q.Ly, pm: q.pm,
                             eqsrc: q.eqsrc }, grid);
    this.nr = q.nx * q.ny;
    this.ns = grid.shell.length;
    this.nPartR = Math.ceil(this.nr / 1024);
    this.nPartM = Math.ceil(this.g.nm / 1024);
    this.nb = nbins(q.nx, q.ny, q.Lx, q.Ly);
    const ad = arrowDims(q.nx, q.ny);
    this.nax = ad.nax; this.nay = ad.nay;
    this.narrow = this.nax * this.nay;
    this.disp = [];                // display chains, one per display card (built on demand)
    this.nsteps = 0;
    this.rng = new Gauss(q.seed);
    this._noise = new Float32Array(2 * this.ns * 2);
    this._raw = new Float32Array(2 * this.ns * 2);
    // blob forcing (BLOBFORCE): off unless the page asks, so the default solver -- the
    // self-test's included -- steps the OU path exactly as before. `_blobs` is the packed
    // upload, 4 floats (x0, y0, sigma, w) per slot, z+ slots then z-.
    this.blobMode = false;
    this._blobs = new Float32Array(2 * BLOB_FORCE_MAX * 4);
    this._buildBuffers();
    this._buildPipelines();
    this.chain(0);
    this.uploadGrid();
    this.uploadCfg();
    this.setIC();
  }

  _buildBuffers() {
    const d = this.device, nm = this.g.nm, nr = this.nr;
    const cx = nm * 8; // one complex field
    const gcx = 2 * Math.max.apply(null, GRAD_CHUNKS_2D.map(ch => ch.length)); // lanes per chunk
    this.buf = {
      fields: d.createBuffer({ size: 2 * cx, usage: SQ }),
      delta: d.createBuffer({ size: 2 * cx, usage: SQ }),
      rhs: d.createBuffer({ size: 2 * cx, usage: SQ }),
      // the gradient chain transforms one CHUNK at a time, so the k-space stack and the
      // column pass's target hold the widest chunk's lanes (GRAD_CHUNKS_2D)
      gradsK: d.createBuffer({ size: gcx * cx, usage: SQ }),
      specTmp: d.createBuffer({ size: gcx * cx, usage: SQ }),
      nlk: d.createBuffer({ size: 2 * cx, usage: SQ }),
      forcing: d.createBuffer({ size: 2 * cx, usage: SQ }),
      gridA: d.createBuffer({ size: nm * 16, usage: SQ }),
      gridB: d.createBuffer({ size: nm * 16, usage: SQ }),
      realGrads: d.createBuffer({ size: 8 * nr * 4, usage: SQ }),
      realNL: d.createBuffer({ size: 2 * nr * 4, usage: SQ }),
      // 12 f32: [dt, t, Ekin, Emag, s+, s-, Ddot, Dint, H_c] (H_c added in Phase H)
      scalars: d.createBuffer({ size: 48, usage: SQ }),
      shell: d.createBuffer({ size: Math.max(4, this.ns * 4), usage: SQ }),
      noise: d.createBuffer({ size: Math.max(16, 2 * this.ns * 8), usage: SQ }),
      // blob forcing: 2 * BLOB_FORCE_MAX vec4 (x0, y0, sigma, w), z+ half then z-
      blobs: d.createBuffer({ size: Math.max(16, 2 * BLOB_FORCE_MAX * 16), usage: SQ }),
      cflPart: d.createBuffer({ size: this.nPartR * 8, usage: SQ }),
      enPart: d.createBuffer({ size: this.nPartM * 16, usage: SQ }),
      specBins: d.createBuffer({ size: 3 * this.nb * 4, usage: SQ }),
      // the cut chart's own line prep: 4 rows of NKY complex -> 4 real lines of NY
      cutK: d.createBuffer({ size: 4 * this.g.nky * 8, usage: SQ }),
      cutR: d.createBuffer({ size: 4 * this.g.ny * 4, usage: SQ }),
      // the eigenfunction chart's column gather (EIGF_PLAN): 2*NX complex (phi then psi),
      // 8 kB at nx = 512, plus the 16-byte uniform carrying its k_y bin
      eigfK: d.createBuffer({ size: 2 * this.g.nx * 8, usage: SQ }),
      eigfU: d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      cfg: d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      stage: [0, 1, 2].map(() => d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }))
    };
    // psi_eq,k for the maintained-flux source: only the equilibrium path has one (J2.3)
    if (this.p.eqsrc) this.buf.eqk = d.createBuffer({ size: cx, usage: SQ });
    for (let s = 0; s < 3; s++) {
      d.queue.writeBuffer(this.buf.stage[s], 0,
        new Float32Array([LSRK33.alpha[s], LSRK33.beta[s], LSRK33.gamma[s], 0]));
    }
  }

  // ---- display chains -----------------------------------------------------
  // One chain = the scratch, bind groups, Mode uniforms, arrow-gather target
  // and output texture of ONE display card. Chain 0 is built with the solver; the
  // others are built the first time a card claims them, so a one-card session pays
  // nothing for them. There is NO chain-0 special case: every chain is complete.
  //
  // Buffer reuse inside a chain (dispK/dispTmp are pure scratch, so BOTH sigma
  // halves run through them):
  //   half 0 (any mode; z+ of sigma_c, u of sigma_r): components -> dispR, dispR2
  //   half 1 (z- of sigma_c, b of sigma_r):           components -> dispR2, dispR3
  // after which magSqA collapses (dispR, dispR2) -> dispR = |z+|^2 (|u|^2) and magSqB
  // collapses (dispR2, dispR3) -> dispR2 = |z-|^2 (|b|^2). Dispatches inside one compute
  // pass are ordered, so the reuse is safe. sigma_r is the SAME four buffers: only the
  // two Mode uniforms differ, so it adds no state at all.
  //
  // The contour overlay (REFINE_PLAN I2.4) extends that discipline instead of
  // allocating: it runs LAST in the frame, when dispK2/dispTmp2/dispR2 -- and, with
  // sigma_c finished, dispR3 -- are dead in every mode, so the two potential planes of
  // the "both" selection (J2.2) land in dispR2 and dispR3 and colorize reads them from
  // there. Their only new state is the CONT_SETS tiny level tables (contMx, contB).
  _makeChain() {
    const d = this.device, nm = this.g.nm, nr = this.nr, cx = nm * 8;
    const B = {
      dispK: d.createBuffer({ size: cx, usage: SQ }),
      dispTmp: d.createBuffer({ size: cx, usage: SQ }),
      dispR: d.createBuffer({ size: nr * 4, usage: SQ }),
      // second display component (vector / sigma_c modes): its own scratch, so the
      // physics buffers are never touched by the display path
      dispK2: d.createBuffer({ size: cx, usage: SQ }),
      dispTmp2: d.createBuffer({ size: cx, usage: SQ }),
      dispR2: d.createBuffer({ size: nr * 4, usage: SQ }),
      dispR3: d.createBuffer({ size: nr * 4, usage: SQ }),
      maxPart: d.createBuffer({ size: this.nPartR * 4, usage: SQ }),
      maxVal: d.createBuffer({ size: 4, usage: SQ }),
      // per-card overlay gathers (8 kB + ny floats -- negligible next to dispR)
      arrows: d.createBuffer({ size: this.narrow * 8, usage: SQ }),
      // contour overlay, per set: the plane's max, and [range, delta, nlev, plain bg]
      // (contLevel owns range/delta, the CPU owns the rest -- and delta = 0 is how a set
      // is switched off)
      contMx: contPer(() => d.createBuffer({ size: 4, usage: SQ })),
      contB: contPer(() => d.createBuffer({ size: 16, usage: SQ })),
      // MODE_BYTES, not 16: prepDisp reads the two band ends past the historical four
      // words out of this same uniform (ISO_PLAN D)
      mode: d.createBuffer({ size: MODE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      // pinned to the sigma chain's second vector: z- for sigma_c, b for sigma_r
      // (rewritten per card by setDisplayMode -- see dispSigmaMate)
      modeM: d.createBuffer({ size: MODE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      // the FIELD EXPORT's pair (IO_PLAN item 4): plain phi and plain psi, no band, no
      // offset, no colormap. Written once below and never again -- setDisplayMode does
      // not touch them -- which is what lets an export run without disturbing whatever
      // this card is showing.
      modeX: d.createBuffer({ size: MODE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      modeX2: d.createBuffer({ size: MODE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      // pinned to each contour set's potential (phi or psi)
      modeC: contPer(() => d.createBuffer({ size: MODE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }))
    };
    // the default pin (a fresh chain is mode 0); setDisplayMode rewrites it with the
    // mate of whatever mode the card asks for, so the pair can never disagree
    d.queue.writeBuffer(B.modeM, 0, modeWords(DISP_ZMINUS, 0, 0, null));
    d.queue.writeBuffer(B.modeX, 0, modeWords(DISP_PHI, 0, 0, null));
    d.queue.writeBuffer(B.modeX2, 0, modeWords(DISP_PSI, 0, 0, null));
    const tex = d.createTexture({
      size: [this.g.nx, this.g.ny], format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
    });
    const P = this.buf;
    const bgOf = (pipe, entries) => d.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: entries.map((r, i) => ({ binding: i, resource: r.buffer ? r : { buffer: r } }))
    });
    const bg = {
      prepDisp: bgOf(this.pl.prepDisp, [P.fields, P.gridA, B.dispK, B.mode, B.dispK2]),
      prepDispM: bgOf(this.pl.prepDisp, [P.fields, P.gridA, B.dispK, B.modeM, B.dispK2]),
      // the export's two preps: phi through the chain's first component, psi through the
      // second (the same two-slot pattern contPrep uses -- each writes the OTHER
      // component's scratch as its throwaway outk2)
      prepDispX: bgOf(this.pl.prepDisp, [P.fields, P.gridA, B.dispK, B.modeX, B.dispK2]),
      prepDispX2: bgOf(this.pl.prepDisp, [P.fields, P.gridA, B.dispK2, B.modeX2, B.dispK]),
      colsInvDisp: bgOf(this.pl.colsInv, [B.dispK, B.dispTmp]),
      colsInvDisp2: bgOf(this.pl.colsInv, [B.dispK2, B.dispTmp2]),
      rowsC2RDisp: bgOf(this.pl.rowsC2R, [B.dispTmp, B.dispR]),
      rowsC2RDisp2: bgOf(this.pl.rowsC2R, [B.dispTmp2, B.dispR2]),
      // the sigma_c second half writes one buffer further along
      rowsC2RDispH1: bgOf(this.pl.rowsC2R, [B.dispTmp, B.dispR2]),
      rowsC2RDisp2H1: bgOf(this.pl.rowsC2R, [B.dispTmp2, B.dispR3]),
      vecMag: bgOf(this.pl.vecMag, [B.dispR, B.dispR2]),
      magSqA: bgOf(this.pl.vecMagSq, [B.dispR, B.dispR2]),
      magSqB: bgOf(this.pl.vecMagSq, [B.dispR2, B.dispR3]),
      maxSumPartial: bgOf(this.pl.maxSumPartial, [B.dispR, B.dispR2, B.maxPart]),
      sigmaCombine: bgOf(this.pl.sigmaCombine, [B.dispR, B.dispR2, B.maxVal]),
      maxPartial: bgOf(this.pl.maxPartial, [B.dispR, B.maxPart]),
      maxFinal: bgOf(this.pl.maxFinal, [B.maxPart, B.maxVal]),
      vecGather: bgOf(this.pl.vecGather, [B.dispR, B.dispR2, B.arrows]),
      // contour prep, per set: the potential through the SAME prepDisp + inverse
      // transform the display modes use (its own Mode uniform, the second component's
      // dead scratch as the throwaway outk2), landing in dispR2 (set 0) / dispR3 (set 1),
      // then the shared max reduction into that set's level table
      contPrep: contPer(i => bgOf(this.pl.prepDisp, [P.fields, P.gridA, B.dispK2, B.modeC[i], B.dispK])),
      contMax: [bgOf(this.pl.maxPartial, [B.dispR2, B.maxPart]),
                bgOf(this.pl.maxPartial, [B.dispR3, B.maxPart])],
      contFin: contPer(i => bgOf(this.pl.maxFinal, [B.maxPart, B.contMx[i]])),
      contLev: contPer(i => bgOf(this.pl.contLevel, [B.contMx[i], B.contB[i]])),
      colorize: d.createBindGroup({
        layout: this.pl.colorize.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: B.dispR } },
                  { binding: 1, resource: { buffer: B.maxVal } },
                  { binding: 2, resource: tex.createView() },
                  { binding: 3, resource: { buffer: B.mode } },
                  { binding: 4, resource: { buffer: B.dispR2 } },
                  { binding: 5, resource: { buffer: B.contB[0] } },
                  { binding: 6, resource: { buffer: B.dispR3 } },
                  { binding: 7, resource: { buffer: B.contB[1] } }]
      }),
      render: d.createBindGroup({
        layout: this.plRender.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: this.sampler },
                  { binding: 1, resource: tex.createView() }]
      })
    };
    // the two sets' inverse transforms are the two the sigma_c chain already declares
    bg.contRows = [bg.rowsC2RDisp2, bg.rowsC2RDisp2H1];
    return { mode: 0, cube: 0, cont: [0, 0], zslice: 0, buf: B, tex, bg };
  }

  // chain ci, built on first use
  chain(ci) {
    const i = ci | 0;
    if (!this.disp[i]) this.disp[i] = this._makeChain();
    return this.disp[i];
  }
  // per-card display state, for the card system and the overlays
  modeOf(ci) { const D = this.disp[ci | 0]; return D ? D.mode : 0; }
  cubeOf(ci) { return 0; }              // the 2D app has no cube view
  zsliceOf(ci) { return 0; }            // ... and no z axis

  _buildPipelines() {
    const d = this.device;
    const S = buildShaders(this.g);
    const mod = shaderModuleFactory(d);
    const cp = (code, name) => d.createComputePipeline({
      layout: "auto", compute: { module: mod(code, name), entryPoint: "main" }
    });
    this.pl = {
      rowsR2C: cp(S.rowsR2C, "rowsR2C"), rowsC2R: cp(S.rowsC2R, "rowsC2R"),
      colsFwd: cp(S.colsFwd, "colsFwd"), colsInv: cp(S.colsInv, "colsInv"),
      prepGrads: GRAD_CHUNKS_2D.map((ch, i) => {
        const n = "prepGrads" + gradChunkSuffix(GRAD_CHUNKS_2D, i);
        return cp(S[n], n);
      }),
      bracket: cp(S.bracket, "bracket"),
      nlAssemble: cp(S.nlAssemble, "nlAssemble"), forcingAdd: cp(S.forcingAdd, "forcingAdd"),
      stage: cp(S.stage, "stage"), cflPartial: cp(S.cflPartial, "cflPartial"),
      cflFinal: cp(S.cflFinal, "cflFinal"), energyPartial: cp(S.energyPartial, "energyPartial"),
      energyFinal: cp(S.energyFinal, "energyFinal"), tick: cp(S.tick, "tick"),
      ou: cp(S.ou, "ou"), scale: cp(S.scale, "scale"), blobBuild: cp(S.blobBuild, "blobBuild"),
      icFinish: cp(S.icFinish, "icFinish"),
      spectrum: cp(S.spectrum, "spectrum"),
      prepDisp: cp(S.prepDisp, "prepDisp"), maxPartial: cp(S.maxPartial, "maxPartial"),
      maxFinal: cp(S.maxFinal, "maxFinal"), colorize: cp(S.colorize, "colorize"),
      vecMag: cp(S.vecMag, "vecMag"), vecGather: cp(S.vecGather, "vecGather"),
      cutPrep: cp(S.cutPrep, "cutPrep"), eigfGather: cp(S.eigfGather, "eigfGather"),
      vecMagSq: cp(S.vecMagSq, "vecMagSq"),
      maxSumPartial: cp(S.maxSumPartial, "maxSumPartial"),
      sigmaCombine: cp(S.sigmaCombine, "sigmaCombine"), contLevel: cp(S.contLevel, "contLevel")
    };
    if (S.srcInit) this.pl.srcInit = cp(S.srcInit, "srcInit");
    const rmod = mod(S.render, "render");
    this.plRender = d.createRenderPipeline({
      layout: "auto",
      vertex: { module: rmod, entryPoint: "vs" },
      fragment: { module: rmod, entryPoint: "fs", targets: [{ format: canvasFormat }] },
      primitive: { topology: "triangle-list" }
    });
    this.sampler = d.createSampler({ magFilter: "linear", minFilter: "linear" });

    const B = this.buf;
    const bg = (pipe, entries) => d.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: entries.map((r, i) => ({ binding: i, resource: r.buffer ? r : { buffer: r } }))
    });
    this.bg = {
      prepGrads: this.pl.prepGrads.map(p => bg(p, [B.fields, B.gridA, B.gradsK])),
      colsInvGrads: bg(this.pl.colsInv, [B.gradsK, B.specTmp]),
      // one target per chunk: the same row kernel, its store landing in the chunk's own
      // lanes of realGrads through the binding's window (physics.js gradChunkWindow)
      rowsC2RGrads: GRAD_CHUNKS_2D.map(ch => bg(this.pl.rowsC2R,
        [B.specTmp, Object.assign({ buffer: B.realGrads }, gradChunkWindow(this.nr, ch))])),
      bracket: bg(this.pl.bracket, [B.realGrads, B.realNL]),
      rowsR2CNL: bg(this.pl.rowsR2C, [B.realNL, B.specTmp]),
      colsFwdNL: bg(this.pl.colsFwd, [B.specTmp, B.nlk]),
      colsInvNL: bg(this.pl.colsInv, [B.nlk, B.specTmp]),          // self-test roundtrip
      rowsC2RNL: bg(this.pl.rowsC2R, [B.specTmp, B.realNL]),       // self-test roundtrip
      // the maintained-flux source is a fifth binding of nlAssemble, present only when
      // the kernel declared it (REFINE_PLAN J2.3)
      nlAssemble: bg(this.pl.nlAssemble, [B.nlk, B.gridA, B.gridB, B.rhs].concat(B.eqk || [])),
      forcingAdd: bg(this.pl.forcingAdd, [B.forcing, B.scalars, B.rhs]),
      stage: [0, 1, 2].map(s => bg(this.pl.stage,
        [B.fields, B.delta, B.rhs, B.gridB, B.scalars, B.stage[s]])),
      cflPartial: bg(this.pl.cflPartial, [B.realGrads, B.cflPart]),
      cflFinal: bg(this.pl.cflFinal, [B.cflPart, B.scalars, B.cfg]),
      energyPartial: bg(this.pl.energyPartial, [B.fields, B.gridA, B.gridB, B.enPart]),
      energyFinal: bg(this.pl.energyFinal, [B.enPart, B.scalars]),
      spectrum: bg(this.pl.spectrum, [B.fields, B.gridA, B.gridB, B.specBins]),
      cutPrep: bg(this.pl.cutPrep, [B.fields, B.gridA, B.cutK]),
      cutRows: bg(this.pl.rowsC2R, [B.cutK, B.cutR]),
      // fields READ-ONLY, the column buffer as the only writable slot (EIGF_PLAN)
      eigfGather: bg(this.pl.eigfGather, [B.fields, B.eigfU, B.eigfK]),
      tick: bg(this.pl.tick, [B.scalars]),
      ou: bg(this.pl.ou, [B.forcing, B.shell, B.noise, B.scalars, B.cfg]),
      scale: bg(this.pl.scale, [B.fields, B.forcing, B.gridA, B.gridB, B.shell, B.scalars, B.cfg]),
      blobBuild: bg(this.pl.blobBuild, [B.forcing, B.gridA, B.gridB, B.blobs]),
      icFinish: bg(this.pl.icFinish, [B.nlk, B.gridB, B.fields])
    };
    if (this.pl.srcInit) this.bg.srcInit = bg(this.pl.srcInit, [B.fields, B.eqk]);
  }

  destroy() {
    const kill = b => { if (Array.isArray(b)) b.forEach(x => x.destroy()); else b.destroy(); };
    for (const k in this.buf) kill(this.buf[k]);
    for (const D of this.disp) {
      if (!D) continue;
      for (const k in D.buf) kill(D.buf[k]);
      D.tex.destroy();
    }
  }

  // ---- parameter / grid uploads ------------------------------------------
  uploadGrid() {
    const d = this.device;
    d.queue.writeBuffer(this.buf.gridA, 0, this.g.gridA);
    d.queue.writeBuffer(this.buf.gridB, 0, this.g.gridB);
    d.queue.writeBuffer(this.buf.shell, 0, Uint32Array.from(this.g.shell.map(e => e.m)));
  }
  refreshDissipation(diss, hyper) {
    this.p.diss = diss; this.p.hyper = hyper;
    const gb = this.g.gridB, ga = this.g.gridA;
    for (let m = 0; m < this.g.nm; m++) gb[4 * m + 1] = -diss * Math.pow(ga[4 * m + 2], hyper);
    this.device.queue.writeBuffer(this.buf.gridB, 0, gb);
  }
  uploadCfg() {
    const q = this.p;
    this.device.queue.writeBuffer(this.buf.cfg, 0, new Float32Array([
      q.Lx / q.nx, q.Ly / q.ny, q.cfl, q.tau, q.epsP, q.epsM, q.smax, 0]));
  }
  // ci = which display card / chain. `zslice` is accepted (and ignored) so both apps
  // present the same signature to the shared card code; `cmap` indexes CMAP_NAMES.
  // `opt.cont` is the contour sets' potentials as display modes (0 = that set off,
  // 2 = phi, 3 = psi), `opt.nlev` their level count and `opt.plain` the blank-background
  // option (J2.1/J2.2); `opt.cube` is 3D-only and ignored here.
  setDisplayMode(ci, mode, zslice, cmap, opt) {
    const D = this.chain(ci), o = opt || {};
    D.mode = mode | 0;
    D.cont = (o.cont || [0, 0]).slice();
    // `o.band` is the card's DISPLAY-ONLY k_perp band (ISO_PLAN D). It rides every uniform
    // this chain preps a field through -- the card's own, the sigma mate and the two contour
    // potentials -- so one card shows one band, contours included, and nothing else moves.
    // `o.offset` is the card's display offset and rides them exactly the same way (so the
    // contours and the sigma mate are rolled with the field, never against it), converted
    // here from the card's box fractions into the lengths prepDisp's phase wants.
    const off = o.offset || [0, 0];
    const u = modeWords(D.mode, 0, cmap, o.band,
                        [off[0] * this.p.Lx, off[1] * this.p.Ly]);
    this.device.queue.writeBuffer(D.buf.mode, 0, u);
    u[0] = dispSigmaMate(D.mode);           // the sigma second half, same colormap
                                            // (z- for sigma_c, b for sigma_r)
    this.device.queue.writeBuffer(D.buf.modeM, 0, u);
    for (let i = 0; i < CONT_SETS; i++) {   // each contour potential's own prep
      u[0] = D.cont[i] || DISP_PHI;
      this.device.queue.writeBuffer(D.buf.modeC[i], 0, u);
    }
    setContLevels(this.device, D, o.nlev, o.plain);
  }

  // ---- initial condition --------------------------------------------------
  // The solver knows exactly two initial conditions: this built-in pair (the
  // large-scale mode pattern of gen_refvectors.py, or an exactly quiescent state) and
  // "here is a real-space (phi, psi)" (setICFromReal). Everything the UI offers as an
  // "IC preset" is built on the CPU and comes in through the latter, so the self-test
  // path -- which calls setIC() / setIC(true) -- is untouched by any of it.
  setIC(zero) {
    const q = this.p, nx = q.nx, ny = q.ny;
    const r = new Float32Array(2 * nx * ny);
    if (!zero) {
      for (let i = 0; i < nx; i++) {
        const x = i * q.Lx / nx;
        for (let j = 0; j < ny; j++) {
          const y = j * q.Ly / ny;
          r[i * ny + j] = 0.1 * (Math.sin(x) * Math.cos(2 * y) + Math.cos(3 * x) * Math.sin(y));
          r[nx * ny + i * ny + j] = 0.1 * (Math.cos(2 * x) * Math.cos(y) + Math.sin(x) * Math.sin(3 * y));
        }
      }
    }
    this._uploadIC(r);
  }

  // phi, psi as real-space Float32Arrays of nx*ny each (row-major ix*ny + iy, the
  // layout of every real-space buffer here). Either may be null = identically zero.
  setICFromReal(phiReal, psiReal) {
    const n = this.nr, r = new Float32Array(2 * n);
    for (const [f, off] of [[phiReal, 0], [psiReal, n]]) {
      if (!f) continue;
      if (f.length !== n) throw new Error("setICFromReal: expected " + n + " values, got " + f.length);
      r.set(f, off);
    }
    this._uploadIC(r);
  }

  // (2*nr) real stack -> forward rfft2 -> dealias (icFinish) -> fields, then the same
  // zeroed forcing state / lagged scale / energy the constructor leaves behind. The
  // dealias is applied HERE and not only in the RHS: unmasked beyond-cutoff IC energy
  // would persist and alias (run.py::initialize).
  _uploadIC(r) {
    const d = this.device, q = this.p, nx = q.nx;
    d.queue.writeBuffer(this.buf.realNL, 0, r);
    d.queue.writeBuffer(this.buf.scalars, 0, new Float32Array(8));
    const enc = d.createCommandEncoder();
    enc.clearBuffer(this.buf.forcing);
    enc.clearBuffer(this.buf.delta);
    const p = enc.beginComputePass();
    p.setPipeline(this.pl.rowsR2C); p.setBindGroup(0, this.bg.rowsR2CNL); p.dispatchWorkgroups(2 * nx);
    p.setPipeline(this.pl.colsFwd); p.setBindGroup(0, this.bg.colsFwdNL); p.dispatchWorkgroups(2 * this.g.nky);
    p.setPipeline(this.pl.icFinish); p.setBindGroup(0, this.bg.icFinish); p.dispatchWorkgroups(Math.ceil(2 * this.g.nm / 64));
    // the maintained equilibrium is whatever this IC's k_y = 0 column is (J2.3)
    if (this.pl.srcInit) {
      p.setPipeline(this.pl.srcInit); p.setBindGroup(0, this.bg.srcInit);
      p.dispatchWorkgroups(Math.ceil(this.g.nm / 64));
    }
    // matches run._refresh_forcing_scale: the very first step uses a scale from the IC
    p.setPipeline(this.pl.scale); p.setBindGroup(0, this.bg.scale); p.dispatchWorkgroups(1);
    p.setPipeline(this.pl.energyPartial); p.setBindGroup(0, this.bg.energyPartial); p.dispatchWorkgroups(this.nPartM);
    p.setPipeline(this.pl.energyFinal); p.setBindGroup(0, this.bg.energyFinal); p.dispatchWorkgroups(1);
    p.end();
    d.queue.submit([enc.finish()]);
    // the scalars were zeroed above and `scale` has just written the OU normalization
    // into sc[4]/sc[5]; blob mode carries its amplitude in the modes themselves, so put
    // the two scales back at 1 or the first frames would force with nothing
    if (this.blobMode) d.queue.writeBuffer(this.buf.scalars, 16, new Float32Array([1, 1]));
    this.nsteps = 0;
    this.rng = new Gauss(q.seed);
  }

  // the eight perpendicular gradients, into realGrads (lanes 0,1 = grad phi, 2,3 = grad
  // psi, 4..7 = grad vorticity / current): one chunk at a time, each prepGrads writing the
  // chunk's lanes of the k-space stack and the two inverse passes carrying them to the
  // chunk's own lanes of realGrads. The only place the chain is encoded.
  encodeGrads(pass) {
    const nm = this.g.nm, nky = this.g.nky, nx = this.g.nx;
    GRAD_CHUNKS_2D.forEach((ch, i) => {
      const lanes = 2 * ch.length;
      pass.setPipeline(this.pl.prepGrads[i]); pass.setBindGroup(0, this.bg.prepGrads[i]);
      pass.dispatchWorkgroups(Math.ceil(nm / 64));
      pass.setPipeline(this.pl.colsInv); pass.setBindGroup(0, this.bg.colsInvGrads);
      pass.dispatchWorkgroups(lanes * nky);
      pass.setPipeline(this.pl.rowsC2R); pass.setBindGroup(0, this.bg.rowsC2RGrads[i]);
      pass.dispatchWorkgroups(lanes * nx);
    });
  }

  // ---- RHS + one LSRK33 step ---------------------------------------------
  encodeRHS(pass, opts) {
    const doCFL = !!(opts && opts.cfl), doForce = !(opts && opts.forcing === false);
    const nm = this.g.nm, nky = this.g.nky, nx = this.g.nx;
    this.encodeGrads(pass);
    if (doCFL) {
      pass.setPipeline(this.pl.cflPartial); pass.setBindGroup(0, this.bg.cflPartial);
      pass.dispatchWorkgroups(this.nPartR);
      pass.setPipeline(this.pl.cflFinal); pass.setBindGroup(0, this.bg.cflFinal);
      pass.dispatchWorkgroups(1);
    }
    pass.setPipeline(this.pl.bracket); pass.setBindGroup(0, this.bg.bracket);
    pass.dispatchWorkgroups(Math.ceil(this.nr / 64));
    pass.setPipeline(this.pl.rowsR2C); pass.setBindGroup(0, this.bg.rowsR2CNL);
    pass.dispatchWorkgroups(2 * nx);
    pass.setPipeline(this.pl.colsFwd); pass.setBindGroup(0, this.bg.colsFwdNL);
    pass.dispatchWorkgroups(2 * nky);
    pass.setPipeline(this.pl.nlAssemble); pass.setBindGroup(0, this.bg.nlAssemble);
    pass.dispatchWorkgroups(Math.ceil(nm / 64));
    if (doForce) {
      pass.setPipeline(this.pl.forcingAdd); pass.setBindGroup(0, this.bg.forcingAdd);
      pass.dispatchWorkgroups(Math.ceil(nm / 64));
    }
  }

  encodeStep(enc, doCFL) {
    for (let s = 0; s < 3; s++) {
      const p = enc.beginComputePass();
      this.encodeRHS(p, { cfl: doCFL && s === 0, forcing: true });
      p.setPipeline(this.pl.stage); p.setBindGroup(0, this.bg.stage[s]);
      p.dispatchWorkgroups(Math.ceil(2 * this.g.nm / 64));
      p.end();
    }
  }

  // ---- blob forcing (BLOBFORCE) -------------------------------------------
  // Blob mode REPLACES the OU shell: `blobBuild` writes the whole forcing buffer from the
  // placed gaussians, and the step dispatches neither `ou` nor `scale`, so sc[4]/sc[5]
  // stay at the 1 written here and each blob's amplitude is its own w. The forcing buffer
  // is cleared on either transition -- the modes it holds mean nothing in the other mode,
  // and with sc[4]/sc[5] at 1 a leftover OU envelope would land on the fields whole.
  setBlobMode(on) {
    const was = this.blobMode, d = this.device;
    this.blobMode = !!on;
    if (was !== this.blobMode) {              // only a real transition throws the modes away
      const enc = d.createCommandEncoder();
      enc.clearBuffer(this.buf.forcing);
      d.queue.submit([enc.finish()]);
    }
    if (this.blobMode) d.queue.writeBuffer(this.buf.scalars, 16, new Float32Array([1, 1]));
  }

  // list: [{ x, y, sigma, amp, pol }], x/y in box coordinates, sigma in the same length
  // units, `amp` the PEAK |grad f| the blob is to force at and `pol` +1 for the z+ channel
  // / -1 for z-. At most BLOB_FORCE_MAX per channel -- the rest are dropped. `amp` is a
  // velocity forcing rate, so the potential peak is icBlobPeak(amp, sigma) (common.js):
  // growing sigma then inflates the blob without changing what it forces the flow at.
  // An empty list zeroes every slot, i.e. zero forcing.
  setBlobs(list) {
    const q = this.p, a = this._blobs, n = BLOB_FORCE_MAX;
    // w = P * 2*pi*sigma^2 * (nx*ny)/(Lx*Ly): the continuous transform at k = 0, times
    // the unnormalized DFT's nx*ny and divided by the cell area (see blobBuildWGSL)
    const w0 = 2 * Math.PI * (q.nx * q.ny) / (q.Lx * q.Ly);
    a.fill(0);
    const used = [0, 0];
    for (const b of (list || [])) {
      const h = b.pol < 0 ? 1 : 0;
      if (used[h] >= n) continue;
      const o = 4 * (h * n + used[h]++), sg = +b.sigma;
      a[o] = +b.x; a[o + 1] = +b.y; a[o + 2] = sg;
      a[o + 3] = icBlobPeak(+b.amp, sg) * sg * sg * w0;
    }
  }

  drawNoise() {
    // unit noise for one OU step: (N(0,1) + i N(0,1))/sqrt(2) * (nx*ny), hermitian-
    // symmetrized on the ky=0 / ky=Nyquist columns (divide by sqrt(2), not 2).
    const ns = this.ns, shell = this.g.shell, raw = this._raw, out = this._noise;
    const gn = this.p.nx * this.p.ny, isq = 1 / Math.SQRT2;
    for (let k = 0; k < 2 * ns; k++) {
      raw[2 * k] = this.rng.next() * isq * gn;
      raw[2 * k + 1] = this.rng.next() * isq * gn;
    }
    out.set(raw);
    const nkyLast = this.g.nky - 1;
    for (let ou = 0; ou < 2; ou++) {
      for (let s = 0; s < ns; s++) {
        const e = shell[s];
        if (e.j !== 0 && e.j !== nkyLast) continue;
        const a = 2 * (ou * ns + s);
        let mr = 0, mi = 0;
        if (e.mir >= 0) { const b = 2 * (ou * ns + e.mir); mr = raw[b]; mi = raw[b + 1]; }
        out[a] = (raw[a] + mr) * isq;
        out[a + 1] = (raw[a + 1] - mi) * isq;
      }
    }
    this.device.queue.writeBuffer(this.buf.noise, 0, out);
  }

  // one full step: LSRK33 (lagged scales), then the OU advance and the new scale --
  // or, in blob mode, the blob rebuild in place of all three
  step(cflEvery) {
    const d = this.device;
    const doCFL = (this.nsteps % Math.max(1, cflEvery | 0)) === 0;
    if (this.blobMode) d.queue.writeBuffer(this.buf.blobs, 0, this._blobs);
    else this.drawNoise();
    const enc = d.createCommandEncoder();
    enc.clearBuffer(this.buf.delta);
    this.encodeStep(enc, doCFL);
    const p = enc.beginComputePass();
    p.setPipeline(this.pl.energyPartial); p.setBindGroup(0, this.bg.energyPartial);
    p.dispatchWorkgroups(this.nPartM);
    p.setPipeline(this.pl.energyFinal); p.setBindGroup(0, this.bg.energyFinal);
    p.dispatchWorkgroups(1);
    p.setPipeline(this.pl.tick); p.setBindGroup(0, this.bg.tick); p.dispatchWorkgroups(1);
    if (this.blobMode) {
      p.setPipeline(this.pl.blobBuild); p.setBindGroup(0, this.bg.blobBuild);
      p.dispatchWorkgroups(Math.ceil(2 * this.g.nm / 64));
    } else {
      p.setPipeline(this.pl.ou); p.setBindGroup(0, this.bg.ou);
      p.dispatchWorkgroups(Math.ceil(2 * this.ns / 64));
      p.setPipeline(this.pl.scale); p.setBindGroup(0, this.bg.scale); p.dispatchWorkgroups(1);
    }
    p.end();
    d.queue.submit([enc.finish()]);
    this.nsteps++;
  }

  // ---- display ------------------------------------------------------------
  // one half of a chain: prepDisp (through the chain's Mode uniform, or its pinned
  // z- one for half 1) followed by the inverse transform of the first component and,
  // for the two-component modes, of the second.
  _dispHalf(p, D, half, twoComp) {
    p.setPipeline(this.pl.prepDisp);
    p.setBindGroup(0, half ? D.bg.prepDispM : D.bg.prepDisp);
    p.dispatchWorkgroups(Math.ceil(this.g.nm / 64));
    p.setPipeline(this.pl.colsInv); p.setBindGroup(0, D.bg.colsInvDisp);
    p.dispatchWorkgroups(this.g.nky);
    p.setPipeline(this.pl.rowsC2R);
    p.setBindGroup(0, half ? D.bg.rowsC2RDispH1 : D.bg.rowsC2RDisp);
    p.dispatchWorkgroups(this.g.nx);
    if (!twoComp) return;
    p.setPipeline(this.pl.colsInv); p.setBindGroup(0, D.bg.colsInvDisp2);
    p.dispatchWorkgroups(this.g.nky);
    p.setPipeline(this.pl.rowsC2R);
    p.setBindGroup(0, half ? D.bg.rowsC2RDisp2H1 : D.bg.rowsC2RDisp2);
    p.dispatchWorkgroups(this.g.nx);
  }

  // the field export (IO_PLAN item 4): plain phi -> dispR and plain psi -> dispR2, each
  // through its own PINNED Mode uniform, so no live display uniform is written. Same
  // three stages the display uses; the psi leg runs second because its prep zeroes dispK,
  // which the phi leg has finished with by then. `readFieldPair` (common.js) owns the
  // rest -- the 3D chain's version of this is the same six lines with one stage more.
  encodeExport(p, D) {
    const legs = [[D.bg.prepDispX, D.bg.colsInvDisp, D.bg.rowsC2RDisp],
                  [D.bg.prepDispX2, D.bg.colsInvDisp2, D.bg.rowsC2RDisp2]];
    for (const leg of legs) {
      p.setPipeline(this.pl.prepDisp); p.setBindGroup(0, leg[0]);
      p.dispatchWorkgroups(Math.ceil(this.g.nm / 64));
      p.setPipeline(this.pl.colsInv); p.setBindGroup(0, leg[1]);
      p.dispatchWorkgroups(this.g.nky);
      p.setPipeline(this.pl.rowsC2R); p.setBindGroup(0, leg[2]);
      p.dispatchWorkgroups(this.g.nx);
    }
  }

  // the contour overlay's potentials (REFINE_PLAN I2.4, J2.2): ONE extra inverse
  // transform per ACTIVE set per card frame, into the now-dead display scratch (dispR2
  // for set 0, dispR3 for set 1 -- which is where colorize reads them), plus the max
  // reduction that feeds each set's adapting level spacing. Runs LAST, after every mode
  // has finished with that scratch.
  _dispCont(p, D) {
    for (let i = 0; i < CONT_SETS; i++) {
      if (!D.cont[i]) continue;
      p.setPipeline(this.pl.prepDisp); p.setBindGroup(0, D.bg.contPrep[i]);
      p.dispatchWorkgroups(Math.ceil(this.g.nm / 64));
      p.setPipeline(this.pl.colsInv); p.setBindGroup(0, D.bg.colsInvDisp2);
      p.dispatchWorkgroups(this.g.nky);
      p.setPipeline(this.pl.rowsC2R); p.setBindGroup(0, D.bg.contRows[i]);
      p.dispatchWorkgroups(this.g.nx);
      contLevelEncode(p, this, D, this.nPartR, i);
    }
  }

  render(ctx, ci) {
    const d = this.device, D = this.chain(ci || 0), mode = D.mode;
    const nrWG = Math.ceil(this.nr / 64);
    const enc = d.createCommandEncoder();
    const p = enc.beginComputePass();
    this._dispHalf(p, D, 0, dispTwoComp(mode));
    if (dispIsSigma(mode)) {
      // |z+|^2 -> dispR, then the z- half -> |z-|^2 in dispR2; the floor uses the
      // max of their sum, and colorize renders the ratio on the fixed +-1 range.
      // sigma_r runs the identical sequence on (|u|^2, |b|^2): the pair of vectors is
      // chosen by the two Mode uniforms alone (mode 9 and its mate), not here.
      p.setPipeline(this.pl.vecMagSq); p.setBindGroup(0, D.bg.magSqA);
      p.dispatchWorkgroups(nrWG);
      this._dispHalf(p, D, 1, true);
      p.setPipeline(this.pl.vecMagSq); p.setBindGroup(0, D.bg.magSqB);
      p.dispatchWorkgroups(nrWG);
      p.setPipeline(this.pl.maxSumPartial); p.setBindGroup(0, D.bg.maxSumPartial);
      p.dispatchWorkgroups(this.nPartR);
      p.setPipeline(this.pl.maxFinal); p.setBindGroup(0, D.bg.maxFinal);
      p.dispatchWorkgroups(1);
      p.setPipeline(this.pl.sigmaCombine); p.setBindGroup(0, D.bg.sigmaCombine);
      p.dispatchWorkgroups(nrWG);
    } else if (dispIsVector(mode)) {
      // subsample the two components for the overlay, then collapse dispR to
      // |(ux,uy)| for the autoscale + colormap
      p.setPipeline(this.pl.vecGather); p.setBindGroup(0, D.bg.vecGather);
      p.dispatchWorkgroups(Math.ceil(this.narrow / 64));
      p.setPipeline(this.pl.vecMag); p.setBindGroup(0, D.bg.vecMag);
      p.dispatchWorkgroups(nrWG);
    }
    if (!dispIsSigma(mode)) {          // sigma_c / sigma_r: FIXED +-1 range, no autoscale
      p.setPipeline(this.pl.maxPartial); p.setBindGroup(0, D.bg.maxPartial);
      p.dispatchWorkgroups(this.nPartR);
      p.setPipeline(this.pl.maxFinal); p.setBindGroup(0, D.bg.maxFinal);
      p.dispatchWorkgroups(1);
    }
    this._dispCont(p, D);
    p.setPipeline(this.pl.colorize); p.setBindGroup(0, D.bg.colorize);
    p.dispatchWorkgroups(Math.ceil(this.g.nx / 8), Math.ceil(this.g.ny / 8));
    p.end();
    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store"
      }]
    });
    rp.setPipeline(this.plRender); rp.setBindGroup(0, D.bg.render); rp.draw(3);
    rp.end();
    d.queue.submit([enc.finish()]);
  }

  async readStats() { return readBuf(this.device, this.buf.scalars, 48); }

  // subsampled (ux,uy) pairs for one card's arrow overlay; filled by the last render()
  // of that chain in a vector mode (<= 32*32*2 f32 = 8 kB). Throttled by the caller:
  // one frame of lag is fine, and this must not turn into a per-step submit.
  async readArrows(ci) { return readBuf(this.device, this.chain(ci).buf.arrows, this.narrow * 8); }

  // the cut chart's four component lines (u_x, u_y, b_x, b_y) along y at x = Lx/2:
  // its OWN prep, so it does not depend on which display cards exist (REFINE_PLAN H.3).
  // `iz` is accepted (and ignored) so both apps present the same signature.
  // Throttled by the caller: one small pass plus a 4*NY map round trip.
  async readCutLine(iz) { return cutLineRead(this, iz); }

  // the eigenfunction chart's column (EIGF_PLAN): (phi, psi) at k_y bin j0, still in kx,
  // as 2*NX complex numbers. One tiny dispatch plus a 4*NX-float map round trip, throttled
  // by the caller exactly as the cut line is; the inverse along kx is CPU work
  // (common.js: eigfProfile). Display-only -- nothing here writes the state.
  // 2D only, so there is no `iz` and no Mode uniform: the k_y bin is the whole selection.
  async readEigf(j0) {
    const d = this.device;
    const j = Math.max(0, Math.min(this.g.nky - 1, j0 | 0));
    d.queue.writeBuffer(this.buf.eigfU, 0, new Uint32Array([j, 0, 0, 0]));
    const enc = d.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pl.eigfGather); p.setBindGroup(0, this.bg.eigfGather);
    p.dispatchWorkgroups(Math.ceil(this.g.nx / 64));
    p.end();
    d.queue.submit([enc.finish()]);
    return readBuf(d, this.buf.eigfK, 2 * this.g.nx * 8);
  }

  // recompute + read back the shell spectra: 2*NB floats ([E_u | E_b]). Throttled by
  // the caller (this is a full pass over the fields plus a mapAsync round trip).
  async readSpectrum() {
    const d = this.device;
    const enc = d.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(this.pl.spectrum); p.setBindGroup(0, this.bg.spectrum);
    p.dispatchWorkgroups(this.nb);
    p.end();
    d.queue.submit([enc.finish()]);
    return { perp: await readBuf(d, this.buf.specBins, 3 * this.nb * 4), par: null, parKfac: 0 };
  }
}
