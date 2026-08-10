"use strict";
// ===========================================================================
// physics.js -- the RMHD physics + display WGSL that the 2D (rmhd2d.html) and
// 3D (rmhd3d.html) apps share, as templates over one constants object.
//
// Loaded as a classic <script src="physics.js"> BETWEEN common.js and each app's
// inline script, so it may call common.js helpers (reduceTail) at emit time and
// its own top-level declarations are visible to the app (and must not be
// redeclared there).
//
// ---------------------------------------------------------------------------
// The template boundary
// ---------------------------------------------------------------------------
// SHARED (here): every kernel whose only 2D/3D difference is how a flat mode
// index is decomposed -- prepGrads, bracket, nlAssemble, energyPartial, ou,
// scale, icFinish -- plus the whole slice display chain (prepDisp, vecMag,
// vecMagSq, maxSumPartial, sigmaCombine, vecGather, cutPrep, colorize,
// contLevel, the blit) and the display-mode predicates the apps and the frame
// loop branch on. The shading itself (dispX + the contour overlay) is one WGSL
// fragment, dispShade(C), so the 3D cube faces shade exactly like a slice.
// The 3D index convention is m = iz*NMP + mp; every template below
// derives ITS OWN text for both dimensions from the single flag `C.hasZ`, so
// the apps hand over sizes, not code. Where a body genuinely restructures
// (ou's block layout, scale's two-plane envelope sum, icFinish's dealias), the
// branch text sits next to its twin here rather than in the app.
//
// NOT shared (per-app, by design): `stage` (a diagonal exponential in 2D, the
// 2x2 Alfven propagator in 3D), `forcingAdd` / `envExpand` (dense shell vs the
// two kz = +-2pi/Lz planes), `spectrum` / `specReduceZ` / `specPar`, and the
// 3D-only `sliceExtract` / `faceExtract` / cube render path. The generic
// reductions (tick, cflPartial/Final, energyFinal, maxPartial/Final) and the
// FFT template stay in common.js: they carry no equation.
//
// ---------------------------------------------------------------------------
// The constants object C (built once per Solver in each app's buildShaders)
// ---------------------------------------------------------------------------
//   pre         the app's WGSL constant block (NX, NY, [NZ,] NM, NR, INVN2, Cfg)
//   hasZ        3D: a gridZ (kz) array exists -> the kz dealias factor, the kz^4
//               damping in the linear diagonal, and the m -> (iz, mp) split
//   wgReal      workgroup size of the real-space kernels (64 in 2D, 256 in 3D)
//   nDisp       element count of the DISPLAYED buffer: "NR" in 2D (the whole
//               domain), "NRS" in 3D (one z slice). The 3D app also instantiates
//               the elementwise display templates at "NFACE" (the three cube
//               faces, packed back to back) -- same kernels, other length.
//   arrow       arrowDims() result for the arrow-overlay subsample
//   ns          forcing-shell size (NS)
//   envFn       3D only: the `envelope()` helper, prepended to `scale` (the app
//               uses the same text for its own forcingAdd / envExpand)
// ===========================================================================

// The display uniform, one declaration for both apps (2D never writes `zslice`,
// which stays 0). `cmap` is the per-display-card colormap index (see CMAP_WGSL).
const MODE_STRUCT = `struct Mode { mode: u32, zslice: u32, cmap: u32, pad: u32 };`;

// ---------------------------------------------------------------------------
// index helpers: everything the 3D app needs to say about m = iz*NMP + mp
// ---------------------------------------------------------------------------
// perpendicular-mode index as an expression in m (for one-off reads)
const _mpExpr = C => (C.hasZ ? "m % NMP" : "m");
// ... and as a local, for kernels that read it more than once
const _mpDecl = C => (C.hasZ ? "  let mp: u32 = m % NMP;\n" : "");
const _mpName = C => (C.hasZ ? "mp" : "m");
// the dealias mask: perpendicular ellipse in 2D, times the |kz| < nz/3 cut in 3D.
// PRECONDITION: the 3D branch reads the local `mp`, so any kernel using this must have
// emitted _mpDecl(C) (which declares it) earlier in the same function body.
const _dealias = C => (C.hasZ ? "gridB[mp].x * gridZ[m / NMP].z" : "gridB[m].x");
// gridZ occupies a binding slot in the 3D versions of the kernels that read it
const _zBind = (C, slot) => (C.hasZ
  ? `@group(0) @binding(${slot}) var<storage, read> gridZ: array<vec4<f32>>;\n` : "");
const _zSlot = (C, slot) => (C.hasZ ? slot + 1 : slot);

// ---------------------------------------------------------------------------
// gradients in k-space: i*k * {phi, psi, vort, jpar}
// ---------------------------------------------------------------------------
// PERPENDICULAR gradients only. In 3D the parallel derivative is not an RHS term
// (it lives in the linear propagator), so the 3D kernel is the 2D one with gridA
// read at the perpendicular index mp; vort/jpar use the perpendicular ksq too.
function prepGradsWGSL(C) {
  return C.pre + `
@group(0) @binding(0) var<storage, read> fields: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> gridA: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> outg: array<vec2<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m: u32 = gid.x;
  if (m >= NM) { return; }
  let g: vec4<f32> = gridA[${_mpExpr(C)}];
  let phi: vec2<f32> = fields[m];
  let psi: vec2<f32> = fields[NM + m];
  let vort: vec2<f32> = -g.z * phi;
  let jpar: vec2<f32> = -g.z * psi;
  outg[m]           = vec2<f32>(-g.x * phi.y,  g.x * phi.x);
  outg[NM + m]      = vec2<f32>(-g.y * phi.y,  g.y * phi.x);
  outg[2u*NM + m]   = vec2<f32>(-g.x * psi.y,  g.x * psi.x);
  outg[3u*NM + m]   = vec2<f32>(-g.y * psi.y,  g.y * psi.x);
  outg[4u*NM + m]   = vec2<f32>(-g.x * vort.y, g.x * vort.x);
  outg[5u*NM + m]   = vec2<f32>(-g.y * vort.y, g.y * vort.x);
  outg[6u*NM + m]   = vec2<f32>(-g.x * jpar.y, g.x * jpar.x);
  outg[7u*NM + m]   = vec2<f32>(-g.y * jpar.y, g.y * jpar.x);
}`;
}

// ---------------------------------------------------------------------------
// Poisson brackets in real space (identical text; only the workgroup differs)
// ---------------------------------------------------------------------------
function bracketWGSL(C) {
  return C.pre + `
@group(0) @binding(0) var<storage, read> gr: array<f32>;
@group(0) @binding(1) var<storage, read_write> nl: array<f32>;
@compute @workgroup_size(${C.wgReal})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= NR) { return; }
  let ax: f32 = gr[i];            let ay: f32 = gr[NR + i];        // grad phi
  let bx: f32 = gr[2u*NR + i];    let by: f32 = gr[3u*NR + i];     // grad psi
  let cx: f32 = gr[4u*NR + i];    let cy: f32 = gr[5u*NR + i];     // grad vort
  let dx_: f32 = gr[6u*NR + i];   let dy_: f32 = gr[7u*NR + i];    // grad jpar
  nl[i]      = (bx * dy_ - by * dx_) - (ax * cy - ay * cx);        // {psi,jpar} - {phi,vort}
  nl[NR + i] = -(ax * by - ay * bx);                               // -{phi,psi}
}`;
}

// ---------------------------------------------------------------------------
// assemble the nonlinear part of the RHS (dealias here and ONLY here)
// ---------------------------------------------------------------------------
function nlAssembleWGSL(C) {
  return C.pre + `
@group(0) @binding(0) var<storage, read> nlk: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> gridA: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gridB: array<vec4<f32>>;
${_zBind(C, 3)}@group(0) @binding(${_zSlot(C, 3)}) var<storage, read_write> rhs: array<vec2<f32>>;
${_eqSrcBind(C)}@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m: u32 = gid.x;
  if (m >= NM) { return; }
${_mpDecl(C)}  let de: f32 = ${_dealias(C)};
  rhs[m]      = (-gridA[${_mpName(C)}].w * de) * nlk[m];
  rhs[NM + m] = de * nlk[NM + m]${_eqSrcAdd(C)};
}`;
}

// ---------------------------------------------------------------------------
// maintained equilibrium flux (REFINE_PLAN J2.3)
// ---------------------------------------------------------------------------
// A static source S = -eta grad^2 psi_eq on the psi equation cancels the resistive decay
// of an equilibrium, so a tearing demo shows the instability and not the layer's own
// diffusion. In k space S = -lin_L * psi_eq,k -- the SAME diagonal gridB.y the stage
// applies, so the source tracks the eta slider with no bookkeeping of its own, and it is
// eta and not nu because gridB.y IS psi's rate (see _pm below). psi_eq,k lives in `eqk`,
// written once per Reset by the app's srcInit; here it is one emit-time term, present ONLY
// when the equilibrium preset asks for it -- with it off, every kernel below is the
// pre-J2 text byte for byte.
const _eqSrcBind = C => (C.eqSrc
  ? `@group(0) @binding(${_zSlot(C, 3) + 1}) var<storage, read> eqk: array<vec2<f32>>;\n` : "");
const _eqSrcAdd = C => (C.eqSrc ? ` - gridB[${_mpName(C)}].y * eqk[m]` : "");

// ---------------------------------------------------------------------------
// per-field dissipation (REFINE_PLAN J.1, in Pm since J2.6)
// ---------------------------------------------------------------------------
// `C.pm` is the magnetic Prandtl number nu/eta. The linear diagonal stored in gridB.y is
// ETA's (-eta ksq^hyper) -- the diss slider IS eta -- so PHI's rate is that times Pm: one
// emit-time factor, never a forked kernel. It is a COMPILE-TIME constant on purpose: at
// the default Pm of 1 every template below emits the pre-J text character for character
// (the byte-diff gate), and changing Pm rebuilds the solver exactly as changing the
// resolution or the forcing band does. Pm = 0 (inviscid phi, pure resistive tearing) is a
// legitimate value and is NOT the default path -- hence the explicit null test.
// 3D never sets it: its 2x2 Alfven propagator needs an equal diagonal, so nu = eta there
// is a constraint, not a default.
const _f32lit = v => (v === Math.round(v) ? v.toFixed(1) : String(v));
const _pm = C => (C.pm === undefined || C.pm === 1 ? null : C.pm);
// the ek term of the dissipation-rate accumulator, and the phi factor of a linear
// diagonal read at flat stack index `idx` (idx < NM is phi)
const _dissEk = C => (_pm(C) === null ? "ek" : `${_f32lit(C.pm)} * ek`);
const _dissLin = (C, expr) => (_pm(C) === null
  ? expr : `(${expr} * select(1.0, ${_f32lit(C.pm)}, idx < NM))`);

// ---------------------------------------------------------------------------
// energy + dissipation rate (first stage; energyFinal is the generic tail)
// ---------------------------------------------------------------------------
// The dissipation rate uses only the DIAGONAL d of the linear operator: in 3D the
// off-diagonal i*kzd Alfven coupling is energy-conserving, so d picks up the kz^4
// damping and nothing else. With nu != eta (2D only) the KINETIC half of the rate
// carries Pm -- one substituted factor, see _dissEk above.
//
// The FOURTH accumulator lane (REFINE_PLAN H.2) is the cross helicity
//   H_c = <u.b> = sum ksq_perp * Re(phi conj(psi)) * yfac * INVN2,
// which is all the Elsasser energies need on top of what is already here:
//   E+- = E_kin + E_mag +- H_c   (so E_tot = (E+ + E-)/2, the repo's convention).
// It rides in the vec4 lane that used to be a hard 0.0 -- no new kernel, no new
// dispatch, no new buffer, and the other three lanes are bit-for-bit unchanged
// (the reduction is componentwise).
function energyPartialWGSL(C) {
  const dcoefLine = C.hasZ ? "      let dcoef: f32 = B.y + gridZ[m / NMP].w;\n" : "";
  const dcoef = C.hasZ ? "dcoef" : "B.y";
  return C.pre + `
@group(0) @binding(0) var<storage, read> fields: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> gridA: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gridB: array<vec4<f32>>;
${_zBind(C, 3)}@group(0) @binding(${_zSlot(C, 3)}) var<storage, read_write> part: array<vec4<f32>>;
var<workgroup> sh: array<vec4<f32>, 256>;
fn add4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> { return a + b; }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid: u32 = lid.x;
  var acc: vec4<f32> = vec4<f32>(0.0);
  var m: u32 = wgid.x * 1024u + tid;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    if (m < NM) {
      let A: vec4<f32> = gridA[${_mpExpr(C)}];
      let B: vec4<f32> = gridB[${_mpExpr(C)}];
${dcoefLine}      let phi: vec2<f32> = fields[m];
      let psi: vec2<f32> = fields[NM + m];
      let ek: f32 = A.z * dot(phi, phi) * B.w;
      let em: f32 = A.z * dot(psi, psi) * B.w;
      acc = acc + vec4<f32>(ek, em, -${dcoef} * (${_dissEk(C)} + em), A.z * dot(phi, psi) * B.w);
    }
    m = m + 256u;
  }
${reduceTail("vec4<f32>", "add4")}
  if (tid == 0u) { part[wgid.x] = sh[0]; }
}`;
}

// ---------------------------------------------------------------------------
// OU update on the shell modes
// ---------------------------------------------------------------------------
// 2D: 2 arrays (z+, z-). 3D: 4, (z+, z-) x (A, B) -- the cos/sin z-envelope
// coefficients -- laid out on the PERPENDICULAR grid, hence the NMP stride.
function ouWGSL(C) {
  const nblk = C.hasZ ? "4u" : "2u";
  const idx = C.hasZ
    ? `  let blk: u32 = n / NS;                       // (ou*2 + ab)
  let s: u32 = n - blk * NS;
  let m: u32 = blk * NMP + shell[s];
`
    : `  let ou: u32 = n / NS;
  let s: u32 = n - ou * NS;
  let m: u32 = ou * NM + shell[s];
`;
  return C.pre + `
@group(0) @binding(0) var<storage, read_write> frc: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> shell: array<u32>;
@group(0) @binding(2) var<storage, read> noise: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> sc: array<f32>;
@group(0) @binding(4) var<uniform> cfg: Cfg;
const NS: u32 = ${C.ns}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n: u32 = gid.x;
  if (n >= ${nblk} * NS) { return; }
${idx}  let decay: f32 = exp(-sc[0] / cfg.tau);
  let diff: f32 = sqrt(max(0.0, 1.0 - decay * decay));
  frc[m] = frc[m] * decay + diff * noise[n];
}`;
}

// ---------------------------------------------------------------------------
// once-per-step power normalization (lagged by design)
// ---------------------------------------------------------------------------
// Two reductions over the forcing shell, sharing one vec4 accumulator:
//   .x/.y  P+-  = sum ksq_perp * Re(conj(z+-) * F+-) * yfac * INVN2   (cross term)
//   .z/.w  F2+- = sum ksq_perp * |F+-|^2          * yfac * INVN2   (self term)
// In 3D only the two kz = +-2pi/Lz planes carry F, so each sum is 2 x (shell size)
// terms and one thread owns both planes of its column (nz = 2 then sums, as it must
// -- exact for P, which is linear in F; the quadratic F2 would need the two planes
// added BEFORE squaring there, but nz >= 32 in the UI so the planes never coincide).
// The epilogue is shared_physics.selfnorm_scale: the scale FACTOR is capped
// (forcing_scale_max), never the denominator P.
function scaleWGSL(C) {
  const accum = C.hasZ
    ? `    let mp: u32 = shell[s];
    let w: f32 = gridA[mp].z * gridB[mp].w;      // ksq_perp * yfac
    let Ap: vec2<f32> = frc[mp];
    let Bp: vec2<f32> = frc[NMP + mp];
    let Am: vec2<f32> = frc[2u * NMP + mp];
    let Bm: vec2<f32> = frc[3u * NMP + mp];
    for (var pl: u32 = 0u; pl < 2u; pl = pl + 1u) {
      let m: u32 = select(1u, NZ - 1u, pl == 1u) * NMP + mp;
      let phi: vec2<f32> = fields[m];
      let psi: vec2<f32> = fields[NM + m];
      let zp: vec2<f32> = phi + psi;
      let zm: vec2<f32> = phi - psi;
      let fp: vec2<f32> = envelope(Ap, Bp, pl);
      let fm: vec2<f32> = envelope(Am, Bm, pl);
      acc = acc + vec4<f32>(w * dot(zp, fp), w * dot(zm, fm),
                            w * dot(fp, fp), w * dot(fm, fm));
    }
`
    : `    let m: u32 = shell[s];
    let w: f32 = gridA[m].z * gridB[m].w;      // ksq * yfac
    let phi: vec2<f32> = fields[m];
    let psi: vec2<f32> = fields[NM + m];
    let zp: vec2<f32> = phi + psi;
    let zm: vec2<f32> = phi - psi;
    let fp: vec2<f32> = frc[m];
    let fm: vec2<f32> = frc[NM + m];
    acc = acc + vec4<f32>(w * dot(zp, fp), w * dot(zm, fm),
                          w * dot(fp, fp), w * dot(fm, fm));
`;
  return C.pre + (C.envFn || "") + `
@group(0) @binding(0) var<storage, read> fields: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> frc: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> gridA: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gridB: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> shell: array<u32>;
@group(0) @binding(5) var<storage, read_write> sc: array<f32>;
@group(0) @binding(6) var<uniform> cfg: Cfg;
var<workgroup> sh: array<vec4<f32>, 64>;
const NS: u32 = ${C.ns}u;
// shared_physics.selfnorm_scale, line for line (FORCING_SPINUP_PLAN Phase 3; the
// derivation is docs/numerics.md "Normalize against the forcing's own self-energy").
// Over one step of length dt the force s*f_raw injects dE = s*P*dt + 0.5*s^2*F2*dt^2;
// requiring dE = target*dt gives 0.5*F2*dt*s^2 + P*s - target = 0. The old
// s = target/P is its |P| -> large limit, and from a quiescent start P = 0 pinned it at
// +-smax, injecting ~0.5*smax^2*F2*dt^2 INDEPENDENTLY of eps -- the spin-up kick this
// removes. Same pinning recurs whenever P fluctuates through zero, so the fix has to be
// continuous in P; the quadratic is.
// NB "tgt", not "target": target is a RESERVED WORD in WGSL (Tint rejects it as an
// identifier; wgsl_reflect does not, so only real devices catch it).
fn selfnormScale(tgt: f32, P: f32, F2: f32, dt: f32, smax: f32) -> f32 {
  let F2dt: f32 = F2 * dt;
  // max(): tgt >= 0 (an injection RATE) makes disc >= P^2 >= 0, so this is
  // unreachable in practice -- it is there so a negative target returns a finite number
  // rather than a NaN poisoning the whole field array.
  let r: f32 = sqrt(max(P * P + 2.0 * F2dt * tgt, 0.0));
  // POSITIVE root (plan Decision 1): the old form followed sign(P) and so flipped the
  // force under an adverse phase, rectifying the OU process; the positive root keeps
  // s > 0 and lets the exact solve absorb an adverse linear term.
  // Two algebraically identical forms, each used where it does NOT suffer catastrophic
  // cancellation -- which matters precisely in the saturated regime 2*F2*dt*target << P^2,
  // where r -> |P| and one numerator cancels:
  //   P >= 0: (r - P)/(F2*dt) cancels  ->  conjugate form 2*target/(P + r)
  //   P <  0: 2*target/(P + r) cancels ->  direct form    (r - P)/(F2*dt)
  // Both denominators are guarded so the UNSELECTED branch cannot produce an inf/NaN
  // (select evaluates both operands): P + r > 0 for P >= 0 whenever target > 0, and the
  // degenerate P = r = 0 case only arises for target == 0, short-circuited below.
  let den: f32 = P + r;
  var s: f32 = select((r - P) / select(1.0, F2dt, F2dt > 0.0),
                      2.0 * tgt / select(1.0, den, den > 0.0),
                      P >= 0.0);
  // F2*dt == 0 (no envelope drawn yet, or dt == 0 at initialization -- see _uploadIC,
  // which zeroes sc[0] and the forcing buffer): fall back to the old clamp, which is the
  // same normalization minus the self term.
  s = select(tgt / P, s, F2dt > 0.0);
  s = select(s, 0.0, tgt == 0.0);
  return clamp(s, -smax, smax);   // last-resort safety, not the normalization
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid: u32 = lid.x;
  var acc: vec4<f32> = vec4<f32>(0.0);
  for (var s: u32 = tid; s < NS; s = s + 64u) {
${accum}  }
  sh[tid] = acc;
  workgroupBarrier();
  var stride: u32 = 32u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { sh[tid] = sh[tid] + sh[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) {
    // dt is sc[0], the step JUST COMPLETED (cflFinal wrote it before this step's stages,
    // tick and ou already consumed it): the same LAGGED dt run._advance_forcing hands to
    // rmhd.forcing_scale; under cfl_every > 1 (dt frozen per block) the lag vanishes
    // within a block and reappears only across block boundaries.
    // It is deliberately NOT a Cfg field -- dt never exists CPU-side here (Cfg is uploaded
    // only on control changes, and sc is read back asynchronously), so sc[0] is both the
    // closest mirror of the jax side and the only synchronous source.
    let dt: f32 = sc[0];
    let Pp: f32 = sh[0].x * INVN2;
    let Pm: f32 = sh[0].y * INVN2;
    let F2p: f32 = sh[0].z * INVN2;
    let F2m: f32 = sh[0].w * INVN2;
    // targets 2*eps+- : the factor 2 of rmhd._forcing_scale_from (E_tot = (E+ + E-)/2)
    sc[4] = selfnormScale(2.0 * cfg.epsP, Pp, F2p, dt, cfg.smax);
    sc[5] = selfnormScale(2.0 * cfg.epsM, Pm, F2m, dt, cfg.smax);
  }
}`;
}

// ---------------------------------------------------------------------------
// initial condition: dealias the transformed IC
// ---------------------------------------------------------------------------
// 2D walks the 2-field stack (2*NM threads, the mask index folded back with
// select); 3D walks one field (NM threads) and applies the shared de to both,
// because its mask is a product of a perpendicular and a kz factor.
function icFinishWGSL(C) {
  const body = C.hasZ
    ? `  let m: u32 = gid.x;
  if (m >= NM) { return; }
  let de: f32 = gridB[m % NMP].x * gridZ[m / NMP].z;
  fields[m] = ick[m] * de;
  fields[NM + m] = ick[NM + m] * de;`
    : `  let idx: u32 = gid.x;
  if (idx >= 2u * NM) { return; }
  let m: u32 = select(idx, idx - NM, idx >= NM);
  fields[idx] = ick[idx] * gridB[m].x;`;
  return C.pre + `
@group(0) @binding(0) var<storage, read> ick: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> gridB: array<vec4<f32>>;
${_zBind(C, 2)}@group(0) @binding(${_zSlot(C, 2)}) var<storage, read_write> fields: array<vec2<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
${body}
}`;
}

// ===========================================================================
// display chain
// ===========================================================================
// Display modes (the `mode` field of the app's Mode uniform):
//   0 vorticity   1 current   2 phi   3 psi       signed scalars, autoscaled +-max
//   4 |u|   5 |b|   6 |z+|   7 |z-|               vector magnitudes (+ arrows)
//   8 sigma_c   9 sigma_r                         signed, FIXED range +-1
// The vector modes show |zhat x grad f| = |grad f| for f = phi, psi, phi+psi,
// phi-psi: u, b and the two Elsasser fields z+- = u +- b. Cross helicity
//   sigma_c = (|z+|^2 - |z-|^2) / (|z+|^2 + |z-|^2)
// is built from the same two vectors, so its chain is the vector chain run twice
// (mode 8 -> the z+ half; a second Mode uniform pinned to DISP_ZMINUS -> the z-
// half) and then combined. That costs 4 inverse transforms per frame, which is
// display cost only -- no physics buffer is touched.
// Residual energy
//   sigma_r = (|u|^2 - |b|^2) / (|u|^2 + |b|^2)
// is the SAME machinery with the other pair of vectors: mode 9 -> the u half (its
// prepDisp branch picks phi, exactly as mode 4 does) and the second Mode uniform
// pinned to DISP_BVEC -> the b half. Everything after the two halves -- the squared
// magnitudes, the quiet-region floor, the ratio and the fixed +-1 colour range -- is
// shared with sigma_c, which is why dispIsSigma covers both. Both apps offer it
// (`C.sigR`), so in 3D it rides the identical chain on a z plane or on the three cube
// faces -- the only 3D-specific piece is that its second half runs through the same
// extraction as the first (see rmhd3d.html's _dispHalf).
const DISP_VEC0 = 4;        // first vector-magnitude mode -- also the u vector
const DISP_BVEC = 5;        // b vector: also the pinned mode of the sigma_r 2nd half
const DISP_ZMINUS = 7;      // z- vector: also the pinned mode of the sigma_c 2nd half
const DISP_SIGMA = 8;       // sigma_c
const DISP_SIGMA_R = 9;     // sigma_r
// the two POTENTIALS, as display modes: the contour overlay (REFINE_PLAN I2.4) prepares
// one of them through the same prepDisp kernel, so its selector value IS its mode.
const DISP_PHI = 2;         // phi -> streamlines
const DISP_PSI = 3;         // psi -> B_perp field lines
const dispIsVector = m => m >= DISP_VEC0 && m <= DISP_ZMINUS;  // magnitude + arrows
const dispIsSigma = m => m === DISP_SIGMA || m === DISP_SIGMA_R;
// which vector the sigma chain's SECOND half is pinned to: z- for sigma_c, b for sigma_r
const dispSigmaMate = m => (m === DISP_SIGMA_R ? DISP_BVEC : DISP_ZMINUS);
// (the colour range is decided in the colorize kernel, which branches on the mode
// itself -- there is no CPU-side signedness predicate any more)
// modes whose display chain needs BOTH components inverse-transformed
const dispTwoComp = m => dispIsVector(m) || dispIsSigma(m);

// ---------------------------------------------------------------------------
// pick the displayed quantity in k-space
// ---------------------------------------------------------------------------
// modes 0..3 are scalars (outk only, outk2 = 0); modes 4..8 are the perpendicular
// vector fields zhat x grad f = (-d_y f, +d_x f), whose two components go to
// outk / outk2 and get their own inverse transform.
// In 3D this is the full (nz,nkx,nky) quantity: it gets a full 3D inverse transform,
// and one z slice is extracted from the real-space result (sliceExtract) -- or its
// three boundary faces are (faceExtract, the cube modes).
function prepDispWGSL(C) {
  // sigma_r's FIRST half is the u vector, so mode 9 has to pick phi like mode 4 does --
  // one extra branch, emitted only where the mode exists (both apps set C.sigR; a C
  // without it still gets the historical text).
  const sigR = C.sigR
    ? `    else if (md.mode == ${DISP_SIGMA_R}u) { f = phi; }   // 9 -> phi (sigma_r's u half)\n`
    : "";
  return C.pre + `
${MODE_STRUCT}
@group(0) @binding(0) var<storage, read> fields: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> gridA: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> outk: array<vec2<f32>>;
@group(0) @binding(3) var<uniform> md: Mode;
@group(0) @binding(4) var<storage, read_write> outk2: array<vec2<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m: u32 = gid.x;
  if (m >= NM) { return; }
${_mpDecl(C)}  var v: vec2<f32> = vec2<f32>(0.0, 0.0);
  var v2: vec2<f32> = vec2<f32>(0.0, 0.0);
  if (md.mode == 0u) { v = -gridA[${_mpName(C)}].z * fields[m]; }
  else if (md.mode == 1u) { v = -gridA[${_mpName(C)}].z * fields[NM + m]; }
  else if (md.mode == 2u) { v = fields[m]; }
  else if (md.mode == 3u) { v = fields[NM + m]; }
  else {
    let phi: vec2<f32> = fields[m];
    let psi: vec2<f32> = fields[NM + m];
    // 4 -> phi (u), 5 -> psi (b), 7 -> z- = phi - psi, 6 and 8 -> z+ = phi + psi
    var f: vec2<f32> = phi;
    if (md.mode == 5u) { f = psi; }
    else if (md.mode == ${DISP_ZMINUS}u) { f = phi - psi; }
${sigR}    else if (md.mode > 5u) { f = phi + psi; }
    let kx: f32 = gridA[${_mpName(C)}].x;
    let ky: f32 = gridA[${_mpName(C)}].y;
    v  = vec2<f32>( ky * f.y, -ky * f.x);   // -i*ky*f  =  -d_y f
    v2 = vec2<f32>(-kx * f.y,  kx * f.x);   // +i*kx*f  =  +d_x f
  }
  outk[m] = v;
  outk2[m] = v2;
}`;
}

// |(ux,uy)| in place: a <- sqrt(a^2 + b^2), so the existing max-reduce and
// colorize see the (non-negative) magnitude in the display buffer without a new binding.
function vecMagWGSL(C) {
  return C.pre + `
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@compute @workgroup_size(${C.wgReal})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= ${C.nDisp}) { return; }
  let x: f32 = a[i];
  let y: f32 = b[i];
  a[i] = sqrt(x * x + y * y);
}`;
}

// the same collapse without the square root: a <- a^2 + b^2 = |z+-|^2, the two
// energy densities sigma_c is built from (kept squared: the ratio needs no roots).
function vecMagSqWGSL(C) {
  return C.pre + `
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@compute @workgroup_size(${C.wgReal})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= ${C.nDisp}) { return; }
  let x: f32 = a[i];
  let y: f32 = b[i];
  a[i] = x * x + y * y;
}`;
}

// max over the displayed field of (a + b) -- the total Elsasser energy density,
// whose maximum sets the sigma_c quiet-region floor. Same shape as maxPartial
// (common.js) and it feeds the same generic maxFinal tail.
function maxSumPartialWGSL(C) {
  return C.pre + `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid: u32 = lid.x;
  var acc: f32 = 0.0;
  var i: u32 = wgid.x * 1024u + tid;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    if (i < ${C.nDisp}) { acc = max(acc, a[i] + b[i]); }
    i = i + 256u;
  }
${reduceTail("f32", "max")}
  if (tid == 0u) { part[wgid.x] = sh[0]; }
}`;
}

// sigma_c = (E+ - E-)/(E+ + E-) in place (a = E+ = |z+|^2, b = E- = |z-|^2), with
// the quiet-region floor: where the total energy density is below FLOOR x its
// maximum over the displayed field the ratio is dominated by noise, so render an
// exact 0. The result is already in [-1,1] and is colorized on the FIXED +-1
// range -- colorize deliberately ignores the autoscale for this mode.
const SIGMA_FLOOR = 1e-4;
function sigmaCombineWGSL(C) {
  return C.pre + `
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> mx: array<f32>;
const FLOOR: f32 = ${SIGMA_FLOOR.toExponential(6)};
@compute @workgroup_size(${C.wgReal})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= ${C.nDisp}) { return; }
  let ep: f32 = a[i];
  let em: f32 = b[i];
  let s: f32 = ep + em;
  var v: f32 = 0.0;
  if (s > 0.0 && s >= FLOOR * mx[0]) { v = (ep - em) / s; }
  a[i] = v;
}`;
}

// subsample the vector field for the arrow overlay (point sample at the cell corner)
function vecGatherWGSL(C) {
  return C.pre + `
const SX: u32 = ${C.arrow.sx}u;
const SY: u32 = ${C.arrow.sy}u;
const NAX: u32 = ${C.arrow.nax}u;
const NAY: u32 = ${C.arrow.nay}u;
@group(0) @binding(0) var<storage, read> ux: array<f32>;
@group(0) @binding(1) var<storage, read> uy: array<f32>;
@group(0) @binding(2) var<storage, read_write> av: array<vec2<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= NAX * NAY) { return; }
  let ax: u32 = i / NAY;
  let ay: u32 = i % NAY;
  let s: u32 = (ax * SX) * NY + (ay * SY);
  av[i] = vec2<f32>(ux[s], uy[s]);
}`;
}

// ---------------------------------------------------------------------------
// cut line: the four in-plane vector components along y at x = Lx/2
// ---------------------------------------------------------------------------
// The cut chart is SELF-CONTAINED (REFINE_PLAN H.3): it no longer gathers a row out
// of whichever display card happens to exist, it prepares its own line. Doing that
// through a display chain would cost two full inverse transforms; instead the two
// transforms the line does NOT need are done analytically, right here:
//
//   f(x0, y) = sum_kx sum_ky F e^{i kx x0} e^{i ky y},  x0 = Lx/2  =>  e^{i kx Lx/2}
//              = (-1)^ix  for the fftfreq index ix (nx even), so the kx sum is a
//   signed sum, and in 3D the kz sum is the same trick with the plane's phase
//   e^{2 pi i * ikz * iz / NZ}. What is left is ONE inverse rfft along y, which the
//   existing rowsC2R does for four lines at once (this kernel's output layout).
//
// It emits all four components of (u, b) -- u = zhat x grad phi = (-d_y phi, d_x phi),
// b likewise from psi -- because the card's pair selector is then pure CPU work:
// z+- = u +- b, so |z+-| along the line is a hypot of what is already here. One
// kernel, one dispatch, no mode uniform, no per-selection variants.
//
// Normalization: the ix (and iz) sums carry the 1/NX (and 1/NZ) that colsInv (and
// zInv) would have applied; rowsC2R still applies its own 1/NY.
function cutPrepWGSL(C) {
  const inner = `      let mp: u32 = ix * NKY + j;
      let m: u32 = ${C.hasZ ? "iz * NMP + mp" : "mp"};
      let wgt: vec2<f32> = select(zw, -zw, (ix & 1u) == 1u);
      let ph: vec2<f32> = cmul(wgt, fields[m]);
      let ps: vec2<f32> = cmul(wgt, fields[NM + m]);
      let kx: f32 = gridA[mp].x;
      a0 = a0 + ph;   a1 = a1 + kx * ph;
      b0 = b0 + ps;   b1 = b1 + kx * ps;`;
  // 3D wraps the same body in the kz loop, whose phase is the only extra factor
  const body = C.hasZ
    ? `  for (var iz: u32 = 0u; iz < NZ; iz = iz + 1u) {
    let th: f32 = 6.2831853071795862 * f32(iz) * f32(md.zslice) / f32(NZ);
    let zw: vec2<f32> = vec2<f32>(cos(th), sin(th));
    for (var ix: u32 = 0u; ix < NX; ix = ix + 1u) {
${inner}
    }
  }`
    : `  {
    let zw: vec2<f32> = vec2<f32>(1.0, 0.0);
    for (var ix: u32 = 0u; ix < NX; ix = ix + 1u) {
${inner}
    }
  }`;
  return C.pre + (C.hasZ ? MODE_STRUCT + "\n" : "") + `
@group(0) @binding(0) var<storage, read> fields: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> gridA: array<vec4<f32>>;
${C.hasZ ? "@group(0) @binding(2) var<uniform> md: Mode;\n" : ""}@group(0) @binding(${C.hasZ ? 3 : 2}) var<storage, read_write> cutk: array<vec2<f32>>;
const INVXZ: f32 = ${C.hasZ ? "1.0 / (f32(NX) * f32(NZ))" : "1.0 / f32(NX)"};
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j: u32 = gid.x;
  if (j >= NKY) { return; }
  var a0: vec2<f32> = vec2<f32>(0.0, 0.0);
  var a1: vec2<f32> = vec2<f32>(0.0, 0.0);
  var b0: vec2<f32> = vec2<f32>(0.0, 0.0);
  var b1: vec2<f32> = vec2<f32>(0.0, 0.0);
${body}
  let ky: f32 = gridA[j].y;
  // rows: u_x = -i*ky*phi, u_y = +i*kx*phi, b_x = -i*ky*psi, b_y = +i*kx*psi
  cutk[j]            = INVXZ * vec2<f32>( ky * a0.y, -ky * a0.x);
  cutk[NKY + j]      = INVXZ * vec2<f32>(-a1.y, a1.x);
  cutk[2u*NKY + j]   = INVXZ * vec2<f32>( ky * b0.y, -ky * b0.x);
  cutk[3u*NKY + j]   = INVXZ * vec2<f32>(-b1.y, b1.x);
}`;
}

// ---------------------------------------------------------------------------
// colormaps: ONE WGSL implementation, selected per display card by md.cmap
// ---------------------------------------------------------------------------
// The coefficients live in common.js (CMAP_COEF) so the WGSL here and the CPU
// mirror the IC editor's preview uses (cmapRGB) can never drift apart; this is
// the emit-time expansion of the same table. Index order is CMAP_NAMES:
//   0 afmhot (the exact matplotlib closed form -- NOT a fit)
//   1 viridis   2 RdBu   3 grayscale
const _cvec = c => `vec3<f32>(${c[0]}, ${c[1]}, ${c[2]})`;
const _cpoly = name => CMAP_COEF[name].map(_cvec).join(",\n                 ");
const CMAP_WGSL = `
// Horner in t, then clamp: a degree-6 fit overshoots [0,1] slightly at the ends.
fn cpoly6(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>,
          e: vec3<f32>, f: vec3<f32>, g: vec3<f32>) -> vec3<f32> {
  let v: vec3<f32> = a + t * (b + t * (c + t * (d + t * (e + t * (f + t * g)))));
  return clamp(v, vec3<f32>(0.0), vec3<f32>(1.0));
}
fn cmap(x: f32, which: u32) -> vec3<f32> {
  let t: f32 = clamp(x, 0.0, 1.0);
  if (which == 1u) {                       // viridis (degree-6 fit)
    return cpoly6(t, ${_cpoly("viridis")});
  }
  if (which == 2u) {                       // RdBu (degree-6 fit)
    return cpoly6(t, ${_cpoly("rdbu")});
  }
  if (which == 3u) { return vec3<f32>(t, t, t); }              // grayscale
  // matplotlib "afmhot": black -> red -> orange -> white, exact, no LUT needed
  return vec3<f32>(clamp(2.0 * t, 0.0, 1.0),
                   clamp(2.0 * t - 0.5, 0.0, 1.0),
                   clamp(2.0 * t - 1.0, 0.0, 1.0));
}`;

// ---------------------------------------------------------------------------
// display shading: value -> colour, and the contour overlay. ONE implementation
// ---------------------------------------------------------------------------
// Every displayed texel goes through the same two steps whether it lands in the z-slice
// texture (colorize, below) or on a cube face (the 3D app's colorizeCube), so both live
// here and neither kernel carries a copy. The bindings they must declare for the overlay
// are `cp` / `cp2` (the two contour potentials on the SAME plane as the texel) and
// `cd` / `cd2` (their level tables, [range, delta, nlev, plain-background]).
//
// dispX: signed fields (modes 0..3) are symmetric about the autoscale
// (imshow(..., vmin=-s, vmax=+s)); magnitude modes (4..7) are already non-negative and
// map straight onto [0,1]; sigma_c is signed on a FIXED +-1 range and ignores the
// autoscale entirely.
//
// contInk: in-plane field lines (REFINE_PLAN I2.4) -- psi contours are the B_perp field
// lines, phi contours the streamlines. A texel is on a contour when the level index
// floor(pot/delta) differs from that of its +x or +y neighbour (both periodic): a
// two-neighbour crossing test, so no derivatives and no fwidth -- this is a compute
// shader. delta is UNIFORM, so line density goes as |grad pot| = |B_perp| (or |u_perp|),
// which is the physically honest picture; delta <= 0 means that set is off.
//
// Since J2.1/J2.2 it also owns the two options that make the overlay legible on its own:
// the BACKGROUND (cd[3] > 0 replaces the field by a flat plate, so the lines are read
// without the colours underneath) and a SECOND set (the "both" selection: psi AND phi at
// once, for alignment inspection).
//
// INK (FEEDBACK_2026-08-08 P0.2). Both sets ink in a FIXED colour that no colormap here
// produces, so the two are always told apart AND neither carries information it does not
// own. Set 0 USED to pick its ink per texel from the background luminance (black over a
// light background, white over a dark one). That is legible, but it makes every line a
// two-tone image of the DISPLAYED field: the black/white boundary is the background's
// lum = 0.5 contour, which for the signed scalars sits at value/autoscale = -0.10
// (afmhot), 0.00 (grayscale), +0.23 (viridis), +-0.60 (RdBu). Over a phi colour map the
// psi field lines then flip colour along a contour of PHI -- exactly the "psi contours
// reflect the sign/structure of phi" that was reported and mistaken for a cp/cp2 mix-up.
// Over the PLATE there is no field to confuse the ink with and no colormap to survive,
// so set 0 keeps its pure black there: that is what the automatic rule already chose on
// the 0.93 plate, so the plain-background option and the 3D lines view (K2.3) are
// pixel-identical to before.
// There is one implementation: the caller passes the six sampled texels (WGSL
// function parameters cannot be storage pointers) through _contArgs.
// The "plain background" plate, as JS numbers first: the 3D lines view (REFINE_PLAN K2.3)
// clears its canvas to exactly this colour, so the ink-only top face it draws over that
// clear is invisible wherever it is not ink -- a transparent face with no plate, no blend
// state and no second kernel. One constant, both consumers.
const CONT_PLATE_RGB = [0.93, 0.93, 0.93];
const CONT_PLATE = `vec3<f32>(${CONT_PLATE_RGB.join(", ")})`;
const CONT_INK = "vec3<f32>(0.0, 0.85, 1.0)";          // set 0's fixed ink over a field
const CONT_ACCENT = "vec3<f32>(1.0, 0.15, 0.85)";      // set 1's fixed ink
const _contArgs = a => `${a}[gid.x * NY + gid.y], ${a}[((gid.x + 1u) % NX) * NY + gid.y], ` +
                       `${a}[gid.x * NY + ((gid.y + 1u) % NY)]`;
const CONT_ARGS = `${_contArgs("cp")}, ${_contArgs("cp2")}`;
// One shade block, two texts: the sigma modes are the ones rendered on the FIXED +-1
// range, and which modes those are is a property of the constants object, so the
// predicate is the single parameter. `dispShade(C)` picks the right one -- and every
// consumer (2D colorize, 3D colorize AND colorizeCube) goes through it, so a slice and
// a cube face can never disagree about which modes are the sigma ones.
const _dispShadeWGSL = sig => `
fn dispX(raw: f32, s: f32, mode: u32) -> f32 {
  if (${sig}) { return 0.5 * (clamp(raw, -1.0, 1.0) + 1.0); }
  let v: f32 = raw / max(s, 1e-30);
  if (mode >= ${DISP_VEC0}u) { return v; }
  return 0.5 * (clamp(v, -1.0, 1.0) + 1.0);
}
fn contHit(p0: f32, pu: f32, pv: f32, dl: f32) -> bool {
  if (!(dl > 0.0)) { return false; }
  let n0: f32 = floor(p0 / dl);
  return !(n0 == floor(pu / dl) && n0 == floor(pv / dl));
}
fn contInk(col: vec3<f32>, a0: f32, au: f32, av: f32, b0: f32, bu: f32, bv: f32) -> vec3<f32> {
  let plate: bool = cd[3] > 0.5;
  var c: vec3<f32> = select(col, ${CONT_PLATE}, plate);
  if (contHit(a0, au, av, cd[1])) {
    c = mix(c, select(${CONT_INK}, vec3<f32>(0.0), plate), 0.8);
  }
  if (contHit(b0, bu, bv, cd2[1])) { c = mix(c, ${CONT_ACCENT}, 0.85); }
  return c;
}`;
// the historical text: sigma_c alone. Both apps now pass C.sigR, so this is the
// fallback for a constants object that does not offer sigma_r.
const DISP_SHADE_WGSL = _dispShadeWGSL(`mode == ${DISP_SIGMA}u`);
const DISP_SHADE_SIGR_WGSL = _dispShadeWGSL(`mode == ${DISP_SIGMA}u || mode == ${DISP_SIGMA_R}u`);
const dispShade = C => (C.sigR ? DISP_SHADE_SIGR_WGSL : DISP_SHADE_WGSL);

// the contour level table, one thread: max |pot| over the displayed plane -> a slowly
// adapting range -> the uniform spacing delta = 2*range/nlev. Adapting on the GPU (rather
// than reading the max back) keeps the overlay off the CPU frame path entirely; the range
// rises at once and falls by 5% of the gap per frame, so the lines do not flicker as the
// plane's extremum wanders. Each contour SET has its own table -- psi and phi have
// unrelated ranges -- and one kernel serves both. st = [range, delta, nlev, plain
// background] with the last two written by the CPU, and delta = 0 turns that set off.
const CONT_RELAX = 0.05;
function contLevelWGSL(C) {
  return C.pre + `
@group(0) @binding(0) var<storage, read> mx: array<f32>;
@group(0) @binding(1) var<storage, read_write> st: array<f32>;
@compute @workgroup_size(1)
fn main() {
  let m: f32 = mx[0];
  var r: f32 = st[0];
  if (!(r > 0.0) || m > r) { r = m; } else { r = r + ${CONT_RELAX} * (m - r); }
  st[0] = r;
  st[1] = 2.0 * r / max(st[2], 1.0);
}`;
}

// colorize the displayed slice into the render texture
function colorizeWGSL(C) {
  return C.pre + `
${MODE_STRUCT}
@group(0) @binding(0) var<storage, read> f: array<f32>;
@group(0) @binding(1) var<storage, read> mx: array<f32>;
@group(0) @binding(2) var tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> md: Mode;
@group(0) @binding(4) var<storage, read> cp: array<f32>;
@group(0) @binding(5) var<storage, read> cd: array<f32>;
@group(0) @binding(6) var<storage, read> cp2: array<f32>;
@group(0) @binding(7) var<storage, read> cd2: array<f32>;
${CMAP_WGSL}
${dispShade(C)}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= NX || gid.y >= NY) { return; }
  let x: f32 = dispX(f[gid.x * NY + gid.y], mx[0], md.mode);
  let col: vec3<f32> = contInk(cmap(x, md.cmap), ${CONT_ARGS});
  textureStore(tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
}`;
}

// full-screen triangle blit of the display texture
const RENDER_WGSL = `
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_2d<f32>;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  var p: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4<f32>(p[vi], 0.0, 1.0);
  o.uv = (p[vi] + vec2<f32>(1.0, 1.0)) * 0.5;
  return o;
}
@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
  return textureSample(tex, samp, vec2<f32>(i.uv.x, 1.0 - i.uv.y));
}`;
