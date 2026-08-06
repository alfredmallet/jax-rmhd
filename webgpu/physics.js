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
// vecMagSq, maxSumPartial, sigmaCombine, vecGather, cutGather, colorize, the
// blit) and the display-mode predicates the apps and the frame loop branch on.
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
//               domain), "NRS" in 3D (one z slice)
//   arrow       arrowDims() result for the arrow-overlay subsample
//   ns          forcing-shell size (NS)
//   modeStruct  the app's `struct Mode` declaration (2D pads what 3D uses for
//               the z slice)
//   envFn       3D only: the `envelope()` helper, prepended to `scale` (the app
//               uses the same text for its own forcingAdd / envExpand)
// ===========================================================================

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
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let m: u32 = gid.x;
  if (m >= NM) { return; }
${_mpDecl(C)}  let de: f32 = ${_dealias(C)};
  rhs[m]      = (-gridA[${_mpName(C)}].w * de) * nlk[m];
  rhs[NM + m] = de * nlk[NM + m];
}`;
}

// ---------------------------------------------------------------------------
// energy + dissipation rate (first stage; energyFinal is the generic tail)
// ---------------------------------------------------------------------------
// The dissipation rate uses only the DIAGONAL d of the linear operator: in 3D the
// off-diagonal i*kzd Alfven coupling is energy-conserving, so d picks up the kz^4
// damping and nothing else.
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
      acc = acc + vec4<f32>(ek, em, -${dcoef} * (ek + em), 0.0);
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
// P+- = sum ksq_perp * Re(conj(z+-) * F+-) * yfac * INVN2. In 3D only the two
// kz = +-2pi/Lz planes carry F, so the sum is 2 x (shell size) terms and one
// thread owns both planes of its column (nz = 2 then sums, as it must).
// The scale FACTOR is capped (forcing_scale_max), never the denominator P.
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
      acc = acc + vec4<f32>(w * dot(zp, envelope(Ap, Bp, pl)),
                            w * dot(zm, envelope(Am, Bm, pl)), 0.0, 0.0);
    }
`
    : `    let m: u32 = shell[s];
    let w: f32 = gridA[m].z * gridB[m].w;      // ksq * yfac
    let phi: vec2<f32> = fields[m];
    let psi: vec2<f32> = fields[NM + m];
    let zp: vec2<f32> = phi + psi;
    let zm: vec2<f32> = phi - psi;
    acc = acc + vec4<f32>(w * dot(zp, frc[m]), w * dot(zm, frc[NM + m]), 0.0, 0.0);
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
    let Pp: f32 = sh[0].x * INVN2;
    let Pm: f32 = sh[0].y * INVN2;
    sc[4] = select(clamp(2.0 * cfg.epsP / Pp, -cfg.smax, cfg.smax), 0.0, cfg.epsP == 0.0);
    sc[5] = select(clamp(2.0 * cfg.epsM / Pm, -cfg.smax, cfg.smax), 0.0, cfg.epsM == 0.0);
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
//   8 sigma_c                                     signed, FIXED range +-1
// The vector modes show |zhat x grad f| = |grad f| for f = phi, psi, phi+psi,
// phi-psi: u, b and the two Elsasser fields z+- = u +- b. Cross helicity
//   sigma_c = (|z+|^2 - |z-|^2) / (|z+|^2 + |z-|^2)
// is built from the same two vectors, so its chain is the vector chain run twice
// (mode 8 -> the z+ half; a second Mode uniform pinned to DISP_ZMINUS -> the z-
// half) and then combined. That costs 4 inverse transforms per frame, which is
// display cost only -- no physics buffer is touched.
const DISP_VEC0 = 4;        // first vector-magnitude mode
const DISP_ZMINUS = 7;      // z- vector: also the pinned mode of the sigma_c 2nd half
const DISP_SIGMA = 8;       // sigma_c
const dispIsVector = m => m >= DISP_VEC0 && m <= DISP_ZMINUS;  // magnitude + arrows
const dispIsSigma = m => m === DISP_SIGMA;
const dispIsSigned = m => m < DISP_VEC0 || m === DISP_SIGMA;   // symmetric colour range
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
  return C.pre + `
${C.modeStruct}
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
    else if (md.mode > 5u) { f = phi + psi; }
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

// one grid line of the displayed scalar at fixed x = Lx/2 (ix = NX/2), for the
// cut-trace chart: NY values gathered from the displayed (slice) buffer.
function cutGatherWGSL(C) {
  return C.pre + `
@group(0) @binding(0) var<storage, read> f: array<f32>;
@group(0) @binding(1) var<storage, read_write> cut: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j: u32 = gid.x;
  if (j >= NY) { return; }
  cut[j] = f[(NX / 2u) * NY + j];
}`;
}

// afmhot colorize of the displayed slice into the render texture
function colorizeWGSL(C) {
  return C.pre + `
${C.modeStruct}
@group(0) @binding(0) var<storage, read> f: array<f32>;
@group(0) @binding(1) var<storage, read> mx: array<f32>;
@group(0) @binding(2) var tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> md: Mode;
// matplotlib "afmhot": black -> red -> orange -> white, no LUT needed
fn cmap(x: f32) -> vec3<f32> {
  let t: f32 = clamp(x, 0.0, 1.0);
  return vec3<f32>(clamp(2.0 * t, 0.0, 1.0),
                   clamp(2.0 * t - 0.5, 0.0, 1.0),
                   clamp(2.0 * t - 1.0, 0.0, 1.0));
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= NX || gid.y >= NY) { return; }
  let raw: f32 = f[gid.x * NY + gid.y];
  let v: f32 = raw / max(mx[0], 1e-30);
  // signed fields: vmin=-vmax, vmax=+vmax (imshow(..., cmap="afmhot", vmin=-s, vmax=s));
  // magnitude modes are already non-negative and map straight onto [0,1];
  // sigma_c is signed on a FIXED +-1 range, so it skips the autoscale entirely.
  var x: f32;
  if (md.mode == ${DISP_SIGMA}u) { x = 0.5 * (clamp(raw, -1.0, 1.0) + 1.0); }
  else if (md.mode >= ${DISP_VEC0}u) { x = v; }
  else { x = 0.5 * (clamp(v, -1.0, 1.0) + 1.0); }
  textureStore(tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(cmap(x), 1.0));
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
