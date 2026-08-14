"use strict";
// ===========================================================================
// common.js -- everything the 2D (rmhd2d.html) and 3D (rmhd3d.html) WebGPU RMHD
// apps share: RNG, reference-vector helpers, the FFT kernel template and the
// dimension-agnostic WGSL snippets, device bring-up, the colormap table, the
// chart / overlay drawing, the DISPLAY-CARD and CHART-CARD system (one class
// each, instantiated N times by both apps), the preset machinery, the self-test
// table and the frame loop.
//
// What deliberately does NOT live here: the equation kernels and the display
// chain (physics.js, which templates them over one 2D/3D constants object), the
// per-app kernels physics.js lists as unshared (stage, forcingAdd/envExpand, the
// spectra, sliceExtract/faceExtract/cube), makeGrid, and the Solver classes.
// What stays: the pieces that carry no equation -- the FFT template, the generic
// reductions (tick, CFL, energy tail, max), device bring-up, charts, overlays,
// the self-test harness and the frame loop.
//
// Loaded as a plain classic <script src="common.js"> BEFORE physics.js and each
// app's inline script, so top-level `let`/`const`/`function` here are visible to
// both (and must not be redeclared there). Works from file://; the reference
// vectors stay inlined in the HTML because fetch() does not.
// ===========================================================================

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
const el = id => document.getElementById(id);
const statusEl = document.getElementById("status");
// The one authoritative per-browser/per-platform WebGPU support matrix (W3C WebGPU
// working group). The two GPU-failure banners link here INSTEAD of naming
// --enable-unsafe-webgpu: the experimental-flag route is documented upstream for anyone
// determined, and this page never tells a visitor to switch off their browser's safety.
const WEBGPU_STATUS_URL = "https://github.com/gpuweb/gpuweb/wiki/Implementation-Status";
function showStatus(msg, kind, linkUrl) {
  statusEl.className = kind || "info";
  statusEl.textContent = msg;
  // Optional trailing link, appended as a NODE: #status stays textContent-only for the
  // message itself, so no status string is ever parsed as HTML (some carry driver text).
  if (linkUrl) {
    const a = document.createElement("a");
    a.textContent = "full browser-support matrix";
    a.href = linkUrl;
    a.target = "_blank";
    a.rel = "noopener";
    statusEl.appendChild(a);
    statusEl.appendChild(document.createTextNode("."));
  }
}
// any uncaught error anywhere -> visible on the page, never a silently dead UI
window.addEventListener("error", e => showStatus("Error: " + e.message, "err"));
window.addEventListener("unhandledrejection", e =>
  showStatus("Error: " + (e.reason && e.reason.message || e.reason), "err"));

// mulberry32 + Box-Muller (polar). jax's threefry is deliberately not reproduced:
// the OU stream is validated statistically, never against recorded values.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
class Gauss {
  constructor(seed) { this.r = mulberry32(seed >>> 0); this.spare = null; }
  next() {
    if (this.spare !== null) { const s = this.spare; this.spare = null; return s; }
    let u, v, s2;
    do { u = 2 * this.r() - 1; v = 2 * this.r() - 1; s2 = u * u + v * v; } while (s2 >= 1 || s2 === 0);
    const f = Math.sqrt(-2 * Math.log(s2) / s2);
    this.spare = v * f;
    return u * f;
  }
}

function relL2(a, b) {
  let num = 0, den = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { const d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; }
  if (den === 0) return Math.sqrt(num);
  return Math.sqrt(num / den);
}
// nested reference list -> flat Float32Array (row-major, same order as our buffers)
function flatR(nested) {
  const out = [];
  (function rec(a) { if (Array.isArray(a[0])) { for (const s of a) rec(s); } else { for (const v of a) out.push(v); } })(nested);
  return Float32Array.from(out);
}
// {re: nested, im: nested} -> flat interleaved (re,im) Float32Array
function flatC(o) {
  const out = [];
  (function rec(r, i) {
    if (Array.isArray(r[0])) { for (let k = 0; k < r.length; k++) rec(r[k], i[k]); }
    else { for (let k = 0; k < r.length; k++) { out.push(r[k], i[k]); } }
  })(o.re, o.im);
  return Float32Array.from(out);
}

// integer-shell spectrum: bin b = round(|k|/kunit), b = 0 .. NB-1. NB is tied to the
// dealias cutoff (min(nx,ny)/3), so every retained mode lands in a bin.
// Number of perpendicular shell bins. The bin unit is kunit = min(2pi/Lx, 2pi/Ly) (what
// makeGrid and the spectrum kernel use), and the shells are counted out to the SMALLER of
// the two AXIS dealias cutoffs measured in that unit. Called with the box lengths since
// REFINE_PLAN J.2 (rectangular 2D boxes); with Lx == Ly -- and with the two-argument form
// the 3D app still uses -- it collapses to the pre-J floor(min(nx,ny)/3) exactly.
function nbins(nx, ny, Lx, Ly) {
  const a = Lx || 1, b = Ly || 1, Lm = Math.max(a, b);
  return Math.floor(Math.min((nx / 3) * (Lm / a), (ny / 3) * (Lm / b)));
}

// arrow-overlay subsampling: at most 32x32 arrows. stride = max(1, n/32), so nx=32
// gives stride 1 (the full grid) and 128/256/512 give exactly 32 arrows per axis.
// Shared by the WGSL constants and the JS readback size -- keep them in one place.
function arrowDims(nx, ny) {
  const sx = Math.max(1, Math.floor(nx / 32)), sy = Math.max(1, Math.floor(ny / 32));
  return { sx, sy, nax: Math.max(1, Math.floor(nx / sx)), nay: Math.max(1, Math.floor(ny / sy)) };
}

// ---------------------------------------------------------------------------
// colormaps (one table, two consumers)
// ---------------------------------------------------------------------------
// The GPU colorize kernel (physics.js: CMAP_WGSL, expanded from this table at
// emit time) and the CPU preview of the IC editor (cmapRGB, below) must agree,
// so the numbers exist exactly once -- here.
//
// afmhot is matplotlib's exact closed form. viridis and RdBu are degree-6
// least-squares fits of each channel to matplotlib 3.10's tables sampled at 256
// points (max abs channel error 0.017 viridis, 0.040 RdBu; regenerate with
//   t = np.linspace(0,1,256); rgb = matplotlib.colormaps[name](t)[:, :3]
//   [np.polyfit(t, rgb[:,c], 6)[::-1] for c in range(3)]
// ). A fit overshoots [0,1] a little at the ends, so both paths clamp.
const CMAP_NAMES = ["afmhot", "viridis", "RdBu", "grayscale"];
const CMAP_COEF = {
  viridis: [
    [0.274455, 0.005768, 0.332664],
    [0.107708, 1.396470, 1.386771],
    [-0.327241, 0.214814, 0.091977],
    [-4.599932, -5.758238, -19.291809],
    [6.203736, 14.153965, 56.656300],
    [4.751787, -13.749439, -65.320968],
    [-5.432077, 4.641571, 26.272108]
  ],
  rdbu: [
    [0.372790, 0.010760, 0.105979],
    [4.951873, -0.041506, 1.495240],
    [-25.589376, 12.710598, -16.350017],
    [90.713136, -14.900622, 110.181831],
    [-170.672778, -23.257965, -254.356137],
    [145.286139, 44.207323, 243.328940],
    [-45.023288, -18.545760, -84.065776]
  ]
};
const _clamp01 = v => Math.max(0, Math.min(1, v));
// which = index into CMAP_NAMES, x already mapped into [0,1]; returns [r,g,b] in [0,1]
function cmapRGB(which, x) {
  const t = _clamp01(x);
  const co = which === 1 ? CMAP_COEF.viridis : (which === 2 ? CMAP_COEF.rdbu : null);
  if (co) {
    const out = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      let v = co[6][c];
      for (let k = 5; k >= 0; k--) v = v * t + co[k][c];
      out[c] = _clamp01(v);
    }
    return out;
  }
  if (which === 3) return [t, t, t];
  return [_clamp01(2 * t), _clamp01(2 * t - 0.5), _clamp01(2 * t - 1)];   // afmhot
}

// ---------------------------------------------------------------------------
// WGSL: generic pieces
// ---------------------------------------------------------------------------
function log2i(n) { let k = 0; while ((1 << k) < n) k++; return k; }
function fftWG(N) { let w = Math.min(256, Math.max(32, N >> 1)); let p = 32; while (p < w) p <<= 1; return p; }

// generic radix-2 Stockham autosort line transform in workgroup memory.
// dir = -1 forward (exp(-2 pi i k n / N)), +1 inverse.  LOAD fills buf[0..N-1],
// STORE reads buf[src + idx].
//
// Dispatch note: with `lpb` (lines per batch) set, the transform is dispatched as
// (lines-per-batch, batch) and the line index is reassembled from wgid.x/wgid.y --
// the 3D app needs that because its line counts (up to 8*nz*nx = 131072) blow past
// the maxComputeWorkgroupsPerDimension limit of 65535. Both factors are exact by
// construction, so there is no partial group and hence no early return before a
// workgroupBarrier. Without `lpb` the line index is just wgid.x (the 2D app).
//
// In-place note: a caller may bind ONE read_write buffer and transform it in place.
// That is safe because a workgroup loads its whole line into workgroup memory before
// the first barrier and stores only after the last one, and lines are owned by
// exactly one workgroup. (Two bindings aliasing one buffer is a WebGPU validation
// error anyway, so a ping-pong needs two distinct buffers.)
function fftKernel(o) {
  const { N, dir, decl, load, store, lpb } = o;
  const WG = fftWG(N), LOGN = log2i(N), H = N >> 1;
  const lineExpr = lpb ? `wgid.y * ${lpb}u + wgid.x` : `wgid.x`;
  return `
${decl}
var<workgroup> buf: array<vec2<f32>, ${2 * N}u>;
const PI: f32 = 3.1415926535897932;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let line: u32 = ${lineExpr};
  let tid: u32 = lid.x;
${load}
  var src: u32 = 0u;
  var dst: u32 = ${N}u;
  var p: u32 = 1u;
  for (var s: u32 = 0u; s < ${LOGN}u; s = s + 1u) {
    workgroupBarrier();
    for (var i: u32 = tid; i < ${H}u; i = i + ${WG}u) {
      let k: u32 = i & (p - 1u);
      let j: u32 = ((i - k) << 1u) + k;
      let u0: vec2<f32> = buf[src + i];
      let u1: vec2<f32> = buf[src + i + ${H}u];
      let ang: f32 = ${dir < 0 ? "-" : ""}PI * f32(k) / f32(p);
      let wc: f32 = cos(ang);
      let ws: f32 = sin(ang);
      let t: vec2<f32> = vec2<f32>(wc * u1.x - ws * u1.y, wc * u1.y + ws * u1.x);
      buf[dst + j] = u0 + t;
      buf[dst + j + p] = u0 - t;
    }
    p = p << 1u;
    let tmp: u32 = src; src = dst; dst = tmp;
  }
  workgroupBarrier();
${store}
}`;
}

// The real <-> complex ROW transforms along y. Both pages emitted these as byte-identical
// twins differing in one argument, so they are written once here (render audit,
// 2026-08-12): the forward packs a real line into the rfft half-spectrum, the inverse
// rebuilds the full line from that half by conjugate symmetry and carries the 1/NY
// normalization. The pair IS the rfft2 layout convention that everything else in both
// apps assumes, which is the argument for it having one home rather than two.
// `lpb` (lines per batch) is the 3D page's dispatch split around
// maxComputeWorkgroupsPerDimension; the 2D page passes nothing and gets wgid.x alone.
// The x (and 3D z) transforms are deliberately NOT folded in with these: they are the
// same SHAPE but the 2D page transforms out of place through its ping-pong pair while the
// 3D page has no second buffer that size and goes in place, and the two strided kernels
// name their index variables differently. Parameterizing all of that costs more lines than
// the three call sites spend, and every one of those kernels is pinned byte-for-byte.
function fftRowPair(ny, nky, lpb) {
  const WGF = fftWG(ny);
  const dims = `const NY_: u32 = ${ny}u;
const NKY_: u32 = ${nky}u;`;
  return {
    r2c: fftKernel({
      N: ny, dir: -1, lpb,
      decl: `@group(0) @binding(0) var<storage, read> rin: array<f32>;
@group(0) @binding(1) var<storage, read_write> cout: array<vec2<f32>>;
${dims}`,
      load: `  let ib: u32 = line * NY_;
  for (var idx: u32 = tid; idx < NY_; idx = idx + ${WGF}u) { buf[idx] = vec2<f32>(rin[ib + idx], 0.0); }`,
      store: `  let ob: u32 = line * NKY_;
  for (var idx: u32 = tid; idx < NKY_; idx = idx + ${WGF}u) { cout[ob + idx] = buf[src + idx]; }`
    }),
    // inverse rows: complex half-spectrum -> real (exactly real: .x only)
    c2r: fftKernel({
      N: ny, dir: +1, lpb,
      decl: `@group(0) @binding(0) var<storage, read> cin: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> rout: array<f32>;
${dims}
const SCL: f32 = ${(1 / ny).toExponential(12)};`,
      load: `  let ib: u32 = line * NKY_;
  for (var idx: u32 = tid; idx < NY_; idx = idx + ${WGF}u) {
    var v: vec2<f32>;
    if (idx < NKY_) { v = cin[ib + idx]; }
    else { let c: vec2<f32> = cin[ib + (NY_ - idx)]; v = vec2<f32>(c.x, -c.y); }
    buf[idx] = v;
  }`,
      store: `  let ob: u32 = line * NY_;
  for (var idx: u32 = tid; idx < NY_; idx = idx + ${WGF}u) { rout[ob + idx] = buf[src + idx].x * SCL; }`
    })
  };
}

// tree reduction tail for a 256-thread workgroup accumulating `acc` into sh[]
// (T is the element type, op a binary function name: max / add4 / ...)
function reduceTail(T, op) {
  return `
  sh[tid] = acc;
  workgroupBarrier();
  var stride: u32 = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { sh[tid] = ${op}(sh[tid], sh[tid + stride]); }
    workgroupBarrier();
    stride = stride >> 1u;
  }`;
}

// ---- dimension-agnostic kernels ------------------------------------------
// Each takes the app's `pre` (its grid-size constants) and whatever counts differ
// between the two apps; the emitted WGSL is otherwise identical text.

// t += dt, and the time integral of the dissipation rate
function tickWGSL(pre) {
  return pre + `
@group(0) @binding(0) var<storage, read_write> sc: array<f32>;
@compute @workgroup_size(1)
fn main() {
  let dt: f32 = sc[0];
  sc[1] = sc[1] + dt;
  sc[7] = sc[7] + sc[6] * dt;
}`;
}

// CFL: max over real space of (|d_y phi| + |d_y psi|, |d_x phi| + |d_x psi|)
function cflPartialWGSL(pre) {
  return pre + `
@group(0) @binding(0) var<storage, read> gr: array<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<vec2<f32>>;
var<workgroup> sh: array<vec2<f32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid: u32 = lid.x;
  var acc: vec2<f32> = vec2<f32>(0.0, 0.0);
  var i: u32 = wgid.x * 1024u + tid;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    if (i < NR) {
      let vy: f32 = abs(gr[i]) + abs(gr[2u*NR + i]);        // |d_x phi| + |d_x psi|
      let vx: f32 = abs(gr[NR + i]) + abs(gr[3u*NR + i]);   // |d_y phi| + |d_y psi|
      acc = max(acc, vec2<f32>(vx, vy));
    }
    i = i + 256u;
  }
${reduceTail("vec2<f32>", "max")}
  if (tid == 0u) { part[wgid.x] = sh[0]; }
}`;
}
function cflFinalWGSL(pre, nPart) {
  return pre + `
@group(0) @binding(0) var<storage, read> part: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> sc: array<f32>;
@group(0) @binding(2) var<uniform> cfg: Cfg;
var<workgroup> sh: array<vec2<f32>, 256>;
const NPART: u32 = ${nPart}u;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid: u32 = lid.x;
  var acc: vec2<f32> = vec2<f32>(0.0, 0.0);
  for (var i: u32 = tid; i < NPART; i = i + 256u) { acc = max(acc, part[i]); }
${reduceTail("vec2<f32>", "max")}
  if (tid == 0u) {
    var ma: f32 = max(sh[0].x / cfg.dx, sh[0].y / cfg.dy);
    ma = max(ma, max(0.1 / cfg.dx, 0.1 / cfg.dy));
    sc[0] = cfg.cfl / ma;
  }
}`;
}

// second stage of the energy reduction: (Ekin, Emag, dissipation rate, H_c) -> scalars
// (sc[8] is the cross helicity <u.b>, the fourth energyPartial lane; REFINE_PLAN H.2)
function energyFinalWGSL(pre, nPart) {
  return pre + `
@group(0) @binding(0) var<storage, read> part: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> sc: array<f32>;
var<workgroup> sh: array<vec4<f32>, 256>;
const NPART: u32 = ${nPart}u;
fn add4(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> { return a + b; }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid: u32 = lid.x;
  var acc: vec4<f32> = vec4<f32>(0.0);
  for (var i: u32 = tid; i < NPART; i = i + 256u) { acc = acc + part[i]; }
${reduceTail("vec4<f32>", "add4")}
  if (tid == 0u) {
    sc[2] = 0.5 * sh[0].x * INVN2;
    sc[3] = 0.5 * sh[0].y * INVN2;
    sc[6] = sh[0].z * INVN2;
    sc[8] = sh[0].w * INVN2;
  }
}`;
}

// display autoscale: |.|max over the displayed field (`count` is NR in 2D, the
// single-slice NRS in 3D)
function maxPartialWGSL(pre, count) {
  return pre + `
@group(0) @binding(0) var<storage, read> f: array<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<f32>;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid: u32 = lid.x;
  var acc: f32 = 0.0;
  var i: u32 = wgid.x * 1024u + tid;
  for (var c: u32 = 0u; c < 4u; c = c + 1u) {
    if (i < ${count}) { acc = max(acc, abs(f[i])); }
    i = i + 256u;
  }
${reduceTail("f32", "max")}
  if (tid == 0u) { part[wgid.x] = sh[0]; }
}`;
}
function maxFinalWGSL(pre, nPart) {
  return pre + `
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> mx: array<f32>;
var<workgroup> sh: array<f32, 256>;
const NPART: u32 = ${nPart}u;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid: u32 = lid.x;
  var acc: f32 = 0.0;
  for (var i: u32 = tid; i < NPART; i = i + 256u) { acc = max(acc, part[i]); }
${reduceTail("f32", "max")}
  if (tid == 0u) { mx[0] = sh[0]; }
}`;
}

// (the display-chain kernels -- vecMag, vecGather, cutPrep, colorize and the
// blit -- moved to physics.js, next to prepDisp, so the whole display path is
// templated in one place.)

// ---------------------------------------------------------------------------
// device / buffers
// ---------------------------------------------------------------------------
const LSRK33 = {
  alpha: [0.0, -5.0 / 9.0, -153.0 / 128.0],
  beta: [1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0],
  gamma: [1.0 / 3.0, 5.0 / 12.0, 1.0 / 4.0]
};
// Lazy: GPUBufferUsage does not exist at all when WebGPU is unavailable, and a bare
// top-level reference would kill the whole script (UI wiring included) with no message.
const SQ = (typeof GPUBufferUsage !== "undefined")
  ? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) : 0;

// Staging buffers for the map-read round trips, POOLED by byte length (render audit,
// 2026-08-12). This used to create and destroy one per call, which is a driver allocation
// on every frame the page runs -- the stats read alone -- plus the arrows, the colorbar,
// the cut line and the spectra at their own rates. The set of sizes is small and fixed
// (one per readback, per grid), so the pool cannot grow without bound, and a buffer is
// popped from the free list BEFORE the first await, so two overlapping reads of the same
// size cannot be handed the same one. Staging buffers belong to the device, not to a
// solver: a rebuild retires the solver's buffers and leaves these alone, which is the
// point -- the sizes are the same across rebuilds at one resolution.
// The pool deliberately has no eviction. A resolution switch strands the old sizes'
// buffers, but there are single digits of them, they are kilobytes each (the biggest is
// the field-line sample block), and the alternative -- a lifetime rule tied to the solver
// -- would reintroduce exactly the per-rebuild bookkeeping this removes.
const _stagePool = new Map();
async function readBuf(device, buf, byteLen) {
  let free = _stagePool.get(byteLen);
  if (!free) { free = []; _stagePool.set(byteLen, free); }
  const st = free.pop() ||
    device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(buf, 0, st, 0, byteLen);
  device.queue.submit([e.finish()]);
  try {
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.unmap();
    free.push(st);                      // returned only on the path that really unmapped it
    return out;
  } catch (err) {
    // a rejected map leaves the buffer in an unknown map state: drop it rather than hand
    // it to the next caller (a device-lost page has bigger problems, but a poisoned pool
    // would turn one failure into every later readback's)
    try { st.destroy(); } catch (e2) {}
    throw err;
  }
}

// The cut chart's line readback (REFINE_PLAN H.3). Identical in both apps -- each
// Solver only supplies the cutPrep / cutRows bind groups and (in 3D) the little Mode
// uniform carrying the plane -- so it lives here and each app's readCutLine is one
// line. Returns 4*ny reals: the (u_x, u_y, b_x, b_y) lines on x = Lx/2.
async function cutLineRead(s, iz) {
  const d = s.device;
  if (s.buf.cutM) d.queue.writeBuffer(s.buf.cutM, 0, new Uint32Array([0, iz | 0, 0, 0]));
  const enc = d.createCommandEncoder();
  const p = enc.beginComputePass();
  p.setPipeline(s.pl.cutPrep); p.setBindGroup(0, s.bg.cutPrep);
  p.dispatchWorkgroups(Math.ceil(s.g.nky / 64));
  p.setPipeline(s.pl.rowsC2R); p.setBindGroup(0, s.bg.cutRows);
  p.dispatchWorkgroups(4);
  p.end();
  d.queue.submit([enc.finish()]);
  return readBuf(d, s.buf.cutR, 4 * s.g.ny * 4);
}

// One display chain's contour level tables (REFINE_PLAN I2.4; CONT_SETS of them since
// J2.2, so that psi and phi can be drawn at once). The CPU owns nlev and the plain-
// background flag, the GPU (contLevel) owns the adapting range and the spacing, and a
// ZERO spacing is what switches a set off inside the shared shader -- so turning contours
// off is one 4-byte write, not a pipeline or bind-group change. Identical in both apps.
const CONT_SETS = 2;
// one buffer / uniform / bind group per contour set: the chains are built in the apps,
// the set count lives here
const contPer = f => Array.from({ length: CONT_SETS }, (_, i) => f(i));
// The adapting range (st[0]) belongs to the potential it was measured FROM, and contLevel
// only ever relaxes it downwards (CONT_RELAX per frame). So a set that changes potential
// -- psi -> phi, or off and back on as the other one -- must start its range again, or the
// new potential is contoured at the old one's spacing until the relaxation catches up:
// ~1/CONT_RELAX frames, and in the tearing preset (|psi| ~ 1.6, |phi| ~ 1e-2) that is a
// 100x-too-coarse delta, i.e. no visible contours at all for a couple of seconds
// (FEEDBACK_2026-08-08 P0.2). A zero range is what contLevel reads as "no history": it
// takes the measured max outright on the next frame. contMx needs no reset -- the max
// reduction overwrites it before contLevel reads it. The set -> potential bookkeeping
// lives HERE, next to the only writer of that field, so neither app carries a copy.
function setContLevels(device, D, nlev, plain) {
  const tail = new Float32Array([Math.max(1, nlev | 0), plain ? 1 : 0]);
  const zero = new Float32Array([0]);
  if (!D.contFor) D.contFor = contPer(() => 0);
  for (let i = 0; i < CONT_SETS; i++) {
    device.queue.writeBuffer(D.buf.contB[i], 8, tail);
    if (!D.cont[i]) device.queue.writeBuffer(D.buf.contB[i], 4, zero);
    if (D.cont[i] !== D.contFor[i]) {
      device.queue.writeBuffer(D.buf.contB[i], 0, zero);
      D.contFor[i] = D.cont[i];
    }
  }
}
// Snap every ACTIVE set's adapting range to whatever the coming frame measures, instead
// of relaxing towards it. The relaxation exists to damp flicker on a MOVING field; on a
// still one it is only a spacing that never arrives -- and a still field is exactly what
// the render gate stops drawing, so without this a pause would freeze the contours
// wherever the relaxation had got to. A zero range is the same "no history" the
// potential-change reset writes, so the kernel needs no new state and no new branch:
// contLevel already takes the measured max outright when it reads one. Called on every
// frame drawn while PAUSED, which on a static field is idempotent.
const _contZero = new Float32Array([0]);
function contSettle(device, D) {
  if (!D || !D.buf || !D.buf.contB) return;
  for (let i = 0; i < CONT_SETS; i++) {
    if (D.cont[i]) device.queue.writeBuffer(D.buf.contB[i], 0, _contZero);
  }
}
// ... and the tail of one set's per-frame prep, once the app's own inverse transform has
// put that potential plane where colorize reads it: max |pot| over the plane (the shared
// reduction) -> the level table. Identical in both apps.
function contLevelEncode(p, s, D, nPart, i) {
  p.setPipeline(s.pl.maxPartial); p.setBindGroup(0, D.bg.contMax[i]);
  p.dispatchWorkgroups(nPart);
  p.setPipeline(s.pl.maxFinal); p.setBindGroup(0, D.bg.contFin[i]);
  p.dispatchWorkgroups(1);
  p.setPipeline(s.pl.contLevel); p.setBindGroup(0, D.bg.contLev[i]);
  p.dispatchWorkgroups(1);
}

// shader-module factory with on-page compile diagnostics (WGSL errors are otherwise
// silent until the first dispatch)
function shaderModuleFactory(device) {
  return (code, name) => {
    const m = device.createShaderModule({ code });
    if (m.getCompilationInfo) {
      m.getCompilationInfo().then(info => {
        for (const msg of info.messages) {
          if (msg.type === "error") {
            showStatus("WGSL error in " + name + " (line " + msg.lineNum + "): " + msg.message, "err");
            console.error(name, msg, code);
          }
        }
      });
    }
    return m;
  };
}

let device = null, canvasFormat = "bgra8unorm";
// set once in initGPUTry when the adapter reports itself, read only by contactBuild's
// mailto body. "" is a legitimate steady state (no adapter, or a browser that does not
// expose adapter.info) and every reader must treat it as one.
let gpuInfo = "";
let solver = null, running = false, stepsPerFrame = 1, spsSmooth = 0;
// the sim time of the last stats readback: the readout prints it, and the save/record
// filenames stamp it (item 13), so it is kept once here rather than parsed back out
let simT = 0;

// a display card's canvas -> a configured WebGPU context (cards are created and
// destroyed at runtime, so this is not part of initGPU)
function gpuCanvasCtx(cv) {
  if (!device || !cv || !cv.getContext) return null;
  const c = cv.getContext("webgpu");
  // COPY_SRC on EVERY display card, recording or not (RECASYNC_PLAN, 2026-08-12): leg 1
  // now takes its frames by copying the canvas TEXTURE into a staging buffer instead of
  // building a VideoFrame from the canvas, and this is the ONE place a card's context is
  // configured -- long before any rec press, so the usage cannot be decided per take.
  // RENDER_ATTACHMENT has to be named as well: it is what an ABSENT `usage` defaults to,
  // and an explicit value REPLACES that default rather than adding to it. A texture that
  // is additionally copyable costs nothing measurable; "renders identically" is an
  // on-device eyes item all the same, since this one line is always on.
  if (c) c.configure({ device, format: canvasFormat, alphaMode: "opaque",
                       usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  return c;
}

// adapter + device. Every failure has already put its own advice in #status, so the
// wrapper only has to make that advice look like a page rather than a dead black canvas
// (ONEPAGE_PLAN B.5): one hook, both apps, and only for the paths that return FALSE --
// a device lost mid-run leaves the run that already happened on screen.
async function initGPU(opts) {
  const ok = await initGPUTry(opts);
  if (!ok) gpuFallback();
  return ok;
}
// opts.maxLimits asks the adapter for its own reported limits (always a legal request):
// the 3D app's 8-field gradient stack is 129 MiB at 256^2x64, past the DEFAULT 128 MiB
// storage-binding limit, and since TEARNL the 2D app's `tall` box runs a 1024-point
// transform line, whose 16 KiB of workgroup storage is EXACTLY the default
// maxComputeWorkgroupStorageSize. That one fits at the default and the grid caps there
// (rmhd2d's NMAX_LINE), so this is margin and not a requirement -- which is why the
// requestDevice below still falls back to a default device rather than failing: every
// grid either app offers must remain buildable on the guaranteed minimums.
// Coarse UA sniff for the two failure banners ONLY -- it picks which safe advice to
// show, never gates a feature, so a wrong guess costs one slightly-off sentence.
// (Support facts as of 2026-08: gpuweb Implementation-Status wiki. Chrome/Linux is
// default-on only for Intel Gen12+ (144+) and NVIDIA >=535.183.01 under Wayland
// (147+); Firefox/Linux is Nightly-only, expected stable during 2026; Firefox/Mac is
// Apple Silicon only (145 on macOS 26+, 147 on all versions), Intel Macs Nightly-only.)
// NB Firefox reports "Intel Mac OS X 10.15" on Apple Silicon too, so the Mac branch
// CANNOT sniff the chip -- it asks the reader to look, and must stay that way.
function gpuFailEnv() {
  const ua = navigator.userAgent || "";
  return { linux: /Linux/.test(ua) && !/Android/.test(ua),
           mac:   /Macintosh|Mac OS X/.test(ua),
           firefox: /Firefox\//.test(ua) };
}
async function initGPUTry(opts) {
  if (!navigator.gpu) {
    const env = gpuFailEnv();
    showStatus("WebGPU is not available in this browser. "
      + (env.firefox && env.linux
        ? "Firefox does not ship WebGPU on Linux yet (expected during 2026); on this machine "
          + "a current Chrome may work. "
        : env.firefox && env.mac
          ? "Firefox ships WebGPU on Apple Silicon Macs only (145 on macOS 26+, 147 on all "
            + "macOS versions); on Intel Macs it is still Nightly-only. If the Apple menu > "
            + "About This Mac shows a Processor rather than a Chip, this machine is Intel: "
            + "use Chrome/Edge 113+ or Safari 26+. Otherwise update Firefox. "
          : "Browsers that have it: Chrome/Edge 113+, Firefox 141+ on Windows or 147+ on "
            + "Apple Silicon Macs, and Safari 26+. ")
      + "See the ", "err", WEBGPU_STATUS_URL);
    return false;
  }
  let adapter = null;
  try { adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }); }
  catch (e) { showStatus("requestAdapter failed: " + e.message, "err"); return false; }
  if (!adapter) {
    const env = gpuFailEnv();
    showStatus("No WebGPU adapter found: the browser has WebGPU, but could not use this "
      + "machine's GPU. "
      + (env.linux
        ? "On Linux, Chrome enables WebGPU by default only for Intel Gen12+ GPUs (Chrome 144+) "
          + "and NVIDIA GPUs with driver 535.183.01 or newer in a Wayland session (Chrome 147+). "
          + "Typing chrome://gpu into the address bar shows why it is off on this machine; "
          + "updating the graphics driver (and, for NVIDIA, logging into a Wayland session) is "
          + "the safe fix. "
        : env.firefox
          ? "Firefox's about:support page (Graphics section) shows why; updating the graphics "
            + "driver is the usual fix. "
          : "Check that hardware acceleration is enabled in the browser's settings and that the "
            + "graphics driver is current; typing chrome://gpu into the address bar shows the "
            + "WebGPU status. ")
      + "See the ", "err", WEBGPU_STATUS_URL);
    return false;
  }
  // The one line the contact link (contactBuild) needs out of here: WHICH GPU. A bug
  // report that names the adapter is worth ten that say "it didn't work". `adapter.info`
  // is a recent-Chrome property and the older async `requestAdapterInfo()` has been
  // removed, so this is guarded twice over and may legitimately stay "" -- the contact
  // link must never depend on it being non-empty. No await, no new failure path.
  try {
    const inf = adapter.info;
    if (inf) gpuInfo = [inf.vendor, inf.architecture, inf.device, inf.description]
      .filter(Boolean).join(" ").trim();
  } catch (e) {}
  const need = {};
  if (opts && opts.maxLimits) {
    for (const k of ["maxStorageBufferBindingSize", "maxBufferSize",
                     "maxComputeWorkgroupStorageSize"]) {
      if (adapter.limits && adapter.limits[k]) need[k] = adapter.limits[k];
    }
  }
  try {
    device = Object.keys(need).length
      ? await adapter.requestDevice({ requiredLimits: need })
      : await adapter.requestDevice();
  } catch (e) {
    try { device = await adapter.requestDevice(); }
    catch (e2) { showStatus("requestDevice failed: " + e2.message, "err"); return false; }
  }
  device.addEventListener("uncapturederror", ev => {
    showStatus("WebGPU error: " + ev.error.message, "err");
    console.error(ev.error);
  });
  device.lost.then(info => showStatus("WebGPU device lost: " + info.message, "err"));
  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  return true;
}

// ---------------------------------------------------------------------------
// charts (plain 2D canvas, no libraries)
// ---------------------------------------------------------------------------
const EW = 420, EH = 200, SW = 420, SH = 240, CW = 420, CH = 140;
const PADE = { l: 54, r: 10, t: 10, b: 20 };
const PADS = { l: 54, r: 10, t: 10, b: 22 };
const PADC = { l: 54, r: 10, t: 10, b: 20 };
// The generated 2D spectrum card gets a box of its own (Alfred, second feedback round): a
// heatmap is read as a MAP, and SW x SH makes its plot area a 356 x 208 letterbox. Width
// stays at SW deliberately -- a chart canvas is drawn in logical pixels and then scaled to
// the card's width, so changing it would change the on-screen size of the 10 px chart font
// relative to every other card. The height is then whatever makes the frame INSIDE PADS
// exactly SQUARE, derived rather than typed so the two can never drift: 10 + 356 + 22 =
// 388. Nothing in the card grid has to move for it -- .card.chart is width: 100% and
// canvas.chart is width: 100%, height: auto on an aspect-ratio box, so this card is simply
// ~1.6x taller in its column than the spectrum card above it, at every viewport.
const GSW = SW, GSH = PADS.t + (GSW - PADS.l - PADS.r) + PADS.b;
const COL = { ek: "#6fb3ff", em: "#ff9c6b", et: "#9ee493",
              zp: "#c58cf5", zm: "#5fd7c8",          // the Elsasser pair, in every chart
              grid: "#232833", grid2: "#1a1e26", axis: "#39404d", txt: "#8a94a3",
              guide: "#7d8798", shell: "#5a6472", cut: "#c9d4e2" };
const HIST_MAX = 2000;
// hc is the cross helicity <u.b> (energyPartial's fourth lane): with it the same
// history serves both trace modes, E+- = E_kin + E_mag +- H_c.
const hist = { t: [], ek: [], em: [], hc: [] };

// a chart card's canvas -> a 2D context at the right backing-store resolution.
// The CSS width is left to the stylesheet (100% inside the card), so charts
// reflow with the card; only the intrinsic aspect ratio is fixed here.
function chartCtx(cv, w, h) {
  const dpr = Math.min(2, (typeof window !== "undefined" && window.devicePixelRatio) || 1);
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const c = cv.getContext("2d");
  if (!c) return null;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);   // set once; all drawing is in logical px
  c.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  c.textBaseline = "alphabetic";
  return c;
}
// ---------------------------------------------------------------------------
// vector overlay: a transparent 2D canvas sitting exactly on top of a display
// card's WebGPU canvas
// ---------------------------------------------------------------------------
// Orientation. The render fragment samples the field texture with v = 1 - uv.y, so
// texel row iy sits at screen row iy counted DOWNWARD (and column ix at screen x = ix,
// counted rightward). The 2D canvas y axis also points down, so the arrow for grid
// point (ix,iy) goes at (ix,iy) with no position flip, and the physical +y component
// -- which points DOWN on screen because of that same texture flip -- is drawn as
// canvas dy = +uy. Relative to a y-up plotting convention that is a sign flip of uy;
// here it is exactly what keeps the arrows consistent with the colour field.
const VEC_SIZE = 512;
// the overlay context of ONE display card. All drawing is in a logical w-by-h space
// (VEC_SIZE square unless the card is aspect-correct for a rectangular box -- J.2); the
// logical size rides on the context so drawArrows needs no extra argument.
function vecCtx(cv, w, h) {
  const dpr = Math.min(2, (typeof window !== "undefined" && window.devicePixelRatio) || 1);
  const W = w || VEC_SIZE, H = h || VEC_SIZE;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const c = cv.getContext("2d");
  if (c) { c.setTransform(dpr, 0, 0, dpr, 0, 0); c.__w = W; c.__h = H; }
  return c;
}
// `X` is the overlay's affine frame: (u,v) in [0,1]^2 -> canvas pixels, applied to the
// arrow ANCHORS and -- unless the frame carries its own `d` sub-frame -- to their
// DIRECTIONS. The default frame is the plane view (u right, v down, the whole square
// canvas) and reproduces the pre-I2 pixels exactly; a cube card passes the projection of
// its visible top face (REFINE_PLAN I2.3), which is an affine parallelogram -- so one
// implementation draws both views.
//
// `d` exists for the RECTANGULAR boxes of REFINE_PLAN J.2. Anchors are grid FRACTIONS, so
// they must go through the (non-square) anchor frame; the gathered vectors are PHYSICAL,
// so on an aspect-correct card (pixels/length equal on both axes) they must go through an
// ISOTROPIC map instead -- otherwise a 4:1 box would shear every arrow. `sc` renormalizes
// to the longest arrow, so only the SHAPE of `d` matters, not its overall scale.
//
// The overlay is CLEARED by its owner (DisplayCard.overlay), not here: since REFINE_PLAN
// K/K2 the same canvas can instead carry the projected field lines (arrows and lines are
// mutually exclusive per view), and the single owner keeps the clear-then-draw order.
const ARROW_FRAME = { ox: 0, oy: 0, ax: VEC_SIZE, ay: 0, bx: 0, by: VEC_SIZE };
function drawArrows(c, a, nax, nay, X) {
  const F = X || ARROW_FRAME, Dm = F.d || F;
  let mx = 0;
  for (let i = 0; i < nax * nay; i++) {
    const m = Math.hypot(a[2 * i], a[2 * i + 1]);
    if (m > mx) mx = m;
  }
  if (!isFinite(mx) || mx <= 0) return;
  // longest arrow ~ 0.9 * subsample cell (in cell fractions of the frame, so the
  // square default gives 0.9 * min(W/nax, H/nay) px exactly as before)
  const sc = 0.9 * Math.min(1 / nax, 1 / nay) / mx;
  const ca = Math.cos(2.6), sa = Math.sin(2.6);
  const path = new Path2D();
  for (let ix = 0; ix < nax; ix++) {
    for (let iy = 0; iy < nay; iy++) {
      const i = ix * nay + iy;
      const dx = sc * (a[2 * i] * Dm.ax + a[2 * i + 1] * Dm.bx);
      const dy = sc * (a[2 * i] * Dm.ay + a[2 * i + 1] * Dm.by);
      const len = Math.hypot(dx, dy);
      if (!(len > 0.6)) continue;             // sub-pixel (or NaN) -> skip
      const u = (ix + 0.5) / nax, v = (iy + 0.5) / nay;
      const x0 = F.ox + u * F.ax + v * F.bx - 0.5 * dx;
      const y0 = F.oy + u * F.ay + v * F.by - 0.5 * dy;
      const x1 = x0 + dx, y1 = y0 + dy;
      const ex = dx / len, ey = dy / len, hl = Math.min(0.4 * len, 4);
      path.moveTo(x0, y0); path.lineTo(x1, y1);
      path.moveTo(x1, y1);
      path.lineTo(x1 + hl * (ex * ca - ey * sa), y1 + hl * (ex * sa + ey * ca));
      path.moveTo(x1, y1);
      path.lineTo(x1 + hl * (ex * ca + ey * sa), y1 + hl * (ey * ca - ex * sa));
    }
  }
  // afmhot spans black -> white, so every arrow gets a dark halo underneath
  c.lineCap = "round"; c.lineJoin = "round";
  c.strokeStyle = "rgba(0,0,0,0.55)"; c.lineWidth = 2.6; c.stroke(path);
  c.strokeStyle = "rgba(190,255,255,0.8)"; c.lineWidth = 1.1; c.stroke(path);
}

// ---------------------------------------------------------------------------
// 3D magnetic field lines: the box frame and the polylines (REFINE_PLAN K.2, K2.3)
// ---------------------------------------------------------------------------
// Both take `F`, the box's own oblique projection (rmhd3d's cubeFrame): box fractions
// (x, y, z) -> canvas pixels, the same three vectors the cube faces are drawn with, so
// everything lands inside the box. Since K2 these are the LINES VIEW's overlay and
// nothing else's: the background under them is the flat contour plate, which is what
// their colours are tuned for (dark stroke, light halo -- legible where a line crosses
// the thin psi ink or the phi accent of the transparent top face).
//
// The frame is the twelve box edges: four parallel to each axis, at the four
// combinations of the other two axes' 0/1 ends.
function drawBoxFrame(c, F) {
  if (!F) return;
  const P = q => [F.ox + q[0] * F.ax + q[1] * F.bx + q[2] * F.cx,
                  F.oy + q[0] * F.ay + q[1] * F.by + q[2] * F.cy];
  const path = new Path2D();
  for (let e = 0; e < 3; e++) {
    for (let q = 0; q < 4; q++) {
      const a = [0, 0, 0];
      a[(e + 1) % 3] = q & 1; a[(e + 2) % 3] = q >> 1;
      const b = a.slice();
      b[e] = 1;
      const p0 = P(a), p1 = P(b);
      path.moveTo(p0[0], p0[1]); path.lineTo(p1[0], p1[1]);
    }
  }
  c.lineCap = "round"; c.lineJoin = "round";
  c.strokeStyle = "rgba(40,48,60,0.45)"; c.lineWidth = 1.0; c.stroke(path);
}
// `L.pos` is the GPU marcher's output: L.nl polylines of L.nz points, each the line's
// PERPENDICULAR position in box fractions and UNWRAPPED (the marcher never folds it), so
// a line that leaves the box shows up as a change in floor(u) -- which is exactly the
// place to lift the pen instead of drawing a stripe across the picture.
// No depth sorting and no occlusion -- 2D canvas, one pass, everything visible. `alpha`
// (default 1) scales both strokes: over the volume view they are drawn faint, because
// there they cross a picture that has its own depth (ISO_PLAN B).
const VOL_LINE_ALPHA = 0.45;
function drawFieldLines(c, L, F, alpha) {
  if (!F) return;
  const al = alpha === undefined ? 1 : alpha;
  const pos = L.pos, nz = L.nz;
  const path = new Path2D();
  for (let l = 0; l < L.nl; l++) {
    let pu = 0, pv = 0, brk = true;
    for (let k = 0; k < nz; k++) {
      const u = pos[2 * (l * nz + k)], v = pos[2 * (l * nz + k) + 1];
      const fu = Math.floor(u), fv = Math.floor(v), w = k / nz;
      const x = F.ox + (u - fu) * F.ax + (v - fv) * F.bx + w * F.cx;
      const y = F.oy + (u - fu) * F.ay + (v - fv) * F.by + w * F.cy;
      if (!isFinite(x) || !isFinite(y)) { brk = true; continue; }
      if (brk || fu !== pu || fv !== pv) path.moveTo(x, y); else path.lineTo(x, y);
      pu = fu; pv = fv; brk = false;
    }
  }
  c.lineCap = "round"; c.lineJoin = "round";
  c.strokeStyle = "rgba(255,255,255," + (0.85 * al) + ")"; c.lineWidth = 2.6; c.stroke(path);
  c.strokeStyle = "rgba(20,60,150," + (0.90 * al) + ")"; c.lineWidth = 1.1; c.stroke(path);
}

const _histCols = () => [hist.t, hist.ek, hist.em, hist.hc];
function histReset() { for (const a of _histCols()) a.length = 0; }
function histPush(t, ek, em, hc) {
  if (hist.t.length >= HIST_MAX) {          // full -> keep every other sample
    for (const a of _histCols()) {
      let w = 0;
      for (let i = 0; i < a.length; i += 2) a[w++] = a[i];
      a.length = w;
    }
  }
  hist.t.push(t); hist.ek.push(ek); hist.em.push(em); hist.hc.push(hc || 0);
}
// keep line coordinates finite but far outside the clip, so the visible slope of a
// series leaving the plot is untouched
const px = v => Math.max(-1e4, Math.min(1e4, v));
function chartFrame(c, W, H, P) {
  c.clearRect(0, 0, W, H);
  c.fillStyle = "#0f1115"; c.fillRect(0, 0, W, H);
  c.strokeStyle = COL.axis; c.lineWidth = 1;
  c.strokeRect(P.l + 0.5, P.t + 0.5, W - P.l - P.r - 1, H - P.t - P.b - 1);
}
// ---------------------------------------------------------------------------
// axis ticks, shared by EVERY chart (FEEDBACK_2026-08-08 item 5)
// ---------------------------------------------------------------------------
// One gridline + one label, in the two orientations the charts need. `major` picks
// the brighter grid colour; a falsy label draws the line only. Both leave textAlign
// set the way the label needed it, so callers that print anything else afterwards
// set their own alignment (they all already did).
function xTick(c, x, y0, y1, ylab, label, major) {
  const xr = Math.round(x) + 0.5;
  c.strokeStyle = major ? COL.grid : COL.grid2; c.lineWidth = 1;
  c.beginPath(); c.moveTo(xr, y0); c.lineTo(xr, y1); c.stroke();
  if (!label) return;
  c.fillStyle = COL.txt; c.textAlign = "center"; c.fillText(label, xr, ylab);
}
function yTick(c, y, x0, x1, label, major) {
  const yr = Math.round(y) + 0.5;
  c.strokeStyle = major ? COL.grid : COL.grid2; c.lineWidth = 1;
  c.beginPath(); c.moveTo(x0, yr); c.lineTo(x1, yr); c.stroke();
  if (!label) return;
  c.fillStyle = COL.txt; c.textAlign = "right"; c.fillText(label, x0 - 5, yr + 3);
}
// Tick VALUES for a log axis running from 10^vlo to 10^vhi across `pxPerDec` pixels
// per decade: the decades are the majors (thinned by a stride so at most ~7 labels
// land on any axis), and inside each decade the 2..9 minors go in as soon as the
// decade is wide enough to be worth subdividing -- standard log-axis practice, and
// what makes a 1.5-decade spectrum readable at all. The subdivision has two tiers --
// 2/3/5 on a narrow decade, the full 2..9 once there is room for them -- and only
// mantissae 2 and 5 are ever labelled, on a wide decade, so labels never collide on
// a narrow card. `fmt(mantissa, decade)` renders a label; returns [value, major,
// label] in increasing value order (no sort: majors precede their decade's minors).
const MINOR_MIN_PX = 45, MINOR_ALL_PX = 90, MINOR_LABEL_PX = 95;
const MINOR_MANT = [2, 3, 4, 5, 6, 7, 8, 9], MINOR_FEW = [2, 3, 5];
const LOG_FMT = (m, d) => (m === 1 ? "1e" + d : m + "e" + d);
function logTicks(vlo, vhi, pxPerDec, fmt) {
  const lab = fmt || LOG_FMT, out = [];
  const d0 = Math.ceil(vlo - 1e-9), d1 = Math.floor(vhi + 1e-9);
  const stride = Math.max(1, Math.ceil((d1 - d0 + 1) / 7));
  const minor = stride === 1 && pxPerDec >= MINOR_MIN_PX;
  const mlab = minor && pxPerDec >= MINOR_LABEL_PX;
  const mant = pxPerDec >= MINOR_ALL_PX ? MINOR_MANT : MINOR_FEW;
  for (let d = d0 - 1; d <= d1; d++) {
    if (d >= d0 && (d - d0) % stride === 0) out.push([Math.pow(10, d), true, lab(1, d)]);
    if (!minor) continue;
    for (const m of mant) {
      const lv = d + Math.log10(m);
      if (lv <= vlo || lv >= vhi) continue;
      out.push([m * Math.pow(10, d), false, mlab && (m === 2 || m === 5) ? lab(m, d) : ""]);
    }
  }
  return out;
}
// Tick FRACTIONS 0..1 of a linear axis: `n` equal intervals, every other one a
// labelled major, so the linear charts get the same major/minor texture.
const linTicks = n => Array.from({ length: n + 1 }, (_, i) => [i / n, i % 2 === 0]);

// items: [label, colour, dash?]. Wraps onto further lines at xmax, so a chart with
// four series plus a guide line still shows every key.
// An entry is [label, colour, dash] -- or a GROUP, a list of such triples in entry[0],
// which is laid out as ONE unwrappable unit: several swatch + label pairs that always share
// a line. (Alfred, second gen2d round: the three theory slopes want to read "GS95 B06 iso"
// on one line, each with its own dash and colour, rather than as three stacked entries.)
// The wrap test is made on the group's TOTAL width, so a group either fits on the current
// line or starts a new one and is never broken across two. A plain triple is a group of
// one, which is why every other chart's legend is untouched by this.
function legend(c, x, y, items, xmax) { return _legendRun(c, x, y, items, xmax, true); }
// The same layout WITHOUT drawing, returning the number of lines it takes. A chart that
// must keep empty canvas under its own legend has to know that height before it can fix
// its axes, and the only honest way to know it is to run the real layout (the wrapping
// depends on measureText, on the labels and on the card's width).
function legendLines(c, x, items, xmax) { return _legendRun(c, x, 0, items, xmax, false); }
function _legendRun(c, x, y, items, xmax, draw) {
  let lx = x, ly = y, lines = 1;
  const segw = s => 15 + c.measureText(s[0]).width + 11;
  for (const it of items) {
    const segs = Array.isArray(it[0]) ? it[0] : [it];
    let w = 0;
    for (const s of segs) w += segw(s);
    if (xmax && lx > x && lx + w > xmax) { lx = x; ly += 12; lines++; }
    for (const s of segs) {
      if (draw) {
        c.strokeStyle = s[1]; c.lineWidth = 2; c.setLineDash(s[2] || []);
        c.beginPath(); c.moveTo(lx, ly - 3); c.lineTo(lx + 12, ly - 3); c.stroke();
        c.setLineDash([]);
        c.fillStyle = s[1]; c.fillText(s[0], lx + 15, ly);
      }
      lx += segw(s);
    }
  }
  return lines;
}

// The two energy-trace modes (REFINE_PLAN H.2), from the SAME history:
//   kmt  E_u, E_b, E_tot                              (the default: unchanged)
//   pmt  E+, E-, E_tot with E+- = E_u + E_b +- H_c
// so E_tot = (E+ + E-)/2, which is the repo's Elsasser convention (taranis
// physics/rmhd.py) and what puts all three curves on one comparable axis.
const ENERGY_MODES = {
  kmt: [["E_u", COL.ek, i => hist.ek[i]], ["E_b", COL.em, i => hist.em[i]],
        ["E_tot", COL.et, i => hist.ek[i] + hist.em[i]]],
  pmt: [["E+", COL.zp, i => hist.ek[i] + hist.em[i] + hist.hc[i]],
        ["E-", COL.zm, i => hist.ek[i] + hist.em[i] - hist.hc[i]],
        ["E_tot", COL.et, i => hist.ek[i] + hist.em[i]]]
};
// ONE time-series chart drawer, shared by the energy trace and the island-width trace
// (REFINE_PLAN J.4): both are "n series over a common t axis with an automatic log / linear
// y". `ts` is the time column, `series` is [[label, colour, i -> value], ...] and
//   o.empty   the placeholder line while there is nothing to draw
//   o.log     true forces a log y axis (the island trace, where the straight linear-stage
//             line IS the point); otherwise log is chosen when every value is positive and
//             the dynamic range is at least 3x -- the pre-J energy-chart rule.
function drawTimeSeries(c, W, H, P, ts, series, o) {
  if (!c) return;
  const x0 = P.l, x1 = W - P.r, y0 = P.t, y1 = H - P.b;
  chartFrame(c, W, H, P);
  const n = ts.length;
  c.textAlign = "left"; c.fillStyle = COL.txt;
  if (n < 2) { c.fillText(o.empty, x0 + 6, y0 + 13); return; }
  let lo = Infinity, hi = -Infinity, allPos = true;
  for (let i = 0; i < n; i++) {
    for (const sr of series) {
      const v = sr[2](i);
      if (!(v > 0)) allPos = false;
      lo = Math.min(lo, v); hi = Math.max(hi, v);
    }
  }
  if (!isFinite(lo) || !isFinite(hi)) return;
  const useLog = allPos && (o.log || hi / lo >= 3);
  let vlo, vhi;
  if (useLog) { vlo = Math.log10(lo); vhi = Math.log10(hi); }
  else { vlo = Math.min(0, lo); vhi = hi; }
  if (!(vhi > vlo)) vhi = vlo + (useLog ? 1 : Math.max(1e-30, Math.abs(vlo)));
  const pad = 0.07 * (vhi - vlo); vlo -= pad; vhi += pad;
  const t0 = ts[0], t1 = ts[n - 1], tspan = (t1 - t0) || 1;
  const X = t => x0 + (t - t0) / tspan * (x1 - x0);
  const Y = v => px(y1 - ((useLog ? Math.log10(Math.max(v, 1e-300)) : v) - vlo) / (vhi - vlo) * (y1 - y0));

  if (useLog) {
    for (const tk of logTicks(vlo, vhi, (y1 - y0) / (vhi - vlo)))
      yTick(c, Y(tk[0]), x0, x1, tk[2], tk[1]);
  } else {
    for (const tk of linTicks(8)) {
      const v = vlo + (vhi - vlo) * tk[0];
      yTick(c, Y(v), x0, x1, tk[1] ? v.toExponential(1) : "", tk[1]);
    }
  }
  // x (time): gridlines across the panel, the two ENDS labelled flush with the frame
  // (a centred label there would hang outside it) and the interior majors centred.
  const ft = v => v.toFixed(tspan < 0.5 ? 3 : 2);
  for (const tk of linTicks(8)) {
    if (tk[0] === 0 || tk[0] === 1) continue;
    xTick(c, X(t0 + tspan * tk[0]), y0, y1, H - 6, tk[1] ? ft(t0 + tspan * tk[0]) : "", tk[1]);
  }
  c.strokeStyle = COL.grid; c.fillStyle = COL.txt; c.textAlign = "right"; c.lineWidth = 1;
  c.fillText(ft(t1), x1, H - 6);
  c.textAlign = "left";
  c.fillText("t = " + ft(t0), x0, H - 6);

  c.save();
  c.beginPath(); c.rect(x0, y0, x1 - x0, y1 - y0); c.clip();
  c.lineWidth = 1.25;
  for (const sr of series) {
    c.strokeStyle = sr[1]; c.beginPath();
    for (let i = 0; i < n; i++) {
      const x = X(ts[i]), y = Y(sr[2](i));
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.stroke();
  }
  c.restore();
  legend(c, x0 + 6, y0 + 12, series.map(s => [s[0], s[1]]), x1 - 40);
  c.fillStyle = COL.txt; c.textAlign = "right";
  c.fillText(useLog ? "log y" : "lin y", x1 - 5, y0 + 12);
  c.textAlign = "left";
}
function drawEnergy(c, o) {
  drawTimeSeries(c, EW, EH, PADE, hist.t,
                 ENERGY_MODES[(o && o.emode)] || ENERGY_MODES.kmt,
                 { empty: "energy vs t — collecting…" });
}

// ---------------------------------------------------------------------------
// island width W(t) (REFINE_PLAN J.4)
// ---------------------------------------------------------------------------
// Same history discipline as the energy trace, filled from the CUT readback (islandPush
// in the frame loop) rather than from the stats one, because W is measured on the
// resonant line. Log y is forced: the linear tearing stage is then a straight line, the
// Rutherford stage bends over to algebraic, and saturation flattens.
const islandHist = { t: [], w: [] };
function islandReset() { islandHist.t.length = 0; islandHist.w.length = 0; }
function drawIsland(c) {
  if (!c) return;
  if (!icEq.on) {
    chartFrame(c, CW, EH, PADC);
    c.textAlign = "left"; c.fillStyle = COL.txt;
    c.fillText("island width — needs the tearing IC preset", PADC.l + 6, PADC.t + 13);
    return;
  }
  // the same instrument the KH mode chart carries (FEEDBACK_2026-08-10 item 9), on the
  // same trailing window and the same R^2 gate; the factor 2 is islandFitGamma's, since
  // W ~ psitilde^(1/2) ~ e^(gamma t / 2).
  const g = islandFitGamma(islandHist.t, islandHist.w);
  drawTimeSeries(c, CW, EH, PADC, islandHist.t,
                 [["W" + (isFinite(g) ? "   γ_fit ≈ " + g.toFixed(4) : ""), COL.zp,
                   i => islandHist.w[i]]],
                 { log: true, empty: "island width W(t) — collecting…" });
}

// ---------------------------------------------------------------------------
// the k_y = 2pi/Ly mode amplitude A(t) -- the KH linear stage
// ---------------------------------------------------------------------------
// The energy trace cannot show that stage at all: the KH equilibrium carries ~1e6 times
// the seed's energy, so E(t) is flat while the mode grows through six decades. What this
// plots instead is the one quantity that is PURE perturbation -- the m = 1 Fourier
// amplitude of u_x (and of b_x) along the same x = Lx/2 line the cut card already reads.
// u_x = -d_y phi, and the equilibrium is y-independent with its flow along y, so u_x has
// exactly ZERO equilibrium content; b_x = -d_y psi likewise. Log y is forced, so the
// linear stage is a straight line whose slope is gamma (fitted into the legend below).
// Same history discipline as islandHist, filled from the same cut readback.
const modeHist = { t: [], u: [], b: [] };
function modeReset() { modeHist.t.length = 0; modeHist.u.length = 0; modeHist.b.length = 0; }
// The gamma fit's trailing window is SIM-TIME, not a sample count: the cut readback is
// wall-clock-throttled (~10 Hz), so a fixed count would span wildly different stretches
// of t on different devices -- on a fast one, more than the whole linear stage
// (~ln(1/seed)/gamma ~ 26 t-units at the defaults), always mixing noise floor, growth
// and roll-up and reading systematically low. 10 t-units is ~2.7/gamma at the KH
// reference rate: comfortably inside the linear stage, a few samples on any device.
const MODE_FIT_DT = 10;
// ... but a sim-time window sampled on a wall clock can come up EMPTY. Samples inside the
// window are 100 / (sim-units per second), so a small grid with a large dt starves it:
// `tearing` at selRes 256 runs ~19 sim-units/s (~6 samples) and a quicker machine reaches
// 38 (~3), below the 4 a slope needs, so the legend blanks in the middle of a clean linear
// stage. fitLogSlope therefore widens the window until it holds MODE_FIT_N samples, and no
// further than MODE_FIT_DT_MAX. Both numbers are about the FIT's conditioning, not about
// physics: 8 points is where a least-squares slope with an R^2 gate stops being swayed by
// one sample, and 4x the window is as far as any preset's linear stage can be assumed to
// extend (tearing's spans ~190 t-units, collapse's ~10 -- but collapse never widens,
// having ~42 samples). The alternative fix, sampling on sim-time too, would mean extra GPU
// readbacks, which is the trade LOOPLAT closed the door on.
const MODE_FIT_N = 8;
const MODE_FIT_DT_MAX = 4 * MODE_FIT_DT;
// ... and a slope is only QUOTED when that window saw real, clean growth: at least
// MODE_FIT_RISE of ln A end to end (~0.43 decades -- the fp32 noise floor's jitter and
// the saturated stage's oscillation cannot fake it), with an R^2 of at least
// MODE_FIT_R2, so a window straddling two stages stays blank instead of quoting a
// number that is an average of neither.
const MODE_FIT_RISE = 1.0;
const MODE_FIT_R2 = 0.98;
// The island chart's own rise gate (FEEDBACK_2026-08-10 item 9). The window and the R^2
// gate are shared; only this number can be, because the two demos run four times a
// decade apart in rate and the gate is a statement about ONE window's worth of growth:
//
//   KH       gamma = 0.267, the fitted quantity is A itself, so a MODE_FIT_DT window
//            rises gamma*DT = 2.67 ln-units. The gate 1.0 is 0.37 of that.
//   tearing  gamma = 0.0287 AND the fitted quantity is W ~ psitilde^(1/2), so the same
//            window rises gamma*DT/2 = 0.143 ln-units -- a factor 19 less. MODE_FIT_RISE
//            would never be met inside a linear stage that only spans ~2.7 ln-units of W
//            in total (W0 ~ 0.09 at the preset's 1e-3 seed, saturation at W ~ a ~ 1.3),
//            i.e. ~190 t-units: the legend would simply never show a number.
//
// So the gate is MODE_FIT_RISE scaled to the tearing demo's own rate: 0.37 * 0.143 = 0.05.
// That keeps the SAME margin over the gate the KH chart has (~2.7x), so the fit still
// arms promptly with maintain-flux OFF too, where the measured slope drops 30-40%
// (0.094 ln-units per window, still ~1.9x the gate), while an oscillating saturated W
// -- which is what the gate exists to reject -- has to clear 5% of coherent, R^2 >= 0.98
// rise across the whole window to fake it.
// It is a LOCAL rate by construction: through the Rutherford stage (W ~ t, so
// d ln W/dt = 1/t) the quoted gamma falls away from the linear value and eventually
// blanks, which is the honest reading of a chart whose y is log W.
const ISLAND_FIT_RISE = 0.05;
function drawMode(c) {
  if (!c) return;
  if (!icEq.kh) {
    chartFrame(c, CW, EH, PADC);
    c.textAlign = "left"; c.fillStyle = COL.txt;
    c.fillText("k_y mode amplitude — needs the KH IC preset", PADC.l + 6, PADC.t + 13);
    return;
  }
  const g = modeFitGamma(modeHist.t, modeHist.u);
  // b_x is IDENTICALLY zero at b0 = 0, and a log axis cannot carry that: drawTimeSeries
  // falls back to a LINEAR y the moment one plotted value is <= 0, which would flatten
  // exactly what this chart exists to show. So the b series is left out entirely until it
  // has something positive, and a zero inside a live series is floored far below its own
  // smallest positive sample rather than dragging the axis.
  let bmin = Infinity;
  for (const v of modeHist.b) if (v > 0 && v < bmin) bmin = v;
  const series = [["A_u" + (isFinite(g) ? "   γ_fit ≈ " + g.toFixed(3) : ""), COL.ek,
                   i => modeHist.u[i]]];
  if (isFinite(bmin)) series.push(["A_b", COL.em, i => Math.max(modeHist.b[i], 1e-3 * bmin)]);
  drawTimeSeries(c, CW, EH, PADC, modeHist.t, series,
                 { log: true, empty: "k_y = 2π/L_y mode A(t) — collecting…" });
}

// The binned data is THREE quantities per bin, [E_u | E_b | H_c] (the spectra kernels'
// third accumulator lane), from which the Elsasser spectra follow without a second
// kernel: E+-(k) = E_u(k) + E_b(k) +- H_c(k). Which of the four curves a card draws is
// its `sq` option; `sd` picks perpendicular (solid), parallel (dashed) or both.
//   d.perp   3*nb shell values, bin 0 = the (zero-energy) DC shell
//   d.par    3*nzb parallel values for |kz| bin 1..nzb (3D only)
//   d.parKfac converts a kz bin index to the same k/kunit units (so with a cubic box
//            the two abscissae coincide)
const SPEC_SETS = {
  ub: [["E_u", COL.ek, (u, b, h) => u], ["E_b", COL.em, (u, b, h) => b]],
  pm: [["E+", COL.zp, (u, b, h) => u + b + h], ["E-", COL.zm, (u, b, h) => u + b - h]]
};

// ---------------------------------------------------------------------------
// the TRUE parallel spectrum, along the field lines (REFINE_PLAN K.3)
// ---------------------------------------------------------------------------
// The GPU marcher samples (u_x, u_y, b_x, b_y) at every z plane of every line, uniformly
// in z (arc length = z to leading order in RMHD). The signal is NOT periodic -- a line
// leaves the box perpendicularly displaced, so its two ends are unrelated -- hence a Hann
// window before the transform; the window's mean square W2 divides back out, so the
// binned values keep the meaning (and the units) of the COORDINATE E(k_par) they are
// plotted beside, and Parseval still holds: the bins sum to the along-line mean square.
//
// The three lanes are the ones every spectrum here carries, [E_u | E_b | H_c] with
// E+- = E_u + E_b +- H_c (SPEC_SETS), so the chart code needs no field-line branch at all:
//   E_u(k) = 1/2 (P[u_x] + P[u_y]),  E_b likewise,  H_c(k) = C[u_x,b_x] + C[u_y,b_y]
// with P the folded power and C the folded co-spectrum. Bins are |kz| = 1..nz/2 in the
// SAME layout as the coordinate parallel spectrum (kz = 0 dropped: no place on a log
// axis), so drawSpectrum's parKfac abscissa is shared too.
// In-place radix-2 complex transform; sign -1 = forward, unnormalized.
function fftPow2(re, im, sign) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let b = n >> 1;
    for (; j & b; b >>= 1) j ^= b;
    j ^= b;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = sign * 2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), h = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < h; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + h] * cr - im[i + k + h] * ci;
        const vi = re[i + k + h] * ci + im[i + k + h] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + h] = ur - vr; im[i + k + h] = ui - vi;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}
// periodic Hann window and its mean square (exactly 3/8 for n >= 3; computed, not assumed)
function flHann(n) {
  const w = new Float64Array(n);
  let s = 0;
  for (let j = 0; j < n; j++) { w[j] = 0.5 * (1 - Math.cos(2 * Math.PI * j / n)); s += w[j] * w[j]; }
  return { w, w2: s / n };
}
function flSpectrum(smp, nl, nz) {
  const nzb = nz >> 1;
  const out = new Float32Array(3 * Math.max(1, nzb));
  if (!(nl > 0) || nzb < 1) return out;
  const H = flHann(nz), nrm = 1 / (nz * nz * H.w2 * nl);
  const re = [], im = [];
  for (let c = 0; c < 4; c++) { re.push(new Float64Array(nz)); im.push(new Float64Array(nz)); }
  for (let l = 0; l < nl; l++) {
    for (let c = 0; c < 4; c++) {
      for (let j = 0; j < nz; j++) { re[c][j] = H.w[j] * smp[4 * (l * nz + j) + c]; im[c][j] = 0; }
      fftPow2(re[c], im[c], -1);
    }
    for (let b = 1; b <= nzb; b++) {
      const f = (2 * b === nz) ? 1 : 2;            // the kz-Nyquist bin has no -kz partner
      let eu = 0, eb = 0, hc = 0;
      for (let c = 0; c < 2; c++) {                // the two perpendicular components
        eu += re[c][b] * re[c][b] + im[c][b] * im[c][b];
        eb += re[c + 2][b] * re[c + 2][b] + im[c + 2][b] * im[c + 2][b];
        hc += re[c][b] * re[c + 2][b] + im[c][b] * im[c + 2][b];
      }
      out[b - 1] += 0.5 * f * eu * nrm;
      out[nzb + b - 1] += 0.5 * f * eb * nrm;
      out[2 * nzb + b - 1] += f * hc * nrm;
    }
  }
  return out;
}
const specSeries = sq => (sq === "both" ? SPEC_SETS.ub.concat(SPEC_SETS.pm)
                                        : (SPEC_SETS[sq] || SPEC_SETS.ub));
// ---------------------------------------------------------------------------
// the spectrum chart's LOWER y limit (FEEDBACK_2026-08-08 item 4)
// ---------------------------------------------------------------------------
// A hyper-dissipative tail runs 15+ decades below the peak, and a floor at the
// smallest drawn value squashes the whole inertial range into a couple of pixels.
// So the floor is pinned to a fraction of the spectrum's own value near the
// dissipation scale, MEASURED off the spectrum rather than derived from eps/diss --
// nothing is assumed about where the energy comes from, exactly as for the
// amplitude-based auto-diss (P2 item 6), so decaying and instability presets are
// handled by the same rule as a forced run:
//   k_d    the first k ABOVE THE CURVE'S OWN PEAK at which the drawn spectrum has
//          fallen SPEC_KNEE decades below the chart's peak. The dissipation fall-off
//          is far steeper than any inertial slope, so k_d sits at the knee whatever
//          the slope above it is; it is a plain crossing, so a non-monotone bump
//          cannot skip it. The walk MUST start at the peak: a rising spectrum (an
//          inverse cascade, peak at the last bin) or a mid-k-peaked one with a deep
//          low-k side would otherwise take the crossing on the low-k side of the peak
//          and read a "dissipation scale" of k = 1.
//   E_a    the spectrum at SPEC_KFRAC * k_d, i.e. a fixed FRACTION of k_d, still
//          inside the cascade; the floor is SPEC_TAIL decades below that.
//   E_pre  the smallest value on the LOW-k side of the peak. The tail rule is about
//          the dissipation range only; low-k content is demo-relevant physics (the
//          inverse cascade of item P2 5, a forcing shell with a quiet large-scale
//          range below it) and must stay inside the frame, so the floor is the MIN of
//          the tail rule and E_pre -- never higher than the deepest pre-peak bin.
// A spectrum that never falls SPEC_KNEE decades (instability presets, early frames,
// a flat/empty chart) has no knee and simply keeps its measured minimum -- which by
// construction is then already less than SPEC_KNEE decades of range. A curve lying
// wholly below the crossing (a series 4+ decades quieter than the loudest one) has no
// knee of its own and is skipped, so it cannot drag k_d down to its first bin.
// SPEC_MAXDEC is the hard floor that keeps the panel readable no matter what the
// crossing and the pre-peak minimum find.
const SPEC_KNEE = 4, SPEC_KFRAC = 0.5, SPEC_TAIL = 3, SPEC_MAXDEC = 9;
// this curve's own peak bin, as an index into its flat (k, v) array (-1 when it is
// empty or carries nothing positive)
function _specPeak(a) {
  let p = -1, pv = 0;
  for (let i = 0; i < a.length; i += 2) if (a[i + 1] > pv) { pv = a[i + 1]; p = i; }
  return p;
}
// k_d, the dissipation knee itself, factored out of the floor rule because a SECOND
// consumer needs exactly this crossing and must not be allowed to invent its own: the
// anisotropy card's level window ends where the cascade does (ANISO_PLAN). Everything
// the paragraph above says about the walk -- start at the peak, plain crossing, a curve
// wholly below the threshold has no knee of its own -- is this function. Infinity means
// no knee: the caller decides what that means for it.
// `rc` are the range-setting curves, each [ [k, v, k, v, ...], ... ] in k order.
function specKnee(rc, hi) {
  const thr = hi * Math.pow(10, -SPEC_KNEE);
  let kd = Infinity;
  for (const cv of rc) {
    const a = cv[0], p = _specPeak(a);
    if (p < 0 || a[p + 1] <= thr) continue;      // empty, or wholly below the crossing
    for (let i = p + 2; i < a.length; i += 2)    // the knee: walk RIGHT from the peak
      if (a[i + 1] <= thr) { kd = Math.min(kd, a[i]); break; }
  }
  return kd;
}
function specFloor(rc, hi, lo) {
  const thr = hi * Math.pow(10, -SPEC_KNEE);
  const kd = specKnee(rc, hi);
  let pre = Infinity;
  for (const cv of rc) {
    const a = cv[0], p = _specPeak(a);
    if (p < 0 || a[p + 1] <= thr) continue;      // (the same curves the knee walk keeps)
    for (let i = 0; i < p; i += 2) pre = Math.min(pre, a[i + 1]);   // low-k side of the peak
  }
  if (!isFinite(kd)) return lo;                 // no knee: the data set their own floor
  const ka = SPEC_KFRAC * kd;
  let ea = 0;
  for (const cv of rc) {                        // E_a = each curve AT k_a (its last bin <= k_a)
    const a = cv[0];
    let v = 0;
    for (let i = 0; i < a.length; i += 2) { if (a[i] > ka) break; v = a[i + 1]; }
    ea = Math.max(ea, v);
  }
  if (!(ea > 0)) ea = hi;                       // knee in the very first bins
  return Math.max(Math.min(ea * Math.pow(10, -SPEC_TAIL), pre),
                  hi * Math.pow(10, -SPEC_MAXDEC));
}
// The y floor the CHART draws to, which is specFloor gated by the card's `clip` option
// plus the SPEC_MAXDEC clamp that applies either way. Factored out of drawSpectrum so the
// choice is testable without a canvas, and so there is exactly one place that knows
// unclipping drops the knee rule and NOT the sanity limit.
function specYFloor(rc, hi, lo, clip) {
  return Math.max(clip === "off" ? lo : specFloor(rc, hi, lo), hi * Math.pow(10, -SPEC_MAXDEC));
}

// ---------------------------------------------------------------------------
// the spectrum chart's FIT LINE (FEEDBACK_2026-08-08 item 8)
// ---------------------------------------------------------------------------
// The fixed k^-5/3 guide becomes a straight line E = A k^p with a settable index p and a
// settable amplitude A, per CHART CARD (so two cards can carry two slopes over the same
// spectrum), plus the mode it always had: PIN the amplitude to the field and only choose
// the index. The controls are the card's own, beside `sq` / `sd`.
//
// The index box holds a plain decimal, which cannot spell -5/3 exactly, so a value within
// FIT_SNAP of one of the two indices this subject actually argues about snaps to the
// exact fraction -- for the drawn line AND for the legend, which is why one function
// answers both. Anything else is drawn and labelled as the number it is.
//
// WHICH fractions those are is per USE, not per file: the spectrum card argues about
// -5/3 and -3/2, the anisotropy card (ANISO_PLAN) about -1/3 (Goldreich-Sridhar critical
// balance), -1/2 (dynamic alignment) and -1 (what the coordinate-z measure trends to once
// field-line wander decorrelates the z frame). So the table is an argument, defaulting to
// the spectrum's pair -- which keeps every existing call site, and its snapping, exactly
// as it was. `sym` names the abscissa in the legend, k on the spectrum and k_perp here.
const FIT_SNAP = 5e-3;
const FIT_FRACS = [[-5 / 3, "-5/3"], [-3 / 2, "-3/2"]];
const FIT_FRACS_ANISO = [[-1 / 3, "-1/3"], [-1 / 2, "-1/2"], [-1, "-1"]];
function _fitFrac(p, fr) {
  for (const f of (fr || FIT_FRACS)) if (Math.abs(p - f[0]) < FIT_SNAP) return f;
  return null;
}
function fitIndex(v, fr) {
  const p = parseFloat(v);
  if (!isFinite(p)) return (fr || FIT_FRACS)[0][0];  // blank / NaN box: the default index
  const f = _fitFrac(p, fr);
  return f ? f[0] : p;
}
function fitLabel(p, fr, sym) {
  const f = _fitFrac(p, fr);
  return (sym || "k") + "^" + (f ? f[1] : String(Math.round(p * 1000) / 1000));
}
// Where the fit line is anchored, and where the anisotropy card's level window starts:
// just above the forcing shell, which is where the fixed k^-5/3 guide was anchored long
// before either was settable. An unforced run has no shell worth speaking of, and the
// max(2, ...) is what then leaves the anchor at the second bin. ONE rule, one site --
// the two cards must not drift apart on where "the inertial range starts" is.
function fitKA(nb, fshell) {
  return Math.max(2, Math.min(nb - 1, Math.round(fshell[1])));
}
// A in E = A k^p, pinned to a drawn series at the anchor wavenumber `ka`: its first point
// at or above ka. 0 when the series has no such point -- an empty or still-filling chart
// draws no line, and the AUTOMATIC anchor below uses that same 0 as its fallback signal.
function fitAnchor(pts, ka, p) {
  for (let i = 0; i < pts.length; i += 2) if (pts[i] >= ka) return pts[i + 1] * Math.pow(pts[i], -p);
  return 0;
}
// WHERE the automatic ("pin to field") amplitude anchors -- on the spectrum card and on the
// anisotropy card, through this one function so the two cannot drift apart (Alfred,
// 2026-08-11). Not at kA any more, but at an INTERMEDIATE scale: halfway LOGARITHMICALLY
// between the box scale and the dissipation scale, i.e. their geometric mean
//     k_match = sqrt(k_box * k_diss),   k_box = FIT_KBOX = 1.
// The fit line is a statement about the WHOLE resolved range, so it is pinned in the middle
// of that range rather than at one end of it: anchored just above the forcing shell (where
// it used to be, and where a forced run is least self-similar -- the injection bump sits
// exactly there) an index that is a little off throws all of its error into the dissipation
// end, and a wobble at the outer scale swings the entire line.
//   k_box  is the box FUNDAMENTAL: 1, in the same kunit the abscissa of both cards is in.
//          Deliberately NOT kA -- "box" is the largest scale in the run, not the largest
//          scale the line happens to be drawn over.
//   k_diss is the dissipation knee, specKnee's peak-then-walk-right crossing, measured by
//          the CALLER on the range-setting curves that card already uses -- the one knee
//          rule in this file, never a second opinion.
// Two fallbacks to the old kA anchor, both of them reached through fitAnchor's 0:
//   * NO KNEE IN VIEW -- specKnee returns Infinity (an early frame, an instability preset,
//     any spectrum that never falls SPEC_KNEE decades). k_match is then Infinity, no drawn
//     point is at or above it, and the anchor falls back to kA.
//   * k_match PAST THE END of the drawn series -- a curve that stops short of it (a
//     still-filling chart, a card drawing a narrow window). Same 0, same fallback.
// fitAnchor's sampling convention is untouched (first point at or above the anchor k), so
// this is a change of WHERE the amplitude is read off and of nothing else. It is the
// automatic path only: a user-set amplitude ("amp") never comes through here, and "off"
// draws no line at all.
const FIT_KBOX = 1;
function fitKMatch(kd) { return Math.sqrt(FIT_KBOX * kd); }
function fitAnchorAuto(pts, kA, kd, p) {
  const a = fitAnchor(pts, fitKMatch(kd), p);
  return a > 0 ? a : fitAnchor(pts, kA, p);
}
// ---------------------------------------------------------------------------
// the spectrum chart's CURVES, as data (PINCURVE Phase A)
// ---------------------------------------------------------------------------
// The front half of drawSpectrum -- series selection, the perpendicular / parallel
// point assembly, the two range pools -- with no canvas and no DOM in it. That seam
// is what lets a PIN snapshot exactly what the card is about to draw (Phase B) and
// what lets node test the assembly directly.
// Returns `curves` as [points (k, v pairs), colour, dash, label] with the
// PERPENDICULAR ones first (`nPerp` of them, so a caller can tell the two apart),
// the perpendicular extremes hiP/loP, and hi/lo -- the parallel extremes, overwritten
// by the perpendicular pair whenever that carries anything, which IS the y-range rule
// below.
function specCurves(d, o) {
  const bins = (d && d.perp) || new Float32Array(3), nb = (d && d.nb) || 1;
  const parKfac = (d && d.parKfac) || 1;
  const set = specSeries(o && o.sq);
  const sd = (o && o.sd) || "both";
  // "fl" swaps the COORDINATE parallel spectrum for the field-line one (REFINE_PLAN K.3):
  // same three lanes, same bins, same abscissa -- only the source differs, so nothing
  // below this line knows which it is drawing beyond the legend label.
  const fl = sd === "fl";
  const par = d && (fl ? d.parFL : d.par);
  const wantPerp = sd !== "par", wantPar = sd !== "perp" && par && par.length >= 3;
  // curves: [points(k,v pairs), colour, dash, label]
  const curves = [];
  // y limits come from the PERPENDICULAR spectra alone (Alfred 2026-08-06): the dashed
  // E(k_par) curves are plotted inside those limits and never stretch them. With only
  // the parallel spectra selected there is nothing else to scale to, so they set it.
  let hi = 0, lo = Infinity, hiP = 0, loP = Infinity;
  if (wantPerp) {
    for (const sr of set) {
      const pts = [];
      for (let b = 1; b < nb; b++) {
        const v = sr[2](bins[b], bins[nb + b], bins[2 * nb + b]);
        if (v > 0 && isFinite(v)) { pts.push(b, v); hiP = Math.max(hiP, v); loP = Math.min(loP, v); }
      }
      curves.push([pts, sr[1], null, sr[0]]);
    }
  }
  if (wantPar) {
    const nzb = Math.floor(par.length / 3);
    for (const sr of set) {
      const pts = [];
      for (let b = 1; b <= nzb; b++) {   // |kz| bins; kz = 0 has no place on a log axis
        const v = sr[2](par[b - 1], par[nzb + b - 1], par[2 * nzb + b - 1]);
        if (v > 0 && isFinite(v)) { pts.push(b * parKfac, v); hi = Math.max(hi, v); lo = Math.min(lo, v); }
      }
      curves.push([pts, sr[1], [5, 3], sr[0] + (fl ? "(k∥ line)" : "(k∥)")]);
    }
  }
  if (hiP > 0) { hi = hiP; lo = loP; }
  return { curves, hiP, loP, hi, lo, nPerp: wantPerp ? set.length : 0 };
}
// one polyline per curve, in the caller's current alpha / width: the live set and the
// pinned ghosts differ only in those two, so the stroking is written once
function specStroke(c, curves, X, Y) {
  for (const cv of curves) {
    const a = cv[0];
    if (a.length < 4) continue;
    c.strokeStyle = cv[1]; c.setLineDash(cv[2] || []); c.beginPath();
    for (let i = 0; i < a.length; i += 2) {
      const x = X(a[i]), y = Y(a[i + 1]);
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.stroke(); c.setLineDash([]);
  }
}
// ---------------------------------------------------------------------------
// pinned ghost spectra (PINCURVE)
// ---------------------------------------------------------------------------
// A pin freezes the card's drawn curves as ghosts under the live ones -- the comparison
// primitive the lessons need (lower the dissipation and compare, change eps and compare,
// pin the decayed spectrum and restart forced, pin at t = 0 and watch the inverse
// cascade). Entirely CPU-side: no kernel, no buffer, no readback of its own.
// Four is the cap (a fifth press is refused, like a fifth display card); each ghost keeps
// its own hue, faded by AGE -- index 0 is the newest.
const PIN_MAX = 4;
const PIN_ALPHA = [0.45, 0.34, 0.26, 0.20];
// Ghosts re-registered on PHYSICAL k: pinned x values are in k/kunit of the pin-time
// grid, so a box change (standard <-> wide 4pi x 2pi) that moves kunit moves them by
// kunit_pin / kunit_live and they stay where the physics put them. With an unchanged box
// the factor is exactly 1. With no live grid to register against (a card drawing ghosts
// alone inside the spectrum throttle) it is 1 as well -- there is nothing to move onto.
// Returns the drawable form of `pins`: rescaled curves + the alpha its age earns.
function pinDraw(pins, kuLive) {
  const out = [];
  const n = pins ? pins.length : 0;
  for (let i = 0; i < n; i++) {
    const p = pins[i];
    const f = (kuLive > 0 && p.kunit > 0) ? p.kunit / kuLive : 1;
    out.push({
      curves: p.curves.map(cv => {
        const a = cv[0].slice();
        if (f !== 1) for (let j = 0; j < a.length; j += 2) a[j] *= f;
        return [a, cv[1], cv[2], cv[3]];
      }),
      nPerp: p.nPerp, t: p.t,
      alpha: PIN_ALPHA[Math.min(n - 1 - i, PIN_ALPHA.length - 1)]
    });
  }
  return out;
}
// the largest k/kunit any ghost reaches, i.e. the x axis a ghosts-only card needs
function pinKmax(P) {
  let m = 0;
  for (const p of P) for (const cv of p.curves) if (cv[0].length) m = Math.max(m, cv[0][cv[0].length - 2]);
  return m;
}
function drawSpectrum(c, d, o, pins) {
  if (!c) return;
  const P = PADS, x0 = P.l, x1 = SW - P.r, y0 = P.t, y1 = SH - P.b;
  chartFrame(c, SW, SH, P);
  c.textAlign = "left"; c.fillStyle = COL.txt;
  const nbLive = (d && d.nb) || 1;
  const fshell = (d && d.fshell) || [1, 3];
  const S = specCurves(d, o);
  const curves = S.curves, nPerp = S.nPerp;
  // the ghosts, already registered on the live grid's physical k
  const G = pinDraw(pins, (d && d.kunit) || 0);
  // Pinned series join the y-range pools exactly as live ones do -- the comparison the pin
  // exists for must not sit off the axis. Which means pinned PARALLEL ghosts never stretch
  // the range either, for the same reason live parallel never does (Alfred 2026-08-06):
  // the perpendicular pool WINS whenever it carries anything, live or pinned. They pool
  // with the live parallel curves only on a par-only card, where the parallel curves are
  // the range-setters -- without that, such a card would fall back to "waiting…" the
  // moment its live data lapsed, with ghosts sitting right there.
  // (a pin whose own hiP > 0 also contributes its perpendicular hi/lo to the parallel
  // pool here; harmless, because hiP > 0 is exactly when the override discards that pool.)
  let hiP = S.hiP, loP = S.loP, hi = S.hi, lo = S.lo;
  for (const p of pins || []) {
    hiP = Math.max(hiP, p.hiP); loP = Math.min(loP, p.loP);
    hi = Math.max(hi, p.hi); lo = Math.min(lo, p.lo);
  }
  if (hiP > 0) { hi = hiP; lo = loP; }
  // A card with ghosts but no live data yet (just after an IC reset, inside the ~300 ms
  // spectrum throttle, or straight after a preset transplant) draws axes + ghosts alone,
  // out to the largest k the ghosts reach.
  const nb = nbLive >= 2 ? nbLive : Math.max(2, Math.ceil(pinKmax(G)));
  if (nb < 2 || !(hi > 0)) { c.fillText("spectra — waiting…", x0 + 6, y0 + 13); return; }
  // the curves the y range is read off: the perpendicular ones whenever they carry
  // anything, else (par-only cards) the parallel ones, which are the tail of `curves`
  const rc = hiP > 0
    ? curves.slice(0, nPerp).concat(...G.map(p => p.curves.slice(0, p.nPerp)))
    : curves.slice(nPerp).concat(...G.map(p => p.curves.slice(p.nPerp)));
  const ymax = Math.log10(hi) + 0.3;
  // The floor rule, or the data's own minimum when the card asks for the full range.
  //
  // `clip` OFF is not a debug view: specFloor's tail rule is calibrated on a HYPER-
  // DISSIPATIVE tail, which runs 15+ decades below the peak and would squash the inertial
  // range into a few pixels (FEEDBACK_2026-08-08 item 4). At hyper = 1 -- which every
  // tearing-family preset locks -- the dissipation range is gentle, only a decade or two
  // below where the rule puts the floor, and it is exactly what has to be read: a bump at
  // the dealias end is how an under-resolved run is diagnosed, and clipped it looks
  // identical to a clean one. So the presets that lock hyper = 1 turn the clip off and
  // every other preset keeps it, rather than one rule pretending to serve both.
  // SPEC_MAXDEC still applies in BOTH states: unclipping drops the knee/tail rule, not the
  // sanity limit, or an early frame whose tail is at the fp32 noise floor (the seed is
  // 1e-3 of the equilibrium) would draw a 12-decade axis nobody can read.
  // at least one decade always, so a flat or single-valued spectrum still has an axis
  const ymin = Math.min(ymax - 1,
    Math.log10(specYFloor(rc, hi, lo, o && o.clip)) - 0.3);
  const xmax = Math.log10(nb);
  const X = k => x0 + Math.log10(k) / xmax * (x1 - x0);
  const Y = v => px(y1 - (Math.log10(v) - ymin) / (ymax - ymin) * (y1 - y0));

  // y decades (+ 2..9 minors once the range is down to a few decades)
  for (const tk of logTicks(ymin, ymax, (y1 - y0) / (ymax - ymin)))
    yTick(c, Y(tk[0]), x0, x1, tk[2], tk[1]);
  // x decades + 2..9 minors, labelled as the integer k they are; the last-bin label
  // owns the right edge, so any tick label that would run into it stays unlabelled
  for (const tk of logTicks(0, xmax, (x1 - x0) / Math.max(xmax, 1e-9), (m, d) => String(m * Math.pow(10, d))))
    xTick(c, X(tk[0]), y0, y1, SH - 8, X(tk[0]) > x1 - 20 ? "" : tk[2], tk[1]);
  c.fillStyle = COL.txt; c.textAlign = "center";
  c.fillText(String(nb), x1, SH - 8);
  c.textAlign = "left";
  c.fillText("k / kunit", x0 + 4, SH - 8);

  c.save();
  c.beginPath(); c.rect(x0, y0, x1 - x0, y1 - y0); c.clip();
  // forcing shell markers
  c.strokeStyle = COL.shell; c.setLineDash([2, 3]); c.lineWidth = 1;
  for (const kf of fshell) {
    if (kf >= 1 && kf <= nb) {
      const x = Math.round(X(kf)) + 0.5;
      c.beginPath(); c.moveTo(x, y0); c.lineTo(x, y1); c.stroke();
    }
  }
  c.setLineDash([]);
  // the ghosts, UNDER everything live: oldest first, so the newest pin lies on top. Each
  // keeps its own hue and its own dash (pinned parallel stays dashed) and is drawn thin
  // and faded, so it reads as a record and never competes with the live curve over it.
  // Forcing markers and the fit line are NOT pinned: the fit line already answers "what
  // slope should this be", the ghost answers "where was the curve".
  c.lineWidth = 1;
  for (const p of G) { c.globalAlpha = p.alpha; specStroke(c, p.curves, X, Y); }
  c.globalAlpha = 1;
  // the fit line E = A k^p, DRAWN from kA (just above the forcing shell, where the old
  // fixed guide was anchored) to the last bin. `fit` = pin / amp / off; in "amp" mode an
  // empty or non-positive amplitude box falls back to the pinned anchor, so switching to it
  // never blanks the line before the user has typed anything.
  // Its automatic AMPLITUDE is pinned elsewhere: at fitKMatch's geometric mean of the box
  // wavenumber and the dissipation knee, with kA as the fallback. The knee comes off `rc`,
  // the very curves the y floor was just measured from -- one knee rule, one measurement.
  const kA = fitKA(nb, fshell);
  const kdFit = specKnee(rc, hi);
  const fitMode = (o && o.fit) || "pin";
  const fitP = fitIndex(o && o.fitp);
  let anch = 0;
  if (fitMode !== "off") {
    anch = fitAnchorAuto(nPerp && curves.length ? curves[0][0] : [], kA, kdFit, fitP);
    const A = parseFloat(o && o.fita);
    // "amp" only ever draws over a DRAWN perpendicular series (hiP > 0): on a par-only
    // card the axis is k_par*parKfac, where a perp-bin fit line would be meaningless
    if (fitMode === "amp" && isFinite(A) && A > 0 && hiP > 0) anch = A;
  }
  if (anch > 0) {
    c.strokeStyle = COL.guide; c.setLineDash([5, 4]);
    c.beginPath();
    c.moveTo(X(kA), Y(anch * Math.pow(kA, fitP)));
    c.lineTo(X(nb), Y(anch * Math.pow(nb, fitP)));
    c.stroke(); c.setLineDash([]);
  }
  c.lineWidth = 1.4;
  specStroke(c, curves, X, Y);
  c.restore();
  const items = curves.filter(cv => cv[0].length >= 4).map(cv => [cv[3], cv[1], cv[2]]);
  if (anch > 0) items.push([fitLabel(fitP), COL.guide, [4, 3]]);
  // ONE collapsed entry for the ghosts, not one per pinned curve. The plan left this open
  // "after seeing it at SW x SH": a 3D card on "both" x "perp + par" draws eight curves,
  // and pushing `label @t=12.3` per pinned curve through this legend's own wrap at
  // SW = 420 takes it from 2 rows to 5 (one pin) and to 14 (four) -- a baseline at y = 178
  // in a panel whose plot ends at y = 218. Collapsed it is 2 rows to 3, worst case. The
  // ghosts already carry the live curves' colours, so the only thing the legend has to add
  // is WHICH MOMENTS are frozen.
  if (G.length) items.push([G.length + " pinned @t=" + G.map(p => p.t.toFixed(1)).join(", "), COL.txt]);
  legend(c, x0 + 6, y0 + 12, items, x1 - 30);
}

// ---------------------------------------------------------------------------
// scale-dependent ANISOTROPY: k_par(k_perp)/k_perp against k_perp (ANISO_PLAN)
// ---------------------------------------------------------------------------
// The critical-balance plot, and entirely CPU arithmetic on spectra that are already in
// hand: the perpendicular shell spectrum, the COORDINATE parallel one (k_par along z)
// and -- whenever the field-line sampler is running -- the along-B one (REFINE_PLAN K.3).
// No kernel, no buffer, no readback of its own; the card rides the spectrum readback the
// way `island` rides the cut line.
//
// k_par(k_perp) is read off by EQUAL-AMPLITUDE MATCHING of the two 1D spectra, drawn on
// their cumulative TAILS
//     Q_perp(k) = sum_{k' >= k} E_perp(k'),   Q_par(k) = sum_{k' >= k} E_par(k'),
// and NOT on the spectral densities. That choice IS the method, so here is the arithmetic
// behind it. Take GS95: E_perp ~ k_perp^-5/3 and E_par ~ k_par^-2. Matching the DENSITIES,
// E_perp(k_perp) = E_par(k_par), gives k_perp^-5/3 = k_par^-2, i.e. k_par ~ k_perp^5/6 --
// a law manufactured by the method, since what critical balance actually asserts is
// k_par ~ k_perp^2/3. What matches physically is energy CONTENT, k E(k) or its integral:
// Q_perp ~ k_perp^-2/3 against Q_par ~ k_par^-1 gives k_perp^-2/3 = k_par^-1, hence
// k_par ~ k_perp^2/3 -- the CB relation, and a RATIO k_par/k_perp ~ k_perp^-1/3, which is
// this card's default fit index. (The 5/6 trap would show up as a ratio slope of -1/6
// instead; devtools/checkaniso.js asserts the difference at a tolerance that excludes it.)
// The tail is preferred over k E(k) itself for two reasons: it is MONOTONE by
// construction, so every matching level has exactly ONE crossing per leg however bumpy
// the live spectrum is, and log-log interpolation between its bins gives sub-bin k_par
// resolution -- which is what softens the integer-k_z quantization at the low-k_perp end.
//
// Both parallel spectra drop the parallel mean (the coordinate one has no kz = 0 bin, the
// field-line one no b = 0 bin) and both already carry the perpendicular one's energy
// units -- the self-test's "sum E(k_perp) vs E_tot" / "sum E(k_par) <= E_tot" rows pin the
// coordinate pair to one normalization, and flSpectrum's W2 division keeps Parseval along
// the lines. So the tails are matched AS THEY ARE, with no renormalization of parFL: a
// residual constant offset between the two normalizations would move the field-line curve
// VERTICALLY on a log axis and cannot change its slope, and the slope is the only thing
// this chart is read for.
//
// Gauge caveat, for the hint and the manual: under the RMHD rescaling symmetry the
// absolute value of k_par/k_perp is a convention (it moves with Lz). Only the slope is
// physical -- and so is the DIVERGENCE of the two curves, which is the Cho-Vishniac
// lesson: the coordinate-z measure saturates at the outer-scale k_z because field-line
// wander decorrelates the z frame, and trends towards ratio ~ k_perp^-1, while the
// field-line measure keeps reporting the real scaling.
//
// ---------------------------------------------------------------------------
// ... and the SECOND ordinate the same matching supports: chi (CHI_PLAN)
// ---------------------------------------------------------------------------
// That gauge caveat is the card apologising for its own y axis, and chi is the quantity it
// is pointing at. Under the rescaling z -> z/eps, Z -> eps Z, t -> t/eps both k_par and db
// pick up a factor eps, so k_par/k_perp moves with Lz while
//     chi(k_perp) = k_perp * db(k_perp) / (k_par(k_perp) * v_A)
// does not. Same readback, same two tails, one more multiplication and a different label --
// and critical balance stops being "a slope of -1/3 in a quantity whose level is a
// convention" and becomes "this is O(1) across the inertial range", which is what CB
// actually asserts. So it is an ORDINATE OPTION on this card (`ay`), not a second card:
// the window, the lane select, the level loop and the legend are all already here.
//
// v_A = 1 in these units (rmhd3d.html: "the line equation is dx_perp/dz = b_perp/B0 with
// B0 = v_A = 1"), and chi = (k_perp/k_par) * db is a pure RATIO times db, so it does not
// care which wavenumber unit the two ks are in as long as it is the same one -- which it
// is: parKfac puts the parallel bins in the perpendicular kunit. Hence chi = kp*sqrt(Q)/kz.
//
// DELTA-B IS THE MATCHED LEVEL ITSELF: db^2 = Q, the tail energy content above k_perp --
// the very currency the matching is built on ("what matches physically is energy CONTENT",
// above). It is defined at every matched level by construction, so it costs no second
// interpolation; and it is a CONVENTION, a factor-of-two choice of what "db at k_perp"
// means. The hint states it rather than hiding it, because chi is an O(1) quantity and
// every O(1) quantity in this subject carries an O(1) convention: the FLATNESS is the sharp
// claim, the level is an order of magnitude and never gets two significant figures.
//
// THE ELSASSER PAIRING. The counterpropagating field is what shears you, so
//     chi+- = k_perp Z-+(k_perp) / (k_par+- v_A),
// and exactly one thing crosses lanes: db. The (kp, kz) matching stays INSIDE the selected
// lane, because that pair IS the measurement of k_par+-(k_perp) from Z+-'s own geometry;
// db is then a lookup of the OPPOSITE lane's perpendicular tail AT kp (_anisoQAt, the
// inverse direction of _anisoAt). Building the perpendicular tail from the opposite lane
// wholesale is the adjacent mistake and a different measurement: it moves which k_perp
// every level maps to, i.e. it measures k_par+ against Z-'s geometry. On the tot lane db^2
// is the matched Q itself and the lookup never happens. All of this is invisible in a
// balanced run and is the whole content of the card on the imbalanced and collision
// presets, so devtools/checkaniso.js section 3 pins kp's PROVENANCE and not just the sign
// of the asymmetry.
//
// WHAT IT SHOWS. The window is kA to the dissipation knee -- of order half a decade in a
// browser-sized forced run -- so a flat line is a weak plot on its own and the content is
// in the departures: chi rising above 1 towards the forcing shell (the outer scale is
// driven, not critically balanced), the roll-off at the knee, the two lanes splitting when
// eps+ != eps-, and chi_z peeling away from chi_B. That last one is NOT an independent
// measurement: the same db divides both curves, so chi_B/chi_z = r_z/r_B exactly. It is the
// Cho-Vishniac divergence of the ratio ordinate, replotted on the axis it bears on.
//
// ESTIMATOR BIAS. flSpectrum must window before transforming (a field line's two ends are
// unrelated), and while the W2 division restores the total variance it does not restore the
// SHAPE: the Hann kernel spreads each parallel line into its neighbours (2/3, 1/6, 1/6),
// which fattens the parallel marginal, makes Q_par decay more slowly and hands back k_par a
// little high -- so chi comes back a little low. checkaniso.js section 8 measures that end
// to end on a synthetic field with a prescribed ridge and reports alpha = chi_meas/chi_true;
// it is REPORTED, never gated, and its job is to keep the hint's wording honest. The
// calibration that looks cheap is null by construction: sweeping Lz cannot expose the bias,
// because the forcing is scattered onto the kz = +-2pi/Lz planes, so the ridge sits at the
// same BIN INDICES whatever Lz is.
const ANISO_NLEV = 16;
// the level of the chi ordinate's reference line -- the CB expectation, an order of
// magnitude and not a fitted number, which is why it is a constant and not a fit index
const ANISO_CHI_REF = 1;
// which energy the tails are built from -- the three lanes every spectrum here carries,
// with E+- = E_u + E_b +- H_c (SPEC_SETS), so this needs no second binning either
const ANISO_LANES = {
  tot: (u, b, h) => u + b,
  zp: (u, b, h) => u + b + h,
  zm: (u, b, h) => u + b - h
};
// the lane whose amplitude shears the selected one, for chi's db lookup. Keyed by the
// OPTION VALUE rather than by the lane function, so `tot` has no opposite (db is its own
// matched level) and neither has an unrecognised aq, which falls back to tot above.
const ANISO_OPP = { zp: "zm", zm: "zp" };
// one leg's (k, E) pairs out of a three-lane bin stack of `n` bins. `j0` is the bin index
// sitting at k = kf: 1 for the parallel stacks, whose bin 0 IS |kz| = 1, and 0 for the
// perpendicular one, whose bin 0 is the (zero-energy) DC shell and is skipped. Bins that
// are not strictly positive and finite are dropped -- which is also what makes the tail
// below strictly decreasing.
function _anisoLeg(a, n, j0, kf, lane) {
  const pts = [];
  if (!a || !(n >= 1)) return pts;
  for (let j = (j0 ? 0 : 1); j < n; j++) {
    const v = lane(a[j], a[n + j], a[2 * n + j]);
    if (v > 0 && isFinite(v)) pts.push((j + j0) * kf, v);
  }
  return pts;
}
// the cumulative tail, as parallel ks[] / qs[] arrays in k order. qs is STRICTLY
// decreasing (every kept bin is positive), which is exactly the property that makes the
// inversion below single-valued on a noisy, non-monotone spectrum.
function _anisoTail(pts) {
  const n = pts.length >> 1, ks = new Float64Array(n), qs = new Float64Array(n);
  let s = 0;
  for (let i = n - 1; i >= 0; i--) { s += pts[2 * i + 1]; ks[i] = pts[2 * i]; qs[i] = s; }
  return { ks, qs, n };
}
// the k at which a tail crosses the level Q, by log-log interpolation between the two
// bracketing bins. NO EXTRAPOLATION PAST THE GRID: a level above the tail's first bin (or
// below its last) would put k outside the resolved range, so it is refused with 0 and the
// caller drops that level rather than fabricating a point.
function _anisoAt(T, Q) {
  if (!(T.n > 0) || !(Q > 0) || Q > T.qs[0] || Q < T.qs[T.n - 1]) return 0;
  let i = 0;
  while (i + 1 < T.n && T.qs[i + 1] >= Q) i++;
  if (i + 1 >= T.n) return T.ks[T.n - 1];
  const dq = Math.log(T.qs[i] / T.qs[i + 1]);
  if (!(dq > 0)) return T.ks[i];                  // flat rung: take the bin, do not divide
  return T.ks[i] * Math.pow(T.ks[i + 1] / T.ks[i], Math.log(T.qs[i] / Q) / dq);
}
// ... and the same inversion the OTHER way round: the level a tail sits at at the
// wavenumber k. The chi ordinate is the only caller, and only on an Elsasser lane, where
// db comes from the opposite lane's perpendicular tail sampled at the k_perp this lane's
// own matching just returned. Same log-log interpolation between bracketing bins and the
// same refusal to extrapolate: a k outside the grid returns 0 and the caller drops that
// level rather than fabricating a point.
function _anisoQAt(T, k) {
  if (!(T.n > 0) || !(k > 0) || k < T.ks[0] || k > T.ks[T.n - 1]) return 0;
  let i = 0;
  while (i + 1 < T.n && T.ks[i + 1] <= k) i++;
  if (i + 1 >= T.n) return T.qs[T.n - 1];
  const dk = Math.log(T.ks[i + 1] / T.ks[i]);
  if (!(dk > 0)) return T.qs[i];                  // repeated k: take the bin, do not divide
  return T.qs[i] * Math.pow(T.qs[i + 1] / T.qs[i], Math.log(k / T.ks[i]) / dk);
}
// the band of levels one leg admits: the tail values at the first bin at or above kLo and
// at the last bin below kHi, returned as [Qlo, Qhi] (Q falls as k rises). Levels outside
// it are never generated, which is the same "drop it, do not extrapolate" rule again --
// only applied to the WINDOW rather than to the grid.
function _anisoWin(T, kLo, kHi) {
  let a = -1, b = -1;
  for (let i = 0; i < T.n; i++) {
    if (a < 0 && T.ks[i] >= kLo) a = i;
    if (T.ks[i] < kHi) b = i;
  }
  return (a < 0 || b < a) ? null : [T.qs[b], T.qs[a]];
}
// the largest value on a flat (k, v) curve -- the "peak" specKnee measures its crossing
// against when the curve is alone in its own range pool
function _anisoPeak(pts) {
  let h = 0;
  for (let i = 1; i < pts.length; i += 2) h = Math.max(h, pts[i]);
  return h;
}
// The card's curves, as data -- the specCurves seam again: no canvas, no DOM, so node can
// test the matching directly and a future pin can snapshot it. Returns the same
// [points (k, ratio pairs), colour, dash, label] curve shape, the global (k_par along z)
// curve first when it is drawn (`nGlob` of them, so a caller can tell the two apart, as
// nPerp does on the spectrum), plus the hi/lo of the drawn ratios.
function anisoCurves(d, o) {
  const lane = ANISO_LANES[(o && o.aq)] || ANISO_LANES.tot;
  const ad = (o && o.ad) || "both";
  const ay = (o && o.ay) || "ratio";       // ordinate: the shipped ratio, or chi
  const nb = (d && d.nb) || 1, parKfac = (d && d.parKfac) || 1;
  const kA = fitKA(nb, (d && d.fshell) || [1, 3]);
  const curves = [];
  let hi = 0, lo = Infinity, nGlob = 0;
  const perp = _anisoLeg(d && d.perp, nb, 0, 1, lane);
  // `kd` is Infinity on the degenerate return for the same reason it is anywhere else: no
  // spectrum, no knee -- and drawAniso's fit anchor reads that as "fall back to kA"
  if (perp.length < 4) return { curves, hi, lo, nGlob, kA, kd: Infinity };  // one bin brackets nothing
  const TP = _anisoTail(perp);
  // The high-k end of the window: the dissipation knee, by the SAME peak-then-walk-right
  // crossing the spectrum card's y floor uses (specKnee) -- one knee rule in this file.
  // With no knee in view (an early frame, an instability preset) the window simply runs to
  // the last resolved bin. The low-k end is the fit line's kA, just above the forcing
  // shell. Note the tails themselves are built over ALL bins: the window restricts which
  // LEVELS are reported, not what energy is counted above a k.
  // It is returned as well as used: the fit line's automatic amplitude anchors halfway (in
  // log) between the box wavenumber and this knee, and must take the SAME one the window
  // ended at rather than measure it again.
  const kd = specKnee([[perp]], _anisoPeak(perp));
  const wP = _anisoWin(TP, kA, kd);
  // chi's db, and ONLY db, crosses lanes (the pairing paragraph above): on zp/zm it is the
  // opposite lane's perpendicular tail, sampled at each matched kp. The matching itself is
  // untouched -- this tail is never inverted, only read. A lane whose opposite carries
  // nothing bracketable (a run so imbalanced that E- has no two positive bins) has no
  // shearing field to report, so the honest answer is no curve rather than a chi built out
  // of the wrong lane.
  const opp = ay === "chi" ? ANISO_OPP[(o && o.aq)] : undefined;
  let TPo = null;
  if (opp) {
    const po = _anisoLeg(d && d.perp, nb, 0, 1, ANISO_LANES[opp]);
    if (po.length < 4) return { curves, hi, lo, nGlob, kA, kd };
    TPo = _anisoTail(po);
  }
  // z first, field line second: the fit line anchors on the field-line curve when it is
  // there, and drawAniso finds it at index nGlob.
  const legs = [];
  if (ad !== "fl") legs.push(["z", d && d.par, COL.ek, null, ay === "chi" ? "χ (k∥z)" : "k∥z / k⊥"]);
  if (ad !== "z") legs.push(["fl", d && d.parFL, COL.em, [5, 3], ay === "chi" ? "χ (k∥B)" : "k∥B / k⊥"]);
  for (const lg of legs) {
    const src = lg[1], nzb = Math.floor((src ? src.length : 0) / 3);
    const par = _anisoLeg(src, nzb, 1, parKfac, lane);
    if (par.length < 4) continue;                 // absent or one-binned: no curve at all
    const TQ = _anisoTail(par);
    const wQ = _anisoWin(TQ, 0, specKnee([[par]], _anisoPeak(par)));
    if (!wP || !wQ) continue;
    // the levels live in the OVERLAP of the two admissible bands, log-spaced from the top
    // (large scales, small k) down, so the points come out in k_perp order
    const Qhi = Math.min(wP[1], wQ[1]), Qlo = Math.max(wP[0], wQ[0]);
    if (!(Qhi > 0) || !(Qlo > 0) || Qlo > Qhi) continue;
    const nl = Qlo < Qhi ? ANISO_NLEV : 1;
    const pts = [];
    let cvHi = 0, cvLo = Infinity;
    for (let m = 0; m < nl; m++) {
      const Q = nl > 1 ? Qhi * Math.pow(Qlo / Qhi, m / (nl - 1)) : Qhi;
      const kp = _anisoAt(TP, Q), kz = _anisoAt(TQ, Q);
      if (!(kp > 0) || !(kz > 0)) continue;
      const r = kz / kp;
      if (!(r > 0) || !isFinite(r)) continue;
      let v = r;
      if (ay === "chi") {
        // db^2 = Q, the matched level itself -- or, on an Elsasser lane, the opposite
        // lane's tail at THIS lane's kp. v_A = 1, and kp/kz is a pure ratio, so
        // chi = kp*sqrt(db^2)/kz carries no unit of its own.
        const db2 = TPo ? _anisoQAt(TPo, kp) : Q;
        if (!(db2 > 0)) continue;                 // kp outside the opposite lane's grid
        v = kp * Math.sqrt(db2) / kz;
        if (!(v > 0) || !isFinite(v)) continue;
      }
      pts.push(kp, v);
      cvHi = Math.max(cvHi, v); cvLo = Math.min(cvLo, v);
    }
    // a single surviving point cannot be stroked (specStroke skips < 2 points) and the
    // legend filters it, so letting it into the range pool would blank the card into a
    // bare frame -- with the fit line possibly anchored on the invisible point. Below
    // two points the leg contributes NOTHING, and a card whose every leg is that
    // degenerate keeps its honest "waiting…" (review 2026-08-10, the one MINOR).
    if (pts.length < 4) continue;
    hi = Math.max(hi, cvHi); lo = Math.min(lo, cvLo);
    curves.push([pts, lg[2], lg[3], lg[4]]);
    if (lg[0] === "z") nGlob = 1;
  }
  return { curves, hi, lo, nGlob, kA, kd };
}
function drawAniso(c, d, o) {
  if (!c) return;
  const P = PADS, x0 = P.l, x1 = SW - P.r, y0 = P.t, y1 = SH - P.b;
  chartFrame(c, SW, SH, P);
  c.textAlign = "left"; c.fillStyle = COL.txt;
  const nb = (d && d.nb) || 1;
  const fshell = (d && d.fshell) || [1, 3];
  const A = anisoCurves(d, o);
  const ay = (o && o.ay) || "ratio";
  const fitMode = (o && o.fit) || "pin";
  // "waiting…" covers everything the matching can legitimately fail on: no readback yet,
  // a quiescent field, a 2D-shaped data object, and (the common one) an open card whose
  // field-line spectrum has not landed yet on its own 2 Hz cadence -- no special casing,
  // the global curve simply appears first. On chi it covers one more: an Elsasser lane
  // whose OPPOSITE lane has no bracketable perpendicular tail, i.e. no shearing field.
  if (nb < 2 || !(A.hi > 0)) {
    c.fillText((ay === "chi" ? "χ vs k⊥" : "k∥/k⊥ vs k⊥") + " — waiting…", x0 + 6, y0 + 13);
    return;
  }
  // The chi reference LEVEL is settled before the axes, because it has to be inside them:
  // "is the measured level 1?" is unanswerable when the 1 is off the top of the frame. On
  // the ratio ordinate nothing changes -- there the reference is a SLOPE and the fit line
  // is anchored to the curve itself, so it cannot leave the frame in the first place.
  let chiRef = 0;
  if (ay === "chi" && fitMode !== "off") {
    const Alev = parseFloat(o && o.fita);
    chiRef = (fitMode === "amp" && isFinite(Alev) && Alev > 0) ? Alev : ANISO_CHI_REF;
  }
  const ymax = Math.log10(chiRef > 0 ? Math.max(A.hi, chiRef) : A.hi) + 0.3;
  // at least one decade always, so a flat ratio still has an axis to sit on
  const ymin = Math.min(ymax - 1, Math.log10(chiRef > 0 ? Math.min(A.lo, chiRef) : A.lo) - 0.3);
  const xmax = Math.log10(nb);
  const X = k => x0 + Math.log10(k) / xmax * (x1 - x0);
  const Y = v => px(y1 - (Math.log10(v) - ymin) / (ymax - ymin) * (y1 - y0));

  for (const tk of logTicks(ymin, ymax, (y1 - y0) / (ymax - ymin)))
    yTick(c, Y(tk[0]), x0, x1, tk[2], tk[1]);
  for (const tk of logTicks(0, xmax, (x1 - x0) / Math.max(xmax, 1e-9), (m, e) => String(m * Math.pow(10, e))))
    xTick(c, X(tk[0]), y0, y1, SH - 8, X(tk[0]) > x1 - 20 ? "" : tk[2], tk[1]);
  c.fillStyle = COL.txt; c.textAlign = "center";
  c.fillText(String(nb), x1, SH - 8);
  c.textAlign = "left";
  c.fillText("k⊥ / kunit", x0 + 4, SH - 8);

  c.save();
  c.beginPath(); c.rect(x0, y0, x1 - x0, y1 - y0); c.clip();
  // the forcing-shell markers are kept: they orient the eye on this x axis exactly as
  // they do on the spectrum's, and the window's low end is one of them
  c.strokeStyle = COL.shell; c.setLineDash([2, 3]); c.lineWidth = 1;
  for (const kf of fshell) {
    if (kf >= 1 && kf <= nb) {
      const x = Math.round(X(kf)) + 0.5;
      c.beginPath(); c.moveTo(x, y0); c.lineTo(x, y1); c.stroke();
    }
  }
  c.setLineDash([]);
  // the reference slope, with the spectrum card's controls and behaviours: pin / set A /
  // off, the index snapping to -1/3, -1/2 or -1. It anchors on the FIELD-LINE curve
  // whenever that is drawn -- that is the curve whose slope the lesson is about -- and on
  // the global one otherwise, at the SAME intermediate scale the spectrum card's automatic
  // amplitude uses: halfway logarithmically between the box wavenumber and the dissipation
  // knee anisoCurves already measured for its level window (fitAnchorAuto, kA as fallback).
  // ... and on the CHI ordinate that whole apparatus is beside the point: an index of -1/3
  // is a statement about the ratio, and what critical balance asserts here is a LEVEL. So
  // the reference becomes the horizontal chi = 1 (settled above; the amplitude box renames
  // the level for anyone who wants to compare against a measured one, and `off` still hides
  // it). Same drawn line, same legend slot, index 0.
  const kA = fitKA(nb, fshell);
  const fitP = ay === "chi" ? 0 : fitIndex(o && o.fitp, FIT_FRACS_ANISO);
  let anch = 0;
  if (ay === "chi") anch = chiRef;
  else if (fitMode !== "off") {
    const fc = A.curves.length > A.nGlob ? A.curves[A.nGlob] : (A.curves[0] || null);
    anch = fc ? fitAnchorAuto(fc[0], kA, A.kd, fitP) : 0;
    const Aamp = parseFloat(o && o.fita);
    if (fitMode === "amp" && isFinite(Aamp) && Aamp > 0) anch = Aamp;
  }
  if (anch > 0) {
    c.strokeStyle = COL.guide; c.setLineDash([5, 4]);
    c.beginPath();
    c.moveTo(X(kA), Y(anch * Math.pow(kA, fitP)));
    c.lineTo(X(nb), Y(anch * Math.pow(nb, fitP)));
    c.stroke(); c.setLineDash([]);
  }
  c.lineWidth = 1.4;
  specStroke(c, A.curves, X, Y);
  c.restore();
  const items = A.curves.filter(cv => cv[0].length >= 4).map(cv => [cv[3], cv[1], cv[2]]);
  if (anch > 0) items.push([ay === "chi" ? "χ = " + String(Math.round(anch * 1000) / 1000)
                                         : fitLabel(fitP, FIT_FRACS_ANISO, "k⊥"),
                            COL.guide, [4, 3]]);
  legend(c, x0 + 6, y0 + 12, items, x1 - 30);
}

// ---------------------------------------------------------------------------
// the generated E(k⊥, k∥) card's k⊥ BAND SET (ANISO_PLAN_2 A)
// ---------------------------------------------------------------------------
// One row of that card is the parallel spectrum of the field band-passed to one k⊥ band,
// so this band set IS the card's k⊥ axis. Its two ends are the anisotropy card's level
// window, through the SAME two functions rather than through a second opinion -- the two
// cards must not drift apart on where the resolved inertial range is:
//   low   fitKA(nb, fshell) -- just above the forcing shell, or the second bin when there
//         is no shell worth speaking of, which is where the fit line anchors too;
//   high  the dissipation knee (specKnee's peak-then-walk-right crossing on the
//         perpendicular spectrum), or the dealias cut nb when there is no knee in view --
//         an early frame, an instability preset, or no spectrum handed in at all. It is
//         never above nb: the band is measured in the same bins the ⊥ spectrum is binned
//         into, and there is nothing retained past that cut.
// GEN_NBAND centres are log-spaced across that window and each band is GEN_BAND_W wide in
// k (an octave: ×÷√2 about its centre), so adjacent bands OVERLAP wherever the window is
// shorter than GEN_NBAND octaves -- which at these resolutions it always is. That is
// deliberate and is the standard filtered-snapshot practice: a partition into disjoint
// octaves would give three or four rows at 256², and the rows are not independent
// measurements to begin with -- they are cuts through one E(k⊥, k∥).
// Ends come back in BOX WAVENUMBERS (the unit bandFac's klo/khi are in, k/kunit) and are
// clipped to the window, so no band reaches past the dealias cut or below the shell.
// An empty return means "no resolved range" (a 16² self-test box, a dead field): the
// caller draws nothing rather than one meaningless row.
const GEN_NBAND = 10, GEN_BAND_W = 2;
function gen2dBands(nb, fshell, perp) {
  const n = Math.max(1, nb | 0);
  const kA = fitKA(n, fshell || [1, 3]);
  let kHi = n;
  if (perp) {
    // the knee, off the SAME three-lane bin stack and the same total lane the anisotropy
    // card's window uses (the tail helpers, so a null / one-binned spectrum is a no-op)
    const pts = _anisoLeg(perp, n, 0, 1, ANISO_LANES.tot);
    if (pts.length >= 4) {
      const kd = specKnee([[pts]], _anisoPeak(pts));
      if (isFinite(kd)) kHi = Math.min(kHi, kd);
    }
  }
  const out = [];
  if (!(kHi > kA)) return out;
  const r = Math.sqrt(GEN_BAND_W);
  const c0 = kA * r, c1 = kHi / r;          // centres whose full band still fits the window
  const nbnd = c1 > c0 ? GEN_NBAND : 1;     // a window under one octave carries ONE band
  for (let j = 0; j < nbnd; j++) {
    const kc = nbnd > 1 ? c0 * Math.pow(c1 / c0, j / (nbnd - 1)) : Math.sqrt(kA * kHi);
    out.push({ kc, lo: Math.max(kA, kc / r), hi: Math.min(kHi, kc * r) });
  }
  return out;
}

// ---------------------------------------------------------------------------
// the generated E(k⊥, k∥) card (ANISO_PLAN_2 C): state, panel, choreography, plot
// ---------------------------------------------------------------------------
// The anisotropy card reduces this picture to one curve; this card is the picture. It is
// the only chart here that is NOT fed by the frame loop: its data comes from a GENERATE
// press (Alfred's model, fixed) that pauses the run and sweeps every k⊥ band over ONE
// frozen state -- the page's `gen2dSpec` (ANISO_PLAN_2 A + B) -- and the result then sits
// there until the next press. So the state is a MODULE-LEVEL snapshot, not a per-card one:
//   * a press is a page-wide act (it pauses the run and costs ~1 s of GPU), and the object
//     it produces says nothing about which card asked for it. Two cards can therefore show
//     the field-line panel and the coordinate panel OF THE SAME SNAPSHOT side by side,
//     which is the comparison the card exists for, without generating twice;
//   * it is the shape the energy / island / mode traces already have (`hist`,
//     `islandHist`, `modeHist`): the chart type draws from module state and the cards are
//     views on it.
// `data` is self-contained -- it carries its own t, bands, nb, nzb, parKfac and fshell --
// so the plot survives a pause, a resume, a preset switch, an IC reset and a solver
// rebuild without ever consulting the live grid. That is the pin convention: a record of a
// moment, legended with the t it was taken at, cleared only by taking another one.
const gen2d = { data: null, busy: false, done: 0, total: 0 };
// afmhot, the display cards' own default ramp, through the SAME cmapRGB table the WGSL is
// expanded from -- the heatmap must not invent a second colour convention.
const GEN2D_CMAP = 0;
// One redraw of every card of this type, with the option sync that enables / disables the
// generate button. It is what the progress callback and the end of a sweep call: these
// cards get no frame-loop draw at all.
function gen2dRedraw() {
  for (const c of cards.chart) if (c.type() === "gen2d") { c._optSync(); c.draw(null); }
}
// The button's whole choreography, and the ONLY place run state is touched: pause, sweep,
// keep the result, leave the page paused. `gen2dSpec` deliberately does not touch `running`
// -- pausing is a UI act -- and it is the 3D page that defines it, hence the typeof guard
// (the card is 3D-only, so in 2D this is unreachable rather than merely harmless).
// The run is NOT auto-resumed: setRunning(false) is a visible state, the plot is a static
// picture of the moment it was taken, and the green Run button resuming when the reader is
// ready is exactly what "the plot persists" means.
async function gen2dRun() {
  if (gen2d.busy || typeof gen2dSpec !== "function") return;
  gen2d.busy = true; gen2d.done = 0; gen2d.total = 0;
  setRunning(false);
  gen2dRedraw();                                   // button disabled before the first await
  let d = null;
  try {
    d = await gen2dSpec({ onProgress: (done, total) => {
      gen2d.done = done; gen2d.total = total;
      gen2dRedraw();
    } });
  } catch (e) {
    showStatus("2D spectrum: " + e.message, "err");
    console.error(e);
  }
  gen2d.busy = false;
  // Four ways a press comes back with nothing: no solver at all, the solver retired
  // mid-sweep (a rebuild), no resolved k⊥ range to band (gen2dSpec returns null for those
  // three), and a snapshot of a field with no energy in it (gen2dLive below -- a
  // well-formed object of zeros, which is NOT a null return and which would otherwise
  // replace a good plot with a blank card). In every one of them the PREVIOUS plot is the
  // honest thing to keep showing, and saying so beats a card that silently goes back to
  // "press generate".
  if (d && gen2dLive(d)) gen2d.data = d;
  else showStatus("2D spectrum: nothing to plot — no solver, a rebuild mid-sweep, no "
                  + "resolved k⊥ range yet, or no energy in the field; any previous plot "
                  + "is kept", "info");
  gen2dRedraw();
}
// Is there anything ON a snapshot? A press over a dead or quiescent field returns a
// perfectly well-formed object whose every bin is zero: gen2dBands still cuts a band set
// (with no knee in a silent ⊥ spectrum, its high end falls back to the dealias cut), the
// sweep still runs, and only gen2dPanel notices, by which point the good plot is gone.
// Any positive bin anywhere is enough -- E_u and E_b cannot be negative, so a field with
// energy in it always has one.
function gen2dLive(d) {
  for (const rows of [(d && d.rows) || [], (d && d.crows) || []])
    for (const r of rows) for (let i = 0; i < r.length; i++)
      if (r[i] > 0 && isFinite(r[i])) return true;
  return false;
}
// The panel the card is currently showing, as data (the specCurves seam again: no canvas,
// no DOM, so the plot and the colorbar labels read ONE computation and node can drive it).
// `gp` picks the frame -- field line (default) or coordinate -- and `gq` the lane, through
// the anisotropy card's own ANISO_LANES, so E± = E_u + E_b ± H_c needs no second table.
// Each row comes back as the charts' flat (k, v) pair array, k∥ in the SAME kunit as k⊥
// (parKfac = (2π/Lz)/kunit), which is what lets one axis pair carry both.
function gen2dPanel(o) {
  const d = gen2d.data;
  if (!d) return null;
  const lane = ANISO_LANES[(o && o.gq)] || ANISO_LANES.tot;
  const rows = (((o && o.gp) === "z") ? d.crows : d.rows) || [];
  const nzb = d.nzb | 0;
  if (!rows.length || nzb < 1) return null;
  const out = [];
  let hi = 0, lo = Infinity;
  for (const r of rows) {
    const pts = [];
    for (let i = 0; i < nzb; i++) {
      const v = lane(r[i], r[nzb + i], r[2 * nzb + i]);
      if (v > 0 && isFinite(v)) { pts.push((i + 1) * d.parKfac, v); hi = Math.max(hi, v); lo = Math.min(lo, v); }
    }
    out.push(pts);
  }
  if (!(hi > 0)) return null;
  // the colour floor is the spectrum chart's floor rule, on these rows as its range-setting
  // curves (specFloor: the dissipation knee, the pre-peak minimum, the SPEC_MAXDEC clamp) --
  // one convention for "below this is noise" in this file. A decade of range is the least
  // that reads as a heatmap at all.
  const floor = Math.min(specFloor(out.map(p => [p]), hi, lo), 0.1 * hi);
  return { d, rows: out, hi, floor, nzb };
}
// The cell colour, as ONE function of a value -- what the `gc` select picks:
//   log  (the default) the rule this card was born with: the panel's own specFloor to the
//        peak, which is this file's single "below this is noise" convention and the only
//        mapping on which an inertial range is visible at all;
//   lin  0 to the peak. It shows almost nothing -- a spectrum falls over decades -- and is
//        exactly why it is worth being able to look: it is the honest picture of how much
//        of the energy sits at the outer scale.
// Returned unclamped; the caller clamps to [0,1] as it always did.
function gen2dCscale(G, o) {
  const lf = Math.log10(G.floor), lh = Math.log10(G.hi);
  return ((o && o.gc) === "lin")
    ? v => v / Math.max(G.hi, 1e-300)
    : v => (Math.log10(v) - lf) / Math.max(lh - lf, 1e-9);
}
// the colorbar's three labels, off the SAME select, because the strip is painted with the
// same ramp: the middle of a LOG strip is the geometric mean of its ends, the middle of a
// LINEAR one the arithmetic mean, and the low end of a linear one is zero, not the floor
function gen2dBarTicks(o) {
  const G = gen2dPanel(o);
  if (!G) return ["", "", ""];
  if ((o && o.gc) === "lin") return [cbarFmt(0), cbarFmt(0.5 * G.hi), cbarFmt(G.hi)];
  return [cbarFmt(G.floor), cbarFmt(Math.sqrt(G.floor * G.hi)), cbarFmt(G.hi)];
}
// the theory slopes, behind one checkbox and OFF by default: THREE reference slopes --
// GS95 (2/3), Boldyrev 2006 (1/2) and isotropic (1) -- drawn as straight lines on this
// log-log frame, each in its own dash and its own grey/white/green so they are told apart
// where they cross. They are the ONLY thing the checkbox draws (Alfred, second round: the
// two measured anisotropy-card curves that used to ride along with them were dropped --
// "don't think they are helpful" -- and with them the measured-vs-prediction distinction
// this card's legend and manual entry used to have to make).
// (The per-band argmax RIDGE used to live here. Nothing drawn ever called it -- the
// reference lines anchor on gen2dTop, below -- so its only consumer was check2dspec's
// recovery leg, which now owns it as `ridgeOf`: render audit, 2026-08-12.)
// The upper BOUNDARY of the excited region in one column: the largest k∥ whose cell is
// still meaningfully above the panel's floor (0 for a column with nothing in it). This, not
// the argmax ridge, is what a critical-balance slope is a statement about -- the cascade
// FILLS k∥ ≲ k∥(k⊥), so the relation is the edge of the filled region and the eye should be
// comparing the slope with that edge (Alfred, first round).
// Measured against `G.floor` -- the panel's own noise floor -- and NOT against the colour
// mapping, so flipping the colour scale does not move the lines.
//
// GEN2D_TOPMARGIN is a NOISE margin, and it is not cosmetic (review of the first round).
// The test used to be `>= floor` exactly, and the boundary is a MAX over bins: one cell
// sitting at 1.05x the floor at the top of one column therefore moves that column's
// boundary to the top of the plot, and because all three anchors are a max over bands
// (below) it drags every reference line with it -- 0.71 decades, 47% of the frame, on the
// reviewer's demonstration. A cell at 1.05x the floor is exactly the fp32 periodogram fuzz
// this card's own manual entry warns about, so that is not a hypothetical: it fires on real
// snapshots and wrecks the one comparison the overlay exists for. Requiring the cell to be
// at least 2x the floor -- half of the panel's LOWEST decade of colour, still far below
// anything the eye reads as filled -- costs nothing on a real edge, where the spectrum is
// falling steeply and the neighbouring bins are orders of magnitude apart, and takes the
// single-fuzzy-cell excursion out. Alfred can veto it on-device; it is one number.
const GEN2D_TOPMARGIN = 2;
function gen2dTop(pts, floor) {
  let k = 0;
  const thr = GEN2D_TOPMARGIN * floor;
  for (let i = 0; i < pts.length; i += 2) if (pts[i + 1] >= thr && pts[i] > k) k = pts[i];
  return k;
}
// [exponent, label, colour, dash]. Labels are Alfred's abbreviations, kept exactly, because
// all three ride on ONE legend line (second round) and the long forms -- "k∥ ∝ k⊥^2/3
// (GS95)" and friends -- took a line each. The long forms live in the checkbox's tooltip
// and in docs.html, where there is room for them. Theory palette: greys and a green, all
// dashed, kept clear of the anisotropy card's own blue / orange.
const GEN2D_SLOPES = [
  [2 / 3, "GS95", COL.guide, [5, 4]],
  [1 / 2, "B06", COL.cut, [2, 3]],
  [1, "iso", COL.et, [8, 3, 2, 3]]
];
// The anchor of one reference slope: A = max over bands of k∥_top(band) / k_c^exponent,
// so the line A·k⊥^p sits JUST ABOVE the measured envelope -- touching it at the band where
// the envelope presses hardest against that slope, and above it everywhere else. The same
// rule for all three, which is what makes them comparable with each other and with the
// boundary. Only the SLOPE is ever a claim here: the intercept is chosen by this rule and
// says nothing about the plasma. 0 when no column has anything in it -- then no line.
//
// "k∥_top" here is the top EDGE of the top filled cell, kt + half a bin, not its centre
// (review of the first round). A cell is drawn as a rectangle half a bin either side of its
// own k∥ (drawGen2D), so anchoring on the centre put the line through the middle of the
// filled pixels at the touching band -- visibly resting ON the data rather than on top of
// it, which is not what "just above the boundary" says.
function gen2dAnchor(G, p) {
  let A = 0;
  for (let j = 0; j < G.rows.length && j < G.d.bands.length; j++) {
    const kt = gen2dTop(G.rows[j], G.floor);
    if (kt > 0) A = Math.max(A, (kt + 0.5 * G.d.parKfac) / Math.pow(G.d.bands[j].kc, p));
  }
  return A;
}
// The plot's FRAME, as data -- the specCurves seam once more (no canvas, no DOM), so the
// axes are one computation that node can read instead of a set of numbers buried in the
// draw. Both ends of both axes are the drawn cells' own edges, and nothing else:
//   x  The extent of a band's COLUMN is the MIDPOINT (in log k⊥) between its neighbours'
//      centres, not its own [lo, hi]: the bands deliberately overlap (gen2dBands), so their
//      own ends would paint over each other and the picture would depend on the draw order.
//      A single-band set falls back to that one band's ends, which is all there is. The
//      axis then runs from the first column's left edge to the last one's right edge
//      (FEEDBACK item 1). It used to run 1 .. nb, and the band set starts at fitKA (3 for
//      the default shell) with the first CENTRE a half-octave above that -- so a fifth of
//      the canvas was axis no band had ever been cut in. Nothing is lost with it: there is
//      no measurement below the first edge to show.
//   y  |k∥| over its own bins, 1..nzb, in the same kunit as k⊥ -- so a k⊥^(2/3) line is a
//      statement and not a unit conversion. The floor is bin 1 ITSELF, not half a bin below
//      it (FEEDBACK item 2), because bin 1 is the injection scale: the 3D forcing's z
//      envelope is A·cos(2πz/Lz) + B·sin(2πz/Lz) (rmhd3d's `envFn`), whose z-FFT is nonzero
//      on kz index ±1 and exactly zero on every other plane. So the axis starts at the
//      forced fundamental -- which is k∥ = 1 only because every preset here fixes Lz = 2π;
//      on an 8π box the same fundamental is 0.25, and pinning the floor at 1 there would
//      CLIP three resolved bins. Nothing resolved is clipped this way: what hangs below the
//      frame is the bottom half of the lowest row of cells, and the frame clip takes it,
//      which is right -- there is no energy below the fundamental to draw.
//
// HEADROOM (Alfred, second round). The legend block sits at the top left of the frame, and
// the top left of the frame is exactly where the low-k⊥ / high-k∥ corner of the data is, so
// the legend was reading over the cells. `hleg` (the legend's own measured height in pixels
// -- see legendLines) and `hplot` (the frame's height in pixels) raise `yhi` ABOVE the top
// of the resolved range by just enough that the whole legend block sits over empty canvas:
// the resolved top (nzb + 1/2) is placed hleg pixels below the top of the frame, so
//   (ytop - ylo) / (yhi - ylo) = (hplot - hleg) / hplot.
// This is a change of AXIS LIMIT and nothing else. The ticks and their labels run on across
// the added range (logTicks is handed the new span), every cell keeps the same k∥, and no
// cell moves relative to the axis -- the whole plot is simply drawn at a slightly smaller
// pixels-per-decade. The reserve is deliberately measured against the top of the RESOLVED
// range rather than against the topmost non-empty cell: the empty space is then the same
// from one press to the next instead of breathing with the data, and since no cell can be
// above the resolved top it clears every one of them. Clamped at a third of the frame, so a
// pathologically tall legend cannot squash the plot into nothing; with hleg = 0 (the
// default) this is exactly the old frame.
function gen2dFrame(G, hleg, hplot) {
  const d = G.d, lb = d.bands.map(b => Math.log10(b.kc)), pk = d.parKfac;
  const edge = j => {
    if (lb.length < 2) return Math.log10(j === 0 ? d.bands[0].lo : d.bands[0].hi);
    if (j === 0) return lb[0] - 0.5 * (lb[1] - lb[0]);
    if (j >= lb.length) return lb[lb.length - 1] + 0.5 * (lb[lb.length - 1] - lb[lb.length - 2]);
    return 0.5 * (lb[j - 1] + lb[j]);
  };
  const ylo = Math.log10(pk), ytop = Math.log10((G.nzb + 0.5) * pk);
  const H = hplot > 0 ? hplot : 0;
  const L = H > 0 ? Math.max(0, Math.min(hleg || 0, H / 3)) : 0;
  const yhi = L > 0 ? ylo + (ytop - ylo) * H / (H - L) : ytop;
  return { lb, edge, xlo: edge(0), xhi: edge(lb.length), ylo, ytop, yhi };
}
// The card's LEGEND and the FRAME that leaves room for it, as one computation -- the
// specCurves seam again, so the draw and any reader (node) get the same answer instead of
// two opinions. The legend has to be settled before the axes are, because the y axis is
// stretched by the block's own height (gen2dFrame's headroom) and that height is not known
// until its wrapping is: it depends on the labels, on measureText and on the card's width.
// The list is
//   * the header, which says which panel and which moment (the pin convention);
//   * ONE grouped entry carrying all three theory slopes, each with its own dash and
//     colour swatch (Alfred, second round) -- present only for the slopes that actually
//     have an anchor, and only with the checkbox on;
//   * the sweep's progress line while a press is in flight.
// The height is MEASURED on a list that always carries the theory group, ticked or not.
// The alternative -- measuring what is actually drawn -- would rescale the whole heatmap
// the moment the reader ticks the box, which is exactly when they are trying to compare a
// slope with the boundary. The cost is a line of empty axis with the box off, which is
// honest axis and costs nothing to read.
function gen2dLayout(c, G, o, prog) {
  const P = PADS, x1 = GSW - P.r, y0 = P.t, y1 = GSH - P.b, d = G.d;
  const lgx = P.l + 6, lgxm = x1 - 30;
  const head = [((o && o.gp) === "z" ? "coordinate k∥z" : "field line k∥B") +
                " — generated @ t = " + (isFinite(d.t) ? d.t.toFixed(2) : "?"), COL.txt];
  const refs = [];
  if ((o && o.gov) === "on")
    for (const s of GEN2D_SLOPES) {
      const A = gen2dAnchor(G, s[0]);
      if (A > 0) refs.push([s[0], A, s[2], s[3], s[1]]);      // p, A, colour, dash, label
    }
  const items = [head], meas = [head, [GEN2D_SLOPES.map(s => [s[1], s[2], s[3]])]];
  if (refs.length) items.push([refs.map(r => [r[4], r[2], r[3]])]);
  if (prog) { items.push([prog, COL.txt]); meas.push([prog, COL.txt]); }
  const hleg = 12 * legendLines(c, lgx, meas, lgxm) + 4;   // baselines 12 apart + descenders
  return { items, refs, lgx, lgxm, hleg, F: gen2dFrame(G, hleg, y1 - y0) };
}
function drawGen2D(c, o) {
  if (!c) return;
  const P = PADS, x0 = P.l, x1 = GSW - P.r, y0 = P.t, y1 = GSH - P.b;
  chartFrame(c, GSW, GSH, P);
  c.textAlign = "left"; c.fillStyle = COL.txt;
  const prog = gen2d.busy
    ? "generating… band " + gen2d.done + "/" + Math.max(gen2d.total, gen2d.done, 1) : "";
  const G = gen2dPanel(o);
  if (!G) { c.fillText(prog || "E(k⊥, k∥) — press generate", x0 + 6, y0 + 13); return; }
  const d = G.d, pk = d.parKfac;
  const LY = gen2dLayout(c, G, o, prog);
  const items = LY.items, refs = LY.refs, lgx = LY.lgx, lgxm = LY.lgxm;
  const F = LY.F, lb = F.lb, edge = F.edge;
  const xlo = F.xlo, xhi = F.xhi, xw = Math.max(xhi - xlo, 1e-9);
  const XL = L => x0 + (L - xlo) / xw * (x1 - x0);   // a log10 k⊥ -> pixels
  const X = k => XL(Math.log10(k));
  const ylo = F.ylo, yhi = F.yhi;
  const YL = L => px(y1 - (L - ylo) / (yhi - ylo) * (y1 - y0));
  const Y = k => YL(Math.log10(k));

  const kfmt = (m, e) => String(Math.round(m * Math.pow(10, e) * 1e6) / 1e6);
  for (const tk of logTicks(ylo, yhi, (y1 - y0) / Math.max(yhi - ylo, 1e-9), kfmt))
    yTick(c, YL(Math.log10(tk[0])), x0, x1, tk[2], tk[1]);
  // the axis is named the way the spectrum and anisotropy charts name theirs -- x on the
  // tick line, y in the hint, no rotated text anywhere in this file. It also OWNS the left
  // of that line exactly as the end-of-axis label owns the right: a tick label whose
  // centred box would run into either stays unlabelled (FEEDBACK item 6). The old label
  // carried "(y: k∥ / kunit)" too, 26 characters at ~6 px each, which reached past the
  // half-decade mark and sat under two of them.
  const xlab = "k⊥ / kunit", xlabR = x0 + 4 + c.measureText(xlab).width + 4;
  const kend = v => String(Math.round(v * (v < 10 ? 10 : 1)) / (v < 10 ? 10 : 1));
  for (const tk of logTicks(xlo, xhi, (x1 - x0) / xw, (m, e) => String(m * Math.pow(10, e)))) {
    const xp = X(tk[0]);
    const clear = xp - 0.5 * c.measureText(tk[2]).width > xlabR && xp <= x1 - 20;
    xTick(c, xp, y0, y1, GSH - 8, clear ? tk[2] : "", tk[1]);
  }
  c.fillStyle = COL.txt; c.textAlign = "center";
  c.fillText(kend(Math.pow(10, xhi)), x1, GSH - 8);
  c.textAlign = "left";
  c.fillText(xlab, x0 + 4, GSH - 8);

  const cs = gen2dCscale(G, o);
  c.save();
  c.beginPath(); c.rect(x0, y0, x1 - x0, y1 - y0); c.clip();
  for (let j = 0; j < G.rows.length && j < lb.length; j++) {
    const xa = Math.round(XL(edge(j))), xb = Math.round(XL(edge(j + 1)));
    const pts = G.rows[j];
    for (let i = 0; i < pts.length; i += 2) {
      const kp = pts[i];
      const t = Math.max(0, Math.min(1, cs(pts[i + 1])));
      const rgb = cmapRGB(GEN2D_CMAP, t);
      c.fillStyle = "rgb(" + rgb.map(v => Math.round(255 * v)).join(",") + ")";
      // one cell = one (band, |k∥| bin): half a bin either side of its own k∥, which is
      // exactly the bin's width and needs no bin index carried alongside the value
      const ya = Math.round(Y(kp + 0.5 * pk)), yb = Math.round(Y(kp - 0.5 * pk));
      c.fillRect(xa, ya, Math.max(1, xb - xa), Math.max(1, yb - ya));
    }
  }
  // The forcing-shell markers, the dashes every other k⊥ chart carries. They are kept ONLY
  // where they still fall inside the axis -- and now that the axis starts at the first
  // band's own left edge (which is above fitKA, which is above the shell) they generally
  // will not, so they simply do not draw. Deliberately not clamped to the frame: a marker
  // pinned to the left edge would claim the shell is at the start of the measured range,
  // which is the one thing it is not.
  c.strokeStyle = COL.shell; c.setLineDash([2, 3]); c.lineWidth = 1;
  for (const kf of (d.fshell || [])) {
    const lk = Math.log10(kf);
    if (kf > 0 && lk >= xlo && lk <= xhi) {
      const x = Math.round(X(kf)) + 0.5;
      c.beginPath(); c.moveTo(x, y0); c.lineTo(x, y1); c.stroke();
    }
  }
  c.setLineDash([]);
  // The three reference slopes, each anchored so it lies JUST ABOVE the upper boundary of
  // the filled region (gen2dAnchor / gen2dTop). Same rule for all three, so their slopes
  // can be read against each other and against the boundary; the intercept is a consequence
  // of the rule and is never a claim. `refs` was settled with the legend above, so a slope
  // with no anchor (no non-empty column anywhere) is neither drawn nor legended.
  const ka = Math.pow(10, xlo), kb = Math.pow(10, xhi);
  c.lineWidth = 1.4;
  for (const [p, A, col, dash] of refs) {
    c.strokeStyle = col; c.setLineDash(dash);
    c.beginPath();
    c.moveTo(X(ka), Y(A * Math.pow(ka, p)));
    c.lineTo(X(kb), Y(A * Math.pow(kb, p)));
    c.stroke(); c.setLineDash([]);
  }
  c.restore();
  // the cells overpaint chartFrame's border, so it is re-stroked over them
  c.strokeStyle = COL.axis; c.lineWidth = 1;
  c.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
  // ... and the block the axis made room for, drawn exactly where it was measured
  legend(c, lgx, y0 + 12, items, lgxm);
}

// ---------------------------------------------------------------------------
// cut trace: a PAIR of in-plane components along y at fixed x = Lx/2 (REFINE_PLAN
// H.3). `d.vals` is solver.readCutLine's 4*ny stack (u_x, u_y, b_x, b_y) of the
// card's own z plane, so the pair selector is pure arithmetic here -- including
// z+- = u +- b, whose magnitudes need no extra readback. Signed pairs get a
// symmetric +-max axis, the magnitude pair [0, max].
// ---------------------------------------------------------------------------
const CUT_PAIRS = {
  u: { t: ["u_x", "u_y"], c: [COL.ek, COL.em], signed: true,
       f: (v, n, j) => [v[j], v[n + j]] },
  b: { t: ["b_x", "b_y"], c: [COL.ek, COL.em], signed: true,
       f: (v, n, j) => [v[2 * n + j], v[3 * n + j]] },
  z: { t: ["|z+|", "|z-|"], c: [COL.zp, COL.zm], signed: false,
       f: (v, n, j) => [Math.hypot(v[j] + v[2 * n + j], v[n + j] + v[3 * n + j]),
                        Math.hypot(v[j] - v[2 * n + j], v[n + j] - v[3 * n + j])] }
};
function drawCut(c, d, o) {
  if (!c) return;
  const P = PADC, x0 = P.l, x1 = CW - P.r, y0 = P.t, y1 = CH - P.b;
  chartFrame(c, CW, CH, P);
  c.textAlign = "left"; c.fillStyle = COL.txt;
  const pair = CUT_PAIRS[(o && o.pair)] || CUT_PAIRS.u;
  const Ly = (d && d.Ly) || 1, signed = pair.signed;
  const n = d && d.vals ? (d.vals.length >> 2) : 0;
  const lines = [[], []];
  let mx = 0;
  for (let j = 0; j < n; j++) {
    const p = pair.f(d.vals, n, j);
    for (let s = 0; s < 2; s++) {
      lines[s].push(p[s]);
      const a = Math.abs(p[s]); if (isFinite(a) && a > mx) mx = a;
    }
  }
  if (n < 2 || !(mx > 0)) { c.fillText("cut along y at x = Lx/2 — waiting…", x0 + 6, y0 + 13); return; }
  const vlo = signed ? -mx : 0, vhi = mx;
  const X = j => x0 + (j / n) * (x1 - x0);
  const Y = v => px(y1 - (v - vlo) / (vhi - vlo) * (y1 - y0));

  // y ticks: the extremes, the midpoints and (for signed pairs) the zero line
  for (const tk of linTicks(signed ? 8 : 4)) {
    const v = vlo + (vhi - vlo) * tk[0];
    yTick(c, Y(v), x0, x1, tk[1] ? (v === 0 ? "0" : v.toExponential(1)) : "", tk[1]);
  }
  // x ticks: quarters of the box, halved again as minors
  for (const tk of linTicks(8))
    xTick(c, x0 + tk[0] * (x1 - x0), y0, y1, CH - 6, tk[1] ? (tk[0] * Ly).toFixed(2) : "", tk[1]);
  c.fillStyle = COL.txt; c.textAlign = "left";
  c.fillText("y", x0 + 4, CH - 6);

  c.save();
  c.beginPath(); c.rect(x0, y0, x1 - x0, y1 - y0); c.clip();
  c.lineWidth = 1;
  for (let s = 0; s < 2; s++) {
    c.strokeStyle = pair.c[s]; c.beginPath();
    for (let j = 0; j < n; j++) {
      const x = X(j), y = Y(lines[s][j]);
      if (j === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.stroke();
  }
  c.restore();
  legend(c, x0 + 6, y0 + 12, [[pair.t[0], pair.c[0]], [pair.t[1], pair.c[1]],
                              ["@ x = Lx/2", COL.cut]], x1 - 30);
}
// ===========================================================================
// cards: ONE display-card class, ONE chart-card interface
// ===========================================================================
// A display card owns one WebGPU canvas + its arrow overlay + the selectors that
// drive it (quantity, per-card colormap, arrows on/off and -- in 3D -- its own z
// slice / peak tracker). Card index `ci` IS the solver's display-chain index, so
// "N cards" costs exactly N chains and nothing else: there is no dual-view flag,
// no second canvas context, no chain-0 special case anywhere. EVERY display card is
// closable down to the last one (the IC editor has its own view since Phase G, so
// nothing is anchored to card 0 any more).
//
// A chart card owns one 2D canvas and a TYPE (energy / spectrum / cut) taken from
// CHART_TYPES below; the frame loop asks each type for its data once per throttle
// window and hands the result to every card of that type.
//
// The app supplies the parts that are genuinely per-app through cardsInit(cfg):
//   fields      [{v, t}] the quantity <option> list
//   zslice      true in 3D: build the per-card z-source select + slice slider
//   cube        true in 3D: that select also offers the cube-faces and field-lines
//               VIEWS (I2.1, K2.1) -- both meaningless on the cut card, hence one flag
//   nz()        current nz, for the slider range
//   zsliceOf(c) resolved plane index of card c (slider or tracked peak)
//   arrowXform() 3D only: the cube top face's (u,v) -> canvas affine, for the arrows
//   lineXform() 3D only: the whole box's (x,y,z) -> canvas affine, for the field lines
//   caption(c)  optional text appended to the card's caption
//   onLayout()  called after any add/remove/close, for app-side label syncing
// Everything else -- DOM, wiring, render order, throttles -- is shared.
const CARD_MAX_DISP = 3;
const CARD_MIN_DISP = 1;
// Each chart type declares its own per-card OPTION selects (REFINE_PLAN H.1-H.3):
// `opts(cfg)` returns [{ id, ti, o: [[value, html], ...], v }], the card builds one
// <select> per entry into its header, and `draw` gets the current values as an object.
// Defaults are the FIRST option of each list and reproduce the pre-Phase-H chart
// exactly. `zslice: true` additionally gives the card its own z source in 3D.
const CHART_TYPES = {
  energy: {
    label: "energy trace", w: EW, h: EH,
    opts: () => [{ id: "emode", ti: "which energies to trace",
                   o: [["kmt", "E_u, E_b"], ["pmt", "E&#8314;, E&#8315;"]] }],
    draw: (c, d, o) => drawEnergy(c, o),
    hint: "vs t; E<sup>&plusmn;</sup> = E<sub>u</sub> + "
      + "E<sub>b</sub> &plusmn; H<sub>c</sub>, so E<sub>tot</sub> = (E<sup>+</sup>+E<sup>&minus;</sup>)/2"
  },
  spectrum: {
    label: "spectra", w: SW, h: SH,
    opts: cfg => [{ id: "sq", ti: "which spectra to bin",
                    o: [["ub", "E_u, E_b"], ["pm", "E&#8314;, E&#8315;"], ["both", "both"]] }]
      .concat(cfg.zslice
        ? [{ id: "sd", ti: "perpendicular (solid) / parallel (dashed) spectra; "
               + "\"field line\" measures k_par ALONG B, not along z",
             o: [["both", "&perp; + &#8741;"], ["perp", "&perp; only"], ["par", "&#8741; only"],
                 ["fl", "&perp; + k&#8741; (field line)"]] }]
        : [],
        // the fit line (item 8): mode, index, amplitude. The two boxes are meaningless
        // with the line off, so they hide with it.
        [{ id: "fit", ti: "power-law fit line E = A k^p",
           o: [["pin", "fit: pin to field"], ["amp", "fit: set A"], ["off", "fit: off"]] },
         { id: "fitp", k: "num", w: 62, step: "any", v: -1.667,
           ti: "fit-line spectral index p in E = A k^p (-5/3 default; -1.5 reads as -3/2)",
           vis: v => v.fit !== "off" },
         { id: "fita", k: "num", w: 74, step: "any", min: 0,
           ti: "fit-line amplitude A at k = 1 in E = A k^p; blank pins it to the spectrum",
           vis: v => v.fit === "amp" },
        // the pins (PINCURVE Phase B). Per card, like the fit line: two cards can carry
        // different frozen states over the same run. `pin` is dead until the card has
        // been handed data; `unpin` only exists while there is something to clear.
        // the y floor (2026-08-13). ON is the historical rule: specFloor's dissipation
        // knee, which is calibrated on a hyper-dissipative tail and deliberately clips it.
        // OFF shows every drawn bin (still under the SPEC_MAXDEC clamp) and is what the
        // hyper = 1 presets ship with -- see drawSpectrum, where the reasoning lives.
         { id: "clip", k: "cbl", t: "clip tail", v: "on",
           ti: "cut the y axis off below the dissipation knee, so an inertial range is not "
             + "squashed by a hyper-dissipative tail 15 decades down. Untick to see every "
             + "bin drawn -- which is what a bump at the small-scale end is read off, and "
             + "what the hyper = 1 (tearing) presets open with" },
         { id: "pin", k: "btn", t: "pin", onClick: c => c.pinAdd(),
           dis: (v, c) => !c.lastData,
           ti: "freeze the curves this card is drawing as grey ghosts, to compare what "
             + "comes next against (at most " + PIN_MAX + ")" },
         { id: "unpin", k: "btn", t: "unpin", onClick: c => c.pinClear(),
           vis: (v, c) => c.pins.length > 0,
           ti: "clear every pinned ghost on this card" }]),
    draw: (c, d, o, pins) => drawSpectrum(c, d, o, pins),
    hint: "shell-binned; E<sup>&plusmn;</sup>(k) = E<sub>u</sub>+E<sub>b</sub>&plusmn;H<sub>c</sub>. "
      + "<b>pin</b> freezes the current curves as ghosts for before/after comparison."
  },
  cut: {
    label: "cut trace", w: CW, h: CH, zslice: true,
    opts: () => [{ id: "pair", ti: "which pair of components to trace",
                   o: [["u", "u_x, u_y"], ["b", "b_x, b_y"], ["z", "|z&#8314;|, |z&#8315;|"]] }],
    draw: (c, d, o) => drawCut(c, d, o),
    hint: "cut along y at x = L<sub>x</sub>/2"
  },
  // REFINE_PLAN J.4. `src: "cut"` says it feeds off the cut readback -- the X and O points
  // live on the resonant surface x = Lx/2, which is the line cutPrep already prepares, so
  // this card adds no kernel, no buffer and no round trip. 2D only: the equilibria are.
  island: {
    label: "island width", w: CW, h: EH, src: "cut", avail: cfg => !cfg.zslice,
    draw: c => drawIsland(c),
    hint: "W = 4&radic;(&Delta;&psi;/2|&psi;&Prime;|) from the &psi; extrema on x = L<sub>x</sub>/2, "
      + "with &psi;&Prime; measured on the equilibrium. the linear stage is a straight line, with "
      + "the local growth rate shown in the legend."
  },
  // the KH counterpart of the island trace, and on the SAME readback (`src: "cut"`): the
  // k_y = 2pi/Ly Fourier amplitude of u_x / b_x on x = Lx/2. Also 2D only -- the
  // equilibria are.
  mode: {
    label: "k_y = 2&pi;/L_y mode", w: CW, h: EH, src: "cut", avail: cfg => !cfg.zslice,
    draw: c => drawMode(c),
    hint: "|&#251;<sub>x</sub>| (and |b&#770;<sub>x</sub>|) at k<sub>y</sub> = 2&pi;/L<sub>y</sub> "
      + "on x = L<sub>x</sub>/2 &mdash; no equilibrium content at all, unlike the energy; log y, "
      + "so the KH linear stage is a straight line and the legend fits its &gamma;. That line lies "
      + "midway between the layers, where the mode is evanescent (~e<sup>&minus;k<sub>y</sub>L<sub>x</sub>/4</sup> "
      + "= e<sup>&minus;&pi;</sup> per layer), so the amplitude is well below its on-layer value "
      + "&mdash; an offset of the line, not a change of its slope."
  },
  // the critical-balance card (ANISO_PLAN). `src: "spectrum"` says it feeds off the
  // spectrum readback -- the two 1D spectra it matches are already in that data object,
  // so this card adds no kernel, no buffer and no round trip, exactly as island/mode add
  // none to the cut line. 3D only: in 2D there is no parallel direction to measure.
  // The hint is Alfred's own copy (his third feedback round), no longer a draft. What it no
  // longer carries -- the Cho-Vishniac divergence of the two curves, the L_z gauge caveat,
  // the integer-k_z flattening at low k_perp and the field-line sampler's own cadence --
  // lives in docs.html under #aniso, in full sentences: the hint is the one-breath version
  // and the manual is the long one. "Solid" / "Dashed" are the dashes anisoCurves gives the
  // two legs (the z leg null, the field-line leg [5, 3]), not a second convention.
  // CHI_PLAN put a second ORDINATE on this card rather than a second card: chi differs from
  // the shipped curve by one multiplication and a label, and a card of its own would
  // duplicate the window, the lane select, the level loop and the legend -- and cost a card
  // slot on a phone. `ay` therefore leads the option row, where it reads as the card's
  // subtitle beside the type name (which is why that name is no longer the ratio's own).
  // Its first value is the shipped one, per the house rule; the hint follows the select.
  aniso: {
    label: "anisotropy", w: SW, h: SH, src: "spectrum",
    avail: cfg => cfg.zslice,
    opts: () => [
      { id: "ay", ti: "what is on the y axis: the anisotropy ratio itself, or the "
          + "critical-balance parameter χ = k⊥δb/(k∥ v_A) built from the same matching "
          + "(δb² = the matched energy content above k⊥, v_A = 1). χ is the combination the "
          + "ratio's L_z convention cancels out of",
        o: [["ratio", "k&#8741;/k&perp;"], ["chi", "&chi; = k&perp;&delta;b/k&#8741;"]] },
      { id: "aq", ti: "which energy the matched spectra are built from; "
          + "E&plusmn; = E_u + E_b &plusmn; H_c",
        o: [["tot", "E_u+E_b"], ["zp", "E&#8314;"], ["zm", "E&#8315;"]] },
      { id: "ad", ti: "k&#8741; measured along the coordinate z axis (solid), along the "
          + "actual field lines (dashed), or both",
        o: [["both", "both k&#8741;"], ["z", "along z only"], ["fl", "field line only"]] },
      { id: "fit", ti: "power-law reference line k&#8741;/k&perp; = A k&perp;^p",
        o: [["pin", "fit: pin to curve"], ["amp", "fit: set A"], ["off", "fit: off"]] },
      // the index box is hidden on the chi ordinate, where a power law is not what the
      // reference line is: there it is the horizontal chi = 1 (ANISO_CHI_REF), and the
      // amplitude box below renames that level for anyone comparing against a measured one
      { id: "fitp", k: "num", w: 62, step: "any", v: -0.333,
        ti: "reference index p in k&#8741;/k&perp; = A k&perp;^p (-1/3 critical balance, "
          + "-1/2 aligned, -1 the wandering-z limit)",
        vis: v => v.fit !== "off" && (v.ay || "ratio") !== "chi" },
      { id: "fita", k: "num", w: 74, step: "any", min: 0,
        ti: "reference amplitude A at k&perp; = 1; blank pins it to the curve "
          + "(on the χ axis: the level of the horizontal reference line, 1 when blank)",
        vis: v => v.fit === "amp" }
    ],
    draw: (c, d, o) => drawAniso(c, d, o),
    // Alfred's copy on the ratio ordinate, verbatim and unchanged. The chi branch is the
    // same shape of sentence for the other axis, and carries the two things the ordinate
    // cannot say for itself: the db convention, spelled out, and the fact that the level is
    // an ORDER OF MAGNITUDE (never two significant figures -- see the block above
    // anisoCurves). It points at the departures because a flat line over half a decade is
    // not the content, and it says outright that the solid/dashed split is the ratio card's
    // divergence replotted rather than a second measurement.
    hint: v => ((v && v.ay) === "chi"
      ? "&chi; = k&perp;&delta;b/(k&#8741;v<sub>A</sub>), from the same cumulative-energy "
        + "matching, with <b>&delta;b&sup2; = Q</b>, the matched energy content above "
        + "k&perp;, and v<sub>A</sub> = 1. Unlike k&#8741;/k&perp; its level does not move "
        + "with L<sub>z</sub>: critical balance says &chi; is of order 1 across the inertial "
        + "range, and &delta;b carries the usual O(1) convention, so read the level as an "
        + "order of magnitude and the flatness as the claim. The departures are the "
        + "interesting part &mdash; &chi; above 1 towards the forcing shell (the outer scale "
        + "is driven, not balanced), the roll-off at the dissipation knee, and E&#8314; "
        + "parting from E&#8315; when the run is imbalanced, each taking &delta;b from the "
        + "<i>other</i> Elsasser field, which is what shears it. Solid and dashed part "
        + "company for the same reason as on the ratio axis and by exactly the same factor "
        + "&mdash; one &delta;b divides both &mdash; so that split is that lesson replotted, "
        + "not a second measurement. experimental feature: imperfect agreement at these "
        + "resolutions."
      : "k&#8741;(k&perp;)/k&perp; as a function of k&perp; by matching cumulative energy "
        + "above k in the perpendicular and parallel spectra. Solid: k<sub>z</sub> (global "
        + "mean field). Dashed: k&#8741; (local mean field along field lines). Eddies "
        + "elongate along B as they shrink, so the ratio falls; somewhere between &minus;1/2 "
        + "and &minus;1/3 is the classic critical balance prediction. experimental feature: "
        + "imperfect agreement at these resolutions.")
  },
  // ... and the distribution that curve is EXTRACTED from (ANISO_PLAN_2 C). The only chart
  // type with no readback source at all: its data is a GENERATE press, so it declares no
  // `src`, the frame loop never hands it anything, and every draw reads the module-level
  // snapshot above. 3D only and placeholderless in 2D (no k∥ to bin), by the same `avail`
  // the anisotropy card uses. `bar: fn` gives it the display cards' small colorbar --
  // a heatmap is the one chart whose y is not the quantity.
  // The hint below is Alfred's own copy, tightened again in his third feedback round. What
  // it deliberately does NOT carry -- the Cho-Vishniac contrast, the per-band-b⊥ caveat on
  // the field-line panel, the legend's GS95 / B06 / iso abbreviations, the rule that lays
  // each slope just clear of the filled region's upper edge, the ~20-bins / fp32 arithmetic
  // behind "imperfect agreement" and the "let the collision develop first" note -- all live
  // in docs.html, in full sentences, under #gen2d: the hint is the one-breath version and
  // the manual is the long one.
  gen2d: {
    // its own box, not the shared SW x SH: a heatmap wants a SQUARE plot area (GSW / GSH)
    label: "2D spectrum E(k&perp;,k&#8741;)", w: GSW, h: GSH,
    avail: cfg => cfg.zslice, bar: o => gen2dBarTicks(o), cmap: GEN2D_CMAP,
    opts: () => [
      { id: "gen", k: "btn", t: "generate", onClick: () => gen2dRun(),
        dis: () => gen2d.busy || !solver,
        ti: "PAUSE the run and sweep every k⊥ band over the frozen state (about a second). "
          + "The plot then stays until you press this again — press Run when you are done "
          + "reading it." },
      // (a `ti` is a title ATTRIBUTE -- plain characters, never entities)
      { id: "gp", ti: "k∥ measured along the field lines, or along the z coordinate — "
          + "the same snapshot in two frames",
        o: [["fl", "field line"], ["z", "coordinate"]] },
      { id: "gq", ti: "which energy is binned; E± = E_u + E_b ± H_c",
        o: [["tot", "E_u+E_b"], ["zp", "E&#8314;"], ["zm", "E&#8315;"]] },
      // the cell colour mapping. Log first, so it is the default (a select with no `v`
      // takes its first option) -- a spectrum falls over decades and linear shows one cell.
      { id: "gc", ti: "colour scale for the cells: log from the noise floor to the peak "
          + "(the default), or linear from 0 to the peak",
        o: [["log", "log colour"], ["lin", "linear colour"]] },
      // "overlays" was the name while this drew the anisotropy card's measured curves too;
      // with those gone (Alfred, second round) the box toggles the three straight reference
      // slopes and nothing else, so it says so.
      { id: "gov", k: "cbl", t: "theory slopes",
        ti: "three reference slopes — GS95 (k∥ ∝ k⊥^2/3), Boldyrev 2006 (k∥ ∝ k⊥^1/2) and "
          + "isotropic (k∥ ∝ k⊥) — each laid just above the upper edge of the filled "
          + "region, so only the slope is ever a claim and never the height" }
    ],
    draw: (c, d, o) => drawGen2D(c, o),
    barTi: o => "colour range of the plotted quantity ("
      + ((o && o.gc) === "lin" ? "linear scale, so the middle tick is the arithmetic mean"
                               : "log scale, so the middle tick is the geometric mean") + ")",
    // The ONE word that is not fixed copy is the colour scale: the sentence has to say what
    // the cells are actually painted on, so "log" becomes "linear" when `gc` is on linear
    // and the rest of it stands exactly as written either way. That is why this hint is a
    // FUNCTION of the card's options -- the `bar(o)` / `barTi(o)` idiom, one row up.
    hint: o => "two-dimensional spectrum E(k&perp;, k&#8741;) from one frozen snapshot "
      + "(<b>generate</b> pauses the run and band-passes the field in k&perp;, band by "
      + "band); colour is " + ((o && o.gc) === "lin" ? "linear" : "log") + " E, y is "
      + "k&#8741;/kunit. overlay lines correspond to GS95 (k&#8741; &prop; "
      + "k&perp;<sup>2/3</sup>), Boldyrev 2006 (k&#8741; &prop; k&perp;<sup>1/2</sup>), and "
      + "isotropic (k&#8741; &prop; k&perp;). experimental feature: imperfect agreement at "
      + "these resolutions."
  }
};
// which chart types this app offers (the equilibrium ones are 2D-only)
const chartTypeKeys = () => Object.keys(CHART_TYPES)
  .filter(k => !CHART_TYPES[k].avail || CHART_TYPES[k].avail(cards.cfg || {}));

const cards = { cfg: null, disp: [], chart: [], hostD: null, hostC: null };

function _mk(tag, cls, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (parent) parent.appendChild(e);
  return e;
}
function _sel(parent, opts, title) {
  const s = _mk("select", null, parent);
  if (title) s.title = title;
  for (const o of opts) {
    const e = document.createElement("option");
    e.value = String(o.v); e.innerHTML = o.t;
    s.appendChild(e);
  }
  return s;
}
// a header range slider, the shape the per-card z slider already has (same class, so it
// shares its flexible width) -- the volume view's level / opacity knobs use it too
function _rng(parent, cls, min, max, step, v, title) {
  const r = _mk("input", cls, parent);
  r.type = "range"; r.min = String(min); r.max = String(max); r.step = String(step);
  r.value = String(v);
  if (title) r.title = title;
  return r;
}
// ... and the same slider with a SHORT LABEL in front of it, the shape the IC editor's
// "z plane" control already has (FEEDBACK 2026-08-10 round 2, item 2: an unlabelled
// header slider is a mystery knob). Label and slider are one control, so the label is
// stashed on the slider and _rngShow moves both.
function _rngLab(parent, cls, min, max, step, v, title, lab) {
  // one flex item, not two: a label ELEMENT wrapping both the text and the slider, so
  // the header's flex-wrap cannot split them onto different lines and the slider's
  // growth is capped by the group's basis (laptop feedback: "opacity" and its slider
  // parted ways, and the bare slider grew to fill the row)
  const l = _mk("label", "rngl", parent);
  const s = _mk("span", null, l);
  s.innerHTML = lab;
  if (title) l.title = title;
  const r = _rng(l, cls, min, max, step, v, title);
  r.lab = l;                              // the whole group, for _rngShow
  r.labSpan = s;                          // the text alone, for anything reading it back
  return r;
}
function _rngShow(r, on) {
  if (!r) return;
  r.style.display = on ? "" : "none";
  if (r.lab) r.lab.style.display = on ? "" : "none";
}

// The per-card z-plane source (3D only): "manual" plus a plane slider, or one of the
// two packet trackers. BOTH card kinds have one -- a display card picks the plane it
// renders, the cut chart the plane it cuts -- so it is written once. Returns the
// elements on `card`, which is all the app's zsliceOf() needs.
//
// Since REFINE_PLAN I2.1 the same select also carries the display card's VIEW: cube
// faces are a view, not a field, so its entries are the cross product of the view
// (slice / cube) with the plane source -- "cube" prefixing the three source values.
// The plane a cube card resolves to is its TOP face (I2.2), which is why the two are one
// control and not two: every combination is meaningful and the header stays one select
// wide on a phone. `cube` is false for the cut chart, where faces mean nothing.
// Since K2.1 the FIELD LINES are a view on the same select: a whole-box object with no
// plane of its own, so it takes no tracker and no slider and is flagged out of the cut
// card with the cube entries.
// Since ISO_PLAN B the VOLUME raymarch is a third such view, and the 3D app's DEFAULT one
// (ratified): the whole box, so like the lines it owns no plane -- and there is no
// "volzp"/tracking variant, because a view that shows every plane makes tracking moot.
const ZSRC_OPTS = [{ v: "manual", t: "z slice" }, { v: "zp", t: "track z&#8314;" },
                   { v: "zm", t: "track z&#8315;" }];
const ZSRC_CUBE = [{ v: "vol", t: "volume" }, { v: "cube", t: "cube faces" },
                   { v: "cubezp", t: "cube + track z&#8314;" },
                   { v: "cubezm", t: "cube + track z&#8315;" }, { v: "lines", t: "field lines" }];
const ZSRC_DEFAULT_CUBE = "vol";        // the 3D display card opens on the volume
function _zSliceControls(card, head, cube) {
  const cfg = cards.cfg;
  if (!cfg.zslice) return;
  card.selZSrc = _sel(head, cube ? ZSRC_OPTS.concat(ZSRC_CUBE) : ZSRC_OPTS,
                      "which z plane this card uses" + (cube ? ", and whether it draws the volume, the cube faces or the field lines" : ""));
  // a display card opens on the volume; the cut chart (cube false) has no such view and
  // keeps the historical first option, its own z plane
  if (cube) card.selZSrc.value = ZSRC_DEFAULT_CUBE;
  card.rSlice = _mk("input", "zslider", head);
  card.rSlice.type = "range"; card.rSlice.min = "0"; card.rSlice.step = "1"; card.rSlice.value = "0";
  card.rSlice.max = String(Math.max(0, cfg.nz() - 1));
}
// the plane source of a card, with the view prefix stripped: every caller that resolves
// a plane (the app's zsliceOf / trackingOn) sees exactly the three pre-I2 values
// ("lines" and "vol" own no plane at all -- reporting them as "manual" is what keeps the
// trackers and the slider out of both whole-box views, K2.5 / ISO_PLAN B)
function _zSrcPlane(v) {
  if (v === "lines" || v === "vol") return "manual";
  return v.indexOf("cube") === 0 ? (v.slice(4) || "manual") : v;
}
// the volume view's two knobs (ISO_PLAN B): the shell level as a fraction of the
// autoscale, and the opacity, which is an extinction per unit BOX length (the march
// integrates 1 - exp(-opacity * shell * dt), so it does not move with the step count).
// The shell WIDTH rides the level (w = 0.4*level in the shader), not a third slider.
const VOL_LEVEL = 0.35, VOL_OPAC = 12;
// the per-card k_perp filter (ISO_PLAN D), in BOTH apps: ONE integer handle in units of the
// box wavenumber, in the forcing band's own idiom (uiFshell). It is a DISPLAY filter -- one
// factor in prepDisp, nothing the solver steps and no chart sees it -- and at 0 it is
// bitwise OFF (see bandFac in physics.js). The travel is the dealias cut, i.e. the last
// shell the spectrum chart bins, so the control speaks the same k as that chart.
//
// It was a two-handle BAND until the on-device pass (feedback round 2, item 3): Alfred ran
// the high handle at the cut every time, so the control is a plain HIGH-PASS now and the
// high end is permanently the cut -- which is the k_hi = 0 case of the same kernel factor,
// i.e. the exact 1.0 that keeps an unfiltered card bitwise unfiltered. The kernel keeps
// both ends (ANISO_PLAN_2 reuses them); only the UI lost one.
const bandTop = () => (solver && solver.nb) || 32;
// ... and the sliders are OPT-IN (item 4): Alfred's "the filter doesn't really work that
// well" -- so one page-wide checkbox in the displays & charts panel decides whether every
// card carries the handle at all. Unticked (the default) the handles are gone AND the
// filter is forced wide open, so an ordinary visit never touches the band arithmetic.
const bandFilterOn = () => { const c = el("cbFilter"); return !!(c && c.checked); };
// the contour overlay's per-card selectors (REFINE_PLAN I2.4), in BOTH apps: in-plane
// field lines of psi (B_perp) or streamlines of phi, on the plane the card displays --
// or BOTH at once (J2.2), which is the alignment view. The value IS the potential's
// display mode, so the solver needs no second mapping; "both" is the pair, in the order
// the two contour sets are drawn (psi in the automatic ink, phi in the fixed accent).
// (a function, not a const: physics.js -- where DISP_PSI lives -- loads after this file)
const _contOpts = () => [{ v: "0", t: "no contours" }, { v: String(DISP_PSI), t: "&psi; contours" },
                         { v: String(DISP_PHI), t: "&phi; contours" },
                         { v: "both", t: "&psi; + &phi;" }];
const CONT_LEVELS = [8, 16, 32];

// ---------------------------------------------------------------------------
// the per-card colorbar (FEEDBACK_2026-08-10 item 12)
// ---------------------------------------------------------------------------
// The STRIP is always the full colormap swept over t in [0,1], because that is exactly
// what `dispX` produces for every display mode: a signed scalar maps -s..+s onto 0..1, a
// magnitude 0..s onto 0..1, and the two sigma modes -1..+1 onto 0..1. So one strip serves
// all ten modes and only the three LABELS differ -- which is what keeps this off the GPU
// entirely. It is painted on a small 2D canvas through `cmapRGB`, the same CMAP_COEF
// table `physics.js` expands into the WGSL at emit time, so the bar cannot drift from the
// pixels above it and NO kernel text changes.
//
// The RANGE the labels quote is the one the GPU colorize actually used: the per-chain
// `maxVal` buffer that maxFinal writes every render (identically named and shaped in both
// apps -- 4 bytes, per display chain -- and it is the *cube-face* maximum in the cube
// view, which is what those pixels were made with too). The CPU never learned it before,
// and no existing readback carries it (readStats is energies, not extrema), so the card
// takes its own: one 4-byte map round trip per card at CBAR_PERIOD, an order of magnitude
// cheaper than the arrow gather already in that loop, skipped entirely for the sigma
// modes (fixed +-1, no autoscale at all) and while the editor view owns the screen.
const CBAR_W = 132, CBAR_H = 9;
const CBAR_PERIOD = 350;                 // ms between label refreshes, per card
const dispMaxRead = (sv, ci) => readBuf(sv.device, sv.chain(ci).buf.maxVal, 4);
// paint the gradient: one 1-logical-px column per step, in the card's own colormap
function cbarPaint(c, which) {
  if (!c) return;
  c.clearRect(0, 0, CBAR_W, CBAR_H);
  for (let i = 0; i < CBAR_W; i++) {
    const rgb = cmapRGB(which, (i + 0.5) / CBAR_W);
    c.fillStyle = "rgb(" + rgb.map(v => Math.round(255 * v)).join(",") + ")";
    c.fillRect(i, 0, 1, CBAR_H);
  }
}
// a tick label: short enough for the 11 px hint font at a 132 px bar
function cbarFmt(v) {
  if (!isFinite(v)) return "";
  const a = Math.abs(v);
  if (a === 0) return "0";
  return (a >= 1e4 || a < 1e-2) ? v.toExponential(1) : v.toPrecision(3);
}

// ---------------------------------------------------------------------------
// save / record (FEEDBACK_2026-08-10 item 13)
// ---------------------------------------------------------------------------
// PNG: the composited view (WebGPU canvas + the overlay canvas + the colorbar) drawn
// into an offscreen 2D canvas and handed to toBlob. WebGPU has no preserveDrawingBuffer:
// `getCurrentTexture` is transient and the canvas keeps only its last PRESENTED image, so
// the card re-renders first and composites in the same task -- what lands in the file is
// then the frame on screen, not a cleared buffer.
// VIDEO: the field canvas alone (the arrows, the field lines and the colorbar are in the
// PNG but not in the video), recorded by whichever of TWO legs the engine can run.
//
//   1. WebCodecs (Chrome; Safari since 16.4). We encode the canvas ourselves with a
//      VideoEncoder and write the .mp4 file ourselves -- see mp4Mux below.
//   2. MediaRecorder, the old leg, kept as the fallback for engines without WebCodecs.
//      It may hand back WebM.
//
// The button is shown when EITHER leg can run and is simply absent otherwise (Alfred's
// "degrade silently", with save unaffected).
//
// WHY leg 1 exists. Chrome's MediaRecorder "video/mp4" writes a FRAGMENTED mp4 -- a moov
// with an empty mvex sample table plus one moof per fragment -- whose delta samples carry
// trun default_sample_flags 0x10000, i.e. "not a sync sample, dependency UNKNOWN", and it
// only emits a keyframe about every 1.4 s. Desktop players ignore the flags and decode
// the lot; iOS AVFoundation believes them, drops every delta sample, and Alfred's iPhone
// played a 30 s recording as about three stills. The codec was never at fault (avc1
// Constrained Baseline with a well-formed avcC). So leg 1: fixed 30 fps timestamps, a
// FORCED keyframe every second, and a PLAIN PROGRESSIVE file -- ftyp + mdat + moov with
// honest sample tables (stts/stss/stsc/stsz/stco) and no moof, no mvex, no trun anywhere.
// That is the shape QuickTime has opened since 2001, and what a phone will play.
const REC_FPS = 30;
const REC_MAX_MS = 30000;                // hard stop, so a forgotten recording stays small
const REC_BITRATE = 5e6;                 // 5 Mbit/s constant: ~19 MB for a full 30 s take
const REC_QMAX = 8;                      // encoder backlog (frames) past which we drop
// RECASYNC_PLAN (2026-08-12): the staging buffers a take's GPU-side captures land in, and
// how long a stop waits for the ones still in flight. THREE buffers is "a couple of frames
// of readback latency is normal, three is the GPU (or the map queue) genuinely behind" --
// with none free the slot is DROPPED, exactly as encoder backpressure drops one, and
// nothing ever waits on the main thread. The drain timeout is the other half of that rule:
// a map that never resolves (a lost device, say) must not hold the finished file hostage,
// so whatever landed within half a second is what gets muxed and the rest are drops.
const REC_POOL = 3;
const REC_DRAIN_MS = 500;
// ?recdebug: while a recording is live, the readout grows one line per recording card --
// frames fed by the rAF loop vs by the watchdog, drops, and the longest gap between two
// loop passes. Diagnostic only, for on-device eyes (a phone has no devtools console);
// harmless to leave in a URL and absent from docs.html on purpose.
const REC_DEBUG = typeof location !== "undefined" && /[?&]recdebug\b/.test(location.search);
// H.264 levels as [level, MaxFS (macroblocks per frame), MaxMBPS (macroblocks per
// second)]. The codec string has to name a level the frame actually fits in or the
// encoder rejects the config; both 512x512 and the 1024x256 wide box are 1024 MBs, which
// is level 3.0, and the 3D cube views are no larger.
const REC_LEVELS = [[0x1e, 1620, 40500], [0x1f, 3600, 108000], [0x20, 5120, 216000],
                    [0x28, 8192, 245760], [0x2a, 8704, 522240], [0x32, 22080, 589824],
                    [0x33, 36864, 983040], [0x34, 36864, 2073600]];
// "avc1.4200<level>": profile_idc 66 (Baseline) with no constraint flags asserted, which
// is the widest thing every H.264 decoder in a phone accepts. No B-frames follow from
// Baseline, hence no reordering, hence no ctts box in the file at all.
function recCodec(w, h) {
  const mb = Math.ceil(w / 16) * Math.ceil(h / 16);
  for (const L of REC_LEVELS) if (mb <= L[1] && mb * REC_FPS <= L[2]) return "avc1.4200" + L[0].toString(16);
  return "avc1.420034";
}
// `avc: { format: "avc" }` is what makes the encoder hand back LENGTH-PREFIXED samples
// and, with the first chunk, a decoderConfig.description holding the avcC payload -- the
// SPS/PPS the stsd box needs. Without it we would get Annex-B and no avcC, and nothing
// would play; the recording bails to leg 2 if a first chunk ever arrives without one.
const recWCConfig = cv =>
  ({ codec: recCodec(cv.width, cv.height), width: cv.width, height: cv.height,
     framerate: REC_FPS, bitrate: REC_BITRATE, bitrateMode: "constant",
     avc: { format: "avc" } });
let recWCOff = false;                    // set when this engine proves the leg unusable
const recProbes = new Map();             // "codec WxH" -> Promise<config|null>, probed once
// ---- RECASYNC_PLAN (2026-08-12): can this engine build a VideoFrame from BYTES? -------
// That question is the whole of the fast capture path. Constructing a VideoFrame from the
// CANVAS is a synchronous readback/conversion on the main thread -- 15-17 ms per capture
// on Alfred's iPhone (?recdebug, on-device), which is what pushed every slot-due loop pass
// past its vsync window and made a live recording stutter. Copying the canvas texture into
// a buffer and building the frame from those bytes when the map resolves moves that cost
// off the hot path entirely. An engine whose WebCodecs cannot take a BufferSource keeps
// today's canvas path: capability probe, degrade silently, never a UA sniff.
// The canvas is alphaMode:"opaque", so the honest four-byte formats are the X variants (an
// alpha channel we never wrote would be a lie the encoder might believe); a canvasFormat
// not in this table means the sync path, not a guess.
const REC_BUF_FMT = { bgra8unorm: "BGRX", rgba8unorm: "RGBX" };
let recBufOff = false;                   // engine latch, same idiom as recWCOff
let recBufTried = false;                 // ... and probed once per session, like recProbes
function recBufProbe() {
  if (recBufTried) return;
  recBufTried = true;
  const fmt = REC_BUF_FMT[canvasFormat];
  if (!fmt) { recBufOff = true; return; }
  try {
    // 2x2 of the canvas's own format = 16 bytes. Built and closed here so the answer costs
    // one frame once, inside the probe the press already awaits.
    const f = new window.VideoFrame(new Uint8Array(16),
      { format: fmt, codedWidth: 2, codedHeight: 2, timestamp: 0 });
    f.close();
  } catch (e) { recBufOff = true; }
}
// ENGINE-level: whether the leg can run at all. The frame size is not part of it -- the
// button's visibility is decided in the card's constructor, before the canvas has been
// sized -- so a size the encoder dislikes is caught by the probe instead, which is
// exactly the path an unsupported config already takes.
const recWCSupported = cv =>
  typeof window !== "undefined" && !recWCOff && !!window.VideoEncoder && !!window.VideoFrame && !!cv;
// VideoEncoder.isConfigSupported is async, so the probe cannot happen inside a click
// handler's synchronous part: it runs on the first press, is cached per config, and a
// rejection (or a throw, on an engine whose isConfigSupported is missing or fussy) means
// a silent fall back to leg 2 rather than a dead button.
function recWCProbe(cv) {
  if (!(cv.width > 0) || !(cv.height > 0)) return Promise.resolve(null);
  const cfg = recWCConfig(cv), key = cfg.codec + " " + cfg.width + "x" + cfg.height;
  let p = recProbes.get(key);
  if (!p) {
    p = Promise.resolve()
      .then(() => (window.VideoEncoder.isConfigSupported
                   ? window.VideoEncoder.isConfigSupported(cfg) : { supported: true }))
      .then(r => {
        // the buffer-path capability rides THIS resolution (RECASYNC_PLAN, 2026-08-12):
        // it is synchronous, so answering it here costs the press no second await, and
        // recStartWC can simply read the settled latch.
        recBufProbe();
        return r && r.supported === false ? null : cfg;
      })
      .catch(() => null);
    recProbes.set(key, p);
  }
  return p;
}
// MP4 (H.264) first on leg 2 as well: it is the one container everything opens, phones
// included -- Alfred's iPhone would not play the VP9 WebM his laptop recorded. Safari's
// MediaRecorder only does MP4; Chrome has recorded video/mp4 since ~126; engines that
// cannot record MP4 fall through to WebM as before.
const REC_MIME = ["video/mp4;codecs=avc1", "video/mp4",
                  "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
// the saved file's extension follows the mime actually negotiated
const recExt = mime => (mime && mime.indexOf("mp4") >= 0 ? "mp4" : "webm");
const recSupported = cv =>
  typeof window !== "undefined" && !!window.MediaRecorder && !!(cv && cv.captureStream);
// either leg: what the rec button's visibility is decided by
const recAnySupported = cv => recWCSupported(cv) || recSupported(cv);
const recMime = () => {
  const M = window.MediaRecorder;
  for (const m of REC_MIME) if (!M.isTypeSupported || M.isTypeSupported(m)) return m;
  return "";
};
// hand a blob to the browser's downloader. A detached <a> is enough in every engine that
// ships WebGPU; the object URL is released once the download has had time to start.
function dlBlob(blob, name) {
  if (!blob) return;
  const u = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = u; a.download = name;
  a.click();
  setTimeout(() => window.URL.revokeObjectURL(u), 10000);
}
// ---- what happens to a FINISHED file (Alfred, 2026-08-11 / 2026-08-12) ------
// Nothing here hands its file straight to dlBlob any more -- neither recording leg, and
// since the second round the PNG save either. On a phone the download is silent: the file
// lands somewhere in Files and is then hard to find and harder to send on, which is the
// opposite of what a picture or a 12 s clip of a simulation is for. So the file waits on
// the card's footer behind a little line of text -- "click start, wait, click stop, then a
// little text appears saying file size, video length, and a download/share button" -- and
// the visitor says where it goes. See DisplayCard.recResult, the ONE place all three
// paths converge on, and its two slots.
//
// Sizes in SI units (kB = 1000 B), which is what a phone's file listing quotes back, and
// only one decimal on the MB -- the number is here to say "small enough to send" or "too
// big to send", not to be exact.
function recSizeText(bytes) {
  const b = Math.max(0, Math.floor(bytes) || 0);
  const k = Math.round(b / 1e3);          // decide the unit on the ROUNDED kB, so
  if (k < 1000) return k + " kB";         // 999.6 kB says "1.0 MB", never "1000 kB"
  return (b / 1e6).toFixed(1) + " MB";
}
// ... and the length in seconds: a decimal only while the clip is short enough for one to
// mean anything. Each leg measures its own honest number -- see the two call sites.
function recLenText(sec) {
  const s = isFinite(sec) && sec > 0 ? sec : 0;
  return (s < 10 ? s.toFixed(1) : String(Math.round(s))) + " s";
}
// The share button exists only where the engine can really share a FILE: Web Share level
// 2, i.e. iOS/Android and a couple of desktop browsers. Capability detection, never a UA
// string (Alfred's standing "degrade silently"): the File is built once, offered to
// canShare, and if that says no the strip simply grows no share button and download stays
// the only -- and on a desktop the obvious -- way out. Both constructor and canShare are
// wrapped because an engine that has the names but dislikes the payload should decline
// quietly rather than take the recording down with it.
function recShareFile(blob, name) {
  try {
    if (!window.File || !navigator.canShare) return null;
    const f = new window.File([blob], name, { type: blob.type || "video/mp4" });
    return navigator.canShare({ files: [f] }) ? f : null;
  } catch (e) { return null; }
}
// taranis-<page>-<field>-t<time>.<ext>
const DISP_SLUG = ["vorticity", "current", "phi", "psi", "u", "b", "zplus", "zminus",
                   "sigma_c", "sigma_r"];
function appSlug() {
  const m = /([^/\\?#]+?)(?:\.html?)?(?:[?#]|$)/.exec(String(location.href || ""));
  const s = m && m[1] ? m[1].replace(/[^A-Za-z0-9_-]/g, "") : "";
  return s || "rmhd";
}
function shotName(mode, ext) {
  return "taranis-" + appSlug() + "-" + (DISP_SLUG[mode] || "field") +
         "-t" + (isFinite(simT) ? simT.toFixed(3) : "0") + "." + ext;
}

// ---------------------------------------------------------------------------
// the MP4 muxer (leg 1 of the recorder)
// ===========================================================================
// Recorder -- the capture legs of a display card (render audit, 2026-08-12)
// ===========================================================================
// This was ~400 lines and fifteen methods inside DisplayCard, more than half of a class
// whose job is to show a field. Nothing about it changed in the move: the same two legs
// (WebCodecs -> mp4Mux, MediaRecorder), the same rAF feeder and visibility-parked
// watchdog (RECRAF_PLAN), the same async GPU-readback capture and its buffer pool
// (RECASYNC_PLAN), the same hard stop, the same bail. What the split buys is that the
// boundary is now written down: the RECORDER owns the encoder, the pool, the timers and
// the two live handles (`wc`, `rec`), and the CARD owns everything a visitor sees -- the
// canvas it captures, the button it relabels through recLive/recIdle, and the result
// strip it hands the finished file to through recResult, which is card UI because the
// SAVE path shares it.
//
// The card keeps `wc` / `rec` / `recBusy` / `recStop` as getters onto this object, so the
// frame loop's REC_DEBUG readout, destroy() and every devtools leg that reads a card's
// live recording state carry on reading exactly what they read before.
class Recorder {
  constructor(card) {
    this.card = card;
    this.rec = null; this.recStop = 0;    // live MediaRecorder, and its hard-stop timer
    this.wc = null;                       // ... or the live WebCodecs recording (leg 1)
    this.recBusy = false;                 // a config probe is in flight
  }
  // toggle: start recording the field canvas, or stop the live recording -- on whichever
  // leg the engine supports (WebCodecs preferred; see the note by REC_FPS). The 30 s
  // timer is a hard stop, not a pause -- an unattended recording must not grow without
  // bound -- and each leg has exactly ONE place that writes the file, so the timer, the
  // button press and destroy() all land in the same path.
  recToggle() {
    if (this.wc || this.rec) { this.recEnd(); return; }
    if (this.recBusy) return;                   // a probe is already in flight
    if (recWCSupported(this.card.cv)) {
      this.recBusy = true;
      recWCProbe(this.card.cv).then(cfg => {
        this.recBusy = false;
        if (this.card.dead || this.wc || this.rec) return;
        if (cfg && recWCSupported(this.card.cv)) this.recStartWC(cfg);
        else if (recSupported(this.card.cv)) this.recStartMR();
      });
      return;
    }
    if (recSupported(this.card.cv)) this.recStartMR();
  }
  recEnd() {
    if (this.wc) this.recStopWC(false);
    else if (this.rec) this.rec.stop();
  }

  // ---- leg 2: MediaRecorder (the fallback; recResult is where it ends) ------
  recStartMR() {
    // a new take replaces the last one's result: two strips on one footer would be two
    // files with the same name a press apart, and the visitor pressing start again has
    // said which one they care about. Cleared HERE, in each leg's start, and not in
    // recToggle: a press whose probe then fails to start anything (a WebCodecs-only
    // engine that dislikes this canvas size) must not have thrown away the one file the
    // visitor still had (adversarial review 2026-08-12, MINOR 1).
    this.card.recClear("video");
    const mime = recMime(), chunks = [];
    const r = new window.MediaRecorder(this.card.cv.captureStream(REC_FPS),
                                       mime ? { mimeType: mime } : undefined);
    const name = shotName(this.card.barMode, recExt(mime));
    // this leg hands back an opaque container built by the engine, so the frames in it are
    // not ours to count: wall clock from start() to onstop is the only length it can
    // honestly quote (leg 1, which writes the file itself, counts samples instead)
    const t0 = Date.now();
    r.ondataavailable = e => { if (e && e.data && e.data.size) chunks.push(e.data); };
    r.onstop = () => {
      clearTimeout(this.recStop);
      this.rec = null; this.recStop = 0;
      this.card.recIdle();
      this.card.recResult("video", new window.Blob(chunks, { type: mime || "video/mp4" }), name,
                     (Date.now() - t0) / 1000);
    };
    this.rec = r;
    r.start();
    this.card.recLive();
    this.recStop = setTimeout(() => { if (this.rec === r) r.stop(); }, REC_MAX_MS);
  }

  // ---- leg 1: WebCodecs -> mp4Mux ------------------------------------------
  recStartWC(cfg) {
    this.card.recClear("video");               // same rule as recStartMR: replace on START
    // ONE clock sample feeds the cadence fields: `due` is the wall-clock time of the next
    // capture slot (recCapture). `lastRaf`/`maxGap` are DIAGNOSTIC only since round 2 --
    // the watchdog parks on visibility, not on a heartbeat -- and `rafN`/`wdN` count which
    // feeder put each frame in the file (?recdebug shows the tallies and maxGap;
    // lastRaf just feeds maxGap).
    const t0 = performance.now();
    const W = { chunks: [], avcC: null, n: 0, drop: 0, timer: 0, bailed: false, done: false,
                due: t0 + 1000 / REC_FPS, lastRaf: t0, maxGap: 0, rafN: 0, wdN: 0,
                tV: 0, tE: 0,               // max ms in the capture / in the encode (recdebug)
                w: this.card.cv.width, h: this.card.cv.height, name: shotName(this.card.barMode, "mp4"),
                // ---- RECASYNC_PLAN (2026-08-12): the GPU-readback capture path --------
                // `bufOn` is decided ONCE per take from the settled probe latch, so a
                // recording never changes paths under its own feet; `fmt` is the
                // VideoFrame format the canvas's bytes really are. `pool` (lazy, three
                // staging buffers, owned by this W) and `pend` are the captures in
                // flight; `chain` is the ordered encode chain -- null until the buffer
                // path first uses it, which is also what keeps the sync path's stop
                // exactly as it was. `gone` is set when the pool is destroyed: a map
                // resolving after that must find it and do nothing.
                bufOn: !recBufOff && !!this.card.ctx && !!device, fmt: REC_BUF_FMT[canvasFormat],
                pool: null, bpr: 0, pad: false, pend: 0, onDrain: null, gone: false,
                chain: null, tL: 0,         // max capture->encode lag in ms (recdebug)
              };
    W.enc = new window.VideoEncoder({
      output: (chunk, meta) => {
        const d = meta && meta.decoderConfig && meta.decoderConfig.description;
        if (d && !W.avcC) W.avcC = mp4Bytes(d);
        // avcC is the ONLY place a progressive file keeps the SPS/PPS -- there is no
        // in-band parameter set to fall back on -- so a first chunk without one means we
        // can never write a playable mp4. Bail to leg 2 NOW, one frame in, rather than
        // at flush time holding 30 s of unusable samples.
        if (!W.avcC) { this.recBailWC(W); return; }
        if (W.bailed) return;
        const b = new Uint8Array(chunk.byteLength);
        chunk.copyTo(b);
        W.chunks.push({ data: b, key: chunk.type === "key" });
      },
      // encoder dead: write what we have. Guarded on ownership -- a stale encoder's
      // error callback can land AFTER this recording stopped and the NEXT one began,
      // and must not terminate the newcomer (adversarial review 2026-08-10, MINOR 1).
      error: () => { if (this.wc === W) this.recStopWC(true); }
    });
    W.enc.configure(cfg);
    this.wc = W;
    // the WATCHDOG interval (RECRAF_PLAN, 2026-08-12): frames normally come from the rAF
    // loop via recCapture, but rAF is throttled to a crawl (or stopped) in a background
    // tab, and the editor view does not render cards at all -- and there a silently
    // stretched recording is exactly what the fixed timestamps must not be paired with.
    // So the timer stays, at the same 1000/30 ms, and feeds the recording itself on a
    // hidden page or under the editor view (the park condition lives in recTick). Its
    // extra render() then costs only where there is no visible display to stutter.
    W.timer = setInterval(() => this.recTick(W), Math.round(1000 / REC_FPS));
    this.card.recLive();
    this.recStop = setTimeout(() => { if (this.wc === W) this.recStopWC(false); }, REC_MAX_MS);
  }
  // The ONE place a recording frame is encoded, whichever feeder brought it (RECRAF_PLAN,
  // 2026-08-12): the frame index, its nominal 1/30 s timestamps, the forced-keyframe
  // cadence and the drop-frame guard exist here once, so the two paths cannot drift apart.
  // It does NOT render -- the rAF path is already inside the render's own task, and the
  // watchdog renders itself immediately before calling this.
  // `src` is the SOURCE of this frame's pixels, and there are two kinds (RECASYNC_PLAN,
  // 2026-08-12): absent means this card's canvas (the sync path -- the watchdog, and any
  // engine that cannot take bytes), and a `{bytes, format, w, h}` record means the mapped
  // GPU copy of it. Everything else -- index, timestamp, keyframe cadence, backpressure --
  // is identical by construction, which is the point of there being one function.
  recEncodeFrame(W, src) {
    // backpressure: when the encoder is behind, DROP this frame instead of queueing it.
    // The frame index is not advanced, so the timestamps stay exactly 1/30 s apart and
    // the forced-keyframe cadence stays exact -- a slow machine records fewer seconds of
    // wall clock, rather than a file whose sample table lies about its own timing.
    if (W.enc.encodeQueueSize > REC_QMAX) { W.drop++; return false; }
    // the two halves of the capture cost, timed separately (?recdebug, round 3): `vf` is
    // what a capture costs the MAIN THREAD and `enc` is what the encode costs. On the sync
    // path that split is VideoFrame-from-canvas (the classic iOS expense, and the half a
    // Worker could NOT take) vs encode(); on the buffer path the main-thread half is the
    // copy+submit timed in recCaptureBuf, and BOTH halves here -- building the frame from
    // bytes and submitting it -- ride `enc`, because they happen a beat later in the chain
    // and no longer stretch a display frame. Two clock reads per frame either way; kept
    // unconditional so the numbers exist the moment anyone asks.
    const t1 = performance.now();
    const init = { timestamp: Math.round(W.n * 1e6 / REC_FPS),
                   duration: Math.round(1e6 / REC_FPS) };
    let f;
    if (src && src.bytes) {
      init.format = src.format; init.codedWidth = src.w; init.codedHeight = src.h;
      f = new window.VideoFrame(src.bytes, init);
    } else {
      f = new window.VideoFrame(this.card.cv, init);
    }
    const t2 = performance.now();
    // a forced keyframe every second: the cadence iOS wanted and MediaRecorder would not
    // give. It is also every seek point the file has, stss being built from these.
    try { W.enc.encode(f, { keyFrame: (W.n % REC_FPS) === 0 }); } finally { f.close(); }
    const t3 = performance.now();
    if (src && src.bytes) { if (t3 - t1 > W.tE) W.tE = t3 - t1; }
    else {
      if (t2 - t1 > W.tV) W.tV = t2 - t1;
      if (t3 - t2 > W.tE) W.tE = t3 - t2;
    }
    W.n++;
    return true;                            // fed: the callers' rafN/wdN tallies key on this
  }
  // ---- the buffer capture path (RECASYNC_PLAN, 2026-08-12) --------------------
  // The hot half: copy this card's canvas texture into a free staging buffer and submit,
  // all synchronously in the render's own task (getCurrentTexture is transient), then let
  // go. Microseconds of command encoding instead of the 15-17 ms a VideoFrame-from-canvas
  // cost the phone -- the readback itself happens on the GPU's clock and lands in
  // recEncodeMapped a beat later, which is fine: the timestamps are ours, not the map's.
  recCaptureBuf(W) {
    const t1 = performance.now();
    // the canvas can be resized under a live take (a preset or a grid change rebuilds the
    // cards): the encoder is configured for W.w x W.h, so a copy of the new size would be
    // a validation error rather than a frame. Drop the slot instead -- the honest-length
    // rule again -- and let the take end at the size it started.
    if (this.card.cv.width !== W.w || this.card.cv.height !== W.h) { W.drop++; return; }
    if (!W.pool && !this.recPoolMake(W)) { W.drop++; return; }   // this slot was due too
    let s = null;
    for (const e of W.pool) if (!e.busy) { s = e; break; }
    // all three buffers are still waiting on their maps: the readback is genuinely behind,
    // so this slot is lost exactly as an encoder-backpressure slot is. Nothing waits.
    if (!s) { W.drop++; return; }
    try {
      const ce = device.createCommandEncoder();
      ce.copyTextureToBuffer({ texture: this.card.ctx.getCurrentTexture() },
                             { buffer: s.b, bytesPerRow: W.bpr, rowsPerImage: W.h },
                             { width: W.w, height: W.h, depthOrArrayLayers: 1 });
      device.queue.submit([ce.finish()]);
    } catch (e) {
      // an engine that accepted the configure but not the copy: fall back to the sync
      // canvas path for the rest of this take rather than lose every remaining frame
      W.bufOn = false; W.drop++;
      return;
    }
    s.busy = true; W.pend++;
    const t2 = performance.now();
    if (t2 - t1 > W.tV) W.tV = t2 - t1;
    if (!W.chain) W.chain = Promise.resolve();
    // THE ORDERED ENCODE CHAIN. mapAsync resolution order across distinct buffers is not
    // something to rely on, VideoEncoder requires monotonic timestamps, and mp4Mux writes
    // a UNIFORM stts (checkmp4 asserts equal deltas) -- so index, timestamp and keyframe
    // are all assigned at ENCODE time, inside one promise chain whose steps cannot
    // interleave. Whichever capture's bytes arrive first is simply frame n: a dropped or
    // failed capture leaves fewer frames and NO hole in the sample table.
    s.b.mapAsync(GPUMapMode.READ).then(
      () => { W.chain = W.chain.then(() => { this.recEncodeMapped(W, s, t2); this.recPend(W); }); },
      () => { s.busy = false; if (!W.gone) W.drop++; this.recPend(W); });
  }
  // three staging buffers, made on the first capture of a take (a take that never captures
  // -- pressed and stopped inside one slot -- allocates nothing). bytesPerRow must be a
  // multiple of 256: at every preset size the row is already aligned, but the box is
  // user-sizable, so the padded case is real and recEncodeMapped compacts it.
  recPoolMake(W) {
    const pool = [];
    try {
      W.bpr = Math.ceil(W.w * 4 / 256) * 256;
      W.pad = W.bpr !== W.w * 4;
      for (let i = 0; i < REC_POOL; i++)
        pool.push({ b: device.createBuffer({ size: W.bpr * W.h,
                      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }), busy: false });
      W.pool = pool;
      return true;
    } catch (e) {
      // no memory: sync path -- and the buffers a mid-loop throw DID create go back
      // (adversarial review 2026-08-12, MINOR 3)
      for (const p of pool) { try { p.b.destroy(); } catch (err) {} }
      W.bufOn = false; W.pool = null; return false;
    }
  }
  // one capture accounted for, whichever way it ended. When the last one lands, a stop
  // that is waiting on the drain can go on to flush and mux.
  recPend(W) {
    if (W.pend > 0) W.pend--;
    if (!W.pend && W.onDrain) W.onDrain();
  }
  // the ASYNC TAIL, running as one step of the chain: the map resolved, so this capture's
  // bytes exist. On the aligned row the mapped range is handed to the VideoFrame
  // constructor as it is (the constructor copies), on the padded one the rows are
  // compacted into a tight buffer first. Then unmap, and the buffer is free again.
  recEncodeMapped(W, s, tCap) {
    if (W.gone) return;                     // torn down while this sat in the chain
    try {
      const ab = s.b.getMappedRange();
      const row = W.w * 4;
      let bytes;
      if (W.pad) {
        bytes = new Uint8Array(row * W.h);
        const src = new Uint8Array(ab);
        for (let y = 0; y < W.h; y++) bytes.set(src.subarray(y * W.bpr, y * W.bpr + row), y * row);
      } else {
        bytes = new Uint8Array(ab);
      }
      // capture-submit to encode: the "arrives a beat late" number, ?recdebug's `lag`
      const lag = performance.now() - tCap;
      if (lag > W.tL) W.tL = lag;
      if (this.recEncodeFrame(W, { bytes: bytes, format: W.fmt, w: W.w, h: W.h })) W.rafN++;
    } catch (e) {
      // a failed capture is a drop, never a throw -- and, like a failed COPY, it also
      // latches this take back onto the sync canvas path: an engine whose 2x2 probe frame
      // passed but whose full-size frames throw would otherwise drop every remaining slot
      // and hand back a 0-chunk take with no file and no explanation (adversarial review
      // 2026-08-12, MINOR 1)
      W.drop++; W.bufOn = false;
    }
    try { s.b.unmap(); } catch (e) {}
    s.busy = false;
  }
  // the pool's ONE teardown, called from every route a recording ends by. `gone` first:
  // the buffers are about to stop existing, so a map still in flight (and the rejection
  // destroy() hands it) must find a recording that wants nothing more from it.
  recPoolFree(W) {
    W.gone = true;
    for (const e of (W.pool || [])) { try { e.b.destroy(); } catch (err) {} }
    W.pool = null;
  }
  // wait for the captures still in flight before the file is written (RECASYNC_PLAN 5).
  // The timeout is the guard against a map that never resolves: half a second, then mux
  // what landed. It cannot strand the chain -- an encode step is synchronous work -- so
  // awaiting the chain afterwards only orders the last step ahead of the flush.
  recDrainWC(W) {
    return new Promise(res => {
      if (!W.pend) { res(); return; }
      // a timed-out capture is a DROP, and says so (plan 5; adversarial review 2026-08-12,
      // MINOR 2): the teardown that follows sets `gone` before destroying the buffers, so
      // the rejection handlers stay silent -- count the stragglers here instead. W.drop is
      // diagnostic (?recdebug); the file itself keys on the chunks that landed.
      const t = setTimeout(() => { W.onDrain = null; W.drop += W.pend; res(); }, REC_DRAIN_MS);
      W.onDrain = () => { clearTimeout(t); W.onDrain = null; res(); };
    }).then(() => W.chain).catch(() => {});
  }
  // leg 1's PRIMARY feeder (RECRAF_PLAN, 2026-08-12), called from loop() in the SAME
  // synchronous task as this card's render(). The old feeder was the interval below, which
  // cost an extra full render() per tick plus the VideoFrame copy and the encode, all on
  // the main thread and at an arbitrary phase against the rAF loop -- which is what made
  // an iPhone stutter visibly for the whole take (Alfred, on-device, 2026-08-12). Riding
  // the render costs zero extra renders and beats against nothing.
  recCapture() {
    const W = this.wc;
    if (!W || W.done) return;               // not recording (or a loop iteration after stop)
    const now = performance.now();
    // the gap between consecutive loop passes, kept as a DIAGNOSTIC (?recdebug): on a
    // phone this is the number that says whether the loop itself is the bottleneck.
    if (now - W.lastRaf > W.maxGap) W.maxGap = now - W.lastRaf;
    W.lastRaf = now;
    const T = 1000 / REC_FPS;
    if (now < W.due) return;                // between slots (a 120 Hz phone): not a drop
    // a slot is due. If the loop was so late that a WHOLE further slot went by, those slots
    // are counted lost and the next one is set from now -- never backfilled, because a
    // backfilled frame would put a stale image at a timestamp it never had. The honest-
    // length rule stands: fewer recorded seconds than wall clock, never a lying sample
    // table. At nominal cadence the `else` keeps `due` drift-free.
    if (now - W.due > T) { W.drop += Math.floor((now - W.due) / T); W.due = now + T; }
    else W.due += T;
    // the fast path since RECASYNC (2026-08-12): submit a GPU copy and return. The frame
    // itself is built from those bytes in the chain when the map resolves, and the rafN
    // tally is incremented THERE -- it counts frames this feeder put in the FILE, and on
    // this path that is not knowable yet.
    if (W.bufOn) return this.recCaptureBuf(W);
    if (this.recEncodeFrame(W)) W.rafN++;
  }
  // the WATCHDOG tick (RECRAF_PLAN, 2026-08-12): identical to the feeder this leg shipped
  // with -- backpressure, render, encode, at the interval's own cadence -- but it runs only
  // where the rAF feeder is KNOWN-ABSENT: a hidden page (rAF stopped or crawling) or the
  // editor view (loop() skips the render/capture pair). Round 1 parked it on a TIMING
  // heuristic instead -- recCapture silent for 3.5 slots -- and Alfred's phone showed why
  // that was wrong: a visible loop whose post-render readback awaits stretch past the
  // threshold got BOTH feeders, rAF captures plus 30 Hz watchdog render+encodes, MORE
  // main-thread work than the pre-RECRAF recorder (on-device, "worse if anything",
  // 2026-08-12). Visibility is the condition the watchdog exists for, so it is the
  // condition it runs on. No slot-due check here: when the watchdog IS the feeder, the
  // tick cadence is the 30 fps the timestamps promise, exactly as it always was. The queue
  // check sits ahead of the render as well as inside the helper, so a tick that is only
  // going to drop the frame does not pay for a render first.
  //
  // The watchdog keeps the SYNCHRONOUS canvas capture unconditionally, buffer path or not
  // (RECASYNC_PLAN, 2026-08-12): it renders off-screen anyway, so a stalled main thread
  // there costs nothing anyone can see, and keeping the async pool out of the
  // background-throttled world means never having to reason about maps that a hidden page
  // is starving of callbacks.
  recTick(W) {
    if (this.wc !== W || W.done) return;
    if (!((typeof document !== "undefined" && document.hidden) || icDraw.on)) return;
    if (W.enc.encodeQueueSize > REC_QMAX) { W.drop++; return; }
    this.card.render();      // same reason as saveShot: getCurrentTexture is transient
    if (this.recEncodeFrame(W)) W.wdN++;
    // a fed tick refreshes the diagnostic clock too, so ?recdebug's `gap` keeps meaning
    // "the longest stretch NOBODY fed" -- without this, an editor-view stint would report
    // its whole dwell time as one loop gap (adversarial review 2026-08-12, r2 MINOR 3)
    W.lastRaf = performance.now();
    // re-base the slot clock: a frame the watchdog just PUT IN THE FILE must not be
    // counted as a dropped slot by the first recCapture after rAF resumes -- with `due`
    // frozen at handoff, every watchdog-fed slot would be double-booked into W.drop, and
    // the returning rAF would double-feed the slot the watchdog had just filled
    // (adversarial review 2026-08-12, MINOR 1).
    W.due = performance.now() + 1000 / REC_FPS;
  }
  // the ONE place a WebCodecs recording ends: button, 30 s timer, destroy(), or encoder
  // error (`broken`, where flushing a dead encoder would only throw).
  recStopWC(broken) {
    const W = this.wc;
    if (!W || W.done) return;
    W.done = true;
    clearInterval(W.timer); clearTimeout(this.recStop);
    this.wc = null; this.recStop = 0;
    this.card.recIdle();
    const fin = () => {
      try { W.enc.close(); } catch (e) {}
      this.recPoolFree(W);      // the staging buffers go here, on every route (RECASYNC 5)
      const mp4 = mp4Mux({ width: W.w, height: W.h, fps: REC_FPS, avcC: W.avcC, chunks: W.chunks });
      // the length is the samples that ended up IN the file over the fixed 30 fps the
      // sample table declares -- honest under the drop-frame guard, where a slow machine
      // records fewer seconds of wall clock than the clip lasts
      if (mp4) this.card.recResult("video", new window.Blob([mp4], { type: "video/mp4" }), W.name,
                              W.chunks.length / REC_FPS);
    };
    // flush() delivers the frames the encoder is still holding, so it has to complete
    // before the mux -- and its rejection is not a reason to lose the file either.
    const flush = () => { try { W.enc.flush().then(fin, fin); } catch (e) { fin(); } };
    // ... and since RECASYNC (2026-08-12) captures can be in FLIGHT when the stop lands:
    // drain them first (W.done above already bars new ones), so the last half-second of a
    // take is in the file rather than in three staging buffers about to be destroyed. A
    // broken encoder skips the drain: there is nothing left that could encode, and the
    // in-flight maps will resolve into a torn-down W and do nothing. A take that never
    // touched the buffer path keeps EXACTLY the stop it always had.
    if (broken) fin();
    else if (!W.bufOn && !W.chain) flush();
    else this.recDrainWC(W).then(flush, flush);
  }
  // no avcC: this engine's WebCodecs leg is unusable, so switch legs mid-press and turn
  // it off for the rest of the session rather than hand back an unplayable file.
  // The card is closing mid-recording: write what we have. The stream (or the canvas the
  // encoder is reading) dies with the card, and losing the file silently would be worse
  // than a short one. The card sets its `dead` flag BEFORE calling this, which is what
  // sends the finished file straight to a download instead of to a strip on a footer that
  // is about to be removed (recResult's dead-card branch).
  destroy() {
    if (this.wc) this.recStopWC(false);
    if (this.rec) { clearTimeout(this.recStop); try { this.rec.stop(); } catch (e) {} }
  }
  recBailWC(W) {
    if (W.bailed || this.wc !== W) return;
    W.bailed = true; W.done = true;
    clearInterval(W.timer); clearTimeout(this.recStop);
    this.wc = null; this.recStop = 0;
    recWCOff = true;
    try { W.enc.close(); } catch (e) {}
    // this leg ends here rather than in recStopWC, so it frees the staging buffers itself
    // (RECASYNC_PLAN 5, "all routes"): the file is being thrown away, so there is nothing
    // to drain -- captures still in flight resolve into a torn-down W and do nothing.
    this.recPoolFree(W);
    if (recSupported(this.card.cv) && !this.card.dead) this.recStartMR();
    else this.card.recIdle();
  }
}

// ---------------------------------------------------------------------------
// display cards: the quantity list, physics.js modes 0..9 plus the two Elsasser
// vorticities. BOTH pages carried this table verbatim (render audit, 2026-08-12) -- the
// modes are physics.js's and neither page adds or drops one, so it is written once here
// and handed to cardsInit as `fields`. It is orthogonal to the 3D page's VIEWS: a cube,
// a volume and the field lines are choices in the card's z-source select (REFINE_PLAN
// I2.1, ISO_PLAN B), and every quantity below renders in all of them, so there are no
// cube or volume entries in it.
// ---------------------------------------------------------------------------
const DISP_FIELDS = [
  { v: 4, t: "velocity |u|", d: "u = z&#770;&times;&nabla;&phi;" },
  { v: 5, t: "magnetic |b|", d: "b = z&#770;&times;&nabla;&psi;" },
  { v: 2, t: "&phi;" }, { v: 3, t: "&psi;" },
  { v: 0, t: "vorticity", d: "&omega; = &nabla;&sup2;&phi; = z&#770;&middot;&nabla;&times;u" },
  { v: 1, t: "current", d: "j = &nabla;&sup2;&psi; = z&#770;&middot;&nabla;&times;b" },
  { v: 10, t: "Elsasser &omega;&#8314;",
    d: "&omega;&#8314; = &nabla;&sup2;&zeta;&#8314; = &omega; + j, &zeta;&#8314; = &phi; + &psi;" },
  { v: 11, t: "Elsasser &omega;&#8315;",
    d: "&omega;&#8315; = &nabla;&sup2;&zeta;&#8315; = &omega; &minus; j, &zeta;&#8315; = &phi; &minus; &psi;" },
  { v: 6, t: "Elsasser |z&#8314;|", d: "z&#8314; = u + b" },
  { v: 7, t: "Elsasser |z&#8315;|", d: "z&#8315; = u &minus; b" },
  { v: 8, t: "cross helicity &sigma;_c",
    d: "&sigma;_c = 2u&middot;b / (|u|&sup2;+|b|&sup2;)" },
  { v: 9, t: "residual energy &sigma;_r",
    d: "&sigma;_r = (|u|&sup2;&minus;|b|&sup2;) / (|u|&sup2;+|b|&sup2;)" }
];

// ---------------------------------------------------------------------------
// An MP4 is a tree of boxes, each [uint32 size][4-char type][payload]. Everything here
// is written by hand because the whole point of the exercise is to control what the
// sample tables say: iOS threw the MediaRecorder file away on the strength of its flags
// (see the note above), so ours state, in the oldest and least surprising way possible,
// that the samples are contiguous, evenly spaced and that these particular ones are sync
// samples. Boxes are assembled bottom-up -- children first, parent size from theirs --
// so no offset in this code is written by hand except stco's, and mdat is emitted before
// moov precisely so that one is known when moov is built. (moov-last is what every
// muxer that cannot seek backwards does; players handle it everywhere. Progressive
// download over HTTP would want it first, but this file goes straight to disk.)
const mp4U32 = n => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
const mp4U16 = n => [(n >>> 8) & 255, n & 255];
const mp4Type = s => [s.charCodeAt(0), s.charCodeAt(1), s.charCodeAt(2), s.charCodeAt(3)];
// flatten byte arrays and Uint8Arrays, in whatever order a box wants them
function mp4Cat(parts) {
  let n = 0;
  for (const p of parts) n += p.length;
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
function mp4Box(type, ...parts) {
  const body = mp4Cat(parts);
  return mp4Cat([mp4U32(body.length + 8), mp4Type(type), body]);
}
// a "full box" carries a version byte and 24 flag bits ahead of its payload
function mp4Full(type, ver, flags, ...parts) {
  return mp4Box(type, [ver & 255, (flags >>> 16) & 255, (flags >>> 8) & 255, flags & 255], ...parts);
}
// an ArrayBuffer or any view of one, as bytes (VideoEncoder metadata is either)
function mp4Bytes(v) {
  if (!v) return null;
  return v.buffer ? new Uint8Array(v.buffer, v.byteOffset || 0, v.byteLength) : new Uint8Array(v);
}
// the unity display matrix, as 9 fixed-point entries
const MP4_MATRIX = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];

// ({width, height, fps, avcC, chunks: [{data: Uint8Array, key: bool}, ...]}) -> Uint8Array
// holding a complete, plain, progressive MP4; null if there is nothing playable to write.
// `data` is exactly what the encoder produced under avc: {format: "avc"} -- length-
// prefixed NAL units, which IS the mp4 sample format, so there is no Annex-B conversion
// anywhere in this file.
function mp4Mux(o) {
  const S = (o && o.chunks) || [], n = S.length;
  const avcC = o && mp4Bytes(o.avcC);
  if (!n || !avcC || !avcC.length) return null;
  const w = o.width | 0, h = o.height | 0, fps = o.fps || REC_FPS;
  // media timescale = 1000*fps, so one frame is exactly 1000 ticks and stts is a single
  // run with no rounding drift over 30 s; the movie timescale stays the customary 1000.
  const ts = 1000 * fps, dt = 1000;
  const durMs = Math.round(1000 * n / fps);

  const ftyp = mp4Box("ftyp", mp4Type("isom"), mp4U32(0x200),
                      mp4Type("isom"), mp4Type("iso2"), mp4Type("avc1"), mp4Type("mp41"));
  let mdatN = 0;
  for (const c of S) mdatN += c.data.length;
  const mdatOff = ftyp.length + 8;            // file offset of the FIRST sample's byte
  // stco is 32-bit. 30 s at 5 Mbit/s is ~19 MB, so co64 would be dead code; assert
  // instead of writing a branch that could never be exercised.
  if (mdatOff + mdatN > 0xffffffff) return null;

  // per-sample tables. stss lists the 1-based indices of the SYNC samples -- the forced
  // keyframes -- and its presence is the statement "the samples not listed are not sync
  // samples, and everything else about them is ordinary": no sdtp, no dependency-unknown.
  const sizes = [], sync = [];
  let nsync = 0;
  for (let i = 0; i < n; i++) {
    sizes.push(mp4U32(S[i].data.length));
    if (S[i].key) { sync.push(mp4U32(i + 1)); nsync++; }
  }
  const avc1 = mp4Box("avc1",
    [0, 0, 0, 0, 0, 0], mp4U16(1),             // reserved, data_reference_index = 1
    mp4U16(0), mp4U16(0), mp4U32(0), mp4U32(0), mp4U32(0),   // pre_defined / reserved
    mp4U16(w), mp4U16(h),
    mp4U32(0x00480000), mp4U32(0x00480000),    // 72 dpi horizontal / vertical, by custom
    mp4U32(0), mp4U16(1),                      // reserved, frame_count = 1
    new Uint8Array(32),                        // compressorname: empty Pascal string
    mp4U16(0x0018), mp4U16(0xffff),            // depth 24, pre_defined = -1
    mp4Box("avcC", avcC));                     // VERBATIM from the encoder's metadata
  const stbl = mp4Box("stbl",
    mp4Full("stsd", 0, 0, mp4U32(1), avc1),
    mp4Full("stts", 0, 0, mp4U32(1), mp4U32(n), mp4U32(dt)),
    mp4Full("stss", 0, 0, mp4U32(nsync), mp4Cat(sync)),
    mp4Full("stsc", 0, 0, mp4U32(1), mp4U32(1), mp4U32(n), mp4U32(1)),
    mp4Full("stsz", 0, 0, mp4U32(0), mp4U32(n), mp4Cat(sizes)),
    mp4Full("stco", 0, 0, mp4U32(1), mp4U32(mdatOff)));
  const minf = mp4Box("minf",
    mp4Full("vmhd", 0, 1, mp4U16(0), mp4U16(0), mp4U16(0), mp4U16(0)),
    // one data reference, flagged self-contained: the samples are in THIS file
    mp4Box("dinf", mp4Full("dref", 0, 0, mp4U32(1), mp4Full("url ", 0, 1))),
    stbl);
  const mdia = mp4Box("mdia",
    mp4Full("mdhd", 0, 0, mp4U32(0), mp4U32(0), mp4U32(ts), mp4U32(n * dt),
            mp4U16(0x55c4), mp4U16(0)),        // language "und", pre_defined
    mp4Full("hdlr", 0, 0, mp4U32(0), mp4Type("vide"), mp4U32(0), mp4U32(0), mp4U32(0),
            mp4Type("Vide"), mp4Type("oHan"), mp4Type("dler"), [0]),
    minf);
  const trak = mp4Box("trak",
    // flags 7 = track enabled, in the movie, in the preview
    mp4Full("tkhd", 0, 7, mp4U32(0), mp4U32(0), mp4U32(1), mp4U32(0), mp4U32(durMs),
            mp4U32(0), mp4U32(0), mp4U16(0), mp4U16(0), mp4U16(0), mp4U16(0),
            mp4Cat(MP4_MATRIX.map(mp4U32)),
            mp4U32(w * 65536), mp4U32(h * 65536)),   // display size, 16.16 fixed point
    mdia);
  const moov = mp4Box("moov",
    mp4Full("mvhd", 0, 0, mp4U32(0), mp4U32(0), mp4U32(1000), mp4U32(durMs),
            mp4U32(0x00010000), mp4U16(0x0100), mp4U16(0), mp4U32(0), mp4U32(0),
            mp4Cat(MP4_MATRIX.map(mp4U32)),
            mp4U32(0), mp4U32(0), mp4U32(0), mp4U32(0), mp4U32(0), mp4U32(0),
            mp4U32(2)),                        // next_track_ID
    trak);
  return mp4Cat([ftyp, mp4U32(mdatN + 8), mp4Type("mdat")]
                .concat(S.map(c => c.data)).concat([moov]));
}

class DisplayCard {
  constructor(ci) {
    const cfg = cards.cfg;
    this.ci = ci;
    const root = _mk("div", "card disp", cards.hostD);
    this.root = root;
    const head = _mk("div", "cardhead", root);
    this.selField = _sel(head, cfg.fields, "displayed quantity");
    _zSliceControls(this, head, !!cfg.cube);
    const al = _mk("label", "cbl", head);
    this.cbArrow = _mk("input", null, al);
    this.cbArrow.type = "checkbox"; this.cbArrow.checked = true;
    // ISO_PLAN B: the vol view has no arrows (there is no plane to draw them on), so the
    // same checkbox carries the FIELD-LINE overlay there -- one control, two independent
    // remembered states (cbState below), because their defaults differ.
    this.cbTx = document.createTextNode("arrows");
    al.appendChild(this.cbTx);
    al.title = "vector overlay on the |u| / |b| / |z±| modes (on the cube: its top face); "
      + "in the volume view, the field lines instead";
    // the volume view's two knobs, hidden (like the contour count) wherever they mean
    // nothing. 3D only: cfg.cube is what says this app has a box to march.
    if (cfg.cube) {
      this.rLevel = _rngLab(head, "zslider", 0.05, 0.95, 0.01, VOL_LEVEL,
                            "volume view: shell level, as a fraction of the colour range "
                            + "(the shells sit at ±level, and their width rides it)",
                            "% range");
      this.rOpac = _rngLab(head, "zslider", 1, 40, 1, VOL_OPAC,
                           "volume view: opacity of the shells", "opacity");
    }
    // the display-only k_perp high-pass (ISO_PLAN D + item 3), both apps, every view: at 0,
    // i.e. off, and absent altogether unless the panel's filter checkbox is ticked (item 4).
    // Its travel follows the grid in apply().
    this.rBLo = _rngLab(head, "zslider", 0, bandTop(), 1, 0,
                        "display filter: hide k⊥ below this, in units of the box wavenumber "
                        + "(left end = off; the top end is always the dealias cut). "
                        + "The simulation itself is NOT filtered.",
                        "filter k<sub>min</sub>");
    this.selCont = _sel(head, _contOpts(), "in-plane field lines: psi -> B_perp, phi -> streamlines");
    this.selLev = _sel(head, CONT_LEVELS.map(n => ({ v: n, t: n + " levels" })), "contour level count");
    this.selBg = _sel(head, [{ v: "0", t: "field bg" }, { v: "1", t: "plain bg" }],
                      "draw the contours over the field, or over a blank plate");
    // both only mean anything with contours on
    this.selLev.style.display = "none";
    this.selBg.style.display = "none";
    this.selCmap = _sel(head, CMAP_NAMES.map((n, i) => ({ v: i, t: n })), "colormap");
    this.btnClose = _mk("button", "x", head);
    this.btnClose.innerHTML = "&times;";
    this.btnClose.title = "close this display";

    const wrap = _mk("div", "cvwrap", root);
    this.wrap = wrap;
    this.cv = _mk("canvas", "cvmain", wrap);
    this.cvVec = _mk("canvas", "cvvec", wrap);
    // The caption line doubles as the card's FOOTER: the colorbar and the two capture
    // buttons are right-aligned on it, under the field and never over it (items 12/13).
    // It wraps, so a narrow card drops them onto their own line instead of squeezing
    // the caption.
    const foot = _mk("div", "viewfoot", root);
    this.foot = foot;                     // the result strip is appended here (recResult)
    this.cap = _mk("div", "viewcap", foot);
    this.bar = _mk("div", "cbar", foot);
    this.barCv = _mk("canvas", "cbarcv", this.bar);
    const tk = _mk("div", "cbartk", this.bar);
    this.barT = [_mk("span", null, tk), _mk("span", null, tk), _mk("span", null, tk)];
    this.bar.title = "colour range of the displayed quantity";
    // the two capture buttons travel as ONE flex item: on a narrow phone the footer wraps,
    // and a wrap that put `save` on one line and `rec` on the next would read as two
    // unrelated controls. The group shrinks for nobody; the caption and the colorbar wrap
    // around it as before. With no recording leg the hidden `rec` takes neither room nor
    // gap, so the group is just `save` (Alfred, 2026-08-12).
    const capg = _mk("div", "capgrp", foot);
    this.btnSave = _mk("button", "capbtn", capg);
    this.btnSave.innerHTML = "save";
    this.btnSave.title = "save this view (field + overlay + colorbar) as a PNG";
    this.btnRec = _mk("button", "capbtn", capg);
    this.btnRec.innerHTML = "rec";
    this.btnRec.title = "record the field canvas to an MP4 file (stops itself after 30 s)";
    if (!recAnySupported(this.cv)) this.btnRec.style.display = "none";

    this.gw = 0; this.gh = 0;
    this._resize();                       // sizes both canvases BEFORE the GPU context
    this.ctx = gpuCanvasCtx(this.cv);
    this.arr = null;                      // last arrow gather, and (3D) the field lines:
    this.lines = null;                    // two sources, one overlay canvas (see overlay())
    this.arrowAt = 0;
    this.wasLines = false;                // edge-trigger for the lines view's psi default
    // the overlay checkbox's two states, [slice/cube arrows, vol field lines]: arrows on
    // by default as they always were, lines over the volume OFF -- they composite in
    // FRONT of the shells (2D overlay canvas), so they are opt-in there. `wasVol` is the
    // edge trigger that swaps them.
    this.cbState = [true, false];
    this.wasVol = false;
    // colorbar state: the strip's context, the mode its labels belong to, the last
    // autoscale read back for it, and that readback's throttle clock
    this.barCx = chartCtx(this.barCv, CBAR_W, CBAR_H);
    this.barMode = -1; this.barMax = NaN; this.barAt = 0; this.barCmap = -1;
    this.recorder = new Recorder(this);   // the two capture legs, and their live handles
    this.resEl = { png: null, video: null };   // the waiting files' strips, one slot each
    this.dead = false;                    // set by destroy(), so a late probe cannot start
    // Render gate: a display chain is re-run only when its picture can have changed --
    // the solver stepped, this card's own controls moved, it was resized, or a capture is
    // taking its frames off this very render. Without it a PAUSED page re-runs every
    // chain at rAF rate forever: in 3D that is a full volume inverse transform per card
    // per frame (two more per active contour set), and in the volume view the whole
    // raymarch. `renderSeq` counts the frames actually drawn, so the readbacks that read
    // what a render LEFT behind (arrows, the colorbar's autoscale) can tell "nothing new
    // to fetch" from "throttled" and go quiet with it.
    this.dirty = true;
    this.seenMark = false;                // never equal to a stateMark(): draws once
    this.renderSeq = 0; this.arrowSeq = -1; this.barSeq = -1;

    const apply = () => { this.apply(); if (cards.cfg.onLayout) cards.cfg.onLayout(); };
    this.selField.onchange = apply;
    this.selCmap.onchange = apply;
    this.selCont.onchange = apply;
    this.selLev.onchange = apply;
    this.selBg.onchange = apply;
    this.cbArrow.onchange = apply;
    if (this.selZSrc) this.selZSrc.onchange = apply;
    if (this.rSlice) this.rSlice.oninput = apply;
    if (this.rLevel) { this.rLevel.oninput = apply; this.rOpac.oninput = apply; }
    this.rBLo.oninput = apply;
    this.btnClose.onclick = () => cardClose(this);
    this.btnSave.onclick = () => this.saveShot();
    this.btnRec.onclick = () => this.recToggle();
  }
  sel() { return parseInt(this.selField.value, 10) | 0; }
  cmap() { return parseInt(this.selCmap.value, 10) | 0; }
  // the PLANE source, view prefix stripped (the app's zsliceOf / trackingOn use this)
  zsrc() { return _zSrcPlane(this.selZSrc ? this.selZSrc.value : "manual"); }
  // ... and the three VIEWS the same select carries instead of the one plane: the cube
  // faces, (K2.1) the whole box's field lines with a transparent top face, or (ISO_PLAN B)
  // the volume raymarch -- the box itself, and the 3D app's default
  cubeView() { return !!this.selZSrc && this.selZSrc.value.indexOf("cube") === 0; }
  linesView() { return !!this.selZSrc && this.selZSrc.value === "lines"; }
  volView() { return !!this.selZSrc && this.selZSrc.value === "vol"; }
  level() { return this.rLevel ? parseFloat(this.rLevel.value) : VOL_LEVEL; }
  opac() { return this.rOpac ? parseFloat(this.rOpac.value) : VOL_OPAC; }
  // the card's k_perp band as prepDisp reads it: [k_lo, k_hi] in box wavenumbers, an end at
  // 0 meaning that end is OFF. The high end is permanently the dealias cut, which is that
  // end off (item 3), so the pair the page writes is [k_min, 0] -- and it is [0, 0], i.e.
  // arithmetic the kernel does not do at all, both at the slider's left stop and whenever
  // the panel's filter checkbox is unticked. k_min stops one shell below the cut: a filter
  // that cut everything would show an empty picture (uiFshell's rule, one handle short).
  band() {
    if (!bandFilterOn()) return [0, 0];
    const lo = Math.max(0, Math.min(parseInt(this.rBLo.value, 10) | 0, bandTop() - 1));
    this.rBLo.value = String(lo);
    return [lo, 0];
  }
  bandOn() { const b = this.band(); return b[0] > 0 || b[1] > 0; }
  // ... and, when it is on, the caption says so: a filtered picture must never be mistaken
  // for the field itself (wording: Alfred, feedback round 2 item 8)
  bandCap() {
    const b = this.band();
    if (!(b[0] > 0 || b[1] > 0)) return "";
    return " &mdash; <i>filter</i>: k&perp; = " + b[0] + ":k<sub>max</sub>";
  }
  // the field lines over the volume: the arrows checkbox, read in its vol state, and only
  // where a field is actually marched (the sigma modes fall back to the cube faces)
  volLinesOn() {
    return !!(this.volView() && this.cbArrow.checked &&
              solver && !dispIsSigma(solver.modeOf(this.ci)));
  }
  // the card's CONT_SETS contour potentials, as display modes (0 = that set is off)
  cont() {
    const v = this.selCont.value;
    return v === "both" ? [DISP_PSI, DISP_PHI] : [parseInt(v, 10) | 0, 0];
  }
  contOn() { return this.selCont.value !== "0"; }
  nlev() { return parseInt(this.selLev.value, 10) | 0; }
  plainBg() { return this.selBg.value === "1"; }
  // Aspect-correct card geometry (REFINE_PLAN J.2): ONE place decides the card's pixel
  // box, and the wrapper's CSS ratio, the WebGPU canvas backing store, the overlay canvas
  // and the arrow frame all follow it. Apps with square boxes supply no cfg.aspect and
  // get the historical 512x512 with no inline style at all.
  _resize() {
    const f = cards.cfg && cards.cfg.aspect, g = (f && f()) || null;
    const w = (g && g.w) || VEC_SIZE, h = (g && g.h) || VEC_SIZE;
    if (w === this.gw && h === this.gh) return;
    this.gw = w; this.gh = h;
    this.dirty = true;                    // a resized canvas has nothing drawn in it yet
    this.wrap.style.aspectRatio = (w === h) ? "" : (w + " / " + h);   // "" = the CSS 1/1
    this.cv.width = w; this.cv.height = h;
    this.vcx = vecCtx(this.cvVec, w, h);
  }
  // push this card's state into the live solver and relabel it
  apply() {
    if (!solver) return;
    const cfg = cards.cfg;
    this._resize();
    if (this.rSlice) this.rSlice.max = String(Math.max(0, cfg.nz() - 1));
    // the filter handle travels out to the live dealias cut; a card sitting at 0 stays off
    // when the grid (and with it the cut) moves under it
    const nb = bandTop();
    if (parseInt(this.rBLo.max, 10) !== nb) {
      this.rBLo.max = String(nb);
      this.rBLo.value = String(Math.min(parseInt(this.rBLo.value, 10), nb));
    }
    // ENTERING the lines view turns psi contours on: the transparent top face is the
    // point of the view (K2.3). From there the card's own contour select rules, "off"
    // included -- hence the edge trigger rather than a forced value.
    const lines = this.linesView();
    if (lines && !this.wasLines && !this.contOn()) this.selCont.value = String(DISP_PSI);
    this.wasLines = lines;
    // ... and the same edge trigger swaps the overlay checkbox between its two meanings
    // (arrows on a plane / field lines over the volume), each keeping its own last value
    const vol = this.volView();
    if (vol !== this.wasVol) {
      this.cbState[this.wasVol ? 1 : 0] = this.cbArrow.checked;
      this.cbArrow.checked = this.cbState[vol ? 1 : 0];
      this.wasVol = vol;
    }
    this.cbState[vol ? 1 : 0] = this.cbArrow.checked;
    this.cbTx.textContent = vol ? "lines" : "arrows";
    solver.setDisplayMode(this.ci, this.sel(), cfg.zsliceOf(this), this.cmap(),
                          { cube: this.cubeView(), lines: lines, vol: vol,
                            level: this.level(), opac: this.opac(), band: this.band(),
                            cont: this.cont(), nlev: this.nlev(),
                            plain: lines || (this.contOn() && this.plainBg()) });
    // the slider drives the displayed plane in the slice view and the TOP face in the
    // cube view, so it is live in both -- and dead whenever a tracker owns the plane, or
    // in either whole-box view (the lines face is the top BOUNDARY, K2.5; the volume has
    // no plane at all)
    if (this.rSlice) {
      this.rSlice.disabled = lines || vol || this.zsrc() !== "manual";
      // ... and a dead plane slider is REMOVED, not just greyed (item 5): a whole-box view
      // has no plane, so the control is not "unavailable", it is meaningless there. The
      // .disabled flag stays as it was for anything that reads it (the checks do).
      _rngShow(this.rSlice, !this.rSlice.disabled);
    }
    // ... but a sigma card in vol view fell back to the cube faces (setDisplayMode), so
    // level/opacity would march nothing and the checkbox would overlay nothing: hide the
    // sliders and grey the checkbox rather than show live-looking controls the card ignores
    const rvol = vol && !!solver && !dispIsSigma(solver.modeOf(this.ci));
    this.cbArrow.disabled = vol && !rvol;
    _rngShow(this.rLevel, rvol);
    _rngShow(this.rOpac, rvol);
    // the filter handle is there only when the panel asks for it (item 4)
    _rngShow(this.rBLo, bandFilterOn());
    // the contour overlay is an IN-PLANE object, so it goes away with the plane: the
    // volume view draws no plate for it and its potential pass is skipped (Solver.render)
    this.selCont.style.display = vol ? "none" : "";
    this.selLev.style.display = (this.contOn() && !vol) ? "" : "none";
    this.selBg.style.display = (this.contOn() && !lines && !vol) ? "" : "none";   // lines: always plain
    // the field selector is inert in the lines view (the lines are psi lines), so the
    // caption is the app's alone there. A field record may carry `d`, a one-line HTML
    // definition of the displayed quantity (FEEDBACK 2026-08-10 item 1) -- it rides the
    // caption line so it sits under the display, next to the app's own caption.
    const o = this.selField.options[this.selField.selectedIndex];
    const fr = !lines && cfg.fields.find(f => String(f.v) === this.selField.value);
    this.cap.innerHTML = (o && !lines ? o.innerHTML : "")
      + (fr && fr.d ? ": " + fr.d : "")
      + this.bandCap()
      + (cfg.caption ? cfg.caption(this) : "");
    this.barSync();                       // ... and so may have retired / relabelled the bar
    this.overlay();                       // the quantity / view may have retired an overlay
    this.dirty = true;                    // every control this reads can change the picture
  }
  // ---- colorbar (item 12) --------------------------------------------------
  // shown for anything that renders a field; the lines view renders none (its GPU canvas
  // carries contour ink over the plate and skips the whole mode chain), so it has no
  // colour range to legend and the bar goes away with the field selector's meaning.
  barOn() { return !this.linesView(); }
  // ... and it only needs the autoscale readback when the range IS the autoscale: the
  // sigma modes are pinned to +-1 in the kernel, and nothing renders while the editor
  // owns the screen.
  barNeedsMax() {
    return this.barOn() && !icDraw.on && !!solver && !dispIsSigma(solver.modeOf(this.ci));
  }
  setBarRange(s) { this.barMax = s; this.barLabel(); }
  // repaint the strip when the colormap changed, and drop a stale range when the MODE
  // did (|u| and psi do not share an autoscale) -- the next readback is <= CBAR_PERIOD away
  barSync() {
    this.bar.style.display = this.barOn() ? "" : "none";
    const m = solver ? solver.modeOf(this.ci) : this.sel();
    if (m !== this.barMode) { this.barMode = m; this.barMax = NaN; this.barAt = 0; }
    if (this.cmap() !== this.barCmap) { this.barCmap = this.cmap(); cbarPaint(this.barCx, this.barCmap); }
    this.barLabel();
  }
  // [left, middle, right] under the strip, in the convention the kernel renders:
  // sigma modes fixed +-1, magnitudes 0..max, everything else symmetric +-max
  barTicks() {
    const m = this.barMode, s = this.barMax;
    if (dispIsSigma(m)) return ["&minus;1", "0", "+1"];
    if (!(isFinite(s) && s > 0)) return ["", "", ""];
    if (dispIsVector(m)) return ["0", cbarFmt(0.5 * s), cbarFmt(s)];
    return ["&minus;" + cbarFmt(s), "0", "+" + cbarFmt(s)];
  }
  barLabel() {
    const t = this.barTicks();
    for (let i = 0; i < 3; i++) this.barT[i].innerHTML = t[i];
  }
  // ---- save / record (item 13) ---------------------------------------------
  // Re-render first: with WebGPU the canvas holds only its last PRESENTED image, so the
  // capture has to be taken in the same task as a fresh present -- there is no
  // preserveDrawingBuffer to ask for. Then one 2D composite of the three layers.
  saveShot() {
    this.render();
    const w = this.gw, h = this.gh;
    const cv = document.createElement("canvas");
    cv.width = w; cv.height = h;
    const c = cv.getContext("2d");
    if (!c || !cv.toBlob) return;
    c.drawImage(this.cv, 0, 0, w, h);                  // the field
    c.drawImage(this.cvVec, 0, 0, w, h);               // arrows / field lines / box frame
    if (this.barOn()) this.barStamp(c, w, h);
    // ... and the picture goes where a recording goes: onto the card, in its own slot, with
    // no length to quote (recResult). Nothing is downloaded until the visitor says so --
    // on a phone the silent download of a save is as hard to find as the silent download
    // of a take was.
    cv.toBlob(b => this.recResult("png", b, shotName(this.barMode, "png")), "image/png");
  }
  // the colorbar, scaled onto the saved image's bottom right over a translucent plate
  barStamp(c, w, h) {
    const sc = Math.max(1, w / 400), bw = CBAR_W * sc, bh = CBAR_H * sc;
    const px = 6 * sc, x = w - bw - 2 * px, y = h - bh - 3.4 * px;
    c.fillStyle = "rgba(20,22,26,0.72)";
    c.fillRect(x - px, y - px, bw + 2 * px, bh + 4.4 * px);
    c.drawImage(this.barCv, x, y, bw, bh);
    c.fillStyle = "#d8dee6";
    c.font = Math.round(9 * sc) + "px ui-monospace, SFMono-Regular, Menlo, monospace";
    c.textBaseline = "top";
    const t = this.barTicks().map(s => String(s).replace(/&minus;/g, "−"));
    const ty = y + bh + 0.5 * px;
    c.textAlign = "left";   c.fillText(t[0], x, ty);
    c.textAlign = "center"; c.fillText(t[1], x + bw / 2, ty);
    c.textAlign = "right";  c.fillText(t[2], x + bw, ty);
    c.textAlign = "left";
  }
  // The recorder's live state, read straight off the card as it always was: the frame
  // loop's REC_DEBUG line, needsRender(), destroy() and the devtools legs all ask the card
  // "are you recording?", and moving the machinery is not a reason to make them ask
  // something else. Read-only on purpose -- only the Recorder writes them.
  get wc() { return this.recorder.wc; }
  get rec() { return this.recorder.rec; }
  get recBusy() { return this.recorder.recBusy; }
  get recStop() { return this.recorder.recStop; }
  // ... and the two entry points its owner drives: the button and the frame loop's
  // per-frame feed. (destroy() hands over wholesale, below.)
  recToggle() { this.recorder.recToggle(); }
  recCapture() { this.recorder.recCapture(); }
  // button state in one place, so both legs and every stop route agree on it
  recLive() { this.btnRec.innerHTML = "stop"; this.btnRec.classList.add("reclive"); }
  recIdle() { this.btnRec.innerHTML = "rec"; this.btnRec.classList.remove("reclive"); }

  // The ONE place a finished file is handed over, whichever path made it -- the same
  // discipline the two stop paths already keep (see the note by recToggle). `seconds` is
  // each recording leg's own honest length: leg 1 counts the frames it actually MUXED
  // (dropped ones never made it into the file), leg 2 has no frame count of its own and
  // quotes wall clock; a PNG has no length at all and is quoted by size alone. Nothing is
  // downloaded here: the strip below is the whole point of the change.
  //
  // `kind` ("png" / "video") is the SLOT the strip lives in, and there are deliberately
  // two: a picture and a recording are two different files, so a save must replace only
  // the last save and a take only the last take. One slot would mean a 30 s take dying
  // because the visitor pressed save a second later -- the same file-losing surprise the
  // strip exists to remove (Alfred, 2026-08-12).
  recResult(kind, blob, name, seconds) {
    if (!blob || !blob.size) return;            // nothing was captured: say nothing
    // ... except on a card that is already gone. destroy() sets `dead` BEFORE leg 1's
    // async flush lands here -- and toBlob's callback is just as async, so a card can be
    // closed between the save press and the picture -- and a strip on a removed footer
    // would be a file the visitor can never reach. So a closed card keeps the old
    // behaviour and downloads on the spot: a surprise file is better than a lost one.
    if (this.dead || !this.foot) { dlBlob(blob, name); return; }
    this.recClear(kind);
    const s = _mk("div", "recres", this.foot);
    this.resEl[kind] = s;
    _mk("span", "recinfo", s).innerHTML = recSizeText(blob.size) +
      (seconds === undefined ? "" : " · " + recLenText(seconds));
    const dl = _mk("button", "capbtn", s);
    dl.innerHTML = "download";
    dl.title = "download " + name;
    dl.onclick = () => dlBlob(blob, name);
    // share is offered only where a file can really be shared (recShareFile); on a desktop
    // that cannot, the strip is just text and a download, which is what a desktop wants.
    const file = recShareFile(blob, name);
    if (file) {
      const sh = _mk("button", "capbtn", s);
      sh.innerHTML = "share";
      sh.title = "send " + name + " to another app";
      sh.onclick = () => {
        // AbortError is the visitor closing the sheet -- a decision, not a failure, so it
        // must not then push the file at them anyway. Anything else (no permission, an
        // engine that lied about canShare) falls back to the download rather than
        // swallowing the recording. share() is called SYNCHRONOUSLY in the click: an
        // engine with strict transient-activation rules (old iOS Safari -- the fallback
        // leg's own audience) can refuse a share deferred even one microtask, which
        // would turn every share press into the silent download this strip exists to
        // avoid. The try/catch folds a synchronous throw into the same rejection path
        // (adversarial review 2026-08-12, MINOR 2).
        let p;
        try { p = navigator.share({ files: [file] }); } catch (e) { p = Promise.reject(e); }
        Promise.resolve(p).catch(e => { if (!e || e.name !== "AbortError") dlBlob(blob, name); });
      };
    }
    const x = _mk("button", "capbtn recx", s);
    x.innerHTML = "&times;";
    x.title = "dismiss this " + (kind === "png" ? "picture" : "recording");
    x.onclick = () => this.recClear(kind);
  }
  // drop ONE kind's strip -- and with the node go the handlers, and with the handlers the
  // only references this card kept to the blob and the File, so the bytes can be
  // collected. The other slot is untouched: dismissing a picture must not take a
  // recording with it.
  recClear(kind) {
    const s = this.resEl[kind];
    this.resEl[kind] = null;
    if (s && s.parentNode) s.parentNode.removeChild(s);
  }

  showArrows() {
    return !!(this.cbArrow.checked && !this.linesView() && !this.volView() &&
              solver && dispIsVector(solver.modeOf(this.ci)));
  }
  // the cube view draws its arrows on the projected top face; a square plane view uses the
  // default frame (identical pixels to pre-J); a rectangular one needs the aspect-correct
  // anchor frame with an ISOTROPIC direction sub-frame (see drawArrows).
  arrowFrame() {
    const f = cards.cfg && cards.cfg.arrowXform;
    if (f && this.cubeView()) return f();
    if (this.gw === this.gh) return null;
    const s = Math.min(this.gw, this.gh);
    return { ox: 0, oy: 0, ax: this.gw, ay: 0, bx: 0, by: this.gh,
             d: { ox: 0, oy: 0, ax: s, ay: 0, bx: 0, by: s } };
  }
  render() {
    if (!this.ctx || !solver) return;
    solver.render(this.ctx, this.ci);
    this.renderSeq++;
  }
  // Does this card's chain have to run this frame? Two independent reasons, plus a
  // veto-free third:
  //   - the STATE moved. Asked as "is the mark I last drew still the current one", not as
  //     a flag somebody has to remember to set: a step, an IC upload, a preset and a
  //     rebuild all move stateMark() by construction, so no caller has to know about this
  //     gate at all. (It shipped as a flag the frame loop set after stepping, which worked
  //     but meant the only test of it was the test setting the flag itself -- adversarial
  //     review, 2026-08-12.)
  //   - this CARD moved: `dirty`, set by apply() and _resize(), i.e. by every control that
  //     feeds setDisplayMode, none of which touches the state.
  //   - a live take, which reads the texture THIS render produced (RECRAF); a skipped
  //     frame would hand the encoder an expired one.
  needsRender() {
    return this.dirty || this.seenMark !== stateMark() || !!this.wc || !!this.rec;
  }
  // The overlay canvas carries the arrow field AND (3D, lines view) the box frame and the
  // projected field lines, whose readbacks land at different rates -- so ONE method owns
  // the canvas: clear once, each source drawn from its last cached data. Called by both
  // setters and by apply(), which is what retires an overlay the new mode/view has no
  // business showing. The two are mutually exclusive since K2 (no arrows in the lines
  // view, no lines over the cube faces), but the traced lines stay cached either way.
  setArrows(a, nax, nay) { this.arr = a ? { a, nax, nay } : null; this.overlay(); }
  // cache always (so entering the lines view shows instantly via apply()'s overlay());
  // redraw only when this card actually displays lines -- the 2 Hz march must not force
  // an overlay repaint on every slice/cube card (reviewer, GATE K2)
  setLines(L) { this.lines = L; if (this.linesView() || this.volLinesOn()) this.overlay(); }
  overlay() {
    const c = this.vcx;
    if (!c) return;
    c.clearRect(0, 0, this.gw, this.gh);
    const X = cards.cfg && cards.cfg.lineXform;
    if (this.linesView() && X) {
      const F = X();                      // the box the lines live in; the GPU canvas
      drawBoxFrame(c, F);                 // under it carries only the top face's ink
      if (this.lines) drawFieldLines(c, this.lines, F);
    }
    // over the VOLUME (ISO_PLAN B) the same polylines, at reduced alpha and with no box
    // frame -- the raymarch draws its own wireframe, and these composite in FRONT of the
    // shells rather than through them (2D overlay canvas; the card's hint says so)
    if (this.volLinesOn() && X && this.lines) drawFieldLines(c, this.lines, X(), VOL_LINE_ALPHA);
    if (this.arr && this.showArrows()) drawArrows(c, this.arr.a, this.arr.nax, this.arr.nay, this.arrowFrame());
  }
  destroy() {
    this.dead = true;                     // set BEFORE the stop: recResult reads it
    this.recorder.destroy();              // a live take writes what it has -- see Recorder
    if (this.root.parentNode) this.root.parentNode.removeChild(this.root);
  }
}

class ChartCard {
  constructor(type) {
    const root = _mk("div", "card chart", cards.hostC);
    this.root = root;
    const head = _mk("div", "cardhead", root);
    this.head = head;
    this.selType = _sel(head, chartTypeKeys().map(k => ({ v: k, t: CHART_TYPES[k].label })),
                        "what this chart shows");
    this.selType.value = type;
    this.btnClose = _mk("button", "x", head);
    this.btnClose.innerHTML = "&times;";
    this.btnClose.title = "close this chart";
    this.cv = _mk("canvas", "chart", root);
    this.hint = _mk("div", "hint", root);
    this.cx = null;
    this.optEls = [];               // this type's option selects, rebuilt on retype
    this.build();
    // a retyped card must not wait out the old type's throttle window before it fills
    this.selType.onchange = () => {
      this.build(); this.draw(null);
      cardsThrottleReset();
    };
    this.btnClose.onclick = () => cardClose(this);
  }
  type() { return this.selType.value; }
  zsrc() { return _zSrcPlane(this.selZSrc ? this.selZSrc.value : "manual"); }
  // the option selects, as { id: value } -- what the type's draw() branches on.
  // Buttons (k: "btn") are actions, not values, so they contribute nothing; a checkbox
  // (k: "cbl") reads as "on" / "off", so every option value stays a plain string.
  optVals() {
    const o = {};
    for (const s of this.optEls) {
      if (s.__optChk) o[s.__optId] = s.__optChk.checked ? "on" : "off";
      else if (!s.__optBtn) o[s.__optId] = s.value;
    }
    return o;
  }
  // the inverse, for a preset's chart spec (addChartCard): write the option values it
  // names and leave the rest at their declared defaults. An id this card's type does not
  // own is IGNORED, exactly as presetWrite ignores an id the page does not build -- so one
  // layout can name `clip` for the spectrum card without every other chart type caring.
  // Buttons carry no value and are skipped.
  optSet(v) {
    if (!v) return;
    let hit = false;
    for (const s of this.optEls) {
      const x = v[s.__optId];
      if (x === undefined || s.__optBtn) continue;
      if (s.__optChk) s.__optChk.checked = (x === "on" || x === true);
      else s.value = String(x);
      hit = true;
    }
    if (hit) this._optSync();
  }
  // show/hide the options whose meaning depends on another one (the fit line's boxes),
  // and enable/disable the ones that depend on the card's STATE (the pin button, which
  // has nothing to snapshot before the card has been handed data). Both hooks see the
  // card, so a button's meaning can depend on more than the other options' values.
  _optSync() {
    const v = this.optVals();
    for (const s of this.optEls) {
      if (s.__optVis) s.style.display = s.__optVis(v, this) ? "" : "none";
      if (s.__optDis) s.disabled = !!s.__optDis(v, this);
    }
  }
  build() {
    const T = CHART_TYPES[this.type()];
    // drop the previous type's controls (a select is appended AFTER the close button,
    // so the button is re-appended last to keep its margin-left:auto place)
    for (const s of this.optEls) this.head.removeChild(s);
    if (this.rSlice) { this.head.removeChild(this.selZSrc); this.head.removeChild(this.rSlice); }
    this.optEls = []; this.selZSrc = null; this.rSlice = null;
    // retyping the card is what drops its spectrum state (PINCURVE): the pins belonged
    // to a chart this card no longer is, and the cache belonged to that chart's data.
    this.pins = []; this.lastData = null;
    const redraw = d => { this._optSync(); this.draw(d || null); cardsThrottleReset(); };
    // an option is a <select> over `o`, (k: "num") a small number box, or (k: "btn") a
    // small header BUTTON -- all three end up in optEls with the same __optId, so
    // optVals() and the type's draw() see one shape. A button carries no value: it calls
    // `onClick(card)` and the card redraws from its own last data (a press must not blank
    // the live curves for the rest of the spectrum throttle window).
    // `vis(vals, card)` optionally hides one when another option or the card's state makes
    // it meaningless; `dis(vals, card)` disables it instead.
    for (const spec of (T.opts ? T.opts(cards.cfg || {}) : [])) {
      let s;
      if (spec.k === "num") {
        s = _mk("input", "optnum", this.head);
        s.type = "number";
        if (spec.min !== undefined) s.min = String(spec.min);
        if (spec.step !== undefined) s.step = String(spec.step);
        if (spec.w) s.style.width = spec.w + "px";
        if (spec.ti) s.title = spec.ti;
        s.oninput = () => redraw();
      } else if (spec.k === "btn") {
        s = _mk("button", "optbtn", this.head);
        s.innerHTML = spec.t;
        if (spec.ti) s.title = spec.ti;
        s.__optBtn = true;
        s.onclick = () => { spec.onClick(this); redraw(this.lastData); };
      } else if (spec.k === "cbl") {
        // a header CHECKBOX, in the `.cbl` idiom the control panel and the display card's
        // arrows box already use (label wrapping box + text, so it is ONE flex item and
        // the layout audit measures it as one). Its value is its checked state, so it
        // rides optVals like any other option and the type's draw() sees "on" / "off".
        s = _mk("label", "cbl", this.head);
        const box = _mk("input", null, s);
        box.type = "checkbox"; box.checked = !!spec.v;
        s.appendChild(document.createTextNode(spec.t));
        if (spec.ti) s.title = spec.ti;
        s.__optChk = box;
        box.onchange = () => redraw(this.lastData);
      } else {
        s = _sel(this.head, spec.o.map(x => ({ v: x[0], t: x[1] })), spec.ti);
        s.onchange = () => redraw();
      }
      if (spec.v !== undefined && !s.__optChk) s.value = String(spec.v);
      s.__optId = spec.id;
      s.__optVis = spec.vis || null;
      s.__optDis = spec.dis || null;
      this.optEls.push(s);
    }
    this._optSync();
    if (T.zslice) {
      _zSliceControls(this, this.head);
      if (this.selZSrc) {
        this.selZSrc.onchange = () => { this.apply(); redraw(); };
        this.rSlice.oninput = () => { this.apply(); redraw(); };
      }
    }
    this.head.removeChild(this.btnClose); this.head.appendChild(this.btnClose);
    this.cv.style.aspectRatio = T.w + " / " + T.h;
    this.cx = chartCtx(this.cv, T.w, T.h);
    this._hintSync(T);
    this._barBuild(T);
  }
  // A type's `hint` is fixed copy for every chart but one: gen2d's names the colour scale
  // its cells are actually painted on, so it is a FUNCTION of the card's options and has to
  // be re-rendered whenever those change -- which is here (retyping) and in draw() (an
  // option select, a generate press). The plain string form is untouched and, being
  // constant, is written once per retype exactly as it always was.
  _hintSync(T) {
    this.hint.innerHTML = typeof T.hint === "function" ? T.hint(this.optVals()) : T.hint;
  }
  // A chart whose quantity is a COLOUR needs the same legend a display card's field does,
  // so the colorbar is the display card's block verbatim -- `.viewfoot` > `.cbar` (strip +
  // three ticks), painted by cbarPaint through the shared colormap table and labelled by
  // cbarFmt. Only types that declare `bar(opts)` get one; retyping the card takes it away
  // again, which is why it is built here and not in the constructor.
  _barBuild(T) {
    if (this.foot) {
      this.root.removeChild(this.foot); this.foot = null; this.barT = null; this.barD = null;
    }
    if (!T.bar) return;
    // appended, then the hint is re-appended after it -- the same "move it back to last"
    // idiom the close button uses, so the card reads canvas / colorbar / hint
    this.foot = _mk("div", "viewfoot", this.root);
    this.root.removeChild(this.hint); this.root.appendChild(this.hint);
    // the bar div is kept because a type whose colour SCALE is an option (gen2d's `gc`)
    // re-titles it per draw, through `barTi(opts)`; without one the static log wording
    // below stands, which is what every other bar-carrying type means
    const bar = this.barD = _mk("div", "cbar", this.foot);
    bar.title = "colour range of the plotted quantity (log scale, so the middle tick is "
      + "the geometric mean)";
    const bcv = _mk("canvas", "cbarcv", bar);
    const tk = _mk("div", "cbartk", bar);
    this.barT = [_mk("span", null, tk), _mk("span", null, tk), _mk("span", null, tk)];
    cbarPaint(chartCtx(bcv, CBAR_W, CBAR_H), T.cmap || 0);
  }
  // keep the z slider in range / enabled only when this card picks its plane by hand
  apply() {
    if (!this.rSlice) return;
    this.rSlice.max = String(Math.max(0, cards.cfg.nz() - 1));
    this.rSlice.disabled = this.zsrc() !== "manual";
  }
  // The last non-null data this card was handed, kept for the spectrum type alone: it is
  // what `pin` snapshots. It deliberately OUTLIVES chartsReset (which draws every card
  // with null) -- an IC reset is exactly the moment a pin is about to be taken.
  draw(data) {
    if (data && this.type() === "spectrum") {
      const first = !this.lastData;
      this.lastData = data;
      // the pin button is dead until the card HAS something to snapshot, and the frame
      // loop's readback is what ends that -- so the first one re-runs the opt sync. Only
      // the first: nothing about the buttons changes on the ones after it.
      if (first) this._optSync();
    }
    const T = CHART_TYPES[this.type()];
    const o = this.optVals();
    T.draw(this.cx, data, o, this.pins);
    // an options-dependent hint follows the options it depends on (only gen2d has one, and
    // gen2d gets no frame-loop draw at all -- this runs on a press or a select, not at 60 Hz)
    if (typeof T.hint === "function") this._hintSync(T);
    // the colorbar's labels come off the SAME options the plot was just drawn from, and
    // through the type's own hook -- the card knows it has a bar, never what is on it
    if (this.barT && T.bar) {
      const t = T.bar(o);
      for (let i = 0; i < 3; i++) this.barT[i].innerHTML = t[i];
      if (this.barD && T.barTi) this.barD.title = T.barTi(o);
    }
  }
  // ---- pinned ghost spectra (PINCURVE Phase B) -------------------------------
  // A pin is a snapshot of the curves this card is DRAWING -- post specSeries /
  // parKfac -- so it is immune to later changes of the card's own sq / sd selectors:
  // a pin taken as "E+ / E-" stays E+/E- however the live view is switched. `kunit`
  // rides along so the ghost can be re-registered on physical k if the box changes,
  // and `t` is the simulation time the legend reports.
  pinAdd() {
    if (this.type() !== "spectrum" || !this.lastData) return;
    if (this.pins.length >= PIN_MAX) {
      showStatus("at most " + PIN_MAX + " pinned spectra — unpin first", "info");
      return;
    }
    const s = specCurves(this.lastData, this.optVals());
    // an all-zero spectrum (quiescent IC before the first injection) would pin an
    // invisible ghost that still burns a cap slot and a legend entry — refuse it
    // (reviewer NOTE, 2026-08-09)
    if (!(s.hi > 0)) { showStatus("nothing to pin yet — spectrum is empty", "info"); return; }
    this.pins.push({
      // deep copy to plain Arrays: the live bins are a reused Float32Array
      curves: s.curves.map(cv => [Array.from(cv[0]), cv[1], cv[2] ? cv[2].slice() : null, cv[3]]),
      nPerp: s.nPerp, hiP: s.hiP, loP: s.loP, hi: s.hi, lo: s.lo,
      t: hist.t.length ? hist.t[hist.t.length - 1] : 0,
      kunit: this.lastData.kunit || 0
    });
  }
  pinClear() { this.pins.length = 0; }
  destroy() { if (this.root.parentNode) this.root.parentNode.removeChild(this.root); }
}

// ---- registry operations ---------------------------------------------------
function cardsInit(cfg) {
  cards.cfg = cfg;
  cards.hostD = el("displays");
  cards.hostC = el("charts");
  el("btnAddDisp").onclick = () => { addDisplayCard(); cardsSync(); };
  el("btnAddChart").onclick = () => { addChartCard("energy"); cardsSync(); };
}
// the lowest free chain index (a closed card frees its slot for the next one)
function _freeCi() {
  for (let i = 0; i < CARD_MAX_DISP; i++) {
    if (!cards.disp.some(d => d.ci === i)) return i;
  }
  return -1;
}
function addDisplayCard(state) {
  const ci = _freeCi();
  if (ci < 0) { showStatus("at most " + CARD_MAX_DISP + " display cards", "info"); return null; }
  const c = new DisplayCard(ci);
  cards.disp.push(c);
  cards.disp.sort((a, b) => a.ci - b.ci);
  if (state) {
    if (state.sel !== undefined) c.selField.value = String(state.sel);
    if (state.cmap !== undefined) c.selCmap.value = String(state.cmap);
    if (state.arrows !== undefined) c.cbArrow.checked = !!state.arrows;
    if (state.zsrc !== undefined && c.selZSrc) c.selZSrc.value = state.zsrc;
    if (state.zslice !== undefined && c.rSlice) c.rSlice.value = String(state.zslice);
    if (state.level !== undefined && c.rLevel) c.rLevel.value = String(state.level);
    if (state.opac !== undefined && c.rOpac) c.rOpac.value = String(state.opac);
    // the k_perp filter, as a preset may ask for it. It is one handle since item 3, but a
    // stored [k_lo, k_hi] pair still reads: the high end is the cut now, so only k_lo is
    // taken and an old pair loads silently rather than erroring.
    if (state.band !== undefined) c.rBLo.value = String(state.band[0]);
    if (state.cont !== undefined) c.selCont.value = String(state.cont);
    if (state.nlev !== undefined) c.selLev.value = String(state.nlev);
    if (state.plain !== undefined) c.selBg.value = state.plain ? "1" : "0";
  }
  return c;
}
// A preset's `charts` entry is either a bare type string, as it has always been, or
// {t: "<type>", <option id>: <value>} -- which is what `disp` entries have always been
// ({sel, cont, nlev}), so this is the two halves of a layout finally agreeing rather than
// a new concept. It exists because the spectrum card's `clip` default is per PRESET and
// not global: the hyper = 1 presets need the tail visible, everything else needs it
// clipped, and a viewer should not have to know that.
function addChartCard(spec) {
  const s = (spec && typeof spec === "object") ? spec : { t: spec };
  const c = new ChartCard(chartTypeKeys().indexOf(s.t) >= 0 ? s.t : "energy");
  cards.chart.push(c);
  c.optSet(s);
  return c;
}
function cardClose(c) {
  const list = (c instanceof DisplayCard) ? cards.disp : cards.chart;
  const i = list.indexOf(c);
  if (i < 0) return;
  if (list === cards.disp && cards.disp.length <= CARD_MIN_DISP) {
    showStatus("at least " + CARD_MIN_DISP + " display card", "info");
    return;
  }
  list.splice(i, 1);
  c.destroy();
  cardsSync();
}
// re-push every card into the solver (after a rebuild, a preset, or add/remove)
function cardsSync() {
  // the last display card cannot be closed -- shown, not just enforced
  for (const d of cards.disp) d.btnClose.disabled = cards.disp.length <= CARD_MIN_DISP;
  for (const d of cards.disp) d.apply();
  for (const c of cards.chart) { c.apply(); if (c.type() === "cut") c.draw(null); }
  cardsThrottleReset();
  if (cards.cfg && cards.cfg.onLayout) cards.cfg.onLayout();
}
// replace the whole layout (used by the presets and by boot)
function cardsLayout(L) {
  // A preset switch rebuilds every card, and that must NOT silently eat the pinned
  // ghosts: "pin the decayed spectrum, pick the forced preset, compare" is the whole
  // universality lesson (PINCURVE Phase C). So the spectrum cards' pins and their
  // last-data cache are carried across POSITIONALLY -- outgoing spectrum cards in DOM
  // order to incoming ones in DOM order, extras dropped, a shortfall simply unfilled.
  // No identity matching: the cards are new objects and the user's mental model is
  // "the spectrum chart", not "that particular card".
  const carry = cards.chart.filter(c => c.type() === "spectrum")
                           .map(c => ({ pins: c.pins, lastData: c.lastData }));
  for (const d of cards.disp.slice()) { d.destroy(); }
  for (const c of cards.chart.slice()) { c.destroy(); }
  cards.disp.length = 0; cards.chart.length = 0;
  for (const s of (L && L.disp) || [{}]) addDisplayCard(s);
  while (cards.disp.length < CARD_MIN_DISP) addDisplayCard();
  for (const t of (L && L.charts) || ["energy", "spectrum", "cut"]) addChartCard(t);
  const inc = cards.chart.filter(c => c.type() === "spectrum");
  for (let i = 0; i < inc.length && i < carry.length; i++) {
    inc[i].pins = carry[i].pins;
    inc[i].lastData = carry[i].lastData;
    // draw the ghosts NOW (not the transplanted live data -- the run it came from is
    // gone); the next readback fills the live curves back in
    inc[i]._optSync();
    inc[i].draw(null);
  }
  cardsSync();
}
// the card the single-instance overlays (IC editor, cut trace) hang off
function primaryCard() { return cards.disp.length ? cards.disp[0] : null; }
// clear the traces after an IC change / rebuild (one call, both apps). The option sync
// rides along because a rebuild can change what a card's BUTTONS mean -- the generated
// E(k⊥,k∥) card's `generate` needs a live solver, and a card built before there was one
// (the no-WebGPU boot) would otherwise stay disabled for the session. It is idempotent
// for every other card: the fit boxes and the pin buttons re-evaluate to what they were.
function chartsReset() {
  histReset(); islandReset(); modeReset();
  cardsThrottleReset();
  // ... and it is also where the state JUMPED without a step being taken, which is the
  // one thing the render gate's step counter cannot see (an IC upload resets nsteps to
  // 0, so the count alone can repeat a value the caches were already holding).
  stateBumped();
  for (const c of cards.chart) { c._optSync(); c.draw(null); }
}

// ===========================================================================
// the control panel, built from a SPEC (REFINE_PLAN H.0)
// ===========================================================================
// The two pages carried byte-identical markup for the sticky top bar, the whole
// forcing group and most of the sim / dissipation / IC / displays rows. That is a
// copy-paste variant of exactly the kind the standing rule forbids, so the markup
// is DATA now: each app hands controlsBuild() a spec, and every row the two pages
// share is a spec FRAGMENT below -- written once.
//
// A spec is { topbar: [item...], groups: [group...] }; a group is
// { id, summary, keep, rows: [row...] }; a row is either an array of items, an
// object { id, hide, items }, or { k: "hintdiv", id } for a bare hint line.
// Item kinds (every item may carry `ti`, the title attribute):
//   lab  <label>t</label> (for: the id it labels)      val  <span class="val" id>
//   sel  <select id> from o: [[value, html], ...]      txt  <span id>t</span>
//   rng  <input type=range> min/max/step/v             hint <span class="hint" id>
//   num  <input type=number> v, w (px)                 btn  <button id>t</button>
//   cb   bare checkbox (v = checked)
//   cbl  checkbox inside <label class="cbl">t</label> (v = checked)
// `t` is HTML (the entities the markup used) EXCEPT for cbl, whose t is a text node --
// give it literal characters (⊥ not &perp;), or the entity shows verbatim; `ti` is
// plain text.
function _ctrlItem(row, it) {
  const put = (tag, cls) => _mk(tag, cls, row);
  let e;
  if (it.k === "sel") {
    e = _sel(row, it.o.map(o => ({ v: o[0], t: o[1] })), it.ti);
    if (it.v !== undefined) e.value = String(it.v);
  } else if (it.k === "rng" || it.k === "num") {
    e = put("input");
    e.type = it.k === "rng" ? "range" : "number";
    if (it.min !== undefined) e.min = String(it.min);
    if (it.max !== undefined) e.max = String(it.max);
    if (it.step !== undefined) e.step = String(it.step);
    if (it.v !== undefined) e.value = String(it.v);
    if (it.w) e.style.width = it.w + "px";
  } else if (it.k === "cb" || it.k === "cbl") {
    const host = it.k === "cbl" ? put("label", "cbl") : row;
    e = _mk("input", null, host);
    e.type = "checkbox";
    e.checked = !!it.v;
    if (it.k === "cbl") host.appendChild(document.createTextNode(it.t));
  } else if (it.k === "btn") {
    e = put("button"); e.innerHTML = it.t;
  } else if (it.k === "lab") {
    e = put("label"); e.innerHTML = it.t;
    if (it.for) e.htmlFor = it.for;
  } else if (it.k === "val") {
    e = put("span", "val");
  } else if (it.k === "hint") {
    e = put("span", "hint"); if (it.t) e.innerHTML = it.t;
  } else {
    e = put("span"); if (it.t) e.innerHTML = it.t;
  }
  if (it.id) e.id = it.id;
  if (it.ti) e.title = it.ti;
  if (it.hide) e.style.display = "none";
  return e;
}
// A NAME AND THE THING IT NAMES ARE ONE FLEX ITEM (2026-08-13 feedback item 4).
// .row wraps freely, which on a phone can leave "diss" at the end of one line and its
// slider at the start of the next -- two orphans that read as unrelated controls. The
// cure is the one .cardhead already uses for its header sliders (label.rngl): put the
// pair in an inline-flex box, so the ROW's flex item is the group and a wrap can only
// ever happen between groups.
//
// The grouping is derived, not declared, so no row spec had to be rewritten and a new
// row gets it for free: a `lab` opens a group, which then absorbs the controls that
// follow it while they are things a label can name -- sliders, number boxes, a bare
// checkbox, and the readout span that belongs to the slider. Anything else (a button, a
// `cbl` with its own built-in label, a hint, the next `lab`) closes it. That is what
// keeps "band n" holding BOTH of its handles and their single readout.
//
// A <select> is DELIBERATELY not groupable. It was, and devtools/layout.js caught what
// that costs: an option list is arbitrarily long text, so "preset" + #selIC became a 361px
// item that cannot wrap internally, against 316px of usable width at 360px -- the 3D
// page's "sinusoidal z+- packets (exact interaction)". A select is a bordered box that
// already reads as one control next to its name, and the complaint this fixes was about
// sliders, whose name is the only thing telling two identical grey tracks apart.
const _CTRL_GROUPABLE = { rng: 1, num: 1, cb: 1, val: 1 };
function _ctrlRow(row, items) {
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    // how many of the items after this label it can take with it
    let n = 0;
    if (it.k === "lab") while (_CTRL_GROUPABLE[(items[i + 1 + n] || {}).k]) n++;
    if (n === 0) { _ctrlItem(row, it); continue; }
    const g = _mk("span", "ctlg", row);
    for (let j = 0; j <= n; j++) _ctrlItem(g, items[i + j]);
    i += n;
  }
}
function controlsBuild(spec) {
  const bar = el("topbar");
  _ctrlRow(bar, spec.topbar || CTRL_TOPBAR);
  const host = el("controls");
  for (const g of spec.groups) {
    const d = _mk("details", null, host);
    d.id = g.id;
    if (g.keep) d.setAttribute("data-keep-open", "");
    _mk("summary", null, d).innerHTML = g.summary;
    for (const r of g.rows) {
      if (r.k === "hintdiv") { const h = _mk("div", "hint", d); h.id = r.id; continue; }
      const row = _mk("div", "row", d);
      if (r.id) row.id = r.id;
      if (r.hide) row.style.display = "none";
      _ctrlRow(row, Array.isArray(r) ? r : r.items);
    }
  }
}

// ---- the rows both pages share --------------------------------------------
const CTRL_TOPBAR = [
  { k: "btn", id: "btnRun", t: "Run" },
  { k: "btn", id: "btnReset", t: "Reset" },
  { k: "lab", t: "preset", for: "selPreset" },
  { k: "sel", id: "selPreset", o: [] },
  { k: "btn", id: "btnParams", t: "show params" },
  { k: "btn", id: "btnText", t: "hide text" },
  { k: "txt", id: "steps" }
];
const CTRL_SEED = [{ k: "lab", t: "seed" }, { k: "num", id: "nSeed", v: 7, w: 70 }];
const CTRL_CFL = [
  { k: "lab", t: "cfl_safety" },
  { k: "rng", id: "rCfl", min: 0.05, max: 0.9, step: 0.01, v: 0.4 }, { k: "val", id: "vCfl" },
  { k: "lab", t: "cfl_every" },
  { k: "rng", id: "rCflEvery", min: 1, max: 16, step: 1, v: 4 }, { k: "val", id: "vCflEvery" }
];
// the hyper / diss row: the slider's default differs between the pages, and only 2D
// offers Pm (REFINE_PLAN J.1, J2.6 -- the 2D propagator is diagonal per field, so nu and
// eta can differ; the 3D 2x2 Alfven propagator needs an equal diagonal).
const ctrlDissRow = (dflt, o) => [
  { k: "lab", t: "hyper" },
  { k: "sel", id: "selHyper", o: [[1, "1"], [2, "2"], [3, "3"], [4, "4"]], v: 4 },
  { k: "lab", t: "diss" },
  // the range is DYNAMIC (FEEDBACK_2026-08-08 item 7): dissRangeSync() rewrites min/max
  // from the live hyper / resolution / box, so the two numbers here are only what the
  // element carries between construction and the first syncLabels. The STEP is the one
  // fixed thing -- every value written to this slider is a multiple of it (see
  // DISS_STEP), which is what keeps a range update from snapping a preset's value.
  { k: "rng", id: "rDiss", min: -20, max: -1, step: DISS_STEP, v: dflt }, { k: "val", id: "vDiss" },
  { k: "cbl", id: "cbAutoDiss", t: "auto", v: true,
    ti: "drive diss continuously from the measured field amplitude near the dissipation "
      + "scale; untick to take the slider back where the controller left it" }
].concat((o && o.pm) ? [
  { k: "lab", t: "Pm" },
  { k: "num", id: "nPm", v: 1, w: 62, min: 0,
    ti: "magnetic Prandtl number nu/eta: the diss slider is eta (it multiplies psi) and "
      + "nu = Pm*eta multiplies phi. 1 is the scalar dissipation every other path uses, "
      + "0 is an inviscid phi; changing it rebuilds the solver." }
] : []);
// the forcing group is IDENTICAL on both pages (this block was the GATE-G MAJOR)
const CTRL_GRP_FORCE = {
  id: "grpForce", summary: "forcing", rows: [
    [{ k: "cb", id: "cbForce", v: true, ti: "forcing on/off (unchecked: decaying turbulence)" },
     { k: "lab", t: "&epsilon;&#8314;" },
     { k: "rng", id: "rEpsP", min: -3, max: 1, step: 0.05, v: -0.82 }, { k: "val", id: "vEpsP" },
     { k: "cbl", id: "cbEpsLock", t: "lock", v: true, ti: "move the two injection rates together" }],
    [{ k: "lab", t: "&epsilon;&#8315;" },
     { k: "rng", id: "rEpsM", min: -3, max: 1, step: 0.05, v: -0.82 }, { k: "val", id: "vEpsM" },
     { k: "lab", t: "tau" },
     { k: "rng", id: "rTau", min: 0.05, max: 5, step: 0.05, v: 1 }, { k: "val", id: "vTau" }],
    [{ k: "lab", t: "band n" },
     { k: "rng", id: "rFmin", min: 1, max: 8, step: 1, v: 1,
       ti: "forcing shell: lower edge, in units of the box wavenumber" },
     { k: "rng", id: "rFmax", min: 2, max: 12, step: 1, v: 3,
       ti: "forcing shell: upper edge (exclusive)" }, { k: "val", id: "vFshell" }]
  ]
};
// the IC group: the paint row and the two amplitude rows are shared. Three slots for the
// per-page rows, because ORDER now carries meaning (2026-08-13 feedback items 2 + 3):
//   pre    before the paint row -- 2D's equilibrium knobs, which belong to presets that
//          never show the paint row at all
//   post   between the paint row and the amplitudes -- 3D's sigma_z, which sits with
//          sigma_perp because the two are the same knob on two axes
//   tail   last -- the derived readout line (3D's chi / packet hint)
// The amplitudes come AFTER the paint selector because the selector is what DECIDES what
// they mean: icAmpBasis reads #selPaint, and ampBasisSync relabels the two sliders from
// it, so cause now reads above effect instead of below it.
const ctrlGrpIC = o => ({
  id: "grpIC", summary: "initial condition", rows: [
    [{ k: "lab", t: "preset" },
     { k: "sel", id: "selIC", o: [["modes", "large-scale modes"], ["letters", o.letters],
                                  ["custom", "custom (drawn blobs)"], ["quiescent", "quiescent (zero)"]]
                                 .concat(o.presets || []) }]
  ].concat(o.pre || [], [
    { id: "rowDraw", hide: true, items: [
      { k: "btn", id: "btnEdit", t: "edit IC", ti: "pause the run and open the IC editor" },
      { k: "lab", t: "paint" },
      // ORDER (FEEDBACK_2026-08-10 item 14): the evolved variables first, the Elsasser
      // potentials after. Nothing indexes these options positionally -- icEditCaption
      // reads options[selectedIndex] and IC_TARGETS is keyed by the VALUE.
      { k: "sel", id: "selPaint", o: [["phi", "&phi;"], ["psi", "&psi;"],
                                      ["zp", "&zeta;&#8314;"], ["zm", "&zeta;&#8315;"]],
        ti: "which field a stroke deposits into; phi and psi are painted through "
          + "zeta+- = phi +- psi, and they make the two amp sliders phi / psi knobs" },
      { k: "lab", t: "&sigma;&perp;" },
      { k: "rng", id: "rSigP" }, { k: "val", id: "vSigP" },
      { k: "lab", t: "negative" },
      { k: "cb", id: "cbNeg", ti: "deposit with a minus sign (or drag with the right button)" }
    ] }
  ], o.post || [], [
    // the two amplitude labels and titles are REWRITTEN by ampBasisSync (item 15): while a
    // drawing is being painted in phi/psi they are the (phi, psi) knobs, not the zeta+- ones.
    // Both rows are HIDDEN, not disabled, for the presets that set their amplitudes some
    // other way (2026-08-13 feedback item 5) -- see icSyncRows.
    { id: "rowAmpP", items: [
      { k: "lab", id: "labAmpP", t: "&zeta;&#8314; amp" },
      { k: "rng", id: "rAmpP", min: -2, max: 1, step: 0.05, v: o.amp }, { k: "val", id: "vAmpP" },
      { k: "cbl", id: "cbAmpLock", t: "lock", v: true, ti: "move the two potential amplitudes together" }
    ] },
    { id: "rowAmpM", items: [
      { k: "lab", id: "labAmpM", t: "&zeta;&#8315; amp" },
      { k: "rng", id: "rAmpM", min: -2, max: 1, step: 0.05, v: o.amp }, { k: "val", id: "vAmpM" }
    ] }
  ], o.tail || [])
});
// displays & charts: the two add buttons, plus whatever page-wide extras follow
const ctrlGrpDisp = extra => ({
  id: "grpDisp", summary: "displays &amp; charts", keep: true, rows: [
    // the two add buttons, the page-wide display toggles, then whatever the page adds.
    // The k_perp filter's handle is opt-in and OFF here (item 4): the panel is where the
    // display-wide switches live, and the hint that used to explain the two band sliders
    // went with the second slider (its content lives in docs.html's scale-filter section).
    [{ k: "btn", id: "btnAddDisp", t: "+ display" },
     { k: "btn", id: "btnAddChart", t: "+ chart" },
     { k: "cbl", id: "cbFilter", t: "k⊥ filter", v: false,
       ti: "give every display card a k_perp filter slider: it hides structure below the "
         + "wavenumber you set, in the PICTURE only -- the run, the spectra and the field "
         + "lines are never filtered" }].concat(extra || [])
  ]
});

// ===========================================================================
// page chrome: the 2D/3D tabs, the "what is all this?" rail, the no-GPU poster
// ===========================================================================
// ONEPAGE_PLAN B. The two apps stay separate documents (separate pipelines), so the
// only thing that can make them feel like one site is chrome built from ONE source --
// the controlsBuild bargain again, applied to the markup above and beside the canvas.
// A visitor cannot tell a styled link from a tab, so "2D | 3D" is a strip in which the
// current page is a plain span and the other one an ordinary <a>.
const TABS = [{ t: "2D", href: "rmhd2d.html", is3d: false },
              { t: "3D", href: "rmhd3d.html", is3d: true }];
// The essay index.html used to carry MOVED here rather than going in the bin (the front
// door is now rmhd2d.html, and an essay nobody reaches is worse than one beside the
// picture it describes): two sentences of lead, then the five panes, all collapsed.
// Alfred's original index.html lead, VERBATIM (review follow-up: a condensed rewrite
// was worse) -- two paragraphs, the body of the #intro pane under the subtitle
const RAIL_LEAD = [
  `this is a browser port of the jax plasma turbulence code
  <a href="https://github.com/alfredmallet/taranis" target="_blank" rel="noopener">taranis</a>,
  running on your gpu and using the same numerical algorithms as the parent code. Reduced
  MagnetoHydroDynamics (RMHD) is the equation set solved here: a simplified model for the
  behavior of a plasma when there is a strong magnetic field present (as is the case in our
  solar system). it is useful because it encodes the physics of Alfv&eacute;n waves traveling
  along magnetic field lines, as well as complex nonlinear interactions between the waves. it
  does a surprisingly good job at describing plasma turbulence in a wide range of systems.`,
  `we hope you can use these solvers to get a feel for how this turbulence
  works. you can change parameters freely, and explore various setups: forced and decaying
  turbulence, collisions between waves, and plasma instabilities. both two-dimensional and
  three-dimensional solvers are provided. for more details on what this all means, please
  click on some of the demonstrations. have fun!`
];
const RAIL_PANES = [
  { summary: "what is turbulence?", html: `
    <blockquote>&ldquo;When I meet God, I am going to ask him two questions: Why relativity?
      And why turbulence? I really believe he will have an answer for the first.&rdquo;<br>
      <span class="attrib">&mdash; possibly apocryphal, attributed to Heisenberg, among
      others&hellip;</span></blockquote>
    <p>perhaps the greatest unsolved mystery of classical physics. a basic, concise working
      definition: multiscale disorder. introducing energy at large scales in the system
      causes a chaotic, nonlinear cascade down to small scales at which dissipation can
      occur.</p>` },
  { summary: "what is plasma?", html: `
    <p>with enough external energy, electrons in atoms can be stripped away from their
      central nuclei, forming a (usually) overall electrically neutral system of
      negatively-charged electrons and positively-charged ions. these particles interact in
      complex ways, resulting in phenomena like plasma turbulence.</p>` },
  { summary: "why should anyone care?", html: `
    <p>most of the universe is full of turbulent plasma: it is hard to find an astrophysics
      problem where understanding this turbulence is not crucially important. here are three
      examples:</p>
    <ol>
      <li>the sun is constantly emitting plasma, which is heated (up to temperatures of
        millions of kelvin) and accelerated (up to hundreds of kilometers per second) by
        plasma turbulence in the solar atmosphere, the corona. understanding this coronal
        heating is an open problem.</li>
      <li>plasma impacting the earth&rsquo;s magnetic field can degrade satellites and
        disrupt power grids: mitigating the impacts of this space weather requires us to
        understand the plasma turbulence in interplanetary space.</li>
      <li>closer to home, plasma turbulence is also a main cause of loss of confinement in
        nuclear fusion devices: solving this problem would be a major step towards accessing
        a near-limitless energy source.</li>
    </ol>` },
  { summary: "why do numerical simulations?", html: `
    <p>the governing equations are strongly nonlinear, coupling an enormous range of length
      and timescales, and cannot be solved only with pen-and-paper theory. simulations are a
      crucial tool: we can perform numerical experiments where every parameter is precisely
      controlled, and every output precisely measured at every point in space and time. we
      then compare the simulations against real measured data: in a first for humanity,
      NASA&rsquo;s Parker Solar Probe has entered our sun&rsquo;s corona, directly sampling
      this turbulent plasma. comparing simulations with these state-of-the-art measurements
      drives improvements in our models, and better understanding of this fundamental and
      omnipresent physical phenomenon.</p>` },
  { summary: "technical details", html: `
    <p>RMHD equations, solved using a pseudospectral method with a Williamson 1980
      low-storage RK3 modified to calculate linear terms (Alfv&eacute;n wave
      propagation and viscous/resistive dissipation) with integrating factors. note that
      the taranis code by default uses a different method for the z-derivatives (centered
      finite differences) which is much better for parallelizing over many gpus; the
      algorithm used here is the same as taranis with <code>z_spectral=True</code>. also,
      WebGPU has no fp64, so this is all fp32. you can validate that the same equations are
      being solved using the <b>self-test</b> button to check a small test problem against
      the taranis solution.</p>
    <p>the <b>preset</b> dropdown in the top bar &mdash; decaying A/B Elsasser packets,
      Kelvin&ndash;Helmholtz shear layers and a tearing mode growing magnetic islands in 2D,
      a counter-propagating Alfv&eacute;n-wave collision in 3D &mdash; sets the controls and
      opens the displays that go with them. everything stays adjustable; presets are
      configurations, not separate programs (<code>?demo=decay</code> /
      <code>?demo=kh</code> / <code>?demo=tearing</code> / <code>?demo=collision</code> are
      deep links to the same thing).</p>` }
];
// the built panes, kept so the no-GPU path can open them without a DOM query
// ---------------------------------------------------------------------------
// the contact line (plans-webgpu/ANALYTICS_PLAN.md phase 2)
// ---------------------------------------------------------------------------
// Analytics on these pages is pageview counts and nothing else -- no events, no adapter
// strings, no "did they press Run". This link is the deliberate substitute: the same
// diagnostic information, VOLUNTEERED, from the few people motivated enough to write.
//
// The address is assembled here and never appears contiguously in the served HTML: a
// mailto: on a page carrying a real name and a publication list gets harvested. This is
// not serious obfuscation and does not pretend to be -- it defeats the regex scrapers,
// which is the entire threat model. Consequence, accepted: no contact link with
// scripting off. These pages need WebGPU, which needs scripting.
const MAIL_USER = "alfred.mallet", MAIL_HOST = "berkeley.edu";
const ISSUES_URL = "https://github.com/alfredmallet/taranis/issues/new";
// DRAFT COPY -- Alfred's to replace, like the preset text and the aniso hints.
const CONTACT_TXT = "email Alfred", ISSUES_TXT = "report a bug";
function contactAddr() { return MAIL_USER + String.fromCharCode(64) + MAIL_HOST; }
// Built at CLICK time, not at build time: on a page that booted, gpuInfo is populated by
// then and the report names the GPU; on one that did not, the line is honestly absent
// rather than a stale "" captured before initGPU ran. Every probe is guarded -- this runs
// during a click handler on the fallback page, where nothing is safe to assume.
function contactBody() {
  const L = [];
  const push = (k, f) => { try { const v = f(); if (v) L.push(k + ": " + v); } catch (e) {} };
  push("page", () => location.href);
  // feature-detected, the same way the controls sweep at the bottom of this file does:
  // the build id has a CLASS and no id (pages.yml seds that exact span, so it must not
  // grow one), which makes querySelector the only way to reach it.
  push("build", () => { const b = document.querySelector && document.querySelector(".buildid");
                        return b && b.textContent; });
  push("webgpu", () => navigator.gpu ? (device ? "yes, device created" : "yes, no device") : "NO");
  push("gpu", () => gpuInfo);
  push("browser", () => navigator.userAgent);
  // NB the guard in push() is "is the value falsy", and a concatenation NEVER is: build
  // the string only once the numbers are known to be numbers, or this line reports
  // "undefinedxundefined" instead of being absent.
  push("screen", () => (typeof window.innerWidth === "number" && typeof window.innerHeight === "number")
    ? window.innerWidth + "x" + window.innerHeight + " dpr " + (window.devicePixelRatio || 1)
    : "");
  return "\n\n\n-- what went wrong, and anything above you would rather delete --\n"
    + L.join("\n");
}
function contactBuild(is3d) {
  const host = el("contact");
  if (!host) return;
  host.textContent = "";
  const PAGE_NAME = is3d ? "3D" : "2D";
  // The leading separator is emitted HERE, not left in the markup: if common.js fails to
  // load, an empty #contact preceded by a markup "·" leaves the acknowledgements line
  // ending in a dangling bullet. Nothing is better than half a list.
  host.appendChild(document.createTextNode("· "));
  const a = _mk("a", null, host);
  a.textContent = CONTACT_TXT;
  a.href = "mailto:" + contactAddr();
  // The subject and the diagnostics are attached on the way out. Kept under ~1500 chars
  // so no mail client trims the body.
  a.onclick = () => {
    const subj = "plasma turbulence in your browser — " + PAGE_NAME;
    let q = "?subject=" + encodeURIComponent(subj)
      + "&body=" + encodeURIComponent(contactBody());
    // Truncate on a CHARACTER boundary of the encoded string: slicing blind can cut a
    // percent-escape in half ("...%2" / "...%"), and what the mail client then gets is
    // not a short URI but a malformed one. Realistic hrefs are ~600-900 chars so this
    // never fires in practice -- which is exactly why it would have been found in the
    // wild, by someone whose report we could not read, rather than here.
    if (q.length > 1500) q = q.slice(0, 1500).replace(/%[0-9A-Fa-f]?$/, "");
    a.href = "mailto:" + contactAddr() + q;
    return true;
  };
  host.appendChild(document.createTextNode(" · "));
  const g = _mk("a", null, host);
  g.textContent = ISSUES_TXT;
  g.href = ISSUES_URL;
  g.target = "_blank";
  g.rel = "noopener";
}

let railPanes = [];
// the three markup hooks (#intro, #tabs, #rail) are empty on both pages, exactly as
// #topbar and #controls are. Called first in each app's boot(), so the chrome is on the
// page even when initGPU is about to fail.
function chromeBuild(o) {
  const is3d = !!(o && o.is3d);
  // FIRST, deliberately: chromeBuild returns early when there is no #rail, and the
  // contact link is the one piece of chrome that must survive every degraded boot.
  // checkgc.js pins this ordering -- it is an argument, not an accident.
  contactBuild(is3d);
  // the intro rides directly under the subtitle so a first-timer is told what the page
  // IS before anything else (Alfred/Charlotte follow-up) -- but as a <details>, so the
  // cost to everyone who has read it is one line, not two paragraphs of canvas push.
  // It boots OPEN and stays open until the visitor closes it once; the choice is
  // remembered exactly like the params toggle (absent/throwing storage = "no memory").
  const intro = el("intro");
  if (intro) {
    _mk("summary", null, intro).innerHTML = "what is all this?";
    const bod = _mk("div", null, intro);
    for (const par of RAIL_LEAD) _mk("p", "lead", bod).innerHTML = par;
    let open = true;
    try { open = localStorage.getItem("taranisIntro") !== "0"; } catch (e) {}
    intro.open = open;
    intro.ontoggle = () => {
      try { localStorage.setItem("taranisIntro", intro.open ? "1" : "0"); } catch (e) {}
    };
  }
  const nav = el("tabs");
  if (nav) {
    for (const t of TABS) {
      const on = t.is3d === is3d;
      const e = _mk(on ? "span" : "a", on ? "tab on" : "tab", nav);
      e.innerHTML = t.t;
      if (on) e.setAttribute("aria-current", "page"); else e.href = t.href;
    }
  }
  const rail = el("rail");
  railPanes = [];
  if (!rail) return;
  for (const p of RAIL_PANES) {
    const d = _mk("details", null, rail);
    _mk("summary", null, d).innerHTML = p.summary;
    // the body goes in its own div: innerHTML on the <details> would eat the <summary>
    _mk("div", null, d).innerHTML = p.html;
    railPanes.push(d);
  }
}
// the no-WebGPU first impression (ONEPAGE_PLAN B.5). Reached only from initGPU's three
// failure paths, every one of which has just written its browser advice into #status --
// so that node is MOVED under the picture (same id, same text, as demohint is moved into
// the topbar) instead of being reworded here. The <img> is created HERE and nowhere else:
// a visit that gets a device must not fetch a 360 kB poster it will never show.
const POSTER = "poster.png";
function gpuFallback() {
  const host = el("displays");
  if (host) {
    const card = _mk("div", "card disp", host);
    const wrap = _mk("div", "cvwrap", card);
    const img = _mk("img", "cvmain", wrap);
    img.alt = "|u| in a 512 by 512 forced-turbulence run, with velocity arrows";
    img.src = POSTER;
    _mk("div", "viewcap", card).innerHTML =
      "a real 512&sup2; run of this solver, recorded earlier &mdash; your browser cannot "
      + "run the live one:";
    const st = el("status");
    if (st) card.appendChild(st);
  }
  const ro = el("readout");
  if (ro) ro.textContent = "";
  // nothing past this point is wired (wireCommonControls is never reached), so the
  // topbar's buttons would be enabled-looking dead chrome -- disable them (review
  // NOTE 11); the tab strip and docs link are plain <a>s and keep working
  for (const id of ["btnRun", "btnReset", "btnParams", "btnText"]) {
    const b = el(id);
    if (b) b.disabled = true;
  }
  // with no run to watch, the explanation is the whole page: open all of it, the
  // intro included (and never mind a remembered dismissal -- this visit is different)
  const intro = el("intro");
  if (intro) intro.open = true;
  for (const d of railPanes) d.open = true;
}

// ---------------------------------------------------------------------------
// collapsible control groups: default-open on a wide viewport, collapsed on a
// narrow one. Deliberately NOT persisted (REFINE_PLAN F.4).
// ---------------------------------------------------------------------------
function groupsInit() {
  let wide = true;
  try { wide = window.matchMedia("(min-width: 900px)").matches; } catch (e) {}
  const gs = document.querySelectorAll ? document.querySelectorAll("#controls details") : [];
  for (const g of gs) g.open = wide || g.hasAttribute("data-keep-open");
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
// never destroy a solver in an event handler: the frame loop may be awaiting a
// readback on its buffers. Retire it here and free it at the top of a later frame.
const graveyard = [];

// the live geometry: the solver's own parameters while one exists, else what the
// controls currently say (before the first rebuild, and between rebuilds).
function liveParams() { return solver ? solver.p : uiParams(); }

// ---- locked slider pairs (zeta+- amplitudes, forcing eps+-) ----------------
// While the lock checkbox is checked, moving either handle writes both. ONE
// implementation; the two pairs differ only in their ids and their callbacks.
//   onInput   every tick (label sync / live parameter upload)
//   onChange  optional, on release only (the amplitudes re-apply the IC there)
function lockPair(idA, idB, idLock, onInput, onChange) {
  const other = id => (id === idA ? idB : idA);
  const mirror = id => { if (el(idLock).checked) el(other(id)).value = el(id).value; };
  for (const id of [idA, idB]) {
    el(id).oninput = () => { mirror(id); onInput(); };
    if (onChange) el(id).onchange = () => { mirror(id); onChange(); };
  }
  el(idLock).onchange = () => { mirror(idA); onInput(); if (onChange) onChange(); };
}
// the two forcing powers, in the SAME units (each is a contribution to dE/dt, so the
// total injection rate is their sum -- rmhd._forcing_scale_from's convention)
function uiEps() {
  const on = el("cbForce").checked;
  return [on ? Math.pow(10, parseFloat(el("rEpsP").value)) : 0,
          on ? Math.pow(10, parseFloat(el("rEpsM").value)) : 0];
}
// the two zeta+- amplitudes (max |zhat x grad zeta+-| after normalization)
function uiAmp() {
  return [Math.pow(10, parseFloat(el("rAmpP").value)), Math.pow(10, parseFloat(el("rAmpM").value))];
}
// the forcing band [n_min, n_max) in units of kunit. Integer handles, kept at least one
// shell apart; whichever handle the user moved wins.
function uiFshell() {
  const a = el("rFmin"), b = el("rFmax");
  let lo = parseInt(a.value, 10) | 0, hi = parseInt(b.value, 10) | 0;
  hi = Math.max(hi, lo + 1); lo = Math.min(lo, hi - 1);
  a.value = String(lo); b.value = String(hi);
  return [lo, hi];
}

// ===========================================================================
// perpendicular dissipation: the slider's dynamic range and the auto-diss
// controller (FEEDBACK_2026-08-08 P2 items 7 and 6)
// ===========================================================================
// The linear operator is -nu * k_perp^(2*hyper) (makeGrid's linL), so "diss" is the
// COEFFICIENT nu and the slider carries log10(nu). Two numbers follow from the GRID and
// the hyper exponent alone -- no field, no forcing power -- and both items below are
// built on them:
//
//   k_d      the wavenumber the cascade is meant to terminate at: DISS_KD_FRAC of the
//            largest RETAINED perpendicular wavenumber. In this app that cutoff is the
//            2/3 dealias, which is exactly what the perpendicular spectrum is binned out
//            to -- nb bins of kunit each -- so k_d = frac * nb * kunit and the controller
//            below measures in the same units it acts in.
//   nu_marg  the MARGINAL coefficient at that scale for an O(1) box-scale amplitude:
//            walk a Kolmogorov u(k) = u_1 (k/k_1)^(-1/3) from u_1 = 1 at k_1 = kunit down
//            to k_d and impose the same balance the controller imposes (dissipation rate
//            = nonlinear rate at k_d, i.e. nu k_d^(2n) = u_d k_d):
//                nu_marg = k_1^(1/3) * k_d^(2/3 - 2*hyper).
//            This is the amplitude-free stand-in for the controller's own answer. It is
//            the old eps-based "auto" button's formula with eps^(1/3) -> u_1 k_1^(1/3)
//            (eps ~ u_1^3 k_1) and lands within ~0.3 decades of it on the 512^2 decaying
//            preset that formula was tuned on -- without the eps dependence, which is
//            precisely what P1 item 3 was about.
const DISS_KD_FRAC = 0.6;
const dissKd = (nb, kunit) => DISS_KD_FRAC * Math.max(1, nb) * kunit;
function dissMarginal(nb, kunit, hyper) {
  return Math.pow(kunit, 1 / 3) * Math.pow(dissKd(nb, kunit), 2 / 3 - 2 * hyper);
}
// nb / kunit / hyper as the CONTROLS currently say -- the same two formulas makeGrid and
// nbins use, so this works with no live solver (a preset writes the diss slider, and the
// range must already fit it, before the first rebuild).
function dissGrid() {
  const q = uiParams();
  return { nb: nbins(q.nx, q.ny, q.Lx, q.Ly), hyper: q.hyper,
           kunit: Math.min(2 * Math.PI / q.Lx, 2 * Math.PI / q.Ly) };
}

// ---- item 7: the slider as a demo instrument -------------------------------
// The old fixed -20 .. -1 spent most of its travel on values nothing can tell apart. One
// sweep of the new range walks the whole physics instead:
//   top     Re ~ 1 at the box scale -- nonlinear rate u k_1 equal to the dissipation rate
//           nu k_1^(2n) with u = O(1), i.e. nu_top = k_1^(1 - 2*hyper) (k_1 = kunit, so
//           exactly 1 in a 2pi box). Above it the box-scale flow is viscous.
//   middle  nu_marg: turbulence with a resolved dissipation range, where the dissipation
//           RATE stops depending on nu at all (the zeroth law).
//   bottom  DISS_DECADES_BELOW under nu_marg: the cascade no longer terminates inside the
//           retained band, energy piles up at the dealias scale and the run stops being a
//           simulation of anything. That bottom is deliberately only 3 decades down: it
//           is also the controller's nu_min (see below), and a controller that bottoms
//           out 3 decades under marginal recovers in a couple of seconds.
// Both ends move with hyper and with the resolution/box (k_d and k_1 do), so the range is
// recomputed by syncCommonLabels rather than written once.
const DISS_STEP = 0.05;                 // log10 per slider notch (unchanged)
const DISS_DECADES_BELOW = 3;
const DISS_LG_OPEN = [-30, 6];          // the hard bounds presetWrite opens the range to
// snap to the step grid, away from (dir < 0) or toward (dir > 0) +infinity
const dissQ = (x, dir) => DISS_STEP * (dir < 0 ? Math.floor(x / DISS_STEP + 1e-9)
                                               : Math.ceil(x / DISS_STEP - 1e-9));
function dissRange(nb, kunit, hyper) {
  const hi = (1 - 2 * hyper) * Math.log10(kunit);
  const lo = Math.log10(dissMarginal(nb, kunit, hyper)) - DISS_DECADES_BELOW;
  return [Math.min(lo, hi - 1), hi];    // never a degenerate (or inverted) axis
}
// A range <input> SANITIZES an assigned value against min/max/step, so narrowing the
// range around a preset's diss would silently rewrite it. presetWrite therefore OPENS the
// range to the hard bounds before it writes anything, and the syncLabels that follows
// narrows it again -- widened outward to whatever value is then stored, so a re-range
// never moves the physical value, only the travel around it.
function dissRangeOpen() {
  const e = el("rDiss");
  if (!e) return;
  e.min = String(DISS_LG_OPEN[0]); e.max = String(DISS_LG_OPEN[1]);
}
function dissRangeSync() {
  const e = el("rDiss");
  if (!e) return;
  const g = dissGrid(), r = dissRange(g.nb, g.kunit, g.hyper), v = parseFloat(e.value);
  let lo = dissQ(r[0], -1), hi = dissQ(r[1], +1);
  if (isFinite(v)) { lo = Math.min(lo, dissQ(v, -1)); hi = Math.max(hi, dissQ(v, +1)); }
  e.min = lo.toFixed(2); e.max = hi.toFixed(2);
}
// the ONE place code (as opposed to a dragging finger) writes the diss slider: quantized
// to the slider's own step, clamped to the live range, and pushed down the LIVE parameter
// path -- never a rebuild -- and only when it actually moved. Returns whether it did.
function dissWriteLog(lg) {
  const e = el("rDiss");
  if (!e || !isFinite(lg)) return false;
  const v = Math.min(parseFloat(e.max),
                     Math.max(parseFloat(e.min), DISS_STEP * Math.round(lg / DISS_STEP)));
  if (Math.abs(v - parseFloat(e.value)) < 1e-9) return false;
  e.value = v.toFixed(2);
  applyControls();
  return true;
}
// The t = 0 SEED. Before the first measurement there is no amplitude to measure, so a
// preset that wants the app to pick its dissipation gets nu_marg. (Presets that quote a
// physical dissipation -- tearing's eta -- set rDiss themselves and turn the controller
// off instead.)
function autoDissSeed() { const g = dissGrid(); dissWriteLog(Math.log10(dissMarginal(g.nb, g.kunit, g.hyper))); }

// ---- item 6: the auto-diss controller --------------------------------------
// A tickbox, ON by default -- not a one-shot button. It sets nu CONTINUOUSLY from the
// MEASURED field amplitude near the dissipation scale, which needs no assumption about
// where the energy came from: forced, decaying, KH, tearing and packet collisions all go
// through one rule. That is what supersedes P1 item 3, whose failure was an amplitude
// taken from the FORCING sliders on a run with the forcing switched off.
//
// THE RULE is Jono Squire's rmhd-gpu (`rmhdgpu/auto_dissipation.py`), ported with its
// structure and its defaults:
//   k_d      = DISS_KD_FRAC * (max retained k_perp)                    [dissKd, above]
//   E_d      = kinetic + magnetic energy in the LOGARITHMIC shell
//              k_d e^-W <= k_perp <= k_d e^+W,  W = AUTODISS_SHELL_W
//   u_d      = sqrt(2 E_d)                              the velocity at that scale
//   nu_targ  = u_d * k_d^(1 - 2*hyper)                  i.e. nu k_d^(2n) = u_d k_d: the
//              dissipation rate at k_d equals the nonlinear rate there, which IS the
//              statement "the cascade terminates at k_d"
//   nu       relaxed toward nu_targ in LOG space by AUTODISS_SMOOTH per update, one
//              update's change capped at a factor AUTODISS_MAX_FACTOR, clamped to
//              [nu_min, nu_max]
// E_d is read off the app's OWN perpendicular spectrum bins (the E_u and E_b lanes), so
// the quadratic density is exactly the energy diagnostic's and there is no new kernel.
//
// DEVIATIONS from the reference, both forced by the app rather than chosen:
//  - cadence. rmhdgpu updates every 10 STEPS; here the loop runs 1..64 steps per frame at
//    a frame rate nobody controls, and each update costs a spectrum pass plus a map round
//    trip, so it is a WALL-CLOCK cadence like every other readback hook in the file
//    (arrows / cut 10 Hz, field lines 2 Hz): AUTODISS_PERIOD, 2 Hz. With the factor-2 cap
//    that is at most 0.6 decades of nu per second, fast enough that a run cannot outgrow
//    the controller and slow enough that each update sees a spectrum that has responded
//    to the last one.
//  - [nu_min, nu_max] is the SLIDER's live range (item 7) instead of a separate pair of
//    constants: its ends are "Re ~ 1" and "3 decades below marginal", which is exactly
//    the window this controller has any business in -- and it guarantees the slider,
//    which follows the controller, can always represent the value.
//  - a shell with NO measurable energy HOLDS nu instead of driving it to nu_min. A
//    quiescent or freshly-seeded IC (forced from rest, KH/tearing at t = 0, a smooth
//    packet pair) has nothing at k_d yet; that is an uninformative measurement, not an
//    instruction. Once anything IS there the walk is capped at a factor 2 per update, so
//    even a badly over-dissipated start descends gradually from the preset's/slider's
//    value and turns round as soon as the amplitude at k_d comes up.
const AUTODISS_SHELL_W = 0.5;           // log half-width of the measurement shell
const AUTODISS_SMOOTH = 0.2;            // log-space relaxation per update
const AUTODISS_MAX_FACTOR = 2;          // hard cap on one update's change, as a ratio
const AUTODISS_PERIOD = 500;            // ms between updates (2 Hz)

// PURE CORE (state in, nu out -- this is what the node checks drive).
// `bins` is the 3*nb [E_u | E_b | H_c] stack, bin b at k_perp = b*kunit.
function autoDissShellE(bins, nb, kunit) {
  const kd = dissKd(nb, kunit);
  const lo = kd * Math.exp(-AUTODISS_SHELL_W), hi = kd * Math.exp(AUTODISS_SHELL_W);
  let e = 0;
  for (let b = 1; b < nb; b++) {
    const k = b * kunit;
    if (k < lo || k > hi) continue;
    const v = bins[b] + bins[nb + b];              // E_u + E_b in this shell
    if (isFinite(v) && v > 0) e += v;
  }
  return e;
}
// nu that would put the cascade's termination exactly at k_d, or 0 for "no measurement"
function autoDissTarget(bins, nb, kunit, hyper) {
  if (!(nb > 1) || !(kunit > 0)) return 0;
  const ed = autoDissShellE(bins, nb, kunit);
  if (!(ed > 0)) return 0;
  return Math.sqrt(2 * ed) * Math.pow(dissKd(nb, kunit), 1 - 2 * hyper);
}
// one update of the controller state: clamp the target, relax in log space, cap the move
function autoDissRelax(nu, target, lo, hi) {
  if (!(nu > 0) || !(target > 0) || !(lo > 0) || !(hi >= lo)) return nu;
  const t = Math.min(hi, Math.max(lo, target)), cap = Math.log(AUTODISS_MAX_FACTOR);
  const d = Math.max(-cap, Math.min(cap, AUTODISS_SMOOTH * Math.log(t / nu)));
  return Math.min(hi, Math.max(lo, nu * Math.exp(d)));
}

// IMPURE EDGE: cadence, readback, slider write. Called from the frame loop with the
// perpendicular bins the spectrum CARDS read this frame, or null when no such card is
// open -- in which case the controller takes its own readback at its own cadence, the
// same "one bounded round trip per period" contract planeTrackHook and fieldLineHook use.
//
// The slider is DISABLED while the box is ticked (syncCommonLabels) and follows the
// controller, so the level is always visible and unticking simply leaves the manual
// slider where the controller last put it. Everything goes through dissWriteLog, hence
// through applyControls -> solver.refreshDissipation: the live path, never a rebuild.
// Its 0.05-decade quantization doubles as the controller's DEAD BAND -- with
// AUTODISS_SMOOTH = 0.2 an update smaller than 0.125 decades of log-distance rounds to no
// change at all, which is what stops a converged controller from re-uploading the
// dissipation array twice a second.
let autoDissAt = 0;
function autoDissOn() { const e = el("cbAutoDiss"); return !!(e && e.checked); }
// the spectrum cards' last readback, kept so the controller can ride it instead of
// paying for a second spectrum pass: the cards fire at ~3.3 Hz and the controller at
// 2 Hz, so with a card open a fresh-enough entry almost always exists. `sv` keys the
// cache to the solver that produced it -- a preset/resolution switch retires the
// solver and with it the cache (nb and kunit change with the grid).
const autoDissCache = { sv: null, at: 0, perp: null };
async function autoDissHook(sv) {
  if (!sv || !autoDissOn() || !running) return;
  const now = performance.now();
  if (now - autoDissAt < AUTODISS_PERIOD) return;
  autoDissAt = now;
  let bins = (autoDissCache.sv === sv && now - autoDissCache.at <= AUTODISS_PERIOD)
    ? autoDissCache.perp : null;
  if (!bins) {                                     // no card open (or its readback stale)
    const sp = await sv.readSpectrum();
    if (sv !== solver) return;                     // retired while we were awaiting
    bins = sp.perp;
  }
  const e = el("rDiss");
  const t = autoDissTarget(bins, sv.nb, sv.g.kunit, sv.p.hyper);
  if (!(t > 0)) return;                            // nothing measurable yet: hold
  dissWriteLog(Math.log10(autoDissRelax(Math.pow(10, parseFloat(e.value)), t,
                                        Math.pow(10, parseFloat(e.min)),
                                        Math.pow(10, parseFloat(e.max)))));
}
// A HYPER change re-parameterizes the operator: nu_targ ~ k_d^(1-2*hyper) moves by many
// decades at once (1 <-> 4 at 256^2 is ~13), and the factor-2-per-update cap -- which is
// for measurement noise, not for a changed exponent -- would leave the run
// mis-dissipated for ~15 s while the controller walked over. So the hyper select JUMPS:
// one fresh measurement, the clamped target written outright. Runs after applyControls,
// so the slider range (dissRangeSync via syncLabels) and sv.p.hyper are already the new
// hyper's; a state with nothing measurable yet falls back to the marginal seed for the
// NEW hyper instead. Manual (unticked) diss is the user's own and is never touched.
async function autoDissRetarget() {
  const sv = solver;
  if (!sv || !autoDissOn()) return;
  autoDissAt = performance.now();                  // the jump IS this period's update
  const sp = await sv.readSpectrum();
  if (sv !== solver) return;                       // retired while we were awaiting
  const e = el("rDiss");
  const t = autoDissTarget(sp.perp, sv.nb, sv.g.kunit, sv.p.hyper);
  if (t > 0) dissWriteLog(Math.log10(Math.min(Math.pow(10, parseFloat(e.max)),
                                              Math.max(Math.pow(10, parseFloat(e.min)), t))));
  else autoDissSeed();
}

// ---------------------------------------------------------------------------
// presets (the in-app dropdown; ?demo=NAME is the same thing as a deep link)
// ---------------------------------------------------------------------------
// A preset is a CONFIGURATION, never new physics:
//   set     control-element id -> value (booleans go to .checked, else .value)
//   prep    runs right after `set` (for the "auto" buttons, which derive one
//           control from the others)
//   layout  the card layout it asks for: {disp: [state...], charts: [type...]}
//   hint    ONE line of HTML for #demohint ("" for the default preset)
// The first entry of a registry is the app's default. Each app owns its registry;
// everything below is shared.
//
// file:// is a first-class target here, and location.search works there; the try/catch
// only covers exotic hosts (and the node smoke test) with no location object at all.
function demoNameFromURL() {
  try {
    const q = (typeof location !== "undefined" && location.search) || "";
    return new URLSearchParams(q).get("demo");
  } catch (e) { return null; }
}
// the user's "hide text" toggle for #demohint (btnText, wired in wireCommonControls):
// presetWrite honours it, so switching preset never resurrects hidden text
let hintHidden = false;
// write one preset's control values + hint; returns the record (or null)
function presetWrite(registry, name) {
  const d = (registry && registry[name]) || null;
  if (!d) return null;
  // the diss slider's range is dynamic and a range <input> sanitizes assignments against
  // it: open it to the hard bounds first so a preset's diss is written verbatim, whatever
  // hyper / resolution the range was last narrowed to (the syncLabels of the bootApply
  // that follows narrows it again, around the value this wrote -- item 7).
  dissRangeOpen();
  for (const id in (d.set || {})) {
    const e = el(id);
    if (!e) { console.warn("preset " + name + ": no control #" + id); continue; }
    const v = d.set[id];
    if (typeof v === "boolean") e.checked = v; else e.value = String(v);
  }
  const h = el("demohint");
  if (h) { h.innerHTML = d.hint || ""; h.style.display = (d.hint && !hintHidden) ? "block" : "none"; }
  if (d.prep) d.prep();
  return d;
}
// fill #selPreset, preselect ?demo=NAME (else the registry's first entry) and write
// it. Called BEFORE the first rebuild, so the solver is built with the preset's grid.
function presetBoot(registry) {
  const keys = Object.keys(registry);
  const s = el("selPreset");
  if (s) {
    s.innerHTML = "";
    for (const k of keys) {
      const o = document.createElement("option");
      o.value = k; o.textContent = registry[k].label || k;
      s.appendChild(o);
    }
  }
  const url = demoNameFromURL();
  const key = (url && registry[url]) ? url : keys[0];
  if (s) s.value = key;
  return presetWrite(registry, key);
}
// ---------------------------------------------------------------------------
// boot wiring shared by both apps
// ---------------------------------------------------------------------------
// These talk to functions each app defines under the SAME names (uiParams, syncIC,
// syncLabels, rebuild, applyIC, applyControls, wireDrawEditor, runSelfTest)
// -- an arrangement the shared code above already relies on: common.js is loaded
// first, but its bodies only run once the app script has declared them.

// the slider readouts every page has. Each app's syncLabels() calls this and then adds
// its own (3D: z_diss_k, sigma_z, chi) -- the shared numbers exist once.
function syncCommonLabels() {
  const eps = uiEps(), amp = uiAmp(), fs = uiFshell();
  // the diss slider's travel follows hyper / resolution / box (item 7), and the auto-diss
  // controller owns the handle while its box is ticked (item 6). Both belong here: this
  // is the one function every path that can change either of them ends in.
  dissRangeSync();
  el("rDiss").disabled = autoDissOn();
  el("vDiss").textContent = Math.pow(10, parseFloat(el("rDiss").value)).toExponential(1);
  el("vEpsP").textContent = el("cbForce").checked ? eps[0].toExponential(1) : "off";
  el("vEpsM").textContent = el("cbForce").checked ? eps[1].toExponential(1) : "off";
  el("vFshell").textContent = fs[0] + "–" + fs[1];
  el("vTau").textContent = parseFloat(el("rTau").value).toFixed(2);
  el("vCfl").textContent = parseFloat(el("rCfl").value).toFixed(2);
  el("vCflEvery").textContent = el("rCflEvery").value;
  el("vAmpP").textContent = amp[0].toPrecision(3);
  el("vAmpM").textContent = amp[1].toPrecision(3);
  el("vSigP").textContent = icSigmaPerp().toPrecision(3);
  ampBasisSync();
}
// The two amp sliders say what they MEAN (FEEDBACK_2026-08-10 item 15). Labels and titles
// only -- icAmpBasis is what actually changes the arithmetic, and it reads the same
// predicate. In the (phi, psi) basis the knobs are the two physical field strengths,
// which is what lets a drawing carry a strong vortex and a weak island at once.
const AMP_TI_Z = ["initial max |z⁺| = max |∇ζ⁺|, the amplitude the stored "
                  + "drawing / glyph is rescaled to at apply time",
                  "initial max |z⁻| = max |∇ζ⁻|, likewise"];
const AMP_TI_PP = ["initial max |u| = max |∇φ|, i.e. the peak flow speed the painted "
                   + "φ is rescaled to at apply time (0 if nothing was painted into φ)",
                   "initial max |b| = max |∇ψ|, i.e. the peak perpendicular field "
                   + "strength the painted ψ is rescaled to (0 if nothing was painted into ψ)"];
function ampBasisSync() {
  const pp = icAmpBasis() === "pp", ti = pp ? AMP_TI_PP : AMP_TI_Z;
  el("labAmpP").innerHTML = pp ? "&phi; amp" : "&zeta;&#8314; amp";
  el("labAmpM").innerHTML = pp ? "&psi; amp" : "&zeta;&#8315; amp";
  el("rAmpP").title = ti[0]; el("labAmpP").title = ti[0];
  el("rAmpM").title = ti[1]; el("labAmpM").title = ti[1];
}
// forcing on/off: the two eps sliders and their lock follow the checkbox
function syncForceEnabled() {
  for (const id of ["rEpsP", "rEpsM", "cbEpsLock"]) el(id).disabled = !el("cbForce").checked;
}

// bring the page into the state a preset (or a fresh boot) asks for.
// NB every visit boots PAUSED (Alfred reversed ONEPAGE_PLAN C's autoplay on 2026-08-10:
// the big green Run is the call to action, and nothing moves before it's pressed) -- if
// autoplay ever comes back, the seam is this function's FIRST call, which is exactly the
// boot one on both pages, past a successful initGPU, with the solver freshly rebuilt.
function bootApply(pre) {
  syncIC();
  syncLabels();
  syncForceEnabled();
  rebuild();
  cardsLayout(pre && pre.layout);
}
// `running` is flipped from several places (topbar click, IC editor enter/leave). The
// hero Run button carries the state as COLOUR as well as text (green = will run, red =
// will pause; .stop class, tones in style.css --run-*), so every visible flip goes
// through here to keep flag, label and class coherent. The self-test wrapper's silent
// save/restore of `running` stays a bare assignment on purpose -- the button must not
// blink red during a test the user perceives as instantaneous.
function setRunning(b) {
  running = b;
  const r = el("btnRun");
  if (r) { r.textContent = b ? "Pause" : "Run"; r.classList.toggle("stop", b); }
}
// every control whose handler is the same in both apps.
//   opts.presets    the app's preset registry
//   opts.sliders    extra live-parameter slider ids beyond the shared ones
//   opts.rebuildOn  extra <select> ids that force a full rebuild (3D: selLz)
function wireCommonControls(opts) {
  const s = el("selPreset");
  if (s) s.onchange = () => bootApply(presetWrite(opts.presets, s.value));
  el("btnRun").onclick = () => setRunning(!running);
  // hide/show the whole #controls block from the always-visible topbar. Pure display
  // toggle: nothing is re-read on show, so hidden controls keep their state. HIDDEN is
  // the boot default (ONEPAGE_PLAN A.1: first-timers get canvas, not sliders); the
  // click is remembered so a returning tinkerer pays the extra click once, ever.
  // Storage can be absent or throwing (Safari private mode, the stub env) -- treat
  // that as "no memory", never as a boot failure.
  const bp = el("btnParams");
  if (bp) {
    const paramsShow = show => {
      el("controls").style.display = show ? "" : "none";
      bp.textContent = show ? "hide params" : "show params";
    };
    bp.onclick = () => {
      const show = el("controls").style.display === "none";
      paramsShow(show);
      try { localStorage.setItem("taranisShowParams", show ? "1" : "0"); } catch (e) {}
    };
    let show = false;
    try { show = localStorage.getItem("taranisShowParams") === "1"; } catch (e) {}
    paramsShow(show);
  }
  // the k_perp filter's opt-in (item 4), remembered exactly like that toggle: off on a
  // first visit, and a visitor who wants the handles asks for them once. Ticking it only
  // has to re-apply every card -- that is what shows the handles and rewrites the band
  // words -- so there is no second path for the filter to come on by.
  const cf = el("cbFilter");
  if (cf) {
    let on = false;
    try { on = localStorage.getItem("taranisFilter") === "1"; } catch (e) {}
    cf.checked = on;
    cf.onchange = () => {
      try { localStorage.setItem("taranisFilter", cf.checked ? "1" : "0"); } catch (e) {}
      cardsSync();
    };
  }
  // the preset's explanatory text rides the sticky topbar as its own full-width last
  // row (Alfred 2026-08-10 follow-up), so it stays on screen while the page scrolls
  // and survives "hide params". The node is MOVED (same id, state kept): presetWrite
  // may already have filled it during presetBoot. Its own show/hide button sits next
  // to the params one; `hintHidden` is honoured by later presetWrites, and display
  // stays "none" for hintless presets whatever the toggle says.
  const tb = el("topbar"), dh = el("demohint");
  if (tb && dh) tb.appendChild(dh);
  const bt = el("btnText");
  if (bt) bt.onclick = () => {
    hintHidden = !hintHidden;
    const h = el("demohint");
    if (h) h.style.display = (!hintHidden && h.innerHTML) ? "block" : "none";
    bt.textContent = hintHidden ? "show text" : "hide text";
  };
  el("btnReset").onclick = () => { solver.p.seed = uiParams().seed; applyIC(); };
  el("selIC").onchange = () => { syncIC(); applyIC(); syncLabels(); };
  // amplitudes are stored OUTSIDE the drawing (REFINE_PLAN G.3): the sliders scale the
  // normalized zeta+- at apply time, so a release re-applies in every preset, drawn
  // ones included -- pause, move the slider, Reset really does rescale.
  lockPair("rAmpP", "rAmpM", "cbAmpLock", syncLabels, () => { if (!icDraw.on) applyIC(); });
  lockPair("rEpsP", "rEpsM", "cbEpsLock", applyControls);
  // hyper first uploads live (applyControls), then the controller jumps to the new
  // hyper's target rather than crawling there under its per-update cap (see
  // autoDissRetarget); fire-and-forget is the event-handler contract for readbacks here
  el("selHyper").onchange = () => { applyControls(); autoDissRetarget(); };
  // auto-diss (item 6): ticking hands the slider to the controller and lets it act on the
  // next frame; unticking gives the slider back exactly where the controller left it.
  el("cbAutoDiss").onchange = () => { autoDissAt = 0; syncLabels(); };
  el("cbForce").onchange = () => { syncForceEnabled(); applyControls(); };
  for (const id of ["selRes"].concat(opts.rebuildOn || [])) el(id).onchange = rebuild;
  for (const id of ["rDiss", "rTau", "rCfl", "rCflEvery"].concat(opts.sliders || [])) {
    el(id).oninput = applyControls;
  }
  // IC-shape sliders (the equilibrium knobs, REFINE_PLAN J.3): a live readout while
  // dragging, a re-apply on release -- the same contract as the zeta amplitudes, since
  // they change the initial condition and not the run.
  for (const id of (opts.icSliders || [])) {
    el(id).oninput = syncLabels;
    el(id).onchange = () => { syncLabels(); applyIC(); };
  }
  // the forcing band changes the fmask AND the shell mode list, which are baked into
  // the grid and into the OU kernel's NS -- a rebuild, not an upload. On release only.
  for (const id of ["rFmin", "rFmax"]) {
    el(id).oninput = syncLabels;
    el(id).onchange = () => { syncLabels(); rebuild(); };
  }
  wireDrawEditor();
  wireTestButton(runSelfTest);
}

// ---------------------------------------------------------------------------
// initial-condition construction (CPU side; uploaded through setICFromReal)
// ---------------------------------------------------------------------------
// Real-space layout, matching the solvers' buffers exactly: index ix*ny + iy in 2D,
// (iz*nx + ix)*ny + iy in 3D. On screen ix runs right and iy runs DOWN (the render
// pass flips v), so a glyph rasterized on an (nx wide, ny tall) canvas at pixel
// (px, py) -> (ix, iy) = (px, py) comes out upright.
//
// WHAT IS PAINTED IS THE ELSASSER POTENTIAL zeta+- (REFINE_PLAN G.1). The Elsasser
// FIELDS are z+- = zhat x grad zeta+- ; the evolved variables are phi = (zeta+ + zeta-)/2
// and psi = (zeta+ - zeta-)/2. Everything below builds zeta+- ; the identifiers stay
// zp / zm for brevity, the labels say zeta.
//
// AMPLITUDE. The knob is the amplitude of the FIELD: "amplitude = a" means
// max |z+-| = max |grad zeta+-| = a. Stored ICs (letters, drawings) are kept at their
// natural unit scale and normalized ONLY at apply time, by icZetaFields -- so the two
// amplitude sliders rescale an unchanged drawing (REFINE_PLAN G.3).

// The two letters of the "letters" IC preset: zeta+ gets the first, zeta- the second.
// Fixed (the free-text input was dropped in the mobile pass -- REFINE_PLAN F.5); both
// apps read it from here.
const IC_LETTERS = "AB";

// ONE perpendicular smoothing length, shared by the letter glyphs and the blob-width
// slider's default (REFINE_PLAN G.2): sigma_letter = Lx/32, a PHYSICAL length, so
// letters at 128 and at 512 carry the same k_perp content and the same chi bookkeeping.
const IC_SIGMA_PERP_FRAC = 1 / 32;
const icSigmaLetter = Lx => IC_SIGMA_PERP_FRAC * Lx;
// the packet / blob length along z is capped at Lz/12, which is what makes the >= 5
// sigma_z packet separation of packetGeom() always fit in the box (REFINE_PLAN G.6).
const IC_SIGMA_Z_FRAC = 1 / 16;         // default: the collision preset's packet length
const IC_SIGMA_Z_MAX_FRAC = 1 / 12;     // hard cap ("increase Lz for longer packets")

// A sigma slider is a FRACTION of its box length. Its range is derived from the shared
// constants here (min = step = max/16), so the default is exactly the constant and the
// two can never drift. Call BEFORE presetBoot -- a preset may set these sliders.
function icSigmaSliderInit(id, frac, maxFrac) {
  const e = el(id), step = maxFrac / 16;
  e.min = String(step); e.max = String(maxFrac); e.step = String(step);
  e.value = String(frac);
  if (Math.abs(frac / step - Math.round(frac / step)) > 1e-9) {
    console.warn(id + ": default " + frac + " is not a multiple of the step " + step);
  }
}
// the live perpendicular blob width (slider, floored at 2 cells) and packet length
function icSigmaPerp() {
  const q = liveParams();
  return icDrawSigma(parseFloat(el("rSigP").value), q.Lx, q.nx);
}
// 3D only (the 2D apps have no rSigZ): the packet / blob length along z
function icSigmaZ() {
  const q = liveParams();
  return q.nz > 1 ? icDrawSigma(parseFloat(el("rSigZ").value), q.Lz, q.nz) : 0;
}

// glyph -> [0,1] coverage mask, or null when there is no usable 2D canvas (node).
// The raster is SHARP by design; the caller always applies icGaussBlur, the exact
// separable periodic gaussian. There used to be a ctx.filter fast path here -- it was
// removed 2026-08-07: iOS Safari reflects the filter attribute while ignoring it at
// rasterization (sharp letters on the phone, the very defect G.2 fixed), and where CSS
// blur does apply it is a box approximation, not the exact gaussian the node gates test.
// One blur implementation, identical on every engine.
function icGlyphRaster(text, nx, ny, cover) {
  let cv = null;
  try { cv = document.createElement("canvas"); } catch (e) { return null; }
  if (!cv || !cv.getContext) return null;
  cv.width = nx; cv.height = ny;
  const c = cv.getContext("2d", { willReadFrequently: true });
  if (!c) return null;
  c.fillStyle = "#000"; c.fillRect(0, 0, nx, ny);
  const target = cover * Math.min(nx, ny);
  // two-pass fit: measure at a trial size, then rescale so the INK box (not the font
  // em box, which is mostly leading) is `cover` of the shorter side
  const fontAt = px => "bold " + px + "px 'Helvetica Neue', Helvetica, Arial, sans-serif";
  let px = Math.max(6, Math.round(target));
  c.font = fontAt(px);
  const m = c.measureText(text);
  const w = m.width || target;
  const h = (m.actualBoundingBoxAscent !== undefined && m.actualBoundingBoxDescent !== undefined)
    ? (m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) : 0.72 * px;
  const big = Math.max(w, h);
  if (big > 0) px = Math.max(6, Math.round(px * target / big));
  c.font = fontAt(px);
  c.textAlign = "center"; c.textBaseline = "middle";
  c.fillStyle = "#fff";
  c.fillText(text, nx / 2, ny / 2);
  let dat;
  try { dat = c.getImageData(0, 0, nx, ny).data; } catch (e) { return null; }
  const out = new Float32Array(nx * ny);
  let peak = 0;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const v = dat[4 * (j * nx + i)] / 255;
      out[i * ny + j] = v;
      if (v > peak) peak = v;
    }
  }
  if (!(peak > 0)) return null;              // nothing was drawn (unknown glyph)
  return out;
}

// Separable PERIODIC gaussian blur, truncated at 3.5 sigma with the kernel renormalized
// (it keeps 0.9995 of the mass). The ONLY letter-smoothing path, on every engine.
//
// It is an exact gaussian at any sigma on purpose: the 3-pass box approximation it
// replaced realized sigma_eff/sigma = 1.118 at 4 px and 1.031 at 16 px, so the SAME
// physical sigma_letter came out 8% wider at 128 than at 512 -- exactly the resolution
// dependence REFINE_PLAN G.2 is about. Cost is O(nx*ny*sigma) and it runs once per IC.
function icGaussBlur(f, nx, ny, sigma) {
  if (!(sigma > 0)) return f;
  const r = Math.max(1, Math.ceil(3.5 * sigma)), w = 2 * r + 1, k = new Float64Array(w);
  let s = 0;
  for (let t = -r; t <= r; t++) { const v = Math.exp(-0.5 * t * t / (sigma * sigma)); k[t + r] = v; s += v; }
  for (let t = 0; t < w; t++) k[t] /= s;
  const a = Float32Array.from(f), b = new Float32Array(nx * ny);
  for (let i = 0; i < nx; i++) {                         // along y (contiguous)
    const row = i * ny;
    for (let j = 0; j < ny; j++) {
      let acc = 0;
      for (let t = -r; t <= r; t++) acc += k[t + r] * a[row + ((((j + t) % ny) + ny) % ny)];
      b[row + j] = acc;
    }
  }
  for (let i = 0; i < nx; i++) {                         // along x (stride ny)
    for (let j = 0; j < ny; j++) {
      let acc = 0;
      for (let t = -r; t <= r; t++) acc += k[t + r] * b[((((i + t) % nx) + nx) % nx) * ny + j];
      a[i * ny + j] = acc;
    }
  }
  return a;
}

// max |grad_perp f| over ONE periodic plane (element offset `off`), 4th-order centred
// differences. This is the estimator icZetaFields inverts, so "amplitude" is defined by
// it; on the smooth (blurred) fields it is used on it agrees with the spectral gradient
// the GPU takes to well under a percent. The dealias mask applied at upload trims the
// achieved value by a little more than that, which is why the knob is approximate.
function icGradMax(f, nx, ny, dx, dy, off) {
  const cx = 1 / (12 * dx), cy = 1 / (12 * dy), o = off | 0;
  let mx = 0;
  for (let i = 0; i < nx; i++) {
    const im2 = o + ((i - 2 + 2 * nx) % nx) * ny, im1 = o + ((i - 1 + nx) % nx) * ny;
    const ip1 = o + ((i + 1) % nx) * ny, ip2 = o + ((i + 2) % nx) * ny, row = o + i * ny;
    for (let j = 0; j < ny; j++) {
      const jm2 = (j - 2 + 2 * ny) % ny, jm1 = (j - 1 + ny) % ny;
      const jp1 = (j + 1) % ny, jp2 = (j + 2) % ny;
      const gx = (f[im2 + j] - 8 * f[im1 + j] + 8 * f[ip1 + j] - f[ip2 + j]) * cx;
      const gy = (f[row + jm2] - 8 * f[row + jm1] + 8 * f[row + jp1] - f[row + jp2]) * cy;
      const g = Math.sqrt(gx * gx + gy * gy);
      if (g > mx) mx = g;
    }
  }
  return mx;
}
// the two numbers icZetaFields needs about a stored potential, WITHOUT mutating it: the
// per-plane mean (a pure gauge for a perpendicular gradient) and max |grad_perp| over
// the whole volume (nz = 1 in 2D). The gradient is blind to the plane means, so they can
// be measured and subtracted in either order.
function icShearStats(f, g) {
  const nrs = g.nx * g.ny, mean = new Float32Array(g.nz);
  let gm = 0;
  for (let k = 0; k < g.nz; k++) {
    const o = k * nrs;
    let s = 0;
    for (let i = 0; i < nrs; i++) s += f[o + i];
    mean[k] = s / nrs;
    gm = Math.max(gm, icGradMax(f, g.nx, g.ny, g.Lx / g.nx, g.Ly / g.ny, o));
  }
  return { gradMax: gm, mean: mean };
}
// THE amplitude step (REFINE_PLAN G.3), and the only place (phi, psi) is formed:
// zero-mean each stored potential plane by plane, scale it so that max |grad_perp zeta|
// is exactly that field's amplitude slider, and map zeta+- -> (phi, psi). The inputs are
// left untouched, so one drawing can be re-applied at any amplitude, any number of times.
//
// BASIS (FEEDBACK_2026-08-10 item 15). Which PAIR the two sliders normalize is a choice,
// and both choices are exact linear maps of the same stored (zeta+, zeta-):
//   undefined / "zeta"  the historical one -- ampP = max |z+| = max |grad zeta+|, likewise
//                       for ampM. Every preset but a phi/psi drawing uses it.
//   "pp"                normalize the COMBINATIONS instead: ampP = max |u| = max |grad phi|
//                       and ampM = max |b| = max |grad psi|. This is the only way to set
//                       the two independently once a drawing carries BOTH phi and psi
//                       strokes -- normalizing zeta+- then rescales an inseparable mixture.
// A drawing with only phi strokes (or only psi ones) comes out numerically identical under
// the two, with the amp lock on: that is the case the old labels happened to describe.
function icZetaFields(zp, zm, g, ampP, ampM, basis) {
  const nrs = g.nx * g.ny, n = g.nz * nrs;
  const phi = new Float32Array(n), psi = new Float32Array(n);
  const pp = basis === "pp";
  let A = zp, B = zm;
  if (pp) {                                  // form (phi, psi) FIRST, then normalize those
    A = new Float32Array(n); B = new Float32Array(n);
    for (let i = 0; i < n; i++) { A[i] = 0.5 * (zp[i] + zm[i]); B[i] = 0.5 * (zp[i] - zm[i]); }
  }
  const P = icShearStats(A, g), M = icShearStats(B, g);
  const kp = P.gradMax > 0 ? ampP / P.gradMax : 0;
  const km = M.gradMax > 0 ? ampM / M.gradMax : 0;
  for (let k = 0; k < g.nz; k++) {
    const o = k * nrs, mp = P.mean[k], mm = M.mean[k];
    for (let i = 0; i < nrs; i++) {
      const a = kp * (A[o + i] - mp), b = km * (B[o + i] - mm);
      phi[o + i] = pp ? a : 0.5 * (a + b);
      psi[o + i] = pp ? b : 0.5 * (a - b);
    }
  }
  return { phi: phi, psi: psi };
}
// Is the drawing being painted in the (phi, psi) basis? -- the ONE predicate behind the
// basis above, the slider labels and the 3D chi line, so those three cannot disagree.
// The paint target is the switch: it is what the user is thinking in, it is only visible
// while the drawing IS the preset (icSyncRows hides #rowDraw otherwise), and it is what
// the caption over the editor already announces.
function icPaintPP() {
  const t = el("selPaint").value;
  return t === "phi" || t === "psi";
}
// ... and only the drawing HAS a paint target, so only the drawing can be in that basis
function icAmpBasis() { return el("selIC").value === "custom" && icPaintPP() ? "pp" : "zeta"; }
// the pair of ZETA amplitudes the sliders imply, whatever basis they are in -- what a
// chi estimate (3D) needs, since chi is written in Elsasser amplitudes. In the (phi, psi)
// basis z+- = u +- b, so co-located strokes of the same shape give max |z+-| =
// |amp_phi +- amp_psi| exactly; for a general drawing it is an estimate, like the k_perp
// that goes into chi beside it.
function icAmpZeta() {
  const a = uiAmp();
  return icAmpBasis() === "pp" ? [a[0] + a[1], Math.abs(a[0] - a[1])] : a;
}

// smooth non-degenerate stand-in when there is no canvas or the glyph drew nothing
function icFallbackBlob(nx, ny) {
  const out = new Float32Array(nx * ny);
  for (let i = 0; i < nx; i++) {
    const cx = Math.sin(2 * Math.PI * i / nx);
    for (let j = 0; j < ny; j++) out[i * ny + j] = cx * Math.sin(2 * Math.PI * j / ny);
  }
  return out;
}

// One Elsasser potential plane from one character: rasterize and gaussian-smooth. The
// scale is left alone -- icZetaFields normalizes. `opts.cover` is the glyph size as a
// fraction of the shorter side (default 0.6), `opts.sigma` the smoothing width as a
// PHYSICAL length (default icSigmaLetter(Lx) = Lx/32), converted to pixels here with a
// 2-px representability floor.
//
// Specifying the blur as a length, not a pixel count (REFINE_PLAN G.2), is what makes
// the letters resolution-independent: the same k_perp content, hence the same chi, at
// 128 and at 512.
//
// The smoothing is not cosmetic. A glyph edge is a step, and the SPECTRAL gradient the
// GPU takes of a step overshoots the finite-difference one badly (~19% for the raw
// raster), so the amplitude knob would mean two different things on the two sides of
// the upload. At sigma >= 2 px the two agree to ~0.1% (node check 2), which is what
// makes "max |zhat x grad zeta+-| = amp" a statement about the displayed field.
function icLetterField(text, nx, ny, Lx, Ly, opts) {
  const o = opts || {};
  const cover = o.cover === undefined ? 0.6 : o.cover;
  const sigL = o.sigma === undefined ? icSigmaLetter(Lx) : o.sigma;
  const sigma = Math.max(2, sigL / (Lx / nx));            // physical length -> pixels
  let f = (text && String(text).length)
    ? icGlyphRaster(String(text).charAt(0), nx, ny, cover) : null;
  if (!f) f = icFallbackBlob(nx, ny);
  else f = icGaussBlur(f, nx, ny, sigma);
  return f;
}

// The "letters" IC as a pair of stored potentials: one glyph per Elsasser potential,
// times a gaussian z-envelope of peak 1 in 3D (`env` = [envPlus, envMinus], each nz
// floats; null in 2D). Shared by both apps -- the only difference between them is that
// pair of envelopes.
// one perpendicular plane times a gaussian z-envelope of peak 1 -> a stored 3D potential.
// `e` is null in 2D (and for a page with no packets), where the plane IS the potential.
function icZExtrude(plane, e, g) {
  if (!e) return plane;
  const nrs = g.nx * g.ny, f = new Float32Array(g.nz * nrs);
  for (let k = 0; k < g.nz; k++) {
    const o = k * nrs, a = e[k];
    for (let i = 0; i < nrs; i++) f[o + i] = a * plane[i];
  }
  return f;
}
function icLetterZeta(g, env) {
  const out = [];
  for (let s = 0; s < 2; s++) {
    const glyph = icLetterField(IC_LETTERS.charAt(s) || IC_LETTERS.charAt(0),
                                g.nx, g.ny, g.Lx, g.Ly);
    out.push(icZExtrude(glyph, env ? env[s] : null, g));
  }
  return { zp: out[0], zm: out[1] };
}

// ---------------------------------------------------------------------------
// the sinusoidal Elsasser pair (FEEDBACK_2026-08-08 item 9) -- 3D only
// ---------------------------------------------------------------------------
// The classic exact-interaction configuration: one counter-propagating packet whose
// perpendicular structure is a single mode in x, the other a single mode in y.
//
// WHAT IS STORED IS THE POTENTIAL. The Elsasser FIELDS are z+- = zhat x grad zeta+- =
// (-d_y zeta+-, d_x zeta+-), so a field that is a pure sine needs a potential that is a
// pure COSINE of the OTHER sign convention:
//     z+ = a+ yhat sin(k1x x)   <-   zeta+ = -cos(k1x x) / k1x     (d_y zeta+ = 0)
//     z- = a- xhat sin(k1y y)   <-   zeta- = +cos(k1y y) / k1y     (d_x zeta- = 0)
// with k1x = 2pi IC_SINE_N / Lx and likewise in y. Each field alone is an exact ideal
// solution (it propagates along z unchanged: a z+ with no z- feels no nonlinearity at
// all). Together they interact through exactly one beat:
//     (z- . grad) z+ = a+ a- k1x sin(k1y y) cos(k1x x) yhat,
// which is why this pair -- and not a pair of blobs -- is the textbook collision.
//
// The 1/k prefactors make the UNNORMALIZED potentials generate unit-amplitude sine
// fields; icZetaFields then rescales by max |grad zeta| = k * (1/k) = 1 per field, so the
// zeta+- amp sliders come out as the field amplitudes a+- exactly, as everywhere else.
// The z-envelope machinery is the letters' (icGaussZ + packetGeom), so sigma_z, chi, the
// packet trackers and the collision timing all keep working unchanged.
const IC_SINE = "sine";
const IC_SINE_N = 1;                     // box-scale mode number of both sinusoids
// the perpendicular gradient scale of the pair, i.e. 1/k1x -- what chiEstimate wants
const icSigmaSine = Lx => Lx / (2 * Math.PI * IC_SINE_N);
function icSinePlanes(g) {
  const nrs = g.nx * g.ny, zp = new Float32Array(nrs), zm = new Float32Array(nrs);
  const kx = 2 * Math.PI * IC_SINE_N / g.Lx, ky = 2 * Math.PI * IC_SINE_N / g.Ly;
  for (let i = 0; i < g.nx; i++) {
    const a = -Math.cos(kx * (i * g.Lx / g.nx)) / kx, row = i * g.ny;
    for (let j = 0; j < g.ny; j++) {
      zp[row + j] = a;
      zm[row + j] = Math.cos(ky * (j * g.Ly / g.ny)) / ky;
    }
  }
  return { zp: zp, zm: zm };
}
function icSineZeta(g, env) {
  const p = icSinePlanes(g);
  return { zp: icZExtrude(p.zp, env ? env[0] : null, g),
           zm: icZExtrude(p.zm, env ? env[1] : null, g) };
}
// the presets whose stored potentials are normalized by the zeta+- amp sliders AND (in
// 3D) carried by a counter-propagating packet envelope. One predicate, so the app's
// applyIC, the chi readout and icSyncRows cannot disagree about what a packet preset is.
const icIsPacketIC = p => (p === "letters" || p === IC_SINE);

// The whole CPU IC path for the two potential-based presets, in one place: pick the
// stored zeta+- pair, then normalize + map through icZetaFields.
//   preset "custom"  the drawing;  IC_SINE  the sinusoids;  anything else  the letters
//   env              3D packet z-envelopes (icLetterZeta / icSineZeta), null in 2D
function icPresetFields(q, preset, ampP, ampM, env) {
  const g = icDrawGrid(q);            // also the geometry record for the letter path
  icEq.on = false; icEq.kh = false;   // only an equilibrium builder turns these back on
  const B = IC_BUILDERS[preset];
  if (B) return B.fields(g);
  const z = preset === "custom" ? { zp: icDraw.zp, zm: icDraw.zm }
          : preset === IC_SINE ? icSineZeta(g, env)
          : icLetterZeta(g, env);
  // only the DRAWING has a paint target, so only it can be in the (phi, psi) basis
  return icZetaFields(z.zp, z.zm, g, ampP, ampM, preset === "custom" ? icAmpBasis() : null);
}

// ---------------------------------------------------------------------------
// equilibrium IC presets: Kelvin-Helmholtz and tearing (REFINE_PLAN J.3)
// ---------------------------------------------------------------------------
// These are NOT potential-amplitude presets. Their knobs are PHYSICAL (U0, b0, psi0, the
// layer width a, the seed amplitude), so they build (phi, psi) directly and skip
// icZetaFields' normalization -- an equilibrium whose amplitude was rescaled to a fixed
// max |grad zeta| would not be the equilibrium anyone asked for. They register exactly
// like the letters and the drawing do, one record per preset:
//   rows    the control-row ids only this preset shows
//   hyper   the hyper exponent it LOCKS the UI to (undefined = the user's choice)
//   src     this equilibrium can be MAINTAINED against resistive decay (J2.3): the
//           #cbEqSrc toggle is live for it, and nothing else ever turns the source on
//   fields  (g) -> {phi, psi}, on the geometry record icDrawGrid returns
// 2D only: the equilibria are 2D objects and the 3D page never lists them in #selIC.
const IC_BUILDERS = {};
function icRegister(name, rec) { IC_BUILDERS[name] = rec; }
// What the island-width chart needs to know about the LIVE equilibrium. `on` and `curv`
// are what islandWidth reads; `a` and `w0` (the initial width) are the record the node
// checks assert the builders against. An equilibrium builder that leaves `on` false --
// KH -- keeps the island chart on its "needs the tearing IC" placeholder. `kh` is the
// parallel flag for the k_y mode chart, which is the KH one: exactly one builder sets
// each, and icPresetFields clears BOTH before any of them runs.
const icEq = { on: false, kh: false, a: 0, curv: 0, w0: 0 };
// a live equilibrium slider (only the page that builds the rows ever calls a builder)
function icEqNum(id, dflt) { const v = parseFloat(el(id).value); return isFinite(v) ? v : dflt; }
// does #selIC offer this preset? -- the ONE test for "this page builds that preset's
// rows", so nothing here ever asks getElementById for an id the page does not have.
// (options is an HTMLOptionsCollection in a browser: index it, do not Array-method it.)
function icHasPreset(name) {
  const o = el("selIC").options;
  for (let i = 0; i < o.length; i++) if (o[i].value === name) return true;
  return false;
}
const icEqPert = () => Math.pow(10, icEqNum("rEqPert", -3));
// is the equilibrium-flux source live? (REFINE_PLAN J2.3) -- only for a preset whose
// builder declares `src`, and only while its toggle is checked. A compile-time constant
// of nlAssemble, so every caller treats a change of it as a rebuild.
function icEqSrcOn() {
  const p = el("selIC").value;
  return !!(icHasPreset(p) && IC_BUILDERS[p] && IC_BUILDERS[p].src && el("cbEqSrc").checked);
}
// 4th-order second derivative of a periodic 1D profile at index i
function icD2(f, i, h) {
  const n = f.length, w = k => f[(((i + k) % n) + n) % n];
  return (-w(-2) + 16 * w(-1) - 30 * w(0) + 16 * w(1) - w(2)) / (12 * h * h);
}
// ln cosh without the overflow: cosh z = e^|z| (1 + e^-2|z|) / 2
function icLogCosh(u) { const z = Math.abs(u); return z + Math.log1p(Math.exp(-2 * z)) - Math.LN2; }
// The DOUBLE shear layer both equilibria build on:
//   f'(x) = A [tanh((x-x1)/a) - tanh((x-x2)/a) - 1],   x1 = Lx/4,  x2 = 3Lx/4
// -- two layers of OPPOSITE sign, which is what periodicity forces, and independent of
// each other while a << |x2 - x1| = Lx/2. What this returns is the POTENTIAL (u_y = d_x phi
// and b_y = d_x psi are both x-derivatives of one), i.e. the analytic antiderivative
//   f(x) = A a [ln cosh((x-x1)/a) - ln cosh((x-x2)/a)] - A (x - Lx/2),
// periodic to O(a e^{-Lx/4a}): the -A x term is exactly what the -A in f' pays for.
function icShearPot(x, Lx, A, a) {
  const x1 = 0.25 * Lx, x2 = 0.75 * Lx;
  return A * a * (icLogCosh((x - x1) / a) - icLogCosh((x - x2) / a)) - A * (x - 0.5 * Lx);
}
// broadcast a 1D x profile (plus an optional k_y seed) into a plane. The seed's y
// dependence is a single ny-long FACTOR, tabulated once instead of being recomputed
// nx times: `yfac` lets a caller hand its own in (TEARNL Phase 1's broadband seed builds
// one out of many modes), and with none the table is exactly the cos(k_y y) this has
// always broadcast -- same expression, same double, so the single-mode plane is bitwise
// what it was before the argument existed.
function icPlaneFromX(prof, seed, g, yfac) {
  const nx = g.nx, ny = g.ny, out = new Float32Array(nx * ny), ky = 2 * Math.PI / g.Ly;
  let yf = yfac;
  if (!yf) { yf = new Float64Array(ny); for (let j = 0; j < ny; j++) yf[j] = Math.cos(ky * j * g.Ly / ny); }
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      out[i * ny + j] = prof[i] + (seed ? seed[i] * yf[j] : 0);
    }
  }
  return out;
}

// KH [20]: u_y is the double-tanh above; B is the same profile scaled by b0, so B lies
// in plane along y. Ideal 2D MHD stabilizes the layer once b0 >= U0 (the shear is then
// slower than the Alfven speed tying the field lines together); dissipation softens that
// threshold rather than removing it. The seed sits on BOTH layers -- they are independent.
// hyper is NOT locked here (J2.5): KH is an ideal instability, so a hyper-dissipation that
// leaves the layer alone and sharpens the secondary structure is a legitimate choice --
// unlike tearing, whose resistive layer IS the physics.
icRegister("kh", {
  rows: ["rowEq", "rowKH"],
  fields: g => {
    const U0 = icEqNum("rEqU0", 1), b0 = icEqNum("rEqB0", 0);
    const a = icEqNum("rEqA", 0.05) * g.Lx, A = icEqPert();
    // the 1D profiles stay fp64 (icPlaneFromX rounds to fp32 on the way into the plane):
    // a second difference of an fp32 profile loses 4 digits to cancellation, and the
    // tearing curvature below is measured off exactly such a profile
    const nx = g.nx, ph = new Float64Array(nx), ps = new Float64Array(nx), sd = new Float64Array(nx);
    const x1 = 0.25 * g.Lx, x2 = 0.75 * g.Lx;
    for (let i = 0; i < nx; i++) {
      const x = i * g.Lx / nx, e1 = (x - x1) / a, e2 = (x - x2) / a;
      ph[i] = icShearPot(x, g.Lx, U0, a);
      ps[i] = icShearPot(x, g.Lx, b0, a);
      sd[i] = A * (Math.exp(-e1 * e1) + Math.exp(-e2 * e2));
    }
    // no resonant surface at x = Lx/2 here (b_y is extremal, not zero), so the island
    // chart stays off for KH (icEq.on stays false) -- its mixing-layer analogue was left
    // out of Phase J. The k_y mode chart is this preset's quantitative view instead.
    icEq.a = a; icEq.kh = true;
    return { phi: icPlaneFromX(ph, sd, g), psi: icPlaneFromX(ps, null, g) };
  }
});

// ---- the broadband-in-k_y seed for the tearing IC (TEARNL Phase 1) ---------
// The single-mode seed below hands the run its answer: there is one k_y, so whatever
// grows is that k_y. Ticking #cbTearBroad seeds modes 1..N together at EQUAL amplitude
// and random phase, and Delta' -- which peaks at some interior k_y and goes negative
// above k_y a = 2.2365 -- picks the winner instead. That is the whole point of the
// `chain` preset, and the reason it is a control on the IC and not a preset flag: the
// selection is worth watching at any (a, L_y) a user can dial.
const icTearBroadOn = () => { const e = el("cbTearBroad"); return !!(e && e.checked); };
// The one seeded source the page has (#nSeed, CTRL_SEED), shared with the OU forcing
// stream so that "same seed, same run" stays a single statement -- Math.random() would
// make this initial condition irreproducible, which is the one thing an initial condition
// may not be. Read at IC-APPLY time, so a change of the box lands on the next apply (an
// equilibrium slider release, an IC or preset switch, Reset, or a rebuild).
function icTearSeed() {
  const e = el("nSeed"), v = e ? parseInt(e.value, 10) : 7;
  return isFinite(v) ? (v | 0) : 7;
}
// How many modes "broadband" is. DERIVED, not a constant: the tearing band of the sech^2
// equilibrium is set by k_y a alone -- Delta' > 0 below k_y a = 2.2365 and negative above,
// measured by bisecting devtools/eqlinear.py's `deltaprime`, and that figure moves by less
// than 0.002 between a = 0.02 L_x and a = 0.2 L_x and between the 2pi and 4pi boxes, so it
// is a property of the profile and not of the geometry. N covers that band with a quarter
// of headroom wherever the a and box sliders are, rather than being a number that happens
// to fit one preset: at the `chain` numbers (a = 0.075 L_x, L_x = 2pi, L_y = 8pi) it comes
// out at exactly 24, and the last mode with gamma > 0 there is 15, so the headroom is real
// and the marginal modes are seeded and simply do not grow -- which is the demonstration.
// Capped at the 2/3 dealias, above which the transform throws the mode away anyway.
const TEAR_KA_MARG = 2.2365;
const TEAR_KA_HEAD = 1.25;
function icTearN(g) {
  const ka1 = (2 * Math.PI / g.Ly) * (icEqNum("rEqA", 0.1) * g.Lx);
  const n = Math.ceil(TEAR_KA_HEAD * TEAR_KA_MARG / Math.max(ka1, 1e-300));
  return Math.max(1, Math.min(n, Math.floor(g.ny / 3)));
}
// The y factor itself: sum_{n=1..N} cos(n k_1 y + phase_n), renormalised so its maximum
// over the grid is exactly 1.
//
// That normalisation is the subtle part and it is forced, not chosen. The x envelope is
// sech^2, which is 1 AT the resonant surface, so the seed's peak perturbed flux there is
// exactly A * max_y |Y(y)| -- and the single-mode Y = cos(k_y y) has max_y |Y| = 1 on the
// grid identically. Normalising to max 1 is therefore the only rule under which the
// #rEqPert slider is the same physical quantity in both branches; leaving the sum raw
// would put ~sqrt(N/2) times the slider in (a factor 3.5 at N = 24, i.e. an order of
// magnitude in the flux), and dividing by N would put ~sqrt(2 ln N / N) times it in
// (a factor 5 the other way). It rescales all N modes by ONE number, so the seed spectrum
// stays flat -- which is what "let Delta' select" requires.
//
// It does NOT make the island chart's W_0 meaningful, and the builder below does not
// pretend otherwise (see icEq.on): a normalisation fixes the total, and W = 4 sqrt(psitilde
// / |psi''|) is about one mode's psitilde.
function icTearYFac(g, N) {
  const ny = g.ny, y = new Float64Array(ny), ph = new Float64Array(N);
  const r = mulberry32(icTearSeed());
  for (let n = 0; n < N; n++) ph[n] = 2 * Math.PI * r();
  const k1 = 2 * Math.PI / g.Ly;
  for (let j = 0; j < ny; j++) {
    const yy = j * g.Ly / ny;
    let s = 0;
    for (let n = 1; n <= N; n++) s += Math.cos(n * k1 * yy + ph[n - 1]);
    y[j] = s;
  }
  let m = 0;
  for (let j = 0; j < ny; j++) { const v = Math.abs(y[j]); if (v > m) m = v; }
  if (m > 0) for (let j = 0; j < ny; j++) y[j] /= m;
  return y;
}

// Tearing [19]: psi_eq = psi0 sech^2((x - Lx/2)/a) (Numata / Loureiro style -- net-flux
// free and exponentially periodic for a << Lx), phi_eq = 0. b_y = psi_eq' vanishes at
// x = Lx/2, so that is the resonant surface of EVERY k_y mode and the line the island
// chart reads. The seed perturbs psi at k_y = 2pi/Ly with the same even-in-x envelope, so
// its value AT the surface is exactly the slider: psitilde(x_s) = A, and the initial
// island width is 4 sqrt(A/|psi_eq''|). With #cbTearBroad ticked the y factor is the
// many-mode one above instead, normalised so that sentence still reads true.
icRegister("tearing", {
  rows: ["rowEq", "rowTear"], hyper: 1, src: true,
  fields: g => {
    const psi0 = icEqNum("rEqPsi0", 1.65);
    const a = icEqNum("rEqA", 0.1) * g.Lx, A = icEqPert();
    const broad = icTearBroadOn();
    const nx = g.nx, ps = new Float64Array(nx), sd = new Float64Array(nx);
    for (let i = 0; i < nx; i++) {
      const s2 = 1 / Math.pow(Math.cosh((i * g.Lx / nx - 0.5 * g.Lx) / a), 2);
      ps[i] = psi0 * s2;
      sd[i] = A * s2;
    }
    // MEASURED curvature on the resonant surface: a 4th-order second difference of the
    // equilibrium profile ON THE GRID THE RUN USES, not the analytic -2 psi0/a^2, so the
    // chart divides by the number the discretization actually has. (fp64 on the profile:
    // a second difference cancels four digits, which fp32 storage would not survive.)
    // The BROADBAND branch leaves icEq.on false, so the island chart keeps its "needs the
    // tearing IC preset" placeholder rather than quoting a W it cannot support: W(t) is
    // 4 sqrt(psitilde / |psi''|) for ONE reconnecting mode measured off max - min of psi
    // on the resonant line, and with 24 modes seeded together that extremum belongs to
    // whichever island is largest at that instant -- and after the first merger not even
    // the count is fixed. Better a placeholder than a plausible wrong number; the spectrum
    // and the picture are what `chain` is read on.
    icEq.on = !broad; icEq.a = a;
    icEq.curv = Math.abs(icD2(ps, Math.round(0.5 * nx), g.Lx / nx));
    icEq.w0 = (!broad && icEq.curv > 0) ? 4 * Math.sqrt(A / icEq.curv) : 0;
    const yfac = broad ? icTearYFac(g, icTearN(g)) : null;
    return { phi: new Float32Array(nx * g.ny), psi: icPlaneFromX(ps, sd, g, yfac) };
  }
});

// ---- island width (REFINE_PLAN J.4) ---------------------------------------
// psi along the resonant line, from the b_x = -d_y psi line the CUT kernel already
// produces: one spectral integration. The line is periodic and band-limited, so the DFT
// inversion is exact where a quadrature rule would be second order; the k = 0 gauge is
// dropped because only max - min is used. O(ny^2 / 2) and only while an island card
// exists, at the cut throttle (~10 Hz).
function icLineIntegrate(bx, n, Ly) {
  const out = new Float64Array(n), k0 = 2 * Math.PI / Ly, half = n >> 1;
  for (let m = 1; m < half; m++) {           // the Nyquist mode of b_x cannot be integrated
    let cr = 0, ci = 0;                      // consistently (and the dealias has killed it)
    for (let j = 0; j < n; j++) {
      const th = 2 * Math.PI * m * j / n;
      cr += bx[j] * Math.cos(th); ci -= bx[j] * Math.sin(th);
    }
    const k = m * k0, pr = -ci / (n * k), pi = cr / (n * k);      // psihat = i bhat / k
    for (let j = 0; j < n; j++) {
      const th = 2 * Math.PI * m * j / n;
      out[j] += 2 * (pr * Math.cos(th) - pi * Math.sin(th));      // + its conjugate partner
    }
  }
  return out;
}
// THE island-width formula. Near the resonant surface
//   psi ~ psi_s + 1/2 psi'' (x - x_s)^2 + psitilde cos(k y),
// so the separatrix half-width w obeys 1/2 |psi''| w^2 = 2 psitilde and the full width is
//   W = 2w = 4 sqrt(psitilde / |psi''|),   with  2 psitilde = psi_X - psi_O on the line.
function islandWidth(vals, ny, Ly) {
  if (!icEq.on || !(icEq.curv > 0)) return NaN;
  const psi = icLineIntegrate(vals.subarray(2 * ny, 3 * ny), ny, Ly);
  let lo = Infinity, hi = -Infinity;
  for (let j = 0; j < ny; j++) { const v = psi[j]; if (v < lo) lo = v; if (v > hi) hi = v; }
  const d = hi - lo;
  return d > 0 ? 4 * Math.sqrt(0.5 * d / icEq.curv) : 0;
}
function islandPush(t, vals, ny, Ly) {
  const w = islandWidth(vals, ny, Ly), H = islandHist;
  if (!isFinite(w)) return;
  if (H.t.length && !(t > H.t[H.t.length - 1])) return;       // paused: no duplicate t
  if (H.t.length >= HIST_MAX) {
    for (const a of [H.t, H.w]) { let k = 0; for (let i = 0; i < a.length; i += 2) a[k++] = a[i]; a.length = k; }
  }
  H.t.push(t); H.w.push(w);
}

// ---- the k_y = 2pi/Ly mode amplitude (KH linear stage) --------------------
// ONE DFT coefficient, not a transform: m = 1 of an ny-point row, accumulated in fp64.
//   A = (2/ny) |sum_j f_j exp(-2 pi i j / ny)|
// normalized so that a row f_j = A cos(2 pi j / ny + phase) returns exactly A, for any
// phase, with any constant offset and any other mode present. O(ny), and only while a
// mode card exists, at the cut throttle (~10 Hz).
function modeAmp1(f, off, n) {
  let cr = 0, ci = 0;
  for (let j = 0; j < n; j++) {
    const th = 2 * Math.PI * j / n;
    cr += f[off + j] * Math.cos(th); ci -= f[off + j] * Math.sin(th);
  }
  return 2 * Math.hypot(cr, ci) / n;
}
// The two rows of the cut stack (u_x, u_y, b_x, b_y) that carry NO equilibrium: it is
// y-independent with u and B along y, so u_x and b_x are pure perturbation -- which is
// the whole reason this chart can show a growth the energy trace cannot.
function modeAmps(vals, ny) {
  return { u: modeAmp1(vals, 0, ny), b: modeAmp1(vals, 2 * ny, ny) };
}
// Least-squares slope of ln(y) vs t over the trailing MODE_FIT_DT of sim-time -- ONE
// implementation, shared by the KH mode chart and the island-width chart (the two log-y
// growth traces). Only finite, positive samples count (a log has nothing to say about the
// others); the window must span time, must RISE by `riseMin` (a flat, decaying, jittering
// or saturated trace has no growth rate to quote), and the fit must actually describe it
// (R^2 >= MODE_FIT_R2 -- a straddle of two stages is not a rate). Otherwise NaN, and the
// legend simply omits the readout.
function fitLogSlope(ts, as, riseMin) {
  const t = [], y = [];
  let tl = NaN;
  for (let i = Math.min(ts.length, as.length) - 1; i >= 0; i--) {
    if (!(isFinite(ts[i]) && isFinite(as[i]) && as[i] > 0)) continue;
    if (!isFinite(tl)) tl = ts[i];
    else {
      // The window is MODE_FIT_DT of SIM time, but the trace is sampled on a WALL clock
      // (the cut readback's ~10 Hz throttle), so the samples it contains are
      // 100 / (sim-units per second) -- and a small grid with a large dt runs sim-time so
      // fast that 10 t-units hold fewer than the 4 samples a slope needs. `tearing` at
      // selRes 256 is exactly that case: ~19 sim-units/s, ~6 samples, and on a quicker
      // machine 3 and the legend goes blank mid-linear-stage for no physical reason.
      // So the window is "MODE_FIT_DT, or as far back as it takes to collect
      // MODE_FIT_N samples, whichever is longer" -- capped at MODE_FIT_DT_MAX. Widening
      // is safe because it cannot fake a rate: the R^2 gate below is what rejects a
      // window straddling two stages, and it does not care how the window was chosen.
      const span = tl - ts[i];
      if (span > MODE_FIT_DT_MAX) break;
      if (span > MODE_FIT_DT && t.length >= MODE_FIT_N) break;
    }
    t.unshift(ts[i]); y.unshift(Math.log(as[i]));
  }
  const m = t.length;
  if (m < 4 || !(t[m - 1] > t[0]) || !(y[m - 1] - y[0] >= riseMin)) return NaN;
  let mt = 0, my = 0;
  for (let i = 0; i < m; i++) { mt += t[i]; my += y[i]; }
  mt /= m; my /= m;
  let sxx = 0, sxy = 0, syy = 0;                       // centered: no cancellation risk
  for (let i = 0; i < m; i++) {
    const dt = t[i] - mt, dy = y[i] - my;
    sxx += dt * dt; sxy += dt * dy; syy += dy * dy;
  }
  const g = sxx > 0 ? sxy / sxx : NaN;
  return g > 0 && sxy * sxy >= MODE_FIT_R2 * sxx * syy ? g : NaN;
}
// the KH chart's gamma: the mode amplitude A grows as e^(gamma t), so the slope IS gamma
// -- the number to compare with the linear reference gamma = 0.267 U0 k_y.
function modeFitGamma(ts, as) { return fitLogSlope(ts, as, MODE_FIT_RISE); }
// the island chart's gamma: in the linear tearing stage psitilde ~ e^(gamma t) and
// W = 4 sqrt(psitilde/|psi''|), so W ~ e^(gamma t / 2) and gamma is TWICE the slope --
// the number to compare with the linear reference 0.0287 quoted in the preset hint.
function islandFitGamma(ts, ws) { return 2 * fitLogSlope(ts, ws, ISLAND_FIT_RISE); }
function modePush(t, vals, ny) {
  const a = modeAmps(vals, ny), H = modeHist;
  if (!(a.u > 0) || !isFinite(a.u) || !isFinite(a.b)) return;   // the log axis needs A_u > 0
  if (H.t.length && !(t > H.t[H.t.length - 1])) return;         // paused: no duplicate t
  if (H.t.length >= HIST_MAX) {
    for (const q of [H.t, H.u, H.b]) { let k = 0; for (let i = 0; i < q.length; i += 2) q[k++] = q[i]; q.length = k; }
  }
  H.t.push(t); H.u.push(a.u); H.b.push(a.b);
}

// The equilibrium presets' own control rows, as controlsBuild spec fragments -- next to
// the builders that read them. The 2D page passes them through ctrlGrpIC({extra: ...});
// the 3D page passes neither these nor the presets, so its #selIC never offers them.
const CTRL_ROWS_EQ = [
  { id: "rowEq", hide: true, items: [
    { k: "lab", t: "a / L<sub>x</sub>" },
    // step 0.00125 and min 0.0125, both down from (0.005, 0.02) for TEARNL: `chain` in the
    // 8pi x 8pi box needs a / Lx = 0.4712 / 8pi = 0.01875, which is below the old floor and
    // off the old grid. Every preset value -- 0.01875, 0.05, 0.075, 0.1 -- is a multiple of
    // 0.00125, checked. A slider that SNAPPED 0.01875 to 0.02 would not merely be 7% off:
    // it moves the fastest mode from 6 to 5 and flattens the band further (gamma 0.2704 vs
    // 0.2694 for its neighbour), i.e. it would change what the preset demonstrates.
    { k: "rng", id: "rEqA", min: 0.0125, max: 0.2, step: 0.00125, v: 0.1,
      ti: "equilibrium layer width, as a fraction of Lx" }, { k: "val", id: "vEqA" },
    { k: "lab", t: "seed" },
    { k: "rng", id: "rEqPert", min: -6, max: -1, step: 0.1, v: -3,
      ti: "seed amplitude at the resonant surface (log10). Single-mode: the amplitude of "
        + "the k_y = 2pi/Ly perturbation. With `broadband seed` ticked: the PEAK over y of "
        + "the many-mode seed, which is the same physical quantity" },
    { k: "val", id: "vEqPert" }
  ] },
  { id: "rowKH", hide: true, items: [
    { k: "lab", t: "U&#8320;" },
    { k: "rng", id: "rEqU0", min: 0, max: 2, step: 0.05, v: 1, ti: "shear-flow amplitude" },
    { k: "val", id: "vEqU0" },
    { k: "lab", t: "b&#8320;" },
    { k: "rng", id: "rEqB0", min: 0, max: 2, step: 0.05, v: 0,
      ti: "in-plane field amplitude; KH is ideally suppressed once b0 >= U0" },
    { k: "val", id: "vEqB0" }
  ] },
  { id: "rowTear", hide: true, items: [
    { k: "lab", t: "&psi;&#8320;" },
    { k: "rng", id: "rEqPsi0", min: 0.1, max: 4, step: 0.05, v: 1.65,
      ti: "flux-function amplitude: psi_eq = psi0 sech^2((x - Lx/2)/a)" },
    { k: "val", id: "vEqPsi0" },
    { k: "cbl", id: "cbEqSrc", t: "maintain equilibrium flux", v: true,
      ti: "add the static source S = -eta grad^2 psi_eq, which holds the equilibrium "
        + "against its own resistive decay (rebuilds the solver)" },
    // TEARNL Phase 1. Unlike cbEqSrc beside it this is NOT a rebuild: eqsrc is a
    // compile-time constant of nlAssemble, whereas this only changes the real-space
    // (phi, psi) handed to setICFromReal -- same grid, same kernels, same everything the
    // solver was built from. It is wired through icSliders below, i.e. it re-applies the
    // IC on change exactly as the equilibrium sliders one row up do, which is the closer
    // precedent for a control that edits the initial condition and not the run.
    { k: "cbl", id: "cbTearBroad", t: "broadband seed", v: false,
      ti: "seed every k_y from 1 up to the marginal one at equal amplitude and random "
        + "phase, and let Delta' select the mode, instead of seeding k_y = 2pi/Ly alone" }
  ] },
  { k: "hintdiv", id: "vEqInfo" }
];
// the equilibrium presets' IC-shape controls, in the order wireCommonControls should wire
// them (each re-applies the IC on release, exactly like the zeta amplitudes). The last is
// a checkbox rather than a slider; it wants the identical contract (its `oninput` is a
// harmless second label sync) and the alternative was a second wiring path saying the
// same thing.
const EQ_SLIDERS = ["rEqA", "rEqPert", "rEqU0", "rEqB0", "rEqPsi0", "cbTearBroad"];
// their readouts, plus the one-line derived summary (max |b|, the initial island width)
function syncEqLabels() {
  if (!icHasPreset("tearing")) return;
  el("vEqA").textContent = icEqNum("rEqA", 0.1).toFixed(3);
  el("vEqPert").textContent = icEqPert().toExponential(1);
  el("vEqU0").textContent = icEqNum("rEqU0", 1).toFixed(2);
  el("vEqB0").textContent = icEqNum("rEqB0", 0).toFixed(2);
  el("vEqPsi0").textContent = icEqNum("rEqPsi0", 1.65).toFixed(2);
  const p = el("selIC").value, q = liveParams(), a = icEqNum("rEqA", 0.1) * q.Lx;
  let s = "";
  if (p === "tearing") {
    const c = 2 * icEqNum("rEqPsi0", 1.65) / (a * a);
    s = "a = " + a.toPrecision(3) + "  max|b| = " + (0.7698 * icEqNum("rEqPsi0", 1.65) / a).toPrecision(3) +
        "  k_y a = " + (2 * Math.PI / q.Ly * a).toPrecision(3) +
        // W(0) is a single-mode number (icEq.w0 above), so the broadband seed says what it
        // actually did instead of quoting one -- and the mode count is the derived N, so a
        // user dragging `a / Lx` can watch the band it covers move.
        (icTearBroadOn()
          ? "  seed: k_y modes 1&ndash;" + icTearN(q) + ", flat, random phase"
          : "  W(0) &asymp; " + (4 * Math.sqrt(icEqPert() / c)).toPrecision(3));
  } else if (p === "kh") {
    const U0 = icEqNum("rEqU0", 1), b0 = icEqNum("rEqB0", 0);
    s = "a = " + a.toPrecision(3) + "  k_y a = " + (2 * Math.PI / q.Ly * a).toPrecision(3) +
        "  b&#8320;/U&#8320; = " + (U0 > 0 ? (b0 / U0).toFixed(2) : "&infin;") +
        (U0 > 0 && b0 >= U0 ? " &mdash; ideally suppressed" : "");
  }
  const e = el("vEqInfo");
  e.innerHTML = s; e.style.display = s ? "" : "none";
}

// periodic gaussian z-envelope, peak normalized to exactly 1 ON THE GRID (so a packet
// whose centre falls between planes still has the requested amplitude).
function icGaussZ(nz, Lz, z0, sigma) {
  const e = new Float32Array(nz);
  let mx = 0;
  for (let k = 0; k < nz; k++) {
    let d = k * Lz / nz - z0;
    d -= Lz * Math.round(d / Lz);            // minimum image
    e[k] = Math.exp(-0.5 * (d * d) / (sigma * sigma));
    if (e[k] > mx) mx = e[k];
  }
  if (mx > 0) for (let k = 0; k < nz; k++) e[k] /= mx;
  return e;
}

// ---------------------------------------------------------------------------
// counter-propagating packets: placement and chi (3D letters / the collision preset)
// ---------------------------------------------------------------------------
// The packets sit symmetrically about the midplane, separated by
//   s = clamp(5*sigma_z, 3Lz/8, Lz/2),
// so they are ALWAYS at least 5 sigma_z apart -- on the direct side by construction and
// on the wrap-around side because s <= Lz/2 <= Lz - s (REFINE_PLAN G.6; the slider's
// Lz/12 cap is what keeps 5*sigma_z <= Lz/2). At the default sigma_z = Lz/16 the floor
// binds and the placement is the historical 11Lz/16 / 5Lz/16.
//
// zeta+ travels toward SMALLER z and zeta- toward LARGER z (see the app's propagator
// note), so putting zeta+ ABOVE the midplane makes them meet head-on at z = Lz/2 at
// t = (s/2)/v_A, before the wrap-around collision at z = 0.
const PACKET_SEP_MIN = 3 / 8;
function packetGeom(Lz, sigmaZ) {
  const s = Math.min(0.5 * Lz, Math.max(PACKET_SEP_MIN * Lz, 5 * sigmaZ));
  return { sep: s, zPlus: 0.5 * (Lz + s), zMinus: 0.5 * (Lz - s), tColl: 0.5 * s };
}
// chi+- = a-+ * kbar_perp * sigma_z / v_A  with  kbar_perp = 1/sigma_perp  and v_A = 1
// (REFINE_PLAN G.6). A gaussian smoothing length IS the gradient scale of the smoothed
// structure, so the perpendicular wavenumber of the FIELD z+- = zhat x grad zeta+- is
// ~1/sigma_perp -- sigma_letter for the letters, the blob-width slider for a drawing,
// which is exactly why G.2 makes those the same constant by default. The nonlinearity
// felt by one packet is set by the amplitude of the OTHER one, hence the swapped
// indices. An ESTIMATE: the packets are not single modes.
function chiEstimate(sigmaPerp, sigmaZ, ampP, ampM) {
  const k = sigmaZ / sigmaPerp;
  return [ampM * k, ampP * k];
}

// ---------------------------------------------------------------------------
// custom ("drawn") IC: the gaussian-blob editor
// ---------------------------------------------------------------------------
// Everything here is CPU-only. The editor keeps the two Elsasser POTENTIALS zeta+ and
// zeta- at the live grid size, paints periodic gaussian blobs into them, previews them
// on an ordinary 2D canvas, and hands them to icZetaFields on save -- the same path the
// letters take. No GPU work happens while editing.
//
// AMPLITUDE. Blobs are deposited at UNIT amplitude (REFINE_PLAN G.3): the sliders scale
// the finished drawing in icZetaFields, so changing one and pressing Reset genuinely
// rescales instead of adding a differently-scaled blob. For f = P exp(-r^2/2 s^2),
// |grad f| = P (r/s^2) exp(-r^2/2 s^2) peaks at r = s with the value P/(s sqrt(e)), so
// the potential peak that realizes an amplitude `a` is P = a s sqrt(e) -- with a = 1
// here, which keeps a wide blob and a narrow one comparable in the stored drawing.
function icBlobPeak(amp, sigma) { return amp * sigma * Math.sqrt(Math.E); }

// minimum-image separation on a periodic axis of length L
function icWrapDelta(d, L) { return d - L * Math.round(d / L); }

// Blobs are deposited PERIODICALLY (minimum image, exactly like icGaussZ) and truncated
// at CUT sigma: exp(-CUT^2/2) = 3.7e-6 at CUT = 5, i.e. the integral keeps 1 - 4e-6 of a
// gaussian. When the window is as wide as the axis the loop simply covers every index
// once, still with the minimum-image distance -- so a blob at the box edge is identical
// (up to translation) to one in the middle.
const IC_BLOB_CUT = 5;

// tabulate the periodic gaussian factor along ONE axis: the grid indices within CUT
// sigma of x0 (or every index once, when that window is not shorter) and their
// exp(-d^2/2 sigma^2) with d the minimum-image separation.
function icBlobAxis(n, L, x0, sigma) {
  const d0 = L / n, inv = 0.5 / (sigma * sigma);
  const h = Math.ceil(IC_BLOB_CUT * sigma / d0);
  const idx = [], fac = [];
  if (2 * h + 1 >= n) {
    for (let i = 0; i < n; i++) idx.push(i);
  } else {
    const i0 = Math.round(x0 / d0);
    for (let t = -h; t <= h; t++) idx.push((((i0 + t) % n) + n) % n);
  }
  for (let p = 0; p < idx.length; p++) {
    const d = icWrapDelta(idx[p] * d0 - x0, L);
    fac.push(Math.exp(-inv * d * d));
  }
  return { idx: idx, fac: fac };
}

// add w * peak * exp(-r^2 / 2 sigma^2) into one perpendicular plane of a real-space
// array (ix*ny + iy layout; `off` is the element offset of the plane, 0 in 2D). The
// gaussian is separable, so the two axis tables above are all it takes.
function icBlobAddPlane(f, nx, ny, Lx, Ly, x0, y0, sigma, peak, off, w) {
  const a = w === undefined ? 1 : w;
  if (!(sigma > 0) || !isFinite(peak) || a === 0) return;
  const X = icBlobAxis(nx, Lx, x0, sigma), Y = icBlobAxis(ny, Ly, y0, sigma);
  const A = a * peak, o = off | 0;
  for (let p = 0; p < X.idx.length; p++) {
    const row = o + X.idx[p] * ny, cx = A * X.fac[p];
    for (let q = 0; q < Y.idx.length; q++) f[row + Y.idx[q]] += cx * Y.fac[q];
  }
}

// 3D: the same blob times a periodic gaussian z-envelope of peak 1 centred on z0
// (which the UI always puts exactly on the displayed plane, so the peak is exact).
function icBlobAdd3D(f, g, x0, y0, z0, sigma, sigmaZ, peak) {
  const nz = g.nz;
  if (!(nz > 1) || !(sigmaZ > 0)) {
    icBlobAddPlane(f, g.nx, g.ny, g.Lx, g.Ly, x0, y0, sigma, peak, 0, 1);
    return;
  }
  const Z = icBlobAxis(nz, g.Lz, z0, sigmaZ), nrs = g.nx * g.ny;
  for (let p = 0; p < Z.idx.length; p++) {
    icBlobAddPlane(f, g.nx, g.ny, g.Lx, g.Ly, x0, y0, sigma, peak,
                   Z.idx[p] * nrs, Z.fac[p]);
  }
}

// ---- editor state ---------------------------------------------------------
// zp / zm are the accumulated Elsasser POTENTIALS at the CURRENT grid; they are dropped
// (and the drawing is lost) whenever the grid changes, which is exactly when their
// layout stops meaning anything. `cfg` is the per-app hook set (icDrawWire); `snap` is
// the copy taken on entering the editor, which "cancel" restores; `plane` is the z plane
// being painted (3D, the editor's own slider -- no display card is involved).
const icDraw = {
  on: false, zp: null, zm: null, key: "", n: 0, has: false, snap: null, plane: 0,
  cfg: null, wired: false, down: false, neg: false, last: null, pending: false,
  cv: null, cap: null, plSl: null          // the editor view's elements (icEditBuild)
};
function icDrawKey(q) { return q.nx + "x" + q.ny + "x" + (q.nz || 1); }
// (re)allocate for the live grid; returns the geometry the rest of the editor uses
function icDrawGrid(q) {
  const g = { nx: q.nx, ny: q.ny, nz: q.nz || 1, Lx: q.Lx, Ly: q.Ly, Lz: q.Lz || 0 };
  const key = icDrawKey(g);
  if (icDraw.key !== key || !icDraw.zp) {
    icDraw.key = key; icDraw.n = g.nx * g.ny * g.nz;
    icDraw.zp = new Float32Array(icDraw.n);
    icDraw.zm = new Float32Array(icDraw.n);
    icDraw.has = false;
  }
  return g;
}
function icDrawClear() {
  if (icDraw.zp) { icDraw.zp.fill(0); icDraw.zm.fill(0); }
  icDraw.has = false; icDraw.last = null;
}
// a blob in the TARGET field, expressed in zeta+- : "phi" is a blob in both, "psi" one
// of each sign, "zp"/"zm" just themselves.
const IC_TARGETS = { zp: [1, 0], zm: [0, 1], phi: [1, 1], psi: [1, -1] };
function icDrawBlob(q, target, x0, y0, iz, sigma, sigmaZ, sign) {
  const g = icDrawGrid(q), w = IC_TARGETS[target] || IC_TARGETS.zp;
  const peak = (sign === undefined ? 1 : sign) * icBlobPeak(1, sigma);   // UNIT amplitude
  const z0 = g.nz > 1 ? Math.max(0, Math.min(g.nz - 1, iz | 0)) * g.Lz / g.nz : 0;
  for (const [f, c] of [[icDraw.zp, w[0]], [icDraw.zm, w[1]]]) {
    if (c === 0) continue;
    if (g.nz > 1) icBlobAdd3D(f, g, x0, y0, z0, sigma, sigmaZ, c * peak);
    else icBlobAddPlane(f, g.nx, g.ny, g.Lx, g.Ly, x0, y0, sigma, c * peak, 0, 1);
  }
  icDraw.has = true;
}
// the plane the preview shows: the target field on the displayed z slice
function icDrawPlane(q, target, iz) {
  const g = icDrawGrid(q), nrs = g.nx * g.ny;
  const o = g.nz > 1 ? Math.max(0, Math.min(g.nz - 1, iz | 0)) * nrs : 0;
  const w = IC_TARGETS[target] || IC_TARGETS.zp;
  // phi = (z+ + z-)/2, psi = (z+ - z-)/2 -> the same weights, halved, for the combinations
  const h = (w[0] && w[1]) ? 0.5 : 1;
  const out = new Float32Array(nrs);
  for (let i = 0; i < nrs; i++) out[i] = h * (w[0] * icDraw.zp[o + i] + w[1] * icDraw.zm[o + i]);
  return out;
}
// sigma floors: anything under ~2 cells is not representable (the 2/3 dealias eats it)
function icDrawSigma(frac, L, n) { return Math.max(frac * L, 2 * L / n); }

// ---- preview --------------------------------------------------------------
// icDraw.cv is the editor view's own plain 2D canvas, laid out exactly like a display
// card's: grid point (ix, iy) is image pixel (ix, iy), i.e. row iy=0 at the TOP, matching
// the render pass's v = 1 - uv.y flip and the arrow overlay. The nx-by-ny image is drawn
// through an offscreen canvas and scaled up by drawImage, so the smoothing matches the
// GPU path's linear sampler.
const icPrev = { cx: null, off: null, ox: null, w: 0, h: 0 };
function icDrawCtx() {
  if (icPrev.cx) return icPrev.cx;
  const cv = icDraw.cv;
  if (!cv || !cv.getContext) return null;
  const dpr = Math.min(2, (typeof window !== "undefined" && window.devicePixelRatio) || 1);
  icPrev.w = cv.width || 512; icPrev.h = cv.height || 512;
  cv.width = Math.round(icPrev.w * dpr); cv.height = Math.round(icPrev.h * dpr);
  cv.style.width = "100%"; cv.style.height = "100%";
  const c = cv.getContext("2d");
  if (!c) return null;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  icPrev.cx = c;
  return c;
}
function icDrawPreview() {
  const cfg = icDraw.cfg, c = icDrawCtx();
  if (!cfg || !c) return;
  const q = cfg.params && cfg.params();
  if (!q) return;
  const g = icDrawGrid(q);
  const v = icDrawPlane(q, cfg.target(), icDraw.plane);
  let mx = 0;
  for (let i = 0; i < v.length; i++) { const a = Math.abs(v[i]); if (a > mx) mx = a; }
  if (!icPrev.off) {
    try { icPrev.off = document.createElement("canvas"); } catch (e) { icPrev.off = null; }
    if (!icPrev.off || !icPrev.off.getContext) { icPrev.off = null; return; }
  }
  if (icPrev.off.width !== g.nx || icPrev.off.height !== g.ny) {
    icPrev.off.width = g.nx; icPrev.off.height = g.ny;
    icPrev.ox = icPrev.off.getContext("2d");
  }
  if (!icPrev.ox) icPrev.ox = icPrev.off.getContext("2d");
  if (!icPrev.ox) return;
  const im = icPrev.ox.createImageData(g.nx, g.ny), d = im.data;
  const inv = mx > 0 ? 1 / mx : 0;
  // exactly the colorize kernel's signed mapping, through the same colormap table,
  // in the first display card's colormap (so the preview matches what Run will show)
  const pc = primaryCard(), which = pc ? pc.cmap() : 0;
  for (let ix = 0; ix < g.nx; ix++) {
    for (let iy = 0; iy < g.ny; iy++) {
      const t = 0.5 * (Math.max(-1, Math.min(1, v[ix * g.ny + iy] * inv)) + 1);
      const rgb = cmapRGB(which, t);
      const p = 4 * (iy * g.nx + ix);
      d[p] = 255 * rgb[0]; d[p + 1] = 255 * rgb[1]; d[p + 2] = 255 * rgb[2];
      d[p + 3] = 255;
    }
  }
  icPrev.ox.putImageData(im, 0, 0);
  c.clearRect(0, 0, icPrev.w, icPrev.h);
  c.drawImage(icPrev.off, 0, 0, icPrev.w, icPrev.h);
}
function icDrawPreviewSoon() {
  if (icDraw.pending) return;
  icDraw.pending = true;
  const run = () => { icDraw.pending = false; icDrawPreview(); };
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(run);
  else run();
}

// ---- pointer -> grid ------------------------------------------------------
// getBoundingClientRect is in CSS pixels, so the responsive width (and any devicePixel
// ratio) divides out. The display puts grid point i at screen fraction (i + 0.5)/n --
// texel centres, as the linear sampler sees them, and as drawArrows places its cells --
// so the continuous grid coordinate under the cursor is u*n - 0.5 and the nearest grid
// point is round(u*n - 0.5) = floor(u*n) inside the canvas.
function icDrawPos(ev, q) {
  const cv = icDraw.cv;
  if (!cv || !cv.getBoundingClientRect) return null;
  const r = cv.getBoundingClientRect();
  if (!(r.width > 0) || !(r.height > 0)) return null;
  const u = (ev.clientX - r.left) / r.width, v = (ev.clientY - r.top) / r.height;
  const dx = q.Lx / q.nx, dy = q.Ly / q.ny;
  let x0 = (u * q.nx - 0.5) * dx, y0 = (v * q.ny - 0.5) * dy;
  x0 -= q.Lx * Math.floor(x0 / q.Lx); y0 -= q.Ly * Math.floor(y0 / q.Ly);
  return { x0: x0, y0: y0,
           ix: ((Math.round(u * q.nx - 0.5) % q.nx) + q.nx) % q.nx,
           iy: ((Math.round(v * q.ny - 0.5) % q.ny) + q.ny) % q.ny };
}

// one deposit from a pointer event (down or drag). Successive drag samples closer
// together than sigma/2 are dropped, so holding the mouse still does not pile up.
function icDrawStroke(ev) {
  const cfg = icDraw.cfg, q = cfg && cfg.params && cfg.params();
  if (!q) return;
  const p = icDrawPos(ev, q);
  if (!p) return;
  const sigma = icSigmaPerp();
  if (icDraw.last) {
    const dx = icWrapDelta(p.x0 - icDraw.last.x, q.Lx), dy = icWrapDelta(p.y0 - icDraw.last.y, q.Ly);
    if (Math.hypot(dx, dy) < 0.5 * sigma) return;
  }
  icDraw.last = { x: p.x0, y: p.y0 };
  const sz = (q.nz || 1) > 1 ? icSigmaZ() : 0;
  // sign: the checkbox XOR the right button. The MAGNITUDE is 1 -- the amplitude
  // sliders scale the finished drawing (REFINE_PLAN G.3).
  const s = ((cfg.neg && cfg.neg()) !== icDraw.neg) ? -1 : 1;
  icDrawBlob(q, cfg.target(), p.x0, p.y0, icDraw.plane, sigma, sz, s);
  icDrawPreviewSoon();
}

// ---- the editor VIEW (REFINE_PLAN G.7) -------------------------------------
// "edit IC" replaces the whole display area with a dedicated editor canvas; the sim is
// hidden and paused, and nothing is ever painted over a live display. Leaving is one of
//   save    keep the drawing as the Reset target (and stay paused)
//   run     save, apply it, resume
//   cancel  restore the snapshot taken on entry
// Because the editor owns its canvas, display card 0 is an ordinary card again.
//
// The view is BUILT here, exactly like a display card, so each app's markup carries only
// the empty <div id="editview">: one implementation, both apps, and the 3D z-plane
// slider is a flag rather than a second copy of the header.
function icEditBuild(cfg) {
  const card = _mk("div", "card disp", el("editview"));
  const head = _mk("div", "cardhead", card);
  _mk("b", null, head).textContent = "IC editor";
  if (cfg.zplane) {
    _mk("label", null, head).textContent = "z plane";
    const s = _mk("input", "zslider", head);
    s.type = "range"; s.min = "0"; s.max = "0"; s.step = "1"; s.value = "0";
    s.oninput = () => {
      icDraw.plane = parseInt(s.value, 10) | 0;
      icEditCaption(); icDrawPreviewSoon();
    };
    icDraw.plSl = s;
  }
  for (const b of [["save &amp; run", "save, apply and resume", () => icEditLeave("run")],
                   ["save", "store this drawing as the Reset target", () => icEditLeave("save")],
                   ["clear", "erase the drawing (there is no undo stack)",
                    () => { icDrawClear(); icDrawPreview(); }],
                   ["cancel", "discard everything drawn since \"edit IC\"",
                    () => icEditLeave("cancel")]]) {
    const e = _mk("button", null, head);
    e.innerHTML = b[0]; e.title = b[1]; e.onclick = b[2];
  }
  const wrap = _mk("div", "cvwrap", card);
  icDraw.cv = _mk("canvas", "cvedit", wrap);
  icDraw.cv.width = 512; icDraw.cv.height = 512;
  icDraw.cap = _mk("div", "viewcap", card);
}
function icEditShow(on) {
  el("display").style.display = on ? "none" : "";
  el("editview").style.display = on ? "" : "none";
}
function icEditCaption() {
  const cfg = icDraw.cfg, q = cfg && cfg.params && cfg.params();
  if (!q || !icDraw.cap) return;
  const t = el("selPaint").options[el("selPaint").selectedIndex];
  icDraw.cap.innerHTML = "painting " + (t ? t.innerHTML : "") +
    "  &sigma;&perp; = " + icSigmaPerp().toPrecision(3) +
    ((q.nz || 1) > 1 ? "  &sigma;<sub>z</sub> = " + icSigmaZ().toPrecision(3) +
                       "  on iz = " + icDraw.plane : "");
}
// the editor canvas follows the display cards' aspect rule (REFINE_PLAN J.2), so a
// rectangular box is painted on a rectangular canvas and icDrawPos' u,v mapping stays
// right. Invalidating icPrev.cx is what makes icDrawCtx re-read the size.
function icEditResize() {
  const f = cards.cfg && cards.cfg.aspect, g = (f && f()) || null;
  const w = (g && g.w) || VEC_SIZE, h = (g && g.h) || VEC_SIZE;
  const cv = icDraw.cv;
  if (!cv || (icPrev.cx && icPrev.w === w && icPrev.h === h)) return;
  if (cv.parentNode) cv.parentNode.style.aspectRatio = (w === h) ? "" : (w + " / " + h);
  cv.width = w; cv.height = h;
  icPrev.cx = null;
}
function icEditEnter() {
  const cfg = icDraw.cfg, q = cfg && cfg.params && cfg.params();
  if (icDraw.on || !q) { if (!q) showStatus("no solver yet", "info"); return; }
  icEditResize();
  icDrawGrid(q);
  icDraw.snap = { zp: Float32Array.from(icDraw.zp), zm: Float32Array.from(icDraw.zm),
                  has: icDraw.has };
  icDraw.on = true; icDraw.down = false; icDraw.last = null;
  setRunning(false);
  el("btnRun").disabled = true; el("btnEdit").disabled = true;
  // the editor's OWN z plane: seeded from the first display card, then its slider's
  if (icDraw.plSl) {
    icDraw.plane = Math.max(0, Math.min((q.nz || 1) - 1, cfg.plane0 ? cfg.plane0() : 0));
    icDraw.plSl.max = String(Math.max(0, (q.nz || 1) - 1));
    icDraw.plSl.value = String(icDraw.plane);
  } else icDraw.plane = 0;
  icEditShow(true);
  icEditCaption();
  icDrawPreview();
}
// mode: "save" | "run" | "cancel"
// The display view comes back: every card has been detached for the length of the edit
// and nothing about the STATE need have changed (save and cancel both leave it alone), so
// the render gate would otherwise let them come back to whatever the compositor still had
// -- if anything (adversarial review, 2026-08-12: whether a WebGPU canvas keeps its last
// presented image across display:none is an engine's business, not something to rely on).
// One frame is a cheap way not to have that argument.
function icEditLeave(mode) {
  if (!icDraw.on) return;
  if (mode === "cancel" && icDraw.snap) {
    icDraw.zp.set(icDraw.snap.zp); icDraw.zm.set(icDraw.snap.zm);
    icDraw.has = icDraw.snap.has;
  }
  icDraw.snap = null;
  icDraw.on = false; icDraw.down = false; icDraw.last = null;
  icEditShow(false);
  for (const d of cards.disp) d.dirty = true;      // see the note above
  el("btnRun").disabled = false; el("btnEdit").disabled = false;
  if (mode === "run") {
    applyIC();
    setRunning(true);
  } else if (mode === "save") {
    showStatus("drawing saved — Reset applies it", "info");
  }
}

// bind the pointer handlers to the editor canvas icEditBuild just made. The editor view
// is built once, so this runs once.
function icDrawAttach() {
  const cv = icDraw.cv;
  if (!cv || !cv.addEventListener) return;
  cv.addEventListener("contextmenu", e => e.preventDefault());
  cv.addEventListener("pointerdown", e => {
    if (!icDraw.on) return;
    e.preventDefault();
    if (cv.setPointerCapture) { try { cv.setPointerCapture(e.pointerId); } catch (err) {} }
    icDraw.down = true;
    icDraw.neg = (e.button === 2) || ((e.buttons & 2) === 2);
    icDraw.last = null;
    icDrawStroke(e);
  });
  cv.addEventListener("pointermove", e => {
    if (!icDraw.on || !icDraw.down) return;
    e.preventDefault();
    icDrawStroke(e);
  });
  for (const t of ["pointerup", "pointercancel"]) {
    cv.addEventListener(t, () => { icDraw.down = false; icDraw.last = null; });
  }
}

// wire the editor. cfg supplies the app's knobs:
//   params()      the live solver's parameter object (nx, ny, nz, Lx, Ly, Lz) or null
//   target()      "zp" | "zm" | "phi" | "psi"     neg()   negative-sign toggle
//   zplane        3D: build the editor's OWN z-plane slider (absent in 2D)
//   plane0()      3D: the plane to open the editor on (the first display card's)
//   icRows        extra control-row ids that only apply to the letters / custom presets
//   sliders       the app's sigma slider ids (["rSigP"] in 2D, + "rSigZ" in 3D)
// The view, the buttons and the slider labels are identical in both apps and live here;
// only the knobs above are per-app.
function icDrawWire(cfg) {
  icDraw.cfg = cfg;
  if (icDraw.wired) return;
  icDraw.wired = true;
  icEditBuild(cfg);
  icDrawAttach();
  el("btnEdit").onclick = icEditEnter;
  // the paint target also picks the amplitude BASIS (item 15), so a change of it relabels
  // the two sliders and -- outside the editor, where there is nothing to save first --
  // genuinely changes the initial condition, exactly as moving an amp slider does
  el("selPaint").onchange = () => {
    syncLabels();
    if (icDraw.on) { icEditCaption(); icDrawPreview(); } else applyIC();
  };
  for (const id of (cfg.sliders || [])) {
    el(id).oninput = () => { syncLabels(); if (icDraw.on) icEditCaption(); };
  }
}

// which IC rows apply to the selected preset -- the same rule in both apps (the app
// names its own extra rows through cfg.icRows, so nothing here guesses at ids).
function icSyncRows() {
  // isP = a PACKET preset (letters, or the item-9 sinusoids): stored potentials that the
  // amp sliders normalize and the sigma_z row shapes. isC = the drawing, which is those
  // things AND owns the paint row.
  const p = el("selIC").value, isP = icIsPacketIC(p), isC = p === "custom";
  // The amplitude rows are HIDDEN, not greyed, for every preset that does not normalize a
  // stored potential (2026-08-13 feedback item 5). That is the equilibria -- whose
  // amplitudes are the PHYSICAL knobs one row up (U0 / b0 / psi0), with max|b| spelled out
  // in #vEqInfo beside them -- and equally `modes` and `quiescent`, which have no amplitude
  // knob at all. One predicate, so the panel never has two ways of saying "not here"; the
  // cost is that the rows are absent on a default boot (selIC = modes) and appear when the
  // user picks letters / sinusoids / the drawing, which is when they first mean anything.
  // Presets still WRITE rAmpP / rAmpM by id -- a hidden input takes a value normally.
  for (const id of ["rowAmpP", "rowAmpM"]) el(id).style.display = (isP || isC) ? "" : "none";
  el("rowDraw").style.display = isC ? "" : "none";
  for (const id of ((icDraw.cfg && icDraw.cfg.icRows) || [])) {
    el(id).style.display = (isP || isC) ? "" : "none";
  }
  // the registered equilibrium builders (REFINE_PLAN J.3): hide every row any of the
  // presets THIS PAGE offers owns, THEN show the selected one's -- two passes, because
  // rowEq belongs to both of them
  const B = (icHasPreset(p) && IC_BUILDERS[p]) || null;
  for (const k in IC_BUILDERS) {
    if (icHasPreset(k)) for (const id of IC_BUILDERS[k].rows || []) el(id).style.display = "none";
  }
  if (B) for (const id of B.rows || []) el(id).style.display = "";
  // hyper LOCK: hyper-dissipation falsifies the tearing / Rutherford physics these
  // presets exist to show, so they PIN the exponent instead of trusting the user to know.
  const want = B && B.hyper;
  const hs = el("selHyper");
  if (want && !hs.disabled) icSyncRows._hyperPrev = hs.value; // remember the user's hyper
  hs.disabled = !!want;
  hs.title = want ? "locked to " + want + " by the " + p + " preset (hyper-dissipation "
                    + "falsifies the resistive-layer physics)" : "";
  if (want && hs.value !== String(want)) { hs.value = String(want); applyControls(); }
  // review I2+J finding 2: leaving a locking preset restores the remembered hyper
  // (demo-preset switches set selHyper explicitly afterwards, so this never fights them)
  if (!want && icSyncRows._hyperPrev !== undefined) {
    if (hs.value !== icSyncRows._hyperPrev) { hs.value = icSyncRows._hyperPrev; applyControls(); }
    icSyncRows._hyperPrev = undefined;
  }
  if (!isC && icDraw.on) icEditLeave("save");
}

// ---------------------------------------------------------------------------
// z-plane trackers (3D display cards; REFINE_PLAN G.8 / note [7])
// ---------------------------------------------------------------------------
// `e` is readPlaneEnergy's 2*nz output, `off` the field's offset (0 for E+, nz for E-).
//
// CENTROID (the default). A plain argmax hops between planes as the discrete peak moves,
// which is what made the collision displays jitter. The CIRCULAR first moment
//   zbar = arg( sum_k E_k exp(2 pi i k / nz) ) * nz / 2pi
// is periodic by construction and moves smoothly: for a packet translating at v_A it is
// linear in t (node check 5). Returns a CONTINUOUS plane coordinate in [0, nz), or -1
// when there is no signal at all.
function trackCentroid(e, off, nz) {
  let cr = 0, ci = 0;
  for (let k = 0; k < nz; k++) {
    const w = Math.max(0, e[off + k]), th = 2 * Math.PI * k / nz;
    cr += w * Math.cos(th); ci += w * Math.sin(th);
  }
  if (!(cr * cr + ci * ci > 0)) return -1;
  let z = Math.atan2(ci, cr) * nz / (2 * Math.PI);
  z -= nz * Math.floor(z / nz);
  return z;
}
// ARGMAX with hysteresis: only leave the current plane when another one beats it by
// TRACK_HYST-1, so a peak wandering between two nearly equal planes stays put.
const TRACK_HYST = 1.1;
function trackArgmax(e, off, nz, cur) {
  let best = 0;
  for (let k = 1; k < nz; k++) if (e[off + k] > e[off + best]) best = k;
  const c = (cur >= 0 && cur < nz) ? cur : best;
  return e[off + best] > TRACK_HYST * e[off + c] ? best : c;
}

// ---------------------------------------------------------------------------
// main loop (identical in both apps: step, render, read back, draw)
// ---------------------------------------------------------------------------
// per-app extension points, both null unless an app sets them:
//   frameHook(solver)   extra per-frame work (the 3D plane tracking readback); it owns
//                       its own throttle and must re-check that `solver` is still the
//                       live one after any await
//   readoutExtra()      extra text lines appended to #readout
//   specExtra()         extra fields merged into the spectrum cards' data object (the 3D
//                       field-line E(k_par), which its own hook refreshes at its own rate)
let frameHook = null, readoutExtra = null, specExtra = null;
// Throttle clocks, and beside each the STATE it was last served for: the throttle says
// "not yet", the marker says "nothing new". A paused page fails the second test, so the
// spectrum's full extra field pass and the cut line's prep stop being taken 3-10 times a
// second over a picture that cannot move. The marker is (solver, nsteps, stateSeq), not
// the run flag, so exactly one more readback still lands AFTER the final step -- the
// charts show the state you paused on, not the one 300 ms before it.
const cardsThrottle = { spec: 0, cut: 0, specAt: null, cutAt: null };
const statsCache = { s: null, at: null };
// "fill these cards NOW": zero the clocks so the window is already over, and forget which
// state they were last served for so the gate cannot answer "nothing new" to a card that
// has never been fed. Both halves are needed -- a newly added or retyped chart on a
// PAUSED page is exactly the case where the state has not moved and the card still wants
// data.
function cardsThrottleReset() {
  cardsThrottle.spec = 0; cardsThrottle.cut = 0;
  cardsThrottle.specAt = null; cardsThrottle.cutAt = null;
}
let stateSeq = 0;
// `rebuild` ends in applyIC -> chartsReset -> stateBumped, so a new solver always moves
// stateSeq: the pair identifies the state without needing a handle on the solver itself.
const stateMark = () => (solver ? solver.nsteps + "/" + stateSeq : null);
// The state jumped without a step: an IC upload, a preset, a rebuild. Moving the counter
// is the whole job -- every consumer (the cards' needsRender, the throttled readbacks,
// the 3D hooks) is keyed on stateMark() and re-opens on its own.
function stateBumped() { stateSeq++; }
const _chartsOf = t => cards.chart.filter(c => c.type() === t);
// cards fed by one readback SOURCE: a type's `src` (island rides the cut line) or, by
// default, its own name. One readback serves every card that consumes it.
const _chartsBySrc = s => cards.chart.filter(c => (CHART_TYPES[c.type()].src || c.type()) === s);
// One frame's worth of display work, gated: a chain runs only when its picture can have
// changed (DisplayCard.needsRender). Returns how many cards actually drew, which is what
// makes "a paused page draws nothing" a measurable statement rather than an intention --
// the stub's requestAnimationFrame is a no-op, so devtools/checkidle.js drives THIS
// instead of re-implementing the gate and then testing its own copy.
//
// `paused` settles each active contour set's adapting range instead of relaxing it, so
// the last frame drawn is the one a still field deserves (see contSettle). recCapture()
// is outside the gate on purpose: it early-outs when no take is live, and when one IS
// live needsRender() is true anyway, so it never sees a frame this did not just draw.
// The catch keeps a capture fault at the OLD blast radius: under the timer feeder a
// VideoFrame/encode throw died inside one interval tick (logged, recording limped on);
// uncaught in the frame loop it would reject loop()'s promise and freeze the whole app --
// the one thing strictly worse than a bad take (adversarial review 2026-08-12, MINOR 2).
// (The editor view hides every card: do not render into detached canvases.)
function renderCards(paused) {
  if (icDraw.on || !solver) return 0;
  let n = 0;
  for (const d of cards.disp) {
    if (d.needsRender()) {
      if (paused) contSettle(solver.device, solver.chain(d.ci));
      d.render();
      d.dirty = false;
      d.seenMark = stateMark();
      n++;
    }
    try { d.recCapture(); } catch (e) { console.error(e); }
  }
  return n;
}
async function loop() {
  for (;;) {
    await new Promise(r => requestAnimationFrame(r));
    while (graveyard.length) graveyard.pop().destroy();
    if (!solver) continue;
    const t0 = performance.now();
    let n = 0;
    if (running) {
      // dt must be recomputed every step while the run spins up from a quiescent /
      // low-amplitude state: a frozen dt block there collapses and NaNs.
      const ce = solver.nsteps < 200 ? 1 : parseInt(el("rCflEvery").value, 10);
      n = stepsPerFrame;
      for (let i = 0; i < n; i++) solver.step(ce);
      // nothing to invalidate by hand: nsteps moved, so stateMark() did, and every card
      // and every gated readback below asks that question for itself
    }
    // One display chain run per card, gated -- and it must happen HERE, in this
    // synchronous task, before any await: a live WebCodecs recording takes its frame off
    // this very render, WebGPU has no preserveDrawingBuffer, so getCurrentTexture is
    // transient and the canvas holds only its last PRESENTED image; a capture deferred
    // even one microtask would wrap an expired texture (the same reason saveShot
    // re-renders first). It is also why leg 1's interval is now only a watchdog: feeding
    // the encoder off a timer meant a second full render per frame at a phase that beat
    // against this loop, which an iPhone showed as a stutter for the length of the take
    // (RECRAF_PLAN, 2026-08-12). Everything else about it is in renderCards.
    renderCards(!running);
    await device.queue.onSubmittedWorkDone();
    const ms = performance.now() - t0;
    if (running) {
      if (ms < 11 && stepsPerFrame < 64) stepsPerFrame++;
      else if (ms > 22 && stepsPerFrame > 1) stepsPerFrame--;
      const sps = n / (Math.max(ms, 0.05) / 1000);
      spsSmooth = spsSmooth ? 0.9 * spsSmooth + 0.1 * sps : sps;
    }
    // the scalars buffer cannot move without a step or a state jump, so a paused page
    // reuses the last read rather than paying a map round trip per frame for the same
    // twelve numbers. The readout text below is rebuilt either way -- the "paused" word
    // and the hooks' extra lines change without the numbers doing.
    const mark = stateMark();
    if (!statsCache.s || statsCache.at !== mark) {
      statsCache.s = await solver.readStats();
      statsCache.at = mark;
      if (!solver) continue;                  // retired while we were awaiting
    }
    const s = statsCache.s;
    if (frameHook) await frameHook(solver);
    if (!solver) continue;                    // retired while we were awaiting
    if (isFinite(s[1])) simT = s[1];          // what the capture filenames stamp
    const extra = readoutExtra ? readoutExtra() : "";
    // the sticky bar carries the one line that must always be visible; the rest
    // goes under the displays. Both come from this one stats readback.
    el("steps").textContent =
      "t " + s[1].toFixed(3) + "  step " + solver.nsteps +
      "  dt " + s[0].toExponential(2) + "  " + (running ? spsSmooth.toFixed(0) + " steps/s" : "paused");
    el("readout").textContent =
      "E_u = " + s[2].toExponential(5) + "  E_b = " + s[3].toExponential(5) +
      "\ns+   = " + s[4].toExponential(3) + "   s- = " + s[5].toExponential(3) +
      (extra ? "\n" + extra : "");
    // ?recdebug (RECRAF round 2, 2026-08-12): one readout line per live recording -- which
    // feeder is putting frames in the file and how stretched the loop is. This is how a
    // phone, which has no devtools console, reports whether the watchdog fired on a
    // visible page (wd must stay 0 there) and whether the loop gap explains a stutter.
    if (REC_DEBUG) for (const d of cards.disp) {
      const W = d.wc;
      // `lag` (RECASYNC, 2026-08-12) is the buffer path's own number: capture submit to
      // encode, i.e. how late the GPU's bytes arrive. Tens of ms are FINE -- the frame is
      // stamped by index, not by arrival -- and it is here only so a phone can say whether
      // the readback is keeping up at all. It stays 0 on the sync canvas path.
      if (W) el("readout").textContent += "\nrec: raf " + W.rafN + "  wd " + W.wdN +
        "  drop " + W.drop + "  gap " + Math.round(W.maxGap) + " ms" +
        "  vf " + W.tV.toFixed(1) + "  enc " + W.tE.toFixed(1) +
        "  lag " + Math.round(W.tL) + " ms";
    }

    // energy trace: one sample per readback, but never a duplicate t while paused.
    // s[8] is the cross helicity H_c, which is what the E+- mode needs.
    if (isFinite(s[1]) && isFinite(s[2]) && isFinite(s[3]) &&
        (!hist.t.length || s[1] > hist.t[hist.t.length - 1])) {
      histPush(s[1], s[2], s[3], isFinite(s[8]) ? s[8] : 0);
      for (const c of _chartsOf("energy")) c.draw(null);
    }
    // arrow overlay, per display card: the gather already ran inside that card's
    // render() pass, so this is only a copy + map round trip. ~10 Hz per card, one
    // frame of lag, and it never runs per step. A cube card gathers the plane its TOP
    // face shows and draws through that face's projection (REFINE_PLAN I2.3).
    // (snapshot: a close button can splice cards.disp while we are awaiting. A card that
    // is not showing arrows is simply skipped -- overlay() gates on showArrows(), so the
    // stale gather cannot reappear, and apply() has already redrawn the canvas.)
    // ... and, on the same snapshot and the same guard, the colorbar's tick labels
    // (FEEDBACK_2026-08-10 item 12): 4 bytes of the per-chain `maxVal` buffer, i.e. the
    // autoscale the colorize kernel of the frame just drawn actually divided by. Its own
    // (slower) throttle, and skipped for the fixed +-1 modes, which need no number at all.
    // Both read what a render LEFT in that card's buffers, so both are also gated on the
    // card's frame counter: once the gate above stops drawing, there is nothing new to
    // fetch and these go quiet with it -- but each still gets ONE read after the last
    // frame drawn, which is what keeps the paused arrows and the paused colorbar showing
    // the state on screen rather than the one before it.
    for (const d of cards.disp.slice()) {
      const tnow = performance.now();
      if (d.showArrows() && d.arrowSeq !== d.renderSeq && tnow - d.arrowAt > 100) {
        d.arrowAt = tnow; d.arrowSeq = d.renderSeq;
        const sv = solver;
        const av = await sv.readArrows(d.ci);
        if (sv === solver && cards.disp.indexOf(d) >= 0) d.setArrows(av, sv.nax, sv.nay);
      }
      if (d.barNeedsMax() && d.barSeq !== d.renderSeq &&
          performance.now() - d.barAt > CBAR_PERIOD) {
        d.barAt = performance.now(); d.barSeq = d.renderSeq;
        const sv = solver;
        const mv = await dispMaxRead(sv, d.ci);
        if (sv === solver && cards.disp.indexOf(d) >= 0) d.setBarRange(mv[0]);
      }
    }

    // cut trace: same throttle / guard idiom as the arrows. SELF-CONTAINED since
    // Phase H -- it runs its own line prep and depends on no display card, only on
    // its own z plane (2D: always 0), so one readback serves every card on that plane.
    const cutCards = _chartsBySrc("cut");
    if (cutCards.length && cardsThrottle.cutAt !== mark &&
        performance.now() - cardsThrottle.cut > 100) {
      cardsThrottle.cut = performance.now(); cardsThrottle.cutAt = mark;
      const sv = solver, planes = new Map();
      for (const c of cutCards) planes.set(cards.cfg.zsliceOf(c), null);
      for (const iz of Array.from(planes.keys())) {
        const vals = await sv.readCutLine(iz);
        if (sv !== solver) break;                 // retired while we were awaiting
        planes.set(iz, { vals, Ly: sv.p.Ly });
      }
      if (sv === solver) {
        // the island trace rides this readback (REFINE_PLAN J.4): psi on the resonant
        // line is one spectral integration of the b_x line already in hand.
        const d0 = planes.get(0);
        if (d0 && cutCards.some(c => c.type() === "island")) {
          islandPush(s[1], d0.vals, sv.p.ny, sv.p.Ly);
        }
        // ... and so does the k_y mode trace: one DFT coefficient of the u_x / b_x rows
        // of the very same line.
        if (d0 && cutCards.some(c => c.type() === "mode")) {
          modePush(s[1], d0.vals, sv.p.ny);
        }
        for (const c of cutCards) c.draw(planes.get(cards.cfg.zsliceOf(c)));
      }
    }

    // spectra: a full extra pass over the fields + a map round trip -> throttle hard
    const specCards = _chartsBySrc("spectrum");
    const now = performance.now();
    if (specCards.length && cardsThrottle.specAt !== mark && now - cardsThrottle.spec > 300) {
      cardsThrottle.spec = now; cardsThrottle.specAt = mark;
      const sv = solver;
      const sp = await sv.readSpectrum();
      if (sv === solver) {
        // `kunit` rides along so a pinned ghost can be re-registered on physical k when
        // the box changes (PINCURVE): the snapshot keeps the value it was taken under.
        const d = Object.assign({ perp: sp.perp, nb: sv.nb, fshell: sv.p.fshell,
                                  par: sp.par, parKfac: sp.parKfac, kunit: sv.g.kunit },
                                specExtra ? specExtra() : null);
        autoDissCache.sv = sv; autoDissCache.at = now; autoDissCache.perp = sp.perp;
        for (const c of specCards) c.draw(d);
      }
    }
    // auto-diss (item 6): the same perpendicular bins, at its own 2 Hz cadence. It rides
    // the cards' cached readback whenever a fresh one exists and takes its own only
    // when none does -- the controller must work with the chart closed.
    await autoDissHook(solver);
  }
}

// ---------------------------------------------------------------------------
// self-test harness
// ---------------------------------------------------------------------------
function testRow(name, err, tol, note) {
  const ok = err <= tol;
  return "<tr><td>" + name + "</td><td>" + (isFinite(err) ? err.toExponential(2) : String(err)) +
    "</td><td>" + tol.toExponential(0) + "</td><td class='" + (ok ? "pass" : "fail") + "'>" +
    (ok ? "PASS" : "FAIL") + "</td><td class='note'>" + (note || "") + "</td></tr>";
}
function showTests(rows, extraHtml) {
  el("tests").innerHTML = "<table><tr><th>test</th><th>rel L2</th><th>tol</th><th></th><th></th></tr>" +
    rows.join("") + "</table>" + (extraHtml || "");
}

// statistical check of the OU path: run the solver from a quiescent start at a known
// total injection rate and compare dE/dt + <D> against it. Identical in both apps.
async function ouInjectionRow(s) {
  // production cap, NOT the refvector-recorded one: the harness solver is built with
  // smax = R.forcing_scale_max so the deterministic rows replay the JSON exactly, but
  // this row is statistical and must run the cap the apps actually ship -- at the
  // saturated eps=1 state the required scale is ~1.8, so a recorded smax=1 pins and
  // caps injection at ~0.64 (the 2026-08-09 failure).
  s.p.epsP = 0.5; s.p.epsM = 0.5; s.p.smax = 1e4; s.uploadCfg();
  s.setIC(true);                                   // quiescent start
  for (let i = 0; i < 600; i++) { s.step(1); if (i % 100 === 99) await device.queue.onSubmittedWorkDone(); }
  const a = await s.readStats();
  let clipped = 0;
  for (let i = 0; i < 2000; i++) {
    s.step(1);
    if (i % 200 === 199) {
      await device.queue.onSubmittedWorkDone();
      const q = await s.readStats();
      const smx = 0.999 * (s.p.smax || 1);
      if (Math.abs(q[4]) >= smx || Math.abs(q[5]) >= smx) clipped++;
    }
  }
  const b = await s.readStats();
  const dtw = b[1] - a[1];
  const dEdt = ((b[2] + b[3]) - (a[2] + a[3])) / dtw;
  const Dbar = (b[7] - a[7]) / dtw;
  const inj = dEdt + Dbar;
  const eps = 1.0;
  return testRow("OU injection rate (statistical)", Math.abs(inj - eps) / eps, 0.2,
    "dE/dt=" + dEdt.toPrecision(4) + " D=" + Dbar.toPrecision(4) + " -> " + inj.toPrecision(4) +
    " vs eps=1" + (clipped ? " [scale clipped in window]" : ""));
}

// wire the "run self-test" button: pause, run, restore
function wireTestButton(fn) {
  el("btnTest").onclick = async () => {
    const wasRunning = running; running = false;
    el("btnTest").disabled = true;
    el("tests").innerHTML = "<div class='hint'>running&hellip;</div>";
    try { await fn(); }
    catch (e) { showStatus("self-test failed: " + e.message, "err"); console.error(e); }
    el("btnTest").disabled = false;
    running = wasRunning;
  };
}
