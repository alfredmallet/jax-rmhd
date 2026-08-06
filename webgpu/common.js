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
function showStatus(msg, kind) {
  statusEl.className = kind || "info";
  statusEl.textContent = msg;
}
// any uncaught error anywhere -> visible on the page, never a silently dead UI
window.addEventListener("error", e => showStatus("Error: " + e.message, "err"));
window.addEventListener("unhandledrejection", e =>
  showStatus("Error: " + (e.reason && e.reason.message || e.reason), "err"));
function clearStatus() { statusEl.className = ""; statusEl.textContent = ""; }

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

async function readBuf(device, buf, byteLen) {
  const st = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const e = device.createCommandEncoder();
  e.copyBufferToBuffer(buf, 0, st, 0, byteLen);
  device.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(st.getMappedRange().slice(0));
  st.unmap(); st.destroy();
  return out;
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

// One display chain's contour level table (REFINE_PLAN I2.4). The CPU owns nlev, the
// GPU (contLevel) owns the adapting range and the spacing, and a ZERO spacing is what
// switches the overlay off inside the shared shader -- so turning contours off is one
// 4-byte write, not a pipeline or bind-group change. Identical in both apps.
function setContLevels(device, D, nlev) {
  device.queue.writeBuffer(D.buf.contB, 8, new Float32Array([Math.max(1, nlev | 0)]));
  if (!D.cont) device.queue.writeBuffer(D.buf.contB, 4, new Float32Array([0]));
}
// ... and the tail of its per-frame prep, once the app's own inverse transform has put
// the potential plane where colorize reads it: max |pot| over that plane (the shared
// reduction) -> the level table. Identical in both apps.
function contLevelEncode(p, s, D, nPart) {
  p.setPipeline(s.pl.maxPartial); p.setBindGroup(0, D.bg.maxPartialCont);
  p.dispatchWorkgroups(nPart);
  p.setPipeline(s.pl.maxFinal); p.setBindGroup(0, D.bg.maxFinalCont);
  p.dispatchWorkgroups(1);
  p.setPipeline(s.pl.contLevel); p.setBindGroup(0, D.bg.contLevel);
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
let solver = null, running = false, stepsPerFrame = 1, spsSmooth = 0;

// a display card's canvas -> a configured WebGPU context (cards are created and
// destroyed at runtime, so this is not part of initGPU)
function gpuCanvasCtx(cv) {
  if (!device || !cv || !cv.getContext) return null;
  const c = cv.getContext("webgpu");
  if (c) c.configure({ device, format: canvasFormat, alphaMode: "opaque" });
  return c;
}

// adapter + device. opts.maxLimits asks the adapter for its own
// reported limits (always a legal request): the 3D app's 8-field gradient stack is
// 129 MiB at 256^2x64, past the DEFAULT 128 MiB storage-binding limit.
async function initGPU(opts) {
  if (!navigator.gpu) {
    showStatus("WebGPU is not available in this browser. Use Chrome 113+ (or Edge) on a machine "
      + "with a supported GPU; on Linux you may need --enable-unsafe-webgpu.", "err");
    return false;
  }
  let adapter = null;
  try { adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }); }
  catch (e) { showStatus("requestAdapter failed: " + e.message, "err"); return false; }
  if (!adapter) { showStatus("No WebGPU adapter found (no compatible GPU, or WebGPU is disabled).", "err"); return false; }
  const need = {};
  if (opts && opts.maxLimits) {
    for (const k of ["maxStorageBufferBindingSize", "maxBufferSize"]) {
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
const COL = { ek: "#6fb3ff", em: "#ff9c6b", et: "#9ee493",
              zp: "#c58cf5", zm: "#5fd7c8",          // the Elsasser pair, in every chart
              grid: "#232833", axis: "#39404d", txt: "#8a94a3",
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
// logical size rides on the context so drawArrows / clearArrows need no extra argument.
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
const ARROW_FRAME = { ox: 0, oy: 0, ax: VEC_SIZE, ay: 0, bx: 0, by: VEC_SIZE };
function drawArrows(c, a, nax, nay, X) {
  const W = c.__w || VEC_SIZE, H = c.__h || VEC_SIZE;
  c.clearRect(0, 0, W, H);
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
// items: [label, colour, dash?]. Wraps onto further lines at xmax, so a chart with
// four series plus a guide line still shows every key.
function legend(c, x, y, items, xmax) {
  let lx = x, ly = y;
  for (const it of items) {
    const w = 15 + c.measureText(it[0]).width + 11;
    if (xmax && lx > x && lx + w > xmax) { lx = x; ly += 12; }
    c.strokeStyle = it[1]; c.lineWidth = 2; c.setLineDash(it[2] || []);
    c.beginPath(); c.moveTo(lx, ly - 3); c.lineTo(lx + 12, ly - 3); c.stroke();
    c.setLineDash([]);
    c.fillStyle = it[1]; c.fillText(it[0], lx + 15, ly);
    lx += w;
  }
}

// The two energy-trace modes (REFINE_PLAN H.2), from the SAME history:
//   kmt  E_kin, E_mag, E_tot                          (the default: unchanged)
//   pmt  E+, E-, E_tot with E+- = E_kin + E_mag +- H_c
// so E_tot = (E+ + E-)/2, which is the repo's Elsasser convention (jax_rmhd
// physics/rmhd.py) and what puts all three curves on one comparable axis.
const ENERGY_MODES = {
  kmt: [["E_kin", COL.ek, i => hist.ek[i]], ["E_mag", COL.em, i => hist.em[i]],
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

  c.strokeStyle = COL.grid; c.fillStyle = COL.txt; c.textAlign = "right"; c.lineWidth = 1;
  if (useLog) {
    const d0 = Math.ceil(vlo), d1 = Math.floor(vhi);
    const stride = Math.max(1, Math.ceil((d1 - d0 + 1) / 6));
    for (let d = d0; d <= d1; d += stride) {
      const y = Math.round(Y(Math.pow(10, d))) + 0.5;
      c.beginPath(); c.moveTo(x0, y); c.lineTo(x1, y); c.stroke();
      c.fillText("1e" + d, x0 - 5, y + 3);
    }
  } else {
    for (let i = 0; i <= 4; i++) {
      const v = vlo + (vhi - vlo) * i / 4, y = Math.round(Y(v)) + 0.5;
      c.beginPath(); c.moveTo(x0, y); c.lineTo(x1, y); c.stroke();
      c.fillText(v.toExponential(1), x0 - 5, y + 3);
    }
  }
  c.fillText(t1.toFixed(2), x1, H - 6);
  c.textAlign = "left";
  c.fillText("t = " + t0.toFixed(2), x0, H - 6);

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
const ISLAND_SERIES = [["W", COL.zp, i => islandHist.w[i]]];
function drawIsland(c) {
  if (!c) return;
  if (!icEq.on) {
    chartFrame(c, CW, EH, PADC);
    c.textAlign = "left"; c.fillStyle = COL.txt;
    c.fillText("island width — needs the tearing IC preset", PADC.l + 6, PADC.t + 13);
    return;
  }
  drawTimeSeries(c, CW, EH, PADC, islandHist.t, ISLAND_SERIES,
                 { log: true, empty: "island width W(t) — collecting…" });
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
const specSeries = sq => (sq === "both" ? SPEC_SETS.ub.concat(SPEC_SETS.pm)
                                        : (SPEC_SETS[sq] || SPEC_SETS.ub));
function drawSpectrum(c, d, o) {
  if (!c) return;
  const P = PADS, x0 = P.l, x1 = SW - P.r, y0 = P.t, y1 = SH - P.b;
  chartFrame(c, SW, SH, P);
  c.textAlign = "left"; c.fillStyle = COL.txt;
  const bins = (d && d.perp) || new Float32Array(3), nb = (d && d.nb) || 1;
  const fshell = (d && d.fshell) || [1, 3];
  const par = d && d.par, parKfac = (d && d.parKfac) || 1;
  const set = specSeries(o && o.sq);
  const sd = (o && o.sd) || "both";
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
      curves.push([pts, sr[1], [5, 3], sr[0] + "(k∥)"]);
    }
  }
  if (hiP > 0) { hi = hiP; lo = loP; }
  if (nb < 2 || !(hi > 0)) { c.fillText("spectra — waiting…", x0 + 6, y0 + 13); return; }
  const ymax = Math.log10(hi) + 0.3;
  const ymin = Math.max(Math.log10(Math.max(lo, hi * 1e-14)) - 0.3, ymax - 14.6);
  const xmax = Math.log10(nb);
  const X = k => x0 + Math.log10(k) / xmax * (x1 - x0);
  const Y = v => px(y1 - (Math.log10(v) - ymin) / (ymax - ymin) * (y1 - y0));

  // y decades
  c.strokeStyle = COL.grid; c.fillStyle = COL.txt; c.textAlign = "right"; c.lineWidth = 1;
  const d0 = Math.ceil(ymin), d1 = Math.floor(ymax);
  const stride = Math.max(1, Math.ceil((d1 - d0 + 1) / 7));
  for (let d = d0; d <= d1; d += stride) {
    const y = Math.round(Y(Math.pow(10, d))) + 0.5;
    c.beginPath(); c.moveTo(x0, y); c.lineTo(x1, y); c.stroke();
    c.fillText("1e" + d, x0 - 5, y + 3);
  }
  // x decades + endpoint
  c.textAlign = "center";
  for (const k of [1, 10, 100, 1000]) {
    if (k > nb) break;
    const x = Math.round(X(k)) + 0.5;
    c.strokeStyle = COL.grid; c.beginPath(); c.moveTo(x, y0); c.lineTo(x, y1); c.stroke();
    c.fillStyle = COL.txt; c.fillText(String(k), x, SH - 8);
  }
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
  // k^-5/3 guide, anchored on the first drawn (perpendicular) series just above the
  // forcing shell
  const kA = Math.max(2, Math.min(nb - 1, Math.round(fshell[1])));
  let anch = 0;
  const first = wantPerp && curves.length ? curves[0][0] : [];
  for (let i = 0; i < first.length; i += 2) {
    if (first[i] >= kA) { anch = first[i + 1] * Math.pow(first[i], 5 / 3); break; }
  }
  if (anch > 0) {
    c.strokeStyle = COL.guide; c.setLineDash([5, 4]);
    c.beginPath();
    c.moveTo(X(kA), Y(anch * Math.pow(kA, -5 / 3)));
    c.lineTo(X(nb), Y(anch * Math.pow(nb, -5 / 3)));
    c.stroke(); c.setLineDash([]);
  }
  c.lineWidth = 1.4;
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
  c.restore();
  const items = curves.filter(cv => cv[0].length >= 4).map(cv => [cv[3], cv[1], cv[2]]);
  if (anch > 0) items.push(["k^-5/3", COL.guide, [4, 3]]);
  legend(c, x0 + 6, y0 + 12, items, x1 - 30);
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

  // y ticks: the two extremes, plus the zero line for signed quantities
  c.strokeStyle = COL.grid; c.fillStyle = COL.txt; c.textAlign = "right"; c.lineWidth = 1;
  for (const v of (signed ? [vhi, 0, vlo] : [vhi, 0])) {
    const y = Math.round(Y(v)) + 0.5;
    c.beginPath(); c.moveTo(x0, y); c.lineTo(x1, y); c.stroke();
    c.fillText(v === 0 ? "0" : v.toExponential(1), x0 - 5, y + 3);
  }
  // x ticks: 0, Ly/2, Ly
  c.textAlign = "center";
  for (const f of [0, 0.5, 1]) {
    const x = Math.round(x0 + f * (x1 - x0)) + 0.5;
    c.beginPath(); c.strokeStyle = COL.grid; c.moveTo(x, y0); c.lineTo(x, y1); c.stroke();
    c.fillStyle = COL.txt; c.fillText((f * Ly).toFixed(2), x, CH - 6);
  }
  c.textAlign = "left";
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
//   cube        true in 3D: that select also offers the cube-faces VIEW (I2.1)
//   nz()        current nz, for the slider range
//   zsliceOf(c) resolved plane index of card c (slider or tracked peak)
//   arrowXform() 3D only: the cube top face's (u,v) -> canvas affine, for the arrows
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
                   o: [["kmt", "E_kin / E_mag"], ["pmt", "E&#8314; / E&#8315;"]] }],
    draw: (c, d, o) => drawEnergy(c, o),
    hint: "vs t (2000 points, decimated 2:1 when full); E<sup>&plusmn;</sup> = E<sub>kin</sub> + "
      + "E<sub>mag</sub> &plusmn; H<sub>c</sub>, so E<sub>tot</sub> = (E<sup>+</sup>+E<sup>&minus;</sup>)/2"
  },
  spectrum: {
    label: "spectra", w: SW, h: SH,
    opts: cfg => [{ id: "sq", ti: "which spectra to bin",
                    o: [["ub", "E_u / E_b"], ["pm", "E&#8314; / E&#8315;"], ["both", "both"]] }]
      .concat(cfg.zslice
        ? [{ id: "sd", ti: "perpendicular (solid) / parallel (dashed) spectra",
             o: [["both", "&perp; + &#8741;"], ["perp", "&perp; only"], ["par", "&#8741; only"]] }]
        : []),
    draw: (c, d, o) => drawSpectrum(c, d, o),
    hint: "shell-binned, ~3&times;/s; E<sup>&plusmn;</sup>(k) = E<sub>u</sub>+E<sub>b</sub>&plusmn;H<sub>c</sub>. "
      + "The y range follows the &perp; spectra."
  },
  cut: {
    label: "cut trace", w: CW, h: CH, zslice: true,
    opts: () => [{ id: "pair", ti: "which pair of components to trace",
                   o: [["u", "u_x, u_y"], ["b", "b_x, b_y"], ["z", "|z&#8314;|, |z&#8315;|"]] }],
    draw: (c, d, o) => drawCut(c, d, o),
    hint: "its own line along y at x = L<sub>x</sub>/2, ~10&times;/s (independent of the displays)"
  },
  // REFINE_PLAN J.4. `src: "cut"` says it feeds off the cut readback -- the X and O points
  // live on the resonant surface x = Lx/2, which is the line cutPrep already prepares, so
  // this card adds no kernel, no buffer and no round trip. 2D only: the equilibria are.
  island: {
    label: "island width", w: CW, h: EH, src: "cut", avail: cfg => !cfg.zslice,
    draw: c => drawIsland(c),
    hint: "W = 4&radic;(&Delta;&psi;/2|&psi;&Prime;|) from the &psi; extrema on x = L<sub>x</sub>/2, "
      + "with &psi;&Prime; measured on the equilibrium; log y, so the linear stage is a straight line"
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
const ZSRC_OPTS = [{ v: "manual", t: "z slice" }, { v: "zp", t: "track z&#8314;" },
                   { v: "zm", t: "track z&#8315;" }];
const ZSRC_CUBE = [{ v: "cube", t: "cube faces" }, { v: "cubezp", t: "cube + track z&#8314;" },
                   { v: "cubezm", t: "cube + track z&#8315;" }];
function _zSliceControls(card, head, cube) {
  const cfg = cards.cfg;
  if (!cfg.zslice) return;
  card.selZSrc = _sel(head, cube ? ZSRC_OPTS.concat(ZSRC_CUBE) : ZSRC_OPTS,
                      "which z plane this card uses" + (cube ? ", and whether it draws the cube faces" : ""));
  card.rSlice = _mk("input", "zslider", head);
  card.rSlice.type = "range"; card.rSlice.min = "0"; card.rSlice.step = "1"; card.rSlice.value = "0";
  card.rSlice.max = String(Math.max(0, cfg.nz() - 1));
}
// the plane source of a card, with the view prefix stripped: every caller that resolves
// a plane (the app's zsliceOf / trackingOn) sees exactly the three pre-I2 values
function _zSrcPlane(v) { return v.indexOf("cube") === 0 ? (v.slice(4) || "manual") : v; }
// the contour overlay's per-card selectors (REFINE_PLAN I2.4), in BOTH apps: in-plane
// field lines of psi (B_perp) or streamlines of phi, on the plane the card displays.
// The value IS the potential's display mode, so the solver needs no second mapping.
// (a function, not a const: physics.js -- where DISP_PSI lives -- loads after this file)
const _contOpts = () => [{ v: "0", t: "no contours" }, { v: String(DISP_PSI), t: "&psi; contours" },
                         { v: String(DISP_PHI), t: "&phi; contours" }];
const CONT_LEVELS = [8, 16, 32];

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
    al.appendChild(document.createTextNode("arrows"));
    al.title = "vector overlay on the |u| / |b| / |z±| modes (on the cube: its top face)";
    this.selCont = _sel(head, _contOpts(), "in-plane field lines: psi -> B_perp, phi -> streamlines");
    this.selLev = _sel(head, CONT_LEVELS.map(n => ({ v: n, t: n + " levels" })), "contour level count");
    this.selLev.style.display = "none";               // only meaningful with contours on
    this.selCmap = _sel(head, CMAP_NAMES.map((n, i) => ({ v: i, t: n })), "colormap");
    this.btnClose = _mk("button", "x", head);
    this.btnClose.innerHTML = "&times;";
    this.btnClose.title = "close this display";

    const wrap = _mk("div", "cvwrap", root);
    this.wrap = wrap;
    this.cv = _mk("canvas", "cvmain", wrap);
    this.cvVec = _mk("canvas", "cvvec", wrap);
    this.cap = _mk("div", "viewcap", root);
    this.gw = 0; this.gh = 0;
    this._resize();                       // sizes both canvases BEFORE the GPU context
    this.ctx = gpuCanvasCtx(this.cv);
    this.vecDrawn = false;
    this.arrowAt = 0;

    const apply = () => { this.apply(); if (cards.cfg.onLayout) cards.cfg.onLayout(); };
    this.selField.onchange = apply;
    this.selCmap.onchange = apply;
    this.selCont.onchange = apply;
    this.selLev.onchange = apply;
    this.cbArrow.onchange = () => { if (!this.cbArrow.checked) this.clearArrows(); apply(); };
    if (this.selZSrc) this.selZSrc.onchange = () => { this.clearArrows(); apply(); };
    if (this.rSlice) this.rSlice.oninput = apply;
    this.btnClose.onclick = () => cardClose(this);
  }
  sel() { return parseInt(this.selField.value, 10) | 0; }
  cmap() { return parseInt(this.selCmap.value, 10) | 0; }
  // the PLANE source, view prefix stripped (the app's zsliceOf / trackingOn use this)
  zsrc() { return _zSrcPlane(this.selZSrc ? this.selZSrc.value : "manual"); }
  // ... and the view the same select carries: cube faces instead of the one plane
  cubeView() { return !!this.selZSrc && this.selZSrc.value.indexOf("cube") === 0; }
  cont() { return parseInt(this.selCont.value, 10) | 0; }
  nlev() { return parseInt(this.selLev.value, 10) | 0; }
  // Aspect-correct card geometry (REFINE_PLAN J.2): ONE place decides the card's pixel
  // box, and the wrapper's CSS ratio, the WebGPU canvas backing store, the overlay canvas
  // and the arrow frame all follow it. Apps with square boxes supply no cfg.aspect and
  // get the historical 512x512 with no inline style at all.
  _resize() {
    const f = cards.cfg && cards.cfg.aspect, g = (f && f()) || null;
    const w = (g && g.w) || VEC_SIZE, h = (g && g.h) || VEC_SIZE;
    if (w === this.gw && h === this.gh) return;
    this.gw = w; this.gh = h;
    this.wrap.style.aspectRatio = (w === h) ? "" : (w + " / " + h);   // "" = the CSS 1/1
    this.cv.width = w; this.cv.height = h;
    this.vcx = vecCtx(this.cvVec, w, h);
    this.vecDrawn = false;
  }
  // push this card's state into the live solver and relabel it
  apply() {
    if (!solver) return;
    const cfg = cards.cfg;
    this._resize();
    if (this.rSlice) this.rSlice.max = String(Math.max(0, cfg.nz() - 1));
    solver.setDisplayMode(this.ci, this.sel(), cfg.zsliceOf(this), this.cmap(),
                          { cube: this.cubeView(), cont: this.cont(), nlev: this.nlev() });
    // the slider drives the displayed plane in the slice view and the TOP face in the
    // cube view, so it is live in both -- and dead whenever a tracker owns the plane
    if (this.rSlice) this.rSlice.disabled = this.zsrc() !== "manual";
    this.selLev.style.display = this.cont() ? "" : "none";
    const o = this.selField.options[this.selField.selectedIndex];
    this.cap.innerHTML = (o ? o.innerHTML : "") + (cfg.caption ? cfg.caption(this) : "");
    if (!this.showArrows()) this.clearArrows();
  }
  showArrows() {
    return !!(this.cbArrow.checked && solver && dispIsVector(solver.modeOf(this.ci)));
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
  render() { if (this.ctx && solver) solver.render(this.ctx, this.ci); }
  drawArrows(a, nax, nay) {
    if (this.vcx) { drawArrows(this.vcx, a, nax, nay, this.arrowFrame()); this.vecDrawn = true; }
  }
  clearArrows() {
    if (this.vcx && this.vecDrawn) { this.vcx.clearRect(0, 0, this.gw, this.gh); this.vecDrawn = false; }
  }
  destroy() { if (this.root.parentNode) this.root.parentNode.removeChild(this.root); }
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
      cardsThrottle.spec = 0; cardsThrottle.cut = 0;
    };
    this.btnClose.onclick = () => cardClose(this);
  }
  type() { return this.selType.value; }
  zsrc() { return _zSrcPlane(this.selZSrc ? this.selZSrc.value : "manual"); }
  // the option selects, as { id: value } -- what the type's draw() branches on
  optVals() {
    const o = {};
    for (const s of this.optEls) o[s.__optId] = s.value;
    return o;
  }
  build() {
    const T = CHART_TYPES[this.type()];
    // drop the previous type's controls (a select is appended AFTER the close button,
    // so the button is re-appended last to keep its margin-left:auto place)
    for (const s of this.optEls) this.head.removeChild(s);
    if (this.rSlice) { this.head.removeChild(this.selZSrc); this.head.removeChild(this.rSlice); }
    this.optEls = []; this.selZSrc = null; this.rSlice = null;
    const redraw = () => { this.draw(null); cardsThrottle.spec = 0; cardsThrottle.cut = 0; };
    for (const spec of (T.opts ? T.opts(cards.cfg || {}) : [])) {
      const s = _sel(this.head, spec.o.map(x => ({ v: x[0], t: x[1] })), spec.ti);
      if (spec.v !== undefined) s.value = String(spec.v);
      s.__optId = spec.id;
      s.onchange = redraw;
      this.optEls.push(s);
    }
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
    this.hint.innerHTML = T.hint;
  }
  // keep the z slider in range / enabled only when this card picks its plane by hand
  apply() {
    if (!this.rSlice) return;
    this.rSlice.max = String(Math.max(0, cards.cfg.nz() - 1));
    this.rSlice.disabled = this.zsrc() !== "manual";
  }
  draw(data) { CHART_TYPES[this.type()].draw(this.cx, data, this.optVals()); }
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
    if (state.cont !== undefined) c.selCont.value = String(state.cont);
    if (state.nlev !== undefined) c.selLev.value = String(state.nlev);
  }
  return c;
}
function addChartCard(type) {
  const c = new ChartCard(chartTypeKeys().indexOf(type) >= 0 ? type : "energy");
  cards.chart.push(c);
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
  cardsThrottle.spec = 0; cardsThrottle.cut = 0;
  if (cards.cfg && cards.cfg.onLayout) cards.cfg.onLayout();
}
// replace the whole layout (used by the presets and by boot)
function cardsLayout(L) {
  for (const d of cards.disp.slice()) { d.destroy(); }
  for (const c of cards.chart.slice()) { c.destroy(); }
  cards.disp.length = 0; cards.chart.length = 0;
  for (const s of (L && L.disp) || [{}]) addDisplayCard(s);
  while (cards.disp.length < CARD_MIN_DISP) addDisplayCard();
  for (const t of (L && L.charts) || ["energy", "spectrum", "cut"]) addChartCard(t);
  cardsSync();
}
// the card the single-instance overlays (IC editor, cut trace) hang off
function primaryCard() { return cards.disp.length ? cards.disp[0] : null; }
// clear the traces after an IC change / rebuild (one call, both apps)
function chartsReset() {
  histReset(); islandReset();
  cardsThrottle.spec = 0; cardsThrottle.cut = 0;
  for (const c of cards.chart) c.draw(null);
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
// `t` is HTML (the entities the markup used); `ti` is plain text.
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
function controlsBuild(spec) {
  const bar = el("topbar");
  for (const it of (spec.topbar || CTRL_TOPBAR)) _ctrlItem(bar, it);
  const host = el("controls");
  for (const g of spec.groups) {
    const d = _mk("details", null, host);
    d.id = g.id;
    if (g.keep) d.setAttribute("data-keep-open", "");
    _mk("summary", null, d).innerHTML = g.summary;
    for (const r of g.rows) {
      if (r.k === "hintdiv") { const h = _mk("div", "hint", d); h.id = r.id; continue; }
      const items = Array.isArray(r) ? r : r.items;
      const row = _mk("div", "row", d);
      if (r.id) row.id = r.id;
      if (r.hide) row.style.display = "none";
      for (const it of items) _ctrlItem(row, it);
    }
  }
}

// ---- the rows both pages share --------------------------------------------
const CTRL_TOPBAR = [
  { k: "btn", id: "btnRun", t: "Run" },
  { k: "btn", id: "btnReset", t: "Reset" },
  { k: "lab", t: "preset", for: "selPreset" },
  { k: "sel", id: "selPreset", o: [] },
  { k: "btn", id: "btnParams", t: "hide params" },
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
// offers the eta/nu ratio (REFINE_PLAN J.1 -- the 2D propagator is diagonal per field,
// so nu and eta can differ; the 3D 2x2 Alfven propagator needs an equal diagonal).
const ctrlDissRow = (dflt, o) => [
  { k: "lab", t: "hyper" },
  { k: "sel", id: "selHyper", o: [[1, "1"], [2, "2"], [3, "3"], [4, "4"]], v: 4 },
  { k: "lab", t: "diss" },
  { k: "rng", id: "rDiss", min: -20, max: -1, step: 0.05, v: dflt }, { k: "val", id: "vDiss" },
  { k: "btn", id: "btnAutoDiss", t: "auto",
    ti: "a marginally-resolved diss for the current hyper / resolution / power" }
].concat((o && o.ratio) ? [
  { k: "lab", t: "&eta;/&nu;" },
  { k: "num", id: "nPm", v: 1, w: 62,
    ti: "inverse magnetic Prandtl number: nu multiplies phi, eta = ratio*nu multiplies psi. "
      + "1 is the historical scalar dissipation; changing it rebuilds the solver." }
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
// the IC group: the two amplitude rows and the paint row are shared; `letters` is
// the wording of the letters preset and `extra` the 3D-only sigma_z row + chi line
const ctrlGrpIC = o => ({
  id: "grpIC", summary: "initial condition", rows: [
    [{ k: "lab", t: "preset" },
     { k: "sel", id: "selIC", o: [["modes", "large-scale modes"], ["letters", o.letters],
                                  ["custom", "custom (drawn blobs)"], ["quiescent", "quiescent (zero)"]]
                                 .concat(o.presets || []) }],
    [{ k: "lab", t: "&zeta;&#8314; amp" },
     { k: "rng", id: "rAmpP", min: -2, max: 1, step: 0.05, v: o.amp }, { k: "val", id: "vAmpP" },
     { k: "cbl", id: "cbAmpLock", t: "lock", v: true, ti: "move the two potential amplitudes together" }],
    [{ k: "lab", t: "&zeta;&#8315; amp" },
     { k: "rng", id: "rAmpM", min: -2, max: 1, step: 0.05, v: o.amp }, { k: "val", id: "vAmpM" }]
  ].concat(o.extra || [], [
    { id: "rowDraw", hide: true, items: [
      { k: "btn", id: "btnEdit", t: "edit IC", ti: "pause the run and open the IC editor" },
      { k: "lab", t: "paint" },
      { k: "sel", id: "selPaint", o: [["zp", "&zeta;&#8314;"], ["zm", "&zeta;&#8315;"],
                                      ["phi", "&phi;"], ["psi", "&psi;"]] },
      { k: "lab", t: "&sigma;&perp;" },
      { k: "rng", id: "rSigP" }, { k: "val", id: "vSigP" },
      { k: "lab", t: "negative" },
      { k: "cb", id: "cbNeg", ti: "deposit with a minus sign (or drag with the right button)" }
    ] }
  ])
});
// displays & charts: the two add buttons, plus whatever page-wide extras follow
const ctrlGrpDisp = extra => ({
  id: "grpDisp", summary: "displays &amp; charts", keep: true, rows: [
    [{ k: "btn", id: "btnAddDisp", t: "+ display" },
     { k: "btn", id: "btnAddChart", t: "+ chart" }].concat(extra || [])
  ]
});

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

// marginally-resolved dissipation for the current hyper / resolution / injected power:
// diss ~ eps^(1/3) * k_c^(2/3 - 2*hyper) with k_c = nx/3, times the safety margin the
// repo's reference notebook uses. (uiParams / applyControls are per-app.)
function autoDiss() {
  const q = uiParams();
  // use the sliders' eps even when the forcing checkbox is off (auto-diss for the
  // power you WOULD inject; with eps identically 0 the formula would degenerate)
  const epsTot = Math.pow(10, parseFloat(el("rEpsP").value)) +
                 Math.pow(10, parseFloat(el("rEpsM").value));
  const kc = q.nx / 3;
  const diss = 30 * Math.pow(epsTot, 1 / 3) * Math.pow(kc, 2 / 3 - 2 * q.hyper);
  const lg = Math.min(-1, Math.max(-20, Math.log10(diss)));
  el("rDiss").value = String(lg);
  applyControls();
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
// write one preset's control values + hint; returns the record (or null)
function presetWrite(registry, name) {
  const d = (registry && registry[name]) || null;
  if (!d) return null;
  for (const id in (d.set || {})) {
    const e = el(id);
    if (!e) { console.warn("preset " + name + ": no control #" + id); continue; }
    const v = d.set[id];
    if (typeof v === "boolean") e.checked = v; else e.value = String(v);
  }
  const h = el("demohint");
  if (h) { h.innerHTML = d.hint || ""; h.style.display = d.hint ? "block" : "none"; }
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
// syncLabels, rebuild, applyIC, applyControls, autoDiss, wireDrawEditor, runSelfTest)
// -- the arrangement autoDiss above already relies on: common.js is loaded first, but
// its bodies only run once the app script has declared them.

// the slider readouts every page has. Each app's syncLabels() calls this and then adds
// its own (3D: z_diss_k, sigma_z, chi) -- the shared numbers exist once.
function syncCommonLabels() {
  const eps = uiEps(), amp = uiAmp(), fs = uiFshell();
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
}
// forcing on/off: the two eps sliders and their lock follow the checkbox
function syncForceEnabled() {
  for (const id of ["rEpsP", "rEpsM", "cbEpsLock"]) el(id).disabled = !el("cbForce").checked;
}

// bring the page into the state a preset (or a fresh boot) asks for
function bootApply(pre) {
  syncIC();
  syncLabels();
  syncForceEnabled();
  rebuild();
  cardsLayout(pre && pre.layout);
}
// every control whose handler is the same in both apps.
//   opts.presets    the app's preset registry
//   opts.sliders    extra live-parameter slider ids beyond the shared ones
//   opts.rebuildOn  extra <select> ids that force a full rebuild (3D: selLz)
function wireCommonControls(opts) {
  const s = el("selPreset");
  if (s) s.onchange = () => bootApply(presetWrite(opts.presets, s.value));
  el("btnRun").onclick = () => { running = !running; el("btnRun").textContent = running ? "Pause" : "Run"; };
  // hide/show the whole #controls block from the always-visible topbar. Pure display
  // toggle: nothing is re-read on show, so hidden controls keep their state.
  const bp = el("btnParams");
  if (bp) bp.onclick = () => {
    const c = el("controls"), hide = c.style.display !== "none";
    c.style.display = hide ? "none" : "";
    bp.textContent = hide ? "show params" : "hide params";
  };
  el("btnReset").onclick = () => { solver.p.seed = uiParams().seed; applyIC(); };
  el("selIC").onchange = () => { syncIC(); applyIC(); syncLabels(); };
  // amplitudes are stored OUTSIDE the drawing (REFINE_PLAN G.3): the sliders scale the
  // normalized zeta+- at apply time, so a release re-applies in every preset, drawn
  // ones included -- pause, move the slider, Reset really does rescale.
  lockPair("rAmpP", "rAmpM", "cbAmpLock", syncLabels, () => { if (!icDraw.on) applyIC(); });
  lockPair("rEpsP", "rEpsM", "cbEpsLock", applyControls);
  el("selHyper").onchange = applyControls;
  el("btnAutoDiss").onclick = autoDiss;
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
// `blurPx`: if the canvas supports ctx.filter, the gaussian is done here; otherwise the
// caller applies icGaussBlur, the exact separable periodic gaussian (icLetterField handles both).
function icGlyphRaster(text, nx, ny, cover, blurPx) {
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
  const canFilter = icCanvasBlurOK(c);
  if (canFilter && blurPx > 0) c.filter = "blur(" + blurPx + "px)";
  c.fillStyle = "#fff";
  c.fillText(text, nx / 2, ny / 2);
  if (canFilter) c.filter = "none";
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
  out.blurred = canFilter && blurPx > 0;
  return out;
}
// does this 2D context honour ctx.filter? (Safari <17 and some engines ignore it)
function icCanvasBlurOK(c) {
  try { c.filter = "blur(2px)"; const ok = c.filter === "blur(2px)"; c.filter = "none"; return ok; }
  catch (e) { return false; }
}

// Separable PERIODIC gaussian blur, truncated at 3.5 sigma with the kernel renormalized
// (it keeps 0.9995 of the mass). This is the fallback for engines whose 2D context
// ignores ctx.filter -- and the only path in node.
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
function icZetaFields(zp, zm, g, ampP, ampM) {
  const nrs = g.nx * g.ny, n = g.nz * nrs;
  const phi = new Float32Array(n), psi = new Float32Array(n);
  const P = icShearStats(zp, g), M = icShearStats(zm, g);
  const kp = P.gradMax > 0 ? ampP / P.gradMax : 0;
  const km = M.gradMax > 0 ? ampM / M.gradMax : 0;
  for (let k = 0; k < g.nz; k++) {
    const o = k * nrs, mp = P.mean[k], mm = M.mean[k];
    for (let i = 0; i < nrs; i++) {
      const a = kp * (zp[o + i] - mp), b = km * (zm[o + i] - mm);
      phi[o + i] = 0.5 * (a + b);
      psi[o + i] = 0.5 * (a - b);
    }
  }
  return { phi: phi, psi: psi };
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
    ? icGlyphRaster(String(text).charAt(0), nx, ny, cover, sigma) : null;
  if (!f) f = icFallbackBlob(nx, ny);
  else if (!f.blurred) f = icGaussBlur(f, nx, ny, sigma);
  return f;
}

// The "letters" IC as a pair of stored potentials: one glyph per Elsasser potential,
// times a gaussian z-envelope of peak 1 in 3D (`env` = [envPlus, envMinus], each nz
// floats; null in 2D). Shared by both apps -- the only difference between them is that
// pair of envelopes.
function icLetterZeta(g, env) {
  const nrs = g.nx * g.ny, out = [];
  for (let s = 0; s < 2; s++) {
    const glyph = icLetterField(IC_LETTERS.charAt(s) || IC_LETTERS.charAt(0),
                                g.nx, g.ny, g.Lx, g.Ly);
    if (!env) { out.push(glyph); continue; }
    const f = new Float32Array(g.nz * nrs), e = env[s];
    for (let k = 0; k < g.nz; k++) {
      const o = k * nrs, a = e[k];
      for (let i = 0; i < nrs; i++) f[o + i] = a * glyph[i];
    }
    out.push(f);
  }
  return { zp: out[0], zm: out[1] };
}

// The whole CPU IC path for the two potential-based presets, in one place: pick the
// stored zeta+- pair, then normalize + map through icZetaFields.
//   preset "custom"  the drawing;  anything else  the letters
//   env              3D letter z-envelopes (see icLetterZeta), null in 2D
function icPresetFields(q, preset, ampP, ampM, env) {
  const g = icDrawGrid(q);            // also the geometry record for the letter path
  icEq.on = false;                    // only an equilibrium builder turns this back on
  const B = IC_BUILDERS[preset];
  if (B) return B.fields(g);
  const z = preset === "custom" ? { zp: icDraw.zp, zm: icDraw.zm } : icLetterZeta(g, env);
  return icZetaFields(z.zp, z.zm, g, ampP, ampM);
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
//   fields  (g) -> {phi, psi}, on the geometry record icDrawGrid returns
// 2D only: the equilibria are 2D objects and the 3D page never lists them in #selIC.
const IC_BUILDERS = {};
function icRegister(name, rec) { IC_BUILDERS[name] = rec; }
// What the island-width chart needs to know about the LIVE equilibrium. `on` and `curv`
// are what islandWidth reads; `a` and `w0` (the initial width) are the record the node
// checks assert the builders against. An equilibrium builder that leaves `on` false --
// KH -- keeps the island chart on its "needs the tearing IC" placeholder.
const icEq = { on: false, a: 0, curv: 0, w0: 0 };
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
// broadcast a 1D x profile (plus an optional k_y seed) into a plane
function icPlaneFromX(prof, seed, g) {
  const nx = g.nx, ny = g.ny, out = new Float32Array(nx * ny), ky = 2 * Math.PI / g.Ly;
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      out[i * ny + j] = prof[i] + (seed ? seed[i] * Math.cos(ky * j * g.Ly / ny) : 0);
    }
  }
  return out;
}

// KH [20]: u_y is the double-tanh above; B is the same profile scaled by b0, so B lies
// in plane along y. Ideal 2D MHD stabilizes the layer once b0 >= U0 (the shear is then
// slower than the Alfven speed tying the field lines together); resistivity softens that
// threshold rather than removing it. The seed sits on BOTH layers -- they are independent.
icRegister("kh", {
  rows: ["rowEq", "rowKH"], hyper: 1,
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
    // out of Phase J.
    icEq.a = a;
    return { phi: icPlaneFromX(ph, sd, g), psi: icPlaneFromX(ps, null, g) };
  }
});

// Tearing [19]: psi_eq = psi0 sech^2((x - Lx/2)/a) (Numata / Loureiro style -- net-flux
// free and exponentially periodic for a << Lx), phi_eq = 0. b_y = psi_eq' vanishes at
// x = Lx/2, so that is the resonant surface of EVERY k_y mode and the line the island
// chart reads. The seed perturbs psi at k_y = 2pi/Ly with the same even-in-x envelope, so
// its value AT the surface is exactly the slider: psitilde(x_s) = A, and the initial
// island width is 4 sqrt(A/|psi_eq''|).
icRegister("tearing", {
  rows: ["rowEq", "rowTear"], hyper: 1,
  fields: g => {
    const psi0 = icEqNum("rEqPsi0", 1.65);
    const a = icEqNum("rEqA", 0.1) * g.Lx, A = icEqPert();
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
    icEq.on = true; icEq.a = a;
    icEq.curv = Math.abs(icD2(ps, Math.round(0.5 * nx), g.Lx / nx));
    icEq.w0 = icEq.curv > 0 ? 4 * Math.sqrt(A / icEq.curv) : 0;
    return { phi: new Float32Array(nx * g.ny), psi: icPlaneFromX(ps, sd, g) };
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

// The equilibrium presets' own control rows, as controlsBuild spec fragments -- next to
// the builders that read them. The 2D page passes them through ctrlGrpIC({extra: ...});
// the 3D page passes neither these nor the presets, so its #selIC never offers them.
const CTRL_ROWS_EQ = [
  { id: "rowEq", hide: true, items: [
    { k: "lab", t: "a / L<sub>x</sub>" },
    { k: "rng", id: "rEqA", min: 0.02, max: 0.2, step: 0.005, v: 0.1,
      ti: "equilibrium layer width, as a fraction of Lx" }, { k: "val", id: "vEqA" },
    { k: "lab", t: "seed" },
    { k: "rng", id: "rEqPert", min: -6, max: -1, step: 0.1, v: -3,
      ti: "amplitude of the k_y = 2pi/Ly perturbation (log10)" }, { k: "val", id: "vEqPert" }
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
    { k: "val", id: "vEqPsi0" }
  ] },
  { k: "hintdiv", id: "vEqInfo" }
];
// the sliders above, in the order wireCommonControls should wire them (each re-applies
// the IC on release, exactly like the zeta amplitudes)
const EQ_SLIDERS = ["rEqA", "rEqPert", "rEqU0", "rEqB0", "rEqPsi0"];
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
        "  W(0) &asymp; " + (4 * Math.sqrt(icEqPert() / c)).toPrecision(3);
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
  running = false; el("btnRun").textContent = "Run";
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
function icEditLeave(mode) {
  if (!icDraw.on) return;
  if (mode === "cancel" && icDraw.snap) {
    icDraw.zp.set(icDraw.snap.zp); icDraw.zm.set(icDraw.snap.zm);
    icDraw.has = icDraw.snap.has;
  }
  icDraw.snap = null;
  icDraw.on = false; icDraw.down = false; icDraw.last = null;
  icEditShow(false);
  el("btnRun").disabled = false; el("btnEdit").disabled = false;
  if (mode === "run") {
    applyIC();
    running = true; el("btnRun").textContent = "Pause";
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
  el("selPaint").onchange = () => { if (icDraw.on) { icEditCaption(); icDrawPreview(); } };
  for (const id of (cfg.sliders || [])) {
    el(id).oninput = () => { syncLabels(); if (icDraw.on) icEditCaption(); };
  }
}

// which IC rows apply to the selected preset -- the same rule in both apps (the app
// names its own extra rows through cfg.icRows, so nothing here guesses at ids).
function icSyncRows() {
  const p = el("selIC").value, isL = p === "letters", isC = p === "custom";
  for (const id of ["rAmpP", "rAmpM", "cbAmpLock"]) el(id).disabled = !isL && !isC;
  el("rowDraw").style.display = isC ? "" : "none";
  for (const id of ((icDraw.cfg && icDraw.cfg.icRows) || [])) {
    el(id).style.display = (isL || isC) ? "" : "none";
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
let frameHook = null, readoutExtra = null;
const cardsThrottle = { spec: 0, cut: 0 };
const _chartsOf = t => cards.chart.filter(c => c.type() === t);
// cards fed by one readback SOURCE: a type's `src` (island rides the cut line) or, by
// default, its own name. One readback serves every card that consumes it.
const _chartsBySrc = s => cards.chart.filter(c => (CHART_TYPES[c.type()].src || c.type()) === s);
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
    }
    // one display chain run per card per rendered frame: same state, own quantity.
    // (the editor view hides them all -- do not render into detached canvases)
    if (!icDraw.on) for (const d of cards.disp) d.render();
    await device.queue.onSubmittedWorkDone();
    const ms = performance.now() - t0;
    if (running) {
      if (ms < 11 && stepsPerFrame < 64) stepsPerFrame++;
      else if (ms > 22 && stepsPerFrame > 1) stepsPerFrame--;
      const sps = n / (Math.max(ms, 0.05) / 1000);
      spsSmooth = spsSmooth ? 0.9 * spsSmooth + 0.1 * sps : sps;
    }
    const s = await solver.readStats();
    if (frameHook) await frameHook(solver);
    if (!solver) continue;                    // retired while we were awaiting
    const extra = readoutExtra ? readoutExtra() : "";
    // the sticky bar carries the one line that must always be visible; the rest
    // goes under the displays. Both come from this one stats readback.
    el("steps").textContent =
      "t " + s[1].toFixed(3) + "  step " + solver.nsteps +
      "  dt " + s[0].toExponential(2) + "  " + (running ? spsSmooth.toFixed(0) + " steps/s" : "paused");
    el("readout").textContent =
      "Ekin = " + s[2].toExponential(5) + "  Emag = " + s[3].toExponential(5) +
      "\ns+   = " + s[4].toExponential(3) + "   s- = " + s[5].toExponential(3) +
      (extra ? "\n" + extra : "");

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
    // (snapshot: a close button can splice cards.disp while we are awaiting)
    for (const d of cards.disp.slice()) {
      if (!d.showArrows()) { d.clearArrows(); continue; }
      const tnow = performance.now();
      if (tnow - d.arrowAt <= 100) continue;
      d.arrowAt = tnow;
      const sv = solver;
      const av = await sv.readArrows(d.ci);
      if (sv === solver && cards.disp.indexOf(d) >= 0 && d.showArrows()) d.drawArrows(av, sv.nax, sv.nay);
    }

    // cut trace: same throttle / guard idiom as the arrows. SELF-CONTAINED since
    // Phase H -- it runs its own line prep and depends on no display card, only on
    // its own z plane (2D: always 0), so one readback serves every card on that plane.
    const cutCards = _chartsBySrc("cut");
    if (cutCards.length && performance.now() - cardsThrottle.cut > 100) {
      cardsThrottle.cut = performance.now();
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
        for (const c of cutCards) c.draw(planes.get(cards.cfg.zsliceOf(c)));
      }
    }

    // spectra: a full extra pass over the fields + a map round trip -> throttle hard
    const specCards = _chartsBySrc("spectrum");
    const now = performance.now();
    if (specCards.length && now - cardsThrottle.spec > 300) {
      cardsThrottle.spec = now;
      const sv = solver;
      const sp = await sv.readSpectrum();
      if (sv === solver) {
        const d = { perp: sp.perp, nb: sv.nb, fshell: sv.p.fshell, par: sp.par, parKfac: sp.parKfac };
        for (const c of specCards) c.draw(d);
      }
    }
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
  s.p.epsP = 0.5; s.p.epsM = 0.5; s.uploadCfg();
  s.setIC(true);                                   // quiescent start
  for (let i = 0; i < 600; i++) { s.step(1); if (i % 100 === 99) await device.queue.onSubmittedWorkDone(); }
  const a = await s.readStats();
  let clipped = 0;
  for (let i = 0; i < 2000; i++) {
    s.step(1);
    if (i % 200 === 199) {
      await device.queue.onSubmittedWorkDone();
      const q = await s.readStats();
      if (Math.abs(q[4]) >= 0.999 || Math.abs(q[5]) >= 0.999) clipped++;
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
