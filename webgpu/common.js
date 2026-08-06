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
function nbins(nx, ny) { return Math.floor(Math.min(nx, ny) / 3); }

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

// second stage of the energy reduction: (Ekin, Emag, dissipation rate) -> scalars
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

// (the display-chain kernels -- vecMag, vecGather, cutGather, colorize and the
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
              grid: "#232833", axis: "#39404d", txt: "#8a94a3",
              guide: "#7d8798", shell: "#5a6472", cut: "#c9d4e2" };
const HIST_MAX = 2000;
const hist = { t: [], ek: [], em: [] };

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
// the overlay context of ONE display card (all drawing in a logical 512x512 space)
function vecCtx(cv) {
  const dpr = Math.min(2, (typeof window !== "undefined" && window.devicePixelRatio) || 1);
  cv.width = Math.round(VEC_SIZE * dpr); cv.height = Math.round(VEC_SIZE * dpr);
  const c = cv.getContext("2d");
  if (c) c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return c;
}
function drawArrows(c, a, nax, nay) {
  const W = VEC_SIZE, H = VEC_SIZE;
  c.clearRect(0, 0, W, H);
  let mx = 0;
  for (let i = 0; i < nax * nay; i++) {
    const m = Math.hypot(a[2 * i], a[2 * i + 1]);
    if (m > mx) mx = m;
  }
  if (!isFinite(mx) || mx <= 0) return;
  const cw = W / nax, ch = H / nay;
  const sc = 0.9 * Math.min(cw, ch) / mx;     // longest arrow ~ 0.9 * subsample cell
  const ca = Math.cos(2.6), sa = Math.sin(2.6);
  const path = new Path2D();
  for (let ix = 0; ix < nax; ix++) {
    for (let iy = 0; iy < nay; iy++) {
      const i = ix * nay + iy;
      const dx = a[2 * i] * sc, dy = a[2 * i + 1] * sc;
      const len = Math.hypot(dx, dy);
      if (!(len > 0.6)) continue;             // sub-pixel (or NaN) -> skip
      const x0 = (ix + 0.5) * cw - 0.5 * dx, y0 = (iy + 0.5) * ch - 0.5 * dy;
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

function histReset() { hist.t.length = 0; hist.ek.length = 0; hist.em.length = 0; }
function histPush(t, ek, em) {
  if (hist.t.length >= HIST_MAX) {          // full -> keep every other sample
    for (const a of [hist.t, hist.ek, hist.em]) {
      let w = 0;
      for (let i = 0; i < a.length; i += 2) a[w++] = a[i];
      a.length = w;
    }
  }
  hist.t.push(t); hist.ek.push(ek); hist.em.push(em);
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
function legend(c, x, y, items) {
  let lx = x;
  for (const it of items) {
    c.strokeStyle = it[1]; c.lineWidth = 2; c.setLineDash(it[2] || []);
    c.beginPath(); c.moveTo(lx, y - 3); c.lineTo(lx + 12, y - 3); c.stroke();
    c.setLineDash([]);
    c.fillStyle = it[1]; c.fillText(it[0], lx + 15, y);
    lx += 15 + c.measureText(it[0]).width + 11;
  }
}

function drawEnergy(c) {
  if (!c) return;
  const P = PADE, x0 = P.l, x1 = EW - P.r, y0 = P.t, y1 = EH - P.b;
  chartFrame(c, EW, EH, P);
  const n = hist.t.length;
  c.textAlign = "left"; c.fillStyle = COL.txt;
  if (n < 2) { c.fillText("energy vs t — collecting…", x0 + 6, y0 + 13); return; }
  let lo = Infinity, hi = -Infinity, allPos = true;
  for (let i = 0; i < n; i++) {
    const a = hist.ek[i], b = hist.em[i], s = a + b;
    if (!(a > 0) || !(b > 0) || !(s > 0)) allPos = false;
    lo = Math.min(lo, a, b, s); hi = Math.max(hi, a, b, s);
  }
  if (!isFinite(lo) || !isFinite(hi)) return;
  const useLog = allPos && hi / lo >= 3;
  let vlo, vhi;
  if (useLog) { vlo = Math.log10(lo); vhi = Math.log10(hi); }
  else { vlo = Math.min(0, lo); vhi = hi; }
  if (!(vhi > vlo)) vhi = vlo + (useLog ? 1 : Math.max(1e-30, Math.abs(vlo)));
  const pad = 0.07 * (vhi - vlo); vlo -= pad; vhi += pad;
  const t0 = hist.t[0], t1 = hist.t[n - 1], ts = (t1 - t0) || 1;
  const X = t => x0 + (t - t0) / ts * (x1 - x0);
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
  c.fillText(t1.toFixed(2), x1, EH - 6);
  c.textAlign = "left";
  c.fillText("t = " + t0.toFixed(2), x0, EH - 6);

  c.save();
  c.beginPath(); c.rect(x0, y0, x1 - x0, y1 - y0); c.clip();
  c.lineWidth = 1.25;
  const series = [[COL.ek, i => hist.ek[i]], [COL.em, i => hist.em[i]],
                  [COL.et, i => hist.ek[i] + hist.em[i]]];
  for (const sr of series) {
    c.strokeStyle = sr[0]; c.beginPath();
    for (let i = 0; i < n; i++) {
      const x = X(hist.t[i]), y = Y(sr[1](i));
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.stroke();
  }
  c.restore();
  legend(c, x0 + 6, y0 + 12, [["E_kin", COL.ek], ["E_mag", COL.em], ["E_tot", COL.et]]);
  c.fillStyle = COL.txt; c.textAlign = "right";
  c.fillText(useLog ? "log y" : "lin y", x1 - 5, y0 + 12);
  c.textAlign = "left";
}

// bins: 2*nb perpendicular shell values [E_u | E_b] (solid).
// par (optional): 2*nzb parallel values [E_u | E_b] for |kz| bin b = 1..nzb, drawn
// DASHED on the same axes; parKfac converts a kz bin index to the same k/kunit units
// (so with a cubic box the two abscissae coincide).
function drawSpectrum(c, bins, nb, fshell, par, parKfac) {
  if (!c) return;
  const P = PADS, x0 = P.l, x1 = SW - P.r, y0 = P.t, y1 = SH - P.b;
  chartFrame(c, SW, SH, P);
  c.textAlign = "left"; c.fillStyle = COL.txt;
  const pu = [], pb = [];
  let hi = 0, lo = Infinity;
  for (let b = 1; b < nb; b++) {          // bin 0 is the (zero-energy) DC shell
    const u = bins[b], m = bins[nb + b];
    if (u > 0 && isFinite(u)) { pu.push(b, u); hi = Math.max(hi, u); lo = Math.min(lo, u); }
    if (m > 0 && isFinite(m)) { pb.push(b, m); hi = Math.max(hi, m); lo = Math.min(lo, m); }
  }
  const qu = [], qb = [];
  if (par && par.length >= 2) {
    const nzb = par.length >> 1, kf = parKfac || 1;
    for (let b = 1; b <= nzb; b++) {      // |kz| bins; kz = 0 has no place on a log axis
      const u = par[b - 1], m = par[nzb + b - 1], k = b * kf;
      if (u > 0 && isFinite(u)) { qu.push(k, u); hi = Math.max(hi, u); lo = Math.min(lo, u); }
      if (m > 0 && isFinite(m)) { qb.push(k, m); hi = Math.max(hi, m); lo = Math.min(lo, m); }
    }
  }
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
  // k^-5/3 guide, anchored on E_u just above the forcing shell
  const kA = Math.max(2, Math.min(nb - 1, Math.round(fshell[1])));
  let anch = 0;
  for (let i = 0; i < pu.length; i += 2) {
    if (pu[i] >= kA) { anch = pu[i + 1] * Math.pow(pu[i], 5 / 3); break; }
  }
  if (anch > 0) {
    c.strokeStyle = COL.guide; c.setLineDash([5, 4]);
    c.beginPath();
    c.moveTo(X(kA), Y(anch * Math.pow(kA, -5 / 3)));
    c.lineTo(X(nb), Y(anch * Math.pow(nb, -5 / 3)));
    c.stroke(); c.setLineDash([]);
  }
  c.lineWidth = 1.4;
  for (const sr of [[pu, COL.ek, null], [pb, COL.em, null],
                    [qu, COL.ek, [5, 3]], [qb, COL.em, [5, 3]]]) {
    const a = sr[0];
    if (a.length < 4) continue;
    c.strokeStyle = sr[1]; c.setLineDash(sr[2] || []); c.beginPath();
    for (let i = 0; i < a.length; i += 2) {
      const x = X(a[i]), y = Y(a[i + 1]);
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
    }
    c.stroke(); c.setLineDash([]);
  }
  c.restore();
  const items = [["E_u", COL.ek], ["E_b", COL.em], ["k^-5/3", COL.guide, [4, 3]]];
  if (qu.length >= 4 || qb.length >= 4) {
    items.splice(2, 0, ["E_u(k∥)", COL.ek, [5, 3]], ["E_b(k∥)", COL.em, [5, 3]]);
  }
  legend(c, x0 + 6, y0 + 12, items);
}

// ---------------------------------------------------------------------------
// cut trace: the displayed scalar along y at fixed x = Lx/2 (one grid line of the
// displayed slice). Signed quantities get a symmetric +-max axis, magnitudes [0,max].
// ---------------------------------------------------------------------------
function drawCut(c, vals, Ly, signed) {
  if (!c) return;
  const P = PADC, x0 = P.l, x1 = CW - P.r, y0 = P.t, y1 = CH - P.b;
  chartFrame(c, CW, CH, P);
  c.textAlign = "left"; c.fillStyle = COL.txt;
  const n = vals ? vals.length : 0;
  let mx = 0;
  for (let i = 0; i < n; i++) { const v = Math.abs(vals[i]); if (isFinite(v) && v > mx) mx = v; }
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
  c.strokeStyle = COL.cut; c.lineWidth = 1; c.beginPath();
  for (let j = 0; j < n; j++) {
    const x = X(j), y = Y(vals[j]);
    if (j === 0) c.moveTo(x, y); else c.lineTo(x, y);
  }
  c.stroke();
  c.restore();
  legend(c, x0 + 6, y0 + 12, [["cut @ x = Lx/2", COL.cut]]);
}
// ===========================================================================
// cards: ONE display-card class, ONE chart-card interface
// ===========================================================================
// A display card owns one WebGPU canvas + its arrow overlay + the selectors that
// drive it (quantity, per-card colormap, arrows on/off and -- in 3D -- its own z
// slice / peak tracker). Card index `ci` IS the solver's display-chain index, so
// "N cards" costs exactly N chains and nothing else: there is no dual-view flag,
// no second canvas context, no chain-0 special case anywhere.
//
// A chart card owns one 2D canvas and a TYPE (energy / spectrum / cut) taken from
// CHART_TYPES below; the frame loop asks each type for its data once per throttle
// window and hands the result to every card of that type.
//
// The app supplies the parts that are genuinely per-app through cardsInit(cfg):
//   fields      [{v, t}] the quantity <option> list (3D adds its cube modes)
//   zslice      true in 3D: build the per-card z-source select + slice slider
//   nz()        current nz, for the slider range
//   zsliceOf(c) resolved plane index of card c (slider or tracked peak)
//   caption(c)  optional text appended to the card's caption
//   onLayout()  called after any add/remove/close, for app-side label syncing
// Everything else -- DOM, wiring, render order, throttles -- is shared.
const CARD_MAX_DISP = 3;
const CHART_TYPES = {
  energy: {
    label: "energy trace", w: EW, h: EH,
    draw: (c, d) => drawEnergy(c),
    hint: "E<sub>kin</sub>, E<sub>mag</sub>, E<sub>tot</sub> vs t (2000 points, decimated 2:1 when full)"
  },
  spectrum: {
    label: "spectra", w: SW, h: SH,
    draw: (c, d) => (d ? drawSpectrum(c, d.perp, d.nb, d.fshell, d.par, d.parKfac)
                       : drawSpectrum(c, new Float32Array(2), 1, [1, 3])),
    hint: "shell-binned E<sub>u</sub>(k), E<sub>b</sub>(k), ~3&times;/s"
  },
  cut: {
    label: "cut trace", w: CW, h: CH,
    draw: (c, d) => drawCut(c, d ? d.vals : null, d ? d.Ly : 1, d ? d.signed : true),
    hint: "the first display's field along y at x = L<sub>x</sub>/2, ~10&times;/s"
  }
};

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

class DisplayCard {
  constructor(ci) {
    const cfg = cards.cfg;
    this.ci = ci;
    const root = _mk("div", "card disp", cards.hostD);
    this.root = root;
    const head = _mk("div", "cardhead", root);
    this.selField = _sel(head, cfg.fields, "displayed quantity");
    if (cfg.zslice) {
      this.selZSrc = _sel(head, [{ v: "manual", t: "z slice" }, { v: "zp", t: "track z&#8314;" },
                                 { v: "zm", t: "track z&#8315;" }], "which z plane this card shows");
      this.rSlice = _mk("input", "zslider", head);
      this.rSlice.type = "range"; this.rSlice.min = "0"; this.rSlice.step = "1"; this.rSlice.value = "0";
      this.rSlice.max = String(Math.max(0, cfg.nz() - 1));
    }
    const al = _mk("label", "cbl", head);
    this.cbArrow = _mk("input", null, al);
    this.cbArrow.type = "checkbox"; this.cbArrow.checked = true;
    al.appendChild(document.createTextNode("arrows"));
    al.title = "vector overlay on the |u| / |b| / |z±| modes";
    this.selCmap = _sel(head, CMAP_NAMES.map((n, i) => ({ v: i, t: n })), "colormap");
    this.btnClose = _mk("button", "x", head);
    this.btnClose.innerHTML = "&times;";
    this.btnClose.title = ci === 0
      ? "the first display cannot be closed (the IC editor draws on it)" : "close this display";
    if (ci === 0) this.btnClose.disabled = true;

    const wrap = _mk("div", "cvwrap", root);
    this.wrap = wrap;
    this.cv = _mk("canvas", "cvmain", wrap);
    this.cv.width = 512; this.cv.height = 512;
    this.cvVec = _mk("canvas", "cvvec", wrap);
    this.cap = _mk("div", "viewcap", root);
    // the IC editor (common.js, one instance) paints on the FIRST display card
    if (ci === 0) {
      wrap.id = "cvwrap";
      this.cvEdit = _mk("canvas", "cvedit", wrap);
      this.cvEdit.id = "cvEdit";
      this.cvEdit.width = 512; this.cvEdit.height = 512;
      icDrawAttach();                 // this canvas replaced the previous card's
    }
    this.ctx = gpuCanvasCtx(this.cv);
    this.vcx = vecCtx(this.cvVec);
    this.vecDrawn = false;
    this.arrowAt = 0;

    const apply = () => { this.apply(); if (cards.cfg.onLayout) cards.cfg.onLayout(); };
    this.selField.onchange = apply;
    this.selCmap.onchange = apply;
    this.cbArrow.onchange = () => { if (!this.cbArrow.checked) this.clearArrows(); apply(); };
    if (this.selZSrc) this.selZSrc.onchange = apply;
    if (this.rSlice) this.rSlice.oninput = apply;
    this.btnClose.onclick = () => cardClose(this);
  }
  sel() { return parseInt(this.selField.value, 10) | 0; }
  cmap() { return parseInt(this.selCmap.value, 10) | 0; }
  zsrc() { return this.selZSrc ? this.selZSrc.value : "manual"; }
  // push this card's state into the live solver and relabel it
  apply() {
    if (!solver) return;
    const cfg = cards.cfg;
    if (this.rSlice) this.rSlice.max = String(Math.max(0, cfg.nz() - 1));
    solver.setDisplayMode(this.ci, this.sel(), cfg.zsliceOf(this), this.cmap());
    if (this.rSlice) this.rSlice.disabled = this.zsrc() !== "manual" || !!solver.cubeOf(this.ci);
    const o = this.selField.options[this.selField.selectedIndex];
    this.cap.innerHTML = (o ? o.innerHTML : "") + (cfg.caption ? cfg.caption(this) : "");
    if (!this.showArrows()) this.clearArrows();
  }
  showArrows() {
    return !!(this.cbArrow.checked && solver && dispIsVector(solver.modeOf(this.ci))
              && !solver.cubeOf(this.ci));
  }
  render() { if (this.ctx && solver) solver.render(this.ctx, this.ci); }
  drawArrows(a, nax, nay) { if (this.vcx) { drawArrows(this.vcx, a, nax, nay); this.vecDrawn = true; } }
  clearArrows() {
    if (this.vcx && this.vecDrawn) { this.vcx.clearRect(0, 0, VEC_SIZE, VEC_SIZE); this.vecDrawn = false; }
  }
  destroy() { if (this.root.parentNode) this.root.parentNode.removeChild(this.root); }
}

class ChartCard {
  constructor(type) {
    const root = _mk("div", "card chart", cards.hostC);
    this.root = root;
    const head = _mk("div", "cardhead", root);
    this.selType = _sel(head, Object.keys(CHART_TYPES).map(k => ({ v: k, t: CHART_TYPES[k].label })),
                        "what this chart shows");
    this.selType.value = type;
    this.btnClose = _mk("button", "x", head);
    this.btnClose.innerHTML = "&times;";
    this.btnClose.title = "close this chart";
    this.cv = _mk("canvas", "chart", root);
    this.hint = _mk("div", "hint", root);
    this.cx = null;
    this.build();
    // a retyped card must not wait out the old type's throttle window before it fills
    this.selType.onchange = () => {
      this.build(); this.draw(null);
      cardsThrottle.spec = 0; cardsThrottle.cut = 0;
    };
    this.btnClose.onclick = () => cardClose(this);
  }
  type() { return this.selType.value; }
  build() {
    const T = CHART_TYPES[this.type()];
    this.cv.style.aspectRatio = T.w + " / " + T.h;
    this.cx = chartCtx(this.cv, T.w, T.h);
    this.hint.innerHTML = T.hint;
  }
  draw(data) { CHART_TYPES[this.type()].draw(this.cx, data); }
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
  }
  return c;
}
function addChartCard(type) {
  const c = new ChartCard(CHART_TYPES[type] ? type : "energy");
  cards.chart.push(c);
  return c;
}
function cardClose(c) {
  const list = (c instanceof DisplayCard) ? cards.disp : cards.chart;
  const i = list.indexOf(c);
  if (i < 0) return;
  if (list === cards.disp && c.ci === 0) return;      // the editor's anchor card
  list.splice(i, 1);
  c.destroy();
  cardsSync();
}
// re-push every card into the solver (after a rebuild, a preset, or add/remove)
function cardsSync() {
  for (const d of cards.disp) d.apply();
  for (const c of cards.chart) if (c.type() === "cut") c.draw(null);
  cardsThrottle.spec = 0; cardsThrottle.cut = 0;
  if (cards.cfg && cards.cfg.onLayout) cards.cfg.onLayout();
}
// replace the whole layout (used by the presets and by boot)
function cardsLayout(L) {
  for (const d of cards.disp.slice()) { d.destroy(); }
  for (const c of cards.chart.slice()) { c.destroy(); }
  cards.disp.length = 0; cards.chart.length = 0;
  for (const s of (L && L.disp) || [{}]) addDisplayCard(s);
  if (!cards.disp.length) addDisplayCard();     // slot 0 always exists (editor anchor)
  for (const t of (L && L.charts) || ["energy", "spectrum", "cut"]) addChartCard(t);
  cardsSync();
}
// the card the single-instance overlays (IC editor, cut trace) hang off
function primaryCard() { return cards.disp.length ? cards.disp[0] : null; }
// clear the traces after an IC change / rebuild (one call, both apps)
function chartsReset() {
  histReset();
  cardsThrottle.spec = 0; cardsThrottle.cut = 0;
  for (const c of cards.chart) c.draw(null);
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

// marginally-resolved dissipation for the current hyper / resolution / injected power:
// diss ~ eps^(1/3) * k_c^(2/3 - 2*hyper) with k_c = nx/3, times the safety margin the
// repo's reference notebook uses. (uiParams / applyControls are per-app.)
function autoDiss() {
  const q = uiParams();
  // use the slider's eps even when the forcing checkbox is off (auto-diss for the
  // power you WOULD inject; with eps identically 0 the formula would degenerate)
  const epsTot = Math.pow(10, parseFloat(el("rEps").value));
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

// bring the page into the state a preset (or a fresh boot) asks for
function bootApply(pre) {
  syncIC();
  syncLabels();
  el("rEps").disabled = !el("cbForce").checked;
  rebuild();
  cardsLayout(pre && pre.layout);
}
// every control whose handler is the same in both apps.
//   opts.presets    the app's preset registry
//   opts.sliders    extra live-parameter slider ids beyond the shared five
//   opts.rebuildOn  extra <select> ids that force a full rebuild (3D: selLz)
function wireCommonControls(opts) {
  const s = el("selPreset");
  if (s) s.onchange = () => bootApply(presetWrite(opts.presets, s.value));
  el("btnRun").onclick = () => { running = !running; el("btnRun").textContent = running ? "Pause" : "Run"; };
  el("btnReset").onclick = () => { solver.p.seed = uiParams().seed; applyIC(); };
  el("selIC").onchange = () => { syncIC(); applyIC(); syncLabels(); };
  el("rAmp").oninput = syncLabels;      // takes effect on the next Reset / IC change
  // in "custom" the amplitude is a per-blob knob: it must NOT re-upload the drawing
  el("rAmp").onchange = () => { if (el("selIC").value !== "custom") applyIC(); };
  el("selHyper").onchange = applyControls;
  el("btnAutoDiss").onclick = autoDiss;
  el("cbForce").onchange = () => { el("rEps").disabled = !el("cbForce").checked; applyControls(); };
  for (const id of ["selRes"].concat(opts.rebuildOn || [])) el(id).onchange = rebuild;
  for (const id of ["rDiss", "rEps", "rTau", "rCfl", "rCflEvery"].concat(opts.sliders || [])) {
    el(id).oninput = applyControls;
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
// The knob these all serve is the AMPLITUDE of the Elsasser field: the ICs are stream
// functions z+- , and what the display (and the physics) sees is the perpendicular
// vector zhat x grad z+- , whose magnitude is |grad z+-|. So "amplitude = a" means
// max |grad z+-| = a, which icNormalizeShear enforces.

// The two letters of the "letters" IC preset: z+ gets the first, z- the second. Fixed
// (the free-text input was dropped in the mobile pass -- REFINE_PLAN F.5); both apps
// read it from here.
const IC_LETTERS = "AB";

// glyph -> [0,1] coverage mask, or null when there is no usable 2D canvas (node).
// `blurPx`: if the canvas supports ctx.filter, the gaussian is done here; otherwise the
// caller's 3-pass box blur does it (icLetterField handles both).
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

// 3-pass periodic box blur (the fallback gaussian, and the only path in node).
// 3 boxes of half-width r approximate a gaussian of sigma^2 = ((2r+1)^2 - 1)/4.
function icBoxRadius(sigma) {
  return Math.max(0, Math.round(0.5 * (Math.sqrt(4 * sigma * sigma + 1) - 1)));
}
function icBoxBlur(f, nx, ny, r, passes) {
  const np = passes === undefined ? 3 : passes;
  if (!(r >= 1) || np < 1) return f;
  const w = 2 * r + 1, inv = 1 / w;
  let a = Float32Array.from(f), b = new Float32Array(nx * ny);
  for (let p = 0; p < np; p++) {
    for (let j = 0; j < ny; j++) {                       // along x (stride ny)
      let s = 0;
      for (let k = -r; k <= r; k++) s += a[(((k % nx) + nx) % nx) * ny + j];
      for (let i = 0; i < nx; i++) {
        b[i * ny + j] = s * inv;
        s += a[((i + r + 1) % nx) * ny + j] - a[((((i - r) % nx) + nx) % nx) * ny + j];
      }
    }
    let t = a; a = b; b = t;
    for (let i = 0; i < nx; i++) {                       // along y (contiguous)
      const row = i * ny;
      let s = 0;
      for (let k = -r; k <= r; k++) s += a[row + (((k % ny) + ny) % ny)];
      for (let j = 0; j < ny; j++) {
        b[row + j] = s * inv;
        s += a[row + ((j + r + 1) % ny)] - a[row + ((((j - r) % ny) + ny) % ny)];
      }
    }
    t = a; a = b; b = t;
  }
  return a;
}

// max |grad f| over the periodic grid, 4th-order centred differences. This is the
// estimator icNormalizeShear inverts, so "amplitude" is defined by it; on the smooth
// (blurred) fields it is used on it agrees with the spectral gradient the GPU takes to
// well under a percent. The dealias mask applied at upload trims the achieved value by
// a little more than that, which is why the knob is documented as approximate.
function icGradMax(f, nx, ny, dx, dy) {
  const cx = 1 / (12 * dx), cy = 1 / (12 * dy);
  let mx = 0;
  for (let i = 0; i < nx; i++) {
    const im2 = ((i - 2 + 2 * nx) % nx) * ny, im1 = ((i - 1 + nx) % nx) * ny;
    const ip1 = ((i + 1) % nx) * ny, ip2 = ((i + 2) % nx) * ny, row = i * ny;
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
// remove the (gauge) mean and scale so that max |grad f| == amp. Returns the factor.
function icNormalizeShear(f, nx, ny, Lx, Ly, amp) {
  const n = nx * ny;
  let s = 0;
  for (let i = 0; i < n; i++) s += f[i];
  s /= n;
  for (let i = 0; i < n; i++) f[i] -= s;
  const gm = icGradMax(f, nx, ny, Lx / nx, Ly / ny);
  const k = gm > 0 ? amp / gm : 0;
  for (let i = 0; i < n; i++) f[i] *= k;
  return k;
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

// One Elsasser stream function from one character: rasterize, gaussian-smooth,
// zero-mean, scale to max |grad| = amp. `opts.cover` is the glyph size as a fraction
// of the shorter side (default 0.6), `opts.blur` the gaussian width as a fraction of it
// (default 1/32 -> 16 px at 512, 2 px at 64, with a hard 2 px floor).
//
// The smoothing is not cosmetic. A glyph edge is a step, and the SPECTRAL gradient the
// GPU takes of a step overshoots the finite-difference one badly (~19% for the raw
// raster), so the amplitude knob would mean two different things on the two sides of
// the upload. At sigma >= 2 px the two agree to ~0.1% (node check 2), which is what
// makes "max |zhat x grad z+-| = amp" a statement about the displayed field.
function icLetterField(text, nx, ny, Lx, Ly, amp, opts) {
  const o = opts || {};
  const cover = o.cover === undefined ? 0.6 : o.cover;
  const sigma = Math.max(2, (o.blur === undefined ? 1 / 32 : o.blur) * Math.min(nx, ny));
  let f = (text && String(text).length)
    ? icGlyphRaster(String(text).charAt(0), nx, ny, cover, sigma) : null;
  if (!f) f = icFallbackBlob(nx, ny);
  else if (!f.blurred) f = icBoxBlur(f, nx, ny, icBoxRadius(sigma), 3);
  icNormalizeShear(f, nx, ny, Lx, Ly, amp);
  return f;
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
// custom ("drawn") IC: the gaussian-blob editor
// ---------------------------------------------------------------------------
// Everything here is CPU-only. The editor keeps the two Elsasser stream functions
// z+ and z- at the live grid size, paints periodic gaussian blobs into them, previews
// them on an ordinary 2D canvas, and hands (phi, psi) = ((z+ + z-)/2, (z+ - z-)/2) to
// setICFromReal on "apply" -- the same convention icLetterField's callers use. No GPU
// work happens while editing.
//
// AMPLITUDE. As everywhere else in the IC code the knob is the amplitude of the
// DISPLAYED field, max |zhat x grad f| = max |grad f|. For f = P exp(-r^2/2 s^2),
// |grad f| = P (r/s^2) exp(-r^2/2 s^2) peaks at r = s with the value P/(s sqrt(e)), so
// the stream-function peak that realizes an amplitude `a` is P = a s sqrt(e).
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
// zp / zm are the accumulated Elsasser stream functions at the CURRENT grid; they are
// dropped (and the drawing is lost) whenever the grid changes, which is exactly when
// their layout stops meaning anything. `cfg` is the per-app hook set (icDrawWire).
const icDraw = {
  on: false, zp: null, zm: null, key: "", n: 0, has: false,
  cfg: null, wired: false, cv: null, down: false, neg: false, last: null, pending: false
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
// z+- -> the evolved variables, the same map icLetterField's callers use
function icDrawFields(q) {
  icDrawGrid(q);
  const n = icDraw.n, phi = new Float32Array(n), psi = new Float32Array(n);
  const zp = icDraw.zp, zm = icDraw.zm;
  for (let i = 0; i < n; i++) { phi[i] = 0.5 * (zp[i] + zm[i]); psi[i] = 0.5 * (zp[i] - zm[i]); }
  return { phi: phi, psi: psi };
}
// a blob in the TARGET field, expressed in z+- : "phi" is a blob in both, "psi" one of
// each sign, "zp"/"zm" just themselves.
const IC_TARGETS = { zp: [1, 0], zm: [0, 1], phi: [1, 1], psi: [1, -1] };
function icDrawBlob(q, target, x0, y0, iz, sigma, sigmaZ, amp) {
  const g = icDrawGrid(q), w = IC_TARGETS[target] || IC_TARGETS.zp;
  const peak = icBlobPeak(amp, sigma);
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
// #cvEdit is a plain 2D canvas laid over #cv (same box, same orientation): grid point
// (ix, iy) is image pixel (ix, iy), i.e. row iy=0 at the TOP, matching the render pass's
// v = 1 - uv.y flip and the arrow overlay. The nx-by-ny image is drawn through an
// offscreen canvas and scaled up by drawImage, so the smoothing matches the GPU path's
// linear sampler.
const icPrev = { cx: null, off: null, ox: null, w: 0, h: 0 };
function icDrawCtx() {
  if (icPrev.cx) return icPrev.cx;
  const cv = el("cvEdit");
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
  const v = icDrawPlane(q, cfg.target(), cfg.plane ? cfg.plane() : 0);
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
  // in the colormap the card being drawn on is using
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
  const cv = el("cvEdit");
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
  const sigma = icDrawSigma(cfg.sigmaFrac(), q.Lx, q.nx);
  if (icDraw.last) {
    const dx = icWrapDelta(p.x0 - icDraw.last.x, q.Lx), dy = icWrapDelta(p.y0 - icDraw.last.y, q.Ly);
    if (Math.hypot(dx, dy) < 0.5 * sigma) return;
  }
  icDraw.last = { x: p.x0, y: p.y0 };
  const nz = q.nz || 1;
  const sz = nz > 1 ? icDrawSigma(cfg.sigmaZFrac(), q.Lz, nz) : 0;
  // sign: the checkbox XOR the right button
  const s = ((cfg.neg && cfg.neg()) !== icDraw.neg) ? -1 : 1;
  icDrawBlob(q, cfg.target(), p.x0, p.y0, cfg.plane ? cfg.plane() : 0, sigma, sz, s * cfg.amp());
  icDrawPreviewSoon();
}

// enter / leave edit mode. Entering pauses the run (the sim must not advance under the
// preview) and shows the overlay; the app's onMode hook syncs its own buttons.
function icDrawSetMode(on) {
  const cfg = icDraw.cfg;
  if (!cfg) return;
  if (on && cfg.enabled && !cfg.enabled()) {
    showStatus("Switch the first display out of cube-face mode to draw an initial condition.", "info");
    return;
  }
  icDraw.on = !!on;
  icDraw.down = false; icDraw.last = null;
  const w = el("cvwrap");
  if (w && w.classList) w.classList.toggle("editing", icDraw.on);
  if (icDraw.on) {
    running = false;
    icDrawPreview();
  }
  if (cfg.onMode) cfg.onMode(icDraw.on);
}

// (re)bind the pointer handlers to the CURRENT #cvEdit. The overlay canvas belongs to
// the first display card, so it is a different element after every card-layout change
// (a preset switch rebuilds the cards) -- binding once at boot would leave the editor
// wired to a detached canvas. Called from the DisplayCard constructor.
function icDrawAttach() {
  const cv = el("cvEdit");
  if (!cv || !cv.addEventListener || icDraw.cv === cv) return;
  if (icDraw.cv) icDraw.on = false;          // the old overlay went away with its card
  icDraw.cv = cv;
  icDraw.down = false; icDraw.last = null;
  icPrev.cx = null;                          // the cached 2D context was that canvas's
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
//   target()      "zp" | "zm" | "phi" | "psi"
//   sigmaFrac()   sigma_perp as a fraction of Lx      sigmaZFrac()  sigma_z / Lz (3D)
//   amp()         amplitude knob (max |grad| of the blob)
//   neg()         negative-sign toggle                plane()       target z plane (3D)
//   enabled()     false where drawing makes no sense (3D cube mode)
//   sliders       the app's sigma slider ids (["rSigP"] in 2D, + "rSigZ" in 3D)
// The buttons, the edit-mode UI sync and the slider labels are identical in both apps
// and live here; only the knobs above are per-app.
function icDrawWire(cfg) {
  icDraw.cfg = cfg;
  cfg.onMode = on => {
    el("btnEdit").textContent = on ? "stop editing" : "edit IC";
    el("btnRun").disabled = on;
    if (on) el("btnRun").textContent = "Run";        // icDrawSetMode paused the run
  };
  icDrawAttach();
  if (icDraw.wired) return;
  icDraw.wired = true;
  el("btnEdit").onclick = () => icDrawSetMode(!icDraw.on);
  el("btnClear").onclick = () => { icDrawClear(); icDrawPreview(); };
  el("btnApply").onclick = () => {
    icDrawSetMode(false);
    applyIC();                                       // uploads the drawing
    running = true; el("btnRun").textContent = "Pause";
  };
  el("selPaint").onchange = () => { if (icDraw.on) icDrawPreview(); };
  for (const id of (cfg.sliders || [])) el(id).oninput = syncLabels;
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
    // one display chain run per card per rendered frame: same state, own quantity
    for (const d of cards.disp) d.render();
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

    // energy trace: one sample per readback, but never a duplicate t while paused
    if (isFinite(s[1]) && isFinite(s[2]) && isFinite(s[3]) &&
        (!hist.t.length || s[1] > hist.t[hist.t.length - 1])) {
      histPush(s[1], s[2], s[3]);
      for (const c of _chartsOf("energy")) c.draw(null);
    }
    // arrow overlay, per display card: the gather already ran inside that card's
    // render() pass, so this is only a copy + map round trip. ~10 Hz per card, one
    // frame of lag, and it never runs per step. Cube mode draws no slice -> cleared.
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

    // cut trace: same throttle / guard idiom as the arrows. It follows the FIRST
    // display card (Phase H makes the cut card self-contained).
    const cutCards = _chartsOf("cut"), pc = primaryCard();
    if (cutCards.length && pc) {
      const tnow = performance.now();
      if (solver.cubeOf(pc.ci)) {
        for (const c of cutCards) c.draw(null);
      } else if (tnow - cardsThrottle.cut > 100) {
        cardsThrottle.cut = tnow;
        const sv = solver;
        const vals = await sv.readCut(pc.ci);
        if (sv === solver && !solver.cubeOf(pc.ci)) {
          const d = { vals, Ly: sv.p.Ly, signed: dispIsSigned(sv.modeOf(pc.ci)) };
          for (const c of cutCards) c.draw(d);
        }
      }
    }

    // spectra: a full extra pass over the fields + a map round trip -> throttle hard
    const specCards = _chartsOf("spectrum");
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
