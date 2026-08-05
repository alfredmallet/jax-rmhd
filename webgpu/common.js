"use strict";
// ===========================================================================
// common.js -- everything the 2D (rmhd2d.html) and 3D (rmhd3d.html) WebGPU RMHD
// apps share: RNG, reference-vector helpers, the FFT kernel template and the
// dimension-agnostic WGSL snippets, device bring-up, the chart / overlay drawing,
// the self-test table and the frame loop.
//
// What deliberately does NOT live here: makeGrid, buildShaders' physics kernels
// (prepGrads, bracket, nlAssemble, forcingAdd, stage, energy, spectrum, scale, ou,
// icFinish, prepDisp, sliceExtract, faceExtract) and the whole Solver class. Those
// are per-app: the 2D and 3D versions differ in index layout, dispatch shape and
// the linear operator, and merging them would cost more in indirection than the
// duplication costs in lines.
//
// Loaded as a plain classic <script src="common.js"> BEFORE each app's inline
// script, so top-level `let`/`const`/`function` here are visible to it (and must
// not be redeclared there). Works from file://; the reference vectors stay inlined
// in the HTML because fetch() does not.
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

// |(ux,uy)| in place: a <- sqrt(a^2 + b^2), so the existing max-reduce and
// colorize see the (non-negative) magnitude in the display buffer without a new binding.
function vecMagWGSL(pre, count, wg) {
  return pre + `
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  if (i >= ${count}) { return; }
  let x: f32 = a[i];
  let y: f32 = b[i];
  a[i] = sqrt(x * x + y * y);
}`;
}

// subsample the vector field for the arrow overlay (point sample at the cell corner)
function vecGatherWGSL(pre, AD) {
  return pre + `
const SX: u32 = ${AD.sx}u;
const SY: u32 = ${AD.sy}u;
const NAX: u32 = ${AD.nax}u;
const NAY: u32 = ${AD.nay}u;
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
function cutGatherWGSL(pre) {
  return pre + `
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
function colorizeWGSL(pre, modeStruct) {
  return pre + `
${modeStruct}
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
  let v: f32 = f[gid.x * NY + gid.y] / max(mx[0], 1e-30);
  // signed fields: vmin=-vmax, vmax=+vmax (imshow(..., cmap="afmhot", vmin=-s, vmax=s));
  // magnitude modes are already non-negative and map straight onto [0,1].
  var x: f32;
  if (md.mode >= 4u) { x = v; } else { x = 0.5 * (clamp(v, -1.0, 1.0) + 1.0); }
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

let device = null, ctx = null, canvasFormat = "bgra8unorm";
let solver = null, running = false, stepsPerFrame = 1, spsSmooth = 0;

// adapter + device + canvas context. opts.maxLimits asks the adapter for its own
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

  const cv = el("cv");
  ctx = cv.getContext("webgpu");
  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format: canvasFormat, alphaMode: "opaque" });
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
let cxEn = null, cxSp = null, cxCut = null, specAt = 0, cutAt = 0, cutDrawn = false;

function chartCtx(id, w, h) {
  const cv = el(id), dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  cv.style.width = w + "px"; cv.style.height = h + "px";
  const c = cv.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);   // set once; all drawing is in logical px
  c.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  c.textBaseline = "alphabetic";
  return c;
}
// ---------------------------------------------------------------------------
// vector overlay: a transparent 2D canvas sitting exactly on top of #cv
// ---------------------------------------------------------------------------
// Orientation. The render fragment samples the field texture with v = 1 - uv.y, so
// texel row iy sits at screen row iy counted DOWNWARD (and column ix at screen x = ix,
// counted rightward). The 2D canvas y axis also points down, so the arrow for grid
// point (ix,iy) goes at (ix,iy) with no position flip, and the physical +y component
// -- which points DOWN on screen because of that same texture flip -- is drawn as
// canvas dy = +uy. Relative to a y-up plotting convention that is a sign flip of uy;
// here it is exactly what keeps the arrows consistent with the colour field.
const VEC_SIZE = 512;
let vecCx = null, vecDrawn = false, arrowAt = 0;
function vecCtx() {
  if (vecCx) return vecCx;
  const cv = el("cvVec"), dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(VEC_SIZE * dpr); cv.height = Math.round(VEC_SIZE * dpr);
  vecCx = cv.getContext("2d");
  vecCx.setTransform(dpr, 0, 0, dpr, 0, 0);   // all drawing in a logical 512x512 space
  return vecCx;
}
function clearArrows() {
  if (!vecDrawn) return;
  vecCtx().clearRect(0, 0, VEC_SIZE, VEC_SIZE);
  vecDrawn = false;
}
function drawArrows(a, nax, nay) {
  const c = vecCtx(), W = VEC_SIZE, H = VEC_SIZE;
  c.clearRect(0, 0, W, H);
  vecDrawn = true;
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

function drawEnergy() {
  if (!cxEn) return;
  const c = cxEn, P = PADE, x0 = P.l, x1 = EW - P.r, y0 = P.t, y1 = EH - P.b;
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
function drawSpectrum(bins, nb, fshell, par, parKfac) {
  if (!cxSp) return;
  const c = cxSp, P = PADS, x0 = P.l, x1 = SW - P.r, y0 = P.t, y1 = SH - P.b;
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
function drawCut(vals, Ly, signed) {
  if (!cxCut) return;
  const c = cxCut, P = PADC, x0 = P.l, x1 = CW - P.r, y0 = P.t, y1 = CH - P.b;
  chartFrame(c, CW, CH, P);
  c.textAlign = "left"; c.fillStyle = COL.txt;
  const n = vals ? vals.length : 0;
  let mx = 0;
  for (let i = 0; i < n; i++) { const v = Math.abs(vals[i]); if (isFinite(v) && v > mx) mx = v; }
  if (n < 2 || !(mx > 0)) { c.fillText("cut along y at x = Lx/2 — waiting…", x0 + 6, y0 + 13); return; }
  cutDrawn = true;
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
function clearCut() {
  if (!cxCut) return;
  cutDrawn = false;
  drawCut(null, 1, true);       // repaints the empty frame + "waiting" label
}

function initCharts() {
  cxEn = chartCtx("cvEnergy", EW, EH);
  cxSp = chartCtx("cvSpec", SW, SH);
  cxCut = chartCtx("cvCut", CW, CH);
  drawEnergy();
  drawSpectrum(new Float32Array(2), 1, [1, 3]);   // "waiting" placeholder
  drawCut(null, 1, true);
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
// main loop (identical in both apps: step, render, read back, draw)
// ---------------------------------------------------------------------------
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
    solver.render(ctx);
    await device.queue.onSubmittedWorkDone();
    const ms = performance.now() - t0;
    if (running) {
      if (ms < 11 && stepsPerFrame < 64) stepsPerFrame++;
      else if (ms > 22 && stepsPerFrame > 1) stepsPerFrame--;
      const sps = n / (Math.max(ms, 0.05) / 1000);
      spsSmooth = spsSmooth ? 0.9 * spsSmooth + 0.1 * sps : sps;
    }
    const s = await solver.readStats();
    el("readout").textContent =
      "t = " + s[1].toFixed(4).padStart(10) + "    step = " + String(solver.nsteps).padStart(7) +
      "\ndt   = " + s[0].toExponential(4) + "   steps/s = " + (running ? spsSmooth.toFixed(1) : "-") +
      "\nEkin = " + s[2].toExponential(5) + "  Emag = " + s[3].toExponential(5) +
      "\ns+   = " + s[4].toExponential(3) + "   s- = " + s[5].toExponential(3);

    // energy trace: one sample per readback, but never a duplicate t while paused
    if (isFinite(s[1]) && isFinite(s[2]) && isFinite(s[3]) &&
        (!hist.t.length || s[1] > hist.t[hist.t.length - 1])) {
      histPush(s[1], s[2], s[3]);
      drawEnergy();
    }
    // arrow overlay: the gather already ran inside render()'s pass, so this is only a
    // copy + map round trip. ~10 Hz, one frame of lag, and it never runs per step.
    // (cube mode draws no slice, so both overlays are cleared there.)
    if (solver.mode >= 4 && !solver.cube) {
      const tnow = performance.now();
      if (tnow - arrowAt > 100) {
        arrowAt = tnow;
        const sv = solver;
        const av = await sv.readArrows();
        if (sv === solver && solver.mode >= 4 && !solver.cube) drawArrows(av, sv.nax, sv.nay);
      }
    } else clearArrows();

    // cut trace: same throttle / guard idiom as the arrows (the gather ran inside
    // render()'s pass; this is a copy + map round trip)
    if (!solver.cube) {
      const tnow = performance.now();
      if (tnow - cutAt > 100) {
        cutAt = tnow;
        const sv = solver;
        const cv = await sv.readCut();
        if (sv === solver && !solver.cube) drawCut(cv, sv.p.Ly, sv.mode < 4);
      }
    } else if (cutDrawn) clearCut();

    // spectra: a full extra pass over the fields + a map round trip -> throttle hard
    const now = performance.now();
    if (now - specAt > 300) {
      specAt = now;
      const sv = solver;
      const sp = await sv.readSpectrum();
      if (sv === solver) drawSpectrum(sp.perp, sv.nb, sv.p.fshell, sp.par, sp.parKfac);
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
