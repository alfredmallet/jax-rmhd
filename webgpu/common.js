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
// No depth sorting and no occlusion -- 2D canvas, one pass, everything visible.
function drawFieldLines(c, L, F) {
  if (!F) return;
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
  c.strokeStyle = "rgba(255,255,255,0.85)"; c.lineWidth = 2.6; c.stroke(path);
  c.strokeStyle = "rgba(20,60,150,0.90)"; c.lineWidth = 1.1; c.stroke(path);
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
// so E_tot = (E+ + E-)/2, which is the repo's Elsasser convention (taranis
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
// ... and a slope is only QUOTED when that window saw real, clean growth: at least
// MODE_FIT_RISE of ln A end to end (~0.43 decades -- the fp32 noise floor's jitter and
// the saturated stage's oscillation cannot fake it), with an R^2 of at least
// MODE_FIT_R2, so a window straddling two stages stays blank instead of quoting a
// number that is an average of neither.
const MODE_FIT_RISE = 1.0;
const MODE_FIT_R2 = 0.98;
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
// `rc` are the range-setting curves, each [ [k, v, k, v, ...], ... ] in k order.
function specFloor(rc, hi, lo) {
  const thr = hi * Math.pow(10, -SPEC_KNEE);
  let kd = Infinity, pre = Infinity;
  for (const cv of rc) {
    const a = cv[0];
    let p = -1, pv = 0;                          // this curve's own peak bin (argmax)
    for (let i = 0; i < a.length; i += 2) if (a[i + 1] > pv) { pv = a[i + 1]; p = i; }
    if (p < 0 || pv <= thr) continue;            // empty, or wholly below the crossing
    for (let i = p + 2; i < a.length; i += 2)    // the knee: walk RIGHT from the peak
      if (a[i + 1] <= thr) { kd = Math.min(kd, a[i]); break; }
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
const FIT_SNAP = 5e-3;
const FIT_FRACS = [[-5 / 3, "-5/3"], [-3 / 2, "-3/2"]];
function _fitFrac(p) {
  for (const f of FIT_FRACS) if (Math.abs(p - f[0]) < FIT_SNAP) return f;
  return null;
}
function fitIndex(v) {
  const p = parseFloat(v);
  if (!isFinite(p)) return FIT_FRACS[0][0];         // blank / NaN box: the default index
  const f = _fitFrac(p);
  return f ? f[0] : p;
}
function fitLabel(p) {
  const f = _fitFrac(p);
  return "k^" + (f ? f[1] : String(Math.round(p * 1000) / 1000));
}
// A in E = A k^p, pinned to a drawn series: its first point at or above kA, exactly where
// the old guide anchored (just above the forcing shell). 0 when the series has no such
// point -- an empty or still-filling chart draws no line.
function fitAnchor(pts, kA, p) {
  for (let i = 0; i < pts.length; i += 2) if (pts[i] >= kA) return pts[i + 1] * Math.pow(pts[i], -p);
  return 0;
}
function drawSpectrum(c, d, o) {
  if (!c) return;
  const P = PADS, x0 = P.l, x1 = SW - P.r, y0 = P.t, y1 = SH - P.b;
  chartFrame(c, SW, SH, P);
  c.textAlign = "left"; c.fillStyle = COL.txt;
  const bins = (d && d.perp) || new Float32Array(3), nb = (d && d.nb) || 1;
  const fshell = (d && d.fshell) || [1, 3];
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
  if (nb < 2 || !(hi > 0)) { c.fillText("spectra — waiting…", x0 + 6, y0 + 13); return; }
  // the curves the y range is read off: the perpendicular ones whenever they carry
  // anything, else (par-only cards) the parallel ones, which are the tail of `curves`
  const nPerp = wantPerp ? set.length : 0;
  const rc = hiP > 0 ? curves.slice(0, nPerp) : curves.slice(nPerp);
  const ymax = Math.log10(hi) + 0.3;
  // at least one decade always, so a flat or single-valued spectrum still has an axis
  const ymin = Math.min(ymax - 1,
    Math.log10(Math.max(specFloor(rc, hi, lo), hi * Math.pow(10, -SPEC_MAXDEC))) - 0.3);
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
  // the fit line E = A k^p, from kA (just above the forcing shell, where the old fixed
  // guide was anchored) to the last bin. `fit` = pin / amp / off; in "amp" mode an empty
  // or non-positive amplitude box falls back to the pinned anchor, so switching to it
  // never blanks the line before the user has typed anything.
  const kA = Math.max(2, Math.min(nb - 1, Math.round(fshell[1])));
  const fitMode = (o && o.fit) || "pin";
  const fitP = fitIndex(o && o.fitp);
  let anch = 0;
  if (fitMode !== "off") {
    anch = fitAnchor(wantPerp && curves.length ? curves[0][0] : [], kA, fitP);
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
  if (anch > 0) items.push([fitLabel(fitP), COL.guide, [4, 3]]);
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
           vis: v => v.fit === "amp" }]),
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
// Since K2.1 the FIELD LINES are a view on the same select: a whole-box object with no
// plane of its own, so it takes no tracker and no slider and is flagged out of the cut
// card with the cube entries.
const ZSRC_OPTS = [{ v: "manual", t: "z slice" }, { v: "zp", t: "track z&#8314;" },
                   { v: "zm", t: "track z&#8315;" }];
const ZSRC_CUBE = [{ v: "cube", t: "cube faces" }, { v: "cubezp", t: "cube + track z&#8314;" },
                   { v: "cubezm", t: "cube + track z&#8315;" }, { v: "lines", t: "field lines" }];
function _zSliceControls(card, head, cube) {
  const cfg = cards.cfg;
  if (!cfg.zslice) return;
  card.selZSrc = _sel(head, cube ? ZSRC_OPTS.concat(ZSRC_CUBE) : ZSRC_OPTS,
                      "which z plane this card uses" + (cube ? ", and whether it draws the cube faces or the field lines" : ""));
  card.rSlice = _mk("input", "zslider", head);
  card.rSlice.type = "range"; card.rSlice.min = "0"; card.rSlice.step = "1"; card.rSlice.value = "0";
  card.rSlice.max = String(Math.max(0, cfg.nz() - 1));
}
// the plane source of a card, with the view prefix stripped: every caller that resolves
// a plane (the app's zsliceOf / trackingOn) sees exactly the three pre-I2 values
// ("lines" owns no plane at all -- reporting it as "manual" is what keeps the trackers
// and the slider out of the lines view, K2.5)
function _zSrcPlane(v) {
  if (v === "lines") return "manual";
  return v.indexOf("cube") === 0 ? (v.slice(4) || "manual") : v;
}
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
    this.cap = _mk("div", "viewcap", root);
    this.gw = 0; this.gh = 0;
    this._resize();                       // sizes both canvases BEFORE the GPU context
    this.ctx = gpuCanvasCtx(this.cv);
    this.arr = null;                      // last arrow gather, and (3D) the field lines:
    this.lines = null;                    // two sources, one overlay canvas (see overlay())
    this.arrowAt = 0;
    this.wasLines = false;                // edge-trigger for the lines view's psi default

    const apply = () => { this.apply(); if (cards.cfg.onLayout) cards.cfg.onLayout(); };
    this.selField.onchange = apply;
    this.selCmap.onchange = apply;
    this.selCont.onchange = apply;
    this.selLev.onchange = apply;
    this.selBg.onchange = apply;
    this.cbArrow.onchange = apply;
    if (this.selZSrc) this.selZSrc.onchange = apply;
    if (this.rSlice) this.rSlice.oninput = apply;
    this.btnClose.onclick = () => cardClose(this);
  }
  sel() { return parseInt(this.selField.value, 10) | 0; }
  cmap() { return parseInt(this.selCmap.value, 10) | 0; }
  // the PLANE source, view prefix stripped (the app's zsliceOf / trackingOn use this)
  zsrc() { return _zSrcPlane(this.selZSrc ? this.selZSrc.value : "manual"); }
  // ... and the two VIEWS the same select carries instead of the one plane: the cube
  // faces, or (K2.1) the whole box's field lines with a transparent top face
  cubeView() { return !!this.selZSrc && this.selZSrc.value.indexOf("cube") === 0; }
  linesView() { return !!this.selZSrc && this.selZSrc.value === "lines"; }
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
    // ENTERING the lines view turns psi contours on: the transparent top face is the
    // point of the view (K2.3). From there the card's own contour select rules, "off"
    // included -- hence the edge trigger rather than a forced value.
    const lines = this.linesView();
    if (lines && !this.wasLines && !this.contOn()) this.selCont.value = String(DISP_PSI);
    this.wasLines = lines;
    solver.setDisplayMode(this.ci, this.sel(), cfg.zsliceOf(this), this.cmap(),
                          { cube: this.cubeView(), lines: lines, cont: this.cont(), nlev: this.nlev(),
                            plain: lines || (this.contOn() && this.plainBg()) });
    // the slider drives the displayed plane in the slice view and the TOP face in the
    // cube view, so it is live in both -- and dead whenever a tracker owns the plane, or
    // in the lines view, whose face is the top BOUNDARY of the box (K2.5)
    if (this.rSlice) this.rSlice.disabled = lines || this.zsrc() !== "manual";
    this.selLev.style.display = this.contOn() ? "" : "none";
    this.selBg.style.display = (this.contOn() && !lines) ? "" : "none";   // lines: always plain
    // the field selector is inert in the lines view (the lines are psi lines), so the
    // caption is the app's alone there
    const o = this.selField.options[this.selField.selectedIndex];
    this.cap.innerHTML = (o && !lines ? o.innerHTML : "") + (cfg.caption ? cfg.caption(this) : "");
    this.overlay();                       // the quantity / view may have retired an overlay
  }
  showArrows() {
    return !!(this.cbArrow.checked && !this.linesView() &&
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
  render() { if (this.ctx && solver) solver.render(this.ctx, this.ci); }
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
  setLines(L) { this.lines = L; if (this.linesView()) this.overlay(); }
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
    if (this.arr && this.showArrows()) drawArrows(c, this.arr.a, this.arr.nax, this.arr.nay, this.arrowFrame());
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
  // show/hide the options whose meaning depends on another one (the fit line's boxes)
  _optSync() {
    const v = this.optVals();
    for (const s of this.optEls) if (s.__optVis) s.style.display = s.__optVis(v) ? "" : "none";
  }
  build() {
    const T = CHART_TYPES[this.type()];
    // drop the previous type's controls (a select is appended AFTER the close button,
    // so the button is re-appended last to keep its margin-left:auto place)
    for (const s of this.optEls) this.head.removeChild(s);
    if (this.rSlice) { this.head.removeChild(this.selZSrc); this.head.removeChild(this.rSlice); }
    this.optEls = []; this.selZSrc = null; this.rSlice = null;
    const redraw = () => { this._optSync(); this.draw(null); cardsThrottle.spec = 0; cardsThrottle.cut = 0; };
    // an option is a <select> over `o`, or (k: "num") a small number box -- both end up
    // in optEls with the same __optId, so optVals() and the type's draw() see one shape.
    // `vis(vals)` optionally hides one when another option makes it meaningless.
    for (const spec of (T.opts ? T.opts(cards.cfg || {}) : [])) {
      let s;
      if (spec.k === "num") {
        s = _mk("input", "optnum", this.head);
        s.type = "number";
        if (spec.min !== undefined) s.min = String(spec.min);
        if (spec.step !== undefined) s.step = String(spec.step);
        if (spec.w) s.style.width = spec.w + "px";
        if (spec.ti) s.title = spec.ti;
        s.oninput = redraw;
      } else {
        s = _sel(this.head, spec.o.map(x => ({ v: x[0], t: x[1] })), spec.ti);
        s.onchange = redraw;
      }
      if (spec.v !== undefined) s.value = String(spec.v);
      s.__optId = spec.id;
      s.__optVis = spec.vis || null;
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
    if (state.plain !== undefined) c.selBg.value = state.plain ? "1" : "0";
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
  histReset(); islandReset(); modeReset();
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

// Tearing [19]: psi_eq = psi0 sech^2((x - Lx/2)/a) (Numata / Loureiro style -- net-flux
// free and exponentially periodic for a << Lx), phi_eq = 0. b_y = psi_eq' vanishes at
// x = Lx/2, so that is the resonant surface of EVERY k_y mode and the line the island
// chart reads. The seed perturbs psi at k_y = 2pi/Ly with the same even-in-x envelope, so
// its value AT the surface is exactly the slider: psitilde(x_s) = A, and the initial
// island width is 4 sqrt(A/|psi_eq''|).
icRegister("tearing", {
  rows: ["rowEq", "rowTear"], hyper: 1, src: true,
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
// Least-squares slope of ln A vs t over the trailing MODE_FIT_DT of sim-time -- the
// number to compare with the linear reference gamma = 0.267 U0 k_y. Only finite,
// positive samples count (a log has nothing to say about the others); the window must
// span time, must RISE by MODE_FIT_RISE (a flat, decaying, jittering or saturated trace
// has no growth rate to quote), and the fit must actually describe it (R^2 >=
// MODE_FIT_R2 -- a straddle of two stages is not a rate). Otherwise NaN, and the legend
// simply omits the readout.
function modeFitGamma(ts, as) {
  const t = [], y = [];
  let tl = NaN;
  for (let i = Math.min(ts.length, as.length) - 1; i >= 0; i--) {
    if (!(isFinite(ts[i]) && isFinite(as[i]) && as[i] > 0)) continue;
    if (isFinite(tl) && ts[i] < tl - MODE_FIT_DT) break;
    if (!isFinite(tl)) tl = ts[i];
    t.unshift(ts[i]); y.unshift(Math.log(as[i]));
  }
  const m = t.length;
  if (m < 4 || !(t[m - 1] > t[0]) || !(y[m - 1] - y[0] >= MODE_FIT_RISE)) return NaN;
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
    { k: "val", id: "vEqPsi0" },
    { k: "cbl", id: "cbEqSrc", t: "maintain equilibrium flux", v: true,
      ti: "add the static source S = -eta grad^2 psi_eq, which holds the equilibrium "
        + "against its own resistive decay (rebuilds the solver)" }
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
  // isP = a PACKET preset (letters, or the item-9 sinusoids): stored potentials that the
  // amp sliders normalize and the sigma_z row shapes. isC = the drawing, which is those
  // things AND owns the paint row.
  const p = el("selIC").value, isP = icIsPacketIC(p), isC = p === "custom";
  for (const id of ["rAmpP", "rAmpM", "cbAmpLock"]) el(id).disabled = !isP && !isC;
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
    // (snapshot: a close button can splice cards.disp while we are awaiting. A card that
    // is not showing arrows is simply skipped -- overlay() gates on showArrows(), so the
    // stale gather cannot reappear, and apply() has already redrawn the canvas.)
    for (const d of cards.disp.slice()) {
      if (!d.showArrows()) continue;
      const tnow = performance.now();
      if (tnow - d.arrowAt <= 100) continue;
      d.arrowAt = tnow;
      const sv = solver;
      const av = await sv.readArrows(d.ci);
      if (sv === solver && cards.disp.indexOf(d) >= 0) d.setArrows(av, sv.nax, sv.nay);
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
    if (specCards.length && now - cardsThrottle.spec > 300) {
      cardsThrottle.spec = now;
      const sv = solver;
      const sp = await sv.readSpectrum();
      if (sv === solver) {
        const d = Object.assign({ perp: sp.perp, nb: sv.nb, fshell: sv.p.fshell,
                                  par: sp.par, parKfac: sp.parKfac },
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
