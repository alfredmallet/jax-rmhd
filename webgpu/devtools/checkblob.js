// The SPACEBAR GAUSSIAN BLOB FORCING gate (c81527d / 0fd7406): the numbers, not the names.
//
//   node checkblob.js [webgpu-dir]        exit code 1 on any failure
//
// CI reports, never gates (the sibling scripts' contract): every row prints PASS / FAIL and
// the exit code is for a human reading the log.
//
// Why this file exists
// --------------------
// The byte-identity gates (`checksolver2d`, `checkiso`, `checkeigf`, `check2dspec`) pin the
// emitted WGSL against a BASE COMMIT. `blobBuild` is a kernel the base never emitted, so
// there is nothing on the other side to diff it against and `ADDED["rmhd2d.html"]` can only
// pin its NAME. Measured on 2026-08-30: flipping the sign inside the kernel
// (`vec2<f32>(cos(th), -sin(th))` -> `vec2<f32>(cos(th), sin(th))`, i.e. the blob placed at
// -x0 instead of x0) leaves ALL FOUR of those gates green. `checksolver2d`'s BLOB_SOLVER /
// BLOB_SHADERS blocks pin the JavaScript that DRIVES the kernel line for line, which is a
// real statement about the OU path being unmoved -- but nothing standing ever re-checked the
// physics. The only numerical check the feature ever had was a one-time out-of-tree DFT
// comparison by the implementing agent (c81527d's message, "3.2e-7 relative"), which is
// evidence about a tree that no longer exists.
//
// This file is that check, made standing. It is a DIFFERENTIAL cross-check: the production
// path -- the real `Solver.setBlobs` and the real emitted `blobBuild` -- against an
// INDEPENDENT transcription of the same mathematics written here, sharing nothing with the
// app but the grid definition and the physical specification of what a blob is.
//
// What is numerically verified, and what is not
// ---------------------------------------------
// EXECUTED and compared at round-off:
//   * `blobBuild`, BYTE FOR BYTE as `buildShaders` emits it, on wgsl_reflect's WGSL
//     interpreter (checkiso leg 9 / check2dspec leg 3's idiom) -- every k-dependent factor,
//     the phase, the dealias mask, the half/m index split and the write.
//   * `Solver.setBlobs`, the real method, called on a bare `{ p, _blobs }` receiver (it
//     touches nothing else, so no device is needed) -- the packed weight, the channel-half
//     placement, the per-channel cap and the zero-fill.
// The reference is built the other way round: the analytic gaussian
// `f(x) = P exp(-|x-x0|^2/2 sigma^2)` laid down in REAL SPACE on the grid, periodised over
// the box, and forward-transformed by a naive O(N^2) DFT written fresh below. `P` is not
// taken from `icBlobPeak` either: the blob's specification is "peak |grad f| = amp", so the
// reference finds `P` by numerically maximising |grad f| over r. So a wrong `2*pi`, a wrong
// `sigma^2`, a wrong `nx*ny/(Lx*Ly)`, a wrong `icBlobPeak`, a wrong sign in the phase or a
// wrong gaussian width all show up as a number that does not match.
// NOT verified here, and deliberately: the STEP wiring (that blob mode dispatches
// `blobBuild` over 2*nm modes in place of `ou` + `scale`, and that `sc[4]`/`sc[5]` are put
// back to 1) is `checksolver2d`'s BLOB_SOLVER text pin, and the buffer/bind-group plumbing
// is its leg 5. This file is about the arithmetic those lines set up.
//
// The interpreter's one quirk, and how the legs live with it
// ----------------------------------------------------------
// wgsl_reflect 1.5.0 reads a `@workgroup_size(N)` attribute as `parseInt(value[0])` -- the
// FIRST CHARACTER of the literal -- so `@workgroup_size(64)` runs 6 lanes per workgroup and
// `@workgroup_size(16)` runs 1. The lanes it does run are contiguous (`gid.x = lane +
// wgid*lanes`), so the house workaround applies unchanged: OVER-DISPATCH workgroups and lean
// on the kernel's own `if (idx >= 2u * NM) { return; }`. Every dispatch below asks for
// `2*NM` workgroups, which covers the whole buffer at any lane count from 1 up. The kernel
// TEXT is never edited -- and leg 1 proves the coverage rather than assuming it, by seeding
// the output buffer with a sentinel and demanding that not one cell still holds it.
//
// Grids are chosen so the comparison is round-off-limited and not aliasing-limited: with
// `Lx = 2*pi` every `kx`, `ky` is an exact integer in f32, `sigma` and the offsets are f32
// -exact, and sigma is fat enough (>= 0.375, i.e. ~2 cells at 32^2) that the periodic
// gaussian's alias sum is below 1e-10 of the peak across the whole retained band. No RNG:
// every field here is analytic or a fixed walk.
"use strict";
const fs = require("fs"), path = require("path");
const { pathToFileURL } = require("url");
const dir = path.resolve(process.argv[2] || path.join(__dirname, ".."));

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};
const e2 = v => (isFinite(v) ? v.toExponential(2) : String(v));

// boot() runs at require, its async tail a macrotask later (the checkonepage idiom)
const boot = (d, page, demo) => new Promise(res => {
  const env = require("./stubenv")(d, page, demo);
  setTimeout(() => res(env));
});
async function wgslMod() {
  const p = path.join(__dirname, "node_modules", "wgsl_reflect", "wgsl_reflect.module.js");
  if (!fs.existsSync(p)) return null;
  return await import(pathToFileURL(p).href);
}
const kconst = (src, n) => {
  const m = new RegExp("const " + n + ": (?:i32|u32|f32) = ([-+\\d.e]+)u?;").exec(src);
  return m ? parseFloat(m[1]) : NaN;
};
const bits = a => new Uint32Array(a.buffer, a.byteOffset, a.length);

// ---------------------------------------------------------------------------
// the two grids the legs run on
// ---------------------------------------------------------------------------
// SQ is the working grid: 32^2, Lx = Ly = 2*pi, so gridA holds exact integer k. RECT is the
// same box halved in y with half the modes, which is the only thing that separates the two
// factors of `w = ... * (nx*ny)/(Lx*Ly)` -- on a square box nx/Lx = ny/Ly and a swapped pair
// is invisible. `diss` / `hyper` / `fshell` are makeGrid's other inputs and reach nothing
// this file looks at (gridB's `lin_L` and `fmask` are the OU path's columns).
const SQ = { nx: 32, ny: 32, Lx: 2 * Math.PI, Ly: 2 * Math.PI, diss: 1e-13, hyper: 4, fshell: [1, 3] };
const RECT = { nx: 32, ny: 16, Lx: 2 * Math.PI, Ly: Math.PI, diss: 1e-13, hyper: 4, fshell: [1, 3] };

// the app's own emission + grid arrays for one parameter set, off the booted page
const GRIDSNIP = `function(P){
  const gr = makeGrid(P);
  const g = Object.assign({ nx: P.nx, ny: P.ny, Lx: P.Lx, Ly: P.Ly }, gr);
  const S = buildShaders(g);
  return { src: S.blobBuild, names: Object.keys(S), nm: gr.nm, nkx: gr.nkx, nky: gr.nky,
           gridA: Array.from(gr.gridA), gridB: Array.from(gr.gridB), nblob: BLOB_FORCE_MAX };
}`;
// the REAL setBlobs, on a bare receiver: it reads `this.p` and writes `this._blobs`, and
// nothing else -- no device, no buffers, no pipelines.
const PACK = `function(P, arr, list){ Solver.prototype.setBlobs.call({ p: P, _blobs: arr }, list); }`;

function gridOf(env, P) {
  const G = env.run(GRIDSNIP, P);
  G.gridA = Float32Array.from(G.gridA);
  G.gridB = Float32Array.from(G.gridB);
  return G;
}
// pack a blob list through the app's setBlobs, into a buffer pre-filled with a sentinel so
// the zero-fill has something to erase
const SENT_PACK = 7.5;
function pack(env, P, G, list) {
  const a = new Float32Array(2 * G.nblob * 4).fill(SENT_PACK);
  env.run(PACK, P, a, list);
  return a;
}

// ---------------------------------------------------------------------------
// the production kernel, executed
// ---------------------------------------------------------------------------
// `2*nm` workgroups is the over-dispatch the interpreter's lane-count quirk needs (header);
// the kernel's own bounds test is what makes it correct. `sentinel` is returned alongside so
// a caller can prove every cell was reached.
const SENT_FRC = 12345.5;
function runBlob(M, G, packed, seed) {
  const frc = seed ? Float32Array.from(seed) : new Float32Array(4 * G.nm).fill(SENT_FRC);
  new M.WgslExec(new M.WgslParser().parse(G.src)).dispatchWorkgroups(
    "main", [2 * G.nm, 1, 1], { 0: { 0: frc, 1: G.gridA, 2: G.gridB, 3: packed } });
  return frc;
}

// ---------------------------------------------------------------------------
// THE INDEPENDENT REFERENCE (nothing below is the app's)
// ---------------------------------------------------------------------------
// The blob's specification is a REAL-SPACE gaussian on the stream function whose peak
// |grad f| is `amp`. For f = P exp(-r^2/2s^2), |grad f| = P (r/s^2) exp(-r^2/2s^2); the
// potential peak P is recovered here by numerically maximising that over r, so the gate
// never borrows common.js's closed form (which is what mutation 8 attacks).
function potentialPeak(amp, sg) {
  const g = r => (r / (sg * sg)) * Math.exp(-0.5 * r * r / (sg * sg));
  let lo = 0, hi = 8 * sg;                       // g is unimodal on [0, inf): ternary search
  for (let i = 0; i < 200; i++) {
    const a = lo + (hi - lo) / 3, b = hi - (hi - lo) / 3;
    if (g(a) < g(b)) lo = a; else hi = b;
  }
  return amp / g(0.5 * (lo + hi));
}
// the 2/3 dealias mask, rebuilt from the rule rather than read off gridB (leg 4 pins the two
// against each other, so the "outside the cut" set this file asserts on is its own)
function dealiasRef(P, nkx, nky) {
  const cutx = P.nx / 3.0, cuty = P.ny / 3.0, de = new Float64Array(nkx * nky);
  for (let i = 0; i < nkx; i++) {
    const ix = (i < P.nx / 2) ? i : i - P.nx;
    for (let j = 0; j < nky; j++)
      de[i * nky + j] = ((ix / cutx) * (ix / cutx) + (j / cuty) * (j / cuty)) < 1.0 ? 1 : 0;
  }
  return de;
}
// the reference spectrum: the analytic gaussians of one channel, laid down in real space and
// periodised over IMG box images each way, then transformed by a naive O(N^2) DFT with the
// solver's own (unnormalized, e^-ikx) convention. Returned interleaved (re, im) over the two
// channel halves, exactly like the forcing buffer.
//
// `pol < 0` is the z- channel and anything else the z+ channel, and at most NBLOB per
// channel, in the order given -- that is the documented contract (solver2d.js's setBlobs
// comment), transcribed here rather than called.
const IMG = 3;
function refSpectrum(P, nkx, nky, nblob, list) {
  const nm = nkx * nky, out = new Float64Array(4 * nm), de = dealiasRef(P, nkx, nky);
  const dx = P.Lx / P.nx, dy = P.Ly / P.ny;
  for (let half = 0; half < 2; half++) {
    const mine = (list || []).filter(b => (b.pol < 0 ? 1 : 0) === half).slice(0, nblob);
    const f = new Float64Array(P.nx * P.ny);
    for (const b of mine) {
      const Pk = potentialPeak(+b.amp, +b.sigma), inv = 0.5 / (b.sigma * b.sigma);
      for (let i = 0; i < P.nx; i++) for (let j = 0; j < P.ny; j++) {
        let s = 0;
        for (let p = -IMG; p <= IMG; p++) for (let q = -IMG; q <= IMG; q++) {
          const ax = i * dx - b.x + p * P.Lx, ay = j * dy - b.y + q * P.Ly;
          s += Math.exp(-inv * (ax * ax + ay * ay));
        }
        f[i * P.ny + j] += Pk * s;
      }
    }
    for (let a = 0; a < nkx; a++) {
      const ix = (a < P.nx / 2) ? a : a - P.nx, kx = ix * 2 * Math.PI / P.Lx;
      for (let q = 0; q < nky; q++) {
        const ky = q * 2 * Math.PI / P.Ly, m = a * nky + q;
        let re = 0, im = 0;
        if (de[m]) for (let i = 0; i < P.nx; i++) {
          const px = kx * (i * dx);
          for (let j = 0; j < P.ny; j++) {
            const th = px + ky * (j * dy), v = f[i * P.ny + j];
            re += v * Math.cos(th); im -= v * Math.sin(th);
          }
        }
        out[2 * (half * nm + m)] = re; out[2 * (half * nm + m) + 1] = im;
      }
    }
  }
  return out;
}
// compare production against reference: the peak-relative worst absolute miss (the fp32
// floor), and the worst PER-MODE relative miss over the modes that carry real amplitude (a
// phase error at 1e-3 of the peak is still a phase error).
const REL_FLOOR = 1e-4;
function compare(prod, ref) {
  let peak = 0;
  for (let i = 0; i < ref.length; i += 2) peak = Math.max(peak, Math.hypot(ref[i], ref[i + 1]));
  let abs = 0, rel = 0, relAt = -1, n = 0, maxIm = 0;
  for (let m = 0; m < ref.length / 2; m++) {
    const a = Math.hypot(ref[2 * m], ref[2 * m + 1]);
    const d = Math.hypot(prod[2 * m] - ref[2 * m], prod[2 * m + 1] - ref[2 * m + 1]);
    abs = Math.max(abs, d);
    maxIm = Math.max(maxIm, Math.abs(ref[2 * m + 1]));
    if (a >= REL_FLOOR * peak) { n++; if (d / a > rel) { rel = d / a; relAt = m; } }
  }
  return { peak, abs: abs / Math.max(1e-300, peak), rel, relAt, n, maxIm: maxIm / Math.max(1e-300, peak) };
}

// ===========================================================================
// 1. the emission, the interpreter, and full coverage of the mode buffer
// ===========================================================================
async function legEmission(state) {
  const M = state.M;
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  const env = state.env = state.env || await boot(dir, "rmhd2d.html");
  const G = state.G = gridOf(env, SQ);
  state.R = gridOf(env, RECT);
  ok("buildShaders emits `blobBuild` on the 2D page", G.names.indexOf("blobBuild") >= 0,
     G.names.length + " kernels");
  ok("  ... and it parses", (() => { try { new M.WgslParser().parse(G.src); return true; }
                                     catch (e) { return false; } })());
  // the packed stride and the kernel's loop bound are ONE number in two files: physics.js's
  // BLOB_FORCE_MAX templates NBLOB, solver2d.js sizes `_blobs` at 2*BLOB_FORCE_MAX*4 floats
  // and the kernel reads slot `half*NBLOB + b`. If they ever drift, the z- half is read at
  // the wrong offset and nothing else in the tree notices.
  const NB = kconst(G.src, "NBLOB");
  const live = env.run("function(){ return solver && solver._blobs ? solver._blobs.length : -1; }");
  ok("NBLOB in the kernel == BLOB_FORCE_MAX == the live solver's packed stride",
     NB === G.nblob && live === 2 * G.nblob * 4 && NB === kconst(state.R.src, "NBLOB"),
     "NBLOB " + NB + ", BLOB_FORCE_MAX " + G.nblob + ", _blobs " + live + " floats");
  // the body is a fixed function of the grid: only physics.js's `C.pre` differs between
  // presets, so every emission's kernel text after the NBLOB line must be one string
  const bodyOf = s => s.slice(s.indexOf("const NBLOB"));
  const bodies = new Set([bodyOf(G.src), bodyOf(state.R.src)]);
  for (const pr of [64, 128, 256, 512, 1024])
    bodies.add(bodyOf(gridOf(env, Object.assign({}, SQ, { nx: pr, ny: pr })).src));
  ok("  ... and the kernel body is byte-identical at every grid the page can emit",
     bodies.size === 1, bodies.size + " distinct bodies over 7 grids");

  // ---- the interpreter, measured rather than assumed -----------------------
  // wgsl_reflect reads `@workgroup_size(64)` as its first character. Recorded here so that
  // the day the library is fixed the over-dispatch below is still correct (it is: more lanes
  // only means the kernel's bounds test does more of the work).
  const probe = `@group(0) @binding(0) var<storage, read_write> o: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>) { o[l.x] = 1.0; }`;
  const po = new Float32Array(64);
  new M.WgslExec(new M.WgslParser().parse(probe)).dispatchWorkgroups("main", [1, 1, 1], { 0: { 0: po } });
  const lanes = po.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
  ok("the WGSL interpreter runs contiguous lanes from 0 (over-dispatch is therefore sound)",
     lanes >= 1 && po.slice(0, lanes).every(v => v === 1) && po.slice(lanes).every(v => v === 0),
     "@workgroup_size(64) runs " + lanes + " lane(s) per workgroup here");
  // ... so every dispatch in this file asks for 2*nm workgroups. Prove the coverage.
  for (const [tag, g, P] of [["32x32", G, SQ], ["32x16", state.R, RECT]]) {
    const packed = pack(env, P, g, [{ x: 0.75, y: 1.5, sigma: 0.5, amp: 0.25, pol: 1 }]);
    const frc = runBlob(M, g, packed);
    let un = 0;
    for (let i = 0; i < frc.length; i++) if (frc[i] === SENT_FRC) un++;
    ok("  ... and the EMITTED kernel, run byte for byte, writes every one of the 2*NM modes ("
       + tag + ")", un === 0,
       un + " of " + frc.length + " f32 words (2 x " + (2 * g.nm)
       + " modes) left holding the sentinel");
  }
}

// ===========================================================================
// 2. setBlobs: the channel half, the per-channel cap, the order, the zero-fill
// ===========================================================================
// The MAGNITUDE of the packed weight is not asserted here -- it is leg 3's business, end to
// end against the DFT. This leg is about WHERE a blob lands and what happens to the slots it
// does not.
function legPack(state) {
  const env = state.env, G = state.G, NB = G.nblob;
  const slot = (a, h, b) => Array.from(a.slice(4 * (h * NB + b), 4 * (h * NB + b) + 4));
  const clean = a => Array.from(a).every(v => v !== SENT_PACK);

  // ---- the channel half ----------------------------------------------------
  const two = pack(env, SQ, G, [{ x: 0.75, y: 1.5, sigma: 0.5, amp: 0.25, pol: 1 },
                                { x: 2.5, y: 0.25, sigma: 0.375, amp: -0.5, pol: -1 }]);
  const p0 = slot(two, 0, 0), p1 = slot(two, 1, 0);
  ok("pol > 0 packs into the z+ half's slot 0, pol < 0 into the z- half's",
     p0[0] === 0.75 && p0[1] === 1.5 && p0[2] === 0.5 &&
     p1[0] === 2.5 && p1[1] === 0.25 && p1[2] === 0.375,
     "z+ " + p0.slice(0, 3) + " / z- " + p1.slice(0, 3));
  ok("  ... the weight carries the amplitude's SIGN (icBlobPeak and sigma^2*w0 are positive)",
     p0[3] > 0 && p1[3] < 0, "w = " + e2(p0[3]) + " / " + e2(p1[3]));
  let rest = 0;
  for (let h = 0; h < 2; h++) for (let b = 1; b < NB; b++)
    if (slot(two, h, b).some(v => v !== 0)) rest++;
  ok("  ... and every other slot in both halves is exactly zero, sentinel erased",
     rest === 0 && clean(two), rest + " of " + (2 * NB - 2) + " unused slots non-zero");
  // pol is a SIGN test, not a flag: 0 and a missing key are both z+
  const pols = pack(env, SQ, G, [{ x: 1, y: 0, sigma: 0.5, amp: 1, pol: 0 },
                                 { x: 2, y: 0, sigma: 0.5, amp: 1 },
                                 { x: 3, y: 0, sigma: 0.5, amp: -1, pol: -0.5 }]);
  ok("  ... pol == 0 and a missing pol are z+; any negative pol is z-",
     slot(pols, 0, 0)[0] === 1 && slot(pols, 0, 1)[0] === 2 && slot(pols, 1, 0)[0] === 3 &&
     slot(pols, 0, 2).every(v => v === 0),
     "z+ slots " + [0, 1, 2].map(b => slot(pols, 0, b)[0]).join(",")
     + " / z- slot0 " + slot(pols, 1, 0)[0]);

  // ---- the per-channel cap -------------------------------------------------
  const many = [];
  for (let i = 0; i < NB + 3; i++) many.push({ x: 0.25 * (i + 1), y: 0, sigma: 0.5, amp: 1, pol: 1 });
  const capped = pack(env, SQ, G, many);
  let order = true;
  for (let b = 0; b < NB; b++) if (slot(capped, 0, b)[0] !== 0.25 * (b + 1)) order = false;
  let zminus = 0;
  for (let b = 0; b < NB; b++) if (slot(capped, 1, b).some(v => v !== 0)) zminus++;
  ok("`n` is a PER-CHANNEL cap: " + (NB + 3) + " z+ blobs keep the first " + NB + " in order",
     order, "x0 = " + [0, 1, NB - 1].map(b => slot(capped, 0, b)[0]).join(", ... ,"));
  ok("  ... the overflow is DROPPED, never wrapped into the other channel's half",
     zminus === 0, zminus + " z- slots written by a z+ overflow");
  // ... and the cap is per channel, so a full z+ list does not stop a z- blob
  const mixed = pack(env, SQ, G, many.concat([{ x: 9.5, y: 0, sigma: 0.5, amp: 1, pol: -1 }]));
  ok("  ... and a full z+ channel does not consume the z- channel's slots",
     slot(mixed, 1, 0)[0] === 9.5, "z- slot 0 x0 = " + slot(mixed, 1, 0)[0]);

  // ---- the zero-fill -------------------------------------------------------
  const empty = pack(env, SQ, G, []);
  ok("an empty list zeroes every slot (`a.fill(0)` runs before anything is written)",
     Array.from(empty).every(v => v === 0), "over " + empty.length + " floats");
  const nul = pack(env, SQ, G, null);
  ok("  ... and so does a null list", Array.from(nul).every(v => v === 0));
  // a SHORTER list must not leave the previous call's blobs standing: setBlobs is handed the
  // whole picture every time, so slot 1 of a 3-then-1 sequence has to be gone
  const a3 = new Float32Array(2 * NB * 4).fill(SENT_PACK);
  env.run(PACK, SQ, a3, many.slice(0, 3));
  env.run(PACK, SQ, a3, many.slice(0, 1));
  let stale = 0;
  for (let b = 1; b < NB; b++) if (slot(a3, 0, b).some(v => v !== 0)) stale++;
  ok("  ... and a shorter list on the SAME buffer erases the slots it no longer fills",
     stale === 0 && slot(a3, 0, 0)[0] === 0.25, stale + " stale slots after 3 blobs then 1");
}

// ===========================================================================
// 3. the cross-check: the production k space against a real-space DFT
// ===========================================================================
// Each case runs the two paths that have nothing in common -- setBlobs + the emitted
// `blobBuild`, and potentialPeak + an analytic gaussian + a naive DFT -- and compares.
const CASES = [
  { tag: "one z+ blob, offset (0.75, 1.5)", grid: "SQ",
    list: [{ x: 0.75, y: 1.5, sigma: 0.375, amp: 0.25, pol: 1 }] },
  { tag: "a DIFFERENT blob in each channel", grid: "SQ",
    list: [{ x: 0.75, y: 1.5, sigma: 0.5, amp: 0.25, pol: 1 },
           { x: 2.5, y: 0.25, sigma: 0.375, amp: -0.5, pol: -1 }] },
  { tag: "two blobs in one channel (superposition)", grid: "SQ",
    list: [{ x: 0.75, y: 1.5, sigma: 0.5, amp: 0.25, pol: 1 },
           { x: 4.25, y: 5.0, sigma: 0.375, amp: 0.75, pol: 1 }] },
  { tag: "both polarities of amplitude, both channels", grid: "SQ",
    list: [{ x: 1.25, y: 0.75, sigma: 0.375, amp: -0.625, pol: 1 },
           { x: 5.5, y: 3.25, sigma: 0.5, amp: 0.375, pol: -1 }] },
  { tag: "a rectangular box (32x16, Ly = Lx/2)", grid: "RECT",
    list: [{ x: 0.75, y: 1.5, sigma: 0.375, amp: 0.5, pol: 1 },
           { x: 3.5, y: 2.25, sigma: 0.5, amp: -0.25, pol: -1 }] }
];
const ABS_TOL = 5e-7, REL_TOL = 2e-6;
function legCross(state) {
  const M = state.M, env = state.env;
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  for (const c of CASES) {
    const P = c.grid === "SQ" ? SQ : RECT, G = c.grid === "SQ" ? state.G : state.R;
    const prod = runBlob(M, G, pack(env, P, G, c.list));
    const ref = refSpectrum(P, G.nkx, G.nky, G.nblob, c.list);
    const r = compare(prod, ref);
    ok(c.tag, r.abs < ABS_TOL && r.rel < REL_TOL && r.peak > 0,
       "peak |f_k| " + e2(r.peak) + ", max|delta|/peak " + e2(r.abs)
       + ", worst per-mode rel " + e2(r.rel) + " over " + r.n + " modes >= "
       + e2(REL_FLOOR) + " of peak");
  }
  // ---- the phase is actually being exercised -------------------------------
  // A blob AT THE ORIGIN has a purely real transform, so a sign error in exp(-i k.x0) is
  // invisible on it. The offset cases above are only a phase test because their imaginary
  // part is O(1) of the peak -- state both halves rather than trusting it.
  const at0 = [{ x: 0, y: 0, sigma: 0.375, amp: 0.25, pol: 1 }];
  const off = CASES[0].list;
  const fa = runBlob(M, state.G, pack(env, SQ, state.G, at0));
  const fb = runBlob(M, state.G, pack(env, SQ, state.G, off));
  const rA = compare(fa, refSpectrum(SQ, state.G.nkx, state.G.nky, state.G.nblob, at0));
  const rB = compare(fb, refSpectrum(SQ, state.G.nkx, state.G.nky, state.G.nblob, off));
  // the PRODUCTION imaginary part, which is exactly zero at zero offset (sin(0) is exact)
  // -- the reference's own is only at the fp64 DFT's cancellation floor
  let imA = 0, imB = 0, pkA = 0;
  for (let m = 0; m < 2 * state.G.nm; m++) {
    imA = Math.max(imA, Math.abs(fa[2 * m + 1])); imB = Math.max(imB, Math.abs(fb[2 * m + 1]));
    pkA = Math.max(pkA, Math.hypot(fa[2 * m], fa[2 * m + 1]));
  }
  ok("a blob at the origin transforms PURELY REAL, and matches", imA === 0
     && rA.maxIm < 1e-12 && rA.abs < ABS_TOL && rA.rel < REL_TOL,
     "kernel max|Im| " + e2(imA) + ", reference max|Im|/peak " + e2(rA.maxIm)
     + ", max|delta|/peak " + e2(rA.abs));
  ok("  ... so the offset cases above are a phase test: their Im is O(1) of the peak",
     rB.maxIm > 0.5 && imB / pkA > 0.5,
     "max|Im|/peak " + e2(rB.maxIm) + " (reference), " + e2(imB / pkA) + " (kernel)");
  // ... and the two agree on |f_k| mode for mode, i.e. the offset moved the blob and did
  // nothing else -- a wrong exp(-0.5 s^2 k^2) would break this even at zero offset
  let dmag = 0, pk = 0;
  for (let m = 0; m < 2 * state.G.nm; m++) {
    const A = Math.hypot(fa[2 * m], fa[2 * m + 1]), B = Math.hypot(fb[2 * m], fb[2 * m + 1]);
    pk = Math.max(pk, A); dmag = Math.max(dmag, Math.abs(A - B));
  }
  ok("  ... and translating a blob changes only its phase: |f_k| is unmoved mode for mode",
     dmag / pk < 1e-6, "max ||f_k| difference| / peak = " + e2(dmag / pk));
}

// ===========================================================================
// 4. what a tolerance cannot see: the dealias cliff, the pure write, the empty table
// ===========================================================================
// Two statements the leg-3 comparison is structurally unable to make. (1) OUTSIDE the 2/3
// cut the kernel must write EXACTLY zero -- `forcingAdd`, unlike `nlAssemble`, does not
// dealias, so a gaussian's tail past the cut would go straight onto the fields and alias.
// A tolerance on the peak cannot see that tail at a resolved sigma. (2) `frc[idx] = acc` is
// a WRITE: the buffer holds whatever the previous mode left there (blob mode never clears it
// per step), so an accumulate would integrate the forcing instead of setting it. And (3) the
// ky = 0 row is exactly hermitian, which is the kernel's licence to write k space with none
// of the OU path's mirror machinery.
function legExact(state) {
  const M = state.M, env = state.env, G = state.G;
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  // the gate's own mask, against the app's
  const de = dealiasRef(SQ, G.nkx, G.nky);
  let mism = 0, nOut = 0;
  for (let m = 0; m < G.nm; m++) { if (de[m] !== G.gridB[4 * m]) mism++; if (!de[m]) nOut++; }
  ok("the 2/3 mask rebuilt here from the rule IS gridB's column 0", mism === 0,
     mism + " modes differ; " + nOut + " of " + G.nm + " modes lie outside the cut");

  // ---- the cliff -----------------------------------------------------------
  // A DELIBERATELY narrow blob (sigma = 0.25, ~1.3 cells at 32^2): its gaussian is still
  // O(0.1) of the peak at the cut, so "zero outside" is a real discontinuity and not a
  // statement about numbers that were negligible anyway.
  const narrow = [{ x: 0.75, y: 1.5, sigma: 0.25, amp: 0.5, pol: 1 },
                  { x: 3.0, y: 4.5, sigma: 0.25, amp: -0.5, pol: -1 }];
  const frc = runBlob(M, G, pack(env, SQ, G, narrow));
  let nz = 0, inMin = Infinity, peak = 0;
  for (let h = 0; h < 2; h++) for (let m = 0; m < G.nm; m++) {
    const i = h * G.nm + m, a = Math.hypot(frc[2 * i], frc[2 * i + 1]);
    peak = Math.max(peak, a);
    if (!de[m]) { if (frc[2 * i] !== 0 || frc[2 * i + 1] !== 0) nz++; }
  }
  // the smallest RETAINED amplitude on the cut's inner rim, as a fraction of the peak: the
  // height of the step the mask cuts
  for (let h = 0; h < 2; h++) for (let m = 0; m < G.nm; m++) {
    if (!de[m]) continue;
    const i = h * G.nm + m, j = m % G.nky, ii = (m - j) / G.nky;
    const nb = [[ii + 1, j], [ii - 1, j], [ii, j + 1], [ii, j - 1]];
    const rim = nb.some(([u, v]) => u >= 0 && u < G.nkx && v >= 0 && v < G.nky && !de[u * G.nky + v]);
    if (rim) inMin = Math.min(inMin, Math.hypot(frc[2 * i], frc[2 * i + 1]));
  }
  ok("every mode outside the 2/3 cut is written EXACTLY zero (the kernel dealiases)",
     nz === 0, nz + " of " + (2 * nOut) + " out-of-band modes non-zero");
  ok("  ... and that is a cliff, not a rounding: the rim just inside the cut is O(1)",
     inMin / peak > 1e-3, "smallest retained rim mode = " + e2(inMin / peak) + " of the peak");

  // ---- the reality constraint ---------------------------------------------
  // The kernel writes k space DIRECTLY, with none of the OU path's hermitian mirror
  // machinery (`_symmetrize_real_line`). Its licence for that is that exp(-i k.x0) is BY
  // CONSTRUCTION the transform of a real field -- so on the ky = 0 row, where rfft2 stores
  // both kx and -kx and reality is a constraint BETWEEN them, F(-kx, 0) must be the exact
  // conjugate of F(kx, 0). It is exact here and not merely close: theta(-kx, 0) = -theta,
  // cos is even and sin is odd. (The ky = Nyquist row carries the same constraint, but the
  // analytic form satisfies it only up to the sampled gaussian's alias sum -- which is moot,
  // because the 2/3 mask zeroes that whole row and the kx-Nyquist column with it. Leg 3's
  // agreement with a DFT of a genuinely real field bounds the residue there at 1e-7 of the
  // peak anyway.)
  const wide = runBlob(M, G, pack(env, SQ, G, CASES[1].list));
  let pairs = 0, broke = 0, nyq = 0;
  for (let h = 0; h < 2; h++) for (let i = 0; i < G.nkx; i++) {
    const m = i * G.nky, mm = ((G.nkx - i) % G.nkx) * G.nky;
    if (!de[m] || !de[mm]) continue;
    const a = h * G.nm + m, b = h * G.nm + mm;
    pairs++;
    if (wide[2 * a] !== wide[2 * b] || wide[2 * a + 1] !== -wide[2 * b + 1]) broke++;
  }
  for (let i = 0; i < G.nkx; i++) {
    if (de[i * G.nky + G.nky - 1]) nyq++;
    if (i === G.nx / 2 && de[i * G.nky]) nyq++;
  }
  ok("the ky = 0 row is EXACTLY hermitian, which is why the kernel needs no symmetrization",
     broke === 0 && pairs > 0,
     broke + " of " + pairs + " (kx, -kx) pairs broken, bit for bit");
  ok("  ... and the rows where the analytic form would NOT be exact are dealiased away",
     nyq === 0, nyq + " retained modes on the ky / kx Nyquist lines");

  // ---- the write -----------------------------------------------------------
  // Seed the buffer with a walk that is nowhere zero, run, and demand the answer is BITWISE
  // the answer from a clean buffer.
  const seed = new Float32Array(4 * G.nm);
  let s = 20261;
  for (let i = 0; i < seed.length; i++) { s = (s * 48271) % 2147483647; seed[i] = s / 2147483647 - 0.5; }
  const packed = pack(env, SQ, G, CASES[1].list);
  const clean = runBlob(M, G, packed, new Float32Array(4 * G.nm));
  const dirty = runBlob(M, G, packed, seed);
  const b1 = Array.from(bits(clean)), b2 = Array.from(bits(dirty));
  let first = -1;
  for (let i = 0; i < b1.length; i++) if (b1[i] !== b2[i]) { first = i; break; }
  ok("`frc[idx] = acc` is a WRITE: the result is bitwise independent of the buffer's contents",
     first < 0, first < 0 ? b1.length + " f32 words compared as raw bits"
                          : "word " + first + " differs");
  // ... and the zero table is the OFF state: no blobs means no forcing, from any buffer
  const offr = runBlob(M, G, pack(env, SQ, G, []), seed);
  ok("  ... so an empty blob table zeroes the forcing buffer outright",
     Array.from(offr).every(v => v === 0), "over " + offr.length + " floats");
}

// ---------------------------------------------------------------------------
const LEGS = [
  ["1. the emission, the interpreter, and full coverage of the mode buffer", legEmission],
  ["2. setBlobs: the channel half, the cap, the order, the zero-fill", legPack],
  ["3. the cross-check: production k space vs an independent real-space DFT", legCross],
  ["4. the dealias cliff, hermitian symmetry, the pure write, the empty table", legExact]
];
(async () => {
  const state = {};
  state.M = await wgslMod();
  for (const [title, fn] of LEGS) { console.log(title); await fn(state); }
  console.log(bad ? "\n" + bad + " FAILURE(S)" : "\nall checks passed");
  process.exit(bad ? 1 : 0);
})();
