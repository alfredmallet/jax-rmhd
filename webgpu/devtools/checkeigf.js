// EIGF_PLAN gate: the eigenfunction chart card -- |psihat(x, k_y)| and |phihat(x, k_y)|
// against x at one k_y, the tearing mode's structure.
//
//   node checkeigf.js [webgpu-dir]        exit code 1 on any failure
//
// The plan's five checks, in its own order, plus the card itself:
//   1  discipline: every emitted kernel parses (+ the reserved-word scan), names.mjs and
//      dup.py clean, the PHYSICS WGSL byte-identical to the base commit, and the RNG
//      reference unmoved. This is a DISPLAY-ONLY plan: nothing it adds may touch the
//      stepping path, and the 3D page must not move at all.
//   2  gather correctness: the emitted `eigfGather`, EXECUTED on a synthetic state
//      (wgsl_reflect's WGSL interpreter), against the CPU-side strided column -- a bit
//      compare, the path being a pure copy.
//   3  the fp64 mirror: gather + inverse-along-kx + modulus (`eigfProfile`, the app's own)
//      against a field whose k_y coefficient is known ANALYTICALLY, with the state built
//      by a direct fp64 forward DFT. That is what says the transform, its direction and
//      its normalization are right -- not that the code agrees with itself.
//   4  equilibrium exclusion: on a pure (y-independent) equilibrium every k_y > 0 column
//      is zero to round-off, which is what makes "minus the equilibrium" free and exact.
//   5  state invariance: a draw leaves (phik, psik) bitwise unchanged -- asserted three
//      ways, because no sandbox makes them all at once (the executed kernel's input array
//      word for word, the emitted WGSL's access qualifiers, and the booted page's encode
//      path traced buffer by buffer) -- and, riding the same bind-group trace, the LAYOUT
//      contract that trace assumes: the declared uniform struct's SizeOf against the size
//      the page allocates for it. That last one is a device-only failure class (nothing in
//      node allocates) and it is here because review 2026-08-14 found one.
//   6  the card: its CHART_TYPES entry and place in the list, its options and defaults,
//      its hint, the readback pool it joins, and the presets it opens on.
//
// The Delta' readout an earlier draft carried was dropped in review (the card is the plot
// alone), so there is no rate or fit anywhere here: everything below is a transform of
// state and is checked as one.
//
// CI reports, never gates.
"use strict";
const fs = require("fs"), os = require("os"), path = require("path");
const { spawnSync } = require("child_process");
const { pathToFileURL } = require("url");
const dir = path.resolve(process.argv[2] || path.join(__dirname, ".."));
const root = path.resolve(dir, "..");
// EIGF_PLAN's base commit (the plan file's own header says 70ec5a8, which is this tree's
// parent; 2cbfadd adds the two plan files and no webgpu/ byte). EIGF_BASE overrides.
const BASE = process.env.EIGF_BASE || "2cbfadd";
// the 64 Gauss(7) draws the forcing path starts from, hashed, RECORDED FROM THE BASE TREE
// before a line of this plan was written. No kernel here goes near the RNG; this is the
// standing rule (record the reference first) rather than a suspicion.
const RNG_SHA = "340f43548a5ed68fcefda6f8e08080075bbd011c247813d93803c8f63519d137";
// what this plan is allowed to ADD to each page's emission, and it may move nothing:
// one gather kernel, on the 2D page only (the card is 2D-only, `avail: cfg => !cfg.zslice`,
// because the equilibria are).
const ADDED = { "rmhd2d.html": ["eigfGather"], "rmhd3d.html": [] };

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "checkeigf-"));
const sh = (cmd, args, opts) => spawnSync(cmd, args, Object.assign(
  { encoding: "utf8", cwd: __dirname, maxBuffer: 1 << 28 }, opts || {}));
const node = (args, opts) => sh(process.execPath, args, opts);
const lastLine = r => ((r.stdout || "") + (r.stderr || "")).trim().split("\n").pop();
const bits = a => new Uint32Array(a.buffer, a.byteOffset, a.length);
const maxAbs = a => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m; };

// every kernel of one page, at every resolution preset + the self-test grid, as a
// { label :: kernel -> source } map, emitted by the real dumpwgsl2 (the checkiso helper)
function dumpKernels(d, page, tag) {
  const out = path.join(tmp, tag + "-" + page.replace(".html", "") + ".wgsl.txt");
  const r = node([path.join(__dirname, "dumpwgsl2.js"), d, page, "", out]);
  if (r.status !== 0) { ok(tag + " " + page + ": dumpwgsl2 ran", false, lastLine(r)); return { file: out, k: {} }; }
  const txt = fs.readFileSync(out, "utf8");
  const parts = txt.split(/^########## (.*) ##########$/m);
  const k = {};
  for (let i = 1; i < parts.length; i += 2) k[parts[i]] = parts[i + 1];
  return { file: out, k };
}
// the base commit's four files, checked out into tmp/base (what stubenv reads)
function baseDir() {
  const d = path.join(tmp, "base");
  fs.mkdirSync(d, { recursive: true });
  for (const f of ["common.js", "physics.js", "rmhd2d.html", "rmhd3d.html"]) {
    const r = sh("git", ["-C", root, "show", BASE + ":webgpu/" + f], { maxBuffer: 1 << 28 });
    if (r.status !== 0) return null;
    fs.writeFileSync(path.join(d, f), r.stdout);
  }
  return d;
}
// wgsl_reflect, from wherever it is installed (the wgslparse / checkiso idiom)
const wgslPath = () => {
  const cands = [process.env.WGSL_REFLECT,
                 path.join(__dirname, "node_modules", "wgsl_reflect", "wgsl_reflect.module.js"),
                 "/tmp/chk/node_modules/wgsl_reflect/wgsl_reflect.module.js"];
  return cands.filter(p => p && fs.existsSync(p))[0] || null;
};
async function wgslMod() {
  const p = wgslPath();
  if (!p) return null;
  try { return await import(pathToFileURL(p).href); } catch (e) { return null; }
}
// boot() runs at require, its async tail a macrotask later (the checkonepage idiom)
const boot = (d, page, demo) => new Promise(res => {
  const env = require("./stubenv")(d, page, demo);
  setTimeout(() => res(env));
});

// ---------------------------------------------------------------------------
// the grid this file measures on, and its analytic fields
// ---------------------------------------------------------------------------
// A WIDE box (Lx = 2*Ly), which is the shape every tearing preset runs in, and a grid
// small enough for an O(nx*ny*nx*nky) fp64 DFT: the reference is a direct sum, never an
// FFT, so a mistake in the app's transform cannot be reproduced by the reference.
const G = { nx: 64, ny: 32, Lx: 4 * Math.PI, Ly: 2 * Math.PI };
G.nky = G.ny / 2 + 1;
G.nm = G.nx * G.nky;
const sech2 = t => Math.pow(2 / (Math.exp(t) + Math.exp(-t)), 2);
// the tearing equilibrium's own profile, and the ODD companion a phihat lobe pair looks
// like: g is even about x0 (psihat's shape), h is odd (phihat's, zero on the surface)
const EQ_A = 0.15;
const gEven = x => sech2((x - 0.5 * G.Lx) / (EQ_A * G.Lx));
const hOdd = x => Math.tanh((x - 0.5 * G.Lx) / (EQ_A * G.Lx)) * gEven(x);
// ... and an OFF-CENTRE pair, which is what actually pins the DIRECTION of the inverse
// along kx. Both profiles above are symmetric about x = Lx/2 and the plotted quantity is a
// MODULUS, so |psihat(Lx - x)| = |psihat(x)| identically: a reversed inverse (fftPow2 at
// sign -1, i.e. ix -> nx - ix) reproduces them to the same 1e-9, and the mirror would be
// measuring the normalization alone (adversarial review, 2026-08-14). These two sit at
// 0.31 Lx and 0.68 Lx instead, so a reversal misses by O(1) of the profile. The analytic
// answer is g/2 and h/2 whatever g and h are -- that is a property of the transform, not
// of the shape, which is exactly why the shape can be moved.
const GX0 = 0.31, HX0 = 0.68;
const gOff = x => sech2((x - GX0 * G.Lx) / (EQ_A * G.Lx));
const hOff = x => Math.tanh((x - HX0 * G.Lx) / (1.7 * EQ_A * G.Lx)) *
                  sech2((x - HX0 * G.Lx) / (EQ_A * G.Lx));
// forward rfft2 of a real field, by DEFINITION and in fp64:
//   F[ix, j] = sum_p sum_q f[p, q] exp(-2 pi i (ix p / nx + j q / ny))
// which is the layout SPEC.md 1 fixes and the layout the app's own rowsR2C + colsFwd
// produce. Returned interleaved (re, im) at m = ix*nky + j, i.e. the state's own order.
function rfft2(f) {
  const { nx, ny, nky } = G, out = new Float64Array(2 * nx * nky);
  for (let ix = 0; ix < nx; ix++) {
    for (let j = 0; j < nky; j++) {
      let re = 0, im = 0;
      for (let p = 0; p < nx; p++) {
        for (let q = 0; q < ny; q++) {
          const th = -2 * Math.PI * ((ix * p) / nx + (j * q) / ny);
          const v = f[p * ny + q];
          re += v * Math.cos(th); im += v * Math.sin(th);
        }
      }
      const m = ix * nky + j;
      out[2 * m] = re; out[2 * m + 1] = im;
    }
  }
  return out;
}
// a (phi, psi) state array in the app's own layout, from two real-space fields
function stateOf(phi, psi) {
  const Fp = rfft2(phi), Fs = rfft2(psi), out = new Float32Array(4 * G.nm);
  out.set(Fp, 0); out.set(Fs, 2 * G.nm);
  return out;
}
// f(x, y) on the grid, row-major p*ny + q as the real-space kernels index it
function sample(f) {
  const { nx, ny, Lx, Ly } = G, out = new Float64Array(nx * ny);
  for (let p = 0; p < nx; p++) for (let q = 0; q < ny; q++) {
    out[p * ny + q] = f(p * Lx / nx, q * Ly / ny);
  }
  return out;
}
// a deterministic state (no RNG anywhere near this file), for the pure-copy legs
function synthState(nm) {
  const f = new Float32Array(4 * nm);
  let s = 12345;
  for (let i = 0; i < f.length; i++) { s = (s * 48271) % 2147483647; f[i] = s / 2147483647 - 0.5; }
  return f;
}
// the emitted gather, run over the whole column. The interpreter runs one invocation per
// (workgroup index x workgroup size), so the dispatch is the item count and the kernel's
// own `ix >= NX` guard retires the rest -- the check2dspec convention.
function runGather(M, src, fields, j0, nx) {
  const col = new Float32Array(4 * nx);
  new M.WgslExec(new M.WgslParser().parse(src)).dispatchWorkgroups("main", [nx, 1, 1],
    { 0: { 0: fields, 1: new Uint32Array([j0, 0, 0, 0]), 2: col } });
  return col;
}

(async () => {
const env2d = await boot(dir, "rmhd2d.html", null);
const M = await wgslMod();
// the gather kernel at THIS file's grid, emitted by the page's own buildShaders (the
// checkk idiom: the kernel under test is the shipped template, not a copy of it)
const gatherSrc = env2d.run(`function(P){
  const g = Object.assign({ nx: P.nx, ny: P.ny, Lx: P.Lx, Ly: P.Ly, pm: 1, eqsrc: false },
                          makeGrid(P));
  return buildShaders(g).eigfGather;
}`, { nx: G.nx, ny: G.ny, Lx: G.Lx, Ly: G.Ly, diss: 1e-13, hyper: 4, fshell: [1, 3] });

// ---------------------------------------------------------------------------
console.log("1. discipline: parse / names / dup, physics WGSL byte-identical, RNG unmoved");
// ---------------------------------------------------------------------------
const cur = {};
{
  const wp = wgslPath();
  const penv = Object.assign({}, process.env);
  if (wp) penv.WGSL_REFLECT = wp;
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const d = dumpKernels(dir, page, "cur");
    cur[page] = d.k;
    const r = node([path.join(__dirname, "wgslparse.mjs"), d.file], { env: penv });
    ok(page + ": every emitted kernel parses (+ reserved-word scan)", r.status === 0, lastLine(r));
  }
  const nm = node([path.join(__dirname, "names.mjs"), dir]);
  ok("names.mjs: no redeclares, every free identifier resolves", nm.status === 0, lastLine(nm));
  const files = [path.join(dir, "common.js"), path.join(dir, "physics.js")];
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const t = fs.readFileSync(path.join(dir, page), "utf8");
    const f = path.join(tmp, page.replace(".html", ".js"));
    fs.writeFileSync(f, t.slice(t.indexOf("<script>\n") + 8, t.lastIndexOf("</script>")));
    files.push(f);
  }
  // dup.py's group COUNT is not a stable number (checkiso's note); what is stable is where
  // the clones are -- a clone inside one file, or one reaching into the shared core, is
  // code that wants sharing.
  const dp = sh("python3", [path.join(__dirname, "dup.py")].concat(files));
  const groups = (dp.stdout || "").split("\n").filter(l => l.indexOf("CLONE") === 0)
    .map(l => l.split("):")[1].split(";").map(s => s.trim().split(":")[0]));
  const bads = groups.filter(g => new Set(g).size !== 2 ||
                                  g.some(f => f === "common.js" || f === "physics.js"));
  ok("dup.py: no clone inside a file, none reaching into the shared core",
     dp.status === 0 && bads.length === 0,
     groups.length + " groups" + (bads.length ? "; BAD: " + bads[0].join(" ") : ""));

  const bd = baseDir();
  if (!bd) ok("base " + BASE + " is readable (git show)", false, "git show failed");
  else for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const base = dumpKernels(bd, page, "base").k;
    const keys = new Set(Object.keys(base).concat(Object.keys(cur[page])));
    const moved = [], gone = [], added = new Set();
    for (const k of keys) {
      if (base[k] === cur[page][k]) continue;
      if (base[k] === undefined) { added.add(k.split(" :: ")[1]); continue; }
      if (cur[page][k] === undefined) { gone.push(k); continue; }
      moved.push(k);
    }
    ok(page + ": every kernel that existed at " + BASE + " is byte-identical",
       moved.length === 0 && gone.length === 0,
       moved.length + " moved, " + gone.length + " vanished" +
       (moved.length ? " (" + moved[0] + ")" : ""));
    // ... and the additions are EXACTLY the expected list: a stale expectation must fail
    // too, or the next kernel slips in behind this one
    const want = ADDED[page].slice().sort().join(",");
    ok("  ... and it adds exactly [" + (want || "nothing") + "]",
       Array.from(added).sort().join(",") === want, Array.from(added).sort().join(",") || "nothing");
  }
  const rng = env2d.run(`function(){
    const g = new Gauss(7), a = [];
    for (let i = 0; i < 64; i++) a.push(g.next());
    return a;
  }`);
  const h = require("crypto").createHash("sha256")
    .update(Buffer.from(Float64Array.from(rng).buffer)).digest("hex");
  ok("the RNG reference is unmoved (64 Gauss(7) draws, hashed)", h === RNG_SHA,
     h.slice(0, 16) + " vs " + RNG_SHA.slice(0, 16));
  ok("  ... which it must be: no kernel or code path here is RNG-adjacent at all",
     !/Gauss|mulberry|noise/.test(gatherSrc) &&
     !/Gauss|mulberry/.test(String(env2d.run("function(){ return String(drawEigf) + String(eigfProfile); }"))));
}

// ---------------------------------------------------------------------------
console.log("2. gather correctness: the executed kernel against the strided column");
// ---------------------------------------------------------------------------
if (!M) {
  ok("wgsl_reflect is installed (npm i wgsl_reflect in devtools/, or WGSL_REFLECT=<path>)", false);
} else {
  const fields = synthState(G.nm);
  let worstBits = 0, worstJ = -1;
  for (const j0 of [0, 1, 2, 5, G.nky - 1]) {
    const col = runGather(M, gatherSrc, fields, j0, G.nx);
    let mism = 0;
    for (let ix = 0; ix < G.nx; ix++) {
      const m = ix * G.nky + j0;
      if (col[2 * ix] !== fields[2 * m] || col[2 * ix + 1] !== fields[2 * m + 1]) mism++;
      if (col[2 * G.nx + 2 * ix] !== fields[2 * (G.nm + m)] ||
          col[2 * G.nx + 2 * ix + 1] !== fields[2 * (G.nm + m) + 1]) mism++;
    }
    if (mism > worstBits) { worstBits = mism; worstJ = j0; }
  }
  ok("the gathered column is the strided column m = ix*NKY + j0, BIT for bit, at 5 bins",
     worstBits === 0, worstBits ? worstBits + " mismatches at j0 = " + worstJ
                                : "phi and psi, j0 = 0, 1, 2, 5, " + (G.nky - 1));
  // an out-of-range bin must clamp rather than read past the array (the kernel's `min`)
  const clamped = runGather(M, gatherSrc, fields, G.nky + 40, G.nx);
  const last = runGather(M, gatherSrc, fields, G.nky - 1, G.nx);
  ok("  ... and a j0 past the last bin clamps to it instead of reading out of bounds",
     bits(clamped).every((v, i) => v === bits(last)[i]));
  // the whole output really is written -- a leg that compares zeros with zeros would pass
  ok("  ... with every one of the 2*NX slots written (no untouched tail)",
     maxAbs(runGather(M, gatherSrc, fields, 3, G.nx)) > 0 &&
     Array.from(runGather(M, gatherSrc, fields, 3, G.nx)).every(v => v !== 0));
}

// ---------------------------------------------------------------------------
console.log("3. the fp64 mirror: gather + inverse along kx + modulus, against analysis");
// ---------------------------------------------------------------------------
// psi = g(x) cos(k_y y + p0) and phi = h(x) sin(k_y y + p0) -- the two 90 degrees apart IN
// Y, which is the plan's whole reason for plotting moduli rather than a real-space cut at
// one y. Then psi = sum_j c_j(x) e^{i k_y y} has c_j0 = (g/2) e^{i p0}, so
//   |psihat(x)| = g(x)/2   and   |phihat(x)| = h(x)/2
// EXACTLY, at every x, for every phase p0 -- which is what the card claims to draw.
//
// The pair is the OFF-CENTRE one, so this leg pins the transform's DIRECTION as well as
// its normalization: a modulus of a profile symmetric about Lx/2 is invariant under the
// reversal a wrong-signed inverse produces (see gOff / hOff above).
const J0 = 2, P0 = 0.7;
// the state, and the card's own path over it, for an arbitrary (g, h)
function mirrorOf(g, h) {
  const ky = J0 * 2 * Math.PI / G.Ly;
  const fields = stateOf(sample((x, y) => h(x) * Math.sin(ky * y + P0)),
                         sample((x, y) => g(x) * Math.cos(ky * y + P0)));
  const prof = j0 => {
    const col = M ? runGather(M, gatherSrc, fields, j0, G.nx) : cpuColumn(fields, j0);
    return env2d.run("function(v, nx, ny){ return eigfProfile(v, nx, ny); }",
                     Array.from(col), G.nx, G.ny);
  };
  return { fields, prof, P: prof(J0), scale: 0.5 * maxAbs(sample(g)) };
}
{
  const { fields, prof, P, scale } = mirrorOf(gOff, hOff);
  let ep = 0, es = 0;
  for (let p = 0; p < G.nx; p++) {
    const x = p * G.Lx / G.nx;
    es = Math.max(es, Math.abs(P.psi[p] - 0.5 * Math.abs(gOff(x))));
    ep = Math.max(ep, Math.abs(P.phi[p] - 0.5 * Math.abs(hOff(x))));
  }
  ok("|psihat(x)| is the analytic g(x)/2 at every x (g OFF-CENTRE)", es / scale < 1e-5,
     "max |delta| / max = " + (es / scale).toExponential(2) + " (tol 1e-5)");
  ok("|phihat(x)| is the analytic h(x)/2 at every x (h OFF-CENTRE)", ep / scale < 1e-5,
     "max |delta| / max = " + (ep / scale).toExponential(2) + " (tol 1e-5)");
  // ... and the direction, said in one number rather than left implicit in the tolerance:
  // the peak is where g put it and NOT at its mirror image, which is where a reversed
  // inverse would have moved it
  let pk = 0;
  for (let p = 0; p < G.nx; p++) if (P.psi[p] > P.psi[pk]) pk = p;
  const want = Math.round(GX0 * G.nx), mir = (G.nx - want) % G.nx;
  ok("  ... so the inverse runs the RIGHT WAY: the peak is at g's own x, not its mirror",
     Math.abs(pk - want) <= 1 && Math.abs(pk - mir) > 2,
     "peak at ix " + pk + ", g at " + want + ", the reversal would put it at " + mir);
  // every OTHER k_y column of this two-mode field is empty -- the card is a single
  // coefficient, not a sum over anything
  let other = 0;
  for (const j of [1, 3, 4]) other = Math.max(other, maxAbs(prof(j).psi), maxAbs(prof(j).phi));
  ok("  ... and every other k_y column of that field is empty", other / scale < 1e-5,
     "max = " + (other / scale).toExponential(2));
  // eigfProfile ALONE against a direct fp64 inverse DFT of the same column, so the CPU
  // half is gated even where the interpreter is not installed. Same arithmetic as the
  // card's, written from the definition rather than through fftPow2.
  const col = cpuColumn(fields, J0);
  const ref = { phi: new Float64Array(G.nx), psi: new Float64Array(G.nx) };
  for (let f = 0; f < 2; f++) {
    const dst = f === 0 ? ref.phi : ref.psi, o = 2 * f * G.nx;
    for (let p = 0; p < G.nx; p++) {
      let re = 0, im = 0;
      for (let ix = 0; ix < G.nx; ix++) {
        const th = 2 * Math.PI * ix * p / G.nx;
        re += col[o + 2 * ix] * Math.cos(th) - col[o + 2 * ix + 1] * Math.sin(th);
        im += col[o + 2 * ix] * Math.sin(th) + col[o + 2 * ix + 1] * Math.cos(th);
      }
      dst[p] = Math.hypot(re, im) / (G.nx * G.ny);
    }
  }
  let ed = 0;
  for (let p = 0; p < G.nx; p++) ed = Math.max(ed, Math.abs(P.psi[p] - ref.psi[p]),
                                                  Math.abs(P.phi[p] - ref.phi[p]));
  ok("eigfProfile is the direct fp64 inverse DFT along kx, 1/(nx*ny) and all",
     ed / scale < 1e-6, "max |delta| / max = " + (ed / scale).toExponential(2));
}
// The two SHAPE claims the card exists for, on the CENTRED pair -- the tearing profile
// itself -- read off the same mirror rather than asserted in prose: psihat peaks on the
// resonant surface, phihat vanishes there and has a lobe either side. (This pair says
// nothing about direction, by construction; the block above is what does.)
{
  const P = mirrorOf(gEven, hOdd).P;
  const mid = G.nx / 2;
  let pk = 0;
  for (let p = 0; p < G.nx; p++) if (P.psi[p] > P.psi[pk]) pk = p;
  const loLobe = Math.max(...Array.from(P.phi.slice(0, mid)));
  const hiLobe = Math.max(...Array.from(P.phi.slice(mid)));
  ok("on the centred tearing pair, psihat peaks ON x = Lx/2 and phihat is zero there, "
     + "with a lobe either side",
     pk === mid && P.phi[mid] < 1e-6 * hiLobe && loLobe > 0.1 * P.psi[pk] && hiLobe > 0.1 * P.psi[pk],
     "peak at ix " + pk + " of " + G.nx + ", |phihat(x0)| = " + P.phi[mid].toExponential(2) +
     " (ABSOLUTE) vs lobes " + loLobe.toExponential(2) + " / " + hiLobe.toExponential(2));
}

// ---------------------------------------------------------------------------
console.log("4. equilibrium exclusion: a pure equilibrium has NOTHING at k_y > 0");
// ---------------------------------------------------------------------------
// The plan's "minus equilibrium" is not a subtraction the code performs: every
// equilibrium seed here is y-INDEPENDENT, so it lives entirely in the k_y = 0 column --
// which is exactly what srcInit relies on when it reads psi_eq,k out of that column -- and
// selecting any other column has removed it, exactly and for free.
{
  const psi = sample(x => 1.65 * gEven(x));                 // psi_eq, phi_eq = 0
  const fields = stateOf(sample(() => 0), psi);
  const col0 = M ? runGather(M, gatherSrc, fields, 0, G.nx) : cpuColumn(fields, 0);
  let worst = 0;
  for (const j of [1, 2, 3, 6, G.nky - 1]) {
    const c = M ? runGather(M, gatherSrc, fields, j, G.nx) : cpuColumn(fields, j);
    worst = Math.max(worst, maxAbs(c));
  }
  ok("the equilibrium is ALL in the k_y = 0 column", maxAbs(col0) > 0,
     "|column 0| = " + maxAbs(col0).toExponential(3));
  ok("  ... and every k_y > 0 column of it is zero to round-off",
     worst / maxAbs(col0) < 1e-6,
     "max |column j>0| / |column 0| = " + (worst / maxAbs(col0)).toExponential(2));
  // ... which the card then plots as a flat zero rather than as a subtracted equilibrium
  const P = env2d.run("function(v, nx, ny){ return eigfProfile(v, nx, ny); }",
                      Array.from(M ? runGather(M, gatherSrc, fields, 1, G.nx) : cpuColumn(fields, 1)),
                      G.nx, G.ny);
  ok("  ... so the drawn profile at k_y = 1 is zero, with no equilibrium left in it",
     maxAbs(P.psi) / (maxAbs(col0) / (G.nx * G.ny)) < 1e-6,
     "max |psihat| = " + maxAbs(P.psi).toExponential(2));
}

// ---------------------------------------------------------------------------
console.log("5. state invariance: a draw leaves (phik, psik) bitwise unchanged");
// ---------------------------------------------------------------------------
{
  if (M) {
    const fields = synthState(G.nm);
    const before = Uint32Array.from(bits(fields));
    for (const j0 of [0, 1, 4, G.nky - 1, G.nky + 9]) runGather(M, gatherSrc, fields, j0, G.nx);
    ok("the executed kernel leaves its input state array bit-identical, word for word",
       bits(fields).every((v, i) => v === before[i]),
       "4 * " + G.nm + " f32 over 5 dispatches");
  }
  ok("the emitted kernel binds `fields` READ-ONLY, and only the column buffer read_write",
     /binding\(0\) var<storage, read> fields/.test(gatherSrc) &&
     /binding\(2\) var<storage, read_write> col/.test(gatherSrc) &&
     (gatherSrc.match(/read_write/g) || []).length === 1,
     (gatherSrc.match(/read_write/g) || []).length + " read_write binding(s)");
  // the ENCODE path on the booted page: what a readback writes, dispatches and copies,
  // named off the solver's own buffer map. A group with `fields` in a writable slot is
  // legal WGSL and legal WebGPU; only a trace says it is not there.
  const trace = await env2d.run(`async function(){
    const sv = solver, d = sv.device, log = { writes: [], copies: [], pipes: [] };
    const nameOf = b => { for (const k in sv.buf) if (sv.buf[k] === b) return k; return "?"; };
    const pipeOf = p => { for (const k in sv.pl) if (sv.pl[k] === p) return k; return "?"; };
    const wb = d.queue.writeBuffer, ce = d.createCommandEncoder;
    d.queue.writeBuffer = function (b, o, v) { log.writes.push(nameOf(b)); return wb.call(this, b, o, v); };
    d.createCommandEncoder = function () {
      const e = ce.call(this), bc = e.beginComputePass, cb = e.copyBufferToBuffer;
      e.beginComputePass = function () {
        const p = bc.call(this), sp = p.setPipeline;
        p.setPipeline = function (q) { log.pipes.push(pipeOf(q)); return sp.call(this, q); };
        return p;
      };
      e.copyBufferToBuffer = function (a, ao, b, bo, n) { log.copies.push(nameOf(a)); return cb.call(this, a, ao, b, bo, n); };
      return e;
    };
    try { await sv.readEigf(2); } finally { d.queue.writeBuffer = wb; d.createCommandEncoder = ce; }
    return log;
  }`);
  ok("readEigf writes ONE buffer (its k_y uniform) and no other",
     trace.writes.join(",") === "eigfU", trace.writes.join(",") || "none");
  ok("  ... runs ONE pipeline, the gather", trace.pipes.join(",") === "eigfGather",
     trace.pipes.join(",") || "none");
  ok("  ... and copies out of the column buffer alone -- `fields` is never a source either",
     trace.copies.join(",") === "eigfK", trace.copies.join(",") || "none");
  // the bind group's RESOURCE LIST, by name and in binding order, recorded off a second
  // solver built with createBindGroup patched (the stub discards the entries otherwise)
  const bgres = env2d.run(`function(){
    const d = solver.device, orig = d.createBindGroup, rec = [];
    d.createBindGroup = function (o) { rec.push(o.entries.map(e => e.resource.buffer)); return orig.call(this, o); };
    let s2;
    try { s2 = new Solver(d, solver.p); } finally { d.createBindGroup = orig; }
    const nameOf = b => { for (const k in s2.buf) if (s2.buf[k] === b) return k; return "?"; };
    const named = rec.map(g => g.map(nameOf).join(","));
    s2.destroy();
    return named;
  }`);
  ok("  ... and one bind group is exactly (fields, eigfU, eigfK), in that order",
     bgres.indexOf("fields,eigfU,eigfK") >= 0,
     bgres.filter(s => s.indexOf("eigf") >= 0).join(" | ") || "none");
  ok("  ... with `fields` appearing in no OTHER eigf group",
     bgres.filter(s => s.indexOf("eigf") >= 0).length === 1);
  // ... and the SIZES in that group agree with the WGSL. This is a whole device-only
  // failure class that no leg above can see, because nothing in node allocates: a uniform
  // bound whole under `layout: "auto"` is validated against the SizeOf of the declared
  // struct, so a struct that rounds UP past the buffer is a GPUValidationError out of
  // createBindGroup at boot -- in the Solver constructor, card open or not. A vec3<u32>
  // pad has align 16 and would put this struct at 32 against a 16-byte buffer, which is
  // exactly what adversarial review found on 2026-08-14; hence three scalar pads (the
  // `struct FL` pattern) and hence this leg, reflected off the EMITTED source and compared
  // with the size the page really asks the device for.
  if (M) {
    const R = new M.WgslReflect(gatherSrc);
    const uni = R.uniforms.filter(v => v.group === 0);
    const alloc = env2d.run("function(){ return solver.buf.eigfU.size; }");
    const declared = uni.length === 1 ? uni[0].size : -1;
    ok("the uniform struct's SizeOf is EXACTLY the eigfU buffer the page allocates",
       uni.length === 1 && uni[0].binding === 1 && declared === alloc && declared === 16,
       uni.map(v => v.name + " @" + v.binding + " = " + v.size + " B").join(",") +
       " vs eigfU " + alloc + " B");
    // the other two bindings are runtime-sized arrays, which carry no size contract at
    // all -- so that one number is the whole of this class for this kernel
    ok("  ... and the group's other two bindings are runtime-sized arrays, size-free",
       R.storage.filter(v => v.group === 0).length === 2 &&
       R.storage.filter(v => v.group === 0)
        .every(v => v.type && v.type.name === "array" && !v.size),
       R.storage.filter(v => v.group === 0).map(v => v.name + ":" + (v.type && v.type.name)).join(","));
  }
}

// ---------------------------------------------------------------------------
console.log("6. the card: its entry, options, hint, readback pool and presets");
// ---------------------------------------------------------------------------
{
  const C = env2d.run(`function(){ return { keys: Object.keys(CHART_TYPES), kmax: EIGF_KMAX }; }`);
  const T = env2d.run("function(){ return CHART_TYPES.eigf; }");
  ok("CHART_TYPES carries an `eigf` entry, with its own readback source",
     !!T && T.src === "eigf", T ? "src=" + T.src : "missing");
  // the placement rule of the plan's Sequencing section: immediately after `mode`, so the
  // concurrent CHI edit inside `aniso` stays a disjoint hunk
  ok("  ... immediately after `mode` in the table (and `aniso` untouched below it)",
     C.keys[C.keys.indexOf("mode") + 1] === "eigf" && C.keys.indexOf("aniso") > C.keys.indexOf("eigf"),
     C.keys.join(","));
  ok("  ... available in 2D and absent in 3D, the island/mode rule",
     env2d.run("function(){ return CHART_TYPES.eigf.avail({}) === true && !CHART_TYPES.eigf.avail({ zslice: true }); }"));
  ok("  ... it declares no zslice of its own (2D has no z to declare)", !T.zslice);
  const os_ = env2d.run("function(){ return CHART_TYPES.eigf.opts({}).map(s => ({ id: s.id, o: s.o })); }");
  ok("  ... its options are the k_y bin and the field selector",
     os_.map(s => s.id).join(",") === "eky,efld", os_.map(s => s.id).join(","));
  ok("  ... defaulting (first option of each) to bin 1 and both fields",
     os_[0].o[0][0] === "1" && os_[1].o[0][0] === "both",
     os_[0].o[0][0] + " / " + os_[1].o[0][0]);
  // island chain's seed is BROADBAND (icTearN, ~24 modes at its slider values), so the
  // list is not sized against the seed at all: it is sized against the growth-rate ladder,
  // whose peak is n = 6, and it has to run PAST the winner for the roll-over to be visible
  // (review 2026-08-14; the ladder itself is in EIGF_KMAX's comment)
  ok("  ... and the bin list runs to " + C.kmax + ", i.e. past island chain's n = 6 winner",
     os_[0].o.length === C.kmax && os_[0].o[C.kmax - 1][0] === String(C.kmax) && C.kmax >= 8,
     os_[0].o.map(o => o[0]).join(","));
  // the hint: what it must say, and the one thing it must NOT claim
  ok("  ... the hint names both moduli, the resonant surface and the linear y axis",
     /\|&psi;&#770;\(x\)\| and \|&phi;&#770;\(x\)\|/.test(T.hint) &&
     /resonant surface x = L<sub>x<\/sub>\/2/.test(T.hint) && /linear y/.test(T.hint),
     T.hint.length + " chars");
  ok("  ... says the k_y = 0 column IS the equilibrium, and that the column is LIVE",
     /k<sub>y<\/sub> = 0 column IS the/.test(T.hint) && /LIVE/.test(T.hint));
  ok("  ... and calls this the OUTER solution rather than promising a resistive layer",
     /OUTER solution/.test(T.hint) && /one cell/.test(T.hint) &&
     !/resolv(e|es|ed) the (resistive )?layer/.test(T.hint));
  // the readback pool, and the throttle that gates it
  const pool = env2d.run(`function(){
    const a = addChartCard("eigf"), b = addChartCard("eigf");
    cardsSync();
    const n = cards.chart.filter(c => (CHART_TYPES[c.type()].src || c.type()) === "eigf").length;
    const bins = [a, b].map(c => eigfBinOf(c.optVals()));
    return { n, bins, a: a, b: b };
  }`);
  ok("the eigf cards join ONE readback source pool", pool.n >= 2, "n = " + pool.n);
  ok("  ... on the same k_y bin by default, so two of them cost one readback",
     pool.bins[0] === 1 && pool.bins[1] === 1, pool.bins.join(","));
  env2d.run(`function(c){ const s = c.optEls.filter(x => x.__optId === "eky")[0]; s.value = "4"; s.onchange(); }`,
            pool.b);
  ok("  ... and moving one card's selector splits them into two",
     env2d.run("function(a, b){ return [eigfBinOf(a.optVals()), eigfBinOf(b.optVals())]; }",
               pool.a, pool.b).join(",") === "1,4");
  ok("  ... cardsThrottleReset re-opens the eigf window as it does the cut one",
     env2d.run(`function(){
       cardsThrottle.eigf = 999; cardsThrottle.eigfAt = "x/1";
       cardsThrottleReset();
       return cardsThrottle.eigf === 0 && cardsThrottle.eigfAt === null;
     }`));
  // The frame loop's own block cannot be DRIVEN here -- `loop()` is an infinite rAF loop
  // and the stub's requestAnimationFrame is a no-op, which is the same reason `renderCards`
  // had to be split out for checkidle -- so its wiring is read off the source instead, and
  // said to be: the marker gate, the 100 ms window, the retirement guard, one readback per
  // distinct bin, and each card handed the data for ITS OWN bin.
  {
    const csrc = fs.readFileSync(path.join(dir, "common.js"), "utf8");
    const blk = (csrc.split('const eigfCards = _chartsBySrc("eigf");')[1] || "").slice(0, 1200);
    const need = [/cardsThrottle\.eigfAt !== mark/, /performance\.now\(\) - cardsThrottle\.eigf > 100/,
                  /await sv\.readEigf\(j0\)/, /if \(sv !== solver\) break;/,
                  /bins\.set\(eigfBinOf\(c\.optVals\(\)\), null\)/,
                  /c\.draw\(bins\.get\(eigfBinOf\(c\.optVals\(\)\)\)\)/];
    ok("the frame loop's eigf readback is the cut line's idiom, bin for plane",
       need.every(re => re.test(blk)),
       need.filter(re => !re.test(blk)).map(re => re.source).join(" | ") || "all six");
  }
  // degenerates: the drawer must survive every shape the frame loop can hand it
  const degen = env2d.run(`function(){
    const rec = [];
    const c = { fillStyle: "", strokeStyle: "", lineWidth: 1, font: "10px x", textAlign: "left",
                textBaseline: "alphabetic" };
    for (const m of ["clearRect", "strokeRect", "beginPath", "moveTo", "lineTo", "stroke",
                     "fill", "clip", "save", "restore", "setLineDash", "rect", "fillRect"]) {
      c[m] = function () { for (const v of arguments) if (typeof v === "number" && !isFinite(v)) rec.push("NONFINITE"); };
    }
    c.measureText = t => ({ width: 6.2 * t.length });
    c.fillText = t => rec.push("T:" + t);
    const zero = new Float32Array(4 * 64);
    const live = new Float32Array(4 * 64);
    for (let i = 0; i < live.length; i++) live[i] = (i % 13) - 6;
    let waits = 0, drew = 0;
    for (const d of [null, {}, { vals: null, nx: 64, ny: 32 }, { vals: zero, nx: 64, ny: 32, Lx: 1 },
                     { vals: live, nx: 64, ny: 32, Lx: 12.5 }]) {
      for (const o of [null, {}, { eky: "junk", efld: "junk" }, { eky: "3", efld: "psi" },
                       { eky: "1", efld: "phi" }]) {
        rec.length = 0;
        drawEigf(c, d, o);
        if (rec.some(s => s === "NONFINITE")) return { bad: "non-finite on " + JSON.stringify(o) };
        if (rec.some(s => /waiting|no amplitude/.test(s))) waits++; else drew++;
      }
    }
    return { waits, drew, none: drawEigf(null, null, null) };
  }`);
  ok("drawEigf survives every (data x options) degenerate with nothing non-finite drawn",
     !degen.bad && degen.waits === 20 && degen.drew === 5 && degen.none === undefined,
     degen.bad || (degen.waits + " placeholder, " + degen.drew + " drawn"));
  // ... and the one shape the sweep above cannot make: ONE field non-finite beside a
  // finite other. The autoscale then comes off the finite field, so the card DRAWS -- and
  // the non-finite field's samples have to be skipped rather than handed to lineTo, which
  // a canvas accepts and silently discards. Found by adversarial review, 2026-08-14.
  const nanLeg = env2d.run(`function(){
    const rec = [];
    const c = { fillStyle: "", strokeStyle: "", lineWidth: 1, font: "10px x", textAlign: "left",
                textBaseline: "alphabetic" };
    for (const m of ["clearRect", "strokeRect", "beginPath", "moveTo", "lineTo", "stroke",
                     "fill", "clip", "save", "restore", "setLineDash", "rect", "fillRect"]) {
      c[m] = function () { for (const v of arguments) if (typeof v === "number" && !isFinite(v)) rec.push("NONFINITE"); };
    }
    c.measureText = t => ({ width: 6.2 * t.length });
    c.fillText = t => rec.push("T:" + t);
    // the phi half of the gather NaN, the psi half finite: eigfProfile transforms the two
    // columns separately, so exactly one of the two curves comes back non-finite at every x
    const v = new Float32Array(4 * 64);
    for (let i = 0; i < 2 * 64; i++) v[i] = NaN;
    for (let i = 2 * 64; i < v.length; i++) v[i] = (i % 13) - 6;
    const d = { vals: v, nx: 64, ny: 32, Lx: 12.5 };
    const out = [];
    for (const efld of ["both", "psi", "phi"]) {
      rec.length = 0;
      drawEigf(c, d, { eky: "1", efld: efld });
      out.push({ efld: efld, non: rec.some(s => s === "NONFINITE"),
                 wait: rec.some(s => /waiting|no amplitude/.test(s)) });
    }
    return out;
  }`);
  const nanBy = k => nanLeg.filter(r => r.efld === k)[0];
  ok("a NaN phi beside a finite psi draws psi, with nothing non-finite reaching the canvas",
     nanLeg.every(r => !r.non) && nanBy("both").wait === false &&
     nanBy("psi").wait === false && nanBy("phi").wait === true,
     nanLeg.map(r => r.efld + ":" + (r.non ? "NONFINITE" : r.wait ? "placeholder" : "drawn")).join(" "));
  // the presets: the card is the tearing preset's third chart, and available-not-default
  // everywhere else (the plan's Defaults section)
  const P = env2d.run(`function(){
    const o = {};
    for (const k in PRESETS) o[k] = (PRESETS[k].layout.charts || []).map(c => (c.t || c));
    return o;
  }`);
  ok("the `tearing` preset opens with island + cut/b + eigf",
     P.tearing.join(",") === "island,cut,eigf", P.tearing.join(","));
  ok("  ... and no other preset opens one (available, not default: collapse, chain, kh)",
     ["collapse", "chain", "kh", "forced", "decay", "rmhdvars"]
       .every(k => !P[k] || P[k].indexOf("eigf") < 0),
     Object.keys(P).map(k => k + ":" + P[k].join("+")).join(" "));
  // the plan's side task: the stale 30-40% clause is gone from the tearing hint
  const src2d = fs.readFileSync(path.join(dir, "rmhd2d.html"), "utf8");
  ok("the tearing hint's stale \"30-40% below the reference\" clause is gone",
     !/30&ndash;40%/.test(src2d) && !/dropping the\s*\"?\s*\+?\s*\"?measured slope/.test(src2d),
     "no 30-40% claim anywhere in rmhd2d.html");
  ok("  ... while the diffusion statement it was attached to still stands",
     /&psi;<sub>eq<\/sub> also diffuses at &sim;&eta;\/a&sup2;/.test(src2d));
  ok("2D boots clean through all of it", env2d.fails.length === 0, env2d.fails.join(" | "));
}
{
  const env3d = await boot(dir, "rmhd3d.html", null);
  const keys = env3d.run("function(){ return chartTypeKeys(); }");
  ok("the 3D app never offers the eigenfunction card", keys.indexOf("eigf") < 0, keys.join(","));
  ok("  ... and has no gather kernel to offer it either",
     env3d.run("function(){ return typeof buildShaders; }") === "function" &&
     Object.keys(cur["rmhd3d.html"] || {}).every(k => k.indexOf("eigfGather") < 0));
  ok("  ... 3D boots clean", env3d.fails.length === 0, env3d.fails.join(" | "));
}

console.log(bad ? "\n" + bad + " EIGF check(s) FAILED" : "\nall EIGF checks passed");
process.exit(bad ? 1 : 0);
})();

// the CPU-side strided column, in the gather kernel's own output layout. It is the
// FALLBACK for the legs that only need the column (3, 4) when wgsl_reflect is not
// installed -- section 2 is what makes it a stand-in rather than an assumption, since it
// asserts the kernel produces exactly this.
function cpuColumn(fields, j0) {
  const nx = G.nx, nky = G.nky, nm = G.nm, out = new Float32Array(4 * nx);
  const j = Math.max(0, Math.min(nky - 1, j0));
  for (let ix = 0; ix < nx; ix++) {
    const m = ix * nky + j;
    out[2 * ix] = fields[2 * m]; out[2 * ix + 1] = fields[2 * m + 1];
    out[2 * nx + 2 * ix] = fields[2 * (nm + m)];
    out[2 * nx + 2 * ix + 1] = fields[2 * (nm + m) + 1];
  }
  return out;
}
