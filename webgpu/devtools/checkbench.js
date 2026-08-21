// The FFTPERF_PLAN phase 0 gate: the ?bench harness and the fftKernel probe seam.
// Usage: node checkbench.js [webgpu-dir]          exit code 1 on any failure
//
// Five legs, the plan's (i)-(v):
//   i    the flag GATES the panel: without ?bench no #benchpanel element exists and
//        window.bench is undefined; with it both do -- on both pages. Then every
//        campaign is driven under the stub, which validates the dispatch shapes and the
//        bind groups the spec lists, and one JSON record per run must land in the
//        textarea.
//   ii   fftKernel / fftRowPair with NO probe are byte-identical to the emission
//        captured from the pre-phase tree (fixtures/fftkernel_<base>.json: every offered
//        line length, both directions, with and without `lpb`). This is the leg that
//        pins the kernel text from here on -- 2A/2B regenerate the capture, and
//        anything else that moves it fails.
//   iii  the two probes: "consttw" differs from the default in EXACTLY the two twiddle
//        lines and keeps every barrier; "copy" has no stage loop at all and still
//        carries the load and store bodies verbatim. All three parse (wgsl_reflect).
//   iv   the bytes-per-step sum reproduces a hand-computed number for 2D 256^2 and
//        3D 128^2 x 64 exactly (appendix A; the arithmetic is in the comment below).
//   v    fftAnalyticCase (the self-test's analytic reference, FFTPERF_PLAN phase 0's
//        other half) returns three nonzero bins at the flat indices it reports, zeros
//        everywhere else, and a real input of the right length.
//
// The fixture was captured from a clean checkout of its `base` commit by booting that
// tree's rmhd2d.html under stubenv and replaying `cases` through fftKernel /
// fftRowPair; regenerate it the same way whenever a phase is allowed to move the text.
"use strict";
const fs = require("fs"), path = require("path");
const { pathToFileURL } = require("url");
const dir = path.resolve(process.argv[2] || path.join(__dirname, ".."));
const FIX = path.join(__dirname, "fixtures", "fftkernel_f83386e.json");

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};
// boot a page and let its async boot() settle (initGPU + bootApply are awaited)
const boot = (page, opts) => new Promise(res => {
  const env = require("./stubenv")(dir, page, null, opts);
  let n = 0;
  const tick = () => (++n < 8 ? setTimeout(tick) : res(env));
  setTimeout(tick);
});
async function wgslMod() {
  const p = path.join(__dirname, "node_modules", "wgsl_reflect", "wgsl_reflect.module.js");
  if (!fs.existsSync(p)) return null;
  return await import(pathToFileURL(p).href);
}
const hasEl = (env, id) => env.allEls.some(e => e.id === id);

// ---------------------------------------------------------------------------
// (i) the flag gates the panel, and the campaigns run
// ---------------------------------------------------------------------------
async function legGate(page) {
  const off = await boot(page);
  ok(page + ": without ?bench there is no #benchpanel element",
     !hasEl(off, "benchpanel") && !hasEl(off, "benchout"),
     "panel " + hasEl(off, "benchpanel") + ", textarea " + hasEl(off, "benchout"));
  ok(page + ": ... and window.bench is undefined",
     off.sandbox.window.bench === undefined, typeof off.sandbox.window.bench);
  ok(page + ": ... and the plain boot raised no stub failures",
     off.fails.length === 0, off.fails.join(" | "));

  const env = await boot(page, { search: "?bench" });
  ok(page + ": with ?bench the panel and its textarea exist",
     hasEl(env, "benchpanel") && hasEl(env, "benchout"),
     "panel " + hasEl(env, "benchpanel") + ", textarea " + hasEl(env, "benchout"));
  const api = env.sandbox.window.bench;
  const want = ["whole", "kernels", "ladder", "chains", "all", "spec", "cfg", "text", "clear"];
  ok(page + ": ... and window.bench exposes the campaigns",
     !!api && want.every(k => api[k] !== undefined),
     api ? Object.keys(api).join(",") : "undefined");
  return env;
}
async function legCampaigns(env, page, chains) {
  const spec = env.run("() => { const s = benchSpec(); return { res: s.res, " +
    "kernels: s.kernels.map(c => c.name + ' ' + JSON.stringify(c.d)), " +
    "ffts: s.ffts.map(c => c.name), chains: (s.chains || []).map(c => c.name) }; }");
  ok(page + ": the spec lists every FFT kernel and the four step kernels",
     spec.kernels.length === spec.ffts.length + 4 && spec.ffts.length === (chains ? 6 : 4),
     spec.kernels.join(" | "));
  ok(page + ": ... and the gradient-chain cells are " + (chains ? "there" : "absent"),
     spec.chains.length === (chains ? 2 : 0), spec.chains.join(" | "));
  // the stub validates every dispatch extent and every bind group as the campaigns run
  env.run("() => { window.bench.cfg.K = 2; window.bench.cfg.R = 1; " +
          "window.bench.cfg.reps = 3; window.bench.cfg.chainReps = 2; }");
  await env.run("() => window.bench.all()");
  const txt = env.run("() => window.bench.text()");
  const recs = txt.trim().split("\n").filter(l => l).map(l => JSON.parse(l));
  ok(page + ": one JSON record landed in the textarea", recs.length === 1,
     recs.length + " records, " + txt.length + " chars");
  const r = recs[0] || {};
  const parts = r.parts || [];
  ok(page + ": ... the record names the page, the GPU and the resolution",
     r.page === page.replace(".html", "") && "gpu" in r &&
     r.nx === spec.res[0] && r.ny === spec.res[1] && r.nz === spec.res[2],
     JSON.stringify({ page: r.page, gpu: r.gpu, nx: r.nx, ny: r.ny, nz: r.nz }));
  ok(page + ": ... and carries all four campaigns", parts.length === 4,
     parts.map(p => p.campaign).join(", "));
  const whole = parts[0] || {};
  ok(page + ": ... the whole-step cell quotes bytes, GB/s and butterflies",
     whole.bytes_per_step > 0 && whole.butterflies_per_step > 0 &&
     isFinite(whole.cells[0].GB_s) && isFinite(whole.cells[0].Gbf_s),
     JSON.stringify(whole.cells && whole.cells[0]));
  const lad = parts[2] || {};
  const lc = (lad.cells || [])[0] || {};
  ok(page + ": ... and every ladder cell carries the three emissions and their shares",
     (lad.cells || []).length === spec.ffts.length &&
     (lad.cells || []).every(c => ["full_us_med", "consttw_us_med", "copy_us_med",
                                   "T_mem_us", "T_bf_us", "T_tw_us",
                                   "T_mem_share"].every(k => k in c)),
     JSON.stringify(lc));
  ok(page + ": the campaigns raised no stub failures (dispatch shapes, bind groups)",
     env.fails.length === 0, env.fails.join(" | "));
  // the app itself never sets a probe: the seam is null again, and the shipped text
  // still computes its twiddles
  const clean = env.run("() => { const s = benchSpec(); const S = s.build(s.g); " +
    "return { probe: FFT_PROBE, tw: /let wc: f32 = cos\\(ang\\);/.test(S.colsInv) }; }");
  ok(page + ": FFT_PROBE is null outside a bench emission, and the app's text is the full one",
     clean.probe === null && clean.tw === true, JSON.stringify(clean));
}

// ---------------------------------------------------------------------------
// (ii) the default emission is the captured one
// ---------------------------------------------------------------------------
function legCapture(env) {
  const fix = JSON.parse(fs.readFileSync(FIX, "utf8"));
  const got = env.run("(cases) => { const o = {};" +
    " for (const c of cases) {" +
    "   if (c.kind === 'kernel') o[c.name] = fftKernel(c.o);" +
    "   else { const p = fftRowPair(c.ny, c.nky, c.lpb); o[c.name + ' r2c'] = p.r2c;" +
    "          o[c.name + ' c2r'] = p.c2r; }" +
    " } return o; }", fix.cases);
  const names = Object.keys(fix.texts);
  const diff = names.filter(n => got[n] !== fix.texts[n]);
  ok("fftKernel / fftRowPair with no probe: byte-identical to " + fix.base,
     names.length === 64 && diff.length === 0 && Object.keys(got).length === names.length,
     names.length + " emissions" + (diff.length ? ", differ: " + diff.join(", ") : ""));
  // the capture really covers the matrix the leg claims
  const N = new Set(fix.cases.map(c => c.o ? c.o.N : c.ny));
  ok("  ... over N = 8..1024, both directions, with and without lpb, and through fftRowPair",
     [8, 16, 32, 64, 128, 256, 512, 1024].every(n => N.has(n)) &&
     fix.cases.filter(c => c.kind === "rowpair").length === 16 &&
     fix.cases.filter(c => c.kind === "kernel" && c.o.dir === 1).length === 16 &&
     fix.cases.filter(c => c.kind === "kernel" && c.o.lpb).length === 16,
     fix.cases.length + " cases");
}

// ---------------------------------------------------------------------------
// (iii) the two probes
// ---------------------------------------------------------------------------
const PROBE_CASE = {
  N: 256, dir: -1,
  decl: "@group(0) @binding(0) var<storage, read_write> dat: array<vec2<f32>>;\n"
      + "const NN: u32 = 256u;",
  load: "  let base: u32 = line * NN;\n"
      + "  for (var idx: u32 = tid; idx < NN; idx = idx + 32u) { buf[idx] = dat[base + idx]; }",
  store: "  for (var idx: u32 = tid; idx < NN; idx = idx + 32u) { dat[base + idx] = buf[src + idx]; }"
};
async function legProbes(env) {
  const emit = p => env.run("(o) => fftKernel(o)", Object.assign({}, PROBE_CASE, { probe: p }));
  const full = emit(undefined), ctw = emit("consttw"), cp = emit("copy");
  const A = full.split("\n"), B = ctw.split("\n");
  const idx = A.length === B.length ? A.map((l, i) => i).filter(i => A[i] !== B[i]) : null;
  ok("consttw: the line COUNT is unchanged and exactly two lines differ",
     idx !== null && idx.length === 2 && idx[1] === idx[0] + 1,
     idx === null ? A.length + " vs " + B.length + " lines" : "lines " + idx.join(", "));
  ok("  ... and they are the two twiddle lines",
     idx !== null && idx.length === 2 &&
     A[idx[0]] === "      let wc: f32 = cos(ang);" && A[idx[1]] === "      let ws: f32 = sin(ang);" &&
     B[idx[0]] === "      let wc: f32 = 1.0;" && B[idx[1]] === "      let ws: f32 = 0.0;",
     idx === null ? "" : JSON.stringify([A[idx[0]], B[idx[0]], A[idx[1]], B[idx[1]]]));
  const bars = s => (s.match(/workgroupBarrier\(\)/g) || []).length;
  ok("  ... every stage, barrier and butterfly survives",
     bars(ctw) === bars(full) && /for \(var s: u32/.test(ctw) &&
     /wc \* u1\.x - ws \* u1\.y/.test(ctw),
     bars(ctw) + " barriers, " + bars(full) + " in the shipped text");
  ok("copy: no stage loop at all", !/for \(var s/.test(cp), cp.split("\n").length + " lines");
  ok("  ... so its only barrier is the one between load and store", bars(cp) === 1,
     bars(cp) + " barriers");
  ok("  ... and the load and store bodies are verbatim",
     cp.indexOf(PROBE_CASE.load) >= 0 && cp.indexOf(PROBE_CASE.store) >= 0 &&
     full.indexOf(PROBE_CASE.load) >= 0 && full.indexOf(PROBE_CASE.store) >= 0);
  const M = await wgslMod();
  if (!M) { ok("  ... all three variants parse (wgsl_reflect)", true, "wgsl_reflect absent: skipped"); return; }
  const parses = [];
  for (const [n, s] of [["full", full], ["consttw", ctw], ["copy", cp]]) {
    try { new M.WgslParser().parse(s); } catch (e) { parses.push(n + ": " + e.message); }
  }
  ok("  ... all three variants parse (wgsl_reflect)", parses.length === 0, parses.join(" | "));
}

// ---------------------------------------------------------------------------
// (iv) bytes per step, against a hand-computed number
// ---------------------------------------------------------------------------
// FFTPERF_PLAN appendix A, with cx = nm*8 (one complex field), rx = nr*4 (one real
// field) and the grid buffers each kernel binds. Per stage: prepGrads 10cx; the 8-lane
// inverse chain 8*(2cx) per complex pass plus 8*(cx + rx) on the rows; bracket 10rx; the
// 2-lane forward chain 2*(rx + cx) on the rows plus 2*(2cx) per complex pass;
// nlAssemble 4cx + its grids; forcingAdd 4cx (2D) / the four (A,B) arrays plus both
// fields of the two kz planes (3D); stage 8cx + its grids. Three stages, plus
// energyPartial's 2cx + one grid once per step.
//
// 2D 256^2: nm = 256*129 = 33024, nr = 65536 -> cx = 264192, rx = 262144, gr = 528384.
//   prepGrads   10cx                    = 2641920
//   colsInv:8   16cx                    = 4227072
//   rowsC2R:8   8cx + 8rx               = 4210688
//   bracket     10rx                    = 2621440
//   rowsR2C:2   2rx + 2cx               = 1052672
//   colsFwd:2   4cx                     = 1056768
//   nlAssemble  4cx + 2gr               = 2113536
//   forcingAdd  4cx                     = 1056768
//   stage       8cx + gr                = 2641920
//   per stage                           = 21622784   x3 = 64868352
//   energyPartial 2cx + gr              = 1056768
//   TOTAL                               = 65925120
//
// 3D 128^2 x 64: nmp = 128*65 = 8320, nm = 64*8320 = 532480, nr = 1048576 ->
// cx = 4259840, rx = 4194304, grp = nmp*16 = 133120, grz = nz*16 = 1024.
//   prepGrads   10cx                    = 42598400
//   zInv:8      16cx                    = 68157440
//   colsInv:8   16cx                    = 68157440
//   rowsC2R:8   8cx + 8rx               = 67633152
//   bracket     10rx                    = 41943040
//   rowsR2C:2   2rx + 2cx               = 16908288
//   colsFwd:2   4cx                     = 17039360
//   zFwd:2      4cx                     = 17039360
//   nlAssemble  4cx + 2grp + grz        = 17306624
//   forcingAdd  12*nmp*8                = 798720
//   stage       8cx + grp + grz         = 34212864
//   per stage                           = 391794688  x3 = 1175384064
//   energyPartial 2cx + grp             = 8652800
//   TOTAL                               = 1184036864
const BYTES = { "rmhd2d.html": { fn: "benchSpec2D", res: [256, 256, 1], bytes: 65925120 },
                "rmhd3d.html": { fn: "benchSpec3D", res: [128, 128, 64], bytes: 1184036864 } };
function legBytes(env, page) {
  const w = BYTES[page];
  const got = env.run("(fn) => { const s = new Solver(device, {}); const sp = globalThis[fn](s);" +
    " const io = benchStepIO(sp); const r = { res: sp.res, bytes: io.bytes, bf: io.bf };" +
    " s.destroy(); return r; }", w.fn);
  ok(page + ": the default solver is the grid the hand count is for",
     String(got.res) === String(w.res), got.res + " vs " + w.res);
  ok(page + ": bytes per step = the hand-computed " + w.bytes,
     got.bytes === w.bytes, got.bytes + " B/step, " + got.bf + " butterflies");
}

// ---------------------------------------------------------------------------
// (v) the analytic self-test reference (fftAnalyticCase -- common.js)
// ---------------------------------------------------------------------------
function legAnalytic(env, g) {
  const tag = "fftAnalyticCase({" + Object.keys(g).map(k => k + ":" + g[k]).join(", ") + "})";
  const c = env.run("(g) => (typeof fftAnalyticCase === 'function' ? fftAnalyticCase(g) : null)", g);
  if (!c) {
    ok(tag, false, "fftAnalyticCase is not defined in common.js -- FFTPERF_PLAN phase 0's " +
       "self-test half has not landed yet");
    return;
  }
  const nz = g.nz || 1, nkx = g.nx, nky = g.ny / 2 + 1;
  const nr = nz * g.nx * g.ny, nm = nz * nkx * nky;
  ok(tag + ": a real input of nr = " + nr,
     c.input.length === nr && Array.prototype.every.call(c.input, v => isFinite(v)),
     c.input.length + " values");
  ok(tag + ": an interleaved spectrum of 2*nm = " + 2 * nm, c.expect.length === 2 * nm,
     c.expect.length + " values");
  const bins = Array.from(c.bins || []);
  ok(tag + ": three modes, each with ky > 0 and off Nyquist, kz = 0 in 2D",
     c.modes.length === 3 && bins.length === 3 &&
     c.modes.every(m => m.ky > 0 && m.ky < nky - 1 && Math.abs(m.kx) < g.nx / 2 &&
                        (nz > 1 || m.kz === 0) && Math.abs(m.kz) < nz),
     JSON.stringify(c.modes));
  const at = c.modes.map(m => (((m.kz % nz) + nz) % nz * nkx + (((m.kx % g.nx) + g.nx) % g.nx)) * nky + m.ky);
  ok(tag + ": the reported bins are the modes' own flat indices",
     bins.length === 3 && at.every((v, i) => v === bins[i]) && new Set(bins).size === 3,
     bins + " vs " + at);
  let stray = 0, zero = 0;
  for (let m = 0; m < nm; m++) {
    const re = c.expect[2 * m], im = c.expect[2 * m + 1];
    if (bins.indexOf(m) >= 0) { if (re === 0 && im === 0) zero++; }
    else if (re !== 0 || im !== 0) stray++;
  }
  ok(tag + ": exactly three nonzero bins, zeros everywhere else",
     stray === 0 && zero === 0, stray + " stray nonzero, " + zero + " of the three empty");
  // amp*nr/2*exp(i*phase), compared as a complex number (a negative amp is a pi flip,
  // not a negative modulus)
  const err = c.modes.map((m, i) => {
    const a = m.amp * nr / 2;
    return Math.hypot(c.expect[2 * bins[i]] - a * Math.cos(m.phase),
                      c.expect[2 * bins[i] + 1] - a * Math.sin(m.phase)) / Math.abs(a);
  });
  ok(tag + ": each bin holds amp*nr/2 at the mode's phase (unnormalized, exp(-i))",
     err.every(e => e < 1e-5), err.map(e => e.toExponential(1)).join(", "));
}

// ---------------------------------------------------------------------------
(async () => {
  console.log("(i) the ?bench flag gates the panel, and the campaigns run");
  const e2 = await legGate("rmhd2d.html");
  await legCampaigns(e2, "rmhd2d.html", false);
  const e3 = await legGate("rmhd3d.html");
  await legCampaigns(e3, "rmhd3d.html", true);
  console.log("(ii) the default fftKernel emission is the captured one");
  legCapture(e2);
  console.log("(iii) the ladder probes");
  await legProbes(e2);
  console.log("(iv) bytes per step against a hand count");
  legBytes(e2, "rmhd2d.html");
  legBytes(e3, "rmhd3d.html");
  console.log("(v) the analytic self-test reference");
  legAnalytic(e2, { nx: 32, ny: 32 });
  legAnalytic(e3, { nx: 16, ny: 16, nz: 8 });
  console.log(bad ? "bench harness: FAILED (" + bad + ")" : "bench harness: all green");
  process.exit(bad ? 1 : 0);
})();
