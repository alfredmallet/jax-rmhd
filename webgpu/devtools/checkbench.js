// The FFTPERF_PLAN phase 0 gate: the ?bench harness and the fftKernel probe seam.
// Usage: node checkbench.js [webgpu-dir]          exit code 1 on any failure
//
// Five legs, the plan's (i)-(v):
//   i    the flag GATES the panel: without ?bench no #benchpanel element exists and
//        window.bench is undefined; with it both do -- on both pages, with FFT_PROBE
//        null on a freshly booted page. Then every campaign is driven under the stub,
//        and one JSON record per run must land in the textarea. The spec's pipelines,
//        dispatch shapes and bind-group buffers are compared against the ones the
//        solver's OWN step encoded, cell by cell; a campaign must hold the frame loop
//        off (no steps, no render, no readStats) and leave the hero button paused, must
//        let the frame ALREADY in flight finish before it times anything, and must
//        restore through the page's own applyIC; and one cell must drain R + 1 times and
//        report the median and min of the reps it kept, the first discarded. The grads
//        hash cell (FFTPERF_PLAN item C's instrument) gets its own leg: the digest
//        against hand-computed FNV-1a values, the cell encoding the step's own gradient
//        chain, and the byte length it reads back and hashes per lane.
//   ii   fftKernel / fftRowPair with NO probe are byte-identical to the emission
//        captured from the pre-phase tree (fixtures/fftkernel_<base>.json: every offered
//        line length, both directions, with and without `lpb`), and so is the text the
//        Solver actually compiles for each FFT pipeline, at the self-test grid, at the
//        default preset and at the longest line the page offers (the fixture's
//        `pipelines`). This is the leg that pins the kernel text from here on -- 2A/2B
//        regenerate the capture, and anything else that moves it fails.
//   iii  the two probes: "consttw" differs from the default in EXACTLY the two twiddle
//        lines and keeps every barrier; "copy" has no stage loop at all and still
//        carries the load and store bodies verbatim. All three parse (wgsl_reflect).
//        A build that THROWS under benchShaders still leaves the seam null.
//   iv   the bytes-per-step sum reproduces a hand-computed number for 2D 256^2 and
//        3D 128^2 x 64 exactly (appendix A; the arithmetic is in the comment below),
//        and the 2D eqsrc solver's extra binding is counted.
//   v    fftAnalyticCase (the self-test's analytic reference, FFTPERF_PLAN phase 0's
//        other half) returns three nonzero bins at the flat indices it reports, zeros
//        everywhere else, and a real input of the right length; and fftAnalyticRows
//        leaves a VISIBLE skipped row, rather than throwing or nothing at all, when the
//        solver is rebuilt during either of its readbacks.
//
// The fixture was captured from a clean checkout of its `base` commit by booting that
// tree's rmhd2d.html under stubenv and replaying `cases` through fftKernel /
// fftRowPair; regenerate it the same way whenever a phase is allowed to move the text.
"use strict";
const fs = require("fs"), path = require("path");
const { pathToFileURL } = require("url");
const dir = path.resolve(process.argv[2] || path.join(__dirname, ".."));
const FIX = path.join(__dirname, "fixtures", "fftkernel_f83386e.json");

// FAILED until the summary says otherwise: a leg that parks forever (a held mapAsync
// nothing releases) drains node's queue and exits, and that exit must not read as green
process.exitCode = 1;

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
  const want = ["whole", "kernels", "ladder", "chains", "gradsHash", "all", "spec", "cfg",
                "text", "clear"];
  ok(page + ": ... and window.bench exposes the campaigns",
     !!api && want.every(k => api[k] !== undefined),
     api ? Object.keys(api).join(",") : "undefined");
  // BEFORE any campaign has run: the module-level seam is null on a page that has just
  // booted, so the solver's own kernels were compiled from the shipped text
  ok(page + ": FFT_PROBE is null on a freshly booted page, before any campaign",
     off.run("() => FFT_PROBE") === null && env.run("() => FFT_PROBE") === null,
     "no flag: " + off.run("() => String(FFT_PROBE)") +
     ", ?bench: " + env.run("() => String(FFT_PROBE)"));
  return env;
}

// ---------------------------------------------------------------------------
// (i.b) the spec is wired to the solver's own step
// ---------------------------------------------------------------------------
// The stub logs every dispatch as {pipeline, bind group, extent}, so what a spec cell
// claims can be compared against what encodeRHS / encodeStep actually did with that same
// pipeline over that same bind group -- shape, lane count, and (for the FFT cells, whose
// ladder variants need their own bind groups) the buffers in binding order.
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const shape = d => [d[0], d[1] || 1, d[2] || 1];
// every dispatch the step made that is NOT in the byte table, with the number of times
// the step makes it: the CFL pair (once per cflEvery block, so once in the step this leg
// records) and the per-step scalar kernels (FFTPERF_PLAN appendix A). Counts, not just
// names -- a kernel dispatched twice where the table assumes once is a byte count that
// has drifted, whatever it is called.
const UNCOUNTED = { cflPartial: 1, cflFinal: 1, energyFinal: 1, tick: 1, ou: 1, scale: 1 };
function legWiring(env, page) {
  const meta = env.run("() => { const sp = benchSpec(), s = solver, pl = {};" +
    " for (const k in s.pl) pl[k] = s.pl[k];" +
    " return { res: sp.res, pl: pl," +
    "   cells: sp.kernels.map(c => ({ name: c.name, pipe: c.pipe, bg: c.bg, d: c.d," +
    "                                 lanes: c.lanes || 0, bufs: c.bufs || null }))," +
    "   stepIO: sp.stepIO.map(e => ({ name: e.name.split(':')[0], n: e.n })) }; }");
  env.gpuReset();
  env.run("() => solver.step(1)");
  const disp = env.gpu.dispatches;
  const nky = meta.res[1] / 2 + 1, nz = meta.res[2];
  const lines = [meta.res[0], nky, nz > 1 ? nz * meta.res[0] : 0, nz > 1 ? nz * nky : 0,
                 nz > 1 ? meta.res[0] * nky : 0];
  for (const c of meta.cells) {
    const mine = disp.filter(e => e.pipe === c.pipe && e.bg === c.bg);
    const bad_ = mine.filter(e => !same(shape(e.d), shape(c.d)));
    ok(page + ": " + c.name + " -- the step dispatches this pipeline over this bind group",
       mine.length > 0, mine.length + " dispatches of " + disp.length);
    ok(page + ":   ... at the spec's own extent " + JSON.stringify(shape(c.d)),
       mine.length > 0 && bad_.length === 0,
       bad_.length ? "the step used " + JSON.stringify(bad_.map(e => e.d)) : "");
    if (c.lanes) {
      // 3D batches lanes in y; 2D folds them into x, so x must be `lanes` whole lines
      const d = shape(c.d);
      ok(page + ":   ... and its " + c.lanes + " lanes are the batch the step used",
         nz > 1 ? d[1] === c.lanes : d[1] === 1 && lines.indexOf(d[0] / c.lanes) >= 0,
         JSON.stringify(d));
    }
    if (c.bufs) {
      const got = (mine[0] && mine[0].bg.__buffers) || [];
      const off = c.bufs.map((b, i) => (b === got[i] ? -1 : i)).filter(i => i >= 0);
      ok(page + ":   ... and its bufs are that bind group's buffers, in binding order",
         same(c.bufs, got),
         off.length ? "binding " + off.join(", ") + " is not what the step bound"
                    : c.bufs.length + " of " + got.length + " bound");
    }
  }
  // the gradient chain's four row-kernel targets: the SAME buffer, four windows into it,
  // in pair order, each two real lanes wide (FFTPERF_PLAN 2C). The stub keeps a binding's
  // {buffer, offset, size}, so a chain that wrote every pair over lane 0 -- or bound the
  // whole eight-lane stack four times -- is visible here and nowhere else.
  const gradRows = env.run("() => { const s = solver;" +
    " return { bg: s.bg.rowsC2RGrads, real: s.buf.realGrads, nr: s.nr," +
    "          prep: s.pl.prepGrads.map(p => p.__name) }; }");
  const npair = gradRows.bg.length, lane2 = 2 * gradRows.nr * 4;
  const wantOff = gradRows.bg.map((b, k) => k * lane2);
  const winOf = e => {
    const w = e.bg.__bindings && e.bg.__bindings.filter(x => x.buffer === gradRows.real)[0];
    return w ? [w.offset, w.size] : null;
  };
  const rows = disp.filter(e => gradRows.bg.indexOf(e.bg) >= 0);
  const perBg = gradRows.bg.map(b => rows.filter(e => e.bg === b).length);
  const firstRows = rows.slice(0, npair).map(winOf);
  ok(page + ": the chain's four row-kernel targets are realGrads at " +
     JSON.stringify(wantOff) + ", two lanes (" + lane2 + " B) each, in pair order",
     firstRows.length === npair &&
     firstRows.every((w, k) => w && w[0] === wantOff[k] && w[1] === lane2),
     JSON.stringify(firstRows));
  ok(page + ":   ... and the step runs each of the four the same number of times",
     rows.length === npair * perBg[0] && perBg.every(n => n === perBg[0]),
     perBg.join(", ") + " dispatches, " + rows.length + " in the step");
  // ... off four DISTINCT prep pipelines: four instantiations of one template, in pair
  // order, not one pipeline dispatched four times
  const preps = disp.filter(e => gradRows.prep.indexOf(e.pipe.__name) >= 0);
  const firstPreps = preps.slice(0, npair);
  ok(page + ":   ... each behind its own prepGrads pipeline, " + gradRows.prep.join(", "),
     preps.length === npair * perBg[0] &&
     same(firstPreps.map(e => e.pipe.__name), gradRows.prep) &&
     new Set(firstPreps.map(e => e.pipe)).size === npair,
     firstPreps.map(e => e.pipe.__name).join(", ") || "none");

  // the byte table's `n` column against the same recorded step, and nothing dispatched
  // that the table neither counts nor names as uncounted
  const per = new Map();
  for (const e of disp) per.set(e.pipe, (per.get(e.pipe) || 0) + 1);
  // a solver pipeline slot may hold a GROUP of pipelines built from one template (the four
  // per-pair prepGrads); the table's `n` counts the group's dispatches together
  const group = n => (Array.isArray(meta.pl[n]) ? meta.pl[n] : [meta.pl[n]]);
  const count = n => group(n).reduce((t, p) => t + (per.get(p) || 0), 0);
  const wrong = meta.stepIO.filter(e => count(e.name) !== e.n);
  ok(page + ": every stepIO entry is dispatched exactly `n` times per step",
     wrong.length === 0,
     wrong.map(e => e.name + ": " + e.n + " claimed, " + count(e.name) + " seen").join(", "));
  const un = Object.keys(UNCOUNTED);
  const named = new Set([].concat.apply([], meta.stepIO.map(e => e.name).concat(un).map(group)));
  const stray = [...per.keys()].filter(p => !named.has(p));
  ok(page + ": ... and the step dispatches nothing the table has not accounted for",
     stray.length === 0, stray.map(p => p.__name).join(", "));
  const overUn = un.filter(n => count(n) !== UNCOUNTED[n]);
  ok(page + ": ... and each kernel the table excuses is dispatched exactly as often as it assumes",
     overUn.length === 0,
     overUn.map(n => n + ": " + UNCOUNTED[n] + " assumed, " + count(n) + " seen").join(", "));
}

// ---------------------------------------------------------------------------
// (i.e) the grads hash cell
// ---------------------------------------------------------------------------
// FFTPERF_PLAN item C's gate is "the real-space gradients, bitwise, before and after",
// and the comparison itself is a readback on a device. What is checkable without one is
// the INSTRUMENT: the digest against hand-computed FNV-1a values, the cell running the
// solver's OWN gradient chain (the step's first dispatches -- same pipelines, same bind
// groups, same extents, in order), and the byte length it reads back and hashes per lane.
// The stub hands back zeros, so the digests it produces are the model's digest of zeros:
// what that pins is the length and the lane split, not the gradients.
const fnv1a = u => {
  let h = 2166136261;
  for (let i = 0; i < u.length; i++) {
    const w = u[i];
    for (let s = 0; s < 32; s += 8) h = Math.imul(h ^ ((w >>> s) & 255), 16777619);
  }
  return h >>> 0;
};
async function legGradsHash(env, page) {
  // the model first, against digests computed by hand from the 32-bit offset basis
  // (2166136261) and prime (16777619) over each word's bytes, low byte first
  ok("the check's own FNV-1a model reproduces hand-computed digests",
     fnv1a(new Uint32Array(0)) === 2166136261 &&
     fnv1a(new Uint32Array([1, 2, 3])) === 2034659765 &&
     fnv1a(new Uint32Array([0xffffffff])) === 3809873841,
     [fnv1a(new Uint32Array(0)), fnv1a(new Uint32Array([1, 2, 3])),
      fnv1a(new Uint32Array([0xffffffff]))].join(", "));
  const cases = [[], [1, 2, 3], [1, 2, 4], [0xffffffff]];
  const got = env.run("(cs) => cs.map(c => benchHash32(new Uint32Array(c)))", cases);
  const want = cases.map(c => fnv1a(new Uint32Array(c)));
  ok(page + ": benchHash32 is that same digest, and one changed word changes it",
     want.every((v, i) => v === got[i]) && got[1] !== got[2],
     got.join(", ") + " vs " + want.join(", "));
  // the cell's chain against the step's own: the gradient chain is what a step encodes
  // first -- one prep and its inverse passes (two in 2D, three in 3D) per gradient pair
  const nchain = env.run("() => GRAD_PAIRS.length") * (env.is3d ? 4 : 3);
  env.gpuReset();
  env.run("() => solver.step(1)");
  const chain = env.gpu.dispatches.slice(0, nchain);
  env.gpuReset();
  const rec = await env.run("() => window.bench.gradsHash()");
  const tail = env.gpu.dispatches.slice(-nchain);
  const off = chain.map((e, i) => (tail[i] && tail[i].pipe === e.pipe && tail[i].bg === e.bg &&
                                   same(shape(tail[i].d), shape(e.d)) ? null : i)).filter(i => i !== null);
  ok(page + ": the grads hash cell encodes the step's own gradient chain, in order",
     tail.length === nchain && off.length === 0,
     tail.map(e => e.pipe.__name + JSON.stringify(e.d)).join(" ") + " vs " +
     chain.map(e => e.pipe.__name + JSON.stringify(e.d)).join(" "));
  ok(page + ": ... and dispatches nothing else after the IC it re-applies",
     env.gpu.dispatches.length > nchain &&
     env.gpu.dispatches.slice(-nchain - 1)[0].pipe !== chain[0].pipe,
     env.gpu.dispatches.length + " dispatches in all");
  // ONE chain, so the four pairs are exactly four dispatches here: four distinct prep
  // pipelines, and four windows into realGrads at 2*k*nr*4, two lanes wide (2C). A chain
  // that wrote every pair over lane 0 would hash the same buffer four times and be
  // invisible to the digest itself.
  const G = env.run("() => ({ real: solver.buf.realGrads, nr: solver.nr," +
    " prep: solver.pl.prepGrads.map(p => p.__name) })");
  const lane2 = 2 * G.nr * 4;
  const wins = tail.filter(e => e.bg.__bindings &&
                                e.bg.__bindings.some(x => x.buffer === G.real))
    .map(e => e.bg.__bindings.filter(x => x.buffer === G.real)
                             .map(x => [x.offset, x.size])[0]);
  ok(page + ": ... the chain's four writes are realGrads at 0, " + lane2 + ", " + 2 * lane2
     + ", " + 3 * lane2 + ", two lanes each",
     wins.length === G.prep.length &&
     wins.every((w, k) => w[0] === k * lane2 && w[1] === lane2),
     JSON.stringify(wins));
  const preps = tail.filter(e => G.prep.indexOf(e.pipe.__name) >= 0);
  ok(page + ": ... and its four preps are the four distinct pipelines, in pair order",
     same(preps.map(e => e.pipe.__name), G.prep) &&
     new Set(preps.map(e => e.pipe)).size === G.prep.length,
     preps.map(e => e.pipe.__name).join(", ") || "none");
  // the readback: eight lanes of one real field each, hashed lane by lane and whole
  const nr = env.run("() => solver.nr");
  const cell = (rec && rec.cells && rec.cells[0]) || {};
  ok(page + ": the record is one grads hash cell over 8 * nr * 4 = " + 8 * nr * 4 + " bytes",
     rec.campaign === "grads hash" && cell.cell === "grads hash" &&
     cell.bytes === 8 * nr * 4 && cell.lane_bytes === nr * 4,
     JSON.stringify({ campaign: rec && rec.campaign, bytes: cell.bytes, lane: cell.lane_bytes }));
  ok(page + ": ... with eight lane digests and one over the whole buffer",
     Array.isArray(cell.hash_lane) && cell.hash_lane.length === 8 &&
     typeof cell.hash_all === "number",
     JSON.stringify({ lanes: cell.hash_lane, all: cell.hash_all }));
  const zl = fnv1a(new Uint32Array(nr)), za = fnv1a(new Uint32Array(8 * nr));
  ok(page + ": ... and they are the model's digests of what the stub read back (zeros)",
     (cell.hash_lane || []).every(h => h === zl) && cell.hash_all === za,
     "lane " + (cell.hash_lane || [])[0] + " vs " + zl + ", all " + cell.hash_all + " vs " + za);
  // a digest that depended on anything but (page, IC, resolution) would move between runs
  const again = await env.run("() => window.bench.gradsHash()");
  const c2 = (again && again.cells && again.cells[0]) || {};
  ok(page + ": ... and a second run of the cell reports the same digests",
     c2.hash_all === cell.hash_all &&
     (c2.hash_lane || []).every((h, i) => h === (cell.hash_lane || [])[i]),
     c2.hash_all + " vs " + cell.hash_all);
}

// ---------------------------------------------------------------------------
// (i.c) a campaign owns the queue: the frame loop is held off for its duration
// ---------------------------------------------------------------------------
// The stub parks rAF callbacks, so a frame happens only when env.frame() fires one --
// here from inside every queue drain, i.e. exactly where a real rAF would land during a
// campaign. What must not happen while the campaign runs: a display chain encoded, a
// stats readback, or a "Pause" hero button.
async function legLoopHold(page) {
  const env = await boot(page, { search: "?bench" });
  env.sandbox.window.__frame = env.frame;
  env.run("() => { setRunning(true); window.bench.cfg.K = 2; window.bench.cfg.R = 1;" +
    " const s = solver, r = s.render.bind(s), rs = s.readStats.bind(s), q = device.queue;" +
    " window.__seen = { frames: 0, renders: 0, stats: 0, running: [], btn: [] };" +
    " window.__drain = q.onSubmittedWorkDone;" +
    " s.render = function (ctx, ci) { window.__seen.renders++; return r(ctx, ci); };" +
    " s.readStats = function () { window.__seen.stats++; return rs(); };" +
    " q.onSubmittedWorkDone = async () => {" +
    "   window.__seen.frames += window.__frame();" +
    "   window.__seen.running.push(running); window.__seen.btn.push(el('btnRun').textContent);" +
    "   return window.__drain(); }; }");
  await env.run("() => window.bench.whole()");
  env.run("() => { device.queue.onSubmittedWorkDone = window.__drain; }");
  const seen = env.run("() => window.__seen");
  ok(page + ": frames really landed inside the campaign", seen.frames >= 2, seen.frames + " frames");
  ok(page + ": ... and drew nothing and read no stats while it ran",
     seen.renders === 0 && seen.stats === 0 && env.gpu.renders === 0,
     seen.renders + " chains, " + seen.stats + " readStats, " + env.gpu.renders + " render passes");
  ok(page + ": ... with the run paused through the hero button",
     seen.running.length > 0 && seen.running.every(v => v === false) &&
     seen.btn.every(t => t === "Run"),
     JSON.stringify(seen.running) + " " + JSON.stringify(seen.btn));
  // ... and the loop is a loop again afterwards: the campaign's steps left every card
  // dirty, so the next frame draws them and reads the scalars back
  env.frame();
  for (let i = 0; i < 12; i++) await new Promise(r => setTimeout(r, 0));
  const after = env.run("() => window.__seen");
  ok(page + ": ... and the very next frame after it draws and reads stats again",
     after.renders > 0 && after.stats > 0,
     after.renders + " chains, " + after.stats + " readStats");
}

// ---------------------------------------------------------------------------
// (i.c2) the frame already in flight when the campaign starts
// ---------------------------------------------------------------------------
// The benchBusy guard only stops the NEXT iteration. This drives the one case it cannot
// see: a frame parked on its own drain (its render, stats readback and frame hooks still
// to come) when the button is pressed. The campaign must not submit a single batch until
// that iteration has finished, and nothing of it may land in a timed rep.
async function legLoopRace(page) {
  const env = await boot(page, { search: "?bench" });
  env.sandbox.window.__frame = env.frame;
  env.run("() => { setRunning(true); window.bench.cfg.K = 2; window.bench.cfg.R = 2;" +
    " const s = solver, rs = s.readStats.bind(s), r = s.render.bind(s), q = device.queue;" +
    " const fh = frameHook;" +
    " window.__seen = { stats: 0, renders: 0, hooks: 0, statsTimed: 0, hooksTimed: 0," +
    "                   rendersTimed: 0, steps: 0, busyAtFirstStep: null };" +
    " const timed = () => benchBusy && window.__seen.steps > 0;" +
    " s.readStats = function () { window.__seen.stats++; if (timed()) window.__seen.statsTimed++; return rs(); };" +
    " s.render = function (ctx, ci) { window.__seen.renders++; if (timed()) window.__seen.rendersTimed++; return r(ctx, ci); };" +
    " frameHook = async sv => { window.__seen.hooks++; if (timed()) window.__seen.hooksTimed++;" +
    "                           if (fh) await fh(sv); };" +
    // the campaign's timed window opens at the first step of its first rep
    " const step = s.step.bind(s);" +
    " s.step = function (ce) { if (benchBusy) { if (window.__seen.steps === 0)" +
    "     window.__seen.busyAtFirstStep = loopBusy; window.__seen.steps++; } return step(ce); };" +
    // one-shot hold on the LOOP's own drain: window.__release() lets that frame finish
    "  const drain = q.onSubmittedWorkDone; let held = null;" +
    " window.__release = () => { const h = held; held = null; if (h) h(); return !!h; };" +
    " q.onSubmittedWorkDone = function () {" +
    "   if (window.__armed && held === null) { window.__armed = false;" +
    "     return new Promise(res => { held = res; }); }" +
    "   return drain(); }; }");
  env.run("() => { window.__armed = true; }");
  env.frame();
  for (let i = 0; i < 4; i++) await new Promise(r => setTimeout(r, 0));
  const parked = env.run("() => ({ armed: window.__armed, loopBusy: loopBusy })");
  ok(page + ": a frame is parked mid-iteration when the button is pressed",
     parked.armed === false && parked.loopBusy === true, JSON.stringify(parked));
  const done = env.run("() => window.bench.whole()");
  for (let i = 0; i < 3; i++) await new Promise(r => setTimeout(r, 0));
  const releasedLate = env.run("() => window.__release()");
  await done;
  for (let i = 0; i < 12; i++) await new Promise(r => setTimeout(r, 0));
  const seen = env.run("() => window.__seen");
  ok(page + ": ... its drain was still held when the campaign was asked to run",
     releasedLate === true, "released by the leg: " + releasedLate);
  ok(page + ": ... and the campaign stepped nothing until that frame was done",
     seen.busyAtFirstStep === false && seen.steps > 0,
     "loopBusy at the first step: " + seen.busyAtFirstStep + ", " + seen.steps + " steps");
  ok(page + ": ... so no stats readback, hook or render landed in a timed rep",
     seen.statsTimed === 0 && seen.hooksTimed === 0 && seen.rendersTimed === 0,
     seen.statsTimed + " readStats, " + seen.hooksTimed + " hooks, " + seen.rendersTimed +
     " renders in the reps (" + seen.stats + " / " + seen.hooks + " / " + seen.renders + " in all)");
  ok(page + ": ... and the parked frame did run its tail, before the campaign",
     seen.stats > 0 && seen.hooks > 0, seen.stats + " readStats, " + seen.hooks + " hooks");
}

// ---------------------------------------------------------------------------
// (i.d) one cell: R + 1 reps, the first discarded, median and min reported
// ---------------------------------------------------------------------------
// performance.now is replaced by a sequence whose FIRST rep is the cheap one, so a kept
// warm-up rep shows up as a `min` below the median. Drains are counted in the stub.
// A rep is "everything up to the next drain", so the clock is driven by the DRAINS and
// not by how many times now() is called: it reads 1000*r during rep r and 1000*r + dur[r]
// on the first call after that rep's drain. Calls from anywhere else (benchGo's wait for
// the frame loop) fall on a rep boundary and change no duration.
const CLOCK = "() => { window.__now = performance.now;" +
  " const q = device.queue, drain = q.onSubmittedWorkDone;" +
  " const dur = [1, 10, 10]; let r = 0, ended = false;" +
  " window.__drain = drain;" +
  " q.onSubmittedWorkDone = async () => { const v = await drain(); ended = true; return v; };" +
  " performance.now = () => { if (!ended) { return 1000 * r; }" +
  "   ended = false; r++; return 1000 * (r - 1) + dur[Math.min(r - 1, dur.length - 1)]; }; }";
async function legReps(env, page) {
  env.run("() => { window.bench.cfg.K = 1; window.bench.cfg.R = 2; }");
  env.run(CLOCK);
  env.gpuReset();
  const rec = await env.run("() => window.bench.whole()");
  env.run("() => { performance.now = window.__now;" +
          " device.queue.onSubmittedWorkDone = window.__drain; }");
  const cell = (rec && rec.cells && rec.cells[0]) || {};
  ok(page + ": one cell drains R + 1 times (the warm-up rep included)",
     env.gpu.drains === 3, env.gpu.drains + " drains for R = 2");
  ok(page + ": ... and reports both the median and the min of the reps it kept",
     typeof cell.ms_med === "number" && typeof cell.ms_min === "number",
     JSON.stringify(cell));
  ok(page + ": ... with the first rep discarded (the cheap one is not the min)",
     cell.ms_med === 10 && cell.ms_min === 10,
     "med " + cell.ms_med + ", min " + cell.ms_min + " of reps 1, 10, 10");
}
async function legCampaigns(env, page, chains) {
  const spec = env.run("() => { const s = benchSpec(); return { res: s.res, " +
    "kernels: s.kernels.map(c => c.name + ' ' + JSON.stringify(c.d)), " +
    "ffts: s.ffts.map(c => c.name), chains: (s.chains || []).map(c => c.name) }; }");
  ok(page + ": the spec lists every FFT kernel and the four step kernels",
     spec.kernels.length === spec.ffts.length + 4 && spec.ffts.length === (chains ? 6 : 4),
     spec.kernels.join(" | "));
  ok(page + ": ... and the gradient-chain cell is " + (chains ? "there" : "absent"),
     spec.chains.length === (chains ? 1 : 0), spec.chains.join(" | "));
  // the stub validates every dispatch extent and every bind group as the campaigns run
  env.run("() => { window.bench.cfg.K = 2; window.bench.cfg.R = 1; " +
          "window.bench.cfg.reps = 3; window.bench.cfg.chainReps = 2; }");
  // ... and the restore goes through the PAGE's applyIC (the selected preset), not the
  // solver's built-in modes IC
  env.run("() => { window.__ic = { applyIC: 0, setIC: 0 };" +
    " const a = benchPage.applyIC, s = solver, si = s.setIC.bind(s);" +
    " benchPage.applyIC = () => { window.__ic.applyIC++; return a(); };" +
    " s.setIC = function (z) { window.__ic.setIC++; return si(z); }; }");
  await env.run("() => window.bench.all()");
  const ic = env.run("() => window.__ic");
  // per kernel, ladder, (chains), and the grads hash cell -- which applies the IC at its
  // START, so that the chain it hashes runs on a state a reload reproduces
  ok(page + ": each campaign that trampled or depended on the fields went through applyIC",
     ic.applyIC === (chains ? 4 : 3) && ic.setIC <= ic.applyIC,
     ic.applyIC + " applyIC, " + ic.setIC + " setIC (expected " + (chains ? 4 : 3) +
     ": per kernel, ladder" + (chains ? ", chains" : "") + ", grads hash)");
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
  ok(page + ": ... and carries all five campaigns", parts.length === 5,
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
// ... and the text the SOLVER compiles, which is what the app runs: each FFT pipeline's
// module against the same capture, at the self-test grid, at the default preset (through
// the page's LIVE solver, the one boot() built before anything here ran) and at the
// LONGEST line the page's selRes offers, where a length-gated emission would hide.
const LONGEST = { "rmhd2d.html": { nx: 1024, ny: 1024 },
                  "rmhd3d.html": { nx: 64, ny: 64, nz: 256 } };
function legPipelines(env, page) {
  const want = JSON.parse(fs.readFileSync(FIX, "utf8")).pipelines[page];
  const grids = Object.keys(want);
  const got = env.run("(grids, keys, long) => { const R = REFVEC, out = {};" +
    " const dims = { selftest: { nx: R.nx, ny: R.ny, nz: R.nz }, preset: null, longest: long };" +
    " for (const g of grids) { const s = dims[g] ? new Solver(device, dims[g]) : solver;" +
    "   const t = { __res: [s.g.nx, s.g.ny, s.g.nz || 1] };" +
    "   for (const k of keys) t[k] = s.pl[k].__code;" +
    "   if (dims[g]) s.destroy(); out[g] = t; }" +
    " return out; }", grids, Object.keys(want[grids[0]]).filter(k => k !== "__res"),
    LONGEST[page]);
  const nk = env.is3d ? 6 : 4;                  // the row / column pair, plus z in 3D
  for (const g of grids) {
    const keys = Object.keys(want[g]).filter(k => k !== "__res");
    const diff = keys.filter(k => got[g][k] !== want[g][k]);
    ok(page + ": the Solver's compiled " + g + " FFT kernels are the captured text",
       String(got[g].__res) === String(want[g].__res) && diff.length === 0 && keys.length === nk,
       got[g].__res + ": " + keys.join(", ") + (diff.length ? " -- differ: " + diff.join(", ") : ""));
  }
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
// benchShaders owns the seam for exactly one build call: a build that throws must still
// hand it back, or every kernel the app emits afterwards is a probe variant
function legSeamRestore(env) {
  const r = env.run("() => { const sp = benchSpec(); let threw = '';" +
    " try { benchShaders(() => { throw new Error('build failed'); }, sp.g, 'copy'); }" +
    " catch (e) { threw = e.message; }" +
    " const S = sp.build(sp.g);" +
    " return { threw: threw, probe: FFT_PROBE, tw: /let wc: f32 = cos\\(ang\\);/.test(S.colsInv) }; }");
  ok("a throwing build leaves FFT_PROBE null and the next emission is the shipped text",
     r.threw === "build failed" && r.probe === null && r.tw === true, JSON.stringify(r));
}

// ---------------------------------------------------------------------------
// (iv) bytes per step, against a hand-computed number
// ---------------------------------------------------------------------------
// FFTPERF_PLAN appendix A, kernel by kernel: cx = nm*8 (one complex field), rx = nr*4
// (one real field), and each grid buffer a kernel binds (gr / grp perpendicular, grz in
// z). A buffer both read and written by the same kernel is counted twice.
//
// Since FFTPERF_PLAN 2C the gradient chain runs one (x, y) PAIR at a time: four preps and
// four two-lane inverse chains per stage, so those rows are dispatched 12 times per step
// and not 3, and each prep reads ONE state field (phi or psi) and writes two lanes. The
// transforms move the same bytes either way -- 12 x 2 lanes is 3 x 8 -- and prepGrads is
// the row that grows: 12(3cx + gr) against 3(10cx + gr).
//
// 2D 256^2: nm = 256*129 = 33024, nr = 65536 -> cx = 264192, rx = 262144, gr = 528384.
//   prepGrads   x12  cx + gr + 2cx          =  1320960 ->  15851520
//   colsInv:2   x12  4cx                    =  1056768 ->  12681216
//   rowsC2R:2   x12  2cx + 2rx              =  1052672 ->  12632064
//   bracket     x3   10rx                   =  2621440 ->   7864320
//   rowsR2C:2   x3   2rx + 2cx              =  1052672 ->   3158016
//   colsFwd:2   x3   4cx                    =  1056768 ->   3170304
//   nlAssemble  x3   2cx + 2gr + 2cx        =  2113536 ->   6340608
//   forcingAdd  x3   2cx + 2cx + 2cx        =  1585152 ->   4755456
//   stage       x3   10cx + gr              =  3170304 ->   9510912
//   energyPartial x1 2cx + 2gr              =  1585152 ->   1585152
//   TOTAL                                                = 77549568
//
// 3D 128^2 x 64: nmp = 128*65 = 8320, nm = 64*8320 = 532480, nr = 1048576 ->
// cx = 4259840, rx = 4194304, grp = nmp*16 = 133120, grz = nz*16 = 1024.
//   prepGrads   x12  cx + grp + 2cx         = 12912640 -> 154951680
//   zInv:2      x12  4cx                    = 17039360 -> 204472320
//   colsInv:2   x12  4cx                    = 17039360 -> 204472320
//   rowsC2R:2   x12  2cx + 2rx              = 16908288 -> 202899456
//   bracket     x3   10rx                   = 41943040 -> 125829120
//   rowsR2C:2   x3   2rx + 2cx              = 16908288 ->  50724864
//   colsFwd:2   x3   4cx                    = 17039360 ->  51118080
//   zFwd:2      x3   4cx                    = 17039360 ->  51118080
//   nlAssemble  x3   2cx + 2grp + grz + 2cx = 17306624 ->  51919872
//   forcingAdd  x3   4*nmp*8 + 2*(4*nmp*8)  =   798720 ->   2396160
//   stage       x3   10cx + grp + grz       = 42732544 -> 128197632
//   energyPartial x1 2cx + 2grp + grz       =  8786944 ->   8786944
//   TOTAL                                                = 1236886528
const BYTES = { "rmhd2d.html": { fn: "benchSpec2D", res: [256, 256, 1], bytes: 77549568,
                                 eqsrc: true },
                "rmhd3d.html": { fn: "benchSpec3D", res: [128, 128, 64], bytes: 1236886528 } };
function legBytes(env, page) {
  const w = BYTES[page];
  const got = env.run("(fn) => { const s = new Solver(device, {}); const sp = globalThis[fn](s);" +
    " const io = benchStepIO(sp); const r = { res: sp.res, bytes: io.bytes, bf: io.bf };" +
    " s.destroy(); return r; }", w.fn);
  ok(page + ": the default solver is the grid the hand count is for",
     String(got.res) === String(w.res), got.res + " vs " + w.res);
  ok(page + ": bytes per step = the hand-computed " + w.bytes,
     got.bytes === w.bytes, got.bytes + " B/step, " + got.bf + " butterflies");
  if (!w.eqsrc) return;
  // the maintained-flux equilibrium binds psi_eq,k as nlAssemble's fifth buffer: one more
  // complex field read per stage, and nothing else moves
  const eq = env.run("(fn) => { const s = new Solver(device, { eqsrc: true });" +
    " const sp = globalThis[fn](s); const io = benchStepIO(sp);" +
    " const r = { eqk: !!s.buf.eqk, bytes: io.bytes, cx: s.g.nm * 8 }; s.destroy(); return r; }", w.fn);
  ok(page + ": ... and an eqsrc solver counts its eqk read, 1cx per stage, on top",
     eq.eqk === true && eq.bytes === w.bytes + 3 * eq.cx,
     eq.bytes + " B/step, " + (eq.bytes - w.bytes) + " over the eqsrc-off count (3cx = " +
     3 * eq.cx + ")");
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
// the rows themselves, driven with an empty encode pair (this leg is about the awaits,
// not the numbers): two rows on a live solver, and a visible "skipped" row -- never a
// throw and never silence -- when a resolution change retires it while a readback is
// parked, whichever of the two readbacks it is parked on.
async function legAnalyticRows(env, page) {
  const call = "() => fftAnalyticRows(solver, { fwd: () => {}, inv: () => {} })" +
    ".then(r => r, e => ['threw: ' + e.message])";
  const live = await env.run(call);
  ok(page + ": fftAnalyticRows adds both rows on a live solver",
     live.length === 2 && !live.some(r => /skipped/.test(r)),
     live.length + " rows");
  // retire the solver while readback `k` (1 or 2) is parked, then release every map that
  // is still held -- the leg must terminate and report, not park with node's queue empty
  const race = async k => {
    env.holdMaps(true);
    const pending = env.run(call);
    for (let i = 0; i < 40 && env.mapsPending() < 1; i++) await new Promise(r => setTimeout(r, 0));
    if (k > 1) { env.maps(false, 1); for (let i = 0; i < 40 && env.mapsPending() < 1; i++) await new Promise(r => setTimeout(r, 0)); }
    env.run("() => { window.__sv = solver; solver = null; }");
    while (env.mapsPending()) { env.maps(); await new Promise(r => setTimeout(r, 0)); }
    const rows = await pending;
    env.run("() => { solver = window.__sv; }");
    env.holdMaps(false);
    return rows;
  };
  const r1 = await race(1);
  ok(page + ": ... a rebuild during the FORWARD readback leaves one visible skipped row",
     r1.length === 1 && /skipped \(solver rebuilt during test\)/.test(r1[0]),
     r1.length + " rows: " + r1.join(" ").slice(0, 120));
  const r2 = await race(2);
  ok(page + ": ... and one during the ROUNDTRIP readback keeps the forward row and adds it",
     r2.length === 2 && /forward transform at/.test(r2[0]) &&
     /skipped \(solver rebuilt during test\)/.test(r2[1]),
     r2.length + " rows: " + r2.map(r => r.slice(4, 60)).join(" | "));
}

// ---------------------------------------------------------------------------
(async () => {
  console.log("(i) the ?bench flag gates the panel, and the campaigns run");
  const e2 = await legGate("rmhd2d.html");
  await legCampaigns(e2, "rmhd2d.html", false);
  const e3 = await legGate("rmhd3d.html");
  await legCampaigns(e3, "rmhd3d.html", true);
  console.log("(i.b) the spec's pipelines, extents and buffers are the step's own");
  legWiring(e2, "rmhd2d.html");
  legWiring(e3, "rmhd3d.html");
  console.log("(i.e) the grads hash cell runs the step's own gradient chain");
  await legGradsHash(e2, "rmhd2d.html");
  await legGradsHash(e3, "rmhd3d.html");
  console.log("(i.c) a campaign holds the frame loop off");
  await legLoopHold("rmhd2d.html");
  await legLoopHold("rmhd3d.html");
  console.log("(i.c2) ... including the frame already in flight when it starts");
  await legLoopRace("rmhd2d.html");
  await legLoopRace("rmhd3d.html");
  console.log("(i.d) R + 1 reps per cell, the first discarded");
  await legReps(e2, "rmhd2d.html");
  console.log("(ii) the default fftKernel emission is the captured one");
  legCapture(e2);
  legPipelines(e2, "rmhd2d.html");
  legPipelines(e3, "rmhd3d.html");
  console.log("(iii) the ladder probes");
  await legProbes(e2);
  legSeamRestore(e2);
  console.log("(iv) bytes per step against a hand count");
  legBytes(e2, "rmhd2d.html");
  legBytes(e3, "rmhd3d.html");
  console.log("(v) the analytic self-test reference");
  legAnalytic(e2, { nx: 32, ny: 32 });
  legAnalytic(e3, { nx: 16, ny: 16, nz: 8 });
  await legAnalyticRows(e2, "rmhd2d.html");
  await legAnalyticRows(e3, "rmhd3d.html");
  console.log(bad ? "bench harness: FAILED (" + bad + ")" : "bench harness: all green");
  process.exitCode = bad ? 1 : 0;
  process.exit(process.exitCode);
})();
