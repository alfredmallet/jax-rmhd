// The RENDER GATE (audit of 2026-08-12) and THE PASS ITSELF (LOOPLAT_PLAN, same day), on
// both booted pages.
//
// The claim under test is "a page that cannot have changed draws nothing", and its two
// halves are equally load-bearing: nothing is drawn when nothing moved, and something IS
// drawn on every event that moves the picture. A gate that only ever answers "no" would
// pass the first half and ship a frozen app, so every leg here that asserts a skip is
// paired with one that asserts a draw.
//
// It drives `renderCards(paused)` -- the app's own per-frame display step, split out of
// loop() so it can be called without a rAF (the stub's requestAnimationFrame is a no-op).
// The counting is done by wrapping `Solver.render`, so what is counted is chains actually
// encoded, not the gate agreeing with itself.
//
// Run: node devtools/checkidle.js [dir]
"use strict";
const DIR = process.argv[2] || __dirname + "/..";
let pass = 0, fail = 0;
function ok(name, cond, note) {
  if (cond) { pass++; console.log("  PASS  " + name + (note ? "   [" + note + "]" : "")); }
  else { fail++; console.log("  FAIL  " + name + (note ? "   [" + note + "]" : "")); }
}

// count the chains a call to renderCards actually encodes
const COUNT = `function(paused){
  const sv = solver, n0 = sv.__rn || 0;
  if (!sv.__wrapped) {
    const o = sv.render.bind(sv);
    sv.render = function(ctx, ci) { sv.__rn = (sv.__rn || 0) + 1; return o(ctx, ci); };
    sv.__wrapped = 1;
  }
  const drew = renderCards(!!paused);
  return { drew: drew, encoded: (sv.__rn || 0) - n0, cards: cards.disp.length }; }`;

// recCapture calls on a frame where the card does NOT draw: it must still be called, or a
// take that started between frames would miss its first one
const frameCaps = env => env.run(`function(){
  const d = cards.disp[0];
  let caps = 0;
  const o = d.recCapture.bind(d);
  d.recCapture = () => { caps++; return o(); };
  const drew = renderCards(true);
  d.recCapture = o;
  return drew === 0 ? caps : -1; }`);

async function run(page) {
  console.log("\n=== " + page + " ===");
  const env = require("./stubenv")(DIR, page, "");
  // boot() is async (it awaits initGPU), and the stub resolves promises immediately, so
  // the solver appears once the microtask queue has drained -- not on the constructor's
  // return
  for (let i = 0; i < 200 && !env.run("function(){ return !!solver; }"); i++)
    await new Promise(r => setTimeout(r, 0));
  if (!env.run("function(){ return !!solver; }")) { ok(page + ": solver came up", false); return; }
  const frame = paused => env.run(COUNT, paused);

  // ---- 1. the gate closes -------------------------------------------------
  // Boot leaves every card dirty (each one's apply() ran), so the FIRST frame draws them
  // all; a second frame over the same state must draw nothing at all.
  const f1 = frame(true);
  ok("the first frame after boot draws every display card",
     f1.drew === f1.cards && f1.encoded === f1.cards,
     f1.drew + " of " + f1.cards + " cards, " + f1.encoded + " chains encoded");
  const f2 = frame(true), f3 = frame(true);
  ok("a paused page then draws NOTHING, frame after frame",
     f2.drew === 0 && f2.encoded === 0 && f3.drew === 0 && f3.encoded === 0,
     "two further frames: " + f2.encoded + " and " + f3.encoded + " chains");

  // ---- 2. ... and opens for every event that moves the picture -------------
  const after = (what, run_) => { run_(); const f = frame(true); return f; };
  // through the WIRED handler, not through apply(): a leg that calls the invalidator
  // itself would pass with the wiring cut
  const ctl = after("a control", () => env.run(
    `function(){ const d = cards.disp[0];
                 d.selCmap.value = String((d.cmap() + 1) % 4); d.selCmap.onchange(); }`));
  ok("a card's own control, fired through its handler, re-opens it",
     ctl.drew === 1, ctl.drew + " card drew");

  // ... and the step leg invalidates NOTHING by hand. needsRender asks stateMark() for
  // itself, so this is a test of the gate and not of its own bookkeeping: cut the mark
  // out of needsRender and this fails.
  const stepped = after("a step", () => env.run("function(){ solver.step(1); }"));
  ok("a step alone re-opens every card, with nothing marked by hand",
     stepped.drew === stepped.cards, stepped.drew + " of " + stepped.cards);

  const reset = after("chartsReset", () => env.run("function(){ chartsReset(); }"));
  ok("an IC change (chartsReset) re-opens every card", reset.drew === reset.cards,
     reset.drew + " of " + reset.cards);

  const synced = after("cardsSync", () => env.run("function(){ cardsSync(); }"));
  ok("cardsSync re-opens every card", synced.drew === synced.cards,
     synced.drew + " of " + synced.cards);

  // a new card has never drawn: it must not inherit the quiet page's clean slate
  const added = env.run(`function(){ addDisplayCard({ sel: 4 }); cardsSync(); return 1; }`);
  const grown = frame(true);
  ok("a card ADDED to a quiet page draws on its first frame",
     !!added && grown.drew === grown.cards && grown.cards === f1.cards + 1,
     grown.drew + " of " + grown.cards);
  frame(true);
  // ... and NOW there are two, so "one card's control re-opens only that card" is a
  // statement with content (it was asserted above while there was one card on both pages,
  // where it could not fail -- adversarial review, 2026-08-12)
  const one = after("one card's control", () => env.run(
    `function(){ const d = cards.disp[1];
                 d.selCmap.value = String((d.cmap() + 1) % 4); d.selCmap.onchange(); }`));
  ok("a control on ONE of two cards re-opens only that one",
     one.cards === 2 && one.drew === 1, one.drew + " of " + one.cards + " drew");

  // closing one goes through cardsSync, which re-applies the survivors -- so let the page
  // go quiet again before the next leg asks what an unchanged frame draws
  env.run(`function(){ cardClose(cards.disp[cards.disp.length - 1]); }`);
  frame(true);

  // ---- 3. a live take overrides the gate ----------------------------------
  // The recorder reads the texture THIS render produced, so a take must keep the frames
  // coming even over a state that is not moving -- the one case where drawing a picture
  // nobody can tell apart from the last one is the correct thing to do.
  const quiet = frame(true);
  // driven all the way through renderCards, not stopped at needsRender: what the RECRAF
  // invariant needs is a FRAME, and it needs recCapture called on every card whether or
  // not that card drew
  const take = leg => env.run(`function(leg){
    const d = cards.disp[0];
    let caps = 0;
    const o = d.recCapture.bind(d);
    d.recCapture = () => { caps++; return o(); };
    d.recorder[leg] = { done: true };      // live enough for the gate, inert for the encoder
    const drew = renderCards(true);
    d.recorder[leg] = null; d.recCapture = o;
    return { drew: drew, caps: caps }; }`, leg);
  const wc = take("wc"), mr = take("rec");
  ok("a live WebCodecs take forces a draw over an unchanged state",
     quiet.drew === 0 && wc.drew === 1,
     "quiet page draws " + quiet.drew + ", with a take " + wc.drew);
  ok("  ... and so does a live MediaRecorder take", mr.drew === 1, mr.drew + " drew");
  ok("  ... and recCapture is called on the card every frame, drawn or not",
     wc.caps === 1 && mr.caps === 1 && frameCaps(env) === 1,
     "take " + wc.caps + ", quiet " + frameCaps(env));

  // ---- 4. the editor owns the screen --------------------------------------
  const edit = env.run(`function(){ icDraw.on = true; const n = renderCards(true);
                                    icDraw.on = false; return n; }`);
  ok("the IC editor's view draws no card at all (detached canvases)", edit === 0,
     edit + " cards drew");

  // ---- 5. contour ranges are SETTLED on a paused frame, relaxed on a live one
  // contSettle writes a zero range, which contLevel reads as "no history" and answers by
  // taking the measured max outright. Without it a pause would freeze the spacing wherever
  // the 0.05-per-frame relaxation had got to. It must fire only for ACTIVE sets, and only
  // while paused.
  const cont = env.run(`function(){
    const d = cards.disp[0];
    d.selCont.value = String(DISP_PSI); d.apply();
    const D = solver.chain(d.ci), seen = { paused: [], running: [] };
    const o = solver.device.queue.writeBuffer.bind(solver.device.queue);
    for (const mode of ["paused", "running"]) {
      solver.device.queue.writeBuffer = (b, off, data) => {
        for (let i = 0; i < CONT_SETS; i++)
          if (b === D.buf.contB[i] && off === 0) seen[mode].push(i);
        return o(b, off, data);
      };
      d.dirty = true;
      renderCards(mode === "paused");
    }
    solver.device.queue.writeBuffer = o;
    return { paused: seen.paused, running: seen.running,
             active: D.cont.map(x => !!x) }; }`);
  const nActive = cont.active.filter(Boolean).length;
  ok("a paused frame settles the range of every ACTIVE contour set",
     cont.paused.length === nActive && nActive > 0,
     "set(s) " + cont.paused.join(",") + " of " + nActive + " active");
  ok("  ... and a running frame settles none of them (the relaxation is the flicker damper)",
     cont.running.length === 0, cont.running.length + " writes while running");

  // ---- 5b. leaving the IC editor ------------------------------------------
  // Save and cancel change no state at all, so nothing else would re-open the cards -- and
  // they have been detached for the length of the edit. Whether a WebGPU canvas keeps its
  // last presented image across display:none is an engine's business, so the cards are
  // marked instead of the question being asked (adversarial review, 2026-08-12).
  for (const mode of ["save", "cancel", "run"]) {
    frame(true);
    const n = after("editor " + mode, () => env.run(
      `function(m){ icEditEnter(); icEditLeave(m); }`, mode));
    ok("leaving the IC editor with " + mode + " re-opens every card",
       n.drew === n.cards, n.drew + " of " + n.cards);
  }

  // ---- 5c. one more readback after the last frame, then quiet ---------------
  // The change's user-visible promise is that a paused page shows the state you paused on,
  // not the one before it. For the two per-card readbacks that means EXACTLY one gather
  // after the last frame drawn -- the seq markers, not the wall clock, have to be what
  // ends it, or a slow frame would race the throttle and lose the last one.
  const seqs = env.run(`function(){
    const d = cards.disp[0];
    if (d.selZSrc) d.selZSrc.value = "manual";     // 3D boots on the volume: no plane
    d.selField.value = "4";                        // |u|, a vector mode
    d.cbArrow.checked = true; d.apply();
    renderCards(true);                       // the frame whose gather we must pick up
    const before = { rs: d.renderSeq, as: d.arrowSeq };
    d.arrowAt = 0; d.barAt = 0;              // throttle wide open, as after a pause
    const first = d.showArrows() && d.arrowSeq !== d.renderSeq;
    d.arrowSeq = d.renderSeq;                // ... the loop's own bookkeeping
    d.arrowAt = 0;
    const second = d.showArrows() && d.arrowSeq !== d.renderSeq;
    const quiet = renderCards(true);
    const third = d.arrowSeq !== d.renderSeq;
    return { on: d.showArrows(), first: first, second: second, third: third,
             quiet: quiet, rs: before.rs }; }`);
  ok("the arrow gather fires once after the last frame drawn",
     seqs.on && seqs.first === true, "showArrows " + seqs.on);
  ok("  ... and not a second time, however open the throttle is",
     seqs.second === false && seqs.third === false && seqs.quiet === 0,
     "quiet frame drew " + seqs.quiet);

  // ---- 5d. the solver-level readbacks really are suppressed -----------------
  // Asserted as the loop asks it -- the marker against stateMark() -- and then that
  // cardsThrottleReset re-permits, which is what feeds a chart card added while paused.
  const thr = env.run(`function(){
    const mk = stateMark();
    cardsThrottle.spec = 0; cardsThrottle.specAt = mk;
    const blocked = !(cardsThrottle.specAt !== mk);
    solver.step(1);
    const afterStep = cardsThrottle.specAt !== stateMark();
    cardsThrottle.specAt = stateMark();
    const quietAgain = !(cardsThrottle.specAt !== stateMark());
    cardsThrottleReset();
    const permitted = cardsThrottle.specAt !== stateMark();
    return { blocked: blocked, afterStep: afterStep,
             quietAgain: quietAgain, permitted: permitted }; }`);
  ok("a served spectrum readback is not taken again over the same state",
     thr.blocked && thr.quietAgain);
  ok("  ... is taken again after a step, and after cardsThrottleReset",
     thr.afterStep && thr.permitted);

  // ---- 5e. the stats cache ------------------------------------------------
  const st = env.run(`function(){
    const a = statsCache.at, mk = stateMark();
    return { keyed: a === mk || a === null, mk: mk }; }`);
  ok("the stats cache is keyed on the state mark, not on the run flag",
     st.keyed !== undefined);

  // ---- 5f. the 3D hooks: a CONSUMER arriving over a still state -------------
  // The bug this leg exists for (adversarial review, 2026-08-12): `fieldLineHook` gated on
  // "the state has not moved AND something is cached", so once ANY consumer had been
  // served, a SECOND one arriving over a still state was locked out for good -- and both
  // pages boot paused, so that is the ordinary way a visitor opens the k∥ chart. The march
  // has three independent consumers and they come and go on their own; the gate now asks
  // whether anybody is asking who has nothing (`flStale`).
  if (env.is3d) {
    const hook = () => env.run(`async function(){ flAt = 0; await fieldLineHook(solver); }`);
    // consumer 1: a card in the lines view. No chart, so the march takes no samples.
    env.run(`function(){ const d = cards.disp[0];
      d.selZSrc.value = "lines"; d.apply(); }`);
    await hook();
    const one = env.run(`function(){ return { lines: !!cards.disp[0].lines, par: !!flPar }; }`);
    ok("3D: a lines-view card gets its polylines", one.lines && !one.par,
       "lines " + one.lines + ", parFL " + one.par);

    // consumer 2, over an UNMOVED state: the k∥ (field line) spectrum leg
    env.run(`function(){ addChartCard("spectrum"); cardsSync();
      const c = cards.chart[cards.chart.length - 1];
      const sd = c.optEls.filter(x => x.__optId === "sd")[0];
      sd.value = "fl"; sd.onchange(); }`);
    await hook();
    const two = env.run(`function(){ return { chart: flChartOn(), par: !!flPar,
                                              extra: !!(specExtra() || {}).parFL }; }`);
    ok("  ... and a k∥ chart opened afterwards, with the state STILL, is fed too",
       two.chart && two.par && two.extra,
       "flChartOn " + two.chart + ", flPar " + two.par + ", specExtra.parFL " + two.extra);

    // consumer 3, still unmoved: a SECOND card entering the lines view
    env.run(`function(){ addDisplayCard({ sel: 4 }); cardsSync();
      const d = cards.disp[cards.disp.length - 1];
      d.selZSrc.value = "lines"; d.apply(); }`);
    await hook();
    const three = env.run(`function(){ return cards.disp.map(d => !!d.lines); }`);
    ok("  ... and so is a second lines card added after it",
       three.length > 1 && three.every(Boolean), JSON.stringify(three));

    // ... and with everybody fed and nothing moving, it stops marching
    const quietHook = await env.run(`async function(){
      let n = 0; const o = solver.readFieldLines.bind(solver);
      solver.readFieldLines = async w => { n++; return o(w); };
      for (let i = 0; i < 3; i++) { flAt = 0; await fieldLineHook(solver); }
      solver.readFieldLines = o; return n; }`);
    ok("  ... then, everybody fed and the state still, it marches no more",
       quietHook === 0, quietHook + " marches over three hook calls");
    env.run(`function(){ while (cards.disp.length > 1)
                           cardClose(cards.disp[cards.disp.length - 1]);
                         for (const c of cards.chart.slice()) if (c.type() === "spectrum")
                           cardClose(c);
                         cards.disp[0].selZSrc.value = "manual"; cards.disp[0].apply(); }`);
    frame(true);
  }

  // ---- 6. the readback markers -------------------------------------------
  // The throttle clocks gained a "which state was this last served for" marker. Two
  // claims: the marker moves when the state does, and cardsThrottleReset clears it, which
  // is what lets a chart card added to a PAUSED page still be filled.
  const mk = env.run(`function(){
    const a = stateMark();
    solver.step(1);
    const b = stateMark();
    chartsReset();                                   // an IC-style jump, no step
    const c = stateMark();
    cardsThrottleReset();
    return { a: a, b: b, c: c,
             cleared: cardsThrottle.specAt === null && cardsThrottle.cutAt === null }; }`);
  ok("the state marker moves on a step", mk.a !== mk.b, mk.a + " -> " + mk.b);
  ok("  ... and on a state JUMP that takes no step (an IC upload resets nsteps to 0)",
     mk.b !== mk.c, mk.b + " -> " + mk.c);
  ok("cardsThrottleReset forgets the marker, so a card added while paused is still fed",
     mk.cleared === true);

  // =========================================================================
  // 7. THE PASS ITSELF (LOOPLAT_PLAN, 2026-08-12)
  // =========================================================================
  // Same discipline as renderCards above: `loopPass(dtPass)` is the app's own frame-loop
  // body, split out of loop() so it can be DRIVEN (the stub's requestAnimationFrame is a
  // no-op) and so the pacing can be fed injected periods instead of the wall clock. Every
  // leg here drives the real function; none re-implements it.
  //
  // The claim is "the pass does not wait for the GPU". Its two halves are again equally
  // load-bearing: nothing is awaited that would stall the pass, AND the numbers still
  // arrive, exactly once each, attached to the state they were read from.

  // ---- 7a. no unconditional drain -----------------------------------------
  // `await device.queue.onSubmittedWorkDone()` used to stand on every pass, paused ones
  // included. It survives only as the in-flight bound's backpressure, which is armed by
  // the STEP batch -- so a paused pass must take none at all, and a running pass exactly
  // one. A reinstated per-pass drain fails the first half; arming it per display card
  // fails the second.
  const drains = await env.run(`async function(){
    const q = device.queue, o = q.onSubmittedWorkDone.bind(q);
    let n = 0;
    q.onSubmittedWorkDone = () => { n++; return o(); };
    const wasRunning = running, sf = stepsPerFrame;
    inflight = 0;
    running = false; await loopPass(20); const paused = n;
    n = 0; inflight = 0;
    running = true;  await loopPass(20); const live = n;
    running = wasRunning; stepsPerFrame = sf; inflight = 0;
    q.onSubmittedWorkDone = o;
    return { paused: paused, live: live }; }`);
  ok("a PAUSED pass drains the queue not at all", drains.paused === 0,
     drains.paused + " drains");
  ok("  ... and a running pass exactly once, as the in-flight bound's backpressure",
     drains.live === 1, drains.live + " drains");

  // ---- 7b. the pass does not await the stats readback ----------------------
  // The sharpest form of the claim: a readStats that NEVER resolves must not stall the
  // pass, and the recorder must still be fed off it. (Restore `await` in front of
  // solver.readStats() and this leg hangs -- which is why it is raced against a timer
  // rather than simply awaited.)
  env.run(`function(){
    const sv = solver, d = cards.disp[0];
    sv.__realStats = sv.readStats.bind(sv);
    sv.__realCap = d.recCapture.bind(d);
    sv.__caps = 0;
    sv.readStats = () => new Promise(() => {});      // never resolves, ever
    d.recCapture = () => { sv.__caps++; return sv.__realCap(); };
    statsReset(); }`);
  const stalled = await Promise.race([
    env.run(`async function(){ await loopPass(20); return "returned"; }`),
    new Promise(r => setTimeout(() => r("HUNG"), 1000))
  ]);
  const stallState = env.run(`function(){
    const sv = solver, d = cards.disp[0], caps = sv.__caps, held = statsCache.s;
    sv.readStats = sv.__realStats; d.recCapture = sv.__realCap;
    statsReset();
    return { caps: caps, held: held === null, busy: statsCache.busy }; }`);
  ok("a pass whose readStats NEVER resolves still returns", stalled === "returned",
     String(stalled));
  ok("  ... and still feeds the recorder off that pass (the whole point of the change)",
     stallState.caps === 1, stallState.caps + " recCapture calls");
  ok("  ... and renders no numbers it does not have", stallState.held === true);

  // ---- 7c. a late arrival for a superseded state is DISCARDED --------------
  // Nothing awaits the read now, so two can be outstanding across a reset and they can
  // land in either order. The `(nsteps, stateSeq)` key answers "has this state been
  // served"; it cannot answer "is this value the newest", so the cache carries a
  // monotonic id as well. Resolving the OLDER read second must not roll the readout back.
  const late = await env.run(`async function(){
    const sv = solver, orig = sv.readStats.bind(sv), gate = [];
    const mk = t => { const a = new Float32Array(12);
                      a[0] = 1e-3; a[1] = t; a[2] = 1; a[3] = 1; return a; };
    sv.readStats = () => new Promise(res => gate.push(res));
    statsReset();
    await loopPass(20);                    // kicks read A over this state
    solver.step(1); statsReset();          // the state moves; the reset frees the busy flag
    await loopPass(20);                    // kicks read B over the new one
    const outstanding = gate.length;
    gate[1](mk(2));                        // the NEWER read lands first ...
    for (let i = 0; i < 5; i++) await Promise.resolve();
    const afterNew = statsCache.s ? statsCache.s[1] : null;
    gate[0](mk(1));                        // ... and the older one crawls in behind it
    for (let i = 0; i < 5; i++) await Promise.resolve();
    const afterOld = statsCache.s ? statsCache.s[1] : null;
    sv.readStats = orig; statsReset();
    return { outstanding: outstanding, afterNew: afterNew, afterOld: afterOld }; }`);
  ok("two stats reads really can be outstanding at once", late.outstanding === 2,
     late.outstanding + " in flight");
  ok("a late stats arrival for a superseded state is discarded, not rendered",
     late.afterNew === 2 && late.afterOld === 2,
     "newest " + late.afterNew + ", then the stale one -> " + late.afterOld);

  // ... and the other half of the same guard, which no id can do: a read whose SOLVER was
  // rebuilt under it is retired outright, whatever its id.
  const retired = await env.run(`async function(){
    const sv = solver, orig = sv.readStats.bind(sv), gate = [];
    const mk = t => { const a = new Float32Array(12);
                      a[0] = 1e-3; a[1] = t; a[2] = 1; a[3] = 1; return a; };
    sv.readStats = () => new Promise(res => gate.push(res));
    statsReset();
    await loopPass(20);
    solver = null;                          // the rebuild, as far as the guard can see
    gate[0](mk(7));
    for (let i = 0; i < 5; i++) await Promise.resolve();
    const held = statsCache.s;
    solver = sv; sv.readStats = orig; statsReset();
    return held === null; }`);
  ok("  ... and a read whose solver was retired under it is discarded too",
     retired === true);

  // ---- 7d. one energy sample per distinct t, with the lag ------------------
  // The trace is pushed from a value that is now one pass old and may be re-read on
  // several passes before a new one lands, so the `s[1] > last` guard is what stands
  // between the chart and a run of duplicate samples.
  const trace = await env.run(`async function(){
    const sv = solver, orig = sv.readStats.bind(sv);
    let k = 0;
    sv.readStats = async () => { k++; const a = new Float32Array(12);
                                 a[0] = 1e-3; a[1] = k; a[2] = 1; a[3] = 1; return a; };
    chartsReset();                              // clears hist AND the held stats
    const wasRunning = running, sf = stepsPerFrame;
    running = true; inflight = 0;
    for (let i = 0; i < 8; i++) await loopPass(25);
    running = wasRunning; stepsPerFrame = sf; inflight = 0;
    const t = Array.from(hist.t);           // BEFORE the cleanup: chartsReset clears it
    sv.readStats = orig; statsReset(); chartsReset();
    return { t: t, reads: k }; }`);
  const mono = trace.t.every((v, i) => i === 0 || v > trace.t[i - 1]);
  ok("the energy trace takes one sample per distinct t and no duplicates",
     trace.t.length >= 3 && mono && new Set(trace.t).size === trace.t.length,
     trace.t.length + " samples from " + trace.reads + " reads: [" + trace.t.join(",") + "]");
  // ... and the LAG is real, stated as arithmetic: one read is kicked per pass and the
  // value it brings is consumed by the NEXT one, so after N passes exactly one read is
  // still out and the trace is one sample short of the reads. Restore the `await` in front
  // of solver.readStats() and this becomes N == N, which is the point of the leg.
  ok("  ... exactly one readback behind, which is what 'one frame late' means",
     trace.t.length === trace.reads - 1,
     trace.t.length + " samples, " + trace.reads + " reads");

  // ---- 7e. the controller, on INJECTED pass periods ------------------------
  // Driven as a pure function of (period, take live), so this is arithmetic and not a
  // race against the machine the checker happens to run on.
  const ctl2 = env.run(`function(){
    const sf = stepsPerFrame, fl = pace.floor, r = {};
    // BOTH edges are relative to the device's period floor, so a leg that did not pin it
    // would be asserting against whatever the legs above happened to leave behind. 16.7 ms
    // is a 60 Hz panel keeping up: lo = max(18, 20.8) = 20.8, hi = max(34, 29.2) = 34, and
    // with a take max(30, 16.7) = 30.
    pace.floor = 1000 / 60;
    const sweep = (start, dt, taking, n) => {
      stepsPerFrame = start; const out = [];
      for (let i = 0; i < n; i++) out.push(paceControl(dt, !!taking));
      return out;
    };
    r.up   = sweep(4, 10, false, 3);       // comfortably inside a frame: take more steps
    r.down = sweep(4, 40, false, 3);       // over the ceiling: take fewer
    r.hold = sweep(4, 25, false, 3);       // inside the band: hold
    r.none = sweep(4, 0,  false, 1);       // no measurement: hold
    r.top  = sweep(64, 5, false, 1);       // the 64 clamp
    r.bot  = sweep(1, 500, false, 1);      // the 1 clamp
    r.cut  = sweep(32, 68, false, 1);      // the phone's own number: proportional cut
    r.noTake = sweep(8, 32, false, 1);     // 32 ms is fine for the DISPLAY ...
    r.take   = sweep(8, 32, true,  1);     // ... and not for a 33.3 ms capture slot
    r.slot = 1000 / REC_FPS;
    r.hiRec = PASS_HI_REC;
    // A SLOW device, where the bare pass already overruns a vsync: 33.3 ms floor, so the
    // panel's cheapest period is two vsyncs. An absolute 18 ms raise edge is unreachable
    // there and pins stepsPerFrame at 1 for the life of the session; relative edges (lo =
    // max(18, 41.6) = 41.6) read 33.3 ms as "still as cheap as this device gets" and raise.
    pace.floor = 100 / 3;
    r.slowRise = sweep(1, 100 / 3, false, 3);
    r.slowFall = sweep(8, 70, false, 1);      // ... and a slipped vsync still cuts
    // ... and with a take live on that same device the ceiling comes down ONTO the floor
    // rather than under it: chasing 30 ms it cannot reach would hold the pass at 1 step and
    // buy no extra slots, so the band closes at the floor and the pass is left alone.
    r.slowTakeHold = sweep(4, 100 / 3, true, 2);
    r.slowTakeCut  = sweep(8, 50, true, 1);   // a whole vsync over the slot: still cuts
    pace.floor = fl;
    stepsPerFrame = sf;
    return r; }`);
  ok("stepsPerFrame RISES while passes are short",
     JSON.stringify(ctl2.up) === "[5,6,7]", JSON.stringify(ctl2.up));
  ok("  ... FALLS while they are long, and holds inside the band",
     JSON.stringify(ctl2.down) === "[3,2,1]" && JSON.stringify(ctl2.hold) === "[4,4,4]",
     JSON.stringify(ctl2.down) + " / " + JSON.stringify(ctl2.hold));
  ok("  ... holds on a pass with no measurement (a tab returning from the background)",
     JSON.stringify(ctl2.none) === "[4]", JSON.stringify(ctl2.none));
  ok("  ... and stays inside the 1..64 clamp at both ends",
     ctl2.top[0] === 64 && ctl2.bot[0] === 1, ctl2.top[0] + " / " + ctl2.bot[0]);
  // the runaway the plain -1 could not treat: 32 steps at a 68 ms pass must not need 31
  // more 68 ms passes to come back
  ok("  ... and cuts PROPORTIONALLY out of a bad overshoot (32 steps @ 68 ms -> 16)",
     ctl2.cut[0] === 16, "-> " + ctl2.cut[0]);
  ok("with a take live the controller targets a period inside one capture slot",
     ctl2.noTake[0] === 8 && ctl2.take[0] < 8,
     "32 ms: no take -> " + ctl2.noTake[0] + ", take -> " + ctl2.take[0]);
  ok("  ... and that ceiling really is under a slot (" + ctl2.slot.toFixed(1) + " ms)",
     ctl2.hiRec < ctl2.slot, "PASS_HI_REC = " + ctl2.hiRec);
  // the latch the adversarial review found: absolute edges make the raise branch
  // unreachable on any device whose bare pass overruns a vsync, and stepsPerFrame then
  // never leaves 1 for the life of the session. Pin PASS_LO back as the only raise edge
  // (drop the `1.25 * fl` term in paceControl) and this leg reads [1,1,1].
  ok("a device whose period FLOOR is over PASS_LO can still raise (no absolute-edge latch)",
     JSON.stringify(ctl2.slowRise) === "[2,3,4]", JSON.stringify(ctl2.slowRise));
  ok("  ... and a slipped vsync on that device still cuts",
     ctl2.slowFall[0] < 8, "8 -> " + ctl2.slowFall[0]);
  ok("  ... and a take there closes the band ON the floor rather than under it",
     JSON.stringify(ctl2.slowTakeHold) === "[4,4]" && ctl2.slowTakeCut[0] < 8,
     "at the floor " + JSON.stringify(ctl2.slowTakeHold) +
     ", a vsync over it 8 -> " + ctl2.slowTakeCut[0]);

  // ... and the take is read off the CARDS, not off a flag the checker sets: a leg that
  // called paceControl(dt, true) by hand would pass with the wiring cut.
  const taking = env.run(`function(){
    const d = cards.disp[0], before = recTaking();
    d.recorder.wc = { done: true };
    const during = recTaking();
    d.recorder.wc = null;
    d.recorder.rec = { done: true };
    const mr = recTaking();
    d.recorder.rec = null;
    return { before: before, during: during, mr: mr, after: recTaking() }; }`);
  ok("  ... and 'a take is live' is read off the cards, on either recording leg",
     taking.before === false && taking.during === true && taking.mr === true &&
     taking.after === false);

  // ---- 7f. the in-flight bound --------------------------------------------
  // Without it the pass would queue step batches as fast as the CPU can encode them and
  // the latency this change removes would come back as a submission backlog. It must bound
  // the STEPS and nothing else: a saturated pass still renders and still captures, which
  // is what keeps a recording at full rate on a GPU-bound device.
  const bound = await env.run(`async function(){
    const d = cards.disp[0], o = d.recCapture.bind(d);
    let caps = 0;
    d.recCapture = () => { caps++; return o(); };
    const wasRunning = running, sf = stepsPerFrame;
    running = true;
    inflight = INFLIGHT_MAX;                      // saturated
    const n0 = solver.nsteps;
    await loopPass(20);
    const sat = { stepped: solver.nsteps - n0, caps: caps };
    caps = 0; inflight = 0;
    const n1 = solver.nsteps;
    await loopPass(20);
    for (let i = 0; i < 5; i++) await Promise.resolve();
    const free = { stepped: solver.nsteps - n1, caps: caps, left: inflight };
    d.recCapture = o; running = wasRunning; stepsPerFrame = sf; inflight = 0;
    return { sat: sat, free: free, max: INFLIGHT_MAX }; }`);
  ok("with the in-flight bound saturated a pass steps NOTHING",
     bound.sat.stepped === 0, bound.sat.stepped + " steps at " + bound.max + " in flight");
  ok("  ... and still captures a frame off that pass",
     bound.sat.caps === 1, bound.sat.caps + " recCapture calls");
  ok("  ... while a pass under the bound steps and captures as usual",
     bound.free.stepped > 0 && bound.free.caps === 1,
     bound.free.stepped + " steps, " + bound.free.caps + " captures");
  ok("  ... and the bound is RELEASED when the submission retires",
     bound.free.left === 0, bound.free.left + " left in flight");

  // ... and the half of the bound that is easy to get backwards (adversarial review): a
  // skipped pass is SHORT because it dropped the work, so handing that period to the
  // controller reads the backpressure as headroom. Eighty saturated passes drove
  // stepsPerFrame 1 -> 64 before this was fixed, so the first batch submitted when the
  // queue freed was the largest one possible.
  const starve = await env.run(`async function(){
    const wasRunning = running, sf = stepsPerFrame, fl = pace.floor;
    running = true; stepsPerFrame = 1; pace.floor = 1000 / 60;
    inflight = INFLIGHT_MAX;
    const n0 = solver.nsteps;
    for (let i = 0; i < 40; i++) { inflight = INFLIGHT_MAX; await loopPass(10); }
    const held = stepsPerFrame, stepped = solver.nsteps - n0;
    // ... and the very same period on a pass that DID step must still raise, or the leg
    // would pass with the controller simply switched off
    inflight = 0; stepsPerFrame = 1;
    await loopPass(10);
    const rose = stepsPerFrame;
    running = wasRunning; stepsPerFrame = sf; pace.floor = fl; inflight = 0;
    return { held: held, stepped: stepped, rose: rose }; }`);
  ok("a pass the bound made SKIP does not vote for more steps",
     starve.held === 1 && starve.stepped === 0,
     "stepsPerFrame " + starve.held + " after 40 skipped passes, " +
     starve.stepped + " steps taken");
  ok("  ... while the same period on a pass that DID step raises as it should",
     starve.rose === 2, "-> " + starve.rose);

  // ---- 7f2. the capture is submitted AHEAD of the step batch ---------------
  // Queue order is submission order. With the steps first, a take's copyTextureToBuffer sat
  // behind this pass's `stepsPerFrame` steps plus every earlier batch still in flight --
  // ~25-39 steps on Alfred's devices -- and its map came back 52-65 ms later against a
  // 33.3 ms slot, saturating the staging pool and dropping slots for want of a buffer
  // (on-device reading, 2026-08-12). Render-and-capture first puts the copy at the head of
  // the pass. Swap the two blocks back in loopPass and this leg reads "step,...,capture".
  const order = await env.run(`async function(){
    const d = cards.disp[0], oc = d.recCapture.bind(d), os = solver.step.bind(solver);
    const seq = [];
    d.recCapture = () => { seq.push("capture"); return oc(); };
    solver.step = ce => { seq.push("step"); return os(ce); };
    const wasRunning = running, sf = stepsPerFrame;
    running = true; stepsPerFrame = 3; inflight = 0;
    await loopPass(20);
    d.recCapture = oc; solver.step = os;
    running = wasRunning; stepsPerFrame = sf; inflight = 0;
    return seq; }`);
  ok("the recorder's frame is taken BEFORE the pass's steps are submitted",
     order[0] === "capture" && order.slice(1).every(x => x === "step") &&
     order.length === 4, order.join(","));

  // ---- 7g. the pass period is a measurement, or it is nothing --------------
  // paceFeed is the only clock the controller has. A first pass has no interval, and a tab
  // returning from the background has one that means nothing at all -- feeding either to
  // the controller would move stepsPerFrame on no evidence.
  const feed = env.run(`function(){
    const at = pace.at, dt = pace.dt, fl = pace.floor;
    pace.at = 0; pace.dt = 0; pace.floor = 0;
    const first = paceFeed(1000);                   // no previous pass: not a measurement
    const second = paceFeed(1020);                  // 20 ms, and the EMA seeds on it
    const third = paceFeed(1040);
    const floor0 = pace.floor;
    // A LONG VISIBLE pass. It is the one the controller most needs to see -- discarding it
    // by length was a latch with no exit, since the discard held stepsPerFrame high and
    // that kept the pass long (adversarial review, BLOCKING).
    const slow = paceFeed(1040 + 600);
    const slowFloor = pace.floor;
    // ... and the CAUSE-based discard: the tab says it was hidden, so the interval spanning
    // the stint is thrown away whatever its length.
    document.dispatchEvent ? document.dispatchEvent(new Event("visibilitychange"))
                           : (pace.at = 0, pace.dt = 0);
    const parked = paceFeed(1040 + 600 + 5 * PASS_MAX);
    const reseed = paceFeed(1040 + 600 + 5 * PASS_MAX + 20);
    // the floor tracks DOWN at once and up only slowly
    pace.floor = 40; paceFeed(pace.at + 12); const down = pace.floor;
    pace.floor = 10; paceFeed(pace.at + 40); const up = pace.floor;
    pace.at = at; pace.dt = dt; pace.floor = fl;
    return { first: first, second: second, third: third, floor0: floor0,
             slow: slow, slowFloor: slowFloor, parked: parked, reseed: reseed,
             down: down, up: up }; }`);
  ok("the first pass of a session yields no pass period at all", feed.first === 0);
  ok("  ... the next seeds the average, and it tracks", feed.second === 20 &&
     Math.abs(feed.third - 20) < 1e-9, feed.second + " -> " + feed.third);
  ok("  ... and seeds the device's period floor with it",
     Math.abs(feed.floor0 - 20) < 1e-9, "floor " + feed.floor0);
  ok("a long VISIBLE pass is measured, not discarded (clamped at PASS_MAX)",
     feed.slow > 20 && feed.slow <= 500,
     "600 ms pass -> smoothed " + feed.slow.toFixed(1) + " ms");
  ok("  ... and it does not drag the floor up with it in one go",
     feed.slowFloor < 30, "floor " + feed.slowFloor.toFixed(2));
  ok("  ... while a HIDDEN stint is discarded by cause and re-seeds after",
     feed.parked === 0 && Math.abs(feed.reseed - 20) < 1e-9,
     "parked " + feed.parked + ", reseed " + feed.reseed);
  ok("the floor drops to a cheaper pass at once and creeps up to a dearer one",
     feed.down === 12 && feed.up > 10 && feed.up < 11,
     "40 -> " + feed.down + " instantly; 10 -> " + feed.up.toFixed(2) + " on a 40 ms pass");

  // ---- 7g2. a dropped slot says WHY -----------------------------------------
  // One `drop` counter with six ways to reach it is one way too few to act on: the
  // on-device reading had it rising on three devices at once and no way to tell a late loop
  // pass from a saturated staging pool from encoder backpressure, which have three
  // different fixes. Driven on the real recCapture, one cause at a time.
  const why = env.run(`function(){
    const d = cards.disp[0], r = d.recorder;
    const mk = back => ({ done: false, drop: 0, why: {}, bufOn: false, n: 0,
                          due: performance.now() - back,
                          lastRaf: performance.now(), maxGap: 0, rafN: 0, wdN: 0,
                          enc: { encodeQueueSize: 0 } });
    // (a) the loop was LATE: five whole slots went by between passes. The encode itself is
    // stubbed out -- what is under test is the accounting ahead of it, and a real encode
    // needs a live VideoEncoder this page has no reason to build.
    const slot = mk(5 * (1000 / REC_FPS));
    const oe = r.recEncodeFrame.bind(r);
    r.recEncodeFrame = () => false;
    r.wc = slot; r.recCapture(); r.wc = null;
    r.recEncodeFrame = oe;
    // (b) the ENCODER is behind. Driven at recEncodeFrame, not through recCapture: the
    // slot clock is wall time and cannot be held still, so routing through the cadence
    // would bill this drop to whatever the clock had done meanwhile. The backpressure
    // check is that function's first line, so this never reaches the encoder either.
    const enc = mk(0);
    enc.enc.encodeQueueSize = REC_QMAX + 1;
    r.wc = enc; r.recEncodeFrame(enc); r.wc = null;
    return { slot: slot.why, slotN: slot.drop, enc: enc.why, encN: enc.drop,
             pool: REC_POOL, slotMs: 1000 / REC_FPS }; }`);
  // the COUNT is wall-clock dependent (the leg cannot hold the slot clock still), so what
  // is asserted is the attribution: every drop this path made is billed to `slot` and to
  // nothing else.
  ok("a slot lost to a late loop pass is counted as 'slot'",
     why.slotN > 0 && why.slot.slot === why.slotN && !why.slot.enc,
     why.slotN + " slots, all billed as " + JSON.stringify(why.slot));
  ok("  ... and one lost to encoder backpressure as 'enc', not as the same number",
     why.encN > 0 && why.enc.enc === why.encN && !why.enc.slot,
     JSON.stringify(why.enc));
  // ... and the pool is sized in SLOTS of readback latency, which is the quantity that
  // exhausts it. Three buffers was 100 ms against a MEASURED 52-65 ms lag plus jitter.
  ok("the capture pool covers well over the measured readback lag",
     why.pool * why.slotMs >= 150,
     why.pool + " buffers x " + why.slotMs.toFixed(1) + " ms = " +
     (why.pool * why.slotMs).toFixed(0) + " ms of latency absorbed");

  // ---- 7h. a state jump drops the held numbers ----------------------------
  // The one consumer that HOLDS a value across passes rather than re-deriving it from
  // stateMark(). Without this the pass after a rebuild would print the old solver's
  // energies and -- the part that leaves a trace on disk -- stamp a capture filename with
  // the old solver's `simT`.
  const jump = env.run(`function(){
    statsCache.s = new Float32Array(12); statsCache.at = "stale"; statsCache.busy = true;
    chartsReset();
    return { s: statsCache.s, at: statsCache.at, busy: statsCache.busy }; }`);
  ok("an IC change / rebuild drops the held stats rather than describing a dead state",
     jump.s === null && jump.at === null && jump.busy === false,
     "s " + jump.s + ", at " + jump.at + ", busy " + jump.busy);

  // ... and the half that dropping the value does NOT cover, which is the defect this
  // whole section found: an IC upload and a preset switch keep the same solver object, so
  // a read still in flight over the discarded state passes the `sv === solver` guard and
  // lands one pass later, putting the forgotten energies back in the readout and its `t`
  // back in `simT`. Retiring by id is what stops it. (Delete the `got = seq` line in
  // statsReset and this leg fails while every other one here still passes.)
  const across = await env.run(`async function(){
    const sv = solver, orig = sv.readStats.bind(sv), gate = [];
    const mk = t => { const a = new Float32Array(12);
                      a[0] = 1e-3; a[1] = t; a[2] = 1; a[3] = 1; return a; };
    sv.readStats = () => new Promise(res => gate.push(res));
    statsReset();
    await loopPass(20);                    // a read goes out over this state ...
    statsReset();                          // ... the state is thrown away under it ...
    gate[0](mk(99));                       // ... and it lands anyway, same solver
    for (let i = 0; i < 5; i++) await Promise.resolve();
    const held = statsCache.s;
    sv.readStats = orig; statsReset();
    return { held: held === null, t: held ? held[1] : null }; }`);
  ok("  ... and retires the reads still in flight over it, same solver or not",
     across.held === true, "held " + (across.t === null ? "nothing" : "t = " + across.t));

  // ... and the busy flag survives that retirement with an OWNER. Without one, the retired
  // read lands later and clears the flag its SUCCESSOR is holding, and from then on the
  // loop kicks a fresh read every pass with two or three always in flight -- three times
  // the map round trips this change removes, and the held value lagging by the whole
  // readback latency instead of by one pass (adversarial review, MAJOR). Measured as reads
  // per pass over a run, which is the consequence and not the flag.
  const owner = await env.run(`async function(){
    const sv = solver, orig = sv.readStats.bind(sv), gate = [];
    let issued = 0;
    sv.readStats = () => { issued++; return new Promise(res => gate.push(res)); };
    const mk = t => { const a = new Float32Array(12);
                      a[0] = 1e-3; a[1] = t; a[2] = 1; a[3] = 1; return a; };
    const wasRunning = running, sf = stepsPerFrame, fl = pace.floor;
    running = true; inflight = 0; pace.floor = 1000 / 60;
    statsReset();
    await loopPass(20);                    // read A goes out
    statsReset();                          // ... and is retired, freeing the slot
    await loopPass(20);                    // read B goes out and OWNS the flag
    gate[0](mk(1));                        // A lands: it must not free B's slot
    for (let i = 0; i < 5; i++) await Promise.resolve();
    const heldByB = statsCache.busy;
    const issuedAfterA = issued;
    for (let i = 0; i < 6; i++) await loopPass(20);   // B still out: no new reads
    const issuedAfter6 = issued;
    for (const g of gate) g(mk(2));
    for (let i = 0; i < 5; i++) await Promise.resolve();
    running = wasRunning; stepsPerFrame = sf; pace.floor = fl; inflight = 0;
    sv.readStats = orig; statsReset();
    return { heldByB: heldByB, issuedAfterA: issuedAfterA, issuedAfter6: issuedAfter6 }; }`);
  ok("a retired read does not free the slot its successor is holding",
     owner.heldByB === true);
  ok("  ... so six further passes over a read still in flight kick no new ones",
     owner.issuedAfter6 === owner.issuedAfterA,
     owner.issuedAfterA + " reads issued, still " + owner.issuedAfter6 + " after 6 passes");
  frame(true);
}

(async () => {
  for (const p of ["rmhd2d.html", "rmhd3d.html"]) await run(p);
  console.log("\n" + (fail ? "FAILED " + fail + "/" + (pass + fail)
                           : "PASS " + pass + "/" + pass) + " checkidle");
  process.exit(fail ? 1 : 0);
})();
