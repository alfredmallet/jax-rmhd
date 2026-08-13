// The RENDER GATE (audit of 2026-08-12), on both booted pages.
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
}

(async () => {
  for (const p of ["rmhd2d.html", "rmhd3d.html"]) await run(p);
  console.log("\n" + (fail ? "FAILED " + fail + "/" + (pass + fail)
                           : "PASS " + pass + "/" + pass) + " checkidle");
  process.exit(fail ? 1 : 0);
})();
