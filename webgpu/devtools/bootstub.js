// Stub-GPU boot test: boot a real app page on the shared stub environment
// (devtools/stubenv.js) and exercise it -- card add / remove / close for every card
// type, every chart card's Phase-H options, preset switching, the forcing controls and
// the IC editor view.  Usage: node bootstub.js <dir> <page> [demo]
"use strict";
const [dir, page, demo] = process.argv.slice(2);
const env = require("./stubenv")(dir, page, demo);
const { run, getEl, fails, fail } = env;

// one frame of the loop's per-frame work, by hand (requestAnimationFrame is a no-op)
const frame = () => run(`async function(){
  if (!solver) return;
  solver.step(1);
  for (const d of cards.disp) d.render();
  const s = await solver.readStats();
  if (frameHook) await frameHook(solver);
  histPush(solver.nsteps * 1e-3, 1e-3, 2e-3, 5e-4);
  for (const c of cards.chart) {
    if (c.type() === "energy") c.draw(null);
    else if (c.type() === "spectrum") { const sp = await solver.readSpectrum();
      c.draw({ perp: sp.perp, nb: solver.nb, fshell: solver.p.fshell, par: sp.par, parKfac: sp.parKfac }); }
    else { c.draw({ vals: await solver.readCutLine(cards.cfg.zsliceOf(c)), Ly: solver.p.Ly }); }
  }
  for (const d of cards.disp) {
    if (!d.showArrows()) { d.clearArrows(); continue; }
    d.drawArrows(await solver.readArrows(d.ci), solver.nax, solver.nay);
  }
}`);
const state = () => run(`function(){
  return { chains: solver.disp.filter(Boolean).length,
           disp: cards.disp.map(d => ({ ci: d.ci, sel: d.sel(), cmap: d.cmap(), zsrc: d.zsrc(),
                                        mode: solver.modeOf(d.ci), cube: solver.cubeOf(d.ci),
                                        zslice: solver.zsliceOf(d.ci), cap: d.cap.innerHTML,
                                        arrows: d.showArrows() })),
           charts: cards.chart.map(c => c.type()),
           preset: document.getElementById("selPreset").value,
           hint: document.getElementById("demohint").style.display,
           ic: document.getElementById("selIC").value,
           nx: solver.p.nx, nz: solver.p.nz, diss: solver.p.diss, eps: solver.p.epsP,
           steps: document.getElementById("steps").textContent };
}`);

setTimeout(async () => {
  const tag = page + (demo ? " ?demo=" + demo : " (plain)");
  try {
    if (!run("function(){ return !!solver; }")) throw new Error("boot() left no solver");
    console.log(tag + " after boot: " + JSON.stringify(state()));
    await frame();

    // --- add display cards up to the cap, and one of every chart type ---------
    run(`function(){
      addDisplayCard({ sel: 6, cmap: 1 }); addDisplayCard({ sel: 8, cmap: 2 });
      addDisplayCard({ sel: 4 });                       // one past the cap -> refused
      cardsSync();
      for (const t of ["cut", "spectrum", "energy"]) addChartCard(t);
    }`);
    let st = state();
    if (st.disp.length !== 3) fail("expected 3 display cards, got " + st.disp.length);
    if (st.charts.length < 4) fail("chart cards did not accumulate: " + st.charts.join(","));
    console.log(tag + " 3 displays + " + st.charts.length + " charts: " + JSON.stringify(st.disp));
    await frame();

    // --- Phase H: every chart-card option, drawn with real data ----------------
    // one card of each type, then walk every value of every option select, running a
    // frame after each so the option is exercised on a real draw, not just on null.
    run(`function(){ while (cards.chart.length) cardClose(cards.chart[0]);
                     for (const t of Object.keys(CHART_TYPES)) addChartCard(t);
                     cardsSync(); }`);
    await frame();
    const combos = run(`function(){
      const out = [];
      cards.chart.forEach((c, i) => c.optEls.forEach(s => s.options.forEach(o =>
        out.push([i, s.__optId, o.value, c.type()]))));
      return out;
    }`);
    if (combos.length < 4) fail("chart cards exposed no options: " + JSON.stringify(combos));
    for (const cb of combos) {
      run(`function(i, id, v){
        const s = cards.chart[i].optEls.filter(e => e.__optId === id)[0];
        s.value = v; s.onchange();
      }`, cb[0], cb[1], cb[2]);
      await frame();
    }
    console.log(tag + " chart options: " + combos.map(c => c[3] + "." + c[1] + "=" + c[2]).join(" "));
    // the cut card owns its z plane in 3D (no display card involved)
    if (page.indexOf("3d") >= 0) {
      run(`function(){ const c = cards.chart.filter(x => x.type() === "cut")[0];
                       c.selZSrc.value = "zp"; c.selZSrc.onchange(); }`);
      await frame();
      if (!run("function(){ return trackingOn(); }"))
        fail("a cut card tracking a packet did not switch the tracker on");
      run(`function(){ const c = cards.chart.filter(x => x.type() === "cut")[0];
                       c.selZSrc.value = "manual"; c.selZSrc.onchange();
                       c.rSlice.value = "5"; c.rSlice.oninput(); }`);
      await frame();
      const iz = run(`function(){ return cards.cfg.zsliceOf(cards.chart.filter(x => x.type() === "cut")[0]); }`);
      if (iz !== 5) fail("the cut card's own z slider did not take: iz " + iz);
      console.log(tag + " cut card z source: tracker + own slider -> iz " + iz);
    }
    // retype / close / re-add with an option selected
    run(`function(){ const c = cards.chart[0]; c.selType.value = "spectrum"; c.selType.onchange();
                     cardClose(cards.chart[1]); addChartCard("cut"); cardsSync(); }`);
    await frame();
    if (run("function(){ return cards.chart.some(c => !c.cx); }")) fail("a retyped chart lost its context");

    // --- every colormap, on a live card ---------------------------------------
    run(`function(){ for (let k = 0; k < CMAP_NAMES.length; k++) { cards.disp[0].selCmap.value = String(k);
                                                                    cards.disp[0].apply(); } }`);
    await frame();

    // --- cube modes (3D only): every card may show faces ----------------------
    if (page.indexOf("3d") >= 0) {
      run(`function(){ cards.disp[1].selField.value = String(CUBE_SEL0); cards.disp[1].apply(); }`);
      st = state();
      if (!st.disp[1].cube) fail("a non-first card could not enter cube-face mode");
      await frame();
      run(`function(){ cards.disp[1].selField.value = "7"; cards.disp[1].apply();
                       cards.disp[1].selZSrc.value = "zm"; cards.disp[1].apply(); }`);
      await frame();
    }

    // --- close: every chart card, then EVERY display but the last -------------
    run(`function(){ while (cards.chart.length) cardClose(cards.chart[0]); }`);
    if (state().charts.length) fail("chart cards would not close");
    await frame();
    // Phase G: card 0 is an ordinary card -- close it FIRST, and keep closing until
    // only one (any one) is left.
    run(`function(){ cardClose(cards.disp[0]); }`);
    if (state().disp.length !== 2) fail("display card 0 would not close");
    run(`function(){ while (cards.disp.length > 1) cardClose(cards.disp[0]); }`);
    st = state();
    if (st.disp.length !== 1) fail("display cards would not close: " + st.disp.length + " left");
    // ... but the LAST one must refuse, and say so on its button
    run(`function(){ cardClose(cards.disp[0]); }`);
    if (state().disp.length !== 1) fail("the last display card was closed");
    if (!run("function(){ return cards.disp[0].btnClose.disabled; }"))
      fail("the last display card's close button is not disabled");
    await frame();
    // a freed slot is reused, and the close buttons come back
    run(`function(){ addDisplayCard({ sel: 5 }); cardsSync(); }`);
    st = state();
    if (st.disp.length !== 2) fail("a card could not be re-added: " + JSON.stringify(st.disp));
    if (run("function(){ return cards.disp.some(d => d.btnClose.disabled); }"))
      fail("close buttons stayed disabled with 2 cards");
    await frame();
    console.log(tag + " after close-all-but-one / reopen: " + JSON.stringify(state().disp));

    // --- preset switching ------------------------------------------------------
    for (const k of run("function(){ return Object.keys(PRESETS); }")) {
      run(`function(k){ const s = document.getElementById("selPreset"); s.value = k; s.onchange(); }`, k);
      const s2 = state();
      console.log(tag + " preset " + k + ": " + JSON.stringify({ disp: s2.disp.map(d => d.sel + "/" + d.zsrc),
        charts: s2.charts, nx: s2.nx, nz: s2.nz, ic: s2.ic, eps: s2.eps, hint: s2.hint }));
      await frame(); await frame();
    }

    // --- eps+- and the forcing band (Phase G.5) --------------------------------
    // the lock mirrors, unlocking separates, and a band change rebuilds the solver
    run(`function(){ const s = document.getElementById("selPreset"); s.value = Object.keys(PRESETS)[0]; s.onchange(); }`);
    run(`function(){ document.getElementById("cbForce").checked = true;
                     document.getElementById("cbForce").onchange();
                     document.getElementById("rEpsP").value = "-1.5";
                     document.getElementById("rEpsP").oninput(); }`);
    let ep = run("function(){ return [solver.p.epsP, solver.p.epsM, document.getElementById('rEpsM').value]; }");
    if (Math.abs(ep[0] - ep[1]) > 1e-12) fail("locked eps did not mirror: " + JSON.stringify(ep));
    run(`function(){ document.getElementById("cbEpsLock").checked = false;
                     document.getElementById("rEpsM").value = "-0.3";
                     document.getElementById("rEpsM").oninput(); }`);
    ep = run("function(){ return [solver.p.epsP, solver.p.epsM]; }");
    if (!(ep[1] > 10 * ep[0])) fail("unlocked eps stayed tied: " + JSON.stringify(ep));
    console.log(tag + " eps+- separated: " + JSON.stringify(ep));
    const nsBefore = run("function(){ return solver.ns; }");
    run(`function(){ document.getElementById("rFmin").value = "3";
                     document.getElementById("rFmax").value = "6";
                     document.getElementById("rFmax").onchange(); }`);
    const band = run("function(){ return [solver.p.fshell[0], solver.p.fshell[1], solver.ns]; }");
    if (band[0] !== 3 || band[1] !== 6) fail("the forcing band did not reach the solver: " + JSON.stringify(band));
    if (band[2] === nsBefore) fail("the forcing shell was not rebuilt (ns unchanged)");
    console.log(tag + " forcing band -> " + JSON.stringify(band) + " (was ns " + nsBefore + ")");
    // the band handles cannot cross
    run(`function(){ document.getElementById("rFmin").value = "9";
                     document.getElementById("rFmin").onchange(); }`);
    const cross = run("function(){ return solver.p.fshell; }");
    if (!(cross[1] > cross[0])) fail("the forcing band handles crossed: " + JSON.stringify(cross));
    await frame();

    // --- amplitudes are a rescale, not a redraw (Phase G.3/G.4) ----------------
    run(`function(){ document.getElementById("selIC").value = "letters";
                     document.getElementById("selIC").onchange(); }`);
    const ampCheck = run(`function(){
      const q = solver.p, g = icDrawGrid(q);
      const z = icLetterZeta(g, q.nz > 1 ? [icGaussZ(q.nz, q.Lz, 0.3 * q.Lz, 0.05 * q.Lz),
                                            icGaussZ(q.nz, q.Lz, 0.7 * q.Lz, 0.05 * q.Lz)] : null);
      const a = icZetaFields(z.zp, z.zm, g, 1.0, 1.0);
      const b = icZetaFields(z.zp, z.zm, g, 3.0, 3.0);
      let mx = 0, mb = 0;
      for (let i = 0; i < a.phi.length; i++) {
        mx = Math.max(mx, Math.abs(3 * a.phi[i] - b.phi[i]));
        mb = Math.max(mb, Math.abs(b.phi[i]));
      }
      return [mx, mb];
    }`);
    if (!(ampCheck[1] > 0) || ampCheck[0] > 1e-6 * ampCheck[1])
      fail("amplitude is not an exact rescale: " + JSON.stringify(ampCheck));

    // --- the IC editor as its own view (Phase G.7) -----------------------------
    run(`function(){ document.getElementById("selIC").value = "custom";
                     document.getElementById("selIC").onchange(); }`);
    const ev = () => run(`function(){ return {
      on: icDraw.on,
      view: document.getElementById("editview").style.display,
      sim: document.getElementById("display").style.display,
      runDis: document.getElementById("btnRun").disabled,
      has: icDraw.has, sum: (function(){ let s = 0; for (let i = 0; i < icDraw.zp.length; i++) s += Math.abs(icDraw.zp[i]); return s; })()
    }; }`);
    // the editor view's own buttons are BUILT by common.js: drive them through the DOM
    const edBtn = t => run(`function(t){
      const head = icDraw.cv.parentNode.parentNode.children[0];
      const b = head.children.filter(c => c.kind === "button" && c.innerHTML === t)[0];
      if (!b) throw new Error("editor button not found: " + t);
      b.onclick();
    }`, t);
    run(`function(){ document.getElementById("btnEdit").onclick(); }`);
    let e0 = ev();
    if (!e0.on || e0.view !== "" || e0.sim !== "none" || !e0.runDis)
      fail("edit IC did not switch to the editor view: " + JSON.stringify(e0));
    if (run("function(){ return running; }")) fail("the editor did not pause the run");
    // paint two strokes, then cancel -> the drawing must come back unchanged
    const paint = `function(){ icDraw.down = true; icDraw.last = null;
      icDrawStroke({ clientX: 100, clientY: 150, preventDefault(){} });
      icDraw.last = null;
      icDrawStroke({ clientX: 300, clientY: 220, preventDefault(){} }); }`;
    run(paint);
    const painted = ev();
    if (!(painted.sum > 0)) fail("painting deposited nothing");
    edBtn("cancel");
    let e1 = ev();
    if (e1.on || e1.view !== "none" || e1.sim !== "" || e1.runDis)
      fail("cancel did not restore the sim view: " + JSON.stringify(e1));
    if (e1.sum !== 0) fail("cancel did not discard the strokes: sum " + e1.sum);
    // paint again and SAVE: the drawing survives, the sim stays paused
    run(`function(){ document.getElementById("btnEdit").onclick(); }`);
    run(paint);
    edBtn("save");
    let e2 = ev();
    if (e2.on || !(e2.sum > 0)) fail("save did not keep the drawing: " + JSON.stringify(e2));
    if (run("function(){ return running; }")) fail("save resumed the run");
    // Reset must now upload that drawing
    run(`function(){ document.getElementById("btnReset").onclick(); }`);
    await frame();
    // clear inside the editor, then "save & run": empty drawing, running again
    run(`function(){ document.getElementById("btnEdit").onclick(); }`);
    // 3D: the editor has its OWN z-plane slider (no display card involved)
    if (page.indexOf("3d") >= 0) {
      const pl = run(`function(){ if (!icDraw.plSl) return null;
        icDraw.plSl.value = "7"; icDraw.plSl.oninput(); return [icDraw.plane, icDraw.plSl.max]; }`);
      if (!pl || pl[0] !== 7) fail("the editor's z-plane slider did not move the paint plane: " + JSON.stringify(pl));
      console.log(tag + " editor plane slider -> iz " + pl[0] + " (max " + pl[1] + ")");
    }
    edBtn("clear");
    edBtn("save &amp; run");
    let e3 = ev();
    if (e3.on || e3.sum !== 0) fail("clear + save&run did not empty the drawing: " + JSON.stringify(e3));
    if (!run("function(){ return running; }")) fail("save & run did not resume");
    await frame();
    console.log(tag + " editor view save/run/cancel OK");
    // the editor view is built once, inside the page's #editview host
    if (!run(`function(){ let p = icDraw.cv; while (p.parentNode) p = p.parentNode;
                          return p === document.getElementById("editview"); }`))
      fail("the editor canvas is not inside #editview");
    // --- the self-test path still runs end to end ------------------------------
    // the numbers are meaningless on a stub GPU (every readback is zeros); what this
    // checks is that the path survives the Phase-H buffer changes -- writeBuffer
    // extents, dispatch sizes and readback lengths on the reference grid.
    await run("function(){ return runSelfTest(); }");
    if (!/<table>/.test(getEl("tests").innerHTML)) fail("the self-test produced no table");
    console.log(tag + " self-test path ran to completion (stub values, not physics)");

  } catch (e) {
    fail("threw: " + e.message + "\n" + (e.stack || "").split("\n").slice(0, 4).join("\n"));
  }
  const stt = getEl("status");
  if (stt.className === "err") fail("page reported: " + stt.textContent);
  if (fails.length) { console.log("FAILURES:"); fails.forEach(f => console.log("  " + f)); process.exit(1); }
  console.log(tag + ": stub-GPU card exercise PASSED");
}, 50);
