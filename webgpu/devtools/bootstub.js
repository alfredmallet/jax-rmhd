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
    else {
      const vals = await solver.readCutLine(cards.cfg.zsliceOf(c));
      if (c.type() === "island") islandPush(solver.nsteps * 1e-3, vals, solver.p.ny, solver.p.Ly);
      c.draw({ vals: vals, Ly: solver.p.Ly });
    }
  }
  for (const d of cards.disp) {
    if (!d.showArrows()) { d.clearArrows(); continue; }
    d.drawArrows(await solver.readArrows(d.ci), solver.nax, solver.nay);
  }
}`);
const state = () => run(`function(){
  return { chains: solver.disp.filter(Boolean).length,
           disp: cards.disp.map(d => ({ ci: d.ci, sel: d.sel(), cmap: d.cmap(), zsrc: d.zsrc(),
                                        view: d.selZSrc ? d.selZSrc.value : "-",
                                        mode: solver.modeOf(d.ci), cube: solver.cubeOf(d.ci),
                                        cont: d.cont(), nlev: d.nlev(),
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

    // --- cube faces as a VIEW (3D only, REFINE_PLAN I2.1-I2.3) ----------------
    // any field renders as a cube, with the top face on the card's own plane (manual
    // slider or tracker) and -- on the vector modes -- arrows on that top face.
    if (page.indexOf("3d") >= 0) {
      for (const [fld, view] of [["0", "cube"], ["3", "cubezp"], ["6", "cubezm"],
                                 ["8", "cube"], ["4", "cube"]]) {
        run(`function(f, v){ const c = cards.disp[1];
                             c.selField.value = f; c.selZSrc.value = v; c.apply(); }`, fld, view);
        st = state();
        const D = st.disp[1];
        if (!D.cube) fail("field " + fld + " / " + view + " did not enter the cube view");
        if (D.mode !== parseInt(fld, 10)) fail("the cube view changed the field: " + JSON.stringify(D));
        if (D.cap.indexOf("cube faces") < 0) fail("cube caption missing: " + D.cap);
        if (fld === "4" && !D.arrows) fail("arrows are not offered on the cube top face");
        await frame();
      }
      // the top plane follows the slider in manual mode, and a tracker otherwise
      run(`function(){ const c = cards.disp[1]; c.selField.value = "0";
                       c.selZSrc.value = "cube"; c.rSlice.value = "3"; c.apply(); }`);
      st = state();
      if (st.disp[1].zslice !== 3 || st.disp[1].cap.indexOf("top iz 3") < 0)
        fail("the cube top face did not follow the manual slider: " + JSON.stringify(st.disp[1]));
      if (run("function(){ return cards.disp[1].rSlice.disabled; }"))
        fail("the cube view disabled its own plane slider");
      run(`function(){ const c = cards.disp[1]; c.selZSrc.value = "cubezp"; c.apply(); }`);
      if (!run("function(){ return trackingOn() && cards.disp[1].rSlice.disabled; }"))
        fail("cube + track did not hand the plane to the tracker");
      console.log(tag + " cube view: 5 fields x {manual, track z+, track z-} " +
                  JSON.stringify(state().disp[1]));
      // the arrow overlay's cube frame is the projection of the top face itself
      const fr = run("function(){ return cards.disp[1].arrowFrame() || null; }");
      if (!fr || !isFinite(fr.ax + fr.by + fr.ox) || Math.abs(fr.ay) < 1e-6)
        fail("the cube arrow frame is not a sheared projection: " + JSON.stringify(fr));
      // ... and the cut chart must NOT offer the view at all
      if (run(`function(){ const c = cards.chart.filter(x => x.type() === "cut")[0];
                           return !c || c.selZSrc.options.some(o => o.value.indexOf("cube") === 0); }`))
        fail("the cut card offers the cube view");
      // card add/remove with a cube card live (the card count is at the cap here, so
      // free a slot first and hand it back afterwards)
      run(`function(){ cardClose(cards.disp[cards.disp.length - 1]);
                       addDisplayCard({ sel: 1, zsrc: "cube" }); cardsSync(); }`);
      st = state();
      if (!st.disp[st.disp.length - 1].cube) fail("a re-added card did not restore the cube view");
      await frame();
      run(`function(){ cardClose(cards.disp[cards.disp.length - 1]);
                       addDisplayCard({ sel: 4 }); cardsSync(); }`);
      await frame();
      run(`function(){ cards.disp[1].selZSrc.value = "manual"; cards.disp[1].apply(); }`);
      await frame();
    }

    // --- contour overlay (both apps, REFINE_PLAN I2.4) ------------------------
    // psi / phi contours at two level counts, on a slice card and (3D) a cube card
    {
      const views = page.indexOf("3d") >= 0 ? ["manual", "cube"] : [null];
      for (const v of views) {
        for (const cont of ["3", "2", "0"]) {
          for (const nl of ["8", "32"]) {
            run(`function(v, c, n){ const d = cards.disp[0];
              if (v && d.selZSrc) d.selZSrc.value = v;
              d.selCont.value = c; d.selLev.value = n; d.apply(); }`, v, cont, nl);
            const d0 = state().disp[0];
            if (d0.cont !== parseInt(cont, 10)) fail("contour selection did not take");
            const shown = run("function(){ return cards.disp[0].selLev.style.display; }");
            if ((cont === "0") !== (shown === "none"))
              fail("the level select is not tied to the contour toggle: " + cont + " / " + shown);
            await frame();
          }
        }
      }
      console.log(tag + " contours: {psi, phi, off} x {8, 32} levels" +
                  (views.length > 1 ? " x {slice, cube}" : "") + " OK");
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

    // --- Phase J: rectangular boxes, equilibria, island card, hyper lock -------
    // 2D only; the 3D app is square-perp by construction and offers none of this.
    if (page.indexOf("3d") < 0) {
      const geom = () => run(`function(){
        const d = cards.disp[0], f = d.arrowFrame();
        return { nx: solver.p.nx, ny: solver.p.ny, Lx: solver.p.Lx, Ly: solver.p.Ly,
                 nb: solver.nb, gw: d.gw, gh: d.gh, ar: d.wrap.style.aspectRatio || "",
                 cw: d.cv.width, ch: d.cv.height, vw: d.vcx.__w, vh: d.vcx.__h,
                 fr: f && [f.ax, f.by, f.d.ax, f.d.by],
                 hyper: document.getElementById("selHyper").value,
                 hyperLock: document.getElementById("selHyper").disabled,
                 pm: solver.p.pm };
      }`);
      for (const k of ["tearing", "kh"]) {
        run(`function(k){ const s = document.getElementById("selPreset"); s.value = k; s.onchange(); }`, k);
        const g = geom();
        if (g.nx !== 512 || g.ny !== 128) fail(k + ": expected a 512x128 grid, got " + g.nx + "x" + g.ny);
        if (Math.abs(g.Lx / g.Ly - 2) > 1e-12) fail(k + ": expected Lx/Ly = 2, got " + g.Lx / g.Ly);
        if (g.nb !== 85) fail(k + ": expected 85 shell bins on the wide box, got " + g.nb);
        // aspect-correct card: equal pixels per unit LENGTH, so 2:1 here
        if (g.gw !== 512 || g.gh !== 256) fail(k + ": card geometry " + g.gw + "x" + g.gh + " is not aspect-correct");
        if (g.ar !== "512 / 256") fail(k + ": wrapper aspect-ratio not set: '" + g.ar + "'");
        if (g.cw !== 512 || g.ch !== 256 || g.vw !== 512 || g.vh !== 256)
          fail(k + ": canvases did not follow the card geometry: " + JSON.stringify(g));
        // the arrow frame: rectangular anchors, ISOTROPIC directions
        if (!g.fr || g.fr[0] !== 512 || g.fr[1] !== 256 || g.fr[2] !== g.fr[3])
          fail(k + ": arrow frame is not {rect anchors, isotropic directions}: " + JSON.stringify(g.fr));
        // hyper is LOCKED to 1 by these presets, in the UI and in the solver
        if (g.hyper !== "1" || !g.hyperLock) fail(k + ": hyper is not locked to 1: " + JSON.stringify(g));
        if (run("function(){ return solver.p.hyper; }") !== 1) fail(k + ": the solver kept a hyper != 1");
        await frame();
        // add / close a display card on the rectangular grid
        run(`function(){ addDisplayCard({ sel: 5 }); cardsSync(); }`);
        if (state().disp.length !== 2) fail(k + ": could not add a card on a rectangular grid");
        const g2 = run("function(){ const d = cards.disp[1]; return [d.gw, d.gh]; }");
        if (g2[0] !== 512 || g2[1] !== 256) fail(k + ": a new card is not aspect-correct: " + JSON.stringify(g2));
        await frame();
        run(`function(){ cardClose(cards.disp[1]); }`);
        await frame();
        console.log(tag + " " + k + ": " + JSON.stringify(g));
      }
      // the island card: add, feed it real cut readbacks, close
      run(`function(){ const s = document.getElementById("selPreset"); s.value = "tearing"; s.onchange();
                       addChartCard("island"); cardsSync(); }`);
      await frame(); await frame();
      const isl = run(`function(){ return { n: islandHist.t.length, w: islandHist.w.slice(-1)[0],
                                            eq: icEq.on, curv: icEq.curv, w0: icEq.w0, a: icEq.a }; }`);
      if (!isl.eq || !(isl.curv > 0)) fail("the tearing preset left no equilibrium record: " + JSON.stringify(isl));
      if (!(isl.n >= 1) || !isFinite(isl.w)) fail("the island card collected no W(t): " + JSON.stringify(isl));
      console.log(tag + " island card: " + JSON.stringify(isl));
      run(`function(){ let c; while ((c = cards.chart.filter(x => x.type() === "island")[0])) cardClose(c); }`);
      if (run(`function(){ return cards.chart.some(c => c.type() === "island"); }`))
        fail("the island card would not close");
      // ... and it must not be offered at all in 3D (checked from this side by its absence
      // from chartTypeKeys when cfg.zslice is on -- see the 3D run's own assertion below)
      if (run(`function(){ return chartTypeKeys().indexOf("island") < 0; }`))
        fail("the 2D page does not offer the island chart");
      // eta/nu: a rebuild knob, and it must reach BOTH the solver and the stage kernel
      run(`function(){ const e = document.getElementById("nPm"); e.value = "4"; e.onchange(); }`);
      if (run("function(){ return solver.p.pm; }") !== 4) fail("the eta/nu ratio did not reach the solver");
      const wg = run(`function(){ const S = buildShaders(solver.g);
                                  return [S.stage, S.energyPartial].join("\\n"); }`);
      if (wg.indexOf("select(1.0, 4.0, idx >= NM)") < 0 || wg.indexOf("4.0 * em") < 0)
        fail("the eta/nu ratio is not in the emitted stage / energyPartial WGSL");
      run(`function(){ const e = document.getElementById("nPm"); e.value = "1"; e.onchange(); }`);
      const wg1 = run(`function(){ const S = buildShaders(solver.g);
                                   return [S.stage, S.energyPartial].join("\\n"); }`);
      if (wg1.indexOf("select(1.0,") >= 0 || wg1.indexOf("* em") >= 0 ||
          wg1.indexOf("gridB[m].y * dt") < 0 || wg1.indexOf("(ek + em)") < 0)
        fail("ratio 1 did not restore the scalar-dissipation kernel text");
      console.log(tag + " eta/nu ratio: solver + WGSL round trip OK");
      // hyper is free again once a non-equilibrium preset is chosen
      run(`function(){ const s = document.getElementById("selPreset"); s.value = "forced"; s.onchange(); }`);
      if (run(`function(){ return document.getElementById("selHyper").disabled; }`))
        fail("hyper stayed locked after leaving the equilibrium preset");
      await frame();
    } else if (run(`function(){ return chartTypeKeys().indexOf("island") >= 0; }`)) {
      fail("the 3D page offers the 2D-only island chart");
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
