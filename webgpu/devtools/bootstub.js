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
      c.draw(Object.assign({ perp: sp.perp, nb: solver.nb, fshell: solver.p.fshell,
                             par: sp.par, parKfac: sp.parKfac }, specExtra ? specExtra() : null)); }
    else {
      const vals = await solver.readCutLine(cards.cfg.zsliceOf(c));
      if (c.type() === "island") islandPush(solver.nsteps * 1e-3, vals, solver.p.ny, solver.p.Ly);
      c.draw({ vals: vals, Ly: solver.p.Ly });
    }
  }
  for (const d of cards.disp) {
    if (!d.showArrows()) continue;
    d.setArrows(await solver.readArrows(d.ci), solver.nax, solver.nay);
  }
}`);
const state = () => run(`function(){
  return { chains: solver.disp.filter(Boolean).length,
           disp: cards.disp.map(d => ({ ci: d.ci, sel: d.sel(), cmap: d.cmap(), zsrc: d.zsrc(),
                                        view: d.selZSrc ? d.selZSrc.value : "-",
                                        mode: solver.modeOf(d.ci), cube: solver.cubeOf(d.ci),
                                        cont: d.cont(), nlev: d.nlev(), plain: d.plainBg(),
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
    // ... and the NUMERIC options (FEEDBACK item 8's fit-line index / amplitude), which
    // carry no <option> list: a sane value, a blank, and a NaN, each drawn on real data
    const nums = run(`function(){
      const out = [];
      cards.chart.forEach((c, i) => c.optEls.forEach(s => { if (s.type === "number") out.push([i, s.__optId]); }));
      return out;
    }`);
    for (const nb of nums) {
      for (const v of ["-1.667", "-2.5", "", "abc", "0.05"]) {
        run(`function(i, id, v){
          const s = cards.chart[i].optEls.filter(e => e.__optId === id)[0];
          s.value = v; s.oninput();
        }`, nb[0], nb[1], v);
        await frame();
      }
    }
    if (!nums.length) fail("the spectrum card exposed no numeric options");
    // back to the built defaults, so nothing downstream draws against a stray fit line
    run(`function(){ for (const c of cards.chart) { c.build(); c.draw(null); } }`);
    await frame();
    console.log(tag + " chart numeric options: " + nums.map(n => n[1]).join(" ") + " x {value, blank, NaN}");
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

      // --- field lines as a VIEW + the true k_par spectrum (K, K2) ------------
      // Two things the stub cannot otherwise see are instrumented here: WHICH readbacks
      // the tracer performs (K2.4 reads the samples only for the chart), and what
      // reaches the contour level tables (K2.3 drives the ink-only face through them).
      run(`function(){
        globalThis.flReads = { pos: 0, smp: 0 };
        const rfl = Solver.prototype.readFieldLines;
        Solver.prototype.readFieldLines = async function (w) {
          const r = await rfl.call(this, w);
          globalThis.flReads.pos++; if (r.smp) globalThis.flReads.smp++;
          return r;
        };
        const scl = setContLevels;
        globalThis.contWrites = {};
        globalThis.setContLevels = function (dev, D, nlev, plain) {
          globalThis.contWrites[solver.disp.indexOf(D)] = [D.cont.slice(), nlev, !!plain];
          return scl(dev, D, nlev, plain);
        };
      }`);
      const setSD = v => run(`function(v){ for (const c of cards.chart) {
        if (c.type() !== "spectrum") continue;
        const s = c.optEls.filter(e => e.__optId === "sd")[0];
        if (s) { s.value = v; s.onchange(); } } }`, v);
      const fl = () => run(`function(){
        return { nl: flData && flData.nl, nz: flData && flData.nz,
                 pos: flData && flData.pos.length, par: flPar && flPar.length,
                 // a COPY: the counters live in the sandbox, and a reference would let a
                 // later frame move a number this side has already recorded
                 reads: { pos: globalThis.flReads.pos, smp: globalThis.flReads.smp },
                 cont: JSON.parse(JSON.stringify(globalThis.contWrites)),
                 held: cards.disp.map(d => !!d.lines),
                 views: cards.disp.map(d => d.selZSrc.value),
                 lv: cards.disp.map(d => d.linesView()) }; }`);
      // drawn through the box projection: every point must land at a finite pixel, and
      // there must be one per line point plus two per box edge (the Path2D stub counts
      // and checks them)
      const drawn = fn => run(`function(fn){
        let got = 0;
        const c = { lineCap: "", lineJoin: "", strokeStyle: "", lineWidth: 0,
                    stroke(p) { got = Math.max(got, p.pts); } };
        const F = cards.cfg.lineXform();
        if (fn === "frame") drawBoxFrame(c, F); else drawFieldLines(c, flData, F);
        return got; }`, fn);
      setSD("both");
      // one card in the lines view, one on the cube, one on a plain slice
      run(`function(){ cards.disp[0].selField.value = "4";           // a vector quantity
                       cards.disp[0].selZSrc.value = "lines"; cards.disp[0].apply();
                       cards.disp[1].selZSrc.value = "cubezp"; cards.disp[1].apply();
                       cards.disp[2].selZSrc.value = "manual"; cards.disp[2].apply(); }`);
      await frame(); await frame();            // the hook runs at ~2 Hz
      {
        const f = fl(), st2 = state(), d0 = st2.disp[0];
        if (f.nl !== 64) fail("the lines view traced " + f.nl + " lines, expected 8x8 = 64");
        if (f.pos !== f.nl * f.nz * 2) fail("polyline readback is not nl*nz*2: " + JSON.stringify(f));
        if (f.reads.smp) fail("the along-line samples were read back with no k_par chart open");
        if (drawn() !== f.nl * f.nz) fail("the projection dropped points: " + drawn() + " of " + f.nl * f.nz);
        if (drawn("frame") !== 24) fail("the box frame is not 12 edges: " + drawn("frame") + " points");
        // the view is the whole box: no arrows, no tracker, no slider, top BOUNDARY face
        if (d0.arrows) fail("the lines view drew arrows on a vector quantity");
        if (d0.zsrc !== "manual") fail("the lines view claimed a tracked plane: " + d0.zsrc);
        if (!run("function(){ return cards.disp[0].rSlice.disabled; }"))
          fail("the lines view left its z slider live");
        if (d0.zslice !== st2.nz - 1)
          fail("the lines face is not the top boundary plane: iz " + d0.zslice + " of " + st2.nz);
        if (d0.cap.indexOf("field lines") < 0) fail("lines caption missing: " + d0.cap);
        if (String(f.cont[0]) !== String([[3, 0], 8, true]))
          fail("the lines view did not open on plain-background psi contours: " + JSON.stringify(f.cont[0]));
        if (!f.held[1] || !f.held[2]) fail("a cube / slice card lost its (undrawn) line cache");
        if (f.lv[1] || f.lv[2]) fail("a cube / slice card entered the lines view");
        console.log(tag + " lines view: " + JSON.stringify({ nl: f.nl, views: f.views, cap: d0.cap,
                    lines: drawn(), frame: drawn("frame"), cont: f.cont[0] }));
      }
      // the per-card contour controls stay LIVE in the lines view and drive the face
      for (const [cv, want] of [["2", [2, 0]], ["both", [3, 2]], ["0", [0, 0]], ["3", [3, 0]]]) {
        run(`function(c){ const d = cards.disp[0]; d.selCont.value = c; d.selLev.value = "16"; d.apply(); }`, cv);
        await frame();
        const w = fl().cont[0];
        if (String(w) !== String([want, 16, true]))
          fail("contour select " + cv + " did not reach the lines face: " + JSON.stringify(w));
        if (run(`function(){ return cards.disp[0].selBg.style.display; }`) !== "none")
          fail("the background select is live in the lines view, whose face is always plain");
      }
      console.log(tag + " lines-view contours: {psi, phi, both, off} -> the top face OK");
      // ... and the cut chart must not offer the view at all
      if (run(`function(){ const c = cards.chart.filter(x => x.type() === "cut")[0];
                           return !c || c.selZSrc.options.some(o => o.value === "lines"); }`))
        fail("the cut card offers the field-lines view");
      // close the lines card and re-add one from a saved state
      run(`function(){ cardClose(cards.disp[0]); addDisplayCard({ sel: 1, zsrc: "lines" }); cardsSync(); }`);
      await frame(); await frame();
      if (!fl().lv.some(v => v)) fail("a re-added card did not restore the lines view");
      // no consumer left: the traces are dropped and the march stops
      run(`function(){ for (const d of cards.disp) { d.selZSrc.value = "manual"; d.apply(); } }`);
      await frame(); await frame();
      const off = fl();
      if (off.nl !== null || off.held.some(v => v))
        fail("leaving the lines view left the traces behind: " + JSON.stringify(off));
      // ... but the k_par CHART option traces them on its own, with no lines card open,
      // and only then are the along-line samples read back (K2.4)
      const before = off.reads;
      setSD("fl");
      await frame(); await frame();
      const chart = fl();
      if (!(chart.par > 0) || chart.par !== 3 * (chart.nz >> 1))
        fail("the k_par (field line) option produced no spectrum: " + JSON.stringify(chart));
      if (chart.reads.pos <= before.pos) fail("the chart-only path did not march the lines");
      if (chart.reads.smp !== before.smp + (chart.reads.pos - before.pos))
        fail("the sample readback did not follow the chart: " + JSON.stringify(chart.reads));
      if (chart.lv.some(v => v)) fail("the chart-only path opened a lines view");
      await frame();
      console.log(tag + " k∥ (field line) spectrum, no lines card: " + JSON.stringify(
        { par: chart.par, reads: chart.reads }));
      setSD("both");
      await frame(); await frame();
      if (fl().nl !== null) fail("the field-line trace outlived its only consumer");
    }

    // --- contour overlay (both apps, REFINE_PLAN I2.4 + J2.1/J2.2) ------------
    // psi / phi / BOTH contours at two level counts, over the field and over the plain
    // plate, on a slice card and (3D) a cube card
    {
      const views = page.indexOf("3d") >= 0 ? ["manual", "cube"] : [null];
      const want = { "3": [3, 0], "2": [2, 0], both: [3, 2], "0": [0, 0] };
      for (const v of views) {
        for (const cont of ["3", "2", "both", "0"]) {
          for (const nl of ["8", "32"]) {
            for (const bg of ["0", "1"]) {
              run(`function(v, c, n, b){ const d = cards.disp[0];
                if (v && d.selZSrc) d.selZSrc.value = v;
                d.selCont.value = c; d.selLev.value = n; d.selBg.value = b; d.apply(); }`,
                  v, cont, nl, bg);
              const d0 = state().disp[0];
              if (String(d0.cont) !== String(want[cont]))
                fail("contour selection did not take: " + cont + " -> " + JSON.stringify(d0.cont));
              if (d0.plain !== (bg === "1")) fail("the contour background option did not take");
              const shown = run(`function(){ const d = cards.disp[0];
                return [d.selLev.style.display, d.selBg.style.display].join(","); }`);
              if ((cont === "0") !== (shown === "none,none"))
                fail("the level / background selects are not tied to the contour toggle: "
                     + cont + " / " + shown);
              await frame();
            }
          }
        }
      }
      console.log(tag + " contours: {psi, phi, both, off} x {8, 32} levels x {field, plain} bg" +
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
                 pm: solver.p.pm, eqsrc: !!solver.p.eqsrc };
      }`);
      // hyper is LOCKED to 1 by tearing (resistive-layer physics) and FREE for KH
      // (an ideal instability) -- REFINE_PLAN J2.5
      const HYPLOCK = { tearing: true, kh: false };
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
        // both presets ASK for hyper 1; only tearing locks the control there
        if (g.hyper !== "1" || g.hyperLock !== HYPLOCK[k])
          fail(k + ": wrong hyper lock state: " + JSON.stringify(g));
        if (run("function(){ return solver.p.hyper; }") !== 1) fail(k + ": the solver kept a hyper != 1");
        // ... and only tearing arms the maintained-flux source (J2.3)
        if (g.eqsrc !== (k === "tearing")) fail(k + ": wrong maintain-flux state: " + g.eqsrc);
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
      // KH's hyper really is live (J2.5): the select moves the solver
      run(`function(){ const s = document.getElementById("selPreset"); s.value = "kh"; s.onchange();
                       const h = document.getElementById("selHyper"); h.value = "3"; h.onchange(); }`);
      if (run("function(){ return solver.p.hyper; }") !== 3)
        fail("KH's unlocked hyper did not reach the solver");
      await frame();
      // Pm: a rebuild knob, and it must reach BOTH the solver and the stage kernel. The
      // substitution is on PHI now (the diss slider is eta) -- J2.6.
      const wgsl = () => run(`function(){ const S = buildShaders(solver.g);
                                          return [S.stage, S.energyPartial, S.nlAssemble].join("\\n"); }`);
      const setPm = v => run(`function(v){ const e = document.getElementById("nPm");
                                           e.value = v; e.onchange(); }`, v);
      for (const [v, lit] of [["4", "4.0"], ["0", "0.0"]]) {
        setPm(v);
        if (run("function(){ return solver.p.pm; }") !== parseFloat(v))
          fail("Pm = " + v + " did not reach the solver");
        const wg = wgsl();
        if (wg.indexOf("select(1.0, " + lit + ", idx < NM)") < 0 || wg.indexOf(lit + " * ek") < 0)
          fail("Pm = " + v + " is not in the emitted stage / energyPartial WGSL");
        await frame();
      }
      setPm("1");
      const wg1 = wgsl();
      if (wg1.indexOf("select(1.0,") >= 0 || wg1.indexOf("* ek") >= 0 ||
          wg1.indexOf("gridB[m].y * dt") < 0 || wg1.indexOf("(ek + em)") < 0)
        fail("Pm 1 did not restore the scalar-dissipation kernel text");
      console.log(tag + " Pm (4, 0, back to 1): solver + WGSL round trip OK");
      // the maintained-flux source is emitted ONLY with the toggle on, and only into
      // nlAssemble; switching it rebuilds (J2.3)
      run(`function(){ const s = document.getElementById("selPreset"); s.value = "tearing"; s.onchange(); }`);
      if (wgsl().indexOf("- gridB[m].y * eqk[m]") < 0 || !run("function(){ return !!solver.pl.srcInit; }"))
        fail("the tearing preset did not emit the maintained-flux source");
      await frame();
      run(`function(){ const e = document.getElementById("cbEqSrc"); e.checked = false; e.onchange(); }`);
      if (run("function(){ return solver.p.eqsrc; }") !== false) fail("the maintain-flux toggle did not rebuild");
      if (wgsl().indexOf("eqk") >= 0 || run("function(){ return !!solver.pl.srcInit; }"))
        fail("the source survived its toggle going off");
      // ... and it goes away with the IC preset too, without touching the toggle
      run(`function(){ const e = document.getElementById("cbEqSrc"); e.checked = true; e.onchange();
                       const s = document.getElementById("selIC"); s.value = "modes"; s.onchange(); }`);
      if (run("function(){ return solver.p.eqsrc; }") !== false)
        fail("a non-equilibrium IC kept the maintained-flux source");
      console.log(tag + " maintain equilibrium flux: on / off / IC switch OK");
      await frame();
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

    // --- the sinusoidal z+- packet IC (FEEDBACK item 9, 3D only) ---------------
    // it must build, carry the packet envelopes, keep the chi/packet readout line and
    // leave the paint row hidden (it is not the drawing preset)
    if (page.indexOf("3d") >= 0) {
      run(`function(){ document.getElementById("selIC").value = IC_SINE;
                       document.getElementById("selIC").onchange(); }`);
      await frame();
      const sine = run(`function(){
        const q = solver.p, g = icDrawGrid(q), sz = icSigmaZ(), pg = packetGeom(q.Lz, sz);
        const env = [icGaussZ(q.nz, q.Lz, pg.zPlus, sz), icGaussZ(q.nz, q.Lz, pg.zMinus, sz)];
        const z = icSineZeta(g, env), nrs = g.nx * g.ny;
        // the packet planes: |zeta| peaks where its own envelope does
        let kp = 0, km = 0, mp = 0, mm = 0;
        for (let k = 0; k < g.nz; k++) {
          let ap = 0, am = 0;
          for (let i = 0; i < nrs; i++) {
            ap = Math.max(ap, Math.abs(z.zp[k * nrs + i]));
            am = Math.max(am, Math.abs(z.zm[k * nrs + i]));
          }
          if (ap > mp) { mp = ap; kp = k; }
          if (am > mm) { mm = am; km = k; }
        }
        return { kp: kp, km: km, zp: Math.round(pg.zPlus / q.Lz * q.nz), zm: Math.round(pg.zMinus / q.Lz * q.nz),
                 draw: document.getElementById("rowDraw").style.display,
                 sigz: document.getElementById("rowSigZ").style.display,
                 amp: document.getElementById("rAmpP").disabled,
                 info: document.getElementById("icinfo").innerHTML.length,
                 nz: q.nz };
      }`);
      if (Math.abs(sine.kp - sine.zp) > 1 || Math.abs(sine.km - sine.zm) > 1)
        fail("sinusoid packets are not on their envelope planes: " + JSON.stringify(sine));
      if (sine.draw !== "none" || sine.sigz !== "" || sine.amp)
        fail("sinusoid IC left the wrong IC rows visible: " + JSON.stringify(sine));
      if (!(sine.info > 0)) fail("sinusoid IC lost the chi / packet readout line");
      console.log(tag + " sinusoid z+- IC: packets at iz " + sine.kp + " / " + sine.km +
                  " of " + sine.nz + ", paint row hidden, chi line live");
    }

    // --- auto-diss as a continuous controller (FEEDBACK item 6) ----------------
    // ticked: the slider is the controller's readout and moves on its own; unticked: the
    // manual slider is live again and stays exactly where the controller left it.
    {
      const dz = () => run(`function(){ const e = document.getElementById("rDiss");
        return { v: +e.value, dis: e.disabled, min: +e.min, max: +e.max, diss: solver.p.diss }; }`);
      run(`function(){ const e = document.getElementById("cbAutoDiss"); e.checked = true; e.onchange(); }`);
      const a0 = dz();
      if (!a0.dis) fail("auto-diss ticked did not disable the manual slider");
      if (!(a0.min < a0.max)) fail("the diss slider range is degenerate: " + JSON.stringify(a0));
      // drive the hook with a synthetic spectrum THROUGH THE CACHE (the cards' ride-along
      // path): plenty of energy at k_d, so the target is well above nu_min and the
      // controller must move UP toward it -- and it must never pay for its own readback
      // while a fresh cached one exists
      const moved = await run(`async function(){
        running = true;
        const sv = solver, bins = new Float32Array(3 * sv.nb);
        for (let b = 1; b < sv.nb; b++) { bins[b] = 1e-2; bins[sv.nb + b] = 1e-2; }
        const before = +document.getElementById("rDiss").value;
        const rs = sv.readSpectrum.bind(sv);
        let own = 0;
        sv.readSpectrum = () => { own++; return rs(); };
        for (let i = 0; i < 20; i++) {
          autoDissAt = 0;
          autoDissCache.sv = sv; autoDissCache.at = performance.now(); autoDissCache.perp = bins;
          await autoDissHook(sv);
        }
        sv.readSpectrum = rs;
        const after = +document.getElementById("rDiss").value;
        running = false;
        return [before, after, solver.p.diss, own];
      }`);
      if (!(moved[1] > moved[0])) fail("auto-diss did not track a loud shell upward: " + JSON.stringify(moved));
      if (Math.abs(Math.log10(moved[2]) - moved[1]) > 1e-6)
        fail("the controller's slider value did not reach the solver: " + JSON.stringify(moved));
      if (moved[3] !== 0) fail("auto-diss read the spectrum itself despite a fresh cache: " + moved[3]);
      // a shell with nothing in it must HOLD, not collapse
      const held = await run(`async function(){
        running = true;
        const sv = solver, zero = new Float32Array(3 * sv.nb);
        const before = +document.getElementById("rDiss").value;
        for (let i = 0; i < 20; i++) {
          autoDissAt = 0;
          autoDissCache.sv = sv; autoDissCache.at = performance.now(); autoDissCache.perp = zero;
          await autoDissHook(sv);
        }
        running = false;
        return [before, +document.getElementById("rDiss").value];
      }`);
      if (held[0] !== held[1]) fail("auto-diss moved on an empty spectrum: " + JSON.stringify(held));
      // with NO spectrum card open (stale cache) it must take its own readback
      const noCard = await run(`async function(){
        while (cards.chart.length) cardClose(cards.chart[0]);
        addChartCard("energy"); cardsSync();
        running = true; autoDissAt = 0;
        autoDissCache.sv = null; autoDissCache.perp = null;
        const sv = solver, rs = sv.readSpectrum.bind(sv);
        let own = 0;
        sv.readSpectrum = () => { own++; return rs(); };
        await autoDissHook(sv);
        sv.readSpectrum = rs;
        running = false;
        return [cards.chart.filter(c => c.type() === "spectrum").length, own];
      }`);
      if (noCard[0] !== 0) fail("the no-spectrum-card auto-diss path was not exercised");
      if (noCard[1] !== 1) fail("auto-diss with no card should read the spectrum exactly once: " + noCard[1]);
      const a1 = run(`function(){ const e = document.getElementById("cbAutoDiss"); e.checked = false; e.onchange();
        const r = document.getElementById("rDiss"); return { v: +r.value, dis: r.disabled }; }`);
      if (a1.dis) fail("unticking auto-diss left the slider disabled");
      if (Math.abs(a1.v - held[1]) > 1e-9)
        fail("unticking auto-diss moved the value: " + a1.v + " vs " + held[1]);
      console.log(tag + " auto-diss: tick -> tracks (" + moved[0] + " -> " + moved[1] +
                  "), empty shell holds, no-card readback OK, untick keeps " + a1.v);
    }

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
