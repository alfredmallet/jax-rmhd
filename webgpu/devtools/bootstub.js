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
      if (c.type() === "mode") modePush(solver.nsteps * 1e-3, vals, solver.p.ny);
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
    // measured against what the BOOT layout left, not against a constant: a preset may
    // legitimately open with no chart cards at all (the rmhd-variables demo does), and
    // what this checks is that the three added ones ACCUMULATE on top of whatever was there
    const chart0 = state().charts.length;
    run(`function(){
      addDisplayCard({ sel: 6, cmap: 1 }); addDisplayCard({ sel: 8, cmap: 2 });
      addDisplayCard({ sel: 4 });                       // one past the cap -> refused
      cardsSync();
      for (const t of ["cut", "spectrum", "energy"]) addChartCard(t);
    }`);
    let st = state();
    if (st.disp.length !== 3) fail("expected 3 display cards, got " + st.disp.length);
    if (st.charts.length !== chart0 + 3)
      fail("chart cards did not accumulate: " + chart0 + " + 3 -> " + st.charts.join(","));
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
                                 ["8", "cube"], ["9", "cube"], ["4", "cube"]]) {
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
      // Since ANISO_PLAN the anisotropy card is a SECOND consumer of the along-line
      // samples (its default `ad: "both"` wants the field-line curve). This section's
      // no-consumer assertions ("no k_par chart open", "outlived its only consumer")
      // are about the GATE, not about which card holds it open, so park every aniso
      // card on its along-z-only option for the duration — the spectrum card's `sd`
      // is then again the only thing that can demand the samples, and the original
      // assertions keep their exact meaning. checkaniso.js owns the aniso side of the
      // gate's truth table.
      run(`function(){ for (const c of cards.chart) {
        if (c.type() !== "aniso") continue;
        const s = c.optEls.filter(e => e.__optId === "ad")[0];
        if (s) { s.value = "z"; s.onchange(); } } }`);
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
                                            eq: icEq.on, kh: icEq.kh, curv: icEq.curv,
                                            w0: icEq.w0, a: icEq.a }; }`);
      if (!isl.eq || !(isl.curv > 0)) fail("the tearing preset left no equilibrium record: " + JSON.stringify(isl));
      if (!(isl.n >= 1) || !isFinite(isl.w)) fail("the island card collected no W(t): " + JSON.stringify(isl));
      console.log(tag + " island card: " + JSON.stringify(isl));
      // ... and its gamma_fit (FEEDBACK_2026-08-10 item 9), the same way the k_y mode card's
      // is driven below: the stub readback is all zeros, so W(t) is flat and the legend must
      // stay blank; a synthetic LINEAR-STAGE island (psitilde ~ e^{gamma t}, hence
      // W ~ e^{gamma t/2}) must then be fitted back at gamma, not at gamma/2.
      const isl2 = run(`function(){
        const ny = solver.p.ny, Ly = solver.p.Ly, k = 2 * Math.PI / Ly, G = 0.0287;
        const line = pt => { const v = new Float32Array(4 * ny);
          for (let j = 0; j < ny; j++) v[2 * ny + j] = pt * k * Math.sin(k * j * Ly / ny);
          return v; };
        const flat = islandFitGamma(islandHist.t, islandHist.w);
        islandReset();
        for (let i = 0; i < 60; i++) islandPush(i, line(1e-3 * Math.exp(G * i)), ny, Ly);
        for (const c of cards.chart) if (c.type() === "island") c.draw({ vals: line(1e-3), Ly: Ly });
        return { flat: flat, n: islandHist.t.length, w: islandHist.w.slice(-1)[0],
                 g: islandFitGamma(islandHist.t, islandHist.w) };
      }`);
      if (isFinite(isl2.flat)) fail("the island fit quoted a rate for a flat W(t): " + JSON.stringify(isl2));
      if (isl2.n !== 60 || !(isl2.w > 0) || !(Math.abs(isl2.g - 0.0287) < 1e-6))
        fail("the island card did not fit a synthetic linear stage: " + JSON.stringify(isl2));
      console.log(tag + " island gamma_fit: " + JSON.stringify(isl2));
      run(`function(){ islandReset(); let c; while ((c = cards.chart.filter(x => x.type() === "island")[0])) cardClose(c); }`);
      if (run(`function(){ return cards.chart.some(c => c.type() === "island"); }`))
        fail("the island card would not close");
      // ... and it must not be offered at all in 3D (checked from this side by its absence
      // from chartTypeKeys when cfg.zslice is on -- see the 3D run's own assertion below)
      if (run(`function(){ return chartTypeKeys().indexOf("island") < 0; }`))
        fail("the 2D page does not offer the island chart");
      // the k_y mode card: the KH preset must OPEN with one, it must collect off the same
      // cut readback, and it must sit on its placeholder for any other IC
      run(`function(){ const s = document.getElementById("selPreset"); s.value = "kh"; s.onchange(); }`);
      if (!state().charts.some(t => t === "mode")) fail("the kh preset did not open a k_y mode card");
      await frame(); await frame();
      const md = run(`function(){ return { n: modeHist.t.length, u: modeHist.u.slice(-1)[0],
                                           b: modeHist.b.slice(-1)[0], kh: icEq.kh, eq: icEq.on,
                                           g: modeFitGamma(modeHist.t, modeHist.u) }; }`);
      if (!md.kh || md.eq) fail("the kh preset left the wrong equilibrium record: " + JSON.stringify(md));
      // the stub's readback is all zeros, so A_u = 0 and the log-y history correctly stays
      // EMPTY (that is the guard, not a defect). Drive the same push with a synthetic
      // growing line to see the record and the legend's fit end to end.
      if (md.n !== 0) fail("the k_y mode card logged a zero amplitude: " + JSON.stringify(md));
      const md2 = run(`function(){
        const ny = solver.p.ny, G = 0.267;
        for (let i = 0; i < 50; i++) {
          const v = new Float32Array(4 * ny), A = 1e-5 * Math.exp(G * 0.1 * i);
          for (let j = 0; j < ny; j++) {
            v[j] = A * Math.cos(2 * Math.PI * j / ny + 0.4);
            v[2 * ny + j] = 0.5 * A * Math.cos(2 * Math.PI * j / ny);
          }
          modePush(0.1 * i, v, ny);
        }
        for (const c of cards.chart) if (c.type() === "mode") c.draw({ vals: new Float32Array(4 * ny), Ly: solver.p.Ly });
        return { n: modeHist.t.length, u: modeHist.u.slice(-1)[0], b: modeHist.b.slice(-1)[0],
                 g: modeFitGamma(modeHist.t, modeHist.u) };
      }`);
      if (md2.n !== 50 || !(md2.u > 0) || !(md2.b > 0) || !(Math.abs(md2.g - 0.267) < 1e-6))
        fail("the k_y mode card did not trace a synthetic exponential: " + JSON.stringify(md2));
      // b0 = 0 (the preset's own default) leaves b_x IDENTICALLY zero: the b series must
      // then be dropped rather than dragging the log axis, and the draw must still run
      const md3 = run(`function(){
        const ny = solver.p.ny;
        modeReset();
        for (let i = 0; i < 20; i++) {
          const v = new Float32Array(4 * ny), A = 1e-5 * Math.exp(0.267 * 0.1 * i);
          for (let j = 0; j < ny; j++) v[j] = A * Math.cos(2 * Math.PI * j / ny);
          modePush(0.1 * i, v, ny);
        }
        for (const c of cards.chart) if (c.type() === "mode") c.draw(null);
        return { n: modeHist.t.length, b: modeHist.b.reduce((a, x) => a + x, 0) };
      }`);
      if (md3.n !== 20 || md3.b !== 0) fail("a zero b_x line was not traced: " + JSON.stringify(md3));
      console.log(tag + " k_y mode card: " + JSON.stringify(md) + " -> " + JSON.stringify(md2));
      // a non-equilibrium IC clears the kh flag (the placeholder comes back), and the
      // tearing one must NOT arm this chart any more than KH arms the island one
      run(`function(){ const s = document.getElementById("selIC"); s.value = "letters"; s.onchange(); }`);
      if (run("function(){ return icEq.kh || modeHist.t.length; }"))
        fail("a non-equilibrium IC kept the k_y mode chart armed / its history");
      run(`function(){ const s = document.getElementById("selPreset"); s.value = "tearing"; s.onchange(); }`);
      if (run("function(){ return icEq.kh || !icEq.on; }")) fail("tearing armed the k_y mode chart");
      await frame();
      run(`function(){ let c; while ((c = cards.chart.filter(x => x.type() === "mode")[0])) cardClose(c); }`);
      if (run(`function(){ return cards.chart.some(c => c.type() === "mode"); }`))
        fail("the k_y mode card would not close");
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
    } else if (run(`function(){ return chartTypeKeys().indexOf("mode") >= 0; }`)) {
      fail("the 3D page offers the 2D-only k_y mode chart");
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
      // a hyper change must JUMP the ticked controller to the new hyper's target
      // (autoDissRetarget), not crawl there under the factor-2 cap
      run(`function(){ const e = document.getElementById("cbAutoDiss"); e.checked = true; e.onchange(); }`);
      const jump = await run(`async function(){
        const sv = solver, bins = new Float32Array(3 * sv.nb);
        for (let b = 1; b < sv.nb; b++) { bins[b] = 1e-2; bins[sv.nb + b] = 1e-2; }
        const rs = sv.readSpectrum.bind(sv);
        sv.readSpectrum = async () => ({ perp: bins, par: null, parKfac: 1 });
        const hs = document.getElementById("selHyper"), h0 = hs.value;
        const before = +document.getElementById("rDiss").value;
        hs.value = h0 === "1" ? "4" : "1"; hs.onchange();
        await new Promise(r => setTimeout(r, 0));       // the handler's readback settles
        const e = document.getElementById("rDiss");
        const after = +e.value, hNew = +hs.value;
        const t = autoDissTarget(bins, sv.nb, sv.g.kunit, hNew);
        const want = Math.min(+e.max, Math.max(+e.min, Math.log10(t)));
        hs.value = h0; hs.onchange();                   // restore for later exercises
        await new Promise(r => setTimeout(r, 0));
        sv.readSpectrum = rs;
        return [before, after, want];
      }`);
      if (Math.abs(jump[1] - jump[2]) > 0.051)          // one 0.05 quantization notch
        fail("hyper change did not jump auto-diss to the new target: " + JSON.stringify(jump));
      if (Math.abs(jump[1] - jump[0]) <= Math.log10(2) + 1e-9)
        fail("the hyper-change move never exceeded the per-update cap -- jump untested: " + JSON.stringify(jump));
      // ... and a hyper change must leave a MANUAL (unticked) slider alone
      const manual = await run(`async function(){
        const e = document.getElementById("cbAutoDiss"); e.checked = false; e.onchange();
        const before = +document.getElementById("rDiss").value;
        const hs = document.getElementById("selHyper"), h0 = hs.value;
        hs.value = h0 === "1" ? "4" : "1"; hs.onchange();
        await new Promise(r => setTimeout(r, 0));
        const after = +document.getElementById("rDiss").value;
        hs.value = h0; hs.onchange();
        await new Promise(r => setTimeout(r, 0));
        return [before, after];
      }`);
      if (manual[0] !== manual[1])
        fail("hyper change moved a MANUAL diss slider: " + JSON.stringify(manual));
      console.log(tag + " auto-diss: tick -> tracks (" + moved[0] + " -> " + moved[1] +
                  "), empty shell holds, no-card readback OK, untick keeps " + a1.v +
                  ", hyper jump " + jump[0] + " -> " + jump[1]);
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
    // --- colorbar + save / record (FEEDBACK_2026-08-10 items 12, 13) -----------
    // LABELS first: the three range conventions the colorize kernel implements, driven
    // off a range handed in by hand (every stub readback is zeros, so a real number has
    // to come from somewhere). The strip itself is one cmapRGB sweep -- what matters here
    // is that the numbers under it follow the mode, and that changing the field DROPS the
    // previous mode's range instead of relabelling the new one with it.
    const bar = run(`function(){
      const d = cards.disp[0], out = { lab: {}, stale: null };
      if (d.selZSrc) { d.selZSrc.value = "manual"; d.apply(); }
      for (const [k, sel] of [["signed", 2], ["mag", 4], ["sigma", 8]]) {
        d.selField.value = String(sel); d.apply();
        d.setBarRange(2.5);
        out.lab[k] = d.barT.map(s => s.innerHTML);
      }
      d.selField.value = "3"; d.apply();          // psi: a DIFFERENT autoscale
      out.stale = d.barT.map(s => s.innerHTML).join("|");
      out.needsSigned = d.barNeedsMax();
      d.selField.value = "8"; d.apply();
      out.needsSigma = d.barNeedsMax();           // fixed +-1: no readback at all
      d.selField.value = "4"; d.apply();
      d.selCmap.value = "1"; d.apply();
      out.cmap = d.barCmap;
      out.shown = d.bar.style.display;
      return out;
    }`);
    if (bar.lab.signed.join(",") !== "&minus;2.50,0,+2.50")
      fail("signed colorbar labels are not +-max: " + JSON.stringify(bar.lab.signed));
    if (bar.lab.mag.join(",") !== "0,1.25,2.50")
      fail("magnitude colorbar labels are not 0..max: " + JSON.stringify(bar.lab.mag));
    if (bar.lab.sigma.join(",") !== "&minus;1,0,+1")
      fail("sigma colorbar labels are not the fixed +-1: " + JSON.stringify(bar.lab.sigma));
    if (bar.stale !== "||") fail("a field change kept the old mode's range: " + bar.stale);
    if (!bar.needsSigned || bar.needsSigma)
      fail("the autoscale readback gate is wrong: signed=" + bar.needsSigned + " sigma=" + bar.needsSigma);
    if (bar.cmap !== 1) fail("the strip did not follow the card's colormap: " + bar.cmap);
    if (bar.shown === "none") fail("the colorbar is hidden on a plain slice card");
    // 3D: the lines view renders no field, so it legends no range
    if (page.indexOf("3d") >= 0) {
      const lv = run(`function(){ const d = cards.disp[0];
        d.selZSrc.value = "lines"; d.apply(); const h = d.bar.style.display;
        d.selZSrc.value = "manual"; d.apply(); return [h, d.bar.style.display]; }`);
      if (lv[0] !== "none" || lv[1] === "none") fail("colorbar in the lines view: " + JSON.stringify(lv));
      console.log(tag + " colorbar hidden in the lines view, back on the slice");
    }
    console.log(tag + " colorbar labels: " + JSON.stringify(bar.lab));

    // The RESULT STRIP (Alfred, 2026-08-11, and the save path 2026-08-12). A finished file
    // is no longer thrown at the downloader: it waits on the card's FOOTER behind its own
    // size (and, for a video, its length), with a download button, a share button where the
    // engine can share files, and a dismiss -- because on a phone a silent download lands in
    // Files and is then hard to send on. These read that strip off the booted page and press
    // its buttons. A card carries TWO slots, keyed by kind: a picture and a recording are
    // two different files, so each replaces only its own.
    const stripOf = kind => run(`function(k){ const d = cards.disp[0], s = d.resEl[k];
      const n = d.foot.children.filter(c => (c.className || "").indexOf("recres") >= 0).length;
      const btn = s ? s.children.filter(c => c.kind === "button") : [];
      return { on: !!s, n: n, slots: (d.resEl.png ? 1 : 0) + (d.resEl.video ? 1 : 0),
               cls: s && s.className, foot: !!s && s.parentNode === d.foot,
               txt: s ? s.children.filter(c => c.kind === "span").map(c => c.innerHTML).join("") : "",
               btns: btn.map(c => c.innerHTML), tips: btn.map(c => c.title) }; }`,
      kind || "video");
    const stripPress = (lab, kind) => run(`function(a){ const s = cards.disp[0].resEl[a[1]];
      const b = s && s.children.filter(c => c.kind === "button" && c.innerHTML === a[0])[0];
      if (!b) return false;
      b.onclick(); return true; }`, [lab, kind || "video"]);
    // what that line should READ, computed here from the stubbed bytes and the pumped
    // frame count rather than read back off the page
    const sizeTxt = b => { const k = Math.round(b / 1e3);
      return k < 1000 ? k + " kB" : (b / 1e6).toFixed(1) + " MB"; };
    const lenTxt = s => (s < 10 ? s.toFixed(1) : String(Math.round(s))) + " s";

    // SAVE: the handler must reach toBlob and produce a real PNG -- and since 2026-08-12
    // that PNG takes the recording's route, onto the footer rather than into the downloads
    // folder. The press itself must therefore write NOTHING.
    const nDl = env.caps.downloads.length;
    run(`function(){ cards.disp[0].btnSave.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));       // toBlob is async, as in a browser
    if (env.caps.downloads.length !== nDl)
      fail("pressing save downloaded the picture by itself -- the strip is the point");
    const pngBlob = env.caps.blobs[env.caps.blobs.length - 1];
    const stPng = stripOf("png");
    if (!stPng.on || !stPng.foot || stPng.n !== 1 || stPng.cls !== "recres")
      fail("no result strip on the footer after save: " + JSON.stringify(stPng));
    else {
      // a picture has no length, so the line is the SIZE alone -- no seconds, and no
      // separator left hanging where they used to be
      if (stPng.txt !== sizeTxt(pngBlob.size))
        fail('the save strip reads "' + stPng.txt + '", not "' + sizeTxt(pngBlob.size) + '"');
      if (/·|\ss$/.test(stPng.txt)) fail("the save strip quotes a length: " + stPng.txt);
      if (stPng.btns.join(",") !== "download,share,&times;")
        fail("the save strip's buttons are " + JSON.stringify(stPng.btns));
      if (stPng.tips.some(t => !t))
        fail("a save-strip button has no title tooltip: " + JSON.stringify(stPng.tips));
      console.log(tag + " save strip: " + stPng.txt + " [" + stPng.btns.join(" ") + "]");
    }
    // ... and the download button is where the picture comes out: the same file the direct
    // download used to produce, name pattern, type and bytes alike
    if (!stripPress("download", "png")) fail("the save strip has no download button");
    const png = env.caps.downloads[env.caps.downloads.length - 1];
    if (!png || env.caps.downloads.length !== nDl + 1) fail("the save strip produced no download");
    else {
      if (!/^taranis-[a-z0-9]+-[a-z_]+-t[-0-9.]+\.png$/.test(png.name))
        fail("save filename is off pattern: " + png.name);
      if (!png.blob || png.blob.type !== "image/png" || !(png.blob.size > 0))
        fail("save produced no PNG blob: " + JSON.stringify(png.blob && png.blob.type));
      const sig = png.blob && png.blob.bytes;
      if (!sig || sig.length !== png.blob.size || sig[0] !== 0x89 || sig[1] !== 0x50)
        fail("the downloaded picture is not the PNG the canvas made");
      console.log(tag + " save -> " + png.name + " (" + png.blob.type + ", " + png.blob.size + " B)");
    }
    // SHARE, with the recording's semantics exactly: offered only where the engine can
    // share FILES, handed the picture itself, AbortError silent and anything else a
    // download.
    const nPgSh = env.caps.shares.length, nPgDl = env.caps.downloads.length;
    if (!stripPress("share", "png")) fail("no share button on the save strip of a sharing engine");
    await new Promise(r => setTimeout(r, 5));
    const shp = env.caps.shares[nPgSh];
    if (!shp || !shp.files || shp.files.length !== 1)
      fail("share() was handed " + JSON.stringify(shp && Object.keys(shp)));
    else if (png && png.blob.bytes) {
      const f = shp.files[0], ref = png.blob.bytes;
      if (f.name !== png.name) fail("shared " + f.name + " but downloaded " + png.name);
      if (f.type !== "image/png") fail("the shared picture's File type is " + f.type);
      if (!f.bytes || f.bytes.length !== ref.length || f.bytes.some((v, i) => v !== ref[i]))
        fail("the shared File is not the saved picture's bytes");
      else console.log(tag + " share -> " + f.name + " (" + f.type + ", " + f.bytes.length +
                       " B, byte-identical to the download)");
    }
    if (env.caps.downloads.length !== nPgDl)
      fail("a share that SUCCEEDED downloaded the picture as well");
    env.share.reject = "NotAllowedError";
    stripPress("share", "png");
    await new Promise(r => setTimeout(r, 5));
    if (env.caps.downloads.length !== nPgDl + 1 ||
        !/\.png$/.test(env.caps.downloads[env.caps.downloads.length - 1].name))
      fail("a failed share of the picture did not fall back to a download");
    env.share.reject = "AbortError";
    const nPgAb = env.caps.downloads.length;
    stripPress("share", "png");
    await new Promise(r => setTimeout(r, 5));
    if (env.caps.downloads.length !== nPgAb)
      fail("a share the visitor closed downloaded the picture anyway");
    env.share.reject = "";
    console.log(tag + " save share rejection: NotAllowedError -> download, AbortError -> nothing");
    // an engine that cannot share files grows no share button here either -- and a second
    // save REPLACES the first, one picture and not a pile of them
    env.share.can = false;
    run(`function(){ window._pngWas = cards.disp[0].resEl.png; cards.disp[0].btnSave.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    const stPng2 = stripOf("png");
    const pngNew = run(`function(){ const r = cards.disp[0].resEl.png !== window._pngWas;
      delete window._pngWas; return r; }`);
    if (!stPng2.on || stPng2.btns.join(",") !== "download,&times;")
      fail("an engine that cannot share files still grew a share button on the save strip: " +
           JSON.stringify(stPng2.btns));
    if (!pngNew || stPng2.n !== 1 || stPng2.slots !== 1)
      fail("a second save did not replace the first strip: " + JSON.stringify(stPng2));
    env.share.can = true;
    stripPress("&times;", "png");
    const stPng3 = stripOf("png");
    if (stPng3.on || stPng3.n !== 0) fail("dismiss left the save strip on the footer");
    console.log(tag + " no file sharing -> download only; a second save replaces the strip");
    // a card closed BETWEEN the press and the picture: toBlob's callback is deferred, as a
    // browser's is, so this is a real race and not a contrivance. The strip would have
    // nowhere to live, so the picture is downloaded on the spot rather than lost -- the
    // recording's dead-card rule, on the save path.
    const deadSave = await (async () => {
      run(`function(){ while (cards.disp.length >= CARD_MAX_DISP) cardClose(cards.disp[cards.disp.length - 1]);
                       addDisplayCard(); cardsSync();
                       const d = cards.disp[cards.disp.length - 1];
                       d.btnSave.onclick();
                       cardClose(d); cardsSync(); }`);
      const n = env.caps.downloads.length;
      await new Promise(r => setTimeout(r, 5));
      return env.caps.downloads.length > n ? env.caps.downloads[env.caps.downloads.length - 1] : null;
    })();
    if (!deadSave || !/\.png$/.test(deadSave.name) || !(deadSave.blob && deadSave.blob.size > 0))
      fail("a card closed between the save press and the picture lost the file: " +
           JSON.stringify(deadSave && deadSave.name));
    else console.log(tag + " card closed mid-save -> picture still written");
    // the CAPTURE GROUP: save and rec are children of one node, so a wrapping footer can
    // never put them on two different lines (Alfred, 2026-08-12)
    const capg = run(`function(){ const d = cards.disp[0];
      const g = d.foot.children.filter(c => (c.className || "").indexOf("capgrp") >= 0);
      return { n: g.length, one: !!g[0] && d.btnSave.parentNode === g[0] && d.btnRec.parentNode === g[0],
               kids: g[0] ? g[0].children.map(c => c.innerHTML) : [],
               loose: d.foot.children.indexOf(d.btnSave) }; }`);
    if (capg.n !== 1 || !capg.one || capg.loose >= 0)
      fail("save and rec are not one footer group: " + JSON.stringify(capg));
    if (capg.kids.join(",") !== "save,rec")
      fail("the capture group holds " + JSON.stringify(capg.kids));
    console.log(tag + " capture group: [" + capg.kids.join(" ") + "] as one footer item");
    // RECORD, leg 1 (WebCodecs -> mp4Mux): the preferred path, and the whole reason the
    // muxer exists -- MediaRecorder's fragmented mp4 is what iOS refused to play. The
    // stub encoder hands back deterministic chunks with an avcC-shaped description, and
    // env.tick() drives the frame pump by hand, so a "30 s" recording costs no wall clock.
    const boxesOf = u8 => {
      const out = [];
      let o = 0;
      while (o + 8 <= u8.length) {
        const sz = ((u8[o] << 24) | (u8[o + 1] << 16) | (u8[o + 2] << 8) | u8[o + 3]) >>> 0;
        out.push(String.fromCharCode(u8[o + 4], u8[o + 5], u8[o + 6], u8[o + 7]));
        if (sz < 8) break;
        o += sz;
      }
      return out;
    };
    // "moof" ANYWHERE in the file, sample payloads included: the stub's sample bytes are
    // a ramp, so the four-byte sequence cannot occur in them by accident
    const hasMoof = u8 => {
      for (let i = 0; i + 3 < u8.length; i++)
        if (u8[i] === 0x6d && u8[i + 1] === 0x6f && u8[i + 2] === 0x6f && u8[i + 3] === 0x66) return true;
      return false;
    };
    const mp4File = what => {
      const dl = env.caps.downloads[env.caps.downloads.length - 1];
      if (!dl || !/\.mp4$/.test(dl.name) || !dl.blob || dl.blob.type !== "video/mp4")
        return fail(what + ": no mp4 download (" + JSON.stringify(dl && dl.name) + ")"), null;
      const b = dl.blob.bytes;
      if (!b || !b.length) return fail(what + ": the mp4 download carried no bytes"), null;
      const bx = boxesOf(b);
      if (bx.join(",") !== "ftyp,mdat,moov")
        fail(what + ": the file is not ftyp+mdat+moov but " + bx.join(","));
      if (hasMoof(b)) fail(what + ": the file contains a moof -- it is FRAGMENTED");
      return { name: dl.name, size: b.length, boxes: bx.join("+") };
    };
    // one take: press, pump n frames, press again -- leg 1's toggle with its async probe
    const wcTake = async n => {
      run(`function(){ cards.disp[0].btnRec.onclick(); }`);
      await new Promise(r => setTimeout(r, 5));
      env.tick(n);
      run(`function(){ cards.disp[0].btnRec.onclick(); }`);
      await new Promise(r => setTimeout(r, 5));
    };
    const wcPress = run(`function(){ const d = cards.disp[0]; d.btnRec.onclick();
      return { busy: d.recBusy, live: !!d.wc, mr: !!d.rec }; }`);
    if (!wcPress.busy || wcPress.live || wcPress.mr)
      fail("the WebCodecs config probe did not gate the start: " + JSON.stringify(wcPress));
    await new Promise(r => setTimeout(r, 5));          // the probe is async, as in a browser
    const wcOn = run(`function(){ const d = cards.disp[0];
      return { live: !!d.wc, mr: !!d.rec, label: d.btnRec.innerHTML, cap: !!d.recStop,
               hot: d.btnRec.classList.contains("reclive"), busy: d.recBusy,
               codec: d.wc && d.wc.enc.config.codec, w: d.wc && d.wc.w, h: d.wc && d.wc.h,
               fmt: d.wc && d.wc.enc.config.avc.format, fps: d.wc && d.wc.enc.config.framerate }; }`);
    if (!wcOn.live || wcOn.mr || wcOn.label !== "stop" || !wcOn.hot || wcOn.busy || !wcOn.cap)
      fail("the WebCodecs recording did not start: " + JSON.stringify(wcOn));
    if (!/^avc1\.4200/.test(wcOn.codec || "") || wcOn.fmt !== "avc" || wcOn.fps !== 30)
      fail("the encoder config is wrong: " + JSON.stringify(wcOn));
    const nFr = env.caps.frames.length;
    env.tick(45);                                      // 45 pumped frames = 1.5 s
    const wcPump = run(`function(){ const d = cards.disp[0], W = d.wc;
      return { n: W.n, frames: W.enc.frames, keys: W.enc.keys, chunks: W.chunks.length,
               sync: W.chunks.map((c, i) => (c.key ? i : -1)).filter(i => i >= 0),
               avcC: !!(W.avcC && W.avcC.length), drop: W.drop }; }`);
    if (wcPump.n !== 45 || wcPump.frames !== 45 || wcPump.chunks !== 45)
      fail("the frame pump did not produce 45 frames: " + JSON.stringify(wcPump));
    // FORCED keyframes once a second: the cadence MediaRecorder would not give, and the
    // one iOS needs to show anything but stills
    if (JSON.stringify(wcPump.sync) !== "[0,30]" || wcPump.keys !== 2)
      fail("the forced keyframe cadence is wrong: " + JSON.stringify(wcPump));
    if (!wcPump.avcC) fail("no avcC arrived from the encoder metadata");
    const fr = env.caps.frames.slice(nFr);
    if (fr.length !== 45) fail("the pump built " + fr.length + " VideoFrames, not 45");
    if (fr.some(f => !f.closed)) fail("a VideoFrame was never closed (that leaks GPU memory)");
    const wantTs = fr.map((f, i) => f.timestamp === Math.round(i * 1e6 / 30));
    if (wantTs.indexOf(false) >= 0)
      fail("frame timestamps are not a fixed 1/30 s apart: " + JSON.stringify(fr.slice(0, 3).map(f => f.timestamp)));
    if (fr[0].codedWidth !== wcOn.w || fr[0].codedHeight !== wcOn.h)
      fail("the frames are not the canvas size: " + fr[0].codedWidth + "x" + fr[0].codedHeight);
    console.log(tag + " WebCodecs " + wcOn.codec + " " + wcOn.w + "x" + wcOn.h +
                ", 45 frames, sync at " + JSON.stringify(wcPump.sync));
    // BACKPRESSURE: with the encoder stalled the queue grows, and past REC_QMAX the pump
    // must DROP frames instead of queueing them -- and must not advance the frame index,
    // or the timestamps would stop matching the sample table's constant delta.
    run(`function(){ cards.disp[0].wc.enc.stall = true; }`);
    env.tick(20);
    const wcDrop = run(`function(){ const W = cards.disp[0].wc;
      return { n: W.n, drop: W.drop, q: W.enc.encodeQueueSize, qmax: REC_QMAX }; }`);
    if (!(wcDrop.drop > 0) || wcDrop.n !== 45 + wcDrop.q || wcDrop.q > wcDrop.qmax + 1)
      fail("the stalled encoder was not backpressured: " + JSON.stringify(wcDrop));
    run(`function(){ cards.disp[0].wc.enc.stall = false; }`);
    env.tick(5);
    const wcAfter = run(`function(){ const W = cards.disp[0].wc;
      return { n: W.n, chunks: W.chunks.length, q: W.enc.encodeQueueSize }; }`);
    if (wcAfter.q !== 0 || wcAfter.chunks !== wcAfter.n || wcAfter.n !== wcDrop.n + 5)
      fail("the pump did not recover after the stall: " + JSON.stringify(wcAfter));
    console.log(tag + " backpressure: " + wcDrop.drop + " frames dropped at queue " +
                wcDrop.q + ", " + wcAfter.n + " kept");
    // STOP by button: flush, mux -- and then the strip, NOT a download. The button goes
    // back to its idle state on the same path it always did.
    const nWcDl = env.caps.downloads.length;
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));          // flush() is a promise
    const wcOff = run(`function(){ const d = cards.disp[0];
      return { live: !!d.wc, label: d.btnRec.innerHTML, cap: d.recStop,
               hot: d.btnRec.classList.contains("reclive") }; }`);
    if (wcOff.live || wcOff.label !== "rec" || wcOff.hot || wcOff.cap)
      fail("the WebCodecs recording did not stop: " + JSON.stringify(wcOff));
    if (env.caps.downloads.length !== nWcDl)
      fail("stopping a recording downloaded the file by itself -- the strip is the point");
    // the file's own size, off the last blob the page built. The strip rewraps it as a
    // File for the share sheet and that File carries the same bytes (asserted below), so
    // either answer is the size of the recording.
    const wcSize = env.caps.blobs[env.caps.blobs.length - 1].size;
    const st1 = stripOf();
    if (!st1.on || !st1.foot || st1.n !== 1 || st1.cls !== "recres")
      fail("no result strip on the footer after the recording: " + JSON.stringify(st1));
    else {
      // 70 muxed frames at 30 fps: the DROPPED ones are not in the file and must not be
      // in the length either
      const want = sizeTxt(wcSize) + " · " + lenTxt(wcAfter.chunks / 30);
      if (st1.txt !== want) fail('the strip reads "' + st1.txt + '", not "' + want + '"');
      if (st1.btns.join(",") !== "download,share,&times;")
        fail("the strip's buttons are " + JSON.stringify(st1.btns));
      if (st1.tips.some(t => !t)) fail("a strip button has no title tooltip: " + JSON.stringify(st1.tips));
      console.log(tag + " result strip: " + st1.txt + " [" + st1.btns.join(" ") + "]");
    }
    // ... and the download button is where the file actually comes out, bytes and all
    if (!stripPress("download")) fail("the strip has no download button");
    const wcMp4 = mp4File("WebCodecs stop");
    if (wcMp4) console.log(tag + " record (WebCodecs) -> " + wcMp4.name + " " +
                           wcMp4.boxes + ", " + wcMp4.size + " B");
    // SHARE: offered only where the engine can share FILES (capability detection, no UA
    // sniffing anywhere), and what it hands the sheet must be the recording itself.
    const nShare = env.caps.shares.length, nShDl = env.caps.downloads.length;
    if (!stripPress("share")) fail("no share button on an engine whose canShare says yes");
    await new Promise(r => setTimeout(r, 5));
    const shd = env.caps.shares[nShare];
    if (!shd || !shd.files || shd.files.length !== 1)
      fail("share() was handed " + JSON.stringify(shd && Object.keys(shd)));
    else if (wcMp4) {
      const f = shd.files[0], ref = env.caps.downloads[nShDl - 1].blob.bytes;
      if (f.name !== wcMp4.name) fail("shared " + f.name + " but downloaded " + wcMp4.name);
      if (f.type !== "video/mp4") fail("the shared File's type is " + f.type);
      if (!f.bytes || f.bytes.length !== ref.length || f.bytes.some((v, i) => v !== ref[i]))
        fail("the shared File is not the downloaded file's bytes");
      else console.log(tag + " share -> " + f.name + " (" + f.type + ", " + f.bytes.length +
                       " B, byte-identical to the download)");
    }
    if (env.caps.downloads.length !== nShDl)
      fail("a share that SUCCEEDED downloaded the file as well");
    // a rejected share: anything but AbortError is a real failure and must fall back to
    // the download rather than lose the recording ...
    env.share.reject = "NotAllowedError";
    stripPress("share");
    await new Promise(r => setTimeout(r, 5));
    if (env.caps.downloads.length !== nShDl + 1)
      fail("a failed share did not fall back to a download");
    else mp4File("share fallback");
    // ... whereas AbortError is the visitor CLOSING the sheet, which is a decision, not a
    // failure: pushing the file at them anyway is exactly the behaviour being removed
    env.share.reject = "AbortError";
    const nAb = env.caps.downloads.length;
    stripPress("share");
    await new Promise(r => setTimeout(r, 5));
    if (env.caps.downloads.length !== nAb)
      fail("a share the visitor closed downloaded the file anyway");
    env.share.reject = "";
    console.log(tag + " share rejection: NotAllowedError -> download, AbortError -> nothing");
    // DISMISS: the strip goes, and a second dismiss is inert rather than an error
    if (!stripPress("&times;")) fail("the strip has no dismiss button");
    const st2 = stripOf();
    if (st2.on || st2.n !== 0) fail("dismiss left the strip on the footer: " + JSON.stringify(st2));
    run(`function(){ const d = cards.disp[0]; d.recClear("video"); d.recClear("video"); }`);
    if (stripOf().on) fail("a second dismiss brought the strip back");
    console.log(tag + " strip dismissed, footer clean, a second dismiss inert");
    // an engine WITHOUT file sharing: the same strip, download only. Turning canShare off
    // is the ONLY difference -- there is no UA string anywhere in this path.
    env.share.can = false;
    await wcTake(9);
    const st3 = stripOf();
    if (!st3.on || st3.btns.join(",") !== "download,&times;")
      fail("an engine that cannot share files still grew a share button: " + JSON.stringify(st3));
    if (st3.txt.indexOf(" · 0.3 s") < 0)
      fail("9 muxed frames at 30 fps are not what the strip says: " + st3.txt);
    // ... and a NEW take REPLACES the result: the old strip goes at the press, and one
    // strip -- not two files a press apart -- comes back at the stop
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    const during = stripOf();
    if (during.on || during.n !== 0)
      fail("starting a new recording kept the old result strip: " + JSON.stringify(during));
    env.tick(12);
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    const st4 = stripOf();
    if (!st4.on || st4.n !== 1 || st4.txt.indexOf(" · 0.4 s") < 0)
      fail("the new take did not replace the strip: " + JSON.stringify(st4));
    env.share.can = true;
    stripPress("&times;");
    console.log(tag + " no file sharing -> download only; a new take replaces the strip");
    // the 30 s HARD STOP takes the same path: fire the armed timeout by hand
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    env.tick(3);
    if (!env.fireTimeout(run("function(){ return REC_MAX_MS; }")))
      fail("no 30 s hard stop was armed for the WebCodecs recording");
    await new Promise(r => setTimeout(r, 5));
    const capped = run(`function(){ const d = cards.disp[0];
      return { live: !!d.wc, label: d.btnRec.innerHTML }; }`);
    if (capped.live || capped.label !== "rec") fail("the 30 s cap did not stop it: " + JSON.stringify(capped));
    // the cap is a stop like any other, so it lands on the strip too
    const stCap = stripOf();
    if (!stCap.on || stCap.txt.indexOf(" · 0.1 s") < 0)
      fail("the 30 s cap left no strip (or the wrong one): " + JSON.stringify(stCap));
    if (!stripPress("download")) fail("the capped recording's strip has no download button");
    if (mp4File("30 s cap")) console.log(tag + " 30 s hard stop -> strip, then file, button reset");
    stripPress("&times;");
    // DESTROY mid-recording: closing the card writes what it has rather than losing it.
    // This is the ONE case that still downloads by itself: destroy() sets `dead` before
    // the async flush lands, the card's footer is gone, and a strip with nowhere to live
    // would lose the file silently -- which is worse than an unasked-for download.
    const destroyed = await (async () => {
      run(`function(){ while (cards.disp.length >= CARD_MAX_DISP) cardClose(cards.disp[cards.disp.length - 1]);
                       addDisplayCard(); cardsSync();
                       cards.disp[cards.disp.length - 1].btnRec.onclick(); }`);
      await new Promise(r => setTimeout(r, 5));
      env.tick(4);
      const n = env.caps.downloads.length;
      run(`function(){ cardClose(cards.disp[cards.disp.length - 1]); cardsSync(); }`);
      await new Promise(r => setTimeout(r, 5));
      return env.caps.downloads.length > n;
    })();
    if (!destroyed) fail("closing a card mid-recording lost the file");
    else if (mp4File("destroy mid-record")) console.log(tag + " card closed mid-record -> file still written");
    // a press that can start NOTHING must not cost the visitor the file they still had:
    // a WebCodecs-only engine (MediaRecorder gone -- the iOS shape) whose probe turns
    // this canvas down must leave the old strip standing. The clear lives in the legs'
    // STARTS, not in recToggle (adversarial review 2026-08-12, MINOR 1).
    await wcTake(6);
    const stKeep = stripOf();
    if (!stKeep.on) fail("no strip to preserve before the dead-probe press");
    run(`function(){
      recProbes.clear();                  // drop the cached yes, so the next press probes
      window._pOldICS = window.VideoEncoder.isConfigSupported;
      window._pOldMR = window.MediaRecorder;
      window.VideoEncoder.isConfigSupported = () => Promise.resolve({ supported: false });
      window.MediaRecorder = undefined;
      cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    const afterDead = run(`function(){ const d = cards.disp[0];
      const r = { wc: !!d.wc, mr: !!d.rec, busy: d.recBusy, label: d.btnRec.innerHTML };
      window.VideoEncoder.isConfigSupported = window._pOldICS;
      window.MediaRecorder = window._pOldMR;
      delete window._pOldICS; delete window._pOldMR;
      recProbes.clear();                  // ... and drop the cached no, for the legs below
      return r; }`);
    if (afterDead.wc || afterDead.mr || afterDead.busy || afterDead.label !== "rec")
      fail("the dead-probe press left something running: " + JSON.stringify(afterDead));
    const stKept = stripOf();
    if (!stKept.on || stKept.txt !== stKeep.txt)
      fail("a press whose probe failed threw the old result away: " + JSON.stringify(stKept));
    stripPress("&times;");
    console.log(tag + " dead probe, no fallback -> nothing starts and the old strip survives");
    // rAF-SIDE CAPTURE (RECRAF_PLAN, 2026-08-12): the frame loop is leg 1's real feeder --
    // `recCapture()` straight after the card's render() -- and the interval above is only
    // the watchdog for when there is no rAF to ride. So this leg drives recCapture by hand,
    // exactly as env.tick() drives the timer. The stub's clock jumps 250 ms per
    // performance.now() CALL, far past one 33 ms slot, so every call is slot-due AND the
    // slots it jumped over must be COUNTED (W.drop) rather than backfilled: the file's
    // sample table is a fixed 1/30 s and a backfilled frame would sit at a time it never
    // had. That is also why every existing leg above still exercises the timer path: under
    // this clock `now - lastRaf` is always stale, so the watchdog always takes over.
    const rafCap = k => run(`function(k){ const d = cards.disp[0];
      for (let i = 0; i < k; i++) d.recCapture(); }`, k);
    // stss, read off the finished file: mp4Mux builds it from the chunks' key flags, so it
    // is the seek table a player would actually use. `moov` is the last box, so the last
    // "stss" in the bytes is the real one (the samples are a ramp and the tables around it
    // hold sizes and offsets in the hundreds -- neither can spell it by accident).
    const stssOf = u8 => {
      let at = -1;
      for (let i = 0; i + 3 < u8.length; i++)
        if (u8[i] === 0x73 && u8[i + 1] === 0x74 && u8[i + 2] === 0x73 && u8[i + 3] === 0x73) at = i;
      if (at < 0) return null;
      const u32 = o => ((u8[o] << 24) | (u8[o + 1] << 16) | (u8[o + 2] << 8) | u8[o + 3]) >>> 0;
      const n = u32(at + 8), out = [];
      for (let i = 0; i < n; i++) out.push(u32(at + 12 + 4 * i));
      return out;
    };
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    // the whole point of the change: a capture rides the render the loop already did, so
    // the rAF path must call render() exactly ZERO times of its own
    run(`function(){ const d = cards.disp[0]; d._rn = 0;
      d.render = function () { this._rn++; return DisplayCard.prototype.render.call(this); }; }`);
    const nRafFr = env.caps.frames.length;
    rafCap(35);
    const rafPump = run(`function(){ const d = cards.disp[0], W = d.wc;
      const r = { n: W.n, frames: W.enc.frames, chunks: W.chunks.length, drop: W.drop, rn: d._rn,
                  sync: W.chunks.map((c, i) => (c.key ? i : -1)).filter(i => i >= 0) };
      delete d.render; delete d._rn;               // back to the prototype's own render
      return r; }`);
    if (rafPump.n !== 35 || rafPump.frames !== 35 || rafPump.chunks !== 35)
      fail("the rAF path did not encode one frame per due slot: " + JSON.stringify(rafPump));
    if (rafPump.rn !== 0)
      fail("the rAF path rendered " + rafPump.rn + " extra times -- it must ride loop()'s render");
    if (!(rafPump.drop > 0))
      fail("the slots the clock jumped over were not counted as drops: " + JSON.stringify(rafPump));
    if (JSON.stringify(rafPump.sync) !== "[0,30]")
      fail("the forced keyframe cadence is wrong on the rAF path: " + JSON.stringify(rafPump.sync));
    const rafFr = env.caps.frames.slice(nRafFr);
    if (rafFr.length !== 35) fail("the rAF path built " + rafFr.length + " VideoFrames, not 35");
    if (rafFr.some(f => !f.closed)) fail("a VideoFrame from the rAF path was never closed");
    if (rafFr.map((f, i) => f.timestamp === Math.round(i * 1e6 / 30)).indexOf(false) >= 0)
      fail("rAF-path timestamps are not a fixed 1/30 s apart: " +
           JSON.stringify(rafFr.slice(0, 3).map(f => f.timestamp)));
    // BETWEEN SLOTS (a 120 Hz phone, where only ~every 4th callback captures): nothing is
    // encoded and nothing is dropped -- but the heartbeat still moves, because that is what
    // keeps the watchdog parked. `due = Infinity` is the only "not yet due" a clock that
    // only counts upward can be given.
    const gap = run(`function(){ const d = cards.disp[0], W = d.wc;
      const b = { n: W.n, drop: W.drop, raf: W.lastRaf };
      W.due = Infinity;
      d.recCapture();
      const a = { n: W.n, drop: W.drop, raf: W.lastRaf };
      W.due = W.lastRaf;                           // back on cadence
      return { b: b, a: a }; }`);
    if (gap.a.n !== gap.b.n || gap.a.drop !== gap.b.drop)
      fail("a capture BETWEEN slots was not free: " + JSON.stringify(gap));
    if (!(gap.a.raf > gap.b.raf))
      fail("a skipped capture did not mark the rAF loop alive: " + JSON.stringify(gap));
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    if (!stripPress("download")) fail("the rAF-fed recording's strip has no download button");
    const rafMp4 = mp4File("rAF capture");
    if (rafMp4) {
      const ss = stssOf(env.caps.downloads[env.caps.downloads.length - 1].blob.bytes);
      if (JSON.stringify(ss) !== "[1,31]")             // stss is 1-BASED sample numbers
        fail("the rAF-fed file's stss is " + JSON.stringify(ss) + ", not the forced keyframes");
      console.log(tag + " rAF capture: 35 frames, no extra render, " + rafPump.drop +
                  " skipped slots counted -> " + rafMp4.boxes + ", stss " + JSON.stringify(ss));
    }
    stripPress("&times;");
    // WATCHDOG HANDOFF: when the rAF loop stops calling (a backgrounded tab, the editor
    // view), the timer must pick the recording up where recCapture left it -- same frame
    // index, same timestamps, no gap in the file.
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    rafCap(3);
    const hand0 = run(`function(){ const W = cards.disp[0].wc; return { n: W.n, chunks: W.chunks.length, due: W.due }; }`);
    env.tick(5);
    const hand1 = run(`function(){ const W = cards.disp[0].wc; return { n: W.n, chunks: W.chunks.length, due: W.due }; }`);
    if (hand0.n !== 3 || hand1.n !== 8 || hand1.chunks !== 8)
      fail("the watchdog did not take over when recCapture went quiet: " +
           JSON.stringify(hand0) + " -> " + JSON.stringify(hand1));
    // ... and each fed tick re-bases the slot clock, so the frames the watchdog put in
    // the file are not double-booked into W.drop by the first recCapture after rAF
    // resumes (adversarial review 2026-08-12, MINOR 1).
    if (!(hand1.due > hand0.due))
      fail("watchdog frames did not re-base W.due: " + hand0.due + " -> " + hand1.due);
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    stripPress("&times;");
    console.log(tag + " watchdog handoff: 3 rAF frames, then 5 on the timer, index unbroken");
    // WATCHDOG PARKED: while the rAF loop IS feeding, the timer must do nothing at all --
    // otherwise every frame would be rendered and encoded twice, which is the iPhone
    // stutter this change removes. Under a clock that only moves forward, `lastRaf =
    // Infinity` is the one deterministic way to make the freshness test pass.
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    rafCap(2);
    const park0 = run(`function(){ const W = cards.disp[0].wc; W.lastRaf = Infinity; return W.n; }`);
    env.tick(5);
    const park1 = run(`function(){ const W = cards.disp[0].wc; return { n: W.n, chunks: W.chunks.length }; }`);
    if (park0 !== 2 || park1.n !== park0 || park1.chunks !== park0)
      fail("a live rAF loop did not park the watchdog: " + park0 + " -> " + JSON.stringify(park1));
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    stripPress("&times;");
    console.log(tag + " watchdog parked by a live rAF loop: 5 timer ticks encoded nothing");
    // NO avcC: an engine whose metadata never carries a decoder description cannot give
    // us a playable mp4, so the app must bail to MediaRecorder on the spot -- one frame
    // in, still recording -- and leave WebCodecs off for the rest of the session.
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    run(`function(){ cards.disp[0].wc.enc.noAvcC = true; }`);
    env.tick(1);
    const bail = run(`function(){ const d = cards.disp[0];
      return { wc: !!d.wc, mr: !!d.rec, mime: d.rec && d.rec.mimeType, off: recWCOff,
               label: d.btnRec.innerHTML, hot: d.btnRec.classList.contains("reclive") }; }`);
    if (bail.wc || !bail.mr || !bail.off || bail.label !== "stop" || !bail.hot)
      fail("a missing avcC did not fall back to MediaRecorder: " + JSON.stringify(bail));
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    console.log(tag + " no avcC -> bailed to MediaRecorder mid-press, WebCodecs off");

    // RECORD, leg 2 (MediaRecorder): the fallback, reached here because the bail above
    // turned WebCodecs off -- exactly the state an engine without it boots in.
    // A toggle: start -> live recorder + relabelled button; stop -> the file. The stub
    // engine supports only WebM/vp9, so this leg exercises the WebM FALLBACK of REC_MIME;
    // the MP4-preferred leg is tested right after.
    if (run("function(){ return recWCSupported(cards.disp[0].cv); }"))
      fail("leg 2 is not being tested: WebCodecs is still on");
    const rec1 = run(`function(){ const d = cards.disp[0]; d.btnRec.onclick();
      return { live: !!d.rec, label: d.btnRec.innerHTML, mime: d.rec && d.rec.mimeType,
               fps: d.rec && d.rec.stream && d.rec.stream.fps, hot: d.btnRec.classList.contains("reclive") }; }`);
    if (!rec1.live || rec1.label !== "stop" || !rec1.hot)
      fail("record did not start: " + JSON.stringify(rec1));
    if (rec1.fps !== run("function(){ return REC_FPS; }")) fail("captureStream fps: " + rec1.fps);
    if (rec1.mime !== "video/webm;codecs=vp9") fail("record picked " + rec1.mime);
    // this leg hands back a container it did not write, so it cannot count samples: its
    // length is WALL CLOCK, and the stub owns the clock (env.advance) exactly as it owns
    // leg 1's frame pump, so "12 s" costs no wall clock here either
    const nMrDl = env.caps.downloads.length;
    env.advance(12000);
    const rec2 = run(`function(){ const d = cards.disp[0]; d.btnRec.onclick();
      return { live: !!d.rec, label: d.btnRec.innerHTML, hot: d.btnRec.classList.contains("reclive") }; }`);
    if (rec2.live || rec2.label !== "rec" || rec2.hot)
      fail("record did not stop: " + JSON.stringify(rec2));
    if (env.caps.downloads.length !== nMrDl)
      fail("the MediaRecorder leg downloaded its file by itself instead of offering it");
    const stMr = stripOf();
    const wantMr = sizeTxt(env.caps.blobs[env.caps.blobs.length - 1].size) + " · 12 s";
    if (!stMr.on || stMr.txt !== wantMr)
      fail('the MediaRecorder leg\'s strip reads "' + (stMr.txt || "") + '", not "' + wantMr + '"');
    else console.log(tag + " result strip (MediaRecorder, wall clock): " + stMr.txt);
    if (!stripPress("download")) fail("the MediaRecorder leg's strip has no download button");
    const webm = env.caps.downloads[env.caps.downloads.length - 1];
    if (!webm || !/\.webm$/.test(webm.name) || !(webm.blob && webm.blob.size > 0))
      fail("record produced no webm download: " + JSON.stringify(webm));
    else console.log(tag + " record -> " + webm.name + " (" + webm.blob.type + ", " + webm.blob.size + " B)");
    // ... and an MP4-capable engine (Safari, current Chrome) must NEGOTIATE MP4: that
    // is the point of the mime order -- VP9 WebM does not open on phones (Alfred's
    // iPhone, 2026-08-10 follow-up). Same toggle path, temporarily widened stub.
    const mp4mime = run(`function(){ const M = window.MediaRecorder, old = M.isTypeSupported;
      M.isTypeSupported = m => m.indexOf("mp4") >= 0 || old(m);
      const d = cards.disp[0]; d.btnRec.onclick();
      const mime = d.rec && d.rec.mimeType; d.btnRec.onclick();
      M.isTypeSupported = old; return mime; }`);
    if (mp4mime !== "video/mp4;codecs=avc1") fail("mp4-capable engine picked " + mp4mime);
    if (!stripPress("download")) fail("the mp4-engine take left no strip to download from");
    const mp4 = env.caps.downloads[env.caps.downloads.length - 1];
    if (!mp4 || !/\.mp4$/.test(mp4.name) || !(mp4.blob && mp4.blob.type.indexOf("mp4") >= 0 && mp4.blob.size > 0))
      fail("mp4 record download wrong: " + JSON.stringify(mp4 && mp4.name));
    else console.log(tag + " record (mp4 engine) -> " + mp4.name + " (" + mp4.blob.type + ")");
    // TWO SLOTS (Alfred, 2026-08-12): a picture and a recording are two different files, so
    // a save replaces only the last save and a take only the last take. One slot would mean
    // a 30 s take dying because the visitor saved a PNG a second later. Driven on leg 2,
    // whose whole take is two presses and a clock.
    run(`function(){ const d = cards.disp[0]; d.recClear("png"); d.recClear("video"); }`);
    run(`function(){ cards.disp[0].btnSave.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    const indPng = stripOf("png");
    if (!indPng.on || indPng.n !== 1) fail("no saved picture to defend: " + JSON.stringify(indPng));
    // a whole take over the top of it: the START is where a take clears the last take, and
    // it must not take the picture with it
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    const midTake = stripOf("png");
    if (!midTake.on || midTake.txt !== indPng.txt)
      fail("starting a recording threw the saved picture away: " + JSON.stringify(midTake));
    env.advance(3000);
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    const indVid = stripOf("video"), indPng2 = stripOf("png");
    if (!indVid.on || indVid.txt.indexOf(" · 3.0 s") < 0)
      fail("the take beside a picture produced no strip of its own: " + JSON.stringify(indVid));
    if (!indPng2.on || indPng2.txt !== indPng.txt || indVid.n !== 2 || indVid.slots !== 2)
      fail("a picture and a recording do not sit on the footer together: " +
           JSON.stringify([indPng2, indVid.n, indVid.slots]));
    // ... and the other way: a save while a recording waits replaces the PICTURE only
    run(`function(){ window._vidWas = cards.disp[0].resEl.video; cards.disp[0].btnSave.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    const afterSave = stripOf("video");
    const vidSame = run(`function(){ const r = cards.disp[0].resEl.video === window._vidWas;
      delete window._vidWas; return r; }`);
    if (!vidSame || !afterSave.on || afterSave.txt !== indVid.txt || afterSave.n !== 2)
      fail("saving a picture disturbed the waiting recording: " + JSON.stringify(afterSave));
    // ... and a second take replaces the VIDEO only, still two strips and not three
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    env.advance(5000);
    run(`function(){ cards.disp[0].btnRec.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    const twoTakes = stripOf("video"), stillPng = stripOf("png");
    if (!twoTakes.on || twoTakes.txt.indexOf(" · 5.0 s") < 0 || twoTakes.n !== 2 || twoTakes.slots !== 2)
      fail("the second take did not replace the first beside the picture: " + JSON.stringify(twoTakes));
    if (!stillPng.on || stillPng.txt !== indPng.txt)
      fail("the second take took the picture with it: " + JSON.stringify(stillPng));
    // dismiss, both ways: each × drops its own file and leaves the other standing
    stripPress("&times;", "png");
    const keptVid = stripOf("video");
    if (!keptVid.on || keptVid.n !== 1 || keptVid.slots !== 1 || stripOf("png").on)
      fail("dismissing the picture disturbed the recording: " + JSON.stringify(keptVid));
    run(`function(){ cards.disp[0].btnSave.onclick(); }`);
    await new Promise(r => setTimeout(r, 5));
    stripPress("&times;", "video");
    const keptPng = stripOf("png");
    if (!keptPng.on || keptPng.n !== 1 || keptPng.slots !== 1 || stripOf("video").on)
      fail("dismissing the recording disturbed the picture: " + JSON.stringify(keptPng));
    stripPress("&times;", "png");
    if (stripOf("png").n !== 0) fail("a strip survived both dismissals");
    console.log(tag + " two slots: save and take replace only their own, dismiss only their own");
    // DESTROY mid-recording on THIS leg too: the dead branch is recResult's, but the stop
    // that reaches it is MediaRecorder's own onstop -- a different route than leg 1's
    // flush, so it gets its own assertion (adversarial review 2026-08-12, MINOR 3)
    const mrDestroyed = await (async () => {
      run(`function(){ while (cards.disp.length >= CARD_MAX_DISP) cardClose(cards.disp[cards.disp.length - 1]);
                       addDisplayCard(); cardsSync();
                       cards.disp[cards.disp.length - 1].btnRec.onclick(); }`);
      await new Promise(r => setTimeout(r, 5));
      env.advance(700);
      const n = env.caps.downloads.length;
      run(`function(){ cardClose(cards.disp[cards.disp.length - 1]); cardsSync(); }`);
      await new Promise(r => setTimeout(r, 5));
      return env.caps.downloads.length > n ? env.caps.downloads[env.caps.downloads.length - 1] : null;
    })();
    if (!mrDestroyed || !/\.webm$/.test(mrDestroyed.name) || !(mrDestroyed.blob && mrDestroyed.blob.size > 0))
      fail("closing a card mid-MediaRecorder-recording lost the file: " + JSON.stringify(mrDestroyed && mrDestroyed.name));
    else console.log(tag + " card closed mid-record (MediaRecorder) -> file still written");
    // ... and with NEITHER leg available the button is simply not there (an engine with
    // no MediaRecorder and no WebCodecs; WebCodecs is still off from the bail above)
    const noRec = run(`function(){
      const M = window.MediaRecorder;
      window.MediaRecorder = undefined;
      const off = recSupported(cards.disp[0].cv);
      while (cards.disp.length >= CARD_MAX_DISP) cardClose(cards.disp[cards.disp.length - 1]);
      const d = addDisplayCard(); cardsSync();
      const hid = d.btnRec.style.display, sav = d.btnSave.style.display;
      const g = d.btnSave.parentNode;
      const grp = { cls: g && g.className, rec: !!g && d.btnRec.parentNode === g,
                    foot: !!g && g.parentNode === d.foot };
      cardClose(d); cardsSync();
      window.MediaRecorder = M;
      return { off: off, hid: hid, sav: sav, grp: grp, back: recSupported(cards.disp[0].cv) };
    }`);
    if (noRec.off !== false || noRec.back !== true)
      fail("MediaRecorder feature detection: " + JSON.stringify(noRec));
    if (noRec.hid !== "none") fail("neither recording leg, but the rec button is shown: " + noRec.hid);
    if (noRec.sav === "none") fail("no MediaRecorder took the SAVE button away too");
    // ... and the group is still there around the one button that is left: a hidden `rec`
    // is out of the flex flow, gap and all, so there is no stray space to degrade into
    if (noRec.grp.cls !== "capgrp" || !noRec.grp.rec || !noRec.grp.foot)
      fail("the capture group did not survive a hidden rec button: " + JSON.stringify(noRec.grp));
    console.log(tag + " no MediaRecorder, no WebCodecs -> rec button absent, save intact in its group");
    // ... whereas MediaRecorder absent but WebCodecs present (iOS Safari 16.4+, which is
    // the case this whole feature is for) must still offer the button
    const wcOnly = run(`function(){
      const M = window.MediaRecorder;
      window.MediaRecorder = undefined; recWCOff = false;
      const any = recAnySupported(cards.disp[0].cv);
      while (cards.disp.length >= CARD_MAX_DISP) cardClose(cards.disp[cards.disp.length - 1]);
      const d = addDisplayCard(); cardsSync();
      const hid = d.btnRec.style.display;
      cardClose(d); cardsSync();
      window.MediaRecorder = M; recWCOff = true;
      return { any: any, hid: hid };
    }`);
    if (!wcOnly.any || wcOnly.hid === "none")
      fail("WebCodecs alone did not keep the rec button: " + JSON.stringify(wcOnly));
    console.log(tag + " WebCodecs but no MediaRecorder -> rec button still offered");
    await frame();

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
