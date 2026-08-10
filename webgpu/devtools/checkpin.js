// PINCURVE gates: the specCurves seam (GATE A), the pin/unpin/ghost arithmetic (GATE B)
// and the four motivating workflows driven through a booted page (GATE C).
// Usage: node checkpin.js <webgpu-dir>            exit code 1 on any failure
//
// Sections 1-2 run the REAL common.js functions in a bare vm (no page): section 1 checks
// specCurves against the PRE-REFACTOR inline loop, ported verbatim below as the reference,
// so the Phase A extraction is measured against what it replaced and not against itself.
// Sections 3-5 boot rmhd2d / rmhd3d on stubenv and press the real header buttons.
"use strict";
const fs = require("fs"), vm = require("vm"), path = require("path");
const dir = process.argv[2] || path.join(__dirname, "..");
let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};

// ---------------------------------------------------------------------------
// a bare vm holding common.js, for the pure functions
// ---------------------------------------------------------------------------
const stubEl = () => ({ value: "", style: {}, textContent: "", innerHTML: "", checked: false,
                        disabled: false, min: "", max: "", step: "", options: [],
                        addEventListener() {}, appendChild() {} });
const sandbox = {
  document: { getElementById: () => stubEl(), createElement: () => stubEl(),
              createTextNode: () => ({}), querySelectorAll: () => [] },
  window: { addEventListener() {}, devicePixelRatio: 1, matchMedia: () => ({ matches: true }) },
  console, Math, JSON, Float32Array, Float64Array, Uint32Array, Uint8ClampedArray, Map, Set,
  Error, Promise, setTimeout, Number, String, Array, Object, isFinite, parseInt, parseFloat,
  URLSearchParams, performance: { now: () => 0 }
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(dir, "common.js"), "utf8"), sandbox, { filename: "common.js" });
// top-level `const`s of a vm script are not context properties: pull them out
const C = Object.assign({}, vm.runInContext(
  "({ specCurves, specSeries, specFloor, SPEC_SETS, PIN_MAX, PIN_ALPHA, pinDraw, pinKmax,"
  + " SPEC_KNEE, SPEC_KFRAC, SPEC_TAIL, SPEC_MAXDEC, drawSpectrum, CHART_TYPES })", sandbox));

// ---------------------------------------------------------------------------
console.log("1. specCurves (GATE A) vs the pre-refactor inline loop");
// ---------------------------------------------------------------------------
// The front half of drawSpectrum EXACTLY as it stood before PINCURVE Phase A
// (common.js @ bdeb53e, lines 1111-1153), reproduced here as the reference.
function refCurves(d, o) {
  const bins = (d && d.perp) || new Float32Array(3), nb = (d && d.nb) || 1;
  const parKfac = (d && d.parKfac) || 1;
  const set = C.specSeries(o && o.sq);
  const sd = (o && o.sd) || "both";
  const fl = sd === "fl";
  const par = d && (fl ? d.parFL : d.par);
  const wantPerp = sd !== "par", wantPar = sd !== "perp" && par && par.length >= 3;
  const curves = [];
  let hi = 0, lo = Infinity, hiP = 0, loP = Infinity;
  if (wantPerp) {
    for (const sr of set) {
      const pts = [];
      for (let b = 1; b < nb; b++) {
        const v = sr[2](bins[b], bins[nb + b], bins[2 * nb + b]);
        if (v > 0 && isFinite(v)) { pts.push(b, v); hiP = Math.max(hiP, v); loP = Math.min(loP, v); }
      }
      curves.push([pts, sr[1], null, sr[0]]);
    }
  }
  if (wantPar) {
    const nzb = Math.floor(par.length / 3);
    for (const sr of set) {
      const pts = [];
      for (let b = 1; b <= nzb; b++) {
        const v = sr[2](par[b - 1], par[nzb + b - 1], par[2 * nzb + b - 1]);
        if (v > 0 && isFinite(v)) { pts.push(b * parKfac, v); hi = Math.max(hi, v); lo = Math.min(lo, v); }
      }
      curves.push([pts, sr[1], [5, 3], sr[0] + (fl ? "(k∥ line)" : "(k∥)")]);
    }
  }
  if (hiP > 0) { hi = hiP; lo = loP; }
  const nPerp = wantPerp ? set.length : 0;
  return { curves, hiP, loP, hi, lo, nPerp };
}
// synthetic bin stacks. `perp` is the 3*nb [E_u | E_b | H_c] stack, `par` the 3*nzb one.
function mkPerp(nb, f) {
  const a = new Float32Array(3 * nb);
  for (let b = 0; b < nb; b++) { const v = f(b); a[b] = v[0]; a[nb + b] = v[1]; a[2 * nb + b] = v[2]; }
  return a;
}
function mkPar(nzb, f) {
  const a = new Float32Array(3 * nzb);
  for (let b = 0; b < nzb; b++) { const v = f(b); a[b] = v[0]; a[nzb + b] = v[1]; a[2 * nzb + b] = v[2]; }
  return a;
}
const kolm = b => { const k = Math.max(b, 1); const e = Math.pow(k, -5 / 3); return [e, 0.8 * e, 0.1 * e]; };
const CASES = [
  ["2D perp-only, 24 bins", { perp: mkPerp(24, kolm), nb: 24, fshell: [1, 3] }],
  ["2D one live bin", { perp: mkPerp(2, kolm), nb: 2 }],
  ["2D empty (all zero)", { perp: mkPerp(16, () => [0, 0, 0]), nb: 16 }],
  ["2D NaN / Inf lanes", { perp: mkPerp(12, b => (b === 3 ? [NaN, Infinity, 0]
                                                          : b === 5 ? [-1, 0, 0] : kolm(b))), nb: 12 }],
  ["2D H_c > E_u+E_b (E- <= 0)", { perp: mkPerp(10, b => { const e = kolm(b); return [e[0], e[1], 9 * (e[0] + e[1])]; }), nb: 10 }],
  ["3D perp + par", { perp: mkPerp(20, kolm), nb: 20, par: mkPar(8, kolm), parKfac: 1.5 }],
  ["3D perp + par + fl", { perp: mkPerp(20, kolm), nb: 20, par: mkPar(8, kolm),
                           parFL: mkPar(6, b => kolm(b).map(x => 3 * x)), parKfac: 1.5 }],
  ["3D par lane empty", { perp: mkPerp(20, kolm), nb: 20, par: mkPar(8, () => [0, 0, 0]), parKfac: 1 }],
  ["3D par too short", { perp: mkPerp(20, kolm), nb: 20, par: new Float32Array(2), parKfac: 1 }],
  ["no data at all", null]
];
const OPTS = [];
for (const sq of ["ub", "pm", "both", undefined, "junk"])
  for (const sd of ["both", "perp", "par", "fl", undefined]) OPTS.push({ sq, sd });
{
  let n = 0, mism = "";
  for (const [nm, d] of CASES) {
    for (const o of OPTS) {
      const a = C.specCurves(d, o), b = refCurves(d, o);
      n++;
      const key = x => JSON.stringify([x.curves.map(cv => [Array.from(cv[0]), cv[1], cv[2], cv[3]]),
                                       x.hiP, x.loP === Infinity ? "inf" : x.loP,
                                       x.hi, x.lo === Infinity ? "inf" : x.lo, x.nPerp]);
      if (key(a) !== key(b) && !mism) mism = nm + " / sq=" + o.sq + " sd=" + o.sd;
    }
  }
  ok("specCurves == pre-refactor loop over " + n + " (case x sq x sd) combinations", !mism, mism || undefined);
  // and the seam actually carries data, so the comparison above is not vacuous
  const s = C.specCurves(CASES[5][1], { sq: "both", sd: "both" });
  ok("  ... the 3D both/both case really builds 8 curves (4 perp + 4 par)",
     s.curves.length === 8 && s.nPerp === 4 && s.curves[0][2] === null &&
     String(s.curves[4][2]) === "5,3" && s.hiP > 0 && s.hi === s.hiP,
     "n=" + s.curves.length + " nPerp=" + s.nPerp);
  const e = C.specCurves(CASES[2][1], { sq: "ub", sd: "perp" });
  ok("  ... an all-zero stack yields empty point lists and hi = 0",
     e.curves.length === 2 && e.curves[0][0].length === 0 && e.hi === 0 && e.hiP === 0);
  const f = C.specCurves(CASES[3][1], { sq: "ub", sd: "perp" });
  ok("  ... NaN / Inf / negative bins are dropped, not drawn",
     f.curves[0][0].every(v => isFinite(v)) && f.curves[0][0].length === 2 * 9,
     "pts=" + (f.curves[0][0].length / 2));
}
// The plan's GATE A also asks for a by-eye check that all four sq x sd combinations draw
// identically. Eyes need a GPU; a byte-diff of the CANVAS CALL LOG does not, and is
// stricter. Point PINCURVE_REF at a pre-refactor common.js (git show <pre>:webgpu/common.js)
// and every drawSpectrum call is replayed against it; without it the check is skipped
// (once this work is committed, HEAD is no longer a reference).
if (process.env.PINCURVE_REF) {
  const rs = Object.assign({}, sandbox, { globalThis: null });
  rs.globalThis = rs;
  vm.createContext(rs);
  vm.runInContext(fs.readFileSync(process.env.PINCURVE_REF, "utf8"), rs, { filename: "ref-common.js" });
  const R = vm.runInContext("({ drawSpectrum })", rs);
  let n = 0, mism = "";
  for (const [nm, d] of CASES) {
    for (const sq of ["ub", "pm", "both"]) for (const sd of ["both", "perp", "par", "fl"]) {
      for (const fit of ["pin", "amp", "off"]) {
        const o = { sq, sd, fit, fitp: -1.667, fita: fit === "amp" ? "2.5" : "" };
        const a = drawLog(C.drawSpectrum, d, o), b = drawLog(R.drawSpectrum, d, o);
        n++;
        if (a !== b && !mism) mism = nm + " / " + sq + " " + sd + " " + fit;
      }
    }
  }
  ok("drawSpectrum canvas call log == pre-refactor over " + n + " combinations (no pins)",
     !mism, mism || undefined);
} else {
  console.log("  SKIP  drawSpectrum vs pre-refactor call log (set PINCURVE_REF)");
}
// every canvas call, with the state that mattered at the time, as one string
function drawLog(fn, d, o) {
  const log = [];
  const c = { fillStyle: "", strokeStyle: "", lineWidth: 1, globalAlpha: 1,
              font: "10px x", textAlign: "left", textBaseline: "alphabetic" };
  for (const m of ["clearRect", "strokeRect", "beginPath", "moveTo", "lineTo", "stroke", "fill",
                   "clip", "save", "restore", "setLineDash", "rect", "fillRect", "fillText"]) {
    c[m] = (...a) => log.push([m, a.map(v => (typeof v === "number" ? v.toPrecision(12) : String(v))).join(","),
                               c.fillStyle, c.strokeStyle, c.lineWidth, c.globalAlpha, c.textAlign].join("|"));
  }
  c.measureText = t => ({ width: 6.2 * t.length, actualBoundingBoxAscent: 7.2, actualBoundingBoxDescent: 0 });
  fn(c, d, o);
  return log.join("\n");
}

// ---------------------------------------------------------------------------
console.log("2. pin arithmetic (GATE B), on the real pinDraw / specFloor");
// ---------------------------------------------------------------------------
{
  // -- deep-copy independence: mutate the live bins after pinning, the pin must not move
  const live = mkPerp(16, kolm);
  const d = { perp: live, nb: 16 };
  const snap = C.specCurves(d, { sq: "ub", sd: "perp" });
  const pin = { curves: snap.curves.map(cv => [Array.from(cv[0]), cv[1], cv[2], cv[3]]),
                nPerp: snap.nPerp, hiP: snap.hiP, loP: snap.loP, hi: snap.hi, lo: snap.lo,
                t: 1.5, kunit: 1 };
  const before = JSON.stringify(pin.curves[0][0]);
  for (let i = 0; i < live.length; i++) live[i] = 1e9;
  const after = JSON.stringify(pin.curves[0][0]);
  ok("snapshot survives mutation of the live bins", before === after);
  const reread = C.specCurves(d, { sq: "ub", sd: "perp" });
  ok("  ... and the live curves really did change (the test is not vacuous)",
     JSON.stringify(reread.curves[0][0]) !== before);
  // the pin must also be plain Arrays, not views on the readback
  ok("  ... pin points are a plain Array, not a Float32Array view",
     Array.isArray(pin.curves[0][0]) && !ArrayBuffer.isView(pin.curves[0][0]));

  // -- kunit rescale
  const one = C.pinDraw([pin], 1);
  ok("kunit rescale: identity when kunit_pin == kunit_live",
     JSON.stringify(one[0].curves[0][0]) === before);
  const half = C.pinDraw([pin], 0.5);
  const xs0 = pin.curves[0][0].filter((_, i) => i % 2 === 0);
  const xs1 = half[0].curves[0][0].filter((_, i) => i % 2 === 0);
  const ys0 = pin.curves[0][0].filter((_, i) => i % 2 === 1);
  const ys1 = half[0].curves[0][0].filter((_, i) => i % 2 === 1);
  ok("  ... pin at kunit=1 drawn at kunit=0.5 doubles every x",
     xs1.length === xs0.length && xs1.every((v, i) => Math.abs(v - 2 * xs0[i]) < 1e-12));
  ok("  ... and moves no y", ys1.every((v, i) => v === ys0[i]));
  const nok = C.pinDraw([{ curves: pin.curves, nPerp: 1, t: 0, kunit: 0 }], 2);
  ok("  ... an unknown kunit (0) on either side does not rescale",
     JSON.stringify(nok[0].curves[0][0]) === before &&
     JSON.stringify(C.pinDraw([pin], 0)[0].curves[0][0]) === before);
  ok("  ... pinDraw does not mutate the stored pin", JSON.stringify(pin.curves[0][0]) === before);

  // -- age ladder
  const four = C.pinDraw([1, 2, 3, 4].map(t => ({ curves: [], nPerp: 0, t, kunit: 1 })), 1);
  ok("age alphas run newest-first down the ladder",
     JSON.stringify(four.map(p => p.alpha)) === JSON.stringify([0.20, 0.26, 0.34, 0.45]),
     four.map(p => p.alpha).join(","));
  ok("  ... a single pin gets the brightest alpha",
     C.pinDraw([{ curves: [], nPerp: 0, t: 0, kunit: 1 }], 1)[0].alpha === C.PIN_ALPHA[0]);
  ok("  ... PIN_MAX is 4", C.PIN_MAX === 4);

  // -- pinKmax (the x axis a ghosts-only card needs)
  ok("pinKmax reads the largest ghost k", C.pinKmax(half) === xs1[xs1.length - 1],
     String(C.pinKmax(half)));
  ok("  ... and is 0 with no ghosts", C.pinKmax([]) === 0);

  // -- range union: a pinned curve a decade below the live one pulls the floor down
  const decLo = [[[1, 1e-6, 2, 1e-6, 3, 1e-6], "#fff", null, "old"]];
  const liveHi = [[[1, 1, 2, 1, 3, 1], "#fff", null, "new"]];
  const f1 = C.specFloor(liveHi, 1, 1);
  const f2 = C.specFloor(liveHi.concat(decLo), 1, 1e-6);
  ok("range union: pinned perp curve a decade below drops the floor",
     f2 < f1 && f2 <= 1e-6 + 1e-18, "floor " + f1 + " -> " + f2);
}

// ---------------------------------------------------------------------------
console.log("3. drawSpectrum with pins: the waiting truth table + draw order");
// ---------------------------------------------------------------------------
// A recording 2D context: every call in order, so the order of the drawn layers and the
// alphas they were drawn at can be asserted without pixels.
function recCtx() {
  const log = [];
  const o = { fillStyle: "", strokeStyle: "", lineWidth: 1, globalAlpha: 1,
              font: "10px x", textAlign: "left", textBaseline: "alphabetic" };
  for (const m of ["clearRect", "strokeRect", "beginPath", "moveTo", "lineTo", "stroke",
                   "fill", "clip", "save", "restore", "setLineDash", "rect", "fillRect"]) {
    o[m] = (...a) => {
      for (const v of a) if (typeof v === "number" && !isFinite(v)) log.push(["NONFINITE", m]);
      if (m === "stroke") log.push(["stroke", o.strokeStyle, o.globalAlpha, o.lineWidth]);
    };
  }
  o.measureText = t => ({ width: 6.2 * t.length, actualBoundingBoxAscent: 7.2,
                          actualBoundingBoxDescent: 0 });
  o.fillText = t => log.push(["text", t]);
  o.log = log;
  return o;
}
{
  const d = { perp: mkPerp(24, kolm), nb: 24, fshell: [1, 3], kunit: 1 };
  const o = { sq: "ub", sd: "perp", fit: "off" };
  const snap = C.specCurves(d, o);
  const pin = { curves: snap.curves.map(cv => [Array.from(cv[0]), cv[1], cv[2], cv[3]]),
                nPerp: snap.nPerp, hiP: snap.hiP, loP: snap.loP, hi: snap.hi, lo: snap.lo,
                t: 12.34, kunit: 1 };
  const waiting = c => c.log.some(e => e[0] === "text" && /waiting/.test(e[1]));
  const c1 = recCtx(); C.drawSpectrum(c1, null, o, []);
  ok("no live data, no pins -> waiting", waiting(c1));
  const c2 = recCtx(); C.drawSpectrum(c2, d, o, []);
  ok("live data, no pins -> drawn", !waiting(c2));
  const c3 = recCtx(); C.drawSpectrum(c3, null, o, [pin]);
  ok("no live data, pins -> drawn (axes + ghosts)", !waiting(c3));
  const c4 = recCtx(); C.drawSpectrum(c4, { perp: mkPerp(16, () => [0, 0, 0]), nb: 16 }, o, [pin]);
  ok("empty live stack, pins -> drawn", !waiting(c4));
  const c5 = recCtx(); C.drawSpectrum(c5, { perp: mkPerp(16, () => [0, 0, 0]), nb: 16 }, o, []);
  ok("empty live stack, no pins -> waiting", waiting(c5));
  ok("  ... nothing non-finite reached the canvas in any of the five",
     ![c1, c2, c3, c4, c5].some(c => c.log.some(e => e[0] === "NONFINITE")));

  // draw order + alpha: ghosts before the live curves, at PIN_ALPHA, at lineWidth 1
  const c6 = recCtx(); C.drawSpectrum(c6, d, o, [pin]);
  const strokes = c6.log.filter(e => e[0] === "stroke");
  const ghost = strokes.filter(e => e[2] === C.PIN_ALPHA[0]);
  const liveS = strokes.filter(e => e[2] === 1 && e[3] === 1.4);
  ok("ghosts stroked at alpha " + C.PIN_ALPHA[0] + ", width 1",
     ghost.length === 2 && ghost.every(e => e[3] === 1), "n=" + ghost.length);
  ok("  ... and BEFORE every live curve",
     ghost.length && liveS.length && strokes.indexOf(ghost[ghost.length - 1]) < strokes.indexOf(liveS[0]));
  ok("  ... the canvas alpha is left at 1", c6.globalAlpha === 1);
  ok("  ... the ghosts keep their own hue",
     ghost.map(e => e[1]).join(",") === snap.curves.map(cv => cv[1]).join(","));

  // legend: one collapsed entry naming the pinned times
  const leg = c6.log.filter(e => e[0] === "text").map(e => e[1]);
  ok("legend carries one collapsed pinned entry",
     leg.filter(t => /pinned/.test(t)).length === 1 &&
     leg.some(t => t === "1 pinned @t=12.3"), leg.filter(t => /pinned/.test(t)).join("|"));
  const c7 = recCtx();
  C.drawSpectrum(c7, d, o, [Object.assign({}, pin, { t: 8.06 }), pin]);
  ok("  ... two pins -> \"2 pinned @t=8.1, 12.3\"",
     c7.log.some(e => e[0] === "text" && e[1] === "2 pinned @t=8.1, 12.3"),
     c7.log.filter(e => e[0] === "text" && /pinned/.test(e[1])).map(e => e[1]).join("|"));

  // pinned PARALLEL ghosts stay dashed and never stretch the range
  const d3 = { perp: mkPerp(20, kolm), nb: 20, par: mkPar(8, kolm), parKfac: 1.5,
               fshell: [1, 3], kunit: 1 };
  const s3 = C.specCurves(d3, { sq: "ub", sd: "both" });
  const pin3 = { curves: s3.curves.map(cv => [Array.from(cv[0]), cv[1], cv[2] ? cv[2].slice() : null, cv[3]]),
                 nPerp: s3.nPerp, hiP: s3.hiP, loP: s3.loP, hi: s3.hi, lo: s3.lo, t: 3, kunit: 1 };
  // a pinned PAR lane 6 decades louder than anything live must not move the axis
  for (const cv of pin3.curves.slice(pin3.nPerp))
    for (let i = 1; i < cv[0].length; i += 2) cv[0][i] *= 1e6;
  const cA = recCtx(); C.drawSpectrum(cA, d3, { sq: "ub", sd: "both", fit: "off" }, []);
  const cB = recCtx(); C.drawSpectrum(cB, d3, { sq: "ub", sd: "both", fit: "off" }, [pin3]);
  const ytx = c => c.log.filter(e => e[0] === "text" && /^1e/.test(e[1])).map(e => e[1]).join(",");
  ok("a 1e6x louder pinned PARALLEL ghost does not move the y axis",
     ytx(cA) === ytx(cB) && ytx(cA).length > 0, ytx(cA) + " vs " + ytx(cB));
  ok("  ... pinned parallel ghosts stay dashed",
     pin3.curves.slice(pin3.nPerp).every(cv => String(cv[2]) === "5,3"));
  // ... while a pinned PERPENDICULAR ghost a decade below DOES pull the floor down
  const pinLo = { curves: s3.curves.slice(0, s3.nPerp).map(
                    cv => [cv[0].map((v, i) => (i % 2 ? v * 1e-2 : v)), cv[1], null, cv[3]]),
                  nPerp: s3.nPerp, hiP: s3.hiP * 1e-2, loP: s3.loP * 1e-2,
                  hi: s3.hi * 1e-2, lo: s3.lo * 1e-2, t: 4, kunit: 1 };
  const cC = recCtx(); C.drawSpectrum(cC, d3, { sq: "ub", sd: "both", fit: "off" }, [pinLo]);
  ok("  ... but a pinned PERPENDICULAR ghost two decades below does",
     ytx(cC) !== ytx(cA), ytx(cA) + " -> " + ytx(cC));

  // a PAR-ONLY card: the parallel curves are that card's range-setters, so its ghosts
  // must be too -- otherwise the card falls back to "waiting..." the moment its live
  // data lapses, with ghosts sitting right there
  const oPar = { sq: "ub", sd: "par", fit: "off" };
  const sPar = C.specCurves(d3, oPar);
  const pinPar = { curves: sPar.curves.map(cv => [Array.from(cv[0]), cv[1], cv[2] ? cv[2].slice() : null, cv[3]]),
                   nPerp: sPar.nPerp, hiP: sPar.hiP, loP: sPar.loP, hi: sPar.hi, lo: sPar.lo,
                   t: 5, kunit: 1 };
  ok("a par-only pin carries no perpendicular content at all",
     sPar.nPerp === 0 && sPar.hiP === 0 && sPar.hi > 0);
  const cD = recCtx(); C.drawSpectrum(cD, null, oPar, [pinPar]);
  ok("  ... and a par-only card with par-only ghosts still draws them (not \"waiting\")",
     !waiting(cD) && cD.log.filter(e => e[0] === "stroke" && e[2] === C.PIN_ALPHA[0]).length === 2,
     cD.log.filter(e => e[0] === "stroke" && e[2] === C.PIN_ALPHA[0]).length + " ghost strokes");
  ok("  ... nothing non-finite reached that canvas", !cD.log.some(e => e[0] === "NONFINITE"));
}

// ---------------------------------------------------------------------------
// Sections 4-5 boot real pages. boot() is async and requestAnimationFrame is a no-op on
// the stub, so (as in bootstub.js) the page is only up on the next tick.
// ---------------------------------------------------------------------------
const stubenv = require("./stubenv");
// the page's boot() is async and requestAnimationFrame is a no-op here, so the layout
// only exists on the next macrotask (bootstub.js relies on the same tick)
const boot = async (page, demo) => {
  const e = stubenv(dir, page, demo);
  await new Promise(r => setTimeout(r, 0));
  return e;
};
(async () => {
console.log("4. the four motivating workflows, on a booted rmhd2d page (GATE C)");
// stubenv boots the real page: the buttons pressed below are the ones the header built.
function specCards(env) {
  return env.run("function(){ return cards.chart.filter(c => c.type() === 'spectrum'); }");
}
function headBtn(env, card, label) {
  return env.run("function(c, t){ return c.optEls.filter(s => s.__optBtn && s.__optId === t)[0]; }",
                 card, label);
}
// hand a spectrum card a synthetic readback exactly as the frame loop does
function feed(env, card, nb, kunit, scale) {
  env.run("function(c, nb, ku, s){"
    + " const a = new Float32Array(3 * nb);"
    + " for (let b = 1; b < nb; b++) { const e = s * Math.pow(b, -5/3); a[b] = e; a[nb+b] = 0.8*e; a[2*nb+b] = 0.1*e; }"
    + " c.draw({ perp: a, nb: nb, fshell: [1, 3], parKfac: 1, kunit: ku });"
    + "}", card, nb, kunit, scale === undefined ? 1 : scale);
}
{
  const env = await boot("rmhd2d.html", null);
  const cs = specCards(env);
  ok("rmhd2d boots with a spectrum card", cs.length >= 1, "n=" + cs.length);
  const card = cs[0];
  const bPin = headBtn(env, card, "pin"), bUnpin = headBtn(env, card, "unpin");
  ok("the card has a pin and an unpin button in its header", !!bPin && !!bUnpin);
  ok("  ... pin starts DISABLED (no data yet) and unpin starts HIDDEN",
     bPin.disabled === true && bUnpin.style.display === "none",
     "dis=" + bPin.disabled + " vis=" + bUnpin.style.display);
  ok("  ... and neither contributes a value to optVals()",
     !("pin" in env.run("function(c){ return c.optVals(); }", card)));

  // NOTE: no _optSync() by hand here -- the frame loop only ever calls draw(), so the
  // button has to come alive off that alone
  feed(env, card, 24, 1);
  ok("one readback (draw() alone) enables the pin button", bPin.disabled === false);

  // WORKFLOW 1 (zeroth law): pin, then halve the dissipation slider
  bPin.onclick();
  ok("workflow 1 (zeroth law): pin -> 1 ghost", card.pins.length === 1);
  ok("  ... unpin became visible", bUnpin.style.display === "");
  const t0 = card.pins[0].t;
  env.run("function(){ const e = document.getElementById('rDiss');"
    + " e.value = String(parseFloat(e.value) - 0.3); syncLabels(); }");
  feed(env, card, 24, 1, 4);
  ok("  ... the ghost survives the diss change and the redraws it triggers",
     card.pins.length === 1 && card.pins[0].t === t0);
  ok("  ... and it did NOT follow the live curves up",
     card.pins[0].curves[0][0][1] < 1.5 * Math.pow(1, -5 / 3),
     "y1 = " + card.pins[0].curves[0][0][1]);

  // WORKFLOW 2 (forcing amplitude): change eps with a pin down
  env.run("function(){ const e = document.getElementById('rEpsP');"
    + " e.value = String(parseFloat(e.value) - 0.5); syncLabels(); }");
  feed(env, card, 24, 1, 9);
  ok("workflow 2 (forcing amplitude): the ghost survives an eps change",
     card.pins.length === 1 && card.pins[0].t === t0);

  // the 4-pin cap and the refusal
  for (let i = 0; i < 5; i++) bPin.onclick();
  ok("workflow 4 (inverse cascade): the 5th press is refused at PIN_MAX",
     card.pins.length === 4, "n=" + card.pins.length);
  ok("  ... and it said so", /at most 4 pinned spectra/.test(env.getEl("status").textContent),
     env.getEl("status").textContent);

  // unpin clears them all in one press
  bUnpin.onclick();
  ok("unpin clears every ghost in one press", card.pins.length === 0);
  ok("  ... and hides itself again", bUnpin.style.display === "none");

  // retyping the card drops its pins and its cache
  feed(env, card, 24, 1); bPin.onclick();
  ok("a re-pinned card has 1 ghost again", card.pins.length === 1);
  env.run("function(c){ c.selType.value = 'energy'; c.selType.onchange(); }", card);
  ok("retyping to another chart type drops the pins and the cache",
     card.pins.length === 0 && card.lastData === null);
  env.run("function(c){ c.selType.value = 'spectrum'; c.selType.onchange(); }", card);

  // a second spectrum card carries its OWN pins
  const c2 = env.run("function(){ const c = addChartCard('spectrum'); cardsSync(); return c; }");
  feed(env, c2, 24, 1);
  const b2 = headBtn(env, c2, "pin");
  b2.onclick(); b2.onclick();
  const card2 = specCards(env)[0];
  ok("pins are PER CARD (2 on the new one, 0 on the old)",
     c2.pins.length === 2 && card2.pins.length === 0,
     c2.pins.length + " / " + card2.pins.length);

  // WORKFLOW 3 (universality): pin, switch preset, the ghosts transplant
  ok("stub boot raised no failures so far", env.fails.length === 0, env.fails.join(" | "));
  env.fails.length = 0;
}
{
  // ?demo=decay -> pin -> pick the forced preset: cardsLayout rebuilds every card
  const env = await boot("rmhd2d.html", "decay");
  const card = specCards(env)[0];
  feed(env, card, 24, 1);
  const bPin = headBtn(env, card, "pin");
  bPin.onclick(); bPin.onclick();
  ok("workflow 3 (universality): 2 ghosts pinned in ?demo=decay", card.pins.length === 2);
  const before = JSON.stringify(card.pins[0].curves[0][0]);
  const keys = env.run("function(){ return document.getElementById('selPreset').options.map(o => o.value); }");
  const cur = env.getEl("selPreset").value;
  const forced = keys.filter(k => k !== cur && /forc|turb|kolm/i.test(k))[0]
              || keys.filter(k => k !== cur)[0];
  ok("  ... the page really opened on the decay preset", cur === "decay", cur);
  env.run("function(k){ const s = document.getElementById('selPreset'); s.value = k; s.onchange(); }", forced);
  const after = specCards(env);
  ok("  ... the preset switch rebuilt the layout (new card objects)",
     after.length >= 1 && after.indexOf(card) < 0, "n=" + after.length);
  ok("  ... and the ghosts were TRANSPLANTED onto the incoming spectrum card",
     after[0].pins.length === 2 && JSON.stringify(after[0].pins[0].curves[0][0]) === before,
     "preset=" + forced + " n=" + after[0].pins.length);
  ok("  ... the lastData cache came with them (a pin is possible before new data)",
     after[0].lastData !== null && headBtn(env, after[0], "pin").disabled === false);
  ok("  ... unpin is visible on the incoming card",
     headBtn(env, after[0], "unpin").style.display === "");
  // the physical-k registration must survive the transplant too: the pin keeps the kunit
  // it was TAKEN under, so a preset that changes the box moves the ghost, not the axis
  ok("  ... the transplanted pins kept their pin-time kunit",
     after[0].pins.every(p => p.kunit === 1), after[0].pins.map(p => p.kunit).join(","));
  const wide = env.run("function(c){"
    + " const a = pinDraw(c.pins, 1)[0].curves[0][0], b = pinDraw(c.pins, 0.5)[0].curves[0][0];"
    + " return b.every((v, i) => (i % 2 ? v === a[i] : Math.abs(v - 2 * a[i]) < 1e-12)); }", after[0]);
  ok("  ... and a halved live kunit doubles their x on the very next draw", wide === true);
  // a card with ghosts but no live data draws them instead of "waiting"
  env.run("function(c){ c.lastData = null; c.draw(null); }", after[0]);
  ok("  ... and a ghosts-only card still draws (no throw, ghosts kept)", after[0].pins.length === 2);
  ok("stub boot raised no failures", env.fails.length === 0, env.fails.join(" | "));
}
{
  // pin at t = 0 on a forced run: the pin is legal before the clock has moved
  const env = await boot("rmhd2d.html", null);
  const card = specCards(env)[0];
  feed(env, card, 24, 1);
  headBtn(env, card, "pin").onclick();
  ok("a pin taken before any history reports t = 0", card.pins[0].t === 0, "t=" + card.pins[0].t);
  ok("stub boot raised no failures", env.fails.length === 0, env.fails.join(" | "));
}

// ---------------------------------------------------------------------------
console.log("5. the 3D check: sd = both, dashed pinned-parallel ghosts (GATE C)");
// ---------------------------------------------------------------------------
{
  const env = await boot("rmhd3d.html", null);
  const cs = specCards(env);
  ok("rmhd3d boots with a spectrum card", cs.length >= 1, "n=" + cs.length);
  const card = cs[0];
  ok("  ... and it offers the perp/par selector", "sd" in env.run("function(c){ return c.optVals(); }", card));
  env.run("function(c){ c.optEls.filter(s => s.__optId === 'sd')[0].value = 'both'; }", card);
  env.run("function(c, nb, nzb){"
    + " const a = new Float32Array(3 * nb), p = new Float32Array(3 * nzb);"
    + " for (let b = 1; b < nb; b++) { const e = Math.pow(b, -5/3); a[b] = e; a[nb+b] = 0.8*e; a[2*nb+b] = 0.1*e; }"
    + " for (let b = 0; b < nzb; b++) { const e = 1e6 * Math.pow(b+1, -2); p[b] = e; p[nzb+b] = e; p[2*nzb+b] = 0; }"
    + " c.draw({ perp: a, nb: nb, fshell: [1,3], par: p, parKfac: 1.5, kunit: 1 });"
    + "}", card, 20, 8);
  headBtn(env, card, "pin").onclick();
  const p = card.pins[0];
  ok("3D pin with sd = both snapshots perp AND par curves",
     p.curves.length === 4 && p.nPerp === 2, "n=" + p.curves.length + " nPerp=" + p.nPerp);
  ok("  ... the pinned parallel pair is DASHED and the perpendicular pair is not",
     p.curves[0][2] === null && String(p.curves[2][2]) === "5,3");
  ok("  ... the pinned par lane is 1e6x the perp one, and hiP ignores it",
     p.hiP < 2 && p.curves[2][0][1] > 1e5, "hiP=" + p.hiP);
  // switching the live view to perp-only must not touch the pin (it is a record of a moment)
  env.run("function(c){ const s = c.optEls.filter(x => x.__optId === 'sd')[0];"
    + " s.value = 'perp'; s.onchange(); }", card);
  ok("  ... switching the live sd to perp leaves the pinned par ghosts alone",
     card.pins[0].curves.length === 4);
  ok("stub boot raised no failures", env.fails.length === 0, env.fails.join(" | "));
}

console.log(bad ? "\n" + bad + " FAILURE(S)" : "\nall checks passed");
process.exit(bad ? 1 : 0);
})();
