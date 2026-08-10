// ANISO_PLAN gates: the k∥(k⊥)/k⊥ matching (Phase A) and the card it feeds (Phase B).
// Usage: node checkaniso.js [webgpu-dir]           exit code 1 on any failure
//
// Sections 1-6 are the plan's six groups, run against the REAL common.js functions in a
// bare vm (no page, no canvas): slope recovery, the density trap as a regression guard,
// uniqueness/robustness, the no-extrapolation rule, the degenerates, and the level
// window. Section 7 boots rmhd3d / rmhd2d on stubenv and drives the card itself -- the
// type selector, the option set, and the field-line readback gate, which is the one
// functional per-app edit the plan allows.
//
// On the synthetic spectra of sections 1-2. The matching is exact only for CONTINUOUS
// power-law tails; on binned ones two known instrument effects survive, and the tolerances
// here are sized to them rather than pretended away:
//   * TRUNCATION. Q(k) = sum_{k' >= k} E(k') on a grid that stops at k_max is short of the
//     continuum tail by ~Q(k_max), which biases the inferred k upward near the top of the
//     window. It is suppressed by giving the synthetic PERPENDICULAR spectrum a grid far
//     wider than the window the knee rule then picks out of it (nb = 32768, window ends at
//     k = 251, where the 4-decade knee falls for k^-5/3).
//   * DISCRETENESS. sum_{k' >= k} k'^-2 is ~1/(k - 1/2), not 1/k, so a k∥ of a few bins
//     reads high by half a bin. It is suppressed by placing the matched k∥ range in the
//     TOP decade of the parallel grid's admissible window, which is what the parallel
//     spectrum's amplitude does here: a constant factor on E∥ multiplies Q∥ and therefore
//     slides k∥ along, i.e. it moves the curve VERTICALLY and cannot touch its slope --
//     the same argument the code comment makes about the fl/coordinate normalizations.
// What is left is ~0.01 of slope, an order of magnitude inside the gap between the -1/3
// this must recover and the -1/6 the density trap would return.
"use strict";
const fs = require("fs"), vm = require("vm"), path = require("path");
const dir = process.argv[2] || path.join(__dirname, "..");
let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};

// ---------------------------------------------------------------------------
// a bare vm holding common.js, for the pure functions (the checkpin idiom)
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
const C = Object.assign({}, vm.runInContext(
  "({ anisoCurves, drawAniso, ANISO_NLEV, ANISO_LANES, _anisoLeg, _anisoTail, _anisoAt,"
  + " _anisoWin, _anisoPeak, specKnee, specFloor, fitKA, fitIndex, fitLabel, FIT_FRACS,"
  + " FIT_FRACS_ANISO, FIT_SNAP, SPEC_KNEE, CHART_TYPES, COL })", sandbox));

// ---- synthetic spectra -----------------------------------------------------
// a three-lane [E_u | E_b | H_c] bin stack of `n` bins, E_u carrying f(bin) and the other
// two lanes zero, so the default `tot` lane reads exactly f
function mk(n, f) {
  const a = new Float32Array(3 * n);
  for (let j = 0; j < n; j++) { a[j] = f(j); a[n + j] = 0; a[2 * n + j] = 0; }
  return a;
}
// ... and one with all three lanes live, so E+- = E_u + E_b +- H_c can be checked
function mk3(n, f) {
  const a = new Float32Array(3 * n);
  for (let j = 0; j < n; j++) { const v = f(j); a[j] = 0.6 * v; a[n + j] = 0.4 * v; a[2 * n + j] = 0.25 * v; }
  return a;
}
const PERP_N = 32768;                       // see the truncation note at the top
const PAR_N = 1024;
const perpLaw = p => mk(PERP_N, j => (j === 0 ? 0 : Math.pow(j, p)));
const parLaw = (A, p) => mk(PAR_N, j => A * Math.pow(j + 1, p));
// the GS95 pair (E⊥ ~ k^-5/3, E∥ ~ k∥^-2 -> ratio slope -1/3) and the aligned pair
// (E⊥ ~ k^-3/2 -> -1/2). The parallel amplitudes place the matched k∥ in the top decade.
const CASE_GS = { perp: perpLaw(-5 / 3), nb: PERP_N, par: parLaw(3, -2),
                  parFL: parLaw(3, -2), parKfac: 1, fshell: [1, 5] };
const CASE_AL = { perp: perpLaw(-3 / 2), nb: PERP_N, par: parLaw(8, -2),
                  parFL: parLaw(8, -2), parKfac: 1, fshell: [1, 5] };
// least-squares slope of log(y) vs log(x) over a flat (x, y) point list
function slope(pts) {
  let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < pts.length; i += 2) {
    const x = Math.log(pts[i]), y = Math.log(pts[i + 1]);
    n++; sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  return n < 2 ? NaN : (n * sxy - sx * sy) / (n * sxx - sx * sx);
}
const allFinite = pts => pts.every(v => isFinite(v) && v > 0);

// ---------------------------------------------------------------------------
console.log("1. slope recovery: does the matching return the exponent that went in?");
// ---------------------------------------------------------------------------
const SLOPE_TOL = 0.03;
{
  for (const [nm, d, want] of [["GS95 (-5/3 vs -2)", CASE_GS, -1 / 3],
                               ["aligned (-3/2 vs -2)", CASE_AL, -1 / 2]]) {
    for (const ad of ["z", "fl"]) {
      const A = C.anisoCurves(d, { aq: "tot", ad });
      const s = slope(A.curves.length ? A.curves[0][0] : []);
      ok(nm + ", k∥ " + (ad === "z" ? "along z" : "along the field line")
         + " -> ratio slope " + want.toFixed(4),
         Math.abs(s - want) < SLOPE_TOL, "got " + s.toFixed(4) + " (tol " + SLOPE_TOL + ")");
    }
  }
  // both curves at once: same spectra on both parallel lanes -> two identical curves
  const B = C.anisoCurves(CASE_GS, { aq: "tot", ad: "both" });
  ok("  ... \"both\" draws two curves, global first (nGlob = 1)",
     B.curves.length === 2 && B.nGlob === 1 && B.curves[0][2] === null &&
     String(B.curves[1][2]) === "5,3",
     "n=" + B.curves.length + " nGlob=" + B.nGlob);
  ok("  ... global solid + field line dashed, distinct colours, both labelled",
     B.curves[0][1] !== B.curves[1][1] && /k∥z/.test(B.curves[0][3]) && /k∥B/.test(B.curves[1][3]),
     B.curves.map(cv => cv[3]).join(" | "));
  ok("  ... " + C.ANISO_NLEV + " levels -> " + C.ANISO_NLEV + " points, in k⊥ order, all finite",
     B.curves.every(cv => cv[0].length === 2 * C.ANISO_NLEV && allFinite(cv[0]) &&
       cv[0].every((v, i) => i < 2 || i % 2 || v >= cv[0][i - 2])),
     "n=" + B.curves[0][0].length / 2);
  // hi/lo really are the drawn extremes
  let hi = 0, lo = Infinity;
  for (const cv of B.curves) for (let i = 1; i < cv[0].length; i += 2) {
    hi = Math.max(hi, cv[0][i]); lo = Math.min(lo, cv[0][i]);
  }
  ok("  ... hi/lo are the extremes of the drawn ratios", B.hi === hi && B.lo === lo,
     B.lo.toExponential(3) + " .. " + B.hi.toExponential(3));
  // the Elsasser lanes: E+- = E_u + E_b +- H_c, so a stack with H_c > 0 gives a LOUDER
  // E+ than E- and the same exponent for both (the lane is a constant factor)
  const dl = { perp: mk3(PERP_N, j => (j === 0 ? 0 : Math.pow(j, -5 / 3))), nb: PERP_N,
               par: mk3(PAR_N, j => 3 * Math.pow(j + 1, -2)), parKfac: 1, fshell: [1, 5] };
  const sp = slope(C.anisoCurves(dl, { aq: "zp", ad: "z" }).curves[0][0]);
  const sm = slope(C.anisoCurves(dl, { aq: "zm", ad: "z" }).curves[0][0]);
  ok("  ... the E+ and E- lanes recover the same exponent",
     Math.abs(sp + 1 / 3) < SLOPE_TOL && Math.abs(sm + 1 / 3) < SLOPE_TOL,
     "E+ " + sp.toFixed(4) + "  E- " + sm.toFixed(4));
  ok("  ... an unknown aq falls back to the total lane",
     JSON.stringify(C.anisoCurves(dl, { aq: "junk", ad: "z" }).curves[0][0]) ===
     JSON.stringify(C.anisoCurves(dl, { aq: "tot", ad: "z" }).curves[0][0]));
  // a constant factor on the parallel spectrum moves the curve VERTICALLY and nothing else
  // -- the argument the code makes for matching the fl tails without renormalizing them
  const dq = Object.assign({}, CASE_GS, { par: parLaw(30, -2) });
  const sq = slope(C.anisoCurves(dq, { aq: "tot", ad: "z" }).curves[0][0]);
  ok("  ... a 10x parallel normalization changes the level, not the slope",
     Math.abs(sq + 1 / 3) < SLOPE_TOL, "got " + sq.toFixed(4));
}

// ---------------------------------------------------------------------------
console.log("2. the density trap: matching E(k) instead of the tails returns -1/6");
// ---------------------------------------------------------------------------
// The regression guard. An implementation that matched spectral DENSITIES on exactly the
// inputs of case 1 would report k∥ ~ k⊥^5/6, i.e. a ratio slope of -1/6 -- a law
// manufactured by the method. That wrong answer is COMPUTED here, from the same bins, so
// the assertion is against the real trap and not against a remembered number.
{
  const lane = (u, b, h) => u + b;
  const perp = C._anisoLeg(CASE_GS.perp, CASE_GS.nb, 0, 1, lane);
  const par = C._anisoLeg(CASE_GS.par, PAR_N, 1, 1, lane);
  // invert a monotone-falling (k, E) list at the level E*, log-log, same interpolation
  const at = (pts, E) => {
    for (let i = 0; i + 3 < pts.length; i += 2) {
      if (pts[i + 1] >= E && pts[i + 3] <= E) {
        const d = Math.log(pts[i + 1] / pts[i + 3]);
        return d > 0 ? pts[i] * Math.pow(pts[i + 2] / pts[i], Math.log(pts[i + 1] / E) / d) : pts[i];
      }
    }
    return 0;
  };
  const dens = [];
  for (let m = 0; m < 16; m++) {
    const E = Math.pow(10, -1 - 3 * m / 15);          // a decade ladder inside both ranges
    const kp = at(perp, E), kz = at(par, E);
    if (kp > 0 && kz > 0) dens.push(kp, kz / kp);
  }
  const sd = slope(dens);
  ok("a density matcher really does return the 5/6 law on these bins",
     dens.length >= 20 && Math.abs(sd + 1 / 6) < 0.02,
     "ratio slope " + sd.toFixed(4) + " (5/6 - 1 = -1/6), " + dens.length / 2 + " points");
  const st = slope(C.anisoCurves(CASE_GS, { aq: "tot", ad: "z" }).curves[0][0]);
  ok("  ... and the TAIL matching does not: it is -1/3, not -1/6",
     Math.abs(st + 1 / 3) < SLOPE_TOL && Math.abs(st + 1 / 6) > 4 * SLOPE_TOL,
     "tails " + st.toFixed(4) + " vs densities " + sd.toFixed(4));
  ok("  ... the two answers are more than a decade of slope apart (the guard has teeth)",
     Math.abs(st - sd) > 0.1, "|difference| = " + Math.abs(st - sd).toFixed(4));
}

// ---------------------------------------------------------------------------
console.log("3. uniqueness and robustness: noise and a bump on the bins");
// ---------------------------------------------------------------------------
// Live spectra are neither smooth nor monotone. What keeps the matching single-valued is
// that the CUMULATIVE TAIL is monotone by construction, whatever the bins do -- so that
// is what is asserted here, by brute force, together with the points surviving intact.
{
  // deterministic multiplicative noise (no RNG anywhere near this file) plus a fat
  // non-monotone bump sitting on top of the perpendicular inertial range
  const wob = j => 1 + 0.6 * Math.sin(1.7 * j) * Math.cos(0.31 * j + 1);
  const bump = j => 1 + 8 * Math.exp(-Math.pow((Math.log(Math.max(j, 1)) - Math.log(40)) / 0.25, 2));
  const d = Object.assign({}, CASE_GS, {
    perp: mk(PERP_N, j => (j === 0 ? 0 : Math.pow(j, -5 / 3) * wob(j) * bump(j))),
    par: mk(PAR_N, j => 3 * Math.pow(j + 1, -2) * wob(3 * j))
  });
  const lane = (u, b, h) => u + b;
  for (const [nm, src, n, j0] of [["perpendicular", d.perp, PERP_N, 0], ["parallel", d.par, PAR_N, 1]]) {
    const pts = C._anisoLeg(src, n, j0, 1, lane);
    const T = C._anisoTail(pts);
    let mono = true, nonMonoBins = 0;
    for (let i = 0; i + 1 < T.n; i++) if (!(T.qs[i] > T.qs[i + 1])) mono = false;
    for (let i = 0; i + 3 < pts.length; i += 2) if (pts[i + 3] > pts[i + 1]) nonMonoBins++;
    ok("the " + nm + " tail is strictly decreasing though " + nonMonoBins + " bins rise",
       mono && nonMonoBins > 10, "n=" + T.n);
    // ... hence exactly ONE crossing per level, counted the dumb way
    let worst = 0;
    for (let m = 0; m < 24; m++) {
      const Q = T.qs[0] * Math.pow(T.qs[T.n - 1] / T.qs[0], (m + 0.5) / 24);
      let cross = 0;
      for (let i = 0; i + 1 < T.n; i++) if ((T.qs[i] >= Q) !== (T.qs[i + 1] >= Q)) cross++;
      worst = Math.max(worst, Math.abs(cross - 1));
    }
    ok("  ... and exactly one crossing at each of 24 levels", worst === 0);
  }
  const A = C.anisoCurves(d, { aq: "tot", ad: "both" });
  ok("noisy + bumpy bins still give finite, ordered, full-length curves",
     A.curves.length === 2 && A.curves.every(cv => cv[0].length === 2 * C.ANISO_NLEV && allFinite(cv[0])),
     A.curves.map(cv => cv[0].length / 2).join(","));
  // No slope assertion on the BUMPED spectrum: a bump is real energy at a real scale and
  // is entitled to move the anisotropy (that 9x one drags the recovered slope to ~ -0.11,
  // and rightly so). What must survive noise alone -- the ~30% bin-to-bin scatter of a
  // live 300 ms readback -- is the exponent, and it does: the tail integrates the scatter
  // away, which is the other half of the reason the method is built on tails.
  const dn = Object.assign({}, CASE_GS, {
    perp: mk(PERP_N, j => (j === 0 ? 0 : Math.pow(j, -5 / 3) * wob(j))),
    par: mk(PAR_N, j => 3 * Math.pow(j + 1, -2) * wob(3 * j))
  });
  const s = slope(C.anisoCurves(dn, { aq: "tot", ad: "z" }).curves[0][0]);
  ok("  ... and with the bump removed, +-60% bin noise leaves the exponent at -1/3",
     Math.abs(s + 1 / 3) < 0.05, "got " + s.toFixed(4));
}

// ---------------------------------------------------------------------------
console.log("4. no extrapolation past the grid");
// ---------------------------------------------------------------------------
{
  const lane = (u, b, h) => u + b;
  const T = C._anisoTail(C._anisoLeg(parLaw(3, -2), PAR_N, 1, 1, lane));
  ok("_anisoAt refuses a level above the tail's FIRST bin (returns 0, invents nothing)",
     C._anisoAt(T, T.qs[0] * 1.000001) === 0 && C._anisoAt(T, 1e9) === 0);
  ok("  ... and one below its LAST bin", C._anisoAt(T, T.qs[T.n - 1] * 0.999999) === 0 &&
     C._anisoAt(T, 1e-30) === 0);
  ok("  ... the two ends themselves invert to the two end bins",
     C._anisoAt(T, T.qs[0]) === T.ks[0] && C._anisoAt(T, T.qs[T.n - 1]) === T.ks[T.n - 1]);
  ok("  ... an exact bin level inverts to that bin, and the midpoint lands between",
     Math.abs(C._anisoAt(T, T.qs[7]) - T.ks[7]) < 1e-9 &&
     C._anisoAt(T, Math.sqrt(T.qs[7] * T.qs[8])) > T.ks[7] &&
     C._anisoAt(T, Math.sqrt(T.qs[7] * T.qs[8])) < T.ks[8]);
  ok("  ... a degenerate (empty) tail inverts to nothing",
     C._anisoAt(C._anisoTail([]), 1) === 0);
  // a SHORT parallel grid: the levels the perpendicular leg would like are mostly below
  // Q∥(first bin) or above Q∥(last bin), and must simply not appear
  const dshort = Object.assign({}, CASE_GS, { par: mk(6, j => 3 * Math.pow(j + 1, -2)) });
  const A = C.anisoCurves(dshort, { aq: "tot", ad: "z" });
  const pts = A.curves.length ? A.curves[0][0] : [];
  let inside = true;
  for (let i = 0; i < pts.length; i += 2) {
    const kz = pts[i] * pts[i + 1];
    if (!(kz >= 1 - 1e-12 && kz <= 6 + 1e-12)) inside = false;
    if (!(pts[i] >= 1 && pts[i] <= dshort.nb - 1)) inside = false;
  }
  ok("a 6-bin parallel spectrum yields only points inside its own 1..6 range",
     pts.length > 0 && inside, pts.length / 2 + " points, k∥ " +
     (pts.length ? (pts[0] * pts[1]).toFixed(3) + ".." +
       (pts[pts.length - 2] * pts[pts.length - 1]).toFixed(3) : "-"));
  ok("  ... and no k⊥ was extrapolated past the perpendicular grid either",
     pts.every((v, i) => i % 2 === 1 || (v >= 1 && v < PERP_N)));
  // a parallel spectrum that is LOUDER than the perpendicular one everywhere: the level
  // bands do not overlap at all, so the honest answer is no curve
  const dloud = Object.assign({}, CASE_GS, { par: mk(4, () => 1e12) });
  const L = C.anisoCurves(dloud, { aq: "tot", ad: "z" });
  ok("  ... non-overlapping level bands give no curve rather than a fabricated one",
     L.curves.length === 0 && L.hi === 0, "n=" + L.curves.length);
}

// ---------------------------------------------------------------------------
console.log("5. degenerates: empty, zero, one bin, missing legs, junk options");
// ---------------------------------------------------------------------------
{
  const empty = A => A.curves.length === 0 && A.hi === 0 && A.lo === Infinity;
  ok("no data at all -> no curves (the \"waiting…\" path)", empty(C.anisoCurves(null, null)));
  ok("  ... an empty options object is the default view", (() => {
    const a = C.anisoCurves(CASE_GS, {}), b = C.anisoCurves(CASE_GS, { aq: "tot", ad: "both" });
    return a.curves.length === b.curves.length && a.curves.length === 2;
  })());
  ok("an all-zero perpendicular stack -> no curves",
     empty(C.anisoCurves({ perp: mk(64, () => 0), nb: 64, par: parLaw(3, -2), parKfac: 1 }, {})));
  ok("a one-bin perpendicular stack -> no curves (one bin brackets nothing)",
     empty(C.anisoCurves({ perp: mk(2, () => 1), nb: 2, par: parLaw(3, -2), parKfac: 1 }, {})));
  ok("nb = 0 / 1 / negative -> no curves and no throw",
     [0, 1, -3].every(nb => empty(C.anisoCurves({ perp: CASE_GS.perp, nb, par: CASE_GS.par, parKfac: 1 }, {}))));
  {
    const A = C.anisoCurves(Object.assign({}, CASE_GS, { parFL: null }), { ad: "both" });
    ok("parFL null -> the GLOBAL curve alone (the fl sampler has not landed yet)",
       A.curves.length === 1 && A.nGlob === 1 && /k∥z/.test(A.curves[0][3]));
  }
  {
    const A = C.anisoCurves(Object.assign({}, CASE_GS, { par: null }), { ad: "both" });
    ok("par null -> the FIELD-LINE curve alone, and nGlob says so",
       A.curves.length === 1 && A.nGlob === 0 && /k∥B/.test(A.curves[0][3]));
  }
  ok("both parallel legs null -> no curves",
     empty(C.anisoCurves(Object.assign({}, CASE_GS, { par: null, parFL: null }), { ad: "both" })));
  ok("ad = \"z\" ignores parFL entirely, ad = \"fl\" ignores par", (() => {
    const z = C.anisoCurves(Object.assign({}, CASE_GS, { parFL: parLaw(300, -2) }), { ad: "z" });
    const f = C.anisoCurves(Object.assign({}, CASE_GS, { par: parLaw(300, -2) }), { ad: "fl" });
    const ref = C.anisoCurves(CASE_GS, { ad: "z" });
    return z.curves.length === 1 && f.curves.length === 1 &&
      JSON.stringify(z.curves[0][0]) === JSON.stringify(ref.curves[0][0]) &&
      JSON.stringify(f.curves[0][0]) === JSON.stringify(ref.curves[0][0]);
  })());
  ok("a tiny nzb (1 bin) is dropped, not matched",
     empty(C.anisoCurves(Object.assign({}, CASE_GS, { par: mk(1, () => 1), parFL: null }), { ad: "both" })));
  ok("NaN / Inf / negative bins are dropped like everywhere else", (() => {
    const p = mk(PERP_N, j => (j === 0 ? 0 : Math.pow(j, -5 / 3)));
    p[7] = NaN; p[9] = Infinity; p[11] = -1;
    const A = C.anisoCurves(Object.assign({}, CASE_GS, { perp: p }), { ad: "z" });
    return A.curves.length === 1 && allFinite(A.curves[0][0]);
  })());
  ok("a missing parKfac defaults to 1 (and a real one scales k∥)", (() => {
    const a = C.anisoCurves(Object.assign({}, CASE_GS, { parKfac: undefined }), { ad: "z" });
    const b = C.anisoCurves(CASE_GS, { ad: "z" });
    return JSON.stringify(a.curves[0][0]) === JSON.stringify(b.curves[0][0]);
  })());
  // drawAniso must survive every one of those against a recording context, with nothing
  // non-finite reaching the canvas and "waiting…" exactly when there is nothing to draw
  const recCtx = () => {
    const log = [];
    const o = { fillStyle: "", strokeStyle: "", lineWidth: 1, globalAlpha: 1,
                font: "10px x", textAlign: "left", textBaseline: "alphabetic" };
    for (const m of ["clearRect", "strokeRect", "beginPath", "moveTo", "lineTo", "stroke",
                     "fill", "clip", "save", "restore", "setLineDash", "rect", "fillRect"]) {
      o[m] = (...a) => {
        for (const v of a) if (typeof v === "number" && !isFinite(v)) log.push(["NONFINITE", m]);
        if (m === "stroke") log.push(["stroke", o.strokeStyle, o.lineWidth]);
      };
    }
    o.measureText = t => ({ width: 6.2 * t.length, actualBoundingBoxAscent: 7.2,
                            actualBoundingBoxDescent: 0 });
    o.fillText = t => log.push(["text", t]);
    o.log = log;
    return o;
  };
  const waiting = c => c.log.some(e => e[0] === "text" && /waiting/.test(e[1]));
  const DEGEN = [null, {}, { nb: 4 }, { perp: mk(64, () => 0), nb: 64 },
                 { perp: CASE_GS.perp, nb: 2, par: CASE_GS.par, parKfac: 1 },
                 Object.assign({}, CASE_GS, { par: null, parFL: null })];
  let anyNon = false, allWait = true;
  for (const d of DEGEN) {
    const c = recCtx(); C.drawAniso(c, d, { fit: "pin" });
    if (c.log.some(e => e[0] === "NONFINITE")) anyNon = true;
    if (!waiting(c)) allWait = false;
  }
  ok("drawAniso draws \"waiting…\" on all " + DEGEN.length + " degenerate data objects", allWait);
  ok("  ... and nothing non-finite ever reached the canvas", !anyNon);
  const live = [];
  for (const ad of ["both", "z", "fl"]) for (const fit of ["pin", "amp", "off"]) {
    const c = recCtx();
    C.drawAniso(c, CASE_GS, { aq: "tot", ad, fit, fitp: -0.333, fita: fit === "amp" ? "2.5" : "" });
    live.push(c);
  }
  ok("  ... and every live (ad x fit) combination draws instead, all finite",
     live.every(c => !waiting(c) && !c.log.some(e => e[0] === "NONFINITE")),
     live.length + " combinations");
  ok("  ... the fit line is drawn for pin/amp and absent for off", (() => {
    const leg = c => c.log.filter(e => e[0] === "text").map(e => e[1]).join("|");
    return /k⊥\^-1\/3/.test(leg(live[0])) && !/k⊥\^/.test(leg(live[2]));
  })(), live.map(c => c.log.filter(e => e[0] === "text" && /k⊥\^/.test(e[1])).length).join(","));
  ok("  ... drawAniso with no context is a no-op, not a throw",
     C.drawAniso(null, CASE_GS, {}) === undefined);
}

// ---------------------------------------------------------------------------
console.log("6. the level window: the kA anchor and the shared dissipation knee");
// ---------------------------------------------------------------------------
{
  // a spectrum with a real hyper-dissipative tail: k^-5/3 up to kd, then a fourth-power
  // exponential fall, which is what the knee rule was written for
  const KD = 300;
  const dd = Object.assign({}, CASE_GS, {
    perp: mk(PERP_N, j => (j === 0 ? 0 : Math.pow(j, -5 / 3) * Math.exp(-Math.pow(j / KD, 4))))
  });
  const lane = (u, b, h) => u + b;
  const perpPts = C._anisoLeg(dd.perp, dd.nb, 0, 1, lane);
  const knee = C.specKnee([[perpPts]], C._anisoPeak(perpPts));
  ok("the perpendicular spectrum has a knee, found by the SHARED specKnee",
     isFinite(knee) && knee > 100 && knee < KD, "k_d = " + knee);
  for (const kA of [3, 8, 20]) {
    const A = C.anisoCurves(Object.assign({}, dd, { fshell: [1, kA] }), { aq: "tot", ad: "z" });
    const pts = A.curves.length ? A.curves[0][0] : [];
    const lo = pts[0], hiK = pts[pts.length - 2];
    ok("  ... with fshell [1," + kA + "] every point sits in [kA, k_d)",
       pts.length > 0 && lo >= C.fitKA(dd.nb, [1, kA]) - 1e-9 && hiK < knee,
       "k⊥ " + lo.toFixed(2) + ".." + hiK.toFixed(2) + ", kA = " + C.fitKA(dd.nb, [1, kA]));
  }
  ok("  ... and the dissipation range is excluded: no point at or beyond the knee", (() => {
    const A = C.anisoCurves(dd, { aq: "tot", ad: "z" });
    return A.curves[0][0].every((v, i) => i % 2 === 1 || v < knee);
  })());
  ok("  ... a spectrum with NO knee runs to its last resolved bin instead", (() => {
    const flat = { perp: mk(64, j => (j === 0 ? 0 : Math.pow(j, -1))), nb: 64,
                   par: mk(32, j => 0.3 * Math.pow(j + 1, -1)), parKfac: 1, fshell: [1, 3] };
    const pts = C.anisoCurves(flat, { ad: "z" }).curves[0][0];
    return !isFinite(C.specKnee([[C._anisoLeg(flat.perp, 64, 0, 1, lane)]], 1)) &&
           pts[pts.length - 2] > 60;
  })());
  ok("fitKA is the fit line's own anchor rule, shared with the spectrum card",
     C.fitKA(128, [1, 3]) === 3 && C.fitKA(128, [1, 1]) === 2 && C.fitKA(4, [1, 30]) === 3,
     "3 / 2 / 3");
  ok("_anisoWin returns null when no bin sits inside the window",
     C._anisoWin(C._anisoTail(C._anisoLeg(parLaw(3, -2), PAR_N, 1, 1, lane)), 1e9, Infinity) === null);
  // the knee factoring must not have moved the spectrum card's floor
  ok("specFloor still answers on the factored knee (a decade-low curve drops the floor)", (() => {
    const hiC = [[[1, 1, 2, 1, 3, 1]]], loC = [[[1, 1e-6, 2, 1e-6, 3, 1e-6]]];
    return C.specFloor(hiC, 1, 1) >= C.specFloor(hiC.concat(loC), 1, 1e-6);
  })());
}

// ---------------------------------------------------------------------------
console.log("7. the card: type entry, fit tables, and the field-line readback gate");
// ---------------------------------------------------------------------------
{
  const T = C.CHART_TYPES.aniso;
  ok("CHART_TYPES carries an `aniso` entry riding the SPECTRUM readback",
     !!T && T.src === "spectrum", T ? "src=" + T.src : "missing");
  ok("  ... available in 3D and absent in 2D (the mirror of island/mode)",
     T.avail({ zslice: true }) === true && !T.avail({}) &&
     C.CHART_TYPES.island.avail({}) === true && !C.CHART_TYPES.island.avail({ zslice: true }));
  ok("  ... it declares no zslice of its own (the spectra are whole-box)", !T.zslice);
  const ids = T.opts({ zslice: true }).map(s => s.id).join(",");
  ok("  ... its options are aq, ad and the fit trio", ids === "aq,ad,fit,fitp,fita", ids);
  const os = T.opts({ zslice: true });
  ok("  ... the defaults are the FIRST option of each list: total energy, both curves",
     os[0].o[0][0] === "tot" && os[1].o[0][0] === "both" && os[2].o[0][0] === "pin");
  ok("  ... the index box defaults to -0.333, which snaps to exactly -1/3",
     C.fitIndex(String(os[3].v), C.FIT_FRACS_ANISO) === -1 / 3, String(os[3].v));
  ok("  ... the two number boxes hide with the fit line, as on the spectrum card",
     os[3].vis({ fit: "off" }) === false && os[3].vis({ fit: "pin" }) === true &&
     os[4].vis({ fit: "pin" }) === false && os[4].vis({ fit: "amp" }) === true);
  ok("  ... and it has a hint", typeof T.hint === "string" && T.hint.length > 80);
  // the FIT_FRACS parameterization: two tables, neither leaking into the other
  ok("the spectrum card's snap table is UNCHANGED at -5/3, -3/2",
     JSON.stringify(C.FIT_FRACS) === JSON.stringify([[-5 / 3, "-5/3"], [-3 / 2, "-3/2"]]));
  ok("  ... a blank box still defaults to -5/3 with no table given, -1/3 with the aniso one",
     C.fitIndex("") === -5 / 3 && C.fitIndex("", C.FIT_FRACS_ANISO) === -1 / 3);
  ok("  ... -0.5 snaps to -1/2 on the aniso table and is literal on the spectrum's",
     C.fitLabel(C.fitIndex("-0.5", C.FIT_FRACS_ANISO), C.FIT_FRACS_ANISO) === "k^-1/2" &&
     C.fitLabel(C.fitIndex("-0.5"), C.FIT_FRACS) === "k^-0.5");
  ok("  ... -1.667 still snaps to -5/3 on the spectrum's table (nothing regressed)",
     C.fitIndex("-1.667") === -5 / 3 && C.fitLabel(-5 / 3) === "k^-5/3");
  ok("  ... the aniso table is exactly -1/3, -1/2, -1 and labels the k⊥ abscissa",
     JSON.stringify(C.FIT_FRACS_ANISO.map(f => f[1])) === JSON.stringify(["-1/3", "-1/2", "-1"]) &&
     C.fitLabel(-1, C.FIT_FRACS_ANISO, "k⊥") === "k⊥^-1");
  ok("  ... and nothing outside FIT_SNAP snaps on either table",
     C.fitIndex(String(-1 / 3 + 2 * C.FIT_SNAP), C.FIT_FRACS_ANISO) !== -1 / 3);
}

// ---------------------------------------------------------------------------
// booted pages: the selector in both apps and flChartOn's gate in 3D
// ---------------------------------------------------------------------------
const stubenv = require("./stubenv");
const boot = async (page, demo) => {
  const e = stubenv(dir, page, demo);
  await new Promise(r => setTimeout(r, 0));
  return e;
};
(async () => {
{
  const env = await boot("rmhd2d.html", null);
  const keys = env.run("function(){ return chartTypeKeys(); }");
  ok("the 2D app never offers the anisotropy card", keys.indexOf("aniso") < 0, keys.join(","));
  ok("  ... and 2D boots clean", env.fails.length === 0, env.fails.join(" | "));
}
{
  const env = await boot("rmhd3d.html", null);
  const keys = env.run("function(){ return chartTypeKeys(); }");
  ok("the 3D app offers it, and still offers no island/mode",
     keys.indexOf("aniso") >= 0 && keys.indexOf("island") < 0, keys.join(","));
  const bySrc = () => env.run("function(){ return cards.chart.filter(c =>"
    + " (CHART_TYPES[c.type()].src || c.type()) === 'spectrum').length; }");
  const nSrc0 = bySrc();                    // the boot layout's own spectrum card(s)
  const card = env.run("function(){ const c = addChartCard('aniso'); cardsSync(); return c; }");
  ok("  ... a card of that type builds, with its five option controls",
     !!card && card.optEls.length === 5, card ? "n=" + card.optEls.length : "none");
  const optId = (c, id) => env.run("function(c, i){ return c.optEls.filter(s => s.__optId === i)[0]; }", c, id);
  // the fl readback gate: OFF for the z-only view, ON for both / field line, and still
  // ON for a spectrum card asking for the field-line spectrum
  const gate = () => env.run("function(){ return flChartOn(); }");
  const setAd = v => env.run("function(c, v){ const s = c.optEls.filter(x => x.__optId === 'ad')[0];"
                             + " s.value = v; s.onchange(); }", card, v);
  ok("flChartOn: an aniso card on \"both\" (the default) asks for the sample readback",
     gate() === true);
  setAd("z");
  ok("  ... switching it to \"along z only\" turns the readback OFF again", gate() === false);
  setAd("fl");
  ok("  ... \"field line only\" turns it back on", gate() === true);
  // a spectrum card is the OTHER consumer: the predicate must still see it
  const sc = env.run("function(){ const c = addChartCard('spectrum'); cardsSync(); return c; }");
  setAd("z");
  ok("  ... with the aniso card back on z, a plain spectrum card does not arm it",
     gate() === false);
  env.run("function(c){ const s = c.optEls.filter(x => x.__optId === 'sd')[0];"
          + " s.value = 'fl'; s.onchange(); }", sc);
  ok("  ... but a spectrum card on \"k∥ (field line)\" does", gate() === true);
  env.run("function(c){ const s = c.optEls.filter(x => x.__optId === 'sd')[0];"
          + " s.value = 'both'; s.onchange(); }", sc);
  ok("  ... and with every consumer switched away it is off again", gate() === false);
  // the card is fed by the SPECTRUM readback: _chartsBySrc must collect it alongside the
  // spectrum cards, so the two added above cost no second round trip
  ok("the aniso + spectrum cards join the one spectrum readback pool",
     bySrc() === nSrc0 + 2, nSrc0 + " at boot -> " + bySrc());
  // and it really draws off that data object, through the card's own draw()
  env.run("function(c, nb, nzb){"
    + " const a = new Float32Array(3 * nb), p = new Float32Array(3 * nzb);"
    + " for (let b = 1; b < nb; b++) { const e = Math.pow(b, -5/3); a[b] = e; a[nb+b] = 0; a[2*nb+b] = 0; }"
    + " for (let b = 0; b < nzb; b++) { const e = 3 * Math.pow(b+1, -2); p[b] = e; p[nzb+b] = 0; p[2*nzb+b] = 0; }"
    + " c.draw({ perp: a, nb: nb, fshell: [1,5], par: p, parFL: p, parKfac: 1, kunit: 1 });"
    + "}", card, 512, 128);
  ok("  ... and one readback through ChartCard.draw() raises nothing", env.fails.length === 0,
     env.fails.join(" | "));
  // retyping to and from the card is clean (its options come and go with it)
  env.run("function(c){ c.selType.value = 'energy'; c.selType.onchange(); }", card);
  ok("  ... retyping away leaves one option control and no fl demand",
     card.optEls.length === 1 && gate() === false, "n=" + card.optEls.length);
  env.run("function(c){ c.selType.value = 'aniso'; c.selType.onchange(); }", card);
  ok("  ... and retyping back rebuilds the five and re-arms the gate",
     card.optEls.length === 5 && gate() === true, "n=" + card.optEls.length);
  ok("  ... optId lookup finds aq/ad (the option ids the draw branches on)",
     !!optId(card, "aq") && !!optId(card, "ad"));
  ok("3D boots clean through all of it", env.fails.length === 0, env.fails.join(" | "));
}

console.log(bad ? "\n" + bad + " FAILURE(S)" : "\nall checks passed");
process.exit(bad ? 1 : 0);
})();
