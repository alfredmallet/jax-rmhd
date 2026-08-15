// ANISO_PLAN gates: the k∥(k⊥)/k⊥ matching (Phase A) and the card it feeds (Phase B).
// Usage: node checkaniso.js [webgpu-dir]           exit code 1 on any failure
//
// Sections 1-6 are the plan's six groups, run against the REAL common.js functions in a
// bare vm (no page, no canvas): slope recovery, the density trap as a regression guard,
// uniqueness/robustness, the no-extrapolation rule, the degenerates, and the level
// window. Section 7 is the card entry itself. Section 8 is CHI_PLAN's five checks on the
// second ordinate (chi = k_perp db / k_par v_A): the bit-identity gate on the shipped
// ratio path, the chi arithmetic, the Elsasser pairing and its provenance, the reported
// estimator bias alpha, and chi's own degenerates. The booted-page block at the end drives
// the card on stubenv -- the type selector, the option set, and the field-line readback
// gate, which is the one functional per-app edit ANISO_PLAN allowed.
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
  + " FIT_FRACS_ANISO, FIT_SNAP, SPEC_KNEE, CHART_TYPES, COL,"
  + " fitAnchor, fitAnchorAuto, fitKMatch, FIT_KBOX,"
  + " _anisoQAt, ANISO_OPP, ANISO_CHI_REF, flSpectrum })", sandbox));

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
  // the knee this window ended at is RETURNED, because the fit line's automatic amplitude
  // anchors off it (section 6b) and must not measure it a second time
  ok("  ... and anisoCurves hands the knee back, the same one the window used", (() => {
    // "no knee" needs a spectrum that never falls SPEC_KNEE decades -- a k^-1 over 64 bins.
    // (CASE_GS does not qualify: k^-5/3 over 32768 bins falls seven decades and HAS one.)
    const flat = { perp: mk(64, j => (j === 0 ? 0 : Math.pow(j, -1))), nb: 64,
                   par: mk(32, j => 0.3 * Math.pow(j + 1, -1)), parKfac: 1, fshell: [1, 3] };
    return C.anisoCurves(dd, { aq: "tot", ad: "z" }).kd === knee &&
           !isFinite(C.anisoCurves(flat, { ad: "z" }).kd) &&
           !isFinite(C.anisoCurves({ nb: 4 }, {}).kd);
  })(),
     "k_d = " + knee + " on the knee'd spectrum, Infinity on a shallow one and on nothing");
  ok("_anisoWin returns null when no bin sits inside the window",
     C._anisoWin(C._anisoTail(C._anisoLeg(parLaw(3, -2), PAR_N, 1, 1, lane)), 1e9, Infinity) === null);
  // the knee factoring must not have moved the spectrum card's floor
  ok("specFloor still answers on the factored knee (a decade-low curve drops the floor)", (() => {
    const hiC = [[[1, 1, 2, 1, 3, 1]]], loC = [[[1, 1e-6, 2, 1e-6, 3, 1e-6]]];
    return C.specFloor(hiC, 1, 1) >= C.specFloor(hiC.concat(loC), 1, 1e-6);
  })());
}

// ---------------------------------------------------------------------------
console.log("6b. the fit line's AUTOMATIC anchor: the intermediate matching scale");
// ---------------------------------------------------------------------------
// Alfred, 2026-08-11: in automatic ("pin") mode the amplitude is matched halfway
// LOGARITHMICALLY between the box scale and the dissipation scale --
//     k_match = sqrt(k_box * k_diss),  k_box = FIT_KBOX = 1
// -- instead of at kA just above the forcing shell, on BOTH cards. The sampling convention
// inside fitAnchor is unchanged (first point at or above the anchor k), so these legs pin
// WHERE it anchors, the two fallbacks to kA, and the fact that the aniso card feeds it the
// knee it already measured rather than a second opinion.
{
  ok("k_box is the box fundamental, 1, and NOT kA", C.FIT_KBOX === 1);
  ok("k_match is the geometric mean of k_box and the knee -- halfway in log",
     C.fitKMatch(100) === 10 && C.fitKMatch(64) === 8 &&
     Math.abs(Math.log(C.fitKMatch(1234)) - 0.5 * (Math.log(1) + Math.log(1234))) < 1e-12,
     "sqrt(1 * k_d)");
  ok("  ... and no knee (specKnee = Infinity) gives no matching scale either",
     C.fitKMatch(Infinity) === Infinity);
  // a straight A k^p series: every anchor point recovers the SAME A, so the leg that proves
  // the anchor moved has to be a series that is NOT a single power law. This one is two
  // power laws joined at k = 50, so the amplitude read at k = 10 differs from the one read
  // at k = 3 by a factor the check states.
  const p = -1 / 3, A0 = 0.7;
  const pts = [];
  for (let k = 1; k <= 400; k++) pts.push(k, k <= 50 ? A0 * Math.pow(k, p) : 4 * A0 * Math.pow(k, p));
  const kd = 100, kM = C.fitKMatch(kd);              // = 10, inside the first branch
  ok("the automatic anchor samples the curve at k_match, not at kA",
     Math.abs(C.fitAnchorAuto(pts, 3, kd, p) - A0) < 1e-12 &&
     C.fitAnchorAuto(pts, 3, kd, p) === C.fitAnchor(pts, kM, p),
     "k_match = " + kM + ", A = " + C.fitAnchorAuto(pts, 3, kd, p).toFixed(4));
  ok("  ... which really is a different point from kA on a curve that is not one power law",
     C.fitAnchorAuto(pts, 60, kd, p) !== C.fitAnchor(pts, 60, p) &&
     Math.abs(C.fitAnchor(pts, 60, p) / A0 - 4) < 1e-12,
     "kA = 60 lands on the upper branch (4x), k_match = 10 on the lower");
  ok("FALLBACK 1, no knee in view: the anchor is kA again, exactly as before this change",
     C.fitAnchorAuto(pts, 60, Infinity, p) === C.fitAnchor(pts, 60, p) &&
     C.fitAnchorAuto(pts, 3, Infinity, p) === C.fitAnchor(pts, 3, p));
  ok("FALLBACK 2, k_match past the end of the drawn series: kA again",
     C.fitAnchorAuto(pts, 3, 1e9, p) === C.fitAnchor(pts, 3, p) &&
     C.fitAnchor(pts, C.fitKMatch(1e9), p) === 0,
     "k_match = " + C.fitKMatch(1e9).toFixed(0) + " > 400");
  ok("  ... and a series with nothing at all still anchors nothing",
     C.fitAnchorAuto([], 3, 100, p) === 0);

  // WIRING: drawAniso must hand fitAnchorAuto the curve it is drawing, its kA, and the knee
  // anisoCurves returned -- checked by intercepting the function in the sandbox rather than
  // by re-deriving pixels off the canvas.
  const KD = 300;
  const dd = Object.assign({}, CASE_GS, {
    perp: mk(PERP_N, j => (j === 0 ? 0 : Math.pow(j, -5 / 3) * Math.exp(-Math.pow(j / KD, 4))))
  });
  const stub = () => {
    const o = { canvas: { width: 420, height: 240 }, calls: [] };
    for (const m of ["save", "restore", "beginPath", "moveTo", "lineTo", "stroke", "fill",
                     "fillRect", "clearRect", "rect", "clip", "setLineDash", "closePath",
                     "fillText", "strokeRect", "translate", "scale", "arc"]) o[m] = () => {};
    o.measureText = t => ({ width: 6.2 * t.length, actualBoundingBoxAscent: 7.2,
                            actualBoundingBoxDescent: 0 });
    return o;
  };
  const spy = (d, o) => {
    const seen = [];
    sandbox.__spy = (pts2, kA2, kd2, p2) => { seen.push([pts2, kA2, kd2, p2]); return 1; };
    vm.runInContext("var __real = fitAnchorAuto;"
      + " fitAnchorAuto = (a, b, c, e) => __spy(a, b, c, e);", sandbox);
    try { C.drawAniso(stub(), d, o); } finally {
      vm.runInContext("fitAnchorAuto = __real;", sandbox);
    }
    return seen;
  };
  const AA = C.anisoCurves(dd, { aq: "tot", ad: "both", fit: "pin" });
  const seen = spy(dd, { aq: "tot", ad: "both", fit: "pin", fitp: -0.333 });
  ok("drawAniso anchors through fitAnchorAuto, on the field-line curve, with THAT knee",
     seen.length === 1 && seen[0][2] === AA.kd && isFinite(AA.kd) &&
     seen[0][1] === C.fitKA(dd.nb, dd.fshell) &&
     JSON.stringify(Array.from(seen[0][0])) ===
       JSON.stringify(Array.from(AA.curves[AA.nGlob][0])),
     "kA = " + seen[0][1] + ", k_d = " + seen[0][2].toFixed(1)
     + ", k_match = " + C.fitKMatch(seen[0][2]).toFixed(2));
  // The two cards share the RULE and the units (k/kunit), so their anchors coincide when
  // they are looking at the same energy: the knee the aniso card feeds in is specKnee on
  // its own perpendicular leg and nothing else. They can still land a bin apart in practice
  // -- the spectrum card measures the knee on the LANES IT IS DRAWING (its `sq` set, plus
  // any pinned ghosts in the range pool) while this one measures on the single `aq` lane --
  // which is a difference of what is being measured, never of the rule.
  ok("  ... and the anchor scale is the shared rule on this card's own range curve",
     (() => {
       const leg = C._anisoLeg(dd.perp, dd.nb, 0, 1, (u, b, h) => u + b);
       return AA.kd === C.specKnee([[leg]], C._anisoPeak(leg)) &&
              C.fitKMatch(AA.kd) === Math.sqrt(AA.kd);
     })(), "k_match = " + C.fitKMatch(AA.kd).toFixed(4));
  ok("the USER-SET amplitude never consults it: \"amp\" wins outright, \"off\" draws nothing",
     spy(dd, { ad: "both", fit: "amp", fita: "2.5" }).length === 1 &&   // computed, then overridden
     spy(dd, { ad: "both", fit: "off" }).length === 0);
  ok("  ... and an \"amp\" box left blank still falls back to the automatic anchor",
     spy(dd, { ad: "both", fit: "amp", fita: "" }).length === 1);
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
  ok("  ... its options are the ordinate, aq, ad and the fit trio",
     ids === "ay,aq,ad,fit,fitp,fita", ids);
  const os = T.opts({ zslice: true });
  ok("  ... the defaults are the FIRST option of each list: the ratio ordinate, total "
     + "energy, both curves",
     os[0].o[0][0] === "ratio" && os[1].o[0][0] === "tot" && os[2].o[0][0] === "both" &&
     os[3].o[0][0] === "pin");
  ok("  ... and the second ordinate is chi (CHI_PLAN), the only other one",
     os[0].o.length === 2 && os[0].o[1][0] === "chi" && /&chi;/.test(os[0].o[1][1]),
     os[0].o.map(x => x[0] + "=" + x[1]).join(" | "));
  ok("  ... the index box defaults to -0.333, which snaps to exactly -1/3",
     C.fitIndex(String(os[4].v), C.FIT_FRACS_ANISO) === -1 / 3, String(os[4].v));
  ok("  ... the two number boxes hide with the fit line, as on the spectrum card",
     os[4].vis({ fit: "off" }) === false && os[4].vis({ fit: "pin" }) === true &&
     os[5].vis({ fit: "pin" }) === false && os[5].vis({ fit: "amp" }) === true);
  // ... and the INDEX box hides on the chi ordinate as well, where a power-law index is
  // not what the reference line is (it is the horizontal chi = 1). The amplitude box
  // stays: there it names the level of that line.
  ok("  ... and the index box also hides on the chi ordinate, where it means nothing",
     os[4].vis({ fit: "pin", ay: "chi" }) === false &&
     os[4].vis({ fit: "amp", ay: "chi" }) === false &&
     os[5].vis({ fit: "amp", ay: "chi" }) === true &&
     os[4].vis({ fit: "pin", ay: "ratio" }) === true);
  // the hint FOLLOWS the ordinate select, so it is a function of the options -- the shape
  // gen2d's colour-scale hint already has
  ok("  ... and it has a hint, one per ordinate",
     typeof T.hint === "function" && T.hint({}).length > 80 &&
     T.hint({ ay: "chi" }).length > 80 && T.hint({}) !== T.hint({ ay: "chi" }));
  // Alfred's copy (third feedback round), and the ONE thing in it that is a claim about the
  // code rather than about the physics: which leg is solid and which is dashed. anisoCurves
  // gives the z leg a null dash and the field-line leg [5, 3], so solid = k_z in the global
  // mean field and dashed = k_par along the local field, which is what the hint says. If a
  // future edit swaps the dashes, this leg fails rather than the copy quietly going wrong.
  const H = T.hint({}), HC = T.hint({ ay: "chi" });
  ok("the hint is Alfred's copy: cumulative-energy matching, the two legs, the CB band",
     /^k&#8741;\(k&perp;\)\/k&perp; as a function of k&perp; by matching cumulative energy/.test(H) &&
     /Solid: k<sub>z<\/sub> \(global mean field\)\./.test(H) &&
     /Dashed: k&#8741; \(local mean field along field lines\)\./.test(H) &&
     /somewhere between &minus;1\/2 and &minus;1\/3 is the classic critical balance prediction/
       .test(H) &&
     /experimental feature: imperfect agreement at these resolutions\.$/.test(H),
     H.length + " chars");
  ok("  ... and the ratio hint is UNCHANGED by CHI_PLAN: `ay` absent reads as `ay: ratio`",
     H === T.hint({ ay: "ratio" }) && H === T.hint(null) && H === T.hint({ aq: "zp" }));
  ok("  ... and SOLID / DASHED are the dashes anisoCurves actually strokes the legs with",
     (() => {
       const A = C.anisoCurves(CASE_GS, { aq: "tot", ad: "both" });
       const z = A.curves[0], fl = A.curves[1];
       return A.nGlob === 1 && A.curves.length === 2 && z[2] === null &&
              JSON.stringify(fl[2]) === "[5,3]" && /k∥z/.test(z[3]) && /k∥B/.test(fl[3]);
     })(), "z leg dash null, field-line leg [5,3]");
  ok("  ... it drops to the manual the things the one-breath version cannot carry",
     !/Cho/i.test(H) && !/gauge/i.test(H) && !/L<sub>z<\/sub>/.test(H) &&
     !/DIVERGE/.test(H), H);
  // the chi hint is Alfred's own copy (2026-08-14, replacing the drafted one wholesale,
  // like the ratio branch before it). What these legs pin is HIS claims -- the db
  // convention still spelled out, chi ~ 1 flat as CB's statement, no two-figure level --
  // plus the honesty clause that replaced the drafted departures list: whether the level
  // is attainable at these resolutions is doubted OUT LOUD, with the literature pointer
  // (Mallet et al. 2015) for the better-resolved test. The pairing disclosure and the
  // solid/dashed-split caveat the draft carried moved to docs.html #chi, and the last leg
  // holds them THERE -- the split one is a physics claim about the code (one db divides
  // both legs) and may not be dropped from both places.
  ok("the chi hint (Alfred's copy) states the db convention and chi ~ 1 flat, no two-figure level",
     /&delta;b&sup2; = Q/.test(HC) && /matched energy content above\s+k&perp;/.test(HC) &&
     /v<sub>A<\/sub> = 1/.test(HC) && /&chi; ~ 1/.test(HC) &&
     /flat across the inertial range/.test(HC) && !/[0-9]\.[0-9]/.test(HC),
     HC.length + " chars, no two-significant-figure level in it");
  ok("  ... doubts the level's attainability out loud and points at the literature",
     /somewhat dubious/.test(HC) && /low-resolution/.test(HC) &&
     /Mallet et al. 2015/.test(HC));
  const DOCS = fs.readFileSync(path.join(dir, "docs.html"), "utf8");
  const DCHI = (DOCS.split(/id="chi"/)[1] || "").slice(0, 4000);
  ok("  ... and the manual's #chi carries what the one-breath copy dropped",
     /Which &delta;b, on E<sup>&plusmn;<\/sup>/.test(DCHI) &&
     /not a new measurement/.test(DCHI) && /same &delta;b divides both/.test(DCHI) &&
     /nothing to shear the other/.test(DCHI));
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
console.log("8. the chi ordinate: χ = k⊥ δb / (k∥ v_A)   (CHI_PLAN checks 1-5)");
// ---------------------------------------------------------------------------
// `ay` picks what the card puts on its y axis: the shipped ratio, or chi. Everything the
// two share -- the tails, the window, the level ladder, the legend -- is shared code, so
// these legs are about the one multiplication, the one lookup that crosses Elsasser lanes,
// and the promise that the shipped path did not move.
//
// The check's own arithmetic below rebuilds the LEVEL LADDER from the public helpers
// (fitKA / specKnee / _anisoLeg / _anisoTail / _anisoWin / _anisoAt / ANISO_NLEV) rather
// than reading it out of anisoCurves, so "chi = kp*sqrt(Q)/kz" is asserted against the
// definition and the level set is available to say WHICH Q and WHICH lane each point came
// from -- which is what check 3 needs.
const LADDER_TOL = 1e-12;
function anisoLadder(d, o) {
  const lane = C.ANISO_LANES[(o && o.aq)] || C.ANISO_LANES.tot;
  const nb = d.nb, kf = d.parKfac === undefined ? 1 : d.parKfac;
  const perp = C._anisoLeg(d.perp, nb, 0, 1, lane);
  const TP = C._anisoTail(perp);
  const kd = C.specKnee([[perp]], C._anisoPeak(perp));
  const wP = C._anisoWin(TP, C.fitKA(nb, d.fshell || [1, 3]), kd);
  const src = (o && o.ad) === "fl" ? d.parFL : d.par;
  const nzb = Math.floor((src ? src.length : 0) / 3);
  const par = C._anisoLeg(src, nzb, 1, kf, lane);
  const TQ = C._anisoTail(par);
  const wQ = C._anisoWin(TQ, 0, C.specKnee([[par]], C._anisoPeak(par)));
  const out = { TP, TQ, lev: [] };
  if (!wP || !wQ) return out;
  const Qhi = Math.min(wP[1], wQ[1]), Qlo = Math.max(wP[0], wQ[0]);
  if (!(Qhi > 0) || !(Qlo > 0) || Qlo > Qhi) return out;
  const nl = Qlo < Qhi ? C.ANISO_NLEV : 1;
  for (let m = 0; m < nl; m++) {
    const Q = nl > 1 ? Qhi * Math.pow(Qlo / Qhi, m / (nl - 1)) : Qhi;
    const kp = C._anisoAt(TP, Q), kz = C._anisoAt(TQ, Q);
    if (kp > 0 && kz > 0) out.lev.push({ Q, kp, kz });
  }
  return out;
}
// the worst relative disagreement between a drawn curve and a list of expected values
const worstRel = (pts, want) => {
  let w = 0;
  for (let i = 0; i < want.length; i++) w = Math.max(w, Math.abs(pts[2 * i + 1] - want[i]) / want[i]);
  return w;
};
// a (k, v) curve sampled at k, log-log -- for comparing two curves drawn over different
// k ranges (the two Elsasser lanes are not drawn over the same one)
const curveAt = (pts, k) => {
  if (pts.length < 4 || k < pts[0] || k > pts[pts.length - 2]) return 0;
  for (let i = 0; i + 3 < pts.length; i += 2) {
    if (k >= pts[i] && k <= pts[i + 2]) {
      const dk = Math.log(pts[i + 2] / pts[i]);
      return dk > 0 ? pts[i + 1] * Math.pow(pts[i + 3] / pts[i + 1], Math.log(k / pts[i]) / dk)
                    : pts[i + 1];
    }
  }
  return 0;
};

// ---------------------------------------------------------------------------
console.log("8.1 CHECK 1, the gate: `ay: ratio` is BIT-IDENTICAL to what shipped");
// ---------------------------------------------------------------------------
// The base is the last commit that touched common.js before CHI_PLAN, read out of git
// rather than vendored as a golden file: what this has to compare against is what SHIPPED,
// and a golden copy is one more thing to keep honest. Both spellings of the default are
// compared -- `ay` absent (every existing caller, including the app's own self-test and
// any preset that never names it) and `ay: "ratio"` written out. Somewhere without the
// repo (a tarball, an export) this reports a SKIP: this file reports, it never gates.
{
  const BASE = "70ec5a8";
  let base = null, why = "";
  try {
    base = require("child_process").execFileSync(
      "git", ["-C", dir, "show", BASE + ":webgpu/common.js"],
      { encoding: "utf8", maxBuffer: 64 << 20, stdio: ["ignore", "pipe", "pipe"] });
  } catch (e) { why = String((e && e.message) || e).split("\n")[0]; }
  if (!base) {
    console.log("  SKIP  " + BASE + ":webgpu/common.js is not readable here (" + why + ")");
  } else {
    const sb = {
      document: { getElementById: () => stubEl(), createElement: () => stubEl(),
                  createTextNode: () => ({}), querySelectorAll: () => [] },
      window: { addEventListener() {}, devicePixelRatio: 1, matchMedia: () => ({ matches: true }) },
      console, Math, JSON, Float32Array, Float64Array, Uint32Array, Uint8ClampedArray, Map, Set,
      Error, Promise, setTimeout, Number, String, Array, Object, isFinite, parseInt, parseFloat,
      URLSearchParams, performance: { now: () => 0 }
    };
    sb.globalThis = sb;
    vm.createContext(sb);
    vm.runInContext(base, sb, { filename: BASE + ":common.js" });
    const B = vm.runInContext("({ anisoCurves })", sb);
    const wob2 = j => 1 + 0.6 * Math.sin(1.7 * j) * Math.cos(0.31 * j + 1);
    const CASES = {
      gs: CASE_GS, aligned: CASE_AL,
      lanes: { perp: mk3(PERP_N, j => (j === 0 ? 0 : Math.pow(j, -5 / 3))), nb: PERP_N,
               par: mk3(PAR_N, j => 3 * Math.pow(j + 1, -2)),
               parFL: mk3(PAR_N, j => 5 * Math.pow(j + 1, -1.8)), parKfac: 1.37, fshell: [1, 5] },
      noisy: Object.assign({}, CASE_GS, {
        perp: mk(PERP_N, j => (j === 0 ? 0 : Math.pow(j, -5 / 3) * wob2(j))),
        par: mk(PAR_N, j => 3 * Math.pow(j + 1, -2) * wob2(3 * j)) }),
      knee: Object.assign({}, CASE_GS, {
        perp: mk(PERP_N, j => (j === 0 ? 0 : Math.pow(j, -5 / 3) * Math.exp(-Math.pow(j / 300, 4)))) }),
      shortpar: Object.assign({}, CASE_GS, { par: mk(6, j => 3 * Math.pow(j + 1, -2)) }),
      loudpar: Object.assign({}, CASE_GS, { par: mk(4, () => 1e12) }),
      flat: { perp: mk(64, j => (j === 0 ? 0 : Math.pow(j, -1))), nb: 64,
              par: mk(32, j => 0.3 * Math.pow(j + 1, -1)),
              parFL: mk(32, j => 0.3 * Math.pow(j + 1, -1)), parKfac: 1, fshell: [1, 3] },
      nofl: Object.assign({}, CASE_GS, { parFL: null }),
      nopar: Object.assign({}, CASE_GS, { par: null }),
      silent: { perp: mk(64, () => 0), nb: 64, par: parLaw(3, -2), parKfac: 1 },
      onebin: { perp: mk(2, () => 1), nb: 2, par: parLaw(3, -2), parKfac: 1 },
      nothing: null, nbonly: { nb: 4 }
    };
    const OPTS = [];
    for (const aq of ["tot", "zp", "zm", "junk", undefined])
      for (const ad of ["both", "z", "fl", undefined]) OPTS.push({ aq, ad });
    OPTS.push(null, {});
    // Infinity is not JSON, and kd carries it on every knee-less spectrum, so it is spelled
    const rep = (k, v) => (typeof v === "number" && !isFinite(v)) ? "#" + String(v) : v;
    let n = 0, diff = 0, first = "";
    for (const cn of Object.keys(CASES)) for (const o of OPTS) {
      const want = JSON.stringify(B.anisoCurves(CASES[cn], o), rep);
      for (const oo of [o, Object.assign({ ay: "ratio" }, o)]) {
        n++;
        if (JSON.stringify(C.anisoCurves(CASES[cn], oo), rep) !== want && !diff++)
          first = cn + " " + JSON.stringify(oo);
      }
    }
    ok("anisoCurves matches base " + BASE + " EXACTLY on every (data x aq x ad) pair",
       diff === 0, n + " comparisons over " + Object.keys(CASES).length + " data cases"
       + (diff ? ", first difference at " + first : ", implicit default and ay:\"ratio\""));
  }
}

// ---------------------------------------------------------------------------
console.log("8.2 CHECK 2, the arithmetic: chi = kp*sqrt(Q)/kz, and the slope that implies");
// ---------------------------------------------------------------------------
{
  // the new helper first: _anisoQAt is _anisoAt's inverse direction, and inherits its
  // no-extrapolation rule (section 4 is the same set of legs the other way round)
  const lane = (u, b, h) => u + b;
  const T = C._anisoTail(C._anisoLeg(CASE_GS.perp, PERP_N, 0, 1, lane));
  ok("_anisoQAt refuses a k below the tail's FIRST bin and above its LAST (returns 0)",
     C._anisoQAt(T, T.ks[0] * 0.999999) === 0 && C._anisoQAt(T, 1e9) === 0 &&
     C._anisoQAt(T, 0) === 0 && C._anisoQAt(C._anisoTail([]), 1) === 0);
  ok("  ... the two ends themselves map to the two end levels",
     C._anisoQAt(T, T.ks[0]) === T.qs[0] && C._anisoQAt(T, T.ks[T.n - 1]) === T.qs[T.n - 1]);
  ok("  ... an exact bin maps to that bin's level, a midpoint lands between",
     Math.abs(C._anisoQAt(T, T.ks[7]) - T.qs[7]) < 1e-12 * T.qs[7] &&
     C._anisoQAt(T, Math.sqrt(T.ks[7] * T.ks[8])) < T.qs[7] &&
     C._anisoQAt(T, Math.sqrt(T.ks[7] * T.ks[8])) > T.qs[8]);
  ok("  ... and it round-trips _anisoAt: Q -> k -> Q to fp64",
     [3, 9, 40, 120].every(i => {
       const Q = Math.sqrt(T.qs[i] * T.qs[i + 1]);
       return Math.abs(C._anisoQAt(T, C._anisoAt(T, Q)) - Q) < 1e-12 * Q;
     }));
  // the ordinate itself, on the two power-law pairs of section 1
  for (const [nm, d, p] of [["GS95 (-5/3 vs -2)", CASE_GS, -5 / 3],
                            ["aligned (-3/2 vs -2)", CASE_AL, -3 / 2]]) {
    for (const ad of ["z", "fl"]) {
      const L = anisoLadder(d, { aq: "tot", ad });
      const r = C.anisoCurves(d, { aq: "tot", ad }).curves[0][0];
      const x = C.anisoCurves(d, { aq: "tot", ad, ay: "chi" }).curves[0][0];
      let sameK = r.length === x.length;
      for (let i = 0; sameK && i < r.length; i += 2) if (r[i] !== x[i]) sameK = false;
      ok(nm + ", " + ad + ": the ordinate MULTIPLIES, it does not re-map k⊥",
         sameK && x.length === 2 * L.lev.length, r.length / 2 + " vs " + x.length / 2 + " points");
      ok("  ... chi = kp*sqrt(Q)/kz from the level ladder, to fp64",
         worstRel(x, L.lev.map(l => l.kp * Math.sqrt(l.Q) / l.kz)) < LADDER_TOL,
         "worst " + worstRel(x, L.lev.map(l => l.kp * Math.sqrt(l.Q) / l.kz)).toExponential(2));
      ok("  ... equivalently chi = sqrt(Q(k⊥))/ratio: the two ordinates differ by sqrt(Q)",
         worstRel(x, r.filter((v, i) => i % 2)
                      .map((rv, i) => Math.sqrt(C._anisoQAt(L.TP, r[2 * i])) / rv)) < 1e-9);
    }
    // ... and what that predicts for the SLOPE, which is the physics statement: with
    // E⊥ ~ k^p the tail goes as k^(p+1), sqrt(Q) as k^((p+1)/2), so chi's slope is
    // (p+1)/2 minus the ratio's. GS95 gives exactly 0 -- flat chi IS critical balance --
    // and the aligned pair +1/4, which is a different claim and worth failing on.
    const r = C.anisoCurves(d, { aq: "tot", ad: "z" }).curves[0][0];
    const x = C.anisoCurves(d, { aq: "tot", ad: "z", ay: "chi" }).curves[0][0];
    const want = (p + 1) / 2 - (p === -5 / 3 ? -1 / 3 : -1 / 2);
    ok("  ... so the chi slope is " + want.toFixed(4) + " where the ratio slope is "
       + slope(r).toFixed(4), Math.abs(slope(x) - want) < SLOPE_TOL,
       "got " + slope(x).toFixed(4) + " (tol " + SLOPE_TOL + ")");
  }
}

// ---------------------------------------------------------------------------
console.log("8.3 CHECK 3, the Elsasser pairing: db crosses lanes, the matching does not");
// ---------------------------------------------------------------------------
// chi± = k⊥ Z∓ / (k∥± v_A): the counterpropagating field is what shears you, so db comes
// from the OPPOSITE lane -- and nothing else does. The adjacent wrong implementation
// builds the whole perpendicular tail from the opposite lane, which also moves which k⊥
// each level maps to; it survives a pure amplitude asymmetry with the right sign, so the
// two lanes here get different perpendicular SLOPES as well, and what is asserted is kp's
// provenance and not just the direction of the split.
{
  // E+ = k^-5/3 against E- = 0.05 k^-4/3 (different amplitude AND slope), carried as
  // E_u = (E+ + E-)/2 and H_c = (E+ - E-)/2, which is what E± = E_u + E_b ± H_c inverts to.
  // The PARALLEL stack carries the same imbalance, so each lane's k∥ comes off its own
  // spectrum as it does in a real imbalanced run.
  const mkPM = (n, fp, fm, j1) => {
    const a = new Float32Array(3 * n);
    for (let j = j1; j < n; j++) {
      const p = fp(j), m = fm(j);
      a[j] = 0.5 * (p + m); a[n + j] = 0; a[2 * n + j] = 0.5 * (p - m);
    }
    return a;
  };
  const IMB = 0.05;
  const parPM = mkPM(PAR_N, j => 3 * Math.pow(j + 1, -2), j => IMB * 3 * Math.pow(j + 1, -2), 0);
  const d = { perp: mkPM(PERP_N, j => Math.pow(j, -5 / 3), j => IMB * Math.pow(j, -4 / 3), 1),
              nb: PERP_N, par: parPM, parFL: parPM, parKfac: 1, fshell: [1, 5] };
  const TPp = C._anisoTail(C._anisoLeg(d.perp, PERP_N, 0, 1, C.ANISO_LANES.zp));
  const TPm = C._anisoTail(C._anisoLeg(d.perp, PERP_N, 0, 1, C.ANISO_LANES.zm));
  ok("the two lanes really are different curves, not one scaled (slopes -5/3 vs -4/3)",
     Math.abs(slope(C.anisoCurves(d, { aq: "zp", ad: "z" }).curves[0][0])
              - slope(C.anisoCurves(d, { aq: "zm", ad: "z" }).curves[0][0])) > 0.2,
     "ratio slopes " + slope(C.anisoCurves(d, { aq: "zp", ad: "z" }).curves[0][0]).toFixed(3)
     + " / " + slope(C.anisoCurves(d, { aq: "zm", ad: "z" }).curves[0][0]).toFixed(3));
  for (const [aq, TPo, TPs] of [["zp", TPm, TPp], ["zm", TPp, TPm]]) {
    const L = anisoLadder(d, { aq, ad: "z" });
    const r = C.anisoCurves(d, { aq, ad: "z" }).curves[0][0];
    const x = C.anisoCurves(d, { aq, ad: "z", ay: "chi" }).curves[0][0];
    // PROVENANCE: kp is this lane's own measurement, bit for bit
    let sameK = r.length === x.length;
    for (let i = 0; sameK && i < r.length; i += 2) if (r[i] !== x[i]) sameK = false;
    ok(aq + ": every k⊥ is the SELECTED lane's own, bit-identical to the ratio curve", sameK);
    ok("  ... and chi takes db from the OPPOSITE lane's tail at that k⊥, to fp64",
       worstRel(x, L.lev.map(l => l.kp * Math.sqrt(C._anisoQAt(TPo, l.kp)) / l.kz)) < LADDER_TOL);
    ok("  ... it is NOT the own-lane level (which is the balanced-run coincidence)",
       worstRel(x, L.lev.map(l => l.kp * Math.sqrt(C._anisoQAt(TPs, l.kp)) / l.kz)) > 0.5,
       "own-lane db would be " +
       (100 * worstRel(x, L.lev.map(l => l.kp * Math.sqrt(C._anisoQAt(TPs, l.kp)) / l.kz))).toFixed(0)
       + "% off");
  }
  // ... and the wrong wiring, COMPUTED here rather than remembered: the same ladder with
  // the perpendicular tail taken from the opposite lane wholesale. It reports different
  // k⊥ for every level, which is what the leg above convicts it on.
  {
    const wrong = (() => {
      const L = anisoLadder(d, { aq: "zm", ad: "z" });      // zm's TP with zp's ladder shape
      return L.lev.map(l => l.kp);
    })();
    const right = anisoLadder(d, { aq: "zp", ad: "z" }).lev.map(l => l.kp);
    let w = 0;
    for (let i = 0; i < Math.min(right.length, wrong.length); i++)
      w = Math.max(w, Math.abs(right[i] - wrong[i]) / right[i]);
    ok("the swapped-tail implementation lands on visibly different k⊥ (the guard has teeth)",
       w > 0.2, "worst k⊥ disagreement " + (100 * w).toFixed(0) + "%");
  }
  // the SIGN, at a common k⊥ (the two lanes are not drawn over the same range): with
  // Z+ >> Z-, it is the WEAK field that is strongly sheared, so chi- > chi+.
  {
    const xp = C.anisoCurves(d, { aq: "zp", ad: "z", ay: "chi" }).curves[0][0];
    const xm = C.anisoCurves(d, { aq: "zm", ad: "z", ay: "chi" }).curves[0][0];
    const kLo = Math.max(xp[0], xm[0]), kHi = Math.min(xp[xp.length - 2], xm[xm.length - 2]);
    let worst = Infinity, at = 0;
    for (let m = 0; m <= 8; m++) {
      const k = kLo * Math.pow(kHi / kLo, m / 8), q = curveAt(xm, k) / curveAt(xp, k);
      if (q < worst) { worst = q; at = k; }
    }
    ok("with E+ " + (1 / IMB) + "x E-, chi- exceeds chi+ at every common k⊥",
       kHi > kLo && worst > 2, "smallest chi-/chi+ = " + worst.toFixed(2) + " at k⊥ = "
       + at.toFixed(1) + " (k⊥ " + kLo.toFixed(1) + ".." + kHi.toFixed(1) + ")");
  }
  // on `tot` there is no opposite lane at all and db is the matched level itself
  ok("the tot lane has no opposite (ANISO_OPP), so db^2 is its own matched Q",
     C.ANISO_OPP.tot === undefined && C.ANISO_OPP.junk === undefined &&
     C.ANISO_OPP.zp === "zm" && C.ANISO_OPP.zm === "zp");
  {
    const L = anisoLadder(CASE_GS, { aq: "tot", ad: "z" });
    const x = C.anisoCurves(CASE_GS, { aq: "tot", ad: "z", ay: "chi" }).curves[0][0];
    ok("  ... and an unknown aq falls back to tot on the chi ordinate too",
       worstRel(x, L.lev.map(l => l.kp * Math.sqrt(l.Q) / l.kz)) < LADDER_TOL &&
       JSON.stringify(C.anisoCurves(CASE_GS, { aq: "junk", ad: "z", ay: "chi" }).curves[0][0]) ===
       JSON.stringify(x));
  }
}

// ---------------------------------------------------------------------------
console.log("8.4 CHECK 4, alpha: the estimator bias, MEASURED and reported (never gated)");
// ---------------------------------------------------------------------------
// flSpectrum must window before transforming -- a field line's two ends are unrelated --
// and the periodic Hann kernel is exactly (1/6, 2/3, 1/6) in bins, so each parallel line
// leaks into its neighbours. That fattens the parallel tail (Q_par(k) picks up
// (E(k-1) - E(k))/6, positive on a falling spectrum), the matching returns k∥ high, and chi
// comes back low. How low is a number, not an argument, so it is measured here:
//
//   a field with a PRESCRIBED ridge k∥(k⊥) = round(k⊥^2/3) and a prescribed db (the
//   perpendicular spectrum, since db^2 = Q by definition), sampled along lines, pushed
//   through the REAL flSpectrum and the REAL anisoCurves, against the same field's exact
//   parallel spectrum through the same matching. alpha = chi_measured / chi_true at equal
//   k⊥ -- and since db is prescribed and identical on both sides, alpha isolates exactly
//   the parallel estimator.
//
// What is synthetic and what is real: there is no GPU here, so the MARCH is BYPASSED --
// the samples are written down along a straight line instead. That is a real exclusion and
// not a formality: the app's marcher (rmhd3d.html, fieldLine) is bilinear in-plane and RK2
// in z at fp32, so it low-passes ALONG the line it is tracing, which pushes k∥ DOWN and chi
// UP -- the opposite sign to the Hann effect measured here, so alpha is not a bound on the
// two together (review 2026-08-14). What alpha IS, exactly: the bias of the parallel
// ESTIMATOR -- window, periodogram, ±kz fold, line average, tail matching -- all of which
// is the app's own code below, and all of which the SHIPPED ratio ordinate runs too, alpha
// reducing to kz_true/kz_meas at matched levels. Phases are deterministic and arranged so that
// the leakage cross terms between adjacent ridge bins cancel exactly over each group of
// four lines (relative phase rotating by a quarter turn), so what comes back is the Hann
// kernel and not one realisation of an interference pattern.
{
  const NB = 128, NZ = 64, NL = 64, KD = 60, NZB = NZ >> 1;
  const frac = x => x - Math.floor(x);
  const E = j => (j < 1 ? 0 : Math.pow(j, -5 / 3) * Math.exp(-Math.pow(j / KD, 4)));
  const perp = mk(NB, E);
  const ridge = j => Math.max(1, Math.min(NZB - 1, Math.round(Math.pow(j, 2 / 3))));
  // the exact parallel stack the ridge implies -- every k⊥ bin's energy lands whole in one
  // k∥ bin, in the [E_u | E_b | H_c] layout flSpectrum itself returns
  const parTrue = new Float32Array(3 * NZB);
  for (let j = 1; j < NB; j++) if (E(j) > 0) parTrue[NZB + ridge(j) - 1] += E(j);
  // ... and the along-line samples: one cosine per ridge bin per line, amplitude
  // 2*sqrt(E) so that each contributes exactly E under flSpectrum's normalization
  const smp = new Float32Array(4 * NL * NZ);
  for (let l = 0; l < NL; l++) for (let k = 1; k < NZB; k++) {
    const e = parTrue[NZB + k - 1];
    if (!(e > 0)) continue;
    const A = 2 * Math.sqrt(e);
    const ph = 2 * Math.PI * (k * (l % 4) / 4 + frac(0.7548776662 * k));
    for (let z = 0; z < NZ; z++)
      smp[4 * (l * NZ + z) + 2] += A * Math.cos(2 * Math.PI * k * z / NZ + ph);
  }
  const parMeas = C.flSpectrum(smp, NL, NZ);
  // the construction is only worth trusting if its bookkeeping is: the periodogram must
  // carry the prescribed energy back, up to the kz = 0 bin flSpectrum does not plot (the
  // Hann kernel leaks a sixth of the first ridge bin into it, and that is a real property
  // of the instrument, not of this test)
  let eT = 0, eM = 0;
  for (let i = 0; i < NZB; i++) { eT += parTrue[NZB + i]; eM += parMeas[NZB + i]; }
  ok("the synthetic field's periodogram carries its prescribed energy (bar the kz=0 leak)",
     eM / eT > 0.85 && eM / eT < 1.0, "measured/prescribed = " + (eM / eT).toFixed(4));
  ok("  ... and the measured parallel spectrum IS the (1/6, 2/3, 1/6) Hann kernel on it",
     (() => {
       let w = 0;
       for (let b = 2; b <= 8; b++) {
         const want = (parTrue[NZB + b - 2] + 4 * parTrue[NZB + b - 1] + parTrue[NZB + b]) / 6;
         w = Math.max(w, Math.abs(parMeas[NZB + b - 1] - want) / want);
       }
       return w < 0.02;
     })(), "bins 2..8, within 2%");
  const dm = { perp, nb: NB, par: null, parFL: parMeas, parKfac: 1, fshell: [1, 3] };
  const dt = { perp, nb: NB, par: null, parFL: parTrue, parKfac: 1, fshell: [1, 3] };
  const o = { aq: "tot", ad: "fl", ay: "chi" };
  const xm = C.anisoCurves(dm, o).curves[0][0], xt = C.anisoCurves(dt, o).curves[0][0];
  let sl = 0, n = 0, amin = Infinity, amax = 0, aLow = 0;
  for (let i = 0; i < xm.length; i += 2) {
    const t = curveAt(xt, xm[i]);
    if (!(t > 0)) continue;
    const a = xm[i + 1] / t;
    if (!n) aLow = a;
    sl += Math.log(a); n++; amin = Math.min(amin, a); amax = Math.max(amax, a);
  }
  const alpha = Math.exp(sl / n);
  console.log("        ALPHA = chi_measured / chi_true = " + alpha.toFixed(4)
    + "   (range " + amin.toFixed(3) + ".." + amax.toFixed(3) + " over " + n
    + " levels; " + aLow.toFixed(3) + " at the low-k⊥ end, where the ridge is a few bins up)");
  console.log("        REPORTED, NEVER GATED (CHI_PLAN): its job is to keep the hint honest. "
    + "Within tens of percent of 1, the hint's \"of order 1\" framing stands as written.");
  ok("alpha is measurable at all: both curves drawn, every ratio finite and positive",
     n >= 8 && isFinite(alpha) && alpha > 0 && amin > 0,
     n + " levels compared, chi_meas " + xm[1].toFixed(3) + ".." + xm[xm.length - 1].toFixed(3));
  // the DIRECTION is the falsifiable part of the argument above, and unlike the magnitude
  // it is not a matter of taste: a fattened parallel tail returns k∥ high, so chi low.
  ok("  ... and it lies on the LOW side, as the Hann-broadening argument says it must",
     alpha < 1.0 && alpha > 0.5, "alpha = " + alpha.toFixed(4));
}

// ---------------------------------------------------------------------------
console.log("8.5 CHECK 5, degenerates on the chi ordinate: empty curves, never NaN");
// ---------------------------------------------------------------------------
{
  const empty = A => A.curves.length === 0 && A.hi === 0 && A.lo === Infinity;
  const CH = { ay: "chi" };
  let IMB = null;                    // the maximally imbalanced case, re-used on the canvas
  ok("no data / silent / one-bin / no-overlap -> no curves, exactly as on the ratio axis",
     empty(C.anisoCurves(null, CH)) &&
     empty(C.anisoCurves({ perp: mk(64, () => 0), nb: 64, par: parLaw(3, -2), parKfac: 1 }, CH)) &&
     empty(C.anisoCurves({ perp: mk(2, () => 1), nb: 2, par: parLaw(3, -2), parKfac: 1 }, CH)) &&
     empty(C.anisoCurves(Object.assign({}, CASE_GS, { par: mk(4, () => 1e12) }),
                         { ad: "z", ay: "chi" })) &&
     empty(C.anisoCurves(Object.assign({}, CASE_GS, { par: null, parFL: null }), CH)));
  // the one degenerate chi has that the ratio does not: an Elsasser lane whose OPPOSITE
  // lane carries nothing. There is then no shearing field to divide by, so the honest
  // answer is no curve -- never the own-lane level quietly standing in for it.
  {
    const n = 4096, a = new Float32Array(3 * n);
    for (let j = 1; j < n; j++) { a[j] = Math.pow(j, -5 / 3); a[2 * n + j] = Math.pow(j, -5 / 3); }
    const d = IMB = { perp: a, nb: n, par: parLaw(3, -2), parFL: parLaw(3, -2), parKfac: 1,
                      fshell: [1, 5] };
    ok("a maximally imbalanced field (E- == 0) draws NO chi+ curve, rather than a wrong one",
       empty(C.anisoCurves(d, { aq: "zp", ad: "z", ay: "chi" })) &&
       empty(C.anisoCurves(d, { aq: "zm", ad: "z", ay: "chi" })),
       "and the ratio ordinate still draws E+: "
       + C.anisoCurves(d, { aq: "zp", ad: "z" }).curves.length + " curve");
    ok("  ... while its tot lane is untouched and finite",
       C.anisoCurves(d, { aq: "tot", ad: "z", ay: "chi" }).curves.length === 1 &&
       allFinite(C.anisoCurves(d, { aq: "tot", ad: "z", ay: "chi" }).curves[0][0]));
  }
  ok("NaN / Inf / negative bins are dropped on the chi ordinate too", (() => {
    const p = mk(PERP_N, j => (j === 0 ? 0 : Math.pow(j, -5 / 3)));
    p[7] = NaN; p[9] = Infinity; p[11] = -1;
    const A = C.anisoCurves(Object.assign({}, CASE_GS, { perp: p }), { ad: "z", ay: "chi" });
    return A.curves.length === 1 && allFinite(A.curves[0][0]);
  })());
  // and the canvas: the same recording context section 5 uses, over the same degenerates
  const recCtx = () => {
    const log = [];
    const o = { fillStyle: "", strokeStyle: "", lineWidth: 1, globalAlpha: 1,
                font: "10px x", textAlign: "left", textBaseline: "alphabetic" };
    for (const m of ["clearRect", "strokeRect", "beginPath", "moveTo", "lineTo", "stroke",
                     "fill", "clip", "save", "restore", "setLineDash", "rect", "fillRect"]) {
      o[m] = (...a) => {
        for (const v of a) if (typeof v === "number" && !isFinite(v)) log.push(["NONFINITE", m]);
      };
    }
    o.measureText = t => ({ width: 6.2 * t.length, actualBoundingBoxAscent: 7.2,
                            actualBoundingBoxDescent: 0 });
    o.fillText = t => log.push(["text", t]);
    o.log = log;
    return o;
  };
  const txt = c => c.log.filter(e => e[0] === "text").map(e => e[1]).join("|");
  const DEGEN = [null, {}, { nb: 4 }, { perp: mk(64, () => 0), nb: 64 },
                 Object.assign({}, CASE_GS, { par: null, parFL: null })];
  let anyNon = false, allWait = true;
  for (const d of DEGEN) {
    const c = recCtx(); C.drawAniso(c, d, { fit: "pin", ay: "chi" });
    if (c.log.some(e => e[0] === "NONFINITE")) anyNon = true;
    if (!/χ vs k⊥ — waiting…/.test(txt(c))) allWait = false;
  }
  ok("drawAniso says \"χ vs k⊥ — waiting…\" on all " + DEGEN.length + " degenerates", allWait);
  ok("  ... and nothing non-finite reached the canvas", !anyNon);
  // ... but the empty-opposite-lane case above is NOT one of them: it is a RESULT, and an
  // unflagged empty return drew "waiting…", promising a curve that was never coming
  // (review 2026-08-14). The flag must be reachable from the chi branch ALONE -- §8.1 is
  // what proves the ratio path did not MOVE, and this is what says it cannot get here.
  {
    const c = recCtx();
    C.drawAniso(c, IMB, { aq: "zp", ad: "z", fit: "pin", ay: "chi" });
    ok("the empty-opposite-lane case says \"no counterpropagating energy\", not \"waiting…\"",
       C.anisoCurves(IMB, { aq: "zp", ad: "z", ay: "chi" }).noShear === true &&
       /χ vs k⊥ — no counterpropagating energy/.test(txt(c)) && !/waiting/.test(txt(c)),
       txt(c));
    const cr = recCtx();
    C.drawAniso(cr, { nb: 1 }, { aq: "zp", ad: "z", fit: "pin" });
    ok("  ... and the flag is unreachable on the ratio ordinate, which still waits",
       C.anisoCurves(IMB, { aq: "zp", ad: "z" }).noShear === undefined &&
       DEGEN.every(d => C.anisoCurves(d, { aq: "zp", ad: "z" }).noShear === undefined) &&
       /k∥\/k⊥ vs k⊥ — waiting…/.test(txt(cr)),
       txt(cr));
  }
  const live = [];
  for (const ad of ["both", "z", "fl"]) for (const fit of ["pin", "amp", "off"]) {
    const c = recCtx();
    C.drawAniso(c, CASE_GS, { aq: "tot", ad, fit, ay: "chi", fitp: -0.333,
                              fita: fit === "amp" ? "2.5" : "" });
    live.push(c);
  }
  ok("  ... every live (ad x fit) combination draws, all finite, none waiting",
     live.every(c => !/waiting/.test(txt(c)) && !c.log.some(e => e[0] === "NONFINITE")),
     live.length + " combinations");
  // the reference line is a LEVEL here, not a slope: chi = 1 by default (ANISO_CHI_REF),
  // the amplitude box renames it, `off` hides it, and the -1/3 legend never appears
  ok("  ... the reference is the horizontal χ = 1, the amplitude box sets the level, "
     + "\"off\" hides it",
     C.ANISO_CHI_REF === 1 && /χ = 1/.test(txt(live[0])) && /χ = 2.5/.test(txt(live[1])) &&
     !/χ = /.test(txt(live[2])) && !/k⊥\^/.test(live.map(txt).join("|")),
     live.map(c => (txt(c).match(/χ = [0-9.]+/) || ["-"])[0]).join(","));
  // ... and that level is legended with SIGNIFICANT FIGURES, not with three decimals:
  // Math.round(x*1000)/1000 turned an amplitude of 4e-4 into "χ = 0" (review 2026-08-14),
  // which is a level the card exists to report claiming to be zero
  const levs = ["0.0004", "0.07", "1", "2.5", "1250"].map(A => {
    const c = recCtx();
    C.drawAniso(c, CASE_GS, { aq: "tot", ad: "z", fit: "amp", ay: "chi", fita: A });
    return (txt(c).match(/χ = [-+0-9.e]+/) || ["χ = ?"])[0].slice(4);
  });
  ok("  ... and a small reference level legends its value, not \"χ = 0\"",
     levs.join(",") === "4.0e-4,0.07,1,2.5,1.25e+3", levs.join(","));
  // ... and it must be INSIDE the frame, or "is the level 1?" is unanswerable. The y range
  // is the drawn extremes padded by 0.3 decades, so a curve sitting well below 1 has to
  // have pulled the top of the axis up to it.
  ok("  ... and the χ = 1 line is inside the axes even when the curve sits far below it",
     (() => {
       const quiet = Object.assign({}, CASE_GS, { par: parLaw(30, -2) });   // chi ~ 0.07
       const A = C.anisoCurves(quiet, { aq: "tot", ad: "z", ay: "chi" });
       const c = recCtx();
       C.drawAniso(c, quiet, { aq: "tot", ad: "z", fit: "pin", ay: "chi" });
       // the top of the axis is log10(max(hi, 1)) + 0.3, so 1 is inside it by construction;
       // assert the curve really is far below 1, so the leg is not vacuous
       return A.hi < 0.1 && /χ = 1/.test(txt(c));
     })());
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
  ok("  ... a card of that type builds, with its six option controls",
     !!card && card.optEls.length === 6, card ? "n=" + card.optEls.length : "none");
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
  ok("  ... and retyping back rebuilds the six and re-arms the gate",
     card.optEls.length === 6 && gate() === true, "n=" + card.optEls.length);
  ok("  ... optId lookup finds ay/aq/ad (the option ids the draw branches on)",
     !!optId(card, "ay") && !!optId(card, "aq") && !!optId(card, "ad"));
  // the ordinate select drives the card end to end: it defaults to the shipped ratio, it
  // does NOT arm the field-line readback by itself (that is `ad`'s job alone), and because
  // the hint is a function of the options the card must RE-RENDER it on the switch --
  // the gen2d colour-scale path, exercised here on a second consumer.
  {
    const feed = c => env.run("function(c, nb, nzb){"
      + " const a = new Float32Array(3 * nb), p = new Float32Array(3 * nzb);"
      + " for (let b = 1; b < nb; b++) { a[b] = Math.pow(b, -5/3); }"
      + " for (let b = 0; b < nzb; b++) { p[b] = 3 * Math.pow(b+1, -2); }"
      + " c.draw({ perp: a, nb: nb, fshell: [1,5], par: p, parFL: p, parKfac: 1, kunit: 1 });"
      + "}", c, 512, 128);
    const setAy = v => env.run("function(c, v){ const s = c.optEls.filter(x => x.__optId === 'ay')[0];"
                               + " s.value = v; s.onchange(); }", card, v);
    ok("  ... the ordinate select defaults to the shipped ratio",
       env.run("function(c){ return c.optVals().ay; }", card) === "ratio");
    feed(card);
    const h0 = env.run("function(c){ return c.hint.innerHTML; }", card);
    setAy("chi"); feed(card);
    const h1 = env.run("function(c){ return c.hint.innerHTML; }", card);
    ok("  ... switching it to χ re-renders the hint and draws without raising",
       h1 !== h0 && /&delta;b&sup2; = Q/.test(h1) && env.fails.length === 0,
       env.fails.join(" | ") || h1.slice(0, 60) + "...");
    ok("  ... and the ordinate does not touch the field-line readback gate (that is `ad`)",
       gate() === true);
    setAy("ratio"); feed(card);
    ok("  ... switching back restores Alfred's ratio copy verbatim",
       env.run("function(c){ return c.hint.innerHTML; }", card) === h0);
  }
  ok("3D boots clean through all of it", env.fails.length === 0, env.fails.join(" | "));
}

console.log(bad ? "\n" + bad + " FAILURE(S)" : "\nall checks passed");
process.exit(bad ? 1 : 0);
})();
