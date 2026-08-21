// GATE: expression initial conditions (IO_PLAN item 1) -- the hand-written parser, the
// non-finite guard, the per-axis periodicity detector, and the REAL app driving all
// three from its two text boxes.
//
//   node checkexpr.js [dir]        (dir defaults to ..)
//
// Everything here is the app's OWN code, reached through the shared stub (stubenv.js) in
// the page's context: both pages are booted, because the name set differs between them
// (z and Lz exist on the 3D page and are unknown names on the 2D one) and the grid layout
// gains an axis. The expected values are hand-computed in THIS file -- an index formula
// written out longhand, an analytic seam jump -- never read back off the thing under test.
"use strict";
const path = require("path");
const dir = process.argv[2] || path.join(__dirname, "..");

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log("  " + (pass ? "PASS" : "FAIL") + "  " + name + (note ? "   [" + note + "]" : ""));
};
const near = (a, b, tol) => Math.abs(a - b) <= (tol === undefined ? 1e-12 : tol) * Math.max(1, Math.abs(b));

// the two pages, and their expression machinery lifted out of the page context
const env2 = require("./stubenv")(dir, "rmhd2d.html");
const env3 = require("./stubenv")(dir, "rmhd3d.html");
const API = e => e.run(`function(){
  return { compile: exprCompile, evaluate: exprEval, field: exprField, seam: exprSeam,
           envOf: exprEnv, seamNote: exprSeamNote, icField: icExprField,
           tol: EXPR_SEAM_TOL, help: EXPR_HELP, name: IC_EXPR };
}`);
const A2 = API(env2), A3 = API(env3);
// the grids the unit sections work on: small, and deliberately NOT square or 2pi-sized,
// so a swapped axis or a hardcoded box length cannot pass
const G2 = { nx: 6, ny: 4, nz: 1, Lx: 2, Ly: 3, Lz: 0 };
const G3 = { nx: 5, ny: 4, nz: 3, Lx: 2, Ly: 3, Lz: 5 };
// compile + evaluate at one point, on the 2D name set unless a 3D api is passed
function ev(src, x, y, z, A, g) {
  const a = A || A2, p = a.compile(src, a.envOf(g || G2));
  if (p.err) return { err: p.err };
  return a.evaluate(p, [x, y, z === undefined ? 0 : z]);
}

// ===========================================================================
console.log("1. tokenizer and shunting-yard: precedence, associativity, calls");
// ===========================================================================
{
  // `^` is right-associative and binds TIGHTER than unary minus -- the two the plan
  // pins by name, because every other reading of them is also defensible
  const CASES = [
    ["1+2*3", 7], ["(1+2)*3", 9], ["1-2-3", -4], ["8/4/2", 1], ["8/(4/2)", 4],
    ["2^3^2", 512], ["-2^2", -4], ["(-2)^2", 4], ["2^-1", 0.5], ["-3^2*2", -18],
    ["+3", 3], ["- -2", 2], ["1 + 2", 3], [".5+1", 1.5], ["2e-3", 0.002], ["1.", 1],
    ["pi", Math.PI], ["e", Math.E], ["Lx", G2.Lx], ["Ly", G2.Ly],
    ["log(e)", 1], ["log10(1000)", 3], ["sqrt(9)", 3], ["abs(0-4)", 4], ["sign(0-2)", -1],
    ["hypot(3,4)", 5], ["pow(2,10)", 1024], ["atan2(1,2)", Math.atan2(1, 2)],
    ["min(1,max(2,3))", 1], ["max(1,min(2,3))", 2], ["floor(1.7)", 1], ["ceil(1.2)", 2],
    ["mod(-1,3)", 2], ["mod(7,3)", 1], ["mod(1,-3)", -2],           // FLOORED modulo
    ["sin(cos(0))", Math.sin(1)], ["atan2(sin(0),cos(0))", 0],
    ["tanh(0)+cosh(0)+sinh(0)", 1], ["asin(1)", Math.PI / 2], ["acos(1)", 0],
    ["atan(1)", Math.PI / 4], ["tan(0)", 0], ["exp(0)", 1]
  ];
  let worst = "", nbadv = 0;
  for (const c of CASES) {
    const v = ev(c[0], 0, 0);
    if (v && v.err) { nbadv++; worst = c[0] + " -> " + v.err; continue; }
    if (!near(v, c[1], 1e-14)) { nbadv++; worst = c[0] + " = " + v + ", want " + c[1]; }
  }
  ok("every precedence / associativity / function case evaluates to its hand value",
     nbadv === 0, nbadv ? worst : CASES.length + " cases");

  // -x^2 with x a VARIABLE, not a literal -- the case a constant-folding shortcut would
  // get right for the wrong reason
  ok("-x^2 is -(x^2)", near(ev("-x^2", 3, 0), -9) && near(ev("(-x)^2", 3, 0), 9),
     "-x^2 = " + ev("-x^2", 3, 0) + ", (-x)^2 = " + ev("(-x)^2", 3, 0));

  // ... and the RPN itself, not just its value. Opcodes: 0 literal, 1 variable,
  // 23 pow (which `^` compiles to), 25 +, 27 *, 29 unary minus.
  const rpn = s => Array.from(A2.compile(s, A2.envOf(G2)).code);
  ok("-x^2 compiles to [x, 2, ^, neg]", JSON.stringify(rpn("-x^2")) === "[1,0,23,29]",
     JSON.stringify(rpn("-x^2")));
  ok("2^3^2 compiles right-associatively to [2, 3, 2, ^, ^]",
     JSON.stringify(rpn("2^3^2")) === "[0,0,0,23,23]", JSON.stringify(rpn("2^3^2")));
  ok("1+2*3 compiles to [1, 2, 3, *, +]", JSON.stringify(rpn("1+2*3")) === "[0,0,0,27,25]",
     JSON.stringify(rpn("1+2*3")));

  // nested calls and mixed arity in one expression
  const nest = ev("atan2(sin(x)*2, max(cos(y), 0.5))", 0.3, 0.7);
  ok("nested calls of mixed arity", near(nest, Math.atan2(Math.sin(0.3) * 2, Math.max(Math.cos(0.7), 0.5))),
     "= " + nest);

  // the evaluator is reused across points and must not accumulate state
  const p = A2.compile("x*x + y", A2.envOf(G2));
  let drift = 0;
  for (let i = 0; i < 50; i++) drift = Math.max(drift, Math.abs(A2.evaluate(p, [2, 3, 0]) - 7));
  ok("one compiled program re-evaluates identically 50 times", drift === 0, "drift " + drift);
}

// ===========================================================================
console.log("2. every error path is a message with the right character position");
// ===========================================================================
{
  // [source, substring of the message, 0-based index it must point at]
  const ERRS = [
    ["sin(x)*sn(y)", "unknown name 'sn'", 7],
    ["sn", "unknown name 'sn'", 0],
    ["x +", "ends where a value was expected", 2],
    ["", "ends where a value was expected", 0],
    ["x + * y", "expected a value before '*'", 4],
    ["(x", "unclosed '('", 0],
    ["x)", "unmatched ')'", 1],
    ["sin(x", "unclosed 'sin('", 0],
    ["atan2(1)", "'atan2' takes 2 arguments, got 1", 0],
    ["sin(1,2)", "'sin' takes 1 argument, got 2", 0],
    ["min(1,2,3)", "'min' takes 2 arguments, got 3", 0],
    ["1+sin", "'sin' is a function", 2],
    ["x y", "expected an operator before 'y'", 2],
    ["2 3", "expected an operator before '3'", 2],
    ["2(x)", "expected an operator before '('", 1],
    ["x $ y", "unexpected character '$'", 2],
    ["1e", "bad number '1e'", 0],
    // the quote is what the USER typed, not the prefix the number matcher stopped at
    ["1e+", "bad number '1e+'", 0],
    ["1e-", "bad number '1e-'", 0],
    ["1e+x", "bad number '1e+x'", 0],
    ["x + 2x", "bad number '2x'", 4],
    ["1x2", "bad number '1x2'", 0],
    ["1..", "bad number '1..'", 0],
    // an open paren still on the stack outranks "ends where a value was expected":
    // both are true here, and only one of them is actionable
    ["(", "unclosed '('", 0],
    ["(1+", "unclosed '('", 0],
    ["sin(1+", "unclosed 'sin('", 0],
    ["1+2*(3", "unclosed '('", 4],
    ["(1,2)", "',' outside a function call", 2],
    ["sin()", "expected a value before ')'", 4],
    ["sin(,1)", "expected a value before ','", 4]
  ];
  // ... and the two messages that are NOT displaced by the paren rule: with nothing open,
  // a trailing operator is still reported as a missing value, at the operator
  for (const c of [["1+", 1], ["x +", 2], ["", 0]]) {
    const r = A2.compile(c[0], A2.envOf(G2));
    ok("a trailing operator with no open paren still says a value was expected ("
       + JSON.stringify(c[0]) + ")",
       !!r.err && /ends where a value was expected/.test(r.err) && r.at === c[1], r.err);
  }
  let nthrow = 0, nmsg = 0, nat = 0, note = "";
  for (const c of ERRS) {
    let r;
    try { r = A2.compile(c[0], A2.envOf(G2)); }
    catch (e) { nthrow++; note = c[0] + " THREW " + e.message; continue; }
    if (!r.err || r.err.indexOf(c[1]) < 0) { nmsg++; note = c[0] + " -> " + (r.err || "(compiled)"); continue; }
    // the message quotes a 1-BASED position; `at` is the 0-based source index it came from
    if (r.at !== c[2] || r.err.indexOf("at character " + (c[2] + 1)) < 0) {
      nat++; note = c[0] + " -> " + r.err + " (at " + r.at + ", want " + c[2] + ")";
    }
  }
  ok("no error path throws", nthrow === 0, nthrow ? note : ERRS.length + " bad inputs");
  ok("every error names what is wrong", nmsg === 0, nmsg ? note : "");
  ok("every error points at the right character", nat === 0, nat ? note : "1-based in the text");
  // ... and nothing silently returns zero
  const z = A2.compile("sn(x)", A2.envOf(G2));
  ok("an unknown name is an error, not a silent zero", !!z.err && z.code === undefined, z.err);
}

// ===========================================================================
console.log("3. the name set: z and Lz exist on the 3D page only");
// ===========================================================================
{
  for (const n of ["z", "Lz"]) {
    const r = A2.compile("x + " + n, A2.envOf(G2));
    ok("2D rejects '" + n + "'", !!r.err && r.err.indexOf("unknown name '" + n + "'") === 0
       && r.err.indexOf("no z axis") > 0 && r.at === 4, r.err);
  }
  ok("3D accepts z", near(ev("z", 0, 0, 1.25, A3, G3), 1.25));
  ok("3D accepts Lz", near(ev("Lz", 0, 0, 0, A3, G3), G3.Lz), "Lz = " + ev("Lz", 0, 0, 0, A3, G3));
  // the names both pages share, so the 2D rejection above is about z and nothing else
  let shared = true;
  for (const n of ["x", "y", "Lx", "Ly", "pi", "e"]) {
    if (A2.compile(n, A2.envOf(G2)).err || A3.compile(n, A3.envOf(G3)).err) shared = false;
  }
  ok("x y Lx Ly pi e resolve on both pages", shared);
  // no RNG smuggled in: the same string must give the same field twice
  ok("no random() in the name set", !!A3.compile("random()", A3.envOf(G3)).err
     && !!A3.compile("rand", A3.envOf(G3)).err);
  // the help line must say which log it is (half the audience reads log as base 10)
  ok("the help line says log is the natural log", /NATURAL/.test(A2.help) && /log10/.test(A2.help),
     A2.help.length + " chars");
}

// ===========================================================================
console.log("4. evaluation over the grid, in setICFromReal's own index order");
// ===========================================================================
{
  // 2D: ix*ny + iy, written out longhand here rather than taken from the app
  const p = A2.compile("x + 10*y", A2.envOf(G2));
  const r = A2.field(p, G2);
  let e = 0;
  for (let ix = 0; ix < G2.nx; ix++) {
    for (let iy = 0; iy < G2.ny; iy++) {
      const want = Math.fround(ix * G2.Lx / G2.nx + 10 * (iy * G2.Ly / G2.ny));
      e = Math.max(e, Math.abs(r.f[ix * G2.ny + iy] - want));
    }
  }
  ok("2D field is x + 10y at ix*ny + iy, in CODE units", e === 0 && r.nbad === 0,
     "max err " + e + ", length " + r.f.length);

  // 3D: (iz*nx + ix)*ny + iy, and a z dependence that would alias onto x if the two
  // outer loops were swapped
  const p3 = A3.compile("x + 10*y + 100*z", A3.envOf(G3));
  const r3 = A3.field(p3, G3);
  let e3 = 0;
  for (let iz = 0; iz < G3.nz; iz++) {
    for (let ix = 0; ix < G3.nx; ix++) {
      for (let iy = 0; iy < G3.ny; iy++) {
        const want = Math.fround(ix * G3.Lx / G3.nx + 10 * (iy * G3.Ly / G3.ny)
                                 + 100 * (iz * G3.Lz / G3.nz));
        e3 = Math.max(e3, Math.abs(r3.f[(iz * G3.nx + ix) * G3.ny + iy] - want));
      }
    }
  }
  ok("3D field is x + 10y + 100z at (iz*nx + ix)*ny + iy", e3 === 0 && r3.nbad === 0,
     "max err " + e3 + ", length " + r3.f.length);
  ok("the field is exactly nx*ny*nz long on both pages",
     r.f.length === G2.nx * G2.ny && r3.f.length === G3.nx * G3.ny * G3.nz);
  // the box ENDS at Lx: the last sample is one cell short of it, never on it
  ok("the grid stops one cell short of the far face",
     near(r.f[(G2.nx - 1) * G2.ny], Math.fround(G2.Lx * (G2.nx - 1) / G2.nx), 1e-6),
     "x_max = " + r.f[(G2.nx - 1) * G2.ny] + " of Lx = " + G2.Lx);
}

// ===========================================================================
console.log("5. the non-finite guard");
// ===========================================================================
{
  const guard = (src, A, g) => A.field(A.compile(src, A.envOf(g)), g);
  const a = guard("1/x", A2, G2);
  ok("1/x is caught: one bad value per y on the x = 0 column",
     a.nbad === G2.ny && a.bad[0] === 0 && a.bad[1] === 0,
     a.nbad + " non-finite, first at (" + a.bad + ")");
  const b = guard("log(y)", A2, G2);
  ok("log(y) is caught on the y = 0 row", b.nbad === G2.nx && b.bad[1] === 0,
     b.nbad + " non-finite, first at (" + b.bad + ")");
  const c = guard("sqrt(0-1)", A2, G2);
  ok("a NaN everywhere is caught everywhere", c.nbad === G2.nx * G2.ny, c.nbad + " non-finite");
  const d = guard("1/z", A3, G3);
  ok("3D reports the offending z too", d.nbad === G3.nx * G3.ny && d.bad[2] === 0,
     d.nbad + " non-finite, first at (" + d.bad + ")");
  const g = guard("sin(2*pi*x/Lx)", A2, G2);
  ok("a finite field trips nothing", g.nbad === 0 && g.bad === null);

  // -- finite as a double, Infinity as the float32 that is actually STORED ----
  // The field is a Float32Array, so the guard has to test the value after the write,
  // not the double before it: exp(100) = 2.7e43 is a perfectly good double and is
  // +Infinity the moment it lands, and a guard reading `r` waves it through into the
  // forward FFT. |f| > 3.4028235e38 is the whole of the extra ground covered here.
  const o1 = guard("exp(100)", A2, G2);
  ok("exp(100) overflows float32 and is caught at EVERY point",
     o1.nbad === G2.nx * G2.ny && o1.bad !== null && o1.bad[0] === 0 && o1.bad[1] === 0,
     o1.nbad + " non-finite of " + (G2.nx * G2.ny) + ", first at (" + o1.bad + ")");
  const o2 = guard("10^40", A2, G2);
  ok("10^40 likewise", o2.nbad === G2.nx * G2.ny, o2.nbad + " non-finite");
  const o3 = guard("0-exp(100)", A2, G2);
  ok("... and -Infinity too, not just +", o3.nbad === G2.nx * G2.ny, o3.nbad + " non-finite");
  // the PARTIAL case, which is what a plausible sharp profile looks like: on G2,
  // x = ix*Lx/nx = 0, 1/3, 2/3, 1, 4/3, 5/3, and exp(100*x) passes float32 at the
  // first three (up to 9.2e28) and overflows at the last three -- ny points each,
  // first at x = 1 in the layout's own order
  const o4 = guard("exp(x*100)", A2, G2);
  ok("exp(x*100) overflows on PART of the grid and names the first point",
     o4.nbad === 3 * G2.ny && o4.bad[0] === G2.Lx * 3 / G2.nx && o4.bad[1] === 0,
     o4.nbad + " non-finite of " + (G2.nx * G2.ny) + ", first at (" + o4.bad + ")");
  // just under the float32 ceiling: still a number, still uploaded
  const o5 = guard("3e38", A2, G2);
  ok("3e38 is under the float32 ceiling and trips nothing", o5.nbad === 0 && o5.bad === null);
  const o6 = guard("1e-60", A2, G2);
  ok("an underflow to zero is a NUMBER, not a refusal", o6.nbad === 0, o6.nbad + " non-finite");
  const o7 = guard("exp(z*100)", A3, G3);
  ok("3D overflows report the offending z", o7.nbad > 0 && o7.bad[2] === G3.Lz / G3.nz,
     o7.nbad + " non-finite, first at (" + o7.bad + ")");

  // ... and the refusal: the field is not handed on, and the line says so
  const ref = A2.icField("1/x", G2, "phi");
  ok("a non-finite field is REFUSED, not uploaded", ref.f === null
     && /not uploaded/.test(ref.note) && /4 non-finite values/.test(ref.note), ref.note);
  const good = A2.icField("sin(2*pi*x/Lx)", G2, "phi");
  ok("a finite periodic field is handed on with no note", good.f !== null && good.note === "");
  const empty = A2.icField("", G2, "phi");
  ok("an empty box is null = exactly zero", empty.f === null && empty.note === "");
  const perr = A2.icField("sn(x)", G2, "phi");
  ok("a parse error is reported and nothing is uploaded",
     perr.f === null && /unknown name 'sn'/.test(perr.note), perr.note);

  // ... and the refusal covers the float32 overflows, with the same count-and-place line
  const ro = A2.icField("exp(100)", G2, "phi");
  ok("an all-Infinity float32 field is REFUSED, not uploaded",
     ro.f === null && /not uploaded/.test(ro.note)
     && /24 non-finite values/.test(ro.note) && /x = 0\.00/.test(ro.note), ro.note);
  const rp = A2.icField("exp(x*100)", G2, "phi");
  ok("a PARTLY infinite field is refused too, and names its first point",
     rp.f === null && /12 non-finite values/.test(rp.note)
     && /x = 1\.00/.test(rp.note) && /y = 0\.00/.test(rp.note), rp.note);
  const rq = A2.icField("10\u005e40", G2, "phi");
  ok("10^40 the same", rq.f === null && /not uploaded/.test(rq.note), rq.note);
}

// ===========================================================================
console.log("6. the periodicity detector: per axis, worst seam named");
// ===========================================================================
{
  const G = { nx: 64, ny: 64, nz: 1, Lx: 4 * Math.PI, Ly: 2 * Math.PI, Lz: 0 };
  const note = src => {
    const p = A2.compile(src, A2.envOf(G));
    return A2.seamNote(p, G, A2.field(p, G).f, "phi");
  };
  const seams = src => {
    const p = A2.compile(src, A2.envOf(G));
    return [A2.seam(p, G, 0), A2.seam(p, G, 1)];
  };

  // -- periodic: nothing to say -------------------------------------------
  ok("sin(2*pi*x/Lx) is periodic -- no warning", note("sin(2*pi*x/Lx)") === "",
     JSON.stringify(seams("sin(2*pi*x/Lx)")));
  ok("sin(2*pi*x/Lx)*cos(4*pi*y/Ly) is periodic in both -- no warning",
     note("sin(2*pi*x/Lx)*cos(4*pi*y/Ly)") === "");
  const s = seams("sin(2*pi*x/Lx)")[0];
  ok("the x seam of a periodic field is round-off in value AND slope",
     s[0] < 1e-14 && s[1] < 1e-14, "j0 = " + s[0].toExponential(1) + ", j1 = " + s[1].toExponential(1));

  // -- discontinuous: x*y jumps across both seams, x worse ----------------
  // f = x*y on the grid: the x seam jumps by Lx*y (max ~Lx*Ly), the y seam by x*Ly
  // (max ~(Lx-h)*Ly), and the range is (Lx-h)(Ly-k), so BOTH exceed 1 and x wins.
  const nxy = note("x*y");
  ok("x*y is not periodic and the WORST seam (x) is the one named",
     /x seam/.test(nxy) && /jump/.test(nxy) && !/kinked/.test(nxy), nxy.slice(0, 60));
  ok("the x*y warning says what will actually run",
     /band-limited projection/.test(nxy) && /broadband spectral tail/.test(nxy));
  const jxy = seams("x*y");
  ok("both of x*y's seams are detected, not just the reported one",
     jxy[0][0] > 1 && jxy[1][0] > 1,
     "j0 x = " + jxy[0][0].toPrecision(3) + ", y = " + jxy[1][0].toPrecision(3));

  // -- periodic in one axis, not the other: the naming is the whole point --
  const nxy2 = note("sin(2*pi*x/Lx)*y");
  ok("sin(2*pi*x/Lx)*y names the y seam, not x", /y seam/.test(nxy2), nxy2.slice(0, 40));

  // -- continuous but kinked ----------------------------------------------
  // abs(x - Lx/2) matches at both ends (Lx/2 either side) but its slope flips: the
  // one-cell increment is -h inside x = 0 and +h arriving at x = Lx, so j1 = 2h exactly
  // and j1/range = 4/nx.
  const nk = note("abs(x-Lx/2)");
  const jk = seams("abs(x-Lx/2)")[0];
  ok("abs(x-Lx/2) is continuous across the seam", jk[0] === 0, "j0 = " + jk[0]);
  ok("... and its slope jump is the analytic 2h", near(jk[1], 2 * G.Lx / G.nx, 1e-6),
     "j1 = " + jk[1].toPrecision(6) + ", 2h = " + (2 * G.Lx / G.nx).toPrecision(6));
  ok("... so it is warned about as KINKED, in different words from a jump",
     /x seam/.test(nk) && /continuous but kinked/.test(nk) && /milder ringing/.test(nk),
     nk.slice(0, 70));
  ok("... at the analytic ratio 4/nx of range",
     new RegExp("slope jumps 0\\.06[0-9]* of range per cell").test(nk),
     "4/nx = " + (4 / G.nx));

  // -- the threshold is a threshold ---------------------------------------
  // a deliberately tiny mismatch: sin(2*pi*x/Lx) + eps*x/Lx jumps by eps of a range ~2
  ok("a jump below 1e-3 of range says nothing", note("sin(2*pi*x/Lx)+0.0001*x/Lx") === "",
     "tol = " + A2.tol);
  ok("a jump above 1e-3 of range says something",
     note("sin(2*pi*x/Lx)+0.01*x/Lx") !== "");

  // -- a field with NO range: nothing to normalize by, and nothing to say ---
  // The normalizer falls back from the range to the magnitude and then to 1 precisely
  // because a constant is a legal expression. Divide by a bare `hi - lo` instead and
  // every ratio here is NaN or Infinity: NaN loses both comparisons, so the seam
  // survives the threshold test AND picks the "kinked" branch, and phi = 1 reports
  // "slope jumps NaN of range per cell".
  for (const c of ["1", "0", "pi", "0*x", "0-1"]) {
    ok("a constant field (" + c + ") is perfectly periodic -- empty note", note(c) === "",
       JSON.stringify(note(c)).slice(0, 70));
  }
  // floor(x/Lx) is the case the code's own comment cites: exactly 0 on the whole grid
  // (x never reaches Lx) and 1 on the far face, so it has zero range and a real jump
  const nfl = note("floor(x/Lx)");
  ok("floor(x/Lx) has zero range on the grid but a REAL x seam, reported as a number",
     /x seam/.test(nfl) && /jump 1\.0 of range/.test(nfl) && !/NaN|Infinity/.test(nfl),
     nfl.slice(0, 60));

  // -- 3D: the z seam is checked and named --------------------------------
  const GZ = { nx: 32, ny: 32, nz: 16, Lx: 2 * Math.PI, Ly: 2 * Math.PI, Lz: 8 };
  const p3 = A3.compile("sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*(1+z/Lz)", A3.envOf(GZ));
  const n3 = A3.seamNote(p3, GZ, A3.field(p3, GZ).f, "psi");
  ok("3D names the z seam when x and y are clean", /z seam/.test(n3) && /^psi:/.test(n3),
     n3.slice(0, 40));
  const pz = A3.compile("sin(2*pi*z/Lz)", A3.envOf(GZ));
  ok("a z-periodic 3D field warns about nothing",
     A3.seamNote(pz, GZ, A3.field(pz, GZ).f, "psi") === "");
  // The seam stencil must be exact for a periodic expression at ANY resolution. A cosine
  // is EVEN about the seam, so its one-cell increments taken from INSIDE the grid are
  // equal and opposite -- an inside-only slope test reads h^2 f'' = (2pi/n)^2 there and
  // calls a perfectly smooth field kinked (it did, at nz = 16: 0.019 of range against a
  // 1e-3 threshold). Evaluating one cell OUTSIDE each face is what removes that floor.
  let coarse = "";
  for (const n of [8, 16, 32]) {
    const gc = { nx: 8, ny: 8, nz: n, Lx: 2 * Math.PI, Ly: 2 * Math.PI, Lz: 8 };
    const pc = A3.compile("cos(2*pi*z/Lz)", A3.envOf(gc));
    const s = A3.seam(pc, gc, 2);
    if (!(s[0] < 1e-12 && s[1] < 1e-12)) coarse += " nz=" + n + " -> " + JSON.stringify(s);
  }
  ok("a COSINE (even about the seam) reads as periodic at every resolution", coarse === "",
     coarse || "nz = 8, 16, 32 all at round-off");
  // ... and the 2D detector never looks for a z seam it has no axis for
  const p2 = A2.compile("x", A2.envOf(G));
  ok("2D checks exactly two seams", A2.seamNote(p2, G, A2.field(p2, G).f, "phi").indexOf("z seam") < 0);
}

// ===========================================================================
// 7. the real app: the preset, its rows, and what reaches setICFromReal
// ===========================================================================
// Driven through the booted page, not through the helpers above: the boxes, the preset
// select's own onchange, and the solver call at the end of applyIC.
function appLegs(env, tag, is3d) {
  const g = env.run("function(){ return { nx: solver.p.nx, ny: solver.p.ny, nz: solver.p.nz || 1, "
                    + "Lx: solver.p.Lx, Ly: solver.p.Ly, Lz: solver.p.Lz || 0 }; }");
  // record every setICFromReal call, then let the real one run
  env.run(`function(){
    window.__icLog = [];
    const real = solver.setICFromReal.bind(solver);
    solver.setICFromReal = function (phi, psi) {
      window.__icLog.push({ phi: phi ? Array.from(phi.subarray(0, 8)) : null,
                            nphi: phi ? phi.length : 0, psi: psi ? Array.from(psi.subarray(0, 8)) : null });
      return real(phi, psi);
    };
  }`);
  const drive = env.run(`function(){ return function (phi, psi) {
    const s = document.getElementById("selIC");
    document.getElementById("tExprP").value = phi;
    document.getElementById("tExprM").value = psi;
    s.value = "expr"; s.onchange();                       // syncIC + applyIC + syncLabels
    const m = document.getElementById("vExprMsg");
    return { rows: ["rowExprP", "rowExprM"].map(i => document.getElementById(i).style.display),
             amp: ["rowAmpP", "rowAmpM"].map(i => document.getElementById(i).style.display),
             help: document.getElementById("vExprHelp").style.display,
             msg: m.innerHTML, msgShown: m.style.display,
             log: window.__icLog.splice(0) };
  }; }`);

  ok(tag + ": the expression preset is offered", env.run(`function(){
    const o = document.getElementById("selIC").options;
    for (let i = 0; i < o.length; i++) if (o[i].value === "expr") return true;
    return false;
  }`));

  // -- a plain periodic expression reaches the solver ----------------------
  const src = is3d ? "sin(2*pi*x/Lx)*cos(2*pi*z/Lz)" : "sin(2*pi*x/Lx)*cos(2*pi*y/Ly)";
  const r = drive(src, "");
  ok(tag + ": selecting it shows both boxes and the help line",
     r.rows[0] === "" && r.rows[1] === "" && r.help === "", JSON.stringify(r.rows));
  ok(tag + ": ... and leaves the amp rows hidden (they normalize nothing here)",
     r.amp[0] === "none" && r.amp[1] === "none", JSON.stringify(r.amp));
  ok(tag + ": one setICFromReal call, phi of nx*ny*nz, psi null",
     r.log.length === 1 && r.log[0].nphi === g.nx * g.ny * g.nz && r.log[0].psi === null,
     JSON.stringify({ n: r.log.length, nphi: r.log[0] && r.log[0].nphi,
                      want: g.nx * g.ny * g.nz }));
  // the first eight entries are iy = 0..7 at ix = iz = 0, i.e. the layout, hand-computed
  let e = 0;
  for (let iy = 0; iy < 8; iy++) {
    const want = is3d ? Math.fround(Math.sin(0) * Math.cos(0)) : Math.fround(Math.sin(0) * Math.cos(2 * Math.PI * (iy * g.Ly / g.ny) / g.Ly));
    e = Math.max(e, Math.abs(r.log[0].phi[iy] - want));
  }
  ok(tag + ": the uploaded values are the expression at those grid points", e === 0, "max err " + e);
  ok(tag + ": a periodic expression draws no warning", r.msg === "" && r.msgShown === "none",
     r.msg.slice(0, 50));

  // -- 1/x refuses to upload ----------------------------------------------
  const bad1 = drive("1/x", "");
  ok(tag + ": 1/x is refused -- phi reaches the solver as null",
     bad1.log.length === 1 && bad1.log[0].phi === null, JSON.stringify(bad1.log));
  ok(tag + ": ... and the line says how many and where",
     /non-finite value/.test(bad1.msg) && /not uploaded/.test(bad1.msg)
     && /x = 0\.00/.test(bad1.msg) && bad1.msgShown === "", bad1.msg);

  // -- the seam warning names its axis, and the run is NOT blocked ---------
  const warn = drive("x*y", "");
  ok(tag + ": x*y warns, names the x seam, and still uploads",
     /x seam/.test(warn.msg) && /jump/.test(warn.msg) && warn.log[0].phi !== null
     && warn.msgShown === "", warn.msg.slice(0, 70));
  const warnY = drive("sin(2*pi*x/Lx)*y", "");
  ok(tag + ": a field periodic in x and not y names the y seam", /y seam/.test(warnY.msg),
     warnY.msg.slice(0, 40));

  // -- both boxes, and psi warned about under its own name ----------------
  const both = drive(src, "y");
  ok(tag + ": both boxes upload, and psi's seam is reported as psi's",
     both.log[0].phi !== null && both.log[0].psi !== null
     && /&psi;: y seam/.test(both.msg) && !/&phi;:/.test(both.msg), both.msg.slice(0, 60));

  // -- typing: the live parse names a bad function before Run -------------
  const typed = env.run(`function(){
    const b = document.getElementById("tExprP");
    b.value = "sin(x)*sn(y)"; b.oninput();
    const m = document.getElementById("vExprMsg");
    return { msg: m.innerHTML, shown: m.style.display, log: window.__icLog.splice(0) };
  }`);
  ok(tag + ": typing an unknown name reports it at once, and uploads nothing",
     /unknown name 'sn' at character 8/.test(typed.msg) && typed.shown === ""
     && typed.log.length === 0, typed.msg);

  // -- leaving the preset takes its warning with it -----------------------
  const away = env.run(`function(){
    const s = document.getElementById("selIC");
    s.value = "modes"; s.onchange();
    const m = document.getElementById("vExprMsg");
    return { msg: m.innerHTML, shown: m.style.display,
             rows: ["rowExprP", "rowExprM"].map(i => document.getElementById(i).style.display),
             help: document.getElementById("vExprHelp").style.display };
  }`);
  ok(tag + ": switching away hides the boxes, the help and the warning",
     away.shown === "none" && away.msg === "" && away.help === "none"
     && away.rows.join() === "none,none", JSON.stringify(away));
}

// ===========================================================================
// 8. cost of one build, and the instrument's own tax
// ===========================================================================
// The parser runs INSIDE stubenv's vm context, where `Math` belongs to the outer realm:
// every Math.sin the evaluator makes is a cross-context call V8 cannot inline. That is
// an artifact of the harness and not of the app, and it is most of what is timed below,
// so the tax is measured first and the browser-side figure quoted with it.
//
// The ASSERTION is a ratio, never an absolute millisecond ceiling. Two calibration loops
// run in the same process and the same vm realm, immediately before the thing they bound
// -- one cross-realm Math.sin, one of plain double arithmetic -- and the build's time is
// compared to a budget built out of THEM. A busy machine inflates the calibration and the
// measurement together, so the ratio holds where a wall-clock ceiling cries wolf (the
// reviewer measured 4.5x inflation under contention against 2.9x of headroom). What the
// ratio still catches is an ALGORITHMIC regression: reinstating the closure table the
// opcode switch replaced costs 2.3x, and HEADROOM below leaves less than that.
const NSIN = 2e7, NARITH = 2e7, ARITH_OPS = 6;
// Each leg's cost as a MULTIPLE of its calibrated budget, measured on a quiet laptop.
// The calibration loop is a unit of this machine's current speed, not a model of the
// interpreter -- the dispatch leg is legitimately ~9x it (a switch, two typed-array
// loads and a bounds check per op against a straight-line flop), and pinning that
// multiple is the point. HEADROOM is what a leg may exceed its reference by before this
// is called a regression: 1.7x, wide enough for another CPU and another V8, and still
// well under the 2.3x that reinstating the closure table costs.
const REF = { disp: 9.4, build: 1.75 }, HEADROOM = 1.7;
function mathTax() {
  const spin = env2.run(`function(){ return function (n) {
    let a = 0;
    for (let i = 0; i < n; i++) a += Math.sin(i * 1e-7);
    return a;
  }; }`);
  // plain double arithmetic in the same realm: ARITH_OPS flops per iteration, no call,
  // no allocation. This is the unit the interpreter's dispatch is priced in.
  const spinA = env2.run(`function(){ return function (n) {
    let a = 0.5, b = 1.25;
    for (let i = 0; i < n; i++) { a = a + b * 0.5; b = b - a * 0.25; a = a * 0.9999; b = b / 1.0001; }
    return a + b;
  }; }`);
  spin(1e6); spinA(1e6);                                    // warm
  const tv = process.hrtime.bigint();
  spin(NSIN);
  const inVm = Number(process.hrtime.bigint() - tv) / 1e6;
  const t0 = process.hrtime.bigint();
  let a = 0;
  for (let i = 0; i < NSIN; i++) a += Math.sin(i * 1e-7);
  const nat = Number(process.hrtime.bigint() - t0) / 1e6;
  const ta = process.hrtime.bigint();
  const b = spinA(NARITH);
  const arith = Number(process.hrtime.bigint() - ta) / 1e6;
  return { inVm: inVm, nat: nat, ratio: inVm / nat, a: a, b: b,
           // ns per cross-realm sin, and ns per plain arithmetic op, both in the vm
           sinNs: 1e6 * inVm / NSIN, opNs: 1e6 * arith / (NARITH * ARITH_OPS) };
}
// One build, priced against the calibration rather than against the clock. The op mix is
// read off the COMPILED program, not hand-counted: `code.length` is exactly the stack ops
// per point, and the transcendental calls are counted from the source (a `^` would be
// priced as arithmetic, which none of these strings has).
const EXPR_TRANS = /\b(?:sin|cos|tan|asin|acos|atan|atan2|sinh|cosh|tanh|exp|log|log10|sqrt|pow)\s*\(/g;
function cost(A, g, src, want, tax) {
  const p = A.compile(src, A.envOf(g));
  A.field(p, g);                                            // warm
  const t0 = process.hrtime.bigint();
  const r = A.field(p, g);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  const n = g.nx * g.ny * g.nz;
  const ntr = (src.match(EXPR_TRANS) || []).length, nop = p.code.length - ntr;
  const budget = n * (ntr * tax.sinNs + nop * tax.opNs) / 1e6;
  // the transcendental calls at the vm's rate vs the native one: the rest of the program
  // is plain arithmetic and pays the vm's general ~2.7x, so this is a BOUND on the
  // browser cost, not a prediction of it
  const trans = ntr * n * (tax.inVm - tax.nat) / NSIN;
  ok("build " + g.nx + "x" + g.ny + (g.nz > 1 ? "x" + g.nz : "") + ": " + ms.toFixed(0)
     + " ms in the stub, < " + (ms - trans).toFixed(0) + " ms with a native Math (plan: "
     + want + ")", r.nbad === 0 && ms < HEADROOM * REF.build * budget,
     (1e6 * ms / n).toFixed(0) + " ns/point raw, " + (n / 1e6).toFixed(2)
     + "e6 points, " + (ms / budget).toFixed(2) + "x its calibrated budget of "
     + budget.toFixed(0) + " ms (reference " + REF.build + "x, ceiling "
     + (HEADROOM * REF.build).toFixed(2) + "x)");
}

setTimeout(() => {
  console.log("7. the real app, both pages");
  appLegs(env2, "rmhd2d", false);
  appLegs(env3, "rmhd3d", true);
  console.log("8. cost of one build (one button press, not a frame)");
  const tax = mathTax();
  ok("the stub's cross-realm Math is the instrument, and it is measured",
     tax.ratio > 1 && tax.a !== 0 && tax.b !== 0 && tax.opNs > 0,
     tax.sinNs.toFixed(0) + " ns/sin in the vm vs "
     + (1e6 * tax.nat / NSIN).toFixed(0) + " ns native -- " + tax.ratio.toFixed(1)
     + "x; plain arithmetic " + tax.opNs.toFixed(2) + " ns/op in the vm");
  // the interpreter's OWN rate, with no transcendental in the program: 17 stack ops of
  // plain arithmetic, so this is the dispatch cost and nothing else
  {
    const g = { nx: 1024, ny: 1024, nz: 1, Lx: 4 * Math.PI, Ly: 2 * Math.PI, Lz: 0 };
    const p = A2.compile("x*y + 0.3*x*x - 0.2*y*y + x/(y+1)", A2.envOf(g));
    A2.field(p, g);
    const t = process.hrtime.bigint();
    A2.field(p, g);
    const ms = Number(process.hrtime.bigint() - t) / 1e6;
    // 17 stack ops per point, priced at the calibration loop's own arithmetic rate: the
    // gap between them IS the dispatch cost, and it is what a slower dispatch inflates
    const budget = 1024 * 1024 * 17 * tax.opNs / 1e6;
    ok("the RPN interpreter's own dispatch rate", ms < HEADROOM * REF.disp * budget,
       ms.toFixed(0) + " ms for 1.05e6 x 17 ops = " + (1e6 * ms / (1.05e6 * 17)).toFixed(1)
       + " ns/op in the vm, " + (ms / budget).toFixed(2) + "x the calibration loop's "
       + tax.opNs.toFixed(2) + " ns/op (reference " + REF.disp + "x, ceiling "
       + (HEADROOM * REF.disp).toFixed(1) + "x)");
  }
  // 29 stack ops of which 4 are transcendental: the shape a real IC has
  const SRC2 = "sin(2*pi*x/Lx)*cos(2*pi*y/Ly) + 0.3*sin(4*pi*x/Lx)*sin(6*pi*y/Ly)";
  const SRC3 = "sin(2*pi*x/Lx)*cos(2*pi*y/Ly)*sin(2*pi*z/Lz) + 0.3*cos(4*pi*x/Lx)*sin(2*pi*y/Ly)";
  cost(A2, { nx: 1024, ny: 1024, nz: 1, Lx: 4 * Math.PI, Ly: 2 * Math.PI, Lz: 0 },
       SRC2, "tens of ms", tax);
  cost(A3, { nx: 256, ny: 256, nz: 64, Lx: 2 * Math.PI, Ly: 2 * Math.PI, Lz: 16 },
       SRC3, "~200 ms", tax);
  console.log(bad ? "\ncheckexpr: " + bad + " FAILED" : "\ncheckexpr: all checks passed");
  process.exit(bad ? 1 : 0);
}, 0);
