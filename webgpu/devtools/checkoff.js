// The 2D DISPLAY OFFSET gate: the per-card x/y sliders that roll the picture inside its
// frame (README: "The display offset (2D)").
// Usage: node checkoff.js [webgpu-dir]           exit code 1 on any failure
//
// Four claims, in the order they matter:
//   1. the emission is GATED (the sigR / band pattern): a constants object without `shift`
//      emits the pre-offset text, and the whole diff `shift` makes is the struct's
//      bpad -> sh rename plus one contiguous block of phase lines. Across the WHOLE page
//      exactly one kernel mentions `md.sh` -- so the cut chart, the spectra and every
//      stepping kernel are untouched by construction, which is what keeps the cut on the
//      REAL x = Lx/2.
//   2. the uniform layout: `shift` writes words 6-7 and moves nothing else, so a Mode
//      uniform is still MODE_BYTES with the same band / mode / cmap words in it.
//   3. the arithmetic, by RUNNING the emitted prepDisp (wgsl_reflect's interpreter, as
//      checkiso runs the raymarch): the factor is exactly e^{-i k.S}; a whole-cell offset
//      really is a ROLL of the real-space picture (a CPU inverse transform in the app's
//      own convention, +x = +ix = rightward on the canvas, +y = +iy = downward); a whole
//      box is the identity; and OFF IS BITWISE OFF -- at [0, 0] the output is bit-identical
//      to the un-gated kernel's.
//   4. the page: the sliders exist in 2D and NOT in 3D, they are live with no panel gate,
//      the words the page actually writes are the card's fractions times the box on every
//      mode buffer of the chain (the sigma mate and both contour potentials included), a
//      centred pair writes exact zeros and captions nothing, and two cards are independent.
"use strict";
const fs = require("fs"), path = require("path");
const { pathToFileURL } = require("url");
const dir = path.resolve(process.argv[2] || path.join(__dirname, ".."));
let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};
const boot = (d, page) => new Promise(res => {
  const env = require("./stubenv")(d, page);
  setTimeout(() => res(env));
});
async function wgslMod() {
  const p = path.join(__dirname, "node_modules", "wgsl_reflect", "wgsl_reflect.module.js");
  if (!fs.existsSync(p)) return null;
  return await import(pathToFileURL(p).href);
}
// the lines `b` adds to `a`, or null if b is not a pure insertion into a (checkiso's helper)
function inserted(a, b) {
  const A = a.split("\n"), add = [];
  let i = 0;
  for (const l of b.split("\n")) { if (i < A.length && A[i] === l) i++; else add.push(l); }
  return i === A.length ? add : null;
}

// A small synthetic grid in the app's own k convention (makeGrid: kx = fftfreq order, ky
// half-line), so the interpreter runs a kernel whose every mode this file can also compute
// by hand -- and so the inverse transform below is 8x8 and not 128x128.
const NX = 8, NY = 8, NKY = NY / 2 + 1, NM = NX * NKY;
const LX = 2 * Math.PI, LY = 4 * Math.PI;
const DX = LX / NX, DY = LY / NY;
function synthGrid() {
  const g = new Float32Array(4 * NM);
  for (let i = 0; i < NX; i++) {
    const ix = i < NX / 2 ? i : i - NX, kx = ix * 2 * Math.PI / LX;
    for (let j = 0; j < NKY; j++) {
      const ky = j * 2 * Math.PI / LY, m = i * NKY + j;
      g[4 * m] = kx; g[4 * m + 1] = ky; g[4 * m + 2] = kx * kx + ky * ky;
      g[4 * m + 3] = g[4 * m + 2] > 0 ? 1 / g[4 * m + 2] : 0;
    }
  }
  return g;
}
// a deterministic (phi, psi) that is a valid rfft2 of a REAL field: the two self-conjugate
// points of each half-line row are real, and the Nyquist row / column are zero -- which is
// what the 2/3 dealias holds them at in the app, and what makes an arbitrary offset a real
// picture at all (README's reality caveat).
function synthFields() {
  const f = new Float32Array(4 * NM);
  let s = 12345;
  const rnd = () => { s = (s * 48271) % 2147483647; return s / 2147483647 - 0.5; };
  for (let fi = 0; fi < 2; fi++) for (let i = 0; i < NX; i++) for (let j = 0; j < NKY; j++) {
    const m = fi * NM + i * NKY + j;
    if (i === NX / 2 || j === NKY - 1) continue;            // the Nyquist column / row: zero
    f[2 * m] = rnd(); f[2 * m + 1] = rnd();
  }
  for (let fi = 0; fi < 2; fi++) {                          // F(-kx, 0) = conj(F(kx, 0))
    const b = fi * NM;
    f[2 * (b + 0) + 1] = 0;                                 // and F(0,0) is real
    for (let i = 1; i < NX / 2; i++) {
      const m = b + i * NKY, mm = b + (NX - i) * NKY;
      f[2 * mm] = f[2 * m]; f[2 * mm + 1] = -f[2 * m + 1];
    }
  }
  return f;
}
// the app's inverse transform, by hand and in fp64: f(x,y) = sum_m w_j Re[F_m e^{i k.x}]
// (unnormalized, ky > 0 rows counted twice through their conjugate mirror -- grids.fft's
// convention, and the sign the "-i*ky*f = -d_y f" comment in prepDisp fixes)
function irfft2(F) {
  const r = new Float64Array(NX * NY);
  for (let ix = 0; ix < NX; ix++) for (let iy = 0; iy < NY; iy++) {
    const x = ix * DX, y = iy * DY;
    let acc = 0;
    for (let i = 0; i < NX; i++) {
      const kxi = (i < NX / 2 ? i : i - NX) * 2 * Math.PI / LX;
      for (let j = 0; j < NKY; j++) {
        const m = i * NKY + j, w = (j === 0 || j === NKY - 1) ? 1 : 2;
        const ph = kxi * x + j * 2 * Math.PI / LY * y;
        acc += w * (F[2 * m] * Math.cos(ph) - F[2 * m + 1] * Math.sin(ph));
      }
    }
    r[ix * NY + iy] = acc;
  }
  return r;
}
const rollIdx = (a, mx, my) => {           // a[(ix - mx, iy - my)], the picture moved by +m
  const r = new Float64Array(a.length);
  for (let ix = 0; ix < NX; ix++) for (let iy = 0; iy < NY; iy++) {
    const sx = ((ix - mx) % NX + NX) % NX, sy = ((iy - my) % NY + NY) % NY;
    r[ix * NY + iy] = a[sx * NY + sy];
  }
  return r;
};
const bits = a => new Uint32Array(a.buffer, a.byteOffset, a.length);
const amax = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };

// ---------------------------------------------------------------------------
// 1-3 run in the 2D page's own context: prepDispWGSL and modeWords are the app's
// ---------------------------------------------------------------------------
const EMIT = `function(pre, kunit){
  const C = { pre: pre, hasZ: false, wgReal: 64, nDisp: "NR", ns: 4, sigR: true, band: kunit };
  return { off: prepDispWGSL(C),
           on:  prepDispWGSL(Object.assign({}, C, { shift: true })),
           bytes: MODE_BYTES }; }`;
const WORDS = `function(mode, band, shift){
  return Array.from(new Uint8Array(modeWords(mode, 0, 2, band, shift).buffer)); }`;
// every kernel the page emits at its self-test grid, for the "one kernel only" scan
const KERNELS = `function(){
  const R = REFVEC;
  const P = { nx: R.nx, ny: R.ny, Lx: R.Lx, Ly: R.Ly, diss: R.diss, hyper: R.hyper, fshell: R.fshell };
  const g = Object.assign({ nx: P.nx, ny: P.ny, Lx: P.Lx, Ly: P.Ly }, makeGrid(P));
  return buildShaders(g); }`;

async function legEmission(env) {
  const pre = `const NX: u32 = ${NX}u;\nconst NY: u32 = ${NY}u;\nconst NM: u32 = ${NM}u;\n`;
  const E = env.run(EMIT, pre, 2 * Math.PI / LX);
  const add = inserted(E.off.replace("bpad: vec2<f32>", "sh: vec2<f32>"), E.on);
  ok("2D: `shift` is a pure INSERTION into the band text (plus the bpad -> sh rename)",
     !!add, add ? add.length + " lines added" : "the pre-offset lines moved");
  if (add) {
    const body = add.filter(l => l.trim().length);
    ok("  ... and the insertion is only the phase block", body.length === 8
       && /^\s*\/\/ the display offset/.test(body[0])
       && /if \(md\.sh\.x != 0\.0 \|\| md\.sh\.y != 0\.0\)/.test(body[1]),
       body.length + " non-blank lines");
  }
  ok("  ... a C without `shift` mentions neither the phase nor the word",
     !/md\.sh/.test(E.off) && !/cos\(ph\)/.test(E.off) && /bpad: vec2<f32>/.test(E.off));
  ok("  ... and with it the struct names the two padding words `sh`",
     /struct Mode \{[^}]*klo: f32, khi: f32, sh: vec2<f32> \}/.test(E.on) && !/bpad/.test(E.on));

  // the whole page: exactly ONE kernel may know about the offset
  const K = env.run(KERNELS);
  const named = Object.keys(K).filter(k => /md\.sh|sh: vec2<f32>/.test(K[k]));
  ok("2D: exactly one emitted kernel mentions the offset, and it is prepDisp",
     named.length === 1 && named[0] === "prepDisp", named.join(", ") || "none");
  ok("  ... so cutPrep (the cut chart's own line at the REAL x = Lx/2) has none",
     !!K.cutPrep && !/md\.sh/.test(K.cutPrep));
}

function legWords(env) {
  const w0 = env.run(WORDS, 3, [4, 0], null);
  const w1 = env.run(WORDS, 3, [4, 0], [1.5, -2.25]);
  ok("modeWords: the shift is 32 bytes wide and its first 24 are unmoved",
     w0.length === 32 && w1.length === 32 &&
     w0.slice(0, 24).join() === w1.slice(0, 24).join());
  const f = a => new Float32Array(Uint8Array.from(a).buffer);
  ok("  ... words 6-7 ARE the shift, and no shift means exact zeros",
     f(w1)[6] === 1.5 && f(w1)[7] === -2.25 && f(w0)[6] === 0 && f(w0)[7] === 0,
     "[" + f(w1)[6] + ", " + f(w1)[7] + "]");
  ok("  ... and the band words are where they were", f(w1)[4] === 4 && f(w1)[5] === 0);
}

// prepDisp, EXECUTED (checkiso's runPrep, with the shift words filled in by modeWords)
function runPrep(M, src, fields, gridA, u8) {
  const outk = new Float32Array(2 * NM), outk2 = new Float32Array(2 * NM);
  new M.WgslExec(new M.WgslParser().parse(src)).dispatchWorkgroups("main", [NM, 1, 1],
    { 0: { 0: fields, 1: gridA, 2: outk, 3: new Uint32Array(Uint8Array.from(u8).buffer), 4: outk2 } });
  return [outk, outk2];
}
async function legArithmetic(env) {
  const M = await wgslMod();
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  const pre = `const NX: u32 = ${NX}u;\nconst NY: u32 = ${NY}u;\nconst NM: u32 = ${NM}u;\n`;
  const E = env.run(EMIT, pre, 2 * Math.PI / LX);
  const gA = synthGrid(), F = synthFields();
  const U = (sh) => env.run(WORDS, 2, [0, 0], sh);     // mode 2 = plain phi, filter OFF
  const base = runPrep(M, E.off, F, gA, U(null))[0];   // the un-gated kernel, for comparison

  // ---- off is bitwise off -------------------------------------------------
  const zero = runPrep(M, E.on, F, gA, U([0, 0]))[0];
  ok("off IS bitwise off: [0,0] matches the un-gated kernel bit for bit",
     bits(zero).every((v, i) => v === bits(base)[i]));

  // ---- the factor is e^{-i k.S} ------------------------------------------
  const S = [3 * DX, -2 * DY];
  const got = runPrep(M, E.on, F, gA, U(S))[0];
  let em = 0, mm = 0;
  for (let m = 0; m < NM; m++) {
    const ph = gA[4 * m] * S[0] + gA[4 * m + 1] * S[1], c = Math.cos(ph), s = Math.sin(ph);
    const a = base[2 * m], b = base[2 * m + 1];
    em = Math.max(em, Math.abs(got[2 * m] - (a * c + b * s)),
                      Math.abs(got[2 * m + 1] - (b * c - a * s)));
    mm = Math.max(mm, Math.abs(Math.hypot(got[2 * m], got[2 * m + 1]) - Math.hypot(a, b)));
  }
  ok("the emitted factor IS the fp64 phase e^{-i k.S}", em < 1e-6, "max |delta| = " + em.toExponential(2));
  ok("  ... and it is a PHASE: every mode keeps its modulus", mm < 1e-6, "max |delta| = " + mm.toExponential(2));

  // ---- ... which is a ROLL of the picture, in the direction claimed -------
  // +Sx moves the picture towards +ix (rightward on the canvas), +Sy towards +iy (down:
  // grid row y = 0 is drawn at the top). This is the claim the tooltip makes.
  const er = amax(irfft2(got), rollIdx(irfft2(base), 3, -2));
  ok("a whole-cell offset is an exact ROLL of the real-space picture (+x right, +y down)",
     er < 1e-4, "max |delta| = " + er.toExponential(2));

  // ---- a whole box is the identity ---------------------------------------
  const boxed = runPrep(M, E.on, F, gA, U([LX, LY]))[0];
  ok("a whole-box offset is the identity", amax(boxed, base) < 1e-4,
     "max |delta| = " + amax(boxed, base).toExponential(2));

  // ---- the vector branch is carried too (mode 4 = |u|, both outputs) -----
  const uw = env.run(WORDS, 4, [0, 0], S), zw = env.run(WORDS, 4, [0, 0], null);
  const v4 = runPrep(M, E.on, F, gA, uw), b4 = runPrep(M, E.off, F, gA, zw);
  let e4 = 0;
  for (const k of [0, 1]) for (let m = 0; m < NM; m++) {
    const ph = gA[4 * m] * S[0] + gA[4 * m + 1] * S[1], c = Math.cos(ph), s = Math.sin(ph);
    const a = b4[k][2 * m], b = b4[k][2 * m + 1];
    e4 = Math.max(e4, Math.abs(v4[k][2 * m] - (a * c + b * s)),
                      Math.abs(v4[k][2 * m + 1] - (b * c - a * s)));
  }
  ok("the second output (the vector modes' u_y / u_x half) takes the same phase",
     e4 < 1e-6, "max |delta| = " + e4.toExponential(2));
}

// ---------------------------------------------------------------------------
// 4. the page: the controls, and the words the page really writes
// ---------------------------------------------------------------------------
// every Mode-uniform write of every display chain, by chain and buffer (checkiso's tracer)
const UTRACE = `function(){
  const d = solver.device, orig = d.queue.writeBuffer.bind(d.queue);
  globalThis.UW = [];
  d.queue.writeBuffer = function (b, off, data) {
    let name = null;
    for (let ci = 0; ci < solver.disp.length; ci++) {
      const D = solver.disp[ci];
      if (!D || name) continue;
      for (const k of Object.keys(D.buf)) {
        const v = D.buf[k];
        if (v === b) name = ci + ":" + k;
        else if (Array.isArray(v) && v.indexOf(b) >= 0) name = ci + ":" + k + "[" + v.indexOf(b) + "]";
      }
    }
    if (name && name.indexOf(":mode") > 0) {
      globalThis.UW.push([name, Array.from(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))]);
    }
    return orig(b, off, data);
  };
}`;
const CARDS = `function(fx, fy){
  while (cards.disp.length < 2) addDisplayCard();
  cardsSync();
  const A = cards.disp[0], B = cards.disp[1];
  A.rOffX.value = String(fx); A.rOffY.value = String(fy);
  globalThis.UW = [];
  A.apply(); B.apply();
  const w = {};
  for (const u of globalThis.UW) if (!w[u[0]]) w[u[0]] = u[1];
  return { a: A.offset(), b: B.offset(), on: A.offsetOn(), cap: A.cap.innerHTML,
           capB: B.cap.innerHTML, w: w, Lx: solver.p.Lx, Ly: solver.p.Ly,
           shown: [A.rOffX.style.display, A.rOffX.lab.style.display],
           lab: [A.rOffX.labSpan.innerHTML, A.rOffY.labSpan.innerHTML],
           range: [A.rOffX.min, A.rOffX.max, A.rOffX.step] }; }`;
const CLAMP = `function(){ const A = cards.disp[0];
  A.rOffX.value = "9"; A.rOffY.value = "-9";       // past both stops, set by hand
  return A.offset(); }`;

async function legPage2d(env) {
  env.run(UTRACE);
  const R = env.run(CARDS, 0, 0);
  ok("2D: the two sliders exist, are visible, and are centred at 0",
     R.a[0] === 0 && R.a[1] === 0 && R.shown[0] === "" && R.shown[1] === "" && !R.on,
     "shown = [" + R.shown + "]");
  ok("  ... labelled, and travelling half a box each way", R.lab[0] === "offset x"
     && R.lab[1] === "offset y" && R.range[0] === "-0.5" && R.range[1] === "0.5"
     && R.range[2] === "0.01", R.lab + " over [" + R.range.slice(0, 2) + "] by " + R.range[2]);
  const f = a => new Float32Array(Uint8Array.from(a).buffer);
  const names = Object.keys(R.w).filter(k => k.indexOf("0:") === 0);
  ok("  ... a centred card writes EXACT zeros on every mode buffer of its chain",
     names.length >= 4 && names.every(k => f(R.w[k])[6] === 0 && f(R.w[k])[7] === 0),
     names.join(", "));
  ok("  ... and captions nothing", R.cap.indexOf("offset") < 0);

  const O = env.run(CARDS, 0.25, -0.1);
  const g = f;
  const mine = Object.keys(O.w).filter(k => k.indexOf("0:") === 0);
  const want = [0.25 * O.Lx, -0.1 * O.Ly];
  const rel = (a, b) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));
  ok("2D: the offset the page writes is the card's fractions times the BOX",
     mine.length >= 4 && mine.every(k => rel(g(O.w[k])[6], want[0]) && rel(g(O.w[k])[7], want[1])),
     mine.length + " mode buffers: [" + g(O.w[mine[0]])[6] + ", " + g(O.w[mine[0]])[7]
     + "] want [" + want + "]");
  const other = Object.keys(O.w).filter(k => k.indexOf("1:") === 0);
  ok("  ... and the OTHER card is untouched (one card, one offset)",
     other.length >= 4 && other.every(k => g(O.w[k])[6] === 0 && g(O.w[k])[7] === 0)
     && O.b[0] === 0 && O.b[1] === 0 && O.capB.indexOf("offset") < 0);
  ok("  ... the caption says so while it is active",
     /offset/.test(O.cap) && /0\.25/.test(O.cap) && /0\.10/.test(O.cap),
     O.cap.replace(/<[^>]*>/g, ""));
  const cl = env.run(CLAMP);
  ok("  ... and a value past either stop is clamped, not written raw",
     cl[0] === 0.5 && cl[1] === -0.5, "[" + cl + "]");
  ok("2D: stub boot raised no failures", env.fails.length === 0, env.fails.join(" | "));
}

async function legPage3d(env) {
  env.run(UTRACE);
  const R = env.run(`function(){
    const A = cards.disp[0];
    globalThis.UW = [];
    A.apply();
    const w = {};
    for (const u of globalThis.UW) if (!w[u[0]]) w[u[0]] = u[1];
    return { has: !!A.rOffX, off: A.offset(), on: A.offsetOn(), cap: A.cap.innerHTML, w: w }; }`);
  ok("3D: no offset sliders at all (a card's picture may be the whole box)",
     !R.has && R.off[0] === 0 && R.off[1] === 0 && !R.on && R.cap.indexOf("offset") < 0);
  const K = env.run(`function(){
    const R = REFVEC;
    const g = Object.assign({ nx: R.nx, ny: R.ny, nz: R.nz },
      makeGrid({ nx: R.nx, ny: R.ny, nz: R.nz, Lx: R.Lx, Ly: R.Ly, Lz: R.Lz,
                 diss: R.diss, hyper: R.hyper, zdiss: R.z_diss_k, fshell: R.fshell }));
    return buildShaders(g); }`);
  const named = Object.keys(K).filter(k => /md\.sh|sh: vec2<f32>/.test(K[k]));
  ok("  ... and no 3D kernel mentions the offset (its prepDisp keeps `bpad`)",
     named.length === 0 && /bpad: vec2<f32>/.test(K.prepDisp), named.join(", "));
  ok("3D: stub boot raised no failures", env.fails.length === 0, env.fails.join(" | "));
}

(async () => {
  console.log("1-2. emission gating + uniform layout (2D)");
  const e2 = await boot(dir, "rmhd2d.html");
  await legEmission(e2);
  legWords(e2);
  console.log("3. the arithmetic, through the emitted kernel");
  await legArithmetic(e2);
  console.log("4. the page's own controls and writes");
  await legPage2d(e2);
  const e3 = await boot(dir, "rmhd3d.html");
  await legPage3d(e3);
  console.log(bad ? "display offset: FAILED (" + bad + ")" : "display offset: all green");
  process.exit(bad ? 1 : 0);
})();