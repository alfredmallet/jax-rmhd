// GATE 4 for the sigma_r display mode (BOTH apps): does the page WIRE the two halves of
// the sigma chain to the u and b vectors, and does that composition really compute
// sigma_r = (|u|^2 - |b|^2)/(|u|^2 + |b|^2)?
//
// Everything the check leans on is READ OUT of the page: the Mode-uniform writes are
// recorded off the stub device, the dispatch sequence off the command encoder, and the
// kernels' own arithmetic (prepDisp's mode -> potential branch chain, its -i*ky / +i*kx
// factors, sigmaCombine's ratio and floor) is parsed out of the EMITTED WGSL and then
// mirrored in fp64 -- never mirrored from memory.
//
// The 3D app runs the SAME sections, twice over: once in the slice view and once in the
// cube-faces view, which is a different TARGET (faceExtract + the face-sized instances of
// the same templates) for the identical mode work. Its extraction kernels are asserted to
// be pure gathers out of the real-space volume, which is what makes the fp64 mirror of
// section 4 -- written on one plane -- the whole story in 3D as well.
"use strict";
const path = require("path");
const dir = process.argv[2] || path.join(__dirname, "..");

let bad = 0;
const ok = (c, m) => { console.log((c ? "  PASS  " : "  FAIL  ") + m); if (!c) bad++; };

// The two tracers, installed once per page: every >=16-byte writeBuffer to a chain-0
// uniform (by BUFFER NAME), and every bind group set inside a compute pass (by NAME, so
// the recorded frame is a readable kernel sequence). Both resolve names by identity
// against the live chain, so nothing here hard-codes the wiring it is checking.
const TRACE = `function(){
  const d = solver.device, origW = d.queue.writeBuffer.bind(d.queue);
  globalThis.UW = [];
  d.queue.writeBuffer = function (b, off, data) {
    const D = solver.disp[0];
    let name = null;
    if (D) for (const k of Object.keys(D.buf)) {
      const v = D.buf[k];
      if (v === b) name = k;
      else if (Array.isArray(v) && v.indexOf(b) >= 0) name = k + "[" + v.indexOf(b) + "]";
    }
    if (name && data.byteLength >= 16) globalThis.UW.push([name, Array.from(new Uint32Array(data.buffer, data.byteOffset, 4))]);
    return origW(b, off, data);
  };
  const origE = d.createCommandEncoder.bind(d);
  globalThis.SEQ = [];
  d.createCommandEncoder = function () {
    const e = origE();
    const bcp = e.beginComputePass.bind(e);
    e.beginComputePass = function () {
      const p = bcp(), sb = p.setBindGroup.bind(p);
      p.setBindGroup = function (i, b) {
        const D = solver.disp[0];
        let nm = "?";
        for (const k of Object.keys(D.bg)) {
          const v = D.bg[k];
          if (nm !== "?") continue;
          if (v === b) nm = k; else if (Array.isArray(v) && v.indexOf(b) >= 0) nm = k + "[" + v.indexOf(b) + "]";
        }
        globalThis.SEQ.push(nm);
        return sb(i, b);
      };
      return p;
    };
    return e;
  };
}`;

// select a field (and, in 3D, a VIEW) on display card 0, then record the uniforms that
// selection writes and the frame it encodes
const PROBE = `function(sel, view, zs){
  const c = cards.disp[0];
  c.selField.value = String(sel);            // the stub FAILS if 9 is not an <option>
  c.selCmap.value = "0";
  if (view && c.selZSrc) c.selZSrc.value = view;
  if (zs !== null && c.rSlice) c.rSlice.value = String(zs);
  globalThis.UW = [];
  c.apply();
  const u = {};
  for (const [n, v] of globalThis.UW) u[n] = v;
  globalThis.SEQ = [];
  c.render();
  return { u: u, seq: globalThis.SEQ.slice() };
}`;

function checkPage(env, page) {
  const is3d = env.is3d;
  const { run, fails } = env;
  console.log("\n=========================== " + page + " ===========================");
  run(TRACE);
  const probe = (sel, view, zs) => run(PROBE, sel, view || null, zs === undefined ? null : zs);

  // the per-VIEW kernel names: a 3D cube runs the face-sized instance of every template
  const views = is3d
    ? [{ v: "manual", tag: "slice view", a: "magSqA", b: "magSqB", msum: "maxSumPartial",
         sig: "sigmaCombine", mpart: "maxPartial", colz: "colorize" },
       { v: "cube", tag: "cube view", a: "magSqFA", b: "magSqFB", msum: "maxSumPartialFace",
         sig: "sigmaCombineFace", mpart: "maxPartialFace", colz: "colorizeCube" }]
    : [{ v: null, tag: "display", a: "magSqA", b: "magSqB", msum: "maxSumPartial",
         sig: "sigmaCombine", mpart: "maxPartial", colz: "colorize" }];

  let mate9 = null;
  for (const V of views) {
    console.log("--- " + V.tag + " ---");
    // -----------------------------------------------------------------------
    // 1. the Mode uniforms the page writes for each selected field
    // -----------------------------------------------------------------------
    const r8 = probe(8, V.v), r9 = probe(9, V.v), r4 = probe(4, V.v);
    const u8 = r8.u, u9 = r9.u, u4 = r4.u;
    mate9 = u9.modeM[0];
    console.log("sigma_c (8): " + JSON.stringify(u8));
    console.log("sigma_r (9): " + JSON.stringify(u9));
    console.log("vector (4):  " + JSON.stringify(u4));
    ok(u8.mode && u8.mode[0] === 8 && u8.modeM && u8.modeM[0] === 7,
       "sigma_c pins the two halves to z+ (mode 8) and z- (mode 7)");
    ok(u9.mode && u9.mode[0] === 9 && u9.modeM && u9.modeM[0] === 5,
       "sigma_r pins the two halves to mode 9 (the u branch) and b (mode 5)");
    ok(u4.mode && u4.mode[0] === 4 && u4.modeM && u4.modeM[0] === 7,
       "a plain vector mode leaves the mate at its z- default");
    ok(u9.mode[2] === 0 && u9.modeM[2] === 0, "both sigma_r halves carry the card's colormap");
    if (is3d) {
      // the mate uniform's OTHER fields must track the card's, so the two halves are the
      // same plane: prepDisp reads only .mode, but nothing may drift (see rmhd3d.html)
      const rz = probe(9, V.v, 3);
      console.log("sigma_r (9) at z slice 3: " + JSON.stringify(rz.u));
      ok(rz.u.mode[1] === 3 && rz.u.modeM[1] === 3,
         "the sigma_r mate uniform rides the card's z slice (both halves on plane 3)");
      probe(9, V.v, 0);
    }

    // -----------------------------------------------------------------------
    // 2. the frame the page ENCODES for mode 9 vs mode 8 (bind groups, in order)
    // -----------------------------------------------------------------------
    const s8 = r8.seq, s9 = r9.seq, s4 = r4.seq;
    console.log("mode 8 frame: " + s8.join(" "));
    console.log("mode 9 frame: " + s9.join(" "));
    ok(s9.join(" ") === s8.join(" "),
       "sigma_r encodes the IDENTICAL kernel sequence as sigma_c (only the uniforms differ)");
    ok(s9[0] === "prepDisp" && s9.indexOf("prepDispM") > 0,
       "sigma_r runs prepDisp twice: the card's Mode uniform first, then the pinned mate");
    ok(s9.indexOf(V.a) > 0 && s9.indexOf(V.b) > s9.indexOf("prepDispM") &&
       s9.indexOf(V.msum) > 0 && s9.indexOf(V.sig) === s9.length - 2,
       "sigma_r squares both halves (" + V.a + "/" + V.b + "), reduces their sum for the " +
       "floor (" + V.msum + "), then combines (" + V.sig + ")");
    ok(s9[s9.length - 1] === V.colz, "... and shades the result with " + V.colz);
    ok(s9.indexOf(V.mpart) < 0, "sigma_r takes no autoscale reduction (fixed +-1)");
    ok(s9.indexOf("vecGather") < 0, "sigma_r gathers no arrows");
    ok(s4.indexOf(V.mpart) >= 0 && s4.indexOf("vecGather") >= 0 && s4.indexOf("prepDispM") < 0,
       "... while a vector mode autoscales, gathers arrows and runs ONE half");
    console.log("mode 4 frame: " + s4.join(" "));
  }

  // -------------------------------------------------------------------------
  // 3. the EMITTED kernels: the branch chain, the derivative factors, the ratio
  // -------------------------------------------------------------------------
  const S = run(is3d
    ? `function(){ const P = solver.p;
         const g = Object.assign({ nx: P.nx, ny: P.ny, nz: P.nz }, makeGrid(P));
         return buildShaders(g); }`
    : `function(){ const g = Object.assign({ nx: solver.p.nx, ny: solver.p.ny,
         Lx: solver.p.Lx, Ly: solver.p.Ly, pm: solver.p.pm, eqsrc: false }, makeGrid(solver.p));
         return buildShaders(g); }`);
  const prep = S.prepDisp, comb = S.sigmaCombine, colz = S.colorize;
  // the if / else-if chain, in source order: [op, n, potential]
  const chain = [];
  for (const m of prep.matchAll(/(?:if|else if) \(md\.mode ([=><]+) (\d+)u\) \{ f = ([^;]+); \}/g))
    chain.push([m[1], parseInt(m[2], 10), m[3].replace(/\s+/g, "")]);
  const potOf = mode => {                       // evaluate the emitted chain for one mode
    for (const [op, n, f] of chain) {
      if (op === "==" && mode === n) return f;
      if (op === ">" && mode > n) return f;
      if (op === "<" && mode < n) return f;
    }
    return "phi";                               // the emitted `var f: vec2<f32> = phi;`
  };
  console.log("emitted prepDisp branch chain: " + JSON.stringify(chain));
  ok(/var f: vec2<f32> = phi;/.test(prep), "the emitted default really is f = phi");
  ok(potOf(9) === "phi", "emitted prepDisp: mode 9 -> phi  (u)      [got " + potOf(9) + "]");
  ok(potOf(5) === "psi", "emitted prepDisp: mode 5 -> psi  (b)      [got " + potOf(5) + "]");
  ok(potOf(4) === potOf(9), "mode 9 selects exactly what the |u| mode 4 selects");
  ok(potOf(8) === "phi+psi" && potOf(7) === "phi-psi", "the sigma_c pair is unchanged");
  ok(/v  = vec2<f32>\( ky \* f\.y, -ky \* f\.x\);/.test(prep) &&
     /v2 = vec2<f32>\(-kx \* f\.y,  kx \* f\.x\);/.test(prep),
     "the emitted components are (-i ky f, +i kx f) = (-d_y f, +d_x f) = zhat x grad f");
  const FLOOR = parseFloat(/const FLOOR: f32 = ([^;]+);/.exec(comb)[1]);
  ok(/v = \(ep - em\) \/ s;/.test(comb) && /let s: f32 = ep \+ em;/.test(comb) && FLOOR > 0,
     "the emitted sigmaCombine is (a - b)/(a + b) with a floor of " + FLOOR);
  const SIGX = /if \(mode == 8u \|\| mode == 9u\) \{ return 0\.5 \* \(clamp\(raw, -1\.0, 1\.0\) \+ 1\.0\); \}/;
  ok(SIGX.test(colz), "the emitted colorize renders modes 8 AND 9 on the fixed +-1 range");
  if (is3d) {
    // the cube shares the slice's shade block, so its faces must carry the SAME predicate
    ok(SIGX.test(S.colorizeCube),
       "the emitted colorizeCube renders modes 8 AND 9 on the fixed +-1 range too");
    // the face-sized instances must be the SLICE text with NFACE for NRS and nothing else:
    // the quiet floor (maxSumPartial's max over this target's own a + b) and the ratio ride
    // into the cube view unchanged, which is why a cube sigma_r needs no new arithmetic
    const bodyOf = s => s.slice(s.indexOf("@group(0)")).replace(/NFACE/g, "NRS");
    ok(bodyOf(S.sigmaCombineFace) === bodyOf(comb),
       "the face-sized sigmaCombine is the slice text with NFACE for NRS (same ratio, same floor)");
    ok(bodyOf(S.maxSumPartialFace) === bodyOf(S.maxSumPartial),
       "the face-sized maxSumPartial is the slice text with NFACE for NRS (the quiet floor " +
       "is the max over the three faces' own a + b)");
    // both extractions are pure GATHERS out of the real-space volume: they move values,
    // they never combine them -- which is what lets section 4 mirror one plane and be done
    const loads = (src, dst) => {
      const out = [];
      for (const m of src.matchAll(new RegExp(dst.replace(/[[\]]/g, "\\$&") + " = ([^;]+);", "g")))
        out.push(m[1].trim());
      return out;
    };
    const sl = loads(S.sliceExtract, "s[i]"), fc = loads(S.faceExtract, "dst[i]");
    const bare = a => a.length > 0 && a.every(e => /^f\[[^=]*\]$/.test(e));
    console.log("sliceExtract loads: " + JSON.stringify(sl));
    console.log("faceExtract loads:  " + JSON.stringify(fc));
    ok(bare(sl) && sl.length === 1 && /md\.zslice/.test(sl[0]),
       "sliceExtract is one bare load from the card's plane -- no arithmetic on the field");
    ok(bare(fc) && fc.length === 3,
       "faceExtract is three bare loads (top + two side faces) -- no arithmetic either");
  }

  // -------------------------------------------------------------------------
  // 4. fp64 mirror: the chain the page just encoded, on an analytic (phi, psi)
  // -------------------------------------------------------------------------
  // Mirror of the emitted kernels (each formula asserted above, in the emitted text):
  //   half h  : f = potOf(mode_h);  components = (-d_y f, +d_x f)      [prepDisp + iFFT]
  //   magSq   : |.|^2                                                   [vecMagSq]
  //   combine : (A - B)/(A + B), zeroed below FLOOR * max(A + B)        [sigmaCombine]
  // with the two modes the PAGE wrote into the uniforms above (9 and its recorded mate).
  // In 3D the transform is a volume one and a plane (or a face) is then gathered out of
  // it, so at a FIXED z this is the same mirror -- see the extraction assertions above.
  const NX = 32, NY = 32, L = 2 * Math.PI;
  const dPHIdx = (x, y) => 0.7 * Math.cos(x) * Math.cos(2 * y);
  const dPHIdy = (x, y) => -1.4 * Math.sin(x) * Math.sin(2 * y);
  const dPSIdx = (x, y) => -1.2 * Math.sin(3 * x) * Math.sin(y) + 0.2 * Math.cos(x + y);
  const dPSIdy = (x, y) => 0.4 * Math.cos(3 * x) * Math.cos(y) + 0.2 * Math.cos(x + y);
  function comps(pot, x, y) {                    // (-d_y f, +d_x f) for f = phi | psi | phi+-psi
    const cphi = (pot === "psi") ? 0 : 1;
    const cpsi = pot === "psi" ? 1 : (pot === "phi+psi" ? 1 : (pot === "phi-psi" ? -1 : 0));
    const gx = cphi * dPHIdx(x, y) + cpsi * dPSIdx(x, y);
    const gy = cphi * dPHIdy(x, y) + cpsi * dPSIdy(x, y);
    return [-gy, gx];
  }
  const modeH0 = 9, modeH1 = mate9;
  const A = [], B = [];
  for (let i = 0; i < NX; i++) for (let j = 0; j < NY; j++) {
    const x = i * L / NX, y = j * L / NY;
    const c0 = comps(potOf(modeH0), x, y), c1 = comps(potOf(modeH1), x, y);
    A.push(c0[0] * c0[0] + c0[1] * c0[1]);
    B.push(c1[0] * c1[0] + c1[1] * c1[1]);
  }
  let mx = 0;
  for (let i = 0; i < A.length; i++) mx = Math.max(mx, A[i] + B[i]);
  let worst = 0, floored = 0, rng = 0;
  for (let i = 0; i < A.length; i++) {
    const x = Math.floor(i / NY) * L / NX, y = (i % NY) * L / NY;
    const ux = -dPHIdy(x, y), uy = dPHIdx(x, y), bx = -dPSIdy(x, y), by = dPSIdx(x, y);
    const eu = ux * ux + uy * uy, eb = bx * bx + by * by;      // the ANALYTIC |u|^2, |b|^2
    const want = (eu + eb) > 0 ? (eu - eb) / (eu + eb) : 0;
    const s = A[i] + B[i];
    if (!(s > 0 && s >= FLOOR * mx)) { floored++; continue; }
    const got = (A[i] - B[i]) / s;
    rng = Math.max(rng, Math.abs(got));
    worst = Math.max(worst, Math.abs(got - want));
  }
  ok(worst < 1e-12, "the mirrored mode-9 chain reproduces analytic sigma_r to " + worst.toExponential(3) +
                    " over " + (NX * NY - floored) + " points");
  ok(rng <= 1, "the mirrored sigma_r stays inside [-1, 1] (max |sigma_r| = " + rng.toFixed(6) + ")");
  // and the same mirror with the sigma_c pair must NOT equal sigma_r (the two modes differ)
  let dmax = 0;
  for (let i = 0; i < NX * NY; i++) {
    const x = Math.floor(i / NY) * L / NX, y = (i % NY) * L / NY;
    const p = comps("phi+psi", x, y), m = comps("phi-psi", x, y);
    const ep = p[0] * p[0] + p[1] * p[1], em = m[0] * m[0] + m[1] * m[1];
    if (!(ep + em > 0) || !(A[i] + B[i] > 0)) continue;
    dmax = Math.max(dmax, Math.abs((ep - em) / (ep + em) - (A[i] - B[i]) / (A[i] + B[i])));
  }
  ok(dmax > 0.1, "sigma_c and sigma_r are genuinely different fields on this state (max |diff| = " +
                 dmax.toFixed(3) + ")");

  if (fails.length) { console.log("stub failures (" + page + "): " + fails.join(" | ")); bad += fails.length; }
}

// each page is booted and then given a macrotask to finish its async bring-up (the same
// wait bootstub.js uses) before it is interrogated
const boot = page => new Promise(res => {
  const env = require("./stubenv")(dir, page, "");
  setTimeout(() => res(env), 50);
});
(async function () {
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) checkPage(await boot(page), page);
  console.log(bad ? "\nsigma_r wiring check: " + bad + " FAILURE(S)" : "\nall sigma_r wiring checks passed");
  process.exit(bad ? 1 : 0);
})();
