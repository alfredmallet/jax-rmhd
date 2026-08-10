// GATE 4 for the sigma_r display mode (2D app): does the page WIRE the two halves of the
// sigma chain to the u and b vectors, and does that composition really compute
// sigma_r = (|u|^2 - |b|^2)/(|u|^2 + |b|^2)?
//
// Everything the check leans on is READ OUT of the page: the Mode-uniform writes are
// recorded off the stub device, the dispatch sequence off the command encoder, and the
// kernels' own arithmetic (prepDisp's mode -> potential branch chain, its -i*ky / +i*kx
// factors, sigmaCombine's ratio and floor) is parsed out of the EMITTED WGSL and then
// mirrored in fp64 -- never mirrored from memory.
"use strict";
const path = require("path");
const dir = process.argv[2] || path.join(__dirname, "..");
const env = require("./stubenv")(dir, "rmhd2d.html", "");
const { run, fails } = env;
setTimeout(function () {
let bad = 0;
const ok = (c, m) => { console.log((c ? "  PASS  " : "  FAIL  ") + m); if (!c) bad++; };

// ---------------------------------------------------------------------------
// 1. the Mode uniforms the page writes for each selected field
// ---------------------------------------------------------------------------
run(`function(){
  globalThis.UW = [];                       // [bufferName, [u32 x 4]]
  const d = solver.device, orig = d.queue.writeBuffer.bind(d.queue);
  d.queue.writeBuffer = function (b, off, data) {
    const D = solver.disp[0];
    let name = null;
    if (D) for (const k of Object.keys(D.buf)) {
      const v = D.buf[k];
      if (v === b) name = k;
      else if (Array.isArray(v) && v.indexOf(b) >= 0) name = k + "[" + v.indexOf(b) + "]";
    }
    if (name && data.byteLength >= 16) globalThis.UW.push([name, Array.from(new Uint32Array(data.buffer, data.byteOffset, 4))]);
    return orig(b, off, data);
  };
}`);

const uniformsFor = sel => run(`function(sel){
  globalThis.UW = [];
  const c = cards.disp[0];
  c.selField.value = String(sel); c.selCmap.value = "0"; c.apply();
  const out = {};
  for (const [n, v] of globalThis.UW) out[n] = v;
  return out;
}`, sel);

const u8 = uniformsFor(8), u9 = uniformsFor(9), u4 = uniformsFor(4);
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

// ---------------------------------------------------------------------------
// 2. the frame the page ENCODES for mode 9 vs mode 8 (pipeline + bind group, in order)
// ---------------------------------------------------------------------------
run(`function(){
  const d = solver.device, orig = d.createCommandEncoder.bind(d);
  globalThis.SEQ = [];
  d.createCommandEncoder = function () {
    const e = orig();
    const bcp = e.beginComputePass.bind(e);
    e.beginComputePass = function () {
      const p = bcp(), sp = p.setPipeline.bind(p), sb = p.setBindGroup.bind(p);
      let cur = null;
      p.setPipeline = function (pl) { cur = pl && pl.__name; return sp(pl); };
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
}`);
const seqFor = sel => run(`function(sel){
  const c = cards.disp[0];
  c.selField.value = String(sel); c.apply();
  globalThis.SEQ = [];
  c.render();
  return globalThis.SEQ.slice();
}`, sel);
const s8 = seqFor(8), s9 = seqFor(9), s4 = seqFor(4);
console.log("mode 8 frame: " + s8.join(" "));
console.log("mode 9 frame: " + s9.join(" "));
ok(s9.join(" ") === s8.join(" "),
   "sigma_r encodes the IDENTICAL kernel sequence as sigma_c (only the uniforms differ)");
ok(s9[0] === "prepDisp" && s9.indexOf("prepDispM") > 0,
   "sigma_r runs prepDisp twice: the card's Mode uniform first, then the pinned mate");
ok(s9.indexOf("magSqA") > 0 && s9.indexOf("magSqB") > s9.indexOf("prepDispM") &&
   s9.indexOf("maxSumPartial") > 0 && s9.indexOf("sigmaCombine") === s9.length - 2,
   "sigma_r squares both halves, reduces their sum for the floor, then combines");
ok(s9.indexOf("maxPartial") < 0, "sigma_r takes no autoscale reduction (fixed +-1)");
ok(s9.indexOf("vecGather") < 0, "sigma_r gathers no arrows");
ok(s4.indexOf("maxPartial") >= 0 && s4.indexOf("vecGather") >= 0 && s4.indexOf("prepDispM") < 0,
   "... while a vector mode autoscales, gathers arrows and runs ONE half");
console.log("mode 4 frame: " + s4.join(" "));

// ---------------------------------------------------------------------------
// 3. the EMITTED kernels: the branch chain, the derivative factors, the ratio
// ---------------------------------------------------------------------------
const S = run(`function(){ const g = Object.assign({ nx: solver.p.nx, ny: solver.p.ny,
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
ok(/if \(mode == 8u \|\| mode == 9u\) \{ return 0\.5 \* \(clamp\(raw, -1\.0, 1\.0\) \+ 1\.0\); \}/.test(colz),
   "the emitted colorize renders modes 8 AND 9 on the fixed +-1 range");

// ---------------------------------------------------------------------------
// 4. fp64 mirror: the chain the page just encoded, on an analytic (phi, psi)
// ---------------------------------------------------------------------------
// Mirror of the emitted kernels (each formula asserted above, in the emitted text):
//   half h  : f = potOf(mode_h);  components = (-d_y f, +d_x f)      [prepDisp + iFFT]
//   magSq   : |.|^2                                                   [vecMagSq]
//   combine : (A - B)/(A + B), zeroed below FLOOR * max(A + B)        [sigmaCombine]
// with the two modes the PAGE wrote into the uniforms above (9 and u9.modeM[0]).
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
const modeH0 = 9, modeH1 = u9.modeM[0];
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

if (fails.length) { console.log("stub failures: " + fails.join(" | ")); bad += fails.length; }
console.log(bad ? "\nsigma_r wiring check: " + bad + " FAILURE(S)" : "\nall sigma_r wiring checks passed");
process.exit(bad ? 1 : 0);
}, 0);
