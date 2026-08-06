// fp64 checks of the colormap table: (a) the WGSL the apps emit carries EXACTLY the
// coefficients cmapRGB uses (single source of truth), (b) afmhot stays the exact
// matplotlib closed form, (c) samples for the python fit-accuracy check.
const fs = require("fs"), vm = require("vm"), path = require("path");
const DIR = process.argv[2] || require("path").join(__dirname, "..");
const cx = { console, Math, JSON, Float32Array, Uint32Array, Map, Error, Promise, setTimeout,
  Number, String, Array, Object, isFinite, parseInt, parseFloat,
  document: { getElementById: () => ({ style: {}, addEventListener() {} }), addEventListener() {} },
  window: { addEventListener() {}, devicePixelRatio: 1 }, navigator: {},
  requestAnimationFrame: () => {}, performance: { now: () => 0 } };
cx.globalThis = cx;
vm.createContext(cx);
for (const f of ["common.js", "physics.js"]) vm.runInContext(fs.readFileSync(path.join(DIR, f), "utf8"), cx, { filename: f });
const CMAP_COEF = vm.runInContext("CMAP_COEF", cx), CMAP_NAMES = vm.runInContext("CMAP_NAMES", cx),
      cmapRGB = vm.runInContext("cmapRGB", cx), CMAP_WGSL = vm.runInContext("CMAP_WGSL", cx);

let bad = 0;
const chk = (n, ok, note) => { if (!ok) bad++; console.log((ok ? "PASS  " : "FAIL  ") + n + (note ? "   " + note : "")); };

// (a) the emitted WGSL vec3 literals == the JS table, in order
const vecs = [...CMAP_WGSL.matchAll(/vec3<f32>\(([-\d.e]+), ([-\d.e]+), ([-\d.e]+)\)/g)]
  .map(m => [+m[1], +m[2], +m[3]]);
const want = CMAP_COEF.viridis.concat(CMAP_COEF.rdbu);
const got = vecs.slice(0, want.length);
let emax = 0;
for (let i = 0; i < want.length; i++) for (let c = 0; c < 3; c++) emax = Math.max(emax, Math.abs(want[i][c] - got[i][c]));
chk("WGSL coefficients == CMAP_COEF", got.length === want.length && emax === 0,
    got.length + " vec3 literals, max |delta| = " + emax);

// (b) afmhot is still the exact formula
let am = 0;
for (let i = 0; i <= 1000; i++) {
  const t = i / 1000, r = cmapRGB(0, t);
  const ex = [Math.min(1, 2 * t), Math.max(0, Math.min(1, 2 * t - 0.5)), Math.max(0, 2 * t - 1)];
  for (let c = 0; c < 3; c++) am = Math.max(am, Math.abs(r[c] - ex[c]));
}
chk("afmhot exact (1001 samples)", am === 0, "max |delta| = " + am);
chk("grayscale identity", cmapRGB(3, 0.37)[0] === 0.37 && cmapRGB(3, 0.37)[2] === 0.37);
// (c) range + endpoint clamping
let out = 0;
for (let w = 0; w < CMAP_NAMES.length; w++) for (let i = -20; i <= 120; i++) {
  const r = cmapRGB(w, i / 100);
  for (const v of r) if (!(v >= 0 && v <= 1) || !isFinite(v)) out++;
}
chk("all colormaps stay in [0,1] (incl. out-of-range x)", out === 0, out + " violations");
// samples for the python side
const S = {};
for (let w = 0; w < CMAP_NAMES.length; w++) {
  S[CMAP_NAMES[w]] = [];
  for (let i = 0; i < 256; i++) S[CMAP_NAMES[w]].push(cmapRGB(w, i / 255));
}
fs.writeFileSync("/tmp/pF/cmap_samples.json", JSON.stringify(S));
console.log(bad ? "cmap check: FAILED" : "cmap check: all green");
process.exit(bad ? 1 : 0);
