// ANISO_PLAN_2 gates: the generated E(k_perp, k_par) card -- the band-gated prepGrads
// sweep (A), the coordinate binning kernel (B) and the card itself (C).
//
//   node check2dspec.js [webgpu-dir]        exit code 1 on any failure
//
// CI reports, never gates (the plan's own line, and the sibling scripts' contract): every
// row prints PASS / FAIL and the exit code is for a human reading the log.
//
// Structure, as checkiso.js: LEGS below is a list of [title, fn] run in order over one
// shared `state`. The base-commit legs shell out to git, so they run from a work tree at
// any cwd.
//
// What is the app's own code here, and what is not
// ------------------------------------------------
//   * the EMITTED kernels come out of the real `dumpwgsl2` and are diffed, kernel by
//     kernel, against the same emission from `git show <BASE>:...`;
//   * `prepGradsBand` is EXECUTED, on wgsl_reflect's WGSL interpreter (checkiso's legs 6
//     and 9 idiom), against an fp64 mirror -- factor, gradients, and the row they feed;
//   * `specParBand` is executed too, with one caveat worth stating plainly. The
//     interpreter runs a workgroup's invocations SEQUENTIALLY (and only the first six of
//     them, which is why every dispatch here OVER-dispatches workgroups and leans on the
//     kernel's own bounds test -- checkiso's leg 9 does the same) and treats
//     workgroupBarrier as a no-op, so a tree reduction over shared memory does not
//     survive it: thread 0 runs the whole reduction and the store before thread 1 has
//     written its slot. The
//     leg therefore runs that kernel on states with exactly ONE live perpendicular mode,
//     at mp = 0. Then every thread but tid = 0 has an identically zero accumulator, the
//     tree adds nothing to sh[0] in whatever order it runs, and the interpreter's answer
//     IS the kernel's answer -- for one mode at a time, swept over k_perp, the band ends,
//     the three lanes and the +-kz pair. (An interpreter that really parallelized would
//     give the same number on these states, so the leg cannot silently change meaning.)
//   * the card, the sweep, the pause choreography and the plot are driven on the REAL
//     rmhd3d page booted on stubenv, with only the four solver READBACKS replaced by
//     synthetic data (bootstub's own idiom) -- `gen2dSpec`, `gen2dBands`, `flSpectrum`,
//     `gen2dPanel`, `gen2dRidge` and `drawGen2D` are all the app's.
// The one thing no sandbox can do is run the physics: the stub device executes no compute
// and there is no GPU. So "a press leaves the state alone" is asserted three ways instead
// of one (leg 3), and the Stage B anchor (a band-[0,0] row IS the 1D parallel spectrum) is
// made at the KERNEL, executed and bitwise, plus at the page as wiring (legs 4 and 5).
// No RNG anywhere: every synthetic field here is analytic or a fixed LCG walk.
"use strict";
const fs = require("fs"), os = require("os"), path = require("path"), vm = require("vm");
const { spawnSync } = require("child_process");
const { pathToFileURL } = require("url");
const dir = path.resolve(process.argv[2] || path.join(__dirname, ".."));
const root = path.resolve(dir, "..");
// This plan's base commit. (ea8c927, the ISO feedback touch-up main sits on, is UI only
// and emits the same WGSL byte for byte -- so pinning the earlier commit is the stricter
// statement, not a looser one.)
const BASE = "ad080a2";
// what the feature is allowed to ADD to the emission, and nothing else: two kernels, both
// on the 3D page, both compiled only if the generate button is ever pressed.
const ADDED = { "rmhd2d.html": [], "rmhd3d.html": ["prepGradsBand", "specParBand"] };
// the selftest grid's instances (16 x 16 x 8: NM = 1152, NMP = 144, NZB = 4) -- the size
// the WGSL interpreter runs in milliseconds
const K_GRADS = "selftest :: prepGrads", K_GRADSB = "selftest :: prepGradsBand";
const K_PAR = "selftest :: specPar", K_PARB = "selftest :: specParBand";
const K_DISP = "selftest :: prepDisp";

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};
const rel = (a, b) => Math.abs(a - b) / Math.max(1e-300, Math.abs(b));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "check2dspec-"));
const sh = (cmd, args, opts) => spawnSync(cmd, args, Object.assign(
  { encoding: "utf8", cwd: __dirname, maxBuffer: 1 << 28 }, opts || {}));
const node = (args, opts) => sh(process.execPath, args, opts);
const lastLine = r => ((r.stdout || "") + (r.stderr || "")).trim().split("\n").pop();

// ---------------------------------------------------------------------------
// helpers (checkiso's set)
// ---------------------------------------------------------------------------
// `ovr` is dumpwgsl2's own 5th argument: a JSON object merged into every parameter set, so
// the SAME templates can be emitted on a different box (leg 2 uses it to get kunit != 1).
function dumpKernels(d, page, tag, ovr) {
  const out = path.join(tmp, tag + "-" + page.replace(".html", "") + ".wgsl.txt");
  const args = [path.join(__dirname, "dumpwgsl2.js"), d, page, "", out];
  if (ovr) args.push(JSON.stringify(ovr));
  const r = node(args);
  if (r.status !== 0) { ok(tag + " " + page + ": dumpwgsl2 ran", false, lastLine(r)); return { file: out, k: {} }; }
  const txt = fs.readFileSync(out, "utf8");
  const parts = txt.split(/^########## (.*) ##########$/m);
  const k = {};
  for (let i = 1; i < parts.length; i += 2) k[parts[i]] = parts[i + 1];
  return { file: out, k };
}
function baseDir() {
  const d = path.join(tmp, "base");
  fs.mkdirSync(d, { recursive: true });
  for (const f of ["common.js", "physics.js", "rmhd2d.html", "rmhd3d.html"]) {
    const r = sh("git", ["-C", root, "show", BASE + ":webgpu/" + f], { maxBuffer: 1 << 28 });
    if (r.status !== 0) return null;
    fs.writeFileSync(path.join(d, f), r.stdout);
  }
  return d;
}
// boot() runs at require, its async tail a macrotask later (the checkonepage idiom)
const boot = (d, page, demo) => new Promise(res => {
  const env = require("./stubenv")(d, page, demo);
  setTimeout(() => res(env));
});
async function wgslMod() {
  const p = path.join(__dirname, "node_modules", "wgsl_reflect", "wgsl_reflect.module.js");
  if (!fs.existsSync(p)) return null;
  return await import(pathToFileURL(p).href);
}
const kconst = (src, n) => {
  const m = new RegExp("const " + n + ": (?:i32|u32|f32) = ([-+\\d.e]+)u?;").exec(src);
  return m ? parseFloat(m[1]) : NaN;
};
const bits = a => new Uint32Array(a.buffer, a.byteOffset, a.length);
// the multiset of lines one text has and the other has not. Order-insensitive on purpose:
// a gate that only MOVED a line is a finding either way, and it shows up as both a
// removal and an addition here.
function lineDiff(a, b) {
  const cnt = t => { const m = new Map(); for (const l of t.split("\n")) m.set(l, (m.get(l) || 0) + 1); return m; };
  const A = cnt(a), B = cnt(b), onlyA = [], onlyB = [];
  for (const [l, n] of A) for (let i = 0; i < n - (B.get(l) || 0); i++) onlyA.push(l);
  for (const [l, n] of B) for (let i = 0; i < n - (A.get(l) || 0); i++) onlyB.push(l);
  return { onlyA, onlyB };
}
// ---- the shared half-cosine band, mirrored in fp64 (checkiso's bandFacJs, verbatim) ----
// The same function deliberately: ISO Phase D's factor and this plan's are ONE factor, and
// a second opinion here would hide the day they stop being one.
const bandEdgeJs = t => (t >= 1 ? 1 : (t <= 0 ? 0 : 0.5 - 0.5 * Math.cos(Math.PI * t)));
const bandFacJs = (kn, lo, hi, e) => (lo > 0 ? bandEdgeJs((kn - lo + 0.5 * e) / e) : 1) *
                                     (hi > 0 ? bandEdgeJs((hi + 0.5 * e - kn) / e) : 1);
// a deterministic (phi, psi) state: no RNG anywhere near this repo's checks
function synthState(nm, seed) {
  const f = new Float32Array(4 * nm);
  let s = seed || 12345;
  for (let i = 0; i < f.length; i++) { s = (s * 48271) % 2147483647; f[i] = s / 2147483647 - 0.5; }
  return f;
}

// ---------------------------------------------------------------------------
// a bare vm holding common.js, for the pure functions (the checkaniso / checkpin idiom)
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
const C = vm.runInContext(
  "({ flSpectrum, gen2dBands, GEN_NBAND, GEN_BAND_W, fitKA, specKnee, _anisoLeg, _anisoPeak,"
  + " ANISO_LANES, CHART_TYPES, gen2dTop, GEN2D_TOPMARGIN, GEN2D_SLOPES, logTicks,"
  + " legend, legendLines, PADS, GSW, GSH, SW, SH })", sandbox);

// ===========================================================================
// 1. the emission: parses, resolves, and is base's plus exactly two kernels
// ===========================================================================
// The plan's WGSL rule has two halves and they are NOT the same statement. "The physics
// stepping kernel is byte-identical" is about `prepGrads`, which the RHS runs every step;
// "the idle-path emission is byte-identical" is about everything else a page emits when
// nobody has pressed the button -- and since the sweep's kernels come from a SECOND
// emission off the same templates, that second half is the whole dump: every kernel base
// has, this tree has, byte for byte, plus two pure additions and nothing removed.
function legEmission(state) {
  const wr = path.join(__dirname, "node_modules", "wgsl_reflect", "wgsl_reflect.module.js");
  const env = Object.assign({}, process.env);
  if (fs.existsSync(wr)) env.WGSL_REFLECT = wr;
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const d = dumpKernels(dir, page, "cur");
    state.cur[page] = d.k;
    const r = node([path.join(__dirname, "wgslparse.mjs"), d.file], { env });
    ok(page + ": every emitted kernel parses (+ reserved-word scan)", r.status === 0, lastLine(r));
  }
  const nm = node([path.join(__dirname, "names.mjs"), dir]);
  ok("names.mjs: no redeclares, every free identifier resolves", nm.status === 0, lastLine(nm));
  const files = [path.join(dir, "common.js"), path.join(dir, "physics.js")];
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const t = fs.readFileSync(path.join(dir, page), "utf8");
    const f = path.join(tmp, page.replace(".html", ".js"));
    fs.writeFileSync(f, t.slice(t.indexOf("<script>\n") + 8, t.lastIndexOf("</script>")));
    files.push(f);
  }
  const dp = sh("python3", [path.join(__dirname, "dup.py")].concat(files));
  // (checkiso's note: dup.py's GROUP COUNT is not a stable number to pin -- WHERE the
  // clones are, is.)
  const groups = (dp.stdout || "").split("\n").filter(l => l.indexOf("CLONE") === 0)
    .map(l => l.split("):")[1].split(";").map(s => s.trim().split(":")[0]));
  const bads = groups.filter(g => new Set(g).size !== 2 ||
                                  g.some(f => f === "common.js" || f === "physics.js"));
  ok("dup.py: no clone inside a file, none reaching into the shared core",
     dp.status === 0 && bads.length === 0,
     groups.length + " groups, all app-script twins" + (bads.length ? "; BAD: " + bads[0].join(" ") : ""));

  const bd = baseDir();
  if (!bd) { ok("base " + BASE + " is readable (git show)", false, "git show failed"); return; }
  state.baseDir = bd;
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const base = dumpKernels(bd, page, "base").k;
    state.base[page] = base;
    const cur = state.cur[page] || {};
    const moved = [], gone = [], added = new Set();
    for (const k of new Set(Object.keys(base).concat(Object.keys(cur)))) {
      if (base[k] === cur[k]) continue;
      if (base[k] === undefined) added.add(k.split(" :: ")[1]);
      else if (cur[k] === undefined) gone.push(k);
      else moved.push(k);
    }
    ok(page + ": the idle-path emission is " + BASE + "'s, kernel for kernel",
       moved.length === 0 && gone.length === 0,
       moved.length || gone.length
         ? moved.length + " changed, " + gone.length + " vanished; first: " + (moved[0] || gone[0])
         : Object.keys(cur).length + " kernels, byte-identical");
    const gk = Object.keys(cur).filter(k => / :: prepGrads$/.test(k));
    ok("  ... including the kernel the RHS steps through, prepGrads, at every preset",
       gk.length >= 4 && gk.every(k => base[k] === cur[k]), gk.length + " emissions");
    const gotA = [...added].sort().join(", "), wantA = ADDED[page].join(", ");
    ok("  ... and what it ADDS is exactly the sweep's two kernels", gotA === wantA, gotA || "none");
  }
}

// ===========================================================================
// 2. the two banded kernels ARE the unbanded templates, plus the shared factor
// ===========================================================================
// The plan's claim is that the sweep, the RHS and the 1D parallel spectrum come off ONE
// template each, so the band is an option and not a fork. That is a statement about the
// TEXT, and this is it: the shared half-cosine block is prepDisp's own, byte for byte, in
// both new kernels; and the lines that differ from the unbanded emission are exactly the
// listed ones -- a line that changed and is not here fails, and so does a line here that
// stopped changing.
const EXPECT = {
  grads: {
    // (unbanded) -> (banded): the two READ lines the factor rides on
    rewrite: [["  let phi: vec2<f32> = fields[m];", "  let phi: vec2<f32> = bf * fields[m];"],
              ["  let psi: vec2<f32> = fields[NM + m];", "  let psi: vec2<f32> = bf * fields[NM + m];"]],
    add: [/^@group\(0\) @binding\(3\) var<uniform> md: Mode;$/,
          /^const INVKU: f32 = /,
          /^  let bf: f32 = bandFac\(sqrt\(g\.z\) \* INVKU, md\.klo, md\.khi\);$/]
  },
  par: {
    rewrite: [["    let w: f32 = 0.5 * gridA[mp].z * gridB[mp].w * INVN2;",
               "    let w: f32 = 0.5 * gridA[mp].z * gridB[mp].w * INVN2 * bf * bf;"],
              ["  if (tid == 0u) { bins[wgid.x] = sh[0].x; bins[NZB + wgid.x] = sh[0].y;",
               "  if (tid == 0u) { let o: u32 = wgid.y * 3u * NZB + wgid.x;"],
              ["                   bins[2u*NZB + wgid.x] = sh[0].z; }",
               "                   bins[o] = sh[0].x; bins[NZB + o] = sh[0].y; bins[2u*NZB + o] = sh[0].z; }"]],
    add: [/^@group\(0\) @binding\(4\) var<storage, read> bnd: array<vec4<f32>>;$/,
          /^const INVKU: f32 = /,
          /^  let klo: f32 = bnd\[wgid\.y\]\.x;$/,
          /^  let khi: f32 = bnd\[wgid\.y\]\.y;$/,
          /^    let bf: f32 = bandFac\(sqrt\(gridA\[mp\]\.z\) \* INVKU, klo, khi\);$/]
  }
};
function legTemplates(state) {
  const cur = state.cur["rmhd3d.html"] || {};
  // the shared factor, lifted out of the DISPLAY kernel that already carried it (ISO
  // Phase D). Both new kernels must contain that text and not a copy of it.
  const disp = cur[K_DISP] || "";
  const i0 = disp.indexOf("const BAND_EDGE:"), TAIL = "return f;\n}\n";
  const j0 = disp.indexOf(TAIL, i0);
  const BLOCK = (i0 >= 0 && j0 > i0) ? disp.slice(i0, j0 + TAIL.length) : null;
  const sm = /^struct Mode \{[^\n]*$/m.exec(disp);
  ok("the half-cosine band block is lifted out of prepDisp's own emission",
     !!BLOCK && BLOCK.length > 200 && /bandFac/.test(BLOCK),
     BLOCK ? BLOCK.split("\n").length + " lines" : "not found");
  for (const [nm, k] of [["prepGradsBand", K_GRADSB], ["specParBand", K_PARB]]) {
    ok(nm + " carries that factor VERBATIM (one factor, three consumers)",
       !!BLOCK && (cur[k] || "").indexOf(BLOCK) >= 0);
  }
  ok("prepGradsBand reads prepDisp's own Mode struct, not a second one",
     !!sm && (cur[K_GRADSB] || "").indexOf(sm[0]) >= 0, sm ? sm[0].slice(0, 60) + "..." : "none");
  // Every preset this app ships fixes Lx = Ly = 2*pi, so kunit == 1 and INVKU == 1 -- which
  // makes "measured in the spectrum chart's own k unit" true of ANY conversion, including
  // none at all. So the same templates are re-emitted through the page's own makeGrid /
  // buildShaders on a 4*pi box, where kunit is 1/2: there a dropped or inverted conversion
  // is a factor of two and the check below has something to fail on. (Leg 4(a) then RUNS
  // that emission, so the multiply is measured in that unit too, not just declared in it.)
  const alt = state.alt = dumpKernels(dir, "rmhd3d.html", "alt",
                                      { Lx: 4 * Math.PI, Ly: 4 * Math.PI }).k;
  const altKU = kconst(alt["selftest :: spectrum"] || "", "INVKU");
  ok("a 4pi box really moves the k unit: the spectrum chart's own INVKU goes 1 -> 2",
     altKU === 2 && kconst(cur["selftest :: spectrum"] || "", "INVKU") === 1,
     "INVKU = " + kconst(cur["selftest :: spectrum"] || "", "INVKU") + " at 2pi, "
     + altKU + " at 4pi");
  for (const [nm, kb, k1, E] of [["prepGradsBand", K_GRADSB, K_GRADS, EXPECT.grads],
                                 ["specParBand", K_PARB, K_PAR, EXPECT.par]]) {
    const a = cur[k1] || "", b = cur[kb] || "";
    const D = lineDiff(a, b);
    const shared = new Set((BLOCK ? BLOCK.replace(/\n$/, "").split("\n") : [])
                           .concat(sm ? [sm[0]] : []));
    const rwA = E.rewrite.map(r => r[0]), rwB = E.rewrite.map(r => r[1]);
    const leftA = D.onlyA.filter(l => rwA.indexOf(l) < 0);
    const leftB = D.onlyB.filter(l => rwB.indexOf(l) < 0 && !shared.has(l)
                                      && !E.add.some(re => re.test(l)));
    ok(nm + " is " + k1.split(" :: ")[1] + "'s own text plus the band, line for line",
       leftA.length === 0 && leftB.length === 0 &&
       rwA.every(l => D.onlyA.indexOf(l) >= 0) && rwB.every(l => D.onlyB.indexOf(l) >= 0),
       leftA.length || leftB.length
         ? "unexpected: " + JSON.stringify((leftA[0] || leftB[0] || "").slice(0, 64))
         : E.rewrite.length + " lines rewritten, " + (D.onlyB.length - E.rewrite.length) + " added");
    // the band ends are measured in the SPECTRUM chart's own k unit, as prepDisp's are:
    // a factor measuring k_perp in anything else would be a lie on the card's x axis. Made
    // at BOTH box sizes, so it is the conversion that is asserted and not the identity.
    const sp = cur["selftest :: spectrum"] || "", ab = kconst(alt[kb] || "", "INVKU");
    ok("  ... and its band is in the spectrum chart's own k unit, in EITHER box",
       kconst(b, "INVKU") === kconst(sp, "INVKU") && isFinite(kconst(b, "INVKU")) &&
       ab === altKU && altKU === 2,
       "INVKU = " + kconst(b, "INVKU") + " at 2pi, " + ab + " at 4pi");
    // read-only state: the whole feature's first promise, made at the declaration
    ok("  ... with `fields` bound READ-ONLY and exactly one read_write target",
       /@group\(0\) @binding\(0\) var<storage, read> fields:/.test(b) &&
       (b.match(/var<storage, read_write>/g) || []).length === 1,
       (b.match(/var<storage, read_write> \w+/g) || []).join(" | "));
  }
}

// ===========================================================================
// 3. state invariance: a generate press leaves (phi_k, psi_k) BITWISE unchanged
// ===========================================================================
// Three statements, because no sandbox makes all three at once:
//   (a) the kernels, EXECUTED: after prepGradsBand and specParBand have run over a band
//       set, the `fields` array is bit-identical, word for word. This is the only one
//       that watches the GPU-side arithmetic actually happen.
//   (b) the press, TRACED: on the booted page nothing in a generate press writes, clears
//       or copies INTO the state buffer -- every buffer is named off `solver.buf`, so the
//       trace is in the app's own vocabulary.
//   (c) the press, READ BACK: with the stub's buffers given real backing stores, the whole
//       4*nm-word state readback before the press is bitwise the readback after it (full
//       buffers, not norms). The stub runs no compute, so on its own (c) only catches a
//       CPU-side write -- exactly the half (a) cannot see, and the other way round.
// ... and two more, because (a)-(c) are all blind to the WIRING between them. (a) executes
// hand-built buffers, not the app's bind groups; (b) records that a buffer was written, not
// what was in it; (c) reads back on a device that runs no compute. So a bind group with
// `fields` in the read_write slot -- legal WGSL, legal WebGPU (the buffer carries STORAGE
// usage), and on a real device a generate press that OVERWRITES the state with gradients --
// would pass all three. Hence:
//   (d) the press, BOUND: every bind group the sweep builds, by solver.buf NAME and in
//       binding order, looked up by identity off `solver._gen` after the press.
//   (e) the press, WRITTEN: the CONTENTS of the three uniforms it fills -- the Mode words
//       per band (modeWords(0,0,0,[lo,hi]), so swapped ends fail), the coordinate pass's
//       band table (lo, hi, 0, 0) per band, and the field-line seed grid's own `genCfg`
//       ([GEN_SIDE,0,0,0], written once in _genInit and never again) -- read out of
//       MEMDEV's backing stores.
//   (f) the press, DISPATCHED: the workgroup counts readGenBand's own pass issues for the
//       banded prep and the march, off MEMDEV's dispatch recorder (leg 5's W.disp idiom).
//       (a)-(e) are all blind to this: (a) hand-dispatches the interpreter itself, (d)
//       checks binding identity and not size, and an undersized march simply leaves stale
//       gradients in 3/4 of the polylines -- invisible to a buffer that is the right one,
//       bound in the right slot, just not fully written.
const MEMDEV = `function(){
  // give every buffer a real backing store, so readBuf really reads what was written:
  // writeBuffer and copyBufferToBuffer move bytes and getMappedRange hands the store out.
  const d = solver.device;
  if (d.__mem) return true;
  const store = new Map();
  const st = b => { if (!store.has(b)) store.set(b, new ArrayBuffer(b.size)); return store.get(b); };
  const bind = b => { if (!b.__mem) { b.__mem = 1; const s = st(b); b.getMappedRange = () => s; } return b; };
  const oc = d.createBuffer.bind(d), ow = d.queue.writeBuffer.bind(d.queue), oe = d.createCommandEncoder.bind(d);
  const obg = d.createBindGroup.bind(d);
  globalThis.TRACE = [];
  // every bind group ever built on this device, by the object the app keeps -> its buffers
  globalThis.BGREC = new Map();
  // one array of dispatchWorkgroups calls per compute pass (leg 5's W.disp idiom, folded
  // into the same patch): a fresh array per beginComputePass, [x, y] per call, in order.
  globalThis.DISP = [];
  d.createBindGroup = o => {
    const g = obg(o);
    globalThis.BGREC.set(g, o.entries.map(e => e.resource.buffer));
    return g;
  };
  const nameOf = b => {
    if (!solver) return null;
    for (const k of Object.keys(solver.buf)) if (solver.buf[k] === b) return k;
    return null;
  };
  d.createBuffer = o => bind(oc(o));
  d.queue.writeBuffer = function (b, off, data) {
    bind(b);
    globalThis.TRACE.push(["write", nameOf(b), off, data.byteLength]);
    new Uint8Array(st(b)).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), off);
    return ow(b, off, data);
  };
  d.createCommandEncoder = function () {
    const e = oe(), occ = e.copyBufferToBuffer.bind(e), ocl = e.clearBuffer.bind(e),
          obp = e.beginComputePass.bind(e);
    e.copyBufferToBuffer = (a, ao, b, bo, n) => {
      bind(a); bind(b);
      globalThis.TRACE.push(["copy", nameOf(a), nameOf(b), n]);
      new Uint8Array(st(b)).set(new Uint8Array(st(a), ao, n), bo);
      return occ(a, ao, b, bo, n);
    };
    e.clearBuffer = b => { globalThis.TRACE.push(["clear", nameOf(b)]); return ocl(b); };
    e.beginComputePass = () => {
      const p = obp(), od = p.dispatchWorkgroups.bind(p), calls = [];
      globalThis.DISP.push(calls);
      p.dispatchWorkgroups = (x, y, z) => { calls.push([x, y === undefined ? 1 : y]); return od(x, y, z); };
      return p;
    };
    return e;
  };
  d.__mem = 1;
  return true; }`;
// a deterministic pattern into the live state buffer, through the app's own queue
const SEEDSTATE = `function(){
  const n = 4 * solver.g.nm, a = new Float32Array(n);
  let s = 20260811;
  for (let i = 0; i < n; i++) { s = (s * 48271) % 2147483647; a[i] = s / 2147483647 - 0.5; }
  solver.device.queue.writeBuffer(solver.buf.fields, 0, a);
  return n; }`;
const READSTATE = `function(){ return readBuf(solver.device, solver.buf.fields, 16 * solver.g.nm)
  .then(a => Array.from(new Uint32Array(a.buffer, a.byteOffset, a.length))); }`;
// (e)'s instrument: the two sweep readbacks are wrapped so that the band they were ASKED
// for and the uniform words the app actually left in the buffer are captured side by side.
// Read AFTER the call returns, which is when the write has certainly happened and before
// anything else can touch either buffer. The wrappers call straight through -- the real
// encode / dispatch / readback path is what leg 3 is watching, and it is untouched.
const BANDREC = `function(){
  const sv = solver, ogb = sv.readGenBand.bind(sv), og2 = sv.readGen2D.bind(sv);
  const words = b => Array.from(new Uint32Array(b.getMappedRange()));
  globalThis.BANDS = []; globalThis.GBAND = null;
  sv.readGenBand = async b => {
    const r = await ogb(b);
    globalThis.BANDS.push([b.lo, b.hi, words(sv.buf.genMode)]);
    return r;
  };
  sv.readGen2D = async bs => {
    const r = await og2(bs);
    globalThis.GBAND = [bs.map(b => [b.lo, b.hi]), words(sv.buf.genBand)];
    return r;
  };
  return true; }`;
// the sweep's bind groups, by solver.buf name, found by IDENTITY off the object the app
// kept -- not by guessing which recorded group was which
const BGNAMES = `function(){
  const sv = solver, rec = globalThis.BGREC, g = sv && sv._gen;
  if (!g || !rec) return null;
  const nameOf = b => { for (const k of Object.keys(sv.buf)) if (sv.buf[k] === b) return k;
                        return "(not the solver's)"; };
  const of = bg => { const e = rec.get(bg); return e ? e.map(nameOf).join(",") : "(unrecorded)"; };
  return { prep: of(g.bgPrep), fl: of(g.bgFL), g2d: of(g.bg2d) }; }`;
// a captured word array back as f32 (the uniforms are written through Float32Array views,
// so the expected values are compared at f32 precision -- Math.fround, not the double)
const f32of = w => new Float32Array(Uint32Array.from(w).buffer);

async function legInvariance(state) {
  const M = state.M, cur = state.cur["rmhd3d.html"] || {};
  // ---- (a) the kernels, executed ------------------------------------------
  if (M) {
    const src = cur[K_GRADSB], psrc = cur[K_PARB];
    const N = { nm: kconst(src, "NM"), nmp: kconst(src, "NMP"), nzb: kconst(psrc, "NZB") };
    const gA = new Float32Array(4 * N.nmp), gB = new Float32Array(4 * N.nmp);
    for (let mp = 0; mp < N.nmp; mp++) {
      gA[4 * mp] = 0.25 * ((mp % 7) - 3); gA[4 * mp + 1] = 0.5 * (mp % 5);
      gA[4 * mp + 2] = 0.25 * (1 + (mp % 13)); gB[4 * mp + 3] = (mp % 2) ? 2 : 1;
    }
    const f0 = synthState(N.nm), before = Array.from(bits(f0));
    const bands = [[0, 0], [1, 3], [2.5, 4], [0, 2]];
    for (const [lo, hi] of bands) {
      const u = new ArrayBuffer(32), uf = new Float32Array(u);
      uf[4] = lo; uf[5] = hi;
      new M.WgslExec(new M.WgslParser().parse(src)).dispatchWorkgroups("main",
        [N.nm, 1, 1],
        { 0: { 0: f0, 1: gA, 2: new Float32Array(16 * N.nm), 3: new Uint32Array(u) } });
      const bnd = new Float32Array(4);
      bnd[0] = lo; bnd[1] = hi;
      new M.WgslExec(new M.WgslParser().parse(psrc)).dispatchWorkgroups("main",
        [N.nzb, 1, 1],
        { 0: { 0: f0, 1: gA, 2: gB, 3: new Float32Array(3 * N.nzb), 4: bnd } });
    }
    const after = Array.from(bits(f0));
    ok("(a) EXECUTED: with both banded kernels run over " + bands.length
       + " bands, (phi_k, psi_k) is bit-identical",
       before.length === after.length && before.every((v, i) => v === after[i]),
       4 * N.nm + " f32 words compared as raw bits");
  } else ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/");

  // ---- (b) + (c) the press, on the real page -------------------------------
  const env = state.env3d || (state.env3d = await boot(dir, "rmhd3d.html"));
  ok("the memory-backed device patch installs", env.run(MEMDEV) === true);
  // only the two SPECTRUM readbacks are synthetic here (they are what gives the sweep a
  // band set); the row readbacks are the solver's REAL ones, so what the trace below
  // watches is the real encode / dispatch / map path of `readGenBand` and `readGen2D`
  installSynth(env, false);
  ok("the band / uniform recorder installs", env.run(BANDREC) === true);
  const nwords = env.run(SEEDSTATE);
  const s0 = await env.run(READSTATE);
  env.run("function(){ globalThis.TRACE = []; globalThis.DISP = []; }");
  const done = await pressGenerate(env);
  ok("a generate press runs to completion on the booted page", done);
  // the trace is COPIED out first (it is a live array in the page, and the readback below
  // is itself a copyBufferToBuffer out of the state buffer -- this check's instrument,
  // not part of the press)
  const tr = env.run("function(){ return globalThis.TRACE.map(t => t.slice()); }");
  const s1 = await env.run(READSTATE);
  let same = s0.length === nwords && s1.length === nwords, firstDiff = -1;
  for (let i = 0; i < Math.min(s0.length, s1.length); i++)
    if (s0[i] !== s1[i]) { same = false; firstDiff = i; break; }
  ok("(c) READ BACK: the whole state buffer is bitwise what it was before the press",
     same, same ? nwords + " f32 words, bit for bit" : "word " + firstDiff + " moved");
  const wrote = tr.filter(t => t[0] === "write" && t[1] === "fields");
  const into = tr.filter(t => t[0] === "copy" && t[2] === "fields");
  const cleared = tr.filter(t => t[0] === "clear" && t[1] === "fields");
  ok("(b) TRACED: the press never writes, copies into, or clears the state buffer",
     wrote.length === 0 && into.length === 0 && cleared.length === 0,
     tr.length + " buffer ops; " + wrote.length + " writes / " + into.length + " copies / "
     + cleared.length + " clears of `fields`");
  // (a null name is a buffer that is not the solver's at all -- a display card's own
  // uniform, written by the pause the press starts with)
  const nm = n => n === null ? "(a card's)" : n;
  const names = [...new Set(tr.filter(t => t[0] === "write").map(t => t[1]))];
  ok("  ... the only solver buffers it writes are the sweep's own uniform / band table",
     names.every(n => n === null || /^gen/.test(n)), names.map(nm).join(", ") || "none");
  const from = [...new Set(tr.filter(t => t[0] === "copy").map(t => t[1]))];
  ok("  ... and the only buffers it copies OUT of are the sweep's own readback targets",
     from.every(n => n === null || /^gen/.test(n)), from.map(nm).join(", ") || "none");

  // ---- (d) the press, BOUND ------------------------------------------------
  const BG = env.run(BGNAMES);
  ok("(d) BOUND: the banded prep reads the STATE and writes the RHS's gradient scratch, "
     + "in prepGradsBand's own binding order",
     !!BG && BG.prep === "fields,gridA,gradsK,genMode",
     BG ? "[" + BG.prep + "]" : "no bind groups recorded");
  ok("  ... the sweep's field-line march reads realGrads and writes its OWN polyline pair",
     !!BG && BG.fl === "realGrads,genPos,genSmp,genCfg", BG ? "[" + BG.fl + "]" : "-");
  ok("  ... and the coordinate pass reads the state and both grids, plus the band table",
     !!BG && BG.g2d === "fields,gridA,gridB,gen2d,genBand", BG ? "[" + BG.g2d + "]" : "-");

  // ---- (e) the press, WRITTEN ----------------------------------------------
  const BW = env.run("function(){ return globalThis.BANDS.map(b => [b[0], b[1], b[2].slice()]); }");
  let badm = 0, firstm = "";
  for (const rec of BW || []) {
    const u = Uint32Array.from(rec[2]), f = f32of(rec[2]);
    const good = u.length === 8 && u[0] === 0 && u[1] === 0 && u[2] === 0 && u[3] === 0 &&
                 f[4] === Math.fround(rec[0]) && f[5] === Math.fround(rec[1]) &&
                 u[6] === 0 && u[7] === 0;
    if (!good && !badm++)
      firstm = "asked [" + rec[0] + ", " + rec[1] + "], wrote [" + Array.from(f).join(", ") + "]";
  }
  ok("(e) WRITTEN: every band's Mode uniform is modeWords(0, 0, 0, [lo, hi]) -- mode / "
     + "zslice / cmap / pad zero and the two ends in THAT order",
     !!BW && BW.length > 2 && badm === 0,
     (BW ? BW.length : 0) + " uniform writes" + (firstm ? "; first bad: " + firstm : ""));
  const GB = env.run("function(){ const g = globalThis.GBAND;"
    + " return g && [g[0].map(b => b.slice()), g[1].slice()]; }");
  let badb = 0, firstb = "";
  if (GB) {
    const f = f32of(GB[1]), bs = GB[0], nB = bs.length;
    for (let j = 0; 4 * j < f.length; j++) {
      const lo = j < nB ? Math.fround(bs[j][0]) : 0, hi = j < nB ? Math.fround(bs[j][1]) : 0;
      if (!(f[4 * j] === lo && f[4 * j + 1] === hi && f[4 * j + 2] === 0 && f[4 * j + 3] === 0)
          && !badb++)
        firstb = "band " + j + ": asked [" + (j < nB ? bs[j].join(", ") : "-") + "], wrote ["
                 + f.slice(4 * j, 4 * j + 4).join(", ") + "]";
    }
  }
  ok("  ... and the coordinate pass's band table is (lo, hi, 0, 0) per band, in the band "
     + "set's own order and zero past the last one",
     !!GB && GB[0].length > 2 && badb === 0,
     GB ? GB[0].length + " bands in a " + (f32of(GB[1]).length / 4) + "-slot table"
          + (firstb ? "; " + firstb : "") : "no band table written");
  // genCfg is written exactly once, in _genInit, not per band like genMode/genBand above --
  // so it is read the same way (out of MEMDEV's backing store) but only once, after the
  // whole press. A mutated seed side here is invisible to (a)-(d): (a) never runs
  // `fieldLine` at all, (b)/(c) do not look at uniform CONTENTS, and (d) checks that the
  // right buffer is bound in the right slot, not what is inside it.
  const CFG = env.run("function(){ const b = solver.buf.genCfg;"
    + " return b ? Array.from(new Uint32Array(b.getMappedRange())) : null; }");
  const sideWant = env.run("function(){ return GEN_SIDE; }");
  ok("  ... and the field-line seed grid's own uniform (genCfg) is [GEN_SIDE, 0, 0, 0]",
     !!CFG && CFG.length === 4 && CFG[0] === sideWant && CFG[1] === 0 && CFG[2] === 0 && CFG[3] === 0,
     CFG ? "[" + CFG.join(", ") + "], GEN_SIDE = " + sideWant : "genCfg not read");

  // ---- (f) the press, DISPATCHED --------------------------------------------
  // leg 5's W.disp idiom, generalized: one array of dispatchWorkgroups calls per compute
  // pass. readGen2D's own pass is a single dispatch (asserted in leg 5); readGenBand's
  // never is -- prep, then encodeInv3D's borrowed inverse transform, then the march -- so
  // a pass with MORE than one call is one of readGenBand's, in call order: first the prep,
  // last (right before p.end()) the march. Expected sizes come off the app's own kernel
  // text (its declared workgroup_size) and its own nm / GEN_SIDE -- nothing here is a
  // number this script chose.
  const DISP = env.run("function(){ return globalThis.DISP.map(c => c.slice()); }");
  const wgSize = src => { const m = /@workgroup_size\((\d+)\)/.exec(src || ""); return m ? parseInt(m[1], 10) : NaN; };
  const nmApp = env.run("function(){ return solver.g.nm; }");
  const sideApp = env.run("function(){ return GEN_SIDE; }");
  const expPrep = Math.ceil(nmApp / wgSize(cur[K_GRADSB]));
  const expMarch = Math.ceil((sideApp * sideApp) / wgSize(cur["selftest :: fieldLine"]));
  const bandPasses = (DISP || []).filter(c => c.length > 1);
  let badd = 0, firstd = "";
  for (const c of bandPasses) {
    const first = c[0], last = c[c.length - 1];
    const good = !!first && !!last && first[0] === expPrep && first[1] === 1 &&
                 last[0] === expMarch && last[1] === 1;
    if (!good && !badd++)
      firstd = "prep " + JSON.stringify(first) + ", march " + JSON.stringify(last);
  }
  ok("(f) DISPATCHED: every per-band pass dispatches ceil(nm/64) workgroups for the banded "
     + "prep and ceil(GEN_SIDE^2/64) for the march (sizes read off the app's own kernels)",
     bandPasses.length > 2 && badd === 0,
     bandPasses.length + " band passes, want prep=" + expPrep + " march=" + expMarch
     + (firstd ? "; first bad: " + firstd : ""));
}

// ===========================================================================
// 4. the fp64 mirror: band factor x gradient x periodogram row
// ===========================================================================
// The plan's chain, one stage at a time and then end to end:
//   (a) the FACTOR the emitted prepGradsBand applies, swept over k_perp, against the fp64
//       half-cosine -- and the same sweep off specParBand's own copy of it, so "one
//       factor, two consumers" is a measurement and not a comment;
//   (b) the GRADIENTS it multiplies: all eight k-space lanes on the page's own grid, per
//       band, against the fp64 mirror bf * (i k f) -- and, with both ends off, bit for bit
//       the RHS's own prepGrads;
//   (c) the ROW those become: the band-passed gradients inverse-transformed by a direct
//       fp64 DFT, sampled on the sweep's own GEN_SIDE^2 seed grid with the marcher's own
//       bilinear gather (psi = 0, so b_perp = 0 and the field lines are straight in z --
//       checkk's analytic case), fed to the app's REAL flSpectrum and compared with an
//       fp64 Hann periodogram of the same samples written here from the definition;
//   (d) the coordinate row: specParBand executed one perpendicular mode at a time (see the
//       header note) against the fp64 mirror of its weight, lanes and +-kz folding -- and
//       its band-[0,0] row against the 1D specPar's own bins, bitwise (the Stage B anchor).
const PROBE = `
@group(0) @binding(5) var<storage, read> pk: array<f32>;
@group(0) @binding(6) var<storage, read_write> po: array<f32>;
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  po[i] = bandFac(pk[i + 2u], pk[0], pk[1]);
}`;
// the page's own selftest grid, straight out of makeGrid (checkiso's gridSnip)
const GRIDSNIP = `function(){
  const R = REFVEC;
  const P = { nx: R.nx, ny: R.ny, nz: R.nz, Lx: R.Lx, Ly: R.Ly, Lz: R.Lz, diss: R.diss,
              hyper: R.hyper, zdiss: R.z_diss_k, fshell: R.fshell };
  const g = makeGrid(P);
  return { gridA: Array.from(g.gridA), gridB: Array.from(g.gridB), kunit: g.kunit,
           nkx: g.nkx, nky: g.nky, nz: g.nz, nx: P.nx, ny: P.ny,
           nb: nbins(P.nx, P.ny), side: GEN_SIDE }; }`;

// half-spectrum -> full complex cube by the rfftn reality condition
// F(-kz,-kx,-ky) = conj(F(kz,kx,ky)), then f(r) = (1/N) sum_k F e^{+2pi i k.r/n} as a
// direct fp64 DFT written from the definition (checksh's toReal, done one axis at a time
// so it costs N*(nz+nx+ny) instead of N^2 -- still no FFT library, still not the app's).
function toReal(comp, G) {
  const { nx, ny, nz, nky } = G, N = nx * ny * nz;
  let re = new Float64Array(N), im = new Float64Array(N);
  for (let iz = 0; iz < nz; iz++) for (let ix = 0; ix < nx; ix++) for (let iy = 0; iy < ny; iy++) {
    let c;
    if (iy < nky) c = comp((iz * nx + ix) * nky + iy);
    else { const q = comp((((nz - iz) % nz) * nx + ((nx - ix) % nx)) * nky + (ny - iy)); c = [q[0], -q[1]]; }
    const f = (iz * nx + ix) * ny + iy;
    re[f] = c[0]; im[f] = c[1];
  }
  const axis = (n, outer, inner) => {
    const r2 = new Float64Array(N), i2 = new Float64Array(N);
    for (let o = 0; o < outer; o++) for (let p = 0; p < n; p++) for (let q = 0; q < inner; q++) {
      let sr = 0, si = 0;
      for (let k = 0; k < n; k++) {
        const a = 2 * Math.PI * k * p / n, c = Math.cos(a), s = Math.sin(a);
        const j = (o * n + k) * inner + q;
        sr += re[j] * c - im[j] * s; si += re[j] * s + im[j] * c;
      }
      const j = (o * n + p) * inner + q;
      r2[j] = sr; i2[j] = si;
    }
    re = r2; im = i2;
  };
  axis(nz, 1, nx * ny);
  axis(nx, nz, ny);
  axis(ny, nz * nx, 1);
  const out = new Float64Array(N);
  for (let i = 0; i < N; i++) out[i] = re[i] / N;
  return out;
}
// the marcher's own bilinear gather (samp2), periodic in x and y, on one z plane
function bilin(f, G, iz, gx, gy) {
  const { nx, ny } = G, fx = Math.floor(gx), fy = Math.floor(gy);
  const wr = (i, n) => ((i % n) + n) % n;
  const x0 = wr(fx, nx), x1 = wr(fx + 1, nx), y0 = wr(fy, ny), y1 = wr(fy + 1, ny);
  const o = iz * nx * ny, tx = gx - fx, ty = gy - fy;
  const lo = f[o + x0 * ny + y0] * (1 - tx) + f[o + x1 * ny + y0] * tx;
  const hi = f[o + x0 * ny + y1] * (1 - tx) + f[o + x1 * ny + y1] * tx;
  return lo * (1 - ty) + hi * ty;
}
// flSpectrum's three lanes, from the definition: fp64, a direct DFT over z, no fftPow2
function rowJs(smp, nl, nz) {
  const nzb = nz >> 1, out = new Float64Array(3 * nzb), w = new Float64Array(nz);
  let w2 = 0;
  for (let j = 0; j < nz; j++) { w[j] = 0.5 * (1 - Math.cos(2 * Math.PI * j / nz)); w2 += w[j] * w[j]; }
  w2 /= nz;
  const nrm = 1 / (nz * nz * w2 * nl);
  for (let l = 0; l < nl; l++) for (let b = 1; b <= nzb; b++) {
    const f = (2 * b === nz) ? 1 : 2, R = [0, 0, 0, 0], I = [0, 0, 0, 0];
    for (let c = 0; c < 4; c++) for (let j = 0; j < nz; j++) {
      const a = -2 * Math.PI * b * j / nz, v = w[j] * smp[4 * (l * nz + j) + c];
      R[c] += v * Math.cos(a); I[c] += v * Math.sin(a);
    }
    let eu = 0, eb = 0, hc = 0;
    for (let c = 0; c < 2; c++) {
      eu += R[c] * R[c] + I[c] * I[c];
      eb += R[c + 2] * R[c + 2] + I[c + 2] * I[c + 2];
      hc += R[c] * R[c + 2] + I[c] * I[c + 2];
    }
    out[b - 1] += 0.5 * f * eu * nrm;
    out[nzb + b - 1] += 0.5 * f * eb * nrm;
    out[2 * nzb + b - 1] += f * hc * nrm;
  }
  return out;
}
// the emitted prepGrads / prepGradsBand, EXECUTED. The interpreter runs six invocations
// per workgroup rather than the 64 the kernel declares, so dispatch NM workgroups and let
// the kernel's own `m >= NM` test discard the surplus: over-dispatch, never under (a mode
// it skipped would show up as a mismatch below, not as a silent pass).
function runGrads(M, src, N, fields, gridA, band) {
  const outg = new Float32Array(16 * N.nm);
  const b = new ArrayBuffer(32), f = new Float32Array(b);
  f[4] = band[0]; f[5] = band[1];
  new M.WgslExec(new M.WgslParser().parse(src)).dispatchWorkgroups("main",
    [N.nm, 1, 1], { 0: { 0: fields, 1: gridA, 2: outg, 3: new Uint32Array(b) } });
  return outg;
}
async function legMirror(state) {
  const M = state.M;
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  const cur = state.cur["rmhd3d.html"] || {};
  const src = cur[K_GRADSB], psrc = cur[K_PARB], src1 = cur[K_GRADS];
  const env = state.env3d || (state.env3d = await boot(dir, "rmhd3d.html"));
  const G0 = env.run(GRIDSNIP);
  const G = { nx: G0.nx, ny: G0.ny, nz: G0.nz, nkx: G0.nkx, nky: G0.nky };
  const N = { nm: kconst(src, "NM"), nmp: kconst(src, "NMP"), nzb: kconst(psrc, "NZB") };
  const E = kconst(src, "BAND_EDGE"), ku = G0.kunit, nb = G0.nb;
  const gA = Float32Array.from(G0.gridA), gB = Float32Array.from(G0.gridB);
  const BAND = [3, 7];

  // ---- (a) the factor, swept over k_perp ----------------------------------
  // A synthetic gridA (k_perp = kn*kunit, kx = 1) is the only way to land samples ON the
  // edges themselves; with phi = (1, 0) the kernel's first output lane is (0, bf), so the
  // sweep reads the factor itself.
  {
    const nS = N.nmp, knAt = i => (i * (nb + 1)) / (nS - 1);
    const gs = new Float32Array(4 * nS), fs2 = new Float32Array(4 * N.nm);
    for (let i = 0; i < nS; i++) { const k = knAt(i) * ku; gs[4 * i] = 1; gs[4 * i + 2] = k * k; }
    for (let m = 0; m < N.nm; m++) fs2[2 * m] = 1;
    const g = runGrads(M, src, N, fs2, gs, BAND);
    let em = 0, out01 = 0, mono = 0, notOne = 0, notZero = 0, prev = -1;
    for (let i = 0; i < nS; i++) {
      const kn = knAt(i), v = g[2 * i + 1];
      em = Math.max(em, Math.abs(v - bandFacJs(kn, BAND[0], BAND[1], E)));
      if (!(v >= 0 && v <= 1)) out01++;
      if (kn >= BAND[0] + 0.5 * E && kn <= BAND[1] - 0.5 * E && v !== 1) notOne++;
      if ((kn <= BAND[0] - 0.5 * E || kn >= BAND[1] + 0.5 * E) && v !== 0) notZero++;
      if (prev >= 0) {
        if (kn <= BAND[0] && v < prev - 1e-7) mono++;
        if (kn >= BAND[1] && v > prev + 1e-7) mono++;
      }
      prev = v;
    }
    ok("(a) the factor prepGradsBand applies IS the fp64 half-cosine band", em < 1e-6,
       "max |delta| = " + em.toExponential(2) + " over " + nS + " samples of k_perp");
    ok("  ... in [0,1], monotone across each edge, EXACTLY 1 inside and EXACTLY 0 outside",
       out01 === 0 && mono === 0 && notOne === 0 && notZero === 0,
       out01 + " out of range, " + mono + " non-monotone, " + notOne + " passband != 1, "
       + notZero + " stopband != 0");
    // ... and specParBand's own copy of bandFac, called directly: its `main` is a workgroup
    // reduction, so the factor is reached through a wrapper entry point (checkiso's PROBE)
    const nP = 65, pk = new Float32Array(2 + nP), po = new Float32Array(nP);
    pk[0] = BAND[0]; pk[1] = BAND[1];
    for (let i = 0; i < nP; i++) pk[2 + i] = (i * (nb + 1)) / (nP - 1);
    new M.WgslExec(new M.WgslParser().parse(psrc + PROBE)).dispatchWorkgroups("probe",
      [nP, 1, 1], { 0: { 0: new Float32Array(4), 1: gA, 2: gB, 3: new Float32Array(4),
                         4: new Float32Array(4), 5: pk, 6: po } });
    let em2 = 0;
    for (let i = 0; i < nP; i++)
      em2 = Math.max(em2, Math.abs(po[i] - bandFacJs(pk[2 + i], BAND[0], BAND[1], E)));
    ok("  ... and the coordinate kernel's copy of it answers the same, k for k", em2 < 1e-6,
       "max |delta| = " + em2.toExponential(2) + " over " + nP + " samples");
    // ... and the SAME sweep off leg 2's 4*pi-box emission, where kunit = 1/2. The kernel
    // is handed k_perp in absolute units and must still land on the fp64 band at k/kunit,
    // so a conversion that is missing, inverted or doubled is a factor of two HERE and
    // exactly nothing on the shipped presets (all of which fix kunit = 1).
    const asrc = (state.alt || {})[K_GRADSB] || "";
    const kua = 1 / kconst(asrc, "INVKU");
    const gsA = new Float32Array(4 * nS);
    for (let i = 0; i < nS; i++) { const k = knAt(i) * kua; gsA[4 * i] = 1; gsA[4 * i + 2] = k * k; }
    const ga = runGrads(M, asrc, N, fs2, gsA, BAND);
    let em3 = 0;
    for (let i = 0; i < nS; i++)
      em3 = Math.max(em3, Math.abs(ga[2 * i + 1] - bandFacJs(knAt(i), BAND[0], BAND[1], E)));
    ok("  ... and the band it applies is in k/kunit and not in k (the same sweep at kunit = 1/2)",
       kua === 0.5 && em3 < 1e-6,
       "kunit = " + kua + ", max |delta| = " + em3.toExponential(2) + " over " + nS + " samples");
  }

  // ---- (b) the gradients, on the page's own grid --------------------------
  {
    const st = synthState(N.nm, 777);
    let worst = 0, n = 0;
    for (const band of [[0, 0], [2, 5], [1.5, 2.5]]) {
      const g = runGrads(M, src, N, st, gA, band);
      for (let m = 0; m < N.nm; m++) {
        const mp = m % N.nmp, kx = gA[4 * mp], ky = gA[4 * mp + 1], ksq = gA[4 * mp + 2];
        const bf = bandFacJs(Math.sqrt(ksq) / ku, band[0], band[1], E);
        const ph = [bf * st[2 * m], bf * st[2 * m + 1]];
        const ps = [bf * st[2 * (N.nm + m)], bf * st[2 * (N.nm + m) + 1]];
        const vo = [-ksq * ph[0], -ksq * ph[1]], jp = [-ksq * ps[0], -ksq * ps[1]];
        const lanes = [[kx, ph], [ky, ph], [kx, ps], [ky, ps],
                       [kx, vo], [ky, vo], [kx, jp], [ky, jp]];
        for (let L = 0; L < 8; L++) {
          const k = lanes[L][0], f = lanes[L][1], want = [-k * f[1], k * f[0]];
          for (let c = 0; c < 2; c++) {
            const got = g[2 * (L * N.nm + m) + c];
            worst = Math.max(worst, Math.abs(got - want[c]) / Math.max(1e-7, Math.abs(want[c])));
            n++;
          }
        }
      }
    }
    ok("(b) the eight band-passed gradient lanes match the fp64 mirror", worst < 1e-5,
       "max relative error " + worst.toExponential(2) + " over " + n + " values, 3 bands");
    const a = runGrads(M, src1, N, st, gA, [0, 0]), b = runGrads(M, src, N, st, gA, [0, 0]);
    const x = bits(a), y = bits(b);
    let dbad = 0;
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) dbad++;
    ok("  ... and with both ends off it is prepGrads BIT for BIT (1.0 * v is v)",
       dbad === 0, dbad ? dbad + " words differ" : 16 * N.nm + " words identical");
  }

  // ---- (c) the row: gradient -> real space -> samples -> periodogram -------
  {
    const nl = G0.side * G0.side, nz = G.nz;
    const st = new Float32Array(4 * N.nm);
    // a handful of analytic modes placed by hand across k_perp, so every band has content
    for (const [ix, iy, iz, re, im] of [[1, 1, 1, 0.7, -0.3], [3, 0, 2, 0.4, 0.5],
                                        [2, 4, 1, -0.6, 0.2], [5, 2, 3, 0.25, 0.15],
                                        [0, 6, 2, 0.3, -0.45]]) {
      const m = (iz * G.nkx + ix) * G.nky + iy;
      st[2 * m] = re; st[2 * m + 1] = im;
    }
    const bands = [[0, 0], [2, 6], [3, 4]];
    let worst = 0, peak = 0;
    for (const band of bands) {
      const g = runGrads(M, src, N, st, gA, band);
      const dx = toReal(m => [g[2 * m], g[2 * m + 1]], G);                     // d_x phi
      const dy = toReal(m => [g[2 * (N.nm + m)], g[2 * (N.nm + m) + 1]], G);   // d_y phi
      const smp = new Float64Array(4 * nl * nz);
      for (let i = 0; i < nl; i++) {
        const gx = (Math.floor(i / G0.side) + 0.5) / G0.side * G.nx;
        const gy = ((i % G0.side) + 0.5) / G0.side * G.ny;
        for (let j = 0; j < nz; j++) {
          smp[4 * (i * nz + j)] = -bilin(dy, G, j, gx, gy);        // u_x = -d_y phi
          smp[4 * (i * nz + j) + 1] = bilin(dx, G, j, gx, gy);     // u_y =  d_x phi
        }
      }
      const app = C.flSpectrum(Float32Array.from(smp), nl, nz);
      const ref = rowJs(smp, nl, nz);
      let hi = 0;
      for (let i = 0; i < ref.length; i++) hi = Math.max(hi, Math.abs(ref[i]));
      peak = Math.max(peak, hi);
      for (let i = 0; i < ref.length; i++)
        worst = Math.max(worst, Math.abs(app[i] - ref[i]) / Math.max(hi, 1e-30));
    }
    ok("(c) the ROW the sweep builds -- factor x gradient x periodogram -- matches fp64",
       worst < 1e-5 && peak > 0, "max error " + worst.toExponential(2) + " of the row peak, "
       + bands.length + " bands x " + nl + " lines x " + nz + " planes");
  }

  // ---- (d) the coordinate kernel, executed one mode at a time -------------
  {
    const nzb = N.nzb, nz = G.nz, INVN2 = kconst(psrc, "INVN2");
    const bandSet = [[0, 0], [2, 5], [3, 3.5], [1, 2]];
    let worst = 0, anchor = 0, live = 0;
    for (const kn of [0.5, 1, 2, 2.75, 4, 6]) {
      // one live perpendicular mode, at mp = 0 (see the header note), its k_perp handed in
      // through a synthetic gridA; everything else on the grid is exactly zero
      const gs = new Float32Array(4 * N.nmp), gs2 = new Float32Array(4 * N.nmp);
      gs[2] = (kn * ku) * (kn * ku); gs2[3] = 2;
      const f = new Float32Array(4 * N.nm);
      let s = 4242;
      for (let iz = 0; iz < nz; iz++) for (const off of [0, 2 * N.nm]) {
        s = (s * 48271) % 2147483647; f[off + 2 * (iz * N.nmp)] = s / 2147483647 - 0.5;
        s = (s * 48271) % 2147483647; f[off + 2 * (iz * N.nmp) + 1] = s / 2147483647 - 0.5;
      }
      // all bands in ONE dispatch, as the app does it: a 2D grid over (|kz| bin, band)
      const bnd = new Float32Array(4 * bandSet.length);
      bandSet.forEach((b, j) => { bnd[4 * j] = b[0]; bnd[4 * j + 1] = b[1]; });
      const bins = new Float32Array(bandSet.length * 3 * nzb);
      new M.WgslExec(new M.WgslParser().parse(psrc)).dispatchWorkgroups("main",
        [nzb, bandSet.length, 1], { 0: { 0: f, 1: gs, 2: gs2, 3: bins, 4: bnd } });
      const bins1 = new Float32Array(3 * nzb);
      new M.WgslExec(new M.WgslParser().parse(cur[K_PAR])).dispatchWorkgroups("main",
        [nzb, 1, 1], { 0: { 0: f, 1: gs, 2: gs2, 3: bins1 } });
      bandSet.forEach((band, j) => {
        const bf = bandFacJs(kn, band[0], band[1], E), w = 0.5 * gs[2] * gs2[3] * INVN2 * bf * bf;
        for (let b = 1; b <= nzb; b++) {
          const idx = [b * N.nmp];
          if (b !== nz - b) idx.push((nz - b) * N.nmp);
          let eu = 0, eb = 0, hc = 0;
          for (const m1 of idx) {
            eu += f[2 * m1] * f[2 * m1] + f[2 * m1 + 1] * f[2 * m1 + 1];
            eb += f[2 * (N.nm + m1)] * f[2 * (N.nm + m1)] + f[2 * (N.nm + m1) + 1] * f[2 * (N.nm + m1) + 1];
            hc += 2 * (f[2 * m1] * f[2 * (N.nm + m1)] + f[2 * m1 + 1] * f[2 * (N.nm + m1) + 1]);
          }
          const want = [w * eu, w * eb, w * hc];
          for (let L = 0; L < 3; L++) {
            const got = bins[j * 3 * nzb + L * nzb + b - 1];
            if (Math.abs(want[L]) > 1e-14) {
              worst = Math.max(worst, Math.abs(got - want[L]) / Math.abs(want[L]));
              live++;
            }
          }
        }
      });
      for (let i = 0; i < 3 * nzb; i++) if (bins[i] !== bins1[i]) anchor++;
    }
    ok("(d) the coordinate kernel's rows match the fp64 mirror (weight, lanes, +-kz fold)",
       worst < 1e-5 && live > 20,
       "max relative error " + worst.toExponential(2) + " over " + live + " live bins");
    ok("  ... and its band-[0,0] row IS the 1D specPar's own bins, BIT for BIT (Stage B's anchor)",
       anchor === 0, anchor ? anchor + " bins differ" : "6 k_perp x 3 lanes x " + nzb + " bins");
  }
}

// ===========================================================================
// 5. ridge recovery, the band set, and the coordinate panel's anchor
// ===========================================================================
// A synthetic snapshot with a KNOWN k_par(k_perp), pushed through the real press: the real
// `gen2dBands` window off a real-shaped perpendicular spectrum, the real `gen2dSpec`
// sweep, the real `flSpectrum` on samples built to carry the law, and the real
// `gen2dPanel` / `gen2dRidge` reading the plot back out. Only the four solver READBACKS
// are synthetic -- there is no GPU here to make them.
//
// The two panels carry DIFFERENT laws on purpose: field line k_par = A k_perp^(2/3) all
// the way, coordinate the same law CAPPED (the Cho-Vishniac flattening the card exists to
// show). Recovering each from its own panel is what says the `gp` select is choosing
// between two datasets and not relabelling one.
//
// A is not a constant here: it is solved for inside the page so the law spans the
// snapshot's OWN resolved range (bands from `gen2dBands`, bins from `nzb`, units from
// `parKfac`), whatever preset the page happens to boot at. The law is then handed back
// out and the assertions are made against it.
// `rows` false leaves `readGenBand` / `readGen2D` ALONE -- the sweep then runs its real
// encode / dispatch / readback path (on a stub device that returns zeros, so the rows are
// empty, which is all leg 3 wants: it is tracing what the press touches, not what it drew).
const SYNTH = `function(rows){
  const sv = solver, nzb = sv.nzb, nz = sv.g.nz, nb = sv.nb, pk = sv.parKfac;
  // a perpendicular spectrum with a real inertial range and a real dissipation knee, so
  // gen2dBands cuts a real window out of it
  const perp = new Float32Array(3 * nb), kd = 0.5 * nb;
  for (let b = 1; b < nb; b++) {
    const e = Math.pow(b, -5/3) * Math.exp(-Math.pow(b / kd, 4));
    perp[b] = 0.6 * e; perp[nb + b] = 0.4 * e; perp[2 * nb + b] = 0.1 * e;
  }
  const par = new Float32Array(3 * nzb);
  for (let i = 0; i < nzb; i++) { par[i] = Math.pow(i + 1, -2); par[nzb + i] = 0.5 * Math.pow(i + 1, -2); }
  // the law, in |kz| BINS: k_par = A * k_perp^(2/3), with A chosen so the ridge crosses
  // the resolved range of THIS grid (2 bins at the lowest band, 0.7*nzb at the highest)
  const bands = gen2dBands(nb, sv.p.fshell, perp);
  const k0 = bands.length ? bands[0].kc : 1, k1 = bands.length ? bands[bands.length - 1].kc : 2;
  const p23 = k => Math.pow(k, 2/3);
  const A = Math.max(0.7 * nzb - 2, 1) / Math.max(p23(k1) - p23(k0), 1e-9);
  const B = 2 - A * p23(k0);
  const kOf = kc => A * p23(kc) + B;                    // in bins
  const flat = 0.5 * (2 + 0.7 * nzb);                   // where the coordinate law saturates
  sv.readSpectrum = async () => ({ perp: perp, par: par, parKfac: pk });
  sv.readStats = async () => Float32Array.from([0, 3.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  // the field-line half: along-line samples whose |kz| content is a one-bin-wide packet
  // about the (continuous) law, so the ROW comes out of the app's own Hann periodogram
  if (!rows) return { A: A, B: B, flat: flat, nzb: nzb, nb: nb, pk: pk, nz: nz,
                      kc: bands.map(b => b.kc) };
  // every call the sweep issues is counted here: since the render audit removed the extra
  // unbanded pass, "one per band and no more" is a claim worth pinning at the source
  globalThis.GBCALL = { n: 0, unbanded: 0 };
  sv.readGenBand = async band => {
    const nl = GEN_SIDE * GEN_SIDE, out = new Float32Array(nl * nz * 4);
    globalThis.GBCALL.n++;
    if (!(band.lo || band.hi)) globalThis.GBCALL.unbanded++;
    const cs = (band.lo || band.hi) ? [kOf(band.kc)] : bands.map(b => kOf(b.kc));
    for (let j = 0; j < nz; j++) {
      let v = 0;
      for (let b = 1; b <= nzb; b++) {
        let a = 0;
        for (const kc of cs) a += Math.exp(-Math.pow(b - kc, 2));
        if (a > 1e-6) v += a * Math.cos(2 * Math.PI * b * j / nz);
      }
      for (let l = 0; l < nl; l++) out[4 * (l * nz + j)] = v;
    }
    return out;
  };
  // ... and the coordinate half: the same packet on a CAPPED law, written straight into
  // the [E_u | E_b | H_c] bin stack the kernel would have produced
  sv.readGen2D = async bands => bands.map(band => {
    const r = new Float32Array(3 * nzb), kc = Math.min(kOf(band.kc), flat);
    for (let b = 1; b <= nzb; b++) {
      const a = Math.exp(-Math.pow(b - kc, 2));
      r[b - 1] = 0.6 * a; r[nzb + b - 1] = 0.4 * a; r[2 * nzb + b - 1] = 0.1 * a;
    }
    return r;
  });
  return { A: A, B: B, flat: flat, nzb: nzb, nb: nb, pk: pk, nz: nz,
           kc: bands.map(b => b.kc) }; }`;
const installSynth = (env, rows) => env.run(SYNTH, rows !== false);
// press the card's generate button and wait the sweep out (it yields between bands)
async function pressGenerate(env) {
  env.run(`function(){
    if (!cards.chart.some(c => c.type() === "gen2d")) { addChartCard("gen2d"); cardsSync(); }
    setRunning(true);
    const c = cards.chart.filter(x => x.type() === "gen2d")[0];
    c.optEls.filter(s => s.__optId === "gen")[0].onclick();
    return true; }`);
  for (let i = 0; i < 2000 && env.run("function(){ return gen2d.busy; }"); i++)
    await new Promise(r => setTimeout(r, 0));
  return env.run("function(){ return gen2d.busy; }") === false;
}
// The ridge of one band's row: the k_par of its largest cell (0 for an empty row). This
// used to live in common.js, where nothing drawn ever called it -- the reference slopes
// anchor on gen2dTop, not on the argmax -- so it was app code with only this file for a
// consumer (render audit, 2026-08-12). It is the checker's MEASURING INSTRUMENT for the
// planted k_par(k_perp) law, and what the recovery leg puts under test is the sweep that
// produced the row, so owning it here weakens nothing.
const ridgeOf = pts => {
  let k = 0, v = 0;
  for (let i = 0; i < pts.length; i += 2) if (pts[i + 1] > v) { v = pts[i + 1]; k = pts[i]; }
  return k;
};
const ridgesOf = P => (P ? Object.assign(P, { ridge: P.rows.map(ridgeOf) }) : P);
const PANEL_ = `function(gp, gq){
  const G = gen2dPanel({ gp: gp, gq: gq, gov: "off" });
  if (!G) return null;
  return { n: G.rows.length, hi: G.hi, floor: G.floor, nzb: G.nzb,
           rows: G.rows.map(r => Array.prototype.slice.call(r)),
           kc: G.d.bands.map(b => b.kc),
           t: G.d.t, pk: G.d.parKfac, npts: G.rows.map(r => r.length / 2) }; }`;
const runPanel = (env, ...a) => ridgesOf(env.run(PANEL_, ...a));

async function legRidge(state) {
  // ---- the band set itself: the anisotropy card's window, in octaves -------
  {
    const nb = 512, fsh = [1, 3];
    const perp = new Float32Array(3 * nb), kd = 200;
    for (let b = 1; b < nb; b++) perp[b] = Math.pow(b, -5 / 3) * Math.exp(-Math.pow(b / kd, 4));
    const bs = C.gen2dBands(nb, fsh, perp);
    const knee = C.specKnee([[C._anisoLeg(perp, nb, 0, 1, C.ANISO_LANES.tot)]],
                            C._anisoPeak(C._anisoLeg(perp, nb, 0, 1, C.ANISO_LANES.tot)));
    ok("GEN_NBAND is the plan's 8-12 bands, GEN_BAND_W one octave",
       C.GEN_NBAND >= 8 && C.GEN_NBAND <= 12 && C.GEN_BAND_W === 2,
       C.GEN_NBAND + " bands x " + C.GEN_BAND_W);
    ok("gen2dBands log-spaces them between the fit anchor and the SHARED knee",
       bs.length === C.GEN_NBAND && bs.every(b => b.lo >= C.fitKA(nb, fsh) - 1e-9 && b.hi <= knee + 1e-9) &&
       bs.every((b, j) => j < 2 || rel(Math.log(b.kc / bs[j - 1].kc), Math.log(bs[1].kc / bs[0].kc)) < 1e-9),
       bs.length + " bands, k_perp " + bs[0].kc.toFixed(2) + ".." + bs[bs.length - 1].kc.toFixed(2)
       + ", knee " + knee.toFixed(1));
    ok("  ... each an octave wide about its centre, clipped to the window (they OVERLAP)",
       bs.every(b => b.hi / b.lo <= C.GEN_BAND_W + 1e-9) &&
       bs.slice(1, -1).every(b => rel(b.hi / b.lo, C.GEN_BAND_W) < 1e-6) &&
       bs.some((b, j) => j > 0 && b.lo < bs[j - 1].hi),
       "hi/lo = " + bs.map(b => (b.hi / b.lo).toFixed(2)).join(" "));
    ok("  ... no resolved range at all -> NO bands (the card draws nothing, not one row)",
       C.gen2dBands(2, [1, 3], null).length === 0 && C.gen2dBands(1, [1, 3], null).length === 0);
    ok("  ... a window under one octave -> exactly ONE band, at its geometric centre",
       (() => { const b = C.gen2dBands(4, [1, 3], null);
                return b.length === 1 && rel(b[0].kc, Math.sqrt(3 * 4)) < 1e-9; })());
    ok("  ... and with no spectrum in hand the high end is the dealias cut nb",
       (() => { const b = C.gen2dBands(64, [1, 3], null);
                return b.length === C.GEN_NBAND && rel(b[b.length - 1].hi, 64) < 1e-9; })());
  }

  // ---- the press, and the ridge it puts on the plot ------------------------
  const env = state.env3d || (state.env3d = await boot(dir, "rmhd3d.html"));
  const S = installSynth(env);
  const done = await pressGenerate(env);
  ok("the press completes and leaves a snapshot", done && env.run("function(){ return !!gen2d.data; }"));
  const d = env.run("function(){ const d = gen2d.data; return d && { nb: d.nb, nzb: d.nzb,"
    + " t: d.t, nl: d.nl, pk: d.parKfac, kc: d.bands.map(b => b.kc),"
    + " lo: d.bands.map(b => b.lo), hi: d.bands.map(b => b.hi), rows: d.rows.length,"
    + " crows: d.crows.length, fshell: d.fshell }; }");
  ok("  ... carrying one field-line row and one coordinate row per band, and its own grid",
     d && d.kc.length > 1 && d.kc.length <= C.GEN_NBAND && d.rows === d.kc.length &&
     d.crows === d.kc.length && d.nzb === S.nzb && d.nb === S.nb,
     d ? d.kc.length + " bands, nb = " + d.nb + ", nzb = " + d.nzb + ", " + d.nl + " lines" : "no data");
  const law = kc => S.A * Math.pow(kc, 2 / 3) + S.B;          // in |k_par| bins
  const P = {};
  for (const [gp, cap, nm] of [["fl", Infinity, "field line"], ["z", S.flat, "coordinate"]]) {
    const p = P[gp] = runPanel(env, gp, "tot");
    if (!p) { ok("the " + nm + " panel has rows", false); continue; }
    let worstBin = 0, checked = 0, off = 0;
    for (let j = 0; j < p.n; j++) {
      const want = Math.min(law(p.kc[j]), cap);               // bins
      if (!(want >= 1.5 && want <= p.nzb - 0.5)) { off++; continue; }   // the resolved range
      worstBin = Math.max(worstBin, Math.abs(p.ridge[j] / p.pk - want));
      checked++;
    }
    ok(nm + " panel: the argmax ridge lands within ONE |k_par| bin of the known law",
       checked >= 3 && worstBin <= 1 + 1e-9,
       checked + " bands in the resolved range (" + off + " outside), worst |delta| = "
       + worstBin.toFixed(3) + " bins");
    ok("  ... and it is monotone in k_perp: the ridge bends, it does not wander",
       p.ridge.every((r, j) => j === 0 || r >= p.ridge[j - 1] - 1e-9),
       p.ridge.map(r => (r / p.pk).toFixed(1)).join(" "));
  }
  // the contrast the card exists for: one snapshot, two frames, and the coordinate ridge
  // FLATTENS where the field-line one keeps climbing
  ok("the two panels really are two datasets: coordinate flattens, field line climbs",
     P.fl && P.z && P.fl.ridge[P.fl.n - 1] > P.z.ridge[P.z.n - 1] + 0.5 * P.fl.pk &&
     Math.abs(P.fl.ridge[0] - P.z.ridge[0]) <= 1.5 * P.fl.pk,
     P.fl && P.z ? "fl " + (P.fl.ridge[0] / P.fl.pk).toFixed(1) + "->"
                   + (P.fl.ridge[P.fl.n - 1] / P.fl.pk).toFixed(1) + " bins, z "
                   + (P.z.ridge[0] / P.z.pk).toFixed(1) + "->"
                   + (P.z.ridge[P.z.n - 1] / P.z.pk).toFixed(1) : "no panel");
  // the lane select is the shared E+- algebra: the coordinate rows carry E_u : E_b : H_c
  // = 0.6 : 0.4 : 0.1, so tot = 1.0, zp = 1.1, zm = 0.9 of the same peak
  // The sweep's extra UNBANDED row is GONE (render audit, 2026-08-12). It existed for ONE
  // consumer -- the k∥B leg of the measured overlay curves -- and Alfred dropped those in
  // his second round, which left it computed, shipped and read by nobody but the assertion
  // that used to stand here. What is pinned now is the shape the sweep was cut back to:
  // the snapshot carries NO parFL, and the press makes exactly one readGenBand call per
  // band, with the progress total to match. Both halves matter -- dropping the field but
  // leaving the pass would save nothing, and dropping the pass but leaving `total` would
  // make the button's own count wrong.
  const OV = env.run(`function(){
    const d = gen2d.data, C = globalThis.GBCALL || {};
    return { has: d.parFL !== undefined, nb: d.bands.length, rows: d.rows.length,
             calls: C.n, unbanded: C.unbanded, done: gen2d.done, total: gen2d.total }; }`);
  ok("the sweep takes ONE readGenBand per band and ships no unbanded row",
     !OV.has && OV.rows === OV.nb && OV.calls === OV.nb && OV.unbanded === 0,
     OV.nb + " bands, " + OV.rows + " rows, " + OV.calls + " calls ("
     + OV.unbanded + " unbanded), parFL " + (OV.has ? "STILL PRESENT" : "gone"));
  ok("  ... and the button's progress total is that band count, so done/total ends at 1",
     OV.total === OV.nb && OV.done === OV.nb,
     "finished at " + OV.done + "/" + OV.total);
  const L = ["tot", "zp", "zm"].map(q => runPanel(env, "z", q));
  ok("the lane select is ANISO_LANES' own E+- = E_u + E_b +- H_c",
     L.every(p => p && p.n === P.z.n) && rel(L[1].hi, 1.1 * L[0].hi) < 1e-5 &&
     rel(L[2].hi, 0.9 * L[0].hi) < 1e-5,
     L.map(p => p && p.hi.toExponential(3)).join(" | "));

  // ---- the coordinate panel's anchor, as WIRING ---------------------------
  // The kernel half of "readGen2D([{lo:0,hi:0}])[0] == readSpectrum().par" is leg 4(d),
  // executed and bitwise. What is left is that the page runs that kernel on the same
  // inputs as the 1D pass: a fresh solver is built with a bind-group recorder installed,
  // so the sweep's bind group and the 1D parallel spectrum's are compared as BUFFERS.
  const W = await env.run(`function(){
    const d = solver.device, ob = d.createBindGroup.bind(d), oe = d.createCommandEncoder.bind(d);
    const rec = [];
    d.createBindGroup = o => { rec.push(o.entries.map(e => e.resource.buffer)); return ob(o); };
    rebuild();                                  // a solver whose readbacks are the REAL ones
    const disp = [];
    d.createCommandEncoder = () => {
      const e = oe(), bp = e.beginComputePass.bind(e);
      e.beginComputePass = () => {
        const p = bp(), od = p.dispatchWorkgroups.bind(p);
        p.dispatchWorkgroups = (x, y, z) => { disp.push([x, y === undefined ? 1 : y]); return od(x, y, z); };
        return p;
      };
      return e;
    };
    return solver.readGen2D([{ lo: 0, hi: 0 }, { lo: 2, hi: 4 }]).then(rows => {
      d.createBindGroup = ob; d.createCommandEncoder = oe;
      const sv = solver;
      const nameOf = b => { for (const k of Object.keys(sv.buf)) if (sv.buf[k] === b) return k; return "?"; };
      return { groups: rec.map(g => g.map(nameOf)), disp: disp, nrow: rows.length,
               len: rows.map(r => r.length), nzb: sv.nzb }; }); }`);
  const g2d = (W.groups || []).filter(g => g.indexOf("gen2d") >= 0)[0] || [];
  const g1d = (W.groups || []).filter(g => g.indexOf("specParBins") >= 0)[0] || [];
  ok("the coordinate pass reads the 1D parallel spectrum's own inputs, in that order",
     g1d.length === 4 && g2d.length === 5 &&
     g2d.slice(0, 3).join(",") === g1d.slice(0, 3).join(",") &&
     g2d[0] === "fields" && g2d[4] === "genBand",
     "1D [" + g1d.join(",") + "]  vs  sweep [" + g2d.join(",") + "]");
  ok("  ... dispatches (nzb, nbands) in ONE pass and hands back 3*nzb floats per band",
     W.disp.length === 1 && W.disp[0][0] === W.nzb && W.disp[0][1] === 2 &&
     W.nrow === 2 && W.len.every(l => l === 3 * W.nzb),
     JSON.stringify(W.disp) + " -> " + W.nrow + " rows x " + W.len[0] + " floats");
  ok("the 3D page boots clean through the whole press", env.fails.length === 0,
     env.fails.join(" | "));
}

// ===========================================================================
// 6. the press: pause / resume choreography and plot persistence
// ===========================================================================
// Alfred's model, checked as a sequence: press while RUNNING -> the page ends PAUSED with
// the plot on it; Run -> the plot does not move; an IC reset and a solver rebuild -> still
// there (it carries its own t, nb, nzb and bands, so it never consults the live grid); a
// second press -> replaced, with the new t.
async function legChoreography(state) {
  const env2 = await boot(dir, "rmhd2d.html");
  ok("the 2D app never offers the card (no k_par to bin)",
     env2.run("function(){ return chartTypeKeys(); }").indexOf("gen2d") < 0);
  ok("  ... through the same `avail` the anisotropy card uses",
     C.CHART_TYPES.gen2d.avail({ zslice: true }) === true && !C.CHART_TYPES.gen2d.avail({}));
  ok("  ... and it declares no readback source at all: its data is a press",
     !C.CHART_TYPES.gen2d.src && typeof C.CHART_TYPES.gen2d.bar === "function" &&
     typeof C.CHART_TYPES.gen2d.hint === "function" &&
     C.CHART_TYPES.gen2d.hint({}).length > 80);

  const env = await boot(dir, "rmhd3d.html");
  const card = env.run(`function(){ const c = addChartCard("gen2d"); cardsSync(); return c; }`);
  const ids = env.run("function(c){ return c.optEls.map(s => s.__optId).join(','); }", card);
  ok("a gen2d card builds with its five controls (generate / panel / lane / colour / theory)",
     card && card.optEls.length === 5 && ids === "gen,gp,gq,gc,gov", ids);
  ok("  ... and a colorbar, because on this chart the plotted quantity IS the colour",
     env.run("function(c){ return !!c.foot && !!c.barT && c.barT.length === 3; }", card));
  // it was "overlays" while it also drew the anisotropy card's measured curves; with those
  // gone (second round) it toggles the three theory slopes alone, and says so
  ok("  ... the theory-slopes box is a checkbox reading on/off, starts OFF, and is no longer "
     + "called `overlays`",
     env.run("function(c){ const o = c.optVals();"
             + " const s = c.optEls.filter(s => s.__optId === 'gov')[0];"
             + " let t = ''; for (const k of s.children || []) if (k.kind === '#text') t += k.textContent;"
             + " return o.gov === 'off' && !!s.__optChk && t.indexOf('theory') >= 0"
             + " && t.indexOf('overlay') < 0; }", card));
  const st = () => env.run("function(){ const c = cards.chart.filter(x => x.type() === 'gen2d')[0];"
    + " const b = c.optEls.filter(s => s.__optId === 'gen')[0];"
    + " return { running: running, busy: gen2d.busy, dis: !!b.disabled, has: !!gen2d.data,"
    + " t: gen2d.data ? gen2d.data.t : null, bars: c.barT.map(x => x.innerHTML) }; }");
  // ---- press while running -------------------------------------------------
  installSynth(env);
  env.run("function(){ setRunning(true); }");
  const mid = env.run(`function(){
    const c = cards.chart.filter(x => x.type() === "gen2d")[0];
    const b = c.optEls.filter(s => s.__optId === "gen")[0];
    b.onclick();
    return { running: running, busy: gen2d.busy, dis: !!b.disabled, lab: el("btnRun").textContent }; }`);
  ok("the press PAUSES the run before its first await, and disables the button",
     mid.running === false && mid.busy === true && mid.dis === true && mid.lab === "Run",
     JSON.stringify(mid));
  for (let i = 0; i < 2000 && env.run("function(){ return gen2d.busy; }"); i++)
    await new Promise(r => setTimeout(r, 0));
  const a = st();
  ok("  ... and when it finishes the page is STILL paused, with the plot on it",
     a.running === false && a.busy === false && a.dis === false && a.has === true && a.t === 3.25,
     JSON.stringify({ running: a.running, busy: a.busy, t: a.t }));
  const leg = env.run(`function(){
    const c = cards.chart.filter(x => x.type() === "gen2d")[0], cx = c.cx, of = cx.fillText;
    const txt = [];
    cx.fillText = (t, x, y) => { txt.push(String(t)); return of(t, x, y); };
    c.draw(null);
    cx.fillText = of;
    return txt.join(" | "); }`);
  ok("  ... and the legend says which frame it is and the t it was taken at",
     /field line k∥B — generated @ t = 3\.25/.test(leg), leg.split(" | ").pop());
  ok("  ... with the colorbar labelled off the same panel (log range, two distinct ends)",
     a.bars.length === 3 && a.bars.every(s => s && s.length) && a.bars[0] !== a.bars[2],
     a.bars.join(" .. "));
  // ---- resume: the plot does not move --------------------------------------
  const before = runPanel(env, "fl", "tot");
  env.run("function(){ setRunning(true); }");
  ok("Run resumes the sim and does NOT clear or move the plot",
     env.run("function(){ return running; }") === true &&
     JSON.stringify(runPanel(env, "fl", "tot")) === JSON.stringify(before));
  // ---- an IC reset and a full rebuild --------------------------------------
  env.run("function(){ chartsReset(); }");
  ok("an IC reset (chartsReset) leaves it standing -- it is a record of a moment",
     JSON.stringify(runPanel(env, "fl", "tot")) === JSON.stringify(before));
  env.run("function(){ rebuild(); }");
  ok("a solver REBUILD leaves it standing too (the snapshot carries its own grid)",
     JSON.stringify(runPanel(env, "fl", "tot")) === JSON.stringify(before),
     "nb / nzb / bands / parKfac all ride in the data");
  ok("  ... and the button comes back live on the new solver",
     env.run("function(){ const c = cards.chart.filter(x => x.type() === 'gen2d')[0];"
             + " return !c.optEls.filter(s => s.__optId === 'gen')[0].disabled; }"));
  // ---- a second press replaces it -----------------------------------------
  installSynth(env);
  env.run("function(){ solver.readStats = async () => Float32Array.from("
          + "[0, 9.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); }");
  const done2 = await pressGenerate(env);
  const b2 = st();
  ok("a second press REPLACES the plot (the pin convention: cleared only by another one)",
     done2 && b2.has === true && b2.t === 9.5, "t = " + b2.t);
  // ---- ... but a press on a DEAD field does not ----------------------------
  // gen2dBands still cuts a band set out of a silent ⊥ spectrum (with no knee in view its
  // high end falls back to the dealias cut), so the sweep runs to completion and hands back
  // a perfectly well-formed object of zeros -- which gen2dPanel would then refuse, leaving
  // the card reading "press generate" where a good plot used to be. gen2dLive sends that
  // down the null path instead: nothing generated, previous plot kept, and a line saying so.
  env.run(`function(){
    const sv = solver, nzb = sv.nzb, nz = sv.g.nz, nb = sv.nb;
    sv.readSpectrum = async () => ({ perp: new Float32Array(3 * nb),
                                     par: new Float32Array(3 * nzb), parKfac: sv.parKfac });
    sv.readStats = async () => Float32Array.from([0, 12.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    sv.readGenBand = async () => new Float32Array(GEN_SIDE * GEN_SIDE * nz * 4);
    sv.readGen2D = async bs => bs.map(() => new Float32Array(3 * nzb)); }`);
  const done3 = await pressGenerate(env);
  const b3 = st();
  ok("  ... and a press on a DEAD field does NOT: an all-zero sweep is the null path",
     done3 && b3.has === true && b3.t === 9.5,
     "t = " + b3.t + " (the dead press was taken at 12.5)");
  ok("  ... which the band set alone would not have caught (it still cuts a window)",
     env.run("function(){ return gen2dBands(solver.nb, solver.p.fshell,"
             + " new Float32Array(3 * solver.nb)).length; }") > 1);
  // ---- and with no solver at all, the button refuses -----------------------
  // The no-WebGPU boot and the moment between a rebuild's retire and its replacement are
  // both "no solver": the button is DISABLED there. Pressed anyway (which a browser would
  // not do, but a stale handler could), the sweep must decline rather than throw -- and
  // must leave the previous plot exactly where it was, since a null return is not news.
  const t0 = env.run("function(){ return gen2d.data.t; }");
  const dis0 = env.run(`function(){
    const c = cards.chart.filter(x => x.type() === "gen2d")[0];
    const b = c.optEls.filter(s => s.__optId === "gen")[0];
    globalThis.SV = solver; solver = null; c._optSync();
    const dis = !!b.disabled;
    b.onclick();
    return dis; }`);
  for (let i = 0; i < 200 && env.run("function(){ return gen2d.busy; }"); i++)
    await new Promise(r => setTimeout(r, 0));
  const after = env.run("function(){ solver = globalThis.SV;"
    + " cards.chart.filter(c => c.type() === 'gen2d')[0]._optSync();"
    + " return { t: gen2d.data ? gen2d.data.t : null, busy: gen2d.busy }; }");
  ok("with no solver the button is disabled, and a forced press declines quietly",
     dis0 === true && after.busy === false, "disabled = " + dis0);
  ok("  ... leaving the plot it could not replace exactly where it was",
     after.t === t0, "t = " + after.t);
  ok("retyping the card away and back takes its controls and its colorbar with it",
     env.run(`function(){
       const c = cards.chart.filter(x => x.type() === "gen2d")[0];
       c.selType.value = "energy"; c.selType.onchange();
       const away = c.optEls.length + ":" + !!c.foot;
       c.selType.value = "gen2d"; c.selType.onchange();
       return away === "1:false" && c.optEls.length === 5 && !!c.foot && !!gen2d.data; }`));
  ok("both boots raised no failures", env.fails.length === 0 && env2.fails.length === 0,
     env.fails.concat(env2.fails).join(" | "));
}

// ===========================================================================
// 7. the plot itself: axis limits, the k∥ floor, the three anchors, the colour scale
// ===========================================================================
// Alfred's first AND second feedback rounds on the card, gated the way the rest of it is:
// the second round drops the two measured overlay curves, gives the legend its y headroom,
// puts the three theory slopes on one legend line, squares the plot area, moves the anchor
// onto the boundary cell's TOP EDGE and puts a noise margin on the boundary itself. The
// frame is
// read as DATA off the app's own `gen2dFrame` (the specCurves seam), and then every claim
// is made a second time in PIXELS off a recording of the real `drawGen2D` on the real card,
// so a correct frame drawn wrongly still fails. The only arithmetic this script owns is the
// affine log -> pixel mirror `MAP` below, which is two lines and is exactly what the check
// is entitled to re-derive.
//
// One thing to be clear about, because it is what the whole leg turns on: the reference
// slopes are anchored on the upper BOUNDARY of the filled region (the largest k∥ above the
// panel's floor, per band), not on the argmax ridge. So a line must lie at or above the
// boundary at EVERY band and touch it at one -- that pair of statements is the anchor rule,
// and either half alone would pass on a line that is merely somewhere near the data.
//
// Everything drawGen2D puts on the canvas, recorded: cells (fillRect + the fill colour),
// stroked polylines (begin / move / line / stroke, with colour and dash), and text (with
// the alignment and the width the app's OWN measureText gives it, so the label-collision
// check measures what the app measured). The card is driven through its real controls and
// its real `draw`, so the option wiring is under test too.
const SETOPT = `function(vals){
  const c = cards.chart.filter(x => x.type() === "gen2d")[0];
  for (const s of c.optEls) {
    const v = vals[s.__optId];
    if (v === undefined) continue;
    if (s.__optChk) s.__optChk.checked = (v === "on"); else s.value = v;
  }
  c._optSync();
  return c.optVals(); }`;
const DRAWREC = `function(vals){
  const c = cards.chart.filter(x => x.type() === "gen2d")[0], cx = c.cx;
  const rec = { cells: [], lines: [], text: [] };
  const keep = {};
  for (const m of ["fillRect", "beginPath", "moveTo", "lineTo", "stroke", "setLineDash", "fillText"])
    keep[m] = cx[m];
  let dash = [], poly = null;
  cx.setLineDash = d => { dash = (d || []).slice(); return keep.setLineDash.call(cx, d); };
  cx.beginPath = () => { poly = []; return keep.beginPath.call(cx); };
  cx.moveTo = (x, y) => { if (poly) poly.push(x, y); return keep.moveTo.call(cx, x, y); };
  cx.lineTo = (x, y) => { if (poly) poly.push(x, y); return keep.lineTo.call(cx, x, y); };
  cx.stroke = () => {
    if (poly && poly.length) rec.lines.push({ p: poly.slice(), col: String(cx.strokeStyle),
                                              dash: dash.slice(), w: cx.lineWidth });
    poly = null;
    return keep.stroke.call(cx);
  };
  cx.fillRect = (x, y, w, h) => {
    rec.cells.push([x, y, w, h, String(cx.fillStyle)]);
    return keep.fillRect.call(cx, x, y, w, h);
  };
  cx.fillText = (t, x, y) => {
    rec.text.push([String(t), x, y, cx.textAlign, cx.measureText(String(t)).width]);
    return keep.fillText.call(cx, t, x, y);
  };
  try { c.draw(null); } finally { for (const m of Object.keys(keep)) cx[m] = keep[m]; }
  rec.bars = c.barT.map(x => x.innerHTML);
  rec.barTi = c.barD ? c.barD.title : "";
  // the RENDERED hint, not the type's copy: this card's hint names the colour scale, so it
  // is a function of the options and the card has to re-render it when they change
  rec.hint = c.hint.innerHTML;
  return rec; }`;
// the frame and the panel, as data. `ylo`/`ytop` are the axis the DATA occupies and `yhi`
// the axis the plot is DRAWN on -- they differ by the legend headroom, and the second-round
// leg below mirrors that relation instead of taking it on trust. The panel's rows ride out
// whole so the boundary rule can be recomputed here from (k, v) pairs + the floor rather
// than by calling the app's `gen2dTop` on both sides of the comparison.
const FRAME = `function(){
  const c = cards.chart.filter(x => x.type() === "gen2d")[0], o = c.optVals();
  const G = gen2dPanel(o);
  if (!G) return null;
  const LY = gen2dLayout(c.cx, G, o, ""), F = LY.F;
  return { hleg: LY.hleg, nleg: legendLines(c.cx, LY.lgx, LY.items, LY.lgxm),
           lgx: LY.lgx, lgxm: LY.lgxm,
           raw: (function(){ const R = gen2dFrame(G); return { ylo: R.ylo, yhi: R.yhi }; })(),
           xlo: F.xlo, xhi: F.xhi, ylo: F.ylo, ytop: F.ytop, yhi: F.yhi, lb: F.lb.slice(),
           e0: F.edge(0), eN: F.edge(F.lb.length), pk: G.d.parKfac, nzb: G.nzb, nb: G.d.nb,
           fshell: (G.d.fshell || []).slice(), kA: fitKA(G.d.nb, G.d.fshell),
           kc: G.d.bands.map(b => b.kc), floor: G.floor, hi: G.hi,
           rows: G.rows.map(r => Array.prototype.slice.call(r)),
           top: G.rows.map(r => gen2dTop(r, G.floor)),
           margin: GEN2D_TOPMARGIN,
           slopes: GEN2D_SLOPES.map(s => [s[0], s[1], s[2], s[3].slice()]),
           anchor: GEN2D_SLOPES.map(s => gen2dAnchor(G, s[0])),
           COL: { shell: COL.shell, ek: COL.ek, em: COL.em, txt: COL.txt } }; }`;
// the gen2d card's OWN box (GSW x GSH), not the shared spectrum one -- second-round item 4
const GEO = `function(){
  const c = cards.chart.filter(x => x.type() === "gen2d")[0];
  return { l: PADS.l, r: PADS.r, t: PADS.t, b: PADS.b, W: GSW, H: GSH, SW: SW, SH: SH,
           ar: c.cv.style.aspectRatio || "" }; }`;

async function legPlot(state) {
  const env = await boot(dir, "rmhd3d.html");
  env.run(`function(){ addChartCard("gen2d"); cardsSync(); }`);
  installSynth(env);
  ok("the press completes, so there is a plot to measure", await pressGenerate(env));
  const P = env.run(GEO);
  const x0 = P.l, x1 = P.W - P.r, y0 = P.t, y1 = P.H - P.b;
  env.run(SETOPT, { gp: "fl", gq: "tot", gc: "log", gov: "off" });
  const F = ridgesOf(env.run(FRAME));
  if (!F) { ok("the panel has rows to plot", false); return; }
  // the app's own affine log -> pixel maps, mirrored (the only arithmetic this leg owns)
  const MX = L => x0 + (L - F.xlo) / (F.xhi - F.xlo) * (x1 - x0);
  const MY = L => y1 - (L - F.ylo) / (F.yhi - F.ylo) * (y1 - y0);
  const L10 = Math.log10;

  // ---- second-round item 4: the card's own box, with a SQUARE plot area ----
  ok("the gen2d card has dimensions of its own, not the shared spectrum SW x SH",
     P.W === C.GSW && P.H === C.GSH && !(P.W === P.SW && P.H === P.SH) &&
     C.CHART_TYPES.gen2d.w === C.GSW && C.CHART_TYPES.gen2d.h === C.GSH &&
     C.CHART_TYPES.spectrum.w === P.SW && C.CHART_TYPES.spectrum.h === P.SH,
     "gen2d " + P.W + "x" + P.H + ", spectrum " + P.SW + "x" + P.SH);
  ok("  ... and the PLOT AREA inside PADS is square, to the pixel",
     (x1 - x0) === (y1 - y0) && (x1 - x0) > 200,
     "plot " + (x1 - x0) + " x " + (y1 - y0) + " (was " + (P.SW - P.l - P.r) + " x "
     + (P.SH - P.t - P.b) + ")");
  ok("  ... and the real canvas carries that ratio, so the card renders it",
     P.ar.replace(/\s/g, "") === C.GSW + "/" + C.GSH, "aspect-ratio: " + P.ar);

  // ---- item 1: the x axis IS the drawn columns ----------------------------
  ok("the x axis starts at the first column's left edge and ends at the last one's right",
     F.xlo === F.e0 && F.xhi === F.eN &&
     rel(F.xlo, F.lb[0] - 0.5 * (F.lb[1] - F.lb[0])) < 1e-12 &&
     rel(F.xhi, F.lb[F.lb.length - 1] + 0.5 * (F.lb[F.lb.length - 1] - F.lb[F.lb.length - 2])) < 1e-12,
     "k⊥ " + Math.pow(10, F.xlo).toFixed(2) + " .. " + Math.pow(10, F.xhi).toFixed(2));
  ok("  ... so the low-k⊥ blank is gone: it no longer starts at k⊥ = 1, below every band",
     F.xlo > L10(F.kA) && F.xlo > 0.5 * L10(F.kc[0]) && F.xlo < L10(F.kc[0]),
     "starts at " + Math.pow(10, F.xlo).toFixed(2) + ", band anchor kA = " + F.kA
     + ", first band centre " + F.kc[0].toFixed(2));

  // ---- item 2: the k∥ floor is the forcing fundamental ---------------------
  ok("the k∥ axis starts AT the first bin -- the forcing fundamental -- not half a bin below",
     F.ylo === L10(F.pk) && F.ylo > L10(0.5 * F.pk) &&
     rel(F.ytop, L10((F.nzb + 0.5) * F.pk)) < 1e-12 &&
     rel(F.raw.yhi, F.ytop) < 1e-12,
     "k∥ " + Math.pow(10, F.ylo) + " .. " + Math.pow(10, F.ytop).toFixed(2)
     + " resolved (drawn to " + Math.pow(10, F.yhi).toFixed(2) + ", parKfac = " + F.pk + ")");
  // ... and it is the FUNDAMENTAL, not the number 1: on a 4x longer box the same forcing
  // injects at k∥ = 0.25, and a floor pinned at 1 would clip two resolved octaves
  const F8 = env.run(`function(pk){
    const d = gen2d.data, was = d.parKfac;
    d.parKfac = pk;
    const F = gen2dFrame(gen2dPanel({ gp: "fl", gq: "tot" }));
    d.parKfac = was;
    return { ylo: F.ylo, ytop: F.ytop }; }`, 0.25 * F.pk);
  ok("  ... and it TRACKS the box: a 4x longer Lz moves the floor to the new fundamental",
     rel(F8.ylo, L10(0.25 * F.pk)) < 1e-12 && rel(F8.ytop, L10((F.nzb + 0.5) * 0.25 * F.pk)) < 1e-12,
     "floor " + Math.pow(10, F8.ylo) + " at parKfac " + (0.25 * F.pk));

  // ---- the pixels: the cells fill the frame exactly ------------------------
  const R = env.run(DRAWREC);
  const cells = R.cells.filter(c => /^rgb\(/.test(c[4]));
  const cx0 = Math.min(...cells.map(c => c[0])), cx1 = Math.max(...cells.map(c => c[0] + c[2]));
  ok("DRAWN: the columns span the frame exactly, left edge to right edge",
     cells.length > 20 && cx0 === x0 && cx1 === x1,
     cells.length + " cells, x " + cx0 + ".." + cx1 + " (frame " + x0 + ".." + x1 + ")");
  const cyTop = Math.min(...cells.map(c => c[1])), cyBot = Math.max(...cells.map(c => c[1] + c[3]));
  const botTop = Math.min(...cells.filter(c => c[1] + c[3] === cyBot).map(c => c[1]));
  ok("  ... nothing above the frame, and the lowest bin's bottom half hangs below it (clipped)",
     cyTop >= y0 && cyBot > y1 && y1 - botTop >= 8,
     "top " + cyTop + " (frame " + y0 + "), bottom " + cyBot + " (frame " + y1
     + "), lowest row " + (y1 - botTop) + " px visible");

  // ---- item 6: the axis label owns the left of the tick line ---------------
  const tl = R.text.filter(t => t[2] === P.H - 8);
  const lab = tl.filter(t => t[3] === "left")[0];
  const box = t => t[3] === "left" ? [t[1], t[1] + t[4]]
                 : t[3] === "right" ? [t[1] - t[4], t[1]] : [t[1] - 0.5 * t[4], t[1] + 0.5 * t[4]];
  let clash = "";
  for (const t of tl) {
    if (t === lab) continue;
    const a = box(lab), b = box(t);
    if (b[0] < a[1] && b[1] > a[0] && !clash) clash = JSON.stringify(t[0]) + " at " + b.join("..");
  }
  // The end-of-axis label is the centred one at the right edge; everything else on this
  // baseline is an INTERIOR tick label. There has to be at least one, or the collision test
  // below is a statement about the empty set -- a mutation that suppressed every interior
  // label (the `clear` test in drawGen2D always false) used to pass it (review, this round).
  const endLab = tl.filter(t => t[3] === "center" && Math.abs(t[1] - x1) < 1e-6)[0];
  const inner = tl.filter(t => t !== lab && t !== endLab && String(t[0]).length);
  ok("interior x tick labels survive the label-owns-its-space rule -- the set is not empty",
     inner.length >= 1,
     inner.length + " interior label(s): " + inner.map(t => t[0]).join(" "));
  ok("the x-axis label collides with no tick label on its own baseline",
     !!lab && !clash, lab ? JSON.stringify(lab[0]) + " spans " + box(lab).map(v => v.toFixed(0)).join("..")
                            + ", " + (tl.length - 1) + " labels beside it" + (clash ? "; HITS " + clash : "")
                          : "no axis label drawn");
  ok("  ... and it names the x axis only, as the other k charts do (y is named in the hint)",
     !!lab && lab[0].indexOf("(y:") < 0 && lab[0].indexOf("k⊥") === 0 &&
     C.CHART_TYPES.gen2d.hint({}).indexOf("k&#8741;/kunit") >= 0,
     lab ? JSON.stringify(lab[0]) : "-");

  // ---- item 1 again: the forcing markers are DROPPED, not clamped ----------
  const shells = R.lines.filter(l => l.col === F.COL.shell);
  ok("the forcing-shell markers fall outside the new axis and are simply not drawn",
     shells.length === 0 && F.fshell.every(k => L10(k) < F.xlo),
     "fshell [" + F.fshell.join(", ") + "], axis starts at "
     + Math.pow(10, F.xlo).toFixed(2) + " -- " + shells.length + " markers");
  const SH2 = env.run(`function(kf){
    const d = gen2d.data, was = d.fshell;
    d.fshell = [kf];
    const c = cards.chart.filter(x => x.type() === "gen2d")[0], cx = c.cx;
    const ob = cx.beginPath, om = cx.moveTo, os = cx.stroke;
    let poly = null, out = [];
    cx.beginPath = () => { poly = []; };
    cx.moveTo = (x, y) => { if (poly) poly.push(x, y); };
    cx.stroke = () => { if (poly && poly.length) out.push([String(cx.strokeStyle), poly.slice()]); poly = null; };
    try { c.draw(null); } finally { cx.beginPath = ob; cx.moveTo = om; cx.stroke = os; d.fshell = was; }
    return out.filter(o => o[0] === COL.shell); }`, Math.pow(10, 0.5 * (F.xlo + F.xhi)));
  ok("  ... but the marker code still bites: a shell INSIDE the range draws one, in place",
     SH2.length === 1 && Math.abs(SH2[0][1][0] - MX(0.5 * (F.xlo + F.xhi))) <= 1,
     SH2.length + " marker(s)" + (SH2.length ? " at x = " + SH2[0][1][0].toFixed(1)
       + " (want " + MX(0.5 * (F.xlo + F.xhi)).toFixed(1) + ")" : ""));

  // ---- items 3 + 4: three slopes, anchored on the upper boundary -----------
  // Second round: the labels are Alfred's abbreviations and are compared VERBATIM, because
  // they are what the one-line legend can fit. The exponents are asserted as numbers (they
  // are the physics), and the long names they abbreviate must survive somewhere a reader
  // can find them -- the checkbox's tooltip and the manual.
  ok("three reference slopes are declared -- 2/3, 1/2 and 1, labelled GS95 / B06 / iso",
     F.slopes.length === 3 &&
     F.slopes.map(s => s[0]).join(",") === [2 / 3, 1 / 2, 1].join(",") &&
     F.slopes.map(s => s[1]).join(",") === "GS95,B06,iso",
     F.slopes.map(s => s[1] + " = " + s[0].toFixed(3)).join(" | "));
  const govTi = C.CHART_TYPES.gen2d.opts().filter(s => s.id === "gov")[0] || {};
  ok("  ... and the long names live where there is room for them: the control's own tooltip",
     /GS95/.test(govTi.ti) && /Boldyrev 2006/.test(govTi.ti) && /isotropic/.test(govTi.ti) &&
     /2\/3/.test(govTi.ti) && /1\/2/.test(govTi.ti) && govTi.t === "theory slopes",
     JSON.stringify(govTi.t) + ": " + govTi.ti);
  ok("  ... visually distinguishable: three different colours and three different dashes",
     new Set(F.slopes.map(s => s[2])).size === 3 &&
     new Set(F.slopes.map(s => s[3].join(","))).size === 3 &&
     F.slopes.every(s => [F.COL.ek, F.COL.em].indexOf(s[2]) < 0),
     F.slopes.map(s => s[2] + " [" + s[3].join(",") + "]").join(" | "));

  // ---- the boundary itself, recomputed here from the RAW rows --------------
  // Not by calling the app's gen2dTop: this leg then mirrors the anchor rule with the app
  // on BOTH sides of the comparison, and a mutation of the boundary test (>= floor to > 0,
  // say) passes unseen -- which is exactly what the reviewer demonstrated on the first
  // round. `F.rows` is the panel's own (k, v) pair list per band and `F.floor` its noise
  // floor; the rule is re-stated here in three lines, margin included.
  const topJs = (r, thr) => {
    let k = 0;
    for (let i = 0; i < r.length; i += 2) if (r[i + 1] >= thr && r[i] > k) k = r[i];
    return k;
  };
  const thr = F.margin * F.floor;
  const topWant = F.rows.map(r => topJs(r, thr));
  ok("the boundary per band, recomputed here from the raw rows and the floor, matches",
     F.top.length === topWant.length && F.top.every((t, j) => t === topWant[j]) &&
     F.top.some(t => t > 0),
     F.top.map(t => (t / F.pk).toFixed(0)).join(" ") + " bins");
  // ... and a cell BELOW the margin must not be able to claim the boundary. Two plants per
  // row: one at half the floor (plain sub-floor noise) and one at 1.05x the floor -- the
  // fp32 periodogram fuzz that the old `>= floor` test would have taken, moving that band's
  // boundary to the top of the plot and, through the max over bands, all three anchors with
  // it (0.71 decades on the reviewer's demonstration).
  const kPlant = F.nzb * F.pk;
  let plantBad = 0, oldWouldMove = 0;
  for (let j = 0; j < F.rows.length; j++) {
    if (!(F.top[j] > 0) || F.top[j] >= kPlant) continue;
    for (const v of [0.5 * F.floor, 1.05 * F.floor]) {
      const r = F.rows[j].concat([kPlant, v]);
      if (topJs(r, thr) !== F.top[j]) plantBad++;
      if (C.gen2dTop(r, F.floor) !== F.top[j]) plantBad++;
      if (v > F.floor && topJs(r, F.floor) !== F.top[j]) oldWouldMove++;   // the old rule
    }
  }
  ok("  ... and a planted sub-floor / 1.05x-floor cell above it is EXCLUDED, by both",
     plantBad === 0 && oldWouldMove > 0 && C.GEN2D_TOPMARGIN >= 2,
     "margin x" + C.GEN2D_TOPMARGIN + "; the bare `>= floor` rule would have moved "
     + oldWouldMove + " band(s) to bin " + (kPlant / F.pk).toFixed(0));

  // the anchor rule, mirrored: A = max over bands of (k∥_top + half a bin) / k_c^p. The
  // half bin is second-round item 5: a cell is DRAWN half a bin either side of its own k∥,
  // so anchoring on the centre put the line through the middle of the filled pixels.
  let anchBad = 0, anchCentre = 0;
  for (let i = 0; i < F.slopes.length; i++) {
    let A = 0, Ac = 0;
    for (let j = 0; j < F.kc.length; j++)
      if (topWant[j] > 0) {
        A = Math.max(A, (topWant[j] + 0.5 * F.pk) / Math.pow(F.kc[j], F.slopes[i][0]));
        Ac = Math.max(Ac, topWant[j] / Math.pow(F.kc[j], F.slopes[i][0]));
      }
    if (rel(F.anchor[i], A) > 1e-12) anchBad++;
    if (rel(F.anchor[i], Ac) < 1e-12) anchCentre++;      // still on the centre: not fixed
  }
  ok("  ... each anchored by max over bands of (k∥_top + half a bin) / k_c^p -- the TOP EDGE "
     + "of the boundary cell, not its centre",
     anchBad === 0 && anchCentre === 0 && F.anchor.every(a => a > 0) &&
     F.top.some((t, j) => t > 0 && Math.abs(t - F.ridge[j]) > 1e-12),
     "A = " + F.anchor.map(a => a.toPrecision(4)).join(", ")
     + "; boundary " + F.top.map(t => t.toFixed(1)).join(" ")
     + " vs ridge " + F.ridge.map(t => t.toFixed(1)).join(" "));
  // ... and in PIXELS, on the real canvas: slopes on, three straight 2-point strokes
  env.run(SETOPT, { gov: "on" });
  const RO = env.run(DRAWREC);
  const cols = F.slopes.map(s => s[2]);
  // (the legend's own 12-px keys are strokes in the same three colours -- the plot lines are
  // the ones that run the full width of the frame, and both sets must be there)
  const inCol = l => l.p.length === 4 && cols.indexOf(l.col) >= 0;
  const refs = RO.lines.filter(l => inCol(l) && Math.abs(l.p[0] - x0) < 1e-6 &&
                                    Math.abs(l.p[2] - x1) < 1e-6);
  const keys = RO.lines.filter(l => inCol(l) && l.p[0] > x0 && l.p[2] - l.p[0] < 20);
  ok("with the theory box ON the three slopes are drawn, each straight across the whole "
     + "axis, each with its own legend key",
     refs.length === 3 && new Set(refs.map(l => l.col)).size === 3 &&
     keys.length === 3 && new Set(keys.map(l => l.col)).size === 3 &&
     refs.every(l => l.dash.length >= 2),
     refs.map(l => l.col + " " + l.p.map(v => v.toFixed(0)).join(",")).join(" | "));
  // THE rule: at every band the line is at or above the TOP EDGE of the boundary cell
  // (pixel y is smaller going up), and at one band it touches that edge. Both halves, per
  // line. Against the cell's CENTRE this used to pass with the line buried in the pixels.
  let above = 0, touch = 0, worst = "";
  for (const l of refs) {
    const yAt = xp => l.p[1] + (l.p[3] - l.p[1]) * (xp - l.p[0]) / (l.p[2] - l.p[0]);
    let bad = 0, hit = 0, mx = 0;
    for (let j = 0; j < F.kc.length; j++) {
      if (!(topWant[j] > 0)) continue;
      // <= 0 means at or above the drawn cell's own top edge
      const dy = yAt(MX(L10(F.kc[j]))) - MY(L10(topWant[j] + 0.5 * F.pk));
      if (dy > 0.5) { bad++; mx = Math.max(mx, dy); }
      if (Math.abs(dy) <= 0.5) hit++;
    }
    if (!bad) above++;
    if (hit) touch++;
    if (bad && !worst) worst = l.col + " dips " + mx.toFixed(1) + " px below";
  }
  ok("  ... and each clears the filled pixels at every band, touching the top edge at one",
     above === 3 && touch === 3, above + "/3 clear of the boundary, " + touch + "/3 touching it"
     + (worst ? "; " + worst : ""));

  // ---- second-round item 3: ONE legend line, three swatches ----------------
  // the legend's own texts: left-aligned, and up at the top of the frame (the x-axis label
  // is left-aligned too, but it sits on the tick baseline at the bottom)
  const legTxt = RO.text.filter(t => t[3] === "left" && t[2] < y0 + 80 && t[1] >= F.lgx);
  const theory = legTxt.filter(t => ["GS95", "B06", "iso"].indexOf(t[0]) >= 0);
  const head = legTxt.filter(t => /generated @ t =/.test(t[0]));
  ok("the legend names the three slopes by Alfred's abbreviations and nothing longer",
     theory.length === 3 && theory.map(t => t[0]).join(",") === "GS95,B06,iso" &&
     head.length === 1,
     legTxt.map(t => JSON.stringify(t[0])).join(" "));
  ok("  ... all three on ONE line, in order, each with its own dash / colour swatch",
     theory.length === 3 && new Set(theory.map(t => t[2])).size === 1 &&
     theory[0][1] < theory[1][1] && theory[1][1] < theory[2][1] &&
     keys.length === 3 && new Set(keys.map(l => l.p[1])).size === 1 &&
     Math.abs(keys[0].p[1] - (theory[0][2] - 3)) < 1e-6 &&
     new Set(keys.map(l => l.dash.join(","))).size === 3,
     "baseline y = " + theory[0][2] + ", labels at x " +
     theory.map(t => t[1].toFixed(0)).join(", "));
  // two lines with the box on (header, then all three slopes together) -- not the four the
  // old one-entry-per-slope legend took -- and one with it off, while the RESERVE stays at
  // the two-line height either way so the axis cannot jump under the reader
  ok("  ... and the whole block is 2 lines, not 4: header, then the three together",
     new Set(legTxt.map(t => t[2])).size === 2 && F.hleg === 12 * 2 + 4 && F.nleg === 1,
     "canvas shows " + new Set(legTxt.map(t => t[2])).size + " lines with the box on, "
     + F.nleg + " with it off; " + F.hleg + " px reserved either way");

  // ---- second-round item 2: y headroom, so the legend clears the data ------
  const legBase = Array.from(new Set(legTxt.map(t => t[2]))).sort((a, b) => a - b);
  const legBot = legBase[legBase.length - 1] + 3;              // baseline + descenders
  ok("the y axis is raised ABOVE the resolved range to make room for the legend",
     F.yhi > F.ytop && rel(F.hleg, 12 * legBase.length + 4) < 1e-12 &&
     Math.abs(MY(F.ytop) - (y0 + F.hleg)) < 1e-6,
     "resolved top k∥ = " + Math.pow(10, F.ytop).toFixed(1) + " at y = "
     + MY(F.ytop).toFixed(1) + ", axis top " + Math.pow(10, F.yhi).toFixed(1)
     + " at y = " + y0 + " (" + (F.hleg).toFixed(0) + " px reserved)");
  const cellsO = RO.cells.filter(c => /^rgb\(/.test(c[4]));
  const cTop = Math.min(...cellsO.map(c => c[1]));
  ok("  ... so the legend block sits over EMPTY canvas: no drawn cell reaches it",
     cellsO.length > 20 && cTop >= legBot && legBot > y0,
     "legend ends at y = " + legBot + ", topmost cell at y = " + cTop);
  // and it is not a decoration: on the un-extended axis the top row WOULD have been under it
  const MYraw = L => y1 - (L - F.ylo) / (F.raw.yhi - F.ylo) * (y1 - y0);
  let kCell = 0;
  for (const r of F.rows) for (let i = 0; i < r.length; i += 2) if (r[i] > kCell) kCell = r[i];
  ok("  ... and it is doing work: without it the topmost cells would be under the legend",
     kCell > 0 && MYraw(L10(kCell + 0.5 * F.pk)) < legBot,
     "top cell would sit at y = " + MYraw(L10(kCell + 0.5 * F.pk)).toFixed(1)
     + " vs the legend's " + legBot);
  // the axis stays honest: the ticks are generated over the DRAWN span, so the added strip
  // carries gridlines and labels like the rest of it
  const tkAll = C.logTicks(F.ylo, F.yhi, (y1 - y0) / (F.yhi - F.ylo),
                           (m, e) => String(Math.round(m * Math.pow(10, e) * 1e6) / 1e6));
  const tkAdd = tkAll.filter(t => L10(t[0]) > F.ytop);
  const grid = RO.lines.filter(l => l.p.length === 4 && Math.abs(l.p[0] - x0) < 1e-6 &&
                                    Math.abs(l.p[2] - x1) < 1e-6 && l.p[1] === l.p[3]);
  const gridUp = grid.filter(l => l.p[1] < MY(F.ytop) - 0.5);
  ok("  ... and the ticks carry on across it -- the added range is real axis, not a margin",
     tkAdd.length >= 1 && gridUp.length >= tkAdd.length &&
     C.logTicks(F.ylo, F.ytop, (y1 - y0) / (F.ytop - F.ylo)).every(
       t => tkAll.some(u => u[0] === t[0])),
     tkAdd.length + " tick(s) above the resolved top ("
     + tkAdd.map(t => t[0]).join(", ") + "), " + gridUp.length + " gridlines drawn there");
  // no cell moves against the axis: every recorded cell is where the frame's own map puts it
  let cellBad = 0;
  for (let j = 0; j < F.rows.length; j++) {
    const r = F.rows[j];
    for (let i = 0; i < r.length; i += 2) {
      const ya = Math.round(MY(L10(r[i] + 0.5 * F.pk))), yb = Math.round(MY(L10(r[i] - 0.5 * F.pk)));
      if (!cellsO.some(c => c[1] === ya && c[1] + c[3] === Math.max(ya + 1, yb))) cellBad++;
    }
  }
  ok("  ... and every cell sits exactly where the frame's map puts it, half a bin either side",
     cellBad === 0, cellBad + " of " + cellsO.length + " cells off the map");

  env.run(SETOPT, { gov: "off" });
  const ROff = env.run(DRAWREC);
  ok("with the theory box OFF none of the three is drawn",
     ROff.lines.filter(l => l.p.length === 4 && cols.indexOf(l.col) >= 0).length === 0);
  ok("  ... but the axis does NOT jump: the headroom is measured on the full legend either way",
     (() => { const F2 = env.run(FRAME);
              return F2 && F2.hleg === F.hleg && rel(F2.yhi, F.yhi) < 1e-12; })(),
     "hleg " + F.hleg + " px with the box on and off alike");

  // ---- second-round item 1: the two measured curves are GONE ---------------
  // They were the anisotropy card's own k∥(k⊥), drawn in ITS colours (COL.ek / COL.em) and
  // legended "measured". Neither the strokes nor the words may be anywhere on this canvas
  // in either state of the checkbox, and the card must no longer even ask for them.
  const anisoCols = [F.COL.ek, F.COL.em];
  const measured = r => r.lines.filter(l => anisoCols.indexOf(l.col) >= 0).length
                      + r.text.filter(t => /measured|aniso/i.test(t[0])).length;
  ok("the two MEASURED anisotropy curves are gone from the card, box on and box off",
     measured(RO) === 0 && measured(ROff) === 0,
     "on: " + measured(RO) + " strokes/labels, off: " + measured(ROff));
  const src = fs.readFileSync(path.join(dir, "common.js"), "utf8");
  const draw = src.slice(src.indexOf("function drawGen2D("));
  ok("  ... and drawGen2D no longer calls anisoCurves at all",
     draw.slice(0, draw.indexOf("\nfunction ", 10)).indexOf("anisoCurves") < 0 &&
     src.indexOf("anisoCurves") > 0,          // still there for the anisotropy card itself
     "anisoCurves survives only in drawAniso");

  // ---- item 7: the colour scale toggle ------------------------------------
  const dflt = env.run("function(){ const c = cards.chart.filter(x => x.type() === 'gen2d')[0];"
    + " const s = c.optEls.filter(s => s.__optId === 'gc')[0];"
    + " return { v: s.value, o: s.options.map(o => o.value).join(',') }; }");
  ok("the colour scale is its own control, log by default",
     dflt.v === "log" && dflt.o === "log,lin", "gc = " + dflt.v + " of [" + dflt.o + "]");
  const Rlog = env.run(DRAWREC);
  env.run(SETOPT, { gc: "lin" });
  const Rlin = env.run(DRAWREC);
  const lum = s => s.slice(4, -1).split(",").reduce((a, v) => a + (+v), 0);
  const bright = r => r.cells.filter(c => /^rgb\(/.test(c[4])).reduce((s, c) => s + lum(c[4]), 0);
  const geom = r => r.cells.filter(c => /^rgb\(/.test(c[4])).map(c => c.slice(0, 4).join(",")).join("|");
  // the brightest CELL, by luminance and not by string order (both scales map the peak
  // value to the top of the ramp, so this one colour must survive the toggle unchanged)
  const peak = r => r.cells.filter(c => /^rgb\(/.test(c[4]))
    .map(c => c[4]).sort((a, b) => lum(a) - lum(b)).pop();
  ok("  ... and it really repaints: same cells, dimmer, with the peak cell unchanged",
     geom(Rlog) === geom(Rlin) && bright(Rlin) < bright(Rlog) && peak(Rlog) === peak(Rlin),
     "brightness " + bright(Rlog).toFixed(0) + " log -> " + bright(Rlin).toFixed(0) + " lin");
  ok("the colorbar labels follow the scale: geometric mean on log, arithmetic on linear",
     Rlog.bars[0] === cbarFmtJs(F.floor) && Rlog.bars[1] === cbarFmtJs(Math.sqrt(F.floor * F.hi)) &&
     Rlin.bars[0] === "0" && Rlin.bars[1] === cbarFmtJs(0.5 * F.hi) &&
     Rlog.bars[2] === Rlin.bars[2] && Rlog.bars[1] !== Rlin.bars[1],
     "log [" + Rlog.bars.join(" .. ") + "]  lin [" + Rlin.bars.join(" .. ") + "]");
  ok("  ... and so does the colorbar's own tooltip",
     /log scale/.test(Rlog.barTi) && /arithmetic mean/.test(Rlin.barTi) &&
     !/log scale/.test(Rlin.barTi), Rlin.barTi);

  // ---- item 8: the hint is Alfred's copy, and docs.html says the same ------
  // Third round: his copy verbatim, with the colour-scale word following the `gc` select.
  // So the hint is a FUNCTION of the options and the two renderings must differ in that ONE
  // word and in nothing else -- checked by putting the word back and comparing strings.
  const hf = C.CHART_TYPES.gen2d.hint;
  const h = hf({ gc: "log" }), hlin = hf({ gc: "lin" });
  ok("the hint is Alfred's copy: his sentence, his three slopes, his register",
     /^two-dimensional spectrum E\(k&perp;, k&#8741;\) from one frozen snapshot/.test(h) &&
     /<b>generate<\/b> pauses the run and band-passes the field in k&perp;, band by band/.test(h) &&
     /overlay lines correspond to GS95/.test(h) && /Boldyrev 2006/.test(h) &&
     /isotropic/.test(h) && /experimental feature: imperfect agreement at these resolutions\.$/.test(h),
     h.length + " chars");
  ok("  ... with the parenthesis after the Boldyrev exponent closed (his one slip, both times)",
     (h.match(/\(/g) || []).length === (h.match(/\)/g) || []).length &&
     /Boldyrev 2006 \(k&#8741; &prop; k&perp;<sup>1\/2<\/sup>\), and isotropic/.test(h),
     (h.match(/\(/g) || []).length + " ( against " + (h.match(/\)/g) || []).length + " )");
  ok("  ... it says \"colour is log E\" on the log scale and \"linear\" on the linear one",
     /colour is log E, y is k&#8741;\/kunit\./.test(h) &&
     /colour is linear E, y is k&#8741;\/kunit\./.test(hlin) &&
     hlin.replace("colour is linear E", "colour is log E") === h,
     "the two renderings differ in that word alone");
  ok("  ... and the CARD re-renders it when the select moves, rather than freezing the "
     + "wording it was built with",
     /colour is log E/.test(Rlog.hint) && /colour is linear E/.test(Rlin.hint) &&
     Rlin.hint === hlin, Rlin.hint.slice(0, 120) + "...");
  ok("  ... it drops the measured-vs-prediction distinction with the curves that made it, "
     + "and the legend abbreviations with it (they live in the manual now)",
     !/measured/i.test(h) && !/aniso/i.test(h) && !/<b>GS95<\/b>/.test(h) &&
     !/<b>B06<\/b>/.test(h) && !/<b>iso<\/b>/.test(h) && !/upper edge/.test(h),
     h);
  const doc = fs.readFileSync(path.join(dir, "docs.html"), "utf8");
  const sec = doc.slice(doc.indexOf('id="gen2d"'), doc.indexOf('id="gen2d"') + 6000);
  ok("  ... and the manual entry carries the same three slopes, the boundary rule and the "
     + "colour toggle",
     /Boldyrev 2006/.test(sec) && /isotropic/.test(sec) && /2\/3/.test(sec) &&
     /linear colour/.test(sec) && /upper edge/.test(sec) && /injection scale/.test(sec));
  ok("  ... names the control <b>theory slopes</b>, gives the three abbreviations, and no "
     + "longer promises any measured curve",
     /<b>theory slopes<\/b>/.test(sec) && /<b>GS95<\/b>/.test(sec) && /<b>B06<\/b>/.test(sec) &&
     /<b>iso<\/b>/.test(sec) && !/<b>measured<\/b>/.test(sec) &&
     !/anisotropy chart's own two curves/.test(sec),
     "gen2d section, " + sec.length + " chars scanned");
  ok("  ... and it states BOTH second-round corrections: the noise margin and the headroom",
     /twice the noise floor/.test(sec) && /top edge of the last filled cell/.test(sec) &&
     /carries on a little past that last bin/.test(sec) && /no cell moves against the axis/.test(sec));
  ok("the 3D page boots clean through all of it", env.fails.length === 0, env.fails.join(" | "));

  // ---- the legend primitive itself, since it grew a GROUP form -------------
  // The generic change (a list of triples in item[0] is one unwrappable unit) must leave
  // the ordinary form bit for bit as it was -- every other chart in the file uses it.
  const mk = () => {
    const rec = [];
    return { rec, cx: { measureText: s => ({ width: 6 * String(s).length }),
                        strokeStyle: "", fillStyle: "", lineWidth: 0,
                        setLineDash(d) { this._d = (d || []).join(","); },
                        beginPath() {}, moveTo(x, y) { rec.push(["m", x, y, this._d]); },
                        lineTo() {}, stroke() {}, fillText(t, x, y) { rec.push(["t", t, x, y]); } } };
  };
  const flat = [["aaa", "#111"], ["bb", "#222", [1, 2]], ["c", "#333", [3, 4]]];
  const grouped = [[flat]];
  const a = mk(), b = mk(), n1 = C.legend(a.cx, 10, 20, flat, 1000),
        n2 = C.legend(b.cx, 10, 20, grouped, 1000);
  ok("legend(): a group laid out on one line is byte-identical to the same items loose",
     JSON.stringify(a.rec) === JSON.stringify(b.rec) && n1 === 1 && n2 === 1,
     a.rec.length + " ops each");
  const narrow = 10 + 15 + 6 * 3 + 11 + 5;      // room for the first entry and no more
  const c1 = mk(), c2 = mk();
  C.legend(c1.cx, 10, 20, flat, narrow);
  C.legend(c2.cx, 10, 20, grouped, narrow);
  const ys = r => Array.from(new Set(r.filter(o => o[0] === "t").map(o => o[3])));
  ok("  ... but at a width that cannot hold it, the group moves as ONE and never splits",
     ys(c1.rec).length === 3 && ys(c2.rec).length === 1 &&
     C.legendLines(c1.cx, 10, flat, narrow) === 3 &&
     C.legendLines(c2.cx, 10, grouped, narrow) === 1,
     "loose wraps onto " + ys(c1.rec).length + " lines, the group stays on "
     + ys(c2.rec).length);
  ok("  ... and legendLines counts what legend() draws, plain items and groups alike",
     C.legendLines(mk().cx, 10, flat, 1000) === 1 &&
     C.legendLines(mk().cx, 10, flat.concat(grouped), narrow) ===
       ys((() => { const m = mk(); C.legend(m.cx, 10, 20, flat.concat(grouped), narrow); return m.rec; })()).length);
}
// cbarFmt, mirrored (the labels are compared against the values this leg computed itself)
function cbarFmtJs(v) {
  if (!isFinite(v)) return "";
  const a = Math.abs(v);
  if (a === 0) return "0";
  return (a >= 1e4 || a < 1e-2) ? v.toExponential(1) : v.toPrecision(3);
}

// ---------------------------------------------------------------------------
const LEGS = [
  ["1. the emission: parses, resolves, and is " + BASE + " plus two kernels", legEmission],
  ["2. the banded kernels ARE the unbanded templates plus the shared factor", legTemplates],
  ["3. state invariance: a press leaves (phi_k, psi_k) bitwise unchanged", legInvariance],
  ["4. the fp64 mirror: band factor x gradient x periodogram row", legMirror],
  ["5. ridge recovery, the band set, and the coordinate panel's anchor", legRidge],
  ["6. the press: pause / resume choreography and plot persistence", legChoreography],
  ["7. the plot: axis limits, the k∥ floor, the three anchors, the colour scale", legPlot]
];
(async () => {
  const state = { cur: {}, base: {} };
  state.M = await wgslMod();
  for (const [title, fn] of LEGS) { console.log(title); await fn(state); }
  try { fs.rmSync(tmp, { recursive: true, force: true }); } catch (e) {}
  console.log(bad ? "\n" + bad + " FAILURE(S)" : "\nall checks passed");
  process.exit(bad ? 1 : 0);
})();
