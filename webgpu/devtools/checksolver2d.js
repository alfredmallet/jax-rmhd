// GAME_PLAN Phase 0 gate: the 2D solver extracted out of rmhd2d.html's inline script
// into the shared classic script solver2d.js, which game.html will load next to it.
//
//   node checksolver2d.js [webgpu-dir]        exit code 1 on any failure
//
// Phase 0 is a PURE refactor, so every leg below is a statement about nothing having
// changed, plus two about the shape the move has to have:
//   1  discipline: every emitted kernel of both pages parses, names.mjs clean, dup.py
//      showing no clone inside a file, none reaching into common.js / physics.js, and
//      none between solver2d.js and the inline script it came out of.
//   2  WGSL byte-identity against the base commit -- every kernel that existed there is
//      unmoved and the additions are EXACTLY nothing, on BOTH pages; plus the whole 2D
//      dump's sha256 against the base dump's, belt and braces.
//   3  the RNG reference unmoved (64 Gauss(7) draws, hashed), the standing rule.
//   4  extraction shape: the three definitions are gone from the inline script and are
//      solver2d.js's only top-level ones, the page loads it after physics.js and before
//      its own script, the 3D page does not load it at all, and no module syntax reaches
//      any of it (Chrome blocks module scripts from file://).
//   5  the verbatim move: each definition's text equals the BASE commit's, byte for byte
//      -- `class Solver` after ONE recorded allowance (IO_PLAN item 4) is stripped.
//   6  reusability: solver2d.js's free identifiers resolve against common.js + physics.js
//      + builtins ALONE -- never against rmhd2d.html's inline script, which is the whole
//      reason a second page can load it.
//   7  pages.yml: solver2d.js in the MISSING check and in the cache-bust sed, and the
//      game.html prune pre-wired.
//   8  boot: both pages run to the end of the self-test path on the stub GPU.
//
// CI reports, never gates.
"use strict";
const fs = require("fs"), os = require("os"), path = require("path");
const { spawnSync } = require("child_process");
const dir = path.resolve(process.argv[2] || path.join(__dirname, ".."));
const root = path.resolve(dir, "..");
// GAME_PLAN's base commit: the tree Phase 0 starts from. SOLVER2D_BASE overrides.
const BASE = process.env.SOLVER2D_BASE || "268100f";
// the 64 Gauss(7) draws the forcing path starts from, hashed, RECORDED FROM THE BASE TREE
// before a line of this phase was written (the same recipe and the same value as
// checkeigf.js's RNG_SHA -- one reference, two gates).
const RNG_SHA = "340f43548a5ed68fcefda6f8e08080075bbd011c247813d93803c8f63519d137";
// a pure refactor adds no kernel to either page. A stale expectation must fail too, so
// the additions are compared against this list rather than merely counted.
const ADDED = { "rmhd2d.html": [], "rmhd3d.html": [] };
// the three definitions that moved, by their header line
const DEFS = ["function makeGrid(p) {", "function buildShaders(g) {", "class Solver {"];

// THE ONE SANCTIONED EDIT to `class Solver` since the move: IO_PLAN item 4's field
// export (webgpu/README.md, "The fields themselves, as `.npz`"). It adds two
// PINNED Mode uniforms and their two prepDisp bind groups to _makeChain, plus the
// encodeExport method -- buffers and bind groups only, which is why leg 2's WGSL
// byte-identity does not move with it. Recorded here as the exact inserted LINES **and
// where they go**, in dispoffsets.js's spirit: leg 5 strips them and then still demands
// the base text byte for byte, so any OTHER change to the class fails, and a block that
// has stopped being there fails as a STALE ALLOWANCE rather than quietly widening the leg.
//
// `at` is the 0-based line of the BASE `class Solver` text the block is inserted before,
// and `tag` names it in the failure message. Position is part of the contract: matching a
// block ANYWHERE in the class would let the same lines MOVED elsewhere pass as unchanged,
// which is the opposite of what the leg claims (adversarial review 2026-08-20, MINOR 6).
const IO4_INSERTS = [
  { tag: "modeX / modeX2, the two pinned Mode uniforms (_makeChain)", at: 124, lines: [
   "      // the FIELD EXPORT's pair (IO_PLAN item 4): plain phi and plain psi, no band, no",
   "      // offset, no colormap. Written once below and never again -- setDisplayMode does",
   "      // not touch them -- which is what lets an export run without disturbing whatever",
   "      // this card is showing.",
   "      modeX: d.createBuffer({ size: MODE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),",
   "      modeX2: d.createBuffer({ size: MODE_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),"] },
  { tag: "the one-time write of the pinned pair (_makeChain)", at: 130, lines: [
   "    d.queue.writeBuffer(B.modeX, 0, modeWords(DISP_PHI, 0, 0, null));",
   "    d.queue.writeBuffer(B.modeX2, 0, modeWords(DISP_PSI, 0, 0, null));"] },
  { tag: "prepDispX / prepDispX2, the export's two bind groups (_makeChain)", at: 142, lines: [
   "      // the export's two preps: phi through the chain's first component, psi through the",
   "      // second (the same two-slot pattern contPrep uses -- each writes the OTHER",
   "      // component's scratch as its throwaway outk2)",
   "      prepDispX: bgOf(this.pl.prepDisp, [P.fields, P.gridA, B.dispK, B.modeX, B.dispK2]),",
   "      prepDispX2: bgOf(this.pl.prepDisp, [P.fields, P.gridA, B.dispK2, B.modeX2, B.dispK]),"] },
  { tag: "the encodeExport method", at: 500, lines: [
   "  // the field export (IO_PLAN item 4): plain phi -> dispR and plain psi -> dispR2, each",
   "  // through its own PINNED Mode uniform, so no live display uniform is written. Same",
   "  // three stages the display uses; the psi leg runs second because its prep zeroes dispK,",
   "  // which the phi leg has finished with by then. `readFieldPair` (common.js) owns the",
   "  // rest -- the 3D chain's version of this is the same six lines with one stage more.",
   "  encodeExport(p, D) {",
   "    const legs = [[D.bg.prepDispX, D.bg.colsInvDisp, D.bg.rowsC2RDisp],",
   "                  [D.bg.prepDispX2, D.bg.colsInvDisp2, D.bg.rowsC2RDisp2]];",
   "    for (const leg of legs) {",
   "      p.setPipeline(this.pl.prepDisp); p.setBindGroup(0, leg[0]);",
   "      p.dispatchWorkgroups(Math.ceil(this.g.nm / 64));",
   "      p.setPipeline(this.pl.colsInv); p.setBindGroup(0, leg[1]);",
   "      p.dispatchWorkgroups(this.g.nky);",
   "      p.setPipeline(this.pl.rowsC2R); p.setBindGroup(0, leg[2]);",
   "      p.dispatchWorkgroups(this.g.nx);",
   "    }",
   "  }",
   ""] }
];

// THE SECOND SANCTIONED EDIT, in the same idiom: FFTPERF_PLAN 2C's gradient chunking. The
// gradient chain runs one (x, y) pair at a time now, which touches the two definitions
// this file pins -- `buildShaders` emits four prepGrads instead of one, and `class Solver`
// builds four pipelines and four row-kernel targets, holds a two-lane k-space stack, and
// encodes the chain in one method the RHS calls. Recorded as REPLACEMENTS (`was` = the
// lines at BASE, `lines` = the lines now) at their base positions, so leg 5 puts base's
// text back and then still demands it byte for byte: any OTHER change to either definition
// fails, and a block that has stopped being there fails as a stale allowance.
const FFT2C_SHADERS = [
  { tag: "the four per-pair prepGrads emissions (the kernel list comment)", at: 53,
    was: ["  //   prepGrads   perpendicular i*k gradients of phi, psi, vort, jpar"],
    lines: [
   "  //   prepGrads   perpendicular i*k gradients of phi, psi, vort, jpar -- one pair per",
   "  //               emission, four of them (physics.js GRAD_PAIRS)"] },
  { tag: "... and the emission itself", at: 56,
    was: ["  S.prepGrads = prepGradsWGSL(C);"],
    lines: [
   "  GRAD_PAIRS.forEach((p, k) => { S[\"prepGrads\" + k] = prepGradsWGSL(Object.assign({}, C, { gpair: k })); });"] }
];
const FFT2C_SOLVER = [
  { tag: "the two-lane gradient stack (_buildBuffers)", at: 44,
    was: ["      gradsK: d.createBuffer({ size: 8 * cx, usage: SQ }),",
          "      specTmp: d.createBuffer({ size: 8 * cx, usage: SQ }),"],
    lines: [
   "      // the gradient chain transforms ONE (x, y) pair at a time, so the k-space stack and",
   "      // the column pass's target hold two lanes, not eight",
   "      gradsK: d.createBuffer({ size: 2 * cx, usage: SQ }),",
   "      specTmp: d.createBuffer({ size: 2 * cx, usage: SQ }),"] },
  { tag: "the four prepGrads pipelines (_buildPipelines)", at: 209,
    was: ["      prepGrads: cp(S.prepGrads, \"prepGrads\"), bracket: cp(S.bracket, \"bracket\"),"],
    lines: [
   "      prepGrads: GRAD_PAIRS.map((p, k) => cp(S[\"prepGrads\" + k], \"prepGrads\" + k)),",
   "      bracket: cp(S.bracket, \"bracket\"),"] },
  { tag: "their bind groups, and the row kernel's per-pair target (_buildPipelines)", at: 240,
    was: ["      prepGrads: bg(this.pl.prepGrads, [B.fields, B.gridA, B.gradsK]),",
          "      colsInvGrads: bg(this.pl.colsInv, [B.gradsK, B.specTmp]),",
          "      rowsC2RGrads: bg(this.pl.rowsC2R, [B.specTmp, B.realGrads]),"],
    lines: [
   "      prepGrads: this.pl.prepGrads.map(p => bg(p, [B.fields, B.gridA, B.gradsK])),",
   "      colsInvGrads: bg(this.pl.colsInv, [B.gradsK, B.specTmp]),",
   "      // one target per pair: the same row kernel, its store landing in the pair's own two",
   "      // lanes of realGrads through the binding's offset (physics.js gradPairOffset)",
   "      rowsC2RGrads: GRAD_PAIRS.map((p, k) => bg(this.pl.rowsC2R,",
   "        [B.specTmp,",
   "         { buffer: B.realGrads, offset: gradPairOffset(this.nr, k), size: 2 * this.nr * 4 }])),"] },
  { tag: "the encodeGrads method", at: 392, lines: [
   "  // the eight perpendicular gradients, into realGrads (lanes 0,1 = grad phi, 2,3 = grad",
   "  // psi, 4..7 = grad vorticity / current): one pair at a time, each prepGrads writing the",
   "  // two-lane k-space stack and the two inverse passes carrying it to the pair's own lanes",
   "  // of realGrads. The only place the chain is encoded.",
   "  encodeGrads(pass) {",
   "    const nm = this.g.nm, nky = this.g.nky, nx = this.g.nx;",
   "    for (let k = 0; k < this.pl.prepGrads.length; k++) {",
   "      pass.setPipeline(this.pl.prepGrads[k]); pass.setBindGroup(0, this.bg.prepGrads[k]);",
   "      pass.dispatchWorkgroups(Math.ceil(nm / 64));",
   "      pass.setPipeline(this.pl.colsInv); pass.setBindGroup(0, this.bg.colsInvGrads);",
   "      pass.dispatchWorkgroups(2 * nky);",
   "      pass.setPipeline(this.pl.rowsC2R); pass.setBindGroup(0, this.bg.rowsC2RGrads[k]);",
   "      pass.dispatchWorkgroups(2 * nx);",
   "    }",
   "  }",
   ""] },
  { tag: "... which encodeRHS now calls instead of encoding the chain itself", at: 396,
    was: ["    pass.setPipeline(this.pl.prepGrads); pass.setBindGroup(0, this.bg.prepGrads);",
          "    pass.dispatchWorkgroups(Math.ceil(nm / 64));",
          "    pass.setPipeline(this.pl.colsInv); pass.setBindGroup(0, this.bg.colsInvGrads);",
          "    pass.dispatchWorkgroups(8 * nky);",
          "    pass.setPipeline(this.pl.rowsC2R); pass.setBindGroup(0, this.bg.rowsC2RGrads);",
          "    pass.dispatchWorkgroups(8 * nx);"],
    lines: ["    this.encodeGrads(pass);"] }
];
// every recorded block per pinned definition, in BASE-line order (which is what lets each
// block's found index BE its base line -- see stripBlocks)
const ALLOWED = {
  "function buildShaders(g) {": FFT2C_SHADERS,
  "class Solver {": FFT2C_SOLVER.concat(IO4_INSERTS).sort((a, b) => a.at - b.at)
};
// the text with every recorded block taken back out (a replacement putting its `was` lines
// back), plus whichever blocks were not found WHERE THEY ARE RECORDED. Blocks are listed
// in BASE-line order and each restores the text above the next one, so the index a block is
// found at IS its line in the base text. A block in the wrong place is left in, so the
// byte-identity leg above goes red with this one rather than silently absorbing the move.
function stripBlocks(txt, blocks) {
  const lines = txt.split("\n"), miss = [];
  let cut = 0;
  for (const blk of blocks) {
    const b = blk.lines, was = blk.was || [];
    let at = -1;
    for (let i = 0; i + b.length <= lines.length && at < 0; i++) {
      if (b.every((l, j) => lines[i + j] === l)) at = i;
    }
    if (at < 0) miss.push(blk.tag + ": not in the text at all");
    else if (at !== blk.at) miss.push(blk.tag + ": at base line " + at + ", recorded " + blk.at);
    else { lines.splice.apply(lines, [at, b.length].concat(was)); cut += b.length; }
  }
  return { txt: lines.join("\n"), miss: miss, cut: cut };
}

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "checksolver2d-"));
const sh = (cmd, args, opts) => spawnSync(cmd, args, Object.assign(
  { encoding: "utf8", cwd: __dirname, maxBuffer: 1 << 28 }, opts || {}));
const node = (args, opts) => sh(process.execPath, args, opts);
const lastLine = r => ((r.stdout || "") + (r.stderr || "")).trim().split("\n").pop();
const read = f => fs.readFileSync(f, "utf8");
const sha = b => require("crypto").createHash("sha256").update(b).digest("hex");
const OFF = require("./dispoffsets");
// a dump's sha with the sections the chunk allowance names taken out of it, on both sides
function shaNoChunk(file) {
  const parts = read(file).split(/^########## (.*) ##########$/m);
  let out = parts[0];
  for (let i = 1; i < parts.length; i += 2)
    if (!OFF.isChunkLabel(parts[i]) && !OFF.isGradsLabel(parts[i]))
      out += "########## " + parts[i] + " ##########" + parts[i + 1];
  return sha(out);
}
// the inline script of a page, exactly as stubenv and names.mjs slice it
const inline = t => t.slice(t.indexOf("<script>\n") + 8, t.lastIndexOf("</script>"));
// one top-level definition as TEXT: its header line down to the closing brace at column 0
// (no definition here contains one -- the WGSL template literals close with "}`;")
function defText(src, header) {
  const lines = src.split("\n");
  const i = lines.indexOf(header);
  if (i < 0) return null;
  for (let j = i + 1; j < lines.length; j++) if (lines[j] === "}") return lines.slice(i, j + 1).join("\n");
  return null;
}
// every kernel of one page, at every resolution preset + the self-test grid, as a
// { label :: kernel -> source } map, emitted by the real dumpwgsl2 (the checkeigf helper)
function dumpKernels(d, page, tag) {
  const out = path.join(tmp, tag + "-" + page.replace(".html", "") + ".wgsl.txt");
  const r = node([path.join(__dirname, "dumpwgsl2.js"), d, page, "", out]);
  if (r.status !== 0) { ok(tag + " " + page + ": dumpwgsl2 ran", false, lastLine(r)); return { file: out, k: {} }; }
  const txt = read(out);
  const parts = txt.split(/^########## (.*) ##########$/m);
  const k = {};
  for (let i = 1; i < parts.length; i += 2) k[parts[i]] = parts[i + 1];
  return { file: out, k };
}
// the base commit's four files, checked out into tmp/base. It predates solver2d.js, so
// its rmhd2d.html carries the solver inline -- which is exactly what leg 5 reads.
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
// wgsl_reflect, from wherever it is installed (the wgslparse / checkeigf idiom)
const wgslPath = () => [process.env.WGSL_REFLECT,
                        path.join(__dirname, "node_modules", "wgsl_reflect", "wgsl_reflect.module.js")]
  .filter(p => p && fs.existsSync(p))[0] || null;
// acorn, the same way names.mjs finds it (npm i acorn; ACORN=<path> overrides)
let parse = null;
const acorn = async () => {
  try { parse = (await import(process.env.ACORN || "acorn")).parse; } catch (e) { parse = null; }
};
// the top-level declared names of one source, names.mjs's own routine
function topNames(src) {
  const s = new Set();
  for (const n of parse(src, { ecmaVersion: 2022 }).body) {
    if (n.type === "FunctionDeclaration" || n.type === "ClassDeclaration") s.add(n.id.name);
    else if (n.type === "VariableDeclaration") for (const d of n.declarations) {
      if (d.id.type === "Identifier") s.add(d.id.name);
      else if (d.id.type === "ArrayPattern") for (const e of d.id.elements) if (e) s.add(e.name);
    }
  }
  return s;
}
// boot() runs at require, its async tail a macrotask later (the checkonepage idiom)
const boot = (d, page, demo) => new Promise(res => {
  const env = require("./stubenv")(d, page, demo);
  setTimeout(() => res(env));
});

const HTML2D = read(path.join(dir, "rmhd2d.html"));
const HTML3D = read(path.join(dir, "rmhd3d.html"));
const SOLVER = read(path.join(dir, "solver2d.js"));

(async () => {
await acorn();
const env2d = await boot(dir, "rmhd2d.html", null);

// ---------------------------------------------------------------------------
console.log("1. discipline: every kernel parses, names.mjs clean, dup.py clean");
// ---------------------------------------------------------------------------
const cur = {};
{
  const wp = wgslPath();
  const penv = Object.assign({}, process.env);
  if (wp) penv.WGSL_REFLECT = wp;
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const d = dumpKernels(dir, page, "cur");
    cur[page] = d;
    const r = node([path.join(__dirname, "wgslparse.mjs"), d.file], { env: penv });
    ok(page + ": every emitted kernel parses (+ reserved-word scan)", r.status === 0, lastLine(r));
  }
  const nm = node([path.join(__dirname, "names.mjs"), dir]);
  ok("names.mjs: no redeclares, every free identifier resolves", nm.status === 0, lastLine(nm));
  // the shared core is THREE files now, and the extracted inline scripts are the other two
  const files = ["common.js", "physics.js", "solver2d.js"].map(f => path.join(dir, f));
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const f = path.join(tmp, page.replace(".html", ".js"));
    fs.writeFileSync(f, inline(read(path.join(dir, page))));
    files.push(f);
  }
  // dup.py's group COUNT is not a stable number (checkiso's note); what is stable is where
  // the clones are -- a clone inside one file, or one reaching into the shared core, is
  // code that wants sharing. "Reaching into" is a statement about the files a page LOADS:
  // common.js and physics.js for both, solver2d.js for the 2D page only. The 2D/3D solver
  // twins dup.py has always reported are therefore still legal as solver2d.js <-> rmhd3d.js
  // -- the 3D page does not load solver2d.js, and the move changed nothing about them.
  const dp = sh("python3", [path.join(__dirname, "dup.py")].concat(files));
  const groups = (dp.stdout || "").split("\n").filter(l => l.indexOf("CLONE") === 0)
    .map(l => l.split("):")[1].split(";").map(s => s.trim().split(":")[0]));
  const bads = groups.filter(g => new Set(g).size !== 2 ||
                                  g.some(f => f === "common.js" || f === "physics.js"));
  ok("dup.py: no clone inside a file, none reaching into common.js / physics.js",
     dp.status === 0 && bads.length === 0,
     groups.length + " groups" + (bads.length ? "; BAD: " + bads[0].join(" ") : ""));
  // ... and specifically: the extraction left no copy behind in the page that loads it
  const left = groups.filter(g => g.indexOf("solver2d.js") >= 0 && g.indexOf("rmhd2d.js") >= 0);
  ok("  ... and no clone between solver2d.js and the page that loads it",
     left.length === 0, left.length ? left[0].join(" ") : "the move left no copy behind");
}

// ---------------------------------------------------------------------------
console.log("2. the emitted WGSL is byte-identical to " + BASE + ", on both pages");
// ---------------------------------------------------------------------------
const bd = baseDir();
if (!bd) ok("base " + BASE + " is readable (git show)", false, "git show failed");
else {
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const b = dumpKernels(bd, page, "base");
    const base = b.k, k = cur[page].k;
    const keys = new Set(Object.keys(base).concat(Object.keys(k)));
    const moved = [], gone = [], added = new Set();
    for (const key of keys) {
      if (base[key] === k[key]) continue;
      // prepGrads is four per-pair emissions since FFTPERF_PLAN 2C (dispoffsets.js)
      if (OFF.isGradsLabel(key) || OFF.isChunkLabel(key)) continue;
      if (base[key] === undefined) { added.add(key.split(" :: ")[1]); continue; }
      if (k[key] === undefined) { gone.push(key); continue; }
      moved.push(key);
    }
    const CH = OFF.chunkAudit(base, k);
    for (const n of CH.added) added.add(n);
    ok(page + ": every kernel that existed at " + BASE + " is byte-identical",
       moved.length === 0 && gone.length === 0,
       moved.length + " moved, " + gone.length + " vanished" +
       (moved.length ? " (" + moved[0] + ")" : ""));
    ok("  ... but for prepGrads, chunked into its four pairs",
       CH.bad.length === 0 && CH.reduced.length + CH.added.length > 0,
       CH.bad[0] || CH.reduced.length + " emissions chunked");
    const want = ADDED[page].slice().sort().join(",");
    ok("  ... and it adds exactly [" + (want || "nothing") + "]",
       Array.from(added).sort().join(",") === want, Array.from(added).sort().join(",") || "nothing");
    // belt and braces: the WHOLE dump, kernel order and headers included -- every kernel
    // the chunk allowance does NOT name, since a chunked emission has no base section to
    // hash against (the leg above is what pins those, text for text)
    const hb = shaNoChunk(b.file), hc = shaNoChunk(cur[page].file);
    ok("  ... and the whole dump hashes the same", hb === hc, hc.slice(0, 16) + " vs " + hb.slice(0, 16));
  }
}

// ---------------------------------------------------------------------------
console.log("3. the RNG reference is unmoved");
// ---------------------------------------------------------------------------
{
  const rng = env2d.run(`function(){
    const g = new Gauss(7), a = [];
    for (let i = 0; i < 64; i++) a.push(g.next());
    return a;
  }`);
  const h = sha(Buffer.from(Float64Array.from(rng).buffer));
  ok("64 Gauss(7) draws, hashed", h === RNG_SHA, h.slice(0, 16) + " vs " + RNG_SHA.slice(0, 16));
}

// ---------------------------------------------------------------------------
console.log("4. extraction shape: what is where, and no module syntax anywhere");
// ---------------------------------------------------------------------------
{
  const in2d = inline(HTML2D);
  const stale = DEFS.filter(h => in2d.split("\n").indexOf(h) >= 0);
  ok("rmhd2d.html's inline script declares none of the three", stale.length === 0, stale.join(" | ") || "gone");
  // solver2d.js's top-level declarations, from the AST: exactly those three names and
  // exactly three declaration statements -- a stray top-level const / let / var / async
  // function is a name the app would have to know about, and fails here.
  if (!parse) ok("acorn is installed (npm i acorn in devtools/, or ACORN=<path-to-acorn.mjs>)", false);
  else {
    const ast = parse(SOLVER, { ecmaVersion: 2022 });
    const decls = ast.body.filter(n => n.type === "FunctionDeclaration" ||
                                       n.type === "ClassDeclaration" || n.type === "VariableDeclaration");
    // everything else at top level, the "use strict" directive excepted
    const other = ast.body.filter(n => decls.indexOf(n) < 0 &&
                                       !(n.type === "ExpressionStatement" && n.directive));
    const got = [...topNames(SOLVER)].sort().join(",");
    const want = DEFS.map(h => h.replace(/^(class|function)\s+/, "").replace(/[\s(].*$/, "")).sort().join(",");
    ok("solver2d.js's top-level names are EXACTLY those three, and nothing else",
       got === want && decls.length === 3 && other.length === 0,
       got + "; " + decls.length + " declarations, " + other.length + " other top-level statements");
  }
  // the tag, and where it sits: after physics.js's and before the page's own <script>
  const tag = '<script src="solver2d.js"></script>';
  const n = HTML2D.split(tag).length - 1;
  const iPhys = HTML2D.indexOf('<script src="physics.js">');
  const iTag = HTML2D.indexOf(tag);
  const iInline = HTML2D.indexOf("<script>\n");
  ok("rmhd2d.html carries exactly one solver2d.js tag", n === 1, n + " tags");
  ok("  ... after physics.js's and before the inline script",
     iTag > iPhys && iTag < iInline, "physics " + iPhys + ", solver2d " + iTag + ", inline " + iInline);
  ok("rmhd3d.html does not load solver2d.js", HTML3D.indexOf("solver2d.js") < 0);
  // Chrome blocks module scripts from file://, so the whole set stays classic. Static
  // `import x` / `import "x"`, the dynamic `import(` and the bare `import{` form, and any
  // `export` at statement position -- deliberately loose (a comment saying "import (" would
  // fire), since the safe direction here is over-firing.
  const MODSYN = [/type=["']module["']/, /(^|[^\w.$])import\s*[({]/m,
                  /(^|[^\w.$])import\s+["'*{]/m, /(^|[^\w.$])import\s+[\w$]+\s+from\s/m,
                  /^\s*export[\s{*]/m];
  const mod = { test: s => MODSYN.some(re => re.test(s)) };
  const units = { "common.js": read(path.join(dir, "common.js")),
                  "physics.js": read(path.join(dir, "physics.js")),
                  "solver2d.js": SOLVER,
                  "rmhd2d.html (inline)": in2d,
                  "rmhd3d.html (inline)": inline(HTML3D) };
  const bads = Object.keys(units).filter(u => mod.test(units[u]));
  ok("no module syntax in any of the three js files or the two inline scripts",
     bads.length === 0 && !mod.test(HTML2D) && !mod.test(HTML3D), bads.join(", ") || "classic scripts only");
}

// ---------------------------------------------------------------------------
console.log("5. the move is verbatim: each definition byte-identical to " + BASE + "'s");
// ---------------------------------------------------------------------------
if (bd) {
  const baseInline = inline(read(path.join(bd, "rmhd2d.html")));
  for (const h of DEFS) {
    const a = defText(baseInline, h), raw = defText(SOLVER, h);
    // `class Solver` is compared with the ONE recorded allowance stripped; the other two
    // definitions are still compared raw, and must never need one.
    const st = raw === null ? null
             : ALLOWED[h] ? stripBlocks(raw, ALLOWED[h]) : { txt: raw, miss: [] };
    const b = st && st.txt;
    ok("  " + h.replace(/ \{$/, "") + ": byte-identical to its text at " + BASE,
       a !== null && b !== null && a === b,
       a === null ? "not found at " + BASE : b === null ? "not found in solver2d.js"
         : a === b ? a.split("\n").length + " lines" : "differs");
    if (!ALLOWED[h]) continue;
    // ... and the allowance is neither stale nor vacuous: every recorded block is really
    // in the file, at its recorded place, and putting base's lines back really did change
    // something
    ok("    ... and the recorded allowance is the WHOLE of the difference",
       !!st && st.miss.length === 0 && raw !== a && b === a,
       !st ? "no text" : st.miss.length ? "allowance: " + st.miss.join(" | ")
         : raw === a ? "the allowance is vacuous -- nothing to strip"
         : st.cut + " lines allowed, each at its recorded place");
  }
}

// ---------------------------------------------------------------------------
console.log("6. reusability: solver2d.js resolves against common.js + physics.js alone");
// ---------------------------------------------------------------------------
{
  // names.mjs checks solver2d.js as a unit of its own, with rmhd2d.html's inline script
  // deliberately NOT in scope -- that line is the proof a second page can load it.
  const r = node([path.join(__dirname, "names.mjs"), dir]);
  const out = (r.stdout || "").split("\n");
  // its report line, read as a whole: the name count AND the verdict, not a substring
  const line = out.filter(l => /^\s*solver2d\.js:/.test(l)).pop() || "";
  const m = /^\s*solver2d\.js: (\d+) top-level names, (.*)$/.exec(line);
  const redecl = out.filter(l => /solver2d\.js REDECLARES/.test(l));
  ok("names.mjs: solver2d.js resolves against common.js + physics.js + builtins alone",
     r.status === 0 && !!m && m[1] === "3" && m[2].trim() === "every free identifier resolves" &&
     redecl.length === 0 && !/UNRESOLVED/.test(line),
     line.trim() || "no solver2d.js line" + (redecl.length ? "; " + redecl[0].trim() : ""));
}

// ---------------------------------------------------------------------------
console.log("7. pages.yml wiring");
// ---------------------------------------------------------------------------
{
  const p = path.join(root, ".github", "workflows", "pages.yml");
  const y = fs.existsSync(p) ? read(p) : "";
  // the `for f in ... ; do` statement itself (line continuations included), so a match
  // cannot be satisfied by the sed block further down
  const iFor = y.indexOf("for f in");
  const iDo = iFor < 0 ? -1 : y.indexOf("; do", iFor);
  const list = iFor >= 0 && iDo > iFor ? y.slice(iFor, iDo) : "";
  ok("pages.yml: solver2d.js is in the MISSING check list",
     list.indexOf("solver2d.js") >= 0, list ? list.replace(/\s+/g, " ").trim() : "no `for f in ... ; do`");
  ok("pages.yml: solver2d.js is cache-busted like common.js / physics.js",
     y.indexOf('s|src=\\"solver2d.js\\"|src=\\"solver2d.js?v=$v\\"|') >= 0);
  ok("pages.yml: game.html is pruned from the deploy", /rm -f\s+_site\/webgpu\/game\.html/.test(y));
}

// ---------------------------------------------------------------------------
console.log("8. both pages still boot to the end of the self-test path");
// ---------------------------------------------------------------------------
for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
  const r = node([path.join(__dirname, "bootstub.js"), dir, page]);
  const out = (r.stdout || "") + (r.stderr || "");
  ok(page + ": bootstub exits 0", r.status === 0, lastLine(r));
  ok("  ... and the self-test path ran to completion",
     out.indexOf("self-test path ran to completion") >= 0);
}

console.log(bad ? "\n" + bad + " solver2d check(s) FAILED" : "\nall solver2d checks passed");
})().catch(e => {
  bad++;
  console.log("  FAIL  checksolver2d threw: " + (e && e.message) + "\n" +
              ((e && e.stack) || "").split("\n").slice(0, 4).join("\n"));
}).finally(() => {
  // the base checkout and the WGSL dumps are scratch: drop them on every route out
  try { fs.rmSync(tmp, { recursive: true, force: true }); } catch (e) { /* best effort */ }
  process.exit(bad ? 1 : 0);
});
