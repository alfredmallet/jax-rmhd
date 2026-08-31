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
//      unmoved and the additions are EXACTLY the recorded list (nothing on the 3D page,
//      the blob forcing's `blobBuild` on the 2D one); plus the whole dump's sha256 against
//      the base dump's over every section neither allowance names, belt and braces.
//   3  the RNG reference unmoved (64 Gauss(7) draws, hashed), the standing rule.
//   4  extraction shape: the three definitions are gone from the inline script and are
//      solver2d.js's only top-level ones, the page loads it after physics.js and before
//      its own script, the 3D page does not load it at all, and no module syntax reaches
//      any of it (Chrome blocks module scripts from file://).
//   5  the verbatim move: each definition's text equals the BASE commit's, byte for byte
//      -- `class Solver` and `buildShaders` after their RECORDED allowances are stripped
//      (IO_PLAN item 4's field export, FFTPERF_PLAN 2C's gradient chunking, c81527d's
//      spacebar blob forcing).
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
// `blobBuild` (c81527d) is the one entry that is not the refactor's: the spacebar blob
// forcing's k-space transform of the placed gaussians, emitted on the 2D page beside `ou`
// and `scale`. It is an ADDITION and nothing else -- the leg above still demands that every
// kernel BASE had is byte-identical, so `ou`, `scale` and the whole step chain are unmoved.
// The JavaScript that drives it is leg 5's business (BLOB_SHADERS / BLOB_SOLVER below).
const ADDED = { "rmhd2d.html": ["blobBuild"], "rmhd3d.html": [] };
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
// gradient chain runs one CHUNK of (x, y) pairs at a time now and each page ships its own
// chunk list, which touches the two definitions this file pins -- `buildShaders` emits one
// prepGrads per chunk, and `class Solver` sizes the k-space stack by the chunk, builds a
// pipeline and a row-kernel window per chunk, and encodes the chain in one method the RHS
// calls. The 2D page's list is ONE whole-stack chunk, so its emitted WGSL is unmoved (the
// byte-identity leg above compares it plainly); what moves is this JavaScript. Recorded as
// REPLACEMENTS (`was` = the lines at BASE, `lines` = the lines now) at their base
// positions, so leg 5 puts base's text back and then still demands it byte for byte: any
// OTHER change to either definition fails, and a block that has stopped being there fails
// as a stale allowance.
const FFT2C_SHADERS = [
  { tag: "the per-chunk prepGrads emissions (the kernel list comment)", at: 53,
    was: ["  //   prepGrads   perpendicular i*k gradients of phi, psi, vort, jpar"],
    lines: [
   "  //   prepGrads   perpendicular i*k gradients of phi, psi, vort, jpar -- one emission",
   "  //               per chunk of this page's chunk list (physics.js GRAD_CHUNKS_2D)"] },
  { tag: "... and the emission itself", at: 56,
    was: ["  S.prepGrads = prepGradsWGSL(C);"],
    lines: [
   "  GRAD_CHUNKS_2D.forEach((ch, i) => {",
   "    S[\"prepGrads\" + gradChunkSuffix(GRAD_CHUNKS_2D, i)] =",
   "      prepGradsWGSL(Object.assign({}, C, { gchunk: ch }));",
   "  });"] }
];
const FFT2C_SOLVER = [
  { tag: "the chunk's lane count (_buildBuffers)", at: 40, lines: [
   "    const gcx = 2 * Math.max.apply(null, GRAD_CHUNKS_2D.map(ch => ch.length)); // lanes per chunk"] },
  { tag: "... which sizes the gradient stack (_buildBuffers)", at: 44,
    was: ["      gradsK: d.createBuffer({ size: 8 * cx, usage: SQ }),",
          "      specTmp: d.createBuffer({ size: 8 * cx, usage: SQ }),"],
    lines: [
   "      // the gradient chain transforms one CHUNK at a time, so the k-space stack and the",
   "      // column pass's target hold the widest chunk's lanes (GRAD_CHUNKS_2D)",
   "      gradsK: d.createBuffer({ size: gcx * cx, usage: SQ }),",
   "      specTmp: d.createBuffer({ size: gcx * cx, usage: SQ }),"] },
  { tag: "one prepGrads pipeline per chunk (_buildPipelines)", at: 209,
    was: ["      prepGrads: cp(S.prepGrads, \"prepGrads\"), bracket: cp(S.bracket, \"bracket\"),"],
    lines: [
   "      prepGrads: GRAD_CHUNKS_2D.map((ch, i) => {",
   "        const n = \"prepGrads\" + gradChunkSuffix(GRAD_CHUNKS_2D, i);",
   "        return cp(S[n], n);",
   "      }),",
   "      bracket: cp(S.bracket, \"bracket\"),"] },
  { tag: "their bind groups, and the row kernel's per-chunk window (_buildPipelines)", at: 240,
    was: ["      prepGrads: bg(this.pl.prepGrads, [B.fields, B.gridA, B.gradsK]),",
          "      colsInvGrads: bg(this.pl.colsInv, [B.gradsK, B.specTmp]),",
          "      rowsC2RGrads: bg(this.pl.rowsC2R, [B.specTmp, B.realGrads]),"],
    lines: [
   "      prepGrads: this.pl.prepGrads.map(p => bg(p, [B.fields, B.gridA, B.gradsK])),",
   "      colsInvGrads: bg(this.pl.colsInv, [B.gradsK, B.specTmp]),",
   "      // one target per chunk: the same row kernel, its store landing in the chunk's own",
   "      // lanes of realGrads through the binding's window (physics.js gradChunkWindow)",
   "      rowsC2RGrads: GRAD_CHUNKS_2D.map(ch => bg(this.pl.rowsC2R,",
   "        [B.specTmp, Object.assign({ buffer: B.realGrads }, gradChunkWindow(this.nr, ch))])),"] },
  { tag: "the encodeGrads method", at: 392, lines: [
   "  // the eight perpendicular gradients, into realGrads (lanes 0,1 = grad phi, 2,3 = grad",
   "  // psi, 4..7 = grad vorticity / current): one chunk at a time, each prepGrads writing the",
   "  // chunk's lanes of the k-space stack and the two inverse passes carrying them to the",
   "  // chunk's own lanes of realGrads. The only place the chain is encoded.",
   "  encodeGrads(pass) {",
   "    const nm = this.g.nm, nky = this.g.nky, nx = this.g.nx;",
   "    GRAD_CHUNKS_2D.forEach((ch, i) => {",
   "      const lanes = 2 * ch.length;",
   "      pass.setPipeline(this.pl.prepGrads[i]); pass.setBindGroup(0, this.bg.prepGrads[i]);",
   "      pass.dispatchWorkgroups(Math.ceil(nm / 64));",
   "      pass.setPipeline(this.pl.colsInv); pass.setBindGroup(0, this.bg.colsInvGrads);",
   "      pass.dispatchWorkgroups(lanes * nky);",
   "      pass.setPipeline(this.pl.rowsC2R); pass.setBindGroup(0, this.bg.rowsC2RGrads[i]);",
   "      pass.dispatchWorkgroups(lanes * nx);",
   "    });",
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
// THE THIRD SANCTIONED EDIT, in the same idiom: c81527d's spacebar gaussian blob forcing,
// the 2D page's interactive alternative to the OU shell. It adds ONE kernel (`blobBuild`, the
// `ADDED` list above) plus the JavaScript that drives it: the mode flag and its packed
// upload, the blobs buffer, the kernel's pipeline and bind group, the two `setBlob*`
// methods, and the branches in `_uploadIC` and `step` that dispatch `blobBuild` in place of
// `ou` + `scale`. Blob mode is OFF by default and every branch keeps BASE's own lines as its
// `else`, so the OU path is untouched -- which the `was` lines below say line for line.
// Recorded, like the two allowances above, as the exact lines and WHERE they go.
const BLOB_SHADERS = [
  { tag: "the blobBuild emission", at: 180,
    lines: [
   "  // ---- forcing, blob mode: the k-space transform of the placed gaussians --",
   "  // emitted at every preset whether or not the page ever turns blob mode on, so the",
   "  // kernel text is a fixed function of the grid; BLOB_FORCE_MAX is a compile-time",
   "  // constant and never a UI number (physics.js)",
   "  S.blobBuild = blobBuildWGSL(Object.assign({}, C, { nblob: BLOB_FORCE_MAX }));"] },
];
const BLOB_SOLVER = [
  { tag: "blobMode / _blobs, the mode flag and its packed upload (constructor)", at: 29,
    lines: [
   "    // blob forcing (BLOBFORCE): off unless the page asks, so the default solver -- the",
   "    // self-test's included -- steps the OU path exactly as before. `_blobs` is the packed",
   "    // upload, 4 floats (x0, y0, sigma, w) per slot, z+ slots then z-.",
   "    this.blobMode = false;",
   "    this._blobs = new Float32Array(2 * BLOB_FORCE_MAX * 4);"] },
  { tag: "the blobs buffer (_buildBuffers)", at: 56,
    lines: [
   "      // blob forcing: 2 * BLOB_FORCE_MAX vec4 (x0, y0, sigma, w), z+ half then z-",
   "      blobs: d.createBuffer({ size: Math.max(16, 2 * BLOB_FORCE_MAX * 16), usage: SQ }),"] },
  { tag: "the blobBuild pipeline (_buildPipelines)", at: 214,
    was: ["      ou: cp(S.ou, \"ou\"), scale: cp(S.scale, \"scale\"), icFinish: cp(S.icFinish, \"icFinish\"),"],
    lines: [
   "      ou: cp(S.ou, \"ou\"), scale: cp(S.scale, \"scale\"), blobBuild: cp(S.blobBuild, \"blobBuild\"),",
   "      icFinish: cp(S.icFinish, \"icFinish\"),"] },
  { tag: "the blobBuild bind group (_buildPipelines)", at: 266,
    lines: [
   "      blobBuild: bg(this.pl.blobBuild, [B.forcing, B.gridA, B.gridB, B.blobs]),"] },
  { tag: "the sc[4]/sc[5] restore after an IC upload (_uploadIC)", at: 388,
    lines: [
   "    // the scalars were zeroed above and `scale` has just written the OU normalization",
   "    // into sc[4]/sc[5]; blob mode carries its amplitude in the modes themselves, so put",
   "    // the two scales back at 1 or the first frames would force with nothing",
   "    if (this.blobMode) d.queue.writeBuffer(this.buf.scalars, 16, new Float32Array([1, 1]));"] },
  { tag: "setBlobMode + setBlobs", at: 432,
    lines: [
   "  // ---- blob forcing (BLOBFORCE) -------------------------------------------",
   "  // Blob mode REPLACES the OU shell: `blobBuild` writes the whole forcing buffer from the",
   "  // placed gaussians, and the step dispatches neither `ou` nor `scale`, so sc[4]/sc[5]",
   "  // stay at the 1 written here and each blob's amplitude is its own w. The forcing buffer",
   "  // is cleared on either transition -- the modes it holds mean nothing in the other mode,",
   "  // and with sc[4]/sc[5] at 1 a leftover OU envelope would land on the fields whole.",
   "  setBlobMode(on) {",
   "    const was = this.blobMode, d = this.device;",
   "    this.blobMode = !!on;",
   "    if (was !== this.blobMode) {              // only a real transition throws the modes away",
   "      const enc = d.createCommandEncoder();",
   "      enc.clearBuffer(this.buf.forcing);",
   "      d.queue.submit([enc.finish()]);",
   "    }",
   "    if (this.blobMode) d.queue.writeBuffer(this.buf.scalars, 16, new Float32Array([1, 1]));",
   "  }",
   "",
   "  // list: [{ x, y, sigma, amp, pol }], x/y in box coordinates, sigma in the same length",
   "  // units, `amp` the PEAK |grad f| the blob is to force at and `pol` +1 for the z+ channel",
   "  // / -1 for z-. At most BLOB_FORCE_MAX per channel -- the rest are dropped. `amp` is a",
   "  // velocity forcing rate, so the potential peak is icBlobPeak(amp, sigma) (common.js):",
   "  // growing sigma then inflates the blob without changing what it forces the flow at.",
   "  // An empty list zeroes every slot, i.e. zero forcing.",
   "  setBlobs(list) {",
   "    const q = this.p, a = this._blobs, n = BLOB_FORCE_MAX;",
   "    // w = P * 2*pi*sigma^2 * (nx*ny)/(Lx*Ly): the continuous transform at k = 0, times",
   "    // the unnormalized DFT's nx*ny and divided by the cell area (see blobBuildWGSL)",
   "    const w0 = 2 * Math.PI * (q.nx * q.ny) / (q.Lx * q.Ly);",
   "    a.fill(0);",
   "    const used = [0, 0];",
   "    for (const b of (list || [])) {",
   "      const h = b.pol < 0 ? 1 : 0;",
   "      if (used[h] >= n) continue;",
   "      const o = 4 * (h * n + used[h]++), sg = +b.sigma;",
   "      a[o] = +b.x; a[o + 1] = +b.y; a[o + 2] = sg;",
   "      a[o + 3] = icBlobPeak(+b.amp, sg) * sg * sg * w0;",
   "    }",
   "  }",
   ""] },
  { tag: "the step() header comment", at: 457,
    was: ["  // one full step: LSRK33 (lagged scales), then the OU advance and the new scale"],
    lines: [
   "  // one full step: LSRK33 (lagged scales), then the OU advance and the new scale --",
   "  // or, in blob mode, the blob rebuild in place of all three"] },
  { tag: "the blobs upload in place of drawNoise (step)", at: 461,
    was: ["    this.drawNoise();"],
    lines: [
   "    if (this.blobMode) d.queue.writeBuffer(this.buf.blobs, 0, this._blobs);",
   "    else this.drawNoise();"] },
  { tag: "the blob dispatch in place of ou + scale (step)", at: 471,
    was: ["    p.setPipeline(this.pl.ou); p.setBindGroup(0, this.bg.ou);",
           "    p.dispatchWorkgroups(Math.ceil(2 * this.ns / 64));",
           "    p.setPipeline(this.pl.scale); p.setBindGroup(0, this.bg.scale); p.dispatchWorkgroups(1);"],
    lines: [
   "    if (this.blobMode) {",
   "      p.setPipeline(this.pl.blobBuild); p.setBindGroup(0, this.bg.blobBuild);",
   "      p.dispatchWorkgroups(Math.ceil(2 * this.g.nm / 64));",
   "    } else {",
   "      p.setPipeline(this.pl.ou); p.setBindGroup(0, this.bg.ou);",
   "      p.dispatchWorkgroups(Math.ceil(2 * this.ns / 64));",
   "      p.setPipeline(this.pl.scale); p.setBindGroup(0, this.bg.scale); p.dispatchWorkgroups(1);",
   "    }"] },
];
// every recorded block per pinned definition, in BASE-line order (which is what lets each
// block's found index BE its base line -- see stripBlocks)
const ALLOWED = {
  "function buildShaders(g) {": FFT2C_SHADERS.concat(BLOB_SHADERS).sort((a, b) => a.at - b.at),
  "class Solver {": FFT2C_SOLVER.concat(IO4_INSERTS, BLOB_SOLVER).sort((a, b) => a.at - b.at)
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
// a dump's sha with the sections the chunk allowance names on THIS page taken out of it,
// on both sides -- on a page that chunks nothing, nothing comes out. The RECORDED
// additions come out too, for the same reason: a kernel BASE never emitted has no section
// on the other side to hash against, so leaving one in would make this leg fail on any
// addition at all, whatever the leg above already said about it. Only the kernels NAMED in
// this page's `ADDED` are skipped -- an unrecorded new kernel still moves the hash, which
// is the whole of what this leg is for.
function shaNoChunk(file, page) {
  const parts = read(file).split(/^########## (.*) ##########$/m);
  let out = parts[0];
  for (let i = 1; i < parts.length; i += 2)
    if (!OFF.isChunked(page, parts[i]) &&
        ADDED[page].indexOf(parts[i].split(" :: ")[1]) < 0)
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
      // where the page chunks prepGrads (FFTPERF_PLAN 2C, dispoffsets.js) the audit below
      // is what judges its emissions
      if (OFF.isChunked(page, key)) continue;
      if (base[key] === undefined) { added.add(key.split(" :: ")[1]); continue; }
      if (k[key] === undefined) { gone.push(key); continue; }
      moved.push(key);
    }
    const CH = OFF.chunkAudit(page, base, k);
    for (const n of CH.added) added.add(n);
    ok(page + ": every kernel that existed at " + BASE + " is byte-identical",
       moved.length === 0 && gone.length === 0,
       moved.length + " moved, " + gone.length + " vanished" +
       (moved.length ? " (" + moved[0] + ")" : ""));
    ok("  ... and prepGrads is " + BASE + "'s own text over this page's chunk list, "
       + JSON.stringify(OFF.CHUNKS[page]),
       CH.bad.length === 0 && CH.reduced.length + CH.added.length > 0,
       CH.bad[0] || CH.reduced.length + " emissions");
    const want = ADDED[page].slice().sort().join(",");
    ok("  ... and it adds exactly [" + (want || "nothing") + "]",
       Array.from(added).sort().join(",") === want, Array.from(added).sort().join(",") || "nothing");
    // belt and braces: the WHOLE dump, kernel order and headers included -- every kernel
    // the chunk allowance does NOT name, since a chunked emission has no base section to
    // hash against (the leg above is what pins those, text for text)
    const hb = shaNoChunk(b.file, page), hc = shaNoChunk(cur[page].file, page);
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
    // a definition with recorded allowances is compared with them stripped (their base
    // lines put back); one without is compared raw, and makeGrid must never need any.
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
