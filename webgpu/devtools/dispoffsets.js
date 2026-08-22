// The recorded moves the WGSL byte-identity gates (checkiso, checkeigf, check2dspec,
// checksolver2d) share: each pins the emitted text against a base commit older than the
// move, so the allowance is written once, here, and each of them names the commit that
// moved its base line -- a stale pin has to stay diagnosable. There are two: prepDisp's
// display offset (below) and prepGrads' gradient chunking (further down).
//
// 1. The one move `prepDisp` has made since those base commits: 268100f (2026-08-18,
// "webgpu perp offsets"), the 2D display card's x/y offset.
//
// 268100f makes exactly two edits to the 2D page's prepDisp text (the offset's own gate is
// checkoff.js; the 3D app sets no `shift` and its prepDisp does not move at all):
//   1. the Mode struct becomes physics.js's MODE_SHIFT_STRUCT -- the band's two padding
//      words are the shift vector. The 2D app passes `shift` unconditionally, so this is
//      also the struct emitted where the BAND is gated off, and there the pre-offset app
//      emitted the plain four-word MODE_STRUCT: both spellings map to the shift one.
//   2. the translation phase e^{-i k.S}, eight lines before the writes, itself gated at run
//      time on `md.sh != 0`.
// Anything else in prepDisp, and any of this going missing, is a change the gates must see.
"use strict";
const COMMIT = "268100f";
const PAGE = "rmhd2d.html";             // the page that offers the offset
const KERNEL = "prepDisp";              // ... in this one kernel, at every preset

const MODE_PLAIN = "struct Mode { mode: u32, zslice: u32, cmap: u32, pad: u32 };";
const MODE_BAND =
  "struct Mode { mode: u32, zslice: u32, cmap: u32, pad: u32, klo: f32, khi: f32, bpad: vec2<f32> };";
const MODE_SHIFT =
  "struct Mode { mode: u32, zslice: u32, cmap: u32, pad: u32, klo: f32, khi: f32, sh: vec2<f32> };";
const ANCHOR = "  outk[m] = v;";        // the phase goes in just before the writes
const PHASE = [
  "  // the display offset, as the translation phase e^{-i k.S}",
  "  if (md.sh.x != 0.0 || md.sh.y != 0.0) {",
  "    let ph: f32 = gridA[m].x * md.sh.x + gridA[m].y * md.sh.y;",
  "    let co: f32 = cos(ph);",
  "    let si: f32 = sin(ph);",
  "    v  = vec2<f32>( v.x * co +  v.y * si,  v.y * co -  v.x * si);",
  "    v2 = vec2<f32>(v2.x * co + v2.y * si, v2.y * co - v2.x * si);",
  "  }"];

// base prepDisp text -> the text 268100f makes of it, or null if the base is not one this
// describes (one Mode struct line in a known spelling, one anchor)
function applied(base) {
  const L = String(base).split("\n");
  const isStruct = l => l === MODE_PLAIN || l === MODE_BAND;
  if (L.filter(isStruct).length !== 1 || L.filter(l => l === ANCHOR).length !== 1) return null;
  const out = [];
  for (const l of L) {
    if (l === ANCHOR) out.push.apply(out, PHASE);
    out.push(isStruct(l) ? MODE_SHIFT : l);
  }
  return out.join("\n");
}

// is `cur` base's text plus 268100f, for this page and this dump label?
function isMove(page, label, base, cur) {
  return page === PAGE && label.split(" :: ").pop() === KERNEL && applied(base) === cur;
}

// the dump labels of `keys` that MUST carry the move, so it cannot silently vanish
function moved(page, keys) {
  return page !== PAGE ? [] : keys.filter(k => k.split(" :: ").pop() === KERNEL);
}

// ---------------------------------------------------------------------------
// 2. the second allowance: FFTPERF_PLAN 2C's gradient chunking
// ---------------------------------------------------------------------------
// `prepGrads` (and the sweep's band-gated twin) writes the eight i*k gradient lanes one
// CHUNK of (x, y) pairs at a time, and each page picks its own chunk list (§9.3): the 2D
// page does all four pairs in one dispatch -- which is BASE's kernel, byte for byte, under
// BASE's name, so nothing is excused there and the pin is plain identity -- and the 3D page
// one pair per dispatch, four emissions where the base commit had one.
//
// The substitution, per chunk, is a pure LINE selection out of the base text:
//   * the `let` lines: only the fields the chunk's pairs read or build, in first-use order;
//   * the eight `outg[...]` writes: only the chunk's, taking the LHS of lanes 0, 1, 2, ...
//     so the chunk lands at the start of its own target.
// Everything else -- bindings, workgroup size, the m / g preamble, the band block where the
// emission has one -- is base's text, line for line, and the whole-stack chunk [0,1,2,3]
// reproduces base exactly. So a gate can keep its base commit and still fail on any OTHER
// change to prepGrads, which is the whole point.
const G_KERNELS = ["prepGrads", "prepGradsBand"];
const NPAIR = 4;
// each page's chunk list, as it ships. A page whose list this does not describe fails the
// audit -- including a 2D page that has started emitting pairs.
const CHUNKS = { "rmhd2d.html": [[0, 1, 2, 3]], "rmhd3d.html": [[0], [1], [2], [3]] };
const LET = /^  let (\w+): vec2<f32> = /;
const OUT = /^(  outg\[[^\]]*\] *)= vec2<f32>\(-g\.([xy]) \* (\w+)\.y, +g\.\2 \* \3\.x\);$/;
// the state field each written field is read from (vort is built out of phi, jpar of psi)
const OF_SRC = { phi: "phi", psi: "psi", vort: "phi", jpar: "psi" };

// base's eight-lane prepGrads text -> the text of one chunk of pairs, or null if `base` is
// not an eight-lane emission or the chunk is not a set of pair indices
function chunkApplied(base, chunk) {
  const L = String(base).split("\n"), out = [];
  for (let i = 0; i < L.length; i++) if (OUT.test(L[i])) out.push(i);
  if (out.length !== 2 * NPAIR) return null;
  if (!Array.isArray(chunk) || !chunk.length ||
      chunk.some(k => !(k >= 0 && k < NPAIR)) || new Set(chunk).size !== chunk.length) return null;
  // the fields the chunk needs, in first-use order, and the lines that make them
  const of = chunk.map(k => OUT.exec(L[out[2 * k]])[3]);
  if (of.some((f, j) => !OF_SRC[f] || OUT.exec(L[out[2 * chunk[j] + 1]])[3] !== f)) return null;
  const want = [];
  of.forEach(f => { for (const n of [OF_SRC[f], f]) if (want.indexOf(n) < 0) want.push(n); });
  const lets = new Map();
  for (const l of L) { const m = LET.exec(l); if (m) lets.set(m[1], l); }
  const res = [];
  let done = false;
  for (let i = 0; i < L.length; i++) {
    if (LET.test(L[i])) {                       // the whole block, once, where it stood
      if (done) continue;
      done = true;
      for (const n of want) if (lets.has(n)) res.push(lets.get(n));
      continue;
    }
    const j = out.indexOf(i);
    if (j < 0) { res.push(L[i]); continue; }
    if (j !== 0) continue;                      // the writes go in as one block too
    chunk.forEach((k, jj) => {
      for (const h of [0, 1])
        res.push(OUT.exec(L[out[2 * jj + h]])[1] + L[out[2 * k + h]].slice(L[out[2 * k + h]].indexOf("= ")));
    });
  }
  return res.join("\n");
}

const kernelOf = label => String(label).split(" :: ").pop();
// the dump labels one base label becomes under a chunk list: a single chunk keeps the
// base name, so its emission is compared as the base kernel it still is
const chunkKeys = (label, chunks) =>
  chunks.map((ch, i) => label + (chunks.length > 1 ? String(i) : ""));
// "prepGrads2" -> "prepGrads" (a per-chunk emission's kernel), else null
function chunkName(name) {
  const m = /^(\w+?)(\d)$/.exec(String(name));
  return m && G_KERNELS.indexOf(m[1]) >= 0 && +m[2] < NPAIR ? m[1] : null;
}
const isChunkLabel = label => !!chunkName(kernelOf(label));
const isGradsLabel = label => G_KERNELS.indexOf(kernelOf(label)) >= 0;
// a label the chunking really moves ON THIS PAGE, i.e. one a byte-identity leg should hand
// to the audit below instead of comparing itself. A page that emits ONE chunk moves
// nothing: its kernel keeps base's name AND base's text, and is compared like any other.
function isChunked(page, label) {
  const chunks = CHUNKS[page];
  if (!chunks || chunks.length < 2) return false;
  return isGradsLabel(label) || isChunkLabel(label);
}

// audit two dumps (label -> text) against the page's own chunk list: every gradient kernel
// the BASE emitted must appear as exactly the emissions that list calls for, each EXACTLY
// base's text reduced to its chunk, and nothing else of that kernel may be emitted. A
// per-chunk emission whose base label does not exist at all is an ADDITION, named by its
// kernel so the gate's own ADDED list still judges it. `bad` is empty or says what is
// wrong, first offender first.
function chunkAudit(page, base, cur) {
  const chunks = CHUNKS[page], bad = [], reduced = [], added = new Set();
  if (!chunks) return { reduced: reduced, added: [], bad: ["no chunk list recorded for " + page] };
  for (const label of Object.keys(base)) {
    if (!isGradsLabel(label)) continue;
    const keys = chunkKeys(label, chunks);
    const stray = [label].concat(chunkKeys(label, [0, 1, 2, 3]))
      .filter(k => keys.indexOf(k) < 0 && cur[k] !== undefined);
    if (stray.length) { bad.push(stray[0] + ": emitted where the page's chunk list has no such kernel"); continue; }
    const miss = keys.filter((k, i) => cur[k] !== chunkApplied(base[label], chunks[i]));
    if (miss.length) bad.push(miss[0] + ": not " + label + "'s text over the chunk the page ships");
    else reduced.push(label);
  }
  for (const label of Object.keys(cur)) {
    if (isChunkLabel(label) && base[label.replace(/\d$/, "")] === undefined)
      added.add(chunkName(kernelOf(label)));
  }
  return { reduced: reduced, added: [...added], bad: bad };
}

module.exports = { COMMIT, PAGE, KERNEL, PHASE, applied, isMove, moved,
                   G_KERNELS, NPAIR, CHUNKS, chunkApplied, chunkKeys, chunkName,
                   isChunkLabel, isGradsLabel, isChunked, chunkAudit };
