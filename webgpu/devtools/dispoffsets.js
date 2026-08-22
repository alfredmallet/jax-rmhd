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
// `prepGrads` (and the sweep's band-gated twin) used to be ONE kernel writing all eight
// i*k gradient lanes. It is now four emissions of one template, each computing a single
// field's (x, y) pair into lanes 0-1 -- so every gate that pins physics WGSL against a
// commit older than 2C sees one kernel vanish and four appear, at every preset.
//
// The substitution, per pair k, is a pure LINE selection out of the base text:
//   * the `let` lines: only the pair's own source field, plus (vort / jpar) the multiply
//     that builds it -- the other two are dropped;
//   * the eight `outg[...]` writes: only the pair's two, and they take the LHS of lanes 0
//     and 1, so the pair lands at the start of a two-lane target.
// Everything else -- bindings, workgroup size, the m / g preamble, the band block where
// the emission has one -- is base's text, line for line. So a gate can keep its base
// commit and still fail on any OTHER change to prepGrads, which is the whole point.
const G_KERNELS = ["prepGrads", "prepGradsBand"];
const NPAIR = 4;
const LET = /^  let (\w+): vec2<f32> = /;
const OUT = /^(  outg\[[^\]]*\] *)= vec2<f32>\(-g\.([xy]) \* (\w+)\.y, +g\.\2 \* \3\.x\);$/;
// the state field each written field is read from (vort is built out of phi, jpar of psi)
const OF_SRC = { phi: "phi", psi: "psi", vort: "phi", jpar: "psi" };

// base's eight-lane prepGrads text -> pair k's text, or null if `base` is not one
function gpairApplied(base, k) {
  const L = String(base).split("\n");
  const out = [];
  for (let i = 0; i < L.length; i++) if (OUT.test(L[i])) out.push(i);
  if (out.length !== 2 * NPAIR || !(k >= 0 && k < NPAIR)) return null;
  const of = OUT.exec(L[out[2 * k]])[3];
  if (!OF_SRC[of] || OUT.exec(L[out[2 * k + 1]])[3] !== of) return null;
  const keep = new Set([OF_SRC[of], of]);
  const res = [];
  for (let i = 0; i < L.length; i++) {
    const ml = LET.exec(L[i]);
    if (ml) { if (keep.has(ml[1])) res.push(L[i]); continue; }
    const j = out.indexOf(i);
    if (j < 0) { res.push(L[i]); continue; }
    if (j === 2 * k || j === 2 * k + 1) res.push(OUT.exec(L[out[j - 2 * k]])[1]
                                                + L[i].slice(L[i].indexOf("= ")));
  }
  return res.join("\n");
}

const kernelOf = label => String(label).split(" :: ").pop();
// the four dump labels one base label becomes
const gpairKeys = label => Array.from({ length: NPAIR }, (_, k) => label + k);
// "prepGrads2" -> "prepGrads" (a pair emission's kernel), else null
function chunkName(name) {
  const m = /^(\w+?)(\d)$/.exec(String(name));
  return m && G_KERNELS.indexOf(m[1]) >= 0 && +m[2] < NPAIR ? m[1] : null;
}
const isChunkLabel = label => !!chunkName(kernelOf(label));
const isGradsLabel = label => G_KERNELS.indexOf(kernelOf(label)) >= 0;

// audit two dumps (label -> text): every chunked kernel the BASE emitted must be gone and
// its four pairs present and EXACTLY the reduction; a pair emission whose base label does
// not exist at all is an ADDITION, named by its kernel so the gate's own ADDED list still
// judges it. `bad` is empty or says what is wrong, first offender first.
function chunkAudit(base, cur) {
  const bad = [], reduced = [], added = new Set();
  for (const label of Object.keys(base)) {
    if (!isGradsLabel(label)) continue;
    if (cur[label] !== undefined) { bad.push(label + ": still emitted whole"); continue; }
    const miss = gpairKeys(label).filter((c, k) => cur[c] !== gpairApplied(base[label], k));
    if (miss.length) bad.push(miss[0] + ": not " + label + "'s text reduced to that pair");
    else reduced.push(label);
  }
  for (const label of Object.keys(cur)) {
    if (isChunkLabel(label) && base[label.replace(/\d$/, "")] === undefined)
      added.add(chunkName(kernelOf(label)));
  }
  return { reduced: reduced, added: [...added], bad: bad };
}

module.exports = { COMMIT, PAGE, KERNEL, PHASE, applied, isMove, moved,
                   G_KERNELS, NPAIR, gpairApplied, gpairKeys, chunkName,
                   isChunkLabel, isGradsLabel, chunkAudit };
