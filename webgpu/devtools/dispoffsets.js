// The one move `prepDisp` has made since checkiso's, checkeigf's and check2dspec's base
// commits: 268100f (2026-08-18, "webgpu perp offsets"), the 2D display card's x/y offset.
// All three gates pin the emitted WGSL against a commit older than that and all three need
// the SAME allowance, so it is written once, here, and each of them names the commit that
// moved their base line -- a stale pin has to stay diagnosable.
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

module.exports = { COMMIT, PAGE, KERNEL, PHASE, applied, isMove, moved };
