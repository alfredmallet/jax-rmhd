// GATE: multi-display recording -- every open display, ONE video (IO_PLAN item 5).
//
// Section A is the TILER and the COMPOSITOR alone: synthetic byte patterns in, one
// composite out, every row asserted at its expected offset, for BOTH layouts (a row and
// a column), padded rows, the halved tiles, a source with no pixels, the label blit
// clipped to its own TILE (recCompose overwrites a bleed, so that one is asserted on
// recBlitPatch itself) and the take's reused composite scratch. Nothing there needs a
// page, a GPU or an encoder -- it is the arithmetic the whole feature rests on.
//
// Section B is the action on a booted page. It asserts on the FILE and on the take's own
// state, never on a handler having run, and it drives the failure modes that matter:
// an all-or-nothing slot (one source fails, the WHOLE slot goes and W.n does not move) in
// its three forms -- a resized source, a rejected map, and the buffer path latched off,
// where a composite has no canvas to fall back to and simply keeps dropping -- the
// isConfigSupported refusal (tiles halve, the card list does NOT), a field retyped under
// a live take (the caption follows the pixels), a card closed mid-recording, and --
// because the no-fork rule means the single-card recorder was refactored underneath --
// that a card's own `rec` still records exactly its own canvas, sync fallback included.
//
// Run: node devtools/checkrecall.js [dir]
"use strict";
const path = require("path");
const DIR = process.argv[2] || path.join(__dirname, "..");
let pass = 0, bad = 0;
function ok(name, cond, note) {
  if (cond) { pass++; console.log("  PASS  " + name + (note ? "   [" + note + "]" : "")); }
  else { bad++; console.log("  FAIL  " + name + (note ? "   [" + note + "]" : "")); }
}
async function boot(page) {
  const env = require("./stubenv")(DIR, page, "");
  for (let i = 0; i < 400 && !env.run("function(){ return !!solver; }"); i++)
    await new Promise(r => setTimeout(r, 0));
  return env;
}
const settle = () => new Promise(r => setTimeout(r, 5));

// ===========================================================================
// A. the tiler and the composite
// ===========================================================================
// A fake source is only ever asked for its canvas size, which is what makes this section
// pure: recTiles takes display cards, and "a display card" here is {cv:{width,height}}.
function tilerLegs(env) {
  console.log("\n=== A. the tiler and the composite ===");
  const lay = (n, w, h) => env.run(`function(n, w, h){
    const cs = []; for (let i = 0; i < n; i++) cs.push({ cv: { width: w, height: h } });
    const L = recTiles(cs);
    return L && { n: L.n, cols: L.cols, rows: L.rows, w: L.w, h: L.h, tw: L.tw, th: L.th,
                  step: L.step, at: L.tiles.map(t => [t.x, t.y]) }; }`, n, w, h);

  // 1. one row or one column, whichever comes out closer to square
  const r2 = lay(2, 512, 512);
  ok("two square displays go side by side, 1024x512",
     r2 && r2.cols === 2 && r2.rows === 1 && r2.w === 1024 && r2.h === 512 &&
     JSON.stringify(r2.at) === "[[0,0],[512,0]]", JSON.stringify(r2));
  const c2 = lay(2, 1024, 256);
  ok("two WIDE-box displays stack instead -- 1024x512 is squarer than 2048x256",
     c2 && c2.cols === 1 && c2.w === 1024 && c2.h === 512 &&
     JSON.stringify(c2.at) === "[[0,0],[0,256]]", JSON.stringify(c2));
  const r3 = lay(3, 512, 512);
  ok("three square displays are a row (the tie goes to side by side), 1536x512",
     r3 && r3.cols === 3 && r3.w === 1536 && r3.h === 512 &&
     JSON.stringify(r3.at) === "[[0,0],[512,0],[1024,0]]", JSON.stringify(r3));
  const c3 = lay(3, 1024, 256);
  ok("three wide-box displays are a column, 1024x768",
     c3 && c3.cols === 1 && c3.w === 1024 && c3.h === 768 &&
     JSON.stringify(c3.at) === "[[0,0],[0,256],[0,512]]", JSON.stringify(c3));
  const one = lay(1, 512, 512);
  ok("ONE card is the same object with one tile: the frame `rec` has always recorded",
     one && one.n === 1 && one.w === 512 && one.h === 512 && one.step === 1 &&
     JSON.stringify(one.at) === "[[0,0]]", JSON.stringify(one));

  // 2. sources that disagree about their size are ASSERTED, not scaled
  const mixed = env.run(`function(){
    return recTiles([{ cv: { width: 512, height: 512 } }, { cv: { width: 256, height: 256 } }]); }`);
  ok("cards of different pixel sizes refuse to tile rather than silently scaling one",
     mixed === null, JSON.stringify(mixed));
  const none = env.run("function(){ return recTiles([]); }");
  ok("no sources at all is no layout", none === null);
  // ... and a source with NO PIXELS is refused the same way rather than dropped out of
  // the list: a card list quietly one short is the silent cap this must not have
  const zero = env.run(`function(){ return [
    recTiles([{ cv: { width: 512, height: 512 } }, { cv: { width: 0, height: 0 } }]),
    recTiles([{ cv: { width: 0, height: 512 } }]),
    recTiles([{}])]; }`);
  ok("a source with no pixels (or no canvas) refuses the layout -- it is never dropped "
     + "quietly out of the card list",
     zero.length === 3 && zero.every(x => x === null), JSON.stringify(zero));

  // 3. the halved fallback: the TILES halve, the card list never does
  const half = env.run(`function(){
    const cs = [{ cv: { width: 512, height: 512 } }, { cv: { width: 512, height: 512 } }];
    const H = recHalve(recTiles(cs));
    return { n: H.n, w: H.w, h: H.h, tw: H.tw, th: H.th, step: H.step,
             at: H.tiles.map(t => [t.x, t.y]), again: recHalve(H) }; }`);
  ok("halving keeps BOTH tiles and halves their pixels: 512x256, step 2",
     half.n === 2 && half.step === 2 && half.tw === 256 && half.th === 256 &&
     half.w === 512 && half.h === 256 && JSON.stringify(half.at) === "[[0,0],[256,0]]",
     JSON.stringify(half));
  ok("... and it halves ONCE: there is no quarter-size fallback behind it", half.again === null);

  // 4. THE COMPOSITE. Two synthetic sources, a PADDED source row (the real
  //    copyTextureToBuffer case), every output row read back and compared with the source
  //    row it must be a copy of.
  //    Pixel bytes are a function of (source, x, y), so a tile at the wrong offset, a row
  //    at the wrong pitch and a swapped pair of sources are all separately visible.
  const compose = (sw, sh, bpr, step, label) => env.run(`function(sw, sh, bpr, step, label){
    const cs = [{ cv: { width: sw, height: sh } }, { cv: { width: sw, height: sh } }];
    let L = recTiles(cs);
    if (step === 2) L = recHalve(L);
    L.bpr = bpr;
    if (label) L.tiles[1].label = { w: label[0], h: label[1],
      bytes: new Uint8Array(4 * label[0] * label[1]).map(() => 7) };
    const ranges = [];
    for (let i = 0; i < 2; i++) {
      const b = new Uint8Array(bpr * sh);
      for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) for (let k = 0; k < 4; k++)
        b[y * bpr + 4 * x + k] = (1 + i * 97 + x * 3 + y * 11 + k) & 255;
      ranges.push(b.buffer);
    }
    const out = recCompose(L, ranges);
    return { w: L.w, h: L.h, cols: L.cols, at: L.tiles.map(t => [t.x, t.y]),
             out: Array.from(out) }; }`, sw, sh, bpr, step, label || null);
  // the value the source SHOULD hold, mirrored here rather than read back out of the page
  const srcByte = (i, x, y, k) => (1 + i * 97 + x * 3 + y * 11 + k) & 255;
  // every row of every tile, at its own offset, plus "everything outside the tiles is
  // untouched" -- which with wall-to-wall tiles means there is no gap anywhere
  function assertComposite(tag, R, sw, sh, step) {
    const tw = Math.floor(sw / step), th = Math.floor(sh / step);
    let wrong = 0, first = null;
    for (let i = 0; i < 2; i++) {
      const [tx, ty] = R.at[i];
      for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) for (let k = 0; k < 4; k++) {
        const got = R.out[(ty + y) * R.w * 4 + 4 * (tx + x) + k];
        const want = srcByte(i, x * step, y * step, k);
        if (got !== want && !first) first = { i, x, y, k, got, want };
        if (got !== want) wrong++;
      }
    }
    ok(tag, !wrong && R.out.length === R.w * R.h * 4,
       wrong ? wrong + " bytes wrong, first " + JSON.stringify(first)
             : R.w + "x" + R.h + ", every row at its offset");
  }
  // a COLUMN (6x4 tiles: 6x8 is squarer than 12x4) and a ROW (4x6 tiles), with the source
  // row PADDED to 256 B exactly as a 256-aligned readback hands it over
  const col = compose(6, 4, 256, 1);
  ok("6x4 sources compose as a column", col.cols === 1 && col.w === 6 && col.h === 8,
     JSON.stringify(col.at));
  assertComposite("... and every row of both tiles lands at its offset (padded rows compacted)",
                  col, 6, 4, 1);
  const row = compose(4, 6, 256, 1);
  ok("4x6 sources compose as a row", row.cols === 2 && row.w === 8 && row.h === 6,
     JSON.stringify(row.at));
  assertComposite("... and every row of both tiles lands at its offset in the row layout",
                  row, 4, 6, 1);
  // tight rows: the same picture with no padding at all
  const tight = compose(4, 6, 16, 1);
  assertComposite("a tight (unpadded) source row composes identically", tight, 4, 6, 1);
  // halved: point-sampled, so output pixel (x, y) is source pixel (2x, 2y)
  const hv = compose(8, 6, 256, 2);
  ok("halved 8x6 sources make a column of two 4x3 tiles",
     hv.w === 4 && hv.h === 6 && JSON.stringify(hv.at) === "[[0,0],[0,3]]",
     JSON.stringify({ w: hv.w, h: hv.h, at: hv.at }));
  assertComposite("... and each output pixel is its own source pixel, point-sampled",
                  hv, 8, 6, 2);

  // 5. the ONE-tile fast path: a card's own recording still hands the mapped range
  //    straight to the constructor -- no copy, no composite
  const solo = env.run(`function(){
    const L = recTiles([{ cv: { width: 4, height: 3 } }]);
    L.bpr = 16;
    const ab = new ArrayBuffer(48);
    const out = recCompose(L, [ab]);
    return { same: out.buffer === ab, len: out.length }; }`);
  ok("one aligned unlabelled tile IS the mapped range: no copy at all",
     solo.same && solo.len === 48, JSON.stringify(solo));

  // 6. the label patch, blitted at its tile's own inset and clipped to the frame
  const lab = compose(4, 6, 16, 1, [2, 2]);
  const PAD = env.run("function(){ return REC_LBL_PAD; }");
  const at = lab.at[1];
  const px = (x, y) => lab.out[y * lab.w * 4 + 4 * x];
  ok("a tile's label is blitted inside THAT tile, at the layout's own inset",
     PAD >= 0 && (at[0] + PAD >= lab.w
       ? px(at[0], at[1]) !== 7                     // clipped away entirely: nothing of it
       : px(at[0] + PAD, at[1] + PAD) === 7 && px(at[0], at[1]) !== 7),
     "tile at " + JSON.stringify(at) + ", pad " + PAD + ", frame " + lab.w + "x" + lab.h);
  const clip = env.run(`function(){
    const L = recTiles([{ cv: { width: 4, height: 4 } }]);
    L.bpr = 16;
    L.tiles[0].label = { w: 99, h: 99, bytes: new Uint8Array(4 * 99 * 99).map(() => 9) };
    const out = recCompose(L, [new ArrayBuffer(64)]);
    return { len: out.length, w: L.w, h: L.h }; }`);
  ok("a label wider and taller than its frame is clipped, not written out of bounds",
     clip.len === clip.w * clip.h * 4, JSON.stringify(clip));

  // 7. ... and clipped to its own TILE, not merely to the frame. recCompose interleaves
  //    the blits with the row copies, so a bleed into the NEXT tile is overwritten there
  //    and invisible -- which is exactly why this is asserted on recBlitPatch itself,
  //    where the arithmetic lives, rather than through a composite that hides it.
  const bleed = (sw, sh) => env.run(`function(sw, sh){
    const L = recTiles([{ cv: { width: sw, height: sh } }, { cv: { width: sw, height: sh } }]);
    const row = L.w * 4, out = new Uint8Array(row * L.h);
    const p = { w: 4 * L.tw, h: 4 * L.th, bytes: new Uint8Array(4 * 4 * L.tw * 4 * L.th).fill(7) };
    const t = L.tiles[0];
    recBlitPatch(out, row, p, t.x + REC_LBL_PAD, t.y + REC_LBL_PAD,
                 L.tw - REC_LBL_PAD, L.th - REC_LBL_PAD);
    let drawn = 0, out_of_tile = 0;
    for (let y = 0; y < L.h; y++) for (let x = 0; x < L.w; x++) {
      if (out[y * row + 4 * x] !== 7) continue;
      drawn++;
      if (x >= L.tw || y >= L.th) out_of_tile++;
    }
    return { w: L.w, h: L.h, tw: L.tw, th: L.th, cols: L.cols, drawn: drawn,
             out: out_of_tile, want: (L.tw - REC_LBL_PAD) * (L.th - REC_LBL_PAD) }; }`,
    sw, sh);
  for (const [sw, sh, shape] of [[64, 32, "column"], [32, 64, "row"]]) {
    const b = bleed(sw, sh);
    ok("a label bigger than its tile stops at the TILE edge, not the frame's (" + shape + ")",
       b.drawn === b.want && b.out === 0,
       JSON.stringify({ drawn: b.drawn, want: b.want, spilled: b.out,
                        tile: b.tw + "x" + b.th, frame: b.w + "x" + b.h }));
  }

  // 8. ONE composite buffer per take, reused every slot: a 3x1024^2 frame is 12 MB, so
  //    allocating one per slot is ~360 MB/s of garbage at 30 fps. The bytes must be
  //    identical to what a fresh allocation gives, or the reuse is leaking a frame.
  const reuse = env.run(`function(sw, sh, bpr, n){
    const cs = []; for (let i = 0; i < n; i++) cs.push({ cv: { width: sw, height: sh } });
    let L = recTiles(cs); L.bpr = bpr;
    const ranges = [];
    for (let i = 0; i < n; i++) {
      const b = new Uint8Array(bpr * sh);
      for (let k = 0; k < b.length; k++) b[k] = (k * 7 + 1 + 31 * i) & 255;
      ranges.push(b.buffer);
    }
    const scratch = new Uint8Array(L.w * 4 * L.h);
    const a = recCompose(L, ranges, scratch), b2 = recCompose(L, ranges, scratch);
    const fresh = recCompose(L, ranges);
    return { reused: a === scratch && b2 === scratch, fresh: fresh !== scratch,
             same: Array.from(a).join() === Array.from(fresh).join(),
             len: a.length, want: L.w * 4 * L.h }; }`, 4, 6, 256, 2);
  ok("a composite handed a scratch buffer WRITES INTO IT and allocates nothing per slot",
     reuse.reused && reuse.fresh && reuse.len === reuse.want, JSON.stringify(reuse));
  ok("... and the bytes are exactly what a freshly allocated frame holds", reuse.same);
  // the one-tile PADDED slot is not the fast path (it has to compact the rows), so it
  // composites -- and must compose the same bytes with the scratch as without it
  const pad1 = env.run(`function(){
    const L = recTiles([{ cv: { width: 4, height: 3 } }]); L.bpr = 256;
    const b = new Uint8Array(256 * 3);
    for (let k = 0; k < b.length; k++) b[k] = (k * 13 + 5) & 255;
    const scratch = new Uint8Array(L.w * 4 * L.h);
    const a = recCompose(L, [b.buffer], scratch), fresh = recCompose(L, [b.buffer]);
    return { reused: a === scratch, len: a.length, want: L.w * 4 * L.h,
             same: Array.from(a).join() === Array.from(fresh).join() }; }`);
  ok("a ONE-tile padded slot composes identically with the scratch and without it",
     pad1.reused && pad1.same && pad1.len === pad1.want, JSON.stringify(pad1));
}

// ===========================================================================
// B. the action, on a booted page
// ===========================================================================
// the stub's VideoFrame-from-bytes capability is OFF by default (RECASYNC): arm it and
// clear the app's own probe latches, as bootstub's own buffer legs do
const bufArm = (env, on) => {
  env.bufFrames(!!on);
  env.run("function(){ recBufOff = false; recBufTried = false; recProbes.clear(); }");
};
const nDisp = (env, n) => env.run(`function(n){
  while (cards.disp.length > n) cardClose(cards.disp[cards.disp.length - 1]);
  while (cards.disp.length < n) addDisplayCard();
  cardsSync();
  return cards.disp.length; }`, n);
const btn = env => env.run(`function(){ const b = el("btnRecAll");
  return { on: !!b, t: b.innerHTML, ti: b.title, off: b.style.display === "none",
           dis: !!b.disabled, live: b.classList.contains("reclive"),
           grp: b.parentNode && b.parentNode.parentNode && b.parentNode.parentNode.id,
           withAdds: b.parentNode.children.indexOf(el("btnAddDisp")) >= 0 }; }`);
const takeOf = env => env.run(`function(){ const W = recAll.recorder && recAll.recorder.wc;
  if (!W) return null;
  return { n: W.n, drop: W.drop, pend: W.pend, w: W.w, h: W.h, sw: W.sw, sh: W.sh,
           name: W.name, bufOn: W.bufOn, cv: !!W.cv, step: W.lay.step, tiles: W.tiles.length,
           labels: W.tiles.map(t => !!t.label), chunks: W.chunks.length,
           pool: W.pool ? W.pool.length : -1, perSlot: W.pool ? W.pool[0].b.length : -1,
           busy: W.pool ? W.pool.filter(e => e.busy).length : -1,
           sync: W.chunks.map((c, i) => (c.key ? i : -1)).filter(i => i >= 0) }; }`);
const stripOf = env => env.run(`function(){ const s = recAll.resEl.video;
  return { on: !!s, foot: !!s && s.parentNode === recAll.foot,
           cls: recAll.foot.className,
           txt: s ? s.children.filter(x => x.kind === "span").map(x => x.innerHTML).join("") : "",
           btns: s ? s.children.filter(x => x.kind === "button").map(x => x.innerHTML) : [] }; }`);
const stripPress = (env, label) => env.run(`function(lab){ const s = recAll.resEl.video;
  if (!s) return false;
  const b = s.children.filter(x => x.kind === "button" && x.innerHTML === lab)[0];
  if (!b) return false;
  b.onclick(); return true; }`, label);
// drive the frame loop's own pass: renderCards is where the slot clock lives, so this is
// the real feeder and not a hand-rolled copy of it
const frames = (env, k) => env.run(`function(k){ for (let i = 0; i < k; i++) renderCards(false); }`, k);
const press = async env => { env.run(`function(){ el("btnRecAll").onclick(); }`); await settle(); };
const cardResClearPage = env => env.run(`function(){ cardResClear(recAll, "video"); }`);
const status = env => env.run(`function(){ const s = el("status"); return (s && s.textContent) || ""; }`);

async function pageLegs(name) {
  console.log("\n=== B. record every display (" + name + ") ===");
  const env = await boot(name);
  if (!env.run("function(){ return !!solver; }")) { ok(name + ": solver came up", false); return; }

  // ---- 1. the offer: >= 2 display cards, and only where leg 1 runs ----------
  nDisp(env, 1);
  const b1 = btn(env);
  ok("a `rec all` button sits in the displays & charts group beside + display",
     b1.on && b1.grp === "grpDisp" && b1.withAdds && b1.t === "rec all", JSON.stringify(b1));
  ok("with ONE display card the action is not offered (it would duplicate that card's rec)",
     b1.dis && /second display/.test(b1.ti), JSON.stringify(b1));
  nDisp(env, 2);
  const b2 = btn(env);
  ok("with two cards it goes live, and its title says what the file will be",
     !b2.dis && !b2.off && /2 tiles/.test(b2.ti) && /1024×512/.test(b2.ti), b2.ti);
  const n3 = nDisp(env, 3);
  const b3 = btn(env);
  ok("with three cards the title follows: 3 tiles at the composite size",
     n3 === 3 && !b3.dis && /3 tiles/.test(b3.ti) && /1536×512/.test(b3.ti), b3.ti);
  nDisp(env, 2);
  // the per-card button is untouched by all of this
  const perCard = env.run(`function(){ return cards.disp.map(d => d.btnRec.innerHTML); }`);
  ok("every display card still has its own `rec` button, unchanged",
     JSON.stringify(perCard) === '["rec","rec"]', JSON.stringify(perCard));
  // an engine with no WebCodecs at all: the action is simply absent (leg 2 records a
  // canvas STREAM and cannot cover several canvases), single-card recording untouched
  const noWC = env.run(`function(){
    const keep = window.VideoEncoder; window.VideoEncoder = undefined;
    cardsSync();
    const off = el("btnRecAll").style.display === "none";
    const recs = cards.disp.map(d => d.btnRec.style.display !== "none");
    window.VideoEncoder = keep; cardsSync();
    return { off: off, recs: recs, back: el("btnRecAll").style.display !== "none" }; }`);
  ok("a WebCodecs-less engine is not offered the action, and keeps its per-card rec",
     noWC.off && noWC.recs.every(Boolean) && noWC.back, JSON.stringify(noWC));

  // ---- 2. a two-card take, end to end --------------------------------------
  bufArm(env, true);
  const nFr0 = env.caps.frames.length, nCp0 = env.caps.copies.length;
  await press(env);
  const T0 = takeOf(env);
  ok("the press starts ONE take over both cards, configured for the COMPOSITE frame",
     !!T0 && T0.tiles === 2 && T0.w === 1024 && T0.h === 512 && T0.sw === 512 && T0.sh === 512,
     JSON.stringify(T0 && { w: T0.w, h: T0.h, tiles: T0.tiles }));
  const cfg = env.run(`function(){ const e = window.__encs = null, c = recAll.recorder.wc.enc.config;
    return { w: c.width, h: c.height, codec: c.codec, fps: c.framerate }; }`);
  ok("... and the encoder itself is configured 1024x512 at 30 fps",
     cfg.w === 1024 && cfg.h === 512 && cfg.fps === 30, JSON.stringify(cfg));
  ok("it takes the bytes path, has no single canvas to fall back on, and names the file "
     + "after the page rather than one field",
     T0.bufOn && !T0.cv && /-displays-t/.test(T0.name) && /\.mp4$/.test(T0.name), T0.name);
  ok("each tile carries a label patch (the picture would be ambiguous without one)",
     T0.labels.length === 2 && T0.labels.every(Boolean), JSON.stringify(T0.labels));
  const lab0 = env.run(`function(){ const W = recAll.recorder.wc;
    return { on: W.tiles.map(t => t.label && t.label.text),
             want: cards.disp.map(d => DISP_SLUG[d.barMode >= 0 ? d.barMode : d.sel()]) }; }`);
  ok("... naming the field that tile is showing",
     JSON.stringify(lab0.on) === JSON.stringify(lab0.want), JSON.stringify(lab0));
  ok("the button says it is live", btn(env).live);

  // the render gate: every source of a live take must render, or the slot would capture
  // an expired texture
  const rendered = env.run(`function(){
    cards.disp.forEach(d => { d._rn = 0; d.render = function () { this._rn++; return DisplayCard.prototype.render.call(this); }; });
    renderCards(false);
    const r = cards.disp.map(d => d._rn);
    cards.disp.forEach(d => { delete d.render; delete d._rn; });
    return r; }`);
  ok("a live all-displays take makes EVERY source render on the loop's pass",
     JSON.stringify(rendered) === "[1,1]", JSON.stringify(rendered));

  await settle();                       // let the render-gate leg's own slot land first
  const n0 = takeOf(env).n, nCp1 = env.caps.copies.length;
  env.holdMaps(true);
  frames(env, 3);
  const held = takeOf(env);
  ok("three slots submit 2 copies each and encode NOTHING yet -- the main thread waits "
     + "on no readback",
     held.n === n0 && held.pend === 3 && held.busy === 3 &&
     env.caps.copies.length - nCp1 === 6 && env.mapsPending() === 6,
     JSON.stringify({ n: held.n - n0, pend: held.pend, copies: env.caps.copies.length - nCp1,
                      maps: env.mapsPending() }));
  ok("the pool is REC_POOL SLOTS deep, each slot holding one buffer per source",
     held.pool === 3 && held.perSlot === 2, JSON.stringify({ pool: held.pool, per: held.perSlot }));
  env.maps();
  await settle();
  const landed = takeOf(env);
  ok("the landed slots encode ONE frame each -- N sources, one slot clock, one ladder",
     landed.n === n0 + 3 && landed.chunks === landed.n && landed.pend === 0 && landed.busy === 0,
     JSON.stringify({ n: landed.n, chunks: landed.chunks }));
  const fr = env.caps.frames.slice(nFr0).filter(f => f.kind === "bytes" && f.codedWidth > 2);
  ok("every frame is built FROM BYTES at the composite size, on the take's own 1/30 s "
     + "ladder, and closed",
     fr.length === landed.n && fr.every(f => f.codedWidth === 1024 && f.codedHeight === 512) &&
     fr.every((f, i) => f.timestamp === Math.round(i * 1e6 / 30)) && fr.every(f => f.closed),
     JSON.stringify(fr.map(f => [f.codedWidth, f.codedHeight, f.timestamp])));

  // ---- 3. ALL-OR-NOTHING: one source cannot be captured ---------------------
  // (a) a source resized under the take -- the encoder is configured for one frame size
  const resized = env.run(`function(){
    const W = recAll.recorder.wc, d = cards.disp[1], w = d.cv.width;
    const before = { n: W.n, drop: W.drop, copies: 0 };
    d.cv.width = 500;
    renderCards(false);
    const mid = { n: W.n, drop: W.drop };
    d.cv.width = w;
    return { before: before, mid: mid }; }`);
  const afterResize = takeOf(env);
  ok("a slot one of whose sources cannot be captured is dropped WHOLE: W.n does not move",
     resized.mid.n === landed.n && afterResize.n === landed.n && afterResize.drop > landed.drop,
     JSON.stringify({ n: afterResize.n - landed.n, drop: afterResize.drop - landed.drop }));
  ok("... and it submits no copies at all: nothing half-captured goes into the pool",
     afterResize.busy === 0 && env.mapsPending() === 0,
     JSON.stringify({ busy: afterResize.busy, maps: env.mapsPending() }));
  // (b) one source's MAP is rejected: the slot is joined before it can be encoded, so the
  //     whole slot goes and the buffers that DID map are released
  frames(env, 1);
  const rej = env.run(`function(){ const W = recAll.recorder.wc;
    const s = W.pool.filter(e => e.busy)[0];
    s.b[1].destroy();                       // a lost buffer under one of the two sources
    return { busy: W.pool.filter(e => e.busy).length, n: W.n, drop: W.drop }; }`);
  env.maps();
  await settle();
  const afterRej = takeOf(env);
  ok("a slot whose second source's map is REJECTED encodes nothing and frees its slot",
     rej.busy === 1 && afterRej.n === landed.n && afterRej.drop > afterResize.drop &&
     afterRej.busy === 0,
     JSON.stringify({ n: afterRej.n, drop: afterRej.drop, busy: afterRej.busy }));
  // (c) the take LATCHES OFF the buffer path. Three shipped catches do it mid-take
  //     (recPoolMake's OOM, a thrown copy, a failed compose) and a take can start with it
  //     off; the fallback is the SYNC canvas path, which needs the one canvas a composite
  //     has not got. Every remaining slot is therefore a drop -- never `VideoFrame(null)`,
  //     which on a real engine throws once per due slot for the rest of the take.
  const nFrL = env.caps.frames.length, fails0 = env.fails.length;
  const latched = env.run(`function(){ const W = recAll.recorder.wc;
    W.bufOn = false;
    const b = { n: W.n, drop: W.drop, chunks: W.chunks.length };
    renderCards(false); renderCards(false);
    const a = { n: W.n, drop: W.drop, chunks: W.chunks.length, cv: W.cv };
    W.bufOn = true;
    return { b: b, a: a }; }`);
  ok("a composite that has latched OFF the buffer path DROPS its due slots: no frame is "
     + "built, W.n does not move, and nothing is asked of a canvas it has not got",
     latched.a.cv === null && latched.a.n === latched.b.n &&
     latched.a.chunks === latched.b.chunks && latched.a.drop > latched.b.drop &&
     env.caps.frames.length === nFrL && env.fails.length === fails0,
     JSON.stringify({ n: latched.a.n - latched.b.n, drop: latched.a.drop - latched.b.drop,
                      frames: env.caps.frames.length - nFrL,
                      fails: env.fails.slice(fails0) }));

  // ---- 3b. the caption follows the field --------------------------------------
  // The field select stays LIVE under a take and the tile's pixels move with it, so the
  // label has to move too -- otherwise tile 0 shows vorticity captioned `u` for the rest
  // of the recording. The patches are rendered ONCE at start; this is the one thing that
  // re-renders one.
  const relab = env.run(`function(){ const W = recAll.recorder.wc, d = cards.disp[0];
    const was = W.tiles[0].label, keep = W.tiles[1].label;
    const alt = cards.cfg.fields.filter(f => String(f.v) !== d.selField.value)[0];
    d.selField.value = String(alt.v);
    d.selField.onchange();
    renderCards(false); renderCards(false);
    return { before: was && was.text, after: W.tiles[0].label && W.tiles[0].label.text,
             want: DISP_SLUG[d.barMode >= 0 ? d.barMode : d.sel()],
             mate: W.tiles[1].label === keep,
             size: [W.sw, W.sh, d.cv.width, d.cv.height] }; }`);
  env.maps();
  await settle();
  ok("a field retyped under a live take is RE-CAPTIONED: the label follows the pixels",
     !!relab.after && relab.after === relab.want && relab.after !== relab.before,
     JSON.stringify(relab));
  ok("... and the tile that did NOT change keeps the very patch it was given at start",
     relab.mate, JSON.stringify(relab.mate));

  // ... and the sample table the take ends with is still uniform: a fixed 1/30 s ladder
  // with no hole where the dropped slots were
  const before = env.caps.downloads.length;
  await press(env);
  await settle();
  const st = stripOf(env);
  ok("the stop leaves the file on the PAGE's strip, not on a card's, and downloads nothing",
     st.on && st.foot && st.cls === "viewfoot" && env.caps.downloads.length === before,
     JSON.stringify({ st: st.on, foot: st.foot, dl: env.caps.downloads.length - before }));
  ok("the strip offers download (and dismiss) for the composite",
     st.btns.indexOf("download") >= 0 && st.btns.indexOf("&times;") >= 0, JSON.stringify(st.btns));
  ok("it does NOT claim the tiles were downscaled -- they were not",
     st.txt.indexOf("half size") < 0, st.txt);
  ok(stripPress(env, "download") ? "the composite downloads on demand" : "download button", true);
  const dl = env.caps.downloads[env.caps.downloads.length - 1];
  const u8 = dl && dl.blob && dl.blob.bytes;
  const boxAt = (b, o) => String.fromCharCode(b[o + 4], b[o + 5], b[o + 6], b[o + 7]);
  const be = (b, o) => ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
  let tops = [], p = 0;
  while (u8 && p + 8 <= u8.length) { tops.push(boxAt(u8, p)); p += be(u8, p); }
  ok("the composite file is a plain progressive mp4: ftyp + mdat + moov, no moof",
     !!u8 && tops.join(",") === "ftyp,mdat,moov" && /-displays-t/.test(dl.name),
     (dl && dl.name) + " " + tops.join(","));
  // THE honesty property, read off the finished file: `stts` is a SINGLE run of equally
  // spaced samples. The two dropped slots left fewer frames and no hole -- nothing was
  // backfilled, and the sample table does not claim a timing it never had. (moov is the
  // last box, so the last "stts" in the bytes is the real one.)
  const sttsOf = b => {
    let at = -1;
    for (let i = 0; i + 3 < b.length; i++)
      if (b[i] === 0x73 && b[i + 1] === 0x74 && b[i + 2] === 0x74 && b[i + 3] === 0x73) at = i;
    if (at < 0) return null;
    const n = be(b, at + 8), runs = [];
    for (let i = 0; i < n; i++) runs.push([be(b, at + 12 + 8 * i), be(b, at + 16 + 8 * i)]);
    return runs;
  };
  const runs = u8 && sttsOf(u8);
  const allFr = env.caps.frames.slice(nFr0).filter(f => f.kind === "bytes" && f.codedWidth > 2);
  ok("every dropped slot left FEWER frames and a still-UNIFORM sample table: one stts run",
     !!runs && runs.length === 1 && runs[0][0] === allFr.length && runs[0][1] === 1000,
     JSON.stringify(runs) + " for " + allFr.length + " frames");
  ok("... and the frame indices are unbroken: no backfill for any dropped slot",
     allFr.every((f, i) => f.timestamp === Math.round(i * 1e6 / 30)),
     JSON.stringify(allFr.map(f => f.timestamp)));
  stripPress(env, "&times;");

  // the WATCHDOG must not feed a composite: it builds a VideoFrame from ONE canvas, and a
  // composite has none. The stub's document.hidden boots TRUE, which is exactly the
  // condition the watchdog runs on, so ticking the timer here is the honest test -- it
  // must encode nothing and, above all, must not try to read a canvas that is null.
  await press(env);
  const wd0 = takeOf(env);
  env.tick(5);
  const wd1 = takeOf(env);
  ok("the watchdog does not feed an all-displays take (fewer frames, never a stretched or "
     + "half-composited one)",
     wd1.n === wd0.n && wd1.chunks === wd0.chunks && wd1.drop === wd0.drop,
     JSON.stringify({ before: wd0.n, after: wd1.n, drop: wd1.drop - wd0.drop }));
  await press(env);
  await settle();
  cardResClearPage(env);

  // the 30 s cap is the same shared timer, so it is asserted where it can end a take that
  // is nobody's card: an unattended composite must stop itself and still leave the file
  await press(env);
  frames(env, 2); env.maps(); await settle();
  const armed = env.fireTimeout(env.run("function(){ return REC_MAX_MS; }"));
  await settle();
  ok("the 30 s hard stop ends a composite take too, onto the page's strip",
     armed && takeOf(env) === null && stripOf(env).on && !btn(env).live,
     JSON.stringify({ armed: armed, live: btn(env).live }));
  stripPress(env, "&times;");

  // ---- 4. the encoder refuses the composite size ---------------------------
  const refuse = env.run(`function(limit){
    window.__ics = window.VideoEncoder.isConfigSupported;
    window.VideoEncoder.isConfigSupported = cfg =>
      Promise.resolve({ supported: cfg.width <= limit, config: cfg });
    recProbes.clear();
    return true; }`, 600);
  await press(env);
  const T1 = takeOf(env);
  ok("a refused composite HALVES the tiles and records anyway",
     !!T1 && T1.step === 2 && T1.w === 512 && T1.h === 256 && T1.tiles === 2,
     JSON.stringify(T1 && { w: T1.w, h: T1.h, step: T1.step, tiles: T1.tiles }));
  ok("... and the card list is never truncated to fit: both cards are still in it",
     T1 && T1.tiles === env.run("function(){ return cards.disp.length; }"),
     JSON.stringify({ tiles: T1 && T1.tiles }));
  const cfg2 = env.run(`function(){ const c = recAll.recorder.wc.enc.config;
    return { w: c.width, h: c.height }; }`);
  ok("the encoder is configured for the HALVED composite", cfg2.w === 512 && cfg2.h === 256,
     JSON.stringify(cfg2));
  frames(env, 2); env.maps(); await settle();
  const nFrH = env.caps.frames.length;
  const hv = env.caps.frames.slice(nFrH - 2);
  ok("the frames really are the halved size", hv.every(f => f.codedWidth === 512 && f.codedHeight === 256),
     JSON.stringify(hv.map(f => [f.codedWidth, f.codedHeight])));
  await press(env);
  await settle();
  const stH = stripOf(env);
  ok("the result strip SAYS the tiles were downscaled",
     stH.on && /half size/.test(stH.txt), stH.txt);
  stripPress(env, "&times;");
  // ... and an encoder that refuses even the halved composite starts nothing and says so
  env.run(`function(){ window.VideoEncoder.isConfigSupported = () => Promise.resolve({ supported: false });
    recProbes.clear(); }`);
  const dl0 = env.caps.downloads.length;
  await press(env);
  const T2 = takeOf(env);
  ok("an encoder that refuses both sizes starts nothing, writes nothing, and says why",
     T2 === null && env.caps.downloads.length === dl0 && /encoder/.test(status(env)) &&
     !stripOf(env).on, JSON.stringify({ take: T2, status: status(env) }));
  ok("... and the button is back to idle", !btn(env).live && btn(env).t === "rec all");
  env.run(`function(){ window.VideoEncoder.isConfigSupported = window.__ics;
    delete window.__ics; recProbes.clear(); }`);

  // ---- 5. a card closed mid-recording --------------------------------------
  // Decision: the take ENDS and the file is written. The composite geometry is the
  // encoder's frame size and cannot move, so a vanished source could only be papered over
  // with a stale or blank tile -- and stopping silently would lose the take.
  await press(env);
  frames(env, 2); env.maps(); await settle();
  const dlBefore = env.caps.downloads.length;
  env.run(`function(){ cardClose(cards.disp[cards.disp.length - 1]); }`);
  await settle();
  const closed = takeOf(env);
  const stC = stripOf(env);
  ok("closing a source card ends the take rather than recording a stale tile",
     closed === null && !btn(env).live, JSON.stringify(closed));
  ok("... and the file is written, to the page's strip -- it does not die with the card",
     stC.on && stC.foot && env.caps.downloads.length === dlBefore, JSON.stringify(stC));
  ok("... and the page says what happened", /closed/.test(status(env)), status(env));
  stripPress(env, "&times;");
  ok("with one card left the action is not offered again", btn(env).dis);

  // ---- 6. the single-card `rec` path is UNCHANGED ---------------------------
  // The no-fork rule means this was refactored underneath, so it is asserted here rather
  // than assumed: one tile, the card's own canvas size, its own field name, no label, and
  // the strip on the CARD's footer.
  nDisp(env, 2);
  const nFrS = env.caps.frames.length;
  env.run(`function(){ cards.disp[0].btnRec.onclick(); }`);
  await settle();
  const S = env.run(`function(){ const d = cards.disp[0], W = d.wc;
    return W && { n: W.tiles.length, w: W.w, h: W.h, cv: W.cv === d.cv, step: W.lay.step,
                  label: !!W.tiles[0].label, name: W.name, cw: d.cv.width, ch: d.cv.height,
                  pool: W.pool ? W.pool[0].b.length : -1 }; }`);
  ok("a card's own take is ONE tile at that canvas's size, off that canvas, with no label",
     !!S && S.n === 1 && S.w === S.cw && S.h === S.ch && S.cv && S.step === 1 && !S.label,
     JSON.stringify(S));
  ok("... and it is still named after the FIELD, not the page",
     S && !/-displays-/.test(S.name) && /\.mp4$/.test(S.name), S && S.name);
  frames(env, 2); env.maps(); await settle();
  const sFr = env.caps.frames.slice(nFrS).filter(f => f.kind === "bytes" && f.codedWidth > 2);
  ok("its frames are its own canvas, and its pool one buffer per slot",
     sFr.length >= 1 && sFr.every(f => f.codedWidth === S.cw && f.codedHeight === S.ch) &&
     env.run(`function(){ const W = cards.disp[0].wc; return W.pool[0].b.length; }`) === 1,
     JSON.stringify(sFr.map(f => [f.codedWidth, f.codedHeight])));
  // the SYNC fallback is a single-source take's own: it HAS one canvas, so latching the
  // buffer path off keeps it recording (from the canvas, as it did before RECASYNC)
  // rather than dropping -- the leg the composite guard above must not have taken away.
  const nFrSync = env.caps.frames.length;
  const sync = env.run(`function(){ const W = cards.disp[0].wc, was = W.n;
    W.bufOn = false;
    renderCards(false);
    const r = { was: was, n: W.n, cv: W.cv === cards.disp[0].cv };
    W.bufOn = true; return r; }`);
  const syncFr = env.caps.frames.slice(nFrSync);
  ok("... and with the buffer path off it still records SYNCHRONOUSLY from that canvas",
     sync.cv && sync.n === sync.was + 1 && syncFr.length === 1 &&
     syncFr[0].kind === "canvas" && syncFr[0].codedWidth === S.cw,
     JSON.stringify({ n: sync.n - sync.was, frames: syncFr.map(f => [f.kind, f.codedWidth]) }));
  env.run(`function(){ cards.disp[0].btnRec.onclick(); }`);
  await settle();
  const cardStrip = env.run(`function(){ const d = cards.disp[0], s = d.resEl.video;
    return { on: !!s, foot: !!s && s.parentNode === d.foot,
             page: !!recAll.resEl.video,
             txt: s ? s.children.filter(x => x.kind === "span").map(x => x.innerHTML).join("") : "" }; }`);
  ok("its file lands on the CARD's footer, with no downscale note and nothing on the page's",
     cardStrip.on && cardStrip.foot && !cardStrip.page && cardStrip.txt.indexOf("half size") < 0,
     JSON.stringify(cardStrip));
  env.run(`function(){ cardResClear(cards.disp[0], "video"); }`);
  bufArm(env, false);
}

(async () => {
  const env = await boot("rmhd2d.html");
  tilerLegs(env);
  await pageLegs("rmhd2d.html");
  await pageLegs("rmhd3d.html");
  console.log("\n" + (bad ? "FAIL" : "PASS") + "  rec all: " + pass + " checks passed, " +
              bad + " failed");
  process.exit(bad ? 1 : 0);
})();
