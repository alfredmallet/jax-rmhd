// Contour-overlay dataflow tracer (FEEDBACK_2026-08-08 P0 item 2).
//
// Boots a real app page on the shared stub (devtools/stubenv.js) and REPLAYS, op by op,
// what the contour overlay actually does: every writeBuffer and every (pipeline, bind
// group) dispatch is recorded in order, buffers are named, and a tiny symbolic +
// numeric interpreter propagates "which potential is in which buffer" and "what level
// spacing does each set's table hold" all the way to the colorize dispatch. So a frame
// answers three questions separately:
//
//   PREPARED  -- which potential landed in cp / cp2 (an index/bind mix-up shows here)
//   INKED     -- which level table (cd / cd2) colorize pairs with each of them
//   SCALED    -- the delta each table carries, i.e. whether contLevel's adapting range
//                (CONT_RELAX per frame, never reset) still belongs to a PREVIOUS
//                potential after the selection changed
//
// The numeric side feeds the max reduction a per-potential amplitude (tearing-like by
// default: |psi| ~ 1, |phi| ~ 1e-2), which is what makes the range lag visible.
//
//   node contrepro.js <dir> <page> [demo]
"use strict";
const [dir, page, demo] = process.argv.slice(2);
if (!dir || !page) { console.error("usage: node contrepro.js <dir> <page> [demo]"); process.exit(2); }
const env = require("./stubenv")(dir, page, demo);
const { run } = env;

// amplitude of each potential on the displayed plane, as the max reduction would find it
const AMP = { 2: 1e-2, 3: 1.0 };          // DISP_PHI = 2, DISP_PSI = 3
const LAB = { 0: "vort", 1: "curr", 2: "phi", 3: "psi", 4: "|u|", 5: "|b|",
              6: "|z+|", 7: "|z-|", 8: "sigma" };

// ---------------------------------------------------------------------------
// tracer: patch the stub device, then rebuild the chains so every bind group is seen
// ---------------------------------------------------------------------------
const install = () => run(`function(){
  const dev = solver.device, T = { ops: [], bg: [], name: [], pl: [] };
  globalThis.__T = T;
  const idOf = (list, o) => { let i = list.indexOf(o); if (i < 0) { i = list.length; list.push(o); } return i; };
  globalThis.__bgId = o => idOf(T.bg, o);
  const oBG = dev.createBindGroup.bind(dev);
  dev.createBindGroup = o => {
    const g = oBG(o);
    g.__e = o.entries.map(e => (e.resource && e.resource.buffer) ? e.resource.buffer : null);
    return g;
  };
  const oW = dev.queue.writeBuffer.bind(dev.queue);
  dev.queue.writeBuffer = (b, off, data) => {
    T.ops.push({ k: "w", b: b, off: off, d: Array.prototype.slice.call(data) });
    return oW(b, off, data);
  };
  const oEnc = dev.createCommandEncoder.bind(dev);
  dev.createCommandEncoder = () => {
    const e = oEnc();
    const oCP = e.beginComputePass.bind(e);
    e.beginComputePass = () => {
      const p = oCP(), sp = p.setPipeline.bind(p), sb = p.setBindGroup.bind(p),
            dw = p.dispatchWorkgroups.bind(p);
      let cur = null;
      p.setPipeline = x => { cur = x; return sp(x); };
      p.setBindGroup = (i, g) => { p.__g = g; return sb(i, g); };
      p.dispatchWorkgroups = (...a) => { T.ops.push({ k: "d", pl: cur, bg: p.__g }); return dw(...a); };
      return p;
    };
    return e;
  };
  // chains built before the patch carry untraced bind groups -> rebuild them all
  solver.disp.length = 0;
  cardsSync();
  return true;
}`);

// name every buffer and pipeline, so the replay can print them
const names = () => run(`function(){
  const T = globalThis.__T, out = { pl: [], buf: [] };
  for (const k in solver.pl) { solver.pl[k].__n = k; out.pl.push(k); }
  const nm = (b, n) => { if (b) { b.__n = n; out.buf.push(n); } };
  for (const k in solver.buf) nm(solver.buf[k], "G." + k);
  solver.disp.forEach((D, ci) => {
    if (!D) return;
    for (const k in D.buf) {
      const b = D.buf[k];
      if (Array.isArray(b)) b.forEach((x, j) => nm(x, "c" + ci + "." + k + "[" + j + "]"));
      else nm(b, "c" + ci + "." + k);
    }
  });
  return out;
}`);

// ---------------------------------------------------------------------------
// the replay interpreter
// ---------------------------------------------------------------------------
// per-pipeline dataflow: [inIdx, outIdx] on the bind-group entries, or a special tag
const FLOW = {
  colsInv: [0, 1], rowsC2R: [0, 1], sliceExtract: [0, 1], faceExtract: [0, 1],
  maxPartial: [0, 1], maxFinal: [0, 1], maxPartialFace: [0, 1], maxFinalFace: [0, 1],
  vecMag: [1, 0], vecMagSq: [1, 0], vecMagFace: [1, 0], vecMagSqFace: [1, 0]
};
const VAL = new Map();            // buffer -> { lab, amp }
const MEM = new Map();            // buffer -> Float32Array(4)  (contB / mode uniforms)
const mem = b => { if (!MEM.has(b)) MEM.set(b, new Float32Array(8)); return MEM.get(b); };
const bn = b => (b && b.__n) || "?";

function replay(ops) {
  const hits = [];
  for (const o of ops) {
    if (o.k === "w") {                       // CPU write: floats and u32 share the slot
      const m = mem(o.b), i0 = o.off / 4;
      for (let i = 0; i < o.d.length; i++) m[i0 + i] = o.d[i];
      continue;
    }
    const n = o.pl && o.pl.__n, e = (o.bg && o.bg.__e) || [];
    if (n === "prepDisp") {
      const md = mem(e[3])[0] | 0;
      VAL.set(e[2], { lab: LAB[md], amp: AMP[md] !== undefined ? AMP[md] : 0.5 });
      VAL.set(e[4], { lab: LAB[md] + "#2", amp: AMP[md] !== undefined ? AMP[md] : 0.5 });
      continue;
    }
    if (n === "contLevel") {                 // [mx, st]; st = [range, delta, nlev, plain]
      const v = VAL.get(e[0]) || { lab: "?", amp: 0 }, st = mem(e[1]);
      const m = v.amp;
      let r = st[0];
      if (!(r > 0) || m > r) r = m; else r = r + 0.05 * (m - r);
      st[0] = r; st[1] = 2 * r / Math.max(st[2], 1);
      st[4] = 1;                             // marker: this table was refreshed this frame
      VAL.set(e[1], { lab: "lev(" + v.lab + ")", amp: r });
      continue;
    }
    if (n === "colorize" || n === "colorizeCube") {
      // colorize: cp,cd,cp2,cd2 = bindings 4..7; colorizeCube: 6..9
      const b0 = n === "colorize" ? 4 : 6;
      const g = i => VAL.get(e[i]) || { lab: "(stale/unset)", amp: 0 };
      hits.push({
        kernel: n, field: g(0).lab,
        cp: g(b0).lab, cd: bn(e[b0 + 1]), cdDelta: mem(e[b0 + 1])[1], cdRange: mem(e[b0 + 1])[0],
        cp2: g(b0 + 2).lab, cd2: bn(e[b0 + 3]), cd2Delta: mem(e[b0 + 3])[1],
        cd2Range: mem(e[b0 + 3])[0]
      });
      continue;
    }
    const f = FLOW[n];
    if (f && e[f[0]] && e[f[1]]) VAL.set(e[f[1]], VAL.get(e[f[0]]) || { lab: "?", amp: 0 });
  }
  return hits;
}

// ---------------------------------------------------------------------------
// driving
// ---------------------------------------------------------------------------
const drain = () => {
  const ops = run("function(){ const o = globalThis.__T.ops; globalThis.__T.ops = []; return o; }");
  return replay(ops);
};
// one rendered frame of card 0 (no solver.step: the physics is irrelevant here)
const frame = () => { run("function(){ cards.disp[0].render(); }"); return drain()[0]; };
// (3D, since ISO_PLAN B: the card opens on the VOLUME, which draws no plane and hence no
// contour overlay at all -- so the tracer puts it on a plane, which is the view whose
// dataflow this is about)
const setCard = (contVal, sel) => run(`function(cv, s){
  const c = cards.disp[0];
  if (s !== null) c.selField.value = String(s);
  if (c.selZSrc) c.selZSrc.value = "manual";
  c.selCont.value = String(cv);
  c.apply();
}`, contVal, sel === undefined ? null : sel);

const fails = [];
const check = (cond, msg) => { if (!cond) { fails.push(msg); console.log("   FAIL " + msg); } };
const fmt = h => "field=" + h.field + "  cp=" + h.cp + " [" + h.cd + " delta=" + h.cdDelta.toExponential(3) +
                 " range=" + h.cdRange.toExponential(3) + "]  cp2=" + h.cp2 + " [" + h.cd2 +
                 " delta=" + h.cd2Delta.toExponential(3) + "]";

// what the card ASKED for, straight from the shared card code
const want = () => run("function(){ return cards.disp[0].cont(); }");

// ---------------------------------------------------------------------------
// INK: does the overlay's colour encode the DISPLAYED field?
// ---------------------------------------------------------------------------
// The contour GEOMETRY can be right and the overlay still carry the other field, if the
// ink is picked per texel from the background luminance: the black/white boundary is
// then the background's lum = 0.5 contour, i.e. a contour of the DISPLAYED quantity
// drawn on top of the contour set. This prints where that boundary sits for each
// colormap (the real cmapRGB, the CPU mirror of the emitted cmap()), and then asserts on
// the EMITTED kernels that set 0's ink no longer depends on the background at all.
function inkSection() {
  console.log("\n-- ink: what set 0's contour colour depends on");
  const flips = run(`function(){
    const res = [];
    for (let w = 0; w < CMAP_NAMES.length; w++) {
      const L = v => { const c = cmapRGB(w, 0.5 * (v + 1));
                       return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) > 0.5 ? "black" : "white"; };
      const f = []; let prev = L(-1);
      for (let i = 1; i <= 4000; i++) {
        const v = -1 + 2 * i / 4000, k = L(v);
        if (k !== prev) f.push(Number(v.toFixed(3)));
        prev = k;
      }
      res.push([CMAP_NAMES[w], L(-1), L(1), f]);
    }
    return res;
  }`);
  for (const r of flips)
    console.log("   " + String(r[0]).padEnd(10) + " a luminance-picked ink would be " + r[1] + " at -max, " +
                r[2] + " at +max, flipping at value/autoscale = " + JSON.stringify(r[3]));
  const S = run("function(){ return buildShaders(solver.g); }");
  for (const k of ["colorize", "colorizeCube"]) {
    if (!S[k]) continue;
    const m = /fn contInk[\s\S]*?\n\}/.exec(S[k]);
    if (!m) { fails.push(k + ": no contInk in the emitted kernel"); continue; }
    const body = m[0];
    console.log("   emitted " + k + " contInk:\n" + body.replace(/^/gm, "     "));
    check(!/lum|dot\(/.test(body),
          k + ": set 0's ink is still derived from the background colour (the overlay " +
          "then draws a contour of the DISPLAYED field in black/white)");
  }
}

// the per-hit assertions, shared by the transition scenarios and the re-add section:
//   (a) PREPARED / INKED -- cp is set 0's potential and cp2 set 1's (that pairing is
//       written by the card's own modeC writeBuffer, read back through prepDisp), and
//       each set is inked from THIS card's chain slot, `c<ci>.contB[0|1]`, so a card
//       reading another slot's level table (a chain-reuse bug) shows up here;
//   (b) SCALED -- an ACTIVE set's spacing matches its OWN potential's amplitude.
function checkHit(title, h, w, ci, nlev) {
  const lab = w.map(m => (m ? LAB[m] : "off"));
  if (w[0]) check(h.cp === lab[0], title + ": set 0 prepared " + h.cp + ", wanted " + lab[0]);
  if (w[1]) check(h.cp2 === lab[1], title + ": set 1 prepared " + h.cp2 + ", wanted " + lab[1]);
  for (const [i, nm] of [[0, h.cd], [1, h.cd2]]) {
    const t = "c" + ci + ".contB[" + i + "]";
    check(nm === t, title + ": set " + i + " inked from " + nm + ", not this card's " + t);
  }
  for (const [i, cp, dl] of [[0, h.cp, h.cdDelta], [1, h.cp2, h.cd2Delta]]) {
    if (!w[i]) { check(dl === 0, title + ": inactive set " + i + " has delta " + dl); continue; }
    const good = 2 * AMP[w[i]] / nlev;
    const err = Math.abs(dl - good) / good;
    console.log("   set " + i + " spacing: " + dl.toExponential(3) + " vs correct " +
                good.toExponential(3) + "  (x" + (dl / good).toFixed(1) + ")");
    check(err < 0.05, title + ": set " + i + " (" + cp + ") drawn at a spacing " +
          (dl / good).toFixed(1) + "x its own -- the range still belongs to the PREVIOUS potential");
  }
}

function scenario(title, contVal, sel, nframes) {
  console.log("\n-- " + title);
  setCard(contVal, sel);
  drain();                                   // the writeBuffers apply() just made
  const w = want(), lab = w.map(m => (m ? LAB[m] : "off"));
  let first = null, last = null;
  for (let f = 0; f < nframes; f++) {
    const h = frame();
    if (!h) { fails.push(title + ": no colorize dispatch"); return; }
    if (f === 0) { first = h; console.log("   frame 1 : " + fmt(h)); }
    last = h;
  }
  console.log("   frame " + nframes + " : " + fmt(last));
  console.log("   asked for: set0=" + lab[0] + "  set1=" + lab[1]);
  checkHit(title, first, w, run("function(){ return cards.disp[0].ci; }"),
           run("function(){ return cards.disp[0].nlev(); }"));
}

setTimeout(() => {
  const tag = page + (demo ? " ?demo=" + demo : "");
  console.log("=== contour overlay trace: " + tag + " ===");
  install();
  const NAMES = names();
  console.log("buffers named: " + NAMES.buf.length + ", pipelines: " + NAMES.pl.length);
  drain();                                   // the rebuild's own writeBuffers
  inkSection();

  // the preset's own card first, then the transitions Alfred's report implies
  scenario("as the preset opens it (psi contours)", 3, undefined, 3);
  scenario("switch the DISPLAYED field to phi, psi contours", 3, 2, 3);
  scenario("switch the contours to phi (display still phi)", 2, 2, 3);
  scenario("... 60 more frames on phi contours", 2, 2, 60);
  scenario("back to psi contours", 3, 2, 3);
  scenario("both", "both", 2, 3);
  scenario("psi only again (set 1 must go quiet)", 3, 2, 3);
  scenario("contours off", 0, 2, 2);
  scenario("phi contours from off", 2, 2, 3);

  // card retype / chain reuse: close card 0, add a new one on the same chain slot
  console.log("\n-- close the card and re-add on the same chain slot");
  run(`function(){ addDisplayCard({ sel: 0, zsrc: "manual" }); cardsSync(); cardClose(cards.disp[0]);
                   addDisplayCard({ sel: 2, cont: 3, zsrc: "manual" }); cardsSync(); }`);
  drain();
  names();                                   // the re-add built a NEW chain slot: name it too
  const st = run(`function(){ return cards.disp.map(d => ({ ci: d.ci, sel: Number(d.sel()),
                                cont: d.cont(), nlev: d.nlev() })); }`);
  console.log("   cards now: " + JSON.stringify(st));
  run("function(){ for (const d of cards.disp) d.render(); }");
  const ops = drain();
  for (const h of ops) console.log("   " + fmt(h));
  // the re-add must NOT leave the two cards sharing a chain slot, and the closed card's
  // slot is the one the new card is expected to take (that is the reuse under test)
  const slots = st.map(d => d.ci);
  check(new Set(slots).size === slots.length,
        "re-add: two cards share a chain slot " + JSON.stringify(slots));
  check(slots.indexOf(0) >= 0,
        "re-add: the closed card's chain slot 0 was not reused " + JSON.stringify(slots));
  check(ops.length === st.length,
        "re-add: " + ops.length + " colorize dispatch(es) for " + st.length + " cards");
  // and every card must colorize ITS OWN field with ITS OWN contour selection out of ITS
  // OWN chain slot -- a stale chain would show the re-added card wearing the closed one's
  // field/levels, or both cards drawing out of slot 0
  st.forEach((d, i) => {
    const h = ops[i], t = "re-add: card " + i + " (slot " + d.ci + ")";
    if (!h) { check(false, t + ": no colorize dispatch"); return; }
    check(h.field === LAB[d.sel], t + ": colorized " + h.field + ", displays " + LAB[d.sel]);
    checkHit(t, h, d.cont, d.ci, d.nlev);
  });

  console.log("\n" + (fails.length ? "FAILURES (" + fails.length + "):\n  " + fails.join("\n  ")
                                   : "all contour-overlay assertions passed"));
  if (env.fails.length) console.log("stub-level failures:\n  " + env.fails.join("\n  "));
  process.exitCode = (fails.length || env.fails.length) ? 1 : 0;
}, 0);
