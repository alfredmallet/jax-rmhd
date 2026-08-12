// DOM-stub layout audit: no single flex ITEM in a control row (or a card header) may be
// wider than the viewport, because a flex item does not wrap internally. Widths are
// estimated from the CSS (which this script reads) plus a 7 px/char text metric for the
// UI font at 13px -- generous, so a pass here is a real pass.
//
// Since Phase H.0 the control rows are BUILT from a spec, so the rows are read off the
// live element tree of a booted page (devtools/stubenv.js) instead of the markup.
// Usage: node layout.js [dir]
"use strict";
const fs = require("fs"), path = require("path");
const DIR = process.argv[2] || path.join(__dirname, "..");
const css = fs.readFileSync(path.join(DIR, "style.css"), "utf8");
const CHAR = 7.0, PAD_SELECT = 26, PAD_BUTTON = 20, GAP = 8, CAP_GAP = 6;
const VIEWPORTS = [360, 768, 1200];
const BODY_PAD = { 360: 2 * 10 + 2 * 12, 768: 2 * 14 + 2 * 12, 1200: 2 * 14 + 2 * 12 };  // body + #controls
let bad = 0;
const need = [/\.row\s*\{[^}]*flex-wrap:\s*wrap/, /\.cardhead\s*\{[^}]*flex-wrap:\s*wrap/,
              /#topbar\s*\{[^}]*flex-wrap:\s*wrap/, /\.recres\s*\{[^}]*flex-wrap:\s*wrap/];
for (const re of need) {
  if (!re.test(css)) { bad++; console.log("FAIL  css: no flex-wrap for " + re.source.slice(0, 12)); }
}
console.log((bad ? "" : "PASS  ") + "every wrapping container declares flex-wrap: wrap");
// ... and the one container that must NOT wrap: save + rec are a pair, and a footer wrap
// that split them would read as two unrelated controls (Alfred, 2026-08-12). It is measured
// below as a single item, which is only honest while the CSS says nowrap.
const nowrap = /\.capgrp\s*\{[^}]*flex-wrap:\s*nowrap/.test(css) &&
               /\.capgrp\s*\{[^}]*flex:\s*0 0 auto/.test(css);
if (!nowrap) { bad++; console.log("FAIL  css: .capgrp must be flex-wrap: nowrap and flex: 0 0 auto"); }
else console.log("PASS  the capture group (save + rec) neither wraps nor shrinks");
// text width of an HTML fragment, entities counted as one char
const tw = h => CHAR * String(h == null ? "" : h).replace(/<[^>]+>/g, "").replace(/&[#\w]+;/g, "x").trim().length;

// one built element -> [kind, width]; null for things that take no space of their own
function itemWidth(e) {
  const cls = e.className || "";
  if (e.kind === "select") {
    let w = 0;
    for (const o of e.options) w = Math.max(w, tw(o.innerHTML));
    return ["select", w + PAD_SELECT];
  }
  if (e.kind === "button") return ["button", tw(e.innerHTML) + PAD_BUTTON];
  if (e.kind === "input") {
    if (e.type === "range") return ["range", 160];
    if (e.type === "checkbox") return ["checkbox", 18];
    const wm = /^(\d+)px$/.exec(e.style.width || "");
    return [e.type || "text", wm ? +wm[1] : 90];
  }
  if (e.kind === "label" && cls.indexOf("cbl") >= 0) {
    // a checkbox + its text: one flex item
    let t = "";
    for (const c of e.children) if (c.kind === "#text") t += c.textContent;
    return ["cbl", 18 + 4 + tw(t)];
  }
  if (e.kind === "label") return ["label", tw(e.innerHTML)];
  if (cls.indexOf("val") >= 0) return ["val", 64];
  // save + rec are ONE flex item of the footer (they never wrap apart), so the PAIR is
  // what has to fit: both buttons plus the group's own gap. A hidden `rec` is out of the
  // flow entirely -- no width, no gap -- so the group is then just `save`.
  if (cls.indexOf("capgrp") >= 0) {
    let w = 0, n = 0;
    for (const c of e.children) {
      if (c.style && c.style.display === "none") continue;
      const it = itemWidth(c);
      if (it) { w += (n ? CAP_GAP : 0) + it[1]; n++; }
    }
    return n ? ["save+rec", w] : null;
  }
  // the colorbar block is a fixed-width flex item (strip + tick row), so it must fit
  if (cls.indexOf("cbar") >= 0) return ["colorbar", 134];
  if (cls.indexOf("viewcap") >= 0) return ["caption", 0];  // wraps as text
  // the finished recording's strip: a full-width footer line that wraps INSIDE itself, so
  // as an item of the footer it costs nothing -- its own buttons are measured as a row of
  // their own (CARD_ROWS below), which is where a "download" too wide to fit would show
  if (cls.indexOf("recres") >= 0) return ["result strip", 0];
  if (cls.indexOf("recinfo") >= 0) return ["result text", 0];  // wraps as text
  if (cls.indexOf("hint") >= 0) return ["hint", 0];   // wraps as text, not a fixed item
  return null;
}
// every ROW (a .row div, or the topbar) of a booted page, as [kind, width] lists
function rowsOf(env) {
  const out = [];
  const collect = host => {
    const items = [];
    for (const c of (host.children || [])) {
      const w = itemWidth(c);
      if (w) items.push(w);
    }
    if (items.length) out.push(items);
  };
  collect(env.getEl("topbar"));
  for (const r of env.descendants(env.getEl("controls"), "div")) {
    if ((r.className || "").indexOf("row") >= 0) collect(r);
  }
  return out;
}
// the widest single item of the flex rows the card system builds at runtime: the header,
// and (FEEDBACK_2026-08-10 items 12/13) the display card's FOOTER, which carries the
// colorbar and the save / record buttons on the caption line
const CARD_ROWS = ["cardhead", "viewfoot", "recres"];
function cardHeadItems(env) {
  const heads = [];
  for (const host of [env.getEl("displays"), env.getEl("charts")]) {
    for (const d of env.descendants(host, "div")) {
      if (CARD_ROWS.indexOf(d.className || "") < 0) continue;
      const items = [];
      for (const c of d.children) { const w = itemWidth(c); if (w) items.push(w); }
      if (items.length) heads.push(items);
    }
  }
  return heads;
}
(async () => {
for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
  const env = require("./stubenv")(DIR, page, null);
  await new Promise(r => setTimeout(r, 30));       // boot() is async
  // make sure every card type (and every chart option select) exists before measuring
  env.run(`function(){
    while (cards.disp.length < 1) addDisplayCard();
    for (const t of Object.keys(CHART_TYPES)) if (!cards.chart.some(c => c.type() === t)) addChartCard(t);
    cardsSync();
  }`);
  // a result strip exists only after a save or a take, so hand a card a finished file
  // straight through recResult -- the ONE place all three capture paths converge on. BOTH
  // kinds are fabricated, because both can wait on one footer (two different files, two
  // slots) and each is a row of its own. The stub engine's canShare says yes, so the widest
  // version of each strip (download + share + dismiss) is the one measured.
  env.run(`function(){ const d = cards.disp[0];
    d.recResult("video", new window.Blob([new Uint8Array(1200)], { type: "video/mp4" }),
                "taranis-rmhd-vorticity-t12.345.mp4", 12.3);
    d.recResult("png", new window.Blob([new Uint8Array(900)], { type: "image/png" }),
                "taranis-rmhd-vorticity-t12.345.png"); }`);
  const rows = rowsOf(env).concat(cardHeadItems(env));
  // the two things this round added, said out loud: the capture pair measured as ONE item
  // (a missing group would silently drop out of every row here), and the two strips as two
  // rows of their own
  const grp = rows.reduce((a, items) => items.reduce((b, [k, w]) => k === "save+rec" ? Math.max(b, w) : b, a), 0);
  const onFoot = rows.reduce((a, items) => a + items.filter(([k]) => k === "result strip").length, 0);
  const strips = rows.filter(items => items.some(([k]) => k === "result text")).length;
  if (!grp) { bad++; console.log("FAIL  " + page + ": the footer's save + rec group was not measured at all"); }
  if (strips !== 2 || onFoot !== 2)
    { bad++; console.log("FAIL  " + page + ": " + strips + " result-strip rows (" + onFoot +
                         " on the footer), not 2 (png + video)"); }
  if (grp && strips === 2 && onFoot === 2)
    console.log("PASS  " + page + ": save + rec measured as one " + grp.toFixed(0) +
                "px item, both result strips audited as rows");
  for (const V of VIEWPORTS) {
    const avail = V - BODY_PAD[V];
    let worst = 0, worstKind = "";
    for (const items of rows) {
      for (const [k, w] of items) if (w > worst) { worst = w; worstKind = k; }
    }
    const ok = worst <= avail;
    if (!ok) bad++;
    console.log((ok ? "PASS  " : "FAIL  ") + page + " @ " + V + "px: " + rows.length +
      " rows (controls + card headers), widest single item " + worst.toFixed(0) + "px (" +
      worstKind + ") vs " + avail + "px available");
  }
  // how many wrapped lines the busiest row needs at 360
  const avail = 360 - BODY_PAD[360];
  let maxLines = 0, which = -1;
  rows.forEach((items, i) => {
    let x = 0, lines = 1;
    for (const [, w] of items) { if (x && x + GAP + w > avail) { lines++; x = w; } else x += (x ? GAP : 0) + w; }
    if (lines > maxLines) { maxLines = lines; which = i; }
  });
  console.log("      " + page + " @ 360px: busiest row (#" + which + ") wraps onto " + maxLines + " lines");
}
process.exit(bad ? 1 : 0);
})();
