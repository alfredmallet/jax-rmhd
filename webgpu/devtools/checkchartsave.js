// GATE: the chart card's `save` button (plans-webgpu/IO_PLAN.md item 2), on both booted
// pages under devtools/stubenv.js.
//
// What is under test is the FILE that comes out, never that a handler ran: the stub logs
// every blob (`caps.blobs`), every <a download> click (`caps.downloads`) and now every
// drawImage into a 2D context, so each leg here asserts on the picture, its size, its
// name and where it was delivered.
//
// Four things this exists to catch, all of them ways the button could look right and be
// wrong:
//   - the silent download. Nothing has handed a finished file straight to dlBlob since
//     2026-08-11: it waits on the card's footer behind a download/share button, because on
//     a phone a silent save lands in Files and is then hard to send on. The chart card had
//     no footer at all, so the lifted strip would have fallen through to exactly that.
//   - the lost scale. A `bar` type (gen2d) paints its colour scale into a SECOND canvas,
//     which a toBlob of the plot alone would leave out -- a saved heatmap with no legend.
//   - the vanishing strip. `_barBuild` used to rebuild the whole footer on every retype,
//     so a file waiting on it would have disappeared when the type select moved.
//   - the strip on a dead card. toBlob's callback is deferred, so a card can be closed
//     between the press and the picture; that file must be downloaded, not appended to a
//     node nobody can see.
//
// Run: node devtools/checkchartsave.js [dir]
"use strict";
const DIR = process.argv[2] || __dirname + "/..";
let pass = 0, bad = 0;
function ok(name, cond, note) {
  if (cond) { pass++; console.log("  PASS  " + name + (note ? "   [" + note + "]" : "")); }
  else { bad++; console.log("  FAIL  " + name + (note ? "   [" + note + "]" : "")); }
}
const settle = () => new Promise(r => setTimeout(r, 5));   // toBlob is async, as in a browser

async function boot(page) {
  const env = require("./stubenv")(DIR, page, "");
  for (let i = 0; i < 400 && !env.run("function(){ return !!solver; }"); i++)
    await new Promise(r => setTimeout(r, 0));
  return env;
}
// the chart card of a given type, added if the page has none
const cardOf = (env, t) => env.run(`function(t){
  const c = cards.chart.filter(x => x.type() === t)[0] || addChartCard(t);
  cardsSync(); return c; }`, t);
// what the card's own footer says: the strip, its slot, its buttons, and where it lives
const stripOf = (env, card) => env.run(`function(c){ const s = c.resEl.png;
  const rows = c.foot.children.filter(x => (x.className || "").indexOf("recres") >= 0);
  const btn = s ? s.children.filter(x => x.kind === "button") : [];
  return { on: !!s, rows: rows.length, foot: !!s && s.parentNode === c.foot,
           txt: s ? s.children.filter(x => x.kind === "span").map(x => x.innerHTML).join("") : "",
           btns: btn.map(x => x.innerHTML),
           barFirst: !!c.barD && c.foot.children.indexOf(c.barD) === 0 }; }`, card);
const pressStrip = (env, card, lab) => env.run(`function(a){ const s = a[0].resEl.png;
  const b = s && s.children.filter(x => x.kind === "button" && x.innerHTML === a[1])[0];
  if (!b) return false;
  b.onclick(); return true; }`, [card, lab]);
// every canvas that has had something composited into it, newest last
const composites = env => env.allEls.filter(e => e.kind === "canvas" && e.__cx2d &&
                                            e.__cx2d.__draws.length);

async function page(name) {
  console.log("\n=== " + name + " ===");
  const env = await boot(name);
  if (!env.run("function(){ return !!solver; }")) { ok(name + ": solver came up", false); return env; }
  const slug = name.replace(".html", "");

  // ---- 1. the button is there, and stays there ------------------------------
  const c = cardOf(env, "energy");
  const head = env.run(`function(c){
    const b = c.head.children.filter(x => x.kind === "button");
    return { n: b.length, last: b.map(x => x.innerHTML).join(","), cls: c.btnSave.className,
             ti: c.btnSave.title, inHead: c.btnSave.parentNode === c.head }; }`, c);
  ok("every chart card carries a save button in its header, before the close button",
     head.inHead && /save,&times;$/.test(head.last), head.last);
  ok("  ... styled as a capture button, with a title that says what it makes",
     head.cls === "capbtn" && /PNG/.test(head.ti), head.cls + " / " + head.ti);

  // ---- 2. a save on a plain chart: a PNG on the strip, and nothing downloaded ----
  const nDl0 = env.caps.downloads.length, nBl0 = env.caps.blobs.length;
  env.run("function(c){ c.btnSave.onclick(); }", c);
  await settle();
  // the FIRST new blob is what toBlob wrote; the strip then rewraps it as a File for the
  // share sheet, which is a blob of its own
  const blob = env.caps.blobs[nBl0];
  const st = stripOf(env, c);
  ok("save on a chart with no colour scale produces a PNG",
     !!blob && blob.type === "image/png" && blob.size > 8,
     blob && blob.type + " " + blob.size + " B");
  ok("  ... it goes onto the card's own result strip, one row, with the usual buttons",
     st.on && st.foot && st.rows === 1 && st.btns.join(",") === "download,share,&times;",
     JSON.stringify(st));
  ok("  ... and NOTHING is downloaded by the press itself (the house rule)",
     env.caps.downloads.length === nDl0, env.caps.downloads.length - nDl0 + " downloads");
  // the plain chart is its own canvas: no composite is built at all, so the bytes are the
  // plot's own (the stub sizes a PNG as 4 * w * h of the canvas that made it)
  const plain = env.run("function(c){ return { w: c.cv.width, h: c.cv.height }; }", c);
  ok("  ... captured from the chart canvas itself, with no re-render and no composite",
     blob.size === 4 * plain.w * plain.h, blob.size + " B vs " + 4 * plain.w * plain.h);
  // ---- 3. the name ----------------------------------------------------------
  ok("the download button writes taranis-<app>-<chart>-t<simT>.png",
     pressStrip(env, c, "download") &&
     env.caps.downloads[env.caps.downloads.length - 1].name === "taranis-" + slug + "-energy-t0.000.png",
     env.caps.downloads[env.caps.downloads.length - 1].name);
  ok("  ... and what comes out is the file that was captured, not a second one",
     env.caps.downloads[env.caps.downloads.length - 1].blob === blob);

  // ---- 4. a retype must not take a waiting file with it ---------------------
  const kept = env.run(`function(c){ const was = c.resEl.png;
    c.selType.value = "spectrum"; c.selType.onchange();
    return { same: c.resEl.png === was, foot: !!was && was.parentNode === c.foot,
             type: c.type() }; }`, c);
  ok("retyping the card keeps the pending result strip on its footer",
     kept.same && kept.foot && kept.type === "spectrum", JSON.stringify(kept));
  ok("  ... and the file it still holds downloads under its ORIGINAL name",
     pressStrip(env, c, "download") &&
     env.caps.downloads[env.caps.downloads.length - 1].name === "taranis-" + slug + "-energy-t0.000.png",
     env.caps.downloads[env.caps.downloads.length - 1].name);
  env.run("function(c){ c.recClear('png'); }", c);

  // ---- 5. capture delivers nothing on its own -------------------------------
  // the split item 3 (save all) will build on: captureShot resolves to the blob and hands
  // it to nobody, so N captures can be collected before one archive is delivered
  const nBl = env.caps.blobs.length, nDl1 = env.caps.downloads.length;
  const capOnly = await env.run("function(c){ return c.captureShot(); }", c);
  const after = stripOf(env, c);
  ok("captureShot() resolves to the blob and delivers nothing",
     !!capOnly && capOnly.type === "image/png" && env.caps.blobs.length === nBl + 1 &&
     env.caps.downloads.length === nDl1 && !after.on,
     JSON.stringify({ strip: after.on, dl: env.caps.downloads.length - nDl1 }));

  // ---- 6. a card closed between the press and the picture -------------------
  const gone = cardOf(env, "cut");
  const nDl2 = env.caps.downloads.length;
  env.run("function(c){ c.btnSave.onclick(); cardClose(c); }", gone);
  await settle();
  const dead = env.caps.downloads.slice(nDl2);
  ok("a chart card closed mid-save downloads the picture instead of losing it",
     dead.length === 1 && dead[0].name === "taranis-" + slug + "-cut-t0.000.png" &&
     env.run("function(c){ return c.dead === true && !c.resEl.png; }", gone),
     JSON.stringify(dead.map(d => d.name)));

  ok("the page raised no stub failures", env.fails.length === 0, env.fails.join(" | "));
  return env;
}

(async () => {
  await page("rmhd2d.html");
  const env = await page("rmhd3d.html");

  // ---- 7. the composite: a `bar` type must save its colour scale -------------
  // gen2d is the one type that declares `bar`, and it is 3D-only. It needs no data for
  // this: what is under test is that the SECOND canvas -- the one this.cv.toBlob() cannot
  // see -- ends up in the file, at the geometry the card itself draws it at.
  console.log("\n=== the bar composite (rmhd3d.html, gen2d) ===");
  const g = cardOf(env, "gen2d");
  const before = composites(env).length;
  const nDl = env.caps.downloads.length, nBl = env.caps.blobs.length;
  env.run("function(c){ c.btnSave.onclick(); }", g);
  await settle();
  const blob = env.caps.blobs[nBl];
  const comp = composites(env)[before];
  const geom = env.run(`function(c){ return { w: c.cv.width, h: c.cv.height,
    bar: c.barCv.width, barh: c.barCv.height, T: CHART_TYPES[c.type()].w }; }`, g);
  ok("a gen2d save composites onto a taller canvas than the plot",
     !!comp && comp.width === geom.w && comp.height > geom.h,
     comp && comp.width + "x" + comp.height + " from " + geom.w + "x" + geom.h);
  const srcs = comp ? comp.__cx2d.__draws.map(d => d.src) : [];
  ok("  ... the plot and the colour-scale canvas are BOTH drawn into it, in that order",
     srcs.length === 2 &&
     env.run("function(a){ return a[0] === a[1].cv && a[2] === a[1].barCv; }", [srcs[0], g, srcs[1]]),
     srcs.length + " drawImage calls");
  // the band under the plot is the on-screen bar's own geometry at the canvas's own dpr:
  // the strip's height plus the label line, i.e. the plate the display card's stamp uses
  const CB = env.run("function(){ return { w: CBAR_W, h: CBAR_H }; }");
  const sc = Math.max(1, geom.w / geom.T);
  ok("  ... in a band sized by the shared colorbar geometry, at the canvas's own scale",
     !!comp && comp.height - geom.h === Math.round(CB.h * sc + 4.4 * 6 * sc),
     comp && (comp.height - geom.h) + " px band, sc = " + sc.toFixed(2));
  ok("  ... and the bytes that came out are that composite, not the plot alone",
     !!blob && blob.size === 4 * comp.width * comp.height,
     blob && blob.size + " B vs " + (comp && 4 * comp.width * comp.height));
  const st = stripOf(env, g);
  ok("  ... delivered to the strip, under the plot's own colour scale, with no download",
     st.on && st.foot && st.rows === 1 && st.barFirst &&
     env.caps.downloads.length === nDl, JSON.stringify(st));
  ok("  ... named for the chart type",
     pressStrip(env, g, "download") &&
     env.caps.downloads[env.caps.downloads.length - 1].name === "taranis-rmhd3d-gen2d-t0.000.png",
     env.caps.downloads[env.caps.downloads.length - 1].name);
  // the retype trap on the type that has a bar: the footer holds both, and the bar must
  // come back ABOVE the waiting strip rather than under it
  const round = env.run(`function(c){ const was = c.resEl.png;
    c.selType.value = "energy"; c.selType.onchange();
    const away = !!c.barD + ":" + (c.resEl.png === was);
    c.selType.value = "gen2d"; c.selType.onchange();
    return { away: away, back: !!c.barD, same: c.resEl.png === was,
             order: c.foot.children.map(x => x.className).join(",") }; }`, g);
  ok("gen2d -> energy -> gen2d keeps the file and puts the rebuilt scale back above it",
     round.away === "false:true" && round.back && round.same && round.order === "cbar,recres",
     JSON.stringify(round));

  // ---- 8. the display card's own path is unchanged ---------------------------
  const d = env.run("function(){ return cards.disp[0]; }");
  const nDlD = env.caps.downloads.length;
  env.run("function(d){ d.btnSave.onclick(); }", d);
  await settle();
  const stD = env.run(`function(d){ const s = d.resEl.png;
    return { on: !!s, foot: !!s && s.parentNode === d.foot }; }`, d);
  ok("the display card still delivers its save to the same strip, through the same code",
     stD.on && stD.foot && env.caps.downloads.length === nDlD, JSON.stringify(stD));
  const was = env.run("function(d){ return d.resEl.png; }", d);
  const nBlD = env.caps.blobs.length;
  const dcap = await env.run("function(d){ return d.captureShot(); }", d);
  ok("  ... and DisplayCard.captureShot() resolves to a PNG without delivering it",
     !!dcap && dcap.type === "image/png" && env.caps.blobs.length === nBlD + 1 &&
     env.run("function(a){ return cards.disp[0].resEl.png === a; }", was) &&
     env.caps.downloads.length === nDlD,
     dcap && dcap.type);

  console.log("\n" + (bad ? "FAIL" : "PASS") + "  chart save: " + pass + " checks passed, " +
              bad + " failed");
  process.exit(bad ? 1 : 0);
})();
