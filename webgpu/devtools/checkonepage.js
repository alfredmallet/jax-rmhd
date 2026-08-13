// ONEPAGE_PLAN gate.
// Phase A: params hidden by default (+ localStorage memory of the toggle) and the hero
// Run button's label/class/flag coherence via setRunning().
// Phase B: the shared chrome -- the 2D/3D tab strip with the right side active, the
// what-is rail injected from common.js (collapsed, and NOT duplicated in either page's
// markup), and the no-WebGPU poster fallback, driven through the real initGPU failure
// path with stubenv's noGpu knob.
// Phase C: a plain visit boots RUNNING on both pages (flag, label and .stop all three),
// a ?demo= lesson still boots paused, and a no-WebGPU boot never starts a clock.
// Boots real pages on stubenv; two boots in one process = two visits by one browser
// (stubenv's localStorage is process-shared, see the note there).
// Usage: node checkonepage.js <dir>
"use strict";
const fs = require("fs"), path = require("path");
const { execFileSync } = require("child_process");
const dir = process.argv[2] || "..";
let n = 0, bad = 0;
const ok = (cond, msg) => { n++; if (!cond) { bad++; console.log("FAIL  " + msg); } };

// boot() runs at require, but its async tail settles a macrotask later (same trick as
// bootstub's setTimeout wrapper) -- hence the promise hop before touching the page
const boot = (page, demo, opts) => {
  const env = require("./stubenv")(dir, page, demo, opts);
  return new Promise(res => setTimeout(() => res(env)));
};
const q = (env, id, f) => env.run(`function(){ const e = document.getElementById("${id}"); return (${f}); }`);
// what a browser would RENDER: the stub keeps innerHTML (creation, _ctrlItem) and
// textContent (setRunning) as separate fields; in a real DOM the later write wins
const label = (env, id) => q(env, id, "e.textContent || e.innerHTML");

(async () => {
// ---- 1. first visit, 2D: hidden controls, paused green Run, toggle writes the memory -
let env = await boot("rmhd2d.html");
env.store.clear();
env = await boot("rmhd2d.html");                 // re-boot with a genuinely empty store
ok(q(env, "controls", "e.style.display") === "none", "2D first visit: controls hidden");
ok(q(env, "btnParams", "e.textContent") === "show params", "2D first visit: show-params label");
// every visit boots PAUSED (Alfred reversed the Phase C autoplay 2026-08-10): the big
// green Run is the call to action -- flag, label and colour, all three
ok(env.run("function(){ return running; }") === false, "2D plain visit: boots paused");
ok(label(env, "btnRun") === "Run", "2D plain visit: Run label");
ok(!q(env, "btnRun", "e.classList._s.stop"), "2D plain visit: no .stop class (green)");
ok(q(env, "selPreset", "e.value") === "forced", "2D plain visit: the default preset loaded");
ok(q(env, "selRes", "e.value") === "256", "2D plain visit: at 256^2");
// the intro pane under the subtitle: OPEN on a first visit, carrying the two original
// lead paragraphs; closing it once is remembered (the taranisIntro key)
ok(q(env, "intro", "e.open") === true, "2D first visit: intro open");
ok(q(env, "intro", 'e.children[0].innerHTML') === "what is all this?", "intro: summary text");
ok(q(env, "intro", 'e.children[1].children.map(c => c.kind + ":" + c.className).join("|")')
   === "p:lead|p:lead", "intro: the two lead paragraphs");
ok(q(env, "intro", 'e.children[1].children[0].innerHTML.indexOf("browser port") >= 0') === true,
   "intro: the original text");
env.run("function(){ const i = document.getElementById('intro'); i.open = false; i.ontoggle(); }");
ok(env.store.getItem("taranisIntro") === "0", "intro close: remembered");
env.run("function(){ document.getElementById('btnParams').onclick(); }");
ok(q(env, "controls", "e.style.display") === "", "params click: controls shown");
ok(q(env, "btnParams", "e.textContent") === "hide params", "params click: hide-params label");
ok(env.store.getItem("taranisShowParams") === "1", "params click: remembered as shown");

// setRunning coherence, both directions, plus the topbar click path
env.run("function(){ document.getElementById('btnRun').onclick(); }");
ok(q(env, "btnRun", "e.textContent") === "Pause", "Run click: Pause label");
ok(q(env, "btnRun", "e.classList._s.stop") === true, "Run click: .stop class (red)");
ok(env.run("function(){ return running; }") === true, "Run click: flag set");
env.run("function(){ setRunning(false); }");
ok(q(env, "btnRun", "e.textContent") === "Run", "setRunning(false): Run label");
ok(!q(env, "btnRun", "e.classList._s.stop"), "setRunning(false): .stop class dropped");
ok(env.run("function(){ return running; }") === false, "setRunning(false): flag dropped");
// ... and a preset switch (a user already on the page) leaves the run state exactly as
// the user left it, in both directions
env.run(`function(){
  const s = document.getElementById("selPreset"); s.value = "decay"; s.onchange(); }`);
ok(env.run("function(){ return running; }") === false, "preset switch: a paused page stays paused");
ok(label(env, "btnRun") === "Run", "preset switch: still the Run label");
env.run(`function(){ setRunning(true);
  const s = document.getElementById("selPreset"); s.value = "forced"; s.onchange(); }`);
ok(env.run("function(){ return running; }") === true, "preset switch: a running page keeps running");

// ---- 2. second visit: the memory holds, and hiding writes it back ------------------
env = await boot("rmhd2d.html");
ok(q(env, "controls", "e.style.display") === "", "2D return visit: controls stay shown");
ok(q(env, "btnParams", "e.textContent") === "hide params", "2D return visit: hide-params label");
ok(q(env, "intro", "e.open") === false, "2D return visit: dismissed intro stays closed");
env.run("function(){ document.getElementById('btnParams').onclick(); }");
ok(env.store.getItem("taranisShowParams") === "0", "hide click: remembered as hidden");

// ---- 3. ?demo= visit: lesson text shows even though params are hidden --------------
env = await boot("rmhd2d.html", "kh");
ok(q(env, "controls", "e.style.display") === "none", "?demo=kh: controls hidden");
ok(q(env, "demohint", "e.innerHTML.length > 0") === true, "?demo=kh: hint filled");
ok(q(env, "demohint", "e.style.display") === "block", "?demo=kh: hint visible");
ok(env.run("function(){ return running; }") === false, "?demo=kh: boots paused");
ok(label(env, "btnRun") === "Run", "?demo=kh: Run label");
ok(!q(env, "btnRun", "e.classList._s.stop"), "?demo=kh: no .stop class");

// ---- 4. 3D page takes the same defaults ---------------------------------------------
env.store.clear();
env = await boot("rmhd3d.html");
ok(q(env, "controls", "e.style.display") === "none", "3D first visit: controls hidden");
ok(env.run("function(){ return running; }") === false, "3D plain visit: boots paused");
ok(label(env, "btnRun") === "Run", "3D plain visit: Run label");
ok(!q(env, "btnRun", "e.classList._s.stop"), "3D plain visit: no .stop class (green)");
ok(q(env, "selPreset", "e.value") === "forced", "3D plain visit: the default preset loaded");
// the 3D page keeps its own default size (the plan only pins 2D at 256^2)
ok(q(env, "selRes", "e.value") === "128,64", "3D plain visit: the default 3D grid");

// ---- 5. the shared tab strip (Phase B) ----------------------------------------------
// one row per tab: kind, class, text, href, aria-current -- so "active" is checked as
// what a visitor sees (filled, not a link) and not just as a class name
const tabs = e => e.run(`function(){ return document.getElementById("tabs").children.map(
  c => [c.kind, c.className, c.innerHTML, c.href || "-", c.attrs["aria-current"] || "-"].join(",")
).join(" | "); }`);
ok(tabs(env) === "a,tab,2D,rmhd2d.html,- | span,tab on,3D,-,page", "3D page: 3D tab active, 2D a link");
const RAIL = ["what is turbulence?", "what is plasma?", "why should anyone care?",
              "why do numerical simulations?", "technical details"];
const rail = e => e.run(`function(){ return document.getElementById("rail").children.map(
  c => c.kind + ":" + c.className).join("|"); }`);
const panes = e => e.run(`function(){ return document.getElementById("rail").children
  .filter(c => c.kind === "details")
  .map(c => c.children[0].innerHTML + "/" + (c.children[1].innerHTML.length > 200 ? "body" : "SHORT")
            + "/" + (c.open ? "open" : "shut")).join(" | "); }`);
const shut = RAIL.map(s => s + "/body/shut").join(" | ");
ok(rail(env) === "details:|details:|details:|details:|details:",
   "3D page: rail is the five panes (the lead lives in #intro now)");
ok(panes(env) === shut, "3D page: the five panes, filled and collapsed");

env = await boot("rmhd2d.html");
ok(tabs(env) === "span,tab on,2D,-,page | a,tab,3D,rmhd3d.html,-", "2D page: 2D tab active, 3D a link");
ok(panes(env) === shut, "2D page: the same five panes, filled and collapsed");
ok(env.allEls.filter(e => e.kind === "img").length === 0, "working boot: no poster node at all");

// the content lives ONCE, in common.js: no page may carry a copy of it
for (const f of ["rmhd2d.html", "rmhd3d.html", "index.html", "docs.html"]) {
  ok(fs.readFileSync(path.join(dir, f), "utf8").indexOf("what is turbulence") < 0,
     f + ": no duplicated pane content");
}

// ---- 6. no-WebGPU poster fallback ---------------------------------------------------
env.store.setItem("taranisIntro", "0");          // a remembered dismissal, deliberately
env = await boot("rmhd2d.html", null, { noGpu: true });
ok(q(env, "status", "e.className") === "err", "no WebGPU: the advice is still an error status");
ok(/^WebGPU is not available/.test(q(env, "status", "e.textContent")), "no WebGPU: advice text kept");
const imgs = env.allEls.filter(e => e.kind === "img");
ok(imgs.length === 1, "no WebGPU: exactly one poster node");
ok(imgs.length === 1 && imgs[0].src === "poster.png", "no WebGPU: it is poster.png");
ok(imgs.length === 1 && (imgs[0].alt || "").length > 10, "no WebGPU: the poster has alt text");
ok(q(env, "displays", "e.children.length") === 1, "no WebGPU: the poster card is in the display area");
ok(q(env, "status", 'e.parentNode ? e.parentNode.className : "-"') === "card disp",
   "no WebGPU: the advice moved under the poster");
ok(panes(env) === RAIL.map(s => s + "/body/open").join(" | "), "no WebGPU: every pane opened");
// ... the intro too, a remembered dismissal notwithstanding (the store still says "0")
ok(q(env, "intro", "e.open") === true, "no WebGPU: intro forced open");
// ... and no autorun: initGPU returned false, so boot() never reached bootApply and there
// is no solver to run. The button must not claim otherwise.
ok(env.run("function(){ return running; }") === false, "no WebGPU: nothing started running");
ok(label(env, "btnRun") === "Run", "no WebGPU: Run label, not Pause");
ok(!q(env, "btnRun", "e.classList._s.stop"), "no WebGPU: no .stop class");
// review MAJOR 2 / NOTE 11: wireCommonControls is never reached on this path, so the
// control panel must be hidden by the MARKUP itself and the dead topbar buttons disabled
ok(q(env, "controls", "e.style.display") === "none", "no WebGPU: control panel never appears");
for (const id of ["btnRun", "btnReset", "btnParams", "btnText"]) {
  ok(q(env, id, "e.disabled") === true, "no WebGPU: dead " + id + " disabled");
}
ok(env.fails.length === 0, "no WebGPU: no stub-level failures (" + env.fails.join("; ") + ")");

// ---- 7. index.html is a coherent redirect (no script to boot: read the markup) -------
const ix = fs.readFileSync(path.join(dir, "index.html"), "utf8");
ok(/<meta http-equiv="refresh" content="0; url=rmhd2d\.html">/.test(ix), "index.html: meta refresh");
ok(/<link rel="canonical" href="rmhd2d\.html">/.test(ix), "index.html: canonical");
ok(/location\.replace\("rmhd2d\.html"\)/.test(ix), "index.html: JS fallback");
ok(/<a href="rmhd2d\.html">/.test(ix), "index.html: plain link for no-JS, no-refresh");
ok(ix.indexOf('href="style.css"') >= 0 && ix.indexOf('<span class="buildid">dev</span>') >= 0,
   "index.html: still carries what pages.yml seds");
ok(!/<script src=|id="topbar"|id="display"/.test(ix), "index.html: nothing but the redirect");

// ---- 8. every local asset the site references is present AND TRACKED BY GIT ----------
// pages.yml deploys `cp -r webgpu/.` out of a fresh CLONE, so a file that exists in the
// working tree but is not committed is a 404 on the deployed site and a working file://
// page here -- exactly how the favicons shipped broken on 2026-08-13 (.gitignore's
// blanket `*.png` swallowed them, with only poster.png excepted). Existence alone would
// have passed. Hence both legs, and hence the JS scan too: poster.png is referenced from
// common.js, not from any markup, so a markup-only sweep would miss the one image the
// no-WebGPU path depends on.
const ASSET = "png|jpe?g|gif|svg|ico|css|js|json|webmanifest|mp4|webm";
const refs = new Map();                          // ref -> the file that names it
for (const f of fs.readdirSync(dir).filter(f => /\.(html|js)$/.test(f))) {
  const src = fs.readFileSync(path.join(dir, f), "utf8");
  // markup attributes and JS string literals both reduce to "a quoted local path"
  for (const m of src.matchAll(new RegExp(`["'\`]([\\w./-]+\\.(?:${ASSET}))(?:[?#][^"'\`]*)?["'\`]`, "g"))) {
    const r = m[1];
    if (r.startsWith("http") || r.startsWith("//") || r.startsWith("../")) continue;
    if (!refs.has(r)) refs.set(r, f);
  }
}
ok(refs.size > 0, "assets: the sweep found references at all");
let tracked = null;                              // null = no git here, skip the git leg
try {
  execFileSync("git", ["rev-parse", "--git-dir"], { cwd: dir, stdio: "ignore" });
  tracked = new Set(execFileSync("git", ["ls-files"], { cwd: dir, encoding: "utf8" })
                    .split("\n").filter(Boolean));
} catch { console.log("NOTE  assets: no git here, checking existence only"); }
for (const [r, from] of [...refs].sort()) {
  ok(fs.existsSync(path.join(dir, r)), `assets: ${r} exists (referenced by ${from})`);
  if (tracked) ok(tracked.has(r), `assets: ${r} is committed, so it deploys (referenced by ${from})`);
}

console.log((bad ? "FAIL " : "PASS ") + (n - bad) + "/" + n + " checkonepage");
process.exit(bad ? 1 : 0);
})();
