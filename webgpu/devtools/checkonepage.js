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
// ---- 1. first visit, 2D: hidden controls, autoplaying Run, toggle writes the memory --
let env = await boot("rmhd2d.html");
env.store.clear();
env = await boot("rmhd2d.html");                 // re-boot with a genuinely empty store
ok(q(env, "controls", "e.style.display") === "none", "2D first visit: controls hidden");
ok(q(env, "btnParams", "e.textContent") === "show params", "2D first visit: show-params label");
// Phase C: a plain visit is already running -- flag, label and colour, all three
ok(env.run("function(){ return running; }") === true, "2D plain visit: autorun set the flag");
ok(label(env, "btnRun") === "Pause", "2D plain visit: Pause label");
ok(q(env, "btnRun", "e.classList._s.stop") === true, "2D plain visit: .stop class (red)");
ok(q(env, "selPreset", "e.value") === "forced", "2D plain visit: the default preset runs");
ok(q(env, "selRes", "e.value") === "256", "2D plain visit: at 256^2");
env.run("function(){ document.getElementById('btnParams').onclick(); }");
ok(q(env, "controls", "e.style.display") === "", "params click: controls shown");
ok(q(env, "btnParams", "e.textContent") === "hide params", "params click: hide-params label");
ok(env.store.getItem("taranisShowParams") === "1", "params click: remembered as shown");

// setRunning coherence, both directions, plus the topbar click path. The autorun left it
// running, so the FIRST click here is the pause.
env.run("function(){ document.getElementById('btnRun').onclick(); }");
ok(q(env, "btnRun", "e.textContent") === "Run", "Run click: back to Run label");
ok(!q(env, "btnRun", "e.classList._s.stop"), "Run click: .stop class dropped");
ok(env.run("function(){ return running; }") === false, "Run click: flag dropped");
env.run("function(){ setRunning(true); }");
ok(q(env, "btnRun", "e.textContent") === "Pause", "setRunning(true): Pause label");
ok(q(env, "btnRun", "e.classList._s.stop") === true, "setRunning(true): .stop class");
ok(env.run("function(){ return running; }") === true, "setRunning(true): flag");
// ... and the autorun is ONE-SHOT: a later preset switch (a user already on the page) is
// not a boot, so it must leave the run state exactly as the user left it
env.run(`function(){ setRunning(false);
  const s = document.getElementById("selPreset"); s.value = "decay"; s.onchange(); }`);
ok(env.run("function(){ return running; }") === false, "preset switch: no second autorun");
ok(label(env, "btnRun") === "Run", "preset switch: still the Run label");
env.run(`function(){ setRunning(true);
  const s = document.getElementById("selPreset"); s.value = "forced"; s.onchange(); }`);
ok(env.run("function(){ return running; }") === true, "preset switch: a running page keeps running");

// ---- 2. second visit: the memory holds, and hiding writes it back ------------------
env = await boot("rmhd2d.html");
ok(q(env, "controls", "e.style.display") === "", "2D return visit: controls stay shown");
ok(q(env, "btnParams", "e.textContent") === "hide params", "2D return visit: hide-params label");
env.run("function(){ document.getElementById('btnParams').onclick(); }");
ok(env.store.getItem("taranisShowParams") === "0", "hide click: remembered as hidden");

// ---- 3. ?demo= visit: lesson text shows even though params are hidden --------------
// ... and a lesson boots PAUSED (Phase C): the hint is there to be read first
env = await boot("rmhd2d.html", "kh");
ok(q(env, "controls", "e.style.display") === "none", "?demo=kh: controls hidden");
ok(q(env, "demohint", "e.innerHTML.length > 0") === true, "?demo=kh: hint filled");
ok(q(env, "demohint", "e.style.display") === "block", "?demo=kh: hint visible");
ok(env.run("function(){ return running; }") === false, "?demo=kh: no autorun");
ok(label(env, "btnRun") === "Run", "?demo=kh: Run label");
ok(!q(env, "btnRun", "e.classList._s.stop"), "?demo=kh: no .stop class");

// ---- 4. 3D page takes the same defaults ---------------------------------------------
env.store.clear();
env = await boot("rmhd3d.html");
ok(q(env, "controls", "e.style.display") === "none", "3D first visit: controls hidden");
ok(env.run("function(){ return running; }") === true, "3D plain visit: autorun set the flag");
ok(label(env, "btnRun") === "Pause", "3D plain visit: Pause label");
ok(q(env, "btnRun", "e.classList._s.stop") === true, "3D plain visit: .stop class (red)");
ok(q(env, "selPreset", "e.value") === "forced", "3D plain visit: the default preset runs");
// the 3D page keeps its own default size (the plan only pins 2D at 256^2)
ok(q(env, "selRes", "e.value") === "128,64", "3D plain visit: the default 3D grid");
// a 3D lesson boots paused too
const env3 = await boot("rmhd3d.html", "collision");
ok(env3.run("function(){ return running; }") === false, "3D ?demo=collision: no autorun");
ok(label(env3, "btnRun") === "Run", "3D ?demo=collision: Run label");

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
ok(rail(env) === "div:railtitle|p:lead|details:|details:|details:|details:|details:",
   "3D page: rail is a title, a lead and five panes");
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

console.log((bad ? "FAIL " : "PASS ") + (n - bad) + "/" + n + " checkonepage");
process.exit(bad ? 1 : 0);
})();
