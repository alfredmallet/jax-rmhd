// checkgc.js -- gate for ANALYTICS_PLAN.md (GoatCounter beacon + the contact line).
// Usage: node checkgc.js
//
// Two halves. The first is text-level over the five HTML files and needs no DOM: it
// pins the SET of counted pages, which is the load-bearing decision of the whole
// feature. The second boots both apps under stubenv -- including the no-WebGPU path,
// where the contact link matters most -- and checks what contactBuild actually produced.
//
// The trap this exists to catch: index.html is a pure redirect to rmhd2d.html AND carries
// <link rel="canonical" href="rmhd2d.html">, so a beacon there would file every arrival
// at the bare /webgpu/ URL under rmhd2d's path and silently inflate the one number the
// feature exists to produce. A later well-meaning "you missed a page" edit must fail
// loudly here rather than quietly corrupt the stats.
"use strict";
const fs = require("fs"), path = require("path");

const DIR = path.join(__dirname, "..");
const COUNTED = ["rmhd2d.html", "rmhd3d.html", "docs.html", "reading.html"];
const UNCOUNTED = ["index.html"];
const ENDPOINT = "https://taranis.goatcounter.com/count";
const PLACEHOLDER = "ALFRED_GOATCOUNTER_CODE";
const ADDR = "alfred.mallet" + String.fromCharCode(64) + "berkeley.edu";

let pass = 0;
const bad = [];
const ok = (cond, what) => { if (cond) pass++; else bad.push(what); };
const read = f => fs.readFileSync(path.join(DIR, f), "utf8");

// ---- part 1: the beacon, text-level -------------------------------------------------
const src = {};
for (const f of COUNTED.concat(UNCOUNTED)) src[f] = read(f);

for (const f of COUNTED) {
  const s = src[f];
  const tags = s.match(/data-goatcounter=/g) || [];
  ok(tags.length === 1, f + ": expected exactly 1 data-goatcounter attribute, got " + tags.length);
  const m = s.match(/data-goatcounter="([^"]+)"/);
  ok(m && m[1] === ENDPOINT, f + ": endpoint is " + (m ? m[1] : "absent") + ", expected " + ENDPOINT);
  ok(/src="\/\/gc\.zgo\.at\/count\.js"/.test(s), f + ": count.js src missing or changed");
  ok(/<script[^>]*\basync\b/.test(s), f + ": the beacon must be async (it must not be able to block boot)");
  // the canonical is what stops ?fresh= / ?demo= scattering one page across many paths
  const c = s.match(/<link rel="canonical" href="([^"]+)">/);
  ok(c && c[1] === f, f + ": canonical is " + (c ? c[1] : "absent") + ", expected its own filename " + f);
}

for (const f of UNCOUNTED) {
  ok(!/data-goatcounter/.test(src[f]),
     f + ": MUST NOT carry a beacon -- it is a redirect to rmhd2d.html and its canonical "
     + "points there, so counting it double-counts every arrival at the bare /webgpu/ URL");
}
ok(/<link rel="canonical" href="rmhd2d\.html">/.test(src["index.html"]),
   "index.html: its existing canonical to rmhd2d.html must stay (bookmarks + crawlers)");

// the site code must be real before this can ship
for (const f of COUNTED) {
  ok(src[f].indexOf(PLACEHOLDER) < 0,
     f + ": still carries the placeholder -- register the site at goatcounter.com and "
     + "replace " + PLACEHOLDER + " in all four files");
}

// ---- part 2: the contact line, text-level -------------------------------------------
for (const f of ["rmhd2d.html", "rmhd3d.html"]) {
  const s = src[f];
  ok(/<span id="contact"><\/span>/.test(s), f + ": #contact hook missing or not empty in the markup");
  ok(/<span class="buildid">/.test(s), f + ": the buildid span pages.yml seds is gone");
  // the leading separator must NOT be in the markup: an empty #contact after a markup
  // "·" leaves a dangling bullet on any page where common.js failed to load
  ok(!/&middot;\s*<span id="contact">/.test(s),
     f + ": the separator before #contact belongs in contactBuild, not the markup");
}

// The address must be absent from EVERY file the deploy publishes -- not just the HTML
// and common.js. pages.yml stages the site with `cp -r webgpu/. _site/webgpu/`, which
// copies this directory wholesale: .md plans, devtools/, all of it. A plan file spelling
// the address out in full would defeat the entire runtime-assembly exercise while every
// narrower check stayed green. (It did, once. That is why this walks the tree.)
const SKIP_DIRS = new Set(["node_modules", "__pycache__"]);
function walk(rel) {
  const abs = path.join(DIR, rel);
  for (const name of fs.readdirSync(abs)) {
    const r = rel ? path.join(rel, name) : name;
    const st = fs.statSync(path.join(DIR, r));
    if (st.isDirectory()) { if (!SKIP_DIRS.has(name)) walk(r); continue; }
    if (/\.(png|jpg|jpeg|gif|webm|mp4|woff2?|ico)$/i.test(name)) continue;
    let body;
    try { body = fs.readFileSync(path.join(DIR, r), "utf8"); } catch (e) { continue; }
    ok(body.indexOf(ADDR) < 0, r + ": the address appears CONTIGUOUSLY in a file the "
       + "deploy publishes (pages.yml copies all of webgpu/) -- de-literal it");
  }
}
walk("");

// contactBuild must be called BEFORE chromeBuild's `if (!rail) return` early exit -- the
// contact link is the one piece of chrome that has to survive a degraded boot, and that
// ordering is the argument, not an accident. Text-level because the early exit cannot be
// reached from a real page (both apps declare #rail), so no boot can pin it.
const CJS = fs.readFileSync(path.join(DIR, "common.js"), "utf8");
const cb = CJS.indexOf("function chromeBuild(");
const call = CJS.indexOf("contactBuild(", cb);
const exit = CJS.indexOf("if (!rail) return", cb);
ok(cb > 0 && call > cb && exit > call,
   "common.js: contactBuild() must be called inside chromeBuild and BEFORE its "
   + "`if (!rail) return` early exit, or a page without a rail loses the contact link");

// ---- part 3: booted, both apps, with and without WebGPU -----------------------------
// async because initGPU is: the stash this checks happens inside an awaited requestAdapter.
async function booted() {
for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
  for (const noGpu of [false, true]) {
    const tag = page + (noGpu ? " [no-WebGPU]" : "");
    let env;
    try {
      env = require("./stubenv")(DIR, page, "", noGpu ? { noGpu: true } : {});
    } catch (e) {
      bad.push(tag + ": boot threw -- " + e.message);
      continue;
    }
    ok(env.fails.length === 0, tag + ": stub reported failures -- " + env.fails.join("; "));

    const host = env.getEl("contact");
    const kids = (host.children || []).filter(e => e.kind === "a" || e.tag === "a");
    ok(kids.length === 2, tag + ": #contact should hold 2 anchors, holds " + kids.length);
    if (kids.length !== 2) continue;

    const [mail, issues] = kids;
    ok(/^mailto:/.test(mail.href || ""), tag + ": first anchor is not a mailto");
    ok((mail.href || "").indexOf(ADDR) >= 0, tag + ": mailto does not carry the assembled address");
    ok(/issues\/new$/.test(issues.href || ""), tag + ": second anchor is not the issues link");
    ok(issues.rel === "noopener" && issues.target === "_blank",
       tag + ": the issues link needs target=_blank rel=noopener like every other external link");

    // stubenv evaluates the page but does NOT run initGPU (bootstub does), so drive it
    // here: gpuInfo is stashed in there, and the point of building the body at click
    // time rather than at build time is that it picks that up.
    if (!noGpu) {
      let gpuThrew = null;
      try { await env.run("function(){ return initGPU(); }"); } catch (e) { gpuThrew = e; }
      ok(!gpuThrew, tag + ": initGPU threw under the stub -- " + (gpuThrew && gpuThrew.message));
    }

    // the diagnostics are attached on the way OUT, so drive the handler
    let threw = null;
    try { if (mail.onclick) mail.onclick(); } catch (e) { threw = e; }
    ok(!threw, tag + ": the mailto click handler threw -- " + (threw && threw.message));
    // decode DEFENSIVELY: the handler truncates the encoded query at 1500 chars, and a
    // cut that lands mid-escape makes this throw. A malformed URI is a finding, not a
    // crash -- report it as the named leg it is.
    let href = "", decodeThrew = null;
    try { href = decodeURIComponent(mail.href || ""); } catch (e) { decodeThrew = e; }
    ok(!decodeThrew, tag + ": the mailto href is not decodable -- the length cap cut a "
       + "percent-escape in half (" + (decodeThrew && decodeThrew.message) + ")");
    ok(/[?&]subject=/.test(mail.href || ""), tag + ": no prefilled subject");
    ok(href.indexOf(page === "rmhd3d.html" ? "3D" : "2D") > 0,
       tag + ": the subject does not name the page, so 2D and 3D reports arrive identical");
    ok(/browser: /.test(href), tag + ": the body carries no userAgent");
    ok(/build: /.test(href), tag + ": the body carries no build id");
    ok(/screen: \d+x\d+/.test(href), tag + ": the body carries no viewport size (or reports "
       + "it as undefinedxundefined -- push() treats a concatenation as always-truthy)");
    ok(/webgpu: /.test(href), tag + ": the body does not say whether WebGPU was available");
    ok((mail.href || "").length <= 1600, tag + ": mailto href is " + (mail.href || "").length
       + " chars -- mail clients trim past ~1500-2000");
    // Drive the length cap for real: a pathological userAgent pushes the encoded query
    // past 1500 so the truncation branch executes, and the result must still decode.
    // Blind slicing fails this ~1 time in 4.
    try {
      env.run("function(s){ navigator.userAgent = s; }", "x".repeat(400) + " (é%20 ünïcode ✓)".repeat(60));
      if (mail.onclick) mail.onclick();
      let cut = null;
      try { decodeURIComponent(mail.href || ""); } catch (e) { cut = e; }
      ok(!cut, tag + ": a body long enough to hit the 1500-char cap produced an "
         + "undecodable href -- the cut split a percent-escape");
      ok((mail.href || "").length <= 1600, tag + ": the length cap did not hold");
    } catch (e) {
      bad.push(tag + ": could not exercise the length cap -- " + e.message);
    }

    if (noGpu) {
      ok(/webgpu: NO/.test(href), tag + ": the no-WebGPU boot must report webgpu: NO");
      ok(!/\bgpu: /.test(href), tag + ": a boot with no adapter must NOT claim to name a GPU");
    } else {
      // gpuInfo is stashed in initGPUTry; on a browser without adapter.info it stays ""
      // and this line is honestly absent, so the assertion belongs to the stub (which
      // does report info) and not to the feature.
      ok(/\bgpu: stub/.test(href), tag + ": the adapter description did not reach the body");
    }
  }
}
}

// -------------------------------------------------------------------------------------
booted().then(() => {
  if (bad.length) {
    console.log("FAIL " + pass + "/" + (pass + bad.length) + " checkgc");
    for (const b of bad) console.log("  - " + b);
    process.exit(1);
  }
  console.log("PASS " + pass + "/" + pass + " checkgc");
}, e => { console.log("FAIL checkgc -- threw: " + (e && e.stack)); process.exit(1); });
