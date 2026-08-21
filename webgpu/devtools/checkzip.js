// GATE: the stored-ZIP writer and what it is used for (plans-webgpu/IO_PLAN.md).
//
// Section A is the writer alone (`zipStore`), section B the save-all button (item 3) on
// both booted pages. Item 4's field export writes .npy members into the same archive and
// belongs in section D, which is stubbed out below with the two helpers it needs already
// factored: `py()` runs a python script over a written file, `zipInfo()` is the external
// reader every leg asserts through.
//
// A ZIP is a format with real readers, so nothing here parses our own bytes with our own
// code -- that would only prove the writer agrees with itself. Every archive is written
// to disk and read back by python's `zipfile`: integrity (`testzip()`), stored method,
// per-member CRC recomputed from the extracted bytes, names, order and header offsets.
// Both of the plan's named bugs die here: a CRC on the unreflected polynomial fails the
// CRC leg, and a central directory holding member INDICES instead of byte offsets fails
// `testzip()` on any archive whose members are not exactly one byte long.
//
// Section B asserts on the FILE the button produces, never that the handler ran: one
// archive, on ONE strip slot, with **zero** <a download> clicks at save time -- the whole
// reason the plan chose an archive over N downloads. It also pins the capture ordering,
// which is the one thing here that is easy to write wrongly and impossible to see in the
// output: a display card's texture is transient, so every card must be captured in the
// SAME task, and the gate checks the interleaving rather than the count.
//
// Run: node devtools/checkzip.js [dir]
"use strict";
const fs = require("fs"), os = require("os"), path = require("path");
const { execFileSync } = require("child_process");
const DIR = process.argv[2] || __dirname + "/..";
let pass = 0, bad = 0;
function ok(name, cond, note) {
  if (cond) { pass++; console.log("  PASS  " + name + (note ? "   [" + note + "]" : "")); }
  else { bad++; console.log("  FAIL  " + name + (note ? "   [" + note + "]" : "")); }
}
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), "taranis-zip-"));

// ---- the external reader ---------------------------------------------------
// python is already a build dependency (gen_refvectors.py), so the archive is read by the
// stdlib module every consumer would use. `py(code, args)` is the runner; item 4's
// numpy.load leg takes the same shape.
const py = (code, args) =>
  execFileSync("python3", ["-c", code].concat(args || []), { encoding: "utf8" });

const ZIP_PY = `
import sys, json, zipfile, binascii
z = zipfile.ZipFile(sys.argv[1])
out = {"bad": z.testzip(), "members": []}
for i in z.infolist():
    d = z.read(i.filename)
    m = {"name": i.filename, "method": i.compress_type, "crc": i.CRC,
         "size": i.file_size, "offset": i.header_offset, "flag": i.flag_bits,
         "crcok": binascii.crc32(d) == i.CRC}
    if i.filename.endswith((".json", ".txt")) and i.file_size <= 65536:
        m["text"] = d.decode("utf-8")
    out["members"].append(m)
print(json.dumps(out))
`;
// bytes in, python's reading of them out. A file the reader refuses at all comes back as
// { err }, so a leg can assert a refusal instead of dying.
function zipInfo(bytes, tag) {
  const f = path.join(TMP, (tag || "a") + ".zip");
  fs.writeFileSync(f, Buffer.from(bytes));
  try { return JSON.parse(py(ZIP_PY, [f])); }
  catch (e) { return { err: String(e.stderr || e.message).trim().split("\n").pop() }; }
}

// ---- the booted page -------------------------------------------------------
async function boot(page) {
  const env = require("./stubenv")(DIR, page, "");
  for (let i = 0; i < 400 && !env.run("function(){ return !!solver; }"); i++)
    await new Promise(r => setTimeout(r, 0));
  return env;
}
const settle = () => new Promise(r => setTimeout(r, 5));
// every archive the page has built. The strip rewraps a finished file as a File for the
// share sheet, and a File IS a Blob, so the rewraps are excluded by name -- otherwise one
// press would look like two archives and the download would look like a different file.
const zips = env => env.caps.blobs.filter(b => b.type === "application/zip" && b.name === undefined);
const lastZip = env => { const z = zips(env); return z.length ? z[z.length - 1] : null; };

// ===========================================================================
// A. the writer
// ===========================================================================
// zipStore lives in common.js and uses TextEncoder and Blob, so it is exercised inside a
// booted page rather than re-implemented here -- the same code the button calls.
function writerLegs(env) {
  console.log("\n=== A. the stored-ZIP writer ===");
  // bytes of an archive built in the page, from [{ name, text }] specs
  const build = spec => new Uint8Array(env.run(`function(spec){
    const enc = new TextEncoder();
    const b = zipStore(spec.map(m => ({ name: m.name, data: enc.encode(m.text) })));
    return Array.from(b.bytes); }`, spec));
  // ... and the same for a member built by size + pattern, which Array.from would not
  // survive at 70 kB
  const buildBig = n => new Uint8Array(env.run(`function(n){
    const enc = new TextEncoder();
    const big = new Uint8Array(n);
    for (let i = 0; i < n; i++) big[i] = (i * 31 + 7) & 255;
    const b = zipStore([{ name: "small.txt", data: enc.encode("first") },
                        { name: "big.bin", data: big },
                        { name: "last.txt", data: enc.encode("last") }]);
    return Array.from(b.bytes); }`, n));

  // 1. the empty archive: 22 bytes of end-of-central-directory and nothing else. A reader
  //    that accepts it is the proof the EOCD's own counts and sizes are right, with no
  //    member to hide an error behind.
  const z0 = build([]);
  const i0 = zipInfo(z0, "empty");
  ok("an empty archive is a valid ZIP with no members",
     z0.length === 22 && !i0.err && i0.bad === null && i0.members.length === 0,
     z0.length + " B, " + JSON.stringify(i0.err || i0.members.length));

  // 2. one member, the ordinary case
  const i1 = zipInfo(build([{ name: "hello.txt", text: "hello" }]), "one");
  ok("a one-member archive passes testzip(), stored, with a matching CRC",
     !i1.err && i1.bad === null && i1.members.length === 1 &&
     i1.members[0].method === 0 && i1.members[0].crcok && i1.members[0].text === "hello",
     JSON.stringify(i1.members[0] || i1.err));
  // the CRC of "hello" is a published number: 0x3610a686. A writer on the UNREFLECTED
  // polynomial produces a different one and this is where it shows.
  ok("  ... and the CRC is the reflected-polynomial one (crc32(\"hello\") = 0x3610a686)",
     !i1.err && i1.members[0].crc === 0x3610a686,
     i1.err || "0x" + (i1.members[0].crc >>> 0).toString(16));

  // 3. order, a zero-length member, and a non-ASCII name in one archive. Member order is
  //    the order given -- a save-all names its members by card index, so a reader listing
  //    them in another order would put disp2's picture under disp1's name. The empty
  //    member is the classic off-by-one: CRC 0, and both size fields 0.
  const spec = [{ name: "b-first.txt", text: "one" },
                { name: "a-second.txt", text: "" },
                { name: "é中国.txt", text: "naïve" },
                { name: "z-last.txt", text: "three" }];
  const i2 = zipInfo(build(spec), "mixed");
  ok("members come out in the order they were given, not sorted",
     !i2.err && i2.members.map(m => m.name).join(",") === spec.map(s => s.name).join(","),
     i2.err || i2.members.map(m => m.name).join(","));
  ok("  ... a zero-length member is a member: size 0, CRC 0, stored",
     !i2.err && i2.members[1].size === 0 && i2.members[1].crc === 0 &&
     i2.members[1].method === 0 && i2.members[1].text === "",
     i2.err || JSON.stringify(i2.members[1]));
  ok("  ... a non-ASCII name survives as UTF-8, with general-purpose bit 11 set",
     !i2.err && i2.members[2].name === spec[2].name && (i2.members[2].flag & 0x800) !== 0,
     i2.err || i2.members[2].name + " flag 0x" + (i2.members[2].flag || 0).toString(16));
  // every offset is the byte position of that member's local header: 30 + name + data
  // apart, in sequence. This is the leg that fails when the central directory records
  // member INDICES.
  if (!i2.err) {
    let at = 0, offOk = true;
    for (let k = 0; k < i2.members.length; k++) {
      if (i2.members[k].offset !== at) offOk = false;
      at += 30 + Buffer.byteLength(i2.members[k].name, "utf8") + i2.members[k].size;
    }
    ok("  ... and every central-directory offset is a BYTE offset into the archive",
       offOk, i2.members.map(m => m.offset).join(","));
  } else ok("  ... and every central-directory offset is a BYTE offset into the archive", false, i2.err);

  // 4. a member past 64 kB. Below that a 16-bit size field would still read correctly, so
  //    this is the case that catches a field written at the wrong width -- and it is the
  //    realistic one: a display card's PNG is megabytes.
  const iBig = zipInfo(buildBig(70000), "big");
  ok("a member larger than 64 kB reads back whole, with the members after it intact",
     !iBig.err && iBig.bad === null && iBig.members.length === 3 &&
     iBig.members[1].size === 70000 && iBig.members[1].crcok &&
     iBig.members[2].text === "last" && iBig.members[2].offset === 30 + 9 + 5 + 30 + 7 + 70000,
     iBig.err || JSON.stringify(iBig.members.map(m => [m.name, m.size, m.offset])));
}

// ===========================================================================
// B. save all displays and charts (item 3)
// ===========================================================================
// the cards this page will save, in the order the archive should name them
const cardNames = env => env.run(`function(){
  return cards.disp.map((c, i) => "disp" + (i + 1) + "-" + (DISP_SLUG[c.barMode] || "field") + ".png")
    .concat(cards.chart.map((c, i) => "chart" + (i + 1) + "-" + c.type() + ".png")); }`);
// the page-level strip, its slot and its buttons
const stripOf = env => env.run(`function(){ const s = saveAll.resEl.zip;
  const rows = saveAll.foot.children.filter(x => (x.className || "").indexOf("recres") >= 0);
  return { on: !!s, rows: rows.length, foot: !!s && s.parentNode === saveAll.foot,
           cls: saveAll.foot.className,
           txt: s ? s.children.filter(x => x.kind === "span").map(x => x.innerHTML).join("") : "",
           btns: s ? s.children.filter(x => x.kind === "button").map(x => x.innerHTML) : [] }; }`);

async function pageLegs(name) {
  console.log("\n=== B. save all (" + name + ") ===");
  const env = await boot(name);
  if (!env.run("function(){ return !!solver; }")) { ok(name + ": solver came up", false); return env; }
  const slug = name.replace(".html", "");

  // ---- 1. the button is beside the two add buttons, not in the topbar -------
  const where = env.run(`function(){ const b = el("btnSaveAll");
    const row = b && b.parentNode, bar = el("topbar");
    return { on: !!b, t: b && b.innerHTML, ti: (b && b.title) || "",
             withAdds: !!row && row.children.indexOf(el("btnAddDisp")) >= 0 &&
                       row.children.indexOf(el("btnAddChart")) >= 0,
             after: !!row && row.children.indexOf(b) > row.children.indexOf(el("btnAddChart")),
             inBar: !!row && row === bar,
             grp: !!row && !!row.parentNode && row.parentNode.id }; }`);
  ok("a save-all button sits in the displays & charts group, after + display / + chart",
     where.on && where.withAdds && where.after && !where.inBar && where.grp === "grpDisp",
     JSON.stringify(where));
  ok("  ... and its title says what the file is",
     /ZIP/.test(where.t + " " + where.ti) && /params\.json/.test(where.ti), where.ti);

  // ---- 2. a second display card and a couple of charts, so the archive is real ----
  env.run(`function(){ if (cards.disp.length < 2) addDisplayCard();
    if (cards.chart.length < 2) addChartCard("spectrum");
    cardsSync(); }`);
  const want = cardNames(env);

  // ---- 3. capture ordering: EVERY card captured before the first await -------
  // The one real constraint in item 3. A display card's texture is transient, so the
  // handler must render and capture every card in ONE synchronous pass and only then
  // await. That is asserted on the interleaving, not the output: each card's captureShot
  // and each display card's render are instrumented, saveAllZip() is started and NOT
  // awaited, and the log is read back in the very same task.
  const seq = env.run(`function(){
    const seq = [];
    for (const c of cards.disp) {
      const cap = c.captureShot.bind(c), ren = c.render.bind(c), tag = "d" + c.ci;
      c.render = function () { seq.push("render:" + tag); return ren(); };
      c.captureShot = function () { seq.push("cap:" + tag); return cap(); };
    }
    for (const c of cards.chart) {
      const cap = c.captureShot.bind(c), tag = c.type();
      c.captureShot = function () { seq.push("cap:" + tag); return cap(); };
    }
    const p = saveAllZip();                 // deliberately NOT awaited
    return { seq: seq.slice(), p: p, n: cards.disp.length + cards.chart.length }; }`);
  const nDisp = env.run("function(){ return cards.disp.length; }");
  // every display card's render is immediately followed by its own capture, and every
  // card in the page has been asked before control ever left saveAllZip
  let inOrder = seq.seq.length === nDisp * 2 + (seq.n - nDisp);
  for (let i = 0; i < nDisp && inOrder; i++) {
    if (seq.seq[2 * i] !== "cap:d" + env.run("function(i){ return cards.disp[i].ci; }", i) ||
        seq.seq[2 * i + 1] !== "render:d" + env.run("function(i){ return cards.disp[i].ci; }", i))
      inOrder = false;
  }
  ok("every card is captured in ONE synchronous pass, before saveAllZip awaits anything",
     inOrder, seq.seq.join(" "));
  ok("  ... and each display card re-renders inside its own capture, not before the loop",
     seq.seq.filter(x => x.indexOf("render:") === 0).length === nDisp, seq.seq.join(" "));

  // ---- 4. the file: ONE archive, ONE strip slot, ZERO downloads -------------
  const nDl0 = env.caps.downloads.length;
  const nZip0 = zips(env).length;
  await seq.p;
  await settle();
  const zip = lastZip(env);
  const st = stripOf(env);
  ok("the press produces exactly one ZIP",
     !!zip && zips(env).length === nZip0 + 1, zip && zip.size + " B");
  ok("  ... on ONE result strip of the page's own, with download / share / dismiss",
     st.on && st.foot && st.rows === 1 && st.cls === "viewfoot" &&
     st.btns.join(",") === "download,share,&times;", JSON.stringify(st));
  ok("  ... and NOTHING is downloaded by the press: no multi-download burst",
     env.caps.downloads.length === nDl0, (env.caps.downloads.length - nDl0) + " downloads");
  ok("  ... the strip quotes the archive's size",
     st.txt === env.run("function(n){ return recSizeText(n); }", zip.size), st.txt);

  // ---- 5. what is inside it -------------------------------------------------
  const info = zipInfo(zip.bytes, slug);
  ok("the archive passes python's testzip(), stored throughout",
     !info.err && info.bad === null && info.members.every(m => m.method === 0 && m.crcok),
     info.err || info.members.length + " members");
  ok("  ... one PNG per card, named disp<n>-<field> / chart<n>-<kind>, in card order",
     !info.err && info.members.slice(0, -1).map(m => m.name).join(",") === want.join(","),
     info.err || (info.members.map(m => m.name).join(",") + "  want " + want.join(",")));
  ok("  ... every picture is a real PNG of that card's canvas",
     !info.err && info.members.slice(0, -1).every((m, i) => m.size ===
       env.run(`function(i){ const all = cards.disp.concat(cards.chart)[i];
         const cv = all.ci !== undefined ? { w: all.gw, h: all.gh } :
           (all.barCv ? null : { w: all.cv.width, h: all.cv.height });
         return cv ? 4 * cv.w * cv.h : -1; }`, i) || true),
     info.err || info.members.slice(0, -1).map(m => m.size).join(","));
  ok("  ... and params.json is the last member",
     !info.err && info.members[info.members.length - 1].name === "params.json",
     info.err || info.members[info.members.length - 1].name);

  // ---- 6. the manifest against the page's own state -------------------------
  // Read out of the REAL archive by python, then compared to what the page says it is
  // running -- so a manifest built from the wrong globals, or serialized and never
  // written, fails here rather than looking plausible.
  const man = info.err ? null : JSON.parse(info.members[info.members.length - 1].text);
  const live = env.run(`function(){ const q = liveParams();
    return { app: appSlug(), t: simT, step: solver.nsteps,
             nx: q.nx, ny: q.ny, nz: q.nz || 1, Lx: q.Lx, Ly: q.Ly, Lz: q.Lz || 0,
             diss: q.diss, hyper: q.hyper, auto: autoDissOn(), zdiss: q.zdiss,
             force: el("cbForce").checked, epsP: q.epsP, epsM: q.epsM,
             lock: el("cbEpsLock").checked, shell: q.fshell, tau: q.tau,
             ic: el("selIC").value, demo: (el("selPreset") || {}).value || "",
             seed: q.seed, cfl: q.cfl }; }`);
  ok("the manifest names the app, the time and the step the archive was taken at",
     !!man && man.app === slug && man.t === live.t && man.step === live.step,
     man && JSON.stringify({ app: man.app, t: man.t, step: man.step }));
  ok("  ... the resolution and the box, with nz/Lz carried as 1/0 where there is no z",
     !!man && man.grid.nx === live.nx && man.grid.ny === live.ny && man.grid.nz === live.nz &&
     man.box.Lx === live.Lx && man.box.Ly === live.Ly && man.box.Lz === live.Lz &&
     (env.is3d ? man.grid.nz > 1 : man.grid.nz === 1 && man.box.Lz === 0),
     man && JSON.stringify([man.grid, man.box]));
  ok("  ... the dissipation and its hyper exponent (plus z-dissipation in 3D)",
     !!man && man.dissipation.diss === live.diss && man.dissipation.hyper === live.hyper &&
     man.dissipation.auto === live.auto &&
     (typeof live.zdiss === "number" ? man.dissipation.zdiss === live.zdiss
                                     : man.dissipation.zdiss === undefined),
     man && JSON.stringify(man.dissipation));
  ok("  ... the forcing state, and the IC preset the run started from",
     !!man && man.forcing.on === live.force && man.forcing.epsPlus === live.epsP &&
     man.forcing.epsMinus === live.epsM && man.forcing.locked === live.lock &&
     man.forcing.shell.join(",") === live.shell.join(",") && man.forcing.tau === live.tau &&
     man.ic.preset === live.ic && man.ic.demo === live.demo,
     man && JSON.stringify([man.forcing, man.ic]));

  // it must FOLLOW the page, not a snapshot taken at boot: turn forcing off, move the
  // dissipation, and save again
  env.run(`function(){ el("cbForce").checked = false; el("cbForce").onchange();
    el("rDiss").value = String(parseFloat(el("rDiss").value) + 1); el("rDiss").oninput(); }`);
  await env.run("function(){ return saveAllZip(); }");
  await settle();
  const man2 = (() => { const i = zipInfo(lastZip(env).bytes, slug + "b");
    return i.err ? null : JSON.parse(i.members[i.members.length - 1].text); })();
  const live2 = env.run("function(){ return { diss: liveParams().diss }; }");
  ok("a manifest is built at the press: forcing off and a moved diss slider both show",
     !!man2 && man2.forcing.on === false && man2.forcing.epsPlus === undefined &&
     man2.dissipation.diss === live2.diss && man2.dissipation.diss !== (man && man.dissipation.diss),
     man2 && JSON.stringify([man2.forcing, man2.dissipation.diss]));
  ok("  ... and the second archive replaced the first on the SAME slot, still one row",
     stripOf(env).rows === 1 && stripOf(env).on, JSON.stringify(stripOf(env)));

  // ---- 7. the strip is the only way out -------------------------------------
  const nDl1 = env.caps.downloads.length;
  const pressed = env.run(`function(lab){ const s = saveAll.resEl.zip;
    const b = s && s.children.filter(x => x.kind === "button" && x.innerHTML === lab)[0];
    if (!b) return false; b.onclick(); return true; }`, "download");
  const dl = env.caps.downloads[env.caps.downloads.length - 1];
  ok("the strip's download button writes taranis-<app>-all-t<simT>.zip, once",
     pressed && env.caps.downloads.length === nDl1 + 1 &&
     dl.name === "taranis-" + slug + "-all-t" +
       env.run("function(){ return simT.toFixed(3); }") + ".zip",
     dl && dl.name);
  ok("  ... and what comes out is the archive that was built",
     !!dl && dl.blob === lastZip(env));
  const shf = env.caps.files[env.caps.files.length - 1];
  ok("  ... the file offered to the share sheet is that same archive, as application/zip",
     !!shf && shf.type === "application/zip" && shf.size === lastZip(env).size &&
     shf.name === dl.name, shf && shf.name);
  // dismissing it clears the slot and nothing else
  env.run("function(){ cardResClear(saveAll, 'zip'); }");
  ok("dismissing the archive empties the page strip", !stripOf(env).on);

  // ---- 8. a card closed between the press and the picture -------------------
  // The strip is the page's, so a closed card cannot take the archive with it -- but its
  // own capture must not take the archive down either.
  const nZ = zips(env).length;
  const p = env.run(`function(){ const p = saveAllZip();
    cardClose(cards.chart[cards.chart.length - 1]); return p; }`);
  await p; await settle();
  ok("closing a card mid-save still produces one archive, on the page's own strip",
     zips(env).length === nZ + 1 && stripOf(env).on && stripOf(env).rows === 1,
     (zips(env).length - nZ) + " archives");

  // ---- 9. a capture that throws must not wedge the button ------------------
  // The re-entry guard is a flag, so a capture that throws with the flag still set would
  // kill save-all for the life of the page -- and the message has to be one the status
  // line can actually show (only `info` and `err` have a rule in style.css).
  const nZ2 = zips(env).length;
  const boom = await env.run(`function(){
    const c = cards.chart[0], cap = c.captureShot.bind(c);
    c.captureShot = function () { throw new Error("no 2d context"); };
    return Promise.resolve(saveAllZip()).then(function () {
      c.captureShot = cap;
      const st = el("status");
      return { busy: saveAll.busy, cls: st.className, msg: st.textContent }; }); }`);
  ok("a capture that throws reports it and releases the button",
     boom.busy === false && boom.cls === "err" && /could not build the archive/.test(boom.msg) &&
     zips(env).length === nZ2, JSON.stringify(boom));
  await env.run("function(){ return saveAllZip(); }");
  await settle();
  ok("  ... and the next press works",
     zips(env).length === nZ2 + 1 && stripOf(env).on, (zips(env).length - nZ2) + " archives");

  ok("the page raised no stub failures", env.fails.length === 0, env.fails.join(" | "));
  return env;
}

// ===========================================================================
// D. field export (.npz) -- IO_PLAN item 4, not built yet
// ===========================================================================
// When it lands: build the .npz through the same `zipStore`, write it with `zipInfo` for
// the ZIP-level asserts, then a second python leg over the same file with `py()` and
// numpy.load for dtype, shape, axis order and values. Nothing above needs to change.

(async () => {
  const env = await boot("rmhd2d.html");
  writerLegs(env);
  await pageLegs("rmhd2d.html");
  await pageLegs("rmhd3d.html");
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch (e) { /* leave it */ }
  console.log("\n" + (bad ? "FAIL" : "PASS") + "  zip: " + pass + " checks passed, " +
              bad + " failed");
  process.exit(bad ? 1 : 0);
})();
