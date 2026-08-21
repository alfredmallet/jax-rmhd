// GATE: the stored-ZIP writer and what it is used for (plans-webgpu/IO_PLAN.md).
//
// Section A is the writer alone (`zipStore`), section B the save-all button (item 3) on
// both booted pages, and section D the field export (item 4) on both -- the same archive
// writer carrying `.npy` members, read back by `numpy.load` rather than by us. `py()`
// runs a python script over a written file; `zipInfo()` and `npzInfo()` are the two
// external readers every leg asserts through.
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
// Section D is the same discipline over the field export. Two things in it can only be
// tested from outside: the AXIS ORDER (a silently transposed field is the classic way to
// waste an afternoon, so the check feeds a field that is not symmetric in x and y through
// the real button and compares the whole array against one numpy builds itself), and the
// PINNED third and fourth Mode uniforms -- the export must not write a live display
// card's mode, which is asserted by tracing every Mode-uniform write across an export
// with two loaded cards open and finding none.
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

// Every ZIP field this writer emits is written TWICE -- once in the member's local header
// and once in the central directory -- and `zipfile` reads only the second. That is not a
// detail: it is why a wrong compressed size or version in a LOCAL header passes every
// zipfile leg while Info-ZIP refuses the archive. So the reader also unpacks each 30-byte
// local header by hand (the one place bytes are read directly, and only to compare the two
// copies against each other) and hands both back.
const ZIP_PY = `
import sys, json, struct, zipfile, binascii
raw = open(sys.argv[1], "rb").read()
z = zipfile.ZipFile(sys.argv[1])
out = {"bad": z.testzip(), "members": []}
for i in z.infolist():
    d = z.read(i.filename)
    o = i.header_offset
    sig, vex, fl, meth, tt, dd, crc, csz, usz, nl, el = struct.unpack("<IHHHHHIIIHH", raw[o:o + 30])
    wd = ((max(0, i.date_time[0] - 1980)) << 9) | (i.date_time[1] << 5) | i.date_time[2]
    wt = (i.date_time[3] << 11) | (i.date_time[4] << 5) | (i.date_time[5] // 2)
    m = {"name": i.filename, "method": i.compress_type, "crc": i.CRC,
         "size": i.file_size, "csize": i.compress_size, "offset": i.header_offset,
         "flag": i.flag_bits, "vex": i.extract_version, "vcr": i.create_version,
         "date": list(i.date_time), "crcok": binascii.crc32(d) == i.CRC,
         "loc": {"sig": sig, "vex": vex, "flag": fl, "method": meth, "crc": crc,
                 "csize": csz, "size": usz, "extralen": el,
                 "name": raw[o + 30:o + 30 + nl].decode("utf-8"),
                 "dateok": dd == wd and tt == wt}}
    if i.filename.endswith((".json", ".txt")) and i.file_size <= 65536:
        m["text"] = d.decode("utf-8")
    if i.file_size <= 64:
        m["hex"] = binascii.hexlify(d).decode()
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
// STORED means the compressed size IS the uncompressed one -- in the central directory
// AND in the local header, which is where zipfile never looks: it extracts a stored member
// by `file_size` and ignores `compress_size` entirely. So the predicate reads both copies,
// and Info-ZIP below is the third opinion that refuses the archive outright.
const storedOk = m => m.method === 0 && m.csize === m.size &&
                      m.loc.method === 0 && m.loc.csize === m.loc.size && m.loc.size === m.size;

// ---- the SECOND external reader --------------------------------------------
// The plan's argument for an external verifier is that our own reader would share our own
// bugs -- and one reader has blind spots for the same reason. `compress_size` on a stored
// member is exactly such a spot: zipfile ignores it, Info-ZIP rejects the whole archive
// ("ucsize 11 <> csize 0 for STORED entry", then a bad CRC). So every archive that
// matters here goes through `unzip -t` as well, which extracts every member to nothing,
// recomputes its CRC and cross-checks the local header against the central directory.
// Where the binary is absent the legs SAY they were skipped rather than passing quietly.
const UNZIP = (() => {
  try { execFileSync("unzip", ["-v"], { stdio: "ignore" }); return true; } catch (e) { return false; }
})();
let skipped = 0;
function okUnzip(name, bytes, tag) {
  if (!UNZIP) { skipped++; console.log("  SKIP  " + name + "   [no unzip binary on this box]"); return; }
  const f = path.join(TMP, (tag || "u") + "-t.zip");
  fs.writeFileSync(f, Buffer.from(bytes));
  let out, good;
  try { out = execFileSync("unzip", ["-t", f], { encoding: "utf8" }); good = true; }
  catch (e) { out = String((e.stdout || "") + (e.stderr || "")); good = false; }
  ok(name, good && /No errors detected/.test(out),
     out.split("\n").map(s => s.trim()).filter(Boolean).pop());
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
     storedOk(i1.members[0]) && i1.members[0].crcok && i1.members[0].text === "hello",
     JSON.stringify((i1.members || [])[0] || i1.err));
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
     storedOk(i2.members[1]) && i2.members[1].text === "",
     i2.err || JSON.stringify(i2.members[1]));
  // STORED's own contract, and the field zipfile alone would never notice: with method 0
  // the compressed size IS the uncompressed one, in the local header and in the central
  // directory both. Info-ZIP refuses an archive where they disagree; the leg after this
  // one is that reader saying so.
  ok("  ... and every member declares compressed == uncompressed, which is what STORED MEANS",
     !i2.err && i2.members.every(storedOk),
     i2.err || i2.members.map(m => m.csize + "/" + m.size).join(" "));
  okUnzip("  ... and Info-ZIP -- a second reader, with different blind spots -- tests it clean",
          build(spec), "mixed");
  // the two header fields nothing else here reads. 2.0 is the version that introduced the
  // fields this writer uses; a 0 there is legal-ish and meaningless, and the DOS stamp is
  // what every unzipper writes on the extracted file, so a day 0 becomes a file dated
  // "the zeroth" or nothing at all.
  ok("  ... version-needed and version-made-by are both 2.0 in every header",
     !i2.err && i2.members.every(m => m.vex === 20 && m.vcr === 20 && m.loc.vex === 20),
     i2.err || i2.members.map(m => m.vex + "/" + m.vcr + "/" + m.loc.vex).join(" "));
  // the other half of every field: zipfile reads the central directory and nothing else,
  // so a wrong LOCAL header is invisible to it (that is exactly how a stored member can
  // declare 0 compressed bytes and still pass). Both copies must say the same thing.
  ok("  ... and each member's LOCAL header repeats the central directory exactly",
     !i2.err && i2.members.every(m => m.loc.sig === 0x04034b50 && m.loc.vex === m.vex &&
       m.loc.flag === m.flag && m.loc.method === m.method && m.loc.crc === m.crc &&
       m.loc.size === m.size && m.loc.csize === m.csize && m.loc.name === m.name &&
       m.loc.extralen === 0 && m.loc.dateok),
     i2.err || i2.members.map(m => JSON.stringify(m.loc)).join(" "));
  // the clock the PAGE reads (stubenv stubs Date, so this is the one _dosStamp saw), in
  // the same six fields the DOS pair decodes to
  const now = env.run(`function(){ const d = new Date();
    return [d.getFullYear(), d.getMonth() + 1, d.getDate(),
            d.getHours(), d.getMinutes(), d.getSeconds()]; }`);
  const asMs = a => Date.UTC(a[0], a[1] - 1, a[2], a[3], a[4], a[5]);
  const stamp = i2.err ? null : i2.members[0].date;
  ok("  ... and the DOS date/time stamp is the local clock at the press, to the 2 s DOS tick",
     !!stamp && stamp[1] >= 1 && stamp[1] <= 12 && stamp[2] >= 1 && stamp[2] <= 31 &&
     stamp[3] < 24 && stamp[4] < 60 && stamp[5] < 60 &&
     Math.abs(asMs(now) - asMs(stamp)) <= 2000,
     stamp && stamp.join("-") + "  (page clock " + now.join("-") + ")");
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
  okUnzip("  ... and Info-ZIP tests the 70 kB archive clean too", buildBig(70000), "big");

  // 5. a member whose data is NOT a byte array. `length` on a Float32Array is its ELEMENT
  //    count and `Uint8Array.set` value-CONVERTS, so the naive writer emits 3 bytes of
  //    rounded values, declares size 3 and CRCs the same wrong bytes -- a perfectly valid
  //    archive holding the wrong data, which no reader anywhere can flag. Every caller
  //    today passes a Uint8Array; this is the shared primitive, so the next one might not.
  const flo = env.run(`function(){
    const a = new Float32Array([1.5, 2.5, 3.5]);
    const b = zipStore([{ name: "f.bin", data: a }]);
    return { bytes: Array.from(b.bytes),
             hex: Array.from(new Uint8Array(a.buffer)).map(v => (v + 256).toString(16).slice(1)).join("") }; }`);
  const iF = zipInfo(flo.bytes, "float");
  ok("a Float32Array member is written as its BYTES: 12 of them, not 3 rounded values",
     !iF.err && iF.bad === null && iF.members.length === 1 && storedOk(iF.members[0]) &&
     iF.members[0].size === 12 && iF.members[0].crcok && iF.members[0].hex === flo.hex,
     iF.err || (iF.members[0].size + " B, " + iF.members[0].hex + " want " + flo.hex));
  okUnzip("  ... and that archive tests clean", flo.bytes, "float");
  // ... while something with no unambiguous byte meaning is refused rather than guessed at
  const guess = env.run(`function(){ const out = [];
    for (const d of ["hello", [1, 2, 3], 7]) {
      try { zipStore([{ name: "x", data: d }]); out.push("no throw"); }
      catch (e) { out.push(e.message); }
    }
    return out; }`);
  ok("  ... and a string, a plain array or a number is refused, never encoded on a guess",
     guess.every(m => /typed array or an ArrayBuffer/.test(m)), guess.join(" | "));
  // an ArrayBuffer IS bytes, so it is taken
  const abuf = env.run(`function(){ const a = new Uint8Array([9, 8, 7]);
    return Array.from(zipStore([{ name: "b.bin", data: a.buffer }]).bytes); }`);
  const iAB = zipInfo(abuf, "abuf");
  ok("  ... and a bare ArrayBuffer is taken as the bytes it is",
     !iAB.err && iAB.members.length === 1 && iAB.members[0].size === 3 &&
     iAB.members[0].hex === "090807", iAB.err || JSON.stringify(iAB.members[0]));

  // 6. a name past the 16-bit name-length field. 70004 bytes writes 4468, and both readers
  //    then refuse the whole archive -- so the writer must refuse it first, and it must do
  //    so on the UTF-8 BYTE length: 40000 two-byte characters is a 40000-character string
  //    and an 80000-byte name.
  const nm = env.run(`function(){
    const t = n => { try { const b = zipStore([{ name: n, data: new Uint8Array(4) }]);
                           return { threw: false, size: b.bytes.length }; }
                     catch (e) { return { threw: true, msg: e.message }; } };
    return { long: t("a".repeat(70004)), utf8: t("\\u00e9".repeat(40000)),
             edge: t("\\u00e9".repeat(30000)) }; }`);
  ok("a member name past 64 kB is refused by the writer, with the length in the message",
     nm.long.threw && /70004/.test(nm.long.msg), JSON.stringify(nm.long));
  ok("  ... on its UTF-8 byte length: 40000 two-byte characters is an 80000-byte name",
     nm.utf8.threw && /80000/.test(nm.utf8.msg), JSON.stringify(nm.utf8));
  ok("  ... and a 60000-byte name is still written, because it FITS the field",
     !nm.edge.threw && nm.edge.size > 60000, JSON.stringify(nm.edge));
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
     !info.err && info.bad === null && info.members.every(m => storedOk(m) && m.crcok),
     info.err || info.members.length + " members");
  okUnzip("  ... and Info-ZIP tests the REAL archive clean, member by member",
          zip.bytes, slug + "-unzip");
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
  ok("  ... and a preset whose content IS its name carries nothing else",
     !!man && man.ic.phi === undefined && man.ic.psi === undefined,
     man && JSON.stringify(man.ic));

  // ---- 6b. an expression IC is its two FORMULAS ----------------------------
  // The one preset whose entire content is typed rather than named (IO_PLAN item 1).
  // `"ic": {"preset": "expr"}` alone names a run nobody could reproduce -- which is the
  // manifest's stated purpose failing on the one IC that needs it most. Driven through
  // the real controls and read back out of the real archive.
  const EPHI = "sin(2*pi*x/Lx)*cos(2*pi*y/Ly)", EPSI = "0.25*cos(2*pi*x/Lx)";
  env.run(`function(a){ const s = el("selIC"); s.value = "expr"; if (s.onchange) s.onchange();
    for (const r of [["tExprP", a[0]], ["tExprM", a[1]]]) {
      const e = el(r[0]); e.value = r[1]; if (e.oninput) e.oninput();
    } }`, [EPHI, EPSI]);
  await env.run("function(){ return saveAllZip(); }");
  await settle();
  const manE = (() => { const i = zipInfo(lastZip(env).bytes, slug + "-expr");
    return i.err ? null : JSON.parse(i.members[i.members.length - 1].text); })();
  ok("an expression IC records BOTH formulas in the manifest, not just the word 'expr'",
     !!manE && manE.ic.preset === "expr" && manE.ic.phi === EPHI && manE.ic.psi === EPSI,
     manE && JSON.stringify(manE.ic));
  // an empty box means exactly zero, which is a statement about the run and is recorded
  env.run(`function(){ const e = el("tExprM"); e.value = ""; if (e.oninput) e.oninput(); }`);
  await env.run("function(){ return saveAllZip(); }");
  await settle();
  const manE2 = (() => { const i = zipInfo(lastZip(env).bytes, slug + "-expr2");
    return i.err ? null : JSON.parse(i.members[i.members.length - 1].text); })();
  ok("  ... an empty box is recorded as the empty formula it is (exactly zero)",
     !!manE2 && manE2.ic.phi === EPHI && manE2.ic.psi === "",
     manE2 && JSON.stringify(manE2.ic));
  env.run(`function(){ const s = el("selIC"); s.value = "modes"; if (s.onchange) s.onchange(); }`);

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

  // ---- 10. the re-entry guard -----------------------------------------------
  // `saveAll.busy` is what stops a double press building the archive twice off the same
  // cards -- two files and two strip slots for one press-worth of intent, on a phone where
  // a double tap is the norm. It is only visible from two calls in ONE task (the second
  // press lands before the first has awaited its captures), so that is what this does:
  // zipStore is counted, both promises awaited, and exactly one archive must come out.
  const nZ3 = zips(env).length;
  const twice = await env.run(`function(){ const zs = zipStore; let n = 0;
    globalThis.zipStore = function (m) { n++; return zs(m); };
    const a = saveAllZip(), b = saveAllZip();          // same task, no await between them
    return Promise.all([a, b]).then(function () {
      globalThis.zipStore = zs;
      return { n: n, busy: saveAll.busy }; }); }`);
  await settle();
  ok("two presses in one task build ONE archive: the re-entry guard holds",
     twice.n === 1 && zips(env).length === nZ3 + 1 && twice.busy === false,
     twice.n + " zipStore calls, " + (zips(env).length - nZ3) + " archives");
  ok("  ... and one strip row, not two files fighting over the slot",
     stripOf(env).rows === 1 && stripOf(env).on, JSON.stringify(stripOf(env)));

  ok("the page raised no stub failures", env.fails.length === 0, env.fails.join(" | "));
  return env;
}

// ===========================================================================
// D. field export -- real-space phi and psi as .npz (item 4)
// ===========================================================================
// The second external reader. `.npz` is a ZIP of `.npy`, so every archive here goes
// through zipInfo first (it is a ZIP or it is nothing) and then through numpy, which is
// the reader the file exists for. numpy reports dtype, shape and C-contiguity; the raw
// member bytes carry the two things numpy hides -- the .npy header text and where the
// data starts -- and both are asserted.
//
// The fields are injected as the FLAT BUFFER INDEX, so the array numpy should see is
// exactly `arange(n).reshape(shape)` in C order and the check compares every element
// against one it builds itself. That is the axis-order leg: a transposed field, a
// Fortran-order flag or a swapped shape tuple each fail on almost every element rather
// than at one lucky index, and the readable probe printed beside it (phi[2, 1] in 2D,
// phi[1, 2, 3] in 3D) is deliberately at an index that is NOT symmetric in x and y.
const NPZ_PY = `
import sys, json, zipfile
import numpy as np
f, spec = sys.argv[1], json.loads(sys.argv[2])
zf = zipfile.ZipFile(f)
z = np.load(f)
out = {"files": list(z.files), "arrays": {}, "npy": {}}
names = [k for k in ("phi", "psi", "x", "y", "z") if k + ".npy" in zf.namelist()]
for k in names:
    a = z[k]
    out["arrays"][k] = {"dtype": a.dtype.str, "shape": list(a.shape),
                        "c": bool(a.flags["C_CONTIGUOUS"])}
    raw = zf.read(k + ".npy")
    hl = raw[8] + 256 * raw[9]
    out["npy"][k] = {"magic": raw[:6].decode("latin1"), "ver": [raw[6], raw[7]],
                     "hlen": hl, "data": 10 + hl,
                     "head": raw[10:10 + hl].decode("latin1")}
# the two fields, element by element, against C-order arange
for i, k in enumerate(("phi", "psi")):
    a, off = z[k], spec["off"][i]
    want = (np.arange(a.size, dtype=np.float64) + off).astype(np.float32).reshape(a.shape)
    p = tuple(spec["probe"])
    out["arrays"][k]["wrong"] = int(np.count_nonzero(a != want))
    out["arrays"][k]["probe"] = [list(p), float(a[p]), float(want[p])]
# ... and the coordinate vectors, exactly i*L/n at float32
for k, n, L in zip(("x", "y", "z"), spec["n"], spec["L"]):
    if k in out["arrays"]:
        want = (np.arange(n, dtype=np.float64) * L / n).astype(np.float32)
        out["arrays"][k]["wrong"] = int(np.count_nonzero(z[k] != want))
out["manifest"] = zf.read("params.json").decode("utf-8")
print(json.dumps(out))
`;
function npzInfo(bytes, spec, tag) {
  const f = path.join(TMP, (tag || "f") + ".npz");
  fs.writeFileSync(f, Buffer.from(bytes));
  try { return JSON.parse(py(NPZ_PY, [f, JSON.stringify(spec)])); }
  catch (e) { return { err: String(e.stderr || e.message).trim().split("\n").pop() }; }
}

// Known bytes through the REAL path. readBufOnce's staging buffers are the only MAP_READ
// buffers the page creates from here on, so filling them by creation order (phi first,
// psi second) puts a field the check knows exactly where the export reads one -- and it
// does so WITHOUT the check touching the export: the uniforms, the pass, the packing and
// the archive are all the page's own.
const INJECT = `function(){ const d = solver.device, orig = d.createBuffer.bind(d);
  globalThis.NSTAGE = 0;
  d.createBuffer = function (o) {
    const b = orig(o);
    if (o.usage & GPUBufferUsage.MAP_READ) {
      const k = globalThis.NSTAGE++;
      b.getMappedRange = function () {
        const n = o.size / 4, a = new Float32Array(n);
        for (let i = 0; i < n; i++) a[i] = i + 1e6 * k;
        return a.buffer;
      };
    }
    return b;
  }; }`;

// every Mode-uniform write of every EXISTING display chain, by chain and buffer name
// (checkoff.js's tracer). `UW` is cleared by hand, so a leg can bracket one action.
const MTRACE = `function(){ const d = solver.device, orig = d.queue.writeBuffer.bind(d.queue);
  globalThis.UW = [];
  d.queue.writeBuffer = function (b, off, data) {
    for (let ci = 0; ci < solver.disp.length; ci++) {
      const D = solver.disp[ci];
      if (!D) continue;
      for (const k of Object.keys(D.buf)) {
        const v = D.buf[k], at = Array.isArray(v) ? v.indexOf(b) : (v === b ? 0 : -1);
        if (at < 0) continue;
        globalThis.UW.push([ci + ":" + k + (Array.isArray(v) ? "[" + at + "]" : ""),
          Array.from(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))]);
      }
    }
    return orig(b, off, data); }; }`;
// ... and the pinned pair, caught where it is written: a chain builds its uniforms before
// the solver has a name for it, so this traces the writes of ONE chain() call and resolves
// the buffers to names afterwards, out of the chain it just built.
const PINNED = `function(ci){
  const d = solver.device, orig = d.queue.writeBuffer.bind(d.queue), W = [];
  d.queue.writeBuffer = function (b, off, data) {
    W.push([b, Array.from(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))]);
    return orig(b, off, data); };
  let D;
  try { D = solver.chain(ci); } finally { d.queue.writeBuffer = orig; }
  const w = {};
  for (const e of W) for (const k of Object.keys(D.buf)) if (D.buf[k] === e[0] && !w[k]) w[k] = e[1];
  return { w: w, has: Object.keys(D.buf).filter(k => k.indexOf("mode") === 0).sort(),
           bgs: Object.keys(D.bg).filter(k => k.indexOf("prepDisp") === 0).sort() }; }`;
// one Mode uniform's 32 bytes, read the way modeWords wrote them
const modeOf = a => {
  const b = Uint8Array.from(a || []).buffer;
  if (b.byteLength !== 32) return { bytes: b.byteLength };
  const u = new Uint32Array(b), f = new Float32Array(b);
  return { bytes: 32, mode: u[0], zslice: u[1], cmap: u[2],
           klo: f[4], khi: f[5], sx: f[6], sy: f[7] };
};
// the page-level strip, in its OWN slot
const fstrip = env => env.run(`function(){ const s = saveAll.resEl.npz;
  const rows = saveAll.foot.children.filter(x => (x.className || "").indexOf("recres") >= 0);
  return { on: !!s, rows: rows.length, zip: !!saveAll.resEl.zip,
           foot: !!s && s.parentNode === saveAll.foot,
           txt: s ? s.children.filter(x => x.kind === "span").map(x => x.innerHTML).join("") : "",
           btns: s ? s.children.filter(x => x.kind === "button").map(x => x.innerHTML) : [] }; }`);

async function fieldLegs(name) {
  console.log("\n=== D. field export (" + name + ") ===");
  const env = await boot(name);          // a page of its own: section B's archives are
  if (!env.run("function(){ return !!solver; }")) {   // application/zip too
    ok(name + ": solver came up", false); return env;
  }
  const slug = name.replace(".html", "");
  const three = env.is3d;

  // ---- 1. the button ------------------------------------------------------
  const where = env.run(`function(){ const b = el("btnSaveFields"), row = b && b.parentNode;
    return { on: !!b, t: (b && b.innerHTML) || "", ti: (b && b.title) || "",
             withSave: !!row && row.children.indexOf(el("btnSaveAll")) >= 0,
             after: !!row && row.children.indexOf(b) > row.children.indexOf(el("btnSaveAll")),
             inBar: !!row && row === el("topbar"),
             grp: !!row && !!row.parentNode && row.parentNode.id }; }`);
  ok("a save-fields button sits in the displays & charts group, after save all",
     where.on && where.withSave && where.after && !where.inBar && where.grp === "grpDisp",
     JSON.stringify(where));
  ok("  ... and its title says what the file is and how to read it",
     /\.npz/.test(where.ti) && /numpy\.load/.test(where.ti) && /phi/.test(where.ti),
     where.ti);

  // ---- 2. the pinned pair, as a fresh chain writes it ---------------------
  // Chain 0 is built at boot, so the pin is caught on a chain built ON PURPOSE here.
  const K = env.run("function(){ return [DISP_PHI, DISP_PSI, MODE_BYTES]; }");
  const pin = env.run(PINNED, 2);
  ok("every display chain carries a THIRD and FOURTH Mode uniform for the export",
     pin.has.join(",") === "mode,modeC,modeM,modeX,modeX2" &&
     pin.bgs.join(",") === "prepDisp,prepDispM,prepDispX,prepDispX2",
     pin.has.join(",") + " / " + pin.bgs.join(","));
  const px = modeOf(pin.w.modeX), pp = modeOf(pin.w.modeX2);
  ok("  ... pinned to PLAIN phi and PLAIN psi",
     px.mode === K[0] && pp.mode === K[1] && px.bytes === K[2] && pp.bytes === K[2],
     JSON.stringify([px.mode, pp.mode]) + " want " + JSON.stringify([K[0], K[1]]));
  ok("  ... with the k_perp band, the display offset and the colormap all OFF",
     [px, pp].every(m => m.klo === 0 && m.khi === 0 && m.sx === 0 && m.sy === 0 &&
                         m.cmap === 0 && m.zslice === 0),
     JSON.stringify([px, pp]));

  // ---- 3. the file, through the real button -------------------------------
  const g = env.run(`function(){ const q = solver.p;
    return { nx: q.nx, ny: q.ny, nz: q.nz || 1, Lx: q.Lx, Ly: q.Ly, Lz: q.Lz || 0,
             nr: solver.nr, t: simT, name: capName("fields", "npz") }; }`);
  env.run(INJECT);
  const nDl0 = env.caps.downloads.length, live0 = env.live.buffers;
  await env.run("function(){ return saveFieldsNpz(); }");
  await settle();
  const npz = lastZip(env);
  const want = ["phi.npy", "psi.npy", "x.npy", "y.npy"]
    .concat(three ? ["z.npy"] : []).concat(["params.json"]);
  ok("the press produces one .npz and downloads nothing",
     !!npz && env.caps.downloads.length === nDl0,
     (npz ? npz.size + " B" : "no file") + ", " +
       (env.caps.downloads.length - nDl0) + " downloads");
  if (!npz) return env;
  // ... which is first of all a valid stored ZIP, read by zipfile
  const zi = zipInfo(npz.bytes, slug + "-npz");
  ok("  ... a valid stored ZIP: testzip() clean, every member's CRC recomputed",
     !zi.err && zi.bad === null && zi.members.every(m => storedOk(m) && m.crcok),
     zi.err || zi.members.length + " members");
  okUnzip("  ... and Info-ZIP tests the .npz clean: numpy is not its only reader",
          npz.bytes, slug + "-npz-unzip");
  ok("  ... holding phi, psi, the coordinate vectors and params.json, in that order",
     !zi.err && zi.members.map(m => m.name).join(",") === want.join(","),
     zi.err || zi.members.map(m => m.name).join(","));

  // ---- 4. what numpy sees --------------------------------------------------
  const probe = three ? [1, 2, 3] : [2, 1];
  const info = npzInfo(npz.bytes, { off: [0, 1e6], probe: probe,
                                    n: [g.nx, g.ny, g.nz], L: [g.Lx, g.Ly, g.Lz] },
                       slug + "-fields");
  const A = info.arrays || {};
  ok("numpy.load opens it and finds the arrays by name",
     !info.err && (info.files || []).join(",") ===
       want.map(n => n.replace(/\.npy$/, "")).join(","),
     info.err || (info.files || []).join(","));
  const shape = three ? [g.nz, g.nx, g.ny] : [g.nx, g.ny];
  ok("  ... phi and psi are float32 '<f4', C-contiguous, shaped " +
       (three ? "(nz, nx, ny)" : "(nx, ny)"),
     !info.err && ["phi", "psi"].every(k => A[k] && A[k].dtype === "<f4" && A[k].c &&
                                            A[k].shape.join(",") === shape.join(",")),
     info.err || JSON.stringify(["phi", "psi"].map(k => A[k] && [A[k].dtype, A[k].shape])));
  // THE axis-order leg: the buffer index read literally, every element
  ok("  ... and every element is the buffer index read literally -- the axis order holds",
     !info.err && A.phi && A.phi.wrong === 0 && A.psi && A.psi.wrong === 0,
     info.err || "phi " + (A.phi && A.phi.wrong) + " wrong, psi " +
       (A.psi && A.psi.wrong) + " wrong of " + g.nr);
  ok("  ... at an index that is NOT symmetric in x and y: phi[" + probe.join(", ") + "]",
     !info.err && A.phi && A.phi.probe[1] === A.phi.probe[2],
     info.err || (A.phi && (A.phi.probe[1] + " want " + A.phi.probe[2])));

  // ---- 5. the coordinate vectors ------------------------------------------
  const axes = ["x", "y"].concat(three ? ["z"] : []);
  const lens = { x: g.nx, y: g.ny, z: g.nz };
  ok("  ... the coordinate vectors are 1-D float32, one per axis, the right length",
     !info.err && axes.every(k => A[k] && A[k].dtype === "<f4" &&
                                  A[k].shape.join(",") === String(lens[k])) &&
     (three || !A.z),
     info.err || JSON.stringify(axes.map(k => A[k] && A[k].shape)));
  ok("  ... and every entry is EXACTLY i*L/n, so pcolormesh needs no grid rebuilt",
     !info.err && axes.every(k => A[k] && A[k].wrong === 0),
     info.err || axes.map(k => k + ":" + (A[k] && A[k].wrong)).join(" "));

  // ---- 6. the .npy header --------------------------------------------------
  const H = info.npy || {};
  ok("every .npy member is v1.0 and starts its DATA on a 64-byte boundary",
     !info.err && want.slice(0, -1).every(n => {
       const h = H[n.replace(/\.npy$/, "")];
       return h && h.magic === "\x93NUMPY" && h.ver.join(",") === "1,0" && h.data % 64 === 0;
     }),
     info.err || want.slice(0, -1).map(n => { const h = H[n.replace(/\.npy$/, "")];
       return n + "@" + (h && h.data); }).join(" "));
  ok("  ... every header declares '<f4' and fortran_order: False",
     !info.err && Object.keys(H).every(k => /'descr': '<f4'/.test(H[k].head) &&
                                            /'fortran_order': False/.test(H[k].head)),
     info.err || (H.phi && H.phi.head.trim()));
  // the shape tuple's two spellings: numpy writes the trailing comma on a 1-tuple
  ok("  ... the shape tuple is spelt (" + shape.join(", ") + ") and, on a vector, (" +
       g.nx + ",)",
     !info.err && H.phi && H.phi.head.indexOf("'shape': (" + shape.join(", ") + "), }") >= 0 &&
     H.x && H.x.head.indexOf("'shape': (" + g.nx + ",), }") >= 0,
     info.err || [H.phi && H.phi.head.trim(), H.x && H.x.head.trim()].join("  |  "));

  // ---- 7. the manifest -----------------------------------------------------
  const man = info.err ? null : JSON.parse(info.manifest);
  ok("params.json is the shared run manifest, carrying the sim time it was taken at",
     !!man && man.app === slug && man.t === g.t && man.grid.nx === g.nx &&
     man.grid.nz === g.nz,
     man && JSON.stringify({ app: man.app, t: man.t, grid: man.grid }));
  ok("  ... plus the AXIS ORDER, so a reader never has to guess it",
     !!man && !!man.export && man.export.dtype === "<f4" && man.export.order === "C" &&
     man.export.shape.join(",") === shape.join(",") &&
     man.export.axes.indexOf(three ? "phi[iz, ix, iy]" : "phi[ix, iy]") === 0,
     man && man.export && man.export.axes);
  ok("  ... and the note that the state is DEALIASED, spectrum and all",
     !!man && !!man.export && /2\/3/.test(man.export.dealiased) &&
     /spectrum/.test(man.export.dealiased),
     man && man.export && man.export.dealiased);

  // ---- 8. the staging buffers are destroyed, never pooled ------------------
  // The plan's one memory rule. _stagePool has no eviction by design, so a field read
  // through it would strand 4 MB per field here (16 MB at the largest 3D grid) for the
  // life of the page. Two things say it did not: the pool has no entry that big, and the
  // device's live buffer count is exactly back where it started.
  const pool = env.run(`function(){ const o = {};
    _stagePool.forEach((v, k) => { o[k] = v.length; }); return o; }`);
  const big = Object.keys(pool).filter(k => Number(k) >= 1e6);
  ok("a " + (g.nr * 4 / 1e6).toFixed(1) + " MB field read leaves NO multi-MB entry in _stagePool",
     big.length === 0, "pool sizes: " + (Object.keys(pool).join(",") || "(empty)"));
  ok("  ... because its staging buffer was destroyed: the live buffer count is unmoved",
     env.live.buffers === live0, env.live.buffers - live0 + " buffers left over");

  // ---- 9. the strip, and its own slot --------------------------------------
  const st = fstrip(env);
  ok("the export waits on the page's result strip, with download / share / dismiss",
     st.on && st.foot && st.btns.join(",") === "download,share,&times;", JSON.stringify(st));
  ok("  ... quoting its size", st.txt === env.run("function(n){ return recSizeText(n); }", npz.size),
     st.txt);
  // an archive of pictures and a field export are different files: one slot each
  await env.run("function(){ return saveAllZip(); }");
  await settle();
  const st2 = fstrip(env);
  ok("  ... and save-all does not take it away: two files, two slots, two rows",
     st2.on && st2.zip && st2.rows === 2, JSON.stringify(st2));
  const nDl1 = env.caps.downloads.length;
  env.run(`function(){ const s = saveAll.resEl.npz;
    s.children.filter(x => x.kind === "button" && x.innerHTML === "download")[0].onclick(); }`);
  const dl = env.caps.downloads[env.caps.downloads.length - 1];
  ok("  ... the download button writes taranis-<app>-fields-t<simT>.npz, once",
     env.caps.downloads.length === nDl1 + 1 && dl.name === g.name && dl.blob === npz,
     (dl && dl.name) + " want " + g.name);

  // ---- 10. an export does NOT disturb a live display card ------------------
  // The failure the plan warns about: B.mode and B.modeM are a card's LIVE uniforms, so
  // an export that borrowed one would corrupt that card until its next apply(). Two cards
  // are loaded up with non-default modes (and, where the page offers them, a band and an
  // offset), every Mode-uniform write is traced, and the export must produce NONE.
  const set = env.run(`function(){
    while (cards.disp.length < 2) addDisplayCard();
    cardsSync();
    const f = el("cbFilter"); if (f) { f.checked = true; if (f.onchange) f.onchange(); }
    const A = cards.disp[0], B = cards.disp[1];
    A.selField.value = String(DISP_PSI); B.selField.value = String(DISP_ZMINUS);
    if (A.selCont) A.selCont.value = String(DISP_PHI);
    if (A.rBLo) A.rBLo.value = "2";
    if (B.rBLo) B.rBLo.value = "3";
    if (A.rOffX) { A.rOffX.value = "0.25"; A.rOffY.value = "-0.125"; }
    A.apply(); B.apply();
    return { modes: [solver.chain(A.ci).mode, solver.chain(B.ci).mode],
             band: [A.band(), B.band()],
             off: A.offset(), hasOff: !!A.rOffX,
             ci: [A.ci, B.ci], cont: solver.chain(A.ci).cont.slice() }; }`);
  ok("two cards loaded up: non-default modes, a k_perp band" +
       (set.hasOff ? " and a display offset" : " (this page has no offset)"),
     set.modes[0] !== 0 && set.modes[1] !== 0 && set.modes[0] !== set.modes[1] &&
     set.band[0][0] > 0 && set.band[1][0] > 0 && set.cont[0] > 0 &&
     (!set.hasOff || (set.off[0] !== 0 && set.off[1] !== 0)),
     JSON.stringify(set));
  env.run(MTRACE);
  const before = env.run(`function(){ return cards.disp.map(c => {
    const D = solver.chain(c.ci);
    return { mode: D.mode, cont: D.cont.slice(), band: c.band(), off: c.offset() }; }); }`);
  env.run("function(){ globalThis.UW = []; }");
  await env.run("function(){ return saveFieldsNpz(); }");
  await settle();
  const wrote = env.run("function(){ return globalThis.UW.map(u => u[0]); }");
  const after = env.run(`function(){ return cards.disp.map(c => {
    const D = solver.chain(c.ci);
    return { mode: D.mode, cont: D.cont.slice(), band: c.band(), off: c.offset() }; }); }`);
  ok("an export writes NO Mode uniform of any live chain -- not mode, not modeM, not modeC",
     wrote.length === 0, wrote.join(",") || "none");
  ok("  ... so both cards' modes, contours, bands and offsets are byte for byte unmoved",
     JSON.stringify(before) === JSON.stringify(after),
     JSON.stringify(before) + " -> " + JSON.stringify(after));
  // ... and the export still worked with two loaded cards open
  const again = (() => { const z = lastZip(env);
    return z ? zipInfo(z.bytes, slug + "-again") : { err: "no file" }; })();
  ok("  ... and the export ran anyway, with two loaded cards open: same member list",
     !again.err && again.members.map(m => m.name).join(",") === want.join(","),
     again.err || again.members.map(m => m.name).join(","));

  // ---- 11. a failed read must not wedge the button -------------------------
  const boom = await env.run(`function(){
    const orig = readFieldPair;
    globalThis.readFieldPair = function () { return Promise.reject(new Error("device lost")); };
    return Promise.resolve(saveFieldsNpz()).then(function () {
      globalThis.readFieldPair = orig;
      const s = el("status");
      return { busy: saveAll.fbusy, cls: s.className, msg: s.textContent }; }); }`);
  ok("a failed read reports it and releases the button",
     boom.busy === false && boom.cls === "err" && /could not export the fields/.test(boom.msg),
     JSON.stringify(boom));
  const nz = zips(env).length;
  await env.run("function(){ return saveFieldsNpz(); }");
  await settle();
  ok("  ... and the next press works", zips(env).length === nz + 1,
     (zips(env).length - nz) + " archives");

  ok("the page raised no stub failures", env.fails.length === 0, env.fails.join(" | "));
  return env;
}

(async () => {
  const env = await boot("rmhd2d.html");
  writerLegs(env);
  await pageLegs("rmhd2d.html");
  await pageLegs("rmhd3d.html");
  await fieldLegs("rmhd2d.html");
  await fieldLegs("rmhd3d.html");
  try { fs.rmSync(TMP, { recursive: true, force: true }); } catch (e) { /* leave it */ }
  console.log("\n" + (bad ? "FAIL" : "PASS") + "  zip: " + pass + " checks passed, " +
              bad + " failed" + (skipped ? ", " + skipped + " skipped (no unzip binary)" : ""));
  process.exit(bad ? 1 : 0);
})();
