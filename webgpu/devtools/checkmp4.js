// GATE: the progressive-MP4 muxer (common.js mp4Mux), driven with REAL H.264.
//
// The muxer exists because Chrome's MediaRecorder "video/mp4" writes a FRAGMENTED file
// whose delta samples are flagged "not a sync sample, dependency unknown", which iOS
// AVFoundation takes at its word: it drops every one of them and plays a 30 s recording
// as three stills. So the thing under test is not "does a file come out" but "is the
// file the plain progressive shape a phone will open" -- ftyp + mdat + moov, real
// sample tables, no moof / mvex / trun / ctts anywhere.
//
// No GPU and no browser here: ffmpeg encodes a synthetic test pattern to a raw Annex-B
// .h264 stream, this script cuts it into samples and builds the avcC (the ONLY Annex-B
// handling in the project, and it lives in the test, not in the app), mp4Mux writes the
// file, and ffprobe/ffmpeg say what it really is.
//
//   Usage: node checkmp4.js [webgpu-dir]     (needs ffmpeg + ffprobe on PATH)
"use strict";
const fs = require("fs"), vm = require("vm"), path = require("path");
const { execFileSync } = require("child_process");
const dir = process.argv[2] || path.join(__dirname, "..");
// scratch for the ffmpeg streams and the muxed files. TMPDIR wins: the names below are
// fixed, so a shared /var/tmp left holding another user's run of this script fails every
// leg with EACCES before a single assertion is made.
const TMP = process.env.TMPDIR || "/var/tmp";
const FPS = 30;
let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};

// ---- pull mp4Mux out of common.js -----------------------------------------
// Same idiom as checks.js: run the shared core on a throwaway DOM and read the top-level
// consts back through an expression (a vm script's `const`s are not context properties).
const stubEl = () => ({ value: "", style: {}, textContent: "", innerHTML: "", checked: false,
                        disabled: false, min: "", max: "", step: "", options: [],
                        addEventListener() {}, appendChild() {} });
const sandbox = {
  document: { getElementById: stubEl, createElement: stubEl, createTextNode: () => ({}),
              querySelectorAll: () => [] },
  window: { addEventListener() {}, devicePixelRatio: 1, matchMedia: () => ({ matches: true }) },
  console, Math, JSON, Float32Array, Float64Array, Uint32Array, Uint8Array, Uint8ClampedArray,
  Map, Set, Error, Promise, setTimeout, Number, String, Array, Object, isFinite, parseInt,
  parseFloat, URLSearchParams, performance: { now: () => 0 }
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(dir, "common.js"), "utf8"), sandbox, { filename: "common.js" });
const { mp4Mux, recCodec, REC_FPS } = vm.runInContext("({ mp4Mux, recCodec, REC_FPS })", sandbox);

// ---- ffmpeg ground truth ---------------------------------------------------
const sh = (cmd, args) => execFileSync(cmd, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
// a real Baseline stream at exactly FPS with a keyframe every FPS frames and no B-frames
// -- the same thing the app asks its VideoEncoder for
function encodeH264(w, h, secs, out) {
  execFileSync("ffmpeg", ["-y", "-v", "error", "-f", "lavfi",
    "-i", "testsrc=size=" + w + "x" + h + ":rate=" + FPS + ":duration=" + secs,
    "-c:v", "libx264", "-profile:v", "baseline", "-pix_fmt", "yuv420p",
    "-g", String(FPS), "-keyint_min", String(FPS), "-sc_threshold", "0", "-bf", "0",
    "-slices", "1", "-f", "h264", out], { stdio: ["ignore", "ignore", "pipe"] });
  return fs.readFileSync(out);
}
// Annex-B -> NAL units (start codes stripped, trailing zero padding trimmed)
function nals(buf) {
  const idx = [];
  for (let i = 0; i + 2 < buf.length; i++)
    if (buf[i] === 0 && buf[i + 1] === 0 && buf[i + 2] === 1) { idx.push(i + 3); i += 2; }
  const out = [];
  for (let k = 0; k < idx.length; k++) {
    let e = (k + 1 < idx.length) ? idx[k + 1] - 3 : buf.length;
    while (e > idx[k] && buf[e - 1] === 0) e--;          // 4-byte start code / padding
    out.push(buf.subarray(idx[k], e));
  }
  return out;
}
// NAL units -> mp4 samples (4-byte length prefixes, which IS the mp4 sample format) plus
// the avcC the stsd box needs, built from the stream's own SPS/PPS.
function samplesOf(buf) {
  let sps = null, pps = null;
  const out = [];
  let cur = [];
  for (const n of nals(buf)) {
    const t = n[0] & 0x1f;
    if (t === 7) { sps = sps || n; continue; }           // parameter sets go in avcC
    if (t === 8) { pps = pps || n; continue; }
    if (t === 9) continue;                               // access unit delimiter
    cur.push(n);
    if (t === 1 || t === 5) { out.push({ nals: cur, key: t === 5 }); cur = []; }
  }
  const chunks = out.map(s => {
    let n = 0;
    for (const x of s.nals) n += 4 + x.length;
    const b = Buffer.alloc(n);
    let o = 0;
    for (const x of s.nals) { b.writeUInt32BE(x.length, o); b.set(x, o + 4); o += 4 + x.length; }
    return { data: new Uint8Array(b), key: s.key };
  });
  const a = [1, sps[1], sps[2], sps[3], 0xff, 0xe1, sps.length >> 8, sps.length & 255];
  const avcC = Buffer.concat([Buffer.from(a), sps,
                              Buffer.from([1, pps.length >> 8, pps.length & 255]), pps]);
  return { chunks: chunks, avcC: new Uint8Array(avcC), sps: sps };
}

// ---- what the muxer actually wrote ----------------------------------------
function boxes(buf, start, end) {
  const out = [];
  let o = start;
  while (o + 8 <= end) {
    const sz = buf.readUInt32BE(o), ty = buf.toString("latin1", o + 4, o + 8);
    if (sz < 8 || o + sz > end) { out.push({ type: "BAD:" + ty, size: sz, off: o }); break; }
    out.push({ type: ty, size: sz, off: o });
    o += sz;
  }
  return out;
}
// where a box's children begin, relative to the box: plain containers hold boxes right
// after the header, while stsd is a full box with an entry count in front of them and an
// avc1 sample entry has its 78-byte visual description before avcC.
const CHILD_AT = { moov: 8, trak: 8, mdia: 8, minf: 8, stbl: 8, dinf: 8, edts: 8,
                   moof: 8, traf: 8, stsd: 16, avc1: 86 };
function allTypes(buf, start, end, into) {
  for (const b of boxes(buf, start, end)) {
    into.push(b.type);
    const at = CHILD_AT[b.type];
    if (at && b.size > at) allTypes(buf, b.off + at, b.off + b.size, into);
  }
  return into;
}
const probe = (f, args) => JSON.parse(sh("ffprobe", ["-v", "error", "-print_format", "json",
                                                     "-select_streams", "v:0"].concat(args, [f])));
// ffprobe 4.x names the per-frame timestamp pkt_pts / best_effort_timestamp, not pts
function frames(f) {
  const j = probe(f, ["-show_frames", "-show_entries",
                      "frame=key_frame,pkt_pts,best_effort_timestamp"]);
  return (j.frames || []).map(x => ({
    key: String(x.key_frame) === "1",
    pts: Number(x.pkt_pts !== undefined && x.pkt_pts !== null ? x.pkt_pts : x.best_effort_timestamp)
  }));
}
// a full decode of every sample: the real question is whether a decoder that BELIEVES the
// sample tables (as iOS does) gets a clean stream out of them
function decodeClean(f) {
  const r = require("child_process").spawnSync("ffmpeg",
    ["-v", "error", "-i", f, "-f", "null", "-"], { encoding: "utf8" });
  const out = String(r.stderr || "").trim();
  return { clean: r.status === 0 && !out, out: out || ("exit " + r.status) };
}

// ---------------------------------------------------------------------------
// one full case: encode -> cut -> mux -> interrogate
// ---------------------------------------------------------------------------
function caseMux(label, w, h, secs, limit) {
  console.log("\n" + label + "  (" + w + "x" + h + ")");
  const raw = encodeH264(w, h, secs, TMP + "/checkmp4-" + w + "x" + h + ".h264");
  const S = samplesOf(raw);
  let chunks = S.chunks;
  if (limit) chunks = chunks.slice(0, limit);
  const want = [];                                   // 1-based sync sample indices
  chunks.forEach((c, i) => { if (c.key) want.push(i + 1); });
  const mp4 = mp4Mux({ width: w, height: h, fps: FPS, avcC: S.avcC, chunks: chunks });
  ok("mp4Mux returned a file", !!mp4 && mp4.length > 0);
  if (!mp4) return;
  const f = TMP + "/checkmp4-" + w + "x" + h + (limit ? "-" + limit : "") + ".mp4";
  const buf = Buffer.from(mp4.buffer, mp4.byteOffset, mp4.byteLength);
  fs.writeFileSync(f, buf);

  const top = boxes(buf, 0, buf.length);
  const names = top.map(b => b.type).join(",");
  console.log("    top-level boxes: " + top.map(b => b.type + "(" + b.size + ")").join(" "));
  ok("top level is exactly ftyp + mdat + moov", names === "ftyp,mdat,moov", names);
  ok("the boxes tile the file with no slack",
     top.reduce((a, b) => a + b.size, 0) === buf.length,
     top.reduce((a, b) => a + b.size, 0) + " vs " + buf.length + " B");
  const inside = allTypes(buf, 0, buf.length, []);
  const banned = ["moof", "mvex", "trun", "traf", "ctts", "sdtp", "co64"].filter(t => inside.indexOf(t) >= 0);
  ok("no fragmented-mp4 or reordering boxes anywhere", banned.length === 0,
     banned.length ? "found " + banned.join(",") : "checked " + inside.length + " boxes");
  for (const need of ["stsd", "stts", "stss", "stsc", "stsz", "stco", "avcC"])
    ok("stbl carries " + need, inside.indexOf(need) >= 0);

  const st = probe(f, ["-show_streams"]).streams[0];
  console.log("    ffprobe stream: " + JSON.stringify({ codec: st.codec_name, profile: st.profile,
    w: st.width, h: st.height, r: st.r_frame_rate, avg: st.avg_frame_rate, fmt: st.pix_fmt }));
  ok("codec is h264", st.codec_name === "h264", st.codec_name);
  ok("size survived the mux", st.width === w && st.height === h, st.width + "x" + st.height);
  const fr = frames(f);
  ok("frame count == samples fed", fr.length === chunks.length, fr.length + " vs " + chunks.length);
  const got = [];
  fr.forEach((x, i) => { if (x.key) got.push(i + 1); });
  console.log("    sync samples (1-based): " + JSON.stringify(got) + "   forced: " + JSON.stringify(want));
  ok("key frames land exactly on the forced indices", JSON.stringify(got) === JSON.stringify(want));
  const pts = fr.map(x => x.pts);
  const d = [];
  for (let i = 1; i < pts.length; i++) if (d.indexOf(pts[i] - pts[i - 1]) < 0) d.push(pts[i] - pts[i - 1]);
  ok("pts deltas are all equal", d.length <= 1, "deltas seen: " + JSON.stringify(d));
  if (chunks.length > 1) {
    ok("frame rate reads back as " + FPS + "/1", st.r_frame_rate === FPS + "/1", st.r_frame_rate);
    ok("one frame is one media tick step", d[0] === 1000, String(d[0]));
  }
  const dec = decodeClean(f);
  console.log("    ffmpeg decode: " + (dec.clean ? "(no output -- clean)" : dec.out));
  ok("ffmpeg decodes the file with zero errors", dec.clean,
     dec.clean ? "no stderr at all" : dec.out.slice(0, 200));
  return f;
}

console.log("1. a full 2 s take, square 512x512 (the app's default card)");
caseMux("square", 512, 512, 2);
console.log("\n2. a NON-square canvas: the 2D wide box");
caseMux("wide box", 1024, 256, 2);
console.log("\n3. edge case: a single frame");
caseMux("one frame", 320, 180, 1, 1);

console.log("\n4. refusals and the codec string");
{
  const S = samplesOf(fs.readFileSync(TMP + "/checkmp4-320x180.h264"));
  ok("no samples -> no file", mp4Mux({ width: 8, height: 8, fps: FPS, avcC: S.avcC, chunks: [] }) === null);
  ok("no avcC -> no file", mp4Mux({ width: 8, height: 8, fps: FPS, avcC: null, chunks: S.chunks }) === null);
  ok("nothing at all -> no file", mp4Mux(null) === null);
  // the level in the codec string has to be one the frame size fits in (MaxFS): 512x512
  // and 1024x256 are both 1024 macroblocks, i.e. level 3.0 = 0x1e.
  ok("512x512 asks for Baseline level 3.0", recCodec(512, 512) === "avc1.42001e", recCodec(512, 512));
  ok("1024x256 asks for the same level", recCodec(1024, 256) === "avc1.42001e", recCodec(1024, 256));
  ok("a 4K frame moves the level up", recCodec(3840, 2160) === "avc1.420033", recCodec(3840, 2160));
  ok("the app records at " + FPS + " fps", REC_FPS === FPS, String(REC_FPS));
}

console.log(bad ? "\n" + bad + " CHECK(S) FAILED" : "\nall mp4 muxer checks passed");
process.exit(bad ? 1 : 0);
