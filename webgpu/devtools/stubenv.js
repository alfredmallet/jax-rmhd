// Shared stub environment for the devtools: a DOM + WebGPU stub good enough to run a
// real app page (common.js + physics.js + the page's inline script) under node, and to
// keep exercising it afterwards.
//
// Extracted at Phase H from bootstub.js, which dumpwgsl2.js had copied wholesale; the
// spec-driven control panel (REFINE_PLAN H.0) also means the controls no longer exist
// in the page markup, so every tool now has to BOOT the page to see them. One stub,
// three consumers (bootstub / dumpwgsl2 / layout).
//
//   const env = require("./stubenv")(dir, page, demo, opts);
//   opts.noGpu   an engine with no navigator.gpu at all: initGPU takes its first failure
//                path, so the no-WebGPU poster fallback (ONEPAGE_PLAN B.5) runs for real
//   env.run("function(){ ... }", args...)   evaluate in the page's context
//   env.getEl(id) / env.els / env.allEls    the element model
//   env.fails / env.fail(msg)               accumulated stub-level failures
"use strict";
const fs = require("fs"), vm = require("vm"), path = require("path");
const MAXWG = 65535;
// localStorage, shared across every env made by this PROCESS (not per-env): two boots
// in one gate script are "two visits by the same browser", which is exactly what the
// params-toggle memory (ONEPAGE_PLAN A.1) needs to be tested against. Also on env.store
// so a consumer can seed or inspect it directly.
const storeMap = new Map();
const store = { getItem: k => (storeMap.has(k) ? storeMap.get(k) : null),
                setItem: (k, v) => { storeMap.set(k, String(v)); },
                removeItem: k => { storeMap.delete(k); },
                clear: () => { storeMap.clear(); } };

module.exports = function makeEnv(dir, page, demo, opts) {
  const noGpu = !!(opts && opts.noGpu);
  const fails = [];
  const fail = m => { if (fails.indexOf(m) < 0) fails.push(m); };

  const html = fs.readFileSync(path.join(dir, page), "utf8");
  const script = html.slice(html.indexOf("<script>\n") + 8, html.lastIndexOf("</script>"));
  const body = html.slice(0, html.indexOf("<script src="));

  // ---- element model ---------------------------------------------------------
  function attrs(tag) {
    const a = {};
    const re = /([a-zA-Z0-9_-]+)(?:\s*=\s*"([^"]*)")?/g;
    let m;
    while ((m = re.exec(tag))) a[m[1].toLowerCase()] = m[2] === undefined ? "" : m[2];
    return a;
  }
  // ---- capture log (FEEDBACK_2026-08-10 item 13) ------------------------------
  // Everything the save / record path hands to the browser lands here, so a consumer can
  // assert that pressing the buttons really produced a blob-shaped download rather than
  // that the handler merely ran: `caps.blobs` (what toBlob / new Blob made),
  // `caps.downloads` ({name, blob} per <a download>.click()), `caps.recs` (MediaRecorders).
  const caps = { blobs: [], downloads: [], recs: [], urls: new Map(),
                 encs: [], frames: [], timeouts: [], files: [], shares: [] };
  let urlN = 0;
  const mkBlob = (type, size) => { const b = { type: type || "", size: size || 0, __blob: 1 }; caps.blobs.push(b); return b; };
  // A blob part is either another stub blob (the MediaRecorder leg, or the finished file
  // rewrapped as a File for the share sheet) or a Uint8Array (the WebCodecs leg, whose
  // whole point is the BYTES it wrote) -- so the bytes are kept and carried through the
  // rewrap, and a consumer can walk the mp4 the muxer produced box by box wherever it
  // ended up.
  function BlobStub(parts, o) {
    let n = 0;
    const keep = [];
    for (const p of (parts || [])) {
      if (p && p.byteLength !== undefined) { n += p.byteLength; keep.push(p); }
      else if (p && p.__blob && p.bytes) { n += p.bytes.byteLength; keep.push(p.bytes); }
      else n += (p && p.size) || 0;
    }
    const b = mkBlob(o && o.type, n);
    this.type = b.type; this.size = b.size; this.__blob = 1;
    if (keep.length) {
      this.bytes = new Uint8Array(n);
      let at = 0;
      for (const p of keep) { this.bytes.set(p, at); at += p.byteLength; }
    }
    caps.blobs[caps.blobs.length - 1] = this;
  }
  // a deterministic "PNG": the real signature, so a consumer can say the bytes that came
  // out are the picture, then a ramp -- content nothing reads, length the size the strip
  // quotes back
  const pngBytes = n => {
    const b = new Uint8Array(Math.max(8, n | 0));
    b.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    for (let i = 8; i < b.length; i++) b[i] = (i * 7 + 13) & 255;
    return b;
  };
  // ---- File + Web Share (the result strip, 2026-08-11) ------------------------
  // A File is a Blob with a name, and it is the object the share sheet is handed -- so it
  // keeps the bytes too (above), and a consumer can assert that what was SHARED is the
  // file that was written, not merely that a handler ran. `caps.files` is every File the
  // page built, `caps.shares` every navigator.share() payload.
  function FileStub(parts, name, o) {
    BlobStub.call(this, parts, o);
    this.name = String(name === undefined ? "" : name);
    this.lastModified = clock;
    caps.files.push(this);
  }
  // Two knobs, because both branches are real engines: `can` is whether this engine can
  // share FILES at all (a desktop that cannot must simply grow no share button), and
  // `reject` makes share() reject with a named error -- "AbortError" being the visitor
  // closing the sheet, which must NOT then download at them, against anything else, which
  // must. Reached from a consumer as `env.share`.
  const share = { can: true, reject: "" };
  const canShareStub = d => !!(share.can && d && d.files && d.files.length);
  const shareStub = d => {
    if (!canShareStub(d)) return Promise.reject(new Error("share() of something unshareable"));
    caps.shares.push(d);
    if (!share.reject) return Promise.resolve();
    const e = new Error(share.reject);
    e.name = share.reject;
    return Promise.reject(e);
  };
  // ---- the clock -------------------------------------------------------------
  // The recorder's leg 2 has no frame count of its own and times itself by wall clock, so
  // the clock is the stub's: `env.advance(ms)` moves it and a "12 s recording" costs no
  // wall clock, exactly as `env.tick(n)` does for leg 1's frame pump. Nothing else in the
  // apps reads Date, so this shadows it wholesale rather than patching `now` in place.
  let clock = 17e11;
  function DateStub(...a) { return a.length ? new Date(...a) : new Date(clock); }
  DateStub.now = () => clock;
  DateStub.parse = Date.parse; DateStub.UTC = Date.UTC;
  const advance = ms => { clock += ms; };
  function MediaRecorderStub(stream, o) {
    this.stream = stream; this.mimeType = (o && o.mimeType) || "video/webm";
    this.state = "inactive"; this.ondataavailable = null; this.onstop = null;
    caps.recs.push(this);
  }
  MediaRecorderStub.isTypeSupported = m => m === "video/webm;codecs=vp9";
  MediaRecorderStub.prototype.start = function () {
    if (this.state === "recording") fail("MediaRecorder.start() while already recording");
    this.state = "recording";
  };
  MediaRecorderStub.prototype.stop = function () {
    if (this.state !== "recording") return fail("MediaRecorder.stop() while not recording");
    this.state = "inactive";
    if (this.ondataavailable) this.ondataavailable({ data: mkBlob(this.mimeType, 4096) });
    if (this.onstop) this.onstop();
  };
  // ---- WebCodecs: the preferred recording leg --------------------------------
  // A deterministic fake encoder: one chunk per encode() call, sizes and bytes derived
  // from the frame index, and the avcC-shaped decoderConfig.description on the first
  // chunk -- enough for the app's VideoEncoder -> mp4Mux path (pump, forced keyframes,
  // flush, download) to run end to end with no GPU and no browser. Two knobs a consumer
  // drives the awkward branches with: `stall` (hold the outputs back, so encodeQueueSize
  // grows and the drop-frame guard fires) and `noAvcC` (an engine that never sends a
  // description, which must make the app bail to MediaRecorder).
  const WC_AVCC = new Uint8Array([1, 0x42, 0x00, 0x1e, 0xff, 0xe1, 0, 4,
                                  0x67, 0x42, 0x00, 0x1e, 1, 0, 4, 0x68, 0xce, 0x3c, 0x80]);
  function VideoFrameStub(src, o) {
    if (!src || !src.getContext) fail("VideoFrame built from something that is not a canvas");
    this.timestamp = (o && o.timestamp) | 0;
    this.duration = o && o.duration;
    this.codedWidth = (src && src.width) | 0; this.codedHeight = (src && src.height) | 0;
    this.closed = false;
    caps.frames.push(this);
  }
  VideoFrameStub.prototype.close = function () { this.closed = true; };
  function EncodedVideoChunkStub(o) {
    this.type = (o && o.type) || "key"; this.timestamp = (o && o.timestamp) | 0;
    this.data = (o && o.data) || new Uint8Array(0);
    this.byteLength = this.data.byteLength;
  }
  EncodedVideoChunkStub.prototype.copyTo = function (d) { d.set(this.data); };
  function VideoEncoderStub(init) {
    this.state = "unconfigured"; this.encodeQueueSize = 0;
    this.stall = false; this.noAvcC = false;
    this.config = null; this.frames = 0; this.keys = 0; this.sent = 0;
    this._out = init && init.output; this._err = init && init.error; this._q = [];
    caps.encs.push(this);
  }
  VideoEncoderStub.isConfigSupported = cfg => Promise.resolve({
    supported: /^avc1\.42/.test((cfg && cfg.codec) || "") && cfg.width > 0 && cfg.height > 0,
    config: cfg });
  VideoEncoderStub.prototype.configure = function (cfg) {
    if (!cfg || !cfg.avc || cfg.avc.format !== "avc")
      fail("VideoEncoder.configure without avc {format:'avc'}: there would be no avcC");
    if (!(cfg.width > 0) || !(cfg.height > 0)) fail("VideoEncoder.configure with no frame size");
    this.config = cfg; this.state = "configured";
  };
  VideoEncoderStub.prototype.encode = function (frame, opt) {
    if (this.state !== "configured") return fail("VideoEncoder.encode before configure()");
    if (!frame || frame.closed) return fail("VideoEncoder.encode of a closed VideoFrame");
    const key = !!(opt && opt.keyFrame);
    this.frames++; if (key) this.keys++;
    const n = key ? 400 : 120;
    const d = new Uint8Array(n);
    d[0] = ((n - 4) >>> 24) & 255; d[1] = ((n - 4) >>> 16) & 255;   // a length-prefixed
    d[2] = ((n - 4) >>> 8) & 255;  d[3] = (n - 4) & 255;            // "NAL", as the real
    d[4] = key ? 0x65 : 0x41;                                       // encoder emits
    for (let i = 5; i < n; i++) d[i] = (i + this.frames) & 255;
    this._q.push(new EncodedVideoChunkStub({ type: key ? "key" : "delta", timestamp: frame.timestamp, data: d }));
    this.encodeQueueSize = this._q.length;
    this.pump();
  };
  VideoEncoderStub.prototype.pump = function () {
    if (this.stall) return;
    while (this._q.length) {
      const c = this._q.shift();
      this.encodeQueueSize = this._q.length;
      const meta = (this.sent++ === 0 && !this.noAvcC) ? { decoderConfig: { description: WC_AVCC } } : {};
      if (this._out) this._out(c, meta);
    }
  };
  VideoEncoderStub.prototype.flush = function () {
    return Promise.resolve().then(() => { this.stall = false; this.pump(); });
  };
  VideoEncoderStub.prototype.close = function () { this.state = "closed"; this._q.length = 0; };
  VideoEncoderStub.prototype.reset = function () { this._q.length = 0; this.encodeQueueSize = 0; };

  // ---- timers ----------------------------------------------------------------
  // setInterval is the recorder's frame pump, so it is NOT armed for real: the stub keeps
  // the callbacks in a list and `env.tick(n)` fires exactly n frames -- deterministic, and
  // a 30 s recording costs no wall clock. setTimeout stays real (the page awaits it), but
  // every pending one is remembered so a consumer can fire the 30 s hard stop by hand.
  const ivals = new Map();
  let ivalN = 0;
  const setIntervalStub = (fn, ms) => { ivals.set(++ivalN, { fn, ms }); return ivalN; };
  const clearIntervalStub = id => { ivals.delete(id); };
  function tick(n) {
    for (let i = 0; i < (n === undefined ? 1 : n); i++) {
      for (const e of caps.encs) e.pump();          // the encoder made progress meanwhile
      for (const t of Array.from(ivals.values())) t.fn();
    }
  }
  const setTimeoutStub = (fn, ms) => {
    const rec = { ms: ms, fn: fn, fired: false };
    rec.id = setTimeout(() => { rec.fired = true; fn(); }, ms);
    caps.timeouts.push(rec);
    return rec.id;
  };
  const clearTimeoutStub = id => {
    clearTimeout(id);
    const i = caps.timeouts.findIndex(t => t.id === id);
    if (i >= 0) caps.timeouts.splice(i, 1);
  };
  // fire the newest pending timeout of a given delay by hand (the 30 s recording cap)
  function fireTimeout(ms) {
    const t = caps.timeouts.filter(x => x.ms === ms && !x.fired).pop();
    if (!t) return false;
    clearTimeoutStub(t.id);
    t.fired = true; t.fn();
    return true;
  }

  const URLStub = {
    createObjectURL(b) {
      if (!b || !b.__blob) fail("createObjectURL of something that is not a blob");
      const u = "blob:stub/" + (++urlN);
      caps.urls.set(u, b);
      return u;
    },
    revokeObjectURL(u) { caps.urls.delete(u); }
  };

  const allEls = [];
  // a real DOM reflects the markup's style attribute into el.style; the app now relies
  // on that (the control panel is hidden in the MARKUP so it cannot flash before boot)
  const parseStyle = s => {
    const o = {};
    for (const d of (s || "").split(";")) {
      const i = d.indexOf(":");
      if (i < 0) continue;
      const k = d.slice(0, i).trim().replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      o[k] = d.slice(i + 1).trim();
    }
    return o;
  };
  function mkEl(id, kind, at) {
    const e = {
      id: id || "", kind: kind || "div", _value: "", _html: "", checked: false, disabled: false,
      textContent: "", className: "", style: at && at.style ? parseStyle(at.style) : {},
      width: 0, height: 0, title: "",
      min: "", max: "", step: "", type: "", options: [], children: [], parentNode: null,
      attrs: at || {},
      classList: { _s: {}, toggle(c, v) { this._s[c] = v === undefined ? !this._s[c] : !!v; },
                   add(c) { this._s[c] = true; }, remove(c) { this._s[c] = false; },
                   contains(c) { return !!this._s[c]; } },
      hasAttribute(a) { return a in this.attrs; },
      setAttribute(a, v) { this.attrs[a] = v === undefined ? "" : String(v); },
      addEventListener() {}, setPointerCapture() {},
      // the capture path (item 13): a canvas is a PNG source and a video source, and the
      // <a download> the blob is handed to is an ordinary element too. The picture keeps
      // real BYTES (2026-08-12): since the save path ends on the result strip too, the PNG
      // is rewrapped as a File for the share sheet, and "what was shared is what was
      // written" is only a checkable statement when there is something to compare. The
      // callback is deferred, as a browser's is -- which is what lets a consumer close the
      // card between the press and the picture and drive the dead-card branch honestly.
      toBlob(cb, type) {
        const b = pngBytes(4 * (this.width | 0) * (this.height | 0));
        setTimeout(() => cb(new BlobStub([b], { type: type || "image/png" })), 0);
      },
      captureStream(fps) { return { __stream: 1, fps: fps }; },
      click() {
        if (this.download === undefined) return;
        const b = caps.urls.get(this.href);
        if (!b) fail('<a download="' + this.download + '"> clicked with no live object URL');
        caps.downloads.push({ name: this.download, blob: b });
      },
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 512, height: 512 }),
      appendChild(c) {
        c.parentNode = this; this.children.push(c);
        if (this.kind === "select") { this.options.push(c); if (this.options.length === 1) this._value = c.value; }
        return c;
      },
      removeChild(c) {
        const i = this.children.indexOf(c);
        if (i < 0) return fail("removeChild of a non-child");
        this.children.splice(i, 1); c.parentNode = null;
        const j = this.options.indexOf(c); if (j >= 0) this.options.splice(j, 1);
        return c;
      },
      getContext: k => (k === "2d" ? ctx2d() : gpuCanvasCtx())
    };
    Object.defineProperty(e, "value", {
      get() { return this._value; },
      set(v) {
        this._value = String(v);
        if (this.options.length && !this.options.some(o => o.value === this._value))
          fail((this.id || this.kind) + ': value "' + v + '" is not one of its <option>s');
      }
    });
    Object.defineProperty(e, "selectedIndex", {
      get() { const i = this.options.findIndex(o => o.value === this._value); return i < 0 ? 0 : i; },
      set() {}
    });
    Object.defineProperty(e, "innerHTML", {
      get() { return this._html; },
      set(v) {
        this._html = String(v);
        if (this._html === "") { this.children.length = 0; this.options.length = 0; }
      }
    });
    allEls.push(e);
    return e;
  }
  const els = {};                                     // ids declared in the page's markup
  for (const m of body.matchAll(/<select\b([^>]*)>([\s\S]*?)<\/select>/g)) {
    const a = attrs(m[1]);
    if (!a.id) continue;
    const e = mkEl(a.id, "select", a);
    for (const o of m[2].matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/g)) {
      const oa = attrs(o[1]);
      const oe = mkEl("", "option", oa);
      oe._value = oa.value !== undefined ? oa.value : o[2].trim();
      oe._html = o[2].trim(); oe.textContent = o[2].trim();
      e.options.push(oe); e.children.push(oe);
      if (oa.selected !== undefined) e._value = oe._value;
    }
    if (!e._value && e.options.length) e._value = e.options[0].value;
    e.disabled = "disabled" in a;
    els[a.id] = e;
  }
  for (const m of body.matchAll(/<input\b([^>]*)>/g)) {
    const a = attrs(m[1]);
    if (!a.id) continue;
    const e = mkEl(a.id, "input", a);
    e._value = a.value || ""; e.checked = "checked" in a; e.type = a.type || "text";
    e.min = a.min || ""; e.max = a.max || ""; e.step = a.step || "";
    e.disabled = "disabled" in a;
    els[a.id] = e;
  }
  // every other id the markup declares (divs, buttons, spans, canvases)
  for (const m of body.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const id = m[3];
    if (els[id]) continue;
    els[id] = mkEl(id, m[1].toLowerCase(), attrs(m[2]));
  }
  const markup = () => new Set(Object.keys(els).map(k => els[k]));
  function attached(e) {
    let p = e;
    while (p.parentNode) p = p.parentNode;
    return markup().has(p);
  }
  function getEl(id) {
    if (els[id]) return els[id];
    // last ATTACHED match: a card that was closed leaves detached elements behind, and
    // getElementById must never hand those out
    const dyn = allEls.filter(e => e.id === id && attached(e)).pop();
    if (dyn) return dyn;
    fail('getElementById("' + id + '") : no such element in the page');
    return (els[id] = mkEl(id));
  }
  // the control panel is BUILT (H.0), so "#controls details" has to walk the live tree
  function descendants(root, kind) {
    const out = [];
    (function rec(e) {
      for (const c of (e.children || [])) { if (c.kind === kind) out.push(c); rec(c); }
    })(root);
    return out;
  }

  // ---- 2D canvas stub: fillText paints a solid block, filter is NOT supported ----
  function ctx2d() {
    const st = { w: 0, h: 0, buf: null };
    const o = { fillStyle: "#000", strokeStyle: "#000", lineWidth: 1, font: "10px x",
                textAlign: "left", textBaseline: "alphabetic", filter: "none",
                lineCap: "butt", lineJoin: "miter" };
    for (const m of ["setTransform", "clearRect", "strokeRect", "beginPath", "moveTo", "lineTo",
                     "stroke", "fill", "clip", "save", "restore", "setLineDash", "rect",
                     "putImageData", "drawImage"]) {
      o[m] = (...a) => { for (const v of a) if (typeof v === "number" && !isFinite(v)) fail("non-finite arg to " + m); };
    }
    const fontPx = () => { const m = /(\d+(?:\.\d+)?)px/.exec(o.font); return m ? parseFloat(m[1]) : 10; };
    o.measureText = t => ({ width: 0.62 * fontPx() * t.length,
                            actualBoundingBoxAscent: 0.72 * fontPx(), actualBoundingBoxDescent: 0 });
    o.fillRect = (x, y, w, h) => { if (!st.buf) { st.w = w | 0; st.h = h | 0; st.buf = new Uint8ClampedArray(4 * st.w * st.h); } };
    o.fillText = (t, cx, cy) => {
      if (!st.buf) return;
      const s = fontPx(), hw = 0.3 * s, hh = 0.36 * s;
      for (let py = 0; py < st.h; py++) for (let px = 0; px < st.w; px++) {
        if (Math.abs(px + 0.5 - cx) <= hw && Math.abs(py + 0.5 - cy) <= hh) {
          const i = 4 * (py * st.w + px);
          st.buf[i] = 255; st.buf[i + 1] = 255; st.buf[i + 2] = 255; st.buf[i + 3] = 255;
        }
      }
    };
    o.getImageData = () => ({ data: st.buf || new Uint8ClampedArray(4) });
    o.createImageData = (w, h) => ({ data: new Uint8ClampedArray(4 * w * h) });
    return o;
  }

  // ---- Path2D: what the overlays (arrows, field lines) build and hand to stroke() ----
  // Every coordinate that reaches it is checked for finiteness and counted, so a caller
  // can assert that an overlay really drew and that nothing NaN'd on the way in.
  function Path2DStub() { this.pts = 0; }
  for (const m of ["moveTo", "lineTo"]) {
    Path2DStub.prototype[m] = function (x, y) {
      if (!isFinite(x) || !isFinite(y)) fail("non-finite " + m + "(" + x + ", " + y + ") on a Path2D");
      this.pts++;
    };
  }

  // ---- WebGPU stub -----------------------------------------------------------
  const live = { buffers: 0, textures: 0 };
  const mkBuf = o => { live.buffers++; return { size: o.size, usage: o.usage,
    destroy() { live.buffers--; }, async mapAsync() {}, getMappedRange() { return new ArrayBuffer(o.size); }, unmap() {} }; };
  function mkPass(kind) {
    return {
      setPipeline(p) { if (!p) fail(kind + ": setPipeline(undefined)"); this._p = p; },
      setBindGroup(i, b) { if (!b) fail(kind + ": setBindGroup(" + i + ") undefined after " + (this._p && this._p.__name)); },
      dispatchWorkgroups(x, y, z) {
        for (const [n, v] of [["x", x], ["y", y === undefined ? 1 : y], ["z", z === undefined ? 1 : z]]) {
          if (!(v >= 1) || !isFinite(v) || v !== Math.floor(v))
            fail(kind + ": bad dispatch " + n + " = " + v + " (" + (this._p && this._p.__name) + ")");
          if (v > MAXWG) fail(kind + ": dispatch " + n + " = " + v + " > 65535 (" + (this._p && this._p.__name) + ")");
        }
      },
      draw() {}, end() {}
    };
  }
  const device = {
    createBuffer: mkBuf,
    createTexture: () => { live.textures++; return { createView: () => ({ __v: 1 }), destroy() { live.textures--; } }; },
    createSampler: () => ({}), createShaderModule: o => ({ code: o.code }),
    createComputePipeline: o => ({ __name: o.compute.module && o.compute.module.__name, getBindGroupLayout: () => ({}) }),
    createRenderPipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup(o) {
      if (!o.layout) fail("bind group with no layout");
      o.entries.forEach(e => {
        if (!e.resource) fail("bind group entry " + e.binding + ": undefined resource");
        else if ("buffer" in e.resource && !e.resource.buffer) fail("bind group entry " + e.binding + ": undefined buffer");
      });
      return { __bg: 1 };
    },
    createCommandEncoder: () => ({
      beginComputePass: () => mkPass("compute"),
      beginRenderPass: o => { if (!o.colorAttachments[0].view) fail("render pass without a view"); return mkPass("render"); },
      clearBuffer(b) { if (!b) fail("clearBuffer(undefined)"); },
      copyBufferToBuffer(a, ao, b) { if (!a || !b) fail("copyBufferToBuffer(undefined)"); },
      finish: () => ({})
    }),
    queue: {
      writeBuffer(b, off, data) {
        if (!b) return fail("writeBuffer(undefined)");
        const n = data.byteLength !== undefined ? data.byteLength : data.length * 4;
        if (off + n > b.size) fail("writeBuffer overflows: " + (off + n) + " > " + b.size);
      },
      submit() {}, onSubmittedWorkDone: async () => {}
    },
    addEventListener() {}, lost: { then() {} }
  };
  const gpuCanvasCtx = () => ({ configure() {}, getCurrentTexture: () => ({ createView: () => ({ __v: 1 }) }) });

  const search = demo ? "?demo=" + demo : "";
  const sandbox = {
    document: {
      // The recorder's watchdog feeds only on a page the rAF loop is known-absent from
      // (document.hidden, or the editor view). This stub has no rAF loop at all, so it
      // reports HIDDEN -- which keeps every pumped recording leg on the timer path, the
      // path env.tick() drives. The parked branch is asserted by a bootstub leg flipping
      // this to false (RECRAF round 2, 2026-08-12).
      hidden: true,
      getElementById: getEl,
      createElement: t => mkEl("", t.toLowerCase()),
      createTextNode: t => ({ kind: "#text", textContent: t }),
      querySelectorAll: sel => (sel === "#controls details" ? descendants(getEl("controls"), "details") : []),
      // ANALYTICS_PLAN phase 2: the ONLY selector the apps pass to querySelector is
      // ".buildid" (contactBody -- the build id has a CLASS and no id, because pages.yml
      // seds that exact span and must keep finding it). The element model above is built
      // from tags that declare an id, so a class-only span is not in it; read it straight
      // out of the markup instead. Deliberately narrow: any other selector returns null,
      // which is the honest answer from a stub that does not implement it.
      querySelector: sel => {
        if (sel !== ".buildid") return null;
        const m = body.match(/<span class="buildid">([^<]*)<\/span>/);
        return m ? { textContent: m[1] } : null;
      }
    },
    // innerWidth/innerHeight: contactBody reports the viewport, and without them that
    // line took the absent branch and was never exercised. 1280x800 is the first-screen
    // size the one-page layout is checked at.
    window: { addEventListener() {}, devicePixelRatio: 1, innerWidth: 1280, innerHeight: 800,
              matchMedia: q => ({ matches: /min-width/.test(q) }),
              // item 13: the browser globals the capture path feature-detects and uses --
              // both recording legs. Present here, so the stub exercises the SUPPORTED
              // branch of each; a page that must survive their absence is checked by
              // deleting them (bootstub).
              URL: URLStub, Blob: BlobStub, MediaRecorder: MediaRecorderStub,
              VideoEncoder: VideoEncoderStub, VideoFrame: VideoFrameStub,
              EncodedVideoChunk: EncodedVideoChunkStub,
              // the result strip feature-detects `window.File` before building one
              File: FileStub },
    location: { search, href: "file:///x/" + page + search },
    URLSearchParams, navigator: {
      // ANALYTICS_PLAN phase 2: contactBody reads userAgent, and every real browser has
      // one -- a stub without it would exercise only the absent branch.
      userAgent: "stubenv/1.0 (node; not a browser)",
      // Web Share level 2, which is what the result strip's share button is detected on
      canShare: canShareStub, share: shareStub,
      gpu: noGpu ? null : {
        getPreferredCanvasFormat: () => "bgra8unorm",
        // `info` mirrors the recent-Chrome adapter.info contactBody stashes into gpuInfo.
        // Browsers that lack it are covered too -- gpuInfo simply stays "", and nothing
        // is allowed to depend on it being non-empty.
        async requestAdapter() { return { limits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 31 },
                                          info: { vendor: "stub", architecture: "stub-arch",
                                                  device: "", description: "stub adapter" },
                                          async requestDevice() { return device; } }; }
      }
    },
    performance: { now: (function () { let t = 1000; return () => (t += 250); })() },
    requestAnimationFrame: () => {},
    Path2D: Path2DStub,
    console, Math, JSON, Float32Array, Float64Array, Uint32Array, Uint8Array, Uint8ClampedArray,
    Map, Set, Error, Promise, Number, String, Array, Object, isFinite, parseInt, parseFloat,
    setTimeout: setTimeoutStub, clearTimeout: clearTimeoutStub,
    setInterval: setIntervalStub, clearInterval: clearIntervalStub, Date: DateStub,
    GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, MAP_READ: 16 },
    GPUTextureUsage: { STORAGE_BINDING: 1, TEXTURE_BINDING: 2 },
    GPUMapMode: { READ: 1 },
    localStorage: store
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  for (const f of ["common.js", "physics.js"]) vm.runInContext(fs.readFileSync(path.join(dir, f), "utf8"), sandbox, { filename: f });
  vm.runInContext(script, sandbox, { filename: page });   // boot() runs at the end

  const run = (src, ...a) => vm.runInContext("(" + src + ")", sandbox)(...a);
  return { sandbox, run, getEl, els, allEls, descendants, fails, fail, live, caps,
           tick, fireTimeout, advance, share, store, is3d: page.indexOf("3d") >= 0 };
};
