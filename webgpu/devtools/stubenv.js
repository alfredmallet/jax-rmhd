// Shared stub environment for the devtools: a DOM + WebGPU stub good enough to run a
// real app page (common.js + physics.js + the page's inline script) under node, and to
// keep exercising it afterwards.
//
// Extracted at Phase H from bootstub.js, which dumpwgsl2.js had copied wholesale; the
// spec-driven control panel (REFINE_PLAN H.0) also means the controls no longer exist
// in the page markup, so every tool now has to BOOT the page to see them. One stub,
// three consumers (bootstub / dumpwgsl2 / layout).
//
//   const env = require("./stubenv")(dir, page, demo);
//   env.run("function(){ ... }", args...)   evaluate in the page's context
//   env.getEl(id) / env.els / env.allEls    the element model
//   env.fails / env.fail(msg)               accumulated stub-level failures
"use strict";
const fs = require("fs"), vm = require("vm"), path = require("path");
const MAXWG = 65535;

module.exports = function makeEnv(dir, page, demo) {
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
  const allEls = [];
  function mkEl(id, kind, at) {
    const e = {
      id: id || "", kind: kind || "div", _value: "", _html: "", checked: false, disabled: false,
      textContent: "", className: "", style: {}, width: 0, height: 0, title: "",
      min: "", max: "", step: "", type: "", options: [], children: [], parentNode: null,
      attrs: at || {},
      classList: { _s: {}, toggle(c, v) { this._s[c] = v === undefined ? !this._s[c] : !!v; },
                   add(c) { this._s[c] = true; }, remove(c) { this._s[c] = false; },
                   contains(c) { return !!this._s[c]; } },
      hasAttribute(a) { return a in this.attrs; },
      setAttribute(a, v) { this.attrs[a] = v === undefined ? "" : String(v); },
      addEventListener() {}, setPointerCapture() {},
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
      getElementById: getEl,
      createElement: t => mkEl("", t.toLowerCase()),
      createTextNode: t => ({ kind: "#text", textContent: t }),
      querySelectorAll: sel => (sel === "#controls details" ? descendants(getEl("controls"), "details") : [])
    },
    window: { addEventListener() {}, devicePixelRatio: 1,
              matchMedia: q => ({ matches: /min-width/.test(q) }) },
    location: { search, href: "file:///x/" + page + search },
    URLSearchParams, navigator: {
      gpu: {
        getPreferredCanvasFormat: () => "bgra8unorm",
        async requestAdapter() { return { limits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 31 },
                                          async requestDevice() { return device; } }; }
      }
    },
    performance: { now: (function () { let t = 1000; return () => (t += 250); })() },
    requestAnimationFrame: () => {},
    Path2D: Path2DStub,
    console, Math, JSON, Float32Array, Float64Array, Uint32Array, Uint8ClampedArray, Map, Set,
    Error, Promise, setTimeout, Number, String, Array, Object, isFinite, parseInt, parseFloat,
    GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, MAP_READ: 16 },
    GPUTextureUsage: { STORAGE_BINDING: 1, TEXTURE_BINDING: 2 },
    GPUMapMode: { READ: 1 }
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  for (const f of ["common.js", "physics.js"]) vm.runInContext(fs.readFileSync(path.join(dir, f), "utf8"), sandbox, { filename: f });
  vm.runInContext(script, sandbox, { filename: page });   // boot() runs at the end

  const run = (src, ...a) => vm.runInContext("(" + src + ")", sandbox)(...a);
  return { sandbox, run, getEl, els, allEls, descendants, fails, fail, live, is3d: page.indexOf("3d") >= 0 };
};
