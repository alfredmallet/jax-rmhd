// ISO_PLAN gate. Phases A-C: the box-unit aspect of `cubeQuads`, the volume raymarch and
// the collision preset built on it, plus the two disciplines the WHOLE plan runs under --
// every emitted kernel parses and resolves, and the PHYSICS WGSL is byte-identical to the
// plan's base commit (this is a render-path plan).
//
//   node checkiso.js [webgpu-dir]        exit code 1 on any failure
//
// Structure: LEGS below is a list of [title, fn]; later phases append their own (the
// k-perp filter) and touch nothing else. The base-commit legs shell out to git, so they
// run from a work tree, at any cwd.
//
// What is the app's own code and what is not: everything. Sections 1-2 emit the REAL
// kernels through `dumpwgsl2` (both pages, every resolution preset and the self-test
// grid) and diff them against the same emission from `git show <BASE>:...`; section 3
// boots the REAL rmhd3d page on stubenv and drives the Lz select, so the aspect is
// measured through the whole path -- select -> uiParams -> new Solver -> cubeQuads ->
// cubeFrame -- and not off a re-implementation of the projection; sections 5-7 EXECUTE
// the emitted raymarch (wgsl_reflect's WGSL interpreter) against a CPU reference march,
// on a ray uniform the page's own volRay computed.
"use strict";
const fs = require("fs"), os = require("os"), path = require("path");
const { spawnSync } = require("child_process");
const { pathToFileURL } = require("url");
const dir = path.resolve(process.argv[2] || path.join(__dirname, ".."));
const root = path.resolve(dir, "..");
const BASE = "c3c7195";                 // ISO_PLAN's base commit (ANISO landed)
// The kernels this plan is allowed to touch. Everything else is PHYSICS and must not move
// by a byte. Phase A was CPU-side, so the expectation was "not even these"; Phase B changes
// that expectation DELIBERATELY, and this is the whole of it:
//   prepDisp                 two branch lines for the Elsasser vorticities omega+- (a
//                            DISPLAY kernel -- it prepares the display chain's k-space
//                            field, and nothing the solver steps reads what it writes),
//                            and, in Phase D, the k_perp band factor multiplied into that
//                            field plus the two extra uniform words it reads. The TOUCHED
//                            set is unchanged by Phase D: the band lives in this one kernel,
//                            it declares its own wider Mode struct rather than widening the
//                            shared one, and so sliceExtract / faceExtract / cutPrep / the
//                            colorize pair are all still byte-identical to base.
//   colorize / colorizeCube  dispX's magnitude test closed from `mode >= 4` to 4..7,
//                            because the modes past the sigma pair are signed again
//   renderVol                the one new shader
//   vecMagVol, maxPartialVol, maxFinalVol
//                            volume-length INSTANTIATIONS of templates the slice and face
//                            targets already emit -- asserted below to be exactly that
const DISPLAY = new Set(["render", "renderCube", "colorize", "colorizeCube", "prepDisp"]);
// ... and what each page's Phase B diff is expected to BE, kernel by kernel. A display
// kernel that moved and is not here fails the leg; so does one here that did not move,
// because a stale expectation hides the next change.
const TOUCHED = { "rmhd2d.html": ["colorize", "prepDisp"],
                  "rmhd3d.html": ["colorize", "colorizeCube", "prepDisp"] };
const ADDED = { "rmhd2d.html": [],
                "rmhd3d.html": ["maxFinalVol", "maxPartialVol", "renderVol", "vecMagVol"] };
// the three added NON-shader kernels, as (instance, template source, substitution): the
// vol target's collapse and autoscale must be the SLICE target's own text at the volume's
// length, or they are a second copy of the arithmetic instead of one template.
const INSTANCES = [["vecMagVol", "vecMag"], ["maxPartialVol", "maxPartial"],
                   ["maxFinalVol", "maxFinal"]];

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};
const rel = (a, b) => Math.abs(a - b) / Math.max(1e-300, Math.abs(b));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "checkiso-"));
const sh = (cmd, args, opts) => spawnSync(cmd, args, Object.assign(
  { encoding: "utf8", cwd: __dirname, maxBuffer: 1 << 28 }, opts || {}));
const node = (args, opts) => sh(process.execPath, args, opts);
const lastLine = r => ((r.stdout || "") + (r.stderr || "")).trim().split("\n").pop();

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
// every kernel of one page, at every resolution preset + the self-test grid, as a
// { label :: kernel -> source } map, emitted from `d` by the real dumpwgsl2
function dumpKernels(d, page, tag) {
  const out = path.join(tmp, tag + "-" + page.replace(".html", "") + ".wgsl.txt");
  const r = node([path.join(__dirname, "dumpwgsl2.js"), d, page, "", out]);
  if (r.status !== 0) { ok(tag + " " + page + ": dumpwgsl2 ran", false, lastLine(r)); return { file: out, k: {} }; }
  const txt = fs.readFileSync(out, "utf8");
  const parts = txt.split(/^########## (.*) ##########$/m);
  const k = {};
  for (let i = 1; i < parts.length; i += 2) k[parts[i]] = parts[i + 1];
  return { file: out, k };
}
// the plan's base commit, checked out into tmp/base (the four files stubenv reads)
function baseDir() {
  const d = path.join(tmp, "base");
  fs.mkdirSync(d, { recursive: true });
  for (const f of ["common.js", "physics.js", "rmhd2d.html", "rmhd3d.html"]) {
    const r = sh("git", ["-C", root, "show", BASE + ":webgpu/" + f], { maxBuffer: 1 << 28 });
    if (r.status !== 0) return null;
    fs.writeFileSync(path.join(d, f), r.stdout);
  }
  return d;
}
// boot() runs at require, its async tail a macrotask later (the checkonepage idiom)
const boot = (d, page, demo) => new Promise(res => {
  const env = require("./stubenv")(d, page, demo);
  setTimeout(() => res(env));
});

// ---------------------------------------------------------------------------
// 1. every kernel parses; names / duplication discipline
// ---------------------------------------------------------------------------
function legDiscipline(state) {
  // wgslparse.mjs imports wgsl_reflect from an absolute path; devtools has it installed
  // locally, so point WGSL_REFLECT there (the ACORN idiom names.mjs already uses)
  const wr = path.join(__dirname, "node_modules", "wgsl_reflect", "wgsl_reflect.module.js");
  const env = Object.assign({}, process.env);
  if (fs.existsSync(wr)) env.WGSL_REFLECT = wr;
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const d = dumpKernels(dir, page, "cur");
    state.cur[page] = d.k;
    const r = node([path.join(__dirname, "wgslparse.mjs"), d.file], { env });
    ok(page + ": every emitted kernel parses (+ reserved-word scan)", r.status === 0, lastLine(r));
  }
  const nm = node([path.join(__dirname, "names.mjs"), dir]);
  ok("names.mjs: no redeclares, every free identifier resolves", nm.status === 0, lastLine(nm));
  // dup.py wants the shared core plus the pages' EXTRACTED inline scripts
  const files = [path.join(dir, "common.js"), path.join(dir, "physics.js")];
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const t = fs.readFileSync(path.join(dir, page), "utf8");
    const f = path.join(tmp, page.replace(".html", ".js"));
    fs.writeFileSync(f, t.slice(t.indexOf("<script>\n") + 8, t.lastIndexOf("</script>")));
    files.push(f);
  }
  const dp = sh("python3", [path.join(__dirname, "dup.py")].concat(files));
  // dup.py's GROUP COUNT is not a stable number to pin: it collapses overlapping windows
  // by absolute line number, so inserting a comment anywhere moves it (base itself reports
  // 17 clone groups, all of them the standing rmhd2d/rmhd3d twins of the app scripts).
  // What is stable, and is the actual discipline, is WHERE the clones are: a clone inside
  // one file, or one reaching into common.js / physics.js, is code that wants sharing.
  const groups = (dp.stdout || "").split("\n").filter(l => l.indexOf("CLONE") === 0)
    .map(l => l.split("):")[1].split(";").map(s => s.trim().split(":")[0]));
  const bads = groups.filter(g => new Set(g).size !== 2 ||
                                  g.some(f => f === "common.js" || f === "physics.js"));
  ok("dup.py: no clone inside a file, none reaching into the shared core",
     dp.status === 0 && bads.length === 0,
     groups.length + " groups, all app-script twins" + (bads.length ? "; BAD: " + bads[0].join(" ") : ""));
}

// ---------------------------------------------------------------------------
// 2. physics WGSL byte-identical to the base commit
// ---------------------------------------------------------------------------
function legByteIdentical(state) {
  const bd = baseDir();
  if (!bd) { ok("base " + BASE + " is readable (git show)", false, "git show failed"); return; }
  state.baseDir = bd;
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const base = dumpKernels(bd, page, "base").k;
    state.base[page] = base;
    const cur = state.cur[page] || dumpKernels(dir, page, "cur").k;
    const keys = new Set(Object.keys(base).concat(Object.keys(cur)));
    const moved = [], touched = new Set(), added = new Set();
    for (const k of keys) {
      const name = k.split(" :: ")[1];
      if (base[k] === cur[k]) continue;
      if (base[k] === undefined) { added.add(name); continue; }
      if (DISPLAY.has(name)) { touched.add(name); continue; }
      moved.push(k);                     // a physics kernel, or one that VANISHED
    }
    ok(page + ": physics WGSL byte-identical to " + BASE,
       moved.length === 0, moved.length ? moved.length + " changed, first: " + moved[0]
                                        : Object.keys(cur).length + " kernels");
    const got = [...touched].sort().join(", "), want = TOUCHED[page].join(", ");
    ok("  ... and the display kernels that moved are exactly Phase B's", got === want,
       got || "none");
    const gotA = [...added].sort().join(", "), wantA = ADDED[page].join(", ");
    ok("  ... and the kernels it ADDS are exactly Phase B's", gotA === wantA, gotA || "none");
    // the three added non-shader kernels are the SLICE target's own text at the volume's
    // length: same template, one instantiation, no second copy of the arithmetic. The
    // preamble is shared, so the comparison is of the kernel BODY, with the slice length
    // (NRS) substituted by the volume's (NR) and maxFinal's partial COUNT normalised --
    // it is a different number by construction (the volume is nz slices deep).
    if (!ADDED[page].length) continue;
    const body = s => s.slice(s.indexOf("@group(0)"));
    const norm = s => body(s).replace(/NRS/g, "NR").replace(/= \d+u;/g, "= Nu;");
    let bad2 = [], np = 0;
    for (const [inst, tmpl] of INSTANCES) {
      for (const k of Object.keys(cur)) {
        if (k.split(" :: ")[1] !== inst) continue;
        const t = k.replace(" :: " + inst, " :: " + tmpl);
        np++;
        if (norm(cur[k]) !== norm(cur[t])) bad2.push(k);
      }
    }
    ok("  ... and the volume-sized ones are the slice templates at NR", bad2.length === 0,
       bad2.join(", ") || np + " instantiations over " + INSTANCES.length + " templates");
  }
}

// ---------------------------------------------------------------------------
// 3. box-unit aspect (Phase A)
// ---------------------------------------------------------------------------
// the three projected box edges, in clip space, straight out of the cubeQuads array:
// face 0 (z = 1) runs u -> x and v -> y, face 1 (x = 1) runs u -> z (see cubeQuads)
const edges = q => ({ x: [q[2] - q[0], q[3] - q[1]], y: [q[4] - q[0], q[5] - q[1]],
                      z: [q[14] - q[12], q[15] - q[13]] });
const len = v => Math.hypot(v[0], v[1]);
// the half-extent of the drawn box in clip x and clip y (the corners are all 12 quad
// corners; entries 8..11 of each 12-float block are the shade + padding, not geometry)
function extent(q) {
  let mx = 0, my = 0;
  for (let f = 0; f < 3; f++) {
    for (let i = 0; i < 8; i += 2) {
      mx = Math.max(mx, Math.abs(q[12 * f + i]));
      my = Math.max(my, Math.abs(q[12 * f + i + 1]));
    }
  }
  return [mx, my];
}
// how the 12 quad corners of the three drawn faces map back to box FRACTIONS: face 0 is
// z = 1, face 1 is x = 1, face 2 is y = 1 (see cubeQuads), each running (u, v) = (a, b)
// over its own two axes, and the four corners sit at floats 0, 2, 4, 6 of its block.
// Used by leg 3 (cubeFrame) and by leg 5 (the ray uniform's inverse).
const CORNER = [(a, b) => [a, b, 1], (a, b) => [1, b, a], (a, b) => [b, 1, a]];
const QCORNER = [[0, 0, 0], [1, 0, 2], [0, 1, 4], [1, 1, 6]];
const READ = `function(){ return { Lz: solver.g.Lz, Lx: solver.g.Lx, Ly: solver.g.Ly,
  q: Array.from(cubeQuads()), F: cubeFrame(), T: cubeTopXform(), cap: ASPECT_CAP }; }`;
async function legAspect() {
  const env = await boot(dir, "rmhd3d.html");
  const opts = env.run(`function(){ return document.getElementById("selLz").options.map(o => o.value); }`);
  ok("selLz still offers 2pi..16pi", opts.join(",") === "2,4,8,16", opts.join(","));
  ok("ASPECT_CAP defaults to 0 (exact ratio, no display cap)",
     env.run("function(){ return ASPECT_CAP; }") === 0);
  // one constant, one edit: declared once and read in exactly one place (comments
  // stripped, or the prose explaining the knob would count as uses of it)
  const src = fs.readFileSync(path.join(dir, "rmhd3d.html"), "utf8");
  const code = src.split("\n").map(l => l.replace(/\/\/.*$/, "")).join("\n");
  ok("  ... and it is a single constant (declared exactly once)",
     (code.match(/const ASPECT_CAP =/g) || []).length === 1 &&
     /ASPECT_CAP > 0 \? Math\.min\(.*ASPECT_CAP\)/.test(code),
     (code.match(/const ASPECT_CAP =/g) || []).length + " declaration(s)");
  let ref = null;
  for (const v of opts) {
    // the REAL path: the select's own rebuild handler makes a new solver at that Lz
    env.run(`function(v){ const s = document.getElementById("selLz"); s.value = v; s.onchange(); }`, v);
    const g = env.run(READ), tag = v + "pi: ";
    const e = edges(g.q), rho = g.Lz / g.Lx;
    ok(tag + "the solver really rebuilt at Lz = " + v + "pi",
       rel(g.Lz, v * Math.PI) < 1e-12, "Lz = " + g.Lz.toFixed(4));
    if (!ref) ref = { zx: len(e.z) / len(e.x), yx: len(e.y) / len(e.x), rho };
    // the z:x edge-length ratio tracks Lz/Lx exactly (the projection's own foreshortening
    // divides out against the Lz = 2pi reference, so this is the box shape and nothing else)
    ok(tag + "the z:x edge ratio tracks Lz/Lx",
       rel((len(e.z) / len(e.x)) / ref.zx, rho / ref.rho) < 1e-5,
       "measured " + ((len(e.z) / len(e.x)) / ref.zx).toFixed(6) + ", want " + (rho / ref.rho).toFixed(6));
    ok(tag + "  ... while y:x is untouched (Ly = Lx)",
       rel(len(e.y) / len(e.x), ref.yx) < 1e-5 && rel(g.Ly, g.Lx) < 1e-12);
    // the 0.92 autoscale still fits the (now elongated) box: inside the square canvas,
    // and TIGHT against it on the axis that fills first
    const [mx, my] = extent(g.q);
    ok(tag + "the 0.92 autoscale still fits the box, tightly",
       mx < 0.92 + 1e-6 && my < 0.92 + 1e-6 && rel(Math.max(mx, my), 0.92) < 1e-5,
       "half-extent (" + mx.toFixed(4) + ", " + my.toFixed(4) + ")");
    // cubeFrame is DERIVED from the same array: it must reproduce all 12 face corners
    // (the field lines, the box wireframe and the arrow overlay ride on exactly this)
    const F = g.F, S = 512;
    const px = i => [(g.q[i] + 1) * 0.5 * S, (1 - g.q[i + 1]) * 0.5 * S];
    const at = (x, y, z) => [F.ox + x * F.ax + y * F.bx + z * F.cx,
                             F.oy + x * F.ay + y * F.by + z * F.cy];
    let df = 0;
    for (let f = 0; f < 3; f++) {
      for (const [a, b, i] of QCORNER) {
        const p = at(...CORNER[f](a, b)), qq = px(12 * f + i);
        df = Math.max(df, Math.abs(p[0] - qq[0]), Math.abs(p[1] - qq[1]));
      }
    }
    ok(tag + "cubeFrame reproduces all 12 projected corners", df < 1e-3,
       "max |delta| = " + df.toExponential(2) + " px");
    // ... and its z vector carries the same elongation (what drawBoxFrame / drawFieldLines
    // and, through cubeTopXform, the arrows are drawn with)
    ok(tag + "  ... and the frame's z:x vector ratio is the same elongation",
       rel(len([F.cx, F.cy]) / len([F.ax, F.ay]), len(e.z) / len(e.x)) < 1e-5);
    ok(tag + "  ... cubeTopXform is still the top face of that frame",
       Math.abs(g.T.ox - (F.ox + F.cx)) < 1e-9 && Math.abs(g.T.ax - F.ax) < 1e-9 &&
       Math.abs(g.T.by - F.by) < 1e-9);
  }
  ok("stub boot raised no failures", env.fails.length === 0, env.fails.join(" | "));
}

// ---------------------------------------------------------------------------
// 4. the display cap, as the on-device edit would make it
// ---------------------------------------------------------------------------
// ASPECT_CAP is meant to be a one-edit switch, so the check makes that edit on a COPY of
// the page and boots it: at 16pi the drawn box must stop at the cap and still fit.
async function legCap() {
  const d = path.join(tmp, "cap");
  fs.mkdirSync(d, { recursive: true });
  for (const f of ["common.js", "physics.js"]) fs.copyFileSync(path.join(dir, f), path.join(d, f));
  const src = fs.readFileSync(path.join(dir, "rmhd3d.html"), "utf8");
  fs.writeFileSync(path.join(d, "rmhd3d.html"), src.replace("const ASPECT_CAP = 0;", "const ASPECT_CAP = 6;"));
  const env = await boot(d, "rmhd3d.html");
  let ref = 0;
  for (const [v, want] of [["2", 1], ["8", 4], ["16", 6]]) {
    env.run(`function(v){ const s = document.getElementById("selLz"); s.value = v; s.onchange(); }`, v);
    const g = env.run(READ), e = edges(g.q);
    if (!ref) ref = len(e.z) / len(e.x);
    const [mx, my] = extent(g.q);
    ok("cap 6: at Lz = " + v + "pi the drawn ratio is " + want,
       rel((len(e.z) / len(e.x)) / ref, want) < 1e-5 && Math.max(mx, my) < 0.92 + 1e-6,
       "measured " + ((len(e.z) / len(e.x)) / ref).toFixed(6));
  }
  ok("cap 6: stub boot raised no failures", env.fails.length === 0, env.fails.join(" | "));
}

// ---------------------------------------------------------------------------
// 5-7. the volume view (Phase B)
// ---------------------------------------------------------------------------
// The raymarch is checked by RUNNING it: wgsl_reflect ships a WGSL interpreter, so the
// EMITTED fragment shader is executed (wrapped in a compute entry that calls `fs` at given
// clip points) and compared with a CPU march written here from the plan's description.
// The ray uniform is the page's OWN volRay output and the box is the page's own cubeQuads,
// so what is under test is the real pairing and not a re-implementation of either.
const PROBE_WGSL = `
@group(0) @binding(4) var<storage, read> px: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i: u32 = gid.x;
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  o.cp = vec2<f32>(px[2u * i], px[2u * i + 1u]);
  let c: vec4<f32> = fs(o);
  outp[3u * i] = c.x; outp[3u * i + 1u] = c.y; outp[3u * i + 2u] = c.z;
}`;
// the self-test grid's kernel: 16 x 16 x 8, i.e. a 2048-float volume the interpreter can
// march in milliseconds, and the same text every other grid emits (only NX/NY/NZ differ)
const VOLK = "selftest :: renderVol";
async function wgslMod() {
  const p = path.join(__dirname, "node_modules", "wgsl_reflect", "wgsl_reflect.module.js");
  if (!fs.existsSync(p)) return null;
  return await import(pathToFileURL(p).href);
}
const kconst = (src, n) => {
  const m = new RegExp("const " + n + ": (?:i32|u32|f32) = ([-+\\d.e]+)u?;").exec(src);
  return m ? parseFloat(m[1]) : NaN;
};
// the shader's knobs, read out of its own text: VOL_STEPS is the on-device fallback knob,
// so the reference march follows it instead of pinning it
const volKnobs = src => ({
  n: ["NX", "NY", "NZ"].map(s => kconst(src, s)),
  steps: kconst(src, "VOL_STEPS"), amax: kconst(src, "VOL_AMAX"),
  wfrac: kconst(src, "VOL_WFRAC"), amb: kconst(src, "VOL_AMB"),
  edge: kconst(src, "VOL_EDGE"), wireA: kconst(src, "VOL_WIRE_A")
});
function runVol(M, src, fld, mx, mode, cmapIdx, U, pix) {
  const ast = new M.WgslParser().parse(src + PROBE_WGSL);
  const out = new Float32Array(3 * pix.length);
  new M.WgslExec(ast).dispatchWorkgroups("probe", [pix.length, 1, 1], { 0: {
    0: fld, 1: new Float32Array([mx]), 2: new Uint32Array([mode, 0, cmapIdx, 0]),
    3: U, 4: Float32Array.from([].concat(...pix)), 5: out } });
  return pix.map((_, i) => [out[3 * i], out[3 * i + 1], out[3 * i + 2]]);
}
// dispX, the display shading's value -> [0,1] map (physics.js), in fp64
const dispXjs = (raw, s, mode) => {
  if (mode === 8 || mode === 9) return 0.5 * (Math.max(-1, Math.min(1, raw)) + 1);
  const v = raw / Math.max(s, 1e-30);
  if (mode >= 4 && mode <= 7) return v;
  return 0.5 * (Math.max(-1, Math.min(1, v)) + 1);
};
// trilinear value + the interpolant's three grid-coordinate derivatives, as the sum over
// the 8 corners -- deliberately NOT the shader's nested mix(), so the two agree only if
// both are right
function refSamp(fld, N, g) {
  const [NX, NY, NZ] = N, f = g.map(Math.floor), t = g.map((x, i) => x - f[i]);
  const wr = (i, n) => ((i % n) + n) % n;
  const ix = [wr(f[0], NX), wr(f[0] + 1, NX)], iy = [wr(f[1], NY), wr(f[1] + 1, NY)];
  const iz = [wr(f[2], NZ), wr(f[2] + 1, NZ)];
  const o = [0, 0, 0, 0];
  for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let c = 0; c < 2; c++) {
    const e = fld[iz[c] * NX * NY + ix[a] * NY + iy[b]];
    const wx = a ? t[0] : 1 - t[0], wy = b ? t[1] : 1 - t[1], wz = c ? t[2] : 1 - t[2];
    o[0] += e * wx * wy * wz;
    o[1] += e * (a ? 1 : -1) * wy * wz;
    o[2] += e * wx * (b ? 1 : -1) * wz;
    o[3] += e * wx * wy * (c ? 1 : -1);
  }
  return o;
}
// ... and the march itself: slab entry/exit, VOL_STEPS midpoint samples, two Gaussian
// shells at +-level*vmax (one for a magnitude mode), Lambert + ambient off the trilinear
// gradient in BOX-UNIT cells, front to back with the early exit, box-edge wireframe.
function refMarch(K, U, fld, mode, mx, cmap, cp) {
  const N = K.n, [NX, NY, NZ] = N;
  const ru = [U[0], U[1], U[2]], uv = [U[4], U[5], U[6]], dv = [U[8], U[9], U[10]];
  const cel = [U[12], U[13], U[14]], s = Math.max(mx, 1e-30);
  const p0 = [0, 1, 2].map(i => 0.5 + cp[0] * ru[i] + cp[1] * uv[i]);
  let t0 = -Infinity, t1 = Infinity;
  for (let i = 0; i < 3; i++) {
    const d = Math.abs(dv[i]) < 1e-12 ? 1e-12 : dv[i];
    const a = -p0[i] / d, b = (1 - p0[i]) / d;
    t0 = Math.max(t0, Math.min(a, b)); t1 = Math.min(t1, Math.max(a, b));
  }
  const acc = [0, 0, 0];
  if (!(t1 > t0)) return { c: acc, t0: t0, t1: t1, hit: false };
  const at = t => [0, 1, 2].map(i => p0[i] + t * dv[i]);
  const wire = [0.55, 0.62, 0.72];
  const ln = Math.hypot(0.3, 0.8, 0.52), lgt = [0.3 / ln, 0.8 / ln, 0.52 / ln];
  const edge = p => {
    let n = 0;
    for (let i = 0; i < 3; i++) if (Math.min(p[i], 1 - p[i]) * cel[i] * N[i] < K.edge) n++;
    return n >= 2;
  };
  let aa = 0;
  if (edge(at(t0))) { for (let c = 0; c < 3; c++) acc[c] = K.wireA * wire[c]; aa = K.wireA; }
  const lev = Math.max(U[16], 1e-4) * s, w = K.wfrac * lev;
  const two = !(mode >= 4 && mode <= 7);
  const cP = cmap(dispXjs(lev, s, mode)), cM = cmap(dispXjs(-lev, s, mode));
  const dt = (t1 - t0) / K.steps;
  for (let k = 0; k < K.steps; k++) {
    const p = at(t0 + (k + 0.5) * dt);
    const v = refSamp(fld, N, [p[0] * NX, p[1] * NY, p[2] * NZ]);
    const qp = (v[0] - lev) / w, qm = (v[0] + lev) / w;
    const gp = Math.exp(-qp * qp), gm = two ? Math.exp(-qm * qm) : 0;
    const dens = gp + gm;
    if (!(dens > 1e-3)) continue;
    const sg = gp >= gm ? -1 : 1;                 // the -lev shell faces the other way
    const gr = [sg * v[1] / cel[0], sg * v[2] / cel[1], sg * v[3] / cel[2]];
    const gl = Math.hypot(gr[0], gr[1], gr[2]);
    const nr = gl > 0 ? gr.map(x => x / Math.max(gl, 1e-30)) : lgt;
    const sh = K.amb + (1 - K.amb) *
               Math.max(nr[0] * lgt[0] + nr[1] * lgt[1] + nr[2] * lgt[2], 0);
    const a = 1 - Math.exp(-U[17] * dens * dt);
    for (let c = 0; c < 3; c++) acc[c] += (1 - aa) * a * sh * ((gp * cP[c] + gm * cM[c]) / dens);
    aa += (1 - aa) * a;
    if (aa > K.amax) break;
  }
  if (edge(at(t1))) for (let c = 0; c < 3; c++) acc[c] += (1 - aa) * K.wireA * wire[c];
  return { c: acc, t0: t0, t1: t1, hit: true };
}
// the synthetic volumes the two marches are compared on: an offset Gaussian blob (signed
// by `sgn`), plus the empty one -- 16 x 16 x 8, indexed iz * NX*NY + ix * NY + iy, which
// is the layout the display chain's dispR carries
function blob(N, sgn) {
  const [NX, NY, NZ] = N, f = new Float32Array(NX * NY * NZ);
  if (!sgn) return f;
  for (let iz = 0; iz < NZ; iz++) for (let ix = 0; ix < NX; ix++) for (let iy = 0; iy < NY; iy++) {
    const x = (ix + 0.5) / NX - 0.42, y = (iy + 0.5) / NY - 0.58, z = (iz + 0.5) / NZ - 0.5;
    f[iz * NX * NY + ix * NY + iy] = sgn * Math.exp(-(x * x + y * y + z * z) / 0.02);
  }
  return f;
}
// the box the legs march: the self-test grid in an ELONGATED box (Lz = 2 Lx), so a wrong
// box-unit metric anywhere -- ray, wireframe or shading normal -- shows up
const VOLG = `function(){ return { u: Array.from(volRay({ Lx: 2 * Math.PI, Ly: 2 * Math.PI,
  Lz: 4 * Math.PI }, 16, 16, 8)),
  q: Array.from(cubeQuads({ Lx: 2 * Math.PI, Ly: 2 * Math.PI, Lz: 4 * Math.PI })),
  lev: VOL_LEVEL, opac: VOL_OPAC }; }`;
// the clip point a box FRACTION projects to, by solving p - 0.5 = cx*ru + cy*uv + t*dv
// (Cramer) -- i.e. the ray uniform read forwards. Leg 5 asserts that this IS cubeQuads.
const det3 = (a, b, c) => a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                        + a[2] * (b[0] * c[1] - b[1] * c[0]);
function clipOf(U, p) {
  const ru = [U[0], U[1], U[2]], uv = [U[4], U[5], U[6]], dv = [U[8], U[9], U[10]];
  const D = det3(ru, uv, dv), v = p.map(x => x - 0.5);
  return [det3(v, uv, dv) / D, det3(ru, v, dv) / D, det3(ru, uv, v) / D];
}
// 2D convex hull (monotone chain, counter-clockwise) of the drawn box's projected corners,
// and a signed distance to it: the silhouette the ray's slab test has to reproduce
function hull(pts) {
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = src => {
    const h = [];
    for (const q of src) {
      while (h.length >= 2 && cr(h[h.length - 2], h[h.length - 1], q) <= 0) h.pop();
      h.push(q);
    }
    h.pop();
    return h;
  };
  const H = half(p).concat(half(p.slice().reverse()));
  let A = 0;
  for (let i = 0; i < H.length; i++) {
    const a = H[i], b = H[(i + 1) % H.length];
    A += a[0] * b[1] - b[0] * a[1];
  }
  return A < 0 ? H.reverse() : H;            // counter-clockwise, so inHull's sign is fixed
}
const inHull = (H, p) => {
  let d = Infinity;
  for (let i = 0; i < H.length; i++) {
    const a = H[i], b = H[(i + 1) % H.length];
    const e = [b[0] - a[0], b[1] - a[1]], L = Math.hypot(e[0], e[1]);
    d = Math.min(d, (e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) / L);
  }
  return d;                                  // > 0 inside, and it is a distance
};

// 5. the ray uniform IS the inverse of the projection the box is drawn with
async function legRay(state) {
  const env = await boot(dir, "rmhd3d.html");
  state.env3d = env;
  const G = env.run(VOLG), U = G.u, q = G.q;
  state.volU = U; state.volLev = [G.lev, G.opac];
  const ru = [U[0], U[1], U[2]], uv = [U[4], U[5], U[6]], dv = [U[8], U[9], U[10]];
  // every drawn corner, projected THROUGH THE RAY UNIFORM, against the clip point
  // cubeQuads actually draws it at
  const corners = [], pts = [];
  for (let f = 0; f < 3; f++) for (const [a, b, i] of QCORNER) {
    corners.push([CORNER[f](a, b), [q[12 * f + i], q[12 * f + i + 1]]]);
    pts.push([q[12 * f + i], q[12 * f + i + 1]]);
  }
  let dm = 0;
  for (const [p, c] of corners) {
    const s = clipOf(U, p);
    dm = Math.max(dm, Math.abs(s[0] - c[0]), Math.abs(s[1] - c[1]));
  }
  ok("the ray uniform inverts cubeQuads at all 12 drawn corners", dm < 1e-6,
     "max |delta clip| = " + dm.toExponential(2));
  // ... and the basis carries the BOX-UNIT metric: in box units (fraction * edge length)
  // the three vectors are orthogonal, the two screen ones share one scale (the projection
  // is orthographic and undistorted) and the depth one is a UNIT vector -- which is what
  // makes the march's t, and hence its step dt and its opacity, a true box-unit length
  const E = [U[12] * 16, U[13] * 16, U[14] * 8];
  const bu = v => v.map((x, i) => x * E[i]);
  const A = bu(ru), Bv = bu(uv), Cv = bu(dv);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  ok("  ... and in BOX UNITS that basis is orthogonal, unit-depth and undistorted",
     Math.abs(Math.hypot(...Cv) - 1) < 1e-6 && rel(Math.hypot(...A), Math.hypot(...Bv)) < 1e-6 &&
     Math.abs(dot(A, Bv)) < 1e-6 && Math.abs(dot(A, Cv)) < 1e-6 && Math.abs(dot(Bv, Cv)) < 1e-6,
     "|dv| = " + Math.hypot(...Cv).toFixed(9) + ", screen scale " + Math.hypot(...A).toFixed(6));
  // the marched box must be entered from the side the cube DRAWS: faces z = 1, x = 1,
  // y = 1 (CORNER above). Sampled over the whole canvas, every entry point sits on a max
  // face and every exit point on a min face -- i.e. dv points away from the camera.
  const H = hull(pts);
  let hits = 0, badFace = 0, badSil = 0, near = 0;
  const K = { n: [16, 16, 8], steps: 1, amax: 1, wfrac: 0.4, amb: 0.35, edge: 0, wireA: 0 };
  const zero = blob([16, 16, 8], 0), black = () => [0, 0, 0];
  for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) {
    const cp = [-1 + i / 20, -1 + j / 20];
    const m = refMarch(K, U, zero, 0, 1, black, cp);
    const d = inHull(H, cp);
    if (Math.abs(d) < 2e-3) { near++; continue; }      // knife edge: not this leg's point
    if (m.hit !== (d > 0)) badSil++;
    if (!m.hit) continue;
    hits++;
    const en = [0, 1, 2].map(k => 0.5 + cp[0] * U[k] + cp[1] * U[4 + k] + m.t0 * U[8 + k]);
    const ex = [0, 1, 2].map(k => 0.5 + cp[0] * U[k] + cp[1] * U[4 + k] + m.t1 * U[8 + k]);
    if (!en.some(v => Math.abs(v - 1) < 1e-5) || !ex.some(v => Math.abs(v) < 1e-5)) badFace++;
  }
  ok("the slab test's silhouette IS the drawn box's outline", badSil === 0 && hits > 200,
     hits + " pixels inside, " + badSil + " disagreements (" + near + " on the edge skipped)");
  ok("  ... entered through the three DRAWN faces, left through the far three",
     badFace === 0, badFace + " of " + hits + " rays entered from behind");
}

// 6. the emitted raymarch, executed, against the CPU reference
async function legMarch(state) {
  const M = await wgslMod();
  const src = (state.cur["rmhd3d.html"] || {})[VOLK];
  if (!M || !src) {
    ok("renderVol is emitted and wgsl_reflect is installed", false,
       M ? "no " + VOLK : "npm i wgsl_reflect in devtools/");
    return;
  }
  const env = state.env3d || (state.env3d = await boot(dir, "rmhd3d.html"));
  const K = volKnobs(src);
  ok("renderVol's knobs read back (steps, shell width, ambient, early exit)",
     [K.steps, K.amax, K.wfrac, K.amb, K.edge, K.wireA].every(v => isFinite(v)) &&
     K.n.join(",") === "16,16,8",
     K.steps + " steps, w = " + K.wfrac + "*level, alpha stops at " + K.amax);
  const U = Float32Array.from(state.volU);
  U[16] = state.volLev[0]; U[17] = state.volLev[1];   // the card's two knobs, as apply() writes them
  const fld = blob(K.n, 1), CM = 2;                   // RdBu: the diverging map the shells use
  const cmap = x => env.run("function(w, x){ return cmapRGB(w, x); }", CM, x);
  // pixels: the box centre, the ray straight through the blob, one off to its side, and
  // one aimed at the corner (1,1,1), which enters ON an edge and so carries the wireframe
  const cl = p => clipOf(U, p).slice(0, 2);
  const pix = [cl([0.5, 0.5, 0.5]), cl([0.42, 0.58, 0.5]), cl([0.3, 0.7, 0.8]), cl([1, 1, 1])];
  for (const [tag, mode] of [["jpar (signed, two shells)", 1], ["|u| (magnitude, one)", 4]]) {
    const got = runVol(M, src, fld, 1, mode, CM, U, pix);
    let em = 0, ink = 0;
    for (let i = 0; i < pix.length; i++) {
      const w = refMarch(K, U, fld, mode, 1, x => cmap(x), pix[i]).c;
      for (let c = 0; c < 3; c++) em = Math.max(em, Math.abs(got[i][c] - w[c]));
      ink = Math.max(ink, ...got[i]);
    }
    ok("the emitted march == the CPU reference, " + tag, em < 1e-5 && ink > 0.05,
       "max |delta| = " + em.toExponential(2) + " over " + pix.length +
       " pixels, brightest " + ink.toFixed(3));
  }
  // the wireframe pixel really is one (otherwise the agreement above proves less than it
  // looks), and an empty volume is all but black -- the shells are what draws
  const wire = refMarch(K, U, fld, 1, 1, () => [0, 0, 0], pix[3]);
  ok("  ... and the corner pixel is a wireframe pixel", Math.max(...wire.c) > 0.05,
     "wire ink " + Math.max(...wire.c).toFixed(3));
  const empty = runVol(M, src, blob(K.n, 0), 1, 1, CM, U, [pix[1]])[0];
  ok("  ... while an empty volume marches to (near) nothing", Math.max(...empty) < 0.15,
     "brightest channel " + Math.max(...empty).toFixed(3));
}

// 7. the omega+- entries, and the shells each field-table entry raises
async function legShells(state) {
  // omega+- = grad_perp^2 zeta+- = omega +- j: prepDisp's two new branches must be the
  // VORTICITY branch with the Elsasser potential phi +- psi in place of phi -- derived
  // from the emitted text of the mode-0 branch, so this cannot drift into a paraphrase.
  // Both apps, every grid: the table and the kernel are one shared block.
  const br = (s, m) => (new RegExp("md\\.mode == " + m + "u\\) \\{ v = (.*?); \\}").exec(s) || [])[1];
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    let n = 0, bad = [];
    for (const [k, s] of Object.entries(state.cur[page] || {})) {
      if (k.split(" :: ")[1] !== "prepDisp") continue;
      n++;
      const b0 = br(s, 0) || "", sub = t => b0.replace("fields[m]", t);
      if (br(s, 1) !== sub("fields[NM + m]") ||
          br(s, 10) !== sub("(fields[m] + fields[NM + m])") ||
          br(s, 11) !== sub("(fields[m] - fields[NM + m])")) bad.push(k);
    }
    ok(page + ": prepDisp's omega+- branches are the vorticity branch on phi +- psi",
       n > 0 && bad.length === 0, bad.join(", ") || n + " emissions");
    const t = fs.readFileSync(path.join(dir, page), "utf8");
    ok("  ... and both carry a field-table entry with its definition",
       /\{ v: 10, t: "Elsasser &omega;&#8314;",\s*\n\s*d: "&omega;&#8314; = &nabla;&sup2;&zeta;&#8314; = &omega; \+ j/.test(t) &&
       /\{ v: 11, t: "Elsasser &omega;&#8315;",\s*\n\s*d: "&omega;&#8315; = &nabla;&sup2;&zeta;&#8315; = &omega; &minus; j/.test(t));
  }
  const M = await wgslMod();
  const src = (state.cur["rmhd3d.html"] || {})[VOLK];
  if (!M || !src) { ok("shell selection: renderVol + wgsl_reflect available", false); return; }
  const env = state.env3d || (state.env3d = await boot(dir, "rmhd3d.html"));
  const K = volKnobs(src);
  const U = Float32Array.from(state.volU);
  U[16] = state.volLev[0]; U[17] = state.volLev[1];
  const fields = env.run(`function(){ return DISP_FIELDS.map(f => [f.v, dispIsVector(f.v),
    dispIsSigma(f.v)]); }`);
  const lum = c => (c[0] + c[1] + c[2]) / 3;
  const pix = [clipOf(U, [0.42, 0.58, 0.5]).slice(0, 2)];   // the ray through the blob
  const zero = blob(K.n, 0), pos = blob(K.n, 1), neg = blob(K.n, -1);
  let bad = [], n = 0;
  for (const [v, isVec, isSig] of fields) {
    if (isSig) continue;                     // never marched: see the page leg below
    n++;
    const L0 = lum(runVol(M, src, zero, 1, v, 2, U, pix)[0]);
    const Lp = lum(runVol(M, src, pos, 1, v, 2, U, pix)[0]);
    const Lm = lum(runVol(M, src, neg, 1, v, 2, U, pix)[0]);
    // a positive blob always raises a shell; a NEGATIVE one does so only where the field
    // is signed, which is exactly the field table's own dispIsVector
    const twoShell = Lm > L0 + 0.05;
    if (!(Lp > L0 + 0.05) || twoShell === isVec) bad.push(v + (isVec ? " (magnitude)" : " (signed)"));
  }
  ok("signed fields raise BOTH shells, magnitudes only the +level one",
     bad.length === 0, bad.length ? "wrong: " + bad.join(", ") : n + " field-table entries");
  // ... and the two sigma modes, which have no volume to march (four real volumes, a chain
  // holds two): the page resolves them to the cube faces, and says so in the caption
  const sg = env.run(`function(){ const c = cards.disp[0], out = [];
    c.selZSrc.value = "vol";
    for (const v of [8, 9, 1]) { c.selField.value = String(v); c.apply();
      out.push([v, solver.volOf(c.ci), solver.cubeOf(c.ci)]); }
    return out; }`);
  ok("  ... and the sigma modes fall back to the cube faces in the vol view",
     sg.every(([v, vo, cu]) => (v === 1 ? (vo === 1 && cu === 0) : (vo === 0 && cu === 1))),
     JSON.stringify(sg));
  ok("stub boot raised no failures (vol legs)", env.fails.length === 0, env.fails.join(" | "));
}

// ---------------------------------------------------------------------------
// 8. the collision preset (Phase C)
// ---------------------------------------------------------------------------
// Phase C is a VIEW change over an untouched initial condition, so the leg asserts both
// halves. First the page: booted at ?demo=collision, it must open on a volume of j with
// the preset's own level/opacity on that card, at Lz = 8pi, with the chi line alive --
// driven through the real preset machinery (presetBoot -> cardsLayout), not read off the
// registry. Then the IC itself: the packet code is compared FUNCTION BY FUNCTION against
// BASE, so "the packet IC stays" is checked and not asserted in a comment.
const IC_FNS = { "common.js": ["icPresetFields", "icGaussZ", "packetGeom", "chiEstimate"],
                 "rmhd3d.html": ["applyIC", "icInfoLine"] };
// ... plus the two one-liners that decide WHICH presets are packet presets and how long
// their packets are (the collision preset writes the second into its slider)
const IC_DECLS = ["const IC_SIGMA_Z_FRAC =", "const icIsPacketIC ="];
// the source of `function NAME(`, brace-matched from its declaration. Every brace in these
// six bodies -- comments and strings included -- is balanced, so the naive count is exact;
// an unbalanced one could only over-extract, which makes the comparison stricter.
function fnSrc(txt, name) {
  const at = txt.indexOf("\nfunction " + name + "(");
  if (at < 0) return null;
  let d = 0;
  for (let i = txt.indexOf("{", at); i < txt.length; i++) {
    if (txt[i] === "{") d++;
    else if (txt[i] === "}" && --d === 0) return txt.slice(at + 1, i + 1);
  }
  return null;
}
const declLine = (txt, pre) =>
  (txt.split("\n").find(l => l.indexOf(pre) === 0) || null);
async function legCollision() {
  // ---- the page, booted on the preset -------------------------------------
  const env = await boot(dir, "rmhd3d.html", "collision");
  const G = env.run(`function(){
    const L = PRESETS.collision.layout.disp;
    return { sel: document.getElementById("selPreset").value,
             ic: document.getElementById("selIC").value,
             Lz: solver.g.Lz, nz: solver.g.nz,
             zsrcKeys: L.map(s => Object.keys(s).sort().join("+")),
             chi: document.getElementById("icinfo").innerHTML,
             cards: cards.disp.map(c => [c.sel(), c.volView(), solver.volOf(c.ci),
                                         c.level(), c.opac(), c.rSlice.disabled]),
             charts: cards.chart.map(c => c.type()),
             defView: ZSRC_DEFAULT_CUBE }; }`);
  ok("?demo=collision selects the collision preset", G.sel === "collision", G.sel);
  ok("  ... on the packet IC, at Lz = 8pi (64^2 x 256)",
     G.ic === "letters" && rel(G.Lz, 8 * Math.PI) < 1e-12 && G.nz === 256,
     G.ic + ", Lz = " + G.Lz.toFixed(4) + ", nz = " + G.nz);
  // the LEAD card is a volume of j (mode 1) -- resolved by the solver, not just by the
  // select -- with the preset's two knobs on it and its plane slider dead
  const c0 = G.cards[0] || [];
  ok("the lead card is a VOLUME of j (mode 1), with its plane slider dead",
     c0[0] === 1 && c0[1] === true && c0[2] === 1 && c0[5] === true,
     JSON.stringify(c0));
  ok("  ... carrying the preset's level / opacity, not the defaults",
     c0[3] === 0.25 && c0[4] === 7, "level " + c0[3] + ", opacity " + c0[4]);
  // ... flanked by the two Elsasser vorticities, one packet each, also marched
  ok("  ... flanked by omega+ and omega- volume cards (one packet each)",
     G.cards.length === 3 && G.cards[1][0] === 10 && G.cards[2][0] === 11 &&
     G.cards.slice(1).every(c => c[1] === true && c[2] === 1 && c[3] === 0.3 && c[4] === 7),
     JSON.stringify(G.cards.slice(1)));
  // the ratified "vol is the default view": the preset sets level/opacity and NOTHING else
  // per card, so it never has to name a view
  ok("  ... and the preset names no view: vol is the default it inherits",
     G.defView === "vol" && G.zsrcKeys.join(" ") === "level+opac+sel level+opac+sel level+opac+sel",
     G.zsrcKeys.join(" "));
  ok("the chi readout is alive under the IC controls",
     /&chi;<sup>\+<\/sup> [\d.]/.test(G.chi) && /meeting at t = [\d.]/.test(G.chi),
     G.chi.replace(/<[^>]*>/g, "").slice(0, 96));
  ok("the energy + spectrum charts are still the preset's",
     G.charts.join(",") === "energy,spectrum", G.charts.join(","));
  ok("collision boot raised no failures", env.fails.length === 0, env.fails.join(" | "));
  // ---- and the IC code itself, against BASE --------------------------------
  const bd = baseDir();
  if (!bd) { ok("base " + BASE + " is readable (git show)", false, "git show failed"); return; }
  for (const [f, names] of Object.entries(IC_FNS)) {
    const cur = fs.readFileSync(path.join(dir, f), "utf8");
    const base = fs.readFileSync(path.join(bd, f), "utf8");
    const bad = [], miss = [];
    for (const n of names) {
      const a = fnSrc(cur, n), b = fnSrc(base, n);
      if (!a || !b) { miss.push(n); continue; }
      if (a !== b) bad.push(n);
    }
    if (f === "common.js") {
      for (const p of IC_DECLS) {
        const a = declLine(cur, p), b = declLine(base, p);
        if (a === null || b === null) miss.push(p); else if (a !== b) bad.push(p);
      }
    }
    ok(f + ": the packet IC is byte-identical to " + BASE,
       bad.length === 0 && miss.length === 0,
       bad.length ? "CHANGED: " + bad.join(", ")
                  : miss.length ? "not found: " + miss.join(", ")
                                : names.length + " functions unmoved");
  }
}

// ---------------------------------------------------------------------------
// 9. the per-scale k_perp filter (Phase D)
// ---------------------------------------------------------------------------
// The filter is ONE factor in prepDisp, so this leg RUNS the emitted prepDisp (wgsl_reflect's
// interpreter, as leg 6 runs the raymarch) and compares it with an fp64 mirror. Three things,
// in the plan's order:
//   (a) the factor is the half-cosine band, and BAND PARSEVAL: shell by shell, the filtered
//       display field's energy is the unfiltered field's with the taper applied -- shells
//       strictly inside the band passed EXACTLY, shells strictly outside EXACTLY zero, on
//       the spectrum chart's own bins and in its own k unit (which is the drift this
//       catches: a filter measuring k in anything else would be a lie on the axis);
//   (b) filter OFF is bitwise off, three ways -- the GATED emission is the base commit's
//       prepDisp byte for byte (the sigR pattern), the executed kernel's output is
//       bit-identical to the base kernel's on the same state, and the display uniforms the
//       page actually writes over a sweep of modes / views / colormaps are the base page's
//       first 16 bytes with two zero band words after them;
//   (c) no factor value outside [0, 1], monotone across each edge, and exactly 1 / exactly 0
//       in the pass and stop bands (the `t >= 1` test, not a cosine evaluated at its end).
const PREPK = "selftest :: prepDisp";
const BAND = [3, 7];                    // the band the leg filters with, in box wavenumbers
// the emitted factor, mirrored in fp64 (`e` = BAND_EDGE, read out of the kernel's own text)
const bandEdgeJs = t => (t >= 1 ? 1 : (t <= 0 ? 0 : 0.5 - 0.5 * Math.cos(Math.PI * t)));
const bandFacJs = (kn, lo, hi, e) => (lo > 0 ? bandEdgeJs((kn - lo + 0.5 * e) / e) : 1) *
                                     (hi > 0 ? bandEdgeJs((hi + 0.5 * e - kn) / e) : 1);
// the page's own self-test grid (the arrays dumpwgsl2 labels "selftest"), plus the emission
// with the filter GATED OFF: buildShaders derives the band's unit from the box k1, so a grid
// without one emits the pre-filter text -- the sigR gating pattern, exercised.
const gridSnip = is3d => `function(){
  const R = REFVEC;
  const P = ${is3d
    ? `{ nx: R.nx, ny: R.ny, nz: R.nz, Lx: R.Lx, Ly: R.Ly, Lz: R.Lz, diss: R.diss,
         hyper: R.hyper, zdiss: R.z_diss_k, fshell: R.fshell }`
    : `{ nx: R.nx, ny: R.ny, Lx: R.Lx, Ly: R.Ly, diss: R.diss, hyper: R.hyper, fshell: R.fshell }`};
  const g = Object.assign(${is3d ? "{ nx: P.nx, ny: P.ny, nz: P.nz }"
                                 : "{ nx: P.nx, ny: P.ny, Lx: P.Lx, Ly: P.Ly, pm: P.pm, eqsrc: P.eqsrc }"},
                          makeGrid(P));
  return { gridA: Array.from(g.gridA), gridB: Array.from(g.gridB), kunit: g.kunit,
           nb: ${is3d ? "nbins(P.nx, P.ny)" : "nbins(P.nx, P.ny, P.Lx, P.Ly)"},
           off: buildShaders(Object.assign({}, g, { kunit: 0 })).prepDisp }; }`;
// every Mode-uniform write, by CHAIN and buffer name, with its bytes (sigrcheck's tracer,
// widened to all chains: two cards showing two bands is the point of the control)
const UTRACE = `function(){
  const d = solver.device, orig = d.queue.writeBuffer.bind(d.queue);
  globalThis.UW = [];
  d.queue.writeBuffer = function (b, off, data) {
    let name = null;
    for (let ci = 0; ci < solver.disp.length; ci++) {
      const D = solver.disp[ci];
      if (!D || name) continue;
      for (const k of Object.keys(D.buf)) {
        const v = D.buf[k];
        if (v === b) name = ci + ":" + k;
        else if (Array.isArray(v) && v.indexOf(b) >= 0) name = ci + ":" + k + "[" + v.indexOf(b) + "]";
      }
    }
    if (name && name.indexOf(":mode") > 0) {   // any chain index width, mode* buffers only
      globalThis.UW.push([name, Array.from(new Uint8Array(data.buffer, data.byteOffset, data.byteLength))]);
    }
    return orig(b, off, data);
  };
}`;
// two cards, two bands: the second card is left wide open, and its uniform must say so
const TWOCARD = `function(){
  while (cards.disp.length < 2) addDisplayCard();
  cardsSync();
  const A = cards.disp[0], B = cards.disp[1];
  for (const c of [A, B]) if (c.selZSrc) c.selZSrc.value = "manual";   // an ordinary field view
  A.rBLo.value = "2"; A.rBHi.value = "6";
  B.rBLo.value = "0"; B.rBHi.value = String(B.rBHi.max);
  globalThis.UW = [];
  A.apply(); B.apply();
  const w = {};
  for (const u of globalThis.UW) if (!w[u[0]]) w[u[0]] = u[1];
  return { a: A.band(), b: B.band(), ca: A.cap.innerHTML, cb: B.cap.innerHTML, w: w }; }`;
// ... over every display mode, view and colormap BOTH commits offer (the omega+- modes and
// the vol view are Phase B's, so they are not in a sweep that has to run on base too)
const USWEEP = `function(modes, views){
  const c = cards.disp[0], out = [];
  for (const v of views) for (const md of modes) for (const cm of [0, 2]) {
    if (c.selZSrc) c.selZSrc.value = v;
    c.selField.value = String(md); c.selCmap.value = String(cm);
    if (c.rSlice) c.rSlice.value = "3";
    globalThis.UW = [];
    c.apply();
    for (const w of globalThis.UW) out.push([v + "/" + md + "/" + cm, w[0], w[1]]);
  }
  return out; }`;
// the emitted prepDisp, EXECUTED. wgsl_reflect's interpreter runs six invocations per
// workgroup rather than the 64 the kernel declares, so dispatch NM workgroups and let the
// kernel's own bounds test discard the surplus: over-dispatch, never under (and a mode the
// interpreter skipped would show up as a mismatch below, not as a silent pass).
function runPrep(M, src, N, fields, gridA, mode, band) {
  const outk = new Float32Array(2 * N.nm), outk2 = new Float32Array(2 * N.nm);
  const b = new ArrayBuffer(32), u = new Uint32Array(b), f = new Float32Array(b);
  u[0] = mode; f[4] = band[0]; f[5] = band[1];
  new M.WgslExec(new M.WgslParser().parse(src)).dispatchWorkgroups("main", [N.nm, 1, 1],
    { 0: { 0: fields, 1: gridA, 2: outk, 3: u, 4: outk2 } });
  return [outk, outk2];
}
// a deterministic (phi, psi) state: no RNG anywhere near this repo's checks
function synthState(nm) {
  const f = new Float32Array(4 * nm);
  let s = 12345;
  for (let i = 0; i < f.length; i++) { s = (s * 48271) % 2147483647; f[i] = s / 2147483647 - 0.5; }
  return f;
}
const bits = a => new Uint32Array(a.buffer, a.byteOffset, a.length);
// the lines `b` adds to `a`, or null if b is not a pure insertion into a (any line of a
// that moved or changed makes it null -- which is the point)
function inserted(a, b) {
  const A = a.split("\n"), add = [];
  let i = 0;
  for (const l of b.split("\n")) { if (i < A.length && A[i] === l) i++; else add.push(l); }
  return i === A.length ? add : null;
}
async function legFilter(state) {
  const M = await wgslMod();
  if (!M) { ok("wgsl_reflect is installed", false, "npm i wgsl_reflect in devtools/"); return; }
  const bd = state.baseDir || baseDir();
  if (!bd) { ok("base " + BASE + " is readable (git show)", false, "git show failed"); return; }
  for (const page of ["rmhd2d.html", "rmhd3d.html"]) {
    const is3d = page === "rmhd3d.html";
    const src = (state.cur[page] || {})[PREPK], bsrc = (state.base[page] || {})[PREPK];
    if (!src || !bsrc) { ok(page + ": prepDisp emitted at the self-test grid", false); continue; }
    const env = is3d ? (state.env3d || (state.env3d = await boot(dir, page))) : await boot(dir, page);
    const G = env.run(gridSnip(is3d));
    const N = { nm: kconst(src, "NM"), nmp: is3d ? kconst(src, "NMP") : kconst(src, "NM") };
    const E = kconst(src, "BAND_EDGE"), ku = G.kunit, nb = G.nb;
    const gA = Float32Array.from(G.gridA), gB = Float32Array.from(G.gridB);
    const [lo, hi] = BAND;

    // ---- the band's k unit is the spectrum chart's ---------------------------
    const sp = (state.cur[page] || {})["selftest :: spectrum"] || "";
    ok(page + ": the band is measured in the spectrum chart's own k unit",
       kconst(src, "INVKU") === kconst(sp, "INVKU") && rel(1 / kconst(src, "INVKU"), ku) < 1e-6,
       "1/INVKU = " + (1 / kconst(src, "INVKU")).toFixed(6) + ", kunit = " + ku.toFixed(6));

    // ---- (c) the factor itself, swept over k_perp ----------------------------
    // mode 2 is the plain phi branch, so with phi = 1 the kernel's output IS the factor. The
    // sweep rides a SYNTHETIC gridA (k_perp = kn * kunit), which is the only way to land
    // samples on the edges themselves.
    const nS = N.nmp, knAt = i => (i * (nb + 1)) / (nS - 1);
    const gs = new Float32Array(4 * nS), fs = new Float32Array(4 * N.nm);
    for (let i = 0; i < nS; i++) { const k = knAt(i) * ku; gs[4 * i + 2] = k * k; }
    for (let m = 0; m < N.nm; m++) fs[2 * m] = 1;
    const fac = runPrep(M, src, N, fs, gs, 2, BAND)[0];
    let em = 0, out01 = 0, mono = 0, notOne = 0, notZero = 0, prev = -1;
    for (let i = 0; i < nS; i++) {
      const kn = knAt(i), v = fac[2 * i];
      em = Math.max(em, Math.abs(v - bandFacJs(kn, lo, hi, E)));
      if (!(v >= 0 && v <= 1)) out01++;
      if (kn >= lo + 0.5 * E && kn <= hi - 0.5 * E && v !== 1) notOne++;
      if ((kn <= lo - 0.5 * E || kn >= hi + 0.5 * E) && v !== 0) notZero++;
      // monotone: up to the pass band it may only rise, past it only fall
      if (prev >= 0) {
        if (kn <= lo && v < prev - 1e-7) mono++;
        if (kn >= hi && v > prev + 1e-7) mono++;
      }
      prev = v;
    }
    ok(page + ": the emitted factor IS the fp64 half-cosine band", em < 1e-6,
       "max |delta| = " + em.toExponential(2) + " over " + nS + " samples of k_perp");
    ok("  ... in [0,1], monotone across each edge", out01 === 0 && mono === 0,
       out01 + " out of range, " + mono + " non-monotone steps");
    ok("  ... EXACTLY 1 in the pass band and EXACTLY 0 outside", notOne === 0 && notZero === 0,
       notOne + " passband != 1, " + notZero + " stopband != 0");

    // ---- (a) band Parseval, on the page's own grid ---------------------------
    // shells binned as the spectrum kernel bins them (b = round(|k|/kunit), weight
    // 0.5*ksq*yfac -- its INVN2 is a common factor), so these ARE its E_u bins.
    const st = synthState(N.nm);
    const raw = runPrep(M, src, N, st, gA, 2, [0, 0])[0];
    const flt = runPrep(M, src, N, st, gA, 2, BAND)[0];
    // ... out to the grid's LAST shell, not the spectrum's nb: past the dealias cut the
    // modes are dead in the solver but still present here, and lumping them into one end
    // bin would compare a filtered shell with an unfiltered one.
    let knMax = 0;
    for (let mp = 0; mp < N.nmp; mp++) knMax = Math.max(knMax, Math.sqrt(gA[4 * mp + 2]) / ku);
    const nbb = Math.round(knMax) + 1, Eraw = new Float64Array(nbb), Eflt = new Float64Array(nbb);
    const Emir = new Float64Array(nbb);
    for (let m = 0; m < N.nm; m++) {
      const mp = m % N.nmp, ksq = gA[4 * mp + 2], w = 0.5 * ksq * gB[4 * mp + 3];
      const kn = Math.sqrt(ksq) / ku, b = Math.round(kn);
      const t = bandFacJs(kn, lo, hi, E);
      const a0 = raw[2 * m], a1 = raw[2 * m + 1], f0 = flt[2 * m], f1 = flt[2 * m + 1];
      Eraw[b] += w * (a0 * a0 + a1 * a1);
      Emir[b] += w * t * t * (a0 * a0 + a1 * a1);
      Eflt[b] += w * (f0 * f0 + f1 * f1);
    }
    let inBad = 0, outBad = 0, worst = 0, live = 0, tot = 0, totM = 0;
    for (let b = 0; b < nbb; b++) {
      tot += Eflt[b]; totM += Emir[b];
      if (!(Eraw[b] > 0)) continue;
      live++;
      if (b >= lo + 1 && b <= hi - 1 && Eflt[b] !== Eraw[b]) inBad++;
      if ((b <= lo - 1 || b >= hi + 1) && Eflt[b] !== 0) outBad++;
      worst = Math.max(worst, rel(Eflt[b], Emir[b]));
    }
    ok(page + ": band Parseval -- every shell is its spectrum bin, tapered",
       worst < 1e-6 && rel(tot, totM) < 1e-6 && live > lo + 2,
       "max shell error " + worst.toExponential(2) + " over " + live + " live shells");
    ok("  ... shells inside the band pass EXACTLY, shells outside are EXACTLY zero",
       inBad === 0 && outBad === 0, inBad + " inside moved, " + outBad + " outside survived");

    // ---- (b) filter off is bitwise off ---------------------------------------
    // Gated off, prepDisp is the BASE text again -- except for Phase B's two omega+- branch
    // lines, which are in the same kernel and are not what this gate switches. So the leg
    // asserts the exact shape of the difference: a pure INSERTION into base's text, of those
    // four lines and nothing else. Anything the band left behind would show up here.
    const add = inserted(bsrc, "\n" + G.off + "\n");
    ok(page + ": gated off, prepDisp is " + BASE + " plus Phase B's four lines and nothing else",
       add !== null && add.length === 4 &&
       add.every(l => /omega\+-|Elsasser potential|md\.mode == 1[01]u/.test(l)),
       add === null ? "not an insertion into the base text: base lines moved"
                    : add.length + " inserted line(s)");
    let dbad = [];
    for (const md of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) {
      const a = runPrep(M, src, N, st, gA, md, [0, 0]), b = runPrep(M, bsrc, N, st, gA, md, [0, 0]);
      for (let h = 0; h < 2; h++) {
        const x = bits(a[h]), y = bits(b[h]);
        for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) { dbad.push(md + ":" + h + ":" + i); break; }
      }
    }
    ok("  ... and the EXECUTED kernel is bit-identical to " + BASE + "'s on the same state",
       dbad.length === 0, dbad.length ? "differ at " + dbad.slice(0, 3).join(" ") : "10 modes, both components");

    // the uniforms the page writes, against the same sweep on the base page
    const modes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      views = is3d ? ["manual", "zp", "zm", "cube", "cubezp", "cubezm", "lines"] : [null];
    env.run(UTRACE);
    const cur = env.run(USWEEP, modes, views);
    const benv = await boot(bd, page);
    benv.run(UTRACE);
    const bas = benv.run(USWEEP, modes, views);
    let ubad = [], zbad = 0;
    for (let i = 0; i < Math.max(cur.length, bas.length); i++) {
      const a = cur[i], b = bas[i];
      if (!a || !b || a[0] !== b[0] || a[1] !== b[1] || a[2].length !== 32 || b[2].length !== 16) {
        ubad.push(i + " " + (a ? a[0] + " " + a[1] + " " + a[2].length : "-"));
        continue;
      }
      for (let j = 0; j < 16; j++) if (a[2][j] !== b[2][j]) { ubad.push(a[0] + " " + a[1] + " byte " + j); break; }
      for (let j = 16; j < 32; j++) if (a[2][j] !== 0) zbad++;
    }
    ok("  ... and the display uniforms it writes are " + BASE + "'s, with two zero band words",
       ubad.length === 0 && zbad === 0 && cur.length === bas.length && cur.length > 0,
       ubad.length ? ubad.slice(0, 2).join(" | ") : cur.length + " writes swept, " + zbad + " non-zero band words");
    // ---- two cards, two bands, and a caption that owns up to it ---------------
    const T = env.run(TWOCARD);
    const bandOf = n => { const a = Uint8Array.from(T.w[n] || []);
                          return a.length === 32 ? Array.from(new Float32Array(a.buffer, 16, 2)) : null; };
    const b0 = bandOf("0:mode"), b1 = bandOf("1:mode"), m0 = bandOf("0:modeM"), c0 = bandOf("0:modeC[0]");
    ok(page + ": two cards carry two bands -- one filtered, one wide open",
       T.a.join(",") === "2,6" && T.b.join(",") === "0,0" &&
       b0 && b0.join(",") === "2,6" && b1 && b1.join(",") === "0,0",
       JSON.stringify({ a: T.a, b: T.b, u0: b0, u1: b1 }));
    ok("  ... on every uniform that card preps a field through (mate, contour potentials)",
       m0 && m0.join(",") === "2,6" && c0 && c0.join(",") === "2,6",
       JSON.stringify({ modeM: m0, modeC0: c0 }));
    ok("  ... and only the filtered card's caption says so",
       /display filter/.test(T.ca) && !/display filter/.test(T.cb),
       T.ca.replace(/<[^>]*>/g, "").slice(-48));
    ok("  ... and both boots raised no failures", env.fails.length === 0 && benv.fails.length === 0,
       env.fails.concat(benv.fails).join(" | "));
  }
}

// ---------------------------------------------------------------------------
const LEGS = [
  ["1. every kernel parses; names / duplication discipline", legDiscipline],
  ["2. physics WGSL byte-identical to " + BASE, legByteIdentical],
  ["3. box-unit aspect: cubeQuads / cubeFrame across every Lz (Phase A)", legAspect],
  ["4. the ASPECT_CAP display cap, as the one-line on-device edit", legCap],
  ["5. the volume ray: volRay inverts cubeQuads, and marches the drawn box (Phase B)", legRay],
  ["6. the EMITTED raymarch, executed, vs a CPU reference march (Phase B)", legMarch],
  ["7. the omega+- fields, and one shell or two per field-table entry (Phase B)", legShells],
  ["8. the collision preset: vol of j, and an untouched packet IC (Phase C)", legCollision],
  ["9. the k_perp display filter: band Parseval, bitwise-off, edge shape (Phase D)", legFilter]
];
(async () => {
  const state = { cur: {}, base: {} };
  for (const [title, fn] of LEGS) { console.log(title); await fn(state); }
  try { fs.rmSync(tmp, { recursive: true, force: true }); } catch (e) {}
  console.log(bad ? "\n" + bad + " FAILURE(S)" : "\nall checks passed");
  process.exit(bad ? 1 : 0);
})();
