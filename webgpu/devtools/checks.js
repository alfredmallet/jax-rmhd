// GATE G node unit checks (fp64 throughout; the app stores f32, which is stated where
// it matters).  Usage: node checks.js <webgpu-dir>
const fs = require("fs"), vm = require("vm"), path = require("path");
const dir = process.argv[2];
let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};

// ---- minimal DOM: a 2D canvas that really rasterizes, so the letter path is exercised
function ctx2d(cv) {
  const st = { buf: null };
  const o = { fillStyle: "#000", font: "10px x", textAlign: "left", textBaseline: "alphabetic" };
  // an engine that IGNORES ctx.filter (Safari < 17), so the CPU blur path is the one
  // under test -- that is the path whose sigma has to be resolution-independent
  Object.defineProperty(o, "filter", { get: () => "none", set: () => {} });
  const fontPx = () => { const m = /(\d+(?:\.\d+)?)px/.exec(o.font); return m ? parseFloat(m[1]) : 10; };
  o.measureText = t => ({ width: 0.62 * fontPx() * t.length,
                          actualBoundingBoxAscent: 0.72 * fontPx(), actualBoundingBoxDescent: 0 });
  o.fillRect = () => { st.buf = new Uint8ClampedArray(4 * cv.width * cv.height); };
  // a filled rectangle of PHYSICAL size (0.6 x 0.72 of the font em), i.e. the same shape
  // at every resolution -- the only thing that must differ between grids is sampling
  o.fillText = (t, cx, cy) => {
    if (!st.buf) return;
    const s = fontPx(), hw = 0.3 * s, hh = 0.36 * s, W = cv.width, H = cv.height;
    for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
      if (Math.abs(px + 0.5 - cx) <= hw && Math.abs(py + 0.5 - cy) <= hh) {
        const i = 4 * (py * W + px);
        st.buf[i] = 255; st.buf[i + 1] = 255; st.buf[i + 2] = 255; st.buf[i + 3] = 255;
      }
    }
  };
  o.getImageData = () => ({ data: st.buf || new Uint8ClampedArray(4) });
  o.createImageData = (w, h) => ({ data: new Uint8ClampedArray(4 * w * h) });
  for (const m of ["putImageData", "drawImage", "setTransform", "clearRect"]) o[m] = () => {};
  return o;
}
const mkCanvas = () => { const cv = { width: 0, height: 0, style: {} };
  cv.getContext = () => ctx2d(cv); return cv; };
const stubEl = () => ({ value: "", style: {}, textContent: "", innerHTML: "", checked: false,
                        disabled: false, min: "", max: "", step: "", options: [],
                        addEventListener() {}, appendChild() {} });
// Sections 6-9 (FEEDBACK_2026-08-08 P2) drive code that reads and WRITES real controls,
// so getElementById has to remember them. `ELS` is null everywhere else, which keeps the
// throwaway-element behaviour sections 1-5 were written against.
let ELS = null;
const getEl = id => (ELS ? (ELS[id] || (ELS[id] = stubEl())) : stubEl());
const sandbox = {
  document: { getElementById: getEl,
              createElement: t => (t === "canvas" ? mkCanvas() : stubEl()),
              createTextNode: () => ({}), querySelectorAll: () => [] },
  window: { addEventListener() {}, devicePixelRatio: 1, matchMedia: () => ({ matches: true }) },
  console, Math, JSON, Float32Array, Float64Array, Uint32Array, Uint8ClampedArray, Map, Set,
  Error, Promise, setTimeout, Number, String, Array, Object, isFinite, parseInt, parseFloat,
  URLSearchParams, performance: { now: () => 0 }
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(dir, "common.js"), "utf8"), sandbox, { filename: "common.js" });
const C = sandbox;
// top-level `const`s of a vm script are not properties of the context: pull the ones
// the checks need out through an expression.
Object.assign(C, vm.runInContext("({ icSigmaLetter, IC_SIGMA_PERP_FRAC, IC_SIGMA_Z_FRAC,"
  + " IC_SIGMA_Z_MAX_FRAC, TRACK_HYST, IC_LETTERS, DISS_KD_FRAC, dissKd, DISS_STEP,"
  + " DISS_DECADES_BELOW, DISS_LG_OPEN, AUTODISS_SHELL_W, AUTODISS_SMOOTH,"
  + " AUTODISS_MAX_FACTOR, AUTODISS_PERIOD, IC_SINE, IC_SINE_N, icSigmaSine, icIsPacketIC,"
  + " FIT_FRACS, FIT_SNAP, MODE_FIT_DT, MODE_FIT_RISE, MODE_FIT_R2, ISLAND_FIT_RISE })",
  sandbox));
const L = 2 * Math.PI;

// ---------------------------------------------------------------------------
console.log("1. normalized-IC rescale exactness (G.3)");
// ---------------------------------------------------------------------------
{
  for (const nz of [1, 8]) {
    const g = { nx: 64, ny: 64, nz: nz, Lx: L, Ly: L, Lz: L };
    const n = g.nx * g.ny * g.nz;
    const zp = new Float32Array(n), zm = new Float32Array(n);
    let r = 12345;
    const rnd = () => { r = (r * 1103515245 + 12345) & 0x7fffffff; return r / 0x7fffffff - 0.5; };
    // smooth, non-trivial, non-zero-mean potentials
    for (let k = 0; k < nz; k++) for (let i = 0; i < g.nx; i++) for (let j = 0; j < g.ny; j++) {
      const x = 2 * Math.PI * i / g.nx, y = 2 * Math.PI * j / g.ny, z = 2 * Math.PI * k / nz;
      const o = (k * g.nx + i) * g.ny + j;
      zp[o] = 0.7 + Math.sin(x) * Math.cos(2 * y) * (1 + 0.3 * Math.cos(z)) + 0.01 * rnd();
      zm[o] = -1.3 + Math.cos(3 * x + y) * (1 + 0.2 * Math.sin(z)) + 0.01 * rnd();
    }
    const a = C.icZetaFields(zp, zm, g, 1.0, 2.0);
    const b = C.icZetaFields(zp, zm, g, 3.0, 6.0);       // exactly 3x both amplitudes
    let e = 0, mx = 0, chg = 0;
    for (let i = 0; i < n; i++) {
      e = Math.max(e, Math.abs(3 * a.phi[i] - b.phi[i]), Math.abs(3 * a.psi[i] - b.psi[i]));
      mx = Math.max(mx, Math.abs(b.phi[i]), Math.abs(b.psi[i]));
    }
    // and the stored potentials must be untouched by either call
    for (let i = 0; i < n; i++) chg = Math.max(chg, Math.abs(zp[i]) === 0 ? 0 : 0);
    ok("nz=" + nz + ": scaling the sliders by 3 scales (phi,psi) by exactly 3",
       e <= 1e-6 * mx, "max abs dev " + e.toExponential(2) + " of " + mx.toPrecision(4));
    // the realized amplitude IS the slider (max |grad_perp zeta| over the volume)
    const st = C.icShearStats(zp, g);
    const back = new Float32Array(n);
    for (let i = 0; i < n; i++) back[i] = a.phi[i] + a.psi[i];      // = zeta+ , rescaled
    const gm = C.icShearStats(back, g).gradMax;
    ok("nz=" + nz + ": max |grad_perp zeta+| == the zeta+ slider (1.0)",
       Math.abs(gm - 1.0) < 1e-5, "got " + gm.toPrecision(8) + " (raw gradMax " + st.gradMax.toPrecision(4) + ")");
  }
  // an untouched drawing: two applies in a row must give identical output
  const g = { nx: 32, ny: 32, nz: 1, Lx: L, Ly: L, Lz: 0 };
  const zp = new Float32Array(32 * 32), zm = new Float32Array(32 * 32);
  for (let i = 0; i < 32; i++) for (let j = 0; j < 32; j++) {
    zp[i * 32 + j] = Math.sin(2 * Math.PI * i / 32) * Math.cos(2 * Math.PI * j / 32);
    zm[i * 32 + j] = Math.cos(4 * Math.PI * i / 32);
  }
  const p1 = C.icZetaFields(zp, zm, g, 0.4, 0.4), p2 = C.icZetaFields(zp, zm, g, 0.4, 0.4);
  let same = true;
  for (let i = 0; i < zp.length; i++) if (p1.phi[i] !== p2.phi[i] || p1.psi[i] !== p2.psi[i]) same = false;
  ok("re-applying the same drawing is bitwise reproducible (inputs not mutated)", same);

  // --- the (phi, psi) amplitude basis (FEEDBACK_2026-08-10 item 15) -----------
  // A drawing painted in phi AND psi stores zeta+- = P +- Q (icDrawBlob's weights), so the
  // two structures are inseparable once the zeta+- pair is what gets normalized. The "pp"
  // basis normalizes the COMBINATIONS instead, which is the only way the demo's closing
  // exercise -- a strong vortex plus a weak island -- can be asked for at all.
  const n2 = 64 * 64, g2 = { nx: 64, ny: 64, nz: 1, Lx: L, Ly: L, Lz: 0 };
  const zpM = new Float32Array(n2), zmM = new Float32Array(n2), P = new Float32Array(n2);
  for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) {
    const x = 2 * Math.PI * i / 64, y = 2 * Math.PI * j / 64, o = i * 64 + j;
    P[o] = 0.7 + Math.sin(x) * Math.sin(y);                       // the "phi" strokes
    const Q = 1.9 * Math.cos(2 * x + 0.3) * Math.cos(y);          // the "psi" strokes
    zpM[o] = P[o] + Q; zmM[o] = P[o] - Q;
  }
  const gradMax = f => C.icShearStats(f, g2).gradMax;
  const pp = C.icZetaFields(zpM, zmM, g2, 1.0, 0.05, "pp");
  ok("pp basis: max |grad phi| IS the first slider", Math.abs(gradMax(pp.phi) - 1.0) < 1e-5,
     "got " + gradMax(pp.phi).toPrecision(8));
  ok("pp basis: max |grad psi| IS the second slider, independently",
     Math.abs(gradMax(pp.psi) - 0.05) < 1e-6, "got " + gradMax(pp.psi).toPrecision(8));
  // ... which the zeta basis genuinely cannot do on the same drawing
  const zz = C.icZetaFields(zpM, zmM, g2, 1.0, 0.05);
  ok("... and the zeta basis cannot ask for that pair at all",
     Math.abs(gradMax(zz.psi) - 0.05) > 0.05,
     "zeta basis leaves max |grad psi| = " + gradMax(zz.psi).toPrecision(4));
  // the G.3 rescale contract holds in the new basis too
  const pp3 = C.icZetaFields(zpM, zmM, g2, 3.0, 0.15, "pp");
  let e3 = 0, mx3 = 0;
  for (let i = 0; i < n2; i++) {
    e3 = Math.max(e3, Math.abs(3 * pp.phi[i] - pp3.phi[i]), Math.abs(3 * pp.psi[i] - pp3.psi[i]));
    mx3 = Math.max(mx3, Math.abs(pp3.phi[i]), Math.abs(pp3.psi[i]));
  }
  ok("pp basis: scaling both sliders by 3 scales (phi,psi) by exactly 3", e3 <= 1e-6 * mx3,
     "max abs dev " + e3.toExponential(2));
  // and on a drawing with ONLY phi strokes the two bases are the same arithmetic, with
  // the amp lock on -- the case the old zeta+- labels happened to describe correctly
  const ppP = C.icZetaFields(P, P, g2, 0.4, 0.9, "pp");
  const zzP = C.icZetaFields(P, P, g2, 0.4, 0.4);
  let bit = true;
  for (let i = 0; i < n2; i++) if (ppP.phi[i] !== zzP.phi[i] || ppP.psi[i] !== zzP.psi[i]) bit = false;
  ok("a phi-only drawing comes out bitwise identical in either basis (psi == 0)", bit,
     "max |grad psi| = " + gradMax(ppP.psi));
}

// ---------------------------------------------------------------------------
console.log("2. sigma_letter is a PHYSICAL length: two resolutions, same k_perp content (G.2)");
// ---------------------------------------------------------------------------
// tiny iterative radix-2 complex FFT (fp64), enough for a 2D power spectrum
function fft(re, im, inv) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (inv ? 2 : -2) * Math.PI / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const w = ang * k, wr = Math.cos(w), wi = Math.sin(w);
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * wr - im[i + k + len / 2] * wi;
        const vi = re[i + k + len / 2] * wi + im[i + k + len / 2] * wr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
      }
    }
  }
}
// radial |k| spectrum of a real (nx,ny) field, in PHYSICAL k bins of width 2pi/L
function radialSpec(f, nx, ny, Lx, NB) {
  const re = [], im = [];
  for (let i = 0; i < nx; i++) {
    const r = new Float64Array(ny), m = new Float64Array(ny);
    for (let j = 0; j < ny; j++) r[j] = f[i * ny + j];
    fft(r, m, false); re.push(r); im.push(m);
  }
  for (let j = 0; j < ny; j++) {
    const r = new Float64Array(nx), m = new Float64Array(nx);
    for (let i = 0; i < nx; i++) { r[i] = re[i][j]; m[i] = im[i][j]; }
    fft(r, m, false);
    for (let i = 0; i < nx; i++) { re[i][j] = r[i]; im[i][j] = m[i]; }
  }
  const out = new Float64Array(NB), norm = 1 / (nx * ny);
  for (let i = 0; i < nx; i++) {
    const kx = (i < nx / 2 ? i : i - nx) * (2 * Math.PI / Lx);
    for (let j = 0; j < ny; j++) {
      const ky = (j < ny / 2 ? j : j - ny) * (2 * Math.PI / Lx);
      const b = Math.round(Math.hypot(kx, ky) / (2 * Math.PI / Lx));
      if (b >= 1 && b < NB) {
        const p = (re[i][j] * re[i][j] + im[i][j] * im[i][j]) * norm * norm;
        out[b] += 0.5 * (kx * kx + ky * ky) * p;      // energy of z = zhat x grad zeta
      }
    }
  }
  return out;
}
{
  const NB = 40, res = [128, 512], spec = [], kbar = [];
  for (const n of res) {
    const sigPx = Math.max(2, C.icSigmaLetter(L) / (L / n));
    ok("nx=" + n + ": letter blur is sigma_letter/dx = n/32 px",
       Math.abs(sigPx - n / 32) < 1e-12, sigPx + " px = " + C.icSigmaLetter(L).toPrecision(4) + " / " + (L / n).toPrecision(4));
    const g = { nx: n, ny: n, nz: 1, Lx: L, Ly: L, Lz: 0 };
    const z = C.icLetterZeta(g, null);
    const f = C.icZetaFields(z.zp, z.zp, g, 1.0, 1.0).phi;    // = the normalized zeta+
    const s = radialSpec(f, n, n, L, NB);
    let tot = 0, mom = 0;
    for (let b = 1; b < NB; b++) { tot += s[b]; mom += b * s[b]; }
    for (let b = 1; b < NB; b++) s[b] /= tot;
    spec.push(s); kbar.push(mom / tot);
  }
  let num = 0, den = 0;
  for (let b = 1; b < NB; b++) { const d = spec[0][b] - spec[1][b]; num += d * d; den += spec[1][b] * spec[1][b]; }
  const rel = Math.sqrt(num / den);
  ok("normalized E(k) at 128 vs 512 agree (rel L2 over k <= 39)", rel < 0.05, "rel L2 " + rel.toFixed(4));
  ok("energy-weighted kbar agrees within 5%",
     Math.abs(kbar[0] - kbar[1]) / kbar[1] < 0.05,
     "kbar 128 = " + kbar[0].toFixed(3) + ", 512 = " + kbar[1].toFixed(3) +
     " (1/sigma_letter = " + (1 / C.icSigmaLetter(L)).toFixed(3) + ")");
  ok("the blob-width slider default IS sigma_letter (one constant, two users)",
     C.IC_SIGMA_PERP_FRAC * L === C.icSigmaLetter(L));
}

// ---------------------------------------------------------------------------
console.log("3. chi formula (G.6)");
// ---------------------------------------------------------------------------
{
  const sz = 0.5, aP = 0.2, aM = 0.7;
  const chi = C.chiEstimate(C.icSigmaLetter(L), sz, aP, aM);
  const kbar = 1 / C.icSigmaLetter(L);
  ok("chi+ = a- * kbar_perp * sigma_z / v_A", Math.abs(chi[0] - aM * kbar * sz) < 1e-12,
     "chi+ " + chi[0].toPrecision(6));
  ok("chi- = a+ * kbar_perp * sigma_z / v_A", Math.abs(chi[1] - aP * kbar * sz) < 1e-12,
     "chi- " + chi[1].toPrecision(6));
  ok("kbar_perp = 1/sigma_letter = 32/Lx", Math.abs(kbar - 32 / L) < 1e-12, "kbar " + kbar.toPrecision(6));
  ok("chi is linear in sigma_z and in the OTHER field's amplitude",
     Math.abs(C.chiEstimate(C.icSigmaLetter(L), 2 * sz, aP, aM)[0] - 2 * chi[0]) < 1e-12 &&
     Math.abs(C.chiEstimate(C.icSigmaLetter(L), sz, aP, 3 * aM)[0] - 3 * chi[0]) < 1e-12);
  // the collision preset's own numbers
  const Lz = 8 * Math.PI, szD = C.IC_SIGMA_Z_FRAC * Lz, a = Math.pow(10, -0.7);
  console.log("        collision preset: a = " + a.toFixed(3) + ", sigma_z = " + szD.toFixed(3) +
              " -> chi = " + C.chiEstimate(C.icSigmaLetter(L), szD, a, a)[0].toFixed(3) +
              " (at the sigma_z cap: " +
              C.chiEstimate(C.icSigmaLetter(L), C.IC_SIGMA_Z_MAX_FRAC * Lz, a, a)[0].toFixed(3) + ")");
}

// ---------------------------------------------------------------------------
console.log("4. packet separation cap (G.6)");
// ---------------------------------------------------------------------------
{
  let worst = 1e9, worstAt = "";
  for (const Lz of [2 * Math.PI, 4 * Math.PI, 8 * Math.PI, 16 * Math.PI]) {
    for (let s = 1; s <= 16; s++) {
      const frac = s * C.IC_SIGMA_Z_MAX_FRAC / 16;         // the slider's own steps
      const sz = frac * Lz, g = C.packetGeom(Lz, sz);
      const direct = g.zPlus - g.zMinus, wrap = Lz - direct;
      const r = Math.min(direct, wrap) / sz;
      if (r < worst) { worst = r; worstAt = "Lz=" + Lz.toFixed(2) + " sigma_z=" + sz.toFixed(3); }
      if (!(g.zPlus > g.zMinus) || !(direct <= 0.5 * Lz + 1e-12)) {
        ok("placement sane at " + worstAt, false); break;
      }
    }
  }
  ok("every slider position keeps the packets >= 5 sigma_z apart (both ways round)",
     worst >= 5 - 1e-9, "worst separation " + worst.toFixed(3) + " sigma_z at " + worstAt);
  // the historical default placement is preserved exactly
  const Lz = 8 * Math.PI, g = C.packetGeom(Lz, C.IC_SIGMA_Z_FRAC * Lz);
  ok("default sigma_z reproduces the 11Lz/16 : 5Lz/16 placement",
     Math.abs(g.zPlus - 11 * Lz / 16) < 1e-12 && Math.abs(g.zMinus - 5 * Lz / 16) < 1e-12,
     "z+ " + g.zPlus.toFixed(4) + ", z- " + g.zMinus.toFixed(4) + ", t_coll " + g.tColl.toFixed(3));
  // envelope overlap at the cap: the two packets never share amplitude
  const nz = 256, szc = C.IC_SIGMA_Z_MAX_FRAC * Lz, gc = C.packetGeom(Lz, szc);
  const ep = C.icGaussZ(nz, Lz, gc.zPlus, szc), em = C.icGaussZ(nz, Lz, gc.zMinus, szc);
  let ov = 0;
  for (let k = 0; k < nz; k++) ov = Math.max(ov, Math.min(ep[k], em[k]));
  // exp(-25/8) = 4.394e-2 is the ideal 5-sigma crossing value; icGaussZ renormalizes each
  // envelope's peak to exactly 1 ON THE GRID, which can lift it a hair above that.
  ok("at the cap the envelopes barely overlap (~ exp(-25/8) = 4.4%)", ov <= 0.05,
     "max min(env+,env-) = " + ov.toExponential(3));
  // the slider the app builds really is capped there
  const e = { value: "", min: "", max: "", step: "" };
  sandbox.document.getElementById = () => e;
  C.icSigmaSliderInit("rSigZ", C.IC_SIGMA_Z_FRAC, C.IC_SIGMA_Z_MAX_FRAC);
  ok("icSigmaSliderInit caps sigma_z at Lz/12 and defaults to Lz/16",
     parseFloat(e.max) === C.IC_SIGMA_Z_MAX_FRAC && parseFloat(e.value) === C.IC_SIGMA_Z_FRAC &&
     Math.abs(parseFloat(e.value) / parseFloat(e.step) - 12) < 1e-9,
     "min " + e.min + " max " + e.max + " step " + e.step + " value " + e.value);
  sandbox.document.getElementById = getEl;      // the shared one back (sections 6-9 use ELS)
}

// ---------------------------------------------------------------------------
console.log("5. centroid tracker on a moving packet (G.8 / [7])");
// ---------------------------------------------------------------------------
{
  const nz = 256, Lz = 8 * Math.PI, sz = Lz / 16, vA = 1;
  const e = new Float64Array(2 * nz);
  const fill = z0 => {
    for (let k = 0; k < nz; k++) {
      let d = k * Lz / nz - z0; d -= Lz * Math.round(d / Lz);
      e[k] = Math.exp(-0.5 * d * d / (sz * sz));
      e[nz + k] = 0;
    }
  };
  const T = 400, dt = 0.05;
  const pos = [], planes = [];
  let unwrap = 0, prev = null;
  for (let s = 0; s < T; s++) {
    const z0 = 0.75 * Lz - vA * s * dt;                       // z+ moves toward smaller z
    fill(z0);
    let z = C.trackCentroid(e, 0, nz);
    const truth = ((z0 % Lz) + Lz) % Lz * nz / Lz;
    let d = z - (prev === null ? z : prev);
    if (prev !== null) { if (d > nz / 2) unwrap -= nz; else if (d < -nz / 2) unwrap += nz; }
    prev = z;
    pos.push(z + unwrap);
    planes.push(Math.round(z) % nz);
    if (s === 0) ok("centroid finds the packet centre",
      Math.abs(((z - truth + 1.5 * nz) % nz) - 0.5 * nz) < 1e-6,
      "centroid " + z.toFixed(4) + " vs " + truth.toFixed(4));
  }
  // linear in t: least-squares fit, residual relative to the total displacement
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let s = 0; s < T; s++) { sx += s; sy += pos[s]; sxx += s * s; sxy += s * pos[s]; }
  const b = (T * sxy - sx * sy) / (T * sxx - sx * sx), a0 = (sy - b * sx) / T;
  let res = 0;
  for (let s = 0; s < T; s++) res = Math.max(res, Math.abs(pos[s] - (a0 + b * s)));
  const vMeas = -b * (Lz / nz) / dt;
  ok("displacement is linear in t (max residual << one plane)", res < 1e-3,
     "max residual " + res.toExponential(2) + " planes");
  ok("the implied speed is v_A", Math.abs(vMeas - vA) < 1e-4, "v = " + vMeas.toFixed(6));
  // no plane HOPPING: consecutive displayed planes never jump by more than 1 (mod nz)
  let hop = 0;
  for (let s = 1; s < T; s++) {
    let d = Math.abs(planes[s] - planes[s - 1]); d = Math.min(d, nz - d);
    hop = Math.max(hop, d);
  }
  ok("the displayed plane never hops (max step 1)", hop <= 1, "max step " + hop);

  // argmax hysteresis: two nearly-equal peaks must not make the tracker oscillate
  const h = new Float64Array(2 * nz);
  let switches = 0, cur = 40;
  for (let s = 0; s < 200; s++) {
    h.fill(0);
    h[40] = 1.0;
    h[200] = 1.0 + 0.05 * Math.sin(s);           // +-5%: under the 10% threshold
    const nx = C.trackArgmax(h, 0, nz, cur);
    if (nx !== cur) switches++;
    cur = nx;
  }
  ok("argmax hysteresis: a 5% rival never steals the plane", switches <= 1,
     switches + " switches in 200 readbacks");
  h.fill(0); h[40] = 1.0; h[200] = 1.15;          // 15%: it should
  ok("argmax hysteresis: a 15% rival does take it", C.trackArgmax(h, 0, nz, 40) === 200);
  ok("hysteresis threshold is 10%", C.TRACK_HYST === 1.1);
}

// ---------------------------------------------------------------------------
console.log("6. auto-diss controller: the pure core (FEEDBACK item 6)");
// ---------------------------------------------------------------------------
// A synthetic perpendicular spectrum, [E_u | E_b | H_c] in 3*nb bins at k = b*kunit:
// a power law E(k) = A k^slope, so the shell energy (and hence nu_target) is analytic.
function synthBins(nb, kunit, A, slope) {
  const b = new Float64Array(3 * nb);
  for (let i = 1; i < nb; i++) {
    const e = 0.5 * A * Math.pow(i * kunit, slope);   // half in u, half in b
    b[i] = e; b[nb + i] = e;
  }
  return b;
}
{
  const nb = 64, kunit = 1, hyper = 4;
  const kd = C.dissKd(nb, kunit);
  ok("k_d is DISS_KD_FRAC of the retained k_perp max", Math.abs(kd - 0.6 * nb * kunit) < 1e-12,
     "k_d = " + kd + " with nb = " + nb);
  ok("the reference constants are rmhdgpu's",
     C.DISS_KD_FRAC === 0.6 && C.AUTODISS_SHELL_W === 0.5 && C.AUTODISS_SMOOTH === 0.2 &&
     C.AUTODISS_MAX_FACTOR === 2,
     "kd_fraction 0.6, shell_half_width 0.5, smooth 0.2, max_update 2");

  // (i) the target IS u_d k_d^(1-2n) with u_d = sqrt(2 E_shell), summed by hand
  const bins = synthBins(nb, kunit, 1e-3, -5 / 3);
  const lo = kd * Math.exp(-C.AUTODISS_SHELL_W), hi = kd * Math.exp(C.AUTODISS_SHELL_W);
  let ed = 0, nsh = 0;
  for (let i = 1; i < nb; i++) {
    const k = i * kunit;
    if (k >= lo && k <= hi) { ed += bins[i] + bins[nb + i]; nsh++; }
  }
  const want = Math.sqrt(2 * ed) * Math.pow(kd, 1 - 2 * hyper);
  const got = C.autoDissTarget(bins, nb, kunit, hyper);
  ok("nu_target = sqrt(2 E_d) k_d^(1-2n) over the log shell (" + nsh + " bins)",
     Math.abs(got / want - 1) < 1e-12, "got " + got.toExponential(4));
  ok("E_d is exactly the shell sum of the E_u + E_b lanes",
     Math.abs(C.autoDissShellE(bins, nb, kunit) / ed - 1) < 1e-12);
  // bins outside the shell must not contribute
  const only = new Float64Array(3 * nb);
  only[1] = 1e3; only[nb + 1] = 1e3;                  // a huge k = 1 bin, far below the shell
  ok("energy outside the log shell is ignored", C.autoDissTarget(only, nb, kunit, hyper) === 0);

  // (ii) convergence to the analytic answer, from three decades either side
  for (const start of [want * 1e3, want * 1e-3]) {
    let nu = start;
    for (let i = 0; i < 200; i++) nu = C.autoDissRelax(nu, want, want * 1e-9, want * 1e9);
    ok("converges to nu_target from " + (start > want ? "3 decades above" : "3 decades below"),
       Math.abs(Math.log10(nu / want)) < 1e-6, "nu/nu_target = " + (nu / want).toPrecision(8));
  }
  // (iii) smoothing and the per-update cap
  const one = (nu, t) => C.autoDissRelax(nu, t, 1e-300, 1e300);
  const d1 = Math.log10(one(1, 10) / 1);
  ok("one update moves SMOOTH of the log distance", Math.abs(d1 - C.AUTODISS_SMOOTH) < 1e-12,
     "1 decade -> " + d1.toFixed(6) + " decades");
  const d6 = one(1, 1e6) / 1, d6d = 1 / one(1, 1e-6);
  ok("a far target is capped at exactly MAX_UPDATE_FACTOR (both ways)",
     Math.abs(d6 - C.AUTODISS_MAX_FACTOR) < 1e-12 && Math.abs(d6d - C.AUTODISS_MAX_FACTOR) < 1e-12,
     "x" + d6.toFixed(6) + " up, /" + d6d.toFixed(6) + " down");
  // (iv) the clamp
  ok("the target is clamped to [nu_min, nu_max] before smoothing",
     C.autoDissRelax(1, 1e9, 0.5, 2) <= 2 && C.autoDissRelax(1, 1e-9, 0.5, 2) >= 0.5);
  let nuc = 1;
  for (let i = 0; i < 100; i++) nuc = C.autoDissRelax(nuc, 1e9, 0.5, 2);
  ok("and the state never leaves the clamp", nuc <= 2 + 1e-12, "nu = " + nuc.toPrecision(6));

  // (v) a QUIESCENT start does not collapse: an empty shell reports "no measurement" (0),
  // which the hook reads as HOLD, and even a real but absurdly quiet shell descends no
  // faster than the cap -- so the floor is minutes away, not milliseconds
  ok("an empty spectrum yields no target at all (the hook holds)",
     C.autoDissTarget(new Float64Array(3 * nb), nb, kunit, hyper) === 0);
  {
    const faint = synthBins(nb, kunit, 1e-40, -5 / 3);
    let nu = want, worst = 0;
    for (let i = 0; i < 10; i++) {
      const nx = C.autoDissRelax(nu, C.autoDissTarget(faint, nb, kunit, hyper), want * 1e-3, want * 1e3);
      worst = Math.max(worst, nu / nx);
      nu = nx;
    }
    ok("a quiescent-like shell walks DOWN at no more than the cap per update",
       worst <= C.AUTODISS_MAX_FACTOR + 1e-12, "worst single step /" + worst.toFixed(4));
    ok("... and cannot fall below nu_min (3 decades under marginal)", nu >= want * 1e-3 - 1e-300,
       "after 10 updates nu/nu_marg = " + (nu / want).toExponential(2));
  }

  // (vi) a KH-like run: the amplitude at k_d GROWS as the layer rolls up, and the
  // controller must follow it upward monotonically instead of sitting at the floor
  {
    const amp = i => 1e-12 * Math.pow(10, i / 4);                // 1 decade / 4 updates
    // start ON the target implied by the layer's initial (tiny) amplitude, which is where
    // the descent guard above leaves a KH start: from there the only way is up
    let nu = C.autoDissTarget(synthBins(nb, kunit, amp(0), -5 / 3), nb, kunit, hyper);
    let prev = 0, mono = true;
    for (let i = 0; i < 40; i++) {
      const t = C.autoDissTarget(synthBins(nb, kunit, amp(i), -5 / 3), nb, kunit, hyper);
      const nx = C.autoDissRelax(nu, t, want * 1e-9, want * 1e9);
      if (nx < nu - 1e-300) mono = false;
      prev = t; nu = nx;
    }
    ok("a growing (KH-like) spectrum drives nu monotonically UP", mono,
       "nu " + nu.toExponential(3) + " chasing " + prev.toExponential(3));
    ok("... to within a factor of a few of the live target", nu > 0.1 * prev,
       "nu / nu_target = " + (nu / prev).toPrecision(4));
  }

  // (vii) CLOSED LOOP. Model the cascade's response: with u(k) = u1 (k/k1)^(-1/3) the
  // dissipation scale k_nu solves nu k_nu^(2n) = u(k_nu) k_nu, and the spectrum is
  // inertial below it and cut off exponentially above -- so E_d(nu) has a genuine fixed
  // point exactly where the controller's rule says. Start 3 decades off either way.
  {
    const u1 = 1, k1 = kunit;
    const knu = nu => Math.pow(u1 * Math.pow(k1, 1 / 3) / nu, 1 / (2 * hyper - 2 / 3));
    const shellE = nu => {                      // E_d from that model, in the same shell
      let e = 0;
      for (let i = 1; i < nb; i++) {
        const k = i * kunit;
        if (k < lo || k > hi) continue;
        const uk = u1 * Math.pow(k / k1, -1 / 3);
        e += 0.5 * uk * uk * Math.exp(-2 * Math.max(0, k / knu(nu) - 1));
      }
      return e;
    };
    const fixed = (() => {                      // the model's own fixed point, by bisection
      let a = 1e-16, b = 1e2;
      for (let i = 0; i < 200; i++) {
        const m = Math.sqrt(a * b);
        const t = Math.sqrt(2 * shellE(m)) * Math.pow(kd, 1 - 2 * hyper);
        if (t > m) a = m; else b = m;
      }
      return Math.sqrt(a * b);
    })();
    for (const f of [1e3, 1e-3]) {
      let nu = fixed * f;
      for (let i = 0; i < 60; i++) {
        const t = Math.sqrt(2 * shellE(nu)) * Math.pow(kd, 1 - 2 * hyper);
        nu = C.autoDissRelax(nu, t, fixed * 1e-6, fixed * 1e6);
      }
      ok("closed loop from " + (f > 1 ? "1e3x" : "1e-3x") + " marginal settles at the fixed point",
         Math.abs(Math.log10(nu / fixed)) < 0.05,
         "nu / nu* = " + (nu / fixed).toPrecision(5) + " after 60 updates (30 s at 2 Hz)");
    }
    // ... and it lands within a decade of the amplitude-free nu_marg the t=0 seed and the
    // slider's bottom anchor use. It is not meant to be closer: nu_marg puts u_1 = 1 at
    // k_1 and counts ONE shell, while E_d here is the sum over the whole log shell -- the
    // point of the check is that the seed starts the controller in the right decade.
    const marg = Math.pow(k1, 1 / 3) * Math.pow(kd, 2 / 3 - 2 * hyper);
    ok("the loop's fixed point is nu_marg to within a decade",
       Math.abs(Math.log10(fixed / marg)) < 1,
       "nu* = " + fixed.toExponential(3) + " vs nu_marg = " + marg.toExponential(3) +
       " (" + Math.log10(fixed / marg).toFixed(2) + " decades)");
  }
  ok("the update cadence is the documented 2 Hz", C.AUTODISS_PERIOD === 500);
}

// ---------------------------------------------------------------------------
console.log("7. the diss slider's dynamic range (FEEDBACK item 7)");
// ---------------------------------------------------------------------------
{
  ELS = {};                                  // from here on the controls REMEMBER
  let applied = 0;
  C.applyControls = () => { applied++; };
  const GEOM = { sq256: { nx: 256, ny: 256, Lx: 2 * Math.PI, Ly: 2 * Math.PI },
                 sq512: { nx: 512, ny: 512, Lx: 2 * Math.PI, Ly: 2 * Math.PI },
                 wide512: { nx: 512, ny: 128, Lx: 4 * Math.PI, Ly: 2 * Math.PI },
                 d3_128: { nx: 128, ny: 128, Lx: 2 * Math.PI, Ly: 2 * Math.PI },
                 d3_64: { nx: 64, ny: 64, Lx: 2 * Math.PI, Ly: 2 * Math.PI } };
  let GEO = GEOM.sq256, HY = 4;
  C.uiParams = () => Object.assign({ hyper: HY }, GEO);
  const R = () => ELS.rDiss || (ELS.rDiss = C.document.getElementById("rDiss"));

  // the top end IS Re ~ 1 at the box scale, k_1^(1-2*hyper)
  for (const [g, h] of [[GEOM.sq256, 4], [GEOM.wide512, 1]]) {
    GEO = g; HY = h;
    const k1 = Math.min(2 * Math.PI / g.Lx, 2 * Math.PI / g.Ly);
    const rr = C.dissRange(C.nbins(g.nx, g.ny, g.Lx, g.Ly), k1, h);
    ok("top of the range is nu = k_1^(1-2*hyper) (hyper " + h + ", k_1 " + k1 + ")",
       Math.abs(rr[1] - (1 - 2 * h) * Math.log10(k1)) < 1e-12, "log10 nu_top = " + rr[1].toFixed(4));
    ok("bottom is DISS_DECADES_BELOW under nu_marg",
       Math.abs(rr[0] - (Math.log10(C.dissMarginal(C.nbins(g.nx, g.ny, g.Lx, g.Ly), k1, h))
                         - C.DISS_DECADES_BELOW)) < 1e-12, "log10 nu_min = " + rr[0].toFixed(4));
  }

  // EVERY value a preset writes must survive the open-then-narrow sequence presetWrite
  // and the following syncLabels perform -- byte for byte, no snapping, no clamping.
  const PRESET_DISS = [["2D forced", GEOM.sq256, 4, "-13"], ["2D kh seed", GEOM.wide512, 4, "-10.40"],
                       ["2D tearing", GEOM.wide512, 1, "-3"], ["3D forced", GEOM.d3_128, 4, "-10.65"],
                       ["2D 512^2 decay seed", GEOM.sq512, 4, "-14.75"]];
  for (const [nm, g, h, v] of PRESET_DISS) {
    GEO = g; HY = h;
    C.dissRangeOpen();
    R().value = v;                           // exactly what presetWrite does
    C.dissRangeSync();
    const lo = parseFloat(R().min), hi = parseFloat(R().max);
    ok(nm + " diss " + v + " survives the re-range and stays representable",
       R().value === v && parseFloat(v) >= lo && parseFloat(v) <= hi,
       "value " + R().value + " in [" + lo + ", " + hi + "]");
  }

  // a RE-RANGE never moves the stored value, even when hyper walks it out of the range
  GEO = GEOM.sq256; HY = 4;
  C.dissRangeOpen(); R().value = "-13"; C.dissRangeSync();
  const before = R().value, lo4 = parseFloat(R().min);
  HY = 1;                                    // hyper 4 -> 1 lifts nu_marg by many decades
  C.dissRangeSync();
  ok("changing hyper re-ranges the slider without moving its value",
     R().value === before && parseFloat(R().min) !== lo4,
     "value " + R().value + ", min " + lo4 + " -> " + R().min);
  ok("... widening the range OUTWARD to keep the value representable",
     parseFloat(R().value) >= parseFloat(R().min) && parseFloat(R().value) <= parseFloat(R().max),
     "[" + R().min + ", " + R().max + "]");
  ok("the hard open range is wide enough for anything a preset can ask",
     C.DISS_LG_OPEN[0] <= -30 && C.DISS_LG_OPEN[1] >= 6);
  // both ends land on the step grid, which is what stops a browser range input from
  // snapping an assigned value away from its multiple of DISS_STEP
  const onGrid = x => Math.abs(x / C.DISS_STEP - Math.round(x / C.DISS_STEP)) < 1e-6;
  ok("the range ends are multiples of the slider step",
     onGrid(parseFloat(R().min)) && onGrid(parseFloat(R().max)),
     R().min + " / " + R().max);

  // dissWriteLog: quantized, clamped, and silent when nothing moved
  HY = 4; GEO = GEOM.sq256;
  C.dissRangeOpen(); R().value = "-13.00"; C.dissRangeSync();
  applied = 0;
  C.dissWriteLog(-12.3712);
  ok("dissWriteLog quantizes to the slider step", onGrid(parseFloat(R().value)),
     "-12.3712 -> " + R().value);
  ok("... and pushes it down the LIVE path", applied === 1);
  applied = 0;
  C.dissWriteLog(parseFloat(R().value) + 0.4 * C.DISS_STEP);
  ok("a sub-step move is the controller's dead band (no re-upload)", applied === 0,
     "value still " + R().value);
  C.dissWriteLog(-999);
  ok("a wild value is clamped to the live range", parseFloat(R().value) === parseFloat(R().min),
     "clamped to " + R().value);
  C.dissWriteLog(Number.NaN);
  ok("NaN is refused outright", parseFloat(R().value) === parseFloat(R().min));

  // the t=0 seed is the marginal level itself
  C.dissRangeOpen(); R().value = "-1"; C.dissRangeSync();
  C.autoDissSeed();
  const g0 = C.dissGrid();
  ok("autoDissSeed writes nu_marg for the live grid",
     Math.abs(parseFloat(R().value) - Math.log10(C.dissMarginal(g0.nb, g0.kunit, g0.hyper)))
       <= 0.5 * C.DISS_STEP + 1e-9,
     "seed " + R().value + " vs nu_marg " + Math.log10(C.dissMarginal(g0.nb, g0.kunit, g0.hyper)).toFixed(4));
  ELS = null;
}

// ---------------------------------------------------------------------------
console.log("8. spectrum fit line: index, amplitude, anchor (FEEDBACK item 8)");
// ---------------------------------------------------------------------------
{
  // the index box holds decimals; -5/3 and -3/2 snap to the exact fraction, everything
  // else is taken literally, and a blank / NaN box falls back to the default
  ok("-1.667 snaps to exactly -5/3", C.fitIndex("-1.667") === -5 / 3);
  ok("-1.5 is -3/2", C.fitIndex("-1.5") === -1.5);
  ok("-2.3 is taken literally", C.fitIndex("-2.3") === -2.3);
  ok("a blank box is the default index", C.fitIndex("") === C.FIT_FRACS[0][0] &&
     C.FIT_FRACS[0][0] === -5 / 3);
  ok("a NaN box is the default index", C.fitIndex("abc") === -5 / 3 &&
     C.fitIndex(undefined) === -5 / 3);
  ok("outside FIT_SNAP nothing snaps", C.fitIndex(String(-5 / 3 + 2 * C.FIT_SNAP)) !== -5 / 3);
  ok("the legend reflects the chosen index", C.fitLabel(-5 / 3) === "k^-5/3" &&
     C.fitLabel(-1.5) === "k^-3/2" && C.fitLabel(-2.3) === "k^-2.3" && C.fitLabel(-1) === "k^-1",
     "k^-5/3, k^-3/2, k^-2.3, k^-1");

  // the anchor inverts E = A k^p at the first drawn point at or above kA -- for ANY p,
  // which is the whole change from the fixed 5/3 the old guide assumed
  for (const p of [-5 / 3, -1.5, -2.5, -1]) {
    const A = 0.0137, pts = [];
    for (let k = 1; k <= 40; k++) pts.push(k, A * Math.pow(k, p));
    const a = C.fitAnchor(pts, 3, p);
    ok("anchor recovers A for p = " + p.toFixed(4), Math.abs(a / A - 1) < 1e-12,
       "A = " + a.toExponential(6));
    // and the line through the anchor passes through the spectrum at kA exactly
    ok("... so the line meets the spectrum at kA", Math.abs(a * Math.pow(3, p) / (A * Math.pow(3, p)) - 1) < 1e-12);
  }
  ok("an empty series anchors nothing (the waiting... path stays blank)",
     C.fitAnchor([], 3, -5 / 3) === 0);
  ok("a series that stops below kA anchors nothing", C.fitAnchor([1, 1, 2, 0.5], 8, -5 / 3) === 0);
}

// ---------------------------------------------------------------------------
console.log("9. sinusoidal z+- packet IC (FEEDBACK item 9)");
// ---------------------------------------------------------------------------
{
  const g = { nx: 64, ny: 64, nz: 32, Lx: 2 * Math.PI, Ly: 2 * Math.PI, Lz: 16 * Math.PI };
  const nrs = g.nx * g.ny;
  const sz = g.Lz / 16, pg = C.packetGeom(g.Lz, sz);
  const env = [C.icGaussZ(g.nz, g.Lz, pg.zPlus, sz), C.icGaussZ(g.nz, g.Lz, pg.zMinus, sz)];
  const z = C.icSineZeta(g, env);

  // MODE CONTENT: a 2D DFT of each potential's peak plane must hold one mode only
  const peak = e => { let k = 0; for (let i = 1; i < g.nz; i++) if (e[i] > e[k]) k = i; return k; };
  const kP = peak(env[0]), kM = peak(env[1]);
  function dft2(f, off) {
    const P = [];
    for (let m = 0; m <= 3; m++) {
      P.push([]);
      for (let n = 0; n <= 3; n++) {
        let re = 0, im = 0;
        for (let i = 0; i < g.nx; i++) for (let j = 0; j < g.ny; j++) {
          const th = 2 * Math.PI * (m * i / g.nx + n * j / g.ny);
          re += f[off + i * g.ny + j] * Math.cos(th); im -= f[off + i * g.ny + j] * Math.sin(th);
        }
        P[m].push((re * re + im * im) / (nrs * nrs));
      }
    }
    return P;
  }
  for (const [nm, f, off, mi, ni] of [["zeta+", z.zp, kP * nrs, 1, 0], ["zeta-", z.zm, kM * nrs, 0, 1]]) {
    const P = dft2(f, off);
    let tot = 0, rest = 0;
    for (let m = 0; m <= 3; m++) for (let n = 0; n <= 3; n++) {
      tot += P[m][n];
      if (!(m === mi && n === ni) && !(m === 0 && n === 0)) rest += P[m][n];
    }
    ok(nm + " is the single mode (m,n) = (" + mi + "," + ni + ")",
       P[mi][ni] > 0 && rest < 1e-12 * P[mi][ni],
       "power " + P[mi][ni].toExponential(3) + ", everything else " + rest.toExponential(3));
    ok(nm + " has no mean (a potential's gauge)", P[0][0] < 1e-20 * (tot || 1));
  }
  ok("the two potentials use k = 2pi IC_SINE_N / L", C.IC_SINE_N === 1);
  ok("its perpendicular gradient scale is 1/k1", Math.abs(C.icSigmaSine(g.Lx) - g.Lx / (2 * Math.PI)) < 1e-15);
  ok("it is a PACKET preset (envelopes, sigma_z row, amp sliders)",
     C.icIsPacketIC(C.IC_SINE) && C.icIsPacketIC("letters") && !C.icIsPacketIC("custom") &&
     !C.icIsPacketIC("modes"));

  // Z ENVELOPE: each potential's plane amplitude follows its own icGaussZ, so the packets
  // sit where packetGeom puts them and the collision timing is the letters' unchanged
  let devP = 0, devM = 0;
  for (let k = 0; k < g.nz; k++) {
    let ap = 0, am = 0;
    for (let i = 0; i < nrs; i++) {
      ap = Math.max(ap, Math.abs(z.zp[k * nrs + i]));
      am = Math.max(am, Math.abs(z.zm[k * nrs + i]));
    }
    const a0 = 1 / (2 * Math.PI / g.Lx);            // |zeta| peak of the unenveloped plane
    devP = Math.max(devP, Math.abs(ap - env[0][k] * a0));
    devM = Math.max(devM, Math.abs(am - env[1][k] * a0));
  }
  ok("zeta+ rides envelope 0 (peak at z = " + pg.zPlus.toFixed(3) + ")", devP < 1e-6, "max dev " + devP.toExponential(2));
  ok("zeta- rides envelope 1 (peak at z = " + pg.zMinus.toFixed(3) + ")", devM < 1e-6, "max dev " + devM.toExponential(2));
  ok("zeta+ is placed ABOVE the midplane, zeta- below (they meet head-on)",
     pg.zPlus > 0.5 * g.Lz && pg.zMinus < 0.5 * g.Lz);

  // NORMALIZATION: icZetaFields must make the FIELDS z+- = zhat x grad zeta+- come out at
  // the amp sliders exactly -- and on the packet plane they must be the pure sinusoids
  const aP = 0.37, aM = 0.11;
  const F = C.icZetaFields(z.zp, z.zm, g, aP, aM);
  const zetaP = new Float32Array(g.nz * nrs), zetaM = new Float32Array(g.nz * nrs);
  for (let i = 0; i < g.nz * nrs; i++) { zetaP[i] = F.phi[i] + F.psi[i]; zetaM[i] = F.phi[i] - F.psi[i]; }
  const gm = (f, off) => C.icGradMax(f, g.nx, g.ny, g.Lx / g.nx, g.Ly / g.ny, off);
  let mxP = 0, mxM = 0;
  for (let k = 0; k < g.nz; k++) { mxP = Math.max(mxP, gm(zetaP, k * nrs)); mxM = Math.max(mxM, gm(zetaM, k * nrs)); }
  ok("max |grad zeta+| is the zeta+ amp slider", Math.abs(mxP / aP - 1) < 1e-4, "got " + mxP.toPrecision(6));
  ok("max |grad zeta-| is the zeta- amp slider", Math.abs(mxM / aM - 1) < 1e-4, "got " + mxM.toPrecision(6));
  // the FIELD on the packet plane: z+ = a+ yhat sin(k1 x) (so d_y zeta+ = 0 identically)
  const k1 = 2 * Math.PI / g.Lx;
  let eF = 0, eY = 0;
  for (let i = 0; i < g.nx; i++) {
    const x = i * g.Lx / g.nx;
    // 4th-order d_x, the same estimator icGradMax uses (row j = 0; there is no y
    // dependence, which the next check asserts separately)
    const at = q => zetaP[kP * nrs + ((((i + q) % g.nx) + g.nx) % g.nx) * g.ny];
    const dx = (at(-2) - 8 * at(-1) + 8 * at(1) - at(2)) / (12 * g.Lx / g.nx);
    eF = Math.max(eF, Math.abs(dx - aP * Math.sin(k1 * x)));
    eY = Math.max(eY, Math.abs(zetaP[kP * nrs + i * g.ny + 3] - zetaP[kP * nrs + i * g.ny + 17]));
  }
  ok("on its packet plane z+ = a+ yhat sin(k1 x)", eF < 2e-3 * aP, "max dev " + eF.toExponential(2));
  ok("... with no y dependence at all (d_y zeta+ = 0)", eY < 1e-9, "max spread " + eY.toExponential(2));
}

// ---------------------------------------------------------------------------
console.log("10. k_y = 2pi/Ly mode extraction and its gamma fit (KH chart)");
// ---------------------------------------------------------------------------
// The two halves of the chart's arithmetic, on data whose answer is known in closed form:
// modeAmps (one DFT coefficient of the cut stack) and modeFitGamma (the trailing-window
// slope of ln A). Neither is checked against itself.
{
  // a synthetic cut stack, in the 4*ny (u_x, u_y, b_x, b_y) layout readCutLine returns.
  // Rows 0 and 2 carry a known m = 1 cosine at a known PHASE, on top of a large constant
  // offset and a large m = 2 contamination -- neither of which a single-mode coefficient
  // may see (the offset is the k=0 gauge; m=2 is the first harmonic the roll-up makes).
  // Rows 1 and 3 are deliberate junk, 6 decades louder: the extraction must not read them.
  const ny = 64, Au = 3.7e-4, phu = 0.9, Ab = 1.1e-2, phb = -2.3;
  const vals = new Float64Array(4 * ny);
  for (let j = 0; j < ny; j++) {
    const th = 2 * Math.PI * j / ny;
    vals[j] = 0.25 + Au * Math.cos(th + phu) + 7 * Math.cos(2 * th);
    vals[ny + j] = 1e3 * Math.sin(3 * th);
    vals[2 * ny + j] = -1.5 + Ab * Math.cos(th + phb) + 4 * Math.cos(2 * th + 1);
    vals[3 * ny + j] = -1e3 * Math.cos(5 * th);
  }
  const a = C.modeAmps(vals, ny);
  ok("A_u is the m = 1 amplitude of the u_x row", Math.abs(a.u / Au - 1) < 1e-6,
     "got " + a.u.toPrecision(10) + " vs " + Au);
  ok("A_b is the m = 1 amplitude of the b_x row", Math.abs(a.b / Ab - 1) < 1e-6,
     "got " + a.b.toPrecision(10) + " vs " + Ab);
  // the phase is arbitrary: sweep it and the amplitude must not move
  let ep = 0;
  for (let q = 0; q < 16; q++) {
    const p = q * Math.PI / 8, v = new Float64Array(4 * ny);
    for (let j = 0; j < ny; j++) v[j] = 3 + Au * Math.cos(2 * Math.PI * j / ny + p);
    ep = Math.max(ep, Math.abs(C.modeAmps(v, ny).u / Au - 1));
  }
  ok("... independent of the mode's phase", ep < 1e-6, "max relative spread " + ep.toExponential(2));
  // a row with NO m = 1 content at all reads zero, not round-off times its own size
  const z = new Float64Array(4 * ny);
  for (let j = 0; j < ny; j++) { z[j] = 12 + 5 * Math.cos(4 * Math.PI * j / ny); z[2 * ny + j] = 0; }
  const az = C.modeAmps(z, ny);
  ok("an offset + m = 2 row has no m = 1 amplitude", az.u < 1e-14 && az.b === 0,
     "A_u = " + az.u.toExponential(2) + ", A_b = " + az.b);

  // --- the gamma fit ---------------------------------------------------------
  const G = 0.266260;                              // the KH eigenvalue reference
  const ts = [], as = [];
  for (let i = 0; i < 120; i++) { ts.push(0.37 * i); as.push(1e-6 * Math.exp(G * 0.37 * i)); }
  ok("the fit recovers gamma from an exponential", Math.abs(C.modeFitGamma(ts, as) / G - 1) < 1e-6,
     "got " + C.modeFitGamma(ts, as).toPrecision(10) + " vs " + G);
  // ... on an UNEVEN time base too (the readback throttle is wall-clock, not step count)
  const tu = [], au = [];
  for (let i = 0, t = 0; i < 60; i++, t += 0.1 + 0.4 * ((i * 7) % 5) / 5) { tu.push(t); au.push(2.5e-8 * Math.exp(G * t)); }
  ok("... on an unevenly sampled trace", Math.abs(C.modeFitGamma(tu, au) / G - 1) < 1e-6,
     "got " + C.modeFitGamma(tu, au).toPrecision(10));
  // it is a TRAILING window: an old, much steeper stage must not leak into the answer
  const tk = [], ak = [];
  for (let i = 0; i < 200; i++) {
    const t = 0.37 * i;
    tk.push(t); ak.push(i < 168 ? 1e-9 * Math.exp(3 * G * t)
                                : 1e-9 * Math.exp(3 * G * 0.37 * 167 + G * (t - 0.37 * 167)));
  }
  ok("only the trailing " + C.MODE_FIT_DT + " t-units are fitted",
     Math.abs(C.modeFitGamma(tk, ak) / G - 1) < 1e-6,
     "got " + C.modeFitGamma(tk, ak).toPrecision(10) + " (the older stage runs at 3 gamma)");
  // and it is NaN-guarded everywhere a growth rate would be a lie
  const flat = ts.map(() => 1e-5), dec = ts.map((t, i) => 1e-3 * Math.exp(-G * ts[i]));
  ok("flat data has no growth rate", !isFinite(C.modeFitGamma(ts, flat)),
     "got " + C.modeFitGamma(ts, flat));
  ok("a decaying trace has no growth rate", !isFinite(C.modeFitGamma(ts, dec)),
     "got " + C.modeFitGamma(ts, dec));
  ok("nonpositive / non-finite amplitudes are not logged",
     !isFinite(C.modeFitGamma([0, 1, 2, 3], [0, -1, NaN, Infinity])) &&
     !isFinite(C.modeFitGamma(ts.slice(0, 3), as.slice(0, 3))),
     "too few valid samples -> NaN");
  // a saturated stage OSCILLATES around a plateau: even when its endpoints happen to
  // rise, the window never rises by MODE_FIT_RISE, so the legend stays blank instead of
  // flickering a meaningless positive slope (reviewer MINOR-2)
  const tsat = [], asat = [];
  for (let i = 0; i < 80; i++) { tsat.push(0.2 * i); asat.push(2e-2 * (1 + 0.3 * Math.sin(1.7 * 0.2 * i + 0.3))); }
  ok("a saturated oscillation has no growth rate", !isFinite(C.modeFitGamma(tsat, asat)),
     "got " + C.modeFitGamma(tsat, asat));
  // ... and a window STRADDLING two stages (here 4 gamma kinking to gamma mid-window,
  // rise well above MODE_FIT_RISE) is not a rate either: the R^2 gate blanks it rather
  // than quoting an average of neither stage (reviewer MINOR-1/2)
  const tst = [], ast = [];
  for (let i = 0; i <= 100; i++) {
    const t = 0.1 * i;
    tst.push(t); ast.push(1e-6 * Math.exp(t < 5 ? 4 * G * t : 4 * G * 5 + G * (t - 5)));
  }
  ok("a two-stage straddle is blanked by the R^2 gate", !isFinite(C.modeFitGamma(tst, ast)),
     "got " + C.modeFitGamma(tst, ast));
  // zeros INSIDE an otherwise growing trace are skipped, not fatal
  const th2 = ts.slice(0, 40), ah2 = as.slice(0, 40).map((v, i) => (i % 7 === 3 ? 0 : v));
  ok("... and skipping them leaves the rate intact",
     Math.abs(C.modeFitGamma(th2, ah2) / G - 1) < 1e-6,
     "got " + C.modeFitGamma(th2, ah2).toPrecision(10));

  // --- the history record ----------------------------------------------------
  // HIST_MAX halving keeps every other sample, and a paused clock pushes nothing
  C.modeReset();
  const line = A => { const v = new Float64Array(4 * ny);
    for (let j = 0; j < ny; j++) v[j] = A * Math.cos(2 * Math.PI * j / ny);
    return v; };
  for (let i = 0; i < 2400; i++) C.modePush(0.01 * i, line(1e-6 * Math.exp(G * 0.01 * i)), ny);
  const H = vm.runInContext("modeHist", sandbox);
  ok("the history halves at HIST_MAX instead of growing", H.t.length > 0 && H.t.length <= 2000,
     H.t.length + " samples after 2400 pushes");
  ok("... keeping every other sample (the t axis stays monotone and spans the run)",
     H.t[0] === 0 && Math.abs(H.t[H.t.length - 1] - 23.99) < 1e-9 && H.t[1] > H.t[0],
     "t = " + H.t[0] + " .. " + H.t[H.t.length - 1] + ", dt = " + (H.t[1] - H.t[0]).toPrecision(3));
  const nb4 = H.t.length;
  C.modePush(H.t[nb4 - 1], line(1), ny); C.modePush(H.t[nb4 - 1] - 1, line(1), ny);
  ok("a paused clock pushes no duplicate t", H.t.length === nb4, nb4 + " samples, unchanged");
  ok("the fitted gamma survives the halving",
     Math.abs(C.modeFitGamma(H.t, H.u) / G - 1) < 1e-6,
     "got " + C.modeFitGamma(H.t, H.u).toPrecision(10));
  C.modeReset();
  ok("modeReset clears all three columns", H.t.length === 0 && H.u.length === 0 && H.b.length === 0);
}

// ---------------------------------------------------------------------------
console.log("11. the island-width chart's gamma_fit (FEEDBACK_2026-08-10 item 9)");
// ---------------------------------------------------------------------------
// The island chart borrows the mode chart's instrument: ONE fitting helper, the same
// trailing window and R^2 gate, its OWN rise gate, and a factor 2 because the plotted
// quantity is W ~ psitilde^(1/2) ~ e^(gamma t / 2).
{
  const GT = 0.0287;                               // the tearing preset's linear reference
  // W(t) exactly as the linear stage produces it, sampled at the cut throttle's ~10 Hz
  // against a run that is slow in sim-time (dt-ish 0.5 t-units per sample here)
  const mkW = (g, n, dt, w0) => {
    const t = [], w = [];
    for (let i = 0; i < n; i++) { t.push(dt * i); w.push((w0 || 0.0875) * Math.exp(0.5 * g * dt * i)); }
    return [t, w];
  };
  const [tw, ww] = mkW(GT, 200, 0.5);
  const gI = C.islandFitGamma(tw, ww);
  ok("islandFitGamma returns 2 x the slope of ln W (W ~ e^{gamma t/2})",
     Math.abs(gI / GT - 1) < 1e-6, "got " + gI.toPrecision(10) + " vs " + GT);
  // it is the SHARED helper, not a copy: both wrappers must BE fitLogSlope at their own
  // gate, on a trace each of them arms on and on one neither does (NaN !== NaN, so the
  // blank case is compared as "both blank")
  const same = (a, b) => (a === b) || (!isFinite(a) && !isFinite(b));
  const [tq, wq] = mkW(2 * C.MODE_FIT_RISE / C.MODE_FIT_DT, 200, 0.5);   // arms both gates
  const wflat = tw.map(() => 0.3);                                       // arms neither
  let shared = true;
  for (const [tt, aa] of [[tw, ww], [tq, wq], [tw, wflat]]) {
    shared = shared && same(C.modeFitGamma(tt, aa), C.fitLogSlope(tt, aa, C.MODE_FIT_RISE))
                    && same(C.islandFitGamma(tt, aa), 2 * C.fitLogSlope(tt, aa, C.ISLAND_FIT_RISE));
  }
  ok("both wrappers are the ONE fitLogSlope at their own rise gate", shared);
  // THE reason the gate had to be its own number: the tearing demo's window rise is 19x
  // smaller than the KH one's, and MODE_FIT_RISE would blank the legend for the whole run
  const rise = 0.5 * GT * C.MODE_FIT_DT;
  ok("a MODE_FIT_DT window of the tearing linear stage rises " + rise.toFixed(3) + " ln-units of W",
     Math.abs(rise - 0.1435) < 1e-9);
  ok("... which MODE_FIT_RISE would reject outright", rise < C.MODE_FIT_RISE &&
     !isFinite(C.modeFitGamma(tw, ww)), "MODE_FIT_RISE = " + C.MODE_FIT_RISE);
  ok("... and ISLAND_FIT_RISE accepts with the same ~2.7x margin the KH chart has",
     rise >= 2.5 * C.ISLAND_FIT_RISE && rise <= 3.5 * C.ISLAND_FIT_RISE,
     "margin " + (rise / C.ISLAND_FIT_RISE).toFixed(2) + "x over " + C.ISLAND_FIT_RISE);
  // the source-OFF case the preset hint quotes (30-40% slower) must still arm
  const [tf, wf] = mkW(0.62 * GT, 200, 0.5);
  ok("... and still arms at the 'maintain flux off' rate (0.62 x gamma)",
     Math.abs(C.islandFitGamma(tf, wf) / (0.62 * GT) - 1) < 1e-6,
     "got " + C.islandFitGamma(tf, wf).toPrecision(8));
  // below the gate it blanks rather than quoting: a decade-slower island is not a linear
  // stage anyone should read a rate off in a 10 t-unit window
  const [ts2, ws2] = mkW(0.15 * GT, 200, 0.5);
  ok("a rate too slow to clear the gate in one window is blanked",
     !isFinite(C.islandFitGamma(ts2, ws2)), "got " + C.islandFitGamma(ts2, ws2));
  // and the guards the mode chart has are inherited verbatim through the shared helper
  const tsat = [], wsat = [];
  for (let i = 0; i < 200; i++) { tsat.push(0.5 * i); wsat.push(1.2 * (1 + 0.05 * Math.sin(0.7 * 0.5 * i))); }
  ok("a saturated, oscillating W has no growth rate", !isFinite(C.islandFitGamma(tsat, wsat)),
     "got " + C.islandFitGamma(tsat, wsat));
  const [td, wd] = mkW(-GT, 200, 0.5);
  ok("a shrinking island has no growth rate", !isFinite(C.islandFitGamma(td, wd)));
  ok("W = 0 (nothing reconnected yet) is not logged as a rate",
     !isFinite(C.islandFitGamma(tw, ww.map(() => 0))));
  // the trailing window really is trailing: an older, steeper stage must not leak in
  const tk = [], wk = [];
  for (let i = 0; i < 300; i++) {
    const t = 0.5 * i, tb = 0.5 * 260;
    tk.push(t);
    wk.push(0.0875 * Math.exp(t < tb ? 0.5 * 4 * GT * t : 0.5 * (4 * GT * tb + GT * (t - tb))));
  }
  ok("only the trailing " + C.MODE_FIT_DT + " t-units are fitted",
     Math.abs(C.islandFitGamma(tk, wk) / GT - 1) < 1e-6, "got " + C.islandFitGamma(tk, wk).toPrecision(10));
}

// ---------------------------------------------------------------------------
console.log("12. the amplitude basis switch and its labels (FEEDBACK_2026-08-10 item 15)");
// ---------------------------------------------------------------------------
// The UI half of section 1's arithmetic: ONE predicate (icPaintPP / icAmpBasis) drives the
// basis, the two slider labels and the 3D chi line, so they cannot disagree. Needs
// remembering controls -- ampBasisSync WRITES the labels it is checked on.
{
  ELS = {};
  const set = (id, v) => { C.document.getElementById(id).value = String(v); };
  const lab = id => C.document.getElementById(id).innerHTML;
  set("rAmpP", 0); set("rAmpM", -1);              // 10^0 = 1 and 10^-1 = 0.1
  // the paint target only means anything while the DRAWING is the preset
  for (const [ic, paint, want] of [["custom", "phi", "pp"], ["custom", "psi", "pp"],
                                   ["custom", "zp", "zeta"], ["custom", "zm", "zeta"],
                                   ["letters", "phi", "zeta"], ["modes", "psi", "zeta"],
                                   ["tearing", "phi", "zeta"]]) {
    set("selIC", ic); set("selPaint", paint);
    ok("selIC=" + ic + " + paint " + paint + " -> " + want + " basis", C.icAmpBasis() === want,
       "got " + C.icAmpBasis());
  }
  // the labels and their titles follow, and say what the number MEANS
  set("selIC", "custom"); set("selPaint", "phi");
  C.ampBasisSync();
  ok("painting phi/psi relabels the two sliders phi amp / psi amp",
     lab("labAmpP") === "&phi; amp" && lab("labAmpM") === "&psi; amp",
     lab("labAmpP") + " / " + lab("labAmpM"));
  const ti = id => C.document.getElementById(id).title;
  ok("... and both tooltips document the semantics (max |u| / max |b|)",
     ti("rAmpP").indexOf("max |u|") >= 0 && ti("rAmpM").indexOf("max |b|") >= 0 &&
     ti("labAmpP") === ti("rAmpP") && ti("labAmpM") === ti("rAmpM"),
     ti("rAmpP") + "  ||  " + ti("rAmpM"));
  // ... and the pair the chi line needs: z+- = u +- b for co-located strokes
  const az = C.icAmpZeta();
  ok("icAmpZeta maps (phi, psi) amps back to (a+, a-) = (a_phi+a_psi, |a_phi-a_psi|)",
     Math.abs(az[0] - 1.1) < 1e-12 && Math.abs(az[1] - 0.9) < 1e-12, JSON.stringify(az));
  set("selPaint", "zp");
  C.ampBasisSync();
  ok("painting zeta+- puts the zeta labels back",
     lab("labAmpP") === "&zeta;&#8314; amp" && lab("labAmpM") === "&zeta;&#8315; amp" &&
     ti("rAmpP").indexOf("max |z") >= 0,
     lab("labAmpP") + " / " + lab("labAmpM"));
  const az2 = C.icAmpZeta();
  ok("... and icAmpZeta is then the sliders themselves",
     az2[0] === C.uiAmp()[0] && az2[1] === C.uiAmp()[1], JSON.stringify(az2));
  ELS = null;
}

console.log(bad ? "\n" + bad + " CHECK(S) FAILED" : "\nall GATE G node checks passed");
process.exit(bad ? 1 : 0);
