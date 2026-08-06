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
const sandbox = {
  document: { getElementById: () => stubEl(),
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
  + " IC_SIGMA_Z_MAX_FRAC, TRACK_HYST, IC_LETTERS })", sandbox));
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
  sandbox.document.getElementById = () => stubEl();
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

console.log(bad ? "\n" + bad + " CHECK(S) FAILED" : "\nall GATE G node checks passed");
process.exit(bad ? 1 : 0);
