// GATE H node unit checks, fp64 throughout (the app runs f32; the tolerances below are
// set accordingly where the comparison is against f32-recorded data).
// Usage: node checksh.js <webgpu-dir>
//
// Everything is computed TWICE from the pages' own inlined reference A states:
//   (a) by mirroring the new WGSL arithmetic exactly (the H_c accumulator lane of
//       energyPartial, the third lane of the spectra kernels, and cutPrep's analytic
//       kx / kz sums followed by rowsC2R's inverse along y), and
//   (b) independently, from the real-space fields obtained with a direct fp64 inverse
//       DFT -- no FFT, no shared code, no shortcut through the same formulas.
// The checks are that the two agree, and that the identities the charts rely on hold:
//   E+- = E_kin + E_mag +- H_c = <|z+-|^2>/2 >= 0,   E_tot = (E+ + E-)/2
//   sum_k [E_u, E_b, H_c](k) = [E_kin, E_mag, H_c]
//   the cut card's (u_x, u_y, b_x, b_y) lines ARE the real fields on x = Lx/2
// The chart layer itself is pulled out of the REAL common.js (below) and fed the same
// numbers, so what is checked is what the app draws, not a restatement of it.
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm");
const dir = process.argv[2] || path.join(__dirname, "..");
let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log((pass ? "  PASS  " : "  FAIL  ") + name + (note ? "   [" + note + "]" : ""));
};
const rel = (a, b) => Math.abs(a - b) / Math.max(1e-300, Math.abs(b));

// the reference vectors, straight out of the page's single inlined JSON line
function refvec(page) {
  const t = fs.readFileSync(path.join(dir, page), "utf8");
  const i = t.indexOf("JSON.parse(String.raw`") + "JSON.parse(String.raw`".length;
  return JSON.parse(t.slice(i, t.indexOf("`)", i)));
}
const TWOPI = 2 * Math.PI;

// ---- the app's own chart layer, in a vm --------------------------------------
const stubEl = () => ({ value: "", style: {}, textContent: "", innerHTML: "", checked: false,
                        disabled: false, min: "", max: "", step: "", options: [],
                        className: "", addEventListener() {}, appendChild() {} });
const sandbox = {
  document: { getElementById: () => stubEl(), createElement: () => stubEl(),
              createTextNode: () => ({}), querySelectorAll: () => [] },
  window: { addEventListener() {}, devicePixelRatio: 1, matchMedia: () => ({ matches: true }) },
  console, Math, JSON, Float32Array, Float64Array, Uint32Array, Uint8ClampedArray, Map, Set,
  Error, Promise, setTimeout, Number, String, Array, Object, isFinite, parseInt, parseFloat,
  URLSearchParams, performance: { now: () => 0 }
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(dir, "common.js"), "utf8"), sandbox, { filename: "common.js" });
const APP = vm.runInContext("({ ENERGY_MODES, SPEC_SETS, CUT_PAIRS, drawSpectrum, hist, histPush, histReset })",
                            sandbox);
// a 2D context that records what was written, so a chart's AXIS can be inspected
function recCtx() {
  const r = { texts: [], style: {} };
  const o = { fillStyle: "", strokeStyle: "", lineWidth: 1, textAlign: "left", font: "10px x", rec: r };
  for (const m of ["clearRect", "fillRect", "strokeRect", "beginPath", "moveTo", "lineTo", "stroke",
                   "fill", "clip", "save", "restore", "setLineDash", "rect", "setTransform"]) o[m] = () => {};
  o.measureText = t => ({ width: 6 * t.length });
  o.fillText = t => r.texts.push(String(t));
  return o;
}
const decades = texts => texts.filter(t => /^1e-?\d+$/.test(t)).join(",");

// ---- geometry ---------------------------------------------------------------
// mode index m = (iz*nkx + ix)*nky + iy (nz = 1 in 2D); a 2-field stack puts psi at +nm
function geom(R) {
  const nx = R.nx, ny = R.ny, nz = R.nz || 1, nkx = nx, nky = ny / 2 + 1;
  const kxOf = ix => ((ix < nx / 2) ? ix : ix - nx) * (TWOPI / R.Lx);
  const kyOf = iy => iy * (TWOPI / R.Ly);
  const yfac = iy => (iy === 0 || iy === nky - 1) ? 1 : 2;
  return { nx, ny, nz, nkx, nky, nm: nz * nkx * nky, kxOf, kyOf, yfac,
           invN2: 1 / ((nz * nz) * (nx * ny) * (nx * ny)),
           kunit: Math.min(TWOPI / R.Lx, TWOPI / R.Ly) };
}
// {re, im} nested arrays -> flat fp64 pair
function flat(o) {
  const re = [], im = [];
  (function rec(r, i) {
    if (Array.isArray(r[0])) { for (let k = 0; k < r.length; k++) rec(r[k], i[k]); }
    else { for (let k = 0; k < r.length; k++) { re.push(r[k]); im.push(i[k]); } }
  })(o.re, o.im);
  return { re: Float64Array.from(re), im: Float64Array.from(im) };
}

// ---- an independent real-space view: direct fp64 inverse DFT ------------------
// half-spectrum (iy < nky) -> the full one, by the rfftn reality condition
// F(-kz,-kx,-ky) = conj(F(kz,kx,ky)); then f(r) = (1/N) sum_k F e^{+2pi i k.r/n}.
function toReal(G, g, comp) {         // comp(m) -> [re, im] of the wanted component
  const { nx, ny, nz, nky } = g, N = nx * ny * nz;
  const Fre = new Float64Array(N), Fim = new Float64Array(N);
  for (let iz = 0; iz < nz; iz++) for (let ix = 0; ix < nx; ix++) for (let iy = 0; iy < ny; iy++) {
    let re, im;
    if (iy < nky) {
      const c = comp((iz * nx + ix) * nky + iy); re = c[0]; im = c[1];
    } else {
      const c = comp((((nz - iz) % nz) * nx + ((nx - ix) % nx)) * nky + (ny - iy));
      re = c[0]; im = -c[1];
    }
    const f = (iz * nx + ix) * ny + iy;
    Fre[f] = re; Fim[f] = im;
  }
  const out = new Float64Array(N);
  for (let pz = 0; pz < nz; pz++) for (let px = 0; px < nx; px++) for (let py = 0; py < ny; py++) {
    let s = 0;
    for (let iz = 0; iz < nz; iz++) {
      const az = TWOPI * iz * pz / nz;
      for (let ix = 0; ix < nx; ix++) {
        const ax = az + TWOPI * ix * px / nx;
        for (let iy = 0; iy < ny; iy++) {
          const a = ax + TWOPI * iy * py / ny, f = (iz * nx + ix) * ny + iy;
          s += Fre[f] * Math.cos(a) - Fim[f] * Math.sin(a);
        }
      }
    }
    out[(pz * nx + px) * ny + py] = s / N;
  }
  return out;
}

// ---- the checks, for one page -------------------------------------------------
function runPage(page) {
  const R = refvec(page), g = geom(page.indexOf("3d") >= 0 ? R : R);
  const F = flat(R.A_fields_k);
  const nm = g.nm, { nx, ny, nz, nkx, nky } = g;
  console.log("\n" + page + "  (" + nx + "x" + ny + (nz > 1 ? "x" + nz : "") + " reference state A)");

  // perpendicular index of a mode, and its k
  const mpOf = m => m % (nkx * nky);
  const kxOfM = m => g.kxOf(Math.floor(mpOf(m) / nky));
  const kyOfM = m => g.kyOf(mpOf(m) % nky);
  const yfOfM = m => g.yfac(mpOf(m) % nky);
  const phi = m => [F.re[m], F.im[m]], psi = m => [F.re[nm + m], F.im[nm + m]];

  // ---- 1. energyPartial's four lanes, mirrored ------------------------------
  let ek = 0, em = 0, hc = 0;
  for (let m = 0; m < nm; m++) {
    const ksq = kxOfM(m) * kxOfM(m) + kyOfM(m) * kyOfM(m), w = ksq * yfOfM(m);
    const p = phi(m), q = psi(m);
    ek += w * (p[0] * p[0] + p[1] * p[1]);
    em += w * (q[0] * q[0] + q[1] * q[1]);
    hc += w * (p[0] * q[0] + p[1] * q[1]);
  }
  ek *= 0.5 * g.invN2; em *= 0.5 * g.invN2; hc *= g.invN2;

  // the same three from real space: u = zhat x grad phi = (-d_y phi, d_x phi)
  const cUx = m => { const p = phi(m), k = kyOfM(m); return [k * p[1], -k * p[0]]; };
  const cUy = m => { const p = phi(m), k = kxOfM(m); return [-k * p[1], k * p[0]]; };
  const cBx = m => { const q = psi(m), k = kyOfM(m); return [k * q[1], -k * q[0]]; };
  const cBy = m => { const q = psi(m), k = kxOfM(m); return [-k * q[1], k * q[0]]; };
  const ux = toReal(F, g, cUx), uy = toReal(F, g, cUy);
  const bx = toReal(F, g, cBx), by = toReal(F, g, cBy);
  const N = nx * ny * nz;
  let rk = 0, rm = 0, rh = 0;
  for (let i = 0; i < N; i++) {
    rk += ux[i] * ux[i] + uy[i] * uy[i];
    rm += bx[i] * bx[i] + by[i] * by[i];
    rh += ux[i] * bx[i] + uy[i] * by[i];
  }
  rk = 0.5 * rk / N; rm = 0.5 * rm / N; rh = rh / N;
  ok("E_kin: energyPartial lane 0 == 0.5<|u|^2> from real space", rel(ek, rk) < 1e-12,
     "kernel " + ek.toPrecision(12) + " vs real " + rk.toPrecision(12));
  ok("E_mag: energyPartial lane 1 == 0.5<|b|^2>", rel(em, rm) < 1e-12,
     "kernel " + em.toPrecision(12) + " vs real " + rm.toPrecision(12));
  ok("H_c:   the NEW lane 3 == <u.b> from real space", rel(hc, rh) < 1e-12,
     "kernel " + hc.toPrecision(12) + " vs real " + rh.toPrecision(12));
  ok("(E_kin, E_mag) still match the recorded jax A_energy",
     rel(ek, R.A_energy[0]) < 1e-9 && rel(em, R.A_energy[1]) < 1e-9,
     "jax " + R.A_energy.map(v => v.toPrecision(8)).join(", "));

  // ---- 2. the Elsasser energies the trace card draws ------------------------
  const Ep = ek + em + hc, Em = ek + em - hc;
  let zp = 0, zm = 0;
  for (let i = 0; i < N; i++) {
    zp += (ux[i] + bx[i]) * (ux[i] + bx[i]) + (uy[i] + by[i]) * (uy[i] + by[i]);
    zm += (ux[i] - bx[i]) * (ux[i] - bx[i]) + (uy[i] - by[i]) * (uy[i] - by[i]);
  }
  zp = 0.5 * zp / N; zm = 0.5 * zm / N;
  ok("E+ = E_kin + E_mag + H_c == 0.5<|z+|^2>", rel(Ep, zp) < 1e-12,
     Ep.toPrecision(12) + " vs " + zp.toPrecision(12));
  ok("E- = E_kin + E_mag - H_c == 0.5<|z-|^2>", rel(Em, zm) < 1e-12,
     Em.toPrecision(12) + " vs " + zm.toPrecision(12));
  ok("E_tot = (E+ + E-)/2 (the repo's Elsasser convention)", rel(0.5 * (Ep + Em), ek + em) < 1e-14);
  ok("both Elsasser energies are non-negative (|H_c| <= E_kin + E_mag)", Ep >= 0 && Em >= 0,
     "sigma_c = " + ((Ep - Em) / (Ep + Em)).toFixed(6));

  // ---- 3. the spectra kernels' three lanes ----------------------------------
  // same weight, binned by round(k_perp/kunit); summing over ALL bins must give the
  // energies back exactly (the chart only PLOTS bins 1..nb-1)
  const nb = Math.floor(Math.min(nx, ny) / 3);
  const bins = [];
  let su = 0, sb = 0, sh = 0, plotted = 0;
  for (let m = 0; m < nm; m++) {
    const kx = kxOfM(m), ky = kyOfM(m), ksq = kx * kx + ky * ky;
    const w = 0.5 * ksq * g.yfac(mpOf(m) % nky) * g.invN2;
    const p = phi(m), q = psi(m);
    const eu = w * (p[0] * p[0] + p[1] * p[1]);
    const eb = w * (q[0] * q[0] + q[1] * q[1]);
    const h = 2 * w * (p[0] * q[0] + p[1] * q[1]);
    const b = Math.round(Math.sqrt(ksq) / g.kunit);
    if (!bins[b]) bins[b] = [0, 0, 0];
    bins[b][0] += eu; bins[b][1] += eb; bins[b][2] += h;
    su += eu; sb += eb; sh += h;
    if (b >= 1 && b < nb) plotted += eu + eb;
  }
  ok("sum_k E_u(k) == E_kin", rel(su, ek) < 1e-12, "sum " + su.toPrecision(12));
  ok("sum_k E_b(k) == E_mag", rel(sb, em) < 1e-12);
  ok("sum_k H_c(k) == H_c", rel(sh, hc) < 1e-12, "sum " + sh.toPrecision(12));
  let worst = 0, negs = 0;
  for (const b in bins) {
    const [u, v, h] = bins[b];
    if (u + v + h < -1e-14 * (u + v) || u + v - h < -1e-14 * (u + v)) negs++;
    worst = Math.max(worst, Math.abs(h) / Math.max(1e-300, u + v));
  }
  ok("every bin's E+-(k) = E_u+E_b+-H_c is non-negative", negs === 0,
     "max |H_c(k)|/(E_u+E_b) = " + worst.toFixed(6));
  // (informational, and unchanged by Phase H: the chart's x range is k < nb = min/3,
  // so the few dealiased-away corner modes above it are not drawn)
  ok("the plotted bins 1..nb-1 hold essentially all the energy", plotted > 0.99 * (ek + em),
     "nb = " + nb + ", plotted " + (100 * plotted / (ek + em)).toFixed(3) + "%");

  // ---- 4. the cut card's four component lines -------------------------------
  // cutPrep: e^{i kx Lx/2} = (-1)^ix, and in 3D the plane's kz phase; then rowsC2R.
  const planes = nz > 1 ? [0, 1, nz >> 1, nz - 1] : [0];
  let cutErr = 0, cutMax = 0, zErr = 0;
  for (const pz of planes) {
    // the analytic kx (and kz) sums, exactly as the kernel does them
    const rows = [new Array(nky), new Array(nky), new Array(nky), new Array(nky)];
    for (let j = 0; j < nky; j++) {
      let a0 = [0, 0], a1 = [0, 0], b0 = [0, 0], b1 = [0, 0];
      for (let iz = 0; iz < nz; iz++) {
        const th = TWOPI * iz * pz / nz, wc = Math.cos(th), ws = Math.sin(th);
        for (let ix = 0; ix < nx; ix++) {
          const s = (ix & 1) ? -1 : 1, wr = s * wc, wi = s * ws;
          const m = (iz * nkx + ix) * nky + j, kx = g.kxOf(ix);
          const p = phi(m), q = psi(m);
          const pr = wr * p[0] - wi * p[1], pi = wr * p[1] + wi * p[0];
          const qr = wr * q[0] - wi * q[1], qi = wr * q[1] + wi * q[0];
          a0 = [a0[0] + pr, a0[1] + pi]; a1 = [a1[0] + kx * pr, a1[1] + kx * pi];
          b0 = [b0[0] + qr, b0[1] + qi]; b1 = [b1[0] + kx * qr, b1[1] + kx * qi];
        }
      }
      const s = 1 / (nx * nz), ky = g.kyOf(j);
      rows[0][j] = [s * ky * a0[1], -s * ky * a0[0]];
      rows[1][j] = [-s * a1[1], s * a1[0]];
      rows[2][j] = [s * ky * b0[1], -s * ky * b0[0]];
      rows[3][j] = [-s * b1[1], s * b1[0]];
    }
    // rowsC2R: hermitian completion along y, inverse transform, 1/ny
    const line = r => {
      const out = new Float64Array(ny);
      for (let py = 0; py < ny; py++) {
        let acc = 0;
        for (let iy = 0; iy < ny; iy++) {
          const h = iy < nky ? r[iy] : [r[ny - iy][0], -r[ny - iy][1]];
          const a = TWOPI * iy * py / ny;
          acc += h[0] * Math.cos(a) - h[1] * Math.sin(a);
        }
        out[py] = acc / ny;
      }
      return out;
    };
    const got = rows.map(line);
    const want = [ux, uy, bx, by];
    const ix0 = nx / 2;
    for (let py = 0; py < ny; py++) {
      for (let k = 0; k < 4; k++) {
        const w = want[k][(pz * nx + ix0) * ny + py];
        cutErr = Math.max(cutErr, Math.abs(got[k][py] - w));
        cutMax = Math.max(cutMax, Math.abs(w));
      }
      // the |z+-| pair the card derives on the CPU
      const zpp = Math.hypot(got[0][py] + got[2][py], got[1][py] + got[3][py]);
      const zmm = Math.hypot(ux[(pz * nx + ix0) * ny + py] + bx[(pz * nx + ix0) * ny + py],
                             uy[(pz * nx + ix0) * ny + py] + by[(pz * nx + ix0) * ny + py]);
      zErr = Math.max(zErr, Math.abs(zpp - zmm));
    }
  }
  ok("cutPrep + rowsC2R == (u_x, u_y, b_x, b_y) on x = Lx/2, every plane",
     cutErr < 1e-12 * cutMax, "max abs err " + cutErr.toExponential(2) +
     " on |max| " + cutMax.toPrecision(6) + " (" + planes.length + " plane(s))");
  ok("the derived |z+| line matches |u+b| in real space", zErr < 1e-12 * cutMax,
     "max abs err " + zErr.toExponential(2));

  // ---- 5. the CHART layer's own series functions (real common.js) -----------
  APP.histReset(); APP.histPush(1.0, ek, em, hc);
  const eSer = APP.ENERGY_MODES.pmt.map(s => s[2](0));
  ok("energy card, E+- mode: the drawn series ARE (E+, E-, E_tot)",
     rel(eSer[0], Ep) < 1e-14 && rel(eSer[1], Em) < 1e-14 && rel(eSer[2], ek + em) < 1e-14,
     APP.ENERGY_MODES.pmt.map((s, i) => s[0] + "=" + eSer[i].toPrecision(8)).join(" "));
  let sp = 0, sm = 0;
  for (const b in bins) {
    sp += APP.SPEC_SETS.pm[0][2](bins[b][0], bins[b][1], bins[b][2]);
    sm += APP.SPEC_SETS.pm[1][2](bins[b][0], bins[b][1], bins[b][2]);
  }
  ok("spectrum card, E+- mode: sum_k of the drawn series == (E+, E-)",
     rel(sp, Ep) < 1e-12 && rel(sm, Em) < 1e-12,
     "sum E+(k) = " + sp.toPrecision(10) + ", sum E-(k) = " + sm.toPrecision(10));
  // the cut card's pair selectors, on the real line data of plane 0
  {
    const ix0 = nx / 2, vals = new Float64Array(4 * ny);
    for (let py = 0; py < ny; py++) {
      vals[py] = ux[ix0 * ny + py]; vals[ny + py] = uy[ix0 * ny + py];
      vals[2 * ny + py] = bx[ix0 * ny + py]; vals[3 * ny + py] = by[ix0 * ny + py];
    }
    let du = 0, db = 0, dz = 0;
    for (let py = 0; py < ny; py++) {
      const pu = APP.CUT_PAIRS.u.f(vals, ny, py), pb = APP.CUT_PAIRS.b.f(vals, ny, py);
      const pz = APP.CUT_PAIRS.z.f(vals, ny, py);
      du = Math.max(du, Math.abs(pu[0] - ux[ix0 * ny + py]), Math.abs(pu[1] - uy[ix0 * ny + py]));
      db = Math.max(db, Math.abs(pb[0] - bx[ix0 * ny + py]), Math.abs(pb[1] - by[ix0 * ny + py]));
      dz = Math.max(dz,
        Math.abs(pz[0] - Math.hypot(ux[ix0 * ny + py] + bx[ix0 * ny + py],
                                    uy[ix0 * ny + py] + by[ix0 * ny + py])),
        Math.abs(pz[1] - Math.hypot(ux[ix0 * ny + py] - bx[ix0 * ny + py],
                                    uy[ix0 * ny + py] - by[ix0 * ny + py])));
    }
    ok("cut card pairs: u / b / |z+-| select the right components",
       du === 0 && db === 0 && dz < 1e-15 * cutMax, "max |z+-| err " + dz.toExponential(2));
  }

  // ---- 6. the 3D spectrum chart's y range is set by the PERP spectra alone ---
  if (nz > 1) {
    const nbb = nb, nzb = 4;
    const perp = new Float64Array(3 * nbb), pare = new Float64Array(3 * nzb);
    for (let b = 1; b < nbb; b++) { perp[b] = 1e-2 / b; perp[nbb + b] = 2e-2 / b; }
    for (let b = 0; b < nzb; b++) { pare[b] = 1e-3; pare[nzb + b] = 1e-3; }
    const draw = (par, o) => {
      const c = recCtx();
      APP.drawSpectrum(c, { perp, nb: nbb, fshell: [1, 3], par, parKfac: 1 }, o);
      return decades(c.rec.texts);
    };
    const big = pare.map(v => v * 1e6);
    const a1 = draw(pare, { sq: "ub", sd: "both" });
    const a2 = draw(big, { sq: "ub", sd: "both" });
    ok("a 1e6x louder E(k_par) does not move the y axis (Alfred 2026-08-06)",
       a1 === a2 && a1.length > 0, "decades " + a1);
    const a3 = draw(big, { sq: "ub", sd: "par" });
    ok("... but with ONLY the parallel spectra selected it does set the range", a3 !== a2,
       "par-only decades " + a3);
    const a4 = draw(pare, { sq: "ub", sd: "perp" });
    ok("perp-only and perp+par share one y range", a4 === a1);
  }
}

console.log("GATE H fp64 checks (E+-, H_c, the spectra lanes, the cut lines)");
runPage("rmhd2d.html");
runPage("rmhd3d.html");
console.log(bad ? "\n" + bad + " GATE H check(s) FAILED" : "\nall GATE H node checks passed");
process.exit(bad ? 1 : 0);
