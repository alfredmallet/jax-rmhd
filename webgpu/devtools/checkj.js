// GATE J physics checks (REFINE_PLAN J): the equilibrium ICs, the island-width
// machinery and the per-field dissipation, at fp64, in node.
//
//   node checkj.js [dir]        (dir defaults to ..)
//
// The initial conditions are built by the REAL app code -- the page is booted on the
// shared stub (stubenv.js), its sliders are set, and icPresetFields() is called in the
// page's own context -- so what is evolved below is exactly what the browser uploads.
//
// The time integration here is an INDEPENDENT fp64 pseudospectral 2D RMHD solver (plain
// complex FFTs + RK4, no integrating factor, same 2/3 elliptical dealias as makeGrid),
// and the growth rates it measures are compared against eqlinear.py -- a 1D GENERALIZED
// EIGENVALUE solve of the linearized system on Fourier differentiation matrices. Two
// different methods, one answer; neither is the other's mirror.
"use strict";
const path = require("path");
const dir = process.argv[2] || path.join(__dirname, "..");

// ---------------------------------------------------------------------------
// eqlinear.py reference values (regenerate: python3 devtools/eqlinear.py)
// ---------------------------------------------------------------------------
// Largest-real-part eigenvalue of the linearized RMHD system at k_y = 2pi/Ly, in the
// SAME box and with the SAME profiles the presets build. Converged in the number of
// Fourier modes: `python3 eqlinear.py 768` reproduces every entry below (it prints this
// table and nothing else needs transcribing).
const REF = {
  Lx: 4 * Math.PI, Ly: 2 * Math.PI,
  tear: { a: 0.1 * 4 * Math.PI, psi0: 1.65,
          g: { "1e-3": 0.028716, "3.1623e-3": 0.051395 },      // eta = nu, i.e. Pm = 1
          gPm0p1: 0.114636,                                    // nu = 1e-3, eta = 1e-2
          gPm0: 0.043646,                                      // nu = 0, eta = 1e-3
          dp_a: 8.3995 },
  kh: { a: 0.05 * 4 * Math.PI, U0: 1, nu: 3.1623e-4,
        g0: 0.266260, gHalf: 0.206229, gSup: 0.003646 }        // b0 = 0, 0.5, 1.2
};

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log("  " + (pass ? "PASS" : "FAIL") + "  " + name + (note ? "   [" + note + "]" : ""));
};
const rel = (a, b) => Math.abs(a - b) / Math.max(1e-300, Math.abs(b));

// ===========================================================================
// a minimal fp64 pseudospectral 2D RMHD solver
// ===========================================================================
// State is the FULL complex spectrum (nx x ny) of phi and psi -- no rfft2 packing, which
// costs 2x and saves 100 lines. Real fields keep their imaginary parts at round-off.
function fft(re, im, n, sign) {                       // in-place radix-2, stride 1
  for (let i = 1, j = 0; i < n; i++) {
    let b = n >> 1;
    for (; j & b; b >>= 1) j ^= b;
    j ^= b;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = sign * 2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}
// 2D transform of a row-major (ix*ny + iy) plane. sign -1 = forward (unnormalized),
// +1 = inverse (normalized by nx*ny).
function fft2(re, im, nx, ny, sign) {
  const br = new Float64Array(Math.max(nx, ny)), bi = new Float64Array(Math.max(nx, ny));
  for (let i = 0; i < nx; i++) {                       // along y (contiguous)
    for (let j = 0; j < ny; j++) { br[j] = re[i * ny + j]; bi[j] = im[i * ny + j]; }
    fft(br, bi, ny, sign);
    for (let j = 0; j < ny; j++) { re[i * ny + j] = br[j]; im[i * ny + j] = bi[j]; }
  }
  for (let j = 0; j < ny; j++) {                       // along x (stride ny)
    for (let i = 0; i < nx; i++) { br[i] = re[i * ny + j]; bi[i] = im[i * ny + j]; }
    fft(br, bi, nx, sign);
    for (let i = 0; i < nx; i++) { re[i * ny + j] = br[i]; im[i * ny + j] = bi[i]; }
  }
  if (sign > 0) { const s = 1 / (nx * ny); for (let i = 0; i < nx * ny; i++) { re[i] *= s; im[i] *= s; } }
}

function makeSolver(o) {
  const nx = o.nx, ny = o.ny, n = nx * ny;
  const kx = new Float64Array(n), ky = new Float64Array(n), ksq = new Float64Array(n);
  const iksq = new Float64Array(n), de = new Float64Array(n);
  const cutx = nx / 3, cuty = ny / 3;
  for (let i = 0; i < nx; i++) {
    const ix = i < nx / 2 ? i : i - nx;
    for (let j = 0; j < ny; j++) {
      const jy = j < ny / 2 ? j : j - ny, m = i * ny + j;
      kx[m] = ix * 2 * Math.PI / o.Lx; ky[m] = jy * 2 * Math.PI / o.Ly;
      ksq[m] = kx[m] * kx[m] + ky[m] * ky[m];
      iksq[m] = ksq[m] > 0 ? 1 / ksq[m] : 0;
      de[m] = ((ix / cutx) * (ix / cutx) + (jy / cuty) * (jy / cuty)) < 1 ? 1 : 0;
    }
  }
  const A = () => new Float64Array(n);
  const gr = [], gi = [], nlr = [A(), A()], nli = [A(), A()];
  for (let k = 0; k < 8; k++) { gr.push(A()); gi.push(A()); }
  // rhs of (phi, psi) in k space; f = [phiR, phiI, psiR, psiI]
  function rhs(f, out) {
    // i*k * {phi, psi, vort, jpar} -> the 8 gradient components
    for (let m = 0; m < n; m++) {
      const pr = f[0][m], pi = f[1][m], sr = f[2][m], si = f[3][m];
      const vr = -ksq[m] * pr, vi = -ksq[m] * pi, jr = -ksq[m] * sr, ji = -ksq[m] * si;
      const src = [[pr, pi], [pr, pi], [sr, si], [sr, si], [vr, vi], [vr, vi], [jr, ji], [jr, ji]];
      for (let c = 0; c < 8; c++) {
        const k = (c & 1) ? ky[m] : kx[m];
        gr[c][m] = -k * src[c][1]; gi[c][m] = k * src[c][0];
      }
    }
    for (let c = 0; c < 8; c++) fft2(gr[c], gi[c], nx, ny, +1);
    // {psi,j} - {phi,w}  and  -{phi,psi}
    for (let m = 0; m < n; m++) {
      const ax = gr[0][m], ay = gr[1][m], bx = gr[2][m], by = gr[3][m];
      const cx = gr[4][m], cy = gr[5][m], dx = gr[6][m], dy = gr[7][m];
      nlr[0][m] = (bx * dy - by * dx) - (ax * cy - ay * cx);
      nlr[1][m] = -(ax * by - ay * bx);
      nli[0][m] = 0; nli[1][m] = 0;
    }
    for (let c = 0; c < 2; c++) fft2(nlr[c], nli[c], nx, ny, -1);
    for (let m = 0; m < n; m++) {
      const h = Math.pow(ksq[m], o.hyper);
      out[0][m] = -iksq[m] * de[m] * nlr[0][m] - o.nu * h * f[0][m];
      out[1][m] = -iksq[m] * de[m] * nli[0][m] - o.nu * h * f[1][m];
      out[2][m] = de[m] * nlr[1][m] - o.eta * h * f[2][m];
      out[3][m] = de[m] * nli[1][m] - o.eta * h * f[3][m];
    }
    // MAINTAIN (o.maintain, REFINE_PLAN J2.3): the static source S = -eta grad^2 psi_eq,
    // psi_eq being the k_y = 0 column of the IC -- exactly what the app's srcInit
    // extracts and what nlAssemble then multiplies by the SAME -lin_L the stage applies.
    // Written with the identical grouping as the damping term above, so it cancels it
    // bitwise while psi is still the equilibrium.
    if (src) for (let m = 0; m < n; m++) {
      const e = o.eta * Math.pow(ksq[m], o.hyper);
      out[2][m] += e * src[0][m];
      out[3][m] += e * src[1][m];
    }
  }
  const f = [A(), A(), A(), A()];
  // FREEZE (o.freeze): restore the k_y = 0 modes after every step, i.e. hold the
  // equilibrium. That is exactly the assumption the eigenvalue problem makes, and it is
  // what makes the two comparable: a free-running initial-value problem also lets psi_eq
  // diffuse (rate ~ eta/a^2), which slowly lowers the growth rate -- see section 4.
  let eqk = null, src = null;
  const k1 = [A(), A(), A(), A()], k2 = [A(), A(), A(), A()];
  const k3 = [A(), A(), A(), A()], k4 = [A(), A(), A(), A()], tmp = [A(), A(), A(), A()];
  const axpy = (dst, a, s, b) => { for (let c = 0; c < 4; c++) for (let m = 0; m < n; m++) dst[c][m] = a[c][m] + s * b[c][m]; };
  return {
    nx, ny, n, kx, ky, de, f,
    setIC(phiReal, psiReal) {
      for (const [src, dr, di] of [[phiReal, f[0], f[1]], [psiReal, f[2], f[3]]]) {
        dr.set(src); di.fill(0);
        fft2(dr, di, nx, ny, -1);
        for (let m = 0; m < n; m++) { dr[m] *= de[m]; di[m] *= de[m]; }   // icFinish
      }
      if (o.freeze) eqk = f.map(a => a.slice());
      if (o.maintain) {
        src = [A(), A()];
        for (let i = 0; i < nx; i++) { src[0][i * ny] = f[2][i * ny]; src[1][i * ny] = f[3][i * ny]; }
      }
    },
    // the k_y = 0 column of psi: the equilibrium, as the maintained source sees it
    eqColumn() {
      const c = new Float64Array(2 * nx);
      for (let i = 0; i < nx; i++) { c[2 * i] = f[2][i * ny]; c[2 * i + 1] = f[3][i * ny]; }
      return c;
    },
    step(dt) {
      rhs(f, k1); axpy(tmp, f, 0.5 * dt, k1);
      rhs(tmp, k2); axpy(tmp, f, 0.5 * dt, k2);
      rhs(tmp, k3); axpy(tmp, f, dt, k3);
      rhs(tmp, k4);
      for (let c = 0; c < 4; c++) for (let m = 0; m < n; m++) {
        f[c][m] += dt / 6 * (k1[c][m] + 2 * k2[c][m] + 2 * k3[c][m] + k4[c][m]);
      }
      if (eqk) for (let c = 0; c < 4; c++) for (let i = 0; i < nx; i++) f[c][i * ny] = eqk[c][i * ny];
    },
    // the k_y = 1 Fourier amplitude of psi ON the resonant surface x = Lx/2: psitilde,
    // the quantity the island width is built from ((-1)^ix is e^{i kx Lx/2}).
    psiTilde() {
      let ar = 0, ai = 0;
      for (let i = 0; i < nx; i++) {
        const s = (i & 1) ? -1 : 1, m = i * ny + 1;
        ar += s * f[2][m]; ai += s * f[3][m];
      }
      return 2 * Math.hypot(ar, ai) / (nx * ny);
    },
    // the app's CUT LINE on x = Lx/2: (u_x, u_y, b_x, b_y) stacked ny at a time, the same
    // 4*ny layout readCutLine returns, in fp32 as the GPU readback is. u_x = -d_y phi and
    // u_y = d_x phi (psi -> b), evaluated at ix = nx/2 by folding the x sum first --
    // e^{i kx Lx/2} = (-1)^ix, exactly as psiTilde above does it -- and inverse
    // transforming the remaining ny column.
    cutLine() {
      const out = new Float32Array(4 * ny), br = new Float64Array(ny), bi = new Float64Array(ny);
      for (let r = 0; r < 4; r++) {
        const c = (r >> 1) * 2, ddx = (r & 1);     // r = 0,1 -> phi;  r = 2,3 -> psi
        br.fill(0); bi.fill(0);
        for (let i = 0; i < nx; i++) {
          const s = (i & 1) ? -1 : 1;
          for (let j = 0; j < ny; j++) {
            const m = i * ny + j, k = ddx ? kx[m] : -ky[m];        // i*k*F, k = kx or -ky
            br[j] += s * (-k * f[c + 1][m]); bi[j] += s * (k * f[c][m]);
          }
        }
        fft(br, bi, ny, +1);                        // unnormalized inverse over the column
        for (let j = 0; j < ny; j++) out[r * ny + j] = br[j] / (nx * ny);
      }
      return out;
    },
    // energy in the |k_y| = 1 harmonic of a field (0 = phi, 2 = psi): the linear mode
    modeEnergy(off) {
      let s = 0;
      for (let i = 0; i < nx; i++) {
        for (const j of [1, ny - 1]) {
          const m = i * ny + j;
          s += f[off][m] * f[off][m] + f[off + 1][m] * f[off + 1][m];
        }
      }
      return s;
    }
  };
}
// least-squares slope of ln(amplitude) over a window
function growthRate(ts, as) {
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  const n = ts.length;
  for (let i = 0; i < n; i++) {
    const y = Math.log(as[i]);
    sx += ts[i]; sy += y; sxx += ts[i] * ts[i]; sxy += ts[i] * y;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}
// run one solver and fit the growth rate of `amp()` over [t0, t1]
function measure(s, amp, dt, t0, t1, every) {
  const ts = [], as = [];
  for (let n = 0, t = 0; t <= t1 + 1e-9; n++, t = n * dt) {
    if (n % every === 0 && t >= t0) { ts.push(t); as.push(amp()); }
    s.step(dt);
  }
  return growthRate(ts, as);
}

// ===========================================================================
// the app's initial conditions, from the app's own code
// ===========================================================================
const env = require("./stubenv")(dir, "rmhd2d.html");
function appIC(preset, nx, ny, Lx, Ly, sliders) {
  return env.run(`function(preset, nx, ny, Lx, Ly, sl){
    for (const k in sl) document.getElementById(k).value = String(sl[k]);
    document.getElementById("selIC").value = preset;
    const f = icPresetFields({ nx: nx, ny: ny, Lx: Lx, Ly: Ly }, preset, 1, 1, null);
    return { phi: Array.from(f.phi), psi: Array.from(f.psi),
             eq: { on: icEq.on, kh: icEq.kh, curv: icEq.curv, a: icEq.a, w0: icEq.w0 } };
  }`, preset, nx, ny, Lx, Ly, sliders);
}
// d/dx of a 1D periodic profile, 4th order
const d1 = (f, i, h) => {
  const n = f.length, w = k => f[(((i + k) % n) + n) % n];
  return (w(-2) - 8 * w(-1) + 8 * w(1) - w(2)) / (12 * h);
};

console.log("1. equilibrium ICs: the app's arrays vs the analytic profiles (J.3)");
{
  const nx = 512, ny = 8, Lx = REF.Lx, Ly = REF.Ly;
  // --- tearing: psi_eq = psi0 sech^2((x - Lx/2)/a), phi_eq = 0 ---------------
  const a = REF.tear.a, psi0 = REF.tear.psi0, A = 1e-3;
  const T = appIC("tearing", nx, ny, Lx, Ly,
                  { rEqA: 0.1, rEqPsi0: psi0, rEqPert: Math.log10(A) });
  let ephi = 0, epsi = 0, epert = 0;
  for (let i = 0; i < nx; i++) {
    const x = i * Lx / nx, s2 = 1 / Math.pow(Math.cosh((x - 0.5 * Lx) / a), 2);
    for (let j = 0; j < ny; j++) {
      const y = j * Ly / ny, m = i * ny + j;
      ephi = Math.max(ephi, Math.abs(T.phi[m]));
      epsi = Math.max(epsi, Math.abs(T.psi[m] - psi0 * s2 - A * s2 * Math.cos(2 * Math.PI / Ly * y)));
    }
    epert = Math.max(epert, Math.abs(T.psi[i * ny] - T.psi[i * ny + ny / 2] - 2 * A * s2));
  }
  ok("tearing phi_eq is identically zero", ephi === 0, "max |phi| = " + ephi);
  ok("tearing psi = psi0 sech^2 + A sech^2 cos(k y)", epsi < 1e-6 * psi0,
     "max abs err " + epsi.toExponential(2));
  ok("the seed value ON the resonant surface is the slider", epert < 1e-6,
     "max err " + epert.toExponential(2));
  // periodicity of the equilibrium across the box edge (exponentially small, not zero)
  const jump = Math.abs(T.psi[0] - T.psi[(nx - 1) * ny]) / psi0;
  ok("tearing psi is periodic to O(e^-Lx/2a)", jump < 1e-3, "relative edge jump " + jump.toExponential(2));
  // measured curvature vs the analytic -2 psi0/a^2
  ok("measured |psi_eq''(x_s)| = 2 psi0/a^2", rel(T.eq.curv, 2 * psi0 / (a * a)) < 1e-6,
     "measured " + T.eq.curv.toPrecision(8) + " vs " + (2 * psi0 / (a * a)).toPrecision(8));
  ok("recorded W(0) = 4 sqrt(A/|psi''|)", rel(T.eq.w0, 4 * Math.sqrt(A / T.eq.curv)) < 1e-12,
     "W(0) = " + T.eq.w0.toPrecision(6));

  // --- KH: u_y = d_x phi is the double tanh ---------------------------------
  const aK = REF.kh.a, U0 = 1, b0 = 0.7;
  const K = appIC("kh", nx, ny, Lx, Ly,
                  { rEqA: 0.05, rEqU0: U0, rEqB0: b0, rEqPert: -9 });
  const prof = i => K.phi[i * ny], sprof = i => K.psi[i * ny];
  const ph = new Float64Array(nx), ps = new Float64Array(nx);
  for (let i = 0; i < nx; i++) { ph[i] = prof(i); ps[i] = sprof(i); }
  let eu = 0, eb = 0;
  const uy = x => U0 * (Math.tanh((x - 0.25 * Lx) / aK) - Math.tanh((x - 0.75 * Lx) / aK) - 1);
  for (let i = 4; i < nx - 4; i++) {                    // away from the periodic seam
    const x = i * Lx / nx;
    eu = Math.max(eu, Math.abs(d1(ph, i, Lx / nx) - uy(x)));
    eb = Math.max(eb, Math.abs(d1(ps, i, Lx / nx) - (b0 / U0) * uy(x)));
  }
  ok("KH u_y = d_x phi is the double tanh", eu < 1e-4 * U0, "max abs err " + eu.toExponential(2));
  ok("KH b_y = d_x psi is the same profile x b0", eb < 1e-4 * U0, "max abs err " + eb.toExponential(2));
  ok("KH phi is periodic to O(a e^-Lx/4a)", Math.abs(ph[0] - ph[nx - 1] - (uy(0) * Lx / nx)) < 1e-3,
     "edge residual " + Math.abs(ph[0] - ph[nx - 1] - uy(0) * Lx / nx).toExponential(2));
  ok("the two layers are independent (a << |x2-x1|)", aK * 8 < 0.5 * Lx,
     "|x2-x1|/a = " + (0.5 * Lx / aK).toFixed(1));
  // KH has no resonant surface on x = Lx/2, so it must NOT arm the island chart -- and
  // the k_y mode chart's flag is the mirror image: exactly one preset arms each
  ok("KH leaves the island record off", K.eq.on === false && T.eq.on === true,
     "icEq.on: kh " + K.eq.on + ", tearing " + T.eq.on);
  ok("... and only KH arms the k_y mode chart", K.eq.kh === true && T.eq.kh === false,
     "icEq.kh: kh " + K.eq.kh + ", tearing " + T.eq.kh);
}

console.log("2. island-width machinery (J.4)");
{
  // an ANALYTIC island: psi(x_s, y) = psi_s + psitilde cos(k y) => b_x = -d_y psi
  const ny = 128, Ly = REF.Ly, k = 2 * Math.PI / Ly;
  const res = env.run(`function(ny, Ly, k, pt, curv){
    const vals = new Float32Array(4 * ny);
    for (let j = 0; j < ny; j++) vals[2 * ny + j] = pt * k * Math.sin(k * j * Ly / ny);
    icEq.on = true; icEq.curv = curv;
    const psi = icLineIntegrate(vals.subarray(2 * ny, 3 * ny), ny, Ly);
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < ny; j++) { lo = Math.min(lo, psi[j]); hi = Math.max(hi, psi[j]); }
    return { span: hi - lo, W: islandWidth(vals, ny, Ly) };
  }`, ny, Ly, k, 3e-3, 2.0894195910618806);
  // the line comes back from the GPU as fp32, so 1e-6 is the honest tolerance
  ok("icLineIntegrate recovers psi from b_x exactly", rel(res.span, 2 * 3e-3) < 1e-6,
     "psi_X - psi_O = " + res.span.toPrecision(10) + " vs 2 psitilde = " + (2 * 3e-3));
  const Wref = 4 * Math.sqrt(3e-3 / 2.0894195910618806);
  ok("W = 4 sqrt(psitilde/|psi''|)", rel(res.W, Wref) < 1e-6,
     "W = " + res.W.toPrecision(8) + " vs " + Wref.toPrecision(8));
  // a doubled psitilde must give exactly sqrt(2) x the width
  const r2 = env.run(`function(ny, Ly, k, pt){
    const vals = new Float32Array(4 * ny);
    for (let j = 0; j < ny; j++) vals[2 * ny + j] = pt * k * Math.sin(k * j * Ly / ny);
    return islandWidth(vals, ny, Ly);
  }`, ny, Ly, k, 6e-3);
  ok("W scales as sqrt(psitilde)", rel(r2 / res.W, Math.SQRT2) < 1e-6, "ratio " + (r2 / res.W).toPrecision(8));
}

console.log("3. per-field dissipation: linear decay rates (J.1)");
{
  // a pure k mode in each field, no nonlinearity to speak of: phi must decay at nu k^2h
  // and psi at eta k^2h, independently.
  const nx = 32, ny = 32, L = 2 * Math.PI, nu = 0.02, eta = 0.05;
  const s = makeSolver({ nx, ny, Lx: L, Ly: L, hyper: 1, nu, eta });
  const phi = new Float64Array(nx * ny), psi = new Float64Array(nx * ny);
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
    phi[i * ny + j] = 1e-6 * Math.sin(2 * i * 2 * Math.PI / nx);       // kx = 2
    psi[i * ny + j] = 1e-6 * Math.sin(3 * j * 2 * Math.PI / ny);       // ky = 3
  }
  s.setIC(phi, psi);
  const amp = (c, i, j) => Math.hypot(s.f[c][i * ny + j], s.f[c + 1][i * ny + j]);
  const a0 = [amp(0, 2, 0), amp(2, 0, 3)];
  const T = 5, dt = 0.01;
  for (let n = 0; n < T / dt; n++) s.step(dt);
  const a1 = [amp(0, 2, 0), amp(2, 0, 3)];
  const gPhi = -Math.log(a1[0] / a0[0]) / T, gPsi = -Math.log(a1[1] / a0[1]) / T;
  ok("phi decays at nu k^2", rel(gPhi, nu * 4) < 1e-6, "measured " + gPhi.toPrecision(8) + " vs " + (nu * 4));
  ok("psi decays at eta k^2 (independently)", rel(gPsi, eta * 9) < 1e-6,
     "measured " + gPsi.toPrecision(8) + " vs " + (eta * 9));
}

console.log("4. tearing linear growth rate vs the eigenvalue reference (J physics target)");
// nx = 256 puts the 2/3 dealias cut at |kx| index 85 -- EXACTLY the app's cut at 512 on
// the 4pi box, so this is the resolution the preset actually runs at. (nx = 512 here
// gives the same rate to 4 digits, which is how that was established.)
{
  const Lx = REF.Lx, Ly = REF.Ly, nx = 256, ny = 8;
  const mk = (nu, eta, o) => {
    const T = appIC("tearing", nx, ny, Lx, Ly,
                    { rEqA: 0.1, rEqPsi0: REF.tear.psi0, rEqPert: (o && o.pert) || -3 });
    const s = makeSolver(Object.assign({ nx, ny, Lx, Ly, hyper: 1, nu, eta }, o));
    s.setIC(Float64Array.from(T.phi), Float64Array.from(T.psi));
    return s;
  };
  for (const C of [{ nu: 1e-3, eta: 1e-3, ref: REF.tear.g["1e-3"], t: [30, 80], tag: "eta = nu = 1e-3" },
                   { nu: 3.1623e-3, eta: 3.1623e-3, ref: REF.tear.g["3.1623e-3"], t: [20, 50],
                     tag: "eta = nu = 3.16e-3" },
                   { nu: 1e-3, eta: 1e-2, ref: REF.tear.gPm0p1, t: [12, 32],
                     tag: "eta = 1e-2, nu = 1e-3 (Pm = 0.1)" },
                   // Pm = 0 grows faster AND takes longer to shed the seed's transient
                   // (nothing damps phi, so its eigenfunction settles resistively): a
                   // smaller seed and a later window, still far from saturation
                   { nu: 0, eta: 1e-3, ref: REF.tear.gPm0, t: [40, 80], pert: -5,
                     tag: "eta = 1e-3, nu = 0 (Pm = 0, J2.6)" }]) {
    const s = mk(C.nu, C.eta, { freeze: true, pert: C.pert });
    const g = measure(s, () => s.psiTilde(), 0.02, C.t[0], C.t[1], 250);
    ok("tearing gamma, " + C.tag, rel(g, C.ref) < 0.02,
       "2D pseudospectral " + g.toPrecision(5) + " vs 1D eigenvalue " + C.ref +
       "  (" + (100 * rel(g, C.ref)).toFixed(2) + "%)");
  }
  // ... and the FREE-RUNNING rate with the source OFF, which is what the demo shows with
  // "maintain equilibrium flux" unchecked: the same run without holding psi_eq, whose own
  // resistive diffusion (rate ~ eta/a^2) widens the layer and steadily lowers Delta'.
  // Documented, not a defect -- record how far it parts company with the eigenvalue.
  {
    const s = mk(1e-3, 1e-3, null);
    const g = measure(s, () => s.psiTilde(), 0.02, 30, 80, 250);
    const r = REF.tear.g["1e-3"];
    ok("free-running tearing gamma (source OFF) is degraded by equilibrium decay",
       g > 0.6 * r && g < r,
       "free " + g.toPrecision(4) + " vs frozen/eigen " + r + "  (eta/a^2 = " +
       (1e-3 / (REF.tear.a * REF.tear.a)).toExponential(2) + ", " +
       (100 * (1 - g / r)).toFixed(0) + "% slower over t = 30..80)");
  }

  console.log("4b. maintained equilibrium flux (J2.3)");
  // (a) with the source on and NO seed (a 1e-60 amplitude underflows to exactly 0 in the
  // fp32 plane the app builds), psi_eq must not move at all -- the source is defined as
  // minus its own damping term. Off, it decays at the resistive rate.
  {
    const run = maintain => {
      const s = mk(1e-3, 1e-3, { maintain, pert: -60 });
      const c0 = s.eqColumn();
      for (let n = 0; n < 2500; n++) s.step(0.02);       // t = 50
      const c1 = s.eqColumn();
      let d = 0, mx = 0;
      for (let i = 0; i < c0.length; i++) {
        d = Math.max(d, Math.abs(c1[i] - c0[i])); mx = Math.max(mx, Math.abs(c0[i]));
      }
      return d / mx;
    };
    const on = run(true), off = run(false);
    // "to round-off" holds for THIS plain-RK4 mirror, where rhs(eq) == 0 identically so
    // any RK is exactly stationary. The app's IF stepper exponentiates L while the source
    // rides the nonlinear quadrature: its psi_eq sits at a discrete fixed point offset
    // O((eta k^2 dt)^2) from the true one -- scheme accuracy, negligible at the sech^2
    // profile's low k, but not round-off.
    ok("maintained psi_eq is stationary to round-off over t = 50", on < 1e-12,
       "max relative drift " + on.toExponential(2));
    ok("... and decays without the source", off > 1e-3,
       "max relative drift " + off.toExponential(2) + " (eta/a^2 t = " +
       (50e-3 / (REF.tear.a * REF.tear.a)).toPrecision(3) + ")");
  }
  // (b) the point of the whole exercise: FREE-RUNNING, source on, the growth rate is the
  // frozen-equilibrium eigenvalue again -- 0.0287, not the 0.018 of the decaying layer.
  {
    const s = mk(1e-3, 1e-3, { maintain: true });
    const g = measure(s, () => s.psiTilde(), 0.02, 30, 80, 250);
    const r = REF.tear.g["1e-3"];
    ok("free-running tearing gamma with the source ON is the eigenvalue rate",
       rel(g, r) < 0.05, "maintained " + g.toPrecision(5) + " vs 1D eigenvalue " + r +
       "  (" + (100 * rel(g, r)).toFixed(2) + "%)");
  }

  console.log("4c. the island chart's gamma_fit (FEEDBACK_2026-08-10 item 9)");
  // What the chart actually quotes, end to end, on physics rather than on a synthetic
  // exponential: the app's OWN islandWidth over the app's OWN cut line, sampled off this
  // fp64 solver at the cut card's cadence, then the app's OWN islandFitGamma over its
  // trailing window. Nothing here is compared against itself -- the trace comes from the
  // pseudospectral run 4b just pinned to the 1D eigenvalue, and the target is that
  // eigenvalue. This is also where the ISLAND_FIT_RISE gate is shown to ARM at all: the
  // whole reason it is not MODE_FIT_RISE (see common.js) is that 10 t-units of this stage
  // only rise gamma*DT/2 = 0.14 ln-units of W.
  {
    const appIslandWidth = env.run("function(){ return islandWidth; }");
    const appIslandFit = env.run("function(){ return islandFitGamma; }");
    const s = mk(1e-3, 1e-3, { maintain: true });   // also (re)arms icEq for islandWidth
    const dt = 0.02, every = 25;                    // ~0.5 t-units/sample, the cut throttle
    const ts = [], ws = [];
    for (let n = 0; n * dt <= 80 + 1e-9; n++) {
      if (n % every === 0 && n * dt >= 30) { ts.push(n * dt); ws.push(appIslandWidth(s.cutLine(), ny, Ly)); }
      s.step(dt);
    }
    const gi = appIslandFit(ts, ws), r = REF.tear.g["1e-3"];
    const rise = Math.log(ws[ws.length - 1] / ws[ws.length - 21]);   // the fitted window
    ok("the island chart's gamma_fit reproduces the linear reference rate",
       isFinite(gi) && rel(gi, r) < 0.05,
       "chart gamma_fit " + gi.toPrecision(5) + " vs 1D eigenvalue " + r +
       "  (" + (100 * rel(gi, r)).toFixed(2) + "%)");
    ok("... and its rise gate arms on that window (which MODE_FIT_RISE would not)",
       rise >= 0.05 && rise < 1.0,
       "ln W rose " + rise.toFixed(4) + " over the trailing 10 t-units");
    // W really is psitilde^(1/2): the fit's factor 2 is not a fudge
    const gW = growthRate(ts, ws);
    ok("W grows at HALF the psitilde rate, which is why gamma = 2 x the slope",
       rel(2 * gW, r) < 0.05, "d(ln W)/dt = " + gW.toPrecision(5) + ", x2 = " +
       (2 * gW).toPrecision(5));
  }
}

console.log("5. KH growth and its magnetic suppression (J physics target)");
{
  const Lx = REF.Lx, Ly = REF.Ly, nx = 256, ny = 8, nu = REF.kh.nu;
  const runKH = (b0, freeze, t0, t1, ampf) => {
    const K = appIC("kh", nx, ny, Lx, Ly, { rEqA: 0.05, rEqU0: 1, rEqB0: b0, rEqPert: -4 });
    const s = makeSolver({ nx, ny, Lx, Ly, hyper: 1, nu: nu, eta: nu, freeze });
    s.setIC(Float64Array.from(K.phi), Float64Array.from(K.psi));
    // the k_y = 1 kinetic energy of the perturbation: the flow IS the KH mode
    return measure(s, ampf ? () => ampf(s) : () => Math.sqrt(s.modeEnergy(0)), 0.01, t0, t1, 100);
  };
  const g0 = runKH(0, true, 8, 20), g0f = runKH(0, false, 8, 20);
  ok("KH gamma at b0 = 0", rel(g0, REF.kh.g0) < 0.02,
     "2D pseudospectral " + g0.toPrecision(5) + " vs 1D eigenvalue " + REF.kh.g0 +
     "  (" + (100 * rel(g0, REF.kh.g0)).toFixed(2) + "%)");
  ok("... and free-running (nu/a^2 << gamma, so the layer barely spreads)",
     rel(g0f, REF.kh.g0) < 0.06,
     "free " + g0f.toPrecision(5) + "  (" + (100 * rel(g0f, REF.kh.g0)).toFixed(1) + "%)");
  // ... and the quantity the k_y MODE CHART plots, taken from the app's own extraction
  // code on a cut line built from these fp64 fields: A_u = the m = 1 amplitude of
  // u_x on x = Lx/2. Same run, same window, same tolerance as g0 -- the chart measures the
  // linear mode itself, not a proxy for it, and its LINE (midway between the layers) only
  // offsets the amplitude, never the rate.
  const appModeAmps = env.run("function(){ return modeAmps; }");
  const gcut = runKH(0, true, 8, 20, s => appModeAmps(s.cutLine(), ny).u);
  ok("the app's k_y mode extraction on the cut line grows at the eigenvalue rate",
     rel(gcut, REF.kh.g0) < 0.02,
     "chart quantity " + gcut.toPrecision(5) + " vs 1D eigenvalue " + REF.kh.g0 +
     "  (" + (100 * rel(gcut, REF.kh.g0)).toFixed(2) + "%)");
  // the approach to the threshold, at a b0 where the mode still grows cleanly
  const gh = runKH(0.5, true, 8, 20);
  ok("KH weakens with b0 below the threshold (b0 = 0.5 U0)", rel(gh, REF.kh.gHalf) < 0.03,
     "gamma " + gh.toPrecision(5) + " vs 1D eigenvalue " + REF.kh.gHalf +
     "  (" + (100 * rel(gh, REF.kh.gHalf)).toFixed(2) + "%)");
  // ... and past it, no growth at all. The eigenvalue solve does keep a tiny positive
  // root there (REF.kh.gSup), but it is 70x smaller than gamma(b0=0) and cannot be
  // separated from the seed's own transient in a 20-time-unit window -- what IS testable,
  // and what the demo claims, is that the mode no longer grows.
  const gs = runKH(1.2, true, 8, 20);
  ok("KH is suppressed at b0 = 1.2 U0 (ideal threshold b0 = U0)", gs < 0.02 * g0,
     "gamma " + gs.toPrecision(3) + " (decaying) vs " + g0.toPrecision(3) +
     "; |ratio| " + Math.abs(g0 / gs).toFixed(0) + "x. Eigenvalue there: " + REF.kh.gSup);
}

console.log(bad ? "\n" + bad + " GATE J check(s) FAILED" : "\nall GATE J node checks passed");
process.exit(bad ? 1 : 0);
