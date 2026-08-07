// GATE K checks (REFINE_PLAN K): the 3D field-line integrator, the along-line sampler
// and the true parallel spectrum, at fp64, in node.
//
//   node checkk.js [dir]        (dir defaults to ..)
//
// What is the app's own code and what is not:
//   * `flSpectrum` / `flHann` / `fftPow2` (section 3) are the REAL common.js functions,
//     called on the booted page -- the windowing, the folding and the [E_u | E_b | H_c]
//     lane algebra are tested as shipped.
//   * `cubeFrame` / `cubeTopXform` (section 0) are likewise the real ones.
//   * the MARCHER is a WGSL compute kernel, which node cannot run, so section 1/2 drive
//     `marchRef` below -- a line-by-line fp64 mirror of rmhd3d's `fieldLine` kernel. It is
//     compared against the ANALYTIC field line of a single-mode psi, never against the
//     kernel, so what it establishes is that the SCHEME (RK2 midpoint over the plane-
//     averaged field, bilinear in plane, periodic wrap, grid-unit stepping) has the
//     accuracy and the orientation claimed. The kernel's own geometry constants are read
//     out of the emitted WGSL and checked against the mirror's (section 0), and the stub
//     boot checks its dispatch and bindings; the picture itself is Alfred's on-device eye.
"use strict";
const path = require("path");
const dir = process.argv[2] || path.join(__dirname, "..");
const env = require("./stubenv")(dir, "rmhd3d.html");

let bad = 0;
const ok = (name, pass, note) => {
  if (!pass) bad++;
  console.log("  " + (pass ? "PASS" : "FAIL") + "  " + name + (note ? "   [" + note + "]" : ""));
};
const rel = (a, b) => Math.abs(a - b) / Math.max(1e-300, Math.abs(b));

// ===========================================================================
// fp64 mirror of the `fieldLine` kernel
// ===========================================================================
// `vol` is laid out exactly as the GPU's realGrads: component c at c*NR + iz*NRS + ix*NY
// + iy, with 0,1 = grad phi and 2,3 = grad psi. Returns the same two outputs the kernel
// writes: pos (box fractions, UNWRAPPED) and smp (u_x, u_y, b_x, b_y).
function marchRef(vol, G, side) {
  const NX = G.nx, NY = G.ny, NZ = G.nz, NRS = NX * NY, NR = NZ * NRS;
  const CX = (G.Lz / NZ) / (G.Lx / NX), CY = (G.Lz / NZ) / (G.Ly / NY);
  const wrap = (i, n) => ((i % n) + n) % n;
  const samp2 = (base, iz, gx, gy) => {
    const fx = Math.floor(gx), fy = Math.floor(gy), tx = gx - fx, ty = gy - fy;
    const x0 = wrap(fx, NX), x1 = wrap(fx + 1, NX), y0 = wrap(fy, NY), y1 = wrap(fy + 1, NY);
    const o = base * NR + iz * NRS;
    const g = (x, y, c) => vol[o + c * NR + x * NY + y];
    const out = [];
    for (let c = 0; c < 2; c++) {
      const lo = g(x0, y0, c) + (g(x1, y0, c) - g(x0, y0, c)) * tx;
      const hi = g(x0, y1, c) + (g(x1, y1, c) - g(x0, y1, c)) * tx;
      out.push(lo + (hi - lo) * ty);
    }
    return out;
  };
  const bp = (iz, gx, gy) => { const g = samp2(2, iz, gx, gy); return [-g[1], g[0]]; };
  const nl = side * side;
  const pos = new Float64Array(nl * NZ * 2), smp = new Float64Array(nl * NZ * 4);
  for (let i = 0; i < nl; i++) {
    let gx = (Math.floor(i / side) + 0.5) / side * NX;
    let gy = (i % side + 0.5) / side * NY;
    for (let iz = 0; iz < NZ; iz++) {
      const o = i * NZ + iz;
      pos[2 * o] = gx / NX; pos[2 * o + 1] = gy / NY;
      const gp = samp2(0, iz, gx, gy), gs = samp2(2, iz, gx, gy);
      smp[4 * o] = -gp[1]; smp[4 * o + 1] = gp[0];
      smp[4 * o + 2] = -gs[1]; smp[4 * o + 3] = gs[0];
      const mx = gx + 0.5 * CX * (-gs[1]), my = gy + 0.5 * CY * gs[0];
      const a = bp(iz, mx, my), b = bp((iz + 1) % NZ, mx, my);
      gx += CX * 0.5 * (a[0] + b[0]);
      gy += CY * 0.5 * (a[1] + b[1]);
    }
  }
  return { pos, smp, nl };
}

// the real-space gradient volume of analytic potentials: f(x, y, z) -> (d_x, d_y) of phi
// and psi, sampled on the grid exactly as the inverse transform of a resolved mode does
function buildVol(G, dphi, dpsi) {
  const NRS = G.nx * G.ny, NR = G.nz * NRS;
  const v = new Float64Array(4 * NR);
  for (let iz = 0; iz < G.nz; iz++) {
    const z = iz * G.Lz / G.nz;
    for (let ix = 0; ix < G.nx; ix++) {
      const x = ix * G.Lx / G.nx;
      for (let iy = 0; iy < G.ny; iy++) {
        const y = iy * G.Ly / G.ny, m = iz * NRS + ix * G.ny + iy;
        const a = dphi(x, y, z), b = dpsi(x, y, z);
        v[m] = a[0]; v[NR + m] = a[1]; v[2 * NR + m] = b[0]; v[3 * NR + m] = b[1];
      }
    }
  }
  return v;
}
const L2P = 2 * Math.PI;

console.log("0. the box projection and the kernel's geometry constants");
{
  // cubeTopXform must still be the pre-K expression (the p00 / p10 / p01 corners of face 0
  // in canvas pixels): the frame was factored out, the numbers must not have moved.
  const g = env.run(`function(){
    return { q: Array.from(cubeQuads()), S: VEC_SIZE, F: cubeFrame(), T: cubeTopXform() };
  }`);
  const px = i => [(g.q[i] + 1) * 0.5 * g.S, (1 - g.q[i + 1]) * 0.5 * g.S];
  const o = px(0), a = px(2), b = px(4);
  const old = { ox: o[0], oy: o[1], ax: a[0] - o[0], ay: a[1] - o[1],
                bx: b[0] - o[0], by: b[1] - o[1] };
  let dt = 0;
  for (const k in old) dt = Math.max(dt, Math.abs(old[k] - g.T[k]));
  ok("cubeTopXform is unchanged by the refactor", dt < 1e-12, "max |delta| = " + dt.toExponential(2));
  // ... and the box frame must reproduce EVERY projected face corner: face 0 (z=1) spans
  // (x, y), face 1 (x=1) spans (z, y), face 2 (y=1) spans (z, x)
  const F = g.F, at = (x, y, z) => [F.ox + x * F.ax + y * F.bx + z * F.cx,
                                    F.oy + x * F.ay + y * F.by + z * F.cy];
  const corner = [(u, v) => [u, v, 1], (u, v) => [1, v, u], (u, v) => [v, 1, u]];
  let df = 0;
  for (let f = 0; f < 3; f++) {
    for (const [u, v, i] of [[0, 0, 0], [1, 0, 2], [0, 1, 4], [1, 1, 6]]) {
      const p = at(...corner[f](u, v)), q = px(12 * f + i);
      df = Math.max(df, Math.abs(p[0] - q[0]), Math.abs(p[1] - q[1]));
    }
  }
  // (cubeQuads is a Float32Array, so "exact" here means to fp32 -- sub-milli-pixel)
  ok("the box frame reproduces all 12 projected face corners", df < 1e-3,
     "max |delta| = " + df.toExponential(2) + " px");
  // the marcher's grid-unit step factors, as the kernel actually emits them
  const P = { nx: 128, ny: 128, nz: 64, Lx: L2P, Ly: L2P, Lz: 4 * L2P,
              diss: 1e-11, hyper: 4, zdiss: 1e-4, fshell: [1, 3] };
  const src = env.run(`function(P){ return buildShaders(Object.assign({ nx: P.nx, ny: P.ny, nz: P.nz },
                                                                      makeGrid(P))).fieldLine; }`, P);
  const con = n => parseFloat((new RegExp("const " + n + ": f32 = ([^;]+);").exec(src) || [])[1]);
  const want = [(P.Lz / P.nz) / (P.Lx / P.nx), (P.Lz / P.nz) / (P.Ly / P.ny)];
  ok("emitted CX / CY are dz/dx and dz/dy", rel(con("CX"), want[0]) < 1e-9 && rel(con("CY"), want[1]) < 1e-9,
     "CX = " + con("CX") + ", CY = " + con("CY") + " on a " + P.nx + "^2 x " + P.nz + ", Lz = 8pi box");
}

console.log("1. the integrator against an analytic b_perp (K.1)");
{
  // psi = A cos(kx x) sin(kz z) with phi = 0:
  //   b = zhat x grad psi = (0, -A kx sin(kx x) sin(kz z))
  // so a line seeded at (x0, y0) keeps x = x0 exactly and
  //   y(z) = y0 + (A kx sin(kx x0) / kz) (cos(kz z) - 1),
  // a pure sinusoidal displacement. The MIRROR image of the same statement -- psi =
  // A sin(ky y) sin(kz z), which moves the line in x instead -- pins the sign convention
  // of zhat x grad = (-d_y, +d_x) on the other component.
  const A = 0.2, kx = 1, kz = 1;
  const run = (nz, axis) => {
    const G = { nx: 64, ny: 64, nz, Lx: L2P, Ly: L2P, Lz: L2P };
    const dpsi = axis === "y"
      ? (x, y, z) => [-A * kx * Math.sin(kx * x) * Math.sin(kz * z), 0]
      : (x, y, z) => [0, A * kx * Math.cos(kx * y) * Math.sin(kz * z)];
    const M = marchRef(buildVol(G, () => [0, 0], dpsi), G, 4);
    // seeds land exactly on grid points here (nx / side = 16 is even), so the bilinear
    // interpolation is EXACT and what is left is purely the RK2 error in z
    let eMove = 0, eStill = 0;
    for (let i = 0; i < M.nl; i++) {
      const u0 = M.pos[2 * i * nz], v0 = M.pos[2 * i * nz + 1];
      // y case: dy/dz = -A kx sin(kx x0) sin(kz z);  x case: dx/dz = -A kx cos(kx y0) sin(kz z)
      const amp = axis === "y" ? A * kx * Math.sin(kx * u0 * L2P) / kz
                               : A * kx * Math.cos(kx * v0 * L2P) / kz;
      for (let iz = 0; iz < nz; iz++) {
        const z = iz * G.Lz / nz, o = 2 * (i * nz + iz);
        const exact = (axis === "y" ? v0 : u0) + amp * (Math.cos(kz * z) - 1) / L2P;
        eMove = Math.max(eMove, Math.abs(M.pos[o + (axis === "y" ? 1 : 0)] - exact));
        eStill = Math.max(eStill, Math.abs(M.pos[o + (axis === "y" ? 0 : 1)] - (axis === "y" ? u0 : v0)));
      }
    }
    return [eMove, eStill];
  };
  for (const axis of ["y", "x"]) {
    const e = [16, 32, 64, 128].map(nz => run(nz, axis));
    ok("line displacement in " + axis + " matches the analytic solution",
       e[3][0] < 1e-5, "max |delta| (box fractions) at nz=128: " + e[3][0].toExponential(2));
    ok("... and the transverse coordinate is EXACTLY fixed (b has no component there)",
       e[3][1] === 0, "max |delta| = " + e[3][1]);
    const p = [0, 1, 2].map(i => Math.log2(e[i][0] / e[i + 1][0]));
    ok("... at second order in dz (RK2 midpoint)", p.every(v => v > 1.9 && v < 2.1),
       "errors " + e.map(v => v[0].toExponential(2)).join(" -> ") +
       ", orders " + p.map(v => v.toFixed(3)).join(", "));
  }
  // a line whose seed is NOT on a grid point still converges -- to the bilinear floor
  {
    const nz = 256, G = { nx: 256, ny: 256, nz, Lx: L2P, Ly: L2P, Lz: L2P };
    const M = marchRef(buildVol(G, () => [0, 0],
      (x, y, z) => [-A * kx * Math.sin(kx * x) * Math.sin(kz * z), 0]), G, 6);
    let e = 0;
    for (let i = 0; i < M.nl; i++) {
      const u0 = M.pos[2 * i * nz], v0 = M.pos[2 * i * nz + 1];
      const amp = A * kx * Math.sin(kx * u0 * L2P) / kz;
      for (let iz = 0; iz < nz; iz++) {
        const z = iz * G.Lz / nz;
        e = Math.max(e, Math.abs(M.pos[2 * (i * nz + iz) + 1] - (v0 + amp * (Math.cos(kz * z) - 1) / L2P)));
      }
    }
    ok("off-grid seeds (6x6 on 256^2) stay within the bilinear floor", e < 2e-5,
       "max |delta| = " + e.toExponential(2) + " box fractions, ~ (k dx)^2/8 = " +
       (Math.pow(kx * L2P / 256, 2) / 8).toExponential(2));
  }
}

console.log("2. the along-line sampler (K.3)");
{
  // phi = P sin(kx x) sin(ky y) rides on top of the case-1 psi, so the line MOVES while
  // the sampled u varies along it: what is checked is that smp records u and b at the
  // position the polyline reports, to bilinear accuracy.
  const A = 0.3, P0 = 0.7, kx = 1, ky = 1, kz = 1, nz = 64;
  const G = { nx: 256, ny: 256, nz, Lx: L2P, Ly: L2P, Lz: L2P };
  const dphi = (x, y, z) => [P0 * kx * Math.cos(kx * x) * Math.sin(ky * y),
                             P0 * ky * Math.sin(kx * x) * Math.cos(ky * y)];
  const dpsi = (x, y, z) => [-A * kx * Math.sin(kx * x) * Math.sin(kz * z), 0];
  const M = marchRef(buildVol(G, dphi, dpsi), G, 6);
  let eu = 0, eb = 0, mvd = 0;
  for (let i = 0; i < M.nl; i++) {
    for (let iz = 0; iz < nz; iz++) {
      const o = i * nz + iz;
      const x = M.pos[2 * o] * L2P, y = M.pos[2 * o + 1] * L2P, z = iz * G.Lz / nz;
      const a = dphi(x, y, z), b = dpsi(x, y, z);
      eu = Math.max(eu, Math.abs(M.smp[4 * o] - (-a[1])), Math.abs(M.smp[4 * o + 1] - a[0]));
      eb = Math.max(eb, Math.abs(M.smp[4 * o + 2] - (-b[1])), Math.abs(M.smp[4 * o + 3] - b[0]));
      mvd = Math.max(mvd, Math.abs(M.pos[2 * o + 1] - M.pos[2 * i * nz + 1]));
    }
  }
  ok("u = zhat x grad phi sampled ON the line", eu < 1e-3 * P0,
     "max abs err " + eu.toExponential(2) + " vs |u| ~ " + P0);
  ok("b = zhat x grad psi sampled ON the line", eb < 1e-3 * A,
     "max abs err " + eb.toExponential(2) + " vs |b| ~ " + A);
  ok("... and the line really wandered while being sampled", mvd > 0.02,
     "max transverse excursion " + mvd.toFixed(4) + " box fractions");
}

console.log("3. window and spectrum on synthetic signals (K.3)");
{
  const call = (name, ...a) => env.run("function(n){ return " + name + "(...arguments[1]); }", name, a);
  // --- the Hann window -------------------------------------------------------
  const H = env.run("function(n){ const h = flHann(n); return { w: Array.from(h.w), w2: h.w2 }; }", 64);
  let ew = 0;
  for (let j = 0; j < 64; j++) ew = Math.max(ew, Math.abs(H.w[j] - 0.5 * (1 - Math.cos(2 * Math.PI * j / 64))));
  ok("flHann is the periodic Hann window", ew < 1e-15, "max |delta| = " + ew.toExponential(2));
  ok("... with mean square exactly 3/8", rel(H.w2, 3 / 8) < 1e-15, "w2 = " + H.w2);

  // --- peak recovery + Parseval ---------------------------------------------
  // u_x = A cos(2 pi m j / nz + phase), one phase per line, everything else zero. Then
  // E_u must peak at bin m and, since m >= 3 keeps the signal's second harmonic clear of
  // the window's own (w^2 has harmonics 0, 1, 2 only), sum_b E_u = <u_x^2>/2 = A^2/4.
  const nz = 64, nzb = nz >> 1, nl = 3, A = 1.7, m = 5;
  const mk = f => {
    const s = new Float64Array(nl * nz * 4);
    for (let l = 0; l < nl; l++) {
      for (let j = 0; j < nz; j++) f(s, 4 * (l * nz + j), j, l);
    }
    return s;
  };
  const smp = mk((s, o, j, l) => { s[o] = A * Math.cos(2 * Math.PI * m * j / nz + l); });
  const S = call("flSpectrum", smp, nl, nz);
  let pk = 1;
  for (let b = 1; b <= nzb; b++) if (S[b - 1] > S[pk - 1]) pk = b;
  let sum = 0;
  for (let b = 1; b <= nzb; b++) sum += S[b - 1];
  ok("E_u(k_par) peaks at the mode's own bin", pk === m, "peak bin " + pk + ", expected " + m);
  // (flSpectrum returns fp32, as every other binned spectrum here does)
  ok("... and the bins sum to <u^2>/2 (Parseval through the window)", rel(sum, A * A / 4) < 1e-6,
     "sum " + sum.toPrecision(10) + " vs " + (A * A / 4).toPrecision(10));
  let eb = 0;
  for (let b = 1; b <= nzb; b++) eb = Math.max(eb, Math.abs(S[nzb + b - 1]), Math.abs(S[2 * nzb + b - 1]));
  ok("... with the E_b and H_c lanes exactly zero for a purely kinetic signal", eb === 0,
     "max |lane| = " + eb);

  // --- the lane algebra: E+- = E_u + E_b +- H_c ------------------------------
  const rho = 0.4;
  const sm2 = mk((s, o, j, l) => {
    const u = A * Math.cos(2 * Math.PI * m * j / nz + l);
    s[o] = u; s[o + 2] = rho * u;                 // b aligned with u: z+ = (1+rho) u
  });
  const S2 = call("flSpectrum", sm2, nl, nz);
  let su = 0, sb = 0, sh = 0;
  for (let b = 1; b <= nzb; b++) { su += S2[b - 1]; sb += S2[nzb + b - 1]; sh += S2[2 * nzb + b - 1]; }
  ok("E_b = rho^2 E_u and H_c = 2 rho E_u for b = rho u",
     rel(sb, rho * rho * su) < 1e-6 && rel(sh, 2 * rho * su) < 1e-6,
     "E_u " + su.toPrecision(8) + ", E_b " + sb.toPrecision(8) + ", H_c " + sh.toPrecision(8));
  ok("... so sum E+ = <|z+|^2>/2", rel(su + sb + sh, Math.pow(1 + rho, 2) * A * A / 4) < 1e-6,
     "E+ " + (su + sb + sh).toPrecision(10) + " vs " + (Math.pow(1 + rho, 2) * A * A / 4).toPrecision(10));

  // --- why the window is there ----------------------------------------------
  // A field line leaves the box perpendicularly displaced, so the sampled signal has a
  // step between its two ends. Model that with a ramp on top of the mode and compare the
  // high-bin contamination with and without the window (the unwindowed spectrum is built
  // here from the same fftPow2, so only the window differs).
  const ramp = mk((s, o, j, l) => { s[o] = A * Math.cos(2 * Math.PI * m * j / nz + l) + 3 * A * j / nz; });
  const Sr = call("flSpectrum", ramp, nl, nz);
  const raw = env.run(`function(smp, nl, nz){
    const nzb = nz >> 1, out = new Float64Array(nzb);
    for (let l = 0; l < nl; l++) {
      const re = new Float64Array(nz), im = new Float64Array(nz);
      for (let j = 0; j < nz; j++) re[j] = smp[4 * (l * nz + j)];
      fftPow2(re, im, -1);
      for (let b = 1; b <= nzb; b++) {
        out[b - 1] += 0.5 * (2 * b === nz ? 1 : 2) * (re[b] * re[b] + im[b] * im[b]) / (nz * nz * nl);
      }
    }
    return Array.from(out);
  }`, ramp, nl, nz);
  const tail = a => { let s = 0; for (let b = nzb / 2; b <= nzb; b++) s += a[b - 1]; return s; };
  ok("the Hann window suppresses the non-periodic signal's high-k_par leakage",
     tail(Sr) < 0.02 * tail(raw),
     "tail power windowed " + tail(Sr).toExponential(2) + " vs unwindowed " + tail(raw).toExponential(2) +
     " (" + (tail(raw) / tail(Sr)).toFixed(0) + "x)");
  // the ramp itself is a genuine low-k_par signal and OWNS bin 1; what matters is that
  // it does not contaminate the mode's own bin
  ok("... and the mode's own bin is untouched by the ramp", rel(Sr[m - 1], S[m - 1]) < 0.05,
     "E_u(bin " + m + ") ramped " + Sr[m - 1].toExponential(4) + " vs clean " + S[m - 1].toExponential(4));

  // --- end to end: marcher -> sampler -> spectrum ----------------------------
  // psi = 0 (straight vertical lines) and phi = P sin(ky y) cos(kz z), so u_x along every
  // line is a pure cosine of kz z: the true parallel spectrum is one bin, and its value is
  // the line-ensemble mean of <u_x^2>/2.
  {
    const P0 = 0.9, ky = 1, mz = 3, side = 4, nzE = 32;
    const G = { nx: 64, ny: 64, nz: nzE, Lx: L2P, Ly: L2P, Lz: L2P };
    const kzE = mz * 2 * Math.PI / G.Lz;
    const M = marchRef(buildVol(G,
      (x, y, z) => [0, P0 * ky * Math.cos(ky * y) * Math.cos(kzE * z)], () => [0, 0]), G, side);
    const SE = call("flSpectrum", M.smp, M.nl, nzE);
    const nb = nzE >> 1;
    let p = 1, tot = 0, want = 0;
    for (let b = 1; b <= nb; b++) { if (SE[b - 1] > SE[p - 1]) p = b; tot += SE[b - 1]; }
    for (let i = 0; i < M.nl; i++) {
      const a = P0 * ky * Math.cos(ky * M.pos[2 * i * nzE + 1] * L2P);
      want += a * a / 4 / M.nl;
    }
    ok("marched + sampled + binned: the mode lands in its own k_par bin", p === mz,
       "peak bin " + p + " of " + nb + ", expected " + mz);
    ok("... at the ensemble-mean <u^2>/2", rel(tot, want) < 1e-6,
       "sum " + tot.toPrecision(10) + " vs " + want.toPrecision(10));
  }
}

if (env.fails.length) { bad += env.fails.length; console.log("STUB FAILURES:", env.fails.join("; ")); }
console.log(bad ? "\n" + bad + " GATE K check(s) FAILED" : "\nall GATE K node checks passed");
process.exit(bad ? 1 : 0);
