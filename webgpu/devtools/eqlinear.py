#!/usr/bin/env python3
"""Linear eigenvalue reference for the Phase-J equilibrium demos (REFINE_PLAN J).

Independent of the WebGPU app AND of the node pseudospectral reference (checkj.js):
this is a 1D GENERALIZED EIGENVALUE problem in x for a single k_y, not a time
integration, so agreement between the two is a real cross-check.

Linearized 2D RMHD about phi_eq(x), psi_eq(x)  (U_y = phi_eq', B_y = psi_eq'),
perturbations ~ exp(i k y):

  d_t grad^2 phi1 = i k [ B_y grad^2 psi1 - B_y'' psi1 ]
                  - i k [ U_y grad^2 phi1 - U_y'' phi1 ] + nu grad^4 phi1
  d_t psi1        = i k B_y phi1 - i k U_y psi1 + eta grad^2 psi1

with grad^2 = D^2 - k^2. Same sign conventions as the app's bracket kernel
(d_t grad^2 phi = {psi, j} - {phi, w}, d_t psi = -{phi, psi}, u = zhat x grad phi).

x is periodic on [0, Lx) and D is the FOURIER differentiation matrix -- the same
spectral operator the solver uses, so no finite-difference error enters.

Usage:  python3 eqlinear.py [n]        (n = Fourier modes, default 384: the benchmark
                                        table checkj.js quotes, then the survey tables)
"""
import sys

import numpy as np

TWO_PI = 2.0 * np.pi


def fourier_D(n, L):
    """Spectral first/second derivative matrices on a periodic grid of n points."""
    k = TWO_PI / L * np.fft.fftfreq(n, d=1.0 / n)
    I = np.eye(n)
    F = np.fft.fft(I, axis=0)
    D1 = np.real(np.fft.ifft(1j * k[:, None] * F, axis=0))
    D2 = np.real(np.fft.ifft((-(k ** 2))[:, None] * F, axis=0))
    return D1, D2


def growth(Uy, By, k, nu, eta, L, n):
    """Largest-real-part eigenvalue gamma of the linearized system (and the spectrum)."""
    D1, D2 = fourier_D(n, L)
    L2 = D2 - (k ** 2) * np.eye(n)
    L2i = np.linalg.inv(L2)
    dU, dB = np.diag(Uy), np.diag(By)
    dUpp, dBpp = np.diag(D2 @ Uy), np.diag(D2 @ By)
    A11 = L2i @ (-1j * k * (dU @ L2 - dUpp) + nu * (L2 @ L2))
    A12 = L2i @ (1j * k * (dB @ L2 - dBpp))
    A21 = 1j * k * dB
    A22 = -1j * k * dU + eta * L2
    A = np.block([[A11, A12], [A21, A22]])
    w = np.linalg.eigvals(A)
    return w[np.argmax(w.real)], w


# ---------------------------------------------------------------------------
# the two equilibria, in EXACTLY the form common.js builds them
# ---------------------------------------------------------------------------
def kh_profile(x, Lx, U0, a):
    """u_y(x) = U0 [tanh((x-x1)/a) - tanh((x-x2)/a) - 1], x1=Lx/4, x2=3Lx/4."""
    x1, x2 = 0.25 * Lx, 0.75 * Lx
    return U0 * (np.tanh((x - x1) / a) - np.tanh((x - x2) / a) - 1.0)


def tearing_By(x, Lx, psi0, a):
    """psi_eq = psi0 sech^2((x-Lx/2)/a)  =>  B_y = psi_eq'."""
    u = (x - 0.5 * Lx) / a
    return -2.0 * psi0 / a * (1.0 / np.cosh(u) ** 2) * np.tanh(u)


def deltaprime(Lx, a, k, n=20001):
    """Delta' a for psi_eq = psi0 sech^2((x-x_s)/a), by shooting the outer (ideal,
    marginal) equation  psi1'' - [k^2 + B_y''/B_y] psi1 = 0  inward from the box edge.

    Far from the layer B_y ~ e^{-2|x|/a}, so B_y''/B_y -> 4/a^2 and the DECAYING
    solution goes as exp(-kappa|x|) with kappa = sqrt(k^2 + 4/a^2) -- NOT e^{-k|x|};
    that is the boundary condition. The profile is even in x - x_s, so the
    reconnecting (even psi1) solution has psi1'(0-) = -psi1'(0+) and
      Delta' = [psi1'(0+) - psi1'(0-)] / psi1(0) = 2 psi1'(0+) / psi1(0+).
    """
    x0 = 0.5 * Lx                            # box edge, measured from the layer
    h = -x0 / (n - 1)                        # integrate inward, toward x -> 0+

    def by(z):
        v = z / a
        return -2.0 / a * (1.0 / np.cosh(v) ** 2) * np.tanh(v)   # up to psi0 (cancels)

    def F(x):
        d = 1e-4 * a
        B = by(x)
        Bpp = (by(x + d) - 2 * by(x) + by(x - d)) / d ** 2
        return k ** 2 + (Bpp / B if abs(B) > 1e-300 else 0.0)

    def rhs(x, y):
        return np.array([y[1], F(x) * y[0]])

    kap = np.sqrt(k ** 2 + 4.0 / a ** 2)
    y = np.array([1.0, -kap])                # psi1(x0) = 1, psi1'(x0) = -kappa
    x = x0
    for _ in range(n - 1):
        k1 = rhs(x, y)
        k2 = rhs(x + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(x + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(x + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h
    return 2.0 * y[1] / y[0] * a


# The benchmark set checkj.js's REF block quotes, in one place so that "regenerate:
# python3 eqlinear.py" is literally true. n = 384 Fourier modes: doubling it moves every
# entry below by < 1e-6 (run `python3 eqlinear.py 768` to confirm; the eigensolve is
# O(n^3), so 1536 takes minutes and was checked once).
BENCH_N = 384
TEAR = dict(Lx=4 * np.pi, Ly=2 * np.pi, a=0.1 * 4 * np.pi, psi0=1.65)
KH = dict(Lx=4 * np.pi, Ly=2 * np.pi, a=0.05 * 4 * np.pi, U0=1.0, nu=10 ** -3.5)


def bench(n=BENCH_N):
    """Print the checkj.js REF table: tearing at two eta = nu (Pm = 1), at Pm = nu/eta =
    0.1 and at Pm = 0 (inviscid phi -- REFINE_PLAN J2.6 allows it), KH at b0/U0 = 0, 1/2
    and 6/5. Both dissipations are decade-and-a-half slider values (10^-3, 10^-2.5,
    10^-3.5), i.e. exactly what the presets' rDiss offers."""
    print("=" * 74)
    print(f"BENCHMARK TABLE for devtools/checkj.js REF  (n = {n} Fourier modes)")
    print("=" * 74)
    Lx, Ly, a, psi0 = TEAR["Lx"], TEAR["Ly"], TEAR["a"], TEAR["psi0"]
    k = TWO_PI / Ly
    x = np.arange(n) * Lx / n
    By, Uy = tearing_By(x, Lx, psi0, a), np.zeros(n)
    print(f"tearing  Lx={Lx:.5f} Ly={Ly:.5f} a={a:.5f} psi0={psi0}  "
          f"Delta'a={deltaprime(Lx, a, k):.4f}  k a={k*a:.4f}")
    for nu, eta, tag in [(1e-3, 1e-3, 'g["1e-3"]'), (10 ** -2.5, 10 ** -2.5, 'g["3.1623e-3"]'),
                         (1e-3, 1e-2, "gPm0p1"), (0.0, 1e-3, "gPm0")]:
        g, _ = growth(Uy, By, k, nu, eta, Lx, n)
        print(f"  nu={nu:.4e} eta={eta:.4e} (Pm={nu/eta:.2f})   gamma = {g.real:.6f}   -> {tag}")
    Lx, Ly, aK, U0, nu = KH["Lx"], KH["Ly"], KH["a"], KH["U0"], KH["nu"]
    x = np.arange(n) * Lx / n
    Uy = kh_profile(x, Lx, U0, aK)
    print(f"KH       Lx={Lx:.5f} Ly={Ly:.5f} a={aK:.5f} U0={U0} nu=eta={nu:.4e}  "
          f"k a={k*aK:.4f}")
    for b0, tag in [(0.0, "g0"), (0.5, "gHalf"), (1.2, "gSup")]:
        g, _ = growth(Uy, kh_profile(x, Lx, b0, aK), k, nu, nu, Lx, n)
        print(f"  b0={b0:.2f} (b0/U0={b0/U0:.2f})        gamma = {g.real:.6f}   -> {tag}")
    print()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else BENCH_N
    bench(n)
    print("=" * 74)
    print("TEARING  psi_eq = psi0 sech^2((x-Lx/2)/a),  phi_eq = 0")
    print("=" * 74)
    Lx, Ly = 4 * np.pi, 2 * np.pi
    k = TWO_PI / Ly
    a = 0.1 * Lx
    psi0 = 1.65                           # the app's tearing preset default
    x = np.arange(n) * Lx / n
    By = tearing_By(x, Lx, psi0, a)
    Uy = np.zeros(n)
    print(f"Lx={Lx:.4f} Ly={Ly:.4f} k_y={k:.4f} a={a:.4f} (a/Lx={a/Lx:.3f}) "
          f"psi0={psi0:.4f} max|B_y|={np.abs(By).max():.4f}")
    print(f"Delta' a = {deltaprime(Lx, a, k):.4f}   k a = {k*a:.4f}")
    print(f"{'eta=nu':>10} {'gamma':>12} {'gamma*a/vA':>12} {'S=a vA/eta':>12}")
    for eta in [3e-3, 1e-3, 3e-4, 1e-4]:
        g, _ = growth(Uy, By, k, eta, eta, Lx, n)
        print(f"{eta:>10.1e} {g.real:>12.6f} {g.real*a:>12.6f} {a/eta:>12.1f}"
              f"   (Im {g.imag:+.2e})")
    # eta scaling exponent between the two smallest
    g1, _ = growth(Uy, By, k, 1e-3, 1e-3, Lx, n)
    g2, _ = growth(Uy, By, k, 1e-4, 1e-4, Lx, n)
    print(f"d ln gamma / d ln eta = {np.log(g2.real/g1.real)/np.log(1e-4/1e-3):.3f}"
          f"   (FKR constant-psi: 3/5)")

    print()
    print("=" * 74)
    print("KELVIN-HELMHOLTZ  u_y = U0[tanh((x-Lx/4)/a) - tanh((x-3Lx/4)/a) - 1]")
    print("=" * 74)
    Lx, Ly = 4 * np.pi, 2 * np.pi
    k = TWO_PI / Ly
    a = 0.05 * Lx
    U0 = 1.0
    x = np.arange(n) * Lx / n
    Uy = kh_profile(x, Lx, U0, a)
    nu = KH["nu"]
    print(f"Lx={Lx:.4f} Ly={Ly:.4f} k_y={k:.4f} a={a:.4f} (a/Lx={a/Lx:.3f}) "
          f"U0={U0} k a={k*a:.4f} nu=eta={nu:.3e}")
    print(f"{'b0':>8} {'b0/U0':>8} {'Re gamma':>12} {'Im gamma':>12} {'gamma a/U0':>12}")
    for b0 in [0.0, 0.25, 0.5, 0.9, 1.0, 1.2, 1.5]:
        By = kh_profile(x, Lx, b0, a)
        g, _ = growth(Uy, By, k, nu, nu, Lx, n)
        print(f"{b0:>8.2f} {b0/U0:>8.2f} {g.real:>12.6f} {g.imag:>12.6f} "
              f"{g.real*a/U0:>12.6f}")


if __name__ == "__main__":
    main()
