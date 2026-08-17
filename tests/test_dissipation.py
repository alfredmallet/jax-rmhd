# Perpendicular hyperdissipation decay law.
# IC phi = psi = cos(x)cos(y)cos(z) lives entirely at k_perp^2 = 2 (kx=+-1, ky=1),
# so with hyper=1 the integrating factor exp(kgrid.lin_L*dt) = exp(-diss*ksq*dt)
# damps each field coefficient by exp(-2*diss*t) and the (quadratic) energy by
# exp(-4*diss*t). The law is exact, not asymptotic: the nonlinear terms vanish
# identically for this IC (grad_perp^2 phi = -2 phi, so {phi, grad_perp^2 phi} = 0,
# and phi = psi kills the remaining brackets), the linear Alfven term conserves
# energy (it only swaps d/dz between the two equations, preserving phi = psi), and
# dissipation commutes with d/dz (diagonal in k_perp vs acting only in z) -- so the
# only deviations are RK3 amplitude error (~ (omega*dt)^4 per step, negligible at
# dt=0.01) and round-off. z_diss is set to 0.0 explicitly: its DEFAULT is 0.25, and
# rmhd.FDLinearTerm applies z_diss*(dz/2)^4 * d4/dz4 as an RHS term, which would add
# spurious decay of the kz=1 envelope on top of the perp prediction.
# pytest: single-process (stub). Savio driver: `mpirun -n 4 python
# tests/test_dissipation.py` (nz=8 divisible by 4; energies are global via the
# allreduce inside diagnostics.energy -> perp_inner_product).
# Debug plot (E_end/E_start vs exp(-4*diss*t)): set RMHD_TEST_PLOTS=1; saves
# rmhd_test_dissipation.png to the cwd.
from _rmhd_testing import alfven_ic, bootstrap, checks, ctx, make_state

bootstrap()

import math
import os

import numpy as np

from taranis import _precision, diagnostics
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

# Fixed dt (ctx default adaptive_timestep=False, dt=0.01) + direct block_of_steps
# gives an exact step count: t_end = 0.5, decay factors span 1 .. e^-2 ~ 0.135 --
# well above round-off at both precisions, nowhere near denormal.
_DISS = (0.0, 0.2, 0.5, 1.0)
_NSTEPS = 50


def _decay_run(d):
    """(E_start, E_end, t_actual) for a 50-step run at diss=(d,d).

    Each d gets its own Parameters+kgrid via ctx (never reuse a kgrid across
    different Parameters); ctx caching means one jit/scan retrace per d --
    expected and cheap at 16x16x8. State is built fresh (never cached: donation
    consumes states). ctx results are shared -- not mutated here.
    """
    params, kgrid = ctx(diss=(d, d), z_diss=0.0)
    state = make_state(params, ic=alfven_ic)
    e_start = sum(float(e) for e in diagnostics.energy(state, kgrid, params))
    stepper, scheme = get_scheme("lsrk33")
    end_state = block_of_steps(state, kgrid, params, _NSTEPS, scheme, stepper)
    e_end = sum(float(e) for e in diagnostics.energy(end_state, kgrid, params))
    # float(end_state.t), never the target time: t accumulates per RK stage.
    return e_start, e_end, float(end_state.t)


def test_dissipation_decay_law():
    log_ratios, ts = [], []
    for d in _DISS:
        e_start, e_end, t_actual = _decay_run(d)
        log_ratios.append(math.log(e_end / e_start))
        ts.append(t_actual)

    # The integrating factor applies the decay exactly; the dominant deviation is a
    # small d-INDEPENDENT offset in log-energy from RK3 amplitude error on the
    # Alfven oscillation (~ -4e-8 at fp64, also visible as the d=0 drift), plus
    # fp32 round-off. Observed per-diss rel err: <=1e-7 (fp64) / <=1.3e-5 (fp32),
    # worst at d=0.2 (smallest denominator). The slope fit cancels the additive
    # offset exactly: observed slope rel err ~1e-15 (fp64) / ~2e-7 (fp32). The
    # plan's 1%/2% tolerances hold with >=800x margin at both precisions, so they
    # are kept precision-independent.
    ratio_tol, slope_tol = 0.01, 0.02
    with checks() as c:
        for d, lr, t_actual in zip(_DISS, log_ratios, ts):
            if d == 0.0:
                continue  # expected log-ratio is 0 -- covered by the conservation test
            expected = -4.0 * d * t_actual
            rel = abs(lr - expected) / abs(expected)
            c.check(f"diss={d}: log(E_end/E_start) within 1% of -4*diss*t_actual",
                    rel < ratio_tol,
                    f"log_ratio={lr:.6e}, expected={expected:.6e}, rel={rel:.3e}")
        slope = float(np.polyfit(np.asarray(_DISS), np.asarray(log_ratios), 1)[0])
        expected_slope = -4.0 * float(np.mean(ts))  # per-run t_actual agree to round-off
        rel = abs(slope - expected_slope) / abs(expected_slope)
        c.check("fitted slope of log(E_end/E_start) vs diss within 2% of -4*t_actual",
                rel < slope_tol,
                f"slope={slope:.6e}, expected={expected_slope:.6e}, rel={rel:.3e}")

    if os.environ.get("RMHD_TEST_PLOTS") == "1":
        # Debugging aid only (MPLBACKEND=Agg set by bootstrap); lazy import keeps
        # matplotlib out of the normal test path.
        import matplotlib.pyplot as plt
        d_arr = np.asarray(_DISS)
        plt.figure()
        plt.semilogy(d_arr, np.exp(np.asarray(log_ratios)), "o", label="measured")
        plt.semilogy(d_arr, np.exp(-4.0 * d_arr * np.mean(ts)), "-", label="exp(-4*diss*t)")
        plt.xlabel(r"$\eta,\nu$")
        plt.ylabel(r"$E_{end}/E_{start}$")
        plt.legend()
        plt.savefig("rmhd_test_dissipation.png")
        plt.close()


def test_zero_dissipation_conserves_energy():
    # diss=(0,0), z_diss=0: the linear operator is identically zero, the nonlinear terms vanish
    # analytically for this IC, and the Alfven term conserves energy -- the drift
    # is RK3 amplitude error on the oscillation plus round-off. Observed
    # |E1-E0|/E0: ~4e-8 (fp64) / ~1.2e-7 (fp32). The plan's 1e-6 is kept at fp64
    # (25x margin); fp32 gets 1e-5 -- the observed drift technically fits 1e-6 too,
    # but only 8x above it, too tight against cross-machine fp32 fft variation
    # (same precedent as test_z_stencils: looser fp32 tolerance, values commented).
    e_start, e_end, _ = _decay_run(0.0)
    tol = 1e-6 if _precision.precision == "64" else 1e-5
    rel = abs(e_end - e_start) / abs(e_start)
    with checks() as c:
        c.check(f"diss=0 run conserves energy to {tol:.0e} (relative)",
                rel < tol, f"E0={e_start:.10e}, E1={e_end:.10e}, rel={rel:.3e}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
