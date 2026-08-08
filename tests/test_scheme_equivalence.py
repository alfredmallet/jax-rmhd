# Timestepping scheme coverage.
#
#   1. lsrk_scan=True (lax.scan stage loop) vs =False (unrolled) agree to round-off.
#      They are NOT bitwise identical at fp64: XLA fuses the scan body and the
#      unrolled 3/5-stage graphs differently, and how much they diverge is
#      machine-dependent (max|diff| ~2e-15 lsrk33 / ~7e-15 lsrk54 after 20 steps
#      under jax 0.6.2/CPU). Asserted at rel < 1e-13 (fp64), which still catches
#      any structural divergence between the two loops.
#   2. rk44 / lsrk54 (never exercised anywhere else) integrate the exact Alfven
#      solution. IC phi = psi = cos(x)cos(y)cos(z) is an exact solution of the FULL
#      RMHD system (nonlinear terms vanish identically -- see test_dissipation's
#      header) with diss=0/z_diss=0, so the semi-discrete (z-FD) system is solved
#      EXACTLY by phi = psi = cos(x)cos(y)cos(z + k_eff*t), where k_eff is the
#      4th-order centered-difference dispersion factor
#          k_eff = (8 sin(k dz) - sin(2 k dz)) / (6 dz),   k = 1
#      (same factor test_z_stencils pins). Against that reference the only error is
#      the RK time integration, so 4th-order schemes must sit orders of magnitude
#      below 3rd-order lsrk33 at dt=0.01.
#   3. get_scheme: known names resolve, unknown names raise ValueError.
#
# mpirun-safe: ICs/references built via jr.initialize (rank-local z slabs), errors
# asserted per rank, nz=8 divisible by 4. Savio: `mpirun -n 4 python tests/...py`.
from _rmhd_testing import alfven_ic, bootstrap, checks, ctx, make_state

bootstrap()

import math

import jax
import jax.numpy as jnp
import numpy as np

import taranis as jr
from taranis import _precision
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme, lsrk_advance, rk_advance

# jitted exactly like run.py's mpi4jax path (params/nblock/scheme/stepper static);
# no donation, so input states stay alive (they are fresh anyway).
_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))


def _run(params, kgrid, ic, nsteps, schemestr):
    stepper, scheme = get_scheme(schemestr)
    end = _advance(make_state(params, ic=ic), kgrid, params, nsteps, scheme, stepper)
    return end


def test_lsrk_scan_vs_unrolled_bitwise():
    # Two Parameters differing only in lsrk_scan -- each gets its OWN kgrid via ctx
    # (never reuse a kgrid across different Parameters).
    tol = 1e-13 if _precision.precision == "64" else 1e-6
    with checks() as c:
        for schemestr in ("lsrk33", "lsrk54"):
            p_scan, kg_scan = ctx(lsrk_scan=True)
            p_unroll, kg_unroll = ctx(lsrk_scan=False)
            end_scan = _run(p_scan, kg_scan, alfven_ic, 20, schemestr)
            end_unroll = _run(p_unroll, kg_unroll, alfven_ic, 20, schemestr)
            fs, fu = np.asarray(end_scan.fields), np.asarray(end_unroll.fields)
            rel = float(np.max(np.abs(fs - fu)) / np.max(np.abs(fu)))
            dt_rel = abs(float(end_scan.t) - float(end_unroll.t)) / float(end_unroll.t)
            c.check(f"{schemestr}: lsrk_scan True/False agree to round-off "
                    f"(rel {rel:.2e} < {tol:.0e}; see header for why not bitwise)",
                    rel < tol)
            c.check(f"{schemestr}: end times agree to round-off "
                    f"(rel {dt_rel:.2e})", dt_rel < tol)


def _alfven_error(schemestr, nsteps=50):
    """Max relative error vs the FD-dispersion-exact traveling Alfven wave."""
    params, kgrid = ctx(diss=(0.0, 0.0), z_diss=0.0)
    end = _run(params, kgrid, alfven_ic, nsteps, schemestr)
    t_act = float(end.t)  # never the target time (t accumulates per stage)
    k_eff = (8.0 * math.sin(params.dz) - math.sin(2.0 * params.dz)) / (6.0 * params.dz)

    def ref_ic(x, y, z):
        phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z + k_eff * t_act)
        return jnp.stack([phi, phi], axis=0)

    ref = jr.initialize(ref_ic, params)
    err = float(jnp.max(jnp.abs(end.fields - ref.fields))
                / jnp.max(jnp.abs(ref.fields)))
    return err, t_act


def test_rk44_lsrk54_integrate_alfven_wave():
    # Per-step local error ~ (w*dt)^(p+1): at dt=0.01, t=0.5 the 4th-order schemes
    # land near 1e-10 (fp64) and lsrk33 near 1e-8 -- tolerances carry >=100x margin.
    # fp32 sits on the round-off floor of the fft/initialize pipeline instead
    # (~1e-6), so all schemes share one loose fp32 tolerance.
    x64 = _precision.precision == "64"
    tols = dict(rk44=1e-8 if x64 else 1e-4,
                lsrk54=1e-8 if x64 else 1e-4,
                lsrk33=1e-6 if x64 else 1e-4)
    with checks() as c:
        errs = {}
        for schemestr, tol in tols.items():
            err, t_act = _alfven_error(schemestr)
            errs[schemestr] = err
            c.check(f"{schemestr}: Alfven-wave error {err:.2e} < {tol:.0e} "
                    f"at t={t_act:.3f}", err < tol)
        if x64:
            # the 4th-order schemes must actually beat the 3rd-order one
            c.check("rk44 and lsrk54 both beat lsrk33 (order visible in practice)",
                    errs["rk44"] < errs["lsrk33"] and errs["lsrk54"] < errs["lsrk33"],
                    f"errs={errs}")


def test_get_scheme_registry():
    with checks() as c:
        stepper44, scheme44 = get_scheme("rk44")
        c.check("rk44 resolves to rk_advance with no LSRK coefficient table",
                stepper44 is rk_advance and scheme44 is None)
        for name, nstages in (("lsrk33", 3), ("lsrk54", 5)):
            stepper, scheme = get_scheme(name)
            c.check(f"{name} resolves to lsrk_advance with {nstages} stages",
                    stepper is lsrk_advance and len(scheme.alphas) == nstages)
        try:
            get_scheme("nope")
            raised = ""
        except ValueError as e:
            raised = str(e)
        c.check("get_scheme('nope') raises ValueError naming the available schemes",
                "nope" in raised and "lsrk33" in raised)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
