# Temporal order of accuracy of the RK schemes, including the integrating-factor
# treatment of dissipation. Both tests refine dt on a FIXED grid and fit the slope of
# log(err) vs log(dt), so they pin the scheme COEFFICIENTS: a wrong alpha/beta/gamma
# changes the slope, not just the constant. (Threshold tests only see the constant --
# that is why the broken lsrk54 A5 showed up as "a bit large" rather than "2nd order".)
#
#   1. Ideal Alfven wave (diss=0, z_diss=0) against the FD-dispersion-exact travelling
#      solution phi = psi = cos(x)cos(y)cos(z + k_eff t). The nonlinear terms vanish
#      identically for phi=psi, so the semi-discrete system is solved EXACTLY by that
#      reference and the only remaining error is the time integration. Nominal orders:
#      3 (lsrk33), 4 (rk44, lsrk54).
#   2. Nonlinear + dissipative. The integrating factor advances the fields by
#      exp(L*dt*gamma) per stage (L = kgrid.lin_L); on a purely linear problem that is exact and
#      therefore says nothing, so the order is only meaningful with the nonlinear terms
#      active. The reference is the same problem integrated with lsrk54 at
#      _NL_T/_NL_REF_STEPS, i.e. 16x finer than the finest sweep point, so the
#      reference's own error is ~16^-4 = 1/65536 of what is being measured.
#
# fp64 only: a 4th-order sweep spans ~4 decades of error, and its fine end would sit on
# the fp32 round-off floor (the same reason test_advection's slow tier is fp64-only).
#
# Both tests were checked against deliberately mutated code. Restoring the historical
# broken lsrk54 A5 takes test 1's fitted order 4.004 -> 0.003. Dropping the integrating
# factor from the low-storage accumulator (`next_delta`) is INVISIBLE to test 1 (4.004,
# unchanged -- the factor is 1 when L=0) and takes test 2's to 1.029: test 2 covers
# something test 1 structurally cannot.
#
# Savio driver: `mpirun -n 4 python tests/test_time_order.py` (nz=8 divisible by 4;
# error norms go through comms.allreduce_max, so multi-rank mode yields global maxima).
from _rmhd_testing import bootstrap

bootstrap()

import math

import jax
import jax.numpy as jnp
import pytest

import jax_rmhd as jr
from _rmhd_testing import alfven_ic, checks, ctx, fit_order, make_state, multimode_ic
from jax_rmhd import diagnostics
from jax_rmhd.comms import allreduce_max
from jax_rmhd.physics import rmhd
from jax_rmhd.run import block_of_steps
from jax_rmhd.timestepping import get_scheme

# jitted exactly like run.py's mpi4jax path; no donation, so states stay alive.
_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))

_NOMINAL = {"lsrk33": 3, "rk44": 4, "lsrk54": 4}
# a fitted order this far below nominal is a real defect, not sweep noise; the same
# 0.5 margin test_advection uses for its spatial fit.
_MARGIN = 0.5

_T_END = 1.0
_NSTEPS = (10, 20, 40, 80)  # dt = _T_END/nsteps: 0.1 -> 0.0125

_NL_T = 0.4
_NL_STEPS = (10, 20, 40, 80)  # dt = _NL_T/nsteps: 0.04 -> 0.005
_NL_REF_STEPS = 80 * 16
# diss=0.2 drains ~39% of the energy over _NL_T while |L|*dt stays <= 0.2 at the
# coarsest point, so the integrating factor matters without the problem going stiff.
_NL_KW = dict(diss=(0.2, 0.2), hyper=1)


def _run(params, kgrid, ic, nsteps, schemestr):
    stepper, scheme = get_scheme(schemestr)
    return _advance(make_state(params, ic=ic), kgrid, params, nsteps, scheme, stepper)


def _global_max_abs(x, params):
    return float(allreduce_max(jnp.max(jnp.abs(x)), params))


# --------------------------------------------------------------- ideal (exact reference)

def _linear_error(schemestr, nsteps):
    """Max |error| vs the FD-dispersion-exact travelling Alfven wave."""
    params, kgrid = ctx(diss=(0.0, 0.0), z_diss=0.0, dt=_T_END / nsteps)
    end = _run(params, kgrid, alfven_ic, nsteps, schemestr)
    t = float(end.t)  # actual accumulated time, never the target
    k_eff = (8.0 * math.sin(params.dz) - math.sin(2.0 * params.dz)) / (6.0 * params.dz)

    def ref_ic(x, y, z):
        phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z + k_eff * t)
        return jnp.stack([phi, phi], axis=0)

    ref = jr.initialize(ref_ic, params)  # same dealias mask as the evolved IC
    return (_global_max_abs(end.fields - ref.fields, params)
            / _global_max_abs(ref.fields, params))


@pytest.mark.fp64
def test_time_order_ideal():
    dts, errs, orders = [_T_END / n for n in _NSTEPS], {}, {}
    for schemestr in _NOMINAL:
        errs[schemestr] = [_linear_error(schemestr, n) for n in _NSTEPS]
        orders[schemestr] = fit_order(dts, errs[schemestr])
        print(f"{schemestr:7s} nominal {_NOMINAL[schemestr]}  fitted {orders[schemestr]:.3f}  "
              f"errs {['%.3e' % e for e in errs[schemestr]]}")
    with checks() as c:
        for schemestr, nominal in _NOMINAL.items():
            c.check(f"{schemestr}: fitted temporal order {orders[schemestr]:.3f} > "
                    f"{nominal - _MARGIN} (nominal {nominal})",
                    orders[schemestr] > nominal - _MARGIN, f"errs={errs[schemestr]}")
            c.check(f"{schemestr}: error decreases monotonically with dt",
                    all(a > b for a, b in zip(errs[schemestr], errs[schemestr][1:])),
                    f"errs={errs[schemestr]}")
        # the sweep must be able to TELL 3rd from 4th order, else the thresholds above
        # would pass on a scheme that had silently lost a stage
        c.check("the sweep separates lsrk33 (3rd) from the 4th-order schemes",
                orders["lsrk33"] < min(orders["rk44"], orders["lsrk54"]) - 0.5,
                f"orders={orders}")


# --------------------------------------------- nonlinear + dissipative (refined reference)

def _nl_reference():
    params, kgrid = ctx(dt=_NL_T / _NL_REF_STEPS, **_NL_KW)
    return _run(params, kgrid, multimode_ic, _NL_REF_STEPS, "lsrk54"), params, kgrid


def _nl_error(schemestr, nsteps, ref):
    params, kgrid = ctx(dt=_NL_T / nsteps, **_NL_KW)
    end = _run(params, kgrid, multimode_ic, nsteps, schemestr)
    return (_global_max_abs(end.fields - ref.fields, params)
            / _global_max_abs(ref.fields, params))


def _nonvacuous(ref, params, kgrid):
    """(energy decay fraction, nonlinear share of the RHS) -- this test is only
    meaningful while both the dissipation and the nonlinear terms do real work."""
    ic = make_state(params, ic=multimode_ic)
    E0 = sum(float(e) for e in diagnostics.energy(ic, kgrid, params))
    E1 = sum(float(e) for e in diagnostics.energy(ref, kgrid, params))
    grads = rmhd.grad(ic, kgrid, params)
    nl = _global_max_abs(rmhd.NonlinearTerm(ic, grads, kgrid, params), params)
    lin = _global_max_abs(rmhd.LinearTerm(ic, grads, kgrid, params), params)
    return 1.0 - E1 / E0, nl / (nl + lin)


@pytest.mark.fp64
def test_time_order_nonlinear_with_dissipation():
    ref, ref_params, ref_kgrid = _nl_reference()
    decay, nl_share = _nonvacuous(ref, ref_params, ref_kgrid)
    dts, errs, orders = [_NL_T / n for n in _NL_STEPS], {}, {}
    for schemestr in _NOMINAL:
        errs[schemestr] = [_nl_error(schemestr, n, ref) for n in _NL_STEPS]
        orders[schemestr] = fit_order(dts, errs[schemestr])
        print(f"{schemestr:7s} nominal {_NOMINAL[schemestr]}  fitted {orders[schemestr]:.3f}  "
              f"errs {['%.3e' % e for e in errs[schemestr]]}")
    with checks() as c:
        # guards against the degenerate versions of this test: an integrating factor
        # that has nothing to do, or a problem the nonlinear terms barely touch (on
        # which the IF would be exact and the order fit would say nothing about it)
        c.check(f"dissipation is doing real work (energy down {100 * decay:.1f}%)",
                decay > 0.20, f"decay={decay}")
        c.check(f"the nonlinear terms are not negligible ({100 * nl_share:.0f}% of the "
                f"ideal RHS at t=0)", nl_share > 0.10, f"nl_share={nl_share}")
        for schemestr, nominal in _NOMINAL.items():
            c.check(f"{schemestr}: fitted order {orders[schemestr]:.3f} > "
                    f"{nominal - _MARGIN} with the integrating factor engaged "
                    f"(nominal {nominal})",
                    orders[schemestr] > nominal - _MARGIN, f"errs={errs[schemestr]}")
            c.check(f"{schemestr}: error decreases monotonically with dt",
                    all(a > b for a, b in zip(errs[schemestr], errs[schemestr][1:])),
                    f"errs={errs[schemestr]}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
