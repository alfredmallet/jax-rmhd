# Alfven-wave z-advection convergence study, asserted (docs/TESTING_PLAN.md Phase 4).
#
# phi = psi = cos(x)cos(y)cos(z) is a single z+ Alfven wave at vA=1; the exact
# linear (ideal, unforced) solution is cos(x)cos(y)cos(z+t). Fixed dt=1e-3 for an
# exact step count to t~0.1 via a direct block_of_steps call (simulate/simulate_scan
# overshoot t_end), then L1/L2 relative error of the k-space fields against the
# exact solution evaluated at float(end_state.t) -- never the target time. nx=ny=16
# is plenty: the IC only carries kx=ky=1, and the perpendicular part is spectral.
#
# Tiers:
#   - fast (default, seconds): nz in _NZ_FAST, fitted z-convergence order > 3.5
#     and monotone-decreasing errors. 3.5 catches "silently 2nd order" without
#     tripping on dt contamination; the sharp 4th-order stencil assertion lives in
#     test_z_stencils.py.
#   - @slow full study (pytest: -m slow --runslow; script: RMHD_RUNSLOW=1): nz in
#     _NZ_SLOW, order > 3.5 on the coarse triple only; the fine end flattens toward
#     the fixed-dt O(dt^3) floor (expected), so only error non-increase is asserted
#     there. fp64-only -- see the comment on the test.
#
# Savio driver: `mpirun -n 4 python tests/test_advection.py` (all nz below are
# divisible by 4; every error reduction goes through comms.allreduce_sum, so
# multi-rank script mode produces global norms).
from _rmhd_testing import bootstrap

bootstrap()

import jax
import jax.numpy as jnp
import pytest

import jax_rmhd as jr
from _rmhd_testing import alfven_ic, checks, ctx, fit_order
from jax_rmhd.comms import allreduce_sum
from jax_rmhd.run import block_of_steps
from jax_rmhd.timestepping import get_scheme

_DT = 1e-3
_NSTEPS = round(0.1 / _DT)  # exact step count; the end time is read back from state.t
_NZ_FAST = (16, 32, 64)
_NZ_SLOW = (64, 128, 256, 512, 1024)  # coarse triple fits the order; fine end floors

# jit exactly like run.py's mpi4jax path (params/nblock/scheme/stepper static).
# Donation is safe here: every state passed in is fresh from initialize, never
# cached or shared (CLAUDE.md donation rule).
_block_jit = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5), donate_argnums=(0,))


def _advect_errors(nz):
    """(dz, L1 rel err, L2 rel err) after _NSTEPS ideal fixed-dt steps at this nz.

    All reductions go through allreduce_sum (identity when cart_comm is None, i.e.
    single-process) so `mpirun -n 4` script mode yields the same global norms.
    """
    params, kgrid = ctx(nz=nz, diss=(0.0, 0.0), dt=_DT)
    state = jr.initialize(alfven_ic, params)
    stepper, scheme = get_scheme("lsrk33")
    end_state, _ = _block_jit(state, kgrid, params, _NSTEPS, scheme, stepper)
    t = float(end_state.t)  # t accumulates in floating point (0.09999974... at fp32)

    def exact_ic(x, y, z):
        phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z + t)
        return jnp.stack([phi, phi], axis=0)

    exact = jr.initialize(exact_ic, params)  # same dealias mask as the evolved IC
    diff = end_state.fields - exact.fields
    l1 = (float(allreduce_sum(jnp.sum(jnp.abs(diff)), params))
          / float(allreduce_sum(jnp.sum(jnp.abs(exact.fields)), params)))
    l2 = (float(jnp.sqrt(allreduce_sum(jnp.sum(jnp.abs(diff) ** 2), params)))
          / float(jnp.sqrt(allreduce_sum(jnp.sum(jnp.abs(exact.fields) ** 2), params))))
    return params.dz, l1, l2


def _sweep(nzs):
    dzs, l1s, l2s = [], [], []
    for nz in nzs:
        dz, l1, l2 = _advect_errors(nz)
        dzs.append(dz)
        l1s.append(l1)
        l2s.append(l2)
        print(f"nz={nz:5d}  dz={dz:.5f}  L1 rel err={l1:.6e}  L2 rel err={l2:.6e}")
    return dzs, l1s, l2s


def test_advection_z_convergence_fast():
    dzs, l1s, l2s = _sweep(_NZ_FAST)
    order = fit_order(dzs, l2s)
    # One 3.5 threshold at both precisions -- measured L2 orders: fp64 3.99 (errs
    # 8.58e-5 / 5.44e-6 / 3.42e-7), fp32 3.85 (errs 8.56e-5 / 5.27e-6 / 4.12e-7).
    # At fp32 the nz=64 z-error (~3.4e-7 at fp64) sits near the round-off floor
    # accumulated over 100 steps -- observed ~20% inflated to 4.1e-7 -- shaving
    # ~0.14 off the fitted slope but staying well clear of 3.5.
    with checks() as c:
        c.check("fitted z-convergence order > 3.5 (nominally 4th order)",
                order > 3.5, f"order={order:.3f}, L2 errs={l2s}")
        c.check("L1 relative error monotone decreasing in nz",
                all(a > b for a, b in zip(l1s, l1s[1:])), f"L1 errs={l1s}")
        c.check("L2 relative error monotone decreasing in nz",
                all(a > b for a, b in zip(l2s, l2s[1:])), f"L2 errs={l2s}")


@pytest.mark.slow
@pytest.mark.fp64
def test_advection_z_convergence_slow():
    # fp64-only: at fp32 every nz>=64 error is pinned at the accumulated round-off
    # floor (measured L2: 4.12e-7 / 4.51e-7 / 4.75e-7 for nz=64/128/256, fitted
    # order -0.10, vs fp64 references 3.42e-7 / 2.14e-8 / 1.34e-9), so neither the
    # coarse-triple order nor error non-increase is meaningful there. The fast-tier
    # test above covers fp32.
    dzs, l1s, l2s = _sweep(_NZ_SLOW)
    order_coarse = fit_order(dzs[:3], l2s[:3])
    # Fine end flattens toward the fixed-dt O(dt^3) floor -- expected, so no order
    # fit there, only non-increase (measured fp64 L2: 3.42e-7 / 2.14e-8 / 1.34e-9 /
    # 8.53e-11 / 7.94e-12: the last pair's local order already sags to ~3.4).
    with checks() as c:
        c.check("coarse-triple (64,128,256) fitted z-convergence order > 3.5",
                order_coarse > 3.5, f"order={order_coarse:.3f}, L2 errs={l2s[:3]}")
        c.check("L1 relative error non-increasing across the full sweep",
                all(b <= a for a, b in zip(l1s, l1s[1:])), f"L1 errs={l1s}")
        c.check("L2 relative error non-increasing across the full sweep",
                all(b <= a for a, b in zip(l2s, l2s[1:])), f"L2 errs={l2s}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
