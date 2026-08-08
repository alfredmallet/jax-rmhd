# params.cfl_every block-CFL coverage.
#
# The IC is the UNFORCED tiny-amplitude Alfven wave (amplitude 1e-3): the CFL
# reduction max_all = max(grad-terms/dx, eps/dx, 1/dz, z_diss) is then pinned by the
# time-independent 1/dz term (1/dz ~ 1.27 vs eps/dx ~ 0.25, z_diss = 0.25, grad
# terms ~ 1e-2), so the adaptive dt is EXACTLY cfl_safety*dz on every step and
# cfl_every=1 vs 4 must integrate the same trajectory. This deliberately avoids the
# documented trap: cfl_every>1 from a QUIESCENT FORCED start silently NaNs (dt
# collapses ~10x during spin-up) -- see CLAUDE.md.
#
# Also covered: adaptive_timestep=False takes the unchanged legacy path regardless
# of cfl_every (bitwise), and nblock rounds UP to whole blocks (10 steps at
# cfl_every=4 -> 12 steps).
#
# mpirun-safe: the CFL allreduce is collective and called identically on all ranks;
# nz=8 divisible by 4. Savio: `mpirun -n 4 python tests/test_cfl_every.py`.
from _rmhd_testing import bootstrap, checks, ctx, make_state, tiny_alfven_ic

bootstrap()

import jax
import numpy as np

from taranis import _precision
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))


def _run(nblock, **overrides):
    params, kgrid = ctx(**overrides)
    stepper, scheme = get_scheme("lsrk33")
    end = _advance(make_state(params, ic=tiny_alfven_ic), kgrid, params,
                      nblock, scheme, stepper)
    return params, end


def _dt_expected(params):
    # 1/dz dominates the CFL max for this IC, so dt = cfl_safety*dz exactly.
    return params.cfl_safety * params.dz


def test_cfl_every_4_matches_every_1():
    p1, end1 = _run(8, adaptive_timestep=True, cfl_every=1)
    p4, end4 = _run(8, adaptive_timestep=True, cfl_every=4)  # 2 blocks of 4
    f1, f4 = np.asarray(end1.fields), np.asarray(end4.fields)
    rel = float(np.max(np.abs(f4 - f1)) / np.max(np.abs(f1)))
    tol = 1e-14 if _precision.precision == "64" else 1e-6
    dt = _dt_expected(p1)
    t_tol = 1e-12 if _precision.precision == "64" else 1e-4
    with checks() as c:
        c.check(f"dt is pinned by the 1/dz term (t after 8 steps == 8*cfl_safety*dz, "
                f"t={float(end1.t):.6f})",
                abs(float(end1.t) - 8 * dt) < t_tol * 8 * dt)
        c.check("cfl_every=1 and =4 end at the same time",
                abs(float(end4.t) - float(end1.t)) < t_tol * float(end1.t))
        c.check(f"trajectories agree (rel {rel:.2e} < {tol:.0e})", rel < tol)


def test_fixed_dt_ignores_cfl_every_bitwise():
    # adaptive_timestep=False: _use_cfl_blocks is False for ANY cfl_every, so both
    # runs take literally the same legacy code path -> bitwise at both precisions.
    _, end1 = _run(10, adaptive_timestep=False, cfl_every=1)
    _, end4 = _run(10, adaptive_timestep=False, cfl_every=4)
    f1, f4 = np.asarray(end1.fields), np.asarray(end4.fields)
    with checks() as c:
        c.check("adaptive_timestep=False + cfl_every=4 is bitwise identical to the "
                "legacy path", np.array_equal(f1, f4),
                f"max|diff|={np.max(np.abs(f4 - f1)):.3e}")
        c.check("... and ends at the same time (10 fixed steps)",
                float(end1.t) == float(end4.t))


def test_nblock_rounds_up_to_whole_blocks():
    # nblock=10 at cfl_every=4 -> ceil(10/4)=3 blocks -> 12 steps, not 10. dt is
    # exactly cfl_safety*dz throughout (see header), so t pins the step count.
    p4, end = _run(10, adaptive_timestep=True, cfl_every=4)
    dt = _dt_expected(p4)
    t = float(end.t)
    t_tol = 1e-10 if _precision.precision == "64" else 1e-3
    with checks() as c:
        c.check(f"10 steps at cfl_every=4 run 12 steps (t={t:.6f} vs 12*dt={12*dt:.6f})",
                abs(t - 12 * dt) < t_tol,
                f"t={t}, 10*dt={10*dt}, 12*dt={12*dt}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
