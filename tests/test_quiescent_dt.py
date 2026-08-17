# rmhd.set_timestep's velocity floor IS rmhd._quiescent_dt: the adaptive dt is
# min(cfl_safety/max_all, _quiescent_dt(params)), so dt <= _quiescent_dt holds by
# construction for every state -- which is the bound the per-stage forcing scale cap
# (rmhd._scale_epilogue) relies on.
#
# From a zero-gradient state the CFL max collapses to the ceiling exactly (2D and 3D
# z_spectral, where no z term enters the max); in 3D finite-difference-z the 1/dz and
# z_diss terms can bind below it, so the ceiling is only an upper bound there.
# Comparisons are exact in the working dtype (the fp32 session rounds the python-float
# ceiling to float32), so both precision sessions run this.
#
# mpirun-safe: the 3D checks call the collective CFL allreduce identically on every rank
# (nz=8 divisible by 4); the dims=2 and z_spectral checks are size==1-only by
# construction and skip together on all ranks.
from _rmhd_testing import bootstrap, checks, ctx, make_state, mpi_size, multimode_ic

bootstrap()

import numpy as np

from taranis.physics import equation_registry, rmhd


def _dt_and_ceiling(ic=None, **overrides):
    params, kgrid = ctx(adaptive_timestep=True, **overrides)
    recipe = equation_registry[params.eqtype]
    grads = recipe.grad_func(make_state(params, ic=ic), kgrid, params)
    dt = np.asarray(recipe.set_timestep_func(grads, params))
    # cast the python-float ceiling into the dt's dtype: fp32 rounds it, and the
    # comparison is about the value set_timestep can produce, not about float64
    return dt, np.asarray(rmhd._quiescent_dt(params), dtype=dt.dtype)


def test_zero_gradient_dt_equals_quiescent_ceiling():
    # no z term in the CFL max -> the floor is the only thing left in it
    if mpi_size() > 1:
        print("[SKIP] test_zero_gradient_dt_equals_quiescent_ceiling -- dims=2 and "
              "z_spectral are single-process only")
        return
    with checks() as c:
        for label, kw in (("2D", dict(dims=2)),
                          ("2D anisotropic grid (dy < dx)", dict(dims=2, ny=32)),
                          ("3D z_spectral", dict(z_spectral=True))):
            dt, dtq = _dt_and_ceiling(**kw)
            c.check(f"{label}: zero-gradient dt == _quiescent_dt exactly "
                    f"(dt={float(dt):.9g})", dt == dtq, f"dt={dt!r}, ceiling={dtq!r}")


def test_finite_difference_z_dt_stays_under_ceiling():
    # 1/dz (and z_diss) stay in the max, so the ceiling is a slack upper bound here
    with checks() as c:
        dt, dtq = _dt_and_ceiling()
        c.check(f"3D FD-z: zero-gradient dt <= _quiescent_dt (dt={float(dt):.9g} <= "
                f"{float(dtq):.9g})", dt <= dtq)
        dt_dev, dtq_dev = _dt_and_ceiling(ic=multimode_ic)
        c.check(f"3D FD-z: developed-state dt <= _quiescent_dt (dt={float(dt_dev):.9g} "
                f"<= {float(dtq_dev):.9g})", dt_dev <= dtq_dev)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
