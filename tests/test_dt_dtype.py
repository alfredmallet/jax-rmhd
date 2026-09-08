# dt dtype regression (the CMHD C1 review's repo-wide trap, plans/CMHD_PLAN.md C1
# landing note): a NUMPY float64 scalar is a STRONG jax type, so a dt computed like
# 0.05/np.sqrt(...) entering a stepper at TARANIS_PRECISION=32 -- as params.dt or as
# dt_override -- used to upcast the propagator in kgrid.lin.scaled(dt) and then the
# whole field graph, surfacing only one stage later at construct_rhs's dtype assert.
# The fix: Parameters stores dt as a weak python float, and timestepping._weak_dt
# normalizes numpy-scalar dt at the stepper dt resolutions and stage_exp_ops (python
# floats and traced values pass through untouched, so existing graphs are unchanged).
#
# This gate steps with a numpy dt through both entry points, over the diagonal,
# separable and putzer2 propagator backends and the IF and IMEX stepper families, and
# asserts FIELD-precision output plus bitwise agreement with the identical run at
# python-float dt. Runs in both precision sessions -- fp32 is where it used to bite.
# mpirun-safe: every config here is single-process only; ranks skip together.
from _rmhd_testing import bootstrap

bootstrap()

import numpy as np

import taranis as jr
from taranis import _precision
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme, stage_exp_ops
from taranis.physics import construct_rhs, equation_registry

from _rmhd_testing import checks, fresh_params, make_state, mpi_size

# the recorded trap shape: a numpy-float64 scalar from scalar arithmetic
_NUMPY_DT = np.float64(0.05) / np.sqrt(2.0)

# label -> (params overrides, scheme): one config per propagator backend, plus IMEX.
# All fixed-dt (adaptive_timestep=False is the _DEFAULTS_3D default) and size==1-only.
_CONFIGS = (
    ("2D diagonal lsrk33", dict(dims=2), "lsrk33"),
    ("z_spectral separable lsrk54 (hoisted)", dict(z_spectral=True), "lsrk54"),
    ("z_spectral putzer2 lsrk33 (hoisted)",
     dict(z_spectral=True, diss=(0.01, 0.02)), "lsrk33"),
    ("2D diagonal imexcb3e", dict(dims=2), "imexcb3e"),
)


def _stepped(dt, overrides, scheme_name):
    params = fresh_params(dt=dt, **overrides)
    kgrid = jr.setup_kgrids(params)
    stepper, scheme = get_scheme(scheme_name)
    return params, block_of_steps(make_state(params), kgrid, params, 2, scheme, stepper)


def test_numpy_dt_in_parameters_keeps_field_precision():
    # params.dt entry point: Parameters normalizes to a python float, and the stepped
    # fields stay at FIELD precision and bitwise-match the python-float-dt run.
    if mpi_size() > 1:
        print("[SKIP] test_numpy_dt_in_parameters_keeps_field_precision -- "
              "single-process-only configs")
        return
    with checks() as c:
        for label, overrides, scheme_name in _CONFIGS:
            params, out = _stepped(_NUMPY_DT, overrides, scheme_name)
            c.check(f"{label}: params.dt is a python float",
                    type(params.dt) is float, f"got {type(params.dt)}")
            c.check(f"{label}: stepped fields keep FIELD precision",
                    out.fields.dtype == _precision.ctype,
                    f"got {out.fields.dtype}, expected {_precision.ctype}")
            _, ref = _stepped(float(_NUMPY_DT), overrides, scheme_name)
            c.check(f"{label}: bitwise-identical to the python-float-dt run",
                    np.array_equal(np.asarray(out.fields), np.asarray(ref.fields))
                    and float(out.t) == float(ref.t))


def test_numpy_dt_override_keeps_field_precision():
    # dt_override entry point: a direct stepper call with a numpy dt (the recorded
    # failure mode) must produce FIELD-precision fields, IF and IMEX alike.
    if mpi_size() > 1:
        print("[SKIP] test_numpy_dt_override_keeps_field_precision -- "
              "single-process-only configs")
        return
    with checks() as c:
        for label, overrides, scheme_name in _CONFIGS:
            params = fresh_params(**overrides)
            kgrid = jr.setup_kgrids(params)
            stepper, scheme = get_scheme(scheme_name)
            recipe = equation_registry[params.eqtype]
            rhs = construct_rhs(recipe)
            out = stepper(make_state(params), kgrid, params, rhs,
                          recipe.set_timestep_func, scheme, dt_override=_NUMPY_DT)
            c.check(f"{label}: dt_override=np.float64 keeps FIELD precision",
                    out.fields.dtype == _precision.ctype,
                    f"got {out.fields.dtype}, expected {_precision.ctype}")


def test_numpy_dt_stage_exp_ops_keeps_field_precision():
    # stage_exp_ops entry point (the hoisted path's own kgrid.lin.scaled(dt)): a numpy
    # dt must not upcast the precomputed per-stage ExpOp arrays.
    if mpi_size() > 1:
        print("[SKIP] test_numpy_dt_stage_exp_ops_keeps_field_precision -- "
              "z_spectral is single-process only")
        return
    with checks() as c:
        params = fresh_params(z_spectral=True)
        kgrid = jr.setup_kgrids(params)
        stepper, scheme = get_scheme("lsrk54")
        ops = stage_exp_ops(kgrid, params, scheme, stepper, _NUMPY_DT)
        c.check("z_spectral lsrk54: one ExpOp per stage", len(ops) == len(scheme.gammas))
        c.check("stage ExpOp arrays keep FIELD precision",
                all(op.P.dtype == _precision.ftype and op.c.dtype == _precision.ftype
                    and op.s.dtype == _precision.ftype for op in ops),
                f"got {[str(op.P.dtype) for op in ops]}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
