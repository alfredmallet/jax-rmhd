# fp32 coverage: TARANIS_PRECISION defaults to 32 (taranis/__init__.py), so this is the
# production default. Everything here is @fp32-marked, so it runs in the
# `TARANIS_PRECISION=32` session of `make test` and prints [SKIP] in the fp64 session
# (script mode included).
#
# The default-precision check spawns a SUBPROCESS with TARANIS_PRECISION scrubbed
# from the environment -- the in-process value is pinned by bootstrap() at import
# and cannot be re-tested here. The subprocess falls back to the MPI stub exactly
# like bootstrap() does, so it works both locally and on Savio login/compute nodes.
#
# Single-process by design (snapshot tmp dirs); listed serially, "32" session
# only, in the Savio manifest. Script: `TARANIS_PRECISION=32 python tests/...py`.
from _rmhd_testing import bootstrap, checks, ctx, make_state, managed_manager, mpi_size, snap_dir, zero_ic

bootstrap()

import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

import taranis as jr
import taranis.snapshot_io as sn
from taranis import _precision, diagnostics

_FORCED = dict(nx=16, ny=16, nz=16, diss=(1e-4, 1e-4), hyper=2,
               adaptive_timestep=True, dt=0.1,
               forcing=True, forcing_mode="elsasser",
               forcing_power_elsasser=(0.5, 0.5), forcing_tau=1.0,
               fshell=(1, 3), forcing_seed=7)


@pytest.mark.fp32
def test_default_precision_is_32():
    # import taranis with NO TARANIS_PRECISION in the environment and read back
    # _precision.precision -- the honest check of the production default FIELD
    # precision. jax_enable_x64 is unconditionally on since A1 (PRECISION_PLAN) and
    # no longer distinguishes fp32 from fp64 sessions -- checked here too, as the
    # new invariant it replaced the old one with.
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(tests_dir)
    code = (
        "import sys\n"
        f"sys.path.insert(0, {tests_dir!r})\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        "try:\n"
        "    from mpi4py import MPI  # noqa: F401\n"
        "except Exception:\n"
        "    import local_mpi_stub  # noqa: F401\n"
        "import taranis, jax\n"
        "from taranis import _precision\n"
        "print('X64_IS', bool(jax.config.jax_enable_x64))\n"
        "print('PRECISION_IS', _precision.precision)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "TARANIS_PRECISION"}
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=300)
    with checks() as c:
        c.check("subprocess import (no TARANIS_PRECISION) succeeded",
                out.returncode == 0, out.stderr[-300:])
        c.check("x64 is unconditionally on regardless of TARANIS_PRECISION",
                "X64_IS True" in out.stdout, out.stdout[-200:])
        c.check("default FIELD precision is 32",
                "PRECISION_IS 32" in out.stdout, out.stdout[-200:])


@pytest.mark.fp32
def test_state_dtypes_are_fp32():
    params, _ = ctx()
    state = make_state(params)
    ftype, ctype = sn.get_precision_types()
    with checks() as c:
        c.check("this session runs at fp32", _precision.precision == "32")
        c.check(f"get_precision_types -> ({ftype.__name__}, {ctype.__name__})",
                ftype is jnp.float32 and ctype is jnp.complex64)
        c.check("fields are complex64", state.fields.dtype == jnp.complex64)
        c.check("forcing_state is complex64",
                state.forcing_state.dtype == jnp.complex64)
        c.check("forcing_scale is float32",
                state.forcing_scale.dtype == jnp.float32)
        # t is float64 at BOTH precisions by design (PRECISION_PLAN.md A2): at fp32,
        # t + dt stops advancing once t/dt exceeds ~1.7e7 steps. It is the one quantity
        # allowed to be 64-bit in an fp32 run, and it never multiplies fields
        # (run._advance_forcing downcasts the one t-difference that does).
        c.check("t is a 64-bit scalar (deliberate: fp32 t freezes at ~1.7e7 steps)",
                jnp.asarray(state.t).dtype == jnp.float64)


@pytest.mark.fp32
def test_forced_16cubed_run_stays_finite():
    # "does the production default even work": forced 16^3 elsasser run with the
    # adaptive timestep to t=0.5, entirely at fp32.
    params, kgrid = ctx(**_FORCED)
    state = make_state(params, ic=zero_ic)
    with snap_dir() as d, managed_manager(params, d, nsnap=2) as m:
        end = jr.simulate(state, kgrid, params, t_snap=10.0, t_end=0.5,
                          mngr=m, save=False)
    E_kin, E_mag = diagnostics.energy(end, kgrid, params)
    with checks() as c:
        c.check("fields stay finite", bool(jnp.all(jnp.isfinite(end.fields))))
        c.check(f"energy is finite and nonzero (E_kin {float(E_kin):.3e})",
                np.isfinite(float(E_kin)) and float(E_kin) > 0.0)
        c.check("forcing_scale stays finite",
                bool(jnp.all(jnp.isfinite(end.forcing_scale))))
        c.check(f"run advanced past t_end (t={float(end.t):.3f})",
                float(end.t) >= 0.5)


@pytest.mark.fp32
def test_fp32_snapshot_roundtrip():
    if mpi_size() > 1:
        print("[SKIP] test_fp32_snapshot_roundtrip -- single-process only (tmp dirs)")
        return
    params, kgrid = ctx(**_FORCED)
    state = make_state(params, ic=zero_ic)
    with snap_dir() as d:
        with managed_manager(params, d, nsnap=2) as m:
            end = jr.simulate(state, kgrid, params, t_snap=10.0, t_end=0.1,
                              mngr=m, save=False)
            want = np.asarray(end.fields)
            sn.save_snapshot(0, end, m, params)
            m.wait_until_finished()
            t_want = float(end.t)
        back = sn.load_snapshot(0, d, params)
        diff = float(np.max(np.abs(np.asarray(back.fields) - want)))
        with checks() as c:
            c.check("restored fields are complex64",
                    back.fields.dtype == jnp.complex64)
            c.check(f"fp32 fields round-trip within 1e-6 (max|diff| {diff:g}; "
                    f"expected bitwise)", diff <= 1e-6)
            c.check("t round-trips", float(back.t) == t_want)
            c.check("forcing_scale restores as concrete float32 (n_ou,)",
                    back.forcing_scale.shape == (params.n_ou,)
                    and back.forcing_scale.dtype == jnp.float32)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
