# Tests for comm_backend="serial" (SERIAL_BACKEND_PLAN.md, agent A3): the no-MPI
# single-process transport that lets `pip install -e .` (no mpi4py/mpi4jax) work.
#
# Everything except test_true_no_mpi_import_path_subprocess runs under the usual
# bootstrap() (local_mpi_stub installed, since this sandbox has no real MPI), which
# means comm_backend="mpi4jax" is ALSO available in-process, as a size-1 stub. Per
# A1/A2's code-inspection (and re-verified below): the stub's sendrecv/allreduce are
# literal identity lambdas, so comms.halo_exchange and comms.allreduce_* return
# bitwise-identical arrays whether comm_backend is "serial" or stubbed "mpi4jax" --
# there is no token/fusion-ordering difference to blur the comparison, unlike REAL
# mpi4jax (which SERIAL_BACKEND_PLAN.md's "Known behavior deltas" section covers with
# tolerances -- not exercisable in this sandbox). So every serial-vs-mpi4jax
# comparison here uses EXACT equality, not a tolerance.
#
# The resolution-matrix test monkeypatches jax_rmhd.config.HAVE_MPI4PY/HAVE_MPI4JAX/
# MPI (the names _resolve_backend actually reads -- config.py imports them BY VALUE
# from _mpi_compat, so patching _mpi_compat itself would not be observed) and calls
# jax_rmhd.config._resolve_backend() directly. jax_rmhd is never re-imported.
#
# pytest: `pytest tests/test_backend_serial.py`.  Script: `python
# tests/test_backend_serial.py`.  No markers -- this is the runs-anywhere tier.
from _rmhd_testing import (bootstrap, checks, ctx, fresh_params, make_state,
                           managed_manager, mpi_size, multimode_ic, snap_dir)

bootstrap()

import contextlib
import json
import os
import subprocess
import sys
import warnings

import jax.numpy as jnp
import numpy as np

import jax_rmhd as jr
import jax_rmhd.config as cfg
import jax_rmhd.snapshot_io as sn
from jax_rmhd import comms


def _single_process_only(what):
    """Soft skip (print + return True). comm_backend='serial' rejects size>1 at
    construction (ValueError), so any test that builds a serial Parameters only
    means something single-process; every rank takes this same branch under
    mpirun, so nothing can deadlock."""
    if mpi_size() > 1:
        print(f"[SKIP] {what} -- single-process only (comm_backend='serial' is "
              "size==1 by construction)")
        return True
    return False


def _multimode_ic_2d(x, y):
    # Same non-degenerate multi-mode structure as _rmhd_testing.multimode_ic, minus z.
    phi = jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y)
    psi = jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([phi, psi], axis=0)


# --------------------------------------------------------------- 1. halo_exchange


def test_halo_exchange_serial_matches_stub_mpi4jax_widths():
    if _single_process_only("test_halo_exchange_serial_matches_stub_mpi4jax_widths"):
        return
    p_ser, _ = ctx(comm_backend="serial")
    p_mpi, _ = ctx(comm_backend="mpi4jax")
    f = make_state(p_ser, ic=multimode_ic).fields
    with checks() as c:
        for width in (1, 2):
            left_ser, right_ser = comms.halo_exchange(f, p_ser, width=width)
            left_mpi, right_mpi = comms.halo_exchange(f, p_mpi, width=width)
            c.check(f"width={width}: recv_left bitwise identical (serial vs stub mpi4jax)",
                    bool(jnp.array_equal(left_ser, left_mpi)))
            c.check(f"width={width}: recv_right bitwise identical (serial vs stub mpi4jax)",
                    bool(jnp.array_equal(right_ser, right_mpi)))
            # SERIAL_BACKEND_PLAN.md's exact self-wrap identities
            c.check(f"width={width}: serial recv_left == f's last {width} plane(s)",
                    bool(jnp.array_equal(left_ser, f[:, -width:, :, :])))
            c.check(f"width={width}: serial recv_right == f's first {width} plane(s)",
                    bool(jnp.array_equal(right_ser, f[:, :width, :, :])))


def test_halo_exchange_width_bounds_raise_both_backends():
    if _single_process_only("test_halo_exchange_width_bounds_raise_both_backends"):
        return
    p_ser, _ = ctx(comm_backend="serial")
    p_mpi, _ = ctx(comm_backend="mpi4jax")
    f = make_state(p_ser, ic=multimode_ic).fields
    nz_local = f.shape[1]
    with checks() as c:
        for p, name in ((p_ser, "serial"), (p_mpi, "mpi4jax")):
            for bad_width in (nz_local + 1, 0):
                try:
                    comms.halo_exchange(f, p, width=bad_width)
                    raised = False
                except ValueError:
                    raised = True
                c.check(f"{name} backend rejects width={bad_width}", raised)


# ------------------------------------------------------------ 2. N-step trajectory


def _traj_kw(dims, forced):
    kw = dict(dims=dims)
    if forced:
        kw.update(forcing=True, forcing_mode="elsasser",
                  forcing_power_elsasser=(0.5, 0.5), forcing_tau=0.5,
                  fshell=(1, 3), forcing_seed=17)
    return kw


def _run_traj_end(comm_backend, dims, forced):
    p, kg = ctx(comm_backend=comm_backend, **_traj_kw(dims, forced))
    ic = multimode_ic if dims == 3 else _multimode_ic_2d
    st = make_state(p, ic=ic)
    with snap_dir() as d, managed_manager(p, d, nsnap=2) as m:
        end = jr.simulate(st, kg, p, t_snap=10.0, t_end=0.1, mngr=m, save=False)
    return (np.asarray(end.fields), float(end.t),
            np.asarray(end.forcing_state), np.asarray(end.forcing_scale))


def _check_traj_matches(label, dims, forced):
    if _single_process_only(label):
        return
    f_ser, t_ser, fs_ser, sc_ser = _run_traj_end("serial", dims, forced)
    f_mpi, t_mpi, fs_mpi, sc_mpi = _run_traj_end("mpi4jax", dims, forced)
    with checks() as c:
        c.check(f"{label}: fields finite and nonzero",
                bool(np.all(np.isfinite(f_ser))) and float(np.max(np.abs(f_ser))) > 0.0)
        c.check(f"{label}: fields bitwise identical (serial vs stub mpi4jax)",
                np.array_equal(f_ser, f_mpi))
        c.check(f"{label}: end time identical ({t_ser!r} vs {t_mpi!r})", t_ser == t_mpi)
        c.check(f"{label}: forcing_state bitwise identical", np.array_equal(fs_ser, fs_mpi))
        c.check(f"{label}: forcing_scale bitwise identical", np.array_equal(sc_ser, sc_mpi))


def test_trajectory_3d_unforced_serial_matches_stub_mpi4jax():
    _check_traj_matches("3D unforced", dims=3, forced=False)


def test_trajectory_3d_forced_serial_matches_stub_mpi4jax():
    _check_traj_matches("3D forced", dims=3, forced=True)


def test_trajectory_2d_unforced_serial_matches_stub_mpi4jax():
    _check_traj_matches("2D unforced", dims=2, forced=False)


def test_trajectory_2d_forced_serial_matches_stub_mpi4jax():
    # dims=2 + forced uses elsasser (CLAUDE.md: momentum mode from a quiescent start
    # is pure hydro in 2D -- irrelevant here since forcing perturbs psi directly, but
    # elsasser is what an actual 2D MHD run would use).
    _check_traj_matches("2D forced", dims=2, forced=True)


# ----------------------------------------------------------- 3. resolution matrix

# Definitive (per-process, launcher-set) vars + the allocation-wide hints they must
# never be confused with (SERIAL_BACKEND_PLAN.md / jax_rmhd/_mpi_compat.py).
_LAUNCHER_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE",
                   "SLURM_NTASKS", "SLURM_STEP_NUM_TASKS", "PMIX_RANK")


class _FakeWorld:
    def __init__(self, size):
        self._size = size

    def Get_size(self):
        return self._size


class _FakeMPI:
    def __init__(self, size):
        self.COMM_WORLD = _FakeWorld(size)


@contextlib.contextmanager
def _patched(have_mpi4py=None, have_mpi4jax=None, mpi_world_size=None, env=None):
    """Monkeypatch the names jax_rmhd.config._resolve_backend actually reads, and
    os.environ, restoring everything on exit. Always clears every launcher/
    allocation env var first, so a real cluster's environment (e.g. a Savio SLURM
    allocation) can never leak into a case. Never re-imports jax_rmhd."""
    saved = dict(HAVE_MPI4PY=cfg.HAVE_MPI4PY, HAVE_MPI4JAX=cfg.HAVE_MPI4JAX, MPI=cfg.MPI)
    saved_env = {k: os.environ.get(k) for k in _LAUNCHER_VARS}
    try:
        for k in _LAUNCHER_VARS:
            os.environ.pop(k, None)
        for k, v in (env or {}).items():
            os.environ[k] = v
        if have_mpi4py is not None:
            cfg.HAVE_MPI4PY = have_mpi4py
        if have_mpi4jax is not None:
            cfg.HAVE_MPI4JAX = have_mpi4jax
        if mpi_world_size is not None:
            cfg.MPI = _FakeMPI(mpi_world_size)
        yield
    finally:
        cfg.HAVE_MPI4PY = saved["HAVE_MPI4PY"]
        cfg.HAVE_MPI4JAX = saved["HAVE_MPI4JAX"]
        cfg.MPI = saved["MPI"]
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _raises(exc_type, fn):
    try:
        fn()
        return None
    except exc_type as e:
        return str(e)
    except Exception as e:  # noqa: BLE001 - wrong exception type is still informative
        return f"WRONG EXCEPTION TYPE {type(e).__name__}: {e}"


def test_resolution_matrix():
    with checks() as c:
        # auto, both present -> "mpi4jax" (today's unchanged default)
        with _patched(have_mpi4py=True, have_mpi4jax=True, mpi_world_size=1):
            c.check("auto + mpi4py + mpi4jax -> mpi4jax",
                    cfg._resolve_backend(None) == "mpi4jax")

        # auto, neither importable, no launcher -> serial (the laptop case)
        with _patched(have_mpi4py=False, have_mpi4jax=False):
            c.check("auto + no MPI at all, no launcher -> serial",
                    cfg._resolve_backend(None) == "serial")

        # half-installed (mpi4py only): size==1 -> serial; size>1 -> RuntimeError
        # naming mpi4jax (common case: mpi4py builds easily, mpi4jax needs a
        # matching jax)
        with _patched(have_mpi4py=True, have_mpi4jax=False, mpi_world_size=1):
            c.check("auto + mpi4py-only + real size==1 -> serial",
                    cfg._resolve_backend(None) == "serial")
        with _patched(have_mpi4py=True, have_mpi4jax=False, mpi_world_size=3):
            msg = _raises(RuntimeError, lambda: cfg._resolve_backend(None))
            c.check("auto + mpi4py-only + real size>1 -> RuntimeError naming mpi4jax",
                    msg is not None and "mpi4jax" in msg, msg)

        # explicit "mpi4jax" without both imports -> ImportError naming jax-rmhd[mpi]
        with _patched(have_mpi4py=False):
            msg = _raises(ImportError, lambda: cfg._resolve_backend("mpi4jax"))
            c.check("explicit mpi4jax, no mpi4py -> ImportError naming jax-rmhd[mpi]",
                    msg is not None and "jax-rmhd[mpi]" in msg, msg)
        with _patched(have_mpi4py=True, have_mpi4jax=False, mpi_world_size=1):
            msg = _raises(ImportError, lambda: cfg._resolve_backend("mpi4jax"))
            c.check("explicit mpi4jax, mpi4py present but mpi4jax absent -> ImportError "
                    "naming jax-rmhd[mpi]", msg is not None and "jax-rmhd[mpi]" in msg, msg)

        # explicit "jax" needs only mpi4py (deliberate A1 deviation from mpi4jax)
        with _patched(have_mpi4py=True, have_mpi4jax=False, mpi_world_size=1):
            c.check("explicit jax + mpi4py present (mpi4jax absent) is accepted",
                    cfg._resolve_backend("jax") == "jax")
        with _patched(have_mpi4py=False):
            msg = _raises(ImportError, lambda: cfg._resolve_backend("jax"))
            c.check("explicit jax without mpi4py -> ImportError naming jax-rmhd[mpi]",
                    msg is not None and "jax-rmhd[mpi]" in msg, msg)

        # definitive launcher vars + no mpi4py -> hard RuntimeError, never silent serial
        for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE"):
            with _patched(have_mpi4py=False, env={var: "4"}):
                msg = _raises(RuntimeError, lambda: cfg._resolve_backend(None))
                c.check(f"auto + {var}=4 (definitive launcher var) + no mpi4py -> RuntimeError",
                        msg is not None, msg)
        # PMIX_RANK promotes the allocation-wide SLURM_NTASKS hint to definitive
        with _patched(have_mpi4py=False, env={"SLURM_NTASKS": "4", "PMIX_RANK": "0"}):
            msg = _raises(RuntimeError, lambda: cfg._resolve_backend(None))
            c.check("auto + SLURM_NTASKS=4 + PMIX_RANK (definitive) + no mpi4py -> "
                    "RuntimeError", msg is not None, msg)

        # SLURM_NTASKS alone is an allocation-wide HINT, not definitive: warn, serial
        with _patched(have_mpi4py=False, env={"SLURM_NTASKS": "4"}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = cfg._resolve_backend(None)
            c.check("auto + SLURM_NTASKS=4 alone (non-definitive) -> serial",
                    result == "serial")
            c.check("... but warns instead of silently degrading",
                    any("serial" in str(w.message) for w in caught),
                    [str(w.message) for w in caught])

        # launcher definitive >1 but mpi4py itself reports size 1 -> mismatched-MPI
        # guard (mpi4py built against a different MPI than the launcher)
        with _patched(have_mpi4py=True, mpi_world_size=1, env={"OMPI_COMM_WORLD_SIZE": "4"}):
            msg = _raises(RuntimeError, lambda: cfg._resolve_backend(None))
            c.check("launcher says 4 ranks, mpi4py reports size 1 -> mismatched-MPI "
                    "RuntimeError", msg is not None and "mpi4py" in msg, msg)

        # explicit "serial" under a definitive launcher -> ValueError, whether or not
        # mpi4py agrees with the launcher's count
        with _patched(have_mpi4py=False, env={"OMPI_COMM_WORLD_SIZE": "4"}):
            msg = _raises(ValueError, lambda: cfg._resolve_backend("serial"))
            c.check("explicit serial under a definitive launcher (no mpi4py) -> ValueError",
                    msg is not None, msg)
        with _patched(have_mpi4py=True, mpi_world_size=4, env={"OMPI_COMM_WORLD_SIZE": "4"}):
            msg = _raises(ValueError, lambda: cfg._resolve_backend("serial"))
            c.check("explicit serial under a definitive launcher (mpi4py size agrees) "
                    "-> ValueError", msg is not None, msg)

        # explicit "serial", mpi4py present, truly single-rank -> accepted
        with _patched(have_mpi4py=True, mpi_world_size=1):
            c.check("explicit serial, mpi4py present, real size==1 -> accepted",
                    cfg._resolve_backend("serial") == "serial")


# ------------------------------------------------------- 4. snapshots / from_snapshot


def test_snapshot_roundtrip_serial_and_cross_restore_with_mpi4jax():
    if _single_process_only("test_snapshot_roundtrip_serial_and_cross_restore_with_mpi4jax"):
        return
    p_ser, _ = ctx(comm_backend="serial")
    p_mpi, _ = ctx(comm_backend="mpi4jax")

    state_ser = make_state(p_ser, ic=multimode_ic)
    want_ser = np.asarray(state_ser.fields)
    with snap_dir() as d1:
        with managed_manager(p_ser, d1, nsnap=2) as m1:
            sn.save_snapshot(0, state_ser, m1, p_ser)
            m1.wait_until_finished()
        layout1 = sn.snapshot_layout(d1)
        back_ser = sn.load_snapshot(0, d1, p_ser)
        back_mpi = sn.load_snapshot(0, d1, p_mpi)  # cross-restore: serial-written

    state_mpi = make_state(p_mpi, ic=multimode_ic)
    want_mpi = np.asarray(state_mpi.fields)
    with snap_dir() as d2:
        with managed_manager(p_mpi, d2, nsnap=2) as m2:
            sn.save_snapshot(0, state_mpi, m2, p_mpi)
            m2.wait_until_finished()
        layout2 = sn.snapshot_layout(d2)
        back_mpi2 = sn.load_snapshot(0, d2, p_mpi)
        back_ser2 = sn.load_snapshot(0, d2, p_ser)  # cross-restore: mpi4jax-written

    with checks() as c:
        c.check(f"serial writer produces the flat layout (got {layout1!r})",
                layout1 == "flat")
        c.check("serial round trip bitwise",
                np.array_equal(np.asarray(back_ser.fields), want_ser))
        c.check("mpi4jax (size-1) params read the serial-written snapshot bitwise "
                "identically", np.array_equal(np.asarray(back_mpi.fields), want_ser))
        c.check(f"mpi4jax (size-1) writer produces the flat layout (got {layout2!r})",
                layout2 == "flat")
        c.check("mpi4jax round trip bitwise",
                np.array_equal(np.asarray(back_mpi2.fields), want_mpi))
        c.check("serial params read the mpi4jax-written snapshot bitwise identically",
                np.array_equal(np.asarray(back_ser2.fields), want_mpi))


def _params_path(d):
    return os.path.join(d, "params.json")


def _read_record(d):
    with open(_params_path(d)) as f:
        return json.load(f)


def _write_record(d, rec):
    with open(_params_path(d), "w") as f:
        json.dump(rec, f, indent=1)


def test_from_snapshot_drops_recorded_backend_and_reresolves():
    if _single_process_only("test_from_snapshot_drops_recorded_backend_and_reresolves"):
        return
    p_written_mpi4jax = fresh_params(comm_backend="mpi4jax", dims=2)
    with snap_dir() as d:
        p_written_mpi4jax.save(d)
        rec = _read_record(d)

        # A Savio-written params.json recording "mpi4jax" must still load on a
        # machine where mpi4py/mpi4jax are unavailable -- from_snapshot re-resolves,
        # it does not trust the recorded value.
        with _patched(have_mpi4py=False, have_mpi4jax=False):
            p_noMPI = jr.Parameters.from_snapshot(d)

        # explicit override wins over both the recorded value and re-resolution
        p_override = jr.Parameters.from_snapshot(d, comm_backend="serial")

        with checks() as c:
            c.check(f"params.json recorded the backend that actually ran "
                    f"(got {rec.get('comm_backend')!r})", rec.get("comm_backend") == "mpi4jax")
            c.check("from_snapshot on a no-MPI machine re-resolves to serial, "
                    "ignoring the recorded 'mpi4jax'", p_noMPI.comm_backend == "serial")
            c.check("explicit comm_backend override wins", p_override.comm_backend == "serial")

    # the other direction: record='serial', override='mpi4jax' (real stub active)
    p_written_serial = fresh_params(comm_backend="serial", dims=2)
    with snap_dir() as d2:
        p_written_serial.save(d2)
        p_override2 = jr.Parameters.from_snapshot(d2, comm_backend="mpi4jax")
        with checks() as c:
            c.check(f"params.json recorded 'serial' (got "
                    f"{_read_record(d2).get('comm_backend')!r})",
                    _read_record(d2).get("comm_backend") == "serial")
            c.check("explicit override wins the other direction too "
                    "(record=serial, override=mpi4jax)", p_override2.comm_backend == "mpi4jax")


# ---------------------------------------------------------------- 5. backfill


def test_backfill_of_missing_comm_backend_uses_resolved_value_not_null():
    if _single_process_only("test_backfill_of_missing_comm_backend_uses_resolved_value_not_null"):
        return
    p = fresh_params(comm_backend="serial", dims=2)
    with snap_dir() as d:
        p.save(d)
        orig = _read_record(d)
        rec = {k: v for k, v in orig.items() if k != "comm_backend"}
        _write_record(d, rec)  # simulate an old params.json predating comm_backend

        p.save(d)  # re-save: must backfill, not error, and not write null
        after = _read_record(d)
        with checks() as c:
            c.check("original save recorded 'serial'", orig.get("comm_backend") == "serial")
            c.check("a record missing comm_backend entirely is accepted (not a "
                    "differing-record error)", "comm_backend" in after)
            c.check(f"the backfilled value is the resolved backend, not null "
                    f"(got {after.get('comm_backend')!r})",
                    after.get("comm_backend") == "serial")
            c.check("every other key is otherwise unchanged by the backfill",
                    {k: v for k, v in after.items() if k not in ("comm_backend", "_created")}
                    == {k: v for k, v in orig.items() if k not in ("comm_backend", "_created")})


# ------------------------------------------------- true no-MPI import path (no stub)


def test_true_no_mpi_import_path_subprocess():
    # bootstrap(stub=False) only means something in a process where it is the FIRST
    # thing to touch jax_rmhd -- this process already bootstrapped with the stub (and
    # jax_rmhd is already imported above), so the true absent-MPI path is exercised in
    # a fresh subprocess instead. If this machine genuinely has mpi4py installed (e.g.
    # a Savio account with the real toolchain), the absent-MPI path doesn't apply here
    # and the subprocess says so; treat that as a soft skip, not a failure.
    if _single_process_only("test_true_no_mpi_import_path_subprocess"):
        return
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    script = f"""
import os, sys
sys.path.insert(0, {tests_dir!r})
os.environ.setdefault("RMHD_PRECISION", "64")
from _rmhd_testing import bootstrap
bootstrap(stub=False)
import jax_rmhd as jr
from jax_rmhd import _mpi_compat as mc
import jax.numpy as jnp
if mc.HAVE_MPI4PY:
    print("REAL_MPI_PRESENT")
else:
    p = jr.Parameters(nx=8, ny=8, Lx=1.0, Ly=1.0, dims=2, diss=(0.0, 0.0), hyper=1,
                      cfl_safety=0.5)
    ok = (p.comm_backend == "serial" and p.rank == 0 and p.size == 1
          and p.cart_comm is None and not mc.HAVE_MPI4JAX)
    st = jr.initialize(lambda x, y: jnp.zeros((2,) + x.shape), p)
    ok = ok and bool(jnp.all(jnp.isfinite(st.fields)))
    print("TRUE_SERIAL_OK" if ok else "TRUE_SERIAL_FAIL")
"""
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=120)
    with checks() as c:
        c.check("subprocess (bootstrap(stub=False), no local_mpi_stub) exits cleanly",
                proc.returncode == 0,
                f"returncode={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")
        if "REAL_MPI_PRESENT" in proc.stdout:
            print("[SKIP] true-no-MPI subprocess check -- mpi4py is genuinely "
                  "importable on this machine; the absent-MPI path doesn't apply here")
        else:
            c.check("the true no-MPI import path auto-resolves comm_backend to "
                    "'serial' and runs initialize() successfully",
                    "TRUE_SERIAL_OK" in proc.stdout,
                    f"stdout={proc.stdout}\nstderr={proc.stderr}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
