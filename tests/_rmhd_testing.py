# Shared test infrastructure for jax_rmhd. Not a test file (leading underscore keeps
# pytest from collecting it).
#
# Import contract -- every test module begins with:
#
#     from _rmhd_testing import bootstrap; bootstrap()
#     import jax_rmhd as jr
#
# bootstrap() must run before jax_rmhd (or jax device work) happens anywhere in the
# process: it sets RMHD_PRECISION (read once at jax_rmhd import), installs the local
# MPI stub when no real MPI toolchain is importable, and requests fake XLA host
# devices for the shard_map ("jax" backend) tests. It is idempotent, so the double
# call under pytest (conftest first, then the module header) is harmless.
#
# This resolves in BOTH execution modes with no PYTHONPATH fiddling:
#   - pytest:  tests/conftest.py puts this directory on sys.path and calls bootstrap()
#              before any test module is imported.
#   - script:  `python tests/test_x.py` / `mpirun -n N python tests/test_x.py` puts
#              the script's own directory (tests/) at sys.path[0] automatically.
#
# Rules inherited from CLAUDE.md -- read them before adding helpers:
#   - NEVER cache or share a SimulationState across tests: simulate/simulate_scan
#     donate their input state (donate_argnums=(0,)), so a shared state dies on first
#     use ("Array has been deleted"). make_state() always builds fresh.
#   - NEVER mutate a cached Parameters (it hashes by identity; jit silently reuses
#     the stale compile). ctx() results are shared -- treat them as frozen. Use
#     fresh_params()/fake_ranked_params() for rank-spoofing tricks.
#   - Never assert exact end times or snapshot counts: simulate overshoots
#     t_snap/t_end by up to one step (cfl_every steps in blocks). Use
#     float(end_state.t).

import contextlib
import functools
import math
import os
import shutil
import sys
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_bootstrapped = False
_stub_active = False


def bootstrap(precision=None, devices=None, stub=None):
    """Prepare env + MPI stub. Must run before jax_rmhd is imported; idempotent.

    stub=True (default) installs tests/local_mpi_stub.py whenever no real MPI
    toolchain is importable -- the normal test-suite path, which keeps
    comm_backend='mpi4jax' testable single-process. stub=False (or env
    RMHD_TEST_NO_STUB=1) skips the stub install even then, so jax_rmhd sees the
    machine's TRUE absent-MPI state (jax_rmhd._mpi_compat.HAVE_MPI4PY/HAVE_MPI4JAX
    both really False, not faked) -- this is what exercises the real
    comm_backend=None -> 'serial' auto-resolve import path end to end, as opposed
    to the stub's simulation of a size-1 mpi4jax install. Only meaningful in a
    process where bootstrap() (this call) is the first thing to touch jax_rmhd;
    combine with a fresh subprocess if the current process already bootstrapped
    with the stub. Precision/XLA-device setup is identical either way -- only the
    `import local_mpi_stub` line is skipped.

    NB the env form is process-wide, so it reaches conftest's bootstrap() too: a
    whole `RMHD_TEST_NO_STUB=1 pytest tests` session leaves every test that asks
    for comm_backend="mpi4jax"/"jax" with no transport to build on and those fail
    (honestly, with ImportError). It is a single-purpose subprocess knob, not a
    supported suite-wide mode -- `make test` must not set it.
    """
    global _bootstrapped, _stub_active
    if _bootstrapped:
        return
    if "jax_rmhd" in sys.modules:
        raise RuntimeError("bootstrap() called after jax_rmhd was imported -- call it "
                           "as the first statement of the test module.")

    if precision is not None:
        os.environ["RMHD_PRECISION"] = str(precision)
    else:
        os.environ.setdefault("RMHD_PRECISION", "64")
    os.environ.setdefault("MPLBACKEND", "Agg")

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    # Script mode without `pip install -e .`: make the repo root importable too.
    import importlib.util
    if importlib.util.find_spec("jax_rmhd") is None:
        sys.path.insert(0, os.path.dirname(_HERE))

    force_stub = os.environ.get("RMHD_TEST_FORCE_STUB") == "1"
    have_mpi = False
    if not force_stub:
        try:
            from mpi4py import MPI  # noqa: F401
            have_mpi = True
        except Exception:
            # bare except on purpose: a broken MPI runtime can raise more than
            # ImportError at import time.
            have_mpi = False

    if stub is None:
        stub = os.environ.get("RMHD_TEST_NO_STUB") != "1"

    if not have_mpi:
        # Fake XLA host devices for the shard_map ("jax" backend) tests. Only when
        # running single-process without real MPI -- never under a real MPI launch.
        # Must be set before the XLA client initializes (first jax device op).
        # Independent of `stub`: harmless (and needed by other modules) even on the
        # true no-stub path, since nothing there constructs comm_backend="jax".
        n = int(devices or os.environ.get("RMHD_TEST_DEVICES", 4))
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" not in flags:
            os.environ["XLA_FLAGS"] = (flags + f" --xla_force_host_platform_device_count={n}").strip()
        if stub:
            import local_mpi_stub  # noqa: F401  (installs fake mpi4py/mpi4jax on import)
            _stub_active = True
        # else: leave mpi4py/mpi4jax genuinely absent -- the true no-MPI import path.

    _bootstrapped = True


def stub_active():
    return _stub_active


def mpi_size():
    # 1 when mpi4py is genuinely absent AND the stub was skipped (bootstrap(stub=False) /
    # RMHD_TEST_NO_STUB=1): that combination is single-process by definition, and letting
    # the ImportError escape here would break conftest's collection hook outright.
    try:
        from mpi4py import MPI
    except Exception:
        return 1
    return MPI.COMM_WORLD.Get_size()


# --------------------------------------------------------------------------- checks

class Checks:
    """Accumulating PASS/FAIL checker. Prints every check; failures raise together."""

    def __init__(self):
        self.failures = []

    def check(self, name, cond, detail=""):
        ok = bool(cond)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))
        if not ok:
            self.failures.append(name + (f" ({detail})" if detail else ""))
        return ok


@contextlib.contextmanager
def checks():
    """with checks() as c: c.check(...)  -- raises one AssertionError listing all
    failures at exit, so every check in the block still runs and prints."""
    c = Checks()
    yield c
    if c.failures:
        raise AssertionError(f"{len(c.failures)} check(s) failed:\n  - " + "\n  - ".join(c.failures))


# --------------------------------------------------------------- params / grids / state

_DEFAULTS_3D = dict(nx=16, ny=16, nz=8, Lx=2.0 * math.pi, Ly=2.0 * math.pi, Lz=2.0 * math.pi,
                    diss=(0.01, 0.01), hyper=1, cfl_safety=0.5, dt=0.01,
                    adaptive_timestep=False, dims=3)


def _param_kwargs(overrides):
    kw = dict(_DEFAULTS_3D)
    kw.update(overrides)
    if kw.get("dims") == 2:
        kw.pop("nz", None)
        kw.pop("Lz", None)
    return kw


def _ctor_kwargs(kw):
    """Parameters(**kw) kwargs. diss/hyper are RMHD *equation* parameters (params.eqpars)
    since 2026-08-01, but they stay flat, HASHABLE knobs here so ctx()'s cache key works;
    this is the only place that folds them into the eqpars dict. An explicit eqpars=
    override wins for keys it sets."""
    kw = dict(kw)
    eqpars = dict(kw.pop("eqpars", None) or {})
    for name in ("diss", "hyper"):
        if name in kw:
            eqpars.setdefault(name, kw.pop(name))
    kw["eqpars"] = eqpars
    return kw


def fresh_params(**overrides):
    """A new (uncached) Parameters. Use when the test mutates attributes."""
    import jax_rmhd as jr
    return jr.Parameters(**_ctor_kwargs(_param_kwargs(overrides)))


def fake_ranked_params(rank, size, **overrides):
    """Uncached Parameters with spoofed rank/size (snapshot-layout tricks). Never
    pass these to jitted physics -- they lie about the decomposition."""
    p = fresh_params(**overrides)
    p.rank = rank
    p.size = size
    return p


@functools.lru_cache(maxsize=None)
def _ctx_cached(key):
    import jax_rmhd as jr
    p = jr.Parameters(**_ctor_kwargs(dict(key)))
    return p, jr.setup_kgrids(p)


def ctx(**overrides):
    """(params, kgrid), cached by kwargs. Sharing the same Parameters identity across
    tests in a module reuses the jit compile (Parameters hashes by identity). The
    returned objects are shared: NEVER mutate them. kgrid comes from setup_kgrids,
    the only sanctioned constructor."""
    kw = _param_kwargs(overrides)
    return _ctx_cached(tuple(sorted(kw.items())))


def make_state(params, ic=None):
    """A FRESH state (never cached/shared -- buffer donation consumes states)."""
    import jax_rmhd as jr
    if ic is None:
        ic = zero_ic if params.spatial_dimensions == 3 else zero_ic_2d
    return jr.initialize(ic, params)


# ------------------------------------------------------------------- initial conditions

def zero_ic(x, y, z):
    import jax.numpy as jnp
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape, z.shape))


def zero_ic_2d(x, y):
    import jax.numpy as jnp
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape))


def alfven_ic(x, y, z):
    # phi = psi: a single z+ Alfven wave; exact linear solution cos(x)cos(y)cos(z+t).
    import jax.numpy as jnp
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
    return jnp.stack([phi, phi], axis=0)


def tiny_alfven_ic(x, y, z):
    # Amplitude 1e-3: keeps the CFL dt pinned by the (time-independent) 1/dz term,
    # which is what makes the cfl_every agreement tests deterministic.
    import jax.numpy as jnp
    phi = 1e-3 * jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
    return jnp.stack([phi, phi], axis=0)


def multimode_ic(x, y, z):
    # Non-degenerate, multi-mode, O(1) amplitudes (from test_energy_parseval).
    import jax.numpy as jnp
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z) + 0.3 * jnp.sin(2 * x + y)
    psi = jnp.sin(x) * jnp.cos(y) * jnp.sin(z) + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([phi, psi], axis=0)


# ------------------------------------------------------------------------ snapshots

@contextlib.contextmanager
def snap_dir(prefix="rmhd_test_"):
    """Temp dir for snapshot output, removed on exit. Always use this instead of
    cwd-relative data/ paths (stale dirs trip the snapshot layout guard)."""
    d = tempfile.mkdtemp(prefix=prefix)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def managed_manager(params, path, nsnap=2):
    """snapshot_manager_setup wrapper that ALWAYS drains async orbax writes before
    the caller (or snap_dir) removes the directory -- skipping this is a flake."""
    import jax_rmhd as jr
    m = jr.snapshot_manager_setup(params, snap_path=path, nsnap=nsnap)
    try:
        yield m
    finally:
        m.wait_until_finished()
        m.close()


# ----------------------------------------------------------------------- numerics

def fit_order(hs, errs):
    """Least-squares slope of log(err) vs log(h): the observed convergence order."""
    import numpy as np
    hs = np.asarray(hs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


# -------------------------------------------------------------------- script driver

def _script_skip_reason(fn):
    """Script-mode equivalent of conftest's marker skip logic (same rules)."""
    names = {m.name for m in getattr(fn, "pytestmark", [])}
    if not names:
        return None
    import jax
    from jax_rmhd import _precision
    # FIELD precision (RMHD_PRECISION), not the jax_enable_x64 config flag: that flag
    # is now unconditionally on (jax_rmhd/__init__.py) -- only _precision knows
    # whether this is an fp32 or fp64 session.
    x64 = _precision.precision == "64"
    if "fp32" in names and x64:
        return "requires an RMHD_PRECISION=32 session"
    if "fp64" in names and not x64:
        return "requires an RMHD_PRECISION=64 session"
    if "mpi" in names and mpi_size() == 1:
        return "needs a real multi-rank MPI launch"
    if "savio" in names and os.environ.get("RMHD_SAVIO") != "1":
        return "needs cluster resources (set RMHD_SAVIO=1)"
    if "slow" in names and os.environ.get("RMHD_RUNSLOW") != "1":
        return "slow; set RMHD_RUNSLOW=1"
    if "multidev" in names and jax.device_count() < 4:
        return f"needs >=4 devices, have {jax.device_count()}"
    return None


def script_main(module_globals):
    """Run every test_* callable in the module (definition order), print the familiar
    [PASS]/[FAIL] lines and ALL PASS banner, allgather ok across MPI ranks, and
    return a 0/1 exit code. This is what keeps `mpirun -n N python tests/test_x.py`
    working on Savio after a file is converted to pytest-style functions.

    Footer for every test module:
        if __name__ == "__main__":
            import sys
            sys.exit(script_main(globals()))
    """
    tests = [(k, v) for k, v in module_globals.items()
             if k.startswith("test_") and callable(v)]
    ok = True
    for name, fn in tests:
        why = _script_skip_reason(fn)
        if why:
            print(f"[SKIP] {name} -- {why}")
            continue
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:
            traceback.print_exc()
            print(f"[FAIL] {name}")
            ok = False

    if mpi_size() > 1:  # a single bad rank must fail the whole mpirun job
        from mpi4py import MPI
        ok = all(MPI.COMM_WORLD.allgather(bool(ok)))

    print()
    print("ALL PASS" if ok else "SOME CHECKS FAILED -- see [FAIL] lines above")
    return 0 if ok else 1
