# pytest entry point. The first executable lines below MUST run before any test
# module is imported: they put tests/ on sys.path and run bootstrap(), which sets
# RMHD_PRECISION / XLA_FLAGS and installs the MPI stub -- all consumed at
# jax/jax_rmhd import time. Do not import jax or jax_rmhd above bootstrap().
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from _rmhd_testing import bootstrap, mpi_size  # noqa: E402

bootstrap()

import pytest  # noqa: E402

# ---------------------------------------------------------------------------
# Collection control.
#
# Legacy script-style test files (module-level code, no test_* functions) are
# executed at import time when pytest collects them -- that works, but only in the
# fp64 session, because each forces RMHD_PRECISION=64 at its own top while the
# process precision is already fixed. As files are converted to pytest-style
# functions (Phase 2 of docs/TESTING_PLAN.md), remove them from _LEGACY_SCRIPTS.
# ---------------------------------------------------------------------------

_LEGACY_SCRIPTS = [
    "test_backend_jax.py",
    "test_energy_parseval.py",
    "test_forcing_norm_per_step.py",
    "test_forcing_smoke.py",
    "test_halo_width.py",
]

# Never collected by pytest (see docs/TESTING_PLAN.md):
#   - test_backend_jax_mpi.py: argv-driven multi-GPU driver, real MPI only
#   - test_restart_resharding.py: two-phase, two rank counts, real MPI only
#   - test_advection.py: Savio-scale convergence study (Phase 4 adds a fast tier)
#   - test_dissipation.py: minutes-scale study with plot output (Phase 4 converts)
collect_ignore = [
    "data",
    "local_mpi_stub.py",
    "test_backend_jax_mpi.py",
    "test_restart_resharding.py",
    "test_advection.py",
    "test_dissipation.py",
]

if os.environ.get("RMHD_PRECISION", "64") == "32":
    collect_ignore += _LEGACY_SCRIPTS


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run tests marked slow (minutes-scale studies)")


def pytest_collection_modifyitems(config, items):
    import jax

    x64 = bool(jax.config.jax_enable_x64)
    size = mpi_size()
    run_slow = config.getoption("--runslow")
    on_savio = os.environ.get("RMHD_SAVIO") == "1"

    skip_mpi = pytest.mark.skip(reason="needs a real multi-rank MPI launch (size==1 here)")
    skip_savio = pytest.mark.skip(reason="needs cluster resources (set RMHD_SAVIO=1)")
    skip_slow = pytest.mark.skip(reason="slow; pass --runslow")
    skip_fp32 = pytest.mark.skip(reason="requires an RMHD_PRECISION=32 session")
    skip_fp64 = pytest.mark.skip(reason="requires an RMHD_PRECISION=64 session")
    skip_multidev = pytest.mark.skip(reason=f"needs >=4 devices, have {jax.device_count()}")

    for item in items:
        if "mpi" in item.keywords and size == 1:
            item.add_marker(skip_mpi)
        if "savio" in item.keywords and not on_savio:
            item.add_marker(skip_savio)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "fp32" in item.keywords and x64:
            item.add_marker(skip_fp32)
        if "fp64" in item.keywords and not x64:
            item.add_marker(skip_fp64)
        if "multidev" in item.keywords and jax.device_count() < 4:
            item.add_marker(skip_multidev)
