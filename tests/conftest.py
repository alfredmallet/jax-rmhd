# pytest entry point. The first executable lines below MUST run before any test
# module is imported: they put tests/ on sys.path and run bootstrap(), which sets
# TARANIS_PRECISION / XLA_FLAGS and installs the MPI stub -- all consumed at
# jax/taranis import time. Do not import jax or taranis above bootstrap().
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
# ---------------------------------------------------------------------------

# Never collected by pytest:
#   - test_backend_jax_mpi.py: argv-driven multi-GPU driver, real MPI only
#   - test_restart_resharding.py: two-phase, two rank counts, real MPI only
collect_ignore = [
    "data",
    "local_mpi_stub.py",
    "test_backend_jax_mpi.py",
    "test_restart_resharding.py",
]

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run tests marked slow (minutes-scale studies)")


def pytest_collection_modifyitems(config, items):
    import jax
    from taranis import _precision

    # FIELD precision (TARANIS_PRECISION), not the jax_enable_x64 config flag: that flag
    # is now unconditionally on (taranis/__init__.py), so it no longer distinguishes
    # an "fp32 session" from an "fp64 session" -- only _precision knows that.
    x64 = _precision.precision == "64"
    size = mpi_size()
    run_slow = config.getoption("--runslow")
    on_savio = os.environ.get("RMHD_SAVIO") == "1"

    skip_mpi = pytest.mark.skip(reason="needs a real multi-rank MPI launch (size==1 here)")
    skip_savio = pytest.mark.skip(reason="needs cluster resources (set RMHD_SAVIO=1)")
    skip_slow = pytest.mark.skip(reason="slow; pass --runslow")
    skip_fp32 = pytest.mark.skip(reason="requires a TARANIS_PRECISION=32 session")
    skip_fp64 = pytest.mark.skip(reason="requires a TARANIS_PRECISION=64 session")
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
