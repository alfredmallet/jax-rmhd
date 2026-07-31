# Installs fake mpi4py/mpi4jax modules for SINGLE-PROCESS test runs on machines without MPI.
# Import this BEFORE jax_rmhd. Semantics are exact for size-1: sendrecv is a self-send
# (periodic topology, matching tags) and allreduce is the identity.
# Usage: PYTHONPATH=.:tests python -c "import local_mpi_stub, runpy; runpy.run_path('tests/test_forcing_smoke.py', run_name='__main__')"
import sys
import types


class _FakeCartComm:
    # Mimics the size-1 periodic cartesian communicator: only neighbor is self (rank 0).
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Shift(self, direction=0, disp=1):
        return (0, 0)


class _FakeComm(_FakeCartComm):
    # Mimics MPI.COMM_WORLD for a single process.
    def Create_cart(self, dims, periods, reorder=False):
        return _FakeCartComm()


def _install():
    # Build fake mpi4py.MPI and mpi4jax modules and register them in sys.modules.
    mpi = types.ModuleType("mpi4py.MPI")
    mpi.COMM_WORLD = _FakeComm()
    mpi.SUM = "SUM"
    mpi.MAX = "MAX"
    mpi.MIN = "MIN"
    mpi.Get_library_version = lambda: "local_mpi_stub (no MPI)"

    mpi4py = types.ModuleType("mpi4py")
    mpi4py.MPI = mpi

    mpi4jax = types.ModuleType("mpi4jax")
    # Self-send: with one rank and matching tags the received buffer is the sent buffer.
    mpi4jax.sendrecv = lambda sendbuf, recvbuf, source, dest, comm=None, sendtag=0, recvtag=0, **kw: sendbuf
    # Reductions over one rank are the identity.
    mpi4jax.allreduce = lambda x, op=None, comm=None, **kw: x

    sys.modules["mpi4py"] = mpi4py
    sys.modules["mpi4py.MPI"] = mpi
    sys.modules["mpi4jax"] = mpi4jax

    # jax.distributed.initialize can hang/crash on machines behind proxies; single-process runs don't need it.
    import jax

    jax.distributed.initialize = lambda *a, **k: print("local_mpi_stub: skipped jax.distributed.initialize")


# Idempotent: a second import (or an already-present real mpi4jax) must not re-install.
if "mpi4jax" not in sys.modules:
    _install()
