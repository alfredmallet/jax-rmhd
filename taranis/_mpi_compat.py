# the only module in taranis allowed to import mpi4py/mpi4jax at module scope. comms.py
# and config.py go through the flags/objects exported here, so the package imports cleanly
# on a machine with no MPI toolchain (where Parameters auto-resolves comm_backend="serial").
import os

try:
    from mpi4py import MPI
    HAVE_MPI4PY = True
except Exception:
    MPI = None
    HAVE_MPI4PY = False

try:
    import mpi4jax
    HAVE_MPI4JAX = True
except Exception:
    mpi4jax = None
    HAVE_MPI4JAX = False


class _NullComm:
    # MPI.COMM_WORLD stand-in for the single-process "serial" backend.
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, obj, root=0):
        return obj


# per-process world size, written by the launcher itself
_LAUNCHER_SIZE_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE")
# allocation-wide, exported to every process in a batch script.
# a hint about the job, not proof that this process is a rank of it.
_ALLOC_SIZE_VARS = ("SLURM_NTASKS", "SLURM_STEP_NUM_TASKS")

def _env_int(name):
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return None

def launcher_world_size():
    # -> (world_size, definitive). 
    # definitive=True: environment proves this process is one rank of a world_size-rank job
    # definitive=False: world_size is only a hint -> warn, never error.
    for var in _LAUNCHER_SIZE_VARS:
        n = _env_int(var)
        if n is not None:
            return n, True
    hint = max([n for n in (_env_int(v) for v in _ALLOC_SIZE_VARS) if n is not None] or [1])
    # PMIx launchers (srun --mpi=pmix) set PMIX_RANK per process but no size variable; the
    # allocation's task count is then the real world size.
    return hint, "PMIX_RANK" in os.environ
