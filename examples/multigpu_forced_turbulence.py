"""Forced 3D RMHD turbulence, sized for several GPUs on a cluster.

This is the reference *production* driver for `comm_backend="jax"` (shard_map/NCCL, the
GPU backend -- docs/performance.md "Backends: why two").  It is meant to be launched from
`slurms/forced_turbulence_multigpu.sh`; everything it needs is an environment variable, so
scaling a run up or restarting one never means editing this file.

    srun ... python examples/multigpu_forced_turbulence.py

Every knob, with its default:

    TARANIS_PRECISION  32          field precision; read at import (32 on Savio's
                                   workstation GPUs -- their fp64 runs at ~1/32 rate)
    TARANIS_BACKEND    jax         "jax" (multi-GPU), "mpi4jax" (CPU cluster / 1 GPU),
                                   "serial" or "auto" for the auto-resolved default
    TARANIS_NX         256         perpendicular resolution (nx = ny)
    TARANIS_NZ         128         z resolution; MUST be divisible by the GPU count
    TARANIS_TEND       10.0        run until t >= this (Alfven times)
    TARANIS_TSNAP      0.5         snapshot cadence in t
    TARANIS_NSNAP      40          how many snapshots to keep on disk (orbax max_to_keep)
    TARANIS_SCHEME     lsrk54      timestepper (the RMHD production choice)
    TARANIS_SNAPDIR    data/multigpu_forced_turbulence
    TARANIS_EPS        0.3         total energy injection rate, split evenly z+/z-
    TARANIS_DISS       1e-5        nu = eta
    TARANIS_HYPER      3           dissipation order ((-1)^h nu k^(2h))

Restarting is just resubmitting: if `TARANIS_SNAPDIR` already holds snapshots the run
picks up from the newest one and extends to `TARANIS_TEND`, so a run longer than one
walltime allocation is a chain of identical `sbatch` calls.  Nothing here is
GPU-specific except the defaults -- `TARANIS_BACKEND=mpi4jax mpirun -n 32 python ...`
runs the same problem on CPU nodes.
"""
import os
import time

import numpy as np


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


NX = _env_int("TARANIS_NX", 256)
NZ = _env_int("TARANIS_NZ", 128)
T_END = _env_float("TARANIS_TEND", 10.0)
T_SNAP = _env_float("TARANIS_TSNAP", 0.5)
NSNAP = _env_int("TARANIS_NSNAP", 40)
SCHEME = os.environ.get("TARANIS_SCHEME", "lsrk54")
SNAP_PATH = os.environ.get("TARANIS_SNAPDIR", "data/multigpu_forced_turbulence")
EPS = _env_float("TARANIS_EPS", 0.3)
DISS = _env_float("TARANIS_DISS", 1e-5)
HYPER = _env_int("TARANIS_HYPER", 3)
BACKEND = os.environ.get("TARANIS_BACKEND", "jax")
if BACKEND in ("", "auto", "none"):
    BACKEND = None

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import taranis as jr
import taranis.diagnostics as diag
import taranis.snapshot_io as sn
from taranis import comms

# --------------------------------------------------------------------------------------
# Parameters FIRST.  Constructing it resolves the transport, and for comm_backend="jax"
# that runs jax.distributed.initialize() and builds the device mesh -- which jax refuses
# once the local backend exists.  So NOTHING above this line may touch a jax device
# (no jax.devices(), no jnp array, no jit call); the per-rank device report below is
# deliberately placed after it.
# --------------------------------------------------------------------------------------
params = jr.Parameters(
    nx=NX, ny=NX, nz=NZ, Lx=2 * np.pi, Ly=2 * np.pi, Lz=2 * np.pi, dims=3,
    eqpars={"diss": (DISS, DISS), "hyper": HYPER},
    cfl_safety=0.5, comm_backend=BACKEND,
    forcing=True, forcing_mode="elsasser",
    forcing_power_elsasser=(EPS / 2, EPS / 2),   # sum to the total target rate EPS
    forcing_tau=1.0, fshell=(1, 3), forcing_seed=42, forcing_scale_max=1.0,
)

# One line per rank, before any timing: this is the binding check.  Every rank must report
# platform=cuda and exactly ONE local device, with distinct ids across the ranks on a node.
# Two ranks on the same id (or one rank listing every GPU) means the launch was wrong --
# see docs/SAVIO_GPU_SETUP.md section 2.  Under comm_backend="jax" the mesh spans all of
# them, so global_devices must equal the total GPU count of the job.
print(f"[rank {params.rank}/{params.size}] backend={params.comm_backend} "
      f"platform={jax.devices()[0].platform} local_devices={jax.local_devices()} "
      f"global_devices={jax.device_count()} precision={os.environ.get('TARANIS_PRECISION', '32')} "
      f"grid={NX}^2x{NZ}", flush=True)

is_control = params.rank == 0
# z is split across the DEVICE MESH on the "jax" backend (one device per process in
# production, so the two agree there) and across MPI ranks on every other backend.
nz_split = jax.device_count() if params.comm_backend == "jax" else params.size
if is_control:
    print(f"# z split {nz_split} ways: nz_local = {NZ // nz_split} planes each, "
          f"snapshots -> {SNAP_PATH}", flush=True)


def energies(state, kgrid):
    """(E_kin, E_mag).

    `diagnostics.energy` allreduces over the z decomposition, and on the "jax" backend
    that allreduce is a `lax.psum` over the mesh axis -- calling it from outside a
    shard_map raises `NameError: unbound axis name: z`.  `comms.shard_call` supplies the
    context; `out_specs` describes the RETURN value, here a pair of replicated scalars.
    Same recipe for any other diagnostic that reduces over z.
    """
    if params.comm_backend == "jax":
        fn = comms.shard_call(lambda s, kg: diag.energy(s, kg, params), params, kgrid,
                              out_specs=(P(), P()))
        return fn(state, kgrid)
    return diag.energy(state, kgrid, params)


# Records the constructor arguments to SNAP_PATH/params.json (and creates the directory).
# Collective -- every rank calls it.  A second run with different physics against the same
# directory is a hard error here rather than a silently mixed snapshot series.
params.save(SNAP_PATH)

kgrid = jr.setup_kgrids(params)
mngr = jr.snapshot_manager_setup(params, SNAP_PATH, nsnap=NSNAP)

# Restart if this directory already holds snapshots, otherwise start from rest and let the
# forcing spin the fields up.  `get_saved_steps` takes params so it reads the right layout
# (the "jax" backend writes ONE shared flat directory of z-sharded arrays; mpi4jax writes
# snap_path/<rank>/<step> -- either backend can restart from either, see
# docs/checkpointing.md).
saved = sn.get_saved_steps(SNAP_PATH, params)
if saved:
    isnap = max(saved)
    state = sn.load_snapshot(isnap, SNAP_PATH, params)
    if is_control:
        print(f"# restarting from snapshot {isnap} at t={float(state.t):.4f}", flush=True)
else:
    state = jr.initialize(
        lambda x, y, z: jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape, z.shape)),
        params)
    if is_control:
        print("# fresh start from rest", flush=True)

if float(state.t) >= T_END:
    if is_control:
        print(f"# nothing to do: t={float(state.t):.4f} already >= TARANIS_TEND={T_END}")
    raise SystemExit(0)

E0 = energies(state, kgrid)
if is_control:
    print(f"# t={float(state.t):.4f} E_kin={float(E0[0]):.6e} E_mag={float(E0[1]):.6e}",
          flush=True)

nblock = jr.estimate_good_nblock(state, kgrid, params, T_SNAP, T_END,
                                 t_last_snap=float(state.t), nblock_min=10)
if is_control:
    print(f"# nblock={nblock} steps per scanned block, scheme={SCHEME}", flush=True)

t0 = time.time()
# simulate_scan (not simulate): it scans a fixed block of steps, which is what keeps the
# GPU busy between host round-trips.  The first call compiles -- tens of seconds is normal,
# and RMHD_COMPILATION_CACHE makes a resubmitted run skip it.
end_state = jr.simulate_scan(state, kgrid, params, nblock, t_snap=T_SNAP, t_end=T_END,
                             mngr=mngr, schemestr=SCHEME, save=True)
wall = time.time() - t0

E1 = energies(end_state, kgrid)
if is_control:
    print(f"# t={float(end_state.t):.4f} E_kin={float(E1[0]):.6e} E_mag={float(E1[1]):.6e}",
          flush=True)
    print(f"# wall {wall:.1f} s; snapshots {sn.get_saved_steps(SNAP_PATH, params)}",
          flush=True)
    print("DONE", flush=True)
