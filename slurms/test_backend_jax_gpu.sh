#!/bin/bash
#SBATCH --job-name=backend_jax_gpu
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#
# savio4_gpu is per-CORE scheduled with a 4:1 CPU:GPU ratio, and FCA regular priority caps
# a user at 16 CPUs = 4 A5000s: 4 tasks x 1 GPU x 4 cpus-per-task is exactly that budget.
# The explicit GPU type in --gres is REQUIRED for FCA jobs (missing it -> QOSMinGRES pend).
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:4
#
#SBATCH --time=00:40:00
#SBATCH --output=backend_jax_gpu_%j.out
#SBATCH --error=backend_jax_gpu_%j.err

# T9 correctness job: comm_backend="jax" (shard_map/NCCL) vs "mpi4jax" on 4 GPUs, same
# seed, plus a cross-backend restart in both directions. fp64 on purpose -- A5000 fp64 is
# ~1/32 fp32, but this is a CORRECTNESS run (tiny grid), and fp64 makes the tolerance sharp.

module purge
module load anaconda3 gcc openmpi cuda

source activate jax_gpu

# The anaconda3 module sets PYTHONPATH to the BASE anaconda site-packages, whose own
# nvidia-* packages shadow the env's (root cause of "cuSPARSE library was not found").
# GPU jobs never need PYTHONPATH (code selection uses RMHD_PKG) -- drop it.
unset PYTHONPATH
echo "python=$(which python)"

# Block ~/.local user-site packages from shadowing the env (stray mpi4py bit us once).
export PYTHONNOUSERSITE=1

# jax's CUDA plugin can fail to dlopen the pip-bundled nvidia libs by bare soname inside
# jobs ("cuSPARSE library was not found") even when the env is complete -- put every
# nvidia/*/lib dir on LD_LIBRARY_PATH so plain-name dlopen always resolves.
NVLIBS=$("$HOME/.conda/envs/jax_gpu/bin/python" -c "import nvidia,os;print(':'.join(os.path.join(p,d,'lib') for p in nvidia.__path__ for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p,d,'lib'))))" 2>/dev/null || true)
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "NVLIBS=${NVLIBS:-EMPTY}"   # visible proof in the .out that this block ran

# One-task compute-node probe: prints what the actual job python sees BEFORE any phase runs.
srun --mpi=$MPI_MODE --ntasks=1 --cpus-per-task=4 "$HOME/.conda/envs/jax_gpu/bin/python" -c "import os,sys,nvidia,jax; print('probe PYTHONPATH=',os.environ.get('PYTHONPATH')); print('probe nvidia=',nvidia.__path__); print('probe devices=',jax.devices())" || true

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/jax_rmhd
DRIVER="$REPO/tests/test_backend_jax_mpi.py"
OUT=data/test_backend_jax
NRANK=4

export RMHD_PRECISION=64
export RMHD_PKG=$REPO          # printed in every result line, per the Phase 3 ground rules
export RMHD_NX=64
export RMHD_NZ=32
export RMHD_TEND=1.0

# Do NOT set/override CUDA_VISIBLE_DEVICES. TWO launch modes, one per backend (job 35861380):
# - mpi4jax NEEDS --gpu-bind=single:1 (one visible GPU per task) or every rank piles onto
#   GPU 0 -- it never calls jax.distributed, so nothing else assigns devices.
# - the jax/NCCL backend must NOT use it: with only its own GPU visible per process, NCCL's
#   first collective dies (ncclGroupEnd, cuda error 101 'invalid device ordinal') because
#   same-node P2P/SHM transport needs peer GPUs visible. Leave all job GPUs visible and
#   comms._local_device_ids pins each process to its node-local rank's ordinal.
MPI_MODE=${MPI_MODE:-pmix}  # probe job 35845619: this openmpi is --without-pmi + external PMIx; pmi2 fails, pmix works   # same knob as the bench scripts; see `srun --mpi=list`
LAUNCH_MPI4JAX="srun --mpi=$MPI_MODE --ntasks=$NRANK --cpus-per-task=$SLURM_CPUS_PER_TASK --gpu-bind=single:1"
LAUNCH_JAX="srun --mpi=$MPI_MODE --ntasks=$NRANK --cpus-per-task=$SLURM_CPUS_PER_TASK"

# Hang forensics (job 35861466: phase 2 stalled silently at the first NCCL collective):
# NCCL_DEBUG=INFO logs communicator bring-up per rank to stderr; `timeout` turns a hang
# into exit 124 after 15 min so later phases still emit their markers.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
TMO="timeout 900"
# NCCL: PCIe P2P between GPUs is BROKEN on savio4_gpu nodes (repro 2026-07-26: rings
# connect, first collective hangs under both CUMEM and legacy-IPC P2P; bench/nccl_repro.py
# passes only with P2P off -> SHM transport). Likely PCIe ACS config -- reported to Savio
# support; revisit if they fix it (SHM adds host-memory hops, so NCCL numbers here
# UNDERSTATE a P2P/NVLink-capable cluster).
export NCCL_P2P_DISABLE=1
# Per-rank python stack dumps every 120 s while hung (see test_backend_jax_mpi.py).
export RMHD_DEBUG_HANG=1

rm -rf "$OUT"

echo "=== phase 1: fresh run, backend=mpi4jax (${NRANK} ranks) ==="
time $TMO $LAUNCH_MPI4JAX "$PY" -u "$DRIVER" mpi4jax "$OUT/mpi4jax"

echo "=== phase 2: fresh run, backend=jax, same seed ==="
time $TMO $LAUNCH_JAX "$PY" -u "$DRIVER" jax "$OUT/jax"

echo "=== phase 3: compare mpi4jax vs jax (expect rel < 1e-12) ==="
"$PY" -u "$DRIVER" --compare "$OUT/mpi4jax" "$OUT/jax"

echo "=== phase 4a: jax backend restarting from the mpi4jax-written snapshot ==="
time $TMO $LAUNCH_JAX "$PY" -u "$DRIVER" jax "$OUT/xr_jax" "$OUT/mpi4jax"

echo "=== phase 4b: mpi4jax restarting from the jax-written snapshot ==="
time $TMO $LAUNCH_MPI4JAX "$PY" -u "$DRIVER" mpi4jax "$OUT/xr_mpi4jax" "$OUT/jax"

echo "=== phase 5: compare the two cross-backend restarts ==="
# looser tolerance: the two restarts start from snapshots that already differ at roundoff
RMHD_CMP_TOL=1e-10 "$PY" -u "$DRIVER" --compare "$OUT/xr_jax" "$OUT/xr_mpi4jax"

echo "=== done: every phase above must end in ALL PASS ==="
