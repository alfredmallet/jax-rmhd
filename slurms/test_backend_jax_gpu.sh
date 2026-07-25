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

# Block ~/.local user-site packages from shadowing the env (stray mpi4py bit us once).
export PYTHONNOUSERSITE=1

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

# Do NOT set/override CUDA_VISIBLE_DEVICES -- srun --gpu-bind=single:1 gives each task
# exactly one GPU, which each rank prints below (local_devices=... must show 1 distinct
# GPU per rank). If this srun launch cannot start mpi4py in your env, fall back to
#   mpirun -n $NRANK ...
# which leaves all 4 GPUs visible to every rank; comms._local_device_ids then claims the
# node-local rank's GPU automatically (read-only on CUDA_VISIBLE_DEVICES).
MPI_MODE=${MPI_MODE:-pmix}  # probe job 35845619: this openmpi is --without-pmi + external PMIx; pmi2 fails, pmix works   # same knob as the bench scripts; see `srun --mpi=list`
LAUNCH="srun --mpi=$MPI_MODE --ntasks=$NRANK --cpus-per-task=$SLURM_CPUS_PER_TASK --gpu-bind=single:1"

rm -rf "$OUT"

echo "=== phase 1: fresh run, backend=mpi4jax (${NRANK} ranks) ==="
time $LAUNCH "$PY" -u "$DRIVER" mpi4jax "$OUT/mpi4jax"

echo "=== phase 2: fresh run, backend=jax, same seed ==="
time $LAUNCH "$PY" -u "$DRIVER" jax "$OUT/jax"

echo "=== phase 3: compare mpi4jax vs jax (expect rel < 1e-12) ==="
"$PY" -u "$DRIVER" --compare "$OUT/mpi4jax" "$OUT/jax"

echo "=== phase 4a: jax backend restarting from the mpi4jax-written snapshot ==="
time $LAUNCH "$PY" -u "$DRIVER" jax "$OUT/xr_jax" "$OUT/mpi4jax"

echo "=== phase 4b: mpi4jax restarting from the jax-written snapshot ==="
time $LAUNCH "$PY" -u "$DRIVER" mpi4jax "$OUT/xr_mpi4jax" "$OUT/jax"

echo "=== phase 5: compare the two cross-backend restarts ==="
# looser tolerance: the two restarts start from snapshots that already differ at roundoff
RMHD_CMP_TOL=1e-10 "$PY" -u "$DRIVER" --compare "$OUT/xr_jax" "$OUT/xr_mpi4jax"

echo "=== done: every phase above must end in ALL PASS ==="
