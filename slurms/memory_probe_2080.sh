#!/bin/bash
#SBATCH --job-name=memprobe_2080
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:4
#SBATCH --time=01:30:00
#SBATCH --output=memprobe_2080_%j.out
#SBATCH --error=memprobe_2080_%j.err
#
# bench/memory_probe.py on one GTX 2080Ti (11 GB), then the 4-GPU sharded FD-z row.
# TAG labels the measurement point: TAG=baseline|postF|postZ sbatch slurms/memory_probe_2080.sh
# Env blocks copied from slurms/bench_phase3_2080_scale.sh (known good on this pool).

set -uo pipefail

module purge
module load anaconda3 gcc openmpi
source activate jax_gpu
unset PYTHONPATH
export PYTHONNOUSERSITE=1
NVLIBS=$("$HOME/.conda/envs/jax_gpu/bin/python" -c "import nvidia,os;print(':'.join(os.path.join(p,d,'lib') for p in nvidia.__path__ for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p,d,'lib'))))" 2>/dev/null || true)
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_P2P_DISABLE=1
export RMHD_REQUIRE_GPU=1
export MPI4JAX_USE_CUDA_MPI=${CUDA_MPI:-0}
MPI_MODE=${MPI_MODE:-pmix}

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=${REPO:-$HOME/taranis}
PROBE=$REPO/bench/memory_probe.py
TAG=${TAG:-baseline}
OUT=$SLURM_SUBMIT_DIR/memprobe_2080_${TAG}_$SLURM_JOB_ID
mkdir -p "$OUT"
echo "python=$PY tag=$TAG out=$OUT"

# The probe isolates each case in a subprocess by default on GPU: peak_bytes_in_use is a
# process high-water mark, so a shared process would report 0 delta for every case after
# the largest one, and an OOM would poison the allocator for the rest of the matrix.

# 1 GPU, fp32: the gtx2080 profile
export TARANIS_PRECISION=32
srun --mpi=$MPI_MODE --ntasks=1 --gres=gpu:GTX2080TI:1 --gpu-bind=single:1 \
     "$PY" -u "$PROBE" --profile gtx2080 --precision 32 --precision-check \
     --tag "$TAG" --out "$OUT/gtx2080_fp32.json" 2>&1 | grep -v "bit precision"

# 1 GPU, fp64: the RMHD rows one size down (the profile halves the grid at precision 64)
export TARANIS_PRECISION=64
srun --mpi=$MPI_MODE --ntasks=1 --gres=gpu:GTX2080TI:1 --gpu-bind=single:1 \
     "$PY" -u "$PROBE" --profile gtx2080 --precision 64 --precision-check \
     --tag "$TAG" --out "$OUT/gtx2080_fp64.json" 2>&1 | grep -v "bit precision"

# F1 granularity pair (post-F runs only; a no-op row with a note on a pre-F1 tree)
export TARANIS_PRECISION=32
for CHUNK in 1 2; do
  srun --mpi=$MPI_MODE --ntasks=1 --gres=gpu:GTX2080TI:1 --gpu-bind=single:1 \
       "$PY" -u "$PROBE" --profile gtx2080 --precision 32 --grad-chunk $CHUNK \
       --cases rmhd_fdz,rmhd_zspec --tag "${TAG}_chunk${CHUNK}" \
       --out "$OUT/gtx2080_fp32_chunk${CHUNK}.json" 2>&1 | grep -v "bit precision"
done

# 4 GPUs, fp32: the sharded FD-z row. Memory is reported PER DEVICE (the probe divides nz
# by the mesh size). The probe forces in-process execution when a comm_backend="jax" case
# is present, and each rank suffixes its own output file (rank 0 keeps the plain name).
srun --mpi=$MPI_MODE --ntasks=4 --ntasks-per-node=4 --gres=gpu:GTX2080TI:4 \
     "$PY" -u "$PROBE" --profile gtx2080-sharded --precision 32 \
     --tag "$TAG" --out "$OUT/gtx2080_fp32_jax4.json" 2>&1 | grep -v "bit precision"

echo "done: $OUT"
