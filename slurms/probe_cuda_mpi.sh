#!/bin/bash
#SBATCH --job-name=probe_cuda_mpi
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:2
#SBATCH --time=00:20:00
#SBATCH --output=probe_cuda_mpi_%j.out
#SBATCH --error=probe_cuda_mpi_%j.err

# T8 pre-flight: does the `openmpi` module do CUDA-aware transfers, and does mpi4jax use it?
# Answers three questions in one submission: (1) ompi_info's CUDA capability, (2) which srun
# PMI modes exist for launching mpi4py, (3) device-buffer sendrecv correctness + latency with
# MPI4JAX_USE_CUDA_MPI=1 vs =0. See docs/SAVIO_GPU_SETUP.md "CUDA-aware MPI audit".
# NOTE: not `set -e` -- a failing/segfaulting CUDA-MPI probe IS one of the two expected
# outcomes and must not abort the rest of the audit.
set -uo pipefail

module purge
module load anaconda3 gcc openmpi
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

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/jax_rmhd
PROBE=$REPO/bench/probe_cuda_mpi.py
# pmi2 is the usual Savio openmpi build; if MPI init fails or every rank reports rank 0/1
# ranks, rerun with MPI_MODE=pmix (the `srun --mpi=list` output below lists what exists).
MPI_MODE=${MPI_MODE:-pmix}  # probe job 35845619: this openmpi is --without-pmi + external PMIx; pmi2 fails, pmix works

echo "=== openmpi module ==="
module list 2>&1
which mpirun ompi_info

echo "=== ompi_info CUDA audit (empty/absent output => NOT CUDA-aware) ==="
ompi_info 2>&1 | grep -i -E "cuda|extensions" || echo "  (no cuda/extensions lines)"
ompi_info --parameters all all 2>&1 | grep -i cuda || echo "  (no cuda MCA parameters)"
echo "smcuda/uct components present?"
ompi_info 2>&1 | grep -i -E "smcuda|ucx" || echo "  (none)"

echo "=== srun PMI modes available ==="
srun --mpi=list 2>&1 || true

echo "=== probe with MPI4JAX_USE_CUDA_MPI=0 (host staging -- the F2(a) baseline) ==="
MPI4JAX_USE_CUDA_MPI=0 srun --mpi=$MPI_MODE --ntasks=2 --cpus-per-task=4 \
    --gres=gpu:A5000:2 --gpu-bind=single:1 "$PY" -u "$PROBE" 512 2>&1
    # --gres is PER NODE: :2 (not :1) is what gives the 2 tasks one distinct GPU each

echo "=== probe with MPI4JAX_USE_CUDA_MPI=1 (device buffers straight to MPI) ==="
# Expected failure mode on a non-CUDA-aware build: segfault or garbage data (correct=False).
MPI4JAX_USE_CUDA_MPI=1 srun --mpi=$MPI_MODE --ntasks=2 --cpus-per-task=4 \
    --gres=gpu:A5000:2 --gpu-bind=single:1 "$PY" -u "$PROBE" 512 2>&1
    # --gres is PER NODE: :2 (not :1) is what gives the 2 tasks one distinct GPU each

echo "=== audit complete: record which setting ran clean, and the us/halo-pair of each ==="
