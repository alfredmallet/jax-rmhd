#!/bin/bash
#SBATCH --job-name=bench_p3_a40
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#
# savio3_gpu is PER-CORE scheduled at 8 CPUs per A40: 16 GPUs x 8 = 128 cores x 3.67 SU/core-hr
# ~= 470 SU/hr, so this job is capped at 1 hour. --gres is PER NODE (4 A40 x 4 nodes = 16).
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:4
#SBATCH --time=01:00:00
#SBATCH --output=bench_p3_a40_%j.out
#SBATCH --error=bench_p3_a40_%j.err
#
# Only some savio3_gpu A40 nodes have 4 GPUs, and the regular-FCA A40 pool is 16 GPUs
# CLUSTER-WIDE -- a 16-GPU job waits for all of them. If it pends too long, switch to
# --nodes=8 --ntasks-per-node=2 --gres=gpu:A40:2 (same 16 GPUs, 2-GPU nodes) and set
# GPUS_PER_NODE=2 below.

# T8 multi-node scaling: 4/8/16 A40 at fp32, the NASA-cluster proxy for NCCL-over-IB. This is
# where comm cost dominates and where A2's jax backend has to win if it wins anywhere.
# Strong scaling on a fixed grid; compare ms/step against the 1-GPU point in bench_phase3_a5000.

set -uo pipefail   # not -e: one crashed case must not abort the matrix

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

# NCCL: PCIe P2P between GPUs is BROKEN on savio4_gpu nodes (repro 2026-07-26: rings
# connect, first collective hangs under both CUMEM and legacy-IPC P2P; bench/nccl_repro.py
# passes only with P2P off -> SHM transport). Likely PCIe ACS config -- reported to Savio
# support; revisit if they fix it (SHM adds host-memory hops, so NCCL numbers here
# UNDERSTATE a P2P/NVLink-capable cluster).
export NCCL_P2P_DISABLE=1
export RMHD_REQUIRE_GPU=1  # abort any case where a rank silently falls back to CPU (job 35894622)

export TARANIS_PRECISION=32   # A40 fp64 is ~1/32 of fp32; the fp64 anchor is the V100 job

# Set CUDA_MPI=1 only if slurms/probe_cuda_mpi.sh showed the openmpi module is CUDA-aware.
# Never set CUDA_VISIBLE_DEVICES -- Slurm scopes it per task.
export MPI4JAX_USE_CUDA_MPI=${CUDA_MPI:-0}

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/taranis
BENCH=$REPO/bench/bench_phase1.py
export RMHD_PKG=$REPO   # bench prints pkg= so the imported package is verifiable

RUN_JAX=${RUN_JAX:-0}          # jax-backend rows are opt-in (script runs if T9 is reverted)
MPI_MODE=${MPI_MODE:-pmix}  # probe job 35845619: this openmpi is --without-pmi + external PMIx; pmi2 fails, pmix works
# derived from the allocation, so switching the header to --nodes=8 --ntasks-per-node=2
# --gres=gpu:A40:2 (the lower-pend-risk 16-GPU shape) needs NO edits below this line
GPUS_PER_NODE=${GPUS_PER_NODE:-$(( SLURM_NTASKS / SLURM_NNODES ))}

NX=512; NZ=256          # nz_local = 16 at 16 GPUs; 512^2x256 fp32 ~ 0.6 GB/state on 48 GB cards
STEPS="nb20 nr4"

# One rank per GPU across nodes; --gres is per node, so ask for ceil(n/GPUS_PER_NODE) nodes
# (ceil + the gpn clamp keep n < GPUS_PER_NODE from asking for --nodes=0).
run() { local n=$1; shift; local gpn=$(( n < GPUS_PER_NODE ? n : GPUS_PER_NODE )); \
        local bind="--gpu-bind=single:1"; case "$*" in *backend=jax*) bind="";; esac; \
        local nn=$(( (n + gpn - 1) / gpn )); \
        srun --mpi=$MPI_MODE --nodes="$nn" --ntasks="$n" --ntasks-per-node="$gpn" \
        --cpus-per-task=8 --gres=gpu:A40:"$gpn" $bind \
        "$PY" -u "$BENCH" "$@" 2>&1 | grep -v "bit precision" || true; }

echo "=== config: precision=$TARANIS_PRECISION cuda_mpi=$MPI4JAX_USE_CUDA_MPI run_jax=$RUN_JAX grid=${NX}^2x$NZ ==="

pass() {
    run 4  a4    3d_forced donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS
    run 8  a8    3d_forced donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS
    run 16 a16   3d_forced donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS
    run 16 a16U  3d        donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS  # unforced pair with a16
    run 16 a16un 3d_forced donate nx$NX nz$NZ nps cfl1 halo_late unroll $STEPS  # T3 GPU candidate
    run 16 a16he 3d_forced donate nx$NX nz$NZ nps cfl1 halo_early $STEPS  # T7 hook on (pair with a16)
}

echo "=== pass 1: mpi4jax backend ==="
pass
echo "=== pass 2: mpi4jax backend ==="
pass

if [ "$RUN_JAX" = "1" ]; then
    echo "=== jax backend (shard_map/NCCL): same scaling points + option pairs, 2 passes ==="
    for _ in 1 2; do
        run 4  j4     3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
        run 8  j8     3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
        run 16 j16    3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
        run 16 j16hl  3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax halo_late $STEPS
        run 16 j16un  3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax unroll    $STEPS
    done
fi

# One profiled case (F2(b) stream syncs at scale); reading it: docs/SAVIO_GPU_SETUP.md "Reading the profile".
echo "=== profile: 16 GPU forced, trace under \$SLURM_SUBMIT_DIR/prof_p3a40 ==="
export RMHD_PROFILE_DIR=$SLURM_SUBMIT_DIR/prof_p3a40_$SLURM_JOB_ID
run 16 a16prof 3d_forced donate nx$NX nz$NZ nps cfl1 halo_late $STEPS --profile
