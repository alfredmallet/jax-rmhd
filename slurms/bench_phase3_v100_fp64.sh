#!/bin/bash
#SBATCH --job-name=bench_p3_v100
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3_gpu
#SBATCH --qos=v100_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#
# savio3_gpu is PER-CORE scheduled at 4 CPUs per V100: 2 tasks x 4 = 8 CPUs on one 2-GPU node.
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:2
#SBATCH --time=00:50:00
#SBATCH --output=bench_p3_v100_%j.out
#SBATCH --error=bench_p3_v100_%j.err
#
# Only 2 V100 GPUs exist for regular FCA priority -- this job takes both, and there is no
# larger V100 configuration available. Expect queue wait.

# T8 fp64 anchor. Phase 1's lesson: fp32 flatters comm savings ~3x, and all keep/revert
# decisions are made on fp64 -- but V100 is the ONLY Savio GPU with usable fp64 (A5000/A40/
# L40/2080Ti run fp64 at ~1/32 of fp32). So: 1-vs-2 GPU fp64 here, scaling on the fp32 jobs.

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

# XLA latency-hiding scheduler (measured 2026-08-21, job 37912751): without it XLA emits the
# halo ppermutes and the two allreduces as async -start/-done pairs but schedules ZERO
# instructions between them, so comms are serialized with compute. Worth 1.31x at 16 GPUs
# (4 nodes) and ~1.02x at 4 on one node, for ~4% more temp memory. Clear it
# (--export=ALL,XLA_FLAGS=) to reproduce a number recorded before that date -- the scaling
# tables in docs/performance.md predate this flag.
LHS_FLAG="--xla_gpu_enable_latency_hiding_scheduler=true"
export XLA_FLAGS="${XLA_FLAGS:-$LHS_FLAG}"
export RMHD_REQUIRE_GPU=1  # abort any case where a rank silently falls back to CPU (job 35894622)

export TARANIS_PRECISION=64   # the whole point of this job

# Set CUDA_MPI=1 only if slurms/probe_cuda_mpi.sh showed the openmpi module is CUDA-aware.
# Never set CUDA_VISIBLE_DEVICES -- Slurm scopes it per task.
export MPI4JAX_USE_CUDA_MPI=${CUDA_MPI:-0}

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/taranis
BENCH=$REPO/bench/bench_phase1.py
export RMHD_PKG=$REPO   # bench prints pkg= so the imported package is verifiable

RUN_JAX=${RUN_JAX:-0}          # jax-backend rows are opt-in (script runs if T9 is reverted)
MPI_MODE=${MPI_MODE:-pmix}  # probe job 35845619: this openmpi is --without-pmi + external PMIx; pmi2 fails, pmix works

NX=256; NZ=128          # fp64 doubles every buffer; 256^2x128 complex128 ~ 0.14 GB/state on 32 GB
STEPS="nb20 nr4"

run() { local n=$1; shift; local bind="--gpu-bind=single:1"; case "$*" in *backend=jax*) bind="";; esac; \
        srun --mpi=$MPI_MODE --ntasks="$n" --cpus-per-task=4 \
        --gres=gpu:V100:"$n" $bind "$PY" -u "$BENCH" "$@" 2>&1 | grep -v "bit precision" || true; }

echo "=== config: precision=$TARANIS_PRECISION cuda_mpi=$MPI4JAX_USE_CUDA_MPI run_jax=$RUN_JAX grid=${NX}^2x$NZ ==="

pass() {
    run 1 v1    3d_forced donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS  # no-comm reference
    run 2 v2    3d_forced donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS
    run 1 v1U   3d        donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS  # unforced pair with v1
    run 2 v2U   3d        donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS  # unforced pair with v2
    run 2 v2unr 3d_forced donate nx$NX nz$NZ nps cfl1 halo_late unroll $STEPS  # T3 GPU candidate
    run 2 v2he  3d_forced donate nx$NX nz$NZ nps cfl1 halo_early $STEPS  # T7 hook on (pair with v2)
}

echo "=== pass 1: mpi4jax backend, fp64 ==="
pass
echo "=== pass 2: mpi4jax backend, fp64 ==="
pass

if [ "$RUN_JAX" = "1" ]; then
    echo "=== jax backend (shard_map/NCCL), fp64, 2 passes ==="
    for _ in 1 2; do
        run 2 j2     3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
        run 2 j2hl   3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax halo_late $STEPS
        run 2 j2unr  3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax unroll    $STEPS
    done
fi

# One profiled case (F2(b) stream syncs at fp64); reading it: docs/SAVIO_GPU_SETUP.md "Reading the profile".
echo "=== profile: 2 GPU forced fp64, trace under \$SLURM_SUBMIT_DIR/prof_p3v100 ==="
export RMHD_PROFILE_DIR=$SLURM_SUBMIT_DIR/prof_p3v100_$SLURM_JOB_ID
run 2 v2prof 3d_forced donate nx$NX nz$NZ nps cfl1 halo_late $STEPS --profile
