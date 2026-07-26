#!/bin/bash
#SBATCH --job-name=bench_p3_a5k
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#
# savio4_gpu is PER-CORE scheduled at 4 CPUs per A5000: 4 tasks x 4 = 16 CPUs, which is
# also the regular-FCA per-user cap (= 4 GPUs max). Never request a whole node here.
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:4
#SBATCH --time=01:00:00
#SBATCH --output=bench_p3_a5k_%j.out
#SBATCH --error=bench_p3_a5k_%j.err

# T8 single-node GPU baseline: 1/2/4 A5000 strong scaling at fp32 (A5000 fp64 is ~1/32 of
# fp32 -- fp64 numbers come from bench_phase3_v100_fp64.sh instead). Answers: how much of a
# step is comm at 2 and 4 GPUs, does lsrk_scan=False (T3's GPU candidate) win here, and does
# the T7 early-halo hook win on GPU. Set RUN_JAX=1 once A2's comm_backend="jax" exists.

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

export RMHD_PRECISION=32

# Host-staged transport is the default assumption; set CUDA_MPI=1 only if slurms/probe_cuda_mpi.sh
# showed the openmpi module is CUDA-aware. Do NOT set CUDA_VISIBLE_DEVICES -- Slurm scopes it.
export MPI4JAX_USE_CUDA_MPI=${CUDA_MPI:-0}

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/jax_rmhd
BENCH=$REPO/bench/bench_phase1.py
export RMHD_PKG=$REPO   # bench prints pkg= so the imported package is verifiable

RUN_JAX=${RUN_JAX:-0}   # jax-backend rows are opt-in: the script still runs if T9 is reverted
MPI_MODE=${MPI_MODE:-pmix}  # probe job 35845619: this openmpi is --without-pmi + external PMIx; pmi2 fails, pmix works   # see slurms/probe_cuda_mpi.sh's `srun --mpi=list` output

NX=512; NZ=128          # 512^2 x 128 fp32 ~ 0.3 GB/state: fills an A5000 usefully, fits 24 GB
STEPS="nb20 nr4"        # 20 warm + 80 timed steps per case

# srun (not mpirun) so SLURM gives each rank exactly one GPU; srun forwards the environment
# by default, so no mpirun-style -x list is needed. If the site's Slurm rejects the step-level
# --gres, drop it and use --gpus-per-task=1 -- either way SLURM, not us, sets CUDA_VISIBLE_DEVICES.
run() { local n=$1; shift; local bind="--gpu-bind=single:1"; case "$*" in *backend=jax*) bind="";; esac; \
        srun --mpi=$MPI_MODE --ntasks="$n" --cpus-per-task=4 \
        --gres=gpu:A5000:"$n" $bind "$PY" -u "$BENCH" "$@" 2>&1 | grep -v "bit precision" || true; }

echo "=== config: precision=$RMHD_PRECISION cuda_mpi=$MPI4JAX_USE_CUDA_MPI run_jax=$RUN_JAX grid=${NX}^2x$NZ ==="

pass() {
    run 1 g1    3d_forced donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS
    run 2 g2    3d_forced donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS
    run 4 g4    3d_forced donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS
    run 1 g1U   3d        donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS  # unforced pair: the
    run 4 g4U   3d        donate nx$NX nz$NZ nps cfl1 halo_late  $STEPS  # forcing allreduce's cost
    run 4 g4unr 3d_forced donate nx$NX nz$NZ nps cfl1 halo_late unroll $STEPS  # T3 GPU candidate
    run 4 g4he  3d_forced donate nx$NX nz$NZ nps cfl1 halo_early $STEPS  # T7 hook on (pair with g4)
}

echo "=== pass 1: mpi4jax backend ==="
pass
echo "=== pass 2: mpi4jax backend ==="
pass

if [ "$RUN_JAX" = "1" ]; then
    echo "=== jax backend (shard_map/NCCL), 2 passes of the scaling points + option pairs ==="
    for _ in 1 2; do
        run 1 j1    3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
        run 2 j2    3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
        run 4 j4    3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax $STEPS
        run 4 j4hl  3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax halo_late  $STEPS
        run 4 j4unr 3d_forced donate nx$NX nz$NZ nps cfl1 backend=jax unroll     $STEPS
    done
fi

# One profiled case (F2(b): per-call stream syncs). Reading the trace: SAVIO_GPU_SETUP.md
# "Reading the profile". Rank 0 only; RMHD_PROFILE_ALL=1 would trace every rank.
echo "=== profile: 4 GPU forced, trace under \$SLURM_SUBMIT_DIR/prof_p3a5k ==="
export RMHD_PROFILE_DIR=$SLURM_SUBMIT_DIR/prof_p3a5k_$SLURM_JOB_ID
run 4 g4prof 3d_forced donate nx$NX nz$NZ nps cfl1 halo_late $STEPS --profile
