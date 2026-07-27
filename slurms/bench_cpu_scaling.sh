#!/bin/bash
#SBATCH --job-name=bench_cpu_scaling
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=01:45:00
#SBATCH --output=bench_cpu_scaling_%j.out
#SBATCH --error=bench_cpu_scaling_%j.err
#SBATCH --mem=0

# CPU strong/weak scaling curves for the mpi4jax backend (Phase 3 context: the GPU bench
# showed mpi4jax scaling 2.3x/4GPU vs jax/NCCL ~4x/4GPU; this job provides the CPU-side
# curves the GPU numbers get compared against, plus two fp32 rows on the EXACT A5000 bench
# grid for a direct hardware-for-hardware throughput comparison).
# savio3 is per-NODE charged: 4 nodes are held for the whole job (~128 SU/hr), so smaller
# -n cases pack ranks onto the first node(s) (mpirun default) rather than resizing the job.

set -euo pipefail

module purge
module load anaconda3 gcc openmpi
source activate jax_cpu
export PYTHONNOUSERSITE=1   # ~/.local shadowing bit us on the GPU env; block it here too

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_CPU_ASYNC_THREAD_COUNT=1
export OMPI_MCA_pml=ucx

export RMHD_PRECISION=64
# Persistent compile cache: pass 2 (and same-shape cases) skip XLA recompilation.
export RMHD_COMPILATION_CACHE=$SLURM_SUBMIT_DIR/.jaxcache_cpu_scaling_$SLURM_JOB_ID

PY=$HOME/.conda/envs/jax_cpu/bin/python
REPO=$HOME/jax_rmhd
BENCH=$REPO/bench/bench_phase1.py

XVARS="-x RMHD_PKG -x RMHD_PRECISION -x RMHD_COMPILATION_CACHE -x PYTHONNOUSERSITE -x OMP_PROC_BIND -x OMP_PLACES -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x NUMEXPR_NUM_THREADS -x VECLIB_MAXIMUM_THREADS -x XLA_CPU_ASYNC_THREAD_COUNT"
# run <nranks> <tag> <case args...>
run() { local n=$1; shift; RMHD_PKG=$REPO mpirun -n "$n" $XVARS "$PY" -u "$BENCH" "$@" 2>&1 | grep -v "bit precision" || true; }

STEPS="nb20 nr4"   # 80 steps/case, same as the GPU benches

echo "=== config: fp64 strong 256^2x256 / weak nz_local=4 / fp32 cross 512^2x128 ==="

pass() {
    # Strong scaling, production fp64: fixed 256^2 x 256, ranks 16 -> 128 (nz_local 16 -> 2).
    run 16  s16  3d_forced donate nx256 nz256 nps cfl1 halo_late $STEPS
    run 32  s32  3d_forced donate nx256 nz256 nps cfl1 halo_late $STEPS
    run 64  s64  3d_forced donate nx256 nz256 nps cfl1 halo_late $STEPS
    run 128 s128 3d_forced donate nx256 nz256 nps cfl1 halo_late $STEPS
    # Weak scaling, fp64: fixed per-rank slab 256^2 x nz_local=4, nz grows with ranks.
    run 16  w16  3d_forced donate nx256 nz64  nps cfl1 halo_late $STEPS
    run 32  w32  3d_forced donate nx256 nz128 nps cfl1 halo_late $STEPS
    run 64  w64  3d_forced donate nx256 nz256 nps cfl1 halo_late $STEPS
    run 128 w128 3d_forced donate nx256 nz512 nps cfl1 halo_late $STEPS
}

echo "=== pass 1 (fp64) ==="
pass
echo "=== pass 2 (fp64) ==="
pass

# Cross-comparison rows: EXACT grid+precision of the A5000 GPU bench (512^2x128, fp32,
# forced+nps) -> direct "N savio3 nodes ~ M A5000s" statement. nz=128 caps the rank count
# at 64 (nz_local=2 = halo width; 128 ranks would give nz_local=1, illegal), so the
# comparison points are 1 node (32) and 2 nodes (64); the s/w curves above cover 128.
export RMHD_PRECISION=32
echo "=== cross-comparison (fp32, 512^2x128 = A5000 bench grid) ==="
for p in 1 2; do
    run 32  x32  3d_forced donate nx512 nz128 nps cfl1 halo_late $STEPS
    run 64  x64  3d_forced donate nx512 nz128 nps cfl1 halo_late $STEPS
done

echo "=== done ==="
