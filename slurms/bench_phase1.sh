#!/bin/bash
#SBATCH --job-name=bench_phase1
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=bench_phase1_%j.out
#SBATCH --error=bench_phase1_%j.err
#SBATCH --mem=0

# Old-vs-new Phase 1 benchmark under real MPI. "Old" = origin/main (the pre-Phase-1
# baseline), extracted read-only from the clone's git database -- no second checkout
# needed; clone with the performance branch checked out and origin/main is available.
# Override with e.g.  sbatch --export=ALL,OLD_REF=<ref> slurms/bench_phase1.sh

set -euo pipefail

module purge
module load anaconda3 gcc openmpi
source activate jax_cpu

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_CPU_ASYNC_THREAD_COUNT=1
export OMPI_MCA_pml=ucx

PY=$HOME/.conda/envs/jax_cpu/bin/python
REPO=$HOME/taranis

# fp32 (the code default) per the plan; uncomment for an fp64 spot check.
# export TARANIS_PRECISION=64

# extract the old package from git into a scratch dir
OLD_REF=${OLD_REF:-origin/main}
OLDDIR=$SLURM_SUBMIT_DIR/old_pkg_$SLURM_JOB_ID
mkdir -p "$OLDDIR"
git -C "$REPO" ls-files taranis | while read -r f; do
    mkdir -p "$OLDDIR/$(dirname "$f")"
    git -C "$REPO" show "$OLD_REF:$f" > "$OLDDIR/$f"
done

BENCH=$REPO/bench/bench_phase1.py
# RMHD_PKG (not PYTHONPATH): a pip-install-e'd taranis registers an import finder that
# beats PYTHONPATH; the bench purges it, imports from RMHD_PKG, and asserts + prints
# which package file it actually imported (check pkg= in the output!).
run_old() { RMHD_PKG=$OLDDIR mpirun -n "$SLURM_NTASKS" -x RMHD_PKG "$PY" -u "$BENCH" "$@"; }
run_new() { RMHD_PKG=$REPO   mpirun -n "$SLURM_NTASKS" -x RMHD_PKG "$PY" -u "$BENCH" "$@"; }

NX=128; NZ=256   # nz_local = NZ/32 = 8 per rank

# two passes to expose run-to-run noise
for pass in 1 2; do
    echo "=== pass $pass: unforced 3D ==="
    run_old old 3d nodonate $NX $NZ
    run_new new 3d donate   $NX $NZ
    echo "=== pass $pass: forced 3D (lsrk54 + elsasser + adaptive dt: comm worst case) ==="
    run_old old 3d_forced nodonate $NX $NZ
    run_new new 3d_forced donate   $NX $NZ
    run_new nps 3d_forced donate   $NX $NZ nps
done
echo "=== extra: unrolled LSRK at scale (expected slower on CPU) ==="
run_new unr 3d_forced donate $NX $NZ unroll

rm -rf "$OLDDIR"
