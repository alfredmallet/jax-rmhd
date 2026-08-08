#!/bin/bash
#SBATCH --job-name=bench_p1_64
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=bench_p1_64_%j.out
#SBATCH --error=bench_p1_64_%j.err
#SBATCH --mem=0

# fp64 re-validation of the Phase 1 decisions (production runs at RMHD_PRECISION=64):
# doubles message sizes and memory traffic, so donation cost, stacked allreduce, shell
# vs full-grid RNG, and the nps gain are all re-measured. Same layout as bench_p1_ab.

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

export RMHD_PRECISION=64

PY=$HOME/.conda/envs/jax_cpu/bin/python
REPO=$HOME/taranis

OLD_REF=${OLD_REF:-origin/main}
OLDDIR=$SLURM_SUBMIT_DIR/old_pkg_$SLURM_JOB_ID
mkdir -p "$OLDDIR"
git -C "$REPO" ls-files taranis | while read -r f; do
    mkdir -p "$OLDDIR/$(dirname "$f")"
    git -C "$REPO" show "$OLD_REF:$f" > "$OLDDIR/$f"
done

BENCH=$REPO/bench/bench_phase1.py
run_old() { RMHD_PKG=$OLDDIR mpirun -n "$SLURM_NTASKS" -x RMHD_PKG -x RMHD_PRECISION "$PY" -u "$BENCH" "$@" 2>&1 | grep -v "bit precision"; }
run_new() { RMHD_PKG=$REPO   mpirun -n "$SLURM_NTASKS" -x RMHD_PKG -x RMHD_PRECISION "$PY" -u "$BENCH" "$@" 2>&1 | grep -v "bit precision"; }

NX=128; NZ=256

echo "=== unforced 3D, fp64 ==="
run_old old  3d nodonate $NX $NZ
run_old oldD 3d donate   $NX $NZ
run_new new  3d donate   $NX $NZ

for pass in 1 2; do
    echo "=== pass $pass: forced 3D, fp64 ==="
    run_old old  3d_forced nodonate $NX $NZ            # true pre-Phase-1 baseline
    run_new new  3d_forced donate   $NX $NZ            # new defaults (full-grid RNG now)
    run_new shl  3d_forced donate   $NX $NZ shellrng   # shell RNG variant (lost at fp32)
    run_new nps  3d_forced donate   $NX $NZ nps        # recommended production config
done

rm -rf "$OLDDIR"
