#!/bin/bash
#SBATCH --job-name=bench_p1_ab
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=bench_p1_ab_%j.out
#SBATCH --error=bench_p1_ab_%j.err
#SBATCH --mem=0

# Follow-up to bench_phase1.sh: first run found the new forced default ~7% SLOWER than
# origin/main at 32 ranks while nps was faster. This job attributes the regression by
# toggling each T4 change independently (sep = revert stacked allreduce, fullrng =
# revert shell-restricted RNG), plus a donation control on the old code.

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
REPO=$HOME/jax_rmhd

OLD_REF=${OLD_REF:-origin/main}
OLDDIR=$SLURM_SUBMIT_DIR/old_pkg_$SLURM_JOB_ID
mkdir -p "$OLDDIR"
git -C "$REPO" ls-files jax_rmhd | while read -r f; do
    mkdir -p "$OLDDIR/$(dirname "$f")"
    git -C "$REPO" show "$OLD_REF:$f" > "$OLDDIR/$f"
done

BENCH=$REPO/bench/bench_phase1.py
# RMHD_PKG (not PYTHONPATH): a pip-install-e'd jax_rmhd registers an import finder that
# beats PYTHONPATH; the bench purges it, imports from RMHD_PKG, and asserts + prints
# which package file it actually imported (check pkg= in the output!).
run_old() { RMHD_PKG=$OLDDIR mpirun -n "$SLURM_NTASKS" -x RMHD_PKG "$PY" -u "$BENCH" "$@" 2>&1 | grep -v "32bit precision"; }
run_new() { RMHD_PKG=$REPO   mpirun -n "$SLURM_NTASKS" -x RMHD_PKG "$PY" -u "$BENCH" "$@" 2>&1 | grep -v "32bit precision"; }

NX=128; NZ=256

for pass in 1 2; do
    echo "=== pass $pass: forced 3D attribution matrix ==="
    run_old old  3d_forced nodonate $NX $NZ            # baseline (origin/main)
    run_old oldD 3d_forced donate   $NX $NZ            # donation control on old code
    run_new new  3d_forced donate   $NX $NZ            # new defaults (regressed run 1)
    run_new sep  3d_forced donate   $NX $NZ sep        # new minus T4b (separate allreduces)
    run_new rng  3d_forced donate   $NX $NZ fullrng    # new minus T4a (full-grid RNG)
    run_new both 3d_forced donate   $NX $NZ sep fullrng  # new minus both
    run_new nps  3d_forced donate   $NX $NZ nps        # per-step norm (fastest in run 1)
done

rm -rf "$OLDDIR"
