#!/bin/bash
#SBATCH --job-name=bench_p2
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=bench_p2_%j.out
#SBATCH --error=bench_p2_%j.err
#SBATCH --mem=0

# Phase 2 gate at 32 ranks, fp64 (the only precision decisions are made on): measures T6
# (params.cfl_every) and T7 (early halo issue) as a 2x3 matrix, {T7 on, halo_late} x
# cfl_every {1,5,20}. No old package is extracted: the Phase 2 baseline is this repo with
# cfl_every=1 + halo_late (T5 is bitwise-identical to Phase 1, T7 is off under halo_late).
# Expect ~353 ms/step for the baseline forced case (Phase 1 fp64 number).

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

export TARANIS_PRECISION=64

PY=$HOME/.conda/envs/jax_cpu/bin/python
REPO=$HOME/taranis

BENCH=$REPO/bench/bench_phase1.py
# RMHD_PKG (not PYTHONPATH): the editable install's import finder beats PYTHONPATH; the
# bench purges it and prints pkg= so the imported package is verifiable in the output.
# -x every perf-relevant env var: OpenMPI only forwards -x-listed vars to remote ranks,
# and thread oversubscription on non-launch nodes would silently corrupt the numbers.
# `|| true`: a crashed case (empty output -> grep exit 1) must not abort the whole matrix.
XVARS="-x RMHD_PKG -x TARANIS_PRECISION -x OMP_PROC_BIND -x OMP_PLACES -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x NUMEXPR_NUM_THREADS -x VECLIB_MAXIMUM_THREADS -x XLA_CPU_ASYNC_THREAD_COUNT"
run() { RMHD_PKG=$REPO mpirun -n "$SLURM_NTASKS" $XVARS "$PY" -u "$BENCH" "$@" 2>&1 | grep -v "bit precision" || true; }

NX=128; NZ=256
# nb20: nblock must be a multiple of every cfl_every compared (1,5,20) so all cases run the
# same 20 steps per block; nr4 -> 20 warm + 80 timed steps per case.
STEPS="nb20 nr4"

forced_matrix() {
    run base   3d_forced donate $NX $NZ nps cfl1  halo_late $STEPS  # Phase 2 baseline (T7 off)
    run t7     3d_forced donate $NX $NZ nps cfl1            $STEPS  # T7 only (current default)
    run cfl5   3d_forced donate $NX $NZ nps cfl5            $STEPS  # T6 N=5  + T7
    run cfl20  3d_forced donate $NX $NZ nps cfl20           $STEPS  # T6 N=20 + T7
    run cfl5L  3d_forced donate $NX $NZ nps cfl5  halo_late $STEPS  # T6 N=5  without T7
    run cfl20L 3d_forced donate $NX $NZ nps cfl20 halo_late $STEPS  # T6 N=20 without T7
}

echo "=== pass 1: forced 3D fp64, T6/T7 matrix ==="
forced_matrix

echo "=== unforced 3D fp64 (per-step comm: 5 halo pairs + the CFL allreduce) ==="
# one variable at a time: cfl20U isolates T6, t7U isolates T7 (review MINOR-1)
run baseU  3d donate $NX $NZ nps cfl1  halo_late $STEPS
run cfl20U 3d donate $NX $NZ nps cfl20 halo_late $STEPS
run t7U    3d donate $NX $NZ nps cfl1            $STEPS

echo "=== pass 2: forced 3D fp64, T6/T7 matrix ==="
forced_matrix
