#!/bin/bash
#SBATCH --job-name=bench_p2_scale
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=bench_p2_scale_%j.out
#SBATCH --error=bench_p2_scale_%j.err
#SBATCH --mem=0

# Same T6/T7 matrix as bench_phase2.sh at 128 ranks (4 nodes) with nz=512 -> nz_local=4.
# This is where the latency savings must show up if they show up at all: per-rank compute
# halves vs the 32-rank/nz=256 job while the per-step collectives cost the same or more.
# nz_local=4 is legal: comms.halo_exchange sends f[:,:2] / f[:,-2:], so the 4th-order
# z stencil only needs nz_local >= halo width 2 (2*halo is NOT required).

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
# -x every perf-relevant env var: OpenMPI only forwards -x-listed vars to remote ranks —
# without these, ranks on nodes 2-4 run unpinned/multithreaded and corrupt the numbers.
# `|| true`: a crashed case (empty output -> grep exit 1) must not abort the whole matrix.
XVARS="-x RMHD_PKG -x TARANIS_PRECISION -x OMP_PROC_BIND -x OMP_PLACES -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x NUMEXPR_NUM_THREADS -x VECLIB_MAXIMUM_THREADS -x XLA_CPU_ASYNC_THREAD_COUNT"
run() { RMHD_PKG=$REPO mpirun -n "$SLURM_NTASKS" $XVARS "$PY" -u "$BENCH" "$@" 2>&1 | grep -v "bit precision" || true; }

NX=128; NZ=512
# nb20: common multiple of the cfl_every values compared, so every case runs 20 steps/block
STEPS="nb20 nr4"

forced_matrix() {
    run base   3d_forced donate $NX $NZ nps cfl1  halo_late $STEPS  # Phase 2 baseline (T7 off)
    run t7     3d_forced donate $NX $NZ nps cfl1            $STEPS  # T7 only (current default)
    run cfl5   3d_forced donate $NX $NZ nps cfl5            $STEPS  # T6 N=5  + T7
    run cfl20  3d_forced donate $NX $NZ nps cfl20           $STEPS  # T6 N=20 + T7
    run cfl5L  3d_forced donate $NX $NZ nps cfl5  halo_late $STEPS  # T6 N=5  without T7
    run cfl20L 3d_forced donate $NX $NZ nps cfl20 halo_late $STEPS  # T6 N=20 without T7
}

echo "=== pass 1: forced 3D fp64, 128 ranks, T6/T7 matrix ==="
forced_matrix

echo "=== pass 2: forced 3D fp64, 128 ranks, T6/T7 matrix ==="
forced_matrix
