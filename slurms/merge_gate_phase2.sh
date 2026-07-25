#!/bin/bash
#SBATCH --job-name=merge_gate_p2
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=merge_gate_p2_%j.out
#SBATCH --error=merge_gate_p2_%j.err
#SBATCH --mem=0

# Phase 2 merge gate: the full Savio battery on the final tree, one submission.
# All runs are 4 MPI ranks max on ONE node ("-n 4" = ranks, not nodes).

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
export MPLBACKEND=Agg

PY=$HOME/.conda/envs/jax_cpu/bin/python
REPO=$HOME/jax_rmhd
cd "$REPO"

echo "=== 1/3: test_advection, 4 ranks (z-convergence scan; longest part) ==="
time mpirun -n 4 "$PY" -u tests/test_advection.py

echo "=== 2/3: test_restart_resharding phase A: fresh run on 2 ranks ==="
rm -rf data/test_restart_resharding
time mpirun -n 2 "$PY" -u tests/test_restart_resharding.py

echo "=== 3/3: test_restart_resharding phase B: restart on 4 ranks ==="
time mpirun -n 4 "$PY" -u tests/test_restart_resharding.py
