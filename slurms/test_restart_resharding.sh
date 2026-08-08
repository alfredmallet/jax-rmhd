#!/bin/bash
#SBATCH --job-name=restart_reshard
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=restart_reshard_%j.out
#SBATCH --error=restart_reshard_%j.err
#SBATCH --mem=0

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

# Adjust these two if your checkout of taranis or conda env don't match this layout.
PY=$HOME/.conda/envs/jax_cpu/bin/python
REPO=$HOME/taranis

# Both phases run in one job: save on 2 ranks, then restart on 4 ranks into the
# same snap_path (relative to cwd). Wipe any previous run so phase detection
# starts fresh.
rm -rf data/test_restart_resharding

echo "=== phase A: fresh run on 2 ranks ==="
time mpirun -n 2 "$PY" -u "$REPO/tests/test_restart_resharding.py"

echo "=== phase B: restart on 4 ranks ==="
time mpirun -n 4 "$PY" -u "$REPO/tests/test_restart_resharding.py"
