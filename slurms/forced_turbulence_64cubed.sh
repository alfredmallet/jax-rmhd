#!/bin/bash
#SBATCH --job-name=forced_turb_64
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=forced_turb_64_%j.out
#SBATCH --error=forced_turb_64_%j.err
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

time mpirun -n $SLURM_NTASKS "$PY" -u "$REPO/tests/forced_turbulence_64cubed.py"
