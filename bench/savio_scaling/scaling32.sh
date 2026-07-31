#!/bin/bash
#SBATCH --job-name=scaling32
#SBATCH --account=fc_kawturb  
#SBATCH --partition=savio3       
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1     
#SBATCH --time=00:30:00           
#SBATCH --output=scaling32_%j.out
#SBATCH --error=scaling32_%j.err
#SBATCH --mem=0

set -euo pipefail

module purge
module load anaconda3 gcc openmpi

export OMP_PROC_BIND=close
export OMP_PLACES=cores

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export XLA_CPU_ASYNC_THREAD_COUNT=1
export OMPI_MCA_pml=ucx

PY=/global/home/users/alfredmallet/.conda/envs/jax_cpu/bin/python 
time mpirun -n $SLURM_NTASKS "$PY" -u savio_scaling.py
