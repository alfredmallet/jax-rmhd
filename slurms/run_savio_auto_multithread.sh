#!/bin/bash
#SBATCH --job-name=auto
#SBATCH --account=fc_kawturb  
#SBATCH --partition=savio3        
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32        
#SBATCH --time=00:30:00           
#SBATCH --output=auto%j.out
#SBATCH --error=auto%j.err

module purge
module load anaconda3

source activate jax_cpu

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

time srun python -u /global/home/users/alfredmallet/taranis/examples/test.py
