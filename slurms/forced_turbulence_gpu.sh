#!/bin/bash
#SBATCH --job-name=forced_turb_gpu
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# V100 requires 4 CPUs/GPU on savio3_gpu (Savio's documented ratio) -- ntasks(1) x
# cpus-per-task(4) = 4 total.
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#
# FCA (fc_*) accounts must explicitly request this QoS for V100 on savio3_gpu, else the
# job is rejected/pends indefinitely -- not needed for condo accounts (use your condo's
# own QoS instead, see scheduler-config docs).
#SBATCH --qos=v100_gpu3_normal
#
#SBATCH --time=00:30:00
#SBATCH --output=forced_turb_gpu_%j.out
#SBATCH --error=forced_turb_gpu_%j.err

# --- Alternative: A40 (more available than V100 -- 16 vs 2 GPUs for regular FCA
# priority -- but its FP64 rate is ~1/32 of FP32, same as any workstation-class Ampere
# card, so it's only a good trade if you can run at RMHD_PRECISION=32). To switch:
#   --partition=savio3_gpu, --gres=gpu:A40:1, --qos=a40_gpu3_normal, --cpus-per-task=8

module purge
module load anaconda3 gcc openmpi

source activate jax_gpu

# Block ~/.local user-site packages from shadowing the env (stray mpi4py bit us once).
export PYTHONNOUSERSITE=1

export RMHD_PRECISION=64

# Do NOT set/override CUDA_VISIBLE_DEVICES -- Slurm's --gres=gpu already scopes this
# process to its assigned GPU.

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/jax_rmhd

# Single GPU -> single rank: no MPI-level parallelism benefit here (this codebase only
# domain-decomposes in z across MPI ranks), but mpi4py/mpi4jax are still hard imports
# (config.py's init_cluster() always touches MPI.COMM_WORLD), so launch via mpirun -n 1
# rather than a bare `python` invocation.
time mpirun -n 1 "$PY" -u "$REPO/tests/forced_turbulence_64cubed.py"
