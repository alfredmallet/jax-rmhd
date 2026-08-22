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
# card, so it's only a good trade if you can run at TARANIS_PRECISION=32). To switch:
#   --partition=savio3_gpu, --gres=gpu:A40:1, --qos=a40_gpu3_normal, --cpus-per-task=8

module purge
module load anaconda3 gcc openmpi

source activate jax_gpu

# The anaconda3 module sets PYTHONPATH to the BASE anaconda site-packages, whose own
# nvidia-* packages shadow the env's (root cause of "cuSPARSE library was not found").
# GPU jobs never need PYTHONPATH (code selection uses RMHD_PKG) -- drop it.
unset PYTHONPATH
echo "python=$(which python)"

# Block ~/.local user-site packages from shadowing the env (stray mpi4py bit us once).
export PYTHONNOUSERSITE=1

# jax's CUDA plugin can fail to dlopen the pip-bundled nvidia libs by bare soname inside
# jobs ("cuSPARSE library was not found") even when the env is complete -- put every
# nvidia/*/lib dir on LD_LIBRARY_PATH so plain-name dlopen always resolves.
NVLIBS=$("$HOME/.conda/envs/jax_gpu/bin/python" -c "import nvidia,os;print(':'.join(os.path.join(p,d,'lib') for p in nvidia.__path__ for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p,d,'lib'))))" 2>/dev/null || true)
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "NVLIBS=${NVLIBS:-EMPTY}"   # visible proof in the .out that this block ran

export TARANIS_PRECISION=64

# XLA latency-hiding scheduler: a no-op on the single GPU this script asks for (no
# collectives to hide), set here so scaling this job to several GPUs inherits the production
# configuration -- 1.31x at 16 GPUs, measured 2026-08-21, job 37912751. See
# docs/performance.md, "XLA latency-hiding scheduler".
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_latency_hiding_scheduler=true}"

# Do NOT set/override CUDA_VISIBLE_DEVICES -- Slurm's --gres=gpu already scopes this
# process to its assigned GPU.

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/taranis

# Single GPU -> single rank: no MPI-level parallelism benefit here (this codebase only
# domain-decomposes in z across MPI ranks), but mpi4py/mpi4jax are still hard imports
# (config.py's init_cluster() always touches MPI.COMM_WORLD), so launch via mpirun -n 1
# rather than a bare `python` invocation.
time mpirun -n 1 "$PY" -u "$REPO/tests/forced_turbulence_64cubed.py"
