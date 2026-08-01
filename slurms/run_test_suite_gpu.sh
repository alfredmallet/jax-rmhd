#!/bin/bash
#SBATCH --job-name=rmhd_tests_gpu
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#
# savio4_gpu is per-CORE scheduled with a 4:1 CPU:GPU ratio, and FCA regular priority caps
# a user at 16 CPUs = 4 A5000s: 4 tasks x 1 GPU x 4 cpus-per-task is exactly that budget.
# The explicit GPU type in --gres is REQUIRED for FCA jobs (missing it -> QOSMinGRES pend).
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:4
#
#SBATCH --time=00:40:00
#SBATCH --output=rmhd_tests_gpu_%j.out
#SBATCH --error=rmhd_tests_gpu_%j.err

# The GPU test suite in one submission: the 5-phase
# mpi4jax-vs-jax/NCCL backend battery from tests/savio_manifest.py, driven by
# tests/run_savio_suite.py (per-phase logs in $RMHD_TEST_OUTDIR, pass rule =
# exit 0 AND the ALL PASS banner). Usage: sbatch slurms/run_test_suite_gpu.sh
# [--only ... | --list]. Preamble + launch modes copied verbatim from the proven
# slurms/test_backend_jax_gpu.sh -- see that file's comments for the full forensics.

# ---- preamble copied verbatim from test_backend_jax_gpu.sh ----
module purge
module load anaconda3 gcc openmpi cuda

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

MPI_MODE=${MPI_MODE:-pmix}  # probe job 35845619: this openmpi is --without-pmi + external PMIx; pmi2 fails, pmix works

# One-task compute-node probe: prints what the actual job python sees BEFORE any phase runs.
srun --mpi=$MPI_MODE --ntasks=1 --cpus-per-task=4 "$HOME/.conda/envs/jax_gpu/bin/python" -c "import os,sys,nvidia,jax; print('probe PYTHONPATH=',os.environ.get('PYTHONPATH')); print('probe nvidia=',nvidia.__path__); print('probe devices=',jax.devices())" || true

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/jax_rmhd

export RMHD_PKG=$REPO          # printed in every result line, per the Phase 3 ground rules
export RMHD_NX=64
export RMHD_NZ=32
export RMHD_TEND=1.0

# Hang forensics: NCCL_DEBUG=INFO logs communicator bring-up per rank to stderr; the
# `timeout 900` inside the launch templates turns a hang into a failed phase (rc 124)
# so later phases still run and the summary table still prints.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
# NCCL: PCIe P2P between GPUs is BROKEN on savio4_gpu nodes (repro 2026-07-26; SHM
# transport works) -- reported to Savio support; revisit if they fix it.
export NCCL_P2P_DISABLE=1
# Per-rank python stack dumps every 120 s while hung (see test_backend_jax_mpi.py).
export RMHD_DEBUG_HANG=1
# ---- end preamble ----

# Launch templates ({n} = the phase's rank count). Do NOT set CUDA_VISIBLE_DEVICES.
# TWO modes, one per backend (job 35861380):
# - mpi4jax NEEDS --gpu-bind=single:1 (one visible GPU per task) or every rank piles
#   onto GPU 0 -- it never calls jax.distributed, so nothing else assigns devices.
# - the jax/NCCL backend must NOT use it: same-node P2P/SHM transport needs peer GPUs
#   visible; comms._local_device_ids pins each process to its node-local ordinal.
export RMHD_LAUNCH_MPI4JAX="timeout 900 srun --mpi=$MPI_MODE --ntasks={n} --cpus-per-task=$SLURM_CPUS_PER_TASK --gpu-bind=single:1"
export RMHD_LAUNCH_JAX="timeout 900 srun --mpi=$MPI_MODE --ntasks={n} --cpus-per-task=$SLURM_CPUS_PER_TASK"

export RMHD_TEST_OUTDIR=${RMHD_TEST_OUTDIR:-$PWD/data/test_suite_gpu_$SLURM_JOB_ID}

time "$PY" -u "$REPO/tests/run_savio_suite.py" --tier gpu "$@"
