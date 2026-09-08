#!/bin/bash
# Production multi-GPU forced 3D RMHD on Savio: comm_backend="jax" (shard_map/NCCL) over
# 4 A5000s on one savio4_gpu node.  Submit as-is:
#
#     sbatch slurms/forced_turbulence_multigpu.sh
#
# and resubmit the same script to continue the run -- the driver restarts from the newest
# snapshot in TARANIS_SNAPDIR.  Setup (the jax_gpu env) and the reasoning behind every
# export below: docs/SAVIO_GPU_SETUP.md.  Scaling numbers: docs/performance.md "GPU".
#
# To use more or different GPUs, change these three lines together -- --ntasks (one task
# per GPU), --gres (PER NODE, explicit type mandatory for fc_* accounts) and --qos -- and
# keep --cpus-per-task at the partition's documented CPUs-per-GPU ratio:
#
#   2 x V100  savio3_gpu  --qos=v100_gpu3_normal   --gres=gpu:V100:2   --cpus-per-task=4
#             the only full-rate-fp64 cards here; run these at TARANIS_PRECISION=64
#   4 x A5000 savio4_gpu  --qos=a5k_gpu4_normal    --gres=gpu:A5000:4  --cpus-per-task=4
#             regular FCA priority caps a user at 16 CPUs = 4 A5000s
#   4 x 2080Ti savio3_gpu --qos=gtx2080_gpu3_normal --gres=gpu:GTX2080TI:4 --cpus-per-task=2
#             (that gres type string really is spelled GTX2080TI)
#  16 x A40   savio3_gpu  --qos=a40_gpu3_normal    --gres=gpu:A40:4    --cpus-per-task=8
#             plus --nodes=4 --ntasks-per-node=4 (--gres is per node); all 16 A40s in the
#             cluster, so expect to wait, and keep the walltime at or under an hour
#
#SBATCH --job-name=turb_multigpu
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:4
#SBATCH --time=04:00:00
#SBATCH --output=turb_multigpu_%j.out
#SBATCH --error=turb_multigpu_%j.err

set -uo pipefail

module purge
module load anaconda3 gcc openmpi
source activate jax_gpu

# The anaconda3 module points PYTHONPATH at the BASE anaconda's site-packages, whose own
# nvidia-* packages then shadow the env's -- the root cause of "The cuSPARSE library was
# not found".  GPU jobs never need PYTHONPATH.
unset PYTHONPATH
# Block ~/.local user-site packages from shadowing the env (a stray --user mpi4py bit us).
export PYTHONNOUSERSITE=1
echo "python=$(which python)"

# jax's CUDA plugin sometimes fails to dlopen the pip-bundled nvidia libs by bare soname
# inside a job even when the env is complete -- put every nvidia/*/lib dir on the path.
# The echo is proof in the .out that this ran: the paths must be under ~/.conda/envs/jax_gpu,
# never under anaconda3/<version>/lib (that means PYTHONPATH is leaking again).
NVLIBS=$("$HOME/.conda/envs/jax_gpu/bin/python" -c "import nvidia,os;print(':'.join(os.path.join(p,d,'lib') for p in nvidia.__path__ for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p,d,'lib'))))" 2>/dev/null || true)
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "NVLIBS=${NVLIBS:-EMPTY}"

# PCIe peer-to-peer between GPUs is BROKEN on savio4_gpu (rings connect, the first
# collective hangs forever; bench/nccl_repro.py is the repro).  SHM transport works.
# Do not remove this until the cluster config changes and the repro says so.
export NCCL_P2P_DISABLE=1

# Overlap collectives with compute.  Without it XLA emits the halo ppermutes and the two
# allreduces as async pairs but schedules ZERO instructions between them: 1.31x at 16 GPUs,
# ~1.02x at 4, for ~4% more temp memory.  Never worth omitting on a multi-GPU run.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_latency_hiding_scheduler=true}"

# Persistent JIT cache: the first block compiles for tens of seconds, and a resubmitted
# continuation run skips it.  Keyed by directory; safe to share across jobs.
export RMHD_COMPILATION_CACHE=$HOME/.taranis_jit_cache

# Field precision, read at IMPORT time -- it is not a runtime flag.  A5000/A40/L40/2080Ti
# run fp64 at ~1/32 of fp32, so 32 is the right choice on them; use 64 on V100.
export TARANIS_PRECISION=32

# The run itself.  Every one of these is optional (the driver has defaults) -- they are
# spelled out here because the job script is where a run's size belongs.
export TARANIS_BACKEND=jax
export TARANIS_NX=512          # nx = ny
export TARANIS_NZ=128          # MUST be divisible by the total GPU count
export TARANIS_TEND=10.0
export TARANIS_TSNAP=0.5
export TARANIS_SCHEME=lsrk54
export TARANIS_SNAPDIR=$SLURM_SUBMIT_DIR/data/turb_multigpu

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/taranis
DRIVER=$REPO/examples/multigpu_forced_turbulence.py

# srun, NEVER mpirun: only Slurm scopes GPUs per task.  --mpi=pmix is required here (this
# openmpi is --without-pmi + external PMIx; pmi2 aborts before MPI_Init).
#
# And NO --gpu-bind: that is the one launch flag that differs between the backends.  The
# jax/NCCL backend needs every job GPU visible to every process -- with only its own GPU
# visible, NCCL's first collective dies with 'invalid device ordinal' -- and
# comms._local_device_ids pins each process to its node-local rank's ordinal instead.  Add
# --gpu-bind=single:1 only if you switch TARANIS_BACKEND to mpi4jax, which never calls
# jax.distributed and would otherwise pile every rank onto GPU 0.
#
# Do NOT set CUDA_VISIBLE_DEVICES by hand either way; Slurm owns it.
time srun --mpi=pmix --ntasks="$SLURM_NTASKS" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     "$PY" -u "$DRIVER"
