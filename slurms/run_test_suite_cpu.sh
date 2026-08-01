#!/bin/bash
#SBATCH --job-name=rmhd_tests_cpu
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=rmhd_tests_cpu_%j.out
#SBATCH --error=rmhd_tests_cpu_%j.err
#SBATCH --mem=0

# The full CPU test suite in one submission:
#     sbatch slurms/run_test_suite_cpu.sh                 # everything
#     sbatch slurms/run_test_suite_cpu.sh --only halo     # substring filter
#     sbatch slurms/run_test_suite_cpu.sh --list          # print the job table
# The job table lives in tests/savio_manifest.py; tests/run_savio_suite.py runs each
# phase under mpirun, tees per-phase logs into $RMHD_TEST_OUTDIR, and fails the job
# if any phase exits nonzero OR fails the ALL PASS banner check (mpirun can mask a
# rank's exit code). savio3 charges per NODE, so we request all 32 cores even though
# the widest phase uses 8.

# ---- preamble copied verbatim from the proven savio3 CPU jobs ----
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
# ---- end preamble ----

# Adjust these two if your checkout of jax_rmhd or conda env don't match this layout.
PY=$HOME/.conda/envs/jax_cpu/bin/python
REPO=$HOME/jax_rmhd

# Per-job clean log/scratch root (every job gets a fresh subdir inside -- stale
# snapshot dirs can never trip the layout guard).
export RMHD_TEST_OUTDIR=${RMHD_TEST_OUTDIR:-$PWD/data/test_suite_cpu_$SLURM_JOB_ID}

# Launcher template for the "mpi" phases ({n} = the phase's rank count).
export RMHD_LAUNCH="mpirun -n {n}"

time "$PY" -u "$REPO/tests/run_savio_suite.py" --tier cpu "$@"
