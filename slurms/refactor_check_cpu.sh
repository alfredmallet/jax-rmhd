#!/bin/bash
#SBATCH --job-name=refactor_check_cpu
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=refactor_check_cpu_%j.out
#SBATCH --error=refactor_check_cpu_%j.err
#SBATCH --mem=0

# REFACTOR_PLAN close-out (2026-08-22): the multi-rank CPU checks the laptop suite cannot
# run, on the merged tree. What each selected job exercises that the refactor touched:
#   restart_resharding        save on 2 ranks, restart + reshard on 4 -- R's Runtime
#                             (cart comm, neighbors, rank/size properties through
#                             snapshot_io's per-rank layout and the rank-0 index bcast)
#   forced_turbulence_64cubed 8 ranks, forcing + checkpoint + reload end to end -- the
#                             one forced-turbulence restart (Phase 0c: a stored nonzero
#                             forcing_scale now survives driver entry)
#   halo_width, z_stencils    the z halo exchange over R's cart communicator at 2/4 ranks,
#                             G's FDLinearTerm selected by fd_linear_active
#   cfl_every                 C's _cfl_block on a real 4-rank CFL allreduce
#   dissipation, scheme_equivalence   L's kgrid.lin through the IF and IMEX steppers at 4 ranks
#   params, forcing_norm_per_step, snapshot_roundtrip, equation_interface,
#   refactor_reference, backend_jax_serial   single-process under the cluster's real mpi4py
#                             (the last two print [SKIP] on Savio: the reference is
#                             host-keyed and the multidev file needs fake devices)
# Everything runs through tests/run_savio_suite.py (per-phase logs under
# $RMHD_TEST_OUTDIR, pass rule = exit 0 AND the ALL PASS banner). Extra arguments are
# forwarded: `--precision 64` narrows to one session, `--list` prints the selection, a
# further `--only X` ADDS jobs (substring match). The whole CPU suite is
# slurms/run_test_suite_cpu.sh; this is its refactor-relevant subset (~15-20 min).
# NB `mpirun -n 4 python tests/test_snapshot_roundtrip.py` is NOT a check: that file
# soft-skips every test at size > 1 (rank-local tmp dirs); its multi-rank content lives in
# restart_resharding and forced_turbulence_64cubed above.

# ---- preamble copied verbatim from slurms/run_test_suite_cpu.sh ----
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

# Adjust these two if your checkout of taranis or conda env don't match this layout.
PY=$HOME/.conda/envs/jax_cpu/bin/python
REPO=$HOME/taranis

export RMHD_TEST_OUTDIR=${RMHD_TEST_OUTDIR:-$PWD/data/refactor_check_cpu_$SLURM_JOB_ID}
export RMHD_LAUNCH="mpirun -n {n}"

cd "$REPO" && echo "taranis HEAD: $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"
cd - >/dev/null

time "$PY" -u "$REPO/tests/run_savio_suite.py" --tier cpu \
    --only restart_resharding --only forced_turbulence_64cubed \
    --only halo_width --only z_stencils --only cfl_every \
    --only dissipation --only scheme_equivalence \
    --only params --only forcing_norm_per_step --only snapshot_roundtrip \
    --only equation_interface --only refactor_reference --only backend_jax_serial \
    "$@"
