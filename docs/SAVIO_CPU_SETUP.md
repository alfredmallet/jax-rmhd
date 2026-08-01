# Setting up and running jax_rmhd on Savio (CPU)

Student-oriented guide: build the `jax_cpu` conda env once, then run 2D interactively and
3D under MPI via Slurm. GPU setup is separate — see `SAVIO_GPU_SETUP.md`. Official cluster
docs: https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/

Prerequisites: a Savio account under the group's allocation (`fc_kawturb`), one-time-password
login working (`ssh <user>@hpc.brc.berkeley.edu`), and a clone of this repo at `~/jax_rmhd`.

## 1. Build the `jax_cpu` env (login node, once)

```bash
module purge
module load anaconda3 gcc openmpi

conda create -n jax_cpu python=3.11 -y
source activate jax_cpu

# Block ~/.local user-site packages: they take precedence over the env's site-packages, so a
# stray `pip install --user` from the past silently shadows everything you install here.
# Symptom: `python -c "import mpi4py; print(mpi4py.__file__)"` prints ~/.local/... — fix by
# `python -m pip uninstall <pkg>` until the env's own copy is the one that imports.
export PYTHONNOUSERSITE=1

pip install -U jax   # CPU jax; jaxlib comes with it

# mpi4py/mpi4jax (an optional `[mpi]` extra in pyproject.toml) are required for any
# multi-rank run (Parameters construction touches MPI.COMM_WORLD once mpi4py is present).
# A single-rank run works serially without them (comm_backend="serial", auto-selected when
# they're absent) but install them anyway on Savio: the launcher-mismatch guard (every rank
# reporting rank 0, below) needs a real mpi4py to detect. Build BOTH from source against the
# loaded openmpi module — a prebuilt wheel may link a different MPI than the one `mpirun` on
# the compute node launches, which is exactly the every-rank-reports-rank-0 symptom.
# --no-cache-dir: otherwise a rebuild after a toolchain change reinstalls a stale cached wheel.
MPICC=$(which mpicc) python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py
python -m pip install --no-cache-dir --no-binary=mpi4jax mpi4jax

pip install orbax-checkpoint tensorstore numpy matplotlib

# NOT `pip install -e ".[mpi]"` here: the extra would let pip resolve its own mpi4py/mpi4jax
# wheels, bypassing the MPICC-pinned from-source builds above and their build ordering
# (mpi4py before mpi4jax). Both are already satisfied, so plain `-e .` picks them up as-is.
cd ~/jax_rmhd && pip install -e .
```

Verify (still on the login node — fine for imports, never for real runs):

```bash
# every path must be under ~/.conda/envs/jax_cpu, NEVER ~/.local
python -c "import mpi4py, mpi4jax; print(mpi4py.__file__); print(mpi4jax.__file__)"
python -c "from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])"  # must name the module's openmpi (e.g. 'Open MPI v4.1.6')
python -c "import jax, jax_rmhd, orbax.checkpoint; print(jax.__version__)"
```

## 2. Things this codebase does differently (read before running)

- **Precision is an env var, not a flag**: `RMHD_PRECISION=64 python script.py` for
  float64/complex128 (production); the default is 32. It is read ONCE at import time.
- **2D runs (`dims=2`) are single-process only.** Only 3D decomposes across MPI ranks
  (along z; max useful ranks ≈ nz/2). A 2D run under `mpirun -n 4` just warns and wastes
  three ranks.
- **Always launch through `mpirun`, even for one rank**: `mpirun -n 1 python script.py`.
- Editable install caveat: `pip install -e .` means jobs import whatever is currently
  checked out in `~/jax_rmhd` — switching git branches changes what your queued jobs run.
  For A/B benchmarking use the `RMHD_PKG` mechanism in `bench/bench_phase1.py`, never
  PYTHONPATH.

## 3. Interactive test drive (small, 2D)

Login nodes are shared — anything heavier than imports belongs in a job. Interactive node:

```bash
srun --pty -A fc_kawturb -p savio3 -N 1 -t 00:30:00 bash
source activate jax_cpu && export PYTHONNOUSERSITE=1 RMHD_PRECISION=64
cd ~/jax_rmhd && mpirun -n 1 python tests/test_forcing_smoke.py   # ends "ALL PASS"
```

`examples/orzag-tang-2D.ipynb` and `forced-turbulence-2D.ipynb` are current worked
examples (some older notebooks predate the API and error as-is — see CLAUDE.md).

## 4. Batch jobs (3D, MPI)

savio3 is **per-node scheduled**: you are charged for whole nodes (32 cores each at
1 SU/core-hr) no matter how many cores you request — so always use all 32
(`--ntasks-per-node=32`), sized so nz divides evenly by total ranks and nz/ranks ≥ 2.
Template (adapted from `slurms/forced_turbulence_64cubed.sh`, which is runnable as-is):

```bash
#!/bin/bash
#SBATCH --job-name=my_run
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=my_run_%j.out
#SBATCH --error=my_run_%j.err
#SBATCH --mem=0

module purge
module load anaconda3 gcc openmpi
source activate jax_cpu
export PYTHONNOUSERSITE=1

# one process per core, no thread oversubscription
export OMP_PROC_BIND=close OMP_PLACES=cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 XLA_CPU_ASYNC_THREAD_COUNT=1
export OMPI_MCA_pml=ucx

export RMHD_PRECISION=64

PY=$HOME/.conda/envs/jax_cpu/bin/python
REPO=$HOME/jax_rmhd
time mpirun -n $SLURM_NTASKS "$PY" -u "$REPO/tests/forced_turbulence_64cubed.py"
```

Submit / watch / cancel: `sbatch job.sh`, `squeue -u $USER`, `scancel <jobid>`. Multi-node:
raise `--nodes` and keep `--ntasks-per-node=32` (e.g. 4 nodes = 128 ranks needs nz ≥ 256).
`sq`-style pending reasons and SU accounting: see the official docs' "Running Your Jobs".

## 5. Sanity battery on the cluster

After any env rebuild or fresh clone, from an interactive node or a small job:

```bash
mpirun -n 1 python tests/test_forcing_smoke.py            # ALL PASS
mpirun -n 1 python tests/test_forcing_norm_per_step.py    # ALL PASS
mpirun -n 4 python tests/test_advection.py                # 4th-order z-convergence
mpirun -n 2 python tests/test_restart_resharding.py && \
mpirun -n 4 python tests/test_restart_resharding.py       # same snap_path, ALL PASS
```

## 6. Common failure modes

- Every rank prints `rank 0 / size 1` under `mpirun -n N`: mpi4py linked against the wrong
  MPI — rebuild from source per section 1 (both mpi4py AND mpi4jax, `--no-cache-dir`).
- Imports resolve to `~/.local/...`: user-site shadowing — see the PYTHONNOUSERSITE note.
- Job pends forever: check `--account=fc_kawturb` and partition spelling; `squeue -u $USER
  --format="%.10i %.9P %.16j %.8T %.10r"` shows the reason code.
- Fields silently NaN with `cfl_every > 1` from a quiescent start: documented hazard — use
  `cfl_every > 1` only from developed states (see CLAUDE.md).
- Snapshot directory errors on restart: params mismatch guard (`params.save` records the
  constructor args; a differing record is a hard error, by design) or a pre-forcing-era
  snapshot needing `snapshot_io.old_snapshot_repair` — read the error text, it points at
  the fix.
