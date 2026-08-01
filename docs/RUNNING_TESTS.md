# Running the tests

A guide for people working on jax_rmhd. Two minutes to read; covers running the
existing tests and writing new ones.

## The one command

From the repo root:

```
pip install -e ".[test]"   # once; needs an MPI toolchain on clusters, none on a laptop
make test
```

`make test` runs the fast tier twice: once at double precision (`RMHD_PRECISION=64`)
and once at single (the production default, 32). Everything in it runs on a laptop
with no MPI installed, in a few minutes. Other targets: `make test-fast` (fp64 only),
`make test-slow` (minutes-scale convergence studies), and
`make test-one T=tests/test_infra.py` for a single file.

You don't need MPI on your laptop. If `mpi4py` isn't importable, the test harness
automatically installs a fake single-process MPI layer (`tests/local_mpi_stub.py`)
and creates 4 fake XLA devices so even the multi-device `comm_backend="jax"` tests
run single-process. If the real thing is installed, it's used and the stub stays out
of the way.

## Two kinds of test runs

**Local (pytest).** `pytest tests` (or `make test`) discovers and runs everything
that works in a single process. By default it excludes tests marked `mpi`, `slow`,
and `savio` — you get the fast tier. Markers:

| marker | meaning | how to include |
|---|---|---|
| `slow` | minutes-scale study | `pytest tests -m slow --runslow` |
| `mpi` | needs a real `mpirun -n N` | run the file as a script under mpirun |
| `savio` | needs cluster hardware/walltime | `RMHD_SAVIO=1` on the cluster |
| `fp32` / `fp64` | needs that precision session | run under matching `RMHD_PRECISION` |
| `multidev` | needs ≥4 XLA devices | automatic (fake devices locally) |

Precision is fixed once per process (read at `import jax_rmhd`), which is why
`make test` runs two separate pytest sessions rather than mixing precisions.

**Cluster (scripts under mpirun).** Every test file is also a standalone script, and
that is how multi-rank testing works — pytest is never run under mpirun:

```
mpirun -n 4 python tests/test_halo_width.py
```

Each script prints `[PASS]`/`[FAIL]` lines, ends with an `ALL PASS` /
`SOME CHECKS FAILED` banner, and exits nonzero on failure (a single bad rank fails
the whole job). Note `test_restart_resharding.py` is two-phase by design: run it
first at `-n 2`, then at `-n 4` against the same snapshot dir (see
`slurms/test_restart_resharding.sh`). `tests/test_backend_jax_mpi.py` is the
multi-GPU driver (see `slurms/test_backend_jax_gpu.sh`).
`bench/savio_scaling/` is a timing benchmark, not a test.

## The whole suite on Savio: one sbatch each

You rarely need to run individual scripts by hand. From the repo root (or anywhere —
logs land under the submission cwd):

```
sbatch slurms/run_test_suite_cpu.sh     # savio3: every CPU job, ~30-45 min
sbatch slurms/run_test_suite_gpu.sh     # savio4_gpu: the 5-phase backend battery
```

Both wrappers call `tests/run_savio_suite.py`, which reads the job table in
`tests/savio_manifest.py` and subprocess-runs each test file in script mode at its
proper rank counts (`mpirun -n 2/4/8` on CPU; the two srun launch modes on GPU).
Per-phase output is teed into `$RMHD_TEST_OUTDIR` (default
`data/test_suite_{cpu,gpu}_<jobid>/<job>_fp<prec>/phaseN_<label>.log`), each job runs
in its own clean scratch cwd (stale snapshot dirs can never trip the layout guard),
and cheap single-process files run at both precisions. The pass rule per phase is
**exit 0 AND the `ALL PASS` banner** — mpirun can mask a rank's exit code — and the
job ends with a summary table and exits nonzero on any failure.

Extra arguments to `sbatch slurms/run_test_suite_*.sh` are forwarded to the runner:

```
sbatch slurms/run_test_suite_cpu.sh --only halo --only parseval   # substring filter
sbatch slurms/run_test_suite_cpu.sh --precision 64                # one session only
sbatch slurms/run_test_suite_cpu.sh --list                        # print the table, run nothing
python tests/run_savio_suite.py --list                            # same, from a login node
```

Adding a test to the suite = adding one entry to `tests/savio_manifest.py` (see the
field comments there). The targeted single-purpose slurm scripts
(`test_backend_jax_gpu.sh`, `test_restart_resharding.sh`, …) still work and remain
the right tool when iterating on one failure.

## Writing a new test

Copy the shape of `tests/test_infra.py`. The skeleton:

```python
from _rmhd_testing import bootstrap, checks, ctx, make_state, snap_dir
bootstrap()                      # MUST be the first thing, before jax_rmhd

import jax.numpy as jnp
import jax_rmhd as jr

def test_my_thing():
    p, kgrid = ctx(nx=32, ny=32)          # cached (params, kgrid); don't mutate
    state = make_state(p)                  # always fresh -- see donation rule
    ...
    with checks() as c:                    # or just plain asserts
        c.check("what this asserts", condition, detail="numbers for the log")

if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))       # keeps `mpirun python tests/...` working
```

Rules that will save you a debugging afternoon (details in CLAUDE.md):

- **`bootstrap()` before `jax_rmhd`.** Precision, the MPI stub, and XLA device
  flags are all consumed at import time. Importing `jax_rmhd` first means they
  silently don't apply.
- **Never reuse a state after passing it to `simulate`/`simulate_scan`.** Buffer
  donation deletes the input arrays ("Array has been deleted"). Build a fresh one
  with `make_state()`; read any diagnostics you need *before* the call.
- **Never mutate `ctx()` results.** They're cached and shared; `Parameters` hashes
  by identity, so mutation silently reuses a stale jit compile. Need a modified
  one? `ctx(**different_kwargs)` or `fresh_params()`.
- **Write snapshots into `snap_dir()`,** never a cwd-relative `data/...` path —
  stale dirs from a previous run trip the snapshot layout guard. If you create a
  snapshot manager yourself, use `managed_manager()` so async orbax writes finish
  before the directory is deleted.
- **Never assert exact end times or snapshot counts.** `simulate` overshoots
  `t_snap`/`t_end` by up to one step (up to `cfl_every` steps in blocks). Compare
  against `float(end_state.t)` and use tolerances or set membership.
- **Keep it cheap.** 16³ grids and ≲50 steps catch almost everything. If it needs
  minutes, mark it `@pytest.mark.slow`; if it needs the cluster, `savio`.
- Tolerances: bitwise (`jnp.array_equal`) for "must be the same code path",
  ~1e-12 relative for "same quantity, different fp path" at fp64, and physics
  bounds (2-3x) only for stochastic injection-rate checks.

## Continuous integration

Two GitHub Actions workflows in `.github/workflows/`. Both **report**; neither gates a
merge. That is deliberate: tests legitimately lag new physics, and a required check
creates pressure to edit tests mid-experiment just to unblock a merge.

- `fast.yml` — jobs `lint` + `test`. Runs `make test` (both precision sessions) and ruff,
  ~3–5 min. The ruff version is read back out of `pyproject.toml` by the workflow, so CI
  and a local run cannot disagree about it.
- `mpi2.yml` — installs a real OpenMPI + mpi4py + mpi4jax and runs
  `tests/run_savio_suite.py --only halo_width --only energy_parseval` under
  `mpirun -n 2`. Reusing the Savio driver rather than hand-rolling mpirun lines is what
  buys the real pass rule (exit 0 **and** an "ALL PASS" banner). This is the only
  multi-rank coverage outside the cluster.

Three things that will bite anyone editing these workflows:

- **`pip install -e .` cannot work on a runner.** mpi4py and mpi4jax are hard
  dependencies and must stay that way — Savio needs them. `fast` uses
  `pip install --no-deps -e .` plus an explicit dependency list omitting those two, which
  is what lets `conftest`'s MPI stub activate.
- **mpi4jax is sdist-only and links mpi4py's ABI**, so `--no-build-isolation` is correct,
  but `setuptools>=82 wheel nanobind` has to be hand-installed first or the build fails
  every time.
- **`continue-on-error` belongs on the test step, not the job.** At job level a broken
  toolchain install stays green forever.

## Troubleshooting

- `Array has been deleted` — you reused a donated state; see above.
- `ValueError: ... holds a per-rank ... tree` — stale snapshot dir; use
  `snap_dir()` or delete the old `data/...` dir.
- A `multidev` test skips with "needs >=4 devices" — something imported jax before
  `bootstrap()` could set `XLA_FLAGS`; check the import order at the top of your file.
- A test passes under pytest but hangs under `mpirun` — usually a collective
  (allreduce, save, `params.save`) called on a subset of ranks. Every rank must make
  every collective call.
- `ModuleNotFoundError: mpi4py` when running a *legacy* script directly: the older
  unconverted scripts don't self-bootstrap. Either run them via pytest, or:
  `PYTHONPATH=.:tests python -c "import local_mpi_stub, runpy; runpy.run_path('tests/<file>.py', run_name='__main__')"`
