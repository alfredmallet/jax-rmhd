# Running the tests

A guide for people working on taranis. Two minutes to read; covers running the
existing tests and writing new ones.

## The one command

From the repo root:

```
pip install -e ".[test]"   # once; on a cluster wanting real MPI, add the extra: ".[test,mpi]"
make test
```

`make test` runs the fast tier twice: once at double precision (`TARANIS_PRECISION=64`)
and once at single (the production default, 32). Everything in it runs on a laptop
with no MPI installed, in a few minutes. Other targets: `make test-fast` (fp64 only),
`make test-slow` (minutes-scale convergence studies), and
`make test-one T=tests/test_infra.py` for a single file.

You don't need MPI on your laptop. If `mpi4py` isn't importable, the test harness
automatically installs a fake single-process MPI layer (`tests/local_mpi_stub.py`)
and creates 4 fake XLA devices so even the multi-device `comm_backend="jax"` tests
run single-process. If the real thing is installed, it's used and the stub stays out
of the way.

The serial tier (`comm_backend="serial"`, auto-selected when mpi4py/mpi4jax are absent)
is what actually runs on laptops and in CI's `fast` lane; the stub exists only to keep
the mpi4jax code path exercised, serially, alongside it.

**A host with real mpi4py loses the multi-device coverage.** The 4 fake XLA devices are
created only when mpi4py does *not* import, so wherever the real one is installed
`make test` skips every `multidev` test and the `comm_backend="jax"` backend is not
exercised at all. Force the devices to run those:

```
XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest tests/test_backend_jax.py
```

Known there on the M1 laptop (jax 0.10.0, CPU):
`test_same_seed_run_matches_serial_reference` fails with `forcing_state` differing by
1.4e-13 between the backends and the final `t` by one ulp. Pre-existing — it reproduces identically on the
pre-refactor tree — and worth a look by whoever owns the jax backend.

**Running from a git worktree.** taranis is installed editable against the main checkout,
so a subprocess — or a script-mode run (`python tests/test_x.py`, where `sys.path[0]` is
`tests/`) — inside a worktree imports the MAIN tree, not the worktree. `python -m pytest`
from the worktree root is unaffected (cwd wins), which is what `make test` runs. Set
`PYTHONPATH=<worktree>` for anything that spawns a subprocess.

## Two kinds of test runs

**Local (pytest).** `pytest tests` (or `make test`) discovers and runs everything
that works in a single process. By default it excludes tests marked `mpi`, `slow`,
and `savio` — you get the fast tier. Markers:

| marker | meaning | how to include |
|---|---|---|
| `slow` | minutes-scale study | `pytest tests -m slow --runslow` |
| `mpi` | needs a real `mpirun -n N` | run the file as a script under mpirun |
| `savio` | needs cluster hardware/walltime | `RMHD_SAVIO=1` on the cluster |
| `fp32` / `fp64` | needs that precision session | run under matching `TARANIS_PRECISION` |
| `multidev` | needs ≥4 XLA devices | fake devices, but only on a host without mpi4py (above) |

Precision is fixed once per process (read at `import taranis`), which is why
`make test` runs two separate pytest sessions rather than mixing precisions.

Note the `slow` marker needs BOTH `-m slow` and `--runslow`: `--runslow` lifts conftest's
skip, while `-m slow` overrides pyproject's default `addopts` deselection. `make test-slow`
passes both. In script mode the equivalent is `RMHD_RUNSLOW=1`.

**Per-equation-set gates.** RMHD's are spread across most of `tests/`; GDI's are in
`test_gdi_linear.py`; compressible MHD (`eqtype="CMHD"`, plans/CMHD_PLAN.md) has four
files:

| file | tier | what it covers |
|---|---|---|
| `test_cmhd_linear.py` | fast | dispersion relations (Alfvén/fast/slow, three angles, both γ), exact dissipation-only decay, the CFL bound, scheme cross-checks |
| `test_cmhd_conservation.py` | fast | bitwise mass and mean B, the div-B random walk, energy/cross-helicity drift, bitwise snapshot restart, eqpars round-trip and every configuration error |
| `test_cmhd_diagnostics.py` | fast | `diagnostics.cmhd` — normalization against numpy, spectra summing to the energies, energy-budget closure, the monitors, read-only safety |
| `test_cmhd_expansion.py` | fast | the expanding box (EBM): expansion-off is bitwise the non-EBM path (the RHS is independent of `state.t`) plus an analytic Beltrami decay, the bitwise ρ′/B′ k=0 pair and the uniform-state u_⊥ ∝ a⁻¹ ODE (its dt-order is the **stage-time** gate, `fp64`), raw backgrounds tracking a⁻²/a⁻²/a⁻¹, div B under expansion, the WKB δu ∝ a^(−1/2) exponent with its ε_WKB discriminator, eqpars round-trip, every configuration error, and a bitwise mid-expansion restart |
| `test_cmhd_orszag_tang.py` | **`slow`** + `fp64` | the Orszag–Tang validation run (three runs, ~2.5 min: 256²×4 to t=0.5 plus two fixed-dt smooth-window runs). Not in `make test`; `make test-slow` picks it up |

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

Two narrower submissions exist for the REFACTOR_PLAN close-out checks (2026-08-22) and
stay useful as the "did the multi-rank paths survive" pair after any core change:
`sbatch slurms/refactor_check_cpu.sh` (savio3, ~15–20 min: the mpirun jobs —
resharding 2→4, the 8-rank forced-turbulence checkpoint/reload, halos, stencils,
`cfl_every`, the IF/IMEX steppers — plus the single-process files the refactor touched)
and `sbatch slurms/refactor_check_gpu.sh` (savio4_gpu: the backend battery, the only
real exercise of `comms.Runtime` bringing up `jax.distributed`, the `shard_call`
boundary and `kgrid.lin` under a real mesh). Both forward extra arguments to the runner.
`mpirun -n 4 python tests/test_snapshot_roundtrip.py` is not a check — that file
soft-skips every test at `size > 1`.

Adding a test to the suite = adding one entry to `tests/savio_manifest.py` (see the
field comments there). The targeted single-purpose slurm scripts
(`test_backend_jax_gpu.sh`, `test_restart_resharding.sh`, …) still work and remain
the right tool when iterating on one failure.

## Writing a new test

Copy the shape of `tests/test_infra.py`. The skeleton:

```python
from _rmhd_testing import bootstrap, checks, ctx, make_state, snap_dir
bootstrap()                      # MUST be the first thing, before taranis

import jax.numpy as jnp
import taranis as jr

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

- **`bootstrap()` before `taranis`.** Precision, the MPI stub, and XLA device
  flags are all consumed at import time. Importing `taranis` first means they
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

- **`pip install -e ".[test]"` is enough on a stock runner.** mpi4py/mpi4jax are the
  optional `mpi` extra (`pyproject.toml`), not base dependencies, so `fast` installs the
  package plainly — no explicit dependency list, no `--no-deps` — and `comm_backend`
  auto-resolves to `"serial"`. `conftest`'s MPI stub still activates on top of that
  (mpi4py is absent), which is what keeps the mpi4jax code path exercised, serially, in
  this job too.
- **mpi4jax is sdist-only and links mpi4py's ABI**, so `--no-build-isolation` is correct
  in `mpi2.yml` (which installs the real thing), but `setuptools>=82 wheel nanobind` has
  to be hand-installed first or the build fails every time.
- **`continue-on-error` belongs on the test step, not the job.** At job level a broken
  toolchain install stays green forever.

## Troubleshooting

- `Array has been deleted` — you reused a donated state; see above.
- `ValueError: ... holds a per-rank ... tree` — stale snapshot dir; use
  `snap_dir()` or delete the old `data/...` dir.
- A `multidev` test skips with "needs >=4 devices" — either mpi4py imports on this host
  (no fake devices are created then; force them with `XLA_FLAGS`, above), or something
  imported jax before `bootstrap()` could set `XLA_FLAGS` — check the import order at the
  top of your file.
- A test passes under pytest but hangs under `mpirun` — usually a collective
  (allreduce, save, `params.save`) called on a subset of ranks. Every rank must make
  every collective call.
- `ModuleNotFoundError: mpi4py` when running a *legacy* script directly: the older
  unconverted scripts don't self-bootstrap. Either run them via pytest, or:
  `PYTHONPATH=.:tests python -c "import local_mpi_stub, runpy; runpy.run_path('tests/<file>.py', run_name='__main__')"`
