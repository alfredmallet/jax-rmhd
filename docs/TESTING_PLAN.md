# TESTING_PLAN.md — systematizing the test suite

Status: Phases 0-2 DONE (2026-07-30) — asserts added, benchmark moved to
bench/savio_scaling (both its bugs fixed), `tests/_rmhd_testing.py` + conftest +
pyproject/Makefile landed, docs/RUNNING_TESTS.md written; all 5 fast scripts
converted to pytest-style (dual-mode, `_LEGACY_SCRIPTS` removed from conftest —
41 tests in the fp64 session). Phase 3 DONE (2026-07-30) — 5 new test files, 68
tests in the fp64 session (55 + skips at fp32). Phase 6 DONE (2026-07-30, landed
ahead of 4/5) — one-sbatch CPU + GPU suites; awaiting first real Savio run.
Phase 4 DONE (2026-07-30) — advection/dissipation converted (71 fp64 / 58 fp32 +
1 slow-tier test), both added to the Savio manifest. Phase 5 DONE (2026-07-30) —
6 new files + 2 additions, all in the Savio manifest; **found and fixed a real
lsrk54 coefficient bug**; suite now 87 fp64 / 77 fp32 (+1 slow), fp32 session
widened to the whole suite. Phase 7 not started.
Produced from an audit of `tests/` (sonnet) and an implementation plan (opus),
2026-07-30. Companion docs: PHASE3_RESULTS.md, checkpointing.md.

## Audit summary (current state)

11 standalone scripts, no pytest/conftest/CI/lint. Pass/fail is inconsistent:

- Hard assert / exit code: `test_backend_jax.py`, `test_restart_resharding.py`,
  `forced_turbulence_64cubed.py`, `test_backend_jax_mpi.py --compare`.
- **Printed "ALL PASS"/"SOME CHECKS FAILED" banner but exit 0 on failure**:
  `test_energy_parseval.py`, `test_forcing_norm_per_step.py`, `test_forcing_smoke.py`,
  `test_halo_width.py`.
- Human-read printouts/plots, no check at all: `test_advection.py` (error table),
  `test_dissipation.py` (plot), `test_savio_scaling.py` (timing benchmark).

Duplication: the `check()`/`all_ok` helper is copy-pasted in 5 files with signature
variants; `RMHD_PRECISION=64` boilerplate, `init_fields`/`zero_ic` closures, and
Parameters/kgrid/initialize construction repeated per file. `local_mpi_stub.py` is the
only shared helper.

Verified problems found along the way:

- `tests/data/` is **gitignored** — the legacy-snapshot fixture used by
  `test_backend_jax.py` §I exists only on this machine; on a fresh clone §I silently
  `[SKIP]`s.
- `tests/savio_scaling/test_savio_scaling.py` is broken (calls
  `snapshot_manager_setup` without `params`, before `params` exists) and its
  `test_*.py` name makes it a pytest collection error. It's a benchmark, not a test.
- `pyproject.toml` `packages = ["jax_rmhd"]` omits `jax_rmhd.physics` — non-editable
  installs ship a package that can't import its own equation registry.
- `test_advection.py` imports `mpi4jax` directly (violates the comms.py-only rule).
- `test_forcing_smoke.py` reuses a kgrid with a different `Parameters` (lines 71-72).
- Several tests write cwd-relative `data/...`; reruns trip the snapshot layout guard.

Coverage gaps (no test at all): dealias mask; Poisson bracket / NonlinearTerm
correctness; z-stencil accuracy in isolation; standalone snapshot round-trip;
`perpspec`/`parspec`; `Parameters.from_snapshot`; momentum↔elsasser relationship;
2D/3D parity; fp32 (every test forces 64; the production default is never tested);
`cfl_every>1` vs `=1` agreement; `lsrk_scan` bitwise claim; `rk44`/`lsrk54` schemes;
`find_items`/`load_slice`; `_AncientCkptState` repair; `allreduce_max` under `"jax"`.

## Design decisions

### A. Import order: `bootstrap()` first

`config.py`/`comms.py` import mpi4py/mpi4jax unconditionally, and `RMHD_PRECISION` /
`XLA_FLAGS` are consumed at import. Every test module starts:

```python
from _rmhd_testing import bootstrap; bootstrap()   # env + stub, before jax_rmhd
import jax_rmhd as jr
```

Works in both modes: pytest's `tests/conftest.py` puts `tests/` on `sys.path` and calls
`bootstrap()` before any test module imports; script mode (`mpirun -n 4 python
tests/test_x.py`) puts `tests/` at `sys.path[0]` automatically. `bootstrap()` installs
`local_mpi_stub` only if real mpi4py import fails (`RMHD_TEST_FORCE_STUB=1` to force),
and sets `--xla_force_host_platform_device_count=4` only when the stub is active.
`_rmhd_testing.py` imports nothing heavy at module scope; `local_mpi_stub` gets an
idempotency guard.

### B. Dual-mode tests: no fixtures, cached builders

Zero-argument `test_*()` functions + a `script_main(globals())` footer that runs them in
definition order, prints the familiar `[PASS]`/`[FAIL]` + banner, allgathers ok across
ranks when size>1, and exits 0/1. So `mpirun -n N python tests/test_X.py` keeps working
on Savio *and* `pytest tests/` works locally, on the same file.

Hard rules (from CLAUDE.md):
- `ctx(**kw)` may lru_cache `(Parameters, K_Grids)` — identity sharing avoids
  re-tracing. It must **never cache a SimulationState** (donation deletes it);
  `make_state()` is always fresh. Never mutate a cached `Parameters`;
  `fake_ranked_params()` gives uncached instances for rank-spoofing tricks.

### C. Precision: two pytest sessions

`RMHD_PRECISION` is read once at import. Default session fp64; `fp32`/`fp64` markers
auto-skip against the live `jax_enable_x64`. `make test` runs pytest twice
(`RMHD_PRECISION=64` then `RMHD_PRECISION=32 -m fp32`). Cross-precision comparisons, if
ever needed, use subprocess isolation.

## Phases (each independently landable)

### Phase 0 — make failing tests fail (~½ day) — DO FIRST

Add `assert all_ok` (allgathered under MPI) to `test_energy_parseval.py`,
`test_forcing_norm_per_step.py`, `test_forcing_smoke.py`, `test_halo_width.py`.
Move `tests/savio_scaling/test_savio_scaling.py` → `bench/savio_scaling.py`, fix its
`snapshot_manager_setup(params, ...)` call, update the 7 `scaling*.sh` paths.
~10 lines + 1 move. Four false-green tests stop lying.

### Phase 1 — infrastructure (1-2 days)

Create:
- `tests/_rmhd_testing.py` (~250 lines): `bootstrap()`, `soft_assert`/`checks()`
  (replaces the 5 copy-pasted `check()`s; raises one AssertionError listing all
  failures), `ctx()`, `make_state()`, ICs (`zero_ic`, `alfven_ic`, `multimode_ic`,
  `tiny_alfven_ic`), `snap_dir()` ctx mgr, `managed_manager()` (always
  `wait_until_finished()` + `close()` before dir removal — orbax async-write flake
  guard), `energy()`, `fit_order()`, `mpi_size()`, `script_main()`.
- `tests/conftest.py` (~90 lines): sys.path insert + `bootstrap()` as first lines;
  register/apply markers (`mpi`, `savio`, `slow`, `fp32`, `fp64`, `multidev`);
  ignore `tests/data`, `tests/savio_scaling`.
- `Makefile`: `test` (fp64 fast tier + fp32 tier), `test-slow`, `lint`.

Modify `pyproject.toml`: `[project.optional-dependencies] test = ["pytest>=7",
"matplotlib"]`; **fix `packages` to include `jax_rmhd.physics`**;
`[tool.pytest.ini_options]` with testpaths/norecursedirs/markers and default
`-m 'not mpi and not slow and not savio'`; minimal `[tool.ruff]` (F, E9 only).
Update CLAUDE.md's "No pytest suite" paragraph with the new contract.

### Phase 2 — convert the 5 fast scripts (2-3 days, one PR each) — DONE 2026-07-30

Notes from landing: energy_parseval + both forcing tests got precision-dependent
tolerances and pass an RMHD_PRECISION=32 session too (not yet in the fp32 tier —
that expands in Phase 5). test_forcing_smoke's kgrid/params mismatch fixed by
giving ForcingTerm-off its own ctx. test_backend_jax's shared end-states live in
lru_cached builders (`_end_run`, `_per_rank_tree`) returning numpy copies + saved
snapshot dirs; the live states never leave the builders, so tests are
order-independent; decoy-tmp-dir tests mutate copies, never the cached dirs; every
test marked multidev+fp64; the untracked tests/data legacy fixture soft-skips via
print+return (pytest.skip would crash script_main). Legacy section of conftest
removed.

Order: `test_halo_width` → `test_energy_parseval` → `test_forcing_norm_per_step` →
`test_forcing_smoke` → `test_backend_jax`. Per file: `bootstrap()` header, delete local
`check()`, split sections into `test_*` functions, `snap_dir()` instead of `data/...`,
`script_main` footer. Specific fixes: forcing tests' hardcoded `complex128` →
`get_precision_types()`; `test_forcing_smoke` kgrid/params mismatch; `test_backend_jax`
shared end-states hoisted into an lru_cached builder returning **numpy** copies (kills
the donation ordering hazard); all of it marked `multidev`.

### Phase 3 — high-value new tests (3-4 days; biggest silent-wrongness gaps) — DONE 2026-07-30

Notes from landing: all 5 files are dual-mode with precision-dependent tolerances
(fp32 sessions run them too). Findings: `z_derivatives` on a constant is ~1e-16/1e-14
round-off, not bitwise zero (d⁴/dz⁴'s /dz⁴ amplifies) — asserted "near machine
precision"; the fp32 d/dz fitted order lands at exactly 3.80 (noise-floor
contamination at nz=32) — threshold 3.5 there, 3.8 at fp64. The 2D ideal-run
invariant test calls `run.block_of_steps` directly for an exact step count; its IC
shares a cos(x)cos(y) component between phi and psi so cross-helicity is O(1), not
round-off. Legacy/ancient checkpoint trees are synthesized via a bare
`StandardCheckpointHandler` — `save()` alone leaves OCDBT unmerged and unreadable;
`handler.finalize(item_dir)` is required (orbax 0.11.x trap, documented in
`_write_raw_step`). `from_snapshot` "warns" via `print()` on rank 0, not
`warnings.warn` — test_params captures stdout AND the warnings module so either
works. Snapshot/params tests soft-skip (print+return) when `mpi_size()>1` —
rank-local tmp dirs + spoofed ranks are single-process by construction. Known nit,
not fixed: `config.from_snapshot` leaks a file handle (`json.load(open(path))` →
ResourceWarning).

- `tests/test_bracket.py`: bracket vs analytic (well below 2/3 cutoff, rel < 1e-12);
  antisymmetry + `bracket(a,a)==0` bitwise; `Σ f·{f,g} == 0` conservation identities;
  NonlinearTerm exactly zero outside the dealias mask; 20-step ideal 2D run conserving
  energy and cross-helicity to 1e-6.
- `tests/test_z_stencils.py`: `z_derivatives` convergence on `cos(2z)` — assert fitted
  order > 3.8 for d/dz, > 1.8 for the d⁴/dz⁴ stencil (nominally 2nd order — pin it);
  exact zero on constants; periodic wraparound vs known FD dispersion factor.
- `tests/test_snapshot_roundtrip.py`: minimal save/load bitwise round-trip incl.
  forcing_key; **`test_reads_never_construct_a_manager`** (monkeypatch
  `CheckpointManager.__init__` to raise, then `load_snapshot`/`get_saved_steps` — 
  enforces the deadlock invariant); `forcing_scale=None` rejected; pruning window;
  **synthesized** `_LegacyCkptState`/`_AncientCkptState` trees written via bare
  `StandardCheckpointHandler` → `old_snapshot_repair` load + idempotency (replaces the
  untracked `tests/data` fixture dependency).
- `tests/test_params.py`: `from_snapshot` round-trip of all ctor args (incl.
  list→tuple restore); identical-resave no-op (bytes+mtime); differing-resave raises;
  backfill of deleted keys; unknown-key warn; precision-mismatch warn; overrides win;
  transport keys not compared.
- `tests/test_dealias.py`: mask cutoff boundary indices exact; kx symmetry;
  `initialize` zeroes an above-cutoff IC mode exactly.

### Phase 4 — advection/dissipation become asserted (1-2 days) — DONE 2026-07-30

Notes from landing: both files call `run.block_of_steps` directly (jitted like
run.py's mpi4jax path for advection) for exact step counts — no snapshot manager
at all. Measured advection L2 orders: fp64 3.99 fast / 4.0 coarse-triple slow;
fp32 fast lands at 3.85 (nz=64 error ~4.1e-7 sits near the accumulated round-off
floor, ~20% inflated), so the single 3.5 threshold holds at both precisions. The
slow tier is fp64-marked: at fp32 every nz≥64 error pins at ~4-5e-7 (fitted order
−0.10 — meaningless). Dissipation finding: **`z_diss` defaults to 0.25** and
`rmhd.LinearTerm` applies it as `z_diss·(dz/2)⁴·d⁴/dz⁴` — it must be set to 0.0
explicitly or it adds spurious kz-envelope decay on top of exp(−4·diss·t) and
breaks the diss=0 conservation check. With it zeroed the law is exact for this IC
(nonlinear terms vanish identically; deviations are a d-independent RK3 amplitude
offset ~−4e-8 in log-energy, which the slope fit cancels): plan tolerances hold
with ≥800x margin at both precisions; only the diss=0 conservation tolerance is
precision-split (1e-6 fp64 / 1e-5 fp32). Both files joined `savio_manifest.py`
as mpi n=4 jobs, both precisions; advection's phase sets `RMHD_RUNSLOW=1` so the
full nz=64→1024 study runs on the cluster. conftest's `collect_ignore` entries
for the two files removed; suite is now 71 fp64 + 58 fp32 (+1 slow, deselected
by default).

- `test_advection.py`: fast tier (nz 16/32/64, ~30 s) asserting fitted z-convergence
  order > 3.5 and monotone decreasing error; `@slow` full tier (nz 64→1024) asserting
  order > 3.5 on the coarse triple (fine end flattens on the fixed-dt O(dt³) floor —
  expected, assert non-increase only). Replace direct `mpi4jax.allreduce` with
  `comms.allreduce_sum`; use `float(end_state.t)`, never the target time (overshoot).
  Threshold rationale: 3.5 catches "silently 2nd order" without tripping on z_diss/dt
  contamination; the sharp 4th-order assertion lives in test_z_stencils.
- `test_dissipation.py`: IC sits entirely at k⊥²=2, so energy decays by
  exp(−4·diss·t). Assert per-diss log-ratio within 1% of −4·diss·t_actual; fitted
  slope within 2%; diss=0 conserves to 1e-6. Drop matplotlib/data writes (plot behind
  `RMHD_TEST_PLOTS=1`).

### Phase 5 — remaining gaps (2-3 days) — DONE 2026-07-30

Notes from landing (all files dual-mode, precision-split tolerances, in the Savio
manifest — mpi n=4 for scheme_equivalence/cfl_every/dims_parity/diagnostics,
serial for forcing_modes and precision_fp32 ("32" session only)):

- **FINDING 1 (real bug, fixed): lsrk54's A5 coefficient was wrong** —
  `-3270041069962/2362476162756` (~-1.3841) broke even the 1st-order condition
  (stability-polynomial z-coefficient 1.0096; Alfven-wave error 4.7e-3 at t=0.5,
  exactly 50*0.0096*dt). Replaced with the published Carpenter & Kennedy (1994)
  value `-1275806237668/842570457699`; the 4th-order conditions were verified in
  exact rational arithmetic and the measured error dropped to 1.6e-11 (vs rk44
  3.9e-11, lsrk33 2.0e-8). The scheme had never been run before this test.
- **FINDING 2: the lsrk_scan bitwise claim is machine-dependent** — under jax
  0.6.2/CPU scan-vs-unrolled differs at round-off for BOTH schemes (~2e-15
  lsrk33 / ~7e-15 lsrk54 after 20 steps; XLA fuses the two loop structures
  differently). CLAUDE.md's "bitwise-identical at fp64" was scoped accordingly;
  the test asserts the plan's fallback rel < 1e-13 (fp64) / 1e-6 (fp32).
- The scheme tests compare against the SEMI-DISCRETE exact Alfven solution
  cos(x)cos(y)cos(z + k_eff*t) with k_eff = (8sin(dz)-sin(2dz))/(6dz) — without
  the z-FD dispersion factor the "exact" comparison bottoms out at ~6e-3.
- cfl_every: tiny-Alfven dt is EXACTLY cfl_safety*dz (1/dz dominates every term),
  so cfl_every=1 vs 4 agree at rel 0.0 observed; nblock rounding pinned via
  t == 12*dt; fixed-dt legacy path bitwise at both precisions.
- dims_parity: z_diss kept at its 0.25 default ON PURPOSE (d4/dz4 goes through
  the same halo path — louder detector); planes bitwise-identical, plane-vs-2D
  observed rel 1.4e-17 (fp64) / 1.5e-8 (fp32) vs 1e-14/1e-5 tolerances.
- forcing_modes: the elsasser->momentum f_phi comparison is round-off (1e-13),
  not bitwise — batch vs flat reduction sums in different order; f_psi == 0 IS
  bitwise. Same-RNG run comparison impossible (n_ou differs) — commented.
- diagnostics: spectrum-integral identities are EXACT (rel 0.0 observed at both
  precisions); note perpspec/parspec return bin CENTERS, edges are center-+dk/2.
- precision_fp32: the default-precision check spawns a subprocess with
  RMHD_PRECISION scrubbed (in-process value is pinned by bootstrap at import);
  fp32 snapshot round-trip observed bitwise. The Makefile fp32 session was
  widened from `-m fp32` to the whole suite (77 passed / 15 skipped) per the
  Phase 2 note.
- test_backend_jax gained test_set_timestep_pmax_matches_serial (z-VARYING
  amplitude IC so shards see different local CFL maxima — shard-identical values
  would not catch a broken pmax); test_snapshot_roundtrip gained the
  find_items/load_slice smoke (slice matches saved fields bitwise).
- Pre-existing ruff F-level debt (19 hits: __init__ re-exports, legacy MPI
  driver, notebook — ruff 0.16 now lints notebooks) left for Phase 7; all new
  files are clean.

- `test_scheme_equivalence.py`: lsrk_scan True/False **bitwise** (`array_equal`) at
  fp64 — separate kgrids per Parameters; if it fails, that's a finding: fix CLAUDE.md
  and relax to rel<1e-13. Exercise `rk44`/`lsrk54` (currently never run) on the exact
  Alfvén solution; `get_scheme("nope")` raises.
- `test_cfl_every.py`: tiny-amplitude Alfvén IC (unforced — avoids the documented
  quiescent-start NaN trap) where the 1/dz CFL term dominates ⇒ dt exactly constant ⇒
  cfl_every=1 vs 4 must agree to rel<1e-14; `adaptive_timestep=False` + cfl_every=4 ⇒
  bitwise identical legacy path; nblock rounding (10 steps @ cfl_every=4 ⇒ 12 steps).
- `test_dims_parity.py`: z-invariant 3D IC, fixed dt (adaptive dt differs 2D vs 3D by
  the 1/dz term) ⇒ every z-plane matches the 2D run to rel<1e-14 and planes stay
  bitwise identical to each other (loud halo-bug detector).
- `test_forcing_modes.py`: elsasser with equal envelopes ⇒ f_psi exactly 0 and f_phi =
  momentum result (same-RNG comparison impossible — n_ou differs; comment it);
  asymmetric `forcing_power_elsasser=(0.6,0.2)` ⇒ energy rate ≈ 0.4 (30%) and
  cross-helicity rate ≈ 0.2 (40%) — catches swapped f_plus/f_minus; 2D momentum
  quiescent ⇒ psi exactly 0.
- `test_diagnostics.py`: perpspec sums to energy (exact identity, rel<1e-12 — guards
  the shared normalization convention); single mode lands in one bin; parspec asserts
  size==1; parspec Parseval.
- `test_precision_fp32.py` (`@fp32` session): default precision is 32; state dtypes;
  forced 16³ run to t=0.5 finite (the "does the production default even work" check);
  fp32 snapshot round-trip to 1e-6.
- `test_backend_jax.py` addition: `set_timestep` under `shard_call` vs serial —
  only direct exercise of `allreduce_max`'s pmax branch.
- Low value, last/skip: `find_items` no-raise smoke, `load_slice` vs
  `load_snapshot` slice.

### Phase 6 — Savio tier (1-2 days) — DONE 2026-07-30 (landed before Phases 4/5)

Notes from landing: `tests/savio_manifest.py` (job table) + `tests/run_savio_suite.py`
(runner) + `slurms/run_test_suite_cpu.sh` / `run_test_suite_gpu.sh` (preambles copied
verbatim from merge_gate_phase2.sh / test_backend_jax_gpu.sh). Launcher templates
come from env (`RMHD_LAUNCH*`, `{n}` = rank count) so cluster knowledge stays in the
wrappers; each (job, precision) runs in a clean scratch cwd under `RMHD_TEST_OUTDIR`
(replaces the plan's "route data/ writers" item); cheap single-process files run
fp64 AND fp32; a failed phase skips the job's remaining phases. Deviation from the
spec: no per-phase timeout in the runner — the GPU wrapper's `timeout 900` lives
inside the launch templates and the sbatch walltime bounds the CPU tier.
advection/dissipation excluded until Phase 4 converts them (exit-0 with no assert
would be a vacuous pass); when Phases 4/5 land, ADD THEIR FILES TO THE MANIFEST.
Runner smoke-tested in the sandbox (`RMHD_LAUNCH="" python tests/run_savio_suite.py
--only ...` runs mpi phases single-process); not yet run on Savio.

pytest never runs under mpirun; scripts stay the Savio driver. Add:
- `tests/savio_manifest.py`: explicit job table (name, script, tier cpu/gpu, phases
  with rank counts, shared_outdir for the 2-then-4-rank resharding sequence, the 5
  GPU backend phases).
- `tests/run_savio_suite.py`: launched inside sbatch; subprocess-runs each phase under
  `mpirun -n N` / `srun`, tees per-phase logs to `$RMHD_TEST_OUTDIR`, pass rule =
  **exit 0 AND banner check** (mpirun can mask rank exits), summary table, exit 1 on
  any failure. `--tier`, `--only`.
- `slurms/run_test_suite_cpu.sh` / `run_test_suite_gpu.sh`: thin wrappers copying the
  existing preambles **verbatim** (esp. `test_backend_jax_gpu.sh`'s NVLIBS / pmix /
  `NCCL_P2P_DISABLE=1` cluster knowledge). Keep the existing targeted scripts.
  Route the cwd-relative `data/` writers through `RMHD_TEST_OUTDIR` with clean per-job
  dirs (stale dirs trip the layout guard).

### Phase 7 — CI + lint (½-1 day) — DONE 2026-07-30

Notes from landing: ruff scope is PINNED on purpose — `[tool.ruff.lint] select = ["F"]`
plus `ruff==0.16.1` in a new `lint` extra, notebooks/examples/a5k excluded. Ruff's
*default* ruleset widens between releases (0.16 both started linting notebooks and
pulled in C408/UP/PL/SIM/...: 237 hits vs 20 for F alone), so inheriting defaults means
an upgrade can turn CI red without a code change. Widening to `I` later is a one-line
config change plus one mechanical `ruff check --fix` commit. The 9 `jax_rmhd/__init__.py`
re-exports are declared in `__all__` rather than deleted; `tests/test_infra.py`'s
`import jax_rmhd` is kept under `# noqa: F401` (it IS the bootstrap-ordering demo).
`fast.yml` = jobs `lint` + `test`; the ruff version is read back out of pyproject by the
workflow (tomllib) so CI and a local run cannot disagree. `mpi2.yml` reuses
`tests/run_savio_suite.py --only halo_width --only energy_parseval` rather than
hand-rolling mpirun lines — that buys the real pass rule (exit 0 AND "ALL PASS" banner).
Two review catches worth remembering: mpi4jax is sdist-only and links mpi4py's ABI, so
`--no-build-isolation` is right but `setuptools>=82 wheel nanobind` must be hand-installed
first or the build fails every time; and `continue-on-error` belongs on the test STEP, not
the job — at job level a broken toolchain install would stay green forever.
NOT YET RUN ON GITHUB: expect 1-2 rounds of iteration on the first push.



Repo is on GitHub (`origin` exists), no `.github/`. `pip install -e .` fails on runners
(mpi4py/mpi4jax hard deps — do NOT demote them; Savio needs them). Workflow:
- `fast` (required): `pip install --no-deps -e .` + explicit dep list, stub kicks in,
  `make test` (both precision sessions) + ruff. ~3-5 min.
- `mpi2` (continue-on-error initially): apt openmpi, real mpi4py/mpi4jax,
  `mpirun --oversubscribe -n 2` on halo_width + energy_parseval — genuine 2-rank
  coverage the local sandbox can never provide.

If Actions is unwanted, `make test` alone is the systematic one-command answer.

## Risk register

| Risk | Mitigation |
|---|---|
| Import order (stub after jax_rmhd) | `bootstrap()` first statement everywhere; tests/ auto on sys.path both modes; idempotent stub |
| Precision fixed at import | two pytest sessions + fp32/fp64 auto-skip markers |
| Buffer donation deletes shared states | never cache SimulationState; numpy copies for shared references |
| jit retrace per fresh Parameters | lru_cached `ctx()` shares identity within a module; never mutate |
| Process-global leaks (`_array_handler_pinned`, `_mesh`, `_dist_initialized`) | document; if order-dependence appears, own pytest session for test_backend_jax |
| Orbax async writes racing tmp cleanup | `managed_manager()` waits + closes before removal |
| Stale `data/` dirs trip layout guard | `snap_dir()` everywhere; per-job dirs on Savio |
| Untracked legacy fixture | synthesize legacy/ancient trees in-test |
| lsrk_scan bitwise claim maybe false | strict assert first; a failure is a documentation finding |
| cfl_every quiescent-start NaN | unforced tiny-Alfvén IC pins dt constant |
| t_end/snapshot overshoot | assert on `float(end_state.t)` and set membership, never counts |
| `jax_rmhd.physics` missing from packages | fix in Phase 1 |

## Effort

Phases 0-3 (≈7 days) deliver most of the value: an honest one-command local suite plus
coverage of the bracket, z stencils, checkpoint round-trip, and Parameters persistence —
none tested today. Full plan ≈ 12-18 days. Every phase lands independently.
