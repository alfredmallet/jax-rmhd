# SERIAL_BACKEND_PLAN.md — make MPI optional via comm_backend="serial"

Goal: `pip install -e .` and full single-process use (2D and 3D size-1) on machines with
no MPI toolchain. mpi4py/mpi4jax move to an optional extra; a third backend `"serial"`
is auto-selected when they are absent. `"mpi4jax"` (CPU production) and `"jax"`
(GPU/NCCL) are untouched when MPI is present.

## Design (agreed 2026-07-31)

### Backend resolution

`Parameters(comm_backend=...)` accepts `"mpi4jax" | "jax" | "serial" | None`.
Default changes from `"mpi4jax"` to `None` = auto-resolve:

1. If a multi-rank MPI launch is detected from the environment but mpi4py is NOT
   importable: **hard RuntimeError**. Never silently degrade under mpirun — every rank
   would run the full domain and overwrite each other's snapshots.
   `launcher_world_size()`: `OMPI_COMM_WORLD_SIZE`, `PMI_SIZE`, `PMIX_RANK`,
   `MV2_COMM_WORLD_SIZE` are definitive (per-process, set only by the launcher).
   `SLURM_NTASKS`/`SLURM_STEP_NUM_TASKS` alone are NOT — they're set for the whole
   allocation, including plain `python` in a batch script → warn, don't error.
2. mpi4py + mpi4jax importable → `"mpi4jax"` (today's default, bit-for-bit unchanged).
3. mpi4py importable, mpi4jax not: real rank/size from COMM_WORLD; if size==1 →
   `"serial"`, else RuntimeError naming mpi4jax. (This half-installed state is common —
   mpi4py builds easily, mpi4jax needs a matching jax.)
4. Neither importable, no launcher detected → `"serial"`, rank=0, size=1, null comm.

Additionally (any resolution path, mpi4py present): if `launcher_world_size() > 1` but
`COMM_WORLD.Get_size() == 1`, hard RuntimeError — mpi4py built against a different MPI
than the launcher; each rank would otherwise run the full domain independently.
(This footgun exists in current code too; the guard fixes it for all backends.)

Explicit `comm_backend="mpi4jax"`/`"jax"` without the imports → ImportError with the
`pip install "taranis[mpi]"` hint. Explicit `"serial"` with size>1 → ValueError.
Auto-resolution to `"serial"` on a no-MPI machine is deliberately silent (the expected
laptop case); the resolved backend is recorded in params.json.
`"serial"` is valid for dims=2 and dims=3 (the dims-2 case is why `"jax"` alone can't
be the fallback: config.py rejects jax+2D, and 2D is single-process by construction).
`"jax"` stays an explicit opt-in — its mesh grabs ALL `jax.devices()`, so making it a
fallback would silently shard over 4 devices on a multi-GPU workstation (or under the
test bootstrap's forced 4 host devices) and then reject nz % 4 != 0.

### Serial semantics (exact, not approximate)

Size-1 periodic z topology: the only neighbor is self.
- `halo_exchange`: `recv_right = f[:, :width]`, `recv_left = f[:, -width:]`
  (same identities the test stub's self-send produces today).
- `allreduce_sum` / `allreduce_max`: identity. (Already the behavior when
  `cart_comm is None`; serial sets `cart_comm=None` even for dims=3, so order the
  serial branch BEFORE any cart_comm use in `halo_exchange`.)
- `params.comm`: a tiny `_NullComm` (Get_rank→0, Get_size→1, bcast→identity) so the
  `size>1`-guarded call sites (`run._start_snapshots`, `Parameters.save`) need no edits.

### Import layer

New module `taranis/_mpi_compat.py`:
- `try: from mpi4py import MPI / import mpi4jax` behind flags `HAVE_MPI4PY`,
  `HAVE_MPI4JAX`; exports `MPI`, `mpi4jax` (None when absent), `_NullComm`,
  `launcher_world_size()` (env sniff above).
- `comms.py` and `config.py` import ONLY from `_mpi_compat` — the "nothing outside
  comms.py imports mpi4jax" rule extends to: nothing outside `_mpi_compat` imports
  mpi4py or mpi4jax at module scope (bench/ and tests/ excepted).

### What does NOT change

- Snapshot layout: size==1 already writes the flat dir; `snapshot_layout` treats
  serial == single-process mpi4jax == jax readers. No snapshot_io layout edits —
  only the `params.size>1` / `comm_backend` branches must tolerate `"serial"`
  (audit, expected no-op).
- `comm_backend` is already in `_TRANSPORT_KEYS`: recorded in params.json, never
  compared → restarts across serial/mpi4jax/jax stay legal with zero format changes.
- `from_snapshot`: strip `comm_backend` from the record before re-init (re-resolve on
  the current machine) unless the caller overrides it explicitly. A Savio-written
  params.json recording "mpi4jax" must load on a laptop without MPI.
- Physics, timestepping, forcing, diagnostics: untouched. Dispatch stays plain
  python `if params.comm_backend == ...` (Parameters is static — never lax.cond).

### Known behavior deltas (document, don't fight)

- Serial is NOT bitwise-identical to size-1 mpi4jax: dropping the mpi4jax ops removes
  their tokens and XLA fuses differently (same class as `lsrk_scan`,
  test_scheme_equivalence). Equivalence tests use tolerances (~1e-13 fp64 relative,
  after O(10) steps), not equality.
- Expected small serial speedup (no token ordering constraints); measure, note in
  docs/performance.md.

## Agent tasks

Order: A1 → A2 → (A3 ∥ A4) → A5. Each agent reads CLAUDE.md + this file first.
Sandbox has no MPI (it IS the target environment for serial) — run `make test` there;
never cp the repo inside the sandbox; no background processes; chunk runs ≤45 s.

### A1 — core implementation (opus)

Files: `taranis/_mpi_compat.py` (new), `taranis/comms.py`, `taranis/config.py`.
- `_mpi_compat` as specced; comms/config switch to it.
- `comms.COMM_BACKENDS += ("serial",)`; serial branches in `halo_exchange`,
  `allreduce_sum`, `allreduce_max` (serial branch first; assert width bounds same as
  mpi4jax path). `init_backend`: no-op for serial.
- `config.py`: `comm_backend=None` default + `_resolve_backend()`; serial sets
  comm=_NullComm(), rank=0, size=1, cart_comm=None, neighbors=None (dims=3 included);
  keep the 2D size>1 warning; `"jax"`+2D rejection unchanged. `_init_args` records the
  RESOLVED backend (params.json documents what actually ran). `from_snapshot` drops
  the recorded `comm_backend` (re-resolves) unless overridden.
- Acceptance: with real mpi4py installed, `git diff` of traced HLO for an mpi4jax run
  is empty (import indirection only); in the no-MPI sandbox,
  `python -c "import taranis; taranis.Parameters(...)"` works with no stub, 2D and 3D.

### A2 — call-site audit (sonnet)

- Grep every `params.comm`, `params.size`, `params.cart_comm`, `comm_backend`,
  `MPI.`, `mpi4jax` use in `taranis/` and confirm serial correctness
  (run.py:141 bcast, config.save bcast, snapshot_io branches at lines ~60/66/75,
  grids.py:83, physics/__init__.py `_halo_start_enabled` — decide: serial in
  `_HALO_START_BACKENDS`? Answer: yes, halo_start is free serial, but verify the
  pre-issued-halo assertion path in z_derivatives tolerates the self-slabs).
- Deliverable: short report appended to this file (## A2 findings) + minimal fixes.

### A3 — tests (sonnet)

Files: `tests/test_backend_serial.py` (new), `tests/_rmhd_testing.py`,
`tests/local_mpi_stub.py`, `tests/conftest.py`.
- New module (standalone-script contract: `from _rmhd_testing import bootstrap;
  bootstrap()` header, `script_main(globals())` footer; never cache states — donation;
  never mutate ctx() results):
  - halo_exchange serial vs stubbed-mpi4jax: exact identity per plane, widths 1..2.
  - N-step trajectory serial vs stubbed-mpi4jax, 2D and 3D, forced and unforced:
    tolerance comparison (see deltas above).
  - resolution matrix: auto→serial under no-MPI; explicit mpi4jax without MPI →
    ImportError; fake `OMPI_COMM_WORLD_SIZE=4` in env + no mpi4py → RuntimeError;
    fake `OMPI_COMM_WORLD_SIZE=4` + (stub) mpi4py reporting size 1 → RuntimeError
    (mismatched-MPI guard). (Monkeypatch env + reload-safe: test via
    `_resolve_backend` directly, not by re-importing taranis.)
  - snapshot roundtrip serial→serial and cross-restore serial↔(stub) mpi4jax size-1
    (flat layout both ways); `from_snapshot` of a params.json recording "mpi4jax"
    resolves to serial on the stub-free path.
- `bootstrap()`: keep the stub install (the mpi4jax code path must stay testable
  serially) but ALSO exercise true-no-stub serial: add `bootstrap(stub=False)` or an
  env knob so test_backend_serial runs against the real absent-MPI import path.
  Keep fake-4-device XLA flags (multidev/"jax"-backend tests unchanged).
- conftest: nothing new to skip — serial tests are the "runs anywhere" tier.

### A4 — packaging + docs (sonnet)

- `pyproject.toml`: move `mpi4py`, `mpi4jax` from `dependencies` to
  `[project.optional-dependencies] mpi = [...]`; `test` extra unchanged (stub covers
  it).
- Install-path docs — three distinct audiences, don't conflate:
  - README / CLAUDE.md setup block: `pip install -e .` (laptop, serial backend) vs
    `pip install -e ".[mpi]"` (generic Linux box with a working MPI toolchain;
    zsh needs the quotes).
  - SAVIO_CPU_SETUP.md / SAVIO_GPU_SETUP.md: the `[mpi]` extra is deliberately NOT
    used there. Both guides hand-build mpi4py (`MPICC=$(which mpicc) ... --no-binary`)
    and mpi4jax (after jax, GPU: after `jax[cuda12]` + cuda module) to pin the
    cluster MPI/ABI; the extra would let pip bypass that. Keep the manual steps and
    the closing plain `pip install -e .` (deps already satisfied); add one sentence
    saying why `[mpi]` is not used on Savio, and update the "hard requirements even
    for single-rank runs" comments (now: required for any multi-rank run; single-rank
    works serially without them, but Savio envs should still install them).
  - docs/RUNNING_TESTS.md ~line 158: the "mpi4py/mpi4jax are hard dependencies and
    must stay that way" note and the CI `--no-deps` + explicit-dep-list workaround
    are now stale — CI `fast` lane becomes plain `pip install -e ".[test]"`. Update
    text (and .github workflow if present).
- docs/numerics.md untouched; do NOT edit slurm scripts/bench.

### A5 — adversarial review + verification (opus)

- Review A1–A4 diffs against: CLAUDE.md invariants (donation, identity-hashed jit
  cache, collectives never under lax.cond, Parameters static), the resolution table
  above, and the "never silently degrade under a launcher" rule specifically.
- Run in sandbox (true no-MPI): `make test` (both precision sessions), plus
  test_backend_serial standalone-script mode.
- Confirm zero-diff behavior for mpi4jax-present environments by code inspection
  (sandbox can't run real MPI — Savio verification below).

## User verification (Savio, after A5)

- `mpirun -n 4 python tests/test_snapshot_roundtrip.py` + one forced-turbulence
  restart from an existing run dir: confirms the mpi4jax path and params.json
  compatibility are untouched.
- One `comm_backend="jax"` GPU job unchanged (backend battery in savio_manifest).

## A2 findings

Audited every `params.comm`/`params.size`/`params.rank`/`params.cart_comm`/
`left_neighbor`/`right_neighbor`/`comm_backend`/`MPI.`/`mpi4jax`/`halo_start` site in
`taranis/` (production package; `bench/`/`tests/` excluded per scope, flagged below).
`grep -rn` found sites in 8 files outside `comms.py`/`config.py`/`_mpi_compat.py`:
`run.py`, `snapshot_io.py`, `grids.py`, `diagnostics.py`, `physics/__init__.py`,
`physics/rmhd.py`, `physics/shared_physics.py` (`timestepping.py`/`__init__.py` matched
the grep only in comments — not real sites).

| Site | Verdict | Note |
|---|---|---|
| `run.py:36,59,72,158,209` `comm_backend=="jax"` branches | OK | serial takes the plain (non-shard_map) else-branch, identical to today's mpi4jax path |
| `run.py:130,146` `params.rank==0` prints | OK | serial rank is always 0 |
| `run.py:140-141` `params.size>1` bcast guard | OK (no-op) | serial size is always 1; `_NullComm.bcast` unreachable here, correct-by-construction |
| `run.py:176` `params.rank==0` progress print | OK | same as above |
| `snapshot_io.py:60` per_rank-tree-vs-`comm_backend=="jax" or size==1` guard | OK | drives off `params.size==1`, not backend name — serial already covered |
| `snapshot_io.py:66` flat-tree-vs-mpi4jax-size>1 guard | OK (no-op) | serial is always size==1, condition never true |
| `snapshot_io.py:72-78` layout selection (`jax` / `size>1` / else) | OK | serial falls into the `else` (flat, same dir as size-1 mpi4jax) |
| `snapshot_io.py:140` `_is_global_array` + `comm_backend!="jax"` guard in `save_snapshot` | OK | serial's `state.fields` is an ordinary single-device array; `_is_global_array` is False |
| `snapshot_io.py:205,266` `comm_backend=="jax"` branches in `load_snapshot` | OK | serial skips both, takes the per-rank-loop-with-size=1 path (degenerates to a single full-domain read) |
| `snapshot_io.py:216-227` `params.size`/`params.rank` in the z-slice restore loop | OK | size=1/rank=0 ⇒ nz_load=nz, z_start_l=0, one iteration, no slicing |
| `grids.py:83` `comm_backend=="jax"` in `setup_kgrids` | OK | serial skips `_kgrid_to_global`, kgrid stays local arrays |
| `grids.py:111-113` `local_z_coords` (`params.nz // params.size`, `params.rank * ...`) | OK | size=1/rank=0 ⇒ full local z range starting at 0 |
| `diagnostics.py:22` `parspec` `assert params.size == 1` | OK | serial satisfies this trivially (it's the same assertion that already allows single-process mpi4jax) |
| `physics/__init__.py` `_HALO_START_BACKENDS` | **changed** | see below |
| `physics/rmhd.py:33-40` `halo_start` (2D returns None, 3D calls `comms.halo_exchange` width=2) | OK | backend-agnostic; dispatches into `comms.halo_exchange`'s own serial branch |
| `physics/shared_physics.py:15-36` `z_derivatives` pre-issued-halo assertions | OK | `tests/test_halo_width.py::test_z_derivatives_halo_width_invariance` and `::test_halo_start_matches_bare_width2_exchange` already show bitwise invariance unconditionally (not gated on backend); reconfirmed against real serial below |
| `config.py` `Parameters.save` backfill of missing ctor keys | **fixed** | see below |

### halo_start decision: **yes**, added `"serial"` to `_HALO_START_BACKENDS`
(`taranis/physics/__init__.py`). Rationale: `comms.halo_exchange`'s serial branch is a
pure array slice (no collective, no token ordering) — pre-issuing it at the top of the
RHS instead of inside `z_derivatives` cannot change results, only *when* the slice
happens. `tests/test_halo_width.py`'s two invariance tests already establish this
generically (not backend-gated); re-verified live in the sandbox (no-MPI, real serial
backend, not the stub): built a forced 3D `Parameters`/`kgrid`, jitted
`construct_rhs(recipe)` once with `halo_start` on (default) and once with
`params.halo_start=False` forced — RHS output `jnp.array_equal` **True**; then ran a
20-step `simulate_scan` trajectory both ways — final `state.fields`
`jnp.array_equal` **True** (exact bitwise match, not just close).

### backfill decision: backfill `comm_backend` with the resolved value, not the ctor default
`Parameters.save`'s backfill loop (for ctor args missing from an old `params.json`) used
`inspect.signature(...).default` for every key, including `comm_backend` (default
`None`), so a legacy record predating this feature would gain `"comm_backend": null`
instead of documenting what actually wrote the backfilled file. Fixed in
`taranis/config.py::Parameters.save`: for keys in `_TRANSPORT_KEYS`, backfill from
`rec[k]` (this save's resolved value, e.g. `"serial"`) instead of the signature default;
every other key keeps using the signature default as before (that's intentionally "what
old code used", not "what this run uses" — `comm_backend` is the one ctor arg where the
resolved-value semantics is correct because it's excluded from the diff comparison
entirely). The "identical re-save is a no-op" invariant is untouched: `comm_backend`
stays in `_TRANSPORT_KEYS`, never compared. Verified in the sandbox: wrote a
`params.json`, stripped `"comm_backend"` to simulate a pre-feature record, re-called
`params.save()` — backfilled value is `"serial"`, not `null`.

### Fixes made
1. `taranis/physics/__init__.py`: `_HALO_START_BACKENDS = ("jax",)` → `("jax", "serial")`, with a comment citing the invariance tests.
2. `taranis/config.py`: `Parameters.save`'s backfill loop special-cases `_TRANSPORT_KEYS` to use the resolved value instead of the ctor signature default.

No other fixes were needed — every other site was already correct by construction (A1's
serial-first dispatch ordering in `comms.py` and the `_NullComm`/`cart_comm=None`/
`rank=0,size=1` semantics in `config.py` made the rest of the call sites no-ops for
serial, matching A1's handoff expectations).

### Flags for A5 / later agents
- `bench/bench_phase1.py` (out of scope, not touched): its `backend=` CLI arg and
  usage comment only document `mpi4jax|jax` (lines 6, 42, 46, 124, 162); it will still
  run under the auto-resolved default (`comm_backend=None` → `"serial"` with no MPI)
  since it just passes through to `Parameters(comm_backend=...)`, but there's no
  `backend=serial` CLI value and no serial-specific measurement path. Not broken, just
  incomplete — worth a follow-up if serial-backend benchmarking is wanted.
- `bench/probe_cuda_mpi.py`, `bench/nccl_repro.py` import `mpi4py`/`mpi4jax` at module
  scope directly (outside `_mpi_compat`), which is correct per the plan's explicit
  bench/ exception — noting it here only so nobody flags it as a violation later.
- `tests/` were out of scope for A2 by instruction, but note for A3: `_rmhd_testing.py`'s
  `ctx()`/ `fresh_params()` always route through the stub-or-real MPI resolution inside
  `bootstrap()`; A3's `bootstrap(stub=False)` (or equivalent) needs to exercise
  `_resolve_backend` truly stub-free the way this audit's sandbox script did (direct
  `taranis.Parameters(...)` construction after `sys.path` insertion, no
  `local_mpi_stub` import) to actually cover the no-MPI import path end to end.

## Verification run (sandbox, no MPI toolchain present)

- `RMHD_PRECISION=64 python -m pytest tests -q` → 92 passed, 5 skipped (expected fp32-only skips), 40.8s.
- `RMHD_PRECISION=32 python -m pytest tests -q` → 80 passed, 17 skipped (expected fp64-only/multidev skips), 26.1s.
- Standalone no-stub script (real absent-MPI import path, not `local_mpi_stub`):
  `Parameters(...)` auto-resolves `comm_backend="serial"`, `rank=0`, `size=1`,
  `cart_comm=None`, neighbors `None`.
  - Forced 3D (`elsasser`, `fshell=(1,2)`) 20-step `simulate_scan` run: finite fields,
    completes.
  - Snapshot save → `get_saved_steps` → `load_snapshot`: fields bitwise-equal
    (`jnp.array_equal`) to the in-memory end state.
  - `params.save(d)` records `"comm_backend": "serial"`; stripping that key from the
    file and re-saving backfills it back to `"serial"` (not `null`) — confirms the fix.
  - `halo_start` on vs off: RHS bitwise-equal; 20-step trajectory bitwise-equal.

## Status

- [x] A1 core
- [x] A2 audit
- [x] A3 tests
- [x] A4 packaging/docs
- [x] A5 review + sandbox verification
- [ ] Savio verification (user)

## A5 review

**Verdict: ship-with-notes.** The mpi4jax path is import-indirection only (verified line by
line); the serial path is exact size-1 semantics; the never-silently-degrade rule holds on
every branch of `_resolve_backend` I could construct. Three fixes made, all in tests/docs —
no production code changed by A5.

### Verification (sandbox, genuinely no MPI toolchain, jax 0.6.2 / py3.10 / CPU)

- `RMHD_PRECISION=64 pytest tests -q` → **103 passed, 5 skipped, 1 deselected**
- `RMHD_PRECISION=32 pytest tests -q` → **91 passed, 17 skipped, 1 deselected**
  (run in file batches to fit the sandbox's per-call time budget; counts are the sums)
- `RMHD_PRECISION=64 python tests/test_backend_serial.py` (standalone-script mode) → ALL PASS
- `ruff==0.16.1 check .` → clean
- Fresh no-stub smoke (`PYTHONPATH=repo python smoke.py`, no `local_mpi_stub`, no test
  helpers): 2D and 3D forced (`elsasser`) `Parameters` auto-resolve to `"serial"`
  (rank 0, size 1, `cart_comm=None`, neighbors `None`), `simulate` with snapshots runs to
  t=0.11, `get_saved_steps` → `load_snapshot` round-trips, `forcing_scale` is a concrete
  `(n_ou,)` array through save/load, `params.json` records `"serial"`, and
  `from_snapshot` of a record hand-edited to `"mpi4jax"` re-resolves to `"serial"`.

### Issues found

1. **(bug, fixed)** `tests/_rmhd_testing.py::mpi_size()` imported mpi4py unconditionally, so
   the *new* `RMHD_TEST_NO_STUB=1` knob crashed the whole pytest session with an
   `INTERNALERROR` inside `conftest.pytest_collection_modifyitems` (no tests ran at all).
   Fixed: `mpi_size()` returns 1 when mpi4py is genuinely absent — correct by definition,
   since `_resolve_backend` already hard-errors if a real launcher is present.
2. **(docs, fixed)** CLAUDE.md still carried "Nothing outside comms.py imports mpi4jax" and
   called `"mpi4jax"` the default. Both are now false. Rewritten to the plan's actual rule
   (`_mpi_compat.py` only, mpi4py *and* mpi4jax, `bench/`+`tests/` excepted) and to
   "auto-selected whenever mpi4py and mpi4jax import".
3. **(docs, fixed)** CLAUDE.md's serial bullet said auto-resolution happens "when
   mpi4py/mpi4jax aren't importable", omitting resolution rule 3 (mpi4py present, mpi4jax
   absent, size 1 → serial). Reworded to "when mpi4jax isn't importable and the
   real/launcher world size is 1".
4. **(docs, fixed)** `bootstrap()`'s docstring advertised `RMHD_TEST_NO_STUB=1` beside
   `stub=False` without saying the env form is process-wide and therefore reaches conftest:
   a whole session under it leaves every `comm_backend="mpi4jax"/"jax"` test with no
   transport (8 honest ImportError failures in `test_backend_serial.py` alone). Documented
   as a single-purpose subprocess knob, not a suite mode.

### Checklist results (no change needed)

- **Zero behavior change for mpi4jax/jax with MPI present.** `comms.py`: the import moved to
  `_mpi_compat` (same object identities); the only new statement on the mpi4jax path is a
  static `if params.comm_backend == "serial"` python branch ahead of it, and the
  `or params.cart_comm is None` disjunct added to the allreduces is redundant-but-inert when
  a cart comm exists. No traced op, no argument, no branch order inside the mpi4jax/jax
  bodies changed. `config.py`: the one new runtime call on the MPI path is a local
  `MPI.COMM_WORLD.Get_size()` (not collective, mpi4py is already initialized at import).
  Every `Parameters` attribute an MPI run ends up with is bit-identical to before,
  `_init_args["comm_backend"]` included ("mpi4jax" then, "mpi4jax" now).
- **Resolution table.** Traced every branch incl. explicit-backend bypasses. Empty/garbage
  env values fall through `_env_int`; `*_SIZE=1` is definitive-and-harmless; explicit
  `"serial"` is *not* a bypass (it is checked against the launcher size, with or without
  mpi4py); explicit `"mpi4jax"`/`"jax"` still hit the mismatched-MPI guard, which runs
  before the `requested is not None` block. `from_snapshot` re-resolves rather than trusting
  the record, and overrides win because `args.pop` precedes `args.update(overrides)`.
- **CLAUDE.md invariants.** `Parameters` still static/not a pytree; dispatch is plain python
  everywhere (no `lax.cond` around a collective); tests build fresh states via `make_state`
  and never mutate `ctx()` results (`test_backend_serial` only ever reads them, and uses
  `fresh_params` where it saves); reads still use a bare `StandardCheckpointHandler`
  (`snapshot_io.py` untouched); `forcing_scale` concrete through the serial save/load path
  (checked live).
- **A1's "jax needs only mpi4py" deviation: agreed.** The `"jax"` paths use `params.comm`
  (mpi4py: `bcast`, `Split_type`), `jax.lax.ppermute/psum/pmax`, and orbax — grep confirms
  no mpi4jax symbol is reachable from `init_backend`, `shard_call`, `to_global/to_local`, or
  any `snapshot_io` jax branch. `_resolve_backend` requires only `HAVE_MPI4PY` for `"jax"`,
  which is exactly right.
- **A3's config-copies-by-value seam: works, acceptably fragile.** `config.py` does
  `from ._mpi_compat import HAVE_MPI4PY, ...`, so `_resolve_backend` reads *config's* module
  globals and patching `taranis.config.HAVE_MPI4PY` is observed (patching `_mpi_compat`
  would not be). The test asserts on `_resolve_backend`'s return value / exception type, so
  it really does exercise the resolution logic, not a mock of it. The coupling is documented
  at the top of the test module; anyone switching config to `from . import _mpi_compat` must
  update the test.
- **`_NullComm` completeness.** The complete set of `params.comm` call sites in `taranis/`
  is `Get_rank`/`Get_size` (config), `Create_cart` (config, inside the non-serial branch),
  `bcast` (config.save and run._start_snapshots, both under `size>1`), `Split_type`
  (comms._local_device_ids, `"jax"` only). `_NullComm` covers the three reachable ones; no
  latent AttributeError. `snapshot_io.py` never touches `params.comm`.
- **halo_start with `"serial"`.** `rmhd.halo_start` pre-issues width=2, `z_derivatives`
  needs `w >= 2` and `recv_left.shape[1] == recv_right.shape[1]`; the serial branch returns
  two width-`width` slabs of the same shape, so both assertions hold. Re-confirmed by
  `test_halo_width.py` passing in both precision sessions.
- **params.json compatibility.** Old record with `"comm_backend": "mpi4jax"` loads and
  re-saves as a no-op (transport key excluded from the diff); a pre-feature record missing
  the key backfills the *resolved* value and rewrites once; every other key is untouched
  (asserted by `test_backfill_of_missing_comm_backend_uses_resolved_value_not_null`).

### Judgment calls (left as designed, flagged for the record)

- **`srun -n 4` without an MPI PMI/PMIx plugin** sets only `SLURM_NTASKS`, which the plan
  deliberately classes as non-definitive → warn + serial → 4 processes each running the full
  domain. This is the plan's agreed trade (a plain `python` in a batch script is
  indistinguishable), and Savio's documented launchers (`mpirun`, `srun --mpi=pmi2/pmix`) all
  set a definitive variable. If it ever bites, `SLURM_PROCID`+`SLURM_STEP_NUM_TASKS` are the
  next-best discriminators.
- **A broken-but-installed mpi4jax now degrades silently at size 1.** `_mpi_compat` catches
  `Exception` (correctly — a bad MPI runtime raises more than ImportError), so a single-rank
  Savio job with an ABI-mismatched mpi4jax quietly runs `"serial"` instead of failing loudly.
  Semantically correct at size 1, and multi-rank still hard-errors naming mpi4jax. Plan rule
  3 chose this; noting it because "my job got slower/quieter" is the symptom.
- **"NOT bitwise-identical to size-1 mpi4jax" is an unverified (conservative) claim.** The
  sandbox stub *is* bitwise-identical, which is why `test_backend_serial` compares exactly;
  the real-mpi4jax delta is asserted from the token/fusion argument, not measured. Nothing
  depends on it being true — it only tells future test authors to use tolerances.
- **`params.json` keeps the backend recorded by the run that created the directory.** A
  serial restart of an mpi4jax run leaves `"comm_backend": "mpi4jax"` on disk (transport keys
  are never compared, so the file is not rewritten). Correct per the plan; just don't read
  the field as "what ran last".
- **`bench/bench_phase1.py` has no `backend=serial` CLI value** (A2's flag). Still true,
  still out of scope, still not broken (it inherits the auto-resolved default).
