# Phase R — transport out of `Parameters.__init__`

Base: `3073df4`, worktree `/private/tmp/taranis-wt-R`, branch `refactor/R`.
Files touched: `taranis/comms.py`, `taranis/config.py`, `tests/_rmhd_testing.py`
(`fake_ranked_params`), `tests/test_backend_jax.py` (the spoof + one import line),
`tests/test_params.py` (one new test + a header line). `taranis/_mpi_compat.py` is
unchanged — nothing there needed to move.

## What changed

### `comms.Runtime`

`@dataclasses.dataclass(frozen=True, eq=False)` with `backend, comm, rank, size,
cart_comm, left_neighbor, right_neighbor`. `eq=False` keeps it identity-hashed and
identity-compared like `Parameters` (a dataclass only synthesizes `__hash__`/`__eq__`
when `eq=True`, and neither an mpi4py communicator nor `_NullComm` has useful value
equality). It is never a pytree and never traced.

`Runtime.resolve(comm_backend=None, *, dims, nz)` performs, in the constructor's previous
order: the `COMM_BACKENDS` membership check, `_resolve_backend`, `MPI.COMM_WORLD` (or
`_NullComm`) plus rank/size, the `TARANIS_PRECISION` allgather, the `"jax"`-needs-`dims==3`
rejection, `nz % size`, the cartesian communicator + `Shift` (dims 3, non-serial), then
`init_backend`. Every message is byte-identical; the two that named `self.<attr>` now name
the local (`rank`, `size`, `nz`) with the same rendered text.

`init_backend(runtime, nz)` and `_local_device_ids(runtime)` take the runtime instead of a
half-built `Parameters` — the only reason they ever took `params` was rank/size/comm/nz.
`init_backend` is called from `Runtime.resolve` only (it had no other caller in the tree).

### `_resolve_backend`

Moved to `comms.py` verbatim, and re-exported from `config.py` as
`_resolve_backend = comms._resolve_backend` so `taranis.config._resolve_backend(...)`
still resolves. **It reads `HAVE_MPI4PY`/`HAVE_MPI4JAX`/`MPI` off the `taranis.config`
module** (`comms._mpi_names()`, a function-local `from . import config`, the same
lazy-import idiom `shard_call` already uses for `grids`). That is deliberate and is what
the plan's "keep the resolver reading the flags from wherever the tests patch them"
requires: `tests/test_backend_serial.py::_patched` substitutes those three names on
`taranis.config` — and not only around direct `cfg._resolve_backend(...)` calls, but also
around a full `Parameters.from_snapshot(d)` construction
(`test_from_snapshot_drops_recorded_backend_and_reresolves`), so the constructor's
resolution path has to see them too. Reading `_mpi_compat`'s or `comms`'s own copies would
silently ignore the patch and that test would fail. `Runtime.resolve` takes
`MPI.COMM_WORLD` from the same place, for the same reason. `config.py` keeps the
by-value import of the three names (`HAVE_MPI4JAX` now only for the resolver to read,
hence a `# noqa: F401`); `_NullComm`/`launcher_world_size` moved with the resolver.

### `Parameters`

- New last ctor argument `runtime=None`. `None` resolves one; an explicit `Runtime` is
  used as is, and `comm_backend` must then be `None` or equal to `runtime.backend`
  (`ValueError` naming both). `runtime` is popped from `_init_args` immediately after the
  `locals()` capture, so it never reaches `params.json` and never takes part in the
  differing-record check; `comm_backend` is still recorded resolved.
- `comm_backend, comm, rank, size, cart_comm, left_neighbor, right_neighbor` are read-only
  properties forwarding to `self.runtime`. No consumer in `taranis/`, `tests/`, `bench/`
  or `examples/` changed (`grep -rnE "params\.(comm|rank|size|cart_comm|left_neighbor|
  right_neighbor)\b"` — 40-odd call sites, all reads). Assignment now raises
  `AttributeError: property 'rank' of 'Parameters' object has no setter`, which is why the
  two rank-spoofing helpers had to change.
- `_validate_compat(self)` holds the whole matrix — `"jax"` × dims, `nz % size`, the
  2D-on-many-ranks warning, `z_spectral` × (dims, size, backend), particles × (eqtype,
  size, backend) and the 3D `B0 == 1` rejection — in that (unchanged) relative order, with
  byte-identical messages. It is called once, as the last statement of `__init__`, because
  the particles half needs the normalized `self.particles`.

### Ordering deltas (all deliberate, none reachable from the suite)

`_validate_compat` sits at the end of `__init__` and `Runtime.resolve` at the start, so
three relative orders moved. In each case both branches raise; only *which* message comes
first could differ:

1. `nz % size` and `"jax"`-needs-3D are checked **twice** — once in `Runtime.resolve` (so
   a direct `resolve()` call and the ordinary construction still fail at the same point as
   before, ahead of `init_backend`) and once in `_validate_compat` (so a `Parameters` built
   on a *shared* `Runtime`, which does not re-run `resolve`, is still checked — plan §9
   decision 2: `nz` is validated against `runtime.size`, not against the `nz` the runtime
   was built for). Same message text in both places.
2. The three `z_spectral` rejections now run *after* `init_backend` instead of just before
   it. Only `z_spectral=True` + `comm_backend="jax"` is affected, and only at `size > 1`,
   where `init_backend` would now bring up `jax.distributed` before the config is refused.
   Untestable here (the 4-fake-device path is `size == 1`); flagged for the Savio run.
   `tests/test_params.py::test_runtime_is_injectable_but_never_recorded` pins the shared-runtime
   half of this: a `dataclasses.replace(rt, size=3)` runtime with `nz=8` is refused with the
   byte-identical "must be divisible by the number of MPI ranks (3)".
3. The particles compatibility rejection now runs *after* `particles.state.normalize_config`
   instead of before it, so a config that is both malformed and incompatible (e.g.
   `eqtype="GDI"` + a bad ensemble dict) reports the malformed-dict error first.
   `tests/test_particles_config.py`'s `eqtype != RMHD` case passes a well-formed dict and is
   unaffected.

`comms.py`'s dispatch functions read `params.runtime` (`rt = params.runtime`, then
`rt.backend`/`rt.cart_comm`/…) rather than the forwarding properties — one convention
throughout the file, as the plan asks. `_unknown_backend` takes the backend string now.

## Gates

Machine: `Alfreds-MacBook-Pro.local`, jax 0.10.0, CPU, python 3.11.5. mpi4py **and**
mpi4jax are importable here, so `bootstrap()` installs no stub and no fake devices:
`comm_backend` auto-resolves to `mpi4jax` at size 1 and the `multidev` tier skips, exactly
as on the base. Other agents were running tests concurrently — no timing is quoted.

| gate | result |
|---|---|
| `make test`, fp64 session | **248 passed, 23 skipped, 1 deselected**, exit 0 |
| `make test`, fp32 session | **225 passed, 46 skipped, 1 deselected**, exit 0 |
| the same fp64 session on the base commit (`git archive 3073df4`, same machine) | **247 passed, 23 skipped, 1 deselected** — exactly the new `test_params` test more, and the skip list is identical line for line (only `test_backend_jax.py`'s numbers shift by 1, from the added `import dataclasses`) |
| `tests/test_refactor_reference.py` (fields + HLO histograms) | passed in both sessions, no regeneration |
| gate 6 / 6b / 6c (`tests/test_particles_coupled.py`) | passed in both sessions |
| `tests/test_forcing_spinup.py`, `tests/test_precision_fp32.py` | passed |
| `tests/test_hoist_propagator.py` (incl. `test_unhoisted_graph_stays_memory_light`) | passed |
| `tests/test_params.py` (15), `test_backend_serial.py`, `test_backend_jax.py`, `test_infra.py` | passed in both sessions |
| `params.json` byte-identity vs the base commit | identical apart from `_created` (869 bytes both) |
| `ruff check .` | All checks passed |

No reference was regenerated, no tolerance touched, no test skipped that the base does not
skip. The probe (§0.4) is not a Phase R gate — this phase compiles nothing new.

Subprocess caveat, this machine only: `taranis` is `pip install -e`-ed against
`/Users/alfy/code/taranis`, so `import taranis` inside a *subprocess* (or in script mode,
`python tests/test_x.py`, where `sys.path[0]` is `tests/`) resolves to the shared main tree,
not to the worktree — `bootstrap()` only prepends the repo root when `find_spec("taranis")`
comes back empty. `python -m pytest` from the worktree root is unaffected (cwd wins), which
is what `make test` runs. To make the three subprocess-spawning files actually exercise this
branch they were re-run as
`PYTHONPATH=/private/tmp/taranis-wt-R python -m pytest tests/test_backend_serial.py
tests/test_precision_fp32.py tests/test_precision_dtypes.py`: **15 passed / 6 skipped** at
fp64 and **21 passed** at fp32. `PYTHONPATH=… python tests/test_params.py` (script mode) is
`ALL PASS`. This is an environment property of running from a worktree, not a code change —
but it applies to every phase branch's `make test` on this laptop.

### `params.json` byte check

A 3D `z_spectral` + particles + forcing configuration (tuple args, `eqpars`, a particles
dict) was saved by a `git archive` copy of the base commit `3073df4` and by this branch;
the two files are 869 bytes each and identical once `_created` is dropped — same keys,
same order, same values.

### Pre-existing failure, not this phase's

`RMHD_TEST_FORCE_STUB=1 pytest tests/test_backend_jax.py::test_same_seed_run_matches_serial_reference`
fails on this branch **and on the base commit**, identically: `forcing_state bit-identical
across backends (max|diff| 1.42817e-13)` on both, plus the paired `final time matches`
check. That mode (forcing the MPI stub + 4 fake devices on a machine that has real mpi4py)
is not what `make test` runs here — under `make test` those tests skip for lack of devices,
on the base and on this branch alike. Worth a look by whoever owns the jax backend, but it
predates Phase R and is not caused by it.

## Noticed, out of scope

- A `Runtime` is still created per `Parameters` unless the caller passes one, so the
  "never freed cartesian communicator per `Parameters`" cost is now *avoidable*, not
  avoided. Nothing in the tree passes `runtime=` yet; the parameter-scan users are
  notebooks and `bench/`.
- `Runtime` is reachable only as `taranis.comms.Runtime` — it is not in
  `taranis/__init__.py`'s `__all__`. If the sweep wants to advertise it, that is a
  one-line addition.
- `Parameters._init_args` still comes from `locals()` (plan §7 keeps it), which is why
  `runtime` has to be popped by name rather than simply not captured.

## CLAUDE.md sentences that should change (sweep)

In **"Parameters / physics registry"**, after the "`Parameters` (`config.py`) is **not a
pytree**" paragraph:

> Transport is not built by the constructor: `comms.Runtime` (a frozen, identity-hashed
> dataclass — `backend, comm, rank, size, cart_comm, left_neighbor, right_neighbor`) is
> built by `comms.Runtime.resolve(comm_backend=None, *, dims, nz)`, which resolves the
> backend, takes `MPI.COMM_WORLD`, checks every rank reads the same `TARANIS_PRECISION`,
> creates the z cartesian communicator and brings the backend up (`jax.distributed` for
> `"jax"`). `Parameters(..., runtime=None)` resolves one; passing an explicit `Runtime`
> reuses it — one set of communicators for a whole parameter scan — and then
> `comm_backend` must be `None` or that runtime's own backend. `params.runtime` holds it
> and `comm_backend/comm/rank/size/cart_comm/left_neighbor/right_neighbor` are read-only
> properties forwarding to it: reads are unchanged everywhere, but **assigning
> `params.rank` now raises** — spoof a rank with `dataclasses.replace(p.runtime, ...)`
> (`tests/_rmhd_testing.py::fake_ranked_params`). `runtime` is never recorded in
> `params.json`. The compatibility matrix (backend × dims × size, `z_spectral`,
> particles) lives in one place, `Parameters._validate_compat`.

In the **`comm_backend` bullets**, the `"jax"` bullet's sentence

> Constructing `Parameters(comm_backend="jax")` brings up `jax.distributed` — must be the
> first jax device work in the process

should become

> `comms.Runtime.resolve` brings up `jax.distributed` — so the first
> `Runtime.resolve`/`Parameters(comm_backend="jax")` in a process must precede any jax
> device work

and the `"serial"` bullet's `comm_backend=None` auto-resolution sentence can point at
`comms._resolve_backend` (it now lives in `comms.py`, re-exported from `config.py`).

`docs/SAVIO_GPU_SETUP.md` states the same ordering constraint in terms of "the first
`Parameters(comm_backend='jax')`"; that phrasing is still true (the constructor resolves a
runtime), so it needs no edit — but if the sweep wants it precise, "the first
`comms.Runtime.resolve`, which `Parameters(comm_backend='jax')` calls" is the exact
statement. Its `comms._local_device_ids` references are still correct (the function now
takes a `Runtime` instead of a `Parameters`).
