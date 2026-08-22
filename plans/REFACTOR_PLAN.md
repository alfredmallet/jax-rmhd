# REFACTOR_PLAN — structural items 1–4: runtime split, uniform carry, typed linear operator, typed equation interface

Written 2026-08-22 from the assessment of an outside agent's six refactoring ideas (this
session). Items 1–4 were accepted in the forms below; item 5 (ETDRK) and item 6 (parallel
z-FFT, halo overlap) and the parts of 1–4 that were rejected are listed in §7 so nobody
re-imports them. Execution per the standing flow (§8): Fable overseer, **opus implementers in
parallel worktrees**, fresh-Fable adversarial review per phase, overseer owns CLAUDE.md and
this file's result notes.

The four phases are **behaviour-preserving moves**: every gate below is bitwise, verified
against references recorded BEFORE the work starts (Phase 0). Nothing here changes a number.
What it buys: the transport side effect out of `Parameters.__init__`, one run-loop body
instead of four, one typed linear-operator slot on `K_Grids` instead of six `Optional`s,
named gradients and statically filtered RHS terms. Net source change is expected to be
negative.

## 0. Principles

Inherited from `plans/MEMORY_PERF_PLAN.md` §0, restated where they bite here:

1. **No new user-facing knobs.** The one new constructor argument (`runtime=`, §4) is an
   injection point for an object the constructor already builds, not a switch.
2. **Comments say what the code does, succinctly.** No history, no "was X, now Y". The
   reviewer checks; a phase is not done while its diff carries narrative comments. Existing
   long comments in touched files are not rewritten wholesale.
3. **Bitwise is stated per phase and verified, never assumed.** Four references pin solver
   bits: `tests/data/particles_gate6_reference_fp{64,32}.npz` (2D RMHD, five driver
   paths), `forcing_spinup_reference.npz`, `precision_fp32_reference.npz`, and the new
   Phase-0 `refactor_reference_fp{64,32}.npz` (3D FD-z, z_spectral separable and putzer2,
   GDI IF and IMEX, adaptive `cfl_every=1`, rk44, imexcb3f, hoist on/off — the paths gate 6
   does not cover). Every phase must leave all four green **with no regeneration**. A
   drift is a bug or a §9 decision for Alfred, never a reason to regenerate — and never a
   reason to restructure floating-point op order to make a comparator clean (bitwise gates
   are evidence, not a goal; MEMORY_PERF_PLAN §0.3).
4. **No performance impact — speed or memory — and it is gated, not asserted.** Memory:
   `tests/test_hoist_propagator.py::test_unhoisted_graph_stays_memory_light` stays green,
   and the phases that touch compiled graphs (C, L) compare the probe
   (`bench/memory_probe.py`, laptop profile, both precisions) against the Phase-0 JSON to
   ≤ 0.05 u per case. Speed: wall-clock on a shared laptop is too noisy to be a gate, so
   the gate is the compiled graph itself — the Phase-0 reference records, per config, the
   opcode histogram (count per HLO op type, plus total instruction and fusion counts) of
   the optimized HLO of the jitted `block_of_steps`, and `tests/test_refactor_reference.py`
   asserts it unchanged. Bitwise-identical output AND an identical op histogram means the
   same compute, hence the same step time. The one allowed difference is Phase G's: in
   configs with an inactive term, FEWER `add`/`broadcast`/`constant` ops and nothing
   else (G records the before/after counts in its notes; the reference is then
   regenerated for the histogram only, fields untouched, as a dated §10 note). After the
   final merge one quiet-machine ms/step run of the probe against the Phase-0 JSON is the
   sanity check (expected within noise, ±3%; quoted as measured against that named
   reference, never chained). Python-side cost (properties on `Parameters`, NamedTuple
   construction, trace-time term filtering) is paid once per trace, not per step.
5. **Disjoint file ownership.** Four agents edit one tree's history concurrently; §6 is the
   ownership table and it is binding. A needed edit to a file you do not own is a message to
   the overseer, not an edit. Shared docs (CLAUDE.md, docs/*.md, this file) are the
   overseer's, via the close-out sweep; agents write their doc text to
   `plans/refactor_notes/<phase>.md` in their own branch.
6. **Public contracts hold.** `block_of_steps`'s documented shape (state in → state out with
   particles off; `((state, pstate), ys)` on), every `params.<attr>` read, the stepper
   signature, `EquationRecipe(...)` with bare callables, `grads` being a tuple, the
   `params.json` record. Tests and benches outside a phase's ownership must pass unedited.

## 1. Phase 0 — references and the restart fix (sequential, one agent, on `main`)

Runs BEFORE the fan-out; the fan-out base is the commit that closes this phase.

**0a. Record the refactor reference.** `tests/_gen_refactor_reference.py` +
`tests/data/refactor_reference_fp{64,32}.npz` (force-added; `tests/data` is gitignored) +
`tests/test_refactor_reference.py`, on the gate-6 pattern: the generator DEFINES the configs,
the test imports them, so the two cannot drift. Driven through the jitted public
`block_of_steps` (as `tests/test_hoist_propagator.py` does — no snapshot dirs, fast), 6
steps, small grids (16²×8 in 3D, 32² in 2D), the non-degenerate multi-mode IC from
`test_hoist_propagator._ic`, forcing on where the physics allows. Configs, each at its
named scheme and dt mode:

| key | config | what it pins |
|---|---|---|
| `fd_fixed_lsrk54` | 3D FD-z, fixed dt, elsasser forcing | diagonal backend, FDLinearTerm + halo |
| `fd_cfl2_lsrk33` | 3D FD-z, adaptive, `cfl_every=2` | `_cfl_block` in 3D |
| `fd_adapt_rk44` | 3D FD-z, adaptive `cfl_every=1`, rk44 | the unhoisted adaptive path, rk44 stage structure |
| `sep_fixed_lsrk54` | z_spectral, ν = η, fixed dt | separable backend, hoisted |
| `put_cfl2_lsrk54` | z_spectral, ν ≠ η, `cfl_every=2` | putzer2 hoisted |
| `put_cfl2_lsrk54_nohoist` | same, `hoist_propagator=False` | the memory-light graph |
| `put_adapt_lsrk33_unrolled` | z_spectral, ν ≠ η, adaptive, `lsrk_scan=False` | unrolled stage loop |
| `rmhd2d_adapt_lsrk33_mom` | 2D, adaptive `cfl_every=1`, momentum forcing | 2D adaptive (gate 6 has fixed and cfl=2 only) |
| `rmhd2d_fixed_imexcb3f` | 2D, fixed dt, imexcb3f | the [3R] unrolled IMEX stepper |
| `gdi2d_fixed_imexcb3e` | GDI 2D, fixed dt | IMEX on a dense 2×2 L |
| `gdi2d_fixed_lsrk33` | GDI 2D, fixed dt, IF | putzer2 under IF, `_lin_dt_safety` not engaged |
| `gdi3d_fixed_imexcb3e` | GDI 3D z_spectral, fixed dt, 16²×8 | z-extent L, `D_par·kz²` |

The npz stores `fields` and `t` per key, and — the speed gate, §0.4 — the opcode histogram
of the optimized HLO of the jitted `block_of_steps` for that config
(`jitted.lower(...).compile().as_text()`, instructions counted by opcode inside and
outside fusions, plus total instruction and fusion counts; `bench/hlo_audit.py` already
parses HLO text — reuse its approach), stored as a JSON sidecar
`tests/data/refactor_reference_hlo_fp{64,32}.json`. The test asserts `np.array_equal` at fp64 AND fp32
on the recording machine (jax 0.10.0, the M1 laptop), under gate 6's host convention:
the npz records `hostname`/`jax_version`/`platform`/`python_version` and the test
print-skips (not fails) when any of the four differs — copy that logic from
`tests/test_particles_coupled.py` (around line 953), do not invent a new one. Both precisions are
separate processes (`TARANIS_PRECISION` is read at import).

**0b. Probe baseline.** `python bench/memory_probe.py --profile laptop --precision 32
--out bench/memory_probe_refactor_base.json` and the same with `--precision 64 --out
bench/memory_probe_refactor_base_fp64.json` (committed; the C and L gates compare
`total_u` per case against these, not against the older `*_laptop_baseline*.json`,
which predate MEMORY_PERF's phases). Quiet machine, no other agents running — the
`total_u` column is deterministic but the ms/step columns are not, and nobody quotes
the latter from this file.

**0c. The forced-restart fix.** `run._refresh_forcing_scale` recomputes `forcing_scale` at
dt = 0 on every `simulate`/`simulate_scan` entry, which overwrites a checkpoint's stored
(lagged, correct) scale — so a forced restart at the default `forcing_norm_per_step=True` is
not bitwise (filed in `plans/TESTPART_PLAN.md` §6 gate 6c, 2026-08-18). Rule: **refresh iff
the stored scale is all zeros** (a fresh `initialize`, a repaired legacy snapshot); a
nonzero scale is the one the uninterrupted run would use next and is kept. Concrete check
outside jit (`bool(jnp.all(state.forcing_scale == 0))`; the jax backend's replicated
global array is addressable, so this holds there too). Consequences, all intended: a
restart from a snapshot reproduces the uninterrupted run bitwise; a second `simulate_scan`
call on a returned state continues it bitwise (today it re-kicks the scale); a run from
`initialize` is unchanged (zeros → refreshed as before, so gate 6 and the spinup reference
are untouched — verify, do not assume).
Gates: a new `tests/test_forcing_norm_per_step.py::test_restart_is_bitwise_at_default_norm`
(forced 2D `simulate_scan` with snapshots; reload the middle step; continue; fields and
`forcing_state` bitwise against the uninterrupted run, both precisions), and
`tests/test_particles_coupled.py` gate 6c run ALSO at the default `forcing_norm_per_step=True`
(its `_RESTART_KWARGS` currently forces it off — keep that cell, add the default one). The
CLAUDE.md line "Restart is bitwise ONLY with `forcing_norm_per_step=False` or
`forcing=False`" and `run.py`'s "(fix filed separately)" comment go in the close-out sweep.

Order inside Phase 0: 0a and 0b on the untouched tree, THEN 0c (0c changes no reference —
none of them restarts — but recording first is the rule). Commit 0a+0b, then 0c separately.

## 2. Phase L — the linear operator as a typed pytree

**Today.** A recipe's `linear_matrix_func` returns a 4-d array, a 5-d array or a
`SeparableL`; `propagators.linear_fields` shape-sniffs it into six `Optional` `K_Grids`
slots (`lin_L/lin_m/lin_s/lin_dperp/lin_dz/lin_kz`); `get_propagator(kgrid, params)` sniffs
them back into a Python class (`DiagonalPropagator`/`Putzer2Propagator`/
`SeparablePropagator`/`IdentityPropagator`) inside every step. Shape dispatch lives in three
places in `propagators.py` (`linear_fields`, `get_propagator`, `dense_operator`), and
`kgrid_specs`/`_kgrid_to_global` special-case the slots by name.

**Target.** One slot, `K_Grids.lin`, holding an OPERATOR that is itself a pytree
(NamedTuple of arrays with methods, exactly as `Putzer2Exp`/`SeparableExp` already are):

```
class IdentityOperator(NamedTuple): ...                 # no fields
class DiagonalOperator(NamedTuple): L
class Putzer2Operator(NamedTuple): L; m; s              # m, s precomputed at setup (putzer2_precompute)
class SeparableL(NamedTuple): dperp; dz; kz             # UNCHANGED NAME: it is what rmhd.linear_matrix returns
```
Each carries the current propagator methods unchanged in arithmetic and op order —
`scaled(factor)`, `exp_op(tau)`, `apply_exp(arr, tau)`, `solve_shifted(arr, a)`,
`apply_L(arr)` — plus `dense()` (the former `dense_operator(kgrid)`; `IdentityOperator.dense`
raises) and `hoistable` as a property (so `op.hoistable is False/True` keeps passing).
`SeparableL` GAINS the methods of `SeparablePropagator`, which is deleted, so
`physics/rmhd.py` (Phase G's file) is not touched. `propagators.build(L, params) -> operator`
is today's `linear_fields` (every validation and error message unchanged — tests match
substrings), returning the operator instead of a dict; `grids._attach_linear_operator` does
`kgrid._replace(lin=build(L, params))`, and `IdentityOperator()` when the recipe has no
`linear_matrix_func` — **`kgrid.lin` is always populated after `setup_kgrids`**, never
`None`. `get_propagator` is deleted; `timestepping.py`'s four call sites read `kgrid.lin`
(`stage_exp_ops` line 22, `rk_advance` 51, `lsrk_advance` 92, `imex2r_advance` 206,
`imex3r_advance` 274). `putzer2_precompute` stays. `kgrid_specs`/`_kgrid_to_global` handle
`lin` with `jax.tree.map` (every leaf replicated, `P()`; the z-extent
`NotImplementedError` for `comm_backend="jax"` stays exactly where it is in `build`).

Two traps: an empty NamedTuple is falsy — never test `kgrid.lin` for truthiness; and the
operator is a tuple — never `jnp.asarray` it or index it positionally outside its own
methods.

**Bitwise expectation: every gate (§0.3), plus memory (§0.4).** The arrays and the op
order are identical; only Python containers change. The one thing that moves is the
flattening order of the jitted `kgrid` argument, which should not change XLA's fusion — if
the fp64 references show drift anyway, characterize it (which config, how many elements,
what magnitude) and raise it; do not reorder fields to chase it.

**Consumers to update (all owned by L):** `tests/test_linear_propagator.py` (lin_L/lin_m/
lin_s reads, `linear_fields` → `build`, `Putzer2Propagator(*putzer2_precompute(..))` →
`Putzer2Operator(*...)`), `tests/test_separable_propagator.py` (backend-selection
assertions become `isinstance(kgrid.lin, ...)`, `dense_operator` → `.dense()`),
`tests/test_gdi_linear.py` (`kgrid.lin_m[...]` → `kgrid.lin.m[...]`, `lin_L.shape` →
`lin.L.shape`), `tests/test_imex.py` (two `kgrid.lin_L` reads → `kgrid.lin.L`),
`tests/test_hoist_propagator.py` (`get_propagator(kgrid, params)` → `kgrid.lin`),
`bench/zspectral_profile.py` (the `_replace(lin_L=None, ...)._replace(**linear_fields(..))`
dance → `_replace(lin=build(L, p))`, and its `get_propagator` import).
`taranis/particles/fields.py::_psi_linear_diagonal` calls `rmhd.linear_matrix` directly and
dispatches on `isinstance(L, SeparableL)` / `L.ndim` — unchanged, verify.

**New/extended tests (L):** in `tests/test_linear_propagator.py`: (i) `kgrid.lin` is a pytree
whose leaves are exactly the former slots' arrays, bitwise, for all four backends (2D, FD-z,
z_spectral ν = η, ν ≠ η); (ii) `jax.jit` of a function taking `kgrid` and applying
`kgrid.lin.apply_exp` traces and matches the eager call bitwise (the pytree rides through jit);
(iii) `IdentityOperator` is what a recipe without `linear_matrix_func` gets (use
`test_imex.py`'s toy-recipe idiom, `linear_matrix_func=None`).

## 3. Phase G — typed equation interface

**Today.** `grads` is an anonymous tuple coupled positionally between `grad_func` and every
term func (`rmhd.set_timestep` reads `grads[0]`, `grads[1]`); `FDLinearTerm` and
`ForcingTerm` return `zeros_like(state.fields)` when inactive and `construct_rhs` adds the
zeros.

**Target, three moves, all in `physics/`:**

1. **Named gradients.** `rmhd.RMHDGrads(NamedTuple): gphi, gpsi, gvort, gjpar` and
   `gdi.GDIGrads(NamedTuple): gphi, gN, gvort`, in the EXISTING `grad_func` order;
   `grad()` returns `RMHDGrads(*grad_fields(...))`. Term funcs and `set_timestep` read by
   name. A NamedTuple IS a tuple, so `grads[0]`, unpacking, `run._block_dt`'s and
   `estimate_good_nblock`'s `set_timestep(grad(...), params)`, and the particles package's
   consumers all keep working unedited — the CLAUDE.md rule "`grads` is a TUPLE" stays
   true and gets names.
2. **Static term filtering.** In `physics/__init__.py`:
   ```
   class Term(NamedTuple):
       func: Callable                          # (state, grads, kgrid, params, halo) -> rhs
       active: Callable = lambda params: True  # static, evaluated at trace time
   ```
   `EquationRecipe.term_funcs` keeps its NAME and accepts bare callables (wrapped as
   `Term(func)` by `construct_rhs`) — `tests/test_imex.py`'s toy recipes (owned by L) must
   pass unedited. The registry lists `Term(rmhd.NonlinearTerm)`,
   `Term(rmhd.FDLinearTerm, active=rmhd.fd_linear_active)`,
   `Term(rmhd.ForcingTerm, active=rmhd.forcing_active)`; `construct_rhs`'s `rhs` sums only
   the terms whose `active(params)` is true (params is static, so this is trace-time
   Python), and raises a clear error at trace time if none is. The predicates are the single
   source of truth: the term funcs' early returns become `if not fd_linear_active(params):
   return zeros_like(...)` etc. — kept, because direct callers exist
   (`particles/fields.py::_forcing_ez` calls `ForcingTerm`), but no longer reached from the
   RHS.
3. Nothing else. `nfields`, the 5-positional-arg term contract, `halo`, `forcing_scale_func`
   and `halo_start_func` are unchanged.

**Bitwise expectation: every gate.** (1) is pure Python. (2) removes an `x + 0` per inactive
term; XLA's algebraic simplifier already folds `add(x, broadcast(0))`, and where it did not,
the only representable difference is the sign of a zero, which `np.array_equal` treats as
equal. Gate 6's `scan_unforced` config and the Phase-0 2D/FD-z configs are the direct
witnesses (2D has both inactive terms; FD-z has one). If a reference drifts beyond
signed zeros, raise it.

**New test (G): `tests/test_equation_interface.py`** — `grads` has the documented names and
order and each entry is a real `(2, nz, nx, ny)` array (both equations); `construct_rhs` calls
only active terms (a counting `Term` in a toy recipe); for a 2D forced RMHD config the RHS
with the inactive terms filtered equals the all-terms sum bitwise; a bare callable in
`term_funcs` still works; `Term.active` is evaluated once per trace, not per step (a
counter across a jitted `block_of_steps`).

**Files (G):** `taranis/physics/__init__.py`, `taranis/physics/rmhd.py`,
`taranis/physics/gdi.py`, the new test. Not `shared_physics.py` (`grad_fields` keeps
returning a plain tuple; the recipe names it), not `particles/`, not `run.py`.

## 4. Phase R — transport out of `Parameters.__init__`

**Today.** `Parameters.__init__` resolves the backend, takes `MPI.COMM_WORLD`, allgathers
the precision, creates a cartesian communicator (a new one per `Parameters` — never freed,
which a long parameter scan notices), and calls `comms.init_backend(self)`, which may run
`jax.distributed.initialize` on a half-built object. Ten compatibility raises are scattered
through the constructor. `_init_args` is captured with `locals()`.

**Target.**

- `comms.Runtime`: a frozen dataclass (NOT a pytree, never traced, identity-hashed like
  `Parameters`) with `backend, comm, rank, size, cart_comm, left_neighbor, right_neighbor`,
  built by `comms.Runtime.resolve(comm_backend=None, *, dims, nz) -> Runtime`, which does —
  in this order, with every message unchanged — what the constructor does today:
  `_resolve_backend` (moves from `config.py` to `comms.py`; check first whether
  `tests/test_backend_serial.py::test_resolution_matrix` monkeypatches module globals, and
  keep the resolver reading the flags from wherever the tests patch them), the
  `"jax"`-needs-`dims==3` rejection, comm/rank/size, the precision allgather, the cartesian
  communicator for `dims==3` on non-serial backends, and `init_backend` (jax.distributed +
  mesh checks; takes `nz` for the divisibility check). `jax.distributed.initialize` must
  still be the first jax device work in the process — the constraint does not move, it
  becomes visible in the API.
- `Parameters(..., runtime=None)`: `None` resolves one; an explicit `Runtime` is used as
  is (`comm_backend` must then be `None` or equal to `runtime.backend`, else `ValueError`),
  which is what lets a parameter scan share one set of communicators. `params.runtime`
  holds it; **`comm_backend, comm, rank, size, cart_comm, left_neighbor, right_neighbor`
  become read-only properties forwarding to it**, so no consumer in `taranis/`, `tests/`,
  `bench/` or `examples/` changes (`snapshot_io.py` reads `params.size` eight times; it
  stays that way). `runtime` is popped from `_init_args` (not JSON); `comm_backend` is
  recorded resolved, as now, and stays excluded from the differing-record comparison.
- `_validate_compat(self)`: the compatibility matrix (`z_spectral` × backend × size,
  particles × eqtype × size × backend, `"jax"` × dims, `nz % size`, 2D-on-many-ranks warning)
  in ONE function called once after the attributes are set — every message byte-identical
  (tests match substrings), the raise ORDER preserved for cases where two conditions hold.
- `comms.py` internals read `params.runtime` (or the properties — either; pick one and be
  consistent).
- NOT done: the `locals()` capture and the `save`/`from_snapshot` legacy folding stay (§7).

**Bitwise expectation: every gate.** No arithmetic is touched; the risk is import-order and
error-ordering, which the suite covers (`test_params`, `test_backend_serial`,
`test_backend_jax` with its 4 fake devices, `test_infra`).

**Consumers R owns:** `tests/_rmhd_testing.py::fake_ranked_params` (`p.rank = ...` →
`p.runtime = dataclasses.replace(p.runtime, rank=..., size=...)`),
`tests/test_backend_jax.py:58` (same idiom), `tests/test_params.py` (add: `runtime` is not
in `_init_args` nor in `params.json`; an explicit `runtime=` round-trips through
`from_snapshot(..., runtime=rt)`; mismatched `comm_backend` vs `runtime.backend` raises),
`tests/test_infra.py` if its `p_real.rank` check needs the property.

**Files (R):** `taranis/config.py`, `taranis/comms.py`, `taranis/_mpi_compat.py` (only if
needed), the three test files above. Not `run.py` (Phase C), not `grids.py` (Phase L).

## 5. Phase C — one run-loop body

**Today.** `run.py` has `_cfl_block`, `_cfl_block_particles`, `_block_of_steps_particles`,
`block_of_steps`, and inside `simulate` four more bodies (`stepper_wrapped`,
`block_wrapped`, `stepper_particles`, `block_particles`) plus two `sim_to_next_snap`
variants — the same step written eight times over two carry shapes.

**Target.** The carry is ALWAYS `(state, aux)` with `aux = pstate` when particles are on
and `None` off (`None` is a leafless pytree: scan/while_loop/donation treat it as nothing),
and per-step outputs are always `ys` — `(t, mom)` on, `None` off:

```
def _step(carry, kgrid, params, rhs, set_timestep, scheme, stepper, dt_override, exp_ops):
    # stepper + _advance_forcing (+ _advance_particles) -> (carry, ys)
def _cfl_block(carry, kgrid, params, rhs, set_timestep, scheme, stepper) -> (carry, ys)
def _advance_block(carry, kgrid, params, nblock, scheme, stepper) -> (carry, ys)
    # cfl blocks or the plain scan; ys reshaped (nblocks, cfl_every, ...) -> (nsteps, ...)
    # via jax.tree.map (a no-op on None)
def block_of_steps(x, kgrid, params, nblock, scheme, stepper)
    # PUBLIC, contract unchanged: particles off -> state in, state out; on -> (state, pstate)
    # in, ((state, pstate), ys) out. The one remaining static `if params.particles` in the
    # driver, and it is a wrapper.
```
`simulate`'s while_loop body is `_step(...)[0]` or `_cfl_block(...)[0]` on the same carry,
with `cond` on `carry[0].t`; `simulate_scan` appends `ys` when particles are on, as now.
The `exp_ops` hoisting stays exactly where it is — computed OUTSIDE the step scan in
`_cfl_block` and `_advance_block` (CLAUDE.md: that placement is the whole point).

**The jax-backend boundary.** `comms.shard_call` builds `in_specs` from `state_specs()`
and cannot spec a `(state, None)` carry; `comms.py` is Phase R's file and C's worktree will
not have R's changes. Particles are rejected under `comm_backend="jax"` in `Parameters`, so
at the jitted boundary on that backend the carry IS the bare state: wrap as
`lambda s, kg: _advance_block((s, None), kg, ...)[0][0]` and keep `shard_call` unchanged.
Two lines, documented in place.

**Bitwise expectation: every gate, and memory-neutral (§0.4).** The scans and while_loops
carry the same leaves in the same order; a `None` leaf adds no HLO. Gate 6 is the direct
witness (it was recorded to catch exactly this class of change, and its five configs cover
both drivers × fixed/cfl × forced/unforced). `test_particles_coupled.py` (gates 6b, 6c,
the particles-on bitwise test) and `test_particles_3d.py` cover the `aux`-on side.

**Tests (C):** none new — the contract is pinned by existing tests. C owns `run.py` only
(the restart fix from Phase 0 is already in the base).

## 6. File ownership (binding)

| phase | owns | must not touch |
|---|---|---|
| 0 | `tests/_gen_refactor_reference.py`, `tests/test_refactor_reference.py`, `tests/data/refactor_reference_*.npz`, `bench/memory_probe_refactor_base*.json`, `taranis/run.py::_refresh_forcing_scale` (0c only), `tests/test_forcing_norm_per_step.py`, `tests/test_particles_coupled.py` (gate 6c cell) | everything else |
| L | `taranis/propagators.py`, `taranis/grids.py`, `taranis/timestepping.py` (the `get_propagator` call sites and import ONLY), `tests/test_linear_propagator.py`, `tests/test_separable_propagator.py`, `tests/test_gdi_linear.py`, `tests/test_imex.py`, `tests/test_hoist_propagator.py`, `bench/zspectral_profile.py` | `physics/`, `run.py`, `config.py`, `comms.py` |
| G | `taranis/physics/__init__.py`, `taranis/physics/rmhd.py`, `taranis/physics/gdi.py`, `tests/test_equation_interface.py` (new) | `shared_physics.py`, `particles/`, `run.py`, `timestepping.py`, `test_imex.py` |
| R | `taranis/config.py`, `taranis/comms.py`, `taranis/_mpi_compat.py`, `tests/_rmhd_testing.py` (`fake_ranked_params` only), `tests/test_backend_jax.py` (line 58 only), `tests/test_params.py`, `tests/test_infra.py` | `run.py`, `grids.py`, `snapshot_io.py`, `propagators.py` |
| C | `taranis/run.py` | everything else |
| sweep | `CLAUDE.md`, `docs/numerics.md`, `docs/performance.md`, `docs/checkpointing.md`, `plans/README.md`, this file §10, `plans/refactor_notes/*` (folded in, then deleted) | source |

Ownership amendment (2026-08-22, during G): consumers of `term_funcs` that nobody owned and
that `Term` breaks — `tests/test_z_stencils.py` (identity-matched `rmhd.FDLinearTerm`, went
silently toothless), `bench/step_accounting.py` (`f.__name__`, `scoped(f)`) and the one
line `bench/zspectral_profile.py:114` (non-overlapping with L's hunks) — are G's.
During L: `webgpu/gen_refvectors3d.py:15,84` and `webgpu/gen_refvectors.py:72` (the deleted
`get_propagator` / `lin_L`) are L's, in its branch; L's `bench/step_accounting.py` edits
(lines 196–204, 253–254: the four deleted `*Propagator` monkeypatches and the
`noexpform` lambda) are deferred to main AFTER the G merge, because G's follow-up edits
the same file at 167/260. Neither bench nor webgpu generator is reached by `make test`
or `ruff` — the §0.6 "benches pass unedited" clause was not enforceable by any gate and
is amended to: every consumer outside `tests/` is listed and fixed by the phase that
broke it, with a minimal run as evidence.

Cross-phase interfaces, fixed now so nobody waits on anybody: G keeps `term_funcs` as the
field name and bare callables valid (L's `test_imex.py` constructs recipes that way); L's
`timestepping.py` edit is confined to replacing `get_propagator(kgrid, params)` with
`kgrid.lin`; C does not rely on any `comms.py` change; R keeps every `params.<attr>` read
working via properties. There is no other coupling: each branch must pass `make test` on
its own against the Phase-0 base.

## 7. Out of scope — rejected or deferred, with the reason

- **Unifying the hoisted/unhoisted propagator paths** ("move dt to the driver, form stage ops
  once per step"): forming every stage's `exp(L·γᵢ·dt)` per step IS the hoisted graph, and on
  putzer2 that is the +7.0 u (P100) the memory-light unhoisted graph exists to avoid —
  `test_unhoisted_graph_stays_memory_light` fails at 0.00 u under exactly that restructure,
  and the diagonal backend is deliberately unhoisted for gate 6 (MEMORY_PERF_PLAN §9.3).
- **Dropping `simulate`** for reverse-mode AD: `plans/AUTODIFF_PLAN.md` commits to FORWARD
  mode, which `while_loop` supports. The two drivers' snapshot-timing difference is real
  (CODE_REVIEW_2026-07-31 B7) but is not this plan's.
- **Comm backend as three classes**: nine string dispatches in `comms.py`; the real
  asymmetry is `shard_call`'s specs, which a protocol does not remove. Cosmetic.
- **Config-representation rewrite** (frozen dataclass tree, versioned migrations, no
  `locals()`): the legacy folding is needed for existing `params.json` records and would
  only be renamed; `save`'s differing-record check is the only cross-mode restart guard and
  must stay byte-compatible with old records; identity hashing is load-bearing.
- **Forcing state out of `SimulationState`**: changes the on-disk layout and the gate-6
  reference for a `(n_ou, 2, nkx, nky)` array.
- **n-field / batched propagator backend**: design it against the real second consumer
  (KRMHD's Hermite coupling is tridiagonal in the moment index, not block-2×2).
- **"Always carry z at size 1 in 2D"**: already the case (`dims=2` gives `(nfields, 1, nx,
  ny//2+1)`); the remaining `dims==2` branches are the z-only attributes and the FD stencil.
- **ETDRK**: `rmhd.set_timestep` is CFL-only, so the modes where IF misweights the nonlinear
  driving are the hyperdissipated ones near the 2/3 cutoff — not an energy-carrying balance.
  The quasi-static balance ETD would fix is GDI's γ∥, which IMEX serves. A plan-file
  candidate if a wave-plus-stiff-damping equation set arrives; costs 4 nonlinear registers
  (vs lsrk54's 2) and three φ-coefficient arrays per stage product.
- fp64 default, `forcing_shell_noise` default-on, embedded-error dt controller, parallel
  z-FFT, `halo_start` re-test under the latency-hiding flag: not structural; the last is an
  open item in `docs/performance.md`.

## 8. Execution

Standing flow (memory `phase-execution-flow`): opus implementers, Fable overseer does not
implement, fresh-Fable adversarial review per phase, commit before review (the reviewer
mutates files to test gates' teeth). `caffeinate -s` for the dispatch; scratchpads vanish on
reboot.

1. **Phase 0** — one opus agent on `main`, sequential. Two commits (0a+0b, then 0c). Review.
   Tag/record the base SHA in §10.
2. **Worktrees** — the overseer builds them (agent-cut worktrees can start stale; memory
   `agent-worktree-stale-base`), all from the Phase-0 base:
   ```
   git worktree add /private/tmp/taranis-wt-L -b refactor/L <base>
   git worktree add /private/tmp/taranis-wt-G -b refactor/G <base>
   git worktree add /private/tmp/taranis-wt-R -b refactor/R <base>
   git worktree add /private/tmp/taranis-wt-C -b refactor/C <base>
   ```
   and passes each agent its absolute worktree path, this file's absolute path, and the §6
   row it owns. Never `git stash` in the shared tree (memory).
3. **Phase L, G, R, C in parallel** — four opus agents. Each brief: the §-text of its phase,
   its §6 row, §0, and the gate list below. Each agent: implements, runs `make test`
   (both precisions) in its worktree, runs the probe comparison where §0.4 requires it,
   writes `plans/refactor_notes/<phase>.md` (what changed, every gate result with numbers,
   anything bitwise-adjacent it noticed, proposed CLAUDE.md wording), commits on its branch.
   Concurrent test runs are fine (nothing here is timed; `memory_analysis()` is
   deterministic) — but the probe's ms/step columns are contaminated and are NOT to be
   quoted (memory `multiagent-timing-measurements`).
4. **Review** — four fresh-Fable adversarial reviews, one per worktree, in parallel: hunt
   for ownership violations, narrative comments, weakened gates, silent behaviour changes,
   and whether the new tests have teeth (mutate, watch them fail). Fix rounds go back to the
   same implementing agent (SendMessage keeps its context).
5. **Merge** in the order **L → G → C → R** (least to most cross-cutting in what the rest of
   the tree imports): for each, rebase the branch onto the current `main` in its worktree,
   `make test`, fast-forward merge, `make test` on `main` again. No conflicts are expected by
   construction; a conflict is an ownership violation and goes back to review.
6. **Integration** — on the merged `main`: full suite both precisions, the four references,
   the probe against the Phase-0 JSON, one `mpirun -n 4` of `tests/test_snapshot_roundtrip.py`
   and `tests/test_backend_jax_mpi.py` is Alfred's (Savio). One fresh-Fable integration review.
7. **Sweep** — one agent: CLAUDE.md (the `K_Grids`/`lin_*` paragraph, "never read `lin_*`
   from a stepper" → "only `kgrid.lin`'s methods, never its arrays", the `grads` and
   `term_funcs` sentences, the carry description in the particles section, the
   `Parameters`/runtime paragraph, the restart-bitwise line), `docs/numerics.md` and
   `docs/performance.md` mentions of the old names (16 hits today across CLAUDE.md and
   docs/), `plans/README.md`, §10 here; fold `plans/refactor_notes/*` in and delete them.
   Overseer reviews the sweep diff line by line.

**Gate list for every phase branch** (all must be green, none may be skipped or loosened):
`make test` (fp64 and fp32 sessions); `tests/test_refactor_reference.py`,
`tests/test_particles_coupled.py` (gate 6 and 6c), `tests/test_forcing_spinup.py`,
`tests/test_precision_fp32.py` bitwise with no regeneration; the HLO opcode histograms in
`tests/test_refactor_reference.py` unchanged (G: fewer adds only, §0.4);
`tests/test_hoist_propagator.py` including the memory-light gate; for C and L the probe
comparison ≤ 0.05 u per case against `bench/memory_probe_refactor_base*.json`; `ruff`
clean as the CI workflow runs it.

## 9. Decisions for Alfred

Answered 2026-08-22 (before dispatch): **1 → (b)** — accept and regenerate with a dated
note once the cause is understood and is container/fusion-class; large drift is tracked
down, not accepted. **2 → yes**, the scan semantics. **3 → ask**, never resolve in the
shared tree.

1. **Any bitwise drift in any phase.** Default: it is a bug until characterized. If a phase
   shows fusion-class drift (a few elements at round-off in one config) that the
   implementer and reviewer both attribute to argument-order or scan-structure changes and
   cannot remove without restructuring arithmetic, the options are (a) find the
   container-level cause and remove it, (b) accept and regenerate the reference with a dated
   note here, as F3 did. Never (c) reorder floating-point ops to make it clean.
2. **`runtime=` sharing semantics.** As specified, an explicit `Runtime` is reused without
   re-validation of `nz % size` against ITS size — `_validate_compat` checks it against
   `runtime.size`, so a `Runtime` built for one `nz` works for any `nz` divisible by its
   size. Confirm that is the wanted semantics (it is the one a parameter scan wants).
3. **Merge order** if a rebase conflicts anywhere — stop and ask rather than resolve in the
   shared tree.

## 10. Results

**Phase 0 (2026-08-22).** `b4de4d5` (0a+0b) and `3073df4` (0c) — **the fan-out base is
`3073df4`**; worktrees `/private/tmp/taranis-wt-{L,G,R,C}` on branches `refactor/{L,G,R,C}`.
Twelve-config reference recorded on Alfreds-MacBook-Pro.local / jax 0.10.0 / cpu / py
3.11.5, determinism double-record bitwise (24 arrays + 12 HLO histograms, both
precisions); probe baseline 28 cases per precision. `make test` at 0c: fp64 247 passed /
23 skipped, fp32 224 / 46. Gate 6, spinup, fp32 and the new reference all green after 0c
with no regeneration; the three new restart tests and the widened gate 6c fail on the
unfixed tree. Side findings for the sweep (from `plans/refactor_notes/phase0.md`):
CLAUDE.md's `GRAD_CHUNK` "bitwise identical" is one ulp too strong in 3D (637/2304
elements at 1.1e-16 between chunk 1 and 2/4; 2D exact); under fp32, `t` is stored fp64
but accumulates fp32-rounded `gamma*dt` increments in the IF steppers (0.059999998 after
6×0.01) while imexcb3f reaches 0.06 exactly — `t` is a live comparator in the reference.
Phase 0 review (fresh Fable, same day): **accepted with one fix** — the HLO histogram regex
(inherited from `bench/hlo_audit.py:119`) never matched XLA's `ROOT %name = ...` lines, so
every computation's final instruction was uncounted (127/3040 on `fd_fixed_lsrk54`, both
`while` loops among them; fusion count 104 vs real 110). The gate still bit on every
mutation tried (a 1+1e-13 factor on `bracket`: 12/12 fields + histogram; `GRAD_CHUNK=2`:
3D fields one ulp, 2D bitwise, histogram in all 12 — the pure-graph case IS caught). Fix +
histogram-only regeneration landed on main before any merge: **`b990d9d`** (ROOT-inclusive
totals 587/2913→3040 on the two configs; npz sha256 unchanged; two further double-records
agree; `bench/hlo_audit.py` got the same one-token fix — its own op tables were
under-counting ROOT lines). Branches cut from `3073df4` carry the old regex; the merge step
re-runs the histogram gate on main with the fixed one. Also verified:
restart rule correct on every entry path incl. the 4-fake-device jax backend and
`forcing_power=0`; gate 6c keeps all ten checks per cell; no ownership/comment/skip
violations. Sweep items from the review: `tests/test_particles_3d.py:849-852` and
`plans/TESTPART_PLAN.md:517,594,1293` now describe the old refresh behaviour (the 3D gate
could be widened to both settings like 6c); CLAUDE.md's `GRAD_CHUNK` sentence is true of
`grad_fields`' output (`test_grad_memory` pins it) — it is the 3D stepped solution that is
one ulp off. Pre-existing, outside the phase: `tests/test_backend_jax.py::
test_same_seed_run_matches_serial_reference` fails under 4 forced host devices on this host
(jax 0.10.0; identical on the unfixed tree) and `make test` skips every `multidev` test
here because mpi4py imports — the jax backend is NOT covered by the local suite.

(per-phase gate numbers, drift findings, the sweep, and what moved to `plans/old/`: below
as they land)
