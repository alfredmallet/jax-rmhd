# Phase G — typed equation interface

Branch `refactor/G`, cut from `3073df4`. Scope: `plans/REFACTOR_PLAN.md` §3, files
`taranis/physics/__init__.py`, `taranis/physics/rmhd.py`, `taranis/physics/gdi.py` plus the
new `tests/test_equation_interface.py`. Nothing outside that row was edited.

## What changed

**1. Named gradients.** `rmhd.RMHDGrads(gphi, gpsi, gvort, gjpar)` and
`gdi.GDIGrads(gphi, gN, gvort)` NamedTuples, in the existing `grad_func` order;
`rmhd.grad`/`gdi.grad` return `<Grads>(*grad_fields(...))`. `rmhd.set_timestep`,
`rmhd.NonlinearTerm`, `gdi.set_timestep` and `gdi.NonlinearTerm` read by name.
`shared_physics.grad_fields` still returns a plain tuple — the recipe names it.
A NamedTuple is a tuple, so every positional consumer (`run._block_dt`,
`run.estimate_good_nblock`, `particles/`, the benches) is unedited and unaffected.

**2. `physics.Term(func, active=lambda params: True)`.** `EquationRecipe.term_funcs` keeps
its name and accepts bare callables — `construct_rhs` wraps anything that is not a `Term`,
which is what keeps `tests/test_imex.py`'s and `tests/test_z_stencils.py`'s recipes valid
unedited. The registry now lists `Term(rmhd.NonlinearTerm)`,
`Term(rmhd.FDLinearTerm, active=rmhd.fd_linear_active)`,
`Term(rmhd.ForcingTerm, active=rmhd.forcing_active)` and `Term(gdi.NonlinearTerm)`.
`construct_rhs` selects `[t.func for t in terms if t.active(params)]` before touching the
halo or the gradients — `params` is static, so this is plain python at trace time — and
raises `ValueError` naming `construct_rhs`, the word "inactive" and the term names when the
selection is empty.

**3. The predicates are the single source of truth.** `rmhd.fd_linear_active(params)` is
`dims == 3 and not z_spectral`; `rmhd.forcing_active(params)` is `bool(params.forcing)`.
`FDLinearTerm`/`ForcingTerm` keep their early `zeros_like` returns, now expressed through
those predicates, because direct callers exist (`particles/fields.py::_forcing_ez`,
`tests/test_forcing_modes.py`, `tests/test_z_spectral.py`). From the RHS they are never
reached.

Nothing else: `nfields`, the 5-positional-arg term contract, `halo`, `forcing_scale_func`
and `halo_start_func` are untouched.

## New test — `tests/test_equation_interface.py`

Both precision sessions, `_rmhd_testing` conventions (`bootstrap()` before `import
taranis`, `script_main(globals())` footer), mpirun-safe.

| test | what it pins |
|---|---|
| `test_grads_are_named_in_the_documented_order` | field names and order for both equations; each entry real `(2,nz,nx,ny)` at field precision; entry *i* is the same array as the name; each equals an independently built `ifft(i k · <documented k-space field>)` |
| `test_terms_and_set_timestep_read_grads_by_name` | rebuilding the NamedTuple from `tuple(grads)` leaves every term and `set_timestep` bitwise — the names label the positional contract, they do not replace it |
| `test_construct_rhs_calls_only_active_terms` | counting `Term`s in a toy recipe: the active func runs once, the inactive func never runs, both predicates are consulted, a bare callable is always active |
| `test_all_terms_inactive_is_a_clear_error` | the empty selection is a `ValueError`, not a `None` RHS |
| `test_filtered_rhs_equals_the_all_terms_sum_bitwise` | 2D forced (FD inactive), 3D FD-z forced (nothing inactive), 3D z_spectral unforced (both optional terms inactive): `jnp.array_equal` against the explicit `NL + FD + Forcing` sum |
| `test_predicates_run_once_per_trace_not_once_per_step` | `jax.jit(block_of_steps)` at nblock 2 and 32 consults the predicate the same 2 times; a repeat call at the same static arguments consults it 0 further times |

## Gates

Run from `/private/tmp/taranis-wt-G`. The machine was running three other phases' suites
concurrently, so **no timing is quoted**.

- `make test`: fp64 `253 passed, 23 skipped, 1 deselected`; fp32 `230 passed, 46 skipped,
  1 deselected`. 276/277 collected, of which 6 are the new module. Nothing failed, nothing
  was loosened or marked skip by this phase.
- `tests/test_refactor_reference.py`, **no regeneration** — `tests/data/refactor_reference_*`
  are byte-identical to the base (sha256 unchanged: npz fp64 `f5b19b56`, fp32 `294cb946`,
  hlo fp64 `91816086`, fp32 `f0db5e75`).
  - fp64: 12/12 configs `fields bitwise identical to the reference`, 12/12 `t bitwise
    identical`, 12/12 `opcode histogram unchanged` (plus instruction and fusion totals).
  - fp32: same, `2 passed`.
- `ruff check .`: clean.

Gate 6 / 6b / 6c (`tests/test_particles_coupled.py`), `tests/test_forcing_spinup.py`,
`tests/test_precision_fp32.py`, `tests/test_hoist_propagator.py` including
`test_unhoisted_graph_stays_memory_light` all run inside `make test` and are green in both
sessions. The probe comparison is not a G gate (§0.4 asks it of C and L).

## HLO opcode deltas — there are none

The committed sidecar comparison passes untouched at both precisions. Because that
sidecar's regex misses `ROOT` lines (being fixed on `main`), the delta was ALSO measured
with a ROOT-aware regex, as a same-process A/B: the shipped (filtered) registry against one
whose `Term`s are all forced active, which reproduces the pre-G graph exactly (the old
`construct_rhs` called every term unconditionally, and the terms' early zero returns are
unchanged). Optimized HLO of the jitted `block_of_steps`, 6 steps, the twelve Phase-0
configs:

| config | inactive term(s) | fp64 instrs before→after | fusions | fp32 instrs before→after | fusions | opcode deltas |
|---|---|---|---|---|---|---|
| fd_fixed_lsrk54 | none | 3040 → 3040 | 110 → 110 | 2374 → 2374 | 110 → 110 | none |
| fd_cfl2_lsrk33 | none | 3239 → 3239 | 122 → 122 | 2574 → 2574 | 122 → 122 | none |
| fd_adapt_rk44 | none | 4003 → 4003 | 132 → 132 | 3337 → 3337 | 132 → 132 | none |
| sep_fixed_lsrk54 | FDLinearTerm | 3077 → 3077 | 111 → 111 | 2411 → 2411 | 111 → 111 | none |
| put_cfl2_lsrk54 | FDLinearTerm | 4540 → 4540 | 147 → 147 | 3875 → 3875 | 147 → 147 | none |
| put_cfl2_lsrk54_nohoist | FDLinearTerm | 3658 → 3658 | 131 → 131 | 2993 → 2993 | 131 → 131 | none |
| put_adapt_lsrk33_unrolled | FDLinearTerm | 3713 → 3713 | 122 → 122 | 3049 → 3049 | 122 → 122 | none |
| rmhd2d_adapt_lsrk33_mom | FDLinearTerm | 2744 → 2744 | 98 → 98 | 2079 → 2079 | 98 → 98 | none |
| rmhd2d_fixed_imexcb3f | FDLinearTerm | 3716 → 3716 | 116 → 116 | 3049 → 3049 | 116 → 116 | none |
| gdi2d_fixed_imexcb3e | none | 587 → 587 | 30 → 30 | 587 → 587 | 30 → 30 | none |
| gdi2d_fixed_lsrk33 | none | 1354 → 1354 | 54 → 54 | 1355 → 1355 | 54 → 54 | none |
| gdi3d_fixed_imexcb3e | none | 587 → 587 | 30 → 30 | 587 → 587 | 30 → 30 | none |

**Every opcode count is identical.** §0.4 allowed "fewer add/broadcast/constant ops"; the
measurement is that XLA's algebraic simplifier had already folded `add(x, broadcast(0))`
away completely, so removing the term at trace time removes nothing the optimizer had left.
No config forced `forcing=False`, so `ForcingTerm` is inactive only in the extra
`z_spectral` unforced cell of the new test, which is not an HLO reference config.

The filtering IS reaching the graph — it is just erased downstream. Pre-optimization
StableHLO of the same jitted block (fp64, lines of module text): `rmhd2d_adapt_lsrk33_mom`
980 → 974, `put_cfl2_lsrk54` 1635 → 1629 (the six lines are the inactive term's zero
constant, its broadcast and the add, in the stage-scan body), `fd_fixed_lsrk54` 1161 → 1161
(nothing inactive). `tests/test_equation_interface.py` pins the same property directly, by
counting calls.

Aside for the sidecar regeneration on `main`: the ROOT-aware regex raises the totals by
~4.4% (e.g. `fd_fixed_lsrk54` fp64 2913 → 3040) and the fusion counts by ~5–6% (104 → 110);
it changes no G-attributable count.

## Bitwise-adjacent notes

- The only arithmetic change is dropping `x + zeros_like(x)` for an inactive term. That is
  exact except for the sign of a zero, and `np.array_equal` treats `-0.0 == 0.0`. Measured:
  no reference moved, at either precision.
- `construct_rhs` now evaluates the predicates BEFORE `halo_start_func`. `rmhd.halo_start`
  already returns `None` in the modes where `FDLinearTerm` is inactive, and the gate on
  issuing it (`_halo_start_enabled`) is unchanged, so no exchange appeared or vanished.
- `Term.active` runs at trace time only. Under `lsrk_scan` the stage body is traced once, so
  the predicate count does not scale with steps or stages (measured: 2 calls per trace of a
  one-term toy recipe, independent of nblock).

## Three consumers of `Term` — found, then fixed (follow-up commit)

The first pass found three places that treated `term_funcs` entries as bare callables. The
overseer amended G's ownership to include them; they are fixed in the follow-up commit
`refactor G: consumers of Term`. `Term` stays a plain record — it was NOT made callable;
direct callers read `.func`.

**1. `tests/test_z_stencils.py::test_fdz_solver_bitwise_over_20_steps` had lost its teeth
silently.** It built its reference term tuple with
`tuple(_padded_fd_linear_term if t is rmhd.FDLinearTerm else t for t in shipped)`. Once
`shipped` held `Term`s the identity test matched nothing, so `reference == shipped` was
`True` (verified in-process) and the test compared the shipped stencil against itself — and
still PASSED, which was the problem. Now:

```python
reference = tuple(t._replace(func=_padded_fd_linear_term)
                  if t.func is rmhd.FDLinearTerm else t for t in shipped)
assert any(t.func is _padded_fd_linear_term for t in reference), \
    "the padded reference term was not substituted into the recipe"
```

The `assert` is the cheap guard against the same class of silent no-op substitution.

**Teeth proof.** `_padded_fd_linear_term`'s return was temporarily scaled by an
environment-set factor, the test run, then the file restored from a backup and re-run.

- fp64, reference scaled by 1 + 1e-12 — **FAILS**, as required:
  ```
  AssertionError: 2 check(s) failed:
    - backend=default: fields bitwise equal to the padded reference after 20 steps (800/2304 differ, max|diff|=1.253e-11)
    - backend=serial: fields bitwise equal to the padded reference after 20 steps (800/2304 differ, max|diff|=1.253e-11)
  1 failed
  ```
- fp32, reference scaled by 1 + 1e-6 (1 + 1e-12 is exactly 1.0 in float32, so it cannot
  perturb anything) — **FAILS**, as required:
  ```
  AssertionError: 2 check(s) failed:
    - backend=default: fields bitwise equal to the padded reference after 20 steps (797/2304 differ, max|diff|=1.490e-05)
    - backend=serial: fields bitwise equal to the padded reference after 20 steps (797/2304 differ, max|diff|=1.490e-05)
  1 failed
  ```
- fp32 control, scale exactly 1.0 — `1 passed`.
- Restored file, whole module: fp64 `6 passed, 1 skipped`; fp32 `6 passed, 1 skipped` (the
  skip is the `multidev` 4-device case).

**2. `bench/step_accounting.py`** — `drop_term` (line 167) and the `scoped(...)` wrap
(line 260) now substitute INSIDE the record, so the recipe keeps its `Term` structure and
the shipped predicates: `t._replace(func=zero_term) if t.func.__name__ == name else t` and
`t._replace(func=scoped(t.func.__name__.upper(), t.func))`.
Proof it runs: a `--single` timing job, `path=fdz`, `variant=nofdlin`, 16²×8, nblock 2 —
which exercises `drop_term` AND the unconditional `scoped` loop — returns `"error": null`.

**3. `bench/zspectral_profile.py:114`** — `zip(..., (t.func for t in recipe.term_funcs))`.
Only that line was touched; Phase L's edits to that file are in the propagator block around
lines 150–160, so the hunks do not overlap. Proof it runs: `python bench/zspectral_profile.py
16 8 2 1` completes all four tables, including the `term: nonlinear / fdlinear / forcing`
rows that go through the fixed line. **Note for L:** unlike `bench/step_accounting.py`,
this bench does not insert the repo root on `sys.path`, so run from a worktree it imports
the *installed* `taranis` unless `PYTHONPATH=.` is set. Pre-existing, not touched here.

Follow-up gates: `ruff check .` clean; `tests/test_z_stencils.py` green at both precisions;
`make test-fast` (fp64) `253 passed, 23 skipped, 1 deselected`, unchanged from the first
commit's fp64 session. fp64 alone is enough for the follow-up — it
changes no solver code, only two benches and one test's substitution.

## CLAUDE.md sentences that should change (sweep)

Architecture → "Parameters / physics registry":

- The `term_funcs` sentence. Today: "`term_funcs` are summed into the RHS
  (`construct_rhs`)". Proposed: "`term_funcs` entries are `physics.Term(func, active)` (a
  bare callable is accepted and means always active) and `construct_rhs` sums only the terms
  whose `active(params)` is true — plain python at TRACE time, `params` being static, so an
  inactive term never enters the graph; an empty selection is a `ValueError`. The predicates
  are the single source of truth: `rmhd.fd_linear_active` (`dims==3 and not z_spectral`) and
  `rmhd.forcing_active`; the term funcs keep their early `zeros_like` returns for direct
  callers such as `particles/fields.py::_forcing_ez`."
- The `grads` sentence. Today: "**`grads` is a TUPLE**, one real-space `(2,nz,nx,ny)` array
  per field in the equation set's `grad_func` order (RMHD: `gphi, gpsi, gvort, gjpar`),
  never a stacked `(nfields,2,…)` array — unpack it, never `.shape`/`grads[:2]` it."
  Proposed: "**`grads` is a NAMED TUPLE** — `rmhd.RMHDGrads(gphi, gpsi, gvort, gjpar)` /
  `gdi.GDIGrads(gphi, gN, gvort)`, one real-space `(2,nz,nx,ny)` array per field in the
  equation set's `grad_func` order — never a stacked `(nfields,2,…)` array. Read it by name;
  it is still a tuple, so positional consumers keep working, but never `.shape`/`grads[:2]`
  it. `shared_physics.grad_fields` returns the plain tuple; the recipe names it."
