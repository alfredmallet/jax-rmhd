# Phase L — the linear operator as a typed pytree

Branch `refactor/L`, base `3073df4`. Spec: `plans/REFACTOR_PLAN.md` §2, ownership row §6.

## What changed

**`taranis/propagators.py`**

- Four operator NamedTuples replace the four propagator classes:
  `IdentityOperator()` (no fields), `DiagonalOperator(L)`, `Putzer2Operator(L, m, s)` and
  `SeparableL(dperp, dz, kz)` — the last keeping its name, since `rmhd.linear_matrix`
  returns it, and gaining `SeparablePropagator`'s methods verbatim (`self.sep.x` → `self.x`).
  `SeparablePropagator`, `DiagonalPropagator`, `Putzer2Propagator`, `IdentityPropagator`,
  `get_propagator` and `dense_operator` are gone.
- The methods are byte-for-byte the old bodies: `scaled`, `exp_op`, `apply_exp`,
  `solve_shifted`, `apply_L`, and putzer2's `_coeffs`. `hoistable` is a property
  (False/False/True/True, same values).
- `dense()` per operator replaces `dense_operator(kgrid)`'s shape sniffing, one branch per
  class, each the old branch verbatim; `IdentityOperator.dense()` raises the old
  `ValueError` with the old message.
- `linear_fields` → `build(L, params) -> operator`. Every validation, every message and the
  order they fire in are unchanged, including the `comm_backend="jax"` z-extent
  `NotImplementedError` in both the dense and the separable path. `_separable_fields` →
  `_separable_operator`, which now returns the validated `SeparableL` instead of a dict.
- `putzer2_precompute` unchanged.

**`taranis/grids.py`**

- The six `lin_*` slots collapse to one, `lin`, defaulting to `IdentityOperator()`.
  `_attach_linear_operator` does `kgrid._replace(lin=build(L, params))`, and
  `_replace(lin=IdentityOperator())` when the recipe has no `linear_matrix_func`, so
  `kgrid.lin` is always populated after `setup_kgrids`.
- `kgrid_specs` / `_kgrid_to_global` handle `lin` with `jax.tree.map` (every leaf
  replicated, `P()`); the other entries take the same branches as before.

**`taranis/timestepping.py`** — only the import and the five `get_propagator(kgrid, params)`
call sites (`stage_exp_ops`, `rk_advance`, `lsrk_advance`, `imex2r_advance`,
`imex3r_advance`), each now `kgrid.lin`. Nothing else in the file moved.

**Tests / bench** — mechanical renames in `test_linear_propagator.py`,
`test_separable_propagator.py`, `test_gdi_linear.py`, `test_imex.py`,
`test_hoist_propagator.py` and `bench/zspectral_profile.py` (whose
`_replace(lin_L=None, ...)._replace(**linear_fields(...))` dance is now
`_replace(lin=build(L, p))`). The former "putzer2 precomputes are absent" /
"lin_L is None" assertions became `_fields` and `isinstance` assertions on `kgrid.lin`,
which say the same thing about the same object.

## New tests (`tests/test_linear_propagator.py`)

1. `test_kgrid_lin_leaves_are_the_operator_arrays` — for all four backends (2D, FD-z,
   z_spectral ν = η, ν ≠ η): `kgrid.lin` is the expected class, and its pytree leaves are
   bitwise (and dtype-) equal to the arrays rebuilt independently from
   `rmhd.linear_matrix` + `grids`' casts + `putzer2_precompute`. This is the direct
   statement that the single slot carries exactly what the six slots did.
2. `test_kgrid_lin_rides_through_jit` — for the same four backends and each of
   `apply_exp`, `scaled(dt).apply_exp`, `solve_shifted`, `apply_L`: jitting a function
   that takes the whole `kgrid` gives bitwise the same array as jitting one that takes the
   bare operator (the pytree rides through the kgrid unchanged), and the traced value
   matches the eager one — bitwise on the elementwise backends, to round-off on the
   separable one (see "bitwise-adjacent" below).
3. `test_a_recipe_without_a_linear_matrix_gets_the_identity_operator` — a toy recipe with
   `linear_matrix_func=None` (the `test_imex.py` `_registered` idiom) yields an
   `IdentityOperator` with no leaves: not hoistable, `scaled` returns the same object,
   `apply_exp`/`solve_shifted`/`exp_op(...).apply` return the input array object,
   `apply_L` is zero, `dense()` raises.

## Gate results

All green, nothing skipped or loosened, no reference regenerated. Machine: the recording
host (`Alfreds-MacBook-Pro.local`, jax 0.10.0, cpu, python 3.11.5), so the host-skip rule in
`test_refactor_reference.py` and gate 6a did NOT fire — the comparisons really ran. Other
agents were running tests concurrently, so no ms/step number from any of this is quoted.

| gate | result |
|---|---|
| `make test`, fp64 session | **250 passed, 23 skipped**, 1 deselected (0 failed) |
| `make test`, fp32 session | **227 passed, 46 skipped**, 1 deselected (0 failed) |
| `tests/test_refactor_reference.py` fp64 | 2 passed; **24/24 bitwise** (12 configs × fields, t) and **36/36 HLO checks** (12 × instructions, fusions, opcode histogram), 0 FAIL lines |
| `tests/test_refactor_reference.py` fp32 | 2 passed; 60 PASS lines, 0 FAIL |
| gate 6 / 6b / 6c (`tests/test_particles_coupled.py`) | 20 passed fp64, 10 passed + 10 precision-skipped fp32 |
| `tests/test_forcing_spinup.py` | 3 passed fp64 (fp32-skipped by its own markers, as before) |
| `tests/test_precision_fp32.py` | 4 passed in the fp32 session (fp64-skipped, as before) |
| `tests/test_hoist_propagator.py` | 5 passed in each session, `test_unhoisted_graph_stays_memory_light` included |
| `tests/test_linear_propagator.py` | 13 passed in each session (10 old + 3 new) |
| probe, laptop profile, fp32 | 28 cases, **max \|Δ total_u\| = 0.0000 u** vs `bench/memory_probe_refactor_base.json` |
| probe, laptop profile, fp64 | 28 cases, **max \|Δ total_u\| = 0.0000 u** vs `bench/memory_probe_refactor_base_fp64.json` |
| `ruff check .` (ruff 0.16.1, the pinned CI version) | All checks passed |

The HLO instruction and fusion counts are unchanged config by config — e.g.
`fd_fixed_lsrk54` 2913 / 104, `sep_fixed_lsrk54` 2949 / 105, `put_cfl2_lsrk54` 4372 / 140,
`put_cfl2_lsrk54_nohoist` 3506 / 124, `gdi2d_fixed_lsrk33` 1295 / 52. The hoisted/unhoisted
putzer2 gap the memory-light gate measures is therefore intact at the graph level too.

## Bitwise-adjacent observations

- **Leaf ORDER through jit is unchanged.** With six `Optional` slots the unpopulated ones
  contributed no pytree leaves, so a diagonal kgrid flattened to `[..., kz?, lin_L]`, a
  putzer2 one to `[..., lin_L, lin_m, lin_s]` and a separable one to
  `[..., lin_dperp, lin_dz, lin_kz]`. The single `lin` slot sits where `lin_L` sat and its
  subtree flattens in the same order, so every jitted `kgrid` argument keeps the leaf order
  it had. Only the treedef (a jit cache key, not a lowering input) changes. That is why the
  HLO histograms are expected to hold — and they do.
- **Eager vs jit on the separable backend is NOT bitwise, and never was.** New test 2
  measures ~5.9e-16 (apply_exp) and ~5.0e-16 (solve_shifted) relative between the eager and
  the traced call at fp64, 8×16²; the same numbers, to the last digit, come out of a
  hand-rolled copy of the pre-Phase-L `SeparablePropagator` driven off the same three
  arrays, and old-eager vs new-eager and old-jit vs new-jit are bitwise equal. It is XLA
  fusing the complex sums of `SeparableExp.apply` / `solve_shifted` differently from eager
  op-by-op evaluation, not a Phase-L effect. Test 2 asserts bitwise on the elementwise
  backends and round-off there, with the reason in the check line.

## Consumers OUTSIDE the L ownership row that now need an edit

None of these is touched on this branch (§6 forbids it), none is reached by `make test`
(`testpaths = ["tests"]`) or by `ruff check .` (pyflakes does not resolve attributes), and
all three break at runtime as they stand:

- `bench/step_accounting.py:196,200,204,253-254` — monkeypatches `propagators.SeparablePropagator`,
  `Putzer2Propagator`, `DiagonalPropagator`, `IdentityPropagator` (`.exp_op` in the
  `noexpform` ablation and in the scoped-timer wiring). Rename to the four `*Operator`
  classes / `SeparableL`, and the `noexpform` lambda's `self.sep.dperp/dz/kz` become
  `self.dperp/dz/kz`. The `*Exp` classes it also patches are unchanged.
- `webgpu/gen_refvectors3d.py:15,84` — `from taranis.propagators import get_propagator`,
  `prop = get_propagator(kgrid, params)` → `prop = kgrid.lin` (drop the import).
- `webgpu/gen_refvectors.py:72` — `kgrid.lin_L[0, 0].real` → `kgrid.lin.L[0, 0].real`.

Comment-only, cosmetic, no runtime effect: `tests/test_dissipation.py:3` and
`tests/test_time_order.py:13` say `kgrid.lin_L` in their header prose, and
`examples/gdi-2D.ipynb` names `propagators.Putzer2Propagator` in four markdown/comment
cells.

`taranis/particles/fields.py::_psi_linear_diagonal` needs NO edit and was verified: it calls
`rmhd.linear_matrix` directly and dispatches on `isinstance(L, SeparableL)` / `L.ndim`, both
of which still hold (`SeparableL` kept its name and its three fields).

## Proposed CLAUDE.md / docs wording

**CLAUDE.md, "Parameters / physics registry"**, the sentence beginning "the k-local LINEAR
part is not an RHS term". Replace the `lin_*` clause with:

> `linear_matrix_func(kgrid, params) -> L` (convention `dt f = L f + N(f)`) is validated by
> `propagators.build` and stored by `setup_kgrids` in the SINGLE slot `kgrid.lin`, as one of
> four operator pytrees — `IdentityOperator` (no `linear_matrix_func`), `DiagonalOperator(L)`
> (dense 4-d L), `Putzer2Operator(L, m, s)` (dense 2×2 5-d L; `m` and `s = sqrt((tr L/2)^2 −
> det L)` are precomputed at SETUP — never `lin_s2`, which Z2 retired) and `SeparableL(dperp,
> dz, kz)` (what `rmhd.linear_matrix` returns for `z_spectral` with `diss[0]==diss[1]`). The
> steppers apply it only through the `taranis.propagators` hook (`apply_exp`,
> `solve_shifted`, `scaled`, `apply_L`, `exp_op`, plus `hoistable`); `dense()` materialises
> the 2×2 L for tests and validation and must never be formed inside a step. **Never read
> `kgrid.lin`'s arrays from a stepper — only its methods**; and never test `kgrid.lin` for
> truthiness (an empty NamedTuple is falsy) or index it positionally outside its own methods.

The old sentences that go: "…is built once by `setup_kgrids` into `kgrid.lin_L` (+
`lin_m`/`lin_s` for putzer2 …) or `lin_dperp`/`lin_dz`/`lin_kz` (a `SeparableL` return;
populated INSTEAD of the dense trio, never alongside)", "backend: a `SeparableL` → separable,
a dense L by shape — diagonal 4-d, putzer2 2x2 5-d", and "Never reintroduce `kgrid.hdiss` or
read `lin_*` from a stepper directly" (keep the `hdiss` half).

**CLAUDE.md, the hoist table's caption** (line ~221): "XLA keeps a 4 u copy of the dense
`lin_L`" → "of the dense `lin.L`".

**docs/numerics.md** — line ~261 "stores it on the `K_Grids` (`lin_L`, plus `lin_m`/`lin_s`
for the 2×2 …)" → "stores it on the `K_Grids` as `lin`, an operator pytree"; line ~278
"`s` is formed once at setup (`kgrid.lin_s`)" → "(`kgrid.lin.s`)"; line ~354 "stores three
small REAL arrays: `lin_dperp` (nkx,nky), `lin_dz` and `lin_kz` (nz,1,1) — against putzer2's
6 u of resident complex full-grid `lin_L/lin_m/lin_s`" → the same three names without the
`lin_` prefix, as `SeparableL`'s / `Putzer2Operator`'s fields; lines ~392-394
"`propagators.dense_operator` helper exists for tests and `particles/fields.py` only" →
"the operators' `dense()` method exists for …".

**docs/performance.md** — lines ~456, 795, 817, 820, 838, 845, 1013 name `lin_L`, `lin_m`,
`lin_s2`, `lin_dperp/lin_dz/lin_kz` in measurement tables. These are records of what was
measured; the names now read `lin.L`, `lin.m`, `lin.dperp` etc. Historical `lin_s2` mentions
(pre-Z2) should stay as they are.

`webgpu/SPEC.md:269` and `webgpu/README.md:1156` use `lin_L` as the name of a reference-vector
entry produced by `webgpu/gen_refvectors.py`; whether that JSON key changes is the webgpu
side's call, not this refactor's.
