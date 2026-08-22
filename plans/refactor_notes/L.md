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

## Which gate catches what (from the review's mutation pass)

- **`tests/test_hoist_propagator.py` cannot see a reassociation inside `exp_op`.** Its
  hoisted-vs-unhoisted cells compare two paths that both go through the SAME `exp_op`, so a
  changed op order there moves both sides together and every cell stays green. The gate that
  catches it is the Phase-0 reference: a mutated `exp_op` shows up as **374-494 elements at
  ~1e-13** on the putzer2 configs of `tests/test_refactor_reference.py`. Anyone tempted to
  treat the hoist test as the propagator-arithmetic gate should read that the other way
  round.
- **The reference's separable config carries no `z_diss_k`.** `sep_fixed_lsrk54` (the only
  separable config in `tests/_gen_refactor_reference.py`) sets `diss=(1e-4, 1e-4), hyper=2`
  and nothing else, so `rmhd.linear_matrix`'s `zdiss` defaults to 0 and `SeparableL.dz` is
  exactly zero throughout it. The separable `dz` arithmetic — the `+ self.dz` in `apply_L` /
  `solve_shifted` and the `exp(dz*tau)` prefactor — is therefore NOT exercised by the
  reference bitwise gate. It is covered elsewhere (`tests/test_separable_propagator.py` runs
  at `z_diss_k=1e-4`, and its `test_particle_psi_diagonal_reads_both_separable_terms` exists
  precisely because every particle gate runs at `z_diss_k=0`), but a reference config with a
  nonzero `z_diss_k` would close the gap. A coverage item for later, not for this plan.

## The webgpu reference-vector generators

Added to the L row by the overseer after the first review pass, and done in the follow-up
commit:

- `webgpu/gen_refvectors3d.py` — the `from taranis.propagators import get_propagator` import
  dropped, `prop = get_propagator(kgrid, params)` → `prop = kgrid.lin`.
- `webgpu/gen_refvectors.py` — `kgrid.lin_L[0, 0].real` → `kgrid.lin.L[0, 0].real`.

Both were run end to end (32² 2D and 16²×8 z_spectral, seconds each) and produce their JSON.
**Their output is byte-identical to the same generators run on the Phase-0 base `3073df4`**
(`git archive` copy in a scratch dir, `PYTHONPATH` at that copy, `cmp` clean on both files;
md5 `fecc1d07…` / `338b3e54…` from both trees). The regenerated `webgpu/refvectors.json` and
`webgpu/refvectors3d.json` were restored with `git checkout --` and are NOT part of any
commit here — the tracked assets keep their committed md5s `ecb55bbd…` / `4831609b…`.

Aside, pre-existing and NOT Phase L's: a fresh run on this laptop does not reproduce the
committed assets bit for bit — 17 of 36 keys in the 2D file and 27 of 46 in the 3D one differ
at relative 1e-15…1e-16. The give-away is `fft_input`, a pure `np.sin/np.cos` expression with
no taranis in it, differing by 1 ulp on 84 of 1024 elements: the committed vectors were
recorded on a different machine or library build. The browser self-test compares with
fp32-appropriate tolerances, so this is harmless, but it means those two JSONs are not a
bitwise gate for anything.

Note for anyone rerunning them from a worktree: `python webgpu/gen_refvectors.py` puts
`webgpu/` on `sys.path[0]`, so `import taranis` resolves to the editable install (the shared
main tree), not the worktree. Run with `PYTHONPATH=<worktree>`.

## Consumers OUTSIDE the L ownership row that still need an edit

`bench/step_accounting.py` is **deferred** by the overseer: Phase G is editing it concurrently
(lines 167/260) and these hunks would conflict at rebase. To be applied on `main` after L and
G are both merged. Not reached by `make test` (`testpaths = ["tests"]`) nor by `ruff check .`
(pyflakes does not resolve attributes), so no gate catches it; it breaks at runtime as it
stands. The exact edit list:

- line 196: `(propagators.SeparablePropagator,` → `(propagators.SeparableL,`
- lines 198-199, the `noexpform` lambda's body for that entry:
  `jnp.ones_like(self.sep.dperp)` → `jnp.ones_like(self.dperp)`,
  `jnp.ones_like(self.sep.dz)` → `jnp.ones_like(self.dz)`,
  `jnp.zeros_like(self.sep.kz)` → `jnp.zeros_like(self.kz)`
- line 200: `(propagators.Putzer2Propagator,` → `(propagators.Putzer2Operator,`
- line 204: `(propagators.DiagonalPropagator,` → `(propagators.DiagonalOperator,`
- lines 253-254: `for cls in (propagators.SeparablePropagator, propagators.Putzer2Propagator,
  propagators.DiagonalPropagator, propagators.IdentityPropagator):` →
  `SeparableL, Putzer2Operator, DiagonalOperator, IdentityOperator`

The `*Exp` classes it also monkeypatches (lines 190-192, 251-252) are unchanged, and
`cls.exp_op = ...` / `cls.apply = ...` assignment still works: a `typing.NamedTuple`
subclass is an ordinary class at runtime. `step_accounting.py` reads no `lin_*` slot.

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
`lin_` prefix, as `SeparableL`'s / `Putzer2Operator`'s fields; line 392
"the `propagators.dense_operator` helper exists for tests and `particles/fields.py` only" →
"the operators' `dense()` method exists for …"; line 394 "which is why `dense_operator` of
the separable entries reproduces the dense L bitwise" → "which is why `SeparableL.dense()`
reproduces …".

**docs/performance.md** — line 354 "of `Putzer2Propagator` and re-times" →
"of `Putzer2Operator`". Lines ~456, 795, 817, 820, 838, 845, 1013 name `lin_L`, `lin_m`,
`lin_s2`, `lin_dperp/lin_dz/lin_kz` in measurement tables. These are records of what was
measured; the names now read `lin.L`, `lin.m`, `lin.dperp` etc. Historical `lin_s2` mentions
(pre-Z2) should stay as they are.

**webgpu/SPEC.md** — line 93 "`lsrk_advance` + `DiagonalPropagator.apply_exp`" →
"`DiagonalOperator.apply_exp`". `webgpu/SPEC.md:269` and `webgpu/README.md:1156` use `lin_L`
as the name of a reference-vector JSON entry produced by `webgpu/gen_refvectors.py` (the key
itself is unchanged by this phase — only the taranis-side attribute it is read from moved);
whether that key is renamed is the webgpu side's call, not this refactor's.
