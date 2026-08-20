# MEMORY_PERF_PLAN — memory and step-time reduction, FD-z and z_spectral

Written 2026-08-19 from the two hand-offs it supersedes, `plans/TARANIS_MEMORY_HANDOFF.md`
(XLA buffer audit, HEAD `c71a3cb`) and `plans/ZSPECTRAL_PROPAGATOR_NOTES.md` (step-time
profile + hoisted propagators), plus this session's CB-IMEX and unrolled-loop probe. Those
two files stay as the measurement record; this file is the plan. Execution per the standing
flow (§8): Fable overseer, opus implementers, fresh-Fable adversarial review per phase.
Reviewed 2026-08-19 (`plans/MEMORY_PERF_PLAN_REVIEW.md`, against the buffer dumps); its four
substantive fixes are folded in below (Z2 cutoffs, F2 buffer model + F3-before-F2, Z1
equal-schemes memory gate, Z1 timing gate) plus the Z1 brief notes; its B1-in-Phase-0
recommendation is held as §9.4 (Alfred's call).

## 0. Principles

1. **No new user-facing knobs.** Every change below is an internal improvement that is
   simply better, or a candidate for *removing* a knob. If an implementer finds they want a
   `Parameters` switch, that is a design smell to bring to the overseer, not a decision to
   make. Module-level constants (not `Parameters` attributes) are acceptable for things a
   benchmark needs to vary.
2. **Comments say what the code does, succinctly.** No design history, no "before 2026-08
   this was…", no measurement numbers, no rationale essays in source. Rationale and history
   live here, in `docs/numerics.md` (derivations) and `docs/performance.md` (measurements).
   The adversarial reviewer checks this; a phase is not done while its diff carries
   narrative comments. Existing long comments in files a phase touches are NOT to be
   rewritten wholesale (scope creep) — only the lines the phase adds or changes.
3. **Bitwise expectations are stated per phase and verified, never assumed.** Three
   reference npz files pin solver bits today (`tests/data/`): `particles_gate6_reference_fp{64,32}.npz`
   (2D RMHD, `tests/test_particles_coupled.py` gate 6), `forcing_spinup_reference.npz` (2D,
   `forcing_state` bitwise), `precision_fp32_reference.npz` (fp32 round-off tolerance). A
   phase marked "bitwise" must leave all three green with no regeneration; if it does not,
   that is a bug or a decision for Alfred (§9), not a reason to regenerate.
4. **Units.** `u` = one field-sized complex array = `nz_local·nkx·nky·itemsize` (8 B fp32,
   16 B fp64). The RMHD state is 2 u. Memory = XLA `compiled.memory_analysis()`
   (`temp + arguments + output`) on the jitted `run.block_of_steps`; on GPU additionally the
   device `memory_stats()['peak_bytes_in_use']` delta. ms/step = median wall time of a
   jitted block divided by its step count. Both come from the probe landed in Phase 0.
   The probe's `total_u` is the §-defined temp+args+out; it also reports the breakdown.
   NOTE (Phase 0 finding, confirmed per-case against the laptop128 baseline): the §1
   memory column is 128²×32 `total_u` throughout (reproduced to ≤0.6 u), with two
   exceptions — the FD-z IF parenthetical "26.5 at 64²×16" is a 64²×16 `temp` (26.58;
   the 64²×16 total is 30.96), and the FD-z IMEX row's original 25.0 reproduced under no
   convention at either grid and is corrected to 28.5 in the table. Phase gates are
   checked as DELTAS in `total_u` against the baseline JSON for the same grid, machine
   and precision, not against the absolute numbers quoted here. Witnesses: `bench/memory_probe_laptop_baseline{,_fp64}.json` (64²×16 / 256²) and
   `bench/memory_probe_laptop128_baseline.json` (128²×32 fp32 — the grid the F1/F2/F3
   probe gates are stated at). The probe measures the NON-donated graph (it reuses one state across reps);
   production jits with `donate_argnums=(0,)`, which can alias input to output — u
   numbers describe the probe's graph, consistently at every measurement point.
5. **Measured on CPU, decided on GPU.** The CPU numbers below are structural (how many
   field-sized buffers are simultaneously live) and have held to 2 significant figures
   across grid sizes; the GPU numbers (§5) are what production sizing uses. Anything whose
   time trade depends on the backend (F1's transform chunking, the putzer2 transcendental
   share) is decided from the GPU runs, not the laptop.

## 1. Baseline (where the memory goes)

CPU, `lsrk_scan=True`, no forcing, no particles, single process. 128²×32 fp32 numbers from
the audit (`u` = 2.03 MiB); 64²×16 fp32 numbers from this session's probe agree within the
per-grid constant overhead.

| config | scheme | memory | ms/step (64²×16) |
|---|---|---|---|
| RMHD FD-z | lsrk33 / lsrk54 | 30.5 u (26.5 at 64²×16) | 5.2 / 9.0 |
| RMHD FD-z | imexcb2 / cb3e / cb3c | 28.5 u (Phase 0 correction; the original 25.0 reproduced under no convention) | 4.9 / 7.2 / 6.5 |
| RMHD FD-z | imexcb3f | 58.8 u | 7.5 |
| RMHD z_spectral | lsrk33 hoisted / unhoisted | 43.2 / 39.8 u | 5.9 / 12.2 |
| RMHD z_spectral | lsrk54 hoisted / unhoisted | 62.0 / 39.8 u | 10.4 / 20.2 |
| RMHD z_spectral | imexcb3e | 27.3 u | 9.7 |
| GDI 2D 256² | lsrk33 / lsrk54 | 48.6 / 68.8 u | 4.4 / 6.8 |
| GDI 2D 256² | imexcb2 / cb3e / cb3c | 30.1 u | 4.9 / 6.8 / 6.8 |
| GDI 3D z_spectral | lsrk33 | 42.0 u | 5.0 |
| GDI 3D z_spectral | imexcb2 / cb3e / cb3c | 23.4 u | 6.0 / 7.9 / 8.0 |

(Phase 0 re-measured every ms/step within ±8% except imexcb3f: 10.96, not 7.5 — the
baseline JSON's number is the one to cite.)

What the FD-z 30.5 u is (XLA buffer table, lsrk54):

| u | buffer |
|---|---|
| 8.0 | `gradk(fk)` — 4 fields × 2 components, k-space, materialised by `jnp.stack` + the opaque FFT boundary |
| 7.9 | the same 8 gradients in real space (a real full-grid array costs ~1 u, not 0.5) |
| 2.1 ×2 | halo-padded `(2, nz+4, nkx, nky)` state copies for the z stencil |
| 2.0 ×4 | LSRK `fields`, `delta`, `init_rhs`, NL/FD term temporaries |
| ~2.4 | parameters / aliased output |

z_spectral adds, on top of that: `kgrid.lin_L/lin_m/lin_s2` 6 u permanently resident
(encoding ~`2nz + 2·nkx·nky` independent numbers — `L01=L10=i·kz`, diagonals real and
separable, `s2=−kz²` at ν=η), and with hoisting `4·nstage` full-grid complex ExpOp arrays
(+3.4 u lsrk33, +22 u lsrk54). The z_spectral step-time premium (~1.9× FD-z) is ~34 ms of
putzer2 complex `sqrt/cosh/sinh/exp` per mode per stage out of a ~37 ms gap; transforms are
~3 ms of it.

Two further facts from this session's probe, both to be recorded in docs/performance.md by
Phase 0: (a) `lsrk_scan=False` costs 1.37–1.92× the memory of the scan path for EVERY
stepper (Phase 0 re-measured in probe totals, correcting this session's hand-derived
numbers: FD-z lsrk54 30.96 → 59.58 u, imexcb3e 28.96 → 39.71 u, lsrk33 30.96 → 42.33 u;
these rows are in the laptop baseline JSON) and on the unrolled path `hoist_propagator`
is a no-op on both axes (XLA hoists the literal-gamma stage exponents itself: witness
rows `rmhd_zspec_64x16_lsrk{33,54}_unroll_hoist{1,0}` — 43.33 u / 54.20 u and the same
speed at both settings); (b) the [2R] CB-IMEX
steppers are already the memory-lightest path in every mode and need no IMEX-specific work;
`imexcb3f`'s 2.2× is its unrolled-only [3R] loop (optional fix, §6).

## 2. Targets

| | today | after this plan | floor |
|---|---|---|---|
| RMHD FD-z lsrk33/54 | 30.5 u | ~15–17 u | ~15 u (2 registers + rhs_k + the real-space bracket working set) |
| RMHD z_spectral lsrk54 | 62.0 u | ~27 u | ~21 u |
| RMHD z_spectral lsrk33 | 43.2 u | ~25 u | |
| z_spectral step time | 1.0 (hoisted) / 1.7 (unhoisted, and all of adaptive `cfl_every=1`) | ~1.0 at every step | |

The "after" column is B2 + A + D measured individually in the audit and summed; the FD-z
column adds the halo change (measured buffer, not yet removed). The GPU runs (§5) confirm or
correct these before anyone sizes a production run from them.

## 3. Part F — FD-z

All four phases are internal, knob-free, and apply to 2D and 3D alike (F2 is 3D-only). F1
and F4 also apply to GDI. None touches the propagator. Expected bitwise: F1, F2; expected
bitwise but unverified: F3; F4 is measure-first.

### F1 — per-field gradient transforms (the audit's A)

**What.** `rmhd.grad` / `gdi.grad` today build `fk = stack([phik, psik, vortk, jpark])`,
then `gradk(fk)` = `stack([i·kx·fk, i·ky·fk], axis=1)` (4,2,nz,nkx,nky), then ONE batched
`ifft`. The 8 u k-space stack is materialised (stack is a concatenate; the FFT is an opaque
kernel so its input must be whole) and is live alongside the 7.9 u real-space output.
Replace with one transform per FIELD: for each of `phik, psik, vortk, jpark` (regenerated
from `phik/psik` — `vortk = −ksq·phik` etc. are one elementwise multiply), form the
(2,nz,nkx,nky) gradient and `ifft` it; return `grads` as a **tuple** of four
(2,nz,nx,ny) real arrays instead of one (4,2,…) array. No stack on either side. Peak
k-space gradient memory: 2 u instead of 8 u.

**Why a tuple.** Every consumer already unpacks: `NonlinearTerm` does
`gphi,gpsi,gvort,gjpar = grads`, `bracket(a,b)` indexes `a[0],a[1]`, `set_timestep` reads
`grads[0]`, `grads[1]`. A tuple removes the need for any concatenate of the outputs, which is
the trap the audit's per-component variant would otherwise fall into (8 separate outputs
then `stack`ed = a second 7.9 u copy). Check `particles/fields.py`,
`bench/zspectral_profile.py` and `diagnostics/` for any consumer that relies on `grads`
being an array (`.shape`, `grads[:2]`) and fix those call sites.

**Shared helper.** One function in `shared_physics` (e.g. `grad_fields(fks, kgrid, params)
-> tuple`) taking a tuple of k-space fields; `rmhd.grad` and `gdi.grad` call it. `gradk`
stays for any remaining caller (the bench uses it); if nothing uses it after F1, delete it.

**Transform granularity is a module constant** (`shared_physics.GRAD_CHUNK`, read at
trace time — the probe's `--grad-chunk` sets it). [As landed, overseer-approved
deviation: `GRAD_CHUNK=1` transforms per COMPONENT with a free per-field restack — XLA
writes both component iffts straight into the (2,…) pair buffer, so the "trap" above
only applies to re-stacking ACROSS fields. Literal per-field transforms measured
24.54 u on FD-z 128²×32, missing the ≤24 gate; per-component lands the audit's 23.55.
`GRAD_CHUNK>1` batches that many whole fields per ifft; 1, 2 and 4 are all bitwise
identical, and chunk=4 reproduces the pre-F1 graph byte-exactly.] The GPU run (§5)
compares granularities; if chunk=1 is measurably slower on GPU (> 5%), the fallback is
a larger chunk — not a `Parameters` knob.

**Gates.**
- `grad` output bitwise equal to the old implementation (`np.array_equal` per field) at
  2D 64², FD-z 64²×16, z_spectral 64²×16, fp64 and fp32. The test keeps a private copy of
  the old `stack + batched ifft` path as its reference function (never `git stash` — the
  tree is shared; see memory note).
- All three reference npz tests green unchanged (gate 6 is 2D and goes through `rmhd.grad`).
- `make test` green both precisions.
- Probe: FD-z ≤ 24 u at 128²×32 (audit measured 23.55), z_spectral unhoisted ≤ 35 u.
- Timing: CPU ms/step within noise of baseline; GPU decision per §5.

**Files.** `taranis/physics/shared_physics.py`, `taranis/physics/rmhd.py` (`grad` only),
`taranis/physics/gdi.py` (`grad` only), consumers found by grep, new test in
`tests/test_bracket.py` or a new `tests/test_grad_memory.py`.

### F2 — z stencil without the padded state copy

**What.** `shared_physics.z_derivatives` does `f_padded = concatenate([recv_left, f,
recv_right], axis=1)`. The optimized HLO (review, measured) shows XLA decomposing this
single 3-way concatenate into TWO 2-way concatenates of nz+2 planes each (2.13 u each,
4.26 u total), both `kLoop` fusions with the halo slice fused in — there is no nz+4
buffer and no per-consumer duplication; the two materialised nz+2 halves are the whole
cost. Their metadata places them inside `cond/branch_0_fun` — the `lax.cond` F3 deletes —
so **F3 runs first and F2's design starts from a fresh post-F3 buffer dump.** The
exchange itself is 4 planes; the copies are the cost. Replace with:
- interior planes `2 .. nz−3` from shifted slices of `f` itself (`f[:,4:]`, `f[:,3:-1]`,
  `f[:,2:-2]`, `f[:,1:-3]`, `f[:,:-4]`) — views that fuse into the elementwise stencil;
- the 2 boundary planes at each end from a 6-plane `concatenate([recv_left, f[:,:4]])`
  (and its mirror at the top) — ~0.4 u at nz=32, shrinking with nz;
- output assembled as `concatenate([lo(2), interior(nz−4), hi(2)], axis=1)` — that is the
  2 u output array needed anyway; check in the buffer dump that the interior is not staged
  through its own buffer first (XLA normally writes concat operands in place).
- Both `df_dz` and `d4f_dz4` from the same slices; per plane the identical expression with
  the identical operands, so the arithmetic is unchanged.
- `nz_local ≥ 4` for a non-empty interior; keep the padded path for `nz_local ∈ {2,3}` (the
  existing `width ≤ nz_local` check allows them) as a static python branch.
- `FDLinearTerm`'s `jnp.stack([df_dz[1], df_dz[0]])` is another potential 2 u copy unless
  XLA fuses the swap into the RHS sum; have `z_derivatives` return per-field results, or
  run the stencil on `state.fields[::-1]`, so the swap is free. Confirm in the dump.

**Gates.**
- `z_derivatives` output bitwise equal to the old padded implementation (private reference
  copy in the test) for serial backend at nz_local ∈ {4, 8, 16}, and under the local MPI
  stub for 4 ranks; the existing `tests/test_z_stencils.py` convergence-order and
  `tests/test_halo_width.py` tests green.
- 3D FD-z RMHD solver output bitwise equal to pre-F2 over 20 steps (the test records a
  reference in-process from the private old stencil, NOT a new npz).
- Probe: FD-z drops by ≥ 3.5 u at 128²×32 (both nz+2 concatenate halves gone — they are
  one decomposed concatenate, so the rewrite removes both or neither; a smaller drop means
  the interior slices are being staged through a buffer — find it in the dump before
  closing the phase).
- gate 6 / spinup / precision references: untouched by construction (2D never calls
  `z_derivatives`) — still run.

**Files.** `taranis/physics/shared_physics.py` (`z_derivatives`), `taranis/physics/rmhd.py`
(`FDLinearTerm` only), `tests/test_z_stencils.py`.

### F3 — peel stage 0 out of the LSRK stage scan (the audit's D)

**What.** `_lsrk_scan_stages` carries `init_rhs` across the whole stage scan so
`lax.cond(istage == 0, …)` can reuse it: one extra field vector (2 u) live for the whole
step. Compute stage 0 inline before the scan (`delta = e0.apply(dt·init_rhs)`, fields
update), then scan stages `1..s−1` with no `cond`; `init_rhs`'s liveness ends before the
scan. `_imex2r_scan_stages` is the in-file model (it already peels stage 1). Do NOT fix it
by dropping the reuse (that is an extra RHS per step). The hoisted `exp_ops` stack is then
indexed `[1:]` for the scan and `[0]` for the peeled stage; the unhoisted path forms
`prop.exp_op(gammas[0])` inline — keep the in-scan scanned-`gamma` form for stages
`1..s−1` exactly as now (the memory-light graph `hoist_propagator=False` depends on).

**Bitwise status: expected, unverified.** The per-stage arithmetic is identical, but the
`cond` wrapper disappears and XLA may fuse the stage body differently. Stage 0's
`alpha·delta + dt·rhs` with `alpha=0`, `delta=0` must still be computed as `0·0 + dt·rhs`
(or proven bitwise equal to `dt·rhs` — it is, `0 + x == x` exactly) — state which.

**Gates.**
- `tests/test_hoist_propagator.py` green (hoisted == unhoisted bitwise in every cell).
- `tests/test_scheme_equivalence.py` green (scan vs unrolled at round-off).
- All three reference npz tests. **If gate 6 or the spinup reference changes bits, stop and
  report** the magnitude (§9 decision: accept + regenerate, or drop F3). The hoist work
  already declined a 15-element 1e-23 change for the diagonal path; the same bar applies.
- Probe: −2.0 u on FD-z and z_spectral unhoisted at 128²×32.

**Files.** `taranis/timestepping.py` (`_lsrk_scan_stages`, `lsrk_advance`), `tests/test_hoist_propagator.py`
if the stack indexing needs it.

### F4 — bracket liveness (the audit's E): measure, then maybe

**What.** At peak all 8 real gradient components are live. `NLTerm_vort = bracket(gpsi,
gjpar) − bracket(gphi, gvort)`, `NLTerm_psi = −bracket(gphi, gpsi)` can be ordered so only 6
components are live. **Python statement order does not control XLA liveness** — after F1
the transforms are separate kernels and XLA's scheduler may already realise this; it may
not. [F1 measured: it does not — the 8-component slot did NOT shrink and is now the
dominant term in the 23.55 u FD-z step. `gvort`/`gjpar` are each consumed by exactly one
bracket and could retire early; `gphi` feeds two. Details in F1's scratch notes → to be
folded here when F4 dispatches.] Measure after F1 from the buffer table: if the
8-component real-space slot is already below 6 u, F4 is done with no change. Otherwise try (a) ordering the per-field transforms
in `grad` so `gvort` is produced last and consumed first, (b) computing the brackets inside
`grad`'s successor in that order. Drop F4 if the gain is < 1 u; do not restructure
`NonlinearTerm` for a sub-u gain.

**Gates.** Bitwise on the NL term (the bracket arithmetic is unchanged; only scheduling
moves); reference npz tests; probe delta reported either way.

**Files.** `taranis/physics/rmhd.py` (`grad`, `NonlinearTerm`) — same owner as F1, run after it.

## 4. Part Z — z_spectral

Z1 is the design change; Z2 is the fallback path's cleanup; Z3 is the knob decision. FD-z
and 2D use the diagonal backend and are untouched by all three (state this in the tests:
an FD-z and a 2D run must be bitwise identical before/after each Z phase).

### Z1 — Elsasser-separable propagator backend (the audit's B2, the notes' recommendation)

**Math.** RMHD z_spectral with ν = η (`eqpars["diss"]` entries equal — compare as python
floats at setup), hyper h, `z_diss_k` = zd (default 0):

```
L = d·I + i·kz·σx,   d(k⊥,kz) = −ν·k⊥^{2h} − zd·kz⁴ = dperp(k⊥) + dz(kz)     (real)
exp(L·τ) = e^{dperp·τ}(k⊥) · e^{dz·τ}(kz) · [cos(kz·τ)·I + i·sin(kz·τ)·σx]
(I − a·L)⁻¹ = [(1 − a·d)·I + i·a·kz·σx] / ((1 − a·d)² + a²·kz²)
```

Exact (`[d·I, i·kz·σx] = 0`), not a splitting. The `solve_shifted` denominator
`(1 − a·d)² + a²·kz²` is bounded below by 1 for `a > 0` since `d ≤ 0` (review, verified):
the separable backend has NO pole, unlike putzer2's `det(I − a·L) = 0` at `λ = 1/a` —
state this in docs/numerics.md, it retires one of the two documented NaN traps in
`propagators.py` for this path. `kz` here is `rmhd._kz_deriv` — the
Nyquist-zeroed kz — used for BOTH the off-diagonal and the `kz⁴` term, exactly as
`linear_matrix` does today, so the dense reconstruction equals the current L to the bit.
For ν ≠ η the diagonal blocks differ and the constant rotation no longer diagonalises —
that case (and GDI, whose L is genuinely full-grid) stays on putzer2.

**Storage.** Three small kgrid entries instead of `lin_L/lin_m/lin_s2` (6 u): `lin_dperp`
(nkx,nky) real, `lin_dz` (nz,1,1) real, `lin_kz` (nz,1,1) real. Per stage the propagator
forms `P = exp(dperp·τ)` (nkx,nky), `c = exp(dz·τ)·cos(kz·τ)` and `s = exp(dz·τ)·sin(kz·τ)`
(nz,1,1): no full-grid array anywhere, so there is nothing to hoist — `hoistable = False`,
`stage_exp_ops` returns None, and `params.hoist_propagator` becomes irrelevant on this path.
Apply: `out0 = P·(c·a0 + i·(s·a1))`, `out1 = P·(i·(s·a0) + c·a1)`, with the multiplication by
`i` written as the real swap it is (no complex multiply). Fix this op order in the class and
keep `apply_exp(arr,τ) == exp_op(τ).apply(arr)` bitwise (the existing contract test).

**Plumbing.**
- `propagators.py`: `SeparablePropagator` (`scaled`, `exp_op` → `SeparableExp(P, c, s)`
  pytree with `.apply`, `apply_exp`, `solve_shifted`, `apply_L`, `hoistable=False`); a
  `SeparableL` NamedTuple `(dperp, dz, kz)` that a recipe may return from
  `linear_matrix_func` instead of a dense array; `linear_fields` accepts both (dense →
  current path; `SeparableL` → the three entries; the Hermitian-compatibility check is
  done analytically for the separable form or plane-by-plane — NOT by materialising the
  dense operator in host numpy and passing it through `_check_hermitian_compatible`,
  whose `np.roll(np.flip(...))` would transiently double a ~4 u host array (~3.2 GB at
  512²×128 fp64) in the setup path of a memory-reduction plan);
  `get_propagator` dispatches on which entries are populated; a
  `dense_operator(kgrid)` helper that materialises `(2,2,nz,nkx,nky)` L from any backend,
  for tests and `particles/fields.py` only (it is 4 u and must never be called in a step).
- `grids.py`: three new Optional `K_Grids` fields (`kgrid_specs` replicates them like
  `lin_*`; z_spectral is size-1 so sharding never sees them).
- `physics/rmhd.py`: `linear_matrix` returns `SeparableL` when `z_spectral and diss[0] ==
  diss[1]`, else the dense array as now. The recipe signature does not change (no new
  registry hook).
- `particles/fields.py::_psi_non_ideal` reads `rmhd.linear_matrix(...)` as an array —
  route it through the ψ diagonal of `SeparableL` (`dperp + dz`, materialised once there)
  without changing its numbers (gate 7 and the 3D particle gates pin it).
- `timestepping.stage_exp_ops`: unchanged in logic (`hoistable` gate); verify it returns
  None for the new backend.

**Bitwise/tolerance expectations.** z_spectral RMHD ν=η results change at round-off versus
putzer2 (measured 5e-14 relative over a step). Nothing pins them bitwise today except
`test_hoist_propagator`'s hoisted==unhoisted cells, which become trivially identical on this
backend (both unhoisted). That test's `test_stage_exp_ops_structure` and the putzer2 hoist
cells must keep exercising putzer2 — add a ν ≠ η config (e.g. `diss=(1e-4, 2e-4)`) so the
putzer2 hoist path stays under test. IMEX on z_spectral RMHD is not a production path (wave
damping, CLAUDE.md) but `solve_shifted`/`apply_L` must be right: test against
`dense_operator` with the putzer2 `solve_shifted` at 1e-13 relative.

**Gates.**
- New `tests/test_separable_propagator.py`: (i) backend selection (ν=η z_spectral →
  separable; ν≠η → putzer2; FD-z/2D → diagonal — assert `kgrid.lin_L is None` on the
  separable path and the three entries populated, and the converse); (ii)
  `SeparableExp.apply` vs putzer2 `Putzer2Exp.apply` on the dense operator at 1e-13 rel
  (fp64) and 1e-5 (fp32) for τ over 4 decades including τ = 0; (iii) `solve_shifted`,
  `apply_L` same; (iv) `exp_op(τ).apply == apply_exp` bitwise; (v) reality preserved: a real
  IC stays real after 20 steps (`ifft` imaginary part at round-off) — the Nyquist-kz rule;
  (vi) `dense_operator` of the separable entries equals `rmhd.linear_matrix`'s dense form
  built with ν≠η logic forced to ν=η, bitwise; (vii) FD-z and 2D solver output bitwise
  identical before/after (reference computed in-process on the same tree by forcing the
  dense path — e.g. a module-level switch the test flips — or from a snapshot recorded at
  phase start; state which).
- `tests/test_z_spectral.py`, `test_hoist_propagator.py` (with the ν≠η addition),
  `test_imex.py`, `test_linear_propagator.py`, `test_dissipation.py`,
  `test_precision_dtypes.py`, `test_time_order.py`, particle 3D gates: green. Any test that
  read `kgrid.lin_L` for RMHD z_spectral switches to `dense_operator`.
- Probe (the review's two-sided gate, restated on the post-F1 floor): after Z1 nothing
  scheme-dependent is stored, so z_spectral lsrk33 and lsrk54 must land on the SAME
  number — equal to each other within 0.1 u, and at (post-F1 unhoisted total − the
  `lin_*` args block) ± 0.5 u = **29.1 ± 0.5 u at 64²×16** (35.10 − 6.05; 128²×32:
  34.92 − 6.0 ≈ 28.9). Catches a partial `lin_*` removal and a stray per-stage live
  array. `kgrid` bytes drop by 6 u. (Original pre-F1 statement was 33.8 ± 0.5; F1
  landed −4.9 u first.)
- Timing: z_spectral lsrk33 fixed-dt step ≤ 1.0× the hoisted putzer2 step on CPU and
  ≤ 0.85× on GPU (per stage, hoisted putzer2 streams 8 u — 4 u ExpOp + 2 in + 2 out —
  where separable streams 4 u; a GPU result > 1.0× means the apply is materialising an
  intermediate, e.g. `c·a0` before the `P·` multiply — find it, don't accept it), and
  adaptive `cfl_every=1` ≤ 0.7× its current value (`bench/zspectral_profile.py`).
- HLO check (as the hoist work did): no `cosh`/`sinh`/complex `sqrt` in the z_spectral RMHD
  step; the only transcendentals are one real exp over (nkx,nky) and exp/cos/sin over (nz)
  per stage.

**Files.** `taranis/propagators.py`, `taranis/grids.py` (K_Grids + `_attach_linear_operator`),
`taranis/physics/rmhd.py` (`linear_matrix` only — F1 owns `grad`; sequence F1 before Z1 or
give Z1 the file after F1 lands), `taranis/particles/fields.py` (`_psi_non_ideal` only),
tests listed.

### Z2 — cheaper putzer2 coefficients (the audit's B3)

**What.** `Putzer2Propagator._coeffs` evaluates complex `sqrt(s2·τ²)`, `cosh`, `sinh` and
`exp(m·τ)` per mode per stage: 10 exp + 5 cos + 5 sin + 7 sqrt + 9 div after lowering.
Store `s = sqrt(s2)` at setup (replace `lin_s2` by `lin_s`; `s2 = s·s` where the Taylor
branch needs it — a 1-ulp change in the small-|z| branch only), and per stage form
`w = exp(s·τ)` once: `cosh = (w + 1/w)/2`, `sinh/s = τ·(w − 1/w)/(2·s·τ)`, `pref = exp(m·τ)`
— 2 complex exps + 1 complex reciprocal per mode per stage. Keep the Taylor branch and the
overflow note, and **raise both Taylor cutoffs ~100× in |z²|** (fp32 to |z²| < 1e-2, fp64
to |z²| < 1e-4) in the SAME commit as the coefficient form: the `w − 1/w` form loses ~2
digits to cancellation that grows as z shrinks (review, measured: 1.2e-5 rel on sinh(z)/z
at the old fp32 cutoff |z| = 1e-2 — over the 1e-5 gate; 8.8e-7 at |z| = 0.1), while the
existing 4-term series truncates at |z|⁸/362880 = 2.8e-14 at |z| = 0.1, covering the
widened branch with room to spare. This is the
fallback path: ν ≠ η RMHD and GDI-IF (GDI's production CB-IMEX path never forms an
exponential and is unaffected). It is the lever for adaptive `cfl_every=1` on those paths,
which hoisting cannot touch.

**Gates.**
- New test: new `_coeffs` vs a private copy of the old `_coeffs` over a sweep of `(m, s2,
  τ)` covering real/imaginary/complex `s2`, the (widened) Taylor branch and its boundary
  — sample AT the new cutoff on both sides, the worst point for the w-form — and
  |Re(sτ)| up to ~600 (fp64) / 80 (fp32): 1e-13 relative (fp64), 1e-5 (fp32); continuity
  across the branch threshold.
- `test_hoist_propagator.py` (hoisted == unhoisted still bitwise on putzer2: both paths go
  through the same `_coeffs`), `test_gdi_linear.py` (8 `lin_*` reads → `lin_s`),
  `test_linear_propagator.py`, `test_imex.py`.
- Timing: GDI 2D 512² lsrk33 adaptive `cfl_every=1`, and z_spectral ν≠η RMHD, before/after
  (`bench/zspectral_profile.py` extended with a `--gdi` case or a small new bench): target
  ≥ 0.75× the current step (the ablation ladder says the transcendentals were 28 of 34 ms).
- Memory unchanged (probe).

**Files.** `taranis/propagators.py` (`Putzer2Propagator`, `putzer2_precompute`,
`linear_fields`), `taranis/grids.py` (`lin_s2` → `lin_s`), tests listed. Sequence after Z1
(same file).

### Z3 — the `hoist_propagator` knob: keep or delete

After Z1 the knob is dead for ν=η RMHD; after Z2 its remaining job (putzer2 on GDI-IF and
ν≠η RMHD at frozen dt) is partly done by Z2 at zero memory. **Decision criterion:** on the
GPU (§5) and CPU timing, if the Z2 putzer2 step at `hoist_propagator=False` is ≥ 0.9× the
hoisted step on GDI-IF, delete the knob and its machinery: `params.hoist_propagator`,
`timestepping.stage_exp_ops`, `propagators.stack_exp_ops`, the `exp_ops=` kwarg on every
stepper, `run._hoisted_exp_ops`/`_fixed_dt`, `tests/test_hoist_propagator.py` (its
contract test `exp_op == apply_exp` moves to the separable/putzer2 tests), and the
`hoistable` attribute. `Parameters.from_snapshot` must warn-and-ignore a recorded
`hoist_propagator` key (records written 2026-08-19 onward carry it). If the criterion
fails, the knob stays, documented as "putzer2 + frozen dt only". **This is Alfred's call
(§9) with the numbers in hand; the implementer prepares both the measurement and the
deletion diff but does not merge the deletion without the decision.**

## 5. Part G — GPU assessment

The CPU buffer structure is indicative; production sizing needs GPU numbers, and two
decisions (F1's transform granularity, Z3) depend on GPU timing. Single-GPU paths are
what this plan optimises (z_spectral and particles are size-1 by construction; FD-z is also
run single-GPU); the `comm_backend="jax"` multi-GPU FD-z row is the one sharded check.

### G0 — the probe (Phase 0 deliverable, lands with the baseline)

`bench/memory_probe.py` — the driver for everything below. Contract:
- CLI: `python bench/memory_probe.py [--profile laptop|p100|gtx2080|a100] [--precision-check]
  [--cases ...] [--out results.json]`; and `main(**kwargs)` with the same options, so the
  lugus kernel can call it (`launch.py` dispatches `main(**kwargs)` when there is no
  `make_data`). Writes the JSON into the current directory (lugus zips the driver's cwd)
  and prints the table to stdout (lands in the kernel log).
- Per case: `Parameters` kwargs, scheme, `lsrk_scan`, `hoist_propagator`; reports
  `memory_analysis` temp/arg/out in bytes and u, `jax.local_devices()[0].memory_stats()`
  peak delta across the block (GPU only), ms/step median of N blocks, and `jax`/backend
  version. An OOM is a valid result row (`"oom": true`), not a crash of the matrix —
  catch `XlaRuntimeError` per case.
- Profiles (fp32 unless noted; u at 512²×128 fp32 = 135 MB):
  - `laptop`: the §1 matrix at 64²×16 / 256² (seconds per case) — the regression table.
  - `p100` (16 GB): RMHD FD-z 512²×128 {lsrk33, lsrk54, imexcb3e} × scan/unrolled; RMHD
    z_spectral 512²×128 {lsrk33, lsrk54} × hoist on/off (lsrk54 hoisted = 8.4 GB — expected
    to fit; an OOM is the result) + imexcb3e; GDI 2D 1024² {lsrk33, imexcb3e}; GDI 3D
    256²×64 {imexcb3e}; plus fp64 of the RMHD rows at 256²×64 via a second launch with
    `--precision 64`. Also the F1 granularity pair (module constant flipped via a
    `--grad-chunk` option that sets it before import) and adaptive `cfl_every=1` timing for
    z_spectral lsrk33.
  - `gtx2080` (11 GB): same rows one size down where needed (512²×64 for z_spectral
    lsrk54 hoisted; 512²×128 FD-z should fit) plus the 4-GPU `comm_backend="jax"` FD-z
    row at 512²×128 (per-GPU memory; the audit's unmeasured case).
  - `a100`: the p100 profile at 1024²×128 when a card is available (Savio A40/A5000 pool
    pends ~a week; not a blocker).
- `comm_backend="serial"` explicitly on single-GPU rows so Kaggle and Savio run the same
  code path (Savio has mpi4py, so the auto-resolution would pick mpi4jax there).

### G1 — Kaggle P100 via lugus (single-GPU paths)

From the taranis checkout (driver must be git-tracked — lugus refuses untracked files):

```bash
python ../lugus/launch.py run bench/memory_probe.py --entry-kwargs '{"profile": "p100"}' --precision 32 --kernel-slug memory-probe-fp32
```

```bash
python ../lugus/launch.py run bench/memory_probe.py --entry-kwargs '{"profile": "p100"}' --precision 64 --kernel-slug memory-probe-fp64
```

[Baseline point run 2026-08-19: both precisions complete, 14 cases each, zero errors,
zero OOMs (hoisted z_spectral lsrk54 fits the P100 at 9.2 GB); JSONs committed as
`bench/memory_probe_p100_baseline_fp{32,64}.json`, table and GPU-only findings in
docs/performance.md. postF/postZ points pending.]

`--dry-run` first to inspect the staged upload. Output: `output/memory-probe-fp32/…zip`
containing `results.json`; the table is also in the kernel log. Budget: the p100 profile
should complete in well under an hour (each case is compile + a few 10-step blocks); if a
run needs bounding, pass `nblock`/`nrep`/`timeout` in `--entry-kwargs` — those are real
`main()` parameters; `main()` rejects unknown kwargs, so a `wall_budget` key would
TypeError in the kernel. Run G1 at three
points, each with the full profile: **baseline** (Phase 0, before any change — this is the
table docs/performance.md cites), **after Part F** (this run carries the F1 granularity
pair), **after Part Z** (this run carries the Z3 hoist on/off pair on GDI-IF).

### G2 — Savio GTX 2080Ti

`slurms/memory_probe_2080.sh` (Appendix A; Phase 0 lands it as a file, parametrised like
the existing `bench_phase3_2080_scale.sh`): one single-GPU task for the `gtx2080` profile,
then one 4-GPU `srun` for the sharded FD-z row. Account/QoS/env blocks are copied from
`slurms/bench_phase3_2080_scale.sh`, which is known to work on that pool (the nvidia-libs
`LD_LIBRARY_PATH` block, `PYTHONNOUSERSITE`, `NCCL_P2P_DISABLE=1`, `RMHD_REQUIRE_GPU=1`).
Same three measurement points as G1. The 11 GB card is the interesting one: it is where
today's hoisted lsrk54 z_spectral row at 512²×128 OOMs (8.4 GB + CUDA context + workspace)
and where after Z1 it should fit — record that flip explicitly.

[Baseline COMPLETE 2026-08-19, job 37775868 (post env-scrub fix, tree f1e343d): all
single-GPU cases ran; the predicted OOM recorded — hoisted z_spectral lsrk54 at 512²×128
fp32 OOMs the 11 GB card, its 512²×64 twin and the unhoisted path fit; sharded row
reproduced exactly. JSONs committed as bench/memory_probe_gtx2080_baseline_*.json,
table in docs/performance.md. postF/postZ points pending.]

[First attempt 2026-08-19, job 37774831: the 4-GPU sharded FD-z row WORKED and is
rank-consistent (24.76 u per device, peak 32.76 u, 75.4 ms/step at 512²×128 fp32), but
every single-GPU isolated case died — the isolation subprocess inherits the parent srun's
PMIx env and `import taranis` runs MPI_Init at import (`_mpi_compat`'s module-scope
mpi4py import, regardless of comm_backend), which aborts in a child joining a live PMIx
context. Fixed in the probe (`_run_case_isolated` scrubs PMI_/PMIX_/OMPI_/SLURM_/
MPI4JAX_/MV2_ from the child env, matching `_LAUNCHER_SIZE_VARS`). Needs one
resubmission at TAG=baseline for the single-GPU tables; the sharded row is in hand.]

### What the GPU numbers decide

- F1 granularity (per-field vs two-field): from the post-F timing pair on P100 (and
  2080Ti). Rule: per-field unless > 5% slower on both cards.
- Z3 knob deletion: from the post-Z GDI-IF hoist on/off pair.
- Whether the putzer2 transcendental share is as large on GPU as on CPU (it may not be —
  GPU transcendentals are cheap next to memory traffic). This does not change the plan
  (Z1 is a memory win regardless) but changes what docs/performance.md says about speed.
- Production grid sizing per card: the "after" column of §2 in bytes, with OOM boundaries.

## 6. Not planned, and why

- **B1 (default `hoist_propagator=False`)**: a stopgap that Z1 makes moot for RMHD; flip it
  only if Z1 slips past the next production z_spectral run (§9).
- **C (broadcastable `L` blocks)**: Z1 removes `lin_*` for the case that matters; for GDI
  the blocks are genuinely full-grid (would save ~1.5 u of real-ness at most); API churn
  for little.
- **2-entry `Putzer2Exp`, generic eigen backend, real-trig branch**: all dominated by Z1 for
  RMHD and not applicable/not better for GDI.
- **Hoisting the IMEX 2×2 inverse, Elsasser-diagonal `solve_shifted`**: the solve is
  ~0.06 ms/stage on the diagonal backend and ~20% of an RHS on putzer2; IMEX on z_spectral
  RMHD is not a production path; hoisting the inverse is the same memory trap as exp
  hoisting for a smaller gain. Z1's `solve_shifted` is already the diagonal form.
- **FSAL reuse in CB-IMEX**: `b_s ≠ 0` in every tableau (1/6, 1/2, …), so `u_s ≠ x_{n+1}`
  and the last explicit derivative cannot seed the next step; the "(FSAL)" in the
  `timestepping.py` IMEX header is loose — fix the comment in passing (Z2 owner), no code.
- **Scanned `imexcb3f`** (58.8 → ~29 u): optional, only if anyone runs cb3f (cb3e is the
  recommended default). Not in scope unless Alfred asks.
- **`lsrk_scan=False` as a GPU default**: contradicted by the memory cost (1.5–2.4×) until
  a GPU measurement shows a speed gain that pays for it; G1/G2 measure it, the doc records
  it, no code change.

## 7. Phase order and ownership

Phases are small; sequence them so shared files have one owner at a time.

| # | phase | files (owner) | parallel with | bitwise |
|---|---|---|---|---|
| 0 | probe + baseline + slurm script + docs stub; G1/G2 baseline runs | `bench/memory_probe.py`, `slurms/memory_probe_2080.sh`, `docs/performance.md` (new section, overseer owns the text) | — | n/a |
| 1 | **F1** grad tuple | `shared_physics.py` (grad helper), `rmhd.py` (`grad`), `gdi.py` (`grad`), consumers (grep `bench/`, `particles/`, `diagnostics/` for array-`grads` uses), test | — (serialised before Z1: both touch `rmhd.py`, F1 is small) | yes |
| 1' | **Z1** separable backend | `propagators.py`, `grids.py`, `rmhd.py` (`linear_matrix`), `particles/fields.py`, tests | — (after F1) | z_spectral round-off; FD-z/2D bitwise |
| 2 | **F3** peel stage 0 | `timestepping.py` | **Z2** (`propagators.py`, `grids.py` `lin_s`) | expected; verify |
| 2' | **Z2** cheap putzer2 coeffs | `propagators.py`, `grids.py`, tests | F3 | putzer2 round-off |
| 3 | **F2** halo-free stencil (design from a fresh post-F3 dump — the nz+2 concats sit inside the cond F3 deletes) | `shared_physics.py` (`z_derivatives`), `rmhd.py` (`FDLinearTerm`), `test_z_stencils.py` | — | yes |
| 4 | **F4** bracket liveness | `rmhd.py` (`grad`/`NonlinearTerm`) | — | yes / measure-first |
| 5 | G1/G2 post-F and post-Z runs; **Z3** decision prep | probe outputs; deletion diff staged, not merged | — | — |
| 6 | docs sweep (§8), move the two hand-off notes to `plans/old/` | CLAUDE.md, docs | — | — |

Each phase: implementer brief fixes the interfaces above (tuple `grads`, `SeparableL`
field names, `lin_s`, the probe JSON schema) so parallel agents do not collide; commit
before dispatching the adversarial review (the reviewer mutates files to test that gates
have teeth); review hunts for: narrative comments (§0.2), any new `Parameters` attribute,
`lax.cond` on static config, `kgrid.lin_*` read from a stepper, bitwise claims without a
test, a gate that passes when the change is reverted.

## 8. Docs to update (overseer owns the text; implementers draft to scratch)

- `docs/performance.md`: new section "Memory: where it goes and what was removed" with the
  §1 table (CPU) and the G1/G2 tables (P100, 2080Ti) at baseline / post-F / post-Z; the
  `lsrk_scan=False` memory cost in "Tuning knobs, measured"; the z_spectral premium entry
  updated (separable backend); "Known, not done" pruned (the `sqrt(s2)` cache item is Z2);
  "Production guidance" gets the per-card grid sizes.
- `docs/numerics.md`: the separable propagator derivation (§4 Z1 math, including why it is
  exact and the ν≠η limit), the new `_coeffs` form, the `grads` tuple convention, the stencil
  assembly.
- `CLAUDE.md`: "Hoisted stage propagators" paragraph rewritten for the post-Z1 world
  (backend table: diagonal / separable / putzer2, selection rule, what is hoistable); the
  `grads` tuple contract under "Term funcs"; the `z_derivatives` interior/boundary rule;
  the `K_Grids` entries; whichever way Z3 goes.
- `plans/README.md`: this plan under Live; the two hand-offs to Finished when §7 row 6 runs.
- `bench/zspectral_profile.py`: the `"grad (gradk+ifft)"` label is stale post-F1 (it times
  the tuple path) — rename in the sweep, it is outside any phase's ownership.

## 9. Decisions for Alfred (raise when reached, with numbers)

1. F3 if gate 6 / spinup reference bits change: accept and regenerate (on the post-F3 tree,
   stating so in the npz generator header) or drop F3 (2 u).
2. F1 granularity if the GPU pair disagrees between cards.
3. Z3: delete the hoist machinery or keep it (criterion in §4 Z3).
4. B1 stopgap if Z1 is not landed before the next production z_spectral run.
   [2026-08-19, Alfred: keep as written — B1 stays deferred; the review's
   land-it-in-Phase-0 recommendation declined.]
5. Whether to scan `imexcb3f` (only if cb3f is in use).

## Appendix A — Savio GTX 2080Ti job script (`slurms/memory_probe_2080.sh`)

SUPERSEDED by the landed `slurms/memory_probe_2080.sh` (Phase 0), which additionally
passes `--precision` with `--precision-check` (the sketch below passes only the check
flag, which no-ops without `--precision`), adds the F1 grad-chunk pair and `--tag`.
Kept for the copied-from provenance: blocks are from `slurms/bench_phase3_2080_scale.sh`
(working on this pool as of 2026-07). `--gres` is per node; the 2080Ti FCA pool is 2:1
CPU:GPU.

```bash
#!/bin/bash
#SBATCH --job-name=memprobe_2080
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:4
#SBATCH --time=01:30:00
#SBATCH --output=memprobe_2080_%j.out
#SBATCH --error=memprobe_2080_%j.err
#
# Memory/step-time probe on one GTX 2080Ti (11 GB), then the 4-GPU sharded FD-z row.
# Profile and output paths are the only things to edit; TAG labels the run point
# (baseline / postF / postZ).

set -uo pipefail

module purge
module load anaconda3 gcc openmpi
source activate jax_gpu
unset PYTHONPATH
export PYTHONNOUSERSITE=1
NVLIBS=$("$HOME/.conda/envs/jax_gpu/bin/python" -c "import nvidia,os;print(':'.join(os.path.join(p,d,'lib') for p in nvidia.__path__ for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p,d,'lib'))))" 2>/dev/null || true)
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_P2P_DISABLE=1
export RMHD_REQUIRE_GPU=1
export MPI4JAX_USE_CUDA_MPI=${CUDA_MPI:-0}
MPI_MODE=${MPI_MODE:-pmix}

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/taranis
PROBE=$REPO/bench/memory_probe.py
TAG=${TAG:-baseline}
OUT=$SLURM_SUBMIT_DIR/memprobe_2080_${TAG}_$SLURM_JOB_ID
mkdir -p "$OUT"
echo "python=$PY tag=$TAG out=$OUT"

# 1 GPU, fp32: the gtx2080 profile (single-process, comm_backend=serial inside the probe)
export TARANIS_PRECISION=32
srun --mpi=$MPI_MODE --ntasks=1 --gres=gpu:GTX2080TI:1 --gpu-bind=single:1 \
     "$PY" -u "$PROBE" --profile gtx2080 --out "$OUT/gtx2080_fp32.json" 2>&1 | grep -v "bit precision"

# 1 GPU, fp64: the RMHD rows one size down
export TARANIS_PRECISION=64
srun --mpi=$MPI_MODE --ntasks=1 --gres=gpu:GTX2080TI:1 --gpu-bind=single:1 \
     "$PY" -u "$PROBE" --profile gtx2080 --precision-check --out "$OUT/gtx2080_fp64.json" 2>&1 | grep -v "bit precision"

# 4 GPUs, fp32: the sharded FD-z row (comm_backend=jax; per-GPU memory is what matters)
export TARANIS_PRECISION=32
srun --mpi=$MPI_MODE --ntasks=4 --ntasks-per-node=4 --gres=gpu:GTX2080TI:4 \
     "$PY" -u "$PROBE" --profile gtx2080-sharded --out "$OUT/gtx2080_fp32_jax4.json" 2>&1 | grep -v "bit precision"

echo "done: $OUT"
```

The probe's `gtx2080-sharded` profile constructs `Parameters(comm_backend="jax", ...)` as
the first jax device work in the process (CLAUDE.md rule) and reports per-rank
`memory_stats()` peaks; all other profiles are single-process.

## Appendix B — the probe's case schema (so implementers and the docs agree)

```
{ "tag": "baseline", "device": "NVIDIA Tesla P100-PCIE-16GB", "jax": "0.10.0",
  "precision": "32", "cases_filter": null, "cases": [
    { "label": "rmhd_fdz_512x512x128_lsrk54_scan",
      "params": {"nx":512,"ny":512,"nz":128,"dims":3,"eqtype":"RMHD","z_spectral":false,
                 "lsrk_scan":true,"hoist_propagator":true,"adaptive_timestep":false,"dt":1e-3,
                 "comm_backend":"serial","eqpars":{"diss":[1e-4,1e-4],"hyper":2}},
      "scheme": "lsrk54", "u_bytes": 135266304,
      "mem_analysis": {"temp": ..., "args": ..., "out": ..., "code": ...},
      "total_bytes": ..., "total_u": 30.5,
      "device_peak_bytes": ..., "device_peak_u": ...,
      "ms_per_step": ..., "nblock": 10, "nrep": 5, "oom": false } ] }

(`total_u` is a sibling of `mem_analysis`, as the landed probe writes it; the header
`cases_filter` records any `--cases` subset, null for a full profile.)
```

## Appendix C — the numbers behind the floor (FD-z, 2-register IF-LSRK)

Live at the RHS peak: `fields` + `delta` (4 u), `rhs_k` (2 u — FFT output is opaque, cannot
fuse into the `delta` update), the real-space bracket working set with optimal ordering
(6 gradient components + 2 accumulators ≈ 8 u; a real full-grid array is ~1 u), one k-space
transform scratch (~1 u), 4 halo planes (≪ 1 u): ≈ 15 u. After F1–F4 the remaining gap to
it is the `rhs`-assembly and output-aliasing slack.
