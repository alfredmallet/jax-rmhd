# z_spectral propagator: profile, hoisting, and the options left on the table

Session notes, 2026-08-19 (Alfred + Claude). Hand-off for the session that combines this
with the separate memory-usage study. Everything measured on the M1 laptop, macOS 14, jax
0.10.0, CPU, fp64, RMHD, lsrk33 unless stated, 128²×16 (nz = 16, Lz = 2π), fixed dt = 1e-3,
elsasser forcing, `nblock=20`. The machine was loaded (load avg ~6) for the later runs, so
compare numbers within a table, not across tables. Full write-up with every number:
docs/performance.md "Where the z_spectral step's extra time goes"; rules in CLAUDE.md
("Hoisted stage propagators"); `exp_op` contract in docs/numerics.md.

## 1. The question and the answer

**Q:** why is a `z_spectral=True` step ~1.9× a finite-difference-z step at the same grid?
The B2 particle benchmark had guessed "rfftn over (z,x,y) vs rfft2 per plane".

**A:** the transforms are ~3 ms of a ~37 ms gap. **~34 ms is the putzer2 2×2 matrix
exponential**, 28 ms of it the complex `sqrt`/`cosh`/`sinh` of `Putzer2Propagator._coeffs`,
re-evaluated per mode per stage. Ablation inside the real scanned step (each row drops one
piece; deliberately wrong numerics):

| variant | ms/step |
|---|---|
| production putzer2 | 79.8 |
| − complex sqrt/cosh/sinh | 51.9 |
| − exp(m·tau) too | 48.1 |
| − the 2×2 apply too (identity) | 45.9 |
| FD-z step | 42.9 |

Why the complex transcendentals cost that much (optimized HLO): `cosh(z)` lowers to
`(exp(z)+exp(−z))/2` with each complex exp as `exp(x)(cos y + i sin y)` → 4 exp + 2 cos +
2 sin; `sinh` the same again, nothing shared; complex `sqrt` is 7 sqrt + 9 div + 18 selects;
`exp(m·tau)` 2 exp + cos + sin. Total **10 exp + 5 cos + 5 sin + 7 sqrt + 9 div per mode per
stage** against the 2 complex exps the math needs (or 1 real exp + cos + sin when `m` is real
and `s²` real one-signed, as for RMHD). Same operator shape and mode count as 2D GDI at 512²
on an IF scheme; GDI's IMEX path is rational and unaffected.

Traps met on the way (both documented): `apply_exp` timed in isolation lies in both
directions (kgrid closed over → constant-folded to 0.3 ms; passed as arg → 19.6 ms, an
over-count); monkeypatch ablations need `jax.clear_caches()` or every variant silently
re-reports the baseline. Tool: `bench/zspectral_profile.py` (new; full step both knob
settings, isolated pieces, operator-swap attribution, ablation ladder, nz sweep).

## 2. What was implemented: hoisted stage propagators

`params.hoist_propagator` (default True, saved/restored). When dt is frozen over a block
(fixed dt, or one `cfl_every` block) `run.py` forms every IF stage's `exp(L·tau)` ONCE per
block and the stepper only applies it.

- `propagators.py`: `exp_op(tau)` on every propagator returns an `ExpOp` pytree
  (`Putzer2Exp` = the four entries m00..m11, `DiagonalExp`, `IdentityExp`) with `.apply(arr)`;
  `apply_exp(arr,tau)` IS `exp_op(tau).apply(arr)`; `stack_exp_ops` stacks a tuple for scan
  xs; `hoistable` class attribute — **True only for putzer2**.
- `timestepping.py`: `stage_exp_ops(kgrid, params, scheme, stepper, dt)` → tuple per stage
  (lsrk: `scaled(dt).exp_op(gamma_s)`, rk44: `(exp_op(dt/2), exp_op(dt))`, IMEX/diagonal/
  identity/knob-off: None). Every stepper takes `exp_ops=None` (IMEX accepts and ignores).
  Unhoisted lsrk keeps the exponent INSIDE the stage scan with the scanned `gamma` — the
  legacy graph, deliberately: XLA's loop-invariant code motion hoists anything formed outside
  the stage scan with a literal gamma (verified in the HLO), so "off" must mean memory-light.
- `run.py`: `_hoisted_exp_ops` / `_fixed_dt`; ops formed outside the step scan in
  `block_of_steps`, `_cfl_block` and both particle variants.
- Tests: `tests/test_hoist_propagator.py` — hoisted == unhoisted bitwise at fp64 in every cell
  (2D/FD-z/z_spectral × lsrk33/54/rk44 × fixed/cfl_every=2/adaptive × scan/unrolled); fp32
  round-off (one 1-ulp fusion cell: z_spectral lsrk54 cfl_every=2 scan); IMEX independent;
  structure; save round-trip. `make test` green, both precisions (220 + 197 passed).
- Docs: CLAUDE.md, docs/numerics.md, docs/performance.md.

Result (same session, loaded machine):

| z_spectral 128²×16 | unhoisted | hoisted |
|---|---|---|
| fixed dt, lsrk33 | 78.7 | **48.7** (0.62×) |
| fixed dt, lsrk54 | 146.2 | **81.5** (0.56×) |
| fixed dt, rk44 | 64.9 | 62.1 (XLA already hoists its two static taus) |
| adaptive, cfl_every=4 | 90.0 | **54.5** (0.61×) |
| adaptive, cfl_every=1 | 89.6 | 91.2 (nothing frozen) |
| FD-z fixed dt lsrk33 | 44.5 | 44.0 (not hoisted) |

HLO check: with hoisting, zero transcendentals inside any step loop for fixed dt, all in the
outer cfl-block loop for `cfl_every>1`; off: 33 per stage inside the stage scan.

**Why the diagonal backend is not hoisted:** the FD-z/2D exp is on the z-broadcast
`(2,1,nkx,nky)` L — free in time AND memory (so "hoisting FD-z" is coded but switched off by
`hoistable=False`) — and a first version that hoisted it broke gate 6's bitwise reference
(`tests/test_particles_coupled.py`, 2D fixed-dt `simulate_scan` configs) by 15 elements at
1e-23: with a literal gamma XLA folds `(L·dt)·gamma` differently. Flipping `hoistable` on
costs a reference regeneration for no measurable gain. Hoisted/unhoisted bitwise agreement
is an op-order statement, not a guarantee against constant folding — never pin a hoisted
putzer2 run bitwise against an unhoisted one across jax versions.

## 3. Memory (XLA `compile().memory_analysis()`, U = one complex full-grid array = nz·nkx·nky·16 B)

| case | kgrid | state | temp | temp/U | total |
|---|---|---|---|---|---|
| spec 128²×16 lsrk33 unhoisted | 12.5 MB | 4.6 | 64.9 | 31.9 | 81.9 |
| spec 128²×16 lsrk33 hoisted | | | 72.0 | 35.4 | 89.0 (+9%) |
| spec 128²×16 lsrk54 hoisted | | | 105.0 | 51.7 | 122.0 (+49%) |
| spec 128²×16 rk44 (either) | | | | 36.2 | 90.5 |
| FD-z 128²×16 lsrk33 | 0.3 | 4.6 | 46.0 | 22.6 | 50.8 |
| spec 256²×64 lsrk33 unhoisted / hoisted | 196 | 66 | 1018 / 1131 | 31.6 / 35.1 | 1281 / 1394 |
| spec 256²×64 lsrk54 hoisted | | | 1661 | 51.5 | 1924 |
| FD-z 256²×64 lsrk33 | 1.1 | 66 | 714 | 22.1 | 781 |

Reading: the z_spectral working set is ~32 U BEFORE hoisting (FD-z ~22 U) — RHS gradient
stack and rfftn intermediates, not the propagator; the persistent putzer2 operator
(`lin_L` 4 U + `lin_m` + `lin_s2`) is 6 U = 196 MB at 256²×64. Hoisting adds 4·nstage U
minus the per-stage coefficient temporaries it removes: measured +3.5 U (lsrk33), +20 U
(lsrk54). `cfl_every` blocks: same numbers as fixed dt.

**The 2N property of LSRK is moot in this regime**: hoisted lsrk33 (35.4 U) sits level with
rk44 (36.2 U, four k-registers and its two hoisted ops); hoisted lsrk54 is the most
memory-hungry option. Per-scheme (128²×16; "stab." = imaginary-axis stability limit of the
explicit part from the stored tableaus: lsrk33 1.73, rk44 2.83, lsrk54 3.34):

| scheme | temp/U | ms/step | ms/stage | cost/unit t at stab.-limited dt (rel.) |
|---|---|---|---|---|
| lsrk33 unhoisted | 31.9 | 80.5 | 26.8 | 1.00 |
| lsrk33 hoisted | 35.4 | 49.2 | 16.4 | 0.61 |
| lsrk54 unhoisted | 31.9 | 130.5 | 26.1 | 0.84 |
| lsrk54 hoisted | 51.7 | 81.3 | 16.3 | 0.52 |
| rk44 | 36.2 | 60.0 | 15.0 | **0.46** |

At a common `cfl_safety` hoisted lsrk33 is the cheapest step; if `cfl_safety` is scaled to
the scheme, rk44 is cheapest per unit time, 4th order, needs no hoisting machinery, and costs
the same memory as hoisted lsrk33. `cfl_safety` is scheme-independent in the code today.

## 4. Options measured but NOT implemented (all change round-off on the putzer2 paths; no bitwise gate pins them; FD diagonal path untouched)

All measured in the real step via monkeypatch + `jax.clear_caches()` (scratch scripts, not in
the repo):

| propagator | step vs production | memory | notes |
|---|---|---|---|
| generic eigen `V diag(e^{λτ}) V⁻¹`, full-grid complex exps | 0.69× | 10 U persistent (+2 U/stage if hoisted) | `λ = m ± L01` (NOT `m ± sqrt(s2)` — sign of kz); defective modes need a fallback |
| real cos/sin path (`m` real, `s² ≤ 0`, decidable at setup) | 0.75× | none | naive two-branch `where` + cast back to complex is SLOWER (1.44×) |
| **Elsasser-separable** `e^{dτ}(k⊥) ⊗ [cos, i sin; i sin, cos](kzτ)` | **0.62×** | **none** (perp (nkx,nky) + z (nz) arrays; also drops the 6 U operator) | exact iff ν = η (and any `z_diss_k`); no change of state variables; agrees with putzer2 to 5e-14 |
| cheaper `_coeffs` (generic): store `s = sqrt(s2)` at setup, `w = e^{sτ}` once, `cosh=(w+1/w)/2`, `sinh/s=(w−1/w)/(2s)`, Taylor branch kept | not measured; ~2.5× fewer transcendentals | none | the lever for adaptive `cfl_every=1` and for GDI-IF |
| 2-entry `Putzer2Exp` when `L00=L11, L01=L10` | n/a | halves hoisted memory | detectable at setup |

Splitting exactness: `exp((A+B)τ) = exp(Aτ)exp(Bτ)` iff `[A,B]=0`. RMHD ν=η: yes (exact).
ν≠η: `diag(d_φ,d_ψ)` vs `σ_x` don't commute — O(τ²(d_φ−d_ψ)kz) Lie-splitting error, not
acceptable. KAW-type (`i·kz·f(k⊥)` entries): exp carries `cos(kz·√(fg)(k⊥)·τ)`, a function
of the product — no kz-only × k⊥-only factorization, no memory saving; memory-free option
there is the per-stage real-trig path, hoisting is the speed-for-memory lever. Eigen
storage vs putzer2+hoist: 10+2·nstage vs 6+4·nstage U (16/20 vs 18/26 for lsrk33/54) —
saves only for lsrk54, costs ~6 extra complex mults per apply; for RMHD it collapses into the
separable form (V is the constant Elsasser transform, λ = d ± i·kz separable at ν = η).

**Recommendation carried over:** implement the separable propagator as a backend selected at
`setup_kgrids` when the RMHD operator has the structure (ν = η), putzer2 + hoisting as the
fallback for ν ≠ η / KAW-type recipes, plus the cheaper `_coeffs` for the remaining putzer2
path. That removes the hoisted-memory term from every scheme row (lsrk33 regains its
register advantage) and is 0.62× at every step including adaptive `cfl_every=1`.

## 5. State of the tree

Uncommitted on `main` (as of this hand-off): `taranis/{config,propagators,run,timestepping}.py`,
`CLAUDE.md`, `docs/{numerics,performance}.md`, new `bench/zspectral_profile.py`,
`tests/test_hoist_propagator.py`, this file. Another session was concurrently editing
`tests/test_particles_{kernel,coupled,3d}.py` and `plans/TESTPART_PLAN.md` (gates 10/11) —
those modifications are theirs, not part of this work. Do not `git stash` in this worktree.
