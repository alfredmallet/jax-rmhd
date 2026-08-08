# Adversarial review — forcing spin-up fix (plans/FORCING_SPINUP_PLAN.md) — 2026-08-08

Reviewer: fresh session, no part in the implementation. Scope: the forcing-fix half of
the working tree only (the jax_rmhd→taranis rename was excluded, and the change surface
was mechanically separated from it — see §2.1). All numbers below were measured in this
review's own probes, not taken from the implementers' claims.

Baseline at review time: HEAD = bee6cfb (pre-fix, pre-rename); full suite on the working
tree: fp64 155 passed / 7 skipped, fp32 137 passed / 25 skipped, 0 failures.

## 0. Verdict

**PASS WITH NITS.** The quadratic normalization is correct in every geometry and sign
regime I could construct, at both precisions; the RNG-stream invariant holds at source
level and bitwise; the webgpu port matches the python implementation exactly; the new
tests bind (a simulated revert fails them by three orders of magnitude); the three
documented deviations are each sound. The findings are one minor doc omission the plan
explicitly required (F1) and four nits. Nothing blocks the commit.

## 1. Findings

### F1 (minor) — examples/forcing-modes-2D still teaches the pre-fix behaviour

Plan Phase 2: "note in examples/forcing-modes-2D that the overshoot remark is
historical". Not done. The notebook's markdown still presents `safe_scale` as current
("The scale is s = clip(ε/P, ±s_max) (`shared_physics.safe_scale` ... see
`rmhd._forcing_scale_from`)") and builds its narrative on the spin-up burst ("the first
few steps deliver a coherent kick that transiently *overshoots* the target", "**The
spin-up burst:** in both runs the first finite-difference point sits far *above* the
target"). On the production path that burst no longer exists, so a re-run of the
notebook will contradict its own text.
**Action:** update the notebook text (and ideally re-run it) per the plan, before or
immediately after the commit.

### F2 (nit) — plan status header is stale

`plans/FORCING_SPINUP_PLAN.md:7` still says "NOT implemented", and Phases 0–2 carry no
DONE markers while Phase 3 does. **Action:** mark Phases 0–2 done (and gate 5's review
as this file) when committing.

### F3 (nit) — "clip engages only if F2·dt underflows AND P ≈ 0" overstates

`taranis/physics/shared_physics.py:176-178` and docs/numerics.md say the ±smax clip is
now engaged "only if F₂·dt underflows and P ≈ 0". Not quite: the positive root
`s* ≈ 2|P|/(F₂·dt)` for adverse P (and `s* = √(2·tgt/(F₂dt))` for large targets) can
exceed smax with perfectly finite F₂·dt, and when clipped at adverse P the realized
injection is below target and can even be transiently negative
(`smax·P·dt + ½smax²F₂dt² < 0` for |P| large) — the one behaviour sign-following could
not produce. Measured relevance: in a 400-step developed window at eps_tot=1e-2 the
candidate `2|P|/(F₂dt)` peaked at 0.07 (smax=1), i.e. it never came close, and the plan
text itself acknowledges the clip case ("still hitting target unless the smax clip
engages"). docs/numerics' "exact under `cfl_every`" is similarly a slight overstatement:
exact within a block, still lagged across block boundaries (1 step in N).
**Action:** soften the two comments; no code change.

### F4 (nit) — webgpu/README.md "injects ε·dt on the first step"

It is the second step (the first *forced* step — step 1's envelope is zero); the jax-side
docs and tests state this correctly. **Action:** one-word fix.

### F5 (nit) — bench_phase1.py's A/B monkeypatch omits the per-stage cap

`bench/bench_phase1.py:100-113` (`_sep`, the separate-reduce A/B variant) reproduces the
production dt-path faithfully but its `dt=None` branch is plain `safe_scale` without the
`_quiescent_dt` cap of `rmhd._scale_epilogue:196-200`. Only ever exercised with the
default path in the bench, so it cannot skew a measurement today.
**Action:** none required; noting so nobody reads `_sep` as a reference implementation.

## 2. Mandated probes — evidence

### 2.1 Change surface is confined (invariants)

Function-level diff of the working tree against HEAD (rename-normalized): in
`shared_physics.py` the ONLY change is the new `selfnorm_scale`; `ou_update`,
`_draw_symmetrized_noise`, `_draw_shell_noise`, `_symmetrize_real_line`,
`reconstruct_envelope` and every `perp_*` reduction are byte-identical. `rmhd.py`:
`_forcing_scale_from`, `forcing_scale`, new `_quiescent_dt`/`_scale_epilogue`,
`set_timestep` comment-only. `run.py`: `_advance_forcing`, `_refresh_forcing_scale`.
`physics/__init__.py`: `EquationRecipe` doc/signature. `gdi.py`, `comms.py`,
`timestepping.py`, `grids.py`, `types.py`, `diagnostics.py`, `snapshot_io.py`:
identical mod rename (`config.py` differs only in the pip-name hint string). GDI recipe
untouched; `test_gdi_linear` green.

### 2.2 Quadratic correctness (probe 4)

Independent forward-Euler check, not trusting any comment: draw a real OU envelope,
compute `rmhd.forcing_scale(state,kgrid,params,dt)` with smax=1e6, apply
`fields += dt·ForcingTerm`, measure ΔE via `diagnostics.energy`. Result: ΔE/(tgt·dt) =
1.0000000000 (fp64) in **all** of: 2D elsasser, 3D z_spectral, 3D finite-difference-z,
2D momentum; for P>0, P<0 (sign-flipped fields) and P=0 (zero fields). fp32: worst ratio
0.99994. This simultaneously validates the elsasser factor-2 bookkeeping (tgt=2·eps± vs
E_tot=(E⁺+E⁻)/2), the F₂ reduction normalization, and the z-envelope factors in both 3D
modes (z_spectral nz/2-on-kz=±1 against perp_reduce's 1/nz², and the z_envcos/sin path).
Notable: z_spectral and FD-z give identical scales from the same seed — the two
envelope representations Parseval-match, as SPEC3D claims. The fp32 two-branch
evaluation shows no cancellation loss (P=±100 regime included).
`test_selfnorm_scale_limits`' assertions match my independent evaluation everywhere.

### 2.3 Sign convention (probe 1, plan Decision 1)

Setup: 2D elsasser 64², plan's diagnosis config, spun up 326 steps to t=20 (E=0.11,
saturated), then 400 further steps with each convention from the same state (fresh
`Parameters` per variant to defeat the identity-hashed jit cache; sign-following
implemented as the quadratic root matching sign(P), which correctly limits to
`safe_scale`'s tgt/P in saturation).

| | positive root | sign-following |
|---|---|---|
| per-step injection / target (analytic, actual dt) | 1.0000 ± 0.0013 | 1.0000 ± 0.0006 |
| smax-clip engagements | 0 / 800 | 0 / 800 |
| adverse-P component-steps | 8 / 800 (1.0%) | 40 / 800 (5.0%) |
| mean |s| | 0.0142 | 0.0130 |

Both conventions hit the target injection exactly whenever unclipped (they are both
exact roots of the same quadratic — the difference is force *amplitude* under adverse
phase, not realized power). No rectification-bias signature; if anything the positive
root **shortens** adverse-phase dwell (1% vs 5% of steps: pushing forward drives P
positive, rectifying lets it linger). E drift over the window is statistically
indistinguishable. **Decision 1 stands; nothing to flag** beyond the F3 clip caveat.

### 2.4 Lagged dt at cfl_safety=0.5 (probe 2)

Quiescent spin-up, per-step realized ΔE vs eps·dt (dissipation negligible at these
amplitudes), fp64:

- eps_tot=0.1: realized/target per step in [0.31, 1.00]; worst single-step dt ratio 0.14.
- eps_tot=1.0: realized/target in [0.33, 1.01]; worst single-step dt collapse **0.042**
  (0.49 → 0.02-class, the plan's scenario); max|s| = 0.45 < smax; E finite throughout.

Structural reason the deviation stays O(1) rather than O(dt_prev/dt): the lagged dt only
matters in the self-term (√dt) regime, where dt sits at the params-static velocity floor
and is *constant*; by the time dt collapses the cross term dominates and s ≈ tgt/P is
dt-insensitive. Under-injection ≤ ~3× on isolated steps, over-injection ≤ 1%, no
instability pathway found. `cfl_every=4` from rest: finite at eps_tot 1e-2 and 1.0
(eps=1: transient overshoot to E=86 before settling at 24 — matches the CLAUDE.md
"~10x during spin-up, prefer developed states" warning). Approximation accepted.

### 2.5 The three deviations (probe 3)

**(a) Bitwise witness redefinition** — sound, and better than it looks:
- The reference npz records `git_commit` bee6cfb = current HEAD, and — decisive —
  replaying the run with `_scale_epilogue` monkeypatched back to plain `safe_scale`
  reproduces the reference **trajectory to all digits** (E[1] = 10.73734918233254 ==
  ref) *and* `forcing_state_final` bitwise. So the reference is genuinely pre-fix AND
  the entire behaviour change is confined to the scale epilogue.
- Witness sensitivity: seed 43 → O(1e4) mismatch; dt[0]·(1+1e-12) → mismatch at 1e-9.
  (A 1e-15 perturbation is absorbed by exp() rounding — the witness detects any change
  that survives one ulp of the decay factor, which is the right resolution.)
- Can it mask an RNG change? The replay pins `ou_update` over 20 steps against a
  pre-fix array; the step-1 tie-in pins the run path's call into it; the per-step run
  path is a scan of one function, so step 1 covers all steps up to the dt coupling —
  which is exactly the part the replay holds fixed. Combined with the source-level
  diff (§2.1: zero changes to any RNG-touching function) I could not construct a
  stream change that passes. The witness does pass under a full revert — by design
  (it is a stream test); the revert is caught by the other two tests (§2.7).

**(b) `_quiescent_dt` returning `params.dt` when `adaptive_timestep=False`** — correct,
no hole: `dt_override` is only ever produced by `run._cfl_block`, reachable only through
`_use_cfl_blocks` = `cfl_every>1 AND adaptive_timestep` (verified in both `simulate` and
`simulate_scan`/`block_of_steps`); with adaptive off every stepper uses `params.dt`
verbatim, so the branch is exact, and under adaptive the velocity floor makes dt_q a
true upper bound (frozen block dts included — they come from a real `set_timestep`
call, which is floored by the same eps=0.1). Measured: per-stage from-rest kick =
0.676·tgt·dt at adaptive dt=0.49, fixed dt=0.05 and fixed dt=0.4909 alike (pre-fix:
185×). A direct stepper call with a hand-rolled `dt_override` bypasses the bound —
test-code-only surface, acceptable. Cross-site comment at `set_timestep` is in place.

**(c) Precision-reference band** — sound, non-vacuous, with sane margins. Measured this
session (fp32): fields rel 2.6e-2…3.4e-2, forcing_scale 5.1e-3…1.1e-2, inside
[1e-4, 1e-1] with ≥3× headroom both sides; forcing_state still at 3.4e-6…3.7e-6 under
the unchanged 1e-5 gate. A revert would put fields back at ~4e-6 and fail the lower
bound. The band's upper edge alone would admit a subtle stream change (3× margin), but
that is not its job — streams are pinned bitwise by the first-noise hex constants and at
1e-5 by forcing_state. Nothing found that depends on the old reference semantics; the
old semantics survive only as the reference data itself, correctly labeled.

### 2.6 WebGPU parity (probe 5)

- **selfnormScale vs selfnorm_scale**: node transcription of the WGSL (fround after
  every op) against the python at fp32 over 1344 cases spanning every branch (target 0,
  adverse/zero/large P, F₂=0, dt=0, clip both signs, fallback with P=0): **0 mismatches
  at 1e-6 rel; exact agreement.** Branch structure (den>0 / F2dt>0 guards, target==0
  short-circuit, select-evaluates-both discipline) is line-for-line.
- **F₂ accumulation**: 2D `w·dot(f,f)` with `w = ksq·yfac`, ×INVN2 = 1/(nx·ny)² —
  matches `perp_inner_product` at nz=1. hasZ: per-plane |envelope|² summed over kz=±1
  with INVN2 = 1/(nz²(nx·ny)²); re-derived Σ|（nz/2)(A∓iB)|² = (nz²/2)(|A|²+|B|²), so the
  nz factors cancel exactly as SPEC3D states; the squared-before-summed subtlety only
  bites at nz=2, unreachable (UI min 32). The jax-side Euler probe (§2.2) independently
  confirms the same normalization chain end-to-end.
- **Lagged dt in the live loop**: verified in both apps' `step()`: cflFinal writes
  sc[0] at stage 0 of the step (only when doCFL), tick/ou/scale run after the stages —
  so at scale-dispatch time sc[0] IS the just-completed step's dt, used by the next
  step's stages after its own cflFinal overwrites sc[0]. Identical to
  `run._advance_forcing`'s convention, incl. the frozen-dt cfl_every case; the IC path
  (zero-initialized scalars buffer → F₂·dt=0 → fallback → clamp) mirrors
  `_refresh_forcing_scale`'s dt=0.
- **Self-test plumbing**: `scal[0] = R.A_dt_last` before the scale dispatch, with A_dt
  explicitly noted as what the CFL block left behind — correct in both rmhd2d and
  rmhd3d self-tests; `A_scale_check` in the generators is computed with the same
  `dt_last` that `_advance_forcing` used.
- **Refvectors**: both generators re-run in this sandbox reproduce the committed
  `refvectors.json` and `refvectors3d.json` **bitwise** (md5-identical).

### 2.7 Test honesty (probe 7)

- Harness: `checks()` accumulates and raises one AssertionError on any failure —
  non-vacuous (`tests/_rmhd_testing.py:152-158`).
- Revert simulation (epilogue → `safe_scale`): `test_first_injection_equals_target_times_dt`
  fails at ratio **2187** vs the allowed 1.5×, and the eps-ratio test would measure
  1.000 vs the required [50, 200]. The `E2 < 0.1·ref` back-stop also fails. The suite
  cannot pass with the fix reverted.
- Tolerances bind: 1.5× on E₂ (measured post-fix ratios ~1.0), [50,200] on an exact-100
  eps ratio, bitwise on the witness. The fp64-only marking is honest (reference is
  fp64) and the lazy `_gen` import correctly avoids poisoning fp32 collection.
- `savio_manifest` gains the module (serial, fp64-only) — consistent with its markers.

## 3. Verification gates (plan §Verification) — status as re-measured here

1. Streams bitwise pre/post: **held** (§2.5a, plus source-level diff §2.1).
2. Quiescent eps sweep, all three geometries: **held** (tests + §2.2; E₂=tgt·dt, no
   pinning after spin-up — max|s| 0.45 at eps_tot=1, clip count 0).
3. Developed-state continuation at target injection: **held** (§2.3: 1.0000±0.0013;
   the plan's "restart a pre-fix checkpoint" variant was not run by the implementers,
   but the same property was measured on a developed state directly).
4. cfl_every>1 quiescent NaN footgun: **held** (finite at eps 1e-2 and 1.0; CLAUDE.md
   wording matches the measured ~10× transient).
5. Full suite + regenerated references: **held** (155/137 green; precision reference
   deliberately NOT regenerated, correctly — §2.5c; spin-up reference is new).

Sandbox caveat: everything here is single-process (local MPI stub, fake devices);
multi-rank mpi4jax and real-GPU shard_map paths were reviewed by trace only
(`test_backend_jax` passes with the 4-arg `forcing_scale_func` under the stub mesh).
The F₂ reduce rides the same batched-allreduce machinery as P, so no new collective is
introduced — Savio re-verification can piggyback on the next routine run.
