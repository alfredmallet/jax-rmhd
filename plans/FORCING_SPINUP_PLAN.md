# Forcing spin-up kick — diagnosis and fix plan

Bug report: plans-webgpu/FEEDBACK_2026-08-08.md P0 item 1 (3D forced app: large initial energy
jump from a quiescent start, independent of eps). The webgpu clamp is a port of the main
solver's `safe_scale`, and the bug **reproduces in the jax code** — this plan fixes it in
taranis first (the app inherits the decision, Phase 3). Status 2026-08-08 (end of day):
ALL PHASES EXECUTED — Phase 0 (reference recorded), Phases 1+2 (jax fix + tests + docs),
Phase 3 (webgpu port); adversarial review PASS WITH NITS
(plans/FORCING_SPINUP_REVIEW.md), all five findings addressed same day. Remaining:
Alfred's on-device app pass (self-tests + quiescent eps sweep on real GPU) and the
multi-rank/Savio paths, which the sandbox review covered by trace only.

## Diagnosis (verified, not conjecture)

Three ingredients conspire; each is individually by design.

1. **The capped scale pins at smax from rest.** `safe_scale(target, P)`
   (`physics/shared_physics.py:149`) returns `clip(target/P, ±forcing_scale_max)`.
   From a quiescent start the cross-power `P = ⟨∇z∓·∇f_raw⟩ = 0`, so the scale pins at
   `smax` (default 1.0, `config.py:107`) **regardless of eps** — eps enters the forcing
   *only* through this scale, and the clamp erases it.
2. **The raw OU envelope is O(1)-amplitude by construction.** `ou_update` draws
   unit-variance complex noise × `grid_norm = nx·ny` (`shared_physics.py:79`), i.e. the
   stationary real-space envelope has ⟨|∇f_raw|²⟩ ≡ F₂ = O(10–100) for fshell=(1,3),
   with **no eps anywhere**. The normalization is deferred entirely to the scale factor
   — the deferral is exactly what the pinned clamp defeats.
3. **The quiescent dt is large.** With no flow, `set_timestep`'s velocity floor
   (`physics/rmhd.py:85-87`, `eps=0.1`) gives `dt = cfl_safety·min(dx,dy)/0.1`; at 64²,
   Lx=2π, cfl_safety=0.5 that is **dt = 0.491**.

The kick is the ballistic self-injection of the full-strength envelope over one step:

    ΔE ≈ ½ · smax² · F₂ · dt²,   with ⟨|f|²⟩ = (1−e^(−2dt/τ))·⟨|f|²⟩_stat  ⇒  ΔE ∝ dt³.

Step 1 injects nothing (`forcing_state` initialized to zero, so f_raw = 0); the kick
lands entirely on step 2, whose scale was computed from the still-zero fields.

### Measurements (2D elsasser, 64², diss=(1e-5,1e-5) hyper=3, τ=1, fshell=(1,3), seed 42, fp64, lsrk33)

| eps_tot | dt₁ | E(step 2) | E(t=5) | E(t=20) |
|---|---|---|---|---|
| 1e-4 | 0.491 | 1.074e+01 | 3.054 | 3.054 |
| 1e-3 | 0.491 | 1.074e+01 | 3.055 | 3.055 |
| 1e-2 | 0.491 | 1.074e+01 | 3.069 | 3.069 |
| 1e-1 | 0.491 | 1.074e+01 | 3.210 | 3.210 |

E(step 2) is eps-independent to 4 digits. Worse than a transient: at these diss values
the box then sits at E ≈ 3 indefinitely — for small eps **the kick is the final state**,
so any "amplitude vs eps" demo (FEEDBACK item, demo catalogue #2) is meaningless.

dt-scaling check (eps=1e-3, cfl_safety 0.5 / 0.1 / 0.02 → dt 0.49 / 0.098 / 0.020):
E(step 2) = 1.07e1 / 8.19e-2 / 7.08e-4 — ratios 131 and 116 vs the predicted 5³ = 125.
Mechanism confirmed.

docs/numerics.md ("Cap the scale factor, never floor the denominator") already names the
cost — "while the cap is engaged the injection is *uncontrolled*" — but treats it as a
cosmetic spin-up overshoot. The measurements show it is O(10) energy, eps-independent,
and persistent. The same pinning also recurs *mid-run* whenever P fluctuates through
zero (OU decorrelation with a weak field), so any fix must be continuous in P, not a
special-case for t=0.

## Fix design

### Rejected options

- **Static `smax = C·eps`** (the obvious "make the cap eps-aware"): wrong form twice
  over. (a) Keeping ΔE ≤ eps·dt requires `smax_eff = √(2·tgt/(F₂·dt))` — √eps, and F₂-
  and dt-dependent; a constant C is a guess against grid/shell/τ/cfl. (b) It caps the
  *equilibrium* scale too: the legitimate saturated scale is tgt/P with P shrinking with
  field amplitude (~eps^{1/3}), so at small eps `s* = tgt/P > C·eps` and the forcing
  persistently under-injects — breaking the normalization exactly where the bug matters.
- **Pre-scaling the OU amplitude by some power of eps**: cosmetic — the kick shrinks
  but remains eps-independent in ratio to the target; equilibrium scale factors shift
  correspondingly and nothing structural improves.
- **Ramp-in envelope on t**: ad hoc, dead weight after spin-up, and does not address
  the mid-run P≈0 recurrence or restarts from weak fields.
- **Forcing-aware CFL / smaller velocity floor**: treats the dt³ symptom, leaves the
  uncontrolled-injection structure intact.
- **Flooring P**: still rejected, same rationale as docs/numerics.md (units, sign
  flips near zero).

### Chosen: self-energy-aware normalization (with a simpler fallback)

Over one step, the force `s·f_raw` acting on fields z injects (per OU component)

    ΔE = s·P·dt + ½·s²·F₂·dt²,       F₂ = ⟨|∇f_raw|²⟩ (per component, batch),

the linear cross term plus the self term the current scheme ignores. Choose s so that
ΔE = tgt·dt **exactly**, i.e. the positive root of the quadratic:

    s = [ −P + sqrt(P² + 2·F₂·dt·tgt) ] / (F₂·dt),    then clip to ±smax as now.

Limits: F₂·dt → 0 or |P| large ⇒ s → tgt/P (current behavior, unchanged in
saturation); P → 0 ⇒ s → √(2·tgt/(F₂·dt)) (controlled: first kick = tgt·dt on the
nose). Continuous in P through zero, so the mid-run recurrence is handled by the same
expression. The cap-not-floor invariant survives: nothing floors P; the *result* is
still clipped to ±smax (now a genuine last-resort safety, engaged only if F₂·dt
underflows AND P ≈ 0).

**Fallback (if review prefers a minimal diff):** keep `safe_scale`'s form and make the
cap dynamic — `s = clip(tgt/P, ±min(smax, √(2·tgt/(F₂·dt))))`. Same quiescent limit,
same inputs (F₂, dt), but overshoots up to ~2× target in the intermediate regime where
both terms contribute. Implementation cost is identical (F₂ and dt threading is the
work; the epilogue formula is three lines either way), so the quadratic is recommended.

Prototype validation (same setup as above, quadratic scale, lagged dt):

| eps_tot | E(step 2) | eps·dt | E(t=30) |
|---|---|---|---|
| 1e-4 | 4.904e-05 | 4.90e-05 | 2.50e-03 |
| 1e-3 | 4.904e-04 | 4.90e-04 | 1.85e-02 |
| 1e-2 | 4.904e-03 | 4.90e-03 | 1.23e-01 |
| 1e-1 | 4.909e-02 | 4.91e-02 | 8.49e-01 |

First injection = eps·dt to 4 digits; saturated energy is now monotone in eps.

### Sign convention (DECIDED 2026-08-08: positive root)

Current `safe_scale` follows sign(P) — with adverse phase it flips the force to keep
injecting, rectifying the OU process. The positive quadratic root instead keeps s > 0
and lets the exact solve absorb an adverse linear term (still hitting target unless the
smax clip engages). Alfred's decision: use the positive root (no rectification bias);
the adversarial reviewer MUST still check both variants against the realized-injection
diagnostic and flag if sign-following materially changes the injected-power statistics.

## Implementation

### Phase 0 — reference recording (before any change)  [DONE 2026-08-08]

Per the standing protocol: record an RNG/trajectory reference on main (a short forced
2D and 3D-z-spectral run, fixed seed, per-step E and forcing_scale dumped) so the
"streams unchanged, trajectories changed only via scale" claim is checkable. The fix
adds **no RNG draws** (F₂ is deterministic in f_raw), so `forcing_state` streams must
be bitwise identical before/after; assert that explicitly.

### Phase 1 — core (production path: forcing_norm_per_step=True)  [DONE 2026-08-08]

- `shared_physics.py`: add `selfnorm_scale(target, P, F2, dt, scale_max)` implementing
  the quadratic + clip, with the `target == 0 → 0` guard and a `F2·dt == 0` branch
  falling back to `safe_scale` (also covers dt=0 at initialization). Keep `safe_scale`
  (still used by the per-stage path and GDI if ever forced).
- `rmhd._forcing_scale_from`: compute `F2 = perp_inner_product(f_raw, f_raw, batch=True)`
  alongside the existing P reduction (same machinery, one extra batched reduce; elsasser
  tgt = 2·eps± exactly as now, momentum tgt = forcing_power).
- **dt threading**: `forcing_scale_func` signature gains dt →
  `forcing_scale(state, kgrid, params, dt)`. `run._advance_forcing` already has dt (the
  step just completed — the scale for the next step uses the **lagged** dt, one more
  O(dt/τ)-class approximation of the same kind norm_per_step already makes; with
  cfl_every blocks dt is frozen anyway). `_refresh_forcing_scale` passes dt=0 (f_raw
  from a fresh initialize is zero, and for checkpoints the stored scale is recomputed
  next step; document). Registry (`EquationRecipe.forcing_scale_func`) and the
  `comm_backend="jax"` shard_call wrapper updated to match.

### Phase 2 — per-stage path, tests, docs  [DONE 2026-08-08]

- `forcing_norm_per_step=False` (exact per-stage, test/reference path): `ForcingTerm`
  has no dt — state carries `t` but not `dt`; dt is a stepper-local chosen by
  `set_timestep` each step (and frozen per block under `cfl_every`), and the RHS
  interface is deliberately dt-agnostic. Recomputing dt inside `ForcingTerm` would
  repeat the CFL allreduce every stage and give the wrong value inside cfl blocks.
  Option (a) thread the stage dt through `construct_rhs` (signature ripple into the
  steppers; alternatively a `dt` field in `SimulationState`, but that changes the
  uniform checkpoint pytree structure — worse),
  option (b) leave the per-stage path on `safe_scale` and document
  that from-rest starts must use the production default.
  **DECIDED 2026-08-08: option (b), flagged for future review** (revisit if a test ever
  needs a quiescent per-stage start), **with a scale_max mitigation**: on the per-stage
  path, tighten the cap to `min(smax, sqrt(2·tgt/(F₂·dt_q)))` with
  `dt_q = cfl_safety·min(dx,dy)/0.1` — the params-STATIC quiescent dt bound (the largest
  dt `set_timestep`'s velocity floor can emit from rest, so no dt threading needed). F₂
  is already computed in `_forcing_scale_from` for the quadratic, so the mitigation is
  one extra line. It is conservative when dt < dt_q, but it only binds when P is small —
  i.e. near-quiescent states, where dt ≈ dt_q anyway. Bounds the per-stage kick at
  ~tgt·dt_q without changing behavior in any developed-field test; if a smoke test
  asserts exact normalization from a state where the mitigation binds, fix the test's
  start state, not the cap.
- Tests: extend `test_forcing_norm_per_step.py` with the quiescent-start eps sweep
  (assert E(step 2) = tgt·dt within RK error, and eps-*dependence*); keep
  `test_safe_scale_uncapped_and_capped` and add the analogous limits test for
  `selfnorm_scale` (P→0, F₂dt→0, adverse-P, clip engagement). Regenerate any pinned
  forced-trajectory references (check `tests/_gen_precision_reference.py` inputs —
  forced or not); RNG streams must NOT need regeneration.
- Docs: rewrite docs/numerics.md "Cap the scale factor, never floor the denominator" →
  keep the no-floor rationale, add the self-energy derivation and the dated behavior
  change (cf. the 2026-07-31 elsasser factor-2 note: runs before the date spin up
  differently); update CLAUDE.md forcing bullet (`shared_physics.py:206` reference) and
  the `cfl_every` footgun note — the quiescent-NaN footgun should be re-tested after
  the fix and the warning softened or kept per result; note in
  examples/forcing-modes-2D that the overshoot remark is historical.

### Phase 3 — webgpu port (after taranis lands; separate change)  [DONE 2026-08-08]

Landed as described below, with one deviation: `dt` reaches the kernel through `sc[0]`,
not a new `Cfg` field. `sc[0]` IS the lagged dt at that point (cflFinal wrote it before
the step's stages; `tick` and `ou` have already consumed it) and dt never exists CPU-side
in the apps — `Cfg` is uploaded only on control changes and the scalars buffer is read
back asynchronously — so `sc[0]` is both the only synchronous source and the closer
mirror of `run._advance_forcing`. Envelope-weighting question below: RESOLVED, the
`hasZ` two-plane sum reproduces jax to 1e-16 rel for both P and F2 (no second 3D bug).

- `physics.js` `scaleWGSL`: accumulate `w·|f|²` per component into the free `acc.z/.w`
  lanes of the existing vec4 shell reduction (2D branch; hasZ branch per plane), add
  `dt` to `Cfg`, apply the same quadratic in the `tid==0` epilogue. `smax` stays in
  uiParams as the safety clip.
- While in `scaleWGSL`, resolve the FEEDBACK open question on the `hasZ` branch: the
  two-plane sum matches the jax z_spectral convention (envelope lives only on kz=±1)
  only if `envelope()` weighting reproduces `reconstruct_envelope`'s nz/2 factors and
  perp_reduce's 1/nz² — verify against `gen_refvectors3d.py` before assuming the 3D
  app's jump is *only* this bug.
- Re-run Alfred's observation on both apps: energy trace from quiescent forced start,
  eps sweep — jump gone, early dE/dt = eps.

### Verification (gates, in order)

1. Phase 0 reference: forcing_state streams bitwise identical pre/post fix.
2. Quiescent eps sweep (2D + 3D z_spectral + 3D finite-difference-z): E(step 2) =
   tgt·dt, saturated E monotone in eps, no smax-pinning events after spin-up
   (log `scale/smax` max over run).
3. Developed-state continuation: restart a saturated pre-fix checkpoint post-fix;
   realized injection stays = tgt (scale change in saturation is O(s²F₂dt/tgt), verify
   it is small).
4. `cfl_every > 1` quiescent forced start: NaN footgun re-tested.
5. Full test suite + regenerated references; adversarial review by a separate fresh
   session per the standing orchestration protocol (review should specifically probe
   the sign convention and the lagged-dt approximation at cfl_safety=0.5).

## Decisions (all resolved 2026-08-08)

1. Sign convention: positive root; adversarial review checks both (above).
2. Per-stage path: option (b) + static-dt_q scale_max mitigation (above); flagged for
   future review.
3. Phase 3 webgpu: use the previous step's dt in the scale kernel's Cfg — the same
   lagged approximation as the jax side. General Phase 3 rule per Alfred: keep the app
   implementation as close as possible to the jax side (same positive root, same
   guard branches, same clip semantics).

Phase 0 reference follows the existing convention: generator
`tests/_gen_forcing_spinup_reference.py` → `tests/data/forcing_spinup_reference.npz`
(mirroring `_gen_precision_reference.py`).
