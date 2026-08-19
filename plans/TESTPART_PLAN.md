# TESTPART_PLAN — charged test particles in taranis

Status: PLAN, 2026-08-18, rev 4 (rev 2 folded in Alfred's answers to the open questions,
§10; rev 3 records Phase A0/A1/A2 landing the same day, rev 4 Phase A3 — see the dated
parentheticals in §2–§8; Phase A is complete, Phase B next). Two scope decisions since the sketch: (1) the resistive contribution to E_z is
switchable for the particles independently of the solver — it is one bit of a general
per-ensemble field mask (§3); (2) the WebGPU port is deferred to Phase C, after Phase B —
§9 keeps only the portability constraints Phases A/B must not break. Sequenced before the
webgpu game (`plans-webgpu/GAME_PLAN.md` eventually consumes the Phase-C kernel); nothing
here blocks on the game or vice versa.

## 1. Motivation and scope

Boris-pushed charged test particles co-stepped with the RMHD solver, to study stochastic
ion heating and current-sheet (E_z) acceleration in the turbulence — the
Chandran-2010-style test-particle program run inside taranis's own fields.

The E_z question is sharper than the sketch stated. In a (hyper-)resistive RMHD run the
parallel electric field at a current sheet has an ideal inductive piece and a resistive
piece; the resistive piece is η j_z for `hyper=1` and a pure numerical regularization
for `hyper>1`, and in a real plasma the reconnection E_∥ is supported by kinetic physics
the fluid model doesn't have. For the collisionless use case the resistive (and
forcing) pieces of E_z are therefore not physical, and particles see the IDEAL E_z only
by default (§10 decision); the non-ideal pieces are toggled on explicitly. Any
parallel-energization claim is still only meaningful quoted BOTH ways: because the
particles never back-react, an ideal-only ensemble and a full-∂ψ/∂t ensemble run in one
simulation on identical fields — the difference is a direct measurement of resistive-E
acceleration, at zero extra field cost. This paired configuration is the standard
production setup, not an afterthought.

Phase A is 2D (`dims=2`, single-process): fields are complete and cheap, and every piece
of machinery (pusher, field assembly, interpolation, diagnostics, gates) is exercised.
Phase B (3D) follows A and precedes the WebGPU port (§10), reusing all of it; nothing in Phase A may
assume ∂_z = 0 in the pusher or the field-assembly interfaces — only in the field
values. §2 shows the E_z expression is literally identical in 2D and 3D, which retires
most of the Phase-B physics risk in advance.

## 2. Fields seen by a particle (code units)

taranis's conventions (the ones that make `NonlinearTerm`'s bracket signs correct MHD:
∂_t ω + u·∇ω = b·∇j) are u = ẑ×∇φ, b_⊥ = ẑ×∇ψ, hence Φ = B₀φ and A_z = −ψ. Then

    B   = B₀ẑ + ẑ×∇ψ
    E_⊥ = −∇_⊥Φ            = −B₀∇φ
    E_z = −∂_zΦ − ∂_t A_z  = −B₀∂_zφ + ∂ψ/∂t

With the code's induction equation ∂ψ/∂t = −{φ,ψ} + B₀∂_zφ + η∇²ψ (the ∂_zφ term is
`FDLinearTerm`/the z_spectral off-diagonal; absent in 2D), the parallel-gradient pieces
cancel exactly and

    E_z = −{φ,ψ} + η∇²ψ        (identical in 2D and 3D; ≡ −(u×b_⊥)_z + η j_z, j_z = +∇²ψ)

Note the sign: E_z = +∂ψ/∂t in 2D, not −∂ψ/∂t as the sketch guessed — because
A_z = −ψ in this convention. Phase A0 writes this derivation into docs/numerics.md and
gates 2 and 7 pin it numerically; if the derivation is wrong the E×B gate fails loudly.
(A0, 2026-08-18: sharpened — gate 2 drives analytic uniform fields, so it pins the PUSHER's
Lorentz-force convention, not the assembly. The assembly's relative signs are pinned by the
exact ideal-Ohm identity E·B = 0 on the raw bracket, asserted in test_particles_coupled; the
absolute sign of b_⊥ and E_z against ψ is gate 4's job in A2.)
(E_∥ = E·b̂ differs from E_z by b_⊥·E_⊥/B₀ corrections; the pusher works in Cartesian
components and never needs E_∥.)

The pieces, in real space at the particle, labeled for the mask in §3:

- **ideal**: −{φ,ψ} = −`bracket(gphi, gpsi)` — pure arithmetic on the gradients
  `rmhd.grad` already produces; no extra transform. (A0, 2026-08-18: it does need one
  extra fft+ifft pair after all — the discrete ψ integrates `dealias·NL_ψ`, so the ideal
  piece the particle must see is `ifft(dealias·fft(−{φ,ψ}))`, not the raw pointwise
  bracket; only the dealiased form lets gate 7 converge and keeps gate 4 exact.)
- **resistive**: `ifft(L_ψ · ψk)` where L_ψ = −diss·ksq^hyper is exactly the ψ diagonal
  of `rmhd.linear_matrix` — the same operator the propagators apply, evaluated once more
  as an array. Equals η j_z for `hyper=1`; the hyper-resistive analogue otherwise. One
  extra ifft, computed only if some ensemble has the bit on (static config).
- **forcing**: +f_ψ from the scaled elsasser envelope (zero in momentum mode, where ψ
  has no forcing source). One extra ifft when live. Like the resistive piece it is not
  physical E for the collisionless use case (it's the stirring EMF), so it is off by
  default; the exactness gates (4 full-mask variant, 7) switch it on.

Unit bookkeeping (decided, §10): B₀ = 1 in code units and q/m carries the entire scale
interpretation — no ε parameter anywhere in the module. (`B0` stays an explicit knob
defaulting to 1.0 because the kernel gates want uniform-field cases; production never
changes it.) The three dimensionless groups that matter are documented with the module:
ρ/dx (gyroradius vs grid), Ω/ω_nl (gyration vs eddy turnover), v_th/v_A.
(A3b, 2026-08-18: superseded — B₀ IS ε, or rather 1/ε, and it is a per-ensemble knob
production does change: see §10 and docs/numerics.md. Ω = qm·B₀ and ρ = v_⊥/(qm·B₀), so
everything below holds with q/m read as qm·B₀.)
With B₀ = 1: Ω = q/m and ρ = v_⊥·m/q. Baseline: ρ ≈ 2–4 dx ⇒ at 256² (dx ≈ 0.0245),
q/m ≈ 10–20, Ω ≈ 10–20, i.e. 60–120 solver steps per gyration — Boris at solver dt is
fully resolved with no substepping (a `substeps` hook is kept for large-q/m sweeps;
fields frozen across substeps).

## 3. Per-ensemble field mask (the resistive switch, generalized)

Each ensemble carries a static mask over the five field pieces:

    {bperp, eperp, ez_ideal, ez_resistive, ez_forcing}
    defaults: bperp = eperp = ez_ideal = True; ez_resistive = ez_forcing = False

The default is the ideal-Ohm particle (§10 decision): the resistive and forcing pieces
of E_z are numerical/stirring artifacts from the collisionless particle's point of
view, so they are opt-in, never silently on. The solver's own dissipation is never
touched by any of this — the mask changes only what the particles see. This one
mechanism covers every planned configuration:

- **Ideal-Ohm** (default): E_z = −{φ,ψ} only; the production physics ensemble.
- **Full-∂ψ/∂t ensemble**: `ez_resistive=True, ez_forcing=True` — E_z is exactly
  ∂ψ/∂t. Paired with a default ensemble in the same run for the
  resistive-acceleration measurement, and required by gates 4 (exact variant) and 7.
- **E = 0 control**: `eperp=ez_*=False`, b_⊥ live. Magnetic forces do no work, so
  kinetic energy is exactly conserved in the continuum even in evolving fields — any
  ⟨v²⟩ drift is the numerical-heating floor of the full pipeline (pusher +
  interpolation + live fields). Every heating claim is quoted against this ensemble,
  run in situ. This supersedes the sketch's "δb = 0 control" as the primary floor.
- **δb = 0 gyration control**: `bperp=eperp=ez_*=False` — pure gyration about B₀ẑ;
  gate 1 running continuously in production.

Config lives in `params.particles` (default `None` = off): a plain-JSON dict recorded in
params.json exactly like `eqpars` (`Parameters` hashes by identity, so a dict attribute
is safe; all keys are compile-time constants). Sketch:

    particles = {
      "seed": 0, "n": 32768, "substeps": 1, "B0": 1.0,
      "ensembles": [
        {"qm": 15.0, "init": {"kind": "maxwellian", "vth": 1.0}},                # ideal-Ohm (default)
        {"qm": 15.0, "init": {"kind": "maxwellian", "vth": 1.0},
         "ez_resistive": True, "ez_forcing": True},                              # full ∂ψ/∂t
        {"qm": 15.0, "init": {"kind": "maxwellian", "vth": 1.0},
         "eperp": False, "ez_ideal": False},                                     # E = 0 floor control (b_⊥ live)
      ],
    }

Init kinds: `maxwellian` (per-ensemble v_th) and `ring` (fixed v_⊥, uniform gyrophase,
optional v_z) — ring init is what a clean ξ-resolved heating measurement wants.
Positions uniform over the box.

(A2, 2026-08-18: landed as `particles/state.py::normalize_config`. `n` is the particle
count PER ENSEMBLE, not the total. Each ensemble's raw per-piece mask keys — e.g.
`ez_resistive=True` inline, as sketched above — are consumed and resolved through
`fields.resolve_mask` into a single `ens["mask"]` dict carrying all five `FIELD_PIECES`;
nothing reads the raw keys after normalization. Added `init_on_restart` (top-level bool,
default False): whether `snapshot_io.load_particles` may fall back to a fresh
`init_particles` draw when a snapshot has no particle checkpoint item, instead of
hard-erroring — not in the original sketch, needed once checkpoint/restart was wired.)

(A3b, 2026-08-18: two per-ensemble keys added. `epar_project` (bool, default False)
projects the numerical E∥ out of the gathered sample — `Ẽ = Ē − (Ē·B̄)B̄/|B̄|²` at every
half-kick, `boris.project_perp` — and REQUIRES the exact ideal-Ohm mask, since with the
resistive/forcing pieces on some of E∥ is real and would be deleted (ValueError otherwise;
`push_tracked` also asserts `nez == 1` statically). `B0` (float > 0) overrides the
top-level value for one ensemble and is ALWAYS present after normalization, the top-level
key keeping its meaning as the default — B₀ is the amplitude parameter 1/ε (§10,
docs/numerics.md), so a run may need several. Both round-trip through params.json with the
raw dict and leave `normalize_config` idempotent. `MOMENTS` widened 9 → 11 with the
local-B `vperpB2`/`vparB2`; gate 9 in `tests/test_particles_coupled.py`.)

## 4. Design (taranis core)

- **Co-stepped, never snapshot-interpolated**: particles advance inside the run at the
  solver dt (time-interpolating saved snapshots corrupts heating statistics). fp64
  particle state always, regardless of TARANIS_PRECISION (the `t` precedent); field
  samples are cast up at the gather. Production heating runs use fp64 fields.
- **Carry, not state**: `ParticleState` (NamedTuple pytree: `x` (N,3), `v` (N,3), per-
  ensemble leading structure, diagnostic accumulators) rides NEXT TO `SimulationState`
  as a carry tuple `(state, pstate)` in `block_of_steps`/`sim_to_next_snap`, NOT as a
  new SimulationState field. Rationale: the on-disk snapshot layout of every existing
  run stays untouched (no forcing_scale-style Optional threading, no snapshot
  migration), and the stepper/physics never see particles — the field→particle one-way
  dependency is enforced structurally. When `params.particles is None` every code path
  is statically identical to today's (gate 6).
  (A2, 2026-08-18: landed as specified — `ParticleState` carries only `x`/`v` in A2,
  diagnostic accumulators stay A3. `block_of_steps`/`_cfl_block` on the particles-on
  carry additionally return scan `ys = (t, moments)` — post-step time and
  `state.py::moments`'s per-ensemble (v_x²+v_y², v_z², v_z) means — which
  `simulate_scan` appends to the sidecar `particle_moments.txt` (the diagnostics-cadence
  bullet below). `simulate`/`simulate_scan` return `(state, pstate)` when particles are
  on, plain `state` when off — same rule as the carry. The particles-off branch in every
  touched function is a static `if params.particles is not None:` at the top, not a
  restructuring of the existing body.)
  (A3, 2026-08-18: the accumulators landed as a third leaf `w` `(n_ens, n, NWORK)` fp64 —
  cumulative work per unit mass split by `fields.WORK_PIECES = (eperp, ez_ideal,
  ez_resistive, ez_forcing)`, the electric members of `FIELD_PIECES`. It is credited inside
  the pusher from the exact half-kick identity `Δ(½|v|²) = h·E·(v_in + v_new)`, not
  integrated separately, so summing the pieces closes against ½|v|² to round-off (~5e-14 of
  KE₀) in BOTH precisions — `w` and the push arithmetic are fp64 whatever the field
  precision is. A piece an ensemble's mask omits stays exactly zero, which is what makes the
  paired resistive-split ensembles differ by `w_ez_resistive` alone. `MOMENTS` widened 3 → 9:
  the three velocity means plus `mu`/`mu2` about the LOCAL B (from the pusher's own last B
  sample — no extra gather) and the four per-piece work means; `push_ensembles` now returns
  `(pstate, mom)` and the run bodies just emit it. Field assembly gained
  `assemble_stacked` — one array `[E_⊥, the live ez pieces, B]` — so a half-kick is ONE
  gather instead of two (the deferred gather-side optimization below, in the form the work
  split needed anyway); `assemble` is its summed `(E, B)` form and `boris.push` a thin
  bitwise wrapper over `push_tracked`. A2 particle checkpoint items (x, v only) are NOT
  restorable by A3 and there is no migration — same-day change, no production data existed.
  Overhead re-measured: unchanged within noise (+30%/+76% for 1/3 ensembles × 32768) — the
  shared cell/weight work bought back what gathering the E_z pieces separately costs, so
  optimization (i) below is now spent without a win; docs/performance.md.)
- **Step placement**: mirror `_advance_forcing`. After the stepper returns `new_state`,
  push with dt = new_state.t − state.t (exact, adaptive-safe) and fields assembled from
  the PRE-step `state` — fields frozen at t_n over the step, first-order in the field
  evolution, which at 60–120 steps/gyration and ω_nl ≪ Ω is far below pusher error.
  (Time-centering with averaged fields is a possible later refinement; not Phase A.)
- **Pusher**: Boris, kick–drift–kick form (well-defined under varying dt), one particle
  → `vmap`; python loop over ensembles (masks and q/m are static per ensemble).
  `boris.py` holds the pure per-particle kernel — no jnp-only idioms that don't
  translate to WGSL (§9). Optional guiding-center integrator later, not Phase A.
- **Field assembly** (`particles/fields.py`): one function
  `particle_fields(state, kgrid, params) -> PFields` producing the real-space arrays
  (u, b_⊥, E_⊥, E_z pieces). It calls `gradk` on (φk, ψk) only (4 iffts — not
  `rmhd.grad`'s 8, vort/jpar gradients are not needed) plus the ≤2 conditional iffts of
  §2. Budget: ≤6 transforms per step vs ~30–36 in the 3-stage RHS — ≤15–20% overhead,
  measured in A2. (A0, 2026-08-18: revised to ≤8 — 4 gradient iffts, +2 for the dealiased
  ideal E_z, +1 resistive, +1 forcing.) (Reusing the stepper's stage-1 grads would shave most of this but
  invasively changes the stepper contract — noted as a later optimization, not Phase A.)
  (A2, 2026-08-18: measured at 256² CPU fp64 — `particle_fields` itself costs 2.24 ms,
  17% of the solver step, inside budget (docs/performance.md "Test particles overhead");
  the observed +26%/+74% total overhead is the O(N) `boris.push` gather, not this
  function. Two optimizations flagged here, not Phase A: (i) gather-side reorganization
  in `interp.gather` — share cell/weight computation across E and B, `jnp.take` on flat
  indices, cast samples not grids; (ii) reuse the stepper's stage-1 gradients here
  (removes 4 of the 6 fixed transforms) — the bigger lever, deferred because it changes
  the stepper contract.)
- **Interpolation** (`particles/interp.py`): periodic bilinear gather from the
  collocation grid, positions folded mod L. Spectral (exact) evaluation kept as a
  validation-only path — too expensive per particle in production, but it pins the
  interpolation error and gives gate 4 its exact variant. Known limitation, stated
  up front: independently interpolated b_⊥ is not exactly ẑ×∇(interpolated ψ);
  gates 4–5 measure the consequence rather than pretending it away.
- **Ensembles**: N ~ 1e4–1e5 total is negligible next to the FFTs (1e5 × O(10²) flops
  ≪ one 256² transform); the gathers are the only memory-irregular op.
- **No back-reaction**, ever, in this plan.
- **RNG**: particle init draws from `jax.random.key(params.particles["seed"])` — its own
  stream, used only at init; the push is deterministic, so the solver's forcing stream
  is untouched by construction. Gate 6 verifies bitwise anyway.
- **Checkpoint/restart**: particle state is saved as a SEPARATE orbax item alongside the
  state item at the same snapshot step (docs/checkpointing.md rules apply: bare
  StandardCheckpointHandler on reads, index broadcast from rank 0 unchanged). Old
  snapshots simply lack the item: restoring one with `params.particles` set is a hard
  error unless `"init_on_restart": true` — never a silent re-init. Restart must
  continue trajectories bitwise (gate 6's restart variant).
- **Diagnostics cadence**: `block_of_steps`'s scan emits per-step per-ensemble moment
  vectors as scan ys when particles are on (`None` ys, unchanged signature, when off);
  the `simulate_scan` driver appends them to a sidecar file in the snapshot dir. The
  while_loop-based `simulate` supports particles but only snapshot-cadence diagnostics —
  production heating runs use `simulate_scan`. Full particle state rides every snapshot;
  optional short high-cadence trajectory dumps exist for §9's future refvectors.
  (A2, 2026-08-18: sidecar is `<mngr.directory>/particle_moments.txt`, written only when
  `simulate_scan(..., save=True)`; off-signature is the plain `final_state`, not a `None`
  ys — block_of_steps returns `(carry, ys)` only on the particles-on branch, so there is
  no ys value to thread through the off path at all.)
- **MPI**: Phase A is `dims=2`, single-process (already the 2D rule). 3D z-decomposed
  runs need particle migration between z-ranks or replicated fields — that design is
  Phase B's first task, not constrained here beyond keeping `ParticleState` free of any
  rank-local assumption.

## 5. Diagnostics (`diagnostics/particles.py`)

Per the house convention: read-only, plain imports, dependency runs
diagnostics → particles, never back.

- ⟨v_⊥²⟩(t), ⟨v_z²⟩(t), ⟨v_z⟩(t) per ensemble (scan-ys time series); μ = v_⊥²/2|B|
  (local B) tracking and its diffusion coefficient.
- Heating rate Q_⊥ vs ξ = δu(ρ)/v_⊥ — the stochastic-heating exponential
  (Q_⊥ ∝ exp(−c₂/ξ)) is the headline science plot; measure c₂ in 2D RMHD and compare
  Chandran et al. 2010 (with the honest caveat that 2D lacks parallel decorrelation —
  quantifying that difference is part of what Phase B is for). δu(ρ) measured from the
  run's own spectra at k_⊥ρ ≈ 1.
- E_z-acceleration statistics conditioned on local |j_z| at the particle (sheet
  acceleration): P(Δv_z) tails, full-E_z vs ideal-Ohm ensembles — the §1 resistive
  split is the second headline plot. Physical caveat, stated wherever this is plotted:
  gate 4's exact 2D invariant means full-mask parallel energization is bounded by
  qΔψ/m along the orbit — 2D constrains v_z gains in a way 3D does not, which is a
  standing argument for Phase B.
- Energy bookkeeping: per-ensemble work integrals ∫q E·v dt split by field piece
  (ideal/resistive/forcing E_z, E_⊥), so the heating attribution is measured, not
  inferred. Quoted against the E = 0 control ensemble's floor.

(A3, 2026-08-18: landed as `taranis/diagnostics/particles.py` — `read_moments`,
`heating_rate`, `mu_diffusion`, `gyroradius`/`gyrofrequency`, `kinetic_spectrum`,
`delta_u`/`xi`/`chandran_fit`, `mu_of`, `jz_at`, `conditional_stats`, `increment_histogram`,
`increment_diffusion`, `work_split`, `energy_budget`; conventions in CLAUDE.md, tests in
`tests/test_particles_diagnostics.py`. The energy bookkeeping is exact rather than
integrated: `ParticleState.w` is credited inside the pusher (§4).

The 2D science run went through TWO passes the same day. Pass 1 (B0 = 1, Q_⊥ about ẑ, no
E∥ projection, hyper=1, rate windows spanning the whole particle run with no upper-limit
handling) found no Chandran exponential (c₂ = 0.02 ± 0.02 over ξ = 0.05–0.43) and argued that
in 2D the parallel channel must dominate because (q/m)ψ_rms/v_⊥ = ψ_rms/ρ ≥ 4 for every
resolved gyroradius. Against Xia, Perez, Chandran & Quataert 2013 (ApJ 776:90) that pass had
three confounds: B0 = 1 is δB/B₀ ~ 1 (Xia cap at 0.47 and flag it — B0 is the RMHD ε,
docs/numerics.md), Q_⊥ was measured about ẑ rather than the local B, and the ideal ensemble
carried a numerical E∥ (dealiased bracket + independent bilinear E and B). Pass 2 (A3b:
B0 = 10, β_i = 0.01, b_rms/B0 = 0.18; `epar_project=True` on every ideal ensemble; Q_⊥ from
`vperpB2`; hyper=3, ν=η=1e-10; Xia's window rule t₀ = 10/Ω, end at 1.2× ⟨v_⊥B²⟩, Q_⊥ ≤ 2σ
treated as an upper limit and held out of every fit; 44 min, 0.69 GB) is what
`examples/test-particles-2D.ipynb` now shows, per §5 bullet: (1) the sidecar carries the 11
moments; D_μ 2–3e-3 for the B0 = 10 ensembles (μ ∝ 1/B0), −6e-7 ± 6e-6 for the E=0 control;
the snapshot-pair estimator is 9–41× larger at Δt ≈ 2 gyroperiods (reversible μ oscillation —
it is an upper bound). (2) **The exponential is there**: 16 measurements + 6 upper limits over
the fitted range ξ = 0.124–0.39 give **c₂ = 0.40 ± 0.13, c₁ ≈ 1.0** (lab-frame δu, local-B Q_⊥;
drift-corrected δu 0.46 ± 0.09) — Chandran 2010 (0.75, 0.34), Xia 2013 (c₂ 0.44 → 0.20 with
resolution, c₁ 0.7–1.1). Sensitivity, all printed in the notebook: including the ≤2σ points
0.32 ± 0.11; ẑ frame same points 0.395; without design (a) 0.31 ± 0.17; design (b) alone
0.34 ± 0.10 (gyro); B0 = 1 cohort under the same protocol 0.33 ± 0.22; NO limit handling
(every Q > 0) 0.59 ± 0.07 — the fit is sensitive to the treatment of the near-zero points,
not to the frame (worth 0.001). The three ring limits at ξ = 0.065/0.125/0.147 require
c₂ ≥ 0.34/0.57/0.78; a plain power law Q ∝ δu^4.5 ρ^−0.45 scores slightly better on the 16
points (R² 0.92 vs 0.88) but overshoots those limits 6–49× where the exponential misses ≤14×.
Design (a) has almost no ξ lever here (δu ∝ ρ^0.18) and its per-host ρ-scaling at fixed
turbulence (+0.67 at eps=0.03, −1.08 at eps=0.3, vs the normalization's −0.46) brackets the
1/ρ of the model — a real statement about the ρ-scaling, not about c₂; design (c) (Xia's:
fixed ρ AT INSERTION only — E×B pickup moves ρ/dx by 1.7–3.7× before t₀) is what reaches
ξ ≤ 0.15 and supplies the limits. What B0 = ε demonstrably controls is the
parallel/perpendicular split, not c₂: the ε = 1 vs ε = 0.1 twins (same Ω, ρ, ξ, fields)
give w_ez/w_tot 0.89–0.95 vs 0.03–0.15 and Q_∥/Q_⊥ 0.5–0.6 vs 0.02 — the pass-1 "2D
structural" argument used ψ_rms/ρ where the correct ratio is ψ_rms/(B0·ρ). The pass-1 null
c₂ is therefore attributable to its measurement protocol (pickup inside the window, no
limit rule; worth up to +0.2 here) and/or its hyper=1 turbulence — the notebook cannot
separate the two and says so; it is a corrected measurement, not a diagnosis. (3) E_z:
numerical-E∥ energization is not resolvable at B0 = 10 (projected vs unprojected Δ⟨KE⟩ =
+2.3%, 1.2σ, independent draws; the sampled E·B before projection is 1.6e-3 of |E||B|);
full-mask p_z drift 0.5% of (q/m)ψ_rms vs ~120% for the ideal ensembles; the |j_z|
conditioning is a NEGATIVE — ⟨Δv_z²⟩ rises 1.8× across quantiles but the E = 0 control's
rises 1.7×, so it is the v×b_⊥ rotation locating small-scale structure, not sheet
acceleration. The physical "resistive acceleration" headline is dropped (hyper=3 makes the
piece a numerical regularization; the full-mask ensemble stays as the exactness check).
Closure 4e-14 of KE₀, controls at 6e-14. Open items: (i) an opt-in shared init draw
(per-ensemble `init_seed`) so projected/unprojected and ideal/full pairs are particle-paired
— the projection comparison would go from 1.2σ to a number at zero cost; (ii) separating
exponential from power law needs a measurable positive Q_⊥ at ξ ≲ 0.1, i.e. much longer
windows on the suppressed rings — the natural next run; (iii) dissipation onset landed at
k ≈ 29 rather than >40 (ν = 3e-11 piles up at the grid scale) — 512² or a longer average
widens the clean band; (iv) a third forcing host helps design (b) more than anything else;
(v) the short high-cadence trajectory dumps §4 anticipates remain wanted for any
particle-level conditioning; (vi) rerunning one pass-1 configuration under the pass-2
protocol would separate protocol from turbulence in the pass-1 null.)

## 6. Validation gates

Kernel gates (no solver; analytic fields through the same interp+push code path):

1. **Gyration**: uniform B₀ẑ, E = 0 — Boris conserves |v| to round-off (the rotation is
   norm-exact); assert that, plus phase error scaling ∝ (Ωdt)² over 1e4 gyrations.
   (A1, 2026-08-18: as implemented the norm-conservation run is the 1e4-gyration one; the
   phase-order fit uses ~20 gyrations, which is all the (Ωdt)² scaling needs.)
2. **E×B drift**: uniform E_⊥, B — gyro-averaged drift = E×B/B² to interpolation
   tolerance; signs pin the §2 conventions. (A1, 2026-08-18: the analytic fields make this
   a check of the PUSHER's convention; the field assembly's relative signs are pinned by
   the E·B = 0 identity in test_particles_coupled, its absolute sign by gate 4.)
3. **Grad-B drift**: static analytic ψ(x) profile — drift vs textbook v_⊥²/(2Ω)·∇B
   expression, convergent in ρ/L. (A1, 2026-08-18: the prescribed b_⊥ = ẑ×∇ψ with
   b_y = ε sin(kx) does NOT give the textbook v_∇B — it gives exactly 2× it for a v_∥ = 0
   launch, 3× for a v_z(0) = 0 launch, because the field direction shears at O(ε) while |B|
   varies only at O(ε²). The exact identity ⟨v_y⟩B₀ = ⟨v_z b_y⟩ plus the p_z invariant gives
   the factor analytically, and an independent RK integration confirms it. The gate is
   therefore split in two: a textbook-|B|(x) case that reproduces v_∇B, and the RMHD-b_⊥
   case against 2·v_∇B with ρ/L convergence. Derivation: tests/test_particles_kernel.py.)

Coupled gates (live solver):

4. **Canonical p_z invariant**: in 2D, z is ignorable, so p_z = m v_z − qψ(x,y,t) is
   conserved EXACTLY per particle — in live, time-dependent fields, not just frozen
   ones — provided the particle's E_z contains every piece of ∂ψ/∂t (full mask; sign
   per §2's A_z = −ψ, pinned in A0). Because it couples E_z, b_⊥, the interpolation
   and the pusher in one scalar, its drift is the sharpest whole-pipeline error gauge:
   frozen-field variant to isolate interp+pusher (bilinear O(dx²); spectral path =
   time-integration error only), live variant runs continuously in production on the
   full-∂ψ/∂t ensemble (one reason that ensemble is standard in production even though
   full mask is no longer the default). For the DEFAULT ideal-only ensembles p_z is
   deliberately NOT conserved — its drift IS the omitted resistive(+forcing)
   acceleration, a free cross-check on §5's work integrals.
   (A2, 2026-08-18: `tests/test_particles_coupled.py` implements four variants. Frozen
   fields, bilinear gather, E_z off (frozen ⇒ ∂ₜψ=0, so the invariant needs E_z=0 by
   construction — leaving `ez_ideal` on is the recorded discriminator, would-be order
   ~0): converges O(dx²), order 2.08 measured. Frozen fields, spectral gather
   (`boris.push(gather=interp.gather_spectral)`): converges O(dt²), order 2.00. Sign
   discrimination, v_z − qmψ (conserved) vs v_z + qmψ (not): ratio ~6000. Live
   (evolving) fields: full-∂ψ/∂t mask converges O(dt), order 0.95; the default
   ideal-only mask does not converge, order −0.02, ratio ~300 at the finest dt — the
   omitted resistive+forcing acceleration. docs/numerics.md's "Test particles" section
   carries the p_z derivation the tests check against.)
5. **E = 0 in-situ floor**: the §3 control ensemble in a live turbulent run — KE drift
   defines the numerical-heating floor; every heating claim quotes it. (A future fp32
   port reruns exactly this to learn what it may claim.)
   (A2, 2026-08-18: `test_e_zero_ensemble_is_the_heating_floor` in
   `tests/test_particles_coupled.py` — the E=0 control ensemble's per-particle |v|²
   drift in the same live run, floor 6.6e-15, against order-unity growth in the two
   E-carrying ensembles.)
6. **Solver untouched**: fields bitwise-identical with particles on vs off, and with
   `params.particles=None` vs current main; restart continues trajectories bitwise.
   The reference run is recorded BEFORE A2 wiring begins (standing RNG-adjacent rule).
   (A2, 2026-08-18: three tests in `tests/test_particles_coupled.py`. (a)
   `test_solver_output_matches_the_pre_a2_reference` — `params.particles=None` output
   bitwise against the pre-wiring npz (recorded by
   `tests/_gen_particles_gate6_reference.py` into
   `tests/data/particles_gate6_reference_fp{64,32}.npz` before the carry-tuple wiring
   landed, per the standing rule), with a host-match (hostname/jax/backend/python)
   soft-skip on the npz comparison. (b)
   `test_particles_on_leaves_the_solver_bitwise_identical` — particles-on vs
   particles-off bitwise in the same session, the `default/` snapshot tree
   byte-structurally unchanged, and a `particles/` item written every step. (c)
   `test_restart_continues_fields_and_trajectories_bitwise` — restart reproduces fields
   and x/v bitwise, run with `forcing_norm_per_step=False`: the default per-step
   normalization's driver-entry `_refresh_forcing_scale` recompute at dt=0 makes even a
   particle-free forced restart non-bitwise, a pre-existing solver issue filed
   separately, not an A2 regression.)
7. **E_z assembly consistency**: all pieces on, assembled E_z vs centered finite
   difference of ψ across a step — agreement at the stepper's order. Catches any sign
   or piece-bookkeeping error in §2 directly. (A0, 2026-08-18: implemented as a centered
   difference across two RAW stepper calls with the forcing state frozen, so ψ(t) is smooth;
   it converges at O(dt²), order 1.94 measured.)

## 7. Module layout

    taranis/particles/state.py     ParticleState, init (maxwellian/ring), checkpoint item helpers
    taranis/particles/boris.py     pure per-particle kernel (the WGSL-portable core)
    taranis/particles/fields.py    particle_fields: piece-decomposed real-space arrays (RMHD-only assert)
    taranis/particles/interp.py    periodic bilinear gather; spectral validation path
    taranis/diagnostics/particles.py
    tests/test_particles_kernel.py    gates 1–3 (no solver)
    tests/test_particles_coupled.py   gates 4–7 (fp64 marker; bootstrap + script_main per convention)
    tests/test_particles_config.py    config validation/normalization, params.json round trip,
                                       init/template/moments, run.py pstate contract, sidecar (A2)

`run.py` changes are confined to the carry tuple, `_advance_particles` (the
`_advance_forcing` mirror), scan-ys plumbing, and the snapshot item — all statically
gated on `params.particles`.

## 8. Phases

- **A0 — conventions + field assembly.** §2 derivation into docs/numerics.md;
  `particle_fields` with the piece decomposition and mask; spectral validation path;
  gate 7 standalone against a short run.
- **A1 — kernel.** `boris.py` + `interp.py` + gates 1–3. No solver coupling.
- **A2 — co-stepping.** Carry tuple, `_advance_particles`, RNG stream, checkpoint item,
  diagnostics ys; gates 4–6 (reference recorded first); overhead measured.
  (Landed 2026-08-18: core wiring + docs + gates 4–6 all done. Overhead measured at
  256², CPU, fp64, quiet machine: `particle_fields`'s fixed transform cost is 17% of
  the solver step — inside the §4 ≤15–20% budget; the O(N) `boris.push` gather, at the
  dense 0.5-particles-per-grid-point loading used here, pushes the observed total to
  +26% (1 default-mask ensemble/32768) / +74% (3 ensembles/32768 each), see
  docs/performance.md "Test particles overhead". Gather optimization and stage-1-grad
  reuse are deferred, flagged in §4.)
- **A3 — physics production.** `diagnostics/particles.py`, paired resistive-split
  ensembles, both ξ-scan designs (§10), the two headline plots, 2D science notebook in
  `examples/`.
  (Landed 2026-08-18 in two passes, A3 then A3b: accumulators + diagnostics module +
  `examples/test-particles-2D.ipynb` with `particles_2d_run.py` (~44 min, ~0.7 GB, M1 laptop,
  fp64). Both headline plots exist. Headline 1 finds the Chandran exponential once the run is
  done Xia-style (B0 = 10, local-B frame, E∥ projected, upper-limit rule): c₂ = 0.40 ± 0.13,
  c₁ ≈ 1.0. The first pass's null result is attributed to its measurement protocol and/or
  base turbulence, NOT to 2D and not to the amplitude alone (§5 A3 note); the amplitude is
  what sets the parallel/perpendicular split. Headline 2 is the E∥/p_z consistency set; the
  resistive-acceleration framing is dropped. Open items in §5.)
- **B — 3D.** Scheduled BEFORE the WebGPU port (§10 decision). First task is the
  distributed-particle design (z-rank migration vs replicated fields), then the same
  gates in 3D — §2's field assembly carries over unchanged — then the 3D science
  notebook in `examples/` (the 2D-vs-3D parallel-energization comparison is the point:
  gate 4's qΔψ/m bound does not constrain 3D).
- **C — WebGPU port** (FUTURE, after B): see §9.

Execution per the standing flow: sonnet/opus implement per phase against this plan,
session Fable oversees, adversarial review by a fresh Fable per phase; comments say what
the code does — design history stays here.

## 9. WebGPU port (Phase C — deferred until after Phase B, constraints only)

Deferred; the game plan consumes it when it exists. Phase A/B choices that keep it
cheap, and which must not be broken meanwhile:

- `boris.py` stays a pure per-particle kernel with a WGSL-translatable structure.
- The mask of §3 is static config — trivially a WGSL specialization constant.
- taranis can dump short recorded trajectory sets (fields + particle states at fixed
  dt) for future refvector-style validation; gates 1, 4, 5 are the on-GPU reruns, and
  gate 5's fp32 floor defines what the port may claim.

## 10. Decisions (Alfred, 2026-08-18 — formerly open questions)

- **Bookkeeping**: B₀ = 1, interpret q/m; no ε parameter in code (§2).
- **ξ-scan: run BOTH designs** — they test different things. Sweeping q/m at fixed
  v_th moves ρ through the spectrum, so ξ varies via δu(ρ) with the turbulence held
  fixed but the gyro-numerics (ρ/dx, Ω dt) changing; sweeping forcing amplitude at
  fixed ρ/dx varies ξ with the particle numerics held fixed but the turbulence
  changing. Agreement between the two Q_⊥(ξ) curves is itself evidence the exponential
  is physics, not numerics. Ring init at several v_⊥ per run supplements both.
- **Particle E_z defaults to the ideal piece only**; resistive and forcing pieces are
  toggleable ON per ensemble. They are not physical E for the collisionless use case
  (§1, §3).
- **Particles see only the ideal E, and its numerical E∥ is projected out** (A3b,
  2026-08-18, after Xia, Perez, Chandran & Quataert 2013, ApJ 776, 90): independently
  interpolated E and B leave `Ē·B̄ ≈ 1.4·10⁻³|E||B|` at the particle even though the grid
  fields are exactly orthogonal, so the ideal production ensembles run with
  `epar_project = True` (their eq. 21). Ensembles carrying the non-ideal E_z pieces must
  NOT be projected — that E∥ is the thing they measure.
- **B₀ is the amplitude parameter, B₀ = 1/ε** (A3b, 2026-08-18 — this supersedes the
  "B₀ = 1, interpret q/m" bookkeeping decision above): B₀ = 1 is δB/B₀ ~ 1, which no RMHD
  ordering supports; production runs B₀ ~ 10 with q/m scaled by 1/B₀, keeping Ω = qm·B₀ and
  therefore ρ, Ω·dt and ξ fixed while β_i = v_th²/B₀². Derivation in docs/numerics.md
  ("E∥ projection and the amplitude parameter"). B₀ is per ensemble.
- **Heating is measured in the local-B frame** (A3b, 2026-08-18; Xia et al. §4.1): the
  `vperpB2`/`vparB2` moment columns, not the ẑ-referred `vperp2`/`vz2` — the field
  direction tilts at O(ε), so the ẑ split mixes ⊥ and ∥ at first order.
- **Base turbulence at `hyper = 3`** for the science run (A3b, 2026-08-18): the inertial
  range is what the ξ-scan needs, and the resistive E_z piece is no longer a physics target
  — only the ∂ψ/∂t-exactness check (gates 4 full-mask variant, 7) still uses it.
- **Phase B (3D) before Phase C (WebGPU)** (§8).
- **Science notebooks (2D, 3D) live in `examples/`** (§8), following the existing
  notebook conventions there.
