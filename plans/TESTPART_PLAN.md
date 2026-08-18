# TESTPART_PLAN — charged test particles in taranis

Status: PLAN, 2026-08-18, rev 2 (fleshed out same day from the sketch; rev 2 folds in
Alfred's answers to the open questions — now recorded as decisions in §10, the headline
one being that particles see the IDEAL E_z only by default). Two scope decisions since
the sketch: (1) the resistive contribution to E_z is switchable for the particles
independently of the solver — it is one bit of a general per-ensemble field mask (§3);
(2) the WebGPU port is deferred to Phase C, after Phase B — §9 keeps only the
portability constraints Phases A/B must not break. Sequenced before the webgpu game
(`plans-webgpu/GAME_PLAN.md` eventually consumes the Phase-C kernel); nothing here
blocks on the game or vice versa.

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
5. **E = 0 in-situ floor**: the §3 control ensemble in a live turbulent run — KE drift
   defines the numerical-heating floor; every heating claim quotes it. (A future fp32
   port reruns exactly this to learn what it may claim.)
6. **Solver untouched**: fields bitwise-identical with particles on vs off, and with
   `params.particles=None` vs current main; restart continues trajectories bitwise.
   The reference run is recorded BEFORE A2 wiring begins (standing RNG-adjacent rule).
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
- **A3 — physics production.** `diagnostics/particles.py`, paired resistive-split
  ensembles, both ξ-scan designs (§10), the two headline plots, 2D science notebook in
  `examples/`.
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
- **Phase B (3D) before Phase C (WebGPU)** (§8).
- **Science notebooks (2D, 3D) live in `examples/`** (§8), following the existing
  notebook conventions there.
