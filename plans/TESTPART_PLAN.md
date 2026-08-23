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
(B1, 2026-08-19: two 3D corrections to this section. (i) The induction equation above is
written with a B₀ coefficient on ∂_zφ; the SOLVER's coefficient is exactly **1**
(`rmhd.linear_matrix`'s off-diagonal is `1j·kz`, `FDLinearTerm` returns a bare `df_dz`),
i.e. taranis writes RMHD in units where the guide field/v_A is 1. The cancellation that makes
E_z dimension-independent therefore holds only at **B₀ = 1**, and `Parameters` now rejects
any other B₀ when `dims == 3` — in 2D there is no Alfvén term and B₀ stays the free
per-ensemble amplitude knob of A3b. The 3D amplitude parameter ε = rms|∇ψ| is set by the run
(forcing amplitude and Lz: v_A = 1 on Lz is the same system as v_A = B₀ on B₀·Lz).
(ii) The **finite-difference-z filter is folded into the resistive piece**: `ez_resistive`
is redefined as the full linear non-ideal EMF on ψ — the ψ diagonal of `rmhd.linear_matrix`
plus `−z_diss·(dz/2)⁴∂_z⁴ψ` when `dims==3 and not z_spectral` (`fields._psi_non_ideal`; the
Alfvén half of `FDLinearTerm` stays out, being the ∂_zφ term already cancelled). It is NOT a
new piece: `FIELD_PIECES` stays 5 long, `WORK_PIECES` 4, `ParticleState.w`, `MOMENTS` and the
checkpoint layout unchanged. Measured on a 16²×8 state, full-mask E_z against the solver's own
`dψ/dt − ∂_zφ`: 7.9e-16 relative in both z modes, with the filter contributing rms 5.4e-4
against the k-local diagonal's 1.1e-3 — omitting it would be a visible O(z_diss) defect, not a
rounding detail. Both derivations are in docs/numerics.md.)
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

(B1, 2026-08-19: `ez_resistive` now means the full linear non-ideal EMF on ψ — the k-local ψ
diagonal plus the finite-difference-z ∂_z⁴ filter in FD-z 3D (§2). The bit's meaning is
otherwise unchanged, it is still off by default, and the "full-∂ψ/∂t ensemble" configuration
below is now exact in 3D as well as 2D. The per-ensemble `B0` is restricted to 1.0 when
`dims == 3` (§2, §10); every other key is dimension-independent. `normalize_config` stays
dimension-agnostic — the B₀ rule is enforced in `Parameters`, which is where `dims` lives.)

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
  (B2, 2026-08-19: **the budget is exceeded in one 3D configuration.** At 128²×16,
  `particle_fields` is 17.4% of the solver step under finite-difference z with the default
  mask and **22.9% with the non-ideal pieces on** — outside the ≤15–20% above; under
  `z_spectral` it is 12.9%/14.4%, but only as a share, that solver step being ~1.8× more
  expensive in absolute terms. The excess is one item: the ∂_z⁴ filter's z-stencil and its
  halo exchange, timed for the first time at 1.86 ms ≈ 4.4% of the step (the resistive piece
  costs 1.50 ms as the k-local ψ diagonal alone, 3.36 ms with the filter). It buys exactness
  — without it gate 7 stops converging (§6) — so it is a priced feature, not a regression.
  Optimization (i) below is the lever if the price ever needs paying down; the trilinear
  gather also roughly doubles the O(N) push, as 8 corners against 4 predicts.
  docs/performance.md "Test particles overhead in 3D".)
- **Interpolation** (`particles/interp.py`): periodic bilinear gather from the
  collocation grid, positions folded mod L. Spectral (exact) evaluation kept as a
  validation-only path — too expensive per particle in production, but it pins the
  interpolation error and gives gate 4 its exact variant. Known limitation, stated
  up front: independently interpolated b_⊥ is not exactly ẑ×∇(interpolated ψ);
  gates 4–5 measure the consequence rather than pretending it away.
  (B1, 2026-08-19: `gather` is periodic **trilinear** when the grid carries a z axis and
  stays exactly the previous bilinear code path at `nz == 1`, so no 2D result moves; the fold
  gained the z coordinate (mod Lz, cell from `params.dz`, wrap at the top cell) and positions
  stay unfolded. `grid_coords` returns `(x, y, z)`, z being the single plane 0 in 2D. The
  validation path is exact for what the representation actually IS, which differs by mode:
  under `z_spectral` the stored arrays are rfftn over (z,x,y), so `gather_spectral` is
  spectral in all three directions; under finite-difference z they are one rfft2 per z plane,
  so it is perp-spectral and linear in z — exact for a z-independent field, which is what the
  embedded-2D gate needs.)
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

  (B1, 2026-08-19: 3D landed **single-process only** — `size == 1` and
  `comm_backend != "jax"`, a ValueError otherwise pointing here; `ParticleState` and every
  interface stayed rank-agnostic, so nothing below is foreclosed. **Neither design is
  implemented.** The two candidates:

  1. **Replicated fields (allgather).** Every rank gathers the full z domain of the ~8
     real-space particle arrays and pushes a replicated copy of every particle, then keeps
     only its own slice of the answer (or all ranks push all particles and agree bitwise).
     Cost: one allgather of `8 · nz · nx · ny` floats per step — at 512³ fp64 that is ~8 GB
     per rank per step, i.e. hopeless at production size, but at modest nz it is a few
     percent and the code change is confined to `particle_fields`/`gather`. No migration, no
     load imbalance, no change to the checkpoint layout, and the push stays bitwise
     reproducible against a serial run.
  2. **Rank-owned particles + padded all-to-all migration.** Each rank owns the particles
     currently inside its z slab and gathers from its own planes plus a 1-plane halo (the
     trilinear stencil needs exactly one neighbour plane, and `comms.halo_exchange` already
     provides it). After the drift, particles that crossed a slab boundary are packed into a
     fixed-size per-neighbour buffer and exchanged; fixed size is what keeps the whole thing
     jittable, so the buffer must be capacity-checked (a `psum` of the overflow count, raised
     as a run-time error rather than silently dropping particles). Cost: O(particles that
     crossed) per step, which at `v_z·dt ≪ dz` is a few percent of N — the only design that
     scales. Costs elsewhere: `ParticleState` leaves become rank-local with a varying
     occupancy mask, so every moment becomes an allreduce, the checkpoint item has to record
     ownership (or be gathered to a canonical order on write), and per-particle bitwise
     reproducibility across decompositions is lost.

  **Build (1) first** if 3D multi-rank is ever wanted: it is a day's work, it is exactly
  right for the small-nz, many-perp-modes runs Phase B actually needs, and it validates the
  physics before (2)'s bookkeeping is worth paying for. (2) is the production answer and
  should not be attempted until a measured field-allgather cost says so.)

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

(B3, 2026-08-19: what the 3D campaign (§11) adds or changes on the diagnostics side.
`diagnostics/particles.py` needed **no** change for 3D — `read_moments`, `heating_rate`,
`mu_diffusion`, `mu_of`, `jz_at`, `kinetic_spectrum`/`delta_u`/`xi`/`chandran_fit`,
`increment_*`, `work_split`, `energy_budget` all work unchanged against a `dims=3` run
(`perpspec` is z-averaged, which is exactly the spectrum `delta_u(rho)` wants, and
`mu_of`/`jz_at` gather through B1's trilinear path). Three additions live in
`examples/test-particles-3D.ipynb` rather than in the module, and are promotion candidates:

- **Xia et al.'s `delta_u`** — `delta_u² = ∫E_u dk` over one e-fold centred on `k_ρ` (their
  eq. 6) — computed beside taranis's `sqrt(2 k E_kin)`. They agree to a few percent on a
  `k^{-3/2}` spectrum but the measured ratio is **0.66–0.91** on the smoke profile's steep one,
  so the ratio is reported per run and `c₂` is fitted both ways in the sensitivity table.
- **the control ensembles run through the same window machinery**, so the numerical floor is
  quoted as a RATE next to the measurements instead of only as a `|v|²` drift; the notebook
  prints smallest-fitted-`Q_⊥` over largest-control-`|Q_⊥|` and calls anything under ~10× a
  measurement of the pipeline. Recommended for the 2D notebook too.
- **`parspec`** (size==1 only, which every particle run is) becomes a *particle* diagnostic in
  3D: the parallel-resolution check with no 2D analogue.

`p_z = v_z − qm·ψ` stops being an invariant in 3D, so the new statistic is the RATIO of its
full-mask drift to the same quantity in a matched 2D twin run, where it is pure discretization
error — 1.65 vs 0.043 of `(q/m)ψ_rms`, a factor 38, already at 32²×16.

Measured while designing, and worth stating because it reads on gate 5: **the E = 0 control is
blind to `TARANIS_PRECISION`.** Its mask makes `E` exactly zero, so the push is a pure fp64
rotation whatever the field precision, and its floor is the same to four digits at fp32 and
fp64 fields (`7.105e-15`). Gate 5 bounds the PUSHER, not the interpolation noise an fp32
*field* run introduces; §9's reading of it — an fp32 particle STATE on GPU — is unaffected, but
an fp32 field run has to be validated statistically instead (§11.6).

New open items, beside (i)–(vi) above which all still stand: (vii) `β_i` and `ξ` are **locked**
in 3D because `B₀` is pinned — Xia's per-cohort `(v_A, L_∥)` rescaling is exactly the
per-ensemble `B₀` the solver forbids there — so unlocking it means per-ensemble `(B₀, L_z)`
pairs, a `Parameters`/`fields.py` change rather than a campaign change; (viii) promote the band
`delta_u` and the control-as-a-rate helper into `diagnostics/particles.py` once §11's notebook
has settled; (ix) item (i)'s per-particle paired init blocks the 3D projected/unprojected
comparison exactly as it blocked the 2D one, and at 32768 particles per ensemble it still
would.)

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
   separately (fixed 2026-08-22, REFACTOR_PLAN Phase 0c), not an A2 regression.)
7. **E_z assembly consistency**: all pieces on, assembled E_z vs centered finite
   difference of ψ across a step — agreement at the stepper's order. Catches any sign
   or piece-bookkeeping error in §2 directly. (A0, 2026-08-18: implemented as a centered
   difference across two RAW stepper calls with the forcing state frozen, so ψ(t) is smooth;
   it converges at O(dt²), order 1.94 measured.)

Two gates added after Phase B (2026-08-19), both closing gaps in the PUSHER's coverage
rather than the assembly's:

10. **Mirror force and μ** (kernel, `tests/test_particles_kernel.py`): gates 1–3 all
    arrange `b̂·∇|B| = 0`, so nothing tested the parallel dynamics. The gap is 3D-only —
    in a static z-independent field with E = 0 the exactly-conserved |v| and
    p_z = v_z − qm·ψ already determine the parallel/perpendicular split from the
    particle's perpendicular position, so the mirror channel is not independent there
    (gates 1/5 and 4c pin both invariants). A static analytic `B = B₀ẑ + ẑ×∇ψ(x,z)` —
    divergence-free for any ψ, the RMHD structure, so it drives B1's trilinear gather —
    gives a mirror along z; the gate asserts the reflection point against
    `|B|_turn = |B|₀·|v|²/v_⊥0²` over a v_∥0 sweep, the SCALING of μ's violation, a
    loss-cone particle, and the z-independent control.
11. **Varying dt**: the KDK driver exists so that dt may change from step to step, and
    nothing tested it. Kernel half (`tests/test_particles_kernel.py`): a jittered dt
    sequence at fixed total time — |v| still exact, the orbit still second order in
    max(dt), the E×B drift still exact. Coupled half
    (`tests/test_particles_coupled.py`): gate 4's live p_z invariant under
    `adaptive_timestep=True`, converging in `cfl_safety` rather than dt.

(B2, 2026-08-19: the 3D coupled gates landed in `tests/test_particles_3d.py`, single-process,
**every one of them run in both z modes**. What changed, gate by gate, and what was measured:

- **Gate 4 becomes the EMBEDDED-2D gate.** In 3D z is ignorable only where the fields have
  no z structure, so the carrier is a 3D box (nz = 8) started z-independent with forcing OFF —
  the O-U z envelope is the thing that would break z-independence. Both halves are measured
  rather than assumed (`test_z_independent_fields_stay_z_independent_unless_forced`): an
  unforced z-independent state stays z-independent to **0.0 exactly** after 50 steps in both
  modes, while 20 forced steps give a z spread of 1.39 on fields of order 1. The four 2D
  variants then port unchanged. Frozen fields + trilinear gather at ρ/dx = 0.51/1.02/2.04:
  **order 2.11** (drifts 3.98e-2 → 2.14e-3 over T = 2.5). Frozen + `gather_spectral`:
  **order 2.00**. Sign discrimination v_z − qmψ vs v_z + qmψ: ratio **1056**. Live (unforced,
  evolving) fields at 64²×8, `diss=0.3`: full mask **order 0.84**, drift 1.6e-3 of qm·rms(ψ);
  ideal-only order −0.00 and **271×** the full-mask drift. The particles carry v_z and their
  positions are left unfolded, so they cross the z boundary 1.4–1.7 box lengths per run — the
  z fold and the z blend are live throughout even though a z-independent field makes the blend
  exact. Both z modes agree to the printed digit on every one of these numbers, which is itself
  the statement that an embedded-2D 3D run is the 2D system.
- **Gate 7 in 3D compares against `Δψ/Δt − ∂_zφ`**, the Alfvén term being the solver's, not the
  particle's. `∂_zφ` is formed the solver's way in each mode (`1j·kz·φk` with the kz-Nyquist
  plane zeroed; the `shared_physics` 4th-order stencil). Converges at **order 1.995 (FD-z) /
  2.000 (z_spectral)**, residual 2.49e-4 of max|E_z| at dt = 0.01, against a max|∂_zφ| that is
  24% of max|E_z| — the subtraction is not cosmetic. **The FD-z filter discriminator**: taking
  the ∂_z⁴ filter back out of `ez_resistive` stops the convergence dead (order 0.033) at a floor
  of 5.98e-3 = 1.25e-3 of max|E_z|, **80×** the filtered residual at the finest dt. B1's fold is
  load-bearing, not a rounding detail.
- **B₀ = 1 in 3D is physics, shown two ways** (`test_B0_must_be_one_in_3d`). (a) On the grid,
  the E_z the derivation demands (`−B₀∂_zφ + ∂ψ/∂t`, ideal + Alfvén part `−{φ,ψ} + ∂_zφ`, raw
  bracket) gives `max|E·B| = B₀|1−B₀|·max|∂_zφ|` to 1e-10 relative at B₀ = 1.5, 2, 10 and
  exactly 0 at B₀ = 1 — while the SAME sweep in 2D stays at 1.7e-16 relative for every B₀,
  which is why B₀ remains a free per-ensemble knob there. (b) Gate 7's residual at B₀ ≠ 1 is
  the dropped `(1−B₀)∂_zφ` to within 1% (0.576/1.15/10.4 vs the B₀ = 1 residual 1.19e-3).
- **Gates 5, 8 and 9** run off one live FORCED, genuinely 3D turbulent run (32²×8, five
  ensembles from one draw, so the ideal/full and projected/unprojected pairs are
  particle-paired). Gate 5: E = 0 control floor **3.1e-15 (FD-z) / 3.7e-15 (z_spectral)**
  per-particle |v|² relative drift, against 1.04–1.68 mean |v|² growth in the four E-carrying
  ensembles. Gate 8: closure **≤ 3.9e-14 of KE₀** at fp64 and **≤ 1.9e-14** at fp32, and every
  piece an ensemble's mask omits exactly zero — so the non-ideal energization between the
  particle-paired ideal and full ensembles is carried entirely by `w_ez_resistive` and
  `w_ez_forcing`, which stay identically zero in the ideal-only twin.
  Gate 9: projected `rms(E'·B)/rms(|E||B|)` **3.5e-17/4.4e-17**, unprojected **4.9e-2/1.14e-1**
  (much larger than 2D's 1.4e-3 — this grid is coarse and strongly forced), orbits separating
  by 5.3e-2 from the same draw, and the work closure surviving the projection.
- **Gate 6 in 3D**: particles-on vs particles-off bitwise in the same session (fields, forcing
  state/scale/key, t), the state item's leaf list and `<step>/default/` tree untouched, a
  `particles/` item at every written step, a sidecar with a whole number of (step, ensemble)
  rows, and a restart that continues fields, trajectories and `w` bitwise — both z modes, with
  `forcing_norm_per_step=False` for the same pre-existing reason as the 2D gate 6c (fixed
  2026-08-22, REFACTOR_PLAN Phase 0c). **No new
  reference npz**: the committed one records pre-A2 2D main and is about the carry wiring,
  which 3D does not touch (`run.py` needed no edit for B1).
- **z-specific interpolation checks**: the trilinear gather reproduces every collocation value
  exactly (0.0), blends the top z cell with plane 0, folds positions 5–7 box lengths outside
  [0, Lz) to the same sample (1.0e-14), converges at **order 1.90 in dz** on a smooth field of z
  alone, matches `gather_spectral` to 3.0e-15 under `z_spectral` (where the trilinear gather is
  off by 0.62), and equals its own bilinear self on a z-independent field to 1.1e-16.

No gate was invented where 3D has no analogue: kernel gates 1–3 drive analytic fields through
the pusher with no solver and no z structure, so the 2D file already exercises the identical
code, and gate 6a's reference npz is 2D by construction. Both statements are in the new file's
module comment. 3D overhead is in docs/performance.md.)

(2026-08-19: **gates 10 and 11 landed**, closing the two pusher gaps above. Both are in the
KERNEL file except gate 11's coupled half — gate 10 needs a 3D grid but no solver, so the
"no 3D kernel analogue" sentence in `tests/test_particles_3d.py` was amended to point at it
rather than moved. Neither is precision-marked except gate 11's coupled half (a convergence
fit): the fp32 numbers below reproduce the fp64 ones to ~5 digits, the fields being analytic
constants stored at field precision and every asserted quantity being ≥ 1e-5.

**Gate 10** — `B = (0, b·cos(k_x x)·cos(k_z z), B₀)`, i.e. ψ = (b/k_x)·sin(k_x x)·cos(k_z z),
at `b/B₀ = 0.5` (mirror ratio 1.1180, deliberately outside the RMHD ordering — it buys a
usable mirror and no solver is in the loop), 64×8×64, four quadrature gyrophases launched in
the well at z = L_z/4 where `cos(k_z z) = 0` and the field is exactly B₀ẑ.
(a) **Turning point**: at ρ·k_z = 0.05, the sweep v_∥0/v_⊥0 = 0.1/0.2/0.3 predicts
`|B|_turn/|B|₀` = 1.01/1.04/1.09 (a 9× lever on the mirror depth) and measures it to
3.9e-4/1.9e-3/4.9e-3 per particle, **2.6e-5/4.2e-5/4.4e-4 averaged over the four gyrophases**
— the per-particle spread being the O(ρ) gyrophase term. `|v|²` conserved to ≤ 1.0e-14
throughout (E = 0). Refinement controls, which is what makes those deviations *physics*:
doubling the steps per gyration moves `|B|_turn` by 4.5e-5 (0.9% of the deviation) and the μ
excursion by 0.06%; doubling the grid moves them by 7.2e-4 (15%) and 0.46%.
(b) **μ is asserted by its SCALING, not by a number.** Over an 8× sweep of ρ·k_z
(0.1 → 0.0125, through q/m) the largest excursion `max|Δμ/μ|` = 2.02e-2, 9.40e-3, 4.64e-3,
2.31e-3 — **order 1.041**, with the ratio to ρ·k_z flat at 0.202/0.188/0.185/0.185. That is
the *reversible finite-Larmor* term (the field at the particle differs from the field at the
guiding centre at O(ρ∇), and the invariant itself carries an O(ρ) gyrophase correction), NOT
a secular drift, and first order is what it should be — the textbook exponentially-small
secular part is far below it and is not what a bouncing orbit exhibits. The reversibility is
measured rather than asserted: the gyrophase-averaged net Δμ/μ back at the launch plane
(where |B| = B₀ for every x, so v_∥ = v_z with no gyro-ripple) is 2.3e-3, 2.5e-3, 1.2e-4,
**1.0e-5** — 39× and 230× below the excursion en route at the two most adiabatic points. The
turning-point error converges at order 0.971 in ρ·k_z on the same sweep.
(c) **Controls**: a particle at v_∥0/v_⊥0 = 0.5, inside the loss cone (the bound is
√(R−1) = 0.3436), never reflects and streams to |z − z₀| = 6.16 > L_z/4 while the trapped
ones never leave 1.13; and the SAME field with its z dependence removed has |B| constant
along a field line, where **v_∥ is constant to 8.9e-4 of v_⊥** and nothing is trapped — the
numerical form of the "this gap is 3D-only" argument above.

**Gate 11** — (kernel) dt jittered ±60% step to step (realized max/min 3.97) and renormalized
to a fixed total time of 20 gyrations, so refining means more steps over the SAME elapsed
time: `|v_⊥|` drift ≤ 2.7e-15 at every resolution, and the orbit converges to the analytic
gyration at **order 1.974** in max(dt) (1.75e-1 → 2.75e-3 over 400 → 3200 steps). The E×B
drift under the same sequence is exact to **3.3e-17/7.3e-18** of |E|, not gate 2's 1e-3 —
for uniform fields the Boris map fixes v = u, so the quadrature average advances at exactly
u per unit time however the steps are chosen.
(coupled) gate 4 live at 64², `adaptive_timestep=True`, cfl_safety 0.5/0.25/0.125 → 48/95/190
steps to t = 0.818/0.816/0.826 (the same elapsed time to 1.3%, since each run takes
`ceil(T/dt(t₀))` steps and dt(t₀) is its largest), with dt varying by **1.39–1.42 within a
run** and mean dt 0.01705/0.00858/0.00435. Full-mask rms|Δp_z| per unit time
4.92e-2/2.43e-2/1.29e-2, **order 0.978** against the fixed-dt gate's 0.948; ideal-only order
−0.025 and 182× the full-mask drift at the finest. Since the runs end at different times,
everything is quoted per unit time, and the O(dt) constant `drift/(t·dt)` is
2.89/2.84/2.97 adaptive against 2.56/2.62/2.75 fixed — **agreeing to 9.8%**, which is the
statement that a varying dt costs the invariant nothing beyond the dt it actually took.)

## 7. Module layout

    taranis/particles/state.py     ParticleState, init (maxwellian/ring), checkpoint item helpers
    taranis/particles/boris.py     pure per-particle kernel (the WGSL-portable core)
    taranis/particles/fields.py    particle_fields: piece-decomposed real-space arrays (RMHD-only assert)
    taranis/particles/interp.py    periodic bi/trilinear gather; spectral validation path
    taranis/diagnostics/particles.py
    tests/test_particles_kernel.py    gates 1–3 (no solver)
    tests/test_particles_coupled.py   gates 4–9, 2D (fp64 marker; bootstrap + script_main per convention)
    tests/test_particles_config.py    config validation/normalization, params.json round trip,
                                       init/template/moments, run.py pstate contract, sidecar (A2)
    tests/test_particles_3d.py        the same gates in 3D, both z modes, plus the B₀ = 1
                                       discriminator and the z interpolation checks (B2)

`run.py` changes are confined to the carry tuple, `_advance_particles` (the
`_advance_forcing` mirror), scan-ys plumbing, and the snapshot item — all statically
gated on `params.particles`.

(B1, 2026-08-19: the layout survived 3D unchanged — `run.py` needed **no** edit at all
(`_advance_particles` is already dimension-agnostic), and neither did `boris.py`,
`snapshot_io.py` or `diagnostics/particles.py`'s numerics. The 3D work landed in `interp.py`
(trilinear gather, z in `grid_coords`, the two exactness modes of `gather_spectral`),
`fields.py` (`_psi_non_ideal`), `state.py` (the z draw in `init_particles`) and `config.py`
(the relaxed gate and the B₀ = 1 rule). 3D config validation, the 3D z draw and the B₀
rejection are in `tests/test_particles_config.py`; the 3D physics gates are B2's.)

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
  (B1 landed 2026-08-19: 3D machinery, single-process, both z modes. Trilinear gather, the
  FD-z filter folded into `ez_resistive`, uniform-in-z init, the relaxed `Parameters` gate
  and the B₀ = 1-in-3D rule (§2, §3, §10); `run.py` unchanged. The distributed design was
  WRITTEN, not built — §4's MPI bullet carries both candidates and the recommendation;
  multi-rank stays a ValueError. Verified end to end in both z modes: full-mask E_z matches
  the solver's own `dψ/dt − ∂_zφ` to 7.9e-16 relative, and the 2D reference gates
  (6/6b/6c, 7, 8, 9) stay bitwise green. Still to come: B2, the 3D gates — the embedded-2D
  p_z gate (z ignorable only for z-independent fields, docs/numerics.md), gate 7 in 3D, the
  E = 0 floor and the E∥ projection at 3D — and B3, the 3D science notebook.)
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
  (B3/review, 2026-08-19: read that last clause narrowly. Gate 5's control masks E to
  exactly zero, so its push is a pure fp64 rotation whose floor is blind to FIELD precision
  — measured identical in both precisions (§5). It bounds pusher arithmetic; it never
  bounded interpolation noise. The port puts the PARTICLE state in fp32 too, where the
  control does move, so the sentence holds there — but even then it bounds the pusher, and
  the interpolation error needs the statistical check of §11.6.)

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
- **B₀ = 1 in 3D, and only in 3D** (B1, 2026-08-19): the solver's Alfvén coefficient is
  exactly 1, so `E_z = −B₀∂_zφ + ∂ψ/∂t` collapses to the ideal `−{φ,ψ}` only at B₀ = 1; any
  other value leaves `(1−B₀)∂_zφ`, i.e. `E·B ≠ 0` and fields that are not ideal-Ohm.
  `Parameters` rejects it. This does NOT retract the A3b decision above: in 2D there is no
  Alfvén term and B₀ stays the per-ensemble amplitude knob. In 3D the amplitude parameter is
  the run's own ε = rms|∇ψ|, tuned through the forcing amplitude and Lz (v_A = 1 on Lz ≡
  v_A = B₀ on B₀·Lz), which is the RMHD ordering L_z/L_⊥ ~ 1/ε made explicit. Derivation in
  docs/numerics.md, "E∥ projection and the amplitude parameter".
- **The FD-z filter is part of `ez_resistive`** (B1, 2026-08-19): the particles' non-ideal
  E_z piece is the full linear non-ideal EMF on ψ, not just the k-local diagonal (§2, §3), so
  full-mask E_z is exactly ∂ψ/∂t in both 3D z modes. No new field piece, no new work column.
- **Heating is measured in the local-B frame** (A3b, 2026-08-18; Xia et al. §4.1): the
  `vperpB2`/`vparB2` moment columns, not the ẑ-referred `vperp2`/`vz2` — the field
  direction tilts at O(ε), so the ẑ split mixes ⊥ and ∥ at first order.
- **Base turbulence at `hyper = 3`** for the science run (A3b, 2026-08-18): the inertial
  range is what the ξ-scan needs, and the resistive E_z piece is no longer a physics target
  — only the ∂ψ/∂t-exactness check (gates 4 full-mask variant, 7) still uses it.
- **Phase B (3D) before Phase C (WebGPU)** (§8).
- **Science notebooks (2D, 3D) live in `examples/`** (§8), following the existing
  notebook conventions there.

## 11. 3D science campaign (phase B3)

Written 2026-08-19 on the B1 tree (B2's gates landing in parallel). Deliverables:
`examples/particles_3d_run.py` (resumable `make_data`, three size profiles selected by
`TARANIS_P3D_PROFILE`) and `examples/test-particles-3D.ipynb`. **Designed and smoke-tested
here; the production run is a Kaggle job, not a laptop job** — §11.6. The 2D campaign (§5,
A3b) is the reference this is written against, and the protocol is deliberately identical so
the two are comparable line for line.

### 11.1 The headline questions, in priority order

**Q1 — Is the Chandran stochastic-heating exponential there in 3D, and at what `c₂`?**
This is a *literature check*, not an extension: Xia, Perez, Chandran & Quataert 2013
(ApJ 776, 90) measured exactly this in 3D RMHD. The 2D campaign got `c₂ = 0.40 ± 0.13` (lab
`v_⊥`, local-B `Q_⊥`, 2σ upper-limit rule) against their resolution-dependent 0.15–0.44 —
which was worth recording and not over-reading, because 2D has no parallel decorrelation.
3D removes that caveat, so the comparison becomes like for like.
**Null:** `c₂` within 1σ of zero over a ξ lever ≥ 3, with every fitted `Q_⊥` an order of
magnitude above the in-situ E = 0 control's rate. That is a result only if the lever and the
floor are both shown, which is why both are printed next to the fit.

**Q2 — What does 3D's freed parallel channel actually buy?** In 2D `p_z = v_z − (q/m)ψ` is an
*exact* invariant of the full-mask dynamics (docs/numerics.md), so parallel energization is
bounded by `(q/m)Δψ` and cannot be secular. In 3D neither step of that argument survives.
Three measurements: the `p_z` drift of the full-mask ensemble normalized by `(q/m)ψ_rms`, in
3D and in a **matched 2D twin run** (§11.3) where the same number is pure discretization
error; `⟨v_∥B²⟩(t)` on a common eddy-time axis, secular vs bounded; and `Q_∥/Q_⊥` against
`β_i`, directly comparable to Xia's Fig. 7.
**Null:** both curves bounded and `Q_∥/Q_⊥` the same in 2D and 3D. That is *also* a result —
it says the 2D invariant was never the binding constraint and that the parallel channel is
suppressed by the amplitude (`E_z/E_⊥ ~ ε` in code units), which is what the 2D `B₀` twins
already concluded. Do not write the campaign so that only a positive answer is publishable.

**Q3 — Can a disagreement be attributed?** Three things differ between the 2D result and a 3D
one: dimensionality, the dissipation model (`hyper=3` in 2D, Laplacian here) and the
amplitude. The matched 2D twin holds dimensionality as the only variable; the `hyper=3` twin
(one host, `full` profile only) holds the dissipation model as the only variable; the
sensitivity table prices the protocol. Anything left over is dimensionality.

**Q4 — Do the two ξ-scan designs agree in 3D as §10 asks?** Design (a) (Ω sweep at fixed
turbulence) and design (b) (forcing sweep at fixed particle numerics) probe different
numerics; agreement is evidence the exponential is physics. In 2D design (a)'s ξ lever was too
short (×2.0) to constrain `c₂` and its two hosts disagreed. 3D's Laplacian dissipation gives a
steeper, better-behaved `δu(ρ)` and design (b)'s ladder is ×6 in `u_rms`, so the lever should
be longer — measured, not assumed.

### 11.2 What 3D takes away, and what it gives

Derived while designing this; every item is a real constraint on the matrix.

- **`B₀` is pinned to 1** (§2, §10, B1). The per-ensemble amplitude knob that made
  `ε = 1/B₀` free in 2D is gone: `ε = u_rms/v_A = u_rms` and `δB/B₀ = b_rms` are properties of
  the RUN, set by the forcing power and the box. **An amplitude contrast is a different run,
  not a different ensemble**, so unlike 2D the fields are not shared between the twins.
- **`β_i` and ξ are locked.** `β_i = v_⊥i²/v_A² = v_⊥i²` in code units, so choosing `v_⊥`
  fixes both `β_i` and `ξ = δu(ρ)/v_⊥`: `β_i = (δu/ξ)²`. Xia et al. break this with RMHD's
  invariance under `(v_A, L_∥) → (ξ v_A, ξ L_∥)`, which lets one run serve six `β_i` cohorts —
  and that rescaling **is** taranis's per-ensemble `B₀`, which 3D forbids (the solver's Alfvén
  coefficient is 1 and `L_z` is not a per-ensemble quantity). Consequence: at
  `ε = 0.2` and `ρ ≈ 2–8 dx`, reaching `ξ ≲ 0.3` forces `β_i ≈ 0.01–0.9` — Xia's *mid-to-high*
  β corner (their B2/B3/C3, `c₂ = 0.37–0.42`), not their `β_i = 0.006` headline. Every `β_i`
  statement in the 3D notebook is entangled with ξ and must say so.
- **`p_z` is no longer an invariant** — Q2. Its 2D value is the pipeline's error floor and its
  3D value is physics, which is why the 2D twin is run rather than remembered.
- **Particles stream in z.** At `v_∥ ~ v_th` and a window of 6 outer times, the parallel
  excursion is `v_th·6L_⊥/u_rms ≈ 6L_⊥·(v_th/u_rms)`, comparable to `L_z = 6L_⊥` — so a
  particle does traverse the box, and the field it samples decorrelates by streaming as well
  as by the Alfvénic evolution at `L_z/v_A` (one outer time, by critical balance). Both are
  absent in 2D. The notebook prints the excursion so a "null" in Q2 can be checked against it
  rather than assumed away.
- **`parspec` becomes a resolution check** with no 2D analogue: whether `n_z` resolves the
  critically-balanced parallel structure at the perpendicular scales the particles sample.

### 11.3 The run matrix

**Base turbulence.** `dims=3`, single process (B1's restriction), **finite-difference z**, box
`L_⊥ = 2π`, `L_z = 6L_⊥` (Xia's aspect), elsasser O-U forcing in the perpendicular shell
`1 ≤ |k|/dk < 3` with the `k_z = ±2π/L_z` envelope, `forcing_tau = 1`, adaptive dt
(`cfl_safety = 0.5`, `cfl_every = 1`), `lsrk33`, fp64.

*Why finite-difference z and not `z_spectral`.* Measured, not assumed: at 256²×64 on this
laptop `z_spectral` costs **1.73×** the FD-z step (1310 vs 759 ms; 1.72× at 128²×64), because
the rfftn over (z,x,y) plus the 2×2 putzer propagator is more work than one rfft2 per plane
plus a z-stencil. Against that, B1 measured full-mask `E_z` against the solver's own
`dψ/dt − ∂_zφ` at 7.9e-16 relative in **both** modes, so exactness buys nothing here, and the
`∂_z⁴` filter FD-z adds lives entirely inside `ez_resistive`, which the production ensembles
do not see. The FD-z parallel CFL (`dt ≤ cfl·dz`) is not binding at this aspect: `1/dz = 1.7`
against a perpendicular `max|∇|/dx ≈ 57` at 256². A `z_spectral` twin at reduced size stays
available as an attribution control if the z discretization is ever suspected.
Scheme: `lsrk33` is integrating-factor, the wave path — **never an IMEX scheme** on a
wave-dominated `L` (CLAUDE.md); with FD-z the Alfvén term is an RHS term anyway.

*Why Laplacian dissipation, against §10's `hyper = 3` decision for 2D.* Xia's `c₂` is
*Reynolds-number dependent* (0.44 at Re 2400, 0.41 at 6000, 0.29 at 15000, 0.20–0.25 at
38000). A `hyper = 3` run has no Reynolds number and cannot be placed on that trend, so the 3D
literature check runs `hyper = 1` with `ν = η = 0.2·L_⊥/Re`, `Re = 6000` at 256²
(`Re ∝ N_⊥^{4/3}` holds `k_d/k_max` fixed as the profile shrinks). The `hyper = 3` twin at the
reference host is the bridge back to the 2D campaign. Side effect worth noting but *not* a
headline: at `hyper = 1` the resistive `E_z` piece is `η j_z` rather than a numerical
regularization, so the full-mask-vs-ideal difference is a resistive acceleration — still a
fluid closure, not the kinetic `E_∥` a collisionless plasma has, so it stays out of the
physics claims (§1's caveat is unchanged).

*Amplitude.* `EPS_LADDER` is a ladder of **target `u_rms`**; the forcing power is
`P = FORCE_C·u_target³/L_⊥` (constant flux) and the achieved `u_rms`, `b_rms` and
`χ = k_⊥u_rms/(k_z v_A)` are **measured in the notebook**, never assumed. `L_z` is held fixed
across the ladder, so `χ` moves with `ε` (0.3 at `ε = 0.05` to 1.8 at 0.3) and only the
`ε = 0.2` host is critically balanced the way Xia's runs are; the alternative — rescaling
`L_z ∝ 1/ε` per host — keeps `χ ≈ 1` everywhere but changes the box, and therefore `δu` at a
fixed physical `ρ`, between hosts. Fixed box was chosen so the design-(b) `δu(ρ)` comparison
is clean; the trade is recorded here so it is not rediscovered.

**Ensembles** (13 per design-(a) run; `n` is per ensemble):

| group | what | mask / flags |
|---|---|---|
| (a) Ω sweep, 5 | `ρ/dx ∈ {2,3,4,6,8}` at fixed `v_th` (fixed `β_i`) | ideal-Ohm, `epar_project=True` |
| (c) rings, 4 | `ρ/dx = 3` fixed at insertion, `v_⊥ ∈ {0.5,1,2,4}×v_⊥,rms` | ideal-Ohm, `epar_project=True` |
| unprojected twin | the `ρ/dx = 4` ensemble without the projection | ideal-Ohm, no projection |
| full-`∂ψ/∂t` | the exactness / `p_z` ensemble | all five pieces |
| E = 0 control | the numerical-heating floor | `bperp` only |
| δb = 0 control | pure gyration about `B₀ẑ` | nothing |

Design (b) is the `ρ/dx = 4` ideal ensemble alone, repeated at every forcing power in its own
run. There is **no `B₀` contrast cohort** — in 3D that is the forcing ladder itself.

**The matched 2D twin.** One 2D run per `TWIN_HOSTS` entry, carrying the same 13 ensembles.
The 2D solver has no Alfvén term, so `B₀` is free there and the mapping is exact — divide
every 3D velocity by `ε` and keep lengths:

    B₀ = 1/ε,  q/m = Ω_3D,  v_th,2D = v_th,3D/ε,  u_rms,2D = 1,  t_2D = ε·t_3D,  ν_2D = ν_3D/ε

which leaves `Ω`, `ρ`, `ξ` and `β_i` identical between the twins. What is **not** matched is
the turbulence: 2D RMHD inverse-cascades `⟨ψ²⟩` to the box scale and 3D does not, so
`b_rms/u_rms` differs and every twin comparison is made at matched ξ, not at matched forcing.
Cost is negligible (2D 256² is ~1/40 of a 3D 256²×64 step), so the twin is not an optional
extra.

**Profiles** (`examples/particles_3d_run.py::PROFILES`, selected by `TARANIS_P3D_PROFILE`):

| | grid | n/ens | hosts | spin-up + window | 2D twins | hyper=3 twin |
|---|---|---|---|---|---|---|
| `smoke` | 32²×16 | 256 | `u_rms` 0.2, 0.3 | 1.5 + 1.5 turnovers | 1 | no |
| `mvp` | 128²×32 | 4096 | 0.1, 0.2, 0.3 | 4 + 5 | 1 | no |
| `full` | 256²×64 | 32768 | 0.05, 0.1, 0.2, 0.3 | 4 + 6 | 2 | yes |

`n_z/n_x = 0.25` matches Xia's 1024²×256 aspect and the anisotropy estimate
`n_z/n_x ~ (k_⊥max/k_0)^{-1/3}`; Xia used 1.0 at 256³, so **parallel resolution is the
standing risk** and `parspec` is the check that fires.

### 11.4 Measurement protocol

Identical to the 2D campaign's, which follows Xia et al. §4.1, so the two are comparable:

- heating in the **local-B frame** (`vperpB2`, `vparB2`), with the ẑ-referred pair reported
  beside it;
- window from `t₀ = 10/Ω` (skipping the E×B pickup) to the first step where the smoothed
  `⟨v_⊥B²⟩` exceeds `1.2×` its `t₀` value, or the run end, never shorter than 400 solver steps
  (`ext` flags where the floor bound it);
- `ρ` and `ξ` from `v_⊥` **at `t₀`**, `δu(ρ)` from the run's own kinetic spectrum averaged over
  the snapshots inside the window;
- **the 2σ upper-limit rule**: a rate not 2σ above zero is a limit, starred, drawn as an arrow,
  held out of every fit. Xia et al. quote **no error bars and no limit rule** — every cohort
  enters their fit — so the sensitivity table carries an explicit **Xia-comparable row** that
  admits every point. In 2D those two differed by 0.2 in `c₂`, which is larger than the gap
  between Xia's coarsest and finest runs; quoting only one of them would be a false comparison.
- the near-zero-point sensitivity table the 2D analysis prints, extended with one new row:
  **`δu` convention**. Xia's `δu² = ∫_{k₋}^{k₊}E_u dk` over one e-fold at `k_ρ` versus
  taranis's `δu = √(2kE_kin)` differ by a few percent on a k^{-3/2} spectrum but by 10–35% on a
  steep one; the notebook computes the band integral (host-side, numpy) and reports the
  measured ratio and a `c₂` fitted with it.
- **the floor is quoted as a rate.** New in 3D and better than the 2D practice: the E = 0 and
  δb = 0 controls are run through the *same* `heating_points` window machinery, so the table
  carries their `Q_⊥` next to the measurements and prints the ratio of the smallest fitted
  `Q_⊥` to the largest control `|Q_⊥|`. Under ~10× means the pipeline was measured, not the
  plasma.

**Q_∥, and what the 2D bound does not constrain.** `Q_∥` is the same OLS slope on `vparB2`
(halved, like every velocity-square column) over the *same* window as `Q_⊥`, quoted per
ensemble alongside `w_ez_ideal/w_tot` from the exact work accumulators. Three things the 2D
`p_z` bound says nothing about, and which are therefore the 3D-only content: (i) whether
`⟨v_∥B²⟩` grows *secularly* rather than oscillating — 2D forbids the former, 3D does not;
(ii) whether `Q_∥/Q_⊥` rises with `β_i` as Xia find (their Fig. 7: `Q_∥ ≪ Q_⊥` at `β_i ≪ 1`,
`Q_∥ > Q_⊥` possible at `β ~ 1`, `ξ ≲ 0.1`), which needs the parallel resonance broadening
that only z-dependence provides; (iii) the size of the `p_z` drift itself, which in 2D is
bounded by construction and in 3D is not. Conversely the bound says nothing about the
*magnitude* of `E_z`: in code units `E_z/E_⊥ ~ ε` in both dimensions, so a small `Q_∥` in 3D
is evidence about the amplitude, not about the invariant. Both readings are written into the
notebook's "reading the parallel channel" section so the answer is attributable either way.

### 11.5 How this differs from Xia et al. 2013

Checked against the paper, not against memory. Differences that could move `c₂`, and why each
was accepted:

| | Xia et al. | this campaign (`full`) |
|---|---|---|
| grid | 128³ … 1024²×256 | 256²×64 (`n_z/n_x` = their D-run aspect, not their 256³ aspect) |
| box | `L_∥/L_⊥ = 6` | 6 |
| dissipation | Laplacian, `ν = η`, Re 2400–38000 | Laplacian, `ν = η`, Re ≈ 6000 at `u_rms = 0.2` |
| forcing | body force on `k_⊥, k_∥ ∈ [1,2]` box modes, Gaussian, refreshed 5×/eddy with cubic interpolation, balanced | O-U elsasser, `k_⊥ ∈ [1,2]` shell, `k_z = ±1`, `τ = 1`, balanced |
| amplitude | `u_rms ≈ v_A/5`, `χ ≈ 1` | ladder 0.05–0.3 at fixed `L_z`, so `χ = 0.3–1.8`; the 0.2 host is theirs |
| `δB_rms/B₀` | ≤ 0.47 (per-cohort, via the rescaling) | `= b_rms`, one value per run, 0.07–0.42 |
| particles | ~1e5 per cohort; 5% rate error at 5.12e4 | 32768 per ensemble → ~2× that finite-N error |
| interpolation | TSC in 4D (three space + **time**) | trilinear in space, fields **frozen** over a solver step |
| substepping | particle dt ≈ RMHD dt / 4 | `substeps = 2`, and the ensemble table prints steps/gyration |
| `E_∥` | their eq. 21 | `epar_project`, algebraically the same operation |
| frame, window | local B; `t₀ = 10/Ω` to 1.2× | same (+ a 400-step floor) |
| limits | none; every cohort is fitted | 2σ rule **and** a Xia-comparable no-limit row |
| `δu` | e-fold band integral at `k_ρ` | `√(2kE_kin)`; band version also computed, ratio reported |
| `β_i` | 0.006–1, **decoupled** from ξ | `= v_⊥²`, **locked** to ξ; ~0.014–0.93 |
| `c₂` | 0.44 / 0.41 / 0.29 / 0.20–0.25 by Re; `c₁` 0.6–1.1 (low β) to 3.6 (β = 1) | the comparison is to the row at *this* run's Re, not to their headline 0.21 |

The three that most plausibly matter: **finite-N** (they resolve 5% at 5e4, we will not do
better than ~10%, which is why the upper-limit rule earns its keep); **frozen fields plus
linear interpolation** where they use TSC in space *and* time (the `p_z` drift of the 2D twin
is the in-situ measurement of what that costs); and **the locked `β_i`**, which means a `c₂`
measured here at `ξ ≈ 0.1` is a `β_i ≈ 1` measurement in their terms and belongs against their
C4/B4 rows, where `c₁ ≈ 3.6`.

### 11.6 Cost, measured

**Measurements** (Apple M1, macOS 14 / Darwin 23.6.0, jax 0.10.0, CPU backend, fp64,
`jax.jit(block_of_steps)` with fixed dt, `lsrk33`, elsasser forcing; median of 9 repetitions;
machine not perfectly quiet, run-to-run spread ~±10%, which is why 4-ensemble and 10-ensemble
columns sometimes cross):

| grid (FD-z) | particles off | 4 ens × 4096 | 10 ens × 4096 |
|---|---|---|---|
| 128²×32 | 96.4 ms/step | 105.0 | 116.8 |
| 128²×64 | 199.7 | 255.3 | 260.0 |
| 192²×48 | 375.9 | 454.8 | 378.3 |
| **256²×64** | **759.3** | **958.6** | **888.9** |

(the 256²×64 row is superseded by the alternating re-measurement below, 707.5 ms particles
off; the 4- and 10-ensemble columns crossing at 192² and 256² is the ±10% run-to-run spread,
not a real effect.)

`z_spectral`, particles off: 344.1 ms at 128²×64 and 1309.7 at 256²×64 — **1.72–1.73× FD-z**,
which is what settles §11.3's z-mode choice.

Particle-count scan at 128²×64, 10 ensembles: 1024 → 205 ms, 4096 → 260, 16384 → 257 —
i.e. flat up to 1.6e5 particles, which is what docs/performance.md predicts (the fixed
`particle_fields` transforms dominate, the O(N) gather does not). **That flatness does not
extend to the production loading**, and assuming it would have been wrong. Alternating
off/on/on at the production size, 7 repetitions, so thermal drift cancels:

| 256²×64, FD-z | ms/step | vs off | IQR |
|---|---|---|---|
| particles off | 707.5 | — | 40 ms |
| 13 ensembles × 8192 (1.1e5 particles) | 906.6 | **+28%** | 103 |
| 13 ensembles × 32768 (4.3e5 particles) | 1220.8 | **+73%** | 67 |

So the gather turns back on somewhere above 1e5 particles: 0.5 ns per particle per step
between the 128²×32 pair (41k particles, +21%) and ~1.0 ns between the two production-size
points, i.e. the rate itself rises with occupancy and is not safe to extrapolate far. The two
profiles' particle counts were each chosen from a measured point, not from the model: 4096 per
ensemble at `mvp` (+21% measured) and 32768 at `full` (+73% measured). The trade is worth taking: 4× the particles is 2× better finite-N statistics
(Xia et al. resolve 5% at 5.12e4 per cohort) for **+25% of the campaign's wall clock**, and
finite-N is what limits the *upper limits*, which are the discriminating points. `full`
therefore runs 32768 per ensemble, and the cost table below uses the measured 1220.8 ms for
the 13-ensemble runs, 870 ms for the 1-ensemble design-(b) runs and 707.5 ms particles-off.

Per grid point the solver is **0.181–0.190 µs/point/step** (fp64, particles off), flat over a
factor 8 in problem size — so the extrapolation below is interpolation, not a guess.

**Steps.** `dt = cfl·dx/max(|∂_xφ|+|∂_xψ|)`, so steps per outer turnover `= κ·N_⊥/cfl` with
`κ = max|∇|/u_rms`, independent of the amplitude (this is why every host on the ladder costs
the same). Measured on the smoke run: `κ = 5.5` at 32² (median `dt = 7.16e-2`,
`u_rms = 0.251`); the 2D campaign implies `κ ≈ 9` at 256², where the gradients are better
resolved. **Planning `κ = 7 ± 2`** → 3584 steps/turnover at 256², 1792 at 128².

**Two options.**

*Minimum viable (`mvp`, 128²×32).* 3 hosts × 4 turnovers of particle-free spin-up + 1
design-(a) × 5 + 3 design-(b) × 5 = 32 turnover-units, 57k steps →
**≈ 1.7 h on this laptop**, ~1.1 GB. What it buys: the whole pipeline, the 2D-vs-3D parallel
comparison (Q2, which needs contrast and not resolution), design (b)'s ×6 forcing lever, and
a `c₂` with a ξ lever of maybe ×2 and a nearly absent inertial range. What it does **not**
buy: a `c₂` placeable on Xia's Re trend, or a `δu(ρ)` worth the name. Run this first; it is
the thing that finds the protocol bugs.

*Full (`full`, 256²×64, 32768 per ensemble).* Priced per phase at the measured rates:

| phase | turnover-units | ms/step | wall |
|---|---|---|---|
| 4 base spin-ups, particles off | 16 | 707.5 | 11.3 h |
| 2 design-(a) runs, 13 ensembles | 12 | 1220.8 | 14.6 h |
| 4 design-(b) runs, 1 ensemble | 24 | ~870 | 20.8 h |
| hyper=3 twin (spin-up + particles) | 4 + 6 | 707.5 / 1220.8 | 10.1 h |
| 2 matched 2D twins at 256² | — | ~25 | ~1 h |
| **total** | **62** (≈ 222k steps) | | **≈ 58 h** |

Disk ≈ **12.4 GB**: 161 field snapshots at 67.6 MB, 1.4 GB of particle items, ~0.2 GB of
moment sidecars.

**Kaggle target.** Assume a **P100** (what `examples/kaggle_forced_turbulence_256cubed.ipynb`
targets), fp64. The extrapolation is a chain and should be treated as **±2×**: P100 fp64 peak
is 4.7 TFLOP/s at 732 GB/s HBM2; taranis's step is FFT/bandwidth-bound, and the closest
measured anchor in docs/performance.md is 1×A5000 at fp32, 294 ms/step for 512²×128
(8.8 ns/point). Halving throughput for fp64's doubled word gives ~17.6 ns/point on a P100
against **185 ns/point measured on this M1 CPU — a factor ≈ 10**. So:

| | steps | M1 CPU fp64 | P100 fp64 (assumed 10×, ±2×) | disk |
|---|---|---|---|---|
| `mvp` 128²×32 | 57k | 1.7 h | ~10 min | 1.1 GB |
| `full` 256²×64 | 222k | 58 h | **≈ 6 h (3–12 h)** | 12.4 GB |

That fits a Kaggle session only if the fast end holds, so **the resumability of `make_data` is
load-bearing, not a convenience** — the campaign is expected to span 2–3 sessions against the
~30 GPU-h/week quota. Memory is not a constraint: the `full` state is 67.6 MB, the real-space
working set ~0.5 GB, peak well under 2 GB on a 16 GB card — there is headroom for 384²×96 if
the GPU turns out to be at the fast end. Disk *is* a constraint on Kaggle: run under
`/kaggle/tmp` and archive only the sidecars, `params.json` and a subset of snapshots
(~0.5 GB), never the whole 12.4 GB tree — `/kaggle/working` caps at 20 GiB **and ~500 files**,
and orbax writes many small files per snapshot.

**Is fp32 admissible?** It roughly halves both time and disk and would put `full` inside one
Kaggle session. The particle state, the pusher and the work accumulators are fp64 whatever
`TARANIS_PRECISION` says (§4), so gate 8's closure is precision-independent — measured on the
smoke run at `6.2e-16` (fp32) and `6.8e-16` (fp64).

That has a consequence which had to be measured rather than assumed: **the E = 0 control is
blind to field precision.** Its mask makes `E` exactly zero, so the push is a pure fp64
rotation and the floor does not move — measured at `7.105e-15` max per-particle `|v|²` drift
in *both* precisions, agreeing to four digits, with `w` bitwise zero in both. So gate 5's
floor bounds the *pusher*, not the interpolation noise fp32 fields would introduce, and it
cannot be the fp32 criterion for a field-precision change. (§9's reading of gate 5 — a WebGPU
port whose particle STATE is also fp32 — is unaffected; there the control does move.)

The criterion is therefore statistical, and there is only one honest form of it: fp32 fields
make the turbulence a different realization within a few eddy times, so **rerun one design-(a)
host at fp32 and require its per-ensemble `Q_⊥`, and the `c₂` it yields, to agree with the
fp64 twin inside the fit error.** Smoke-scale precedent (32²×16, one host, the reference ideal
ensemble): `Q_⊥ = 2.20e-3 ± 2.9e-4` at fp64 against `3.20e-3 ± 1.3e-3` at fp32 — consistent,
but only because the fp32 error bar is 4× wider, which is exactly the kind of "agreement" the
production test must not accept. Until that test is green at the production window length,
**production is fp64**. The prize is real (≈6 h → ≈3 h on a P100, 12.4 → 6.2 GB, two Kaggle
sessions → one), so run the test; do not assume the answer.

### 11.7 Smoke run (2026-08-19, verified)

`TARANIS_P3D_PROFILE=smoke`, 32²×16, `L_z = 6L_⊥`, `Re = 375`, 13 ensembles × 256 particles,
two 3D hosts + the 2D twin + the high-cadence tail: **46 s of wall clock, 10 MiB**, and the
notebook executes end to end with **zero errors**. What it verified (the numbers are a
pipeline check, not physics — this profile has no inertial range):

- **work closure** `6.8e-16`, i.e. `2.9e-15` of the initial kinetic energy, over the 11
  ensembles that do work; a piece an ensemble's mask omits is bitwise zero;
- **E = 0 floor** max per-particle `|v|²` drift `7.1e-15` (δb = 0 control `6.3e-15`) against
  ideal-ensemble `|v|²` changes of order 3.7;
- **nonzero `Q_⊥`**: best point `7.8e-3 ± 2e-3`, with 4 measurements and 7 upper limits out of
  11 points, so the limit rule fires and `chandran_fit` returns finite numbers on both the
  2σ and the Xia-comparable branch (the fitted `c₂` is meaningless on 4 points and the
  notebook says so);
- **the floor-as-a-rate check works and correctly complains**: largest control `|Q_⊥|`
  `1.8e-4` against a smallest fitted `Q_⊥` of `1.6e-3` — a factor 8.7, i.e. *below* the ~10×
  the notebook demands. At smoke size that is the expected answer and it is the sign the check
  has teeth;
- **`mu_of` vs the sidecar `mu` column** agree to `7.5e-4`; the high-cadence tail's pair
  estimator of `D_μ` is 6–10× the sidecar's variance-slope value on the maxwellian ensembles,
  reproducing the 2D campaign's finding that the pair estimator is an upper bound;
- **snapshots and restart**: a second `make_data` call recognizes every run as already at its
  target and does nothing; the moments sidecar is appended without duplicate times
  (`read_moments`' repeated-`(t, ensemble)` check passes);
- **the amplitude is measured, not assumed**: at `FORCE_C = 2.0` the runs reached
  `u_rms/u_target = 0.83, 0.84`, so the constant was retuned to `3.4` and now reaches
  `0.94, 0.96`. The calibration loop the notebook prints works; the constant must still be
  re-checked at production Reynolds number;
- **`p_z`, already visible at smoke size**: the full-mask ensemble's drift is `0.043` of its
  `(q/m)ψ_rms` scale in the 2D twin (pure discretization error) and `1.65` in 3D — a factor
  **38**. This is the Q2 signature, and finding it at 32²×16 says the production run is
  measuring something rather than hunting for it. Note the flip side, which the smoke also
  shows: `⟨v_∥B²⟩` grew by 0.49 in 3D and 0.60 in the 2D twin over their windows, so the
  *magnitude* of the parallel channel is not obviously different — consistent with the
  `E_z/E_⊥ ~ ε` expectation of §11.4, and exactly the "null" the production run must be able
  to state cleanly;
- **fp32 fields** (a second smoke pass at `TARANIS_PRECISION=32`, `data/test-particles-3D-fp32`):
  closure `6.2e-16`, E = 0 floor `7.105e-15` — identical to fp64 to four digits, which is the
  measurement behind §11.6's fp32 criterion.

### 11.8 Open risks

- **Parallel resolution.** `n_z/n_x = 0.25` follows Xia's largest runs but they used 1.0 at
  256³. If `parspec` shows power piling at the `k_z` cutoff, `n_z = 128` doubles the cost of
  the whole campaign; there is no cheaper fix.
- **Finite N.** 32768 per ensemble gives ~10% rate error against Xia's 5% at 5.12e4. This
  bites the *limits* hardest, which are the points that discriminate exponential from power
  law (§5's open item (ii)).
- **The locked `β_i`.** §11.2. Nothing in the current solver can unlock it; the honest fix
  would be per-ensemble `(B₀, L_z)` pairs, i.e. running the same fields against several box
  interpretations — cheap in principle (the fields do not change), but it is a `Parameters`
  and `fields.py` change, not a campaign change, and it is not proposed here.
- **Frozen fields.** Xia interpolate in time as well as space. The 2D twin's `p_z` residual
  measures the cost in situ, but if it is not ≪ the 3D drift the Q2 comparison is bounded
  rather than measured.
- **The GPU factor.** ±2× on a chain of inferences from a different card at a different
  precision. The first production session should time 20 steps and re-plan before committing.

## 12. Time interpolation of the particle fields (PLAN, 2026-08-19 — not implemented)

Decided with Alfred after Phase B: **quadratic (TSC) interpolation in time**, matching Xia et
al.'s 4D TSC. This section is the design; nothing here is built.

### 12.1 What is wrong now

`_advance_particles` assembles the fields from the PRE-step state and holds them fixed across
the step, so the impulse `∫E(x(t),t)dt` is evaluated as `(dt/2)[E(x_n,t_n) + E(x_{n+1},t_n)]`:
trapezoid in SPACE along the trajectory, left-endpoint rectangle in TIME. That is the O(dt)
term gate 4's live variant measures (order 0.838 fixed dt, 0.978 adaptive — §6), and it is
first order where everything else in the push is second.

The parameter is the field's Eulerian decorrelation rate at the gyroscale against dt. With
`dt = cfl·dx/(κ·u_rms)` and sweeping `ω ~ u_rms/ρ`, it collapses to

    ω·dt = cfl / (κ · ρ/dx)     = 3.6% / 2.4% / 1.8% / 1.2% / 0.9%  at ρ/dx = 2/3/4/6/8
                                  (cfl = 0.5, κ = 7)

independent of amplitude, so it is the same on every rung of the ε ladder. **It is worst
exactly where the exponential is measured**: design (a) sweeps q/m at fixed v_th, so small ρ
means small δu(ρ) means small ξ, and `d ln Q_⊥/d ln ξ = 3 + c₂/ξ ≈ 7` at ξ = 0.1.

Three things that do NOT bound it, each a natural assumption that fails:

- **The E = 0 and δb = 0 controls are blind to it** — their masks make E exactly zero, so
  there is no E-time-sampling error to commit. The §11.4 floor does not cover this.
- **`substeps` does not reduce it.** Substeps refine `Ω·dt`, not `ω·dt`; the fields stay frozen
  across the whole solver step whatever `substeps` says. Today `substeps` is decorative for
  this error.
- **The `p_z` drift is a weaker proxy than §11.8 assumes.** It is a systematic first-order
  quantity; spurious heating is diffusive and second order in the same error. Small `p_z`
  drift is necessary, not sufficient.

### 12.2 Which quadratic, and why it is a smoothness argument

Two different objects hide behind "quadratic in time", and only one gives what is wanted:

- **Lagrange quadratic** through `(t_{n−1}, t_n, t_{n+1})` — interpolating, third-order
  accurate, but adjacent intervals use different quadratics, so it is still only C⁰ at the
  joins. The kink survives, and the third order is wasted against a second-order pusher.
- **Quadratic B-spline — which is what TSC is** — C¹, genuinely kink-free, but APPROXIMATING:
  the curve does not pass through the field values. It slightly low-passes E in time.

Take the B-spline. The overall scheme stays second order either way, so this is a
**smoothness** decision, not an accuracy one: the point is to remove the step-boundary
discontinuity whose error signal is a sawtooth with fundamental `1/dt`. The price is that
TSC's temporal low-pass is itself a small systematic in the direction of LESS stochasticity,
so it must be measured, not assumed (§12.5).

**Non-uniform knots.** dt is adaptive (`cfl_every = 1`), so the textbook three-constant TSC
weights do not apply. Non-uniform quadratic B-splines are still C¹ — the weights just become
knot-dependent (Cox–de Boor on the actual `h₁ = t_n − t_{n−1}`, `h₂ = t_{n+1} − t_n`). This
costs nothing: the weights are THREE SCALARS per half-kick, ~10 scalar flops, computed once
per half-kick and not per particle or per grid point. Carry the two past dt values alongside
the field history; they are traced scalars under `scan`, needing no `lax.cond` and no shape
dependence. At fixed dt the non-uniform weights must collapse to the textbook constants —
a free unit test.

### 12.3 Where the cost actually is: blend the grids, do not gather three times

    blend three grid levels, then ONE gather   O(N_grid × 3)                  streaming
    gather three times, blend the samples      O(N_particle × 8 corners × 3)  irregular

Element counts are within a small factor at production loading (4.3e5 particles against
4.19M grid points), but the gather is already the largest particle cost — B2's numbers imply
≈400 ms of a 707 ms step at 13 ensembles on the M1, and **tripling that is fatal**, while the
blend is one fused pass (3 reads, 1 write) over ~7 arrays ≈ 1.3 ms per half-kick on a P100.
Blend the grids.

That moves the cost driver onto `substeps`: one blend per half-kick means `2 × substeps`
blends per step, ≈ +14% at `substeps = 4`. Measure it rather than trusting the estimate.

### 12.4 Carry, memory, and the restart ladder

Quadratic needs `F(t_{n−1})`, `F(t_n)`, `F(t_{n+1})`. `_advance_particles` receives only
`prev_state` and `new_state`, so one past level must be carried. With a carried history the
extra field assembly DISAPPEARS — at step n the two older levels are already in hand and only
the new one is assembled, so it stays **one `particle_fields` per step, exactly as today**,
plus a ring-buffer shift. Quadratic in time costs memory, not transforms:

| carry | memory at 256²×64 fp64 | compute |
|---|---|---|
| two past `PFields` | ~470 MB (~600 with the optional pieces) | unchanged |
| two past `(φ,ψ)` states, reassemble each step | ~68 MB | 3 assemblies/step (+34%) |

Both fit a 16 GB P100 against a ~0.5 GB working set; prefer the memory.

**The history is NOT checkpointed.** Saving it would add ~68 MB (states) or ~470 MB (PFields)
per snapshot against a 67.6 MB field snapshot — +11 GB or +48 GB over the campaign's 161
snapshots, against a 12.4 GB budget and Kaggle's 20 GiB / ~500 file cap. Instead **degrade by
available history**: 0 levels → frozen, 1 → linear, 2+ → quadratic. Two reduced-order steps
per restart out of ~1e5, and synchronization is never broken (the alternative, "start pushing
on the second step", leaves the particles one dt behind the fields).

Resetting the history at block boundaries — so that an uninterrupted run degrades at the same
steps a restarted one does, keeping restart bitwise — was **considered and rejected**. It
injects a PERIODIC first-order perturbation at a cadence set by `nblock`, a pure performance
knob; on a measurement whose whole content is spurious-versus-real stochasticity, a periodic
scheme change is the artifact one would least want, and it is worse than a one-off warm-up.
The property it protected is not there anyway: production restarts are already non-bitwise
(`forcing_norm_per_step=True` recomputes the forcing scale at dt = 0 on driver entry;
fixed 2026-08-22, REFACTOR_PLAN Phase 0c), and
gate 6c only achieves bitwise by turning that off. **Gate 6c is therefore re-specified**: a
restart is deterministic, and reproduces the uninterrupted run bitwise AFTER the two-step
warm-up, with the warm-up difference measured and reported.

### 12.5 How it gets measured

As a **paired ensemble**, like the resistive split: frozen and TSC-interpolated particles in
the same run, on identical fields, differing only in this. The difference in `Q_⊥` is then
measured with zero realization noise rather than argued about — which is the only honest way
to settle both the sign of the frozen-field bias and the size of TSC's temporal low-pass.
Run the pair at `mvp` scale before `full` commits.

The cheaper alternative — rerun one host at `cfl_safety/2` — works but is noisier: halving dt
gives a different turbulence realization, so the comparison has to fight through window
statistics instead of differencing per particle.

Once fields vary within a step, `substeps` finally does something, so `substeps = 2` (§11.3)
should be revisited against Xia's ≈4 at the same time.
