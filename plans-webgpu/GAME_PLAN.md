# GAME_PLAN — sailing the turbulence: a 2D MHD boat on the rmhd2d solver

Rev 2, 2026-08-18. Supersedes the rev-1 draft (delivered in chat). Changes from rev 1,
all from the design discussion: separate page `game.html` with a solver-extraction
Phase 0; passive tracers replaced by real charged test particles (Boris pusher, imported
from the TESTPART project, which is sequenced FIRST — see `taranis/plans/TESTPART_PLAN.md`);
the hull-frame keel replaced by the derived tether force law with the Neubauer finite-M_A
correction; waypoint controls; dipole gradient force added. Rev-1 content that stands
unchanged: no full-field readback (entity pass samples the RHS gradient buffers), sim-time
integration (never wall clock), code units (box Lx×Ly, u,b in v_A units, B_0 = 1),
256² default preset, and the two rev-1 sign fixes are folded into the force laws below.

## 0. Physics picture (one paragraph)

The boat is a non-back-reacting test body in the perpendicular plane of 2D forced RMHD:
u = ẑ×∇φ is the wind, b_⊥ = ẑ×∇ψ the in-plane field. Its hardware is a conducting hull
plus one z-directed conducting tether of length ℓ. The hull moving relative to the plasma
launches Alfvén wings on the guide field → isotropic linear drag γ₀w (Drell–Foley–Ruderman
1965; Neubauer 1980). The tether, shorted, converts cross-field-line slip into a motional
EMF and a J×B drag — the magnetic keel: sliding along in-plane field lines is cheap,
crossing them costs, O-points are slippery, current sheets grip. The same tether driven
with stored charge is the thruster. A magnetic dipole moment gives the heading lock
(τ = m×b) and a gradient force m∇|b| that reels a locked boat into strong-field regions.
Charge is harvested by capturing charged test particles, which gyrate, drift, and heat in
the same fields (TESTPART kernel). Wings, return currents, and the tether's z-extent live
outside the simulated plane; the solver never hears about any of it.

## 1. Sequencing

- **TESTPART (separate project, parallel track)** — Boris test-particle module for
  taranis core (fp64, science: stochastic heating) + validated WGSL port. See
  `taranis/plans/TESTPART_PLAN.md`. It gates ONLY the final form of §5: the boat track
  does NOT wait for it. Until the kernel lands, the game runs an E×B tracer stub in the
  SAME particle buffer layout (§2) — tracers are the guiding-center limit, so
  attraction/capture/economy tuning survives the swap; the swap itself is one kernel
  function. Reverse dependency: the game's entity pass (field binding, bilinear
  sampling, readback pool) is the harness TESTPART's WGSL port lands in.
- **Phase 0: solver extraction.** Move the inlined solver WGSL + Solver class out of
  `rmhd2d.html` into a shared module (`solver2d.js`) imported by both `rmhd2d.html` and
  `game.html`. Pure refactor, zero behavior change. Gates: refvector self-test green
  through the same GPU pipelines; RNG reference byte-identical; rmhd2d.html
  feature-identical by eye on one desktop + one phone. Game work (not TESTPART) starts
  only after Phase 0 ships. Timing (Alfred, 2026-08-18): START NOW; game.html ships
  unlinked (no references from index.html/docs) and pruned from the Pages deploy
  (pages.yml, next to the existing devtools/ prune) until ready — local testing only.
  The extraction itself does touch deployed rmhd2d.html and deploy = push, so the first
  push after Phase 0 carries it live: run the full gate set plus an on-device browser
  self-test before whatever push happens next, and keep the extraction commit clean so
  it reverts alone.
- **Phase 1: game.html** — boat + controls + particle economy (stub, then Boris) + HUD
  (this document).
- Later (not specced here): regenerative-braking mode, equilibrium/island presets
  (those inherit the tearing-preset resolution rules), scoring/goals.

## 2. Architecture (unchanged from rev 1 in structure)

The solver's RHS already holds real-space (∂xφ, ∂yφ, ∂xψ, ∂yψ) every stage. One entity
compute pass per frame, after the last solver step, binds those buffers read-only:

- **Particles**: GPU-resident buffer (N_p ~ 2–8k × {pos, v_⊥ (2), v_z, alive}), advanced
  by the TESTPART Boris kernel; rendered as point sprites on the overlay canvas; never
  read back individually.
- **Boat uniform in**: position/velocity/heading/ω, input state, Δt_sim, radii, params.
- **Tiny buffer out** (~64 B): bilinear u(x_b), b_⊥(x_b), optional j(x_b); atomic capture
  count (charge-weighted, see §5). Read back async with a small buffer pool (RECASYNC
  idiom); boat ODE integrates on CPU from the latest arrived sample; 1–2 frames of field
  staleness accepted (field correlation time ≫ frame time). Inputs act on CPU state
  immediately, so control feel never sees readback latency.

Bilinear sampling wraps periodically (index mod nx/ny; row-major ix·ny + iy per
`vecGatherWGSL`). The entity pass consumes ZERO solver RNG (particle seeding uses its own
stream); with the game page closed / feature off, the solver step loop is byte-identical.

**Time base — pinned sim rate.** Two separate mechanisms, do not conflate them:

1. *Dynamics are in sim time* (correctness): boat and particles integrate the Δt_sim the
   solver actually advanced (boat in ≤4 substeps; particles at solver dt inside the
   pass). Never wall clock anywhere in dynamics or fits. This alone makes trajectories
   device-independent but leaves the PACE device-dependent (steps/s spans ~10× across
   devices; params tuned in sim units would feel 5× different on a phone).
2. *The wall↔sim exchange rate is pinned* (feel): a fixed target T_sim (sim units per
   wall second, a game constant, `?simrate=` override for tuning) with an accumulator:
   per rAF, request enough CFL steps to cover min(T_sim·Δt_wall, 100 ms·T_sim), carry
   the remainder. Handles 60 vs 120 Hz displays and variable CFL dt automatically; the
   100 ms clamp kills the catch-up spiral after tab-hide.

   This is NOT the LOOPLAT adaptive stepsPerFrame controller (see its post-mortem before
   touching this): that was a closed-loop maximizer; this is an open-loop THROTTLE below
   device capability. Fast devices idle — which is a feature: at 256² ~2 ms/step, a
   T_sim needing ~150 steps/s uses ~30% of a desktop GPU, leaving headroom for particles,
   render, and recording (the iPhone rec-slot starvation was GPU saturation).

   Choose T_sim by MEASUREMENT on the slowest target device: T_sim ≈ 0.5× that device's
   sustainable sim rate. Ballpark: ~0.5–1.0 sim units/s ⇒ an eddy turnover plays out
   over ~5 s. If a device still can't keep up, the accumulator grows and the game slows
   down uniformly (the rev-1 behavior remains as the graceful fallback); NEVER skip sim
   time to catch up (teleporting fluid breaks boat consistency), never auto-change
   `selRes` (surface a "below target rate" notice instead; resolution is the player's).
   Pause = T_sim 0; slow-mo effects come free.

## 3. Boat equations of motion

State: x (periodic), v, heading θ (ĥ = (cosθ, sinθ)), ω, sail angle θ_s, charge Q.
Relative flow w = u(x_b) − v. Local field b_⊥, |b|, b̂, and the in-plane cross-field
direction n̂ = ẑ×b̂. All coefficients in §7; all placeholders pending on-device tuning.

### 3.1 Hull drag (isotropic, linear — Alfvén-wing drag on the guide field)

F_hull = γ₀ w

Linear-in-w is not Stokes hand-waving: wing drag is radiation reaction, linear in the
source. Guide-field Mach number w/v_A0 ~ ε in the RMHD ordering — asymptotically valid.

### 3.2 Tether, shorted: the magnetic keel (with Neubauer correction)

Motional EMF along the tether: E_z = |b|(w·n̂) — exactly zero for along-line slip.
Circuit: internal resistance R_t in series with the wing return impedance
R_A = ½|c|, where |c| = √(|b|² + (w·n̂)²) is the characteristic speed the kink actually
propagates at in the boat frame (Neubauer's √(1+M_A²), transplanted; ½ = two wings in
parallel; code units μ₀ = ρ = 1, Σ_A = 1/|c|). Then I_z^ind = ℓ|b|(w·n̂)/(R_t + R_A) and

F_keel = ℓ² |b|² (w·n̂) / (R_t + ½√(|b|² + (w·n̂)²)) · n̂

Limits (all correct by construction): R_t-dominated → γ_b|b|²(w·n̂)n̂; plasma-dominated
sub-Alfvénic → 2ℓ²|b|(w·n̂)n̂; near nulls force → 0 like |b|² with bounded conductance;
fast slip saturates at ≈ 2ℓ²|b|² sgn(w·n̂) (planing regime). Keep R_t > 0 (smoothness).
Lenz sign verified componentwise: relaxes v toward u, same convention as γ₀w.
R_t/R_A is a real design knob: it selects the grip-map exponent (|b|² vs |b|).

Ordering note (Alfred's call, and it holds): O(1) keel-to-hull ratio needs ℓ/a ~ B₀/b ~
1/ε, which is exactly the RMHD parallel/perp aspect ratio — the long-tether boat is an
ordering-consistent object, not a code-units fudge.

### 3.3 Tether, driven: discharge thrust

F_thrust = I_z ℓ (ẑ×b_⊥),  I_z = I_max·trigger,  dQ/dt = −β|I_z|

Same hardware, same force law, current now player-set. Thrust is strictly ⊥ local field
lines and dies at O-points — the mechanic that makes the player read the field. Optional
(later phase): shorted-mode regeneration dQ/dt = +η_r I_ind²R_t with
η_r ≤ R_t/(R_t + R_A) — motor/generator physics, the efficiency is set by the same knob
as the grip map.

### 3.4 Sail (flat plate, apparent wind)

ŵ = w/|w|; signed attack angle α = wrap(θ_s − atan2(w_y, w_x)) ∈ (−π, π]:

F_sail = [ C_L sin(2α)(ẑ×ŵ) + C_D(1 − cos2α)ŵ ] |w|²

Applied at the center of mass (no sail torque in Phase 1). The α → α+π symmetry is
physical for an uncambered plate. The keel that makes sailing work is §3.2 — note its
resistance axis is the LOCAL FIELD (n̂), not the hull: upwind progress depends on the
local field geometry cooperating, and "you can't beat upwind in a becalmed region" is
gameplay, not a bug. (Tuning escape hatch if that proves too cruel: a small hull-frame
γ_perp term, off by default.)

### 3.5 Dipole forces and rotation

I dω/dt = τ_rudder + τ_dipole − b_rot ω
τ_rudder = K_r · steerAxis · |w·ĥ|             (flow-scaled; constant authority = thruster)
τ_dipole = +k_dip (m_x b_y − m_y b_x), m̂ = (cosθ, sinθ)   (ALIGNING sign — rev-1 fix)

`dipoleLock` on ⇒ k_dip active (optionally boosted b_rot); off ⇒ k_dip = 0. Locked, the
dipole also feels the gradient force

F_dip = m ∇|b|      (locked, m̂ ∥ b̂; sampled by finite difference of |b| in the entity pass)

— pulled toward strong field, i.e. into current sheets, where the charge carriers spawn
(§5). Lock to refuel, at the cost of being reeled into the most violent structures; m < 0
is a repulsion/safety mode. Optional passive charge deflection F = q_s·Q(−w_y, w_x)
(rev-1 sign-fixed Q(v−u)×B₀ẑ force) is kept behind a flag, default off — handling that
changes with stored charge; small gameplay value, one more force to tune.

## 4. Controls (waypoint scheme)

- **Tap/click sets a waypoint**; d̂ = minimum-image direction boat→waypoint. Rudder runs a
  PD loop on heading error toward d̂ (gain K_r, damping via b_rot), authority scaled by
  |w·ĥ| as in §3.5. Velocity-matching toward the point (arrive behavior), NOT
  acceleration-matching — acceleration matching in turbulence is jerky and marginally
  stable.
- **Auto-trim**: coarse scan over θ_s (~36 angles per substep, trivial) maximizing
  F_sail·d̂. No tacking autopilot: dead-upwind waypoints stall visibly and the player
  places tack waypoints — that is the skill. Manual sail mode deferred past Phase 1.
- **Discharge**: hold (Space / touch button); sign auto-selected, I_z sign =
  sign((ẑ×b̂)·d̂) — "boost toward the waypoint when the geometry allows."
- **Dipole lock**: toggle (D / touch button).

## 5. Charged-particle economy

Carriers are real charged particles (TESTPART WGSL Boris kernel; q/m chosen so
ρ = v_⊥/Ω ≈ 2–4 dx: Ω ≈ 10–20 at 256², i.e. 60–120 solver steps per gyration — resolved
at solver dt with no substepping). They gyrate visibly, E×B-drift with the flow, and heat
in the turbulence.

- **Attraction** (r ≤ R_att, minimum-image): velocity bias k_pull·r̂ toward the boat.
- **Capture** (r ≤ R_col): mark dead, atomicAdd charge credit. Credit is v_⊥-weighted
  (ΔQ ∝ ½v_⊥², clamped): hotter particles are worth more — the player is literally
  harvesting stochastically heated ions, and sheets (where heating and E_z acceleration
  concentrate) become the rich fishing grounds by physics rather than by fiat.
- **Respawn**: fixed budget N_p; spawn laws `uniform` or `sheet` (rejection-sample on
  |j| from a finite-difference Laplacian of the bound ψ gradients), flag-selected,
  `sheet` default. Spawn velocities Maxwellian at a set v_th; own RNG stream.

## 6. HUD / overlay

On the transparent overlay canvas (arrow-overlay idiom), from CPU boat state + the async
sample: w vector at the bow (blue); local b̂ line through the boat (magenta); thrust
vector while firing (yellow); charge arc around the hull; dipole tick when locked; sail α
indicator; waypoint marker + minimum-image bearing line. Particle sprites sized by v_⊥
(gyroradius IS the size — it is real). No extra readbacks beyond the §2 buffer.

## 7. Baseline parameters — ALL placeholders, code units

Box 2π, u_rms ~ O(1), eddy turnover ~ few sim units, dt ~ 5e-3 at 256².

```js
export const GAME_PARAMS = {
  boatMass: 1.0, rotationalInertia: 0.15,
  hullDrag: 0.5,                    // gamma_0
  tetherLen: 1.0, tetherRt: 0.3,    // keel: F ~ l^2 |b|^2 w_n / (Rt + 0.5*sqrt(|b|^2+w_n^2))
  liftCoeff: 1.5, dragCoeff: 0.4,
  rudderGain: 2.0, rotDamping: 1.5,
  dipoleTorque: 3.0, dipoleMoment: 0.5,   // k_dip; m for grad-|b| force when locked
  qMax: 100.0, qPerHeat: 30.0,      // DeltaQ = min(qPerHeat * v_perp^2/2, 25)
  dischargeRate: 25.0, iMax: 5.0,   // full-charge burn ~0.8 sim units
  nParticles: 4096, qOverM: 15.0, vth: 0.5,   // rho ~ 3 dx at 256^2
  attractRadius: 0.08 * L, collectRadius: 0.02 * L, pullStrength: 1.5,
  maxSubsteps: 4,
};
```

Tuning targets, not spec: sail-only cruise ≈ 0.5·u_rms; full-charge discharge beats any
sail speed but costs the tank in < 1 eddy turnover; keel grip in an average-|b| region
roughly 5× hull drag for cross-line slip.

## 8. Solver configuration

256² default, forced turbulence (demo defaults scaled per SPEC §2: hyper=2,
eps±=(0.15,0.15), fshell=(1,3)); 512² offered, not default; `selRes` never locked. The
tearing-campaign resolution rules do not gate this preset (nothing fits a published
number); they DO gate any future equilibrium preset.

## 9. Exit criteria

1. Phase 0 gates green (self-test, byte-identical RNG reference) and stay green with
   game.html open in another tab.
2. Solver regression: entity pass on/off leaves solver state after N steps identical.
3. Keel test: cross-line slip visibly damped vs along-line slip; from rest, sail-only,
   the boat makes net progress against the local flow where the field geometry allows.
4. Dipole lock settles ALIGNED within ~1 sim unit, no limit cycle; locked boat drifts
   measurably up ∇|b| toward a sheet.
5. Thrust bends the trajectory across field lines, drains Q at β|I_z|, and produces no
   force at an O-point center (fly through one and check).
6. Particles: TESTPART WGSL kernel unmodified; capture credit arrives via the pooled
   async readback; frame time at 256² + 4096 particles within ~1 ms of the plain preset
   on a phone.
7. Sim-time invariance: machines advancing 1 and 20 steps/frame produce the same boat
   trajectory over 10 recorded sim units (the pin changes pace, never the trajectory).
8. Pinned pace: on the reference desktop and the slowest target phone, measured sim
   units per wall second within 10% of T_sim over a 60 s run; on a device below
   capability, the game slows uniformly and the "below target rate" notice shows.
9. Wall of shame check: no wall-clock quantity anywhere in dynamics or fits (grep for
   performance.now outside render/recording/pacing code; the pacing accumulator is the
   ONE sanctioned consumer of wall time, and it only sets how much sim time to request).

## 10. Execution notes (for the implementing session)

- Standard flow: opus/sonnet agents implement; oversight by the orchestrating model;
  adversarial review by a separate FRESH agent that has not seen the implementation
  conversation. Fix majors before anything is committed.
- Record the RNG reference BEFORE touching anything solver-adjacent (Phase 0 included);
  re-record after and require byte-identity. The devtools node suite (stubenv/bootstub +
  check2dspec.js etc., wgslparse.mjs) is the GPU-less gate set; the in-browser refvector
  self-test on a real device is the final gate and is run by Alfred.
- Phase 0 constraint: pages must keep working from file:// — no build step, no ES-module
  imports (Chrome blocks module scripts from file://). solver2d.js loads exactly the way
  common.js does (classic script tag); follow common.js's namespace conventions.
- Comment style: comments say what the code does, succinctly; design history lives in
  this plan, not in source.
- Entity/game code consumes ZERO solver RNG; particle/game seeding uses its own stream.
