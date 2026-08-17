# Autodiff plan — forward-mode sensitivities: growth rates, saturation, statistics

**Status: DRAFT** (written 2026-08-15; nothing started). Three rungs; rung 1 needs no
repo changes at all. Each rung that touches `taranis/` lands via the usual
implement → adversarial-review flow.

## Goal

Make physically interesting sensitivities computable by autodiff, in increasing order of
invasiveness and difficulty:

1. **Rung 1 — IC/equilibrium parameters** (e.g. current-sheet width a), target = linear
   growth rate. Pure notebook work.
2. **Rung 2 — physics parameters** (η first; in principle anything in `eqpars`), target =
   growth rate. Introduces the one repo change: an `overrides` seam.
3. **Rung 3 — beyond the linear phase**: (3a) saturated island amplitude (deterministic
   steady state), (3b) statistical steady-state amplitudes in forced/chaotic runs.

The long-range motivation is fusion-relevant instabilities, where the quantities that
matter are growth rates *and* saturated amplitudes as functions of equilibrium and
physics parameters.

## Why forward mode (decision)

- Objectives are scalars; parameter counts are O(1)–O(5). `jax.jvp` costs ~2× a primal
  run per tangent with O(1) memory (one tangent copy of the state pytree). Reverse-mode
  cost scales with the number of *outputs* (= 1) — it buys nothing here.
- `run.py`'s snapshot driver uses `lax.while_loop`: forward-differentiable, NOT
  reverse-differentiable in JAX. Reverse would force a scan rewrite plus trajectory
  storage (128×1024, ~2e4 steps: ~200 GB naive, ~300 MB with sqrt-T remat — possible,
  pointless).
- The linear phase is non-chaotic: tangents grow with the mode, not with a Lyapunov
  exponent, so dγ/dp is well-conditioned. (Rung 3b is where this breaks; see below.)

**Rejected: reverse mode / adjoints** (above). **Rejected: full traced-Parameters
redesign** — see rung 2 for the override design that replaces it.

## What is already AD-friendly (audited 2026-08-15)

- The integrating factor is evaluated *inside* the traced region: propagators store L
  (array data on `kgrid`, built once in `grids.setup_kgrids`) and compute exp(L·τ) at
  call time with τ = dt traced (Putzer 2×2 and the sinh(z)/z Taylor branch are smooth
  jnp code). Adaptive dt forced this design; AD inherits it for free.
- LSRK stage loops and step blocks are `lax.scan` — forward-differentiates.
- `kgrid` is a pytree of arrays (not static config), so L can carry tangents today.
- The IC path (`run.py::initialize`) is user jnp code + a constant dealias-mask multiply.

Things that are static and must stay static (never in the differentiable surface):
grid shapes, dims/flags, the integer `hyper` exponent, `cfl_every`.

## Common experimental protocol (all rungs)

- `RMHD_PRECISION=64`. Seeds are 1e-7 and tangents ride on top; fp32 is not enough.
- **Fixed dt** (`adaptive_timestep` off) inside anything differentiated: removes the
  dt(fields) derivative path and its kinks. Choose dt from a primal reconnaissance run.
- **Frozen windows**: no data-dependent window/threshold selection inside the
  differentiated function. Pick fit windows from the primal run, then freeze.
- **FD gate** (acceptance criterion, every rung): central finite differences at ≥2 step
  sizes must match the jvp with the expected convergence order before any derivative is
  believed or used downstream. Record the comparison in the notebook.
- Rungs 1–2 run unforced (no RNG in the differentiated path). Rung 3b is forced: fix
  `forcing_key` (common random numbers) so the derivative is pathwise at a frozen noise
  realization. Repo rule applies: record the RNG reference before any forcing-adjacent
  change.

## Rung 1 — equilibrium width via the IC (no repo changes)

**Equilibrium family:** B_y(x; a) = tanh(sin x / a) / tanh(1/a) (normalization pins
B_y(π/2) = 1). Exactly 2π-periodic; odd under x → x+π so the mean flux is zero and ψ
stays periodic; near each null it is locally a Harris sheet of width exactly a
(B ≈ tanh(x/a)); a ~ 1 sits near the cos(x) baseline. ψ is obtained spectrally,
ψ̂ = B̂_y/(i k_x) (k_x ≠ 0, zero mean) — all jnp, differentiable in a. Seed modes and
box as in `tearing-growth-vs-k.ipynb` (long-box trick: k stepped over box harmonics).

**Objective:** γ(a) = [ln A(t₂) − ln A(t₁)] / (t₂ − t₁) with A = |ψ̂(k_x=0, k_seed)|,
window frozen in the clean exponential phase, with the same equilibrium-decay
correction the existing tearing notebooks use (the decaying background is itself
differentiable, so the correction goes through the jvp).

**Targets, in order:**

1. dγ/da via `jax.jvp`, FD-gated.
2. Local logarithmic derivative a·dlnγ/da against layer theory through the chain rule
   on Harris Δ′(k,a): Δ′a = 2(1/ka − ka), FKR regime γ ∝ η^{3/5} Δ′^{4/5}. Compare in
   the small-Δ′ part of the band where FKR holds.
3. **Marginal curve:** Newton on γ(a) = 0 at fixed k, using forward-over-forward
   (jvp-of-jvp) for the second derivative. Expected answer: the boundary lands on
   ka = 1 (Δ′ = 0). An AD-computed marginal curve reproducing Δ′ = 0 is the headline
   validation of the whole approach.

Cost: 128×1024 primal run is minutes on a laptop; jvp ≈ 2×; Newton needs a handful of
iterations. Deliverable: `examples/tearing-sensitivity-2D.ipynb` alongside the
existing tearing pair.

## Rung 2 — physics parameters via an overrides seam

**Design decision: params keeps everything; differentiability is per-experiment.**
`Parameters` stays the single static, hashable, self-documenting record it is now. A
new optional `overrides` pytree (fixed key structure at trace time, traced values) is
threaded `simulate → stepper → rhs / linear_matrix_func`. Which parameters are
differentiable is a property of the *call* (the override pytree's structure), mirroring
JAX's own argnums philosophy — not a global property of the config model.

Rules:

- `overrides=None` is a **Python-level branch**: the compiled graph must be literally
  today's. **GATE: bitwise regression on the three reference configs** (same harness as
  PRECISION_PLAN A5).
- All parameter reads funnel through one helper (`getp(params, overrides, key)`), at the
  *read sites*, not patched into derived arrays — η enters through L *and* through the
  CFL dt estimate; a read-site seam catches every use, ad-hoc L patching silently
  misses secondary reads. The refactor's real content is grepping down every
  `params.eqpars` read once.
- `params.save` stamps active override values into the run directory (the
  self-documenting-run-dir invariant survives).
- Static/structural parameters are excluded from the override space by construction.

Side benefit, AD aside: traced values mean η sweeps reuse the compiled step instead of
retracing per value — dispersion-style scans get this for free.

**Targets:** dγ/dη FD-gated and cross-checked against the `tearing-mode-2D` γ(η) sweep;
then the one-run local exponent dlnγ/dlnη (FKR 3/5 → Coppi 1/3 crossover along the
dispersion curve, no parameter sweep).

## Rung 3a — saturated island amplitude (deterministic steady state)

The saturated island is a stable fixed point (up to the slow resistive decay of the
equilibrium, timescale ~1/η — quasi-static relative to saturation at t ~ few hundred;
carry the usual decay caveat). Two routes, same machinery:

- **Converged tangent:** integrate the jvp past saturation; the tangent's transients
  decay at the island's own stability rate and the tangent converges to dx*/dp. Read
  dW_sat/dp (or d of any island functional) off the converged tangent. No new code.
- **Implicit differentiation** (sharper, optional): at the fixed point solve
  (∂F/∂x)·δx = −∂F/∂p matrix-free (GMRES with jvps of the RHS).

Validation target: saturation theory W_sat ∝ Δ′ (White/POEM-type), so dW_sat/da checks
against the same Harris Δ′(k,a) as rung 1.

## Rung 3b — statistical steady state (the honest frontier)

**Step zero — measure λ₁ with the machinery we already have.** A jvp with zero
parameter-tangent and a random state tangent *is* the tangent-linear model: Benettin
renormalization of its norm gives λ₁, and extra orthonormalized tangents count the
positive exponents m. This is the decision point, and it is nearly free.

**If λ₁ ≈ 0** (steady or periodic saturated state): a time-averaged scalar amplitude
⟨A⟩_T over a frozen window is a fine objective — accumulate the average (and its
tangent) online inside the scan, O(1) memory, differentiate directly. Done.

**If λ₁ > 0** (chaotic): the naive estimator fails, and fails *counterintuitively*:
d⟨A⟩_T/dp of a single trajectory does not converge to d⟨A⟩_∞/dp — the tangent grows
like e^{λ₁t}, so the derivative estimate diverges as the averaging window grows.
Longer averaging makes it worse, not better (Lea–Allen–Haine). Options, ranked by
implementation cost on top of rungs 1–2:

1. **Ensemble of short-window tangents:** vmap the jvp over N independent ICs, window
   T_seg ~ a few 1/λ₁; variance ~ e^{2λ₁ T_seg}/N. Embarrassingly parallel, zero new
   machinery, logarithmically slow convergence — but plausibly sufficient at 2D-RMHD λ₁.
2. **Forward NILSS:** m+1 tangent trajectories, segmented QR + a small least-squares
   problem; forward-mode-only, no adjoint needed for O(1) parameters. Cost scales with
   m — which step zero measures. The serious tool if (1) is too noisy.
3. **Linear-response/FDT estimate** from correlation functions of a single run: cheap,
   uncontrolled bias; cross-check only, never primary.

**State of the art in fusion (checked 2026-08-15).** Differentiable gyrokinetics now
exists: iGENE (differentiable flux-tube GENE in TensorFlow, Phys. Plasmas 33, 083901
(2026), arXiv:2605.03086) reverse-mode-differentiates *nonlinear time-averaged fluxes*
and does profile-matching optimization with the results — but handles chaos by
truncation: gradients computed over short windows from a saturated state, empirically
divergent beyond ~512 steps (≈ the flux autocorrelation time, i.e. a few 1/λ₁ — exactly
the bias-variance wall above), landing at 15–50% of the FD reference, "directionally
correct". No shadowing, no ensembles. gyaradax (JAX flux-tube GK on the GKW model,
arXiv:2604.06085) has AD for inverse problems/sensitivities. So the *gap* is not
differentiable plasma codes — it is a *controlled* statistical-sensitivity estimator
(shadowing or quantified-bias ensemble) in any plasma turbulence code. RMHD is a far
cheaper place to build that than gyrokinetics. This rung is a paper-scale project, not
a notebook.

**Task (Alfred): reading list before committing to a 3b approach.**

- [ ] Lea, Allen & Haine, Tellus A 52, 523 (2000) — the phenomenon on Lorenz '63;
      ensemble-of-short-windows fix. Read first.
- [ ] Ruelle, Commun. Math. Phys. 187, 227 (1997) + his Nonlinearity 22, 855 (2009)
      review — why ⟨A⟩(p) is differentiable yet the naive tangent diverges
      (stable/unstable split of the response formula).
- [ ] Eyink, Haine & Lea, Nonlinearity 17, 1867 (2004) — heavy tails / diverging
      variance of ensemble gradient estimators (caveat on option (i)).
- [ ] Wang, Hu & Blonigan, JCP 267, 210 (2014) — least-squares shadowing; then
      Ni & Wang, JCP 347, 56 (2017) — NILSS (the forward-mode-friendly version).
- [ ] Baladi, ICM proceedings (2014), "Linear response, or else" — when linear
      response genuinely fails; why noise/chaotic hypothesis rescues physics.
- [ ] iGENE paper (arXiv:2605.03086) — what truncated-window gradients buy in practice
      in GK, and where they stop; gyaradax (arXiv:2604.06085) for the JAX-GK landscape.

## Order of work

1. Rung 1 notebook (no repo changes) — includes the FD gate and the marginal-curve
   Newton demo.
2. Overrides seam + bitwise gate; then rung 2 targets.
3. λ₁/m diagnostic notebook (step zero of 3b, also validates 3a's stability-rate
   assumption); then 3a; 3b as its own planned project once m is known.
