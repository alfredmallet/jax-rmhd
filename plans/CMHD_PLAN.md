# CMHD_PLAN — compressible MHD (z_spectral, serial) in taranis

Status: **COMPLETE, 2026-08-30** — all phases landed on main (C0 `fd814c6`, C1 `38d37c1`,
C2 `210ad69`, C3a `8e2d229`, C3b `f7271ae`, each with its close-out commit), every phase
opus-implemented (C0/C3a by the session directly, Alfred's call) and fable-adversarially
reviewed before landing; the dated landing notes in §5 carry the review verdicts and
measured numbers. Remaining follow-ups live in §11, none blocking.
(History: rev 1 folds in Alfred's §10 answers: isothermal
default, EBM equations from Squire et al. 2020, radial axis x; rev 2: OT reference =
Athena, first-target regime β = 0.3, δB/B₀ = 1, C0 executed by the session directly
rather than a subagent (Alfred's call), §3.3 div-B claim corrected from "exactly zero"
to "round-off with no systematic source" — the pairwise products differ in fp grouping). Phases C0–C2 are the MVP (polytropic compressible MHD,
`dims==3` + `z_spectral=True`, single process, isotropic hyperdissipation, IF-LSRK);
Phase C3 adds expanding-box (EBM) terms. Execution follows the house flow: Opus
implementer subagents per phase, session oversight, fresh adversarial review before a
phase is recorded as landed. §9 is the digest of repo rules every implementer must obey;
read it before writing code.

## 1. Motivation and scope

A fully-spectral compressible MHD solver sharing taranis's core (steppers, propagators,
run loop, checkpointing) via the equation registry, for compressible-turbulence and
expanding-box studies. Scope decisions, fixed:

- `dims==3` + `z_spectral=True` + `size==1` ONLY (which `Parameters` already forces
  together). No FD-z path, no MPI, no `comm_backend="jax"`. 2D physics runs as 2.5D
  (z-independent fields at small nz — an unforced z-independent state stays exactly
  z-independent, the same trick `tests/test_particles_3d.py` uses).
- Polytropic closure `p = K ρ^γ` (γ ≥ 1; the enthalpy branch is trace-time python).
  **Default γ = 1, isothermal** (§10 answer) — consistent with the Squire et al. 2020
  EBM target; γ > 1 stays supported and gated. No energy equation, no shocks: this is a
  pseudospectral code, target regime is smooth compressible turbulence at sonic Mach
  M_s ≲ 1 with hyperdissipation absorbing the small scales.
- Isotropic (hyper-)dissipation `−ν (k_x²+k_y²+k_z²)^h` per field, the ONLY content of
  the linear operator L (§3.1).
- No forcing, no particles in the MVP (both rejected at validation; forcing is a natural
  later phase, particles already assert `eqtype=="RMHD"`).
- Time integration: the existing IF-LSRK schemes (lsrk54 default). CB-IMEX also works
  unchanged on a pure-dissipation diagonal L (`solve_shifted` is elementwise) — no scheme
  restriction needs enforcing, unlike GDI's IF caveat.

## 2. Equations, variables, conventions (code units)

Fields, in state order (nfields = 7):

    fields[0] = ρ̂          density (k=0 mode carries ρ₀; convention ρ₀ = 1)
    fields[1:4] = û_x,y,z   velocity
    fields[4:7] = B̂_x,y,z   magnetic field (Alfvén units; k=0 mode carries B₀)

Equations (ideal part; D = isotropic hyperdissipation added to each):

    ∂ρ/∂t = −∇·(ρu)                                  (flux form: mass exact, §3.2)
    ∂u/∂t = −(∇×u)×u − ∇(u²/2 + h(ρ)) + (∇×B)×B/ρ    (rotational form, §3.4)
    ∂B/∂t = ∇×(u×B)                                  (curl form: div B exact, §3.3)

with the polytropic enthalpy (the pressure force is exactly ∇h since (1/ρ)∇p = ∇h):

    h(ρ) = (γK/(γ−1)) ρ^(γ−1)     γ > 1,   K = c_s0²/γ  (so c_s(ρ₀=1) = c_s0)
    h(ρ) = c_s0² ln ρ             γ = 1    (isothermal)

There is NO separate linear wave term: the background B₀ and ρ₀ live in the k = 0 modes
of the evolved fields, so Alfvén/fast/slow propagation emerges from the quadratic terms
(u×B contains u×B₀, etc.). Both k = 0 modes are exactly conserved: the curl and flux
forms have no k = 0 source and hyperdissipation vanishes at k = 0. This is the single
most load-bearing design fact — it removes any B₀/ρ₀ `eqpars` entry, any linear term
func, and any background/fluctuation split, and it means the dispersion gates test the
production RHS, not a separate linear code path.

Conserved ideal invariants and their discrete status (C0 writes the derivations into
docs/numerics.md, C1 gates them):

| invariant | continuous | discrete status |
|---|---|---|
| mass ∫ρ | exact | BITWISE: the k=0 RHS is exact fp zero, exp(0)=1, mask=1 (numerics.md) |
| mean B | exact | BITWISE (same argument, curl form) |
| div B = 0 | exact | round-off random walk, no systematic source (§3.3) |
| E = ∫ ρu²/2 + B²/2 + ρe(ρ) | exact (ν=0) | O(dt^p) + aliasing residual (§3.5); gate asserts dt-order + smallness, never round-off |
| cross helicity ∫u·B | exact (ν=0) | same class as E |

with ρe(ρ) = Kρ^γ/(γ−1) for γ > 1 and c_s0² ρ ln ρ for γ = 1.

Linear waves about (ρ₀=1, B₀): ω² branches — Alfvén ω² = (k·v_A)², fast/slow
ω² = ½(c_s²+v_A²)k² [1 ± √(1 − 4 c_s² (k·v_A)² / (c_s²+v_A²)²/k²)] — C0 records the
exact expressions and the eigenvectors the gates project onto.

## 3. Design decisions (with rationale)

### 3.1 L = dissipation only; waves explicit. IF-LSRK is correct under exactly this split.

The compressible wave operator couples all 7 fields per k-mode; the propagator framework
tops out at dense 2×2 (`Putzer2Operator`), and both alternatives are bad: a 7×7
matrix-exponential backend is ~50 grid-sized precomputed complex arrays and a new
propagator family, and IMEX-ing the waves violates the standing rule (never an L-stable
solve on a wave-dominated L at |ω|dt ≳ 1). With waves explicit the only stiffness left is
hyperdissipation, which IF treats exactly — the same division of labor as RMHD's perp
dissipation. Cost: dt ≲ cfl_safety·min(dx,dy,dz_eff)/max(|u|+c_f), c_f² = c_s²+v_A²
(the angle-maximised fast speed, a safe upper bound) — at low M_s the step shrinks ∝ 1/c_s.
That is inherent to explicit compressible spectral codes and accepted; M_s ≪ 1
semi-implicit acoustics is out of scope (§8).

Verified against the tree (2026-08-29): `propagators.build` (propagators.py:340) accepts
a 4-d diagonal `(1 or nfields, nz-or-1, nkx, nky)` L; kz-dependent entries are allowed
off the `"jax"` backend (the :368 guard); `DiagonalOperator` is elementwise and supports
both `exp_op` (IF) and `solve_shifted` (IMEX). So the MVP needs ZERO new propagator code.
A uniform ν is the broadcast leading-axis-1 diagonal (one exp per stage, broadcast over
fields); per-group ν uses leading axis nfields. `DiagonalOperator.hoistable` is False —
deliberate elsewhere, and irrelevant here: a broadcast diagonal exp is cheap. Do NOT make
it hoistable for CMHD; that touches the FD-z/2D bitwise gates for no measurable win.

### 3.2 Evolve ρ (not ln ρ), flux form.

Flux form `−ik·FFT(ρu)` conserves mass to round-off (k=0 component is exactly zero) and
gives a clean gate. ln ρ would guarantee positivity but surrenders exact mass
conservation. At M_s ≲ 1 with adequate hyperdissipation, ρ stays comfortably positive;
`set_timestep` gates on it anyway (§4, the NaN path: c_s(ρ<0) is NaN at non-integer γ−1
and poisons dt loudly rather than silently). If a later high-Mach campaign hits
positivity, ln ρ is a contained change to three term funcs — noted in §8, not built.

### 3.3 Induction in curl form: div B = 0 to round-off, free.

`∂B̂ = ik×FFT(u×B)` (then dealias). `k·(ik×Ê)` cancels pairwise ANALYTICALLY; in fp the
paired products differ in grouping (`k_x·(k_y·E_z)` vs `k_y·(k_x·E_z)` round
differently), so each RHS evaluation deposits O(ε_mach·|k|²|Ê|) into k·∂B̂, a random
walk with NO systematic source: the IF exponential and the dealias mask multiply all
three B components by the same factor and cannot create divergence, and nothing feeds
div B back into the dynamics. No cleaning step, no vector potential, no projection. The
C1 gate asserts max_k |k·B̂|/(|k| |B̂|) stays at a stated round-off-scale tolerance
(~ε·√N_steps class) with no secular trend after O(100) steps — round-off, not bitwise
zero. Corollary: any future term added to the induction equation
must also be a curl (or k-local and divergence-free) — record this as a rule in the
module docstring.

### 3.4 Momentum in rotational form; one combined gradient; one combined curl-force fft.

`u·∇u = (∇×u)×u + ∇(u²/2)` lets `∇(u²/2)` merge with `∇h` into a single scalar gradient
`ik·FFT(u²/2 + h)`, and `(∇×B)×B/ρ − (∇×u)×u` is summed in real space and fft'd ONCE
(3 component transforms, not 6). Transform tally per RHS evaluation, the C1 reference
(assert it in review, measure it in C2): ifft ρ, u(3), B(3), ω(3), j(3) = 13; fft
combined-curl-force(3), scalar(1), u×B(3), ρu(3) = 10; total 23 3-D FFTs. RMHD
z_spectral is 10 (8 grad iffts + 2 NonlinearTerm ffts), so expect ~2–2.5× the RMHD step
at the same grid; state memory is 7/2 of RMHD. C2 measures the real number
(docs/performance.md); estimates are never quoted as measurements.

### 3.5 Dealiasing stance: 2/3 rule, non-polynomial residual accepted and documented.

The quadratic terms are exactly dealiased by the existing 2/3 mask (already
(nz,nkx,nky) under z_spectral, grids.py:38 `dealias_mask`). h(ρ), 1/ρ (and ln ρ at γ=1)
are non-polynomial: no finite padding dealiases them, every spectral compressible code
lives with this, and at M_s ≲ 1 with smooth ρ the residual sits below the
time-discretization error the conservation gates already budget for. C0 writes one
paragraph in docs/numerics.md so it is a recorded decision, and the conservation gates
assert convergence ORDER in dt plus absolute smallness — never round-off — so the
aliasing residual is inside the gate's stated tolerance, not hidden by it.

### 3.6 `CMHDGrads` carries VALUES, not just gradients.

`set_timestep(grads, params)` receives only what `grad_func` returned (run.py:105), and
the term funcs need real-space values anyway. So `grad_func` returns

    CMHDGrads(rho, u, B, omega, j, t)
    # rho (nz,nx,ny); u, B, omega=∇×u, j=∇×B each (3,nz,nx,ny), real space; t = state.t

— the 13 iffts of §3.4, computed once, consumed by both the terms and the CFL. `t` rides
along because `set_timestep` has no other access to it; the MVP ignores it, the EBM CFL
(§7) reads it. This is per-equation freedom the recipe contract explicitly grants (GDI
and RMHD each define their own NamedTuple); do NOT try to route this through
`shared_physics.grad_fields`, which is a (d/dx,d/dy)-pair machine for bracket equations.

## 4. Module design: `physics/cmhd.py`

Follow gdi.py as the structural template (module docstring with the physics source and
normalization; `_check_supported`; `_eqpars` with required/optional key validation
rejecting unknowns; recipe functions; nothing user-facing/read-only — that goes in
`diagnostics/cmhd.py`).

- `_check_supported(params)`: require `spatial_dimensions == 3`, `z_spectral`,
  `not params.forcing`, and (belt-and-braces; `Parameters` already forces it) `size == 1`.
  Error messages name this plan.
- eqpars schema: required `cs0` (> 0), `diss` (scalar, or length-3
  `(D_rho, nu, eta)` expanded to the 7-field diagonal), `hyper` (int ≥ 1). Optional:
  `gamma` (float ≥ 1, **default 1.0** = isothermal); C3 adds `expansion`. Unknown keys rejected (GDI precedent, rmhd's
  `_diss_hyper` precedent).
- `linear_matrix(kgrid, params)`: `−diss_f · (ksq + kz_deriv²)^hyper` with `kz_deriv`
  the Nyquist-zeroed kz (copy `rmhd._kz_deriv`'s rule exactly — even-nz Nyquist plane set
  to 0; k² is even in kz so hermitian-compat holds, `propagators._check_hermitian_compatible`
  will verify). Shape `(1, nz, nkx, nky)` for uniform diss, `(7, nz, nkx, nky)` otherwise.
- `grad(state, kgrid, params)` → `CMHDGrads` (§3.6). ω and j are formed in k-space
  (ik× is k-local, using the SAME Nyquist-zeroed kz for every ∂_z anywhere in the module
  — one module-level helper, used by grad, the terms, and nothing else) and ifft'd.
- Term funcs (5 positional args `(state, grads, kgrid, params, halo)`, `halo=None`
  ignored — no halo under z_spectral):
  - `NonlinearTerm`: the whole ideal RHS of §2 as ONE term (the three equations share
    the real-space products; splitting them into separate Terms would recompute or
    re-transform). Multiply by `kgrid.dealias` exactly once, on the assembled
    `(7, nz, nkx, nky)` RHS.
  - That is the ONLY MVP term. `construct_rhs` requires a non-empty active set — fine.
- `set_timestep(grads, params)`: `c_s² = γK·rho^(γ−1)` (γ=1: `c_s0²`), `v_A² = |B|²/rho`,
  `c_f = sqrt(c_s² + v_A²)`; `max_all = max_i max_grid (|u_i| + c_f)/d_i` over the three
  directions (`params.dz` exists for every dims==3 run — config.py sets it
  unconditionally; z_spectral does not remove it); `allreduce_max` (identity under
  serial, keep it for uniformity); return `cfl_safety/max_all` clamped by the fixed-dt
  ceiling as rmhd does. No quiescent floor is needed, but the argument is the GRID max,
  not pointwise (C0 review finding): at γ > 1, c_s(ρ) < c_s0 wherever ρ < 1; mass
  conservation pins max_x ρ ≥ 1, so max_x c_f ≥ c_s0 — say that in the comment rather
  than importing `QUIESCENT_EPS`.
- Registry entry in `physics/__init__.py`: `EquationRecipe(set_timestep, (Term(NonlinearTerm),),
  grad, nfields=7, linear_matrix_func=linear_matrix)` — no forcing_scale_func, no
  halo_start_func.
- `config.py`: nothing structural — `eqtype` validation is registry-driven (config.py:67),
  `nfields` flows from the recipe (config.py:70). Add the CMHD row to `_validate_compat`'s
  comment block only if a check actually lands there; prefer keeping every CMHD rule in
  `_check_supported` (GDI precedent) so config stays equation-agnostic.

Initialization: `run.initialize` is already equation-generic (func → real space → fft →
dealias). Provide `cmhd.uniform_plus_perturbation`-style IC HELPERS in the diagnostics
or examples layer, not in physics/: eigenmode ICs for the gates are built in REAL space
(cos/sin of k·x with the analytic eigenvector) — never written directly into k-space, so
the rfftn reality constraint (both mirrors, CLAUDE.md) is satisfied by construction.

## 5. Phases

### Phase C0 — derivation and conventions (docs only; small; blocks C1 review, not C1 start)

Executed by the session directly (Alfred, 2026-08-29: "for derivation do it yourself"),
with the fresh-session adversarial review kept as the gate.

**Landed 2026-08-29.** The docs/numerics.md § "Compressible MHD" section is written; the
adversarial review (fresh session, independent re-derivation of items A–H) returned
PASS — every gate-bearing formula confirmed — with six non-blocking findings, all folded
in: the γ>1 quiescent-floor argument corrected to the grid-max form (§4 here), the
params.dz claim corrected (§7 here), the div-B deposit scale corrected to ε|k|²|Ê|
(§3.3 here), a signed-zero footnote on the bitwise-k=0 claim, the γ=1 δE/δρ constant
noted as provably dropping out of the budget, and the Alfvén δB/δu relation pinned to
the unambiguous −(k_∥B₀/ω) form for gate transcription.

Write docs/numerics.md § "Compressible MHD": §2's equations with the code-unit
conventions, the rotational-form identity and the combined-gradient trick, h(ρ) both
branches, the k=0-carries-background argument and its exactness, the invariant table with
discrete status, the dispersion relations WITH eigenvectors (the gates project onto
them), the div-B exactness argument (§3.3), the dealiasing stance (§3.5), and the
23-transform tally. Deliverable: the docs section + any correction back into this plan.
Gate: adversarial review by a fresh session against a plasma-physics textbook derivation
— every sign checked, since the gates in C1 are built FROM this section (TESTPART_PLAN
§2's E_z sign lesson: derive once, in writing, before coding).

### Phase C1 — core physics + gates (the big one)

Files: `taranis/physics/cmhd.py` (new), `taranis/physics/__init__.py` (registry entry +
import), `tests/test_cmhd_linear.py`, `tests/test_cmhd_conservation.py` (new).

Implementation per §4. Tests (house form: `from _rmhd_testing import bootstrap;
bootstrap()` first, `script_main(globals())` footer, `fresh_params(...)` not `ctx()` when
passing dict-valued kwargs — eqpars dicts hash fine through ctx's sorted-items key ONLY
if hashable; dicts are not, so use `fresh_params`):

1. **Dispersion gates** (test_cmhd_linear.py, fp64; fp32 versions with loosened
   tolerances only where they pass without weakening the assertion): single-eigenmode
   real-space IC at amplitude ε = 1e-6 on (ρ₀=1, B₀ẑ via the k=0 mode of the IC), ν = 0,
   fixed small dt; measure the oscillation frequency of the projected eigenamplitude over
   a few periods. Assert relative error vs analytic ω(k) ≤ tol(dt, ε) for: Alfvén, fast,
   slow, each at ≥ 3 propagation angles including exactly-parallel and
   exactly-perpendicular (where slow → degenerate and fast → magnetosonic — assert the
   degenerate limits explicitly), at both γ = 5/3 and γ = 1, and at c_s0/v_A ∈ {0.5, 2}.
   Also assert no spurious growth: |amplitude| constant to O(ε²·t) with ν = 0.
2. **Dissipation-only exact decay**: u = B = 0 perturbation... rather: tiny-amplitude
   single mode with ν > 0, h ∈ {1, 2}; the IF step applies exp(L dt) exactly, so each
   field's mode decays by exp(−ν k^{2h} dt) per step to round-off when the nonlinear
   contribution is negligible — assert at the ε where the quadratic terms sit below
   round-off of the linear decay.
3. **Invariant gates** (test_cmhd_conservation.py): random smooth IC (band-limited,
   M_s ≈ 0.3, δB/B₀ ≈ 0.3), ν = 0, ~50 steps. Mass and each mean-B component: BITWISE
   (the k=0 modes are exact invariants of the discrete step — numerics.md derivation).
   max_k |k·B̂|/(|k||B̂|): round-off-scale tolerance, no secular trend. Energy and cross helicity: run at dt and dt/2, assert the
   drift ratio matches the scheme order p (lsrk54: measured order ≥ 4 within a stated
   band) AND absolute drift below a stated small bound. Never assert these at round-off
   (§3.5).
4. **Scheme cross-checks**: lsrk54 vs rk44 vs imexcb3e on the same IC agree to
   O(dt^min-order); `lsrk_scan` True/False agree to round-off (the fusion caveat from
   CLAUDE.md applies — tolerance, not bitwise).
5. **Plumbing gates**: `params.save`/`from_snapshot` round-trips the CMHD eqpars;
   snapshot save/load restart continues bitwise (fixed dt, ν > 0, the standard
   `forcing_scale` zeros path); unknown/missing eqpars raise; `forcing=True` raises;
   FD-z (`z_spectral=False`) raises; `hoist_propagator` on/off agree (diagonal is
   unhoistable — assert `stage_exp_ops` returns the working ops or None per its contract
   rather than assuming).

Explicitly NOT touched: `tests/data/refactor_reference_*` and gate-6 references (frozen,
RMHD/GDI-only — a new eqtype must not require regenerating them; if any existing gate
goes red, that is a C1 BUG by definition). `make test` runs both precision sessions —
every new test must pass or be marked fp64-only via the existing markers.

Gate to close C1: all of the above green under `make test` on the laptop, plus
adversarial review (fresh session) of cmhd.py against the C0 docs section, sign by sign.

**Landed 2026-08-30** (main `38d37c1`, opus implementer + fable adversarial review,
PASS-WITH-FIXES, all fixes applied and re-verified). Highlights: dispersion ≤2.1e-8
relative (fp64) over 3 angles × 3 branches × both γ × two c_s0/v_A, plus a non-cubic
ky≠0 case (≤2.1e-8) added after review so 2π/L factors cannot cancel; mass/mean-B
bitwise; div B ≤8.7e-17 field-scale-normalized over 100 steps; decay ICs along all
three axes after the review's mutation test exposed a k⊥-blind gate (the mutation now
fails 12 checks); schemes lsrk54/rk44 order 4.00, vs imexcb3e 2.90; full `make test`
regression-clean both precisions. Deviations recorded in the test docstrings: the
plan-regime energy gate asserts dt-INDEPENDENCE + smallness (the drift is the §3.5
truncation residual, verified ~amp^6.8 and resolution-falling; dt-order 3.64 asserted
at half amplitude), scheme cross-check and drift gates fp64-only (measured fp32 noise
floors), negative diss now rejected. The 23-FFT tally is verified by inspection only —
C2 measures. Repo-wide trap found in passing (pre-existing, untouched): a numpy-float64
`params.dt` upcasts the fp32 field graph via `lin.scaled(dt)` — applies to RMHD/GDI too.

### Phase C2 — diagnostics, validation science, performance

Files: `taranis/diagnostics/cmhd.py` + `diagnostics/__init__.py` `__all__` entry (NO
top-level name re-exports — GDI precedent; the top-level surface stays RMHD-historical),
`tests/test_cmhd_diagnostics.py`, `examples/cmhd_orszag_tang.py` (+ notebook),
docs/performance.md § addition, CLAUDE.md + docs/RUNNING_TESTS.md + examples/README.md
sweep.

- Diagnostics: `energies(state, kgrid, params)` → (kinetic, magnetic, internal) on the
  SHARED perp_reduce normalization (CLAUDE.md: new energy-like diagnostics keep the
  rfft2 ky-doubling `/ nz²(nx·ny)²` z_spectral convention or their numbers are not
  comparable); `mach_numbers` (M_s rms, M_A rms); `divB_max`; `spectra` (kinetic,
  magnetic, density) via `diagnostics.core._binned`; `energy_budget` (−dE/dt vs the
  hyperdissipation sink, closure gate in the test).
- **Orszag–Tang gate** (marked `slow`): the standard compressible OT vortex (γ = 5/3,
  the usual IC) as 2.5D (nz = 4, z-independent IC), 256², to t = π; compare E_kin(t),
  E_mag(t) traces and the density field against published spectral-code references
  (pick the reference in-phase and cite it in the test docstring). Also assert exact
  z-independence is preserved (max over kz≠0 modes at round-off) — this doubles as the
  2.5D-embedding gate.
- **Performance**: measure ms/step vs z_spectral RMHD at 128²×16 and 256²×16, fp64,
  laptop CPU, quiet machine, same-session interleaved A/B (the standing measurement
  rules); record in docs/performance.md with the transform-count context from §3.4.
  Expected ballpark 2–2.5×; whatever is measured is what gets written.
- Docs sweep: CLAUDE.md gains a short CMHD paragraph (what it is, the L split rule, the
  curl-form rule from §3.3, eqpars schema, "no forcing/particles"); examples/README.md
  ordering updated.

Gate to close C2: OT within stated tolerance of the cited reference, budget closure
green, performance section written from measurements, review pass.

**Landed 2026-08-30** (main `210ad69` + close-out fixes; opus implementer + fable
adversarial review, verdict PASS — normalization verified by independent hand
computation including the budget's D_ρ term at 3e-15, the Athena unit mapping re-derived
exactly, the t ≤ 0.12 window's Snow et al. 2021 citation verified verbatim, the
uniform-entropy polytropic≡adiabatic argument confirmed EXACT, perf reporting compliant.
Review fixes applied in the close-out commit: the C1 `_div_b` NaN-masking broadcast, two
performance.md wording nits per the quote-measured rule, the OT docstring's unsourced
1e-12 figure replaced, a RUNNING_TESTS coverage omission. Status lines below were
confirmed by the review.) Files: `taranis/diagnostics/cmhd.py`
(new), `taranis/diagnostics/__init__.py` (`__all__` entry, no name re-exports),
`tests/test_cmhd_diagnostics.py` (new), `tests/test_cmhd_orszag_tang.py` (new, `slow`+`fp64`),
`examples/cmhd_orszag_tang.py` (new), `bench/cmhd_perf.py` (new), plus the CLAUDE.md /
docs/performance.md / docs/RUNNING_TESTS.md / examples/README.md sweep. Nothing under
`taranis/physics/`, `run.py`, `timestepping.py`, `propagators.py`, `grids.py`, `comms.py`,
`config.py` or `snapshot_io.py` was touched, and no frozen reference was regenerated.

Measured numbers (Apple M1, macOS 14.6, jax 0.10.0, CPU, fp64 unless stated):

- **Performance**, `bench/cmhd_perf.py`, same-session interleaved A/B, 9 reps, lsrk54, fixed
  `dt=1e-3`, jitted `block_of_steps`, against **z_spectral RMHD at ν=η (separable backend)**:
  128²×16 → 55.79 vs 103.16 ms/step = **1.85×**; 256²×16 → 237.08 vs 481.25 ms/step =
  **2.03×**. Spreads 1.0–2.4%; a repeat session gave 1.89× / 2.03×. The plan's 2–2.5×
  estimate is NOT the result — the measurement is, and it sits slightly below the 23-vs-10
  transform ratio because RMHD's separable propagator is not free and CMHD's diagonal exp is
  cheap. `memory_analysis()` in the `bench/memory_probe.py` u convention: 48.71 u (128²×16)
  and 48.76 u (256²×16) against RMHD's 18.69/18.70 u — 2.61×, against 3.5× the state.
- **Diagnostics gates**: spectra sum to the energies at 1.9e-15/5.4e-15 relative (fp64,
  both binnings, three bin_factors, both γ); energy-budget closure 3.5e-8 (γ=1) and 2.8e-8
  (γ=5/3) relative, gated at 1e-5 and fp64-only; dropping the `D_rho` work term breaks it by
  61% (the term is 38% of the sink).
- **Orszag–Tang**: exact initial energies to 1e-13; z-independence identically 0.0 through
  all three runs; div B ≤1.3e-15 and ρ_min ≥0.176 through the shocks; E conserved to
  **5.6e-9** at 256² over t ≤ 0.12 and falling with resolution (2.5e-7 at 128², same dt);
  128²/256² traces converged to 3.7e-7 (E_kin) and 5.9e-7 (E_mag); E declines 5.9% by t=0.51.

Deviations from this §, all recorded in the files themselves:

1. **"to t = π" does not apply.** That is the 2π-box normalization; the Athena reference is
   the [0,1] box, where the equivalent is t = 1/2 (Stone's own mapping, §VIII.4). The run
   goes to t = 0.5.
2. **No published reference data exists.** Stone et al. 2008 §VIII.4 gives three OT figures,
   all snapshots/slices at t_f = 1/2, and not one time trace anywhere in the paper;
   Stone et al. 2020 (Athena++) does not contain the OT test at all; the Athena test-suite
   pages are HTTP 404 and their archived copies hold GIFs with no data files; the only
   published OT energy-vs-time curves (Orszag & Tang 1979 fig 5 — incompressible, 2π box,
   unstated factor-of-2 normalization; Dahlburg & Picone 1989 fig 5; Picone & Dahlburg 1991
   fig 8 — different M and β, marginal DTIC scans) are a different problem. So the fallback
   applies: the quantitative content is self-contained physics (exact initial energies, ideal
   invariant conservation, resolution convergence) plus a **labelled self-generated
   regression table**, and the test docstring names the concrete route to a real reference
   (build Athena++, run its shipped `athinput.orszag-tang`, whose `hst` output already emits
   E_kin(t)/E_mag(t) at dt=0.01; rescale by 1/ρ₀ = 36π/25). **Open item, carried to §11.**
3. **The gate window is stronger than "polytropic ≈ adiabatic while smooth".** The Athena IC
   has uniform ρ and uniform p, hence uniform entropy, and smooth isentropic ideal-gas flow
   obeys p = Kρ^γ with one global K — so the polytropic closure is EXACTLY the adiabatic one
   in the window, not an approximation. Window t ≤ 0.12, below every literature shock-onset
   estimate (Snow et al. 2021 §3, already in these units: "After t = 0.15, large-scale
   fast-mode shocks are generated"; Tóth 2000 §6.4 in the 2π box → t ≈ 0.159) and below where
   this code's own smoothness monitors move.
4. **The smooth-window runs use fixed dt, not the adaptive one.** With adaptive dt the two
   resolutions record at different times and comparing INTERPOLATED traces reported 3.5e-3 of
   "disagreement" that was entirely linear interpolation of a curved E_kin(t) over a 0.026
   sample interval. Fixed dt at both resolutions gives bitwise-identical sample times and the
   real answer, 4e-7.
5. **`spectra` bins to the grid corner**, not to `diagnostics.rmhd.perpspec`'s
   `min(nx,ny)//2·kunit`, and the kinetic spectrum is built from `w = √ρ·u` rather than `u`.
   Both are forced by the sum rule: `|û|²` integrates to `⟨|u|²⟩/2`, not to the kinetic
   energy, and `w` is a non-polynomial product that carries power past the dealias cut.
6. **`energy_budget` returns both signs** (`total` = the docs' sink ε, `dEdt` = −ε), because
   `diagnostics.gdi.energy_budget`'s `total` is dE/dt and silently inheriting either
   convention would be a trap.
7. **The energy-budget and OT gates are fp64-only.** dE over two steps is ~4e-8 on an E of
   order 1, below the fp32 noise floor of the difference — the same reason the GDI closure
   gates are fp64-gated. `test_cmhd_diagnostics.py`'s other six gates run at both precisions.

One pre-existing C1 observation, reported and NOT fixed: `divB_max`'s C1 formulation
(`tests/test_cmhd_conservation.py::_div_b`) broadcasts `kmag` with `+ 0.0*d`, which turns a
NaN field into an empty mask and a confusing "zero-size reduction" error instead of a NaN.
The diagnostics copy drops the trick (the k arrays broadcast on their own); the C1 test was
left alone.

### Phase C3 — expanding box (EBM)

Two sub-phases, strictly ordered:

**C3a (derivation, docs only)**: EBM equations for THIS variable set in comoving
coordinates. **Source of truth: Squire et al. 2020's expanding-box formulation** (§10
answer; their compressible EBM runs are isothermal, matching the γ default — C3a pins
the exact paper/equation numbers and cross-checks against Grappin & Velli 1996 /
Dong, Verdini & Grappin 2014 as secondary derivation checks, flagging any convention
difference rather than silently mixing them). a(t) = 1 + ȧt, **radial axis = x**
(confirmed), transverse (y,z) expanding. Deliverables: docs/numerics.md § "Expanding box" with every
term's power of a and ȧ/a tabulated per field, the anisotropic-metric factors on each
derivative, the WKB/analytic gate predictions (uniform-state decay laws: ρ ∝ a⁻²,
B_x ∝ a⁻², B_⊥ ∝ a⁻¹, polytropic T scaling; WKB Alfvén amplitude ∝ a^(−1/2)), and the
comoving-vs-physical dissipation decision (default: comoving-k dissipation, keeping L
static — the physical-ν(t) alternative would force per-block L rebuilds or an explicit
dissipation term; documented, not built). Same adversarial-review gate as C0.

**C3a drafted 2026-08-30** (by the session directly, like C0; pending its adversarial
review): docs/numerics.md § "Expanding box" now carries the derivation from Squire et
al. 2020 eqs (1)–(3) — a(t) = 1 + ȧt, ∇̃ = (∂_x, a⁻¹∂_y, a⁻¹∂_z), T = diag(0,1,1),
Λ = diag(2,1,1) — with two design results that SUPERSEDE the C3b sketch below where
they conflict: (i) **EBM is isothermal-only** (γ = 1 enforced with expansion on; Squire's
closure, and a γ > 1 polytrope would need its own expansion terms), with cooling
`c_s²(t) = c_s0²·a^(−q)`, default q = 4/3; (ii) **rescaled evolution variables**
ρ′ = a²ρ, B′ = (a²B_x, aB_y, aB_z) kill the ρ and B expansion terms identically and —
via the verified identity A·(∇̃×E) = ∇×E′ with E′ = (E_x, aE_y, aE_z) — keep induction
a pure STATIC-k curl, so the C1 round-off div-B property and the bitwise ρ′/B′ k=0
gates survive expansion unchanged (physical div = a⁻²·K·B̂′; raw backgrounds track
ρ ∝ a⁻², B_x ∝ a⁻², B_⊥ ∝ a⁻¹ exactly). Only −(ȧ/a)T·u survives as an additive term.
Transform tally stays 23 (rescalings are elementwise).

**C3a reviewed 2026-08-30, PASS after two one-line fixes** (fable adversarial review:
Squire eqs (1)–(3) transcription and Dong/Verdini/Grappin 2014 cross-check verified —
identical conventions, DVG14's physical-ν and adiabatic-pressure divergences being
exactly the two recorded taranis choices; the rescaling identity confirmed in all five
sub-items; every exponent confirmed, WKB δu ∝ a^(−1/2) re-derived from wave action with
k_x static and ω ∝ a⁻¹). The two blocking fixes, applied: a sign flip in the
anisotropic rotational-identity transcription, and the u_x(0)-bitwise claim qualified
to the uniform-state gate IC (in general the mean stresses source u(0)). Also folded
in: the isothermal-only rationale corrected (barotropic γ > 1 is exactly
self-consistent in EBM — D_t(p/ρ^γ) = 0, expansion sources cancel — so the restriction
is a scope pin to Squire's closure, deferred not blocked), the Γ_sim quote completed,
a(t) ≤ 0 rejection added.

**C3b (implementation)**: per the docs section, gated on `"expansion" in params.eqpars`
as trace-time static python: the metric/rescaling factors enter grad and the term funcs
(k̃ = (k_x, k_y/a, k_z/a), the elementwise unscalings ρ̂ = ρ̂′/a², B̂ = A⁻¹B̂′, the E′
scaling, c_s²(t)), plus ONE new additive `Term` for −(ȧ/a)T·u with
`active=lambda p: "expansion" in p.eqpars`; expansion OFF must leave the graph BITWISE
identical to the pre-C3 tree (every factor a literal 1 that never enters the trace —
gate, and the standing bitwise-gates-are-evidence rule applies: if it drifts, find out
why, never widen). eqpars gains optional `expansion = {"adot": float > 0, "cs_q":
float ≥ 0, default 4/3}`, rejected unless γ = 1; note Γ_sim = ȧ·L_x/v_A is the
run-design dial — Γ_sim = (ȧ/a)(L_x/v_A) at t = 0, decaying during a run; Squire et
al.'s Athena++ runs use 0.2–0.5 and their Snoopy runs 2. Time enters through `grads.t` (§3.6, cast
to ftype before touching fields — CMHDGrads.t is float64): verified 2026-08-29 that
every stepper sets stage-correct times (`state._replace(t=state.t + c_k·dt)`,
timestepping.py throughout), so a(t) is evaluated at the right stage abscissae with no
order loss. CFL: physical spacings (d_x, a·d_y, a·d_z) with speeds from the UNPRIMED
fields and c_s(t). Snapshots store the PRIMED state and t — restarts reconstruct a(t)
from t; params.save records `expansion`, and the differing-record check stops a
cross-mode restart exactly as for z_spectral; validation rejects a(t) ≤ 0. Gates:
bitwise ρ′/B′(k=0) and, ON A UNIFORM STATE (u(0) is stress-sourced in general — docs),
the u_x(0) bitwise / u_⊥(0) ∝ a⁻¹ exact-ODE pair (the latter doubles as the stage-time
regression gate — its convergence order collapses if a stepper stops setting stage
times); K·B̂′ round-off; the WKB δu ∝ a^(−1/2) exponent over a decade of a (tolerance
budgeted from O(ȧ/(aω)) corrections); expansion-off bitwise; dispersion gates unchanged
at ȧ = 0. Diagnostics: energies/spectra are documented as COMOVING-primed quantities in
C3b (physical conversions are a-scalings; a `diagnostics.cmhd` helper may unscale, but
the sidecar convention is decided there and recorded).

**C3b LANDED 2026-08-30** (main `f7271ae` + close-out; opus implementer in a worktree off
main `8e2d229`; adversarial review round 1 FAIL on gate coverage — three blind mutations —
round 2 **PASS** after the fixes: every mutation M1–M5 + M3b independently re-run and
caught, the raw-frame differential gate's independence audited with no cancellation
channel, the oblique-WKB substitution adjudicated acceptable, and the review's own
correction to the gate's agreement mechanism — the 2/3 rule, not band-limiting — folded
into the docstring and this note. The numbers below were review-confirmed in round 2.) Files:
`taranis/physics/cmhd.py` (277 → 467 lines), `tests/test_cmhd_expansion.py` (new, 16 tests
/ 918 lines), plus the CLAUDE.md, docs/RUNNING_TESTS.md and this-§ sweep. **Nothing else
was touched** — no `physics/__init__.py`, `run.py`, `timestepping.py`, `propagators.py`,
`grids.py`, `comms.py`, `config.py`, `snapshot_io.py` or `diagnostics/`, and no frozen
reference was regenerated.

Implementation, per the C3a docs section: `eqpars["expansion"] = {"adot": > 0, "cs_q":
>= 0, default 4/3}` behind the single trace-time switch `cmhd._expansion(params)` (None
off); `grad` unscales `B^ = A^-1 B'^` in k-space and forms `omega`/`j` with
`k~ = (kx, ky/a, kz/a)` through the same Nyquist-zeroed `_kz_deriv`; `NonlinearTerm` runs
continuity on the primed flux `rho' u`, `h = cs^2(t) ln rho'` (the uniform `-2 cs^2 ln a`
dropped BY the gradient, not added back), the Lorentz force on the UNPRIMED `rho`, the
rotational form with the C3a-fixed sign, and induction as the STATIC-k curl of
`E' = (E_x, aE_y, aE_z)`; `set_timestep` uses `(dx, a*dy, a*dz)` with UNPRIMED speeds and
`cs(t)`; `a(t)` is built from `grads.t` and cast to `_precision.ftype` before it touches a
field. **`linear_matrix` has ZERO code change** — the dissipation already IS the static
comoving-k diagonal acting on the primed state, the recorded truncation choice; confirmed
by inspection and by the expansion-off bitwise check below. Transform tally stays 23.

Measured (Apple M1, jax 0.10.0, CPU; both precisions unless stated):

- **Expansion-off is bitwise the pre-C3b tree.** One-time out-of-tree check (a scratchpad
  script, deliberately NOT `tests/data`): three configs — isothermal ideal, isothermal +
  scalar diss, polytropic γ=5/3 + length-3 diss under rk44 — recorded on the pristine
  worktree and re-run after the edits. All three matched in `fields` AND `t` bitwise (max
  |Δ| exactly 0.0) AND in the optimized-HLO opcode histogram of the jitted
  `block_of_steps`, at BOTH fp64 and fp32. The STANDING gate in the test file is the
  durable half: the expansion-off RHS is bitwise independent of `state.t` (nothing else in
  CMHD reads `t`, so no a-factor can be in that graph), with the expansion-on twin
  differing by 4.8e-1 of the RHS scale as the discriminator; plus an analytic Beltrami
  `exp(-nu k^2 t)` decay at 2.3e-15 (fp64) / 2.8e-6 (fp32) and a two-compilation bitwise
  check.
- **k=0**: `rho'` and every `B'` component bitwise over 100 uniform-state steps AND over 50
  turbulent steps (with `u(k=0)` moving, as the discriminator); `u_x(k=0)` bitwise on the
  uniform state; `u_perp(k=0)` tracks `a^-1` at 1.3e-11 (fp64) / 7.2e-7 (fp32), converging
  at **order 4.007** in dt (lsrk54, nominal 4) — the stage-time gate, fp64-only because the
  fp32 errors sit on a ~5e-7 storage floor.
- **Raw backgrounds** through `grad`'s own unscaling: `rho ~ a^-2`, `B_x ~ a^-2`,
  `B_perp ~ a^-1` at 1.1e-16…4.4e-16 (fp64), ≤3.6e-7 (fp32).
- **div B under expansion**: `max_k |K.B'^|/(|K| max|B'^|)` ≤ 6.2e-17 over 100 steps
  (fp64), growth ×3.80 from 10 to 100 steps (√10 = 3.2 is a random walk, 10 would be a
  source); ≤2.9e-8 / ×1.04 at fp32.
- **WKB** (run at BOTH transverse polarizations since review round 1 — identical physics,
  but E′ rotates with the polarization, so the y-polarized wave leaves E′_y multiplying an
  exact zero): Alfvén wave along x, `k_x = 1`, `v_A0 = B0 = 1`, `adot = 0.02`, a spanning
  1 → 4 (a factor 4, not a decade — a decade costs ~4× the steps for no extra
  discrimination; the run is already 3000 steps). Because `omega ~ a^-1` with `k_x`
  static, `eps_WKB = adot/(a*omega) = adot/(k_x v_A0) = 0.02` is CONSTANT over the run, so
  the exponent budget is `eps_WKB/Δln a = 0.02/ln 4 = 0.0144`. Measured `|u_y|` exponent
  **−0.49816** and `|B'_y|` **−0.50181** against the WKB −1/2, i.e. |Δ| = 1.8e-3, 8×
  inside budget and IDENTICAL at fp32. Discriminator: halving `adot` moves |Δ| to 5.4e-4
  (×3.4), confirming the residual is first order in `adot/(a*omega)`.
- **Plumbing**: the nested `expansion` dict round-trips through `params.json` unchanged
  (`config._lists_to_tuples` recurses into dicts); a differing `adot` — and dropping
  `expansion` entirely — is a hard save error, which is what stops a cross-mode restart;
  a mid-expansion restart at a = 1.05 is bitwise in fields and `t`; `gamma != 1`,
  `adot <= 0`, `cs_q < 0`, a missing `adot`, an unknown sub-key and a non-dict `expansion`
  all raise. `cs_q = 0` vs 4/3 changes the run by 6.9e-3 relative, so the cooling law
  demonstrably reaches the RHS.
- **Raw-frame RHS cross-check** (added in review round 1, below): one RHS evaluation at
  a = 1.7000 on a band-limited random state, converted to raw-frame time derivatives and
  compared against an independent ADVECTIVE transcription of eqs (1)–(3). Per-row relative
  residuals **3.2e-17 … 1.4e-15** (rho 3.2e-17; u 1.4e-15/2.2e-16/1.4e-15; B
  4.0e-16/3.6e-16/1.2e-16), gated at 1e-11, fp64-only.
- **CFL under expansion**: on IDENTICAL fields in a transverse-limited box
  (ny = nz = 2·nx) at adot = 0.4 (a = 1.8 at t = 2), dt goes **0.0877 → 0.1579 (×1.8011)
  at cs_q = 0** — where the physical spacings are almost the whole effect — and
  0.0877 → 0.1684 (×1.9215) at cs_q = 4/3 with the cooling on top; the expansion-absent
  control returns a bitwise identical dt at both times. An adaptive-dt expansion run is
  finite through a = 5.95.
- **Dispersion unchanged with expansion absent**: the C1 exactly-parallel Alfvén config
  (8³, 2π box, B0 = 1, cs0/v_A = 0.5) measures ω = 0.999999979 against the analytic 1, rel
  2.1e-8, amplitude drift 6.1e-9.
- Regression: `test_cmhd_linear.py` + `test_cmhd_conservation.py` +
  `test_cmhd_diagnostics.py` 24 passed at fp64; full `make test` clean at both precisions.

Deviations from the C3b paragraph above, all recorded in the files themselves:

1. **The `-(adot/a)T·u` term is folded into `NonlinearTerm`'s assembled RHS, not
   registered as its own `Term`.** `physics/__init__.py` is not a C3b file (the phase's
   file list forbids touching the registry), and a second registry entry is the only way to
   add a `Term`. The predicate that would have been its `active=` is the same trace-time
   `_expansion(params) is None` switch, so the off-graph guarantee is identical; the module
   comment says so.
2. **`CMHDGrads` gains a trailing `rho_p` field** (the primed density, `None` when
   expansion is off). `grads.rho` stays the UNPRIMED density, as the phase spec requires,
   so `set_timestep` and the Lorentz force need no unscaling — but `h` is built on `rho'`
   as the docs prescribe, and that needs `rho'` carried. `rho'` is therefore the
   transformed quantity and `rho = rho' * a^-2` is done elementwise in REAL space rather
   than in k-space (the docs say k-space); the two differ only by rounding on an
   elementwise scaling and the tally is unchanged at 13 inverse either way. Using
   `ln rho'` rather than `ln rho` is not cosmetic: they differ by the uniform `-2 ln a`,
   which the gradient kills, but `ln rho ~ -2 ln a` grows with the run and costs
   conditioning in the gradient while `ln rho'` stays O(1).
3. **`_enthalpy` gains an optional 4th argument `cs2`** and `_eqpars` keeps its 4-tuple
   return, because `taranis/diagnostics/cmhd.py` (not a C3b file) imports both and unpacks
   the 4-tuple. The EBM configuration is read through the separate `cmhd._expansion`
   instead. `_eqpars` calls `_expansion` purely for validation, so every entry point that
   reads eqpars — diagnostics included — rejects a malformed expansion block.
4. **`_curl_real`'s signature changed** from `(vk, kgrid, kz, params)` to
   `(vk, kx, ky, kz, params)`: the physical curls need `k~` while the induction curl needs
   the static `K`. Module-private, no external callers.
5. **`a(t) <= 0` is guarded only where it is reachable.** `adot > 0` is validated, which
   makes `a >= 1` for every `t >= 0` a run can reach. A DOCTORED restart at `t < -1/adot`
   cannot be validated anywhere — `t` is a traced value at every use site — so it carries a
   comment in `_expansion` and nothing more, as the phase spec allows.
6. **The WKB gate uses a factor 4 in a, not a decade** (per the measurement note above),
   and is NOT marked `slow`: its two runs together take ~5 s.
7. **Diagnostics are left EBM-unaware — this is the C3b DECISION, with the code change
   deferred.** With expansion on, `diagnostics.cmhd`'s `energies` / `spectra` /
   `mach_numbers` / `divB_max` are COMOVING-PRIMED quantities: they read `state.fields`,
   which under EBM holds `rho'`, `B'`, `u`. The physical conversions are pure a-scalings
   (`rho = a^-2 rho'`, `B = A^-1 B'`, `div_phys = a^-2 K.B'^`), so nothing there is wrong,
   only differently normalized. A future unscaling helper should take `a` (or `params` +
   `t`) EXPLICITLY rather than reading `params.eqpars["expansion"]` behind the caller's
   back, so the two conventions cannot be silently mixed in one plot. Carried to §11.

**C3b review round 1, 2026-08-30 — physics PASS, gate coverage FAIL; fixes applied,
pending re-review.** The fresh-session adversarial review verified the physics
transcription, the off-path guarantee, the WKB budget arithmetic and all seven deviations
above as correct, but **mutation-tested the gate set and found three blind spots**. The
lesson, worth carrying into any future metric-factor work: the *structural* gates that make
this equation set attractive — bitwise k=0 modes, the div-B random walk, the uniform-state
ODE — are exactly the gates that CANNOT see a wrong metric factor. A k=0 mode has no
metric, `K·B̂′` survives ANY diagonal rescaling of E′, and a uniform state has zero curl and
zero gradient. Structural invariance is not physics coverage.

The blind spots (each mutation applied to cmhd.py alone, the original 15 tests run):

| # | mutation | before | after the fixes |
|---|---|---|---|
| M1 | drop the `a` on E′_y in the induction scaling | **15 passed — BLIND** | caught by the raw-frame cross-check (`d_t B_x` rel 3.18e-1, `d_t B_z` 9.71e-2) AND by the new z-polarized WKB twin (exponent −0.248 vs −1/2) |
| M3 | `ky` instead of `ky/a` in `grad`'s physical curls | **15 passed — BLIND** | caught by the cross-check (`d_t u_y` rel 2.27e-1, `u_x` 2.19e-2, `u_z` 9.55e-2) |
| M3b | the same on the term func's `k̃` (continuity + pressure gradient) — the reviewer predicted this would be equally blind | (predicted blind) | caught by the cross-check (`d_t rho` rel 5.25e-2, `d_t u_y` 3.42e-1) |
| M5 | static spacings in `set_timestep` | **15 passed — BLIND** (the 1.05× threshold was met by cooling alone: the mutant measures ×1.0675) | caught by both CFL cells (cs_q = 0: ×1.0006 vs the real ×1.8011; cs_q = 4/3: ×1.0675 vs ×1.9215) |
| M2 | `T·u` sign flip | caught by 4 tests | unchanged |
| M4 | cooling law dropped | caught by 1 test | unchanged |

Fixes, all additive test work plus one threshold — `taranis/physics/cmhd.py` was NOT
touched (restored byte-identical, sha256 verified, after every mutation):

1. **`test_raw_frame_rhs_matches_the_advective_reference` (new, fp64)** — the primary gate,
   and the one that reads every metric factor at once. One RHS evaluation at a = 1.7 is
   converted to RAW-frame time derivatives (`d_t ρ = (d_t ρ′)/a² − 2(ȧ/a)ρ`, and
   `B_i = B′_i/A_i` with `A = diag(a²,a,a)` whose logarithmic derivative is exactly
   `Λ_i·(ȧ/a)` — the ȧ terms reappear, which is the rescaling working in reverse) and
   compared against an independent transcription of eqs (1)–(3) in **advective** form, with
   its own k̃ arrays, its own a-powers and its own Nyquist zeroing. Two discretization
   points, both recorded in the test docstring: **the reference must carry the same dealias
   mask** (`1/ρ` and `ln ρ` put power past the 2/3 cut; the mask is part of the
   discretization under test, and the ȧ terms sit outside it in both — which cannot matter
   anyway, since `initialize` leaves the state mask-supported, asserted); and **the two
   forms agree at round-off because of the 2/3 rule, not band-limiting** (round-2 review
   correction: the compared state fills the mask band after 100 steps and the gate still
   reads 1.4e-15 — the retained modes of quadratic products are exact for any
   mask-supported state, and the non-polynomial pieces are common-mode; the |n| ≤ 2 IC is
   a convenience, not load-bearing).
2. **The WKB gate now runs BOTH transverse polarizations.** y and z are the two expanding
   directions and every EBM diagonal treats them identically, so the prediction is the same
   −1/2 — but `u × B` is along −ẑ for a y-polarized wave and along +ŷ for a z-polarized
   one, so only the z-polarized twin makes E′_y load-bearing in a measured frequency. Both
   measure −0.49816 / −0.50181. Cost: +2 s. (The reviewer's suggested oblique-k transverse
   twin was NOT built: its WKB amplitude law is not derivable from the docs' framework
   without new work, and asserting an underived exponent is worse than not gating it. The
   polarization rotation gets the same coverage with zero derivation risk.)
3. **The CFL gate gained a cs_q = 0 cell**, where the physical spacings are almost the
   whole relaxation, at a ×1.4 threshold; the cs_q = 4/3 cell's threshold went 1.05 → 1.5,
   above the cooling-only ×1.0675. Note the cs_q = 0 mutant is ×1.0006, **not** exactly 1:
   `grad` still unscales the fields, so `v_A² = |B|²/ρ` carries its own a-dependence. That
   figure is stated in the test rather than the tempting "exactly ×1".
4. CLAUDE.md now qualifies the three-config bitwise+HLO verification as the implementer's
   one-time out-of-tree check rather than stating it as a reproducible fact.

## 6. What is verified NOT to need changes (checked against the tree, 2026-08-29)

- `run.py`: fully registry-driven (`recipe.*` at :85, :105, :142, :279); `initialize`
  equation-generic; forcing advance keyed off `forcing_scale_func`/`params.forcing`.
- `timestepping.py`, `propagators.py`: no new backends, no new schemes (§3.1).
- `grids.py`: `dealias_mask` already 3-D under z_spectral; `kz` already built;
  `_attach_linear_operator` generic.
- `snapshot_io.py`, `comms.py` (serial), `config.py` structurally (§4 last bullet).
- Frozen references (gate 6, refactor reference): untouched by construction.

## 7. Risks / traps for implementers

- **kz Nyquist**: every i·kz anywhere in cmhd.py goes through the one Nyquist-zeroed
  helper (§4). A raw `kgrid.kz` derivative breaks reality and the div-B gate will catch
  it confusingly late — this is the most likely silent-wrongness bug in the phase.
- **`set_timestep` sees grads only** — no state, no t except what CMHDGrads carries.
  Don't reach for `state` in it; the signature is fixed by the registry contract.
- **`params.dz` DOES exist** under z_spectral (config.py sets dz = Lz/nz for every
  dims==3 run; rev 2 of this plan claimed otherwise — the C0 review corrected it). The
  guarded-attribute rule covers dims==2, which CMHD rejects anyway.
- **Dict-valued kwargs and `ctx()`**: eqpars dicts make `ctx()` unusable for CMHD test
  configs — `fresh_params` everywhere.
- **Never assert energy/cross-helicity at round-off** (§3.5) — order + smallness.
- **`state._replace` only**, never positional `SimulationState` (forcing fields must
  thread through even though CMHD never uses them).
- **Non-integer γ−1 needs ρ > 0**: `rho**(gamma-1)` on a negative ρ is NaN at fp — this
  is the intended loud failure, don't "fix" it with abs() or clipping.

## 8. Deferred, explicitly out of scope

- Forcing (compressible O-U momentum/Elsasser forcing — needs its own normalization
  derivation; the hooks exist).
- Particles in CMHD fields; MPI/z-decomposition; FD-z CMHD; `dims==2` CMHD.
- ln ρ variable (positivity at high Mach, §3.2); semi-implicit acoustics for M_s ≪ 1;
  a dense-nfields exponential backend (wave-IF).
- Shock capturing of any kind.

## 9. Execution rules digest (per-phase, for the implementer agents)

- Worktree agents: verify the worktree base is current main; subprocess/script-mode
  tests import the EDITABLE install — set `PYTHONPATH=<worktree>` or the test runs
  main's taranis. No `git stash` in the shared tree, ever.
- New test modules: bootstrap-before-import-taranis header, `script_main` footer,
  markers per conftest (`fp32`/`fp64`/`slow`); must pass BOTH `make test` precision
  sessions or carry the precision marker.
- Never cache a SimulationState across `simulate` calls (donation); never mutate `ctx()`
  results; rebuild kgrid for any new/changed Parameters.
- Frozen bitwise references are evidence — a red frozen gate means stop and report, never
  regenerate, never restructure fp op order to satisfy a comparator.
- Performance numbers: same-session interleaved A/B on a quiet machine, or they don't go
  in docs. Quote measured, not chained; ratios name their reference.
- Every phase ends with: docs updated in the same commit series, this plan's § updated
  with a dated landing note, adversarial review by a fresh session before the phase is
  called done.

## 10. Open questions — answered 2026-08-29 unless noted

1. ~~γ default~~ **Answered: isothermal, γ = 1 default** (folded into §1/§4; γ > 1 kept
   and gated).
2. ~~OT reference~~ **Answered: an Athena one if available** — target the Athena method
   paper's Orszag–Tang test (Stone et al. 2008, ApJS 178, 137; the C2 implementer
   verifies the exact figure/section and cites it, falling back to the Athena++ method
   paper, Stone et al. 2020, if the data there is easier to digitize). The reference is
   ADIABATIC γ = 5/3; our polytropic γ = 5/3 run is isentropic, identical to adiabatic
   only while the flow is smooth — so the quantitative gate window is the pre-shock
   smooth phase (implementer determines the shock-formation time from the reference and
   states the window in the test), with post-shock agreement checked qualitatively only.
   A grid-code reference is fine for this purpose; tolerance set accordingly.
3. ~~EBM formulation~~ **Answered: Squire et al. 2020, radial axis = x** (folded into
   §5 C3a).
4. ~~First-target regime~~ **Answered: β = 0.3, δB/B₀ = 1.** Implications, recorded:
   isothermal β = 2c_s0²/v_A² so c_s0 = √(β/2)·v_A ≈ 0.39 B₀ (code units, ρ₀ = 1).
   The explicit-acoustics dt penalty is a NON-issue here — the fast speed is
   v_A-dominated, so the CFL is Alfvénic and CMHD's dt is comparable to z_spectral
   RMHD's at the same B₀. The real flag is the other direction: trans-Alfvénic δu with
   c_s ≈ 0.4 v_A means M_s up to ~2.6·(δu/v_A) — sonic-transonic, so steepening is
   expected and the hyperdissipation must be sized to absorb it (the OT gate is directly
   relevant experience); if production ICs turn out strongly supersonic, revisit §3.2's
   ρ-positivity note. C2 measures perf at 128²×16 and 256²×16 as written.

## 11. Open items (carried out of C2, 2026-08-30)

1. **A real external Orszag–Tang reference.** The C2 search established that no published,
   digitizable E_kin(t)/E_mag(t) exists for this problem in any normalization close to ours
   (details and citations in §5's C2 note and in `tests/test_cmhd_orszag_tang.py`). The route
   that works, and is not in this repository: build Athena++, run its shipped
   `inputs/mhd/athinput.orszag-tang` (which already requests `file_type = hst` at dt = 0.01;
   `src/outputs/history.cpp` writes volume-integrated `1-KE, 2-KE, 3-KE, 1-ME, 2-ME, 3-ME`
   per row), and compare to our t ≤ 0.12 window after multiplying its energies by
   1/ρ₀ = 36π/25 = 4.5238934. Its pgen uses the box [−0.5,0.5]² and
   `Az = B0/(4π)(cos 4πx − 2cos 2πy)`, i.e. the Stone-2008 field translated by (½,½) — same
   physics and same energies, opposite signs; do not chase that as a discrepancy. Until then
   the stored table in that test is a regression gate and is labelled as one.
2. **A density image comparison against Stone et al. 2008 figure 22** (contours at
   t_f = 1/2) and its figure 23 pressure slices. Post-shock, so qualitative only, and it needs
   a human eye rather than an assertion — the example script already produces the ρ image.
3. **CMHD forcing** (§8) is the natural next phase after C3; it needs its own power
   normalization derivation, and until it exists `diagnostics.cmhd`'s shared-convention
   energies are the thing a forcing power will have to be comparable to.
4. **EBM-aware diagnostics** (carried out of C3b, decision item 7 in §5). `diagnostics.cmhd`
   reports COMOVING-PRIMED quantities when expansion is on. If a production campaign wants
   physical energies/spectra, add an unscaling helper that takes `a` explicitly; do not make
   the existing functions branch on `params.eqpars["expansion"]` silently.
5. **A barotropic `gamma > 1` under expansion** is exactly self-consistent (EBM preserves
   `D_t(p/rho^gamma) = 0`) and would drop in as
   `h = cs0^2 a^(-2(gamma-1)) rho'^(gamma-1)/(gamma-1)`, with γ = 5/3 reproducing the
   a^(−4/3) cooling automatically. C3b enforces `gamma == 1` as a scope pin to Squire et
   al.'s closure — deferred, not blocked.
6. **`spectra`'s corner binning is honest but wasteful**: the bins past the dealias cut carry
   only the non-polynomial residual of `√ρ·u`. If a production campaign wants a clean inertial
   range it should slice the returned arrays, not narrow `kmax` — narrowing it silently breaks
   the sum rule the C2 gate rests on.
