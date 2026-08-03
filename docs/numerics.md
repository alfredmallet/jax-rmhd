# Numerics notes

The *reasoning* behind the numerical choices in `jax_rmhd`. CLAUDE.md states the rules;
this file says why they are what they are, so neither has to do both. Written for someone
reading or extending the solver, not for someone running it (for that see
`docs/performance.md`).

## Fourier conventions

`grids.fft/ifft` are unnormalized `rfft2`, so an O(1) real field has O(nx·ny)
coefficients. This matters whenever an amplitude is synthesized directly in k-space
rather than transformed into it: the stochastic forcing multiplies its noise by
`nx·ny` (`shared_physics.ou_update`) precisely so that the resulting real-space forcing
is resolution-independent.

`rfft2` keeps `kx` two-sided and `ky` non-negative. Reality of the underlying field is
therefore *not* a per-coefficient property — it is a constraint **between** `(kx, ky)` and
`(−kx, ky)`, and it only bites on the two rows where both are represented: `ky = 0` and
`ky = ky_Nyquist`. Anything that writes k-space directly has to impose it explicitly.
Anything that gets there through `ifft`/`fft` gets it for free.

### One normalization for every energy-like quantity

All the `perp_*` reductions in `shared_physics`, and both spectra plus `energy` in
`diagnostics`, share one convention: multiply by the `ky`-doubling factor `kgrid.yfac`
(2 everywhere except the `ky = 0` and Nyquist rows, which are not duplicated), sum, then
divide by `nz·(nx·ny)²`. The consequence relied on in `tests/test_diagnostics.py` is that
the integral of either spectrum equals `diagnostics.energy` exactly, to round-off. A new
diagnostic that invents its own normalization silently stops being comparable to the
forcing power and the dissipation rate.

## Dealiasing

The 2/3 rule, applied as an ellipse in *mode-index* space with semi-axes `nx/3`, `ny/3`
(`grids.dealias_mask`). Box lengths cancel out of the condition entirely, so the mask
depends only on resolution.

The elliptical region is a strict subset of the box rule `|n_x| < nx/3 and |n_y| < ny/3`,
and any subset of a dealiased set is dealiased, so this is valid — slightly conservative
at the corners, which is the price of a mask that is isotropic in index space.

`run.initialize` applies the mask to the initial condition as well as the evolution
applying it to the nonlinear term. Without that, energy in an IC beyond the cutoff would
never be removed (the linear terms and the integrating factor do not truncate) and would
alias for the whole run.

## z discretization

Fourth-order centered differences for `∂/∂z`, plus a five-point `∂⁴/∂z⁴` hyperdissipation
term, both in `shared_physics.z_derivatives`. Together they need a two-plane halo, which
is what fixes `width=2` throughout `comms.halo_exchange` and `rmhd.halo_start`.

**The semi-discrete dispersion relation.** A fourth-order centered first derivative does
not reproduce `ik` exactly; for a mode `e^{ikz}` it gives `i·k_eff` with

```
k_eff = (8 sin(k·dz) − sin(2·k·dz)) / (6·dz)
```

This is why an Alfvén wave in the discrete system travels at `k_eff`, not 1. It is the
difference between a test that measures *spatial* accuracy and one that measures
*temporal* accuracy: `tests/test_advection.py` compares against `cos(z + t)` and sees the
O(dz⁴) dispersion error, while `tests/test_time_order.py` and
`tests/test_scheme_equivalence.py` compare against `cos(z + k_eff·t)`, which the
semi-discrete system solves *exactly*, leaving only the time-integration error.

**Hyperdissipation stability.** The coefficient is `z_diss·(dz/2)⁴`, and the five-point
`∂⁴/∂z⁴` stencil has maximum eigenvalue `16/dz⁴`, so the damping rate at the grid scale is
`z_diss·(dz/2)⁴·16/dz⁴ = z_diss` — independent of resolution. That is why
`rmhd.set_timestep` can bound the timestep with a bare `max(..., z_diss)` term.

## Spectral z (`params.z_spectral`)

An optional single-process mode (`dims=3`, `size==1`, never `comm_backend="jax"`) in which the
z axis is Fourier rather than finite-difference. `grids.fft/ifft` become `rfftn`/`irfftn` over
`(z, x, y)`, so `state.fields` keeps its `(nfields, nz, nkx, nky)` shape and axis 1 simply means
`kz`. Everything downstream follows from three consequences:

- **The parallel operator is k-local**, so it leaves the RHS entirely and joins `L`:
  `∂ₜφ = i·kz·ψ`, `∂ₜψ = i·kz·φ` — the exact spectral form of `rmhd.LinearTerm`'s `∂ψ/∂z` in the
  (vorticity/−k⊥²) equation and `∂φ/∂z` in the ψ equation. The 2×2 putzer2 backend gives
  eigenvalues `±i·kz`, i.e. `z± = φ±ψ` propagating at unit speed, *exactly*: no wave CFL and no
  dispersion error, which is why `rmhd.set_timestep` drops both its `1/dz` and `z_diss` entries
  there and `halo_start`/`LinearTerm` are skipped (plain `if` on the static parameter).
  `eqpars['z_diss_k']` adds an optional `−z_diss_k·kz⁴` diagonal; it is a truncation choice, not
  a stability need, and is rejected outside this mode.
- **`i·kz` needs the same Nyquist fix as gdi's `ky_deriv`** (`rmhd._kz_deriv`). The reality
  constraint of the layout is now `F(−kx, −kz, ky) = conj(F(kx, kz, ky))` on the `ky = 0` and
  Nyquist rows — `propagators._check_hermitian_compatible` mirrors both axes — and at the
  self-conjugate `kz` Nyquist plane a bare `i·kz` violates it. The 2/3 `kz` cut removes that
  plane from every nonlinear and IC path anyway, so zeroing it is inconsequential.
- **Parseval, once, everywhere.** The z-FFT is unnormalized like the perpendicular one, so the
  z-average `(1/nz)Σ_z` becomes `(1/nz²)Σ_kz`: `shared_physics.perp_reduce` divides by `nz²`
  instead of `nz` in this mode and every energy-like quantity built on it — `diagnostics.energy`,
  `perpspec`, the forcing power — stays numerically identical to the real-z computation of the
  same physical state (`tests/test_z_spectral.py::test_forcing_power_parseval_matches_real_z`
  measures ~1e-16 relative). `parspec` becomes a plain sum over `kz` (its own `1/(nx·ny·nz)²`
  is unchanged: it always worked in the transformed variable).

The forcing's `A·cos(2πz/Lz) + B·sin(2πz/Lz)` envelope is scattered exactly rather than
evaluated: the unnormalized z-FFT of that envelope is `(A − iB)·nz/2` at `kz` index `+1` and
`(A + iB)·nz/2` at index `−1` (`nz−1`), and nothing anywhere else. Reality comes for free —
`A`, `B` are already kx-Hermitian on the `ky = 0`/Nyquist rows, and mirroring `kz` there swaps
the two planes while conjugating. The scatter uses `.at[].add` so the degenerate `nz = 2` case,
where the two planes coincide, still sums correctly.

Dealiasing gains a plain 2/3 cut in `kz` multiplied onto the perpendicular ellipse
(`grids.dealias_mask`): the brackets are pointwise in z, so products alias in `kz` too. The
product mask is a subset of the full 3D box rule, hence valid, and both the `* kgrid.dealias` in
the nonlinear terms and `run.initialize`'s IC masking pick it up with no physics-code change.

Snapshots are **not** cross-mode compatible (identical shapes, different meaning): `z_spectral`
is recorded in `params.json` and `params.save`'s differing-record check is the only guard.

## Time integration

Dissipation is not an RHS term. It is applied as an integrating factor: the perpendicular
hyperdissipation operator `L = −diss·k⊥^(2·hyper)` is diagonal in k-space, so
`exp(L·dt)` can be applied exactly rather than integrated.

For the low-storage schemes each stage advances the fields by `exp(L·dt·γᵢ)` and the
accumulator `delta` by the same factor. Since `Σγᵢ = 1` for both tables, a run with no
nonlinear term reproduces `exp(L·dt)` per step exactly. Dropping the factor from
`delta` is a silent bug on any dissipative problem and invisible on a non-dissipative one
— `tests/test_time_order.py` exists to catch exactly that, and its header records the
mutation test that proves it does.

**Scan and unrolled stage loops are not bitwise identical.** `params.lsrk_scan` selects a
`lax.scan` stage loop or a Python-unrolled one. They are algebraically the same, but XLA
fuses the two graph shapes differently, so at fp64 they diverge at round-off (measured
max|diff| ~2e-15 for lsrk33 and ~7e-15 for lsrk54 after 20 steps, under jax 0.6.2 on CPU;
the size is machine- and version-dependent). Earlier notes in this repo claimed bitwise
identity — that claim was wrong, and `tests/test_scheme_equivalence.py` asserts agreement
to a relative 1e-13 instead.

### The linear propagator (`jax_rmhd/propagators.py`)

The integrating factor above is one instance of a general hook. An equation set may
declare `linear_matrix_func(kgrid, params) → L` in its `EquationRecipe`; `setup_kgrids`
builds it once and stores it on the `K_Grids` (`lin_L`, plus `lin_m`/`lin_s2` for the 2×2
backend), and the timesteppers only ever call `apply_exp(arr, τ)` and — from the IMEX
schemes — `solve_shifted(arr, a) = (I − a·L)⁻¹ arr`. No stepper sees a matrix.

**Sign convention, fixed repo-wide:** `L` is defined by `∂ₜf = L f + N(f)`, so the
propagator is `exp(L·τ)` and a damped mode has `Re λ(L) < 0`. (RMHD's `L` is exactly the
old `hdiss`; the `∂ₜf + M f = 0` convention of the earlier gdi branch is *not* used here.)

The backend is selected by the shape of `L`, which carries an optional z/kz axis
(broadcast, size 1, whenever the operator is perpendicular-only):

- **diagonal**, `(nfields, nz-or-1, nkx, nky)`: `exp(L·τ)·arr`, elementwise.
- **putzer2**, `(2, 2, nz-or-1, nkx, nky)`: the closed form
  `exp(Lτ) = e^{mτ}[cosh(sτ)·I + (sinh(sτ)/s)·(L − m·I)]` with `m = tr L/2` and
  `s² = m² − det L` (Putzer/Sylvester). No eigendecomposition and no eigenvector storage,
  and it is smooth through the defective points where an eigenbasis does not exist.

Two traps in the 2×2 form, both guarded in `tests/test_linear_propagator.py`:

- **everything is complex.** `s²` is negative for any oscillatory mode and complex in
  general; a real `sqrt` NaNs silently. `cosh(sτ)` and `sinh(sτ)/s` are *even* in `s`, so
  they are single-valued functions of `s²` and the branch of the sqrt does not matter.
- **`sinh(sτ)/s` is 0/0 at a defective mode**, so it switches to its Taylor series
  `τ·(1 + z²/6 + z⁴/120 + z⁶/5040)`, `z = sτ`, below a cutoff on `|z²|` (`1e-6` at fp64,
  `1e-4` at fp32 — the coefficients are evaluated at fp64 and cast). The truncation error
  there is `|z|⁸/362880`, i.e. far below round-off at either precision, and the branch
  is taken with a `jnp.where` on a *safe* denominator so the unused branch cannot NaN.

`setup_kgrids` also asserts that `L(−kx, ky) = conj(L(kx, ky))` on the `ky = 0` and
Nyquist rows: without it the exponential would break the reality constraint of the rfft2
layout (see "Fourier conventions").

`apply_exp`'s optional third argument (`coef`) multiplies the propagator *factor* rather
than the array, and `scaled(c)` returns the propagator of `c·L`. Both exist so the
steppers can preserve the exact floating-point op order of the pre-propagator code
(`exp(hdiss·dt·γ)`, `dt·exp(hdiss·dt/2)·k₃`): with them the refactor is bitwise identical
on RMHD, without them it moves at round-off.

## Stochastic forcing

An Ornstein-Uhlenbeck process on a shell of perpendicular wavenumbers, rescaled every step
to inject power at a fixed target rate.

### Reality symmetrization: divide by √2, not 2

On the `ky = 0` and Nyquist rows the noise has to be made Hermitian in `kx`. The obvious
`(c[k] + conj(c[−k]))/2` is wrong, and wrong in two different ways at once:

- for a paired `(kx, −kx)` it averages two independent complex Gaussians, halving the
  variance;
- for the self-conjugate points `kx = 0` and `kx = Nyquist` it produces `Re(c)`, whose
  target variance is that of a *real* Gaussian, not a complex one.

Dividing the symmetrized combination by `√2` restores both at once. Using `/2` instead
anisotropically underforces exactly the modes that vary only in x.

### Cap the scale factor, never floor the denominator

`safe_scale(target, P)` computes `target/P` and clips the *result* to `±forcing_scale_max`.
The tempting alternative — flooring `P` away from zero — misbehaves whenever `P` is small
but nonzero (enormous or sign-flipped scales) and has no principled floor value, because
`P` carries units. Capping the scale bounds the worst case regardless of what `P` does,
including the `P = 0` first step from a quiescent initial condition.

The cost is that while the cap is engaged the injection is *uncontrolled* rather than
equal to the target — visible as a transient overshoot during spin-up from rest.

### Power conventions

`forcing_power` (momentum mode) and both entries of `forcing_power_elsasser` are in the
same units: a contribution to the **total** energy injection rate. Total `dE/dt` is
`forcing_power` in momentum mode and `ε₊ + ε₋` in elsasser mode, so `(p/2, p/2)` matches
`forcing_power = p`.

This needs a factor of 2 inside `rmhd._forcing_scale_from`, because with
`E_± = ⟨|∇z_±|²⟩/2` the total energy is `E = (E₊ + E₋)/2`, not `E₊ + E₋`. The cross-helicity
injection rate is then `dH_c/dt = ε₊ − ε₋`. Before 2026-07-31 the elsasser entries meant
half this; runs from before that date inject twice as much at the same numbers.

### Per-step normalization

`forcing_norm_per_step` (production default) computes the normalization scale once per
step from the step's starting fields and reuses it across the RK sub-stages, instead of
renormalizing at every stage. The realized injection then tracks the target to O(dt/τ)
rather than exactly. Worth about +8% at fp64 (see `docs/performance.md`).

### 2D momentum forcing is pure hydrodynamics

In 2D with `forcing_mode="momentum"` from a quiescent start, `ψ` stays *exactly* zero —
bitwise, not approximately. Its only source in 2D is the `{φ, ψ}` bracket, which vanishes
identically when `ψ = 0`, and momentum forcing drives only `φ`. Use `forcing_mode="elsasser"`
for actual 2D MHD from rest.

## GDI (2D, `physics/gdi.py`)

Normalization `ρₛ = cₛ = Ωᵢ = 1` (eqs 5.4-5.5 of docs/"GDI_nonlinear_equations (10).pdf"),
fields `(N, φ)` in that order — `N = δn/n₀`, `φ = eφ_electrostatic/T_e`. The 3D model's
parallel-diffusion term `D_∥ ∂²/∂z²` is replaced by a perp-k-local closure,
`γ_∥(k) = gpar_fac · ν_in · k⊥²` (`gpar_fac` scales eq 4.3's current-closure floor; in eq
4.7's parametrization α = 1 + gpar_fac, so the default `gpar_fac=1` gives α=2, the
minimum-γ_∥ result quoted at eq 4.6) — valid until P4b adds a real kz axis.

**Deriving `L`.** `physics/gdi.linear_matrix` derives `L` directly from the scalar PDEs
(5.1)-(5.2), with `γ_∥` substituted for `D_∥ ∂²/∂z²`; the result agrees entry-by-entry
with the paper's eq (5.3) matrix under `L = -M` (full derivation and the (3.7)
factorization: docs/gdi_linear_matrix_note.tex):

```
L[N,N]     = -γ_∥ - diss·k⊥^(2·hyper)
L[N,φ]     =  γ_∥ + i·ky_deriv/Ln
L[φ,N]     =  gpar_fac·ν_in - i·ky_deriv·ν_in·v0·inv_ksq
L[φ,φ]     = -ν_in - gpar_fac·ν_in - diss·k⊥^(2·hyper)
```

`tests/test_gdi_linear.py` cross-checks this `L`'s eigenvalues against an independently
re-derived exact quadratic (eq 3.7, ω-form), and against eq (2.8)'s collisional (2.9) and
inertial/Keskinen-Ossakow (2.11) asymptotic limits at `γ_∥ = 0` — the dispersion
relations are the arbiter for the sign convention.

**The Nyquist-row fix.** `ky_deriv` is `kgrid.ky` with the last (Nyquist) row zeroed, used
everywhere an `i·ky` cross term appears in `L`. A bare `i·ky` there breaks the propagator
setup's `L(-kx,ky) = conj(L(kx,ky))` reality check at that row: this is the standard
spectral-differentiation subtlety that odd-order derivatives are ill-defined at the
self-conjugate Nyquist point (Trefethen). RMHD's nonlinear terms never hit this because
the 2/3 dealiasing mask removes the Nyquist row from every nonlinear/IC path; `L` is
applied unconditionally by the exact propagator (no dealiasing), so it needs the explicit
fix. Physically inconsequential (a single edge row with no resolved dynamics).

**dt ceiling.** The propagator applies `L` exactly, so it imposes no *stability* limit —
but the E×B CFL alone (mirroring `rmhd.set_timestep`) doesn't bound the timestep needed for
IF-RK to correctly weight the nonlinear forcing of a stiffly-damped mode (`γΔt ≲ 1`, see
plans/GDI_PLAN.md's "Design background"). `gdi.set_timestep` therefore also caps `dt` at
`lin_dt_safety / max|Re λ(L)|`, computed once with plain numpy at trace time
(`gdi._max_re_lambda`, `functools.lru_cache`d on `Parameters`' identity hash), with
`lin_dt_safety` (`params.eqpars`, default 0.5) a separate safety knob from `cfl_safety`.

The max is taken over the **dealiased** (2/3-rule) region only, not the whole k-grid.
`γ_∥ = gpar_fac·ν_in·k⊥²` (the density equation's electron-adiabaticity relaxation rate) is
genuinely unbounded in `k⊥` — real physics, not a bug — so including the grid-cutoff modes
would make `max|Re λ|`, and therefore `dt`, scale with resolution and `diss` for no
accuracy benefit: those modes receive *zero* nonlinear forcing (`run.initialize` masks the
IC, `NonlinearTerm` masks every step), so the "misweighted nonlinear forcing of a stiff
mode" concern the ceiling exists for cannot apply there, and the propagator applies `L`
there exactly regardless of how large `γ_∥` gets.

**Energy budget.** `gdi.energy_budget` extends eq (3.18) (which is derived at `v0=0,
1/Ln=0` — no drive) with the Ln/Pedersen drive terms and the perp-hyperdissipation sink,
since the bracket/advection nonlinearity conserves `E = ½⟨N² + |∇⊥φ|²⟩` exactly under
periodic BCs — so the full linear operator's contraction against `(N, φ)` accounts for the
*entire* measured `dE/dt` of a nonlinear run. Measured closure in
`tests/test_gdi_linear.py::test_energy_budget_closure_nonlinear`: relative error ~4e-8 at
fp64 over a short random-IC run (centered-difference `dE/dt` vs the budget's `total`).

## Reading 2D MHD results

Energy cascades **forward** in 2D MHD — the opposite of 2D hydrodynamics' inverse cascade
— but `⟨ψ²⟩` inverse-cascades regardless. A plateaued energy spectrum is therefore not on
its own evidence of a saturated state; check that `⟨ψ²⟩` has stopped climbing too.

Per the zeroth law of turbulence, and given adequate resolution, `diss` sets the
time-to-saturation and the dissipation-range cutoff — not the saturated amplitude.
