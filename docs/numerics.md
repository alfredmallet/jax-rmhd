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

## Time integration

Dissipation is not an RHS term. It is applied as an integrating factor: the perpendicular
hyperdissipation operator `kgrid.hdiss = −diss·k⊥^(2·hyper)` is diagonal in k-space, so
`exp(hdiss·dt)` can be applied exactly rather than integrated.

For the low-storage schemes each stage advances the fields by `exp(hdiss·dt·γᵢ)` and the
accumulator `delta` by the same factor. Since `Σγᵢ = 1` for both tables, a run with no
nonlinear term reproduces `exp(hdiss·dt)` per step exactly. Dropping the factor from
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

## Reading 2D MHD results

Energy cascades **forward** in 2D MHD — the opposite of 2D hydrodynamics' inverse cascade
— but `⟨ψ²⟩` inverse-cascades regardless. A plateaued energy spectrum is therefore not on
its own evidence of a saturated state; check that `⟨ψ²⟩` has stopped climbing too.

Per the zeroth law of turbulence, and given adequate resolution, `diss` sets the
time-to-saturation and the dissipation-range cutoff — not the saturated amplitude.
