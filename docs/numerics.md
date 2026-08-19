# Numerics notes

The *reasoning* behind the numerical choices in `taranis`. CLAUDE.md states the rules;
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

## Precision model

`jax_enable_x64` is turned on unconditionally at import (`taranis/__init__.py`) — x64
availability and *field* precision are no longer the same question. `TARANIS_PRECISION`
(`"32"`/`"64"`, default `"32"`, read exactly once by `taranis/_precision.py` at import)
sets `_precision.ftype`/`ctype`, the dtypes that `fields`/`forcing_state`/`forcing_scale`
are pinned to; it does not touch `SimulationState.t`, which is float64 at *both* field
precisions. This exists to fix a real failure mode: at fp32, `t + dt == t` exactly once
`t/dt > 1/eps32 ≈ 1.7e7`, so `while state.t < t_end` never terminates, and even before
that point roundoff random-walks `t` by `~sqrt(N)·eps32`. With x64 always on, `t` can be
fp64 for free while every field array stays at the dtype `TARANIS_PRECISION` selects — an
x64-enabled JAX process compiles an all-fp32/complex64 op graph to the same kernels it
always did; the flag only changes *default* dtypes and the promotion of any 64-bit input
that leaks in. The entire risk of the design is exactly that leak, which is why
`physics/__init__.py::construct_rhs` asserts `state.fields.dtype == _precision.ctype` and
`state.forcing_state.dtype == _precision.ctype` on every trace (`tests/test_precision_dtypes.py`
is the regression net; a pre-change reference lives in `tests/data/`).

Four rules govern any new code that touches both `t` and fields:

1. **Python scalars are weak-typed.** `0.5*fields`, IMEX tableau entries (plain python
   floats from the scheme tuples, e.g. `(b[k]*dt)*z`) do not upcast anything — leave them
   as bare python floats.
2. **Bare `jnp.array(...)`/`np.array(...)` (no dtype) are STRONG float64 under x64** and
   poison every field-math op they touch — this is the dangerous case. Any such array
   that ever multiplies fields (kgrid arrays, LSRK coefficient tables, …) must be pinned
   explicitly to `ftype`/`ctype` at construction, not patched after the fact.
3. **`jax.random.*` draws need an explicit `dtype=ftype`.** Without it, a draw under x64
   is float64 — not just the wrong dtype for fields, but a *different bitstream* than the
   float32 draw it replaces. Pinning the dtype is what keeps the fp32 RNG stream bitwise
   identical to before x64 was unconditional (`shared_physics`'s two `jax.random.normal`
   calls do this).
4. **Any `t`-derived quantity must downcast to `ftype` before it re-enters field math.**
   There is exactly one such path today: `run.py::_advance_forcing`'s
   `dt = (new_state.t - prev_t).astype(_precision.ftype)`, which feeds the OU decay
   factor multiplying complex64/128 `forcing_state`. A future time-dependent term func
   that reads `t` must do the same at the point it touches fields — it will not be caught
   by rule 2's pins because `t` itself is legitimately fp64.

### The fp32 spectral noise shelf

Storing a field at complex64 quantizes every Fourier coefficient to relative precision
`eps32 ≈ 6e-8`, so a coefficient's *energy* (`~|f|²`) carries a relative roundoff floor of
`eps32² ≈ 4e-15` against the peak of the spectrum, `E_peak`: below `~eps²·E_peak` in
absolute terms, a perpendicular spectrum is showing fp32 storage/arithmetic roundoff, not
physics. Don't fit a dissipation range below that floor, and don't mistake a flattening
spectrum there for a bottleneck or condensate effect — check the same run at
`TARANIS_PRECISION=64` before drawing a physics conclusion from spectral content that close
to `eps²·E_peak`. This is a separate mechanism from the z-stencil error below (which is
about `k∥`-resolution, not overall field storage) and from the fp64 budget-closure gap
fixed by promoting the *reductions* rather than the fields themselves (Appendix B of
`plans/PRECISION_PLAN.md`).

### The fp32 z-stencil error floor

For a mode with parallel wavenumber `k∥` stored at fp32, the 4th-order first-derivative
stencil (`shared_physics.z_derivatives`) has relative error

```
(k∥·dz)⁴/30                     truncation
+ eps/(k∥·dz)                   roundoff (dominated by fp32 STORAGE quantization of f —
                                 upcasting only the stencil arithmetic recovers nothing)
```

Both terms are relative to that mode's own amplitude — the stencil acts per `(kx,ky)`
coefficient, so the error floor is a property of each perpendicular mode independently,
not of the field as a whole. The two terms trade off oppositely in `k∥·dz`, so the sum is
minimized at

```
k∥·dz ≈ (7.5·eps)^(1/5)
```

which is `≈0.05` at fp32 (`eps = 6e-8`), giving an irreducible relative-error floor of
`~1e-6`: refining `dz` *below* this makes `∂∥` **worse**, because the roundoff term grows
as `1/(k∥·dz)`. At fp64 the same balance puts the sweet spot at `k∥·dz ≈ 1e-3` with a
floor of `~1e-13` — invisible in any practical run.

Three consequences:

- **Critical balance puts the worst conditioning on the outer scale.** Since the error is
  per-perp-mode relative to that mode's own amplitude, and critical balance scales
  `k∥ ∝ k⊥^(2/3)`, the outer-scale modes (smallest `k⊥`, smallest `k∥`) sit furthest from
  the `k∥·dz ≈ 0.05` optimum for a fixed `dz` — they are the modes most likely to be
  under-resolved in `k∥·dz` even when the inertial range is fine.
- **The spurious content lands at grid-scale `k_z`,** with amplitude `~eps·|f|` — exactly
  the wavenumber `z_diss`'s hyperdissipation targets. Keep `z_diss` on at fp32; it is
  what removes this roundoff-sourced grid-scale content rather than letting it
  accumulate.
- **`z_spectral` sidesteps the cancellation entirely.** `params.z_spectral` makes `∂∥`
  the exact `±i·k_z` diagonal of `L`, applied as a unitary phase by `apply_exp` — there is
  no finite-difference truncation/roundoff trade-off at all, so it is the fp32-robust
  parallel formulation. It is currently `dims==3`, `size==1` only.

Practical rule: at fp32, size `nz` so that the outer-scale `k∥·dz ≳ 0.05` — below that,
adding z resolution actively degrades the parallel derivative rather than improving it.

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
  `∂ₜφ = i·kz·ψ`, `∂ₜψ = i·kz·φ` — the exact spectral form of `rmhd.FDLinearTerm`'s `∂ψ/∂z` in the
  (vorticity/−k⊥²) equation and `∂φ/∂z` in the ψ equation. The 2×2 putzer2 backend gives
  eigenvalues `±i·kz`, i.e. `z± = φ±ψ` propagating at unit speed, *exactly*: no wave CFL and no
  dispersion error, which is why `rmhd.set_timestep` drops both its `1/dz` and `z_diss` entries
  there and `halo_start`/`FDLinearTerm` are skipped (plain `if` on the static parameter).
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

### The linear propagator (`taranis/propagators.py`)

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

`scaled(c)` returns the propagator of `c·L`. `lsrk_advance` uses it to pre-scale by `dt`
so a stage exponent is formed as `exp((L·dt)·γ)` — the floating-point op order of the
pre-propagator code, which keeps the LSRK schemes bitwise identical to it on RMHD.

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

The scale factor is always **clipped** to `±forcing_scale_max`, and the denominator is
never floored. Flooring `P` away from zero misbehaves whenever `P` is small but nonzero
(enormous or sign-flipped scales) and has no principled floor value, because `P` carries
units. Capping the scale bounds the worst case regardless of what `P` does. That part is
unchanged; what changed is what gets clipped.

### Normalize against the forcing's own self-energy (behaviour change 2026-08-08)

Over one step of length `dt` the force `s·f_raw` acting on the fields `z` injects

```
ΔE = s·P·dt + ½·s²·F₂·dt²,     P = ⟨∇z·∇f_raw⟩,   F₂ = ⟨|∇f_raw|²⟩
```

— the linear cross term **plus** the forcing's self term. `safe_scale(target, P)` keeps
only the first: `s = target/P`. That is right in saturation, and wrong from rest. From a
quiescent start `P = 0` exactly, so `safe_scale` pinned at `±forcing_scale_max` and the
first forced step injected `½·s_max²·F₂·dt²` — a quantity in which **`forcing_power` does
not appear at all**, because ε enters the forcing *only* through the scale and the clamp
erased it. Measured at 64², `fshell=(1,3)`, `cfl_safety=0.5` (where the quiescent velocity
floor gives `dt = 0.491`): `E` after step 2 was `1.074e+01` for `ε_tot = 1e-4, 1e-3, 1e-2`
and `1e-1` alike, identical to four digits, and at small dissipation that kick *was* the
final state. Because `ΔE ∝ ½s_max²F₂dt²` with `⟨|f|²⟩ ∝ (1 − e^{−2dt/τ})`, it scales as
`dt³`: measured 1.07e1 / 8.19e-2 / 7.08e-4 at `dt` = 0.49 / 0.098 / 0.020, versus the
predicted ratio `5³ = 125`.

`selfnorm_scale(target, P, F₂, dt)` (`shared_physics.py`) instead solves

```
½·F₂·dt·s² + P·s − target = 0        →        s = [−P + √(P² + 2·F₂·dt·target)] / (F₂·dt)
```

so `ΔE = target·dt` **exactly**, and then clips to `±forcing_scale_max` as before. Limits:
`F₂·dt → 0` or `|P|` large gives `s → target/P` (saturation is unchanged); `P → 0` gives
`s → √(2·target/(F₂·dt))` (the first kick is `target·dt` on the nose). The expression is
continuous in `P` through zero, which matters because the same pinning recurred *mid-run*
whenever `P` fluctuated through zero with a weak field — a fix special-cased on `t = 0`
would not have covered that. The clip survives as a last-resort safety: it engages if
`F₂·dt` underflows while `P ≈ 0`, or under a strongly adverse `P < 0` (there
`s ~ 2|P|/(F₂·dt)`; a clipped step under-injects, transiently even net-negative through
the linear term). Measured margins in developed runs sit more than an order of magnitude
from the clip.

Three details worth knowing:

- **The positive root, not the sign-following one.** `safe_scale` follows `sign(P)`, so an
  adverse phase flips the force to keep injecting — a rectification of the O-U process. The
  positive root keeps `s > 0` and lets the exact solve absorb an adverse linear term
  instead (still hitting the target unless the clip engages).
- **Numerically, the root is evaluated in two branches.** `(−P + √D)/(F₂dt)` cancels
  catastrophically for `P > 0` in the saturated regime `2F₂·dt·target ≪ P²`, and the
  conjugate form `2·target/(P + √D)` cancels for `P < 0`. Each is used where it does not
  cancel; both are algebraically identical.
- **`dt` is lagged by one step.** `run._advance_forcing` passes the dt of the step just
  completed, since that is what is known when the scale is stored. This is one more
  `O(dt/τ)`-class approximation of exactly the kind `forcing_norm_per_step` already makes,
  and it is *exact* under `cfl_every > 1`, where dt is frozen for the block.

**Dated behaviour change.** Runs from before **2026-08-08** spin up differently from a
quiescent (or weak-field) start: they take an ε-independent `O(½s_max²F₂dt²)` kick on step
2 that later runs do not, and at low dissipation that kick can dominate the saturated
state. Saturated statistics are affected only at `O(s²F₂dt/target)`. Same class of
non-comparability as the 2026-07-31 elsasser factor-2 note below. `tests/test_forcing_spinup.py`
pins the new behaviour; `tests/data/forcing_spinup_reference.npz` records the old one.

**The per-stage path (`forcing_norm_per_step=False`) keeps `safe_scale`.** `ForcingTerm`
has no `dt` — the RHS interface is deliberately dt-agnostic, `dt` is a stepper-local chosen
once per step (and frozen per block under `cfl_every`), and recomputing it inside the term
would repeat the CFL allreduce every stage *and* give the wrong value inside a block.
Rather than thread `dt` through `construct_rhs` and every stepper, that path gets a
mitigation instead: its clip is tightened to `min(s_max, √(2·target/(F₂·dt_q)))`, with
`dt_q` the **params-static** bound on the quiescent step length (`params.dt` when
`adaptive_timestep=False`, where it is exact; otherwise
`cfl_safety·min(dx,dy)/QUIESCENT_EPS`, which `rmhd.set_timestep` applies as an explicit
`jnp.minimum` ceiling — so `dt ≤ dt_q` holds by construction for every state, not by
keeping two expressions in sync).
That bounds the from-rest kick at `~target·dt_q` instead of `~½s_max²F₂dt_q²`, and only
binds when `P` is small, i.e. near-quiescent states. **From-rest starts that need exact
normalization should use the production default.**

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

## Test particles: fields seen by a particle

The reference derivation for `taranis/particles/fields.py` (design and phasing:
`plans/TESTPART_PLAN.md`). Charged test particles are pushed in Cartesian `(E, B)`, so the
only question this section answers is what `(E, B)` the RMHD fields `(φ, ψ)` *mean*.

**Conventions.** `shared_physics.bracket(a, b) = a_x b_y − a_y b_x = ẑ·(∇a × ∇b)`, and
`rmhd.NonlinearTerm` builds `∂ₜω + {φ, ω} = {ψ, j_z}` with `ω = ∇²φ`, `j_z = ∇²ψ` — i.e.
`∂ₜω + u·∇ω = b·∇j_z` — exactly when

```
u   = ẑ×∇φ = (−∂_yφ, ∂_xφ)
b_⊥ = ẑ×∇ψ = (−∂_yψ, ∂_xψ)
```

Those two identifications fix the potentials: `b_⊥ = ∇×(A_z ẑ) = −ẑ×∇A_z`, so `A_z = −ψ`,
and `u = E×B/B² = ẑ×∇(Φ/B₀)` gives `Φ = B₀φ`. Hence

```
B   = B₀ẑ + ẑ×∇ψ
E_⊥ = −∇_⊥Φ           = −B₀∇φ
E_z = −∂_zΦ − ∂_t A_z = −B₀∂_zφ + ∂ψ/∂t
```

**The parallel-gradient terms cancel.** The code's induction equation is
`∂ψ/∂t = −{φ, ψ} + B₀∂_zφ + η∇²ψ (+ f_ψ)` — the `∂_zφ` piece being `rmhd.FDLinearTerm`'s
z-stencil (or, under `z_spectral`, the `±i·kz` off-diagonal of `rmhd.linear_matrix`),
absent in 2D. Substituting it into `E_z` above kills the `B₀∂_zφ` terms exactly:

```
E_z = −{φ, ψ} + η∇²ψ (+ f_ψ)        the physics terms, identical in 2D and 3D
```

so the field assembly is dimension-independent in the physics and Phase B inherits it
unchanged. Note the sign: `E_z = +∂ψ/∂t`, because `A_z = −ψ` in this convention.

**Caveat: the finite-difference-z filter is a fourth, unrepresented piece.** In 3D without
`z_spectral`, `rmhd.FDLinearTerm` adds `−z_diss·(dz/2)⁴ ∂_z⁴ψ` to `∂ψ/∂t` alongside the
Alfvén stencil (`physics/rmhd.py`, `FDLinearTerm`). It is a numerical filter, not a physics
term, and it is in none of the three pieces above — so `E_z` (full mask) equals `∂ψ/∂t`
*exactly* only in 2D and under `z_spectral` 3D. Under `z_spectral` the analogous
`−z_diss_k·kz⁴` IS part of the ψ diagonal of `rmhd.linear_matrix` and therefore lands inside
the resistive piece automatically. Whether the FD-z filter should be added to the particles'
`E_z` (it is an artificial EMF, like the resistive piece, but unlike it not part of any
`L`) is a Phase B decision; Phase A is 2D, where the question does not arise.

**Cross-check against `E = −u×B`** (ideal Ohm, which is what the ideal piece must be):
`(u×B)_z = u_x b_y − u_y b_x = (−∂_yφ)(∂_xψ) − (∂_xφ)(−∂_yψ) = {φ, ψ}`, so
`−(u×B)_z = −{φ, ψ}`, the ideal `E_z` above; and `−u×B₀ẑ = −(ẑ×∇φ)×B₀ẑ = −B₀∇φ = E_⊥`.
Running that backwards, the E×B drift of `(E_⊥, B₀ẑ)` is `ẑ×∇φ = u` — which is what kernel
gate 2 pins numerically.

**What pins which sign.** Kernel gate 2 (`tests/test_particles_kernel.py`) pins the
*pusher's* Lorentz-force convention only: it drives uniform analytic `(E, B)` arrays, so it
never touches the field assembly. The assembly's *relative* signs are pinned by the ideal-Ohm
identity `E·B = 0`, which holds exactly for the raw (undealiased) ideal `E_z`
(`E_x b_x + E_y b_y + B₀·(−{φ,ψ}) = 0` identically, since both terms are `B₀{φ,ψ}` with
opposite signs) — asserted in `tests/test_particles_coupled.py`. That fixes `b_⊥` relative to
`E_⊥` and `E_z`, and `B_z = +B₀`; it does not fix the overall sign of `b_⊥` and `E_z` against
`ψ`, which is what gate 4's canonical `p_z = m v_z − qψ` invariant does.

**The canonical p_z invariant.** In 2D, z is ignorable (∂_z = 0 on every field), so the
canonical momentum conjugate to z, `p_z = m v_z + qA_z = m v_z − qψ` (using `A_z = −ψ`), is
conserved exactly along a particle orbit — for ANY time-dependent `(φ, ψ)`, not just frozen
fields. Newton's law gives `m dv_z/dt = q(E_z + (v×B)_z)`. `B_⊥ = ∇×(A_zẑ)` with `A_z`
z-independent makes `(v×B)_z = v_xB_y − v_yB_x = −(v_x∂_xA_z + v_y∂_yA_z) = −v·∇A_z`
identically (algebra alone, no dynamics), and `E_z = −∂_zΦ − ∂_tA_z = −∂_tA_z` since
`Φ = B₀φ` has no z-dependence either. So `m dv_z/dt = q(−∂_tA_z − v·∇A_z) = −q·dA_z/dt`
(the total derivative of `A_z(x(t),y(t),t)` along the orbit), i.e. `d/dt(m v_z + qA_z) = 0`
— the standard ignorable-coordinate argument, specialized to 2D. This holds **only with the
FULL mask** (`ez_resistive=ez_forcing=True`): `E_z` must equal the true `∂ψ/∂t` the discrete
ψ obeys, piece for piece, or the cancellation is incomplete and `p_z` drifts by exactly the
omitted acceleration (the default ideal-only ensembles are the worked example — see the
plan's gate 4). Because the invariant needs both `b_⊥` (through `(v×B)_z`) and `E_z` at
their correct relative AND absolute sign against `ψ`, it is the gate that pins the absolute
sign left open above (gate 4). With fields frozen over a step (the run.py push convention),
the per-step defect in the identity is the time-discretization error of an otherwise-exact
cancellation, `O(dt²)` per step, so the accumulated drift over a fixed physical time is
`O(dt)` — consistent with the KDK push being exact in space but first-order in how it
samples the field's time dependence.

`E_∥ = E·b̂` differs from `E_z` by `b_⊥·E_⊥/B₀` corrections; the pusher is Cartesian and
never needs it.

### The pieces, and why `∂ψ/∂t` is dealiased

`particle_fields` returns the piece decomposition rather than a summed `E`, because which
pieces a particle sees is a per-ensemble choice (`FIELD_PIECES` /
`FIELD_MASK_DEFAULTS`, assembled by `assemble_stacked` — `assemble` is its summed `(E, B)`
form; all mask logic is static python):

- **ideal**, `−{φ, ψ}`. On by default.
- **resistive**, `L_ψ ψk` with `L_ψ` the ψ diagonal of `rmhd.linear_matrix` — `η j_z` at
  `hyper=1`, its hyper-resistive analogue otherwise. Only the diagonal: under `z_spectral`
  the full 2×2 `L` also carries the `±i·kz` off-diagonal, which is the `B₀∂_zφ` term that
  has already cancelled out of `E_z`, so `propagators.apply_L` is the wrong tool here.
  Off by default.
- **forcing**, `f_ψ` (the scaled elsasser envelope's ψ half; identically zero in momentum
  mode). Off by default.

`PFields.ex/ey` hold `E_⊥/B₀ = −∇φ`, not `E_⊥` itself: `B₀` enters in exactly one place,
`assemble_stacked(pf, mask, B0)`, which scales `(ex, ey)` by `B₀` and sets `B_z = B₀`. Physically
`E_⊥ = −B₀∇φ` as derived above; storing it per unit `B₀` keeps a single knob.

Defaults are the ideal-Ohm particle: for a collisionless test particle the resistive piece
is a fluid-closure/numerical-regularization artifact and the forcing piece is a stirring
EMF, so neither is physical `E` — but the three pieces together are exactly `∂ψ/∂t` (2D; see
the FD-z caveat above), and an `ez_resistive = ez_forcing = True` ensemble run alongside a
default one measures the resistive-acceleration difference on identical fields.

**The ideal piece must be dealiased.** The discrete ψ obeys

```
∂ₜ ψk = dealias·NL_ψ,k + F_ψ,k + L_ψ ψk
```

(`rmhd.NonlinearTerm` applies `kgrid.dealias`; the propagator applies `L` unmasked). The
raw pointwise bracket `−{φ, ψ}` has spectral content out to `2k_c` that *never enters ψ*,
so the field a particle should see is

```
E_z,ideal = ifft(dealias · fft(−{φ, ψ}))
```

not the raw product. This is not cosmetic: with the raw bracket the gate-7 comparison
against a centered difference of ψ across a step stalls at the (dt-independent) size of
the discarded beyond-cutoff content instead of converging at O(dt²), and gate 4's exact
2D invariant `p_z = m v_z − qψ` stops being exact. The cost is one extra `fft`+`ifft` pair,
which is the whole reason to state it here rather than rediscover it.

**Transform budget of `particle_fields`**: 4 `ifft`s for the `(∂_x, ∂_y)(φ, ψ)` gradients,
`+2` for the dealiased ideal `E_z`, `+1` when the resistive piece is requested, `+1` when
the forcing piece is — so ≤8 per step, against the ~30–36 of a 3-stage RHS.

### Work bookkeeping

Heating attribution is *measured*, not inferred: `ParticleState.w` carries, per particle,
the cumulative work per unit mass done by each electric piece
(`fields.WORK_PIECES = (eperp, ez_ideal, ez_resistive, ez_forcing)`, the electric members
of `FIELD_PIECES`). The split is exact, not a quadrature.

Write one Boris half-kick with `h = q·dt_k/(2m)`: `v⁻ = v + hE`, a rotation `v⁻ → v⁺` with
`|v⁺| = |v⁻|`, then `v_new = v⁺ + hE`. Then

```
½|v_new|² = ½|v⁺|² + h E·v⁺ + ½h²|E|²
½|v|²     = ½|v⁻|² − h E·v⁻ + ½h²|E|²
```

and `|v⁺| = |v⁻|`, so subtracting gives `½|v_new|² − ½|v|² = h E·(v⁺ + v⁻) = h E·(v_new + v)`
— the rotation drops out identically, no small-`dt_k` expansion. `boris.push_tracked`
accumulates exactly that expression restricted to each E component (`E_⊥` → the `x,y`
terms, each `E_z` piece → the `z` term with its own piece's sampled value), so summing the
columns of `w` reproduces `½|v|² − ½|v₀|²` per particle to round-off, over any number of
steps and substeps and whatever the mask. A piece an ensemble does not see contributes
exactly zero, forever. Gate 8 (`tests/test_particles_coupled.py`) asserts both, measured
at `~5e-14` of `KE₀` in *both* precisions — `w` and the push are fp64 whatever the field
precision is, so the closure does not degrade at fp32.

Two consequences worth stating: the work is credited with the *same* field sample the kick
used (any other choice would leave an `O(dt)` residual in the closure), and the piece
columns are attributions of the *pusher's* energy change, so `w_ez_*` is not by itself the
parallel heating — the rotation exchanges ⊥ and ∥ energy within a step.

**The magnetic moment uses the local field.** `state.moments`'s `mu` is
`μ = v_⊥B²/(2|B|)` with
`v_⊥B² = |v|² − (v·B)²/|B|²` and `B = B₀ẑ + b_⊥` *sampled at the particle*, not
`v_x²+v_y²/(2B₀)`. In RMHD `|B|` differs from `B₀` only at `O(ε²)` while the field
*direction* tilts at `O(ε)`, so the ẑ-referred perpendicular energy and the true one differ
at first order in `b_⊥/B₀` — the same effect that makes the naive grad-B drift wrong by an
O(1) factor (below). `push_tracked` returns the B sample of its last half-kick precisely so
`μ` costs no extra gather.

### E∥ projection and the amplitude parameter

**The gathered field has a numerical E∥.** On the grid the ideal pieces are exactly
orthogonal: `E_⊥·b_⊥ + B₀·(−{φ,ψ}) = 0` identically (above). At the particle they are not.
`interp.gather` interpolates each component of `E` and `B` independently, and the bilinear
interpolant of a product is not the product of the interpolants, so the sampled
`Ē·B̄ = O(dx²)·|E||B|` — a parallel electric field of purely numerical origin, which
accelerates particles along `B` and contaminates exactly the quantity a heating run
measures. Measured on a developed 64² state: `rms(Ē·B̄)/rms(|Ē||B̄|) ≈ 1.4·10⁻³`
(gate 9, `tests/test_particles_coupled.py`).

The per-ensemble flag `epar_project` removes it the way Xia, Perez, Chandran & Quataert
2013 (ApJ 776, 90, their eq. 21) do, at every half-kick, per particle:

```
Ẽ = Ē − (Ē·B̄) B̄/|B̄|²        so  Ẽ·B̄ = 0 exactly (4·10⁻¹⁷ of |E||B| measured)
```

`boris.project_perp` is the kernel; `push_tracked(..., project=True)` applies it to the
sample *before* the kick and before the work columns are formed, so `w` credits the
projected field and the closure identity above stays exact. The projection is defined only
for the exact ideal-Ohm mask (`bperp = eperp = ez_ideal = True`, the non-ideal `E_z` pieces
off) and any other mask is a config error: with `ez_resistive`/`ez_forcing` on, `E∥` is
partly *real* (that is the whole point of the paired resistive-split ensembles) and
projecting would delete it; without the three ideal pieces there is nothing to project. It
also mixes the `E_z` pieces, which is the second reason it needs a single one.

**`B₀` is the RMHD amplitude parameter, `B₀ = 1/ε`.** Nondimensionalize the real particle
equation in RMHD units — `u = ε v_A ũ`, `ψ = ε B₀ L ψ̃`, `t = L t̃/(ε v_A)`, velocities in
flow units `ε v_A`:

```
dv̂/dt̃ = Ω̂ [ −∇φ̃ + ε(∂_t ψ̃) ẑ + v̂ × (ẑ + ε b̃_⊥) ],      Ω̂ = ΩL/(ε v_A)
```

while the code pushes `qm·B₀·[ −∇φ + (1/B₀)(∂_tψ) ẑ + v × (ẑ + b_⊥/B₀) ]`. The two are
identical with `qm·B₀ = Ω̂` and `B₀ = 1/ε`. So `B₀` is not a free unit: `B₀ = 1` means
`δB/B₀ ~ 1`, an amplitude no RMHD ordering supports (Xia et al. cap their runs at
`δB_rms/B₀ ≤ 0.47` and flag even that). Production runs `B₀ ~ 10` with `q/m` scaled by
`1/B₀` so that `qm·B₀` — hence `ρ = v_⊥/(qm·B₀)`, `Ω·dt` and `ξ` — is unchanged; the one
thing that does change is `β_i = v_⊥²/v_A² = v_th²/B₀²` (0.01 for `v_th = 1`, `B₀ = 10`).
`B₀` is therefore a PER-ENSEMBLE key (the top-level one is only its default), so one run
can hold several amplitudes; with every field piece off it changes no orbit at all
(gate 9's control).

**Heating is measured in the local-B frame.** `MOMENTS` carries `vperpB2` and `vparB2` —
`v_∥B² = (v·B)²/|B|²` and `v_⊥B² = |v|² − v_∥B²` with the pusher's own `B` sample — beside
the ẑ-referred `vperp2`/`vz2`, and `diagnostics.particles.heating_rate` halves all four.
Perpendicular heating is the local-B one (Xia et al. §4.1): the field direction tilts at
`O(ε)` while `|B|` varies only at `O(ε²)`, so the ẑ-referred split mixes ⊥ and ∥ energy at
first order in `b_⊥/B₀` — the same effect as the `μ` and grad-B remarks above.

### Dimensionless groups

`q/m` carries the scale interpretation of the gyration: `Ω = qm·B₀` and
`ρ = v_⊥/(qm·B₀)`, so at `B₀ = 1` (the default) `Ω = q/m` and `ρ = v_⊥ m/q`. `B₀` itself is
the amplitude parameter `1/ε` (above), not a unit choice. Three groups govern whether a
test-particle run means anything:

- `ρ/dx` — gyroradius against the grid, i.e. whether the fields the particle samples are
  resolved. Baseline `ρ ≈ 2–4 dx`.
- `Ω/ω_nl` — gyration against eddy turnover; the stochastic-heating parameter
  `ξ = δu(ρ)/v_⊥` is the same statement locally.
- `v_th/v_A` — thermal speed against the Alfvén speed.

At 256² (`dx ≈ 0.0245`) the baseline is `Ω = qm·B₀ ≈ 10–20`, i.e. 60–120 solver steps per
gyration: Boris at the solver `dt` is fully resolved with no substepping. `v_th/v_A` is
`v_th/B₀`.

**Guiding-centre formulas do not transfer naively.** Kernel gate 3
(`tests/test_particles_kernel.py`, derivation in its header) measures the perpendicular drift
in an RMHD `b_⊥ = ẑ×∇ψ` field and finds **2× the textbook `v_∇B`** (3× for a `v_z(0) = 0`
launch): the field *direction* shears at `O(ε)` while `|B| = √(B₀² + b_⊥²)` varies only at
`O(ε²)`, so shear terms enter the drift at the same order as the `|B|` gradient. Relevant
wherever a guiding-centre expression is used on these fields — including the `μ = v_⊥²/2|B|`
tracking of `plans/TESTPART_PLAN.md` §5.

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

**Energy budget.** `diagnostics/gdi.py`'s `energy_budget` extends eq (3.18) (which is derived at `v0=0,
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
