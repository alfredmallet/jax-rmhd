# WebGPU 2D RMHD — implementation spec

> **File layout note (2026-08-05):** the app was split into `webgpu/rmhd2d.html` (this
> spec's app: UI, physics WGSL, Solver, inlined reference vectors) plus `webgpu/common.js`
> (machinery shared with the 3D app) and `webgpu/style.css`; `webgpu/index.html` is now a
> landing page. Everything below is unchanged as a contract — read "index.html" as
> "rmhd2d.html + common.js".

Target: a single self-contained `webgpu/index.html` (no build step, no server — must work
from `file://`) implementing the 2D forced-RMHD solver of this repo (`jax_rmhd`) on
WebGPU, with the LSRK33 integrating-factor scheme. fp32 throughout (WebGPU has no f64).
This spec is extracted from the repo source (`jax_rmhd/physics/rmhd.py`,
`shared_physics.py`, `timestepping.py`, `grids.py`, `run.py`, `config.py`) — it is the
contract; match it exactly. Reference test vectors: `webgpu/refvectors.json` (generated
at fp64 by `webgpu/gen_refvectors.py`), to be **inlined** into index.html for the
self-test mode.

## 1. Grids and conventions

Doubly periodic box `Lx × Ly`, grid `nx × ny` (powers of two; default 512², selector for
128/256/512). Spectral representation: numpy-style rfft2 layout, complex array of shape
`(nkx, nky) = (nx, ny/2+1)`:

- `kx[i] = fftfreq(nx)*nx * 2π/Lx` — two-sided: `i < nx/2` → `i*2π/Lx`, else `(i-nx)*2π/Lx`
  (the Nyquist index `nx/2` sits in the NEGATIVE branch, matching numpy `fftfreq`).
- `ky[j] = j * 2π/Ly`, `j = 0..ny/2` — one-sided.
- `ksq = kx² + ky²`; `inv_ksq = 1/ksq` with `inv_ksq[0,0] = 0`.
- `yfac[j] = 2` except `yfac = 1` at `j=0` and `j=ny/2` (rfft2 ky-doubling for sums).
- Dealias mask (elliptical 2/3 rule, **mode-index** space, Lx/Ly cancel):
  `(ix/(nx/3))² + (iy/(ny/3))² < 1` with `ix = fftfreq(nx)*nx`, `iy = 0..ny/2`.
- FFTs are **unnormalized** (numpy convention): forward rfft2 has no prefactor; inverse
  irfft2 carries `1/(nx*ny)`. An O(1) real field has O(nx*ny) k-coefficients.
- **Reality constraint**: on the `j=0` and `j=ny/2` columns, `F(-kx) = conj(F(kx))`
  (kx index mirror `i -> (nx-i) mod nx`). Anything writing k-space directly (forcing
  noise) must enforce it explicitly. Physics kernels (products of ifft'd fields) preserve
  it automatically IF the inverse/forward transforms are exact-real; if you implement
  ifft via complex FFTs, make sure the real-space arrays are real by construction.

## 2. State and equations (2D RMHD)

Evolved fields in k-space: `phik` (stream function φ), `psik` (flux function ψ).
Derived: vorticity `vortk = -ksq*phik`, current `jpark = -ksq*psik`.

RHS convention `dt f = L f + N(f)`; L is handled by the integrating factor, N is the
explicit RHS (nonlinear + forcing).

**Linear operator (diagonal, real):** `L_f = -diss_f * ksq^hyper` per field
(`diss` scalar or per-field `(nu, eta)`; `hyper` integer, 1 = laplacian). Demo defaults
match `examples/forced-turbulence-2D.ipynb`: `diss=1e-5, hyper=2`, eps±=(0.15,0.15),
fshell=(1,3) (self-test uses `diss=0.01, hyper=1` from refvectors.json). hyper=1 needs
diss large enough that k_η ≈ (ε/diss³)^(1/4) < nx/3, i.e. ~1.5e-3 at 512², ε=1 —
underresolved laplacian runs pile up at the grid scale.

**Nonlinear term.** With `grad(f) = (∂x f, ∂y f)` obtained as `ifft(i*kx*fk)`,
`ifft(i*ky*fk)` for each of `f ∈ {phi, psi, vort, jpar}` (8 inverse FFTs), and the
Poisson bracket `{a,b} = a_x*b_y - a_y*b_x` evaluated pointwise in real space:

```
NL_vort = {psi, jpar} - {phi, vort}          (real space)
NL_psi  = -{phi, psi}
RHS_phi_k = -inv_ksq * fft(NL_vort) * dealias
RHS_psi_k =            fft(NL_psi)  * dealias
```

(2 forward FFTs. The dealias mask multiplies ONLY the nonlinear term — not the fields,
not the forcing.)

**Forcing term** (elsasser mode — the only correct 2D MHD mode; momentum forcing keeps
ψ≡0): add to the RHS

```
f_phi = 0.5*(s⁺*F⁺ + s⁻*F⁻)
f_psi = 0.5*(s⁺*F⁺ - s⁻*F⁻)
```

where `F± (nkx,nky complex)` are the two OU forcing-state arrays and `s±` the two scalar
normalization scales (§4). No dealias multiply (the shell is far inside the cutoff).

## 3. LSRK33 with integrating factor

Williamson (1980) 3-stage scheme; coefficients EXACT:

```
alpha = (0, -5/9, -153/128)
beta  = (1/3, 15/16, 8/15)
gamma = (1/3, 5/12, 1/4)          # sum = 1
```

One step of size `dt`, with `E_i = exp(L * dt * gamma_i)` (per-mode real factor;
exponent is `L*dt*gamma_i` — the propagator is pre-scaled by dt, matching
`lsrk_advance` + `DiagonalPropagator.apply_exp`):

```
delta = 0
u = (phik, psik), t
for i in 0..2:
    r = RHS(u)                          # nonlinear + forcing, at current u
    delta = E_i * (alpha_i * delta + dt * r)      # stage 0: alpha_0 = 0
    u     = E_i * u + beta_i * delta
    t    += gamma_i * dt
```

Note the op order: the exponential multiplies the *combination* `(alpha*delta + dt*r)`,
and separately the fields; `beta*delta` is added un-exponentiated. Keep it.

## 4. Forcing: Ornstein–Uhlenbeck process (once per FULL step)

Per full step (AFTER the LSRK33 update, using that step's actual `dt` — see §6 loop):

**Shell:** `kunit = min(2π/Lx, 2π/Ly)`; mask `fmask = (nmin <= sqrt(ksq)/kunit < nmax)`,
default `fshell = (1,2)`. Use **shell-restricted noise** (sanctioned by the repo's
`forcing_shell_noise=True`; statistically identical): draw noise only at the shell modes.

**OU update** for each of the two states `F±`:

```
noise_k = (N(0,1) + i*N(0,1)) / sqrt(2) * (nx*ny)        # at shell modes; 0 elsewhere
# hermitian-symmetrize the ky=0 and ky=Nyquist columns:
#   col <- (col + conj(col[(nx-i) mod nx])) / sqrt(2)     (divide by sqrt(2), NOT 2)
decay = exp(-dt/tau);  diffusion = sqrt(1 - decay²)
F <- (F*decay + diffusion*noise) * fmask
```

`tau = forcing_tau` (default 1.0). Any decent Gaussian RNG is fine CPU-side in JS
(xorshift128+ / mulberry32 + Box–Muller); jax's threefry need not be reproduced.
The shell mode list is tiny (~12 modes for fshell=(1,2)) — keep `F±` CPU-side and upload
the shell values (index list + complex values) each step, or keep them GPU-side; either
way the dense `F±` seen by kernels is nonzero only on the shell.

**Normalization scale** (per-step, LAGGED — the production `forcing_norm_per_step=True`):
after the OU update, compute from the NEW fields and NEW forcing state the two scalars

```
z± = phik ± psik
P± = Σ_k  ksq * Re( conj(z±) * F± ) * yfac   / (nx*ny)²        # nz=1
s± = clip( 2*eps± / P±,  -scale_max, +scale_max );   s± = 0 if eps± == 0
```

`eps± = forcing_power_elsasser` (each a contribution to the TOTAL energy injection rate;
`(p/2, p/2)` ≡ total power p). The factor 2 is load-bearing (E_tot = (E⁺+E⁻)/2). The
clip caps the SCALE, never floors P. `scale_max` default 1.0. The scales `s±` computed
here are used for ALL sub-stages of the NEXT step (they lag one step; the very first
step after init uses scales computed from the initial state). Since `F±` is shell-only,
P± is a ~12-term sum — do it CPU-side from a tiny GPU readback of the shell values of
`z±`, or GPU-side; if a readback is used it may be async/lagged (the scale is lagged by
design), but the self-test one-step path must use the exact recorded scales.

## 5. Adaptive timestep (CFL)

From the real-space gradients of φ and ψ (already computed for the bracket):

```
max_vy = max over grid of |∂x phi| + |∂x psi|
max_vx = max over grid of |∂y phi| + |∂y psi|
m = max( max_vx/dx, max_vy/dy, max(0.1/dx, 0.1/dy) )     # 0.1 = velocity floor
dt = cfl_safety / m
```

`dx = Lx/nx` etc. Default `cfl_safety = 0.4`. GPU max-reduction; a readback per step
stalls the pipeline, so recompute dt every `cfl_every` steps (default 4–8, UI knob) and
freeze it in between (matches the repo's `cfl_every` blocks; use the *block-start* state's
dt). Never start a frozen-dt block from a quiescent state (dt collapses during spin-up)
— recompute every step for the first ~100 steps or until energy is O(1).

## 6. Full step loop

```
compute dt (every cfl_every steps, from current grads)
u <- LSRK33_step(u, dt, scales s±)         # §3, RHS = NL + forcing with LAGGED s±
F± <- OU_update(F±, dt)                    # §4
s± <- normalization from (u, F±)           # §4, used by the NEXT step
```

Initialization: real-space IC → forward FFT → multiply by dealias mask (the IC IS
masked; evolution masks only NL). Demo default: start from small random large-scale
fields or zero (zero is fine — forcing spins it up; keep per-step dt recompute during
spin-up).

## 7. Diagnostics / UI

- Energy: `E_kin = 0.5 * Σ ksq*|phik|²*yfac / (nx*ny)²`, `E_mag` same with `psik`
  (this normalization is shared with P± above — keep it).
- Display: time t, step count, dt, E_kin, E_mag, steps/s. Render vorticity (default) /
  current / φ / ψ / |u| / |b| to canvas each animation frame (one extra ifft of the
  selected field — two for the vector modes, whose components are u = ẑ×∇φ and
  b = ẑ×∇ψ; render at most once per rAF, stepping multiple times per frame if fast).
  Colormap (matplotlib afmhot by default): signed fields map x = (v/vmax + 1)/2 over a
  symmetric autoscaled range, the non-negative magnitude modes map x = v/vmax. The two
  vector modes also overlay a subsampled (≤32×32) arrow field on a transparent 2D canvas,
  gathered on the GPU and read back at ~10 Hz (one frame of lag is fine).
- Controls: run/pause, reset, resolution (128/256/512), diss, hyper (1/2), eps⁺ and eps⁻
  (independent, lockable; each is a contribution to dE/dt, so the total is their sum),
  the forcing band [n_min, n_max), tau, cfl_safety, cfl_every, seed. Changing
  resolution/hyper/band rebuilds; the other sliders take effect live.
- Status line for WebGPU init failures (no adapter → clear message).
- Outside this contract (display/IC features, no physics): the Elsasser and σ_c display
  modes, the display/chart CARD system (N ≤ 3 display chains, per-card quantity, colormap
  — afmhot / viridis / RdBu / grayscale, one shared WGSL `cmap(x, which)` — and arrows;
  addable chart cards, each with its own options: the energy trace draws E_kin/E_mag or
  E±, the spectra E_u/E_b, E± or both, and the cut card its own selected pair of
  components on its own prepared line — the CPU-built IC presets uploaded through
  `setICFromReal` — the
  Elsasser POTENTIALS ζ± (z± = ẑ×∇ζ±), stored unnormalized and scaled to their two
  amplitude sliders only at apply time, including the `custom` gaussian-blob editor,
  which is CPU-only, lives in its own view and paints into the same upload path — and the
  preset dropdown / `?demo=` URL configurations. See README.md.

## 8. FFT requirements (the hard part)

WGSL compute-shader FFT, radix-2 Stockham (or radix-4/8 if you like), f32.
Requirements, not implementation choices:

- Must reproduce the numpy rfft2 layout of §1 exactly (self-test compares mode-by-mode).
- Batched: the RHS needs 8 inverse + 2 forward transforms per stage evaluation; batch
  rows/columns and multiple fields into single dispatches where possible.
- Real transforms may be implemented via complex FFTs (e.g. pack two real fields into
  one complex transform, or use a full complex ny transform and discard) — but the
  round-trip must be exactly real (write the imaginary part as 0, don't leave residue).
- Sizes: 128–512, power of two, `nx == ny` may be assumed (keep Lx≠Ly working in the
  math anyway). (1024 would need 16KB of workgroup shared memory per line — exactly the
  WebGPU guaranteed minimum, zero margin — so it is not offered.)
- Workgroup shared memory per 1D transform line is the standard approach (a 512-point
  complex f32 line = 4KB — fits easily; one workgroup per line, `workgroupBarrier`
  between butterfly stages).

## 9. Self-test mode (inlined reference vectors)

`refvectors.json` (nx=ny=32, Lx=Ly=2π, diss=0.01, hyper=1, eps±=0.5, tau=1, fshell=(1,2),
cfl_safety=0.4, scale_max=1.0; complex arrays as `{re,im}` nested lists, field arrays
shaped `(2, 32, 17)`) contains:

| key | test |
|---|---|
| `fft_input` (32×32 real), `fft_output` | forward rfft2 matches; roundtrip identity |
| `dealias`, `fmask`, `lin_L` (=-diss*ksq) | static grid arrays match |
| `ic_fields_k` | (context only) |
| `A_fields_k`, `A_forcing_state` (2×32×17), `A_forcing_scale` (2,) | load as state A |
| `A_nonlinear_k` | NL term at A matches |
| `A_forcing_term_k` | forcing term at A (with recorded scales) matches |
| `A_scale_check` (2,) | recomputed s± from A matches |
| `A_energy` (2,) | E_kin, E_mag at A match |
| `A_dt` | CFL dt at A matches |
| `B_fields_k` | ONE LSRK33 step from A with dt=`A_dt`, recorded scales, NO OU update → matches |

Run all tests at 32² through the same GPU pipelines as production (no special CPU path).
Report per-test relative L2 error against thresholds (reference is fp64; solver is fp32):
FFT/static/energy/dt ≤ 1e-5; NL/forcing/scale ≤ 1e-4; one-step B ≤ 5e-4. Also run a
statistical OU check: with quiescent-start forcing at 32², after ~500 steps the measured
dE/dt (energy gain + dissipation) should be within ~20% of eps⁺+eps⁻ (this validates the
OU/noise path the vectors can't).

## 10. Non-goals

No z direction, no MPI, no snapshots/checkpointing, no momentum forcing mode, no IMEX
schemes. Don't touch anything outside `webgpu/`.
