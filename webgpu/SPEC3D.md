# WebGPU 3D forced RMHD (spectral-z) — implementation spec

> **File layout note (2026-08-05):** the app now lives in `webgpu/rmhd3d.html` (UI,
> physics WGSL, Solver, inlined reference vectors) on top of the shared `webgpu/common.js`
> + `webgpu/style.css`; `webgpu/index.html` is a landing page and `index3d.html` is gone.
> Everything below is unchanged as a contract — read "index3d.html" as "rmhd3d.html +
> common.js", and "index.html (2D)" as "rmhd2d.html".

Target: `webgpu/index3d.html`, a single self-contained file implementing the repo's
**z_spectral** 3D RMHD path (`params.z_spectral=True`) on WebGPU with LSRK33. It is the
3D sibling of `index.html` (2D): reuse that file's architecture, kernel-template style,
UI, charts, self-test harness, and everything in SPEC.md that this file does not
override. fp32. Reference vectors: `webgpu/refvectors3d.json` (fp64, from
`gen_refvectors3d.py`), inlined for self-test.

Physics source of truth: `jax_rmhd/physics/rmhd.py` (`linear_matrix` z_spectral branch,
`_kz_deriv`), `shared_physics.py` (`reconstruct_envelope` z_spectral branch,
`perp_reduce` nz² norm), `grids.py` (`dealias_mask`, rfftn), `propagators.py`
(Putzer2), plus docs/numerics.md.

## 1. Grid and transforms

Triply periodic `Lx × Ly × Lz` (all fixed 2π), grid `nx × ny × nz`. Spectral in ALL
THREE directions: numpy `rfftn(f, axes=(z,x,y))` layout — complex array shape
`(nkz, nkx, nky) = (nz, nx, ny/2+1)`; kz and kx two-sided (fftfreq), ky one-sided.
Storage layout choice is free, but state the indexing convention once and use it
everywhere; suggested flat index `m = (iz*nx + ix)*nky + iy`.

- `kz[iz] = fftfreq(nz)*nz * 2π/Lz` (two-sided, Nyquist at iz=nz/2 in the negative branch).
- 3D FFT = the 2D machinery (rows-y, cols-x) plus a strided complex pass along z of
  length nz. Forward: rfft rows(y) → complex cols(x) → complex z. Inverse: reverse
  order, total normalization 1/(nx·ny·nz) applied on the inverse (numpy convention).
  The z pass is an ordinary complex Stockham line FFT; batch all `nx·nky` lines (and both
  fields where possible) per dispatch like the existing passes.
- **Dealias mask**: the 2D perpendicular ellipse `(ix/(nx/3))² + (iy/(ny/3))² < 1`
  AND a plain kz cut `|iz_signed| < nz/3` (iz_signed = fftfreq index). Masks only the
  nonlinear term, exactly as in 2D.
- **Reality constraint**: on the ky=0 and ky=Nyquist columns,
  `F(-kx, -kz) = conj(F(kx, kz))` — the mirror is over BOTH kx and kz indices
  (`i → (nx−i) mod nx`, `iz → (nz−iz) mod nz`). Anything writing k-space directly
  (forcing) must preserve it.

## 2. Equations and linear propagator (THE key 3D difference)

Fields `phik, psik` shaped `(nz, nkx, nky)` complex. Nonlinear term: **identical to 2D**
(SPEC.md §2) — brackets are pointwise in real (z,x,y) space; the 8 inverse + 2 forward
transforms are now 3D; `-inv_ksq` uses the PERPENDICULAR ksq only (broadcast over kz);
dealias is the 3D mask of §1. There is NO separate z-derivative RHS term: the whole
parallel (Alfvén) coupling lives in the linear operator.

**Linear operator** (per mode, 2×2 over the (phi,psi) pair):

```
L = [[ d(k),  i*kzd ],
     [ i*kzd, d(k)  ]]
d(k)  = -diss*ksq_perp^hyper - z_diss_k*kz^4      (same for both fields: scalar diss)
kzd   = kz with the kz-Nyquist plane ZEROED (iz = nz/2 → 0)   [rmhd._kz_deriv]
```

(kz-Nyquist zeroing is required for the two-axis reality constraint; the kz dealias cut
removes that plane from the nonlinear path anyway. `z_diss_k*kz^4` uses the TRUE kz,
not the zeroed one — it's even, so reality-safe.)

**Exact propagator.** Because both diagonal entries are equal, the general Putzer2 form
collapses (m = d real, s² = −kzd², cosh(sτ)=cos(kzd·τ), sinh(sτ)/s = sin(kzd·τ)/kzd):

```
exp(L·τ) = e^{d·τ} · [[ cos(kzd·τ),   i·sin(kzd·τ) ],
                      [ i·sin(kzd·τ), cos(kzd·τ)   ]]
```

i.e. with c = e^{dτ}·cos(kzd·τ), s = e^{dτ}·sin(kzd·τ), and i·(re,im) = (−im,re):

```
phi' = c*phi + i*s*psi
psi' = i*s*phi + c*psi
```

Eigenvalues d ± i·kzd: exact Alfvén waves, **no wave CFL** — this is the whole point of
z_spectral. `sin(kzd·τ)/kzd` needs no care here because we use sin directly (not /kzd);
the only branch to mind is nothing: cos/sin are well-behaved at kzd=0 (c=e^{dτ}, s=0 →
degenerates to the 2D diagonal propagator — good sanity check). The LSRK33 stepper is
UNCHANGED (SPEC.md §3) except every `E_i * x` becomes this 2×2 apply with
τ = dt·gamma_i, applied jointly to the (phi,psi) pair — the stage kernel must process
both fields together now, not independently. (This closed form is valid ONLY for equal
diagonals; per-field diss would need full Putzer2 — out of scope, scalar diss only.)

## 3. Forcing (3D z-envelope)

OU state per Elsasser component is TWO perpendicular arrays [A, B] `(nkx,nky)` complex
(the 2D app already stores both and draws noise for both — the 2D path just never read
B; now both are read). OU update, shell mask, noise draw, symmetrization: exactly the
2D ones (SPEC.md §4), applied to A and B independently (independent RNG draws).

**Envelope reconstruction** (`shared_physics.reconstruct_envelope`, z_spectral branch):
the forcing k-space field is nonzero ONLY on the kz index +1 and nz−1 planes:

```
F(iz=1)     = (nz/2) * (A − i·B)
F(iz=nz−1)  = (nz/2) * (A + i·B)          (ADD, so nz=2 degenerates correctly)
```

(This is the exact unnormalized z-FFT of A·cos(2πz/Lz) + B·sin(2πz/Lz), and it
preserves the two-axis reality constraint for free given A,B are kx-Hermitian on the
ky=0/Nyquist columns.) The forcing RHS term is then the same z± → (f_phi, f_psi)
mapping as 2D, applied on those two planes.

**Normalization scale**: same formula as 2D but the inner product runs over the full 3D
grid with the z_spectral Parseval norm:

```
P± = Σ_{kz,kx,ky} ksq_perp * Re( conj(z±) * F± ) * yfac / (nz² * (nx*ny)²)
```

— only the two kz planes contribute, so the sum is 2 × (shell size) terms. Everything
else (2·eps±/P clip, lagged per-step scale, s±=0 iff eps±=0) is unchanged.
**Every energy-like quantity shares the nz² norm**: E_kin/E_mag, the spectrum bins, P±.
With that, `forcing_power` means the same thing as in 2D and as in jax.

## 4. Timestep

`set_timestep` under z_spectral is EXACTLY the 2D one (perpendicular advection + 0.1
velocity floor; no 1/dz term, no z_diss term — the propagator handles waves exactly).
The CFL reduction runs over the full 3D real-space gradient arrays. cfl_every blocks,
spin-up per-step recompute: as in 2D.

## 5. Diagnostics / UI / defaults

- E_kin, E_mag: 2D formulas with the nz² norm (sum over all modes).
- Perpendicular spectrum: same integer-|k_perp| shell binning, now also summed over kz
  (one more loop level in the bin kernel), same norm.
- Display: one z-SLICE of the selected field, chosen by a "z slice" slider (0..nz−1);
  the slice is extracted AFTER the inverse FFT of the display quantity (a 3D ifft per
  displayed frame is acceptable — it's 1/10th of one step's transforms; alternatively
  ifft only in perp after selecting... no: z is spectral, the full 3D ifft is required
  and fine). All display modes of the 2D app (vorticity, current, phi, psi, |u|+arrows,
  |b|+arrows — vectors are the PERPENDICULAR components on the slice) carry over, with the
  same colormap set.
- Controls: those of the 2D app, plus: resolution presets `64²×32`, `128²×32`,
  `128²×64` (default), `256²×64` (warn "needs a good GPU" in the hint); z-slice slider;
  `z_diss_k` slider (log10, default `3/kz_c³` with kz_c = (nz/3)·(2π/Lz) — sets the
  kz⁴ damping rate at the kz cutoff to ~3× the Alfvén frequency there — plus an "auto"
  that recomputes it; 0 allowed via the same checkbox pattern... a simple "0" position
  at the slider bottom is fine here, it's a stabilizer not a power input).
  Default perpendicular params: hyper=4, auto diss at 128², eps=(0.15,0.15), tau=1,
  fshell=(1,3), cfl 0.4, cfl_every 4.
- Energy trace + spectrum panels, self-test panel, acknowledgements: as in 2D.
- Outside this contract (display/IC features, no physics): Lz as a UI parameter with the
  added `64²×128` / `64²×256` presets (every kz-derived quantity already reads `p.Lz`),
  the letter-packet IC via `setICFromReal` (and the `custom` gaussian-blob editor, which
  deposits on the z plane the FIRST display card is showing and uploads through the same
  path), the per-real-z-plane Elsasser energy reduction that drives the per-card
  "track z± peak" slice sources, the display/chart card system (every chain carries the
  cube path, so any card may show faces), and the `collision` preset / `?demo=collision`
  URL configuration. See README.md.
- Add a small cross-link: 2D page ↔ 3D page (one <a> in each header).

## 6. Self-test (refvectors3d.json, 16²×8, inlined)

Same harness pattern as 2D, through the REAL GPU pipelines at 16²×8. Params recorded in
the JSON (diss=0.01, hyper=1, z_diss_k=1e-3, eps±=0.5, tau=1, fshell=(1,2), seed 7).
Field arrays shaped `(2, 8, 16, 9)` as {re,im}; forcing A,B recorded separately.

| key | test |
|---|---|
| `fft_input` (8×16×16 real), `fft_output` (8×16×9) | forward rfftn matches; roundtrip |
| `dealias` (8×16×9), `fmask` (16×9), `kzd` (8,) | static arrays match |
| `prop_in` (2,8,16,9), `prop_tau`, `prop_out` | ONE apply of exp(L·τ) matches (§2 closed form vs jax Putzer2) |
| `A_fields_k`, `A_forcing_A`, `A_forcing_B` (2,16,9 each), `A_forcing_scale` (2,) | state A |
| `A_envelope_k` (2,8,16,9) | reconstructed 3D envelope matches |
| `A_nonlinear_k` | NL term at A |
| `A_forcing_term_k` | forcing RHS term at A (recorded scales) |
| `A_scale_check` (2,) | recomputed s± |
| `A_energy` (2,), `A_dt` | energy, CFL dt |
| `B_fields_k` | one LSRK33 step from A, dt=`A_dt`, recorded scales, no OU |

Thresholds as in 2D (fp32 vs fp64: 1e-5 static/energy/dt, 1e-4 terms, 5e-4 step), plus
the statistical injection-rate check (quiescent 16²×8 run, dE/dt+D within 20% of
eps⁺+eps⁻).

## 7. Performance targets / notes

~30 3D transforms per step; at 128²×64 that's ~0.8–1GB of traffic/step → tens of
steps/s on integrated GPUs, 100+ discrete. The z pass adds one more shared-memory line
FFT; nz ≤ 64 lines are tiny — batch aggressively. Total GPU memory at 256²×64 stays
under ~500MB. fp32 throughout; 1024² perp is out of scope.
