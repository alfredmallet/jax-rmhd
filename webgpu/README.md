# WebGPU forced RMHD (2D + 3D)

Two browser apps over one shared core: `rmhd2d.html` (2D) and `rmhd3d.html` (3D
spectral-z, the `z_spectral=True` path — Alfvén coupling applied exactly via the
closed-form 2×2 wave propagator, no wave CFL; z-slice or three-face cube display,
resolutions 64²×32 to 256²×64, `z_diss_k` kz⁴ dissipation with auto). `index.html` is a
landing page linking the two. The 3D contract is `SPEC3D.md`; its reference vectors are
`refvectors3d.json` from `gen_refvectors3d.py` (16²×8, fp64, including a dedicated
exp(L·τ) propagator vector). The rest of this README describes the 2D app; everything
carries over to 3D except where SPEC3D.md says otherwise.

## 2D app

A browser-based port of the repo's 2D forced-RMHD solver: pseudospectral (numpy-rfft2
layout), LSRK33 integrating-factor stepper, elsasser Ornstein–Uhlenbeck forcing with
per-step power normalization, adaptive CFL dt with `cfl_every`-style blocks. fp32
(WebGPU has no f64). All physics runs in WGSL compute shaders, including a
workgroup-shared-memory Stockham FFT; the only CPU work per step is drawing the
shell-restricted OU noise (~12 modes).

## Run

Open `index.html` (or either app directly) in a WebGPU-capable browser (Chrome, Firefox,
Safari — 2026 versions all ship it). Works from `file://`, no server or build step:
`<script src>` and `<link rel=stylesheet>` are allowed there, which is why the reference
vectors stay *inlined* in each app (`fetch` is not). If no adapter is found the page
says so.

## Self-test

The **Self-test** button runs the 32² reference vectors (inlined from
`refvectors.json`, generated at fp64 by `gen_refvectors.py` against `jax_rmhd` itself)
through the same GPU pipelines as production: FFT layout, static grids, nonlinear term,
forcing term + normalization scales, energy, CFL dt, and one full deterministic LSRK33
step (recorded state → recorded dt and scales → compare), plus a statistical
injection-rate check of the OU path. Thresholds are fp32-appropriate (1e-5…5e-4
relative L2; 20% on the statistical check). The 3D page adds the exp(L·τ) propagator,
the forcing envelope on the kz = ±2π/Lz planes, and two energy-budget rows for the
spectra.

To regenerate the vectors after changing the physics:

```
PYTHONPATH=<repo root> RMHD_PRECISION=64 python3 webgpu/gen_refvectors.py
```

then re-inline the JSON into `rmhd2d.html` (it sits on a single marked line — splice it
programmatically; do not try to hand-edit a 180 kB line).

## Files

- `index.html` — landing page (links, description, contributors). No solver code.
- `rmhd2d.html`, `rmhd3d.html` — the two apps. Each holds ONLY its inlined reference
  vectors, its physics/dimension-specific WGSL and `Solver`, and its UI layout and
  defaults, on top of `<script src="common.js">`.
- `common.js` — everything shared: RNG, reference-vector flatteners, the FFT kernel
  template and the dimension-agnostic WGSL (CFL, energy tail, max-reduce, vecMag,
  arrow/cut gathers, afmhot colorize, blit), device bring-up, the chart + overlay
  drawing (energy trace, spectra, cut trace, arrows), the self-test table and the frame
  loop. Physics kernels and the `Solver` classes deliberately stay per-app.
- `style.css` — the shared dark theme.
- `SPEC.md`, `SPEC3D.md` — the implementation contracts, extracted from the JAX source;
  read these before touching the math.
- `gen_refvectors.py`, `refvectors.json`, `gen_refvectors3d.py`, `refvectors3d.json` —
  reference-vector generators and their output.

## Display modes

Both apps: vorticity / current / phi / psi as signed fields (afmhot, symmetric ±max),
and "velocity |u|" / "magnetic |b|" magnitudes with a ≤32×32 arrow overlay. Grid row
y=0 is drawn at the top of the canvas. Under the spectrum panel, a **cut trace** plots
the displayed quantity along y at x = Lx/2 (the current z slice in 3D), autoscaled
symmetrically for signed fields and to [0,max] for magnitudes.

3D only: the **cube-face** modes draw the three visible boundary faces of the box
(z = Lz, x = Lx, y = Ly) in the oblique view `examples/forced-turbulence-3D.ipynb` uses
(matplotlib `view_init(elev=30, azim=45)`), colorized with one common ±max across the
three faces and depth-cued by a per-face darkening of 1.0 / 0.85 / 0.7. Arrows, the cut
trace and the z-slice slider are inactive there. The 3D spectrum panel also carries the
parallel spectra E_u(k∥), E_b(k∥) as dashed curves (|kz| bins 1…nz/2, ±kz paired,
kz = 0 omitted from the log axis).

## Expected long-time behavior

Defaults match `examples/forced-turbulence-2D.ipynb` (hyper=2, diss=1e-5, eps=(0.15,0.15),
fshell=(1,3)). On timescales much longer than the notebook's t=20, 2D RMHD piles magnetic
energy up at the box scale (inverse cascade of the mean-square flux function): verified
with the JAX solver itself at 128², E_mag grows roughly linearly (≈1 at t=20 → ≈5 at
t=140) while E_kin saturates, and the flow organizes into large coherent structures.
Stable large vortices/islands at late times are physics, not a solver bug — there is no
large-scale friction term to absorb the inverse cascade. Stronger forcing or a shell
closer to the box scale gets there proportionally faster. hyper=1 requires much larger
diss to resolve (k_η ≈ (ε/diss³)^¼ must stay below nx/3, e.g. diss ≳ 1.5e-3 at 512²,
ε=1); underresolved laplacian runs develop grid-scale pileup.

## Limitations

- fp32: fine for eyeballs and demos, expect slow energy-budget drift over long runs;
  not a substitute for the JAX solver for science runs.
- Resolutions 128/256/512 (1024 would hit the WebGPU minimum workgroup-storage limit
  exactly; see SPEC §8).
- Elsasser forcing only, no snapshots.
- The perpendicular spectrum dispatches `floor(min(nx,ny)/3)` bins, but
  `round(|k|/kunit)` reaches that value in the corners of the dealias ellipse, so those
  few modes are binned nowhere: `sum(bins)` is a hair below the total energy (1.5% at
  16², far less at production resolutions). Long-standing, cosmetic.
