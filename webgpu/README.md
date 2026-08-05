# WebGPU 2D forced RMHD

A single-file, browser-based port of the repo's 2D forced-RMHD solver: pseudospectral
(numpy-rfft2 layout), LSRK33 integrating-factor stepper, elsasser Ornstein–Uhlenbeck
forcing with per-step power normalization, adaptive CFL dt with `cfl_every`-style
blocks. fp32 (WebGPU has no f64). All physics runs in WGSL compute shaders, including
a workgroup-shared-memory Stockham FFT; the only CPU work per step is drawing the
shell-restricted OU noise (~12 modes).

## Run

Open `index.html` in a WebGPU-capable browser (Chrome, Firefox, Safari — 2026 versions
all ship it). Works from `file://`, no server or build step. If no adapter is found the
page says so.

## Self-test

The **Self-test** button runs the 32² reference vectors (inlined from
`refvectors.json`, generated at fp64 by `gen_refvectors.py` against `jax_rmhd` itself)
through the same GPU pipelines as production: FFT layout, static grids, nonlinear term,
forcing term + normalization scales, energy, CFL dt, and one full deterministic LSRK33
step (recorded state → recorded dt and scales → compare), plus a statistical
injection-rate check of the OU path. Thresholds are fp32-appropriate (1e-5…5e-4
relative L2; 20% on the statistical check).

To regenerate the vectors after changing the physics:

```
PYTHONPATH=<repo root> RMHD_PRECISION=64 python3 webgpu/gen_refvectors.py
```

then re-inline the JSON into `index.html` (it sits on a single marked line).

## Files

- `index.html` — the whole app (JS + WGSL + inlined reference vectors).
- `SPEC.md` — the implementation contract, extracted from the JAX source; read this
  before touching the math.
- `gen_refvectors.py`, `refvectors.json` — reference-vector generator and its output.

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
- 2D only, elsasser forcing only, no snapshots. The display draws grid row y=0 at the
  top of the canvas.
