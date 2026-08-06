# WebGPU forced RMHD (2D + 3D)

Two browser apps over one shared core: `rmhd2d.html` (2D) and `rmhd3d.html` (3D
spectral-z, the `z_spectral=True` path — Alfvén coupling applied exactly via the
closed-form 2×2 wave propagator, no wave CFL; z-slice or three-face cube display,
resolutions 64²×32 to 256²×64 plus the long-box 64²×128 / 64²×256, free Lz,
`z_diss_k` kz⁴ dissipation with auto). `index.html` is a landing page linking the two
apps; the preconfigured runs are presets *inside* each app (dropdown, or `?demo=` as a
deep link), not separate pages. The 3D contract is `SPEC3D.md`; its reference vectors are
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
  vectors, its dimension-specific WGSL and `Solver`, and its UI layout and defaults, on
  top of `<script src="common.js">` and `<script src="physics.js">` (in that order).
- `common.js` — the shared pieces that carry no equation: RNG, reference-vector
  flatteners, the FFT kernel template, the generic reductions (CFL, energy tail,
  max-reduce), device bring-up, the **colormap table** (`CMAP_COEF` / `cmapRGB`, the one
  source both the WGSL and the editor preview read), the chart + overlay drawing (energy
  trace, spectra, cut trace, arrows), the **card system** (`DisplayCard`, `ChartCard`,
  `cardsInit`/`cardsLayout`/`cardsSync`), the preset machinery (`presetBoot`,
  `presetWrite`) and the boot wiring both apps share (`bootApply`,
  `wireCommonControls`), the CPU-side IC construction (glyph raster, periodic box blur,
  gradient-amplitude normalization, gaussian z-envelope, and the whole `custom` blob
  editor — deposit math, preview, pointer→grid mapping, driven by one per-app hook object
  through `icDrawWire`), the self-test table and the frame loop (with two per-app hooks,
  `frameHook` and `readoutExtra`).
- `physics.js` — the shared RMHD kernels and the whole slice display chain, as
  templates over one constants object per app (`{pre, hasZ, wgReal, nDisp, arrow, ns,
  envFn}`), plus the shared `struct Mode` and `CMAP_WGSL` (four colormaps, one
  implementation, expanded from common.js's coefficient table at emit time):
  `prepGrads`, `bracket`, `nlAssemble`, `energyPartial`, `ou`,
  `scale`, `icFinish`, `prepDisp`, `vecMag`/`vecMagSq`, `maxSumPartial`,
  `sigmaCombine`, the arrow/cut gathers, the colorize and the blit (the 3D-only
  `zpPrep`/`planeEnergy` tracking kernels stay in the 3D app). Every 2D/3D
  difference is derived inside the templates from the single flag `hasZ` (the 3D index
  split m = iz·NMP + mp, the kz dealias factor, the kz⁴ term in the linear diagonal),
  so the apps hand over sizes, not code. Deliberately NOT shared, and still per-app:
  `stage` (diagonal exponential vs the 2×2 Alfvén propagator), `forcingAdd`/`envExpand`,
  the spectra, and the 3D `sliceExtract`/`faceExtract`/cube path — plus `makeGrid` and
  the `Solver` classes. The refactor that moved this out of the apps was gated on the
  generated WGSL staying byte-identical at every resolution preset in both apps.
- `style.css` — the shared dark theme.
- `SPEC.md`, `SPEC3D.md` — the implementation contracts, extracted from the JAX source;
  read these before touching the math.
- `gen_refvectors.py`, `refvectors.json`, `gen_refvectors3d.py`, `refvectors3d.json` —
  reference-vector generators and their output.

## Cards: displays and charts

The UI is built from **cards**. A *display card* is one WebGPU canvas plus its own
quantity selector, colormap, arrows checkbox and — in 3D — its own z slice (manual
slider or `track z⁺ / z⁻ peak`); "+ display" adds one (up to three), the × on its header
closes it. Card index *is* the solver's display-chain index, so N cards cost exactly N
chains (scratch, bind groups, gather targets, textures), each built lazily on first use.
The first card cannot be closed: it anchors the IC editor's overlay canvas. A *chart
card* is one 2D canvas with a type selector — energy trace / spectra / cut — and "+ chart"
adds another; several cards of the same type are allowed and share one throttled
readback per frame. Nothing here is a special case of anything else: there is no
dual-view flag and no fixed chart stack.

The **preset** dropdown in the sticky top bar (Run / Reset / t·step·dt·steps-per-second)
picks a whole configuration — controls *and* card layout. `?demo=NAME` is the same thing
as a deep link. Controls live in collapsible `<details>` groups (simulation /
dissipation / forcing / initial condition / displays & charts), open by default on a wide
viewport and collapsed on a narrow one; the state is deliberately not persisted.

## Display modes

Both apps: vorticity / current / phi / psi as signed fields (symmetric ±max), and the
perpendicular vector magnitudes "velocity |u|", "magnetic |b|" and the two Elsasser
fields "|z⁺|", "|z⁻|" (z± = φ±ψ, displayed as |ẑ×∇z±|), each with an optional ≤32×32
arrow overlay. Grid row y=0 is drawn at the top of the canvas. A **cut trace** chart card
plots the *first* display card's quantity along y at x = Lx/2 (its current z slice in 3D),
autoscaled symmetrically for signed fields and to [0,max] for magnitudes.

**Colormaps** are per display card: `afmhot` (default), `viridis`, `RdBu`, `grayscale`.
One WGSL `cmap(x, which)` serves the slice colorize and the 3D cube colorize; afmhot is
matplotlib's exact closed form, viridis and RdBu are degree-6 least-squares fits of each
channel to matplotlib 3.10's tables sampled at 256 points (max abs channel error 0.017
and 0.040; both clamped, since a fit overshoots slightly at the ends). The coefficients
live once, in `common.js: CMAP_COEF` — `physics.js` expands them into the WGSL at emit
time and the IC editor's CPU preview reads the same table through `cmapRGB`, so the two
cannot drift.

**Cross helicity σ_c** = (|z⁺|²−|z⁻|²)/(|z⁺|²+|z⁻|²), pointwise, is the one mode with a
**fixed** colour range (±1, symmetric — no autoscale, so colours are comparable
between frames and between runs). Where the local Elsasser energy density |z⁺|²+|z⁻|²
falls below 1e-4 × its maximum over the displayed field, σ_c is rendered as exactly 0:
in quiet regions the ratio is pure noise. It costs four inverse transforms per frame
(both components of both z±) instead of the vector modes' two — display cost only, no
physics buffer is touched. Its cut trace is still autoscaled to the data.

3D only: the **cube-face** modes draw the three visible boundary faces of the box
(z = Lz, x = Lx, y = Ly) in the oblique view `examples/forced-turbulence-3D.ipynb` uses
(matplotlib `view_init(elev=30, azim=45)`), colorized with one common ±max across the
three faces and depth-cued by a per-face darkening of 1.0 / 0.85 / 0.7. Any display card
may select them (each chain owns its own face buffer and three face textures — a rounding
error next to its nz·nx·ny scratch volume). Arrows, the cut trace and that card's z-slice
slider are inactive there. The 3D spectrum panel also carries the
parallel spectra E_u(k∥), E_b(k∥) as dashed curves (|kz| bins 1…nz/2, ±kz paired,
kz = 0 omitted from the log axis).

## Initial conditions

The solvers know exactly two ICs: the built-in large-scale mode pattern (`setIC()`, what
the reference vectors use) and an exactly quiescent state (`setIC(true)`). Everything
else is built on the CPU and uploaded through **`setICFromReal(phi, psi)`** — real-space
`Float32Array`s in the buffers' own layout (`ix*ny + iy` in 2D, `(iz*nx + ix)*ny + iy` in
3D) — which forward-transforms them and applies the 2/3 dealias exactly like the
constructor does (unmasked beyond-cutoff IC energy would persist and alias). The **IC**
selector then offers `large-scale modes`, `quiescent`, `letters` and `custom`; **Reset**
re-applies the current one.

The **letters** preset builds one Elsasser stream function per field from a rasterized
glyph (A → z⁺, B → z⁻; fixed, `common.js: IC_LETTERS` — the free-text input was dropped
in the mobile pass): the glyph is drawn on an
offscreen canvas at 60% of the box, gaussian-smoothed (`ctx.filter="blur(Npx)"` where the
engine supports it, otherwise a 3-pass periodic box blur), zero-meaned, and scaled so
that **max |ẑ×∇z±| equals the amplitude knob**; then φ = (z⁺+z⁻)/2, ψ = (z⁺−z⁻)/2. In 3D
each field is additionally multiplied by a gaussian z-envelope of peak exactly 1
(σ_z = Lz/16) to make a *wave packet*. The smoothing is not cosmetic: the spectral
gradient the GPU takes of an unsmoothed glyph edge overshoots the finite-difference one
used for the normalization by ~19%, and at σ ≥ 2 px they agree to ~0.1%, which is what
makes the amplitude a statement about the displayed field. (The dealias at upload still
trims it slightly, so the knob is approximate.)

## Drawn ICs: the blob editor (`custom`)

Selecting **custom** shows one extra row of controls; **edit IC** pauses the run and puts
an opaque 2D canvas (`#cvEdit`) over the display. Click or drag on it and each sample
deposits a **periodic gaussian blob** into the chosen target field — `z⁺`, `z⁻`, `φ` or
`ψ`. The editor keeps only the two Elsasser stream functions on the CPU: a φ blob is one
blob in *both* z±, a ψ blob one of each sign, and **apply & run** converts once with
φ = (z⁺+z⁻)/2, ψ = (z⁺−z⁻)/2 and uploads through `setICFromReal`. Nothing touches the GPU
while editing — the canvas is redrawn from the CPU arrays (the signed-afmhot mapping of
the `colorize` kernel, autoscaled) at most once per animation frame. **clear** is the
undo (there is no undo stack), **Reset** re-applies the same drawing, and the drawing is
dropped when the grid changes (resolution, or Lz in 3D) because its layout is per-grid.

- **Amplitude** keeps the same meaning as everywhere else in the IC code: the peak
  |ẑ×∇| of the blob. For f = P·exp(−r²/2σ²) that peak is P/(σ√e), so the editor deposits
  a stream-function peak P = a·σ·√e. Overlapping strokes add, so a scribble goes higher.
- **σ⊥** (and **σ_z** in 3D) are fractions of the box, floored at 2 cells — anything
  narrower is eaten by the 2/3 dealias at upload.
- **Sign**: the "negative" checkbox XOR the right mouse button, so a right-drag is a
  negative stroke without touching the controls.
- Blobs are truncated at 5σ (keeping 1−4·10⁻⁶ of the gaussian) and wrapped with the
  minimum-image convention `icGaussZ` already used, so a blob painted on the box edge is
  exactly a translate of one painted in the middle.
- **3D**: each blob is a perpendicular gaussian times a z-envelope of peak 1 centred on
  the z plane display 1 is currently showing (slider or tracked peak — the preview shows
  that plane too). Cube-face modes have no plane to draw on, so editing is refused there.

Mouse → grid: `getBoundingClientRect` is in CSS pixels, so the responsive canvas width and
the device pixel ratio both divide out. The display puts grid point i at screen fraction
(i+0.5)/n — texel centres as the linear sampler sees them, the same convention
`drawArrows` uses to place its cells — and row iy = 0 is at the **top** (the render pass
samples v = 1−uv.y). So the continuous grid coordinate under the cursor is u·n−0.5 and the
nearest grid point is round(u·n−0.5); the deposit uses the continuous position, wrapped
into [0,L).

## Presets (the dropdown, and `?demo=`)

A preset is a **configuration**, never new physics. A registry entry carries `set`
(control id → value), `prep` (the "auto" buttons), `layout` (the display + chart cards it
wants) and a one-line `hint`. `presetBoot` (common.js) fills the dropdown, preselects
`?demo=NAME` if the URL carries one, and writes the controls *before* the first solver is
built; picking another entry from the dropdown runs the same path plus a rebuild.
Everything stays adjustable afterwards. There are no separate demo pages.

- **2D `decaying A / B packets`** (`rmhd2d.html?demo=decay`) — letters A/B at 512²,
  forcing off, hyper=4 with auto diss, two displays showing |z⁺| and |z⁻|. Watch the
  spectrum settle onto the quasi-universal decaying slope, and switch a display to σ_c
  for dynamic alignment.
- **3D `Alfven-wave collision`** (`rmhd3d.html?demo=collision`) — 64²×256, Lz = 8π,
  forcing off, two counter-propagating letter packets, each display card tracking the
  peak plane of one of them.

## Alfvén-wave collision: directions, placement, χ

The propagation direction is read off the *implemented* propagator, not assumed. The 3D
stage kernel applies exp(L·τ) with L = [[d, i·kzd], [i·kzd, d]] to (φ, ψ); its
eigenvectors are z± = φ±ψ with eigenvalues d ± i·kz, so dz±/dt = ±i·kz·z±, and with the
inverse transform's e^{+i kz z} convention that is **z±(z,t) = z±(z±t, 0)**: z⁺ travels
toward *smaller* z, z⁻ toward *larger* z, both at v_A = 1. (Same statement as RMHD's
∂_t z± ∓ v_A ∂_z z± = …, with B₀ = ẑ.) The packets are therefore placed at
z₀⁺ = 11Lz/16 and z₀⁻ = 5Lz/16 — 6 σ_z apart, so they start well separated — and meet
head-on at **z = Lz/2 at t = 3Lz/16 ≈ 4.7** for Lz = 8π; the next (wrap-around)
collision at z = 0 comes a full half-box later (t = 11Lz/16 — the closure speed is
2·v_A, so collisions repeat every Lz/2 in time), which is why the offsets are 3Lz/16
and not Lz/4.

Next to the amplitude slider the page shows **χ ≈ a·k̄⊥/(k̄∥·v_A)** with k̄⊥ = 2π/(0.3·Lx)
(the letter scale), k̄∥ = 1/σ_z and v_A = 1 — an *estimate*, the packets are not single
modes. χ ≪ 1 is the weak regime: the packets pass through each other nearly unchanged
and distortion accumulates over many transits (the box is periodic, so they collide again
and again); χ ≳ 1 is strong, a single collision already shreds them. The demo starts at
a ≈ 0.2, χ ≈ 1.

**Max-energy plane tracking.** Because z is spectral, "the energy in plane iz" is not a
k-space partial sum: `readPlaneEnergy` forms z± = φ±ψ, inverse-transforms *along z only*
(borrowing the gradient stack as scratch — a separate submit, and submits execute in
order), and reduces one workgroup per plane with the usual perpendicular-energy weight
minus the nz² Parseval factor, so the plane values average to E±. The frame loop reads
back 2·nz floats at ~10 Hz; every display card's z-slice source can be `manual`,
`track z⁺ peak` or `track z⁻ peak`, and the tracked plane indices and their z coordinates
are printed in the readout.

**Lz** is a free parameter of the 3D grid (selector: 2π/4π/8π/16π), with `64²×128` and
`64²×256` added to the resolution presets. Everything kz-derived already read `p.Lz` —
kz/kzd/z_diss_k in `makeGrid`, `parKfac` for the parallel-spectrum abscissa, the
`autoZDiss` cutoff, the IC's z coordinate — and no generated WGSL depends on Lz at all
(verified: identical bytes at 2π/8π/16π for five grids). The one thing Lz does not
stretch is the cube-face view, which always draws the box as a cube.

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
