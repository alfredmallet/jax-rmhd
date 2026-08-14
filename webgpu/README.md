# WebGPU RMHD (2D + 3D)

**This file is repo-facing**: how the thing is built, what the contracts are, why each
piece is the way it is, and how to verify a change. The **user-facing** documentation —
every button and every chart, in plain language — is `docs.html`, and the reading list
that takes a visitor from the demo to the literature is `reading.html`. Both are pages of
the app set itself and both are linked from the `.sub` line of `rmhd2d.html` /
`rmhd3d.html`. Nothing here is linked from those pages. When a control changes,
`docs.html` changes with it; when the *reason* for a control changes, this file does.

Two browser apps over one shared core: `rmhd2d.html` (2D) and `rmhd3d.html` (3D
spectral-z, the `z_spectral=True` path — Alfvén coupling applied exactly via the
closed-form 2×2 wave propagator, no wave CFL; z-slice or three-face cube display,
resolutions 64²×32 to 256²×64 plus the long-box 64²×128 / 64²×256, free Lz,
`z_diss_k` kz⁴ dissipation with auto). `rmhd2d.html` is the front door: `index.html` is an
immediate redirect to it (ONEPAGE_PLAN B), the two apps carry the same 2D/3D tab strip,
and the background essay that used to be the landing page now lives in two places: its
lead paragraphs are the "what is all this?" pane right under the subtitle (open until
dismissed once — remembered like the params toggle), and the five background panes are
the rail beside the canvas — intro, tabs, rail and no-WebGPU poster all built by
`chromeBuild` in `common.js`, so every piece of that text exists once. The preconfigured runs are presets *inside* each
app (dropdown, or `?demo=` as a deep link), not separate pages. The 3D contract is `SPEC3D.md`; its reference vectors are
`refvectors3d.json` from `gen_refvectors3d.py` (16²×8, fp64, including a dedicated
exp(L·τ) propagator vector). The rest of this README describes the 2D app; everything
carries over to 3D except where SPEC3D.md says otherwise.

## 2D app

A browser-based port of the repo's 2D RMHD solver: pseudospectral (numpy-rfft2
layout), LSRK33 integrating-factor stepper, optional elsasser Ornstein–Uhlenbeck forcing
with per-step power normalization, adaptive CFL dt with `cfl_every`-style blocks. fp32
(WebGPU has no f64). All physics runs in WGSL compute shaders, including a
workgroup-shared-memory Stockham FFT; the only CPU work per step is drawing the
shell-restricted OU noise (~12 modes).

## Run

Open `rmhd2d.html` (or `rmhd3d.html`, or `index.html`, which redirects to the 2D app) in a
WebGPU-capable browser (Chrome, Firefox, Safari — 2026 versions all ship it). Works from
`file://`, no server or build step: `<script src>` and `<link rel=stylesheet>` are allowed
there, which is why the reference vectors stay *inlined* in each app (`fetch` is not). If
no adapter is found the page says so and shows `poster.png` — a still of a real run — in
place of the live one, with the what-is rail opened. Every visit boots **paused** on its
preset (plain visits get the default forced-turbulence one at 256²): the big green **Run**
in the top bar is the call to action, and turns into a red **Pause** while running.

## Self-test

The **Self-test** button runs the 32² reference vectors (inlined from
`refvectors.json`, generated at fp64 by `gen_refvectors.py` against `taranis` itself)
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

- `index.html` — an immediate redirect to `rmhd2d.html` (meta refresh + `location.replace`
  + a plain link, `rel=canonical`). It exists because it is the URL GitHub Pages serves
  for `webgpu/` and the one old links point at; there is no landing page any more. Note
  that `pages.yml` seds this file too, so its `style.css` reference and its `buildid` span
  have to stay.
- `poster.png` — a 512² still of a real forced-turbulence run (|u|, afmhot, arrows,
  colorbar), shown *only* where `initGPU` fails. The `<img>` is created on that path and
  nowhere else, so a working visit never fetches it.
- `docs.html` — the user-facing manual: the topbar, every control group, the display and
  chart cards and their options, the colorbar, save/record, and one plain line per preset.
  Same stylesheet, no script, no build step. It is the page the apps link at the top; the
  accuracy rule when editing it is that every control named there must exist by id in the
  built panel (`controlsBuild`'s spec fragments and each app's own spec).
- `reading.html` — the annotated reading list: four "basics" sections (RMHD's origins,
  plasma physics, turbulence, heliophysics) then Alfred's own tour of the arguments the
  field has had, one `<details>` per thread. Same skeleton and stylesheet as `docs.html`,
  no script, no build step; linked from the `.sub` line of both apps and of `docs.html`.
  It is the ONLY page with outbound links, so two rules apply when editing it: every
  external `<a>` carries `target="_blank" rel="noopener"`, and every citation is a
  publisher DOI where one exists (arXiv only as a fallback). Its handful of page-specific
  rules (`h2`, `ul.refs`, `.todo`) live in a `<style>` block in the file rather than in
  `style.css`, which stays owned by the apps.
- `rmhd2d.html`, `rmhd3d.html` — the two apps. Each holds ONLY its inlined reference
  vectors, its dimension-specific WGSL and `Solver`, and its UI layout and defaults, on
  top of `<script src="common.js">` and `<script src="physics.js">` (in that order).
- `common.js` — the shared pieces that carry no equation: RNG, reference-vector
  flatteners, the FFT kernel template and the **rfft row pair** (`fftRowPair`: the y
  forward/inverse kernels, which both pages emitted as byte-identical twins), the generic
  reductions (CFL, energy tail,
  max-reduce), device bring-up and the **pooled readback** (`readBuf`, whose staging
  buffers are keyed by byte length and reused rather than allocated per call), the
  **colormap table** (`CMAP_COEF` / `cmapRGB`, the one
  source both the WGSL and the editor preview read), the **display quantity table**
  (`DISP_FIELDS`, physics.js's modes with their definitions — one table, both pages), the
  chart + overlay drawing (energy
  trace, spectra, cut trace, arrows), the **card system** (`DisplayCard`, `Recorder`,
  `ChartCard`,
  `cardsInit`/`cardsLayout`/`cardsSync`), the **control-panel builder**
  (`controlsBuild(spec)` and the row/group fragments the two pages share — the sticky
  top bar, the cfl row, the hyper/diss row, the whole forcing group, the IC group and
  the displays group; each app's page markup is now an empty `#topbar` / `#controls`
  plus its own spec), the **page chrome** (`chromeBuild`: the 2D/3D tab strip, whose
  inactive side is a plain link, the `RAIL_LEAD` / `RAIL_PANES` text of the what-is rail,
  and `gpuFallback`, the poster + opened rail an engine with no WebGPU gets), the preset
  machinery (`presetBoot`,
  `presetWrite`) and the boot wiring both apps share (`bootApply`,
  `wireCommonControls`, the locked slider pairs,
  `syncCommonLabels`), the CPU-side IC
  construction (glyph raster, periodic gaussian blur, the ζ± → (φ,ψ) normalization
  `icZetaFields`, gaussian z-envelope, packet placement + χ, and the whole `custom` blob
  editor — deposit math, its own **view**, preview, pointer→grid mapping, driven by one
  per-app hook object through `icDrawWire`), the z-plane trackers (`trackCentroid`,
  `trackArgmax`), the self-test table and the frame loop (with two per-app hooks,
  `frameHook` and `readoutExtra`).

  **The render gate** (2026-08-12). A display chain runs only when its picture can have
  changed: `renderCards(paused)` draws a card when `DisplayCard.needsRender()` says so.
  That asks three things — has the STATE moved (`seenMark !== stateMark()`, so a step, an
  IC upload, a preset and a rebuild all invalidate by construction and no caller has to
  know this gate exists), has this CARD moved (`dirty`, set by `apply()` and `_resize()`,
  i.e. by the controls that change no state), and is a take reading frames off it. Before this, a PAUSED page re-ran every chain at rAF
  rate forever — in 3D a full volume inverse transform per card per frame, two more per
  active contour set, and in the volume view the whole 512²×256 raymarch — over a state
  that could not move. The readbacks that read what a render left behind (arrows, the
  colorbar autoscale) are gated on the card's own frame counter; the ones that read the
  SOLVER (stats, the cut line, the spectra, and 3D's plane tracker and field-line march)
  are gated on `stateMark()`, the pair `(nsteps, stateSeq)`. `stateSeq` is what catches a
  state that jumped without a step — an IC upload resets `nsteps` to 0 — and is bumped by
  `chartsReset`, which both apps already call on exactly those paths. Each gate still lets
  ONE readback through after the last step, so a paused page shows the state you paused
  on, not the one 300 ms before it, and `cardsThrottleReset` clears the markers so a chart
  card added to a still page is filled rather than told "nothing new".

  The rule the 3D hooks got wrong first time, and the one to keep in mind when adding a
  fourth consumer of anything gated here: **a marker may only ever suppress a REPEAT.**
  `fieldLineHook` shipped gating on "the state has not moved AND something is cached",
  which is true as soon as ANY of its three consumers has been served — so a second one
  appearing over a still state (and both pages boot paused, so that is the ordinary case)
  was locked out for good, leaving the k∥ curve empty and a second lines card blank. It
  now asks `flStale()`: is anybody asking who has nothing? `planeTrackHook` asks its
  drift test the same way, before the early-out rather than after it.

  Three more things the gate has to get right, all pinned by `devtools/checkidle.js`: a
  live recording forces a draw (the encoder reads the texture that render produced, and a
  skipped frame would hand it an expired one — RECRAF), and a paused frame *settles* each
  active contour set's adapting range instead of relaxing it (`contSettle` writes the zero
  range that `contLevel` reads as "no history", so the last frame drawn takes the measured
  max outright). The relaxation exists to damp flicker on a moving field; on a still one it
  is a spacing that would never arrive, frozen wherever the gate stopped the frames. And
  leaving the IC editor marks every card, because save and cancel change no state at all
  and the cards have been detached for the length of the edit.
- `physics.js` — the shared RMHD kernels and the whole slice display chain, as
  templates over one constants object per app (`{pre, hasZ, wgReal, nDisp, arrow, ns,
  envFn}`), plus the shared `struct Mode` and `CMAP_WGSL` (four colormaps, one
  implementation, expanded from common.js's coefficient table at emit time):
  `prepGrads`, `bracket`, `nlAssemble`, `energyPartial`, `ou`,
  `scale`, `icFinish`, `prepDisp`, `vecMag`/`vecMagSq`, `maxSumPartial`,
  `sigmaCombine`, the arrow gather, the cut card's `cutPrep`, the colorize (whose shading
  — value→colour plus the contour overlay — is one fragment the 3D cube faces share),
  the contour level table and the blit (the 3D-only
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
- `devtools/` — the sandbox verification tooling (no GPU needed): the shared DOM+WebGPU
  stub every tool boots a real page on, the WGSL byte-diff harness, the fp64 physics and
  arithmetic checks, the clone detector and the layout audit. `devtools/README.md` says
  what each one covers; the standing rule is that a phase captures a FRESH WGSL baseline
  from clean git state before editing and diffs against it at the end.

## Cards: displays and charts

The UI is built from **cards**. A *display card* is one WebGPU canvas plus its own
quantity selector, colormap, arrows checkbox, contour-overlay selectors and — in 3D — its
own z slice / view select (manual slider, `track z⁺ / z⁻`, the same three with the
cube-faces view, or the field-lines view). Card index *is* the solver's display-chain
index, so N cards cost exactly N chains (scratch, bind groups, gather targets, textures),
each built lazily on first use.
*Any* card can be closed down to the last one (whose × goes disabled) — the IC editor has
its own view and anchors nothing. A *chart card* is one 2D canvas with a type selector — energy trace / spectra / cut — plus that
type's own options; several cards of the same type are
allowed and share one throttled readback per frame. Nothing here is a special case of
anything else: there is no dual-view flag and no fixed chart stack. (What each control
*does* is `docs.html`; below is why it is built this way.)

**Chart options** (each is a small select in the card's header; the first value of each
is the default and is what the chart drew before they existed):

| chart | option | values |
| --- | --- | --- |
| energy trace | which energies | `E_u, E_b` (with E_tot) · `E⁺, E⁻` (with E_tot) |
| spectra | which spectra | `E_u, E_b` · `E⁺, E⁻` · both |
| spectra (3D) | direction | ⊥ + ∥ · ⊥ only · ∥ only · ⊥ + k∥ (field line) |
| spectra | fit line | `pin to field` · `set A` · `off`, plus a numeric index `p` and amplitude `A` |
| spectra | pin / unpin | buttons, not selects: freeze the drawn curves as ghosts, or clear them |
| cut trace | component pair | `u_x, u_y` · `b_x, b_y` · `|z⁺|, |z⁻|` |
| cut trace (3D) | z source | manual slider · track z⁺ · track z⁻ |
| island width (2D) | — | log W(t) with a fitted γ = 2·d(ln W)/dt; needs the tearing IC (see below) |
| k_y = 2π/L_y mode (2D) | — | log A(t) of u_x / b_x at that k_y; needs the KH IC (see below) |
| eigenfunction ψ̂(x) (2D) | k_y bin | `1`…`6`, the box fundamental first; picks which column is read back |
| eigenfunction ψ̂(x) (2D) | fields | `ψ + φ` · `ψ only` · `φ only` |

**The spectrum fit line** (FEEDBACK item 8) is a straight E = A k<sup>p</sup> from just
above the forcing shell to the last bin, per CHART CARD, so two cards can carry two
slopes over the same spectrum. `p` defaults to −5/3 (the index box holds decimals, and a
value within `FIT_SNAP` of −5/3 or −3/2 snaps to the exact fraction and is *labelled* as
it, so the legend reads `k^-5/3` / `k^-3/2` and anything else reads as the number it is).
`pin to field` takes A off the spectrum itself, exactly as the old fixed guide did;
`set A` uses the amplitude box (A at k = 1) and falls back to the pinned anchor while the
box is empty; `off` hides the line and both boxes. NaN in either box is ignored.

**Pinned ghost spectra.** `pin` freezes the curves the spectrum card is *drawing* as faint
ghosts under the live ones, so the next thing you do can be compared against the last
thing you did (the workflows it exists for are in `docs.html`). It costs nothing on the
GPU — the snapshot is the CPU-side `[points, colour, dash, label]` list `specCurves` just
built, deep-copied.

Because it is the *drawn* curves that are frozen, a pin taken as `E⁺ / E⁻` stays E⁺/E⁻
however the card's own `sq` / `sd` selectors are switched afterwards: the ghost is a record
of a moment, not a live view. Details:

- **Per card**, like the fit line, and at most **4**; a fifth press is refused. `unpin`
  appears only when there is something to clear and clears all of them in one press
  (pins are cheap to retake).
- **Physical k.** A pin remembers the `kunit` it was taken under, and every ghost x is
  redrawn at `kunit_pin / kunit_live`, so changing the box (standard ↔ wide 4π × 2π) leaves
  the ghost at the same physical k instead of the same bin. An unchanged box is a factor of
  exactly 1. Resolution changes need nothing — the axis is log k/kunit out to the live
  `nb` and the clip rect crops any ghost tail beyond it.
- **Range.** Pinned ⊥ curves join the y-range pool and the `specFloor` knee walk exactly as
  live ⊥ curves do (otherwise the comparison can sit off the axis); pinned ∥ curves never
  stretch it, the same rule live ∥ obeys.
- **Drawing.** Under the live curves and under the fit line, at `lineWidth` 1 and
  `globalAlpha` 0.45 / 0.34 / 0.26 / 0.20 by age, each keeping its own hue and dash (so a
  pinned parallel spectrum stays dashed). Forcing markers and the fit line are not pinned.
  The legend gets **one** collapsed entry, `2 pinned @t=8.1, 12.3` — a 3D card on both × both
  draws eight curves, so one entry per pinned curve would cost eight legend items per pin.
- **Lifetime.** Pins survive parameter changes, IC resets, `chartsReset`, pause and
  rebuilds, and a preset switch transplants them onto the incoming spectrum cards
  positionally (`cardsLayout`) — which is what makes "pin the decayed spectrum, pick the
  forced preset" work. They are cleared by `unpin`, by retyping the card, and by closing
  it. Nothing is persisted across a reload. A card with ghosts but no live data yet draws
  axes and ghosts instead of `spectra — waiting…`.

The Elsasser energies are E<sup>±</sup> = E_kin + E_mag ± H_c with the cross helicity
H_c = ⟨u·b⟩, i.e. E<sup>±</sup> = ½⟨|z<sup>±</sup>|²⟩ and E_tot = (E⁺+E⁻)/2 — the
convention `taranis/physics/rmhd.py` uses for the forcing powers, which is what lets all
three curves share one axis. H_c costs nothing extra: it rides in the fourth (previously
zero) lane of `energyPartial`'s vec4 accumulator, and the spectra kernels bin it as a
third lane, so E<sup>±</sup>(k) = E_u(k) + E_b(k) ± H_c(k) needs no second kernel.

The **preset** dropdown in the sticky top bar (Run / Reset / t·step·dt·steps-per-second)
picks a whole configuration — controls *and* card layout. `?demo=NAME` is the same thing
as a deep link. Controls live in collapsible `<details>` groups (simulation /
dissipation / forcing / initial condition / displays & charts), open by default on a wide
viewport and collapsed on a narrow one; the state is deliberately not persisted.

## Display modes

Both apps: vorticity / current / phi / psi as signed fields (symmetric ±max), and the
perpendicular vector magnitudes "velocity |u|", "magnetic |b|" and the two Elsasser
fields "|z⁺|", "|z⁻|" (z± = φ±ψ, displayed as |ẑ×∇z±|), each with an optional ≤32×32
arrow overlay. Grid row y=0 is drawn at the top of the canvas.

The **cut trace** chart card is self-contained: it plots its own selected pair of
components along y at x = Lx/2 (of its own z plane in 3D — manual or tracked), and does
not depend on which display cards exist. It prepares its own line rather than gathering
a row out of a display chain, and does it without a 2D transform: e<sup>i k_x L_x/2</sup>
is just (−1)<sup>ix</sup>, so the k_x sum (and in 3D the k_z sum, with the plane's phase)
is analytic in one small kernel (`cutPrep`), and only the inverse along y is a transform
— four rows through the existing `rowsC2R`. The kernel always emits all four of
(u_x, u_y, b_x, b_y), so the pair selector, |z<sup>±</sup>| = |u ± b| included, is pure
CPU arithmetic on 4·ny numbers. Signed pairs are autoscaled symmetrically, the
magnitude pair to [0,max].

The **eigenfunction ψ̂(x)** chart card (2D only, EIGF_PLAN) is that card's transpose, and
the only chart with a readback of its own. It plots |ψ̂(x, k_y)| and |φ̂(x, k_y)| against x
at one selected k_y, on a **linear, autoscaled** y axis — the mode grows exponentially, so
the shape is the content, and a linear axis also makes |φ̂| = 0 on the resonant surface a
zero rather than a −∞ spike needing a floor clamp. The path is deliberately the cheapest
one that exists:

- `eigfGather` (`physics.js`) is a **pure gather**: the strided column m = ix·NKY + j₀ of
  φ and ψ into a compact 2·NX complex buffer (4 kB at nx = 256), one dispatch, no
  arithmetic, `fields` bound read-only, j₀ from a 16-byte uniform so the k_y selector costs
  a `writeBuffer` rather than a pipeline. It exists only on the 2D page.
- the inverse along **kx** is CPU work on those few kB (`common.js: eigfProfile`), through
  the same `fftPow2` the field-line spectrum uses — nx is a power of two in every box. The
  1/(nx·ny) is the full inverse-rfft2 normalization, so what is plotted is the coefficient
  c_j(x) of ψ = Σ_j c_j(x) e^{i k_y y}.
- `src: "eigf"` is a third throttled readback in the frame loop, on the cut line's idiom
  (100 ms, keyed on `stateMark()`), with the card's **k_y bin** where the cut card has its
  z plane: one gather per distinct bin on screen, so two cards on the same k_y cost one
  round trip and two on different ones cost two.

Two things it does *not* do. It never subtracts an equilibrium: every equilibrium seed here
is y-independent, so it lives entirely in the k_y = 0 column (the same fact `srcInit` uses
to read ψ_eq,k out of that column), and any other column has it removed exactly and for
free — and the column is the LIVE one, so once the run is nonlinear the profile is measured
against the state's own mean profile, flattened current and Reynolds-driven mean flow
included. And it quotes no Δ′: a legend readout fitting ψ̂′/ψ̂ either side of x₀ was drafted
and dropped in review (window-dependent, and not interesting enough at this resolution to
earn a legend slot) — the card is the plot alone. What is drawn is the **outer** solution;
the resistive layer is ~1 cell wide at the shipped presets (δ_lin/dx = 1.03 on `tearing`),
which is why neither the hint nor `docs.html` promises a layer.

**Colormaps** are per display card: `afmhot` (default), `viridis`, `RdBu`, `grayscale`.
One WGSL `cmap(x, which)` serves the slice colorize and the 3D cube colorize; afmhot is
matplotlib's exact closed form, viridis and RdBu are degree-6 least-squares fits of each
channel to matplotlib 3.10's tables sampled at 256 points (max abs channel error 0.017
and 0.040; both clamped, since a fit overshoots slightly at the ends). The coefficients
live once, in `common.js: CMAP_COEF` — `physics.js` expands them into the WGSL at emit
time and the IC editor's CPU preview reads the same table through `cmapRGB`, so the two
cannot drift.

**The colorbar** (FEEDBACK_2026-08-10 item 12) is a third consumer of that same table, and
deliberately not a fourth code path: the STRIP is always the full colormap swept over
t ∈ [0,1], because that is exactly what `dispX` produces for every mode (signed −s…+s →
0…1, magnitude 0…s → 0…1, σ_c/σ_r −1…+1 → 0…1), so one 132×9 2D canvas painted through
`cmapRGB` serves all ten modes and **no WGSL changes** — the gate on the whole feature was
both dumps staying byte-identical. Only the three LABELS branch, on the same three
classes. It lives at the right of the card's caption line (`.viewfoot`, which wraps), so
it never covers the field canvas, and it is hidden in the 3D lines view, which renders no
colour field at all.

The number it quotes is the one the kernel used, not a CPU estimate of it: the autoscale
lives only in the per-chain `maxVal` buffer that `maxFinal` writes every render (the same
buffer in both apps, and the three-face maximum in the cube view — which is what those
pixels were divided by too), and no existing readback carries it, `readStats` being
energies rather than extrema. So the frame loop takes its own, on the arrow readback's
snapshot-and-guard idiom: 4 bytes per card at `CBAR_PERIOD` = 350 ms, an order of
magnitude cheaper than the 8 kB arrow gather beside it, skipped entirely for the σ modes
(fixed ±1, no autoscale exists) and while the editor view owns the screen. A field change
drops the stale range rather than relabelling the new mode with it.

**Save / record** (item 13) are per display card, in the same footer. `save` composites
the WebGPU canvas, the overlay canvas and the colorbar into an offscreen 2D canvas and
hands it to `toBlob`; the card **re-renders first**, because WebGPU has no
`preserveDrawingBuffer` — `getCurrentTexture` is transient and the canvas holds only its
last *presented* image, so the capture has to be taken in the same task as a fresh
present. `rec` records the field canvas (that canvas alone: the arrows, the field lines
and the colorbar are in the PNG only) with a 30 s hard stop, on whichever of **two legs**
the engine supports.

**Leg 1, WebCodecs (preferred).** A `VideoEncoder` configured as `avc1.4200<level>`
Constrained-Baseline-class H.264, `avc: {format:"avc"}` (so the chunks are length-prefixed
samples and the first chunk's `metadata.decoderConfig.description` is the avcC payload the
`stsd` box needs), 5 Mbit/s constant, level picked from the frame's macroblock count.
The frames come off **the frame loop itself**: `loop()` calls each display card's
`recCapture()` in the same synchronous task as that card's `render()`, before any `await`
— WebGPU has no `preserveDrawingBuffer`, so a capture deferred even one microtask would
wrap an expired `getCurrentTexture` (the same reason `save` re-renders). A **slot cadence**
keeps it at 30 fps: `W.due` is the next slot's wall clock, a callback before it captures
nothing (a 120 Hz phone captures on about every 4th and drops nothing), and a loop so late
that whole further slots went by counts those into `W.drop` and re-bases `due` on *now* —
never backfilling, since a backfilled frame would sit at a timestamp it never had.
**What a due slot does** is the RECASYNC part (2026-08-12): building a `VideoFrame`
*from the canvas* is a synchronous readback on the main thread — 15–17 ms per capture on
a real iPhone (`?recdebug`), which pushed every slot-due pass past its vsync window and
was the stutter itself. So on an engine whose WebCodecs can build a `VideoFrame` from
**bytes** (probed once with a 2×2 frame, `recBufOff` latch, degrade silently), a due slot
only *submits* a GPU-side `copyTextureToBuffer` of the canvas texture into one of three
staging buffers (the one always-on cost: every card's context is configured with
`COPY_SRC`) and returns — microseconds. The bytes arrive when `mapAsync` resolves, a beat
later, and are encoded inside one **ordered promise chain**, where frame index, timestamp
`round(n·10⁶/30)` µs and `keyFrame: n % 30 === 0` are all assigned at *encode* time — map
resolution order across buffers is nothing to rely on, `VideoEncoder` wants monotonic
timestamps, and `mp4Mux` writes a uniform `stts`, so a dropped or failed capture leaves
fewer frames and never a hole. No free buffer (three maps in flight), a mid-take canvas
resize, a failed copy, or `encodeQueueSize` past `REC_QMAX` all *drop* the slot without
advancing `n`: the timestamps stay an exact 1/30 s apart and a slow machine records fewer
seconds of wall clock rather than a sample table that lies about its timing. Stop drains
the in-flight captures first (500 ms cap, so a hung map cannot hold the file hostage,
and the stragglers count as drops), then flushes, then closes the encoder, frees the
pool and muxes — on every route out. Engines without the
bytes constructor keep the old direct `VideoFrame(canvas)` path unchanged.
The `setInterval` at 1000/30 ms is still there but **demoted to a watchdog**
(RECRAF_PLAN, 2026-08-12): it re-renders and encodes only where the rAF feeder is
**known-absent** — `document.hidden` or the editor view. Round 1 parked it on a timing
heuristic (no capture for ~117 ms) and an on-device test showed why that fails: a loaded
phone's loop gaps beat any threshold, waking the watchdog *alongside* a perfectly live
rAF feeder — both feeders at once, more main-thread work than the old recorder. So the
watchdog now keys on the condition it exists for, i.e. exactly where there is no rAF to
ride — a
background tab, whose rAF is throttled to a stop, the editor view, which renders no cards,
and the headless stub. It was feeding leg 1 from that timer that made a live recording
stutter visibly on an iPhone: an *extra* full `render()` per tick plus the frame copy and
the encode, all on the main thread and at an arbitrary phase against the rAF loop. Riding
the render costs zero extra renders, and the watchdog's second render now recurs only where
there is no visible display to stutter. A watchdog-fed frame re-bases `W.due` too, so the
slots it put in the file are never double-booked as drops when the rAF loop resumes. The
watchdog keeps the *synchronous* canvas capture even when the buffer path is on: it renders
off-screen anyway, so a stall costs nothing visible, and the async pool stays out of the
background-throttled world where a hidden page starves maps of callbacks. On stop (button,
timer, `destroy()`, encoder error) it drains, flushes and `mp4Mux` writes the file.
`?recdebug` in the URL adds one readout line per live recording — frames fed by rAF vs
the watchdog (wd must stay 0 on a visible page), drops, the longest gap between loop
passes, and the capture cost split: `vf` is the max ms a capture cost the MAIN THREAD
(copy+submit on the buffer path — expected ≲1 — or the full `VideoFrame(canvas)` readback
on the sync path), `enc` the max ms in the encode step (on the buffer path that includes
building the frame from bytes, off the hot path in the chain), and `lag` the max
capture-submit→encode delay — tens of ms are fine, frames are stamped by index, not
arrival; it stays 0 on the sync path. This is how a phone, with no devtools console,
reports what a stutter is made of. Deliberately absent from `docs.html`.

**Why we mux it ourselves.** Chrome's `MediaRecorder` `video/mp4` is a *fragmented* MP4:
`moov` + `mvex`, one `moof` per fragment, `trun default_sample_flags 0x10000` — "not a
sync sample, dependency unknown" — and a keyframe only every ~1.4 s. Desktop players
ignore the flags; iOS AVFoundation believes them, drops every delta sample, and plays a
30 s recording as about three stills. The codec was never the problem. `mp4Mux` writes the
plain progressive shape instead: `ftyp` + `mdat` + `moov`, real `stts`/`stss`/`stsc`/
`stsz`/`stco` tables (media timescale 1000·fps, so one frame is exactly 1000 ticks),
`avcC` verbatim in `stsd`, `stco` 32-bit with an assert instead of a `co64` branch that
30 s could never reach, and no `moof`, `mvex`, `trun`, `ctts`, `edts` or `sdtp` anywhere.
`moov` comes last, which is why `mdat`'s offset is known when the tables are built.

**Leg 2, `MediaRecorder`** over `captureStream(30)`, unchanged, for engines without
WebCodecs: MP4 (H.264) where `isTypeSupported` allows it, WebM otherwise, `onstop` the
single write path. It is also where leg 1 bails to, mid-press and for the rest of the
session, if a first chunk ever arrives without a `decoderConfig.description` — without an
avcC there is no playable progressive file to write. The button is shown when **either**
leg can run and is simply absent otherwise. Filenames are
`taranis-<page>-<field>-t<time>.{png,mp4,webm}`.

`devtools/stubenv.js` stubs `toBlob`, `captureStream`, `MediaRecorder`, `VideoEncoder` /
`VideoFrame` / `EncodedVideoChunk`, `Blob` (keeping the *bytes*), `URL.createObjectURL`,
and `setInterval` as a hand-driven pump (`env.tick(n)`) with `env.fireTimeout(ms)` for the
30 s cap — so `bootstub.js` runs the whole WebCodecs path headlessly (pump, forced
keyframes, backpressure, flush, an `ftyp+mdat+moov` download with no `moof`, the 30 s cap,
destroy-mid-record, the no-avcC bail) and then the same MediaRecorder legs as before. Its
stub `requestAnimationFrame` is a no-op and its `performance.now` jumps 250 ms per *call*,
so `env.tick(n)` always drives the **watchdog** path; the rAF-side feeder is driven by
calling `recCapture()` directly, which is how the slot cadence, the skipped-slot counting
and the handoff in both directions are covered.
`devtools/checkmp4.js` drives `mp4Mux` with **real** H.264: ffmpeg encodes a test pattern,
the script cuts it into samples and builds the avcC (the only Annex-B code in the project,
and it is in the test), and ffprobe/ffmpeg check the result — top-level boxes, sync samples
exactly on the forced indices, equal pts deltas, `30/1`, and a decode with zero errors, for
a square canvas, the 1024×256 wide box and a one-frame file.

**Contour overlays** are per display card too: ψ contours (= the perpendicular magnetic
field lines), φ contours (= the streamlines), or **both at once** (ψ + φ — the alignment
view), at 8 / 16 / 32 levels, over whatever field the card displays or over a **plain
background** (the second per-card select: contour ink on a blank plate, when the colours
underneath are in the way). They are drawn inside the shared colorize kernel, not as a
second pass: each active set's potential reaches the display scratch through ONE
extra inverse transform per card frame (reusing scratch that is dead by then — the second
component's, and for the second set the σ_c half-1 buffer), and a texel is inked when
`floor(pot/Δ)` differs from that of its +x or +y
neighbour — a crossing test, so no derivatives and nothing a compute shader cannot do.
Δ is **uniform**, so the line density is proportional to |B⊥| (or |u⊥|), which is the
honest picture rather than a prettier equal-area one. Δ = 2·range/nlev with the range
adapted on the GPU (up at once, down by 5% of the gap per frame) so the lines do not
flicker from frame to frame, and each set has its own range (ψ and φ are unrelated in
size). The first set's ink is black over a light background and white over a dark one, so
it survives every colormap and the plain plate; the second set is a fixed magenta accent
that none of the four colormaps produces, so the two sets are always told apart. On the
cube view the contours are drawn on the **top face only** — that is the one face whose
in-plane potential the chain already has.

**Cross helicity σ_c** = (|z⁺|²−|z⁻|²)/(|z⁺|²+|z⁻|²), pointwise, is the one mode with a
**fixed** colour range (±1, symmetric — no autoscale, so colours are comparable
between frames and between runs). Where the local Elsasser energy density |z⁺|²+|z⁻|²
falls below 1e-4 × its maximum over the displayed field, σ_c is rendered as exactly 0:
in quiet regions the ratio is pure noise. It costs four inverse transforms per frame
(both components of both z±) instead of the vector modes' two — display cost only, no
physics buffer is touched.

**Residual energy σ_r** = (|u|²−|b|²)/(|u|²+|b|²) is the same two-half machinery with
the u and b vectors as the pair, and shares σ_c's fixed ±1 range and relative
quiet-region floor. Both apps offer it, and in 3D it goes wherever σ_c goes — a z slice
or the three cube faces — since the two modes differ only in which pair of vectors the
chain's two halves are pinned to (the floor is then the maximum over that same target:
the plane in a slice view, the three faces together in a cube view). Note the floor's
rendering convention: quiet pixels show the neutral mid-colour, which for σ_r is also
the equipartition colour — a grey quiet region means "too little energy to measure",
not "measured equipartition".

3D only: **cube faces** are a *view*, not a field — three of the last four entries of a
display card's z-source select ("cube faces", "cube + track z⁺", "cube + track z⁻"; the
fourth is the field-lines view below), orthogonal
to the field selector, so **any** quantity (σ_c, σ_r and the vector magnitudes included)
can be drawn as a cube. It draws the three visible boundary faces of the box in the oblique
view `examples/forced-turbulence-3D.ipynb` uses (matplotlib `view_init(elev=30,
azim=45)`), colorized through the same shared shading as a slice (so each mode keeps its
own colour range) with one common autoscale across the three faces, depth-cued by a
per-face darkening of 1.0 / 0.85 / 0.7. The **top face is the card's own plane** — its
slider in manual mode, its tracker otherwise, so a cube card riding a packet puts the
collision front on top; the two side faces stay x = Lx and y = Ly boundary slices. On the
vector modes the arrow overlay is drawn on the top face, the same anchors and directions
put through that face's affine projection (CPU-side, on the overlay canvas; side faces
get none — u⊥/b⊥ is not tangent to them). Any display card may take the view (each chain
owns its own three face buffers and three face textures — a rounding error next to its
nz·nx·ny scratch volume); the cut chart does not offer it. The 3D spectrum
card also carries the parallel spectra as dashed curves (|kz| bins 1…nz/2, ±kz paired,
kz = 0 omitted from the log axis); **the y limits are set by the perpendicular spectra
alone**, so the dashed curves are plotted inside that range and never stretch it (unless
∥ only is selected, when there is nothing else to scale to).

### 3D field lines, and k∥ measured along them

**Field lines** are the last *view* in a display card's z-source select, next to the cube
entries (and, like them, not offered by the cut chart). It traces a fixed 8×8 grid of
magnetic field lines seeded on the bottom face and draws them — with the twelve box edges
— through the same oblique box projection on the card's overlay canvas. The march is a compute
kernel solving dx⊥/dz = b⊥/B₀ with B₀ = v_A = 1: RK2 midpoint (the midpoint field is the
mean of the two bracketing planes, so it is second order in dz with no extra storage),
bilinear in plane, uniform dz, periodic ⊥ wrap. Its volume is the RHS's *own* gradient
prep — ∇⊥φ and ∇⊥ψ in real space — so b⊥ = ẑ×∇ψ needs no new gradient or FFT kernel, and
the volume never leaves the GPU: what comes back is the **polylines only** (N_lines × nz
positions plus the same number of (u, b) samples, tens of kB). It runs at ~2 Hz, not per
step, in its own submit. There is no depth sorting and no occlusion: a 2D canvas, one
pass, every line visible. A line that leaves the box is drawn wrapped, with the pen
lifted at the seam.

Behind the lines the GPU canvas carries no field: it is cleared to the contour plate and
draws the box's **top boundary plane as contour ink only**, through the cube projection
and the same shared colorize kernel — every texel of that face which is not ink is exactly
the background colour, so the face is transparent with no plate, no blend state and no
second kernel (the whole mode chain is skipped in this view, since nothing reads its
colour). The card's own contour selectors stay live and drive that face — ψ (the default
on entering the view), φ, both, or off — so the line endpoints are seen puncturing the
exit plane on its own ⊥ structure. The field selector and the arrow overlay are inert
here, and so are the tracker and the z slider: the view is the whole box, and its face is
the top boundary, not a tracked plane.

The same march samples (u, b) **along** each line, which is what the spectra card's
`⊥ + k∥ (field line)` option bins: the true parallel spectrum, measured along B rather
than along the z coordinate. The samples are uniform in z (arc length = z to leading
order in RMHD) but not periodic — a line exits perpendicularly displaced — so each line
is Hann-windowed before the transform, the window's mean square divides back out (so
Parseval still holds and the numbers mean the same thing as the coordinate E(k∥) plotted
beside them), the ±kz bins are folded, and the ensemble of lines is averaged. The three
binned lanes are the usual [E_u | E_b | H_c], so E<sup>±</sup>(k∥) needs no separate
path, and the y-limit rule is unchanged: the ⊥ spectra set the range. The two consumers
are independent: with such a chart open and no card in the lines view the lines are still
traced and simply not drawn, and the along-line samples — four times the polylines' size —
come back only while that chart is open.

## Initial conditions

The solvers know exactly two ICs: the built-in large-scale mode pattern (`setIC()`, what
the reference vectors use) and an exactly quiescent state (`setIC(true)`). Everything
else is built on the CPU and uploaded through **`setICFromReal(phi, psi)`** — real-space
`Float32Array`s in the buffers' own layout (`ix*ny + iy` in 2D, `(iz*nx + ix)*ny + iy` in
3D) — which forward-transforms them and applies the 2/3 dealias exactly like the
constructor does (unmasked beyond-cutoff IC energy would persist and alias). The **IC**
selector then offers `large-scale modes`, `quiescent`, `letters` and `custom` (plus the
two equilibrium presets in 2D, and the sinusoidal packet pair in 3D); **Reset** re-applies
the current one.

### ζ±, amplitudes, and where the normalization happens

What the letter and blob ICs build is the Elsasser **potential** ζ±; the Elsasser
*fields* are z± = ẑ×∇ζ±, and the evolved variables are φ = (ζ⁺+ζ⁻)/2, ψ = (ζ⁺−ζ⁻)/2.
The UI says ζ± wherever a potential is meant (paint target, amplitude sliders) and z±
wherever a field is (display modes, tracking) — except in the (φ, ψ) paint basis below.
Identifiers stay `zp`/`zm`.

Stored ICs are kept at their natural scale and normalized **only at apply time**, by
`common.js: icZetaFields`: each potential is zero-meaned plane by plane (a pure gauge for
a perpendicular gradient) and scaled so that max |ẑ×∇ζ±| over the volume equals **its
own amplitude slider** — ζ⁺ and ζ⁻ have separate sliders, linked by a lock checkbox that
is on by default. Because the stored arrays are never mutated, pause → move a slider →
**Reset** genuinely rescales the same drawing, any number of times (before Phase G the
amplitude was baked in per blob at deposit, and Reset re-uploaded the old one).
Consequences worth knowing: the amplitude is a property of the whole drawing, not of one
blob, so overlapping strokes redistribute instead of stacking past it; and a ζ⁺ drawing
and a ζ⁻ drawing are normalized independently, so their *relative* size is set by the two
sliders, not by how hard you scribbled.

**The (φ, ψ) basis.** *Which* pair those two sliders normalize follows the **paint**
target (`icAmpBasis`, and only while `custom` — the drawing — is the selected IC, since
nothing else has a paint target). Painting ζ⁺/ζ⁻ keeps the historical basis above.
Painting φ or ψ switches `icZetaFields` to normalize the *combinations* instead: the
labels become **φ amp** and **ψ amp**, and they mean max |∇φ| = max |u| and
max |∇ψ| = max |b| — the peak flow speed and the peak perpendicular field strength,
independently. Both are exact linear maps of the same stored ζ± pair, and a drawing with
only φ strokes (or only ψ ones) comes out *bitwise identical* under either basis with the
amp lock on. The basis exists for the mixed case: a drawing carrying both φ and ψ strokes
stores ζ± = P ± Q, so normalizing ζ± can only rescale an inseparable mixture — "a strong
vortex plus a weak island" is not expressible in that basis at all, and it is what the
`rmhd variables` demo closes with. χ (3D) is written in Elsasser amplitudes, so its
readout goes through `icAmpZeta`, which maps a (φ, ψ) pair back to
(a⁺, a⁻) = (a_φ+a_ψ, |a_φ−a_ψ|) — exact for co-located strokes of the same shape,
an estimate otherwise, like the k̄⊥ beside it.

The **letters** preset builds one potential per field from a rasterized glyph
(A → ζ⁺, B → ζ⁻; fixed, `common.js: IC_LETTERS` — the free-text input was dropped in the
mobile pass): the glyph is drawn on an offscreen canvas at 60% of the box and
gaussian-smoothed at a **physical length σ_letter = Lx/32** (`IC_SIGMA_PERP_FRAC`, the
single constant that is also the blob-width slider's default), via
`ctx.filter="blur(Npx)"` with N = σ_letter/dx where the engine supports it and an exact
separable periodic gaussian otherwise. Specifying the blur as a length, not a pixel
count, is what makes the letters resolution-independent: identical k⊥ content, hence
identical χ, at 128² and 512² (node check: normalized E(k) agrees to 3% rel L2, k̄ to
0.2%). In 3D each field is additionally multiplied by a gaussian z-envelope of peak
exactly 1 to make a *wave packet* (σ_z from its own slider, see below). The smoothing is
not cosmetic: the spectral gradient the GPU takes of an unsmoothed glyph edge overshoots
the finite-difference one used for the normalization by ~19%, and at σ ≥ 2 px they agree
to ~0.1%, which is what makes the amplitude a statement about the displayed field. (The
dealias at upload still trims it slightly, so the knob is approximate.)

## Drawn ICs: the blob editor (`custom`)

Selecting **custom** shows one extra row of controls; **edit IC** pauses the run and
switches the main area to a **dedicated editor view** (`#editview` replaces `#display`).
The view is one empty host div per app, filled by `common.js: icEditBuild` exactly as the
card system builds its cards — so its canvas is an ordinary card canvas, nothing is ever
painted over a live display, and no display card is special because of it (the 3D z-plane
slider is a flag on the hook object, not a second copy of the header). Click or drag and
each sample
deposits a **periodic gaussian blob** into the chosen target potential — `ζ⁺`, `ζ⁻`, `φ`
or `ψ`. The editor keeps only the two potentials on the CPU: a φ blob is one blob in
*both* ζ±, a ψ blob one of each sign. Leaving the view is one of

- **save & run** — keep the drawing, apply it (`icZetaFields` → `setICFromReal`), resume;
- **save** — keep it as the **Reset** target and stay paused;
- **cancel** — restore the snapshot taken when the editor opened (everything drawn since
  is discarded);

and **clear** empties the drawing without leaving (there is no undo stack). Nothing
touches the GPU while editing — the canvas is redrawn from the CPU arrays (the signed
mapping of the `colorize` kernel, in the first display card's colormap, autoscaled) at
most once per animation frame. The drawing is dropped when the grid changes (resolution,
or Lz in 3D) because its layout is per-grid.

- **Amplitude** is NOT a deposit knob: blobs go in at unit amplitude (peak P = 1·σ·√e for
  f = P·exp(−r²/2σ²), whose |∇f| peaks at P/(σ√e)) and the ζ± sliders scale the finished
  drawing — see the normalization section above.
- **σ⊥** (and **σ_z** in 3D) are fractions of the box, floored at 2 cells — anything
  narrower is eaten by the 2/3 dealias at upload. Their sliders' range, step and default
  come from the shared constants (`icSigmaSliderInit`), so σ⊥ defaults to exactly
  σ_letter and σ_z cannot exceed the packet cap Lz/12.
- **Sign**: the "negative" checkbox XOR the right mouse button, so a right-drag is a
  negative stroke without touching the controls.
- Blobs are truncated at 5σ (keeping 1−4·10⁻⁶ of the gaussian) and wrapped with the
  minimum-image convention `icGaussZ` already used, so a blob painted on the box edge is
  exactly a translate of one painted in the middle.
- **3D**: each blob is a perpendicular gaussian times a z-envelope of peak 1 centred on
  the plane the editor's **own z-plane slider** selects (opened on whatever plane the
  first display card was showing). The cube view is irrelevant now — the editor no
  longer borrows a card's slice, so editing is never refused.

Mouse → grid: `getBoundingClientRect` is in CSS pixels, so the responsive canvas width and
the device pixel ratio both divide out. The display puts grid point i at screen fraction
(i+0.5)/n — texel centres as the linear sampler sees them, the same convention
`drawArrows` uses to place its cells — and row iy = 0 is at the **top** (the render pass
samples v = 1−uv.y). So the continuous grid coordinate under the cursor is u·n−0.5 and the
nearest grid point is round(u·n−0.5); the deposit uses the continuous position, wrapped
into [0,L).

## Presets (the dropdown, and `?demo=`)

A preset is a **configuration**, never new physics. A registry entry carries `set`
(control id → value), `prep` (the "auto" seeds), `layout` (the display + chart cards it
wants) and a one-line `hint`. `presetBoot` (common.js) fills the dropdown, preselects
`?demo=NAME` if the URL carries one, and writes the controls *before* the first solver is
built; picking another entry from the dropdown runs the same path plus a rebuild.
Everything stays adjustable afterwards. There are no separate demo pages.

- **2D `decaying A / B packets`** (`rmhd2d.html?demo=decay`) — letters A/B at 512²,
  forcing off, hyper=4 seeded at ν_marg and then driven by auto-diss (see
  *Dissipation* below), two displays showing |z⁺| and |z⁻|. Watch the
  spectrum settle onto the quasi-universal decaying slope, and switch a display to σ_c
  for dynamic alignment. Its hint states the one thing the amplitude slider does *not*
  do here: ideal 2D RMHD is self-similar in amplitude ((a,t) → (a/λ, λt)), and dt is
  adaptive, so the per-frame dynamics are amplitude-independent — amplitude only matters
  against the fixed dissipation. (In 3D it matters through χ.)
- **2D `rmhd variables (φ, ψ)`** (`rmhd2d.html?demo=rmhdvars`) — the only preset that
  opens on an **empty** drawing: 256², forcing off, fixed ν (no auto-diss — a laminar
  vortex has nothing at the dissipation scale to measure), IC `custom` with **paint** on
  φ and a big blob (σ⊥ = L_x/8), one display of φ with its own contours and no chart
  cards. The user's single click *is* the initial condition, and the hint walks the
  variables from there: φ blob → a stationary vortex whose φ contours are the streamlines
  of u; the same field as vorticity ω = ∇²φ; then the same click in ψ → a magnetic island
  whose ψ contours are the B⊥ field lines; then a strong φ blob plus a weak ψ one, which
  is what the (φ, ψ) amplitude basis above exists for.
- **2D `Kelvin–Helmholtz shear layers`** (`rmhd2d.html?demo=kh`) and
  **2D `tearing mode`** (`rmhd2d.html?demo=tearing`) — the two equilibrium demos; see
  the next section.
- **3D `Alfven-wave collision`** (`rmhd3d.html?demo=collision`) — 64²×256, Lz = 8π,
  forcing off, two counter-propagating **sinusoidal** packets, on one volume view of j∥.
  (It opened on the letter packets until the on-device pass: the sinusoids fill the box,
  so their shells read as surfaces rather than as two blobs in a long column. The letters
  are one entry away in the IC dropdown, unchanged.)

## Equilibrium demos: Kelvin–Helmholtz and tearing (2D)

Unlike every other IC these two are **equilibria plus a seed**, so their knobs are
physical (U₀, b₀, ψ₀, the layer width a, the seed amplitude) and they skip the ζ±
normalization entirely — rescaling an equilibrium to a fixed max |∇ζ| would not be the
equilibrium anyone asked for. They are registered exactly like the letters and the
drawing (`icRegister` in common.js: `rows` = the control rows the preset shows, `hyper` =
the exponent it locks, `src` = whether it offers the maintained-flux source, `fields(g)` =
the (φ, ψ) pair), so adding a third costs one record.

The two SHIPPED presets run in a **rectangular box**: 512 × 128 on 4π × 2π, chosen by the
`box` select next to the resolution (which now means n_x — n_y and both box lengths follow
the box). A long x holds the layer's far field inside the periodic box, a short y carries
exactly one unstable wavelength k_y = 2π/L_y. (The two TEARNL presets below take the other
two boxes: `collapse` the square one, `chain` the large 8π × 8π.) Rectangular boxes are 2D-only; everything downstream
that used to assume a square perpendicular plane is now min-based or aspect-aware
(the shell-bin count `nbins`, the arrow subsample, the display and editor canvases,
which get equal pixels per unit *length* so that contours and arrows are not sheared).

- **KH**: u_y(x) = U₀[tanh((x−L_x/4)/a) − tanh((x−3L_x/4)/a) − 1] — two layers of
  opposite sign, which is what periodicity forces, and independent of each other while
  a ≪ |x₂−x₁| = L_x/2 (the preset's default is L_x/2 = 10a). ψ_eq is the same profile
  scaled by b₀, i.e. an in-plane field along ŷ. The potentials are the analytic
  antiderivative, a·ln cosh(·) written overflow-safe, so u_y and b_y are exact to
  round-off. Ideal 2D MHD stabilizes the layer at **b₀ ≥ U₀** (the shear is then slower
  than the Alfvén speed tying the field lines together); dissipation softens that
  threshold rather than removing it.
- **Tearing**: ψ_eq = ψ₀ sech²((x−L_x/2)/a) (Numata/Loureiro-style — net-flux free, and
  periodic to O(e^{−L_x/2a})), φ_eq = 0. b_y = ψ_eq′ **vanishes on x = L_x/2**, so that
  line is the resonant surface of every k_y mode; the seed perturbs ψ there with the same
  even-in-x envelope, so ψ̃(x_s) is exactly the seed slider. L_y/a sets Δ′a (8.40 at the
  defaults, from a shooting solve of the outer equation — `devtools/eqlinear.py`).

**hyper is LOCKED to 1 by the tearing preset** (the select goes disabled and shows why):
hyper-dissipation has no resistive layer and no Rutherford stage, so it would falsify
exactly the physics that demo exists to show. KH does **not** lock it — it is an ideal
instability, and hyper is a legitimate way to sharpen its secondary structure.

**Maintaining the equilibrium flux.** Tearing carries a `maintain equilibrium flux`
checkbox (on by default) that adds the static source **S = −η∇²ψ_eq** to the ψ equation,
cancelling the equilibrium's own resistive decay so that the demo shows the instability
and not the layer spreading. ψ_eq,k is extracted once per Reset from the k_y = 0 column of
the uploaded IC — which *is* the equilibrium, every seed here having zero mean along y —
by the tiny `srcInit` kernel, and `nlAssemble` then adds −lin_L·ψ_eq,k: the SAME diagonal
the stage applies, so the source follows the η slider with no bookkeeping of its own and
uses η (ψ's coefficient), never ν. Like Pm it is emitted at WGSL-generation time and only
when the preset asks, so every other path keeps its byte-identical kernel text. With the
source on the measured growth rate is the frozen-equilibrium eigenvalue (0.0284 vs
0.028716 at the benchmark, `devtools/checkj.js` §4b) and the maintained ψ_eq is stationary
to round-off; with it off, see the caveat under the table below.

**The nonlinear pair: `tearing: X-point collapse` and `tearing: island chain`** (keys
`collapse` / `chain`; TEARNL, 2026-08-13). Two more presets on
the SAME tearing equilibrium and with no new physics — only the controls move — sitting
either side of the instability's nonlinear fork. All three tearing presets run at
`selRes` 256 and open on **island width + cut trace b_x/b_y**, except `chain`, which has no
island chart (see the broadband seed below) and opens on **cut b + cut u** instead. Both run the maintained-flux source
**off** (neither quotes a rate against a held equilibrium, and a source feeding flux in
would make the collapse externally driven) and auto-diss off (the controller places the
*cascade termination* at the dealias scale and has no term that knows a reconnection layer
exists; on a quiescent equilibrium it drives η to ~1e-9).

- **`collapse`** — the square box at a = 0.1L_x, so Δ′a = **37.8** against the shipped
  preset's 8.4. The island passes the critical width W_c ~ 1/Δ′ immediately and the
  X-point collapses into a sheet, which itself tears and sheds a secondary island
  (Loureiro et al., *PRL* **95**, 235003 (2005)). Its `a / L_x` slider walks Δ′a from 37.8
  down to 8.40 at a = 0.2, which is the shipped preset's own value — 0.2×2π is its
  0.1×4π, and Δ′a depends only on k_y a in this profile, so the two are *exactly* equal
  and the preset contains `tearing` as a slider endpoint.
- **`chain`** — the LARGE box (8π × 8π, i.e. the square one four times bigger, not a long
  thin one), seeded broadband (below), so the layer selects its own k_y out of 24 offered:
  Δ′ falls monotonically with k_y while the resistive layer gets faster, and the two peak
  together at k_y = 1.5, i.e. six islands, which then coalesce in pairs. Expect *roughly*
  six rather than exactly six, and expect it to move with `#nSeed`: γ is flat near its peak
  (modes 4–8 within 9%), so over the linear stage the winner leads its neighbours by only
  ~7% and the phases settle the rest. What the preset actually demonstrates is that the box
  FUNDAMENTAL loses, its γ being 3× down. Square cells are mandatory and not a nicety: the
  Sweet–Parker sheets between merging islands lie normal to y. L_x/a = 53 because an
  island's extent across the sheet grows with its wavelength along it and each merger
  doubles that wavelength — size the box on the LAST merger, not the first.

**Broadband seed.** `rowTear` carries a `broadband seed` checkbox (default off, so the
shipped preset is untouched). On, the seed's y factor is Σ_{n=1..N} cos(nk₁y + φ_n) at
equal amplitude and random phase instead of a single cos(k₁y), normalized so its maximum
over the grid is 1 — which is what keeps the `seed` slider meaning the same physical thing
(peak perturbed flux at the resonant surface) in both branches, the x envelope being sech²
and hence 1 there. N is derived, not a constant: Δ′ > 0 below k_y a = √5 (an analytic
property of the sech² outer solution, independent of a and of the box), and N covers that
band with a quarter of headroom, which is 24 at the `chain` parameters. The phases come
from the page's own `#nSeed` stream, so the IC stays reproducible. The island-width chart
turns **off** for a broadband seed (`icEq.on` stays false): W = 4√(ψ̃/|ψ″|) is a
single-mode formula read off max − min of ψ on the resonant line, and with 24 modes that
extremum belongs to whichever island is largest at that instant — after the first merger
not even the count is fixed.

**Per-field dissipation (Pm).** The 2D linear operator is diagonal per field, so ν (on φ)
and η (on ψ) need not be equal: the **diss slider is η**, and the `Pm` box next to it is
the magnetic Prandtl number ν/η, default 1 (Pm = 0 is allowed — a completely inviscid φ,
which in a long run piles kinetic energy up at the grid scale). It is a compile-time
constant of the stage kernel — at Pm = 1 the emitted WGSL is character-for-character the
scalar-dissipation text every other path uses, and changing it rebuilds the solver like a
resolution change. Exactly two kernels carry it: `stage` (the φ half of the integrating
factor) and `energyPartial` (the kinetic half of the dissipation-rate lane). 3D keeps
ν = η — its 2×2 Alfvén propagator needs an equal diagonal.

**Island width.** The `island width` chart card (2D only) plots W(t) on a log axis, so
the linear tearing stage is a straight line, the Rutherford stage bends over, and
saturation flattens. Near the resonant surface
ψ ≈ ψ_s + ½ψ″(x−x_s)² + ψ̃ cos(k_y y), so the separatrix half-width obeys
½|ψ″|w² = 2ψ̃ and

> **W = 2w = 4·√(ψ̃/|ψ″|) = 4·√((ψ_X−ψ_O)/2|ψ″|)**

with ψ_X−ψ_O the peak-to-peak of ψ along x = L_x/2 and ψ″ **measured** on the
equilibrium profile (a 4th-order second difference at x_s, on the grid the run actually
uses — not the analytic 2ψ₀/a²). The card costs no kernel and no extra round trip: the
cut chart already reads b_x = −∂_yψ on that line at ~10 Hz, and ψ along it is one
spectral integration of that line (`icLineIntegrate`, exact for a band-limited periodic
line where a quadrature rule would be second order; the k = 0 gauge drops out because
only max − min is used).

Its legend carries the same fitted growth rate the k_y mode card does, through the same
helper (`fitLogSlope`): the trailing `MODE_FIT_DT` = 10 t-units of ln W, least squares,
R² ≥ `MODE_FIT_R2`, quoted as **γ = 2 × d(ln W)/dt** because W ∝ ψ̃^½ ∝ e^{γt/2} in the
linear stage. Only the *rise* gate is its own number (`ISLAND_FIT_RISE` = 0.05 against
`MODE_FIT_RISE` = 1.0): the tearing demo runs an order of magnitude slower than KH *and*
plots the square root, so one window of its linear stage rises γ·Δt/2 = 0.14 ln-units,
19× less than KH's 2.67 — the KH gate would blank the legend for the whole run. 0.05 is
`MODE_FIT_RISE` scaled to the tearing rate, keeping the same ~2.7× margin over the gate,
which is still ~1.9× with **maintain equilibrium flux** off (where the measured slope
drops 30–40%). It is a *local* rate by construction, so it falls away from 0.0287 and
eventually blanks as the Rutherford stage (W ∼ t) bends the curve over. The fp64 mirror
(`devtools/checkj.js` §4c) drives the app's own `islandWidth` + `islandFitGamma` off the
pseudospectral solver and gets 0.0278 against the 1D eigenvalue's 0.0287 (3.2%).

**The k_y = 2π/L_y mode.** KH's counterpart of the island chart, and on the same readback:
the `k_y = 2π/L_y mode` card (2D only, so it is off the 3D page exactly as `island width`
is) plots the m = 1 Fourier amplitude of u_x — and of b_x — along x = L_x/2 on a log
axis, and fits γ over the trailing `MODE_FIT_DT` = 10 t-units of *sim* time into its
legend (`γ_fit`, blank unless that window is finite, positive, rose by `MODE_FIT_RISE`
and fits with R² ≥ `MODE_FIT_R2`; a sample count would span wildly different stretches of
t on different devices, since the cut readback is wall-clock throttled). It exists because the **energy trace cannot show
the linear stage at all**: the equilibrium shear carries ~10⁶ times the seed's energy, so
E(t) is flat while the mode grows through six decades. u_x = −∂_yφ and b_x = −∂_yψ are the
two rows of the cut stack with *exactly zero* equilibrium content (the equilibrium is
y-independent with u and B along ŷ), so what is plotted is pure perturbation. b_x is
identically 0 at b₀ = 0, which a log axis cannot carry, so that series is simply absent
until there is something positive to draw. One caveat on the amplitude, not the rate: the
cut line sits midway between the two layers, where the mode is evanescent — A_u there is
0.17 of its peak on-layer value at the preset's defaults (a vortex-sheet estimate,
e^{−k_y L_x/4} = e^{−π} per layer, would say 0.09: the finite-width layer's eigenfunction
decays slower than e^{−k_y|x|}) — the log-y line is offset downward, its slope is not affected
(`devtools/checkj.js` §5 fits 0.26738 off this very line, 0.42 % from the eigenvalue).

**Linear-theory references** (`devtools/eqlinear.py`: a 1D generalized eigenvalue solve
of the linearized system on Fourier differentiation matrices, at k_y = 2π/L_y, converged
in the mode count; `devtools/checkj.js` reproduces each with an independent fp64
pseudospectral 2D run of the app's own ICs):

| case | parameters | γ (eigenvalue) | γ (2D run) |
| --- | --- | --- | --- |
| tearing | η = ν = 10⁻³ (Pm = 1), ψ₀ = 1.65, a = 0.1L_x | 0.028716 | 0.028609 |
| tearing | η = ν = 10^−2.5 | 0.051395 | 0.051300 |
| tearing | η = 10⁻², ν = 10⁻³ (Pm = 0.1) | 0.114636 | 0.11401 |
| tearing | η = 10⁻³, ν = 0 (Pm = 0) | 0.043646 | 0.043373 |
| KH | b₀ = 0, U₀ = 1, a = 0.05L_x, ν = 10^−3.5 | 0.266260 | 0.26657 |
| KH | b₀ = 0.5 U₀ | 0.206229 | 0.20377 |
| KH | b₀ = 1.2 U₀ | 0.003646 (resistive residue) | decaying |

Those rates are for a **frozen equilibrium**, which is what an eigenvalue problem assumes
— and what `maintain equilibrium flux` reproduces in the demo (0.0283 free-running with
the source on, 1.3 % off the eigenvalue). **With the source off** ψ_eq also diffuses at
~η/a² and slowly lowers Δ′: over t = 30…80 at η = 10⁻³ the measured tearing rate is
≈ 0.018, some 37 % below the frozen-equilibrium value. KH has no such source and does not
need one (ν/a² ≪ γ there: 3.6 %).

## Dissipation: the slider's range, and auto-diss

The perpendicular operator is −ν·k⊥<sup>2·hyper</sup>, so the **diss** slider carries
log₁₀ ν. Two grid-only numbers set everything here (`common.js`, beside `ctrlDissRow`):

- **k_d** = `DISS_KD_FRAC`·(largest retained k⊥). In this app that cutoff is the 2/3
  dealias, which is exactly what the ⊥ spectrum is binned out to, so k_d = 0.6·nb·kunit.
- **ν_marg** = k₁<sup>1/3</sup>·k_d<sup>2/3−2·hyper</sup> — the marginal coefficient for
  an O(1) box-scale amplitude: walk a Kolmogorov u(k) = u₁(k/k₁)<sup>−1/3</sup> from
  u₁ = 1 at k₁ = kunit down to k_d and set the dissipation rate there equal to the
  nonlinear rate. It is the old "auto" button's formula with ε<sup>1/3</sup> → u₁k₁<sup>1/3</sup>,
  i.e. *without* its dependence on the forcing sliders.

**The slider is a demo instrument** (FEEDBACK item 7). Its range is recomputed from the
live hyper / resolution / box by `dissRangeSync` (called from `syncCommonLabels`, so every
path that can change them ends in it): the top is Re ~ 1 at the box scale,
ν_top = k₁<sup>1−2·hyper</sup>; the bottom is `DISS_DECADES_BELOW` = 3 decades under
ν_marg, where the cascade no longer terminates inside the retained band. One sweep
therefore goes viscous → turbulent with an ε-independent dissipation rate → numerically
unresolved. A range <input> sanitizes assigned values against min/max/step, so
`presetWrite` **opens** the range to the hard bounds before writing and the following
`syncLabels` narrows it again — widened outward around whatever is stored, so a re-range
never moves the physical value, and every value written from code goes through
`dissWriteLog` (quantized to the slider's own 0.05-decade step, clamped, live path only).

**Auto-diss is a tickbox, ON by default** (FEEDBACK item 6), not the old one-shot button.
It sets ν continuously from the *measured* amplitude near the dissipation scale, so it
needs no assumption about where the energy came from — which is what makes it work for
KH and tearing as well as for a forced run. The rule is Jono Squire's rmhd-gpu
(`rmhdgpu/auto_dissipation.py`), ported with its structure and defaults: E_d = kinetic +
magnetic energy in the logarithmic shell k_d·e<sup>∓0.5</sup>, u_d = √(2E_d),
ν_target = u_d·k_d<sup>1−2·hyper</sup> (i.e. ν k_d<sup>2n</sup> = u_d k_d: the cascade
terminates at k_d), relaxed in log space by 0.2 per update with the change capped at a
factor 2. Three app-forced deviations: the cadence is wall-clock (`AUTODISS_PERIOD`,
2 Hz — the loop's step count per frame is not ours to choose, and every other readback
hook here is wall-clock too); [ν_min, ν_max] *is* the slider's dynamic range above; and a
shell with no measurable energy **holds** ν rather than driving it to ν_min, because a
quiescent or freshly-seeded IC has nothing at k_d yet. E_d comes off the app's own ⊥
spectrum bins — the same kernels the spectra card uses. With a spectrum card open the
controller rides that card's cached readback (the cards refresh faster than the
controller updates, so a fresh one almost always exists); only with no card open does it
take a readback of its own — so there is no new kernel and no new buffer, and at most
one spectrum pass is ever added.

While ticked the slider is disabled and *follows* the controller, so the level is always
visible and unticking simply leaves the manual slider where the controller left it.
Changing **hyper** makes the controller jump straight to the new exponent's target (one
fresh measurement, no relaxation cap — the cap is for measurement noise, not for a
re-parameterized operator); with the box unticked a hyper change never touches your
slider. The
pure core (`autoDissShellE` / `autoDissTarget` / `autoDissRelax`) takes state in and
returns ν, which is what `devtools/checks.js` §6 drives — including a closed-loop cascade
model whose fixed point is ν_marg to within a decade. Before the first measurement a
preset that wants the app to choose gets `autoDissSeed()` = ν_marg; the one preset that
turns the controller **off** is 2D `tearing`, whose whole point is a quoted η against a
linear reference growth rate (the same argument that makes it lock hyper).

## Forcing controls

ε⁺ and ε⁻ are separate log sliders with a lock checkbox (on by default); they are in the
same units — each is a contribution to dE/dt, so the total injection rate is their sum
(`rmhd._forcing_scale_from`'s convention). Unlocking them drives an imbalanced cascade.
The scale that realizes those rates solves `½·F₂·dt·s² + P·s = ε` for its positive root
(`shared_physics.selfnorm_scale`, mirrored in `physics.js` `scaleWGSL`), so a quiescent
start injects `ε·dt` on the first forced step (the second step overall — the O-U
envelope is still zero on step one) instead of an ε-independent kick; `smax` is only a
last-resort clip. Derivation and the dated behaviour change: `docs/numerics.md`.
The **band n** pair of handles sets the forcing shell [n_min, n_max) in units of the box
wavenumber; they cannot cross. The band is baked into the grid (`fmask`) *and* into the
OU kernel's `NS`, so changing it triggers the ordinary rebuild path on handle release —
which is why it is an `onchange`, not an `oninput`. Nothing about the generated WGSL
changes except the `NS` constant, exactly as a resolution change does.

## Alfvén-wave collision: directions, placement, χ

The propagation direction is read off the *implemented* propagator, not assumed. The 3D
stage kernel applies exp(L·τ) with L = [[d, i·kzd], [i·kzd, d]] to (φ, ψ); its
eigenvectors are ζ± = φ±ψ with eigenvalues d ± i·kz, so dζ±/dt = ±i·kz·ζ±, and with the
inverse transform's e^{+i kz z} convention that is **ζ±(z,t) = ζ±(z±t, 0)**: ζ⁺ travels
toward *smaller* z, ζ⁻ toward *larger* z, both at v_A = 1. (Same statement as RMHD's
∂_t z± ∓ v_A ∂_z z± = …, with B₀ = ẑ.)

**Placement** (`common.js: packetGeom`) is symmetric about the midplane with separation
s = clamp(5·σ_z, 3Lz/8, Lz/2), i.e. z₀± = (Lz ± s)/2. That is ≥ 5 σ_z on the direct side
by construction and on the wrap-around side too, because s ≤ Lz/2 ≤ Lz − s — which is
what the **σ_z slider's Lz/12 cap** buys (5·Lz/12 ≤ Lz/2; the slider shows "increase Lz
for longer packets" when it is sitting on the cap). Overlap is therefore impossible: at
the cap the two envelopes cross at 4.4% = exp(−25/8). At the default σ_z = Lz/16 the
3Lz/8 floor binds and the placement is the historical z₀⁺ = 11Lz/16, z₀⁻ = 5Lz/16; the
head-on collision is at z = Lz/2 at **t = s/2** (= 3Lz/16 ≈ 4.7 for Lz = 8π), strictly
before the wrap-around one at z = 0 (the closure speed is 2·v_A, so collisions repeat
every Lz/2 in time). The live line under the IC controls prints s and t.

**χ.** σ_z is the knob that sets the nonlinearity: the same line shows
**χ± = a∓·k̄⊥·σ_z/v_A** with k̄⊥ = 1/σ⊥ (σ_letter = Lx/32 for the letters, the blob-width
slider for a drawing — they share a default) and v_A = 1. Two values because the
nonlinearity one packet feels is set by the *other* one's amplitude (χ⁺ ∝ z⁻), which now
matters — the ζ± amplitudes are separate sliders. k̄⊥ comes from the shared smoothing
length because the smoothing width *is* the gradient scale of a smoothed glyph, so it is
the perpendicular wavenumber of the field z± = ẑ×∇ζ±; it is an *estimate*, the packets
are not single modes (an energy-weighted k̄ of the actual glyph runs lower). χ ≪ 1 is the
weak regime: the packets pass through each other nearly unchanged and distortion
accumulates over many transits (the box is periodic, so they collide again and again);
χ ≳ 1 is strong, a single collision already shreds them. The demo starts at a ≈ 0.2,
σ_z = Lz/16, χ ≈ 1.6, and reaches χ ≈ 2.1 at the σ_z cap.

**The sinusoidal pair** (3D IC preset `sinusoidal z± packets`, FEEDBACK item 9) is the
textbook version of the same collision: one packet whose perpendicular structure is a
single mode in x, the other a single mode in y. What is *stored* is the potential, so a
field that is a pure sine needs a potential that is a cosine of the other coordinate's
sign convention — ζ⁺ = −cos(k₁ˣx)/k₁ˣ and ζ⁻ = +cos(k₁ʸy)/k₁ʸ give
**z⁺ = a⁺·ŷ·sin(k₁ˣx)** and **z⁻ = a⁻·x̂·sin(k₁ʸy)** after `icZetaFields` normalizes each
to max|∇ζ| = its own amp slider (the 1/k prefactors are what make that exact). Each field
alone is an exact ideal solution — a z⁺ with no z⁻ propagates unchanged — and together
they interact through exactly one beat, (z⁻·∇)z⁺ = a⁺a⁻k₁ˣ sin(k₁ʸy)cos(k₁ˣx)ŷ. It is a
PACKET preset in every other respect: the same `icGaussZ` + `packetGeom` envelopes, so
σ_z, the ≥ 5σ_z placement, the trackers and the meeting time above are unchanged, and its
k̄⊥ in the χ line is the mode's own 1/k₁ = Lx/2π (the one case where it is not an
estimate). It is the collision preset's *default* IC since the on-device pass (a
space-filling pair is far easier to read as isosurfaces than two localized blobs); the
letters remain an entry in the same dropdown, and neither IC's code moved.

**Plane tracking.** Because z is spectral, "the energy in plane iz" is not a k-space
partial sum: `readPlaneEnergy` forms z± = φ±ψ, inverse-transforms *along z only*
(borrowing the gradient stack as scratch — a separate submit, and submits execute in
order), and reduces one workgroup per plane with the usual perpendicular-energy weight
minus the nz² Parseval factor, so the plane values average to E±. The frame loop reads
back 2·nz floats at ~10 Hz; every display card's z-slice source can be `manual`,
`track z⁺` or `track z⁻` (in the cube view they pick the TOP face's plane instead), and
the cut chart has its own. The tracked planes and their (continuous) z coordinates are
printed in the readout. How a tracker follows its packet is one page-wide choice:

- **energy centroid** (default) — the *circular* first moment
  z̄ = arg(Σ_k E_k e^{2πik/nz})·nz/2π. Periodic by construction, smooth, and exactly
  linear in t for a packet translating at v_A (node check: max residual 1.7e-13 planes
  over 400 readbacks, implied speed 1.000000, displayed plane never steps by more than 1).
- **peak plane** — the argmax, with **10% hysteresis**: it only leaves the current plane
  when another beats it by more than that. A raw per-readback argmax is what made the
  collision displays jitter; a 5% rival now never steals the plane, a 15% one does.

Both still cost one readback per ~100 ms and that readback is a `mapAsync` round trip
inside the frame loop, so a slow device pays a pipeline stall per tracked frame
regardless of the tracker — worth checking on-device if the collision preset still feels
uneven with the centroid.

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
- Resolutions 128/256/512 — n_x, since 2D also offers rectangular boxes: the wide 4π × 2π,
  where n_y is n_x/4, and the large 8π × 8π, which `chain` runs at 256². A
  1024-point line is exactly the WebGPU minimum workgroup-storage limit, so the 2D page
  asks for the adapter's own limit at boot and caps the longest line there (`NMAX_LINE`);
  a resolution a box cannot run is disabled in the select rather than silently clamped —
  which as of the 8π × 8π box disables nothing, every box having fy ≤ 1, but the rule is
  about the box table and not its current contents. See
  SPEC §8. Only 2D boxes are rectangular; the 3D perpendicular plane stays square.
- Elsasser forcing only, no snapshots.
- The perpendicular spectrum dispatches `nbins` = the smaller of the two axis dealias
  cutoffs in units of kunit = min(2π/L_x, 2π/L_y) (`floor(min(nx,ny)/3)` on a square box), but
  `round(|k|/kunit)` reaches that value in the corners of the dealias ellipse, so those
  few modes are binned nowhere: `sum(bins)` is a hair below the total energy (1.5% at
  16², far less at production resolutions). Long-standing, cosmetic.
