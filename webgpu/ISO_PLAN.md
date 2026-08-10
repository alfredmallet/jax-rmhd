# Isosurface plan: box-unit aspect + semi-transparent volume view + scale filter

Four phases, executed by agents in order, each gated by adversarial review (fresh
reviewer, SEND-BACK on majors). Base: current working tree (f79b14a + the uncommitted
2026-08-10 feedback batch); rebase over ANISO if it lands first — no file conflicts
expected beyond `rmhd3d.html` hunk adjacency. **Physics WGSL is untouched: every
physics kernel must remain byte-identical, and the checks assert it. No RNG-adjacent
changes anywhere in this plan.**

## What it is / why

Two features, one goal: make the 3D app show an Alfvén-wave collision the way the
Howes/Verniero/Klein 2016 movie does — an elongated column threaded by B₀, with the
interaction rendered as semi-transparent signed isosurfaces of j∥.

1. **Box-unit aspect (Phase A).** `Lz` is a free grid parameter (2π…16π) but
   `cubeQuads()` draws a unit cube regardless. Draw the box with edge lengths
   ∝ (Lx, Ly, Lz) instead. Side effect, intended: the field-line view gets cleaner —
   the same perpendicular wander is spread over a longer screen-z, so lines read as
   gently perturbed threads instead of scribble.
2. **Volume view (Phases B–C).** A new per-card view that raymarches the display
   chain's real-space volume and renders fuzzy shells at ±ℓ of the selected field.
   Field-line tracing shows δb⊥/B₀, which is small and undramatic in a collision; the
   physics — mutual shearing, growth of perpendicular structure, sheet formation —
   lives in j∥ = −∇⊥²ψ, and signed translucent shells are the visualization that
   shows it. Capstone: a guided "AW collision" preset (sinusoid z± IC, Lz = 8π,
   volume view of j∥) — the Howes movie, live and steerable, in a browser.

## Fixed decisions (do not relitigate)

- **Raymarch, not marching cubes.** No geometry extraction, no triangle sorting;
  front-to-back accumulation gives correct transparency for free; periodic wrap is a
  modulo in the sampler. The projection is affine/orthographic (`cubeQuads`), so rays
  are parallel: ray setup is nine numbers from the CPU, same discipline as the cube
  faces.
- **No 3D textures.** r16float is not a storage-writable format and r32float sampling
  needs an optional device feature. The fragment shader reads the display volume
  storage buffer directly with manual trilinear (8 taps) — the same pattern as
  `samp2`'s manual bilinear in the field-line kernel. Zero new texture machinery.
- **Fuzzy shells, not hard surfaces.** Opacity a·exp(−(f∓ℓ)²/w²) per signed shell.
  At 128²×64 a hard isosurface shows the grid; a Gaussian shell hides it and reads
  better at low opacity. Shading: central-difference normal (differences divided by
  the BOX-UNIT cell sizes Lx/nx, Ly/ny, Lz/nz — with elongated boxes the metric
  matters), fixed light, Lambert + ambient. Shell colors from the existing diverging
  colormap sampled at ±ℓ, so volume and slice views agree chromatically.
- **The volume marched is the display chain's own `dispR`** — the real-space volume
  `sliceExtract` already slices. Scalar fields (phi, psi, vort, jpar) are there as-is.
  Vector-magnitude fields get one new kernel INSTANTIATION, not a new kernel: the
  existing `vecMagWGSL` template with nDisp = NR (exactly how the face variants set
  nDisp = NFACE), writing the magnitude volume in place of the slice path. Level
  normalization reuses the max-reduction templates the same way (Vol variants of
  maxPartial/maxFinal). Buffer reuse discipline: extend the dispK/dispK2 → dispR/dispR2
  reuse chain; allocate nothing volume-sized.
- **Cost envelope:** 128³-ish volume × ≲256 steps × 8 taps at canvas res, at the 2 Hz
  display cadence — ≲1 G buffer loads per display frame. If integrated GPUs complain
  on-device, the fallback is half-res + upscale, a knob, not a redesign.

## Phase A — box-unit aspect

`cubeQuads()`: scale the centered corner coordinates per-axis by (Lx, Ly, Lz)/Lx
before projecting; the existing `sc = 0.92/max(hx,hy)` autoscale then fits the
elongated box in the square canvas unchanged. Everything downstream — `cubeFrame`,
the field-line polylines, the arrow overlay, the lines-view plate — reads off this one
projection, so consistency is free; the audit task is to CONFIRM that (grep for any
other consumer of the 0.92 scale or hard-coded unit-cube assumption), not to touch
them. Physics constants (`flC`, kz, parKfac) already read Lz and are out of scope.

Default: exact ratio, no cap. At Lz = 16π the box is an 8:1 stick and the
perpendicular faces get small — that is the true shape; whether a display cap (~6:1
with a hint line saying the render is compressed) feels better is an ON-DEVICE
decision, wired as a single constant either way.

Deliverables: the `cubeQuads` change; aspect assertions in the new check (edge-length
ratios track Lz/Lx for all four selLz values); on-device screenshot pass over
lines/cube views at 2π and 8π.

## Phase B — volume view

- **UI:** extend the z-source select with a single "vol" entry (NO "volzp"/tracking
  variants — ratified: the vol view shows the whole box, plane tracking is moot)
  and a `volView()` helper beside `cubeView()`. **vol is the 3D app's DEFAULT view**
  (ratified) — the collision preset no longer needs to switch views, only set
  level/opacity; audit bootstub/check legs that assert the old default.
- **New rendered fields: Elsasser vorticities ω± = ∇⊥²ζ± = ω ± j∥** — two new
  DISP_FIELDS entries (both apps; the table is a shared block) with caption
  definitions (ω±=∇²ζ±, ζ±=φ±ψ), computed in prepDisp as −k⊥²(φk±ψk): new branch
  lines, no new kernel. Signed → diverging cmap and ± shells automatically. Physics
  point: a packet is purely z⁺ or z⁻, so an ω⁺ vol card shows one packet with the
  other invisible — two side-by-side ω± cards separate the colliding packets, which
  largely supersedes the Phase C two-hue stretch. σ-mate table untouched.
  Two sliders in the card, live-active only in vol view: level ℓ (fraction of the
  volume max, default ≈ 0.35) and opacity. Shell width w rides ℓ (w = 0.4·ℓ),
  not a third slider.
- **WGSL (render-side only, one new shader):** full-canvas quad; fragment
  reconstructs the box-entry point and parallel ray direction from a uniform (origin
  + edge vectors, the inverse of the `cubeQuads` mapping, CPU-computed); marches in
  grid coordinates with the box-unit metric carried in the step; accumulates
  front-to-back with early exit at α ≈ 0.98; two shells at ±ℓ·vmax for signed
  fields, one at +ℓ·vmax for magnitude fields (the field table already knows which
  is which). Faint box-edge wireframe so the silhouette reads. Run it through
  `wgslparse.mjs` (reserved-words scan) like every other kernel.
- **Compositing with field lines:** the polylines live on the 2D overlay canvas, so
  they composite on top, not depth-correctly. Accept that: in vol view the lines
  overlay is OFF by default, available as the same checkbox the cube view uses, at
  reduced alpha, with one hint line noting lines draw in front. Depth-correct line
  compositing means moving lines into WebGPU — out of scope, noted as a possible
  follow-up, not attempted here.
- **Autoscale:** vmax from the Vol max-reduction per display tick, same symmetric
  convention as the faces' shared vscale.

Deliverables: shader + pipeline + bind groups riding the existing display encoder;
UI wiring; hint text (draft copy — Alfred rewrites); check coverage below.

## Phase C — collision preset + polish

- **Modify the EXISTING `?demo=collision` preset — do not add a new one.** It is
  already the Howes setup: Lz = 8π (so Phase A elongates it with no preset change),
  localized packet ICs launched toward each other meeting at Lz/2, χ readout. The
  packet IC STAYS — discrete packets passing through each other is the movie; the
  P2#9 sinusoid IC remains the space-filling alternative, not this preset's default.
  Changes: default view → vol of jpar, preset level/opacity, hint text extended to
  walk the collision (watch the shells shear where the packets overlap;
  perpendicular structure appearing there is the nonlinear interaction the charts
  measure; mention switching back to cube/track-z± views, since vol makes plane
  tracking moot). Draft copy only — Alfred owns the final text, per the preset-text
  precedent.
- **Two-hue z± mode (stretch, cuttable without review penalty, likely CUT):** one
  extra vol z-source entry rendering both packets as two hue families in one box —
  the second volume rides dispR2/the second chain's buffers under the existing
  reuse discipline. With the ω± fields in Phase B, two side-by-side ω± vol cards
  already separate the packets cleanly; build this only if the single-box version
  is judged worth it at review, and cut without discussion if the buffer
  choreography is ugly.
- docs.html user-manual section for the vol view; save/record (PNG/WebM/MP4) should
  ride the same canvas path untouched — verify, don't modify.

## Phase D — per-scale filter (both apps)

Anisotropy of structures as a function of scale, by eye: a per-card k⊥ BAND-PASS on
the displayed field. Slide the band up during a developed run and structures get
thinner in ⊥ while staying extended along z — the visual counterpart of the ANISO
chart's k∥(k⊥) curve (cross-link the two in docs.html if ANISO has landed). In 2D it
shows small-scale sheet/alignment structure. This is the Cho–Vishniac /
Maron–Goldreich filtered-snapshot technique as an interactive control.

- **DISPLAY-ONLY, emphatically not physics** (Alfred's explicit requirement): the
  filter is one factor multiplied into `prepDisp` — k-space, before the display
  inverse transform, after the state update. The solver state, stepping kernels,
  spectra/energy charts, and the field-line march (which reads the RHS gradient
  stack, not the display chain) are all unfiltered. One hint line says so.
- **Filter:** band-pass in k⊥ only (kz untouched — ≤~20 usable kz modes, and ⊥ is
  the anisotropy knob). Two ends (k_lo, k_hi) in units of the box k1; k_hi at the
  dealias cut = plain high-pass, defaults = wide open = OFF. Smooth half-cosine
  edges about one bin wide — a sharp cutoff's Gibbs ringing would read as fake
  structure. The passthrough value must be EXACTLY 1.0 (select(), not the cosine
  evaluated at its endpoint), so filter-off emissions and pixels are bitwise
  unchanged.
- **Wiring:** factor computed in-kernel from a two-float extension of the existing
  Mode/display uniform — no mask buffer, no new pass, no new allocation. Autoscale
  rides the existing max reduction, so the (much smaller) filtered amplitude
  renormalizes for free. Per-card controls (two sliders, or the two-ended pattern
  the pin/fit cards use — builder's pick within house style), so two cards can show
  two bands side by side.
- `prepDisp` is a shared physics.js template: both apps get the feature from one
  edit. The vol view's volume is `dispR`, i.e. Phase B inherits filtering with zero
  extra work — filtered j∥ shells in the elongated box is the payoff frame.

Deliverables: prepDisp factor + uniform + UI + hint; check coverage below (§6);
docs.html paragraph (draft copy).

## Checks (devtools/checkiso.js, follows checkpin/checkonepage patterns)

1. All WGSL parses (wgslparse.mjs), names/dup discipline (names.mjs, dup.py).
2. **Physics WGSL byte-identical** to base — this plan is render-path only.
3. Raymarch reference: synthetic analytic volume (offset Gaussian blob) → CPU
   reference march at a handful of pixels, tolerance ~1e-5 (fp32 accumulation order).
4. Aspect: cubeQuads edge ratios ∝ Lz/Lx across all selLz options; cubeFrame
   consistency (frame vectors derived from the same array).
5. Signed vs magnitude shell selection per field-table entry.
6. Filter: fp64 mirror of the emitted factor (band Parseval — energy of the
   filtered display field equals the spectrum summed over the band with the edge
   taper applied, ~1e-6); filter-off ⇒ WGSL emission AND a captured display-uniform
   sweep bitwise identical to base (the sigR gating pattern); edge smoothness (no
   value outside [0,1], monotone across each edge).
CI reports, never gates, as everywhere else.

## On-device checklist (owed after merge)

Perf + step-count feel on the integrated-GPU laptop; level/opacity defaults during an
actual collision run; aspect cap decision at 16π; lines-over-volume alpha; preset
caption timing; poster.png unaffected (onepage boots 2D — confirm); recorder round
trip in vol view (iPhone MP4 eyes already owed from the recorder rebuild — fold in);
filter feel — band width vs noise at 128²-class resolution, slider granularity, and
whether filtered+vol wants its own level default (band-passed fields are
near-zero-mean, so the symmetric ±ℓ convention should just work — verify by eye).

## Ratified decisions (Alfred, 2026-08-10 — no longer open)

- vol is the 3D app's default view.
- No volzp; vol-only.
- ω± = ∇⊥²ζ± added to the rendered-field table (both apps).
