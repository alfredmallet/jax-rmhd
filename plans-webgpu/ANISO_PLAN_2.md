# Anisotropy plan 2: generated 2D spectrum E(k⊥, k∥) card (3D app)

One chart card, one phase, adversarial-review gated. **Separate file from
ANISO_PLAN.md deliberately: that plan is being executed by another agent
concurrently — do not edit ANISO_PLAN.md, and BASE THIS WORK ON THE TREE AFTER
ANISO LANDS** (it owns the fl-readback gate and the chart-card regions of
common.js this feature will sit next to; rebase, verify disjoint hunks).
Second constraint: **do not execute concurrently with ISO_PLAN** — both edit
physics.js (prepDisp there, prepGrads here), common.js card controls, and
per-card band UI. Agreed order: ANISO → ISO → this plan. Run after ISO Phase D
lands and REUSE its k⊥ band factor (half-cosine edges, exact-1.0 passthrough),
its per-card band-control pattern, and its checkiso §6 fp64 factor mirror. (If
ISO is abandoned or Phase D cut, this plan defines the factor itself — the
soft-dependency fallback, not the expected path.) Suggested by a collaborator.

## What it is / why

The ANISO chart reduces anisotropy to a curve k∥(k⊥). This card shows the
distribution the curve is extracted from: a static heatmap of E(k⊥, k∥), with k∥
measured ALONG FIELD LINES (band-pass the field in k⊥, then parallel spectrum of
the band-passed field along traced lines — the standard filtered-snapshot
technique). The ridge bending over is critical balance seen whole; the ANISO
curve overlays it as the extracted skeleton.

**Generate-button model (Alfred's design, fixed — do not relitigate):** the card
has a "generate" button (the k:"btn" opt kind from PINCURVE). Press → pause the
sim (setRunning(false)) → sweep all bands over the FROZEN state → render a static
plot. This is what makes the feature honest and cheap: every band row comes from
the SAME snapshot (a live per-update band cycle would time-skew rows — wrong
mid-collision), and the ~8 iFFTs × NBANDS cost is paid once at a button press,
not per frame. The plot persists (legend "generated @ t = …", the pin
convention) until regenerated; Run resumes normally and does NOT clear it.

## The method

- **Bands:** ~8–12 log-spaced k⊥ octave-ish bands between the forcing shell and
  the dealias cut (reuse the ANISO level-window thinking: kA anchor, knee walk —
  mirror, don't reinvent). Band factor = the shared Phase D half-cosine.
- **Per band:** apply the band factor to the (φk, ψk) READ path of the gradient
  prep — a uniform-gated multiply in prepGrads with exact-1.0 passthrough
  (sigR/Phase-D gating pattern; NO scratch state copies, NO new volume buffers;
  the state itself is NEVER written) → existing gradient iFFTs → field-line
  march → flSmp readback → existing CPU Hann periodogram → one row E±(k∥ | band).
- **Generate-time seeds:** the sweep may use a denser one-shot seed grid
  (GEN_SIDE = 16 → 256 lines) than the live view's FL_SIDE = 8 — affordable
  because one-shot; keeps the live path untouched.
- **Coordinate companion, nearly free:** the same press also bins the frozen
  spectral state into coordinate E(k⊥, |kz|) (one GPU binning pass in the
  spirit of the existing 1D spectra kernels). A card select toggles the displayed
  panel: "field line" (default) / "coordinate". The pair is the Cho–Vishniac
  lesson as two pictures from one snapshot: the coordinate ridge flattens at
  high k⊥ (line wander decorrelates the z frame), the field-line one keeps
  bending. One hint line says exactly that.
- **Rendering:** CPU canvas heatmap (~NBANDS × ~20 usable k∥ bins), log-log
  axes via the shared tick helpers, the small-colorbar machinery from the
  2026-08-10 batch, log-color with a floor tied to the specFloor convention.
  Optional overlay, cuttable: k∥ ∝ k⊥^(2/3) (GS95) reference line, and the live
  ANISO curve if that card is present.
- **Expectations, stated in the hint not oversold:** ≤ ~20 usable k∥ bins and
  fp32 make the ridge fuzzy at webgpu resolutions; the quantitative version of
  this plot stays a Kaggle-class exercise (2026-08-07 decision). The demo point
  is the SHAPE and the coordinate-vs-field-line contrast, not exponents.

## Constraints

- E± lanes as everywhere (E_u+E_b±H_c convention); display-only end to end — the
  state, charts, and live fl spectrum are untouched by a generate press.
- 3D only; the card is placeholderless in 2D (no kz).
- WGSL footprint: the prepGrads gate line(s) only; byte-identical emission when
  the feature is idle. Run new WGSL through wgslparse.mjs. No RNG anywhere.
- UI while generating: button disabled, brief progress text (NBANDS sequential
  encoder submissions with readbacks — order ~1 s; keep the browser responsive,
  no long single submission).

## Checks (devtools/check2dspec.js, follows checkk/checkaniso patterns)

1. wgslparse/names/dup; physics stepping WGSL byte-identical; idle-path emission
   byte-identical to base.
2. State invariance: generate press leaves (φk, ψk) bitwise unchanged.
3. fp64 mirror: band factor × gradient × periodogram row vs a direct fp64
   computation on a synthetic field, per band, ~1e-5.
4. Ridge recovery: synthetic anisotropic spectrum with known k∥(k⊥) → heatmap
   argmax ridge within a bin over the resolved range; coordinate panel binning
   Parseval-checks against the 1D spectra.
5. Pause/resume choreography + plot persistence via bootstub leg.
CI reports, never gates.

## On-device checklist (owed after merge)

Generate latency at 256²×64 (NBANDS × 8 iFFTs + readbacks); heatmap legibility
at phone widths (NBANDS rows × 208-px panels); floor/colormap feel; whether the
GS95 overlay helps or clutters; ridge visibility in forced turbulence vs
collision (collision is transient — hint may want "let it develop first").
