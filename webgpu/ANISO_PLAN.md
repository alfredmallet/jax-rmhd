# Anisotropy plan: k∥/k⊥ vs k⊥ chart card (3D app)

One chart card, three phases, executed by agents in order, each gated by review. Base:
current main (f79b14a). Guiding principle: **the whole feature is CPU-side arithmetic on
readbacks that already exist** — no new WGSL, no new buffers, no new readbacks, no
change to any kernel or to SPEC3D.md. The only per-app (rmhd3d.html) edits are the
fl-readback gate and text. Everything else lands in `common.js` + `docs.html` + one new
devtools check.

## What it is / why

A new 3D-only chart card plotting the scale-dependent anisotropy k∥(k⊥)/k⊥ against k⊥,
with TWO curves — k∥ measured along the coordinate z axis (the "global" measure, from the
existing coordinate parallel spectrum) and k∥ measured along the actual field lines (from
the existing REFINE_PLAN-K field-line spectrum) — plus the settable power-law fit line the
spectrum card already has, with reference slopes −1/3 (Goldreich–Sridhar critical
balance), −1/2 (dynamically aligned / Boldyrev), and −1.

This is the critical-balance plot. The physics lesson is twofold: (a) eddies get more
elongated along B as they get smaller, at a measurable power law; (b) the global and
field-line curves DIVERGE — the coordinate-z measure saturates at the outer-scale k_z
(field-line wander decorrelates the z frame), trending toward ratio ∝ k⊥^{−1}, while the
field-line-following measure shows the real scaling. That divergence is the
Cho–Vishniac (2000) lesson, live. No gating to any preset: colliding packets develop
anisotropy too, and watching the curve be noise until the first collision is itself
instructive.

## The method (fixed — do not relitigate)

k∥(k⊥) is extracted by **equal-amplitude matching of the two 1D spectra**, drawn on
cumulative tail energies, NOT on spectral densities:

- Q⊥(k⊥) = Σ_{k′⊥ ≥ k⊥} E(k′⊥) and Q∥(k∥) = Σ_{k′∥ ≥ k∥} E(k′∥), each from its own
  binned spectrum. For a matching level Q*, the pair (k⊥*, k∥*) is where the two tails
  cross Q*. Scan ~16 log-spaced levels → the curve.
- **Why tails and not densities** (put this arithmetic in a code comment): matching raw
  densities E⊥(k⊥) = E∥(k∥) with GS95 slopes −5/3 and −2 gives k∥ ∝ k⊥^{5/6} — a wrong
  law manufactured by the method. Matching energy CONTENT (k E(k), or its integral, the
  tail) gives Q⊥ ∝ k⊥^{−2/3} against Q∥ ∝ k∥^{−1}, hence k∥ ∝ k⊥^{2/3}: the CB relation.
  The tail is preferred over kE(k) because it is monotone by construction — a unique
  crossing even on a noisy, bumpy live spectrum — and log-log interpolation between bins
  gives sub-bin k∥ resolution, which is what softens the integer-k_z quantization at the
  low-k⊥ end.
- Both parallel spectra drop the parallel mean (the coordinate one has no kz = 0 bin, the
  fl one no b = 0 bin) and the self-test already pins perp and coordinate-par to one
  energy normalization (its "sum E(k_perp) vs E_tot" / "sum E(k_par) <= E_tot" rows); the
  fl spectrum keeps Parseval along lines by its W2 normalization. So the tails are
  matched AS THEY ARE — no renormalization of parFL. A residual constant offset between
  the fl and coordinate normalizations would shift its curve vertically on a log axis and
  cannot change a slope; say so in a comment, and leave it.
- **No extrapolation past the grid**: a level whose Q* exceeds Q∥ at the first parallel
  bin would put k∥ below the first resolved bin — that level is dropped, not
  extrapolated. Same at every array end. At 128²×64 the surviving low-k⊥ points are
  still quantization-flattened; that is the instrument, not a bug — one line in the hint.
- **Level window**: levels are restricted to the inertial range the same way the fit
  line and the y-floor already think about it — low-k end anchored just above the forcing
  shell (the fit line's kA = max(2, round(fshell[1])), falling back to 2 when unforced),
  high-k end excluding the dissipation knee, REUSING the specFloor knee walk (factor its
  peak-then-walk-right crossing out, or mirror it exactly — do not invent a second knee
  rule). Levels then live in the overlap of the admissible Q⊥ and Q∥ ranges.

## Design decisions (fixed here so the phases don't relitigate them)

- **New chart type `aniso` in CHART_TYPES**, `src: "spectrum"` — it rides the existing
  ~300 ms spectrum readback exactly as `island`/`mode` ride the cut readback. 3D only:
  `avail: cfg => cfg.zslice` (the mirror image of island/mode's `!cfg.zslice`; the 2D
  app never sees it).
- **Data**: the same `d` object spectrum cards get (perp, par, parFL, nb, parKfac,
  kunit, fshell). The axes are k/kunit on both legs (parKfac already puts the parallel
  bins on that axis), so the plotted ratio is dimensionless in box units. **Gauge
  caveat** for hint + docs: under the RMHD rescaling symmetry the absolute value of
  k∥/k⊥ is a convention (it moves with Lz); only the slope is physical.
- **The fl gate**: `flChartOn()` in rmhd3d.html currently fires the along-line sample
  readback only for spectrum cards with `sd === "fl"`. It must also fire for an open
  aniso card whose curve selector wants the field-line curve. Keep it one predicate;
  this is the only functional per-app edit.
- **Curve assembly is a pure function** `anisoCurves(d, o)` in common.js — the
  specCurves seam, node-testable, no canvas/DOM. Returns the same
  `[points, colour, dash, label]` curve shape plus hi/lo, so drawing and any future pin
  support inherit the established form.
- **Card options** (the CHART_TYPES `opts` idiom):
  - `aq`: which energy the tails are built from — total E_u+E_b (default), z⁺, z⁻
    (the three lanes are already in every spectrum; E± = E_u+E_b±H_c).
  - `ad`: both curves (default) / global only / field line only. Global solid,
    field-line dashed, one colour each from the existing palette; legend labels
    "k∥z /k⊥" and "k∥B /k⊥" (exact strings are draft copy — see Phase C).
  - fit controls: same `fit`/`fitp`/`fita` trio as the spectrum card, same behaviours
    (pin-to-field anchors on the FIELD-LINE curve when drawn, else the global one, at
    its first point ≥ kA).
- **FIT_FRACS becomes per-use**: the snap table [[−5/3],[−3/2]] is currently a module
  const. Parameterize (`fitIndex`/`fitLabel` take a fracs table, spectrum keeps its
  pair) and give aniso [[−1/3, "−1/3"], [−1/2, "−1/2"], [−1, "−1"]], default index
  −1/3 (box default −0.333, snapped). −1 is there because the GLOBAL curve is expected
  to trend to it — a teaching slope, same standing as the other two.
- **No pins on the aniso card in v1.** The pin machinery is spectrum-card-specific
  (pinAdd goes through specCurves). The curve-shape compatibility above keeps the door
  open; a follow-up item, not this plan.
- **Draw**: log-log, drawSpectrum's frame idiom (chartFrame, logTicks, clip rect,
  "waiting…" until points exist). Forcing-shell markers kept on the x axis — they
  orient the eye the same way they do on the spectrum. y range from the drawn points
  with the at-least-one-decade rule. The fl curve appears when flPar first lands (its
  own 2 Hz cadence); until then the card draws the global curve alone — no special
  casing.
- **Presets untouched.** The card is reachable from every chart card's type selector;
  whether any preset should open it by default is Alfred's call later.
- **Hint/caption/docs text is DRAFT copy** flagged for Alfred's pass — house rule from
  the preset-text history.

## Phases

### Phase A — the math, and its check (common.js + devtools/checkaniso.js)

`anisoCurves(d, o)` + the FIT_FRACS parameterization + the knee-walk factoring, plus
`devtools/checkaniso.js` (stubenv idiom, exit-code gate like checkpin/checkj) covering:

1. **Slope recovery**: synthetic bins with E⊥ ∝ k⊥^{−5/3}, E∥ ∝ k∥^{−2} (both parallel
   lanes) → least-squares slope of log(ratio) vs log(k⊥) = −1/3 within tolerance;
   E⊥ ∝ k⊥^{−3/2} case → −1/2.
2. **The density trap, as a regression guard**: an implementation that matched densities
   would return 5/6-law anisotropy (ratio slope −1/6) on case 1's inputs — assert the
   recovered slope is −1/3 specifically, tolerance tight enough to exclude −1/6.
3. **Uniqueness/robustness**: multiplicative noise and a non-monotone bump on the bins →
   every level still yields exactly one crossing per leg (monotone tails), finite
   points, no NaN.
4. **No extrapolation**: levels beyond either tail's range are dropped; a Q* above
   Q∥(first bin) yields no point rather than a fabricated one.
5. **Degenerates**: empty/zero/one-bin spectra, `parFL: null` (global curve only),
   par null (no curves, "waiting…" path returns empty), tiny nb.
6. **Window**: points respect the kA anchor and the knee exclusion on a spectrum with a
   built-in hyper-diss tail.

Gate: checkaniso green; node devtools/{checkj,checkpin,checks}.js and dup.py/names.mjs
still green (the FIT_FRACS refactor touches shared code).

### Phase B — the card (common.js CHART_TYPES + rmhd3d.html gate)

CHART_TYPES entry, drawAniso, opts wiring, the flChartOn widening. Card opens/closes
cleanly in 3D, absent from the 2D app's selector. Fit line snaps and labels −1/3, −1/2,
−1. Gate: all devtools checks green including checkonepage (74/74 — the onepage embeds
the 2D app, which must be byte-identical in behaviour; WGSL byte-identity is trivially
required since no WGSL is touched) and wgslparse (nothing new to scan, must stay green).

### Phase C — text (docs.html + hints), self-test row

One user-manual section in docs.html (structure follows the FEEDBACK_2026-08-10 split),
the card hint with the gauge caveat and the quantization line, both marked DRAFT for
Alfred. One structural self-test row in the rmhd3d harness: `anisoCurves` on the
self-test grid's real spectra returns finite points and no NaN (no slope assertion — a
16² self-test box has no inertial range).

## Hard constraints

- No WGSL, no buffers, no readbacks added or changed; refvectors3d.json untouched; no
  RNG anywhere near this. `git diff --stat` should show common.js, rmhd3d.html,
  docs.html, devtools/checkaniso.js, this file — nothing else. AMENDED post-Phase-B
  (overseer call): devtools/bootstub.js is admitted too — its lines-view section
  asserted "no k_par chart open" from the days when the spectrum card was the only
  along-line-sample consumer; the aniso card (default `ad: "both"`) is legitimately a
  second one, so the harness now parks aniso cards on `ad: "z"` for that section,
  keeping the original assertions' exact meaning. The gate itself is proven by
  checkaniso's truth table.
- The spectrum card's behaviour is bit-for-bit unchanged (its snap table keeps −5/3,
  −3/2 through the refactor — checkpin/checkj prove the assembly, eyeball the fit
  legend).
- House comment style: explain the why in paragraphs at the site, the way specFloor and
  flSpectrum do. New constants get the SPEC_* / FIT_* naming treatment.

## Review

Adversarial review of the full diff against this plan by a fresh reviewer after Phase C
(or after any phase that ends a session): verdict MAJOR/MINOR per finding, majors block.
Reviewer instructions: check the matching math against the "why tails" argument
independently, re-derive the 5/6 trap, verify the level-window reuse didn't fork the
knee rule, verify flChartOn still gates the smp readback OFF when no consumer wants it,
and run every devtools check cold.
