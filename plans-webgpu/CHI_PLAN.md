# Critical-balance plan: χ = k⊥δb/k∥ on the anisotropy card (3D app)

**Status: EXECUTED 2026-08-14, then SWITCHED OFF the same day** — built, adversarially
reviewed and fixed, then Alfred ran it on-device: χ ~ 2 at the defaults and strongly
scale-dependent, so the ordinate is **not offered** (`ANISO_CHI_UI = false` in `common.js`;
the code, checks and docs all kept — see the final section). Drafted and revised the same
day after Alfred's review (δb is now the matched level itself; the pairing wiring is
pinned). Built as written — no redesign, one deviation, recorded in the EXECUTION NOTES
together with the measured α. **Not a new card: a y-axis mode on the existing `aniso`
card.** Sibling plan `EIGF_PLAN.md` was drafted in the same session
and touches the 2D app; the two are near-disjoint and EIGF's **Sequencing** section owns the
rules where they meet.

Provenance: Alfred, 2026-08-14 — "new plot option for 3d: chi = kperp db / kpar, with kpar
the field-line-following one calculated in the same method as for the kpar/kperp vs kpar
plot." The estimator-bias section below records an exchange in which he was right twice and
the plan is narrower for it.

## What it is / why

The anisotropy card plots k∥/k⊥ against k⊥ and has to apologise for its own ordinate. From
`common.js`, immediately above `anisoCurves`:

> Gauge caveat, for the hint and the manual: under the RMHD rescaling symmetry the absolute
> value of k_par/k_perp is a convention (it moves with Lz). Only the slope is physical.

**χ is exactly the invariant that caveat is pointing at.** The RMHD rescaling is z → z/ε,
Z → εZ, t → t/ε; under it k∥ → εk∥ and δb → εδb, so k∥/k⊥ → ε(k∥/k⊥) while

    χ(k⊥) = k⊥ δb(k⊥) / (k∥(k⊥) v_A)

is unchanged. The same data, plotted on the axis that does not move with the box. Critical
balance stops being "a slope of −1/3 in a quantity whose level means nothing" and becomes
"this is O(1) across the inertial range", which is what CB actually asserts.

Cost: **nothing new**. v_A = 1 in these units (`rmhd3d.html`: "the line equation is
dx_perp/dz = b_perp/B0 with B0 = v_A = 1"), the ratio r(k⊥) = k∥/k⊥ is already what
`anisoCurves` returns, and δb² is the matched energy level already in hand in the same loop
(see *The method*). Pure CPU arithmetic on an existing readback — the ANISO_PLAN
principle, unchanged.

## Placement: an ordinate select, not a second card

`aniso` already has `src: "spectrum"`, the window logic (`fitKA` → `specKnee`), the lane
select (`aq`: tot/zp/zm), the measure select (`ad`: z/fl/both), and the fit line. χ differs
from the shipped curve by **one multiplication and the y label**. A second card would
duplicate all of that and cost a card slot on a phone.

So: add an ordinate option to `aniso`'s `opts`, e.g. `{ id: "ay", o: [["ratio", "k∥/k⊥"],
["chi", "χ = k⊥δb/k∥"]] }`, default `ratio` (the shipped behaviour, per the house rule that
the first option reproduces what shipped). `anisoCurves` gains the χ branch; `drawAniso`
gains the axis label, the reference line and the hint text. The card's title follows the
selection.

Consequence worth stating: the fit line's default index (−1/3, the CB ratio slope) is
meaningless on the χ ordinate, where the CB expectation is **slope 0 at level ~1**. The
reference line must switch with the ordinate — a horizontal χ = 1 — or be hidden.

## The method

Inside the existing level loop in `anisoCurves`, where `kp`, `kz` **and the matched level
`Q` itself** are already in hand:

    r   = kz / kp                       // shipped ordinate
    chi = kp * sqrt(Q) / (kz * vA)      // δb² = Q, vA = 1

with **δb²(k⊥) = Q** — the tail energy content above k⊥, i.e. the very currency the level
matching is built on ("what matches physically is energy CONTENT", the comment above
`anisoCurves`). An earlier draft interpolated sqrt(k⊥ E(k⊥)) at kp instead; the two differ
by the (α−1) O(1) factor of the local slope, which the estimator ruling below already
covers, and Q needs no interpolation at all — it is defined at every matched level by
construction (Alfred, 2026-08-14: this is what he expected it to do). State the convention
in the hint verbatim; it is an O(1) choice and χ is an O(1) quantity.

**Which energy, and the Elsasser pairing.** The `aq` lane select already offers tot/zp/zm.
Get the pairing right: the counterpropagating field is what shears you, so

    χ± = k⊥ Z∓(k⊥) / (k∥± v_A)

— χ⁺ takes δb from the **Z⁻** lane and k∥ from the **Z⁺** measurement. The wiring, precisely,
because there is a wrong implementation adjacent to the right one: the (kp, kz) level
matching stays **within** the selected lane — that pair *is* the measurement of k∥±(k⊥)
from Z±'s own geometry — and only δb crosses lanes, as a lookup of the *opposite* lane's
perpendicular tail at kp (a k → Q interpolation, the inverse direction of `_anisoAt`; one
small helper). On the tot lane δb² is the matched Q itself and even that helper is idle.
Building the perpendicular tail TP from the opposite lane wholesale is the adjacent
mistake: it changes which k⊥ every level maps to and measures k∥⁺ against Z⁻'s geometry.
This is the single detail in the plan most likely to be wired backwards, it is invisible in
a balanced run, and it is exactly what makes the card say something on the collision and
imbalanced presets. The check below must pin it — and must pin *which lane kp came from*,
not just the sign of the asymmetry.

**Both measures, kept.** Draw χ for the coordinate and field-line k∥ as the card already
draws both ratios: χ_fl approximately flat, χ_z falling as k⊥ rises because field-line
wander decorrelates the z frame. That divergence is the Cho–Vishniac lesson stated in the
variable it actually bears on, and — see below — it is a **ratio**, so it survives the
calibration question intact. (It is also *only* a restatement: the same δb divides both
curves, so χ_fl/χ_z = r_z/r_fl exactly — the hint should not present the split as an
independent measurement, just as the same lesson on the axis where it bears.)

## The estimator-bias question, and what it does and does not gate

An earlier draft of this plan claimed χ inherited a normalisation problem. Two rounds of
Alfred's pushback narrowed it to something much smaller, and the record matters because the
narrowed version is still worth one check.

**Not an issue: the L_z gauge.** It cancels, as above. That is the plan's motivation, not an
objection to it.

**Not an issue: δb's factor-of-two convention.** Definitional. State it, move on.

**Not a defect: an O(1) prefactor in the answer.** Alfred's ruling, 2026-08-14: *"there is
always an O(1) convention; in most turbulence predictions this is the ~ that appears
everywhere — so this is instructive rather than a defect."* Correct, and it is the same rule
he set during TEARNL for δ_SP versus δ_lin: the *level* is an order-of-magnitude statement
and never gets two significant figures in user-facing text; the *flatness* is the sharp
claim. Write the hint that way and a measured χ of 0.4 is a result, not an error bar.

**What remains: a possible estimator bias, which unlike the theory's ~ need not be a
constant.** `flSpectrum` must window before transforming — a field line's two ends are
unrelated (`common.js`) — and while the W2 division restores total variance it does not
restore shape: the Hann kernel convolves the parallel spectrum over ~2 bins. Broadening the
parallel marginal pushes content to higher k∥, so Q_par decays more slowly, the tail
matching returns k∥ too high, and χ comes back biased **low**. With nz = 64 there are 32
usable bins, and the aniso window's low end (kA, just above the forcing shell) is where the
ridge sits only a few bins up — the fractional smear is worst exactly where one wants to
read the level.

**The cheap test does not work, and the reason is structural** (Alfred, same day). Sweeping
`selLz` (2π…16π) looks like it should expose an L_z-dependent bias. It cannot: the forcing
is *"a k_perp shell, scattered onto the kz = ±2π/Lz planes"* (`rmhd3d.html`), i.e. pinned at
|kz| bin 1 whatever L_z is. The parallel outer scale moves with the box, the ridge stays at
the same **bin indices**, and the ~2-bin kernel is the same fractional smear at every L_z.
The sweep is null by construction. This also means the situation is not a tunable corner
case — every run in the app has its parallel outer scale at bin 1.

**So the only test is synthetic**, and it is a check, not a blocker:

> Build a field with a prescribed ridge k∥(k⊥) and prescribed δb, push it through the real
> path — field-line march, Hann, periodogram, tail matching — and read α = χ_measured/χ_true.

If α is within tens of percent, ship it and let the hint say χ is O(1) with a stated
convention. If α is a factor of ~3, that is worth knowing before the axis is labelled χ, and
the response is a sentence in the hint rather than abandoning the card. **Either way the
card ships**; α only decides how the level is described. The ratios the card is most
interesting for — χ_fl against χ_z, χ⁺ against χ⁻ — divide α out and are unaffected.

## What the card will actually show

Be realistic in the hint. The window runs from kA to the dissipation knee, which in a forced
256²×64 run is on the order of half a decade, and a flat line over half a decade is a weak
plot on its own. The content is in the departures, and they should be what the hint points
at: χ rising above 1 toward the forcing shell (the outer scale is not critically balanced,
it is driven), the roll-off at the knee, the two lanes splitting when ε⁺ ≠ ε⁻, and χ_z
peeling away from χ_fl.

## Defaults

Default ordinate stays `ratio` everywhere — the shipped card is Alfred's own copy after
three feedback rounds and this plan does not relitigate it. No preset layout change is
required, and the honest position is that χ earns a default only if on-device shows the
curve is legible over the available window. Revisit on the AW-collision and imbalanced
presets specifically, where the ± split gives it something to say.

## Constraints

- **No new WGSL, no new buffers, no new readbacks, no kernel change, no SPEC3D change.** If
  the implementation finds itself writing a shader, it has left the plan.
- Display-only; state never touched; RNG reference byte-identical.
- The `ratio` path must be **numerically unchanged** — same curves, same fit, same legend —
  when the new option is at its default. Assert it, do not assume it.
- No copy-paste: the tails, the window, the level loop, the log axes and the legend all
  exist.

## Checks (extend `devtools/checkaniso.js` rather than adding a file — same data, same
entry points)

1. **Default-path invariance**: with `ay: "ratio"`, `anisoCurves` output is bit-identical to
   base across every existing (`aq`, `ad`) combination. This is the regression gate.
2. **Analytic χ**: on a synthetic spectrum with known E(k⊥) and a known ridge, the χ branch
   reproduces k⊥δb/k∥ from the same inputs to fp64 tolerance — i.e. the *arithmetic* is
   right, independent of question 4.
3. **Elsasser pairing**: an asymmetric synthetic field (Z⁺ ≫ Z⁻) returns χ⁺ built from Z⁻
   and χ⁻ from Z⁺. Assert the asymmetry has the right *sign*; a symmetric test cannot catch
   a swap. Give the two lanes **different perpendicular slopes** as well as different
   amplitudes, and assert kp's provenance: the swapped-TP implementation (the "adjacent
   mistake" above) can survive a pure amplitude asymmetry with the right sign, and only a
   slope difference moves the levels' k⊥ mapping enough to convict it.
4. **α, the estimator bias**: the synthetic end-to-end measurement described above, reported
   as a number in the CI output. Reported, never gated — its job is to inform the hint.
5. Degenerate inputs (silent spectrum, one-bin tail, no overlap window) return empty curves
   rather than NaN, as the shipped card already does.

CI reports, never gates.

## On-device checklist (owed after merge)

Whether the χ window is wide enough to be worth an ordinate; the level that actually comes
back on `forced` (and how it compares to α from check 4); legibility of four curves
(two measures × two lanes) at phone width, and whether the lane split needs the measure
select pinned to one leg; whether the χ = 1 reference line helps or preaches; the AW
collision, where χ ought to be transient and large; whether the hint's O(1) framing reads
as honest or as hedging.

---

## EXECUTION NOTES (2026-08-14)

Built as drafted. `webgpu/common.js` (+~150 lines, most of them the comment block above
`anisoCurves`) and `webgpu/devtools/checkaniso.js` (+~430 lines, the plan's checks 1–5 as
§8.1–§8.5) are the whole of the code; `docs.html`, `README.md` and this file are the text.
No WGSL, no buffer, no readback, no kernel, no `SPEC3D.md`, no per-app edit at all — the
`ay` option is inert to `flChartOn`, so `rmhd3d.html` was not touched.

**α = 0.988.** Measured, reported, not gated (checkaniso §8.4): a synthetic field with a
prescribed ridge k∥(k⊥) = round(k⊥^2/3) and a prescribed δb, sampled along lines, pushed
through the real `flSpectrum` (Hann window, periodogram, ±kz fold, line average) and the
real `anisoCurves`, against the same field's exact parallel spectrum through the same
matching. The bias is on the LOW side, as the Hann-broadening argument in the plan says it
must be, and it is a **percent**, not a factor: 0.951 at the low-k⊥ end where the ridge is
only a few bins up, 1.000 at the top. So the plan's "within tens of percent → ship it and
let the hint say χ is O(1)" branch is the one taken, and no hedging sentence was added to
the hint. Two things make that number trustworthy and both are asserted beside it: the
synthetic periodogram returns 91.7% of the prescribed energy (the missing sixth of the first
ridge bin leaks into the kz = 0 bin, which `flSpectrum` does not plot — a property of the
instrument), and the measured parallel spectrum IS the (1/6, 2/3, 1/6) kernel on the exact
one, to 2%. The window, the transform, the fold, the average, the tails and the inversion
are the app's own code. Line phases are arranged so
the leakage cross terms between adjacent ridge bins cancel exactly over each group of four
lines — otherwise what comes back is one realisation of an interference pattern rather than
the kernel (the first attempt did exactly that and read α = 1.71, in the wrong direction).

**What α is a bias OF.** The first draft of this note called it end to end, and that was
wrong (adversarial review, 2026-08-14). It is the **parallel ESTIMATOR** — window,
periodogram, ±kz fold, line average, tail matching — with the field-line march **excluded**,
which is deviation 1 below. Two of the reviewer's measurements are worth keeping beside it.
With **random per-line phases** (what real field lines give, rather than this file's arranged
cancellation) the ensemble α = **0.993** over 24 realisations at the app's own 64-line
ensemble — so the arranged phases are a variance reduction, not the result. And α reduces to
**kz_true / kz_meas** at matched levels, which makes it a bias of the **shipped ratio
ordinate** exactly as much as of χ: χ does not introduce it, and nothing here is a reason to
prefer one ordinate over the other.

**Two deviations from the plan**, both recorded:

**1. The α measurement BYPASSES the march.** The plan asks for the synthetic field to be
pushed through "the real path — field-line march, Hann, periodogram, tail matching". There
is no GPU in node, so the samples are written down along a straight line instead. An earlier
version of this note treated that as a formality ("which is all a march produces"); it is
not. The app's marcher (`rmhd3d.html`, `fieldLine`) is bilinear in-plane and RK2 in z at
fp32, so it low-passes ALONG the line it traces — pushing k∥ **down** and χ **up**, the
opposite sign to the Hann broadening α measures. The two therefore do not add, and α is not
a bound on their sum. Measuring the march wants a GPU and is on the on-device list, not this
one.

**2. The card's title.** The plan asks for the card title to follow the
selection. `CHART_TYPES[k].label` is read once, into the type dropdown, for every card of
that type — following a per-card option would mean editing `ChartCard`, outside this plan's
edit surface. What was done instead is the shipped idiom: the label is now just
`anisotropy`, and `ay` LEADS the option row, so the header reads
`anisotropy | k∥/k⊥ | E_u+E_b | both k∥ | fit: pin` and, switched,
`anisotropy | χ = k⊥δb/k∥ | …`. The ordinate is named in the header, in the legend
(`χ (k∥z)` / `χ (k∥B)`), in the waiting line and in the hint, which is a function of the
options — the `gen2d` colour-scale shape, now with a second consumer.

**Two small things the plan did not settle**, both decided the conservative way:

- An Elsasser lane whose OPPOSITE lane has no bracketable perpendicular tail (E⁻ ≡ 0, i.e.
  maximal imbalance) draws **nothing** rather than falling back to the own-lane level. There
  is no shearing field, so there is no χ; a silent fallback would have been the mispairing
  the whole of check 3 exists to prevent. checkaniso §8.5 pins it.
- The χ = 1 reference line is included in the y range. Without it a card whose χ sits at
  0.07 draws its reference off the top of the frame and the legend claims a line nobody can
  see — and "is the level 1?" is the question the ordinate exists to answer. `fit: off`
  still hides the line, and `fit: amp` renames the level (the index box hides on χ, where a
  power-law index is not what the reference is).

**The gate.** checkaniso §8.1 is the plan's check 1 and it is a real regression gate, not a
promise: it reads `70ec5a8:webgpu/common.js` through `git show`, runs both implementations
over 14 data cases × 22 option sets × {`ay` absent, `ay: "ratio"`} and compares the
serialised output — 616 comparisons, all identical. (An out-of-repo copy with no git reports
a SKIP; this file reports, it never gates.) The same comparison was run standalone before
any edit, over a wider option matrix including the fit options, with the same answer.
`refvectors.json` / `refvectors3d.json` are byte-unchanged, which is trivially true — the
diff does not go near the solver.

**Checks run cold**, all green: `checkaniso` (the new §8 included), `checks`, `checkpin`,
`check2dspec`, `checkk`, `checkiso`, `checkonepage` (89/89), `checkidle`, `checksh`,
`checkgc` green (the leg count is not stable between trees and was quoted here as if it
were), `layout` (the sixth header control does not widen the busiest row —
the widest single item on 3D at 360 px is unchanged at 313 px), `names.mjs`. `checkj` was
not run to completion (it is minutes long and is the 2D tearing/eigenvalue path, untouched
here).

### Adversarial review, 2026-08-14 — verdict PASS-WITH-MINORS; nothing structural

Fresh reviewer against this plan and the merged diff. The pairing is wired the right way
round, the ratio path is untouched, and the two things the review did that are worth naming
by name:

- **Mutation testing of the check file.** Four mutants, including the plan's own "adjacent
  mistake" (building the perpendicular tail TP wholesale from the opposite lane, which is
  invisible in a balanced run) — **all four convicted**, by §8.3's kp-provenance legs rather
  than by the sign of an asymmetry. That is the check the plan asked for, doing the job the
  plan asked it to do.
- **The §8.1 gate is real**, not a promise: 616 draw-level comparisons against
  `70ec5a8:webgpu/common.js` read through `git show`, over 14 data cases × 22 option sets ×
  {`ay` absent, `ay: "ratio"`}, all identical — and it stayed identical through this round's
  fixes, which is what says the `noShear` flag below cannot reach the shipped ordinate.

Five minors, all fixed on `main` in `webgpu: review fixes — chi no-shear line, alpha's
scope, legend format`: the deliberate empty-opposite-lane return rendered as
"χ vs k⊥ — waiting…", promising data that never comes, while `docs.html` claimed the card
"honestly draws nothing" (that return is flagged `noShear` now and the card says
"no counterpropagating energy"; a new §8.5 leg asserts the flag is *unreachable* from the
ratio ordinate rather than merely unset there); the `fit` select's tooltip had no χ clause
though both its siblings did; α was labelled end to end (corrected above, with the
reviewer's two measurements kept); the reference level legended through
`Math.round(x*1000)/1000`, so a χ of 4e-4 read "χ = 0" — the one number the ordinate exists
to report, claiming to be zero (now `cbarFmt`'s 3-significant-figure rule with trailing
zeros dropped, gated over 4e-4 … 1250); and `docs.html`'s cross-reference to the
"anisotropy k∥/k⊥ chart" was stale, the card being titled just "anisotropy" since the
ordinate select landed.

Gates after the fixes: `checkaniso` all-green with §8.1 still 616/616 identical and
α = 0.9884 reported, plus `checkeigf`, `checkiso`, `checkgc`, `layout`, `names.mjs`,
`wgslparse` over fresh dumps and `bootstub` on both pages. `refvectors.json` /
`refvectors3d.json` byte-unchanged.

## On-device, and the switch-off (2026-08-14)

Alfred ran it: on `forced` at the default settings χ tops out around 2 — order unity,
which is what the plan promised the level could only be — but the curve is **strongly
scale-dependent**, and the flatness is the sharp claim. His hint copy had already said the
quiet part ("whether we can attain this in a low-resolution simulation in real time is
somewhat dubious"); measured, the doubt wins. His call: **the ordinate is not offered**,
and the code stays.

Mechanically that is one flag, `ANISO_CHI_UI = false` in `common.js`: the `ay` select is
absent from the header (absent, not greyed), the two tooltips' χ clauses gate with it, and
`docs.html`'s #chi section is hidden in place beside the same note. Nothing below the flag
is dead — the χ branch, the Elsasser pairing, the `noShear` return, the χ = 1 reference
and Alfred's hint copy are all still driven directly by `checkaniso` (154 legs green with
the flag off, §8.1 ratio identity untouched), so `true` brings the whole feature back as
reviewed. The α machinery and its 0.988 stay in the check output either way.

The plan's own postscript: the physics motivation (χ is the rescaling invariant) survives
intact; what failed is the resolution, exactly as the "What the card will actually show"
section feared — half a decade of window is not enough for a flat line to be flat in.
See Mallet et al. 2015 for the better-resolved test the hint pointed at.
