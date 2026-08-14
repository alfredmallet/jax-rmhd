# Critical-balance plan: χ = k⊥δb/k∥ on the anisotropy card (3D app)

**Status: DRAFTED 2026-08-14, REVISED same day after Alfred's review (δb is now the
matched level itself; the pairing wiring is pinned), NOT STARTED.** Base: `70ec5a8`, tree clean. **Not a new card:
a y-axis mode on the existing `aniso` card.** Sibling plan `EIGF_PLAN.md` was drafted in the
same session and touches the 2D app; the two are near-disjoint and EIGF's **Sequencing**
section owns the rules where they meet.

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
