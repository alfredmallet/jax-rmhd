# Display-offset plan: x / y sliders that slide the picture inside its frame (2D app)

**Status: EXECUTED 2026-08-18** (asked for and scoped by Alfred in one round: 2D displays
only, the IC blob editor out of scope, the cut chart to stay on the real x = L_x/2, and no
preset key — "since this is all periodic the person writing a preset can just shift the IC
themselves"). Built on `main` at `89f263f`. Gate: `devtools/checkoff.js`. Two sliders per
2D display card, one multiply in one display kernel; no new kernel, no new buffer, no new
uniform byte.

## What it is

A structure that happens to straddle a box edge — a current sheet, an island, an arriving
packet — is hard to look at. The ⊥ box is periodic, so the fix is a change of origin, and
in k-space a change of origin is one phase factor. Each 2D display card gets `offset x` /
`offset y` in **box fractions** over ±0.5 (every distinct offset there is: ±L/2 are the same
picture), 0 in the middle, and `prepDisp` multiplies that card's display field by
e<sup>−i k·S</sup> on its way to the display inverse transform. The card then draws
f(x − S): an **exact roll**, nothing cropped, interpolated or wrapped in from a neighbouring
copy. +x is rightward, +y downward (grid row y = 0 is drawn at the top).

## What shipped

- `physics.js`: `MODE_SHIFT_STRUCT` (the band struct with `bpad` renamed `sh` — a
  `vec2<f32>` aligns to 8, so the two words the band already left as padding are reused at
  their existing offsets and every Mode uniform is still `MODE_BYTES`); `modeWords` grew a
  fifth argument writing words 6–7; `prepDispWGSL` grew the gated `shiftMul` block.
- `common.js`: `OFF_MAX` / `OFF_STEP`, the two `_rngLab` sliders behind `cfg.offset`,
  `DisplayCard.offset()` / `offsetOn()` / `offsetCap()`, the `offset:` word in `apply()`'s
  opt bag and the caption fragment.
- `rmhd2d.html`: `shift: true` in the constants object, `offset: true` in `cardsInit`, and
  `setDisplayMode` converting the card's fractions into lengths with its own `Lx`/`Ly`.
- Docs: `docs.html` (card controls, DRAFT-marked for Alfred's wording pass),
  `webgpu/README.md` ("The display offset (2D)"), `devtools/README.md`.

Three things needed no code, and that is the point of putting the factor where it is: the
offset rides **every** mode uniform the chain preps a field through (the card's own, the σ
mate, both contour potentials), so the contours and the second σ half roll *with* the field;
the arrow overlay gathers the real-space buffer the shifted chain just wrote, at the same
anchors; and the colorbar range is unchanged by construction, a roll being a permutation.

## Decisions

- **2D only.** `cfg.offset` builds the sliders, `buildShaders`'s `shift` emits the phase, and
  the 3D app sets neither: a 3D card's picture may be the whole box (cube faces, field
  lines, the volume march), where "slide the picture" is not a statement about anything. The
  3D `prepDisp` is byte-identical to the pre-offset text (verified), the `sigR`/`band`
  gating pattern.
- **Off is bitwise off**, the band's rule: at `[0, 0]` the whole `if` is skipped, so an
  unshifted picture is unshifted bit for bit and an ordinary visit does no offset arithmetic.
- **The cut chart does not move** (Alfred's call, and it is the right one): it keeps cutting
  the real x = L_x/2, which is what makes `e^{i k_x L_x/2} = (−1)^{ix}` in `cutPrep` true and
  is where the resonant surface of the tearing and KH equilibria actually is. Nothing outside
  `prepDisp` learns the word — asserted over every emitted kernel of the page.
- **No preset key.** `addDisplayCard`'s state bag deliberately has none: a preset author who
  wants the structure centred shifts the IC, which is free on a periodic box and moves the
  charts with it instead of only the picture.
- **No panel gate.** Unlike the k⊥ filter (opt-in behind `cbFilter` because Alfred did not
  trust it), the sliders are simply present. They are self-explanatory, harmless at rest, and
  the 2D card head is the one with room — no z source, no volume knobs. The cost is width:
  each is a `label.rngl` (label and slider one unwrappable flex item, `flex 0 1 150px`), so on
  a 360 px phone the 2D display head wraps one line further than it did. `devtools/layout.js`
  reports that line count rather than gating it — worth a look on-device, and a `cbOffset`
  checkbox in the displays & charts panel is the obvious retreat if it reads as clutter.
- **The IC blob editor is untouched** (out of scope): it paints on its own view and its own
  plane, in box coordinates.

## Reality caveat, for anyone taking the phase elsewhere

At an offset that is not a whole number of cells, e<sup>−i k·S</sup> breaks the rfft2 reality
constraint at the self-conjugate points (k_x-Nyquist and k_y-Nyquist). The displayed field is
real anyway only because the 2/3 dealias holds those modes at *exactly* zero — the IC is
masked (`run.py::initialize`) and so is every nonlinear term. A path feeding an unmasked
field through this phase would have to handle the mirrors explicitly.

## Verification notes (read before re-running the gate)

`devtools/checkoff.js` has four legs: emission gating (pure insertion + the one-kernel scan),
the uniform layout, the arithmetic through the **executed** kernel (wgsl_reflect: the factor
is the fp64 phase, it preserves every modulus, a whole-cell offset is a roll of the
real-space picture in the direction the tooltip claims, a whole box is the identity, off is
bit-identical to the un-gated kernel), and both pages booted on `stubenv`.

**The sandbox this was implemented in had no `node`**, so the legs that need one — the
wgsl_reflect execution and both page boots — are *written but unrun*; run `checkoff.js`
(and `checkiso.js` beside it) on a machine that has node before trusting them. What WAS
verified here, and how: the emission, insertion and `modeWords` legs plus the check's own
roll/direction reference machinery were run against the real `common.js` + `physics.js`
through JavaScriptCore (`osascript -l JavaScript`, both files eval'd in one scope — the
cheap substitute for `vm` when node is missing); the 3D `prepDisp` was byte-diffed against
`git show HEAD:webgpu/physics.js` the same way and is identical; and the phase ↔ roll
equivalence, the fp32 error (4e-7), the composition of two half-cell offsets and the
Nyquist-row argument above were checked independently in numpy.
