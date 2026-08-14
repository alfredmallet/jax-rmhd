# Eigenfunction plan: ψ̂(x), φ̂(x) at fixed k_y — a tearing-mode card (2D app)

**Status: EXECUTED 2026-08-14** (drafted and revised the same day after Alfred's review —
Δ′ readout dropped, y axis now linear, stale hint clause to remove). Built on branch `eigf`
off `2cbfadd` (this plan's own commit; `70ec5a8` below is that commit's parent and the two
are byte-identical under `webgpu/`). Gate: `devtools/checkeigf.js`, all passing, with
`checkaniso.js` still all-passing beside it. One chart card, 2D only. Sibling plan
`CHI_PLAN.md` was drafted in the same session and touches the 3D app; the two are
near-disjoint and the **Sequencing** section below says exactly where they meet.

**What shipped, against the plan below.** Everything, with no design departures:
`eigfGatherWGSL` in `physics.js` (a pure strided gather, `fields` read-only, j₀ from a
16-byte uniform, emitted by the 2D page alone), `eigfProfile` + `drawEigf` in `common.js`
(the inverse along kx through the existing `fftPow2`, at the full 1/(nx·ny) inverse-rfft2
normalization, so the plotted quantity is the coefficient c_j(x) of ψ = Σ_j c_j(x)e^{ik_y y}),
the `eigf` entry inserted immediately after `mode` in `CHART_TYPES` with `aniso` untouched,
`Solver.readEigf` + the third throttled `src: "eigf"` readback in the frame loop (the cut
line's idiom, with the card's k_y BIN where the cut card has its z plane — one gather per
distinct bin on screen), the k_y (1…9, raised from 6 in review — see below) and field
(ψ/φ/both) selectors, `avail: cfg =>
!cfg.zslice`, the card as `tearing`'s third chart and available-not-default everywhere
else, and the side task (the stale 30–40 % clause deleted from the `tearing` hint, the
diffusion statement it hung off kept). Docs: `docs.html` #eigf, `webgpu/README.md` (chart
options table + a "Display modes" subsection), `SPEC.md` §7.

Execution notes worth keeping:

- **The plot is a genuine measurement of the transform, not of itself.** Check 3 builds the
  state by a direct fp64 forward DFT of ψ = g(x)cos(k_y y + p₀), φ = h(x)sin(k_y y + p₀)
  and compares the card's output with the ANALYTIC g/2 and h/2 — agreement 8.9e-9 and
  2.5e-9 (tol 1e-5), which pins the normalization and, since **review 2026-08-14**, the
  DIRECTION as well. As first shipped it did not pin the direction: g and h were both
  symmetric about L_x/2 and the plotted quantity is a modulus, so a reversed inverse
  reproduced them to the same 1e-9 and the note here claiming otherwise was wrong. The
  envelopes are off-centre now (peaks at 0.31 L_x and 0.68 L_x — g/2 and h/2 is a property
  of the transform, not of the shape, so the shape was free to move), plus an explicit
  peak-position leg; reversing `fftPow2`'s sign fails four legs at O(1) and failed none
  before. The two shape claims the card exists for keep the centred pair and fall out of
  the same mirror rather than out of prose: ψ̂ peaks exactly on ix = NX/2, and |φ̂(x₀)| is
  2.35e-9 — an ABSOLUTE number, i.e. 1.2e-8 of its 1.92e-1 lobes.
- **The gather is bit-exact**, so "minus the equilibrium" is a property of the STATE and
  not of any code: check 4 puts a y-independent ψ_eq in and finds every k_y > 0 column at
  9.3e-16 of the k_y = 0 one.
- **Three cards on `tearing`** was shipped rather than the plan's fallback (`eigf` replacing
  the cut trace). `devtools/layout.js` passes at 360 px with the new card's header measured
  — but note what that measurement is and is not: the widest single 2D item is 284 px of
  316 available and it is a **label + range + range + val control row**, not the eigf
  header, which is nowhere near the limit. It is unchanged by this card, and unchanged
  again by the k_y list going to 9 (review 2026-08-14). The plan's real question — whether
  three cards *read* well on a phone — is an eye question and stays on the on-device list.
- The same stale 30–40 % number survives in two DEVELOPER-facing places the side task did
  not name: `common.js`'s `ISLAND_FIT_RISE` comment (where it is load-bearing arithmetic for
  a gate constant) and one parenthetical in `webgpu/README.md`. Left alone deliberately —
  the ruling was about the user-facing hint, and re-deriving the gate constant is not this
  plan's business.
- **On-device round 1 (Alfred, 2026-08-14):** title cut to plain "eigenfunction" and the
  hint replaced wholesale with his copy (the check legs now pin his claims); the lone "x"
  axis glyph collided with the 0.00 tick and became the cut card's end-label idiom
  ("x = 0" … L_x); the island-chain hint moved to "up to k_y = 9" and the out-of-date
  512-vs-256 resolution comment on the chain preset was removed at his call. His "φ̂
  doesn't look odd around L_x/2" was run down via `eqlinear.py`'s eigenvector at the
  shipped `tearing` numbers: φ̂ IS exactly odd (parity residual 6e-8; ψ̂ even at 3e-10),
  but its lobes peak at 0.13a from x₀ — **≈ 3 cells at 256, back to half-max by 10** — so
  the odd structure is layer-localised and reads as a single narrow spike with a ~1-cell
  notch. Physics, not plumbing: FKR's outer φ is small and the flow concentrates in the
  layer. Anyone captioning φ̂'s shape should caption a spike, not lobes.

### Adversarial review, 2026-08-14 — verdict FAIL; one major, six minors, all fixed

Fresh reviewer against this plan and the merged diff. Fixed on `main` in
`webgpu: review fixes — eigf uniform size, k_y range, direction check`.

**The major, and it is a whole class.** `struct EigSel` was padded with a `vec3<u32>`,
which has align 16 — so the declared struct's WGSL SizeOf is **32** against the **16**-byte
`eigfU` the page allocates. The pipelines use `layout: "auto"` with a whole-buffer binding
and the bind groups are built in the `Solver` constructor, so that is a `GPUValidationError`
out of `createBindGroup` **at boot, on every real device, card open or not**. Nothing
above could see it: no CPU gate allocates, so the entire check file passed a shader that
could not bind. Fixed with three scalar pads (`rmhd3d.html`'s `struct FL` is the house
pattern; the buffer did not move), and `checkeigf` gained the leg that is the **regression
gate for the class**: reflect the EMITTED kernel and assert the uniform struct's SizeOf
against the buffer size read off the booted solver. Reads 16 B vs 16 B; putting the vec3
back reads 32 B vs 16 B.

`devtools/checkiso.js` FAILED on the merged tree — its exact-additions list did not know
about `eigfGather`. Listed there now, with the note the other borrowed kernels carry.

The other minors, each fixed where it lived: check 3 could not pin the transform's
DIRECTION and the note above claimed it did (both now true — see that note); "seeds every
mode up to 6" was wrong by 4× in four places and `EIGF_KMAX` went 6 → 9 so the growth-rate
roll-over past the n = 6 winner is on screen; the x = L_x/2 guide is drawn on every preset,
so the hint now carries both readings (resonant surface on tearing, plain midline on KH,
whose layers are at L_x/4 and 3L_x/4); `drawEigf` handed non-finite samples to `lineTo`
when one field went NaN and the other did not, which a canvas drops silently while the
degenerate leg claimed nothing non-finite was drawn (samples now break the polyline, and
the NaN-φ/finite-ψ case that convicts the old code is in the suite); and two loose
execution notes here — the 2.3e-9 and the 284 px — were corrected in place above. On
terminology: a **y-independent** field lives entirely in the k_y = 0 column; it does not
have "zero mean along y", which would mean that column is *empty*. Corrected in
`physics.js`, `common.js`, `rmhd2d.html`'s `srcInit` note, `webgpu/README.md` and above.

Owed, not done: the `island chain` PRESET HINT (`rmhd2d.html`) still says "seeded with
random modes up to k_y = 6" — the same 4× error, in Alfred's own user-facing copy, which
this round deliberately did not rewrite.

Gates after the fixes, all green: `checkeigf` (with the two new legs), `checkaniso`,
`checkiso` (now), `checkgc`, `layout` at 360 px, `names.mjs`, `wgslparse` over fresh dumps
of both pages (124 / 270 kernels, 0 failures), `bootstub` both pages. `refvectors.json` /
`refvectors3d.json` byte-unchanged.

Provenance: Alfred, 2026-08-14 — "solution minus equilibrium, cut along x at a specified
location: phi and psi: shows eigenmode structure at early times." The idea is his; two
things below depart from the literal form of it, and both departures are argued rather than
assumed.

## What it is / why

The tearing family (`tearing`, `tearing: X-point collapse`, `tearing: island chain`) shows
the *consequence* of the instability — reconnected flux, an island, a chart of log W(t) —
but never its *structure*. The eigenfunction is the thing a reader of FKR has in their head
and has no way to see here: ψ₁ peaked on the resonant surface with a kink in its slope, φ₁
odd about it with a pair of lobes, the whole outer solution spanning the box.

A new 2D-only chart card draws, against x across the full box:

- **|ψ̂(x, k_y)|** and **|φ̂(x, k_y)|** — the moduli of the Fourier coefficients at one
  chosen k_y, on a linear, autoscaled y axis (see *Axes*).

## Two departures from the request, both deliberate

**1. Fourier modulus, not a real-space cut at a specified y.** Linearising about the
equilibrium, γψ₁ = i k_y ψ_eq′(x) φ₁ + η∇²ψ₁. γ and ψ_eq′ are real for the growing mode, so
φ₁ is *90° out of phase with ψ₁ in y*: ψ₁ even in x and real, φ₁ odd and imaginary. At the y
where ψ is maximal, φ is identically zero, and vice versa. **No single y shows both fields.**
Plotting |ψ̂| and |φ̂| is phase-free, shows both at once, needs no y control, and is the
eigenfunction itself rather than one realisation of it. (If a real-space trace is wanted
later it is the same data times a phase — a follow-on option, not this card.)

**2. "Minus equilibrium" = drop the k_y = 0 column, taken LIVE.** The equilibrium *is* the
k_y = 0 column by construction: every equilibrium seed is y-INDEPENDENT, so it lives
entirely in that column — which is
exactly what `srcInit` relies on when it extracts ψ_eq,k (`rmhd2d.html`, the eqsrc block —
`eqk[m] = select(0, fields[NM + m], (m % NKY) == 0u)`). So the subtraction is free and
exact, with no stored reference field, and selecting a k_y ≠ 0 column performs it
automatically.

It must be the **live** column — which is also the cheap choice: selecting a k_y ≠ 0 column
performs the subtraction automatically, exactly, with no stored reference field, while a
t = 0 snapshot would need one and be wrong once anything has evolved. (An earlier draft
argued this from a diffusion rate "η/a² = 0.1 ≫ γ"; that arithmetic read `rEqA` as an
absolute length. It is a/L_x — the island-chain preset comment pins the convention:
a = 0.4712 = 0.01875 · 8π — so on `tearing` a = 0.1 · 4π ≈ 1.26 and η/a² ≈ 6×10⁻⁴, some
45× *slower* than γ = 0.0287. The live column needs no rate argument; it is exact by
construction.)

Side task for this plan's `rmhd2d.html` commit: the shipped `tearing` hint's clause that
switching the source off "drops the measured slope some 30–40% below the reference" is
probably a stale measurement (Alfred, 2026-08-14), and the corrected diffusion arithmetic
above cannot support it — delete the clause, do not re-measure.

Note what the live column also contains once the run is nonlinear: the flattened current
profile and any Reynolds-stress-driven mean flow. That is the right thing to call
"equilibrium" at time t, and the hint should say so in one clause rather than implying the
card is showing a linear eigenmode forever.

## Dropped: the Δ′ readout

An earlier draft fitted ψ̂′/ψ̂ on either side of x₀ and quoted Δ′a in the legend beside the
preset's analytic value (8.4 / 37.8). Dropped (Alfred, 2026-08-14): the fit is entirely
window-dependent, and at this solver's resolution the number is not interesting in enough
detail to earn a legend slot. **The card is the plot alone.** If it is ever revisited, the
review notes to start from: fit the measured ψ̂ against an fp64 integration of the outer
ODE rather than extrapolating log-derivatives; fix the fit window in grid cells inside the
check and ship those same cells; and gate the readout at both ends (the seed is not the
eigenmode early, the outer solution is not ideal late).

## Honesty about what is resolved

The shipped `tearing` preset is 256×64 with δ_lin/dx = 1.03 (`rmhd2d.html` preset comment;
the measured ladder is in TEARNL's execution notes). **The inner layer is about one cell:
this card shows the OUTER solution and must not be captioned as showing the layer.** That is
not a defect — the outer solution is where the structure a reader recognises lives — but a hint
that promises "the resistive layer" would be writing a cheque the grid cannot cash. At 512
(one click away) the layer is ~2 cells, still not a picture of a layer.

## Axes

Linear y, autoscaled per frame — the mode grows exponentially, so the shape, not the level,
is the content. An earlier draft specified log y, but its only real justification was
reading the exponential far field for the Δ′ fit, which is dropped. Linear is what shows
the eigenfunction as a reader of FKR pictures it — ψ̂ peaked with the slope kink at x₀,
φ̂'s odd pair of lobes — and it dissolves a log-axis defect for free: φ̂ is odd about x₀,
so |φ̂(x₀)| = 0 would be a −∞ spike on a log axis needing a floor clamp; on a linear axis
it is simply a zero, and its sitting exactly on x = L_x/2 is a free diagnostic.

## The method

**Readback.** The existing cut path is no help: `cutPrepWGSL` collapses x by direct summation
using the (−1)^ix trick and leaves one inverse rfft *along y* — the exact transpose of what
this card needs. What it needs instead:

- a small **gather kernel** in `physics.js`: given a k_y index j₀ from a uniform, write the
  strided column m = ix·NKY + j₀ for both fields into a compact buffer (NX × vec2 × 2 ≈ 4 KB
  at NX = 256). Ten lines, one dispatch, no new state.
- **CPU inverse along kx**, in `common.js`, reusing `fftPow2` (already there for
  `flSpectrum`). NX is a power of two in every preset. Then |·| per x.
- the card is therefore a **new readback source** (`src: "eigf"`), throttled like the cut and
  spectrum readbacks and keyed on `stateMark()` the same way. This is the one piece of new
  plumbing in the plan and the one place to look hardest during review.

**Controls (`opts`).** k_y selector — default the box fundamental (bin 1), which is the
seeded mode on the single-mode presets; the list runs to a few bins for `island chain`,
where Δ′ selects the mode and watching a *different* k_y win is the whole point of that
preset. Plus a field selector (ψ only / φ only / both) if the two moduli crowd one axis.

**Availability.** `avail: cfg => !cfg.zslice`, as `island` and `mode` already are — the
equilibria are 2D. It is not restricted to the tearing ICs: it is meaningful on `KH` too
(where `mode` already plots the k_y amplitude *on one line*, and this card generalises that
to the full x profile). Worth saying in the hint; not worth a preset gate.

## Defaults

- **`tearing`** — add it. This is the preset the card is for: single-mode, quoted γ, quoted
  Δ′a. Layout becomes three cards (`island`, `cut`/b, `eigf`); check the phone width before
  committing to three, and if it is too tight, `eigf` replaces the cut trace rather than
  joining it.
- **`tearing: X-point collapse`** — available, **not** default. At Δ′a = 37.8 and rEqPert
  −4 the interesting phase is the ideal collapse, not an eigenmode, and a card that says
  "eigenfunction" invites a reader to over-interpret the early frames.
- **`tearing: island chain`** — available, not default; it ships with two cut traces already
  and the k_y selector makes it a *manual* exploration, which is the right register for that
  preset. Reconsider after on-device.
- **`KH`** — leave alone. `mode` already covers it.

## Constraints

- Display-only, end to end: the state is never written, no kernel in the stepping path
  changes, physics WGSL byte-identical. Run the new WGSL through `wgslparse.mjs`
  (reserved words — see the pitfall note).
- No RNG anywhere; the RNG reference must be byte-identical before and after.
- 2D only; no `docs.html` claim that is not on the plot.
- No copy-paste: the axis and legend helpers and `fftPow2` all exist.

## Checks (`devtools/checkeigf.js`, following `checkk` / `checkaniso`)

1. `wgslparse` / `names` / dup; physics stepping WGSL byte-identical; RNG reference
   unchanged.
2. **Gather correctness**: the extracted column equals the CPU-side column of a synthetic
   uploaded state, exactly (fp32 bit compare where the path is a pure copy).
3. **fp64 mirror**: gather + inverse-along-kx + modulus against a direct fp64 computation on
   a synthetic field, ~1e-5.
4. **Equilibrium exclusion**: on a pure equilibrium state (no perturbation), every k_y ≠ 0
   column is zero to round-off — i.e. the subtraction really is exact.
5. State invariance: a draw leaves (φk, ψk) bitwise unchanged.

With the Δ′ readout dropped, checks 2 and 3 are the plan's real gate: the card is a pure
transform of state, and the fp64 mirror is what says the transform is right.

CI reports, never gates.

## On-device checklist (owed after merge)

Readback cost at 256×64 and 256²; legibility of two curves at phone width; whether three
cards fit on `tearing` or the cut trace has to yield; whether the k_y selector is
discoverable enough to be worth its width on `island chain`.

## Sequencing with CHI_PLAN.md

Near-disjoint. `physics.js` and `rmhd2d.html` are this plan's alone; `rmhd3d.html` is
CHI's alone. They meet in exactly three places:

1. **`CHART_TYPES` in `common.js`** — this plan appends a new entry; CHI edits *inside* the
   existing `aniso` entry. Adjacent (~15 lines apart) but disjoint hunks. Rule: **this plan
   inserts its entry immediately after `mode` and does not touch `aniso`.**
2. **`_chartsBySrc` / the readback dispatch** — this plan adds a third source; CHI adds
   none (it rides `src: "spectrum"`). One-sided, so no conflict, but it is this plan's
   riskiest edit and the reviewer should treat it as such.
3. **`docs.html`, `webgpu/README.md`, `SPEC.md`, `plans-webgpu/README.md`** — both plans add
   sections and status rows. Guaranteed textual conflict. Rule: **whichever lands second
   owns the doc reconciliation**, in its own commit.

Either order works. Running them concurrently is fine given rules 1 and 3; running them in
sequence is simpler if one agent is doing both.
