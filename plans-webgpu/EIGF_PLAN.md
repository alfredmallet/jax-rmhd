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
distinct bin on screen), the k_y (1…6) and field (ψ/φ/both) selectors, `avail: cfg =>
!cfg.zslice`, the card as `tearing`'s third chart and available-not-default everywhere
else, and the side task (the stale 30–40 % clause deleted from the `tearing` hint, the
diffusion statement it hung off kept). Docs: `docs.html` #eigf, `webgpu/README.md` (chart
options table + a "Display modes" subsection), `SPEC.md` §7.

Execution notes worth keeping:

- **The plot is a genuine measurement of the transform, not of itself.** Check 3 builds the
  state by a direct fp64 forward DFT of ψ = g(x)cos(k_y y + p₀), φ = h(x)sin(k_y y + p₀)
  and compares the card's output with the ANALYTIC g/2 and h/2 — agreement 9.3e-9 and
  7.8e-9 (tol 1e-5), which also pins the direction of the transform and the normalization.
  The two shape claims the card exists for fall out of the same mirror rather than out of
  prose: ψ̂ peaks exactly on ix = NX/2 and |φ̂(x₀)| is 2.3e-9 of its lobes.
- **The gather is bit-exact**, so "minus the equilibrium" is a property of the STATE and
  not of any code: check 4 puts a y-independent ψ_eq in and finds every k_y > 0 column at
  9.3e-16 of the k_y = 0 one.
- **Three cards on `tearing`** was shipped rather than the plan's fallback (`eigf` replacing
  the cut trace). `devtools/layout.js` passes at 360 px with the new card's header measured
  (widest single item 284 px of 316 available), but the plan's real question — whether three
  cards *read* well on a phone — is an eye question and stays on the on-device list.
- The same stale 30–40 % number survives in two DEVELOPER-facing places the side task did
  not name: `common.js`'s `ISLAND_FIT_RISE` comment (where it is load-bearing arithmetic for
  a gate constant) and one parenthetical in `webgpu/README.md`. Left alone deliberately —
  the ruling was about the user-facing hint, and re-deriving the gate constant is not this
  plan's business.

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
k_y = 0 column by construction: every equilibrium seed has zero mean along y, which is
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
