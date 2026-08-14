# Nonlinear tearing plan: two new 2D presets (X-point collapse; island chain + coalescence)

**Status: written 2026-08-13. Phase 0 gate PASSED the same day — all four diagnostic runs
are done and eqSrc off is confirmed for both presets. Phases 1–4 EXECUTED 2026-08-13,
adversarially reviewed, fixes applied; uncommitted pending Alfred's on-device pass. Phase 5
is partly owed — see the note at the end of that section.** The parameters
below are settled on their evidence. Base:
current `rmhd2d.html` `PRESETS`, `common.js` `icRegister("tearing")` / `BOXES` /
`icSyncRows`. Shipped `tearing` preset keeps the FKR/Rutherford comparison and stays the
quantitative one. It ended up modified in two ways during execution, both deliberate and
both recorded below: it must pin `cbTearBroad: false` (or it inherits CHAIN's broadband
seed, `presetWrite` writing only the ids a preset names), and Alfred dropped its default
`selRes` to 256, which costs 2.4% in the γ it quotes.

Provenance: Alfred found this by loading `forced`, switching the IC dropdown to `tearing`
(which locks `hyper` to 1) and leaving low-amplitude OU forcing on as a seed. The physics
turned out not to be what the recipe suggested — see *What the forcing band is actually
doing* — but the two runs it produced are both worth shipping.

## What it is / why

Two presets that sit either side of the tearing instability's nonlinear fork, both reusing
the existing `tearing` equilibrium ψ_eq = ψ₀ sech²((x − Lx/2)/a) with no new physics:

- **COLLAPSE** — large Δ′. The island exceeds the critical width, the X-point collapses
  into a current sheet, the sheet itself tears and produces a secondary island which then
  coalesces with the original. This is Loureiro, Cowley, Dorland, Haines & Schekochihin,
  *PRL* **95**, 235003 (2005), and the run reproduces its abstract point for point.
- **CHAIN** — a long box seeded broadband in k_y, so the sheet selects its own mode from
  Δ′ and grows a six-island chain that then coalesces 6 → 3 → 1.

The pairing with the shipped preset is the real asset: `rEqA` alone moves COLLAPSE from
Δ′a = 37.8 to Δ′a = 8.40, and **8.40 is the shipped preset's own value** (a = 0.2 × 2π
equals its 0.1 × 4π). So COLLAPSE contains `tearing` as a slider endpoint, and one control
walks a viewer across the Loureiro threshold.

## What the forcing band is actually doing (correction to the original recipe)

The recipe was "forcing at eps = 1e-3, band 8–9 → island chain". Measured y-FFT of the
rendered current in a ±40 px strip around the sheet, over all three of Alfred's original
movies, peaks at **n = 1 in essentially every frame**. Linear theory says why — square
box, a = 0.1Lx = 0.628, ψ₀ = 1.65, η = 1e-3, Pm = 1:

| k_y | 1 | 2 | 3 | 4 | 8 | 9 |
|---|---|---|---|---|---|---|
| Δ′a | 37.8 | 8.40 | 1.94 | −1.21 | −8.38 | −9.84 |
| γ | **0.225** | 0.115 | 0.021 | stable | stable | stable |

**The forcing band is tearing-stable.** It is a broadband seed and nothing more; Δ′ picks
the mode. That is the teaching point CHAIN is built around, and it is why CHAIN uses a
deterministic broadband IC seed rather than the OU forcing (which would keep acting
through the merger phase and muddy it).

## The resistive-layer numbers everything below rests on

All computed with the repo's own `devtools/eqlinear.py`, which reproduces the shipped
preset's quoted Δ′a = 8.399 and γ = 0.0287 to all published digits — so the machinery is
validated against the app before being used on anything new. Converged at n = 192.

### Two resolution criteria, not one

Write k_max = (n/3)(2π/L) for the dealiased cutoff; since dx = L/n, **k_max·δ =
(2π/3)(δ/dx) ≈ 2.09 δ/dx**, i.e. "how many cells span the thinnest structure", in
wavenumber units. Two different structures bind at two different stages:

- **k_max,x · δ_lin** where δ_lin = (γη / k_y²B_y′(x_s)²)^¼ and |B_y′(x_s)| = 2ψ₀/a².
  The linear tearing layer. Thin in **x**. Governs the growth phase.
- **k_max,y · δ_SP** where δ_SP = √(L_sheet η / v_A). The Sweet–Parker thickness of the
  sheets formed *nonlinearly*. In coalescence these sit between merging islands, **normal
  to y** — so it is the y grid that must resolve them. Measured off Alfred's test-2 frames,
  L_sheet ≈ 0.35 × island spacing.

**Benchmark: 4.3.** That is the shipped `tearing` preset's k_max·δ_lin (δ/dx = 2.05), and
it reproduces its eigenvalue reference, so ~2 cells is the empirical floor in this solver.
Anything below it rings.

> **Superseded in part, 2026-08-13.** "~2 cells is the floor" was inferred from that ONE
> working configuration; nobody had measured below it. Alfred has now done so and it is
> conservative by about 2× — 1 cell costs 2.4% in γ, and only ~0.5 cells breaks it. The
> measured table is at the end of Phase 4. It is why CHAIN ships at 256² and not the 1024²
> this benchmark, applied literally, demanded. It still stands as a *design* target; it is
> no longer evidence that anything below it "rings".

Getting this wrong once already cost a design: an earlier draft of CHAIN used 512×256 in
the 8π box (dy/dx = 8), which scores k_max,y·δ_SP = **0.7**. Isotropic cells are mandatory
whenever the payload is nonlinear.

### Why COLLAPSE cannot also show a plasmoid chain

Plasmoids need S = L v_A/η ≳ 10⁴, which wants η small; resolution wants δ_SP large, which
wants η large. Square box, L ≈ 2.2, v_A = 2.02:

| η | S | δ_SP | k_max·δ_SP @512² |
|---|---|---|---|
| 3e-3 | 1.5e3 | 0.057 | 9.8 |
| **1e-3** | 4.4e3 | 0.033 | **5.6** ← COLLAPSE ships here |
| 5e-4 | 8.9e3 | 0.023 | 4.0 |
| **3e-4** | 1.5e4 | 0.018 | **3.1** ← the under-resolved teaching case |
| 1e-4 | 4.4e4 | 0.010 | 1.8 |

Imposing both at once needs nx ≈ 614, i.e. 1024². The resolution is to ship the resolved
run and expose the under-resolved one as a labelled experiment (Phase 3 hint text).

*(Execution note: 1024² was ruled out here for every target device, and decision 4 below
said so. Alfred reopened it on 2026-08-13 for CHAIN alone — see Phase 2 — on the grounds
that 1.05 Mpts is exactly the 3D app's own default grid. COLLAPSE still ships at 512² and
this paragraph still describes why it cannot have both S ≳ 10⁴ and a resolved δ_SP.)*

## Runs already performed

| | config | result |
|---|---|---|
| test 1 | η = 1e-3, 512², sq, eqSrc **on**, `rEqPert` −3 | Collapses. Peak \|J\| 8.36 → **82** (×9.8). **One** secondary. Spectrum clean, distinct rolloff. |
| test 2 | as above, η = 3e-4 | 1–3 plasmoids at a time; **spectral bump + Gibbs ringing** — under-resolved, as 3.1 predicts. |
| test 3 | η = 1e-3, eqSrc **off** | **Collapse still happens, secondary island still forms.** |

**Do not quote a current-intensification factor from these runs.** The frames on hand are
single grabs at different times of different runs (test 1 read 82, test 3 read 47 near the
collapse and 25.4 at island formation), and peak |J| varies strongly *through* a collapse —
so those numbers compare nothing. The equilibrium value 2ψ₀/a² = **8.36** is solid; the
collapsed value the hint quotes must be a **maximum over the run**, taken from the shipped
configuration (eqSrc off), with the time it occurs recorded. **Owed — see Phase 5.**

Test 1 settles the question that motivated the whole exercise: the collapse survives fixed,
resolved η, so it is **not** an auto-diss artifact. What distinguishes COLLAPSE from the
shipped preset is Δ′a = 37.8 vs 8.4 — geometry, not dissipation.

## Fixed decisions (do not relitigate)

1. **auto-diss OFF in both.** `autoDissTarget` places the *cascade termination* at the
   dealias scale; it has no term that knows a reconnection layer exists. In Alfred's
   original run, with eps = 1e-3 the shell energy is tiny and ν_target = √(2E_shell)/k_d
   at hyper = 1, so it drove η to **3.2e-9** — γ ≈ 1.8e-3 (e-folding 550), δ ≈ 5e-4, which
   is 0.044 cells at 512² and would need nx ≈ 24,500. Two further reasons: γ ∝ η^0.38 here,
   so drifting η fakes the acceleration the demo is about; and `AUTODISS_PERIOD` is
   wall-clock, so updates per unit *simulation* time scale with steps/s (500/s desktop vs
   36/s on the iPhone 11 per LOOPLAT) — two viewers would get different movies.
2. **hyper = 1, Pm = 1** in both. Pm is load-bearing for CHAIN specifically: γ falls as
   ≈Pm^−0.24 at large Pm and at Pm = 100 the short-wavelength end of the chain goes
   outright stable, so a viewer nudging `nPm` would silently truncate the chain.
3. **eqSrc off in both** — *pending Phase 0*. Neither preset quotes a rate against a
   maintained equilibrium, and a source feeding flux in makes the collapse externally
   driven, which undercuts "watch it relax". Resistive spreading is not a threat: a²/η is
   400 growth times for COLLAPSE, 22 for CHAIN.
4. **No 1024².** ~~See above.~~ Reopened 2026-08-13 when CHAIN needed x room, then
   **closed again the same day**: measuring the resolution floor (end of Phase 4) showed
   512² is inside the good range, so nothing ships at 1024² after all and the decision
   stands as written. `selRes` keeps the 1024 option — it costs nothing in workgroup
   storage, the constraint being per LINE — but no preset uses it.
5. **`selRes` is NOT locked.** Alfred's call: an under-resolved run is instructive and
   should be reachable. See the hint text in Phases 3–4.
6. Shipped `tearing` preset untouched. *(Amended during execution — see the header. The
   FKR/Rutherford comparison and every physics parameter are untouched; what changed is
   one flag it must pin to keep that true, and its default resolution.)*

## Phase 0 — run test 3 (gate) — **DONE, PASSED 2026-08-13**

η = 1e-3, 512², sq, `rEqA` 0.1, **eqSrc off**, everything else as test 1. Question: does
the collapse survive with the drive removed? **Yes** — collapse happens and the secondary
island still forms, at peak |J| = 47 rather than 82.

So decision 3 stands: eqSrc off in both presets, and COLLAPSE is a genuine relaxation
demo, not forced reconnection. Nothing about the design changes; only the |J| figure the
hint quotes. Start at Phase 1.

## Phase 1 — broadband-in-k_y seed for the tearing IC

`icRegister("tearing")` in `common.js` currently seeds one mode:

```js
sd[i] = A * s2;                    // single k_y = 2pi/Ly, even envelope
```

`icPlaneFromX(ps, sd, g)` multiplies that x-envelope by cos(k_y y) at the box's lowest
k_y. CHAIN needs equal-amplitude, random-phase seeding across k_y = 1..N so that Δ′ does
the selecting. This is a change to the **IC builder**, not a preset — it is the one place
this plan breaches "a preset is a UI concept only", and it needs its own review.

- New control on `rowTear` (a checkbox is enough: **broadband seed**), default off so the
  shipped `tearing` preset is byte-identical.
- When on, sum modes 1..N with equal amplitude and random phase, normalised so the *total*
  seed at the resonant surface is still the `rEqPert` slider value — otherwise the slider
  stops meaning what `icEq.w0` says it means and the island chart's W₀ is wrong.
- N should cover the unstable band with headroom; N = 24 at the CHAIN parameters.
  *(Corrected during execution: the stated reason — "unstable to k_y ≈ 6 = box mode 24" —
  is wrong. Δ′ goes marginal at k_y a = √5, i.e. k_y = 4.75 = mode 19, and γ goes negative
  at mode 15/16. N = 24 is still right, by the other route: ⌈1.25 × 18.98⌉ = 24. The code
  derives N that way, so it tracks the `a` and box sliders rather than being a constant
  that happens to fit one preset. Do not "fix" the code back to the plan here.)*
- `icEq.w0` = 4√(A/curv) assumes a single mode. Decide explicitly what it means for a
  broadband seed — simplest defensible choice is to leave `icEq.on = false` for the
  broadband case so the island-width chart does not quote a number it cannot support.

**RNG note.** This is RNG-adjacent. Per the standing rule, **record the RNG reference
before touching anything**, so the existing seeded-IC reproducibility can be shown intact
afterwards.

## Phase 2 — new `big` box

`BOXES` in `rmhd2d.html` (line ~115) gains a third entry. `fy` is ny/nx:

```js
big: { t: "large 8&pi; &times; 8&pi;", fy: 1, Lx: 8 * Math.PI, Ly: 8 * Math.PI }
```

At `selRes` 1024 that is **1024 × 1024**, dx = dy = 0.02454 — isotropic, which Phase 4
requires — and 1.05 Mpts.

**Revised TWICE from the drafted 2π × 8π at `selRes` 256**, both times because Alfred ran
it and the islands did not fit: 2π → 4π → 8π, and the box is now square, so `tall` became
`big`. The reason it kept happening is that an island's extent *across* the sheet grows
with its wavelength *along* it, and each coalescence doubles that wavelength — so the last
merger is much the widest, and sizing the box on the first one under-provisions it.
Separatrices reaching the periodic x boundary both look wrong and destroy the far field
Δ′ is defined against. Lx/a is now **53.3**, from 13.3.

`a` is held at its absolute 0.4712 throughout, so `rEqA` tracks Lx down to 0.01875 — which
needed the slider's min lowered to 0.0125 and its step to 0.00125 (every preset value is a
multiple; 0.02, the old min, moves the fastest mode from 6 to 5, so snapping was not an
option). γ shifts 0.3113 → 0.3065 with the wider box, and the resolution scores are
unchanged at 5.29 / 5.87 because dx never moved.

**Raising η instead was considered and rejected.** Buying the resolution back at 512² takes
5× the resistivity to reach the 4.3 benchmark and drops S to 91, which stops the mergers
being the clean reconnection the preset is about. The cost is points and only points.

**`selRes` gains a 1024 option.** This costs nothing in workgroup storage — the constraint
is per line and the longest line was already 1024 — and it is the plan's fixed decision 4
("no 1024²") knowingly reopened by Alfred on the grounds that 1.05 Mpts is exactly the 3D
app's default grid. Note the consequence: 1024² is now reachable for the *square* box too,
so a viewer can drag `collapse` there. Nothing ships at 1024² except `chain`.

Check as part of this phase, not later:

- `selBox` is already in `rebuildOn`, so the rebuild path is free.
- Spectrum `kunit` moves with the box (`common.js` ~1488 has the precedent for the 4π
  case); confirm the chart axes and the band sliders follow.
- `cardAspect` gives Lx/Ly = 0.25 → a **128 × 512** display card. Given the recorder
  history, **verify the H.264 encoder accepts that geometry on device** — 512×512 and the
  1024×256 wide box are both 1024 MBs and this one is not.

## Phase 3 — preset COLLAPSE

Square box, and the interesting control is `rEqA`.

```
set: { selRes: "256", selBox: "sq", selHyper: "1", nPm: "1", cbForce: false,
       rDiss: "-3", cbAutoDiss: false,
       selIC: "tearing", rEqA: "0.1", rEqPert: "-4", rEqPsi0: "1.65",
       cbEqSrc: false, rCflEvery: "4" }   // rEqPert -4 and selRes 256, both revised: below
layout: { disp: [{ sel: 1, cont: 3, nlev: 16 }], charts: ["island", "energy", "spectrum"] }
```

The spectrum chart is not decoration — it is what the hint's resolution experiment points
at. `rDiss: "-3"` is η = 1e-3.

`rEqA` sweep, all at 512² / η = 1e-3, computed for the hint text. **Seed revised to
`rEqPert` −4 during execution** (Alfred's call — see the note under the table):

| `rEqA` | a | Δ′a | γ | W_c = 1/Δ′ | W₀ at −4 | W₀/W_c | W₀/δ_lin | δ_lin/dx @256² | linear stage |
|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.628 | **37.8** | 0.225 | 0.0166 | 0.0138 | 0.83 | 0.33 | 1.73 | 10.0 |
| 0.125 | 0.785 | 23.9 | 0.131 | 0.0328 | 0.0173 | 0.53 | 0.37 | 1.88 | 15.1 |
| 0.15 | 0.942 | 16.3 | 0.078 | 0.0579 | 0.0208 | 0.36 | 0.43 | 1.98 | 22.0 |
| 0.175 | 1.100 | 11.6 | 0.047 | 0.0951 | 0.0242 | 0.25 | 0.48 | 2.04 | 30.9 |
| 0.20 | 1.257 | **8.40** | **0.029** | 0.1496 | 0.0277 | 0.18 | 0.54 | 2.07 | 41.8 |

δ_lin/dx never drops below 1.73 (it was 3.45 at the drafted 512²), so a viewer cannot break
it by dragging — every row is above the 1.03 cells the calibration measures at −2.4%. The bottom row is
the shipped preset's physics to within 1%.

**Why −4 and not the −3 this plan was written with.** At −3, W₀ = 0.0438 against
δ_lin = 0.0423 — the seed island is *already as wide as the resistive layer it is supposed
to grow out of* (W₀/δ_lin = 1.03), so there is no linear stage and the island chart's
headline claim of two straight segments at different slopes is a segment short. −4 gives
W₀/δ_lin = 0.33 and ~10 time units of clean exponential growth first.

It also repairs the narrative. At −3, W₀/W_c = **2.63**: the island starts *past* the
critical width the hint describes it as passing, and the old "W₀/W_c crosses 1 at
`rEqA` ≈ 0.16, mid-slider" line was the only thing making that coherent. At −4 the ratio
is 0.83 at `rEqA` 0.1 and falls monotonically to 0.18 — below 1 across the whole slider —
so the island grows *through* W_c at the collapse end and never reaches it at the
Rutherford end. The contrast the slider exists to draw now happens in time, which is what
the log W(t) chart actually plots. Cost: ~10 time units before anything is visible.

Hint text must carry:

- ψ_eq and what Δ′ is, then: at `rEqA` 0.1, Δ′a = 37.8 and the island passes the critical
  width W_c ~ 1/Δ′ almost immediately, so instead of saturating it collapses the X-point
  into a sheet — cite Loureiro et al. 2005.
- The colourbar is doing quantitative work: equilibrium peak |J| = 2ψ₀/a² = **8.36**, and
  the collapse drives it to *N*× that. **Do not fill in N from the existing frame grabs**
  — they are single frames from different runs at different times and are not comparable.
  Measure max |J| over the shipped configuration first (Phase 5).
- **The slider is the experiment**: drag `rEqA` to 0.2 and Δ′a falls to 8.4, the same value
  the `tearing` preset runs at, and the island saturates into a Rutherford island instead.
- **The resolution experiment**: drop diss to 3e-4 and the single secondary becomes a chain
  of them — then look at the spectrum, which has grown a bump at the high-k end, and at the
  ripples in the island interiors. The Sweet–Parker sheet is ~1.5 cells across at that η,
  so the run cannot represent it, and you cannot tell from the picture whether those extra
  islands are plasmoids or numerics. Optionally: tick auto-diss and watch the bump vanish
  because ν rose to meet the grid — spectrum fixed, physics now wrong.
- Do **not** claim plasmoid instability at the shipped η. S = 4.4e3 there, below the ~10⁴
  threshold, and the run shows *one* secondary, which is ordinary secondary tearing —
  exactly Loureiro's "the sheet can itself become tearing-mode unstable".

Second quotable if the island chart cooperates: the paper's collapse-phase reconnection is
exponential with rate ∝ η^½, distinct from the linear γ ∝ η^0.38 measured here, so log W(t)
should show two straight segments with different slopes.

## Phase 4 — preset CHAIN

```
set: { selRes: "256", selBox: "big", selHyper: "1", nPm: "1", cbForce: false,
       rDiss: "-2.5", cbAutoDiss: false,
       selIC: "tearing", rEqA: "0.01875", rEqPert: "-3", rEqPsi0: "0.60",
       cbEqSrc: false, cbTearBroad: true, rCflEvery: "4" }
// selRes/selBox/rEqA revised twice during execution -- see Phase 2. a is unchanged at
// 0.4712 throughout; only the box it sits in grew, so every number below still holds
// except gamma, which the wider box moves 0.3113 -> 0.3065.
layout: { disp: [{ sel: 1, cont: 3, nlev: 16 }],
          charts: [{ t: "cut", pair: "b" }, { t: "cut", pair: "u" }] }
// charts revised 2026-08-13 (Alfred): the two cut traces, not spectrum + energy
```

(`cbTearBroad` is Phase 1's new control; name it as that phase decides.)

Resulting numbers, at the slider-quantised values above:

- a = 0.4712, max|b| = v_A = 0.980, η = 3.2e-3
- fastest mode **k_y = 1.5 → 6 islands**, γ = 0.3065, e-folding 3.3 (0.3113 in the drafted 2π box)
- k_max,x·δ_lin = **2.64**, k_max,y·δ_SP = **2.93** at the shipped 512² — BELOW the 4.3
  benchmark, deliberately, on Alfred's measurement rather than in spite of it. See below.

Alfred has accepted six islands — but see Phase 5 item 3: γ is flat near its peak, his run
showed **four**, and the shipped hint now promises "roughly six" and warns the count moves
with the seed. A nine-island variant (`rEqA` 0.05, ψ₀ 0.40) scores 5.0/6.0 but only at
320×1280, which `selRes` still does not offer. Not worth it, and now doubly so: at a
9-island spacing the mode degeneracy is worse, not better.

Hint text: seed everything flat, watch the sheet pick k_y = 1.5 out of 24 available
unstable modes because Δ′ is largest there, then three generations of coalescence
6 → 3 → 1. The resolution experiment here is `selRes` → 128, which drops the scores to
2.7/2.9 and puts ripples in the merger sheets.

## Phase 5 — verification

1. Both presets reproduce the uploaded movies qualitatively (COLLAPSE against test 1;
   CHAIN has no prior movie, so judge it against the predicted six islands and γ = 0.311).
2. **Measure max |J| over the run** for COLLAPSE in the shipped configuration, and record
   the time it peaks. This is the number Phase 3's hint text needs and it is deliberately
   left blank there. A single frame grab is not an acceptable substitute.
2. **Spectrum check on every shipped configuration** — no bump at the dealias end. This is
   now the house test for these presets; it is what caught test 2.
3. Measured island count in CHAIN = 6, and the fastest mode is not the lowest box mode
   (if it is, Δ′ selection is not being demonstrated and something is wrong).

   **Read this before judging the count.** γ is FLAT near its peak: mode 6 is only 1.8%
   faster than mode 5 and 2.6% faster than mode 7 (γ = 0.3113 / 0.3056 / 0.3033). Over the
   ~14 time units of linear growth from the `rEqPert` −3 seed, mode 6 ends just **8% above
   mode 5** and 12% above mode 7 — so with equal-amplitude random-phase seeding the
   realised pattern is a superposition of modes ~5–7 at comparable amplitude, and the count
   you see may well be 5 or 7, with irregular spacing, and may vary with `#nSeed`. That is
   the physics, not a bug: Δ′ selection here is broad. The check that actually discriminates
   is the one in the parenthesis — the winner must not be the box fundamental (γ = 0.106,
   3× down) — plus "6 ± 1 islands, roughly evenly spaced". A clean, repeatable 6 would be
   a bonus, not the pass condition. Lowering the seed does not buy much: at −5 the linear
   stage roughly doubles and mode 6's margin over mode 5 only reaches 18%.

   **Decision owed once the run exists** (Alfred, 2026-08-13): the shipped hint currently
   promises "k_y = 1.5, i.e. six islands". If the run shows the spread above, soften it to
   the band peaking near k_y = 1.5 and make the breadth of the selection the teaching point.
4. Preset switching: into and out of `big` rebuilds cleanly; `icSyncRows`' hyper lock
   engages and releases correctly across all four tearing-family presets; the broadband
   checkbox does not leak into the shipped `tearing` preset.
5. RNG reference from Phase 0 still reproduces.
6. On-device pass including the recorder on the 128 × 512 card.

**Execution note, 2026-08-13.** Items verifiable without a GPU are DONE: preset switching
and the hyper lock across all four tearing-family presets, the broadband flag not leaking
into shipped `tearing` (every preset selecting the tearing IC now pins it explicitly —
`presetWrite` only writes the ids a preset names), the single-mode IC bitwise unchanged,
the RNG reference reproducing byte-for-byte, and every linear number in this plan
recomputed independently. **Still owed, and needing Alfred at a browser:** items 1 and 3
(qualitative match, measured island count = 6), item 2 in both its senses — the max |J|
measurement the COLLAPSE hint carries a placeholder for, and the no-bump spectrum check on
both shipped configurations — and item 6.

Two blockers found in adversarial review and fixed here, both about the 1024-point line
the long-y box needed: it takes 16 KiB of workgroup storage, which is exactly the WebGPU
guaranteed minimum, and SPEC §8 had declared that size closed for precisely that reason.
The 2D page now boots `initGPU({maxLimits: true})` and `NMAX_LINE` caps n_x (never n_y —
that would shear the cells) so every offered grid still fits a default device; SPEC §8 and
the README record the reversal. Before the cap, the then-`tall` box at `selRes` 512 asked
for 32 KiB and would have failed pipeline creation outright — reachable in two clicks from
`chain`. The box has since become square, so nothing in `BOXES` trips that clamp today; it
is kept because the rule is about the table and not about its current contents.

## Hint-text convention (applies to both)

**Write dissipation as a number, not as the slider's log.** "drop diss to 3e-4", never
"drop diss to −3.5". The control is log₁₀ but nobody reads it that way. Same for any η
quoted in prose.

## Loose ends found while planning — not part of this plan

1. **The shipped `tearing` hint claims γ ∝ η^{3/5}.** Measured η^0.49 over
   η = 3e-4…1e-2, and η^0.54 locally at 1e-3. Δ′a = 8.4 is not small enough for the
   constant-ψ asymptotic. This is public-facing text on a page Kunz is linking from a
   problem sheet — worth fixing on its own.
2. **ψ contours speckle into broken cyan dashes over large parts of the frame.** Visible
   in test 1, and much worse in test 3, where they cover most of the box. **This is now a
   blocker for COLLAPSE, not a cosmetic note** — the preset ships with `cont: 3, nlev: 16`
   and a viewer's first impression would be that the run is full of noise.

   Working hypothesis: ψ_eq = ψ₀sech²((x−Lx/2)/a) is a localised bump, so ψ → 0 across
   most of the box. Evenly-spaced contour levels then put a level *inside* the near-zero
   far field, where ψ is flat to well within the level spacing, and the tracer speckles.
   This predicts the speckle grows with time as ψ flattens — which is consistent with the
   frames on hand, but note they are at different times, so they do **not** establish that
   eqSrc off is worse than eqSrc on. Cheap discriminators, in order: set `nlev` to 0 and check the
   underlying J field is smooth there (if it is not, this is ringing and a physics
   problem); raise `nlev` and see whether the speckle densifies smoothly (real structure)
   or stays pinned to a lattice (tracer artifact); try quantile-spaced levels or a
   |∇ψ| threshold below which no contour is drawn.
3. **auto-diss + hyper = 1 is an untested combination** and produced η = 3.2e-9 from a
   weak forcing state. Not a bug — the controller is doing what it says — but it is a
   trap for anyone repeating Alfred's original recipe. A guard, or a line in the docs.
4. **eqSrc tracks η correctly** — the source is −lin_L·ψ_eq,k evaluated against the live
   dissipation array, so auto-diss does not desynchronise it. Checked, no action needed;
   recorded so nobody re-investigates.


### The 4.3 benchmark, measured (2026-08-13)

The benchmark was inferred from ONE working configuration — the shipped `tearing` preset at
δ_lin/dx = 2.05 reproduces its published γ — and nothing had ever probed the other side of
it. Alfred did, on the only preset whose growth rate the app actually fits and whose
reference value is published:

| `tearing` `selRes` | δ_lin/dx | fitted γ | error vs 0.0287 |
|---|---|---|---|
| 512 | 2.06 | 0.0287 | — (the calibration point) |
| 256 | 1.03 | 0.0280 | −2.4% |
| 128 | 0.52 | 0.0315 | +9.8%, and visible junk early on |

So **~1 cell across the linear layer costs a couple of percent**, and the error only becomes
serious — and non-monotonic, which is the real tell — at ~0.5 cells. The benchmark is
conservative by about a factor of two.

CHAIN therefore ships at **512²** (δ_lin/dx = 1.26, inside the measured-good range) and not
the 1024² this plan reached by applying 4.3 literally. At 256² it would be 0.63 cells, i.e.
the row that misbehaves.

**What this does NOT license.** The clean spectra Alfred pinned across 1024²/512²/256² are
necessary but not sufficient, and were never the test: the linear layer sits ~4 decades down
the spectrum and carries far too little energy to disturb it, so a spectrum can look perfect
while γ is wrong. The rows above are the evidence; the spectra only rule out pile-up. Note
also that this calibration measured **δ_lin**, which is the sharp criterion: δ_lin comes out
of the eigenvalue problem with a definite coefficient, so "1.03 cells" means something and
the γ column can be read against it.

**δ_SP is not that kind of number and should not be quoted as if it were.**
δ_SP = √(L_sheet η / v_A) carries an undetermined O(1) prefactor, and L_sheet itself is the
0.35 × island-spacing eyeballed off Alfred's test-2 frames. Its k_max·δ_SP scores are
therefore order-of-magnitude guidance for *design* — they correctly rejected the
dy/dx = 8 draft, which scored 0.7 — and nothing finer. Alfred's convergence check (island
count and spectrum holding across 1024²/512²/256²) is the appropriate evidence at that
level of precision, and is what the merger phase rests on. Do not tighten a δ_SP score into
a pass/fail the way δ_lin's can be, and do not quote δ_SP cell counts to two figures in
user-facing text.

### Default resolution dropped to 256 across the family (2026-08-13)

Alfred's call, having run all of them: 256 is 4× cheaper and he judged every one of them
fine at it. Where each lands against the measured floor above:

| preset | grid | δ_lin/dx | cost |
|---|---|---|---|
| `collapse` | 256² | **1.73** | inside the measured-good range; nothing owed |
| `tearing` | 256×64 | **1.03** | exactly the −2.4% row: the app now fits γ = 0.0280 against the 0.0287 its own hint quotes |
| `chain` | 256² | **0.63** | the +9.8% row — but `chain` displays no fitted rate, and its γ is prose from linear theory |

`kh` stays at 512: it is an ideal instability with no resistive layer, so none of this
applies to it, and it was not part of the change.

Two things to be conscious of rather than to fix blindly. **`tearing` is the preset whose
entire purpose is a quoted growth rate**, and it now reproduces its reference to 2.4% rather
than to the sub-percent it did at 512 — which is still an agreement a viewer reads as
success, and is the honest number they will actually measure, but it is a deliberate
softening of the one quantitative claim on the page. **`chain` sits below the floor**, and
what protects it is only that it makes no measured claim: its six-island selection and clean
spectrum both survive (Alfred checked across three grids), and its γ is a linear-theory
number in prose. If either preset is ever made to quote a *fitted* rate, revisit this.

### Island-chart fit starvation, found and fixed (2026-08-13)

Alfred: "the width chart often fails to display a growth rate". Cause, and it is a
regression from the resolution drop above rather than anything about tearing:

`islandPush` rides the cut readback, which is throttled on a **wall clock** (~10 Hz), while
`fitLogSlope`'s window is `MODE_FIT_DT` = 10 units of **sim** time. So the samples in the
window are `100 / (sim-units per second)` — a quantity that depends on the grid and the
machine, not on the physics. `tearing` at selRes 512 ran ~4 sim-units/s (~26 samples); at
256 the point count fell 4× and dx doubled, so it runs ~19 (~6 samples) and a quicker
machine reaches ~38 (**3 samples**), below the 4 a least-squares slope needs. The legend
then blanks in the middle of a perfectly clean linear stage.

Nothing was wrong with the fit — on a clean exponential it returns the exact γ whenever it
has the samples. It was being starved. Fix: `fitLogSlope` now widens the window until it
holds `MODE_FIT_N` = 8 samples, capped at `MODE_FIT_DT_MAX` = 4×`MODE_FIT_DT`. Widening
cannot fake a rate — the R² ≥ 0.98 gate is what rejects a window straddling two stages, and
it does not care how the window was chosen. Regression test in `devtools/checks.js` §11
sweeps 4–100 sim-units/s and requires the exact γ throughout, checks the cap still gives up
past it, and re-runs the saturated / decaying guards at a spacing that triggers widening.

The alternative fix — sampling on sim-time as well as wall-clock — was rejected: it means
extra GPU readbacks per unit sim-time exactly on the fast configurations, which is the
trade LOOPLAT closed.

**Watch for this anywhere a sim-time window meets a wall-clock sample.** The KH `mode`
chart shares the helper and so is fixed too; its own γ = 0.267 is fast enough that a
`MODE_FIT_DT` window was never starved at the grids it ships on, but the same arithmetic
applies if it is ever moved down the resolution ladder.

### Spectrum y floor: the `clip tail` option (2026-08-13)

Alfred: the spectra "are routinely cut off at too high of a power in these tearing runs, so
it is hard to see if there is instability at the small-scale end" — i.e. the exact chart the
house test (Phase 5 item 2, "no bump at the dealias end") is read off.

Cause. `specFloor` puts the y floor SPEC_TAIL = 3 decades below the spectrum at half the
dissipation knee. That rule was calibrated (FEEDBACK_2026-08-08 item 4) on a **hyper = 4**
tail, which runs 15+ decades below the peak and would otherwise squash the inertial range
into a few pixels. Every tearing-family preset locks **hyper = 1**, where the fall-off is
gradual: on a representative run the floor sits 5.6 decades down while the data span 6.9, so
the last ~1.3 decades — the dealias end — are drawn below the axis. Worse, a run *with* a
bump and one without are then clipped to the same picture, so the test cannot distinguish
the case it exists to catch.

Fix, Alfred's design: a per-card **`clip tail`** checkbox. On (default) is the historical
rule exactly; off draws every bin. SPEC_MAXDEC still clamps **both** states — unclipping
drops the knee rule, not the sanity limit, or an early frame whose tail is at the fp32 noise
floor would draw a 12-decade axis.

Two things worth keeping:

- The default is **per preset, not global**, so `collapse` and `chain` open unclipped and
  `forced` / `decay` / the 3D presets are untouched bit for bit. That needed a preset's
  `charts` entry to be able to carry options — `{t: "spectrum", clip: "off"}` — which is
  what `disp` entries have always been (`{sel, cont, nlev}`), so the two halves of a layout
  now agree rather than one gaining a new concept.
- Rejected: making SPEC_TAIL follow `hyper`. It would have been narrower, but the floor rule
  is deliberately *measured off the spectrum* and assumes nothing about where the energy
  came from (the same principle as the amplitude-based auto-diss); reading a UI setting
  would have broken that. Also rejected: "show everything that fits in SPEC_MAXDEC", which
  fixes tearing but takes `forced` from 4.5 to 9.0 decades — precisely the regression item 4
  exists to prevent.

Tests: `checks.js` §11b pins both states on synthetic gentle and hyper-dissipative tails
(including that a dealias-end bump is invisible clipped and visible unclipped, and that
clip-on is bit-for-bit the historical `specFloor` under the clamp); `bootstub.js` pins the
per-preset defaults and that the checkbox toggles live.

### Alfred's text rewrite (2026-08-13) — and what it deliberately drops

Both presets retitled into a family: **`tearing: X-point collapse`** and **`tearing: island
chain`**, alongside the shipped `tearing`. Hint text replaced with Alfred's own, and the
island-width chart's caption shortened to match. The style is short, lowercase, and asks
questions rather than listing conclusions.

One consequence worth recording: COLLAPSE's hint now *asks* "what is the peak current in the
collapsed sheet compared to the initial equilibrium?" instead of quoting a factor. That
**retires the placeholder** this plan left for Phase 5 item 2 — the colourbar answers it
live against the analytic 8.36, and nothing user-facing is waiting on a measurement any
more. Measuring max |J| over a run is still worth doing for the record; it is no longer
blocking.

Claims the previous text carried that the new text does not. None of these is an oversight
to be reinstated without asking — the brevity is the point — but they are no longer written
anywhere a viewer can see, so they are recorded here:

- **The `Pm` = 1 warning (CHAIN).** Fixed decision 2: γ falls ~Pm^−0.25 over Pm 1–10 and the
  short-wavelength end of the band is stable by Pm = 100, so a viewer who nudges `Pm` gets a
  quietly truncated chain and is not told. The preset's code comment is now the only record.
- **"This is not a plasmoid instability" (COLLAPSE).** The S ≈ 4.4e3 vs ~10⁴ argument is
  gone from the hint. The replacement covers the same ground differently — the η = 3e-4
  experiment is now flagged as "isn't physical … the collapsed sheet wanting to be thinner
  than the grid" — so the under-resolution warning survives; the explicit plasmoid
  disclaimer does not.
- **The `rEqA` slider walk (COLLAPSE).** "Drag a / L_x to 0.2 and Δ′a falls to 8.4, the
  value the tearing preset runs at" — the exact-equivalence this plan called "the real
  asset". The new text states the 37.8 vs 8.4 contrast but does not tell the viewer the
  slider gets them there.
- The η-exponent figures, the W₀/W_c numbers across the slider, and the spectrum-bump
  discussion. All still in this plan; none now on the page.

### CHAIN's charts: the two cut traces (2026-08-13)

Alfred's call: `chain` opens on **cut trace b_x/b_y** and **cut trace u_x/u_y** rather than
spectrum + energy. The cut line is x = L_x/2, the resonant surface, and the equilibrium is
y-independent with its field along y — so u_x and b_x carry **exactly zero** equilibrium
content there (the same property the KH mode chart is built on). What those traces show is
the perturbation alone, mode by mode along the chain, which reads the island count far more
directly than the picture does — and the picture is the thing the hint now warns "runs too
fast to check island numbers" on a quick machine.

Consequence: `chain` no longer opens a spectrum card, so the `clip tail` default set for it
above now only applies to `collapse`. A viewer who adds a spectrum card to `chain` by hand
gets the default (clipped) — there is no per-preset memory for a card the preset did not
open. Worth knowing when running the Phase 5 spectrum check on CHAIN: untick `clip tail`
first, or the dealias end is below the axis.

This is also the first use of the chart-options-in-a-layout mechanism for a SELECT rather
than a checkbox (`{t: "cut", pair: "b"}`), which is what makes two cards of the same type
open on different components.