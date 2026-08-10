# Pin-curve plan: frozen ghost spectra on the spectrum chart card

One small feature, three phases, executed by agents in order, each gated. Base: current
layout (`common.js` chart-card system, `rmhd2d.html`, `rmhd3d.html`, `style.css`).
Guiding principle: **a pin is a snapshot of what the card is drawing, taken and drawn
entirely on the CPU side** — no new WGSL, no new buffers, no readback changes, no per-app
edits. Everything lands in `common.js` (+ a line of `style.css` if the header buttons need
one). The physics contract (SPEC.md/SPEC3D.md) is untouched.

## What it is / why

A "pin" button on each spectrum chart card freezes the currently drawn curve set as grey
ghost lines that stay on the chart while the live spectra evolve. This is the comparison
primitive several planned lessons need: zeroth law (pin, lower diss, compare), forcing
amplitude (pin, change eps, compare), universality (pin the decayed A/B spectrum, restart
forced, compare), inverse cascade (pin at t=0, watch E_b pile up leftward of the forcing
shell). One diagnostic serves all four, at zero GPU cost.

## Design decisions (fixed here so the phases don't relitigate them)

- **Per card**, like the fit line: each spectrum card owns its pins; two cards can carry
  different pinned states over the same run. Max **4 pins per card**; a 5th press is
  refused with `showStatus("at most 4 pinned spectra — unpin first", "info")` (matches
  the card-count idiom).
- **Snapshot = the drawn curves**, not the raw bins: the `[pts, colour, dash, label]`
  list exactly as `drawSpectrum` builds it (post `specSeries`/`parKfac` transform),
  deep-copied to plain arrays, plus `{ t, kunit, sq, sd }` at pin time (`t` = last
  `hist.t` entry, 0 if none; `kunit` = `solver.g.kunit`). Snapshotting drawn curves means
  the ghost is immune to later changes of the card's `sq`/`sd` selectors — a pin taken as
  "E+ / E-" stays E+/E- however the live view is switched. That is a feature, not a bug:
  the pin is a record of a moment.
- **Physical-k registration across box changes**: pinned x values are in k/kunit units of
  the *pin-time* grid. At draw time every pinned x is multiplied by
  `kunit_pin / kunit_live`, so a pinned curve stays at the same **physical k** if the box
  (and hence kunit) changes (e.g. standard ↔ wide 4π×2π). With an unchanged box the
  factor is 1 and nothing moves. Resolution changes (nb changes) need nothing: the x axis
  is log k/kunit out to the live nb, and the existing clip rect crops any pinned tail
  beyond it.
- **y-range and floor**: pinned PERPENDICULAR series participate in hi/lo and in the
  `specFloor` knee walk exactly as live perpendicular series do — otherwise the
  comparison the feature exists for can sit outside the axis. Pinned parallel series
  never stretch the range, same rule as live parallel (Alfred 2026-08-06). A card with
  pins but no live data yet (just after an IC reset, inside the ~300 ms spectrum
  throttle) draws axes + ghosts from the pins alone instead of "spectra — waiting…".
- **Rendering**: each pinned curve keeps its own hue but drawn first (under the live
  curves) at `globalAlpha` 0.45 for the newest pin, 0.34 / 0.26 / 0.20 for older ones,
  `lineWidth` 1.0, dash preserved (so pinned-parallel stays dashed). Forcing-shell
  markers and fit lines are NOT pinned — the fit line already covers "where should the
  slope be", the pin covers "where was the curve".
- **Legend**: pinned entries appended after live ones as `label @t=12.3` (t to 1
  decimal), same colour at the same reduced alpha, relying on the existing legend wrap.
  If that crowds small cards, the fallback (Phase B may choose after looking at it) is a
  single collapsed entry `2 pinned @t=8.1, 12.3` in `COL.txt`.
- **Lifetime**: pins live on the ChartCard instance. They survive parameter changes, IC
  resets, `chartsReset`, pause, resolution/box rebuilds. They are cleared by: the unpin
  button, retyping the card to another chart type, closing the card. `cardsLayout`
  (preset switches rebuild the whole layout) must NOT silently eat them — see Phase C.
  No cross-reload persistence (no storage APIs).
- **UI**: two header buttons beside the fit controls — `pin` (always visible once the
  card has received data; disabled before) and `unpin` (visible only when pins exist;
  one press clears ALL pins on the card — pins are cheap to retake, an undo stack is
  not worth its UI). Buttons are a new `k: "btn"` item kind in the CHART_TYPES `opts`
  spec, since the existing kinds are value-carrying (select / num) and buttons are not.

## Phase A — seams, no behavior change

1. **`specCurves(d, o)`**: factor the curve-building front half of `drawSpectrum`
   (series selection, perp/par point assembly, hi/lo/hiP/loP accumulation) into a pure
   function returning `{ curves, hiP, loP, hi, lo, nPerp }`; `drawSpectrum` consumes it.
   Pure = no canvas, no DOM, no globals beyond the COL constants — node-testable.
2. **Last-data cache**: `ChartCard.draw(data)` keeps `this.lastData = data` when data is
   non-null and the type is spectrum (cleared on retype). This is what the pin button
   snapshots — it must NOT be cleared by `chartsReset` (a reset is exactly when a pin
   was probably just taken).
3. **`k: "btn"` opt kind** in `ChartCard.build()`: `{ id, k: "btn", t, ti, onClick }`,
   rendered as a small header button, `onClick(card)`; `vis` hook works as for the num
   boxes but is re-evaluated from a `card._optSync()` call sites already make. No
   spectrum-specific logic in this step — the kind is generic.

GATE A: `node --check` on common.js; a node harness feeds `specCurves` synthetic bin
stacks (2D-style perp-only and 3D-style perp+par+fl, incl. empty/one-point/NaN lanes)
and asserts its output reproduces the pre-refactor inline logic (port the old loop into
the test as the reference); both apps' self-tests still green (standing gate — nothing
GPU-side changed, so a failure means the refactor broke page boot); visual spot-check
that all four `sq`×`sd` combinations draw pixel-identically by eye in both apps.

## Phase B — pin, unpin, ghost drawing

1. **State**: `this.pins = []` on ChartCard; each entry
   `{ curves, t, kunit, hiP, loP }` (curves deep-copied to plain Arrays from
   `specCurves(this.lastData, this.optVals())` at press time).
2. **Buttons** (via the Phase A btn kind, spectrum type only): `pin` — disabled until
   `lastData`; appends a snapshot or refuses at 4. `unpin` — visible iff pins exist,
   clears them, redraws.
3. **Drawing**: `drawSpectrum` gains an optional `pins` argument (threaded from
   `ChartCard.draw` — the CHART_TYPES draw signature grows a 4th param only for
   spectrum). Order: frame/ticks → forcing markers → **pinned curves (aged alphas,
   kunit-rescaled x)** → fit line → live curves → legend. Range logic per the decisions
   above: pinned perp series join the hi/lo pool and the `specFloor` walk; the
   "waiting…" early-out becomes "no live data AND no pins".
4. **Legend** per the decision above (pick full-entry vs collapsed after seeing it at
   SW×SH on the narrowest card layout).

GATE B: `node --check`; node unit checks: snapshot deep-copy independence (mutating the
live bins after pinning must not move the pin), 4-pin cap, kunit rescale (pin at
kunit=1, draw at kunit=0.5 → x doubles; identity when equal), range union (pinned curve
a decade below live pulls ymin down; pinned-par does not), waiting-early-out truth
table; both self-tests green; sonnet review of the diff.

## Phase C — lifetime across preset switches + docs

1. **`cardsLayout` transplant**: before destroying the outgoing cards, collect
   `{ pins, lastData }` from outgoing spectrum cards in DOM order; after building the
   incoming layout, reassign them to incoming spectrum cards in order (extras dropped,
   shortfall fine). One loop, no identity matching. This is what makes the universality
   lesson work: pin the decayed spectrum, pick the forced preset, ghosts persist.
2. **Hint text**: the spectrum card hint gains one clause ("pin freezes the current
   curves as ghosts for before/after comparison; ghosts keep their physical k if the box
   changes").
3. **README.md**: short section under the existing chart-card docs. SPEC.md/SPEC3D.md:
   no change (display feature, outside the physics contract).

GATE C: `node --check`; manual pass of the four motivating workflows end-to-end in the
2D app (zeroth-law diss halving, forcing eps change, decay→forced preset switch with
transplant, inverse-cascade watch) and one 3D check (pin with `sd = both`, confirm
dashed pinned-par ghosts and no range stretch); both self-tests green; **adversarial
review by a separate fresh Fable over the whole A+B+C diff** (orchestration default:
sonnet/opus implement, Fable oversees, fresh Fable reviews).

## Cross-cutting rules

- No WGSL, no new buffers, no readback or kernel changes anywhere in this plan. If a
  phase finds itself wanting one, it has misread the plan — stop and re-check.
- `common.js` is the only substantive file touched. **Coordination**: a KH-demo agent is
  working concurrently (2026-08-09) and may touch `common.js` (presets section, ~line
  2170) and `rmhd2d.html`. Do not start Phase A until that work is committed; implement
  on top of its commit. The chart-card region (~1020–1720) and the presets region are
  disjoint, so a rebase should be clean — verify, don't assume.
- No RNG-adjacent changes (rule noted for completeness; nothing here steps the solver).
- All the usual constraints: file:// operation, no repo copies in the sandbox,
  `node --check` everything, self-tests are sacred, giant inlined JSON lines never
  touched.
