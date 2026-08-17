# One-page plan: first-visit layout overhaul

Source: non-physicist feedback (a sketch from an outside reader, 2026-08-10). That
layout, top to bottom: title → one line of text → 2D/3D tabs → toolbar with a big RUN and
a hide toggle → the canvas → "what is…" as a side panel. Notes on the sketch: *hide params by default*,
*run/pause is green/red*, *put "what is…" on the side?*

Guiding principle: **a first-time visitor sees something pretty within one screen and one
click (ideally zero), without losing anything a returning power user has today.** No
physics, no WGSL, no SPEC changes; this is chrome only (`common.js` UI regions,
`style.css`, the three html files).

## Coordination — read before starting

1. ~~Recorder rebuild in flight~~ **CLEARED 2026-08-10**: recorder work landed and is
   committed (`5366639` mp4/WebCodecs). Base commit for this plan = `5366639`.
2. ~~Uncommitted 2026-08-10 batch~~ **CLEARED**: batch committed (`e494cd4`, `2e3eb54`).
3. Per repo default: opus/sonnet implement, Fable oversees, adversarial review by a
   separate fresh Fable at the end.

## What it is / why

The pages were built and reviewed exclusively under expert eyes; the wall of controls
above the fold and the essay-only landing page are invisible costs to us and the first
thing a lay visitor hits. The sketch is standard demo UX: hero canvas, one obvious
action, explanation off to the side. Almost all of it is cheap because the pieces
already exist — `btnParams` already toggles `#controls`, the topbar is already sticky,
the "what is…" panes already exist as `<details>` on index.html.

## Design decisions (fixed here so the phases don't relitigate them)

- **Tabs, not a merge.** 2D and 3D stay separate documents (separate GPU pipelines,
  1.4k/2.3k lines; merging buys nothing a user can see). A shared tab strip renders
  "2D | 3D" with identical chrome on both pages; the inactive tab is a plain link.
  Visitors cannot tell a styled link from a tab.
- **rmhd2d.html is the front door.** index.html becomes an immediate redirect to
  `rmhd2d.html` (meta refresh + JS fallback, `<link rel="canonical">`). Nothing else is
  renamed: every existing deep link (`?demo=`, docs.html, README anchors) keeps working.
- **The index essay moves into the apps, not the bin.** The five panes (what is
  turbulence / plasma / why care / why simulate / technical details) plus a trimmed
  two-sentence version of the lead paragraphs become a "what is all this?" rail of
  collapsed `<details>`. Content lives ONCE, in `common.js`, as a spec injected on both
  pages — the controlsBuild pattern ("every row the two apps share exists once").
- **Rail placement:** right of the display block at ≥1400px (a third, narrower column
  after `#displaycol` and `#charts`); below everything at narrower widths. All panes
  boot collapsed. The preset hint (`#demohint`) stays where it is — riding the sticky
  topbar (2026-08-10 follow-up); the rail must not touch that mechanism.
- **Params hidden by default.** `#controls` boots `display:none`; `btnParams` boots
  labelled "show params". Pure display toggle as today — hidden controls keep state,
  presets still write them, guided-preset text is unaffected (it rides the topbar).
  Returning-user affordance: remember the toggle in `localStorage` so power users pay
  the extra click once, ever.
- **Run is the hero button.** First item in the topbar, larger (own `#btnRun` style
  block: bigger font/padding), green when it says "Run", red when it says "Pause".
  **NOT lurid** (Alfred): desaturated, dark-palette tones — think green ≈ `#2f6b4f`,
  red ≈ `#7a3d3d` backgrounds with the existing `#d8dee6` text, never pure `#0f0`/
  `#f00`. Two constants in `style.css` so retuning after on-device is a two-line edit.
  No other topbar button grows.
- **~~Autoplay on a plain visit~~ REVERSED by Alfred 2026-08-10 (after build):** every
  visit boots PAUSED on the default forced-turbulence preset at **256²** (2D; 3D keeps
  its current default size); the big green Run is the call to action. The autorun seam
  (bootApply's first call) is documented in a comment there if it ever comes back.
  Trim the h1 + `.sub` to one line each so title → tabs → topbar → canvas fits one
  laptop screen (unchanged).
- **index.html redirects immediately** (ratified; no splash).
- **No-WebGPU fallback is IN scope** (consequence of the redirect: today a failed
  `initGPU` leaves dead black canvases plus one `#status` line, acceptable behind an
  essay front door, not as the first impression). Minimal version: on `initGPU()`
  returning false, show the checked-in PNG of a real 512² run (**`webgpu/poster.png`**,
  512², |u| afmhot + arrows + colorbar, supplied by Alfred 2026-08-10) in the display
  area, keep the existing browser-advice message under it, and boot the what-is rail
  expanded. No capability sniffing beyond what initGPU
  already does; the three failure paths (no `navigator.gpu`, no adapter, device lost)
  all already land in `showStatus(..., "err")` — hook there.
- **Out of scope:** any preset/lesson text changes; docs.html restructure beyond
  fixing links.

## Phases (agent-executed, in order, each gated)

### Phase A — cheap wins (common.js + style.css only)

Params hidden by default (+ localStorage memory of the toggle), Run button styling and
green/red state, topbar ordering. **Gate:** both pages; toggle state survives preset
switches and Reset; guided presets (`?demo=`) render their text with params hidden;
recorder buttons (fresh from the other instance) unharmed; no layout shift when the
demohint row appears.

### Phase B — front door (all three html files + common.js)

Shared tab strip; one-line h1/.sub on both app pages; "what is…" pane spec in common.js
injected on both pages (rail markup + responsive CSS, panes collapsed); index.html →
redirect; no-WebGPU poster fallback (decision above); sweep README/docs.html for links
that said "index.html is the overview".
**Gate:** every `?demo=` deep link unchanged; docs.html reachable from both pages;
GitHub Pages entry lands on a working 2D page; rail collapses below the display block
at phone widths; no duplicated pane content in any html file; poster + expanded rail
render when `navigator.gpu` is stubbed away (bootstub leg).

### Phase C — autoplay (common.js boot path)

Plain-visit autorun at the reduced default resolution; `?demo=` boots paused exactly as
today; steps/s readout confirms it is actually running. **Gate:** no autorun under any
`?demo=`; IC editor, self-test, and record flows all still pause/restore correctly
around the running boot (grep the `wasRunning` sites); energy trace starts cleanly from
the autorun (no duplicate-t guard trips).
**BUILT 2026-08-10.** The seam is `bootAutorun()` in `common.js`, fired from `bootApply`'s
FIRST call — the boot one, past a successful `initGPU`, with the solver already built — and
one-shot, so the preset dropdown never touches `running`. No page-level edits: neither
inline boot script changed. Gate legs added to `devtools/checkonepage.js` (64/64).

### Review

Adversarial review by a fresh Fable instance (checklist: the three gates, plus
"first screen, cold cache, 1280×800 — is the canvas visible without scrolling?").
Then Alfred on-device: phone widths, autoplay feel/battery, whether muted green/red
reads at a glance, tab strip on iOS Safari.

**DONE 2026-08-10.** Verdict SEND-BACK (narrowly) → both MAJORs + MINORs fixed by the
overseer same session, re-gated to SHIP quality: (1) root `.gitignore`'s `*.png` was
silently swallowing poster.png — negation rule added (+ `node_modules/`); (2) params-
hidden only applied after `initGPU` resolved — `#controls` now hidden in the MARKUP of
both pages (no flash during adapter wait; no dead panel on the poster page), spec label
fixed, and stubenv taught to reflect markup style attributes. MINORs: docs.html "press
Run" clause updated for autoplay; hero hover/border hexes now color-mix-derived so the
two `--run-*` lines really are the whole retune; gpuFallback disables the dead topbar
buttons; both `<title>`s match the new h1. Recorded NOTEs (defensible, unfixed):
boot-time-lost device gets no poster (pre-existing at HEAD); index.html redirect drops
query strings (no documented `index.html?demo=` links exist); invalid `?demo=typo`
boots paused instead of autoplaying; self-test over an autorun shows Pause while
frozen (pre-existing save/restore mechanics); SPEC.md/SPEC3D.md still say "landing
page" (SPEC edits out of scope). Final gates: checkonepage 69/69, bootstub 8/8,
checks/layout/names/dup green, WGSL byte-identical (120/159126 B, 240/357390 B).
Alfred's on-device list above stands, plus: color-mix rendering of the Run tones,
poster page in a real non-WebGPU browser (Firefox mac/Linux).

## Round 2 (Alfred, 2026-08-10, after the build was committed as 0286eee)

1. **Autoplay REVERSED** — every visit boots paused; the big green Run is the call to
   action (see the struck-through decision above; seam documented at `bootApply`).
2. **Original lead text restored VERBATIM** — Phase B's condensed rewrite of the index
   essay's two lead paragraphs was worse than Alfred's original; all five panes had
   gone over verbatim and stay. One deliberate exception kept: the technical-details
   pane says "the preset dropdown in the top bar" instead of the original "each app has
   a preset dropdown in its top bar" (the text now lives ON the app page).
3. **Intro moved under the subtitle** — the two lead paragraphs are now a
   "what is all this?" `<details>` (`#intro`, built by `chromeBuild`) directly under
   the h1/.sub, ABOVE the tabs: open until the visitor closes it once (localStorage
   `taranisIntro`, same pattern as the params toggle), forced open on the no-WebGPU
   page. The rail keeps just the five background panes. Chosen over Alfred's
   floated "intro preset" idea (two paragraphs riding the STICKY topbar would stay
   glued to the top of a phone screen) and over an always-visible block (permanent
   ~150 px canvas push).
4. **rmhdvars hint** now opens its second sentence with "Press show params above," —
   the walkthrough needs `edit IC`, which lives in the hidden panel.

Gates after round 2: checkonepage 74/74 (paused-boot + intro legs), bootstub 6 legs
re-run green, checks/layout/names green, WGSL byte-identical.

## Decisions ratified (Alfred, 2026-08-10)

1. ~~Autoplay, not pulsing-Run.~~ REVERSED after build (2026-08-10, on-device): boots
   paused; `bootAutorun` removed, gates flipped to assert the paused boot.
2. Plain-visit default: forced-turbulence preset at 256².
3. index.html: immediate redirect, no splash.
4. No-WebGPU poster fallback: in scope, Phase B (see decision above). Alfred supplies
   the poster PNG.
5. Green/red must be muted, not lurid — hexes in the Run decision are the target
   register; final tuning on-device.
