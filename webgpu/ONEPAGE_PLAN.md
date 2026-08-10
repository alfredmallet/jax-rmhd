# One-page plan: first-visit layout overhaul

Source: non-physicist feedback (Alfred's wife, sketch, 2026-08-10). Her layout, top to
bottom: title → one line of text → 2D/3D tabs → toolbar with a big RUN and a hide toggle
→ the canvas → "what is…" as a side panel. Notes on the sketch: *hide params by default*,
*run/pause is green/red*, *put "what is…" on the side?*

Guiding principle: **a first-time visitor sees something pretty within one screen and one
click (ideally zero), without losing anything a returning power user has today.** No
physics, no WGSL, no SPEC changes; this is chrome only (`common.js` UI regions,
`style.css`, the three html files).

## Coordination — read before starting

1. **Another instance is rebuilding the video recorder right now** (FEEDBACK_2026-08-10
   item 13; recorder logic ~common.js 2000–2100, plus possible `CTRL_TOPBAR` /
   display-card header edits). Do NOT start until that lands. Phases A and B edit
   `CTRL_TOPBAR`, `wireCommonControls`, and `style.css`'s topbar block — guaranteed
   conflict zones. Rebase this plan's base commit on the recorder work.
2. The working tree carries the (large) uncommitted 2026-08-10 batch. This plan builds
   on top of it *after* it is committed, not alongside.
3. Per repo default: opus/sonnet implement, Fable oversees, adversarial review by a
   separate fresh Fable at the end.

## What it is / why

The pages were built and reviewed exclusively under expert eyes; the wall of controls
above the fold and the essay-only landing page are invisible costs to us and the first
thing a lay visitor hits. Her sketch is standard demo UX: hero canvas, one obvious
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
  Muted green/red consistent with the dark palette — not stoplight saturation.
  No other topbar button grows.
- **Autoplay on a plain visit.** A visit with no `?demo=` boots the default forced-
  turbulence preset at a modest resolution (256² in 2D; 3D keeps its current default
  size) and starts running (`running = true`, button shows red "Pause"). Any `?demo=`
  visit keeps today's paused boot — lessons want the visitor to read first. Trim the
  h1 + `.sub` to one line each so title → tabs → topbar → canvas fits one laptop screen.
- **Out of scope:** static poster fallback for non-WebGPU browsers (nice, separate);
  any preset/lesson text changes; docs.html restructure beyond fixing links.

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
redirect; sweep README/docs.html for links that said "index.html is the overview".
**Gate:** every `?demo=` deep link unchanged; docs.html reachable from both pages;
GitHub Pages entry lands on a working 2D page; rail collapses below the display block
at phone widths; no duplicated pane content in any html file.

### Phase C — autoplay (common.js boot path)

Plain-visit autorun at the reduced default resolution; `?demo=` boots paused exactly as
today; steps/s readout confirms it is actually running. **Gate:** no autorun under any
`?demo=`; IC editor, self-test, and record flows all still pause/restore correctly
around the running boot (grep the `wasRunning` sites); energy trace starts cleanly from
the autorun (no duplicate-t guard trips).

### Review

Adversarial review by a fresh Fable instance (checklist: the three gates, plus
"first screen, cold cache, 1280×800 — is the canvas visible without scrolling?").
Then Alfred on-device: phone widths, autoplay feel/battery, whether muted green/red
reads at a glance, tab strip on iOS Safari.

## Open questions (Alfred decides, not the builders)

1. Autoplay vs. load-paused-with-pulsing-green-Run. Plan says autoplay; flip to the
   pulse if the battery/CPU cost on lurkers feels rude on-device.
2. Which preset is the plain-visit default, and at what resolution (256² assumed).
3. Does index.html redirect, or keep a minimal splash (title + two cards + redirect
   after N seconds)? Plan says immediate redirect.
4. Non-WebGPU poster fallback — worth a follow-up item?
