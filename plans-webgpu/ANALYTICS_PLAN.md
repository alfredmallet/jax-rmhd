# Analytics + contact plan: GoatCounter pageviews and a JS-assembled contact line

One small feature, two phases, executed in order, each gated, then one adversarial review
by a fresh reviewer. Base: current `webgpu/` at HEAD (`common.js` chrome system,
`rmhd2d.html`, `rmhd3d.html`, `docs.html`, `reading.html`, `index.html`,
`.github/workflows/pages.yml`).

Guiding principle: **this touches chrome only**. No WGSL, no solver, no card, no chart, no
`style.css` rule that any existing element resolves against. Every generated kernel must
come out byte-identical to the pre-phase baseline, and `checkonepage.js` must still pass
at its current leg count. If either fails, the phase is wrong.

## What it is / why

Two unrelated-but-adjacent bits of page furniture, done in one pass because they land in
the same three lines of markup:

1. **Pageview counts.** The site has no server logs (GitHub Pages) and no analytics at
   all, so "does anyone actually load this" is currently unanswerable. GoatCounter:
   cookieless, no personal data, free for non-commercial use, one `<script>` tag.
2. **A contact link.** The acknowledgements line already carries Alfred's name; a visitor
   whose GPU falls over has nowhere to report it. Email + a GitHub issues link, side by
   side, in that same line.

Explicitly **out of scope**: custom events (which preset, whether Run was pressed,
adapter strings, WebGPU-support rate). Discussed 2026-08-11 and declined — pageviews
only. The prefilled diagnostics in the mailto body (Phase 2) are the deliberate
substitute: the same information, volunteered, from the handful of people motivated to
write. If the events question is reopened later, GoatCounter takes them on the free tier
(`goatcounter.count({event: true, ...})`) and the hooks are named in "If events come
back" at the bottom.

## Decisions (fixed here so the phases don't relitigate them)

- **Four pages, not five.** `rmhd2d.html`, `rmhd3d.html`, `docs.html`, `reading.html`
  get the beacon. **`index.html` does NOT.** It is a pure redirect to `rmhd2d.html`
  (meta refresh + `location.replace`), so every visitor arriving at the bare
  `/taranis/webgpu/` URL would be counted twice — and worse, its existing
  `<link rel="canonical" href="rmhd2d.html">` means GoatCounter would file that first hit
  *under `rmhd2d.html`'s path*, silently inflating the one number this whole exercise
  exists to produce. Nothing is lost: the redirect's destination counts the same visitor
  a few hundred ms later.
- **Plain inline `<script>` tags, one per page — NOT injected from `common.js`.** This
  cuts against the house "shared chrome exists once" rule, and does so knowingly:
  `docs.html` and `reading.html` do not load `common.js`, so a `common.js` injector would
  cover two of the four pages and the other two would need the tag anyway. One mechanism
  on four pages beats two mechanisms on four pages. The tag is three lines and identical
  in each; `checkgc.js` (Phase 1) is what keeps the four copies honest.
- **Path is the bare pathname**, via a self-referencing `<link rel="canonical">` in each
  of the four heads. GoatCounter sends `location.pathname + location.search` by default,
  and this site generates query strings in normal use: the "load latest build" link
  appends `?fresh=<timestamp>` (a *unique* value per click) and `?demo=` deep-links carry
  a preset name. Without the canonical, one page's traffic scatters across an unbounded
  set of near-identical paths and the dashboard is useless. Cost: `?demo=` links stop
  being separable — accepted, that is an event question, and events are out of scope.
- **Site code lives literally in the markup.** It is public in the page source by
  construction; there is nothing to protect and no reason to thread it through Actions.
  The plan originally called for landing a placeholder that `checkgc.js` would fail on,
  because registering the site is the one step no agent can do. **Superseded 2026-08-11:
  Alfred registered `taranis` before the phases ran and supplied the snippet**, so the
  four files carry the real endpoint `https://taranis.goatcounter.com/count` and
  `checkgc.js` pins that exact string instead. The placeholder leg survives as a
  regression guard: if anyone reintroduces `ALFRED_GOATCOUNTER_CODE`, the gate fails.
  A reviewer coming to this cold cannot tell a registered code from an invented one —
  the provenance is here, in this bullet, on purpose.
- **The address is assembled in JS**, never present as a contiguous string in the served
  HTML: `mailto:` on a page carrying a real name and a publication list gets harvested.
  Assembly is a `String.fromCharCode`-free, plain concatenation of parts held in
  separate variables plus the `@` from a char code — enough to defeat the regex scrapers,
  not a serious obfuscation, and not pretending to be one. Consequence: **the contact
  link does not exist with scripting off.** Accepted — the pages it appears on need
  WebGPU, which needs scripting.
- **Address: `alfred.mallet [at] berkeley.edu`** (institutional, per 2026-08-11; taken
  from the repo's own commit authorship). Not the personal gmail. Written de-literalled
  **here too**, and that is not fussiness: `pages.yml` stages the site with
  `cp -r webgpu/. _site/webgpu/`, so every `.md` in this directory — including this plan
  — is published next to the pages whose entire runtime-assembly dance exists to keep the
  address out of the served bytes. `checkgc.js` walks every file the deploy copies, not
  just the HTML and JS.
- **Contact goes on the two app pages only**, in the existing `#acknowledgements` div.
  `docs.html` and `reading.html` keep their current footers. They get the beacon, not the
  link — a reader of the reading list has nothing to report.
- **No cookie banner, and no claim of one being unnecessary in the UI.** GoatCounter sets
  no cookies and stores no personal data; the page says nothing about it. If a privacy
  line is ever wanted it belongs in `docs.html`, not the app chrome.

## Phase 1 — the beacon

Files: `rmhd2d.html`, `rmhd3d.html`, `docs.html`, `reading.html`, plus a new
`devtools/checkgc.js`. `index.html` is NOT touched.

1. In each of the four `<head>`s, directly after the existing
   `<link rel="stylesheet" href="style.css">`:

   ```html
   <link rel="canonical" href="rmhd2d.html">   <!-- the page's own filename -->
   <!-- Pageview counts only: no cookies, no personal data, no custom events (see
        ANALYTICS_PLAN.md). The canonical above is what keeps ?fresh= and ?demo= from
        scattering one page's traffic across unbounded distinct paths. -->
   <script data-goatcounter="https://taranis.goatcounter.com/count"
           async src="//gc.zgo.at/count.js"></script>
   ```

   Each page's canonical `href` is **its own filename**, not `rmhd2d.html` — the snippet
   above shows the 2D case. `index.html`'s existing canonical stays exactly as it is.

2. The tag is `async` and third-party: it must not be able to affect boot. Confirm by
   inspection that nothing in `common.js` or either app's inline script reads
   `window.goatcounter`, and that the tag sits after the stylesheet and before the
   `common.js`/`physics.js` script tags at the bottom of body (i.e. it cannot reorder
   them).

3. `devtools/checkgc.js` — a text-level gate over the five HTML files, no DOM boot
   needed. Legs:
   - the four counted pages each contain exactly one `data-goatcounter=` attribute;
   - `index.html` contains none (the double-count trap, asserted so a later well-meaning
     "you missed one" edit fails loudly instead of quietly inflating);
   - the four endpoint URLs are byte-identical to each other;
   - the placeholder `ALFRED_GOATCOUNTER_CODE` is absent (a regression guard now that the
     real code is registered, not a live gate);
   - each of the four has a self-referencing canonical whose `href` equals its own
     filename;
   - `index.html`'s canonical still points at `rmhd2d.html`;
   - **no file the deploy copies contains the address contiguously** — the whole of
     `webgpu/`, `.md` files included, because `pages.yml` publishes all of it.

   Wire it into `devtools/README.md` the way `checkonepage.js` and `check2dspec.js` are.

**Gate:** `checkgc.js` passes clean; `checkonepage.js`
still passes at its current leg count; `dumpwgsl2.js` output byte-identical to a baseline
captured from clean HEAD before the phase.

## Phase 2 — the contact line

Files: `rmhd2d.html`, `rmhd3d.html` (markup hook), `common.js` (the builder),
`devtools/checkgc.js` (extra legs).

1. In both app pages, inside the existing `#acknowledgements` div, after the Claude link,
   add an empty hook: `&middot; <span id="contact"></span>`. That is the whole markup
   change, and it is identical in both files.

2. In `common.js`, a new `contactBuild()` called from `chromeBuild()` (verified: both apps
   call `chromeBuild` before `initGPU` — `rmhd2d.html:1348` vs `1358`, `rmhd3d.html:2883`
   vs `2897` — so the link is present *on the no-WebGPU fallback page too*, which is
   precisely the visitor most likely to need it).
   It fills `#contact` with two anchors:
   - **"email Alfred"** — `href` built at runtime. Local part and domain held as separate
     string literals, `@` from a char code, joined only inside the function. The `href`
     carries a prefilled `subject` (`plasma turbulence in your browser — <page>`) and a
     `body` seeded with, each on its own line and each individually guarded with
     try/catch or `?.` so a missing API cannot throw during chrome build:
     `location.href`, the build id (the `.buildid` span's text), `navigator.userAgent`,
     `navigator.gpu ? "webgpu: yes" : "webgpu: NO"`, and — only if a device was already
     obtained by the time the link is clicked — an adapter description. That last line
     needs a **two-line addition to `initGPUTry`** (`common.js`, the `adapter` local at
     ~line 494 is currently discarded): stash a module-level `gpuInfo` string from
     `adapter.info` when it exists, `""` otherwise, inside a try/catch. `adapter.info` is
     a recent-Chrome property and the older `requestAdapterInfo()` is async and now
     removed — do **not** call either unguarded, and do not make the contact line's
     correctness depend on the string being non-empty. This is the only edit either phase
     makes outside chrome; it adds no await and cannot change a boot outcome. The body is
     assembled **at click time**, not at build time, so the adapter line is populated on
     a page that booted fine and honestly absent on one that did not. Everything is
     `encodeURIComponent`-ed; keep the total under ~1500 chars so no mail client trims it.
   - **"report a bug"** — a plain `<a>` to
     `https://github.com/alfredmallet/taranis/issues/new`, `target="_blank"`,
     `rel="noopener"`, matching every other external link in the file.

   Wording is chrome copy and therefore **Alfred's to write** — the phase ships the two
   labels above as a draft and flags them for replacement, exactly as the preset text and
   the aniso hints were handled.

3. Style: reuse the `.hint` context the div already sits in. **No new `style.css` rule**
   unless the two anchors visibly collide, and if one is genuinely needed, it goes in the
   shared sheet with a comment, not inline.

**Gate:** `checkgc.js` grows legs asserting that neither served HTML file contains the
contiguous address string, that `#contact` exists and is empty in both markup files, and
— booted under `stubenv.js`, both pages, including the `{noGpu: true}` fallback boot —
that `#contact` ends up holding exactly two anchors, that the mailto `href` starts with
`mailto:` and contains the correctly assembled address, and that the body contains the
build id and the userAgent. `checkonepage.js` unchanged and still passing; WGSL
byte-identical.

## Review

One fresh reviewer, no prior context in this plan, adversarial, with the brief: *the
counted-page set is the load-bearing decision here — verify the `index.html` exclusion is
right and that no page counts a visitor twice; then verify the mailto cannot throw during
`chromeBuild` on any boot path, including the no-WebGPU one, and that the address is
absent from both served files as a contiguous string.* Send-back on any MAJOR.

## After the phases (Alfred, not an agent)

1. ~~Register the site~~ — DONE 2026-08-11, code `taranis`, already in the four files.
2. Set the dashboard to public or keep it private — either is fine, GoatCounter defaults
   to private.
3. Ignore your own visits. `count.js` already skips `localhost` and `file:` on its own;
   for the deployed page, load it once as
   `…/rmhd2d.html#toggle-goatcounter`, which sets `localStorage.skipgc = "t"` in that
   browser and reports what it did. (The key is `skipgc` and the toggle is a URL **hash**
   — verified against upstream `count.js` 2026-08-11, since the obvious guesses
   `goatcounter-ignore` and a settings checkbox are both wrong.) Repeat per browser and
   per device; it is per-origin localStorage, not an account setting.
4. Replace the two draft link labels with your own wording.
5. On-device: check the link on the phone (does `mailto:` open the right client with the
   body intact?) and confirm a real pageview lands in the dashboard within ~10 s.

Expect undercounting: this page's audience runs adblockers at a high rate, and both
`goatcounter.com` and `gc.zgo.at` are on the common lists. Read the numbers as ratios and
trends, never as a census.

## If events come back

Not now, but so the decision is cheap to revisit: the natural hooks already exist and are
named here so nobody has to re-find them — `gpuFallback()` in `common.js` (the
WebGPU-unsupported rate, the single most valuable number), the `selPreset` change handler,
`btnRun`, and `initGPU`'s three failure paths. Each is a one-line
`window.goatcounter?.count({path: ..., event: true})`. Free tier, no new dependency.
