# IO_PLAN — data in and out: expressions and files in, PNGs, ZIPs, `.npz` and video out

Written 2026-08-20, **not started**. From a collaborator's reply to the round-2 outreach
email: "add an option for saving simulation data" and "allow users to define their own
initial conditions", plus one of Alfred's own (item 5). Five items, one theme — the demo
currently has exactly two exits (a PNG of one display card, an MP4 of one field canvas) and
no numeric one at all, and its only IC authoring surface is the mouse.

**Decisions already taken (Alfred, 2026-08-20), not to be re-litigated:**

- the periodicity check runs **per axis** — x and y, and z in 3D — and names the seam it
  fails on;
- expression variables `x, y, z` are in **code units** — the same coordinates `setIC` uses,
  x = ix·Lx/nx, so the box ends are Lx, Ly, Lz and nothing is normalized to 2π or to 1;
- a non-periodic expression is **warned about, never forbidden**, and the warning says what
  will happen (ringing) rather than just that something is wrong;
- save-all produces a **ZIP**, not a burst of separate downloads;
- field export is **real-space φ and ψ** (not the k-space state), packaged as **.npz**.

**Not an item, but the reason item 1 was asked for.** The drawing editor already does
"define your own initial conditions" — `custom` + **edit IC**, and the `rmhdvars` preset
opens on an empty canvas precisely to introduce it — and a capable user explored the
presets without finding it. Expressions are worth having on their own merits, but they do
not fix that; the discoverability of the **edit IC** button is a separate open UI question
and belongs in whatever the next feedback round is.

## 1. Expression ICs

### Where it attaches

`icPresetFields(q, preset, ampP, ampM, env)` (`common.js`) is the one place every
*potential-based* UI IC preset is built — `quiescent` and `modes` bypass it for
`setIC(true)` / `setIC()` and stay untouched by any of this — and `IC_BUILDERS` is the
registry for presets that build (φ, ψ)
**directly** — the equilibria live there because their knobs are physical and an amplitude
renormalization would destroy the equilibrium they name. An expression is the same kind of
object: `φ = sin(x)cos(y)` has to mean what it says. So expressions register through
`icRegister` like an equilibrium does, declare their own `rows` (the two text boxes), and
**skip `icZetaFields` normalization**, which makes the φ amp / ψ amp sliders meaningless for
them and so requires hiding those rows.

Two things the registry does **not** do for us, both easy to assume it does:

- **The amp rows are not hidden by a builder's `rows` list.** `icSyncRows` hides
  `rowAmpP` / `rowAmpM` on its own predicate — `icIsPacketIC(p) || p === "custom"`
  (`common.js` ~7067) — which is separate from the `IC_BUILDERS[k].rows` loop below it. The
  expression preset must be added to that predicate; declaring `rows: []` hides nothing.
- **There is no text-input control kind.** `_ctrlItem` (`common.js` ~4854) knows
  `sel rng num cb cbl btn lab val hint` and a bare-span fallback. A `k: "text"` item is a
  prerequisite of this item, and it is the one place item 1 touches shared UI code.

The builder is `(g) => {phi, psi}` on the geometry record `icDrawGrid(q)` returns
(`{nx, ny, nz, Lx, Ly, Lz}`), and the result goes to `setICFromReal`, whose layout is
`ix*ny + iy` in 2D and `(iz*nx + ix)*ny + iy` in 3D. Either field may be `null` = exactly
zero, so an empty box is not a special case to invent — it already means that.

Nothing in the solver changes. `_uploadIC` forward-transforms and dealiases exactly as it
does for a drawing, which is also why the ringing story below is honest: what runs is the
band-limited periodic projection of what was typed, never the expression itself.

### The parser

Do **not** use `eval` or `new Function`. Not mainly for security (the page has no secrets
to steal), but because a thrown `SyntaxError` from the engine is useless to a student and
because the accepted language would then be all of JavaScript, which we would have to
document. A hand-written parser is ~150 lines and gives error messages with a character
position.

Three stages, all pure and all node-testable:

1. **Tokenizer** — numbers (`1`, `.5`, `2e-3`), identifiers, operators `+ - * / ^`,
   parentheses, comma. Every token carries its source index, which is what the error
   messages point at.
2. **Shunting-yard to RPN** — precedence `+ -` < `* /` < unary minus < `^`, with `^`
   **right-associative** and binding tighter than unary minus, so `-x^2` is `-(x^2)` and
   `2^3^2` is `2^9`. Function calls push an arity; a name that is not a known function or
   variable is an error at its own index ("unknown name 'sn' at character 7"), not a
   silent zero.
3. **Evaluator** — a flat loop over the RPN array with a small numeric stack, run once per
   grid point. No allocation inside the loop.

Names: variables `x, y` (and `z` in the 3D app only — in 2D `z` is an unknown name, which
is the correct error rather than a silent zero); constants `pi`, `e`, and the box lengths
`Lx`, `Ly` (and `Lz` in 3D only — `icDrawGrid` returns `q.Lz || 0`, so a 2D page must reject
the name rather than hand out a zero that turns `x/Lz` into the non-finite refusal below and
a confusing message with it), so `sin(2*pi*x/Lx)` is writable without hardcoding the box; functions
`sin cos tan asin acos atan atan2 sinh cosh tanh exp log log10 sqrt abs sign min max mod
floor ceil hypot pow`. `log` is natural log — say so in the placeholder text, because half
the audience will assume base 10. No RNG: a seeded noise IC is a reasonable future ask and
a bad thing to smuggle in through an expression, because the same string would then not
reproduce the same run.

Cost: 2D at its 1024² maximum is 1.0e6 points × ~20 stack ops, tens of ms. 3D at its largest
offered grid — 256²×64; the 3D resolutions are 64²×32, 128²×32, 128²×64, 256²×64, 64²×128,
64²×256 (`rmhd3d.html` ~96), and there is **no** 128³ — is 4.2e6 points, ~200 ms — one
button press, not a frame, so no chunking. Evaluate in the layout's
own index order so the write is sequential.

**Non-finite guard, before anything is uploaded.** `1/x` at x = 0 and `log(y)` at y = 0 are
the first two things anyone types. Scan the result; if any entry is not finite, refuse the
upload and report the count and the first offending (x, y[, z]). A NaN reaching
`_uploadIC` propagates through the forward FFT into every mode and the run is silently dead
from step 0 — an unrecoverable state that looks like a solver bug.

### The periodicity warning

The box is periodic; an expression is not required to be. Detect and explain, do not block.

**Check every axis, and say which one failed.** The box is periodic in x and y (and z in
3D), and an expression can easily be periodic in one and not another — `sin(x)*y` is fine in
x and discontinuous in y, and a warning that only says "not periodic" sends the user hunting
in the wrong place. So the test runs **per axis**, and each field is tested separately: two
seams in 2D × two fields, three in 3D.

The grid never samples the far edge (x = ix·Lx/nx stops one cell short), so the far face has
to be evaluated explicitly rather than read off the array. For the x seam, evaluate at
x = Lx over the whole opposite face and compare to x = 0; likewise y at y = Ly, and z at
z = Lz. Per seam:

- value jump `J0 = max|f(Lx, y, z) − f(0, y, z)|` over that face, reported relative to the
  field's own range `max f − min f`;
- gradient jump, one-sided differences either side of the seam, same normalization — a field
  that is continuous but kinked (`abs(x − Lx/2)`) rings much less than one that is not, and
  the two cases deserve different-sounding warnings.

Threshold for showing anything: 1e-3 of the range. Below that it is a rounding artifact.
Report the worst seam by name (`φ: y seam, jump 0.42 of range`), not a single verdict for
the whole expression.

The warning is one line under the box, in Alfred's register — state the number, say the
field that will actually run is the periodic band-limited projection of what was typed, and
say what that looks like: oscillations near the seam that do not decay, and a spectrum with
a broadband tail that is not turbulence. Then let the user press Run. Half the value of the
feature is that a student can type `x*y`, see the ringing, and learn why spectral codes want
periodic data.

## 2. Save button on chart cards

`ChartCard` (`common.js` ~4485) has a close button, a type `<select>`, a hint line and its
option row — and no capture button at all; `DisplayCard` has **save** and **rec**. Add
**save** to `ChartCard`, mirroring the display
card's button placement and title.

This is the *easy* half of the pair: a chart draws into its own persistent 2D canvas
(`this.cv`, class `chart`, sized by `chartCtx` at `dpr` capped to 2 with the transform set
once, so `toBlob` already yields the physical-pixel image), so it is `this.cv.toBlob(...)`
with **no** re-render — `DisplayCard.saveShot` re-renders and composites three layers only
because WebGPU keeps no `preserveDrawingBuffer`, which does not apply to a 2D canvas.

**One chart type still needs a composite.** A type declaring `bar` — today `gen2d`, the
generated 2D spectrum — paints its colour scale into a *second* canvas built by
`ChartCard._barBuild` (`common.js` ~4636), which `this.cv.toBlob()` would not capture, so the
saved heatmap would lose its scale. For those types the save draws plot and bar onto one 2D
canvas: the chart-side analogue of `DisplayCard.barStamp`, and the reason to keep the bar's
geometry in one place rather than re-deriving it in the save path.

Filename through a `chartName(kind, ext)` alongside `shotName(mode, ext)`, same shape:
`taranis-<app>-<chart>-t<simT>.png`.

**Delivery follows the house rule, not `dlBlob`.** Nothing has handed a finished file
straight to the downloader since 2026-08-11: files wait on the card footer behind a line of
text with a download/share button (`DisplayCard.recResult`), because on a phone a silent
download lands somewhere in Files and is then hard to find and harder to send on.

The chart card needs the same footer slot **and does not have one**, so lifting `recResult`
across as-is would break the rule it is meant to honour:

- `ChartCard.foot` exists only for `bar` types, and `_barBuild` removes and recreates it on
  every `build()` — a pending strip would silently vanish on a retype.
- `recResult`'s own guard is `if (this.dead || !this.foot) { dlBlob(blob, name); return; }`
  (`common.js` ~4369), and `ChartCard` never sets `dead` (its `destroy` only removes the
  node). For every non-`bar` type `this.foot` is `undefined`, so the lifted strip would fall
  straight through to exactly the silent download this rule exists to prevent.

So the shared strip wants an always-present footer host on both card classes and a `dead`
flag set in `ChartCard.destroy`, and only then is it worth lifting out of `DisplayCard`
rather than copy-pasted (the standing no-copy-paste rule from REFINE_PLAN).

## 3. Save all displays and charts, as one ZIP

A button beside `+ display` / `+ chart` — **not** in the topbar: those two are `{k:"btn", …}` rows
in the shared UI group (`common.js` ~5056), so this is one more row and one more handler
beside `btnAddDisp` / `btnAddChart`.

**Why a ZIP and not N downloads.** Firing several `<a download>` clicks in a row raises
Chrome's "site wants to download multiple files" prompt, and on iOS each save is silent with
no share sheet — the exact failure the result strip exists to avoid, multiplied by the
number of cards. One archive is one file, one strip slot, one share.

Contents: every display card's composite PNG (the existing `saveShot` capture, colorbar
stamp and all), every chart card's PNG, and a `params.json` manifest — resolution, box,
sim time, dissipation and hyper exponent, forcing state, IC preset, and the app slug — so a
folder of PNGs is still self-describing a month later. Names inside the archive
`disp1-<field>.png`, `chart2-<kind>.png`, in card order.

**Capture ordering is the one real constraint.** A display card's texture is transient: its
capture must happen in the same task as its re-render. So the handler renders and captures
*every* display card in one synchronous pass, collecting the `toBlob` promises, and only
then awaits them all and builds the archive. Do not `await` between cards.

**`saveShot` cannot be reused as it stands.** It returns nothing and ends by handing its blob
to `this.recResult`, so calling it per card would spawn N result strips beside the one
archive. Split it: a capture that resolves to a blob, and a deliver step the single-card
button keeps using unchanged.

**Recording is not part of save-all.** "Record every card" as a blanket action stays out:
it is N encoders on a device that is already bandwidth-bound (LOOPLAT: ~2 ms/step, GB/s
pinned) and, on an iPhone 11, already dropping about half its recorder slots. What *is*
wanted is item 5 below — the open display cards, at most three of them, in one
frame-synchronous video.

## 4. Download fields — real-space φ, ψ as .npz

### Getting the data

The display chain already produces exactly what is wanted, but not on a spare path.
`prepDisp` → `colsInv` → `rowsC2R` lands a real-space field in `dispR` (`nr*4` bytes) in 2D;
3D has one stage more, `prepDisp` → `zInv` → `colsInv` → `rowsC2R` (`rmhd3d.html` ~1885).

What it does **not** already have is a spare pin. The chain that reaches `dispR` is driven by
`B.mode` (a display card's live mode, carrying its band filter and offset) or `B.modeM` (the
sigma mate), both written only by `setDisplayMode`; the contour path's pinned modes feed
`dispK2` → `dispR2`/`dispR3` (2D) or `sliceB`/`sliceC` (3D), never `dispR`. Overwriting
either live uniform would corrupt that card's display until its next `apply()`. So export
needs a **third Mode uniform and one bind group per stage of the chain**, pinned to plain φ
and plain ψ with no band, no offset and no scale filter, and it must run at a frame boundary
rather than inside a render. No new pipeline, kernel or WGSL — but it is not allocation-free,
and the plan should not be read as saying it is. `readBuf(device, buf, byteLen)` already
exists, `SQ` includes
`COPY_SRC` unconditionally on every buffer, and no new pipeline or kernel is needed.

The alternative — read `buf.fields` (2·cx bytes, φ̂ and ψ̂ in k-space, the whole state in one
buffer) and inverse-transform on the CPU — is rejected: it duplicates an FFT that already
exists on the GPU, and k-space is not what was asked for.

**Staging-pool caveat, and it matters in 3D.** `_stagePool` is keyed by byte length and has
**no eviction**, deliberately, because every pooled size today is kilobytes. A 2D 512²
readback is 1 MB per field; a 3D readback is 4 MB per field at 128²×64 and **16 MB** at the
largest offered grid, 256²×64. Do
**not** pool those: allocate the staging buffer for a field export and `destroy()` it when
the read resolves, or a single 3D export strands tens of MB for the life of the page. Add
the rule as a comment at the pool, since the pool's own comment currently promises the sizes
are small.

### The file

`.npz` is a ZIP of `.npy` members — the same stored-ZIP writer as item 3, which is the
reason to build that writer once and well. Members:

- `phi.npy`, `psi.npy` — dtype `<f4`, C-order, **shape `(nx, ny)` in 2D and `(nz, nx, ny)`
  in 3D**, which is the buffer layout read literally (`ix*ny+iy`; `(iz*nx+ix)*ny+iy`). Say
  the axis order in the docs and in the manifest; a silently transposed field is the classic
  way to waste someone's afternoon.
- `x.npy`, `y.npy` (and `z.npy`) — the coordinate vectors, `ix*Lx/nx`. A few kB, and it
  makes `plt.pcolormesh(x, y, phi.T)` work without the reader reconstructing the grid.
- `params.json` — the same manifest as item 3, plus sim time and an explicit note that the
  state is dealiased (the 2/3 mask is applied at IC upload and in the RHS, so the saved
  field has no content above the cutoff and its spectrum should not be read as if it did).

`.npy` v1.0 is a 10-byte magic-and-version prefix, an ASCII dict header padded so the data
starts 64-byte aligned, then raw little-endian data — ~30 lines to write, no dependency.
`numpy.load` reads a **stored** (uncompressed) archive fine, so no deflate implementation is
needed anywhere in this plan.

Sizes to expect: 2D 512² both fields ≈ 2 MB, 1024² (the 2D maximum) ≈ 8 MB; 3D at 256²×64,
the largest offered grid, ≈ **32 MB** for both fields — that is the number the phone
checklist and any memory argument should be sized against. Single time
slice only — a time series is a different feature with different design questions and is
explicitly out of scope.

**Not in scope, but designed for:** the download layout is byte-for-byte what
`setICFromReal` consumes, so "load fields" (an `.npz` back in as an IC) stays a small
follow-up rather than a rewrite. Do not add it here.

## 5. Multi-display recording — every open display, one video

### Why this and not "record everything"

The case that motivates it (Alfred, 2026-08-20): **z+ and z- side by side in an imbalanced
run**, where the anomalous coherence between the two is obvious to the eye and hard to convey
any other way. That reading depends on the two panels being the *same frames* — two separate
files that have to be lined up in an editor lose exactly the thing the recording is for. So
the feature is not N recorders; it is **one recorder with two sources and one output file**.

The ingredients for the run already exist: modes 6 and 7 are `zplus` / `zminus`
(`DISP_SLUG`), and forcing imbalance is the `eps+` / `eps-` pair with `cbEpsLock` unlocked.
A preset that opens on that configuration (forced, unlocked eps, two displays) would make the
feature self-demonstrating — worth considering, Alfred's call, not assumed here.

### Why it is affordable now, and what it actually costs

It would not have been before RECASYNC. The old capture was `new VideoFrame(canvas)` at
**15–17 ms of main thread per slot** — a second source would have been unshippable. The
shipped path is `copyTextureToBuffer` (microseconds on the main thread) → `mapAsync` →
`VideoFrame(BufferSource)` (~1 ms) → encode (~1 ms). A second source therefore costs about
**2 ms of main thread per slot** at `REC_FPS` 30, ~6% of a 33 ms budget.

The real cost is GPU-side: one more texture copy per slot and a second staging pool, on
devices that are already saturated. So the honest claim is that pairing costs **simulation
steps per second**, not display smoothness, and the on-device check below must report the
steps/s delta rather than assert there isn't one.

### Design

**It records every open display card — there is no selection UI, and none is needed.**
`CARD_MAX_DISP = 3` (`common.js` ~2745), so "all of them" is at most three tiles; the user
composes the recording by opening the displays they want, exactly as they already compose
what they are looking at. This also matches the loop that exists: `renderCards` already
walks `cards.disp` calling `d.recCapture()` on every card every frame, so a multi-source
recorder is that loop with one slot clock over it rather than a new traversal.

- **One recorder, N sources, one slot clock.**
  One `W.n`, one timestamp ladder, one `VideoEncoder`, one mp4. Two independent recorders
  would drift, and a side-by-side that drops a frame on one side only is worse than no
  recording at all.
- **All-or-nothing slots.** If any source cannot be captured this slot (no free staging
  buffer, a rejected map), the *whole* slot is dropped and `W.n` does not advance — the
  RECRAF/RECASYNC honesty rule verbatim: fewer frames, never a lying sample table.
- **Composite at encode time, not capture time.** Each source's mapped bytes are copied
  row-wise into one combined frame buffer at its tile offset — N memcpy loops, no 2D canvas
  on the WebCodecs leg — and one `VideoFrame(BufferSource)` is built from the result. The
  existing ordered encode chain is what guarantees arrival order = encode order; nothing
  about it changes except that one entry now carries N byte blocks.
- **Layout: one row or one column, whichever comes out closer to square.** With N ≤ 3 that
  is the whole rule and there are never ragged or empty tiles. All cards show the same run,
  so their `gw`/`gh` match by construction — assert it rather than scaling.
- **Probe the composite size before starting.** Three 1024² displays in a row is 3072×1024,
  which is past what some mobile encoders will accept. Ask `VideoEncoder.isConfigSupported`
  for the composite config; if it is refused, halve the tile size and ask again, and say in
  the result strip that the tiles were downscaled. Never fail silently and never truncate the
  card list to fit — the no-silent-caps rule.
- **Labels.** Each tile needs its field name or the picture is ambiguous. On the bytes path
  there is no 2D context, so render each label **once** at recording start into a small RGBA
  patch and blit it into the tile per frame; do not add a per-frame canvas round trip to get
  text. (Compositing through a 2D canvas instead is simpler and costs an upload per frame —
  the fallback leg has to do it anyway, see below.)
- **Leg 2 (MediaRecorder).** It records a *canvas stream* — `this.card.cv.captureStream(REC_FPS)`
  (`common.js` ~3450) — so the fallback cannot cover several WebGPU canvases without
  compositing into a real 2D canvas and calling `captureStream()` on that. Either do that,
  or — preferred, and consistent with the standing degrade-silently rule — offer the
  all-displays recording only where leg 1 runs, and leave single-card recording as it is
  everywhere else.
- **UI.** One action in the displays & charts group, live only when there are ≥ 2 display
  cards (with one card it would duplicate the card's own button). The per-card `rec` button
  keeps today's behaviour untouched, and the action's title should say what it will produce:
  N tiles at the composite size.
- **Charts stay out of v1.** Their canvases are 2D and trivially capturable, so a field with
  its live spectrum beside it is a natural later ask, but it makes the tiling ragged (chart
  aspect ≠ display aspect) and is not what this feature is for.
- `REC_FPS` (30), `REC_MAX_MS` (30 s) and `REC_QMAX` (8) are unchanged; the file is roughly
  N times the bytes of a single-card one.

## Shared: the stored-ZIP writer

One function, used by items 3 and 4: local file headers, central directory, end-of-central-
directory, method 0 (stored), CRC-32 per member. No compression — PNGs and float arrays
compress by a few percent at best, and it keeps the writer to ~80 lines with no dependency
and no worker. Watch the two easy bugs: CRC-32 must be the standard reflected polynomial
(`0xEDB88320`, table-driven), and the central-directory offsets must be byte offsets into
the finished buffer, not member indices. ZIP64 is not needed below 4 GB.

## Gates

House pattern — node checks under `devtools/`, run with the stub, plus a Python leg where
the file format has an external reader.

- **`devtools/checkexpr.js`** — tokenizer and RPN unit cases (precedence, right-associative
  `^`, unary minus, `-x^2`, nested calls, `atan2` arity); every error path returns a message
  with the right character index and does not throw; `z` rejected in the 2D variable set;
  evaluation against hand-computed values at a few grid points, in the exact index layout;
  the non-finite guard catching `1/x`; the periodicity detector agreeing with known cases
  (`sin(2*pi*x/Lx)` periodic; `x*y` not; `abs(x-Lx/2)` continuous but kinked).
- **`devtools/checkzip.js`** — build an archive with the writer, then verify externally:
  `python3 -c` with `zipfile` (integrity, stored method, CRCs) and `numpy.load` on a
  generated `.npz`, asserting dtype, shape, axis order and values against arrays the check
  itself generated. The repo already runs Python for `gen_refvectors.py`, so this needs no
  new tooling.
- **Multi-display recording** — extend the existing mp4 check (it already asserts a uniform `stts`)
  over the composite path, plus a node check of the tiler alone: two synthetic byte patterns
  in, one composite out, every row asserted at its expected offset for both layouts.
- **Physics invariance** — nothing here touches WGSL or the solver. Assert it the way
  CHI_PLAN did: `git show` byte-identity of `physics.js` and the solver files across the
  change, and the existing reference-vector checks green.

## On-device checklist (owed after merge)

- iPhone: save-all produces one file on the strip with a working share sheet, and Chrome
  desktop raises no multi-download prompt.
- Safari: `toBlob` on the chart canvases (the display path is known good; the chart canvases
  are 2D and dpr-scaled, which is where Safari has surprised before).
- A 3D field export at 256²×64 on a phone (≈32 MB of `.npz`, 16 MB per staged field): peak
  memory, and that the staging buffer is actually released rather than pooled.
- Multi-display recording on an iPhone, at 2 and at 3 cards: **steps/s with and without
  it**, reported as a number, not
  as "seems fine"; the share sheet on the composite file; and what a WebCodecs-less engine
  does with the action (it should simply not offer it).
- One expression IC typed on a phone keyboard — the boxes must not fight the numeric
  keypad, and the warning line must not push the Run button off-screen.

## Sequencing

Item 2 first (smallest, introduces no new primitive and shakes out the shared result strip),
then the ZIP writer, then item 3, then item 4 on top of it. Item 1 is independent of all
three and is the largest single piece; it can go first or last but should not be interleaved
with the export work — the two touch disjoint files apart from the UI row table.

Item 5 is independent of items 1–4 and touches only `Recorder`, which items 1–4 do not.
It should be done **last or in a separate branch**: it is the only item that can regress
something already shipped and already fixed twice on-device (RECRAF, RECASYNC), so it wants
its own on-device pass rather than sharing one with four unrelated changes.
