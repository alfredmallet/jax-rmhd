# RECRAF_PLAN — capture recording frames in the rAF render, demote the timer to a watchdog

## Why

On iPhone the display stutters while a WebCodecs recording is live (Alfred, on-device,
2026-08-12). The recorder rebuild (2026-08-10) moved iPhone from MediaRecorder/
`captureStream` — capture off the main thread — to leg 1, whose `recTick` runs on a
33 ms `setInterval`: an EXTRA full `render()` per tick plus a `VideoFrame` GPU copy plus
encode, all on the main thread, at arbitrary phase relative to the rAF loop. The double
rendering and the timer/rAF beating are removable: the rAF loop already renders every
card every frame, and a capture taken in the SAME TASK as that render sees a live
`getCurrentTexture` and costs zero extra renders. The share-button commits are innocent
(post-finish UI only); this plan changes only leg 1's FEEDING side. Muxer, timestamps,
keyframe cadence, both stop paths, leg 2, and the result strip are all untouched.

## Design

**Split capture from cadence.** The frame index `n`, its timestamps
`round(n·1e6/REC_FPS)` µs, `keyFrame: n % REC_FPS === 0`, and the `REC_QMAX`
backpressure drop all stay exactly as they are. What changes is who calls the encode.

1. **Shared encode helper.** Factor the body of today's `recTick` from the backpressure
   check down (queue check → `VideoFrame` → `encode`/`finally close` → `n++`) into ONE
   helper (e.g. `recEncodeFrame(W)`), so the VideoFrame/timestamp/keyframe logic exists
   once and both paths below call it. The helper does NOT render.

2. **rAF path (new, the fix).** `DisplayCard.recCapture()`, called from `loop()`
   immediately after `d.render()`, in the same synchronous task, BEFORE any await —
   WebGPU has no preserveDrawingBuffer, so a capture deferred even one microtask sees an
   expired texture (same reason `saveShot` re-renders). Loop change is exactly:

   ```js
   if (!icDraw.on) for (const d of cards.disp) { d.render(); d.recCapture(); }
   ```

   `recCapture()` body:
   - gate: `const W = this.wc; if (!W || W.done) return;`
   - mark the rAF loop alive: `W.lastRaf = performance.now()` — on EVERY call, captured
     or skipped; this is what parks the watchdog.
   - slot cadence: capture only when the next 1/REC_FPS slot is due. `W.due` is the
     wall-clock time of that slot. If `now < W.due` return (between slots — not a drop).
     Otherwise advance: if `now - W.due > T` (rAF stalled past a whole slot) count the
     missed slots into `W.drop` and set `W.due = now + T` — never backfill; the honest-
     length rule stands (fewer recorded seconds than wall clock, never a lying sample
     table). Else `W.due += T` (drift-free at nominal cadence). `T = 1000/REC_FPS`.
   - then the shared helper (which may still drop on `REC_QMAX`).

   On a 120 Hz phone this captures on ~every 4th callback and renders NOTHING extra.

3. **Timer path (kept, demoted).** `W.timer` stays a `setInterval` at `1000/REC_FPS`;
   its tick becomes: if `this.wc !== W || W.done` return; if
   `performance.now() - W.lastRaf < REC_RAF_STALE` return — rAF is feeding frames;
   otherwise EXACTLY today's `recTick` body: backpressure check, `this.render()`, shared
   helper. No slot-due check on this path — when the watchdog is the feeder (background
   tab, editor view owning the screen, the stub harness) the tick cadence IS the 30 fps
   promise, as it is today. `REC_RAF_STALE = 2.5 * 1000 / REC_FPS` (~83 ms: two missed
   display frames at 30 Hz means the rAF loop is not delivering). New module constant
   beside `REC_QMAX`, with a comment saying what it means.

   Neatly, the watchdog's extra-render cost recurs only exactly where there is no
   visible display to stutter.

4. **`W` gains `due` and `lastRaf`**, both initialized in `recStartWC` from ONE
   `performance.now()` sample: `lastRaf = t0; due = t0 + T`. (Initializing `lastRaf`
   live gives rAF the first REC_RAF_STALE ms to show up; a genuinely dead rAF hands over
   after ~83 ms, well inside one slot's worth of loss.)

5. **Everything else unchanged**: `recStopWC` / `recBailWC` / `destroy()` already clear
   `W.timer` and null `this.wc` — `recCapture`'s gate makes a late loop iteration a
   no-op. Mixed feeding (watchdog frames, then rAF resumes) is safe by construction:
   `n` is monotonic and timestamps are nominal `n/30` on both paths. Leg 2 untouched.
   `mp4Mux` untouched. No WGSL anywhere near this.

## Stub/test consequences (devtools)

- stub `requestAnimationFrame` is a no-op and stub `performance.now` advances 250 ms per
  CALL, so in bootstub the watchdog check `now - lastRaf` is ALWAYS stale → every
  existing `env.tick(n)` recording leg drives the timer path ≡ old `recTick` and must
  pass UNCHANGED (pump counts, stall/drop, bail, 30 s cap, destroy-mid-record, mux
  shape). Any existing-test edit is a red flag to raise, not a thing to do silently.
- New bootstub legs (both pages boot the same common.js, one page suffices unless a leg
  is cheap to run on both):
  a. rAF pump: press rec, then call `cards.disp[0].recCapture()` k times via `run()`.
     The 250 ms/call clock makes every call slot-due, so `W.n` advances per call;
     assert n, the drop count of the skipped slots' accounting (clock jumps 250 ms
     ≫ T, so `W.drop` grows by the missed slots — assert it is counted, not
     backfilled: chunk count == n), stop → download still `ftyp+mdat+moov`,
     stss on the forced-keyframe indices.
  b. Watchdog handoff: rec, a few `recCapture()` calls, then STOP calling it and
     `env.tick(m)` — frames continue on the timer path (stale clock), `W.n` grows by m.
  c. Watchdog parked: set `W.lastRaf = Infinity` via `run()` (the one deterministic way
     to make `now - lastRaf` negative under the stub clock), `env.tick(5)`, assert
     `W.n` unchanged — proves a live rAF loop parks the timer.
- `checkmp4.js` (real ffmpeg/ffprobe gate) must still pass untouched — it drives
  `mp4Mux` directly and nothing in the mux changed.

## Docs

- `webgpu/README.md`, the leg-1 paragraph ("A `setInterval` at 1000/30 ms …"): rewrite
  to describe rAF-side capture in the render task + slot cadence + the watchdog timer,
  and WHY (the iPhone stutter: double render + timer/rAF beating on the main thread).
- `webgpu/devtools/README.md`, bootstub bullet: add the three new legs.
- `docs.html`: NO change (no user-visible control changed).

## Gates (in order)

1. BEFORE editing, from clean git state: `dumpwgsl2.js` baselines for BOTH pages.
2. Build.
3. `dumpwgsl2.js` + `kdiff.py` both pages: byte-identical (pure JS change; any WGSL
   diff is an automatic send-back).
4. `node bootstub.js .. rmhd2d.html` and `.. rmhd3d.html`: all legs green, old
   recording legs UNEDITED.
5. `node checkmp4.js`: green, untouched.
6. Adversarial review by a fresh reviewer with the diff and this plan; fix findings.

Uncommitted at the end; Alfred commits after on-device eyes (the stutter itself is
only observable on the phone).

## Execution note (2026-08-12)

Built (opus) + adversarially reviewed (fresh reviewer): PASS-WITH-MINORS, all three
fixed — (1) watchdog-fed frames re-base `W.due` so they are never double-booked into
`W.drop` when rAF resumes (+ bootstub assertion), (2) `recCapture()` wrapped in
try/catch in `loop()` so a VideoFrame/encode fault keeps the old one-tick blast radius
instead of freezing the app, (3) `REC_RAF_STALE` widened 2.5 → 3.5 slots (~117 ms) for
the loop's post-render readback awaits. Gates after fixes: WGSL byte-identical vs clean
HEAD (2D 120/120, 3D 270/270), bootstub both pages green (old recording legs unedited),
checkmp4 all-pass (via a TMP-redirected copy — sandbox `/var/tmp` sticky-dir EACCES;
re-run the real one where `/var/tmp` is writable). On-device iPhone check owed: stutter
gone while recording, file still plays.
