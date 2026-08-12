# RECASYNC_PLAN — capture recording frames by async GPU readback, not VideoFrame(canvas)

## Why

RECRAF rounds 1–3 (RECRAF_PLAN.md) moved leg 1's feeding into the rAF loop and proved,
with on-device numbers, where the remaining iPhone display stutter lives: `?recdebug`
reads `wd 0 gap 36 ms vf 15-17 enc 1.0`. The 15–17 ms is `new VideoFrame(this.cv)` — a
synchronous canvas readback/conversion on the main thread, every slot-due pass, which
pushes exactly those passes over their vsync window ~30 times a second. The encoder is
free (1 ms), so a Worker is pointless; 15 fps would only halve the hitch rate.

The fix: never construct a VideoFrame from the canvas on the hot path. Copy the canvas
texture GPU-side (`copyTextureToBuffer` — microseconds of main-thread encoding), map the
staging buffer with `mapAsync` (non-blocking by design), and when the map resolves build
the `VideoFrame` from bytes (`VideoFrame(BufferSource)` — a ~1 MB copy, ~1 ms) and feed
the encoder. The main thread never waits on a readback; a frame simply arrives a beat
late, and the timestamps are ours anyway. Engines whose WebCodecs cannot construct a
VideoFrame from bytes keep today's canvas path — capability probe, degrade silently,
never a UA sniff (Alfred's standing rule).

Scope: leg 1 only. Leg 2 (MediaRecorder), PNG save, mp4Mux, result strip, watchdog park
condition, slot cadence (`W.due`), and all WGSL are untouched.

## Design

### 1. Canvas usage

`gpuCanvasCtx` (common.js ~488, the ONE configure site) gains
`usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC`. Unspecified
usage defaults to RENDER_ATTACHMENT alone, so the explicit value must include it. This
applies to every display card always (recording or not); a texture that is additionally
copyable costs nothing measurable, but "rendering identical" is an on-device eyes item.
The stub's `GPUTextureUsage` object needs the two constants added (it currently holds
only STORAGE_BINDING/TEXTURE_BINDING).

### 2. Probe

`recWCProbe` additionally answers "can this engine build a VideoFrame from bytes?":
construct a 2×2 frame from a 16-byte buffer with the format matching the canvas
(`canvasFormat` "bgra8unorm" → `"BGRX"`, "rgba8unorm" → `"RGBX"`; the canvas is
`alphaMode:"opaque"`, so the X variants are the honest formats) inside try/catch,
close it. Failure sets a session latch (`recBufOff`, same idiom as `recWCOff`) and the
recording runs exactly today's canvas path. The probe result rides the cfg object the
existing probe already resolves with — no second click-time await.

### 3. Capture (the hot path)

`recCapture`'s slot logic (`lastRaf`/`maxGap`, `due`, catch-up clamp, missed-slot
drops) is UNCHANGED. What changes is what a due slot does when the buffer path is on:

- Synchronously, same task as the render (the texture is transient):
  `this.ctx.getCurrentTexture()` → command encoder → `copyTextureToBuffer` into a free
  staging buffer from the pool → `queue.submit`. `bytesPerRow` is the 256-aligned
  `ceil(w*4/256)*256` (at the preset sizes 256/512/1024 the row is already aligned, but
  the code must handle padding — the box is user-sizable).
- Pool: 3 buffers per recording, created lazily on first use, owned by `W`, destroyed
  in `recStopWC`/`recBailWC` (all routes). No free buffer (3 maps in flight — the GPU
  or the map queue is genuinely behind) → `W.drop++`, slot lost, same honesty rule as
  encoder backpressure. Nothing ever waits.
- Async tail: `buf.mapAsync(READ).then(...)` hands the mapped bytes to the ORDERED
  ENCODE CHAIN (below). On the padded-row case the rows are compacted into a tight
  copy; on the aligned case the mapped view is passed straight to the VideoFrame
  constructor (the spec says it copies during construction), then `unmap` and the
  buffer returns to the pool. A rejected map (device loss, destroyed buffer at stop)
  is a drop, never a throw that escapes.

### 4. Ordered encode chain — the correctness core

`VideoEncoder` requires monotonically increasing timestamps, `mp4Mux` writes a uniform
stts (checkmp4 asserts equal deltas), and `mapAsync` resolution order across distinct
buffers is not something to rely on. So: frame index `W.n`, timestamp
`round(n·1e6/REC_FPS)` and `keyFrame: n % REC_FPS === 0` are ALL assigned at ENCODE
time, inside a single promise chain (`W.chain = W.chain.then(encodeOne)`), exactly as
`recEncodeFrame` does today — the chain guarantees arrival order = encode order, and a
dropped/failed capture leaves NO hole in the timestamps (fewer frames, never a lying
sample table; the honest-length rule from RECRAF stands verbatim). The existing
`recEncodeFrame` stays as the sync fallback path and the two share the
timestamp/keyframe/`n++` logic in one place — one function that encodes "a source"
(canvas or bytes), not two copies of the cadence rules.

Backpressure: the chain also applies the `REC_QMAX` queue check at encode time (drop,
`W.n` not advanced) — same as today.

### 5. Stop / bail / destroy drainage

`recStopWC` currently flushes then muxes. With captures in flight it must first drain:
`W.done = true` (no new captures), then `W.chain` settles (guard with a
`Promise.race` timeout of ~500 ms so a hung map cannot hold the file hostage — a
timed-out capture is a drop), THEN `enc.flush()`, then mux, then destroy the pool.
`recBailWC` (no-avcC) and `destroy()` follow the same route they do today —
`recStopWC` remains the one place a WebCodecs recording ends; the pool teardown lives
there. A late map resolving after teardown must find `W.done` and do nothing (buffers
may already be destroyed — the rejection path above covers it).

### 6. Watchdog

The watchdog (hidden page / editor view) keeps the SYNC canvas path unconditionally:
it renders off-screen anyway, a stall there costs nothing visible, and keeping the
async pool out of the background-throttled world avoids reasoning about starved maps.
One sentence of comment explaining exactly that.

### 7. ?recdebug

`vf` keeps its meaning — max main-thread ms in the capture step — which on the buffer
path is the encoder-encoding + submit (expected ~0.1–1); the async tail's
VideoFrame-from-bytes cost rides `enc`'s slot in the chain (rename display to
`cap`/`enc` only if the builder finds `vf` misleading — README must match either way).
Add `lag`: max ms from capture submit to encode (map latency — the "arrives a beat
late" number, purely diagnostic).

## Stub / bootstub

- stubenv: `GPUTextureUsage` constants; command-encoder `copyTextureToBuffer` (record
  the copy against MEMDEV's backing store semantics the other readbacks use — zeros
  are fine, the recording legs assert structure, not pixels); buffer `mapAsync`
  already exists as a resolved-promise stub — it needs to become awaitable in a way
  `env` can SEQUENCE (a hand-driven pending-map list, like the timer pump: e.g.
  `env.maps()` resolves all pending maps, so a leg can interleave captures and
  resolutions deterministically); `VideoFrameStub` accepts `(BufferSource, init)` and
  records format/codedWidth/codedHeight so legs can assert them.
- bootstub, new legs (existing legs must pass UNEDITED — the stub probe must default
  the buffer path ON so the rAF pump/handoff/parked legs run through it; if that flips
  any existing assertion the builder STOPS and reports rather than editing the leg —
  plan expectation: the pump legs assert counts/timestamps/keyframes which the chain
  preserves, so with `env.maps()` pumped between `recCapture()` calls they should hold;
  if sequencing existing legs is intractable, default the stub probe OFF and drive the
  buffer path only in the new legs — the builder documents which of the two it built):
  (a) buffer-path pump: captures then `env.maps()`, frames encode in slot order with
  exact timestamps/keyframes, VideoFrames constructed FROM BYTES (assert stub kind),
  zero canvas VideoFrames, download still `ftyp+mdat+moov`, stss on cadence;
  (b) out-of-order maps: resolve maps in reverse order, timestamps still monotonic
  (the chain reorders), no hole in stts;
  (c) pool exhaustion: 4th capture with 3 maps pending → drop counted, nothing queued;
  (d) stop-with-in-flight: stop, then `env.maps()` — file contains the drained frames,
  pool destroyed, late map after teardown is a no-op;
  (e) probe-fail fallback: `recBufOff` latched → canvas path byte-for-byte as today
  (the existing rAF pump leg's assertions re-run under the latch).
- checkmp4 untouched.

## Docs

README leg-1 paragraph: the buffer path, the ordered chain, the probe/fallback, the
watchdog staying sync, `lag` in the recdebug sentence. devtools/README: the new legs +
`env.maps()`. docs.html: NO change.

## Gates (in order)

1. WGSL baselines from clean state BEFORE edits; byte-identical after (this plan has
   no WGSL surface at all — any diff is a send-back).
2. bootstub both pages: old legs unedited-and-green, new legs green.
3. checkmp4 untouched and green (TMP caveat as before).
4. checkonepage/checks quick regression (loop and canvas-configure were touched).
5. Fresh-Fable adversarial review with the diff and this plan; fix; re-gate.

Uncommitted at the end; Alfred commits after on-device: stutter actually gone while
recording (`?recdebug`: vf ≲1, wd 0, drop ~0, lag tens of ms is fine), file plays in
Photos, and normal (non-recording) rendering looks untouched on both his laptop and
phone (the COPY_SRC configure is the one always-on change).

## Execution note (2026-08-12)

Built per plan (+ good builder calls: mid-take canvas resize drops the slot; a thrown
copy latches the take back to the sync path; bail frees its own pool). Builder choices:
stub bytes-capability boots OFF (plan's fallback option — old legs untouched on the
canvas path, each new leg arms `bufArm(true)` itself, which resets latch+tried+probe
cache as ONE lifetime); `vf` keeps its name = max main-thread ms per capture on either
path. 8 bootstub legs, all green; existing legs unedited.

Fresh-Fable adversarial review: PASS-WITH-MINORS, 0 major. Load-bearing traces verified:
chain reassignment is single-threaded-safe; encode-after-flush impossible through the
normal drain (recPend runs INSIDE the chain step) and harmless on the timeout route
(positional mux, uniform stts, never a hole); `gone`-before-destroy on all four end
routes with the stub honestly rejecting maps on destroy; PROBE CACHE SAFE (recStartWC
only reachable through a probe chain that ran recBufProbe; cache/latch share one
lifetime). Minors fixed: recEncodeMapped throw now latches bufOn=false (a probe-passing
engine failing full-size frames must degrade, not hand back a 0-chunk nothing);
timed-out drains + pool-make failure count their drops; partial pool-creation leak
destroyed on catch; stub validates bytesPerRow ≥ 4·width; README stop-order wording;
devtools/README documents bufFrames/bufArm + the default-OFF choice. Reviewer note kept
as-is: a one-beat content reorder is possible exactly at a mid-take path handoff
(timestamps stay monotonic — structurally clean).

Gates after fixes: WGSL byte-identical vs HEAD 3eabb95 (120/120, 270/270), bootstub
both pages, checkonepage 74/74, checks GATE G, checkmp4 all-pass (TMP-redirected copy;
sticky /var/tmp caveat as before). UNCOMMITTED. On-device (the acceptance test): record
with `?recdebug` — expect vf ≲1, lag tens of ms, wd 0, stutter gone; file in Photos;
and one plain non-recording look at both apps since COPY_SRC is always on.
