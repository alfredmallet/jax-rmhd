# VOLTEX_PLAN — the volume raymarch samples a 3D texture, not a storage buffer

Status: **written 2026-08-12, not started.** Item 7 of `AUDIT_2026-08-12.md`. Items 1–5 of
that audit are done; item 6 is deliberately not scheduled (last section).

## Why

`vsamp` (`rmhd3d.html`, in `volRay`'s fragment shader) does software trilinear
interpolation out of a storage buffer: **eight scalar loads per sample**, returning the
value and — from the same eight corners — the finite-difference gradient the shading needs.

At the shipped 512² canvas and `VOL_STEPS = 256`:

- 67 M samples per frame per volume card
- **537 M scalar buffer loads per frame**

The early-out at `VOL_AMAX` helps on opaque views and nothing on the wispy ones the feature
exists for. There is no empty-space skipping. This is, by a wide margin, the most expensive
thing either app does, and it is the path a phone hits in the volume view.

**It is NOT the iPhone 11 slot-drop story** (Alfred, 2026-08-12). That was on the **2D**
page, which has no volume view and never runs this shader — an earlier draft of this plan
claimed it as motivation and was simply wrong. See the last section: the arithmetic rules
this out, rules item 6 out too, and points somewhere else entirely. This plan stands on the
3D volume view's own cost and nothing else, which means it is a **quality-of-experience
change for the 3D page, not a fix for a known device problem** — worth knowing when Phase C
decides whether it earned its keep.

A `texture_3d<f32>` with a linear sampler replaces those eight loads with **one hardware
fetch**: filtering in fixed-function hardware, and a tiled/swizzled memory layout that the
texture cache is built for instead of a linear one that it is not.

The render gate from the audit (item 1) already removed this cost when the page is *paused*.
This is the other half: the cost while it is *running*, which is when a visitor is actually
looking at it.

## This reopens an ISO_PLAN fixed decision — on a fact it did not consider

`ISO_PLAN.md` says, under "do not relitigate":

> **No 3D textures.** r16float is not a storage-writable format and r32float sampling needs
> an optional device feature.

Both halves are **true**, and neither is the whole list. **`rgba16float` is both
storage-writable and filterable in core WebGPU** — it is in the storage-texture format
table (write-only access, which is all we need) and 16-bit float formats are filterable
without any optional feature. `r32float` sampling needs `float32-filterable`; `rgba16float`
needs nothing.

That is the entire justification for revisiting this. It is not a disagreement with the
original decision; it is a format the original decision did not enumerate. If that claim is
wrong, this plan dies at Phase A and ISO_PLAN's decision stands — **verify it against the
spec and against a real adapter before writing any other code.**

The four channels are not waste, either: the shading needs the gradient, and `vsamp`
currently derives it from the same eight corners it loads. Packing
`(value, ∂x, ∂y, ∂z)` into the texture makes one filtered fetch return everything the
march needs, which is why this is `rgba16float` and not `r16float` with three more taps.

## Fixed decisions

- **`dispR` does not change.** It stays the fp32 storage buffer it is, feeding `colorize`,
  `colorizeCube`, the reductions and the slice path exactly as now. The volume texture is a
  SEPARATE resource written by one new kernel. This is the difference between a change with
  a blast radius of two kernels and one that touches every display kernel in the file, and
  it is what lets the byte-identity gates stay meaningful.
- **Written after the max reduction, normalized.** fp16 tops out at 65504 and carries ~3
  decimal digits; raw field values are neither bounded nor conditioned for that. The writer
  kernel runs *after* `maxPartial`/`maxFinal` in the same pass and stores `f / max`, so the
  texture holds values in [−1, 1] where fp16 has ~11 bits of mantissa. The march's `lev`
  is already a fraction of the max (`vu.par.x * s`), so it becomes a bare fraction and the
  `* s` drops out.
- **Lazily allocated, volume view only.** 256²×64 × 8 B = **33.5 MB** per volume-showing
  chain. Built on the first frame a card is in the volume view and destroyed with the
  chain, the same discipline as `_genInit`. A session that never opens a volume allocates
  none of it.
- **`vsamp` keeps its name and signature.** The texture version is a body swap, not a new
  call shape. This is a testability constraint, not an aesthetic one — see Gates.
- **Sigma modes are out of scope.** They already fall back to the cube faces in the volume
  view (ISO_PLAN B), so the texture path never sees them.
- **No empty-space skipping in this plan.** It is the obvious next lever and it is a
  different change with a different risk profile (an occupancy mip, and a march that is no
  longer a fixed step count, which is what the convergence gate pins). Not here.

## Phases

### Phase A — establish the premise (no app changes)

Confirm on a real adapter, and write the numbers into this file:

1. `rgba16float` accepts `STORAGE_BINDING | TEXTURE_BINDING` on a `"3d"` texture.
2. A filtering sampler on it works — sample a known ramp, check interpolation actually
   happened rather than snapping to texels.
3. `maxTextureDimension3D` ≥ 256 (core minimum is 2048; this is a formality, but 64²×256
   long boxes make z the interesting axis).
4. Baseline timing of the volume view at 256²×64, so Phase C has something to beat. Use
   the existing `?recdebug` readout's frame gap, or a timestamp query if it is available.

**If (1) or (2) fails, stop.** ISO_PLAN's decision was right and the fallback in Phase D is
the whole remaining change.

### Phase B — the writer kernel + the texture path

- New kernel `volTexWrite`: reads `dispR` and `maxVal`, writes
  `(f/max, ∂x f, ∂y f, ∂z f)` into `texture_storage_3d<rgba16float, write>`. Gradients in
  the BOX-UNIT metric (`Lx/nx, Ly/ny, Lz/nz`) — with elongated boxes the metric matters,
  same rule the current `vsamp` follows. Periodic wrap in the differences, as now.
- `renderVol` gains a `texture_3d<f32>` + `sampler` binding; `vsamp`'s body becomes one
  `textureSampleLevel`. The march, the slab test, the shell Gaussians, the shading, the
  wireframe and the early-out are **untouched**.
- `Solver.render`'s volume branch encodes `volTexWrite` between the max reduction and the
  render pass. Everything else in that branch is unchanged.

Nothing else in either app changes. The 2D page does not change at all.

### Phase C — measure, then decide

Compare against Phase A's baseline at 128²×64 and 256²×64, on a desktop GPU and on the
iPhone. Report frames/s in the volume view and the `?recdebug` gap during a take.

**The change is justified only if this is a real speedup on the phone.** If it is not — if
the app was bound by something else all along — say so in this file and revert. A 33.5 MB
texture and a reopened fixed decision are not worth a desktop-only win.

### Phase D — the fallbacks (independent of A–C)

Both are cheap, neither needs the texture, and either can ship alone if Phase A kills the
plan:

- **Half-resolution volume + upscale.** ISO_PLAN already names this as the intended
  fallback ("a knob, not a redesign"). 4× off the top.
- **`VOL_STEPS` while moving.** 256 was measured as *converged for a still picture*, which
  is a strictly stronger requirement than a frame shown for 16 ms. Fewer steps while
  running, full quality on the paused frame — which the render gate now draws exactly once,
  so the still image the visitor studies is the converged one. This composes well with the
  gate and is probably the best effort-to-payoff item in the whole plan.

Given how Phase A might go, **do Phase D first.** It is cheap, it needs no format
verification, and if it is enough on device then A–C are optional.

## Gates

The hard part. **The current Phase B leg of `devtools/checkiso.js` EXECUTES the emitted
fragment shader** on wgsl_reflect's WGSL interpreter and compares it pixel by pixel against
an independently written CPU reference march (~1e-6 achieved, 1e-5 tolerance). That is the
strongest test in this repo and a hardware texture fetch almost certainly breaks it — the
interpreter will not model `textureSampleLevel` on a filtered 3D texture, and if it does it
will not model the hardware's filtering precision.

The verification therefore SPLITS, and the plan must be honest that this is a downgrade:

1. **The march stays proved.** Because `vsamp` keeps its signature, the interpreter leg
   injects a software `vsamp` and executes the march exactly as it does today — slab entry,
   step count, shell Gaussians, shading, accumulation, early-out — against the same CPU
   reference. What is being tested is the marching, which is where the physics-visible
   behaviour lives, and it is untouched by this plan.
2. **The texture path is pinned as WIRING, not as pixels.** In the style of
   `sigrcheck.js`: the emitted `volTexWrite` is parsed, its normalization and its
   box-unit gradient are mirrored in fp64 against an analytic field, and the bind groups,
   texture descriptor (format, dimension, usage) and encode order are asserted by name off
   the stub. The stub gains a 3D storage texture that keeps its bytes, so "what was written
   is what the march would read" is checkable; the FILTERING is not, and that is the gap.
3. **The gap is closed on device, by eye, once.** A volume view of the collision preset
   before and after, same seed, same state: the shells must sit in the same places. This is
   a screenshot comparison and it should be said plainly rather than dressed up as a test.
4. **`VOL_STEPS` convergence leg is unaffected** — it is about the reference march.
5. **Byte-identity:** `checkiso`'s changed-kernel list gains `renderVol` (changed) and
   `volTexWrite` (added), and the three volume-length kernels assertion gains the new one.
   Every physics kernel must still be byte-identical to the base commit; if any is not, the
   change has leaked out of the display path and is wrong.
6. **fp16 quantization is measured, not assumed.** One leg mirrors the write-read round
   trip in fp64 with fp16 rounding applied and reports the worst-case relative error on a
   realistic normalized field. If it is worse than ~1e-3 the shells will band and Phase C
   should say so.

Also: `dup.py` clean, `names.mjs` clean, every emitted kernel parses, and the full suite
(`bootstub` ×2, `checkonepage`, `checkidle`, `check2dspec`, `checkgc`, `checks`, `checkj`,
`checkk`, `checksh`, `checkaniso`, `checkpin`, `sigrcheck`, `contrepro` ×2, `layout`)
re-run green. Capture a FRESH WGSL baseline from clean git state before editing.

## Open questions

- **Does the writer pass eat the win?** It is 4.2 M texel writes per volume frame against
  537 M loads saved, so on paper it is nothing — but it is a full volume pass at display
  cadence, and the display cadence is per rendered frame. If Phase C shows it dominating,
  the answer is to write the texture only when `dirty` (the render gate already knows) and
  let the march re-read an unchanged texture, which is free.
- **Long boxes.** 64²×256 puts 256 on the z axis; check the sampler's periodic addressing
  mode (`address-mode: repeat`) actually wraps on all three axes rather than clamping, or
  the wrap `vwrap` does by hand today is silently lost at the boundary.
- **Does anything else want this texture?** The field-line marcher does its own manual
  bilinear gather (`samp2`). Not in scope, but if the texture proves out, that is the next
  consumer and the reason to keep the writer kernel general.

## What the iPhone 11 slot drop probably IS — and it is neither 6 nor 7

Alfred corrected the record on 2026-08-12: the ~half-slot drop was on the **2D** page. That
page has no volume view, so item 7 cannot be the cause. The arithmetic then rules out item 6
as well, and points at the item the audit deprioritized.

At 256², from `encodeRHS`: one step is **3 stages × ~10 field transforms**, i.e. **11.8 M
butterflies**, in **29 dispatch calls across 3 compute passes**. The report is 36 steps/s,
so one step takes **27.8 ms**. What that step demands:

| | demand at 36 steps/s | A13 capability | fraction |
|---|---|---|---|
| ALU (18 flop-equivalents/butterfly) | ~7.7 GFLOP/s | ~0.7–1.0 TFLOP/s | **~1%** |
| FFT memory traffic | ~1.1 GB/s | ~34 GB/s | **~3%** |

At 512² it is still only ~4% and ~13%. **The device was not saturated on either axis** — the
work in a step accounts for well under a millisecond of the 27.8 ms it takes. "GPU-saturated"
in the campaign log is not supported by the numbers and should be treated as an open question
rather than a finding.

That leaves latency and overhead, and the frame loop has an obvious candidate — audit §4.3,
which this round explicitly deferred:

```js
await device.queue.onSubmittedWorkDone();   // full pipeline drain
...
const s = await solver.readStats();          // + a map round trip
```

With `stepsPerFrame` adapted down to 1 (which it will be at 27.8 ms/frame), that is a **full
GPU drain plus a buffer map round trip per step**, with the GPU idle across both. On iOS
Safari a drain-plus-map is plausibly tens of milliseconds, and it would show up exactly as
"one step takes 27.8 ms while the step's own work is ~1 ms". The render-gate work of this
round already removed the *paused* half of this cost (`statsCache` skips the map entirely
when nothing moved), but the running path is untouched.

**Alfred confirmed (2026-08-12): standard 256² forced, `drop` rising steadily, ~half the
slots.** That pins the loop period without another measurement.

**Why a recorder counter measures the LOOP.** Since RECRAF the loop is the encoder's feeder:
`recCapture()` runs once per loop pass and takes at most one frame, never backfilling. So
encoded frames/s = passes/s, and `drop` counts the 30 fps slots that had no pass to fill
them. The encoder is not slow; it is starved. (`drop` is leg 1's counter, i.e. WebCodecs,
i.e. the iOS path — consistent with where this was seen.)

So: half of 30 slots lost → ~15 passes/s → **one pass ≈ 67 ms**. With 36 steps/s that gives
**stepsPerFrame ≈ 2.4**. And the adaptive controller holds its own measurement at `ms ≤ 22`
(above that it decrements) — but `ms` is timed from `t0` to `onSubmittedWorkDone()`, i.e. it
stops *before* `readStats`, the arrow gather, the colorbar, the cut line and the spectra:

> **≥ 45 ms of every ~67 ms pass — about two thirds — is spent after the controller stops
> looking, in waits it cannot see and therefore cannot adapt to.**

**What the waits actually are.** Not "waiting for the GPU to compute": the GPU work in a pass
is single-digit ms and finishes early. The wait is for a *completion message* to be delivered
back into JavaScript, which a browser services on task-queue granularity — on Safari/iOS
roughly frame granularity, so ~16 ms of latency for a signal about 2 ms of work. The GPU is
idle across it because of the ORDERING: the loop will not submit the next batch until the
previous batch's message arrives. Two such round trips per pass (`onSubmittedWorkDone`, then
`readStats`'s `mapAsync`) plus the throttled chart readbacks account for the missing 45 ms.

At stepsPerFrame ≈ 2.4 the controller's equilibrium also implies ~8 ms of GPU per step
against **~0.3–1.7 ms of actual FFT arithmetic** (11.8 M butterflies ≈ 0.2 GFLOP). So the
step is mostly overhead as well — 29 dispatch calls in 3 compute passes, ~0.27 ms apiece,
the right ballpark for barriers between dependent compute dispatches on a tile-based mobile
GPU.

Two structural costs, in priority order — and **neither is item 6 or item 7**:

1. **The post-render waits (audit §4.3), ~45 ms of ~67.** Fix: pipeline it. Submit the next
   pass's work immediately and consume the previous pass's numbers when they land — the
   one-frame-late pattern arrows, the colorbar and the cut line already use. The
   unconditional drain exists only to make `ms` a true GPU time; a periodic drain or a
   timestamp query would serve the controller as well. The render gate landed in this audit
   already removed this cost when PAUSED (`statsCache`); this is the running half, and it is
   much the larger one. Because capture rate = pass rate, halving the pass period roughly
   doubles the frames that reach the encoder.
2. **Dispatch/barrier count inside the step, ~8 ms.** 29 calls per step, ~0.3–1.7 ms of it
   arithmetic. Fewer, larger dispatches, or fewer compute-pass boundaries. Worth doing only
   after (1).

`?recdebug`'s `gap` from that session confirms or kills this outright — it is the
loop-pass interval, so the prediction is **gap ≈ 67 ms**. If it reads ≈28 ms then
stepsPerFrame was 1, the drops came from somewhere else, and this analysis is wrong.

## Item 6 (FFT twiddle table) — not scheduled, and why

`AUDIT_2026-08-12.md` §3 proposes replacing the per-butterfly `cos`/`sin` in `fftKernel`
with a precomputed table. The arithmetic there is right — ~1 G butterflies and ~2 G
transcendental evaluations per step at 256²×64 — and a table computed in f64 on the CPU is
*more* accurate than an fp32 `PI*k/p` division followed by an fp32 `cos`, so it is not the
usual speed-for-precision trade.

**But the 20–40%-of-FFT-time estimate in that audit is a guess, and it assumes the FFT is
ALU-bound — which the section above shows it is not, at least on the one device there is
evidence about: ~1% of an A13's ALU at the reported step rate.** A workgroup-shared-memory Stockham kernel is quite plausibly bound by LDS
bandwidth and barriers instead, in which case removing ALU work buys nothing at all — and
this change costs a fresh WGSL baseline, a reference-vector regeneration, and an update to
every byte-identity gate.

**Measure first.** The cheap discriminator: emit a variant kernel with the twiddles
replaced by a constant (numerically wrong, fine for timing) and time one RHS. If the FFT
does not get materially faster, it is not ALU-bound and item 6 is worth nothing; close it
in the audit file and move on. Only if it does should the table be built.
