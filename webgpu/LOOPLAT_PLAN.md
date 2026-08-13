# LOOPLAT_PLAN — take the sync round trips out of the frame loop

Status: **written 2026-08-12; §1–§3 EXECUTED the same day, §4 deliberately not taken.**
This is audit item §4.3 (`AUDIT_2026-08-12.md`), promoted to the top of the list by an
on-device measurement. Items 1–5 of that audit are done; items 6 and 7 are
`VOLTEX_PLAN.md`'s problem and are **not** what the phone is complaining about.

## What landed (2026-08-12)

`loop()` is now the rAF driver only; **`loopPass(dtPass)`** is the pass, split out for the
same reason `renderCards` was — so `devtools/checkidle.js` drives the real function instead
of a copy. §1 stats fire-and-forget, §2 the rAF-period controller, §3 the in-flight bound.
Emitted WGSL byte-identical on both pages (120/120 and 270/270 kernels), which was the point
of taking a baseline for a change that touches none. `checkidle` 54 → 108 legs; whole
devtools suite green.

Three deliberate departures from the text above, each because implementing it exposed
something the plan could not have known:

- **The recording target is TIGHTER, not looser.** The plan says both "target ~16–20 ms" and
  "with a take live the controller targets ≤ 33 ms rather than ~16 ms", which read together
  make the take *loosen* the ceiling — but the plan's own success criterion is `gap ≈ 17 ms`
  *while recording*, and its prose says "prefer pass rate over steps-per-pass during a take".
  Implemented as: band `[PASS_LO = 18, PASS_HI = 34]`, and while a take is live the upper
  edge tightens to `PASS_HI_REC = 30`, which is inside one 33.3 ms capture slot with margin.
  A take therefore costs steps-per-pass and buys capture rate, which is what the exercise
  was for. `checkidle` pins it as a discriminating statement: a 32 ms period is tolerated
  without a take and forces a decrease with one.
- **The decrease is proportional**, not the plan's plain −1. `ms` billed an over-raised step
  count within the same pass; a pass period is one pass late *and* smoothed, so the rise
  overshoots by construction, and a plain −1 would crawl back one decrement per pass while
  each pass is the length the overshoot made it (32 steps at 68 ms → 31 further 68 ms
  passes). `round(sf·hi/dt)`, floored at −1, clears a 2× overshoot in one pass and still
  steps by exactly 1 at the band edge, so the settled behaviour is unchanged and only the
  runaway is treated. This *is* the damping the Risks section asks for; the EMA is the other
  half.
- **`statsReset` retires reads by id, not just by solver identity.** The plan lists the
  `sv === solver` retirement guard as the thing to preserve. It is not sufficient: an IC
  upload and a preset switch move the state *inside the same solver object*, so a read still
  in flight over the discarded state passes that guard and lands one pass later, putting the
  forgotten energies back in the readout and its `t` back in `simT` — which stamps capture
  filenames, the one value the plan's Risks section says must not go stale across a rebuild.
  Found by the `checkidle` leg written for the plan's own "discard a superseded arrival"
  gate, which failed on the first run. Fixed by advancing the cache's `got` id to the last
  id issued, so every read outstanding at the reset is retired whoever owns it.

**§4 (cap display renders at ~30/s) not taken**, per the plan's own instruction to decide it
with a measurement: the sandbox has no GPU. The argument against taking it blind is now
stronger than when the plan was written — the controller measures the pass period
*including* display cost, so display work converts into a lower `stepsPerFrame` rather than
into a stalled loop. On-device discriminator: `steps/s` with three display cards open versus
one. If the gap is large, the cap is ~5 lines in `needsRender` plus one `checkidle` leg.

## Adversarial review (fresh reviewer, same day) — four real defects, all in the controller

The reviewer was pointed at the diff and told to find defects. It found four, all in the
part the plan itself flagged as the risky one, and none of which the first round of legs
covered. Each is now fixed AND has a `checkidle` leg that was mutation-tested: the fix was
reverted, the leg failed, nothing else did.

1. **BLOCKING — `PASS_MAX` was a latch with no exit.** Discarding any period over 500 ms as
   "not a measurement" made the controller blind to exactly the passes it must react to: a
   genuinely slow *visible* pass is indistinguishable by length from a background stint, and
   the discard turned one into the other. Once the period crossed 500 ms the controller saw
   `dt = 0` on every pass, held `stepsPerFrame` where it was, and that kept the period over
   500 ms — a stable fixed point. Reached by ordinary UI: let `sf` climb on a cheap
   configuration (~1 s), then open a volume card. `rebuild()` is the only thing in the tree
   that resets `stepsPerFrame`, and opening a card is not a rebuild. **Fixed** by clamping
   rather than discarding, and dealing with a hidden tab by *cause* — a `visibilitychange`
   listener forgets the previous timestamp, so the resume pass is "no measurement" and the
   next re-seeds. Same discriminator `recTick` already uses.
2. **MAJOR — the controller read its own backpressure as headroom.** A pass the in-flight
   bound made skip is short *because* it dropped the work, and `paceControl` was still fed
   that period: 40 saturated passes drove `stepsPerFrame` 1 → 41 with zero steps taken, so
   the first batch submitted when the queue freed was the largest one possible — the
   opposite of what the bound is for. **Fixed**: `paceControl(n ? dtPass : 0, …)`.
3. **MAJOR — the absolute raise edge was unreachable on the target device.** `dt < PASS_LO`
   asks the pass to fit inside one 60 Hz vsync. On any device whose *bare* pass already
   overruns one — which, by this plan's own arithmetic, is probably the iPhone 11 at ~20 ms
   — the raise branch never fires and `stepsPerFrame` stays at 1 for the session, reducing
   the promised 3–12× to the pass-rate win alone. A 30 Hz or 50 Hz panel latches the same
   way. **Fixed** by making both edges relative to `pace.floor`, the cheapest period the
   device has recently managed (instant down, 1%/pass up), with the constants as floors on
   the edges. While a take is live the ceiling is `max(PASS_HI_REC, floor)`: inside a slot
   where the device can deliver one, at the floor where it cannot.
4. **MAJOR — `statsCache.busy` desynchronised permanently at the first reset.** `statsReset`
   clears the flag while a read is still out; that retired read then landed and cleared the
   flag its *successor* was holding, and from then on the loop kicked a fresh read every
   pass with two or three always in flight. Three times the map round trips this change
   exists to remove, and the held value lagging by the whole readback latency instead of by
   one pass — which `simT`, the readout and the energy trace all ride on. **Fixed** with
   `busyId`: the flag is released by its own read and by no other.

Also from that review, and taken: `statsReset` zeroes `simT` (every caller is `applyIC`, so
a filename stamped in the pass before the first read lands now says 0 rather than the
retired state's `t`); `paceFeed` no longer throws away the smoothed period on a zero-length
interval; the two weakest legs were replaced (the energy-trace one now asserts the lag as
`samples === reads - 1`, which reads `N === N` if the `await` is restored); and
`bootstub.js`'s `frame()` comment no longer claims to be the loop's per-frame work — it is
the ungated card exerciser, deliberately, and `loopPass` is what `checkidle` drives.

Ruled out by that review and worth recording: the RECRAF invariant still holds on every
path through `loopPass` (`renderCards` is above the first `await`, and both early returns
are on the far side of it); `inflight` cannot fail to decrement or decrement for an
uncounted batch; the staging pool gives each concurrent read its own buffer; and the
graveyard cannot race a pending copy, since `readBuf` submits synchronously before the
first `await`.

**Still owed: the on-device reading.** Success is judged on the phone and nothing in the
sandbox can judge it — and defect 3 above means the *shape* of the win now depends on a
number nobody has measured, the bare pass period at `stepsPerFrame = 1`. `?recdebug` prints
a `loop:` line unconditionally for exactly this: `pass <smoothed period> ms
sf <stepsPerFrame>  ifl <in flight>`, beside the per-take line. What to look for, in order:
`pass` should be a vsync multiple and not 68 ms; `sf` should be well above 1 (if it sits at
1 with `pass` at 33 ms, the bare pass is the bottleneck and §4's display cap is the next
move, not more steps); `drop` should be ≈ 0; `ifl` should mostly be 0–1.

## The measurement (Alfred, iPhone 11, 2026-08-12)

2D page, standard forced preset at 256², recording a take:

| | |
|---|---|
| `?recdebug` `gap` (worst loop-pass interval) | **68 ms** |
| `steps/s` | **40** |
| `drop` | rising steadily, ~half the slots |

Predicted from the drop rate before it was read: 67 ms. So the loop pass is ~50–70 ms and
`stepsPerFrame` ≈ 40 × 0.068 ≈ **2.7** (it will be oscillating between 2 and 3).

## What is actually wrong

**The recorder is starved by the loop, not slow.** Since RECRAF the loop is leg 1's feeder:
`recCapture()` runs once per pass and takes at most one frame, never backfilling. So
encoded frames/s = passes/s, and `drop` counts the 30 fps slots that had no pass to fill
them. At ~15 passes/s, half the slots are empty by construction.

**Almost all of the pass is message latency, not work.** Per pass the loop does:

```js
for (...) solver.step(ce);            // ~2.7 steps
renderCards(!running);                // one display chain
await device.queue.onSubmittedWorkDone();   // (1) full drain
const s = await solver.readStats();         // (2) copy + submit + mapAsync
...arrows / colorbar / cut / spectra, each its own submit + mapAsync
```

(1) and (2) are not "waiting for the GPU to compute" — the GPU work is single-digit ms and
finishes early. They are waiting for a **completion message to be delivered back into
JavaScript**, which a browser services on task-queue granularity; on Safari/iOS that is
roughly frame granularity, so ~16 ms of latency per await for a signal about ~2 ms of work.
The GPU is idle across it because of the ORDERING: the loop will not submit the next batch
until the previous batch's message arrives.

The adaptive controller cannot see any of this. `ms` is timed `t0 → onSubmittedWorkDone`,
so it stops before every readback, and it holds `ms ≤ 22`. With a 68 ms pass that leaves
**≥ 46 ms per pass — about two thirds — outside what the controller measures.**

Corollary worth stating: `ms` currently *includes* the drain's own latency, so the ~8 ms/step
it implies is an over-estimate. The step is probably 1–2 ms, against ~0.3 ms of pure FFT
arithmetic. That is why item 6 (twiddle table) is not on this list — it optimises ~0.3 ms of
a 68 ms pass.

## The change

Pipeline the loop: submit the next pass's work immediately, consume the previous pass's
numbers when they land.

### 1. Stats one frame late

`readStats` becomes fire-and-forget: kick off the read, do not await it in the pass, and
render the readout from the value that arrived. This is the pattern arrows, the colorbar and
the cut line already use. Invariants to preserve:

- the `sv === solver` retirement guard after the await (a rebuild mid-read);
- `statsCache`'s `(nsteps, stateSeq)` key from the render gate — it stays, and now also
  stops a late arrival from overwriting a newer read;
- the energy-trace push guard `s[1] > hist.t[last]`, which is what stops duplicate samples;
- `simT`, which stamps capture filenames — one frame of lag is fine, a missing value is not.

### 2. Drop the unconditional drain, and give the controller a new signal

`onSubmittedWorkDone()` exists only to make `ms` a true GPU time. Remove it from the per-pass
path and the controller loses its input — `ms` would then measure CPU encode time
(sub-millisecond) and `stepsPerFrame` would run away to 64.

Replace it with the **rAF-to-rAF pass period**, which needs no GPU sync at all, is what the
recorder actually cares about (capture rate = pass rate), and is already being computed:
`W.maxGap` is exactly this number. Target a pass of ~16–20 ms; raise `stepsPerFrame` when
passes are short, lower it when they are long, same 1–64 clamp.

**While a take is live, the pass period is the capture rate**, so the target must be ≤ 33 ms
or the recording drops slots by construction. Prefer pass rate over steps-per-pass during a
take; that is a one-line change to the target and it is the whole point of the exercise.

### 3. Bound the work in flight

Without any await the CPU can queue unboundedly, which converts latency into a growing
submission backlog — the same problem wearing a different hat. Keep a cheap bound: at most N
(2 or 3) outstanding `onSubmittedWorkDone` promises, and skip stepping on a pass while the
bound is saturated. This keeps the drain as **backpressure** rather than as a per-pass
barrier, and it is what stops a fast CPU from running the queue away on a slow GPU.

### 4. Watch the display cost this exposes

At ~15 passes/s the display chain runs 15×/s. At 60 passes/s it runs 60×/s — the render gate
does not help while running, since every pass is dirty. On 2D a display chain is
prepDisp + two inverse transforms + a max reduction + colorize, i.e. comparable to a solver
step, so this could eat a real share of the win.

Capture is 30 fps and the eye does not need more, so **cap display renders at ~30/s** (a
minimum interval in `needsRender`, forced true while a take is live). Optional, measurable,
and it composes with the gate that is already there. Decide it with a measurement, not up
front.

## Expected outcome

Passes go from ~15/s to rAF-bound (60/s on this device). `steps/s` follows
`stepsPerFrame × passes/s`:

| if one step really costs | stepsPerFrame settles ~ | steps/s | vs 40 now |
|---|---|---|---|
| 8 ms | 2 | ~120 | 3× |
| 2 ms | 8 | ~480 | 12× |
| 1 ms | 16 | bandwidth-capped ~300–500 | ~10× |

The bracket is wide because the true step cost is unknown *for the reason this plan
exists* — `ms` currently measures latency as much as work. **The fix is also the
measurement.** On a desktop expect 10–20%, not this: the round trips are cheap there. This
is an old-mobile-device change.

Success is judged on the phone, on the same 2D 256² forced preset, recording:
`gap` ≈ 17 ms, `drop` ≈ 0, `steps/s` up by ≥ 3×.

## Gates

No WGSL changes at all — this is JS control flow, so the whole byte-identity suite should
stay green untouched, and that is itself an assertion worth making. Capture a fresh baseline
first anyway (`devtools/dumpwgsl2.js` + `kdiff.py`).

New/extended legs in `devtools/checkidle.js`, which already has the harness:

- the loop does **not** await stats inline (a stub whose `readStats` never resolves must not
  stall the pass — drive `renderCards`/the pass and assert it completed);
- a late stats arrival for a superseded state is **discarded**, not rendered (the
  `statsCache` key does this; pin it);
- the energy trace still gets exactly one sample per distinct `t`, with the lag;
- `stepsPerFrame` rises when passes are short and falls when they are long, driven by
  injected pass periods rather than by real time;
- with a take live, the controller targets ≤ 33 ms rather than ~16 ms;
- the in-flight bound: with N submissions outstanding, a pass steps nothing and still
  captures;
- (if §4 is taken) the display render cap holds at ~30/s while running, and is bypassed
  while recording.

`bootstub.js`'s `frame()` helper mimics the loop by hand and will need to follow whatever
the loop's shape becomes — check it does not silently diverge.

## Risks

- **The controller change is the risky part**, not the await removal. A pacing loop that
  reads its own output can oscillate; damp it the way `spsSmooth` already damps steps/s, and
  assert stability in the gate with injected pass periods.
- **`simT` and filenames.** One frame of lag is fine; a `NaN` or a stale value across a
  rebuild is not. The retirement guard matters more once nothing is awaited inline.
- **This is the third change in a row to the frame loop** (RECRAF, the render gate, now
  this). Re-read the RECRAF invariant before touching `renderCards`: `recCapture()` must stay
  in the same synchronous task as the render, before any await, or captures wrap an expired
  texture.
