# LOOPLAT_PLAN — take the sync round trips out of the frame loop

> # STATUS: TRIED OVER FOUR ROUNDS AND REVERTED (2026-08-12)
>
> Implemented, adversarially reviewed, measured on three devices, and backed out at Alfred's
> call — the code in `common.js`, `devtools/checkidle.js` and `devtools/bootstub.js` is
> byte-identical to `989ec07` again, checkidle back to its 54 legs, WGSL byte-identical
> throughout. **Nothing below this banner describes the shipped app.** It is kept because
> the four rounds produced one measurement that is worth more than the change was, and that
> measurement closes other items in the audit too. Read the post-mortem immediately below
> before re-opening any of this.
>
> ## Post-mortem: what was actually learned
>
> **1. The loop was never the cost. `tB 1.7 ms` of a `pass 44.3 ms`.** With the latency
> removed and the pass instrumented, the whole synchronous pass — every display chain, the
> capture submit, and the encode of 21 solver steps — is 1.7 ms. `tR 0.4`, `tS 1.6`. The loop
> is **idle 96% of every pass**, waiting for rAF, which waits for the GPU to drain enough to
> present. The app is GPU-bound at roughly 2 ms of GPU per step, and no arrangement of
> JavaScript control flow moves that.
>
> **2. So three audit items are answered, and two of them are dead.**
> - **§4.3 (this plan) is dead.** It removed ~46 ms of real latency per pass and the pass
>   period genuinely fell (68 → 25–35 ms). It did not deliver throughput, because throughput
>   was never latency-limited.
> - **§4.2's display-render cap (the "§4" of this plan) is dead.** `tR 0.4 ms` is what it
>   would have been capping. The dirty-render *gate* that already shipped is about idle
>   power and stands; capping the *rate* of something that is 1% of a pass cannot move a
>   pass.
> - **Item 6, the FFT twiddle table, is the one that is now MORE interesting, not less.**
>   The postscript closed it on the grounds that it optimises ~0.3 ms of a 68 ms pass. That
>   arithmetic assumed the pass was latency. It is not: the pass is ~2 ms/step of GPU work,
>   the FFT is essentially all of a step, and the twiddles are plausibly 20–40% of the FFT.
>   Same for the dispatch count inside the step (ranked 2 in the postscript). **Those are
>   where the remaining throughput is.**
>
> **3. The adaptive `stepsPerFrame` controller is a trap, and three designs failed in three
> different ways.** All of them try to steer one number by watching a period they cannot
> measure independently of it:
> - *Absolute band* `[18, 34]` / `[18, 30]` recording: on a machine whose minimum achievable
>   pass is 31.7 ms, no period is ever under 18 (never raises) and every period is over 30
>   (always cuts). **Latches at `sf = 1`, 30 steps/s.** Measured, on the laptop.
> - *Floor-relative band* `[max(18, 1.25·floor), max(30, floor)]`: `lo` collapses onto `hi`
>   the moment `floor` > ~24 ms. **Zero-width dead band, so it cannot hold and flaps** —
>   `steps/s` swinging 500 ↔ 30 for the length of a take. Measured, on all three devices.
> - *Bucket-relative band* `dt/floor ∈ [1.25, 1.6]`: reproduces neither reading in
>   simulation and settles 8% lower where it can be modelled. Thrown away before shipping.
>
> The common root is that on a GPU-bound device the pass period is an **output** of the
> controller (it scales with `stepsPerFrame`), so `pace.floor` is not a property of the
> hardware and any band anchored on it reads its own tail. Anyone reopening this should
> consider whether the loop needs an adaptive controller at all, rather than which band
> shape to try fourth.
>
> **4. Process notes.** The plan's own baseline numbers were wrong and that cost two rounds:
> `steps/s` was `n / ms` with `ms` timed to the drain, so it measured busy time and
> over-reported ~3×, and the `stepsPerFrame ≈ 2.7` derived from it was really 1. The
> adversarial review caught this and it was written down and then not applied when the first
> results came in. **Instrument the thing before optimising it**: `tB` / `tR` / `tS` took
> fifteen lines and would have stopped this plan being written.
>
> What did work, and is worth remembering if the recorder is ever touched again: with the
> per-pass drain gone, a capture's `copyTextureToBuffer` queued behind the step batch and its
> map came back 65 ms later, saturating a 3-buffer staging pool and dropping slots. Rendering
> and capturing ahead of the steps, plus a 6-buffer pool, took `lag` 65 → 7 ms and pool drops
> to zero. That coupling is real and will come back if the drain is ever removed again.

Original status: **written 2026-08-12; §1–§3 EXECUTED the same day, §4 deliberately not taken.**
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

## The on-device reading (Alfred, 2026-08-12, three devices, 2D forced, recording)

| | before (iPhone 11) | laptop | Alfred's phone | second iPhone 11 |
|---|---|---|---|---|
| `pass` (smoothed) | 68 ms | 24.7 | 21.4 | **34.9** |
| passes/s | 14.7 | 40.5 | 46.7 | **28.7** |
| `sf` | **1** | 13 | 11 | **1** |
| steps/s, TRUE | **~15** | ~525 | ~515 | **~29** |
| `drop` share of slots | ~50% | 9% | 8% | **25%** |
| `lag` | — | 65 ms | 52 ms | 37 ms |
| `ifl` | — | 2 | 2 | 1 |

**Read the bold column pair and nothing else.** It is the only controlled comparison in the
table: iPhone 11 before, iPhone 11 after. The laptop and Alfred's phone have no
before-reading, so no multiple can be claimed for them, and comparing their absolute numbers
against an iPhone 11 baseline — which is what a first pass over this table invites — is
meaningless.

**Two corrections to the numbers this plan was written on**, both of which shrink the result:

- **`sf` was 1 before, not 2.7.** The "≈ 2.7" at the top of this file was back-derived from
  `steps/s = sf × passes/s`, i.e. on the assumption that the readout's `steps/s` was true
  throughput. It was not: the old `spsSmooth` was `n / ms`, and `ms` was timed to the drain,
  which the controller held at ≤ 22 ms — about a third of a 68 ms pass. 40 steps/s displayed
  with `ms ≈ 25 ms` means `sf = 1`, already clamped, which is also why the controller could
  not lower it further. The adversarial reviewer flagged this and it was recorded above; it
  was then not applied to the first reading of these results, which is the error.
- **So the readout's `steps/s` changed meaning across this change** — `n / pass` now, true
  throughput, where it was `n / busy time` before. The displayed number is roughly 3x
  smaller for the same real work. Anyone comparing a remembered 40 against a fresh 29 will
  conclude it got worse; it did not.

**Like-for-like, then: ~15 → ~29 steps/s and drops halved. About 2x, not the 3–12x this plan
predicted.** The pass-period win is real and large (68 → 35 ms, passes 14.7 → 28.7/s), and
all of the throughput gain comes from it. None comes from `sf`, which is still 1 — because
the premise that removing the latency would leave room for more steps is false on this
device. At `sf = 1` the pass is already 34.9 ms: one step plus one display chain plus the
throttled readbacks. **The remaining bottleneck is the display chain, not the loop**, and the
next move is measuring the bare pass rather than more loop surgery.

`drop` also did not reach zero, and the reason has moved: it is no longer the loop either.

**The new bottleneck is the capture staging pool, and this change caused it.** `REC_POOL`
was 3, sized against the loop as it then was — an unconditional drain per pass meant the
queue was empty when a capture's `copyTextureToBuffer` was submitted, so its map came back
promptly. With the drain gone, that copy was queued behind this pass's `stepsPerFrame` steps
*and* up to `INFLIGHT_MAX - 1` earlier batches: `ifl 2` x `sf` 11-13 is ~25-39 steps of GPU
work ahead of it, and at ~2.5 ms/step that is 60-70 ms — which is exactly the measured
`lag`. Against a 33.3 ms slot that is two captures permanently in flight, three on jitter,
and the next slot dropped for want of a buffer. Latency was not removed, it was moved.

Fixed two ways, both landed: `renderCards` (and with it `recCapture`) now runs **before** the
step batch, so the copy is at the head of the pass's submissions rather than behind them;
and `REC_POOL` is 6, sizing the pool in slots of readback latency (200 ms) rather than
against a drain that no longer exists. `drop` is also **split by cause** under `?recdebug`
now — `[slot N pool N enc N ...]` — because one counter with six ways to reach it had drops
rising on three devices at once with no way to tell a late pass from a saturated pool from
encoder backpressure, and those have three different fixes.

**The second iPhone 11 is a different problem and is not fixed.** `sf 1` at `pass 34.9 ms`
means one step plus one display chain plus the readbacks already overruns a capture slot:
the device is display-bound, not step-bound, and the take ceiling correctly refuses to add
steps it cannot afford. That is the case this plan's own hand-off note predicted ("if `sf`
sits at 1 with `pass` ≈ 33 ms, the bare pass is the bottleneck and §4 is the next move, not
more steps"). It costs throughput against the old behaviour — 40 steps/s became ~29 — because
the old controller banked steps into a 68 ms pass and this one holds the pass down to protect
the capture rate. Whether that trade is the right one on a slow phone is a judgment call, not
a measurement, and it is open. `gap 627 ms` there is a separate one-off stall worth a look;
`gap` is a max over the whole take and includes the take's own startup, so `pass` is now the
steady-state number to read and `gap` the outlier detector.

## Round 4 (2026-08-12): `tB 1.7` of a `pass 44.3`, and the flapping was a closed dead band

```
loop: pass 44.3  fl 31.7  sf 21  ifl 2
busy: tB 1.7  tR 0.4  tS 1.6 ms
```

**The loop is idle for 96% of every pass.** All display chains plus the capture submit cost
0.4 ms of main thread; encoding 21 steps costs 1.6 ms; the whole synchronous pass is 1.7 ms
of 44.3. Nothing in this file's remit is spending that time — the pass period is set by how
fast rAF comes back, and rAF comes back when the GPU has drained enough to present. **The app
is GPU-bound, and the loop's own cost is noise.** Two consequences, and they close two
questions that have been open since the audit:

- **§4 (cap display renders at ~30/s) is dead.** `tR 0.4 ms` is what it would be cutting.
  Whatever the display costs, it is not costing it on the CPU, and capping the *rate* of a
  thing that is 1% of the pass cannot move the pass. Audit §4.2's premise (a paused page
  burning rAF on full raymarches) was about idle power and was already fixed by the render
  gate; it is not a throughput lever here.
- **The remaining `[slot]` drops are not lateness in any sense this loop can fix.** A pass is
  late for its slot because the GPU queue in front of it is deep, and the queue is deep
  because `stepsPerFrame` put it there. That is the controller's trade, not a defect.

**And the flapping was mine.** `lo = min(1.25 * floor, hi)` collapses onto `hi` the moment
`floor` exceeds `PASS_HI_REC / 1.25` ≈ 24 ms — raise edge and cut edge on the same number,
zero hysteresis, so the controller *cannot* hold and swings at whatever amplitude the
proportional cut gives it. `fl 31.7` is squarely in that range, as were all three devices in
round 3. That floor-relative band came in as the fix for an adversarial review's theoretical
"absolute edges latch `sf` at 1 on a slow device"; it was the wrong fix, and it was wrong
twice: the collapse above, and — now that `tB` says the loop is GPU-bound — the fact that
`pace.floor` is an *output* of this controller (the period scales with `stepsPerFrame`), so a
band anchored on it is reading its own tail. On a vsync-bound device the floor would be the
panel's interval and the idea would have been sound, which is what made it plausible.

Reverted to the round-2 absolute constants, which were *measured* stable on the same laptop
at `sf 13`, `pass 24.7`, drop 9%. `pace.floor` stays in the `?recdebug` line as a diagnostic
— it is how the degenerate band was spotted — and drives nothing. The new gate states the
property as behaviour rather than as an inequality between two constants: fifty passes at a
period anywhere inside the band must not move `stepsPerFrame`, at any `pace.floor`. Restoring
the round-3 band fails it at exactly the slow entries.

## Round 3 (2026-08-12): the pool half is fixed, and the loop is no longer the question

Reading after the pool + ordering fix. Laptop, recording: `pass 33.3  sf 1  ifl 1`,
`drop 10 [slot 10]`, `lag 7 ms`, `raf 453`. iPhone 11 and Alfred's phone: the same shape,
every drop `[slot]`.

- **`lag` 65 → 7 ms and pool drops → 0.** Submitting the capture copy ahead of the step
  batch, plus `REC_POOL` 6, did exactly what the arithmetic said. That half is closed.
- **Every remaining drop is `slot`** — the loop arriving after a slot was due — and `pass` is
  33.3 ms, which is *two vsyncs* on a 60 Hz panel, with `sf 1`. So the bare pass (one step,
  one display chain, the throttled readbacks, the capture submit) has overrun a single
  vsync, and the take ceiling is then correct to refuse a second step.
- Alfred also reports `steps/s` flapping 500 ↔ 30 during a take.

**A controller rewrite was written for that flapping and thrown away.** The theory was that a
band in milliseconds can contain no period the panel can deliver (16.7 below it, 33.3 above
it, nothing in between), so it must sawtooth; the fix was to express the band in buckets of
`dt / floor` with a dead band containing 1.5, the EMA of an alternating pass. Simulated
against a cost model of both readings, it **reproduces round 2 (`sf` 12–13, pass ~25 ms) and
does not reproduce round 3 at all**, and it settles 8% lower in the regime it can model. To
get `sf 1` out of *either* controller you need the bare pass to exceed one vsync — and then
both latch identically, because what binds is not the band's shape but `dt > max(PASS_HI_REC,
floor)`, which is the take ceiling doing its job.

So the controller is not the open question and no third design of it is going in. **The open
question is what the bare pass spends 33 ms on**, which two rounds have now been spent
guessing at. `?recdebug` grew a second line for exactly that:

```
busy: tB 4.2  tR 3.1  tS 0.6 ms
```

`tB` is the pass's whole synchronous cost before its first await, `tR` the display chains
plus the capture submit, `tS` the step encode; all maxima over the session. The reading is
`tB` against `pass`:

- **`tB` small (single-digit ms of a 33 ms pass)** — the loop is *idle*, waiting on vsync or
  on the compositor, and cutting display work will not help. The cost is on the GPU or in
  presentation, and the next move is neither this plan nor §4.
- **`tB` most of the pass** — then `tR` says whether the display chain is the thing to cut,
  which is what audit §4.1 / §4.2 have been waiting on since they were written.

Until that number exists, both remaining options — hold the pass at a slot and accept ~29
steps/s, or let it grow and accept the drops — are being chosen blind.

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

> **WRONG, corrected after the change landed — see "The on-device reading" below.** That
> derivation assumes `steps/s` is throughput. It was `n / ms`, and `ms` stopped at the drain,
> so it measured busy time and over-reported by ~3x. `stepsPerFrame` was **1**, clamped, and
> true throughput was ~15 steps/s. Every estimate below that leans on 2.7 — including the
> 3–12x outcome table — is inflated by the same factor.

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
