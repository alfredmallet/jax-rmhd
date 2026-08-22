# FFTPERF_PLAN — what the step's transforms cost, then twiddle table / radix-4 / gradient chunking

Written 2026-08-21. **Phase 0 landed and Phase 1 measured 2026-08-21** (execution notes at the end;
the Phase 1 record is in §4): A and B not taken (§9.1), **2C in progress**. Prompted by `plans/MEMORY_PERF_PLAN.md` closing on the
solver: of its four shipped phases, Z1 (the Elsasser-separable propagator) is what
`rmhd3d.html`'s stage kernel has always been, Z2/F2 have no counterpart here, F4 is dead
(`realGrads` is shared with the field-line pass), and F1's diagnosis — the eight-lane gradient
stack is the largest allocation in the step — transfers exactly. What does NOT transfer is any
timing number: the P100 ladder says the jax step is transform-bound (79–89% of the step in
the FFTs, propagator gone from the accounting), and the demo's step has only ever been
measured as a whole.

**This reopens `AUDIT_2026-08-12.md` item 6, and says why.** The audit closed the twiddle
table on a scaling measurement: 256² → 512² took `steps/s` 500 → 125, ratio 4.00, GB/s
pinned at 31, "a kernel running at 1–2% of ALU peak". Three things about that closure:

1. The ALU figure counts a transcendental as one op (79 Gflop/s + 16 G transcendental/s).
   A precise f32 `cos` is a range reduction plus a polynomial — tens of instructions on every
   backend, the exact count depending on Metal/D3D/Vulkan and a fast-math policy we do not
   control. At 16 G/s that is plausibly a few hundred G instructions/s, a real fraction of a
   laptop GPU's issue rate, not 1–2% of it.
2. The ratio test separates *launch-bound* from *work-bound* (a 4× grid through the same 34
   dispatches cost 4× — that conclusion stands). It does not separate bandwidth from ALU from
   barrier latency: bytes scale as N, butterflies and twiddles as N log N, and over one
   doubling those differ by 12%, inside the noise of an adaptive controller's `steps/s`.
3. 31 GB/s is, in the audit's own words, "3–10× under any laptop GPU's peak". A kernel that
   far below the memory roof is not at the memory roof; whatever it is waiting on is unmeasured.

So the audit's ranking ("**Measure first**", item 6 after a discriminator) is followed here
rather than its postscript. Alfred's 2026-08-12 call — reopen only if the algorithm changes or
someone wants the resolution ceiling raised — is honoured by construction: item C below IS
the resolution-ceiling item, and nothing in A or B ships without the Phase 1 numbers saying
it moves the needle. If Phase 1 confirms the closure, this plan records that with numbers and
stops.

Execution per the standing flow: Fable overseer, opus implementers, fresh-Fable adversarial
review per phase. **Every timing number in this plan comes from Alfred's devices** — the
sandbox has no GPU, so Phase 1 and the timing gates of Phase 2 are his runs, with the harness
Phase 0 builds for exactly that.

## 0. Principles

1. **No new user-facing knobs.** A `?bench` URL flag (precedent: `?recdebug`, `?demo=`) is a
   developer entry point, not a control; it adds nothing to the page without the flag. No
   kernel variant ships behind a switch — Phase 1 decides, Phase 2 ships one form, the loser
   is deleted. Module-level emit options that only the bench reads (`fftKernel`'s `probe`) are
   the one allowed seam, and a gate asserts the app never sets them.
2. **Comments say what the code does.** History, numbers and rationale live here and in
   `webgpu/README.md`.
3. **Byte-identity expectations are stated per phase.** `devtools/dumpwgsl2.js` + `kdiff.py`
   before and after every phase; the set of kernels allowed to differ is named in the phase.
   The `checkiso`/`check2dspec`/`checkeigf` pins on `physics.js` kernels against their BASE
   commits fire by design only where a phase says so (item C's `prepGrads`), never as a side
   effect.
4. **Measured, not chained.** A ratio is new/reference against a NAMED reference (device,
   page, resolution, median-of-R, which kernel), quoted from the bench's JSON. No "table × the
   FFT share" arithmetic: the table's gain is measured after it lands, on the same cells.
5. **One variant per page load, nothing else running, plugged in.** The LOOPLAT post-mortem's
   lesson — the old `steps/s` was `n / ms` to a drain the controller itself was steering —
   is why the bench has no controller: it saturates the queue and times the drain of a known
   batch. Interleave A/B/A/B within a session; discard the first rep (pipeline warm-up).

## 1. What is known (the record, not new measurement)

- Laptop (Apple M1 class, Chrome), 2D forced 256²: ~525 steps/s true, i.e. ~1.9 ms/step of
  GPU; loop idle 96% (LOOPLAT). 512²: 125 steps/s. Alfred's phone ~515 steps/s at 256²;
  iPhone 11 ~29.
- One 2D step = 3 stages × (prepGrads → 8-lane inverse 2D transform → bracket → 2-lane
  forward 2D transform → nlAssemble → forcingAdd → stage), + energy/tick/OU/scale once.
  Each 2D transform is a row pass (`fftRowPair`: real↔half-spectrum, N = ny) and a column
  pass (`colsFwd`/`colsInv`, N = nx, stride `NKY`); 3D adds a z pass (`zFwd`/`zInv`, N = nz,
  stride `NMP`). 30 field-transforms per step in 2D, 11.8 M butterflies at 256².
- `fftKernel` (`common.js:198`): radix-2 Stockham, `WG = fftWG(N)` = N/2 clamped to
  [32, 256], `buf` = 2N `vec2<f32>` of workgroup memory (16 KiB at N = 1024 — exactly the
  default `maxComputeWorkgroupStorageSize`, which is why `NMAX_LINE = 1024`), log₂N stages
  each behind a `workgroupBarrier`, and per butterfly `ang = ∓PI·k/p; cos(ang); sin(ang)`.
- Allocation (field-equivalents `cx = nm·8` bytes): 2D `gradsK` 8, `specTmp` 8 (its 8-lane
  size is set ONLY by the gradient chain; the NL chain uses 2), `realGrads` 8·nr·4; 3D
  `gradsK` 8 (in place, no `specTmp`), `realGrads` 8·nr·4. At 256²×64: `gradsK` 129 MiB,
  `realGrads` exactly 128 MiB — the first is why the 3D page asks for `maxLimits`.
- The jax P100 accounting (`docs/performance.md`, "P100 addendum"): transforms 79–89% of the
  step, inverse transforms alone 69–77%. A hypothesis for the demo, not a number.

## 2. The three candidates

| | what it removes | what it cannot touch | moves `steps/s`? |
|---|---|---|---|
| **A. twiddle table** | the per-butterfly `cos`/`sin` (2·N/2·log₂N transcendentals per line) | bytes moved, barrier count | only if the kernel is ALU-bound |
| **B. radix-4** | half the stages: half the barriers and half the workgroup-memory round trips; 25% fewer twiddle multiplies | bytes moved | only if the kernel is barrier/shared-memory-bound |
| **C. gradient chunking** | 6 of 8 lanes of `gradsK` (and of 2D's `specTmp`): the step's peak allocation | bytes moved (identical), transcendentals, barriers | **no** — memory and the resolution ceiling only |

Neither A nor B changes global-memory traffic: every line is loaded once and stored once
regardless. If Phase 1 says the kernels are at the memory roof, both are dead and the audit's
"fuse the two 1D directions through shared memory / transpose so columns read coalesced" is
the only route left — which stays closed per 2026-08-12 (no path to 10×, not a production
code). C is in this plan because it shares the harness and is the one item the reopening
condition names; its own gate is "no slower", not "faster".

## 3. Phase 0 — the harness (`?bench`)

**Where.** In-page, both apps, because the 3D `Solver` lives inline in `rmhd3d.html` and has
no module to import from. Behind `?bench` in `location.search`: a small panel under the
readout (phones have no console) with one button per campaign cell and a `<textarea>` that
accumulates one JSON record per run — copy-paste is the transport. A console API
(`window.bench`) exposes the same functions for the laptop.

**What it measures.** Three things, all by saturating the queue and timing the drain:

1. *Whole step*: `K` back-to-back `solver.step` calls (`cflEvery` = the page's setting,
   the frame loop held off so nothing renders or reads back meanwhile), one
   `await onSubmittedWorkDone()` at the end, median of `R` reps, ms/step. [As landed: K
   submits with one drain, not K steps in one submit — `step` owns its submit and its
   per-step tail, and an encode-only step would be a second copy of it. So the number
   includes the per-step JS encode and the OU noise draw, exactly as the app's loop does;
   read it as "a step as the app runs it", not as pure GPU time.] Plus the
   same with `solver.step`'s forcing off — not a decision input, a consistency check against
   the solver's own structure.
2. *Per kernel, isolated*: one pipeline (`rowsC2R`, `colsInv`, `zInv`, `rowsR2C`, `colsFwd`,
   `zFwd`, `prepGrads`, `bracket`, `nlAssemble`, `stage`) dispatched `reps` times in ONE
   compute pass on the solver's own buffers with the solver's own dispatch shape, timed the
   same way, µs per dispatch. Results are garbage afterwards; the bench calls `setIC` when
   done. This is per-kernel timing with nothing beyond core WebGPU — `timestamp-query` is
   optional polish, not a dependency (not every target ships it).
3. *The ladder*: the FFT kernels re-emitted from `fftKernel` with `o.probe` set, compiled by
   the bench only, each run exactly as (2):
   - `probe: "copy"` — `load` and `store` only, no stages: the kernel's memory-traffic floor
     (the strided column/z reads included, which is the point).
   - `probe: "consttw"` — every stage, every barrier, every butterfly, with `wc = 1.0`,
     `ws = 0.0` in place of `cos`/`sin`. The multiply is kept in the text, but `1.0·x` folds
     under any compiler, so `T_bf` is a LOWER bound on the butterfly cost and `T_tw` an upper
     bound on the transcendentals' — read the ladder with that sign in mind.
   - default — the shipped text, byte for byte.
   A pipeline's `auto` layout is its own, so the bench builds its own bind groups for the
   variants over the same buffers.

From the three: `T_mem = copy`, `T_bf = consttw − copy` (butterflies + barriers +
shared-memory round trips), `T_tw = full − consttw` (transcendentals). The bench prints each
as µs and as a share of `full`, per kernel.

**What it reports**, per record: page, `gpuInfo`, resolution, `cflEvery`, K, R, each cell's
median and min, bytes moved per step (computed from `nm`/`nr` and the dispatch list —
Appendix A gives the formula so the number is checkable by hand) → GB/s, butterflies per step
→ G butterflies/s. Never a ratio against another device.

**What it must not do.** Steer anything (no `stepsPerFrame`, no controller); run while the
loop runs (`running = false` first, like the self-test button); leave a variant pipeline
where the app can reach it; exist without the flag.

**Also in Phase 0, because Phase 2 needs it before any kernel changes:** a self-test row at
the PRODUCTION transform sizes. Today the FFT is verified at 32² (2D) and 8×16×16 (3D); a
radix-4 rewrite at N = 256 or 1024 has no on-device check. Add to both pages' self-test a
row "forward transform at the selected resolution, analytic": fill real space with a sum of
three cosines at known `(kx, ky[, kz])` and amplitudes, transform with the production
pipelines, and compare the whole spectrum against the analytic one (three nonzero bins, zero
elsewhere) at `relL2 ≤ 1e-5`. No refvector file — the reference is computed in JS at any N.
Plus the existing roundtrip row at the selected resolution. Both rows are GREEN on the
shipped kernels before Phase 2 touches them — that is the baseline the rows exist for.

**Gates.**
- `dumpwgsl2` + `kdiff`: every kernel on both pages byte-identical to the pre-phase dump, at
  every preset and every override the dump knows. The bench's variants are not in the dump
  and must not be.
- New `devtools/checkbench.js` under `stubenv`: (i) without `?bench` the panel element does
  not exist and `window.bench` is undefined; with it, both exist; (ii) `fftKernel(o)` with no
  `probe` is string-equal to the pre-phase emission for N ∈ {8, 16, 32, 64, 128, 256, 512,
  1024} and both directions (a captured copy lives in the check); (iii) `probe: "consttw"`
  differs from the default in exactly the two twiddle lines, `probe: "copy"` has no
  `workgroupBarrier` inside a stage loop; (iv) the bytes-per-step formula reproduces a
  hand-computed number for 2D 256² and 3D 128²×64 (Appendix A) exactly; (v) the analytic
  self-test row's reference generator returns the three bins and zeros elsewhere for two
  `(N, k)` choices (pure JS, no GPU).
- [As landed, after two review rounds, `checkbench.js` also carries: the Solver's OWN
  compiled FFT text pinned against the fixture at the self-test grid, the default preset
  and the largest offered grid; `FFT_PROBE === null` on a freshly booted page before any
  campaign; every spec cell's pipeline, bind group, dispatch extent, lane count and buffer
  order compared against a recorded `solver.step`, with every `stepIO` entry dispatched
  exactly `n` times and nothing dispatched the table neither counts nor excuses; the frame
  loop held off during a campaign, including the iteration in flight at the click; R+1
  drains with the first discarded; the `finally` on the probe seam; the self-test's
  mid-test solver rebuild; `process.exitCode` non-zero unless the summary line is reached.]
- `bootstub`, `checkidle`, the whole devtools suite: green untouched.
- The self-test rows: green on-device on the shipped kernels (Alfred, once, both pages, at
  the default and the largest resolution).

**Files.** `common.js` (`fftKernel` gains `o.probe`; the bench module — one function block,
gated, placed with the self-test helpers; the analytic self-test row's JS reference),
`rmhd2d.html` / `rmhd3d.html` (the `?bench` panel spec and the per-page kernel/bind-group
list the bench iterates; the new self-test rows), `devtools/checkbench.js`,
`devtools/README.md` (one entry).

## 4. Phase 1 — the campaign (Alfred, on device)

Devices: the laptop (names the GPU via `gpuInfo`), Alfred's phone, and the iPhone 11 if it
is still to hand — the slowest device is where `steps/s` decides whether a recording drops
slots, and a phone GPU's balance of ALU to bandwidth is not a laptop's.

| page | resolutions | cells |
|---|---|---|
| 2D | 256², 512², 1024² | whole step; `rowsC2R`, `colsInv`, `rowsR2C`, `colsFwd` × {full, consttw, copy}; `prepGrads`, `bracket`, `nlAssemble`, `stage` |
| 3D | 128²×64, 256²×64, 64²×256 | whole step; the six FFT kernels × {full, consttw, copy}; the rest; **the gradient chain at batch 8 vs 2** (four `encodeInv3D(p, 2, …)` calls against one `(p, 8, …)` — a one-argument variant the bench can drive without item C's code) |

R ≥ 5 per cell, interleaved, first rep discarded, one page load per resolution. The
`64²×256` cell is the one where N_z = 256 makes the z kernel the largest line and where the
strided `NMP` read is most hostile — it is there to see whether the z pass behaves like the
column pass or differently.

**Decision table** (per kernel, on the laptop unless the phone disagrees in direction, in
which case both are reported and §9 decides):

| finding | consequence |
|---|---|
| `T_tw ≥ 15%` of the kernel on the kernels that carry the step (by (2), the per-kernel share of the whole step) | **A goes ahead** (§5). 15%, not 0: a table costs one cached load per butterfly, so expect to recover most but not all of `T_tw`. |
| `T_bf ≥ 30%` | **B goes ahead** (§6). Radix-4 removes about half of `T_bf` (half the stages) and a quarter of `T_tw`; the threshold is set where that half-share is worth a rewrite of the one kernel everything depends on. |
| `T_mem ≥ 70%` | both dead. Recorded in this file with the numbers, the audit's closure confirmed, and the column-stride note stands as the only route. |
| batch-2 gradient chain ≤ 1.03× the batch-8 time | **C goes ahead** (§7) on both pages. Above 1.03× on any device: C is dropped; the memory it would save is recorded. The Phase 1 cell re-transforms lanes 0–1 four times (no per-pair bind groups exist yet), so a two-lane working set that fits a laptop's last-level cache flatters batch-2; it is a proxy, and 2C's own timing gate re-measures with the real per-pair bind groups. |

Both A and B can be true; then A lands first (the table is what radix-4's three twiddles
index) and B is measured against post-A.

**Deliverable.** A table in this file — one row per (device, page, resolution, kernel), the
three ladder numbers and the shares — pasted from the bench's JSON. Nothing in Phase 2 is
briefed until it exists.

### Phase 1 record

**Laptop, `apple metal-3` (Chrome), 2D 256², `cflEvery` 4, K 20 / R 5 / reps 50 (2026-08-21).**
Whole step **1.515 ms** (min 1.49); by the Appendix A convention 47.0 GB/s, 7.8 G butterflies/s.
Per-kernel µs per dispatch at the step's lane count, and ×3 against the step (the eight cells
sum to 1,422 µs = 94% of it):

| kernel | lanes | µs med (min) | ×3 / step |
|---|---|---|---|
| rowsC2R | 8 | 150 (148) | 30% |
| colsInv | 8 | 94 (90) | 19% |
| rowsR2C | 2 | 52 (52) | 10% |
| colsFwd | 2 | 38 (36) | 7.5% |
| prepGrads | — | 38 (36) | 7.5% |
| bracket | — | 34 (30) | 6.7% |
| nlAssemble | — | 30 (28) | 5.9% |
| stage | — | 38 (36) | 7.5% |

Ladder (µs med: full / consttw / copy, then the shares):

| kernel | full | consttw | copy | T_mem | T_bf | T_tw |
|---|---|---|---|---|---|---|
| rowsC2R | 148 | 100 | 48 | 32% | 35% | 32% |
| rowsR2C | 42 | 32 | 16 | 38% | 38% | 24% |
| colsInv | 88 | 72 | 70 | 80% | 2% | 18% |
| colsFwd | 48 (min 36) | 40 | 40 | 83% | 0% | 17% |

Reading: FFTs are 66% of the step (rows 40%, columns 26%). The row kernels split three ways;
the column kernels are 80% memory floor — the strided `NKY` read (their `copy` rung moves
bytes at ~60 GB/s against the rows' ~88) with the butterflies hidden behind it. Summed over
a step, `T_tw` = 246 µs = 16% (an upper bound, §3) and `T_bf` = 210 µs, of which radix-4
could take about half on the row kernels only. `colsFwd` is the one noisy cell (258
workgroups; med/min 48/36); re-run the 2-lane cells at `reps` 200. A second standalone
"per kernel" run drifted ≤ 5% from the one inside `all`.

**Laptop, same device and settings, 2D 512² and 1024², 3D 128²×64 (2026-08-21).** Whole step
6.705 ms (512², 42.4 GB/s, 7.93 G bf/s), 30.18 ms (1024², 37.6 GB/s, 7.82 G bf/s),
25.475 ms (3D, 47.5 GB/s, 8.40 G bf/s). **Step time scales as N log N, not N**: the
256²→512²→1024² ratios are 4.43 and 4.50 (N log N predicts 4.50 and 4.44; bytes 3.99), the
butterfly rate is flat at 7.8–7.9 G/s across the three grids and 8.4 G/s in 3D while GB/s
falls 47→42→38. The audit's 4.00 was read through the adaptive loop; the direct step
measurement pins the step to butterflies, not bytes.

Per-kernel shares of the step (cells sum to 93% / 94% / 99% of the whole step):

| kernel (lanes) | 2D 512² µs | share | 2D 1024² µs | share | 3D 128²×64 µs | share |
|---|---|---|---|---|---|---|
| rowsC2R (8) | 672 | 30% | 3158 | 31% | 2254 | 26.5% |
| colsInv (8) | 384 | 17% | 1806 | 18% | 1210 | 14.2% |
| zInv (8) | — | — | — | — | 1202 | 14.2% |
| rowsR2C (2) | 172 | 7.7% | 794 | 7.9% | 548 | 6.5% |
| colsFwd (2) | 108 | 4.8% | 482 | 4.8% | 328 | 3.9% |
| zFwd (2) | — | — | — | — | 310 | 3.7% |
| prepGrads | 196 | 8.8% | 884 | 8.8% | 782 | 9.2% |
| bracket | 186 | 8.3% | 728 | 7.2% | 724 | 8.5% |
| nlAssemble | 142 | 6.4% | 590 | 5.9% | 306 | 3.6% |
| stage | 222 | 9.9% | 1048 | 10.4% | 752 | 8.9% |
| all FFTs | | 60% | | 62% | | 69% |

Ladder shares (T_mem / T_bf / T_tw, %; full / consttw / copy µs in the JSON):

| kernel | 2D 512² | 2D 1024² | 3D 128²×64 |
|---|---|---|---|
| rowsC2R | 44 / 23 / 33 | 37 / 38 / 26 | 51 / 24 / 25 |
| rowsR2C | 31 / 36 / 33 | 36 / 39 / 25 | 55 / 15 / 30 |
| colsInv | 88 / 0 / 12 | 76 / 8 / 16 | 96 / 1 / 3 |
| colsFwd | 75 / 2 / 23 | 76 / 8 / 16 | 87 / 1 / 12 |
| zInv | — | — | 101 / 0 / −2 (noise) |
| zFwd | — | — | 97 / −1 / 4 |

3D grad chain (C's cell): batch 8 = 4,720 µs, 4 × batch 2 = 4,600 µs → **0.975×** (the
cache-optimistic proxy; 2C's gate re-measures with real per-pair bind groups).

`T_tw` summed over a step: 16% (256²), 16% (512²), 14% (1024²), 9.5% (3D) — upper bounds;
`T_bf`: 14%, 10%, 17%, 7%, of which radix-4 could take about half, on the row kernels only.

**Laptop verdicts against the decision table.** **A goes ahead**: `T_tw` 12–33% on every
row/column kernel at every grid (the z kernels show none — they are a memory floor), realistic
whole-step gain ~10% in 2D, ~6% in 3D, almost all from the row kernels. **B is held until
post-A**: it clears the 30% `T_bf` bar on the 2D row kernels (35–39%), not on the 3D rows
(15–24%) nor on any column/z kernel; half the rows' `T_bf` is 5–8% of a 2D step, ~3.5% of 3D.
**C goes ahead** (0.975× ≤ 1.03×). "Both dead" does not apply. The column and z kernels —
18–36% of the step — are 76–100% memory floor (the strided `NKY`/`NMP` read) and outside this
plan's reach; the audit's fuse/transpose route stays closed.

**Laptop, 3D 256²×64 (2026-08-21).** Whole step **99.355 ms** (48.4 GB/s, 9.53 G bf/s); the
ten cells sum to 99.2% of it. Per kernel (µs, share of step): zInv 4,646 (14.0%), colsInv
4,948 (14.9%), rowsC2R 8,558 (25.8%), rowsR2C 2,028 (6.1%), colsFwd 1,260 (3.8%), zFwd 1,182
(3.6%), prepGrads 3,024 (9.1%), bracket 2,874 (8.7%), nlAssemble 1,216 (3.7%), stage 3,102
(9.4%); all FFTs 68% (rows 32%, cols 19%, z 18%). Ladder (T_mem / T_bf / T_tw, %): rowsC2R
52 / 13 / 34, rowsR2C 57 / 7 / 37, colsInv 97 / 1 / 1, colsFwd 97 / 0 / 3, zInv 102 / −3 / 1,
zFwd 101 / −1 / 1. `T_tw` over a step ≈ 11.7% (upper bound), `T_bf` ≈ 3.8%. Grad chain:
batch 8 = 18,780 µs, 4 × batch 2 = 18,530 → **0.987×**. The 256-point row lines carry the
largest transcendental share measured anywhere (34–37%) with the smallest butterfly share
(7–13%): A is carried by the row kernels at every grid on both pages, B has nothing to take
in 3D, C stays under its bar. Laptop complete; 3D 64²×256 not run (the z kernel at its
longest line — optional, the z kernels show a pure memory floor at every measured grid).

**Phone, `apple apple apple apple` (Safari), 2D 256² and 3D 128²×64 (2026-08-21).** Safari
quantizes `performance.now()` to 1 ms, so per-dispatch cells resolve to 20 µs at reps 50:
the 2D 256² ladder (cells of 20–260 µs) is ±1 quantum and is NOT read for shares; the 3D
cells are hundreds to thousands of µs and are. Whole step 1.75 ms (2D 256², 40.7 GB/s,
6.76 G bf/s — 1.16× the laptop) and 35.15 ms (3D, 34.4 GB/s, 6.09 G bf/s — 1.38× the
laptop); the ten 3D cells sum to 99% of the step. 3D ladder (T_mem / T_bf / T_tw, %): zInv
82 / 13 / 5, colsInv 70 / 24 / 6, rowsC2R 68 / 26 / 6, rowsR2C 69 / 27 / 4, colsFwd
71 / 25 / 4, zFwd 76 / 19 / 5. Grad chain: batch 8 = 7,400 µs, 4 × batch 2 = 7,400 →
**1.00×**. (2D 256² whole step per kernel, ±20 µs: rowsC2R 260, colsInv 140, rowsR2C 60,
colsFwd 40, the four non-FFT kernels 20 each.)

**The phone disagrees with the laptop in direction on A**: `T_tw` is 4–6% on every kernel
(laptop rows 25–37%), `T_bf` 13–27% on every kernel (laptop: rows 24–39%, cols/z ~0), the
memory floor 68–82% (laptop: rows 32–57%, cols/z 76–102%). Likely cause, unmeasured: the
browser's fast-math policy (WebKit → Metal with hardware sin/cos; Dawn → precise software
sin/cos), not the GPU — which would also make the phone's transform the less accurate one
today; the analytic self-test row's relL2 on both devices is the one-click check. Verdicts
with both devices: **A** go on the laptop (realistic ~10% 2D, ~6% 3D), ≤5% ceiling and a
possible small regression on the phone — 2A's timing gate becomes two-sided (laptop gain
≥ 0.7·T_tw on the row kernels; phone whole step ≤ 1.02×; analytic relL2 not worse on
either); **B** held (under the bar on the phone everywhere, marginal on the laptop); **C** go
(1.00× phone, 0.975–0.987× laptop). §9.1 raised to Alfred.

**§9.1 decision (Alfred, 2026-08-21): A and B NOT taken** — 5–10% of the step on one device
and ≤5% on the other is not worth a rewrite of the kernel everything depends on; the audit's
2026-08-12 call stands, now with the ladder numbers behind it. **C goes ahead**, as the
memory/resolution-ceiling item it is, with its "no slower on either device" gate. Phase 1
closed; Phase 2A/2B stay in this file as designs not executed.

## 5. Phase 2A — the twiddle table

**Design.** One read-only storage buffer per distinct line length the page transforms (2D:
`ny` and `nx` — equal except in the rectangular boxes; 3D: `nx`, `ny`, `nz`), holding
`w_N^j = exp(−2πi·j/N)` for `j ∈ [0, N)` as `vec2<f32>` — N entries, not N/2, so radix-4's
three twiddles `w^{qk'}` index directly without a sign trick; 8 KiB at N = 1024. Computed in
JS in float64 and rounded once; uploaded once at solver construction. The inverse uses the
conjugate (`vec2(w.x, −w.y)`), emitted by direction exactly as the sign of `ang` is today.

Per butterfly at stage `s` (p = 2^s): the angle `∓π·k/p` is `w_N^{k·N/(2p)}`, i.e. index
`k << (LOGN − 1 − s)`; a shift, no division, no transcendental. Stage 0 has `k = 0` always,
so its twiddle is exactly `(1, 0)`; the multiply may be skipped there — `1·x − 0·y` is `x`
bit for bit for finite operands, so the specialisation is neutral, and the emitter keeps it
as a separate first stage only if the bench shows it matters.

**Why a storage buffer and not the two alternatives.** `var<workgroup>`: the 1024-point line
already sits at the 16 KiB default workgroup-storage limit with its ping-pong `buf`; a
workgroup table would break `NMAX_LINE` on default-limit devices. A WGSL module-scope `const
array<vec2<f32>, N>`: dynamically indexed const arrays are lowered per backend, and Tint has
been known to materialise them as per-invocation private copies — a risk the implementer
checks on the laptop's backend by the bench before choosing it; the storage buffer has no
such question (one L1/L2-cached 8 KiB read stream, the same on every backend) and is the
audit's own proposal. The binding is appended LAST in each FFT pipeline's group so the
existing entries keep their numbers and the bind-group lists change by one appended entry.

**Bitwise expectation: NOT bitwise, by design, and characterised.** The device's `cos`/`sin`
are replaced by float64-rounded values; every transform output moves at the ulp level.
Gate: the self-test's forward-transform row (32² / 8×16×16) and the Phase 0 analytic row at
production N stay ≤ 1e-5 (they should IMPROVE; the implementer records before/after
`relL2` for both rows on the laptop in this file). The `kdiff` set allowed to differ:
`rowsR2C`, `rowsC2R`, `colsFwd`, `colsInv`, `zFwd`, `zInv` — nothing else, on either page.
The `physics.js` pins (`checkiso`, `check2dspec`, `checkeigf`) do not see these kernels and
must stay green without a base bump.

**Timing gate.** Re-run the Phase 1 cells for the FFT kernels and the whole step, same
device, same session, interleaved against a pre-2A page load: gain on each kernel ≥ 0.7·T_tw
(the table's cost is what eats the rest), whole-step ≥ the kernel gains weighted by their
Phase 1 shares, within noise. Falls short → find the cost (a miscompiled table read, a
bank conflict in the lookup) before closing; a gain under 5% of the step with no found cause
is a §9 decision (ship as a precision improvement, or revert).

**Files.** `common.js` (`fftKernel`, `fftRowPair`; a `twiddleTable(N)` helper; the bench's
variant list gains the new default), `solver2d.js` (`_buildBuffers`, the FFT bind groups,
`colsFwd`/`colsInv` decl), `rmhd3d.html` (same three places plus `zFwd`/`zInv`),
`devtools/checkbench.js` (the captured default emission is regenerated — the check's
"default is byte-identical to the capture" leg is what pins this phase's text from then on),
`devtools/checkfft.js` (new, §6 — its table leg lands here: the JS table vs
`Math.cos`/`Math.sin` at every N, exact after f32 rounding).

## 6. Phase 2B — radix-4

**Design.** The stage sequence is fixed by LOGN at emit time: `⌊LOGN/2⌋` radix-4 stages, then
one radix-2 stage if LOGN is odd (128, 512, 8 and 32 have it; 16, 64, 256, 1024 do not). A
radix-4 stage at `p ∈ {1, 4, 16, …}` over `i ∈ [0, N/4)`:

```
k = i mod p;  j = ((i − k) << 2) + k
a_q = src[i + q·N/4],  q = 0..3
b_q = a_q · w^{q·k·N/(4p)}          (three table reads; q = 0 is the identity)
y0 = (b0 + b2) + (b1 + b3)
y2 = (b0 + b2) − (b1 + b3)
y1 = (b0 − b2) ∓ i·(b1 − b3)        (∓i·(x, y) = (±y, ∓x): a swap and a sign, no multiply)
y3 = (b0 − b2) ± i·(b1 − b3)
dst[j + q·p] = y_q
```

— the Stockham indexing whose R = 2 case is the kernel's `((i − k) << 1) + k` today. The
trailing radix-2 stage is the existing butterfly verbatim. `WG` stays `fftWG(N)`: a radix-4
stage has N/4 butterflies against N/2 threads, so half the threads idle in those stages
unless the loop is rewritten to give each thread two butterflies; which of the two the
implementer emits is decided by the bench on the laptop, not up front. Workgroup memory is
unchanged (the same 2N ping-pong).

**B without A** is possible — three inline `cos`/`sin` pairs per radix-4 butterfly, which is
still 25% fewer transcendentals per line than today — and is the form to measure if Phase 1
passed B and failed A. With A landed, the three twiddles are table reads.

**Gates.**
- `devtools/checkfft.js`: a pure-JS model of the emitted schedule — the same `(LOGN → stage
  list)` function the emitter uses, the same index arithmetic, float64 — run against a naive
  DFT for N ∈ {8, 16, 32, 64, 128, 256, 512, 1024}, both directions, ≤ 1e-12 relative. This
  pins the INDEXING, which is the thing a Stockham rewrite gets wrong, and it runs with no
  GPU. The model reads the emitter's stage list, not a copy of it, so a change in one is a
  change in both.
- On-device: the self-test's transform rows (32² exercises r4+r4+r2; 3D's 8 and 16 exercise
  r4+r2 and r4+r4), the Phase 0 analytic row at every production N, the roundtrip row, the
  nonlinear-term and full-step rows — all within their existing tolerances. The
  `exp(L·τ)` row and everything that does not transform are bitwise unchanged (they do not
  go through the FFT) — say so by reading the self-test's `relL2` for those rows as exactly
  what they were.
- `kdiff`: the same six-kernel set as 2A and nothing else.
- Timing: per kernel ≥ 0.4·T_bf gained against the pre-2B page (same session,
  interleaved), whole step accordingly. Short of it: the likely causes are the half-idle
  threads (rewrite the loop) or register pressure on the phone (four complex values and
  three twiddles per thread); find it, then §9 if it still falls short.

**Files.** `common.js` (`fftKernel` only — the schedule function, the radix-4 stage text,
the trailing radix-2), `devtools/checkfft.js`, `devtools/checkbench.js` (capture
regenerated).

## 7. Phase 2C — gradient chunking

**Design.** A chunk is one field's `(x, y)` gradient pair — the F1 granularity that the jax
side measured as free on two of three devices and that keeps `vort`/`jpar` recomputation to
one multiply per chunk. `prepGrads` gains a compile-time pair index `C.gpair ∈ {0, 1, 2, 3}`
(phi, psi, vort, jpar) and writes lanes 0–1 of a 2-lane `gradsK`; four pipelines from one
template, and the `gband` variant follows the same template (the sweep's banded emission is
built lazily from it today). The inverse chain transforms 2 lanes (3D: `encodeInv3D(p, 2,
…)` four times; 2D: the cols/rows dispatches at batch 2); the row kernel's store lands in
`realGrads` at lane offset `2·gpair` through the bind group's buffer `offset` (`2·gpair·nr·4`
bytes — 256-byte aligned for any `nr ≥ 64`, asserted at construction), so the row kernel's
text does not change and every consumer of `realGrads` (bracket, CFL, the 3D field-line and
display passes) sees the eight lanes exactly where they are. 2D: `specTmp` drops to 2 lanes
with it. 3D: the generate path's `encodeInv3D(p, 8, …)` (`rmhd3d.html:2202`) and the
field-line prep go through the same chunked helper — there is one gradient-chain encoder,
used by all three callers.

**Expected: bitwise identical real-space gradients.** Each lane's arithmetic is the same
kernel text on the same inputs, one lane at a time instead of eight at once; nothing in an
FFT line depends on its batch neighbours. Gate: the bench reads back `realGrads` after one
`encodeGrads` on the reference IC before and after, `Uint32Array` equality, both pages, two
resolutions — and the self-test's nonlinear-term and full-step rows read exactly what they
read before.

**Memory and the ceiling** (allocation arithmetic from the sizes, not measured):

| | `gradsK` today | after | `realGrads` | page needs `maxLimits` for |
|---|---|---|---|---|
| 3D 256²×64 | 129 MiB (over the 128 MiB default binding limit) | 32 MiB | 128 MiB (= the default, allowed) | nothing — buildable on a default device |
| 3D 256²×128 (not offered) | 258 MiB | 64.5 MiB | 256 MiB | `realGrads` only |
| 2D 1024² | 32 MiB (+ `specTmp` 32) | 8 (+ 8) | 32 MiB | nothing, as now |

Whether a 256²×128 (or 512²×64) rung is then offered is a separate, §9 question with its own
costs (the volume view, the spectra passes, `realGrads`' own 256 MiB); this phase makes the
gradient stack stop being the reason. [As landed (`createBuffer` sizes): `gradsK` 2D 1024²
32.06 → 8.02 MiB and `specTmp` the same; 3D 128²×64 32.5 → 8.13 MiB; 3D 256²×64
**129.0 → 32.25 MiB**, `realGrads` 134,217,728 B = exactly the 128 MiB default binding
limit (≤, allowed). The page still asks for `maxLimits` as margin — a §9 question, not a
requirement of this phase.]

**Timing gate.** Whole step and the gradient chain at the Phase 1 cells: ≤ 1.03× the pre-2C
page on every device measured (the Phase 1 batch-2 number is the prediction; this is the
measurement of the shipped code). Above it on any device → revert, record.

**Byte identity.** `kdiff`: `prepGrads` changes (four instantiations where there was one);
nothing else. `checkiso`'s "physics WGSL byte-identical to BASE" pin WILL fire on
`prepGrads` — by design, and `dispoffsets.js` is the precedent for an allowance: the pin
keeps its BASE and learns that `prepGrads` is BASE's text plus exactly the pair-index
substitution, so it still fails on any other change. The implementer writes the allowance;
the reviewer mutates `prepGrads` elsewhere to check the pin still bites.

**Files.** `physics.js` (`prepGradsWGSL`), `solver2d.js` (`_buildBuffers`, `_buildPipelines`,
bind groups, `encodeRHS`), `rmhd3d.html` (the same, `encodeGrads`, the generate path),
`devtools/dispoffsets.js` (the allowance), `devtools/checkbench.js` (the readback-equality
leg), `devtools/checkiso.js` / `check2dspec.js` / `checkeigf.js` (consume the allowance).

## 8. Phase order and ownership

| # | phase | files (one owner) | depends on | byte identity |
|---|---|---|---|---|
| 0 | harness + analytic self-test rows | `common.js` (bench + row reference), both pages (panel, rows), `checkbench.js` | — | all kernels identical |
| 1 | campaign | this file (§4 table) | 0, Alfred's devices | — |
| 2A | twiddle table | `common.js` (`fftKernel`/`fftRowPair`), both pages' buffers/bind groups, `checkfft.js` | 1 says A | six FFT kernels differ |
| 2B | radix-4 | `common.js` (`fftKernel`), `checkfft.js` | 1 says B; after 2A if both | six FFT kernels differ |
| 2C | gradient chunking | `physics.js`, both pages' buffers/encoders, `dispoffsets.js` | 1 says C | `prepGrads` differs, with an allowance |
| 3 | docs sweep | `webgpu/README.md`, `devtools/README.md`, `plans-webgpu/README.md`, `AUDIT_2026-08-12.md` item 6 line | all | — |

2C is disjoint from 2A/2B in files except the bench's variant list and can run in parallel
with either once Phase 1 has spoken; 2A and 2B share `fftKernel` and are sequential. Each
phase: commit before dispatching the adversarial review (the reviewer mutates files to test
gates; uncommitted work has been lost that way).

## 9. Decisions for Alfred (raised when reached, with the numbers inline)

1. Phase 1 verdicts where the laptop and a phone disagree in DIRECTION (e.g. ALU-bound on
   the phone, memory-bound on the laptop): which device the demo is optimised for.
2. 2A lands with a gain under 5% of the step and no found cause: ship it as the precision
   improvement it is, or revert to keep the kernel text as it was.
3. 2C passes its memory gate but measures 1.01–1.03× on one device: the ceiling against the
   1–3%, per device.
4. After 2C: whether to offer a 256²×128 / 512²×64 rung, with the other costs listed.
5. Phase 1 confirms the audit's closure (`T_mem ≥ 70%` everywhere): whether the column-stride
   route stays closed — the plan's default is yes, this file records the numbers and stops.

## 10. Docs to update (overseer owns the text; implementers draft to scratch)

- `webgpu/README.md`: a "Benchmark (`?bench`)" paragraph under "Run"/"Self-test" — what it
  measures, how to read the three ladder numbers, the copy-paste transport; the new
  self-test rows in the "Self-test" list; the FFT kernel description wherever it says
  radix-2 / recomputed twiddles (the `fftKernel` comment block in `common.js` is the one
  source comment that says WHAT the kernel is and gets the one-line update).
- `devtools/README.md`: `checkbench.js`, `checkfft.js`, the `dispoffsets` allowance.
- `plans-webgpu/README.md`: this file under Live, then under Executed; the AUDIT line's
  "item 6 (twiddle table) is dead on the LOOPLAT evidence" becomes whatever Phase 1 found.
- `AUDIT_2026-08-12.md`: one line at item 6 pointing here.
- `plans/MEMORY_PERF_PLAN.md` is not touched; `docs/performance.md` is the solver's.

## Appendix A — bytes per step, for the bench's GB/s (checkable by hand)

The count is **buffer bytes bound and read or written by each dispatch** (not cache traffic,
not the CFL pair, not `tick`/`ou`/`scale`/`energyFinal`, not the `clearBuffer(delta)`). Units: `cx = nm·8`
(one complex field), `rx = nr·4` (one real field); the static grids are `grA = grB = nm·16`
in 2D and `nmp·16` in 3D (`gridA`/`gridB` are perpendicular-only there), `grZ = nz·16`
(3D only). Per stage, with lane counts as the step dispatches them:

| kernel | 2D | 3D |
|---|---|---|
| `prepGrads` ×4 (2C: one field pair each) | read `cx + grA`, write `2cx` (each) | same |
| `zInv` ×4 at 2 lanes | — | `4cx` (each) = `16cx` |
| `colsInv` ×4 at 2 lanes | `4cx` (each) = `16cx` | same |
| `rowsC2R` ×4 at 2 lanes | read `2cx`, write `2rx` (each) = `8cx + 8rx` | same |
| `bracket` | read `8rx`, write `2rx` | same |
| `rowsR2C` ×2 | read `2rx`, write `2cx` | same |
| `colsFwd` ×2 | `4cx` | `4cx` |
| `zFwd` ×2 | — | `4cx` |
| `nlAssemble` | read `2cx + grA + grB`, write `2cx` | read `2cx + grA + grB + grZ`, write `2cx` |
| `forcingAdd` | read `frc 2cx + rhs 2cx`, write `rhs 2cx` = `6cx` | `frc` 4 arrays of `nmp·8`, plus `rhs` on two kz planes for both fields read and written: `32·nmp + 64·nmp` |
| `stage` | read `fields 2cx + delta 2cx + rhs 2cx + grB`, write `fields 2cx + delta 2cx` = `10cx + grB` | `10cx + grB + grZ` |

Per step: three stages plus `energyPartial` (read `2cx + grA + grB` in 2D, `+ grZ` in 3D).
Before 2C the gradient chain was one `prepGrads` (read `2cx + grA`, write `8cx`) and one
8-lane pass per transform — the same transform bytes, and `prepGrads` at `10cx + grA` per
stage against 2C's `4(3cx + grA)`: every chunk re-reads the grid and reads one state field.
The two hand numbers `checkbench.js` leg (iv) pins, pre-2C (Phase 0/1) and as landed (2C):

- **2D 256²**: `nm = 33,024`, `nr = 65,536`, `cx = 264,192`, `rx = 262,144`, `grA = 528,384`.
  Per stage 3,170,304 + 4,227,072 + 4,210,688 + 2,621,440 + 1,052,672 + 1,056,768 +
  2,113,536 + 1,585,152 + 3,170,304 = 23,207,936; ×3 = 69,623,808; + `energyPartial`
  1,585,152 = **71,208,960 B/step** pre-2C. Butterflies: 11,827,200. 2C: `prepGrads`
  4 × 1,320,960 = 5,283,840 per stage in place of 3,170,304 → **77,549,568 B/step**
  (+6,340,608, all `prepGrads`).
- **3D 128²×64**: `nmp = 8,320`, `nm = 532,480`, `nr = 1,048,576`, `cx = 4,259,840`,
  `rx = 4,194,304`, `grA = 133,120`, `grZ = 1,024`. Per stage 42,731,520 + 68,157,440 +
  68,157,440 + 67,633,152 + 41,943,040 + 16,908,288 + 17,039,360 + 17,039,360 + 17,306,624 +
  798,720 + 42,732,544 = 400,447,488; ×3 = 1,201,342,464; + `energyPartial` 8,786,944 =
  **1,210,129,408 B/step** pre-2C. Butterflies: 213,934,080. 2C: `prepGrads` 4 × 12,912,640
  = 51,650,560 per stage in place of 42,731,520 → **1,236,886,528 B/step** (+26,757,120).

The bench sums the same table from the page's own dispatch list so a kernel added later is
counted; a disagreement between the check's literal and the function is a change to one of
them that the other did not follow.

## Phase 0 execution notes (2026-08-21)

Two opus implementers in the shared tree (A: harness, probe seam, `checkbench.js`; B:
`fftAnalyticCase` and the production-N rows), fresh-Fable adversarial review, one fix round,
fresh-Fable close-out. Landed as `58c3fd7` and `4972e0e`. WGSL byte-identical to `f83386e`
on both pages (124/124, 270/270) throughout; the whole devtools suite green; `checkbench.js`
135 legs.

**Deviations from §3, accepted:**
- `buildShaders` did not gain an `opts` argument: `checksolver2d.js` pins its header and
  `solver2d.js`'s top-level names against `268100f`. The probe reaches the emitter through
  `FFT_PROBE`, a module-level seam in `common.js` set only inside `benchShaders` (try/finally)
  — §0.1's "module-level emit options that only the bench reads". The gate now asserts it is
  null on a freshly booted page before any campaign, and pins the Solver's own compiled FFT
  text against the `f83386e` fixture.
- The whole-step cell is K `solver.step` submits with one drain (§3 item 1, as amended): an
  encode-only step would be a second copy of `step`.
- The bench spec is a page-level function (`benchSpec2D`/`benchSpec3D`) next to the solver,
  not a `Solver` method (the same pin).
- The analytic rows run on the LIVE solver (`realNL`/`nlk` are RHS scratch) — a throwaway
  solver at 256²×64 would double the page's largest allocation.
- Appendix A was rewritten as the per-kernel table after the first review found the
  implementers' byte count and the appendix's prose both short of what the kernels bind
  (`gridA`/`gridB`/`gridZ` reads, 2D `forcingAdd` = 6cx); the fix round then caught that the
  reviewer and the appendix had `stage` at 8cx where the kernel reads three fields and writes
  two (10cx). Three people, three numbers; the table is now the single statement and the
  check's two literals follow from it: 71,208,960 (2D 256²), 1,210,129,408 (3D 128²×64).

**What the first review found and the fix round closed:** the `FFT_PROBE` assertion ran after
the bench's own `finally` had reset it (toothless; now asserted on a fresh boot); nothing pinned
the spec's dispatch shapes/buffers to the solver's (the stub only rejected non-integers — it now
logs every dispatch and bind group, and `legWiring` compares against a recorded `step`); the
whole-step cell let the rAF loop render and read back inside the timed window (`benchBusy`
holds `loop()` off; gated with a parked-rAF stub); `running = false` → `setRunning(false)`;
first-rep discard, `finally`, and the self-test's mid-test solver rebuild are gated.

**What the close-out review found and the second fix round closed:** the `loop()` iteration
already in flight at the click finished its tail (`readStats`, the 3D hooks) inside a kept
rep — `benchGo` now waits for it; 1024² and 64²×128/256 were outside every byte-identity
gate (`dumpwgsl2`'s preset list predated them — added, baselines regenerated from `f83386e`,
the largest grids pinned in the fixture); a deadlocked check exited 0 without a summary
(`process.exitCode`); `benchRestore` re-applied the hard-coded "modes" IC, not the page's
selected one (now `applyIC`). Three independent recomputations of the byte table now agree.

**Only checkable on device** (handed to Alfred before Phase 1): the two new self-test rows
green on both pages at the default and largest resolutions; the `?bench` panel renders and
a campaign completes with no WebGPU validation error (the variant pipelines' `auto` layouts
and the hand-listed `bufs` get their only real validation there); per kernel
`copy ≤ consttw ≤ full`; whole-step `ms_med` stable across two runs.

## Phase 2C execution notes (2026-08-21)

One opus implementer, two steps: the `grads hash` bench cell first (`6831d53`, WGSL
byte-identical — the bitwise gate's instrument, so Alfred can record the eight-lane digests
on the pre-2C page), then the chunking (`fb30cab` + review fixes). Fresh-Fable adversarial
review, one fix round.

**As landed.** `prepGradsWGSL(C)` takes `C.gpair ∈ {0..3}` off a `GRAD_PAIRS` table; each
kernel reads ONE state field (phi for pairs 0 and 2, psi for 1 and 3), forms the pair's
field where it is derived, writes lanes 0–1 of a 2-lane `gradsK` — four pipelines from one
template, the `gband` variant through the same template. The row kernel's text is unchanged;
its store lands in `realGrads` through the bind group's buffer `offset` (`gradPairOffset`,
256-byte alignment thrown at construction). One chain encoder per page (`encodeGrads`) used
by the RHS, the 3D field-line prep and the generate sweep. 2D `specTmp` drops to 2 lanes
with `gradsK`. The `checksolver2d` `class Solver` pin gained a second recorded-replacement
block (the IO_PLAN idiom, generalised to `stripBlocks`); the `prepGrads` byte pins
(`checkiso`/`checkeigf`/`check2dspec`) keep their BASE commits through a `dispoffsets.js`
allowance that reduces BASE's eight-lane text to each pair mechanically.

**What the review found and the fix round closed.** The lane offsets — the whole mechanism —
were invisible to every gate (the stub kept only an entry's buffer): `gradPairOffset` halved,
or every pair bound at offset 0, stayed green. The stub now keeps `{buffer, offset, size}`
and throws on misalignment or overrun as WebGPU does, and `checkbench` asserts the chain's
four row-kernel targets at `[0, 2nr4, 4nr4, 6nr4]` in pair order behind four distinct
`prepGrads0..3` pipelines. The allowance had no negative test (a permissive edit to
`gpairApplied` retired all four pins silently) — `checkiso` leg 2b now feeds it six wrong
pair sets and the correct one. The "one encoder" claim is pinned by `check2dspec` leg (f)
(each band pass = 4 × (prepGradsBand_k, zInv, colsInv, rowsC2R@2) + march). Bench
semantics: the per-kernel `prepGrads`/`colsInv`/`rowsC2R` cells now time one two-lane chunk
(×12 per step) and need ×4 before comparison with the Phase 1 rows.

**Byte identity.** Every FFT kernel and everything else byte-identical to `f83386e` at every
offered grid; the differing set is exactly `prepGrads` (gone) and `prepGrads0..3` (+ the
`Band` four on 3D). Bytes per step as in Appendix A (77,549,568 / 1,236,886,528).

**Only checkable on device** (the phase's two gates, Alfred): the `grads hash` records from
the `6831d53` page and the chunked page equal in all eight `hash_lane` and `hash_all`, both
pages, two resolutions each, same IC; whole step and the `gradient chain` cell ≤ 1.03× the
Phase 1 record; the 3D generate sweep and field-line view clean of validation errors
(the offset bind groups' only real validation); self-test rows unchanged.

**Pre-2C `grads hash` record (laptop `apple metal-3`, page `6831d53`, default IC, 2026-08-21):**

| page, grid | bytes | hash_lane[0..7] | hash_all |
|---|---|---|---|
| 2D 256² | 2,097,152 | 482638133 2368582853 4180176569 1854853189 3807947513 3579484697 676813825 1675221881 | 4061148817 |
| 2D 1024² | 33,554,432 | 2504137769 3206017749 3678507661 495144193 1079874917 2578950449 649279321 1275090241 | 58098025 |
| 3D 128²×64 | 33,554,432 | 1935610801 3236346983 698820014 889683997 1981461617 2571723835 2451374745 1792438921 | 1028719198 |
| 3D 256²×64 | 134,217,728 | 318975641 781519921 710674129 2467220641 1396472425 3371760281 1795061589 2403523413 | 4222161269 |

The chunked page (`cc50ec4`) must reproduce every entry of this table.

**Bitwise gate PASSED (2026-08-22):** the chunked page (`07ea844`, same laptop, same IC)
reproduced all 32 lane digests and the four `hash_all` values exactly — compared by script
against the table above, not by eye. Real-space gradients are bit-identical to the eight-lane
chain on both pages at both grids. Timing gate pending (whole step and `gradient chain` vs
the Phase 1 record).
