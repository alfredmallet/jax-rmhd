# FFTPERF_PLAN — what the step's transforms cost, then twiddle table / radix-4 / gradient chunking

Written 2026-08-21, **not started**. Prompted by `plans/MEMORY_PERF_PLAN.md` closing on the
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

1. *Whole step*: encode `K` steps (`solver.step`, `cflEvery` = the page's setting, display
   off) in one submit, `await onSubmittedWorkDone()`, median of `R` reps, ms/step. Plus the
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
     `ws = 0.0` in place of `cos`/`sin` (the multiply stays, so only the transcendentals go).
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
| batch-2 gradient chain ≤ 1.03× the batch-8 time | **C goes ahead** (§7) on both pages. Above 1.03× on any device: C is dropped; the memory it would save is recorded. |

Both A and B can be true; then A lands first (the table is what radix-4's three twiddles
index) and B is measured against post-A.

**Deliverable.** A table in this file — one row per (device, page, resolution, kernel), the
three ladder numbers and the shares — pasted from the bench's JSON. Nothing in Phase 2 is
briefed until it exists.

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
gradient stack stop being the reason.

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

`cx = nm·8` (one complex field), `rx = nr·4` (one real field). Per 2D transform of `b` lanes:
rows real→complex read `b·rx`, write `b·cx`; cols read and write `b·cx` each; inverse the
mirror. Per stage (2D): `prepGrads` read `2cx` write `8cx`; inverse 8 lanes `8·(2cx) +
8·(cx + rx)`; `bracket` read `8rx` write `2rx`; forward 2 lanes `2·(rx + cx) + 2·(2cx)`;
`nlAssemble` read `2cx + 2·nm·16` write `2cx`; `forcingAdd` read/write ~`4cx`; `stage` read
`4cx + nm·16` write `4cx`. Per step: 3 stages + `energyPartial` read `2cx + nm·16` + the
small kernels. 3D adds the z pass (`2·b·cx` per transform) and reads `gridZ` instead of a
full-grid `gridB` in `stage`. The bench sums this from the page's own dispatch list so a
kernel added later is counted; `checkbench.js` leg (iv) pins the sum for two grids against
the numbers this formula gives by hand.
