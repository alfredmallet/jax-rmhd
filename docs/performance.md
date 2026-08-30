# Performance and scaling

Measured numbers, the tuning knobs they justify, and the negative results — so nobody
re-runs a cluster job to rediscover an answer. All figures from Savio (account
fc_kawturb), 2026-07-25 → 2026-07-27. Setup instructions live in
`docs/SAVIO_CPU_SETUP.md` / `docs/SAVIO_GPU_SETUP.md`; the reasoning behind the numerics
is in `docs/numerics.md`.

## Read this first: fp32 flatters communication savings

Every accept/revert decision in this code was made on **fp64** numbers, because fp32
halves message sizes and therefore roughly triples the apparent benefit of any
communication optimization. Measured example: per-step forcing normalization was +27% at
fp32 and +8% at fp64 on the same job. Benchmark at the precision you will actually run.

## Architecture: what sets the ceiling

The decomposition is **z-only**. The perpendicular plane is never distributed — the
`rfft2` is process-local on every rank — so:

- the maximum useful rank count is about `nz/2` (the halo is 2 planes wide);
- a 2D run has no parallelism to express at all, which is why `dims=2` is single-process
  and `comm_backend="jax"` rejects it at construction;
- going further would need a pencil decomposition and distributed FFTs, which is a
  different architecture, not a tuning change.

## Backends: why two

Three exist, but only two are production: `"serial"` is the single-process fallback
(auto-selected with no MPI installed — exact size-1 semantics, no collectives, no
tokens). Its performance has **not** been measured against size-1 `mpi4jax`; expect a
small win from the absent token-ordering constraints, and note it is not bitwise-identical
to size-1 `mpi4jax` (dropping the mpi4jax ops changes XLA fusion — the same class of
difference as `lsrk_scan`, so compare with tolerances, not equality).

`mpi4jax` is the CPU-cluster backend and `"jax"` (shard_map/NCCL) is the GPU one. This is
not arbitrary:

- XLA:CPU collectives are slow; shard_map was tried on CPU and performed badly.
- On GPU, mpi4jax has three specific costs: without CUDA-aware MPI every transfer stages
  through host memory; each mpi4jax op is an XLA custom call that forces a CUDA stream
  sync (negligible on CPU, a pipeline stall on GPU, ×10–20 per step); and the token chain
  is opaque to XLA, so there is no compute/comm overlap.

Savio's MPI is **not** CUDA-aware (built `--without-cuda`; forcing `MPI4JAX_USE_CUDA_MPI=1`
segfaults in UCX), so mpi4jax-on-GPU is a fallback or single-GPU option only.

## CPU (savio3, mpi4jax, fp64)

Baseline, 32 ranks, 128²×256, lsrk54 + elsasser forcing + adaptive dt, production config:

| | fp32 | fp64 |
|---|---|---|
| unforced | 153 ms/step | 302 ms/step |
| forced | 174 | 353 |

One allreduce costs ~4 ms at 32 ranks and ~22 ms at 128 — which is why the
communication knobs below only pay at high rank counts.

Benchmark caveat (found during the 2026-08 precision work): before fp64-t landed,
`initialize` returned a weak-typed `t`, so the SECOND jitted `block_of_steps` call
retraced (weak → strong scalar), and `bench_phase1.py`'s single warmup call left that
retrace inside the timed region — historical numbers taken with small `nrep` include
one recompile (measured up to ~2× at 64²×16 serial/CPU with `nrep=4`). Post-fp64-t the
carry is stable from the first call (steady-state per-step time is unchanged). When
comparing against pre-2026-08 numbers, warm up twice or use a large `nrep`.

Scaling, 256²×256 strong: 1331 / 1363 / 762 / 434 ms/step at 16 / 32 / 64 / 128 ranks.
Within a node the cores saturate memory bandwidth by about 16 (32 ranks is no faster than
16); across nodes it holds ~88% per doubling. Weak scaling at 256²×4 per rank: 370 / 750 /
762 / 771 — flat from 1 to 4 nodes (97%), which is the production regime.

## GPU

Single node, 4×A5000, fp32, 512²×128, forced:

| GPUs | mpi4jax | jax/NCCL | advantage |
|---|---|---|---|
| 1 | 294 ms/step | 310 | −5% |
| 2 | 176 | 158 | +11% |
| 4 | 126 | 77.0 | +63% |

Scaling 1→4: jax ~4.0× (ideal), mpi4jax ~2.3×. The −5% at one GPU is fixed
shard_map/global-array overhead and disappears as soon as there is communication to do.

Multi-node, 16×GTX2080Ti over 4 nodes, fp32, 512²×256:

| GPUs (nodes) | mpi4jax | jax/NCCL | advantage |
|---|---|---|---|
| 4 (1) | 284 ms/step | 200 | 1.4× |
| 8 (2) | 204 | 102 | 2.0× |
| 16 (4) | 165 | 48 | 3.4× |

Scaling 4→16: jax 4.15×, mpi4jax 1.72×. The jax curve holds **across node boundaries on
plain TCP** — no InfiniBand userspace, PCIe peer-to-peer disabled. On NVLink/IB hardware
the margin can only grow.

### XLA latency-hiding scheduler (16×GTX2080Ti, 2026-08-21, job 37912751)

`slurms/bench_xla_flags_2080.sh` (probe → HLO audit → timed matrix → profile) with
`bench/hlo_audit.py`. 512²×256 fp32, `nps cfl1`, `backend=jax`, lsrk54, `halo_start` at its
jax default (on), 2 passes × 120 timed steps per case. jax 0.10.2. The whole matrix — 28
`srun` steps — cost 16 minutes and ~32 SU; compile is ~12 s per case, so a flag matrix is
cheap even with the compilation cache deliberately off (each flagset must compile its own
binary or the experiment is meaningless).

**Static, from the optimized+scheduled HLO.** Collectives are async (`-start`/`-done`)
even with no flags — but at base the scheduler emits the `-done` immediately after the
`-start`:

| flagset | async pairs (scan) | overlap window, median instrs | temp (scan) |
|---|---|---|---|
| base (no flags) | 6 | **0** | 219 MB (6.8 u) |
| `lhs` = `--xla_gpu_enable_latency_hiding_scheduler=true` | 6 | **23** | 227 MB (7.1 u) |
| `lhsdec` = `lhs` + `--xla_gpu_collective_permute_decomposer_threshold=0` | 6 | 23 | 227 MB (7.1 u) |

`--xla_gpu_enable_pipelined_collectives` no longer exists in this XLA build and was dropped
by the script's phase-0 probe. `lhsdec` produced HLO byte-identical to `lhs`, which makes it
a free control rather than a third data point.

**Dynamic**, mean ms/step over both passes:

| 16 GPUs (4 nodes) | base | `lhs` | `lhsdec` | `lhs` vs base |
|---|---|---|---|---|
| scan | 60.46 | 46.22 | 45.31 | **1.31×** |
| unrolled | 65.95 | 54.79 | 57.04 | **1.20×** |

| 4 GPUs (1 node) | base | `lhs` | `lhs` vs base |
|---|---|---|---|
| scan | 145.54 | 142.24 | 1.02× |
| unrolled | 172.82 | 168.15 | 1.03× |

**The noise floor is measured, not assumed:** `lhs` and `lhsdec` compile to identical HLO,
so their timing gap is pure run-to-run spread — 2.0% (scan), 4.1% (unrolled), with
pass-to-pass spread inside one config ≤3.2%. The 16-GPU win clears that by ~6×. The 4-GPU
repeats are tighter (<0.3%), so even 2.3% there is real, just small. The win scales with
communication, which is what the overlap story predicts: ~2% single-node where there is
bandwidth to spare, ~24% across four nodes on plain TCP.

**What this says about `halo_start`.** The hook is enabled by default on the jax backend, so
every base row above already pre-issues the halo exchange — and the schedule still leaves a
zero-instruction window. That is why it "buys nothing measurable": not a physics result, a
scheduling one. Under `lhs` there is finally a window for the pre-issued exchange to fill,
so the `halo_early`/`halo_late` A/B is worth running again. It has not been.

**What this says about unrolling.** `lsrk_scan=False` loses at every GPU count here — it
costs 18.7% more time per step (4 GPUs, base), 18.2% (4, `lhs`), 18.5% (16, `lhs`), and
9.1% at 16 GPUs base, the one outlier. The consistent ~18% is unroll's extra compute; at 16 GPUs base the step is
comm-bound, so the stall *hides* half the penalty. Latency hiding does not flip the sign, it
unmasks it. The audit agrees: unrolling exposes 12 async collectives instead of 6, but each
gets a worse window (median 6 vs 23). The earlier −38% at this same configuration did not
reproduce.

**One number not to chain.** Base here is 60.46 ms/step where the scaling table above records
48 for what is nominally the same case (`j16`: 512²×256 fp32, `nps cfl1`, backend=jax,
16×2080Ti). Different job, different allocation, different nodes — not a controlled
comparison, and it is not called a regression on this evidence. The base-vs-`lhs` delta *is*
controlled (one job, one allocation) and stands on its own.

### Cost per timestep (fp32, 512²×128)

| Hardware | ms/step | SU/hr | SU per 1000 steps |
|---|---|---|---|
| 1 savio3 node (32c) | 1563 | 32 | 13.9 |
| 2 nodes | 866 | 64 | 15.4 |
| 1 × A5000 | 294 | 18.7 | 1.53 |
| 4 × A5000, mpi4jax | 126 | 74.7 | 2.61 |
| 4 × A5000, jax | 77 | 74.7 | 1.60 |

CPU cost per step *rises* under strong scaling — waiting cores still bill — while the jax
backend holds it flat while quadrupling throughput. Roughly **9× cheaper per timestep on
GPU at fp32**. Savio's workstation GPUs run fp64 at 1/32 rate, so fp64 production stays on
CPU there; on full-rate-fp64 hardware (A100/GH200/H100) the economics carry over.

## Tuning knobs, measured

| Knob | Effect | Guidance |
|---|---|---|
| `forcing_norm_per_step` | +8% at fp64/32 ranks | default on |
| `cfl_every` | +1.3% (N=20) at 32 ranks, +8.9% at 128 | 10–20 at ≥128 ranks, **developed states only** |
| `lsrk_scan=True` (scan) | ~20% faster than unrolled on CPU | default |
| `lsrk_scan=False` (unrolled) | +21% mpi4jax-GPU, +12% jax single-node; on the **jax** backend a loss at every GPU count re-measured 2026-08-21 (−9% at 16 GPUs base, −18% at 4, −18% at 16 under latency hiding; 1.10× on the P100) | per-machine knob, benchmark it; scan is the jax default |
| `forcing_shell_noise` | faster single-device, ~5% *slower* on Savio CPU at 32 ranks | opt-in; revisit on GPU |
| `halo_start` | neutral (≤2%, sub-noise) everywhere measured — but every jax measurement predates the latency-hiding flag, under which the overlap window it feeds was structurally zero | see below; worth one re-test under `lhs` |
| `hoist_propagator` | **putzer2 only** — 1.22× (P100) / 1.80× (CPU) slower off on GDI-IF; scheme-dependent memory, see below | default on; turn off only on a memory-bound putzer2 grid |
| `XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true` | **1.31× at 16 GPUs** (jax/NCCL, 4 nodes), 1.02× at 4 GPUs on one node | set it on every multi-GPU jax run — see below |
| `GRAD_CHUNK` (module constant, not a `Parameters` attribute) | chunk 2 costs 1.25–1.54× the step on XLA:CPU and gains only 1.8–3.3% on the 2080Ti, but gains 3–20% on the P100 | default 1 everywhere; the one per-card tuning worth knowing |

**`hoist_propagator` after Z1/Z2 (2026-08-20).** It only reaches the putzer2 backend now:
FD-z and 2D use the diagonal backend, which is never hoisted, and z_spectral RMHD at ν = η
uses the separable backend, whose whole stage stack is ≤0.1 u (hoisted unconditionally in
effect — there is no trade to make). On putzer2 the memory cost is **scheme-dependent**: at
lsrk54 the hoisted/unhoisted gap is ~8 u (44.4 vs 36.0 at 128²×32 ν≠η z_spectral; 45.2 vs
38.2 on the P100's GDI 2D 1024²), at lsrk33 it has collapsed to under 1 u because Z2's
`w`-form removed the full-grid `sqrt`/`cosh`/`sinh` temporaries the unhoisted stage used to
carry. And it is a **scheduler** property, not an arithmetic one: on XLA:CPU the GDI 2D 256²
lsrk33 pair is memory-*neutral* (41.130 u hoisted, 41.091 u unhoisted) where the same pair
on the P100 costs a real +7.0 u. Size it on the device you will run on. On adaptive
`cfl_every=1` nothing is frozen, so the knob has no effect at all.

`cfl_every > 1` costs one extra standalone gradient evaluation per block, because the
stage-0 RHS no longer doubles as the dt source. At 32 ranks that cancels the saving at
N=5; it only pays once the allreduce is expensive.

**The `cfl_every` hazard is real:** from a quiescent forced start the CFL dt collapses by
~10× within a few steps of spin-up, so a frozen dt NaNs — measured, N=20 dies by t≈2 while
N=5 survives. Use N>1 only from developed states.

## Negative results

Recorded so they are not re-investigated.

- **Early halo issue (`halo_start`) buys nothing measurable — and on the jax backend the
  reason is now known.** On mpi4jax there is no fp64 win at any rank count (−0.6% at 32,
  sub-noise at 128) because the token chain serializes communication with compute
  regardless. On the jax backend it is neutral on Savio at bench sizes, including
  multi-node NCCL — but the 2026-08-21 HLO audit shows why: without the latency-hiding
  scheduler XLA leaves a **zero-instruction** window between each collective's `-start` and
  `-done`, so pre-issuing the exchange cannot help by construction. The hook is kept and
  enabled for `"jax"`. **This entry is not settled under `lhs`**, where the window is a
  median of 23 instructions; that A/B has not been run.
- **Unrolled LSRK is not a universal win.** It helps mpi4jax-GPU (+21%) and jax
  single-node on the A5000 (+12%). On the **jax** backend on the 2080Ti it loses at every
  GPU count (2026-08-21: +9% at 16 GPUs base, +18% at 4 and at 16 under `lhs`), and the
  P100 probe has it at 1.10× for FD-z lsrk54. The earlier −38% at 16 GPUs did not
  reproduce; it is now +9%, and the flag that removes the comm stall widens unroll's loss
  rather than reversing it.
- **shard_map on CPU** — measured slow, which is why `mpi4jax` remains the CPU backend.

## Test particles overhead (laptop)

### 2D (A2/A3, 2026-08-18)

`bench/particles_overhead.py`: `jax.jit(block_of_steps)` at 256², `Lx=Ly=2π`, fp64,
lsrk33, fixed `dt=1e-3` (`adaptive_timestep=False`, for determinism), elsasser forcing
(`forcing_power_elsasser=(1,1)`, `eqpars={"diss":(1e-4,1e-4),"hyper":2}`). Particles:
`n=32768` per ensemble, `qm=15`, `vth=1` (plan §2 baseline). Apple M1, macOS 14 (Darwin
23.6.0), jax 0.10.0, CPU backend, quiet machine.

| case | ms/step | overhead vs off |
|---|---|---|
| off | 13.2 | — |
| on, 1 ensemble (default mask), 32768 particles | 16.65 | +26% |
| on, 3 ensembles (ideal, full ∂ψ/∂t, E=0 control), 32768 each | 22.9 | +74% |

Breakdown, fixed (transforms) vs O(N) (gather/push):

| part | cost | share of solver step | scaling |
|---|---|---|---|
| `particle_fields` (4 gradient iffts + fft/ifft pair for the dealiased ideal E_z; optional resistive/forcing iffts add ~nothing measurable) | 2.24 ms | 17% | grid-like, same as the solver's own transforms |
| `boris.push` per ensemble of 32768 (4 bilinear gathers + 2 kicks) | 1–2.5 ms | — | O(N), resolution-independent per particle |

Verdict: the fixed part (`particle_fields`) lands INSIDE the plan's ≤15–20% budget
(plans/TESTPART_PLAN.md §4) — that budget was an FFT-count estimate of exactly this
piece. The O(N) push is what exceeds it: at 256² with 32768/ensemble the loading is 0.5
particles per grid point, dense, and XLA's CPU gather is a scalar loop. In 3D at
production sizes (e.g. 256²×128 with ~1e5 particles) the push is ~1% and only the fixed
transform part matters; on GPU the gather is cheap.

**Re-measured after A3 (2026-08-18, same machine and script, `nrep=25`).** A3 replaced the
two gathers per half-kick (E, then B) with a single gather of `assemble_stacked`'s combined
array, and added the per-piece work accumulation. Net effect on the total: nothing
measurable.

| case | ms/step (A2) | ms/step (A3) | overhead vs off |
|---|---|---|---|
| off | 13.2 | 12.9 | — |
| on, 1 ensemble (default mask), 32768 particles | 16.65 | 16.8 | +30% |
| on, 3 ensembles (ideal, full ∂ψ/∂t, E=0 control), 32768 each | 22.9 | 22.8 | +76% |

(The A2-vs-A3 overhead percentages differ mostly because `off` benchmarked ~2% faster in
the A3 run; the particle-on absolute times are unchanged within the run-to-run IQR, ~1–2
ms/step.) Isolated jitted pieces at the same size: `particle_fields` 2.16 ms with the
default mask, 2.50 ms with the resistive and forcing pieces on (17% / 19% of the solver
step, still inside the plan's budget); `boris.push_tracked` for one 32768-particle ensemble
2.30 ms at the default mask (6 gathered components) and 5.23 ms at the full mask (8
components — the three E_z pieces are gathered separately so their work can be attributed,
where A2 gathered one pre-summed E_z). So sharing the cell/weight work across E and B
bought back roughly what keeping the E_z pieces separate costs, and the O(N) push remains
the part that exceeds the budget at this 2D loading.

Deferred (Alfred, 2026-08-18): gather optimization would attack only the O(N) part
(est. 1.5–2×) — revisit if A3's 2D science runs at 3×32768 particles feel slow. Two
items flagged: (i) gather-side reorganization (`interp.gather`: share cell/weight
computation across E and B — done in A3, see above, no measurable win; `jnp.take` on flat
indices, cast samples not grids — still open); (ii)
reuse the stepper's stage-1 gradients in `particle_fields` (removes 4 of the 6 fixed
transforms) — the bigger lever, but changes the stepper contract, so not Phase A.

### Test particles overhead in 3D (B2, 2026-08-19)

Same script and machine (`bench/particles_overhead.py`, now taking a case argument:
`2d` | `3dfd` | `3dspec`; Apple M1, macOS 14 (Darwin 23.6.0), jax 0.10.0, CPU backend,
fp64, quiet machine, `nblock=20`, `nrep=12`). 3D runs **128²×16** — a smaller perpendicular
grid than the 2D case so one configuration fits a laptop — with everything else identical
(`dt=1e-3` fixed, lsrk33, elsasser forcing, `eqpars={"diss":(1e-4,1e-4),"hyper":2}`,
`n=32768` particles per ensemble, `qm=15`, `vth=1`). What transfers between grids is the
RATIO to the solver step, not the ms.

| case | finite-difference z, ms/step | z_spectral, ms/step |
|---|---|---|
| off | 42.2 | 75.8 |
| on, 1 ensemble (default mask) | 53.9 (**+27.7%**) | 97.9 (**+29.1%**) |
| on, 3 ensembles (ideal, full ∂ψ/∂t, E=0 control) | 74.3 (**+76.2%**) | 125.0 (**+65.0%**) |

Isolated jitted `particle_fields` at the same grid:

| mask | finite-difference z | z_spectral |
|---|---|---|
| default (4 gradient iffts + the dealiased-E_z fft/ifft pair) | 7.36 ms = **17.4%** of the solver step | 9.76 ms = **12.9%** |
| + resistive and forcing pieces | 9.66 ms = **22.9%** | 10.94 ms = **14.4%** |

**The finite-difference-z filter, timed for the first time** (B1 folded `−z_diss·(dz/2)⁴∂_z⁴ψ`
into `ez_resistive`; it costs a 4th-order z stencil and a `comms.halo_exchange`, which no
earlier measurement covered). Isolated at 128²×16 in a separate timing session, so compare
these two with each other and not with the table above: the resistive piece as the k-local ψ
diagonal alone is **1.50 ms**, and `fields._psi_non_ideal` with the filter is **3.36 ms** — the
stencil plus halo is **1.86 ms**, about **4.4%** of the 42.2 ms solver step. That is the whole
gap between the FD-z and z_spectral "+ resistive and forcing" rows, and it is why FD-z with the
optional pieces on is the one configuration that leaves the plan's ≤15–20% budget
(plans/TESTPART_PLAN.md §4) at 22.9%. It buys exactness: without it, full-mask E_z misses
∂ψ/∂t by 1.2e-3 of max|E_z| and gate 7 stops converging (`tests/test_particles_3d.py`).

Two more observations:

- **z_spectral looks cheaper only because its solver step is nearly twice as expensive**:
  `particle_fields` costs about the same absolute time in both modes, so its *share* halves.
  Nothing about the particle path is faster there. (The parenthetical this bullet first
  carried — "rfftn/irfftn over (z,x,y) against one rfft2 per plane" — was a guess, and the
  profile below shows it is wrong: the transforms account for ~3 ms of the ~38 ms gap.)
- **The trilinear gather roughly doubles the O(N) push cost per particle**, as its 8 corners
  against 4 predict: subtracting `particle_fields` from the one-ensemble overhead gives ≈1.9 ms
  in 2D at 32768 particles and ≈4.3 ms in 3D. It is still the part that exceeds the budget at
  this dense loading (0.5 particles per grid point in 2D, 0.125 in 3D); at production 3D sizes
  with ~1e5 particles it is a few percent, and only the fixed transform part matters.

(2D re-measured in the same session for comparability: off 12.65, on1 16.35 (+29.2%), on3 22.64
(+79.0%) ms/step; `particle_fields` 1.80 ms = 14.2% and 2.10 ms = 16.6% — the A3 numbers within
run-to-run scatter.)

## CMHD: the compressible step against z_spectral RMHD (C2, 2026-08-30)

`bench/cmhd_perf.py`, Apple M1, macOS 14.6 (Darwin 23.6.0), jax 0.10.0, CPU backend,
fp64, `comm_backend="serial"`, quiet machine (bench lock held). Same-session **interleaved
A/B**: one timed `nblock=10` call of each equation set per repetition, alternating, 9
repetitions, median ms/step. Both sides are the jitted `run.block_of_steps` at fixed
`dt=1e-3` with **lsrk54**, `Lx=Ly=Lz=2π`, `z_spectral=True`, `hyper=2`, default
`hoist_propagator=True`.

The reference is **z_spectral RMHD at ν = η** (`diss=(1e-4,1e-4)`, `z_diss_k=1e-6`), which
resolves to the **separable** backend (`SeparableL`) — the z_spectral production path, and
hoisted. CMHD (`cs0=1`, `diss=1e-4`, `gamma=1`) resolves to **`DiagonalOperator`**, which
is deliberately unhoistable (CMHD_PLAN §3.1: a broadcast diagonal exp is cheap and making
it hoistable would touch the FD-z/2D bitwise gates for no win).

| grid | RMHD z_spectral, separable | CMHD, diagonal | ratio |
|---|---|---|---|
| 128²×16 | 55.79 ms/step (spread 1.5%) | 103.16 ms/step (spread 1.0%) | **1.85×** |
| 256²×16 | 237.08 ms/step (spread 1.2%) | 481.25 ms/step (spread 2.4%) | **2.03×** |

Repeated in a second session on the same machine minutes later: 54.99 / 103.97 (1.89×) and
236.61 / 481.29 (2.03×) — the 128² ratio moves by 2%, the 256² one not at all. Spread is
max−min over the 9 repetitions as a percentage of the median; nothing here is above 4.2%.

**Context: the transform count.** CMHD does **23** 3-D FFTs per RHS evaluation (13 inverse:
ρ, u(3), B(3), ω(3), j(3); 10 forward: the combined curl force(3), the combined scalar(1),
u×B(3), ρu(3) — docs/numerics.md, CMHD_PLAN §3.4) against z_spectral RMHD's **10** (8
gradient iffts + 2 `NonlinearTerm` ffts), i.e. a 2.3× transform ratio. The measured step
ratio is 1.85–2.03×, slightly BELOW that, because RMHD's separable propagator and its
per-stage work are not free either and CMHD's diagonal exp is cheaper per stage. State is
7/2 = 3.5× RMHD's; the step is not.

`memory_analysis()` at the same configurations, in the `bench/memory_probe.py` u convention
(u = one field-sized complex array, `nz*nkx*nky*itemsize`; at 256²×16 fp64 that is 8.45 MB):

| grid | RMHD total | CMHD total |
|---|---|---|
| 128²×16 | 18.69 u | 48.71 u |
| 256²×16 | 18.70 u | 48.76 u |

2.61× the RMHD total against 3.5× the state — the fixed per-stage and propagator costs do
not scale with `nfields`. CMHD's 48.7 u is flat in grid size, as RMHD's is. (For scale, the
committed probes put GDI 2D 256² lsrk33 putzer2 at 41.1 u and RMHD z_spectral lsrk54 ν=η at
18.4 u; the 18.69/18.70 here reproduce that.)

Reproduce: `python bench/cmhd_perf.py --grids 128 256 --nz 16 --nrep 9`. It refuses to start
if `/private/tmp/taranis_bench.lock` exists and writes/removes its own.

## Where the z_spectral step's extra time goes (2026-08-19)

`bench/zspectral_profile.py`, same machine as the particle benchmarks (Apple M1, macOS 14,
jax 0.10.0, CPU, fp64, quiet machine), RMHD at **128²×16**, lsrk33, fixed `dt=1e-3`,
elsasser forcing, `nblock=20`, `nrep=10`. The whole gap is the **linear propagator**, not
the transforms.

| | finite-difference z | z_spectral | ratio |
|---|---|---|---|
| ms/step | 42.9 | 81.6 | 1.90 |
| GFLOP/step (XLA cost analysis) | 0.065 | 0.255 | 3.91 |

Ablation ladder inside the real (scanned, fused) z_spectral step — each row drops one piece
of `Putzer2Operator` and re-times; every variant is numerically wrong on purpose:

| variant | ms/step | that piece costs |
|---|---|---|
| baseline (production) | 79.8 | — |
| − the complex `sqrt`/`cosh`/`sinh` in `_coeffs` | 51.9 | **27.9** |
| − `exp(m·tau)` as well | 48.1 | 3.8 |
| − the 2×2 apply as well (propagator = identity) | 45.9 | 2.2 |
| finite-difference-z step, for comparison | 42.9 | — |

So of the +37 ms, **33.9 ms is the putzer2 matrix exponential** (27.9 of that the complex
transcendentals alone, evaluated per stage on a complex (nz,nkx,nky) grid) and the remaining
3.0 ms is everything else — the (z,x,y) transforms net of the finite-difference-z stencil the
spectral mode does not run. The isolated pieces say the same: the whole RHS times 14.8 ms (FD)
against 15.8 ms (spectral), and the `ifft` of the gradient stack 4.17 vs 4.25 ms, while FD
alone pays 1.3 ms for `FDLinearTerm`. An independent check swapping only the operator on a fixed grid agrees:
with L forced to the diagonal backend, fd 42.4 vs spec 47.1 ms/step; with L forced to a 2×2
putzer2 operator of the same z extent in both modes, fd 82.8 vs spec 88.6.

**Isolated timings mislead here, in both directions.** `apply_exp` timed on its own with the
kgrid *closed over* reports 0.3 ms because XLA constant-folds `exp(L·tau)` away entirely; with
the kgrid passed as an argument it reports 19.6 ms, which then over-counts because the fused
step shares work across the two `apply_exp` calls of a stage. The ablation ladder is the number
to trust. Monkeypatching for an ablation needs `jax.clear_caches()` — the jaxpr trace cache is
keyed on the function and avals, so without it every variant silently re-reports the baseline
(this bit the first run of this profile).

Scaling with nz at nx=128 (ms/step, fd → spec): nz=4 11.4 → 18.1 (1.59×), nz=8 22.3 → 40.5
(1.82×), nz=16 42.9 → 88.5 (2.06×), nz=32 88.6 → 180.1 (2.03×). Both sides are linear in nz;
the ratio saturates near 2 once the propagator's per-mode cost dominates the fixed overheads.

**Why the complex transcendentals cost what they do** (optimized HLO, `jax.jit(...).compile()
.as_text()`, XLA CPU): `cosh(z)` on complex128 lowers to `(exp(z) + exp(−z))/2` with each
complex `exp` expanded as `exp(x)·(cos y + i sin y)` plus overflow guards — 4 real `exp`, 2
`cos`, 2 `sin`, 6 selects; `sinh` the same again with nothing shared (XLA does not recognise
`exp(−z)` as `1/exp(z)`, nor `cos(−y)` as `cos(y)`); complex `sqrt` is 7 real `sqrt`, 9
divides, 18 selects (branch-cut and overflow handling); complex `exp` is 2 `exp` + `cos` +
`sin`. So one putzer2 `_coeffs` + `exp(m·tau)` evaluates **10 exp + 5 cos + 5 sin + 7 sqrt + 9
div per mode**, where the mathematics needs 2 complex exps (4 exp + 2 cos + 2 sin) — or, for an
L whose `m` is real and `s²` real and one-signed, 1 real exp + 1 cos + 1 sin. And on XLA CPU
each real transcendental is 3–6 ns/element (`exp` is a vectorised polynomial, `sin`/`cos` are
slower library-class calls), so 20-odd of them per mode per stage over 133k modes is the
~10 ms per `apply_exp` the ablation found. The same operator shape and mode count occurs for
2D GDI at 512² on an IF scheme (putzer2 on (2,2,1,nkx,nky) = 131k modes), so the IF path
there pays it too; the IMEX path (GDI production) is rational — `solve_shifted` has no
transcendentals — and is unaffected.

**Done: hoisted stage propagators (2026-08-19, `params.hoist_propagator`, default True).**
The exponent depends on `tau = gamma_s·dt` only, so whenever dt is frozen over a block — fixed
dt, or one `cfl_every` block — `run.py` forms every stage's `exp(L·tau)` ONCE per block
(`timestepping.stage_exp_ops` → `propagators.ExpOp` pytrees, stacked as the stage scan's xs)
and each stage only applies it. Same arrays in the same op order, so **hoisted == unhoisted
bitwise** at fp64 in every cell of `tests/test_hoist_propagator.py` (2D/FD-z/z_spectral ×
lsrk33/lsrk54/rk44 × fixed/cfl_every=2/adaptive × scan/unrolled; at fp32 one cell is 1 ulp
off from a fusion difference). Re-measured
(same session, loaded machine, so compare within the row):

| z_spectral 128²×16 | unhoisted ms/step | hoisted | |
|---|---|---|---|
| fixed dt, lsrk33 | 78.7 | **48.7** | 0.62× |
| fixed dt, lsrk54 | 146.2 | **81.5** | 0.56× |
| fixed dt, rk44 | 64.9 | 62.1 | 0.96× (rk44's two taus were already shared) |
| adaptive, `cfl_every=4` | 90.0 | **54.5** | 0.61× |
| adaptive, `cfl_every=1` | 89.6 | 91.2 | nothing frozen, nothing to hoist |
| FD-z, fixed dt, lsrk33 | 44.5 | 44.0 | diagonal L is z-broadcast: nothing to gain — and not hoisted |

The remaining gap to the FD-z step (48.7 vs 44) is the (z,x,y) transforms plus the 2×2 apply
itself. Verified in the optimized HLO (`bench`-style count of `exponential`/`cosine`/`sine`/
`sqrt` per while body): with hoisting, zero transcendentals in the step loop for fixed dt and
all of them in the outer cfl-block loop for `cfl_every>1`; with `hoist_propagator=False` the
legacy graph (33 per stage, inside the stage scan, where `gamma` is a scanned value XLA cannot
hoist past). That knob exists for memory: one `ExpOp` per stage — for putzer2 4 complex
arrays of L's full shape, i.e. 4·nstage·nz·nkx·nky·16 B at fp64 (128²×16, lsrk33: 16 MB;
256²×64, lsrk54: 0.7 GB; 512²×128 fp32 lsrk54: 2.7 GB). Note XLA's own loop-invariant code
motion would hoist these too once they are formed outside the stage scan with static `gamma`
(observed in the HLO) — which is why the unhoisted path deliberately keeps the exponent
inside the stage scan: `False` must mean memory-light, not "hoisted by XLA instead".
Only the putzer2 backend is hoisted: the diagonal backend's exp is one real exp per mode per
stage and z-broadcast for FD-z, so there is nothing to gain, and a first version that hoisted
it too broke gate 6's bitwise reference on the 2D fixed-dt `simulate_scan` configs by 15
elements at 1e-23 absolute (with a literal `gamma`, XLA folds `(L·dt)·gamma` differently) —
the hoisted/unhoisted bitwise agreement is an op-order statement, not a guarantee against
constant folding, so expect round-off-level differences on other grids/versions and never
pin a hoisted putzer2 run bitwise against an unhoisted one across jax versions.

**Memory, measured** (`jit(block_of_steps).lower(...).compile().memory_analysis()`, fp64,
lsrk, elsasser forcing; U = one complex full-grid array = nz·nkx·nky·16 B; "temp" is XLA's
working set for the program, "total" adds the live arguments — state + kgrid):

| case | U | kgrid | state | temp | temp/U | total |
|---|---|---|---|---|---|---|
| spec 128²×16 lsrk33, unhoisted | 2.0 MB | 12.5 MB | 4.6 MB | 64.9 MB | 31.9 | 81.9 MB |
| spec 128²×16 lsrk33, hoisted | | | | 72.0 | 35.4 | 89.0 (+9%) |
| spec 128²×16 lsrk54, hoisted | | | | 105.0 | 51.7 | 122.0 (+49%) |
| FD-z 128²×16 lsrk33 | 2.0 | 0.3 | 4.6 | 46.0 | 22.6 | 50.8 |
| spec 256²×64 lsrk33, unhoisted | 32.2 | 196 | 66.5 | 1018 | 31.6 | 1281 |
| spec 256²×64 lsrk33, hoisted | | | | 1131 | 35.1 | 1394 (+9%) |
| spec 256²×64 lsrk54, hoisted | | | | 1661 | 51.5 | 1924 (+49%) |
| FD-z 256²×64 lsrk33 | 32.2 | 1.1 | 66.5 | 714 | 22.1 | 781 |

Reading it: the z_spectral step's working set is ~32 U before any hoisting (FD-z: ~22 U) —
the RHS's real-space gradient stack and the rfftn intermediates, not the propagator — and the
persistent putzer2 operator (`lin.L` 4 U + `lin.m` + `lin_s2`) is 6 U = 196 MB at 256²×64.
Hoisting adds 4·nstage U of live arrays less the per-stage coefficient temporaries it
removes: measured +3.5 U for lsrk33 (+9% of the program) and +20 U for lsrk54 (+49%);
`cfl_every` blocks give the same numbers as fixed dt. So lsrk33 hoisted is cheap; lsrk54
hoisted is the case to think about on a memory-bound grid.

**Splitting the operator into perp-only and z-only factors** would make the hoisted memory
vanish (perp arrays (nkx,nky) plus z arrays (nz,) per stage) and also drop the 6 U operator
itself — but `exp((A+B)τ) = exp(Aτ)exp(Bτ)` only when `[A,B] = 0`. RMHD spectral-z with
ν = η: `L = D(k⊥)·I + i·kz·σ_x` (+ `−z_diss_k·kz⁴·I`), everything commutes, the split is EXACT
(it is the Elsasser-separable form measured at 0.62× above, which needs no hoisting and no
change of state variables). ν ≠ η: `diag(d_φ,d_ψ)` does not commute with `σ_x`, the split is a
Lie splitting with O(τ²·(d_φ−d_ψ)·kz) error — not acceptable for a scheme whose point is the
exact linear propagator. KAW-type operators (entries `i·kz·f(k⊥)`): the exponential carries
`cos(kz·√(fg)(k⊥)·τ)`, a function of the *product*, which no product of a kz-only and a
k⊥-only array reproduces — there the memory-free choice is the per-stage real-trig
evaluation (`m` real, `s²` real ≤ 0: 1 real exp on (nkx,nky) + cos + sin on the full grid,
0.75×) and hoisting stays the speed-for-memory lever. Generic halving available either way:
a 2×2 with `L00 = L11` and `L01 = L10` has `m00 = m11`, `m01 = m10`, so `Putzer2Exp` could
store 2 arrays per stage instead of 4 (detectable at setup).

**Precomputed eigenvectors, reassessed against these numbers.** `V`, `V⁻¹`, `λ` are 10 U
persistent; hoisting on top stores `exp(λτ)` = 2 U per stage and the apply costs 10 complex
mults instead of putzer2's 4: totals 10 + 2·nstage (16/20 U for lsrk33/54) against putzer2's
6 + 4·nstage (18/26 U) — a saving only for lsrk54, bought with ~2 ms/step of extra multiplies.
Unhoisted it is the 0.69× per-step path. For RMHD specifically `V` is the constant Elsasser
transform (nothing stored) and `λ = d ± i·kz` is separable at ν = η (nothing stored): the
eigen route collapses into the separable form above. So: RMHD wants the separable propagator
(zero memory, 0.62× at every step, adaptive included); generic 2×2 operators want putzer2 +
hoisting for speed, eigen storage only if lsrk54's hoisted 26 U is the constraint.

**Schemes against each other under z_spectral** (same run, 128²×16, fixed dt, hoisting on
unless stated; memory at 256²×64 is the same picture). "stab." is the scheme's own
imaginary-axis stability limit `|ω·dt|_max` for the explicit (advective) part — L is exact
under IF so only the nonlinear term sees the RK stability polynomial (lsrk33 1.73, rk44 2.83,
lsrk54 3.34, computed from the stored tableaus); the last column is ms/step ÷ stab., i.e. cost
per unit simulated time IF `cfl_safety` were raised to each scheme's limit — at a common
`cfl_safety` the comparison is just ms/step:

| scheme | temp/U | ms/step | ms/stage | cost per unit t at stab.-limited dt (rel.) |
|---|---|---|---|---|
| lsrk33, unhoisted | 31.9 | 80.5 | 26.8 | 1.00 |
| lsrk33 | 35.4 | 49.2 | 16.4 | 0.61 |
| lsrk54, unhoisted | 31.9 | 130.5 | 26.1 | 0.84 |
| lsrk54 | 51.7 | 81.3 | 16.3 | 0.52 |
| rk44 (hoist on or off: identical) | 36.2 | 60.0 | 15.0 | **0.46** |

Two conclusions. (i) The 2N (two-register) property of the LSRK schemes buys nothing here:
the z_spectral working set is ~32 U of RHS temporaries either way, hoisted lsrk33 (35.4 U)
sits level with rk44 (36.2 U, four k-registers AND its two hoisted ops), and hoisted lsrk54
(51.7 U) is the most memory-hungry option of all. (ii) rk44 needs no hoisting machinery:
its two taus (`dt/2`, `dt`) are fixed per step, so XLA's loop-invariant code motion already
lifts them out of the step loop (hoist on/off identical), it is 4th order, its per-stage cost
is the lowest (no stage scan, no `cond`), and it tolerates the largest dt of the three per
stage but one. At a common `cfl_safety` hoisted lsrk33 remains the cheapest step (49 vs 60
ms); if `cfl_safety` is scaled to the scheme, rk44 is the cheapest per unit time and 4th
order — worth considering as the z_spectral default once the separable propagator lands
(which removes the hoisted-memory term from every row and leaves only the registers, where
lsrk33 wins again by 4 U). **FD-z is a different regime:** the diagonal z-broadcast exp costs
nothing per stage, so hoisting is neither needed nor enabled there (`hoistable=False`), and
the scheme choice is the classical one — lsrk33 for cost, lsrk54/rk44 for order.

**DONE for RMHD ν=η (Z1, 2026-08-20): the Elsasser-separable backend landed** — see
"Memory: where it goes and what was removed" below and docs/numerics.md for the
derivation. Measured: z_spectral RMHD total 41.7 → 20.4 u at 128²×32 (scheme- and
hoist-independent; the stage ExpOps are 0.02–0.10 u against putzer2's 3.4–27.3 u),
fixed-dt step 0.94–0.98× and adaptive `cfl_every=1` 0.40–0.47× the pre-Z1 hoisted
putzer2 step, HLO transcendentals down from 30 exp + 30 cos + 30 sin + 21 sqrt full-grid
to one perp-plane exp plus exp/cos/sin on (nz,1,1) per stage. ν ≠ η and GDI stay on
putzer2 — for those the paragraph below is the record of what was tried and what shipped:

**DONE for putzer2 (Z2, 2026-08-20) — the adaptive `cfl_every=1` path.** Nothing is frozen
there, so the per-stage evaluation stays and only a cheaper evaluation helps. The generic,
no-memory, any-L option below is what shipped: store `s = sqrt(s2)` at setup (kills the
complex sqrt — its 7 sqrt/9 div/18 selects per mode —
from the step; `tau > 0` so the branch is immaterial and cosh/sinh·z are even anyway) and form
`w = exp(s·tau)` once, `cosh = (w + 1/w)/2`, `sinh/s = (w − 1/w)/(2s)` with the small-|z|
Taylor branch kept and widened ~100× in `|z²|` to cover the `w − 1/w` cancellation
(docs/numerics.md). Measured stage-body HLO: `sqrt` 14 → 0, `exp` 20 → 8, `cos`/`sin`
16 → 6, `div` 21 → 5. Step time, adaptive `cfl_every=1`, interleaved medians: GDI 2D 512²
lsrk33 **0.68×** (fp64) / 0.76× (fp32); RMHD z_spectral ν≠η 128²×32 lsrk33 0.79/0.81× —
the plan's ≤0.75× target met on the GDI production path, the RMHD corner Amdahl-limited by
its transform share. It also came with a memory windfall the bar did not ask for (the old
`_coeffs` materialised full-grid `sqrt`/`cosh`/`sinh` intermediates): −1.4 u on
`rmhd_zspec_64x16_lsrk33_uneq`, −3.4 on `gdi2d_256_lsrk33` and `gdi3d_64x16_lsrk33`,
−13.5 on `gdi2d_256_lsrk54`. Two alternatives were measured alongside it and not taken on this
path. Structure-aware, when
`m` is real and `s2` real one-signed (RMHD: `m = −νk^{2h}`, `s2 = −kz²` — concrete at setup):
1 real exp + cos + sin — measured **0.75×** the step; and the Elsasser-separable form
(`e^{dτ}` on (nkx,nky) ⊗ `e^{±ikzτ}` on (nz,), no change of state variables) **0.62×** — the
same ratio as hoisting, but for every step including adaptive `cfl_every=1`, at no memory;
that one IS what Z1 shipped, for the ν = η RMHD case where it applies, which is why it is
not also a putzer2 option.
Both change round-off on the putzer2 paths (no bitwise gate pins them; the FD diagonal path is
untouched). A naive real-trig version that evaluates both `cosh`/`cos` branches under a `where`
and casts back to complex measured *slower* (114.8 ms/step). The full per-mode
eigendecomposition (V, V⁻¹, λ stored) was measured at 0.69× and rejected: 10 full-grid arrays
for less than the separable form gives.

## Memory: where it goes and what was removed

The measurement instrument is `bench/memory_probe.py` (Phase 0 of
`plans/MEMORY_PERF_PLAN.md`), which reports per case: XLA
`compiled.memory_analysis()` as temp/args/out in bytes and in **u** — one field-sized
complex array, `nz_local·nkx·nky·itemsize` (8 B fp32, 16 B fp64; the RMHD state is
2 u) — plus `total_u = temp+args+out`, the device `peak_bytes_in_use` on GPU, and the
median ms/step of a jitted `block_of_steps`. Three conventions to hold when reading any
number in this section: (i) `total_u` is the quotable one — `lin_*` lives in *args*
(the hoisted ExpOps are formed inside the jitted block and sit in *temp*), so temp-only
understates z_spectral by ~8–10 u; (ii) the probe
measures the **non-donated** graph (it reuses one state across reps), while production
jits with `donate_argnums=(0,)` and may alias input to output — u values describe the
probe's graph, consistently at every measurement point, and phase gates in the plan are
**deltas** between probe runs, never absolute targets; (iii) under `comm_backend="jax"`
both `memory_analysis()` and u are per-device (verified against a fake-device mesh).
(The "Memory, measured" block above uses U = the fp64 u at other grids — same idea,
different absolute numbers.)

**CPU baseline** (M1 laptop, jax 0.10.0, fp32, `bench/memory_probe_laptop_baseline.json`
— the regression reference. The fp64 twin `..._fp64.json` has the same memory to
≤0.38 u — every difference sits in the z_spectral/GDI args block (fp64-always scalars),
FD-z rows agree to ≤0.005 u — and costs 1.3–2.1× the time. The 128²×32 gate grid is in
`bench/memory_probe_laptop128_baseline.json`. u = 0.26 MB at 64²×16, 0.25 MB at 256²):

| case (64²×16 RMHD/GDI-3D, 256² GDI-2D) | temp u | args u | total u | ms/step |
|---|---|---|---|---|
| rmhd_fdz lsrk33 / lsrk54 | 26.58 | 2.26 | 30.96 | 5.07 / 8.28 |
| rmhd_fdz imexcb2 / cb3e / cb3c | 24.57 | 2.26 | 28.96 | 5.09 / 7.54 / 6.84 |
| rmhd_fdz imexcb3f | 56.45 | 2.26 | 60.83 | 10.96 |
| rmhd_fdz lsrk33 / lsrk54 / cb3e unrolled | 37.95 / 55.20 / 35.32 | 2.26 | 42.33 / 59.58 / 39.71 | 7.36 / 21.15 / 7.11 |
| rmhd_zspec lsrk33 hoisted / unhoisted | 32.89 / 29.51 | 8.31 | 43.33 / 39.95 | 6.04 / 12.02 |
| rmhd_zspec lsrk54 hoisted / unhoisted | 51.76 / 29.51 | 8.31 | 62.19 / 39.95 | 10.08 / 19.70 |
| rmhd_zspec lsrk33 / lsrk54 unrolled (hoist on = off) | 32.89 / 43.76 | 8.31 | 43.33 / 54.20 | 5.70 / 9.50 |
| rmhd_zspec imexcb3e | 18.89 | 6.31 | 27.33 | 9.52 |
| rmhd_zspec lsrk33 ν≠η (putzer2) | 32.89 | 8.31 | 43.33 | 5.95 |
| gdi2d_256 lsrk33 / lsrk54 | 33.47 / 53.64 | 11.13 | 48.60 / 68.77 | 4.07 / 6.77 |
| gdi2d_256 imexcb2 / cb3e / cb3c | 16.97 | 9.13 | 30.10 | 4.89 / 6.73 / 6.83 |
| gdi3d lsrk33 | 31.57 | 8.31 | 42.01 | 4.86 |
| gdi3d imexcb2 / cb3e / cb3c | 14.95 | 6.31 | 23.39 | 5.59 / 7.67 / 7.64 |

What the table says, structurally (buffer breakdown in `plans/old/TARANIS_MEMORY_HANDOFF.md`):
the FD-z IF working set is dominated by the batched gradient transforms (8 u k-space
stack + ~8 u real-space output — the plan's F1) and the halo-concatenate pair (~4.3 u,
F2/F3); z_spectral adds the resident putzer2 operator (6 u of args) and, hoisted,
4·nstage u of ExpOps (+3.4 u lsrk33, +22 u lsrk54 — the plan's Z1 removes both for
ν=η); the [2R] CB-IMEX steppers are the memory floor everywhere (no exponentials, one
shifted solve per stage); `imexcb3f` is unrolled-only and pays 2.1×. `lsrk_scan=False`
costs 1.37–1.92× the scan path's total for every stepper measured, and on the unrolled
path `hoist_propagator` is a no-op on both axes — XLA hoists the literal-gamma stage
exponents itself (the `_unroll_hoist{1,0}` rows: identical totals, identical speed at
both settings, both precisions). Memory in u reproduced identically (0.00 u per case) on a second
machine and across jax 0.10.0/0.10.2 during Phase 0 validation — timings are this
laptop's only.

**What has been removed so far** (deltas at 128²×32 fp32, scratchpad post-phase probe
runs vs the committed baselines):

- **F1** (per-field gradient transforms, `grads` a tuple): FD-z 30.48 → 23.55 u,
  z_spectral unhoisted 39.85 → 34.92 u, IMEX FD-z 28.48 → 22.57 u — the 8 u stacked
  k-space gradient is gone, and on a bandwidth-bound CPU that was also an 18–31%
  step-time win. `shared_physics.GRAD_CHUNK` (module constant, default 1 = per-component
  transforms) batches fields per ifft for the GPU comparison; chunks 1/2/4 are bitwise
  identical and chunk 4 reproduces the pre-F1 graph byte-exactly.
- **Z1** (Elsasser-separable propagator, ν = η z_spectral RMHD): 34.92 → **20.35 u**,
  identical for lsrk33/lsrk54 and hoist on/off (nothing scheme- or dt-dependent is
  stored; the kgrid args block dropped 6 u and the per-stage putzer2 coefficient
  temporaries went with the backend). Hoisted lsrk54's 61.97 u → the same 20.39 u.
  Fixed-dt step 0.94–0.98× and adaptive 0.40–0.47× the pre-Z1 hoisted putzer2 step.
  FD-z, 2D, GDI and ν ≠ η rows: unchanged to the last digit.

**GPU baseline, G1 Kaggle P100 16 GB** (jax 0.11.1, isolated subprocess per case,
`bench/memory_probe_p100_baseline_fp{32,64}.json`; fp32 at 512²×128 / GDI-2D 1024², u =
135 MB; fp64 halves the grid. `peak` is the device allocator's `peak_bytes_in_use` — it
runs a near-constant ~16 u above `total_u` on every row, the state/kgrid/warm-up
residue):

| case (fp32) | total u | peak u | ms/step |
|---|---|---|---|
| rmhd_fdz lsrk33 / lsrk54 | 30.10 | 46.07 | 182.7 / 305.2 |
| rmhd_fdz imexcb3e | 24.06 | 40.03 | 290.3 |
| rmhd_fdz lsrk54 unrolled | 28.09 | 44.06 | 336.8 |
| rmhd_zspec lsrk33 hoisted / unhoisted | 54.17 / 36.17 | 70.14 / 52.14 | 178.0 / 209.8 |
| rmhd_zspec lsrk54 hoisted / unhoisted | 70.17 / 36.17 | 86.14 / 52.14 | 338.3 / 345.6 |
| rmhd_zspec imexcb3e | 30.30 | 48.27 | 222.7 |
| rmhd_zspec lsrk33 adaptive cfl_every=1 | 36.17 | 52.14 | 211.0 |
| rmhd_zspec lsrk33 ν≠η (putzer2) | 54.17 | 70.14 | 178.0 |
| gdi2d_1024 lsrk33 / imexcb3e | 53.63 / 30.25 | 66.12 / 44.74 | 4.4 / 5.3 |
| gdi3d_256x64 imexcb3e | 26.34 | 41.30 | 22.7 |

Zero OOMs: the hoisted z_spectral lsrk54 row (70.2 u ≈ 9.2 GB + peak 11.6 GB) fits the
16 GB card, as the plan predicted.

**P100 post-F/Z point** (same profile, tree with F1+Z1+F3+Z2+F2-behind-constant;
`bench/memory_probe_p100_postFZ_*.json`). Memory, fp32 512²×128 total u, baseline →
postFZ (default `GRAD_CHUNK=1`, `Z_STENCIL_BLOCKS=False`): FD-z lsrk 30.10 → 20.10
(17.82 with `--z-blocks 1`); **every z_spectral ν=η row 30–70 → 17.30** — scheme-,
hoist- and dt-independent; ν≠η 54.2 → 43.3; GDI-2D 53.6 → 45.2 (hoist0 38.2); the
fp64 twin shows the same collapse (all RMHD rows ≈ 17.2 u at 256²×64). The GPU
scheduler does what the CPU one would not (the F4 finding inverts here): FD-z drops
far below the CPU's 22.7 u without any bracket reordering. Timing tells a
granularity story: at the default per-component transforms (`GRAD_CHUNK=1`) FD-z/IMEX
rows run +4–17% vs baseline, while `--grad-chunk 2` beats baseline nearly everywhere
(FD-z lsrk −5.5%, z_spectral lsrk54 267 vs 338 ms = 0.79×, adaptive 160 vs 211 =
0.76×) at +2–3 u — the P100 wants chunk 2 where the M1 CPU wants chunk 1
(chunk 2 measures 1.25–1.54× on CPU); the F1 granularity decision is per-device.
`--z-blocks 1` on the P100: −2.3 u for +1.5–4.7% time (the scheduler already
recovers most of the halo waste on the default path). GDI-IF putzer2 hoist pair
(Z3 input): unhoisted = 1.22× the hoisted step for 7 u — hoisting still pays on
GDI-IF after Z2.

`Z_STENCIL_BLOCKS` and the probe's `--z-blocks` no longer exist: the block-stencil path
was a one-day experiment, all three platforms preferred the padded slab, and it was
deleted in the plan's docs sweep (2026-08-20; the surviving `z_derivatives` is the
verbatim pre-experiment code, pinned by an optimized-HLO identity check). The rows
quoting the flag are kept because they are the measurement that decided it.

**GPU baseline, G2 Savio GTX 2080Ti 11 GB** (jax 0.10.2, job 37775868,
`bench/memory_probe_gtx2080_baseline_fp{32,64}.json` + `..._fp32_jax4.json`; same grids
as the P100 profile plus a 512²×64 twin of the OOM candidate):

| case (fp32, 512²×128) | total u | peak u | ms/step |
|---|---|---|---|
| rmhd_fdz lsrk33 / lsrk54 | 28.06 | 36.07 | 162 / 275 |
| rmhd_zspec lsrk33 hoisted / unhoisted | 54.17 / 36.17 | 54.2 / 44.2 | 149 / 199 |
| rmhd_zspec lsrk54 **hoisted: OOM** / unhoisted | 70.17 / 36.17 | — / 44.2 | OOM / 328 |
| rmhd_zspec lsrk54 hoisted, 512²×**64** | 70.21 | 79.2 | 135 |
| rmhd_zspec imexcb3e / adaptive lsrk33 | 30.30 / 36.17 | 40.3 / 44.2 | 204 / 201 |
| gdi2d_1024 lsrk33 / imexcb3e | 53.63 / 30.25 | 60.1 / 38.8 | 3.8 / 4.5 |
| FD-z lsrk54, 4-GPU sharded (`comm_backend="jax"`) | 24.76 /dev | 32.76 /dev | 75.3 |

The headline row is the recorded "before" of the Z1 flip: **hoisted z_spectral lsrk54 at
512²×128 fp32 OOMs the 11 GB card** (9.2 GB program + context; the allocator fails on a
7.5 GB request) while the unhoisted path fits — after Z1 removes the hoist memory this
row must fit, and that flip is the G2 post-Z deliverable.

**2080Ti post-F/Z point** (`bench/memory_probe_gtx2080_postFZ_*.json`): **the flip is
recorded** — the row that OOM'd now runs at 17.3 u, 18.3 u peak = 2.4 GB, 240 ms. Every
RMHD row lands at 17.1–17.3 u (FD-z included: 28.1 → 17.1 — this card's scheduler beats
even the P100's 20.1), all FASTER than baseline at the default settings (FD-z lsrk33
148 vs 162 ms; z_spectral adaptive 145 vs 201 = 0.72×; z_spectral lsrk54 238 vs the
old unhoisted 328 = 0.73×). Decision pairs, closing the plan's §9 items (Alfred,
2026-08-20): the chunk pair reads 1.8–3.3% for chunk 2 — under the 5% bar, so
**GRAD_CHUNK stays 1 everywhere** (P100-class cards gain 3–20% from setting 2 — the one
per-card tuning worth knowing); the z-blocks pair is negative on this card in BOTH
memory (19.8 vs 17.1 u) and time (+9–14%), so with all three platforms preferring the
old path **the block-stencil path and Z_STENCIL_BLOCKS were deleted** (docs sweep,
2026-08-20);
GDI-IF hoisting still pays (1.22× unhoisted on P100, 2080Ti consistent), so
**hoist_propagator stays, putzer2-only**. Z1's GPU timing gate final tally: lsrk54
0.72–0.79× and adaptive 0.72–0.76× meet ≤0.85; lsrk33 fixed-dt 0.89–0.97× does not
(3-stage coefficient work amortizes less) — moot at ¼ the memory, recorded for honesty. The 2080Ti is otherwise
~10–15% faster than the P100 per step, the 4-GPU sharded row matches the earlier run
exactly (near-perfect weak scaling at nz_local = 32, per-GPU per-point parity with the
P100), and fp64 mirrors the P100 structure. Absolute u differs a little from the P100
JSONs on some rows (e.g. FD-z 28.1 vs 30.1) — jax 0.10.2 vs 0.11.1 buffer accounting;
compare within a card/version, not across. What the GPU says that the CPU
could not: (i) hoisting buys lsrk33 0.85× but lsrk54 only ~0.98× — streaming 22 u of
ExpOps per step costs about what the transcendentals cost, exactly the bandwidth trade
Z1 sidesteps by storing nothing full-grid; (ii) z_spectral hoisted lsrk33 (178 ms) is
*at parity with FD-z lsrk33* (183 ms) on GPU — the z_spectral premium is a CPU
phenomenon at this size; (iii) adaptive `cfl_every=1` (211 ms) sits at the unhoisted
step, as expected (nothing frozen to hoist); (iv) FD-z lsrk54 unrolled costs LESS
memory than the scan on GPU (28.1 vs 30.1 u — the reverse of CPU, where it is 59.6 vs
31.0) and 1.10× the time, so `lsrk_scan=False` remains a no-win on this card. fp64 at
256²×64: same u structure to ≤2.1 u, 46.1/75.6 ms FD-z lsrk33/54. Post-Part-F and
post-Part-Z reruns of both G1 launches (`../lugus/launch.py run bench/memory_probe.py
--entry-kwargs '{"profile": "p100", "tag": "postF"}'` etc.) fill in the deltas here.

## Memory and time accounting after MEMORY_PERF_PLAN (2026-08-20)

Measured 2026-08-20 on HEAD `72840fa` (every MEMORY_PERF_PLAN build phase landed), Apple M1
laptop, macOS 14, jax 0.10.0, CPU, **fp32**, quiet machine under the scratchpad bench lock.
Working tree clean throughout: nothing under `taranis/` was edited, every variant below is a
monkeypatch applied in a scratch script before the first trace. Provenance: the memory numbers
are compile-time (`memory_analysis` + XLA buffer dumps) and contention-immune; the timing
numbers all come from the **second** interleaved campaign (`acct_campaign2.jsonl`), run with
the machine confirmed quiet, and its base rows have a 1.1–2.8% round-to-round scatter. A first
campaign (`acct_campaign.jsonl`, same variants, machine at load ~2–3) agrees with it to within
1–3% on every row and 4% on the base rows; it is kept as the contention control, not quoted.

Canonical configuration: RMHD 128²×32 (u = 2.03 MiB), GDI 2D 256² (u = 258 KiB), defaults
(`GRAD_CHUNK=1`, the padded z stencil, `hoist_propagator=True`, `lsrk_scan=True`), fixed
`dt = 1e-3`, no forcing, no particles, `comm_backend="serial"`, single process. The three
paths are the ones the plan sizes production from:

| path | probe label | propagator backend |
|---|---|---|
| RMHD finite-difference z, lsrk54 | `rmhd_fdz_128x32_lsrk54` | diagonal (real, z-broadcast) |
| RMHD `z_spectral` ν=η, lsrk54 | `rmhd_zspec_128x32_lsrk54_hoist1` | separable |
| GDI 2D 256², lsrk33 | `gdi2d_256_lsrk33` | putzer2 (dense 2×2) |

Scripts (scratchpad): `acct_dump.py` (compile + XLA dump + `memory_analysis`),
`acct_scope.py` (same compile with `jax.named_scope` wrappers monkeypatched around the RHS
seams, for attribution), `acct_table.py` / `acct_args.py` (lane and argument tables),
`acct_time.py` (one ablation variant per process), `acct_campaign.sh` (the interleaved
repeat campaign).

### Where the memory goes

`memory_analysis()` on the jitted `run.block_of_steps`, the probe's non-donated graph.

| path | temp | args | out | **total** |
|---|---|---|---|---|
| RMHD FD-z lsrk54 | 18.471 u | 2.129 u | 2.063 u | **22.663 u** |
| RMHD z_spectral lsrk54 (separable, hoisted) | 14.127 u | 2.235 u | 2.063 u | **18.425 u** |
| GDI 2D 256² lsrk33 (putzer2, hoisted) | 25.999 u | 11.131 u | 4.000 u | **41.130 u** |

Against the plan's §1 baseline table (same grids, pre-plan): FD-z lsrk54 30.48 → 22.66 u,
z_spectral lsrk54 hoisted 61.97 → 18.43 u, GDI 2D 256² lsrk33 48.6 → 41.13 u.

### Attribution method

The optimized-HLO `op_name` metadata carries only the jaxpr-level op path
(`jit(block_of_steps)/while/body/closed_call/mul`) and a `stack_frame_id` that the text dump
does not resolve, so a buffer cannot be attributed from the dump alone. Wrapping the RHS
seams (`grids.fft`/`ifft`, `grad_fields`, `bracket`, `z_derivatives`, `halo_exchange`, the
term funcs, the `ExpOp.apply`/`exp_op` methods) in `jax.named_scope` puts the seam name into
every instruction's `op_name`. `named_scope` is metadata only: the scoped compile reproduces
the plain compile's total **byte for byte** on all three paths, which is the check that the
attribution costs nothing.

Lanes below are distinct temp-arena **offsets**, sized by the largest buffer placed there.
Buffers at overlapping addresses are lifetime-disjoint reuse, so the lane-max sum exceeds the
arena high-water; both numbers are given.

### RMHD FD-z lsrk54 — 22.663 u (33 lanes, lane-max sum 26.42 u, arena 18.47 u)

| u | lanes | shape | what |
|---|---|---|---|
| 2.125 | 2 | `c64[2,34,128,65]` | the halo-padded z slab of `shared_physics.z_derivatives` — (nz+2)/nz × 2 u, one per RHS instantiation (peeled stage 0, and the stage scan) |
| 2.000 | 2 | `c64[2,32,128,65]` | LSRK stage-scan carry: `fields` and `delta` |
| 2.000 | 2 | `c64[2,32,128,65]` | `NonlinearTerm`'s k-space output, one per RHS instantiation |
| 2.000 | 1 | `c64[2,32,128,65]` | `FDLinearTerm`'s output — the `jnp.stack([df_dz[1],df_dz[0]])` field swap, materialised |
| 1.000 | 5 | `c64[32,128,65]` | per-component k-space gradient `i·k·f̂` before its inverse transform (`GRAD_CHUNK=1`) |
| 0.985 | 7 | `f32[32,128,128]` | real-space gradient components out of `irfft2` |
| (1.969) | 2 occupants | `f32[2,32,128,128]` | the two real-space bracket results stacked for the forward transform (they share offsets with 2 u k-space lanes) |
| | | | remaining lanes < 0.4 u: 0.28 u |

args 2.129 u = `state.fields` 2.000 + `forcing_state` `c64[1,2,128,65]` 0.062 +
`kgrid.lin.L` `f32[2,1,128,65]` 0.031 (the diagonal backend's real, z-broadcast L) + change.
out 2.063 u = fields + forcing_state.

**What bounds it:** the RHS gradient working set. The arena holds 16 k-space gradient values
(1 u each) and 16 real-space ones (0.985 u each) — eight per RHS instantiation, in 5 and 7
distinct lanes after reuse — plus the two 2.125 u halo-padded z slabs; F1's reading that all
four `(2,nz,nx,ny)` gradient pairs are co-live at the peak is what sets the peak, and F4 established that no source-level reordering retires one
early on the XLA CPU scheduler, so the ~2 u ideal-ordering prize is unreachable from source.

### RMHD z_spectral lsrk54, ν=η, hoisted — 18.425 u (34 lanes, lane-max sum 19.10 u, arena 14.13 u)

| u | lanes | shape | what |
|---|---|---|---|
| 2.000 | 2 | `c64[2,32,128,65]` | LSRK stage-scan carry: `fields` and `delta` |
| 2.000 | 1 | `c64[2,32,128,65]` | `NonlinearTerm`'s k-space output |
| 1.000 | 8 | `c64[32,128,65]` | per-component k-space gradient `i·k·f̂` |
| 1.000 | 1 | `c64[32,128,65]` | `NonlinearTerm` convert/scale temporary |
| 0.985 | 4 | `f32[32,128,128]` | real-space gradient components out of `irfftn` |
| 0.062 | 1 | `f32[4,128,65]` | **the hoisted stage propagators**: `SeparableExp.P = exp(dperp·τ)` for the four scanned stages |
| ~0 | 5 | `f32[4,32,1,1]`, `f32[32,1,1]` | the same ops' `c`, `s` (nz,1,1) envelopes |

args 2.235 u = fields 2.000 + `forcing_state` 0.062 + `dealias` `pred[32,128,65]` 0.125 +
the separable `lin.dperp`/`lin.dz`/`lin.kz` (≈ 0.03 u together).

**What bounds it:** the same gradient working set as FD-z, and nothing else — the operator has
left the accounting. The dense `lin.L`/`lin.m`/`lin_s2` that cost 6 u of arguments in the
baseline are gone (Z1), and hoisting five stages of `exp(L·τ)` now costs **0.06 u** instead of
the 22 u the putzer2 backend needed, because `SeparableExp` is `(nkx,nky) + 2×(nz,1,1)` reals
per stage rather than four full-grid complex arrays.

### GDI 2D 256² lsrk33, putzer2, hoisted — 41.130 u (35 lanes, lane-max sum 33.49 u, arena 26.00 u)

| u | lanes | shape | what |
|---|---|---|---|
| 2.000 | 4 | `c64[2,1,256,129]` | **hoisted `Putzer2Exp` entries** `m00,m01,m10,m11`, stacked over the two scanned stages (`stack_exp_ops`) |
| 1.000 | 7 | `c64[1,256,129]` | stage-0's four `Putzer2Exp` entries plus `_coeffs`' `cosh`/`sinhc`/`pref` intermediates |
| 0.500 | 3 | `f32[1,256,129]` | real intermediates inside `_coeffs` (the `|z²| < tol` branch) |
| 2.000 | 2 | `c64[2,1,256,129]` | LSRK stage-scan carry: `fields` and `delta` |
| 2.000 | 1 | `c64[2,1,256,129]` | the stage's `EXPAPPLY` output |
| 1.000 | 5 | `c64[1,256,129]` | per-component k-space gradient (three fields × two components) |
| 0.992 | 3 | `f32[1,256,256]` | real-space gradient components |
| 1.000 | 3 | `c64[256,129]` | `NonlinearTerm` k-space output/temporaries |

args 11.131 u = `kgrid.lin.L` `c64[2,2,1,256,129]` **4.000** + `lin.m` 1.000 + `lin_s2` 1.000
+ fields 2.000 + `forcing_state` `c64[1,2,256,129]` 2.000 + `ksq`/`inv_ksq` 0.500 each +
`dealias` 0.125. out 4.000 u = fields + forcing_state.

**What bounds it:** the operator, not the RHS. The putzer2 propagator owns 16.5 u of the
26.0 u temp arena (12 u of hoisted stage coefficients — four complex full-grid entries for
stage 0 plus four stacked over the two scanned stages — and 4.5 u of `_coeffs` intermediates)
and 6 u of the 11.1 u of arguments (`lin.L`/`lin.m`/`lin_s2`). The whole RHS working set —
gradients, brackets, transforms — is under 10 u. `hoist_propagator=False` is NOT the memory
knob here that the plan expects it to be: the same case with hoisting off measures 41.091 u,
so hoisting is memory-neutral on this backend and grid (§2 has why, and the GPU contrast).

### Two cases the memory probe has never carried

| case | temp | args | out | total |
|---|---|---|---|---|
| z_spectral lsrk54, **elsasser forcing on** | 16.409 u | 2.301 u | 2.125 u | **20.836 u** |
| the same case unforced (the row above) | 14.127 u | 2.235 u | 2.063 u | 18.425 u |

**Forcing costs +2.41 u**, and every existing probe row runs unforced, so this has never been
on the books. It is +0.066 u of arguments (`forcing_state` is `c64[2,2,128,65]` in elsasser
mode against `c64[1,2,128,65]` in momentum), +0.062 u of output, and **+2.28 u of temp** — the
O-U update, `reconstruct_envelope`, and the two full-grid reductions
(`perp_inner_product`/`perp_mean_square`) that `selfnorm_scale` needs. The lane count goes from
34 to 92: the forcing adds many sub-u lanes rather than one large one, and the largest new
lanes are 2 u field-vector temporaries in the once-per-step `_advance_forcing`, not in the
stage scan.

**Snapshot peak sequence** (block → `save_snapshot` → block, device `peak_bytes_in_use`
sampled at each stage): the driver runs it, and on CPU it reports null peaks with a note —
`jax.local_devices()[0].memory_stats()` is unavailable on the CPU backend. This case exists for
the GPU: the save materialises host copies of every state array while orbax holds its own
buffers, so a run's true high-water mark can sit at the checkpoint rather than in the step,
which a step-only probe cannot see. Run it on the P100 to get the number.

### Cross-check against the committed probes

`bench/memory_probe_p100_postFZ_fp32.json` is the committed post-plan probe (GPU, P100,
fp32, 512²×128 for the RMHD rows and 1024² for GDI 2D):

| path | this measurement (laptop, 128²×32 / 256²) | committed p100 postFZ (512²×128 / 1024²) |
|---|---|---|
| RMHD FD-z lsrk54 | 22.663 u | 20.096 u |
| RMHD z_spectral lsrk54 hoisted | 18.425 u | 17.309 u |
| GDI 2D lsrk33 hoisted | 41.130 u | 45.244 u |

The RMHD rows sit 1.1–2.6 u above the GPU numbers, which is the per-grid constant overhead the
plan's §0.4 note describes (the smaller grid pays proportionally more for the sub-u lanes and
for the peeled-stage duplication); the structural counts — 8 co-live gradient pairs, 2 carry
registers, one NL output per RHS instantiation — are identical. The GDI row moves the other
way (41.1 laptop vs 45.2 p100) because its putzer2 coefficient arrays scale with the grid
while `forcing_state` and the small kgrid entries do not.
The FD-z total also reproduces the F4 phase's landed figure (22.663 u) exactly, on the same
grid and tree.

### Where the time goes

Two independent methods, reported side by side because they disagree in known places.

**Primary — ablation ladder.** One variant per process (the F3 trace-cache rule; no
`jax.clear_caches()` anywhere), each variant swapping ONE part of the step for a cheap
stand-in of identical shapes and dtypes, re-timing the same jitted `run.block_of_steps`.
Five interleaved rounds (`acct_campaign.sh`, `NREP=21`, 240 measurements), so drift shows up
as round-to-round scatter rather than as a bias on one variant; the tables quote the
**minimum over rounds of the per-round minimum**, with the base row's round scatter as the
noise floor — 0.74 ms on 44.60 (FD-z), 0.76 on 67.40 (z_spectral), 0.07 on 2.50 (GDI), i.e.
1.7 / 1.1 / 2.8 %.

**Cross-check — op-level XLA:CPU profile.** `jax.profiler.trace` on the *scoped* build emits
one trace event per optimized-HLO instruction on the CPU backend, which rolls up by
`named_scope` into a per-part share (`acct_profscope.py`). Two caveats: the summed op time is
~1.9× the wall time because XLA:CPU runs the ops on a thread pool and the trace sums
per-thread intervals — read the profile as **shares, not wall time**; and container ops
(`while`) nest their body's duration and are excluded.

Step times against the plan's §1 baseline, same grids and machine: FD-z lsrk54 79.7 → 45.6
ms/step, z_spectral lsrk54 hoisted 98.2 → 68.7, z_spectral lsrk33 hoisted 64.2 → 39.6, GDI 2D
256² lsrk33 4.07 → 2.65 (medians).

### RMHD FD-z lsrk54 — 44.60 ms/step

Build-up ladder (each row restores one part; monotone, every delta positive):

| the step contains | ms/step | the part just added | % of base |
|---|---|---|---|
| stepper skeleton only (`norhs_noprop`) | 1.12 | scan machinery + `_replace` state updates | 2.5 |
| + propagator apply (`norhs`) | 1.79 | diagonal `exp(L·τ)` apply | 1.5 |
| + the FD-z linear term (`nonlin`) | 7.52 | z stencil + halo + field swap | 12.8 |
| + the gradient/nonlinear pipeline = **base** | **44.60** | everything below | 83.1 |

Inside the pipeline, by subtraction from base:

| part | ms/step | % of base | op profile |
|---|---|---|---|
| inverse transforms (8 per stage, `noifft`) | 26.33 | 59.0 | 72.8 |
| forward transform (2 fields, one op per stage, `nofft`) | 6.09 | 13.7 | 7.8 |
| *both* (`notrans`) | 32.99 | 74.0 | 80.7 |
| k-space gradient multiply `i·k·f̂` (`nogradk`) | 0.11 | 0.3 | 4.8 |
| bracket ALU (`nobracket`, all operands still live) | −0.07 | −0.2 | 2.3 |
| z stencil + halo (from the ladder) | 5.73 | 12.8 | 10.6 |
| propagator (from the ladder) | 0.67 | 1.5 | 0.1 |
| stepper skeleton (from the ladder) | 1.12 | 2.5 | 1.0 |
| **residual** (dealias/`inv_ksq` scaling, the stacks, real-space traffic) | 3.97 | **8.9** | 0.5 (NL) |

Rows sum to 100.0% with the residual carried explicitly. The residual and the two arithmetic
rows are the one place the methods disagree, and they disagree consistently: the ablation
credits 0.1% to the gradient multiply and the bracket where the profile credits 7.1%, and the
ablation's 8.9% residual is that same work. The stand-ins keep every read and only cheapen the
arithmetic (a complex scalar multiply instead of an `i·k_perp` broadcast array; three adds
instead of two multiplies and a subtract), so their deltas are lower bounds on what the fused
kernels actually spend there. Read the combined result as: transforms **74–81%**, z stencil +
halo **11–13%**, gradient and bracket arithmetic **0–7%**, propagator **≤1.5%**, stepper
**1–2.5%**.

### RMHD z_spectral lsrk54, ν=η, hoisted — 67.40 ms/step

| the step contains | ms/step | the part just added | % of base |
|---|---|---|---|
| stepper skeleton (`norhs_noprop`) | 1.10 | scan + state updates | 1.6 |
| + propagator apply (`norhs`) | 5.35 | separable apply (2 complex mults + the `i` swap per stage) | 6.3 |
| + the pipeline = **base** | **67.40** | | 92.1 |

`nonlin` (5.34) reproduces `norhs` (5.35) to within noise, as it must: with no
finite-difference-z term there is nothing between the two.

| part | ms/step | % of base | op profile |
|---|---|---|---|
| inverse transforms (`noifft`) | 48.02 | 71.2 | 73.9 |
| forward transform (`nofft`) | 11.42 | 16.9 | 11.5 |
| *both* (`notrans`) | 56.86 | 84.4 | 85.4 |
| propagator apply | 4.25 (build-up) / 9.23 (`noprop` subtraction) | 6.3–13.7 | 11.5 |
| hoisted stage-coefficient formation (`noexpform`) | −0.02 | 0.0 | 0.03 |
| k-space gradient multiply / bracket ALU | ≈ 0 | 0.0 | 2.0 |
| stepper skeleton | 1.10 | 1.6 | 0.5 |
| **residual** | 5.19 | **7.7** | 0.6 (NL) |

**The z_spectral premium is now the transform, not the propagator.** At this grid the
z_spectral step is 1.51× the FD-z step (67.40 vs 44.60 ms), and the gap decomposes as
transforms +23.9 ms (the 3-D `rfftn` over (z,x,y) costs log₂(nz·nx·ny) where the FD-z mode
pays nz × log₂(nx·ny)), propagator apply +3.6 ms, minus the 5.7 ms of z stencil and halo the
spectral mode does not run — **+21.7 ms predicted against +22.8 ms measured**. That inverts
the pre-Z1 finding recorded above ("of the +37 ms, 33.9 ms is the putzer2 matrix
exponential"): the separable backend has taken the propagator out of the accounting on both
axes, and what is left of the premium is irreducible transform work.

### GDI 2D 256² lsrk33, putzer2, hoisted — 2.50 ms/step

| the step contains | ms/step | the part just added | % of base |
|---|---|---|---|
| stepper skeleton (`norhs_noprop`) | 0.08 | scan + state updates | 3.2 |
| + propagator apply and hoisted formation (`norhs`) | 0.55 | putzer2 2×2 apply + the once-per-block `_coeffs` | 18.8 |
| + the pipeline = **base** | **2.50** | | 78.2 |

| part | ms/step | % of base | op profile |
|---|---|---|---|
| inverse transforms (`noifft`) | 1.19 | 47.8 | 61.7 |
| *both* transforms (`notrans`) | 1.70 | 68.2 | 74.7 |
| forward transform (by difference) | 0.51 | 20.4 | 13.0 |
| putzer2 apply | ≈ 0.40 | 16.0 | 15.3 |
| hoisted `_coeffs` formation (`noexpform`) | 0.07 | 2.8 | 4.1 |
| bracket ALU / gradient multiply | ≈ 0 | 0.0 | 2.5 |
| stepper skeleton | 0.08 | 3.2 | 1.5 |
| **residual** | 0.26 | **10.4** | 1.7 (NL) |

The direct forward-transform stand-in is invalid at this shape — `nofft` lands **17% above**
base, because a 256-wide strided reduction on XLA:CPU costs more than the `rfft2` it replaces.
The 0.51 ms above is `notrans` minus `noifft`, and the op profile's 13.0% is the independent
confirmation that the forward transform is real work here, not a measurement artifact.

`hoist_propagator` on this path is worth **1.80×** in time (2.50 ms/step hoisted against 4.50
unhoisted, `nblock=10`) and costs **nothing** in memory on this backend and grid: 41.130 u
hoisted against 41.091 u unhoisted. That is not what the plan's §1 note predicts ("cost of
True on putzer2: 4 complex arrays of L's full shape per stage") and not what the GPU shows —
the committed `bench/memory_probe_p100_postFZ_fp32.json` has `gdi2d_1024_lsrk33` at 45.244 u
hoisted against 38.242 u unhoisted, a real +7.0 u. The arena tables explain the CPU result:
unhoisted, XLA keeps a 4.000 u copy of the dense `lin.L` plus seven 1 u `_coeffs` lanes live
inside the stage scan, and those cost as much as the twelve 1 u-equivalents of hoisted stage
coefficients they replace. The memory price of hoisting is a scheduler property, not an
arithmetic one; size it on the GPU, per the plan's §0.5.

### Adaptive `z_spectral` lsrk33 (`cfl_every=1`, nothing frozen) — 43.26 ms/step

Recorded because it is the one production path where no propagator work is hoisted. Against
the same scheme at fixed dt with hoisting (38.87 ms/step) the adaptive step costs **+11.3%**.

| part | ms/step | % of base | op profile |
|---|---|---|---|
| transforms (`notrans`) | 35.83 | 82.8 | 80.3 |
| per-stage separable coefficient formation (`noexpform`) | 5.43 | 12.6 | 12.4 |
| propagator apply + formation (`noprop`, a subtraction row: over-counts) | 8.11 | 18.7 | 14.1 |
| CFL dt: reduction, `allreduce_max`, the `minimum` (`nocfl`) | 1.51 | 3.5 | 0.4 |
| stepper skeleton, bracket, gradient multiply, residual | ≈ 0.5 | ≈ 1 | 5.1 |

The three exp/cos/sin evaluations per step cost **1.8 ms per stage** here (5.43 ms over three
stages), and hoisting removes 4.4 of those 5.4 ms — the +11.3% above. The putzer2 backend paid
~9 ms per stage for the same job at a comparable grid (the 2026-08-19 ablation above); that
ratio is the separable operator's whole point. `nocfl` (3.5%) and the profile's `SETDT` (0.4%)
bracket the adaptive dt itself: the ablation also removes the dependency that forces `grads`
to be materialised before dt is known, so 0.4–3.5% is the honest range.

### Two things the ablation ladder cannot measure

**Removing a consumer can make the step slower.** Four FD-z variants land *above* the
baseline: `noexpform` +28.8%, `nofdlin` +20.2%, `nozarith` +13.9%, `noprop` +8.5% (and GDI's
`nofft` +17.0%). These are not noise — the base row's round scatter is 1.7% and these repeat
across all five rounds. XLA re-fuses the RHS around whatever is left, and on FD-z the schedule
it finds without the z-stencil's materialised halo slab, or without a real `exp(L·τ)` to
multiply by, is worse than the production one. Any part whose ablation is non-monotone must be
read from the build-up ladder or the op profile, never from base-minus-variant; that is the
concrete form of the warning recorded above ("isolated timings mislead here, in both
directions").

**`GRAD_CHUNK > 1` loses on both axes on XLA CPU**, at every path measured — it is slower
*and* larger, which settles the CPU half of F1's deferred trade:

| path | `GRAD_CHUNK=1` | `=2` | `=4` |
|---|---|---|---|
| RMHD FD-z lsrk54 | 44.2 ms, 22.66 u | 67.9 ms, 24.66 u | 75.5 ms, 28.60 u |
| RMHD z_spectral lsrk54 | 67.4 ms, 18.42 u | 78.5 ms, 21.39 u | 89.6 ms, 25.33 u |
| GDI 2D 256² lsrk33 | 2.50 ms, 41.13 u | 3.69 ms, 43.11 u | 3.85 ms, 46.10 u |

The intuition that batching the inverse transforms should pay — the op profile shows the
per-component *inverse* transform costing ~3× the per-component *forward* one, and the forward
is the batched op — is wrong: batching the inverses makes them worse, so the asymmetry belongs
to `irfft2`/`irfftn` itself and not to how many components share an op. Whether the GPU agrees
is still the open half of the trade.

### P100 addendum (512²×128 fp32, `bench/step_accounting_p100.json`)

The same driver on the Kaggle P100 (3 rounds, nrep 21; memory totals reproduce the
committed postFZ probe rows exactly: 20.096 / 17.309 / 45.244 u). Time shares from the
same ablation vocabulary, minima over rounds:

| part | FD-z lsrk54 (317.9 ms) | z_spec lsrk54 (294.3 ms) | GDI-2D lsrk33 (4.8 ms) |
|---|---|---|---|
| both transforms (`notrans`) | 52.6% | **88.7%** | 79.3% |
| inverse transforms (`noifft`) | 43.3% | 76.7% | 68.5% |
| z stencil + FD term (`nozarith`/`nofdlin`) | 16.3–22.7% | — | — |
| propagator | ~2.4% | ≤0 (non-monotone) | ~6% |
| RHS in total (`nonlin`) | 87.2% | 92.4% | 81.8% |

The GPU agrees with the CPU verdict — transform-bound everywhere, the propagator gone
from the accounting — with z_spectral even more transform-dominated (88.7%) and the
whole z_spectral step FASTER than FD-z (294 vs 318 ms, no stencil/halo to pay).
Non-monotone ablations occur on GPU too (`nofft` on FD-z lands 10% ABOVE base,
`noprop` on z_spectral 5% above): the same read-from-the-ladder rule applies. The
forced-case and snapshot-peak GPU numbers (`bench/step_accounting_p100_special.json`)
settle the production-sizing questions: (i) elsasser forcing, +2.41 u on CPU, costs
~+0.4 u on the P100 (`zspec_forced` totals 17.69 u at 128²×32) — the scheduler absorbs
the O-U temporaries; (ii) the snapshot sequence (block → `save_snapshot` → block) puts
the device high-water at 19.81 u, reached in the step AFTER the save — checkpointing
never sets the peak. Together: the max isotropic fp32 z_spectral box on a 16 GB P100 is
**576³ with forcing and snapshots on** (~14 GB peak), with 512³ leaving ~5 GB of margin.

### Reproduce

The whole accounting is packaged as the git-tracked driver `bench/step_accounting.py`, with a
lugus-compatible `main(**kwargs)` and `memory` / `timing` / `both` modes; it carries the same
six paths (the three above, the adaptive case, the forced case and the snapshot-peak
sequence) and reproduces every total in §1 exactly, with `scope_neutral=true` on each.

```
python bench/step_accounting.py --mode memory --profile laptop
python bench/step_accounting.py --mode timing --profile laptop --rounds 5   # quiet machine
python bench/step_accounting.py --mode memory --paths zspec_forced,snapshot_peak
python bench/step_accounting.py --mode both --profile p100 --precision 32 --rounds 3
```

The scratchpad scripts the 2026-08-20 numbers came from are listed in `README_accounting.md`;
`S` is that directory. Nothing under `taranis/` is modified by any of them.

```
# memory: compile-only dump, then the lane / argument tables
python $S/acct_dump.py  rmhd_fdz_128x32_lsrk54 $S/acct_fdz
python $S/acct_scope.py rmhd_fdz_128x32_lsrk54 $S/acct_fdz_sc $S/acct_fdz   # asserts MATCH
python $S/acct_table.py $S/acct_fdz_sc 0.4
python $S/acct_args.py  $S/acct_fdz_sc
#   (labels: rmhd_zspec_128x32_lsrk54_hoist1, gdi2d_256_lsrk33, gdi2d_256_lsrk33_hoist0)

# time: one variant per process, then the interleaved campaign and its reduction
python $S/acct_time.py rmhd_fdz_128x32_lsrk54 noifft 21
zsh    $S/acct_ladder.sh          # first pass, one round
zsh    $S/acct_campaign.sh        # 5 interleaved rounds -> acct_campaign.jsonl
python $S/acct_analyze.py $S/acct_campaign.jsonl

# op-level profile cross-check
python $S/acct_profscope.py rmhd_fdz_128x32_lsrk54 /tmp/acct_pf_fdz
```

Raw results: `acct_ladder.jsonl`, `acct_campaign.jsonl` (240 measurements, 5 rounds),
`acct_profiles.txt`, `acct_fdz_table.txt`, `acct_zspec_table.txt`, `acct_gdi_table.txt`,
`acct_gdi0_table.txt`.

Variant vocabulary in `acct_time.py`: `noifft`/`nofft`/`notrans` (transforms → a shape- and
read-preserving reduction), `noifft_dead`/`nobracket_dead` (crude twins that also dead-code
their producer, i.e. upper bounds), `nogradk` (the `i·k_perp` array multiply → a complex
scalar), `nobracket` (2 mul + 1 sub → 3 adds, all operands live), `nozarith` (the 4th-order
stencils → two of their operands, halo and concatenate kept), `nohalo`, `nofdlin`/`nonlin`
(a term → zeros), `noprop` (`ExpOp.apply` → identity), `noexpform` (the operator's `exp_op` →
same-shaped constants), `norhs`/`norhs_noprop`, `nocfl`, `chunk2`/`chunk4`.

## Known, not done

`run.py` calls `mngr.wait_until_finished()` immediately after every `save_snapshot`, which
defeats orbax's asynchronous save: checkpoint I/O is serialized with compute rather than
overlapping the next block of steps. Waiting lazily instead — before the *next* save and
at the end of the run — was planned and never implemented. It needs care around
`max_to_keep` deletion.

## Production guidance

**CPU clusters:** `mpi4jax`, fp64, `forcing_norm_per_step=True`, `cfl_every` 10–20 at
≥128 ranks from developed states.

**Savio GPU:** `comm_backend="jax"`, fp32 workloads, sizes per the cost table above, and
`export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"` on anything multi-GPU
(1.31× at 16 GPUs; ~2% at 4, so it is never worth omitting). `lsrk_scan` stays at its
default True on this backend.

**z_spectral (single process):** at ν = η (equal `diss` entries) the separable backend
(Z1, 2026-08-20) makes every step cheap — adaptive `cfl_every=1` included — at ~20 u
total with nothing scheme-dependent stored; no hoisting trade-off to think about (the
hoisted stage ops are ≤0.1 u and on by default). At ν ≠ η the putzer2 guidance still
applies: run `cfl_every > 1` or fixed dt so hoisting amortises the coefficient cost, and
budget 4·nstage complex full-grid arrays for the hoisted ops (or turn `hoist_propagator`
off on a memory-bound grid).

**fp64 GPU production** needs full-rate-fp64 hardware. Verified candidates as of
2026-07-27: NASA HECC Cabeus (A100 NVLink, plus GH200 nodes), NSF ACCESS DeltaAI, TACC
Vista. Note a quoted state size is aggregate, not per-GPU — a 270 GB state fits on one
4-GPU A100/GH200 node.
