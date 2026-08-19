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
| `lsrk_scan=False` (unrolled) | +21% mpi4jax-GPU, +12% jax single-node, **−38% jax multi-node** | per-machine knob, benchmark it |
| `forcing_shell_noise` | faster single-device, ~5% *slower* on Savio CPU at 32 ranks | opt-in; revisit on GPU |
| `halo_start` | neutral (≤2%, sub-noise) everywhere measured | see below |

`cfl_every > 1` costs one extra standalone gradient evaluation per block, because the
stage-0 RHS no longer doubles as the dt source. At 32 ranks that cancels the saving at
N=5; it only pays once the allreduce is expensive.

**The `cfl_every` hazard is real:** from a quiescent forced start the CFL dt collapses by
~10× within a few steps of spin-up, so a frozen dt NaNs — measured, N=20 dies by t≈2 while
N=5 survives. Use N>1 only from developed states.

## Negative results

Recorded so they are not re-investigated.

- **Early halo issue (`halo_start`) buys nothing measurable.** On mpi4jax there is no fp64
  win at any rank count (−0.6% at 32, sub-noise at 128) because the token chain serializes
  communication with compute regardless. On the jax backend it is neutral on Savio at
  bench sizes, including multi-node NCCL. The hook is kept and enabled for `"jax"` because
  the answer plausibly changes with a different scheme or on NVLink/IB hardware, but it is
  not currently earning its keep.
- **Unrolled LSRK is not a universal win.** It helps every configuration except the one
  that matters most for scaling, jax multi-node, where it costs 38%.
- **shard_map on CPU** — measured slow, which is why `mpi4jax` remains the CPU backend.

## Test particles overhead (2D, laptop, 2026-08-18)

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

Deferred (Alfred, 2026-08-18): gather optimization would attack only the O(N) part
(est. 1.5–2×) — revisit if A3's 2D science runs at 3×32768 particles feel slow. Two
items flagged: (i) gather-side reorganization (`interp.gather`: share cell/weight
computation across E and B, `jnp.take` on flat indices, cast samples not grids); (ii)
reuse the stepper's stage-1 gradients in `particle_fields` (removes 4 of the 6 fixed
transforms) — the bigger lever, but changes the stepper contract, so not Phase A.

## Known, not done

`run.py` calls `mngr.wait_until_finished()` immediately after every `save_snapshot`, which
defeats orbax's asynchronous save: checkpoint I/O is serialized with compute rather than
overlapping the next block of steps. Waiting lazily instead — before the *next* save and
at the end of the run — was planned and never implemented. It needs care around
`max_to_keep` deletion.

## Production guidance

**CPU clusters:** `mpi4jax`, fp64, `forcing_norm_per_step=True`, `cfl_every` 10–20 at
≥128 ranks from developed states.

**Savio GPU:** `comm_backend="jax"`, fp32 workloads, sizes per the cost table above.

**fp64 GPU production** needs full-rate-fp64 hardware. Verified candidates as of
2026-07-27: NASA HECC Cabeus (A100 NVLink, plus GH200 nodes), NSF ACCESS DeltaAI, TACC
Vista. Note a quoted state size is aggregate, not per-GPU — a 270 GB state fits on one
4-GPU A100/GH200 node.
