# Phase 3 results — GPU backends for jax_rmhd

Closes PERFORMANCE_PLAN.md Phase 3 (T8 GPU baseline, T9 jax-native GPU backend).
Execution detail and per-job logs live in PHASE3_PLAN.md; this file is the consolidated
outcome. All Savio jobs 2026-07-25 → 2026-07-27, account fc_kawturb.

## What was built

`comm_backend="jax"`: a shard_map/NCCL backend behind the Phase-2 comms.py abstraction.
One process per GPU; mpi4py stays as the control plane (init, rank discovery,
params.save, snapshot-index broadcast); the three device-side ops switch transport —
`lax.ppermute` for the 2-wide z halo, `psum`/`pmax` for the reductions — inside a
shard_map context provided by the jitted steppers over a 1D z mesh of global sharded
arrays. Physics code is untouched and sees the same local `(nfields, nz_local, nkx,
nky)` shards as the mpi4jax path, which remains the default and is verified
bit-identical to pre-Phase-3. Checkpointing under the jax backend is orbax-native
global (one shared flat `snap_path/<step>/` directory, global arrays, per-process shard
writes) — a return to the code's original pre-MPI design; the mpi4jax per-rank layout
is unchanged and canonical. `snapshot_layout()` auto-detects flat / per-rank / legacy
trees; cross-backend restarts work in both directions.

## Correctness

- Local (no GPU): `tests/test_backend_jax.py`, 43 checks on 4 forced host devices at
  fp64 — fields vs serial reference rel 1.3e-16, halo-stencil orientation check
  (verified sensitive: flipped ppermute fails at O(1)), save/restore roundtrips 0.0,
  cross-backend and pre-Phase-3 legacy restores 0.0, layout detection with decoy tmp
  dirs, writer-mixing guards. Verified against orbax 0.12.1 (Savio's version).
- Hardware gate (job 35861902, 4×A5000, fp64): all 5 phases ALL PASS — jax/NCCL fresh
  run matches mpi4jax to ~3e-16; cross-backend restarts agree both directions to
  ~1e-15 with exact t and energies.
- mpi4jax path: bit-identical to HEAD (byte-equal orbax _METADATA, max|diff| = 0.0
  on both steppers, all cfl_every/lsrk variants).

## Benchmarks

### Single node — 4×A5000 (savio4_gpu, fp32, 512²×128, forced+nps; jobs 35861931/34)

| GPUs | mpi4jax | jax/NCCL | jax advantage |
|---|---|---|---|
| 1 | 294 ms/step | 310 | −5% |
| 2 | 176 | 158 | +11% |
| 4 | 126 | 77.0 | +63% |
| 4, unrolled | 104 | 68.5 | +52% |

Scaling 1→4 GPUs: jax ~4.0× (ideal), mpi4jax ~2.3×. The −5% at 1 GPU is fixed
shard_map/global-array overhead; irrelevant once communication exists.

### Multi-node — 16×GTX2080TI over 4 nodes (savio3_gpu, fp32, 512²×256; job 35895464)

| GPUs (nodes) | mpi4jax | jax/NCCL | jax advantage |
|---|---|---|---|
| 4 (1) | 284 ms/step | 200 | 1.4× |
| 8 (2) | 204 | 102 | 2.0× |
| 16 (4) | 165 | 48 | 3.4× |

Scaling 4→16 GPUs: **jax 4.15× (ideal+), mpi4jax 1.72×** — and the jax curve holds
across node boundaries on plain TCP (no InfiniBand userspace, PCIe P2P disabled).
This is the multi-node verdict, achieved under every Savio handicap; on NVLink/IB
hardware the margin only grows. (Substituted for the A40 job, which pends ~a week for
the full FCA pool; same question, ~4× cheaper.)

### CPU reference — savio3, mpi4jax, fp64 (job 35893732)

Strong (256²×256): 1331 / 1363 / 762 / 434 ms/step at 16/32/64/128 ranks. Within a
node, cores saturate memory bandwidth by ~16 (s32 == s16); across nodes ~88% per
doubling. Weak (256²×4 per rank): 370 / 750 / 762 / 771 — flat from 1→4 nodes (97%),
the production regime.

### Hardware economics (identical grid + precision: fp32, 512²×128)

| Hardware | ms/step | SU/hr | SU per 1000 steps |
|---|---|---|---|
| 1 savio3 node (32c) | 1563 | 32 | 13.9 |
| 2 nodes (grid's CPU ceiling) | 866 | 64 | 15.4 |
| 1 × A5000 | 294 | 18.7 | 1.53 |
| 4 × A5000, mpi4jax | 126 | 74.7 | 2.61 |
| 4 × A5000, jax backend | 77 | 74.7 | 1.60 |

CPU cost per step *rises* under strong scaling (waiting cores still bill); the jax
backend holds it flat while quadrupling speed. Net: **~9× cheaper per timestep on
GPUs at fp32**, bounded by the grid's CPU rank ceiling (nz caps decomposition at 64
ranks here). On Savio's workstation GPUs (fp64 = 1/32 fp32), fp64 production stays on
CPU; on full-rate-fp64 hardware (A100/GH200/H100) the economics carry to production
precision. SU rates from the Savio scheduler-config table (2026-07-25).

## Verdicts (PERFORMANCE_PLAN gate rules)

- **T9 jax backend: KEEP.** +63% at 4 GPUs single-node, 3.4× at 16 GPUs multi-node,
  ideal scaling in both regimes; correctness bit-clean; mpi4jax untouched as CPU
  production backend.
- **T8 baseline: complete.** mpi4jax+GPU works but scales at 1.7–2.3× per 4× hardware
  (host-staged comm, F2 confirmed); Savio MPI is not CUDA-aware (probe job 35845686:
  built --without-cuda; CUDA_MPI=1 segfaults in UCX). Use mpi4jax-GPU only single-GPU
  or as a fallback.
- **T3 on GPU (lsrk_scan):** unroll helps mpi4jax-GPU consistently (+21–22%) and jax
  single-node (+12%) but hurts jax multi-node (−38%). Default stays `lsrk_scan=True`
  everywhere; treat unroll as a per-config tuning knob, benchmark per machine.
- **T7 halo_start hook: stays unregistered.** Neutral in every measured config
  (≤2%, sub-noise), including multi-node NCCL. Infrastructure retained.
- **Forcing/production flags:** unchanged from Phases 1–2 (nps default on, cfl_every
  guidance per PHASE2_PLAN).

## Cluster/environment findings (all baked into scripts + SAVIO_GPU_SETUP.md)

PCIe GPU P2P is broken on Savio GPU nodes (known issue per Savio staff; NCCL_P2P_DISABLE=1
is the documented workaround — `bench/nccl_repro.py` is a 60-second reproducer).
The anaconda3 module leaks base site-packages via PYTHONPATH into batch shells (shadows
the env's nvidia libs → "cuSPARSE not found"); GPU scripts unset PYTHONPATH and set
LD_LIBRARY_PATH over env nvidia/*/lib. jax.distributed mis-parses CUDA_VISIBLE_DEVICES
physical ids as ordinals (fixed in comms._local_device_ids). mpi4jax launches need
--gpu-bind=single:1; jax/NCCL launches must NOT use it (peer visibility). srun needs
--mpi=pmix. Transient cuInit failures can silently drop ranks to CPU — benchmarks now
abort via RMHD_REQUIRE_GPU=1. rdma-core userspace is incomplete (no mlx5 driver), so
multi-node traffic is TCP.

## Production guidance

CPU clusters: mpi4jax backend, unchanged (Phase 1/2 config: fp64, nps, cfl_every 10–20
on ≥128 ranks from developed states). Savio GPU: jax backend, fp32 workloads, sizes
per the SU table; expect ~an order of magnitude better SU/timestep than CPU. Target
machines for fp64 GPU production (verified 2026-07-27): NASA HECC Cabeus (128×4 A100
NVLink nodes + 350 GH200 nodes) or NSF ACCESS DeltaAI / TACC Vista (GH200). A "270 GB"
state is aggregate, not per-GPU — one 4-GPU A100/GH200 node holds it.

## Open items

V100×2 fp64 anchor job still queued (consistency check only; target hardware is
full-rate fp64). Savio ticket on ACS/P2P and rdma-core — chatbot confirmed the known issue; escalate to a human with
the reproducer if direct P2P is wanted. A40 normal-QoS job: cancel. Before merge: Savio CPU battery
(test_restart_resharding -n 2→4, test_advection -n 4) per the Phase 2 convention.
