# Phase 3 execution plan — GPU backends (agent handoff)

Expands PERFORMANCE_PLAN.md Phase 3 (T8 GPU baseline → T9 shard_map/NCCL backend) into
concrete tasks for Opus agents. PERFORMANCE_PLAN.md findings (F2 especially) and
CLAUDE.md invariants are binding. Phase 2 outcome that matters here: comms.py exists
(T5), `halo_start_func` hook + `rmhd.halo_start` are pre-staged for a backend with real
overlap (T7 reverted on CPU), `lsrk_scan=False` unrolled loop is the GPU candidate (T3).

USER DECISION (2026-07-25): T9 is built NOW, not gated on T8's numbers — the GPU
benchmark round decides keep-vs-revert afterwards (same pattern as T7). Rationale: the
submit-and-wait loop costs days; agent implementation time is cheap.

## Savio GPU facts (verified against docs-research-it.berkeley.edu, 2026-07-25)

| Partition | GPU | GPUs/node | CPU:GPU | FCA QoS | FCA regular-priority pool |
|---|---|---|---|---|---|
| savio4_gpu | A5000 24 GB | 8 | 4:1 | `a5k_gpu4_normal` | 136 GPUs, but **16 CPUs/user = 4 GPUs max per user** |
| savio4_gpu | L40 46 GB | 8 | 8:1 | `savio_lowprio` only | — |
| savio3_gpu | V100 32 GB | 2 | 4:1 | `v100_gpu3_normal` | 2 GPUs (the only good-fp64 GPUs; A5000/A40/L40 fp64 ≈ 1/32 fp32) |
| savio3_gpu | A40 48 GB | 2 or 4 | 8:1 | `a40_gpu3_normal` | 16 GPUs — cluster-wide pool, a 16-GPU job waits for ALL of them |
| savio3_gpu | 2080Ti 11 GB | 4 | 2:1 | `gtx2080_gpu3_normal` | 28 GPUs |

- GPU partitions are **per-core scheduled** (not per-node like savio3): request exactly
  `cpus-per-task = ratio × GPUs`, never a whole node. SU rate 3.67/core-hr (savio3_gpu),
  4.67 (savio4_gpu) → 16×A40 ≈ 470 SU/hr: keep scale jobs ≤ 1 hr.
- `--gres=gpu:<TYPE>:<n>` with the explicit type is REQUIRED for FCA jobs, and A40/V100
  additionally require the explicit `--qos`. Missing type → `QOSMinGRES` pend forever.
- Do not touch `CUDA_VISIBLE_DEVICES`; Slurm sets it per job/task.
- Benchmark targets this round: **A5000×{1,2,4}** (single node, fp32, low wait — the
  workhorse), **A40×{4,8,16}** (multi-node NCCL-over-IB scaling — the NASA-cluster
  proxy; 16 = 4–8 nodes), **V100×2 fp64** (anchor: fp32 flatters comm ~3×, Phase 1
  lesson). Account `fc_kawturb` everywhere.

## Ground rules for every agent (unchanged from Phase 2 + GPU additions)

- Obey CLAUDE.md invariants (donation, `_replace` construction, rfft2 reality rows,
  static-params plain-python branching, energy normalization, kgrid–params binding,
  forcing_scale lifecycle, checkpoint structure).
- Comments: ~1 line per new function, ~1 line per change. No walls of text. NO git
  commits — leave the working tree for review. Branch: `performance`.
- Sandbox has NO MPI and NO GPU. Local verification uses `tests/local_mpi_stub.py`
  (fakes mpi4py/mpi4jax single-rank incl. dims=3):
  `MPLBACKEND=Agg PYTHONPATH=.:tests python3 -c "import local_mpi_stub, runpy;
  runpy.run_path('tests/<script>.py', run_name='__main__')"`.
  Keep every run < 40 s (bash calls hard-capped at 45 s); grids ≤ 64², nz ≤ 16; fp64
  via `RMHD_PRECISION=64`. NEVER copy the repo tree (multi-GB); extract old versions
  with `git show <ref>:<path>` selectively.
- **GPU-less T9 testing trick**: `XLA_FLAGS=--xla_force_host_platform_device_count=4`
  gives 4 fake CPU devices in ONE process — mesh/shard_map/ppermute/psum all exercise
  the real code path without MPI or GPUs. This is the primary local correctness tool
  for the jax backend.
- Local battery after every task: `tests/test_forcing_smoke.py`,
  `tests/test_forcing_norm_per_step.py` (both "ALL PASS"), `tests/test_dissipation.py`
  (completes). Savio battery before merge: `test_restart_resharding.py` (-n 2 → -n 4),
  `test_advection.py` (-n 4), both per backend where applicable.
- Savio jobs: agents PREPARE sbatch scripts; the user submits and drops `.out` files
  back. One submission must answer one question. Every result line must print `pkg=`
  (RMHD_PKG mechanism from bench_phase1.py — never PYTHONPATH) and the jax platform +
  device list.

## A1 — T8: GPU environment + mpi4jax GPU baseline (prep; user runs the jobs)

Deliverables, all documentation/scripts — no solver-code changes:

1. **Rewrite `SAVIO_GPU_SETUP.md`** as the complete from-scratch guide (env does NOT
   exist yet): conda env `jax_gpu` (python 3.11, `pip install -U "jax[cuda12]"`,
   mpi4py built against the system `openmpi` module, mpi4jax, orbax-checkpoint,
   tensorstore, numpy, matplotlib, `pip install -e ~/jax_rmhd`); login-node import
   sanity check; srun single-GPU device-visibility check. Add the multi-GPU sections:
   - **CUDA-aware MPI audit** (this decides how T8 is interpreted): a small probe
     script + instructions — `ompi_info --parameters | grep -i cuda`, and a 2-rank
     mpi4jax GPU-buffer sendrecv probe run on a GPU node. Document both outcomes:
     if the `openmpi` module is not CUDA-aware, `MPI4JAX_USE_CUDA_MPI=0` (host
     staging, the F2(a) cost we're measuring) is the baseline — record which one ran.
   - **GPU binding**: one MPI rank per GPU via Slurm (`--ntasks-per-node = #GPUs`,
     `--gpu-bind` or `CUDA_VISIBLE_DEVICES` as Slurm scopes it per task — verify with
     a `jax.devices()` print per rank; each rank must see exactly 1 distinct GPU).
   - The per-core scheduling / QoS / gres table above, SU cost note, and the
     fp64-only-on-V100 warning.
2. **Bench harness**: extend `bench/bench_phase1.py` (or a thin `bench_phase3.py`
   wrapper if cleaner) with: platform/device reporting per rank (`jax.devices()`,
   backend name) printed into the results line; a `backend=<mpi4jax|jax>` switch
   (jax path exercised by A2's backend); grid-size flags big enough for GPU
   utilization (perp ≥ 256², nz per the case) while fitting 11–48 GB.
3. **Sbatch scripts** (`slurms/`): `bench_phase3_a5000.sh` (savio4_gpu, 1 node,
   cases 1/2/4 GPU, fp32, forced 3D + one unforced pair, 2 passes),
   `bench_phase3_a40_scale.sh` (savio3_gpu, 4/8/16 A40 across nodes, fp32, ≤ 1 hr),
   `bench_phase3_v100_fp64.sh` (2×V100, fp64, the anchor). Each runs BOTH backends
   once A2 lands (mpi4jax first; jax rows behind a flag so the script still runs if
   A2's backend is reverted). Include the `lsrk_scan=False` unrolled-loop case on GPU
   (T3's GPU candidate) and a `halo_start` on/off pair (T7's hook — real overlap is
   plausible on NCCL).
4. **Profiling**: one `--profile` case per script emitting a `jax.profiler` trace to
   quantify F2(b) per-call stream syncs, with a one-paragraph how-to-read note.

Acceptance: local battery still passes (nothing behavioral changed); scripts
lint-clean under `bash -n`; every case prints pkg=, backend, devices, ms/step.

## A2 — T9: `comm_backend="jax"` (shard_map/NCCL) — the big one

Design constraints (pinned; deviations need a written reason in the status line):

- **Control plane stays MPI.** mpi4py init, rank/size discovery, `params.save`
  collectives, orbax per-rank checkpointing, snapshot-index broadcast — all unchanged.
  Only the three DEVICE-side ops in comms.py (halo_exchange, allreduce_sum,
  allreduce_max) switch transport. Launch remains `mpirun`/`srun` with one process
  per GPU. `jax.distributed.initialize(coordinator, num_processes=size,
  process_id=rank)` with the coordinator address obtained via an mpi4py broadcast
  from rank 0 (do not depend on Slurm env autodetection — must also work under plain
  mpirun).
- **Physics code untouched.** Inside `shard_map` every array keeps its local
  `(nfields, nz_local, nkx, nky)` shape — identical to what mpi4jax ranks see today.
  comms.py's jax branch: `lax.ppermute` (both directions) for the 2-wide z halo,
  `lax.psum`/`lax.pmax` over the z mesh axis for the reductions. These primitives are
  only valid inside shard_map — the jit wrapper in run.py provides that context for
  the "jax" backend; plain-python dispatch on the static `params.comm_backend` as now.
- Mesh: 1D over z, one device per process (multi-controller). `check_rep=False`
  where required; forcing_state/forcing_key are replicated (identical on all ranks by
  construction — preserve that invariant, don't shard them).
- mpi4jax stays the default and the CPU-cluster backend. `Parameters.__init__`
  validates `comm_backend in ("mpi4jax", "jax")`; "jax" + dims==2 is an error (no
  decomposition to map).
- Buffer donation, `dt_override`/cfl_every hoisting, `forcing_norm_per_step`, and the
  snapshot-cadence semantics must all survive under the new backend — reuse the
  existing steppers, don't fork them.
- Re-register `halo_start_func=rmhd.halo_start` FOR THE JAX BACKEND ONLY (plain-python
  branch on params) — overlap is the point of NCCL; keep it None for mpi4jax per the
  Phase 2 measurement.
- Checkpointing: per-rank orbax managers keyed by MPI rank work unchanged IF each
  process can hand orbax a host-local array for its shard — verify, and if a global
  jax.Array needs explicit shard extraction before save, do it in snapshot_io behind
  `params.comm_backend`, keeping the on-disk layout IDENTICAL between backends
  (a run must be restartable across backends; add that to the correctness test).

**AMENDMENT (2026-07-26, user-approved).** Two Savio runs showed that per-rank orbax
managers cannot be made to work under `comm_backend="jax"`: at `process_count>1` orbax
rejects host-local `jax.Array`s outright, and scoping each rank's manager with
`MultiprocessingOptions(primary_host=rank, active_processes={rank})` — which verified
against orbax 0.11.39 — fails on Savio's orbax 0.12.1 (only rank 0 finalized its step;
ranks 1–3 stranded `*.orbax-checkpoint-tmp` dirs, then a `FileExistsError`, a ~5 min
barrier hang and an abort). The design is therefore replaced by orbax-native GLOBAL
checkpointing: ONE shared `CheckpointManager` over ONE shared directory, handed the
GLOBAL z-sharded `jax.Array`s directly. Consequently the "identical on-disk layout"
constraint above is **withdrawn** — layouts may differ per backend. The binding
requirements are now: (i) cross-backend restartability in BOTH directions, (ii)
pre-Phase-3 snapshot dirs still restorable via the existing path/repair, (iii) the
mpi4jax writer path BIT-IDENTICAL to HEAD (its per-rank layout is the canonical
production format and stays untouched).

Verification (local, no GPU): (a) battery passes with default backend — zero
regression; (b) `XLA_FLAGS=--xla_force_host_platform_device_count=4` single-process
test — new `tests/test_backend_jax.py`: same global 3D grid run (i) serial mpi4jax
path (stub, size=1) and (ii) jax backend over 4 forced host devices; fields must
match at fp64 to ~1e-14 (bitwise not required — different reduction orders), forcing
stream identical, energies match; include a restart roundtrip. (c) Prepared Savio
scripts: `slurms/test_backend_jax_gpu.sh` (4×A5000: mpi4jax vs jax backend same-seed
comparison + cross-backend restart) — user submits.

## A3 — adversarial review + fix round (mandatory, after A1+A2)

Fresh Opus agent, read-only against `git diff` of the Phase 3 work, brief as Phase 2's:
rank findings CRITICAL/MAJOR/MINOR; specifically probe: multi-controller init ordering
(MPI before jax.distributed? double-init guards?), ppermute edge correctness vs the
sendrecv tags/directions it replaces (left/right neighbor orientation!), reality rows
at ky=0/Nyquist after any new k-space op, donation vs global-array aliasing,
orbax layout identity across backends, forcing replication invariant, stub
compatibility, and that mpi4jax paths are bit-identical to pre-Phase-3. Runs the local
battery + the forced-host-device test itself. Then a fix round for CRITICAL/MAJOR.

## Benchmark gate (user submits, then we interpret)

fp64 (V100) and fp32 (A5000/A40) numbers decide, per PERFORMANCE_PLAN's rule —
keep what pays for itself on GPU, revert what doesn't:
- jax backend vs mpi4jax(+CUDA-MPI if available) at 2/4 GPUs single-node and 4/8/16
  A40 multi-node; scaling curves, not single points.
- `lsrk_scan` False-vs-True on GPU; `halo_start` on/off under the jax backend.
- If the jax backend loses everywhere, revert per the T7 rule and record the negative
  result here; the comms.py abstraction stays either way.

## Sequencing

A1 and A2 in parallel (disjoint files: A1 = SAVIO_GPU_SETUP.md, bench/, slurms/bench_*;
A2 = jax_rmhd/*, tests/test_backend_jax.py, slurms/test_backend_jax_gpu.sh) → A3 review
+ fixes → user creates env, runs probes + correctness job → benchmark round → gate.
Each agent updates ONLY its own status line below.

## Status

- A1 (T8 prep): DONE — `SAVIO_GPU_SETUP.md` rewritten (env build, per-rank GPU binding via
  srun `--gpu-bind`, CUDA-aware-MPI audit + `slurms/probe_cuda_mpi.sh`/`bench/probe_cuda_mpi.py`,
  QoS/gres/SU table, fp64-only-on-V100, profile how-to); `bench/bench_phase1.py` gained
  `nx<N>`/`nz<N>`, `backend=` (pass-through to `comm_backend`), `halo_early`, `--profile`, and
  per-rank platform/device reporting; `slurms/bench_phase3_{a5000,a40_scale,v100_fp64}.sh`
  written (jax-backend rows behind `RUN_JAX=1`). Local battery ALL PASS; all scripts `bash -n`
  clean. Awaiting user submission.
- A2 (T9 jax backend): DONE, pinned design followed — mpi4py control plane + ppermute/psum/pmax
  inside a 1D-z shard_map from the jitted steppers; state/kgrid become global z-sharded arrays,
  snapshot_io localizes before orbax so the on-disk layout is backend-independent. mpi4jax
  BIT-IDENTICAL (max|diff| 0.0, both steppers), battery unchanged, tests/test_backend_jax.py
  19/19 PASS on 4 forced host devices (fields rel 1.3e-16). Additions beyond the pinned list:
  comm_backend exempted from params.json's differing-record check (cross-backend restart),
  comms._local_device_ids GPU binding (reads CUDA_VISIBLE_DEVICES, never sets it), and
  tests/test_backend_jax_mpi.py as the driver for slurms/test_backend_jax_gpu.sh.
  Multi-host checkpointing, FINAL design (see the A2 AMENDMENT above; supersedes the per-rank
  manager attempts that failed in jobs 35845687 and 35852476): under comm_backend="jax"
  snapshot_manager_setup builds ONE shared CheckpointManager over ONE shared directory and
  save_snapshot hands orbax the GLOBAL z-sharded arrays unchanged — orbax's supported multihost
  path (its host-local guard only rejects fully-addressable arrays; typed PRNG keys are unwrapped
  to key_data and rewrapped on restore, so forcing_key needs no special handling). All the
  attempt-2 machinery is deleted (_mp_options, _new_manager, _to_host, localize-before-save).
  Layouts now differ by backend — jax: snap_path/<step>, mpi4jax: snap_path/<rank>/<step> — and
  snapshot_layout()/get_saved_steps()/load_snapshot detect which is which by the
  _CHECKPOINT_METADATA marker inside snap_path/<n> (immune to stranded *.orbax-checkpoint-tmp
  dirs). Reads never build a CheckpointManager: a bare StandardCheckpointHandler is barrier-free,
  which is what lets ranks read different directories (or different numbers of them) without
  deadlocking. Verified locally on 4 forced host devices at orbax 0.12.1: test_backend_jax.py
  40/40 PASS, mpi4jax writer byte-for-byte unchanged vs HEAD.
- A3 (review): DONE, 3 CRITICAL + 4 MAJOR found and FIXED. Criticals: (1)+(2) the jax backend
  could not run on >1 process at all — `comms.to_global`/`to_local` used `jax.device_put` onto /
  off a non-fully-addressable sharding (rejects orbax-restored committed arrays, and its
  multihost `assert_equal` raises TypeError on the PRNG-key leaf); both now build/extract via
  `make_array_from_single_device_arrays` / `addressable_shards`. (3) `bench_phase1.py backend=jax`
  jitted `block_of_steps` with no shard_map ("unbound axis name: z") — every `j*` row of all three
  bench scripts would have crashed; now wrapped in `comms.shard_call`. Majors: `halo_early` was a
  silent no-op under mpi4jax (backend gate) so the T7 on/off GPU pairs measured the baseline twice
  → `params.halo_start` override, all 4 (backend,flag) combos verified; `save_snapshot`'s
  `params=None` default now refuses a global/sharded state instead of forking the on-disk layout;
  jax.distributed coordinator port derived from SLURM_JOB_ID (per-core-scheduled GPU nodes are
  shared) + `is_initialized()` double-init guard and a pointed error on late bring-up;
  `slurms/probe_cuda_mpi.sh` asked for `--gres=gpu:A5000:1` for 2 tasks (gres is PER NODE — both
  ranks would have shared one GPU) and the A40 script's node arithmetic now derives from the
  allocation so the 8×2 fallback shape is a header-only edit. Verified: battery ALL PASS,
  test_dissipation completes, test_backend_jax.py 19/19 on 4 forced host devices (a deliberately
  flipped ppermute makes it fail at rel 8.7e-01, so the halo check is genuinely O(1)-sensitive),
  and mpi4jax fp64 A/B vs `git show HEAD` is BITWISE identical for scan/unroll/cfl_every=2.
  Unfixed MINOR: params.json keeps the *first* run's comm_backend (exempt from the diff check);
  the A40 16-GPU job takes all four 4-GPU A40 nodes (pend risk — 8×2 fallback documented) and with
  RUN_JAX=1 packs ~23 srun steps into 1 hr (submit the two backends as separate jobs).
