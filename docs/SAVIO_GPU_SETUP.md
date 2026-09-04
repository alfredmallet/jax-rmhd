# taranis on Savio GPUs — setup, running multi-GPU, benchmarking

Two paths through this file:

- **I want to run a simulation on several GPUs.** §0 → §5. Build the env once (§2), submit
  `slurms/forced_turbulence_multigpu.sh` (§0), read §4 before changing how it launches and
  §5 before writing your own driver. §6 is the symptom → cause → fix table.
- **I want to benchmark or audit the machine.** §7 onwards: the CUDA-aware-MPI audit of
  record, the Phase 3 benchmark jobs and how to read a profile. Companion to
  `plans/old/PHASE3_PLAN.md` and `plans/old/PERFORMANCE_PLAN.md`.

Setup for CPU nodes is a separate, simpler story: `docs/SAVIO_CPU_SETUP.md`. Measured
scaling numbers live in `docs/performance.md`; the numerics are in `docs/numerics.md`.

## 0. Quickstart

Once per account (§2 explains every line):

```bash
# login node
module purge && module load anaconda3 gcc openmpi
conda create -n jax_gpu python=3.11 -y && source activate jax_gpu
export PYTHONNOUSERSITE=1
pip install -U "jax[cuda12]"
MPICC=$(which mpicc) python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py
module load cuda
python -m pip install --no-cache-dir --no-binary=mpi4jax mpi4jax
pip install orbax-checkpoint tensorstore numpy matplotlib
cd ~/taranis && pip install -e .
```

Then a real run — forced 3D RMHD turbulence on 4 A5000s: the most a regular-FCA user can
hold on savio4_gpu (§1), on one node, so nothing here depends on inter-node bandwidth:

```bash
cd ~/taranis
sbatch slurms/forced_turbulence_multigpu.sh
squeue -u $USER
```

That script is the production template: it sets every environment variable §4 says it must,
and runs `examples/multigpu_forced_turbulence.py`, whose size and duration are environment
variables (`TARANIS_NX`, `TARANIS_NZ`, `TARANIS_TEND`, …) set in the job script rather than
edited into the python. **Resubmitting the same script continues the run** from the newest
snapshot, so a run longer than one walltime allocation is a chain of identical `sbatch`
calls.

The first thing to check in the `.out` is the per-rank line the driver prints before it
times anything:

```
[rank 0/4] backend=jax platform=cuda local_devices=[cuda:0] global_devices=4 precision=32 grid=512^2x128
[rank 1/4] backend=jax platform=cuda local_devices=[cuda:1] global_devices=4 precision=32 grid=512^2x128
```

Every rank must say `platform=cuda`, list exactly **one** local device, with **distinct**
ids across the ranks on a node, and report `global_devices` equal to the job's total GPU
count. Anything else means the launch was wrong, and no number from that job means
anything — §6 has the specific failures.

## 1. Which partition to ask for

Verified against docs-research-it.berkeley.edu on 2026-07-25.

| Partition | GPU | GPUs/node | CPUs per GPU | FCA QoS | Regular-priority pool |
|---|---|---|---|---|---|
| savio4_gpu | A5000 24 GB | 8 | 4 | `a5k_gpu4_normal` | 136 GPUs, but **16 CPUs/user cap = 4 GPUs max per user** |
| savio4_gpu | L40 46 GB | 8 | 8 | `savio_lowprio` only | — |
| savio3_gpu | V100 32 GB | 2 | 4 | `v100_gpu3_normal` | 2 GPUs — the only good-fp64 cards |
| savio3_gpu | A40 48 GB | 2 or 4 | 8 | `a40_gpu3_normal` | 16 GPUs cluster-wide (a 16-GPU job waits for all of them) |
| savio3_gpu | 2080Ti 11 GB | 4 | 2 | `gtx2080_gpu3_normal` | 28 GPUs (gres type string: `GTX2080TI`) |

Rules that will otherwise cost you a day of pending jobs:

- **GPU partitions are per-core scheduled**, unlike savio3's per-node charging. Request
  exactly `cpus-per-task × ntasks = (CPUs per GPU) × (number of GPUs)` — never a whole node.
- `--gres=gpu:<TYPE>:<n>` **with the explicit type is mandatory for FCA (`fc_*`) accounts**.
  Omitting the type pends forever with reason `QOSMinGRES`.
- A40 and V100 additionally require the explicit `--qos=` from the table. A5000 does too in
  practice (`a5k_gpu4_normal`); L40 is `savio_lowprio` only.
- `--gres` is **per node**. 16 A40 = `--nodes=4 --ntasks-per-node=4 --gres=gpu:A40:4`.
- One MPI task per GPU, always: `--ntasks` (or `--ntasks-per-node`) must equal the GPU count.
- SU rate: 3.67/core-hr on savio3_gpu, 4.67 on savio4_gpu. 16×A40 = 128 cores ≈ 470 SU/hr —
  keep scaling jobs at or under 1 hour of walltime.
- Account is `fc_kawturb` everywhere.
- **fp64 only on V100.** A5000/A40/L40/2080Ti are workstation-class: fp64 runs at ~1/32 of
  fp32. Run those at `TARANIS_PRECISION=32` and take fp64 numbers from the 2×V100 job. Phase 1's
  lesson still applies — fp32 flatters comm savings by ~3×, so keep/revert decisions get made
  on the fp64 anchor.

## 2. Build the `jax_gpu` env (login node, once)

```bash
module purge
module load anaconda3 gcc openmpi

conda create -n jax_gpu python=3.11 -y
source activate jax_gpu

# Block ~/.local user-site packages: they take precedence over the env's site-packages and a
# stray `pip install --user` mpi4py there silently satisfies pip and shadows the env's copy.
# If `python -c "import mpi4py; print(mpi4py.__file__)"` ever prints ~/.local/..., run
# `python -m pip uninstall mpi4py` (repeat for jax_cpu if it relied on that copy).
export PYTHONNOUSERSITE=1

# JAX's cuda12 wheel bundles its own CUDA/cuDNN runtime via nvidia-* packages, so no system
# cuda module is needed for JAX itself.
pip install -U "jax[cuda12]"

# mpi4py/mpi4jax (an optional `[mpi]` extra in pyproject.toml) are required for any
# multi-rank run, INCLUDING every comm_backend="jax" run: that backend's control plane
# (rank/size, params.save, orbax, index broadcasts) is still mpi4py, and only the three
# device ops become NCCL collectives. A single-rank run works serially without them
# (comm_backend="serial", auto-selected when they're absent), but install them anyway on
# Savio -- the launcher-mismatch guard (every rank reporting rank 0, below) needs a real
# mpi4py to detect. Build mpi4py FROM SOURCE against the loaded openmpi -- a prebuilt wheel
# may link a different MPI than the one `srun`/`mpirun` on the compute node actually
# launches, which is exactly that every-rank-reports-rank-0 symptom.
# --no-cache-dir: pip caches locally-built wheels; without it a rebuild after an MPI/toolchain
# change silently reinstalls the stale cached wheel.
MPICC=$(which mpicc) python -m pip install --no-cache-dir --no-binary=mpi4py mpi4py

# mpi4jax: MUST come after jax[cuda12] and mpi4py are in the env, and needs a CUDA toolkit
# visible AT BUILD TIME or setup.py silently skips the CUDA extension ("CUDA path not found
# (GPU extensions will not be built)" in the -v output) and GPU buffers stage through host
# memory forever. mpi4jax 0.9.x detects CUDA from (a) the nvidia-* pip packages jax[cuda12]
# installed, else (b) nvcc on PATH -- loading the cuda module covers (b) as belt-and-braces.
# The build compiles with mpicc (or set MPI4JAX_BUILD_MPICC), so keep gcc+openmpi loaded.
module load cuda   # 12.x; build-time only, jobs do NOT need it (libcudart is rpath-linked)
python -m pip install --no-cache-dir --no-binary=mpi4jax mpi4jax

pip install orbax-checkpoint tensorstore numpy matplotlib

# NOT `pip install -e ".[mpi]"` here: the extra would let pip resolve its own mpi4py/mpi4jax
# wheels, bypassing the from-source builds above (MPICC pinning, and mpi4jax needing jax +
# a visible CUDA toolkit at build time -- see the ordering notes above). Both are already
# satisfied, so plain `-e .` picks them up as-is.
cd ~/taranis && pip install -e .
```

Login-node sanity check (no GPU there — this only confirms imports and MPI wiring):

```bash
# every path must be under ~/.conda/envs/jax_gpu, NEVER ~/.local (user-site shadowing)
python -c "import mpi4py, mpi4jax; print(mpi4py.__file__); print(mpi4jax.__file__)"
python -c "import jax, orbax.checkpoint, tensorstore; print(jax.__version__)"
python -c "from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])"   # must name the module's openmpi (e.g. 'Open MPI v4.1.6')
# CUDA XLA bridge actually built? (module renamed from ..._gpu to ..._cuda in mpi4jax 0.9.x)
python -c "import mpi4jax._src.xla_bridge.mpi_xla_bridge_cpu; print('CPU bridge OK')"
python -c "import mpi4jax._src.xla_bridge.mpi_xla_bridge_cuda; print('CUDA bridge OK')"
```

If the CUDA bridge import fails: rerun the mpi4jax install with `-v` and look for the
`CUDA INFO: {...}` line (detection worked) vs the `CUDA path not found` warning (it didn't
— check `module load cuda` was active, or export `CUDA_ROOT` to the toolkit prefix and
rebuild). Verified working 2026-07-26 with mpi4jax 0.9.1 + jax 0.10.2 + cuda/12.6.0 +
openmpi/4.1.6.

Single-GPU device visibility (must run inside a GPU allocation, not on the login node):

```bash
srun --pty -A fc_kawturb -p savio3_gpu --qos=v100_gpu3_normal --gres=gpu:V100:1 \
  --cpus-per-task=4 -t 00:10:00 \
  bash -c "source activate jax_gpu && export PYTHONNOUSERSITE=1 && python -c \"import jax; print(jax.default_backend(), jax.devices())\""
```

Expect `cuda [CudaDevice(id=0)]`. `cpu [CpuDevice(id=0)]` means either no GPU was allocated or
the `jax[cuda12]` wheel didn't install cleanly (check `python -c "import jax; jax.devices()"`
stderr for a plugin load error).

**Does the whole thing work?** `sbatch slurms/run_test_suite_gpu.sh` runs the five-phase
backend battery on 4×A5000 (see `docs/RUNNING_TESTS.md`); `sbatch slurms/refactor_check_gpu.sh`
is the shorter version — the only real exercise of `comms.Runtime` bringing up
`jax.distributed`, the `shard_call` boundary and `kgrid.lin` under a real mesh. Run one of
them after any env rebuild, before trusting a production job.

## 3. Which backend, and why it matters

Three transports exist (`taranis/comms.py`); on GPU the choice is between two:

| | `comm_backend="jax"` | `comm_backend="mpi4jax"` |
|---|---|---|
| device ops | `ppermute`/`psum`/`pmax` inside a `shard_map` (NCCL) | mpi4py + mpi4jax custom calls |
| state arrays | global `jax.Array`s, z-sharded over the device mesh | process-local per rank |
| launch | `srun --mpi=pmix`, **no** `--gpu-bind` | `srun --mpi=pmix --gpu-bind=single:1` |
| scaling 4 → 16 GPUs (512²×256 fp32) | **4.15×** | 1.72× |
| snapshot layout | one shared flat `snap_path/<step>/` | `snap_path/<rank>/<step>/` |
| use it for | **every multi-GPU run** | one GPU, or a cross-check |

**Use `"jax"` for anything with more than one GPU.** The mpi4jax path pays three specific
costs on GPU: Savio's openmpi is not CUDA-aware (§7), so every transfer stages through host
memory; each mpi4jax op is an XLA custom call forcing a CUDA stream sync (×10–20 per step);
and its token chain is opaque to XLA, so nothing overlaps. At 4 A5000s that is 126 ms/step
against 77; at 16 GPUs across 4 nodes, 165 against 48. Full tables: `docs/performance.md`.

Either backend can restart from the other's snapshots (`tests/test_backend_jax_mpi.py`
phases 4a/4b prove it in both directions), so this is not a decision you are locked into.

## 4. Launching a multi-GPU job

`slurms/forced_turbulence_multigpu.sh` is the template; this is what each part of it is for.
Every one of these has cost someone a job.

**Slurm.** Use **`srun`, not `mpirun`** — only Slurm scopes GPUs per task. `--mpi=pmix` is
required (Savio's openmpi is `--without-pmi` + external PMIx, so pmi2 aborts before
`MPI_Init`; `srun --mpi=list` shows what a site supports). One task per GPU. **Never set
`CUDA_VISIBLE_DEVICES` yourself** — hand-setting it fights Slurm's own scoping.
(`comms._local_device_ids` only *reads* it, and only for `comm_backend="jax"` under a
launcher that left several GPUs visible to one process; override with
`RMHD_LOCAL_DEVICE_IDS`.)

**`--gpu-bind` is the one flag that differs between the backends**, and getting it backwards
fails loudly in both directions:

- `comm_backend="jax"`: **omit it.** All job GPUs must stay visible to every process —
  same-node NCCL transport needs peer GPUs visible, and with only its own GPU visible a
  process dies at the first collective (`ncclGroupEnd`, cuda error 101 "invalid device
  ordinal"). Each process pins itself to its node-local rank's ordinal.
- `comm_backend="mpi4jax"`: **`--gpu-bind=single:1` is mandatory.** That backend never calls
  `jax.distributed`, so every process just takes its own default device — with all GPUs
  visible, all N ranks pile onto GPU 0 (symptom: an immediate OOM from the second rank,
  since JAX preallocates ~75% of a device, or a "scaling" run that gets *slower* with more
  ranks). If your Slurm rejects the step-level `--gres`, use `--gpus-per-task=1`.

`srun` forwards the whole environment by default, so no `mpirun -x VAR` list is needed
(contrast with the CPU scripts `slurms/bench_phase2*.sh`).

**Environment.** Six exports, none optional:

| export | why |
|---|---|
| `unset PYTHONPATH` | the `anaconda3` module points it at the BASE anaconda's site-packages, whose `nvidia-*` packages precede the env's in `sys.path` — the root cause of "The cuSPARSE library was not found" (jobs 35861001/35861191). GPU jobs never need it. |
| `PYTHONNOUSERSITE=1` | blocks `~/.local` user-site shadowing |
| the `NVLIBS` block | jax's CUDA plugin can fail to `dlopen` the pip-bundled nvidia libs by bare soname inside a job even when the env is complete; putting every `nvidia/*/lib` dir on `LD_LIBRARY_PATH` fixes it. The block echoes its value into the `.out` as proof it ran — if the echoed paths show `.../anaconda3/<version>/lib/...` instead of `~/.conda/envs/jax_gpu/...`, PYTHONPATH is leaking again. |
| `NCCL_P2P_DISABLE=1` | **PCIe P2P between GPUs is broken on savio4_gpu nodes** (root-caused 2026-07-26 via `bench/nccl_repro.py`, interactive): NCCL rings connect, then the first collective hangs forever, under both CUMEM and legacy-IPC P2P — the signature of PCIe ACS misconfiguration, the GPUs report peer-capable but transfers stall. SHM transport works. Never remove this export until the cluster config changes; verify with the repro first. Consequences: same-node NCCL bandwidth is host-memory-limited, so jax-backend numbers here *understate* a P2P/NVLink cluster, and this is worth a Savio support ticket with `bench/nccl_repro.py` attached. |
| `XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true` | without it XLA emits the halo ppermutes and the two allreduces as async `-start`/`-done` pairs but schedules **zero** instructions between them, serializing comms with compute. Worth 1.31× at 16 GPUs (4 nodes) and ~1.02× at 4 GPUs on one node, for ~4% more temp memory (measured 2026-08-21, job 37912751). Never worth omitting on a multi-GPU run. Confirm the flag took with `bench/hlo_audit.py` — the overlap column should read a median of ~23 instructions, not 0 — before believing any timing that depends on it, and note that flag names drift between XLA versions: `--xla_gpu_enable_pipelined_collectives` was accepted once and no longer exists in jax 0.10.2, so probe before you rely on one. Full result: `docs/performance.md`, "XLA latency-hiding scheduler". |
| `TARANIS_PRECISION` | 32 or 64, **read at import time** — it is not a runtime flag, so it must be exported before the process starts. 32 on A5000/A40/L40/2080Ti; 64 only on V100. |

One more that is not required but that you want on any run you will resubmit:
`RMHD_COMPILATION_CACHE=$HOME/.taranis_jit_cache` turns on jax's persistent compilation
cache (`taranis/__init__.py` reads it), so a continuation job skips the tens of seconds the
first block otherwise spends compiling. Keyed by directory; safe to share across jobs.
Deliberately *off* in the benchmark scripts, where each flagset must compile its own binary
or the experiment is meaningless.

## 5. Writing your own multi-GPU driver

Start from `examples/multigpu_forced_turbulence.py` — it is short, and it is the shape every
one of these has to have. Five things the GPU backend requires that a laptop script does not:

- **Build `Parameters` before touching a jax device.** Constructing it resolves the
  transport, and for `comm_backend="jax"` that runs `jax.distributed.initialize()` and builds
  the device mesh — which jax refuses once the local backend exists. So no `jax.devices()`,
  no `jnp` array, no jit call above that line. The failure is explicit
  ("construct the first `Parameters(comm_backend='jax')` BEFORE any jax device work"), but
  it costs a queue wait to discover. Print the per-rank device report *after* it.
- **`nz` must be divisible by the total GPU count**, not just by the process count — the
  decomposition is over the device mesh. `comms.init_backend` checks it and says so.
- **`dims=3` only, and no `z_spectral`.** `comm_backend="jax"` rejects `dims=2` (there is no
  z decomposition to map) and rejects `z_spectral=True` (the z-FFT needs the whole z domain
  on one process — the jax backend exists to split it). Test particles are rejected on this
  backend too. The maximum useful device count is about `nz/2`: the halo is 2 planes wide.
- **Diagnostics that reduce over z need a `shard_map` context.** `diagnostics.energy`,
  `perpspec` and anything else built on `shared_physics.perp_reduce` end in an allreduce,
  which on this backend is a `lax.psum` over the mesh axis. Calling one from ordinary
  eager code raises `NameError: unbound axis name: z`. Wrap it:

  ```python
  from jax.sharding import PartitionSpec as P
  from taranis import comms
  fn = comms.shard_call(lambda s, kg: diag.energy(s, kg, params), params, kgrid,
                        out_specs=(P(), P()))   # out_specs describes the RETURN value
  E_kin, E_mag = fn(state, kgrid)
  ```

  The alternative, when a diagnostic is easier to write on the host, is
  `comms.to_local(state.fields, params, z_axis=1)` — this process's addressable shard, in
  the same per-rank layout mpi4jax produces (that is what `tests/test_backend_jax_mpi.py`
  does to compare the two backends).
- **Snapshots are one shared flat directory** (`snap_path/<step>/`) of global z-sharded
  arrays — orbax's native multihost path — not the per-rank tree the other backends write.
  Pass `params` to `get_saved_steps`/`load_snapshot` so they read the right layout. Either
  backend restarts from either layout, and restarting on a different GPU count works through
  `load_snapshot`'s z-slice union. Details and the rules for touching `snapshot_io.py`:
  `docs/checkpointing.md`.

**Sizing.** A `(2, nz_local, nx, nx/2+1)` state array is `2·nz_local·nx·(nx/2+1)·8` bytes at
fp32 (16 at fp64), and LSRK keeps several of those live plus FFT workspace — budget ~10× one
state array per GPU. JAX preallocates ~75% of the device by default; if you hit a
preallocation OOM with correct binding, lower `XLA_PYTHON_CLIENT_MEM_FRACTION` rather than
disabling preallocation (fragmentation hurts more). 512²×128 at fp32 is ~0.3 GB/state and
fills an A5000 usefully. A per-lane account of where the memory actually goes is in
`docs/performance.md`, "Memory: where it goes and what was removed".

**Editable-install caveat.** `pip install -e .` means a queued job imports whatever is
checked out in `~/taranis` when it *runs* — switching branches changes what your pending
jobs do. For A/B work use the `RMHD_PKG` mechanism (`bench/bench_phase1.py`,
`tests/test_backend_jax_mpi.py`), never `PYTHONPATH`: the editable install's import finder
beats it.

## 6. When it goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Job pends forever, reason `QOSMinGRES` | `--gres=gpu:<n>` without the explicit type, on an FCA account | `--gres=gpu:A5000:4` (§1), and the matching `--qos=` |
| `The cuSPARSE library was not found` | the `anaconda3` module's `PYTHONPATH` shadows the env's `nvidia-*` packages | `unset PYTHONPATH` + the `NVLIBS` block (§4) |
| Imports resolve to `~/.local/...` | user-site shadowing | `PYTHONNOUSERSITE=1`; `pip uninstall` the stray copy |
| Every rank prints `rank 0 / size 1` under a multi-rank launch | mpi4py linked against a different MPI than the launcher | rebuild mpi4py (and mpi4jax) from source with `MPICC=$(which mpicc)`, `--no-cache-dir` (§2). taranis detects this case and raises rather than letting every rank run the full domain. |
| Ranks report `platform=cpu` / `CpuDevice` | no GPU allocated, or the `jax[cuda12]` plugin failed to load | check `--gres`; run the §2 single-GPU visibility check and read its stderr |
| Two ranks report the same device id, or one rank lists every GPU | binding | `--gpu-bind=single:1` for mpi4jax; for the jax backend this is *expected* during bring-up but `local_devices` must still be one per process — check `RMHD_LOCAL_DEVICE_IDS` isn't set |
| First NCCL collective dies: `ncclGroupEnd`, cuda error 101, "invalid device ordinal" | `--gpu-bind=single:1` used with `comm_backend="jax"` | drop the flag (§4) |
| First NCCL collective hangs forever | PCIe P2P on savio4_gpu | `export NCCL_P2P_DISABLE=1` (§4). `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV` logs bring-up per rank; wrap phases in `timeout` so a hang becomes exit 124 instead of a dead allocation. |
| Silent multi-process hang with no output | anything collective called on a subset of ranks | `RMHD_DEBUG_HANG=1` (in the test drivers) dumps every thread's python stack every 120 s. Check that `params.save`, snapshot saves and allreduces run on *every* rank. |
| `RuntimeError: jax.distributed.initialize() failed` | jax device work happened before `Parameters` | move every `jax.devices()`/`jnp`/jit call below the `Parameters(...)` line (§5) |
| `NameError: unbound axis name: z` | a z-reducing diagnostic called outside `shard_map` | wrap it in `comms.shard_call` (§5) |
| OOM at startup with correct binding | JAX preallocates ~75% of the device | lower `XLA_PYTHON_CLIENT_MEM_FRACTION`; don't disable preallocation |
| `ValueError: nz=… must be divisible by …` | `nz` vs the device count | pick `nz` a multiple of the total GPU count (§5) |
| Snapshot directory errors on restart | the `params.json` differing-record guard, or a pre-forcing-era snapshot | read the error text — it names the fix (`Parameters.from_snapshot`, or `snapshot_io.old_snapshot_repair`) |
| Fields NaN with `cfl_every > 1` from a quiescent start | documented hazard | use `cfl_every > 1` only from developed states (CLAUDE.md) |

Two more standing constraints, easy to trip over: `dims=2` runs are single-process only, and
`TARANIS_PRECISION` is read at import so it cannot be changed inside a script.

---

The rest of this file is the benchmark and audit record, not the running instructions.

## 7. CUDA-aware MPI audit (decides how T8 is interpreted)

PERFORMANCE_PLAN F2(a): without CUDA-aware MPI, every halo and allreduce stages through host
memory. Which regime you're in changes what the T8 numbers mean, so measure it explicitly:

```bash
sbatch slurms/probe_cuda_mpi.sh      # 2×A5000, ~5 min of runtime
```

That job does three things: greps `ompi_info` for CUDA support, prints `srun --mpi=list`, and
runs `bench/probe_cuda_mpi.py` (the exact two-`sendrecv` ring of `comms.halo_exchange`, on
device buffers) once with `MPI4JAX_USE_CUDA_MPI=0` and once with `=1`.

Two possible outcomes — **record which one you got in the benchmark .out header**:

- **Not CUDA-aware** (no `cuda` lines from `ompi_info`, and/or the `=1` run segfaults or
  reports `correct=False`): run everything with `MPI4JAX_USE_CUDA_MPI=0`. Host staging is then
  the F2(a) cost we are measuring, and that is the honest T8 baseline. This is the default in
  all three benchmark scripts (`CUDA_MPI` env var, default `0`).
- **CUDA-aware** (`ompi_info` shows CUDA MCA parameters / smcuda / a CUDA-enabled UCX, and the
  `=1` run reports `correct=True` with a lower µs/halo-pair): submit the benchmark jobs with
  `CUDA_MPI=1 sbatch ...` and note it. F2(b) — the per-call stream sync — remains regardless;
  that is what the profile case measures.

**AUDIT RESULT (job 35845619, 2026-07-26): NOT CUDA-aware.** `openmpi/4.1.6` is configured
`--without-cuda` (the "MPI extensions: cuda" line is only the API stub; there are no cuda MCA
parameters and no smcuda). All jobs run `MPI4JAX_USE_CUDA_MPI=0`; host staging is the honest
mpi4jax baseline on Savio. Same job also showed this openmpi is `--without-pmi` + external
PMIx: **`srun` needs `--mpi=pmix`** (pmi2 aborts in `ext3x_client.c` before `MPI_Init`) —
now the default `MPI_MODE` in every GPU script. Rerun under pmix (job 35845686): binding
verified (1 distinct GPU per rank); `=0` host staging `correct=True` at ~2,900–7,500
µs/halo-pair for a 4.2 MB payload (1.1–2.9 GB/s — ≈30 ms/step of comm at 10 halo
exchanges/step, the number NCCL has to beat); `=1` segfaults in UCX `ucp_dt_pack` as
expected for a non-CUDA-aware build handed device pointers.

## 8. Running the Phase 3 benchmarks

```bash
sbatch slurms/bench_phase3_a5000.sh       # 1/2/4 A5000, fp32, single node (the workhorse)
sbatch slurms/bench_phase3_a40_scale.sh   # 4/8/16 A40 across nodes, fp32, ≤ 1 hr
sbatch slurms/bench_phase3_v100_fp64.sh   # 2×V100, fp64 anchor
```

Env knobs (all optional, all defaulted so a bare `sbatch` works):

| Var | Default | Meaning |
|---|---|---|
| `CUDA_MPI` | `0` | sets `MPI4JAX_USE_CUDA_MPI`; `1` only after a clean probe |
| `RUN_JAX` | `0` | `1` adds the `comm_backend="jax"` (shard_map/NCCL) rows — leave `0` until T9 lands, so the scripts still run if it's reverted |
| `MPI_MODE` | `pmix` | `srun --mpi=` mode (pmi2 aborts on Savio's openmpi; see §7) |
| `GPUS_PER_NODE` | `4` | A40 script only: set `2` if you switch to 2-GPU A40 nodes |

Each case prints one result line:

```
g4     3d_forced  nx=512 nz=128 ranks=4 [scan+nps+cfl1+halo_late] steps=80  ...  ms/step  backend=mpi4jax plat=cuda dev=[cuda:0] pkg=/global/home/users/.../taranis/__init__.py
```

`pkg=` proves which package version was imported (the `RMHD_PKG` mechanism — never PYTHONPATH,
the editable install's import finder beats it), `backend=` is `params.comm_backend`, `dev=` is
rank 0's bound device, and every option under test appears in the `[tags]` bracket.
`RMHD_REQUIRE_GPU=1` (a bench-script variable, not a taranis one) aborts any case where a rank
silently falls back to CPU.

`slurms/bench_xla_flags_2080.sh` is the XLA compiler-flag matrix (probe → HLO audit → timed
matrix → profile) and `bench/hlo_audit.py` its compile-only static half — that one runs on a
login node or a laptop, and `BENCH_SETS_16`, `PASSES`, `NX`, `NZ` are all env-overridable.

Bench flags used by these jobs (`bench/bench_phase1.py`): `nx<N>`/`nz<N>` grid sizes,
`backend=mpi4jax|jax`, `halo_early`/`halo_late` (T7 hook forced on/off, the GPU overlap pair),
`unroll` (`lsrk_scan=False`, T3's GPU candidate), `nps`, `cfl<N>`, `nb<N>`/`nr<N>`, `--profile`.
Both passes of each matrix exist so you can judge run-to-run spread before believing a delta.

## 9. Reading the profile

Each script ends with one `--profile` case, which wraps only the timed loop in
`jax.profiler.trace` and writes a TensorBoard-format trace to
`$SLURM_SUBMIT_DIR/prof_<job>_<jobid>/rank0/` (rank 0 only unless you set `RMHD_PROFILE_ALL=1`).
Copy it back and open it with `tensorboard --logdir prof_...` (needs `tensorboard` +
`tensorboard-plugin-profile`), or load the `*.trace.json.gz` in Perfetto/`chrome://tracing`. What
to look for, in order: on the GPU-stream row, the gaps between consecutive XLA kernels — with
mpi4jax each halo/allreduce is an XLA **custom call** that forces a CUDA stream sync, so the
signature of F2(b) is a repeating pattern of ~10–12 short stalls per step, one per comm op,
each bracketed by device→host/host→device copies if `MPI4JAX_USE_CUDA_MPI=0` (that's F2(a),
visible as memcpy activity around every custom call). Sum those gaps and divide by the step
time: that fraction is the ceiling on what an NCCL/`shard_map` backend (T9) can recover, and it
is the number to quote in the Phase 3 gate. A backend with real overlap looks different —
collectives on their own stream, running concurrently with FFT kernels rather than between them.
