# jax_rmhd on Savio GPUs — from-scratch setup, binding, and benchmarking

Everything needed to go from "no GPU env" to "T8 benchmark numbers": build the `jax_gpu`
conda env, verify one GPU per MPI rank, audit whether the site MPI is CUDA-aware, and run
the Phase 3 benchmark jobs. Companion to `PHASE3_PLAN.md` (task A1 / PERFORMANCE_PLAN T8).

## 0. Which partition to ask for

Verified against docs-research-it.berkeley.edu on 2026-07-25.

| Partition | GPU | GPUs/node | CPUs per GPU | FCA QoS | Regular-priority pool |
|---|---|---|---|---|---|
| savio4_gpu | A5000 24 GB | 8 | 4 | `a5k_gpu4_normal` | 136 GPUs, but **16 CPUs/user cap = 4 GPUs max per user** |
| savio4_gpu | L40 46 GB | 8 | 8 | `savio_lowprio` only | — |
| savio3_gpu | V100 32 GB | 2 | 4 | `v100_gpu3_normal` | 2 GPUs — the only good-fp64 cards |
| savio3_gpu | A40 48 GB | 2 or 4 | 8 | `a40_gpu3_normal` | 16 GPUs cluster-wide (a 16-GPU job waits for all of them) |
| savio3_gpu | 2080Ti 11 GB | 4 | 2 | `gtx2080_gpu3_normal` | 28 GPUs |

Rules that will otherwise cost you a day of pending jobs:

- **GPU partitions are per-core scheduled**, unlike savio3's per-node charging. Request
  exactly `cpus-per-task × ntasks = (CPUs per GPU) × (number of GPUs)` — never a whole node.
- `--gres=gpu:<TYPE>:<n>` **with the explicit type is mandatory for FCA (`fc_*`) accounts**.
  Omitting the type pends forever with reason `QOSMinGRES`.
- A40 and V100 additionally require the explicit `--qos=` from the table. A5000 does too in
  practice (`a5k_gpu4_normal`); L40 is `savio_lowprio` only.
- `--gres` is **per node**. 16 A40 = `--nodes=4 --ntasks-per-node=4 --gres=gpu:A40:4`.
- SU rate: 3.67/core-hr on savio3_gpu, 4.67 on savio4_gpu. 16×A40 = 128 cores ≈ 470 SU/hr —
  keep scaling jobs at or under 1 hour of walltime.
- Account is `fc_kawturb` everywhere.
- **fp64 only on V100.** A5000/A40/L40/2080Ti are workstation-class: fp64 runs at ~1/32 of
  fp32. Run those at `RMHD_PRECISION=32` and take fp64 numbers from the 2×V100 job. Phase 1's
  lesson still applies — fp32 flatters comm savings by ~3×, so keep/revert decisions get made
  on the fp64 anchor.

## 1. Build the `jax_gpu` env (login node, once)

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
# multi-rank run (config.py touches MPI.COMM_WORLD once mpi4py is present). A single-rank
# run works serially without them (comm_backend="serial", auto-selected when they're
# absent), but install them anyway on Savio -- the launcher-mismatch guard (every rank
# reporting rank 0, below) needs a real mpi4py to detect. Build mpi4py FROM SOURCE against
# the loaded openmpi -- a prebuilt wheel may link a different MPI than the one `srun`/`mpirun`
# on the compute node actually launches, which is exactly that every-rank-reports-rank-0
# symptom.
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
cd ~/jax_rmhd && pip install -e .
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

Runtime note — ROOT CAUSE of "The cuSPARSE library was not found" (jobs 35861001/35861191):
the `anaconda3` module sets `PYTHONPATH` to the BASE anaconda's site-packages, which contains
its own (incompatible) `nvidia-*` packages; `PYTHONPATH` precedes the env's site-packages in
`sys.path`, so jax's CUDA plugin imported the base `nvidia` tree instead of the env's. Every
GPU sbatch script therefore does `unset PYTHONPATH` right after `source activate jax_gpu`
(GPU jobs never need PYTHONPATH — code selection uses `RMHD_PKG`) and additionally exports
`LD_LIBRARY_PATH` over every env `nvidia/*/lib` dir (the `NVLIBS` block, which also echoes
its value into the .out as proof it ran — if the echoed paths ever show
`.../anaconda3/<version>/lib/...` instead of `~/.conda/envs/jax_gpu/...`, PYTHONPATH is
leaking again). Keep both blocks in any new GPU job script.

NCCL note (root-caused 2026-07-26 via `bench/nccl_repro.py`, interactive): **PCIe P2P
between GPUs is broken on savio4_gpu nodes** — NCCL rings connect, then the first
collective hangs forever, under both CUMEM and legacy-IPC P2P (signature of PCIe ACS
misconfiguration; the GPUs report peer-capable but transfers stall). All GPU job scripts
export `NCCL_P2P_DISABLE=1` (SHM transport, works). Consequences: (a) never remove that
export until the cluster config changes — verify with the repro first; (b) same-node NCCL
bandwidth is host-memory-limited, so jax-backend numbers understate what a P2P/NVLink
cluster would achieve; (c) worth a Savio support ticket with `bench/nccl_repro.py`
attached. Also required for the jax backend: launch WITHOUT `--gpu-bind` (all job GPUs
visible; `comms._local_device_ids` pins per-process ordinals) — mpi4jax launches keep
`--gpu-bind=single:1`.

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

## 2. GPU binding: one MPI rank per GPU

This codebase decomposes only along z across MPI ranks, and with `size > 1` it never calls
`jax.distributed.initialize` on the mpi4jax backend — every process just uses **its own
default device**. So if all N GPUs are visible to all N ranks, all N ranks pile onto GPU 0
(symptom: an immediate OOM from the second rank, since JAX preallocates ~75% of a device, or a
"scaling" run that gets slower with more ranks). Binding is therefore mandatory, and it must
come from Slurm:

```bash
srun --mpi=pmix --ntasks=4 --cpus-per-task=4 --gres=gpu:A5000:4 --gpu-bind=single:1 \
     python -u bench/bench_phase1.py ...
```

- Use **`srun`, not `mpirun`**, for multi-GPU steps: only Slurm scopes `CUDA_VISIBLE_DEVICES`
  per task. **Never set `CUDA_VISIBLE_DEVICES` yourself** — hand-setting it fights Slurm's own
  scoping. (`comms._local_device_ids` only *reads* it, and only for `comm_backend="jax"` under
  a launcher that left several GPUs visible to one process; override with
  `RMHD_LOCAL_DEVICE_IDS`.)
- `--gpu-bind=single:1` gives each task one distinct GPU. If your Slurm rejects the step-level
  `--gres`, use `--gpus-per-task=1` instead; both are Slurm-side scoping.
- `srun` forwards the whole environment by default, so no `mpirun -x VAR` list is needed
  (contrast with the CPU scripts `slurms/bench_phase2*.sh`).
- `--mpi=pmix` is REQUIRED on Savio (audit result, section 3: this openmpi is
  `--without-pmi` + external PMIx, so pmi2 aborts before `MPI_Init`); `srun --mpi=list`
  (printed by the probe job) shows what a site supports. All Phase 3 scripts honor
  `MPI_MODE=<mode>` as an env override.

**Verification**: every script prints one line per rank before timing anything:

```
[rank 0] platform=cuda local_devices=[cuda:0] global_device_count=1
[rank 1] platform=cuda local_devices=[cuda:1] global_device_count=1
```

Each rank must show `platform=cuda` and exactly **one** local device, and the ids must differ
across ranks on a node. Two ranks reporting the same id, or a rank listing all 4 devices, means
binding failed — fix that before believing any number.

## 3. CUDA-aware MPI audit (decides how T8 is interpreted)

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

## 4. Running the Phase 3 benchmarks

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
| `MPI_MODE` | `pmix` | `srun --mpi=` mode (pmi2 aborts on Savio's openmpi; see section 3) |
| `GPUS_PER_NODE` | `4` | A40 script only: set `2` if you switch to 2-GPU A40 nodes |

Each case prints one result line:

```
g4     3d_forced  nx=512 nz=128 ranks=4 [scan+nps+cfl1+halo_late] steps=80  ...  ms/step  backend=mpi4jax plat=cuda dev=[cuda:0] pkg=/global/home/users/.../jax_rmhd/__init__.py
```

`pkg=` proves which package version was imported (the `RMHD_PKG` mechanism — never PYTHONPATH,
the editable install's import finder beats it), `backend=` is `params.comm_backend`, `dev=` is
rank 0's bound device, and every option under test appears in the `[tags]` bracket.

Bench flags used by these jobs (`bench/bench_phase1.py`): `nx<N>`/`nz<N>` grid sizes,
`backend=mpi4jax|jax`, `halo_early`/`halo_late` (T7 hook forced on/off, the GPU overlap pair),
`unroll` (`lsrk_scan=False`, T3's GPU candidate), `nps`, `cfl<N>`, `nb<N>`/`nr<N>`, `--profile`.
Both passes of each matrix exist so you can judge run-to-run spread before believing a delta.

## 5. Reading the profile

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

## 6. Codebase-specific gotchas

- `RMHD_PRECISION` (32/64) is read **at import time**, so it must be exported before the
  process starts — it is not a runtime flag. The scripts set it per job.
- Do not set `CUDA_VISIBLE_DEVICES` (see §2). Do not launch multi-GPU steps with `mpirun`.
- `dims=2` runs are single-process only; all GPU benchmark cases are 3D.
- Memory sizing: a `(2, nz_local, nx, nx/2+1)` state array is `2·nz_local·nx·(nx/2+1)·8` bytes
  at fp32 (16 at fp64), and LSRK keeps several of those live plus FFT workspace — budget ~10×
  one state array. JAX preallocates ~75% of the device by default; if you hit a preallocation
  OOM with correct binding, lower `XLA_PYTHON_CLIENT_MEM_FRACTION` rather than disabling
  preallocation (fragmentation hurts more).
- Snapshots are per-MPI-rank orbax directories; restarting on a different GPU count works
  through `load_snapshot`'s z-slice union, exactly as on CPU.
