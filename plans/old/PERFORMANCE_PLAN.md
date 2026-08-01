# jax_rmhd performance refactor plan

Target: better throughput and scaling on CPU clusters and GPU clusters.
Structure: ranked findings, then a phased task list sized for handoff to Sonnet agents.

## Findings (ranked by expected impact)

### F1. Communication latency dominates the 3D hot loop at scale
Worst case (lsrk54 + elsasser forcing + adaptive dt), per timestep:
- 5 RHS evals × 2 `mpi4jax.sendrecv` halo exchanges (`shared_physics.z_derivatives`) = 10 point-to-point ops
- 5 RHS evals × 2 `mpi4jax.allreduce` (Pp, Pm in `rmhd.ForcingTerm` → `_perp_reduce`) = 10 global syncs
- 1 `allreduce(MAX)` for CFL (`rmhd.set_timestep`)
mpi4jax's token threading serializes all of these with compute — no overlap. On CPU
clusters with many ranks this is latency-bound; on GPUs it also stalls the stream.
Levers: forcing normalization once per step instead of per stage (removes 8 allreduces),
stack Pp/Pm into one allreduce, CFL recompute every N steps, overlap halos with
perpendicular work.

### F2. mpi4jax has GPU-specific costs (but stays the CPU backend)
mpi4jax is the right transport on CPU clusters — XLA:CPU collectives are known-slow
(shard_map was already tried there and performed badly). On GPUs the same design has
three specific costs: (a) without CUDA-aware MPI every halo/allreduce stages through
host memory; (b) even with it, each mpi4jax op is an XLA custom call that forces a CUDA
stream sync — negligible on CPU, a pipeline stall on GPU, ×10–20 comm calls per step;
(c) the token chain is opaque to XLA, so no compute/comm overlap (NCCL would get
side-stream scheduling). Plan: comm abstraction layer with mpi4jax as the permanent
CPU-cluster backend; benchmark mpi4jax+CUDA-aware-MPI on GPU first; build a
`shard_map`/NCCL backend only if those numbers justify it. Note the architecture
survives either way: inside `shard_map` code sees the same local `nz_device`-shaped
shards, `ppermute` is a drop-in for the halo sendrecv, `psum`/`pmax` for allreduces —
only init (mesh vs. cart_comm) and the ~3 comm primitives change.

### F3. No buffer donation in the jitted steppers
`block_of_steps_jit` and `sim_to_next_snap_jit` don't donate the state → XLA keeps
input and output copies of the full `(nfields, nz_local, nkx, nky)` complex arrays
live. Free memory + copy savings via `donate_argnums`.

### F4. LSRK stage loop structure (`timestepping.lsrk_advance`)
`lax.scan` over stages with `lax.cond(istage==0, init_rhs, rhs(...))`:
- `init_rhs` stays live in memory across all 5 stages
- scan blocks per-stage constant folding of alpha/beta/gamma and cross-stage fusion
- the cond adds control-flow overhead on GPU
The scan was added because it helped CPU memory behavior — so make unrolled
(Python-loop) vs. scan a config choice and benchmark both per backend.

### F5. Static k-space arrays recomputed every call
`kgrid` is a traced jit argument, so `ksq()`, `inv_ksq()`, `dealias_filter()`,
`hdiss_exponents()`, `_perp_yfac()`, `_shell_mask()`, and the forcing cos/sin
z-envelopes are recomputed (extra kernels) at every RHS eval / OU update. Precompute
once at setup and store as concrete arrays.

### F6. Forcing wastes RNG and reductions (`shared_physics`)
`ou_update` draws complex Gaussian noise over the full `(n_ou, 2, nkx, nky)` grid,
then masks down to a thin shell (typically a few dozen modes). Generate only for the
static shell index set. Also F1's per-stage normalization and separate Pp/Pm reductions.

### F7. Checkpoint I/O serialized with compute (`run.py`)
`mngr.wait_until_finished()` immediately after every `save_snapshot` defeats orbax's
async save. Wait lazily (before the *next* save and at the end) so I/O overlaps the
next block of steps.

### F8. No persistent compilation cache
Every cluster job recompiles from scratch. Opt-in `jax.config` compilation-cache dir
saves minutes per job at scale.

### F9. Scaling ceiling: z-only decomposition (noted, out of scope)
Max ranks ≈ nz/2 (halo width 2). True perpendicular (pencil) decomposition of the
FFTs would be a much larger project; not in this plan unless requested.

## Task list for Sonnet agents

Every task must: (a) keep `tests/test_advection.py`, `tests/test_dissipation.py`,
`tests/test_forcing_smoke.py` passing (serial and `mpirun -n 4` for 3D-relevant
changes), (b) report before/after numbers from the Phase 0 benchmark, (c) follow the
invariants in CLAUDE.md — `state._replace(...)` never positional `SimulationState`,
rfft2 reality rows at ky=0/Nyquist, plain-Python branching on static `params`, the
shared energy normalization convention, and (d) comment style: ~1 line per new
function saying what it does, ~1 line noting any change from the previous version —
no walls of text.

### Phase 0 — measurement (prerequisite)
**T0. Benchmark harness.** New `bench/bench_step.py`: times steps/sec after warmup for
2D 512² and 3D 64³ (and 128³ flag), forcing off/momentum/elsasser, rk44/lsrk33/lsrk54,
adaptive dt on/off; JSON output; `--profile` flag emitting a `jax.profiler` trace;
works serial and under mpirun. No physics changes. Benchmarks run at fp32 (the code
default) — comm-latency and structural results carry over to fp64; note fp32 halves
message sizes so comm/compute ratios shift slightly. Savio GPU notes: A5000 (savio4),
A40/TITAN RTX/2080Ti (savio3) are all poor at fp64 (~1/32); the two 2×V100 savio3
nodes are the only good-fp64 option for single-node fp64 spot checks.

### Phase 1 — low-risk single-device wins (independent; parallelizable)
**T1. Jit hygiene.** Add `donate_argnums` for state in `block_of_steps_jit` /
`sim_to_next_snap_jit` (F3); opt-in persistent compilation cache via env var (F8);
make per-block `print(state.t)` optional/every-N (host-sync reduction).

**T2. Precompute static grid arrays (F5).** Extend `K_Grids` / setup to carry concrete
precomputed `ksq`, `inv_ksq`, `dealias_filter`, y-doubling factor, forcing shell mask,
and cached per-params `hdiss_exponents`; precompute forcing z-envelopes. Keep method
API backward compatible where cheap.

**T3. LSRK restructure (F4).** Replace scan+cond with a statically unrolled stage loop
(stage coefficients become Python constants; `init_rhs` used directly in stage 0, freed
after). Keep the scan variant behind `params.lsrk_scan=True` for CPU comparison.
Verify identical trajectories to current code at float64.

**T4. Forcing efficiency (F6 + part of F1).** (a) Shell-restricted noise: draw noise
only at precomputed static shell indices, scatter into the k-grid, preserving the
hermitian symmetrization at ky=0/Nyquist rows. (b) Stack Pp/Pm into a single allreduce.
(c) New option `forcing_norm_per_step=True`: compute the power-normalization scale once
per full step (at stage 0) and reuse across sub-stages — document the approximation.

### Phase 2 — communication (sequential: T5 → T6, T7)
**T5. Comm abstraction layer.** New `jax_rmhd/comms.py` defining `halo_exchange(f)`,
`allreduce_sum(x)`, `allreduce_max(x)` selected by `params.comm_backend`; port
`z_derivatives`, `_perp_reduce`, `set_timestep` onto it. mpi4jax backend reproduces
current behavior exactly. Pure refactor.

**T6. Fewer syncs per step (F1).** `params.cfl_every` (recompute dt every N steps
inside the step loop, reuse otherwise; N=1 default preserves behavior). Combine with
T4b/c this cuts worst-case global syncs per step from ~11 to ~1–2.

**T7. Halo overlap.** Reorder the RHS so the halo exchange for `LinearTerm` is issued
before the perpendicular FFT/nonlinear work, letting the exchange proceed concurrently
where the backend allows. Benchmark under mpirun; keep if it wins.

### Phase 3 — GPU backends (T8 first; T9 conditional on its results; T9 depends on T5)
**T8. GPU baseline on mpi4jax.** Verify/set up CUDA-aware MPI on Savio
(`MPI4JAX_USE_CUDA_MPI`, module audit), GPU binding per rank; run the T0 benchmarks
multi-GPU (fp32, savio3_gpu/savio4_gpu); profile to quantify the per-call stream-sync
and staging costs (F2). Also covers CPU thread-pinning guidance and `slurms/`
template updates. Output: numbers that decide whether T9 is worth building.

**T9. JAX-native GPU backend (conditional, gated on T8).** Add `comm_backend="jax"`
to the T5 layer: `jax.distributed.initialize`, 1D device mesh over z, `shard_map`
around the step function, `lax.ppermute` halos, `psum`/`pmax` reductions. NCCL +
XLA compute/comm overlap on GPU clusters. mpi4jax remains the CPU-cluster backend
(shard_map already measured slow on CPU — do not switch CPU runs). Largest task;
includes multi-process correctness test vs. the mpi4jax backend and a check of
orbax per-rank checkpointing interplay.

### Phase 4 — I/O
**T10. Async checkpointing (F7).** Move `wait_until_finished()` to before the next
save and simulation end; ensure clean interaction with `max_to_keep` deletion.

## Suggested execution order
T0 → (T1, T2, T3, T4 in parallel) → T5 → (T6, T7, T10 in parallel) → T8 → T9 only if
T8's numbers show mpi4jax+CUDA-aware-MPI is still comm-bound on GPU. Benchmark gate
after each phase; revert any change that doesn't pay for itself on either backend.
