# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pseudospectral solver for reduced MHD (RMHD) and related plasma fluid models, written in
JAX. Spectral (rfft2) in the perpendicular (x,y) plane, finite-difference (4th order
centered) in z, domain-decomposed across MPI ranks along z only. Currently only RMHD is
implemented; the architecture is designed to add other equation sets (gradient drift,
compressible RMHD, KRMHD, gyrokinetics, ...) without touching the core solver.

## Setup / running

```
pip install -e .
```

`pyproject.toml`'s `dependencies` list is complete as of 2026-07-28 (`jax`, `jaxlib`,
`orbax-checkpoint`, `numpy`, `mpi4py`, `mpi4jax`, `tensorstore`, `etils[epath]`), so
`pip install -e .` needs a working MPI toolchain on the machine (mpi4py/mpi4jax build
against it). `matplotlib` is the optional `examples` extra (`pip install -e ".[examples]"`),
needed only for the example notebooks/plotting test scripts.

Precision is controlled by an env var read at import time, not a runtime flag:
```
RMHD_PRECISION=64 python your_script.py   # float64/complex128; default is 32
```

There's no pytest suite and no lint/format config in this repo. "Tests" under `tests/` are
standalone scripts (not pytest — no fixtures/assertions framework, they print/plot results
for a human to interpret), run directly:
```
python tests/test_dissipation.py
python tests/test_advection.py
python tests/test_forcing_smoke.py     # does use plain assert-and-print PASS/FAIL checks
```
For MPI-parallel (z-decomposed 3D) runs, launch under `mpirun`, e.g.:
```
mpirun -n 4 python tests/test_advection.py
```
2D runs (`dims=2`) are single-process only — `Parameters.__init__` prints a warning if run
with `size>1`. `tests/savio_scaling/` and `slurms/` are SLURM/cluster scaling-test scripts
for the Savio HPC cluster specifically (hardcoded paths), not general-purpose examples.

`examples/*.ipynb` are worked examples of varying freshness — several predate the current
API (`jr.Fields`, positional `SimulationState(...)` construction) and will error as-is;
`orzag-tang-2D.ipynb`, `orzag-tang-3d.ipynb`, `forced-turbulence-2D.ipynb`, and
`forced-turbulence-3D.ipynb` are current.

## Architecture

### Field representation

`SimulationState` (NamedTuple, `types.py`) is `(t, fields, forcing_state, forcing_key,
forcing_scale)` — see the forcing section for `forcing_scale`'s lifecycle.
`fields` has shape `(nfields, nz_local, nkx, nky)`: real-space grid in z, rfft2 spectral in
(x,y). **This axis is never dropped, even in 2D** — `dims=2` still produces
`(nfields, 1, nx, ny//2+1)`, not a 3D-rank array; `run.py::initialize` reshapes the x,y
coordinate grids with a leading axis of 1 unconditionally, 2D or 3D. `initialize` also
applies the 2/3 dealias mask (`grids.dealias_mask`, the same mask baked into
`kgrid.dealias`) to the transformed initial condition: the nonlinear term is the only
place the mask is applied during evolution, so an unmasked IC's beyond-cutoff energy
would otherwise persist forever and alias the brackets. `dims==3` requires
`nz % size == 0` — validated in `Parameters.__init__` (before 2026-07-28 the mpi4jax
path silently truncated to `size*(nz//size)` planes, misplacing the periodic z-seam).

rfft2 convention: `kx` (`grids.py::K_Grids`) is full two-sided (`fftfreq`), `ky` is
half/non-negative (`rfftfreq`) — real-space reality is a constraint *between* `(kx,ky)` and
`(-kx,ky)` at `ky=0` and `ky=Nyquist`, not a per-mode constraint. Anything that writes
directly into k-space rather than deriving it via fft of a real field (e.g. stochastic
forcing) must enforce this explicitly or the reconstructed field silently isn't real at
those rows. When enforcing it on *noise*, divide the symmetrized combination by sqrt(2),
not 2 (`shared_physics._symmetrize_real_line`): a plain average halves the variance both
of the paired modes and of the self-conjugate kx=0/Nyquist points (whose target variance
is that of a real, not complex, Gaussian — the single sqrt(2) restores both at once),
anisotropically underforcing the purely-x-varying shell modes. Fixed 2026-07-28 — forced
runs from before
and after do not reproduce bitwise (same RNG stream, different amplitudes at those rows),
so old forced benchmarks need re-baselining. `jnp.fft.rfft2`/`irfft2`
(`grids.py::fft`/`ifft`) are unnormalized transforms:
an O(1) real-space field has raw coefficients of magnitude O(nx*ny), not O(1) — matters for
any synthetic k-space process whose amplitude is meant to be grid-resolution-independent.

### Parameters / physics registry

`params.save(snap_path)` records the *constructor arguments* (not derived attrs) plus
precision to `snap_path/params.json` (identical re-save is a no-op, a differing
existing record is a hard error — the guard against continuing a directory with wrong
parameters). `save` is **collective under MPI**: rank 0 alone checks/writes and
broadcasts the outcome so every rank raises or returns together — never call it from a
subset of ranks. numpy/jax 0-d scalar ctor args are unwrapped to python scalars for the
JSON record (`_json_scalar`); anything else unserializable is a clear `TypeError`. `Parameters.from_snapshot(snap_path, **overrides)` reconstructs via
`__init__` (all validation/derivation reruns), explicit overrides win, unknown keys from
other code versions are ignored with a warning, precision mismatch warns (it's env-set at
import). Both are explicit calls — nothing writes params.json automatically.

`Parameters` (`config.py`) is **not a pytree** (the registration was removed 2026-07-28 —
it was never flattened anywhere, and its aux dict carried MPI comm handles). It is only
ever closed over by jitted functions or passed via `static_argnums`, so every attribute is
a compile-time constant under `jax.jit` and plain Python `if params.foo:` branching in
physics code is correct and preferred over `jax.lax.cond`. Never pass a `Parameters` as a
traced (non-static) jit argument or put one inside a scanned/mapped tree — it is not a
valid JAX type. z-related attributes (`dz`, `Lz`, `z_diss`, `cart_comm`,
`left_neighbor`/`right_neighbor`) only exist when `dims==3`; guard access to them.
`z_diff_order`/`z_diss_hyper` are accepted by `Parameters.__init__` and stored, but
`physics/rmhd.py::LinearTerm` doesn't read them back — z-derivatives are hardcoded to 4th
order and z-hyperdissipation to the `z_diss_hyper=2` form regardless of what's passed in
(see the `#TODO` in `LinearTerm`); passing a non-default value silently does nothing.

New equation sets register via `physics/__init__.py`'s `equation_registry`: an
`EquationRecipe(set_timestep_func, term_funcs, grad_func, forcing_scale_func=None,
halo_start_func=None)` per `eqtype`. `term_funcs` are
summed to build the RHS (`construct_rhs`); dissipation is *not* one of them — it's applied
separately as an integrating factor in `timestepping.py` (the precomputed `kgrid.hdiss`), not
as an RHS term. **Term funcs take 5 positional args**: `(state, grads, kgrid, params,
halo)` — `construct_rhs` always passes `halo` (the result of `halo_start_func`, or None),
so every term func must accept it (declare `halo=None` and ignore it if unused).
`halo_start_func` issues the equation's z-halo exchange at the top of the RHS, before the
perpendicular FFT work (T7); it returns whatever the consuming term expects
(`(recv_left, recv_right)` for RMHD's `z_derivatives`) or None for dims=2. RMHD registers
`rmhd.halo_start`, but `construct_rhs` only USES the hook per backend
(`_halo_start_enabled`): off under mpi4jax (measured no fp64 win — the token chain
serializes comm with compute anyway), on by default under `comm_backend="jax"` (measured
neutral on Savio at bench sizes; kept for NVLink/IB hardware), overridable either way with
`params.halo_start=True/False` (how the benchmark measures the on/off pair). When
disabled, `z_derivatives` issues its own exchange at the old call point.
`comms.halo_exchange(f, params, width=2)` takes an explicit halo width (default 2, today's
behavior); RMHD's `halo_start` and `z_derivatives`' internal fallback both pass `width=2`
explicitly and must agree — `z_derivatives` derives its stencil offsets from the received
slab's width rather than hardcoding them, so a pre-issued halo narrower than the stencil
needs is a clear assertion failure, not silently wrong output.
`physics/shared_physics.py` holds equation-agnostic helpers (`gradk`,
`bracket`, z-derivative stencils, the O-U forcing mechanics); `physics/rmhd.py` holds the
RMHD-specific term functions and maps generic building blocks onto the (phi,psi) fields.

All distributed transport goes through `jax_rmhd/comms.py`: `halo_exchange(f, params)`,
`allreduce_sum(x, params)` (array x allowed), `allreduce_max(x, params)`, dispatched in
plain Python on the static `params.comm_backend` (validated at `Parameters` construction).
Two backends exist (Phase 3, see PHASE3_RESULTS.md for verdicts/benchmarks):

- `"mpi4jax"` (default, the CPU-cluster production backend): mpi4py communicators +
  mpi4jax device ops; arrays stay process-local; verified bit-identical to pre-Phase-3.
  Nothing outside comms.py imports mpi4jax — keep it that way.
- `"jax"` (the GPU backend, shard_map/NCCL): the CONTROL plane is still mpi4py (rank/size,
  params.save, snapshot-index broadcast) — only the three device ops become
  `lax.ppermute`/`psum`/`pmax`, valid ONLY inside the shard_map context that
  `comms.shard_call` wraps around the jitted steppers (run.py's four call sites). State
  and kgrid become GLOBAL z-sharded jax.Arrays (`comms.to_global`); inside shard_map every
  device sees the same local `(nfields, nz_local, nkx, nky)` shapes the mpi4jax ranks see,
  so physics code is backend-agnostic. `Parameters(comm_backend="jax")` brings up
  `jax.distributed` (coordinator broadcast over mpi4py; must be the FIRST jax device work
  in the process); `"jax"`+`dims==2` is rejected. `forcing_state`/`forcing_key` are
  replicated, never sharded. Launch: one process per GPU, jax backend WITHOUT
  `--gpu-bind` (NCCL needs peer GPUs visible; `comms._local_device_ids` pins ordinals),
  mpi4jax WITH `--gpu-bind=single:1`; on Savio also `NCCL_P2P_DISABLE=1` and the env
  blocks in SAVIO_GPU_SETUP.md. GPU production config: `lsrk_scan=True` (unroll hurts
  multi-node jax; helps mpi4jax-GPU — per-machine knob).

### Timestepping

`rk_advance`/`lsrk_advance` (`timestepping.py`) rebuild intermediate `SimulationState`s at
every RK/LSRK sub-stage. **Always use `state._replace(...)`, never positional
`SimulationState(t, fields)` construction** — the latter silently drops/misaligns any
fields beyond the first two now that the tuple has grown (this bit us adding forcing:
`forcing_state`/`forcing_key` must survive unchanged across sub-stages within a timestep,
since they're only updated once per full step, not per sub-stage).

`lsrk_advance` has two stage-loop structures selected by `params.lsrk_scan`: `lax.scan`
(default — measured ~20% faster on CPU) and statically unrolled (`lsrk_scan=False`, the
GPU candidate). Bitwise-identical trajectories at fp64.

`params.cfl_every` (int >= 1, default 1) recomputes the adaptive dt — and thus the CFL
`allreduce(MAX)` — only once per block of `cfl_every` steps. dt is *hoisted out of the
stepper*: `run._cfl_block` computes it from the block's starting state (`grad_func` +
`set_timestep_func`, exactly `estimate_good_nblock`'s pattern) and passes it into
`rk_advance`/`lsrk_advance` as `dt_override`, which then skips its own `set_timestep`
(`dt_override=None` = historical per-step behavior). Never put the collective under a
`lax.cond` and never reuse a rank-local dt — one collective, one dt, all ranks. `cfl_every=1`
takes a literally unchanged legacy code path (bitwise-identical trajectories), as does
`adaptive_timestep=False` (fixed dt has no reduction to skip, so `cfl_every` is ignored
there). With dt frozen for N steps the CFL condition can be transiently violated while the
flow accelerates — compensate with `cfl_safety`. **From a quiescent start this is not a
small effect: the CFL dt collapses ~10x within a few steps of forced spin-up, so a frozen
quiescent dt at N>~5 runs far over CFL and silently NaNs (measured: N=20 NaNs by t~2,
N=5 survives). Use N>1 only from developed states**; also note snapshot/t_end overshoot
grows to up to N steps of the (possibly large) frozen dt. Costs one extra standalone `grad_func`
evaluation per block (the stepper's stage-0 rhs no longer doubles as the dt source), so
N>1 only pays off when the allreduce is expensive (high rank counts). The forcing update
(`_advance_forcing`) still runs every step in both paths; `nblock` in `block_of_steps`
still counts *steps*, rounded **up** to a whole number of `cfl_every`-step blocks.

**Buffer donation consumes input states.** `simulate`/`simulate_scan` jit their steppers
with `donate_argnums=(0,)`: any state passed in is invalidated — touching it afterwards
raises "Array has been deleted". Always continue from the *returned* state; read anything
you need (energies, fields) from a state *before* passing it to simulate. This also means
Phase 4's async-checkpointing task (T10) must not remove the `wait_until_finished()`
barrier without first decoupling orbax's buffer from the next donating call.

`kgrid` is bound to the `params` it was built from: `setup_kgrids` bakes `diss`, `hyper`,
`fshell`, `Lz`, rank layout etc. into precomputed arrays. `K_Grids` is a **dumb pytree
container** — plain fields (`kgrid.ksq`, `.inv_ksq`, `.dealias`, `.hdiss`, `.yfac`, plus
forcing-only `fmask`/`fidx_*`/`z_env*` which are None when forcing is off), no methods, no
lazy fallbacks (removed 2026-07-27), and **`setup_kgrids` is the only sanctioned
constructor**. Computation must never move into the type itself: jax rebuilds the
NamedTuple constantly with tracers, PartitionSpecs (`kgrid_specs`) and global arrays
(`_kgrid_to_global`) as field values. Never reuse a `kgrid` with a different or
mutated `Parameters` — rebuild it (and note mutating `params` attributes after a jit trace
silently reuses the stale compile, since `Parameters` hashes by identity).

### Stochastic forcing (`params.forcing`)

Ornstein-Uhlenbeck process injecting power into a shell of perpendicular wavenumbers,
sustaining turbulence instead of letting it freely decay. `forcing_state` shape
`(n_ou, 2, nkx, nky)`: axis 0 is 1 (`forcing_mode="momentum"`, forces phi only) or 2
(`"elsasser"`, forces z+ = phi+psi and z- = phi-psi independently, each with its own
`forcing_power_elsasser` target); axis 1 is the [A,B] cosine/sine z-envelope coefficients
(dims=3) or just uses A directly (dims=2, no z to project onto).

- `shared_physics.ou_update`/`reconstruct_envelope`/`perp_inner_product`/
  `perp_mean_square` are equation-agnostic; `rmhd.ForcingTerm` does the RMHD-specific power
  normalization and (phi,psi) mapping.
- Power normalization (`perp_inner_product`, `safe_scale`) targets exact injection power
  regardless of current field amplitude — cap the *scale factor* (`forcing_scale_max`), not
  the denominator (`P`): flooring `P` near zero produces wildly wrong (sign-flipped or
  enormous) results when `P` is small-but-nonzero or exactly 0 (e.g. the very first
  forcing evaluation from an all-zero initial condition), whereas capping the resulting
  scale factor bounds the worst case directly regardless of `P`'s units/scale.
- All `perp_*` energy-like reductions share one normalization convention (rfft2 `ky`
  y-doubling factor, divide by `nz*(nx*ny)^2`) to give a volume-averaged,
  grid-resolution-independent physical quantity — matches `diagnostics.perpspec`'s
  convention (`diagnostics.energy` now wraps `perp_inner_product_batch` — MPI-correct,
  returns `(E_kin, E_mag) = 0.5*<|grad|^2>` per field, same values the old local-slab
  real-space version gave at size==1; its former independent real-space Parseval check
  moved to `tests/test_energy_parseval.py`. `parspec` remains size==1-only per its
  assert). Keep any new energy-like diagnostic on this same convention or its
  numbers won't be comparable to `forcing_power`.
- `forcing_norm_per_step` (**default True**, the production config: ~+8% at fp64/32 ranks)
  computes the power-normalization scale once per full step — stored in
  `SimulationState.forcing_scale`, shape `(n_ou,)`, updated in `run.py::_advance_forcing`
  right after `ou_update` — and reuses it across RK sub-stages, replacing ~8 per-stage
  allreduces with 1 per step. The scale lags by one step (error O(dt/tau); larger bounded
  overshoot during quiescent spin-up — user signed off). Lifecycle (post-C1 design, see
  Checkpointing): in any state from `initialize`/`load_snapshot` it is ALWAYS a concrete
  `(n_ou,)` array (zeros when the flag or forcing is off — never None, or the on-disk
  orbax structure would fork), it IS serialized with the rest of the state, and it is
  recomputed by `_refresh_forcing_scale` at `simulate`/`simulate_scan` start regardless;
  hand-built states with `forcing_scale=None` fail with a clear trace-time error in
  `ForcingTerm` and are rejected by `save_snapshot`.
- `forcing_shell_noise` (default False) draws OU noise only at precomputed shell indices
  (`kgrid.fidx_x/fidx_y`) instead of the full k-grid: statistically identical but a
  *different RNG stream* (no bitwise reproduction of full-grid runs), faster single-device
  but measured ~5% slower on Savio CPU at 32 ranks — hence opt-in, revisit on GPU.
- 2D MHD (RMHD's `dims=2` limit) is *not* 2D hydro: with `forcing_mode="momentum"` and a
  quiescent start, `psi` stays exactly zero forever (its only 2D source term,
  `-bracket(gphi,gpsi)`, vanishes identically when `psi=0`) — that's pure hydro, use
  `"elsasser"` for actual MHD turbulence. Energy cascades forward/direct in 2D MHD
  (opposite of 2D hydro's inverse cascade), but mean-square flux function `<psi^2>`
  (`perp_mean_square`) inverse-cascades to large scales regardless — don't read a plateaued
  energy spectrum as "fully saturated" without also checking `<psi^2>` isn't still
  climbing. Per the zeroth law of turbulence / dissipative anomaly, `visc`/`res` set the
  time to reach saturation and the dissipation-range cutoff, not the saturated amplitude
  itself (given adequate resolution).

### Checkpointing

`snapshot_io.py` save/restore is orbax-based, one `CheckpointManager` per MPI rank
(`snapshot_manager_setup`) writing `snap_path/<rank>/<step>/` — except under
`comm_backend="jax"`, where all processes share ONE manager over ONE directory
(`snap_path/<step>/`) and hand orbax the global z-sharded arrays (orbax's native multihost
path). The two layouts are distinguished by `snapshot_layout()` (a `_CHECKPOINT_METADATA`
marker directly inside `snap_path/<n>` means `<n>` is a step, so the dir is "flat"), and
each backend can restart from either. **Reads never construct a `CheckpointManager`** — a
bare `StandardCheckpointHandler` is barrier-free, whereas manager/checkpointer restores
call `multihost.sync_global_processes` and deadlock as soon as ranks read different
directories. The serialized tree is the full `SimulationState`, and
`forcing_scale` is **always a concrete `(n_ou,)` array, never None**, in any state that
reaches `save_snapshot` (`initialize`/`load_snapshot` guarantee this) — orbax records
`None` children in its metadata, so a None-bearing save would fork the on-disk structure
between configs. `save_snapshot` hard-errors on a
None `forcing_scale` (hand-built states) rather than letting orbax fork the on-disk
structure. Snapshots written before `forcing_scale` existed (pre-Phase-1) fail to
restore with a structure mismatch; upgrade the snapshot directory once, in place, with
`snapshot_io.old_snapshot_repair(snap_path, params)` (handles single- and per-rank
layouts, plus the even older pre-forcing t/fields-only layout, for which it synthesizes
zero `forcing_state` and a key from `forcing_seed`; `load_snapshot` catches the mismatch
and points at it). The repair is interruption-safe: each step is staged fully before the
original is swapped aside via same-filesystem renames, and a rerun first auto-recovers
`.repair_old_*`/`.repair_tmp_*` leftovers of an interrupted swap. Note the repair makes
directories unreadable by pre-Phase-1 code. Any direct `StandardRestore` template must
include `forcing_scale` as a `(n_ou,)` ShapeDtypeStruct (see
`tests/test_restart_resharding.py`). `snapshot_manager_setup`'s `nsnap` arg is passed straight
through as orbax's `max_to_keep` — it used to be silently unwired (every run kept every
snapshot forever regardless of `nsnap`); it's now honored, so old runs' checkpoint
directories may hold more snapshots than a run made after this fix would produce.
`load_snapshot` supports restoring onto a different rank count
than was saved (`p_save` vs `params.size`) by unioning overlapping z-slices per field —
`forcing_state`/`forcing_key` are **not** part of that union (they have no z-axis and are
identical across all saved ranks by construction, since forcing is perpendicular-only and
kept in sync across ranks); they're restored once, directly from rank 0's checkpoint.
`forcing_key`'s dtype is obtained via `jax.eval_shape(lambda: jax.random.key(0)).dtype`
rather than a guessed public constant. Use `get_saved_steps(snap_path)` rather than
`mngr.all_steps()` to enumerate saved snapshot indices — `get_saved_steps` detects and
correctly handles a resharded (multi-rank-saved) layout, whereas a bare `mngr.all_steps()`
on a top-level manager over such a layout misreads the numbered rank subfolders as step
numbers.

Snapshot indices must mean the same simulation time on **every** rank — `load_snapshot`
assumes it, and orbax's `max_to_keep` prunes each rank's directory independently, so any
per-rank numbering skew compounds into different rank groups holding different index
windows. `run.py`'s `simulate`/`simulate_scan` therefore broadcast rank 0's starting index
(`max(all_steps())+1`) to all ranks rather than letting each rank derive its own: computing
it per-rank desynchronizes on restart-with-more-ranks, where pre-existing rank dirs resume
from their old latest step while brand-new (empty) dirs restart from 0 (the observed
symptom: after a 32→64 restart with `nsnap=20`, ranks 0–31 held snapshots 20–39 and ranks
32–63 held 16–35 — same data, offset indices, independently pruned). Regression test:
`tests/test_restart_resharding.py` (+ `slurms/test_restart_resharding.sh`), run `-n 2` then
`-n 4` on the same `snap_path`. The correct invariant it checks is NOT "all rank dirs hold
identical index sets": pre-existing ranks may legitimately retain *older* snapshots the new
ranks never existed to write. It is (a) identical latest index everywhere and (b) exact
agreement from the newest "oldest index" upward, plus identical `t` per common index.

Snapshot cadence caveat: `simulate`'s inner while-loop steps until `t >= target`, so with
large dt (e.g. near-zero fields early in a forced run from quiescence) it can overshoot a
`t_snap` target or `t_end` by a whole step — snapshots are "at least `t_snap` apart" and
the final time is ">= t_end", not exact. With `cfl_every>1` the while-loop iterates over
whole blocks, so the overshoot grows to up to `cfl_every` steps. Don't write tests (or
diagnostics postprocessing) that assume an exact snapshot count or exact end time. The snapshot target
(`t_last_snapshot`) advances every outer iteration regardless of `save` — it used to be
updated only when saving, which made `simulate(..., save=False)` with `t_snap < t_end`
spin forever once `state.t` passed the first frozen target (the inner while_loop returned
immediately, the outer python loop never advanced).
