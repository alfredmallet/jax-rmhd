# CLAUDE.md

Guidance for Claude Code working in this repository. Rationale and perf measurements
behind these rules: docs/PHASE3_RESULTS.md (incl. its appendix). Checkpointing detail:
docs/checkpointing.md.

## What this is

A pseudospectral solver for reduced MHD (RMHD) and related plasma fluid models, in JAX.
Spectral (rfft2) in the perpendicular (x,y) plane, 4th-order finite-difference in z,
MPI-decomposed along z only. Only RMHD is implemented; the architecture supports adding
other equation sets without touching the core solver.

## Setup / running

```
pip install -e .                     # needs a working MPI toolchain (mpi4py/mpi4jax)
pip install -e ".[examples]"         # adds matplotlib for notebooks/plot scripts
RMHD_PRECISION=64 python script.py   # float64/complex128; default 32. Read at import.
```

Tests: `make test` locally (two pytest sessions, fp64 then fp32; no MPI needed —
`tests/conftest.py` auto-installs `tests/local_mpi_stub.py` + 4 fake XLA devices when
mpi4py is absent). Test files are ALSO standalone scripts: `mpirun -n 4 python
tests/...` is the multi-rank driver on Savio — pytest is never run under mpirun.
New/converted test modules start with `from _rmhd_testing import bootstrap;
bootstrap()` BEFORE `import jax_rmhd`, and end with the `script_main(globals())`
footer; helpers live in `tests/_rmhd_testing.py` (never cache a SimulationState —
donation; never mutate `ctx()` results — identity-hashed jit cache). Markers: mpi,
savio, slow, fp32/fp64, multidev (skip logic in conftest + `_script_skip_reason`).
Legacy script-style files still run whole-module at pytest collection; they're listed
in `conftest._LEGACY_SCRIPTS` until converted. 2D (`dims=2`) is single-process only.
`bench/savio_scaling/` (scaling benchmark, not a test) and `slurms/` are
Savio-cluster-specific. How-to: docs/RUNNING_TESTS.md; roadmap: docs/TESTING_PLAN.md. Current notebooks: `orzag-tang-2D/3d`, `forced-turbulence-2D/3D`;
the others predate the API and will error.

## Architecture

### Field representation

`SimulationState` (NamedTuple, `types.py`) = `(t, fields, forcing_state, forcing_key,
forcing_scale)`. `fields` shape `(nfields, nz_local, nkx, nky)`: real-space in z, rfft2
in (x,y). **The z axis is never dropped** — `dims=2` gives `(nfields, 1, nx, ny//2+1)`.
`run.py::initialize` applies the 2/3 dealias mask to the IC (evolution masks only the
nonlinear term, so unmasked beyond-cutoff IC energy would persist and alias). `dims==3`
requires `nz % size == 0` (validated in `Parameters.__init__`).

rfft2 convention: `kx` full two-sided, `ky` half/non-negative — reality is a constraint
*between* `(kx,ky)` and `(-kx,ky)` at the ky=0 and Nyquist rows. Anything writing k-space
directly (e.g. stochastic forcing) must enforce it explicitly; when symmetrizing *noise*,
divide by sqrt(2), not 2 (`shared_physics._symmetrize_real_line`; derivation in the
PHASE3_RESULTS appendix). `grids.fft/ifft` are unnormalized: an O(1) real field has
O(nx*ny) coefficients — matters for resolution-independent synthetic k-space amplitudes.

### Parameters / physics registry

`params.save(snap_path)` records constructor args + precision to `params.json`; identical
re-save is a no-op, a differing existing record is a hard error. **Collective under
MPI** — never call from a subset of ranks. `Parameters.from_snapshot(snap_path,
**overrides)` re-runs `__init__`; overrides win, unknown keys warn, precision mismatch
warns. Both are explicit calls — nothing writes params.json automatically.

`Parameters` (`config.py`) is **not a pytree**. It is only closed over or passed static,
so every attribute is a compile-time constant — plain `if params.foo:` is correct and
preferred over `lax.cond`. Never pass it as a traced jit arg or inside a scanned tree.
z attributes (`dz`, `Lz`, `z_diss`, `cart_comm`, neighbors) exist only when `dims==3` —
guard access. `z_diff_order`/`z_diss_hyper` are accepted and stored but not read back by
`rmhd.LinearTerm` (see its TODO) — non-default values silently do nothing.

Equation sets register in `physics/__init__.py::equation_registry`:
`EquationRecipe(set_timestep_func, term_funcs, grad_func, nfields,
forcing_scale_func=None, halo_start_func=None)` per `eqtype`. `term_funcs` are summed
into the RHS (`construct_rhs`); dissipation is applied separately as an integrating
factor (`kgrid.hdiss` in `timestepping.py`), not as an RHS term. **Term funcs take 5
positional args** `(state, grads, kgrid, params, halo)` — declare `halo=None` and ignore
if unused. `halo_start_func` pre-issues the z-halo exchange at the top of the RHS;
enabled per backend (`_halo_start_enabled`: off for mpi4jax, on for `"jax"`),
overridable via `params.halo_start`. When off, `z_derivatives` does its own exchange.
`comms.halo_exchange(f, params, width=2)`: a pre-issued width narrower than the stencil
is an assertion failure (`z_derivatives` derives offsets from the received slab).
`physics/shared_physics.py` holds equation-agnostic helpers (`gradk`, `bracket`,
z-stencils, O-U forcing mechanics); `physics/rmhd.py` maps them onto (phi,psi).

All distributed transport goes through `comms.py`: `halo_exchange`, `allreduce_sum`,
`allreduce_max`, dispatched on the static `params.comm_backend`:

- `"mpi4jax"` (default, CPU production): mpi4py + mpi4jax; arrays stay process-local.
  Nothing outside comms.py imports mpi4jax — keep it that way.
- `"jax"` (GPU, shard_map/NCCL): control plane stays mpi4py; the three device ops become
  `ppermute`/`psum`/`pmax`, valid only inside `comms.shard_call` around the jitted
  steppers. State/kgrid become global z-sharded arrays (`comms.to_global`); inside
  shard_map physics sees the same local shapes, so physics code is backend-agnostic.
  `forcing_state`/`forcing_key` are replicated, never sharded. Constructing
  `Parameters(comm_backend="jax")` brings up `jax.distributed` — must be the first jax
  device work in the process; `"jax"`+`dims==2` is rejected. Launch flags/env:
  docs/SAVIO_GPU_SETUP.md + docs/PHASE3_RESULTS.md.

### Timestepping

RK/LSRK sub-stages rebuild states: **always `state._replace(...)`, never positional
`SimulationState(...)`** — positional construction silently drops/misaligns the forcing
fields, which must survive unchanged within a step (updated once per step, not per
sub-stage). `lsrk_advance` has scan (`lsrk_scan=True`, default) and unrolled stage
loops — agree to round-off at fp64 (~1e-15 after 20 steps; bitwise identity is
machine/jax-version dependent — held where first measured, does NOT hold under jax
0.6.2/CPU: XLA fuses the two loop structures differently; test_scheme_equivalence).
Per-machine perf knob.

`params.cfl_every` (default 1) recomputes the adaptive dt (and its CFL allreduce) once
per N-step block: `run._cfl_block` computes dt from the block's start state and passes
`dt_override` to the stepper. Never put the collective under `lax.cond`, never use a
rank-local dt — one collective, one dt, all ranks. `cfl_every=1` (and
`adaptive_timestep=False`) take the unchanged legacy path. A frozen dt can transiently
violate CFL — compensate with `cfl_safety`; **N>1 from a quiescent forced start silently
NaNs** (dt collapses ~10x during spin-up) — use only from developed states. Snapshot/t_end
overshoot grows to N steps. The forcing update still runs every step; `nblock` counts
steps, rounded up to whole blocks.

**Buffer donation consumes input states** (`donate_argnums=(0,)`): a state passed to
`simulate`/`simulate_scan` is invalidated ("Array has been deleted"). Continue from the
returned state; read diagnostics before the call.

`K_Grids` is a dumb pytree container — no methods, no lazy fallbacks; **`setup_kgrids` is
the only sanctioned constructor** (jax rebuilds the NamedTuple with tracers/specs as
field values). Never reuse a kgrid with a different or mutated `Parameters` — rebuild it;
mutating params after a jit trace silently reuses the stale compile (`Parameters` hashes
by identity).

### Stochastic forcing (`params.forcing`)

Ornstein-Uhlenbeck process injecting power into a perpendicular-wavenumber shell.
`forcing_state` shape `(n_ou, 2, nkx, nky)`: axis 0 is 1 (`forcing_mode="momentum"`, phi
only) or 2 (`"elsasser"`, z± = phi±psi independently, each with its own
`forcing_power_elsasser`); axis 1 is the [A,B] cos/sin z-envelope coefficients (dims=3;
dims=2 uses A only).

`forcing_power` and both entries of `forcing_power_elsasser` are in the SAME units — a
contribution to the total energy injection rate. Total dE/dt is `forcing_power` in
momentum mode and `eps_plus + eps_minus` in elsasser mode, so `(p/2, p/2)` matches
`forcing_power=p`. `rmhd._forcing_scale_from` carries the factor 2 this needs (E_tot =
(E+ + E-)/2); before 2026-07-31 the elsasser entries meant half this.

- `shared_physics` (`ou_update`, `reconstruct_envelope`, `perp_inner_product`,
  `perp_mean_square`) is equation-agnostic; `rmhd.ForcingTerm` does the RMHD power
  normalization and (phi,psi) mapping.
- Normalization targets exact injection power: cap the *scale factor*
  (`forcing_scale_max`), never floor the denominator `P` (rationale in the appendix).
- All `perp_*` reductions share one normalization (rfft2 ky-doubling, `/ nz*(nx*ny)^2`),
  matching `diagnostics.perpspec`/`energy` and `forcing_power` — keep new energy-like
  diagnostics on this convention or their numbers won't be comparable. `parspec` is
  size==1-only.
- `forcing_norm_per_step` (default True, production): computes the normalization scale
  once per step — stored in `SimulationState.forcing_scale` (`(n_ou,)`, updated in
  `run._advance_forcing`), reused across sub-stages; the scale lags one step. Lifecycle:
  in any state from `initialize`/`load_snapshot` it is ALWAYS a concrete `(n_ou,)` array
  (zeros when off — never None), it is serialized, and it is refreshed at
  `simulate`/`simulate_scan` start; `forcing_scale=None` errors at trace time and is
  rejected by `save_snapshot`.
- `forcing_shell_noise` (default False): draws OU noise only at shell indices —
  statistically identical but a *different RNG stream*; opt-in.
- `dims=2` + `"momentum"` from a quiescent start is pure hydro (`psi` stays exactly 0 —
  its only 2D source vanishes); use `"elsasser"` for actual 2D MHD. Physics context in
  the appendix.

### Checkpointing

**Read docs/checkpointing.md before touching `snapshot_io.py`** — layouts, restore rules,
resharding, index sync, old-snapshot repair. Invariants you must not break:

- **Reads never construct a `CheckpointManager`** (manager restores barrier and deadlock
  when ranks read different dirs); use a bare `StandardCheckpointHandler`.
- `forcing_scale` in any saved/loaded state is a concrete `(n_ou,)` array, never None.
- The snapshot starting index is broadcast from rank 0 — never derived per-rank.
- Enumerate saved steps with `get_saved_steps(snap_path)`, not `mngr.all_steps()`.
- `simulate` overshoots `t_snap`/`t_end` by up to one step (`cfl_every` steps in blocks):
  never assume exact snapshot counts or end times in tests/postprocessing.
