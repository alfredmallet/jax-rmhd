# CLAUDE.md

Guidance for Claude Code working in this repository. This file carries the rules; the
reasoning behind them lives in docs/numerics.md (derivations and conventions),
docs/performance.md (measurements and tuning) and docs/checkpointing.md.

## What this is

A pseudospectral solver for reduced MHD (RMHD) and related plasma fluid models, in JAX.
Spectral (rfft2) in the perpendicular (x,y) plane, 4th-order finite-difference in z,
MPI-decomposed along z only. Implemented equation sets: RMHD and 2D GDI; the
architecture supports adding others without touching the core solver.

## Setup / running

```
pip install -e .                        # laptop / no MPI toolchain: comm_backend="serial"
pip install -e ".[mpi]"                 # generic Linux box with a working MPI toolchain (mpi4py/mpi4jax); zsh needs the quotes
pip install -e ".[examples]"            # adds matplotlib for notebooks/plot scripts
TARANIS_PRECISION=64 python script.py   # FIELD precision only (float64/complex128; default 32); x64 always on, t always fp64. Read at import.
```

`RMHD_PRECISION` (the pre-2026-08-17 name) is rejected with an error, never honored:
setting it without `TARANIS_PRECISION` raises at import.

Tests: `make test` locally (two pytest sessions, fp64 then fp32; no MPI needed —
`tests/conftest.py` auto-installs `tests/local_mpi_stub.py` + 4 fake XLA devices when
mpi4py is absent). Test files are ALSO standalone scripts: `mpirun -n 4 python
tests/...` is the multi-rank driver on Savio — pytest is never run under mpirun.
New/converted test modules start with `from _rmhd_testing import bootstrap;
bootstrap()` BEFORE `import taranis`, and end with the `script_main(globals())`
footer; helpers live in `tests/_rmhd_testing.py` (never cache a SimulationState —
donation; never mutate `ctx()` results — identity-hashed jit cache). Markers: mpi,
savio, slow, fp32/fp64, multidev (skip logic in conftest + `_script_skip_reason`).
2D (`dims=2`) is single-process only.
`bench/savio_scaling/` (scaling benchmark, not a test) and `slurms/` are
Savio-cluster-specific. How-to: docs/RUNNING_TESTS.md. Every notebook in `examples/`
uses the current API; `examples/README.md` gives a suggested reading order.

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
divide by sqrt(2), not 2 (`shared_physics._symmetrize_real_line`; derivation in
docs/numerics.md). `grids.fft/ifft` are unnormalized: an O(1) real field has
O(nx*ny) coefficients — matters for resolution-independent synthetic k-space amplitudes.

**`params.z_spectral`** (default False; `dims==3` + `size==1` only, `comm_backend="jax"`
rejected) flips `grids.fft/ifft` (BOTH take `params`) from rfft2 to rfftn/irfftn over
`(z,x,y)`: shapes are unchanged, axis 1 means **kz**. Then reality reads
`F(-kx,-kz,ky)=conj(F(kx,kz,ky))` on the ky=0/Nyquist rows (both mirrors — anything writing
k-space directly must preserve it, and `i*kz` needs the kz-Nyquist plane zeroed exactly like
gdi's `ky_deriv`); `dealias` gains a 2/3 kz cut; `perp_reduce` divides by nz^2, not nz
(Parseval), which is what keeps `energy`/`perpspec`/forcing power identical to the real-z
computation; the parallel operator lives in `rmhd.linear_matrix` as `+-i*kz` off-diagonals
(`FDLinearTerm`/`halo_start` skipped, `set_timestep` drops 1/dz and z_diss); optional
`eqpars['z_diss_k']` (`-z_diss_k*kz^4`). Fields have the SAME shape in both modes and
different meaning — `z_spectral` is recorded in params.json and `params.save`'s
differing-record check is the ONLY thing stopping a cross-mode restart. Derivations:
docs/numerics.md; tests: `tests/test_z_spectral.py`.

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
`rmhd.FDLinearTerm`; `Parameters` warns when either is set away from its default (so is
`z_diss` under `z_spectral`, where every finite-difference-z knob is dead).

Per-equation physics parameters live in `params.eqpars` (a plain-JSON dict, recorded in
params.json, `{}` by default): RMHD reads `diss`/`hyper` (and the z_spectral-only `z_diss_k`) from it — they were ctor args
until 2026-08-01, and old records are folded into `eqpars` with a warning by
`from_snapshot`/`save`. `Parameters` hashes by identity, so a dict attribute is safe.

Equation sets register in `physics/__init__.py::equation_registry`:
`EquationRecipe(set_timestep_func, term_funcs, grad_func, nfields,
forcing_scale_func=None, halo_start_func=None, linear_matrix_func=None)` per `eqtype`.
`term_funcs` are summed into the RHS (`construct_rhs`); the k-local LINEAR part is not an
RHS term — `linear_matrix_func(kgrid, params) -> L` (convention `dt f = L f + N(f)`) is
built once by `setup_kgrids` into `kgrid.lin_L`/`lin_m`/`lin_s2`, and the steppers apply
it only through the `taranis.propagators` hook (`apply_exp`, `solve_shifted`, `scaled`;
backend chosen by L's shape — diagonal 4-d, putzer2 2x2 5-d). Never reintroduce
`kgrid.hdiss` or read `lin_*` from a stepper directly; the op order inside `apply_exp` is
the RMHD bitwise-equivalence gate (docs/numerics.md). **Term funcs take 5
positional args** `(state, grads, kgrid, params, halo)` — declare `halo=None` and ignore
if unused. `halo_start_func` pre-issues the z-halo exchange at the top of the RHS;
enabled per backend (`_halo_start_enabled`: off for mpi4jax, on for `"jax"` and `"serial"`
— serial's exchange is a pure slice, so pre-issuing it cannot change results),
overridable via `params.halo_start`. When off, `z_derivatives` does its own exchange.
`comms.halo_exchange(f, params, width=2)`: a pre-issued width narrower than the stencil
is an assertion failure (`z_derivatives` derives offsets from the received slab).
`physics/shared_physics.py` holds equation-agnostic helpers (`gradk`, `bracket`,
z-stencils, O-U forcing mechanics); `physics/rmhd.py` maps them onto (phi,psi).

`physics/<eq>.py` holds ONLY what the solver consumes: the recipe functions and their
helpers (including `_max_re_lambda`/`_lin_dt_safety`, which `set_timestep` calls).
Everything read-only and user-facing lives in `diagnostics/<eq>.py` — `diagnostics.rmhd`
(`energy`, `perpspec`, `parspec`) and `diagnostics.gdi` (`energy_enstrophy`,
`energy_budget`, `perp_spectrum`, `cross_phase_spectrum`, `kperp_break`, `measure_alpha`,
`theory_cross_phase`), with `diagnostics/core.py` for the shared machinery (`_binned`).
`diagnostics.<eqtype>` is the naming convention; the registry stays a solver contract, so
there is no diagnostics hook on `EquationRecipe` — diagnostics are plain imports, and the
dependency runs diagnostics -> physics, never back. The RMHD names plus `_binned` are
re-exported at the top level (`diagnostics.energy(...)` etc.) — that is the historical
surface, keep it working.

All distributed transport goes through `comms.py`: `halo_exchange`, `allreduce_sum`,
`allreduce_max`, dispatched on the static `params.comm_backend`:

- `"mpi4jax"` (CPU production; auto-selected whenever mpi4py **and** mpi4jax import):
  mpi4py + mpi4jax; arrays stay process-local. Nothing outside `_mpi_compat.py` imports
  mpi4py or mpi4jax at module scope (`bench/` and `tests/` excepted) — keep it that way.
- `"jax"` (GPU, shard_map/NCCL): control plane stays mpi4py; the three device ops become
  `ppermute`/`psum`/`pmax`, valid only inside `comms.shard_call` around the jitted
  steppers. State/kgrid become global z-sharded arrays (`comms.to_global`); inside
  shard_map physics sees the same local shapes, so physics code is backend-agnostic.
  `forcing_state`/`forcing_key` are replicated, never sharded. Constructing
  `Parameters(comm_backend="jax")` brings up `jax.distributed` — must be the first jax
  device work in the process; `"jax"`+`dims==2` is rejected. Launch flags/env:
  docs/SAVIO_GPU_SETUP.md; measured scaling in docs/performance.md.
- `"serial"` (single-process, no MPI installed): `comm_backend=None` (the default)
  auto-resolves to `"serial"` when mpi4jax isn't importable and the real/launcher world
  size is 1 — the expected laptop case, silent. Semantics are exact size-1, not approximate (halo exchange self-sends,
  allreduce is identity), but `"serial"` is NOT bitwise-identical to size-1 `"mpi4jax"`
  (dropping the mpi4jax ops changes XLA fusion — same class as `lsrk_scan`; compare with
  tolerances). Running under a detected multi-rank launcher without mpi4py installed is a
  hard `RuntimeError`, never a silent single-domain fallback — every rank would otherwise
  run the full domain and overwrite the others' output.

### Timestepping

RK/LSRK/IMEX sub-stages rebuild states: **always `state._replace(...)`, never positional
`SimulationState(...)`** — positional construction silently drops/misaligns the forcing
fields, which must survive unchanged within a step (updated once per step, not per
sub-stage). `lsrk_advance` has scan (`lsrk_scan=True`, default) and unrolled stage
loops — agree to round-off at fp64 (~1e-15 after 20 steps; bitwise identity is
machine/jax-version dependent — held where first measured, does NOT hold under jax
0.6.2/CPU: XLA fuses the two loop structures differently; test_scheme_equivalence).
Per-machine perf knob.

Two scheme families in `_scheme_registry`, one contract
(`stepper(state,kgrid,params,rhs,set_timestep,scheme,dt_override=None)`):

- **IF (integrating-factor)** — `rk44`, `lsrk33`, `lsrk54` (RMHD production). L applied
  exactly via `apply_exp`; treats the linear physics exactly but misweights the nonlinear
  forcing of stiffly DAMPED modes at |Reλ|·dt ≳ 1 (gives u ~ dt·N, not the quasi-static
  u ≈ N/γ — test_imex.py records the flat-in-γ error).
- **CB-IMEX** — `imexcb2`/`imexcb3e`/`imexcb3c`/`imexcb3f` (Cavaglieri & Bewley, JCP
  286:172 (2015); `imex2r_advance` 3 registers, `imex3r_advance` 4). ALL of L is implicit
  (dissipation included, no exponential in this path): one `solve_shifted(·, aᵢᵢ·dt)` per
  implicit stage plus `apply_L` — still only the `propagators` hook, never `kgrid.lin_*`.
  Recovers the quasi-static limit (L-stable, stiffly accurate) — the GDI/γ∥ path. The flip
  side: an L-stable solve artificially DAMPS oscillatory linear terms at |ω|·dt ≳ 1 —
  **never use an IMEX scheme on a wave-dominated L (e.g. z_spectral RMHD's ±i·kz) at large
  dt**; IF is the wave path. Error at fixed γ·dt is O(dt), not O(dt^order) (stage-order-1
  reduction — expected, all four schemes). `imexcb3e` is the recommended default (exactly
  rational coefficients, smallest stiff error constant). The [2R] stepper honors
  `lsrk_scan` (scan default; agrees with unrolled to round-off, same fusion caveat as
  lsrk); `imexcb3f`'s [3R] stepper is unrolled only (lookahead coefficients — the knob is
  not read). Coefficient tables are verified against order/coupling/
  L-stability conditions rebuilt from the stored values in test_imex.py — any edit to a
  tableau must keep that test green (history: a transcribed lsrk54 coefficient was wrong
  for years).

`params.cfl_every` (default 1) recomputes the adaptive dt (and its CFL allreduce) once
per N-step block: `run._cfl_block` computes dt from the block's start state and passes
`dt_override` to the stepper. Never put the collective under `lax.cond`, never use a
rank-local dt — one collective, one dt, all ranks. `cfl_every=1` (and
`adaptive_timestep=False`) take the unchanged legacy path. A frozen dt can transiently
violate CFL — compensate with `cfl_safety`. **N>1 from a quiescent forced start used to
blow up silently** (the spin-up kick collapsed dt ~10x inside a frozen block); the
2026-08-08 forcing-normalization fix removes the kick and the blow-up with it — re-measured
2026-08-08 at 64², elsasser, `cfl_every=4`, 240 steps from rest: finite at every
`eps_tot` in 1e-2…1 (pre-fix: `E ~ 6e58` within 60 steps at all three). It is still not
free: at `eps_tot=1` the frozen-dt block overshoots the `cfl_every=1` energy by ~10x
during spin-up before recovering, so **prefer developed states** for N>1 and check the
early energy trace if you do start from rest. Snapshot/t_end overshoot grows to N steps. The forcing update still runs every step; `nblock` counts
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
- Normalization targets exact injection power over ONE STEP, self term included
  (`shared_physics.selfnorm_scale`, since 2026-08-08): it solves
  `s·P·dt + ½·s²·F₂·dt² = target·dt` for the POSITIVE root, with `P = ⟨∇z·∇f_raw⟩` and
  `F₂ = ⟨|∇f_raw|²⟩`, then caps the *scale factor* (`forcing_scale_max`) — never floors the
  denominator `P`. The old `safe_scale` (`s = target/P`, the `F₂dt → 0` limit) is kept and
  is still what the per-stage path and the `F₂·dt == 0`/`dt == 0` guards use. Rationale,
  the two-branch cancellation-free evaluation, and the dated behaviour change:
  docs/numerics.md. **`forcing_scale_func(state, kgrid, params, dt)` takes a dt** —
  `run._advance_forcing` passes the just-completed step's (lagged, exact under `cfl_every`),
  `_refresh_forcing_scale` passes `0.0` (guard → `safe_scale`; a fresh `initialize` has
  `f_raw = 0` anyway). Never re-derive `dt` inside a term func.
- `rmhd._quiescent_dt` IS `rmhd.set_timestep`'s velocity floor: `set_timestep` returns
  `min(cfl_safety/max_all, _quiescent_dt(params))`, so the bound the per-stage scale cap
  relies on (`dt <= _quiescent_dt`) holds by construction on the adaptive path
  (`tests/test_quiescent_dt.py`). The floor value lives once, in
  `shared_physics.QUIESCENT_EPS` (`gdi.set_timestep` reads it too). On the fixed-dt path
  `_quiescent_dt` is `params.dt`, so a direct `set_timestep` call there also caps at
  `params.dt`.
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
  docs/numerics.md.

### Test particles (`taranis/particles/`, plans/TESTPART_PLAN.md — Phase A0/A1 landed 2026-08-18)

Boris-pushed charged test particles that see the RMHD fields and never back-react.
Derivation and conventions: docs/numerics.md "Test particles". Rules:

- Conventions (from `NonlinearTerm`'s bracket signs): u = ẑ×∇φ, b_⊥ = ẑ×∇ψ, Φ = B₀φ,
  A_z = −ψ, so E_⊥ = −B₀∇φ and **E_z = +∂ψ/∂t = −{φ,ψ} + L_ψψ + f_ψ** (2D and 3D; the
  finite-difference-z filter is NOT represented — Phase B decides). B₀ = 1, q/m carries the
  scale; no ε parameter. `assemble(pf, mask, B0)` is the ONLY place B₀ enters.
- `particles/fields.py::particle_fields(state,kgrid,params,*,resistive,forcing)` builds the
  piece-decomposed real-space `PFields` (RMHD-only assert). **`ez_ideal` is the DEALIASED
  bracket** `ifft(dealias·fft(−{φ,ψ}))` — the raw pointwise bracket is not the ∂ψ/∂t the
  discrete ψ obeys (gate 7 fails at O(1) with it). Resistive piece = ψ diagonal of
  `rmhd.linear_matrix` (never `apply_L`: under `z_spectral` that carries the ±i·kz term).
  Forcing piece = `ForcingTerm(...)[1]`. Per-ensemble mask over
  `FIELD_PIECES = (bperp, eperp, ez_ideal, ez_resistive, ez_forcing)`; defaults
  `ez_resistive = ez_forcing = False` (ideal-Ohm particle); `full_mask()` is the exact-∂ψ/∂t
  ensemble. Mask logic is static python — never `lax.cond`.
- Particle state is **fp64 always** (positions/velocities `(N,3)` float64 regardless of
  TARANIS_PRECISION); `interp.gather` casts samples up. The z axis is carried in every
  interface even though Phase A implements `nz_local == 1` only (assert). Positions are
  left UNFOLDED; only the gather folds mod L. `interp.gather_spectral` is validation-only.
- `boris.py`: `boris_kick`/`drift` are pure per-particle kernels (elementwise arithmetic
  on length-3 vectors, no `Parameters` — WGSL-portable, plan §9); `push` is the KDK
  driver (x, v synchronized at step boundaries, one gather per half-kick, fields frozen
  over the step). Keep the kernel free of jnp-only idioms.
- Gates: kernel gates 1–3 in `tests/test_particles_kernel.py` (both precisions), gate 7
  in `tests/test_particles_coupled.py` (fp64). In RMHD b_⊥ fields the naive grad-B drift
  is off by an O(1) factor (shear enters at O(ε), |B| at O(ε²)) — measured and derived
  in the kernel test; do not "fix" it toward the textbook value.
- **Gate-6 reference** (`tests/_gen_particles_gate6_reference.py`,
  `tests/data/particles_gate6_reference_fp{64,32}.npz`, force-added — `tests/data` is
  gitignored) was recorded on the pre-A2 tree: solver output with `params.particles=None`
  must stay bitwise identical to it. Regenerate only on a tree that has no particle wiring.
- Not yet (Phase A2+): `params.particles`, `ParticleState`, run.py carry tuple, checkpoint
  item, `diagnostics/particles.py`. Do not pre-empt them ad hoc.

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
