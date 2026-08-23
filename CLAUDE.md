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
The 4 fake devices are installed only when mpi4py does NOT import, so on a host that has
mpi4py `make test` skips every `multidev` test and **the `"jax"` backend is not covered
locally**; exercise it with `XLA_FLAGS=--xla_force_host_platform_device_count=4`, where
`tests/test_backend_jax.py::test_same_seed_run_matches_serial_reference` fails at 1.4e-13
in `forcing_state` on this host under jax 0.10.0 — pre-existing, identical on the
pre-refactor tree.
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
requires `nz % size == 0` (validated in `comms.Runtime.resolve` and in
`Parameters._validate_compat`).

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
z attributes (`dz`, `Lz`, `z_diss`) exist only when `dims==3` — guard access (`cart_comm`
and the neighbors always exist and are `None` outside a 3D non-serial run).
`z_diff_order`/`z_diss_hyper` are accepted and stored but not read back by
`rmhd.FDLinearTerm`; `Parameters` warns when either is set away from its default (so is
`z_diss` under `z_spectral`, where every finite-difference-z knob is dead).

Transport is not built by the constructor. `comms.Runtime` (a frozen, identity-hashed
dataclass — `backend, comm, rank, size, cart_comm, left_neighbor, right_neighbor`) comes
from `comms.Runtime.resolve(comm_backend=None, *, dims, nz, z_spectral=False)`, which
resolves the backend, takes `MPI.COMM_WORLD`, checks every rank reads the same
`TARANIS_PRECISION`, creates the z cartesian communicator and brings the backend up
(`jax.distributed` + the device mesh for `"jax"`). `Parameters(..., runtime=None)` resolves
one; an explicit `Runtime` is reused as is — one set of communicators for a whole parameter
scan, re-validated against that `Parameters`' own `dims`/`nz` (REFACTOR_PLAN §9.2: `nz` is
checked against `runtime.size`, so one runtime serves any `nz` its size divides) — and
`comm_backend` must then be `None` or that runtime's own backend. `params.runtime` holds it,
and `comm_backend`, `comm`, `rank`, `size`, `cart_comm`, `left_neighbor`, `right_neighbor`
are read-only properties forwarding to it: every read is unchanged, but **they are not
assignable** — spoof a rank with `dataclasses.replace(p.runtime, ...)`
(`tests/_rmhd_testing.py::fake_ranked_params`). `runtime` is never recorded in
`params.json`. The compatibility matrix (backend × dims × size, `nz % size`, `z_spectral`,
particles) is validated in `Parameters._validate_compat`, which is the one thing a
`Parameters` built on a shared `Runtime` runs; `nz % size`, `"jax"` × dims, the `z_spectral`
rejections and the jax device-count checks are ALSO in `Runtime.resolve`, guarding the
communicators and mesh that call creates — both carry them, with identical messages.

Per-equation physics parameters live in `params.eqpars` (a plain-JSON dict, recorded in
params.json, `{}` by default): RMHD reads `diss`/`hyper` (and the z_spectral-only `z_diss_k`) from it — they were ctor args
until 2026-08-01, and old records are folded into `eqpars` with a warning by
`from_snapshot`/`save`. `Parameters` hashes by identity, so a dict attribute is safe.

Equation sets register in `physics/__init__.py::equation_registry`:
`EquationRecipe(set_timestep_func, term_funcs, grad_func, nfields,
forcing_scale_func=None, halo_start_func=None, linear_matrix_func=None)` per `eqtype`.
`term_funcs` entries are `physics.Term(func, active=...)` (a bare callable is accepted and
wrapped, meaning always active) and `construct_rhs` sums only the terms whose
`active(params)` is true — plain python at TRACE time, `params` being static, so an
inactive term never enters the graph; an empty selection is a `ValueError`. The predicates
are the single source of truth: `rmhd.fd_linear_active` (`dims==3 and not z_spectral`,
which `rmhd.halo_start` reads too) and `rmhd.forcing_active`. The term funcs keep their
early `zeros_like` returns, expressed through those predicates, for the direct callers that
reach them (`tests/test_forcing_smoke.py:125`, `tests/test_z_spectral.py:284`); from the
RHS they are never reached.

The k-local LINEAR part is not an RHS term — `linear_matrix_func(kgrid, params) -> L`
(convention `dt f = L f + N(f)`) is validated by `propagators.build` and stored by
`setup_kgrids` in the SINGLE slot `kgrid.lin`, always populated, as one of four operator
pytrees (NamedTuples of arrays carrying the methods): `IdentityOperator()` (a recipe with
no `linear_matrix_func`), `DiagonalOperator(L)` (a 4-d diagonal L — FD-z and 2D),
`Putzer2Operator(L, m, s)` (dense 2×2 5-d L, with `m` and `s = sqrt((tr L/2)^2 - det L)`
precomputed at SETUP — never `lin_s2`, which Z2 retired) and `SeparableL(dperp, dz, kz)`
(what `rmhd.linear_matrix` returns for `z_spectral` with `diss[0]==diss[1]`). The steppers
apply it only through the `taranis.propagators` hook — `apply_exp`, `solve_shifted`,
`scaled`, `apply_L`, `exp_op` and the `hoistable` property; `dense()` materialises the 2×2
L for tests and validation and is never formed inside a step. Never reintroduce
`kgrid.hdiss`, and never read `kgrid.lin`'s arrays from a stepper — steppers use its
methods only; the op order inside `apply_exp` is the RMHD bitwise-equivalence gate
(docs/numerics.md). Two traps: an empty NamedTuple is falsy, so never test `kgrid.lin` for
truthiness; and never index the operator positionally outside its own methods.

**Term funcs take 5 positional args** `(state, grads, kgrid, params, halo)` — declare
`halo=None` and ignore if unused. **`grads` is a NAMED TUPLE** —
`rmhd.RMHDGrads(gphi, gpsi, gvort, gjpar)` / `gdi.GDIGrads(gphi, gN, gvort)`, one
real-space `(2,nz,nx,ny)` array per field in the equation set's `grad_func` order, never a
stacked `(nfields,2,…)` array. Read it by name or unpack it — a NamedTuple is a tuple, so
positional consumers keep working — but never `.shape`/`grads[:2]` it.
`shared_physics.grad_fields` returns the plain tuple; the recipe names it. It comes from
`shared_physics.grad_fields(fks, kgrid, params)`, which transforms
`shared_physics.GRAD_CHUNK` fields per inverse transform (module constant, read at TRACE
time, not a `Parameters` attribute; the probe's `--grad-chunk` sets it). **`GRAD_CHUNK = 1`
is the default everywhere** and transforms per COMPONENT with a free per-field restack;
`>1` batches whole fields per ifft. `grad_fields`' OUTPUT is bitwise identical at every
chunk size (`tests/test_grad_memory.py`) and `4` reproduces the pre-F1 graph byte-exactly;
the STEPPED 3D solution is not — chunk 2 and 4 differ from chunk 1 by one ulp (637/2304
elements, rel 1.1e-16, 16²×8 FD-z, 6 steps, from XLA fusing the restacked gradient
differently inside the step), 2D exactly bitwise. So it is a per-device speed/memory knob,
not a no-op on the solution (measured splits: docs/performance.md — CPU wants 1 outright,
P100 gains 3–20% from 2, 2080Ti is indifferent). `halo_start_func` pre-issues the z-halo exchange at the top of the RHS;
enabled per backend (`_halo_start_enabled`: off for mpi4jax, on for `"jax"` and `"serial"`
— serial's exchange is a pure slice, so pre-issuing it cannot change results),
overridable via `params.halo_start`. When off, `z_derivatives` does its own exchange.
`comms.halo_exchange(f, params, width=2)`: a pre-issued width narrower than the stencil
is an assertion failure (`z_derivatives` derives offsets from the received slab).
`physics/shared_physics.py` holds equation-agnostic helpers (`grad_fields`/`gradk`,
`bracket`, z-stencils, O-U forcing mechanics — `grad_fields` is the one physics calls;
`gradk` is the batched k-space gradient it uses at `GRAD_CHUNK > 1`, otherwise a
bench/test entry point); `physics/rmhd.py` maps them onto (phi,psi).

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
  `forcing_state`/`forcing_key` are replicated, never sharded. `comms.Runtime.resolve`
  brings up `jax.distributed`, so the first `Runtime.resolve` (directly, or via the first
  `Parameters(comm_backend="jax")`) in a process must precede any jax device work in it;
  `"jax"`+`dims==2` is rejected. Launch flags/env:
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
(`stepper(state,kgrid,params,rhs,set_timestep,scheme,dt_override=None,exp_ops=None)`):

- **IF (integrating-factor)** — `rk44`, `lsrk33`, `lsrk54` (RMHD production). L applied
  exactly via `apply_exp`; treats the linear physics exactly but misweights the nonlinear
  forcing of stiffly DAMPED modes at |Reλ|·dt ≳ 1 (gives u ~ dt·N, not the quasi-static
  u ≈ N/γ — test_imex.py records the flat-in-γ error).
- **CB-IMEX** — `imexcb2`/`imexcb3e`/`imexcb3c`/`imexcb3f` (Cavaglieri & Bewley, JCP
  286:172 (2015); `imex2r_advance` 3 registers, `imex3r_advance` 4). ALL of L is implicit
  (dissipation included, no exponential in this path): one `solve_shifted(·, aᵢᵢ·dt)` per
  implicit stage plus `apply_L` — still only the `propagators` hook, never `kgrid.lin`'s
  arrays.
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

**Hoisted stage propagators** (`params.hoist_propagator`, default True): whenever dt is frozen
over a block — fixed dt, or one `cfl_every` block — `run.py` forms every IF stage's
`exp(L·tau)` ONCE per block (`timestepping.stage_exp_ops(kgrid, params, scheme, stepper, dt)`
→ a tuple of `propagators.ExpOp` pytrees: `Putzer2Exp`/`SeparableExp`/`DiagonalExp`/
`IdentityExp`, each with `.apply(arr)`) and passes it to the stepper's `exp_ops=` kwarg
(every stepper in the registry takes it; IMEX ignores it and `stage_exp_ops` returns None
for them). Whether a backend is hoisted is `kgrid.lin.hoistable`, and the three answers are
different in kind:

| backend | selected for | hoisted | per-stage ops | what hoisting buys |
|---|---|---|---|---|
| diagonal | FD-z, 2D | **no** | one real exp per mode, z-broadcast | nothing — and staying unhoisted keeps the FD-z/2D fixed-dt graph byte-identical to the pre-hoist solver (gate 6's reference: with a literal `gamma` XLA folds `(L·dt)·gamma` differently, 15 elements at 1e-23 in the 64² gate-6 config) |
| separable | z_spectral RMHD, ν = η | yes | `(nkx,nky)` + 2×`(nz,1,1)` reals per stage, ~free (≤0.1 u for the whole stack) | amortises the per-stage exp/cos/sin EVALUATION: 1.02–1.13× unhoisted vs 0.94–0.98× hoisted against the old putzer2 fixed-dt step |
| putzer2 | GDI-IF, ν ≠ η z_spectral RMHD | yes | 4 complex arrays of L's full shape per stage | the real time/memory trade — see below |

**The knob is a putzer2 knob** (§9.3 decision, 2026-08-20): it is the only backend where
turning it off saves anything, and turning it off there costs time. Measured: z_spectral
ν ≠ η lsrk54 is 44.4 u hoisted against 36.0 u unhoisted (~8 u over 5 stages — the gate in
`tests/test_hoist_propagator.py`), while at lsrk33 the same gap is under 1 u (Z2's `w`-form
removed the full-grid temporaries the unhoisted stage used to carry); GDI-IF is
1.22× slower unhoisted on the P100 and 1.80× on XLA:CPU. **The memory price is a scheduler
property, not an arithmetic one** — on XLA:CPU the GDI 2D 256² lsrk33 pair is memory-NEUTRAL
(41.130 u hoisted, 41.091 u unhoisted: unhoisted, XLA keeps a 4 u copy of the dense `lin.L`
plus seven 1 u `_coeffs` lanes live inside the stage scan, which costs what the hoisted stage
coefficients cost), where the same pair on the P100 is a real +7.0 u. Size the trade on the
device you will run on. Nothing to decide on the separable path (hoisting is ~free and on),
and nothing to decide on FD-z/2D (the knob does not reach them).

The ExpOps are exactly the arrays `apply_exp` forms, in the same op order — `apply_exp(arr,
tau)` IS `exp_op(tau).apply(arr)` — so hoisted and unhoisted agree bitwise at fp64 on the test
grids (`tests/test_hoist_propagator.py`; fp32 has one 1-ulp fusion cell). `exp_ops=None` is
the legacy graph: the exponent evaluated inside each stage — under `lsrk_scan` inside the
stage scan, where `gamma` is a scanned value — which is what keeps `hoist_propagator=False`
memory-light (XLA's own loop-invariant code motion would otherwise hoist ops formed outside
the stage scan with static `gamma`; do not "simplify" the unhoisted branch into that form —
`test_unhoisted_graph_stays_memory_light` is the guard, and it fails at exactly 0.00 u of gap
under that restructure). Both `run.py` block functions — `_cfl_block` and `_advance_block` —
compute the ops OUTSIDE their step scan (`_hoisted_exp_ops` after `_block_dt`, or from
`_fixed_dt(params)`) — keep it there, that placement is the whole point. **The rule is
review-enforced and nothing else**: forming the ops inside the scans instead leaves the
memory-light gate green (it measures the `exp_ops=None` branch) and every HLO histogram
identical, because XLA's LICM hoists the loop-invariant formation itself. On adaptive
`cfl_every=1` nothing is frozen, so nothing is hoisted on any backend; that is the path Z2's cheaper putzer2 coefficients and Z1's separable
backend address instead.

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
  `simulate`/`simulate_scan` start only when it is all zeros — a stored nonzero scale
  survives entry unchanged, which is what makes a restart bitwise;
  `forcing_scale=None` errors at trace time and is rejected by `save_snapshot`.
- `forcing_shell_noise` (default False): draws OU noise only at shell indices —
  statistically identical but a *different RNG stream*; opt-in.
- `dims=2` + `"momentum"` from a quiescent start is pure hydro (`psi` stays exactly 0 —
  its only 2D source vanishes); use `"elsasser"` for actual 2D MHD. Physics context in
  docs/numerics.md.

### Test particles (`taranis/particles/`, plans/TESTPART_PLAN.md — Phase A landed 2026-08-18, Phase B (3D, single-process) 2026-08-19)

Boris-pushed charged test particles that see the RMHD fields and never back-react.
Derivation and conventions: docs/numerics.md "Test particles". Rules:

- Conventions (from `NonlinearTerm`'s bracket signs): u = ẑ×∇φ, b_⊥ = ẑ×∇ψ, Φ = B₀φ,
  A_z = −ψ, so E_⊥ = −B₀∇φ and **E_z = +∂ψ/∂t = −{φ,ψ} + L_ψψ + f_ψ** (2D and 3D).
  `assemble_stacked(pf, mask, B0)` is the ONLY place B₀ enters the fields the pusher sees (diagnostics read the
  same `ens["B0"]`; B_z is stored at FIELD precision, so the sidecar `mu` and `mu_of` are
  bit-consistent at fp64, and at fp32 only for an fp32-representable B₀). **B₀ IS the RMHD amplitude parameter, B₀ = 1/ε**
  (derivation: docs/numerics.md): B₀ = 1 means δB/B₀ ~ 1, production runs B₀ ~ 10 with q/m
  scaled by 1/B₀ so Ω = qm·B₀ — hence ρ, Ω·dt, ξ — is unchanged; β_i = v_th²/B₀². It is a
  PER-ENSEMBLE key (top-level = default), so one run can hold several amplitudes —
  **in 2D only**. **`B₀` must be 1.0 for every ensemble when `dims==3`** (rejected in
  `Parameters`): the solver's Alfvén coefficient is exactly 1 (`linear_matrix`'s
  off-diagonal is `1j·kz`, `FDLinearTerm` a bare `df_dz`), so any other B₀ leaves
  (1−B₀)∂_zφ in E_z and the field stops satisfying ideal Ohm. In 3D ε = rms|∇ψ| is set by
  the forcing amplitude and Lz instead (v_A = 1 on Lz ≡ v_A = B₀ on B₀·Lz).
- `particles/fields.py::particle_fields(state,kgrid,params,*,resistive,forcing)` builds the
  piece-decomposed real-space `PFields` (RMHD-only assert). **`ez_ideal` is the DEALIASED
  bracket** `ifft(dealias·fft(−{φ,ψ}))` — the raw pointwise bracket is not the ∂ψ/∂t the
  discrete ψ obeys (gate 7 fails at O(1) with it). Resistive piece = the FULL linear
  non-ideal EMF on ψ: the ψ diagonal of `rmhd.linear_matrix` (never `apply_L`: under
  `z_spectral` that carries the ±i·kz term)
  PLUS `FDLinearTerm`'s −z_diss·(dz/2)⁴∂_z⁴ψ filter when `dims==3 and not z_spectral`
  (`_psi_non_ideal`; its Alfvén half is the ∂_zφ term already cancelled out of E_z, and
  stays out). Still 5 `FIELD_PIECES`, 4 `WORK_PIECES` — the filter is not a new piece.
  Forcing piece = `ForcingTerm(...)[1]`. Per-ensemble mask over
  `FIELD_PIECES = (bperp, eperp, ez_ideal, ez_resistive, ez_forcing)`; defaults
  `ez_resistive = ez_forcing = False` (ideal-Ohm particle); `full_mask()` is the exact-∂ψ/∂t
  ensemble. Mask logic is static python — never `lax.cond`.
- `assemble_stacked(pf, mask, B0) -> (F, ez_on)` is the production assembly: ONE array
  `[B0·ex, B0·ey, <the ez pieces the mask keeps, in order>, bx, by, B0]` plus the static
  tuple of kept piece names, so the pusher gathers E and B in one call and can attribute
  the work per piece. `assemble` is its summed `(E, B)` form — same code path, no second
  implementation. `WORK_PIECES = (eperp, ez_ideal, ez_resistive, ez_forcing)` (`NWORK = 4`)
  are the ELECTRIC pieces, in the order `ParticleState.w`'s last axis stores them.
- Particle state is **fp64 always** (positions/velocities `(N,3)` float64 regardless of
  TARANIS_PRECISION); `interp.gather` casts samples up. Positions are
  left UNFOLDED; only the gather folds mod L. `interp.gather` is periodic bilinear at
  `nz == 1` (the 2D path, bitwise unchanged) and periodic **trilinear** when the grid has a
  z axis; `init_particles` draws z uniform over Lz in 3D and exactly 0 in 2D (the 2D RNG
  stream draws `(n,2)`, unchanged). `interp.gather_spectral` is validation-only and is
  exact for what the representation is: fully spectral under `z_spectral`, perp-spectral
  and linear-in-z (hence exact for a z-independent field) under finite-difference z.
- `boris.py`: `boris_kick`/`drift`/`project_perp` are pure per-particle kernels (elementwise
  arithmetic on length-3 vectors, no `Parameters` — WGSL-portable, plan §9); `push_tracked` is the KDK
  driver (x, v synchronized at step boundaries, ONE gather of the stacked `F` per half-kick,
  fields frozen over the step), and `push(x,v,E3,B3,...)` is a thin bitwise wrapper over it
  (`nez=1`, E and B concatenated) that the kernel gates use. Keep the kernel free of
  jnp-only idioms.
- `push_tracked` also returns `(dw, bsample)`: `dw` `(N, 1+nez)` fp64 is the work per unit
  mass over the call, column 0 by E_⊥ and column 1+i by ez piece i, accumulated per
  half-kick as `h·E·(v_in + v_new)` with `h = qm·dt_k/2`. That is EXACT Boris algebra (the
  rotation is norm-exact), so `sum(dw) == ½|v_out|² − ½|v_in|²` to round-off — gate 8.
  `bsample` `(N,3)` is the last half-kick's B sample, i.e. B at the returned position, so
  the local-B μ costs no extra gather. Derivation: docs/numerics.md "Work bookkeeping".
- **E∥ projection** (`boris.project_perp`, `push_tracked(..., project=True)`, static flag):
  replaces each gathered sample by `E − (E·B)/(B·B)·B` BEFORE the kick and before the work
  columns are formed, so the particle sees E·B = 0 exactly and the closure stays exact. It
  removes the numerical E∥ that independent bilinear interpolation of E and B leaves
  (measured 1.4e-3 of |E||B| at 64², down to 4e-17 — gate 9). Per-ensemble
  `epar_project` (default False) REQUIRES the exact ideal-Ohm mask
  (`bperp=eperp=ez_ideal=True`, non-ideal ez off) — anything else is a ValueError naming the
  ensemble, and `nez == 1` is a static assert in `push_tracked`: with the resistive/forcing
  pieces on, E∥ is partly real and projecting would delete it.
- Gates: kernel gates 1–3 in `tests/test_particles_kernel.py` (both precisions), gate 7
  in `tests/test_particles_coupled.py` (fp64), gate 8 (work closure, off pieces exactly
  zero, the resistive-split discriminator, `push` vs `push_tracked` bitwise) and gate 9
  (E∥ projection: E'·B round-off at the particles, unprojected NOT zero, closure and orbits
  still right; per-ensemble B0 leaves the δb=0 control bitwise) there too, in
  both precisions — the closure holds at fp32 fields because `w` and the push are fp64.
  In RMHD b_⊥ fields the naive grad-B drift
  is off by an O(1) factor (shear enters at O(ε), |B| at O(ε²)) — measured and derived
  in the kernel test; do not "fix" it toward the textbook value.
- **Gate 10 (mirror force, `tests/test_particles_kernel.py`, both precisions)** is the only
  parallel-dynamics gate: static analytic `B = B₀ẑ + ẑ×∇ψ(x,z)` on a 3D grid, E = 0, launched
  in the well at z = Lz/4 where the field is exactly B₀ẑ. It pins the reflection point against
  `|B|_turn = |B|₀·|v|²/v_⊥0²` over a v_∥0 sweep. **μ's violation is asserted by its SCALING,
  never as a fixed number**: a bouncing orbit shows the REVERSIBLE finite-Larmor excursion,
  first order in ρ·k_z (0.19·ρk_z, flat over an 8× sweep), NOT the exponentially small secular
  drift — which the gate separates by gyrophase-averaging back at the launch plane (1e-5, 230×
  below the excursion). The gap is 3D-only: in a static z-independent field the exact `|v|` and
  `p_z` already fix the parallel/perpendicular split from the perpendicular position, which the
  gate's z-independent control measures.
- **Gate 11 (varying dt)** is the pusher's only adaptive-dt coverage: kernel half a
  ±60%-jittered dt sequence at fixed total time (|v| exact, orbit O(max dt²), E×B exact),
  coupled half gate 4 live under `adaptive_timestep=True` (order 0.978 in `cfl_safety` against
  the fixed-dt gate's 0.948). Drift is quoted PER UNIT TIME — `block_of_steps` takes a step
  count, so runs at different `cfl_safety` end at different t.
- `tests/test_particles_3d.py` is the 3D gate set, every gate run in BOTH z modes: the
  embedded-2D `p_z` invariant (z is ignorable only for z-independent fields — an unforced
  z-independent 3D state stays z-independent exactly, the O-U z envelope is what breaks
  it), gate 7 as E_z vs `Δψ/Δt − ∂_zφ` (the Alfvén term is NOT part of E_z), the FD-z
  filter discriminator, the B₀ = 1 discriminator (grid `max|E·B|` = B₀|1−B₀|·max|∂_zφ|
  exactly, and 2D stays at round-off for every B₀), and gates 5/6/8/9 in 3D. Kernel gates
  1–3 have no 3D analogue (analytic fields, no z structure) and no new gate-6 reference
  npz was recorded — the existing one is a 2D artifact of the A2 wiring.
- **Gate-6 reference** (`tests/_gen_particles_gate6_reference.py`,
  `tests/data/particles_gate6_reference_fp{64,32}.npz`, force-added — `tests/data` is
  gitignored) was recorded on the pre-A2 tree and is what `tests/test_particles_coupled.py`
  compares against: solver output with `params.particles=None` must stay bitwise identical
  to it. Regenerate only on a tree that has no particle wiring.
- **Refactor reference** (`tests/_gen_refactor_reference.py`,
  `tests/data/refactor_reference_fp{64,32}.npz` + `refactor_reference_hlo_fp{64,32}.json`,
  force-added) is the same kind of standing gate for the SOLVER paths gate 6 does not
  cover: twelve configs (3D FD-z, z_spectral separable and putzer2, GDI IF and IMEX,
  fixed/cfl-block/adaptive dt, rk44, imexcb3f, hoist on/off), compared by
  `tests/test_refactor_reference.py` in `fields` and `t` bitwise AND in the optimized-HLO
  opcode histogram (plus instruction and fusion totals) of the jitted `block_of_steps` — so
  a structural change that keeps the numbers but moves the compute is caught too. The
  generator DEFINES the configs and the test imports them. Host-keyed like gate 6
  (print-skip off the recording host) and under the same rule: never regenerate to make it
  pass.
- `params.particles` (default `None` = off): a plain-JSON dict like `eqpars`, normalized
  by `taranis.particles.state.normalize_config` (imported inside the ctor, not at module
  scope, so `config` itself does not depend on the particle package — there is NO import
  cycle; `run.py` imports the package eagerly anyway) into `self.particles`;
  `self._init_args["particles"]` keeps the raw dict
  so `save()`/`from_snapshot()` round-trip it (list/tuple-tolerant, re-save is a no-op).
  Requires `eqtype=="RMHD"`, `size==1` and a non-sharded backend (`comm_backend!="jax"`);
  `dims` 2 or 3, both z modes, with `B0 == 1.0` enforced in 3D (above). z-decomposed
  particles are unimplemented — ValueError pointing at plans/TESTPART_PLAN.md §4, which
  carries the design note. Schema: `seed`/`substeps`/`B0`/`init_on_restart` plus a
  non-empty `ensembles` tuple, each `{qm, init: {kind: "maxwellian"|"ring", ...}}` with the
  optional `B0` (> 0; the top-level one is its default and `ens["B0"]` is ALWAYS present
  after normalization — `push_ensembles`/`mu_of` read it, never `cfg["B0"]`) and
  `epar_project`; `n` is
  the particle count PER ENSEMBLE, not the total. Each ensemble's raw `FIELD_PIECES` keys
  (bperp/eperp/ez_ideal/ez_resistive/ez_forcing) are resolved through `fields.resolve_mask`
  into `ens["mask"]` — the raw per-piece keys are consumed, not kept alongside it, and an
  already-normalized `"mask"` is accepted back (`normalize_config` is IDEMPOTENT; a piece
  given both inline and in `"mask"` must agree or it is a ValueError).
  `params.n_ens = len(params.particles["ensembles"])` is absent when particles are off —
  guard access like the z attributes.
- `ParticleState` (`particles/state.py`; `x`/`v` each `(n_ens, n, 3)` fp64, `w`
  `(n_ens, n, NWORK)` fp64 = cumulative work per unit mass per piece since init, zero from
  `init_particles` and EXACTLY zero forever for a piece the ensemble's mask omits) rides in
  the run CARRY next to `SimulationState` — never a `SimulationState`
  field, so the on-disk state layout and every particles-off code path stay untouched
  (gate 6). The carry is ALWAYS `(state, aux)`, `aux` being the `ParticleState` when
  particles are on and `None` off (a leafless pytree: scan/while_loop/donation see the same
  leaves either way), and `_step`/`_cfl_block`/`_advance_block` always return `(carry, ys)`
  with `ys = (t, mom)` on and `None` off. `block_of_steps` is the public wrapper and the
  only branch on the carry SHAPE, keeping its contract: off, state in → state out; on,
  `(state, pstate)` in → `((state, pstate), ys)` out. Particles are rejected on
  `comm_backend="jax"`, so on that backend the jitted `shard_call` boundary carries the
  bare state — `run.py` wraps `(s, None)` around it inside the jit and unwraps on the way
  out; `comms.shard_call` never sees the tuple carry.
  `simulate`/`simulate_scan(..., pstate=)` is REQUIRED iff `params.particles` is
  set (a pstate with particles off is also a ValueError); both return `(state, pstate)`
  when on, plain `state` when off.
- `push_ensembles(...) -> (pstate, mom)` and `moments(pstate, bsample) -> (n_ens, NMOM)`:
  `MOMENTS = (vperp2, vz2, vz, vperpB2, vparB2, mu, mu2, w_eperp, w_ez_ideal,
  w_ez_resistive, w_ez_forcing)` (`NMOM = 11`), per-ensemble means. `vparB2 = (v·B)²/|B|²`,
  `vperpB2 = |v|² − vparB2` and `mu = v_⊥B²/(2|B|)` all use the LOCAL B from the push's
  `bsample`, not v_z²/v_x²+v_y² about ẑ — **heating is measured in the local-B frame**
  (docs/numerics.md); `w_<piece>` is the mean of `w[..., i]`.
  Only `push_ensembles` computes moments — the run bodies just emit what it returns.
- `_advance_particles(pstate, prev_state, new_state, kgrid, params)` mirrors
  `_advance_forcing`'s placement: runs after the stepper and after `_advance_forcing`
  (whose result it does not read), with fields assembled from the PRE-step state (frozen
  at t_n) and `dt = new_state.t - prev_state.t`.
- `simulate_scan(..., save=True)` appends each `advance()` call's `ys` to a sidecar
  `<mngr.directory>/particle_moments.txt` (header `# t ensemble ` + `MOMENTS`, one row per
  step per ensemble, `%.17g`, append mode — a restart just continues the file; rows with
  `t > t_restart` are dropped on entry, so a restart from an older snapshot never leaves
  duplicate times). `simulate`'s
  while_loop carries the tuple too but the while_loop can't emit scan ys, so it only gets
  snapshot-cadence particle diagnostics — production heating runs use `simulate_scan`.
- Checkpoint: `pstate` rides the same snapshot step as a SEPARATE orbax item
  (`snapshot_io.PARTICLES_ITEM = "particles"`, `ocp.args.Composite` save); the state item
  `_ITEM = "default"` stays STRUCTURALLY identical (same subtree, keys, array metadata and
  values — what gate 6b tests), not byte-for-byte (ocdbt data-file names are hashed per
  write, and the step's `_CHECKPOINT_METADATA` lists the extra item handler).
  `load_particles(isnap, snap_path, params)` reads
  it under the same bare-`StandardCheckpointHandler` rule as `load_snapshot`. A snapshot
  missing the `particles` item hard-errors (`FileNotFoundError` naming `init_on_restart`)
  UNLESS `params.particles["init_on_restart"]` is set, in which case it prints a notice and
  returns `init_particles(params)` — never a silent re-init; a missing STEP is a
  `FileNotFoundError` either way (the flag covers a missing ITEM only). Particle items
  written by the A2 tree (x, v only) are NOT restorable — `template()` now carries `w` and
  there is no migration.
- RNG: `jax.random.key(params.particles["seed"])` is used ONLY in `init_particles`
  (`jax.random.fold_in(key, ensemble_index)`, then split for x/v) — the push itself is
  deterministic, so the forcing RNG stream is untouched by construction.
- `boris.push_tracked/push(..., gather=interp.gather)` is swappable: validation drives the
  identical push through `interp.gather_spectral` on rfft2 arrays.
- Restart is bitwise at every `forcing_norm_per_step` setting (2026-08-22):
  `_refresh_forcing_scale` computes a scale on `simulate`/`simulate_scan` entry only for a
  state whose `forcing_scale` is all zeros (a fresh `initialize`, a repaired legacy
  snapshot), so a checkpoint's stored lagged scale is the one the next step uses and a
  particle restart reproduces the uninterrupted trajectory bitwise. Gates:
  `tests/test_forcing_norm_per_step.py::test_restart_is_bitwise_at_default_norm` and gate
  6c at both settings.
- `ctx()` (`tests/_rmhd_testing.py`) caches on `tuple(sorted(kwargs.items()))`, so it
  CANNOT take `particles=` (a dict is unhashable) — use `fresh_params(particles=...)`.
- `diagnostics/particles.py` is the read-only observer side (plain imports, dependency runs
  diagnostics → particles, never back; listed in `diagnostics/__init__.py`'s `__all__` with
  NO name re-exports). Host numpy; `jnp` only where a gather/fft is needed (`mu_of`,
  `jz_at`, `kinetic_spectrum`). Conventions to keep: `read_moments` parses the sidecar BY
  HEADER NAME (widening `MOMENTS` never breaks old files; a repeated `(t, ensemble)` row is
  a ValueError; the file name is duplicated there, not imported from `run.py`);
  `heating_rate` halves ONLY the velocity-square columns (`vperp2`, `vz2`, `vperpB2`,
  `vparB2`) and its `err` is
  the block-slope scatter, never the OLS residual error; `kinetic_spectrum` is `perpspec`'s
  phi column so `0.5⟨u²⟩ = ∫E_kin dk` and `delta_u(rho) = sqrt(2 k E_kin(k))` at `k = 1/rho`
  — any new velocity-at-a-scale estimator keeps that factor 2; `gyroradius` uses `|q/m|`;
  `mu_of` is the snapshot-cadence counterpart of the sidecar `mu` column and honours each
  ensemble's `bperp` bit and its own `B0` (a `bperp=False` ensemble sees `B = B₀ẑ`, as its particles do —
  `tests/test_particles_diagnostics.py` pins it against `push_ensembles`' moment);
  `energy_budget`'s `closure = dke − total_work` is round-off by construction — nonzero
  means a piece is doing work outside the accumulators.
- Overhead (re-measured A3, 2026-08-18, 256² CPU fp64, quiet machine): `particle_fields`'s
  fixed transform cost is 17% of the solver step (19% with the optional pieces), inside the
  ≤15–20% budget; the O(N) push gather at the dense 2D loading used here pushes the observed
  total to +30%/+76% (1/3 ensembles × 32768) — unchanged from A2 within noise, the single
  gather per half-kick paying for the separately-gathered E_z pieces. docs/performance.md
  "Test particles overhead". `jnp.take`-style gather work and stage-1-grad reuse in
  `particle_fields` are still deferred, flagged in plans/TESTPART_PLAN.md §4.
  In 3D (B2, 2026-08-19, 128²×16) `particle_fields` is 17.4%/22.9% of the solver step under
  FD-z and 12.9%/14.4% under `z_spectral` (default mask / with the optional pieces): **FD-z
  with the non-ideal pieces is the one configuration outside the ≤15–20% budget**, and the
  1.86 ms ∂_z⁴ stencil + halo is the whole of the gap. `z_spectral` is cheaper only as a
  share — its own solver step is ~1.8× more expensive. The trilinear gather roughly doubles
  the O(N) push, as 8 corners against 4 predicts.
- Science: `examples/test-particles-2D.ipynb` + `examples/particles_2d_run.py` (resumable
  `make_data`, ~44 min/0.7 GB on the M1 laptop, fp64) is the 2D production reference: hyper=3
  base turbulence, production ensembles at **B0 = 10 with `epar_project=True`**, Q_⊥ from the
  local-B `vperpB2` with Xia's window rule (skip 10/Ω of E×B pickup, stop at 1.2×, Q_⊥ ≤ 2σ is
  an upper limit and stays out of the fit), designs (a) q/m sweep, (b) forcing sweep, (c) Xia's
  fixed-ρ-at-insertion v_⊥ sweep, plus B0 = 1 contrast twins and the controls. Result:
  c₂ = 0.40 ± 0.13, c₁ ≈ 1.0 (Chandran 0.34/0.75, Xia 0.2–0.44), with the sensitivity table
  printed — the number moves with the near-zero-point rule (0.32–0.59), not with the frame.
  The SAME analysis at B0 = 1 gives c₂ = 0.33 ± 0.22: amplitude sets the parallel/perpendicular
  split (E_z work 0.9 → 0.03–0.15 of the total), NOT c₂; the first pass's c₂ ≈ 0 was its
  protocol/turbulence (plan §5). Do not quote a c₂ without its rule and frame. Open items
  (particle-paired init, longer windows at ξ ≲ 0.1, wider clean spectral band, a pass-1
  configuration under the pass-2 protocol) are in plans/TESTPART_PLAN.md §5.
- 3D science: `examples/test-particles-3D.ipynb` + `examples/particles_3d_run.py`
  (`TARANIS_P3D_PROFILE` = smoke/mvp/full) is designed and smoke-tested, NOT run — the
  production campaign, its cost estimate and the Xia-comparison table are plan §11. In 3D
  β_i = v_th² is tied to ξ through the pinned B₀ (§11.2), so the reachable ξ band comes with
  its β band; quote both.
- Not yet: the 3D production run, and z-decomposed (multi-rank) particles — designs for the
  latter are in plans/TESTPART_PLAN.md §4, neither is built.

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
