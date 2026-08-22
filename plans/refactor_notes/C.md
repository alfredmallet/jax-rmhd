# Phase C — one run-loop body (`taranis/run.py`)

Branch `refactor/C`, base `3073df4` (Phase-0 close). Scope: `taranis/run.py` only; no other
file touched.

## What changed

`taranis/run.py`: **371 → 322 lines** (`git diff --stat`: 64 insertions, 113 deletions).

The run carry is now ALWAYS `(state, aux)` — `aux` is the `ParticleState` with particles on
and `None` off (a leafless pytree, so scan/while_loop/donation see the same leaves either
way) — and the per-step output is always `ys`: `(post-step t, per-ensemble moments)` on,
`None` off.

Deleted, folded into the three bodies below: `_carry_state`, `_carry_pstate`,
`_cfl_block_particles`, `_block_of_steps_particles`, and inside `simulate`
`stepper_wrapped`, `block_wrapped`, `stepper_particles`, `block_particles` and the second
`sim_to_next_snap` variant.

New/rewritten:

- `_step(carry, kgrid, params, rhs, set_timestep, scheme, stepper, dt_override=None,
  exp_ops=None) -> (carry, ys)` — one full timestep: stepper, `_advance_forcing`,
  `_advance_particles`. The only place the step is written.
- `_cfl_block(carry, ...) -> (carry, ys)` — `params.cfl_every` steps sharing one dt;
  `_block_dt` then `_hoisted_exp_ops` **outside** the step scan, unchanged.
- `_advance_block(carry, kgrid, params, nblock, scheme, stepper) -> (carry, ys)` — builds
  `rhs`/`set_timestep` once, then either the scan over cfl blocks (ys reshaped
  `(nblocks, cfl_every, ...) -> (nsteps, ...)` via `jax.tree.map`, a no-op on the
  particles-off `None`) or the plain step scan, with `_fixed_dt` + `_hoisted_exp_ops`
  computed outside it.
- `block_of_steps(x, ...)` is now a thin wrapper and the only remaining static
  `if params.particles is not None:` in the driver stack. Contract unchanged: particles off
  → state in, state out; on → `(state, pstate)` in, `((state, pstate), ys)` out.

`simulate`: one `loop_body` (`_cfl_block(...)[0]` or `_step(...)[0]`, chosen statically by
`_use_cfl_blocks`), one `sim_to_next_snap` with `cond` on `carry[0].t`, one host loop
(`state, pstate = carry`).

`simulate_scan`: one `advance` returning `(carry, ys)`; the moments-sidecar append moved out
of a closure redefinition and into the host loop (`if params.particles is not None and save`).
`_truncate_moments`/`_append_moments` semantics and call order are unchanged.

**The jax-backend boundary.** `comms.shard_call` builds `in_specs` from `state_specs()` and
cannot spec a `(state, None)` carry, and `comms.py` is Phase R's file. Particles are rejected
under `comm_backend="jax"` in `Parameters`, so on that backend the jitted boundary carries the
bare state: `simulate_scan` wraps `lambda s, kg: _advance_block((s, None), kg, ...)[0][0]` and
`simulate` wraps `lambda s, kg, target_t: sim_to_next_snap((s, None), kg, target_t)[0]`, each
re-tupling on the host side. `comms.py` untouched.

Hoisting placement is unchanged: `_hoisted_exp_ops` is called in `_cfl_block` (after
`_block_dt`) and in `_advance_block` (from `_fixed_dt`), both outside their step scans.

## Gates

All run from `/private/tmp/taranis-wt-C`. Nothing skipped, nothing loosened, no reference
regenerated. Four agents were testing concurrently, so no ms/step number is quoted anywhere
below.

### `make test`, both precision sessions

| session | result |
|---|---|
| `TARANIS_PRECISION=64 pytest tests` | **247 passed, 23 skipped, 1 deselected** |
| `TARANIS_PRECISION=32 pytest tests` | **224 passed, 46 skipped, 1 deselected** |

Every skip is a standing environment/precision gate, none new: fp64 — 16 × `needs >=4
devices, have 1`, 7 × `requires a TARANIS_PRECISION=32 session`; fp32 — 44 × `requires a
TARANIS_PRECISION=64 session`, 2 × `needs >=4 devices, have 1`.

### `tests/test_refactor_reference.py` — fields bitwise AND HLO histograms

Both precisions, all 12 configs, run with `-s` to confirm the host-mismatch print-skip did
NOT fire. Every config printed `fields bitwise identical to the reference`, `t bitwise
identical to the reference`, `<N> HLO instructions`, `<N> fusions`, `opcode histogram
unchanged`. fp64: `2 passed`. fp32: `2 passed`.

Instruction/fusion counts (identical to the Phase-0 record; sample):

| config | fp64 instr / fusions | fp32 instr / fusions |
|---|---|---|
| `put_cfl2_lsrk54_nohoist` | 3506 / 124 | 2841 / 124 |
| `put_adapt_lsrk33_unrolled` | 3574 / 117 | 2910 / 117 |
| `rmhd2d_adapt_lsrk33_mom` | 2627 / 92 | 1962 / 92 |
| `rmhd2d_fixed_imexcb3f` | 3585 / 111 | 2918 / 111 |
| `gdi2d_fixed_imexcb3e` | 552 / 28 | 552 / 28 |
| `gdi2d_fixed_lsrk33` | 1295 / 52 | 1296 / 52 |
| `gdi3d_fixed_imexcb3e` | 552 / 28 | 552 / 28 |

Not one opcode moved in any of the 12 configs at either precision — the `None` leaf adds no
HLO, as §5 expected. (This branch still carries the Phase-0 histogram regex, which does not
count `ROOT` lines; the comparison is self-consistent within the tree, and the merge step
re-runs it against the fixed regex on `main`.)

### The other named gates, per file

| file | fp64 | fp32 |
|---|---|---|
| `test_refactor_reference.py` | 2 passed | 2 passed |
| `test_particles_coupled.py` (gates 6, 6b, 6c, particles-on bitwise) | 20 passed | 10 passed, 10 skipped |
| `test_particles_3d.py` | 18 passed | 9 passed, 9 skipped |
| `test_forcing_spinup.py` | 3 passed | 3 skipped (fp64-only reference) |
| `test_precision_fp32.py` | 4 skipped (fp32-only) | 4 passed |
| `test_cfl_every.py` | 3 passed | 3 passed |
| `test_hoist_propagator.py` | 5 passed | 5 passed |

`test_hoist_propagator.py`'s five are `test_hoisted_matches_unhoisted_bitwise`,
`test_imex_independent_of_knob`, `test_stage_exp_ops_structure`,
**`test_unhoisted_graph_stays_memory_light`** and `test_knob_round_trips_through_save` — the
memory-light gate is in the passing set at both precisions, not skipped.

Gate 6 covers all five driver configs (`scan_fixed`, `while_fixed`, `scan_cflblock`,
`while_cflblock`, `scan_unforced`) — i.e. `simulate_scan` and `simulate`, fixed and
`cfl_every=2`, forced and unforced — bitwise against the pre-A2 npz with no regeneration.

### Probe — `bench/memory_probe.py --profile laptop`

`--precision 32 --out /tmp/probe_C_32.json` against `bench/memory_probe_refactor_base.json`
and `--precision 64 --out /tmp/probe_C_64.json` against
`bench/memory_probe_refactor_base_fp64.json`, comparing `total_u` per case:

| precision | cases | max abs delta total_u |
|---|---|---|
| fp32 | 28 | **0.0000 u** |
| fp64 | 28 | **0.0000 u** |

Every case matches the Phase-0 baseline exactly — not one case differs at all, so there is no
"worst case" to name. (ms/step columns were produced under concurrent agents and are not
quoted.)

### `ruff check .`

`All checks passed!`

## Noticed

- `jax.tree.map` is what makes the one `_advance_block` work for both branches: on the
  particles-off side `ys` is `None`, a pytree with no leaves, so the `(nblocks, cfl_every,
  ...) -> (nsteps, ...)` reshape is a no-op rather than a branch. That is the only place the
  unified body needed anything beyond deleting a duplicate.
- `_step`'s `dt_override=None, exp_ops=None` defaults reproduce the old `simulate` bodies,
  which called `stepper(state, kgrid, params, rhs, set_timestep, scheme)` with six positional
  args and let the stepper default the last two. Same values, same graph.
- Donation is unchanged in leaf terms: `jax.jit(..., donate_argnums=(0,))` now sees a
  `(state, None)` tuple where it used to see a bare state, and `None` contributes no buffer.
- `run._block_dt` and `run._refresh_forcing_scale` are read by tests outside C's ownership
  (`tests/test_particles_coupled.py:575`, `tests/test_z_stencils.py:149`); both keep their
  names and signatures.
- The session scratchpad is shared between the four phase worktrees. An early `make test` pair
  of mine wrote to the same `mt64.log`/`mt32.log` filenames another phase's agent was using;
  the numbers above come from a clean re-run into `/private/tmp/taranis-wt-C/.testlogs/`
  (untracked). Later parallel dispatches should namespace log paths by worktree.

## CLAUDE.md sentences that should change (for the sweep)

In the "Test particles" section:

- "`ParticleState` ... rides as a CARRY TUPLE `(state, pstate)` next to `SimulationState`" —
  still true, and the tuple is now the carry shape on BOTH branches: **the run carry is always
  `(state, aux)`, with `aux = pstate` when particles are on and `None` when off**. `None` is a
  leafless pytree, so the particles-off graph, donation and on-disk layout are untouched.
- "`block_of_steps`/`_cfl_block` return `((state, pstate), ys)` with `ys = (t, mom)` when on,
  the unchanged `final_state` when off; the off branch is a static `if params.particles is not
  None:` at the top of each function, never restructured "for symmetry"" — replace with:
  `_step`/`_cfl_block`/`_advance_block` always return `(carry, ys)` on the uniform carry
  (`ys = (t, mom)` on, `None` off); the ONE static `if params.particles is not None:` left in
  the driver stack is in the public `block_of_steps`, a thin wrapper that preserves the
  documented shape (off: state in → state out; on: `(state, pstate)` in → `((state, pstate),
  ys)` out). `simulate`'s while_loop body is `_step(...)[0]` or `_cfl_block(...)[0]` on the
  same carry, with `cond` on `carry[0].t`.
- Worth adding: particles are rejected on `comm_backend="jax"`, so the jitted `shard_call`
  boundary carries the bare state — `run.py` wraps `(s, None)` around it inside the jit and
  unwraps on the way out; `comms.shard_call` never sees the tuple carry.

In "Hoisted stage propagators": "Every `run.py` block function computes the ops OUTSIDE its
step scan (`_hoisted_exp_ops` after `_block_dt`, or from `_fixed_dt(params)`) — keep it there,
that placement is the whole point" — still exactly true; "every block function" is now the two
of them, `_cfl_block` and `_advance_block`.

In `params.cfl_every`: "`run._cfl_block` computes dt from the block's start state and passes
`dt_override` to the stepper" — unchanged (it reads `carry[0]` and passes `dt` through
`_step`).
