# Phase 0 — references, probe baseline, forced-restart fix

Base: `4fe4841`. Two commits: 0a+0b on the untouched solver tree, then 0c.

Recording host, stamped into every reference and checked by the host-skip rule:
`Alfreds-MacBook-Pro.local` / jax `0.10.0` / backend `cpu` / python `3.11.5`
(`macOS-14.8.3-arm64-arm-64bit`). `comm_backend` auto-resolves to `mpi4jax` at size 1
here (mpi4py and mpi4jax are both importable on this machine).

## 0a — the refactor reference

New files (the two data pairs force-added; `tests/data` is gitignored):

- `tests/_gen_refactor_reference.py` — DEFINES the twelve configs, the IC, the driver and
  the HLO parse.
- `tests/test_refactor_reference.py` — imports them, so the reference and the comparison
  cannot drift.
- `tests/data/refactor_reference_fp{64,32}.npz` — `fields` and `t` per config.
- `tests/data/refactor_reference_hlo_fp{64,32}.json` — per config, the instruction count
  per HLO opcode of the optimized module, plus `total_instructions` and `fusions`.

Driver: `jax.jit(run.block_of_steps, static_argnums=(2,3,4,5))`, 6 steps, 16²×8 in 3D and
32² in 2D, from `test_hoist_propagator`'s multi-mode IC (copied into the generator so the
recorded runs do not depend on another test module). Forcing is on for every RMHD config
(elsasser, `forcing_power_elsasser=(1,1)`; the 2D adaptive cell uses momentum) and off for
GDI, which rejects it. GDI eqpars are `bench/memory_probe.py`'s, at `dt=0.01`.

### Determinism check (before recording)

Two independent recordings per precision, separate processes, compared bitwise:

| precision | arrays compared | mismatches | HLO histograms | mismatches |
|---|---|---|---|---|
| fp64 | 24 (12 × fields, t) | none | 12 | none |
| fp32 | 24 | none | 12 | none |

### What was recorded

| config | t_final (fp64) | HLO instrs / fusions (fp64) | (fp32) |
|---|---|---|---|
| fd_fixed_lsrk54 | 0.06 | 2913 / 104 | 2247 / 104 |
| fd_cfl2_lsrk33 | 0.2973286 | 3096 / 115 | 2431 / 115 |
| fd_adapt_rk44 | 0.29090542 | 3854 / 127 | 3188 / 127 |
| sep_fixed_lsrk54 | 0.06 | 2949 / 105 | 2283 / 105 |
| put_cfl2_lsrk54 | 0.29579613 | 4372 / 140 | 3707 / 140 |
| put_cfl2_lsrk54_nohoist | 0.29579613 | 3506 / 124 | 2841 / 124 |
| put_adapt_lsrk33_unrolled | 0.28895684 | 3574 / 117 | 2910 / 117 |
| rmhd2d_adapt_lsrk33_mom | 0.16124561 | 2627 / 92 | 1962 / 92 |
| rmhd2d_fixed_imexcb3f | 0.06 | 3585 / 111 | 2918 / 111 |
| gdi2d_fixed_imexcb3e | 0.06 | 552 / 28 | 552 / 28 |
| gdi2d_fixed_lsrk33 | 0.06 | 1295 / 52 | 1296 / 52 |
| gdi3d_fixed_imexcb3e | 0.06 | 552 / 28 | 552 / 28 |

Every config's fields are finite; the hoisted and unhoisted putzer2 cells agree bitwise in
fields (as `test_hoist_propagator` claims) and differ by 866 instructions / 16 fusions in
the graph, which is the memory-light difference the knob exists for.

### Teeth

- Fields: a `(1 + 1e-13)` factor on `shared_physics.bracket` fails all 12 configs
  (`rel` 5e-15 … 2e-14).
- HLO: raising `shared_physics.GRAD_CHUNK` from its default 1 moves `fft`, `fusion`,
  `broadcast`, `parameter`, `slice`, `bitcast`, `get-tuple-element` counts in every config
  it was tried on, with a per-opcode diff printed.

## 0b — probe baseline

`bench/memory_probe_refactor_base.json` (fp32) and
`bench/memory_probe_refactor_base_fp64.json` (fp64), laptop profile, **28 cases each**,
run with nothing else on the machine. `total_u` is what the C and L gates compare
(≤ 0.05 u per case); the ms/step columns in these files are not to be quoted.

## 0c — the forced-restart fix

`run._refresh_forcing_scale` now returns the state untouched when the stored
`forcing_scale` is not all zeros; the dt = 0 computation runs only for a state that
carries none (a fresh `initialize`, a repaired legacy snapshot). The check is concrete and
outside jit (`bool(jnp.all(state.forcing_scale == 0))`); the `comm_backend="jax"` branch
for the computation itself is unchanged.

Gates:

- `tests/test_forcing_norm_per_step.py` gains three tests (the file already existed; the
  plan's new test goes in it):
  `test_restart_is_bitwise_at_default_norm`,
  `test_second_call_continues_the_returned_state_bitwise`,
  `test_refresh_computes_a_scale_only_when_the_state_carries_none`.
  All three fail on the unfixed tree (fields and `forcing_scale` differ; `forcing_state`
  and `forcing_key` do NOT — the O-U stream never saw the scale).
- `tests/test_particles_coupled.py` gate 6c now runs its whole body at BOTH
  `forcing_norm_per_step` settings (`_RESTART_KWARG_SETS`); every existing check is kept,
  labelled with the setting.

Verified unchanged with no regeneration: gate 6a
(`test_solver_output_matches_the_pre_a2_reference`), gate 6b,
`tests/test_forcing_spinup.py`, `tests/test_precision_fp32.py`, and the new refactor
reference — all start from `initialize`, whose zero scale is refreshed exactly as before.

## Gate results

| gate | fp64 session | fp32 session |
|---|---|---|
| `make test` at commit 1 (0a+0b, `b4de4d5`) | 244 passed, 23 skipped, 1 deselected, 787.9 s | 221 passed, 46 skipped, 1 deselected, 626.7 s |
| `make test` at commit 2 (0c) | 247 passed, 23 skipped, 1 deselected, 772.1 s | 224 passed, 46 skipped, 1 deselected, 653.5 s |

`ruff check` clean on every touched file at both commits (ruff 0.16.1, the version pinned
in `pyproject.toml`'s `lint` extra, which is what the `fast` workflow installs and runs as
`ruff check .`).

Nothing was skipped or loosened; the skip counts are the suite's own precision/marker
skips (`fp32`/`fp64`/`multidev`/`mpi`), unchanged from before Phase 0.

## Bitwise-adjacent observations

1. **`GRAD_CHUNK` is not bitwise across chunk sizes in 3D on this host.** CLAUDE.md says
   "All values are bitwise identical and `4` reproduces the pre-F1 graph byte-exactly".
   Measured here (jax 0.10.0 / CPU / fp64, 16²×8 FD-z, 6 steps): `GRAD_CHUNK` 2 and 4 both
   differ from 1 in 637 of 2304 elements, max 1.42e-14, rel 1.08e-16 — one ulp, the fusion
   class. The 2D config is exactly bitwise at every chunk size. Nothing depends on this
   (the default is 1 everywhere) but the CLAUDE.md sentence is stronger than the machine.
2. **`t` is not always the same fp64 number across schemes at fp32.** At
   `TARANIS_PRECISION=32` the IF schemes accumulate `t` to 0.059999998 over 6 fixed steps
   of 0.01 while `imexcb3f` lands on exactly 0.06, and `gdi2d_fixed_lsrk33` on 0.059999999.
   `t` is documented as always fp64; the difference is the dt the stepper adds being cast
   to the field precision on some paths. Recorded as found — the reference pins whatever
   each path does — but a later phase that touches `timestepping.py` should expect `t` to
   be a live comparator, not a formality.
3. Hoisted vs unhoisted putzer2 is bitwise in fields but a 25 % larger graph
   (4372 vs 3506 instructions at fp64), consistent with `test_unhoisted_graph_stays_memory_light`.

## Proposed CLAUDE.md wording (for the close-out sweep)

- Test particles section, the restart bullet — replace
  "Restart is bitwise ONLY with `forcing_norm_per_step=False` or `forcing=False` — this is
  pre-existing (`_refresh_forcing_scale` recomputes the forcing scale at dt=0 on
  `simulate`/`simulate_scan` entry, not particle-specific), but it also bounds when a
  particle restart reproduces the uninterrupted trajectory bitwise."
  with
  "Restart is bitwise at every `forcing_norm_per_step` setting: `_refresh_forcing_scale`
  computes a scale on `simulate`/`simulate_scan` entry only for a state whose
  `forcing_scale` is all zeros (a fresh `initialize`, a repaired legacy snapshot), so a
  checkpoint's stored lagged scale is the one the next step uses and a particle restart
  reproduces the uninterrupted trajectory bitwise (`tests/test_forcing_norm_per_step.py`,
  gate 6c at both settings)."
- Stochastic forcing section, the `forcing_norm_per_step` bullet — after "the scale lags
  one step" add: "and a stored nonzero scale survives `simulate`/`simulate_scan` entry
  unchanged, which is what makes a restart bitwise."
- Checkpointing invariants — no change needed.
- A line for the new reference, next to the gate-6 one: "`tests/data/refactor_reference_fp{64,32}.npz`
  plus `refactor_reference_hlo_fp{64,32}.json` (`tests/_gen_refactor_reference.py`,
  force-added) pin twelve solver paths — 3D FD-z, z_spectral separable and putzer2, GDI IF
  and IMEX, adaptive/cfl-block/fixed dt, rk44, imexcb3f, hoist on/off — bitwise in fields
  and t AND in the optimized-HLO opcode histogram."
