# Checkpointing (`snapshot_io.py`)

Orbax-based save/restore. Read this before touching `snapshot_io.py`.

## Layouts

One `CheckpointManager` per MPI rank (`snapshot_manager_setup`), writing
`snap_path/<rank>/<step>/` — except under `comm_backend="jax"`, where all processes share
one manager over one flat directory (`snap_path/<step>/`) holding global z-sharded arrays
(orbax's native multihost path). `snapshot_layout()` distinguishes them: a
`_CHECKPOINT_METADATA` marker directly inside `snap_path/<n>` means `<n>` is a step, so
the dir is flat. Each backend can restart from either layout.

## Restore rules

- **Reads never construct a `CheckpointManager`.** Manager/checkpointer restores call
  `multihost.sync_global_processes` and deadlock as soon as ranks read different
  directories; use a bare, barrier-free `StandardCheckpointHandler`.
- The serialized tree is the full `SimulationState`. `forcing_scale` is always a concrete
  `(n_ou,)` array, never None — orbax records None children in its metadata, so a
  None-bearing save would fork the on-disk structure between configs. `save_snapshot`
  hard-errors on None (hand-built states). Any direct `StandardRestore` template must
  include `forcing_scale` as a `(n_ou,)` ShapeDtypeStruct (see
  `tests/test_restart_resharding.py`).
- `forcing_key`'s dtype comes from `jax.eval_shape(lambda: jax.random.key(0)).dtype`, not
  a guessed public constant.
- `t` is float64 at both field precisions (`TARANIS_PRECISION` only sets field dtype; see
  docs/numerics.md "Precision model"). `load_snapshot`/`old_snapshot_repair` restore `t`
  with its STORED dtype first (`snapshot_io._stored_t_dtype`, float32 for pre-precision-plan
  snapshots, float64 for current ones), then widen explicitly with
  `jnp.asarray(t, dtype=snapshot_io.T_DTYPE)` — the same repair pattern as
  `forcing_scale`. Any direct `StandardRestore` template must give `t` a
  `T_DTYPE`-typed `ShapeDtypeStruct`: orbax silently downcasts a stored float64 `t` into a
  float32 template slot rather than erroring, so a template built with the wrong dtype
  loses precision with no warning.
- Enumerate saved steps with `get_saved_steps(snap_path)`, never `mngr.all_steps()` — a
  top-level manager over a per-rank layout misreads rank subfolders as step numbers.
- Async saving (Phase 4 T10): don't remove the `wait_until_finished()` barrier without
  first decoupling orbax's buffer from the next donating stepper call —
  `donate_argnums=(0,)` invalidates it.

## Resharding (restore onto a different rank count)

`load_snapshot` unions overlapping z-slices per field across the saved ranks (`p_save` vs
`params.size`). `forcing_state`/`forcing_key` are excluded from the union (no z-axis,
identical across saved ranks by construction) — restored once, from rank 0's checkpoint.

## Snapshot index synchronization

Indices must mean the same simulation time on every rank: `load_snapshot` assumes it, and
orbax's `max_to_keep` prunes each rank's directory independently, so per-rank skew
compounds into different ranks holding different index windows. `simulate`/`simulate_scan`
broadcast rank 0's starting index (`max(all_steps())+1`) to all ranks — never derive it
per-rank (on restart-with-more-ranks, brand-new empty rank dirs would restart from 0 while
pre-existing ones resume from their old latest step). Regression test:
`tests/test_restart_resharding.py` (+ `slurms/test_restart_resharding.sh`), run `-n 2`
then `-n 4` on the same `snap_path`. Its invariant is NOT "identical index sets in every
rank dir" (pre-existing ranks may retain older snapshots the new ranks never wrote): it is
(a) identical latest index everywhere, (b) exact agreement from the newest "oldest index"
upward, (c) identical `t` per common index.

## Retention / old directories

`snapshot_manager_setup`'s `nsnap` is orbax's `max_to_keep` (honored since 2026-07-28;
older runs' directories may hold more snapshots than `nsnap`).

Pre-Phase-1 snapshots (no `forcing_scale`, or the older t/fields-only layout) fail to
restore with a structure mismatch; upgrade once, in place, with
`snapshot_io.old_snapshot_repair(snap_path, params)` — handles single- and per-rank
layouts, synthesizes zero `forcing_state` and a key from `forcing_seed` for the oldest
layout; `load_snapshot`'s error points at it. The repair is interruption-safe: each step
is staged fully, then swapped via same-filesystem renames; a rerun auto-recovers
`.repair_old_*`/`.repair_tmp_*` leftovers. Repaired directories are unreadable by
pre-Phase-1 code.

## Cadence caveats

`simulate`'s inner loop steps until `t >= target`, so large dt (e.g. early forced spin-up
from quiescence) can overshoot a `t_snap` target or `t_end` by a whole step — up to
`cfl_every` steps when stepping in blocks. Snapshots are "at least `t_snap` apart" and the
final time is ">= t_end", not exact — never assume exact snapshot counts or end times in
tests or postprocessing. The snapshot target advances every outer iteration regardless of
`save`, so `simulate(..., save=False)` terminates normally.
