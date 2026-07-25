# Code-review fixes — 2026-07-24

Follow-up to the code review of the uncommitted `performance`-branch changes. All six
review findings are fixed, plus one pre-existing infinite-loop bug in `run.py::simulate`
discovered while verifying the fixes. Every fix was verified by running the relevant code
path (see [Verification](#verification)).

## 1. `Parameters.save` could deadlock or desync MPI ranks

**Was** (`jax_rmhd/config.py`): every rank ran the `os.path.exists` check and the
diff comparison itself, but only some code paths reached the `comm.Barrier()`. On a lazy
shared filesystem (Savio Lustre/NFS), ranks can briefly disagree on whether the file
exists; whichever subset returned early or raised left the other ranks stuck in the
barrier forever.

**Now**: rank 0 alone checks, compares, and writes; the outcome (an error string or
`None`) is `bcast` to all ranks, which then raise or return **together**. The broadcast
also replaces the barrier — it inherently orders every rank after the write.
`save` is therefore collective under MPI and must be called by all ranks (as before,
but now documented).

## 2. `Parameters.save` crashed on numpy/jax scalar constructor args

**Was**: `json.dump` raised a bare `TypeError` on `np.float64` / 0-d `jnp` array values
(easy to hit via a `dt` or `Lx` pulled out of an array).

**Now**: a `default=_json_scalar` hook unwraps any 0-d `.item()`-bearing scalar to its
python value; genuinely unserializable values raise a clear
`TypeError: Parameter value ... can't be recorded in params.json`. The record is
round-tripped through JSON *before* the equality comparison, so a numpy-typed re-save of
identical values correctly compares equal to the file on disk (tuples→lists and
scalar unwrapping are normalized on both sides).

## 3. `save_snapshot` could silently fork the on-disk checkpoint structure

**Was** (`jax_rmhd/snapshot_io.py`): a hand-built `SimulationState` (default
`forcing_scale=None`) in a **non-forced** run sailed past every existing check
(`ForcingTerm`'s trace-time guard only fires when forcing + `forcing_norm_per_step` are
on) and orbax recorded the `None` child — forking the on-disk tree structure between
runs, exactly the failure the always-concrete-`forcing_scale` design exists to prevent.

**Now**: `save_snapshot` raises a `ValueError` naming the fix
(`state._replace(forcing_scale=jnp.zeros((params.n_ou,)))` or build states via
`run.initialize` / `load_snapshot`) before anything touches disk.

## 4. `old_snapshot_repair` had a data-stranding crash window

**Was**: the swap sequence was `mngr.delete(step)` (a full recursive delete) followed by
`shutil.move` of the staged copy. An interruption between the two left the step absent
from the main tree, present only in a `.repair_*` temp dir — and a rerun couldn't recover
it, because it iterates `mngr.all_steps()`, which no longer listed the step. The code
comment claimed "an interruption can never lose data (worst case: temp dir left behind)",
which overstated the guarantee.

**Now**: the staged copy is written fully first (unchanged), then the swap is two
same-filesystem **renames**: original → `.repair_old_<step>`, staged → `<step>`, then
both leftovers are removed. A new `_recover_interrupted_repair` pass runs at the start
of every repair (per rank dir) and self-heals any interruption point: a
`.repair_old_<step>` whose step is missing is renamed back into place (and re-repaired);
one whose step exists means the swap completed and is just deleted; stale
`.repair_tmp_*` staging dirs are always safe to discard. Interrupted repairs now recover
automatically on rerun with no manual step.

## 5. Pre-forcing-era snapshots hit a dead end between two error messages

**Was**: a snapshot old enough to predate forcing entirely (tree = `t`/`fields` only)
failed `load_snapshot` with a pointer at `old_snapshot_repair` — but repair's only legacy
template also mismatched it, printed "not a legacy snapshot, skipping", and the user
bounced between the two messages with no way forward. The same bare `except ValueError`
also mislabeled genuinely corrupt steps as "not legacy".

**Now**: repair tries three templates in order — pre-`forcing_scale` (adds zero scale),
already-current (prints "already current, skipping"), and the new pre-forcing
`_AncientCkptState` (`t`/`fields` only), for which it synthesizes zero `forcing_state`,
a `forcing_key` from `params.forcing_seed`, and zero `forcing_scale`. Steps matching
none of the three get an honest message ("structure matches neither the current nor a
known legacy layout (corrupt, or shapes don't match params?)"). Additionally,
`_restore_or_advise` now matches any of `forcing_scale`/`forcing_state`/`forcing_key`
in orbax's error text — orbax names only the *first* mismatched key, so pre-forcing
snapshots (which trip on `forcing_state`) previously surfaced the raw cryptic error
instead of the repair pointer.

## 6. Two comment corrections

- `old_snapshot_repair`'s layout-detection comment claimed it detects "exactly like
  `load_snapshot` does"; it actually uses `get_saved_steps`' heuristic (`isdir("0")`
  without a `"default"` child), while `load_snapshot` probes for its specific step under
  `snap_path/0`. The comment now says so (both agree on the two real layouts).
- `_restore_or_advise` now carries an explicit note that its advice depends on orbax's
  error wording: if an orbax upgrade rewords the message, the raw error still propagates
  unchanged — only the friendlier pointer is lost.

## Bonus: pre-existing infinite loop in `simulate(..., save=False)` (found during verification)

Not a review finding — the verification script for fixes 3–5 hung reproducibly, and
bisecting the trigger (`save=True` completed; both `save=False` variants spun forever at
`run.py`'s outer while-loop, confirmed via `faulthandler` stack dumps) exposed a
pre-existing bug unrelated to the fixes above:

**Was** (`jax_rmhd/run.py::simulate`): `t_last_snapshot` was only updated inside
`if save:`. With `save=False` and `t_snap < t_end`, `t_next_snapshot =
min(t_last_snapshot + t_snap, t_end)` stayed frozen at the first target; once `state.t`
passed it, the jitted inner `lax.while_loop` returned immediately (condition false on
entry) and the outer python `while state.t < t_end` spun forever at full CPU. It went
unnoticed because every existing caller either saves (`save=True`) or runs a single
segment (`t_snap > t_end`, as in `tests/test_forcing_norm_per_step.py`).

**Now**: `t_last_snapshot = float(state.t)` runs unconditionally at the end of each
outer iteration (identical behavior when `save=True`). `simulate_scan` was checked and
does not share the bug — its loop advances by `nblock` steps regardless of `save`.

## Documentation

`CLAUDE.md` updated to match: `params.save` collectivity and scalar normalization, the
`save_snapshot` None guard, repair's pre-forcing support and interruption self-healing,
and the `simulate(save=False)` loop fix.

## Verification

All run on this machine at `RMHD_PRECISION=64`:

| Check | Result |
|---|---|
| Targeted fix suite (13 checks: numpy-scalar save/round-trip/re-save, unserializable `TypeError`, `save_snapshot` None guard, ancient load pointer + repair + rerun, legacy repair, interrupted-swap recovery A/B) | 13/13 PASS |
| `mpirun -n 2` `Parameters.save`: collective write, identical re-save, differing save raises on **both** ranks without deadlock | PASS |
| Pre-fix regression suite (params/snapshot round trips, legacy repair via `load_snapshot` pointer, repair idempotency) | 11/11 PASS |
| `tests/test_forcing_norm_per_step.py` | 9/9 ALL PASS |
| `tests/test_forcing_smoke.py` | ALL PASS |
| `tests/test_restart_resharding.py` (`mpirun -n 2` then `-n 4`, fresh `snap_path`) | ALL PASS |
| `simulate` bisect repro (plain / save=True / pre-saved mngr, `t_snap < t_end`) after loop fix | all 3 complete at t≈0.11 |
