# Checkpoint round-trip / invariant tests for jax_rmhd.snapshot_io.
# Everything here is self-contained: the legacy checkpoint
# fixtures are SYNTHESIZED in a tmp dir, so there is no dependency on the untracked
# tests/data tree.
#
# Invariants under test (docs/checkpointing.md + CLAUDE.md):
#   - the serialized tree is the full SimulationState, forcing_key/forcing_scale included,
#     and it round-trips bitwise;
#   - reads NEVER construct a CheckpointManager (manager restores barrier -> deadlock);
#   - forcing_scale is always a concrete (n_ou,) array, never None;
#   - nsnap == orbax max_to_keep, enumerated with get_saved_steps (never mngr.all_steps());
#   - old_snapshot_repair upgrades both legacy layouts in place and is idempotent.
#
# pytest: `pytest tests/test_snapshot_roundtrip.py`.  Script: `python tests/...py`.
from _rmhd_testing import (bootstrap, checks, ctx, fake_ranked_params, make_state,
                           managed_manager, mpi_size, multimode_ic, snap_dir)

bootstrap()

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from etils import epath

import jax_rmhd as jr
import jax_rmhd.snapshot_io as sn

# orbax's item subdirectory inside a step dir, and the handler name its step-level
# _CHECKPOINT_METADATA records (mirrors what a real CheckpointManager writes).
_ITEM = "default"
_HANDLER = ("orbax.checkpoint._src.handlers.standard_checkpoint_handler"
            ".StandardCheckpointHandler")


def _single_process_only(what):
    """Soft skip (print + return True): these tests own their own tmp dirs and spoof
    rank/size, so they are meaningful in a single process only. Every rank takes the
    same branch, so nothing can deadlock under mpirun."""
    if mpi_size() > 1:
        print(f"[SKIP] {what} -- single-process only (rank-local tmp dirs / spoofed ranks)")
        return True
    return False


def _kd(k):
    return np.asarray(jax.random.key_data(k))


def _is_key(k):
    return jax.dtypes.issubdtype(k.dtype, jax.dtypes.prng_key)


def _rand_c(shape, rng, ctype):
    return jnp.asarray((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)),
                       dtype=ctype)


def _decorated_state(params, seed=0):
    """A FRESH state with non-trivial values in every leaf (a zero forcing_state would
    make the round-trip check vacuous for three of the five leaves)."""
    ftype, ctype = sn.get_precision_types()
    rng = np.random.default_rng(seed)
    nkx, nky = params.nx, params.ny // 2 + 1
    st = make_state(params, ic=multimode_ic)
    return st._replace(
        t=jnp.asarray(0.375, dtype=ftype),
        forcing_state=_rand_c((params.n_ou, 2, nkx, nky), rng, ctype),
        forcing_key=jax.random.split(jax.random.key(7))[1],
        forcing_scale=jnp.asarray(rng.standard_normal((params.n_ou,)), dtype=ftype))


def _write_raw_step(snap_path, step, tree, rank=None):
    """Write `tree` into snap_path[/rank]/<step>/default with a BARE
    StandardCheckpointHandler -- the only writer that can produce a checkpoint whose
    pytree is NOT a SimulationState, which is what the legacy fixtures need.

    handler.save() alone is not enough: it leaves the per-process ocdbt files unmerged
    and every leaf then fails to open ("Metadata at fields/.zarray ... does not exist").
    finalize() is the step that makes the directory readable.
    """
    base = snap_path if rank is None else os.path.join(snap_path, str(rank))
    step_dir = os.path.join(base, str(step))
    item_dir = epath.Path(os.path.join(step_dir, _ITEM))
    handler = ocp.StandardCheckpointHandler()
    try:
        handler.save(item_dir, args=ocp.args.StandardSave(tree))
        handler.finalize(item_dir)
    finally:
        getattr(handler, "close", lambda: None)()
    # the step-level marker a real CheckpointManager leaves behind; _is_step_dir()
    # accepts either this or the item subdir, but write it so the fixture is faithful.
    with open(os.path.join(step_dir, "_CHECKPOINT_METADATA"), "w") as f:
        json.dump({"item_handlers": {_ITEM: _HANDLER}, "metrics": {},
                   "performance_metrics": {}, "custom_metadata": {}}, f)


def _legacy_trees(params, nz_save, seed=3):
    """(_LegacyCkptState, _AncientCkptState, reference dict) sharing one fields array.

    _LegacyCkptState  = pre-forcing_scale layout (t/fields/forcing_state/forcing_key)
    _AncientCkptState = pre-forcing layout (t/fields only)
    Both shapes are read straight off snapshot_io so they cannot drift from the library.
    """
    ftype, ctype = sn.get_precision_types()
    nkx, nky = params.nx, params.ny // 2 + 1
    rng = np.random.default_rng(seed)
    fields = _rand_c((params.nfields, nz_save, nkx, nky), rng, ctype)
    forcing_state = _rand_c((params.n_ou, 2, nkx, nky), rng, ctype)
    key = jax.random.split(jax.random.key(21))[0]
    legacy = sn._LegacyCkptState(t=jnp.asarray(1.25, dtype=ftype), fields=fields,
                                 forcing_state=forcing_state, forcing_key=key)
    ancient = sn._AncientCkptState(t=jnp.asarray(2.5, dtype=ftype), fields=fields)
    ref = dict(fields=np.asarray(fields), forcing_state=np.asarray(forcing_state),
               key_data=_kd(key))
    return legacy, ancient, ref


def _repair_leftovers(path):
    return sorted(n for n in os.listdir(path) if n.startswith(".repair_"))


# --------------------------------------------------------------------- round-trip


def test_save_load_bitwise_roundtrip():
    if _single_process_only("test_save_load_bitwise_roundtrip"):
        return
    params, _ = ctx()
    state = _decorated_state(params)
    want = dict(t=float(state.t), fields=np.asarray(state.fields),
                forcing_state=np.asarray(state.forcing_state),
                forcing_scale=np.asarray(state.forcing_scale),
                key_data=_kd(state.forcing_key))
    with snap_dir() as d:
        with managed_manager(params, d, nsnap=2) as m:
            sn.save_snapshot(0, state, m, params)
            m.wait_until_finished()
        back = sn.load_snapshot(0, d, params)
        with checks() as c:
            c.check(f"single-process save produced the flat layout "
                    f"(got {sn.snapshot_layout(d)!r})", sn.snapshot_layout(d) == "flat")
            c.check("get_saved_steps sees the saved step", sorted(sn.get_saved_steps(d)) == [0])
            c.check("t round-trips exactly", float(back.t) == want["t"])
            c.check("fields round-trip bitwise",
                    np.array_equal(np.asarray(back.fields), want["fields"])
                    and back.fields.shape == want["fields"].shape
                    and back.fields.dtype == state.fields.dtype)
            c.check("forcing_state round-trips bitwise",
                    np.array_equal(np.asarray(back.forcing_state), want["forcing_state"]))
            c.check("forcing_scale round-trips bitwise as a concrete (n_ou,) array",
                    back.forcing_scale is not None
                    and back.forcing_scale.shape == (params.n_ou,)
                    and np.array_equal(np.asarray(back.forcing_scale), want["forcing_scale"]))
            c.check("forcing_key comes back as a typed PRNG key with identical key data",
                    _is_key(back.forcing_key)
                    and np.array_equal(_kd(back.forcing_key), want["key_data"]))


def test_reads_never_construct_a_manager():
    # docs/checkpointing.md: a CheckpointManager (and Checkpointer.restore) calls
    # multihost.sync_global_processes and DEADLOCKS the moment ranks read different
    # directories. Booby-trap the constructor so any read path that reaches for one
    # fails loudly here instead of hanging on a cluster.
    if _single_process_only("test_reads_never_construct_a_manager"):
        return
    params, _ = ctx()
    state = _decorated_state(params, seed=1)
    want_fields = np.asarray(state.fields)
    with snap_dir() as d:
        with managed_manager(params, d, nsnap=2) as m:
            sn.save_snapshot(0, state, m, params)
            m.wait_until_finished()

        original_init = ocp.CheckpointManager.__init__

        def _boom(*args, **kwargs):
            raise AssertionError("read path constructed an orbax CheckpointManager "
                                 "(barrier -> multi-rank deadlock); use a bare "
                                 "StandardCheckpointHandler")

        try:
            ocp.CheckpointManager.__init__ = _boom
            steps = sorted(sn.get_saved_steps(d))
            layout = sn.snapshot_layout(d)
            back = sn.load_snapshot(0, d, params)
        finally:
            ocp.CheckpointManager.__init__ = original_init

        with checks() as c:
            c.check("get_saved_steps needs no manager", steps == [0])
            c.check("snapshot_layout needs no manager", layout == "flat")
            c.check("load_snapshot needs no manager and returns the saved fields",
                    np.array_equal(np.asarray(back.fields), want_fields))


def test_save_rejects_none_forcing_scale():
    if _single_process_only("test_save_rejects_none_forcing_scale"):
        return
    params, _ = ctx()
    with snap_dir() as d:
        with managed_manager(params, d, nsnap=2) as m:
            bad = make_state(params)._replace(forcing_scale=None)
            try:
                sn.save_snapshot(0, bad, m, params)
                raised = ""
            except ValueError as e:
                raised = str(e)
            # a good state still saves through the same manager
            sn.save_snapshot(0, make_state(params), m, params)
            m.wait_until_finished()
        with checks() as c:
            c.check("save_snapshot rejects forcing_scale=None with a ValueError",
                    "forcing_scale" in raised)
            c.check("... and the error names the fix (concrete (n_ou,) array)",
                    "n_ou" in raised)
            c.check("a concrete forcing_scale still saves", sorted(sn.get_saved_steps(d)) == [0])


def test_pruning_window_keeps_last_nsnap():
    # nsnap is orbax's max_to_keep. Enumerate with get_saved_steps, NEVER
    # mngr.all_steps() (a top-level manager over a per-rank tree reads rank dirs as steps).
    if _single_process_only("test_pruning_window_keeps_last_nsnap"):
        return
    params, _ = ctx()
    with snap_dir() as d:
        with managed_manager(params, d, nsnap=2) as m:
            state = _decorated_state(params, seed=2)
            for i in range(5):
                sn.save_snapshot(i, state, m, params)
                m.wait_until_finished()
            kept = sorted(sn.get_saved_steps(d))
        newest = sn.load_snapshot(4, d, params)
        with checks() as c:
            c.check(f"nsnap=2 keeps only the last two steps (kept {kept})", kept == [3, 4])
            c.check("the surviving newest step is still readable",
                    np.array_equal(np.asarray(newest.fields), np.asarray(state.fields)))
            try:
                sn.load_snapshot(0, d, params)
                pruned_gone = False
            except FileNotFoundError:
                pruned_gone = True
            c.check("a pruned step raises FileNotFoundError", pruned_gone)


# ------------------------------------------------- synthesized legacy checkpoints


def test_legacy_flat_snapshot_repair_and_idempotency():
    # SYNTHESIZED pre-forcing_scale checkpoint (t/fields/forcing_state/forcing_key),
    # written with a bare handler because its pytree is not a SimulationState.
    if _single_process_only("test_legacy_flat_snapshot_repair_and_idempotency"):
        return
    params, _ = ctx()
    legacy, _, ref = _legacy_trees(params, params.nz)
    with snap_dir() as d:
        _write_raw_step(d, 4, legacy)
        with checks() as c:
            c.check(f"synthesized legacy tree reads as the flat layout, step 4 "
                    f"(got {sn.snapshot_layout(d)!r}, {sorted(sn.get_saved_steps(d))})",
                    sn.snapshot_layout(d) == "flat" and sorted(sn.get_saved_steps(d)) == [4])
            try:
                sn.load_snapshot(4, d, params)
                advised = False
            except ValueError as e:
                advised = "old_snapshot_repair" in str(e)
            c.check("loading it un-repaired points the user at old_snapshot_repair", advised)

            sn.old_snapshot_repair(d, params)
            st = sn.load_snapshot(4, d, params)
            c.check("repaired snapshot loads with t/fields/forcing_state preserved bitwise",
                    float(st.t) == 1.25
                    and np.array_equal(np.asarray(st.fields), ref["fields"])
                    and np.array_equal(np.asarray(st.forcing_state), ref["forcing_state"]))
            c.check("forcing_key survives the repair as a typed key",
                    _is_key(st.forcing_key)
                    and np.array_equal(_kd(st.forcing_key), ref["key_data"]))
            c.check("forcing_scale is synthesized as concrete (n_ou,) zeros",
                    st.forcing_scale is not None
                    and st.forcing_scale.shape == (params.n_ou,)
                    and not np.any(np.asarray(st.forcing_scale)))
            c.check("repair leaves no .repair_* staging directories", _repair_leftovers(d) == [])

            # idempotent: the second pass recognizes the current layout and skips
            sn.old_snapshot_repair(d, params)
            again = sn.load_snapshot(4, d, params)
            c.check("re-running old_snapshot_repair is a no-op (bitwise identical reload)",
                    float(again.t) == float(st.t)
                    and np.array_equal(np.asarray(again.fields), np.asarray(st.fields))
                    and np.array_equal(np.asarray(again.forcing_state),
                                       np.asarray(st.forcing_state))
                    and np.array_equal(_kd(again.forcing_key), ref["key_data"])
                    and np.array_equal(np.asarray(again.forcing_scale),
                                       np.asarray(st.forcing_scale)))
            c.check("... and still leaves no staging directories", _repair_leftovers(d) == [])


def test_ancient_snapshot_repair_synthesizes_forcing():
    # SYNTHESIZED pre-forcing checkpoint (t/fields only): forcing_state (zeros),
    # forcing_key (from params.forcing_seed) and forcing_scale (zeros) are all invented
    # by the repair.
    if _single_process_only("test_ancient_snapshot_repair_synthesizes_forcing"):
        return
    seed = 13
    params = fake_ranked_params(0, 1, forcing_seed=seed)
    _, ancient, ref = _legacy_trees(params, params.nz)
    with snap_dir() as d:
        _write_raw_step(d, 2, ancient)
        with checks() as c:
            try:
                sn.load_snapshot(2, d, params)
                advised = False
            except ValueError as e:
                advised = "old_snapshot_repair" in str(e)
            c.check("un-repaired ancient snapshot points at old_snapshot_repair", advised)

            sn.old_snapshot_repair(d, params)
            st = sn.load_snapshot(2, d, params)
            c.check("repaired ancient snapshot preserves t and fields bitwise",
                    float(st.t) == 2.5
                    and np.array_equal(np.asarray(st.fields), ref["fields"]))
            c.check("forcing_state synthesized as zeros with the current shape",
                    st.forcing_state.shape == (params.n_ou, 2, params.nx, params.ny // 2 + 1)
                    and not np.any(np.asarray(st.forcing_state)))
            c.check(f"forcing_key synthesized from params.forcing_seed={seed}",
                    _is_key(st.forcing_key)
                    and np.array_equal(_kd(st.forcing_key), _kd(jax.random.key(seed))))
            c.check("forcing_scale synthesized as concrete (n_ou,) zeros",
                    st.forcing_scale is not None
                    and st.forcing_scale.shape == (params.n_ou,)
                    and not np.any(np.asarray(st.forcing_scale)))

            sn.old_snapshot_repair(d, params)
            again = sn.load_snapshot(2, d, params)
            c.check("re-running old_snapshot_repair is a no-op",
                    float(again.t) == float(st.t)
                    and np.array_equal(np.asarray(again.fields), np.asarray(st.fields))
                    and np.array_equal(_kd(again.forcing_key), _kd(st.forcing_key)))
            c.check("no staging directories left behind", _repair_leftovers(d) == [])


def test_legacy_per_rank_tree_repair_and_reshard_read():
    # The mpi4jax production layout: snap_path/<rank>/<step>/. Repairing it must fix
    # every rank dir, and the repaired tree must still union correctly across ranks.
    # rank/size are spoofed (never handed to jitted physics) -- single process only.
    if _single_process_only("test_legacy_per_rank_tree_repair_and_reshard_read"):
        return
    p0 = fake_ranked_params(0, 2)
    p1 = fake_ranked_params(1, 2)
    nz_local = p0.nz // 2
    legacy_full, _, ref = _legacy_trees(p0, p0.nz)
    with snap_dir() as d:
        for r, p_w in enumerate((p0, p1)):
            _write_raw_step(d, 3, legacy_full._replace(
                fields=legacy_full.fields[:, r * nz_local:(r + 1) * nz_local]), rank=r)
        with checks() as c:
            c.check(f"synthesized 2-rank tree reads as the per-rank layout "
                    f"(got {sn.snapshot_layout(d)!r})",
                    sn.snapshot_layout(d) == "per_rank"
                    and sorted(os.listdir(d)) == ["0", "1"])
            c.check("get_saved_steps enumerates per-rank steps from rank 0's dir",
                    sorted(sn.get_saved_steps(d)) == [3])
            try:
                sn.load_snapshot(3, d, p0)
                advised = False
            except ValueError as e:
                advised = "old_snapshot_repair" in str(e)
            c.check("un-repaired per-rank tree points at old_snapshot_repair", advised)

            sn.old_snapshot_repair(d, p0)
            s0 = sn.load_snapshot(3, d, p0)
            s1 = sn.load_snapshot(3, d, p1)
            c.check("rank 0 reads its own z-half bitwise",
                    s0.fields.shape == (p0.nfields, nz_local, p0.nx, p0.ny // 2 + 1)
                    and np.array_equal(np.asarray(s0.fields), ref["fields"][:, :nz_local]))
            c.check("rank 1 reads its own z-half bitwise",
                    np.array_equal(np.asarray(s1.fields), ref["fields"][:, nz_local:]))
            c.check("forcing leaves come from rank 0 for both ranks",
                    np.array_equal(np.asarray(s0.forcing_state), ref["forcing_state"])
                    and np.array_equal(np.asarray(s1.forcing_state), ref["forcing_state"])
                    and np.array_equal(_kd(s1.forcing_key), ref["key_data"]))
            c.check("no staging directories left in either rank dir",
                    all(_repair_leftovers(os.path.join(d, r)) == [] for r in ("0", "1")))

            sn.old_snapshot_repair(d, p0)
            again = sn.load_snapshot(3, d, p1)
            c.check("re-running old_snapshot_repair on a per-rank tree is a no-op",
                    np.array_equal(np.asarray(again.fields), np.asarray(s1.fields))
                    and float(again.t) == float(s1.t))


def test_find_items_and_load_slice():
    # low-value smoke: find_items lists the stored keys without raising,
    # and load_slice reads a z-slice of `fields` bitwise via tensorstore (the
    # laptop-diagnostics path -- flat single-process layout only).
    if _single_process_only("test_find_items_and_load_slice"):
        return
    params, _ = ctx()
    state = _decorated_state(params, seed=5)
    with snap_dir() as d:
        with managed_manager(params, d, nsnap=2) as m:
            sn.save_snapshot(0, state, m, params)
            m.wait_until_finished()
        sn.find_items(0, d)  # prints the key listing; smoke: must not raise
        iz, nzslice = 2, 3
        sl = np.asarray(sn.load_slice(0, iz, nzslice, d, item="fields"))
        want = np.asarray(state.fields)[:, iz:iz + nzslice]
        with checks() as c:
            c.check(f"load_slice returns the ({params.nfields},{nzslice},...) z-slice",
                    sl.shape == want.shape)
            c.check("load_slice slice matches load_snapshot's fields bitwise",
                    np.array_equal(sl, want))


def test_repaired_snapshot_restarts_a_run():
    # end-to-end: a repaired legacy directory is a valid restart point (this is what
    # the repair exists for), and the run's own writer can continue in a fresh dir.
    if _single_process_only("test_repaired_snapshot_restarts_a_run"):
        return
    params, kgrid = ctx()
    legacy, _, _ = _legacy_trees(params, params.nz, seed=9)
    with snap_dir() as d:
        _write_raw_step(d, 6, legacy)
        sn.old_snapshot_repair(d, params)
        restored = sn.load_snapshot(6, d, params)
        t0 = float(restored.t)
        with snap_dir() as out, managed_manager(params, out, nsnap=2) as m:
            end = jr.simulate(restored, kgrid, params, t_snap=10.0,
                              t_end=t0 + 0.02, mngr=m, save=False)
            t1 = float(end.t)
            finite = bool(np.all(np.isfinite(np.asarray(end.fields))))
    with checks() as c:
        c.check(f"simulate advanced past t_end from the repaired snapshot "
                f"({t0:.4f} -> {t1:.4f})", t1 >= t0 + 0.02)
        c.check("the continued state is finite", finite)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
