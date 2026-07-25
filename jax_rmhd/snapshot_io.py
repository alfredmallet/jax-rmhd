import jax
import jax.numpy as jnp
import tensorstore as ts
import orbax.checkpoint as ocp
import os
import shutil
import tempfile
from typing import NamedTuple, Any
from .types import SimulationState

class _LegacyCkptState(NamedTuple):
    # Pre-forcing_scale checkpoint tree, used only by old_snapshot_repair to read
    # snapshots written before SimulationState gained forcing_scale.
    t: Any
    fields: Any
    forcing_state: Any
    forcing_key: Any

class _AncientCkptState(NamedTuple):
    # Pre-forcing checkpoint tree (t/fields only), the oldest layout old_snapshot_repair
    # understands; forcing_state/forcing_key/forcing_scale are synthesized on repair.
    t: Any
    fields: Any

def get_precision_types():
    if jax.config.read("jax_enable_x64"):
        return jnp.float64, jnp.complex128
    else:
        return jnp.float32, jnp.complex64

def get_key_dtype():
    return jax.eval_shape(lambda: jax.random.key(0)).dtype
    
# Setting up Orbax stuff
def snapshot_manager_setup(params,snap_path="data",nsnap=1000):
    if params.size>1:
        checkpoint_path = os.path.abspath(snap_path+f'/{params.rank}')
    else:
        checkpoint_path = os.path.abspath(snap_path)
    options = ocp.CheckpointManagerOptions(max_to_keep=nsnap)
    return ocp.CheckpointManager(directory=checkpoint_path,options=options)

def save_snapshot(isnap,state,mngr):
    # saves the full SimulationState; forcing_scale is always a concrete (n_ou,) array
    # (see run.initialize), so all new checkpoints share one tree structure. Snapshots
    # written before forcing_scale existed must be upgraded once with old_snapshot_repair.
    if state.forcing_scale is None:
        # orbax records None children in its metadata, so letting this through would fork
        # the on-disk tree structure between runs (hand-built states hit this; the
        # ForcingTerm check only catches it when forcing_norm_per_step is on)
        raise ValueError("state.forcing_scale is None; checkpoints must always carry a "
                         "concrete (n_ou,) array so all snapshots share one tree structure. "
                         "Build states via run.initialize / load_snapshot, or use "
                         "state._replace(forcing_scale=jnp.zeros((params.n_ou,))).")
    return mngr.save(isnap,args=ocp.args.StandardSave(state))

def get_saved_steps(snap_path):
    snap_path = str(snap_path)
    rank0_dir = os.path.join(snap_path, "0")
    options = ocp.CheckpointManagerOptions()
    if os.path.isdir(rank0_dir) and not os.path.exists(os.path.join(rank0_dir, "default")):
        mngr_rank0 = ocp.CheckpointManager(os.path.abspath(rank0_dir), options=options)
        return mngr_rank0.all_steps()
    mngr = ocp.CheckpointManager(os.path.abspath(snap_path), options=options)
    return mngr.all_steps()

def load_snapshot(isnap,snap_path,params):
    snap_path = str(snap_path)

    p_save = 1
    if os.path.exists(os.path.join(snap_path, "0", str(isnap))):
        while os.path.exists(os.path.join(snap_path, str(p_save), str(isnap))):
            p_save += 1

    ftype, ctype = get_precision_types()
    
    #grid sizes for the current load and the original save
    nz_load = params.nz // params.size if params.spatial_dimensions == 3 else 1
    nz_save = params.nz // p_save if params.spatial_dimensions == 3 else 1
    assert params.nz % params.size == 0, f"Current z-domain {params.nz} not cleanly divisible by params.size {params.size}"
    assert params.nz % p_save == 0, f"Saved z-domain {params.nz} not cleanly divisible by p_save {p_save}"

    #initialize the restored field array for the current rank (r_l)
    restored_fields = jnp.zeros((params.nfields, nz_load, params.nx, params.ny // 2 + 1), dtype=ctype)
    restored_t = 0.0

    #index range for current rank
    z_start_l = params.rank * nz_load
    z_end_l = (params.rank + 1) * nz_load

    # Setup the expected ShapeDtypeStruct for loading from the saved ranks.
    if params.spatial_dimensions == 3:
        shape_complex_s = (params.nfields, nz_save, params.nx, params.ny // 2 + 1)
    else:
        shape_complex_s = (params.nfields, 1, params.nx, params.ny // 2 + 1)

    nkx, nky = params.nx, params.ny // 2 + 1
    # forcing_state/forcing_key have no z-axis and are identical on every saved rank
    forcing_state_like_s = jax.ShapeDtypeStruct((params.n_ou, 2, nkx, nky), ctype)
    forcing_key_like_s = jax.ShapeDtypeStruct((), get_key_dtype())

    fields_like_s = jax.ShapeDtypeStruct(shape_complex_s, ctype)
    forcing_scale_like_s = jax.ShapeDtypeStruct((params.n_ou,), ftype)
    state_like_s = SimulationState(t=jax.ShapeDtypeStruct((), ftype), fields=fields_like_s,
                                    forcing_state=forcing_state_like_s, forcing_key=forcing_key_like_s,
                                    forcing_scale=forcing_scale_like_s)
    options = ocp.CheckpointManagerOptions()

    #iterate over saved ranks and extract overlapping z-slices (fields/t only)
    for rank_s in range(p_save):
        z_start_s = rank_s * nz_save
        z_end_s = (rank_s + 1) * nz_save

        #check for overlap
        g_start = max(z_start_l, z_start_s)
        g_end = min(z_end_l, z_end_s)

        if g_start < g_end:
            #overlap exists. Setup a checkpoint manager to read from rank_s
            path_s = os.path.abspath(os.path.join(snap_path, str(rank_s))) if p_save > 1 else os.path.abspath(snap_path)
            mngr_s = ocp.CheckpointManager(path_s, options=options)
            state_s = _restore_or_advise(mngr_s, isnap, state_like_s)

            #slice coordinates
            s_start = g_start - z_start_s
            s_end = g_end - z_start_s
            l_start = g_start - z_start_l
            l_end = g_end - z_start_l

            restored_fields = restored_fields.at[:, l_start:l_end, :, :].set(state_s.fields[:, s_start:s_end, :, :])
            restored_t = state_s.t

    # forcing_state/forcing_key/forcing_scale: no z-axis, identical on all saved ranks by
    # construction -> restore once from rank 0
    path_0 = os.path.abspath(os.path.join(snap_path, "0")) if p_save > 1 else os.path.abspath(snap_path)
    mngr_0 = ocp.CheckpointManager(path_0, options=options)
    state_0 = _restore_or_advise(mngr_0, isnap, state_like_s)
    return SimulationState(t=restored_t, fields=restored_fields,
                            forcing_state=state_0.forcing_state, forcing_key=state_0.forcing_key,
                            forcing_scale=state_0.forcing_scale)

def _restore_or_advise(mngr, isnap, state_like):
    # restore, turning the cryptic orbax structure-mismatch error on old snapshots into a
    # pointer at old_snapshot_repair. Orbax names only the FIRST mismatched key, so match
    # any of the forcing fields: pre-forcing_scale snapshots trip on "forcing_scale",
    # pre-forcing (t/fields-only) ones on "forcing_state".
    # NB the match is on orbax's error text; if an orbax upgrade rewords it, the raw
    # error still propagates unchanged — we just lose this friendlier pointer.
    try:
        return mngr.restore(isnap, args=ocp.args.StandardRestore(state_like))
    except ValueError as e:
        if any(k in str(e) for k in ("forcing_scale", "forcing_state", "forcing_key")):
            raise ValueError(
                "This snapshot predates the current SimulationState layout. Upgrade the "
                "snapshot directory once with snapshot_io.old_snapshot_repair(snap_path, params), "
                "then retry."
            ) from e
        raise

def _recover_interrupted_repair(path_r):
    # Self-heal leftovers of a previously interrupted old_snapshot_repair run in path_r:
    # a ".repair_old_<step>" dir is the original step moved aside during the swap — if the
    # step is missing from the main tree, put it back (it'll be re-repaired); if the step
    # is present the swap completed, so the backup is just cleanup. ".repair_tmp_*" dirs
    # are staging areas that are always safe to discard (the original was never touched
    # until the staged copy was complete).
    for name in sorted(os.listdir(path_r)):
        full = os.path.join(path_r, name)
        if name.startswith(".repair_old_"):
            step_name = name[len(".repair_old_"):]
            dest = os.path.join(path_r, step_name)
            if not os.path.exists(dest):
                os.rename(full, dest)
                print(f"recovered step {step_name} from an interrupted repair (will re-repair)")
            else:
                shutil.rmtree(full)
        elif name.startswith(".repair_tmp_"):
            shutil.rmtree(full, ignore_errors=True)

def old_snapshot_repair(snap_path, params, ranks=None):
    # One-time, in-place upgrade of an old snapshot directory to the current tree
    # structure. Understands two legacy layouts:
    #   pre-forcing_scale (t/fields/forcing_state/forcing_key): forcing_scale is added as
    #     zeros — it's a per-step derived quantity recomputed by simulate on start, so
    #     zeros loses nothing;
    #   pre-forcing (t/fields only): forcing_state (zeros), forcing_key (from
    #     params.forcing_seed) and forcing_scale (zeros) are all synthesized.
    # Handles both single-rank (snap_path/<step>) and per-rank (snap_path/<rank>/<step>)
    # directory layouts. Interruption-safe: each step is staged fully before the original
    # is moved aside, both moves are same-filesystem renames, and a rerun first recovers
    # any leftovers of an interrupted swap. Run this from a single process; pass
    # ranks=[...] to restrict (e.g. splitting work across a job array).
    snap_path = str(snap_path)
    ftype, ctype = get_precision_types()
    nkx, nky = params.nx, params.ny // 2 + 1

    # same layout heuristic as get_saved_steps (load_snapshot instead probes for its
    # specific step under snap_path/0; both agree on these two layouts)
    p_save = 1
    if os.path.isdir(os.path.join(snap_path, "0")) and not os.path.exists(os.path.join(snap_path, "0", "default")):
        # top-level numbered dirs that are themselves manager dirs -> per-rank layout
        while os.path.isdir(os.path.join(snap_path, str(p_save))):
            p_save += 1
    nz_save = params.nz // p_save if params.spatial_dimensions == 3 else 1

    t_like = jax.ShapeDtypeStruct((), ftype)
    fields_like = jax.ShapeDtypeStruct((params.nfields, nz_save, nkx, nky), ctype)
    forcing_state_like = jax.ShapeDtypeStruct((params.n_ou, 2, nkx, nky), ctype)
    forcing_key_like = jax.ShapeDtypeStruct((), get_key_dtype())
    legacy_like = _LegacyCkptState(t=t_like, fields=fields_like,
                                   forcing_state=forcing_state_like, forcing_key=forcing_key_like)
    ancient_like = _AncientCkptState(t=t_like, fields=fields_like)
    current_like = SimulationState(t=t_like, fields=fields_like,
                                   forcing_state=forcing_state_like, forcing_key=forcing_key_like,
                                   forcing_scale=jax.ShapeDtypeStruct((params.n_ou,), ftype))
    options = ocp.CheckpointManagerOptions()

    for rank_s in (ranks if ranks is not None else range(p_save)):
        path_r = os.path.abspath(os.path.join(snap_path, str(rank_s))) if p_save > 1 else os.path.abspath(snap_path)
        _recover_interrupted_repair(path_r)
        mngr = ocp.CheckpointManager(path_r, options=options)
        for step in mngr.all_steps():
            try:
                old = mngr.restore(step, args=ocp.args.StandardRestore(legacy_like))
                new = SimulationState(t=old.t, fields=old.fields, forcing_state=old.forcing_state,
                                      forcing_key=old.forcing_key,
                                      forcing_scale=jnp.zeros((params.n_ou,), dtype=ftype))
            except ValueError:
                try:
                    mngr.restore(step, args=ocp.args.StandardRestore(current_like))
                    print(f"rank {rank_s} step {step}: already current, skipping")
                    continue
                except ValueError:
                    try:
                        old = mngr.restore(step, args=ocp.args.StandardRestore(ancient_like))
                        new = SimulationState(t=old.t, fields=old.fields,
                                              forcing_state=jnp.zeros((params.n_ou, 2, nkx, nky), dtype=ctype),
                                              forcing_key=jax.random.key(params.forcing_seed),
                                              forcing_scale=jnp.zeros((params.n_ou,), dtype=ftype))
                    except ValueError:
                        print(f"rank {rank_s} step {step}: structure matches neither the "
                              f"current nor a known legacy layout (corrupt, or shapes "
                              f"don't match params?) — skipping")
                        continue
            # stage the repaired step fully BEFORE touching the original; then swap via
            # two same-filesystem renames (tmp_root lives inside path_r). If interrupted
            # between them, _recover_interrupted_repair on the next run restores the
            # original from .repair_old_<step> automatically.
            tmp_root = tempfile.mkdtemp(prefix=".repair_tmp_", dir=path_r)
            tmp_mngr = ocp.CheckpointManager(tmp_root, options=ocp.CheckpointManagerOptions())
            tmp_mngr.save(step, args=ocp.args.StandardSave(new))
            tmp_mngr.wait_until_finished()
            tmp_mngr.close()
            old_backup = os.path.join(path_r, f".repair_old_{step}")
            os.rename(os.path.join(path_r, str(step)), old_backup)
            os.rename(os.path.join(tmp_root, str(step)), os.path.join(path_r, str(step)))
            shutil.rmtree(old_backup)
            shutil.rmtree(tmp_root, ignore_errors=True)
            print(f"rank {rank_s} step {step}: repaired")


def find_items(isnap,snap_path):
    db_path=os.path.join(snap_path, str(isnap), "default")

    kv_spec = {
        'driver': 'ocdbt',
        'base': {'driver': 'file', 'path': db_path}
    }
    kvs = ts.KvStore.open(kv_spec).result()

    print("Found these keys in the database:")
    for key in kvs.list().result():
        print(f"  {key.decode()}")

def load_slice(isnap,iz,nzslice,snap_path,item='fields'):
    #This loads a slice of a snapshot into memory: useful for laptop diagnostics
    #Use find_items to check what to put as item here.
    db_path = os.path.join(snap_path, str(isnap), "default")
    spec = {'driver': 'zarr', 'kvstore': {
        'driver': 'ocdbt', 'base': {
            'driver': 'file', 'path': db_path,
            }
        },
        'path': item,       
    }
    f = ts.open(spec).result()
    return f[iz:iz+nzslice , :, :].read().result()