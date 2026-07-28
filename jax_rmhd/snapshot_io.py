import jax
import jax.numpy as jnp
import tensorstore as ts
import orbax.checkpoint as ocp
from etils import epath
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

# orbax's default item subdirectory inside a step dir (snap_path/<step>/default/...)
_ITEM = "default"

_array_handler_pinned = False

def _pin_array_handler():
    # comm_backend="jax" only, idempotent: turn OFF orbax's replica-parallel writing.
    # Replica-parallel splits a REPLICATED array's write across the hosts holding copies;
    # ours (t/forcing_*) are a few kB, so it buys nothing, while orbax 0.12.1 implements the
    # split with `with jax.sharding.set_mesh(...)`
    # (_src/serialization/replica_slices.py::ReplicaSlice.data) — a context manager only on
    # newer jax, AttributeError: __enter__ on jax 0.6.x. Flipping the flag on the REGISTERED
    # handler (rather than registering a fresh one) keeps every other orbax default,
    # including the array_metadatas sidecar and write_shape metadata. A no-op if orbax
    # renames the attribute: its default path is correct on a matching jax.
    global _array_handler_pinned
    if _array_handler_pinned:
        return
    _array_handler_pinned = True
    from orbax.checkpoint import type_handlers as _th
    handler = _th.get_type_handler(jax.Array)
    if hasattr(handler, "_use_replica_parallel"):
        handler._use_replica_parallel = False

# Setting up Orbax stuff
def snapshot_manager_setup(params,snap_path="data",nsnap=1000):
    # comm_backend="jax": ONE shared manager over ONE shared directory driven by every
    # process — orbax's native multihost mode (each process writes its own addressable
    # shards, atomicity/pruning coordinated upstream). create=True is race-free here: only
    # the primary host mkdirs and everyone barriers (checkpoint_manager.py::
    # _create_root_directory, orbax 0.12.1). mpi4jax keeps the per-rank layout unchanged.
    # Refuse to WRITE a layout into a directory already holding the other one: the manager
    # enumerates numbered subdirs as steps, so a shared jax manager over a per-rank tree
    # would read rank dirs as steps and let max_to_keep delete them. Reading across layouts
    # (load_snapshot) is fine and stays supported — only mixing writers is blocked.
    _layout = snapshot_layout(snap_path)
    # ANY flat-layout writer (jax backend, or a size==1 run of either backend) is refused
    # here: its manager sits directly over snap_path, so orbax reads the rank dirs as steps
    # and max_to_keep PRUNES them (verified on orbax 0.12.1 — silent loss of saved data).
    if _layout=="per_rank" and (params.comm_backend=="jax" or params.size==1):
        raise ValueError(
            f"{snap_path} holds a per-rank (mpi4jax size>1) snapshot tree; this run writes "
            f"a single flat directory and its manager would mistake the rank dirs for steps "
            f"(max_to_keep would prune/DELETE them). Point the run at a new snap_path "
            f"(load_snapshot can still restart from this one).")
    if params.comm_backend!="jax" and params.size>1 and _layout=="flat":
        raise ValueError(
            f"{snap_path} holds a flat snapshot directory (jax-backend or single-process "
            f"layout); mpi4jax with size={params.size} writes per-rank subdirs and the two "
            f"would interleave. Point the run at a new snap_path (load_snapshot can still "
            f"restart from this one).")
    if params.comm_backend=="jax":
        _pin_array_handler()
        checkpoint_path = os.path.abspath(snap_path)
    elif params.size>1:
        checkpoint_path = os.path.abspath(snap_path+f'/{params.rank}')
    else:
        checkpoint_path = os.path.abspath(snap_path)
    return ocp.CheckpointManager(directory=checkpoint_path,
                                 options=ocp.CheckpointManagerOptions(max_to_keep=nsnap))

def _wrap_key(k):
    # Keys are stored as real key arrays in both layouts (orbax unwraps a typed PRNG key to
    # its uint32 key_data and records the impl, then rewraps on restore), so this is a no-op
    # for anything current; it only rescues a leaf that came back as raw key_data.
    if jax.dtypes.issubdtype(k.dtype, jax.dtypes.prng_key):
        return k
    return jax.random.wrap_key_data(jnp.asarray(k))

def _local_sharding():
    # Pin process-local restores to THIS process's device. Without an explicit sharding
    # orbax follows the sharding recorded in the checkpoint
    # (_src/handlers/standard_checkpoint_handler.py::_get_sharding_for_target_leaf), which
    # names devices this process may not own once jax.distributed is up ("Device cpu:0 was
    # not found in jax.local_devices()" / "Topology mismatch").
    return jax.sharding.SingleDeviceSharding(jax.local_devices()[0])

def _is_global_array(x):
    # True for a z-sharded / multi-process jax.Array (what the "jax" backend's states hold)
    return (hasattr(x, "addressable_shards")
            and (not x.is_fully_addressable or len(x.addressable_shards) > 1))

############################
# on-disk layout detection #
############################

def _numbered_subdirs(path):
    # Numeric subdirectory names only: this is what makes the probes below immune to
    # stranded "<step>.orbax-checkpoint-tmp" leftovers (".isdigit()" rejects them).
    if not os.path.isdir(path):
        return []
    return sorted(int(n) for n in os.listdir(path)
                  if n.isdigit() and os.path.isdir(os.path.join(path, n)))

def _is_step_dir(path):
    # An orbax STEP dir carries _CHECKPOINT_METADATA / the item subdir directly; a per-rank
    # dir holds numbered STEP dirs instead. snap_path/0 exists in both layouts, so this
    # marker (not the mere existence of "0") is what tells them apart.
    return (os.path.exists(os.path.join(path, "_CHECKPOINT_METADATA"))
            or os.path.isdir(os.path.join(path, _ITEM)))

def _step_dirs(path):
    # finalized step indices directly under `path`
    return [n for n in _numbered_subdirs(path) if _is_step_dir(os.path.join(path, str(n)))]

def snapshot_layout(snap_path):
    # "flat"     snap_path/<step>/...        — the jax backend's shared global dir, and also
    #                                          a single-process mpi4jax run (both hold the
    #                                          FULL global z range, so readers treat them alike)
    # "per_rank" snap_path/<rank>/<step>/... — mpi4jax with size>1
    # "empty"    nothing saved yet
    snap_path = str(snap_path)
    subs = _numbered_subdirs(snap_path)
    if not subs:
        return "empty"
    if any(_is_step_dir(os.path.join(snap_path, str(n))) for n in subs):
        return "flat"
    return "per_rank"

def save_snapshot(isnap,state,mngr,params=None):
    # saves the full SimulationState; forcing_scale is always a concrete (n_ou,) array
    # (see run.initialize), so all new checkpoints share one tree structure. Snapshots
    # written before forcing_scale existed must be upgraded once with old_snapshot_repair.
    # comm_backend="jax" states are GLOBAL z-sharded jax.Arrays and are handed to orbax as
    # they are — that is orbax's supported multihost path (the host-local guard in
    # _src/serialization/jax_array_handlers.py only rejects fully-addressable arrays).
    if _is_global_array(state.fields) and (params is None or params.comm_backend != "jax"):
        # a global array only belongs in the jax backend's single shared directory; through
        # an mpi4jax per-rank manager it would write global data into a per-rank tree
        raise ValueError("save_snapshot(...) got a sharded/multi-process global state but "
                         "no comm_backend='jax' params. Pass the jax-backend params (4th "
                         "argument) and a manager built by snapshot_manager_setup.")
    if state.forcing_scale is None:
        # orbax records None children in its metadata, so letting this through would fork
        # the on-disk tree structure between runs (hand-built states hit this; the
        # ForcingTerm check only catches it when forcing_norm_per_step is on)
        raise ValueError("state.forcing_scale is None; checkpoints must always carry a "
                         "concrete (n_ou,) array so all snapshots share one tree structure. "
                         "Build states via run.initialize / load_snapshot, or use "
                         "state._replace(forcing_scale=jnp.zeros((params.n_ou,))).")
    return mngr.save(isnap,args=ocp.args.StandardSave(state))

def get_saved_steps(snap_path, params=None):
    # Plain directory scan (params accepted for API compatibility, unused): handles the
    # per-rank, flat/global and legacy layouts alike, needs no CheckpointManager — and so
    # no multihost barriers — and ignores stranded *.orbax-checkpoint-tmp dirs.
    snap_path = str(snap_path)
    if snapshot_layout(snap_path) == "per_rank":
        return _step_dirs(os.path.join(snap_path, "0"))
    return _step_dirs(snap_path)

def _restore_step(snap_path, isnap, state_like):
    # PROCESS-LOCAL read: a bare StandardCheckpointHandler, never a CheckpointManager.
    # CheckpointManager construction and Checkpointer.restore both call
    # multihost.sync_global_processes (checkpoint_manager.py::_create_root_directory,
    # _src/checkpointers/checkpointer.py::restore), which deadlocks the moment different
    # ranks read different directories (the mpi4jax per-rank tree) or a different number of
    # them (resharding). The handler itself is barrier-free.
    d = os.path.join(os.path.abspath(str(snap_path)), str(isnap), _ITEM)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"no snapshot {isnap} under {snap_path} (expected {d})")
    handler = ocp.StandardCheckpointHandler()
    try:
        return handler.restore(epath.Path(d), args=ocp.args.StandardRestore(state_like))
    finally:
        getattr(handler, "close", lambda: None)()

def _state_template(params, nz, sharding=None, fields_sharding=None):
    # SimulationState of ShapeDtypeStructs with `nz` z-planes; sharding=None lets orbax
    # follow the checkpoint's own recorded sharding.
    ftype, ctype = get_precision_types()
    nkx, nky = params.nx, params.ny // 2 + 1
    fs = sharding if fields_sharding is None else fields_sharding
    return SimulationState(
        t=jax.ShapeDtypeStruct((), ftype, sharding=sharding),
        fields=jax.ShapeDtypeStruct((params.nfields, nz, nkx, nky), ctype, sharding=fs),
        # forcing_state/key/scale have no z-axis and are identical on every rank
        forcing_state=jax.ShapeDtypeStruct((params.n_ou, 2, nkx, nky), ctype, sharding=sharding),
        forcing_key=jax.ShapeDtypeStruct((), get_key_dtype(), sharding=sharding),
        forcing_scale=jax.ShapeDtypeStruct((params.n_ou,), ftype, sharding=sharding))

def _load_flat_global(isnap, snap_path, params):
    # comm_backend="jax" reading a flat dir: restore straight into the z-sharded GLOBAL
    # arrays, so each process only reads its own shards and the rank count at save time is
    # irrelevant (the file holds the whole z range).
    from . import comms
    from jax.sharding import NamedSharding, PartitionSpec as P
    _pin_array_handler()
    mesh = comms.get_mesh()
    tmpl = _state_template(params, params.nz, sharding=NamedSharding(mesh, P()),
                           fields_sharding=NamedSharding(mesh, P(None, comms.Z_AXIS)))
    st = _restore_or_advise(snap_path, isnap, tmpl)
    return st._replace(forcing_key=_wrap_key(st.forcing_key))

def load_snapshot(isnap,snap_path,params):
    snap_path = str(snap_path)
    layout = snapshot_layout(snap_path)
    if params.comm_backend == "jax" and layout != "per_rank":
        return _load_flat_global(isnap, snap_path, params)

    p_save = 1
    if layout == "per_rank":
        while os.path.exists(os.path.join(snap_path, str(p_save), str(isnap))):
            p_save += 1

    ctype = get_precision_types()[1]

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

    # Expected tree for one saved rank's file, pinned to this process's own device.
    state_like_s = _state_template(params, nz_save if params.spatial_dimensions == 3 else 1,
                                   sharding=_local_sharding())

    #iterate over saved ranks and extract overlapping z-slices (fields/t only)
    for rank_s in range(p_save):
        z_start_s = rank_s * nz_save
        z_end_s = (rank_s + 1) * nz_save

        #check for overlap
        g_start = max(z_start_l, z_start_s)
        g_end = min(z_end_l, z_end_s)

        if g_start < g_end:
            #overlap exists: read that rank's file (its own dir in a per-rank tree). The
            #subdir choice follows the LAYOUT, not p_save>1: a per-rank tree pruned down to
            #a single rank dir is still snap_path/0/<step>.
            path_s = os.path.join(snap_path, str(rank_s)) if layout == "per_rank" else snap_path
            state_s = _restore_or_advise(path_s, isnap, state_like_s)

            #slice coordinates
            s_start = g_start - z_start_s
            s_end = g_end - z_start_s
            l_start = g_start - z_start_l
            l_end = g_end - z_start_l

            restored_fields = restored_fields.at[:, l_start:l_end, :, :].set(state_s.fields[:, s_start:s_end, :, :])
            restored_t = state_s.t

    # forcing_state/forcing_key/forcing_scale: no z-axis, identical on all saved ranks by
    # construction -> restore once from rank 0
    path_0 = os.path.join(snap_path, "0") if layout == "per_rank" else snap_path
    state_0 = _restore_or_advise(path_0, isnap, state_like_s)
    restored = SimulationState(t=jnp.asarray(restored_t), fields=restored_fields,
                               forcing_state=jnp.asarray(state_0.forcing_state),
                               forcing_key=_wrap_key(state_0.forcing_key),
                               forcing_scale=jnp.asarray(state_0.forcing_scale))
    if params.comm_backend == "jax":
        from . import comms
        restored = comms.state_to_global(restored, params)  # local shards -> global z-sharded arrays
    return restored

def _restore_or_advise(snap_path, isnap, state_like):
    # restore, turning the cryptic orbax structure-mismatch error on old snapshots into a
    # pointer at old_snapshot_repair. Orbax names only the FIRST mismatched key, so match
    # any of the forcing fields: pre-forcing_scale snapshots trip on "forcing_scale",
    # pre-forcing (t/fields-only) ones on "forcing_state".
    # NB the match is on orbax's error text; if an orbax upgrade rewords it, the raw
    # error still propagates unchanged — we just lose this friendlier pointer.
    try:
        return _restore_step(snap_path, isnap, state_like)
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

    # one shared layout detector with get_saved_steps/load_snapshot
    per_rank = snapshot_layout(snap_path) == "per_rank"
    p_save = 1
    if per_rank:
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
        path_r = os.path.abspath(os.path.join(snap_path, str(rank_s))) if per_rank else os.path.abspath(snap_path)
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