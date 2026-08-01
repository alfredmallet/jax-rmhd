# Correctness test for comm_backend="jax" (shard_map / ppermute / psum) WITHOUT MPI
# or GPUs: bootstrap() forces 4 fake XLA host devices in one process (when the MPI stub
# is active), so the real mesh/collective code path runs and can be compared against the
# serial mpi4jax path. pytest: `pytest tests/test_backend_jax.py` (skipped unless >=4
# devices, fp64 session only). Script: `python tests/test_backend_jax.py`.
#
# Shared end-states live in lru_cached builders that return ONLY numpy copies + the
# saved snapshot dir -- the live jax states never leave the builders, so no test can
# hit the "Array has been deleted" donation hazard, in any execution order.
from _rmhd_testing import bootstrap, checks, managed_manager, snap_dir, zero_ic

bootstrap()

import functools
import json
import os
import shutil
import tempfile

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import PartitionSpec as P

import jax_rmhd as jr
import jax_rmhd.snapshot_io as sn
from jax_rmhd import comms
from jax_rmhd.physics import rmhd, shared_physics

_KW = dict(nx=16, ny=16, nz=16, Lx=2 * np.pi, Ly=2 * np.pi, Lz=2 * np.pi,
           diss=(1e-4, 1e-4), hyper=2, cfl_safety=0.5, dims=3,
           forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(0.5, 0.5),
           forcing_tau=1.0, fshell=(1, 3), forcing_seed=7)
T_END = 0.5


def _kd(k):
    return np.asarray(jax.random.key_data(k))


def _is_key(k):
    return jax.dtypes.issubdtype(k.dtype, jax.dtypes.prng_key)


@functools.lru_cache(maxsize=None)
def _bctx(backend, alt=False):
    kw = dict(_KW, cfl_every=2, lsrk_scan=False, forcing_norm_per_step=False) if alt else _KW
    p = jr.Parameters(comm_backend=backend, **kw)
    return p, jr.setup_kgrids(p)


def _fresh_mpi4jax_params(rank=None, size=None, **overrides):
    """Uncached mpi4jax Parameters on _KW, optionally with spoofed rank/size
    (snapshot-layout tricks only -- never hand these to jitted physics)."""
    p = jr.Parameters(comm_backend="mpi4jax", **dict(_KW, **overrides))
    if size is not None:
        p.size, p.rank = size, rank
    return p


@functools.lru_cache(maxsize=None)
def _end_run(backend):
    """Simulate to T_END once per backend and save one snapshot at step 0 into a
    module-lifetime tmp dir. Returns only numpy/python data + the dir path."""
    p, kg = _bctx(backend)
    st = jr.initialize(zero_ic, p)
    scratch = tempfile.mkdtemp()
    with managed_manager(p, scratch, nsnap=2) as m:
        end = jr.simulate(st, kg, p, t_snap=10.0, t_end=T_END, mngr=m, save=False)
    snap = tempfile.mkdtemp()
    mngr = jr.snapshot_manager_setup(p, snap, nsnap=2)
    manager_dir_is_root = os.path.abspath(str(mngr.directory)) == os.path.abspath(snap)
    sn.save_snapshot(0, end, mngr, p)
    mngr.wait_until_finished()
    mngr.close()
    return dict(fields=np.asarray(end.fields), t=float(end.t),
                forcing_state=np.asarray(end.forcing_state),
                forcing_scale=np.asarray(end.forcing_scale),
                key_data=_kd(end.forcing_key),
                nshards=len(end.fields.addressable_shards),
                shard_nz=end.fields.addressable_shards[0].data.shape[1],
                snap=snap, manager_dir_is_root=manager_dir_is_root)


@functools.lru_cache(maxsize=None)
def _per_rank_tree():
    """A genuine 2-rank mpi4jax per-rank tree (what `mpirun -n 2` would write),
    built by faking rank/size on the real mpi4jax writer path. Holds the mpi4jax
    reference end fields split in half along z."""
    ref = _end_run("mpi4jax")
    template = sn.load_snapshot(0, ref["snap"], _bctx("mpi4jax")[0])
    hz = _KW["nz"] // 2
    d = tempfile.mkdtemp()
    for r in range(2):
        p_w = _fresh_mpi4jax_params(rank=r, size=2)
        m = jr.snapshot_manager_setup(p_w, d, nsnap=2)
        sn.save_snapshot(0, template._replace(
            fields=jnp.asarray(ref["fields"][:, r * hz:(r + 1) * hz])), m, p_w)
        m.wait_until_finished()
        m.close()
    return d


@pytest.mark.multidev
@pytest.mark.fp64
def test_forced_devices_present():
    assert jax.device_count() == 4, f"got {jax.device_count()}"


@pytest.mark.multidev
@pytest.mark.fp64
def test_dims2_rejected_for_jax_backend():
    with pytest.raises(ValueError):
        jr.Parameters(nx=8, ny=8, Lx=1.0, Ly=1.0, diss=(0.0, 0.0), hyper=1,
                      cfl_safety=0.5, dims=2, comm_backend="jax")


@pytest.mark.multidev
@pytest.mark.fp64
def test_same_seed_run_matches_serial_reference():
    ref, jx = _end_run("mpi4jax"), _end_run("jax")
    p_ref, kg_ref = _bctx("mpi4jax")
    ndev, nz = jax.device_count(), _KW["nz"]

    def energy(fields):
        fk = jnp.asarray(fields)
        return 0.5 * float(shared_physics.perp_inner_product(fk[0], fk[0], kg_ref, p_ref)
                           + shared_physics.perp_inner_product(fk[1], fk[1], kg_ref, p_ref))

    rel = np.max(np.abs(jx["fields"] - ref["fields"])) / np.max(np.abs(ref["fields"]))
    fs_diff = float(np.max(np.abs(jx["forcing_state"] - ref["forcing_state"])))
    sc_rel = float(np.max(np.abs(jx["forcing_scale"] - ref["forcing_scale"]))
                   / np.max(np.abs(ref["forcing_scale"])))
    E_ref, E_jax = energy(ref["fields"]), energy(jx["fields"])
    with checks() as c:
        c.check(f"jax-backend fields have the global shape {ref['fields'].shape}",
                jx["fields"].shape == ref["fields"].shape)
        c.check(f"fields sharded over {ndev} devices along z",
                jx["nshards"] == ndev and jx["shard_nz"] == nz // ndev)
        c.check(f"final fields agree with the serial reference (rel {rel:.2e} < 1e-14)",
                rel < 1e-14)
        c.check(f"final time matches ({ref['t']:.12f})", jx["t"] == ref["t"])
        # forcing is replicated, never sharded: the O-U stream must be bit-identical
        c.check(f"forcing_state bit-identical across backends (max|diff| {fs_diff:g})",
                fs_diff == 0.0)
        c.check("forcing_state replicated (single global shard shape preserved)",
                jx["forcing_state"].shape == ref["forcing_state"].shape)
        c.check(f"forcing_scale agrees (rel {sc_rel:.2e} < 1e-12)", sc_rel < 1e-12)
        # energies (same normalization convention as diagnostics.perpspec)
        c.check(f"energy matches: ref {E_ref:.12f} vs jax {E_jax:.12f}",
                abs(E_jax - E_ref) <= 1e-12 * abs(E_ref))
        c.check(f"energy is finite and nonzero ({E_ref:.6f})",
                np.isfinite(E_ref) and E_ref > 0.0)


@pytest.mark.multidev
@pytest.mark.fp64
def test_z_derivatives_across_ppermute_halo():
    # a wrong ppermute direction would silently mirror the z stencil
    p_ref, _ = _bctx("mpi4jax")
    p_jax, _ = _bctx("jax")
    nx, ny, nz = _KW["nx"], _KW["ny"], _KW["nz"]
    rng = np.random.default_rng(0)
    f_loc = jnp.asarray(rng.standard_normal((p_ref.nfields, nz, nx, ny // 2 + 1))
                        + 1j * rng.standard_normal((p_ref.nfields, nz, nx, ny // 2 + 1)))
    dz_ref = shared_physics.z_derivatives(f_loc, p_ref)[0]
    dz_fn = comms._shard_map(lambda f: shared_physics.z_derivatives(f, p_jax)[0],
                             (P(None, comms.Z_AXIS),), P(None, comms.Z_AXIS))
    dz_jax = jax.jit(dz_fn)(comms.to_global(f_loc, p_jax, z_axis=1))
    # tolerance, not equality: the reference runs op-by-op and the sharded one is
    # jitted, so XLA fusion differs -- a flipped ppermute direction shows up as an
    # O(1) error, not 1e-15
    dz_err = float(np.max(np.abs(np.asarray(dz_jax) - np.asarray(dz_ref)))
                   / np.max(np.abs(np.asarray(dz_ref))))
    assert dz_err < 1e-14, f"rel {dz_err:.2e}"


@pytest.mark.multidev
@pytest.mark.fp64
def test_checkpoint_roundtrip_and_cross_backend_restart():
    # The jax backend writes ONE shared global directory and hands orbax the
    # z-sharded global jax.Arrays; layouts may differ per backend -- what must hold
    # is cross-restartability both ways.
    ref, jx = _end_run("mpi4jax"), _end_run("jax")
    p_ref, _ = _bctx("mpi4jax")
    p_jax, _ = _bctx("jax")
    ndev, nz = jax.device_count(), _KW["nz"]
    p_ref.save(jx["snap"])
    p_jax.save(jx["snap"])  # comm_backend must not count as a differing parameter
    back_jax = sn.load_snapshot(0, jx["snap"], p_jax)
    rt = float(np.max(np.abs(np.asarray(back_jax.fields) - jx["fields"])))
    back_ref = sn.load_snapshot(0, jx["snap"], p_ref)
    cross = float(np.max(np.abs(np.asarray(back_ref.fields) - jx["fields"])))
    # ... and with z-slicing: an mpi4jax rank that owns only part of z. No MPI here,
    # so rank/size are spoofed -- load_snapshot reads only those two.
    hz = nz // 2
    half = sn.load_snapshot(0, jx["snap"], _fresh_mpi4jax_params(rank=1, size=2))
    h_err = float(np.max(np.abs(np.asarray(half.fields) - jx["fields"][:, hz:])))
    with checks() as c:
        c.check("params.save accepts both backends in one run directory", True)
        c.check("jax backend's manager is ONE shared directory (no per-rank subdir)",
                jx["manager_dir_is_root"])
        c.check(f"jax-backend save produced the flat layout "
                f"(got {sn.snapshot_layout(jx['snap'])!r})",
                sn.snapshot_layout(jx["snap"]) == "flat")
        c.check(f"jax-backend snapshot roundtrip exact (max|diff| {rt:g})", rt == 0.0)
        c.check(f"restored fields re-sharded over {ndev} devices along z",
                len(back_jax.fields.addressable_shards) == ndev
                and back_jax.fields.addressable_shards[0].data.shape[1] == nz // ndev)
        c.check("restored global fields have the full global shape",
                back_jax.fields.shape == ref["fields"].shape)
        c.check(f"cross-backend restart (jax-written global dir read by mpi4jax) "
                f"exact (max|diff| {cross:g})",
                cross == 0.0 and float(back_ref.t) == jx["t"])
        c.check(f"mpi4jax rank 1/2 reads only its z-slice of the jax global dir "
                f"(shape {tuple(half.fields.shape)}, max|diff| {h_err:g})",
                half.fields.shape == (p_ref.nfields, hz, _KW["nx"], _KW["ny"] // 2 + 1)
                and h_err == 0.0)
        # keys are stored as real key arrays in both layouts (orbax unwraps/rewraps
        # typed PRNG keys)
        c.check("forcing_key roundtrips as a key array under the jax backend",
                _is_key(back_jax.forcing_key)
                and np.array_equal(_kd(back_jax.forcing_key), jx["key_data"]))
        c.check("forcing_key survives the cross-backend read",
                _is_key(back_ref.forcing_key)
                and np.array_equal(_kd(back_ref.forcing_key), jx["key_data"]))


@pytest.mark.multidev
@pytest.mark.fp64
def test_restart_continues_identically():
    # Fresh loads for both continuations -- simulate donates its input state.
    jx = _end_run("jax")
    p_jax, kg_jax = _bctx("jax")
    p_ref, kg_ref = _bctx("mpi4jax")
    with snap_dir() as d2, managed_manager(p_jax, d2, nsnap=2) as m2:
        cont_jax = jr.simulate(sn.load_snapshot(0, jx["snap"], p_jax), kg_jax, p_jax,
                               t_snap=10.0, t_end=T_END + 0.2, mngr=m2, save=False)
    with snap_dir() as d3, managed_manager(p_ref, d3, nsnap=2) as m3:
        cont_ref = jr.simulate(sn.load_snapshot(0, jx["snap"], p_ref), kg_ref, p_ref,
                               t_snap=10.0, t_end=T_END + 0.2, mngr=m3, save=False)
    c_rel = np.max(np.abs(np.asarray(cont_jax.fields) - np.asarray(cont_ref.fields))) \
        / np.max(np.abs(np.asarray(cont_ref.fields)))
    assert c_rel < 1e-14, f"rel {c_rel:.2e}"


@pytest.mark.multidev
@pytest.mark.fp64
def test_scan_cfl_block_unrolled_per_stage_norm_agree():
    # the other stepper paths: simulate_scan + cfl_every>1 + unrolled LSRK +
    # per-stage normalization + snapshot writing, again ref-vs-jax
    def run_scan(backend):
        p, kg = _bctx(backend, alt=True)
        st = jr.initialize(zero_ic, p)
        nblock = jr.estimate_good_nblock(st, kg, p, t_snap=0.2, t_end=0.2, nblock_min=4)
        with snap_dir() as d, managed_manager(p, d, nsnap=4) as m:
            end = jr.simulate_scan(st, kg, p, nblock=nblock, t_snap=0.1, t_end=0.2,
                                   mngr=m, save=True)
            return np.asarray(end.fields), float(end.t), nblock

    f_ref, t_ref, nb_ref = run_scan("mpi4jax")
    f_jax, t_jax, nb_jax = run_scan("jax")
    s_rel = np.max(np.abs(f_jax - f_ref)) / np.max(np.abs(f_ref))
    with checks() as c:
        c.check(f"estimate_good_nblock agrees across backends ({nb_ref})",
                nb_ref == nb_jax)
        c.check(f"simulate_scan + cfl_every=2 + lsrk_scan=False + per-stage norm "
                f"agree (rel {s_rel:.2e} < 1e-14)", s_rel < 1e-14 and t_jax == t_ref)


def _leaf_meta(path, isnap=0):
    m = json.load(open(os.path.join(path, str(isnap), "default", "_METADATA")))["tree_metadata"]
    return sorted(m.keys()), {v["value_metadata"]["value_type"] for v in m.values()}


@pytest.mark.multidev
@pytest.mark.fp64
def test_on_disk_leaf_sets_pruning_and_reverse_cross_restore():
    # the on-disk layouts may DIFFER between backends. What must hold is the leaf SET
    # (one pytree structure), both cross-restores, and an untouched mpi4jax writer path.
    ref, jx = _end_run("mpi4jax"), _end_run("jax")
    p_jax, _ = _bctx("jax")
    keys_ref, types_ref = _leaf_meta(ref["snap"])
    keys_jax, types_jax = _leaf_meta(jx["snap"])
    with checks() as c:
        c.check(f"on-disk leaf set identical across backends ({len(keys_ref)} leaves)",
                keys_ref == keys_jax)
        c.check(f"both backends store jax.Array leaves (mpi4jax {types_ref} / jax {types_jax})",
                types_ref == {"jax.Array"} and types_jax == {"jax.Array"})
        # one shared manager -> max_to_keep prunes once, globally
        with snap_dir() as prune_dir:
            with managed_manager(p_jax, prune_dir, nsnap=2) as m_p:
                state = sn.load_snapshot(0, jx["snap"], p_jax)
                for i in range(4):
                    sn.save_snapshot(i, state, m_p, p_jax)
                    m_p.wait_until_finished()
                kept = sorted(sn.get_saved_steps(prune_dir, p_jax))
                c.check(f"nsnap/max_to_keep honored under the jax backend (kept {kept})",
                        kept == [2, 3])
        back_x = sn.load_snapshot(0, ref["snap"], p_jax)  # the other direction
        xr = float(np.max(np.abs(np.asarray(back_x.fields) - ref["fields"])))
        c.check(f"cross-backend restart (mpi4jax-written snapshot read by the jax "
                f"backend) exact (max|diff| {xr:g})",
                xr == 0.0 and float(back_x.t) == ref["t"] and _is_key(back_x.forcing_key))


@pytest.mark.multidev
@pytest.mark.fp64
def test_jax_backend_reads_per_rank_mpi4jax_tree():
    # the Savio phase-4a case: the jax backend restarts from a real 2-rank tree
    ref = _end_run("mpi4jax")
    pr_dir = _per_rank_tree()
    p_ref, _ = _bctx("mpi4jax")
    p_jax, _ = _bctx("jax")
    ndev = jax.device_count()
    with checks() as c:
        c.check(f"2-rank mpi4jax writer produced the per-rank layout "
                f"(got {sn.snapshot_layout(pr_dir)!r})",
                sn.snapshot_layout(pr_dir) == "per_rank"
                and sorted(os.listdir(pr_dir)) == ["0", "1"])
        pr_jax = sn.load_snapshot(0, pr_dir, p_jax)
        pr_err = float(np.max(np.abs(np.asarray(pr_jax.fields) - ref["fields"])))
        c.check(f"jax backend unions a 2-rank mpi4jax tree into global z-sharded "
                f"arrays (max|diff| {pr_err:g})",
                pr_err == 0.0 and len(pr_jax.fields.addressable_shards) == ndev
                and float(pr_jax.t) == ref["t"] and _is_key(pr_jax.forcing_key))
        pr_ref = sn.load_snapshot(0, pr_dir, p_ref)
        c.check("... and the mpi4jax backend reads the same tree identically",
                float(np.max(np.abs(np.asarray(pr_ref.fields) - ref["fields"]))) == 0.0)


@pytest.mark.multidev
@pytest.mark.fp64
def test_layout_detection_ignores_stranded_tmp_dirs():
    # snap/0 exists in BOTH layouts; the marker is _CHECKPOINT_METADATA / the item
    # subdir directly inside it. Stranded "<step>.orbax-checkpoint-tmp" dirs (real
    # Savio leftovers) must confuse neither the detector nor get_saved_steps.
    # Decoys go into COPIES so the shared cached dirs stay pristine for other tests.
    jx = _end_run("jax")
    pr_src = _per_rank_tree()
    with snap_dir() as work:
        flat = os.path.join(work, "flat")
        per_rank = os.path.join(work, "per_rank")
        shutil.copytree(jx["snap"], flat)
        shutil.copytree(pr_src, per_rank)
        for decoy_root, decoy in ((flat, "0.orbax-checkpoint-tmp"),
                                  (per_rank, "3.orbax-checkpoint-tmp"),
                                  (os.path.join(per_rank, "0"), "7.orbax-checkpoint-tmp")):
            os.makedirs(os.path.join(decoy_root, decoy), exist_ok=True)
            open(os.path.join(decoy_root, decoy, "_CHECKPOINT_METADATA"), "w").close()
        empty_dir = os.path.join(work, "empty")
        os.makedirs(empty_dir)
        cases = [(flat, "flat", [0]), (per_rank, "per_rank", [0]),
                 (empty_dir, "empty", [])]
        repo = os.path.dirname(os.path.dirname(os.path.abspath(sn.__file__)))
        legacy_tree = os.path.join(repo, "tests", "data", "forced_turbulence_64cubed")
        if os.path.isdir(legacy_tree):  # untracked local fixture, read-only
            cases.append((legacy_tree, "per_rank", [0, 1, 2, 3, 4]))
        with checks() as c:
            for path, want, want_steps in cases:
                got, got_steps = sn.snapshot_layout(path), sorted(sn.get_saved_steps(path))
                c.check(f"layout of {os.path.basename(path) or path} is {want!r} with "
                        f"steps {want_steps} (got {got!r}, {got_steps})",
                        got == want and got_steps == want_steps)


@pytest.mark.multidev
@pytest.mark.fp64
def test_mixed_layout_writers_refused():
    # mixing WRITERS of the two layouts in one directory is refused (reading across
    # layouts stays supported -- that is what the cross-restore tests check)
    jx = _end_run("jax")
    pr_dir = _per_rank_tree()
    p_ref, _ = _bctx("mpi4jax")
    p_jax, _ = _bctx("jax")
    with pytest.raises(ValueError):
        jr.snapshot_manager_setup(p_jax, pr_dir, nsnap=2)
    with pytest.raises(ValueError):
        jr.snapshot_manager_setup(_fresh_mpi4jax_params(rank=0, size=2), jx["snap"],
                                  nsnap=2)
    # single-process mpi4jax may still continue a flat dir
    with managed_manager(p_ref, jx["snap"], nsnap=2):
        pass


@pytest.mark.multidev
@pytest.mark.fp64
def test_restores_pinned_to_local_device():
    # With jax.distributed up each process owns different device ids, so a template
    # leaf WITHOUT a sharding makes orbax follow the checkpoint's recorded device
    # ("Device cpu:0 was not found in jax.local_devices()"). Every process-local
    # restore must pin this process's own device; hiding cpu:0 is the stand-in for
    # "this process doesn't own the writer's device".
    ref = _end_run("mpi4jax")
    pr_dir = _per_rank_tree()
    _real_local = jax.local_devices
    try:
        jax.local_devices = lambda *a, **k: _real_local(*a, **k)[1:]
        with checks() as c:
            for name in ("mpi4jax", "jax"):
                p_, _ = _bctx(name)
                try:
                    again = sn.load_snapshot(0, pr_dir, p_)  # per-rank: local templates
                    c.check(f"{name} restore pinned to this process's device (cpu:0 hidden)",
                            float(again.t) == ref["t"])
                except Exception as e:  # noqa: BLE001 - reported as a failed check
                    c.check(f"{name} restore pinned to this process's device (cpu:0 hidden)",
                            False, f"raised {type(e).__name__}: {str(e)[:160]}")
    finally:
        jax.local_devices = _real_local


@pytest.mark.multidev
@pytest.mark.fp64
def test_set_timestep_pmax_matches_serial():
    # the only DIRECT exercise of allreduce_max's pmax branch. The IC's
    # amplitude varies with z, so each z-shard sees a DIFFERENT local CFL maximum
    # and the global dt is wrong unless pmax really reduces across the mesh (the
    # forced-run comparisons only ever feed it shard-identical values early on).
    def zvar_ic(x, y, z):
        # amplitude 5*(1.5+sin z): grad terms ~ tens >> 1/dz, and z-dependent
        phi = 5.0 * (1.5 + jnp.sin(z)) * jnp.cos(x) * jnp.cos(y)
        return jnp.stack([phi, 0.5 * phi])

    p_ref, kg_ref = _bctx("mpi4jax")
    p_jax, kg_jax = _bctx("jax")
    st_ref = jr.initialize(zvar_ic, p_ref)                     # process-local
    dt_ref = float(rmhd.set_timestep(rmhd.grad(st_ref, kg_ref, p_ref), p_ref))
    st_jax = jr.initialize(zvar_ic, p_jax)                     # global z-sharded
    f = comms.shard_call(lambda s, kg: rmhd.set_timestep(rmhd.grad(s, kg, p_jax), p_jax),
                         p_jax, kg_jax, out_specs=P())
    dt_jax = float(jax.jit(f)(st_jax, kg_jax))
    rel = abs(dt_jax - dt_ref) / dt_ref
    with checks() as c:
        c.check("serial CFL dt is grad-limited, not 1/dz-limited (test not vacuous)",
                dt_ref < p_ref.cfl_safety * p_ref.dz)
        c.check(f"set_timestep under shard_call (pmax) matches serial "
                f"({dt_jax:.12e} vs {dt_ref:.12e}, rel {rel:.2e})", rel < 1e-13)


@pytest.mark.multidev
@pytest.mark.fp64
def test_legacy_pre_phase3_snapshot_repair():
    # tests/data/forced_turbulence_64cubed is the reference pre-forcing_scale layout,
    # written long before either the jax backend or forcing_scale existed. It is an
    # untracked local fixture (tests/data is gitignored): absent on fresh clones.
    # Read-only: ONE step dir (~230 kB) is copied out and the copy repaired.
    repo = os.path.dirname(os.path.dirname(os.path.abspath(sn.__file__)))
    legacy_src = os.path.join(repo, "tests", "data", "forced_turbulence_64cubed", "0", "4")
    if not os.path.isdir(legacy_src):
        print("[SKIP] legacy snapshot tests/data/forced_turbulence_64cubed not present")
        return
    lkw = dict(nx=64, ny=64, nz=8, forcing_seed=42)
    with snap_dir() as legacy_dir:
        shutil.copytree(legacy_src, os.path.join(legacy_dir, "4"))
        # nz=8/nx=ny=64 reproduces that run's per-rank shard shape (2,8,64,33) in a
        # 1-rank layout
        p_leg = _fresh_mpi4jax_params(**lkw)
        with checks() as c:
            try:
                sn.load_snapshot(4, legacy_dir, p_leg)
                pre_ok = False
            except ValueError as e:
                pre_ok = "old_snapshot_repair" in str(e)
            c.check("legacy snapshot still routed to old_snapshot_repair "
                    "(structure check first)", pre_ok)
            sn.old_snapshot_repair(legacy_dir, p_leg)
            leg_ref = sn.load_snapshot(4, legacy_dir, p_leg)
            leg_jax = sn.load_snapshot(4, legacy_dir,
                                       jr.Parameters(comm_backend="jax", **dict(_KW, **lkw)))
            ld = float(np.max(np.abs(np.asarray(leg_jax.fields) - np.asarray(leg_ref.fields))))
            c.check(f"repaired legacy snapshot reads identically under both backends "
                    f"(max|diff| {ld:g})",
                    ld == 0.0 and float(leg_jax.t) == float(leg_ref.t)
                    and np.all(np.isfinite(np.asarray(leg_ref.fields)))
                    and float(np.max(np.abs(np.asarray(leg_ref.fields)))) > 0.0)
            c.check("... and its forcing_key comes back as a key array under the jax backend",
                    _is_key(leg_jax.forcing_key) and _is_key(leg_ref.forcing_key))


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
