# T9 correctness test for comm_backend="jax" (shard_map / ppermute / psum) WITHOUT MPI or
# GPUs: --xla_force_host_platform_device_count=4 gives 4 fake devices in one process, so the
# real mesh/collective code path runs and can be compared against the serial mpi4jax path.
# Run (from the repo root):
#   MPLBACKEND=Agg PYTHONPATH=.:tests python3 -c \
#     "import local_mpi_stub, runpy; runpy.run_path('tests/test_backend_jax.py', run_name='__main__')"
import os
os.environ.setdefault("RMHD_PRECISION", "64")
# must be set before the XLA backend is created (jax import alone doesn't create it)
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import jax_rmhd as jr
import jax_rmhd.snapshot_io as sn
from jax_rmhd import comms
from jax_rmhd.physics import shared_physics
from jax.sharding import PartitionSpec as P

all_ok = True
def check(msg, ok):
    global all_ok
    print(("[PASS] " if ok else "[FAIL] ") + msg)
    all_ok &= bool(ok)
    return ok

NDEV = jax.device_count()
print(f"devices: {NDEV} x {jax.devices()[0].platform}")
check(f"4 forced host devices available (got {NDEV})", NDEV == 4)

nx = ny = 16
nz = 16
kw = dict(nx=nx, ny=ny, Lx=2*np.pi, Ly=2*np.pi, nz=nz, Lz=2*np.pi,
          diss=(1e-4, 1e-4), hyper=2, cfl_safety=0.5, dims=3,
          forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(0.5, 0.5),
          forcing_tau=1.0, fshell=(1, 3), forcing_seed=7)
t_end = 0.5

def zero_ic(x, y, z):
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape, z.shape))

def run(backend):
    p = jr.Parameters(comm_backend=backend, **kw)
    kg = jr.setup_kgrids(p)
    st = jr.initialize(zero_ic, p)
    mngr = jr.snapshot_manager_setup(p, tempfile.mkdtemp(), nsnap=2)
    return p, kg, jr.simulate(st, kg, p, t_snap=10.0, t_end=t_end, mngr=mngr, save=False)

# --- A. dims=2 is rejected for the jax backend -----------------------------------------
try:
    jr.Parameters(nx=8, ny=8, Lx=1.0, Ly=1.0, diss=(0.0, 0.0), hyper=1, cfl_safety=0.5,
                  dims=2, comm_backend="jax")
    check("comm_backend='jax' + dims=2 raises", False)
except ValueError:
    check("comm_backend='jax' + dims=2 raises", True)

# --- B. same-seed run: serial mpi4jax reference vs the 4-device jax backend -------------
p_ref, kg_ref, end_ref = run("mpi4jax")
p_jax, kg_jax, end_jax = run("jax")

f_ref = np.asarray(end_ref.fields)
f_jax = np.asarray(end_jax.fields)
check(f"jax-backend fields have the global shape {f_ref.shape}", f_jax.shape == f_ref.shape)
check(f"fields sharded over {NDEV} devices along z",
      len(end_jax.fields.addressable_shards) == NDEV
      and end_jax.fields.addressable_shards[0].data.shape[1] == nz // NDEV)
rel = np.max(np.abs(f_jax - f_ref)) / np.max(np.abs(f_ref))
check(f"final fields agree with the serial reference (rel {rel:.2e} < 1e-14)", rel < 1e-14)
check(f"final time matches ({float(end_ref.t):.12f})", float(end_jax.t) == float(end_ref.t))

# forcing is replicated, never sharded: the O-U stream must be bit-identical
fs_diff = float(np.max(np.abs(np.asarray(end_jax.forcing_state) - np.asarray(end_ref.forcing_state))))
check(f"forcing_state bit-identical across backends (max|diff| {fs_diff:g})", fs_diff == 0.0)
check("forcing_state replicated (single global shard shape preserved)",
      np.shape(end_jax.forcing_state) == np.shape(end_ref.forcing_state))
sc_rel = float(np.max(np.abs(np.asarray(end_jax.forcing_scale) - np.asarray(end_ref.forcing_scale)))
               / np.max(np.abs(np.asarray(end_ref.forcing_scale))))
check(f"forcing_scale agrees (rel {sc_rel:.2e} < 1e-12)", sc_rel < 1e-12)

# energies (same normalization convention as diagnostics.perpspec / forcing_power)
def energy(fields):
    fk = jnp.asarray(fields)
    return 0.5 * float(shared_physics.perp_inner_product(fk[0], fk[0], kg_ref, p_ref)
                       + shared_physics.perp_inner_product(fk[1], fk[1], kg_ref, p_ref))
E_ref, E_jax = energy(f_ref), energy(f_jax)
check(f"energy matches: ref {E_ref:.12f} vs jax {E_jax:.12f}",
      abs(E_jax - E_ref) <= 1e-12 * abs(E_ref))
check(f"energy is finite and nonzero ({E_ref:.6f})", np.isfinite(E_ref) and E_ref > 0.0)

# --- C. halo orientation: z_derivatives under shard_map vs the serial stencil -----------
# a wrong ppermute direction would silently mirror the z stencil
rng = np.random.default_rng(0)
f_loc = jnp.asarray(rng.standard_normal((p_ref.nfields, nz, nx, ny//2+1))
                    + 1j*rng.standard_normal((p_ref.nfields, nz, nx, ny//2+1)))
dz_ref = shared_physics.z_derivatives(f_loc, p_ref)[0]
dz_fn = comms._shard_map(lambda f: shared_physics.z_derivatives(f, p_jax)[0],
                         (P(None, comms.Z_AXIS),), P(None, comms.Z_AXIS))
dz_jax = jax.jit(dz_fn)(comms.to_global(f_loc, p_jax, z_axis=1))
# tolerance, not equality: the reference runs op-by-op and the sharded one is jitted, so
# XLA fusion differs — a flipped ppermute direction shows up as an O(1) error, not 1e-15
dz_err = float(np.max(np.abs(np.asarray(dz_jax) - np.asarray(dz_ref))) / np.max(np.abs(np.asarray(dz_ref))))
check(f"z_derivatives across the ppermute halo match the serial stencil (rel {dz_err:.2e} < 1e-14)",
      dz_err < 1e-14)

# --- D. checkpoint roundtrip + cross-backend restart ------------------------------------
# The jax backend writes ONE shared global directory (snap_path/<step>/...) and hands orbax
# the z-sharded global jax.Arrays; layouts may differ per backend, cross-restartability
# both ways is what must hold.
snap_path = tempfile.mkdtemp()
p_ref.save(snap_path)
p_jax.save(snap_path)  # comm_backend must not count as a differing parameter
check("params.save accepts both backends in one run directory", True)

mngr = jr.snapshot_manager_setup(p_jax, snap_path, nsnap=2)
check("jax backend's manager is ONE shared directory (no per-rank subdir)",
      os.path.abspath(str(mngr.directory)) == os.path.abspath(snap_path))
sn.save_snapshot(0, end_jax, mngr, p_jax)
mngr.wait_until_finished()
check(f"jax-backend save produced the flat layout (got {sn.snapshot_layout(snap_path)!r})",
      sn.snapshot_layout(snap_path) == "flat")

back_jax = sn.load_snapshot(0, snap_path, p_jax)
rt = float(np.max(np.abs(np.asarray(back_jax.fields) - f_jax)))
check(f"jax-backend snapshot roundtrip exact (max|diff| {rt:g})", rt == 0.0)
check(f"restored fields re-sharded over {NDEV} devices along z",
      len(back_jax.fields.addressable_shards) == NDEV
      and back_jax.fields.addressable_shards[0].data.shape[1] == nz // NDEV)
check("restored global fields have the full global shape", back_jax.fields.shape == f_ref.shape)

# an mpi4jax process reads the jax backend's global directory
back_ref = sn.load_snapshot(0, snap_path, p_ref)
cross = float(np.max(np.abs(np.asarray(back_ref.fields) - f_jax)))
check(f"cross-backend restart (jax-written global dir read by mpi4jax) exact (max|diff| {cross:g})",
      cross == 0.0 and float(back_ref.t) == float(end_jax.t))

# ... and with z-slicing: an mpi4jax rank that owns only part of z. There is no MPI here, so
# rank/size are set directly on a throwaway Parameters — load_snapshot reads only those two.
p_half = jr.Parameters(comm_backend="mpi4jax", **kw)
p_half.size, p_half.rank = 2, 1
half = sn.load_snapshot(0, snap_path, p_half)
hz = nz // 2
h_err = float(np.max(np.abs(np.asarray(half.fields) - f_jax[:, hz:])))
check(f"mpi4jax rank 1/2 reads only its z-slice of the jax global dir (shape "
      f"{tuple(half.fields.shape)}, max|diff| {h_err:g})",
      half.fields.shape == (p_ref.nfields, hz, nx, ny//2+1) and h_err == 0.0)

# keys are stored as real key arrays in both layouts (orbax unwraps/rewraps typed PRNG keys).
# NB checked here, before section E donates back_jax's buffers.
kd = lambda k: np.asarray(jax.random.key_data(k))
is_key = lambda k: jax.dtypes.issubdtype(k.dtype, jax.dtypes.prng_key)
kd_saved = kd(end_jax.forcing_key)
check("forcing_key roundtrips as a key array under the jax backend",
      is_key(back_jax.forcing_key) and np.array_equal(kd(back_jax.forcing_key), kd_saved))
check("forcing_key survives the cross-backend read (mpi4jax reading a jax-written snapshot)",
      is_key(back_ref.forcing_key) and np.array_equal(kd(back_ref.forcing_key), kd_saved))

# --- E. restart continues correctly under the jax backend -------------------------------
mngr2 = jr.snapshot_manager_setup(p_jax, tempfile.mkdtemp(), nsnap=2)
cont_jax = jr.simulate(back_jax, kg_jax, p_jax, t_snap=10.0, t_end=t_end+0.2, mngr=mngr2, save=False)
mngr3 = jr.snapshot_manager_setup(p_ref, tempfile.mkdtemp(), nsnap=2)
cont_ref = jr.simulate(sn.load_snapshot(0, snap_path, p_ref), kg_ref, p_ref,
                       t_snap=10.0, t_end=t_end+0.2, mngr=mngr3, save=False)
c_rel = np.max(np.abs(np.asarray(cont_jax.fields) - np.asarray(cont_ref.fields))) \
        / np.max(np.abs(np.asarray(cont_ref.fields)))
check(f"continued run after restart agrees with the reference (rel {c_rel:.2e} < 1e-14)", c_rel < 1e-14)

# --- F. the other stepper paths: simulate_scan + cfl_every>1 + unrolled LSRK + per-stage
#        normalization + snapshot writing, again ref-vs-jax --------------------------------
alt = dict(kw, cfl_every=2, lsrk_scan=False, forcing_norm_per_step=False)
def run_scan(backend):
    p = jr.Parameters(comm_backend=backend, **alt)
    kg = jr.setup_kgrids(p)
    st = jr.initialize(zero_ic, p)
    nblock = jr.estimate_good_nblock(st, kg, p, t_snap=0.2, t_end=0.2, nblock_min=4)
    mngr = jr.snapshot_manager_setup(p, tempfile.mkdtemp(), nsnap=4)
    return p, jr.simulate_scan(st, kg, p, nblock=nblock, t_snap=0.1, t_end=0.2, mngr=mngr, save=True), nblock

p_sref, s_ref, nb_ref = run_scan("mpi4jax")
p_sjax, s_jax, nb_jax = run_scan("jax")
check(f"estimate_good_nblock agrees across backends ({nb_ref})", nb_ref == nb_jax)
s_rel = np.max(np.abs(np.asarray(s_jax.fields) - np.asarray(s_ref.fields))) \
        / np.max(np.abs(np.asarray(s_ref.fields)))
check(f"simulate_scan + cfl_every=2 + lsrk_scan=False + per-stage norm agree (rel {s_rel:.2e} < 1e-14)",
      s_rel < 1e-14 and float(s_jax.t) == float(s_ref.t))

# --- G. on-disk shape of the two layouts + pruning ----------------------------------------
# AMENDED (2026-07-26): the layouts may DIFFER between backends. What must hold is the leaf
# SET (one pytree structure), both cross-restores, and an untouched mpi4jax writer path.
import json
def leaf_meta(path, isnap=0):
    m = json.load(open(os.path.join(path, str(isnap), "default", "_METADATA")))["tree_metadata"]
    return sorted(m.keys()), {v["value_metadata"]["value_type"] for v in m.values()}

snap_ref = tempfile.mkdtemp()
mngr_r = jr.snapshot_manager_setup(p_ref, snap_ref, nsnap=2)
sn.save_snapshot(0, end_ref, mngr_r, p_ref)
mngr_r.wait_until_finished()
keys_ref, types_ref = leaf_meta(snap_ref)
keys_jax, types_jax = leaf_meta(snap_path)
check(f"on-disk leaf set identical across backends ({len(keys_ref)} leaves)", keys_ref == keys_jax)
check(f"both backends store jax.Array leaves (mpi4jax {types_ref} / jax {types_jax})",
      types_ref == {"jax.Array"} and types_jax == {"jax.Array"})

# one shared manager -> max_to_keep prunes once, globally
prune_dir = tempfile.mkdtemp()
mngr_p = jr.snapshot_manager_setup(p_jax, prune_dir, nsnap=2)
for i in range(4):
    sn.save_snapshot(i, end_jax, mngr_p, p_jax)
    mngr_p.wait_until_finished()
kept = sorted(sn.get_saved_steps(prune_dir, p_jax))
check(f"nsnap/max_to_keep honored under the jax backend (kept {kept})", kept == [2, 3])

back_x = sn.load_snapshot(0, snap_ref, p_jax)  # the other direction
xr = float(np.max(np.abs(np.asarray(back_x.fields) - f_ref)))
check(f"cross-backend restart (mpi4jax-written snapshot read by the jax backend) exact (max|diff| {xr:g})",
      xr == 0.0 and float(back_x.t) == float(end_ref.t) and is_key(back_x.forcing_key))

# --- G2. the jax backend reading a real mpi4jax PER-RANK tree (the Savio phase-4a case) ----
# Built with the genuine mpi4jax writer path by faking rank/size (no MPI here): two rank
# dirs each holding half of z, exactly what `mpirun -n 2` would write.
pr_dir = tempfile.mkdtemp()
p_w = jr.Parameters(comm_backend="mpi4jax", **kw)
p_w.size = 2
for r in range(2):
    p_w.rank = r
    m_w = jr.snapshot_manager_setup(p_w, pr_dir, nsnap=2)
    sn.save_snapshot(0, end_ref._replace(fields=jnp.asarray(f_ref[:, r*hz:(r+1)*hz])), m_w, p_w)
    m_w.wait_until_finished()
check(f"2-rank mpi4jax writer produced the per-rank layout (got {sn.snapshot_layout(pr_dir)!r})",
      sn.snapshot_layout(pr_dir) == "per_rank" and sorted(os.listdir(pr_dir)) == ["0", "1"])
pr_jax = sn.load_snapshot(0, pr_dir, p_jax)
pr_err = float(np.max(np.abs(np.asarray(pr_jax.fields) - f_ref)))
check(f"jax backend unions a 2-rank mpi4jax tree into global z-sharded arrays "
      f"(max|diff| {pr_err:g})",
      pr_err == 0.0 and len(pr_jax.fields.addressable_shards) == NDEV
      and float(pr_jax.t) == float(end_ref.t) and is_key(pr_jax.forcing_key))
pr_ref = sn.load_snapshot(0, pr_dir, p_ref)
check("... and the mpi4jax backend reads the same tree identically",
      float(np.max(np.abs(np.asarray(pr_ref.fields) - f_ref))) == 0.0)

# --- G3. layout detection on every layout, with a decoy stranded tmp dir -------------------
# snap_path/0 exists in BOTH layouts; the marker is _CHECKPOINT_METADATA / the item subdir
# directly inside it. Stranded "<step>.orbax-checkpoint-tmp" dirs (real Savio leftovers)
# must not confuse either the detector or get_saved_steps.
for decoy_root, decoy in ((snap_path, "0.orbax-checkpoint-tmp"), (pr_dir, "3.orbax-checkpoint-tmp"),
                          (os.path.join(pr_dir, "0"), "7.orbax-checkpoint-tmp")):
    os.makedirs(os.path.join(decoy_root, decoy), exist_ok=True)
    open(os.path.join(decoy_root, decoy, "_CHECKPOINT_METADATA"), "w").close()
empty_dir = tempfile.mkdtemp()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(sn.__file__)))
legacy_tree = os.path.join(REPO, "tests", "data", "forced_turbulence_64cubed")  # read-only
cases = [(snap_path, "flat", [0]), (pr_dir, "per_rank", [0]), (empty_dir, "empty", [])]
if os.path.isdir(legacy_tree):
    cases.append((legacy_tree, "per_rank", [0, 1, 2, 3, 4]))
for path, want, want_steps in cases:
    got, got_steps = sn.snapshot_layout(path), sorted(sn.get_saved_steps(path))
    check(f"layout of {os.path.basename(path) or path} is {want!r} with steps {want_steps} "
          f"(got {got!r}, {got_steps})", got == want and got_steps == want_steps)

# --- G4. mixing WRITERS of the two layouts in one directory is refused ---------------------
# (reading across layouts stays supported — that is what G2/D check)
def raises_valueerror(fn):
    try:
        fn()
        return False
    except ValueError:
        return True
p_w2 = jr.Parameters(comm_backend="mpi4jax", **kw)
p_w2.size, p_w2.rank = 2, 0
check("jax backend refuses to write into an existing mpi4jax per-rank tree",
      raises_valueerror(lambda: jr.snapshot_manager_setup(p_jax, pr_dir, nsnap=2)))
check("multi-rank mpi4jax refuses to write into an existing flat dir",
      raises_valueerror(lambda: jr.snapshot_manager_setup(p_w2, snap_path, nsnap=2)))
check("single-process mpi4jax may still continue a flat dir",
      not raises_valueerror(lambda: jr.snapshot_manager_setup(p_ref, snap_path, nsnap=2)))

# --- H. restores never follow the device recorded in the checkpoint ------------------------
# With jax.distributed up each process owns different device ids, so a template leaf WITHOUT
# a sharding makes orbax follow the checkpoint's recorded device ("Device cpu:0 was not found
# in jax.local_devices()"). Every process-local restore now pins this process's own device;
# hiding cpu:0 is the stand-in for "this process doesn't own the writer's device".
_real_local = jax.local_devices
try:
    jax.local_devices = lambda *a, **k: _real_local(*a, **k)[1:]
    pinned_ok = True
    for name, p_ in (("mpi4jax", p_ref), ("jax", p_jax)):
        try:
            again = sn.load_snapshot(0, pr_dir, p_)  # per-rank tree: process-local templates
            pinned_ok &= float(again.t) == float(end_ref.t)
        except Exception as e:  # noqa: BLE001 - reported as a failed check
            pinned_ok = False
            print(f"    {name} restore raised {type(e).__name__}: {str(e)[:160]}")
finally:
    jax.local_devices = _real_local
check("process-local restores are pinned to this process's device (cpu:0 hidden)", pinned_ok)

# --- I. a real PRE-Phase-3 (pre-forcing_scale) snapshot dir is still readable -------------
# tests/data/forced_turbulence_64cubed is the reference legacy layout, written long before
# either the jax backend or forcing_scale existed. Read-only: ONE step dir (~230 kB) is
# copied out and the copy is repaired, so tests/data is never touched.
import shutil
REPO = os.path.dirname(os.path.dirname(os.path.abspath(sn.__file__)))
legacy_src = os.path.join(REPO, "tests", "data", "forced_turbulence_64cubed", "0", "4")
if not os.path.isdir(legacy_src):
    print("[SKIP] legacy snapshot tests/data/forced_turbulence_64cubed not present")
else:
    legacy_dir = tempfile.mkdtemp()
    shutil.copytree(legacy_src, os.path.join(legacy_dir, "4"))
    # nz=8/nx=ny=64 reproduces that run's per-rank shard shape (2,8,64,33) in a 1-rank layout
    lkw = dict(kw, nx=64, ny=64, nz=8, forcing_seed=42)
    p_leg = jr.Parameters(comm_backend="mpi4jax", **lkw)
    try:
        sn.load_snapshot(4, legacy_dir, p_leg)
        pre_ok = False
    except ValueError as e:
        pre_ok = "old_snapshot_repair" in str(e)
    check("legacy snapshot still routed to old_snapshot_repair (structure check first)", pre_ok)
    sn.old_snapshot_repair(legacy_dir, p_leg)
    leg_ref = sn.load_snapshot(4, legacy_dir, p_leg)
    leg_jax = sn.load_snapshot(4, legacy_dir, jr.Parameters(comm_backend="jax", **lkw))
    ld = float(np.max(np.abs(np.asarray(leg_jax.fields) - np.asarray(leg_ref.fields))))
    check(f"repaired legacy snapshot reads identically under both backends (max|diff| {ld:g})",
          ld == 0.0 and float(leg_jax.t) == float(leg_ref.t)
          and np.all(np.isfinite(np.asarray(leg_ref.fields)))
          and float(np.max(np.abs(np.asarray(leg_ref.fields)))) > 0.0)
    check("... and its forcing_key comes back as a key array under the jax backend",
          is_key(leg_jax.forcing_key) and is_key(leg_ref.forcing_key))

print("\n" + ("ALL PASS" if all_ok else "SOME CHECKS FAILED"))
assert all_ok
