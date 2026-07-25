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
snap_path = tempfile.mkdtemp()
p_ref.save(snap_path)
p_jax.save(snap_path)  # comm_backend must not count as a differing parameter
check("params.save accepts both backends in one run directory", True)

mngr = jr.snapshot_manager_setup(p_jax, snap_path, nsnap=2)
sn.save_snapshot(0, end_jax, mngr, p_jax)
mngr.wait_until_finished()

back_jax = sn.load_snapshot(0, snap_path, p_jax)
rt = float(np.max(np.abs(np.asarray(back_jax.fields) - f_jax)))
check(f"jax-backend snapshot roundtrip exact (max|diff| {rt:g})", rt == 0.0)
check(f"restored fields re-sharded over {NDEV} devices",
      len(back_jax.fields.addressable_shards) == NDEV)

# the whole point of localizing before orbax: an mpi4jax process reads the same directory
back_ref = sn.load_snapshot(0, snap_path, p_ref)
cross = float(np.max(np.abs(np.asarray(back_ref.fields) - f_jax)))
check(f"cross-backend restart (jax-written snapshot read by mpi4jax) exact (max|diff| {cross:g})",
      cross == 0.0 and float(back_ref.t) == float(end_jax.t))

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

print("\n" + ("ALL PASS" if all_ok else "SOME CHECKS FAILED"))
assert all_ok
