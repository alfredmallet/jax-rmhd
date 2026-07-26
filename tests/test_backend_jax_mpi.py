# Real-MPI (multi-GPU) correctness driver for comm_backend="jax" (T9): runs the SAME
# same-seed forced 3D problem under either backend, dumps each rank's LOCAL fields, and
# diffs two phases. Also does cross-backend restarts. The two backends write DIFFERENT
# on-disk layouts (mpi4jax: snap_path/<rank>/<step>; jax: one shared snap_path/<step> of
# global z-sharded arrays) — what these phases prove is that each reads the other's.
#
# Usage (see slurms/test_backend_jax_gpu.sh):
#   mpirun -n P python tests/test_backend_jax_mpi.py <mpi4jax|jax> <outdir> [restart_from]
#   python tests/test_backend_jax_mpi.py --compare <dirA> <dirB>      # single process
import sys, os, shutil

# Select the package version via RMHD_PKG=<dir> (same mechanism as bench/bench_phase1.py):
# an editable install's meta-path finder otherwise silently beats PYTHONPATH.
_pkgdir = os.environ.get("RMHD_PKG")
if _pkgdir:
    sys.meta_path = [f for f in sys.meta_path
                     if "jax_rmhd" not in (getattr(f, "__module__", "") or "")]
    sys.path.insert(0, _pkgdir)

import numpy as np

def compare(dir_a, dir_b):
    # Diff the per-rank field dumps of two phases; tolerance follows the precision.
    tol = float(os.environ.get("RMHD_CMP_TOL",
                               1e-12 if os.environ.get("RMHD_PRECISION", "32") == "64" else 1e-4))
    ok = True
    ranks = sorted(int(f.split("rank")[1].split(".")[0])
                   for f in os.listdir(dir_a) if f.startswith("fields_rank"))
    for r in ranks:
        a = np.load(os.path.join(dir_a, f"fields_rank{r}.npy"))
        b = np.load(os.path.join(dir_b, f"fields_rank{r}.npy"))
        rel = np.max(np.abs(a - b)) / np.max(np.abs(a))
        # amax>0 guards against a vacuous comparison of two all-zero runs (too few steps)
        good = bool(np.isfinite(rel) and rel < tol and np.max(np.abs(a)) > 0.0)
        ok &= good
        print(f"[compare] rank {r}: rel {rel:.3e} (tol {tol:.0e}) -> {'PASS' if good else 'FAIL'}")
    ta = float(np.load(os.path.join(dir_a, "t.npy")))
    tb = float(np.load(os.path.join(dir_b, "t.npy")))
    print(f"[compare] final t: {ta!r} vs {tb!r} -> {'PASS' if ta == tb else 'FAIL'}")
    ok &= (ta == tb)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok

if sys.argv[1] == "--compare":
    raise SystemExit(0 if compare(sys.argv[2], sys.argv[3]) else 1)

backend, outdir = sys.argv[1], sys.argv[2]
restart_from = sys.argv[3] if len(sys.argv) > 3 else None

import jax
import jax.numpy as jnp
import jax_rmhd as jr
import jax_rmhd.snapshot_io as sn
from jax_rmhd import comms
from jax_rmhd.physics import shared_physics

nx = ny = int(os.environ.get("RMHD_NX", 64))
nz = int(os.environ.get("RMHD_NZ", 32))
t_end = float(os.environ.get("RMHD_TEND", 1.0))  # long enough that forcing has spun the fields up

# NB Parameters() brings up jax.distributed for the "jax" backend, so nothing above may
# touch jax devices — the device print below deliberately comes after this line.
params = jr.Parameters(nx=nx, ny=ny, Lx=2*np.pi, Ly=2*np.pi, nz=nz, Lz=2*np.pi,
                       diss=(1e-4, 1e-4), hyper=2, cfl_safety=0.5, dims=3,
                       comm_backend=backend, forcing=True, forcing_mode="elsasser",
                       forcing_power_elsasser=(0.5, 0.5), forcing_tau=1.0,
                       fshell=(1, 3), forcing_seed=7)
print(f"pkg={_pkgdir or 'default'} backend={backend} rank={params.rank}/{params.size} "
      f"platform={jax.devices()[0].platform} local_devices={jax.local_devices()} "
      f"global_devices={jax.device_count()} precision={os.environ.get('RMHD_PRECISION','32')}",
      flush=True)

kgrid = jr.setup_kgrids(params)
if restart_from is None:
    state = jr.initialize(lambda x, y, z: jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape, z.shape)),
                          params)
else:
    # params: scopes the reader's CheckpointManager to this process under comm_backend="jax"
    last = max(sn.get_saved_steps(restart_from, params))
    if params.rank == 0:
        print(f"restarting from {restart_from} snapshot {last}", flush=True)
    state = sn.load_snapshot(last, restart_from, params)
    t_end = float(state.t) + t_end

# Start from a clean output dir: a previous aborted run can leave stranded
# "<step>.orbax-checkpoint-tmp" dirs (and snapshots in an older on-disk format) that would
# make this phase's save collide. Rank 0 wipes, then everyone waits.
_same = restart_from is not None and os.path.realpath(outdir) == os.path.realpath(restart_from)
if params.rank == 0:
    if not _same:  # never wipe the directory we just restarted from
        shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)
if params.size > 1:
    params.comm.Barrier()
mngr = jr.snapshot_manager_setup(params, outdir, nsnap=2)
end = jr.simulate(state, kgrid, params, t_snap=10.0, t_end=t_end, mngr=mngr, save=True)
if params.rank == 0:
    print(f"[{backend}] on-disk layout of {outdir}: {sn.snapshot_layout(outdir)!r} "
          f"steps={sn.get_saved_steps(outdir)}", flush=True)

# per-rank LOCAL fields, identical layout under both backends
local_fields = np.asarray(comms.to_local(end.fields, params, z_axis=1) if backend == "jax" else end.fields)
np.save(os.path.join(outdir, f"fields_rank{params.rank}.npy"), local_fields)

# energy on the host (same normalization as diagnostics.perpspec / forcing_power), so the
# number is computed identically for both backends without needing a shard_map context
ksq = np.asarray(kgrid.ksq())
yfac = np.asarray(shared_physics._perp_yfac(kgrid))
E_local = float(np.sum(ksq * np.abs(local_fields)**2 * yfac))
E_total = params.comm.allreduce(E_local) if params.size > 1 else E_local
E = 0.5 * E_total / (params.nz * float(params.nx*params.ny)**2)
if params.rank == 0:
    np.save(os.path.join(outdir, "t.npy"), np.asarray(float(end.t)))
    print(f"[{backend}] done: t={float(end.t)!r} energy={E:.12f}", flush=True)
