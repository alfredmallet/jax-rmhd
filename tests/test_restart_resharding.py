import os
os.environ["RMHD_PRECISION"] = "64"
import jax
import jax.numpy as jnp
import jax_rmhd as jr
import jax_rmhd.snapshot_io as sn
import orbax.checkpoint as ocp
from jax_rmhd.types import SimulationState

# ---------------------------------------------------------------------------
# Regression test for cross-rank snapshot-numbering desynchronization when
# restarting on more ranks than a run was saved with (the 32->64 incident:
# pre-existing rank dirs resumed numbering from their old latest step while
# brand-new rank dirs started from 0, and max_to_keep pruning then left the
# two groups holding different snapshot windows). Fixed in run.py by
# broadcasting rank 0's starting snapshot index.
#
# Usage (two phases, same snap_path, shared filesystem):
#   mpirun -n 2 python tests/test_restart_resharding.py   # phase A: fresh run, save
#   mpirun -n 4 python tests/test_restart_resharding.py   # phase B: restart + checks
# To rerun from scratch: rm -rf data/test_restart_resharding
#
# The phase is auto-detected from whether snap_path already holds snapshots.
# nsnap is chosen small enough that phase B triggers max_to_keep pruning --
# the checks verify every rank dir ends up with the *identical* pruned window.
# Before the fix, phase B fails check 1 (ranks 2,3 number from 0).
# ---------------------------------------------------------------------------

nx = ny = nz = 16
Lx = Ly = Lz = 2.0 * jnp.pi
snap_path = "data/test_restart_resharding"

nsnap = 6          # small max_to_keep so phase B exercises pruning
t_snap = 0.1
t_end_A = 0.4
t_end_B = 1.0

params = jr.Parameters(nx=nx, ny=ny, Lx=Lx, Ly=Ly, nz=nz, Lz=Lz,
                       eqpars={"diss": (1e-5, 1e-5), "hyper": 3}, cfl_safety=0.5, dims=3,
                       forcing=True, forcing_mode="elsasser",
                       forcing_power_elsasser=(0.3, 0.3), forcing_tau=1.0,
                       fshell=(1, 3), forcing_seed=42, forcing_scale_max=1.0)

is_control = (params.rank == 0)

# Phase detection BEFORE snapshot_manager_setup (which creates empty rank dirs).
prior_steps = sn.get_saved_steps(snap_path) if os.path.isdir(snap_path) else []
restarting = len(prior_steps) > 0

mngr = jr.snapshot_manager_setup(params=params, snap_path=snap_path, nsnap=nsnap)
kgrid = jr.setup_kgrids(params)

if not restarting:
    if is_control:
        print(f"[phase A] fresh start on {params.size} ranks")
    state = jr.initialize(lambda x, y, z: jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape, z.shape)), params)
    t_end = t_end_A
else:
    last = max(prior_steps)
    if is_control:
        print(f"[phase B] restarting on {params.size} ranks from snapshot {last} (saved steps: {prior_steps})")
    state = sn.load_snapshot(last, snap_path, params)
    t_end = t_end_B

end_state = jr.simulate(state, kgrid, params, t_snap=t_snap, t_end=t_end, mngr=mngr, save=True)

if not restarting:
    if is_control:
        print("[phase A] done. Now rerun with more ranks, e.g.: mpirun -n 4 python tests/test_restart_resharding.py")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Phase B checks
# ---------------------------------------------------------------------------
mngr.wait_until_finished()
params.comm.Barrier()
ok = True

# Check 1: snapshot numbering is synchronized across rank dirs. Ranks that
# existed before the restart may retain OLDER snapshots that the new ranks
# never wrote (pruning trims each dir independently) -- that's benign. The
# invariant is: (a) every rank dir has the same LATEST step, and (b) from the
# newest "oldest step" over all dirs upward, all dirs agree exactly. The
# original bug fails (a): pre-existing dirs ended on a higher index than
# fresh ones (e.g. 20-39 vs 16-35).
if is_control:
    per_rank = []
    for r in range(params.size):
        m = ocp.CheckpointManager(os.path.abspath(os.path.join(snap_path, str(r))),
                                  options=ocp.CheckpointManagerOptions())
        per_rank.append(sorted(m.all_steps()))
    same_latest = len({s[-1] for s in per_rank}) == 1
    floor = max(s[0] for s in per_rank)  # newest "oldest step" over all dirs
    common = [x for x in per_rank[0] if x >= floor]
    same_window = all([x for x in s if x >= floor] == common for s in per_rank)
    match = same_latest and same_window
    print(f"[check 1] snapshot indices per rank dir: {per_rank}")
    print(f"[check 1] same latest={same_latest}, common window {common} identical={same_window} -> {'PASS' if match else 'FAIL'}")
    ok &= match
    common_steps = common
else:
    common_steps = None
ok = params.comm.bcast(ok, root=0)
common_steps = params.comm.bcast(common_steps, root=0)

# Check 2: every common snapshot holds the same simulation time on every rank.
last = max(mngr.all_steps())
ftype, ctype = sn.get_precision_types()
nz_local = params.nz // params.size
state_like = SimulationState(
    t=jax.ShapeDtypeStruct((), ftype),
    fields=jax.ShapeDtypeStruct((params.nfields, nz_local, params.nx, params.ny // 2 + 1), ctype),
    forcing_state=jax.ShapeDtypeStruct((params.n_ou, 2, params.nx, params.ny // 2 + 1), ctype),
    forcing_key=jax.ShapeDtypeStruct((), sn.get_key_dtype()),
    forcing_scale=jax.ShapeDtypeStruct((params.n_ou,), ftype))
t_match = True
for step in common_steps:
    own = mngr.restore(step, args=ocp.args.StandardRestore(state_like))
    ts_all = params.comm.allgather(float(own.t))
    step_match = (max(ts_all) - min(ts_all)) < 1e-12
    t_match &= step_match
    if is_control:
        print(f"[check 2] t of snapshot {step} across ranks: min={min(ts_all):.6f} max={max(ts_all):.6f} -> {'PASS' if step_match else 'FAIL'}")
ok &= t_match

# Check 3: load_snapshot roundtrip of the latest snapshot matches end_state.
reloaded = sn.load_snapshot(last, snap_path, params)
t_diff = float(jnp.abs(reloaded.t - end_state.t))
f_diff = float(jnp.max(jnp.abs(reloaded.fields - end_state.fields)))
rt = (t_diff < 1e-8) and (f_diff < 1e-8)
print(f"[check 3][rank {params.rank}] roundtrip isnap={last}: t_diff={t_diff:.2e} fields_diff={f_diff:.2e} -> {'PASS' if rt else 'FAIL'}")
ok &= rt

ok = all(params.comm.allgather(bool(ok)))
assert ok, f"[rank {params.rank}] restart-resharding test FAILED"
if is_control:
    print("ALL PASS")
