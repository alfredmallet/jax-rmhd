import os
os.environ["RMHD_PRECISION"] = "64"
import jax
import jax_rmhd as jr
import jax.numpy as jnp
import jax_rmhd.snapshot_io as sn
from jax_rmhd.physics import shared_physics

# ---------------------------------------------------------------------------
# Simple 3D forced-turbulence reference run at 64^3. Meant as a quick
# (few-minute), multi-rank, end-to-end smoke test: forcing + checkpointing +
# reload, on the actual cluster/MPI environment -- not a physics result to
# read anything into. Structure/logging mirrors tests/test_advection.py;
# forcing parameters match examples/forced-turbulence-3D.ipynb.
#
# NB: gate rank-0-only prints on params.rank (MPI rank), NOT
# jax.process_index(): under the mpi4jax backend every MPI rank is its own
# single-process JAX runtime -- parallelism here is handled entirely by MPI/mpi4jax, not
# by JAX's own multi-controller distributed system. jax.process_index() is
# therefore 0 on every rank and can't be used to distinguish them.
# ---------------------------------------------------------------------------

#grid + physical parameters
nx = ny = nz = 64
Lx = Ly = Lz = 2.0 * jnp.pi

nsnap = 20
t_snap = 0.25
t_end = 1.0
cfl_safety = 0.5
spatial_dimensions = 3
snap_path = "data/forced_turbulence_64cubed"

#hyperviscosity (same choice as examples/forced-turbulence-3D.ipynb at this resolution)
visc = 1e-5
res = 1e-5
hyper = 3

#forcing parameters (same choice as examples/forced-turbulence-3D.ipynb)
forcing = True
forcing_mode = "elsasser"
forcing_power_elsasser = (0.3, 0.3)   #target energy injection rate, one per Elsasser variable
forcing_tau = 1.0                      #O-U decorrelation time of the forcing pattern
fshell = (1, 3)                        #force the shell 1 <= |k_perp|/dk < 3 (large scales)
forcing_seed = 42                      #random seed
forcing_scale_max = 1.0                #caps the energy added per timestep

params = jr.Parameters(nx=nx, ny=ny, Lx=Lx, Ly=Ly, nz=nz, Lz=Lz, diss=(visc, res),
                        hyper=hyper, cfl_safety=cfl_safety, dims=spatial_dimensions,
                        forcing=forcing, forcing_mode=forcing_mode,
                        forcing_power_elsasser=forcing_power_elsasser,
                        forcing_tau=forcing_tau, fshell=fshell, forcing_seed=forcing_seed,
                        forcing_scale_max=forcing_scale_max)

is_control = (params.rank == 0)
if is_control:
    print(f"params.size={params.size} MPI rank(s), nz_local={nz // params.size}")

mngr = jr.snapshot_manager_setup(params=params, snap_path=snap_path, nsnap=nsnap)
kgrid = jr.setup_kgrids(params)

def zero_init(x, y, z):
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape, z.shape))

state = jr.initialize(zero_init, params)

nblock = jr.estimate_good_nblock(state, kgrid, params, t_snap, t_end, nblock_min=10)
if is_control:
    print("nblock estimate:", nblock)

end_state = jr.simulate_scan(state, kgrid, params, nblock, t_snap=t_snap, t_end=t_end,
                              mngr=mngr, save=True)

# ---------------------------------------------------------------------------
# Sanity checks -- run (and print) on every rank, not just rank 0, so an
# MPI-wide checkpointing problem shows up in every rank's log rather than
# being hidden behind whichever rank happens to be "in control".
# ---------------------------------------------------------------------------
phik, psik = end_state.fields[0], end_state.fields[1]
E_kin = 0.5 * float(shared_physics.perp_inner_product(phik, phik, kgrid, params))
E_mag = 0.5 * float(shared_physics.perp_inner_product(psik, psik, kgrid, params))
print(f"[rank {params.rank}] final t={float(end_state.t):.4f}  E_kin={E_kin:.4e}  E_mag={E_mag:.4e}")

last_isnap = max(sn.get_saved_steps(snap_path))
reloaded = sn.load_snapshot(last_isnap, snap_path, params)

t_diff = float(jnp.abs(reloaded.t - end_state.t))
fields_diff = float(jnp.max(jnp.abs(reloaded.fields - end_state.fields)))
forcing_state_diff = float(jnp.max(jnp.abs(reloaded.forcing_state - end_state.forcing_state)))
key_match = bool(jnp.array_equal(jax.random.key_data(reloaded.forcing_key),
                                  jax.random.key_data(end_state.forcing_key)))
ok = (t_diff < 1e-8) and (fields_diff < 1e-8) and (forcing_state_diff < 1e-8) and key_match

print(f"[rank {params.rank}] reload check (isnap={last_isnap}): "
      f"t_diff={t_diff:.2e} fields_diff={fields_diff:.2e} "
      f"forcing_state_diff={forcing_state_diff:.2e} key_match={key_match} "
      f"-> {'PASS' if ok else 'FAIL'}")
assert ok, f"[rank {params.rank}] reloaded snapshot does not match in-memory end_state"

if is_control:
    print("ALL PASS")
