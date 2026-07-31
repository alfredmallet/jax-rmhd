import os
os.environ["RMHD_PRECISION"] = "64"
import jax
import jax.numpy as jnp
import jax_rmhd as jr
from jax_rmhd.physics import shared_physics

# ---------------------------------------------------------------------------
# Smoke tests for the O-U forcing implementation.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    return cond

all_ok = True

# --- A. Hermitian symmetry is preserved by ou_update across many steps -----
nx, ny, nz = 16, 16, 8
Lx = Ly = Lz = 2.0*jnp.pi
params = jr.Parameters(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, diss=(0.0, 0.0),
                        hyper=1, cfl_safety=0.5, dt=0.01, adaptive_timestep=False, dims=3,
                        forcing=True, forcing_mode="momentum", forcing_power=1.0,
                        forcing_tau=0.5, fshell=(1, 5), forcing_seed=1,
                        forcing_norm_per_step=False)  # this test checks exact per-stage normalization
kgrid = jr.setup_kgrids(params)

nkx, nky = nx, ny//2 + 1
forcing_state = jnp.zeros((params.n_ou, 2, nkx, nky), dtype=jnp.complex128)
forcing_key = jax.random.key(params.forcing_seed)

for _ in range(50):
    forcing_state, forcing_key = shared_physics.ou_update(forcing_state, forcing_key, 0.01, params, kgrid)

nkx_dim = forcing_state.shape[-2]
mirror_idx = (-jnp.arange(nkx_dim)) % nkx_dim
for ky_idx in (0, -1):
    col = forcing_state[..., ky_idx]
    mirror = jnp.conj(col[..., mirror_idx])
    err = float(jnp.max(jnp.abs(col - mirror)))
    all_ok &= check(f"Hermitian symmetry holds at ky index {ky_idx} after 50 ou_update steps",
                     err < 1e-10, f"max |A(kx)-conj(A(-kx))| = {err:.2e}")

# --- B. safe_scale: sign-correct where unclipped, capped at +-scale_max otherwise ---
target = 2.0
scale_max = 1.0  # default
for P_val in (3.0, -3.0):
    # |target/P| < scale_max here, so the cap shouldn't engage: exact match expected.
    scale = shared_physics.safe_scale(target, jnp.array(P_val), scale_max)
    result = float(scale) * P_val
    all_ok &= check(f"safe_scale: scale*P == target for P={P_val:+.0e} (uncapped regime)",
                     abs(result - target) < 1e-8, f"got {result}")
for P_val in (1e-40, -1e-40, 0.0):
    # |target/P| >> scale_max here (or exactly target/0), so the cap must engage:
    # scale should land exactly at +-scale_max with the sign of target/P, not blow up.
    scale = float(shared_physics.safe_scale(target, jnp.array(P_val), scale_max))
    expected_sign = 1.0 if P_val >= 0.0 else -1.0
    all_ok &= check(f"safe_scale: near-zero P={P_val:+.0e} is capped at +-scale_max, not blown up",
                     jnp.isfinite(scale) and abs(abs(scale) - scale_max) < 1e-12 and jnp.sign(scale) == expected_sign,
                     f"scale={scale}")

# --- C. ForcingTerm is an exact no-op when params.forcing is False ---------
params_off = jr.Parameters(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, diss=(0.0, 0.0),
                            hyper=1, cfl_safety=0.5, dt=0.01, adaptive_timestep=False, dims=3,
                            forcing=False)
def init_zero(x, y, z):
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape, z.shape))
state_off = jr.initialize(init_zero, params_off)
from jax_rmhd.physics import rmhd
grads_off = rmhd.grad(state_off, kgrid, params_off)
f_off = rmhd.ForcingTerm(state_off, grads_off, kgrid, params_off)
all_ok &= check("ForcingTerm returns exact zeros when params.forcing=False",
                bool(jnp.all(f_off == 0)))

# --- D. Power-injection sanity check (momentum mode, no dissipation) --
# checks kinetic energy is roughly what we expect
mngr = jr.snapshot_manager_setup(params, snap_path="data/test_forcing_smoke", nsnap=10)
state0 = jr.initialize(init_zero, params)
t_end = 0.5
nblock = 50
end_state = jr.simulate_scan(state0, kgrid, params, nblock, t_end, t_end, mngr, save=False)

phik = end_state.fields[0]
E_kin = 0.5 * float(shared_physics.perp_inner_product(phik, phik, kgrid, params))
rate = E_kin / float(end_state.t)
target = params.forcing_power
all_ok &= check("Kinetic energy injection rate is within 3x of forcing_power (loose bound)",
                target/3.0 < rate < target*3.0,
                f"measured rate={rate:.4f}, target={target}, t_end={float(end_state.t)}")

# --- E. Forcing works in dims=2 ---
params_2d = jr.Parameters(nx=nx, ny=ny, Lx=Lx, Ly=Ly, diss=(0.0, 0.0),
                           hyper=1, cfl_safety=0.5, dt=0.01, adaptive_timestep=False, dims=2,
                           forcing=True, forcing_mode="momentum", forcing_power=1.0,
                           forcing_tau=0.5, fshell=(1, 5), forcing_seed=1,
                           forcing_norm_per_step=False)  # this test checks exact per-stage normalization
kgrid_2d = jr.setup_kgrids(params_2d)

def init_zero_2d(x, y):
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape))

state0_2d = jr.initialize(init_zero_2d, params_2d)

all_ok &= check("2D fields have a singleton leading axis (shape is (nfields,1,nx,nky))",
                state0_2d.fields.shape == (2, 1, nx, ny//2+1), f"fields.shape={state0_2d.fields.shape}")

grads_2d = rmhd.grad(state0_2d, kgrid_2d, params_2d)
f_2d = rmhd.ForcingTerm(state0_2d, grads_2d, kgrid_2d, params_2d)
all_ok &= check("2D ForcingTerm output shape matches state.fields",
                f_2d.shape == state0_2d.fields.shape,
                f"f_2d.shape={f_2d.shape}, fields.shape={state0_2d.fields.shape}")

mngr_2d = jr.snapshot_manager_setup(params_2d, snap_path="data/test_forcing_smoke_2d", nsnap=10)
end_state_2d = jr.simulate_scan(state0_2d, kgrid_2d, params_2d, nblock, t_end, t_end, mngr_2d, save=False)
phik_2d = end_state_2d.fields[0]
E_kin_2d = 0.5 * float(shared_physics.perp_inner_product(phik_2d, phik_2d, kgrid_2d, params_2d))
rate_2d = E_kin_2d / float(end_state_2d.t)
all_ok &= check("2D kinetic energy injection rate is within 3x of forcing_power (loose bound)",
                target/3.0 < rate_2d < target*3.0,
                f"measured rate={rate_2d:.4f}, target={target}, t_end={float(end_state_2d.t)}")

print()
if params.size > 1:  # a single bad rank must fail the whole mpirun job
    all_ok = all(params.comm.allgather(bool(all_ok)))
print("ALL PASS" if all_ok else "SOME CHECKS FAILED -- see [FAIL] lines above")
assert all_ok, f"[rank {params.rank}] forcing smoke checks failed"
