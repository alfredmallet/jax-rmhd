import os
os.environ["RMHD_PRECISION"] = "64"
import jax
import jax.numpy as jnp
import jax_rmhd as jr
from jax_rmhd import comms
from jax_rmhd.physics import shared_physics, rmhd

# ---------------------------------------------------------------------------
# Standalone checks for the parameterized-width z-halo exchange (comms.halo_exchange).
# Run under `mpirun -n N python tests/test_halo_width.py` for N in 1, 2, 4 (no MPI in
# the dev sandbox -- this is a Savio-only correctness run). Keep it cheap: small grid,
# few steps.
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    return cond

all_ok = True

nx, ny, nz = 16, 16, 16  # nz=16 keeps nz_local >= 3 (the width=3 case below) up to -n 4
Lx = Ly = Lz = 2.0 * jnp.pi

params = jr.Parameters(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, diss=(0.01, 0.01),
                        hyper=1, cfl_safety=0.5, dt=0.01, adaptive_timestep=False, dims=3)
kgrid = jr.setup_kgrids(params)

def init_fields(x, y, z):
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
    psi = jnp.sin(x) * jnp.cos(y) * jnp.sin(z)
    return jnp.stack([phi, psi], axis=0)

state0 = jr.initialize(init_fields, params)
fields_shape0 = state0.fields.shape

# NB: the donating smoke run (simulate_scan, donate_argnums=(0,)) is Section D, LAST --
# every section that touches state0's buffers (B, C) must run before it, or they hit
# "Array has been deleted" (see the buffer-donation section of CLAUDE.md).

# --- A. z_derivatives is bitwise-identical whether fed a pre-issued width-2 halo, a
#        pre-issued (wider-than-needed) width-3 halo, or no halo at all (internal fallback,
#        which itself requests width=2). This exercises option 3(a) from the plan: extra
#        halo planes must be sliced away exactly, not just "close". -----------------------
f = state0.fields  # (nfields, nz_local, nkx, nky) spectral, from a real (non-degenerate) IC

halo2 = comms.halo_exchange(f, params, width=2)
halo3 = comms.halo_exchange(f, params, width=3)

df_dz_2, d4f_dz4_2 = shared_physics.z_derivatives(f, params, halo=halo2)
df_dz_3, d4f_dz4_3 = shared_physics.z_derivatives(f, params, halo=halo3)
df_dz_none, d4f_dz4_none = shared_physics.z_derivatives(f, params, halo=None)

all_ok &= check("z_derivatives: df/dz bitwise-equal for pre-issued width=2 vs width=3 halo",
                bool(jnp.array_equal(df_dz_2, df_dz_3)))
all_ok &= check("z_derivatives: d4f/dz4 bitwise-equal for pre-issued width=2 vs width=3 halo",
                bool(jnp.array_equal(d4f_dz4_2, d4f_dz4_3)))
all_ok &= check("z_derivatives: df/dz bitwise-equal for pre-issued width=2 vs no-halo fallback",
                bool(jnp.array_equal(df_dz_2, df_dz_none)))
all_ok &= check("z_derivatives: d4f/dz4 bitwise-equal for pre-issued width=2 vs no-halo fallback",
                bool(jnp.array_equal(d4f_dz4_2, d4f_dz4_none)))

# --- B. rmhd.halo_start (width=2) feeds z_derivatives identically to a bare width=2
#        halo_exchange call -- the one coupling the plan calls out explicitly. -----------
halo_start_result = rmhd.halo_start(state0, kgrid, params)
df_dz_hs, d4f_dz4_hs = shared_physics.z_derivatives(f, params, halo=halo_start_result)
all_ok &= check("rmhd.halo_start width=2 output matches bare width=2 halo_exchange in z_derivatives",
                bool(jnp.array_equal(df_dz_hs, df_dz_2)) and bool(jnp.array_equal(d4f_dz4_hs, d4f_dz4_2)))

# --- C. width > nz_local raises ValueError, not a silently-wrong short exchange. --------
nz_local = f.shape[1]
raised = False
try:
    comms.halo_exchange(f, params, width=nz_local + 1)
except ValueError:
    raised = True
all_ok &= check(f"halo_exchange: width={nz_local + 1} > nz_local={nz_local} raises ValueError",
                raised)

# width < 1 also guarded
raised_low = False
try:
    comms.halo_exchange(f, params, width=0)
except ValueError:
    raised_low = True
all_ok &= check("halo_exchange: width=0 raises ValueError", raised_low)

# --- D. Default path (width=2 everywhere, unchanged): a few steps stay finite and the
#        right shape -- confirms nothing broke structurally in the refactor. LAST because
#        simulate_scan donates state0's buffers (see note at top). -----------------------
mngr = jr.snapshot_manager_setup(params, snap_path="data/test_halo_width", nsnap=5)
nblock = 20
t_end = 0.05
end_state = jr.simulate_scan(state0, kgrid, params, nblock, t_end, t_end, mngr, save=False)

all_ok &= check("default path: output fields shape unchanged",
                end_state.fields.shape == fields_shape0,
                f"{end_state.fields.shape} vs {fields_shape0}")
all_ok &= check("default path: output fields finite after a few steps",
                bool(jnp.all(jnp.isfinite(end_state.fields))))

print()
print("ALL PASS" if all_ok else "SOME CHECKS FAILED -- see [FAIL] lines above")
