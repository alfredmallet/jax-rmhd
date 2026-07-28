import os
os.environ["RMHD_PRECISION"] = "64"
import jax.numpy as jnp
import jax_rmhd as jr
from jax_rmhd import comms, grids, diagnostics
from jax_rmhd.physics.shared_physics import gradk, perp_inner_product_batch

# ---------------------------------------------------------------------------
# Independent real-space Parseval cross-check on the spectral energy path
# (shared_physics._perp_reduce / perp_inner_product_batch, which diagnostics.energy
# now wraps). This used to live inside diagnostics.energy itself; it moved here when
# energy became a thin wrapper over the spectral path -- a check is only worth having
# if it shares no code with the thing it checks. The real-space path below goes
# ifft -> pointwise square -> local sum -> allreduce, touching none of the _perp_reduce
# normalization machinery.
# Run under `mpirun -n N python tests/test_energy_parseval.py` (N=1,2,4 on Savio;
# no MPI in the dev sandbox).
# ---------------------------------------------------------------------------

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    return cond

def realspace_grad_meansq(fieldk, kgrid, params):
    # <|grad_perp f|^2>, volume-averaged over the GLOBAL domain: independent real-space
    # path (grad components summed explicitly; local z-slab sum, then allreduce).
    g = grids.ifft(gradk(fieldk, kgrid), params)  # (nz_local, 2, nx, ny) real space
    local_sum = jnp.sum(g**2)
    total = comms.allreduce_sum(local_sum, params)  # no-op unless z-decomposed
    return total / (params.nz * params.nx * params.ny)

all_ok = True

nx, ny, nz = 16, 16, 8
Lx = Ly = Lz = 2.0 * jnp.pi

params = jr.Parameters(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, diss=(0.01, 0.01),
                        hyper=1, cfl_safety=0.5, dt=0.01, adaptive_timestep=False, dims=3)
kgrid = jr.setup_kgrids(params)

def init_fields(x, y, z):
    # non-degenerate, multi-mode, O(1) amplitudes
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z) + 0.3 * jnp.sin(2*x + y)
    psi = jnp.sin(x) * jnp.cos(y) * jnp.sin(z) + 0.2 * jnp.cos(x - 2*y)
    return jnp.stack([phi, psi], axis=0)

state = jr.initialize(init_fields, params)
phik, psik = state.fields[0], state.fields[1]

# --- A. Parseval: spectral batch reduction == independent real-space reduction ---------
spec = perp_inner_product_batch(state.fields[:2], state.fields[:2], kgrid, params)
real_phi = realspace_grad_meansq(phik, kgrid, params)
real_psi = realspace_grad_meansq(psik, kgrid, params)

rtol = 1e-12  # same quantity, different fp path; fp64
all_ok &= check("Parseval: spectral <|grad phi|^2> matches real-space path",
                bool(jnp.allclose(spec[0], real_phi, rtol=rtol)),
                f"spectral={spec[0]:.15e}, real={real_phi:.15e}")
all_ok &= check("Parseval: spectral <|grad psi|^2> matches real-space path",
                bool(jnp.allclose(spec[1], real_psi, rtol=rtol)),
                f"spectral={spec[1]:.15e}, real={real_psi:.15e}")

# sanity: nonzero (a bug that zeros both sides would "pass" the equality above)
all_ok &= check("Parseval: energies are nonzero", bool(spec[0] > 0) and bool(spec[1] > 0))

# --- B. diagnostics.energy is exactly half the batch reduction (E = 0.5*<|grad|^2>) ----
E_kin, E_mag = diagnostics.energy(state, kgrid, params)
all_ok &= check("diagnostics.energy == 0.5 * perp_inner_product_batch",
                bool(jnp.array_equal(E_kin, 0.5*spec[0])) and bool(jnp.array_equal(E_mag, 0.5*spec[1])))

print()
print("ALL PASS" if all_ok else "SOME CHECKS FAILED -- see [FAIL] lines above")
