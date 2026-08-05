"""Reference vectors for the WebGPU 3D (z_spectral) RMHD port. fp64. Deterministic:
the one-step test starts from recorded (fields, forcing_state, scale) with dt_override;
the OU/RNG path is validated statistically in-browser. See webgpu/SPEC3D.md section 6."""
import os, json
os.environ["RMHD_PRECISION"] = "64"
import numpy as np
import jax
import jax.numpy as jnp

from jax_rmhd.config import Parameters
from jax_rmhd.grids import setup_kgrids
from jax_rmhd.run import initialize, _advance_forcing, _refresh_forcing_scale
from jax_rmhd.timestepping import get_scheme
from jax_rmhd.physics import equation_registry, construct_rhs
from jax_rmhd.physics import rmhd, shared_physics
from jax_rmhd.propagators import get_propagator
from jax_rmhd.diagnostics import energy

NX = NY = 16
NZ = 8
params = Parameters(nx=NX, ny=NY, Lx=2*np.pi, Ly=2*np.pi, cfl_safety=0.4,
                    dims=3, nz=NZ, Lz=2*np.pi, z_spectral=True,
                    eqpars={"diss": 0.01, "hyper": 1, "z_diss_k": 1e-3},
                    forcing=True, forcing_mode="elsasser",
                    forcing_power_elsasser=(0.5, 0.5), forcing_tau=1.0,
                    fshell=(1, 2), forcing_seed=7)
kgrid = setup_kgrids(params)
recipe = equation_registry["RMHD"]
rhs = construct_rhs(recipe)
stepper, scheme = get_scheme("lsrk33")
set_timestep = recipe.set_timestep_func


def ic(x, y, z):
    phi = 0.1*(jnp.sin(x)*jnp.cos(2*y)*jnp.cos(z) + jnp.cos(3*x)*jnp.sin(y))
    psi = 0.1*(jnp.cos(2*x)*jnp.cos(y)*jnp.sin(z) + jnp.sin(x)*jnp.sin(3*y))
    return jnp.stack([phi, psi])


def c2j(a):
    a = np.asarray(a)
    return {"re": a.real.tolist(), "im": a.imag.tolist()}


def r2j(a):
    return np.asarray(a, dtype=np.float64).tolist()


out = {"nx": NX, "ny": NY, "nz": NZ, "Lx": params.Lx, "Ly": params.Ly, "Lz": params.Lz,
       "diss": 0.01, "hyper": 1, "z_diss_k": 1e-3, "cfl_safety": params.cfl_safety,
       "forcing_power_elsasser": [0.5, 0.5], "forcing_tau": 1.0, "fshell": [1, 2],
       "forcing_scale_max": params.forcing_scale_max}

# ---- FFT vector: real (z,x,y) field and its unnormalized rfftn ----
x = np.linspace(0, 2*np.pi, NX, endpoint=False).reshape(1, -1, 1)
y = np.linspace(0, 2*np.pi, NY, endpoint=False).reshape(1, 1, -1)
z = np.linspace(0, 2*np.pi, NZ, endpoint=False).reshape(-1, 1, 1)
ftest = (np.sin(x)*np.cos(2*y)*np.cos(z) + 0.3*np.cos(3*x + y)*np.sin(2*z)
         + 0.05*np.sin(5*x)*np.sin(7*y + z))
out["fft_input"] = r2j(ftest)
out["fft_output"] = c2j(np.fft.rfftn(ftest, axes=(0, 1, 2)))

# ---- static arrays ----
out["dealias"] = r2j(np.asarray(kgrid.dealias, dtype=np.float64))     # (nz,nkx,nky)
out["fmask"] = r2j(np.asarray(kgrid.fmask, dtype=np.float64))         # (nkx,nky)
out["kzd"] = r2j(np.asarray(rmhd._kz_deriv(kgrid, params)).reshape(-1))

# ---- state A: init + spin-up ----
state = initialize(ic, params)
state = _refresh_forcing_scale(state, kgrid, params)
for _ in range(30):
    prev_t = state.t
    state = stepper(state, kgrid, params, rhs, set_timestep, scheme)
    state = _advance_forcing(state, prev_t, kgrid, params)
A = state
out["A_t"] = float(A.t)
out["A_fields_k"] = c2j(A.fields)                     # (2, nz, nkx, nky)
out["A_forcing_A"] = c2j(A.forcing_state[:, 0])       # (2, nkx, nky)
out["A_forcing_B"] = c2j(A.forcing_state[:, 1])
out["A_forcing_scale"] = r2j(A.forcing_scale)

# ---- propagator vector: one exp(L*tau) application ----
prop = get_propagator(kgrid, params)
tau = 0.1
out["prop_in"] = out["A_fields_k"]
out["prop_tau"] = tau
out["prop_out"] = c2j(prop.apply_exp(A.fields, tau))

# ---- granular terms at A ----
grads = rmhd.grad(A, kgrid, params)
out["A_envelope_k"] = c2j(shared_physics.reconstruct_envelope(A.forcing_state, kgrid, params))
out["A_nonlinear_k"] = c2j(rmhd.NonlinearTerm(A, grads, kgrid, params))
out["A_forcing_term_k"] = c2j(rmhd.ForcingTerm(A, grads, kgrid, params))
out["A_scale_check"] = r2j(rmhd.forcing_scale(A, kgrid, params))
ek, em = energy(A, kgrid, params)
out["A_energy"] = [float(ek), float(em)]
dtA = float(set_timestep(grads, params))
out["A_dt"] = dtA

# ---- one LSRK33 step, recorded scales, no OU ----
B = stepper(A, kgrid, params, rhs, set_timestep, scheme, dt_override=dtA)
out["B_t"] = float(B.t)
out["B_fields_k"] = c2j(B.fields)

dst = "/sessions/relaxed-determined-brahmagupta/mnt/outputs/refvectors3d.json"
with open(dst, "w") as f:
    json.dump(out, f)
print("wrote", dst, os.path.getsize(dst), "bytes")
print("A energy:", out["A_energy"], "dt:", dtA, "scale:", out["A_forcing_scale"])
