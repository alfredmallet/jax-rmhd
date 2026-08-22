"""Generate reference test vectors for the WebGPU RMHD port.

Run at fp64 (TARANIS_PRECISION=64) so the reference is cleaner than the browser's fp32;
the browser self-test compares with fp32-appropriate tolerances.

Everything recorded is DETERMINISTIC given the recorded inputs: the one-step
comparison starts from a recorded (fields, forcing_state, forcing_scale) triple and
uses dt_override, so no RNG needs to be reproduced in the browser (jax threefry is
not portable). The OU update itself is validated statistically in-browser instead.
"""
import os, json
os.environ["TARANIS_PRECISION"] = "64"
import numpy as np
import jax.numpy as jnp

from taranis.config import Parameters
from taranis.grids import setup_kgrids
from taranis.run import initialize, _advance_forcing, _refresh_forcing_scale
from taranis.timestepping import get_scheme
from taranis.physics import equation_registry, construct_rhs
from taranis.physics import rmhd
from taranis.diagnostics import energy

NX = NY = 32
params = Parameters(nx=NX, ny=NY, Lx=2*np.pi, Ly=2*np.pi, cfl_safety=0.4,
                    eqpars={"diss": 0.01, "hyper": 1}, dims=2,
                    forcing=True, forcing_mode="elsasser",
                    forcing_power_elsasser=(0.5, 0.5), forcing_tau=1.0,
                    fshell=(1, 2), forcing_seed=7)
kgrid = setup_kgrids(params)
recipe = equation_registry["RMHD"]
rhs = construct_rhs(recipe)
stepper, scheme = get_scheme("lsrk33")
set_timestep = recipe.set_timestep_func


def ic(x, y):
    phi = 0.1*(jnp.sin(x)*jnp.cos(2*y) + jnp.cos(3*x)*jnp.sin(y))
    psi = 0.1*(jnp.cos(2*x)*jnp.cos(y) + jnp.sin(x)*jnp.sin(3*y))
    return jnp.stack([phi, psi])


def c2j(a):
    """complex array -> {re: nested list, im: nested list} (z axis squeezed)."""
    a = np.asarray(a)
    if a.ndim == 4:            # (nf, 1, nkx, nky) -> (nf, nkx, nky)
        a = a[:, 0]
    elif a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    return {"re": a.real.tolist(), "im": a.imag.tolist()}


def r2j(a):
    return np.asarray(a).tolist()


out = {"nx": NX, "ny": NY, "Lx": params.Lx, "Ly": params.Ly,
       "diss": 0.01, "hyper": 1, "cfl_safety": params.cfl_safety,
       "forcing_power_elsasser": [0.5, 0.5], "forcing_tau": 1.0,
       "fshell": [1, 2], "forcing_scale_max": params.forcing_scale_max}

# ---- FFT test vector: a deterministic real field and its (unnormalized) rfft2 ----
x = np.linspace(0, 2*np.pi, NX, endpoint=False).reshape(-1, 1)
y = np.linspace(0, 2*np.pi, NY, endpoint=False).reshape(1, -1)
ftest = np.sin(x)*np.cos(2*y) + 0.3*np.cos(3*x + y) + 0.05*np.sin(5*x)*np.sin(7*y)
out["fft_input"] = r2j(ftest)
out["fft_output"] = c2j(np.fft.rfft2(ftest))

# ---- static grids the browser must reproduce ----
out["dealias"] = np.asarray(kgrid.dealias, dtype=np.float64).tolist()
out["fmask"] = np.asarray(kgrid.fmask, dtype=np.float64).tolist()
out["lin_L"] = r2j(np.asarray(kgrid.lin.L[0, 0].real))   # -diss*ksq^hyper (diagonal, real)

# ---- initial condition (k-space, dealiased) ----
state = initialize(ic, params)
state = _refresh_forcing_scale(state, kgrid, params)
out["ic_fields_k"] = c2j(state.fields)

# ---- spin-up: 30 real steps so forcing_state/fields are nontrivial ----
dt_last = 0.0   # the dt _advance_forcing hands to forcing_scale (needed for A_scale_check)
for _ in range(30):
    prev_t = state.t
    state = stepper(state, kgrid, params, rhs, set_timestep, scheme)
    state = _advance_forcing(state, prev_t, kgrid, params)
    dt_last = float(state.t) - float(prev_t)

A = state
out["A_t"] = float(A.t)
out["A_fields_k"] = c2j(A.fields)
out["A_forcing_state"] = c2j(A.forcing_state[:, 0])   # (2, nkx, nky); 2D uses A-coef only
out["A_forcing_scale"] = r2j(A.forcing_scale)

# ---- granular diagnostics at A ----
grads = rmhd.grad(A, kgrid, params)
out["A_nonlinear_k"] = c2j(rmhd.NonlinearTerm(A, grads, kgrid, params))
out["A_forcing_term_k"] = c2j(rmhd.ForcingTerm(A, grads, kgrid, params))
# recompute from A. dt_last is the same step length _advance_forcing used above, so this
# still reproduces A_forcing_scale exactly (the normalization is dt-dependent since
# 2026-08-08 -- self-energy-aware quadratic; the app port is FORCING_SPINUP_PLAN Phase 3).
out["A_scale_check"] = r2j(rmhd.forcing_scale(A, kgrid, params, dt_last))
# ... and the browser has to put that same dt in sc[0] before dispatching `scale`
out["A_dt_last"] = dt_last
ek, em = energy(A, kgrid, params)
out["A_energy"] = [float(ek), float(em)]
dtA = float(set_timestep(grads, params))
out["A_dt"] = dtA

# ---- one deterministic LSRK33 step from A (no OU update afterwards) ----
B = stepper(A, kgrid, params, rhs, set_timestep, scheme, dt_override=dtA)
out["B_t"] = float(B.t)
out["B_fields_k"] = c2j(B.fields)

# next to this script, i.e. the copy the apps fetch
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refvectors.json")
with open(dst, "w") as f:
    json.dump(out, f)
print("wrote", dst, os.path.getsize(dst), "bytes")
print("A energy:", out["A_energy"], "dt:", dtA, "scale:", out["A_forcing_scale"])
