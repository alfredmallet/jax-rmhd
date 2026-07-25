import jax
from jax import jit
import jax.numpy as jnp
from .types import SimulationState
from typing import NamedTuple,Tuple

# Standard RK4 with integrating factor.
# Problem is it uses a lot of memory on k1-k4.
# I think it is always better to use the LSRK schemes

def rk_advance(state,kgrid,params,rhs,set_timestep,scheme=None,dt_override=None):
    #print("---COMPILING rk_advance---") #add this back in to check that jit is working properly
    #RK4 substep 1
    k1, grads = rhs(state,kgrid,params)
    # dt_override: dt already computed for a whole cfl_every block by run.py (skips the
    # per-step CFL allreduce); None means the historical per-step behavior below.
    if dt_override is not None:
        dt = dt_override
    elif params.adaptive_timestep==True:
        dt = set_timestep(grads,params)
    else:
        dt = params.dt
    #dissipation factors 
    diss_full = jnp.exp(kgrid.hdiss_exponents(params)*dt)
    diss_half = jnp.exp(kgrid.hdiss_exponents(params)*dt/2)
    f1 = diss_half * (state.fields + 0.5 * dt * k1)
    #RK4 substep 2
    # NB: forcing_state/forcing_key are threaded through unchanged at every sub-stage via
    # _replace (they're only updated once per full step, in run.block_of_steps).
    k2,_ = rhs(state._replace(t=state.t+dt/2.0,fields=f1),kgrid,params)
    f2 = diss_half * state.fields + 0.5*dt*k2
    #RK4 substep 3
    k3,_ = rhs(state._replace(t=state.t+dt/2.0,fields=f2),kgrid,params)
    f3 = diss_full * state.fields + dt * diss_half * k3
    #RK4 final step
    k4,_ = rhs(state._replace(t=state.t+dt,fields=f3),kgrid,params)
    f_end = diss_full * state.fields + (dt/6.0) * (diss_full * k1 + 2.0*diss_half*k2+2.0*diss_half*k3+k4)
    return state._replace(t=state.t + dt, fields=f_end)

# object defining low-storage Runge-Kutta (lsrk) schemes
class LSRK_Scheme(NamedTuple):
    alphas: Tuple[float,...]
    betas: Tuple[float,...]
    gammas: Tuple[float,...]
    nstages: int

# LSRK timestepper: includes an integrating factor for the dissipative terms.
# Two stage-loop structures selected by params.lsrk_scan: lax.scan (default — measured
# ~20% faster on CPU) or statically unrolled (lsrk_scan=False; candidate for GPU, where
# constant folding / freeing init_rhs after stage 0 / no cond should help — benchmark in
# Phase 3). Both give bitwise-identical trajectories at fp64.
def lsrk_advance(state, kgrid, params, rhs, set_timestep, scheme, dt_override=None):
    init_rhs,grads = rhs(state,kgrid,params)
    # dt_override: block-frozen dt from run.py (params.cfl_every>1); None = per-step CFL as before.
    if dt_override is not None:
        dt = dt_override
    elif params.adaptive_timestep==True:
        dt = set_timestep(grads,params)
    else:
        dt = params.dt

    diss_exponents = kgrid.hdiss_exponents(params) * dt

    if params.lsrk_scan:
        return _lsrk_scan_stages(state, kgrid, params, rhs, scheme, init_rhs, dt, diss_exponents)

    current_state = state
    delta = None
    for istage in range(scheme.nstages):
        alpha, beta, gamma = scheme.alphas[istage], scheme.betas[istage], scheme.gammas[istage]
        # stage 0 reuses init_rhs (also used for dt above); alphas[0]=0 so delta starts at dt*rhs
        stage_rhs = init_rhs if istage == 0 else rhs(current_state,kgrid,params)[0]
        diss_factors = jnp.exp(diss_exponents*gamma)
        delta = diss_factors * (dt*stage_rhs if istage == 0 else alpha*delta + dt*stage_rhs)
        # forcing fields threaded through unchanged via _replace (see rk_advance comment above).
        current_state = current_state._replace(t=current_state.t + gamma*dt,
                                               fields=diss_factors*current_state.fields + beta*delta)
    return current_state

# Original lax.scan-over-stages variant of the LSRK stage loop (params.lsrk_scan=True).
def _lsrk_scan_stages(state, kgrid, params, rhs, scheme, init_rhs, dt, diss_exponents):
    alphas_arr = jnp.array(scheme.alphas)
    betas_arr = jnp.array(scheme.betas)
    gammas_arr = jnp.array(scheme.gammas)

    init_delta = jnp.zeros_like(state.fields)
    init_carry = (state,init_delta)

    stage_pars = (alphas_arr, betas_arr, gammas_arr, jnp.arange(scheme.nstages))

    def scan_stage_func(carry,stage_vals):
        current_state, delta = carry
        alpha, beta, gamma, istage = stage_vals

        stage_rhs = jax.lax.cond(istage == 0,lambda: init_rhs,
                                 lambda: rhs(current_state,kgrid,params)[0])

        diss_factors = jnp.exp(diss_exponents*gamma)

        next_delta = diss_factors * (alpha * delta + dt * stage_rhs)
        next_fields = diss_factors * current_state.fields + beta*next_delta
        next_t = current_state.t + gamma*dt
        # forcing_state/forcing_key threaded through unchanged (see rk_advance comment above).
        return (current_state._replace(t=next_t,fields=next_fields),next_delta), None

    (final_state, _), _ = jax.lax.scan(scan_stage_func,init_carry,stage_pars)

    return final_state

#Code to set up a couple of common lsrk schemes

def _get_LSRK54_gammas():
    c2 = 1432997174477 / 9575080441755
    c3 = 2526269341429 / 6820363962896
    c4 = 2006345519317 / 3224310063776
    c5 = 2802321613138 / 2924317926251
    return (c2, c3 - c2, c4 - c3, c5 - c4, 1.0 - c5)

_scheme_registry = {
    "rk44": (rk_advance, None),
    "lsrk33": (lsrk_advance,
               LSRK_Scheme(
                   alphas = (0.0, -5.0 / 9.0, -153.0 / 128.0),
                   betas  = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0),
                   gammas = (1.0 / 3.0, 5.0 / 12.0, 1.0 / 4.0),
                   nstages = 3,
                   )), #This is the original Williamson 1980 RK3 scheme
    "lsrk54": (lsrk_advance,
               LSRK_Scheme(
                   alphas = (
                       0.0,
                       -567301805773 / 1357537059087,
                       -2404267990393 / 2016746695238,
                       -3550918686646 / 2091501179385,
                       -3270041069962 / 2362476162756
                       ),
                   betas = (
                       1432997174477 / 9575080441755,
                       5161836677717 / 13612068292357,
                       1720146321549 / 2090206949498,
                       3134564353537 / 4481467310338,
                       2277821191437 / 14882151754819
                       ),
                   gammas = _get_LSRK54_gammas(),
                   nstages = 5,
                   )) #5-stage, 4th-order scheme from Carpenter & Kennedy 1994
}

def get_scheme(name: str):
    if name not in _scheme_registry:
        raise ValueError(f"Unknown scheme '{name}'. Available: {list(_scheme_registry.keys())}")
    return _scheme_registry[name]