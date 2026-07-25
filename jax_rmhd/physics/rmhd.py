import jax.numpy as jnp
from .. import grids
from . import shared_physics
from .shared_physics import gradk,bracket,z_derivatives
from .. import comms

def grad(state,kgrid,params):
    phik=state.fields[0]
    psik=state.fields[1]
    vortk = -kgrid.ksq()*phik
    jpark = -kgrid.ksq()*psik
    fk = jnp.stack([phik,psik,vortk,jpark])
    gradients = grids.ifft(gradk(fk,kgrid),params)
    return gradients

def set_timestep(grads,params):
    #Sets the timestep according to the CFL condition.
    gphi = grads[0]
    gpsi = grads[1]    
    max_vy_eff = jnp.max(jnp.abs(gphi[0])+jnp.abs(gpsi[0]))
    max_vx_eff = jnp.max(jnp.abs(gphi[1])+jnp.abs(gpsi[1]))
    eps=0.1
    max_eps = jnp.maximum(eps/params.dx,eps/params.dy)
    max_all = jnp.maximum(max_vx_eff/params.dx, max_vy_eff/params.dy)
    max_all = jnp.maximum(max_all,max_eps)
    if params.spatial_dimensions==3:
        max_all = jnp.maximum(max_all,1.0/params.dz)
        max_all = jnp.maximum(max_all,params.z_diss)
    max_all = comms.allreduce_max(max_all,params)  # no-op unless z-decomposed
    return params.cfl_safety / max_all

def halo_start(state,kgrid,params):
    # T7: pre-issues LinearTerm's z halo exchange at the top of the RHS; None in 2D (no halo).
    if params.spatial_dimensions==2:
        return None
    return comms.halo_exchange(state.fields,params)

def NonlinearTerm(state,grads,kgrid,params,halo=None):
    gphi,gpsi,gvort,gjpar = grads
    NLTerm_vort = bracket(gpsi,gjpar) - bracket(gphi,gvort)
    NLTerm_psi = - bracket(gphi,gpsi)
    (NLTerm_vort_k , NLTerm_psi_k) = grids.fft(jnp.stack([NLTerm_vort,NLTerm_psi]))
    NLTerm_fields = jnp.stack([-kgrid.inv_ksq()*NLTerm_vort_k,NLTerm_psi_k])*kgrid.dealias_filter()
    return NLTerm_fields

def LinearTerm(state,grads,kgrid,params,halo=None):
    #TODO: add a check on z_diff_order and z_diss_hyper here. For now use 4th order centered f.d. 
    # and d_z^4 hyperdissipation for stability
    if params.spatial_dimensions==2:
        return jnp.zeros_like(state.fields)
    dz=params.dz
    diss=params.z_diss * (dz/2)**4
    df_dz,d4f_dz4 = z_derivatives(state.fields,params,halo=halo)  # T7: reuse the pre-issued halo
    #RMHD only logic: the z-derivatives belong to the opposite equations
    df_dz_rmhd = jnp.stack([df_dz[1],df_dz[0]])
    return df_dz_rmhd - diss * d4f_dz4

def _forcing_scale_from(fields, f_raw, kgrid, params):
    # (n_ou,) power-normalization scale factor(s) for the given fields and forcing envelope.
    phik = fields[0]
    psik = fields[1]
    if params.forcing_mode == "momentum":
        P = shared_physics.perp_inner_product(phik,f_raw[0],kgrid,params)
        return jnp.reshape(shared_physics.safe_scale(params.forcing_power,P,params.forcing_scale_max),(1,))
    # elsasser: (Pp,Pm) now computed with a single stacked allreduce instead of two.
    za = jnp.stack([phik + psik, phik - psik])
    Ppm = shared_physics.perp_inner_product_batch(za,f_raw,kgrid,params)
    eps = jnp.asarray(params.forcing_power_elsasser)
    return shared_physics.safe_scale(eps,Ppm,params.forcing_scale_max)

def forcing_scale(state,kgrid,params):
    # Once-per-full-step scale for params.forcing_norm_per_step, called from run.py right
    # after ou_update (registered as forcing_scale_func in the equation registry).
    f_raw = shared_physics.reconstruct_envelope(state.forcing_state,kgrid,params)
    return _forcing_scale_from(state.fields,f_raw,kgrid,params)

def ForcingTerm(state,grads,kgrid,params,halo=None):
    # RMHD-specific forcing: either in the momentum equation or elsasser forcing
    if not params.forcing:
        return jnp.zeros_like(state.fields)
    # z-envelopes come precomputed from kgrid (setup_kgrids) when available, so no need
    # to recompute local_z_coords here every call.
    f_raw = shared_physics.reconstruct_envelope(state.forcing_state,kgrid,params)
    if params.forcing_norm_per_step:
        # approximation: reuse the scale computed once per step (start-of-step fields and
        # this step's OU state), skipping the per-sub-stage allreduce entirely.
        if state.forcing_scale is None:
            # trace-time check: catches hand-built states; go through run.initialize/simulate
            raise ValueError("forcing_norm_per_step=True requires state.forcing_scale "
                             "(build states via run.initialize / restore via load_snapshot)")
        scale = state.forcing_scale
    else:
        scale = _forcing_scale_from(state.fields,f_raw,kgrid,params)
    if params.forcing_mode == "momentum":
        f_phi = f_raw[0] * scale[0]
        f_psi = jnp.zeros_like(f_phi)
    else:
        f_plus = f_raw[0] * scale[0]
        f_minus = f_raw[1] * scale[1]
        f_phi = 0.5*(f_plus+f_minus)
        f_psi = 0.5*(f_plus-f_minus)
    return jnp.stack([f_phi,f_psi])
