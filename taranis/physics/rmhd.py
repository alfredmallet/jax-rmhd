import jax.numpy as jnp
import numpy as np
from .. import grids
from .. import _precision
from . import shared_physics
from .shared_physics import bracket,grad_fields,z_derivatives,z_stencil_blocks
from .. import comms
from ..propagators import SeparableL

def grad(state,kgrid,params):
    # (gphi, gpsi, gvort, gjpar), one real-space (2,nz,nx,ny) gradient per field
    phik=state.fields[0]
    psik=state.fields[1]
    return grad_fields((phik,psik,-kgrid.ksq*phik,-kgrid.ksq*psik),kgrid,params)

# when False, z_spectral returns the dense 2x2 operator (the putzer2 backend) even at
# nu == eta; benchmarks and tests flip it, nothing else reads it
SEPARABLE_L = True

def linear_matrix(kgrid,params):
    # finite-difference z: just perp (hyper)dissipation L = -diss*k_perp^(2*hyper)
    # z_spectral: includes the AW propagation.  Applied as exp(L*tau).
    diss_par, hyper = _diss_hyper(params)
    zdiss = _z_diss_k(params)   # validated in both modes, meaningless without kz
    if not params.z_spectral:
        diss = jnp.array(diss_par, dtype=_precision.ftype).reshape(-1,1,1,1)
        return -diss*kgrid.ksq**hyper
    diss = jnp.broadcast_to(jnp.array(diss_par, dtype=_precision.ftype).reshape(-1),
                            (params.nfields,))
    kz = _kz_deriv(kgrid,params)
    if SEPARABLE_L and _equal_dissipation(diss_par, params):
        # L = d*I + i*kz*sigma_x with d = dperp + dz: the separable backend
        return SeparableL(dperp=-diss[0]*kgrid.ksq**hyper, dz=-zdiss*kz**4, kz=kz)
    # (nz,1,1) kz against (nkx,nky) k_perp broadcasts every entry to (nz,nkx,nky)
    dphi = -diss[0]*kgrid.ksq**hyper - zdiss*kz**4
    dpsi = -diss[1]*kgrid.ksq**hyper - zdiss*kz**4
    off = jnp.broadcast_to(1j*kz, dphi.shape)
    zero = jnp.zeros_like(off)
    return jnp.stack([jnp.stack([dphi + zero, off]),
                      jnp.stack([off, dpsi + zero])])

def _equal_dissipation(diss_par, params):
    # nu == eta as python floats: the condition for L to be d*I + i*kz*sigma_x
    d = np.broadcast_to(np.asarray(diss_par, dtype=float), (params.nfields,))
    return bool(np.all(d == d[0]))

def _kz_deriv(kgrid,params):
    # kz with the Nyquist plane zeroed to maintain reality constraint
    kz = kgrid.kz
    return kz.at[params.nz//2].set(0.0) if params.nz % 2 == 0 else kz

def _z_diss_k(params):
    #kz hyperdissipation coefficient (-z_diss_k*kz^4, both fields); z_spectral only
    zdiss = params.eqpars.get("z_diss_k", 0.0)
    if zdiss and not params.z_spectral:
        raise ValueError("eqpars['z_diss_k'] is a spectral-z knob (-z_diss_k*kz^4) and needs "
                         "z_spectral=True; the finite-difference-z filter is params.z_diss")
    return zdiss

def _diss_hyper(params):
    missing = [k for k in ("diss","hyper") if k not in params.eqpars]
    unknown = [k for k in params.eqpars if k not in ("diss","hyper","z_diss_k")]
    if missing or unknown:
        raise ValueError(f"RMHD eqpars problem (missing {missing}, unknown {unknown}): "
                         f"expected exactly {{'diss': (nu, eta), 'hyper': 1}}-style keys; "
                         f"got eqpars={params.eqpars!r}")
    diss = params.eqpars["diss"]
    if np.shape(diss) not in ((), (1,), (params.nfields,)):
        raise ValueError(f"eqpars['diss'] must be a scalar (applied to every field) or a "
                         f"length-{params.nfields} sequence (one per RMHD field), got {diss!r}")
    return diss, params.eqpars["hyper"]

def set_timestep(grads,params):
    #Sets the timestep according to the CFL condition.
    gphi = grads[0]
    gpsi = grads[1]    
    max_vy_eff = jnp.max(jnp.abs(gphi[0])+jnp.abs(gpsi[0]))
    max_vx_eff = jnp.max(jnp.abs(gphi[1])+jnp.abs(gpsi[1]))
    max_all = jnp.maximum(max_vx_eff/params.dx, max_vy_eff/params.dy)
    if params.spatial_dimensions==3 and not params.z_spectral:
        max_all = jnp.maximum(max_all,1.0/params.dz)
        max_all = jnp.maximum(max_all,params.z_diss)
    max_all = comms.allreduce_max(max_all,params)
    # the velocity floor is the _quiescent_dt ceiling: dt <= _quiescent_dt for every state
    return jnp.minimum(params.cfl_safety / max_all, _quiescent_dt(params))

def halo_start(state,kgrid,params):
    # width must match what shared_physics.z_derivatives' stencil expects
    if params.spatial_dimensions==2 or params.z_spectral:
        return None
    return comms.halo_exchange(state.fields,params,width=2)

def NonlinearTerm(state,grads,kgrid,params,halo=None):
    gphi,gpsi,gvort,gjpar = grads
    NLTerm_vort = bracket(gpsi,gjpar) - bracket(gphi,gvort)
    NLTerm_psi = - bracket(gphi,gpsi)
    (NLTerm_vort_k , NLTerm_psi_k) = grids.fft(jnp.stack([NLTerm_vort,NLTerm_psi]),params)
    NLTerm_fields = jnp.stack([-kgrid.inv_ksq*NLTerm_vort_k,NLTerm_psi_k])*kgrid.dealias
    return NLTerm_fields

def FDLinearTerm(state,grads,kgrid,params,halo=None):
    # the NON-k-local remainder of the linear physics: the finite-difference-z Alfven
    # stencil plus the d4/dz4 z filter. Live only for dims==3 without z_spectral; the
    # k-local part lives in linear_matrix and is applied by the propagators.
    # todo: add functionality for variable z_order
    if params.spatial_dimensions==2 or params.z_spectral:
        return jnp.zeros_like(state.fields)
    dz=params.dz
    diss=params.z_diss * (dz/2)**4
    #RMHD only logic: the z-derivatives belong to the opposite equations
    if shared_physics.Z_STENCIL_BLOCKS:
        #assembled one z block at a time: the concatenate is the term's own output, the
        #[::-1] field swap rides it, and neither z derivative becomes a slab
        blocks = z_stencil_blocks(state.fields,params,halo=halo)
        return jnp.concatenate([df_dz[::-1] - diss * d4f_dz4 for df_dz,d4f_dz4 in blocks],
                               axis=1)
    df_dz,d4f_dz4 = z_derivatives(state.fields,params,halo=halo)
    df_dz_rmhd = jnp.stack([df_dz[1],df_dz[0]])
    return df_dz_rmhd - diss * d4f_dz4

def _quiescent_dt(params):
    # static upper bound on the timestep: the ceiling set_timestep itself enforces on the
    # adaptive path (its velocity floor), which is what bounds the per-stage forcing cap.
    if not params.adaptive_timestep:
        return params.dt
    return params.cfl_safety*min(params.dx,params.dy)/shared_physics.QUIESCENT_EPS

def _forcing_scale_from(fields, f_raw, kgrid, params, dt=None):
    # (n_ou,) power-normalization scale factor(s) for the given fields and forcing envelope.
    # dt=None is the (non-default) per-stage path (forcing_norm_per_step=False): needs checking
    phik = fields[0]
    psik = fields[1]
    if params.forcing_mode == "momentum":
        P = shared_physics.perp_inner_product(phik,f_raw[0],kgrid,params)
        # F2 = <|grad f_raw|^2>, the forcing's self-energy rate. One extra reduction of the
        # same machinery as P; no RNG is involved, so forcing streams are untouched.
        F2 = shared_physics.perp_inner_product(f_raw[0],f_raw[0],kgrid,params)
        scale = _scale_epilogue(params.forcing_power,P,F2,dt,params)
        return jnp.reshape(scale,(1,))
    za = jnp.stack([phik + psik, phik - psik])
    Ppm = shared_physics.perp_inner_product(za,f_raw,kgrid,params,batch=True)
    # batched exactly like Ppm: one stacked reduce gives <|grad f+|^2>, <|grad f-|^2>
    F2pm = shared_physics.perp_inner_product(f_raw,f_raw,kgrid,params,batch=True)
    #factor 2: E_tot = (E+ + E-)/2: 
    #forcing_power_elsasser is contribution to the TOTAL energy injection rate
    eps = 2.0*jnp.asarray(params.forcing_power_elsasser, dtype=_precision.ftype)
    return _scale_epilogue(eps,Ppm,F2pm,dt,params)

def _scale_epilogue(tgt,P,F2,dt,params):
    # the two normalization paths, sharing the (tgt,P,F2) reductions computed above.
    if dt is not None:
        return shared_physics.selfnorm_scale(tgt,P,F2,dt,params.forcing_scale_max)
    F2dtq = F2*_quiescent_dt(params)
    cap = jnp.where(F2dtq > 0.0,
                    jnp.sqrt(2.0*jnp.abs(tgt)/jnp.where(F2dtq > 0.0, F2dtq, 1.0)),
                    params.forcing_scale_max)
    return shared_physics.safe_scale(tgt,P,jnp.minimum(params.forcing_scale_max,cap))

def forcing_scale(state,kgrid,params,dt):
    # called from run.py right after ou_update (forcing_scale_func in equation_registry).
    # dt is the last completed step dt (the step just completed): an approximation
    f_raw = shared_physics.reconstruct_envelope(state.forcing_state,kgrid,params)
    return _forcing_scale_from(state.fields,f_raw,kgrid,params,dt)

def ForcingTerm(state,grads,kgrid,params,halo=None):
    # RMHD-specific forcing: either in the momentum equation or elsasser forcing
    if not params.forcing:
        return jnp.zeros_like(state.fields)
    f_raw = shared_physics.reconstruct_envelope(state.forcing_state,kgrid,params)
    if params.forcing_norm_per_step:
        # reuse the scale computed once per step: approximation
        if state.forcing_scale is None:
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
