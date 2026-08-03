import jax.numpy as jnp
import numpy as np
from .. import grids
from . import shared_physics
from .shared_physics import gradk,bracket,z_derivatives
from .. import comms

def grad(state,kgrid,params):
    phik=state.fields[0]
    psik=state.fields[1]
    vortk = -kgrid.ksq*phik
    jpark = -kgrid.ksq*psik
    fk = jnp.stack([phik,psik,vortk,jpark])
    gradients = grids.ifft(gradk(fk,kgrid),params)
    return gradients

def linear_matrix(kgrid,params):
    # RMHD's linear operator. Finite-difference z: just the diagonal perpendicular
    # (hyper)dissipation L = -diss*k_perp^(2*hyper) (the z derivatives are an RHS term).
    # z_spectral: the Alfven coupling is k-local too, so it joins L as a 2x2 per (kz,k_perp)
    # and the putzer2 backend propagates it EXACTLY -- no wave CFL. Applied as exp(L*tau).
    # diss/hyper live in params.eqpars (they were Parameters ctor args before 2026-08-01).
    diss_par, hyper = _diss_hyper(params)
    zdiss = _z_diss_k(params)   # validated in BOTH modes: it is meaningless without kz
    if not params.z_spectral:
        diss = jnp.array(diss_par).reshape(-1,1,1,1)
        return -diss*kgrid.ksq**hyper
    # phi/psi equations: dt phi = i*kz*psi + ..., dt psi = i*kz*phi + ... -- exactly the
    # spectral-z form of LinearTerm's d(psi)/dz in the (vorticity/-k_perp^2) equation and
    # d(phi)/dz in the psi equation. Eigenvalues +-i*kz: Alfven waves, phase speed 1.
    diss = jnp.broadcast_to(jnp.array(diss_par).reshape(-1), (params.nfields,))
    kz = _kz_deriv(kgrid,params)
    # (nz,1,1) kz against (nkx,nky) k_perp broadcasts every entry to (nz,nkx,nky)
    dphi = -diss[0]*kgrid.ksq**hyper - zdiss*kz**4
    dpsi = -diss[1]*kgrid.ksq**hyper - zdiss*kz**4
    off = jnp.broadcast_to(1j*kz, dphi.shape)
    zero = jnp.zeros_like(off)
    return jnp.stack([jnp.stack([dphi + zero, off]),
                      jnp.stack([off, dpsi + zero])])

def _kz_deriv(kgrid,params):
    # kz for the ODD-order (i*kz) coupling, with the kz-Nyquist plane zeroed: at kz_Nyq the
    # mirror kz -> -kz is the identity, so a bare i*kz breaks the reality constraint
    # L(-kx,-kz,ky) = conj(L(kx,kz,ky)) that propagators._check_hermitian_compatible enforces
    # (same subtlety as gdi's ky_deriv; the 2/3 kz cut removes that plane from every
    # nonlinear/IC path anyway).
    kz = kgrid.kz
    return kz.at[params.nz//2].set(0.0) if params.nz % 2 == 0 else kz

def _z_diss_k(params):
    # optional kz hyperdissipation coefficient (-z_diss_k*kz^4, both fields); z_spectral only
    zdiss = params.eqpars.get("z_diss_k", 0.0)
    if zdiss and not params.z_spectral:
        raise ValueError("eqpars['z_diss_k'] is a spectral-z knob (-z_diss_k*kz^4) and needs "
                         "z_spectral=True; the finite-difference-z filter is params.z_diss")
    return zdiss

def _diss_hyper(params):
    # pull (and validate) RMHD's dissipation parameters out of the equation-parameter dict
    missing = [k for k in ("diss","hyper") if k not in params.eqpars]
    unknown = [k for k in params.eqpars if k not in ("diss","hyper","z_diss_k")]
    if missing or unknown:
        # unknown keys are rejected: a typo would otherwise be silently ignored
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
    #velocity floor: caps dt at cfl_safety*min(dx,dy)/eps for a near-quiescent field
    eps=0.1
    max_eps = jnp.maximum(eps/params.dx,eps/params.dy)
    max_all = jnp.maximum(max_vx_eff/params.dx, max_vy_eff/params.dy)
    max_all = jnp.maximum(max_all,max_eps)
    if params.spatial_dimensions==3 and not params.z_spectral:
        # both terms are finite-difference-z artefacts: the Alfven wave CFL (1/dz, speed 1)
        # and the z filter's rate. In spectral z the propagator handles both exactly, so dt
        # is set by perpendicular advection alone -- the payoff of the mode.
        max_all = jnp.maximum(max_all,1.0/params.dz)
        max_all = jnp.maximum(max_all,params.z_diss)
    max_all = comms.allreduce_max(max_all,params)  # no-op unless z-decomposed
    return params.cfl_safety / max_all

def halo_start(state,kgrid,params):
    # pre-issues LinearTerm's z halo exchange at the top of the RHS; None in 2D (no halo).
    # width must match what shared_physics.z_derivatives' stencil expects (RMHD: 4th-order
    # centered + 5-point d4 => 2); the pre-issued halo here and the fallback exchange inside
    # z_derivatives MUST use the same width -- the one coupling in this design.
    # z_spectral has no z stencil at all (the parallel term lives in the propagator), so
    # there is nothing to exchange there either.
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

def LinearTerm(state,grads,kgrid,params,halo=None):
    # fixed at 4th-order centered f.d. + d_z^4 hyperdissipation: params.z_diff_order and
    # z_diss_hyper are not read here (Parameters warns when they are set).
    # z_spectral: the whole parallel operator is in linear_matrix (exact propagator), so
    # this term is skipped -- plain `if` on a static param, never lax.cond.
    if params.spatial_dimensions==2 or params.z_spectral:
        return jnp.zeros_like(state.fields)
    dz=params.dz
    diss=params.z_diss * (dz/2)**4
    df_dz,d4f_dz4 = z_derivatives(state.fields,params,halo=halo)
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
    za = jnp.stack([phik + psik, phik - psik])
    Ppm = shared_physics.perp_inner_product(za,f_raw,kgrid,params,batch=True)
    #factor 2: E_tot = (E+ + E-)/2, so this makes each forcing_power_elsasser entry a
    #contribution to the TOTAL energy injection rate, in the same units as forcing_power
    eps = 2.0*jnp.asarray(params.forcing_power_elsasser)
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
