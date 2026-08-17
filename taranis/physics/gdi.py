# Gradient-Drift-Instability (GDI) equation set, built
# Physics source: docs/"GDI_nonlinear_equations.pdf". 
# Normalization: rho_s = c_s = Omega_i = 1 (eqs 5.4-5.5/3.5-3.6)
# fields are (N, phi) in that order, N = delta n/n0, phi = e(electrostatic potential)/T_e.
#
# Closure: eq (4.3)'s 2D floor gamma_par(k) = gpar_fac*nu_in*k_perp^2 (P4a) and the real
# 3D parallel-diffusion term gamma_par = D_par*kz^2 (eqs 3.5-3.7, P4b, z_spectral only)
# are added: gamma_par_total = gpar_fac*nu_in*k_perp^2 + D_par*kz^2. 
# in 3D default is gpar_fac=0, which is the "physical" case
#
# Linear matrix L (dt [N,phi] = L [N,phi] + NL), derived directly from eqs (5.1)-(5.2) /
# (3.5)-(3.6) with gamma_par_total substituted for the bare D_par*d^2/dz^2 term, cross-
# checked against the exact dispersion relation (3.7)/(2.8) and (2D-floor-only)
# against the paper's eq (5.3) matrix under L = -M (docs/gdi_linear_matrix_note.tex); the
# 3D quartic (3.11) and its named limits (2D 3.6, Hasegawa-Wakatani 3.12, nearly-adiabatic
# 3.9, stabilization boundary 3.15) are checked in tests/test_gdi_linear.py:
#   L[N,N]     = -gamma_par - diss*k_perp^(2*hyper)
#   L[N,phi]   =  gamma_par + i*ky_deriv/Ln
#   L[phi,N]   =  gamma_par*inv_ksq - i*ky_deriv*nu_in*v0*inv_ksq
#   L[phi,phi] = -nu_in - gamma_par*inv_ksq - diss*k_perp^(2*hyper)
# ky_deriv is kgrid.ky with the Nyquist row zeroed: this keeps reality constraint satisfied
#
# 3D requires params.z_spectral=True (raised at _check_supported: single GPU)
# CB-IMEX (imexcb3e default) should be used with the 3D equations: IF-LSRK only allowed
# with a fixed dt<~ 1/max(gamma_par/k_perp^2/rho_s^2)
#
# This module holds only what the solver consumes (the recipe functions and their helpers);
# the read-only GDI observers live in taranis/diagnostics/gdi.py.

import functools

import jax.numpy as jnp
import numpy as np

from .. import comms, grids
from . import shared_physics
from .shared_physics import bracket, gradk


def _check_supported(params):
    if params.spatial_dimensions == 3 and not params.z_spectral:
        raise NotImplementedError("GDI dims==3 requires z_spectral=True: the parallel "
                                  "closure gamma_par=D_par*kz^2 needs a kz axis "
                                  "(kgrid.kz), which only exists in that mode.")
    if params.forcing:
        raise ValueError("GDI is instability-driven (linear growth from Ln/v0): "
                         "forcing=True is not supported")


_EQPARS_REQUIRED_2D = ("Ln", "nu_in", "v0", "diss", "hyper")
_EQPARS_REQUIRED_3D = _EQPARS_REQUIRED_2D + ("D_par",)
_EQPARS_OPTIONAL = ("gpar_fac", "lin_dt_safety")

def _eqpars(params):
    # pull & validate GDI eqpars out of params.eqpars.
    # Unknown keys are rejected
    _check_supported(params)
    is3d = params.spatial_dimensions == 3
    required = _EQPARS_REQUIRED_3D if is3d else _EQPARS_REQUIRED_2D
    missing = [k for k in required if k not in params.eqpars]
    unknown = [k for k in params.eqpars if k not in required + _EQPARS_OPTIONAL]
    if missing or unknown:
        raise ValueError(f"GDI eqpars problem (missing {missing}, unknown {unknown}): "
                         f"required {required}, optional {_EQPARS_OPTIONAL}; "
                         f"got eqpars={params.eqpars!r}")
    ep = params.eqpars
    gpar_default = 0.0 if is3d else 1.0
    D_par = ep["D_par"] if is3d else 0.0
    return ep["Ln"], ep["nu_in"], ep["v0"], ep.get("gpar_fac", gpar_default), ep["diss"], ep["hyper"], D_par


def _lin_dt_safety(params):
    # safety factor on the dt ceiling from max|Re lambda(L)|
    return float(params.eqpars.get("lin_dt_safety", 0.5))


def _closure_terms(ksq, inv_ksq, kz, gpar_fac, nu_in, D_par):
    gpar_ratio = gpar_fac * nu_in
    gamma_par = gpar_ratio * ksq
    if kz is not None:
        gamma_par_kz = D_par * kz**2
        gamma_par = gamma_par + gamma_par_kz
        gpar_ratio = gpar_ratio + gamma_par_kz*inv_ksq
    return gamma_par, gpar_ratio


def _L_entries(ksq, ky_deriv, inv_ksq, Ln, nu_in, v0, gamma_par, gpar_ratio, diss, hyper):
    # shared between linear_matrix (jnp) and _max_re_lambda (np, setup-time only)
    hyperdiss = diss * ksq**hyper
    L_NN = -gamma_par - hyperdiss
    L_Nphi = gamma_par + 1j*ky_deriv/Ln
    L_phiN = gpar_ratio - 1j*ky_deriv*nu_in*v0*inv_ksq
    L_phiphi = -nu_in - gpar_ratio - hyperdiss
    return L_NN, L_Nphi, L_phiN, L_phiphi


def linear_matrix(kgrid, params):
    Ln, nu_in, v0, gpar_fac, diss, hyper, D_par = _eqpars(params)
    ky_deriv = kgrid.ky.at[..., -1].set(0.0)   # zero the Nyquist row: maintain real fields
    kz = kgrid.kz if params.spatial_dimensions == 3 else None
    gamma_par, gpar_ratio = _closure_terms(kgrid.ksq, kgrid.inv_ksq, kz, gpar_fac, nu_in, D_par)
    L00, L01, L10, L11 = _L_entries(kgrid.ksq, ky_deriv, kgrid.inv_ksq,
                                    Ln, nu_in, v0, gamma_par, gpar_ratio, diss, hyper)
    L = jnp.stack([jnp.stack([L00, L01]), jnp.stack([L10, L11])])
    return L if params.spatial_dimensions == 3 else L[:, :, None, :, :]


def _perp_grids_np(params):
    # numpy float64 rebuild of the perpendicular k-grid and dealias region used by
    # _max_re_lambda (setup-time only; see its comment for why grids/kgrid are not used).
    # kx, ky are broadcast-shaped (nkx,1) and (1,nky) like kgrid.kx/ky; perp_dealias is the
    # (nkx,nky) boolean 2/3 ellipse, i.e. grids.dealias_mask's perpendicular factor.
    # tests/test_gdi_linear.py pins both against grids.setup_kgrids/dealias_mask.
    kx = np.fft.fftfreq(params.nx) * params.nx * 2*np.pi/params.Lx
    ky = np.fft.rfftfreq(params.ny) * params.ny * 2*np.pi/params.Ly
    kx_grid, ky_grid = kx.reshape(-1, 1), ky.reshape(1, -1)
    ksq = kx_grid**2 + ky_grid**2
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_ksq = np.where(ksq > 0, 1.0/ksq, 0.0)
    ky_deriv = ky_grid.copy()
    ky_deriv[..., -1] = 0.0   # Nyquist row: matches linear_matrix's ky_deriv
    ix = np.fft.fftfreq(params.nx) * params.nx
    iy = np.fft.rfftfreq(params.ny) * params.ny
    perp_dealias = ((ix.reshape(-1, 1)/(params.nx/3.0))**2 +
                    (iy.reshape(1, -1)/(params.ny/3.0))**2) < 1.0
    return kx_grid, ky_grid, ksq, inv_ksq, ky_deriv, perp_dealias


def _kz_values_np(params):
    # kz planes surviving the 2/3 rule, as a list of (kz, mode index iz) float pairs in
    # fftfreq order. Companion to _perp_grids_np; requires dims==3 (z_spectral).
    kzs = np.fft.fftfreq(params.nz) * params.nz * 2*np.pi/params.Lz
    iz = np.fft.fftfreq(params.nz) * params.nz
    return [(float(k), float(i)) for k, i in zip(kzs, iz) if abs(i) < params.nz/3.0]


@functools.lru_cache(maxsize=32)   # bounded: params is identity-hashed, so an unbounded
def _max_re_lambda(params):        # cache would pin every Parameters in a long param scan
    # dt ceiling for nonlinear-vs-linear accuracy near saturation: the fastest growing
    # linear rate max(Re lambda(L), 0)
    Ln, nu_in, v0, gpar_fac, diss, hyper, D_par = _eqpars(params)
    _, _, ksq, inv_ksq, ky_deriv, perp_dealias = _perp_grids_np(params)

    def _plane_gmax(kz_val):
        # fastest-growing signed rate over one kz plane
        gamma_par, gpar_ratio = _closure_terms(ksq, inv_ksq, kz_val, gpar_fac, nu_in, D_par)
        L00, L01, L10, L11 = _L_entries(ksq, ky_deriv, inv_ksq, Ln, nu_in, v0,
                                        gamma_par, gpar_ratio, diss, hyper)
        m, s2 = shared_physics.eig2_ms(L00, L01, L10, L11)
        s = np.sqrt(s2.astype(complex))
        re_max = np.maximum((m + s).real, (m - s).real)
        return float(np.max(np.where(perp_dealias, re_max, -np.inf)))

    if params.spatial_dimensions == 3:
        gmax = max(_plane_gmax(kz_val) for kz_val, _ in _kz_values_np(params))
    else:
        gmax = _plane_gmax(None)
    return max(gmax, 0.0)


def grad(state, kgrid, params):
    # everything needed for the brackets
    Nk = state.fields[0]
    phik = state.fields[1]
    vortk = -kgrid.ksq*phik
    fk = jnp.stack([phik, Nk, vortk])
    return grids.ifft(gradk(fk, kgrid), params)


def set_timestep(grads, params):
    # ExB CFL from |grad phi|
    gphi, gN, gvort = grads
    max_vy = jnp.max(jnp.abs(gphi[0]))
    max_vx = jnp.max(jnp.abs(gphi[1]))
    eps = shared_physics.QUIESCENT_EPS
    max_eps = jnp.maximum(eps/params.dx, eps/params.dy)
    max_all = jnp.maximum(max_vx/params.dx, max_vy/params.dy)
    max_all = jnp.maximum(max_all, max_eps)
    max_all = comms.allreduce_max(max_all, params)
    dt_cfl = params.cfl_safety / max_all
    # growth-rate-only ceiling (see _max_re_lambda)
    gmax = _max_re_lambda(params)
    if gmax > 0.0:
        dt_cfl = jnp.minimum(dt_cfl, _lin_dt_safety(params)/gmax)
    return dt_cfl


def NonlinearTerm(state, grads, kgrid, params, halo=None):
    gphi, gN, gvort = grads
    NLTerm_N = -bracket(gphi, gN)
    NLTerm_vort = -bracket(gphi, gvort)
    NLTerm_N_k, NLTerm_vort_k = grids.fft(jnp.stack([NLTerm_N, NLTerm_vort]), params)
    NLTerm_fields = jnp.stack([NLTerm_N_k, -kgrid.inv_ksq*NLTerm_vort_k]) * kgrid.dealias
    return NLTerm_fields
