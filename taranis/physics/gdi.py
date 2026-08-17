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


@functools.lru_cache(maxsize=32)   # bounded: params is identity-hashed, so an unbounded
def _max_re_lambda(params):        # cache would pin every Parameters in a long param scan
    # dt ceiling for nonlinear-vs-linear accuracy near saturation: the fastest growing
    # linear rate max(Re lambda(L), 0)
    Ln, nu_in, v0, gpar_fac, diss, hyper, D_par = _eqpars(params)
    kx = np.fft.fftfreq(params.nx) * params.nx * 2*np.pi/params.Lx
    ky = np.fft.rfftfreq(params.ny) * params.ny * 2*np.pi/params.Ly
    kx_grid, ky_grid = kx.reshape(-1, 1), ky.reshape(1, -1)
    ksq = kx_grid**2 + ky_grid**2
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_ksq = np.where(ksq > 0, 1.0/ksq, 0.0)
    ky_deriv = ky_grid.copy()
    ky_deriv[..., -1] = 0.0
    ix = np.fft.fftfreq(params.nx) * params.nx
    iy = np.fft.rfftfreq(params.ny) * params.ny
    perp_dealias = ((ix.reshape(-1, 1)/(params.nx/3.0))**2 +
                    (iy.reshape(1, -1)/(params.ny/3.0))**2) < 1.0

    def _plane_gmax(kz_val):
        # fastest-growing signed rate over one kz plane
        gamma_par, gpar_ratio = _closure_terms(ksq, inv_ksq, kz_val, gpar_fac, nu_in, D_par)
        L00, L01, L10, L11 = _L_entries(ksq, ky_deriv, inv_ksq, Ln, nu_in, v0,
                                        gamma_par, gpar_ratio, diss, hyper)
        m = 0.5*(L00 + L11)
        s = np.sqrt((m*m - (L00*L11 - L01*L10)).astype(complex))
        re_max = np.maximum((m + s).real, (m - s).real)
        return float(np.max(np.where(perp_dealias, re_max, -np.inf)))

    if params.spatial_dimensions == 3:
        kzs = np.fft.fftfreq(params.nz) * params.nz * 2*np.pi/params.Lz
        iz = np.fft.fftfreq(params.nz) * params.nz
        gmax = max(_plane_gmax(float(k)) for k, i in zip(kzs, iz) if abs(i) < params.nz/3.0)
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
    eps = 0.1
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


def energy_enstrophy(state, kgrid, params):
    # E = 0.5<N^2 + |grad_perp phi|^2> (eq 3.16), Z = 0.5<(N - laplacian(phi))^2> (eq 3.17).
    Nk, phik = state.fields[0], state.fields[1]
    vortk = -kgrid.ksq*phik
    E = 0.5*(shared_physics.perp_mean_square(Nk, Nk, kgrid, params) +
             shared_physics.perp_inner_product(phik, phik, kgrid, params))
    Wk = Nk - vortk
    Z = 0.5*shared_physics.perp_mean_square(Wk, Wk, kgrid, params)
    return E, Z


def energy_budget(state, kgrid, params):
    # dE/dt source/sink decomposition
    Ln, nu_in, v0, gpar_fac, diss, hyper, D_par = _eqpars(params)
    Nk, phik = state.fields[0], state.fields[1]
    ksq, yfac, inv_ksq = kgrid.ksq, kgrid.yfac, kgrid.inv_ksq
    ky_deriv = kgrid.ky.at[..., -1].set(0.0)
    kz = kgrid.kz if params.spatial_dimensions == 3 else None
    gamma_par, gpar_ratio = _closure_terms(ksq, inv_ksq, kz, gpar_fac, nu_in, D_par)
    cross_im = jnp.imag(jnp.conj(Nk)*phik) * yfac
    drive_Ln = shared_physics.perp_reduce(-(ky_deriv/Ln)*cross_im, params)
    drive_ped = shared_physics.perp_reduce(-ky_deriv*nu_in*v0*cross_im, params)
    visc = -nu_in*shared_physics.perp_inner_product(phik, phik, kgrid, params)
    dk = Nk - phik
    gpar_term = shared_physics.perp_reduce(
        (-gamma_par*jnp.real(jnp.conj(Nk)*dk)
         + (ksq*gpar_ratio)*jnp.real(jnp.conj(phik)*dk)) * yfac, params)
    hyper_N = shared_physics.perp_reduce(-diss*ksq**hyper*jnp.abs(Nk)**2*yfac, params)
    hyper_phi = shared_physics.perp_reduce(-diss*ksq**(hyper+1)*jnp.abs(phik)**2*yfac, params)
    total = drive_Ln + drive_ped + visc + gpar_term + hyper_N + hyper_phi
    return dict(drive_Ln=drive_Ln, drive_pedersen=drive_ped, viscous=visc,
               gamma_par=gpar_term, hyper_N=hyper_N, hyper_phi=hyper_phi, total=total)


def perp_spectrum(state, kgrid, params, bin_factor=2.0):
    # perpendicular density/electric-field spectra
    from .. import diagnostics
    Nk, phik = state.fields[0], state.fields[1]
    N2 = jnp.real(jnp.conj(Nk)*Nk)*kgrid.yfac
    spec_N = shared_physics.perp_reduce(N2, params, axis=0)
    spec_phi = shared_physics.perp_reduce(0.5*shared_physics.perp_gradsq(phik, phik, kgrid), params, axis=0)
    kunit = min(2*jnp.pi/params.Lx, 2*jnp.pi/params.Ly)
    kmax = min(params.nx//2, params.ny//2)*kunit
    return diagnostics._binned(jnp.sqrt(kgrid.ksq), (spec_N, spec_phi), kunit, kmax, bin_factor)


def cross_phase_spectrum(state, kgrid, params, bin_factor=2.0, kz_index=0):
    # N-phi cross-phase and amplitude ratio |N|/|phi| vs k_perp
    # kz_index selects a single kz (or none in 2d)
    from .. import diagnostics
    Nk, phik = state.fields[0, kz_index], state.fields[1, kz_index]
    yfac = kgrid.yfac
    N2 = jnp.abs(Nk)**2 * yfac
    phi2 = jnp.abs(phik)**2 * yfac
    cross = jnp.conj(phik)*Nk * yfac
    kunit = min(2*jnp.pi/params.Lx, 2*jnp.pi/params.Ly)
    kmax = min(params.nx//2, params.ny//2)*kunit
    kbins, N2b, phi2b, cross_re, cross_im = diagnostics._binned(
        jnp.sqrt(kgrid.ksq), (N2, phi2, jnp.real(cross), jnp.imag(cross)), kunit, kmax, bin_factor)
    amp_ratio = jnp.sqrt(N2b/phi2b)
    cross_phase = jnp.arctan2(cross_im, cross_re)
    return kbins, amp_ratio, cross_phase


def kperp_break(params, kz=0.0, ky_lo=1e-3, ky_hi=None, tol=1e-6, max_iter=60):
    # marginal-stability k_perp, computed from the linear dispersion relation
    Ln, nu_in, v0, gpar_fac, diss, hyper, D_par = _eqpars(params)
    gamma_par_kz = D_par*kz*kz

    def max_re(ky):
        ksq = ky*ky
        inv_ksq = 0.0 if ksq == 0.0 else 1.0/ksq
        gamma_par = gpar_fac*nu_in*ksq + gamma_par_kz
        gpar_ratio = gpar_fac*nu_in + gamma_par_kz*inv_ksq
        L00, L01, L10, L11 = _L_entries(ksq, ky, inv_ksq, Ln, nu_in, v0,
                                        gamma_par, gpar_ratio, diss, hyper)
        m = 0.5*(L00 + L11)
        s = np.sqrt(complex(m*m - (L00*L11 - L01*L10)))
        return max((m + s).real, (m - s).real)

    if ky_hi is None:
        ky_hi = 20.0*max(2*np.pi/params.Lx, 2*np.pi/params.Ly)   # generous default band
    if max_re(ky_lo) <= 0.0 or max_re(ky_hi) > 0.0:
        return None
    lo, hi = ky_lo, ky_hi
    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        if max_re(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5*(lo + hi)


def measure_alpha(state, kgrid, params, kz_index, modes):
    # eq (4.7)
    Ln, nu_in, v0, gpar_fac, diss, hyper, D_par = _eqpars(params)
    kz = float(kgrid.kz[kz_index, 0, 0]) if params.spatial_dimensions == 3 else 0.0
    gamma_par_kz = D_par*kz*kz
    out = []
    for ikx, iky in modes:
        Nk = complex(state.fields[0, kz_index, ikx, iky])
        phik = complex(state.fields[1, kz_index, ikx, iky])
        ky = float(kgrid.ky[0, iky])
        ksq = float(kgrid.ksq[ikx, iky])
        if phik == 0.0 or ky == 0.0 or ksq == 0.0:
            continue
        ratio = Nk/phik
        alpha_measured = ratio.imag * v0 * ky / ksq
        alpha_theory = 1.0 + (gpar_fac*nu_in*ksq + gamma_par_kz)/(nu_in*ksq)
        out.append(dict(ikx=ikx, iky=iky, kz=kz, ratio=ratio,
                        alpha_measured=alpha_measured, alpha_theory=alpha_theory))
    return out
