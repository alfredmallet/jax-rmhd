# GDI diagnostics: read-only observers on a GDI state (energy/enstrophy, energy budget,
# spectra, cross-phase, marginal-stability k_perp, adiabaticity alpha).
# The linear-operator helpers are shared with the solver: _eqpars/_closure_terms/_L_entries
# are imported from physics/gdi.py, i.e. underscore names crossing a module boundary.
import jax.numpy as jnp
import numpy as np

from ..physics import shared_physics
from ..physics.gdi import _closure_terms, _eqpars, _L_entries
from .core import _binned


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
    Nk, phik = state.fields[0], state.fields[1]
    N2 = jnp.real(jnp.conj(Nk)*Nk)*kgrid.yfac
    spec_N = shared_physics.perp_reduce(N2, params, axis=0)
    spec_phi = shared_physics.perp_reduce(0.5*shared_physics.perp_gradsq(phik, phik, kgrid), params, axis=0)
    kunit = min(2*jnp.pi/params.Lx, 2*jnp.pi/params.Ly)
    kmax = min(params.nx//2, params.ny//2)*kunit
    return _binned(jnp.sqrt(kgrid.ksq), (spec_N, spec_phi), kunit, kmax, bin_factor)


def cross_phase_spectrum(state, kgrid, params, bin_factor=2.0, kz_index=0):
    # N-phi cross-phase and amplitude ratio |N|/|phi| vs k_perp
    # kz_index selects a single kz (or none in 2d)
    Nk, phik = state.fields[0, kz_index], state.fields[1, kz_index]
    yfac = kgrid.yfac
    N2 = jnp.abs(Nk)**2 * yfac
    phi2 = jnp.abs(phik)**2 * yfac
    cross = jnp.conj(phik)*Nk * yfac
    kunit = min(2*jnp.pi/params.Lx, 2*jnp.pi/params.Ly)
    kmax = min(params.nx//2, params.ny//2)*kunit
    kbins, N2b, phi2b, cross_re, cross_im = _binned(
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
        m, s2 = shared_physics.eig2_ms(L00, L01, L10, L11)
        s = np.sqrt(complex(s2))
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


def theory_cross_phase(params, kz, ky_list, kx=0.0):
    # the THEORETICAL N/phi eigenvector ratio (dominant eigenvalue, exact -- not a nonlinear
    # measurement) vs ky at fixed kz: the clean way to see eq (4.6)<->(4.8)'s adiabaticity
    # transition, since a real run's cross_phase_spectrum can be dominated by nonlinear
    # mode-coupling long before/after the linear structure is cleanly resolved at every k.
    # Under the pure D_par*kz^2 closure (gpar_fac=0, the 3D default) gamma_par does NOT grow
    # with k_perp (unlike the 2D floor gpar_fac*nu_in*k_perp^2), so at fixed kz!=0 the
    # crossover runs the OPPOSITE way from the 2D-floor picture: ADIABATIC (small phase,
    # N~phi) at SMALL k_perp (gamma_par/k_perp^2 -> large there) and GDI-like (phase ->
    # 90deg) at LARGE k_perp (gamma_par/k_perp^2 -> 0) -- kperp_break's marginal-STABILITY
    # k_perp (eq 4.5's analog, where the GROWTH RATE crosses zero) is a genuinely different
    # scale from this adiabaticity crossover in the pure-D_par model.
    Ln, nu_in, v0, gpar_fac, diss, hyper, D_par = _eqpars(params)
    gamma_par_kz = D_par*kz*kz
    out = []
    for ky in ky_list:
        ksq = kx*kx + ky*ky
        inv_ksq = 0.0 if ksq == 0.0 else 1.0/ksq
        gamma_par = gpar_fac*nu_in*ksq + gamma_par_kz
        gpar_ratio = gpar_fac*nu_in + gamma_par_kz*inv_ksq
        L00, L01, L10, L11 = _L_entries(ksq, ky, inv_ksq, Ln, nu_in, v0,
                                        gamma_par, gpar_ratio, diss, hyper)
        m, s2 = shared_physics.eig2_ms(L00, L01, L10, L11)
        s = np.sqrt(complex(s2))
        lam1, lam2 = m + s, m - s
        lam = lam1 if lam1.real > lam2.real else lam2
        ratio = -L01/(L00 - lam)
        out.append(dict(ky=ky, growth=lam.real, amp_ratio=abs(ratio),
                        phase_deg=float(np.degrees(np.angle(ratio)))))
    return out
