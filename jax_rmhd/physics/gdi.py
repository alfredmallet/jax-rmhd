# 2D Gradient-Drift-Instability (GDI) equation set (plans/GDI_PLAN.md P4a), built on the
# P1 linear-propagator machinery. Physics source: docs/"GDI_nonlinear_equations (10).pdf".
# Normalization: rho_s = c_s = Omega_i = 1 (eqs 5.4-5.5); fields are (N, phi) in that
# order, N = delta n/n0, phi = e(electrostatic potential)/T_e. 2D closure (eq 4.3):
# gamma_par(k) = gpar_fac*nu_in*k_perp^2 replaces the 3D parallel-diffusion term
# D_par*d^2/dz^2 (kept until P4b, which adds the real kz axis).
#
# Linear matrix L (dt [N,phi] = L [N,phi] + NL), derived directly from eqs (5.1)-(5.2)
# with gamma_par substituted for D_par*d^2/dz^2, cross-checked against the exact
# quadratic dispersion relation (3.7)/(2.8) (tests/test_gdi_linear.py), and agreeing
# entry-by-entry with the paper's eq (5.3) matrix under L = -M (full derivation:
# docs/gdi_linear_matrix_note.tex):
#   L[N,N]     = -gamma_par - diss*k_perp^(2*hyper)
#   L[N,phi]   =  gamma_par + i*ky_deriv/Ln
#   L[phi,N]   =  gpar_fac*nu_in - i*ky_deriv*nu_in*v0*inv_ksq
#   L[phi,phi] = -nu_in - gpar_fac*nu_in - diss*k_perp^(2*hyper)
# ky_deriv is kgrid.ky with the Nyquist row zeroed: the "iky" cross terms are a genuine
# odd-order spectral derivative, which is ill-defined at the self-conjugate Nyquist row
# (same reason gradk's fields there only survive because dealiasing removes that row from
# every nonlinear/IC path -- L is applied unconditionally, with no dealiasing, so it needs
# the explicit fix to keep the rfft2 reality constraint the propagator setup checks for).
import functools

import jax.numpy as jnp
import numpy as np

from .. import comms, grids
from . import shared_physics
from .shared_physics import bracket, gradk


def _check_supported(params):
    # GDI is 2D-only until P4b, and instability-driven (no forcing term/mechanism).
    if params.spatial_dimensions != 2:
        raise NotImplementedError("GDI: dims==3 is not implemented until P4b (2D closure "
                                  "gpar_fac only; the parallel-diffusion axis is missing)")
    if params.forcing:
        raise ValueError("GDI is instability-driven (linear growth from Ln/v0): "
                         "forcing=True is not supported")


_EQPARS_REQUIRED = ("Ln", "nu_in", "v0", "diss", "hyper")
_EQPARS_OPTIONAL = ("gpar_fac", "lin_dt_safety")

def _eqpars(params):
    # pull (and validate) GDI's equation parameters out of params.eqpars (rmhd._diss_hyper
    # pattern). gpar_fac scales the eq (4.3) closure floor: gamma_par = gpar_fac*nu_in*k^2
    # (in eq (4.7)'s parametrization alpha = 1 + gpar_fac, so the default 1 gives alpha=2,
    # the minimum-gamma_par result). Unknown keys are rejected -- a typo'd optional key
    # would otherwise silently fall back to its default.
    _check_supported(params)
    missing = [k for k in _EQPARS_REQUIRED if k not in params.eqpars]
    unknown = [k for k in params.eqpars if k not in _EQPARS_REQUIRED + _EQPARS_OPTIONAL]
    if missing or unknown:
        raise ValueError(f"GDI eqpars problem (missing {missing}, unknown {unknown}): "
                         f"required {_EQPARS_REQUIRED}, optional {_EQPARS_OPTIONAL}; "
                         f"got eqpars={params.eqpars!r}")
    ep = params.eqpars
    return ep["Ln"], ep["nu_in"], ep["v0"], ep.get("gpar_fac", 1.0), ep["diss"], ep["hyper"]


def _lin_dt_safety(params):
    # safety factor on the dt ceiling from max|Re lambda(L)| (default: fairly conservative,
    # since it's an ACCURACY floor, not a stability one -- L is applied exactly)
    return float(params.eqpars.get("lin_dt_safety", 0.5))


def _L_entries(ksq, ky_deriv, inv_ksq, Ln, nu_in, v0, gpar_fac, diss, hyper):
    # the four (2,2) entries of L, shared between linear_matrix (jnp, kgrid-based) and
    # _max_re_lambda (numpy, setup-time only) so the two never drift apart.
    gpar_over_ksq = gpar_fac * nu_in            # = gamma_par/k_perp^2 (rho_s=1): k-independent
    gamma_par = gpar_over_ksq * ksq
    hyperdiss = diss * ksq**hyper
    L_NN = -gamma_par - hyperdiss
    L_Nphi = gamma_par + 1j*ky_deriv/Ln
    L_phiN = gpar_over_ksq - 1j*ky_deriv*nu_in*v0*inv_ksq
    L_phiphi = -nu_in - gpar_over_ksq - hyperdiss
    return L_NN, L_Nphi, L_phiN, L_phiphi


def linear_matrix(kgrid, params):
    # dt [N,phi] = L [N,phi] + NL(N,phi); shape (2,2,1,nkx,nky), z broadcast (perp-only op).
    Ln, nu_in, v0, gpar_fac, diss, hyper = _eqpars(params)
    ky_deriv = kgrid.ky.at[..., -1].set(0.0)   # zero the Nyquist row: see module docstring
    L00, L01, L10, L11 = _L_entries(kgrid.ksq, ky_deriv, kgrid.inv_ksq,
                                    Ln, nu_in, v0, gpar_fac, diss, hyper)
    L = jnp.stack([jnp.stack([L00, L01]), jnp.stack([L10, L11])])
    return L[:, :, None, :, :]


@functools.lru_cache(maxsize=32)   # bounded: params is identity-hashed, so an unbounded
def _max_re_lambda(params):        # cache would pin every Parameters in a long param scan
    # dt ceiling for nonlinear-vs-linear accuracy near saturation: max|Re lambda(L)|,
    # computed once (numpy, at trace time; params hashes by identity so lru_cache makes
    # this genuinely once-per-run, not once-per-jit-retrace), restricted to the DEALIASED
    # (2/3-rule) region: gamma_par = gpar_fac*nu_in*k_perp^2 grows without
    # bound past it (the electron-adiabaticity relaxation rate is genuinely k^2-fast --
    # real physics, not a bug), but those modes receive zero nonlinear forcing (run.initialize
    # masks the IC, NonlinearTerm masks every step), so the "misweight the nonlinear forcing
    # of a stiff mode" accuracy concern this ceiling exists for cannot apply there; the
    # propagator applies L there exactly regardless. Including them anyway would make dt
    # scale with resolution/diss for no accuracy benefit -- see the P4a report.
    Ln, nu_in, v0, gpar_fac, diss, hyper = _eqpars(params)
    kx = np.fft.fftfreq(params.nx) * params.nx * 2*np.pi/params.Lx
    ky = np.fft.rfftfreq(params.ny) * params.ny * 2*np.pi/params.Ly
    kx_grid, ky_grid = kx.reshape(-1, 1), ky.reshape(1, -1)
    ksq = kx_grid**2 + ky_grid**2
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_ksq = np.where(ksq > 0, 1.0/ksq, 0.0)
    ky_deriv = ky_grid.copy()
    ky_deriv[..., -1] = 0.0
    L00, L01, L10, L11 = _L_entries(ksq, ky_deriv, inv_ksq, Ln, nu_in, v0, gpar_fac, diss, hyper)
    m = 0.5*(L00 + L11)
    s2 = (m*m - (L00*L11 - L01*L10)).astype(complex)
    s = np.sqrt(s2)
    lam1, lam2 = m + s, m - s
    ix = np.fft.fftfreq(params.nx) * params.nx
    iy = np.fft.rfftfreq(params.ny) * params.ny
    dealias = ((ix.reshape(-1, 1)/(params.nx/3.0))**2 +
              (iy.reshape(1, -1)/(params.ny/3.0))**2) < 1.0
    re_max = np.maximum(np.abs(lam1.real), np.abs(lam2.real))
    return float(np.max(np.where(dealias, re_max, 0.0)))


def grad(state, kgrid, params):
    # gradients of phi, N, and vort=laplacian(phi) -- everything the two brackets need.
    Nk = state.fields[0]
    phik = state.fields[1]
    vortk = -kgrid.ksq*phik
    fk = jnp.stack([phik, Nk, vortk])
    return grids.ifft(gradk(fk, kgrid), params)


def set_timestep(grads, params):
    # ExB CFL from |grad phi| (mirrors rmhd.set_timestep) capped by the static linear-op
    # dt ceiling; L imposes no STABILITY limit (exact propagator), only an accuracy one.
    gphi, gN, gvort = grads
    max_vy = jnp.max(jnp.abs(gphi[0]))
    max_vx = jnp.max(jnp.abs(gphi[1]))
    eps = 0.1
    max_eps = jnp.maximum(eps/params.dx, eps/params.dy)
    max_all = jnp.maximum(max_vx/params.dx, max_vy/params.dy)
    max_all = jnp.maximum(max_all, max_eps)
    max_all = comms.allreduce_max(max_all, params)   # no-op in 2D; keeps the rmhd pattern
    dt_cfl = params.cfl_safety / max_all
    dt_lin = _lin_dt_safety(params) / _max_re_lambda(params)
    return jnp.minimum(dt_cfl, dt_lin)


def NonlinearTerm(state, grads, kgrid, params, halo=None):
    # dN/dt|_NL = -{phi,N}; the phi equation is the vorticity equation {phi,vort}-advection
    # divided by -k_perp^2 (inv_ksq, zero mode masked), same pattern as rmhd.NonlinearTerm.
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
    # dE/dt source/sink decomposition: eq (3.18)'s viscous term plus the 2D-closure analog
    # of the D_par(d_z(phi-N))^2 sink (-gamma_par*(N-phi)^2), PLUS the Ln/Pedersen drive
    # terms (eq 3.18 is only exactly zero-drive: sec 3.9 sets v0=0, 1/Ln=0) and the perp
    # hyperdiss sink -- the bracket/advection nonlinearity conserves E exactly (standard
    # property of {phi,.} terms under periodic BCs), so this decomposition of L accounts
    # for the WHOLE measured dE/dt of a nonlinear run (tests/test_gdi_linear.py).
    Ln, nu_in, v0, gpar_fac, diss, hyper = _eqpars(params)
    Nk, phik = state.fields[0], state.fields[1]
    ksq, yfac = kgrid.ksq, kgrid.yfac
    ky_deriv = kgrid.ky.at[..., -1].set(0.0)
    gamma_par = gpar_fac*nu_in*ksq
    cross_im = jnp.imag(jnp.conj(Nk)*phik) * yfac
    drive_Ln = shared_physics.perp_reduce(-(ky_deriv/Ln)*cross_im, params)
    drive_ped = shared_physics.perp_reduce(-ky_deriv*nu_in*v0*cross_im, params)
    visc = -nu_in*shared_physics.perp_inner_product(phik, phik, kgrid, params)
    diff2 = jnp.abs(Nk - phik)**2 * yfac
    gpar_term = shared_physics.perp_reduce(-gamma_par*diff2, params)
    hyper_N = shared_physics.perp_reduce(-diss*ksq**hyper*jnp.abs(Nk)**2*yfac, params)
    hyper_phi = shared_physics.perp_reduce(-diss*ksq**(hyper+1)*jnp.abs(phik)**2*yfac, params)
    total = drive_Ln + drive_ped + visc + gpar_term + hyper_N + hyper_phi
    return dict(drive_Ln=drive_Ln, drive_pedersen=drive_ped, viscous=visc,
               gamma_par=gpar_term, hyper_N=hyper_N, hyper_phi=hyper_phi, total=total)


def perp_spectrum(state, kgrid, params, bin_factor=2.0):
    # perpendicular density/potential spectra (density: |N|^2; potential: |grad_perp phi|^2,
    # same z-then-radial-bin convention as diagnostics.perpspec), z-averaged (trivial in 2D).
    from .. import diagnostics
    Nk, phik = state.fields[0], state.fields[1]
    N2 = jnp.real(jnp.conj(Nk)*Nk)*kgrid.yfac
    spec_N = shared_physics.perp_reduce(N2, params, axis=0)
    spec_phi = shared_physics.perp_reduce(0.5*shared_physics.perp_gradsq(phik, phik, kgrid), params, axis=0)
    kunit = min(2*jnp.pi/params.Lx, 2*jnp.pi/params.Ly)
    kmax = min(params.nx//2, params.ny//2)*kunit
    return diagnostics._binned(jnp.sqrt(kgrid.ksq), (spec_N, spec_phi), kunit, kmax, bin_factor)


def cross_phase_spectrum(state, kgrid, params, bin_factor=2.0):
    # N-phi cross-phase and amplitude ratio |N|/|phi| vs k_perp (eqs 4.6-4.8), binned like
    # diagnostics.perpspec (local import: avoids a module-load-order dependency on physics
    # being fully initialized, same rationale as grids._attach_linear_operator's).
    from .. import diagnostics
    Nk, phik = state.fields[0, 0], state.fields[1, 0]   # dims=2: drop the size-1 z axis
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
