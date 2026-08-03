# Gradient-Drift-Instability (GDI) equation set (plans/GDI_PLAN.md P4a 2D, P4b 3D), built
# on the P1 linear-propagator machinery. Physics source:
# docs/"GDI_nonlinear_equations (10).pdf". Normalization: rho_s = c_s = Omega_i = 1
# (eqs 5.4-5.5/3.5-3.6); fields are (N, phi) in that order, N = delta n/n0, phi =
# e(electrostatic potential)/T_e.
#
# Closure: eq (4.3)'s 2D floor gamma_par(k) = gpar_fac*nu_in*k_perp^2 (P4a) and the real
# 3D parallel-diffusion term gamma_par = D_par*kz^2 (eqs 3.5-3.7, P4b, z_spectral only)
# are ADDED: gamma_par_total = gpar_fac*nu_in*k_perp^2 + D_par*kz^2. Decision (P4b): the
# 2D floor is NOT retired in 3D but its default is flipped to OFF (gpar_fac=0) -- with a
# real kz axis resolved there is no sub-grid current-closure gap to patch, so the honest
# default is the real D_par*kz^2 term alone; gpar_fac stays available as an optional
# large-scale supplement (e.g. modeling closure below the lowest resolved |kz|) and
# nonzero gpar_fac still adds exactly the P4a floor on every kz plane, including kz=0.
# This additive form guarantees the REQUIRED consistency property: at kz=0 with gpar_fac=0
# (the 3D default), gamma_par_total==0 identically and dims==3's L collapses to EXACTLY
# the dims==2 L at gpar_fac=0 (tests/test_gdi_linear.py); with gpar_fac=1 (2D's default)
# the kz=0 plane instead matches the dims==2 model at gpar_fac=1.
#
# Linear matrix L (dt [N,phi] = L [N,phi] + NL), derived directly from eqs (5.1)-(5.2) /
# (3.5)-(3.6) with gamma_par_total substituted for the bare D_par*d^2/dz^2 term, cross-
# checked against the exact quadratic dispersion relation (3.7)/(2.8) and (2D-floor-only)
# against the paper's eq (5.3) matrix under L = -M (docs/gdi_linear_matrix_note.tex); the
# 3D quartic (3.11) and its named limits (2D 3.6, Hasegawa-Wakatani 3.12, nearly-adiabatic
# 3.9, stabilization boundary 3.15) are checked in tests/test_gdi_linear.py:
#   L[N,N]     = -gamma_par - diss*k_perp^(2*hyper)
#   L[N,phi]   =  gamma_par + i*ky_deriv/Ln
#   L[phi,N]   =  gamma_par*inv_ksq - i*ky_deriv*nu_in*v0*inv_ksq
#   L[phi,phi] = -nu_in - gamma_par*inv_ksq - diss*k_perp^(2*hyper)
# ky_deriv is kgrid.ky with the Nyquist row zeroed: the "iky" cross terms are a genuine
# odd-order spectral derivative, which is ill-defined at the self-conjugate Nyquist row
# (same reason gradk's fields there only survive because dealiasing removes that row from
# every nonlinear/IC path -- L is applied unconditionally, with no dealiasing, so it needs
# the explicit fix to keep the rfft2 reality constraint the propagator setup checks for).
# The [phi,N]/[phi,phi] entries divide gamma_par by k_perp^2 (the vorticity-equation
# division, same as the rest of the phi row): for the 2D floor this ratio is EXACTLY
# k-independent (gpar_fac*nu_in, no masking needed, matching P4a bit-for-bit) but for the
# 3D D_par*kz^2 term it is NOT k_perp-independent (diverges as k_perp->0 at kz!=0), so it
# is masked with inv_ksq (zero at k_perp=0) -- the same zero-mode convention used
# everywhere else in this file and in rmhd.py. gamma_par*kz^2 is EVEN in kz, so (unlike
# rmhd's +-i*kz off-diagonals) no kz-Nyquist fix is needed; propagators.
# _check_hermitian_compatible verifies this at every kgrid setup regardless.
#
# 3D requires params.z_spectral=True (raised at _check_supported): the parallel-diffusion
# axis needs a real kz grid (kgrid.kz), which only exists in that mode. dims==2 is
# completely unaffected by any of the above -- every 3D-only branch is gated on
# params.spatial_dimensions==3, which implies z_spectral (config.py enforces dims==3 for
# z_spectral=True) and never fires for dims==2.
#
# Production scheme (P4b): CB-IMEX (imexcb3e default -- plans/GDI_PLAN.md P3), which is
# L-stable and recovers the quasi-static N~phi balance that D_par*kz^2 (stiff by design,
# gamma_par/k_perp^2 -> D_par*kz^2/k_perp^2 -> infinity as k_perp->0) makes essential for
# any 3D run with an adaptive timestep. IF-LSRK is permitted only with a manually bounded
# fixed dt <~ 1/max(gamma_par/k_perp^2/rho_s^2) (the same quantity _max_re_lambda computes
# a ceiling from) -- GDI's L has no +-i*kz wave term (all-real growth/damping spectrum,
# unlike RMHD's Alfven coupling), so there is no wave-damping caveat against IMEX here,
# only the usual accuracy-not-stability role _max_re_lambda already plays for IF schemes.
#
# Sizing (plans/GDI_PLAN.md P4b): single GPU. At 512^3, fp32, one field register is
# ~0.54 GB (512^3*4 bytes); with nfields=2 and CB-IMEX's up to 4 live registers ([3R]
# imexcb3f) that is ~4*2*0.54 ~= 4.3 GB of field storage -- fits comfortably on a single
# 40-80 GB part with headroom for k-grid/propagator arrays and nonlinear-term temporaries.
import functools

import jax.numpy as jnp
import numpy as np

from .. import comms, grids
from . import shared_physics
from .shared_physics import bracket, gradk


def _check_supported(params):
    # GDI is instability-driven (no forcing term/mechanism); dims==3 needs a real kz axis
    # for the parallel-diffusion closure, i.e. params.z_spectral (P4b) -- the 2D floor
    # alone (gpar_fac) is not a 3D model.
    if params.spatial_dimensions == 3 and not params.z_spectral:
        raise NotImplementedError("GDI dims==3 requires z_spectral=True: the parallel "
                                  "closure gamma_par=D_par*kz^2 needs a real kz axis "
                                  "(kgrid.kz), which only exists in that mode -- the 2D "
                                  "closure floor (gpar_fac) alone is not a 3D model")
    if params.forcing:
        raise ValueError("GDI is instability-driven (linear growth from Ln/v0): "
                         "forcing=True is not supported")


_EQPARS_REQUIRED_2D = ("Ln", "nu_in", "v0", "diss", "hyper")
_EQPARS_REQUIRED_3D = _EQPARS_REQUIRED_2D + ("D_par",)
_EQPARS_OPTIONAL = ("gpar_fac", "lin_dt_safety")

def _eqpars(params):
    # pull (and validate) GDI's equation parameters out of params.eqpars (rmhd._diss_hyper
    # pattern). gpar_fac scales the eq (4.3) closure floor: gamma_par = gpar_fac*nu_in*k^2
    # (in eq (4.7)'s parametrization alpha = 1 + gpar_fac, so the 2D default 1 gives
    # alpha=2, the minimum-gamma_par result). D_par (3D only, required) scales the real
    # parallel closure gamma_par = D_par*kz^2 (eq 3.5-3.7); gpar_fac's 3D default is 0
    # (module docstring). Unknown keys are rejected -- a typo'd optional key would
    # otherwise silently fall back to its default, and D_par in a 2D run would silently
    # do nothing (there is no kz axis to apply it to).
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
    # safety factor on the dt ceiling from max|Re lambda(L)| (default: fairly conservative,
    # since it's an ACCURACY floor, not a stability one -- L is applied exactly)
    return float(params.eqpars.get("lin_dt_safety", 0.5))


def _closure_terms(ksq, inv_ksq, kz, gpar_fac, nu_in, D_par):
    # gamma_par(k) [feeds L_NN/L_Nphi and the energy_budget N-phi sink] and its ratio to
    # k_perp^2 [feeds L_phiN/L_phiphi] -- shared by linear_matrix (jnp), _max_re_lambda
    # (numpy) and energy_budget so the three views can never drift apart. kz=None: 2D
    # (module docstring's floor-only formula, bit-for-bit P4a); kz given (any shape
    # broadcastable against ksq with a leading kz axis, e.g. (nz,1,1)): adds the 3D
    # D_par*kz^2 term, masked by inv_ksq in the ratio (see module docstring) but not in
    # gamma_par itself (D_par*kz^2 is already k_perp-independent, nothing to mask there).
    gpar_ratio = gpar_fac * nu_in
    gamma_par = gpar_ratio * ksq
    if kz is not None:
        gamma_par_kz = D_par * kz**2
        gamma_par = gamma_par + gamma_par_kz
        gpar_ratio = gpar_ratio + gamma_par_kz*inv_ksq
    return gamma_par, gpar_ratio


def _L_entries(ksq, ky_deriv, inv_ksq, Ln, nu_in, v0, gamma_par, gpar_ratio, diss, hyper):
    # the four (2,2) entries of L given precomputed gamma_par/gpar_ratio (_closure_terms),
    # shared between linear_matrix (jnp, kgrid-based) and _max_re_lambda (numpy, setup-time
    # only) so the two never drift apart.
    hyperdiss = diss * ksq**hyper
    L_NN = -gamma_par - hyperdiss
    L_Nphi = gamma_par + 1j*ky_deriv/Ln
    L_phiN = gpar_ratio - 1j*ky_deriv*nu_in*v0*inv_ksq
    L_phiphi = -nu_in - gpar_ratio - hyperdiss
    return L_NN, L_Nphi, L_phiN, L_phiphi


def linear_matrix(kgrid, params):
    # dt [N,phi] = L [N,phi] + NL(N,phi). 2D: shape (2,2,1,nkx,nky), z broadcast (perp-only
    # op) -- bit-for-bit the P4a operator. 3D (z_spectral only): shape (2,2,nz,nkx,nky);
    # at the kz=0 plane with gpar_fac=0 (the 3D default) this is EXACTLY the 2D operator at
    # gpar_fac=0 -- required consistency check, tests/test_gdi_linear.py.
    Ln, nu_in, v0, gpar_fac, diss, hyper, D_par = _eqpars(params)
    ky_deriv = kgrid.ky.at[..., -1].set(0.0)   # zero the Nyquist row: see module docstring
    kz = kgrid.kz if params.spatial_dimensions == 3 else None
    gamma_par, gpar_ratio = _closure_terms(kgrid.ksq, kgrid.inv_ksq, kz, gpar_fac, nu_in, D_par)
    L00, L01, L10, L11 = _L_entries(kgrid.ksq, ky_deriv, kgrid.inv_ksq,
                                    Ln, nu_in, v0, gamma_par, gpar_ratio, diss, hyper)
    L = jnp.stack([jnp.stack([L00, L01]), jnp.stack([L10, L11])])
    return L if params.spatial_dimensions == 3 else L[:, :, None, :, :]


@functools.lru_cache(maxsize=32)   # bounded: params is identity-hashed, so an unbounded
def _max_re_lambda(params):        # cache would pin every Parameters in a long param scan
    # dt ceiling for nonlinear-vs-linear accuracy near saturation: the fastest GROWING
    # linear rate max(Re lambda(L), 0) over the DEALIASED region (2/3-rule, perp and -- in
    # 3D -- kz; beyond it modes receive zero nonlinear forcing, so no accuracy constraint
    # can originate there). Returns 0.0 when the operator is linearly stable everywhere --
    # set_timestep then applies no ceiling (CFL binds). Computed once (numpy, at trace
    # time; identity-hashed params + lru_cache = once per run), one kz plane at a time so
    # the setup cost stays O(nkx*nky) memory at any nz (review round 2, m2: the dense
    # (nz,nkx,nky) build OOM'd at the docstring's own 512^3 sizing).
    #
    # Growth-only, NOT max|Re lambda| (review round 2, M1): the stiffly DAMPED branch
    # (|Re lambda| ~ gamma_par*(1+ksq)/ksq, unbounded in D_par*kz^2) is handled
    # quasi-statically by the CB-IMEX production path BY DESIGN (P3's stiff test: u = g +
    # O(1/gamma) at gamma*dt >> 1), so resolving it buys nothing there and throttled 3D
    # runs ~10^2-10^5x. The flip side is deliberate and documented here: under an IF-LSRK
    # scheme stiffly damped modes ARE misweighted at gamma_par*dt >~ 1, and this ceiling no
    # longer protects against that -- IF schemes on 3D GDI require a hand-set dt <~
    # 1/max(gamma_par) (module docstring; plans/GDI_PLAN.md P4b "IF-LSRK permitted only
    # with documented dt"). 2D note: the old max|Re lambda| ceiling (e.g. 0.10 for the
    # gdi_2d_run weak-drive family) is gone too; there the damped branch is mild
    # (~gpar_fac*nu_in*(1+ksq), gamma*dt < 1 at any CFL-bound dt), so lsrk54 stays accurate.
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
        # fastest-growing SIGNED rate over one kz plane (kz_val=None: the 2D/no-kz case)
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
    # growth-rate-only ceiling (see _max_re_lambda); 0.0 = linearly stable = no ceiling.
    # _max_re_lambda(params) is a static python float, so plain `if` is correct here.
    gmax = _max_re_lambda(params)
    if gmax > 0.0:
        dt_cfl = jnp.minimum(dt_cfl, _lin_dt_safety(params)/gmax)
    return dt_cfl


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
    # dE/dt source/sink decomposition: eq (3.18)'s viscous term plus the D_par(d_z(phi-N))^2
    # sink (2D: its closure-floor analog -gamma_par*(N-phi)^2; 3D: the same expression with
    # gamma_par = D_par*kz^2 [+ gpar_fac's floor if nonzero] -- Parseval turns
    # D_par*(d_z(phi-N))^2 into D_par*kz^2*|phi-N|^2 per kz plane, exactly this term), PLUS
    # the Ln/Pedersen drive terms (eq 3.18 is only exactly zero-drive: sec 3.9 sets v0=0,
    # 1/Ln=0) and the perp hyperdiss sink -- the bracket/advection nonlinearity conserves E
    # exactly (standard property of {phi,.} terms under periodic BCs), so this decomposition
    # of L accounts for the WHOLE measured dE/dt of a nonlinear run (tests/test_gdi_linear.py).
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
    # exact per-row gamma_par transfer (review round 2, M2): the N row contributes
    # -gamma_par*Re(N*(N-phi)); the phi row's energy weight is ksq*gpar_ratio, which equals
    # gamma_par everywhere EXCEPT k_perp=0 (inv_ksq mask in _closure_terms), where phi
    # carries no energy and L's phi-row gamma_par is masked. Off k_perp=0 the two rows sum
    # to the familiar -gamma_par*|N-phi|^2; at k_perp=0 (kz!=0, 3D) only the N row acts --
    # the old -gamma_par*|N-phi|^2 form overcounted there (1e-2 closure error if such
    # modes are seeded; no shipped IC does, but the contract says exact).
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
    # perpendicular density/potential spectra (density: |N|^2; potential: |grad_perp phi|^2,
    # same z-then-radial-bin convention as diagnostics.perpspec), z-averaged (trivial in 2D;
    # in 3D perp_reduce's axis=0 Parseval-sums over kz, dividing by the shared nz^2 -- no
    # code change needed, same as diagnostics.perpspec under z_spectral).
    from .. import diagnostics
    Nk, phik = state.fields[0], state.fields[1]
    N2 = jnp.real(jnp.conj(Nk)*Nk)*kgrid.yfac
    spec_N = shared_physics.perp_reduce(N2, params, axis=0)
    spec_phi = shared_physics.perp_reduce(0.5*shared_physics.perp_gradsq(phik, phik, kgrid), params, axis=0)
    kunit = min(2*jnp.pi/params.Lx, 2*jnp.pi/params.Ly)
    kmax = min(params.nx//2, params.ny//2)*kunit
    return diagnostics._binned(jnp.sqrt(kgrid.ksq), (spec_N, spec_phi), kunit, kmax, bin_factor)


def cross_phase_spectrum(state, kgrid, params, bin_factor=2.0, kz_index=0):
    # N-phi cross-phase and amplitude ratio |N|/|phi| vs k_perp (eqs 4.6-4.8), binned like
    # diagnostics.perpspec (local import: avoids a module-load-order dependency on physics
    # being fully initialized, same rationale as grids._attach_linear_operator's).
    # kz_index selects a single kz PLANE (default 0, the only choice in 2D -- unchanged
    # behavior there); in 3D (z_spectral) axis 1 of state.fields is kz, so kz_index=0 picks
    # the kz=0 plane (== the 2D-limit cross-phase, gamma_par floor only) while a nonzero
    # index picks a plane with a finite D_par*kz^2 closure, showing the eq (4.6)->(4.8),
    # 180deg->90deg transition the real parallel closure drives (module docstring).
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
    # marginal-stability k_perp (the eq 4.5 "break" location), computed EXACTLY from the
    # linear dispersion relation itself (numpy bisection on max Re(lambda(L))) rather than
    # the paper's approximate closed form -- eq (4.5) is explicitly hedged in the source
    # ("this argument could be formalized...", valid only at k_perp^2*rho_s^2<<1 under the
    # eq (4.3) floor) and its self-referential algebra is easy to mistranscribe from a PDF
    # (the CLAUDE.md-flagged P4a lesson); bisecting the actual L avoids that risk entirely
    # and works for the real D_par*kz^2 closure too, not just the floor. Along the fastest-
    # growing direction (kx=0, section 2.7 "growth rate maximized for k_perp in the y
    # direction"): bisects ky in [ky_lo, ky_hi] for the sign change of max Re(eigenvalue),
    # at the given real kz (module docstring's gamma_par = gpar_fac*nu_in*k_perp^2 +
    # D_par*kz^2). Returns None if [ky_lo, ky_hi] does not bracket a single unstable-to-
    # stable crossing (e.g. stable everywhere, or kz so large the whole band is damped).
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
    # eq (4.7)'s alpha, measured mode-by-mode from an EVOLVED state's N/phi ratio (purely
    # imaginary in the GDI/small-k regime, eq 4.6-4.7: N = i*ky*(k_perp^2/ky^2)*(alpha/v0)*phi)
    # at the given kz plane and (ikx,iky) index list, vs the linear prediction alpha =
    # 1 + gamma_par/(nu_in*k_perp^2) (review-round-1 NOTE: this is 1+gpar_fac in the 2D
    # floor-only case, NOT gpar_fac itself -- generalized here to the full 3D gamma_par).
    # `modes` is an explicit [(ikx,iky), ...] list (not radially binned like perp_spectrum/
    # cross_phase_spectrum): alpha depends on ky specifically, not just k_perp magnitude, so
    # binning by |k_perp| would mix modes with different theoretical alpha.
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
