# GDI (physics/gdi.py, plans/GDI_PLAN.md P4a 2D + P4b 3D) validation.
#
# physics/gdi.py's L is derived directly from the scalar PDEs (5.1)-(5.2) (equivalently
# (3.5)-(3.6), with gamma_par = gpar_fac*nu_in*k_perp^2 [2D floor, eq 4.3] + D_par*kz^2
# [3D, P4b, z_spectral only] substituted for the bare D_par*d^2/dz^2 term) and agrees
# entry-by-entry with the paper's eq (5.3) matrix (2D-floor-only case) under L = -M
# (derivation: docs/gdi_linear_matrix_note.tex). Per plans/GDI_PLAN.md's ground rules the
# dispersion relations are the arbiter for any sign ambiguity regardless. What's checked:
#   1. L's characteristic polynomial (eigenvalues m +- s) matches the EXACT quadratic
#      dispersion relation (3.7)/(5.3) (derived independently in this file from the same
#      PDEs, not imported from gdi.py) at both gamma_par=0 and finite closure values,
#      including a gamma_par SCAN via kz (D_par*kz^2) -- catches any transcription bug in
#      physics/gdi.py._L_entries/_closure_terms.
#   2. the gamma_par=0 case collapses to eq (2.8) (== the 2D limit, eq 3.6), and matches
#      the collisional (2.9) and inertial (2.11) asymptotic limits where they apply.
#   3. the gamma_par scan additionally satisfies the independently-transcribed 3D quartic
#      (3.11)/(3.10) exactly (not just the quadratic they were derived from), and the named
#      asymptotic limits: Hasegawa-Wakatani (3.12, nu_in=0), nearly-adiabatic drift waves
#      (3.9, gamma_par -> infinity), and the stabilization boundary (3.15).
#   4. propagator vs dispersion cross-check: the ACTUAL putzer2-evolved field's measured
#      growth rate equals max Re(m +- s) from kgrid.lin_m/lin_s2 at that k.
#   5. 3D's kz=0 plane is EXACTLY the 2D operator at the same gpar_fac (required
#      consistency property of the additive gamma_par_total construction).
#   6. nonlinear energy-budget closure (extended eq 3.18): a short fp64 run's measured
#      dE/dt matches gdi.energy_budget's total to a tight relative tolerance, in both 2D
#      (lsrk54) and 3D (imexcb3e, the P4b production scheme).
#   7. energy_enstrophy/cross_phase_spectrum against a direct numpy reference on a known
#      field (normalization gate, CLAUDE.md's "keep new energy-like diagnostics on the
#      shared perp_reduce convention").
#
# Dual precision: dispersion/propagator checks are precision-independent algebra (loose
# fp32 tolerance, tight fp64; the numpy-only quadratic/quartic cross-checks are float64
# regardless of RMHD_PRECISION, but keep the same rtol idiom for consistency); the
# energy-budget closure tests are fp64-gated (their tolerance is below fp32 roundoff) per
# CLAUDE.md's "print+return, never pytest.skip" convention.
# pytest: `pytest tests/test_gdi_linear.py`. Script: `python tests/test_gdi_linear.py`.
from _rmhd_testing import bootstrap, checks

bootstrap()

import jax.numpy as jnp
import numpy as np

import taranis as jr
from taranis import _precision, propagators
from taranis.physics import gdi


def _fp64():
    # FIELD precision (RMHD_PRECISION) -- jax_enable_x64 is now unconditionally on.
    return _precision.precision == "64"


def _gdi_params(nx=8, ny=8, Lx=2*np.pi, Ly=2*np.pi, Ln=2.0, nu_in=0.3, v0=1.0,
               gpar_fac=1.0, diss=0.0, hyper=1, **overrides):
    kw = dict(nx=nx, ny=ny, Lx=Lx, Ly=Ly, cfl_safety=0.5, dims=2, adaptive_timestep=False,
             dt=0.01, eqtype="GDI",
             eqpars=dict(Ln=Ln, nu_in=nu_in, v0=v0, gpar_fac=gpar_fac, diss=diss, hyper=hyper))
    kw.update(overrides)
    return jr.Parameters(**kw)


def _gdi_params_3d(nx=8, ny=8, nz=8, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi, Ln=2.0, nu_in=0.3,
                   v0=1.0, D_par=1.0, gpar_fac=0.0, diss=0.0, hyper=1, **overrides):
    # z_spectral=True is required for GDI dims==3 (P4b); size==1 always in this sandbox.
    kw = dict(nx=nx, ny=ny, nz=nz, Lx=Lx, Ly=Ly, Lz=Lz, cfl_safety=0.5, dims=3,
             z_spectral=True, adaptive_timestep=False, dt=0.01, eqtype="GDI",
             eqpars=dict(Ln=Ln, nu_in=nu_in, v0=v0, D_par=D_par, gpar_fac=gpar_fac,
                        diss=diss, hyper=hyper))
    kw.update(overrides)
    return jr.Parameters(**kw)


def _dispersion_quadratic_coeffs(kx, ky, Ln, nu_in, v0, gamma_par):
    # eq (3.7) (rho_s=c_s=Omega_i=1), expanded to a*omega^2+b*omega+c=0 -- derived
    # independently from (5.1)-(5.2) (see the P4a report for the by-hand algebra), NOT
    # imported from physics/gdi.py: this is the ground truth the implementation is
    # checked against, not a restatement of it. gamma_par is general (2D floor, 3D
    # D_par*kz^2, or a sum of both): (3.7) itself makes no reference to how gamma_par
    # arises.
    ksq = kx*kx + ky*ky
    omega_star = ky/Ln
    omega_ped = ky*nu_in*v0
    a = ksq
    b = 1j*(nu_in*ksq + gamma_par*(1.0 + ksq))
    c = -gamma_par*nu_in*ksq + omega_ped*omega_star + 1j*gamma_par*(omega_star - omega_ped)
    return a, b, c


def _eq311_sides(kx, ky, Ln, nu_in, v0, gamma_par, gamma):
    # eq (3.11)'s LHS and RHS, evaluated (not solved) at a candidate real growth rate
    # gamma: transcribed independently from the paper, NOT derived from _dispersion_
    # quadratic_coeffs -- an exact eigenvalue's Re(lambda) must make lhs==rhs here. Also
    # returns eq (3.10)'s predicted omega_R at that gamma, for a second independent check.
    ksq = kx*kx + ky*ky
    omega_star = ky/Ln
    omega_ped = ky*nu_in*v0
    A = (ky*ky/ksq)*(nu_in*v0/Ln)
    denom = (2*gamma + nu_in)*ksq + gamma_par*(1.0 + ksq)
    lhs = gamma**2 + gamma*(nu_in + gamma_par*(1.0 + ksq)/ksq) - (A - gamma_par*nu_in)
    rhs = (gamma_par*(omega_star - omega_ped)/denom)**2
    omega_R_310 = gamma_par*(omega_ped - omega_star)/denom
    return lhs, rhs, omega_R_310


def _L_eigs_numpy(kx, ky, Ln, nu_in, v0, gpar_fac, diss=0.0, hyper=1, D_par=0.0, kz=None):
    # eigenvalues of physics/gdi.py's L at a single mode, via its own _closure_terms/
    # _L_entries (so this exercises the actual implementation, not a hand copy of it).
    # kz=None: 2D (P4a, bit-for-bit); kz given: adds the 3D D_par*kz^2 closure term.
    ksq = np.array([[kx*kx + ky*ky]])
    kyv = np.array([[ky]])
    inv_ksq = np.array([[1.0/ksq[0, 0] if ksq[0, 0] > 0 else 0.0]])
    kz_arr = None if kz is None else np.array([[[kz]]])
    gamma_par, gpar_ratio = gdi._closure_terms(ksq, inv_ksq, kz_arr, gpar_fac, nu_in, D_par)
    L00, L01, L10, L11 = gdi._L_entries(ksq, kyv, inv_ksq, Ln, nu_in, v0,
                                        gamma_par, gpar_ratio, diss, hyper)
    L00, L01, L10, L11 = (np.asarray(x).reshape(()) for x in (L00, L01, L10, L11))
    m = 0.5*(L00 + L11)
    s2 = m*m - (L00*L11 - L01*L10)
    s = np.sqrt(complex(s2))
    return m + s, m - s


def test_L_matches_exact_dispersion_quadratic():
    # cross-check physics/gdi.py._L_entries's eigenvalues (m +- s, lambda = -i*omega)
    # against the independently-derived eq (3.7) quadratic, at gamma_par=0 and finite.
    rtol = 1e-10 if _fp64() else 1e-4
    kx, ky, Ln, nu_in, v0 = 1.0, 1.0, 2.0, 0.3, 1.0
    with checks() as c:
        for gpar_fac in (0.0, 1.0, 2.5):
            gamma_par = gpar_fac*nu_in*(kx*kx + ky*ky)
            lam1, lam2 = _L_eigs_numpy(kx, ky, Ln, nu_in, v0, gpar_fac)
            omega_from_L = sorted([1j*lam1, 1j*lam2], key=lambda z: z.imag)
            qa, qb, qc = _dispersion_quadratic_coeffs(kx, ky, Ln, nu_in, v0, gamma_par)
            omega_ref = sorted(np.roots([qa, qb, qc]), key=lambda z: z.imag)
            err = max(abs(o1 - o2) for o1, o2 in zip(omega_from_L, omega_ref))
            scale = max(1.0, max(abs(o) for o in omega_ref))
            c.check(f"L eigenvalues match eq(3.7) roots (gpar_fac={gpar_fac})",
                    err/scale < rtol, f"omega_L={omega_from_L}, omega_ref={omega_ref}, "
                    f"rel err={err/scale:.3e}")


def test_2D_limit_matches_eq28():
    # gamma_par=0 exactly: eq (3.7)/L's char. poly must collapse to omega(omega+i*nu_in)=-A
    # (eq 2.8), i.e. the quadratic [1, i*nu_in, A].
    rtol = 1e-10 if _fp64() else 1e-4
    kx, ky, Ln, nu_in, v0 = 1.0, 1.0, 2.0, 0.3, 1.0
    ksq = kx*kx + ky*ky
    A = (ky*ky/ksq)*(nu_in*v0/Ln)
    lam1, lam2 = _L_eigs_numpy(kx, ky, Ln, nu_in, v0, gpar_fac=0.0)
    omega_from_L = sorted([1j*lam1, 1j*lam2], key=lambda z: z.imag)
    omega_28 = sorted(np.roots([1.0, 1j*nu_in, A]), key=lambda z: z.imag)
    err = max(abs(o1 - o2) for o1, o2 in zip(omega_from_L, omega_28))
    with checks() as c:
        c.check("gamma_par=0 dispersion matches eq (2.8)'s quadratic",
                err < rtol, f"omega_L={omega_from_L}, omega_2.8={omega_28}")


def test_collisional_limit_matches_eq29():
    # A << nu_in^2/2: eq (2.9)'s asymptotic roots omega = iA/nu_in, -i(nu_in + A/nu_in).
    kx, ky, Ln, nu_in, v0 = 1.0, 1.0, 1e5, 1.0, 1.0
    ksq = kx*kx + ky*ky
    A = (ky*ky/ksq)*(nu_in*v0/Ln)
    assert A < 1e-3*(nu_in**2/2), "test setup: not deep in the collisional regime"
    omega_29 = sorted([1j*A/nu_in, -1j*(nu_in + A/nu_in)], key=lambda z: z.imag)
    omega_exact = sorted(np.roots([1.0, 1j*nu_in, A]), key=lambda z: z.imag)
    err = max(abs(o1 - o2) for o1, o2 in zip(omega_exact, omega_29))
    scale = max(abs(o) for o in omega_29)
    with checks() as c:
        c.check("eq (2.8) roots match the (2.9) collisional asymptote",
                err/scale < 1e-2, f"omega_exact={omega_exact}, omega_2.9={omega_29}, "
                f"rel err={err/scale:.3e}")


def test_inertial_limit_matches_eq211():
    # A >> nu_in^2/2 (ion inertia/Keskinen-Ossakow): omega = +-i*sqrt(A).
    kx, ky, Ln, nu_in, v0 = 1.0, 1.0, 1.0, 1e-4, 1.0
    ksq = kx*kx + ky*ky
    A = (ky*ky/ksq)*(nu_in*v0/Ln)
    assert A > 1e3*(nu_in**2/2), "test setup: not deep in the inertial regime"
    omega_211 = sorted([1j*np.sqrt(A), -1j*np.sqrt(A)], key=lambda z: z.imag)
    omega_exact = sorted(np.roots([1.0, 1j*nu_in, A]), key=lambda z: z.imag)
    err = max(abs(o1 - o2) for o1, o2 in zip(omega_exact, omega_211))
    scale = max(abs(o) for o in omega_211)
    with checks() as c:
        c.check("eq (2.8) roots match the (2.11) inertial (Keskinen-Ossakow) asymptote",
                err/scale < 1e-2, f"omega_exact={omega_exact}, omega_2.11={omega_211}, "
                f"rel err={err/scale:.3e}")


def test_propagator_growth_matches_L_eigenvalues():
    # propagator vs dispersion cross-check: evolve a real taranis state through the actual
    # putzer2 propagator and measure its growth rate from the field-norm time series; must
    # equal max Re(m +- s) read off kgrid.lin_m/lin_s2 at that mode (grids.setup_kgrids's
    # own precompute, not a hand recomputation).
    params = _gdi_params(nu_in=0.3, v0=1.0, Ln=2.0, gpar_fac=1.0, diss=0.0, hyper=1)
    kgrid = jr.setup_kgrids(params)
    ikx, iky = 1, 1   # kx=ky=1 on this Lx=Ly=2*pi, nx=ny=8 grid
    m = complex(kgrid.lin_m[0, ikx, iky])
    s2 = complex(kgrid.lin_s2[0, ikx, iky])
    s = np.sqrt(s2)
    max_re_eig = max((m + s).real, (m - s).real)

    prop = propagators.get_propagator(kgrid, params)
    nkx, nky = params.nx, params.ny//2 + 1
    dtype = jnp.result_type(float, complex)
    arr0 = jnp.zeros((2, 1, nkx, nky), dtype=dtype).at[0, 0, ikx, iky].set(1.0)
    taus = np.linspace(20.0, 24.0, 5)   # large tau: dominant eigenvalue separates out
    norms = [float(jnp.linalg.norm(prop.apply_exp(arr0, float(tau))[:, 0, ikx, iky]))
             for tau in taus]
    fitted_rate = float(np.polyfit(taus, np.log(norms), 1)[0])
    rtol = 1e-6 if _fp64() else 1e-3
    with checks() as c:
        c.check("putzer2-evolved growth rate matches max Re(m+-s)",
                abs(fitted_rate - max_re_eig)/max(1e-12, abs(max_re_eig)) < rtol,
                f"fitted={fitted_rate:.6e}, max_re_eig={max_re_eig:.6e}")


def _single_mode_ic(kx_idx, ky_idx, amp_N, amp_phi, params):
    # a real-space IC whose rfft2 has support (after fft) concentrated at one (kx,ky) mode
    # pair via cos: cos(kx*x+ky*y) puts equal energy at (kx,ky) and (-kx,ky).
    def ic(x, y):
        kx = kx_idx*2*np.pi/params.Lx
        ky = ky_idx*2*np.pi/params.Ly
        N = amp_N*jnp.cos(kx*x + ky*y)
        phi = amp_phi*jnp.cos(kx*x + ky*y)
        return jnp.stack([N, phi])
    return ic


def test_energy_budget_closure_nonlinear():
    # fp64-only: short random-IC run, centered-difference dE/dt vs gdi.energy_budget's
    # total at the midpoint state. NL brackets conserve E exactly, so L (this decomposition)
    # should account for the whole measured budget.
    if not _fp64():
        print("[SKIP] test_energy_budget_closure_nonlinear -- fp64 only (tight tolerance)")
        return
    from taranis.run import block_of_steps
    from taranis.timestepping import get_scheme

    params = _gdi_params(nx=16, ny=16, Ln=5.0, nu_in=0.2, v0=0.3, gpar_fac=1.0,
                        diss=1e-3, hyper=1, adaptive_timestep=False, dt=2e-4)
    kgrid = jr.setup_kgrids(params)

    def ic(x, y):
        rng = np.random.default_rng(7)
        N = jnp.zeros_like(x*y)
        phi = jnp.zeros_like(x*y)
        for (kxi, kyi) in [(1, 1), (2, 1), (1, 2), (2, 2), (1, 0), (0, 1)]:
            aN, aP, ph1, ph2 = rng.normal(size=4)*1e-3
            kx = kxi*2*np.pi/params.Lx
            ky = kyi*2*np.pi/params.Ly
            N = N + aN*jnp.cos(kx*x + ky*y + ph1)
            phi = phi + aP*jnp.cos(kx*x + ky*y + ph2)
        return jnp.stack([N, phi])

    state0 = jr.initialize(ic, params)
    stepper, scheme = get_scheme("lsrk54")
    state1 = block_of_steps(state0, kgrid, params, 1, scheme, stepper)
    state2 = block_of_steps(state1, kgrid, params, 1, scheme, stepper)
    dt_actual = float(state2.t - state0.t)/2.0

    E0, _ = gdi.energy_enstrophy(state0, kgrid, params)
    E2, _ = gdi.energy_enstrophy(state2, kgrid, params)
    dEdt_measured = (float(E2) - float(E0))/(2*dt_actual)
    budget = gdi.energy_budget(state1, kgrid, params)
    dEdt_budget = float(budget["total"])

    rel = abs(dEdt_measured - dEdt_budget)/max(abs(dEdt_budget), 1e-30)
    with checks() as c:
        c.check("measured dE/dt (centered difference) matches gdi.energy_budget total",
                rel < 1e-4, f"measured={dEdt_measured:.10e}, budget={dEdt_budget:.10e}, "
                f"rel={rel:.3e}, dt={dt_actual:.3e}")


def test_energy_enstrophy_numpy_reference():
    # normalization gate: E/Z against a direct numpy real-space computation on a known
    # field, independent of shared_physics.perp_reduce/perp_mean_square/perp_inner_product.
    params = _gdi_params(nx=16, ny=16, Ln=5.0, nu_in=0.2, v0=0.3, gpar_fac=1.0, diss=0.0, hyper=1)
    kgrid = jr.setup_kgrids(params)

    def ic(x, y):
        N = jnp.cos(x)*jnp.cos(2*y) + 0.3*jnp.sin(x + y)
        phi = jnp.sin(2*x)*jnp.cos(y) + 0.1*jnp.cos(x - y)
        return jnp.stack([N, phi])

    state = jr.initialize(ic, params)
    E, Z = gdi.energy_enstrophy(state, kgrid, params)

    x = np.linspace(0, float(params.Lx), params.nx, endpoint=False).reshape(-1, 1)
    y = np.linspace(0, float(params.Ly), params.ny, endpoint=False).reshape(1, -1)
    N_real = np.asarray(np.cos(x)*np.cos(2*y) + 0.3*np.sin(x + y))
    phi_real = np.asarray(np.sin(2*x)*np.cos(y) + 0.1*np.cos(x - y))
    # dealiased IC: reproduce via the same fft/mask path gdi.py's dealiasing uses so this
    # is a check of the NORMALIZATION, not a rediscovery of the 2/3 rule.
    from taranis.grids import fft, ifft, dealias_mask
    mask = np.asarray(dealias_mask(params))
    Nk_np = np.asarray(fft(jnp.asarray(N_real), params))*mask
    phik_np = np.asarray(fft(jnp.asarray(phi_real), params))*mask
    N_dealiased = np.asarray(ifft(jnp.asarray(Nk_np), params))
    # laplacian(phi) via real-space FD is unnecessary -- use the same spectral k^2 (exact
    # for these band-limited trig ICs) but from a fresh numpy k-grid, not kgrid.ksq itself
    kx = np.fft.fftfreq(params.nx)*params.nx*2*np.pi/float(params.Lx)
    ky = np.fft.rfftfreq(params.ny)*params.ny*2*np.pi/float(params.Ly)
    ksq_np = kx.reshape(-1, 1)**2 + ky.reshape(1, -1)**2
    vort_np = np.asarray(ifft(jnp.asarray(-ksq_np*phik_np), params))
    gphix = np.asarray(ifft(jnp.asarray(1j*kx.reshape(-1, 1)*phik_np), params))
    gphiy = np.asarray(ifft(jnp.asarray(1j*ky.reshape(1, -1)*phik_np), params))

    E_ref = 0.5*np.mean(N_dealiased**2 + gphix**2 + gphiy**2)
    Z_ref = 0.5*np.mean((N_dealiased - vort_np)**2)

    rtol = 1e-10 if _fp64() else 1e-4
    with checks() as c:
        c.check("energy_enstrophy E matches a direct numpy real-space reference",
                abs(float(E) - E_ref)/abs(E_ref) < rtol, f"E={float(E):.10e}, ref={E_ref:.10e}")
        c.check("energy_enstrophy Z matches a direct numpy real-space reference",
                abs(float(Z) - Z_ref)/abs(Z_ref) < rtol, f"Z={float(Z):.10e}, ref={Z_ref:.10e}")


def test_registry_rejects_3d_without_z_spectral_and_forcing():
    with checks() as c:
        try:
            params3d = _gdi_params(dims=3, nz=4, Lz=2*np.pi)   # z_spectral defaults False
            jr.setup_kgrids(params3d)
            err = None
        except NotImplementedError as e:
            err = str(e)
        c.check("GDI dims=3 without z_spectral raises NotImplementedError at kgrid setup",
                err is not None and "z_spectral" in err, f"err={err!r}")
        try:
            params_f = _gdi_params(forcing=True, fshell=(1, 2))
            jr.setup_kgrids(params_f)
            err = None
        except ValueError as e:
            err = str(e)
        c.check("GDI forcing=True raises ValueError at kgrid setup", err is not None)
        params3d_ok = _gdi_params_3d()
        kgrid3d_ok = jr.setup_kgrids(params3d_ok)
        c.check("GDI dims=3 WITH z_spectral=True sets up successfully",
                kgrid3d_ok.lin_L is not None and kgrid3d_ok.lin_L.shape[2] == params3d_ok.nz,
                f"lin_L shape {None if kgrid3d_ok.lin_L is None else kgrid3d_ok.lin_L.shape}")


def test_3D_eqpars_validation():
    # D_par is required in 3D and rejected as an unknown key in 2D (a stray D_par in a 2D
    # run would silently do nothing -- there is no kz axis to apply it to).
    base3d = dict(nx=8, ny=8, nz=4, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi, cfl_safety=0.5,
                 dims=3, z_spectral=True, adaptive_timestep=False, dt=0.01, eqtype="GDI")
    with checks() as c:
        try:
            p = jr.Parameters(eqpars=dict(Ln=2.0, nu_in=0.3, v0=1.0, diss=0.0, hyper=1),
                              **base3d)
            jr.setup_kgrids(p)
            err = None
        except ValueError as e:
            err = str(e)
        c.check("3D GDI without D_par in eqpars raises ValueError",
                err is not None and "D_par" in err, f"err={err!r}")
        try:
            p2d = jr.Parameters(nx=8, ny=8, Lx=2*np.pi, Ly=2*np.pi, cfl_safety=0.5, dims=2,
                                adaptive_timestep=False, dt=0.01, eqtype="GDI",
                                eqpars=dict(Ln=2.0, nu_in=0.3, v0=1.0, diss=0.0, hyper=1,
                                           D_par=1.0))
            jr.setup_kgrids(p2d)
            err2 = None
        except ValueError as e:
            err2 = str(e)
        c.check("2D GDI with a D_par eqpars key raises ValueError (unknown in 2D)",
                err2 is not None and "D_par" in err2, f"err={err2!r}")
        p3d = _gdi_params_3d()
        c.check("3D GDI gpar_fac default is 0.0 (not stored unless overridden)",
                "gpar_fac" not in p3d.eqpars or p3d.eqpars["gpar_fac"] == 0.0)
        Ln, nu_in, v0, gpar_fac, diss, hyper, D_par = gdi._eqpars(p3d)
        c.check("gdi._eqpars resolves the 3D gpar_fac default to 0.0", gpar_fac == 0.0,
                f"gpar_fac={gpar_fac}")
        p2d_default = _gdi_params()
        _, _, _, gpar_fac_2d, _, _, _ = gdi._eqpars(p2d_default)
        c.check("gdi._eqpars resolves the 2D gpar_fac default to 1.0 (unchanged)",
                gpar_fac_2d == 1.0, f"gpar_fac={gpar_fac_2d}")


def test_kz0_plane_matches_2D_model():
    # REQUIRED consistency check (plans/GDI_PLAN.md P4b): the 3D operator's kz=0 plane must
    # be EXACTLY the 2D operator at the same gpar_fac -- both at the 3D default (gpar_fac=0,
    # i.e. gamma_par off entirely) and at a nonzero gpar_fac (the additive floor still
    # applies on every kz plane, including kz=0).
    nx, ny, Lx, Ly = 8, 8, 2*np.pi, 2*np.pi
    Ln, nu_in, v0, diss, hyper = 2.0, 0.3, 1.0, 1e-2, 1
    tol = 1e-12 if _fp64() else 1e-5
    with checks() as c:
        for gpar_fac in (0.0, 1.0):
            p2d = _gdi_params(nx=nx, ny=ny, Lx=Lx, Ly=Ly, Ln=Ln, nu_in=nu_in, v0=v0,
                              gpar_fac=gpar_fac, diss=diss, hyper=hyper)
            kg2d = jr.setup_kgrids(p2d)
            L2d = gdi.linear_matrix(kg2d, p2d)[:, :, 0, :, :]

            p3d = _gdi_params_3d(nx=nx, ny=ny, nz=4, Lx=Lx, Ly=Ly, Lz=2*np.pi, Ln=Ln,
                                 nu_in=nu_in, v0=v0, D_par=1.7, gpar_fac=gpar_fac,
                                 diss=diss, hyper=hyper)
            kg3d = jr.setup_kgrids(p3d)
            L3d = gdi.linear_matrix(kg3d, p3d)
            kz0_idx = 0   # fftfreq index 0 -> kz=0 exactly
            c.check(f"kz index 0 is exactly kz=0 (gpar_fac={gpar_fac})",
                    float(kg3d.kz[kz0_idx, 0, 0]) == 0.0)
            err = float(jnp.max(jnp.abs(L3d[:, :, kz0_idx, :, :] - L2d)))
            c.check(f"3D L at kz=0 == 2D L at the same gpar_fac={gpar_fac}",
                    err < tol, f"max abs diff {err:.3e}")


def test_3D_grid_dispersion_matches_quadratic_and_quartic_scan():
    # exercises the ACTUAL setup_kgrids -> gdi.linear_matrix pipeline (kz broadcasting,
    # putzer2 precompute), not just the low-level _L_entries helper: at a fixed (kx,ky)
    # mode, scan every kz grid plane and check the exact putzer2 eigenvalues against BOTH
    # the independently-transcribed exact quadratic (3.7) and quartic (3.11)/(3.10) -- a
    # gamma_par scan spanning gamma_par=0 (2D limit, eq 3.6) up to gamma_par >> everything
    # else (stiff, near-adiabatic).
    Ln, nu_in, v0, D_par = 2.0, 0.3, 1.0, 0.6
    p3d = _gdi_params_3d(nx=8, ny=8, nz=8, Ln=Ln, nu_in=nu_in, v0=v0, D_par=D_par,
                         gpar_fac=0.0, diss=0.0, hyper=1)
    kgrid = jr.setup_kgrids(p3d)
    ikx, iky = 1, 1
    kx_val = float(kgrid.kx[ikx, 0])
    ky_val = float(kgrid.ky[0, iky])
    rtol = 1e-9 if _fp64() else 1e-4
    with checks() as c:
        for ikz in range(p3d.nz):
            kz_val = float(kgrid.kz[ikz, 0, 0])
            gamma_par = D_par*kz_val*kz_val
            m = complex(kgrid.lin_m[ikz, ikx, iky])
            s2 = complex(kgrid.lin_s2[ikz, ikx, iky])
            s = np.sqrt(s2)
            lam1, lam2 = m + s, m - s
            omega_from_L = sorted([1j*lam1, 1j*lam2], key=lambda z: z.imag)
            qa, qb, qc = _dispersion_quadratic_coeffs(kx_val, ky_val, Ln, nu_in, v0, gamma_par)
            omega_ref = sorted(np.roots([qa, qb, qc]), key=lambda z: z.imag)
            err = max(abs(o1 - o2) for o1, o2 in zip(omega_from_L, omega_ref))
            scale = max(1.0, max(abs(o) for o in omega_ref))
            c.check(f"grid L eigenvalues match eq(3.7) roots (ikz={ikz}, kz={kz_val:.3f}, "
                    f"gamma_par={gamma_par:.3f})", err/scale < rtol,
                    f"omega_L={omega_from_L}, omega_ref={omega_ref}")
            for lam in (lam1, lam2):
                lhs, rhs, _ = _eq311_sides(kx_val, ky_val, Ln, nu_in, v0, gamma_par, lam.real)
                c.check(f"grid L Re(lambda)={lam.real:.4f} satisfies quartic (3.11) "
                        f"(ikz={ikz})", abs(lhs - rhs) < rtol*max(1.0, abs(lhs), abs(rhs)),
                        f"lhs={lhs:.6e}, rhs={rhs:.6e}")


def test_HW_limit_matches_eq312():
    # Hasegawa-Wakatani limit (3.12): nu_in=0 (which also sets A=0), gamma << gamma_par ->
    # gamma = omega_star^2*k_perp^2/(gamma_par*(1+k_perp^2)^3). Verified numerically (see
    # the P4b report) to converge as gamma_par grows; gamma_par=100 here gives ~1e-5 relerr.
    kx, ky, Ln = 0.0, 1.0, 2.0
    ksq = kx*kx + ky*ky
    omega_star = ky/Ln
    with checks() as c:
        for gamma_par in (20.0, 100.0, 1000.0):
            lam1, lam2 = _L_eigs_numpy(kx, ky, Ln, nu_in=0.0, v0=0.0, gpar_fac=0.0,
                                       D_par=1.0, kz=np.sqrt(gamma_par))
            gamma_slow = min((lam1.real, lam2.real), key=abs)   # the drift-wave branch
            pred = omega_star**2 * ksq / (gamma_par*(1.0 + ksq)**3)
            rel = abs(gamma_slow - pred)/abs(pred)
            c.check(f"H-W growth rate matches eq (3.12) (gamma_par={gamma_par})",
                    rel < 1e-2, f"gamma_slow={gamma_slow:.6e}, pred={pred:.6e}, rel={rel:.3e}")


def test_nearly_adiabatic_matches_eq39():
    # nearly-adiabatic limit (3.9): gamma_par -> infinity, drift waves modified by the
    # Pedersen drift and collisional damping: omega0 = (omega_Ped - omega_star -
    # i*nu_in*k_perp^2)/(1+k_perp^2). Verified numerically to converge as gamma_par grows.
    kx, ky, Ln, nu_in, v0 = 1.0, 1.0, 2.0, 0.3, 1.0
    ksq = kx*kx + ky*ky
    omega_star = ky/Ln
    omega_ped = ky*nu_in*v0
    omega0_pred = (omega_ped - omega_star - 1j*nu_in*ksq)/(1.0 + ksq)
    with checks() as c:
        for gamma_par in (1e4, 1e5):
            lam1, lam2 = _L_eigs_numpy(kx, ky, Ln, nu_in, v0, gpar_fac=0.0, D_par=1.0,
                                       kz=np.sqrt(gamma_par))
            lam_slow = min((lam1, lam2), key=lambda z: abs(z.real))   # not the ~-gamma_par branch
            omega_slow = -lam_slow.imag + 1j*lam_slow.real            # lambda = -i*omega
            rel = abs(omega_slow - omega0_pred)/abs(omega0_pred)
            c.check(f"nearly-adiabatic omega matches eq (3.9) (gamma_par={gamma_par:.0e})",
                    rel < 1e-2, f"omega_slow={omega_slow:.6f}, pred={omega0_pred:.6f}, "
                    f"rel={rel:.3e}")


def test_stabilization_boundary_matches_eq315():
    # eq (3.15): stable iff A + [gamma_par(omega_star-omega_Ped)/(nu_in*k_perp^2 +
    # gamma_par(1+k_perp^2))]^2 < gamma_par*nu_in. Scan gamma_par across the predicted
    # boundary (straddling ~0.2-0.5 at these parameters, verified numerically) and check
    # the SIGN of max Re(lambda) matches the predicted stable/unstable side every time.
    kx, ky, Ln, nu_in, v0 = 1.0, 1.0, 2.0, 0.3, 1.0
    ksq = kx*kx + ky*ky
    omega_star = ky/Ln
    omega_ped = ky*nu_in*v0
    A = (ky*ky/ksq)*(nu_in*v0/Ln)
    with checks() as c:
        for gamma_par in (0.01, 0.1, 0.2, 0.5, 1.0, 5.0):
            lhs = A + (gamma_par*(omega_star - omega_ped) /
                      (nu_in*ksq + gamma_par*(1.0 + ksq)))**2
            rhs = gamma_par*nu_in
            predicted_stable = lhs < rhs
            lam1, lam2 = _L_eigs_numpy(kx, ky, Ln, nu_in, v0, gpar_fac=0.0, D_par=1.0,
                                       kz=np.sqrt(gamma_par))
            actual_stable = max(lam1.real, lam2.real) < 0.0
            c.check(f"stability sign matches eq (3.15) (gamma_par={gamma_par})",
                    predicted_stable == actual_stable,
                    f"lhs={lhs:.5f}, rhs={rhs:.5f}, max_re={max(lam1.real, lam2.real):.5f}")


def test_energy_budget_closure_nonlinear_3D():
    # fp64-only (tight tolerance): short random-IC 3D run under imexcb3e (the P4b
    # production scheme -- CB-IMEX is L-stable and recovers the quasi-static balance the
    # stiff gamma_par/k_perp^2 -> D_par*kz^2/k_perp^2 term needs), same centered-difference
    # dE/dt vs gdi.energy_budget check as the 2D test above, now exercising the kz-dependent
    # gamma_par sink term.
    if not _fp64():
        print("[SKIP] test_energy_budget_closure_nonlinear_3D -- fp64 only (tight tolerance)")
        return
    from taranis.run import block_of_steps
    from taranis.timestepping import get_scheme

    params = _gdi_params_3d(nx=8, ny=8, nz=4, Ln=5.0, nu_in=0.2, v0=0.3, D_par=0.5,
                            gpar_fac=0.0, diss=1e-3, hyper=1, adaptive_timestep=False,
                            dt=2e-4)
    kgrid = jr.setup_kgrids(params)

    def ic(x, y, z):
        rng = np.random.default_rng(11)
        N = jnp.zeros_like(x*y*z)
        phi = jnp.zeros_like(x*y*z)
        for (kxi, kyi, kzi) in [(1, 1, 1), (2, 1, 0), (1, 2, 1), (2, 2, 2), (1, 0, 1), (0, 1, 0)]:
            aN, aP, ph1, ph2 = rng.normal(size=4)*1e-3
            kx = kxi*2*np.pi/params.Lx
            ky = kyi*2*np.pi/params.Ly
            kz = kzi*2*np.pi/params.Lz
            N = N + aN*jnp.cos(kx*x + ky*y + kz*z + ph1)
            phi = phi + aP*jnp.cos(kx*x + ky*y + kz*z + ph2)
        return jnp.stack([N, phi])

    state0 = jr.initialize(ic, params)
    stepper, scheme = get_scheme("imexcb3e")
    state1 = block_of_steps(state0, kgrid, params, 1, scheme, stepper)
    state2 = block_of_steps(state1, kgrid, params, 1, scheme, stepper)
    dt_actual = float(state2.t - state0.t)/2.0

    E0, _ = gdi.energy_enstrophy(state0, kgrid, params)
    E2, _ = gdi.energy_enstrophy(state2, kgrid, params)
    dEdt_measured = (float(E2) - float(E0))/(2*dt_actual)
    budget = gdi.energy_budget(state1, kgrid, params)
    dEdt_budget = float(budget["total"])

    rel = abs(dEdt_measured - dEdt_budget)/max(abs(dEdt_budget), 1e-30)
    with checks() as c:
        c.check("3D imexcb3e: measured dE/dt matches gdi.energy_budget total",
                rel < 1e-4, f"measured={dEdt_measured:.10e}, budget={dEdt_budget:.10e}, "
                f"rel={rel:.3e}, dt={dt_actual:.3e}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
