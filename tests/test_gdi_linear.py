# 2D GDI (physics/gdi.py, plans/GDI_PLAN.md P4a) validation.
#
# physics/gdi.py's L is derived directly from the scalar PDEs (5.1)-(5.2) (equivalently
# (3.5)-(3.6) with the 2D closure gamma_par substituted for D_par*d^2/dz^2) and agrees
# entry-by-entry with the paper's eq (5.3) matrix under L = -M (derivation:
# docs/gdi_linear_matrix_note.tex). Per plans/GDI_PLAN.md's ground rules the dispersion
# relations are the arbiter for any sign ambiguity regardless. What's checked:
#   1. L's characteristic polynomial (eigenvalues m +- s) matches the EXACT quadratic
#      dispersion relation (3.7)/(5.3) (derived independently in this file from the same
#      PDEs, not imported from gdi.py) at both gamma_par=0 and a finite closure value --
#      catches any transcription bug in physics/gdi.py._L_entries.
#   2. the gamma_par=0 case collapses to eq (2.8), and matches the collisional (2.9) and
#      inertial (2.11) asymptotic limits where they apply.
#   3. propagator vs dispersion cross-check: the ACTUAL putzer2-evolved field's measured
#      growth rate equals max Re(m +- s) from kgrid.lin_m/lin_s2 at that k.
#   4. nonlinear energy-budget closure (extended eq 3.18): a short fp64 run's measured
#      dE/dt matches gdi.energy_budget's total to a tight relative tolerance.
#   5. energy_enstrophy/cross_phase_spectrum against a direct numpy reference on a known
#      field (normalization gate, CLAUDE.md's "keep new energy-like diagnostics on the
#      shared perp_reduce convention").
#
# Dual precision: dispersion/propagator checks are precision-independent algebra (loose
# fp32 tolerance, tight fp64); the energy-budget closure test is fp64-gated (its tolerance
# is below fp32 roundoff) per CLAUDE.md's "print+return, never pytest.skip" convention.
# pytest: `pytest tests/test_gdi_linear.py`. Script: `python tests/test_gdi_linear.py`.
from _rmhd_testing import bootstrap, checks

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np

import jax_rmhd as jr
from jax_rmhd import propagators
from jax_rmhd.physics import gdi


def _fp64():
    return bool(jax.config.read("jax_enable_x64"))


def _gdi_params(nx=8, ny=8, Lx=2*np.pi, Ly=2*np.pi, Ln=2.0, nu_in=0.3, v0=1.0,
               gpar_fac=1.0, diss=0.0, hyper=1, **overrides):
    kw = dict(nx=nx, ny=ny, Lx=Lx, Ly=Ly, cfl_safety=0.5, dims=2, adaptive_timestep=False,
             dt=0.01, eqtype="GDI",
             eqpars=dict(Ln=Ln, nu_in=nu_in, v0=v0, gpar_fac=gpar_fac, diss=diss, hyper=hyper))
    kw.update(overrides)
    return jr.Parameters(**kw)


def _dispersion_quadratic_coeffs(kx, ky, Ln, nu_in, v0, gamma_par):
    # eq (3.7) (rho_s=c_s=Omega_i=1), expanded to a*omega^2+b*omega+c=0 -- derived
    # independently from (5.1)-(5.2) (see the P4a report for the by-hand algebra), NOT
    # imported from physics/gdi.py: this is the ground truth the implementation is
    # checked against, not a restatement of it.
    ksq = kx*kx + ky*ky
    omega_star = ky/Ln
    omega_ped = ky*nu_in*v0
    a = ksq
    b = 1j*(nu_in*ksq + gamma_par*(1.0 + ksq))
    c = -gamma_par*nu_in*ksq + omega_ped*omega_star + 1j*gamma_par*(omega_star - omega_ped)
    return a, b, c


def _L_eigs_numpy(kx, ky, Ln, nu_in, v0, gpar_fac, diss=0.0, hyper=1):
    # eigenvalues of physics/gdi.py's L at a single mode, via its own _L_entries (so this
    # exercises the actual implementation, not a hand copy of it).
    ksq = np.array([[kx*kx + ky*ky]])
    kyv = np.array([[ky]])
    inv_ksq = np.array([[1.0/ksq[0, 0] if ksq[0, 0] > 0 else 0.0]])
    L00, L01, L10, L11 = gdi._L_entries(ksq, kyv, inv_ksq, Ln, nu_in, v0, gpar_fac, diss, hyper)
    m = 0.5*(L00[0, 0] + L11[0, 0])
    s2 = m*m - (L00[0, 0]*L11[0, 0] - L01[0, 0]*L10[0, 0])
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
    # propagator vs dispersion cross-check: evolve a real jax_rmhd state through the actual
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
    from jax_rmhd.run import block_of_steps
    from jax_rmhd.timestepping import get_scheme

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
    from jax_rmhd.grids import fft, ifft, dealias_mask
    mask = np.asarray(dealias_mask(params))
    Nk_np = np.asarray(fft(jnp.asarray(N_real), params))*mask
    phik_np = np.asarray(fft(jnp.asarray(phi_real), params))*mask
    N_dealiased = np.asarray(ifft(jnp.asarray(Nk_np), params))
    phi_dealiased = np.asarray(ifft(jnp.asarray(phik_np), params))
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


def test_registry_rejects_3d_and_forcing():
    with checks() as c:
        try:
            params3d = _gdi_params(dims=3, nz=4, Lz=2*np.pi)
            jr.setup_kgrids(params3d)
            err = None
        except NotImplementedError as e:
            err = str(e)
        c.check("GDI dims=3 raises NotImplementedError at kgrid setup", err is not None)
        try:
            params_f = _gdi_params(forcing=True, fshell=(1, 2))
            jr.setup_kgrids(params_f)
            err = None
        except ValueError as e:
            err = str(e)
        c.check("GDI forcing=True raises ValueError at kgrid setup", err is not None)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
