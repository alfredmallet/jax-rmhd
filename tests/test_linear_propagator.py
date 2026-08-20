# Linear-propagator machinery (taranis/propagators.py).
#
# The steppers only ever call the hook (apply_exp / solve_shifted / scaled) on the object
# built from a recipe's linear_matrix_func, with the convention dt f = L f + N(f), i.e.
# propagator = exp(L*tau). What is checked here:
#   1. putzer2 (closed-form 2x2 exponential) against scipy.linalg.expm on random batches
#      covering the four regimes that break naive implementations: general complex,
#      DEFECTIVE (s -> 0, where sinh(s*tau)/s is 0/0), growing (s^2 > 0) and oscillatory
#      (s^2 < 0, where a real sqrt would NaN);
#   2. the Taylor branch is continuous across its |s*tau| cutoff;
#   3. a rotation test: conjugating a diagonal system by a constant 2x2 rotation and
#      evolving it with putzer2 reproduces the diagonal backend's trajectory;
#   4. solve_shifted: (I - a L) applied to solve_shifted(arr, a) returns arr, both backends;
#   5. scaled(c) (used by lsrk_advance to keep the pre-P1 (L*dt)*gamma op order) agrees
#      with rescaling tau;
#   6. setup wiring: RMHD selects the diagonal backend, reproduces -diss*ksq**hyper, and
#      an L that breaks the rfft2 reality constraint is rejected at setup;
#   7. the coefficient evaluation (cosh(s*tau), sinh(s*tau)/s built from w = exp(s*tau))
#      against a private copy of the previous sqrt/cosh/sinh form, over a sweep of
#      (m, s^2, tau) covering both sides of the Taylor cutoff and |Re(s*tau)| up to the
#      overflow limit, plus the cutoff neighbourhood against an accurate complex128
#      reference (the w-form's cancellation is what sets the cutoff).
# The eqpars migration of old params.json records is tested in tests/test_params.py
# (test_legacy_toplevel_diss_hyper_folds_into_eqpars), with the rest of the record tests.
#
# Dual precision: every test runs in BOTH sessions with precision-dependent tolerances
# (no pytest.skip -- fp32 is the interesting case for the Taylor branch).
# pytest: `pytest tests/test_linear_propagator.py`. Script: `python tests/...`.
from _rmhd_testing import bootstrap, checks, ctx, fresh_params

bootstrap()

import jax.numpy as jnp
import numpy as np
import scipy.linalg

import taranis as jr
from taranis import _precision, propagators
from taranis.physics import rmhd


def _fp64():
    # FIELD precision (TARANIS_PRECISION) -- jax_enable_x64 is now unconditionally on.
    return _precision.precision == "64"


def _tols():
    """(relative tolerance vs the fp64 scipy reference, label) for this session."""
    return (2e-12, "fp64") if _fp64() else (2e-4, "fp32")


def _batch(mats):
    """(2, 2, N) propagator input from a list of 2x2 numpy matrices."""
    return jnp.asarray(np.stack(mats, axis=-1))


def _putzer(mats):
    return propagators.Putzer2Propagator(*propagators.putzer2_precompute(_batch(mats)))


def _columns(prop, n, tau):
    """The propagator's matrix, column by column: apply_exp to the two unit vectors."""
    cols = []
    for j in range(2):
        e = np.zeros((2, n))
        e[j] = 1.0
        cols.append(np.asarray(prop.apply_exp(jnp.asarray(e), tau)))
    return np.stack(cols, axis=1)      # (2, 2, N) = [row, column, mode]


def _sample_matrices(seed=3):
    """Named 2x2 cases spanning the regimes the closed form has to survive."""
    rng = np.random.default_rng(seed)
    cases = {}
    cases["general complex"] = [rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))
                                for _ in range(12)]
    # defective: equal eigenvalues with a nonzero off-diagonal -> s = 0 exactly
    cases["defective (s = 0)"] = [np.array([[a, b], [0.0, a]])
                                  for a, b in zip(rng.normal(size=6), rng.normal(size=6))]
    # near-defective: s^2 tiny but nonzero, i.e. right on the Taylor branch
    cases["near-defective"] = [np.array([[a, 1.0], [eps, a]])
                               for a, eps in zip(rng.normal(size=5),
                                                 [1e-14, 1e-10, 1e-7, 1e-4, 1e-2])]
    # growing: real, distinct, positive-real-part eigenvalues (s^2 > 0)
    cases["growing (s^2 > 0)"] = [np.diag([l1, l2]) + np.array([[0.0, c], [0.0, 0.0]])
                                  for l1, l2, c in zip(rng.uniform(0.1, 1.5, 5),
                                                       rng.uniform(-1.5, -0.1, 5),
                                                       rng.normal(size=5))]
    # oscillatory: purely imaginary eigenvalues from a real matrix (s^2 < 0)
    cases["oscillatory (s^2 < 0)"] = [np.array([[0.0, w], [-w, 0.0]])
                                      for w in (0.3, 1.0, 2.5)]
    # dissipative + wave coupling: the RMHD/GDI-shaped case (s^2 complex)
    cases["damped waves"] = [np.array([[-d, 1j*kz], [1j*kz, -d]])
                             for d, kz in ((0.1, 1.0), (2.0, 0.5), (0.05, 3.0))]
    return cases


def test_putzer2_matches_scipy_expm():
    rtol, label = _tols()
    with checks() as c:
        for name, mats in _sample_matrices().items():
            prop = _putzer(mats)
            for tau in (0.0, 1e-3, 0.37, 1.0):
                got = _columns(prop, len(mats), tau)
                ref = np.stack([scipy.linalg.expm(np.asarray(m, dtype=complex)*tau)
                                for m in mats], axis=-1)
                scale = max(1.0, float(np.max(np.abs(ref))))
                err = float(np.max(np.abs(got - ref)))/scale
                c.check(f"putzer2 == scipy.expm ({label}, {name}, tau={tau})",
                        err < rtol and np.all(np.isfinite(got)), f"rel err {err:.3e}")


def test_taylor_branch_is_continuous_across_its_cutoff():
    # sinh(s*tau)/s switches to a Taylor series at small |s*tau|: straddle the cutoff with
    # matrices whose s^2 sits just either side of it and check both branches agree with expm.
    rtol, label = _tols()
    tol_z2 = propagators._tol_z2()
    taus = (1.0,)
    mats = []
    for frac in (0.2, 0.9, 0.999, 1.001, 1.1, 5.0):
        s2 = frac*tol_z2                  # tau = 1 so z^2 = s^2
        mats.append(np.array([[0.5, 1.0], [s2, 0.5]]))       # tr/2 = 0.5, det = 0.25 - s2
        mats.append(np.array([[0.5, 1.0], [-s2, 0.5]]))      # s^2 < 0 side
    prop = _putzer(mats)
    with checks() as c:
        for tau in taus:
            got = _columns(prop, len(mats), tau)
            ref = np.stack([scipy.linalg.expm(np.asarray(m, dtype=complex)*tau)
                            for m in mats], axis=-1)
            err = float(np.max(np.abs(got - ref)))/max(1.0, float(np.max(np.abs(ref))))
            c.check(f"both sides of the |s*tau| cutoff match expm ({label}, "
                    f"tol_z2={tol_z2:g})", err < rtol, f"rel err {err:.3e}")


def test_putzer2_of_a_rotated_diagonal_system_matches_the_diagonal_backend():
    # L = R diag(l1,l2) R^T with a constant rotation R: evolving with putzer2 and rotating
    # back must reproduce the diagonal backend's trajectory (exp is basis-covariant).
    rtol, label = _tols()
    rng = np.random.default_rng(11)
    n = 24
    lam = np.stack([-rng.uniform(0.0, 2.0, n) + 1j*rng.normal(size=n),
                    -rng.uniform(0.0, 2.0, n) + 1j*rng.normal(size=n)])
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    L_rot = np.einsum("ij,jn,kj->ikn", R, lam, R)          # R diag(lam) R^T, per mode
    u0 = rng.normal(size=(2, n)) + 1j*rng.normal(size=(2, n))

    diag = propagators.DiagonalPropagator(jnp.asarray(lam))
    rot = propagators.Putzer2Propagator(*propagators.putzer2_precompute(jnp.asarray(L_rot)))
    tau, nstep = 0.13, 7
    u_d = jnp.asarray(np.einsum("ji,jn->in", R, u0))       # R^T u0: the diagonal basis
    u_r = jnp.asarray(u0)
    for _ in range(nstep):
        u_d = diag.apply_exp(u_d, tau)
        u_r = rot.apply_exp(u_r, tau)
    back = np.einsum("ji,jn->in", R, np.asarray(u_r))      # rotate the putzer2 result back
    err = float(np.max(np.abs(back - np.asarray(u_d))))/max(1.0, float(np.max(np.abs(u_d))))
    with checks() as c:
        c.check(f"rotated putzer2 trajectory == diagonal trajectory ({label}, "
                f"{nstep} steps)", err < rtol, f"rel err {err:.3e}")


def test_solve_shifted_inverts_the_shifted_operator():
    # (I - a L) . solve_shifted(arr, a) == arr, for both backends (the P3 IMEX hook).
    rtol, label = _tols()
    rng = np.random.default_rng(5)
    n = 16
    arr = jnp.asarray(rng.normal(size=(2, n)) + 1j*rng.normal(size=(2, n)))
    Ld = jnp.asarray(-rng.uniform(0.1, 5.0, size=(2, n)))
    mats = [rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2)) for _ in range(n)]
    L2 = _batch(mats)
    diag = propagators.DiagonalPropagator(Ld)
    put = propagators.Putzer2Propagator(*propagators.putzer2_precompute(L2))
    with checks() as c:
        for a in (0.01, 0.5, 3.0):
            got = diag.solve_shifted(arr, a)
            back = got - a*Ld*got
            err = float(np.max(np.abs(np.asarray(back - arr))))
            c.check(f"diagonal solve_shifted inverts I - {a}L ({label})",
                    err < rtol*10, f"abs err {err:.3e}")
            got2 = put.solve_shifted(arr, a)
            g = np.asarray(got2)
            M = np.eye(2)[:, :, None] - a*np.asarray(L2)
            back2 = np.einsum("ijn,jn->in", M, g)
            err2 = float(np.max(np.abs(back2 - np.asarray(arr))))
            c.check(f"putzer2 solve_shifted inverts I - {a}L ({label})",
                    err2 < rtol*10, f"abs err {err2:.3e}")


def test_scaled_matches_rescaling_tau():
    # lsrk_advance uses scaled(dt) so the stage exponent is (L*dt)*gamma; it must agree
    # with exp(L*(dt*gamma)) to round-off on both backends.
    rtol, label = _tols()
    rng = np.random.default_rng(9)
    n = 12
    arr = jnp.asarray(rng.normal(size=(2, n)) + 1j*rng.normal(size=(2, n)))
    diag = propagators.DiagonalPropagator(jnp.asarray(-rng.uniform(0.0, 3.0, size=(2, n))))
    put = _putzer([rng.normal(size=(2, 2)) for _ in range(n)])
    dt, gamma = 0.021, 0.37
    with checks() as c:
        for name, prop in (("diagonal", diag), ("putzer2", put)):
            a = np.asarray(prop.scaled(dt).apply_exp(arr, gamma))
            b = np.asarray(prop.apply_exp(arr, dt*gamma))
            err = float(np.max(np.abs(a - b)))/max(1.0, float(np.max(np.abs(b))))
            c.check(f"scaled(dt).apply_exp(.,gamma) == apply_exp(.,dt*gamma) "
                    f"({label}, {name})", err < rtol*10, f"rel err {err:.3e}")


def test_rmhd_uses_the_diagonal_backend_with_eqpars_dissipation():
    # setup_kgrids builds L from the recipe; RMHD's is the old hdiss array, and the
    # hdiss field itself is gone (the steppers only see the hook now).
    params, kgrid = ctx(dims=2, diss=(0.01, 0.02), hyper=2)
    prop = propagators.get_propagator(kgrid, params)
    # dtype=ftype: kgrid.lin_L is built at FIELD precision (grids/rmhd pin it), so the
    # reference has to be too -- a bare jnp.array is a strong float64 under x64.
    expected = -jnp.array((0.01, 0.02), dtype=_precision.ftype).reshape(-1, 1, 1, 1)*kgrid.ksq**2
    arr = jnp.ones((params.nfields, 1, params.nx, params.ny//2 + 1),
                   dtype=jnp.result_type(float, complex))
    with checks() as c:
        c.check("RMHD gets the diagonal backend",
                isinstance(prop, propagators.DiagonalPropagator))
        c.check("kgrid.lin_L is -diss*ksq**hyper",
                bool(jnp.array_equal(kgrid.lin_L, expected)))
        c.check("putzer2 precomputes are absent for a diagonal operator",
                kgrid.lin_m is None and kgrid.lin_s is None)
        c.check("kgrid.hdiss is gone", not hasattr(kgrid, "hdiss"))
        c.check("apply_exp is exp(L*tau)*arr",
                bool(jnp.array_equal(prop.apply_exp(arr, 0.25),
                                     jnp.exp(kgrid.lin_L*0.25)*arr)))


def test_setup_rejects_operators_that_break_reality_or_shape():
    # L must satisfy L(-kx,ky) = conj(L(kx,ky)) on the ky=0 and Nyquist rows, or the
    # exponential would destroy the reality of the fields it acts on.
    params = fresh_params(dims=2, diss=(0.01, 0.01), hyper=1)
    kgrid = jr.setup_kgrids(params)
    good = rmhd.linear_matrix(kgrid, params)
    bad = good.at[:, :, 1, 0].add(1.0)                     # single kx entry, ky=0 row
    nkx, nky = params.nx, params.ny//2 + 1
    wrong_shape = jnp.zeros((3, 1, nkx, nky))
    putzer_wrong_nfields = jnp.zeros((2, 2, 1, nkx, nky))
    with checks() as c:
        try:
            propagators.linear_fields(good, params)
            err = None
        except ValueError as e:
            err = str(e)
        c.check("the RMHD operator passes the reality check", err is None, f"raised {err!r}")
        for name, L in (("a non-hermitian-compatible ky=0 row", bad),
                        ("a leading axis that is neither 1 nor nfields", wrong_shape)):
            try:
                propagators.linear_fields(L, params)
                err = None
            except ValueError as e:
                err = str(e)
            c.check(f"setup rejects {name}", err is not None)
        # nfields=2 IS satisfied by RMHD, so the putzer2 shape is accepted here: check the
        # complementary failure, a 3-d operator that matches no backend
        try:
            propagators.linear_fields(jnp.zeros((nkx, nky)), params)
            err = None
        except ValueError as e:
            err = str(e)
        c.check("setup rejects an operator matching no backend", err is not None)
        c.check("a (2,2,1,nkx,nky) operator is accepted for nfields=2 and precomputed",
                set(propagators.linear_fields(putzer_wrong_nfields, params))
                == {"lin_L", "lin_m", "lin_s"})


# ---------------------------------------------------------------- coefficient evaluation
# The reference for the two tests below: the previous _coeffs, verbatim, with the cutoffs
# it shipped with. Those are 100x smaller in |z^2| than the current ones, so over the whole
# widened Taylor band the reference is jnp.sinh(z)/z -- an independent accurate value there,
# not the same series.
_OLD_TOL_Z2 = {"64": 1e-6, "32": 1e-4}


def _coeffs_old(s2, tau):
    z2 = s2*(tau*tau)
    z = jnp.sqrt(z2)
    small = jnp.abs(z2) < _OLD_TOL_Z2[_precision.precision]
    z_safe = jnp.where(small, jnp.ones_like(z), z)
    sinhc = jnp.where(small,
                      1.0 + z2*(1.0/6.0 + z2*(1.0/120.0 + z2/5040.0)),
                      jnp.sinh(z_safe)/z_safe)
    return jnp.cosh(z), tau*sinhc


def _exp_entries_old(L, m, s2, tau):
    # the previous exp_op, in the same op order, off the previous coefficients
    cosh_z, sinh_over_s = _coeffs_old(s2, tau)
    pref = jnp.exp(m*tau)
    d = cosh_z - sinh_over_s*m
    return (pref*(d + sinh_over_s*L[0, 0]), pref*(sinh_over_s*L[0, 1]),
            pref*(sinh_over_s*L[1, 0]), pref*(d + sinh_over_s*L[1, 1]))


def _coeff_tols():
    return (1e-13, "fp64") if _fp64() else (1e-5, "fp32")


# Phase samples on the cutoff circle. The w-form's cancellation error at fixed |z| swings by
# several times with arg(z), so a sparse sweep can sit in a lucky phase and pass at a cutoff
# that is actually too small: this count is what makes a shrunken cutoff fail the test.
_CUTOFF_PHASES = 1024


def _re_limit():
    # |Re(s*tau)| the exponential still has to survive (cosh overflows at ~710 / ~88)
    return 600.0 if _fp64() else 80.0


def _z_grid():
    """Complex z = s*tau spanning the branch boundary, the defective limit and overflow."""
    zb = np.sqrt(propagators._tol_z2())          # |z| at the Taylor cutoff
    lim = _re_limit()
    mags = np.array([0.0, 1e-12, 1e-6, 0.3*zb, 0.9*zb, 0.999*zb, 1.001*zb, 1.1*zb, 3*zb,
                     0.5, 1.0, 7.3, lim/3.0, lim])
    phases = np.array([0.0, 0.37, np.pi/4, 1.1, np.pi/2, 2.0, np.pi - 0.3])
    return (mags[:, None]*np.exp(1j*phases[None, :])).ravel()


def _rel(new, ref, floor):
    """Elementwise |new - ref| / max(|ref|, floor)."""
    num = np.abs(np.asarray(new, dtype=complex) - np.asarray(ref, dtype=complex))
    return num/np.maximum(np.abs(np.asarray(ref, dtype=complex)), floor)


def _eps():
    return float(np.finfo(np.dtype(_precision.ftype)).eps)


def _within(new, ref, sens, rtol):
    """Worst ratio of |new - ref| to (rtol*|ref| + 32*eps*sens); < 1 passes.

    The two forms do not evaluate at the same z: the old one round-trips through
    sqrt(s^2*tau^2), which moves z by ~eps*|z|. `sens` is |dcoeff/dz|*|z|, so eps*sens is
    the size that perturbation alone puts into the answer; it dominates the relative term
    wherever the coefficient is ill-conditioned in z (large |z|, and the zeros of cosh and
    sinh on the imaginary axis)."""
    num = np.abs(np.asarray(new, dtype=complex) - np.asarray(ref, dtype=complex))
    return float(np.max(num/(rtol*np.abs(np.asarray(ref, dtype=complex)) + 32*_eps()*sens)))


def test_coeffs_match_the_previous_sqrt_cosh_sinh_form():
    rtol, label = _coeff_tols()
    z = _z_grid()
    with checks() as c:
        for tau in (1.0, 0.017, 43.0):
            s = jnp.asarray(z/tau, dtype=_precision.ctype)
            s2 = jnp.asarray((z/tau)**2, dtype=_precision.ctype)
            prop = propagators.Putzer2Propagator(None, None, s)
            got_c, got_s = prop._coeffs(tau)
            ref_c, ref_s = _coeffs_old(s2, tau)
            # d(cosh z)/dz = sinh z and d(tau*sinh(z)/z)/dz ~ tau*cosh(z)/z, both bounded
            # by cosh(Re z); times the eps*|z| that the sqrt round-trip moves z by
            cre = np.cosh(np.real(z))
            sens_c = np.abs(z)*cre
            sens_s = abs(tau)*cre
            ec = _within(got_c, ref_c, sens_c, rtol)
            es = _within(got_s, ref_s, sens_s, rtol)
            c.check(f"cosh(s*tau) matches the previous form ({label}, tau={tau})",
                    ec < 1.0 and np.all(np.isfinite(np.asarray(got_c))),
                    f"worst tolerance ratio {ec:.3f}; worst plain rel err "
                    f"{float(np.max(_rel(got_c, ref_c, 0.0))):.3e}")
            c.check(f"sinh(s*tau)/s matches the previous form ({label}, tau={tau})",
                    es < 1.0 and np.all(np.isfinite(np.asarray(got_s))),
                    f"worst tolerance ratio {es:.3f}; worst plain rel err "
                    f"{float(np.max(_rel(got_s, ref_s, 0.0))):.3e}")


def test_coeffs_at_the_taylor_cutoff_match_an_accurate_reference():
    # Both sides of the cutoff, against complex128 cosh / sinh(z)/z: the series side must be
    # accurate (truncation |z|^8/362880) and the w = exp(z) side must not have lost the gate
    # to cancellation. Also the two sides of the threshold must agree with each other.
    rtol, label = _coeff_tols()
    tol_z2 = propagators._tol_z2()
    zb = np.sqrt(tol_z2)
    phases = np.linspace(0.0, 2*np.pi, _CUTOFF_PHASES, endpoint=False)
    tau = 1.0
    with checks() as c:
        sides = {}
        for side, frac in (("Taylor side", 0.999), ("w = exp(z) side", 1.001)):
            s = jnp.asarray(np.sqrt(frac)*zb*np.exp(1j*phases)/tau, dtype=_precision.ctype)
            got_c, got_s = propagators.Putzer2Propagator(None, None, s)._coeffs(tau)
            sides[side] = (got_c, got_s)
            # reference at the z the kernel really evaluates at: s is stored at field
            # precision, so this measures the evaluation, not the input quantization
            z = np.asarray(np.asarray(s), dtype=np.complex128)*tau
            ref_c = np.cosh(z)
            ref_s = tau*np.sinh(z)/z
            ec = float(np.max(_rel(got_c, ref_c, 0.0)))
            es = float(np.max(_rel(got_s, ref_s, 0.0)))
            c.check(f"cosh at the cutoff, {side} ({label}, |z|={np.abs(z[0]):.3g})",
                    ec < rtol, f"rel err {ec:.3e}")
            c.check(f"sinh(z)/z at the cutoff, {side} ({label}, |z|={np.abs(z[0]):.3g})",
                    es < rtol, f"rel err {es:.3e}")
        below, above = sides["Taylor side"], sides["w = exp(z) side"]
        for name, a, b in (("cosh", below[0], above[0]), ("sinh/s", below[1], above[1])):
            # the two samples differ in |z^2| by 0.002*tol_z2, so cosh and sinh(z)/z
            # themselves differ by ~1e-3*tol_z2: anything above that is a branch jump
            jump = float(np.max(_rel(a, b, 0.0)))
            c.check(f"{name} is continuous across the |z^2| = {tol_z2:g} threshold "
                    f"({label})", jump < 2e-3*tol_z2 + 10*rtol, f"jump {jump:.3e}")


def test_exp_op_matches_the_previous_evaluation():
    # the coefficients inside the full exp(L*tau): sweeps m as well as s^2 and tau
    rtol, label = _coeff_tols()
    rng = np.random.default_rng(23)
    z = _z_grid()
    n = z.size
    ms = np.concatenate([np.zeros(3), rng.normal(size=n - 6) + 1j*rng.normal(size=n - 6),
                         np.array([-3.0, -0.5j, 2.0 + 1.0j])])
    with checks() as c:
        for tau in (0.11, 1.0, 9.0):
            s2 = (z/tau)**2
            m = ms/max(1.0, float(np.max(np.abs(ms)))*tau)      # keep exp(m*tau) in range
            # L with the right trace and determinant: [[m + a, b], [(s2 - a^2)/b, m - a]]
            a = 0.3 + 0.2j
            b = 1.7 - 0.4j
            L = np.stack([np.stack([m + a, np.full(n, b)]),
                          np.stack([(s2 - a*a)/b, m - a])]).astype(complex)
            Lj = jnp.asarray(L, dtype=_precision.ctype)
            mj = jnp.asarray(m, dtype=_precision.ctype)
            prop = propagators.Putzer2Propagator(Lj, mj,
                                                 jnp.asarray(z/tau, dtype=_precision.ctype))
            got = prop.exp_op(tau)
            ref = _exp_entries_old(Lj, mj, jnp.asarray(s2, dtype=_precision.ctype), tau)
            # the entries are pref*(cosh, sinh/s)*(L, m), so the sensitivity of the
            # coefficients carries the size of L and m with it
            sens = (np.exp(np.real(m)*tau)*np.cosh(np.real(z))
                    *(np.abs(z) + 1.0 + np.max(np.abs(L), axis=(0, 1))))
            err = max(_within(g, r, sens, rtol) for g, r in zip(got, ref))
            c.check(f"exp(L*tau) entries match the previous evaluation ({label}, "
                    f"tau={tau})", err < 1.0, f"worst tolerance ratio {err:.3f}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
