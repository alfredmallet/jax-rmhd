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
#      an L that breaks the rfft2 reality constraint is rejected at setup.
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
                kgrid.lin_m is None and kgrid.lin_s2 is None)
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
                == {"lin_L", "lin_m", "lin_s2"})


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
