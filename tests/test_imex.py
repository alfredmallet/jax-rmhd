# CB-IMEX low-storage implicit/explicit RK schemes (plans/GDI_PLAN.md P3).
#
# The schemes are Cavaglieri & Bewley, JCP 286:172-193 (2015),
# http://robotics.ucsd.edu/pubs/CB15.pdf -- IMEXRKCB2 (eq. 24), IMEXRKCB3c (eq. 28a),
# IMEXRKCB3e (eq. 30) and IMEXRKCB3f (eq. 32c). In taranis the WHOLE k-local linear
# operator L is the implicit part (dissipation included; no exponential anywhere in this
# path) and construct_rhs's summed term_funcs are the explicit part.
#
# A wrong transcribed coefficient survived in this repo's lsrk54 table for years, so
# nothing here trusts the numbers in timestepping.py:
#   1. test_tableau_structure_and_order_conditions -- rebuilds the Butcher tableaux and
#      checks row sums, stiff accuracy, the [2R]/[3R] low-storage structure (i.e. that the
#      entries the steppers IGNORE really are b_j), and every order condition of both
#      tableaux AND the coupling conditions up to the claimed order (plus that a scheme
#      claiming order 2 really does fail order 3).
#   2. test_implicit_part_is_L_stable -- |R(z)| <= 1 sampled over the left half plane and
#      R(z) -> 0 as z -> -inf.
#   3. test_low_storage_matches_dense_tableau -- the actual repo steppers vs a plain
#      dense-tableau IMEX integrator on the same ODE, to round-off.
# and the acceptance items of the plan:
#   4. test_convergence_order -- measured order on a manufactured NONLINEAR problem with an
#      exact solution (u' = L u + u^2 + S(t), S chosen so u = a(t)F is exact).
#   5. test_stiff_quasi_static -- u' = -gamma(u - g(t)) + weak NL at gamma*dt >> 1. The
#      IMEX schemes must track u ~ g + O(1/gamma); the IF-LSRK error on the SAME problem is
#      measured and reported in the check line (it is the motivation for the whole task).
#   6. test_rmhd_forced_2d_tracks_lsrk54 -- statistical regression on a small forced run.
#   7. test_z_spectral_3d_runs_and_converges_to_if -- the putzer2 solve_shifted path with a
#      REAL kz extent. L-stable solves damp waves at |omega|*dt >~ 1 (expected, that is what
#      L-stability means), so this only asserts finiteness and dt -> 0 convergence to the
#      IF answer, never wave fidelity at large dt.
#
# Tests 4-7 drive the REAL machinery (equation_registry recipes, setup_kgrids, run.py's
# block_of_steps), not hand-rolled stepper loops. Tests 3-5 register a throw-away equation
# set for the duration of the test and remove it again.
#
# Dual precision: the tableau/L-stability algebra is done in numpy float64 in both
# sessions (it is a property of the coefficients, not of the run precision); the ODE tests
# carry precision-dependent tolerances. No pytest.skip -- the two heavier physics runs are
# marked fp64 in the same style as the neighbouring modules.
# pytest: `pytest tests/test_imex.py`. Script: `python tests/test_imex.py`.
from _rmhd_testing import (bootstrap, checks, ctx, fit_order, fresh_params, make_state,
                           multimode_ic, zero_ic_2d)

bootstrap()

import contextlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import taranis as jr
from taranis import _precision, grids, timestepping
from taranis.diagnostics import energy
from taranis.physics import EquationRecipe, equation_registry
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme, imex2r_advance, imex3r_advance

IMEX_SCHEMES = ("imexcb2", "imexcb3e", "imexcb3c", "imexcb3f")

# jitted exactly like run.py's non-"jax" path (params/nblock/scheme/stepper static)
_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))


def _fp64():
    # FIELD precision (TARANIS_PRECISION) -- jax_enable_x64 is now unconditionally on.
    return _precision.precision == "64"


def _run(state, kgrid, params, nsteps, schemestr):
    stepper, scheme = get_scheme(schemestr)
    return _advance(state, kgrid, params, nsteps, scheme, stepper)


# ------------------------------------------------------------------ tableau algebra

def _tableaux(name):
    """(A_im, A_ex, b, c, scheme) as float64 numpy arrays, from the registry entry."""
    _, sch = get_scheme(name)
    return (np.array(sch.a_im, dtype=np.float64), np.array(sch.a_ex, dtype=np.float64),
            np.array(sch.b, dtype=np.float64), np.array(sch.c, dtype=np.float64), sch)


def test_tableau_structure_and_order_conditions():
    atol = 1e-13
    with checks() as c:
        for name in IMEX_SCHEMES:
            A_im, A_ex, b, cc, sch = _tableaux(name)
            s = len(b)
            c.check(f"{name}: square {s}-stage tableaux with c = (0, ..., 1) (FSAL)",
                    A_im.shape == (s, s) and A_ex.shape == (s, s)
                    and cc[0] == 0.0 and cc[-1] == 1.0)
            # triangularity: DIRK lower triangular incl. diagonal, ERK strictly lower
            c.check(f"{name}: A_im lower triangular, A_ex strictly lower triangular",
                    np.all(np.triu(A_im, 1) == 0.0) and np.all(np.triu(A_ex, 0) == 0.0))
            c.check(f"{name}: sum b == 1 (order 1)", abs(b.sum() - 1.0) < atol,
                    f"residual {b.sum() - 1.0:.3e}")
            c.check(f"{name}: A_im row sums == c",
                    np.max(np.abs(A_im.sum(1) - cc)) < atol,
                    f"residual {np.max(np.abs(A_im.sum(1) - cc)):.3e}")
            c.check(f"{name}: A_ex row sums == c",
                    np.max(np.abs(A_ex.sum(1) - cc)) < atol,
                    f"residual {np.max(np.abs(A_ex.sum(1) - cc)):.3e}")
            c.check(f"{name}: stiff accuracy a^IM_(s,i) = b_i (what makes the implicit part "
                    f"L-stable and gives the quasi-static limit)",
                    np.max(np.abs(A_im[-1] - b)) < atol,
                    f"residual {np.max(np.abs(A_im[-1] - b)):.3e}")
            # low-storage structure: the entries the steppers never read must BE b_j.
            # [2R] (paper eq. 16): a_(k,j) = b_j for j <= k-2. [3R] (eq. 17): j <= k-3.
            lead = 2 if sch.structure == "2R" else 3
            worst = 0.0
            for k in range(s):
                for j in range(max(0, k - lead + 1)):
                    worst = max(worst, abs(A_im[k, j] - b[j]), abs(A_ex[k, j] - b[j]))
            c.check(f"{name}: [{sch.structure}] structure holds (a_(k,j) = b_j for "
                    f"j <= k-{lead}), so the low-storage stepper sees the whole tableau",
                    worst < atol, f"worst deviation {worst:.3e}")

            # Order conditions. b and c are SHARED between the two tableaux (the paper's
            # design constraint), so orders 1-2 and "sum b c^2" are common and the
            # remaining third-order conditions are one per tableau -- which is also
            # exactly the IMEX coupling set at this order.
            conds = {"sum b c = 1/2": (b @ cc, 0.5)}
            if sch.order >= 3:
                conds["sum b c^2 = 1/3"] = (b @ (cc**2), 1.0 / 3.0)
                for lbl, A in (("IM", A_im), ("EX", A_ex)):
                    conds[f"sum b a{lbl} c = 1/6"] = (b @ (A @ cc), 1.0 / 6.0)
            for lbl, (got, want) in conds.items():
                c.check(f"{name}: order-{sch.order} condition {lbl}",
                        abs(got - want) < atol, f"got {got!r}, residual {got - want:.3e}")
            if sch.order == 2:
                # a 2nd-order scheme must NOT accidentally satisfy the 3rd-order set
                third = max(abs(b @ (cc**2) - 1.0 / 3.0), abs(b @ (A_ex @ cc) - 1.0 / 6.0))
                c.check(f"{name}: is genuinely 2nd order (3rd-order conditions fail)",
                        third > 1e-3, f"max third-order residual {third:.3e}")


def _stability_function(A, b, z):
    """R(z) = 1 + z b^T (I - z A)^-1 1, elementwise over an array of z."""
    s = len(b)
    one = np.ones(s)
    z = np.asarray(z)
    out = np.empty(z.shape, dtype=complex)
    for idx in np.ndindex(z.shape):
        out[idx] = 1.0 + z[idx] * (b @ np.linalg.solve(np.eye(s) - z[idx] * A, one))
    return out


def _R_exact_real(A, b, z):
    """|R(z)| for real z, in exact rational arithmetic.

    Needed for the z -> -inf tail: R(z) ~ C/z there, and IMEXRKCB3f's tableau has
    O(4) entries of both signs, so a float evaluation bottoms out at a roundoff floor
    (~1e-7) long before |R| does. A is lower triangular, so (I - zA)x = 1 is a forward
    substitution and Fraction makes it exact for the coefficients AS STORED.
    """
    from fractions import Fraction as Fr
    s = len(b)
    zf = Fr(z)
    x = []
    for k in range(s):
        num = Fr(1) + zf*sum(Fr(float(A[k, j]))*x[j] for j in range(k))
        x.append(num / (Fr(1) - zf*Fr(float(A[k, k]))))
    return abs(float(Fr(1) + zf*sum(Fr(float(b[i]))*x[i] for i in range(s))))


def test_implicit_part_is_L_stable():
    # A-stability (|R| <= 1 on Re z <= 0) plus R(-inf) = 0 is L-stability. The samples run
    # out to |z| = 1e6 so this really covers the stiff corner the schemes exist for.
    re = -np.concatenate([[0.0], np.logspace(-3, 6, 60)])
    im = np.concatenate([-np.logspace(-3, 6, 40)[::-1], [0.0], np.logspace(-3, 6, 40)])
    Z = re.reshape(-1, 1) + 1j * im.reshape(1, -1)
    with checks() as c:
        for name in IMEX_SCHEMES:
            A_im, A_ex, b, cc, sch = _tableaux(name)
            worst = float(np.max(np.abs(_stability_function(A_im, b, Z))))
            c.check(f"{name}: implicit part is A-stable over Re z <= 0 (max |R| = "
                    f"{worst:.12f})", worst <= 1.0 + 1e-10)
            zs = (-1e4, -1e6, -1e8, -1e10)
            decay = np.array([_R_exact_real(A_im, b, z) for z in zs])
            # R(z) ~ C/z: each factor-100 step in |z| must cut |R| by ~100
            ratios = decay[:-1] / decay[1:]
            c.check(f"{name}: R(z) -> 0 as z -> -inf, i.e. L-stable not merely A-stable "
                    f"(|R| = {np.array2string(decay, precision=2)} at z = -1e4..-1e10)",
                    decay[-1] < 1e-8 and np.all(np.abs(ratios - 100.0) < 5.0))
            big = abs(_stability_function(A_ex, b, np.array([-1e3]))[0])
            c.check(f"{name}: the explicit tableau really is explicit "
                    f"(|R_ex(-1e3)| = {big:.2e} is unbounded)", big > 1e3)


# ------------------------------------------- throw-away equation sets for the ODE tests

@contextlib.contextmanager
def _registered(name, recipe):
    """Add a test-only entry to equation_registry for the duration of a test."""
    equation_registry[name] = recipe
    try:
        yield
    finally:
        equation_registry.pop(name, None)


def _one_field_zero_ic(x, y):
    return jnp.zeros((1,) + jnp.broadcast_shapes(x.shape, y.shape))


def _toy_L(kgrid, params):
    # diagonal (1, 1, nkx, nky) relaxation operator L = -(gam0 + gam1*k_perp^2)
    g0, g1 = params.eqpars["gam0"], params.eqpars["gam1"]
    return (-(g0 + g1 * kgrid.ksq)).reshape(1, 1, *kgrid.ksq.shape)


def _toy_set_timestep(grads, params):
    return params.dt


def _toy_grad(state, kgrid, params):
    return state.fields


def _real_mode(params, f):
    """k-space representation of the real (1,1,nx,ny) field built by f(x,y)."""
    # dtype=ftype (PRECISION_PLAN.md A2 pattern, e.g. run.py::initialize): x64 is
    # unconditionally on, so an unpinned linspace is a STRONG float64 array that poisons
    # f(x,y) and its fft -- the resulting Fh would be complex128 under TARANIS_PRECISION=32,
    # which is exactly the leak construct_rhs's dtype tripwire (A3) now catches when this
    # helper's output gets _replace'd straight into state.fields, bypassing initialize's
    # own choke-point .astype(ctype).
    x = jnp.linspace(0, params.Lx, params.nx, endpoint=False,
                     dtype=_precision.ftype).reshape(1, 1, -1, 1)
    y = jnp.linspace(0, params.Ly, params.ny, endpoint=False,
                     dtype=_precision.ftype).reshape(1, 1, 1, -1)
    return grids.fft(jnp.broadcast_to(f(x, y), (1, 1, params.nx, params.ny)), params)


def _mfg_F(params):
    return _real_mode(params, lambda x, y: jnp.cos(x) + 0.5 * jnp.sin(2 * y))


def _mfg_exact(params, t):
    # u(t) = a(t)*F with a(t) = 1 + sin(t)/2
    return (1.0 + 0.5 * jnp.sin(t)) * _mfg_F(params)


def _mfg_N(fields, t, kgrid, params):
    # explicit part: the nonlinearity u^2 (formed in real space) plus the manufactured
    # source that makes u = a(t)F an EXACT solution of dt u = L u + u^2 + S(t)
    Fh = _mfg_F(params)
    F2h = grids.fft(grids.ifft(Fh, params)**2, params)
    a = 1.0 + 0.5 * jnp.sin(t)
    src = (0.5 * jnp.cos(t)) * Fh - kgrid.lin_L * (a * Fh) - (a * a) * F2h
    return grids.fft(grids.ifft(fields, params)**2, params) + src


def _mfg_term(state, grads, kgrid, params, halo=None):
    # PRECISION_PLAN.md A3: this manufactured term func is the second place (besides
    # run.py::_advance_forcing) found to read state.t (float64, every precision) straight
    # into field math -- _mfg_N mixes it with Fh (ftype/ctype-pinned) via jnp.sin/cos,
    # which promotes the whole RHS to complex128 under TARANIS_PRECISION=32 and trips
    # construct_rhs's dtype tripwire one scan step later. Downcast before the field mix,
    # exactly like _advance_forcing's dt.
    return _mfg_N(state.fields, state.t.astype(_precision.ftype), kgrid, params)


_MFG_RECIPE = EquationRecipe(set_timestep_func=_toy_set_timestep, term_funcs=(_mfg_term,),
                             grad_func=_toy_grad, nfields=1, linear_matrix_func=_toy_L)


def _stiff_drive(params, t):
    # G(t): the O(1) explicit drive of the stiff mode. NB it is the DRIVE that is O(1) and
    # the quasi-static state g = G/gamma that is small -- that is the physical situation
    # (GDI: an O(1) nonlinear/adiabatic drive against a large gamma_par), and it is what
    # separates the schemes. An O(gamma) explicit source instead would just be a hard
    # explicit-RK problem and would penalise IF and IMEX alike.
    G = _real_mode(params, lambda x, y: jnp.cos(x) * (1.0 + 0.3 * jnp.cos(y)))
    return jnp.cos(params.eqpars["omega"] * t) * G


def _stiff_target(params, t):
    # g(t): the quasi-static state the relaxation is chasing, i.e. the instantaneous root of
    # 0 = -gamma*u + G(t) + nl*u^2. That is G/gamma corrected by the (weak) nonlinearity,
    # reached by Picard iteration -- the contraction factor is ~2*nl*|u|/gamma ~ 4%, so five
    # sweeps are converged to well under the O(omega/gamma) lag the test is measuring.
    gam, nl = params.eqpars["gam0"], params.eqpars["nl"]
    G = _stiff_drive(params, t)
    u = G / gam
    for _ in range(5):
        u = (G + nl * grids.fft(grids.ifft(u, params)**2, params)) / gam
    return u


def _stiff_term(state, grads, kgrid, params, halo=None):
    # G(t) + weak nonlinearity: with L = -gamma this is u' = -gamma(u - g(t)) + nl*u^2.
    # nl is scaled outside as a fraction of gamma^2, so that at the quasi-static amplitude
    # |u| ~ |G|/gamma the nonlinear term is that fraction of the drive -- weak but present.
    nl = params.eqpars["nl"]
    # PRECISION_PLAN.md A3: same state.t-into-field-math leak as _mfg_term above --
    # downcast before _stiff_drive multiplies it against G (ftype/ctype-pinned).
    return (_stiff_drive(params, state.t.astype(_precision.ftype))
            + nl * grids.fft(grids.ifft(state.fields, params)**2, params))


_STIFF_RECIPE = EquationRecipe(set_timestep_func=_toy_set_timestep, term_funcs=(_stiff_term,),
                               grad_func=_toy_grad, nfields=1, linear_matrix_func=_toy_L)


def _toy_params(eqtype, dt, eqpars, nx=8, ny=8):
    # fixed dt (adaptive_timestep=False), 2D, one field
    return jr.Parameters(nx=nx, ny=ny, Lx=2 * math.pi, Ly=2 * math.pi, cfl_safety=0.5,
                         dt=dt, adaptive_timestep=False, dims=2, eqtype=eqtype,
                         eqpars=dict(eqpars))


def _toy_run(params, kgrid, u0, nsteps, schemestr):
    state = jr.initialize(_one_field_zero_ic, params)._replace(fields=u0)
    return _run(state, kgrid, params, nsteps, schemestr)


# ------------------------------------------- low-storage vs dense-tableau reference

def _dense_imex_steps(u, t0, dt, nsteps, L, N, sch, kgrid, params):
    """Textbook dense-tableau IMEX-RK for a DIAGONAL L. Deliberately not low storage."""
    A_im = np.array(sch.a_im, dtype=np.float64)
    A_ex = np.array(sch.a_ex, dtype=np.float64)
    b = np.array(sch.b, dtype=np.float64)
    cc = np.array(sch.c, dtype=np.float64)
    s = len(b)
    L = jnp.asarray(L)
    t = t0
    for _ in range(nsteps):
        Z, Y = [None]*s, [None]*s
        for k in range(s):
            acc = u
            for j in range(k):
                acc = acc + (dt * A_im[k, j]) * Z[j] + (dt * A_ex[k, j]) * Y[j]
            U = acc / (1.0 - (dt * A_im[k, k]) * L)     # (I - a_kk dt L)^-1, L diagonal
            Z[k] = L * U
            Y[k] = N(U, t + cc[k] * dt, kgrid, params)
        for k in range(s):
            u = u + (dt * b[k]) * (Z[k] + Y[k])
        t = t + dt
    return u, t


def test_low_storage_matches_dense_tableau():
    # The point of a low-storage formulation is that it is ALGEBRAICALLY the tableau. If a
    # register recurrence is wrong, this is where it shows.
    tol = 1e-12 if _fp64() else 3e-4
    dt, nsteps = 0.03, 6
    with _registered("IMEXMFG", _MFG_RECIPE), checks() as c:
        params = _toy_params("IMEXMFG", dt, dict(gam0=0.5, gam1=0.1))
        kgrid = jr.setup_kgrids(params)
        u0 = _mfg_exact(params, 0.0)
        for name in IMEX_SCHEMES:
            _, sch = get_scheme(name)
            end = _toy_run(params, kgrid, u0, nsteps, name)
            ref, _ = _dense_imex_steps(u0, 0.0, dt, nsteps, kgrid.lin_L, _mfg_N, sch,
                                       kgrid, params)
            err = float(jnp.max(jnp.abs(end.fields - ref)) / jnp.max(jnp.abs(ref)))
            c.check(f"{name}: the {sch.registers}-register [{sch.structure}] stepper == the "
                    f"dense tableau after {nsteps} steps", err < tol, f"rel err {err:.3e}")


# ------------------------------------------------------------- (a) convergence order

def test_convergence_order():
    # Manufactured NONLINEAR problem with an exact solution: dt u = L u + u^2 + S(t) is
    # solved exactly by u = (1 + sin(t)/2)F. L = -(0.5 + 0.1 k_perp^2) is diagonal and
    # mild, so the schemes show their asymptotic (non-stiff) order.
    t_end = 0.4
    # fp32: a 3rd-order scheme is already on the round-off floor (~1e-7 relative) at
    # dt = 0.02, so the refinement ladder stops well short of it there.
    nsteps = (10, 20, 40, 80) if _fp64() else (3, 5, 8, 12)
    slack = 0.25 if _fp64() else 0.7
    with _registered("IMEXMFG", _MFG_RECIPE), checks() as c:
        for name in IMEX_SCHEMES:
            _, sch = get_scheme(name)
            hs, errs = [], []
            for n in nsteps:
                dt = t_end / n
                params = _toy_params("IMEXMFG", dt, dict(gam0=0.5, gam1=0.1))
                kgrid = jr.setup_kgrids(params)
                end = _toy_run(params, kgrid, _mfg_exact(params, 0.0), n, name)
                ref = _mfg_exact(params, float(end.t))
                errs.append(float(jnp.max(jnp.abs(end.fields - ref))
                                  / jnp.max(jnp.abs(ref))))
                hs.append(dt)
            order = fit_order(hs, errs)
            c.check(f"{name}: measured order {order:.2f} ~ {sch.order} "
                    f"(errors {['%.2e' % e for e in errs]})",
                    abs(order - sch.order) < slack)


# --------------------------------------------------------- (b) stiff quasi-static test

def _quasi_static_error(schemestr, gamma, dt, nsteps, omega=1.0, nl_frac=0.02):
    """max|u - g(t_end)| / max|g| for one scheme on u' = -gamma(u - g(t)) + nl*u^2."""
    params = _toy_params("IMEXSTIFF", dt, dict(gam0=gamma, gam1=0.0, omega=omega,
                                               nl=nl_frac * gamma * gamma))
    kgrid = jr.setup_kgrids(params)
    u0 = _stiff_target(params, 0.0)          # start ON the quasi-static manifold
    end = _toy_run(params, kgrid, u0, nsteps, schemestr)
    g = _stiff_target(params, float(end.t))
    return float(jnp.max(jnp.abs(end.fields - g)) / jnp.max(jnp.abs(g))), float(end.t)


def test_stiff_quasi_static():
    # THE motivating physics (plans/GDI_PLAN.md "Design background"): a mode relaxed at rate
    # gamma against an O(1) explicit drive G(t). The solution sits on the quasi-static
    # manifold u = g + O(omega/gamma), g = the instantaneous root of the RHS. An integrating
    # factor cannot reproduce that at gamma*dt >> 1: it applies a fixed-node quadrature to a
    # boundary-layer integral and gives u ~ dt*G instead of G/gamma, i.e. a factor ~gamma*dt
    # too large. An L-stable, stiffly accurate IMEX scheme gets it right.
    #
    # The sharp statement, and what is asserted: hold gamma*dt = 10 (deep in the stiff
    # regime) and take gamma from 200 to 2000. |u - g|/|g| must fall like 1/gamma for the
    # IMEX schemes -- that IS "u = g + O(1/gamma)" -- while the IF schemes' error stays
    # O(1). Their numbers are measured and RECORDED in the check lines rather than asserted,
    # since documenting them is the point.
    #
    # NB the IMEX schemes' error is O(dt) at fixed gamma*dt, not O(dt^order): the classic
    # stage-order-one order reduction in the stiff limit. IMEXRKCB3f has stage order two on
    # its IMPLICIT part only, and the drive here lives entirely in the explicit part, so it
    # does not help; IMEXRKCB3e simply has the smallest stiff error constant of the four.
    cases = ((200.0, 0.05, 40), (2000.0, 0.005, 400))     # (gamma, dt, nsteps), t_end = 2
    with _registered("IMEXSTIFF", _STIFF_RECIPE), checks() as c:
        errs = {n: [_quasi_static_error(n, *cs)[0] for cs in cases]
                for n in ("lsrk33", "lsrk54", "rk44") + IMEX_SCHEMES}
        if_report = ", ".join(f"{k} {errs[k][1]:.3e}" for k in ("lsrk33", "lsrk54", "rk44"))
        for name in IMEX_SCHEMES:
            e200, e2000 = errs[name]
            c.check(f"{name}: |u - g|/|g| = {e200:.3e} -> {e2000:.3e} as gamma goes 200 -> "
                    f"2000 at fixed gamma*dt = 10, i.e. u = g + O(1/gamma) "
                    f"[IF at gamma=2000 on the SAME problem: {if_report}]",
                    e2000 < 0.2 * e200 and e2000 < 0.02)
        for name in ("lsrk33", "lsrk54", "rk44"):
            e200, e2000 = errs[name]
            c.check(f"{name} (IF): error does NOT vanish with gamma ({e200:.3e} -> "
                    f"{e2000:.3e}): the integrating factor misses the quasi-static balance",
                    e2000 > 0.5 * e200)
        worst = max(errs[n][1] for n in IMEX_SCHEMES)
        best_if = min(errs[n][1] for n in ("lsrk33", "lsrk54", "rk44"))
        c.check(f"at gamma = 2000 every IMEX scheme beats every IF scheme by >5x "
                f"(worst IMEX {worst:.3e}, best IF {best_if:.3e})", worst * 5 < best_if)


# ------------------------------------------------------- (c) forced RMHD 2D regression

# Fixed dt on purpose: every scheme then sees the SAME O-U forcing realization and ends at
# exactly the same t, so the comparison is a truncation-error comparison and nothing else.
_FORCED = dict(dims=2, nx=32, ny=32, diss=(2e-3, 2e-3), hyper=1, forcing=True,
               forcing_mode="elsasser", forcing_power_elsasser=(0.5, 0.5),
               forcing_tau=0.5, fshell=(1, 3), forcing_seed=7, cfl_safety=0.4,
               adaptive_timestep=False, dt=0.02)


def _forced_energy(schemestr, nsteps=250):
    params, kgrid = ctx(**_FORCED)
    end = _run(make_state(params, ic=zero_ic_2d), kgrid, params, nsteps, schemestr)
    Ek, Em = energy(end, kgrid, params)
    return float(Ek) + float(Em), float(end.t)


@pytest.mark.fp64
def test_rmhd_forced_2d_tracks_lsrk54():
    # Statistical regression: total energy at t = 5 is set by the injection rate, and the
    # schemes must agree on it to a few percent.
    ref, t_ref = _forced_energy("lsrk54")
    with checks() as c:
        for name in IMEX_SCHEMES:
            got, t_got = _forced_energy(name)
            rel = abs(got - ref) / ref
            c.check(f"{name}: forced-2D total energy {got:.5f} at t = {t_got:.2f} is within "
                    f"5% of lsrk54's {ref:.5f} at t = {t_ref:.2f}", rel < 0.05,
                    f"rel diff {rel:.3%}")


# ------------------------------------------------- scan vs unrolled ([2R] stage loop)

def test_imex2r_scan_vs_unrolled():
    # imex2r_advance has scan (params.lsrk_scan=True, default) and unrolled stage loops;
    # like lsrk they agree to round-off, NOT bitwise (XLA fuses the two graphs
    # differently). imexcb3f does not read the knob ([3R] is unrolled only): its two runs
    # trace the identical graph and must match bitwise.
    tol = 1e-13 if _fp64() else 1e-5
    nsteps = 30
    with checks() as c:
        for name in ("imexcb2", "imexcb3e", "imexcb3c"):
            p_s, kg_s = ctx(lsrk_scan=True, **_FORCED)
            p_u, kg_u = ctx(lsrk_scan=False, **_FORCED)
            end_s = _run(make_state(p_s, ic=zero_ic_2d), kg_s, p_s, nsteps, name)
            end_u = _run(make_state(p_u, ic=zero_ic_2d), kg_u, p_u, nsteps, name)
            fs, fu = np.asarray(end_s.fields), np.asarray(end_u.fields)
            rel = float(np.max(np.abs(fs - fu)) / np.max(np.abs(fu)))
            c.check(f"{name}: lsrk_scan True/False agree to round-off "
                    f"(rel {rel:.2e} < {tol:.0e})", rel < tol)
        p_s, kg_s = ctx(lsrk_scan=True, **_FORCED)
        p_u, kg_u = ctx(lsrk_scan=False, **_FORCED)
        end_s = _run(make_state(p_s, ic=zero_ic_2d), kg_s, p_s, nsteps, "imexcb3f")
        end_u = _run(make_state(p_u, ic=zero_ic_2d), kg_u, p_u, nsteps, "imexcb3f")
        c.check("imexcb3f: does not read lsrk_scan (identical graph -> bitwise match)",
                bool(np.array_equal(np.asarray(end_s.fields), np.asarray(end_u.fields))))


# ------------------------------------ (extra) spectral-z 3D: the putzer2 solve_shifted path

@pytest.mark.fp64
def test_z_spectral_3d_runs_and_converges_to_if():
    # z_spectral RMHD puts +-i*kz off-diagonals in L, so this exercises putzer2's
    # solve_shifted with a REAL kz extent. An L-stable implicit treatment DAMPS those waves
    # at |kz|*dt >~ 1 -- expected, that is what L-stability means -- so the assertions are
    # only: it runs, it stays finite, and it converges to the (wave-exact) IF answer as
    # dt -> 0. No wave-fidelity claim at large dt.
    t_end = 0.4
    with checks() as c:
        for name in ("imexcb3e", "imexcb3f"):
            errs, hs = [], []
            for n in (8, 16, 32):
                dt = t_end / n
                params = fresh_params(dims=3, nz=8, nx=16, ny=16, z_spectral=True,
                                      diss=(0.02, 0.02), hyper=1, dt=dt,
                                      adaptive_timestep=False)
                kgrid = jr.setup_kgrids(params)
                imex = _run(make_state(params, ic=multimode_ic), kgrid, params, n, name)
                ifr = _run(make_state(params, ic=multimode_ic), kgrid, params, n, "lsrk54")
                a, b = np.asarray(imex.fields), np.asarray(ifr.fields)
                c.check(f"{name}: z_spectral 3D run at dt = {dt:.4f} stays finite",
                        bool(np.all(np.isfinite(a))))
                errs.append(float(np.max(np.abs(a - b)) / np.max(np.abs(b))))
                hs.append(dt)
            order = fit_order(hs, errs)
            c.check(f"{name}: converges to the IF (wave-exact) result as dt -> 0 "
                    f"(diffs {['%.2e' % e for e in errs]}, slope {order:.2f})",
                    errs[-1] < errs[0] / 3.0 and order > 0.8)


# ------------------------------------------------------------------- registry wiring

def test_imex_schemes_are_registered():
    with checks() as c:
        for name in IMEX_SCHEMES:
            stepper, sch = get_scheme(name)
            want = imex2r_advance if sch.structure == "2R" else imex3r_advance
            c.check(f"{name}: resolves to {want.__name__}, order {sch.order}, "
                    f"{sch.registers} registers",
                    stepper is want and isinstance(sch, timestepping.IMEX_Scheme)
                    and sch.registers == (3 if sch.structure == "2R" else 4))
        # the IF production schemes must not have moved
        c.check("the IF production schemes are untouched",
                get_scheme("lsrk54")[0] is timestepping.lsrk_advance
                and get_scheme("rk44")[1] is None)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
