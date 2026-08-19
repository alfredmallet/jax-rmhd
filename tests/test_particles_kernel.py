# Kernel gates 1-3 of plans/TESTPART_PLAN.md §6 plus the interpolation unit checks: the
# Boris pusher and the periodic gather driven by ANALYTIC fields, no solver in the loop.
#
#   1. Gyration      uniform B0*zhat, E = 0: |v| conserved to round-off (the Boris
#                    rotation is norm-exact), gyrophase error ~ (Omega*dt)^2, and the
#                    sense of rotation (qm > 0, B = +zhat rotates v clockwise seen from
#                    +z, i.e. dv_x/dt = +qm*v_y*B0).
#   2. E x B drift   uniform E_perp, B: gyro-averaged drift = E x B / B^2, both
#                    components, signs included.
#   3. grad-B drift  static analytic B(x), E = 0: measured gyrocentre drift against the
#                    textbook v_gradB for a pure |B| gradient, and against 2*v_gradB for
#                    the RMHD b_perp = zhat x grad(psi) structure, where the field shear
#                    contributes at the same order (derived where the gate is defined).
#                    Convergent in rho/L.
#  10. mirror force  static analytic B = B0*zhat + zhat x grad(psi(x,z)) on a 3D grid,
#                    E = 0: the PARALLEL dynamics gates 1-3 arrange away. The reflection
#                    point matches the mu prediction |B|_turn = |B|_0*|v|^2/v_perp0^2 over
#                    a sweep of v_par0, mu's excursion falls at first order in rho*k_z, a
#                    particle inside the loss cone passes instead of reflecting, and the
#                    SAME field without z structure leaves v_par constant. Drives the
#                    trilinear 3D gather.
#  11. varying dt    the KDK driver under a jittered dt sequence (x and v are synchronized
#                    at step boundaries, so dt may change from step to step): |v| still
#                    exact in a uniform field, the orbit still converges to the analytic
#                    one at second order in max(dt), and the E x B drift is still exact.
#
# Not precision-marked: particle state is fp64 at every TARANIS_PRECISION, and gates 1, 2
# and 11 use exactly representable field constants, so they are bitwise identical in both
# sessions. The analytic field arrays of gates 3 and 10 are stored at field precision and
# so differ at fp32 round-off; their tolerances are far above it (the smallest number
# either gate asserts on is ~1e-5, against an fp32 field error of ~6e-8), so they pass in
# both sessions. Only the interpolation checks take an explicitly precision-dependent
# tolerance.
#
# Gates 2, 3 and 10 fire four particles from the same point at quadrature gyrophases. For
# uniform fields the Boris map is affine in v, so that average is EXACTLY the drift
# motion (the gyration cancels identically); in the weakly nonuniform field of gate 3 it
# suppresses the gyration to the 4th harmonic, and a least-squares slope over hundreds of
# gyroperiods removes the rest; in gate 10 it turns the O(rho) gyrophase spread of a
# single-particle measurement into an average whose scaling can be read.
from _rmhd_testing import bootstrap

bootstrap()

import math
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from _rmhd_testing import checks, ctx, fit_order
from taranis import _precision
from taranis.particles import boris, interp

# field-precision floor for the interpolation checks (the particles themselves are fp64)
_FEPS = float(np.finfo(np.float64 if _precision.precision == "64" else np.float32).eps)

_QUAD = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)  # quadrature gyrophases


def _params(**kw):
    p, _ = ctx(dims=2, **kw)
    return p


def _uniform(params, vec):
    # (3, 1, nx, ny) grid array holding the constant vector `vec`
    return jnp.broadcast_to(jnp.asarray(vec, _precision.ftype).reshape(3, 1, 1, 1),
                            (3, 1, params.nx, params.ny))


@partial(jax.jit, static_argnums=(6, 7))
def _orbit(x, v, E3, B3, qm, dt, params, nsteps):
    def step(carry, _):
        x, v = carry
        return boris.push(x, v, E3, B3, qm, dt, params), None
    (x, v), _ = jax.lax.scan(step, (x, v), None, length=nsteps)
    return x, v


# ------------------------------------------------------------------ gate 1: gyration

_G1 = dict(nx=32, ny=32, Lx=2.0 * math.pi, Ly=2.0 * math.pi)
_G1_QM = 2.0          # Omega = qm*B0 = 2, exactly representable at fp32
_G1_V = 0.5
_G1_NGYRO = 10000
_G1_SPG = 50          # steps per gyration: Omega*dt = 2*pi/50


def test_gyration_norm_and_sense():
    """|v| conserved to round-off over 1e4 gyrations, and v rotates clockwise (from +z)."""
    params = _params(**_G1)
    B3, E3 = _uniform(params, (0.0, 0.0, 1.0)), _uniform(params, (0.0, 0.0, 0.0))
    omega = _G1_QM * 1.0
    dt = 2.0 * math.pi / (omega * _G1_SPG)
    # starts at the origin so the orbit (radius v/Omega) straddles x = y = 0 and the
    # gather sees folded, negative positions every gyration
    x0 = jnp.zeros((1, 3), dtype=jnp.float64)
    v0 = jnp.asarray([[_G1_V, 0.0, 0.0]], dtype=jnp.float64)
    x, v = _orbit(x0, v0, E3, B3, _G1_QM, dt, params, _G1_NGYRO * _G1_SPG)
    drift = abs(float(jnp.linalg.norm(v, axis=1)[0]) / _G1_V - 1.0)

    # a quarter turn from v = +x must land on -y for qm > 0, B = +z (100 steps per
    # gyration so that a quarter period is a whole number of steps)
    dtq = 2.0 * math.pi / (omega * 100)
    _, vq = _orbit(x0, v0, E3, B3, _G1_QM, dtq, params, 25)
    vq = np.asarray(vq)[0]
    print(f"gate 1: |v| relative drift over {_G1_NGYRO} gyrations = {drift:.3e}; "
          f"quarter-turn v = {vq}")
    with checks() as c:
        c.check(f"|v| conserved to {drift:.2e} over {_G1_NGYRO} gyrations (push+gather, "
                f"periodic wrap)", drift < 1e-12, f"drift={drift}")
        c.check("quarter turn from +x lands on -y (clockwise viewed from +z)",
                vq[1] < -0.99 * _G1_V and abs(vq[0]) < 0.05 * _G1_V, f"v={vq}")
        c.check("no out-of-plane velocity is generated", abs(vq[2]) < 1e-14, f"v_z={vq[2]}")


def _phase_error(params, spg, ngyro):
    # gyrophase of v after ngyro exact gyroperiods; the exact answer is the initial phase
    omega = _G1_QM * 1.0
    dt = 2.0 * math.pi / (omega * spg)
    x0 = jnp.zeros((1, 3), dtype=jnp.float64)
    v0 = jnp.asarray([[_G1_V, 0.0, 0.0]], dtype=jnp.float64)
    B3, E3 = _uniform(params, (0.0, 0.0, 1.0)), _uniform(params, (0.0, 0.0, 0.0))
    _, v = _orbit(x0, v0, E3, B3, _G1_QM, dt, params, ngyro * spg)
    v = np.asarray(v)[0]
    return abs(math.atan2(v[1], v[0])), omega * dt


def test_gyrophase_error_order():
    """Phase error after a fixed total time scales as (Omega*dt)^2."""
    params = _params(**_G1)
    spgs = (50, 100, 200)
    errs, odts = zip(*[_phase_error(params, s, 20) for s in spgs])
    order = fit_order(odts, errs)
    print(f"gate 1b: Omega*dt {['%.4f' % o for o in odts]} -> phase error "
          f"{['%.3e' % e for e in errs]}, fitted order {order:.3f}")
    with checks() as c:
        c.check(f"gyrophase error order {order:.3f} in (1.8, 2.2)", 1.8 < order < 2.2,
                f"errs={errs}")
        c.check("phase error decreases with dt",
                all(a > b for a, b in zip(errs, errs[1:])), f"errs={errs}")


# ------------------------------------------------------------------- gate 2: E x B

def _exb_drift(params, Evec, qm=2.0, B0=1.0, vperp=0.5, ngyro=40, spg=50):
    B3, E3 = _uniform(params, (0.0, 0.0, B0)), _uniform(params, Evec)
    omega = qm * B0
    dt = 2.0 * math.pi / (omega * spg)
    nsteps = ngyro * spg
    u = np.asarray([Evec[1] * B0, -Evec[0] * B0, 0.0]) / B0**2
    x0 = jnp.zeros((4, 3), dtype=jnp.float64)
    v0 = jnp.asarray([[u[0] + vperp * math.cos(p), u[1] - vperp * math.sin(p), 0.0]
                      for p in _QUAD], dtype=jnp.float64)
    x, _ = _orbit(x0, v0, E3, B3, qm, dt, params, nsteps)
    return np.asarray(jnp.mean(x, axis=0)) / (nsteps * dt), u, nsteps * dt


def test_exb_drift():
    """Gyro-averaged drift = E x B / B^2, magnitudes and signs.

    This pins plans/TESTPART_PLAN.md §2's convention: with E_perp = -B0*grad(phi) the
    drift is zhat x grad(phi) = u, the code's flow velocity. E_y > 0 with B = B0*zhat
    must drift along +x, and E_x > 0 along -y.
    """
    params = _params(**_G1)
    meas_y, pred_y, tend = _exb_drift(params, (0.0, 0.25, 0.0))
    meas_x, pred_x, _ = _exb_drift(params, (0.25, 0.0, 0.0))
    scale = 0.25
    ey = np.max(np.abs(meas_y - pred_y)) / scale
    ex = np.max(np.abs(meas_x - pred_x)) / scale
    print(f"gate 2: E=(0,0.25,0) drift {meas_y} vs {pred_y} (rel {ey:.2e}); "
          f"E=(0.25,0,0) drift {meas_x} vs {pred_x} (rel {ex:.2e})")
    with checks() as c:
        c.check(f"E_y > 0 drifts along +x at E_y/B0, rel err {ey:.2e} < 1e-3", ey < 1e-3,
                f"{meas_y} vs {pred_y}")
        c.check("E_y > 0 drift is along +x (sign)", meas_y[0] > 0.5 * pred_y[0], f"{meas_y}")
        c.check(f"E_x > 0 drifts along -y at E_x/B0, rel err {ex:.2e} < 1e-3", ex < 1e-3,
                f"{meas_x} vs {pred_x}")
        c.check("E_x > 0 drift is along -y (sign)", meas_x[1] < 0.5 * pred_x[1], f"{meas_x}")
        # the drift carries the particles several boxes away: push() leaves positions
        # unfolded and gather() folds them
        c.check(f"positions stay unfolded ({abs(meas_y[0]) * tend:.1f} >> Lx = "
                f"{params.Lx:.2f})", abs(meas_y[0]) * tend > 4.0 * params.Lx,
                f"displacement={abs(meas_y[0]) * tend}")


def test_substeps_split_the_step_exactly():
    """push(..., substeps=n) is exactly n consecutive push() calls at dt/n, on fields
    the gather actually has to interpolate."""
    params = _params(**_G1)
    x, y, _z = interp.grid_coords(params)
    xg, yg = x.reshape(1, -1, 1), y.reshape(1, 1, -1)
    ones = jnp.ones((1, params.nx, params.ny))
    B3 = jnp.stack([0.1 * jnp.sin(yg) * ones, 0.1 * jnp.sin(xg) * ones,
                    ones]).astype(_precision.ftype)
    E3 = jnp.stack([0.05 * jnp.cos(yg) * ones, 0.02 * jnp.sin(xg + yg) * ones,
                    0.03 * jnp.cos(xg) * ones]).astype(_precision.ftype)
    rng = np.random.default_rng(3)
    x0 = jnp.asarray(rng.uniform(-params.Lx, 2.0 * params.Lx, (8, 3)))
    v0 = jnp.asarray(rng.normal(size=(8, 3)))
    dt = 0.125
    xa, va = boris.push(x0, v0, E3, B3, _G1_QM, dt, params, substeps=4)
    xb, vb = x0, v0
    for _ in range(4):
        xb, vb = boris.push(xb, vb, E3, B3, _G1_QM, 0.25 * dt, params)
    dx = float(jnp.max(jnp.abs(xa - xb)))
    dv = float(jnp.max(jnp.abs(va - vb)))
    print(f"substeps: max |dx| {dx:.2e}, max |dv| {dv:.2e}")
    with checks() as c:
        c.check(f"substeps=4 reproduces four dt/4 steps ({dx:.1e}, {dv:.1e})",
                dx == 0.0 and dv == 0.0, f"dx={dx} dv={dv}")


# ---------------------------------------------------------------- gate 3: grad-B drift
#
# Two static analytic fields, both driven through push+gather, both with grad|B| along
# xhat so the drift is along +y (B x grad|B|) and the gyrocentre never leaves its
# neighbourhood in x:
#
#   (a) B = (0, 0, B0(1 + a*sin(kx))) — the textbook configuration. The measured drift
#       IS v_gradB = (v_perp^2/(2*qm*|B|)) (B x grad|B|)/|B|^2, to a fraction of a percent.
#
#   (b) B = B0*zhat + zhat x grad(psi), psi = -eps*(Lx/2pi)*cos(kx), i.e. b_perp =
#       (0, eps*sin(kx), 0) — the RMHD field structure of plans/TESTPART_PLAN.md §2. Here
#       the naive v_gradB is NOT the leading drift: b_perp shears at O(eps) while |B| =
#       sqrt(B0^2 + b_y^2) varies only at O(eps^2), so second-order-in-shear terms enter
#       at the same order as the |B| gradient. The exact drift follows from the two
#       invariants of an (x)-only field: v_y = -qm*B0*x + const and v_z = qm*Psi(x) + const
#       with Psi' = b_y, plus <dv_x/dt> = 0, which give the identity drift = <v_z*b_y>.
#       Expanding about the gyrocentre in delta = x - x0 (<delta> = rho*sin(phi),
#       <delta^2> = rho^2 after averaging the four quadrature gyrophases) leaves three
#       terms of order v_perp*eps^2*k*rho*sin(kx0)*cos(kx0):
#           qm*b*b'*<delta^2>        = +1     b_y sampled across the orbit
#           qm*b*b'*<delta^2>/2      = +1/2   second order in the v_z invariant
#           v_z(0)*b'*<delta>        = -1/2   the v_par = 0 launch tilt
#       against the naive v_gradB = +1/2 in the same units: the drift is 2*v_gradB. (The
#       -1/2 is launch-dependent: firing the same ensemble with v_z(0) = 0 instead of
#       v_par = 0 gives 3*v_gradB, which the numerics reproduce.) This is a standing
#       warning for the coupled phase — particles in RMHD fields do not obey the naive
#       grad-B formula.
#
# Both cases launch with v_par = 0. Because b_x = 0 in (b) and B is purely parallel in
# (a), bhat.grad|B| = 0: there is no mirror force, v_par stays 0 and no curvature drift
# contaminates the measurement.
_G3 = dict(nx=256, ny=8, Lx=2.0 * math.pi, Ly=2.0 * math.pi)
_G3_AMP = 0.05        # a (case a) / eps (case b)
_G3_B0 = 1.0
_G3_VPERP = 1.0
_G3_NGYRO = 400
_G3_SPG = 50
_G3_QMS = (10.0, 20.0, 40.0)     # rho/L = 1/10, 1/20, 1/40 with L = Lx/2pi = 1


def _g3_grid(params, vecs):
    # (3,1,nx,ny) field from three functions of x
    x, _y, _z = interp.grid_coords(params)
    cols = [jnp.broadcast_to(jnp.asarray(f(x)).reshape(1, -1, 1), (1, params.nx, params.ny))
            for f in vecs]
    return jnp.stack(cols).astype(_precision.ftype)


def _field_bz(params):
    k = 2.0 * math.pi / params.Lx
    return _g3_grid(params, [lambda x: 0.0 * x, lambda x: 0.0 * x,
                             lambda x: _G3_B0 * (1.0 + _G3_AMP * jnp.sin(k * x))])


def _analytic_bz(params, x0):
    k = 2.0 * math.pi / params.Lx
    bz = _G3_B0 * (1.0 + _G3_AMP * math.sin(k * x0))
    return (np.array([0.0, 0.0, bz]), abs(bz),
            np.array([_G3_B0 * _G3_AMP * k * math.cos(k * x0), 0.0, 0.0]))


def _field_bperp(params):
    k = 2.0 * math.pi / params.Lx
    return _g3_grid(params, [lambda x: 0.0 * x, lambda x: _G3_AMP * jnp.sin(k * x),
                             lambda x: 0.0 * x + _G3_B0])


def _analytic_bperp(params, x0):
    k = 2.0 * math.pi / params.Lx
    by = _G3_AMP * math.sin(k * x0)
    dby = _G3_AMP * k * math.cos(k * x0)
    Bmag = math.hypot(by, _G3_B0)
    return (np.array([0.0, by, _G3_B0]), Bmag, np.array([by * dby / Bmag, 0.0, 0.0]))


def _g3_drift(params, B3, analytic, qm, x0):
    """(measured drift, naive v_gradB, rho) for four particles launched with v_par = 0."""
    B, Bmag, grad = analytic(params, x0)
    omega = qm * Bmag
    dt = 2.0 * math.pi / (omega * _G3_SPG)
    nsteps = _G3_NGYRO * _G3_SPG
    bhat = B / Bmag
    e1 = np.array([1.0, 0.0, 0.0])          # b has no x component, so xhat is perpendicular
    e2 = np.cross(bhat, e1)
    x_init = jnp.asarray(np.tile([x0, 0.0, 0.0], (4, 1)), dtype=jnp.float64)
    v_init = jnp.asarray(np.stack([_G3_VPERP * (math.cos(p) * e1 + math.sin(p) * e2)
                                   for p in _QUAD]), dtype=jnp.float64)
    E3 = jnp.zeros_like(B3)

    def gc_y(x, v):
        # guiding centre y: X = x + (v x B)/(qm*|B|^2), averaged over the four particles.
        # The quadrature average kills every gyroharmonic but the 4th; the least-squares
        # slope over _G3_NGYRO gyroperiods removes what is left.
        Bp = interp.gather(B3, x, params)
        bsq = jnp.sum(Bp * Bp, axis=1)
        return jnp.mean(x[:, 1] + (v[:, 2] * Bp[:, 0] - v[:, 0] * Bp[:, 2]) / (qm * bsq))

    @partial(jax.jit, static_argnums=(2,))
    def run(x, v, params):
        def step(carry, _):
            x, v = carry
            x, v = boris.push(x, v, E3, B3, qm, dt, params)
            return (x, v), gc_y(x, v)
        _, ys = jax.lax.scan(step, (x, v), None, length=nsteps)
        return ys

    ys = np.asarray(run(x_init, v_init, params))
    t = dt * np.arange(1, nsteps + 1)
    measured = float(np.polyfit(t, ys, 1)[0])
    v_gradb = _G3_VPERP**2 * np.cross(B, grad)[1] / (2.0 * qm * Bmag**3)
    return measured, v_gradb, _G3_VPERP / omega


def _g3_cell_centre(params):
    # gyrocentre at a cell centre near Lx/8, where sin(kx)cos(kx) is near its maximum
    dx = params.Lx / params.nx
    return (math.floor(0.25 * math.pi / dx) + 0.5) * dx


def _g3_sweep(params, B3, analytic, factor, label):
    x0 = _g3_cell_centre(params)
    rows = []
    for qm in _G3_QMS:
        meas, v_gradb, rho = _g3_drift(params, B3, analytic, qm, x0)
        pred = factor * v_gradb
        rows.append((rho, meas, pred, abs(meas / pred - 1.0)))
        print(f"gate 3 [{label}]: rho/L={rho:.4f}  measured {meas:.6e}  predicted {pred:.6e}"
              f"  rel err {rows[-1][3]:.3e}")
    return x0, rows


def test_gradb_drift_textbook():
    """Uniform-direction B with |B| = B0(1 + a sin kx): the drift is the textbook
    v_gradB = (v_perp^2/(2*qm*|B|)) (B x grad|B|)/|B|^2, sign included."""
    params = _params(**_G3)
    _, rows = _g3_sweep(params, _field_bz(params), _analytic_bz, 1.0, "|B| gradient")
    with checks() as c:
        for rho, meas, pred, rel in rows:
            c.check(f"textbook grad-B drift to {rel:.2e} at rho/L = {rho:.4f}", rel < 0.02,
                    f"{meas} vs {pred}")
        c.check("drift is along +y, the sign of B x grad|B| for qm > 0",
                all(meas > 0 and pred > 0 for _, meas, pred, _ in rows), f"rows={rows}")


def test_gradb_drift_rmhd_bperp():
    """b_perp = zhat x grad(psi) on top of B0*zhat: the drift is 2*v_gradB (the field
    shears at O(eps) but |B| only at O(eps^2) — see the derivation above), converging in
    rho/L, with the bilinear interpolation error subdominant."""
    params = _params(**_G3)
    B3 = _field_bperp(params)
    x0, rows = _g3_sweep(params, B3, _analytic_bperp, 2.0, "RMHD b_perp")
    rels = [r[3] for r in rows]

    # interpolation error at the particle: bilinear vs exact spectral evaluation
    B3k = jnp.fft.rfft2(B3, axes=(-2, -1))
    rng = np.random.default_rng(0)
    probe = jnp.asarray(np.stack([x0 + rows[0][0] * rng.uniform(-1.0, 1.0, 64),
                                  rng.uniform(0.0, params.Ly, 64), np.zeros(64)], axis=1),
                        dtype=jnp.float64)
    b_bilin = np.asarray(interp.gather(B3, probe, params))
    b_spec = np.asarray(interp.gather_spectral(B3k, probe, params))
    interp_rel = float(np.max(np.abs(b_bilin[:, 1] - b_spec[:, 1]))
                       / np.max(np.abs(b_spec[:, 1])))
    print(f"gate 3 [RMHD b_perp]: bilinear vs spectral b_y over the orbit, max rel diff "
          f"{interp_rel:.3e}")

    with checks() as c:
        c.check(f"sheared-field drift within 5% of 2*v_gradB at rho/L = {rows[2][0]:.4f} "
                f"(rel err {rels[2]:.3e})", rels[2] < 0.05, f"rows={rows}")
        c.check("drift is along +y, the sign of B x grad|B| for qm > 0",
                all(r[1] > 0 and r[2] > 0 for r in rows), f"rows={rows}")
        c.check(f"error falls monotonically with rho/L {['%.2e' % r for r in rels]}",
                rels[0] > rels[1] > rels[2], f"rels={rels}")
        c.check(f"finite-Larmor-radius error is second order in rho/L: ratios "
                f"{['%.2f' % (a / b) for a, b in zip(rels, rels[1:])]} (~4)",
                all(2.5 < a / b < 6.0 for a, b in zip(rels, rels[1:])), f"rels={rels}")
        c.check(f"bilinear interpolation of b_y is subdominant ({interp_rel:.2e} << "
                f"{rels[2]:.2e})", interp_rel < 0.2 * rels[2], f"interp_rel={interp_rel}")


# ------------------------------------------------ gate 10: the mirror force and mu
#
# Gates 1-3 all hold v_par fixed by construction (bhat.grad|B| = 0), so nothing above
# tests the PARALLEL dynamics. That gap is 3D-only. In a static z-INDEPENDENT field with
# E = 0 the parallel motion is not an independent channel at all: |v| is exactly conserved
# (magnetic forces do no work, and the Boris rotation is norm-exact), and z is ignorable so
# p_z = v_z - qm*psi is exactly conserved too (docs/numerics.md), which gives
# v_z(t) = v_z(0) + qm*(psi(x(t)) - psi(x(0))) and v_perp^2 = |v|^2 - v_z^2. The split
# between parallel and perpendicular energy is then a function of the particle's
# PERPENDICULAR position alone, and both invariants are already pinned -- by gates 1 and 5
# (|v|) and gate 4c (p_z). Only when the field varies along z does v_par acquire dynamics
# of its own, the mirror force -mu*grad_par|B|. The last check below measures exactly that
# statement: in the z-independent version of this field v_par is constant to 1e-3.
#
# The field is B = B0*zhat + zhat x grad(psi) with psi = (b/kx)*sin(kx*x)*cos(kz*z):
#
#     B = (0, b*cos(kx*x)*cos(kz*z), B0)
#
# which is exactly divergence-free for ANY psi (zhat x grad(psi) has no z component and its
# x,y divergences cancel identically; here b_y has no y dependence, so div B = 0 term by
# term), periodic on the grid, and the RMHD field structure of plans/TESTPART_PLAN.md §2 --
# so the gate drives B1's trilinear 3D gather the way a real 3D run does. Field lines run
# at fixed x, so |B| = sqrt(B0^2 + b_y^2) varies ALONG a line through cos^2(kz*z): a
# magnetic mirror with throats at z = 0, Lz/2 and a well at z = Lz/4, of ratio
# R = sqrt(1 + (b/B0)^2). b/B0 = 0.5 (R = 1.1180) is DELIBERATELY outside the RMHD ordering
# -- it buys a usable mirror, and this is a kernel gate on an analytic field, not an RMHD
# state.
#
# Particles are launched in the well at z = Lz/4, where cos(kz*z) = 0 and the field is
# exactly B0*zhat: v_par0 = v_z(0), v_perp0 = |v_xy(0)| and |B|_0 = B0, with no gyrophase
# ambiguity. Conservation of mu = v_perp^2/(2|B|) and of |v| then puts the reflection at
#
#     |B|_turn = |B|_0 * |v|^2 / v_perp0^2
#
# which is what the v_par0 sweep measures. mu is NOT an exact invariant, so its excursion is
# asserted by its SCALING, not by a fixed small number: what these runs show is the
# reversible finite-Larmor term, first order in rho*k_z (the field at the particle differs
# from the field at the guiding centre at O(rho*grad), and the invariant itself carries an
# O(rho) gyrophase-dependent correction). The measured excursion is 0.19*rho*k_z at every
# point of an 8x sweep. The gyrophase-AVERAGED net change after half a bounce -- back at
# the launch plane, where |B| = B0 again for every x, so v_par there is v_z with no
# gyro-ripple -- falls far faster, which is what makes "reversible" a measurement rather
# than a hope.
_G10 = dict(dims=3, nx=64, ny=8, nz=64, Lx=2.0 * math.pi, Ly=2.0 * math.pi,
            Lz=2.0 * math.pi)
_G10_B0 = 1.0
_G10_B = 0.5              # b_perp amplitude; mirror ratio R = hypot(1, b/B0) = 1.1180
_G10_R = math.hypot(1.0, _G10_B / _G10_B0)
_G10_VPERP = 1.0
_G10_SPG = 50             # steps per gyration
_G10_T = 16.0             # launch -> turning point -> back through the launch plane
_G10_QMS = (10.0, 20.0, 40.0, 80.0)     # rho*kz = 0.1, 0.05, 0.025, 0.0125
_G10_VPARS = (0.10, 0.20, 0.30)         # trapped: v_par0/v_perp0 < sqrt(R-1) = 0.3436
_G10_LOSS = 0.5                         # inside the loss cone, so it must pass through


def _g10_params(n=_G10["nx"]):
    p, _ = ctx(**{**_G10, "nx": n, "nz": n})
    return p


def _g10_field(params, zdep=True):
    # B = (0, b*cos(kx x)*cos(kz z), B0), or the z-INDEPENDENT field of the same amplitude
    kx, kz = 2.0 * math.pi / params.Lx, 2.0 * math.pi / params.Lz
    x, _y, z = interp.grid_coords(params)
    zfac = jnp.cos(kz * z.reshape(-1, 1, 1)) if zdep else jnp.ones((params.nz, 1, 1))
    by = jnp.broadcast_to(_G10_B * jnp.cos(kx * x.reshape(1, -1, 1)) * zfac,
                          (params.nz, params.nx, params.ny))
    zero = jnp.zeros_like(by)
    return jnp.stack([zero, by, zero + _G10_B0]).astype(_precision.ftype)


@partial(jax.jit, static_argnums=(5, 6))
def _g10_orbit(x, v, B3, qm, dt, params, nsteps):
    # per step: v_par and |B| at the particle, mu about the LOCAL B (state.moments'
    # convention), and z. E = 0, so |v| is conserved exactly and mu's only error is
    # adiabatic.
    E3 = jnp.zeros_like(B3)

    def step(carry, _):
        x, v = boris.push(carry[0], carry[1], E3, B3, qm, dt, params)
        B = interp.gather(B3, x, params)
        bmag = jnp.sqrt(jnp.sum(B * B, axis=1))
        vpar = jnp.sum(v * B, axis=1) / bmag
        vsq = jnp.sum(v * v, axis=1)
        return (x, v), (vpar, bmag, (vsq - vpar ** 2) / (2.0 * bmag), x[:, 2], vsq)
    _, ys = jax.lax.scan(step, (x, v), None, length=nsteps)
    return ys


def _g10_cross(series, down=False):
    """(index, fraction) of the first zero crossing of `series` -- any crossing, or only a
    descending one -- with the fraction linearly interpolating between samples i and i+1.
    None if it never crosses."""
    above = series > 0.0
    hit = np.nonzero(above[:-1] & ~above[1:] if down else above[:-1] != above[1:])[0]
    if hit.size == 0:
        return None
    i = hit[0]
    return i, series[i] / (series[i] - series[i + 1])


@lru_cache(maxsize=None)
def _g10_bounce(qm, vpar0, spg=_G10_SPG, zdep=True, n=_G10["nx"]):
    """Four particles at quadrature gyrophases launched in the well at z = Lz/4. Host
    arrays: |B| at the first reflection (nan if the particle never reflects), the largest
    excursion of mu, the change in mu on returning to the launch plane, and the |v|^2
    drift."""
    params = _g10_params(n)
    B3 = _g10_field(params, zdep)
    dt = 2.0 * math.pi / (qm * _G10_B0 * spg)
    nsteps = int(_G10_T / dt)
    z0 = 0.25 * params.Lz
    x0 = jnp.asarray(np.tile([0.0, 0.0, z0], (len(_QUAD), 1)), dtype=jnp.float64)
    v0 = jnp.asarray([[_G10_VPERP * math.cos(p), _G10_VPERP * math.sin(p), vpar0]
                      for p in _QUAD], dtype=jnp.float64)
    vpar, bmag, mu, z, vsq = [np.asarray(a) for a in
                              _g10_orbit(x0, v0, B3, qm, dt, params, nsteps)]
    b_launch = np.asarray(interp.gather(B3, x0, params))
    bmag0 = np.linalg.norm(b_launch, axis=1)
    mu0 = _G10_VPERP ** 2 / (2.0 * bmag0)
    bturn, dmu_return = [], []
    for p in range(len(_QUAD)):
        turn = _g10_cross(vpar[:, p] * np.sign(vpar[0, p]))
        bturn.append(np.nan if turn is None
                     else bmag[turn[0], p] * (1.0 - turn[1]) + bmag[turn[0] + 1, p] * turn[1])
        # back DOWN through the launch plane after the reflection (the particle starts
        # there moving up, so the first descending crossing is the return): |B| = B0 there
        # for every x, so v_par is v_z with no gyro-ripple and mu is
        # (|v|^2 - v_par^2)/(2*B0), i.e. dmu/mu is (v_par0^2 - v_par^2)/v_perp0^2
        back = _g10_cross(z[:, p] - z0, down=True)
        if back is None:
            dmu_return.append(np.nan)
        else:
            i, f = back
            vz = vpar[i, p] * (1.0 - f) + vpar[i + 1, p] * f
            dmu_return.append((vpar0 ** 2 - vz ** 2) / _G10_VPERP ** 2)
    return {"bturn": np.array(bturn), "bmag0": bmag0,
            "excursion": float(np.max(np.abs(mu / mu0[None, :] - 1.0))),
            "dmu_return": np.array(dmu_return),
            "vsq_drift": float(np.max(np.abs(vsq / np.sum(np.asarray(v0) ** 2, axis=1) - 1.0))),
            "vpar": vpar, "z": z, "nsteps": nsteps,
            "rho_k": _G10_VPERP / (qm * _G10_B0) * (2.0 * math.pi / params.Lz),
            "rho_dx": _G10_VPERP / (qm * _G10_B0) / params.dx}


def _g10_predicted(res, vpar0):
    # mu and |v| conservation put the reflection at |B|_0*|v|^2/v_perp0^2
    return res["bmag0"] * (_G10_VPERP ** 2 + vpar0 ** 2) / _G10_VPERP ** 2


def test_mirror_turning_point_matches_mu_conservation():
    """Gate 10a: particles launched in the well reflect, and they reflect where mu and |v|
    conservation say they must -- over a sweep of v_par0 that moves |B|_turn across a ninth
    of the way up the mirror to nearly all of it. |v| is exact throughout (E = 0), which is
    the control that the 3D field and the trilinear gather are wired sanely: a mirror force
    with the wrong sign or a missing term would put the turning point somewhere else or
    stop the particle reflecting at all."""
    qm = 20.0
    rows = []
    with checks() as c:
        for vpar0 in _G10_VPARS:
            res = _g10_bounce(qm, vpar0)
            pred = _g10_predicted(res, vpar0)
            rel = res["bturn"] / pred - 1.0
            rows.append((vpar0, res, pred, rel))
            print(f"gate 10a: v_par0 = {vpar0} (rho/dx = {res['rho_dx']:.2f}, "
                  f"rho*kz = {res['rho_k']:.3f}): |B|_turn predicted {pred[0]:.6f} of "
                  f"|B|_max = {_G10_R:.4f}, measured "
                  f"{['%.6f' % b for b in res['bturn']]}, rel "
                  f"{['%+.2e' % r for r in rel]}, quadrature mean {np.mean(rel):+.2e}")
            c.check(f"every particle reflects at v_par0 = {vpar0}",
                    not np.any(np.isnan(res["bturn"])), f"bturn={res['bturn']}")
            c.check(f"|B|_turn matches |B|_0*|v|^2/v_perp0^2 to {np.max(np.abs(rel)):.2e} "
                    f"per particle at v_par0 = {vpar0} (< 1e-2; the spread is the O(rho) "
                    f"gyrophase term)", np.max(np.abs(rel)) < 1e-2, f"rel={rel}")
            c.check(f"... and to {abs(np.mean(rel)):.2e} averaged over the four quadrature "
                    f"gyrophases (< 2e-3)", abs(np.mean(rel)) < 2e-3,
                    f"mean={np.mean(rel)}")
            c.check(f"|v|^2 is conserved to {res['vsq_drift']:.2e} over "
                    f"{res['nsteps']} steps (E = 0, <= 1e-12)",
                    res["vsq_drift"] <= 1e-12, f"drift={res['vsq_drift']}")
        # the sweep really moves the prediction, so this is the mu law and not one number
        preds = [float(p[0]) for _v, _r, p, _rel in rows]
        c.check(f"the sweep moves |B|_turn - |B|_0 by 9x "
                f"({['%.4f' % (p - 1.0) for p in preds]}), so the three checks above pin "
                f"the mu law rather than a single reflection",
                preds[-1] - 1.0 > 8.0 * (preds[0] - 1.0), f"preds={preds}")
        # the same measurement refined in time and in space: the deviations above are the
        # physical finite-Larmor error, not the pusher's and not the gather's
        base = _g10_bounce(qm, _G10_VPARS[-1])
        worst = float(np.max(np.abs(rows[-1][3])))
        for label, ref in (("2x the steps per gyration",
                            _g10_bounce(qm, _G10_VPARS[-1], spg=2 * _G10_SPG)),
                           ("2x the grid",
                            _g10_bounce(qm, _G10_VPARS[-1], n=2 * _G10["nx"]))):
            moved = float(np.max(np.abs(ref["bturn"] / base["bturn"] - 1.0)))
            moved_mu = abs(ref["excursion"] / base["excursion"] - 1.0)
            print(f"gate 10a: {label} moves |B|_turn by {moved:.2e} ({moved / worst:.1%} "
                  f"of its deviation from the mu prediction) and the mu excursion by "
                  f"{moved_mu:.1%} ({base['excursion']:.5e} -> {ref['excursion']:.5e})")
            c.check(f"{label} moves |B|_turn by only {moved / worst:.0%} of its "
                    f"{worst:.1e} deviation from the mu prediction (< 25%), so that "
                    f"deviation is the finite-Larmor error and not the pusher's or the "
                    f"gather's", moved < 0.25 * worst, f"moved={moved}")
            c.check(f"... and moves the mu excursion gate 10b fits by {moved_mu:.2%} "
                    f"(< 2%)", moved_mu < 0.02, f"moved_mu={moved_mu}")


def test_mu_excursion_is_first_order_in_the_gyroradius():
    """Gate 10b: mu is conserved adiabatically, not exactly, so what is asserted is the
    SCALING of its violation. Over an 8x sweep of rho*k_z (through qm, i.e. Omega against
    the bounce frequency) the largest excursion of mu falls at first order and its ratio to
    rho*k_z is flat -- that excursion is the reversible finite-Larmor term, not a secular
    drift. The gyrophase-averaged net change on returning to the launch plane, which the
    reversible term does not survive, is orders of magnitude smaller and falls faster; it is
    printed at every point and asserted only where it is unambiguous."""
    vpar0 = _G10_VPARS[-1]
    res = [_g10_bounce(qm, vpar0) for qm in _G10_QMS]
    rho_k = [r["rho_k"] for r in res]
    exc = [r["excursion"] for r in res]
    turn = [float(np.max(np.abs(r["bturn"] / _g10_predicted(r, vpar0) - 1.0))) for r in res]
    ret = [abs(float(np.mean(r["dmu_return"]))) for r in res]
    o_exc, o_turn = fit_order(rho_k, exc), fit_order(rho_k, turn)
    for r, e, t, q in zip(res, exc, turn, ret):
        print(f"gate 10b: rho*kz = {r['rho_k']:.4f} (rho/dx = {r['rho_dx']:.2f}): "
              f"max|dmu/mu| = {e:.4e} = {e / r['rho_k']:.3f}*rho*kz; max turning-point "
              f"error {t:.3e}; |mean dmu/mu| back at the launch plane {q:.2e} "
              f"({e / q:.0f}x smaller than the excursion)")
    print(f"gate 10b: orders in rho*kz -- mu excursion {o_exc:.3f}, turning point "
          f"{o_turn:.3f}")
    ratios = [e / r["rho_k"] for e, r in zip(exc, res)]
    with checks() as c:
        c.check(f"the mu excursion falls at order {o_exc:.3f} in rho*kz over an 8x sweep "
                f"(0.85 < order < 1.25: the reversible finite-Larmor term is first order)",
                0.85 < o_exc < 1.25, f"exc={['%.3e' % e for e in exc]}")
        c.check(f"... and its ratio to rho*kz is flat, {['%.3f' % r for r in ratios]} "
                f"(within 15% of the mean), which is the same statement without a fit",
                max(ratios) / min(ratios) < 1.15, f"ratios={ratios}")
        c.check(f"the turning point converges to the mu prediction at order {o_turn:.3f} "
                f"in rho*kz too (0.8 < order < 1.3)", 0.8 < o_turn < 1.3,
                f"turn={['%.3e' % t for t in turn]}")
        for i in (-2, -1):
            c.check(f"at rho*kz = {rho_k[i]:.4f} the excursion is reversible: the "
                    f"gyrophase-averaged mu change back at the launch plane, {ret[i]:.2e}, "
                    f"is {exc[i] / ret[i]:.0f}x below the excursion en route (>= 10x)",
                    exc[i] > 10.0 * ret[i], f"exc={exc[i]} return={ret[i]}")


def test_loss_cone_particles_pass_and_a_z_independent_field_traps_nothing():
    """Gate 10c, two negative controls for gate 10a. (i) A particle with too much v_par0 --
    |v|^2/v_perp0^2 above the mirror ratio -- must cross the throat instead of reflecting.
    (ii) The SAME field with its z dependence removed has |B| constant along a field line,
    so there is no mirror force at all and v_par is constant: the parallel channel gate 10
    tests exists only because the field varies along z, which is why it has no 2D
    counterpart."""
    qm = 20.0
    trapped = _g10_bounce(qm, _G10_VPARS[-1])
    loss = _g10_bounce(qm, _G10_LOSS)
    flat = _g10_bounce(qm, _G10_VPARS[-1], zdep=False)
    z0 = 0.25 * _g10_params().Lz
    dz_trap = float(np.max(np.abs(trapped["z"] - z0)))
    dz_loss = float(np.min(np.abs(loss["z"][-1] - z0)))
    dz_flat = float(np.min(np.abs(flat["z"][-1] - z0)))
    spread = [float(np.ptp(flat["vpar"][:, p])) for p in range(len(_QUAD))]
    print(f"gate 10c: loss cone needs v_par0/v_perp0 < {math.sqrt(_G10_R - 1.0):.4f}; "
          f"at {_G10_LOSS} the particles reach |z - z0| = {dz_loss:.2f} without reflecting "
          f"(trapped ones stay inside {dz_trap:.2f})")
    print(f"gate 10c: z-independent field -- v_par ranges "
          f"{['%.4f..%.4f' % (flat['vpar'][:, p].min(), flat['vpar'][:, p].max()) for p in range(len(_QUAD))]}"
          f", |z - z0| reaches {dz_flat:.2f}")
    with checks() as c:
        c.check(f"a particle inside the loss cone never reflects (no v_par sign change) "
                f"and crosses the throat, |z - z0| = {dz_loss:.2f} > Lz/4",
                not np.any(np.isnan(trapped["bturn"])) and np.all(np.isnan(loss["bturn"]))
                and dz_loss > z0, f"bturn={loss['bturn']} dz={dz_loss}")
        c.check(f"with the z dependence removed, |B| is constant along a field line and "
                f"v_par is constant to {max(spread):.1e} of v_perp (< 1e-2): no mirror "
                f"force, no parallel dynamics", max(spread) < 1e-2 * _G10_VPERP,
                f"spread={spread}")
        c.check(f"... so nothing is trapped there either: every particle streams to "
                f"|z - z0| >= {dz_flat:.2f}, outside the {dz_trap:.2f} the mirrored ones "
                f"never leave", dz_flat > 1.5 * dz_trap and np.all(np.isnan(flat["bturn"])),
                f"flat={dz_flat} trapped={dz_trap} bturn={flat['bturn']}")


# --------------------------------------------- gate 11: the pusher under a varying dt
#
# The push is kick-drift-kick precisely so that dt may change from step to step: x and v
# are synchronized at step boundaries, unlike leapfrog. Production runs adaptive dt, so
# that property is exercised here on analytic fields (its coupled counterpart is gate 4
# live under adaptive_timestep in tests/test_particles_coupled.py). The sequence is
# jittered per step by +-60% about the mean and renormalized to a fixed total time, so
# refining it means more steps over the SAME elapsed time, not a shorter run.
_G11_T = 20.0 * 2.0 * math.pi / (_G1_QM * 1.0)     # 20 gyrations
_G11_NS = (400, 800, 1600, 3200)
_G11_JITTER = 0.6


def _g11_dts(n, jitter=_G11_JITTER, seed=11):
    # n step lengths summing exactly to _G11_T; returns them with the max/min ratio
    d = 1.0 + jitter * np.random.default_rng(seed).uniform(-1.0, 1.0, n)
    return jnp.asarray(_G11_T * d / d.sum()), float(d.max() / d.min())


@partial(jax.jit, static_argnums=(5,))
def _g11_orbit(x, v, E3, B3, qm, params, dts):
    def step(carry, dt):
        return boris.push(carry[0], carry[1], E3, B3, qm, dt, params), None
    (x, v), _ = jax.lax.scan(step, (x, v), dts)
    return x, v


def _g11_exact(x0, v0, t, omega):
    # uniform B0*zhat with E = 0: v rotates clockwise at omega (gate 1's sense)
    c, s = math.cos(omega * t), math.sin(omega * t)
    v = np.array([v0[0] * c + v0[1] * s, -v0[0] * s + v0[1] * c, v0[2]])
    x = np.array([x0[0] + (v0[0] * s - v0[1] * (c - 1.0)) / omega,
                  x0[1] + (v0[0] * (c - 1.0) + v0[1] * s) / omega,
                  x0[2] + v0[2] * t])
    return x, v


def test_varying_dt_conserves_v_and_converges_at_second_order():
    """Gate 11a: with dt jittered by 4x from step to step, |v| is still conserved to
    round-off in a uniform field with E = 0 (the rotation is norm-exact whatever dt is --
    if this failed, the KDK driver would be mis-splitting the half-kicks), and the orbit
    still converges to the analytic solution at second order in max(dt)."""
    params = _params(**_G1)
    B3, E3 = _uniform(params, (0.0, 0.0, 1.0)), _uniform(params, (0.0, 0.0, 0.0))
    omega = _G1_QM * 1.0
    x0 = np.zeros(3)
    v0 = np.array([_G1_V, 0.0, 0.1 * _G1_V])
    xe, ve = _g11_exact(x0, v0, _G11_T, omega)
    hs, errs, norms, ratios = [], [], [], []
    for n in _G11_NS:
        dts, ratio = _g11_dts(n)
        x, v = _g11_orbit(jnp.asarray(x0[None]), jnp.asarray(v0[None]), E3, B3, _G1_QM,
                          params, dts)
        x, v = np.asarray(x)[0], np.asarray(v)[0]
        hs.append(float(jnp.max(dts)))
        errs.append(float(np.max(np.abs(np.concatenate([x - xe, v - ve])))))
        norms.append(abs(float(np.hypot(v[0], v[1])) / _G1_V - 1.0))
        ratios.append(ratio)
        print(f"gate 11a: {n} steps, max dt {hs[-1]:.4f} (Omega*dt {omega * hs[-1]:.3f}), "
              f"dt max/min {ratio:.2f}: |v_perp| drift {norms[-1]:.2e}, max orbit error "
              f"{errs[-1]:.3e}")
    order = fit_order(hs, errs)
    print(f"gate 11a: orbit error order in max(dt) = {order:.3f}")
    with checks() as c:
        c.check(f"the dt sequence really varies (max/min {min(ratios):.2f} >= 3)",
                min(ratios) >= 3.0, f"ratios={ratios}")
        c.check(f"|v_perp| is conserved to {max(norms):.2e} under a varying dt (<= 1e-14)",
                max(norms) <= 1e-14, f"norms={norms}")
        c.check(f"the orbit converges to the analytic solution at order {order:.3f} in "
                f"max(dt) (1.8 < order < 2.2)", 1.8 < order < 2.2,
                f"errs={['%.3e' % e for e in errs]}")


def test_varying_dt_preserves_the_exb_drift():
    """Gate 11b: the E x B drift is still exactly E x B/B^2 under a varying dt. For uniform
    fields the Boris map is affine in v and fixes v = u, so the quadrature-gyrophase average
    advances at exactly u per unit time however the step lengths are chosen -- which makes
    this a round-off check rather than gate 2's 1e-3 one, and any dt-dependent bias in the
    kick/drift splitting would break it."""
    params = _params(**_G1)
    B3 = _uniform(params, (0.0, 0.0, 1.0))
    dts, ratio = _g11_dts(1000)
    with checks() as c:
        for Evec in ((0.0, 0.25, 0.0), (0.25, 0.0, 0.0)):
            u = np.array([Evec[1], -Evec[0], 0.0])       # E x B/B^2 at B = zhat
            x0 = jnp.zeros((4, 3), dtype=jnp.float64)
            v0 = jnp.asarray([[u[0] + 0.5 * math.cos(p), u[1] - 0.5 * math.sin(p), 0.0]
                              for p in _QUAD], dtype=jnp.float64)
            x, _v = _g11_orbit(x0, v0, _uniform(params, Evec), B3, _G1_QM, params, dts)
            meas = np.asarray(jnp.mean(x, axis=0)) / _G11_T
            rel = float(np.max(np.abs(meas - u))) / 0.25
            print(f"gate 11b: E = {Evec} over {len(dts)} jittered steps (max/min "
                  f"{ratio:.2f}): drift {meas} vs {u}, rel {rel:.2e}")
            c.check(f"E = {Evec} drifts at E x B/B^2 under a varying dt to {rel:.1e} "
                    f"(<= 1e-12)", rel <= 1e-12, f"{meas} vs {u}")


# --------------------------------------------------------------- interpolation checks

_GI = dict(nx=32, ny=32, Lx=2.0 * math.pi, Ly=2.0 * math.pi)


def _analytic(x, y):
    return jnp.cos(3.0 * x) * jnp.sin(2.0 * y) + 0.5 * jnp.sin(x + 4.0 * y)


def _analytic_field(params):
    x, y, _z = interp.grid_coords(params)
    return _analytic(x.reshape(1, -1, 1), y.reshape(1, 1, -1)).astype(_precision.ftype)


def test_gather_exact_at_nodes_and_on_ramps():
    params = _params(**_GI)
    f = _analytic_field(params)
    x, y, _z = interp.grid_coords(params)
    ii, jj = np.meshgrid(np.arange(params.nx), np.arange(params.ny), indexing="ij")
    pos = jnp.stack([x[ii.ravel()], y[jj.ravel()], jnp.zeros(ii.size)], axis=1)
    node_err = float(jnp.max(jnp.abs(interp.gather(f[None], pos, params)[:, 0]
                                     - f[0].reshape(-1).astype(jnp.float64))))

    # a linear ramp is reproduced exactly away from the wrap cell
    ramp = jnp.broadcast_to(x.reshape(1, -1, 1), (1, params.nx, params.ny)).astype(_precision.ftype)
    rng = np.random.default_rng(1)
    xr = rng.uniform(0.0, params.Lx - params.dx, 200)
    pr = jnp.asarray(np.stack([xr, rng.uniform(0.0, params.Ly, 200), np.zeros(200)], axis=1))
    ramp_err = float(jnp.max(jnp.abs(interp.gather(ramp[None], pr, params)[:, 0] - xr)))

    # the last cell blends column nx-1 with column 0
    pw = jnp.asarray([[params.Lx - 0.5 * params.dx, 0.0, 0.0]], dtype=jnp.float64)
    wrapped = float(interp.gather(f[None], pw, params)[0, 0])
    expect = 0.5 * (float(f[0, -1, 0]) + float(f[0, 0, 0]))
    print(f"interp: node err {node_err:.2e}, ramp err {ramp_err:.2e}, "
          f"wrap {wrapped:.12f} vs {expect:.12f}")
    with checks() as c:
        # not bitwise: x_i/dx is only integral to within a ulp, so a node can land a ulp
        # inside the neighbouring cell and blend
        c.check(f"gather reproduces the grid values at collocation points ({node_err:.2e})",
                node_err < 1e-13, f"err={node_err}")
        c.check(f"gather of a linear ramp is exact between nodes ({ramp_err:.2e})",
                ramp_err < 50.0 * _FEPS * params.Lx, f"err={ramp_err}")
        c.check("gather at Lx - dx/2 blends the last and first columns",
                abs(wrapped - expect) < 1e-15, f"{wrapped} vs {expect}")


def test_gather_spectral_exact_and_bilinear_order():
    params = _params(**_GI)
    f = _analytic_field(params)
    fk = jnp.fft.rfft2(f, axes=(-2, -1))
    rng = np.random.default_rng(2)
    npos = 64
    px = rng.uniform(0.0, params.Lx, npos)
    py = rng.uniform(0.0, params.Ly, npos)
    pos = jnp.asarray(np.stack([px, py, np.zeros(npos)], axis=1))
    exact = np.asarray(_analytic(jnp.asarray(px), jnp.asarray(py)))
    spec = np.asarray(interp.gather_spectral(fk[None], pos, params)[:, 0])
    spec_err = float(np.max(np.abs(spec - exact)))

    # bilinear error against the same analytic function, refined
    dxs, errs = [], []
    for nx in (32, 64, 128):
        p = _params(**{**_GI, "nx": nx, "ny": nx})
        fp = _analytic_field(p)
        errs.append(float(np.max(np.abs(np.asarray(interp.gather(fp[None], pos, p)[:, 0])
                                        - exact))))
        dxs.append(p.dx)
    order = fit_order(dxs, errs)
    print(f"interp: spectral err {spec_err:.3e}; bilinear errs "
          f"{['%.3e' % e for e in errs]} -> order {order:.3f}")
    with checks() as c:
        c.check(f"gather_spectral reproduces a band-limited field to round-off "
                f"({spec_err:.2e})", spec_err < 2000.0 * _FEPS, f"err={spec_err}")
        c.check(f"bilinear gather converges at order {order:.3f} (~2)",
                1.8 < order < 2.3, f"errs={errs}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
