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
#
# Not precision-marked: particle state is fp64 at every TARANIS_PRECISION, and gates 1-2
# use exactly representable field constants, so they are bitwise identical in both
# sessions. Gate 3's analytic field arrays are stored at field precision and so differ at
# fp32 round-off; its tolerances are loose enough that it passes in both sessions. Only
# the interpolation checks take an explicitly precision-dependent tolerance.
#
# Gates 2 and 3 fire four particles from the same point at quadrature gyrophases. For
# uniform fields the Boris map is affine in v, so that average is EXACTLY the drift
# motion (the gyration cancels identically); in the weakly nonuniform field of gate 3 it
# suppresses the gyration to the 4th harmonic, and a least-squares slope over hundreds of
# gyroperiods removes the rest.
from _rmhd_testing import bootstrap

bootstrap()

import math
from functools import partial

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
    x, y = interp.grid_coords(params)
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
    x, _ = interp.grid_coords(params)
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


# --------------------------------------------------------------- interpolation checks

_GI = dict(nx=32, ny=32, Lx=2.0 * math.pi, Ly=2.0 * math.pi)


def _analytic(x, y):
    return jnp.cos(3.0 * x) * jnp.sin(2.0 * y) + 0.5 * jnp.sin(x + 4.0 * y)


def _analytic_field(params):
    x, y = interp.grid_coords(params)
    return _analytic(x.reshape(1, -1, 1), y.reshape(1, 1, -1)).astype(_precision.ftype)


def test_gather_exact_at_nodes_and_on_ramps():
    params = _params(**_GI)
    f = _analytic_field(params)
    x, y = interp.grid_coords(params)
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
