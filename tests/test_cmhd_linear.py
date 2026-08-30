# Compressible MHD (physics/cmhd.py, plans/CMHD_PLAN.md Phase C1), gates 1, 2 and 4:
#
#   1. DISPERSION. A single analytic eigenmode of the linearized system, built in REAL space
#      at amplitude eps on the (rho0 = 1, B0 zhat) background, evolved with nu = 0 at fixed
#      small dt. The measured oscillation frequency of the projected eigenamplitude is
#      compared with the analytic omega(k) of docs/numerics.md "Compressible MHD" -- Alfven,
#      fast and slow, at exactly parallel / oblique / exactly perpendicular propagation, at
#      gamma in {1, 5/3} and cs0/v_A in {0.5, 2}. Because there is no background/fluctuation
#      split in this equation set, this measures the PRODUCTION RHS: the waves come out of the
#      quadratic terms acting on the k = 0 modes, so a wrong sign anywhere is an O(1) error
#      here.
#      Branches are labelled by SPEED, never by name: at cs0 = 0.5 the parallel sound wave is
#      the slow branch and the transverse one is fast; at cs0 = 2 they swap. The test checks
#      both, which is what makes the theta = 0 polarization selection non-vacuous.
#
#   2. DISSIPATION-ONLY EXACT DECAY. Three states whose IDEAL RHS vanishes identically (a
#      force-free B, a Beltrami u, and a pure density perturbation at negligible cs0), so the
#      IF step reduces to exp(L*dt) and each field's mode must decay by exp(-diss_f*k^(2h)*dt)
#      per step to round-off. Run with a length-3 (D_rho, nu, eta) diss, so the three
#      coefficients are distinct and the 7-field expansion is gated too.
#
#   4. SCHEME CROSS-CHECKS. lsrk54 vs rk44 vs imexcb3e agree at the expected order (fp64
#      only -- at fp32 the differences sit at the storage noise floor and no order can be
#      measured honestly), and lsrk_scan / hoist_propagator do not change the answer.
#
# Single process by construction (CMHD is z_spectral, which is size==1 only). pytest, or
# `python tests/test_cmhd_linear.py` -- never under mpirun.
from _rmhd_testing import bootstrap, checks, fit_order, fresh_params

bootstrap()

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme, stage_exp_ops

# 8^3 is enough for every gate here: the modes used are |n| <= 2, well inside the 2/3 cut
# (|n| < 8/3), and the representation of a single Fourier mode is exact.
_N = 8
_BOX = dict(nx=_N, ny=_N, nz=_N, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi, dims=3,
            z_spectral=True, eqtype="CMHD", adaptive_timestep=False, cfl_safety=0.5)
_B0 = 1.0

# (kx, kz) of the tested modes: exactly parallel, oblique, exactly perpendicular. ky = 0
# throughout (the (k, B0) plane is (x, z), matching the docs' linearization).
_ANGLES = ((0.0, 1.0), (1.0, 2.0), (1.0, 0.0))


def _fp64():
    return _precision.precision == "64"


# Perturbation amplitude. At fp64 eps = 1e-6 keeps the O(eps^2) self-interaction ~1e-6
# relative, far under the tolerances. At fp32 the background rho = 1 quantizes the
# perturbation at eps32/eps relative, so eps = 1e-6 would leave only ~6% of the mode
# intact -- 1e-3 puts that quantization at 6e-5 instead, at the price of an O(eps) = 1e-3
# relative self-interaction (which is what sets the fp32 amplitude tolerance below).
_EPS = 1e-6 if _fp64() else 1e-3

# Tolerances. Frequency: the residual is the lsrk54 time error, O((omega*dt)^4) at
# omega*dt <= 0.05, measured <= 2.1e-8 (fp64) and <= 5.4e-5 (fp32, where the eps^2/eps32
# floor above dominates). Amplitude: nu = 0 and the IF propagator is the identity, so the
# only motion is the O(eps) self-interaction plus round-off; measured <= 6.1e-9 (fp64),
# <= 2.0e-3 (fp32, on the slow branch, whose eigenvector is du_z-dominated).
_W_TOL = 1e-6 if _fp64() else 2e-4
_AMP_TOL = 1e-6 if _fp64() else 1e-2
# omega = 0 branches (both degenerate limits at theta = pi/2): the projected amplitude must
# not move at all. Measured <= 1.0e-10 (fp64), <= 5.4e-5 (fp32).
_STATIC_TOL = 1e-6 if _fp64() else 5e-4

_CTX = {}


def _ctx(cs0, gamma, dt):
    # one Parameters (hence one jit trace) per physics config; dt is a plain python float,
    # never a numpy scalar -- a numpy float64 dt would make DiagonalOperator.scaled(dt)
    # strong-typed float64 and poison the fp32 field math
    key = (cs0, gamma, float(dt))
    if key not in _CTX:
        p = fresh_params(eqpars=dict(cs0=cs0, diss=0.0, hyper=1, gamma=gamma),
                         dt=float(dt), **_BOX)
        _CTX[key] = (p, jr.setup_kgrids(p))
    return _CTX[key]


# ------------------------------------------------------------- analytic linear theory
# Transcribed from docs/numerics.md "Compressible MHD" -> "Linear waves". Linearized about
# rho = 1, B = B0 zhat, k = (kx, 0, kz):
#     omega drho = k.du
#     omega du   = c^2 k drho - (k x dB) x B0
#     omega dB   = -kpar B0 du + (k.du) B0 zhat

def _omega_ms(kx, kz, cs, vA, fast):
    # omega^2 = 0.5 k^2 (c^2+vA^2) [1 +- sqrt(1 - 4 c^2 vA^2 cos^2(theta)/(c^2+vA^2)^2)]
    k2 = kx*kx + kz*kz
    s = cs*cs + vA*vA
    disc = max(1.0 - 4.0*cs*cs*vA*vA*(kz*kz/k2)/(s*s), 0.0)
    w2_fast = 0.5*k2*s*(1.0 + np.sqrt(disc))
    if fast:
        return np.sqrt(w2_fast)
    # slow root from the product of the quartic's roots, w2f*w2s = c^2 vA^2 k^2 kpar^2:
    # cancellation-free, and exactly 0 at kpar = 0 (the perpendicular degenerate limit)
    return np.sqrt(cs*cs*vA*vA*k2*kz*kz/w2_fast)


def _eigenmode(branch, kx, kz, cs, vA):
    """(omega, v) with v = (drho, du_x, du_y, du_z, dB_x, dB_y, dB_z), all real."""
    if branch == "alfven":
        # du along yhat (out of the (k, B0) plane), drho = 0, omega^2 = kpar^2 vA^2, and
        # dB_y = -(kpar*B0/omega)*du_y -- for the omega = +kpar*vA root that is exactly
        # -du_y, which is also the kz -> 0 limit used at exactly perpendicular (where
        # omega = 0 and the ratio itself is 0/0).
        return np.sqrt(kz*kz*vA*vA), np.array([0.0, 0, 1, 0, 0, -1.0, 0])
    w = _omega_ms(kx, kz, cs, vA, branch == "fast")
    if kx == 0.0:
        # theta = 0: the branches decouple, so use the polarizations DIRECTLY -- the
        # du_z/du_x ratio below is 0/0 here. Which of fast/slow is the sound wave depends on
        # c vs vA, so decide by which omega^2 this root matches (labels are speeds).
        if abs(w*w - cs*cs*kz*kz) < abs(w*w - vA*vA*kz*kz):
            # sound wave: du parallel to zhat, omega = +-k*c, dB = 0
            return w, np.array([kz/w, 0, 0, 1.0, 0, 0, 0])
        # transverse x-polarized wave at omega = +-k*vA (degenerate with the Alfven branch)
        return w, np.array([0.0, 1.0, 0, 0, -kz*_B0/w, 0, 0])
    if w == 0.0:
        # theta = pi/2, slow: omega -> 0. The eigenvector table degenerates (it divides by
        # omega), so take the omega = 0 solution of the linearized system directly: du = 0,
        # k.dB = 0 forces dB = dB_z zhat, and x-momentum gives total-pressure balance
        # c^2 drho + B0 dB_z = 0.
        return 0.0, np.array([-_B0/(cs*cs), 0, 0, 0, 0, 0, 1.0])
    # magnetosonic eigenvector, docs/numerics.md verbatim:
    #   du_z/du_x = c^2 kx kz/(omega^2 - c^2 kz^2)
    #   drho = (kx du_x + kz du_z)/omega
    #   dB_x = -kz B0 du_x/omega,  dB_y = 0,  dB_z = +kx B0 du_x/omega   (k.dB = 0)
    dux = 1.0
    duz = cs*cs*kx*kz/(w*w - cs*cs*kz*kz)*dux
    drho = (kx*dux + kz*duz)/w
    return w, np.array([drho, dux, 0.0, duz, -kz*_B0*dux/w, 0.0, kx*_B0*dux/w])


def _eigen_ic(v, kx, kz, eps):
    # real-space IC: the uniform background plus eps*v*cos(k.x). Built in real space, so the
    # rfftn reality constraint holds by construction (never write k-space directly).
    base = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, _B0)

    def ic(x, y, z):
        shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
        ph = jnp.broadcast_to(jnp.cos(kx*x + kz*z), shp)
        return jnp.stack([base[i] + eps*float(v[i])*ph for i in range(7)])
    return ic


def _project(fields, v, kx, kz, params):
    # complex amplitude of the eigenmode: the single k-space coefficient at (kz, kx, ky=0),
    # contracted with the (real) eigenvector. numpy's fftn stores the e^{+ik.x} amplitude at
    # index +n, so a mode ~cos(k.x - omega t) reads as A*exp(-i*omega*t) here.
    c = np.asarray(fields[:, int(kz) % params.nz, int(kx) % params.nx, 0])
    return complex(np.dot(v, c)/np.dot(v, v))


def _measure(branch, kx, kz, cs0, gamma, block=20, target_phase=6.0, wdt=0.05):
    """(omega_analytic, relative frequency error, relative amplitude variation)."""
    vA = _B0
    w, v = _eigenmode(branch, kx, kz, cs0, vA)
    # dt is fixed by the fastest wave of the angle set, so one Parameters serves every branch
    dt = wdt/max(_omega_ms(a, b, cs0, vA, True) for a, b in _ANGLES)
    params, kgrid = _ctx(cs0, gamma, dt)
    state = jr.initialize(_eigen_ic(v, kx, kz, _EPS), params)
    stepper, scheme = get_scheme("lsrk54")
    advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))
    ts = [float(state.t)]
    ps = [_project(state.fields, v, kx, kz, params)]
    # enough blocks for ~1 radian per sample (so np.unwrap is unambiguous) and
    # target_phase radians overall; the omega = 0 branches just run a fixed 200 steps
    nsamp = int(np.ceil(target_phase/(w*float(dt)*block))) if w > 0 else 10
    for _ in range(nsamp):
        state = advance(state, kgrid, params, block, scheme, stepper)
        ts.append(float(state.t))
        ps.append(_project(state.fields, v, kx, kz, params))
    ts, ps = np.array(ts), np.array(ps)
    amp = np.abs(ps)
    ampvar = float(np.max(np.abs(amp/amp[0] - 1.0)))
    if w == 0.0:
        return w, float(np.max(np.abs(ps - ps[0]))/abs(ps[0])), ampvar
    w_meas = -float(np.polyfit(ts, np.unwrap(np.angle(ps)), 1)[0])
    return w, abs(w_meas/w - 1.0), ampvar


# --------------------------------------------------------------------- gate 1: dispersion

def _dispersion_cell(c, branch, cs0, gamma):
    vA = _B0
    for kx, kz in _ANGLES:
        w, err, ampvar = _measure(branch, kx, kz, cs0, gamma)
        label = f"gamma={gamma:.3f} cs0={cs0} {branch} k=({kx:.0f},{kz:.0f})"
        if w == 0.0:
            c.check(f"{label}: degenerate omega = 0, the mode is static "
                    f"(drift {err:.2e})", err < _STATIC_TOL, f"drift {err:.3e}")
        else:
            c.check(f"{label}: omega = {w:.6f} measured to {err:.2e}", err < _W_TOL,
                    f"relative error {err:.3e} > {_W_TOL}")
        c.check(f"{label}: no spurious growth at nu = 0 ({ampvar:.2e})",
                ampvar < _AMP_TOL, f"amplitude varies by {ampvar:.3e}")


def test_dispersion_alfven():
    with checks() as c:
        for gamma in (1.0, 5.0/3.0):
            for cs0 in (0.5, 2.0):
                _dispersion_cell(c, "alfven", cs0, gamma)


def test_dispersion_fast():
    with checks() as c:
        for gamma in (1.0, 5.0/3.0):
            for cs0 in (0.5, 2.0):
                _dispersion_cell(c, "fast", cs0, gamma)


def test_dispersion_slow():
    with checks() as c:
        for gamma in (1.0, 5.0/3.0):
            for cs0 in (0.5, 2.0):
                _dispersion_cell(c, "slow", cs0, gamma)


def test_degenerate_limits_of_the_dispersion_relation():
    """The analytic limits the gates above rely on, asserted on the formulas themselves:
    perpendicular slow and perpendicular Alfven both go to zero frequency, perpendicular fast
    is the magnetosonic k*sqrt(c^2+vA^2), and at theta = 0 the two magnetosonic roots are
    exactly {k*c, k*vA} -- which of them is 'fast' swaps with cs0/vA."""
    vA = _B0
    with checks() as c:
        for cs0 in (0.5, 2.0):
            k = 1.0
            c.check(f"cs0={cs0}: perpendicular slow omega is exactly 0",
                    _omega_ms(k, 0.0, cs0, vA, False) == 0.0)
            c.check(f"cs0={cs0}: perpendicular Alfven omega is exactly 0",
                    _eigenmode("alfven", k, 0.0, cs0, vA)[0] == 0.0)
            wf = _omega_ms(k, 0.0, cs0, vA, True)
            c.check(f"cs0={cs0}: perpendicular fast is k*sqrt(c^2+vA^2)",
                    abs(wf - k*np.sqrt(cs0**2 + vA**2)) < 1e-14, f"{wf}")
            par = sorted([_omega_ms(0.0, k, cs0, vA, False),
                          _omega_ms(0.0, k, cs0, vA, True)])
            c.check(f"cs0={cs0}: parallel roots are {{k*c, k*vA}}",
                    abs(par[0] - k*min(cs0, vA)) < 1e-14
                    and abs(par[1] - k*max(cs0, vA)) < 1e-14, f"{par}")
            sound_is_fast = abs(_omega_ms(0.0, k, cs0, vA, True) - k*cs0) < 1e-14
            c.check(f"cs0={cs0}: the parallel sound wave is the "
                    f"{'fast' if sound_is_fast else 'slow'} branch",
                    sound_is_fast == (cs0 > vA))


# ------------------------------------------------------- gate 2: dissipation-only decay
# Three ideal-RHS-free states. Each is an exact steady state of the ideal equations, so the
# IF step is exactly exp(L*dt) and the decay is the propagator's alone:
#   B   : rho = 1, u = 0, B = (cos kz, sin kz, 0) -- curl B = -k B, so j x B cancels
#         pointwise (a x a = 0 bitwise in fp), and u x B = 0 exactly.
#   u   : rho = 1, B = 0, u = (cos kz, sin kz, 0) -- omega x u cancels the same way,
#         |u|^2 is uniform, and u_z = 0 makes div(rho u) vanish.
#   rho : u = B = 0 and a density perturbation, at cs0 = 1e-8 so the pressure force it
#         drives is O(cs0^2) = 1e-16 and the sound response stays under the tolerance.
_DECAY_DISS = (0.05, 0.1, 0.2)      # (D_rho, nu, eta): three distinct coefficients
_DECAY_KZ = 2.0
# fp64: measured <= 2.6e-15 relative over 40 steps (the exp(L*gamma_i*dt) stage composition).
# fp32: measured <= 3.4e-6, i.e. ~50 ulps of the same composition.
_DECAY_TOL = 1e-13 if _fp64() else 2e-5
# leak into the fields the configuration did not excite, relative to the excited amplitude
_LEAK_TOL = 1e-13 if _fp64() else 1e-6


def _decay_ic(kind):
    def ic(x, y, z):
        shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
        one = jnp.ones(shp)
        zero = jnp.zeros(shp)
        cs = jnp.broadcast_to(jnp.cos(_DECAY_KZ*z), shp)
        sn = jnp.broadcast_to(jnp.sin(_DECAY_KZ*z), shp)
        if kind == "B":
            return jnp.stack([one, zero, zero, zero, cs, sn, zero])
        if kind == "u":
            return jnp.stack([one, cs, sn, zero, zero, zero, zero])
        return jnp.stack([one + 0.1*cs, zero, zero, zero, zero, zero, zero])
    return ic


def _decay_case(c, hyper, nsteps=40, dt=0.02):
    for kind, comps, diss, cs0 in (("B", (4, 5), _DECAY_DISS[2], 1.0),
                                   ("u", (1, 2), _DECAY_DISS[1], 1.0),
                                   ("rho", (0,), _DECAY_DISS[0], 1e-8)):
        params = fresh_params(eqpars=dict(cs0=cs0, diss=_DECAY_DISS, hyper=hyper,
                                          gamma=1.0), dt=dt, **_BOX)
        kgrid = jr.setup_kgrids(params)
        state = jr.initialize(_decay_ic(kind), params)
        stepper, scheme = get_scheme("lsrk54")
        advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))
        iz = int(_DECAY_KZ)
        c0 = np.asarray(state.fields[:, iz, 0, 0])
        end = advance(state, kgrid, params, nsteps, scheme, stepper)
        c1 = np.asarray(end.fields[:, iz, 0, 0])
        want = np.exp(-diss*(_DECAY_KZ**2)**hyper*nsteps*dt)
        err = max(abs(c1[i]/c0[i] - want)/want for i in comps)
        leak = (max(abs(c1[i]) for i in range(7) if i not in comps)
                / max(abs(c0[i]) for i in comps))
        c.check(f"hyper={hyper} {kind}: decays by exp(-diss*k^{2*hyper}*t) = {want:.4e} "
                f"(relative error {err:.2e})", err < _DECAY_TOL, f"{err:.3e}")
        c.check(f"hyper={hyper} {kind}: the other fields stay at zero ({leak:.2e})",
                leak < _LEAK_TOL, f"{leak:.3e}")


def test_dissipation_only_decay_is_exact():
    """Each field group decays at its OWN coefficient: the length-3 (D_rho, nu, eta) diss is
    expanded over the 7 fields, so a wrong expansion shows up as the wrong decay rate."""
    with checks() as c:
        for hyper in (1, 2):
            _decay_case(c, hyper)


# ------------------------------------------------------------- gate 4: scheme cross-checks

def _scheme_ic(x, y, z):
    shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
    a = jnp.broadcast_to(jnp.cos(x + 0.4)*jnp.cos(y)*jnp.cos(z), shp)
    b = jnp.broadcast_to(jnp.sin(2*x)*jnp.cos(y + 1.1)*jnp.sin(z), shp)
    d = jnp.broadcast_to(jnp.cos(x)*jnp.sin(y)*jnp.cos(2*z + 0.7), shp)
    return jnp.stack([1.0 + 0.1*a, 0.2*b, 0.2*d, 0.15*a, 0.2*d, 0.2*a, 1.0 + 0.2*b])


def _evolve(dt, nsteps, scheme_name, lsrk_scan=True, hoist=True, diss=0.02):
    params = fresh_params(eqpars=dict(cs0=1.0, diss=diss, hyper=1, gamma=1.0), dt=float(dt),
                          lsrk_scan=lsrk_scan, hoist_propagator=hoist, **_BOX)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_scheme_ic, params)
    stepper, scheme = get_scheme(scheme_name)
    advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))
    ops = stage_exp_ops(kgrid, params, scheme, stepper, float(dt))
    return np.asarray(advance(state, kgrid, params, nsteps, scheme, stepper).fields), ops


@pytest.mark.fp64
def test_schemes_agree_at_the_expected_order():
    """Different schemes on the same problem differ at the lower of their two orders. fp64
    only: at fp32 the pairwise differences (~5e-8 relative) are the storage noise floor, so
    no order can be measured honestly."""
    grids = ((0.02, 20), (0.01, 40), (0.005, 80))       # same total time t = 0.4
    hs, d_rk, d_imex = [], [], []
    for dt, ns in grids:
        a, _ = _evolve(dt, ns, "lsrk54")
        b, _ = _evolve(dt, ns, "rk44")
        e, _ = _evolve(dt, ns, "imexcb3e")
        scale = float(np.abs(a).max())
        hs.append(dt)
        d_rk.append(float(np.abs(a - b).max()/scale))
        d_imex.append(float(np.abs(a - e).max()/scale))
    with checks() as c:
        o_rk, o_imex = fit_order(hs, d_rk), fit_order(hs, d_imex)
        c.check(f"lsrk54 vs rk44 converge at order {o_rk:.2f} (both are 4th order)",
                3.5 < o_rk < 4.5, f"errors {d_rk}")
        c.check(f"lsrk54 vs imexcb3e converge at order {o_imex:.2f} (imexcb3e is 3rd order)",
                2.5 < o_imex < 3.5, f"errors {d_imex}")
        c.check(f"the differences are small in absolute terms ({d_rk[0]:.2e}, "
                f"{d_imex[0]:.2e})", d_rk[0] < 1e-6 and d_imex[0] < 1e-6)


def test_lsrk_scan_and_hoist_propagator_do_not_change_the_answer():
    """lsrk_scan is a loop-structure knob (agreement to round-off, not bitwise by contract),
    and hoist_propagator cannot reach CMHD at all: L is the diagonal backend, whose
    `hoistable` is False, so stage_exp_ops returns None either way and the two runs take the
    same code path."""
    tol = 1e-12 if _fp64() else 1e-5
    with checks() as c:
        base, ops_on = _evolve(0.02, 20, "lsrk54", hoist=True)
        _, ops_off = _evolve(0.02, 20, "lsrk54", hoist=False)
        hoisted, _ = _evolve(0.02, 20, "lsrk54", hoist=False)
        unrolled, _ = _evolve(0.02, 20, "lsrk54", lsrk_scan=False)
        imex_scan, _ = _evolve(0.02, 20, "imexcb3e")
        imex_unrolled, _ = _evolve(0.02, 20, "imexcb3e", lsrk_scan=False)
        c.check("stage_exp_ops returns None at hoist_propagator=True (the diagonal backend "
                "is not hoistable)", ops_on is None, f"{ops_on!r}")
        c.check("stage_exp_ops returns None at hoist_propagator=False", ops_off is None,
                f"{ops_off!r}")
        c.check("hoist_propagator on/off give the same fields",
                np.array_equal(base, hoisted))
        err = float(np.abs(base - unrolled).max()/np.abs(base).max())
        c.check(f"lsrk_scan True/False agree to {err:.2e}", err < tol, f"{err:.3e}")
        err = float(np.abs(imex_scan - imex_unrolled).max()/np.abs(imex_scan).max())
        c.check(f"imexcb3e lsrk_scan True/False agree to {err:.2e}", err < tol, f"{err:.3e}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
