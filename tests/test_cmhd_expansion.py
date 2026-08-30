# Compressible MHD in an EXPANDING BOX (physics/cmhd.py, plans/CMHD_PLAN.md Phase C3b).
# Derivation of record: docs/numerics.md "Compressible MHD" -> "Expanding box". Comoving
# frame of Squire, Chandran & Meyrand 2020: a(t) = 1 + adot*t, radial axis x, transverse
# (y, z) expanding, and the EVOLVED state is the RESCALED (primed) one
#
#     rho' = a^2 rho     B' = (a^2 B_x, a B_y, a B_z)     u unscaled
#
# so the only surviving additive expansion term is -(adot/a)*diag(0,1,1)*u. The gates, in
# the order of the plan's C3b paragraph:
#
#   (a) EXPANSION OFF IS THE PRE-C3 SOLVER. Three standing assertions, none of which needs
#       a stored reference: the expansion-off RHS is bitwise INDEPENDENT of state.t (no
#       a(t) can be in that graph if its output cannot see t -- and the expansion-on twin
#       is the discriminator); an expansion-off run reproduces an ANALYTIC non-EBM result
#       (the exact exp(-nu k^2 t) decay of a Beltrami state, whose ideal RHS vanishes
#       identically); and two independent compilations of one expansion-off config agree
#       bitwise. The stronger one-time check -- that this tree's expansion-off output and
#       its optimized-HLO opcode histogram are bitwise/identically those of the PRE-C3b
#       tree -- cannot live in a test (there is no pre-C3b tree to import here); it was run
#       out of tree over three configs at both precisions and is recorded in the Phase C3b
#       landing note in plans/CMHD_PLAN.md §5.
#   (b) THE k = 0 EXACT-ODE PAIR. rho'(k=0) and B'(k=0) are BITWISE invariants of the
#       discrete step, exactly as in the non-expanding case (the rescaling is what keeps
#       continuity a k~-divergence and induction a static-k curl). On a SPATIALLY UNIFORM
#       state, where the mean stresses that source u(0) in general vanish identically,
#       u_x(0) is bitwise too (T_x = 0) and u_perp(0) obeys du/dt = -(adot/a)u exactly,
#       i.e. u_perp(0) ~ a^-1 at scheme order.
#   (c) THE RAW BACKGROUNDS follow rho ~ a^-2, B_x ~ a^-2, B_perp ~ a^-1 -- measured
#       through cmhd.grad's own unscaling, so it gates the code and not the algebra.
#   (d) div B: K.B'^ stays a round-off random walk with expansion ON.
#   (e) WKB: a small-amplitude Alfven wave along x on a B_x background conserves wave
#       action, so its velocity amplitude follows delta_u ~ a^(-1/2).
#   (f) PLUMBING: the expansion eqpars round-trip through params.json, every configuration
#       error, and a bitwise restart mid-expansion (a(t) is reconstructed from t alone).
#   (g) The C1 dispersion physics is unchanged when expansion is absent.
#
# Single process by construction (CMHD is z_spectral, which is size==1 only). pytest, or
# `python tests/test_cmhd_expansion.py` -- never under mpirun.
from _rmhd_testing import (bootstrap, checks, fit_order, fresh_params, managed_manager,
                           snap_dir)

bootstrap()

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision, grids, snapshot_io
from taranis.physics import cmhd, construct_rhs, equation_registry
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

_B0 = 1.0
_BOX = dict(dims=3, z_spectral=True, eqtype="CMHD", adaptive_timestep=False,
            cfl_safety=0.5, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi)


def _fp64():
    return _precision.precision == "64"


def _params(n, dt, adot=None, cs_q=None, nx=None, ny=None, nz=None, **kw):
    """A fresh CMHD Parameters. `adot=None` leaves the `expansion` key ABSENT, which is the
    off switch -- never an `adot` of 0 (cmhd rejects that, and the whole point of the C3b
    contract is that off means the key is not there)."""
    eqpars = dict(kw.pop("eqpars", dict(cs0=1.0, diss=0.0, hyper=1)))
    if adot is not None:
        exp = dict(adot=adot)
        if cs_q is not None:
            exp["cs_q"] = cs_q
        eqpars["expansion"] = exp
    # dt as a plain python float: a numpy scalar would make DiagonalOperator.scaled(dt)
    # strong-typed float64 and poison the fp32 field math (the C1 close-out note)
    return fresh_params(eqpars=eqpars, dt=float(dt),
                        nx=nx or n, ny=ny or n, nz=nz or n, **dict(_BOX, **kw))


def _advance(state, kgrid, params, nsteps, schemestr="lsrk54"):
    stepper, scheme = get_scheme(schemestr)
    return jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))(
        state, kgrid, params, nsteps, scheme, stepper)


def _a_of(t, adot):
    return 1.0 + adot*float(t)


# ------------------------------------------------------------------- initial conditions

def _uniform_ic(rho=1.0, u=(0.2, 0.3, -0.15), B=(0.8, 0.3, -0.2)):
    """A spatially UNIFORM state. It stays uniform exactly (every product is uniform, so
    every k != 0 RHS mode is an exact fp zero), which is what makes the k = 0 ODE gate an
    ODE gate."""
    def ic(x, y, z):
        o = jnp.ones(jnp.broadcast_shapes(x.shape, y.shape, z.shape))
        return jnp.stack([rho*o, u[0]*o, u[1]*o, u[2]*o, B[0]*o, B[1]*o, B[2]*o])
    return ic


def _beltrami_ic(amp=1e-3):
    """u = amp*(0, sin x, cos x), B = 0, rho = 1: a Beltrami flow whose IDEAL RHS vanishes
    identically -- curl u = u so (curl u) x u = 0, |u|^2 is uniform so grad(|u|^2/2) = 0,
    j = 0, and div(rho u) = d_x u_x = 0. So an IF step reduces to exp(L*dt) and every mode
    must decay by exactly exp(-nu*k^2h*dt) per step. cs0 is set tiny by the callers so the
    round-off residual of grad(h) cannot matter either."""
    def ic(x, y, z):
        o = jnp.ones(jnp.broadcast_shapes(x.shape, y.shape, z.shape))
        return jnp.stack([1.0*o, 0.0*o, amp*jnp.sin(x)*o, amp*jnp.cos(x)*o,
                          0.0*o, 0.0*o, 0.0*o])
    return ic


def _random_ic(ms=0.3, db=0.3, drho=0.1, seed=7, nmode=6):
    """Band-limited smooth random state, the C1 conservation-gate IC (same construction, so
    the div-B gate below is directly comparable to its non-expanding twin). Each B mode is
    built perpendicular to its own k, so the IC is divergence-free to round-off."""
    rng = np.random.default_rng(seed)
    cand = [(a, b, c) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2)
            for c in (-2, -1, 0, 1, 2)
            if 1 <= a*a + b*b + c*c <= 4 and (a > 0 or (a == 0 and (b > 0 or
                                                                   (b == 0 and c > 0))))]
    modes = []
    for i in rng.choice(len(cand), size=nmode, replace=False):
        k = np.array(cand[i], dtype=float)
        modes.append((k, rng.normal(size=3), rng.normal(size=3), rng.normal(),
                      rng.uniform(0, 2*np.pi, size=3)))

    def ic(x, y, z):
        shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
        u = [jnp.zeros(shp) for _ in range(3)]
        B = [jnp.zeros(shp) for _ in range(3)]
        r = jnp.zeros(shp)
        for k, au, cb, ar, ph in modes:
            arg = k[0]*x + k[1]*y + k[2]*z
            ab = np.cross(k/np.linalg.norm(k), cb)
            cu, cbp, cr = jnp.cos(arg + ph[0]), jnp.cos(arg + ph[1]), jnp.cos(arg + ph[2])
            for i in range(3):
                u[i] = u[i] + float(au[i])*cu
                B[i] = B[i] + float(ab[i])*cbp
            r = r + float(ar)*cr
        su = ms/jnp.sqrt(jnp.mean(u[0]**2 + u[1]**2 + u[2]**2))
        sb = db*_B0/jnp.sqrt(jnp.mean(B[0]**2 + B[1]**2 + B[2]**2))
        sr = drho/jnp.sqrt(jnp.mean(r**2))
        return jnp.stack([1.0 + sr*r, su*u[0], su*u[1], su*u[2],
                          sb*B[0], sb*B[1], sb*B[2] + _B0])
    return ic


def _alfven_x_ic(eps, pol="y", B0=_B0):
    """A single FORWARD-travelling Alfven wave along x on a B = B0 xhat, rho = 1 background:
    k = (1,0,0), delta_u = eps cos(x) along `pol`, and the docs' unambiguous polarization
    delta_B = -(kpar*B0/omega) delta_u, which at omega = +kpar*v_A and v_A = B0 (rho = 1) is
    exactly delta_B = -delta_u. At t = 0, a = 1, so the primed IC IS the raw one.

    `pol` MATTERS FOR COVERAGE, not for physics. y and z are both transverse (expanding)
    directions and EBM treats them identically -- T = diag(0,1,1), Lambda = diag(2,1,1),
    A = diag(a^2,a,a) -- so the WKB prediction is the same for both. But the ELECTRIC field
    rotates with the polarization: u x B is along -zhat for a y-polarized wave (so E'_z
    carries the whole wave and E'_y is an exact zero) and along +yhat for a z-polarized one.
    The C3b review's mutation testing found that a y-only WKB gate is BLIND to dropping the
    `a` on E'_y, because it multiplies nothing. Both are run."""
    row = 2 if pol == "y" else 3

    def ic(x, y, z):
        o = jnp.ones(jnp.broadcast_shapes(x.shape, y.shape, z.shape))
        du = eps*jnp.cos(x)*o
        f = [1.0*o, 0.0*o, 0.0*o, 0.0*o, B0*o, 0.0*o, 0.0*o]
        f[row] = du
        f[row + 3] = -du
        return jnp.stack(f)
    return ic


def _div_b(fields, params, kgrid):
    """max_k |K.B'^| / (|K| * max|B'^|), the C1 metric applied to the PRIMED field. Under
    expansion the physical divergence is a^-2 times this (docs), so the two vanish
    together and the a-scaling cancels out of the normalized ratio entirely."""
    f = np.asarray(fields)
    kx, ky = np.asarray(kgrid.kx), np.asarray(kgrid.ky)
    kz = np.asarray(kgrid.kz).copy()
    kz[params.nz//2] = 0.0                       # cmhd._kz_deriv's rule
    d = np.abs(kx*f[4] + ky*f[5] + kz*f[6])
    kmag = np.broadcast_to(np.sqrt(kx**2 + ky**2 + kz**2), d.shape)
    bmag = np.sqrt(np.abs(f[4])**2 + np.abs(f[5])**2 + np.abs(f[6])**2)
    m = kmag > 0
    return float(np.max(d[m]/kmag[m])/bmag.max())


# ------------------------------------------- (a) expansion off is the pre-C3 solver

def test_expansion_off_rhs_does_not_depend_on_t():
    """THE STANDING off-is-bitwise GATE. a(t) is the ONLY thing expansion adds that can
    depend on time, and nothing else in the CMHD RHS reads state.t at all -- so if the
    expansion-off RHS output is bitwise identical at two different t on identical fields,
    no a-factor is in that graph. The expansion-ON twin is the discriminator: it must
    differ, and by an O(1)-of-the-RHS amount, not by round-off.

    (The full claim -- that the off graph is bitwise the PRE-C3b one, HLO histogram
    included -- was checked once out of tree; see the module header and the C3b landing
    note. This test is what survives here, and it is what would catch an a-factor being
    reintroduced into the off path later.)"""
    rhs = jax.jit(construct_rhs(equation_registry["CMHD"]), static_argnums=(2,))
    eqpars = dict(cs0=0.8, diss=0.01, hyper=1)
    with checks() as c:
        for label, adot in (("expansion off", None), ("expansion on", 0.25)):
            p = _params(8, 0.01, adot=adot, eqpars=eqpars)
            kgrid = jr.setup_kgrids(p)
            s = jr.initialize(_beltrami_ic(), p)
            r0, g0 = rhs(s._replace(t=jnp.float64(0.0)), kgrid, p)
            r1, _ = rhs(s._replace(t=jnp.float64(3.7)), kgrid, p)
            r0, r1 = np.asarray(r0), np.asarray(r1)
            same = np.array_equal(r0, r1)
            scale = np.abs(r0).max()
            if adot is None:
                c.check(f"{label}: the RHS is bitwise independent of state.t "
                        f"(no a(t) is in this graph)", same,
                        f"max |delta| {np.abs(r0 - r1).max():.3e}")
                c.check(f"{label}: grads.rho_p is None (no primed density is formed)",
                        g0.rho_p is None, repr(type(g0.rho_p)))
            else:
                c.check(f"{label}: the RHS DOES depend on state.t, by "
                        f"{np.abs(r0 - r1).max()/scale:.2e} of the RHS scale (the "
                        f"discriminator: the check above is not vacuous)",
                        not same and np.abs(r0 - r1).max() > 1e-3*scale,
                        f"max |delta| {np.abs(r0 - r1).max():.3e}, scale {scale:.3e}")
                c.check(f"{label}: grads.rho_p is the primed density",
                        g0.rho_p is not None, repr(type(g0.rho_p)))


def test_expansion_off_reproduces_the_analytic_beltrami_decay():
    """An analytic, reference-free statement of "the non-EBM solver still does non-EBM
    physics": on a Beltrami state the ideal RHS vanishes identically, so the IF step is
    exactly exp(L*dt) and each mode decays by exp(-nu*k^2*t) -- to round-off, computed
    here from scratch and not from any stored table. Measured relative error 2.3e-15
    (fp64), 2.8e-6 (fp32, the storage floor of a 4x decay)."""
    nu, dt, nsteps = 0.05, 0.01, 50
    p = _params(8, dt, eqpars=dict(cs0=1e-8, diss=nu, hyper=1))
    kgrid = jr.setup_kgrids(p)
    s = jr.initialize(_beltrami_ic(), p)
    a0 = np.abs(np.asarray(s.fields)[2][0, 1, 0])            # (kz, kx, ky) = (0, +1, 0)
    end = _advance(s, kgrid, p, nsteps)
    f = np.asarray(end.fields)
    pred = a0*np.exp(-nu*1.0**2*float(end.t))                # k^2 = kx^2 = 1, hyper = 1
    tol = 1e-13 if _fp64() else 1e-4
    with checks() as c:
        for i, nm in ((2, "u_y"), (3, "u_z")):
            rel = abs(np.abs(f[i][0, 1, 0])/pred - 1.0)
            c.check(f"{nm} decays by exactly exp(-nu k^2 t) (rel {rel:.2e} < {tol:.0e})",
                    rel < tol, f"{rel:.3e}")
        c.check("rho stays exactly at its k = 0 background (L(0) = 0)",
                f[0][0, 0, 0] == np.asarray(s.fields)[0][0, 0, 0],
                f"{np.asarray(s.fields)[0][0,0,0]!r} -> {f[0][0,0,0]!r}")


def test_expansion_off_is_reproducible_across_compilations():
    """Two independent Parameters (hence two traces and two compiles) of the same
    expansion-off config give bitwise identical output. Cheap, but it is what says a
    reported bitwise match elsewhere in this file is a property of the code and not of one
    lucky compile."""
    outs = []
    for _ in range(2):
        p = _params(8, 0.02, eqpars=dict(cs0=0.9, diss=0.01, hyper=1))
        kgrid = jr.setup_kgrids(p)
        end = _advance(jr.initialize(_random_ic(), p), kgrid, p, 10)
        outs.append((np.asarray(end.fields).copy(), float(end.t)))
    with checks() as c:
        c.check("two compilations of one expansion-off config agree bitwise",
                np.array_equal(outs[0][0], outs[1][0]) and outs[0][1] == outs[1][1],
                f"max |delta| {np.abs(outs[0][0] - outs[1][0]).max():.3e}")


# --------------------------------------------------- (b) the k = 0 exact-ODE pair

def test_uniform_state_is_the_exact_expansion_ode():
    """On a spatially uniform state the rescaling makes three of the four k = 0 statements
    EXACT and the fourth an ODE:
      - rho'(k=0) and B'(k=0) bitwise, exactly as without expansion (the primed continuity
        is still a k~-divergence and the primed induction still a static-k curl, so both
        RHS are exact fp zeros at k = 0);
      - u_x(0) bitwise, because T = diag(0,1,1) leaves the radial component alone AND the
        uniform state's mean Reynolds/Maxwell stresses vanish identically (on a turbulent
        state they do not -- that is the next test's discriminator);
      - u_perp(0) = u_perp(0)|_{t=0} / a(t), the solution of du/dt = -(adot/a)u.
    """
    adot, dt, nsteps = 0.3, 0.02, 100
    p = _params(8, dt, adot=adot)
    kgrid = jr.setup_kgrids(p)
    s = jr.initialize(_uniform_ic(), p)
    m0 = np.asarray(s.fields[:, 0, 0, 0]).copy()
    end = _advance(s, kgrid, p, nsteps)
    m1 = np.asarray(end.fields[:, 0, 0, 0])
    a = _a_of(end.t, adot)
    tol = 1e-9 if _fp64() else 1e-5
    with checks() as c:
        c.check(f"a({float(end.t):.3f}) = {a:.4f}: rho'(k=0) is bitwise invariant",
                m0[0] == m1[0], f"{m0[0]!r} -> {m1[0]!r}")
        c.check("every B'(k=0) component is bitwise invariant",
                bool(np.all(m0[4:7] == m1[4:7])),
                f"max |delta| {np.abs(m1[4:7] - m0[4:7]).max():.3e}")
        c.check("u_x(k=0) is bitwise invariant (T_x = 0 on a uniform state)",
                m0[1] == m1[1], f"{m0[1]!r} -> {m1[1]!r}")
        for i, nm in ((2, "u_y"), (3, "u_z")):
            pred = m0[i]/a
            rel = abs(m1[i] - pred)/abs(pred)
            c.check(f"{nm}(k=0) tracks a^-1 (rel {rel:.2e} < {tol:.0e})", rel < tol,
                    f"{m1[i]!r} vs {pred!r}")


@pytest.mark.fp64
def test_uniform_state_ode_converges_at_scheme_order():
    """THE STAGE-TIME GATE. a(t) is evaluated from grads.t, i.e. at whatever time the
    stepper put in the sub-stage state -- so the u_perp(0) ODE above converges at the
    scheme's order ONLY IF every stage carries its correct abscissa. If a stepper ever
    stops setting stage times (state._replace(t=state.t + c_k*dt)), this order collapses to
    1 and this check is what says so. Measured 4.007 for lsrk54 (nominal 4).

    fp64 only: at fp32 the ODE error at these dt sits on the ~5e-7 storage floor (measured
    5.6e-7 / 1.6e-7 / 7.9e-7 over the sweep) and no order survives."""
    adot, dts = 0.3, (0.04, 0.02, 0.01)
    errs = []
    for dt in dts:
        p = _params(8, dt, adot=adot)
        kgrid = jr.setup_kgrids(p)
        s = jr.initialize(_uniform_ic(), p)
        m0 = np.asarray(s.fields[:, 0, 0, 0]).copy()
        end = _advance(s, kgrid, p, int(round(2.0/dt)))
        m1 = np.asarray(end.fields[:, 0, 0, 0])
        pred = m0[2]/_a_of(end.t, adot)
        errs.append(float(abs(m1[2] - pred)/abs(pred)))
    order = fit_order(dts, errs)
    with checks() as c:
        c.check(f"the u_perp(0) ODE converges at order {order:.3f} (lsrk54 is 4th order; "
                f"errors {['%.2e' % e for e in errs]})", 3.0 < order < 5.0, f"{order:.4f}")


def test_primed_backgrounds_are_bitwise_on_a_turbulent_state():
    """The same rho'/B' k = 0 bitwise invariance on a band-limited random state, i.e. with
    the nonlinear terms fully active -- and with the u(k=0) discriminator, which DOES move
    here (the mean stresses source it, exactly as the non-expanding docs say)."""
    adot = 0.2
    p = _params(16, 0.02, adot=adot)
    kgrid = jr.setup_kgrids(p)
    s = jr.initialize(_random_ic(), p)
    m0 = np.asarray(s.fields[:, 0, 0, 0]).copy()
    end = _advance(s, kgrid, p, 50)
    m1 = np.asarray(end.fields[:, 0, 0, 0])
    with checks() as c:
        c.check(f"a = {_a_of(end.t, adot):.4f}: rho'(k=0) is bitwise invariant",
                m0[0] == m1[0], f"{m0[0]!r} -> {m1[0]!r}")
        c.check("every B'(k=0) component is bitwise invariant",
                bool(np.all(m0[4:7] == m1[4:7])),
                f"max |delta| {np.abs(m1[4:7] - m0[4:7]).max():.3e}")
        c.check("u(k=0) is NOT invariant on a turbulent state (the checks above are not "
                "vacuous)", bool(np.any(m0[1:4] != m1[1:4])),
                f"max |delta| {np.abs(m1[1:4] - m0[1:4]).max():.3e}")


# ------------------------------------------------------- (c) raw background scalings

def test_raw_backgrounds_track_the_expansion_scalings():
    """rho ~ a^-2, B_x ~ a^-2, B_perp ~ a^-1. Definitional GIVEN the bitwise primed k = 0
    modes above -- which is the point: it is asserted through cmhd.grad's OWN unscaling
    (rho'^/a^2, A^-1 B'^ followed by the same 13 inverse transforms), so a wrong exponent
    or a swapped A entry in the code fails here even though the algebra is trivial."""
    adot = 0.2
    p = _params(8, 0.02, adot=adot)
    kgrid = jr.setup_kgrids(p)
    grad = jax.jit(cmhd.grad, static_argnums=(2,))
    s = jr.initialize(_uniform_ic(), p)
    g0 = grad(s, kgrid, p)
    rho0 = float(np.mean(np.asarray(g0.rho)))
    B0m = np.asarray(g0.B).mean(axis=(1, 2, 3))
    end = _advance(s, kgrid, p, 100)
    g1 = grad(end, kgrid, p)
    rho1 = float(np.mean(np.asarray(g1.rho)))
    B1m = np.asarray(g1.B).mean(axis=(1, 2, 3))
    a = _a_of(end.t, adot)
    tol = 1e-12 if _fp64() else 1e-5
    with checks() as c:
        rel = abs(rho1/rho0/a**-2 - 1.0)
        c.check(f"a = {a:.4f}: the raw rho follows a^-2 (rel {rel:.2e})", rel < tol,
                f"{rho1/rho0!r} vs {a**-2!r}")
        for i, ex, nm in ((0, -2.0, "B_x"), (1, -1.0, "B_y"), (2, -1.0, "B_z")):
            rel = abs(B1m[i]/B0m[i]/a**ex - 1.0)
            c.check(f"the raw {nm} follows a^{ex:.0f} (rel {rel:.2e})", rel < tol,
                    f"{B1m[i]/B0m[i]!r} vs {a**ex!r}")


# ---------------------------------------------------------------------- (d) div B

def test_div_b_stays_a_round_off_random_walk_under_expansion():
    """The C1 div-B property survives expansion for a structural reason: E' rescaling keeps
    induction a curl on the STATIC integer-grid K, so K.dB' is still a pairwise-cancelling
    sum with no systematic source, and the propagator and dealias mask still scale all
    three B' components identically. Same gate shape as the non-expanding twin: the
    eps*sqrt(N) class plus the growth SHAPE (a random walk multiplies by ~sqrt(10) over a
    10x longer run, a systematic source by 10). Measured max 6.2e-17 (fp64, growth 3.8) and
    2.9e-8 (fp32, growth 1.0 -- already saturated at its floor by step 10)."""
    p = _params(16, 0.02, adot=0.2)
    kgrid = jr.setup_kgrids(p)
    state = jr.initialize(_random_ic(), p)
    trace = [_div_b(state.fields, p, kgrid)]
    for _ in range(10):
        state = _advance(state, kgrid, p, 10)
        trace.append(_div_b(state.fields, p, kgrid))
    eps = np.finfo(np.float64 if _fp64() else np.float32).eps
    bound = 20.0*eps*np.sqrt(500)                # 5 stages/step * 100 steps, 20x margin
    with checks() as c:
        c.check(f"the IC is divergence-free to round-off ({trace[0]:.2e})",
                trace[0] < 100*eps, f"{trace[0]:.3e}")
        c.check(f"div B' stays in the eps*sqrt(N) class under expansion: max "
                f"{max(trace):.2e} < {bound:.2e}", max(trace) < bound,
                f"trace {[f'{v:.2e}' for v in trace]}")
        growth = trace[-1]/trace[1]
        c.check(f"the growth from 10 to 100 steps is a random walk, not linear "
                f"(x{growth:.2f}; sqrt(10) = 3.2, linear would be 10)", growth < 5.0,
                f"trace {[f'{v:.2e}' for v in trace]}")


# ------------------------------------------------------------------------ (e) WKB

# WKB gate configuration. Box 2*pi in x with nx = 16, so the mode index (1,0,0) is
# k_x = 1 exactly; ny = nz = 4 because the wave is y- and z-independent and those axes
# only have to exist. cs0 = 0.5 keeps the CFL comfortable at the fixed dt below and is
# irrelevant to an Alfven wave. eps = 1e-5 puts the O(eps) self-interaction ~1e-5 relative,
# far under the exponent tolerance.
_WKB = dict(nx=16, ny=4, nz=4, cs0=0.5, eps=1e-5, dt=0.05, nchunk=40)


def _wkb_run(adot, nstep_chunk, pol="y"):
    """Return (a samples, |du^(kx=1)| samples, |dB'^(kx=1)| samples) for the `pol`-polarized
    wave. For a single TRAVELLING wave the complex Fourier coefficient at k_x = +1 has
    constant modulus and rotating phase, so the modulus IS the WKB envelope -- no fitting of
    an oscillation."""
    p = _params(None, _WKB["dt"], adot=adot, nx=_WKB["nx"], ny=_WKB["ny"], nz=_WKB["nz"],
                eqpars=dict(cs0=_WKB["cs0"], diss=0.0, hyper=1))
    kgrid = jr.setup_kgrids(p)
    s = jr.initialize(_alfven_x_ic(_WKB["eps"], pol), p)
    row = 2 if pol == "y" else 3
    f = np.asarray(s.fields)
    idx = (0, 1, 0)                                          # (kz, kx, ky) = (0, +1, 0)
    ts, au, ab = [float(s.t)], [abs(f[row][idx])], [abs(f[row + 3][idx])]
    for _ in range(_WKB["nchunk"]):
        s = _advance(s, kgrid, p, nstep_chunk)
        f = np.asarray(s.fields)
        ts.append(float(s.t))
        au.append(abs(f[row][idx])); ab.append(abs(f[row + 3][idx]))
    return 1.0 + adot*np.array(ts), np.array(au), np.array(ab)


def test_wkb_alfven_amplitude_follows_a_minus_half():
    """WKB / wave-action conservation for an Alfven wave along the RADIAL axis:
    delta_u ~ a^(-1/2), which is the amplitude growth relative to the background that drives
    switchback formation (docs/numerics.md "Expanding box"). |B'_y| must follow the same
    exponent, because delta_b = delta_B/sqrt(rho) = B'_y/sqrt(rho'_0) exactly under the
    rescaling -- so the primed magnetic amplitude IS the normalized one.

    TOLERANCE BUDGET, stated as arithmetic rather than picked. The WKB small parameter is
    adot/(a*omega). Here omega(t) = k_x*v_A(t) with k_x STATIC (radial) and
    v_A = B_x/sqrt(rho) ~ a^-2/a^-1 = a^-1, so omega ~ a^-1 and

        eps_WKB = adot/(a*omega) = adot/(k_x*v_A0)      -- CONSTANT over the whole run.

    At k_x = 1, v_A0 = B0 = 1 and adot = 0.02 that is eps_WKB = 0.02. The next-order WKB
    correction is O(eps_WKB) in the amplitude, i.e. at most eps_WKB in log-amplitude, which
    over the fitted range Delta ln a = ln 4 = 1.386 is an exponent budget of
    eps_WKB/Delta ln a = 0.0144. That is the gate. Measured |p + 1/2| = 1.8e-3 for u_y and
    1.8e-3 for B'_y, 8x inside the budget and IDENTICAL at fp32 (the amplitudes are 1e-5 in
    their own otherwise-empty field rows, so fp32 costs nothing here).

    The discriminator is the second half: halving adot must halve the deviation, since the
    residual is first order in eps_WKB. Measured 1.84e-3 -> 5.4e-4, a factor 3.4 -- which is
    what says the residual is the WKB correction and not a coincidence at one adot. The gate
    only asserts that it FALLS by at least 1.5x, since the sub-leading terms are not gated.

    a spans 1 -> 4 (a factor 4, not the docs' full decade: a decade at this eps_WKB costs
    ~4x the steps for no extra discrimination -- the run is 3000 steps as it stands).

    BOTH transverse polarizations are run. The physics and the prediction are identical
    (y and z are the two expanding directions and every EBM diagonal treats them the same),
    but the electric field rotates with the polarization -- see `_alfven_x_ic`. A y-only
    gate leaves E'_y multiplying an exact zero, which is how the C3b review's E'_y mutation
    slipped through the first version of this file."""
    adot_ref = 0.02
    eps_wkb = adot_ref/(1.0*_B0)                             # adot/(k_x v_A0), k_x = 1
    budget = eps_wkb/np.log(4.0)
    with checks() as c:
        for pol in ("y", "z"):
            # 40*75 steps * 0.05 = t 150; E' is along z for pol y and along y for pol z
            a, au, ab = _wkb_run(adot_ref, 75, pol)
            pu = float(np.polyfit(np.log(a), np.log(au), 1)[0])
            pb = float(np.polyfit(np.log(a), np.log(ab), 1)[0])
            e = "z" if pol == "y" else "y"
            c.check(f"pol {pol} (E' along {e}): a spans {a[0]:.2f} -> {a[-1]:.2f} and "
                    f"eps_WKB = adot/(k_x v_A0) = {eps_wkb:.3f}, so the exponent budget is "
                    f"eps_WKB/ln(a_end) = {budget:.4f}", abs(a[-1] - 4.0) < 1e-6,
                    f"a_end {a[-1]}")
            c.check(f"pol {pol}: |u_{pol}| exponent {pu:+.5f} vs the WKB -1/2 (|delta| "
                    f"{abs(pu+0.5):.2e} < {budget:.4f})", abs(pu + 0.5) < budget,
                    f"{pu:.6f}")
            c.check(f"pol {pol}: |B'_{pol}| exponent {pb:+.5f} vs the WKB -1/2 (|delta| "
                    f"{abs(pb+0.5):.2e})", abs(pb + 0.5) < budget, f"{pb:.6f}")
            c.check(f"pol {pol}: the amplitude actually falls by ~a^-1/2 over the run "
                    f"({au[-1]/au[0]:.4f} vs {a[-1]**-0.5:.4f}) -- not a flat fit",
                    abs(au[-1]/au[0]/a[-1]**-0.5 - 1.0) < 0.02, f"{au[-1]/au[0]:.5f}")
            if pol == "y":
                pu_ref = pu
        # the discriminator: same a range, half the WKB parameter, twice the steps
        a2, au2, _ = _wkb_run(adot_ref/2, 150)
        pu2 = float(np.polyfit(np.log(a2), np.log(au2), 1)[0])
        c.check(f"halving eps_WKB shrinks the deviation ({abs(pu_ref+0.5):.2e} -> "
                f"{abs(pu2+0.5):.2e}), i.e. the residual is first order in adot/(a*omega)",
                abs(pu2 + 0.5) < abs(pu_ref + 0.5)/1.5,
                f"{abs(pu_ref+0.5):.3e} -> {abs(pu2+0.5):.3e}")


# ------------------------------------------------------------------- (f) plumbing

_PLUMB = dict(cs0=1.1, diss=(0.01, 0.02, 0.03), hyper=2,
              expansion=dict(adot=0.15, cs_q=4.0/3.0))


def test_expansion_eqpars_round_trip_through_params_json():
    """The nested expansion dict is plain JSON, so it round-trips like every other eqpars
    entry (config._lists_to_tuples recurses into dicts, so the sub-dict comes back a dict).
    params.save's differing-record check then stops a cross-mode restart exactly as it does
    for z_spectral -- asserted here by trying to save a DIFFERENT adot over the record, and
    by trying to save a non-expanding record over an expanding one."""
    params = fresh_params(eqpars=dict(_PLUMB), dt=0.01, nx=8, ny=8, nz=8, **_BOX)
    with snap_dir("cmhd_ebm_params_") as d, checks() as c:
        params.save(d)
        back = jr.Parameters.from_snapshot(d)
        c.check(f"eqpars round-trip: {back.eqpars!r}", back.eqpars == _PLUMB,
                f"{back.eqpars!r} != {_PLUMB!r}")
        c.check("the expansion block comes back as a dict",
                isinstance(back.eqpars["expansion"], dict),
                repr(type(back.eqpars["expansion"])))
        c.check("cmhd._expansion reads it back as (adot, cs_q)",
                cmhd._expansion(back) == (0.15, 4.0/3.0), repr(cmhd._expansion(back)))
        params.save(d)          # identical re-save is a no-op
        c.check("identical re-save is a no-op", True)
        for label, ep in (("a different adot",
                           dict(_PLUMB, expansion=dict(adot=0.3, cs_q=4.0/3.0))),
                          ("dropping expansion entirely",
                           {k: v for k, v in _PLUMB.items() if k != "expansion"})):
            with pytest.raises(ValueError, match="eqpars"):
                fresh_params(eqpars=ep, dt=0.01, nx=8, ny=8, nz=8, **_BOX).save(d)
            c.check(f"saving {label} over the record is a hard error", True)


def test_cs_q_defaults_to_four_thirds_and_is_honoured():
    """cs_q is optional and defaults to 4/3 (Squire et al.'s Athena++ "adiabatic" mimic);
    cs_q = 0 is constant temperature. Both reach the RHS: at cs_q = 0 the sound speed never
    cools, so a run started from the same state diverges from the default one."""
    with checks() as c:
        p_default = _params(8, 0.01, adot=0.2)
        c.check("cs_q defaults to 4/3", cmhd._expansion(p_default) == (0.2, 4.0/3.0),
                repr(cmhd._expansion(p_default)))
        c.check("an explicit cs_q is read back",
                cmhd._expansion(_params(8, 0.01, adot=0.2, cs_q=0.5)) == (0.2, 0.5),
                repr(cmhd._expansion(_params(8, 0.01, adot=0.2, cs_q=0.5))))
        ends = []
        for cs_q in (4.0/3.0, 0.0):
            p = _params(8, 0.02, adot=0.3, cs_q=cs_q)
            kgrid = jr.setup_kgrids(p)
            end = _advance(jr.initialize(_random_ic(), p), kgrid, p, 40)
            ends.append(np.asarray(end.fields))
        d = np.abs(ends[0] - ends[1]).max()/np.abs(ends[0]).max()
        c.check(f"cs_q = 0 (no cooling) gives a different run from cs_q = 4/3 "
                f"(rel {d:.2e}) -- the cooling law reaches the RHS", d > 1e-4, f"{d:.3e}")


def test_expansion_configuration_errors_raise():
    """Everything the expansion block rejects. All at setup_kgrids time: config.py stays
    equation-agnostic, so these are cmhd._expansion errors reached through _eqpars."""
    def raises(c, label, match, expansion, **eq):
        eqpars = dict(cs0=1.0, diss=0.0, hyper=1, **eq)
        if expansion is not None:
            eqpars["expansion"] = expansion
        try:
            jr.setup_kgrids(fresh_params(eqpars=eqpars, dt=0.01, nx=8, ny=8, nz=8, **_BOX))
            raised = ""
        except (ValueError, NotImplementedError) as exc:
            raised = str(exc)
        c.check(f"{label} raises, naming {match!r}", match in raised,
                f"raised: {raised!r}")

    with checks() as c:
        raises(c, "gamma != 1 with expansion", "isothermal", dict(adot=0.1),
               gamma=5.0/3.0)
        raises(c, "adot = 0", "adot", dict(adot=0.0))
        raises(c, "adot < 0", "adot", dict(adot=-0.1))
        raises(c, "cs_q < 0", "cs_q", dict(adot=0.1, cs_q=-1.0))
        raises(c, "a missing adot", "adot", dict(cs_q=1.0))
        raises(c, "an unknown expansion key", "unknown", dict(adot=0.1, q=1.0))
        raises(c, "a non-dict expansion", "dict", 0.1)
        # gamma > 1 without expansion is still fine, and so is the default cs_q
        c.check("gamma = 5/3 is still accepted with expansion ABSENT",
                jr.setup_kgrids(fresh_params(
                    eqpars=dict(cs0=1.0, diss=0.0, hyper=1, gamma=5.0/3.0), dt=0.01,
                    nx=8, ny=8, nz=8, **_BOX)) is not None)
        c.check("gamma = 1 stated explicitly is accepted with expansion",
                jr.setup_kgrids(fresh_params(
                    eqpars=dict(cs0=1.0, diss=0.0, hyper=1, gamma=1.0,
                                expansion=dict(adot=0.1)), dt=0.01,
                    nx=8, ny=8, nz=8, **_BOX)) is not None)


def test_restart_mid_expansion_is_bitwise():
    """Snapshots store the PRIMED state and t, and nothing else: a(t) is reconstructed from
    t at every use site, so a run stopped mid-expansion and resumed reproduces the
    uninterrupted trajectory bitwise. Fixed dt, nu > 0, the standard forcing_scale zeros
    path -- the C1 restart gate with expansion turned on."""
    p = _params(8, 0.02, adot=0.25, eqpars=dict(cs0=1.0, diss=0.01, hyper=1))
    kgrid = jr.setup_kgrids(p)
    ic = _random_ic()
    straight = _advance(jr.initialize(ic, p), kgrid, p, 20)
    ref_fields, ref_t = np.asarray(straight.fields).copy(), float(straight.t)

    mid = _advance(jr.initialize(ic, p), kgrid, p, 10)
    with snap_dir("cmhd_ebm_restart_") as d:
        with managed_manager(p, d, nsnap=2) as mngr:
            snapshot_io.save_snapshot(0, mid, mngr, p)
            mngr.wait_until_finished()
        reloaded = snapshot_io.load_snapshot(0, d, p)
    resumed = _advance(reloaded, kgrid, p, 10)
    with checks() as c:
        c.check(f"the restart happens mid-expansion (a = {_a_of(mid.t, 0.25):.4f}, not 1)",
                _a_of(mid.t, 0.25) > 1.04, f"{_a_of(mid.t, 0.25)}")
        c.check("the reloaded state is bitwise the checkpointed one",
                np.array_equal(np.asarray(reloaded.fields), np.asarray(mid.fields))
                and float(reloaded.t) == float(mid.t))
        c.check(f"the resumed run ends at the same t ({float(resumed.t)!r})",
                float(resumed.t) == ref_t, f"{float(resumed.t)!r} vs {ref_t!r}")
        c.check("the resumed run is bitwise the uninterrupted one",
                np.array_equal(np.asarray(resumed.fields), ref_fields),
                f"max |delta| "
                f"{np.abs(np.asarray(resumed.fields) - ref_fields).max():.3e}")


# ------------------------------- the raw-frame RHS cross-check (the metric-factor gate)
#
# WHY THIS GATE EXISTS. The structural gates above (bitwise k=0, div B, the uniform-state
# ODE) all hold for a WIDE class of wrong a-factors: a k=0 mode sees no metric at all, div
# B' survives ANY diagonal rescaling of E', and a uniform state has zero curl and zero
# gradient. The C3b adversarial review's mutation testing made that concrete -- dropping the
# `a` on E'_y, and using ky instead of ky/a in the physical curls, both passed the whole
# original gate set. This test is the one that reads every metric factor at once: it
# reconstructs the RAW-frame time derivatives from the production primed RHS and compares
# them against an independent transcription of Squire et al.'s raw equations.

_LAMBDA = (2.0, 1.0, 1.0)               # B expansion diag; A = diag(a^2, a, a) = a^_LAMBDA
_T_DIAG = (0.0, 1.0, 1.0)               # u expansion diag


def _reference_raw_rhs(state, kgrid, params, cs0, adot, cs_q):
    """An INDEPENDENT transcription of docs/numerics.md "Expanding box" eqs (1)-(3), in
    RAW variables and ADVECTIVE form -- deliberately none of the three forms the production
    code uses:

        d_t rho = -u.grad~ rho - rho div~ u          - 2(adot/a) rho     [not flux form]
        d_t u   = -(u.grad~)u - grad~ h + j x B/rho  - (adot/a) T u      [not rotational]
        d_t B   = B.grad~ u - B div~ u - u.grad~ B   - (adot/a) Lambda B [not curl form]

    with its own k~ arrays, its own a-powers, and its own Nyquist zeroing. It shares only
    grids.fft/ifft (the transform primitive is not what any metric mutation touches) and
    the dealias mask, for the reason in the test docstring. `div~ B` is dropped from the
    induction identity: it is 1e-17 here, i.e. below the tolerance by four orders.

    Both this and `_production_raw_rhs` run EAGERLY, not jitted: `a` is read off the state
    as a host float, which is part of what makes this an independent construction rather
    than a second copy of `cmhd._a_of`. One RHS evaluation on a 16^3 grid is cheap."""
    a = 1.0 + adot*float(state.t)
    ah = adot/a
    cs2 = cs0*cs0 * a**(-cs_q)
    # k~ = (kx, ky/a, kz/a), built here from scratch -- numpy so a shared jnp expression
    # cannot be the thing that agrees
    kzn = np.asarray(kgrid.kz, dtype=np.float64).copy()
    if params.nz % 2 == 0:
        kzn[params.nz//2] = 0.0
    kt = (jnp.asarray(np.asarray(kgrid.kx, dtype=np.float64)),
          jnp.asarray(np.asarray(kgrid.ky, dtype=np.float64)/a),
          jnp.asarray(kzn/a))
    mask = kgrid.dealias

    def dd(fkc, i):
        """the real-space d~_i of a k-space field"""
        return grids.ifft(1j*kt[i]*fkc, params)

    # unscale to raw variables IN K-SPACE (the production path unscales rho in real space
    # and B in k-space; doing both here in k-space is one more place the two differ)
    fk = state.fields
    rhok = fk[0]/a**2
    uk = fk[1:4]
    bk = jnp.stack([fk[4]/a**2, fk[5]/a, fk[6]/a])
    rho = grids.ifft(rhok, params)
    u = jnp.stack([grids.ifft(uk[i], params) for i in range(3)])
    B = jnp.stack([grids.ifft(bk[i], params) for i in range(3)])

    du_dx = [[dd(uk[i], j) for j in range(3)] for i in range(3)]     # du_dx[i][j] = d~_j u_i
    dB_dx = [[dd(bk[i], j) for j in range(3)] for i in range(3)]
    divu = du_dx[0][0] + du_dx[1][1] + du_dx[2][2]

    # continuity, advective
    drho = -(u[0]*dd(rhok, 0) + u[1]*dd(rhok, 1) + u[2]*dd(rhok, 2)) - rho*divu

    # momentum: advective inertia + the Lorentz force with an independently formed j
    jj = jnp.stack([dd(bk[2], 1) - dd(bk[1], 2),
                    dd(bk[0], 2) - dd(bk[2], 0),
                    dd(bk[1], 0) - dd(bk[0], 1)])
    lor = _cross_np(jj, B)/rho
    duu = [lor[i] - (u[0]*du_dx[i][0] + u[1]*du_dx[i][1] + u[2]*du_dx[i][2])
           for i in range(3)]
    hk = grids.fft(cs2*jnp.log(rho), params)

    # induction, advective (the curl identity expanded)
    dBB = [B[0]*du_dx[i][0] + B[1]*du_dx[i][1] + B[2]*du_dx[i][2] - B[i]*divu
           - (u[0]*dB_dx[i][0] + u[1]*dB_dx[i][1] + u[2]*dB_dx[i][2]) for i in range(3)]

    rows = [grids.fft(drho, params)*mask - 2.0*ah*rhok]
    for i in range(3):
        rows.append((grids.fft(duu[i], params) - 1j*kt[i]*hk)*mask - ah*_T_DIAG[i]*uk[i])
    for i in range(3):
        rows.append(grids.fft(dBB[i], params)*mask - ah*_LAMBDA[i]*bk[i])
    return jnp.stack(rows)


def _cross_np(a, b):
    return jnp.stack([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2],
                      a[0]*b[1] - a[1]*b[0]])


def _production_raw_rhs(state, kgrid, params, adot):
    """The production PRIMED RHS, converted to raw-frame time derivatives. The a-dot terms
    reappear here, which is the whole point of the rescaling: with rho = rho'/a^2,

        d_t rho = (d_t rho')/a^2 - 2(adot/a) rho,

    and likewise B_i = B'_i/A_i with A = diag(a^2, a, a), whose logarithmic derivative
    Adot_i/A_i is exactly Lambda_i*(adot/a). u is unscaled, so its rows pass through."""
    a = 1.0 + adot*float(state.t)
    ah = adot/a
    prod, _ = construct_rhs(equation_registry["CMHD"])(state, kgrid, params)
    fk = state.fields
    rows = [prod[0]/a**2 - 2.0*ah*(fk[0]/a**2)]
    rows += [prod[1+i] for i in range(3)]
    for i in range(3):
        apow = a**_LAMBDA[i]
        rows.append(prod[4+i]/apow - ah*_LAMBDA[i]*(fk[4+i]/apow))
    return jnp.stack(rows)


@pytest.mark.fp64
def test_raw_frame_rhs_matches_the_advective_reference():
    """THE METRIC-FACTOR GATE. One RHS evaluation on a smooth random state at a != 1,
    converted to the raw frame, against the independent advective transcription above.
    Every a and every k~ in the production path is load-bearing here: get one wrong and
    the two forms disagree at O(1) of the row, not at round-off.

    Two discretization points, both settled by the band limit:

    1. THE REFERENCE MUST CARRY THE SAME DEALIAS MASK. `1/rho` and `ln rho` are not
       polynomial, so both forms put power past the 2/3 cut; the mask is part of the
       discretization under test, not part of the physics, and comparing a masked RHS with
       an unmasked one would differ by that residual (~1e-8 here) rather than by round-off.
       The a-dot terms are OUTSIDE the mask in both -- they are exact algebra of the
       variable change, not a nonlinear term -- and it makes no difference anyway because
       `initialize` leaves the state mask-supported, which the test asserts.
    2. WHY THE TWO FORMS AGREE AT ROUND-OFF (round-2 review correction): NOT because the
       state is band-limited below the products' aliasing threshold -- the compared state
       is 100 nonlinear steps past the IC and fills the whole mask band (measured 3.8e-3
       relative power at per-axis |n| = 5), so quadratic products reach |n| = 10 > Nyquist.
       The real mechanism is the 2/3 rule itself: for any MASK-SUPPORTED state the RETAINED
       modes of a quadratic product are exact (aliases from |n| <= 2N/3 inputs land only on
       modes the mask kills), so the advective and rotational/flux/curl forms agree on the
       retained band identically; the non-polynomial pieces (ln rho, 1/rho) are common-mode
       between the two constructions. Measured here: max rel 1.4e-15 on that filled-band
       state. The |n| <= 2 IC is a convenience, not a load-bearing assumption.

    fp64 only: the comparison is a round-off-scale one and there is nothing to say at fp32.
    """
    cs0, adot, cs_q = 1.0, 0.35, 4.0/3.0
    p = _params(16, 0.02, adot=adot, cs_q=cs_q, eqpars=dict(cs0=cs0, diss=0.0, hyper=1))
    kgrid = jr.setup_kgrids(p)
    state = _advance(jr.initialize(_random_ic(), p), kgrid, p, 100)
    a = _a_of(state.t, adot)
    ref = np.asarray(_reference_raw_rhs(state, kgrid, p, cs0, adot, cs_q))
    got = np.asarray(_production_raw_rhs(state, kgrid, p, adot))
    names = ("rho", "u_x", "u_y", "u_z", "B_x", "B_y", "B_z")
    tol = 1e-11
    with checks() as c:
        c.check(f"the state is dealias-mask-supported, so the a-dot terms' placement "
                f"relative to the mask cannot matter",
                np.array_equal(np.asarray(state.fields*kgrid.dealias),
                               np.asarray(state.fields)))
        c.check(f"a = {a:.4f} is meaningfully away from 1", a > 1.5, f"{a}")
        for i, nm in enumerate(names):
            scale = np.abs(ref[i]).max()
            rel = np.abs(got[i] - ref[i]).max()/scale
            c.check(f"d_t {nm} matches the advective raw-form reference "
                    f"(rel {rel:.2e} < {tol:.0e}, row scale {scale:.3e})", rel < tol,
                    f"{rel:.3e}")


# --------------------------------------------------------- the CFL under expansion

def test_cfl_uses_physical_spacings_under_expansion():
    """set_timestep's EBM branch: physical spacings (dx, a*dy, a*dz) and the cooled cs(t),
    with the speeds taken from the UNPRIMED fields (which is exactly what grads carries, so
    nothing is unscaled twice). Both effects RELAX the timestep as the box expands, so on
    IDENTICAL fields at a later t the returned dt must be larger -- and with the expansion
    key absent it must be bitwise the same at both times, which is the discriminator.

    The box is deliberately transverse-limited (ny = nz = 2*nx, so dy = dz = dx/2 at a = 1
    and a*dy stays under dx for the whole a range used): the CFL is a max over the three
    directions, and on a cubic box the fixed radial direction would mask the transverse
    stretching entirely.

    THE cs_q = 0 CELL IS THE LOAD-BEARING ONE. With cs_q = 4/3 the two effects are summed,
    and a mutant using STATIC spacings still reaches x1.0675 on cooling alone (measured, on
    the mutant, by the C3b review and again here) -- so a threshold anywhere near 1 gates
    nothing. At cs_q = 0 the sound speed is constant and the physical spacings are almost
    the whole relaxation: the static-spacing mutant is then measured at x1.0006 -- not
    exactly 1, because `grad` still unscales the fields and v_A^2 = |B|^2/rho carries its
    own a-dependence -- against x1.8011 for the real code. The x1.4 threshold separates
    them by a mile. The cs_q = 4/3 cell is kept as a second cell with a threshold above
    the cooling-only figure."""
    box = dict(nx=8, ny=16, nz=16)
    st = jax.jit(lambda s, kg, p: cmhd.set_timestep(cmhd.grad(s, kg, p), p),
                 static_argnums=(2,))

    def dt_pair(adot, cs_q):
        p = _params(None, 0.5, adot=adot, cs_q=cs_q, adaptive_timestep=True, **box)
        kgrid = jr.setup_kgrids(p)
        s = jr.initialize(_random_ic(), p)
        return (float(st(s._replace(t=jnp.float64(0.0)), kgrid, p)),
                float(st(s._replace(t=jnp.float64(2.0)), kgrid, p)))

    with checks() as c:
        # (1) the discriminating cell: no cooling, so only a*dy and a*dz can move dt
        d0, d1 = dt_pair(0.4, 0.0)
        c.check(f"cs_q = 0, a = 1.8: the PHYSICAL SPACINGS alone relax the CFL on identical "
                f"fields ({d0:.6g} -> {d1:.6g}, x{d1/d0:.4f} > 1.4 -- a static-spacing "
                f"mutant measures x1.0006 here)", d1 > d0*1.4, f"{d0!r} -> {d1!r}")
        # (2) the default cooling law on top; the threshold clears the cooling-only x1.0675
        d0, d1 = dt_pair(0.4, 4.0/3.0)
        c.check(f"cs_q = 4/3, a = 1.8: spacings and cooling together give x{d1/d0:.4f} "
                f"(> 1.5; cooling alone reaches only x1.07)", d1 > d0*1.5,
                f"{d0!r} -> {d1!r}")
        # (3) the control: with no expansion key, dt cannot move with t at all
        p = _params(None, 0.5, adaptive_timestep=True, **box)
        kgrid = jr.setup_kgrids(p)
        s = jr.initialize(_random_ic(), p)
        e0 = float(st(s._replace(t=jnp.float64(0.0)), kgrid, p))
        e1 = float(st(s._replace(t=jnp.float64(2.0)), kgrid, p))
        c.check("expansion off: dt is bitwise the same at t = 0 and t = 2 (the control for "
                "the two checks above)", e0 == e1, f"{e0!r} {e1!r}")
        # (4) and an actual adaptive run stays finite through a factor-6 expansion
        p = _params(None, 0.5, adot=0.4, adaptive_timestep=True, **box)
        kgrid = jr.setup_kgrids(p)
        end = _advance(jr.initialize(_random_ic(), p), kgrid, p, 60)
        f = np.asarray(end.fields)
        c.check(f"an adaptive-dt expansion run is finite through a = "
                f"{_a_of(end.t, 0.4):.2f} (max |field| {np.abs(f).max():.3e})",
                bool(np.all(np.isfinite(f))) and _a_of(end.t, 0.4) > 2.0,
                f"t {float(end.t)}")


# --------------------------------------------- (g) C1 dispersion, expansion absent

def test_parallel_alfven_dispersion_is_unchanged_with_expansion_absent():
    """One C1 dispersion configuration driven through the C3b code path: the exactly-
    parallel Alfven wave of tests/test_cmhd_linear.py's _ANGLES[0] on the same 8^3 2*pi box
    and the same B0 = 1 background, at cs0/v_A = 0.5. With no `expansion` key the measured
    omega must still be k_par*v_A. (test_cmhd_linear.py is the full dispersion coverage --
    every branch, angle, gamma and cs0/v_A; this is the in-file statement that the EBM
    edits did not move it.)"""
    eps, dt, nchunk, nstep = 1e-6 if _fp64() else 1e-3, 0.05, 8, 25
    p = _params(8, dt, eqpars=dict(cs0=0.5, diss=0.0, hyper=1))
    kgrid = jr.setup_kgrids(p)

    def ic(x, y, z):
        # k = (0,0,1) zhat, B = B0 zhat: Alfven branch, du ~ yhat, drho = 0, and the docs'
        # dB = -(kpar B0/omega) du, which at omega = +kpar v_A and v_A = B0 is dB_y = -du_y
        o = jnp.ones(jnp.broadcast_shapes(x.shape, y.shape, z.shape))
        du = eps*jnp.cos(z)*o
        return jnp.stack([1.0*o, 0.0*o, du, 0.0*o, 0.0*o, -du, _B0*o])

    s = jr.initialize(ic, p)
    idx = (1, 0, 0)                                          # (kz, kx, ky) = (+1, 0, 0)
    ts, ph, amp = [float(s.t)], [np.angle(np.asarray(s.fields)[2][idx])], []
    amp.append(abs(np.asarray(s.fields)[2][idx]))
    for _ in range(nchunk):
        s = _advance(s, kgrid, p, nstep)
        f = np.asarray(s.fields)
        ts.append(float(s.t)); ph.append(np.angle(f[2][idx])); amp.append(abs(f[2][idx]))
    # omega from the unwrapped phase: a travelling mode is u^ ~ exp(-i omega t), and
    # omega*dt_chunk = 1.25 rad < pi per sample, so np.unwrap is unambiguous
    w = float(-np.polyfit(np.array(ts), np.unwrap(np.array(ph)), 1)[0])
    w_exact = 1.0*_B0                                        # kpar = 1, v_A = B0/sqrt(rho)
    tol = 1e-6 if _fp64() else 2e-4
    amp_tol = 1e-6 if _fp64() else 1e-2
    with checks() as c:
        rel = abs(w/w_exact - 1.0)
        c.check(f"the parallel Alfven frequency is {w:.9f} vs the analytic "
                f"{w_exact:.9f} (rel {rel:.2e} < {tol:.0e})", rel < tol, f"{w!r}")
        drift = abs(max(amp)/min(amp) - 1.0)
        c.check(f"the amplitude does not drift (nu = 0; {drift:.2e} < {amp_tol:.0e})",
                drift < amp_tol, f"{drift:.3e}")
        c.check("this config carries no expansion key", cmhd._expansion(p) is None,
                repr(cmhd._expansion(p)))


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
