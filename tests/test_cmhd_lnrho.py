# Compressible MHD in the ln rho DENSITY VARIABLE (physics/cmhd.py, plans/CMHD_PLAN.md
# Phase C4). Derivation of record: docs/numerics.md "Compressible MHD" -> "The ln rho
# density variable". With eqpars["density_var"] = "lnrho", field 0 is s = ln rho instead of
# rho:
#
#     d_t s = -u.grad s - div u          (the flux form divided by rho)
#     h     = cs^2 s                     gamma = 1  (linear: a k-local pressure force)
#     h     = cs0^2 e^((gamma-1)s)/(gamma-1)   gamma > 1
#
# with 1/rho = e^-s the only non-polynomial factor left at gamma = 1. The gates, in the
# order of the plan's C4 paragraph:
#
#   (a) THE "rho" PATH IS UNCHANGED. density_var absent and density_var="rho" produce a
#       bitwise identical RHS and a bitwise identical run, two compilations agree bitwise,
#       and grads' lnrho members (s, gs, inv_rho) are None on that path -- with the lnrho
#       twin as the discriminator for each. The stronger one-time claim (this tree's rho
#       path is bitwise the PRE-C4 tree's, optimized-HLO histogram included) cannot live in
#       a test; it was run out of tree over three configs at both precisions and is recorded
#       in the C4 landing note in plans/CMHD_PLAN.md §5.
#   (b) DISPERSION. The C1 eigenmodes driven through the lnrho RHS, with delta_s = delta_rho
#       about (rho0 = 1, s0 = 0): Alfven, fast and slow, at both gamma.
#   (c) EXACT DECAY of a single s mode under dissipation, along all three axes (the C1
#       review's lesson: a z-only decay gate cannot see kgrid.ksq being dropped from L).
#       Plus set_timestep against a numpy CFL bound rebuilt on rho = e^s -- a cell the plan's
#       list does not name, added because every other gate here runs at FIXED dt and the
#       mutation "gamma > 1 uses the constant cs0^2" therefore passed all of them.
#   (d) MASS int e^s. NOT bitwise any more, and that is the documented price of the variable
#       change: the conserved functional is nonlinear in the evolved field, so RK preserves
#       it only to scheme order. Gated by drift ORDER plus absolute smallness, never at
#       round-off -- with mean B bitwise and div B in the eps*sqrt(N) class asserted on the
#       SAME run, since induction never sees field 0 and must be untouched.
#   (e) THE UNIFORM STATE. Every s-RHS term is an exact fp zero there, so s^(k=0) is bitwise
#       (and the whole state stays exactly uniform); on a turbulent state <s> drifts, which
#       is the physics and the discriminator.
#   (f) THE ENERGY BUDGET at both gamma, and the documented INVERSION trap: at gamma = 1 the
#       constant in delta E/delta s multiplies <e^s D_s s>, whose mean is NOT zero, so the
#       +cs^2 term MUST be kept -- unlike the rho form, where the analogous constant
#       provably dropped out. Built here from the docs' expressions, NOT from
#       diagnostics.cmhd, which is rho-only (see the note on that test).
#   (g) THE POSITIVITY DISCRIMINATOR -- the motivating gate. One deep-rarefaction config
#       where the rho form provably reaches rho <= 0 and NaNs, and the identical-physics
#       lnrho run stays finite with min e^s > 0.
#   (h) lnrho x EXPANSION: s'(k=0) and B'(k=0) bitwise, u_perp(0) ~ a^-1, and the raw
#       backgrounds tracked through grad's OWN a^2 e^-s' unscaling.
#   (i) PLUMBING: density_var round-trips through params.json, a differing record is a hard
#       save error (which is what blocks a cross-mode restart), invalid values raise, and an
#       lnrho restart is bitwise.
#
# Plus the cross-variable RHS gate, which is the one that reads every new term at once:
# the two variable forms must compute the SAME physical time derivatives, to a residual
# that falls with amplitude.
#
# Single process by construction (CMHD is z_spectral, which is size==1 only). pytest, or
# `python tests/test_cmhd_lnrho.py` -- never under mpirun.
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


def _params(n, dt, lnrho=False, nx=None, ny=None, nz=None, eqpars=None, **kw):
    """A fresh CMHD Parameters. `lnrho=False` leaves density_var ABSENT, which is the
    default and the byte-identical path -- an explicit "rho" is a separate case, and gate
    (a) asserts the two agree bitwise rather than assuming it.

    dt is forced to a plain python float: a numpy scalar would make
    DiagonalOperator.scaled(dt) strong-typed float64 and poison the fp32 field math (the C1
    close-out note)."""
    ep = dict(cs0=1.0, diss=0.0, hyper=1) if eqpars is None else dict(eqpars)
    if lnrho:
        ep["density_var"] = "lnrho"
    return fresh_params(eqpars=ep, dt=float(dt), nx=nx or n, ny=ny or n, nz=nz or n,
                        **dict(_BOX, **kw))


def _advance(state, kgrid, params, nsteps, schemestr="lsrk54"):
    stepper, scheme = get_scheme(schemestr)
    return jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))(
        state, kgrid, params, nsteps, scheme, stepper)


def _a_of(t, adot):
    return 1.0 + adot*float(t)


def _mass(state, params):
    """int e^s over the box, as a volume average -- the conserved functional in this
    variable. Promoted to float64 for the reduction, so the number the gate quotes is not
    itself fp32-noisy."""
    s = np.asarray(grids.ifft(state.fields[0], params), dtype=np.float64)
    return float(np.mean(np.exp(s)))


def _div_b(fields, params, kgrid):
    """max_k |k.B^| / (|k| max|B^|), the C1 metric verbatim (tests/test_cmhd_conservation
    .py::_div_b) so the numbers here are directly comparable to the rho-form twin's."""
    f = np.asarray(fields)
    kx, ky = np.asarray(kgrid.kx), np.asarray(kgrid.ky)
    kz = np.asarray(kgrid.kz).copy()
    kz[params.nz//2] = 0.0                       # cmhd._kz_deriv's rule
    d = np.abs(kx*f[4] + ky*f[5] + kz*f[6])
    kmag = np.broadcast_to(np.sqrt(kx**2 + ky**2 + kz**2), d.shape)
    bmag = np.sqrt(np.abs(f[4])**2 + np.abs(f[5])**2 + np.abs(f[6])**2)
    m = kmag > 0
    return float(np.max(d[m]/kmag[m])/bmag.max())


# ------------------------------------------------------------------- initial conditions
#
# Every IC is built in REAL space (never written into k-space), so the rfftn reality
# constraint holds by construction. The rho-form and lnrho-form builders below take the SAME
# physical (u, B, rho) and differ ONLY in field 0, which is what makes the cross-variable
# gate a comparison of two discretizations of one physical state.

def _random_parts(ms=0.3, db=0.3, drho=0.1, seed=7, nmode=6):
    """The C1 band-limited random state (six modes, 1 <= |n|^2 <= 4), returned as its
    (u, B, delta_rho) pieces so both variable forms can be built from one draw. Each B mode
    is perpendicular to its own k, so div B starts at round-off."""
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

    def parts(x, y, z):
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
        return [su*u[i] for i in range(3)], [sb*B[0], sb*B[1], sb*B[2] + _B0], sr*r
    return parts


def _rho_ic(parts):
    def ic(x, y, z):
        u, B, r = parts(x, y, z)
        return jnp.stack([1.0 + r] + u + B)
    return ic


def _lnrho_ic(parts):
    """The SAME physical state with s = ln rho in field 0 -- the convention `initialize`'s
    user function must follow under density_var="lnrho" (cmhd module docstring)."""
    def ic(x, y, z):
        u, B, r = parts(x, y, z)
        return jnp.stack([jnp.log(1.0 + r)] + u + B)
    return ic


def _uniform_ic(s0=np.log(1.3), u=(0.2, 0.3, -0.15), B=(0.8, 0.3, -0.2)):
    """A spatially UNIFORM lnrho state. It stays uniform exactly -- every product is
    uniform, so every k != 0 RHS mode is an exact fp zero -- which is what makes the
    s^(k=0) claim a bitwise one."""
    def ic(x, y, z):
        o = jnp.ones(jnp.broadcast_shapes(x.shape, y.shape, z.shape))
        return jnp.stack([s0*o, u[0]*o, u[1]*o, u[2]*o, B[0]*o, B[1]*o, B[2]*o])
    return ic


# ------------------------------------------ (a) the "rho" path is untouched by the switch

def test_the_rho_path_is_unchanged_by_the_density_var_switch():
    """THE STANDING rho-path GATE. Three statements, none of which needs a stored reference:
    density_var ABSENT and density_var="rho" give a bitwise identical RHS and a bitwise
    identical 20-step run (so the default really is the pre-C4 path and not a re-derived
    equivalent); two independent compilations of one config agree bitwise (so a bitwise
    match reported anywhere in this file is a property of the code, not of one lucky
    compile); and grads carries NO lnrho members on that path -- `s`, `gs` and `inv_rho` are
    all None, i.e. not one exponential or extra transform is in that graph.

    The lnrho twin is the discriminator throughout: its RHS differs at O(1) of the RHS
    scale (field 0 is a different variable), and its grads carry `s`/`gs`/`inv_rho` and a
    None `rho` -- the density array is deliberately never formed on that path."""
    rhs = jax.jit(construct_rhs(equation_registry["CMHD"]), static_argnums=(2,))
    base = dict(cs0=0.8, diss=0.01, hyper=1)
    parts = _random_parts()
    with checks() as c:
        # absent vs explicit "rho"
        outs = []
        for label, ep in (("absent", dict(base)), ("explicit 'rho'",
                                                   dict(base, density_var="rho"))):
            p = _params(8, 0.01, eqpars=ep)
            kgrid = jr.setup_kgrids(p)
            s0 = jr.initialize(_rho_ic(parts), p)
            r, g = rhs(s0, kgrid, p)
            end = _advance(s0, kgrid, p, 20)
            outs.append((np.asarray(r), np.asarray(end.fields), float(end.t), g))
            c.check(f"density_var {label}: grads.s / grads.gs / grads.inv_rho are all None "
                    f"(no lnrho machinery in this graph)",
                    g.s is None and g.gs is None and g.inv_rho is None,
                    f"{type(g.s)} {type(g.gs)} {type(g.inv_rho)}")
            c.check(f"density_var {label}: grads.rho IS the density array",
                    g.rho is not None, repr(type(g.rho)))
        c.check("density_var absent and density_var='rho' give a bitwise identical RHS",
                np.array_equal(outs[0][0], outs[1][0]),
                f"max |delta| {np.abs(outs[0][0] - outs[1][0]).max():.3e}")
        c.check("... and a bitwise identical 20-step run",
                np.array_equal(outs[0][1], outs[1][1]) and outs[0][2] == outs[1][2],
                f"max |delta| {np.abs(outs[0][1] - outs[1][1]).max():.3e}")

        # the lnrho discriminator: same physical IC, a different graph and a different RHS
        p = _params(8, 0.01, lnrho=True, eqpars=base)
        kgrid = jr.setup_kgrids(p)
        sl = jr.initialize(_lnrho_ic(parts), p)
        rl, gl = rhs(sl, kgrid, p)
        rl = np.asarray(rl)
        scale = np.abs(outs[0][0]).max()
        c.check(f"lnrho: the RHS differs from the rho path by "
                f"{np.abs(rl - outs[0][0]).max()/scale:.2e} of the RHS scale (the checks "
                f"above are not vacuous)",
                np.abs(rl - outs[0][0]).max() > 1e-3*scale,
                f"max |delta| {np.abs(rl - outs[0][0]).max():.3e}, scale {scale:.3e}")
        c.check("lnrho: grads carries s, grad s and 1/rho, and NO rho array",
                gl.s is not None and gl.gs is not None and gl.inv_rho is not None
                and gl.rho is None and gl.rho_p is None,
                f"s={type(gl.s)} gs={type(gl.gs)} inv_rho={type(gl.inv_rho)} "
                f"rho={type(gl.rho)}")
        c.check("lnrho: grad s has one real-space component per direction",
                np.asarray(gl.gs).shape == (3, p.nz, p.nx, p.ny),
                f"{np.asarray(gl.gs).shape}")

        # two compilations of one lnrho config
        ends = []
        for _ in range(2):
            q = _params(8, 0.02, lnrho=True, eqpars=dict(cs0=0.9, diss=0.01, hyper=1))
            kg = jr.setup_kgrids(q)
            e = _advance(jr.initialize(_lnrho_ic(parts), q), kg, q, 10)
            ends.append((np.asarray(e.fields).copy(), float(e.t)))
        c.check("two compilations of one lnrho config agree bitwise",
                np.array_equal(ends[0][0], ends[1][0]) and ends[0][1] == ends[1][1],
                f"max |delta| {np.abs(ends[0][0] - ends[1][0]).max():.3e}")


# ------------------------------------------------------------------- (b) dispersion
#
# Analytic linear theory, transcribed from docs/numerics.md "Compressible MHD" -> "Linear
# waves" (linearized about rho = 1, B = B0 zhat, k = (kx, 0, kz)). Deliberately re-derived
# here rather than imported from tests/test_cmhd_linear.py: importing another test module
# would couple two bootstrap()s, and the point of this cell is that the SAME analytic
# omega(k) comes out of a different discretization. In the lnrho variable the linearized
# field-0 perturbation is delta_s = delta_rho exactly, since s = ln(1 + delta_rho) =
# delta_rho + O(delta_rho^2) about s0 = ln 1 = 0 -- so the eigenvectors are reused verbatim.

_ANGLES = {"parallel": (0, 0, 1), "oblique": (1, 0, 2)}
_EPS = 1e-6 if _fp64() else 1e-3
_W_TOL = 1e-6 if _fp64() else 2e-4
_AMP_TOL = 1e-6 if _fp64() else 1e-2


def _omega_ms(kx, kz, cs, vA, fast):
    k2 = kx*kx + kz*kz
    s = cs*cs + vA*vA
    disc = max(1.0 - 4.0*cs*cs*vA*vA*(kz*kz/k2)/(s*s), 0.0)
    w2_fast = 0.5*k2*s*(1.0 + np.sqrt(disc))
    if fast:
        return np.sqrt(w2_fast)
    # slow root from the product of the quartic's roots: cancellation-free, exactly 0 at
    # kpar = 0
    return np.sqrt(cs*cs*vA*vA*k2*kz*kz/w2_fast)


def _eigenmode(branch, kx, kz, cs, vA):
    """(omega, v) with v = (delta_s, du_x, du_y, du_z, dB_x, dB_y, dB_z), all real."""
    if branch == "alfven":
        # du along yhat, delta_rho = 0 (hence delta_s = 0), omega^2 = kpar^2 vA^2, and the
        # docs' unambiguous dB_y = -(kpar B0/omega) du_y, which at omega = +kpar vA and
        # vA = B0 is exactly -du_y
        return np.sqrt(kz*kz*vA*vA), np.array([0.0, 0, 1, 0, 0, -1.0, 0])
    w = _omega_ms(kx, kz, cs, vA, branch == "fast")
    dux = 1.0
    duz = cs*cs*kx*kz/(w*w - cs*cs*kz*kz)*dux
    drho = (kx*dux + kz*duz)/w
    return w, np.array([drho, dux, 0.0, duz, -kz*_B0*dux/w, 0.0, kx*_B0*dux/w])


def _eigen_ic(v, k, eps):
    """The uniform background plus eps*v*cos(k.x), with FIELD 0 CARRYING s: the background
    is s0 = ln(rho0) = ln 1 = 0, and the perturbation is eps*delta_s*cos(k.x)."""
    base = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _B0)
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])

    def ic(x, y, z):
        shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
        ph = jnp.broadcast_to(jnp.cos(kx*x + ky*y + kz*z), shp)
        return jnp.stack([base[i] + eps*float(v[i])*ph for i in range(7)])
    return ic


def _measure(branch, idx, cs0, gamma, block=20, target_phase=6.0, wdt=0.05):
    """(omega_analytic, relative frequency error, relative amplitude variation) for one
    eigenmode driven through the lnrho RHS."""
    k = np.array([float(idx[0]), float(idx[1]), float(idx[2])])       # 2*pi box: k = n
    w, v = _eigenmode(branch, float(k[0]), float(k[2]), cs0, _B0)
    dt = wdt/_omega_ms(float(k[0]), float(k[2]), cs0, _B0, True)
    p = _params(8, dt, lnrho=True,
                eqpars=dict(cs0=cs0, diss=0.0, hyper=1, gamma=gamma))
    kgrid = jr.setup_kgrids(p)
    state = jr.initialize(_eigen_ic(v, k, _EPS), p)
    sl = (slice(None), idx[2] % p.nz, idx[0] % p.nx, idx[1])
    ts, ps = [float(state.t)], [complex(np.dot(v, np.asarray(state.fields[sl]))
                                        / np.dot(v, v))]
    nsamp = int(np.ceil(target_phase/(w*float(dt)*block)))
    for _ in range(nsamp):
        state = _advance(state, kgrid, p, block)
        ts.append(float(state.t))
        ps.append(complex(np.dot(v, np.asarray(state.fields[sl]))/np.dot(v, v)))
    ts, ps = np.array(ts), np.array(ps)
    amp = np.abs(ps)
    w_meas = -float(np.polyfit(ts, np.unwrap(np.angle(ps)), 1)[0])
    return w, abs(w_meas/w - 1.0), float(np.max(np.abs(amp/amp[0] - 1.0)))


def test_dispersion_through_the_lnrho_path():
    """The three branches come out of the lnrho RHS at the analytic omega(k) -- and, exactly
    as in the rho form, this measures the PRODUCTION RHS: there is no background/fluctuation
    split, so the waves emerge from the quadratic terms acting on the k = 0 modes.

    A sign error in either new term is O(1) here: -div u is the whole of the acoustic
    coupling (drop it and delta_s stops responding to compression), and -u.grad s carries
    the advection of the background gradient. The Lorentz e^-s, by contrast, is INVISIBLE to
    this gate -- it multiplies a first-order force by 1 + O(eps) -- which is why the
    cross-variable and budget gates below exist.

    One config per branch, as the plan's C4 list asks: parallel Alfven, oblique fast, oblique
    slow, at gamma = 1, plus the oblique fast/slow pair at gamma = 5/3 (the Alfven branch is
    gamma-independent). Tolerances are C1's verbatim; measured 2026-08-30 <= 2.1e-8 (fp64)
    and <= 6.2e-6 (fp32) relative in omega, amplitude drift <= 4.3e-9 / 4.0e-6. The fp32
    figures are an order better than C1's rho-form twin, for a structural reason worth
    recording: the lnrho background is s0 = ln 1 = 0, so an eps-sized field-0 perturbation
    costs no quantization against a background of 1 the way delta_rho does."""
    cells = [("alfven", "parallel", 1.0), ("fast", "oblique", 1.0),
             ("slow", "oblique", 1.0), ("fast", "oblique", 5.0/3.0),
             ("slow", "oblique", 5.0/3.0)]
    with checks() as c:
        for branch, angle, gamma in cells:
            idx = _ANGLES[angle]
            w, err, ampvar = _measure(branch, idx, 0.5, gamma)
            label = f"lnrho gamma={gamma:.3f} {branch} {angle} n={idx}"
            c.check(f"{label}: omega = {w:.6f} measured to {err:.2e}", err < _W_TOL,
                    f"relative error {err:.3e} > {_W_TOL}")
            c.check(f"{label}: no spurious growth at nu = 0 ({ampvar:.2e})",
                    ampvar < _AMP_TOL, f"amplitude varies by {ampvar:.3e}")


# --------------------------------------------------------- (c) exact dissipation decay

_DECAY_DISS = (0.05, 0.1, 0.2)      # (D_rho, nu, eta): three distinct coefficients
_DECAY_K = 2.0
_DECAY_TOL = 1e-13 if _fp64() else 2e-5
_LEAK_TOL = 1e-13 if _fp64() else 1e-6
_DECAY_AXES = {"x": 0, "y": 1, "z": 2}
_DECAY_IDX = {"x": (int(_DECAY_K), 0, 0), "y": (0, int(_DECAY_K), 0),
              "z": (0, 0, int(_DECAY_K))}


def _s_decay_ic(axis):
    icoord = _DECAY_AXES[axis]

    def ic(x, y, z):
        shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
        arg = _DECAY_K*(x, y, z)[icoord]
        return jnp.stack([0.1*jnp.broadcast_to(jnp.cos(arg), shp)]
                         + [jnp.zeros(shp) for _ in range(6)])
    return ic


def test_dissipation_only_decay_of_s_modes_is_exact():
    """u = B = 0 and a single s mode, at cs0 = 1e-8 so the pressure force it drives is
    O(cs0^2) = 1e-16 and the sound response stays under the tolerance. The ideal s-RHS is
    then identically zero (-u.grad s and -div u both vanish with u), so the IF step reduces
    to exp(L dt) and the mode must decay by exactly exp(-D_rho k^(2h) t) -- the field-0 row
    of the SAME diagonal L, now acting on s, which is the whole of `linear_matrix`'s
    density_var story.

    ALL THREE AXES, per the C1 review's lesson: a z-only state has k_perp = 0, so
    L = -diss*(ksq + kz^2)^hyper reduces to the kz term and deleting kgrid.ksq would leave
    the gate green. The length-3 (D_rho, nu, eta) diss also pins the 7-field expansion --
    a decay at nu or eta instead of D_rho is the wrong number here.

    Measured 2026-08-30: relative error <= 2.1e-15 (fp64) and <= 3.4e-6 (fp32) over 40
    steps, with the u/B leak at <= 1.3e-16 in BOTH precisions (it is the O(cs0^2) pressure
    response, not a rounding effect)."""
    nsteps, dt = 40, 0.02
    with checks() as c:
        for hyper in (1, 2):
            p = _params(8, dt, lnrho=True,
                        eqpars=dict(cs0=1e-8, diss=_DECAY_DISS, hyper=hyper, gamma=1.0))
            kgrid = jr.setup_kgrids(p)
            for axis in ("x", "y", "z"):
                idx = _DECAY_IDX[axis]
                sl = (slice(None), idx[2], idx[0], idx[1])
                state = jr.initialize(_s_decay_ic(axis), p)
                c0 = np.asarray(state.fields[sl])
                end = _advance(state, kgrid, p, nsteps)
                c1 = np.asarray(end.fields[sl])
                want = np.exp(-_DECAY_DISS[0]*(_DECAY_K**2)**hyper*nsteps*dt)
                err = abs(c1[0]/c0[0] - want)/want
                leak = max(abs(c1[i]) for i in range(1, 7))/abs(c0[0])
                c.check(f"hyper={hyper} s along {axis}: decays by exp(-D_rho k^{2*hyper} t) "
                        f"= {want:.4e} (relative error {err:.2e})", err < _DECAY_TOL,
                        f"{err:.3e}")
                c.check(f"hyper={hyper} s along {axis}: u and B stay at zero ({leak:.2e})",
                        leak < _LEAK_TOL, f"{leak:.3e}")


# ----------------------------------------------------------------------- set_timestep
#
# ADDED AFTER MUTATION TESTING (2026-08-30). Every other gate in this file runs at fixed dt,
# so set_timestep's lnrho branch was reached by nothing: the mutation "gamma > 1 uses the
# constant cs0^2 instead of cs0^2 e^((gamma-1)s)" passed all 14 tests at both precisions.
# This cell is the fix, and it is the C1 test_set_timestep_matches_the_cfl_bound pattern
# rebuilt on e^s.

def _cfl_ic(x, y, z):
    # every factor carries its own phase offset, so s is NOT 0 at the grid point where the
    # CFL speed peaks -- otherwise c_s(e^s) = cs0 there and the gamma branch is vacuous
    shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
    a = jnp.broadcast_to(jnp.cos(x + 0.37)*jnp.cos(y + 0.11)*jnp.cos(z + 0.53), shp)
    b = jnp.broadcast_to(jnp.sin(2*x)*jnp.cos(y + 1.1)*jnp.sin(z), shp)
    d = jnp.broadcast_to(jnp.cos(x)*jnp.sin(y)*jnp.cos(2*z + 0.7), shp)
    return jnp.stack([jnp.log(1.0 + 0.3*a), 0.3*b, 0.3*d, 0.2*a, 0.3*d, 0.3*a,
                      1.0 + 0.3*b])


def _cfl_reference(state, params, cs0, gamma):
    """The same bound, rebuilt from the real-space fields in host numpy -- with rho = e^s
    formed HERE, so the code's e^-s / e^((gamma-1)s) forms are compared against an
    independent construction rather than against themselves."""
    f = np.asarray(grids.ifft(state.fields, params), dtype=np.float64)
    s, u, B = f[0], f[1:4], f[4:7]
    rho = np.exp(s)
    cs2 = cs0**2 if gamma == 1.0 else cs0**2*np.exp((gamma - 1.0)*s)
    cf = np.sqrt(cs2 + (B**2).sum(0)/rho)
    d = (params.dx, params.dy, params.dz)
    return params.cfl_safety/max((np.abs(u[i]) + cf).max()/d[i] for i in range(3))


def _cfl_reference_ebm(state, params, cs0, a, cs_q):
    """The expanding-box bound in host numpy. Three things are different from above and all
    three are the point: the physical spacings are (dx, a dy, a dz); field 0 is s', so the
    PHYSICAL rho is a^-2 e^s'; and fields[4:7] are B', so the physical B is
    (B'_x/a^2, B'_y/a, B'_z/a). The sound speed is the cooled cs0^2 a^-cs_q (EBM is
    gamma = 1)."""
    f = np.asarray(grids.ifft(state.fields, params), dtype=np.float64)
    sp, u, Bp = f[0], f[1:4], f[4:7]
    rho = np.exp(sp)/a**2
    B = np.stack([Bp[0]/a**2, Bp[1]/a, Bp[2]/a])
    cf = np.sqrt(cs0**2*a**(-cs_q) + (B**2).sum(0)/rho)
    d = (params.dx, params.dy*a, params.dz*a)
    return params.cfl_safety/max((np.abs(u[i]) + cf).max()/d[i] for i in range(3))


def test_set_timestep_on_the_lnrho_path_matches_the_cfl_bound():
    """set_timestep is the one recipe function every fixed-dt gate misses, and on this path
    it carries two lnrho-specific expressions: c_s^2 = cs0^2 e^((gamma-1)s) and
    v_A^2 = |B|^2 e^-s (the precomputed grads.inv_rho). Both are checked against an
    independent numpy rebuild that forms rho = e^s itself.

    The gamma cells are asserted to give DIFFERENT dt, which is what stops the gamma branch
    from passing vacuously -- the mutation that motivated this test (gamma > 1 falling back
    to the constant cs0^2) is exactly that vacuity.

    The expansion cell is included because inv_rho carries the extra a^2 there, so the
    physical v_A the CFL sees is a^2 e^-s'|B|^2 and the spacings are (dx, a dy, a dz)."""
    tol = 1e-12 if _fp64() else 1e-5
    st = jax.jit(lambda s, kg, p: cmhd.set_timestep(cmhd.grad(s, kg, p), p),
                 static_argnums=(2,))
    seen = {}
    with checks() as c:
        for gamma in (1.0, 5.0/3.0):
            for cs0 in (0.5, 2.0):
                p = _params(8, 0.01, lnrho=True, adaptive_timestep=True,
                            eqpars=dict(cs0=cs0, diss=0.0, hyper=1, gamma=gamma))
                kgrid = jr.setup_kgrids(p)
                state = jr.initialize(_cfl_ic, p)
                got = float(st(state, kgrid, p))
                want = _cfl_reference(state, p, cs0, gamma)
                err = abs(got/want - 1.0)
                c.check(f"lnrho gamma={gamma:.3f} cs0={cs0}: dt = {got:.6e} matches the CFL "
                        f"bound rebuilt on rho = e^s ({err:.2e})", err < tol,
                        f"{got} vs {want}")
                seen[(gamma, cs0)] = got
        for cs0 in (0.5, 2.0):
            a, b = seen[(1.0, cs0)], seen[(5.0/3.0, cs0)]
            c.check(f"cs0={cs0}: the gamma branch is not vacuous -- gamma=1 and gamma=5/3 "
                    f"give different dt ({a:.6e} vs {b:.6e})", a != b)
        # expansion: physical spacings, the unprimed B, and the a^2 inside inv_rho
        for cs_q in (0.0, 4.0/3.0):
            adot = 0.3
            p = _params(8, 0.01, lnrho=True, adaptive_timestep=True,
                        eqpars=dict(cs0=1.0, diss=0.0, hyper=1,
                                    expansion=dict(adot=adot, cs_q=cs_q)))
            kgrid = jr.setup_kgrids(p)
            state = jr.initialize(_cfl_ic, p)._replace(t=jnp.float64(2.0))
            aa = _a_of(2.0, adot)
            got = float(st(state, kgrid, p))
            want = _cfl_reference_ebm(state, p, 1.0, aa, cs_q)
            err = abs(got/want - 1.0)
            c.check(f"lnrho + expansion at a = {aa:.2f}, cs_q = {cs_q:.3f}: dt = "
                    f"{got:.6e} matches the bound rebuilt with physical spacings, the "
                    f"unprimed B and rho = a^-2 e^s' ({err:.2e})", err < tol,
                    f"{got} vs {want}")


# ------------------------------------------------------------------- (d) mass int e^s

# The mass-drift cell. n = 16, half the C1 conservation amplitude and a fixed t = 4 at three
# dt: measured 2026-08-30 (fp64) 4.986e-06 -> 2.201e-07 -> 2.641e-08, order 3.780. At the
# C1 amplitude and dt the drift is instead the dt-INDEPENDENT non-polynomial truncation
# residual (3.648e-06 flat over dt 0.02/0.01/0.005), exactly as the C1 energy gate found --
# so this cell is deliberately placed where the TIME error dominates: larger dt, smaller
# amplitude. That is what makes the order meaningful.
_MASS_DTS = (0.16, 0.08, 0.04)
_MASS_T = 4.0
_MASS_AMP = 0.5


@pytest.mark.fp64
def test_mass_drift_is_order_consistent_and_small():
    """MASS CHANGES DISCRETE CLASS, and this is the gate that says so honestly. In
    rho-variables mass is rho^(k=0), whose RHS projection is an exact fp zero -- bitwise for
    free. int e^s is nonlinear in the evolved field and has no linear substitute, so RK
    preserves it only to scheme order (docs/numerics.md). The assertion is therefore drift
    ORDER plus absolute smallness, and NEVER round-off -- the gate also asserts the drift is
    comfortably ABOVE round-off, so it cannot pass by measuring nothing.

    Asserted on the SAME run, because the variable switch must not have touched induction:
    every mean-B component still BITWISE invariant, and div B still in the eps*sqrt(N)
    class. Field 0 never enters the induction equation, so anything else would be a bug.

    fp64 only: the drift at the finest dt is 2.6e-8 on a mass of order 1, an order of
    magnitude below the fp32 storage floor of the difference, and no order survives there.
    The mean-B/div-B halves are re-run at both precisions in the next test."""
    if not _fp64():
        print("[SKIP] test_mass_drift_is_order_consistent_and_small -- fp64 only")
        return
    parts = _random_parts(ms=0.3*_MASS_AMP, db=0.3*_MASS_AMP, drho=0.1*_MASS_AMP)
    drifts, first = [], None
    for dt in _MASS_DTS:
        p = _params(16, dt, lnrho=True)
        kgrid = jr.setup_kgrids(p)
        st = jr.initialize(_lnrho_ic(parts), p)
        m0, b0, d0 = _mass(st, p), np.asarray(st.fields[4:7, 0, 0, 0]).copy(), \
            _div_b(st.fields, p, kgrid)
        end = _advance(st, kgrid, p, int(round(_MASS_T/dt)))
        drifts.append(abs(_mass(end, p) - m0)/m0)
        if first is None:
            first = (b0, np.asarray(end.fields[4:7, 0, 0, 0]), d0,
                     _div_b(end.fields, p, kgrid))
    order = fit_order(_MASS_DTS, drifts)
    eps = np.finfo(np.float64).eps
    b0, b1, d0, d1 = first
    with checks() as c:
        c.check(f"the mass drift converges at order {order:.3f} over dt {_MASS_DTS} "
                f"(lsrk54 is 4th order; drifts {['%.3e' % v for v in drifts]})",
                3.0 < order < 5.0, f"{order:.4f}")
        c.check(f"... and is small in absolute terms ({drifts[0]:.2e} relative at the "
                f"coarsest dt)", drifts[0] < 1e-4, f"{drifts[0]:.3e}")
        c.check(f"... and is NOT round-off -- the finest-dt drift {drifts[-1]:.2e} is still "
                f"{drifts[-1]/eps:.0f} eps, so an order was actually measured",
                drifts[-1] > 1e4*eps, f"{drifts[-1]:.3e}")
        c.check("mean B is still BITWISE invariant on the same run (induction never sees "
                "field 0)", bool(np.all(b0 == b1)),
                f"max |delta| {np.abs(b1 - b0).max():.3e}")
        c.check(f"div B is still a round-off random walk on the same run "
                f"({d0:.2e} -> {d1:.2e})", d1 < 20.0*eps*np.sqrt(5*int(_MASS_T/_MASS_DTS[0])),
                f"{d1:.3e}")


def test_mean_b_and_div_b_are_unchanged_by_the_variable_switch():
    """The induction half of gate (d), at BOTH precisions and against its rho-form twin: the
    same physical IC, run in both variables, must give the same bitwise mean B and the same
    class of div B. Field 0 is not an input to `d_t B = curl(u x B)`, so this is a
    structural statement -- and one worth pinning, since it is the property the variable
    change is NOT allowed to cost (only mass was)."""
    parts = _random_parts()
    eps = np.finfo(np.float64 if _fp64() else np.float32).eps
    bound = 20.0*eps*np.sqrt(250)                # 5 stages/step * 50 steps, 20x margin
    with checks() as c:
        for label, lnrho, ic in (("rho", False, _rho_ic(parts)),
                                 ("lnrho", True, _lnrho_ic(parts))):
            p = _params(16, 0.02, lnrho=lnrho)
            kgrid = jr.setup_kgrids(p)
            st = jr.initialize(ic, p)
            b0, d0 = np.asarray(st.fields[4:7, 0, 0, 0]).copy(), _div_b(st.fields, p, kgrid)
            end = _advance(st, kgrid, p, 50)
            b1, d1 = np.asarray(end.fields[4:7, 0, 0, 0]), _div_b(end.fields, p, kgrid)
            c.check(f"{label}: every mean-B component is bitwise invariant over 50 steps",
                    bool(np.all(b0 == b1)), f"max |delta| {np.abs(b1 - b0).max():.3e}")
            c.check(f"{label}: div B stays in the eps*sqrt(N) class "
                    f"({d0:.2e} -> {d1:.2e} < {bound:.2e})", d1 < bound, f"{d1:.3e}")
            c.check(f"{label}: u(k=0) DOES move (the bitwise check above is not vacuous)",
                    bool(np.any(np.asarray(st.fields[1:4, 0, 0, 0])
                                != np.asarray(end.fields[1:4, 0, 0, 0]))))


# ---------------------------------------------------------------- (e) the uniform state

def test_uniform_state_s_at_k_zero_is_bitwise():
    """docs/numerics.md's exact-fp-zero argument, asserted: on a spatially uniform state
    grad~s and grad~.u are identically zero, so BOTH terms of d_t s are exact fp zeros, the
    dealias mask is 1 at k = 0 and exp(L(0) dt) = 1 exactly. So s^(k=0) is a bitwise
    invariant there even though int e^s is not one in general -- and the whole state stays
    exactly uniform, which is asserted too (every k != 0 mode still exactly zero).

    The discriminator is the turbulent twin: <s> DOES drift there. That is the physics, not
    a bug -- d<s>/dt = <s div u>, and Jensen pushes <s> ~ -sigma_s^2/2 downward as
    fluctuations grow at fixed mass -- and it is what says the bitwise claim above is a
    statement about the uniform state and not about a dead RHS."""
    p = _params(8, 0.02, lnrho=True)
    kgrid = jr.setup_kgrids(p)
    st = jr.initialize(_uniform_ic(), p)
    f0 = np.asarray(st.fields).copy()
    end = _advance(st, kgrid, p, 100)
    f1 = np.asarray(end.fields)
    off = f1.copy()
    off[:, 0, 0, 0] = 0.0
    with checks() as c:
        c.check("s^(k=0) is bitwise invariant over 100 uniform-state steps",
                f0[0, 0, 0, 0] == f1[0, 0, 0, 0], f"{f0[0,0,0,0]!r} -> {f1[0,0,0,0]!r}")
        c.check("the state stays EXACTLY uniform (every k != 0 mode identically zero)",
                bool(np.all(off == 0.0)), f"max |k!=0| {np.abs(off).max():.3e}")
        c.check("every B(k=0) component is bitwise invariant too",
                bool(np.all(f0[4:7, 0, 0, 0] == f1[4:7, 0, 0, 0])),
                f"max |delta| {np.abs(f1[4:7,0,0,0] - f0[4:7,0,0,0]).max():.3e}")
        # the turbulent discriminator
        q = _params(16, 0.02, lnrho=True)
        kg = jr.setup_kgrids(q)
        t0 = jr.initialize(_lnrho_ic(_random_parts()), q)
        s0 = complex(np.asarray(t0.fields[0, 0, 0, 0]))
        t1 = _advance(t0, kg, q, 50)
        s1 = complex(np.asarray(t1.fields[0, 0, 0, 0]))
        nrm = q.nx*q.ny*q.nz               # grids.fft is unnormalized: s^(0)/N is <s>
        c.check(f"<s> DOES drift on a turbulent state ({s0.real/nrm:.6e} -> "
                f"{s1.real/nrm:.6e}, and downward as Jensen requires) -- int e^s is the "
                f"invariant, not <s>", s0 != s1, f"{s0!r} -> {s1!r}")


# ----------------------------------------------------------------- (f) the energy budget
#
# Built here from docs/numerics.md's expressions and NOT from taranis/diagnostics/cmhd.py,
# which is rho-ONLY: it reads state.fields[0] as rho, so on an lnrho state its energies and
# budget are wrong rather than differently normalized. Making it density_var-aware is
# outside this phase's file list and is carried as plans/CMHD_PLAN.md §11.
#
#     E       = < e^s |u|^2/2 + |B|^2/2 + rho_e(s) >
#     rho_e   = cs^2 e^s s                      gamma = 1
#             = cs0^2 e^(gamma s)/(gamma(gamma-1))   gamma > 1
#     dE/du   = e^s u      dE/dB = B
#     dE/ds   = e^s(|u|^2/2 + h + cs^2)         gamma = 1   <-- THE INVERSION TRAP
#             = e^s(|u|^2/2 + h)                gamma > 1
#     eps     = -< (dE/du).(D_u u) + (dE/dB).(D_B B) + (dE/ds)(D_s s) >
#
# The closure tolerance. NOT round-off and it must not be tightened into one: CMHD's ideal
# RHS conserves E only to the O(dt^p)-plus-non-polynomial-aliasing residual
# (plans/CMHD_PLAN.md §3.5), and the lnrho form has MORE non-polynomial content in E itself
# (e^s and e^s s), not less. Measured 2026-08-30 at 16^3, amp 0.5 of the C1 regime:
# 7.4e-9 (gamma=1) and 9.7e-9 (gamma=5/3) at diss=1e-3, 7.2e-9 / 9.9e-9 at diss=1e-2 --
# i.e. a fixed absolute floor, exactly as the aliasing story predicts. The rho-form gate
# sits at the same 1e-5 with a comparable measured figure.
_BUDGET_TOL = 1e-5


def _lnrho_energy(state, params, cs0, gamma):
    s = np.asarray(grids.ifft(state.fields[0], params), dtype=np.float64)
    f = np.asarray(grids.ifft(state.fields[1:7], params), dtype=np.float64)
    u, B = f[0:3], f[3:6]
    rho = np.exp(s)
    rho_e = (cs0**2*rho*s if gamma == 1.0
             else cs0**2/(gamma*(gamma - 1.0))*np.exp(gamma*s))
    return float(np.mean(0.5*rho*(u*u).sum(0) + 0.5*(B*B).sum(0) + rho_e))


def _lnrho_sink(state, params, kgrid, cs0, gamma, with_cs2=True):
    """(eps, the +cs^2 piece of it). D_f f = ifft(L_row * field), exactly as the rho-form
    budget does it."""
    L = np.asarray(cmhd.linear_matrix(kgrid, params))
    fk = np.asarray(state.fields)

    def diss(i):
        row = L[0] if L.shape[0] == 1 else L[i]
        return np.asarray(grids.ifft(jnp.asarray(row*fk[i]), params), dtype=np.float64)

    s = np.asarray(grids.ifft(state.fields[0], params), dtype=np.float64)
    f = np.asarray(grids.ifft(state.fields[1:7], params), dtype=np.float64)
    u, B = f[0:3], f[3:6]
    rho = np.exp(s)
    d_s = diss(0)
    d_u = np.stack([diss(1), diss(2), diss(3)])
    d_B = np.stack([diss(4), diss(5), diss(6)])
    h = cs0**2*s if gamma == 1.0 else cs0**2/(gamma - 1.0)*np.exp((gamma - 1.0)*s)
    const = cs0**2 if (gamma == 1.0 and with_cs2) else 0.0
    dEds = rho*(0.5*(u*u).sum(0) + h + const)
    eps = -(np.mean(rho*(u*d_u).sum(0)) + np.mean((B*d_B).sum(0)) + np.mean(dEds*d_s))
    cs2_piece = -np.mean(rho*cs0**2*d_s) if gamma == 1.0 else 0.0
    return float(eps), float(cs2_piece)


def _closure(gamma, amp=0.5, diss=1e-3, hyper=1, dt=2e-4, n=16, cs0=1.0):
    """(measured dE/dt, eps, the +cs^2 piece) -- centered difference over two steps against
    the sink at the midpoint state, the tests/test_cmhd_diagnostics.py pattern."""
    p = _params(n, dt, lnrho=True, eqpars=dict(cs0=cs0, diss=diss, hyper=hyper,
                                               gamma=gamma))
    kgrid = jr.setup_kgrids(p)
    stepper, scheme = get_scheme("lsrk54")
    parts = _random_parts(ms=0.3*amp, db=0.3*amp, drho=0.1*amp)
    s0 = jr.initialize(_lnrho_ic(parts), p)
    s1 = block_of_steps(s0, kgrid, p, 1, scheme, stepper)
    s2 = block_of_steps(s1, kgrid, p, 1, scheme, stepper)
    dt_actual = float(s2.t - s0.t)/2.0
    E0, E2 = _lnrho_energy(s0, p, cs0, gamma), _lnrho_energy(s2, p, cs0, gamma)
    eps, cs2_piece = _lnrho_sink(s1, p, kgrid, cs0, gamma)
    return (E2 - E0)/(2*dt_actual), eps, cs2_piece


@pytest.mark.fp64
def test_energy_budget_closes_at_both_gamma():
    """dE/dt + eps ~ 0 in the ln rho variable, at gamma = 1 and gamma = 5/3.

    THIS GATE IS WHAT SEES THE LORENTZ e^-s. The dispersion gate above cannot: e^-s
    multiplies an already-first-order force, so a wrong 1/rho there is second order in the
    perturbation. Here it is not -- get the Lorentz factor wrong and the ideal RHS stops
    conserving E, which shows up as a closure failure far outside this tolerance.

    NOT a round-off gate and it must not be tightened into one -- see the _BUDGET_TOL note
    above for the measured figures and why. fp64 only, for the same reason the rho-form
    closure gate is: dE over two steps is far below the fp32 noise floor of the
    difference."""
    if not _fp64():
        print("[SKIP] test_energy_budget_closes_at_both_gamma -- fp64 only")
        return
    with checks() as c:
        for gamma in (1.0, 5.0/3.0):
            for diss in (1e-3, 1e-2):
                meas, eps, _ = _closure(gamma, diss=diss)
                rel = abs(meas + eps)/max(abs(eps), 1e-30)
                tag = f"lnrho gamma={gamma:.3f} diss={diss:g}"
                c.check(f"{tag}: dE/dt = -eps (rel {rel:.2e}, eps {eps:.6e})",
                        rel < _BUDGET_TOL, f"measured {meas:.10e}, -eps {-eps:.10e}")
                c.check(f"{tag}: eps > 0 for diss > 0", eps > 0.0, f"eps={eps:.10e}")
        # cmhd._enthalpy_s's gamma = 1 branch is NOT reached from NonlinearTerm (there the
        # force is k-local, which is the transform the 24-tally does not spend), so pin both
        # branches here against the inline h this file's sink uses. Without this the gamma=1
        # branch would be untested code sitting next to a production one.
        sv = np.linspace(-0.7, 0.7, 11)
        for gamma in (1.0, 5.0/3.0):
            want = 1.21*sv if gamma == 1.0 else 1.21/(gamma - 1.0)*np.exp((gamma - 1.0)*sv)
            got = np.asarray(cmhd._enthalpy_s(jnp.asarray(sv, dtype=jnp.float64), 1.1,
                                              gamma), dtype=np.float64)
            err = float(np.max(np.abs(got - want))/np.max(np.abs(want)))
            c.check(f"cmhd._enthalpy_s matches h(e^s) at gamma={gamma:.3f} ({err:.2e})",
                    err < 1e-14, f"{err:.3e}")


@pytest.mark.fp64
def test_the_gamma1_budget_needs_the_plus_cs2_term():
    """THE DOCUMENTED INVERSION TRAP, and its discriminator. In the rho form the gamma = 1
    strict derivative d(rho e)/drho = cs0^2(ln rho + 1) exceeds h by the constant cs0^2, and
    that constant multiplies <D_rho rho>, whose only surviving mode would be k = 0 where
    L(0) = 0 exactly -- so it provably drops out and the rho-form budget uses h alone.

    IT DOES NOT DROP OUT HERE. In s-variables the same constant multiplies <e^s D_s s>,
    whose mean is NOT zero (e^s is not constant), so delta E/delta s = e^s(|u|^2/2 + h +
    cs^2) and the +cs^2 term is load-bearing. Copying the rho-form note across is the likely
    first bug of this phase, so this test rebuilds the sink WITHOUT it and asserts the
    closure fails -- at the measured scale, quoted in the check line.

    The density amplitude is deliberately 2x the closure gate's (drho = 0.1, not 0.05): the
    dropped term's weight is the SPREAD of e^s, so it grows with drho. Measured 2026-08-30
    at 16^3, drho = 0.05 / 0.1 / 0.15 / 0.25: the term carries 4.1% / 13.3% / 22.7% / 36.3%
    of eps and breaking it moves the closure to 4.3e-2 / 1.5e-1 / 2.9e-1 / 5.7e-1, while the
    intact closure holds at 7.4e-9 / 1.2e-7 / 3.0e-6 / 2.0e-4. drho = 0.1 is the point where
    the discriminator is 15x clear of its threshold and the intact closure is still 80x
    inside _BUDGET_TOL."""
    if not _fp64():
        print("[SKIP] test_the_gamma1_budget_needs_the_plus_cs2_term -- fp64 only")
        return
    p = _params(16, 2e-4, lnrho=True, eqpars=dict(cs0=1.0, diss=1e-3, hyper=1, gamma=1.0))
    kgrid = jr.setup_kgrids(p)
    stepper, scheme = get_scheme("lsrk54")
    parts = _random_parts(ms=0.15, db=0.15, drho=0.10)
    s0 = jr.initialize(_lnrho_ic(parts), p)
    s1 = block_of_steps(s0, kgrid, p, 1, scheme, stepper)
    s2 = block_of_steps(s1, kgrid, p, 1, scheme, stepper)
    meas = (_lnrho_energy(s2, p, 1.0, 1.0) - _lnrho_energy(s0, p, 1.0, 1.0)) \
        / (2*float(s2.t - s0.t)/2.0)
    eps, cs2_piece = _lnrho_sink(s1, p, kgrid, 1.0, 1.0)
    eps_no, _ = _lnrho_sink(s1, p, kgrid, 1.0, 1.0, with_cs2=False)
    with checks() as c:
        rel = abs(meas + eps)/abs(eps)
        rel_no = abs(meas + eps_no)/abs(eps_no)
        c.check(f"with the +cs^2 term the gamma = 1 budget closes (rel {rel:.2e})",
                rel < _BUDGET_TOL, f"{rel:.3e}")
        c.check(f"WITHOUT it the closure fails by {rel_no:.2e} -- the term carries "
                f"{cs2_piece/eps:.1%} of eps and does NOT drop out the way the rho form's "
                f"constant does", rel_no > 100*max(rel, 1e-12) and rel_no > 1e-2,
                f"rel without {rel_no:.3e} vs with {rel:.3e}")


# ------------------------------------------------------- (g) THE POSITIVITY DISCRIMINATOR

# A deep rarefaction: a single-mode, strongly supersonic converging/diverging flow along x
# on a cheap 3D box (the y and z axes only have to exist -- the IC is x-only and stays so).
# cs0 = 0.1 against u = 2 is M_s = 20, the flow evacuates the diverging half exponentially,
# and rho reaches zero on a timescale ~1/max(du/dx). ny = nz = 4 keeps the run cheap.
_POS = dict(nx=32, ny=4, nz=4, cs0=0.1, uamp=2.0, drho=0.6, dt=0.005, nsteps=150)


def _rarefaction_ic(lnrho):
    def ic(x, y, z):
        shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
        o = jnp.ones(shp)
        rho = 1.0 + _POS["drho"]*jnp.broadcast_to(jnp.cos(x), shp)
        ux = _POS["uamp"]*jnp.broadcast_to(jnp.sin(x), shp)
        f0 = jnp.log(rho) if lnrho else rho
        return jnp.stack([f0, ux, 0.0*o, 0.0*o, 0.0*o, 0.0*o, 0.0*o])
    return ic


def test_lnrho_survives_a_rarefaction_that_kills_the_rho_form():
    """THE MOTIVATING GATE. Identical physics, identical numerics, one eqpars key apart.

    The rho form evolves rho in flux form, which conserves mass exactly but guarantees
    NOTHING about positivity: in a deep rarefaction the spectral representation of a
    collapsing rho rings, undershoots through zero, and then log(rho) is NaN -- the loud
    failure plans/CMHD_PLAN.md §3.2 and §7 deliberately keep (never abs() or clip it). The
    lnrho form cannot do that: rho = e^s > 0 is STRUCTURAL, so the same run stays finite
    with min e^s comfortably positive.

    The gate asserts BOTH halves, and the rho-form half loudly -- a positivity gate whose
    control silently survives is measuring nothing. Config and measurements, 2026-08-30:
    32x4x4, cs0 = 0.1, u = 2 sin(x) (M_s = 20), rho = 1 + 0.6 cos(x), nu = 0, dt = 0.005,
    150 steps. rho form: min rho 0.802 -> 0.574 -> NaN by step 150. lnrho form: min e^s
    0.802 -> 0.685 -> 3.5e-2, finite everywhere, and still falling like the exponential the
    physics asks for (1.0e-7 at 200 steps, 3.0e-30 at 300). Tuned to be robust, not
    marginal: the rho form is already NaN a third of the way before this gate looks."""
    nst = _POS["nsteps"]
    out = {}
    for label, lnrho in (("rho", False), ("lnrho", True)):
        p = _params(None, _POS["dt"], lnrho=lnrho, nx=_POS["nx"], ny=_POS["ny"],
                    nz=_POS["nz"], eqpars=dict(cs0=_POS["cs0"], diss=0.0, hyper=1))
        kgrid = jr.setup_kgrids(p)
        st = jr.initialize(_rarefaction_ic(lnrho), p)
        trace = []
        for _ in range(3):
            st = _advance(st, kgrid, p, nst//3)
            f0 = np.asarray(grids.ifft(st.fields[0], p), dtype=np.float64)
            rho = np.exp(f0) if lnrho else f0
            trace.append((float(np.nanmin(rho)) if np.any(np.isfinite(rho)) else np.nan,
                          bool(np.all(np.isfinite(np.asarray(st.fields))))))
        out[label] = trace
    r_min, r_fin = zip(*out["rho"])
    l_min, l_fin = zip(*out["lnrho"])
    with checks() as c:
        c.check(f"the rho form REALLY fails on this rarefaction within {nst} steps "
                f"(min rho {['%.3e' % v for v in r_min]}, all-finite "
                f"{list(r_fin)}) -- if this ever passes, the gate below is vacuous",
                (not r_fin[-1]) or np.isnan(r_min[-1]) or r_min[-1] <= 0.0,
                f"min rho {r_min[-1]!r}, finite {r_fin[-1]!r}")
        c.check(f"the lnrho form stays finite through the same run (min e^s "
                f"{['%.3e' % v for v in l_min]})", all(l_fin),
                f"all-finite {list(l_fin)}")
        c.check(f"... with min e^s = {l_min[-1]:.3e} > 0, structurally (rho = e^s cannot "
                f"be <= 0)", l_min[-1] > 0.0, f"{l_min[-1]!r}")
        c.check("... and the density really did collapse -- this is a deep rarefaction, not "
                f"a quiet run (min e^s fell {1.0/l_min[-1]:.1f}x below 1)",
                l_min[-1] < 0.2, f"{l_min[-1]!r}")


# ------------------------------------------------------------- (h) lnrho x expansion

def test_lnrho_composes_with_the_expanding_box():
    """EBM in s-variables. The evolved field is s' = ln rho' = s + 2 ln a, and the -2(adot/a)
    expansion term of the raw continuity CANCELS IDENTICALLY against d(2 ln a)/dt -- so
    lnrho continuity carries NO expansion term at all (docs/numerics.md). On a uniform state
    that makes s'(k=0) a BITWISE invariant, which is simultaneously the discriminator for a
    spurious expansion term: add one and s' would decay visibly.

    The rest is the C3b uniform-state pair driven through the lnrho path: B'(k=0) bitwise,
    u_x(0) bitwise (T_x = 0), u_perp(0) ~ a^-1 from du/dt = -(adot/a)u, and the RAW
    backgrounds through grad's own unscaling -- rho ~ a^-2 read off `grads.inv_rho`, which
    is where the a^2 e^-s' factor lives and the one place a wrong exponent on this path
    would hide."""
    adot, dt, nsteps = 0.3, 0.02, 100
    p = _params(8, dt, lnrho=True,
                eqpars=dict(cs0=1.0, diss=0.0, hyper=1, expansion=dict(adot=adot)))
    kgrid = jr.setup_kgrids(p)
    grad = jax.jit(cmhd.grad, static_argnums=(2,))
    st = jr.initialize(_uniform_ic(), p)
    m0 = np.asarray(st.fields[:, 0, 0, 0]).copy()
    g0 = grad(st, kgrid, p)
    rho0 = float(np.mean(1.0/np.asarray(g0.inv_rho, dtype=np.float64)))
    B0m = np.asarray(g0.B).mean(axis=(1, 2, 3))
    end = _advance(st, kgrid, p, nsteps)
    m1 = np.asarray(end.fields[:, 0, 0, 0])
    g1 = grad(end, kgrid, p)
    rho1 = float(np.mean(1.0/np.asarray(g1.inv_rho, dtype=np.float64)))
    B1m = np.asarray(g1.B).mean(axis=(1, 2, 3))
    a = _a_of(end.t, adot)
    tol = 1e-9 if _fp64() else 1e-5
    btol = 1e-12 if _fp64() else 1e-5
    with checks() as c:
        c.check(f"a({float(end.t):.3f}) = {a:.4f}: s'(k=0) is BITWISE invariant -- lnrho "
                f"continuity carries no expansion term (a spurious one would decay it)",
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
        rel = abs(rho1/rho0/a**-2 - 1.0)
        c.check(f"the RAW rho = 1/grads.inv_rho follows a^-2 (rel {rel:.2e}) -- this is the "
                f"a^2 e^-s' unscaling, read through grad's own code", rel < btol,
                f"{rho1/rho0!r} vs {a**-2!r}")
        for i, ex, nm in ((0, -2.0, "B_x"), (1, -1.0, "B_y"), (2, -1.0, "B_z")):
            rel = abs(B1m[i]/B0m[i]/a**ex - 1.0)
            c.check(f"the raw {nm} follows a^{ex:.0f} (rel {rel:.2e})", rel < btol,
                    f"{B1m[i]/B0m[i]!r} vs {a**ex!r}")


# ---------------------------------------------- the cross-variable RHS gate (the net)

def _physical_rhs(state, kgrid, params, lnrho):
    """One RHS evaluation converted to PHYSICAL time derivatives: (d_t rho, d_t u, d_t B) in
    real space. In the rho form field 0's row IS d_t rho; in the lnrho form it is d_t s, and
    d_t rho = rho d_t s exactly. u and B rows are the same variables in both.

    Under expansion every row here is the PRIMED quantity (d_t rho' from e^s' d_t s', and
    d_t B'), which is what both forms evolve -- the a-factors relating them to the raw frame
    are common to the two and cancel out of the comparison, so nothing is lost."""
    r, _ = construct_rhs(equation_registry["CMHD"])(state, kgrid, params)
    f0 = np.asarray(grids.ifft(state.fields[0], params), dtype=np.float64)
    d0 = np.asarray(grids.ifft(r[0], params), dtype=np.float64)
    drho = d0*np.exp(f0) if lnrho else d0
    rest = np.asarray(grids.ifft(r[1:7], params), dtype=np.float64)
    return np.concatenate([drho[None], rest])


@pytest.mark.fp64
def test_the_two_variable_forms_compute_the_same_physical_rhs():
    """THE CROSS-VARIABLE GATE, and the one that reads every new term at once. The rho form
    and the lnrho form are two discretizations of ONE set of equations, so on the same
    physical state they must compute the same physical d_t rho, d_t u and d_t B. Drop the
    div u term, flip the sign of u.grad s, or drop the e^-s on the Lorentz force and the two
    disagree at O(1) of the row -- these are exactly the mutations the plan names.

    WHY THIS IS NOT A ROUND-OFF COMPARISON, and why the gate is an amplitude SCALING rather
    than a fixed tolerance: the two states are not the same object. `initialize` dealiases
    both, and ln(1 + r) is not polynomial, so the masked s and the masked rho are not
    exp/log of each other -- they differ by the non-polynomial truncation residual that
    plans/CMHD_PLAN.md §3.5 accepts and never hides. That residual vanishes with amplitude
    (measured ~amp^3.2 on the density row) while a wrong TERM would not vanish at all. So
    the assertion is: small at the working amplitude, and falling by at least 4x per halving
    of it.

    Both expansion modes are run: under EBM the new terms are built on the comoving k~, and
    the rho-form twin carries its own independently-written k~ in the flux and the enthalpy
    gradient, so a metric factor dropped from only one of them shows up here.

    Measured 2026-08-30, expansion off, 16^3, amplitudes 1 / 0.5 / 0.25 of the C1
    conservation regime: d_t rho 1.40e-4, 1.49e-5, 1.69e-6; d_t u and d_t B rows
    <= 2.35e-6, 1.15e-7, 6.38e-9. fp64 only -- at fp32 the amp/4 residual is under the
    storage noise floor and no scaling survives."""
    if not _fp64():
        print("[SKIP] test_the_two_variable_forms_compute_the_same_physical_rhs -- fp64")
        return
    names = ("rho", "u_x", "u_y", "u_z", "B_x", "B_y", "B_z")
    with checks() as c:
        for label, extra in (("expansion off", {}),
                             ("expansion on", dict(expansion=dict(adot=0.25)))):
            worst = []
            for amp in (1.0, 0.5, 0.25):
                parts = _random_parts(ms=0.3*amp, db=0.3*amp, drho=0.1*amp)
                got = {}
                for key, lnrho, ic in (("rho", False, _rho_ic(parts)),
                                       ("lnrho", True, _lnrho_ic(parts))):
                    p = _params(16, 0.01, lnrho=lnrho,
                                eqpars=dict(cs0=1.0, diss=0.0, hyper=1, **extra))
                    kgrid = jr.setup_kgrids(p)
                    st = jr.initialize(ic, p)
                    if extra:
                        st = st._replace(t=jnp.float64(2.0))     # a = 1.5, away from 1
                    got[key] = _physical_rhs(st, kgrid, p, lnrho)
                rels = [float(np.abs(got["lnrho"][i] - got["rho"][i]).max()
                              / np.abs(got["rho"][i]).max()) for i in range(7)]
                worst.append(max(rels))
                c.check(f"{label}, amp {amp:.2f}: the two forms agree on every physical row "
                        f"(worst {names[int(np.argmax(rels))]} at {max(rels):.2e})",
                        max(rels) < 1e-2, f"{['%.2e' % v for v in rels]}")
            for i in range(len(worst) - 1):
                c.check(f"{label}: halving the amplitude shrinks the residual "
                        f"{worst[i]:.2e} -> {worst[i+1]:.2e} (x{worst[i]/worst[i+1]:.1f} "
                        f"> 4) -- it is the non-polynomial truncation residual, not a term "
                        f"error", worst[i]/worst[i+1] > 4.0,
                        f"{worst[i]:.3e} -> {worst[i+1]:.3e}")


# --------------------------------------------------------------------- (i) plumbing

_PLUMB = dict(cs0=1.3, diss=(0.01, 0.02, 0.03), hyper=2, gamma=5.0/3.0,
              density_var="lnrho")


def test_density_var_round_trips_and_blocks_a_cross_mode_restart():
    """density_var is a plain-JSON eqpars string, so it round-trips like every other entry;
    params.save's differing-record check is then the ONLY thing stopping a cross-mode
    restart, exactly as for z_spectral and for the expansion block. That matters more here
    than usual: field 0 changes MEANING between the two modes while keeping its shape and
    dtype, so a snapshot restored under the wrong density_var would run silently and be
    wrong (rho read as s, or s exponentiated twice)."""
    params = fresh_params(eqpars=dict(_PLUMB), dt=0.01, nx=8, ny=8, nz=8, **_BOX)
    with snap_dir("cmhd_lnrho_params_") as d, checks() as c:
        params.save(d)
        back = jr.Parameters.from_snapshot(d)
        c.check(f"eqpars round-trip: {back.eqpars!r}", back.eqpars == _PLUMB,
                f"{back.eqpars!r} != {_PLUMB!r}")
        c.check("cmhd._density_var reads it back", cmhd._density_var(back) == "lnrho",
                repr(cmhd._density_var(back)))
        params.save(d)          # identical re-save is a no-op
        c.check("identical re-save is a no-op", True)
        for label, ep in (("switching to 'rho'", dict(_PLUMB, density_var="rho")),
                          ("dropping density_var entirely",
                           {k: v for k, v in _PLUMB.items() if k != "density_var"})):
            with pytest.raises(ValueError, match="eqpars"):
                fresh_params(eqpars=ep, dt=0.01, nx=8, ny=8, nz=8, **_BOX).save(d)
            c.check(f"saving with {label} over the record is a hard error", True)


def test_invalid_density_var_values_raise():
    """Everything the schema rejects, at setup_kgrids time (config.py stays
    equation-agnostic, so these are cmhd._density_var errors reached through _eqpars)."""
    def raises(c, label, match, value):
        eqpars = dict(cs0=1.0, diss=0.0, hyper=1, density_var=value)
        try:
            jr.setup_kgrids(fresh_params(eqpars=eqpars, dt=0.01, nx=8, ny=8, nz=8, **_BOX))
            raised = ""
        except (ValueError, NotImplementedError) as exc:
            raised = str(exc)
        c.check(f"{label} raises, naming {match!r}", match in raised, f"raised: {raised!r}")

    with checks() as c:
        raises(c, "a misspelling", "density_var", "lnrho ")
        raises(c, "a plausible alternative name", "density_var", "log_rho")
        raises(c, "an empty string", "density_var", "")
        raises(c, "a non-string", "density_var", 1)
        raises(c, "None (absent is how you turn it off, not None)", "density_var", None)
        for value in ("rho", "lnrho"):
            c.check(f"density_var={value!r} is accepted",
                    jr.setup_kgrids(fresh_params(
                        eqpars=dict(cs0=1.0, diss=0.0, hyper=1, density_var=value),
                        dt=0.01, nx=8, ny=8, nz=8, **_BOX)) is not None)
        c.check("lnrho composes with gamma > 1",
                jr.setup_kgrids(fresh_params(
                    eqpars=dict(cs0=1.0, diss=0.0, hyper=1, gamma=5.0/3.0,
                                density_var="lnrho"), dt=0.01, nx=8, ny=8, nz=8,
                    **_BOX)) is not None)
        c.check("lnrho composes with the expanding box",
                jr.setup_kgrids(fresh_params(
                    eqpars=dict(cs0=1.0, diss=0.0, hyper=1, density_var="lnrho",
                                expansion=dict(adot=0.1)), dt=0.01, nx=8, ny=8, nz=8,
                    **_BOX)) is not None)


def test_lnrho_restart_is_bitwise():
    """Fixed dt, nu > 0: an lnrho run stopped, checkpointed and resumed reproduces the
    uninterrupted trajectory bitwise (fields and t). The on-disk layout is unchanged -- only
    field 0's meaning is -- which is why the params.json check above is the whole of the
    cross-mode protection."""
    p = _params(8, 0.02, lnrho=True, eqpars=dict(cs0=1.0, diss=0.01, hyper=1))
    kgrid = jr.setup_kgrids(p)
    ic = _lnrho_ic(_random_parts())
    straight = _advance(jr.initialize(ic, p), kgrid, p, 20)
    ref_fields, ref_t = np.asarray(straight.fields).copy(), float(straight.t)

    mid = _advance(jr.initialize(ic, p), kgrid, p, 10)
    with snap_dir("cmhd_lnrho_restart_") as d:
        with managed_manager(p, d, nsnap=2) as mngr:
            snapshot_io.save_snapshot(0, mid, mngr, p)
            mngr.wait_until_finished()
        reloaded = snapshot_io.load_snapshot(0, d, p)
    resumed = _advance(reloaded, kgrid, p, 10)
    with checks() as c:
        c.check("the reloaded state is bitwise the checkpointed one",
                np.array_equal(np.asarray(reloaded.fields), np.asarray(mid.fields))
                and float(reloaded.t) == float(mid.t))
        c.check(f"the resumed run ends at the same t ({float(resumed.t)!r})",
                float(resumed.t) == ref_t, f"{float(resumed.t)!r} vs {ref_t!r}")
        c.check("the resumed run is bitwise the uninterrupted one",
                np.array_equal(np.asarray(resumed.fields), ref_fields),
                f"max |delta| "
                f"{np.abs(np.asarray(resumed.fields) - ref_fields).max():.3e}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
