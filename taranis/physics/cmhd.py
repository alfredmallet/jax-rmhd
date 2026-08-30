# Compressible (polytropic) MHD, fully spectral.
#
# Derivation of record: docs/numerics.md, section "Compressible MHD" (Phase C0). Plan and
# scope decisions: plans/CMHD_PLAN.md. Every expression below is a transcription of that
# docs section -- change the docs first, then this file.
#
# Seven fields, in state order (rho, u_x, u_y, u_z, B_x, B_y, B_z). Alfven units
# (B_code = B/sqrt(mu0*rho0), so v_A = |B|/sqrt(rho)), <rho> = rho0 = 1 by convention,
# p = K*rho^gamma with K = cs0^2/gamma so that c_s(rho=1) = cs0.
#
#     d_t rho = -div(rho u)                                     flux form
#     d_t u   = -(curl u) x u - grad(|u|^2/2 + h(rho)) + (curl B) x B / rho
#     d_t B   = curl(u x B)                                     curl form
#
#     h(rho) = cs0^2 * rho^(gamma-1)/(gamma-1)   gamma > 1
#     h(rho) = cs0^2 * ln(rho)                   gamma = 1  (isothermal, the default)
#
# There is NO background/fluctuation split and no linear wave term: B0 and rho0 live in the
# k = 0 modes of the evolved fields, and the waves come out of the quadratic terms. Both
# k = 0 modes are preserved BITWISE by the discrete step (docs/numerics.md), which is what
# the C1 mass / mean-B gates assert.
#
# Three standing rules this module is built on:
#
#  1. L IS DISSIPATION ONLY; THE WAVES STAY EXPLICIT. `linear_matrix` returns the isotropic
#     (hyper-)dissipation diagonal and nothing else. The compressible wave operator couples
#     all seven fields per mode -- there is no 7x7 propagator backend, and IMEX-ing the waves
#     would put an L-stable solve on a wave-dominated L, which the house rule forbids. Never
#     move a wave term into L.
#  2. ANY FUTURE INDUCTION TERM MUST BE A CURL (or k-locally divergence-free). div B is not
#     cleaned, projected or evolved: `i*k x fft(u x B)` makes `k . dB` a pairwise-cancelling
#     sum, so div B only random-walks at machine epsilon, and the propagator and the dealias
#     mask scale all three B components identically (they commute with `k .`). A
#     non-curl induction term would break that for good.
#  3. THE NON-POLYNOMIAL RESIDUAL IS ACCEPTED, NOT HIDDEN. The quadratic products are exactly
#     dealiased by the existing 2/3 mask; h(rho), ln(rho) and 1/rho are not polynomial and no
#     finite padding dealiases them. At smooth rho and M_s <~ 1 the residual sits inside the
#     time-discretization error, which is why the conservation gates assert convergence ORDER
#     plus absolute smallness and NEVER round-off.
#
# Scope (enforced in _check_supported): dims == 3 with z_spectral=True, single process, no
# forcing, no particles. This module holds only what the solver consumes; read-only CMHD
# observers belong in taranis/diagnostics/cmhd.py (Phase C2).
#
# EXPANDING BOX (EBM, Phase C3b; docs/numerics.md "Expanding box"). Optional
# eqpars["expansion"] = {"adot": > 0, "cs_q": >= 0, default 4/3}, isothermal (gamma == 1)
# only. Comoving frame of Squire, Chandran & Meyrand 2020: a(t) = 1 + adot*t, radial axis x,
# transverse (y, z) expanding, comoving gradient grad~ = (d_x, d_y/a, d_z/a). The EVOLVED
# state is then the RESCALED (primed) one,
#
#     rho' = a^2 rho     B' = (a^2 B_x, a B_y, a B_z)     u unscaled
#
# which kills the -2(adot/a)rho and -(adot/a)diag(2,1,1)B expansion terms identically and
# keeps induction a STATIC-k curl of E' = ((u x B)_x, a(u x B)_y, a(u x B)_z). What is left:
#
#     d_t rho' = -grad~ . (rho' u)
#     d_t u    = (the ideal RHS with grad~, h = cs^2(t) ln rho') - (adot/a) diag(0,1,1) u
#     d_t B'   = curl E'                                        (static integer-grid k)
#
# with cs^2(t) = cs0^2 a^(-cs_q). Consequences inherited verbatim from the non-expanding
# argument, and gated in tests/test_cmhd_expansion.py: K.B'^ is a round-off random walk, and
# the k = 0 modes of rho' and B' are BITWISE invariants (so the raw backgrounds track
# rho ~ a^-2, B_x ~ a^-2, B_perp ~ a^-1 exactly).
#
# THE OFF PATH IS BITWISE THE PRE-C3 GRAPH. Everything above is gated on
# `_expansion(params) is None`, plain python at TRACE time (params is static), so not one
# a-factor enters the graph of a non-expanding run -- never a lax.cond, never a multiply by
# a literal 1.0. `test_expansion_off_rhs_does_not_depend_on_t` is the standing gate.
#
# Dissipation is UNCHANGED by expansion: L stays the static diagonal
# -diss_f*(kperp^2 + kz^2)^hyper acting on the PRIMED fields at COMOVING k, which is the
# recorded truncation choice (docs; the physical-wavenumber alternative would need nu(t) or
# per-block L rebuilds). linear_matrix therefore has no expansion branch at all.
#
# THE ln rho DENSITY VARIABLE (Phase C4; docs/numerics.md "The ln rho density variable").
# Optional eqpars["density_var"] in {"rho" (default), "lnrho"}. With "lnrho", FIELD 0 IS
# s = ln rho instead of rho -- an opt-in variable change for POSITIVITY: rho = e^s > 0 is
# structural, so the rarefaction NaN channel the rho form carries is retired, and at the
# transonic target s is the smooth near-Gaussian field where rho has decade-deep valleys.
#
#     d_t s = -u.grad s - div u                     (the flux form divided by rho)
#
#     h = cs^2 * s                     gamma = 1   LINEAR: the pressure force is the
#                                                  k-local -cs^2 * i k~ s^, no transform
#     h = cs0^2 e^((gamma-1)s)/(gamma-1)  gamma > 1  (pointwise, joins the scalar transform)
#
# with 1/rho = e^-s the only non-polynomial factor left at gamma = 1. WHAT THIS COSTS, and
# it is a real cost, not a wash: MASS CHANGES DISCRETE CLASS. In rho-variables mass is the
# single coefficient rho^(k=0) whose RHS projection is an exact fp zero, i.e. bitwise for
# free; the s-form's conserved functional int e^s is NONLINEAR in the evolved field and has
# no linear substitute (<s> is not conserved even in the continuum), so RK preserves it only
# to scheme order. Mass therefore joins the class energy is already in -- gated by drift
# ORDER plus absolute smallness, NEVER at round-off. mean B and div B are untouched
# (induction never sees field 0), and on a SPATIALLY UNIFORM state every s-RHS term is an
# exact fp zero, so s^(k=0) is bitwise there.
#
# EBM composes: with expansion on the evolved field is s' = ln rho' = s + 2 ln a, and the
# -2(adot/a) expansion term cancels identically against d(2 ln a)/dt, so continuity carries
# NO expansion term at all. The raw 1/rho is a^2 e^-s', and h's uniform -2 cs^2(t) ln a dies
# under the gradient exactly as in the rho form. Transform tally 24 (16 inverse / 8 forward),
# one more than the rho form's 23: grad gains grad~s (3) and s (1) = 16, and the term func
# loses (rho u)(3) and gains u.grad s (1) = 8. div u = i k~ . u^ is k-local.
#
# THE "rho" PATH MUST STAY BYTE-IDENTICAL. Like the expansion switch, density_var is a
# TRACE-TIME python branch on the static params (`_density_var(params) == "lnrho"`), never a
# lax.cond and never a select between two computed values -- verified once out of tree over
# three configs (one of them expansion-on) at both precisions, in fields, t and the
# optimized-HLO opcode histogram (plans/CMHD_PLAN.md §5 C4). linear_matrix has ZERO change:
# the field-0 row of the same diagonal L simply acts on s instead of rho.
#
# INITIALIZATION AND DIAGNOSTICS. run.initialize is equation-generic, so nothing there
# changes -- but the user IC function MUST RETURN s = ln rho IN FIELD 0 under "lnrho"
# (fields 1..6 are unchanged u and B). density_var rides eqpars into params.json, so
# params.save's differing-record check is what blocks a cross-mode restart. taranis/
# diagnostics/cmhd.py is rho-ONLY and is NOT density_var-aware: it reads state.fields[0] as
# rho, so on an lnrho state its energies/Mach numbers/spectra/budget are wrong, not merely
# differently normalized. Unscale by hand (rho = e^s) until a helper lands
# (plans/CMHD_PLAN.md §11).

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .. import comms, grids
from .. import _precision


class CMHDGrads(NamedTuple):
    # What one RHS evaluation needs in real space, computed once by grad() and shared by the
    # term func and by set_timestep. Not (d/dx, d/dy) pairs -- CMHD is not a bracket equation
    # set -- so this deliberately does NOT go through shared_physics.grad_fields.
    # the PHYSICAL density, real space (nz, nx, ny) -- and None on the density_var="lnrho"
    # path, where rho is DELIBERATELY never formed: e^s would be one more full-grid array
    # and nothing needs it (the Lorentz force and v_A^2 want 1/rho, h and c_s want s), and
    # not forming it is what keeps the positivity guarantee structural rather than a
    # convention. Read `s`/`inv_rho` instead, under the same trace-time density_var switch.
    rho: jnp.ndarray        # (nz, nx, ny)
    u: jnp.ndarray          # (3, nz, nx, ny)
    B: jnp.ndarray          # (3, nz, nx, ny)
    omega: jnp.ndarray      # (3, nz, nx, ny), curl u
    j: jnp.ndarray          # (3, nz, nx, ny), curl B
    # state.t rides along because set_timestep sees only what grad_func returned; the
    # non-expanding path ignores it and the expanding-box CFL reads it. Deliberately
    # float64 -- a(t) is cast to _precision.ftype before it touches field math.
    t: jnp.ndarray
    # EXPANSION ONLY, None otherwise: the PRIMED density rho' = a^2 rho, real space. rho
    # above is always the UNPRIMED (physical) one, so set_timestep and the Lorentz force
    # need no unscaling; rho' is what the continuity flux and h(rho') are built from
    # (docs/numerics.md "Expanding box"). None off, so the off graph carries no extra leaf.
    # Also None on the lnrho path, where there is no density array at all, primed or not.
    rho_p: jnp.ndarray = None
    # LNRHO ONLY (None on the rho path, the rho_p precedent): the EVOLVED field 0 in real
    # space -- s = ln rho, or s' = ln rho' with expansion on. h and c_s(gamma>1) are built
    # from it, and it is what `initialize`'s IC function must return in field 0.
    s: jnp.ndarray = None
    # LNRHO ONLY: grad~ s, (3, nz, nx, ny) real space -- the three extra inverse transforms
    # of the 24-transform tally. Under expansion these are grad~ s' on the comoving k~.
    gs: jnp.ndarray = None
    # LNRHO ONLY: the PHYSICAL 1/rho = e^-s (a^2 e^-s' under expansion, since rho = a^-2
    # rho'). Formed once here and shared by the Lorentz force and by set_timestep's
    # v_A^2 = |B|^2/rho, exactly as `rho` is shared on the rho path.
    inv_rho: jnp.ndarray = None


# --------------------------------------------------------------------------- validation

def _check_supported(params):
    if params.eqtype != "CMHD":
        raise ValueError(f"physics.cmhd was called with eqtype={params.eqtype!r}: these "
                         f"recipe functions are the CMHD equation set (plans/CMHD_PLAN.md)")
    if params.spatial_dimensions != 3 or not params.z_spectral:
        raise NotImplementedError(
            f"CMHD requires dims=3 with z_spectral=True (got dims="
            f"{params.spatial_dimensions}, z_spectral={params.z_spectral}): the equations are "
            f"fully spectral, so every d/dz is i*kz and there is no finite-difference-z or 2D "
            f"path (plans/CMHD_PLAN.md §1). Run 2D physics as 2.5D: dims=3, small nz, a "
            f"z-independent initial condition.")
    if params.size != 1:
        raise NotImplementedError(
            f"CMHD is single-process only (z_spectral needs the whole z domain on one rank), "
            f"but this process is one of {params.size} (plans/CMHD_PLAN.md §1)")
    if params.forcing:
        raise NotImplementedError(
            "CMHD has no forcing in the MVP: compressible O-U forcing needs its own power "
            "normalization derivation (plans/CMHD_PLAN.md §8). Use forcing=False.")


_EQPARS_REQUIRED = ("cs0", "diss", "hyper")
_EQPARS_OPTIONAL = ("gamma", "expansion", "density_var")

# eqpars["density_var"] (docs/numerics.md "The ln rho density variable")
_DENSITY_VARS = ("rho", "lnrho")

# eqpars["expansion"] schema (docs/numerics.md "Expanding box")
_EXPANSION_KEYS = ("adot", "cs_q")
_CS_Q_DEFAULT = 4.0/3.0             # Squire et al.'s Athena++ "adiabatic" mimic


def _expansion(params):
    """The expanding-box configuration as (adot, cs_q), or None when expansion is off.

    THE ONE TRACE-TIME SWITCH for the whole EBM path. `params` is static, so every caller
    writes `if _expansion(params) is None:` as plain python and an off run's graph never
    sees an a-factor -- which is what makes the off path BITWISE the pre-C3 one. Never
    lax.cond this, and never express "off" as a multiply by a literal 1.0.
    """
    ep = params.eqpars
    if "expansion" not in ep:
        return None
    exp = ep["expansion"]
    if not isinstance(exp, dict):
        raise ValueError(f"eqpars['expansion'] must be a dict like "
                         f"{{'adot': 0.1, 'cs_q': {_CS_Q_DEFAULT}}} (or absent, for a "
                         f"non-expanding run), got {exp!r}")
    unknown = [k for k in exp if k not in _EXPANSION_KEYS]
    if unknown or "adot" not in exp:
        raise ValueError(f"eqpars['expansion'] problem (unknown {unknown}, "
                         f"{'adot missing' if 'adot' not in exp else 'adot present'}): the "
                         f"schema is {{'adot': float > 0, 'cs_q': float >= 0, default "
                         f"{_CS_Q_DEFAULT}}}; got {exp!r}")
    adot = float(exp["adot"])
    if not adot > 0.0:
        # a(t) = 1 + adot*t: adot > 0 is what keeps a(t) >= 1 for every t >= 0 a run can
        # reach, so the a^-1 / a^-2 unscalings and ln a are always finite. The one hole
        # this cannot close is a DOCTORED restart at t < -1/adot (a negative t in a
        # snapshot) -- unreachable from initialize/simulate, and nothing here can validate
        # it, since t is a traced value at every use site.
        raise ValueError(f"eqpars['expansion']['adot'] is the expansion rate in "
                         f"a(t) = 1 + adot*t and must be > 0 (adot <= 0 would let a(t) "
                         f"reach 0 and the a^-1 unscalings blow up), got {exp['adot']!r}")
    cs_q = float(exp.get("cs_q", _CS_Q_DEFAULT))
    if cs_q < 0.0:
        raise ValueError(f"eqpars['expansion']['cs_q'] is the cooling exponent in "
                         f"cs^2(t) = cs0^2*a^(-cs_q) and must be >= 0 (cs_q = 0 is constant "
                         f"temperature), got {exp['cs_q']!r}")
    gamma = float(ep.get("gamma", 1.0))
    if gamma != 1.0:
        # scope, not mathematics: EBM preserves D_t(p/rho^gamma) = 0 exactly, so a
        # barotropic gamma > 1 would drop in with h = cs0^2 a^(-2(gamma-1)) rho'^(gamma-1).
        # Deferred -- Squire et al.'s closure is isothermal and pinning to it is the whole
        # justification for the EBM terms here (docs/numerics.md "Expanding box").
        raise ValueError(f"eqpars['expansion'] requires the isothermal closure gamma = 1 "
                         f"(EBM is isothermal-only: the cooling law cs^2(t) = cs0^2*a^-q IS "
                         f"the closure), got gamma={gamma!r}")
    return adot, cs_q


def _density_var(params):
    """"rho" (the default) or "lnrho" -- THE ONE TRACE-TIME SWITCH for the whole ln rho path.

    `params` is static, so every caller writes `if _density_var(params) == "lnrho":` as
    plain python and the "rho" graph is byte-identical to the pre-C4 one: not a lax.cond,
    not a select between two computed branches, and never an `s = log(rho)` round trip.
    """
    dv = params.eqpars.get("density_var", "rho")
    if dv not in _DENSITY_VARS:
        raise ValueError(f"eqpars['density_var'] selects the evolved density variable and "
                         f"must be one of {_DENSITY_VARS} ('rho' evolves rho in field 0, "
                         f"'lnrho' evolves s = ln rho there and is positivity-structural; "
                         f"docs/numerics.md 'The ln rho density variable'), got {dv!r}")
    return dv


def _a_of(t, adot):
    # a(t) = 1 + adot*t, evaluated in float64 (CMHDGrads.t / state.t are float64 by
    # construction) and cast to the FIELD dtype HERE, before a ever multiplies a field: an
    # fp32 run must not be silently upcast by a strong-typed float64 scalar leaking into
    # the field graph (the params.dt trap recorded in the C1 close-out note).
    return jnp.asarray(1.0 + adot*t, dtype=_precision.ftype)


def _eqpars(params):
    # pull & validate the CMHD eqpars out of params.eqpars. Unknown keys are rejected.
    _check_supported(params)
    missing = [k for k in _EQPARS_REQUIRED if k not in params.eqpars]
    unknown = [k for k in params.eqpars if k not in _EQPARS_REQUIRED + _EQPARS_OPTIONAL]
    if missing or unknown:
        raise ValueError(f"CMHD eqpars problem (missing {missing}, unknown {unknown}): "
                         f"required {_EQPARS_REQUIRED}, optional {_EQPARS_OPTIONAL} "
                         f"(gamma defaults to 1.0, isothermal); got eqpars={params.eqpars!r}")
    ep = params.eqpars
    cs0 = float(ep["cs0"])
    if not cs0 > 0.0:
        raise ValueError(f"eqpars['cs0'] is the sound speed at rho=1 and must be > 0, got "
                         f"{ep['cs0']!r}")
    hyper = ep["hyper"]
    if isinstance(hyper, bool) or not isinstance(hyper, (int, np.integer)) or hyper < 1:
        raise ValueError(f"eqpars['hyper'] must be an int >= 1 (L = -diss*k^(2*hyper)), got "
                         f"{hyper!r}")
    gamma = float(ep.get("gamma", 1.0))
    if gamma < 1.0:
        raise ValueError(f"eqpars['gamma'] must be >= 1 (gamma=1 is the isothermal default; "
                         f"gamma<1 is not a physical polytrope and h(rho) would flip sign), "
                         f"got {ep['gamma']!r}")
    diss = ep["diss"]
    if np.shape(diss) not in ((), (1,), (3,)):
        raise ValueError(f"eqpars['diss'] must be a scalar (one coefficient for every field) "
                         f"or a length-3 (D_rho, nu, eta) sequence expanded over "
                         f"(rho, u, B), got {diss!r}")
    if np.any(np.asarray(diss, dtype=float) < 0.0):
        # a negative coefficient makes L = -diss*k^(2h) POSITIVE: exp(L*tau) then amplifies
        # the smallest scales exponentially, which is anti-dissipation, not a knob
        raise ValueError(f"eqpars['diss'] must be >= 0 (it is the coefficient of "
                         f"L = -diss*k^(2*hyper); a negative entry makes the propagator "
                         f"amplify the grid scale instead of damping it), got {diss!r}")
    # validate the optional expanding-box block here too, so EVERY entry point that reads
    # eqpars (including diagnostics) rejects a malformed one. The 4-tuple return is fixed by
    # taranis/diagnostics/cmhd.py's unpacking -- callers that need the EBM configuration
    # call _expansion(params) separately. _density_var is validated here for the same
    # reason -- note that taranis/diagnostics/cmhd.py, which is one of those callers, is
    # rho-ONLY: it will happily read an lnrho state's field 0 as rho and report nonsense
    # (module header; plans/CMHD_PLAN.md §11).
    _expansion(params)
    _density_var(params)
    return cs0, diss, int(hyper), gamma


def _diss_per_field(diss):
    # numpy float64 dissipation coefficients for the leading axis of L: a scalar stays length
    # 1 (propagators.DiagonalOperator broadcasts it over the 7 fields), a length-3
    # (D_rho, nu, eta) is expanded to the 7 fields in state order.
    d = np.asarray(diss, dtype=float).reshape(-1)
    if d.size == 1:
        return d
    return np.array([d[0], d[1], d[1], d[1], d[2], d[2], d[2]])


# ------------------------------------------------------------------ k-space helpers

def _kz_deriv(kgrid, params):
    # THE kz of this module: kgrid.kz with the Nyquist plane zeroed, so that i*kz respects
    # the rfftn reality constraint F(-kx,-kz,ky) = conj(F(kx,kz,ky)) on the ky = 0 / Nyquist
    # rows (docs/numerics.md "Spectral z"). Every d/dz in this file goes through it -- the
    # curls, the three divergences, and the even power inside L. For the even power it is
    # optional (kz^2 is reality-safe at the Nyquist plane), and the 2/3 kz cut removes that
    # plane from every IC and nonlinear path regardless; one rule, no exceptions, is simply
    # cheaper to audit. Using a bare kgrid.kz anywhere here is the most likely
    # silent-wrongness bug in this module (plans/CMHD_PLAN.md §7).
    kz = kgrid.kz
    return kz.at[params.nz // 2].set(0.0) if params.nz % 2 == 0 else kz


def _cross(a, b):
    # real-space cross product of two (3, nz, nx, ny) vector fields
    return jnp.stack([a[1]*b[2] - a[2]*b[1],
                      a[2]*b[0] - a[0]*b[2],
                      a[0]*b[1] - a[1]*b[0]])


def _curl_real(vk, kx, ky, kz, params):
    # real-space curl of a k-space vector field (3, nz, nkx, nky). i*k x v is k-local, so each
    # component is formed in k-space and inverse-transformed on its own: only one k-space
    # component is live at a time. That is the grad_fields memory rationale
    # (docs/numerics.md "Gradients are a tuple, one transform at a time") applied here by
    # hand -- grad_fields itself is a (d/dx, d/dy)-pair machine for bracket equations and is
    # deliberately not used (plans/CMHD_PLAN.md §3.6).
    #
    # The wavenumbers are passed in rather than read off kgrid because under expansion the
    # PHYSICAL curls (omega = grad~ x u, j = grad~ x B) use the comoving-scaled
    # k~ = (kx, ky/a, kz/a), while the INDUCTION curl stays on the static integer-grid k
    # (docs/numerics.md: A.(grad~ x E) = grad x E'). Off, every caller passes kgrid's own.
    cx = grids.ifft(1j*(ky*vk[2] - kz*vk[1]), params)
    cy = grids.ifft(1j*(kz*vk[0] - kx*vk[2]), params)
    cz = grids.ifft(1j*(kx*vk[1] - ky*vk[0]), params)
    return jnp.stack([cx, cy, cz])


def _enthalpy(rho, cs0, gamma, cs2=None):
    # h with h'(rho) = c_s^2(rho)/rho, so that (1/rho) grad p is exactly grad h. The branch is
    # trace-time python (params is static). rho**(gamma-1) is NaN for rho < 0 at non-integer
    # exponents, and log(rho) is NaN for rho < 0: that loud failure is INTENDED -- never
    # abs() or clip it (plans/CMHD_PLAN.md §7).
    #
    # cs2 overrides cs0^2 for the expanding box, where the sound speed is time-dependent
    # (cs^2(t) = cs0^2 a^-cs_q) and is therefore a traced value rather than a constant. When
    # it is None the expression is the python float cs0*cs0 exactly as before, so the
    # non-expanding graph is unchanged; taranis/diagnostics/cmhd.py calls the 3-arg form.
    c2 = cs0*cs0 if cs2 is None else cs2
    if gamma == 1.0:
        return c2 * jnp.log(rho)
    return (c2/(gamma - 1.0)) * rho**(gamma - 1.0)


def _enthalpy_s(s, cs0, gamma, cs2=None):
    # _enthalpy in the ln rho variable: h(e^s). Same h, no log and no negative-rho NaN
    # channel -- e^s > 0 structurally, which is the whole point of the variable
    # (docs/numerics.md "The ln rho density variable"):
    #
    #     h = cs^2 * s                          gamma = 1
    #     h = cs0^2 e^((gamma-1)s)/(gamma-1)    gamma > 1
    #
    # NonlinearTerm calls this only on the gamma > 1 branch: at gamma = 1 h is LINEAR in the
    # evolved field, so its gradient is the k-local -cs^2 i k~ s^ and forming h in real space
    # would cost a transform the 24-tally does not have. The gamma = 1 branch here is exactly
    # what that k-local force is the transform of, and is what the budget gates use.
    c2 = cs0*cs0 if cs2 is None else cs2
    if gamma == 1.0:
        return c2 * s
    return (c2/(gamma - 1.0)) * jnp.exp((gamma - 1.0)*s)


# ------------------------------------------------------------------- recipe functions

def linear_matrix(kgrid, params):
    # L = -diss_f * (k_perp^2 + kz^2)^hyper, the ENTIRE linear operator (rule 1 in the header).
    # Shape (1, nz, nkx, nky) for a uniform diss -- propagators.DiagonalOperator broadcasts the
    # leading axis over the fields -- or (7, nz, nkx, nky) for the (D_rho, nu, eta) schema.
    # Real and even in both kx and kz, so propagators._check_hermitian_compatible passes.
    #
    # NO EXPANSION BRANCH, by design: under EBM the dissipation acts on the PRIMED fields at
    # COMOVING k, which is exactly this operator (docs/numerics.md "Expanding box", the
    # recorded truncation choice -- physical-wavenumber dissipation would need nu(t) or a
    # per-block L rebuild). L is static in both modes.
    #
    # NO density_var BRANCH EITHER, and that is a verified property, not an oversight: L is
    # built from diss/hyper and the wavenumbers alone and never reads the field it acts on,
    # so the field-0 row simply damps s instead of rho when density_var = "lnrho". Diffusing
    # s is undershoot-proof (it cannot drive rho <= 0), affects int e^s only through the
    # variance, and is still budget-accounted through delta E/delta s (docs/numerics.md).
    _, diss, hyper, _ = _eqpars(params)
    kz = _kz_deriv(kgrid, params)
    ksq_tot = kgrid.ksq + kz*kz
    d = jnp.array(_diss_per_field(diss), dtype=_precision.ftype).reshape(-1, 1, 1, 1)
    return -d * ksq_tot**hyper


def grad(state, kgrid, params):
    # The 13 inverse transforms of one RHS evaluation: rho, u(3), B(3), omega = curl u (3),
    # j = curl B (3), the two curls formed k-locally first (docs/numerics.md, transform tally).
    # EXPANSION DOES NOT CHANGE THE TALLY -- every rescaling below is elementwise. The ln rho
    # path makes it 16: field 0's inverse transform gives s rather than rho, and grad~s adds
    # three more (one component at a time, as everywhere else in this module).
    _check_supported(params)
    exp = _expansion(params)                     # None off: TRACE-TIME python, see _expansion
    lnrho = _density_var(params) == "lnrho"      # "rho" off: TRACE-TIME python too
    kz = _kz_deriv(kgrid, params)
    fk = state.fields
    if exp is None:
        rho_p = None
        if lnrho:
            rho = None
            s = grids.ifft(fk[0], params)        # s = ln rho, the EVOLVED field 0
            inv_rho = jnp.exp(-s)                # 1/rho, the only density any consumer wants
        else:
            s = inv_rho = None
            rho = grids.ifft(fk[0], params)
        bk = fk[4:7]
        kxt, kyt, kzt = kgrid.kx, kgrid.ky, kz
    else:
        # unscale the PRIMED state elementwise, and build the comoving k~ = (kx, ky/a, kz/a).
        # a >= 1 for every reachable t (adot > 0, t >= 0), so one reciprocal and multiplies:
        # at t = 0, a = inv_a = inv_a2 = 1.0 exactly and the scalings are exact.
        adot, _ = exp
        a = _a_of(state.t, adot)
        inv_a = 1.0/a
        inv_a2 = inv_a*inv_a
        if lnrho:
            # the evolved field is s' = ln rho' = s + 2 ln a. The raw 1/rho is then
            # a^2 e^-s' -- the exact analogue of rho = rho'/a^2, done on the reciprocal so
            # that no exponential of a growing -2 ln a offset is ever formed. No rho' array
            # exists on this path: h is built on s' directly (docs).
            rho = rho_p = None
            s = grids.ifft(fk[0], params)
            inv_rho = (a*a)*jnp.exp(-s)
        else:
            s = inv_rho = None
            rho_p = grids.ifft(fk[0], params)    # rho' = a^2 rho, the EVOLVED density
            rho = rho_p*inv_a2                   # the physical rho every consumer wants
        bk = jnp.stack([fk[4]*inv_a2, fk[5]*inv_a, fk[6]*inv_a])     # B^ = A^-1 B'^
        kxt, kyt, kzt = kgrid.kx, kgrid.ky*inv_a, kz*inv_a
    u = jnp.stack([grids.ifft(fk[1], params), grids.ifft(fk[2], params),
                   grids.ifft(fk[3], params)])
    B = jnp.stack([grids.ifft(bk[0], params), grids.ifft(bk[1], params),
                   grids.ifft(bk[2], params)])
    # omega and j are the PHYSICAL curls: comoving k~ under expansion, kgrid's own off
    omega = _curl_real(fk[1:4], kxt, kyt, kzt, params)
    j = _curl_real(bk, kxt, kyt, kzt, params)
    if lnrho:
        # grad~ s, one component per inverse transform. Under expansion these are the
        # comoving k~ derivatives of s', which is what -u.grad~s' needs.
        gs = jnp.stack([grids.ifft(1j*kxt*fk[0], params),
                        grids.ifft(1j*kyt*fk[0], params),
                        grids.ifft(1j*kzt*fk[0], params)])
    else:
        gs = None
    return CMHDGrads(rho=rho, u=u, B=B, omega=omega, j=j, t=state.t, rho_p=rho_p,
                     s=s, gs=gs, inv_rho=inv_rho)


def NonlinearTerm(state, grads, kgrid, params, halo=None):
    # The whole ideal RHS as ONE term: the three equations share the real-space products, so
    # splitting them into separate Terms would re-transform them. halo is unused (z_spectral
    # has no halo). 10 forward transforms: rho*u(3), the combined curl force(3), the combined
    # scalar(1), u x B(3) -- 23 in total with grad()'s 13 inverse, in BOTH expansion modes.
    # On the ln rho path it is 8 forward (u.grad s(1) replaces rho*u(3); grad~.u is k-local
    # and h at gamma = 1 needs no transform at all) against grad()'s 16 inverse -- 24.
    #
    # The -(adot/a)*diag(0,1,1)*u expansion term is folded into du below rather than
    # registered as its own Term: physics/__init__.py's registry is not a C3b file, and a
    # 2-element sum reaching construct_rhs would be a registry change. It is still gated by
    # the same trace-time predicate, so the off graph is untouched either way.
    cs0, _, _, gamma = _eqpars(params)
    exp = _expansion(params)                     # None off: TRACE-TIME python
    lnrho = _density_var(params) == "lnrho"      # "rho" off: TRACE-TIME python too
    kx, ky = kgrid.kx, kgrid.ky
    kz = _kz_deriv(kgrid, params)
    rho, u, B, omega, j = grads.rho, grads.u, grads.B, grads.omega, grads.j
    if exp is None:
        kxt, kyt, kzt = kx, ky, kz
        cs2 = None                               # _enthalpy falls back to the constant cs0^2
        rho_e = rho                              # the evolved density IS the physical one
    else:
        adot, cs_q = exp
        a = _a_of(grads.t, adot)
        inv_a = 1.0/a
        kxt, kyt, kzt = kx, ky*inv_a, kz*inv_a   # k~ = (kx, ky/a, kz/a)
        # cs^2(t) = cs0^2 a^-cs_q, the prescribed cooling law. cs_q = 0 (constant
        # temperature) drops the pow at trace time.
        cs2 = cs0*cs0 if cs_q == 0.0 else cs0*cs0 * a**(-cs_q)
        rho_e = grads.rho_p                      # rho' = a^2 rho, what the state evolves

    # continuity. In rho-variables, FLUX form: mass is exact because -i*k.(rho u)^ vanishes
    # identically at k = 0. Under expansion this is d_t rho' = -grad~.(rho' u): the flux
    # carries the PRIMED density.
    #
    # In ln rho-variables it is the same equation divided by rho,
    #
    #     d_t s = -u.grad~ s - grad~ . u
    #
    # with the advection one forward transform and the divergence K-LOCAL (i k~ . u^, read
    # straight off the state -- u is unscaled in both modes, so state.fields[1:4] ARE u^).
    # Under expansion this carries NO expansion term: the -2(adot/a) of the raw continuity
    # cancels identically against d(2 ln a)/dt when s' = s + 2 ln a is the evolved variable
    # (docs/numerics.md). Neither piece is exactly zero at k = 0 on a turbulent state -- int
    # e^s is a nonlinear invariant now, gated by drift order, never bitwise -- but on a
    # UNIFORM state grad~s and grad~.u are exact fp zeros and s^(k=0) is bitwise.
    if lnrho:
        gs = grads.gs
        uk = state.fields[1:4]
        drho = (-grids.fft(u[0]*gs[0] + u[1]*gs[1] + u[2]*gs[2], params)
                - 1j*(kxt*uk[0] + kyt*uk[1] + kzt*uk[2]))
    else:
        mk = grids.fft(rho_e*u, params)
        drho = -1j*(kxt*mk[0] + kyt*mk[1] + kzt*mk[2])

    # momentum, rotational form. The two curl forces are summed in REAL space and transformed
    # once (3 transforms, not 6), and grad(|u|^2/2) merges with grad(h) into one scalar
    # transform -- that merge is the whole point of the rotational form. The rotational-form
    # identity survives the anisotropic metric: for any DIAGONAL d~_i = c_i d_i,
    # [(grad~ x u) x u]_i = (u.grad~)u_i - d~_i(u^2/2) (docs/numerics.md). The Lorentz force
    # divides by the UNPRIMED rho, and h is built on rho' -- the uniform -2 cs^2 ln a between
    # ln rho and ln rho' is killed BY the gradient (it only shifts sk at k = 0, where the
    # k-multiply is exactly zero), so it is not added back; rho' is used because ln rho' stays
    # O(1) while ln rho ~ -2 ln a drifts and costs conditioning in the gradient.
    #
    # In ln rho-variables the Lorentz force divides by rho through the precomputed
    # e^-s (a^2 e^-s' under expansion) rather than by a rho array that is never formed, and
    # the pressure force splits by gamma. At gamma = 1, h = cs^2 s is LINEAR in the evolved
    # field, so h^ IS cs^2 * s^ exactly and the force is k-local: fold it into the scalar
    # BEFORE the -i k~ multiply (algebraically identical to adding -cs^2 i k~ s^ separately,
    # one multiply cheaper) and the |u|^2/2 transform is the only one left -- this is the
    # transform the 24-tally does not spend. At gamma > 1, h is pointwise again and joins the
    # scalar exactly as it does in the rho form.
    if lnrho:
        fk = grids.fft(_cross(j, B)*grads.inv_rho - _cross(omega, u), params)
        if gamma == 1.0:
            sk = (grids.fft(0.5*(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]), params)
                  + (cs0*cs0 if cs2 is None else cs2)*state.fields[0])
        else:
            sk = grids.fft(0.5*(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
                           + _enthalpy_s(grads.s, cs0, gamma, cs2), params)
    else:
        fk = grids.fft(_cross(j, B)/rho - _cross(omega, u), params)
        sk = grids.fft(0.5*(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
                       + _enthalpy(rho_e, cs0, gamma, cs2), params)
    dux = fk[0] - 1j*kxt*sk
    duy = fk[1] - 1j*kyt*sk
    duz = fk[2] - 1j*kzt*sk
    if exp is not None:
        # the ONE new additive piece: -(adot/a) diag(0,1,1) u. u is unscaled, so its k-space
        # rows ARE state.fields[1:4]. x is the radial axis and is untouched (T_x = 0).
        eu = adot*inv_a
        duy = duy - eu*state.fields[2]
        duz = duz - eu*state.fields[3]
    du = jnp.stack([dux, duy, duz])

    # induction, curl form (header rule 2): k . dB is a pairwise-cancelling sum, so div B has
    # no systematic source. Under expansion this is the STATIC-k curl of
    # E' = ((u x B)_x, a(u x B)_y, a(u x B)_z) -- kx, ky, kz here, never k~, which is exactly
    # what preserves the pairwise cancellation and the bitwise B'(k=0) invariance.
    ek = grids.fft(_cross(u, B), params)
    if exp is not None:
        ek = jnp.stack([ek[0], a*ek[1], a*ek[2]])
    dB = jnp.stack([1j*(ky*ek[2] - kz*ek[1]),
                    1j*(kz*ek[0] - kx*ek[2]),
                    1j*(kx*ek[1] - ky*ek[0])])

    # one dealias multiply, on the assembled (7, nz, nkx, nky) RHS. The 2/3 mask needs no
    # expansion branch: products are formed on the comoving grid and alias in comoving mode
    # indices, so the mask never has to know about a(t) (docs/numerics.md).
    return jnp.concatenate([drho[None], du, dB]) * kgrid.dealias


def set_timestep(grads, params):
    # CFL on the angle-maximised fast speed: w^2/k^2 <= c_s^2 + v_A^2 from the dispersion
    # relation, so c_f(x) = sqrt(c_s^2(rho) + |B|^2/rho) bounds every wave pointwise.
    # d_z = params.dz = Lz/nz, which config.py sets for EVERY dims==3 run (z_spectral does not
    # remove it). set_timestep sees only what grad_func returned -- there is no state here.
    #
    # No quiescent floor is needed, and the reason is a GRID-MAX argument, not a pointwise one:
    # at gamma > 1, c_s(rho) = cs0*rho^((gamma-1)/2) falls BELOW cs0 wherever rho < 1, but mass
    # conservation pins max_x rho >= <rho> = 1 (a bitwise invariant of the discrete step), so
    # max_x c_f >= cs0 and the denominator is bounded away from zero for any reachable state.
    #
    # Under expansion the grid spacings are PHYSICAL -- (dx, a*dy, a*dz), the transverse
    # directions being the ones that stretch -- and the speeds come from the UNPRIMED fields,
    # which is exactly what grads already carries, plus the cooled cs^2(t). The grid-max
    # argument above is unaffected (EBM is gamma == 1, where cs = cs0*a^(-cs_q/2) > 0 is a
    # pointwise floor on c_f outright).
    cs0, _, _, gamma = _eqpars(params)
    exp = _expansion(params)                     # None off: TRACE-TIME python
    lnrho = _density_var(params) == "lnrho"      # "rho" off: TRACE-TIME python too
    rho, u, B = grads.rho, grads.u, grads.B
    if exp is None:
        if lnrho:
            # c_s(s) = cs0 e^((gamma-1)s/2), so c_s^2 = cs0^2 e^((gamma-1)s). THE NaN CHANNEL
            # IS GONE: rho = e^s > 0 structurally, so there is no negative-rho poisoning of
            # dt to derive a quiescent floor against -- and at the gamma = 1 DEFAULT c_s is
            # constant outright, so the CFL denominator's floor c_f >= cs0 is exact and
            # pointwise. At gamma > 1 the grid-max argument the rho form leans on (max_x rho
            # >= <rho> = 1, hence max_x c_f >= cs0) survives only to MASS-DRIFT order, since
            # int e^s is no longer a bitwise invariant -- the bound is as good as the drift
            # the conservation gate measures, which is the honest statement.
            cs2 = cs0*cs0 if gamma == 1.0 else cs0*cs0 * jnp.exp((gamma - 1.0)*grads.s)
        else:
            cs2 = cs0*cs0 if gamma == 1.0 else cs0*cs0 * rho**(gamma - 1.0)
        dx, dy, dz = params.dx, params.dy, params.dz
    else:
        adot, cs_q = exp
        a = _a_of(grads.t, adot)
        cs2 = cs0*cs0 if cs_q == 0.0 else cs0*cs0 * a**(-cs_q)      # gamma == 1 under EBM
        dx, dy, dz = params.dx, params.dy*a, params.dz*a
    if lnrho:
        # v_A^2 = |B|^2 e^-s on the PHYSICAL 1/rho grad already unscaled (a^2 e^-s' under
        # expansion), so this line reads the same physical speed as the rho branch below.
        vA2 = (B[0]*B[0] + B[1]*B[1] + B[2]*B[2])*grads.inv_rho
    else:
        vA2 = (B[0]*B[0] + B[1]*B[1] + B[2]*B[2])/rho
    cf = jnp.sqrt(cs2 + vA2)
    max_all = jnp.max(jnp.maximum(jnp.maximum((jnp.abs(u[0]) + cf)/dx,
                                              (jnp.abs(u[1]) + cf)/dy),
                                  (jnp.abs(u[2]) + cf)/dz))
    # identity under serial; kept for uniformity with the other equation sets
    max_all = comms.allreduce_max(max_all, params)
    dt = params.cfl_safety/max_all
    if not params.adaptive_timestep:
        # fixed-dt path: a direct call must not report more than the dt the run actually uses
        # (rmhd.set_timestep's _quiescent_dt ceiling does the same)
        dt = jnp.minimum(dt, params.dt)
    return dt
