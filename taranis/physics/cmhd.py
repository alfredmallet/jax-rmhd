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

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .. import comms, grids
from .. import _precision


class CMHDGrads(NamedTuple):
    # What one RHS evaluation needs in real space, computed once by grad() and shared by the
    # term func and by set_timestep. Not (d/dx, d/dy) pairs -- CMHD is not a bracket equation
    # set -- so this deliberately does NOT go through shared_physics.grad_fields.
    rho: jnp.ndarray        # (nz, nx, ny)
    u: jnp.ndarray          # (3, nz, nx, ny)
    B: jnp.ndarray          # (3, nz, nx, ny)
    omega: jnp.ndarray      # (3, nz, nx, ny), curl u
    j: jnp.ndarray          # (3, nz, nx, ny), curl B
    # state.t rides along because set_timestep sees only what grad_func returned; the MVP
    # ignores it and the expanding-box CFL (plan §7 / Phase C3) reads it.
    t: jnp.ndarray


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
_EQPARS_OPTIONAL = ("gamma",)


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


def _curl_real(vk, kgrid, kz, params):
    # real-space curl of a k-space vector field (3, nz, nkx, nky). i*k x v is k-local, so each
    # component is formed in k-space and inverse-transformed on its own: only one k-space
    # component is live at a time. That is the grad_fields memory rationale
    # (docs/numerics.md "Gradients are a tuple, one transform at a time") applied here by
    # hand -- grad_fields itself is a (d/dx, d/dy)-pair machine for bracket equations and is
    # deliberately not used (plans/CMHD_PLAN.md §3.6).
    kx, ky = kgrid.kx, kgrid.ky
    cx = grids.ifft(1j*(ky*vk[2] - kz*vk[1]), params)
    cy = grids.ifft(1j*(kz*vk[0] - kx*vk[2]), params)
    cz = grids.ifft(1j*(kx*vk[1] - ky*vk[0]), params)
    return jnp.stack([cx, cy, cz])


def _enthalpy(rho, cs0, gamma):
    # h with h'(rho) = c_s^2(rho)/rho, so that (1/rho) grad p is exactly grad h. The branch is
    # trace-time python (params is static). rho**(gamma-1) is NaN for rho < 0 at non-integer
    # exponents, and log(rho) is NaN for rho < 0: that loud failure is INTENDED -- never
    # abs() or clip it (plans/CMHD_PLAN.md §7).
    if gamma == 1.0:
        return cs0*cs0 * jnp.log(rho)
    return (cs0*cs0/(gamma - 1.0)) * rho**(gamma - 1.0)


# ------------------------------------------------------------------- recipe functions

def linear_matrix(kgrid, params):
    # L = -diss_f * (k_perp^2 + kz^2)^hyper, the ENTIRE linear operator (rule 1 in the header).
    # Shape (1, nz, nkx, nky) for a uniform diss -- propagators.DiagonalOperator broadcasts the
    # leading axis over the fields -- or (7, nz, nkx, nky) for the (D_rho, nu, eta) schema.
    # Real and even in both kx and kz, so propagators._check_hermitian_compatible passes.
    _, diss, hyper, _ = _eqpars(params)
    kz = _kz_deriv(kgrid, params)
    ksq_tot = kgrid.ksq + kz*kz
    d = jnp.array(_diss_per_field(diss), dtype=_precision.ftype).reshape(-1, 1, 1, 1)
    return -d * ksq_tot**hyper


def grad(state, kgrid, params):
    # The 13 inverse transforms of one RHS evaluation: rho, u(3), B(3), omega = curl u (3),
    # j = curl B (3), the two curls formed k-locally first (docs/numerics.md, transform tally).
    _check_supported(params)
    kz = _kz_deriv(kgrid, params)
    fk = state.fields
    rho = grids.ifft(fk[0], params)
    u = jnp.stack([grids.ifft(fk[1], params), grids.ifft(fk[2], params),
                   grids.ifft(fk[3], params)])
    B = jnp.stack([grids.ifft(fk[4], params), grids.ifft(fk[5], params),
                   grids.ifft(fk[6], params)])
    omega = _curl_real(fk[1:4], kgrid, kz, params)
    j = _curl_real(fk[4:7], kgrid, kz, params)
    return CMHDGrads(rho=rho, u=u, B=B, omega=omega, j=j, t=state.t)


def NonlinearTerm(state, grads, kgrid, params, halo=None):
    # The whole ideal RHS as ONE term: the three equations share the real-space products, so
    # splitting them into separate Terms would re-transform them. halo is unused (z_spectral
    # has no halo). 10 forward transforms: rho*u(3), the combined curl force(3), the combined
    # scalar(1), u x B(3).
    cs0, _, _, gamma = _eqpars(params)
    kx, ky = kgrid.kx, kgrid.ky
    kz = _kz_deriv(kgrid, params)
    rho, u, B, omega, j = grads.rho, grads.u, grads.B, grads.omega, grads.j

    # continuity, flux form: mass is exact because -i*k.(rho u)^ vanishes identically at k = 0
    mk = grids.fft(rho*u, params)
    drho = -1j*(kx*mk[0] + ky*mk[1] + kz*mk[2])

    # momentum, rotational form. The two curl forces are summed in REAL space and transformed
    # once (3 transforms, not 6), and grad(|u|^2/2) merges with grad(h) into one scalar
    # transform -- that merge is the whole point of the rotational form.
    fk = grids.fft(_cross(j, B)/rho - _cross(omega, u), params)
    sk = grids.fft(0.5*(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]) + _enthalpy(rho, cs0, gamma),
                   params)
    du = jnp.stack([fk[0] - 1j*kx*sk, fk[1] - 1j*ky*sk, fk[2] - 1j*kz*sk])

    # induction, curl form (header rule 2): k . dB is a pairwise-cancelling sum, so div B has
    # no systematic source.
    ek = grids.fft(_cross(u, B), params)
    dB = jnp.stack([1j*(ky*ek[2] - kz*ek[1]),
                    1j*(kz*ek[0] - kx*ek[2]),
                    1j*(kx*ek[1] - ky*ek[0])])

    # one dealias multiply, on the assembled (7, nz, nkx, nky) RHS
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
    cs0, _, _, gamma = _eqpars(params)
    rho, u, B = grads.rho, grads.u, grads.B
    cs2 = cs0*cs0 if gamma == 1.0 else cs0*cs0 * rho**(gamma - 1.0)
    vA2 = (B[0]*B[0] + B[1]*B[1] + B[2]*B[2])/rho
    cf = jnp.sqrt(cs2 + vA2)
    max_all = jnp.max(jnp.maximum(jnp.maximum((jnp.abs(u[0]) + cf)/params.dx,
                                              (jnp.abs(u[1]) + cf)/params.dy),
                                  (jnp.abs(u[2]) + cf)/params.dz))
    # identity under serial; kept for uniformity with the other equation sets
    max_all = comms.allreduce_max(max_all, params)
    dt = params.cfl_safety/max_all
    if not params.adaptive_timestep:
        # fixed-dt path: a direct call must not report more than the dt the run actually uses
        # (rmhd.set_timestep's _quiescent_dt ceiling does the same)
        dt = jnp.minimum(dt, params.dt)
    return dt
