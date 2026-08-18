# Coupled (live-solver) test-particle gates, plans/TESTPART_PLAN.md §6.
#
# This module holds gate 7 -- the E_z assembled from taranis.particles.fields against a
# centered difference of psi across a step -- plus the field-assembly sign and mask checks.
# Gate 7 is the direct numerical check of docs/numerics.md's "Test particles" derivation:
# E_z = +dpsi/dt = -{phi,psi} + L_psi psi + f_psi, with the ideal piece DEALIASED (the
# discrete psi integrates dealias*NL_psi). A sign error, a missing piece, or an undealiased
# bracket all show up as a failure to converge at O(dt^2).
#
# Gates 4 (canonical p_z), 5 (E = 0 heating floor) and 6 (solver untouched) need particle
# co-stepping in run.py and are added here in Phase A2.
#
# 2D, single-process, fp64 for the convergence tests. Script: `python tests/test_particles_coupled.py`.
from _rmhd_testing import bootstrap, checks, fit_order, fresh_params

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import taranis as jr
from taranis import grids, run
from taranis.particles.fields import (FIELD_MASK_DEFAULTS, assemble, full_mask,
                                      particle_fields)
from taranis.physics import construct_rhs, equation_registry
from taranis.physics.shared_physics import bracket, gradk
from taranis.timestepping import get_scheme

N = 64
DT_WARM = 0.01     # warm-up step length, the same for every dt tested
N_WARM = 5         # warm-up steps: makes forcing_state/forcing_scale and the fields nontrivial
DTS = (0.01, 0.005, 0.0025)


def _params(dt):
    # elsasser forcing (the only mode that drives psi) and diss large enough that the
    # resistive piece of E_z is not negligible at 64^2.
    return fresh_params(dims=2, nx=N, ny=N, Lx=2 * np.pi, Ly=2 * np.pi, dt=dt,
                        adaptive_timestep=False, diss=1e-2, hyper=1, cfl_safety=0.5,
                        forcing=True, forcing_mode="elsasser",
                        forcing_power_elsasser=(0.5, 0.5), forcing_tau=1.0, fshell=(1, 3))


def _ic(x, y):
    # outer-scale modes plus one pair inside the 2/3 cutoff (|n| = 21 at 64^2) whose
    # product is outside it: phi's (13,11) (|n| = 17.0) beating against psi's (12,-9)
    # (|n| = 15.0) gives (25,2) and (25,20), both beyond the cutoff. That is what makes
    # the dealiasing of the ideal E_z piece a measurable effect, not a round-off one.
    phi = (jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y)
           + 0.03 * jnp.sin(13 * x) * jnp.cos(11 * y))
    psi = (jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)
           + 0.03 * jnp.cos(12 * x - 9 * y))
    return jnp.stack([phi, psi], axis=0)


def _warm_state(params, kgrid):
    # a developed state with live forcing. block_of_steps advances the O-U state, so
    # forcing_state and forcing_scale are nonzero afterwards. Not donated: jax.jit here
    # has no donate_argnums, so the caller's state stays valid (unlike simulate*).
    stepper, scheme = get_scheme("lsrk33")
    state = run._refresh_forcing_scale(jr.initialize(_ic, params), kgrid, params)
    step = jax.jit(run.block_of_steps, static_argnums=(2, 3, 4, 5))
    return step(state, kgrid, params, N_WARM, scheme, stepper)


def _raw_stepper(params, kgrid):
    # the RAW stepper: no run._advance_forcing, so forcing_state/forcing_scale stay frozen
    # across the two steps and psi(t) is smooth (what a centered difference needs).
    recipe = equation_registry[params.eqtype]
    stepper, scheme = get_scheme("lsrk33")
    rhs = construct_rhs(recipe)
    return jax.jit(lambda s: stepper(s, kgrid, params, rhs, recipe.set_timestep_func, scheme))


def _psi_real(state, params):
    return grids.ifft(state.fields[1], params)


def _fd_and_ez(dt, mask):
    """max|E_z(mask) - (psi(s2) - psi(s0))/(2 dt)| at the midpoint state s1, and the E_z scale."""
    params = _params(dt)
    kgrid = jr.setup_kgrids(params)
    # warm-up at DT_WARM for every dt, so the three runs share the same s0 fields
    warm_params = _params(DT_WARM)
    s0 = _warm_state(warm_params, jr.setup_kgrids(warm_params))
    step = _raw_stepper(params, kgrid)
    s1 = step(s0)
    s2 = step(s1)
    pf = particle_fields(s1, kgrid, params, resistive=True, forcing=True)
    E, _ = assemble(pf, mask)
    fd = (_psi_real(s2, params) - _psi_real(s0, params)) / (2.0 * dt)
    return float(jnp.max(jnp.abs(E[2] - fd))), float(jnp.max(jnp.abs(E[2])))


@pytest.mark.fp64
def test_ez_matches_dpsi_dt():
    """Gate 7: with every piece on, E_z is dpsi/dt to the centered difference's O(dt^2)."""
    errs, scales = zip(*[_fd_and_ez(dt, full_mask()) for dt in DTS])
    order = fit_order(DTS, errs)
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    with checks() as c:
        c.check(f"full-mask E_z converges to (psi(t+dt)-psi(t-dt))/2dt at order "
                f"{order:.2f} (>= 1.8; ratios {['%.2f' % r for r in ratios]})", order >= 1.8,
                f"errs {['%.3e' % e for e in errs]}")
        c.check(f"... and the error is small next to max|E_z| = {scales[0]:.3e} "
                f"(rel {errs[0] / scales[0]:.2e} at dt={DTS[0]})",
                errs[0] < 1e-2 * scales[0])


@pytest.mark.fp64
def test_ez_pieces_are_all_needed():
    """Dropping the forcing or the resistive piece breaks the agreement at O(1) in dt:
    each piece is really live, not a rounding-level decoration."""
    full = [_fd_and_ez(dt, full_mask())[0] for dt in DTS]
    with checks() as c:
        for piece in ("ez_forcing", "ez_resistive"):
            errs = [_fd_and_ez(dt, dict(full_mask(), **{piece: False}))[0] for dt in DTS]
            order = fit_order(DTS, errs)
            c.check(f"without {piece} the FD comparison does NOT converge (order "
                    f"{order:.2f} < 0.5; errs {['%.3e' % e for e in errs]})", order < 0.5)
            c.check(f"... and its error at dt={DTS[-1]} ({errs[-1]:.3e}) dwarfs the "
                    f"full-mask error ({full[-1]:.3e})", errs[-1] > 30.0 * full[-1])


@pytest.mark.fp64
def test_perp_fields_signs_and_dealiasing():
    """PFields holds E_perp/B0 = -grad(phi) and b_perp = zhat x grad(psi) to round-off, and
    ez_ideal really is dealiased. (Kernel gate 2 pins the PUSHER's Lorentz-force convention
    on analytic fields; the assembly's relative signs are pinned by the E.B = 0 identity
    below, and their absolute sign against psi by gate 4 in Phase A2.)"""
    params = _params(DT_WARM)
    kgrid = jr.setup_kgrids(params)
    state = _warm_state(params, kgrid)
    pf = particle_fields(state, kgrid, params, resistive=True, forcing=True)
    g = grids.ifft(gradk(state.fields[:2], kgrid), params)
    gphi, gpsi = np.asarray(g[0]), np.asarray(g[1])
    tol = 1e-12 * max(1.0, float(np.max(np.abs(gphi))), float(np.max(np.abs(gpsi))))
    ideal_k = grids.fft(pf.ez_ideal, params)
    leak = float(jnp.max(jnp.abs(ideal_k * (1 - kgrid.dealias))))
    # the raw pointwise bracket: what ez_ideal must NOT be
    raw = -bracket(g[0], g[1])
    dropped = float(jnp.max(jnp.abs(raw - pf.ez_ideal)))
    with checks() as c:
        c.check("E_x/B0 = -d_x phi", np.allclose(np.asarray(pf.ex), -gphi[0], atol=tol, rtol=0.0))
        c.check("E_y/B0 = -d_y phi", np.allclose(np.asarray(pf.ey), -gphi[1], atol=tol, rtol=0.0))
        c.check("b_x = -d_y psi", np.allclose(np.asarray(pf.bx), -gpsi[1], atol=tol, rtol=0.0))
        c.check("b_y = +d_x psi", np.allclose(np.asarray(pf.by), gpsi[0], atol=tol, rtol=0.0))
        c.check("u = zhat x grad(phi) = (-d_y phi, d_x phi)",
                np.allclose(np.asarray(pf.ux), -gphi[1], atol=tol, rtol=0.0)
                and np.allclose(np.asarray(pf.uy), gphi[0], atol=tol, rtol=0.0))
        c.check(f"fft(ez_ideal) vanishes outside the 2/3 mask (leak {leak:.3e})",
                leak < 1e-10 * float(jnp.max(jnp.abs(ideal_k))))
        c.check(f"... and the mask actually removes something on this state: "
                f"max|raw bracket - ez_ideal| = {dropped:.3e} (so the convergence gate "
                f"above discriminates the two forms)", dropped > 1e-3)


@pytest.mark.fp64
def test_ideal_ohm_identity():
    """E.B = 0 exactly for the RAW (undealiased) ideal E_z -- the ideal-Ohm identity. It
    pins the field assembly's RELATIVE signs: b_perp against E_perp and E_z, and B_z = +B0
    (both E_perp and B_z carry B0, so a nontrivial B0 also pins assemble as the only place
    it enters). The absolute sign of b_perp/E_z against psi is gate 4's, in Phase A2."""
    params = _params(DT_WARM)
    kgrid = jr.setup_kgrids(params)
    state = _warm_state(params, kgrid)
    B0 = 1.7
    pf = particle_fields(state, kgrid, params)
    E, B = assemble(pf, dict(FIELD_MASK_DEFAULTS), B0=B0)
    g = grids.ifft(gradk(state.fields[:2], kgrid), params)
    ez_raw = -bracket(g[0], g[1])            # E_z,ideal before the dealiasing mask
    resid = float(jnp.max(jnp.abs(E[0] * B[0] + E[1] * B[1] + ez_raw * B0)))
    scale = (float(jnp.max(jnp.abs(E[:2]))) * float(jnp.max(jnp.abs(B[:2]))))
    print(f"ideal Ohm: max|E.B| (raw E_z) = {resid:.3e} vs |E_perp||b_perp| = {scale:.3e}")
    with checks() as c:
        c.check(f"E_perp.b_perp + B0*E_z,ideal(raw) = 0 to {resid:.2e} "
                f"(<= 1e-12 * {scale:.3e})", resid <= 1e-12 * scale, f"resid={resid}")
        c.check(f"B_z = +B0 = {B0}", bool(jnp.all(B[2] == B0)))
        c.check("E_perp scales with B0 (assemble is the only place it enters)",
                bool(jnp.allclose(E[0], B0 * pf.ex, rtol=0.0, atol=0.0))
                and bool(jnp.allclose(E[1], B0 * pf.ey, rtol=0.0, atol=0.0)))


def test_assemble_mask():
    """assemble() is static-python bookkeeping: zeros where a bit is off, B_z = B0, and
    loud errors for a bad key or a piece that was never computed. Cheap grid, both
    precisions (no convergence involved)."""
    params = fresh_params(dims=2, nx=16, ny=16, dt=0.01, adaptive_timestep=False,
                          diss=1e-2, hyper=1, cfl_safety=0.5)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_ic, params)
    pf = particle_fields(state, kgrid, params)          # ideal piece only
    B0 = 2.5
    with checks() as c:
        c.check("defaults are the ideal-Ohm particle",
                FIELD_MASK_DEFAULTS == dict(bperp=True, eperp=True, ez_ideal=True,
                                            ez_resistive=False, ez_forcing=False))
        E, B = assemble(pf, {}, B0=B0)
        c.check("default mask: E_z is the ideal piece",
                bool(jnp.all(E[2] == pf.ez_ideal)))
        c.check("B_z = B0 everywhere", bool(jnp.all(B[2] == B0)))
        c.check("b_perp live by default",
                bool(jnp.all(B[0] == pf.bx)) and bool(jnp.all(B[1] == pf.by)))
        E0, B0f = assemble(pf, dict(bperp=False, eperp=False, ez_ideal=False), B0=B0)
        c.check("everything off: E = 0 and b_perp = 0, B_z untouched",
                bool(jnp.all(E0 == 0.0)) and bool(jnp.all(B0f[:2] == 0.0))
                and bool(jnp.all(B0f[2] == B0)))
        c.check("E = 0 control keeps b_perp",
                bool(jnp.all(assemble(pf, dict(eperp=False, ez_ideal=False))[1][0] == pf.bx)))
        try:
            assemble(pf, dict(ez_ohmic=True))
            c.check("unknown mask key raises", False, "no ValueError")
        except ValueError:
            c.check("unknown mask key raises ValueError", True)
        try:
            assemble(pf, full_mask())
            c.check("requesting a None piece raises", False, "no ValueError")
        except ValueError:
            c.check("requesting a piece that was not computed raises ValueError", True)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
