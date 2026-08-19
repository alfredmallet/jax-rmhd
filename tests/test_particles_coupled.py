# Coupled (live-solver) test-particle gates, plans/TESTPART_PLAN.md §6.
#
#   4. Canonical p_z   in 2D z is ignorable, so p_z = v_z - qm*psi(x,t) is conserved per
#                      particle whenever the particle's E_z is exactly dpsi/dt. Frozen
#                      fields (dpsi/dt = 0, so E_z = 0) isolate interpolation + pusher:
#                      bilinear converges O(dx^2), the exact spectral gather leaves only
#                      the Boris O(dt^2). The invariant's sign pins b_perp and E_z against
#                      psi absolutely. Live fields with the full mask converge O(dt) (the
#                      fields are frozen across a step); the DEFAULT ideal-only ensemble
#                      does not converge at all -- its drift IS the omitted
#                      resistive + forcing acceleration.
#   5. E = 0 floor     the control ensemble (b_perp live, E off) in the same live run:
#                      magnetic forces do no work and the Boris rotation is norm-exact, so
#                      |v| conservation is the whole pipeline's numerical-heating floor.
#   6. Solver untouched  with params.particles=None the solver output is bitwise identical
#                      to the pre-A2 reference npz; with particles on it is bitwise
#                      identical to the particles-off run of the same session; and a
#                      restart from a snapshot continues both fields and trajectories
#                      bitwise.
#   7. E_z assembly    the assembled E_z against a centered difference of psi across a
#                      step -- the direct numerical check of docs/numerics.md's "Test
#                      particles" derivation, E_z = +dpsi/dt = -{phi,psi} + L_psi psi +
#                      f_psi with the ideal piece DEALIASED (the discrete psi integrates
#                      dealias*NL_psi). A sign error, a missing piece or an undealiased
#                      bracket all show up as a failure to converge at O(dt^2).
#   8. Work closure    the per-piece work accumulator w: summed over pieces it equals the
#                      per-particle kinetic-energy change to round-off (exact Boris
#                      algebra), a piece an ensemble's mask omits stays exactly zero, and
#                      w_ez_resistive alone separates the paired resistive-split
#                      ensembles. Plus: boris.push is bitwise its push_tracked wrapper.
#   9. E_par projection  `epar_project` replaces each gathered E sample by its component
#                      perpendicular to the gathered B, so the numerical E_parallel that
#                      independent bilinear interpolation of E and B leaves at the particle
#                      is removed exactly, closure included. Plus the per-ensemble B0
#                      (= 1/epsilon): at fixed Omega = qm*B0 it changes no orbit.
#
# Plus the field-assembly sign/mask checks that gate 7 rests on.
#
# 2D, single-process; gates 4/5/7 are fp64 (convergence), gates 6, 8 and 9 run in both
# sessions (the reference npz exists for both; the particle state, the work arithmetic and
# the gathered samples are fp64 whatever the field precision).
# Script: `python tests/test_particles_coupled.py`.
from _rmhd_testing import (bootstrap, checks, fit_order, fresh_params, make_state,
                           managed_manager, snap_dir)

bootstrap()

import functools
import os
import platform

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import taranis as jr
from taranis import grids, run
from taranis.particles import boris, interp
from taranis.particles.fields import (FIELD_MASK_DEFAULTS, WORK_PIECES, assemble,
                                      assemble_stacked, full_mask, particle_fields)
from taranis.particles.state import ParticleState, init_particles
from taranis.physics import construct_rhs, equation_registry
from taranis.physics.shared_physics import bracket, gradk
from taranis.snapshot_io import get_saved_steps, load_particles, load_snapshot
from taranis.timestepping import get_scheme

# The gate-6 reference npz and the runs that produced it are DEFINED in the generator, so
# the committed reference and the test reading it cannot drift apart (see its header).
from _gen_particles_gate6_reference import (CONFIGS, NBLOCK_CFL, NBLOCK_SCAN, T_END, T_SNAP,
                                            _snapshot_skeleton, _state_leaves, config_ctx,
                                            reference_path, run_config)
from _gen_particles_gate6_reference import _ic as _gate6_ic

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
    below, and their absolute sign against psi by gate 4 below.)"""
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
    it enters). The absolute sign of b_perp/E_z against psi is gate 4's, below."""
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


# ------------------------------------------------ gate 4: canonical p_z, frozen fields
#
# In 2D, z is ignorable and p_z = v_z - qm*psi is an exact per-particle invariant:
#   dv_z/dt = qm*(E_z + (v x B)_z) = qm*(E_z + v_perp . grad psi)   [b_perp = zhat x grad psi]
#   dpsi/dt = d_t psi + v_perp . grad psi
# so dp_z/dt = qm*(E_z - d_t psi). With the fields FROZEN, d_t psi = 0 and the invariant
# holds iff E_z = 0 -- which is what makes this variant a pure measurement of
# interpolation + pusher error, with no field time-integration in it. The residual is the
# known mismatch between the independently gathered b_perp and the gradient of the
# gathered psi (plans/TESTPART_PLAN.md §4).
#
# The frozen field is a band-limited ANALYTIC state, not a warmed turbulent one: a warm
# state is resolution-dependent (the O-U stream is shaped (nkx,nky)), and the dx
# refinement needs literally the same function at 32^2, 64^2 and 128^2. Every mode below
# has |n| <= 4, well inside the 2/3 cutoff (|n| = 10) of the coarsest grid.
_FROZ_QM = 5.0           # Omega = qm*B0 = 5, rho = vperp/Omega = 0.2 -> rho/dx = 1, 2, 4
_FROZ_VPERP = 1.0
_FROZ_B0 = 1.0
_FROZ_NPART = 128
_FROZ_DT = 0.01          # Omega*dt = 0.05, held fixed across the dx refinement
_FROZ_STEPS = 500        # T = 5, about 4 gyroperiods
_FROZ_NXS = (32, 64, 128)
_FROZ_DTS = (0.02, 0.01, 0.005)   # spectral variant, T held fixed
_FROZ_T = _FROZ_DT * _FROZ_STEPS

# frozen fields => E_z must be 0 for p_z to be invariant: b_perp and E_perp live, every
# E_z piece off
_FROZ_MASK = dict(FIELD_MASK_DEFAULTS, ez_ideal=False)


def _froz_ic(x, y):
    psi_amp = 0.2      # |b_perp| << B0, so the orbit is a clean weakly perturbed gyration
    phi = 0.5 * (jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y)
                 + 0.1 * jnp.cos(3 * x - 2 * y))
    psi = psi_amp * (jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)
                     + 0.1 * jnp.sin(4 * x + y))
    return jnp.stack([phi, psi], axis=0)


def _froz_params(nx):
    return fresh_params(dims=2, nx=nx, ny=nx, Lx=2 * np.pi, Ly=2 * np.pi, dt=_FROZ_DT,
                        adaptive_timestep=False, diss=1e-2, hyper=1, cfl_safety=0.5)


def _froz_seeds():
    # ring in gyrophase at random positions; drawn on the host so every resolution and
    # every dt starts from bitwise identical particles
    rng = np.random.default_rng(0)
    th = rng.uniform(0.0, 2.0 * np.pi, _FROZ_NPART)
    x = np.stack([rng.uniform(0.0, 2.0 * np.pi, _FROZ_NPART),
                  rng.uniform(0.0, 2.0 * np.pi, _FROZ_NPART),
                  np.zeros(_FROZ_NPART)], axis=1)
    v = np.stack([_FROZ_VPERP * np.cos(th), _FROZ_VPERP * np.sin(th),
                  np.zeros(_FROZ_NPART)], axis=1)
    return jnp.asarray(x), jnp.asarray(v)


@functools.lru_cache(maxsize=None)
def _froz_drift(nx, dt, nsteps, spectral=False, sign=-1.0):
    """max_i |p_z,i(T) - p_z,i(0)| for p_z = v_z + sign*qm*psi, pushed in the frozen
    fields of an analytic state. `spectral` swaps BOTH the push's gather and the psi
    sample for the exact rfft2 evaluation, so the two are always consistent."""
    params = _froz_params(nx)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_froz_ic, params)
    E, B = assemble(particle_fields(state, kgrid, params), _FROZ_MASK, B0=_FROZ_B0)
    if spectral:
        gather = interp.gather_spectral
        E, B = grids.fft(E, params), grids.fft(B, params)
        psi_field = state.fields[1][None]                       # (1, nz, nkx, nky)
    else:
        gather = interp.gather
        psi_field = grids.ifft(state.fields[1], params)[None]    # (1, nz, nx, ny)

    @jax.jit
    def orbit(x, v):
        def step(c, _):
            return boris.push(c[0], c[1], E, B, _FROZ_QM, dt, params, gather=gather), None
        (x, v), _ = jax.lax.scan(step, (x, v), None, length=nsteps)
        return x, v

    def pz(x, v):
        return v[:, 2] + sign * _FROZ_QM * gather(psi_field, x, params)[:, 0]

    x0, v0 = _froz_seeds()
    x1, v1 = orbit(x0, v0)
    return float(jnp.max(jnp.abs(pz(x1, v1) - pz(x0, v0))))


@pytest.mark.fp64
def test_pz_frozen_bilinear_converges_in_dx():
    """Gate 4a: with the production bilinear gather the p_z drift is the interpolation
    error alone (dt fixed, so no time-integration error varies), and it converges at
    O(dx^2) -- the rate a consistent b_perp/psi pair must give. What flattens it: a gather
    that samples b_perp and psi inconsistently, a mis-scaled grid spacing, or an E_z that
    is not zero in frozen fields (leaving ez_ideal on gives order ~0 and 3000x the drift
    at 128^2)."""
    dxs = [2.0 * np.pi / nx for nx in _FROZ_NXS]
    errs = [_froz_drift(nx, _FROZ_DT, _FROZ_STEPS) for nx in _FROZ_NXS]
    order = fit_order(dxs, errs)
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print(f"gate 4a: rho/dx {['%.2f' % ((_FROZ_VPERP / _FROZ_QM) / d) for d in dxs]} -> "
          f"max|dp_z| {['%.3e' % e for e in errs]} over T = {_FROZ_T}, order {order:.3f}")
    with checks() as c:
        c.check(f"bilinear p_z drift converges at order {order:.2f} in dx (>= 1.5; ratios "
                f"{['%.2f' % r for r in ratios]})", order >= 1.5,
                f"errs {['%.3e' % e for e in errs]}")
        c.check(f"... and at rho/dx = 4 the drift {errs[-1]:.2e} is tiny next to the p_z "
                f"scale qm*max|psi| ~ {_FROZ_QM * 0.2:.2f}", errs[-1] < 1e-2)


@pytest.mark.fp64
def test_pz_frozen_spectral_converges_in_dt():
    """Gate 4b: driving the SAME push through the exact spectral gather removes the
    interpolation error entirely, leaving only the Boris time-integration error -- which
    must be O(dt^2). Anything in the field assembly that is inconsistent with
    b_perp = zhat x grad psi survives interpolation refinement and would show up here as a
    dt-independent floor."""
    nsteps = [int(round(_FROZ_T / dt)) for dt in _FROZ_DTS]
    errs = [_froz_drift(64, dt, n, spectral=True) for dt, n in zip(_FROZ_DTS, nsteps)]
    order = fit_order(_FROZ_DTS, errs)
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print(f"gate 4b: Omega*dt {['%.3f' % (_FROZ_QM * dt) for dt in _FROZ_DTS]} -> "
          f"max|dp_z| {['%.3e' % e for e in errs]}, order {order:.3f}")
    with checks() as c:
        c.check(f"spectral-gather p_z drift converges at order {order:.2f} in dt (>= 1.8; "
                f"ratios {['%.2f' % r for r in ratios]})", order >= 1.8,
                f"errs {['%.3e' % e for e in errs]}")
        c.check(f"... and is well below the bilinear drift at the same resolution "
                f"({errs[1]:.2e} vs {_froz_drift(64, _FROZ_DT, _FROZ_STEPS):.2e})",
                errs[1] < _froz_drift(64, _FROZ_DT, _FROZ_STEPS))


@pytest.mark.fp64
def test_pz_sign_pins_bperp_and_ez_against_psi():
    """Gate 4c: the invariant is p_z = v_z - qm*psi, not v_z + qm*psi. The ideal-Ohm
    identity E.B = 0 fixes b_perp, E_perp and E_z only RELATIVE to each other; flipping
    all three together leaves it satisfied. This check is what pins their absolute sign
    against psi (A_z = -psi, docs/numerics.md): the wrong sign leaves 2*qm*dpsi along the
    orbit, orders of magnitude above the conserved combination's drift."""
    right = _froz_drift(128, _FROZ_DT, _FROZ_STEPS)
    wrong = _froz_drift(128, _FROZ_DT, _FROZ_STEPS, sign=+1.0)
    print(f"gate 4c: max|d(v_z - qm psi)| = {right:.3e} vs max|d(v_z + qm psi)| = "
          f"{wrong:.3e} (ratio {wrong / right:.1f})")
    with checks() as c:
        c.check(f"v_z - qm*psi is conserved and v_z + qm*psi is not (ratio "
                f"{wrong / right:.0f} >= 30)", wrong >= 30.0 * right,
                f"right={right} wrong={wrong}")


# ------------------------------- gates 4 (live) and 5: co-stepped in evolving fields

_LIVE_NX = 64
_LIVE_QM = 5.0            # rho = 0.2 = 2*dx at 64^2
_LIVE_VPERP = 1.0
_LIVE_NPART = 256
_LIVE_DTS = (0.01, 0.005, 0.0025)
_LIVE_T = 1.0             # about 0.8 gyroperiods of strongly forced evolution
# heavy dissipation on purpose: it keeps psi smooth, so the (dt-independent) bilinear
# interpolation floor stays below the O(dt) field-freezing error the gate measures, and it
# makes the resistive piece of E_z large enough to separate the ensembles cleanly
_LIVE_DISS = 1e-1

# ensemble 0 sees the exact dpsi/dt (p_z conserved), 1 is the production DEFAULT
# ideal-Ohm particle (p_z drift = the omitted resistive+forcing acceleration), 2 is the
# E = 0 control of gate 5. Same qm and the same ring speed, so the three are comparable.
_LIVE_ENSEMBLES = [
    {"qm": _LIVE_QM, "init": {"kind": "ring", "vperp": _LIVE_VPERP},
     "ez_resistive": True, "ez_forcing": True},
    {"qm": _LIVE_QM, "init": {"kind": "ring", "vperp": _LIVE_VPERP}},
    {"qm": _LIVE_QM, "init": {"kind": "ring", "vperp": _LIVE_VPERP},
     "eperp": False, "ez_ideal": False},
]
_LIVE_LABELS = ("full dpsi/dt", "ideal-only (default)", "E = 0 control")


def _live_params(dt, particles=None):
    return fresh_params(dims=2, nx=_LIVE_NX, ny=_LIVE_NX, Lx=2 * np.pi, Ly=2 * np.pi, dt=dt,
                        adaptive_timestep=False, diss=_LIVE_DISS, hyper=1, cfl_safety=0.5,
                        forcing=True, forcing_mode="elsasser",
                        forcing_power_elsasser=(0.5, 0.5), forcing_tau=1.0, fshell=(1, 3),
                        particles=particles)


def _pz_per_ensemble(state, pstate, params):
    # p_z = v_z - qm*psi with psi sampled by the SAME bilinear gather the push uses
    psi = grids.ifft(state.fields[1], params)[None]
    return [np.asarray(pstate.v[e][:, 2]
                       - ens["qm"] * interp.gather(psi, pstate.x[e], params)[:, 0])
            for e, ens in enumerate(params.particles["ensembles"])]


@functools.lru_cache(maxsize=None)
def _live_run(dt):
    """One co-stepped run of length _LIVE_T through run.block_of_steps' (state, pstate)
    carry. Returns host arrays only: rms p_z drift per ensemble, the per-particle |v|^2
    drift of every ensemble, and the p_z scale qm*rms(psi)."""
    warm_params = _live_params(_LIVE_DTS[0])         # identical warm state for every dt
    s0 = _warm_state(warm_params, jr.setup_kgrids(warm_params))
    params = _live_params(dt, particles={"seed": 0, "n": _LIVE_NPART,
                                         "ensembles": _LIVE_ENSEMBLES})
    kgrid = jr.setup_kgrids(params)
    p0 = init_particles(params)
    stepper, scheme = get_scheme("lsrk33")
    step = jax.jit(run.block_of_steps, static_argnums=(2, 3, 4, 5))
    (sT, pT), _ys = step((s0, p0), kgrid, params, int(round(_LIVE_T / dt)), scheme, stepper)
    a = _pz_per_ensemble(s0, p0, params)
    b = _pz_per_ensemble(sT, pT, params)
    v0, vT = np.asarray(p0.v), np.asarray(pT.v)
    return {
        "drift": np.array([float(np.sqrt(np.mean((y - x) ** 2))) for x, y in zip(a, b)]),
        "vsq_rel": np.abs(np.sum(vT ** 2, axis=2) / np.sum(v0 ** 2, axis=2) - 1.0),
        "pz_scale": _LIVE_QM * float(jnp.sqrt(jnp.mean(grids.ifft(s0.fields[1],
                                                                  params) ** 2))),
    }


@pytest.mark.fp64
def test_pz_live_full_mask_converges_and_ideal_only_does_not():
    """Gate 4, live variant. The full-dpsi/dt ensemble's only error is that the fields are
    frozen across each step, so its p_z drift is O(dt). The default ideal-only ensemble
    omits the resistive and forcing pieces of E_z, which are real accelerations, not
    discretization: its drift is two orders of magnitude larger and does not shrink with
    dt at all. That contrast is the gate -- a missing or mis-signed E_z piece in the FULL
    mask would put the two ensembles on the same footing."""
    res = [_live_run(dt) for dt in _LIVE_DTS]
    full = [r["drift"][0] for r in res]
    ideal = [r["drift"][1] for r in res]
    o_full, o_ideal = fit_order(_LIVE_DTS, full), fit_order(_LIVE_DTS, ideal)
    ratio = ideal[-1] / full[-1]
    scale = res[-1]["pz_scale"]
    for lbl, col in zip(_LIVE_LABELS, np.array([r["drift"] for r in res]).T):
        print(f"gate 4 live [{lbl}]: rms|dp_z| {['%.3e' % v for v in col]}")
    print(f"gate 4 live: orders full {o_full:.3f}, ideal-only {o_ideal:.3f}; "
          f"qm*rms(psi) = {scale:.3f}")
    with checks() as c:
        c.check(f"full-mask p_z drift converges at order {o_full:.2f} in dt (>= 0.8: the "
                f"fields are frozen across a step, so O(dt))", o_full >= 0.8,
                f"drifts {['%.3e' % v for v in full]}")
        c.check(f"... and is small next to the p_z scale qm*rms(psi) = {scale:.2f} "
                f"({full[-1] / scale:.2e} of it)", full[-1] < 0.02 * scale)
        c.check(f"ideal-only p_z drift does NOT converge (order {o_ideal:.2f} < 0.3)",
                o_ideal < 0.3, f"drifts {['%.3e' % v for v in ideal]}")
        c.check(f"... and is {ratio:.0f}x the full-mask drift at dt = {_LIVE_DTS[-1]} "
                f"(>= 50): the omitted resistive+forcing acceleration", ratio >= 50.0,
                f"ideal={ideal[-1]} full={full[-1]}")


@pytest.mark.fp64
def test_e_zero_ensemble_is_the_heating_floor():
    """Gate 5: in the same live run the E = 0 control ensemble (b_perp live, no electric
    field) can do no work on the particles, and the Boris rotation is norm-exact -- so
    |v| is conserved to round-off through the whole pipeline (gather of a time-varying
    b_perp included). This is the numerical-heating floor every heating claim is quoted
    against; a leak here (an E piece bleeding through the mask, a non-orthogonal rotation)
    would be invisible in the drifting ensembles."""
    res = _live_run(_LIVE_DTS[-1])
    floor = float(np.max(res["vsq_rel"][2]))
    others = [float(np.mean(res["vsq_rel"][e])) for e in (0, 1)]
    print(f"gate 5: E=0 ensemble max per-particle |v|^2 relative drift = {floor:.3e}; "
          f"mean |v|^2 growth of the accelerated ensembles {['%.2f' % v for v in others]}")
    with checks() as c:
        c.check(f"E = 0 ensemble conserves |v|^2 per particle to {floor:.2e} (<= 1e-12)",
                floor <= 1e-12, f"floor={floor}")
        c.check("... while the E-carrying ensembles heat by order unity, so the floor is "
                "not a trivially static run",
                all(v > 0.1 for v in others), f"{others}")


# --------------------------------------------- gate 8: per-piece work closure
#
# The Boris rotation is norm-exact, so each half-kick's kinetic-energy change is exactly
# h*E.(v_in + v_new) with h = qm*dt_k/2 (docs/numerics.md, "Work bookkeeping"). Splitting
# that expression by E component is what ParticleState.w accumulates, so summing w over
# the pieces must reproduce 1/2|v|^2 - 1/2|v0|^2 per particle to round-off -- through the
# whole co-stepped run, whatever the mask. That closure is what makes the heating
# attribution a measurement rather than an inference.

_CLOSE_DT = 0.005
_CLOSE_STEPS = 100
# the three standard ensembles plus one with b_perp off (E_perp and the ideal E_z live,
# B = B0 zhat): four different (nez, eperp, bperp) combinations through the same
# accumulator
_CLOSE_ENSEMBLES = _LIVE_ENSEMBLES + [
    {"qm": _LIVE_QM, "init": {"kind": "ring", "vperp": _LIVE_VPERP}, "bperp": False},
]
_CLOSE_LABELS = _LIVE_LABELS + ("b_perp = 0",)


@functools.lru_cache(maxsize=None)
def _closure_run():
    """A short co-stepped run of the four gate-8 ensembles. Returns host arrays: the
    per-particle work split w, the kinetic-energy change, and the initial energy scale."""
    warm_params = _live_params(_LIVE_DTS[0])
    s0 = _warm_state(warm_params, jr.setup_kgrids(warm_params))
    params = _live_params(_CLOSE_DT, particles={"seed": 0, "n": _LIVE_NPART,
                                                "ensembles": _CLOSE_ENSEMBLES})
    kgrid = jr.setup_kgrids(params)
    p0 = init_particles(params)
    v0 = np.asarray(p0.v)
    stepper, scheme = get_scheme("lsrk33")
    step = jax.jit(run.block_of_steps, static_argnums=(2, 3, 4, 5))
    (_sT, pT), _ys = step((s0, p0), kgrid, params, _CLOSE_STEPS, scheme, stepper)
    return {"w": np.asarray(pT.w),
            "dke": 0.5 * np.sum(np.asarray(pT.v) ** 2, axis=2) - 0.5 * np.sum(v0 ** 2, axis=2),
            "ke0": float(np.mean(0.5 * np.sum(v0 ** 2, axis=2))),
            "masks": [e["mask"] for e in params.particles["ensembles"]]}


def test_work_split_closes_against_the_kinetic_energy():
    """Gate 8: per particle, the accumulated per-piece work sums to the kinetic-energy
    change -- exact Boris algebra, so the residual is round-off however the fields are
    interpolated and whatever TARANIS_PRECISION is (w and the push arithmetic are fp64
    always). A residual above round-off would mean the work is being credited with a
    different E than the kick used, which would silently bias every heating
    attribution."""
    res = _closure_run()
    w, dke, ke0 = res["w"], res["dke"], res["ke0"]
    resid = np.max(np.abs(np.sum(w, axis=2) - dke), axis=1) / ke0
    for lbl, r, m in zip(_CLOSE_LABELS, resid, np.mean(np.abs(dke), axis=1) / ke0):
        print(f"gate 8 [{lbl}]: max|sum w - dKE|/KE0 = {r:.2e}, mean|dKE|/KE0 = {m:.2e}")
    with checks() as c:
        c.check(f"the work split closes to {np.max(resid):.2e} of KE0 for every ensemble "
                f"(<= 1e-12)", np.max(resid) <= 1e-12, str(resid))
        c.check("... on a run where the E-carrying ensembles actually gained energy, so "
                "the closure is not trivial",
                np.mean(np.abs(dke[0])) > 0.01 * ke0, str(np.mean(np.abs(dke), axis=1)))


def test_work_pieces_track_the_mask():
    """Gate 8's discriminator: a piece an ensemble does not see contributes EXACTLY zero
    work forever, so the paired resistive-split ensembles (identical fields, identical
    draw) are separated by w_ez_resistive alone -- the direct measurement plans/
    TESTPART_PLAN.md §1 asks for. The E = 0 control's w is identically zero, the same
    statement as gate 5's |v| conservation read through the accumulator."""
    res = _closure_run()
    w, masks = res["w"], res["masks"]
    off_leak = {}
    for e, mask in enumerate(masks):
        for i, piece in enumerate(WORK_PIECES):
            if not mask[piece]:
                off_leak[(_CLOSE_LABELS[e], piece)] = float(np.max(np.abs(w[e, :, i])))
    ir = WORK_PIECES.index("ez_resistive")
    full_res = float(np.mean(np.abs(w[0, :, ir])))
    ideal_res = float(np.max(np.abs(w[1, :, ir])))
    print(f"gate 8: full-mask mean|w_ez_resistive| = {full_res:.3e}, ideal-only = "
          f"{ideal_res:.1e}; E=0 max|w| = {float(np.max(np.abs(w[2]))):.1e}")
    with checks() as c:
        c.check("every piece an ensemble's mask switches off carries exactly zero work",
                all(v == 0.0 for v in off_leak.values()),
                str({k: v for k, v in off_leak.items() if v != 0.0}))
        c.check("the E = 0 control ensemble's w is identically zero",
                bool(np.all(w[2] == 0.0)), str(np.max(np.abs(w[2]))))
        c.check(f"the resistive split separates the paired ensembles: full-mask "
                f"mean|w_ez_resistive| = {full_res:.2e} > 0, ideal-only exactly 0",
                full_res > 0.0 and ideal_res == 0.0)
        c.check("the ideal piece is live in both paired ensembles (they differ only by "
                "the non-ideal pieces)",
                float(np.mean(np.abs(w[0, :, WORK_PIECES.index("ez_ideal")]))) > 0.0
                and float(np.mean(np.abs(w[1, :, WORK_PIECES.index("ez_ideal")]))) > 0.0)


def test_push_wrapper_matches_push_tracked():
    """boris.push is a thin wrapper over push_tracked (one E_z piece, E and B
    concatenated), so the kernel gates that drive push and the production path that drives
    push_tracked must be the SAME arithmetic, bit for bit."""
    params = _froz_params(64)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_froz_ic, params)
    pf = particle_fields(state, kgrid, params)
    E, B = assemble(pf, dict(FIELD_MASK_DEFAULTS), B0=_FROZ_B0)
    F, ez_on = assemble_stacked(pf, dict(FIELD_MASK_DEFAULTS), B0=_FROZ_B0)
    x0, v0 = _froz_seeds()
    xa, va = boris.push(x0, v0, E, B, _FROZ_QM, _FROZ_DT, params, substeps=2)
    xb, vb, dw, bs = boris.push_tracked(x0, v0, F, len(ez_on), _FROZ_QM, _FROZ_DT, params,
                                        substeps=2)
    with checks() as c:
        c.check("a default-mask assemble_stacked is exactly [E, B] concatenated",
                ez_on == ("ez_ideal",)
                and bool(jnp.all(F == jnp.concatenate([E, B]))))
        c.check("push and push_tracked agree bitwise on x and v",
                np.array_equal(np.asarray(xa), np.asarray(xb))
                and np.array_equal(np.asarray(va), np.asarray(vb)))
        c.check("push_tracked's extra outputs have the documented shapes and dtypes",
                dw.shape == (v0.shape[0], 2) and bs.shape == v0.shape
                and dw.dtype == jnp.float64 and bs.dtype == jnp.float64,
                f"{dw.shape} {bs.shape}")
        c.check("bsample is B at the RETURNED position (the last half-kick's gather)",
                np.allclose(np.asarray(bs), np.asarray(interp.gather(B, xb, params)),
                            rtol=0.0, atol=0.0))


# ------------------------------------------- gate 9: the E_parallel projection
#
# On the grid the ideal fields satisfy E.B = 0 exactly, but bilinear interpolation samples
# E and B independently, so what the particle sees carries a numerical E_parallel.
# `epar_project` removes it: at every half-kick the gathered E is replaced by
# E - (E.B)/(B.B) B, so the sample the kick uses is exactly perpendicular to the sampled B.
# It is defined only for the ideal-Ohm mask (with the resistive/forcing pieces on it would
# delete a REAL parallel field) -- that validation is in tests/test_particles_config.py.

_PROJ_DT = 0.005
_PROJ_STEPS = 100
# the default ideal-Ohm particle, the same ensemble with the projection on, and the full
# dpsi/dt ensemble (which must not be projected)
_PROJ_ENSEMBLES = [
    {"qm": _LIVE_QM, "init": {"kind": "ring", "vperp": _LIVE_VPERP}},
    {"qm": _LIVE_QM, "init": {"kind": "ring", "vperp": _LIVE_VPERP}, "epar_project": True},
    {"qm": _LIVE_QM, "init": {"kind": "ring", "vperp": _LIVE_VPERP},
     "ez_resistive": True, "ez_forcing": True},
]


@functools.lru_cache(maxsize=None)
def _proj_run():
    """A short co-stepped run of the gate-9 ensembles, all three started from ONE draw so
    the projected and unprojected ideal ensembles differ by the projection alone. Returns
    host arrays: E.B at the particles before and after projecting the sample, its |E||B|
    scale, the work/kinetic-energy pair for the closure, and the two trajectories."""
    warm_params = _live_params(_LIVE_DTS[0])
    s0 = _warm_state(warm_params, jr.setup_kgrids(warm_params))
    params = _live_params(_PROJ_DT, particles={"seed": 0, "n": _LIVE_NPART,
                                               "ensembles": _PROJ_ENSEMBLES})
    kgrid = jr.setup_kgrids(params)
    p0 = init_particles(params)
    p0 = p0._replace(x=jnp.stack([p0.x[0]] * 3), v=jnp.stack([p0.v[0]] * 3))
    v0 = np.asarray(p0.v)
    stepper, scheme = get_scheme("lsrk33")
    step = jax.jit(run.block_of_steps, static_argnums=(2, 3, 4, 5))
    (sT, pT), _ys = step((s0, p0), kgrid, params, _PROJ_STEPS, scheme, stepper)
    # the ideal-Ohm (E, B) the two paired ensembles ride, sampled at their end positions
    F, _ez_on = assemble_stacked(particle_fields(sT, kgrid, params),
                                 dict(FIELD_MASK_DEFAULTS),
                                 B0=params.particles["ensembles"][0]["B0"])
    project = jax.vmap(boris.project_perp, in_axes=(0, 0))
    dots = {}
    for e in (0, 1):
        s = interp.gather(F, pT.x[e], params)
        E, B = s[:, :3], s[:, -3:]
        Ep = project(E, B)
        E, B, Ep = np.asarray(E), np.asarray(B), np.asarray(Ep)
        dots[e] = {"eb": np.sum(E * B, axis=1), "eb_proj": np.sum(Ep * B, axis=1),
                   "scale": np.linalg.norm(E, axis=1) * np.linalg.norm(B, axis=1)}
    return {"dots": dots, "w": np.asarray(pT.w),
            "dke": 0.5 * np.sum(np.asarray(pT.v) ** 2, axis=2) - 0.5 * np.sum(v0 ** 2, axis=2),
            "ke0": float(np.mean(0.5 * np.sum(v0 ** 2, axis=2))),
            "x": np.asarray(pT.x), "v": np.asarray(pT.v)}


def test_epar_projection_removes_the_numerical_parallel_field():
    """Gate 9: at the particle, the projected sample's E'.B is round-off, while the
    unprojected one carries an O(interpolation error) parallel field -- the numerical
    E_parallel the projection exists to remove. The two paired ensembles start from the
    same draw, so their trajectories separating is the projection acting, and the work
    closure still holds for the projected one (the work columns credit the PROJECTED
    field, so the exact half-kick identity is unaffected)."""
    res = _proj_run()
    raw, proj = res["dots"][0], res["dots"][1]
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    rel_raw = rms(raw["eb"]) / rms(raw["scale"])
    rel_proj = rms(proj["eb_proj"]) / rms(proj["scale"])
    resid = np.max(np.abs(np.sum(res["w"], axis=2) - res["dke"]), axis=1) / res["ke0"]
    sep = float(np.max(np.abs(res["x"][1] - res["x"][0])))
    print(f"gate 9: unprojected rms(E.B)/rms(|E||B|) = {rel_raw:.3e}, projected "
          f"{rel_proj:.3e}; max|x_proj - x_raw| = {sep:.3e}")
    with checks() as c:
        c.check(f"the projected sample is exactly perpendicular to B: "
                f"rms(E'.B)/rms(|E||B|) = {rel_proj:.2e} (<= 1e-15)", rel_proj <= 1e-15,
                f"max|E'.B| = {np.max(np.abs(proj['eb_proj'])):.3e}")
        c.check(f"... while the unprojected gather leaves a numerical E_parallel: "
                f"rms(E.B)/rms(|E||B|) = {rel_raw:.2e} (>= 1e-6)", rel_raw >= 1e-6,
                f"rel_raw={rel_raw}")
        c.check(f"the projection changes the orbits (same draw, max|dx| = {sep:.2e})",
                sep > 1e-6, f"sep={sep}")
        c.check(f"the work closure survives the projection for every ensemble "
                f"(max residual {np.max(resid):.2e} of KE0 <= 1e-12)",
                np.max(resid) <= 1e-12, str(resid))
        c.check("... on a run where the ensembles actually gained energy",
                np.mean(np.abs(res["dke"][1])) > 0.01 * res["ke0"],
                str(np.mean(np.abs(res["dke"]), axis=1)))


def test_projection_flag_matches_projecting_the_gathered_sample():
    """The `project` flag is applied to EVERY half-kick's sample, not once per step: the
    pushed trajectory is bitwise the one obtained by projecting the gather's output
    instead. Frozen analytic fields, so the comparison is pure pusher arithmetic."""
    params = _froz_params(64)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_froz_ic, params)
    F, ez_on = assemble_stacked(particle_fields(state, kgrid, params),
                                dict(FIELD_MASK_DEFAULTS), B0=_FROZ_B0)
    project = jax.vmap(boris.project_perp, in_axes=(0, 0))

    def proj_gather(fields, pos, prm):
        s = interp.gather(fields, pos, prm)
        return jnp.concatenate([project(s[:, :3], s[:, -3:]), s[:, -3:]], axis=1)

    x0, v0 = _froz_seeds()
    args = (x0, v0, F, len(ez_on), _FROZ_QM, _FROZ_DT, params)
    xa, va, dwa, _ba = boris.push_tracked(*args, substeps=2, project=True)
    xb, vb, dwb, _bb = boris.push_tracked(*args, substeps=2, gather=proj_gather)
    xc, vc, _dwc, _bc = boris.push_tracked(*args, substeps=2)
    try:
        boris.push_tracked(x0, v0, jnp.concatenate([F[:2], F[2:3], F[2:3], F[-3:]]), 2,
                           _FROZ_QM, _FROZ_DT, params, project=True)
        nez_msg = "no AssertionError"
    except AssertionError as e:
        nez_msg = str(e)
    with checks() as c:
        c.check("project=True is bitwise a per-sample projection of the gather",
                np.array_equal(np.asarray(xa), np.asarray(xb))
                and np.array_equal(np.asarray(va), np.asarray(vb))
                and np.array_equal(np.asarray(dwa), np.asarray(dwb)),
                str(np.max(np.abs(np.asarray(va) - np.asarray(vb)))))
        c.check("... and it is not a no-op (the unprojected push differs)",
                not np.array_equal(np.asarray(vc), np.asarray(va)),
                str(np.max(np.abs(np.asarray(vc) - np.asarray(va)))))
        c.check("project=True with more than one E_z piece is an AssertionError naming nez",
                "nez" in nez_msg, nez_msg)


def test_per_ensemble_B0_is_the_amplitude_parameter():
    """B0 is per ensemble (B0 = 1/epsilon), so one run can hold several amplitudes. With
    q/m scaled to keep Omega = qm*B0 fixed, the gyroradius rho = v_perp/(qm*B0) is
    unchanged -- and for the delta_b = 0 gyration control (every field piece off, so the
    particle sees only B0 zhat and E = 0) the two ensembles' orbits are bitwise the same
    trajectory, which is the statement that B0 alone carries no dynamics."""
    off = dict(bperp=False, eperp=False, ez_ideal=False)
    ensembles = [dict(off, qm=10.0, B0=1.0, init={"kind": "ring", "vperp": _LIVE_VPERP}),
                 dict(off, qm=1.0, B0=10.0, init={"kind": "ring", "vperp": _LIVE_VPERP})]
    params = _live_params(_PROJ_DT, particles={"seed": 0, "n": 64, "ensembles": ensembles})
    kgrid = jr.setup_kgrids(params)
    warm_params = _live_params(_LIVE_DTS[0])
    s0 = _warm_state(warm_params, jr.setup_kgrids(warm_params))
    p0 = init_particles(params)
    p0 = p0._replace(x=jnp.stack([p0.x[0]] * 2), v=jnp.stack([p0.v[0]] * 2))
    stepper, scheme = get_scheme("lsrk33")
    step = jax.jit(run.block_of_steps, static_argnums=(2, 3, 4, 5))
    (_sT, pT), _ys = step((s0, p0), kgrid, params, _PROJ_STEPS, scheme, stepper)
    x, v = np.asarray(pT.x), np.asarray(pT.v)
    v0 = np.asarray(p0.v)
    rho = [np.hypot(v0[e, :, 0], v0[e, :, 1]) / abs(ens["qm"] * ens["B0"])
           for e, ens in enumerate(params.particles["ensembles"])]
    vsq_rel = np.abs(np.sum(v ** 2, axis=2) / np.sum(v0 ** 2, axis=2) - 1.0)
    with checks() as c:
        c.check("each ensemble's B0 is stored on the ensemble, the top-level one is the "
                "default",
                [e["B0"] for e in params.particles["ensembles"]] == [1.0, 10.0]
                and params.particles["B0"] == 1.0)
        c.check("Omega = qm*B0 equal => identical gyroradius rho = v_perp/(qm*B0)",
                np.array_equal(rho[0], rho[1]), str(np.max(np.abs(rho[0] - rho[1]))))
        c.check("the delta_b = 0 control orbits are bitwise identical at B0 = 1 and 10",
                np.array_equal(x[0], x[1]) and np.array_equal(v[0], v[1]),
                f"max|dx| = {np.max(np.abs(x[0] - x[1])):.3e}")
        c.check(f"... and the control conserves |v|^2 (max rel drift "
                f"{float(np.max(vsq_rel)):.2e} <= 1e-12)", float(np.max(vsq_rel)) <= 1e-12)


# ---------------------------------------------------- gate 6: the solver is untouched

_CONFIG_BY_NAME = {name: (kwargs, driver) for name, kwargs, driver in CONFIGS}

# gate-6 particle config: the gate-4 ensembles at a small n -- the point is the SOLVER's
# bitwise output, so the particles only have to be live, not statistically meaningful
_G6_PARTICLES = {"seed": 0, "n": 256, "ensembles": _LIVE_ENSEMBLES}

_G6_ARRAYS = ("fields_final", "forcing_state_final", "forcing_scale_final",
              "forcing_key_final", "t_final")


@functools.lru_cache(maxsize=None)
def _reference():
    """The committed pre-A2 npz plus whether this host can be expected to reproduce it
    bitwise (same machine, jax, backend and python)."""
    ref = np.load(reference_path(), allow_pickle=False)
    host_ok = (str(ref["hostname"]) == platform.node()
               and str(ref["jax_version"]) == jax.__version__
               and str(ref["platform"]) == jax.default_backend()
               and str(ref["python_version"]) == platform.python_version())
    return ref, host_ok


@functools.lru_cache(maxsize=None)
def _off_result(name):
    kwargs, driver = _CONFIG_BY_NAME[name]
    return run_config(kwargs, driver)


@functools.lru_cache(maxsize=None)
def _on_result(name):
    """run_config's twin with particles on. Mirrors it exactly for the SimulationState
    outputs, so the comparison against the off run is apples to apples; ctx() cannot key
    its cache on a particles dict, so the Parameters come from fresh_params (which builds
    the same object otherwise)."""
    kwargs, driver = _CONFIG_BY_NAME[name]
    params = fresh_params(particles=_G6_PARTICLES, **kwargs)
    kgrid = jr.setup_kgrids(params)
    state0 = make_state(params, ic=_gate6_ic)
    pstate0 = init_particles(params)
    # the whole (state, pstate) carry is donated, so keep host copies of the initial
    # particle draw for the round-trip check below
    p0 = (np.asarray(pstate0.x).copy(), np.asarray(pstate0.v).copy())
    with snap_dir() as d, managed_manager(params, d, nsnap=10) as mngr:
        if driver == "scan":
            nblock = NBLOCK_CFL if params.cfl_every > 1 else NBLOCK_SCAN
            end, pend = jr.simulate_scan(state0, kgrid, params, nblock, T_SNAP, T_END, mngr,
                                         save=True, pstate=pstate0)
        else:
            end, pend = jr.simulate(state0, kgrid, params, T_SNAP, T_END, mngr, save=True,
                                    pstate=pstate0)
        steps = get_saved_steps(d)
        first_step = min(steps)
        listing = _snapshot_skeleton(os.path.join(d, str(first_step)))
        leaves = _state_leaves(d, first_step, params)
        n_snap_files = len(steps)
        every_step_has_particles = all(
            os.path.isdir(os.path.join(d, str(s), "particles")) for s in steps)
        # the particle item must not disturb the state item's own on-disk tree
        state_item_listing = _snapshot_skeleton(os.path.join(d, str(first_step), "default"))
        side = os.path.join(d, "particle_moments.txt")
        nrows = len(open(side).read().splitlines()) - 1 if os.path.exists(side) else None
        restored = load_particles(first_step, d, params)
        initial_roundtrip = (np.array_equal(np.asarray(restored.x), p0[0])
                             and np.array_equal(np.asarray(restored.v), p0[1])
                             and restored.x.dtype == jnp.float64)
    return {
        "fields_final": np.asarray(end.fields),
        "forcing_state_final": np.asarray(end.forcing_state),
        "forcing_scale_final": np.asarray(end.forcing_scale),
        "forcing_key_final": np.asarray(jax.random.key_data(end.forcing_key)),
        "t_final": np.asarray(float(end.t)),
        "snapshot_listing": listing,
        "state_item_listing": state_item_listing,
        "state_leaves": leaves,
        "n_snapshots": n_snap_files,
        "px": np.asarray(pend.x),
        "pv": np.asarray(pend.v),
        "every_step_has_particles": every_step_has_particles,
        "initial_roundtrip": initial_roundtrip,
        "nrows": nrows,
        "n_ens": params.n_ens,
    }


def test_solver_output_matches_the_pre_a2_reference():
    """Gate 6a: with params.particles=None every run.py / snapshot_io.py path must be
    STATICALLY identical to pre-A2 main -- fields, forcing state, forcing key, the
    snapshot layout and the state item's own leaf list, bitwise, for all five recorded
    driver/dt configurations.

    Bitwise reproduction is host-specific, so the npz comparison only runs when the
    recorded machine/jax/backend/python match (print + return otherwise, per the house
    convention; gates 6b and 6c are separate tests and still run)."""
    ref, host_ok = _reference()
    if not host_ok:
        print(f"[SKIP] gate 6a npz comparison -- reference recorded on "
              f"{ref['hostname']} / jax {ref['jax_version']} / {ref['platform']} / py "
              f"{ref['python_version']}, this host is {platform.node()} / jax "
              f"{jax.__version__} / {jax.default_backend()} / py "
              f"{platform.python_version()} (bitwise reproduction is host-specific)")
        return
    print(f"gate 6a: comparing against {os.path.basename(reference_path())} "
          f"(git {str(ref['git_commit'])[:9]}, precision {ref['precision']})")
    with checks() as c:
        for name, _kwargs, _driver in CONFIGS:
            off = _off_result(name)
            for key in _G6_ARRAYS + ("snapshot_listing", "state_leaves"):
                c.check(f"{name}: {key} bitwise identical to the reference",
                        np.array_equal(off[key], ref[f"{name}_{key}"]),
                        f"{off[key]!r} vs {ref[f'{name}_{key}']!r}")
            c.check(f"{name}: same number of snapshots as the reference",
                    off["n_snapshots"] == int(ref[f"{name}_n_snapshots"]),
                    f"{off['n_snapshots']} vs {int(ref[f'{name}_n_snapshots'])}")


def test_particles_on_leaves_the_solver_bitwise_identical():
    """Gate 6b: turning particles ON must not perturb the solver by a single bit. The
    particle RNG is its own stream and the push never writes back, so any difference here
    would mean the carry/scan restructuring changed the solver's own arithmetic. The
    on-disk side is checked the same way: the state item's tree is untouched and the
    particle carry rides the SAME step as a separate `particles/` item."""
    with checks() as c:
        for name, _kwargs, driver in CONFIGS:
            off, on = _off_result(name), _on_result(name)
            for key in _G6_ARRAYS:
                c.check(f"{name}: {key} bitwise identical with particles on",
                        np.array_equal(off[key], on[key]))
            c.check(f"{name}: the state item's leaves are unchanged (the particle carry "
                    f"is NOT a new SimulationState field)",
                    np.array_equal(off["state_leaves"], on["state_leaves"]),
                    f"{on['state_leaves']} vs {off['state_leaves']}")
            c.check(f"{name}: same number of snapshots", off["n_snapshots"] == on["n_snapshots"])
            minus = np.array(sorted(e for e in on["snapshot_listing"]
                                    if not e.startswith("particles" + os.sep)))
            c.check(f"{name}: snapshot listing minus particles/ is identical to the off run",
                    np.array_equal(minus, off["snapshot_listing"]),
                    f"{sorted(set(minus) ^ set(off['snapshot_listing']))}")
            c.check(f"{name}: <step>/default/ is byte-structurally the off run's listing",
                    np.array_equal(on["state_item_listing"],
                                   np.array(sorted(e[len("default") + 1:]
                                                   for e in off["snapshot_listing"]
                                                   if e.startswith("default" + os.sep)))),
                    str(on["state_item_listing"]))
            c.check(f"{name}: every written step carries a particles/ item",
                    on["every_step_has_particles"])
            c.check(f"{name}: the initial snapshot's particle state round-trips bitwise",
                    on["initial_roundtrip"])
            c.check(f"{name}: particle state stayed finite",
                    bool(np.all(np.isfinite(on["px"])) and np.all(np.isfinite(on["pv"]))))
            if driver == "scan":
                c.check(f"{name}: sidecar holds a whole number of (step, ensemble) rows",
                        on["nrows"] is not None and on["nrows"] > 0
                        and on["nrows"] % on["n_ens"] == 0, f"rows={on['nrows']}")
            else:
                c.check(f"{name}: the while_loop driver writes no sidecar",
                        on["nrows"] is None)


# restart is bitwise only when the driver's start-up does not recompute anything: with
# forcing_norm_per_step=True (the default) `run._refresh_forcing_scale` recomputes the
# forcing scale with dt = 0 at every simulate*/ call, so even a PARTICLE-FREE restart is
# not bitwise. That is pre-existing solver behaviour, not A2, so this gate runs forcing
# live with forcing_norm_per_step=False (the per-stage normalization path).
_RESTART_KWARGS = dict(_CONFIG_BY_NAME["scan_fixed"][0], forcing_norm_per_step=False)
_RESTART_T_SNAP = 0.04     # < the 0.05 a 5-step block covers, so every block snapshots


def test_restart_continues_fields_and_trajectories_bitwise():
    """Gate 6c: load_snapshot + load_particles from a mid-run snapshot and continuing must
    reproduce the uninterrupted run bitwise in BOTH the fields and the particle
    trajectories -- the point of storing the carry at snapshot cadence. Also the
    missing-item policy: a snapshot written without particles is never silently re-inited;
    it is a FileNotFoundError naming init_on_restart, and only that flag turns it into a
    fresh ensemble -- while a snapshot step that does not exist at all stays a
    FileNotFoundError whatever the flag says."""
    params = fresh_params(particles=_G6_PARTICLES, **_RESTART_KWARGS)
    kgrid = jr.setup_kgrids(params)
    with snap_dir() as d1, managed_manager(params, d1, nsnap=10) as mngr:
        end, pend = jr.simulate_scan(make_state(params, ic=_gate6_ic), kgrid, params,
                                     NBLOCK_SCAN, _RESTART_T_SNAP, T_END, mngr, save=True,
                                     pstate=init_particles(params))
        ref = (np.asarray(end.fields), np.asarray(pend.x), np.asarray(pend.v), float(end.t),
               np.asarray(pend.w))
        steps = get_saved_steps(d1)
        mid = steps[len(steps) // 2]
        state_mid = load_snapshot(mid, d1, params)
        pstate_mid = load_particles(mid, d1, params)
        t_mid = float(state_mid.t)
        with snap_dir() as d2, managed_manager(params, d2, nsnap=10) as mngr2:
            end2, pend2 = jr.simulate_scan(state_mid, kgrid, params, NBLOCK_SCAN,
                                           _RESTART_T_SNAP, T_END, mngr2, save=True,
                                           pstate=pstate_mid)
            got = (np.asarray(end2.fields), np.asarray(pend2.x), np.asarray(pend2.v),
                   float(end2.t), np.asarray(pend2.w))

    # a snapshot written by a particle-free run: the missing-item policy
    off_params, off_kgrid = config_ctx(_CONFIG_BY_NAME["scan_fixed"][0])
    with snap_dir() as d3, managed_manager(off_params, d3, nsnap=10) as mngr3:
        jr.simulate_scan(make_state(off_params, ic=_gate6_ic), off_kgrid, off_params,
                         NBLOCK_SCAN, T_SNAP, T_END, mngr3, save=True)
        step0 = min(get_saved_steps(d3))
        try:
            load_particles(step0, d3, params)
            missing_msg = "no FileNotFoundError"
        except FileNotFoundError as e:
            missing_msg = str(e)
        reinit_params = fresh_params(particles=dict(_G6_PARTICLES, init_on_restart=True),
                                     **_RESTART_KWARGS)
        reinit = load_particles(step0, d3, reinit_params)
        fresh = init_particles(reinit_params)
        # a snapshot index that does not exist at all: init_on_restart must NOT cover it
        try:
            load_particles(max(get_saved_steps(d3)) + 1000, d3, reinit_params)
            absent_msg = "no FileNotFoundError"
        except FileNotFoundError as e:
            absent_msg = str(e)

    print(f"gate 6c: restarted from snapshot {mid} at t = {t_mid} of {ref[3]}")
    with checks() as c:
        c.check(f"the restart lands on the same final time ({got[3]!r} vs {ref[3]!r})",
                got[3] == ref[3])
        c.check("restarted fields are bitwise identical", np.array_equal(got[0], ref[0]))
        c.check("restarted particle positions are bitwise identical",
                np.array_equal(got[1], ref[1]))
        c.check("restarted particle velocities are bitwise identical",
                np.array_equal(got[2], ref[2]))
        c.check("restarted work accumulators are bitwise identical (w rides the same "
                "checkpoint item, so the heating attribution survives a restart)",
                np.array_equal(got[4], ref[4]))
        c.check("a mid-run snapshot really was mid-run (not the final state)",
                0.0 < t_mid < ref[3], f"t_mid={t_mid}, t_end={ref[3]}")
        c.check("restoring a particle-less snapshot raises FileNotFoundError naming "
                "init_on_restart", "init_on_restart" in missing_msg, missing_msg)
        c.check("a MISSING SNAPSHOT is a FileNotFoundError even with init_on_restart=True "
                "(the flag covers a missing particles item, not a missing step)",
                "no snapshot" in absent_msg and "init_on_restart" not in absent_msg,
                absent_msg)
        c.check("init_on_restart=True returns init_particles(params) bitwise instead",
                np.array_equal(np.asarray(reinit.x), np.asarray(fresh.x))
                and np.array_equal(np.asarray(reinit.v), np.asarray(fresh.v)))
        c.check("the restored carry is a ParticleState of fp64 leaves",
                isinstance(pstate_mid, ParticleState)
                and pstate_mid.x.dtype == jnp.float64 and pstate_mid.v.dtype == jnp.float64
                and pstate_mid.w.dtype == jnp.float64)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
