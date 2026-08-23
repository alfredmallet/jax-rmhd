# 3D test-particle gates, plans/TESTPART_PLAN.md §6 (Phase B2). Single-process, both z
# modes (finite-difference z and z_spectral).
#
#   4. Canonical p_z   in 3D z is ignorable ONLY where the fields have no z structure, so
#                      the 2D gate carries over as an EMBEDDED-2D run: a 3D box (nz > 1)
#                      whose (phi, psi) start z-independent, with forcing off -- the O-U
#                      z-envelope is what would break z-independence, and the first test
#                      below measures both halves of that claim. Then
#                      p_z = v_z - qm*psi(x,y,t) is exactly conserved again, and the same
#                      four variants run: frozen fields + trilinear gather O(dx^2), frozen
#                      fields + spectral gather O(dt^2), live full mask O(dt), live
#                      ideal-only not convergent. The particles carry v_z, so they cross
#                      the z boundary repeatedly -- one gate covering the trilinear z
#                      interpolation, the 3D assembly and the co-stepping at once.
#   7. E_z assembly    in 3D the solver's dpsi/dt also carries the Alfven term +d_z(phi),
#                      which is NOT part of E_z (it cancels against -B0*d_z(phi) in
#                      E_z = -B0*d_z(phi) + dpsi/dt). So the gate compares the assembled
#                      full-mask E_z against (psi(t+dt) - psi(t-dt))/2dt - d_z(phi), with
#                      d_z(phi) formed the way the solver forms it in each mode
#                      (z_spectral: i*kz*phik with the Nyquist plane zeroed; FD z: the
#                      shared_physics 4th-order z stencil). It converges at the centered
#                      difference's O(dt^2). Dropping the finite-difference-z filter from
#                      the resistive piece leaves an O(z_diss) floor instead.
#   B0 = 1 in 3D       Parameters rejects any other value (tests/test_particles_config.py);
#                      this file shows the rule is physics: the field a particle at
#                      amplitude B0 would see carries E.B = B0*(1-B0)*d_z(phi), and gate
#                      7's residual grows like |1-B0|. The same sweep in 2D leaves E.B at
#                      round-off for every B0.
#   5. E = 0 floor     the control ensemble in a live FORCED, genuinely 3D turbulent run.
#   8. Work closure    the per-piece work accumulator against the kinetic-energy change in
#                      the same run: round-off, and every piece an ensemble's mask omits
#                      exactly zero -- so the non-ideal energization the paired ideal/full
#                      ensembles differ by is measured, in w_ez_resistive and w_ez_forcing,
#                      which are identically zero in the ideal-only twin.
#   9. E_par projection  `epar_project` drives E'.B at the particles to round-off while the
#                      unprojected sample is not zero, orbits and closure still right.
#   6. Solver untouched  particles on vs off bitwise in the same session, a particles/
#                      checkpoint item at every step, and a restart that continues fields
#                      AND trajectories bitwise. No new reference npz: the committed one is
#                      a 2D artifact about the A2 wiring (plans/TESTPART_PLAN.md §6).
#
# Plus the z-specific interpolation checks (nodes, O(dz^2), wrapping far outside [0,Lz),
# agreement with gather_spectral where that path is exact).
#
# Gates 1-3 have no 3D analogue to add HERE: they drive analytic uniform/1-D fields through
# the pusher with no solver, so the kernel file exercises the identical code. The one
# kernel gate that does need a z axis -- gate 10, the mirror force, whose parallel dynamics
# exists only in a field that varies along z -- lives with them in
# tests/test_particles_kernel.py, since no solver is in its loop. Gate 6a (the pre-A2
# reference npz) is 2D-only by construction.
#
# Convergence gates are fp64; gates 5, 6, 8, 9 run in both sessions (the particle state,
# the work arithmetic and the gathered samples are fp64 whatever the field precision).
# Script: `python tests/test_particles_3d.py`.
from _rmhd_testing import (bootstrap, checks, fit_order, fresh_params, managed_manager,
                           snap_dir)

bootstrap()

import functools
import os

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
from taranis.physics import rmhd
from taranis.physics.shared_physics import bracket, gradk, z_derivatives
from taranis.snapshot_io import get_saved_steps, load_particles, load_snapshot
from taranis.timestepping import get_scheme

from _gen_particles_gate6_reference import _snapshot_skeleton, _state_leaves

NZ = 8
LZ = 2 * np.pi
Z_MODES = (False, True)          # params.z_spectral
_ZLABEL = {False: "finite-difference z", True: "z_spectral"}


def _dz_phi(state, kgrid, params):
    """d_z(phi) in real space, formed exactly as the solver forms it: i*kz*phik under
    z_spectral (kz-Nyquist plane zeroed, rmhd._kz_deriv), the 4th-order z stencil of
    shared_physics.z_derivatives otherwise."""
    if params.z_spectral:
        return grids.ifft(1j * rmhd._kz_deriv(kgrid, params) * state.fields[0], params)
    return grids.ifft(z_derivatives(state.fields, params)[0], params)[0]


def _zspread(fields, params):
    # max deviation of the real-space fields from their first z plane
    r = np.asarray(grids.ifft(fields, params))
    return float(np.max(np.abs(r - r[:, :1])))


# ---------------------------------------- the embedded-2D configuration (gate 4 in 3D)
#
# In 3D, p_z = v_z - qm*psi is conserved only where z is ignorable, i.e. only for fields
# with no z structure. A z-independent initial condition stays z-independent under RMHD --
# every z derivative in the solver annihilates it and the nonlinear term of z-independent
# fields is z-independent -- UNLESS the stochastic forcing is on, whose z envelope is
# cos/sin(kz*z) by construction (shared_physics.reconstruct_envelope). The first test
# measures both halves; the gates below then run with forcing off.

_E2D_QM = 5.0
_E2D_VPERP = 1.0
_E2D_VZ = 3.0            # nonzero on purpose: the particles must cross the z boundary
_E2D_B0 = 1.0
_E2D_NPART = 64
_E2D_DT = 0.01           # Omega*dt = 0.05, held fixed across the dx refinement
_E2D_STEPS = 250         # T = 2.5, about 2 gyroperiods
_E2D_NXS = (16, 32, 64)  # rho/dx = 0.51, 1.02, 2.04
_E2D_DTS = (0.02, 0.01, 0.005)      # spectral variant, T held fixed
_E2D_SPEC_NX = 32
_E2D_T = _E2D_DT * _E2D_STEPS

# frozen fields => E_z must be 0 for p_z to be invariant: b_perp and E_perp live, every
# E_z piece off (the same discriminator as the 2D gate)
_E2D_MASK = dict(FIELD_MASK_DEFAULTS, ez_ideal=False)


def _e2d_ic(x, y, z):
    # the 2D gate's band-limited analytic state, broadcast over z: every mode has
    # |n| <= 4.2, inside the 2/3 elliptical cutoff (radius 16/3) of the coarsest grid below
    ones = jnp.ones_like(z)
    phi = 0.5 * (jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y)
                 + 0.1 * jnp.cos(3 * x - 2 * y)) * ones
    psi = 0.2 * (jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)
                 + 0.1 * jnp.sin(4 * x + y)) * ones
    return jnp.stack([phi, psi], axis=0)


def _e2d_params(nx, zs, dt=_E2D_DT, diss=1e-2, particles=None):
    return fresh_params(dims=3, nx=nx, ny=nx, nz=NZ, Lx=2 * np.pi, Ly=2 * np.pi, Lz=LZ,
                        dt=dt, adaptive_timestep=False, diss=diss, hyper=1, cfl_safety=0.5,
                        z_spectral=zs, forcing=False, particles=particles)


def _forced_e2d_params(zs):
    return fresh_params(dims=3, nx=32, ny=32, nz=NZ, Lx=2 * np.pi, Ly=2 * np.pi, Lz=LZ,
                        dt=_E2D_DT, adaptive_timestep=False, diss=1e-2, hyper=1,
                        cfl_safety=0.5, z_spectral=zs, forcing=True,
                        forcing_mode="elsasser", forcing_power_elsasser=(0.5, 0.5),
                        forcing_tau=1.0, fshell=(1, 3))


def _evolve(state, kgrid, params, nsteps):
    # block_of_steps under jax.jit: no donate_argnums here, so the caller's state survives
    stepper, scheme = get_scheme("lsrk33")
    step = jax.jit(run.block_of_steps, static_argnums=(2, 3, 4, 5))
    return step(state, kgrid, params, nsteps, scheme, stepper)


def test_z_independent_fields_stay_z_independent_unless_forced():
    """The premise of every embedded-2D gate below: RMHD preserves z-independence exactly
    (both z modes), and the stochastic forcing destroys it (its z envelope is cos/sin(kz z)
    -- shared_physics.reconstruct_envelope), which is why the gates run unforced. Measured
    rather than assumed: if a solver change ever leaked z structure into an unforced
    z-independent run, gate 4 below would fail for a reason that has nothing to do with
    particles."""
    with checks() as c:
        for zs in Z_MODES:
            params = _e2d_params(32, zs)
            kgrid = jr.setup_kgrids(params)
            state = jr.initialize(_e2d_ic, params)
            spread0 = _zspread(state.fields, params)
            evolved = _evolve(state, kgrid, params, 50)
            spread = _zspread(evolved.fields, params)
            amp = float(jnp.max(jnp.abs(grids.ifft(evolved.fields, params))))
            fparams = _forced_e2d_params(zs)
            fkgrid = jr.setup_kgrids(fparams)
            fstate = run._refresh_forcing_scale(jr.initialize(_e2d_ic, fparams), fkgrid,
                                                fparams)
            fspread = _zspread(_evolve(fstate, fkgrid, fparams, 20).fields, fparams)
            print(f"embedded 2D [{_ZLABEL[zs]}]: z spread {spread0:.1e} at t=0, "
                  f"{spread:.1e} after 50 steps (|fields| ~ {amp:.2f}); forced run "
                  f"{fspread:.2e}")
            c.check(f"[{_ZLABEL[zs]}] an unforced z-independent state stays z-independent "
                    f"({spread:.1e} after 50 steps, fields of order {amp:.2f})",
                    spread <= 1e-12 * amp, f"spread={spread}")
            c.check(f"[{_ZLABEL[zs]}] ... while the O-U forcing breaks it at O(1) "
                    f"({fspread:.2f}), so gate 4 must run unforced", fspread > 0.1,
                    f"fspread={fspread}")


def _e2d_seeds():
    # ring in gyrophase at random positions (z included), drawn on the host so every
    # resolution and every dt starts from bitwise identical particles
    rng = np.random.default_rng(0)
    th = rng.uniform(0.0, 2.0 * np.pi, _E2D_NPART)
    x = np.stack([rng.uniform(0.0, 2.0 * np.pi, _E2D_NPART),
                  rng.uniform(0.0, 2.0 * np.pi, _E2D_NPART),
                  rng.uniform(0.0, LZ, _E2D_NPART)], axis=1)
    v = np.stack([_E2D_VPERP * np.cos(th), _E2D_VPERP * np.sin(th),
                  np.full(_E2D_NPART, _E2D_VZ)], axis=1)
    return jnp.asarray(x), jnp.asarray(v)


@functools.lru_cache(maxsize=None)
def _e2d_frozen(nx, dt, nsteps, zs, spectral=False, sign=-1.0):
    """(max_i |p_z,i(T) - p_z,i(0)|, box lengths of z travelled) for p_z = v_z + sign*qm*psi
    in the frozen fields of a z-independent analytic 3D state. `spectral` swaps BOTH the
    push's gather and the psi sample for interp.gather_spectral, which is exact here in
    both z modes (fully spectral under z_spectral; perp-spectral and linear in z
    otherwise, and the field has no z structure)."""
    params = _e2d_params(nx, zs, dt=dt)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_e2d_ic, params)
    E, B = assemble(particle_fields(state, kgrid, params), _E2D_MASK, B0=_E2D_B0)
    if spectral:
        gather = interp.gather_spectral
        E, B = grids.fft(E, params), grids.fft(B, params)
        psi_field = state.fields[1][None]                        # (1, nz, nkx, nky)
    else:
        gather = interp.gather
        psi_field = grids.ifft(state.fields[1], params)[None]     # (1, nz, nx, ny)

    @jax.jit
    def orbit(x, v):
        def step(c, _):
            return boris.push(c[0], c[1], E, B, _E2D_QM, dt, params, gather=gather), None
        (x, v), _ = jax.lax.scan(step, (x, v), None, length=nsteps)
        return x, v

    def pz(x, v):
        return v[:, 2] + sign * _E2D_QM * gather(psi_field, x, params)[:, 0]

    x0, v0 = _e2d_seeds()
    x1, v1 = orbit(x0, v0)
    travel = float(jnp.max(jnp.abs(x1[:, 2] - x0[:, 2]))) / params.Lz
    return float(jnp.max(jnp.abs(pz(x1, v1) - pz(x0, v0)))), travel


@pytest.mark.fp64
def test_pz_embedded_2d_frozen_trilinear_converges_in_dx():
    """Gate 4a in 3D: frozen fields (so dpsi/dt = 0 and the invariant needs E_z = 0), the
    production trilinear gather, dt fixed -- the drift is interpolation error alone and
    converges at O(dx^2). The particles carry v_z and their positions are left unfolded, so
    they cross the z boundary and both the z fold and the z blend of the gather are live,
    even though a z-independent field makes the blend exact."""
    with checks() as c:
        dxs = [2.0 * np.pi / nx for nx in _E2D_NXS]
        for zs in Z_MODES:
            errs, travels = zip(*[_e2d_frozen(nx, _E2D_DT, _E2D_STEPS, zs)
                                  for nx in _E2D_NXS])
            order = fit_order(dxs, errs)
            ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
            print(f"gate 4a [{_ZLABEL[zs]}]: rho/dx "
                  f"{['%.2f' % ((_E2D_VPERP / _E2D_QM) / d) for d in dxs]} -> max|dp_z| "
                  f"{['%.3e' % e for e in errs]} over T = {_E2D_T}, order {order:.3f}, "
                  f"z travel {max(travels):.2f} box lengths")
            c.check(f"[{_ZLABEL[zs]}] trilinear p_z drift converges at order {order:.2f} "
                    f"in dx (>= 1.5; ratios {['%.2f' % r for r in ratios]})", order >= 1.5,
                    f"errs {['%.3e' % e for e in errs]}")
            c.check(f"[{_ZLABEL[zs]}] ... and at rho/dx = 2 the drift {errs[-1]:.2e} is "
                    f"small next to the p_z scale qm*max|psi| ~ "
                    f"{_E2D_QM * 0.2:.2f}", errs[-1] < 1e-2)
            c.check(f"[{_ZLABEL[zs]}] the particles crossed the z boundary "
                    f"({max(travels):.2f} box lengths of unfolded z travel)",
                    max(travels) > 1.0, f"travel={max(travels)}")


@pytest.mark.fp64
def test_pz_embedded_2d_frozen_spectral_converges_in_dt():
    """Gate 4b in 3D: the SAME push through the exact spectral gather removes the
    interpolation error, leaving only the Boris O(dt^2). A 3D assembly inconsistent with
    b_perp = zhat x grad(psi) -- or a z axis wired into the gather wrongly -- would survive
    dt refinement as a floor."""
    with checks() as c:
        nsteps = [int(round(_E2D_T / dt)) for dt in _E2D_DTS]
        for zs in Z_MODES:
            errs = [_e2d_frozen(_E2D_SPEC_NX, dt, n, zs, spectral=True)[0]
                    for dt, n in zip(_E2D_DTS, nsteps)]
            order = fit_order(_E2D_DTS, errs)
            ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
            tril = _e2d_frozen(_E2D_SPEC_NX, _E2D_DT, _E2D_STEPS, zs)[0]
            print(f"gate 4b [{_ZLABEL[zs]}]: Omega*dt "
                  f"{['%.3f' % (_E2D_QM * dt) for dt in _E2D_DTS]} -> max|dp_z| "
                  f"{['%.3e' % e for e in errs]}, order {order:.3f}")
            c.check(f"[{_ZLABEL[zs]}] spectral-gather p_z drift converges at order "
                    f"{order:.2f} in dt (>= 1.8; ratios {['%.2f' % r for r in ratios]})",
                    order >= 1.8, f"errs {['%.3e' % e for e in errs]}")
            c.check(f"[{_ZLABEL[zs]}] ... and is well below the trilinear drift at the "
                    f"same resolution ({errs[1]:.2e} vs {tril:.2e})", errs[1] < tril)


@pytest.mark.fp64
def test_pz_embedded_2d_sign_pins_bperp_and_ez_against_psi():
    """Gate 4c in 3D: the invariant is v_z - qm*psi, not v_z + qm*psi. The ideal-Ohm
    identity fixes b_perp, E_perp and E_z only relative to each other; this is what pins
    their absolute sign against psi (A_z = -psi) once the z axis is real rather than
    notional."""
    with checks() as c:
        for zs in Z_MODES:
            right = _e2d_frozen(_E2D_NXS[-1], _E2D_DT, _E2D_STEPS, zs)[0]
            wrong = _e2d_frozen(_E2D_NXS[-1], _E2D_DT, _E2D_STEPS, zs, sign=+1.0)[0]
            print(f"gate 4c [{_ZLABEL[zs]}]: max|d(v_z - qm psi)| = {right:.3e} vs "
                  f"max|d(v_z + qm psi)| = {wrong:.3e} (ratio {wrong / right:.0f})")
            c.check(f"[{_ZLABEL[zs]}] v_z - qm*psi is conserved and v_z + qm*psi is not "
                    f"(ratio {wrong / right:.0f} >= 30)", wrong >= 30.0 * right,
                    f"right={right} wrong={wrong}")


def test_trilinear_gather_of_a_z_independent_field_is_the_bilinear_one():
    """The embedded-2D gates rest on this: with every z plane equal, the trilinear blend
    (1-wz)*f + wz*f must return the plane itself, so a 3D run of a z-independent field
    reproduces the 2D answer. Not bitwise -- the blend is two roundings -- but at the last
    ulp."""
    params = _e2d_params(32, False)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_e2d_ic, params)
    F, _ez_on = assemble_stacked(particle_fields(state, kgrid, params), _E2D_MASK,
                                 B0=_E2D_B0)
    params2d = fresh_params(dims=2, nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dt=_E2D_DT,
                            adaptive_timestep=False, diss=1e-2, hyper=1, cfl_safety=0.5)
    x0, _v0 = _e2d_seeds()
    tri = np.asarray(interp.gather(F, x0, params))
    bil = np.asarray(interp.gather(F[:, :1], x0, params2d))
    err = float(np.max(np.abs(tri - bil)))
    scale = float(np.max(np.abs(bil)))
    print(f"trilinear vs bilinear on a z-independent field: max diff {err:.3e} "
          f"(sample scale {scale:.3f})")
    with checks() as c:
        c.check(f"the trilinear gather of a z-independent field is the bilinear gather of "
                f"one plane to {err:.1e} (<= 1e-15 of the sample scale)",
                err <= 1e-15 * scale, f"err={err}")


# ------------------------------------------------- gate 4, live: co-stepped in 3D fields

_LIVE_NX = 64            # rho/dx = 2, the 2D gate's loading
_LIVE_QM = 5.0
_LIVE_VPERP = 1.0
_LIVE_VZ = 10.0          # ~1.7 box lengths of z travel over _LIVE_T
_LIVE_NPART = 256
_LIVE_DTS = (0.01, 0.005, 0.0025)
_LIVE_T = 1.0
# heavy dissipation on purpose: it keeps psi smooth (so the dt-independent interpolation
# floor stays below the O(dt) field-freezing error the gate measures) and makes the
# resistive piece of E_z large enough to separate the ensembles cleanly. Unforced, so the
# fields evolve by the bracket and the dissipation alone and stay z-independent.
_LIVE_DISS = 3e-1

_LIVE_INIT = {"kind": "ring", "vperp": _LIVE_VPERP, "vz": _LIVE_VZ}
_LIVE_ENSEMBLES = [
    {"qm": _LIVE_QM, "init": _LIVE_INIT, "ez_resistive": True, "ez_forcing": True},
    {"qm": _LIVE_QM, "init": _LIVE_INIT},
    {"qm": _LIVE_QM, "init": _LIVE_INIT, "eperp": False, "ez_ideal": False},
]
_LIVE_LABELS = ("full dpsi/dt", "ideal-only (default)", "E = 0 control")


def _pz_per_ensemble(state, pstate, params):
    # p_z = v_z - qm*psi with psi sampled by the SAME trilinear gather the push uses
    psi = grids.ifft(state.fields[1], params)[None]
    return [np.asarray(pstate.v[e][:, 2]
                       - ens["qm"] * interp.gather(psi, pstate.x[e], params)[:, 0])
            for e, ens in enumerate(params.particles["ensembles"])]


@functools.lru_cache(maxsize=None)
def _e2d_live(dt, zs):
    """One co-stepped embedded-2D run of length _LIVE_T through run.block_of_steps'
    (state, pstate) carry. Host arrays only: rms p_z drift per ensemble, the p_z scale and
    the z distance travelled."""
    params = _e2d_params(_LIVE_NX, zs, dt=dt, diss=_LIVE_DISS,
                         particles={"seed": 0, "n": _LIVE_NPART,
                                    "ensembles": _LIVE_ENSEMBLES})
    kgrid = jr.setup_kgrids(params)
    s0 = jr.initialize(_e2d_ic, params)
    p0 = init_particles(params)
    x0 = np.asarray(p0.x)
    a = _pz_per_ensemble(s0, p0, params)
    (sT, pT), _ys = _evolve((s0, p0), kgrid, params, int(round(_LIVE_T / dt)))
    b = _pz_per_ensemble(sT, pT, params)
    return {
        "drift": np.array([float(np.sqrt(np.mean((y - x) ** 2))) for x, y in zip(a, b)]),
        "travel": float(np.max(np.abs(np.asarray(pT.x)[..., 2] - x0[..., 2]))) / params.Lz,
        "zspread": _zspread(sT.fields, params),
        "pz_scale": _LIVE_QM * float(jnp.sqrt(jnp.mean(grids.ifft(s0.fields[1],
                                                                 params) ** 2))),
    }


@pytest.mark.fp64
def test_pz_embedded_2d_live_full_mask_converges_and_ideal_only_does_not():
    """Gate 4, live variant in 3D. The full-dpsi/dt ensemble's only error is that the
    fields are frozen across a step, so its p_z drift is O(dt); the default ideal-only
    ensemble omits real accelerations (the resistive piece -- the forcing one is off here)
    and does not converge at all. The contrast is the gate: it would collapse if a piece of
    the FULL mask were missing or mis-signed in 3D, or if the 3D assembly leaked the
    Alfven d_z(phi) term into E_z."""
    with checks() as c:
        for zs in Z_MODES:
            res = [_e2d_live(dt, zs) for dt in _LIVE_DTS]
            full = [r["drift"][0] for r in res]
            ideal = [r["drift"][1] for r in res]
            o_full, o_ideal = fit_order(_LIVE_DTS, full), fit_order(_LIVE_DTS, ideal)
            ratio = ideal[-1] / full[-1]
            scale, travel = res[-1]["pz_scale"], res[-1]["travel"]
            for lbl, col in zip(_LIVE_LABELS, np.array([r["drift"] for r in res]).T):
                print(f"gate 4 live [{_ZLABEL[zs]}][{lbl}]: rms|dp_z| "
                      f"{['%.3e' % v for v in col]}")
            print(f"gate 4 live [{_ZLABEL[zs]}]: orders full {o_full:.3f}, ideal-only "
                  f"{o_ideal:.3f}; qm*rms(psi) = {scale:.3f}; z travel {travel:.2f} box "
                  f"lengths; field z spread {res[-1]['zspread']:.1e}")
            c.check(f"[{_ZLABEL[zs]}] the live fields stayed z-independent "
                    f"({res[-1]['zspread']:.1e})", res[-1]["zspread"] <= 1e-12)
            c.check(f"[{_ZLABEL[zs]}] full-mask p_z drift converges at order {o_full:.2f} "
                    f"in dt (>= 0.7: the fields are frozen across a step, so O(dt))",
                    o_full >= 0.7, f"drifts {['%.3e' % v for v in full]}")
            c.check(f"[{_ZLABEL[zs]}] ... and is small next to the p_z scale qm*rms(psi) = "
                    f"{scale:.2f} ({full[-1] / scale:.2e} of it)", full[-1] < 0.02 * scale)
            c.check(f"[{_ZLABEL[zs]}] ideal-only p_z drift does NOT converge (order "
                    f"{o_ideal:.2f} < 0.3)", o_ideal < 0.3,
                    f"drifts {['%.3e' % v for v in ideal]}")
            c.check(f"[{_ZLABEL[zs]}] ... and is {ratio:.0f}x the full-mask drift at "
                    f"dt = {_LIVE_DTS[-1]} (>= 50): the omitted resistive acceleration",
                    ratio >= 50.0, f"ideal={ideal[-1]} full={full[-1]}")
            c.check(f"[{_ZLABEL[zs]}] the co-stepped particles crossed the z boundary "
                    f"({travel:.2f} box lengths)", travel > 1.0, f"travel={travel}")


# ------------------------------------------------------------- gate 7: E_z assembly in 3D

_G7_N = 16
_G7_DT_WARM = 0.01
_G7_N_WARM = 5
_G7_DTS = (0.01, 0.005, 0.0025)


def _g7_params(dt, zs):
    # elsasser forcing (the only mode that drives psi) and diss large enough that the
    # resistive piece of E_z is not negligible; genuinely 3D, so d_z(phi) is O(1)
    return fresh_params(dims=3, nx=_G7_N, ny=_G7_N, nz=NZ, Lx=2 * np.pi, Ly=2 * np.pi,
                        Lz=LZ, dt=dt, adaptive_timestep=False, diss=1e-2, hyper=1,
                        cfl_safety=0.5, z_spectral=zs, forcing=True,
                        forcing_mode="elsasser", forcing_power_elsasser=(0.5, 0.5),
                        forcing_tau=1.0, fshell=(1, 3))


def _g7_ic(x, y, z):
    # 3D structure in both fields, plus one perp pair inside the 2/3 elliptical cutoff
    # (radius 16/3 at 16^2) whose product is outside it -- phi's (3,4) beating against
    # psi's (4,-3) gives (7,1) and (-1,7) -- so the dealiasing of the ideal E_z piece is a
    # measurable effect here, not a round-off one
    phi = (jnp.cos(x) * jnp.cos(y) * jnp.cos(z) + 0.3 * jnp.sin(2 * x + y)
           + 0.2 * jnp.sin(x) * jnp.sin(z) + 0.03 * jnp.sin(3 * x) * jnp.cos(4 * y))
    psi = (jnp.sin(x) * jnp.cos(y) * jnp.sin(z) + 0.2 * jnp.cos(x - 2 * y)
           + 0.1 * jnp.cos(y) * jnp.cos(z) + 0.03 * jnp.cos(4 * x - 3 * y))
    return jnp.stack([phi, psi], axis=0)


def _g7_warm(params, kgrid):
    # a developed state with live forcing; not donated (jax.jit without donate_argnums)
    state = run._refresh_forcing_scale(jr.initialize(_g7_ic, params), kgrid, params)
    return _evolve(state, kgrid, params, _G7_N_WARM)


def _g7_raw_stepper(params, kgrid):
    # the RAW stepper: no run._advance_forcing, so forcing_state/forcing_scale stay frozen
    # across the two steps and psi(t) is smooth (what a centered difference needs)
    recipe = equation_registry[params.eqtype]
    stepper, scheme = get_scheme("lsrk33")
    rhs = construct_rhs(recipe)
    return jax.jit(lambda s: stepper(s, kgrid, params, rhs, recipe.set_timestep_func,
                                     scheme))


def _unfiltered(pf, state, params):
    """pf with the finite-difference-z d4/dz4 filter TAKEN BACK OUT of the resistive piece:
    _psi_non_ideal subtracts it, so adding it back leaves the k-local psi diagonal alone --
    the discriminator for whether the fold is load-bearing."""
    filt = params.z_diss * (params.dz / 2) ** 4 * z_derivatives(state.fields, params)[1][1]
    return pf._replace(ez_resistive=pf.ez_resistive + grids.ifft(filt, params))


@functools.lru_cache(maxsize=None)
def _g7_residual(dt, zs, unfiltered=False, B0=1.0):
    """max|E_z(full mask) - ((psi(s2) - psi(s0))/2dt - B0*d_z(phi))| at the midpoint state,
    with the E_z scale and max|d_z(phi)|. In 3D the solver's dpsi/dt carries the Alfven
    term +d_z(phi), which is not part of E_z = -B0*d_z(phi) + dpsi/dt."""
    params = _g7_params(dt, zs)
    kgrid = jr.setup_kgrids(params)
    # warm up at _G7_DT_WARM for every dt, so the three runs share the same s0 fields
    warm_params = _g7_params(_G7_DT_WARM, zs)
    s0 = _g7_warm(warm_params, jr.setup_kgrids(warm_params))
    step = _g7_raw_stepper(params, kgrid)
    s1 = step(s0)
    s2 = step(s1)
    pf = particle_fields(s1, kgrid, params, resistive=True, forcing=True)
    if unfiltered:
        pf = _unfiltered(pf, s1, params)
    E, _B = assemble(pf, full_mask())
    psi = lambda s: grids.ifft(s.fields[1], params)
    fd = (psi(s2) - psi(s0)) / (2.0 * dt)
    dzphi = _dz_phi(s1, kgrid, params)
    err = float(jnp.max(jnp.abs(E[2] - (fd - B0 * dzphi))))
    return err, float(jnp.max(jnp.abs(E[2]))), float(jnp.max(jnp.abs(dzphi)))


@pytest.mark.fp64
def test_ez_matches_dpsi_dt_minus_dz_phi_3d():
    """Gate 7 in 3D: with every piece on, E_z is dpsi/dt - d_z(phi) to the centered
    difference's O(dt^2), in both z modes. The subtracted d_z(phi) is the Alfven term the
    SOLVER adds to dpsi/dt, which cancels against -B0*d_z(phi) in the particle's
    E_z = -B0*d_z(phi) + dpsi/dt at B0 = 1; a sign error, a missing piece, an undealiased
    bracket or a mis-formed z derivative all show up as a failure to converge."""
    with checks() as c:
        for zs in Z_MODES:
            errs, scales, dzs = zip(*[_g7_residual(dt, zs) for dt in _G7_DTS])
            order = fit_order(_G7_DTS, errs)
            ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
            print(f"gate 7 [{_ZLABEL[zs]}]: max|E_z - (dpsi/dt - d_z phi)| "
                  f"{['%.3e' % e for e in errs]} -> order {order:.3f}; max|E_z| "
                  f"{scales[0]:.3e}, max|d_z phi| {dzs[0]:.3e}")
            c.check(f"[{_ZLABEL[zs]}] full-mask E_z converges to the centered difference "
                    f"at order {order:.2f} (>= 1.8; ratios "
                    f"{['%.2f' % r for r in ratios]})", order >= 1.8,
                    f"errs {['%.3e' % e for e in errs]}")
            c.check(f"[{_ZLABEL[zs]}] ... and the error is small next to max|E_z| = "
                    f"{scales[0]:.3e} (rel {errs[0] / scales[0]:.2e} at dt={_G7_DTS[0]})",
                    errs[0] < 1e-2 * scales[0])
            c.check(f"[{_ZLABEL[zs]}] the Alfven term is a real O(1) part of dpsi/dt "
                    f"(max|d_z phi| = {dzs[0]:.2f}), so subtracting it is not cosmetic",
                    dzs[0] > 0.1 * scales[0], f"dzphi={dzs[0]} Ez={scales[0]}")


@pytest.mark.fp64
def test_fd_z_filter_belongs_to_the_resistive_piece():
    """B1's decision, measured: with finite-difference z the solver's dpsi/dt also carries
    FDLinearTerm's -z_diss*(dz/2)^4 d4psi/dz4, which lives outside the k-local L. Folded
    into ez_resistive (fields._psi_non_ideal) gate 7 converges; left out, the comparison
    stops converging at an O(z_diss) floor. Under z_spectral the analogous -z_diss_k*kz^4
    is already inside the diagonal, so there is nothing to fold and nothing to test."""
    full = [_g7_residual(dt, False)[0] for dt in _G7_DTS]
    errs = [_g7_residual(dt, False, unfiltered=True)[0] for dt in _G7_DTS]
    scale = _g7_residual(_G7_DTS[0], False)[1]
    order = fit_order(_G7_DTS, errs)
    print(f"gate 7 filter: without the d4/dz4 filter {['%.3e' % e for e in errs]} "
          f"(order {order:.3f}, rel {errs[-1] / scale:.2e} of max|E_z| = {scale:.3f}); "
          f"with it {['%.3e' % e for e in full]} (rel {full[-1] / scale:.2e})")
    with checks() as c:
        c.check(f"without the filter the comparison does NOT converge (order "
                f"{order:.2f} < 0.5; errs {['%.3e' % e for e in errs]})", order < 0.5)
        c.check(f"... and its floor {errs[-1]:.2e} is a visible O(z_diss) defect, "
                f"{errs[-1] / scale:.1e} of max|E_z| -- not round-off",
                errs[-1] > 1e-5 * scale, f"floor={errs[-1]} scale={scale}")
        c.check(f"... {errs[-1] / full[-1]:.0f}x the filtered residual at dt = "
                f"{_G7_DTS[-1]} (>= 10)", errs[-1] > 10.0 * full[-1],
                f"unfiltered={errs[-1]} filtered={full[-1]}")


# ------------------------------------------------------- why B0 must be 1 in 3D
#
# Parameters rejects B0 != 1 whenever dims == 3 (tests/test_particles_config.py). The rule
# is physics, not taste: the particle's E_z = -B0*d_z(phi) + dpsi/dt collapses to the ideal
# -{phi,psi} only because the solver supplies +1*d_z(phi). At any other B0 the field a
# particle would see keeps (1-B0)*d_z(phi), so it is no longer ideal-Ohm -- and no
# assembly can repair it, since d_z(phi) is not one of the pieces.

_B0_SWEEP = (1.0, 1.5, 2.0, 10.0)


def _b0_ic_2d(x, y):
    return _g7_ic(x, y, jnp.zeros_like(x))


def _b0_eb(dims, zs=False):
    """max|E.B| of the ideal-Ohm field at each B0 of _B0_SWEEP, with E_z taken from the
    DERIVATION (-B0*d_z(phi) + dpsi/dt, whose ideal + Alfven part is -{phi,psi} + d_z(phi))
    and the raw undealiased bracket, so the identity is exact by construction at B0 = 1."""
    if dims == 3:
        params = _g7_params(_G7_DT_WARM, zs)
        state = jr.initialize(_g7_ic, params)
    else:
        params = fresh_params(dims=2, nx=_G7_N, ny=_G7_N, Lx=2 * np.pi, Ly=2 * np.pi,
                              dt=_G7_DT_WARM, adaptive_timestep=False, diss=1e-2, hyper=1,
                              cfl_safety=0.5, forcing=False)
        state = jr.initialize(_b0_ic_2d, params)
    kgrid = jr.setup_kgrids(params)
    pf = particle_fields(state, kgrid, params)
    g = grids.ifft(gradk(state.fields[:2], kgrid), params)
    ez_raw = -bracket(g[0], g[1])
    dzphi = _dz_phi(state, kgrid, params) if dims == 3 else jnp.zeros_like(ez_raw)
    out = {}
    for B0 in _B0_SWEEP:
        E, B = assemble(pf, dict(FIELD_MASK_DEFAULTS), B0=B0)
        ez = ez_raw + (1.0 - B0) * dzphi
        resid = float(jnp.max(jnp.abs(E[0] * B[0] + E[1] * B[1] + ez * B[2])))
        scale = (float(jnp.max(jnp.abs(E[:2]))) * float(jnp.max(jnp.abs(B[:2]))))
        out[B0] = (resid, scale)
    return out, float(jnp.max(jnp.abs(dzphi)))


@pytest.mark.fp64
def test_B0_must_be_one_in_3d():
    """The discriminator behind the config rule, in two independent forms. (a) On the grid:
    the ideal-Ohm identity E.B = 0 holds at B0 = 1 and fails by exactly B0*|1-B0|*|d_z phi|
    otherwise -- while the SAME sweep in 2D stays at round-off for every B0, because there
    is no Alfven term to leave behind. (b) Through gate 7: the assembled E_z carries no B0
    at all, so the residual against the derivation's -B0*d_z(phi) + dpsi/dt grows like
    |1-B0|."""
    with checks() as c:
        for zs in Z_MODES:
            out, mdz = _b0_eb(3, zs)
            for B0, (resid, scale) in out.items():
                print(f"B0 [{_ZLABEL[zs]}] B0={B0:5.1f}: max|E.B| {resid:.3e} "
                      f"(|E||B| ~ {scale:.3e}), B0*|1-B0|*max|d_z phi| = "
                      f"{B0 * abs(1 - B0) * mdz:.3e}")
            r1, s1 = out[1.0]
            c.check(f"[{_ZLABEL[zs]}] at B0 = 1 the 3D field is ideal-Ohm: max|E.B| = "
                    f"{r1:.2e} (<= 1e-12 * {s1:.2e})", r1 <= 1e-12 * s1, f"resid={r1}")
            for B0 in _B0_SWEEP[1:]:
                resid, scale = out[B0]
                predicted = B0 * abs(1.0 - B0) * mdz
                c.check(f"[{_ZLABEL[zs]}] at B0 = {B0} it is not, and the defect is "
                        f"exactly B0*(1-B0)*d_z(phi) ({resid:.3e} vs {predicted:.3e})",
                        abs(resid - predicted) <= 1e-10 * predicted,
                        f"resid={resid} predicted={predicted}")
        out2d, mdz2d = _b0_eb(2)
        worst = max(resid / scale for resid, scale in out2d.values())
        print(f"B0 [2D]: max|d_z phi| = {mdz2d:.1e}, worst max|E.B|/(|E||B|) over "
              f"{list(_B0_SWEEP)} = {worst:.2e}")
        c.check(f"in 2D every B0 leaves the field ideal-Ohm (worst relative E.B "
                f"{worst:.1e} <= 1e-14), which is why B0 stays a free per-ensemble knob "
                f"there", worst <= 1e-14, f"worst={worst}")
        # (b) gate 7's residual against the derivation at each B0
        base = _g7_residual(_G7_DTS[0], False)[0]
        dzphi = _g7_residual(_G7_DTS[0], False)[2]
        for B0 in _B0_SWEEP[1:]:
            err = _g7_residual(_G7_DTS[0], False, B0=B0)[0]
            predicted = abs(1.0 - B0) * dzphi
            print(f"B0 [gate 7]: B0={B0:5.1f} residual {err:.3e} vs "
                  f"|1-B0|*max|d_z phi| = {predicted:.3e} (B0 = 1: {base:.3e})")
            c.check(f"gate 7's residual at B0 = {B0} is the dropped (1-B0)*d_z(phi) "
                    f"({err:.2e}, within 10% of {predicted:.2e}) and dwarfs the B0 = 1 "
                    f"residual {base:.2e}",
                    abs(err - predicted) <= 0.1 * predicted and err > 100.0 * base,
                    f"err={err} predicted={predicted} base={base}")


# --------------------------------- gates 5, 8 and 9: one live FORCED, genuinely 3D run

_F3D_NX = 32
_F3D_DT = 0.005
_F3D_WARM = 40           # forced spin-up before the particles are inserted
_F3D_STEPS = 100
_F3D_QM = 5.0
_F3D_VPERP = 1.0
_F3D_NPART = 128
_F3D_INIT = {"kind": "ring", "vperp": _F3D_VPERP, "vz": 1.0}
# the three standard ensembles of plan §3, plus one with b_perp off (E live, B = B0 zhat)
# and the projected twin of the default -- five (nez, eperp, bperp, project) combinations
# through one accumulator
_F3D_ENSEMBLES = [
    {"qm": _F3D_QM, "init": _F3D_INIT, "ez_resistive": True, "ez_forcing": True},
    {"qm": _F3D_QM, "init": _F3D_INIT},
    {"qm": _F3D_QM, "init": _F3D_INIT, "eperp": False, "ez_ideal": False},
    {"qm": _F3D_QM, "init": _F3D_INIT, "bperp": False},
    {"qm": _F3D_QM, "init": _F3D_INIT, "epar_project": True},
]
_F3D_LABELS = ("full dpsi/dt", "ideal-only (default)", "E = 0 control", "b_perp = 0",
               "ideal-only, projected")


def _f3d_params(zs, particles=None):
    return fresh_params(dims=3, nx=_F3D_NX, ny=_F3D_NX, nz=NZ, Lx=2 * np.pi, Ly=2 * np.pi,
                        Lz=LZ, dt=_F3D_DT, adaptive_timestep=False, diss=1e-1, hyper=1,
                        cfl_safety=0.5, z_spectral=zs, forcing=True,
                        forcing_mode="elsasser", forcing_power_elsasser=(0.5, 0.5),
                        forcing_tau=1.0, fshell=(1, 3), particles=particles)


@functools.lru_cache(maxsize=None)
def _f3d_run(zs):
    """One co-stepped run in live forced 3D turbulence, every ensemble started from ONE
    draw so the paired ideal/full and projected/unprojected comparisons differ by the mask
    alone. Host arrays only."""
    params = _f3d_params(zs, particles={"seed": 0, "n": _F3D_NPART,
                                        "ensembles": _F3D_ENSEMBLES})
    kgrid = jr.setup_kgrids(params)
    # a developed forced state (particles inserted after the spin-up): the O-U z envelope
    # is what puts real z structure into the fields
    warm_params = _f3d_params(zs)
    warm_kgrid = jr.setup_kgrids(warm_params)
    s0 = _evolve(run._refresh_forcing_scale(jr.initialize(_g7_ic, warm_params), warm_kgrid,
                                            warm_params), warm_kgrid, warm_params,
                 _F3D_WARM)
    p0 = init_particles(params)
    n_ens = params.n_ens
    p0 = p0._replace(x=jnp.stack([p0.x[0]] * n_ens), v=jnp.stack([p0.v[0]] * n_ens))
    v0 = np.asarray(p0.v)
    (sT, pT), _ys = _evolve((s0, p0), kgrid, params, _F3D_STEPS)
    # the ideal-Ohm (E, B) the paired ensembles ride, sampled at their end positions
    F, _ez_on = assemble_stacked(particle_fields(sT, kgrid, params),
                                 dict(FIELD_MASK_DEFAULTS), B0=1.0)
    project = jax.vmap(boris.project_perp, in_axes=(0, 0))
    dots = {}
    for e in (1, 4):
        s = interp.gather(F, pT.x[e], params)
        E, B = s[:, :3], s[:, -3:]
        Ep = np.asarray(project(E, B))
        E, B = np.asarray(E), np.asarray(B)
        dots[e] = {"eb": np.sum(E * B, axis=1), "eb_proj": np.sum(Ep * B, axis=1),
                   "scale": np.linalg.norm(E, axis=1) * np.linalg.norm(B, axis=1)}
    return {
        "w": np.asarray(pT.w),
        "dke": 0.5 * np.sum(np.asarray(pT.v) ** 2, axis=2) - 0.5 * np.sum(v0 ** 2, axis=2),
        "ke0": float(np.mean(0.5 * np.sum(v0 ** 2, axis=2))),
        "vsq_rel": np.abs(np.sum(np.asarray(pT.v) ** 2, axis=2)
                          / np.sum(v0 ** 2, axis=2) - 1.0),
        "x": np.asarray(pT.x), "dots": dots,
        "masks": [e["mask"] for e in params.particles["ensembles"]],
        "zspread": _zspread(sT.fields, params),
    }


def test_e_zero_ensemble_is_the_heating_floor_3d():
    """Gate 5 in 3D: in a live FORCED, genuinely 3D turbulent run the E = 0 control
    ensemble (b_perp live, no electric field) can do no work, and the Boris rotation is
    norm-exact -- so |v| is conserved to round-off through the whole 3D pipeline, trilinear
    gather of a time-varying b_perp included. That is the numerical-heating floor every 3D
    heating claim is quoted against."""
    with checks() as c:
        for zs in Z_MODES:
            res = _f3d_run(zs)
            floor = float(np.max(res["vsq_rel"][2]))
            others = [float(np.mean(res["vsq_rel"][e])) for e in (0, 1, 3, 4)]
            print(f"gate 5 [{_ZLABEL[zs]}]: E=0 ensemble max per-particle |v|^2 relative "
                  f"drift = {floor:.3e}; mean |v|^2 growth of the E-carrying ensembles "
                  f"{['%.2f' % v for v in others]}; field z spread "
                  f"{res['zspread']:.2f} (a genuinely 3D run)")
            c.check(f"[{_ZLABEL[zs]}] E = 0 ensemble conserves |v|^2 per particle to "
                    f"{floor:.2e} (<= 1e-12)", floor <= 1e-12, f"floor={floor}")
            c.check(f"[{_ZLABEL[zs]}] ... while the E-carrying ensembles heat by order "
                    f"unity, so the floor is not a trivially static run",
                    all(v > 0.1 for v in others), str(others))
            c.check(f"[{_ZLABEL[zs]}] the run really is 3D (field z spread "
                    f"{res['zspread']:.2f})", res["zspread"] > 0.1)


def test_work_split_closes_against_the_kinetic_energy_3d():
    """Gate 8 in 3D: per particle the accumulated per-piece work sums to the kinetic-energy
    change. Exact Boris algebra, so the residual is round-off however the 3D fields are
    interpolated and whatever TARANIS_PRECISION is (w and the push arithmetic are fp64
    always)."""
    with checks() as c:
        for zs in Z_MODES:
            res = _f3d_run(zs)
            w, dke, ke0 = res["w"], res["dke"], res["ke0"]
            resid = np.max(np.abs(np.sum(w, axis=2) - dke), axis=1) / ke0
            for lbl, r, m in zip(_F3D_LABELS, resid, np.mean(np.abs(dke), axis=1) / ke0):
                print(f"gate 8 [{_ZLABEL[zs]}][{lbl}]: max|sum w - dKE|/KE0 = {r:.2e}, "
                      f"mean|dKE|/KE0 = {m:.2e}")
            c.check(f"[{_ZLABEL[zs]}] the work split closes to {np.max(resid):.2e} of KE0 "
                    f"for every ensemble (<= 1e-12)", np.max(resid) <= 1e-12, str(resid))
            c.check(f"[{_ZLABEL[zs]}] ... on a run where the E-carrying ensembles actually "
                    f"gained energy, so the closure is not trivial",
                    np.mean(np.abs(dke[0])) > 0.01 * ke0,
                    str(np.mean(np.abs(dke), axis=1)))


def test_work_pieces_track_the_mask_3d():
    """Gate 8's discriminator in 3D: a piece an ensemble does not see contributes EXACTLY
    zero work forever. The paired ideal/full ensembles run on identical fields from an
    identical draw, so the non-ideal energization between them is carried entirely by
    w_ez_resistive and w_ez_forcing, which stay identically zero in the ideal-only twin --
    the direct measurement plans/TESTPART_PLAN.md §1 asks for."""
    with checks() as c:
        for zs in Z_MODES:
            res = _f3d_run(zs)
            w, masks = res["w"], res["masks"]
            off_leak = {}
            for e, mask in enumerate(masks):
                for i, piece in enumerate(WORK_PIECES):
                    if not mask[piece]:
                        off_leak[(_F3D_LABELS[e], piece)] = float(np.max(np.abs(w[e, :, i])))
            ir, ifo = WORK_PIECES.index("ez_resistive"), WORK_PIECES.index("ez_forcing")
            full_res = float(np.mean(np.abs(w[0, :, ir])))
            full_for = float(np.mean(np.abs(w[0, :, ifo])))
            print(f"gate 8 [{_ZLABEL[zs]}]: full-mask mean|w_ez_resistive| = "
                  f"{full_res:.3e}, mean|w_ez_forcing| = {full_for:.3e}; E=0 max|w| = "
                  f"{float(np.max(np.abs(w[2]))):.1e}")
            c.check(f"[{_ZLABEL[zs]}] every piece an ensemble's mask switches off carries "
                    f"exactly zero work",
                    all(v == 0.0 for v in off_leak.values()),
                    str({k: v for k, v in off_leak.items() if v != 0.0}))
            c.check(f"[{_ZLABEL[zs]}] the E = 0 control ensemble's w is identically zero",
                    bool(np.all(w[2] == 0.0)), str(np.max(np.abs(w[2]))))
            c.check(f"[{_ZLABEL[zs]}] the non-ideal pieces are live in the full-mask "
                    f"ensemble and exactly zero in the ideal-only one",
                    full_res > 0.0 and full_for > 0.0
                    and float(np.max(np.abs(w[1, :, [ir, ifo]]))) == 0.0)
            c.check(f"[{_ZLABEL[zs]}] the ideal piece is live in both paired ensembles "
                    f"(they differ only by the non-ideal pieces)",
                    float(np.mean(np.abs(w[0, :, WORK_PIECES.index("ez_ideal")]))) > 0.0
                    and float(np.mean(np.abs(w[1, :, WORK_PIECES.index("ez_ideal")]))) > 0.0)


def test_epar_projection_removes_the_numerical_parallel_field_3d():
    """Gate 9 in 3D: at the particle, the projected sample's E'.B is round-off while the
    unprojected one carries the numerical E_parallel that independent trilinear
    interpolation of E and B leaves. The paired ensembles start from the same draw, so
    their separating is the projection acting, and the work closure survives it."""
    with checks() as c:
        for zs in Z_MODES:
            res = _f3d_run(zs)
            raw, proj = res["dots"][1], res["dots"][4]
            rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
            rel_raw = rms(raw["eb"]) / rms(raw["scale"])
            rel_proj = rms(proj["eb_proj"]) / rms(proj["scale"])
            sep = float(np.max(np.abs(res["x"][4] - res["x"][1])))
            print(f"gate 9 [{_ZLABEL[zs]}]: unprojected rms(E.B)/rms(|E||B|) = "
                  f"{rel_raw:.3e}, projected {rel_proj:.3e}; max|x_proj - x_raw| = "
                  f"{sep:.3e}")
            c.check(f"[{_ZLABEL[zs]}] the projected sample is exactly perpendicular to B: "
                    f"rms(E'.B)/rms(|E||B|) = {rel_proj:.2e} (<= 1e-15)",
                    rel_proj <= 1e-15,
                    f"max|E'.B| = {np.max(np.abs(proj['eb_proj'])):.3e}")
            c.check(f"[{_ZLABEL[zs]}] ... while the unprojected gather leaves a numerical "
                    f"E_parallel: rms(E.B)/rms(|E||B|) = {rel_raw:.2e} (>= 1e-6)",
                    rel_raw >= 1e-6, f"rel_raw={rel_raw}")
            c.check(f"[{_ZLABEL[zs]}] the projection changes the orbits (same draw, "
                    f"max|dx| = {sep:.2e})", sep > 1e-6, f"sep={sep}")
            resid = float(np.max(np.abs(np.sum(res["w"][4], axis=1) - res["dke"][4]))
                          / res["ke0"])
            c.check(f"[{_ZLABEL[zs]}] the work closure survives the projection "
                    f"({resid:.2e} of KE0 <= 1e-12): the work columns credit the PROJECTED "
                    f"field, so the exact half-kick identity is unaffected",
                    resid <= 1e-12, f"resid={resid}")


# ------------------------------------------- gate 6 in 3D: the solver is untouched
#
# No new reference npz: the committed one records pre-A2 2D main and is about the carry
# wiring, which 3D does not touch (run.py needed no edit for B1). What 3D can and must
# show is the rest of gate 6 -- particles on vs off bitwise in the same session, a
# particles/ item at every written step, and a bitwise restart.
#
# run._refresh_forcing_scale computes a scale on simulate*/ entry only for a state whose
# forcing_scale is all zeros, so a forced restart is bitwise at either
# forcing_norm_per_step setting. This gate pins the =False setting; the 2D gate 6c runs
# both.

_G6_NX = 16
_G6_NBLOCK = 5
_G6_T_SNAP = 0.04     # < the 0.05 a 5-step block covers, so every block snapshots
_G6_T_END = 0.2
_G6_PARTICLES = {"seed": 0, "n": 64, "ensembles": _F3D_ENSEMBLES}


def _g6_params(zs, particles=None):
    return fresh_params(dims=3, nx=_G6_NX, ny=_G6_NX, nz=NZ, Lx=2 * np.pi, Ly=2 * np.pi,
                        Lz=LZ, dt=0.01, adaptive_timestep=False, diss=1e-3, hyper=1,
                        cfl_safety=0.5, z_spectral=zs, forcing=True,
                        forcing_mode="elsasser", forcing_power_elsasser=(5e-3, 5e-3),
                        forcing_tau=1.0, fshell=(1, 3), forcing_seed=42,
                        forcing_norm_per_step=False, particles=particles)


def _g6_run(zs, particles):
    """simulate_scan over a 3D run, with or without particles. Returns the solver outputs
    plus (particles on) the snapshot-tree facts gate 6 asserts."""
    params = _g6_params(zs, particles)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_g7_ic, params)
    pstate = init_particles(params) if particles else None
    p0 = (np.asarray(pstate.x).copy(), np.asarray(pstate.v).copy()) if particles else None
    with snap_dir() as d, managed_manager(params, d, nsnap=10) as mngr:
        out = jr.simulate_scan(state, kgrid, params, _G6_NBLOCK, _G6_T_SNAP, _G6_T_END,
                               mngr, save=True, pstate=pstate)
        end, pend = out if particles else (out, None)
        steps = get_saved_steps(d)
        first = min(steps)
        res = {"fields": np.asarray(end.fields),
               "forcing_state": np.asarray(end.forcing_state),
               "forcing_scale": np.asarray(end.forcing_scale),
               "forcing_key": np.asarray(jax.random.key_data(end.forcing_key)),
               "t": float(end.t),
               "listing": _snapshot_skeleton(os.path.join(d, str(first))),
               "state_leaves": _state_leaves(d, first, params),
               "n_snapshots": len(steps)}
        if particles:
            res["px"], res["pv"], res["pw"] = (np.asarray(pend.x), np.asarray(pend.v),
                                               np.asarray(pend.w))
            res["every_step_has_particles"] = all(
                os.path.isdir(os.path.join(d, str(s), "particles")) for s in steps)
            res["state_item_listing"] = _snapshot_skeleton(os.path.join(d, str(first),
                                                                       "default"))
            side = os.path.join(d, "particle_moments.txt")
            res["nrows"] = len(open(side).read().splitlines()) - 1
            restored = load_particles(first, d, params)
            res["initial_roundtrip"] = (np.array_equal(np.asarray(restored.x), p0[0])
                                        and np.array_equal(np.asarray(restored.v), p0[1])
                                        and restored.x.dtype == jnp.float64)
            res["n_ens"] = params.n_ens
    return res


def test_particles_on_leaves_the_3d_solver_bitwise_identical():
    """Gate 6b in 3D: turning particles on must not perturb the 3D solver by a single bit,
    in either z mode -- the particle RNG is its own stream and the push never writes back.
    The on-disk side too: the state item's tree is untouched and the particle carry rides
    the SAME step as a separate particles/ item."""
    with checks() as c:
        for zs in Z_MODES:
            off, on = _g6_run(zs, None), _g6_run(zs, _G6_PARTICLES)
            for key in ("fields", "forcing_state", "forcing_scale", "forcing_key", "t"):
                c.check(f"[{_ZLABEL[zs]}] {key} bitwise identical with particles on",
                        np.array_equal(off[key], on[key]))
            c.check(f"[{_ZLABEL[zs]}] the state item's leaves are unchanged (the particle "
                    f"carry is NOT a new SimulationState field)",
                    np.array_equal(off["state_leaves"], on["state_leaves"]),
                    f"{on['state_leaves']} vs {off['state_leaves']}")
            c.check(f"[{_ZLABEL[zs]}] same number of snapshots",
                    off["n_snapshots"] == on["n_snapshots"])
            minus = np.array(sorted(e for e in on["listing"]
                                    if not e.startswith("particles" + os.sep)))
            c.check(f"[{_ZLABEL[zs]}] snapshot listing minus particles/ is identical to "
                    f"the off run",
                    np.array_equal(minus, off["listing"]),
                    f"{sorted(set(minus) ^ set(off['listing']))}")
            c.check(f"[{_ZLABEL[zs]}] <step>/default/ is byte-structurally the off run's "
                    f"listing",
                    np.array_equal(on["state_item_listing"],
                                   np.array(sorted(e[len("default") + 1:]
                                                   for e in off["listing"]
                                                   if e.startswith("default" + os.sep)))),
                    str(on["state_item_listing"]))
            c.check(f"[{_ZLABEL[zs]}] every written step carries a particles/ item",
                    on["every_step_has_particles"])
            c.check(f"[{_ZLABEL[zs]}] the initial snapshot's particle state round-trips "
                    f"bitwise", on["initial_roundtrip"])
            c.check(f"[{_ZLABEL[zs]}] the sidecar holds a whole number of "
                    f"(step, ensemble) rows",
                    on["nrows"] > 0 and on["nrows"] % on["n_ens"] == 0,
                    f"rows={on['nrows']}")
            c.check(f"[{_ZLABEL[zs]}] particle state stayed finite",
                    bool(np.all(np.isfinite(on["px"])) and np.all(np.isfinite(on["pv"]))))


def test_restart_continues_3d_fields_and_trajectories_bitwise():
    """Gate 6c in 3D: load_snapshot + load_particles from a mid-run snapshot and continuing
    reproduces the uninterrupted 3D run bitwise in the fields, the trajectories and the
    work accumulators."""
    with checks() as c:
        for zs in Z_MODES:
            params = _g6_params(zs, _G6_PARTICLES)
            kgrid = jr.setup_kgrids(params)
            with snap_dir() as d1, managed_manager(params, d1, nsnap=10) as mngr:
                end, pend = jr.simulate_scan(jr.initialize(_g7_ic, params), kgrid, params,
                                             _G6_NBLOCK, _G6_T_SNAP, _G6_T_END, mngr,
                                             save=True, pstate=init_particles(params))
                ref = (np.asarray(end.fields), np.asarray(pend.x), np.asarray(pend.v),
                       float(end.t), np.asarray(pend.w))
                steps = get_saved_steps(d1)
                mid = steps[len(steps) // 2]
                state_mid = load_snapshot(mid, d1, params)
                pstate_mid = load_particles(mid, d1, params)
                t_mid = float(state_mid.t)
                with snap_dir() as d2, managed_manager(params, d2, nsnap=10) as mngr2:
                    end2, pend2 = jr.simulate_scan(state_mid, kgrid, params, _G6_NBLOCK,
                                                   _G6_T_SNAP, _G6_T_END, mngr2, save=True,
                                                   pstate=pstate_mid)
                    got = (np.asarray(end2.fields), np.asarray(pend2.x),
                           np.asarray(pend2.v), float(end2.t), np.asarray(pend2.w))
            print(f"gate 6c [{_ZLABEL[zs]}]: restarted from snapshot {mid} at t = {t_mid} "
                  f"of {ref[3]}")
            c.check(f"[{_ZLABEL[zs]}] the restart lands on the same final time "
                    f"({got[3]!r} vs {ref[3]!r})", got[3] == ref[3])
            c.check(f"[{_ZLABEL[zs]}] restarted fields are bitwise identical",
                    np.array_equal(got[0], ref[0]))
            c.check(f"[{_ZLABEL[zs]}] restarted particle positions are bitwise identical",
                    np.array_equal(got[1], ref[1]))
            c.check(f"[{_ZLABEL[zs]}] restarted particle velocities are bitwise identical",
                    np.array_equal(got[2], ref[2]))
            c.check(f"[{_ZLABEL[zs]}] restarted work accumulators are bitwise identical",
                    np.array_equal(got[4], ref[4]))
            c.check(f"[{_ZLABEL[zs]}] a mid-run snapshot really was mid-run",
                    0.0 < t_mid < ref[3], f"t_mid={t_mid}, t_end={ref[3]}")
            c.check(f"[{_ZLABEL[zs]}] the restored carry is a ParticleState of fp64 leaves",
                    isinstance(pstate_mid, ParticleState)
                    and pstate_mid.x.dtype == jnp.float64
                    and pstate_mid.v.dtype == jnp.float64
                    and pstate_mid.w.dtype == jnp.float64)


# ------------------------------------------------------- the z axis of the interpolation

_GI_NX = 16
_GI_NZS = (8, 16, 32)


def _gi_params(nz, zs=False):
    return fresh_params(dims=3, nx=_GI_NX, ny=_GI_NX, nz=nz, Lx=2 * np.pi, Ly=2 * np.pi,
                        Lz=LZ, dt=0.01, adaptive_timestep=False, diss=1e-2, hyper=1,
                        cfl_safety=0.5, z_spectral=zs, forcing=False)


def _fz(z):
    # a function of z ALONE: the perpendicular interpolation is then exact, so what is
    # measured below is the z blend and nothing else
    return jnp.cos(3.0 * z) + 0.5 * jnp.sin(z) - 0.25 * jnp.cos(2.0 * z)


def _fz_field(params):
    _x, _y, z = interp.grid_coords(params)
    return jnp.broadcast_to(_fz(z).reshape(1, -1, 1, 1),
                            (1, params.nz, params.nx, params.ny))


def test_trilinear_gather_at_nodes_and_across_the_z_boundary():
    """The trilinear gather reproduces the grid values at every collocation point, blends
    the top cell with plane 0, and folds positions many box lengths outside [0, Lz) --
    positions are left UNFOLDED by the pusher, so this fold is the only thing keeping a
    long 3D run on the grid."""
    params = _gi_params(8)
    f = _fz_field(params)
    x, y, z = interp.grid_coords(params)
    kk, ii, jj = np.meshgrid(np.arange(params.nz), np.arange(params.nx),
                             np.arange(params.ny), indexing="ij")
    pos = jnp.stack([x[ii.ravel()], y[jj.ravel()], z[kk.ravel()]], axis=1)
    node_err = float(jnp.max(jnp.abs(interp.gather(f, pos, params)[:, 0]
                                     - f[0].reshape(-1).astype(jnp.float64))))
    # the last z cell blends plane nz-1 with plane 0
    pw = jnp.asarray([[0.0, 0.0, params.Lz - 0.5 * params.dz]], dtype=jnp.float64)
    wrapped = float(interp.gather(f, pw, params)[0, 0])
    expect = 0.5 * (float(f[0, -1, 0, 0]) + float(f[0, 0, 0, 0]))
    # unfolded positions: 7 box lengths up and 5 down must gather what [0, Lz) gathers
    rng = np.random.default_rng(3)
    base = np.stack([rng.uniform(0.0, params.Lx, 64), rng.uniform(0.0, params.Ly, 64),
                     rng.uniform(0.0, params.Lz, 64)], axis=1)
    inside = np.asarray(interp.gather(f, jnp.asarray(base), params))
    shifts = {}
    for nbox in (-5, 7):
        far = base.copy()
        far[:, 2] += nbox * params.Lz
        shifts[nbox] = float(np.max(np.abs(
            np.asarray(interp.gather(f, jnp.asarray(far), params)) - inside)))
    print(f"interp z: node err {node_err:.2e}, wrap {wrapped:.12f} vs {expect:.12f}, "
          f"unfolded {shifts}")
    with checks() as c:
        c.check(f"the trilinear gather reproduces the grid values at the collocation "
                f"points ({node_err:.2e})", node_err < 1e-13, f"err={node_err}")
        c.check("gather at Lz - dz/2 blends the last and first z planes",
                abs(wrapped - expect) < 1e-14, f"{wrapped} vs {expect}")
        c.check(f"positions many box lengths outside [0, Lz) fold to the same sample "
                f"({max(shifts.values()):.2e})", max(shifts.values()) < 1e-13,
                str(shifts))


@pytest.mark.fp64
def test_trilinear_gather_converges_in_dz():
    """Refining nz alone on a smooth field of z alone: the z blend converges at O(dz^2),
    the same rate the perpendicular blend has. A z axis wired with the wrong spacing, or a
    fold that is off by a plane, flattens this."""
    rng = np.random.default_rng(4)
    npos = 128
    pz = rng.uniform(-2.0 * LZ, 3.0 * LZ, npos)     # deliberately unfolded
    pos = jnp.asarray(np.stack([rng.uniform(0.0, 2 * np.pi, npos),
                                rng.uniform(0.0, 2 * np.pi, npos), pz], axis=1))
    exact = np.asarray(_fz(jnp.asarray(pz)))
    dzs, errs = [], []
    for nz in _GI_NZS:
        p = _gi_params(nz)
        errs.append(float(np.max(np.abs(
            np.asarray(interp.gather(_fz_field(p), pos, p)[:, 0]) - exact))))
        dzs.append(p.dz)
    order = fit_order(dzs, errs)
    print(f"interp z: dz {['%.3f' % d for d in dzs]} -> err "
          f"{['%.3e' % e for e in errs]}, order {order:.3f}")
    with checks() as c:
        c.check(f"the trilinear z blend converges at order {order:.3f} (~2)",
                1.8 < order < 2.3, f"errs={errs}")


@pytest.mark.fp64
def test_gather_spectral_agrees_where_it_is_exact():
    """gather_spectral is exact for what the representation IS: fully spectral under
    z_spectral (so a band-limited 3D field is reproduced to round-off), perp-spectral and
    linear in z otherwise (so it is exact for a z-independent field, which is what the
    embedded-2D gate above leans on). Both, against the trilinear gather it validates."""
    rng = np.random.default_rng(5)
    npos = 64
    pos = jnp.asarray(np.stack([rng.uniform(0.0, 2 * np.pi, npos),
                                rng.uniform(0.0, 2 * np.pi, npos),
                                rng.uniform(-LZ, 2 * LZ, npos)], axis=1))
    with checks() as c:
        # (a) z_spectral: a band-limited field of z alone, evaluated exactly
        p = _gi_params(8, zs=True)
        f = _fz_field(p)
        fk = grids.fft(f, p)
        exact = np.asarray(_fz(pos[:, 2]))
        spec = np.asarray(interp.gather_spectral(fk, pos, p)[:, 0])
        tril = np.asarray(interp.gather(f, pos, p)[:, 0])
        spec_err = float(np.max(np.abs(spec - exact)))
        tril_err = float(np.max(np.abs(tril - exact)))
        print(f"interp z: z_spectral gather_spectral err {spec_err:.2e} vs trilinear "
              f"{tril_err:.2e}")
        c.check(f"under z_spectral, gather_spectral reproduces a band-limited 3D field to "
                f"round-off ({spec_err:.2e})", spec_err < 1e-12, f"err={spec_err}")
        c.check(f"... which the trilinear gather does not ({tril_err:.2e})",
                tril_err > 1e-4, f"err={tril_err}")
        # (b) finite-difference z: exact for a z-independent field, where the linear z
        # blend of two identical planes is the plane itself
        p2 = _e2d_params(32, False)
        state = jr.initialize(_e2d_ic, p2)
        real = grids.ifft(state.fields, p2)
        spec2 = np.asarray(interp.gather_spectral(state.fields, pos, p2))
        tril2 = np.asarray(interp.gather(real, pos, p2))
        err2 = float(np.max(np.abs(spec2 - tril2)))
        scale2 = float(np.max(np.abs(tril2)))
        print(f"interp z: FD-z z-independent field, gather_spectral vs trilinear "
              f"{err2:.2e} (interpolation error, scale {scale2:.2f})")
        c.check(f"under finite-difference z, gather_spectral of a z-independent field "
                f"agrees with the trilinear gather to its interpolation error "
                f"({err2:.2e} on a field of order {scale2:.2f})",
                err2 < 0.05 * scale2, f"err={err2}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
