# Run configuration and data generation for examples/test-particles-2D.ipynb: charged test
# particles (plans/TESTPART_PLAN.md phase A3/A3b) in forced 2D RMHD turbulence.
#
# Fields: 256^2, Lx=Ly=2*pi, dims=2, fp64, elsasser O-U forcing in the shell 1 <= |k|/dk < 3
# with forcing_tau=1, adaptive dt (cfl_safety=0.5, cfl_every=1), lsrk33. Dissipation is
# HYPERVISCOUS (hyper=3, nu = eta = NU below, i.e. -nu*k^6), which buys the inertial range
# the xi-scan needs; the resistive piece of E_z is then hyper-resistive and is used only for
# the dpsi/dt-exactness check, never as a physics quantity.
#
# Particles: B0 is the RMHD amplitude parameter (B0 = 1/epsilon, docs/numerics.md "E-parallel
# projection and the amplitude parameter"). Production ensembles run at B0 = B0_PROD with
# q/m = Omega/B0, so Omega = qm*B0 -- and therefore rho, Omega*dt and xi -- is what the
# configuration below actually fixes; beta_i = v_th^2/B0^2. All ideal-Ohm production
# ensembles set epar_project=True (Xia, Perez, Chandran & Quataert 2013, eq. 21), which
# removes the numerical E_parallel left by interpolating E and B independently.
#
# Three scan designs, all driven by make_data():
#   design (a)  Omega sweep at fixed v_th = 1 inside one run: rho moves through the
#               spectrum, the turbulence is fixed. Run at TWO forcing powers (A_HOSTS).
#   design (b)  the same (Omega = OM_REF, v_th = 1) ensemble at four forcing powers, one run
#               each: the particle numerics are fixed, the turbulence moves.
#   design (c)  Xia et al.'s design, riding the design-(a) runs: rho held FIXED at RHO_C
#               (rings at v_perp in VPERP_C with Omega = v_perp/RHO_C), so delta_u(rho) is
#               fixed too and only xi = delta_u/v_perp moves.
# The design-(a) runs also carry the contrast and control ensembles: a B0 = 1 cohort at three
# of the sweep's Omega values (identical rho and xi, delta-B/B0 ten times larger), the
# unprojected twin of the reference ideal ensemble, the full-dpsi/dt ensemble, an E = 0 floor
# control and a delta-b = 0 gyration control.
#
# Dimensionless groups (plans/TESTPART_PLAN.md §2; dx = 2*pi/256 = 0.0245):
#   rho/dx       rho = v_perp/Omega. v_perp,rms = sqrt(2)*v_th = 1.41 for the maxwellian
#                ensembles, so Omega in 5..20 gives rho/dx = 11.5..2.9 at insertion (larger
#                once they heat); design (c) pins rho/dx = 4 at every v_perp. All are above
#                the >= 2 floor bilinear interpolation needs, and every k_rho = 1/rho is
#                below 15, well under the k ~ 29 where the hyperviscous dissipation range
#                starts (the notebook measures both).
#   Omega/omega_nl ~ 1/xi   Omega = 5..41 against an outer-scale rate ~ 0.5; locally
#                Omega*rho/delta_u(rho) = 1/xi, the stochastic-heating parameter.
#   v_th/v_A     = v_th/B0 = 0.1 at B0 = 10 (beta_i = 0.01), 1.0 for the B0 = 1 cohort.
#   delta_B/B0   = 1/B0 by construction: b_rms/B0 = 0.18 at eps_tot = 0.3 for the production
#                ensembles, 1.8 for the contrast cohort.
#   Omega*dt     dt ~ 1.3e-3 at eps_tot = 0.3, so Omega*dt <= 0.06 at the largest Omega = 41:
#                >= 100 solver steps per gyration, no substepping (substeps = 1).
#
# make_data() is resumable, idempotent and wall-clock bounded, like examples/gdi_2d_run.py:
# call it repeatedly (from the notebook or as
# `python -c "from particles_2d_run import make_data; make_data()"`) until it returns True.
# Every phase resumes from its own snapshot directory through the standard
# params.save/get_saved_steps/load_snapshot(+load_particles) contract. Particle runs need
# their OWN directories: params.json records `particles`, and params.save refuses to write a
# differing record into a directory that already holds one. The whole set costs ~45 minutes
# and ~690 MiB on an M1 laptop (CPU, fp64).
import os
import time

import numpy as np
import jax.numpy as jnp

import taranis as jr
import taranis.snapshot_io as sn
from taranis.particles.state import init_particles

NX = NY = 256
LX = LY = 2.0*np.pi
NU = ETA = 1.0e-10
HYPER = 3
FSHELL = (1, 3)
FORCING_TAU = 1.0
FORCING_SEED = 234
SCHEME = "lsrk33"
NBLOCK = 200

DX = LX/NX

# forcing powers of design (b), and the two of them that also host designs (a) and (c).
EPS_LADDER = (0.01, 0.03, 0.1, 0.3)
A_HOSTS = (0.03, 0.3)
EPS_SHOW = 0.3        # the run the notebook shows in detail (spectra, budget, E_z pairs)

# spin-up and particle-window lengths scale with the outer-scale turnover time
# (~eps_tot^(-1/3)), so every run covers the same number of eddy turnovers; the number of
# STEPS is nearly the same for all four because the adaptive dt scales the same way.
T_SPINUP = {0.01: 60.0, 0.03: 42.0, 0.1: 29.0, 0.3: 20.0}
T_PART = {0.01: 90.0, 0.03: 62.0, 0.1: 43.0, 0.3: 30.0}
NSNAP_PART = 25       # snapshots per particle run (t_snap = T_PART/NSNAP_PART)

# A short high-cadence tail on the EPS_SHOW design (a) run, continuing its particles into
# its own directory: snapshot-pair statistics need an interval over which a particle does
# NOT cross the box.
T_FAST = 0.4
T_SNAP_FAST = 0.019
NBLOCK_FAST = 10

NPART = 4096          # particles PER ENSEMBLE
PARTICLE_SEED = 7
VTH = 1.0
B0_PROD = 10.0        # production amplitude: delta_B/B0 = 1/B0 = 0.1
B0_LOW = 1.0          # the contrast cohort: delta_B/B0 = 1
OM_SWEEP = (5.0, 7.5, 10.0, 15.0, 20.0)   # design (a): Omega = qm*B0, fixed v_th
OM_REF = 10.0                              # the ensemble design (b) repeats at every eps_tot
OM_CONTRAST = (5.0, 10.0, 20.0)            # the B0 = 1 twins of these design-(a) points
RHO_C = 4.0*DX                             # design (c): the fixed gyroradius
VPERP_C = (0.5, 1.0, 2.0, 4.0)             # design (c): ring speeds, Omega = v_perp/RHO_C

_MAXW = {"kind": "maxwellian", "vth": VTH}


def _ideal(om, b0, init):
    """An ideal-Ohm production ensemble at gyrofrequency `om` and amplitude B0 = b0, with
    the numerical E_parallel projected out."""
    return {"qm": om/b0, "B0": b0, "init": dict(init), "epar_project": True}


# Design (a): the Omega sweep and design (c)'s fixed-rho rings, all projected ideal-Ohm at
# B0_PROD; then the B0 = 1 contrast cohort (same Omega, so same rho and xi), the unprojected
# twin of the OM_REF ensemble (its difference from the projected one is the numerical
# E_parallel energization), the full-dpsi/dt ensemble (the p_z / E_z-exactness check), the
# E = 0 numerical-heating floor and the delta-b = 0 pure-gyration control.
A_ENSEMBLES = ([_ideal(om, B0_PROD, _MAXW) for om in OM_SWEEP]
               + [_ideal(vp/RHO_C, B0_PROD, {"kind": "ring", "vperp": vp})
                  for vp in VPERP_C]
               + [_ideal(om, B0_LOW, _MAXW) for om in OM_CONTRAST]
               + [{"qm": OM_REF/B0_PROD, "B0": B0_PROD, "init": dict(_MAXW)},
                  {"qm": OM_REF/B0_PROD, "B0": B0_PROD, "init": dict(_MAXW),
                   "ez_resistive": True, "ez_forcing": True},
                  {"qm": OM_REF/B0_PROD, "B0": B0_PROD, "init": dict(_MAXW),
                   "eperp": False, "ez_ideal": False},
                  {"qm": OM_REF/B0_PROD, "B0": B0_PROD, "init": dict(_MAXW),
                   "bperp": False, "eperp": False, "ez_ideal": False}])
A_LABELS = ([f"(a) Omega = {om:g}, B0 = {B0_PROD:g}" for om in OM_SWEEP]
            + [f"(c) ring v_perp = {vp:g}, Omega = {vp/RHO_C:.1f}" for vp in VPERP_C]
            + [f"B0 = 1, Omega = {om:g}" for om in OM_CONTRAST]
            + [f"Omega = {OM_REF:g}, unprojected", f"Omega = {OM_REF:g}, full dpsi/dt",
               "E = 0 control", "delta-b = 0 control"])
# indices into A_ENSEMBLES that the notebook refers to by name
_NA, _NC, _NL = len(OM_SWEEP), len(VPERP_C), len(OM_CONTRAST)
A_IDX = {"sweep": tuple(range(_NA)),
         "rings": tuple(range(_NA, _NA + _NC)),
         "lowB0": tuple(range(_NA + _NC, _NA + _NC + _NL)),
         "ideal_ref": OM_SWEEP.index(OM_REF),
         "unprojected": _NA + _NC + _NL,
         "full": _NA + _NC + _NL + 1,
         "e_zero": _NA + _NC + _NL + 2,
         "gyration": _NA + _NC + _NL + 3}
# the design-(a) point each B0 = 1 ensemble twins
A_TWIN = {A_IDX["lowB0"][j]: OM_SWEEP.index(om) for j, om in enumerate(OM_CONTRAST)}

# Design (b): the reference ideal ensemble, its B0 = 1 twin, the full-mask ensemble and the
# E = 0 control, repeated at every forcing power.
B_ENSEMBLES = [_ideal(OM_REF, B0_PROD, _MAXW), _ideal(OM_REF, B0_LOW, _MAXW),
               {"qm": OM_REF/B0_PROD, "B0": B0_PROD, "init": dict(_MAXW),
                "ez_resistive": True, "ez_forcing": True},
               {"qm": OM_REF/B0_PROD, "B0": B0_PROD, "init": dict(_MAXW),
                "eperp": False, "ez_ideal": False}]
B_LABELS = [f"(b) Omega = {OM_REF:g}, B0 = {B0_PROD:g}", f"B0 = 1, Omega = {OM_REF:g}",
            f"Omega = {OM_REF:g}, full dpsi/dt", "E = 0 control"]
B_IDX = {"ideal_ref": 0, "lowB0": (1,), "full": 2, "e_zero": 3}
B_TWIN = {1: 0}


def particle_config(ensembles):
    return {"seed": PARTICLE_SEED, "n": NPART, "substeps": 1, "B0": B0_PROD,
            "ensembles": ensembles}


def make_params(eps_tot, particles=None):
    # one place for the run configuration -- identical every call, which is what makes
    # params.save's identical-record-is-a-no-op guard the resume path's safety net.
    return jr.Parameters(nx=NX, ny=NY, Lx=LX, Ly=LY, dims=2,
                         eqpars={"diss": (NU, ETA), "hyper": HYPER},
                         cfl_safety=0.5, adaptive_timestep=True, cfl_every=1,
                         forcing=True, forcing_mode="elsasser",
                         forcing_power_elsasser=(eps_tot/2, eps_tot/2),
                         forcing_tau=FORCING_TAU, fshell=FSHELL,
                         forcing_seed=FORCING_SEED, forcing_scale_max=1.0,
                         particles=particles)


def run_dirs(root="data/test-particles-2D"):
    """Every snapshot directory this module writes, by role."""
    return {"base": {e: os.path.join(root, "base_eps%g" % e) for e in EPS_LADDER},
            "a": {e: os.path.join(root, "design_a_eps%g" % e) for e in A_HOSTS},
            "b": {e: os.path.join(root, "design_b_eps%g" % e) for e in EPS_LADDER},
            "fast": os.path.join(root, "design_a_eps%g_fast" % EPS_SHOW)}


def particle_run(design, eps_tot, root="data/test-particles-2D"):
    """(params, snap_path, labels, index map) of one particle run: design "a" (the sweep,
    the rings and the contrast/control ensembles, at the A_HOSTS forcing powers) or design
    "b" (the reference quartet, at every EPS_LADDER forcing power)."""
    ens, labels, idx = ((A_ENSEMBLES, A_LABELS, A_IDX) if design == "a"
                        else (B_ENSEMBLES, B_LABELS, B_IDX))
    return (make_params(eps_tot, particles=particle_config(ens)),
            run_dirs(root)[design][eps_tot], labels, idx)


def _zero_ic(x, y):
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape))


def _drive(params, snap_path, t_end, t_snap, wall_left, seed, chunk_t=None, nsnap=2000,
           nblock=None):
    """Advance one run to t_end, resuming from its own snapshots, in wall-clock-bounded
    chunks. `seed` returns (state, pstate) for a fresh start. Returns (done, wall_used)."""
    t0_wall = time.time()
    params.save(snap_path)   # no-op when the identical record is already there
    kgrid = jr.setup_kgrids(params)
    steps = sn.get_saved_steps(snap_path)
    if steps:
        state = sn.load_snapshot(steps[-1], snap_path, params)
        pstate = (sn.load_particles(steps[-1], snap_path, params)
                  if params.particles is not None else None)
    else:
        state, pstate = seed()
    if float(state.t) >= t_end:
        print(f"  {snap_path}: already at t = {float(state.t):.2f} >= {t_end} "
              f"({len(steps)} snapshots)")
        return True, 0.0
    if chunk_t is None:
        chunk_t = 4.0*t_snap
    mngr = jr.snapshot_manager_setup(params=params, snap_path=snap_path, nsnap=nsnap)
    while float(state.t) < t_end and time.time() - t0_wall < wall_left:
        stop = min(t_end, float(state.t) + chunk_t)
        out = jr.simulate_scan(state, kgrid, params, nblock or NBLOCK, t_snap=t_snap, t_end=stop,
                               mngr=mngr, schemestr=SCHEME, save=True, print_every=10**9,
                               pstate=pstate)
        state, pstate = out if params.particles is not None else (out, None)
    mngr.wait_until_finished()
    done = float(state.t) >= t_end
    wall = time.time() - t0_wall
    print(f"  {snap_path}: t = {float(state.t):.2f} / {t_end} in {wall:.0f}s this call; "
          f"done={done}")
    return done, wall


def make_data(root="data/test-particles-2D", wall_budget=3600.0, only=None):
    """Generate every dataset the notebook needs, resuming wherever the last call stopped.

    Phase order: the four base spin-ups (no particles), then the particle runs restarted
    from each spin-up's final snapshot. Returns True once everything has reached its
    target time; call again after a False. `only` restricts the work to a subset of
    EPS_LADDER (every phase of those runs)."""
    t0 = time.time()
    eps_list = tuple(EPS_LADDER if only is None else only)
    d = run_dirs(root)
    for eps in eps_list:
        print(f"base spin-up, eps_tot = {eps:g} (to t = {T_SPINUP[eps]:g})")
        p = make_params(eps)
        done, _ = _drive(p, d["base"][eps], T_SPINUP[eps], T_SPINUP[eps]/6.0,
                         wall_budget - (time.time() - t0),
                         seed=lambda p=p: (jr.initialize(_zero_ic, p), None))
        if not done:
            return False
    jobs = ([("a", eps) for eps in eps_list if eps in A_HOSTS]
            + [("b", eps) for eps in eps_list])
    for design, eps in jobs:
        params, path, labels, _idx = particle_run(design, eps, root)
        base = d["base"][eps]
        t_end = T_SPINUP[eps] + T_PART[eps]
        print(f"particles, design ({design}), eps_tot = {eps:g}, {len(labels)} ensembles "
              f"x {NPART} (to t = {t_end:g})")

        def seed(params=params, base=base):
            step = sn.get_saved_steps(base)[-1]
            return sn.load_snapshot(step, base, params), init_particles(params)

        done, _ = _drive(params, path, t_end, T_PART[eps]/NSNAP_PART,
                         wall_budget - (time.time() - t0), seed=seed)
        if not done:
            return False
    if EPS_SHOW in eps_list:
        params, src, labels, _idx = particle_run("a", EPS_SHOW, root)
        src_step = sn.get_saved_steps(src)[-1]
        # the tail continues the source run's own final time, which overshoots its target
        t_end = float(sn.load_snapshot(src_step, src, params).t) + T_FAST
        print(f"particles, high-cadence tail of design (a) eps_tot = {EPS_SHOW:g} "
              f"(to t = {t_end:g}, snapshot every {T_SNAP_FAST:g})")

        def seed(params=params, src=src, step=src_step):
            return (sn.load_snapshot(step, src, params),
                    sn.load_particles(step, src, params))

        done, _ = _drive(params, d["fast"], t_end, T_SNAP_FAST,
                         wall_budget - (time.time() - t0), seed=seed,
                         chunk_t=T_FAST, nblock=NBLOCK_FAST)
        if not done:
            return False
    print(f"make_data: complete in {time.time() - t0:.0f}s this call")
    return True
