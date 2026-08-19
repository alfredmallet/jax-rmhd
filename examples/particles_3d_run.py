# Run configuration and data generation for examples/test-particles-3D.ipynb: charged test
# particles (plans/TESTPART_PLAN.md phase B3) in forced 3D RMHD turbulence, with matched 2D
# twin runs for the parallel-energization comparison.
#
# Fields: NX^2 x NZ, Lx=Ly=2*pi, Lz=ASPECT*Lx, dims=3, finite-difference z, fp64, elsasser
# O-U forcing in the perpendicular shell 1 <= |k|/dk < 3 with a kz = +-2*pi/Lz envelope,
# adaptive dt (cfl_safety=0.5, cfl_every=1), lsrk33. Dissipation is LAPLACIAN (hyper=1,
# nu = eta), so the run has a Reynolds number that can be placed on Xia, Perez, Chandran &
# Quataert (2013, ApJ 776, 90) Table 1's c2(Re) trend; a hyper=3 twin at the reference
# forcing bridges to the hyper=3 2D campaign of phase A3b.
#
# Amplitude: in 3D B0 is pinned to 1.0 (taranis writes RMHD with v_A = 1; Parameters rejects
# anything else -- docs/numerics.md "B0 in 3D"). The RMHD amplitude parameter is therefore
# the run's own eps = u_rms/v_A = u_rms, set by the forcing power, and delta_B/B0 = b_rms.
# EPS_LADDER below is a ladder of TARGET u_rms; the achieved value is measured in the
# notebook, and FORCE_C is the one constant to retune if it is off.
#
# Particles: all ideal-Ohm production ensembles set epar_project=True (Xia et al. eq. 21).
# beta_i = v_perp_i^2/v_A^2 = v_perp_i^2 in code units, so in 3D the thermal speed sets
# beta_i and xi together -- the per-ensemble B0 that decoupled them in 2D is unavailable.
#
# Three scan designs, all driven by make_data():
#   design (a)  Omega sweep at fixed v_th inside one run: rho moves through the spectrum,
#               the turbulence is fixed. Run at TWO forcing powers (A_HOSTS).
#   design (b)  the same reference ensemble at every forcing power, one run each: the
#               particle numerics are fixed, the turbulence (and with it eps) moves.
#   design (c)  Xia et al.'s design, riding the design-(a) runs: rho held fixed at RHO_C
#               (rings at v_perp in VPERP_C with Omega = v_perp/RHO_C).
# The design-(a) runs also carry the unprojected twin of the reference ideal ensemble, the
# full-dpsi/dt ensemble, an E = 0 floor control and a delta-b = 0 gyration control.
#
# 2D twins (TWIN_HOSTS): the same particles in 2D RMHD at the same eps, for headline 2. The
# 2D solver has no Alfven term, so eps enters only through the particles' B0 = 1/eps; code
# velocities there are in flow units eps*v_A, which maps the 3D configuration onto the 2D
# one by  B0 = 1/eps,  qm = Omega_3D,  v_th_2D = v_th_3D/eps,  u_rms_2D = 1,
# t_2D = eps*t_3D,  nu_2D = nu_3D/eps -- leaving Omega, rho, xi and beta_i identical.
#
# make_data() is resumable, idempotent and wall-clock bounded, like examples/gdi_2d_run.py:
# call it repeatedly (from the notebook or as
# `python -c "from particles_3d_run import make_data; make_data()"`) until it returns True.
# Every phase resumes from its own snapshot directory through the standard
# params.save/get_saved_steps/load_snapshot(+load_particles) contract. Particle runs need
# their OWN directories: params.json records `particles`, and params.save refuses to write a
# differing record into a directory that already holds one.
#
# The size of the campaign is one config change, not an edit: set TARANIS_P3D_PROFILE to
# "smoke", "mvp" or "full" (see PROFILES) before importing, or pass profile="..." through.
import os
import time

import numpy as np
import jax.numpy as jnp

import taranis as jr
import taranis.snapshot_io as sn
from taranis.particles.state import init_particles

LX = LY = 2.0*np.pi
ASPECT = 6.0             # Lz/Lx; Xia et al. use L_par/L_perp = 6 with u_rms ~ v_A/5
LZ = ASPECT*LX
FSHELL = (1, 3)
FORCING_TAU = 1.0
FORCING_SEED = 234
SCHEME = "lsrk33"        # integrating-factor: the wave path (CLAUDE.md), never IMEX here

# forcing power that reaches a target u_rms: P = FORCE_C*u_rms^3/Lx (constant flux). The
# notebook reports the achieved u_rms and the factor to retune this one number by; the value
# below was calibrated on the "smoke" profile, whose Reynolds number is far below
# production.
FORCE_C = 3.4

# Reynolds number the Laplacian dissipation targets at NX = 256, and how it is scaled down
# with resolution: k_d ~ Re^(3/4), so holding k_d/k_max fixed means Re ~ NX^(4/3).
RE_256 = 6000.0          # Xia et al. run B (256^3, k_d L/2pi = 42, c2 = 0.41)

# design (a): gyroradii in grid cells, at fixed v_th (so beta_i is fixed along the sweep)
RHO_DX_SWEEP = (2.0, 3.0, 4.0, 6.0, 8.0)
RHO_DX_REF = 4.0         # the design-(a) point design (b) repeats at every forcing power
RHO_DX_C = 3.0           # design (c): the fixed gyroradius, in grid cells
VPERP_C_MUL = (0.5, 1.0, 2.0, 4.0)   # design (c) ring speeds, in units of v_perp,rms

PROFILES = {
    # nx, nz, n particles per ensemble, substeps, target-u_rms ladder, design-(a) hosts,
    # the host shown in detail, the hosts with a 2D twin, v_th, spin-up and particle-window
    # lengths in outer-scale turnover times, snapshots per particle run, hyper=3 twin
    "smoke": dict(nx=32, nz=16, npart=256, substeps=1, eps=(0.2, 0.3), a_hosts=(0.3,),
                  show=0.3, twins=(0.3,), vth=0.30, n_spin=1.5, n_part=1.5, nsnap=5,
                  hyper3_twin=False, nblock=25),
    "mvp": dict(nx=128, nz=32, npart=4096, substeps=2, eps=(0.1, 0.2, 0.3), a_hosts=(0.2,),
                show=0.2, twins=(0.2,), vth=0.22, n_spin=4.0, n_part=5.0, nsnap=20,
                hyper3_twin=False, nblock=100),
    "full": dict(nx=256, nz=64, npart=32768, substeps=2, eps=(0.05, 0.1, 0.2, 0.3),
                 a_hosts=(0.1, 0.2), show=0.2, twins=(0.1, 0.2), vth=0.17, n_spin=4.0,
                 n_part=6.0, nsnap=16, hyper3_twin=True, nblock=200),
}

PROFILE = os.environ.get("TARANIS_P3D_PROFILE", "smoke")


def cfg(profile=None):
    """The profile dict, defaulting to $TARANIS_P3D_PROFILE (or "smoke")."""
    name = profile or PROFILE
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; expected one of {list(PROFILES)}")
    return PROFILES[name]


def geometry(profile=None):
    """(dx, nu, T_eddy(eps)) of a profile: grid spacing, Laplacian coefficient at the
    profile's resolution, and the outer-scale turnover time at a given target u_rms."""
    c = cfg(profile)
    dx = LX/c["nx"]
    re = RE_256*(c["nx"]/256.0)**(4.0/3.0)
    return dx, 0.2*LX/re, (lambda eps: LX/eps)


def forcing_power(eps):
    """Total injection rate that should reach u_rms = eps (see FORCE_C)."""
    return FORCE_C*eps**3/LX


############################
# ensembles                #
############################

def _ideal(om, init, b0=1.0):
    """An ideal-Ohm production ensemble at gyrofrequency `om`, with the numerical
    E_parallel projected out. qm = Omega/B0."""
    return {"qm": om/b0, "B0": b0, "init": dict(init), "epar_project": True}


def ensembles(design, profile=None, b0=1.0):
    """(ensembles, labels, index map) for design "a" (the Omega sweep, design (c)'s rings
    and the contrast/control ensembles) or "b" (the reference ensemble alone).

    `b0` is 1.0 in 3D (Parameters enforces it) and 1/eps for a 2D twin; the velocity scale
    follows it, so Omega, rho, xi and beta_i are the same at either B0."""
    c = cfg(profile)
    dx, _nu, _te = geometry(profile)
    vth = c["vth"]*b0
    vperp = np.sqrt(2.0)*vth
    maxw = {"kind": "maxwellian", "vth": vth}
    om = tuple(vperp/(r*dx) for r in RHO_DX_SWEEP)
    om_ref = vperp/(RHO_DX_REF*dx)
    if design == "b":
        return ([_ideal(om_ref, maxw, b0)],
                [f"(b) rho/dx = {RHO_DX_REF:g}"],
                {"ideal_ref": 0})
    rho_c = RHO_DX_C*dx
    ens = ([_ideal(o, maxw, b0) for o in om]
           + [_ideal(m*vperp/rho_c, {"kind": "ring", "vperp": m*vperp}, b0)
              for m in VPERP_C_MUL]
           + [{"qm": om_ref/b0, "B0": b0, "init": dict(maxw)},
              {"qm": om_ref/b0, "B0": b0, "init": dict(maxw),
               "ez_resistive": True, "ez_forcing": True},
              {"qm": om_ref/b0, "B0": b0, "init": dict(maxw),
               "eperp": False, "ez_ideal": False},
              {"qm": om_ref/b0, "B0": b0, "init": dict(maxw),
               "bperp": False, "eperp": False, "ez_ideal": False}])
    labels = ([f"(a) rho/dx = {r:g}, Omega = {o:.2f}" for r, o in zip(RHO_DX_SWEEP, om)]
              + [f"(c) ring v_perp = {m:g} v_th_rms" for m in VPERP_C_MUL]
              + [f"rho/dx = {RHO_DX_REF:g}, unprojected",
                 f"rho/dx = {RHO_DX_REF:g}, full dpsi/dt",
                 "E = 0 control", "delta-b = 0 control"])
    na, nc = len(RHO_DX_SWEEP), len(VPERP_C_MUL)
    idx = {"sweep": tuple(range(na)),
           "rings": tuple(range(na, na + nc)),
           "ideal_ref": RHO_DX_SWEEP.index(RHO_DX_REF),
           "unprojected": na + nc,
           "full": na + nc + 1,
           "e_zero": na + nc + 2,
           "gyration": na + nc + 3}
    return ens, labels, idx


def particle_config(ens, profile=None):
    c = cfg(profile)
    return {"seed": 7, "n": c["npart"], "substeps": c["substeps"], "ensembles": ens}


############################
# runs                     #
############################

def make_params(eps, particles=None, profile=None, hyper=1, dims=3):
    # one place for the run configuration -- identical every call, which is what makes
    # params.save's identical-record-is-a-no-op guard the resume path's safety net.
    c = cfg(profile)
    _dx, nu, _te = geometry(profile)
    if dims == 2:
        # the 2D twin: velocities in flow units eps*v_A, so nu and the forcing power scale
        # with 1/eps and the target u_rms is 1
        nu, power = nu/eps, forcing_power(1.0)
        extra = {}
    else:
        power = forcing_power(eps)
        extra = dict(nz=c["nz"], Lz=LZ)
    if hyper != 1:
        # the hyper=3 twin: -nu3*k^6 dissipating at the same wavenumber as -nu*k^2
        nu = nu*_hyper3_scale(c["nx"], hyper)
    return jr.Parameters(nx=c["nx"], ny=c["nx"], Lx=LX, Ly=LY, dims=dims,
                         eqpars={"diss": (nu, nu), "hyper": hyper},
                         cfl_safety=0.5, adaptive_timestep=True, cfl_every=1,
                         forcing=True, forcing_mode="elsasser",
                         forcing_power_elsasser=(power/2, power/2),
                         forcing_tau=FORCING_TAU, fshell=FSHELL,
                         forcing_seed=FORCING_SEED, forcing_scale_max=1.0,
                         particles=particles, **extra)


def _hyper3_scale(nx, hyper):
    # nu_h = nu/k_d^(2*hyper-2) puts -nu_h*k^(2*hyper) at the same wavenumber as -nu*k^2,
    # with k_d estimated as half the 2/3 dealias cutoff.
    kd = 0.5*(2.0/3.0)*(nx//2)*(2.0*np.pi/LX)
    return kd**(2.0 - 2.0*hyper)


def run_dirs(root="data/test-particles-3D", profile=None):
    """Every snapshot directory this module writes, by role."""
    c = cfg(profile)
    tag = lambda name, e: os.path.join(root, "%s_eps%g" % (name, e))
    d = {"base": {e: tag("base", e) for e in c["eps"]},
         "a": {e: tag("design_a", e) for e in c["a_hosts"]},
         "b": {e: tag("design_b", e) for e in c["eps"]},
         "twin2d": {e: tag("twin2d", e) for e in c["twins"]},
         "twin2d_base": {e: tag("twin2d_base", e) for e in c["twins"]},
         "fast": tag("design_a_fast", c["show"])}
    if c["hyper3_twin"]:
        d["base_h3"] = {c["show"]: tag("base_h3", c["show"])}
        d["h3"] = {c["show"]: tag("design_a_h3", c["show"])}
    return d


def particle_run(design, eps, root="data/test-particles-3D", profile=None, hyper=1,
                 dims=3):
    """(params, snap_path, labels, index map) of one particle run."""
    b0 = 1.0 if dims == 3 else 1.0/eps
    ens, labels, idx = ensembles("a" if design in ("a", "twin2d", "h3") else "b",
                                 profile, b0=b0)
    params = make_params(eps, particles=particle_config(ens, profile), profile=profile,
                         hyper=hyper, dims=dims)
    key = {"a": "a", "b": "b", "twin2d": "twin2d", "h3": "h3"}[design]
    return params, run_dirs(root, profile)[key][eps], labels, idx


def times(eps, profile=None):
    """(t_spinup, t_particles) of one host, in code time units."""
    c = cfg(profile)
    _dx, _nu, t_eddy = geometry(profile)
    te = t_eddy(eps)
    return c["n_spin"]*te, c["n_part"]*te


def _zero_ic_3d(x, y, z):
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape, z.shape))


def _zero_ic_2d(x, y):
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape))


def _drive(params, snap_path, t_end, t_snap, wall_left, seed, chunk_t=None, nsnap=2000,
           nblock=None, profile=None):
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
        print(f"  {snap_path}: already at t = {float(state.t):.2f} >= {t_end:.2f} "
              f"({len(steps)} snapshots)")
        return True, 0.0
    if chunk_t is None:
        chunk_t = 4.0*t_snap
    mngr = jr.snapshot_manager_setup(params=params, snap_path=snap_path, nsnap=nsnap)
    while float(state.t) < t_end and time.time() - t0_wall < wall_left:
        stop = min(t_end, float(state.t) + chunk_t)
        out = jr.simulate_scan(state, kgrid, params, nblock or cfg(profile)["nblock"],
                               t_snap=t_snap, t_end=stop, mngr=mngr, schemestr=SCHEME,
                               save=True, print_every=10**9, pstate=pstate)
        state, pstate = out if params.particles is not None else (out, None)
    mngr.wait_until_finished()
    done = float(state.t) >= t_end
    wall = time.time() - t0_wall
    print(f"  {snap_path}: t = {float(state.t):.2f} / {t_end:.2f} in {wall:.0f}s this call; "
          f"done={done}")
    return done, wall


def _spinup(params, path, eps, wall_left, profile, dims=3):
    t_spin, _ = times(eps, profile)
    if dims == 2:
        t_spin *= eps                      # 2D twin code time is eps * the 3D time
    ic = _zero_ic_3d if dims == 3 else _zero_ic_2d
    return _drive(params, path, t_spin, t_spin/6.0, wall_left,
                  seed=lambda: (jr.initialize(ic, params), None), profile=profile)


def _particles_from(params, path, base, eps, wall_left, profile, dims=3):
    c = cfg(profile)
    t_spin, t_part = times(eps, profile)
    if dims == 2:
        t_spin, t_part = t_spin*eps, t_part*eps

    def seed():
        step = sn.get_saved_steps(base)[-1]
        return sn.load_snapshot(step, base, params), init_particles(params)

    return _drive(params, path, t_spin + t_part, t_part/c["nsnap"], wall_left, seed=seed,
                  profile=profile)


def make_data(root="data/test-particles-3D", wall_budget=3600.0, profile=None, only=None):
    """Generate every dataset the notebook needs, resuming wherever the last call stopped.

    Phase order: the particle-free spin-ups, then the particle runs restarted from each
    spin-up's final snapshot, then the 2D twins, the optional hyper=3 twin and a short
    high-cadence tail. Returns True once everything has reached its target time; call again
    after a False. `only` restricts the work to a subset of the profile's eps ladder."""
    t0 = time.time()
    c = cfg(profile)
    d = run_dirs(root, profile)
    eps_list = tuple(c["eps"] if only is None else only)
    left = lambda: wall_budget - (time.time() - t0)

    for eps in eps_list:
        print(f"3D base spin-up, eps = {eps:g}")
        if not _spinup(make_params(eps, profile=profile), d["base"][eps], eps, left(),
                       profile)[0]:
            return False
    jobs = ([("a", eps) for eps in eps_list if eps in c["a_hosts"]]
            + [("b", eps) for eps in eps_list])
    for design, eps in jobs:
        params, path, labels, _idx = particle_run(design, eps, root, profile)
        print(f"3D particles, design ({design}), eps = {eps:g}, {len(labels)} ensembles "
              f"x {c['npart']}")
        if not _particles_from(params, path, d["base"][eps], eps, left(), profile)[0]:
            return False

    for eps in (e for e in c["twins"] if e in eps_list):
        print(f"2D twin spin-up, eps = {eps:g} (B0 = {1/eps:g})")
        p2 = make_params(eps, profile=profile, dims=2)
        if not _spinup(p2, d["twin2d_base"][eps], eps, left(), profile, dims=2)[0]:
            return False
        params, path, labels, _idx = particle_run("twin2d", eps, root, profile, dims=2)
        print(f"2D twin particles, eps = {eps:g}, {len(labels)} ensembles x {c['npart']}")
        if not _particles_from(params, path, d["twin2d_base"][eps], eps, left(), profile,
                               dims=2)[0]:
            return False

    if c["hyper3_twin"] and c["show"] in eps_list:
        eps = c["show"]
        print(f"3D hyper=3 twin, eps = {eps:g}")
        if not _spinup(make_params(eps, profile=profile, hyper=3), d["base_h3"][eps], eps,
                       left(), profile)[0]:
            return False
        params, path, labels, _idx = particle_run("h3", eps, root, profile, hyper=3)
        if not _particles_from(params, path, d["base_h3"][eps], eps, left(), profile)[0]:
            return False

    if c["show"] in eps_list:
        # a short high-cadence tail continuing the shown design-(a) run's particles into its
        # own directory: snapshot-pair statistics need an interval over which a particle does
        # not cross the box
        params, src, _labels, _idx = particle_run("a", c["show"], root, profile)
        src_step = sn.get_saved_steps(src)[-1]
        _dx, _nu, t_eddy = geometry(profile)
        t_fast = 0.02*t_eddy(c["show"])
        t_end = float(sn.load_snapshot(src_step, src, params).t) + t_fast
        print(f"high-cadence tail of design (a) eps = {c['show']:g} (to t = {t_end:.2f})")

        def seed(step=src_step):
            return (sn.load_snapshot(step, src, params),
                    sn.load_particles(step, src, params))

        if not _drive(params, d["fast"], t_end, t_fast/20.0, left(), seed=seed,
                      chunk_t=t_fast, nblock=max(5, c["nblock"]//20), profile=profile)[0]:
            return False
    print(f"make_data: complete in {time.time() - t0:.0f}s this call")
    return True


if __name__ == "__main__":
    while not make_data(wall_budget=1800.0):
        print("--- continuing")
