# Alfven-speed channel in compressible MHD: phase mixing (stage 1) -> KHI/turbulence (stage 2).
#
#   python examples/cmhd_channel_run.py --stage 1                     # production stage 1
#   python examples/cmhd_channel_run.py --stage 1 --nx 256            # the reduced-res gate
#   python examples/cmhd_channel_run.py --stage 2 --nx 64 --ny 64 --nz 64 --t-end 10
#   python examples/cmhd_channel_run.py --stage 3 --nx 128 --ny 128 --nz 32 --t-end 12
#
# Always run with TARANIS_PRECISION=64. CMHD is dims=3 + z_spectral + single process
# (taranis/physics/cmhd.py::_check_supported), so there is no MPI path here; on a laptop the
# transport resolves to "serial" by itself.
#
# ------------------------------------------------------------------------------ the setup
#
# Isothermal (gamma = 1) compressible MHD in a triply periodic box Lx = Ly = Lz = 2 pi, mean
# field along z, with a DENSE SLAB occupying the middle half of the box in x:
#
#     f(x)    = 0.5*(tanh((x-x1)/delta) - tanh((x-x2)/delta)),  x1 = Lx/4, x2 = 3Lx/4
#     rho(x)  = 1 + (chi-1) f(x)                                chi = 3
#     B_z(x)  = sqrt(1 - 2 cs0^2 (rho(x)-1))                    cs0 = 0.3
#     v_A(x)  = B_z(x)/sqrt(rho(x))
#
# B_z comes from total-pressure balance cs0^2 rho + B_z^2/2 = cs0^2 + 1/2, i.e. the state is
# an exact static equilibrium of the continuum equations (the pressure force -grad h with
# h = cs0^2 ln rho exactly cancels the tension/pressure of B_z(x) z-hat). delta = Lx/40, so
# the profile is periodic to e^-20 at the box edges. Numbers: rho_in = 3, B_z,in = 0.8,
# v_A,in = 0.8/sqrt(3) = 0.4619, beta_out = 2 cs0^2 = 0.18, beta_in = 2 cs0^2 chi/B_z,in^2 =
# 0.84. 1 - 2 cs0^2 (chi-1) = 0.64 > 0, so B_z is real everywhere (checked at config time).
#
# UNITS. Alfven units with v_A = 1 and rho = 1 OUTSIDE the channel. taranis's CMHD module
# documents "<rho> = rho0 = 1", but that is a CONVENTION of the module header and nothing in
# the solver enforces it: here the volume mean is <rho> = 1 + (chi-1)<f> ~ 2, and the
# normalization that matters (the one every diagnostic is quoted in) is the ambient one.
#
# THE WAVE is a STANDING, y-polarised Alfven wave,
#
#     u_y = a cos(k_z z),  B_y = u_x = u_z = B_x = 0   at t = 0,   k_z = 2 pi/Lz (m = 1),
#
# i.e. a counter-propagating pair -- there is no Alfvenic sign convention to get wrong, and
# the E_kin <-> E_mag exchange at 2 omega is a free correctness check. Field order is
# (rho, u_x, u_y, u_z, B_x, B_y, B_z).
#
# WHY THIS PHASE MIXES EXACTLY. With d/dy = 0 the y-polarised pair obeys
#
#     d_t u_y = (B_z/rho) d_z B_y,      d_t B_y = B_z d_z u_y
#
# with NO x-derivatives anywhere: each x evolves independently at omega(x) = k_z v_A(x), and
# the only coupling in x is the dissipation. So k_x(x,t) = k_z v_A'(x) t exactly. taranis's
# linear operator is L = -diss_f (k_perp^2 + k_z^2)^hyper applied to the field variables u
# and B; with nu = eta = D the eigenmode (u_y, B_y) damps at exactly D k^(2 hyper), and the
# z-averaged wave energy density obeys, per x, to O(D) and O(a^2),
#
#     E(x,t)/E(x,0) = exp( -2 D int_0^t (k_z^2 v_A'(x)^2 t'^2 + k_z^2)^hyper dt' )      (*)
#
# with E(x,t) = < rho u_y^2 + B_y^2 >_{y,z} / 2 (for the undamped standing wave this is
# rho(x) a^2/4, constant in time -- the oscillation cancels between the two halves). (*) is
# THE STAGE-1 GATE, evaluated pointwise in x. The k_z^2 term is kept and the integral is done
# numerically (np.trapezoid on a fine t' grid); do not drop it and do not close it in
# closed form for hyper != 2. D_rho = 0 in stage 1, so the background profile is untouched by
# dissipation and E(x,0) stays the right normalization for all time.
#
# WHAT (*) IS AND IS NOT, MEASURED (2026-09-04, nx = 256 to t_end, fp64). (*) is the LEADING
# WKB envelope law and it has an error of its own, which is not a solver error:
#
#   - the run agrees with the EXACT linearised problem -- the same L, the same profile, the
#     same k_z, integrated rather than expanded (report()'s linear_reference, cross-checked
#     against an independent throwaway RK4 version) -- to max|dln| = 7.6e-2 over the whole
#     run and 1.8% relative wherever the exact damping is at least one e-fold. THAT is the
#     verification of the solver, the energy normalization and the dissipation form;
#   - (*) itself departs from that exact solution by 0.044 in ln at 1 e-fold of damping,
#     0.24 at 2, and 3.2 (a factor 25 in E) at t_end, where it predicts -14.9 and the truth
#     is -4.6. The departure is RESOLUTION INDEPENDENT (the reference at nx = 256, 1024 and
#     4096 agrees to 3 digits) and is the finite-layer-width failure of the envelope
#     picture: at fixed x the field carries a band of k_x of width ~1/delta, the hyper = 2
#     damping varies by e^38 across that band by t_end, and the surviving field at a deeply
#     damped x is the low-k_x tail of its own band, not the WKB mode. In units of the
#     accumulated damping, the band-edge variation is 20/(k_x delta), so (*) is a few-percent
#     law only while |ln E/E0| * 20/(k_x delta) << 1;
#   - so (*) gets BETTER with nx only because a finer grid buys a larger t_half at fixed
#     k_x delta: at the stage-1 PRODUCTION nx = 1024 the reference says (*) is good to 0.004
#     in ln at 0.3 t_end, 0.043 (3.5% relative) at 1 e-fold, and still 2.5 (34%) at t_end.
#
# Nothing here is tuned to make anything pass: report() prints (*) exactly as specified and
# then prints the exact-linear comparison next to it, so a future reader can see which of
# the two moved. Checked and excluded as explanations: energy normalization
# (mean_x e_kin_x == E_kin to 7.8e-4, E(x,0) = <rho> a^2/4 exactly), k_y leakage (E_ky is
# identically 0.0), the k_z^2 term (kept; it is the whole prediction at small v'), and the
# dissipation form (the reference uses the same operator).
#
# STAGE 1 (linear, 2.5D in (x,z); ny = 4 is a dead axis) sizes D from a resolution rule: the
# steepest layer has damped to e^-3 IN AMPLITUDE by the time its k_x reaches half the dealias
# cut. With k_max = (nx/3)(2 pi/Lx), v'_max = max_x |dv_A/dx| measured NUMERICALLY on the
# grid (np.gradient -- it is resolution dependent, and the number the run actually sees is
# the discrete one), t_half = (k_max/2)/(k_z v'_max) and, for hyper = 2,
#
#     D (k_z v'_max)^4 t_half^5 / 5 = 3        ->    D = 15 / ((k_z v'_max)^4 t_half^5)
#
# (the k_z^2 term is negligible at that k_x). t_end = 1.2 t_half by default.
#
# STAGE 2 (nonlinear, 3D) turns the amplitude up to a = 0.1 and lets the phase-mixed shear
# layer go unstable: generalized phase mixing -> Kelvin-Helmholtz rolls (Magyar & Van
# Doorsselaere 2016; Antolin et al. 2014's "TWIKH rolls") -> turbulence. diss = 2e-6 (scalar,
# all seven fields) gives ~100 per Alfven time at the 256^3 dealias cut, so k_x ~ 20 is
# reached before dissipation bites and gamma_KH ~ a k_x/2 ~ 1. A small random-phase k_y != 0
# SEED in u_x and u_y (rms seed_amp = 1e-3 = 1% of a) is added so KHI grows from a controlled
# amplitude rather than from round-off.
#
# STAGE 2 HAS NOT BEEN RUN AT ITS PRODUCTION SIZE, and the reduced-resolution check does NOT
# show the instability: at 64^3 with the default diss, to t = 50 (2026-09-04, fp64), the run
# stays finite with rho_min = 0.926 and div B at round-off, but E_ky never leaves its initial
# 2.4e-7 - 5.1e-7 band -- no exponential window at all. Two reasons to expect that and not
# read it as a null result for the design: (i) the design's own onset is k_x ~ 20, i.e.
# t = k_x/(k_z v'_max) ~ 12 at 256^3, so t = 10 is before it; (ii) at 64^3 the phase-mixing
# k_x reaches the 2/3 cut k_max = 21.3 at t ~ 15 with only D k_max^4 = 0.41 per unit time of
# damping there, so past t ~ 15 the 64^3 run is under-dissipated at the grid scale and is not
# a faithful reduction of the 256^3 problem. Whether KHI appears is a question for the
# production run.
#
# STAGE 3 (nonlinear 3D, the "uniturbulence" configuration) replaces the y-polarised wave
# by a PROPAGATING, x-polarised SURFACE ALFVEN WAVE on the two interfaces -- the kink mode
# of the slab in the k_perp >> k_z limit. For the sharp-interface incompressible problem the
# kink phase speed is
#
#     c_k^2 = (B_in^2 + B_out^2)/(rho_in + rho_out)      = 1.64/4  ->  c_k = 0.64031
#
# which sits BETWEEN v_A,in = 0.4619 and v_A,out = 1. That is the whole point of the run:
# a single surface wave propagating at one speed c_k cannot be a pure Elsasser state
# anywhere, because the local Alfven speed differs from c_k on both sides. With
#
#     b_perp = -(B_z(x)/c_k) u_perp        (linear induction, d_t b = B_z d_z u, w = c_k k_z)
#
# the local Elsasser fields z^+- = u_perp +- b_perp/sqrt(rho) are
#
#     z^-  = (1 + v_A/c_k) u_perp   (the +z-going MAJOR component)
#     z^+  = (1 - v_A/c_k) u_perp   (the co-propagating MINOR component)
#
# so the minor/major AMPLITUDE ratio is (c_k - v_A(x))/(c_k + v_A(x)): -0.2193 outside the
# slab and +0.1618 inside. A nonlinearity u.grad u ~ z^+ z^- is therefore built into a
# single "unidirectional" wave -- Magyar, Van Doorsselaere & Goossens 2019's uniturbulence.
# The run measures the projection: the incompressible sharp-interface eigenfunction is only
# approximate here (tanh layer of width delta, M_s ~ 0.33), which is accepted, not tuned.
#
# THE IC, per interface i at x_i with the PERIODIC offset d_i(x) = x - x_i wrapped into
# [-Lx/2, Lx/2) and s_i = sign(d_i), sgn_1 = +1, sgn_2 = +1 (kink) or -1 (sausage):
#
#     u_x += U0 sgn_i exp(-k_y|d_i|) cos(k_y y) cos(k_z z)
#     u_y += U0 sgn_i s_i exp(-k_y|d_i|) sin(k_y y) cos(k_z z)
#     u_z  = 0,   B_perp = -(B_z(x)/c_k) u_perp (propagating) or 0 (standing)
#
# with k_y = ky0 2pi/Ly. d_x u_x + d_y u_y = 0 EXACTLY on each side of each interface (the
# x-derivative of exp(-k_y|d|) brings down -k_y s_i, which is exactly what the s_i in u_y
# cancels), so the IC is incompressible away from the two kinks; the k_y != 0 seed of stage
# 2 is then ADDED to (u_x, u_y) unchanged (no B seed), and it is NOT divergence-free.
# The construction is periodic but the envelope has not decayed at half a box: exp(-k_y Lx/2)
# = 1.9e-3 at ky0 = 2, so u_y carries a jump of 2*1.9e-3*U0 at each interface's antipode --
# which, because the two interfaces are exactly Lx/2 apart, lands on the OTHER interface.
# That 0.2%-of-U0 artifact is inherent to the truncated eigenfunction and is left in place.
#
# DIV B, AND WHY IT IS NOT ROUND-OFF HERE. B_perp = -(B_z(x)/c_k) u_perp is by itself NOT
# solenoidal: div B = d_x B_x + d_y B_y = -(B_z'(x)/c_k) u_x (the perpendicular divergence of
# u vanishes, the derivative of the PROFILE does not). taranis's induction is curl form, so
# div B is preserved exactly -- an IC divergence is frozen in for the whole run. Measured on
# the seed-free IC, max|div B| is 0.53 of k_y max|B_perp| and diagnostics.cmhd.divB_max is
# 9.12e-4 at nx = 128 and 9.14e-4 at nx = 256: RESOLUTION INDEPENDENT, i.e. real. This file
# ships the briefed IC as the default (bz_wave=False) and the cure next to it: the exact
# linear eigenfunction b = curl(xi x B) also carries b_z = -B_z'(x) xi_x, and adding it
# (bz_wave=True, _bz_wave below) cancels the divergence identically -- the same metric then
# reads 2.33e-5 at nx = 128 and 5.84e-6 at nx = 256, falling as 1/nx^2 because all that is
# left is the |d| kink. Turn it on for production unless the monopole is wanted.
#
# WHAT STAGE 3 ADDS TO THE DIAGNOSTICS: the Elsasser split (e_zp_x/e_zm_x profiles, E_zp/
# E_zm and the minor fraction), E_ky_other (the k_y content OUTSIDE the driven ky0, which is
# the instability monitor here -- E_ky itself is dominated by the wave), div u, and
# ux_iface = u_x(x_1, y=0, z), whose k_z = mz Fourier phase gives the MEASURED phase speed
# to compare against c_k. Expect ~10%: resonant absorption in the smooth layer and the
# finite delta both shift it.
#
# ---------------------------------------------------------------------------- what is here
#
#   make_data(snap_path, stage=1, ...) -> bool    resumable/idempotent, the lugus contract
#   report(snap_path)                             the gate + the science summary
#   __main__                                      argparse loop: make_data until True, report
#
# Lightweight diagnostics go to <dirname(snap_path)>/diag: config.json, trace.npz (rewritten
# at every record), timing.json, and (stage 2) slices_NNNN.npz. Snapshots are the restart
# path only; nothing in report() needs them.
import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import taranis as jr
import taranis.snapshot_io as sn
from taranis import _precision
from taranis.diagnostics import cmhd as dcmhd
# the module's own Nyquist-zeroed d/dz -- imported rather than rebuilt from kgrid.kz so the
# div u monitor here cannot drift from the d/dz the solver's induction term and
# diagnostics.cmhd.divB_max use (that module crosses the same boundary, for the same reason)
from taranis.physics.cmhd import _kz_deriv as _cmhd_kz
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

_trapz = getattr(np, "trapezoid", None) or np.trapz

# ------------------------------------------------------------------------------ the config

# knobs shared by both stages; STAGES[stage] overrides these, kwargs to make_data override
# both. Anything here is a legal make_data override (an unknown key is an error).
BASE = dict(
    Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi,
    chi=3.0,                 # density contrast rho_in/rho_out
    cs0=0.3,                 # sound speed at rho = 1 (gamma = 1 -> everywhere)
    delta=2*np.pi/40.0,      # layer width; Lx/40
    hyper=2,                 # L = -diss (k_perp^2 + kz^2)^hyper
    cfl_safety=0.4,
    a=1e-2,                  # standing-wave amplitude in u_y
    mz=1,                    # wave parallel mode number (k_z = mz*2pi/Lz)
    diss=None,               # None -> stage-1 rule below; else scalar or (D_rho, nu, eta)
    t_end=None,              # None -> 1.2*t_half
    seed_amp=0.0,            # rms of the ky != 0 seed in (u_x, u_y); 0 disables it
    seed=1,                  # np.random.default_rng seed for that seed field
    seed_nmax=8,             # 1 <= |ikx|,|iky| <= seed_nmax
    seed_kzmax=2,            # ikz in 0..seed_kzmax
    nx=256, ny=4, nz=32,
    # ---- stage-3 only (the propagating kink / surface Alfven wave). Unused by stages 1-2.
    U0=0.1,                  # surface-wave velocity amplitude at the interface
    ky0=2,                   # perpendicular mode NUMBER; k_y = ky0*2pi/Ly
    kink=True,               # True: both interfaces displaced the same way (the slab
                             # translates); False: sausage (opposite displacements)
    propagating=True,        # True: B_perp = -(B_z/c_k) u_perp, a +z-going wave;
                             # False: B_perp = 0, a standing pair
    bz_wave=False,           # add the eigenfunction's b_z = -B_z'(x) xi_x, which is what
                             # makes the propagating IC EXACTLY solenoidal. Default False
                             # = the IC as briefed; see the div B note in the header.
)

STAGES = {
    # stage 1: linear phase mixing, 2.5D in (x,z). ny=4 is a dead axis (a y-independent IC
    # stays y-independent EXACTLY -- only ky=0 products are ever formed).
    1: dict(nx=1024, ny=4, nz=32, a=1e-2, hyper=2, diss=None, t_end=None, seed_amp=0.0),
    # stage 2: nonlinear 3D. diss is a scalar (all seven fields), t_end fixed, seeded.
    2: dict(nx=256, ny=256, nz=256, a=0.1, hyper=2, diss=2e-6, t_end=50.0, seed_amp=1e-3),
    # stage 3: the propagating kink (surface Alfven) wave. `a` is UNUSED here -- the
    # amplitude knob is U0. hyper=3 with diss=5.6e-13 are the stage-2b production values,
    # defaulted here so a production launch needs fewer overrides.
    3: dict(nx=256, ny=256, nz=256, a=0.0, hyper=3, diss=5.6e-13, t_end=80.0,
            seed_amp=1e-3, U0=0.1, ky0=2, kink=True, propagating=True, bz_wave=False),
}

# per-stage driver defaults (block size, diagnostic and snapshot cadence). A None entry in
# DIAG_EVERY/SNAP_EVERY is filled from t_half at resolve time.
STAGE_DRIVER = {
    1: dict(nblock=200, diag_every=None, snap_every=None),   # t_half/60, t_half/4
    2: dict(nblock=20, diag_every=0.5, snap_every=10.0),
    3: dict(nblock=20, diag_every=0.5, snap_every=10.0),
}


def _profile(x, cfg, xp=np):
    """(rho, B_z) of the static equilibrium at coordinate x. Works on numpy or jax arrays."""
    x1, x2 = cfg["Lx"]/4.0, 3.0*cfg["Lx"]/4.0
    f = 0.5*(xp.tanh((x - x1)/cfg["delta"]) - xp.tanh((x - x2)/cfg["delta"]))
    rho = 1.0 + (cfg["chi"] - 1.0)*f
    return rho, xp.sqrt(1.0 - 2.0*cfg["cs0"]**2*(rho - 1.0))


def resolve_config(stage, **overrides):
    """The one place the run configuration is decided: pure function of (stage, overrides).

    Returns a plain-JSON dict carrying both the inputs and every derived number (k_max,
    v'_max, t_half, D, diss, t_end, dt_est). params.save's identical-record guard is what
    makes make_data's resume safe, and this function being deterministic is what makes that
    guard meaningful."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}, got {stage!r}")
    cfg = dict(BASE)
    cfg.update(STAGES[stage])
    unknown = [k for k in overrides if k not in cfg]
    if unknown:
        raise ValueError(f"unknown make_data override(s) {unknown}; legal keys are "
                         f"{sorted(cfg)}")
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["stage"] = int(stage)

    if not cfg["chi"] >= 1.0:
        raise ValueError(f"chi is the density contrast rho_in/rho_out and must be >= 1, got "
                         f"{cfg['chi']!r}")
    bz2_in = 1.0 - 2.0*cfg["cs0"]**2*(cfg["chi"] - 1.0)
    if bz2_in <= 0.0:
        raise ValueError(f"no pressure-balanced equilibrium: 1 - 2 cs0^2 (chi-1) = {bz2_in:.4g} "
                         f"<= 0 for cs0={cfg['cs0']}, chi={cfg['chi']} (the slab is too dense "
                         f"or too warm for B_z^2 to stay positive)")

    # ---- derived, on the grid the run will actually use
    nx, Lx, Lz = cfg["nx"], cfg["Lx"], cfg["Lz"]
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    rho_x, bz_x = _profile(x, cfg, xp=np)
    vA_x = bz_x/np.sqrt(rho_x)
    vp_max = float(np.max(np.abs(np.gradient(vA_x, Lx/nx))))
    kz = cfg["mz"]*2.0*np.pi/Lz
    k_max = (nx/3.0)*(2.0*np.pi/Lx)               # the 2/3 dealias cut in x
    t_half = (0.5*k_max)/(kz*vp_max)              # k_x reaches k_max/2 at the steepest layer
    # amplitude e^-3 at t_half, hyper = 2: D (kz v')^4 t_half^5/5 = 3. Quoted for every
    # hyper, but only USED as the default when hyper == 2 (see below).
    D_rule = 15.0/((kz*vp_max)**4 * t_half**5)

    diss = cfg["diss"]
    if diss is None:
        if cfg["hyper"] != 2:
            raise ValueError(f"the D-sizing rule in this file is written for hyper=2 "
                             f"(D (kz v')^4 t_half^5/5 = 3); with hyper={cfg['hyper']} pass "
                             f"diss=... explicitly")
        diss = [0.0, D_rule, D_rule]              # (D_rho, nu, eta): D_rho = 0 keeps rho exact
    diss = float(diss) if np.shape(diss) == () else [float(v) for v in np.ravel(diss)]
    cfg["diss"] = diss
    cfg["t_end"] = float(cfg["t_end"]) if cfg["t_end"] is not None else 1.2*t_half

    # ---- the kink / surface-Alfven numbers (stage 3; pure functions of chi and cs0, so
    # they are recorded for every stage). c_k^2 = (B_in^2 + B_out^2)/(rho_in + rho_out) is
    # the sharp-interface incompressible kink speed; the minor/major Elsasser AMPLITUDE
    # ratio of a +z-going wave at that speed is (c_k - v_A)/(c_k + v_A), whose square is the
    # ENERGY ratio E_minor/E_major and r^2/(1+r^2) the minor FRACTION E_minor/(E_+ + E_-).
    vA_in = np.sqrt(bz2_in/cfg["chi"])
    c_k = float(np.sqrt((bz2_in + 1.0)/(cfg["chi"] + 1.0)))
    r_out, r_in = (c_k - 1.0)/(c_k + 1.0), (c_k - vA_in)/(c_k + vA_in)
    ky_wave = cfg["ky0"]*2.0*np.pi/cfg["Ly"]

    # the wave amplitude that sets the CFL estimate: U0 at stage 3, a otherwise
    u_amp = cfg["U0"] if cfg["stage"] == 3 else cfg["a"]
    cf_max = float(np.max(np.sqrt(cfg["cs0"]**2 + bz_x**2/rho_x)))
    dmin = min(Lx/nx, cfg["Ly"]/cfg["ny"], Lz/cfg["nz"])
    dt_est = cfg["cfl_safety"]*dmin/(cf_max + u_amp)
    cfg.update(c_k=c_k, ky_wave=float(ky_wave),
               kink_period=float(2.0*np.pi/(kz*c_k)),
               minor_ratio_out=float(r_out), minor_ratio_in=float(r_in),
               minor_frac_out=float(r_out**2/(1.0 + r_out**2)),
               minor_frac_in=float(r_in**2/(1.0 + r_in**2)))
    cfg.update(k_max=float(k_max), kz=float(kz), vp_max=vp_max, t_half=float(t_half),
               D_rule=float(D_rule), dt_est=float(dt_est),
               nsteps_est=int(round(cfg["t_end"]/dt_est)),
               rho_in=float(1.0 + (cfg["chi"] - 1.0)), bz_in=float(np.sqrt(bz2_in)),
               vA_in=float(np.sqrt(bz2_in/cfg["chi"])),
               beta_out=float(2.0*cfg["cs0"]**2),
               beta_in=float(2.0*cfg["cs0"]**2*cfg["chi"]/bz2_in),
               precision=int(_precision.precision))
    return cfg


def make_params(cfg):
    """The Parameters for a resolved config -- identical every call, which is what params.save's
    identical-record guard needs."""
    return jr.Parameters(nx=cfg["nx"], ny=cfg["ny"], nz=cfg["nz"],
                         Lx=cfg["Lx"], Ly=cfg["Ly"], Lz=cfg["Lz"],
                         dims=3, z_spectral=True, eqtype="CMHD",
                         adaptive_timestep=True, cfl_safety=cfg["cfl_safety"], dt=1e-3,
                         eqpars=dict(cs0=cfg["cs0"], diss=cfg["diss"],
                                     hyper=int(cfg["hyper"]), gamma=1.0))


# --------------------------------------------------------------------------------- the IC

def _seed_coeffs(cfg):
    """Host-side complex coefficients c[comp, ikx, iky, ikz] for the ky != 0 seed.

    Modes are cos(k.x + phase) with random phase and gaussian amplitude, ikx >= 1 (so no mode
    is another's conjugate), 1 <= |iky| <= seed_nmax, 0 <= ikz <= seed_kzmax. They are
    mutually orthogonal and none sits at k = 0, so the seed's mean square is exactly
    sum |c|^2/2 -- which is why the coefficients can be rescaled here, on the host, to make
    seed_amp the EXACT rms of sqrt(u_x^2 + u_y^2). ("amplitude 1e-3, absolute" is then a
    statement about the seed field, not about one of ~800 modes.)"""
    n, mz = int(cfg["seed_nmax"]), int(cfg["seed_kzmax"])
    ikx = np.arange(1, n + 1)
    iky = np.concatenate([np.arange(-n, 0), np.arange(1, n + 1)])
    ikz = np.arange(0, mz + 1)
    rng = np.random.default_rng(int(cfg["seed"]))
    shape = (2, ikx.size, iky.size, ikz.size)
    c = rng.normal(size=shape)*np.exp(1j*rng.uniform(0.0, 2.0*np.pi, size=shape))
    c *= cfg["seed_amp"]/np.sqrt(0.5*np.sum(np.abs(c)**2))
    return ikx, iky, ikz, c


def _seed_field(x, y, z, cfg):
    """(u_x, u_y) seed on the (nz,nx,ny) grid, or (None, None) when seed_amp == 0.

    Evaluated as three sequential contractions rather than a python loop over modes: a loop
    over ~800 modes each touching a 256^3 array is 10^10 flops of IC, the contractions are
    nx*ny*nz*seed_nmax."""
    if not cfg["seed_amp"] > 0.0:
        return None, None
    ikx, iky, ikz, c = _seed_coeffs(cfg)
    kux, kuy, kuz = 2*np.pi/cfg["Lx"], 2*np.pi/cfg["Ly"], 2*np.pi/cfg["Lz"]
    ex = jnp.exp(1j*kux*x.reshape(-1, 1)*ikx.reshape(1, -1))     # (nx, na)
    ey = jnp.exp(1j*kuy*y.reshape(-1, 1)*iky.reshape(1, -1))     # (ny, nb)
    ez = jnp.exp(1j*kuz*z.reshape(-1, 1)*ikz.reshape(1, -1))     # (nz, nc)
    out = []
    for comp in range(2):
        t1 = jnp.einsum("abc,zc->abz", jnp.asarray(c[comp]), ez)
        t2 = jnp.einsum("abz,yb->azy", t1, ey)
        out.append(jnp.real(jnp.einsum("azy,xa->zxy", t2, ex)))
    return out[0], out[1]


def _interfaces(cfg):
    """((x_1, sgn_1), (x_2, sgn_2)) -- the two slab edges and the relative sign of their
    displacement: both +1 is the KINK (the slab translates bodily), opposite is the
    SAUSAGE (the slab is squeezed)."""
    Lx = cfg["Lx"]
    return ((0.25*Lx, 1.0), (0.75*Lx, 1.0 if cfg["kink"] else -1.0))


def _surface_wave(x, y, z, cfg, xp=jnp):
    """(u_x, u_y) of the surface Alfven wave, summed over the two interfaces.

    Per interface, with d_i the PERIODIC offset x - x_i wrapped into [-Lx/2, Lx/2):

        u_x = U0 sgn_i exp(-k_y|d_i|) cos(k_y y) cos(k_z z)
        u_y = U0 sgn_i sign(d_i) exp(-k_y|d_i|) sin(k_y y) cos(k_z z)

    d_x u_x + d_y u_y = 0 identically on each side of x_i: d_x exp(-k_y|d|) =
    -k_y sign(d) exp(-k_y|d|) and d_y sin(k_y y) = +k_y cos(k_y y), and the sign(d) in u_y
    is exactly what makes the two cancel. sign(0) = 0, so u_y vanishes on the interface
    column itself -- the correct midpoint value of an odd function."""
    U0, ky, kz, Lx = cfg["U0"], cfg["ky_wave"], cfg["kz"], cfg["Lx"]
    ux = uy = 0.0
    for xi, sgn in _interfaces(cfg):
        d = xp.mod(x - xi + 0.5*Lx, Lx) - 0.5*Lx
        env = (U0*sgn)*xp.exp(-ky*xp.abs(d))
        ux = ux + env*xp.cos(ky*y)*xp.cos(kz*z)
        uy = uy + env*xp.sign(d)*xp.sin(ky*y)*xp.cos(kz*z)
    return ux, uy


def _dbz_dx(x, cfg, xp=jnp):
    """dB_z/dx of the equilibrium, analytically: B_z^2 = 1 - 2 cs0^2 (rho-1) and
    rho = 1 + (chi-1) f, so B_z B_z' = -cs0^2 (chi-1) f'(x)."""
    x1, x2 = cfg["Lx"]/4.0, 3.0*cfg["Lx"]/4.0
    d = cfg["delta"]
    fp = 0.5*((1.0/xp.cosh((x - x1)/d))**2 - (1.0/xp.cosh((x - x2)/d))**2)/d
    _, bz = _profile(x, cfg, xp=xp)
    return -cfg["cs0"]**2*(cfg["chi"] - 1.0)*fp/bz


def _bz_wave(x, y, z, cfg, xp=jnp):
    """The eigenfunction's PARALLEL magnetic perturbation, b_z = -B_z'(x) xi_x.

    b = curl(xi x B) for a purely perpendicular displacement xi of B = B_z(x) zhat gives
    b_perp = B_z d_z xi_perp -- which IS the briefed B_perp = -(B_z/c_k) u_perp, since
    u = d_t xi and w = c_k k_z -- AND b_z = -B_z (div_perp xi) - B_z' xi_x = -B_z' xi_x
    (div_perp xi = 0 for these forms). With xi_x = int u_x dt =
    -(U0/(c_k k_z)) sum_i sgn_i e^{-k_y|d_i|} cos(k_y y) sin(k_z z),

        b_z = +(B_z'(x)/(c_k k_z)) U0 sum_i sgn_i e^{-k_y|d_i|} cos(k_y y) sin(k_z z)

    and div b = -(B_z'/c_k) u_x + d_z b_z = 0 EXACTLY, by the same cancellation that makes
    div u = 0. Without it the IC carries a real, O(1)-of-scale monopole field which curl
    induction then preserves for the whole run -- see the div B note in the header."""
    U0, ky, kz, Lx = cfg["U0"], cfg["ky_wave"], cfg["kz"], cfg["Lx"]
    env = 0.0
    for xi, sgn in _interfaces(cfg):
        d = xp.mod(x - xi + 0.5*Lx, Lx) - 0.5*Lx
        env = env + (U0*sgn)*xp.exp(-ky*xp.abs(d))
    return (_dbz_dx(x, cfg, xp=xp)/(cfg["c_k"]*kz))*env*xp.cos(ky*y)*xp.sin(kz*z)


def _make_ic_stage3(cfg):
    """The stage-3 IC: the surface wave above on the same static slab equilibrium.

    B_perp = -(B_z(x)/c_k) u_perp when propagating (linear induction at w = c_k k_z, which
    sends the wave toward +z since B_z > 0), 0 when standing. The seed is added to
    (u_x, u_y) AFTER B is formed, exactly as stage 2 adds it and with no B counterpart."""
    ck = cfg["c_k"]

    def ic(x, y, z):
        one = jnp.ones(jnp.broadcast_shapes(x.shape, y.shape, z.shape))
        rho, bz = _profile(x, cfg, xp=jnp)
        ux, uy = _surface_wave(x, y, z, cfg)
        ux, uy = ux*one, uy*one
        bz = bz*one
        if cfg["propagating"]:
            bx, by = -(bz/ck)*ux, -(bz/ck)*uy
            if cfg["bz_wave"]:
                bz = bz + _bz_wave(x, y, z, cfg)*one
        else:
            bx, by = 0.0*one, 0.0*one
        sx, sy = _seed_field(x, y, z, cfg)
        if sx is not None:
            ux, uy = ux + sx, uy + sy
        zero = 0.0*one
        return jnp.stack([rho*one, ux, uy, zero, bx, by, bz])

    return ic


def make_ic(cfg):
    """The IC callable for run.initialize (which hands it x (1,nx,1), y (1,1,ny), z (nz,1,1)
    and applies the 2/3 dealias mask to the result)."""
    if cfg["stage"] == 3:
        return _make_ic_stage3(cfg)
    a, kz = cfg["a"], cfg["kz"]

    def ic(x, y, z):
        one = jnp.ones(jnp.broadcast_shapes(x.shape, y.shape, z.shape))
        rho, bz = _profile(x, cfg, xp=jnp)
        ux, uy_seed = _seed_field(x, y, z, cfg)
        uy = a*jnp.cos(kz*z)*one
        if ux is None:
            ux = 0.0*one
        else:
            uy = uy + uy_seed
        zero = 0.0*one
        return jnp.stack([rho*one, ux, uy, zero, zero, zero, bz*one])

    return ic


# ------------------------------------------------------------------------- the diagnostics

def diag_dir(snap_path):
    """<the run directory>/diag -- a sibling of snap_path, so on Kaggle the whole workdir
    (checkpoints + diag) zips as one artifact."""
    return os.path.join(os.path.dirname(os.path.abspath(str(snap_path))), "diag")


# scalar trace columns; the per-x / per-k arrays are handled separately
_SCALARS = ("t", "E_kin", "E_mag", "E_int", "M_s", "M_A", "rho_min", "divB_max",
            "E_u", "E_ky")
_PROFILES = ("e_kin_x", "e_mag_x", "Pkx")
_SPECTRA = ("spec_kin", "spec_mag", "spec_rho")
# stage 3 only: the Elsasser split, the off-ky0 instability monitor, div u, and the
# interface column u_x(x_1, y=0, z) the phase-speed fit reads.
_SCALARS3 = ("E_zp", "E_zm", "E_minor_frac", "E_ky_other", "divu_max", "divu_max_clear")
_PROFILES3 = ("e_zp_x", "e_zm_x", "ux_iface")


def _empty_trace(stage):
    cols = list(_SCALARS) + list(_PROFILES) + (list(_SPECTRA) if stage in (2, 3) else [])
    if stage == 3:
        cols += list(_SCALARS3) + list(_PROFILES3)
    return {c: [] for c in cols}


def _load_trace(diag, stage, t_keep):
    """Resume the trace: keep records with t <= t_keep, drop the rest (the particles-sidecar
    rule -- a restart from an older snapshot must never leave duplicate or future times)."""
    tr = _empty_trace(stage)
    path = os.path.join(diag, "trace.npz")
    if not os.path.exists(path):
        return tr, {}
    with np.load(path) as z:
        stored = {k: z[k] for k in z.files}
    t = stored.get("t", np.zeros(0))
    keep = np.nonzero(t <= t_keep + 1e-12)[0]
    for c in tr:
        if c in stored and len(stored[c]) >= len(t):
            tr[c] = [np.asarray(stored[c][i]) for i in keep]
    static = {k: v for k, v in stored.items() if k not in tr}
    if len(keep) < len(t):
        print(f"  trace.npz: dropped {len(t)-len(keep)} record(s) with t > {t_keep:.6g} "
              f"(restart), kept {len(keep)}")
    return tr, static


def _save_trace(diag, trace, static):
    out = dict(static)
    for c, v in trace.items():
        out[c] = np.asarray(v) if v else np.zeros(0)
    np.savez(os.path.join(diag, "trace.npz"), **out)


def _prune_slices(diag, nkeep):
    for name in sorted(os.listdir(diag)):
        if name.startswith("slices_") and name.endswith(".npz"):
            if int(name[7:-4]) >= nkeep:
                os.remove(os.path.join(diag, name))


def _div_u(state, kgrid, params, cfg, clear=0.4):
    """(max|div u|, max|div u| at least `clear` in x from either interface), both in units
    of k_y U0 -- the size of the two terms that cancel in the IC.

    Spectral d/dx, d/dy and the module's own Nyquist-zeroed d/dz, then ONE inverse
    transform. On the grid this sees three things the analytic statement does not: the |d|
    kinks at the two interfaces, the k_y != 0 seed (which is NOT solenoidal) and, later in
    the run, the genuine compressive response. The clearance is a PHYSICAL distance, not a
    column count, because the kink's contribution is a local one: at fixed clearance the
    seed-free IC number falls as 1/nx (0.045, 0.029, 0.015, 0.0071 at nx = 64, 128, 256,
    512 measured with clear = 0.4), which is the spectral derivative of a C^0 function
    converging, while at a fixed number of COLUMNS it would look resolution-independent
    only because the window shrinks with the grid. The clean analytic statement -- exactly
    zero on each side -- is div_u_analytic()."""
    kz = _cmhd_kz(kgrid, params)
    uk = state.fields[1:4]
    dk = 1j*(kgrid.kx*uk[0] + kgrid.ky*uk[1] + kz*uk[2])
    d = np.abs(np.asarray(jr.grids.ifft(dk, params)))
    scale = max(cfg["ky_wave"]*cfg["U0"], 1e-300)
    xg = np.linspace(0.0, cfg["Lx"], cfg["nx"], endpoint=False)
    far = np.ones(cfg["nx"], dtype=bool)
    for xi, _ in _interfaces(cfg):
        far &= np.abs((xg - xi + 0.5*cfg["Lx"]) % cfg["Lx"] - 0.5*cfg["Lx"]) > clear
    return float(d.max()/scale), float(d[:, far].max()/scale)


def div_u_analytic(cfg, npts=64, h=1e-5, clear=0.1, seed=0):
    """max |d_x u_x + d_y u_y| / (k_y U0) for the ANALYTIC stage-3 surface wave, by central
    differences at random points at least `clear` from either interface.

    "It is divergence-free on each side of each interface" checked rather than asserted:
    the cancellation is between two O(k_y U0) terms, so anything above the differencing
    error (~k_y^2 h^2/6 in these units, ~7e-11 at the defaults) would mean the eigenfunction
    is written down wrong. Host float64 throughout (xp=np), independent of
    TARANIS_PRECISION."""
    Lx = cfg["Lx"]
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, Lx, 8*npts)
    ok = np.ones(xs.shape, dtype=bool)
    for xi, _ in _interfaces(cfg):
        ok &= np.abs((xs - xi + 0.5*Lx) % Lx - 0.5*Lx) > clear
    x = xs[ok][:npts].reshape(-1, 1, 1)
    y = rng.uniform(0.0, cfg["Ly"], npts).reshape(1, -1, 1)
    z = rng.uniform(0.0, cfg["Lz"], npts).reshape(1, 1, -1)
    dxux = (_surface_wave(x + h, y, z, cfg, xp=np)[0]
            - _surface_wave(x - h, y, z, cfg, xp=np)[0])/(2.0*h)
    dyuy = (_surface_wave(x, y + h, z, cfg, xp=np)[1]
            - _surface_wave(x, y - h, z, cfg, xp=np)[1])/(2.0*h)
    return float(np.max(np.abs(dxux + dyuy))/max(cfg["ky_wave"]*cfg["U0"], 1e-300))


def record(state, kgrid, params, cfg, trace, diag, static=None, verbose=True):
    """One diagnostic record, appended to `trace` (and, stage 2, one slices_NNNN.npz).

    Everything here is on the repo's one energy-like convention (volume averages;
    CLAUDE.md): diagnostics.cmhd.energies is <rho|u|^2>/2 etc., and the 1-D e_kin_x /
    e_mag_x below are real-space means over (z,y) of the same integrands, so
    mean_x e_kin_x == E_kin exactly whenever u_y is the only velocity component. report()
    prints that identity as its normalization check."""
    stage = cfg["stage"]
    ek, em, ei = dcmhd.energies(state, kgrid, params)
    ms, ma = dcmhd.mach_numbers(state, kgrid, params)

    # per-field inverse transforms, not one stacked (7,nz,nx,ny) array: at 256^3 the stack is
    # ~0.9 GB and only four fields are wanted.
    def real(i):
        return np.asarray(jr.grids.ifft(state.fields[i], params))

    rho, uy, by = real(0), real(2), real(5)
    if stage == 3:
        # stage 3's wave is x-polarised, so the 1-D profiles carry BOTH perpendicular
        # components: mean_x e_kin_x is then E_kin_perp (u_z excluded) and
        # e_kin_x + e_mag_x == e_zp_x + e_zm_x is the Elsasser identity report() prints.
        ux, bx = real(1), real(4)
        e_kin_x = 0.5*np.mean(rho*(ux*ux + uy*uy), axis=(0, 2))
        e_mag_x = 0.5*np.mean(bx*bx + by*by, axis=(0, 2))
    else:
        e_kin_x = 0.5*np.mean(rho*uy*uy, axis=(0, 2))  # <rho u_y^2>_{y,z}/2, over x
        e_mag_x = 0.5*np.mean(by*by, axis=(0, 2))      # <B_y^2>_{y,z}/2, over x

    # |u_y^(kx, ky=0, kz=+mz)|^2 along kx: the phase-mixing wavenumber drift. grids.fft is
    # unnormalized, so divide by N^2 to read as a contribution to <|u_y|^2>.
    nrm = float(params.nx*params.ny*params.nz)**2
    Pkx = np.fft.fftshift(np.abs(np.asarray(state.fields[2][int(cfg["mz"]), :, 0]))**2/nrm)

    # 0.5<|u|^2> split by ky == 0 / ky != 0 (Parseval with the rfft2 y-doubling factor).
    uk = np.asarray(state.fields[1:4])
    q = 0.5*(np.abs(uk)**2).sum(0)*np.asarray(kgrid.yfac)/nrm
    row = dict(t=float(state.t), E_kin=float(ek), E_mag=float(em), E_int=float(ei),
               M_s=float(ms), M_A=float(ma),
               rho_min=dcmhd.rho_min(state, kgrid, params),
               divB_max=dcmhd.divB_max(state, kgrid, params),
               E_u=float(q.sum()), E_ky=float(q[:, :, 1:].sum()),
               e_kin_x=e_kin_x, e_mag_x=e_mag_x, Pkx=Pkx)

    if stage == 3:
        # Elsasser split z^+- = u_perp +- B_perp/sqrt(rho), E^+- = <rho|z^+-|^2>_{y,z}/4.
        # With B_z > 0 and B_perp = -(B_z/c_k) u_perp the +z-going wave is z^-, so z^+ is
        # the MINOR (co-propagating) component whose existence is the point of the run.
        sr = np.sqrt(rho)
        zpx, zpy = ux + bx/sr, uy + by/sr
        zmx, zmy = ux - bx/sr, uy - by/sr
        e_zp_x = 0.25*np.mean(rho*(zpx*zpx + zpy*zpy), axis=(0, 2))
        e_zm_x = 0.25*np.mean(rho*(zmx*zmx + zmy*zmy), axis=(0, 2))
        ezp, ezm = float(e_zp_x.mean()), float(e_zm_x.mean())
        # E_ky_other: 0.5<|u|^2> at every |iky| except 0 and the driven ky0. E_ky itself is
        # dominated by the wave here, so it is E_ky_other that monitors an instability.
        keep = np.ones(q.shape[-1], dtype=bool)
        keep[0] = False
        if int(cfg["ky0"]) < q.shape[-1]:
            keep[int(cfg["ky0"])] = False
        dmax, dclear = _div_u(state, kgrid, params, cfg)
        ix1 = int(round(0.25*cfg["nx"])) % cfg["nx"]      # grid column nearest x_1 = Lx/4
        row.update(e_zp_x=e_zp_x, e_zm_x=e_zm_x, E_zp=ezp, E_zm=ezm,
                   E_minor_frac=ezp/max(ezp + ezm, 1e-300),
                   E_ky_other=float(q[:, :, keep].sum()),
                   divu_max=dmax, divu_max_clear=dclear,
                   ux_iface=np.array(ux[:, ix1, 0]))

    if stage in (2, 3):
        kb, sk, sm, sd = dcmhd.spectra(state, kgrid, params)
        row.update(spec_kin=np.asarray(sk), spec_mag=np.asarray(sm), spec_rho=np.asarray(sd))
        if static is not None and "kbins" not in static:
            static["kbins"] = np.asarray(kb)     # bin edges are static per grid
        idx = len(trace["t"])
        if stage == 2:
            ux = real(1)
        planes = [("rho", rho), ("u_x", ux), ("u_y", uy), ("B_y", by)]
        if stage == 3:
            planes.append(("B_x", bx))           # the wave is x-polarised at stage 3
        np.savez(os.path.join(diag, f"slices_{idx:04d}.npz"), t=np.float64(state.t),
                 # [x,y] plane at z index 0 and [x,z] plane at y index 0, float32
                 **{f"{n}_xy": f[0].astype(np.float32) for n, f in planes},
                 **{f"{n}_xz": f[:, :, 0].T.astype(np.float32) for n, f in planes})

    for c in trace:
        trace[c].append(row[c])
    if verbose:
        msg = ("  t={t:9.4f} E_kin={E_kin:.6e} E_mag={E_mag:.6e} E_int={E_int:.6e} "
               "M_s={M_s:.4f} M_A={M_A:.4f} rho_min={rho_min:.5f} divB={divB_max:.3e} "
               "E_ky={E_ky:.4e}")
        if stage == 3:
            msg += ("\n            E_zp={E_zp:.6e} E_zm={E_zm:.6e} "
                    "minor={E_minor_frac:.5f} E_ky_other={E_ky_other:.4e} "
                    "divu={divu_max:.3e}")
        print(msg.format(**row), flush=True)
    return row


def _static_arrays(cfg):
    """Grid axes stored once in trace.npz so report() never has to rebuild a kgrid.
    (stage 2's spectral bin centres join them at the first record.)"""
    nx, Lx = cfg["nx"], cfg["Lx"]
    return dict(x=np.linspace(0.0, Lx, nx, endpoint=False),
                kx=np.fft.fftshift(np.fft.fftfreq(nx)*nx*2*np.pi/Lx))


# ------------------------------------------------------------------------------- make_data

_CONFIG_LOCKED = ("stage", "nx", "ny", "nz", "a", "chi", "cs0", "delta", "hyper", "mz",
                  "seed_amp", "seed", "seed_nmax", "seed_kzmax", "Lx", "Ly", "Lz")
# stage-3-only IC keys. Locked only at stage 3, so a config.json written by an earlier
# version of this file (which has no U0/ky0/kink/propagating record) still resumes at
# stages 1-2 instead of tripping the "<absent>" comparison.
_CONFIG_LOCKED_3 = ("U0", "ky0", "kink", "propagating", "bz_wave")


def _check_config(diag, cfg):
    """params.save guards the grid/eqpars; this guards the parts of the config that live only
    in the IC (a, the seed) and would otherwise change silently on a resume."""
    path = os.path.join(diag, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        locked = _CONFIG_LOCKED + (_CONFIG_LOCKED_3 if cfg["stage"] == 3 else ())
        diffs = {k: (old.get(k, "<absent>"), cfg[k]) for k in locked
                 if old.get(k, "<absent>") != cfg[k]}
        if diffs:
            raise ValueError(f"{path} records a different run configuration (saved, current): "
                             f"{diffs}. These enter the initial condition, so resuming with a "
                             f"new value would be meaningless; point at a new snap_path.")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=1, sort_keys=True)


def _append_timing(diag, entry):
    path = os.path.join(diag, "timing.json")
    log = []
    if os.path.exists(path):
        with open(path) as f:
            log = json.load(f)
    log.append(entry)
    with open(path, "w") as f:
        json.dump(log, f, indent=1)


def make_data(snap_path, stage=1, wall_budget=3300.0, nblock=None, diag_every=None,
              snap_every=None, nsnap=3, schemestr="lsrk54", **overrides):
    """Advance the stage-`stage` run in `snap_path`, resumably. Returns True when t >= t_end.

    Idempotent and restart-safe exactly like examples/gdi_3d_run.py::make_data: call it
    repeatedly (a bash loop on Kaggle, a python loop locally) until it returns True. Every
    knob of the physics is an override kwarg (nx, ny, nz, a, diss, t_end, seed_amp, chi,
    cs0, delta, hyper, cfl_safety, ...), so a launch changes resolution without editing this
    file."""
    cfg = resolve_config(stage, **overrides)
    drv = STAGE_DRIVER[cfg["stage"]]
    nblock = drv["nblock"] if nblock is None else int(nblock)
    diag_every = drv["diag_every"] if diag_every is None else float(diag_every)
    snap_every = drv["snap_every"] if snap_every is None else float(snap_every)
    if diag_every is None:                       # stage-1 defaults are relative to t_half
        diag_every = cfg["t_half"]/60.0
    if snap_every is None:
        snap_every = cfg["t_half"]/4.0
    cfg["diag_every"], cfg["snap_every"], cfg["scheme"] = diag_every, snap_every, schemestr

    params = make_params(cfg)
    params.save(snap_path)
    diag = diag_dir(snap_path)
    os.makedirs(diag, exist_ok=True)
    _check_config(diag, cfg)

    # a block must not be longer than the requested diagnostic cadence, or the records come
    # out at the block cadence instead. At the production resolutions nblock is unchanged.
    nblk = min(nblock, max(1, int(round(diag_every/cfg["dt_est"]))))
    print(f"=== cmhd channel, stage {cfg['stage']}: {cfg['nx']}x{cfg['ny']}x{cfg['nz']}, "
          f"a={cfg['a']}, hyper={cfg['hyper']}, scheme={schemestr}, "
          f"TARANIS_PRECISION={cfg['precision']} ===")
    print(f"  profile: chi={cfg['chi']} cs0={cfg['cs0']} delta={cfg['delta']:.5f} -> "
          f"rho_in={cfg['rho_in']:.3f} B_z,in={cfg['bz_in']:.4f} v_A,in={cfg['vA_in']:.4f} "
          f"beta_out={cfg['beta_out']:.3f} beta_in={cfg['beta_in']:.3f}")
    print(f"  k_max={cfg['k_max']:.4f} (2/3 cut)  v'_max={cfg['vp_max']:.6f} (np.gradient on "
          f"the grid)  k_z={cfg['kz']:.4f}")
    print(f"  t_half={cfg['t_half']:.4f}  D_rule={cfg['D_rule']:.6e}  diss={cfg['diss']}  "
          f"t_end={cfg['t_end']:.4f}")
    if cfg["stage"] == 3:
        print(f"  surface wave: U0={cfg['U0']} ky0={cfg['ky0']} (k_y={cfg['ky_wave']:.4f}) "
              f"{'kink' if cfg['kink'] else 'sausage'}, "
              f"{'propagating +z' if cfg['propagating'] else 'standing'}"
              f"{', bz_wave (solenoidal)' if cfg['bz_wave'] else ''}")
        print(f"  c_k={cfg['c_k']:.6f} (between v_A,in={cfg['vA_in']:.4f} and 1); "
              f"period 2pi/(k_z c_k)={cfg['kink_period']:.4f}; minor/major amplitude ratio "
              f"{cfg['minor_ratio_out']:+.4f} out, {cfg['minor_ratio_in']:+.4f} in")
    print(f"  dt_est={cfg['dt_est']:.6g} -> ~{cfg['nsteps_est']} steps; nblock={nblk}"
          + (f" (capped from {nblock} by diag_every={diag_every:.4g})" if nblk != nblock else "")
          + f", diag_every={diag_every:.4g}, snap_every={snap_every:.4g}, nsnap={nsnap}")

    kgrid = jr.setup_kgrids(params)
    stepper, scheme = get_scheme(schemestr)
    advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))

    steps = sn.get_saved_steps(snap_path)
    if steps:
        state = sn.load_snapshot(steps[-1], snap_path, params)
        next_snap = steps[-1] + 1
        print(f"  resuming from snapshot {steps[-1]} at t={float(state.t):.6f}")
    else:
        state = jr.initialize(make_ic(cfg), params)
        next_snap = 0

    trace, static = _load_trace(diag, cfg["stage"], float(state.t))
    static = {**_static_arrays(cfg), **static}
    if cfg["stage"] in (2, 3):
        _prune_slices(diag, len(trace["t"]))

    mngr = jr.snapshot_manager_setup(params=params, snap_path=snap_path, nsnap=nsnap)
    if next_snap == 0:
        sn.save_snapshot(0, state, mngr, params)
        mngr.wait_until_finished()
        next_snap = 1
    if not trace["t"]:
        record(state, kgrid, params, cfg, trace, diag, static)
        _save_trace(diag, trace, static)

    if float(state.t) >= cfg["t_end"] - 1e-12:
        print(f"make_data: already complete (t={float(state.t):.6f} >= t_end={cfg['t_end']:.6f})")
        return True

    # compile once, timed on its own, so the ms/step below is steps and not XLA
    t0 = time.perf_counter()
    advance.lower(state, kgrid, params, nblk, scheme, stepper).compile()
    compile_s = time.perf_counter() - t0
    print(f"  compiled block_of_steps(nblock={nblk}) in {compile_s:.1f}s", flush=True)

    t_last_diag = float(trace["t"][-1])
    t_last_snap = float(state.t)
    wall0 = time.perf_counter()
    step_wall, nsteps = 0.0, 0
    while float(state.t) < cfg["t_end"] - 1e-12 and time.perf_counter() - wall0 < wall_budget:
        t1 = time.perf_counter()
        state = advance(state, kgrid, params, nblk, scheme, stepper)
        state.fields.block_until_ready()
        step_wall += time.perf_counter() - t1
        nsteps += nblk
        t_now = float(state.t)
        if not np.isfinite(t_now):
            raise RuntimeError("t went non-finite -- the run blew up (check rho_min in the "
                               "trace above; under-dissipated for this grid?)")
        if t_now - t_last_diag >= diag_every or t_now >= cfg["t_end"] - 1e-12:
            record(state, kgrid, params, cfg, trace, diag, static)
            _save_trace(diag, trace, static)
            t_last_diag = t_now
            if not np.isfinite(trace["E_kin"][-1]):
                raise RuntimeError("the run went non-finite; see rho_min in the trace above")
        if t_now - t_last_snap >= snap_every or t_now >= cfg["t_end"] - 1e-12:
            sn.save_snapshot(next_snap, state, mngr, params)
            mngr.wait_until_finished()
            next_snap += 1
            t_last_snap = t_now
    if float(state.t) > t_last_snap:
        sn.save_snapshot(next_snap, state, mngr, params)
        next_snap += 1
    mngr.wait_until_finished()
    _save_trace(diag, trace, static)

    ms = 1e3*step_wall/nsteps if nsteps else float("nan")
    _append_timing(diag, dict(t_reached=float(state.t), nsteps=nsteps,
                              step_wall_s=step_wall, ms_per_step=ms, compile_s=compile_s,
                              nblock=nblk, scheme=schemestr, nx=cfg["nx"], ny=cfg["ny"],
                              nz=cfg["nz"], precision=cfg["precision"],
                              when=time.strftime("%Y-%m-%d %H:%M:%S")))
    done = float(state.t) >= cfg["t_end"] - 1e-12
    print(f"make_data: t={float(state.t):.6f} (target {cfg['t_end']:.6f}); {nsteps} steps in "
          f"{step_wall:.1f}s = {ms:.2f} ms/step (+{compile_s:.1f}s compile); done={done}")
    return done


# ---------------------------------------------------------------------------- the gate law

def predicted_ln_ratio(t, vp, cfg, nt=4001):
    """ln E(x,t)/E(x,0) from (*): -2 D int_0^t (kz^2 v'(x)^2 t'^2 + kz^2)^hyper dt'.

    Numerical in t' (np.trapezoid on a fine grid) for every hyper, and the k_z^2 term is
    kept -- at small v' it is the whole of the damping."""
    D = cfg["diss"][1] if np.shape(cfg["diss"]) != () else cfg["diss"]
    kz2, hyper = cfg["kz"]**2, int(cfg["hyper"])
    if t <= 0.0:
        return np.zeros_like(vp)
    tp = np.linspace(0.0, t, nt)
    integrand = (kz2*vp[:, None]**2*tp[None, :]**2 + kz2)**hyper
    return -2.0*D*_trapz(integrand, tp, axis=1)


def linear_reference(cfg, times, h=0.02):
    """ln E(x,t)/E(x,0) from the EXACT linearised phase-mixing problem, on the run's own x
    grid -- the reference that says whether a gap in the gate above is the SOLVER or the WKB
    law itself.

    With d/dy = 0 and a single k_z, the y-polarised pair is (u_y, B_y) = Re[(u^, b^)(x,t)
    e^{i k_z z}] obeying, with no approximation beyond linearity,

        d_t u^ = i k_z (B_z/rho) b^ + L u^,   d_t b^ = i k_z B_z u^ + L b^,
        L = -D (-d_x^2 + k_z^2)^hyper,        u^(x,0) = a,  b^(x,0) = 0

    and E(x,t) = (rho |u^|^2 + |b^|^2)/4, which at t = 0 is rho a^2/4 exactly. Both halves
    have exact sub-flows -- the advective one is a pointwise rotation at omega = k_z v_A(x)
    in the variables u^ +- b^/sqrt(rho), the dissipative one a multiply in k_x -- so this is
    Strang over two exact flows and its only error is the splitting one (converged to 5
    digits at h = 0.04 on both stage-1 grids; h = 0.02 is the default here).

    This is NOT the gate. It is the same physics the gate law approximates, integrated
    instead of expanded, and its cost is a second on the production grid."""
    D = cfg["diss"][1] if np.shape(cfg["diss"]) != () else cfg["diss"]
    nx, Lx, kz, hyper = cfg["nx"], cfg["Lx"], cfg["kz"], int(cfg["hyper"])
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    rho, bz = _profile(x, cfg, xp=np)
    sr = np.sqrt(rho)
    omega = kz*bz/sr
    kx = np.fft.fftfreq(nx)*nx*2*np.pi/Lx
    ksq = kx*kx + kz*kz
    u = np.full(nx, cfg["a"], dtype=complex)
    b = np.zeros(nx, dtype=complex)
    E0 = (rho*np.abs(u)**2 + np.abs(b)**2)/4.0
    out, t = [], 0.0
    for T in times:
        while t < T - 1e-12:
            step = min(h, T - t)
            f = np.exp(-D*ksq**hyper*(step/2))
            u = np.fft.ifft(f*np.fft.fft(u))
            b = np.fft.ifft(f*np.fft.fft(b))
            p = (u + b/sr)*np.exp(1j*omega*step)
            mn = (u - b/sr)*np.exp(-1j*omega*step)
            u, b = 0.5*(p + mn), sr*0.5*(p - mn)
            u = np.fft.ifft(f*np.fft.fft(u))
            b = np.fft.ifft(f*np.fft.fft(b))
            t += step
        out.append(np.log(((rho*np.abs(u)**2 + np.abs(b)**2)/4.0)/E0))
    return np.array(out)


def _kx_edges(P, kx, eps=1e-3):
    """(k at 99.9% of the k_x > 0 power, largest k_x carrying eps of the peak) for one
    P(kx) record -- two parameter-light estimators of the phase-mixing spectral edge."""
    g = kx > 0
    k, p = kx[g], P[g]
    c = np.cumsum(p)/max(p.sum(), 1e-300)
    kq = k[min(np.searchsorted(c, 0.999), len(k) - 1)]
    sm = np.maximum.reduce([np.r_[p[1:], 0.0], p, np.r_[0.0, p[:-1]]])
    idx = np.nonzero(sm >= eps*sm.max())[0]
    return float(kq), float(k[idx[-1]]) if len(idx) else np.nan


def _fit_exponential_window(t, y, minlen=4):
    """Longest window over which ln y is a straight line to R^2 >= 0.99 with positive slope.
    Returns (i, j, slope, r2) or None."""
    ly = np.log(np.where(y > 0, y, np.nan))
    ok = np.isfinite(ly)
    best = None
    n = len(t)
    for i in range(n):
        for j in range(i + minlen, n + 1):
            s = slice(i, j)
            if not ok[s].all():
                continue
            p = np.polyfit(t[s], ly[s], 1)
            if p[0] <= 0:
                continue
            resid = ly[s] - np.polyval(p, t[s])
            ss = np.sum((ly[s] - ly[s].mean())**2)
            r2 = 1.0 - np.sum(resid**2)/ss if ss > 0 else 0.0
            if r2 >= 0.99 and (best is None or (j - i) > best[1] - best[0]
                               or ((j - i) == best[1] - best[0] and r2 > best[3])):
                best = (i, j, float(p[0]), float(r2))
    return best


def report(snap_path):
    """Read diag/config.json + diag/trace.npz and print the stage's summary. No snapshots."""
    diag = diag_dir(snap_path)
    with open(os.path.join(diag, "config.json")) as f:
        cfg = json.load(f)
    with np.load(os.path.join(diag, "trace.npz")) as z:
        tr = {k: z[k] for k in z.files}
    t = tr["t"]
    print(f"=== cmhd channel report: {snap_path} ===")
    print(f"  stage {cfg['stage']}, {cfg['nx']}x{cfg['ny']}x{cfg['nz']}, a={cfg['a']}, "
          f"diss={cfg['diss']}, hyper={cfg['hyper']}, t_end={cfg['t_end']:.4f}, "
          f"precision={cfg.get('precision')}")
    print(f"  {len(t)} records, t in [{t.min():.4f}, {t.max():.4f}]")
    tpath = os.path.join(diag, "timing.json")
    if os.path.exists(tpath):
        with open(tpath) as f:
            for e in json.load(f):
                print(f"  timing: {e['nsteps']:7d} steps  {e['ms_per_step']:8.2f} ms/step  "
                      f"(compile {e['compile_s']:.1f}s, nblock={e['nblock']}, "
                      f"scheme={e['scheme']}) -> t={e['t_reached']:.4f}")

    print("\n-- energies (volume averages; E_mag and E_int include the k=0 background) --")
    ew_kin = tr["e_kin_x"].mean(axis=1)
    ew_mag = tr["e_mag_x"].mean(axis=1)
    for i in range(len(t)):
        print(f"  t={t[i]:9.4f} E_kin={tr['E_kin'][i]:.6e} E_mag={tr['E_mag'][i]:.6e} "
              f"E_int={tr['E_int'][i]:.6e} Ew_kin={ew_kin[i]:.6e} Ew_mag={ew_mag[i]:.6e} "
              f"Ew_tot={ew_kin[i]+ew_mag[i]:.6e} rho_min={tr['rho_min'][i]:.5f} "
              f"M_s={tr['M_s'][i]:.4f} M_A={tr['M_A'][i]:.4f} divB={tr['divB_max'][i]:.2e}")

    # normalization check: mean_x e_kin_x IS <rho|u|^2>/2 whenever u_y is the only component
    if cfg["stage"] == 3:
        frac = ew_kin/np.maximum(tr["E_kin"], 1e-300)
        print("\n-- normalization: at stage 3 e_kin_x/e_mag_x are the PERPENDICULAR energy "
              "profiles\n   (both of u_x,u_y and B_x,B_y), so mean_x e_kin_x = E_kin_perp <= "
              "E_kin, not E_kin itself:\n   "
              f"mean_x e_kin_x/E_kin runs over [{frac.min():.6f}, {frac.max():.6f}] "
              f"(the deficit is u_z)")
    else:
        rel = np.abs(ew_kin - tr["E_kin"])/np.maximum(tr["E_kin"], 1e-300)
        print(f"\n-- normalization: max |mean_x e_kin_x - E_kin|/E_kin = {rel.max():.3e} "
              f"(exact identity while u_y is the only velocity component)")

    kyfrac = tr["E_ky"]/np.maximum(tr["E_u"], 1e-300)
    print(f"-- k_y != 0 content of u: max E_ky = {tr['E_ky'].max():.6e}, "
          f"max E_ky/E_u = {kyfrac.max():.6e}")

    if cfg["stage"] == 1:
        _report_stage1(cfg, tr, diag, ew_kin, ew_mag)
    elif cfg["stage"] == 3:
        _report_stage3(cfg, tr, diag, ew_kin, ew_mag)
    else:
        _report_stage2(cfg, tr, diag)
    _plots(cfg, tr, diag)


def _report_stage1(cfg, tr, diag, ew_kin, ew_mag):
    t, x = tr["t"], tr["x"]
    rho_x, bz_x = _profile(x, cfg, xp=np)
    vA = bz_x/np.sqrt(rho_x)
    vp = np.abs(np.gradient(vA, cfg["Lx"]/cfg["nx"]))

    print("\n-- ky != 0 gate (a y-independent IC must stay y-independent EXACTLY) --")
    print(f"  max E_ky over the run = {tr['E_ky'].max():.6e} -> "
          f"{'PASS (identically zero)' if tr['E_ky'].max() == 0.0 else 'see above'}")

    E = tr["e_kin_x"] + tr["e_mag_x"]            # E(x,t) = <rho u_y^2 + B_y^2>_{y,z}/2
    E0 = E[0]
    ln_meas = np.log(np.where(E > 0, E, np.nan)/E0)
    worst_abs, worst_rel, worst_at = 0.0, 0.0, None
    strict_rel = 0.0
    print("\n-- THE GATE: pointwise-in-x phase-mixing law (*), per record --")
    print("     t      npts   max|dln|   max|dln|/|ln_pred|   (over -8 <= ln_pred <= -1e-3)")
    for i in range(1, len(t)):
        ln_pred = predicted_ln_ratio(float(t[i]), vp, cfg)
        m = (ln_pred >= -8.0) & (ln_pred <= -1e-3) & np.isfinite(ln_meas[i])
        if not m.any():
            print(f"  {t[i]:9.4f}      0   (no x in the mask yet)")
            continue
        d = np.abs(ln_meas[i][m] - ln_pred[m])
        r = d/np.abs(ln_pred[m])
        s = (np.abs(ln_pred) >= 1.0) & m
        if s.any():
            strict_rel = max(strict_rel, float(np.max(np.abs(ln_meas[i][s]-ln_pred[s])
                                                      / np.abs(ln_pred[s]))))
        if d.max() > worst_abs:
            worst_abs, worst_at = float(d.max()), float(t[i])
        worst_rel = max(worst_rel, float(r.max()))
        print(f"  {t[i]:9.4f}  {m.sum():5d}   {d.max():.4e}   {r.max():.4e}")
    print(f"  GATE: max over x and records of |ln(E/E0)_meas - ln(E/E0)_pred| = "
          f"{worst_abs:.4e} (at t={worst_at})")
    print(f"        the same as a fraction of |ln predicted|            = {worst_rel:.4e}")
    print(f"        restricted to |ln predicted| >= 1 (damping is O(1)) = {strict_rel:.4e}")

    # The gate law is leading-order WKB. Where it and the run disagree, this says which of
    # the two is wrong: the same linear physics, integrated instead of expanded.
    ref = linear_reference(cfg, [float(v) for v in t[1:]])
    sr_abs, sr_rel, wr_abs = 0.0, 0.0, 0.0
    print("\n-- the same comparison against the EXACT linear solution (linear_reference) --")
    print("     t      npts   max|ln_meas - ln_exact|   rel(|ln_exact|>=1)  |  "
          "max|ln_exact - ln_pred|")
    for k in range(len(ref)):
        i = k + 1
        ln_pred = predicted_ln_ratio(float(t[i]), vp, cfg)
        mm = (ln_pred >= -8.0) & (ln_pred <= -1e-3) & np.isfinite(ln_meas[i])
        if not mm.any():
            continue
        d = np.abs(ln_meas[i] - ref[k])
        w = np.abs(ref[k] - ln_pred)
        st = mm & (np.abs(ref[k]) >= 1.0)
        r = float(np.max((d/np.abs(ref[k]))[st])) if st.any() else np.nan
        sr_abs, wr_abs = max(sr_abs, float(d[mm].max())), max(wr_abs, float(w[mm].max()))
        sr_rel = max(sr_rel, r) if np.isfinite(r) else sr_rel
        if i % max(1, len(t)//12) == 0 or i == len(t) - 1:
            print(f"  {t[i]:9.4f}  {mm.sum():5d}   {d[mm].max():.4e}          {r:.4e}   |  "
                  f"{w[mm].max():.4e}")
    print(f"  SOLVER vs EXACT LINEAR: max|dln| = {sr_abs:.4e}, max relative where "
          f"|ln_exact| >= 1 = {sr_rel:.4e}")
    print(f"  EXACT LINEAR vs THE GATE LAW: max|dln| = {wr_abs:.4e}  <- the WKB law's own "
          f"error; it is resolution independent and grows with the accumulated damping")

    print("\n-- k_x drift: the high-k_x EDGE of P(kx) = |u_y^(kx, ky=0, kz=+1)|^2 vs "
          "k_z v'_max t --")
    kx = tr["kx"]
    print("      t     kx_pred    kx(99.9% power)  ratio    kx(1e-3 of peak)  ratio")
    for i in np.unique(np.linspace(1, len(t)-1, 9).astype(int)):
        kx_pred = cfg["kz"]*cfg["vp_max"]*float(t[i])
        kq, ke = _kx_edges(tr["Pkx"][i], kx)
        print(f"  {t[i]:9.4f}  {kx_pred:9.4f}  {kq:15.2f}  {kq/kx_pred:6.3f}  "
              f"{ke:16.2f}  {ke/kx_pred:6.3f}")
    print("  The observable is the spectral EDGE, not a peak: most of the box is uniform and\n"
          "  contributes at k_x = 0, and the stationary-phase caustic at k_x = k_z v'_max t\n"
          "  sits exactly where the damping is strongest, so it never stands proud of the\n"
          "  pedestal. (P(kx) also alternates strongly even/odd -- the two layers are Lx/2\n"
          "  apart, so the structure factor 1 + (-1)^{i k_x} kills odd k_x; the 1e-3 estimator\n"
          "  runs a 3-bin max first for that reason.) Expect ratios near 1 through the middle\n"
          "  of the run and a fall-off at the end, when dissipation has removed the leading\n"
          "  edge faster than phase mixing creates it.")

    print("\n-- E_kin <-> E_mag exchange (wave parts, mean_x of the 1-D profiles) --")
    tot = ew_kin + ew_mag
    print(f"  E(0) = {tot[0]:.6e}  (expected <rho> a^2/4 = "
          f"{np.mean(rho_x)*cfg['a']**2/4:.6e})")
    print(f"  max E_kin/E_tot = {np.max(ew_kin/tot):.4f}, min = {np.min(ew_kin/tot):.4f} "
          f"(a standing wave exchanges the full amplitude at 2 omega; a spread of omega(x) "
          f"over the box damps the box-averaged swing)")


def _report_stage2(cfg, tr, diag):
    t = tr["t"]
    print("\n-- E_ky (0.5<|u|^2> restricted to k_y != 0): the KHI monitor --")
    for i in range(len(t)):
        print(f"  t={t[i]:9.4f}  E_ky={tr['E_ky'][i]:.6e}  E_u={tr['E_u'][i]:.6e}  "
              f"E_ky/E_u={tr['E_ky'][i]/max(tr['E_u'][i], 1e-300):.4e}")
    fit = _fit_exponential_window(t, tr["E_ky"])
    if fit is None:
        print("  no clean exponential window (R^2 >= 0.99 over >= 4 records with a positive "
              "slope) -- report the trace above as it is, not a growth rate")
    else:
        i, j, slope, r2 = fit
        print(f"  exponential window t in [{t[i]:.4f}, {t[j-1]:.4f}] ({j-i} records): "
              f"d ln E_ky/dt = {slope:.5f} (R^2 = {r2:.5f}) -> amplitude growth rate "
              f"gamma = {slope/2:.5f}")
        print(f"  cf. the KHI expectation gamma ~ a k_x/2 with k_x ~ k_z v'_max t: "
              f"{cfg['a']*cfg['kz']*cfg['vp_max']*t[(i+j)//2]/2:.4f} at the window centre")
    print(f"\n-- rho_min over the run: {tr['rho_min'].min():.6f} "
          f"({'POSITIVE' if tr['rho_min'].min() > 0 else 'NON-POSITIVE -- rarefaction'})")
    print(f"-- max |div B| metric: {tr['divB_max'].max():.3e}")
    ns = len([n for n in os.listdir(diag) if n.startswith("slices_")])
    print(f"-- {ns} slice files in {diag} (slices_NNNN.npz: rho/u_x/u_y/B_y as float32 "
          f"[x,y] planes at z index 0 and [x,z] planes at y index 0)")


def _phase_speed(cfg, t, ux_iface, nper=2.0):
    """The measured phase speed from ux_iface = u_x(x_1, y = 0, z), or None.

    The discrete Fourier coefficient of that column at +mz,
    c(t) = (1/nz) sum_z u_x e^{-i k_z z}, is (A/2) e^{-i w t} for a +z-going wave
    u_x ~ A cos(k_z z - w t): the unwrapped phase falls LINEARLY at -w, so a straight-line
    fit over the first `nper` periods gives c_meas = -slope/k_z. (A standing wave has a
    stationary phase that flips by pi, i.e. slope ~ 0 -- which is the point of measuring it
    this way rather than by tracking a zero crossing.)"""
    mz, nz = int(cfg["mz"]), ux_iface.shape[1]
    c = np.fft.fft(np.asarray(ux_iface), axis=1)[:, mz]/nz
    amp = np.abs(c)
    m = (t <= min(t.max(), nper*cfg["kink_period"]) + 1e-9) & (amp > 0.05*max(amp[0], 1e-300))
    if m.sum() < 3:
        return None
    ph = np.unwrap(np.angle(c[m]))
    pf = np.polyfit(t[m], ph, 1)
    resid = ph - np.polyval(pf, t[m])
    return dict(c_meas=float(-pf[0]/cfg["kz"]), slope=float(pf[0]), npts=int(m.sum()),
                t0=float(t[m][0]), t1=float(t[m][-1]),
                max_resid=float(np.max(np.abs(resid))),
                dphase_per_record=float(np.max(np.abs(np.diff(ph)))),
                amp0=float(amp[0]), amp_end=float(amp[m][-1]))


def _report_stage3(cfg, tr, diag, ew_kin, ew_mag):
    t, x, ck = tr["t"], tr["x"], cfg["c_k"]
    rho_x, bz_x = _profile(x, cfg, xp=np)
    vA = bz_x/np.sqrt(rho_x)

    print("\n-- Elsasser split: E^+- = <rho |u_perp +- B_perp/sqrt(rho)|^2>_{y,z}/4 --")
    print("   B_perp = -(B_z/c_k) u_perp sends the wave toward +z, so z^- is the MAJOR")
    print("   component and z^+ the co-propagating MINOR one (the uniturbulence driver).")
    ident, perp = tr["E_zp"] + tr["E_zm"], ew_kin + ew_mag
    for i in range(len(t)):
        print(f"  t={t[i]:9.4f} E_zp={tr['E_zp'][i]:.6e} E_zm={tr['E_zm'][i]:.6e} "
              f"minor={tr['E_minor_frac'][i]:.6f}  E_zp+E_zm={ident[i]:.6e}  "
              f"E_kinperp+E_magperp={perp[i]:.6e}")
    rel = np.abs(ident - perp)/np.maximum(perp, 1e-300)
    print(f"  IDENTITY E_zp + E_zm == E_kin_perp + E_mag_perp: max relative gap = "
          f"{rel.max():.3e} (round-off; both are the same two real-space quadratics)")

    print("\n-- t = 0 minor content vs the sharp-interface theory, at x_i +- 0.4 --")
    print("   r(x) = (c_k - v_A(x))/(c_k + v_A(x)) is the minor/major AMPLITUDE ratio, so")
    print("   E_minor/E_major = r^2 and E_minor/(E^+ + E^-) = r^2/(1 + r^2). Both are")
    print("   printed: the brief's 'minor fraction' is the second, r^2 alone is the first.")
    zp0, zm0 = tr["e_zp_x"][0], tr["e_zm_x"][0]
    print("       x     side    v_A       r        r^2      E+/E- meas   r^2/(1+r^2)  "
          "frac meas")
    for xi, _ in _interfaces(cfg):
        for off in (-0.4, 0.4):
            xq = (xi + off) % cfg["Lx"]
            j = int(np.argmin(np.abs((x - xq + cfg["Lx"]/2) % cfg["Lx"] - cfg["Lx"]/2)))
            r = (ck - vA[j])/(ck + vA[j])
            tot = max(zp0[j] + zm0[j], 1e-300)
            print(f"  {x[j]:7.4f}  {'in ' if rho_x[j] > 2.0 else 'out'}  {vA[j]:7.4f}  "
                  f"{r:+.5f}  {r*r:.6f}   {zp0[j]/max(zm0[j], 1e-300):.6f}     "
                  f"{r*r/(1+r*r):.6f}    {zp0[j]/tot:.6f}")
    print(f"  box minor fraction at t=0: {tr['E_minor_frac'][0]:.6f}  (a box mean over an")
    print("   x-varying r, so it is not either of the two single-x numbers above)")

    print("\n-- phase speed from ux_iface = u_x(x_1, y=0, z), k_z = mz Fourier phase --")
    fit = _phase_speed(cfg, t, tr["ux_iface"])
    if fit is None:
        print("  too few usable records for a fit")
    else:
        print(f"  fit over t in [{fit['t0']:.4f}, {fit['t1']:.4f}] ({fit['npts']} records, "
              f"max |dphase| between records {fit['dphase_per_record']:.4f} rad -- unwrap is "
              f"safe below pi)")
        print(f"  d(phase)/dt = {fit['slope']:.6f} -> c_meas = {fit['c_meas']:.6f} vs "
              f"c_k = {ck:.6f}  ({100*(fit['c_meas']-ck)/ck:+.2f}%)")
        print(f"  max residual about the straight line = {fit['max_resid']:.4f} rad; "
              f"|c(mz)| {fit['amp0']:.6e} -> {fit['amp_end']:.6e} over the window")
        print("  A NEGATIVE slope is the wave going toward +z. Expect a few percent: the")
        print("  sharp-interface c_k ignores the finite layer width and resonant absorption.")

    print("\n-- E_ky_other (0.5<|u|^2> at |iky| not in {0, ky0}): the stage-3 instability "
          "monitor --")
    for i in range(len(t)):
        print(f"  t={t[i]:9.4f}  E_ky_other={tr['E_ky_other'][i]:.6e}  "
              f"E_ky={tr['E_ky'][i]:.6e}  E_u={tr['E_u'][i]:.6e}  "
              f"E_ky_other/E_u={tr['E_ky_other'][i]/max(tr['E_u'][i], 1e-300):.4e}")
    f3 = _fit_exponential_window(t, tr["E_ky_other"])
    if f3 is None:
        print("  no clean exponential window (R^2 >= 0.99 over >= 4 records with a positive "
              "slope) -- report the trace above as it is, not a growth rate")
    else:
        i, j, slope, r2 = f3
        print(f"  exponential window t in [{t[i]:.4f}, {t[j-1]:.4f}] ({j-i} records): "
              f"d ln E_ky_other/dt = {slope:.5f} (R^2 = {r2:.5f}) -> amplitude growth rate "
              f"gamma = {slope/2:.5f}")

    print("\n-- div u --")
    print(f"  analytic IC, central differences away from the interfaces: "
          f"max|div u|/(k_y U0) = {div_u_analytic(cfg):.3e}")
    print(f"  on the grid (spectral, dealiased IC, SEED INCLUDED -- the seed is not "
          f"solenoidal): t=0\n     max = {tr['divu_max'][0]:.4e} (at the interface kinks), "
          f"more than 0.4 from either interface {tr['divu_max_clear'][0]:.4e} "
          f"(units of k_y U0)")
    print(f"  over the run: max = {tr['divu_max'].max():.4e}, max outside the interfaces = "
          f"{tr['divu_max_clear'].max():.4e}")

    print("\n-- div B: NOT round-off at stage 3, and constant, BY CONSTRUCTION --")
    print(f"  divB_max over the run: min {tr['divB_max'].min():.4e}, max "
          f"{tr['divB_max'].max():.4e} (curl induction preserves div B exactly, so this is")
    print("  the IC's own divergence, frozen for the whole run -- not a solver failure).")
    if cfg["propagating"] and not cfg.get("bz_wave", False):
        print("  B_perp = -(B_z/c_k) u_perp ALONE is not solenoidal: div B = -(B_z'/c_k) u_x,")
        print("  which peaks in the layers where B_z' is largest. Measured on the seed-free")
        print("  IC: max|div B| is 0.53 of k_y max|B_perp| and does NOT fall with resolution")
        print("  (the divB_max metric is 9.12e-4 at nx=128 and 9.14e-4 at nx=256). Setting")
        print("  bz_wave=True adds the eigenfunction's b_z = -B_z'(x) xi_x (see _bz_wave),")
        print("  after which the cancellation is exact and the residual is only the |d| kink:")
        print("  the same metric becomes 2.33e-5 at nx=128 and 5.84e-6 at nx=256, i.e. 1/nx^2.")
    elif cfg.get("bz_wave", False):
        print("  bz_wave=True: the IC is the exact linear b = curl(xi x B), so what is left")
        print("  is the |d|-kink truncation only and it converges as ~1/nx^2.")

    print(f"\n-- rho_min over the run: {tr['rho_min'].min():.6f} "
          f"({'POSITIVE' if tr['rho_min'].min() > 0 else 'NON-POSITIVE -- rarefaction'})")
    ns = len([n for n in os.listdir(diag) if n.startswith("slices_")])
    print(f"-- {ns} slice files in {diag} (slices_NNNN.npz: rho/u_x/u_y/B_x/B_y as float32 "
          f"[x,y] planes at z index 0 and [x,z] planes at y index 0)")
    print(f"-- spectra: trace.npz carries spec_kin/spec_mag/spec_rho on kbins "
          f"({len(tr['kbins'])} bins to the grid corner; diagnostics.cmhd.spectra)")


def _plots(cfg, tr, diag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                      # a bare laptop may not have it; Kaggle does
        print(f"\n(no plots: {exc})")
        return
    t = tr["t"]
    if cfg["stage"] == 3:
        _plots_stage3(cfg, tr, diag, plt)
        return
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    a = ax[0, 0]
    a.semilogy(t, tr["E_kin"], label="E_kin")
    a.semilogy(t, tr["e_mag_x"].mean(axis=1), label="E_mag (wave)")
    if tr["E_ky"].max() > 0:
        a.semilogy(t, tr["E_ky"], label="E_ky")
    a.set_xlabel("t")
    a.legend()
    a.set_title(f"stage {cfg['stage']} energies")

    a = ax[0, 1]
    for i in np.unique(np.linspace(0, len(t)-1, 5).astype(int)):
        a.plot(tr["x"], tr["e_kin_x"][i] + tr["e_mag_x"][i], label=f"t={t[i]:.2f}")
    a.set_xlabel("x")
    a.set_ylabel("E(x,t)")
    a.set_yscale("log")
    a.legend(fontsize=7)
    a.set_title("wave energy vs x")

    a = ax[1, 0]
    for i in np.unique(np.linspace(1, len(t)-1, 5).astype(int)):
        a.semilogy(tr["kx"], np.maximum(tr["Pkx"][i], 1e-40), label=f"t={t[i]:.2f}")
    a.set_xlabel("k_x")
    a.set_xlim(0, cfg["k_max"])
    a.legend(fontsize=7)
    a.set_title("P(kx) at ky=0, kz=+1")

    a = ax[1, 1]
    if cfg["stage"] == 1:
        rho_x, bz_x = _profile(tr["x"], cfg, xp=np)
        vp = np.abs(np.gradient(bz_x/np.sqrt(rho_x), cfg["Lx"]/cfg["nx"]))
        E = tr["e_kin_x"] + tr["e_mag_x"]
        sel = np.unique(np.linspace(1, len(t)-1, 4).astype(int))
        ref = linear_reference(cfg, [float(t[i]) for i in sel])
        for n, i in enumerate(sel):
            lm = np.log(np.where(E[i] > 0, E[i], np.nan)/E[0])
            lp = predicted_ln_ratio(float(t[i]), vp, cfg)
            ln = a.plot(tr["x"], lm, label=f"run t={t[i]:.2f}")[0]
            a.plot(tr["x"], lp, "--", color=ln.get_color(), label=f"WKB law t={t[i]:.2f}")
            a.plot(tr["x"], ref[n], ":", lw=2, color=ln.get_color(),
                   label=f"exact linear t={t[i]:.2f}")
        a.set_ylim(-12, 1)
        a.set_xlabel("x")
        a.set_ylabel("ln E/E0")
        a.legend(fontsize=6, ncol=2)
        a.set_title("the gate")
    else:
        for i in np.unique(np.linspace(0, len(t)-1, 4).astype(int)):
            a.loglog(tr["kbins"], np.maximum(tr["spec_kin"][i], 1e-40), label=f"t={t[i]:.1f}")
        a.set_xlabel("k_perp")
        a.legend(fontsize=7)
        a.set_title("kinetic spectrum")
    fig.tight_layout()
    out = os.path.join(diag, f"stage{cfg['stage']}_summary.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\n(plots written to {out})")


def _plots_stage3(cfg, tr, diag, plt):
    """The stage-3 2x2: E^+-(t) with the minor fraction, E_ky_other(t), the e_zm_x/e_zp_x
    profiles at a few times, and the kinetic spectrum."""
    t, x = tr["t"], tr["x"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    a = ax[0, 0]
    a.semilogy(t, tr["E_zm"], label="E- (major, +z)")
    a.semilogy(t, tr["E_zp"], label="E+ (minor)")
    a.semilogy(t, tr["E_kin"], ":", label="E_kin")
    a.set_xlabel("t")
    a.legend(fontsize=7, loc="lower left")
    a.set_title("Elsasser energies")
    a2 = a.twinx()
    a2.plot(t, tr["E_minor_frac"], "k--", lw=1)
    a2.set_ylabel("minor fraction E+/(E+ + E-)")

    a = ax[0, 1]
    a.semilogy(t, np.maximum(tr["E_ky_other"], 1e-300), label="E_ky_other")
    a.semilogy(t, np.maximum(tr["E_ky"], 1e-300), ":", label="E_ky (incl. ky0)")
    a.set_xlabel("t")
    a.legend(fontsize=7)
    a.set_title("k_y content outside the driven ky0")

    a = ax[1, 0]
    for i in np.unique(np.linspace(0, len(t)-1, 4).astype(int)):
        ln = a.semilogy(x, np.maximum(tr["e_zm_x"][i], 1e-40), label=f"E- t={t[i]:.2f}")[0]
        a.semilogy(x, np.maximum(tr["e_zp_x"][i], 1e-40), "--", color=ln.get_color(),
                   label=f"E+ t={t[i]:.2f}")
    for xi, _ in _interfaces(cfg):
        a.axvline(xi, color="0.7", lw=0.8)
    a.set_xlabel("x")
    a.legend(fontsize=6, ncol=2)
    a.set_title("e_zp_x / e_zm_x")

    a = ax[1, 1]
    for i in np.unique(np.linspace(0, len(t)-1, 4).astype(int)):
        a.loglog(tr["kbins"], np.maximum(tr["spec_kin"][i], 1e-40), label=f"t={t[i]:.1f}")
    a.set_xlabel("k_perp")
    a.legend(fontsize=7)
    a.set_title("kinetic spectrum")

    fig.tight_layout()
    out = os.path.join(diag, "stage3_summary.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\n(plots written to {out})")


# ------------------------------------------------------------------------------- __main__

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, default=1, choices=sorted(STAGES))
    p.add_argument("--snap-path", default=None,
                   help="default examples/data/cmhd-channel-s<stage>/checkpoints")
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--nx", type=int, default=None)
    p.add_argument("--ny", type=int, default=None)
    p.add_argument("--nz", type=int, default=None)
    p.add_argument("--a", type=float, default=None)
    p.add_argument("--diss", type=float, default=None, help="scalar diss (stage 2 form)")
    p.add_argument("--seed-amp", type=float, default=None)
    p.add_argument("--hyper", type=int, default=None)
    p.add_argument("--U0", type=float, default=None, help="stage-3 wave amplitude")
    p.add_argument("--ky0", type=int, default=None,
                   help="stage-3 perpendicular mode number (k_y = ky0*2pi/Ly)")
    p.add_argument("--sausage", action="store_true",
                   help="stage 3: opposite interface displacements (default kink)")
    p.add_argument("--standing", action="store_true",
                   help="stage 3: B_perp = 0 at t=0 (default a +z-propagating wave)")
    p.add_argument("--nblock", type=int, default=None)
    p.add_argument("--diag-every", type=float, default=None)
    p.add_argument("--snap-every", type=float, default=None)
    p.add_argument("--nsnap", type=int, default=3)
    p.add_argument("--wall-budget", type=float, default=3300.0)
    p.add_argument("--no-report", action="store_true")
    args = p.parse_args()

    snap_path = args.snap_path or f"examples/data/cmhd-channel-s{args.stage}/checkpoints"
    ov = {k: v for k, v in dict(nx=args.nx, ny=args.ny, nz=args.nz, a=args.a,
                                diss=args.diss, t_end=args.t_end, hyper=args.hyper,
                                U0=args.U0, ky0=args.ky0,
                                seed_amp=args.seed_amp).items() if v is not None}
    if args.sausage:
        ov["kink"] = False
    if args.standing:
        ov["propagating"] = False
    while not make_data(snap_path, stage=args.stage, wall_budget=args.wall_budget,
                        nblock=args.nblock, diag_every=args.diag_every,
                        snap_every=args.snap_every, nsnap=args.nsnap, **ov):
        pass
    if not args.no_report:
        report(snap_path)


if __name__ == "__main__":
    main()
