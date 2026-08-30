# The Orszag-Tang vortex: CMHD's validation gate (plans/CMHD_PLAN.md Phase C2), marked
# slow (three runs, ~2.5 min total on an M1 laptop) and fp64.
#
#   TARANIS_PRECISION=64 python -m pytest tests/test_cmhd_orszag_tang.py -m slow --runslow
#   TARANIS_PRECISION=64 RMHD_RUNSLOW=1 python tests/test_cmhd_orszag_tang.py  # script mode
#
# (`--runslow` alone is not enough under pytest: pyproject's addopts deselects `slow` with
# -m, which is why the Makefile's test-slow target passes `-m slow` too.)
#
# examples/cmhd_orszag_tang.py runs the same problem with plots. The IC is re-declared here
# rather than imported: tests do not depend on examples/.
#
# ============================================================================ the problem
#
# Stone, Gardiner, Teuben, Hawley & Simon 2008, "Athena: A New Code for Astrophysical MHD",
# ApJS 178, 137, section VIII.4 "Two-Dimensional MHD" (published numbering 8.4), figures
# 22-24, run to t_f = 1/2. Verbatim from that section: constant initial density and
# pressure rho0 = 25/(36 pi) and P0 = 5/(12 pi), periodic (Lx,Ly) = (1,1), initial velocity
# (vx,vy) = (-sin 2pi y, sin 2pi x), and B from the vector potential
# Az = (B0/4pi) cos 4pi x + (B0/2pi) cos 2pi y with B0 = 1/sqrt(4 pi) -- i.e.
# B = curl(Az zhat) = (-B0 sin 2pi y, B0 sin 4pi x, 0). gamma = 5/3 (stated on the Athena
# test page and in the shipped athinput.orszag-tang, not in that paragraph). The IC
# originates with Ryu, Jones & Frank 1995, ApJ 452, 785, figure 8.
#
# In taranis code units (Alfven units, <rho> = 1; docs/numerics.md "Compressible MHD") that
# IC is exactly
#
#     rho = 1, u unchanged, B_code = 0.6 * (-sin 2pi y, sin 4pi x, 0), cs0 = 1, gamma = 5/3
#
# since c_s = sqrt(gamma P0/rho0) = 1 and v_A = B0/sqrt(rho0) = 0.6 exactly, giving
# beta = 2 P0/B0^2 = 10/3 and the EXACT initial energies E_kin = 1/2, E_mag = 0.18,
# E_int = 0.9 (asserted below -- the cheapest possible check that the unit mapping is
# right). The time unit is 1:1 with Athena's; to compare against raw Athena output multiply
# its energies by 1/rho0 = 36 pi/25 = 4.5238934.
#
# ============================================== polytropic vs adiabatic: exact, then not
#
# taranis is a dealiased pseudospectral code with a POLYTROPIC closure (p = K rho^gamma, no
# energy equation) and no shock capturing; Athena solves the adiabatic ideal-gas equations.
# The two are not an approximation of each other in general -- but for THIS IC they
# coincide EXACTLY while the flow is smooth: rho0 and P0 are both uniform, so the entropy
# is uniform, and smooth isentropic ideal-gas flow satisfies p = K rho^gamma with a single
# global K, which is precisely the polytropic closure. Nothing is being approximated in the
# smooth phase.
#
# Once shocks form they diverge for good: shock heating raises entropy non-uniformly,
# Athena's p leaves the K rho^gamma curve, and a barotropic solver cannot represent that.
# Concretely, an adiabatic code turns the shock's kinetic energy into heat and conserves
# total energy, while this model has no heat reservoir and loses it (measured: -5.9% by
# t = 0.51). So the quantitative window is exactly the pre-shock smooth phase, and after it
# the assertions here are qualitative by construction.
#
# ======================================================== the window, and how it was set
#
# Literature, converted with Stone's own mapping between the Athena [0,1] box and the 2 pi
# box used by Toth 2000 (Stone 2008 section VIII.4: figure 22 at t_f = 1/2 "can be compared
# directly to the results in, e.g., T2000 at a time of t_f = pi"), i.e. t = t_2pi/(2 pi):
#
#   Snow, Hillier et al. 2021, Exp. Results 2, e35, section 3 (already in Athena units):
#       "At very early times (t < 0.15), the detections of shocks are sporadic... After
#       t = 0.15, large-scale fast-mode shocks are generated due to the initial conditions
#       of the OT vortex."                                            -> onset t ~ 0.15
#   Toth 2000, JCP 161, 605, section 6.4 (2 pi box): "At time t = 1 the flow is still
#       quite smooth, although some discontinuities are already present."   -> t ~ 0.159
#   Picone & Dahlburg 1991 (less compressible, M_s = 0.71): shocks "around t = 1.5" in the
#       2 pi box                                                            -> t ~ 0.24
#
# Measured here (2026-08-30, fp64, diss = 0, fixed dt = 5e-4):
#
#   t      |dE/E| at 256^2   fraction of spectral energy above k_max/2 (256^2)
#   0.00   0                 3e-32
#   0.05   ~1e-9             3e-32
#   0.10   ~2e-9             5e-16
#   0.12   5.6e-9            ~1e-11
#   0.13   2.9e-7            3e-10
#   0.18   ~1e-5             1e-6
#   0.51   -5.9e-2           5e-6   (adaptive dt with hyperdissipation; see the shocked run)
#
# THE GATE WINDOW IS t <= 0.12, comfortably inside every literature estimate of the shock
# time and inside where this code's own smoothness monitors are still clean.
#
# In it, three quantitative assertions, all on FIXED-dt inviscid runs so that 128^2 and
# 256^2 record at bitwise-identical times (an earlier adaptive-dt version of this file
# compared INTERPOLATED traces and read 3.5e-3 of "disagreement" that was entirely the
# linear interpolation of a curved E_kin(t) over a 0.026 sample interval -- do not
# reintroduce it):
#
#   (a) E is conserved to 1e-7 at 256^2 (measured 5.6e-9; 2.5e-7 at 128^2 -- the
#       plans/CMHD_PLAN.md §3.5 non-polynomial aliasing residual, falling with resolution
#       exactly as the C1 energy gate found),
#   (b) E_kin(t) and E_mag(t) at 128^2 and 256^2 agree to 1e-5 (measured 3.7e-7 and
#       5.9e-7), i.e. the traces are a property of the equations and not of this grid,
#   (c) the 256^2 trace matches a stored table -- see
#       test_energy_traces_against_a_reference for what that table is and, importantly,
#       what it is not.
#
# Single process by construction (CMHD is z_spectral, which is size==1 only).
from _rmhd_testing import bootstrap, checks, fresh_params

bootstrap()

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision
from taranis.diagnostics import cmhd as dcmhd
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

VA = 0.6                 # v_A = B0/sqrt(rho0) for the Athena OT IC, in code units
CS0 = 1.0
GAMMA = 5.0/3.0
DISS_256 = 1.25e-8       # k^4 hyperdissipation for the shocked run at 256^2
T_SMOOTH = 0.12          # the pre-shock window (header)
T_END = 0.50             # Athena's figure-22 comparison time
DT_SMOOTH = 5e-4         # fixed dt for the smooth-window runs (CFL-safe at 256^2)
NREC_SMOOTH = 24         # records of 10 steps: 24*10*5e-4 = T_SMOOTH


def _fp64():
    return _precision.precision == "64"


def _ot_ic(x, y, z):
    shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
    one = jnp.ones(shp)
    tp = 2*jnp.pi
    return jnp.stack([one,
                      -jnp.sin(tp*y)*one, jnp.sin(tp*x)*one, 0.0*one,
                      -VA*jnp.sin(tp*y)*one, VA*jnp.sin(2*tp*x)*one, 0.0*one])


def _params(n, diss, nz=4, dt=None):
    fixed = dt is not None
    return fresh_params(nx=n, ny=n, nz=nz, Lx=1.0, Ly=1.0, Lz=1.0, dims=3, z_spectral=True,
                        eqtype="CMHD", adaptive_timestep=not fixed, cfl_safety=0.4,
                        dt=float(dt if fixed else 1e-3),
                        eqpars=dict(cs0=CS0, diss=diss, hyper=2, gamma=GAMMA))


_COLS = ("t", "E_kin", "E_mag", "E_int", "E_tot", "M_s", "M_A", "rho_min", "divB", "kz_max")


def _run(n, diss, *, dt=None, t_end=None, nrec=None, nblock=10, nz=4, keep=()):
    """One OT run. Either fixed dt (dt=, nrec= records of nblock steps) or adaptive
    (t_end=). Returns (params, kgrid, trace, kept-states-by-requested-time)."""
    params = _params(n, diss, nz=nz, dt=dt)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_ot_ic, params)
    stepper, sch = get_scheme("lsrk54")
    advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))
    trace = {c: [] for c in _COLS}
    kept = {}

    def record(s):
        ek, em, ei = dcmhd.energies(s, kgrid, params)
        ms, ma = dcmhd.mach_numbers(s, kgrid, params)
        # every kz != 0 coefficient of every field: the 2.5D embedding monitor
        kzmax = float(np.abs(np.asarray(s.fields)[:, 1:]).max())
        for c, v in zip(_COLS, (float(s.t), float(ek), float(em), float(ei),
                                float(ek + em + ei), float(ms), float(ma),
                                dcmhd.rho_min(s, kgrid, params),
                                dcmhd.divB_max(s, kgrid, params), kzmax)):
            trace[c].append(v)
        for tk in keep:
            if tk not in kept and float(s.t) >= tk:
                kept[tk] = s

    record(state)
    nleft = nrec if nrec is not None else -1
    while nleft != 0 and (t_end is None or float(state.t) < t_end):
        state = advance(state, kgrid, params, nblock, sch, stepper)
        record(state)
        nleft -= 1
        assert np.isfinite(trace["E_tot"][-1]), (
            f"the {n}^2 OT run went non-finite at t={trace['t'][-1]}; rho_min trace "
            f"{trace['rho_min']}")
    return params, kgrid, {c: np.asarray(v) for c, v in trace.items()}, kept


# The three runs serve every test in this module and are not repeated per test: 256^2
# inviscid fixed-dt through the smooth window (31 s), the same at 128^2 (8 s), and the
# adaptive dissipative 256^2 run through the shocks to t = 0.5 (110 s).
_CACHE = {}


def _runs():
    if not _CACHE:
        _CACHE["hi"] = _run(256, 0.0, dt=DT_SMOOTH, nrec=NREC_SMOOTH, keep=(0.0,))
        _CACHE["lo"] = _run(128, 0.0, dt=DT_SMOOTH, nrec=NREC_SMOOTH)
        _CACHE["shock"] = _run(256, DISS_256, t_end=T_END, nblock=20, keep=(T_END,))
    return _CACHE["hi"], _CACHE["lo"], _CACHE["shock"]


# ------------------------------------------------------------------- the unit mapping

@pytest.mark.slow
@pytest.mark.fp64
def test_the_initial_energies_are_the_exact_athena_values():
    """The cheapest check that the Athena -> code-unit mapping in the header is right:
    E_kin = <rho|u|^2>/2 = 1/2, E_mag = <|B|^2>/2 = v_A^2/2 = 0.18 and
    E_int = <cs0^2 rho^gamma/(gamma(gamma-1))> = 0.9 are exact rationals for this IC, with
    no time integration involved. M_s = u_rms/c_s = 1 and M_A = u_rms/v_A = 5/3 likewise.
    A wrong beta, a dropped sqrt(4 pi) or a mis-scaled density each move one of them."""
    if not _fp64():
        print("[SKIP] test_the_initial_energies_are_the_exact_athena_values -- fp64 only")
        return
    (params, kgrid, trace, kept), _, _ = _runs()
    s0 = kept[0.0]
    ek, em, ei = (float(v) for v in dcmhd.energies(s0, kgrid, params))
    ms, ma = (float(v) for v in dcmhd.mach_numbers(s0, kgrid, params))
    with checks() as c:
        for name, got, ref in (("E_kin", ek, 0.5), ("E_mag", em, 0.5*VA**2),
                               ("E_int", ei, CS0**2/(GAMMA*(GAMMA - 1.0))),
                               ("M_s", ms, 1.0), ("M_A", ma, 1.0/VA)):
            c.check(f"{name} = {ref!r} exactly at t = 0", abs(got - ref) < 1e-13*abs(ref),
                    f"{got!r} vs {ref!r}")
        c.check("rho_min = 1 at t = 0", abs(trace["rho_min"][0] - 1.0) < 1e-13,
                f"{trace['rho_min'][0]!r}")


# --------------------------------------------------------- 2.5D embedding + constraints

@pytest.mark.slow
@pytest.mark.fp64
def test_z_independence_is_preserved_exactly():
    """The 2.5D-embedding gate. A z-independent IC has every kz != 0 coefficient exactly
    zero, and nothing in the CMHD RHS can create one: every product of z-independent fields
    is z-independent, and i*kz multiplies only coefficients that are already zero. At
    nz = 4 the z-FFT twiddles are exactly +-1, +-i, so this holds BITWISE rather than merely
    to round-off -- measured identically 0.0 for the whole of all three runs, and at nz = 8
    too. Gated at a round-off bound, because exact zero is a property of the FFT
    implementation while the physics claim is the round-off one; the bitwise fact is
    asserted separately, and it going red alone means the FFT changed, not the equations."""
    if not _fp64():
        print("[SKIP] test_z_independence_is_preserved_exactly -- fp64 only")
        return
    runs = _runs()
    eps = np.finfo(np.float64).eps
    with checks() as c:
        for label, (params, _, trace, _) in zip(("smooth 256", "smooth 128", "shocked 256"),
                                                runs):
            scale = 0.5*params.nx*params.ny*params.nz    # an O(1) field's coefficient scale
            worst = float(trace["kz_max"].max())
            c.check(f"{label}: max |field(kz != 0)| stays at round-off ({worst:.3e}, "
                    f"coefficient scale {scale:.3e})", worst < 1e3*eps*scale, f"{worst!r}")
            c.check(f"{label}: ...and is in fact exactly 0.0", worst == 0.0,
                    "the physics claim above is the round-off one; this line going red "
                    "alone means the nz=4 FFT stopped producing exact zeros -- report it")


@pytest.mark.slow
@pytest.mark.fp64
def test_div_b_and_positivity_survive_the_shocks():
    """The two things a compressible spectral run can lose without saying so. div B is a
    round-off random walk for the whole shocked run (curl-form induction, docs/numerics.md
    and the C1 gate), and rho never reaches zero -- if it did, rho**(gamma-1) would be NaN
    and set_timestep's c_s would be poisoned, which is exactly what
    diagnostics.cmhd.rho_min is for. Measured: div B peaks at 1.3e-15, rho_min bottoms at
    0.176."""
    if not _fp64():
        print("[SKIP] test_div_b_and_positivity_survive_the_shocks -- fp64 only")
        return
    _, _, (_, _, trace, _) = _runs()
    with checks() as c:
        c.check(f"div B stays at round-off through the shocks "
                f"(max {trace['divB'].max():.2e})", trace["divB"].max() < 1e-12,
                f"{trace['divB'].max()!r}")
        c.check(f"rho stays positive with margin (min {trace['rho_min'].min():.4f})",
                trace["rho_min"].min() > 0.05, f"{trace['rho_min'].min()!r}")
        c.check("the run stayed finite throughout",
                bool(np.all(np.isfinite(trace["E_tot"]))), "")


# ------------------------------------------------------- the smooth window: quantitative

@pytest.mark.slow
@pytest.mark.fp64
def test_energy_is_conserved_in_the_pre_shock_window():
    """E = <rho|u|^2/2 + |B|^2/2 + rho e(rho)> is an exact invariant of the ideal polytropic
    equations (docs/numerics.md). With diss = 0 in the smooth window the only budget left is
    the O(dt^p)-plus-non-polynomial-aliasing residual, so this is a real ideal-invariant
    gate at a large (M_s = 1) amplitude -- and NOT a round-off assertion, per
    plans/CMHD_PLAN.md §3.5. Measured 5.6e-9 at 256^2 against 2.5e-7 at 128^2: it falls with
    resolution, which is the signature of the aliasing residual rather than of the
    timestepper (whose dt is the same at both)."""
    if not _fp64():
        print("[SKIP] test_energy_is_conserved_in_the_pre_shock_window -- fp64 only")
        return
    (_, _, hi, _), (_, _, lo, _), _ = _runs()
    d_hi = float(np.abs(hi["E_tot"]/hi["E_tot"][0] - 1.0).max())
    d_lo = float(np.abs(lo["E_tot"]/lo["E_tot"][0] - 1.0).max())
    with checks() as c:
        c.check(f"|dE/E| <= 1e-7 at 256^2 through t <= {T_SMOOTH} (max {d_hi:.2e} over "
                f"{hi['t'].size} samples)", d_hi < 1e-7, f"{d_hi!r}")
        c.check(f"and it FALLS with resolution at the same dt: 128^2 gives {d_lo:.2e}, "
                f"256^2 {d_hi:.2e} (the aliasing residual, not the stepper)",
                d_lo > 3*d_hi, f"128^2 {d_lo!r}, 256^2 {d_hi!r}")


@pytest.mark.slow
@pytest.mark.fp64
def test_the_smooth_window_traces_are_resolution_converged():
    """E_kin(t) and E_mag(t) at 128^2 and 256^2, sampled at bitwise-identical times (fixed
    dt at both resolutions -- see the header on why interpolation is not used), agree to
    1e-5 relative through the smooth window. Measured 3.7e-7 (E_kin) and 5.9e-7 (E_mag).
    This is what makes the traces a property of the equations rather than of this grid, and
    it is the precondition for comparing them to anyone else's code."""
    if not _fp64():
        print("[SKIP] test_the_smooth_window_traces_are_resolution_converged -- fp64 only")
        return
    (_, _, hi, _), (_, _, lo, _), _ = _runs()
    with checks() as c:
        c.check("the two resolutions sampled at bitwise-identical times",
                bool(np.array_equal(hi["t"], lo["t"])), f"{hi['t'][:3]} vs {lo['t'][:3]}")
        for key in ("E_kin", "E_mag"):
            d = np.abs(hi[key]/lo[key] - 1.0)
            worst = int(np.argmax(d))
            c.check(f"{key}(t) at 128^2 and 256^2 agree to 1e-5 through t <= {T_SMOOTH} "
                    f"(max rel {d.max():.2e})", d.max() < 1e-5,
                    f"worst at t={hi['t'][worst]:.4f}: 256^2 {hi[key][worst]!r}, "
                    f"128^2 {lo[key][worst]!r}")


# ---------------------------------------------------------------- the reference table
#
# SELF-GENERATED, NOT PUBLISHED DATA. Recorded 2026-08-30 from the 256^2 fixed-dt inviscid
# run in this file (Apple M1, macOS 14.6 / Darwin 23.6.0, jax 0.10.0, CPU, fp64), every 20
# steps of dt = 5e-4. See test_energy_traces_against_a_reference for exactly why, and for
# what would replace it.
_REFERENCE = (   # (t, E_kin, E_mag)
    (0.0000, 4.999999999999986e-01, 1.800000000000001e-01),
    (0.0100, 5.001173706202353e-01, 1.798775482574133e-01),
    (0.0200, 5.003871058935423e-01, 1.795326651846362e-01),
    (0.0300, 5.005721038364845e-01, 1.790301419471755e-01),
    (0.0400, 5.003057601924195e-01, 1.784709056625498e-01),
    (0.0500, 4.991220001941303e-01, 1.779864119305644e-01),
    (0.0600, 4.964845924997381e-01, 1.777385245732305e-01),
    (0.0700, 4.918309015653177e-01, 1.779299034807137e-01),
    (0.0800, 4.846829507203799e-01, 1.788279034639075e-01),
    (0.0900, 4.748592239145760e-01, 1.807853515009844e-01),
    (0.1000, 4.626153818761274e-01, 1.841916165930883e-01),
    (0.1100, 4.484783321703140e-01, 1.893135240195070e-01),
    (0.1200, 4.329323112814121e-01, 1.961498144118838e-01),
)


@pytest.mark.slow
@pytest.mark.fp64
def test_energy_traces_against_a_reference():
    """REGRESSION reference, self-generated -- read this before trusting it.

    TODO / KNOWN GAP: there is NO published, digitizable E_kin(t) or E_mag(t) for the
    Athena-normalized Orszag-Tang vortex to gate against. Searched 2026-08-30:

      - Stone et al. 2008 (ApJS 178, 137) section VIII.4 publishes exactly three OT
        figures -- 22 (contours at t_f = 1/2), 23 (pressure slices at t_f = 1/2), 24 (the
        isothermal variant). Not one of the paper's 36 figures is a time trace of anything.
      - Stone et al. 2020 (Athena++, ApJS 249, 4) does not contain the OT test at all
        (only the shipped src/pgen/orszag_tang.cpp does), and the Athena++ regression
        suite has no OT case and no stored reference data.
      - The Athena test-suite web pages are HTTP 404 at their live URL; the archived
        copies carry GIF images and movies only -- no .tab/.dat/.txt/.csv anywhere.
      - The only published OT energy-vs-time curves are Orszag & Tang 1979 (JFM 90, 129)
        figure 5 -- INCOMPRESSIBLE, 2 pi box, and on an unstated factor-of-2
        normalization -- and Dahlburg & Picone 1989 figure 5 / Picone & Dahlburg 1991
        figure 8, which are compressible but at different Mach number and beta and are
        readable only off marginal-quality DTIC scans. None is this problem.

    WHAT WOULD REPLACE THIS TABLE, concretely: Athena++'s shipped
    inputs/mhd/athinput.orszag-tang already requests `file_type = hst` at dt = 0.01, and
    src/outputs/history.cpp writes volume-integrated 1-KE, 2-KE, 3-KE, 1-ME, 2-ME, 3-ME
    per row. Building Athena++ and running that input therefore produces E_kin(t) and
    E_mag(t) directly, in the right normalization and with no digitizing -- multiply its
    energies by 1/rho0 = 36 pi/25 = 4.5238934 to reach these units. That is the correct
    external reference and it is NOT in this repository. (Note that pgen uses the box
    [-0.5,0.5]^2 and Az = B0/(4 pi)(cos 4 pi x - 2 cos 2 pi y), which is the Stone-2008
    field translated by (1/2,1/2): same physics, same energies, opposite signs.)

    So this table is a REGRESSION gate, not a validation gate: it says the solver still
    produces the trace it produced when C2 landed, and nothing more. The validation weight
    in this module is carried by the tests above -- exact initial energies, ideal-invariant
    conservation, and 128^2/256^2 convergence -- plus the fact (header) that polytropic and
    adiabatic coincide EXACTLY for this uniform-entropy IC while the flow is smooth, which
    is what makes an external comparison meaningful in this window at all.

    The 1e-9 tolerance is deliberately far tighter than any physical scale here (the
    resolution difference is 4e-7): a real change in the equations, the diagnostics or the
    stepper moves these numbers by orders more, while cross-host FFT re-association is
    ~1e-12 over 240 steps. If this alone goes red with everything above green, suspect the
    host or the jax version and report it -- do not regenerate the table to make it pass."""
    if not _fp64():
        print("[SKIP] test_energy_traces_against_a_reference -- fp64 only")
        return
    (_, _, hi, _), _, _ = _runs()
    with checks() as c:
        for t_ref, ek_ref, em_ref in _REFERENCE:
            i = int(np.argmin(np.abs(hi["t"] - t_ref)))
            c.check(f"t = {t_ref}: the sample time is on the grid",
                    abs(hi["t"][i] - t_ref) < 1e-12, f"nearest {hi['t'][i]!r}")
            for name, got, ref in (("E_kin", hi["E_kin"][i], ek_ref),
                                   ("E_mag", hi["E_mag"][i], em_ref)):
                rel = abs(got - ref)/abs(ref)
                c.check(f"t = {t_ref}: {name} matches the stored trace (rel {rel:.1e})",
                        rel < 1e-9, f"got {got!r}, stored {ref!r}")


# ------------------------------------------------------------- post-shock, qualitative

@pytest.mark.slow
@pytest.mark.fp64
def test_the_post_shock_state_is_qualitatively_right():
    """Post-shock the comparison is qualitative by construction (header). What is asserted
    is the shape every published OT run shows and a broken code would not:

      - the vortex decays: M_s starts at exactly 1 and ends well below it, and E declines,
      - magnetic energy GROWS at the expense of kinetic (measured E_mag 0.180 -> 0.289,
        E_kin 0.500 -> 0.227 by t = 0.51),
      - a broadband tail develops: the fraction of kinetic+magnetic energy above k_max/2
        goes from 3e-33 to 5e-6 -- small in absolute terms because hyperdissipation is
        absorbing it, but 27 orders above where it started.

    M_s is NOT monotone and is not asserted to be (it dips to 0.786 near t ~ 0.28, rises to
    0.831 as the compressed regions rebound, then falls); only the net decline is."""
    if not _fp64():
        print("[SKIP] test_the_post_shock_state_is_qualitatively_right -- fp64 only")
        return
    _, _, (params, kgrid, trace, kept) = _runs()
    ms = trace["M_s"]
    E0, Eend = trace["E_tot"][0], trace["E_tot"][-1]
    with checks() as c:
        c.check(f"M_s falls from 1 over the run ({ms[0]:.3f} -> {ms[-1]:.3f})",
                abs(ms[0] - 1.0) < 1e-12 and ms[-1] < 0.85, f"{ms[-1]!r}")
        c.check(f"E declines once shocks form ({100*(Eend/E0 - 1):+.2f}% by "
                f"t = {trace['t'][-1]:.3f}) -- the polytropic model has no heat reservoir",
                -0.20 < Eend/E0 - 1 < -0.01, f"{Eend/E0 - 1!r}")
        c.check(f"E_mag grows and E_kin falls (E_kin {trace['E_kin'][0]:.3f} -> "
                f"{trace['E_kin'][-1]:.3f}, E_mag {trace['E_mag'][0]:.3f} -> "
                f"{trace['E_mag'][-1]:.3f})",
                trace["E_mag"][-1] > 1.3*trace["E_mag"][0]
                and trace["E_kin"][-1] < 0.6*trace["E_kin"][0], "")
        kb, sk, sm, _ = dcmhd.spectra(kept[T_END], kgrid, params, bin_factor=1.0)
        kb = np.asarray(kb)
        tot = np.asarray(sk) + np.asarray(sm)
        frac = float(tot[kb > 0.5*kb.max()].sum()/tot.sum())
        c.check(f"a broadband tail has developed by t = {T_END} ({frac:.2e} of the energy "
                f"above k_max/2)", 1e-9 < frac < 1e-2, f"{frac!r}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
