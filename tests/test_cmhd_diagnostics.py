# CMHD diagnostics (taranis/diagnostics/cmhd.py, plans/CMHD_PLAN.md Phase C2). Four gates:
#
#   1. NORMALIZATION. energies() and mach_numbers() against a direct numpy real-space
#      computation that shares no code with them, and spectra() summing to the energies to
#      round-off (Parseval). That sum rule is the repo's standing convention gate: it is
#      what ties a CMHD number to diagnostics.rmhd.energy, to perpspec and to a future CMHD
#      forcing power (CLAUDE.md, "keep new energy-like diagnostics on this convention").
#
#   2. ENERGY BUDGET. -dE/dt from a centered difference of a short dissipating run against
#      energy_budget()'s eps, the tests/test_gdi_linear.py closure pattern. NEVER at
#      round-off: CMHD's ideal RHS conserves E only up to the non-polynomial aliasing
#      residual of plans/CMHD_PLAN.md §3.5, so the gate lives in the order-plus-aliasing
#      class (measured 3.5e-8 relative at fp64, gated at 1e-5). Plus the discriminator the
#      docs ask for: dropping the D_rho work term breaks the closure by ~60%.
#
#   3. MONITORS. rho_min against numpy's own min, and div B max against an IC built with a
#      KNOWN divergence, so the metric is shown to see one.
#
#   4. BITWISE SAFETY. Every diagnostic leaves state.fields and every kgrid leaf bitwise
#      unchanged. They are read-only observers and are called between simulate() calls on
#      live states.
#
# The Orszag-Tang validation gate lives in tests/test_cmhd_orszag_tang.py (marked slow).
#
# Single process by construction (CMHD is z_spectral, which is size==1 only). pytest, or
# `python tests/test_cmhd_diagnostics.py` -- never under mpirun.
from _rmhd_testing import bootstrap, checks, fresh_params

bootstrap()

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision, grids
from taranis.diagnostics import cmhd as dcmhd
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

_B0 = 1.0
_CS0 = 1.0


def _fp64():
    return _precision.precision == "64"


def _params(n=16, dt=2e-4, gamma=1.0, diss=1e-3, hyper=1, cs0=_CS0, **kw):
    # dt as a plain python float: a numpy scalar would make DiagonalOperator.scaled(dt)
    # strong-typed float64 and poison the fp32 field math (the C1 note).
    box = dict(nx=n, ny=n, nz=n, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi, dims=3,
               z_spectral=True, eqtype="CMHD", adaptive_timestep=False, cfl_safety=0.5)
    return fresh_params(eqpars=dict(cs0=cs0, diss=diss, hyper=hyper, gamma=gamma),
                        dt=float(dt), **dict(box, **kw))


# ------------------------------------------------------------------------- the test IC
# Seven band-limited modes at a modest amplitude, each B mode perpendicular to its own k so
# the state starts divergence-free. The amplitude is deliberately small: rho = 1 + O(amp)
# must stay positive (rho**(gamma-1) and log(rho) are NaN below zero -- the intended loud
# failure, plans/CMHD_PLAN.md §7), and the budget gate wants the ideal aliasing residual
# well below the dissipation sink, which it is at amp^~7 (the C1 energy-gate measurement).

_MODES = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, -1, 1)]


def _random_ic(amp=0.05, seed=3, divergent_b=False):
    rng = np.random.default_rng(seed)
    coeff = [(rng.normal(size=3), rng.normal(size=3), rng.normal(),
              rng.uniform(0, 2*np.pi, size=3)) for _ in _MODES]

    def ic(x, y, z):
        shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
        u = [jnp.zeros(shp) for _ in range(3)]
        B = [jnp.zeros(shp) for _ in range(3)]
        r = jnp.zeros(shp)
        for kk, (au, cb, ar, ph) in zip(_MODES, coeff):
            k = np.array(kk, dtype=float)
            arg = k[0]*x + k[1]*y + k[2]*z
            # k x c is perpendicular to k -> div B = 0; the raw c is not
            ab = cb if divergent_b else np.cross(k/np.linalg.norm(k), cb)
            for i in range(3):
                u[i] = u[i] + amp*float(au[i])*jnp.cos(arg + ph[0])
                B[i] = B[i] + amp*float(ab[i])*jnp.cos(arg + ph[1])
            r = r + amp*float(ar)*jnp.cos(arg + ph[2])
        return jnp.stack([1.0 + r, u[0], u[1], u[2], B[0], B[1], B[2] + _B0])
    return ic


def _setup(**kw):
    params = _params(**kw)
    kgrid = jr.setup_kgrids(params)
    return params, kgrid


def _numpy_reference(state, params, gamma, cs0=_CS0):
    """(E_kin, E_mag, E_int, M_s, M_A, rho_min) from the real-space fields with numpy only:
    no perp_reduce, no diagnostics code, promoted to float64 for the reduction."""
    f = np.asarray(grids.ifft(state.fields, params), dtype=np.float64)
    rho, u, B = f[0], f[1:4], f[4:7]
    u2 = (u*u).sum(0)
    cs2 = cs0**2*np.ones_like(rho) if gamma == 1.0 else cs0**2*rho**(gamma - 1.0)
    ie = (cs0**2*rho*np.log(rho) if gamma == 1.0
          else cs0**2/(gamma*(gamma - 1.0))*rho**gamma)
    return (0.5*np.mean(rho*u2), 0.5*np.mean((B*B).sum(0)), np.mean(ie),
            np.sqrt(np.mean(u2)/np.mean(cs2)),
            np.sqrt(np.mean(rho*u2)/np.mean((B*B).sum(0))), rho.min())


# ------------------------------------------------------- gate 1: normalization + Parseval

def test_energies_and_mach_numbers_match_a_numpy_reference():
    """The normalization gate. Every CMHD scalar is a volume average <.> over the box, the
    one convention shared with shared_physics.perp_reduce, diagnostics.rmhd.energy and
    forcing power (CLAUDE.md). Checked here against a numpy real-space computation that
    imports none of it, at both gamma branches."""
    tol = 1e-12 if _fp64() else 3e-6
    with checks() as c:
        for gamma in (1.0, 5.0/3.0):
            params, kgrid = _setup(gamma=gamma)
            state = jr.initialize(_random_ic(), params)
            ek, em, ei = dcmhd.energies(state, kgrid, params)
            ms, ma = dcmhd.mach_numbers(state, kgrid, params)
            rk, rm, ri, rms, rma, rrho = _numpy_reference(state, params, gamma)
            g = f"gamma={gamma:.3f}"
            for name, got, ref in (("E_kin", ek, rk), ("E_mag", em, rm), ("E_int", ei, ri),
                                   ("M_s", ms, rms), ("M_A", ma, rma)):
                rel = abs(float(got) - ref)/max(abs(ref), 1e-30)
                c.check(f"{g}: {name} matches the numpy reference (rel {rel:.2e})",
                        rel < tol, f"got {float(got)!r}, numpy {ref!r}")
            c.check(f"{g}: rho_min matches numpy's min",
                    abs(dcmhd.rho_min(state, kgrid, params) - rrho) < 1e-12*abs(rrho) + 1e-30,
                    f"got {dcmhd.rho_min(state, kgrid, params)!r}, numpy {rrho!r}")


def test_spectra_integrate_to_the_energies():
    """The standing convention gate: sum(spectrum)*dk reproduces the energy the spectrum
    decomposes, to round-off, in BOTH binning modes.

        kinetic  -> <rho|u|^2>/2       (the sqrt(rho)*u variable; |u^|^2 alone would not)
        magnetic -> <|B|^2>/2
        density  -> <(rho - <rho>)^2>/2

    This is what forces the CMHD bins to run to the grid CORNER rather than to perpspec's
    min(nx,ny)//2 * kunit: sqrt(rho)*u is a non-polynomial real-space product and carries
    power past the dealias cutoff, so an early truncation would break the sum rule
    silently. Run at both gammas and at several bin_factors, since a sum rule that only
    holds at one binning is a coincidence."""
    tol = 1e-11 if _fp64() else 5e-5
    with checks() as c:
        for gamma in (1.0, 5.0/3.0):
            params, kgrid = _setup(gamma=gamma)
            state = jr.initialize(_random_ic(), params)
            ek, em, _ = dcmhd.energies(state, kgrid, params)
            rho = np.asarray(grids.ifft(state.fields[0], params), dtype=np.float64)
            edens = 0.5*np.mean((rho - rho.mean())**2)
            for iso in (False, True):
                for bf in (1.0, 2.0, 3.0):
                    kb, sk, sm, sd = dcmhd.spectra(state, kgrid, params, bin_factor=bf,
                                                   isotropic=iso)
                    dk = float(kb[1] - kb[0])
                    tag = f"gamma={gamma:.3f} isotropic={iso} bin_factor={bf}"
                    for name, spec, ref in (("kinetic", sk, float(ek)),
                                            ("magnetic", sm, float(em)),
                                            ("density", sd, edens)):
                        got = float(jnp.sum(spec))*dk
                        rel = abs(got - ref)/max(abs(ref), 1e-30)
                        c.check(f"{tag}: sum({name})*dk = the energy (rel {rel:.2e})",
                                rel < tol, f"got {got!r}, ref {ref!r}")


def test_the_kinetic_spectrum_is_not_just_the_velocity_spectrum():
    """Discriminator for the sqrt(rho)*u choice: at a compressible amplitude, <rho|u|^2>/2
    and <|u|^2>/2 differ well outside the sum-rule tolerance, so the gate above is testing
    something. If they ever agree to round-off the IC has stopped being compressible."""
    params, kgrid = _setup()
    state = jr.initialize(_random_ic(amp=0.3), params)
    f = np.asarray(grids.ifft(state.fields, params), dtype=np.float64)
    u2 = (f[1:4]**2).sum(0)
    weighted, plain = 0.5*np.mean(f[0]*u2), 0.5*np.mean(u2)
    rel = abs(weighted - plain)/plain
    with checks() as c:
        c.check(f"<rho|u|^2>/2 differs from <|u|^2>/2 at this amplitude (rel {rel:.2e})",
                rel > 1e-4, f"{weighted!r} vs {plain!r}")


# ------------------------------------------------------------------- gate 2: the budget

def _closure(gamma, amp=0.05, diss=1e-3, hyper=1, dt=2e-4, n=16):
    """(measured dE/dt, budget dict) -- centered difference over two steps against
    energy_budget at the midpoint state, the tests/test_gdi_linear.py pattern."""
    params, kgrid = _setup(n=n, dt=dt, gamma=gamma, diss=diss, hyper=hyper)
    stepper, scheme = get_scheme("lsrk54")
    s0 = jr.initialize(_random_ic(amp=amp), params)
    s1 = block_of_steps(s0, kgrid, params, 1, scheme, stepper)
    s2 = block_of_steps(s1, kgrid, params, 1, scheme, stepper)
    dt_actual = float(s2.t - s0.t)/2.0
    E0 = sum(float(v) for v in dcmhd.energies(s0, kgrid, params))
    E2 = sum(float(v) for v in dcmhd.energies(s2, kgrid, params))
    return (E2 - E0)/(2*dt_actual), dcmhd.energy_budget(s1, kgrid, params)


@pytest.mark.fp64
def test_energy_budget_closes_against_the_measured_dedt():
    """dE/dt + eps ~ 0. fp64 only: dE over two steps is ~4e-8 on an E of order 1, which is
    below the fp32 noise floor of the difference itself -- the same reason the GDI closure
    gates are fp64-gated.

    The tolerance is NOT round-off and must not be tightened to it: CMHD's ideal RHS
    conserves E only up to the O(dt^p)-plus-non-polynomial-aliasing residual
    (plans/CMHD_PLAN.md §3.5, docs/numerics.md). Measured 2026-08-30 at amp=0.05, 16^3:
    3.5e-8 (gamma=1) and 2.8e-8 (gamma=5/3) relative, and 1.5e-9 / 1.4e-9 at diss=1e-2
    where the sink is 10x larger -- i.e. the residual is a fixed absolute floor, exactly as
    the aliasing story predicts. Gated at 1e-5."""
    if not _fp64():
        print("[SKIP] test_energy_budget_closes_against_the_measured_dedt -- fp64 only")
        return
    with checks() as c:
        for gamma in (1.0, 5.0/3.0):
            for diss in (1e-3, 1e-2):
                meas, b = _closure(gamma, diss=diss)
                eps = float(b["total"])
                rel = abs(meas - float(b["dEdt"]))/max(abs(eps), 1e-30)
                tag = f"gamma={gamma:.3f} diss={diss:g}"
                c.check(f"{tag}: dE/dt = -eps (rel {rel:.2e})", rel < 1e-5,
                        f"measured {meas:.10e}, budget dEdt {float(b['dEdt']):.10e}")
                c.check(f"{tag}: eps > 0 for diss > 0", eps > 0.0, f"eps={eps:.10e}")


@pytest.mark.fp64
def test_the_budget_needs_the_density_work_term():
    """docs/numerics.md names the D_rho term -- mass diffusion doing pdV-like work through
    h -- as the likely first bug in a budget implementation. Here is the discriminator: it
    carries ~38% of the sink at this configuration, so a budget without it misses the
    measured dE/dt by ~60%, three orders outside the gate above."""
    if not _fp64():
        print("[SKIP] test_the_budget_needs_the_density_work_term -- fp64 only")
        return
    with checks() as c:
        for gamma in (1.0, 5.0/3.0):
            meas, b = _closure(gamma)
            without = -(float(b["total"]) - float(b["density"]))
            rel = abs(meas - without)/max(abs(without), 1e-30)
            frac = float(b["density"])/float(b["total"])
            c.check(f"gamma={gamma:.3f}: dropping the D_rho term breaks the closure "
                    f"(rel {rel:.2e}, the term is {frac:.1%} of eps)", rel > 0.1,
                    f"measured {meas:.10e}, budget-without-density {without:.10e}")


@pytest.mark.fp64
def test_the_budget_pieces_vanish_with_the_dissipation_they_name():
    """Each piece is the sink of ONE field's diagonal of L, so the (D_rho, nu, eta) eqpars
    schema switches them off one at a time. This is what pins the piece->field mapping;
    a transposed _Lrow index would leave the total unchanged and pass everything above."""
    if not _fp64():
        print("[SKIP] test_the_budget_pieces_vanish_with_the_dissipation_they_name -- fp64")
        return
    with checks() as c:
        for i, (name, diss) in enumerate((("density", (0.0, 1e-3, 1e-3)),
                                          ("kinetic", (1e-3, 0.0, 1e-3)),
                                          ("magnetic", (1e-3, 1e-3, 0.0)))):
            _, b = _closure(1.0, diss=diss)
            others = [k for k in ("density", "kinetic", "magnetic") if k != name]
            c.check(f"diss[{i}]=0 zeroes the {name} piece exactly",
                    float(b[name]) == 0.0, f"{name}={float(b[name])!r}")
            for o in others:
                c.check(f"diss[{i}]=0 leaves the {o} piece nonzero", float(b[o]) != 0.0,
                        f"{o}={float(b[o])!r}")


# ------------------------------------------------------------------- gate 3: the monitors

def test_div_b_max_sees_a_divergence_that_is_really_there():
    """divB_max is only useful if it can see one. The same IC built with B amplitudes NOT
    projected perpendicular to their own k carries a real divergence; the projected one
    starts at round-off. Also checks the monitor stays at round-off through a short run
    (the curl-form guarantee the C1 gate covers at length).

    On the SCALE of the dirty reading: the metric divides by max_k|B^|, which here is the
    k = 0 background B0 = 1, while the divergent part is the amp = 0.05 fluctuation -- so
    an unprojected IC reads O(amp), measured 2.71e-2 at fp64, not O(1). That is the
    field-scale normalization doing exactly what it is for (the C1 test's rationale); the
    discriminating fact is the fifteen orders of magnitude between 2.7e-2 and 1.1e-17."""
    eps = np.finfo(np.float64 if _fp64() else np.float32).eps
    params, kgrid = _setup(diss=0.0)
    clean = jr.initialize(_random_ic(divergent_b=False), params)
    dirty = jr.initialize(_random_ic(divergent_b=True), params)
    stepper, scheme = get_scheme("lsrk54")
    evolved = block_of_steps(clean, kgrid, params, 20, scheme, stepper)
    d_clean = dcmhd.divB_max(clean, kgrid, params)
    d_dirty = dcmhd.divB_max(dirty, kgrid, params)
    d_evolved = dcmhd.divB_max(evolved, kgrid, params)
    with checks() as c:
        c.check(f"a k-perpendicular B IC reads round-off ({d_clean:.2e})",
                d_clean < 100*eps, f"{d_clean!r}")
        c.check(f"an unprojected B IC reads a real divergence, O(amp) against the field "
                f"scale ({d_dirty:.2e})", d_dirty > 1e-3, f"{d_dirty!r}")
        # measured 2.5e15 at fp64 (clean 1.1e-17) and 1.1e7 at fp32 (clean 2.4e-9)
        gap = 1e10 if _fp64() else 1e6
        c.check(f"the dirty IC is >= {gap:.0e} x the clean one "
                f"({d_dirty/max(d_clean, 1e-300):.1e})", d_dirty > gap*d_clean,
                f"clean {d_clean!r}, dirty {d_dirty!r}")
        c.check(f"20 curl-form steps leave it at round-off ({d_evolved:.2e})",
                d_evolved < 1e4*eps, f"{d_evolved!r}")


def test_rho_min_tracks_a_rarefaction():
    """The rarefaction monitor moves the way the density does: a larger IC amplitude drives
    rho_min further below 1, and it equals the numpy minimum of the real-space field."""
    params, kgrid = _setup()
    mins = []
    for amp in (0.05, 0.15, 0.30):
        state = jr.initialize(_random_ic(amp=amp), params)
        mins.append(dcmhd.rho_min(state, kgrid, params))
    with checks() as c:
        c.check(f"rho_min falls monotonically with the IC amplitude {mins}",
                mins[0] > mins[1] > mins[2], f"{mins}")
        c.check("rho_min stays below the mean density 1.0", max(mins) < 1.0, f"{mins}")


# ------------------------------------------------------------ gate 4: read-only observers

def test_diagnostics_do_not_mutate_state_or_kgrid():
    """Diagnostics are read-only observers called between simulate() calls on live states.
    Every entry point is run and both the state fields and every kgrid leaf are compared
    BITWISE afterwards. (kgrid.lin is a NamedTuple of arrays, so the tree walk covers the
    propagator's arrays too.)"""
    params, kgrid = _setup()
    state = jr.initialize(_random_ic(), params)
    f_before = np.asarray(state.fields).copy()
    t_before = float(state.t)
    k_before = [np.asarray(v).copy() for v in jax.tree.leaves(kgrid)]

    dcmhd.energies(state, kgrid, params)
    dcmhd.mach_numbers(state, kgrid, params)
    dcmhd.rho_min(state, kgrid, params)
    dcmhd.divB_max(state, kgrid, params)
    dcmhd.spectra(state, kgrid, params)
    dcmhd.spectra(state, kgrid, params, isotropic=True)
    dcmhd.energy_budget(state, kgrid, params)

    k_after = [np.asarray(v) for v in jax.tree.leaves(kgrid)]
    with checks() as c:
        c.check("state.fields is bitwise unchanged",
                bool(np.array_equal(f_before, np.asarray(state.fields))), "")
        c.check("state.t is unchanged", float(state.t) == t_before, "")
        c.check(f"every kgrid leaf is bitwise unchanged ({len(k_after)} leaves)",
                all(bool(np.array_equal(a, b)) for a, b in zip(k_before, k_after)), "")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
