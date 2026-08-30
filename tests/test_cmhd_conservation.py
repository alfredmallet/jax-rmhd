# Compressible MHD (physics/cmhd.py, plans/CMHD_PLAN.md Phase C1), gates 3 and 5:
#
#   3. IDEAL INVARIANTS on a band-limited random smooth state (M_s ~ 0.3, dB/B0 ~ 0.3,
#      nu = 0):
#        - mass rho^(k=0) and each mean-B component are BITWISE invariants of the discrete
#          step (docs/numerics.md: the k = 0 RHS is exact fp zero, exp(0) = 1, mask = 1).
#          The mean of u is NOT -- that is the discriminator that the bitwise check is not
#          vacuous.
#        - max_k |k.B^|/(|k| max|B^|) is a machine-epsilon random walk with no systematic
#          source (curl-form induction), asserted in the ~eps*sqrt(N) class AND against
#          linear growth.
#        - energy and cross helicity: NEVER at round-off. See
#          test_energy_and_cross_helicity_drift for what is asserted and why it is not
#          exactly what CMHD_PLAN §5 gate 3 wrote.
#
#   5. PLUMBING: eqpars round-trip through params.json, a bitwise snapshot restart, and every
#      configuration error the module rejects.
#
# Single process by construction (CMHD is z_spectral, which is size==1 only). pytest, or
# `python tests/test_cmhd_conservation.py` -- never under mpirun.
from _rmhd_testing import (bootstrap, checks, fresh_params, managed_manager, snap_dir)

bootstrap()

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision, grids, snapshot_io
from taranis.physics import cmhd
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

_B0 = 1.0
_CS0 = 1.0


def _fp64():
    return _precision.precision == "64"


def _box(n):
    return dict(nx=n, ny=n, nz=n, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi, dims=3,
                z_spectral=True, eqtype="CMHD", adaptive_timestep=False, cfl_safety=0.5)


def _params(n, dt, gamma=1.0, diss=0.0, hyper=1, cs0=_CS0, **kw):
    # dt as a plain python float: a numpy scalar would make DiagonalOperator.scaled(dt)
    # strong-typed float64 and poison the fp32 field math
    return fresh_params(eqpars=dict(cs0=cs0, diss=diss, hyper=hyper, gamma=gamma),
                        dt=float(dt), **dict(_box(n), **kw))


# --------------------------------------------------------------- band-limited random IC
# Six random modes with 1 <= |n|^2 <= 4, fixed seed. u and rho are unconstrained; each B mode
# amplitude is built as khat x c, i.e. perpendicular to its own k, so the IC is
# divergence-free to round-off and the div B gate starts from zero rather than from the IC.

def _random_modes(seed=7, nmode=6):
    rng = np.random.default_rng(seed)
    cand = [(a, b, c) for a in (-2, -1, 0, 1, 2) for b in (-2, -1, 0, 1, 2)
            for c in (-2, -1, 0, 1, 2)
            if 1 <= a*a + b*b + c*c <= 4 and (a > 0 or (a == 0 and (b > 0 or
                                                                   (b == 0 and c > 0))))]
    out = []
    for i in rng.choice(len(cand), size=nmode, replace=False):
        k = np.array(cand[i], dtype=float)
        out.append((k, rng.normal(size=3), rng.normal(size=3), rng.normal(),
                    rng.uniform(0, 2*np.pi, size=3)))
    return out


def _random_ic(ms, db, drho, seed=7):
    modes = _random_modes(seed)

    def ic(x, y, z):
        shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
        u = [jnp.zeros(shp) for _ in range(3)]
        B = [jnp.zeros(shp) for _ in range(3)]
        r = jnp.zeros(shp)
        for k, au, cb, ar, ph in modes:
            arg = k[0]*x + k[1]*y + k[2]*z
            ab = np.cross(k/np.linalg.norm(k), cb)      # perpendicular to k: div B = 0
            cu, cbp, cr = jnp.cos(arg + ph[0]), jnp.cos(arg + ph[1]), jnp.cos(arg + ph[2])
            for i in range(3):
                u[i] = u[i] + float(au[i])*cu
                B[i] = B[i] + float(ab[i])*cbp
            r = r + float(ar)*cr
        su = ms*_CS0/jnp.sqrt(jnp.mean(u[0]**2 + u[1]**2 + u[2]**2))
        sb = db*_B0/jnp.sqrt(jnp.mean(B[0]**2 + B[1]**2 + B[2]**2))
        sr = drho/jnp.sqrt(jnp.mean(r**2))
        return jnp.stack([1.0 + sr*r, su*u[0], su*u[1], su*u[2],
                          sb*B[0], sb*B[1], sb*B[2] + _B0])
    return ic


def _advance(state, kgrid, params, nsteps, schemestr="lsrk54"):
    stepper, scheme = get_scheme(schemestr)
    return jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))(
        state, kgrid, params, nsteps, scheme, stepper)


def _invariants(fields, params, gamma):
    # E = sum[ rho|u|^2/2 + |B|^2/2 + rho e(rho) ] and the cross helicity sum u.B, on the
    # real-space fields, promoted to float64 for the reduction. rho e(rho) is
    # cs0^2 rho^gamma/(gamma(gamma-1)) for gamma > 1 and cs0^2 rho ln rho at gamma = 1
    # (docs/numerics.md).
    f = np.asarray(grids.ifft(fields, params), dtype=np.float64)
    rho, u, B = f[0], f[1:4], f[4:7]
    ke = 0.5*np.sum(rho*(u**2).sum(0))
    me_fluc = 0.5*np.sum((B**2).sum(0) - _B0**2)
    ie = (_CS0**2*np.sum(rho*np.log(rho)) if gamma == 1.0
          else _CS0**2/(gamma*(gamma - 1.0))*np.sum(rho**gamma))
    E = ke + 0.5*np.sum((B**2).sum(0)) + ie
    return float(E), float(np.sum((u*B).sum(0))), float(ke + me_fluc), float(rho.min())


def _div_b(fields, params, kgrid):
    # max_k |k.B^| / (|k| * max|B^|): the worst divergence-carrying component of B^ anywhere
    # on the grid, measured against the FIELD scale. The field scale is the right
    # normalization because the round-off deposit the docs describe (O(eps*|k|^2|E^|)) is set
    # by the field, not by the individual mode's own amplitude -- dividing each mode by its
    # OWN |B^| instead inflates a mode a factor a below the peak by 1/a and reports the
    # amplitude ratio rather than div B. No mode cut is needed in this form.
    f = np.asarray(fields)
    kx, ky = np.asarray(kgrid.kx), np.asarray(kgrid.ky)
    kz = np.asarray(kgrid.kz).copy()
    kz[params.nz//2] = 0.0                       # cmhd._kz_deriv's rule
    d = np.abs(kx*f[4] + ky*f[5] + kz*f[6])
    # broadcast_to, not "+ 0.0*d": a NaN field would poison kmag through 0.0*NaN, empty
    # the kmag > 0 mask and raise "zero-size array" instead of surfacing the NaN
    kmag = np.broadcast_to(np.sqrt(kx**2 + ky**2 + kz**2), d.shape)
    bmag = np.sqrt(np.abs(f[4])**2 + np.abs(f[5])**2 + np.abs(f[6])**2)
    m = kmag > 0
    return float(np.max(d[m]/kmag[m])/bmag.max())


# ------------------------------------------------------------ gate 3: bitwise invariants

def test_mass_and_mean_b_are_bitwise_invariant():
    """The k = 0 modes of rho and B are exact invariants of the discrete step, not merely
    conserved to round-off: the flux/curl forms have no k = 0 source, exp(L*tau) is exactly 1
    there and the dealias mask is 1. The mean of u is NOT an invariant (<(curl u) x u> does
    not vanish) -- checked here so the bitwise assertion cannot pass vacuously."""
    with checks() as c:
        for gamma in (1.0, 5.0/3.0):
            params = _params(16, 0.02, gamma=gamma)
            kgrid = jr.setup_kgrids(params)
            state = jr.initialize(_random_ic(0.3, 0.3, 0.1), params)
            m0 = np.asarray(state.fields[:, 0, 0, 0]).copy()
            end = _advance(state, kgrid, params, 50)
            m1 = np.asarray(end.fields[:, 0, 0, 0])
            g = f"gamma={gamma:.3f}"
            c.check(f"{g}: mass rho^(k=0) is bitwise invariant over 50 steps",
                    m0[0] == m1[0], f"{m0[0]!r} -> {m1[0]!r}")
            c.check(f"{g}: every mean-B component is bitwise invariant",
                    bool(np.all(m0[4:7] == m1[4:7])),
                    f"max |delta| {np.abs(m1[4:7] - m0[4:7]).max():.3e}")
            c.check(f"{g}: the mean of u is NOT invariant (the bitwise checks above are not "
                    f"vacuous)", bool(np.any(m0[1:4] != m1[1:4])),
                    f"max |delta| {np.abs(m1[1:4] - m0[1:4]).max():.3e}")


def test_div_b_stays_a_round_off_random_walk():
    """Curl-form induction: k.dB is a pairwise-cancelling sum, so each RHS evaluation deposits
    only round-off into it, and nothing amplifies or reads it. The gate is the ~eps*sqrt(N)
    class the docs name, plus the shape of the growth: a random walk multiplies by ~sqrt(10)
    over a 10x longer run, a systematic source by 10."""
    params = _params(16, 0.02)
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_random_ic(0.3, 0.3, 0.1), params)
    trace = [_div_b(state.fields, params, kgrid)]
    for _ in range(10):
        state = _advance(state, kgrid, params, 10)
        trace.append(_div_b(state.fields, params, kgrid))
    eps = np.finfo(np.float64 if _fp64() else np.float32).eps
    # 5 stages/step * 100 steps = 500 RHS evaluations; the walk bound is a stated 20x margin
    # on eps*sqrt(500). Measured: 8.7e-17 (fp64), 1.1e-8 (fp32).
    bound = 20.0*eps*np.sqrt(500)
    with checks() as c:
        c.check(f"the IC is divergence-free to round-off ({trace[0]:.2e})",
                trace[0] < 100*eps, f"{trace[0]:.3e}")
        c.check(f"div B stays in the eps*sqrt(N) class: max {max(trace):.2e} < {bound:.2e}",
                max(trace) < bound, f"trace {[f'{v:.2e}' for v in trace]}")
        growth = trace[-1]/trace[1]
        c.check(f"the growth from 10 to 100 steps is a random walk, not linear "
                f"(x{growth:.2f}; sqrt(10) = 3.2, linear would be 10)", growth < 5.0,
                f"trace {[f'{v:.2e}' for v in trace]}")


# ------------------------------------------ gate 3: energy and cross helicity (fp64 only)

@pytest.mark.fp64
def test_energy_and_cross_helicity_drift():
    """DEVIATION FROM CMHD_PLAN §5 gate 3, measured 2026-08-29, recorded here rather than
    tuned away. The plan asks for a dt-convergence order on BOTH invariants at
    M_s = dB/B0 = 0.3. Cross helicity does that (order 3.6-3.9 at every amplitude and
    resolution tried). Energy does not, at that amplitude: at 32^3 the drift is
    2.2859e-3 at dt = 0.02 and 2.2406e-3 at dt/2 -- a ratio of 1.02, because the drift there
    is NOT the time integrator, it is the spectral truncation residual of §3.5 (the
    non-polynomial 1/rho and ln rho, plus the cascade reaching the 2/3 cut of an nu = 0 run).
    That residual is dt-independent, falls ~15x when the resolution is doubled 16 -> 32, and
    falls ~300x when the amplitude is halved. So the energy gate has two halves:

      (a) at the plan's regime, absolute smallness AND dt-INDEPENDENCE -- which is itself the
          statement that the time integrator is not the source of the drift;
      (b) at half that amplitude, where the residual drops below the time error, the
          dt-convergence order the plan asked for (measured ratio 12.5 = order 3.6).

    fp64 only: at fp32 both drifts sit on a ~1e-3/1e-7 storage-noise floor (measured) and no
    order survives.
    """
    n, gamma = 32, 1.0

    def drift(amp, dt, nsteps):
        params = _params(n, dt, gamma=gamma)
        kgrid = jr.setup_kgrids(params)
        state = jr.initialize(_random_ic(0.3*amp, 0.3*amp, 0.1*amp), params)
        E0, H0, Efluc, rmin = _invariants(state.fields, params, gamma)
        end = _advance(state, kgrid, params, nsteps)
        E1, H1, _, _ = _invariants(end.fields, params, gamma)
        return abs(E1 - E0), abs(H1 - H0)/abs(H0), Efluc, rmin

    with checks() as c:
        # (a) the plan's regime: M_s = 0.3, dB/B0 = 0.3, t = 1
        dE_a, dH_a, Efluc, rmin = drift(1.0, 0.02, 50)
        dE_b, dH_b, _, _ = drift(1.0, 0.01, 100)
        c.check(f"rho stays well positive (min {rmin:.3f})", rmin > 0.5, f"{rmin}")
        c.check(f"energy drift is small: {dE_a:.3e} = {dE_a/Efluc:.2e} of the fluctuation "
                f"energy {Efluc:.3e}", dE_a/Efluc < 1e-5, f"{dE_a/Efluc:.3e}")
        c.check(f"... and is dt-INDEPENDENT ({dE_a:.6e} vs {dE_b:.6e} at dt/2), i.e. it is "
                f"the spectral truncation residual, not the scheme",
                abs(dE_a/dE_b - 1.0) < 0.05, f"ratio {dE_a/dE_b:.4f}")
        c.check(f"cross helicity converges at order {np.log2(dH_a/dH_b):.2f} "
                f"({dH_a:.3e} -> {dH_b:.3e})", 8.0 < dH_a/dH_b < 40.0,
                f"ratio {dH_a/dH_b:.2f}")
        c.check(f"... and is small in absolute terms ({dH_a:.2e})", dH_a < 1e-6,
                f"{dH_a:.3e}")

        # (b) half the amplitude: the truncation residual drops under the time error and the
        # energy shows the scheme order too
        dE_a, dH_a, Efluc, _ = drift(0.5, 0.02, 50)
        dE_b, dH_b, _, _ = drift(0.5, 0.01, 100)
        c.check(f"at half amplitude the energy drift converges at order "
                f"{np.log2(dE_a/dE_b):.2f} ({dE_a:.3e} -> {dE_b:.3e})",
                8.0 < dE_a/dE_b < 40.0, f"ratio {dE_a/dE_b:.2f}")
        c.check(f"... and is small in absolute terms ({dE_a/Efluc:.2e} of the fluctuation "
                f"energy)", dE_a/Efluc < 1e-8, f"{dE_a/Efluc:.3e}")
        c.check(f"cross helicity converges at order {np.log2(dH_a/dH_b):.2f} there too",
                8.0 < dH_a/dH_b < 40.0, f"ratio {dH_a/dH_b:.2f}")


# ------------------------------------------------------------------------ gate 5: plumbing

_PLUMB_EQPARS = dict(cs0=1.3, diss=(0.01, 0.02, 0.03), hyper=2, gamma=5.0/3.0)


def test_eqpars_round_trip_through_params_json():
    params = fresh_params(eqpars=dict(_PLUMB_EQPARS), dt=0.01, **_box(8))
    with snap_dir("cmhd_params_") as d, checks() as c:
        params.save(d)
        back = jr.Parameters.from_snapshot(d)
        c.check("eqtype round-trips", back.eqtype == "CMHD", back.eqtype)
        c.check("nfields is 7 from the registry", back.nfields == 7, back.nfields)
        c.check(f"eqpars round-trip: {back.eqpars!r}", back.eqpars == _PLUMB_EQPARS,
                f"{back.eqpars!r} != {_PLUMB_EQPARS!r}")
        c.check("the length-3 diss comes back as a tuple, not a list",
                isinstance(back.eqpars["diss"], tuple), repr(back.eqpars["diss"]))
        params.save(d)          # identical re-save is a no-op
        c.check("identical re-save is a no-op", True)
        with pytest.raises(ValueError, match="eqpars"):
            fresh_params(eqpars=dict(_PLUMB_EQPARS, cs0=2.0), dt=0.01, **_box(8)).save(d)
        c.check("saving different eqpars over the record is a hard error", True)


def test_restart_from_a_snapshot_is_bitwise():
    """Fixed dt, nu > 0: a run stopped, checkpointed and resumed reproduces the uninterrupted
    trajectory bitwise (fields and t)."""
    params = _params(8, 0.02, diss=0.01, hyper=1)
    kgrid = jr.setup_kgrids(params)
    ic = _random_ic(0.3, 0.3, 0.1)
    straight = _advance(jr.initialize(ic, params), kgrid, params, 20)
    ref_fields, ref_t = np.asarray(straight.fields).copy(), float(straight.t)

    mid = _advance(jr.initialize(ic, params), kgrid, params, 10)
    with snap_dir("cmhd_restart_") as d:
        with managed_manager(params, d, nsnap=2) as mngr:
            snapshot_io.save_snapshot(0, mid, mngr, params)
            mngr.wait_until_finished()
        reloaded = snapshot_io.load_snapshot(0, d, params)
    resumed = _advance(reloaded, kgrid, params, 10)
    with checks() as c:
        c.check("the reloaded state is bitwise the checkpointed one",
                np.array_equal(np.asarray(reloaded.fields), np.asarray(mid.fields))
                and float(reloaded.t) == float(mid.t))
        c.check("forcing_scale survives as a concrete (n_ou,) array of zeros",
                reloaded.forcing_scale.shape == (params.n_ou,)
                and bool(np.all(np.asarray(reloaded.forcing_scale) == 0)))
        c.check(f"the resumed run ends at the same t ({float(resumed.t)!r})",
                float(resumed.t) == ref_t, f"{float(resumed.t)!r} vs {ref_t!r}")
        c.check("the resumed run is bitwise the uninterrupted one",
                np.array_equal(np.asarray(resumed.fields), ref_fields),
                f"max |delta| "
                f"{np.abs(np.asarray(resumed.fields) - ref_fields).max():.3e}")


def _raises(c, label, match, **kw):
    eqpars = kw.pop("eqpars", dict(cs0=1.0, diss=0.0, hyper=1))
    n = kw.pop("n", 8)
    try:
        jr.setup_kgrids(fresh_params(eqpars=eqpars, dt=0.01, **dict(_box(n), **kw)))
        raised = ""
    except (ValueError, NotImplementedError) as exc:
        raised = str(exc)
    c.check(f"{label} raises, naming {match!r}", match in raised,
            f"raised: {raised!r}")


def test_configuration_errors_raise():
    """Everything cmhd rejects, at setup_kgrids time (config.py stays equation-agnostic, so
    these are _check_supported/_eqpars errors, not Parameters errors)."""
    with checks() as c:
        _raises(c, "z_spectral=False", "z_spectral", z_spectral=False)
        _raises(c, "forcing=True", "forcing", forcing=True, fshell=(1, 3))
        _raises(c, "a missing eqpars key", "missing", eqpars=dict(diss=0.0, hyper=1))
        _raises(c, "an unknown eqpars key", "unknown",
                eqpars=dict(cs0=1.0, diss=0.0, hyper=1, z_diss_k=1.0))
        _raises(c, "cs0 <= 0", "cs0", eqpars=dict(cs0=0.0, diss=0.0, hyper=1))
        _raises(c, "gamma < 1", "gamma",
                eqpars=dict(cs0=1.0, diss=0.0, hyper=1, gamma=0.9))
        _raises(c, "hyper < 1", "hyper", eqpars=dict(cs0=1.0, diss=0.0, hyper=0))
        _raises(c, "a non-integer hyper", "hyper",
                eqpars=dict(cs0=1.0, diss=0.0, hyper=1.5))
        _raises(c, "a length-2 diss", "diss",
                eqpars=dict(cs0=1.0, diss=(0.1, 0.2), hyper=1))
        # a negative coefficient flips L's sign: exp(L*tau) would amplify the grid scale
        _raises(c, "a negative scalar diss", "diss",
                eqpars=dict(cs0=1.0, diss=-0.01, hyper=1))
        _raises(c, "one negative entry in a length-3 diss", "diss",
                eqpars=dict(cs0=1.0, diss=(0.01, -0.01, 0.01), hyper=1))
        c.check("diss = 0 is still accepted (the ideal runs above need it)",
                jr.setup_kgrids(fresh_params(eqpars=dict(cs0=1.0, diss=0.0, hyper=1),
                                             dt=0.01, **_box(8))) is not None)
        # the eqtype guard: these recipe functions are CMHD's, not a generic hook
        rmhd_params = fresh_params(nz=8, eqpars={"diss": (0.0, 0.0), "hyper": 1},
                                   dims=3, z_spectral=True, nx=8, ny=8,
                                   Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi, cfl_safety=0.5)
        try:
            cmhd._check_supported(rmhd_params)
            raised = ""
        except ValueError as exc:
            raised = str(exc)
        c.check("calling the cmhd recipe functions with a non-CMHD Parameters raises",
                "eqtype" in raised, f"raised: {raised!r}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
