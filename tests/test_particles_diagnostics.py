# Test-particle diagnostics, plans/TESTPART_PLAN.md §5 (taranis/diagnostics/particles.py).
#
# These are read-only post-processing tools, so almost everything here is checked against
# an analytic answer rather than against the solver:
#
#   - read_moments parses the particle_moments.txt sidecar by HEADER NAME (column order
#     and a narrower column set are both tolerated) and refuses a file with duplicated
#     times -- a silently mis-mapped column would rename a heating rate;
#   - heating_rate / mu_diffusion recover a planted slope, honour the [t0,t1] window and
#     apply the 0.5 factor to the velocity-square columns only;
#   - delta_u / xi / chandran_fit round-trip a synthetic power-law spectrum and a
#     synthetic Q = c1 (du^3/rho) exp(-c2/xi);
#   - kinetic_spectrum's normalization: sum(Ekin)*dk = 0.5<u^2> for a single mode, which
#     is the convention delta_u = sqrt(2 k Ekin) rests on;
#   - mu_of and jz_at against analytic fields (psi = 0 and psi = a cos x), to bilinear
#     tolerance;
#   - conditional_stats / increment_histogram / increment_diffusion on synthetic arrays;
#   - work_split / energy_budget closure on a short LIVE co-stepped run.
#
# Cheap grids; the field-dependent tolerances are the same in both precision sessions
# (the bilinear interpolation floor dominates fp32 round-off), so nothing here is
# precision-marked. Script: `python tests/test_particles_diagnostics.py`.
from _rmhd_testing import bootstrap, checks, fresh_params

bootstrap()

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

import taranis as jr
from taranis import run
from taranis.diagnostics import particles as pdiag
from taranis.particles.fields import WORK_PIECES
from taranis.particles.state import MOMENTS, init_particles, push_ensembles
from taranis.timestepping import get_scheme

NX = 64
B0 = 2.0
NPART = 96


def _params(particles=None, **over):
    kw = dict(dims=2, nx=NX, ny=NX, Lx=2 * np.pi, Ly=2 * np.pi, dt=0.01,
              adaptive_timestep=False, diss=1e-2, hyper=1, cfl_safety=0.5,
              particles=particles)
    kw.update(over)
    return fresh_params(**kw)


def _pconfig(ensembles, n=NPART, **over):
    return dict({"seed": 7, "n": n, "B0": B0, "ensembles": ensembles}, **over)


def _fields(phi, psi, x, y):
    # initialize() passes x as (1,nx,1) and y as (1,1,ny): broadcast both fields to the
    # full (1,nx,ny) plane before stacking
    zero = jnp.zeros_like(x + y)
    return jnp.stack([phi + zero, psi + zero], axis=0)


# --------------------------------------------------------------------- read_moments

_HAND_COLS = ("t", "ensemble") + MOMENTS


def _write_sidecar(directory, cols, rows):
    with open(os.path.join(directory, "particle_moments.txt"), "w") as f:
        f.write("# " + " ".join(cols) + "\n")
        for r in rows:
            f.write(" ".join("%.17g" % v for v in r) + "\n")


def _hand_value(t, e):
    # a distinct base per (t, ensemble), so a mis-mapped column or a mis-grouped
    # ensemble cannot pass by coincidence
    return 1e6 * t + 1000.0 * e


def _hand_rows(times, ens, ncol):
    return [[t, float(e)] + [_hand_value(t, e) + c for c in range(ncol)]
            for t in times for e in ens]


def test_read_moments():
    """The sidecar is parsed by header name, sorted by t and grouped per ensemble."""
    times, n_ens = [0.03, 0.01, 0.02], 2
    ncol = len(MOMENTS)
    rows = _hand_rows(times, range(n_ens), ncol)
    with tempfile.TemporaryDirectory() as d:
        _write_sidecar(d, _HAND_COLS, rows[::-1])          # rows in reverse order
        mom = pdiag.read_moments(d)
    # a header in a different order, with only some of the moment columns
    with tempfile.TemporaryDirectory() as d:
        _write_sidecar(d, ("vz2", "ensemble", "t", "vperp2"),
                       [[r[2 + MOMENTS.index("vz2")], r[1], r[0],
                         r[2 + MOMENTS.index("vperp2")]] for r in rows])
        narrow = pdiag.read_moments(d)
    with tempfile.TemporaryDirectory() as d:
        _write_sidecar(d, _HAND_COLS, rows + _hand_rows([0.02], range(n_ens), ncol))
        dup = _message(pdiag.read_moments, d)
    with tempfile.TemporaryDirectory() as d:
        # two times each carrying only one of the two ensembles
        _write_sidecar(d, _HAND_COLS, rows + _hand_rows([0.04], [0], ncol)
                       + _hand_rows([0.05], [1], ncol))
        ragged = _message(pdiag.read_moments, d)
    # and a file written by the real writer, from synthetic ys
    ts = np.array([0.01, 0.02, 0.03])
    vals = np.arange(ts.size * n_ens * len(MOMENTS), dtype=float).reshape(ts.size, n_ens, -1)
    with tempfile.TemporaryDirectory() as d:
        run._append_moments(d, (ts, vals))
        written = pdiag.read_moments(d)

    ref = np.array([[_hand_value(t, e) for e in range(n_ens)] for t in sorted(times)])
    with checks() as c:
        c.check("t is sorted and 1-D", np.allclose(mom["t"], sorted(times))
                and mom["t"].shape == (3,), str(mom["t"]))
        c.check("every moment column is returned as (nsteps, n_ens)",
                set(mom) == {"t"} | set(MOMENTS)
                and all(mom[k].shape == (3, n_ens) for k in MOMENTS), str(sorted(mom)))
        c.check("'t' and 'ensemble' are consumed as row keys, not returned as data",
                "ensemble" not in mom)
        c.check("columns land on the right (step, ensemble) despite the shuffled rows",
                all(np.allclose(mom[name], ref + i) for i, name in enumerate(MOMENTS)),
                str(mom[MOMENTS[0]]))
        c.check("a reordered header with only some columns parses by name",
                set(narrow) == {"t", "vz2", "vperp2"}
                and np.allclose(narrow["vperp2"], mom["vperp2"])
                and np.allclose(narrow["vz2"], mom["vz2"]), str(sorted(narrow)))
        c.check("a duplicated time is a ValueError", "repeated time" in dup, dup)
        c.check("a missing ensemble row is a ValueError",
                "missing ensemble row" in ragged, ragged)
        c.check("round-trips run._append_moments' own output",
                np.allclose(written["t"], ts)
                and all(np.allclose(written[name], vals[:, :, i])
                        for i, name in enumerate(MOMENTS)), str(written["t"]))


# ------------------------------------------------------------ heating_rate / mu_diffusion

def test_heating_rate_and_mu_diffusion():
    """Planted slopes, the 0.5 factor on the velocity-square columns only, the [t0,t1]
    window, and an error bar that is zero for a noiseless line and grows with scatter."""
    t = np.arange(400) * 0.01
    rates = np.array([0.3, -0.05])
    mom = {"t": t,
           "vperp2": 2.0 * (rates * t[:, None] + 1.0),      # f = 0.5*vperp2 = rate*t + 1
           "vperpB2": 2.0 * (rates * t[:, None] + 1.0),     # the local-B columns too
           "vparB2": 2.0 * (rates * t[:, None] + 1.0),
           "w_eperp": rates * t[:, None],                   # already an energy: no 0.5
           "mu": np.full((t.size, 2), 0.7),
           "mu2": 0.49 + 2.0 * np.array([0.02, 0.005]) * t[:, None]}
    rate, err = pdiag.heating_rate(mom)
    localb = [pdiag.heating_rate(mom, key=k)[0] for k in ("vperpB2", "vparB2")]
    wrate, _ = pdiag.heating_rate(mom, key="w_eperp")
    D, Derr = pdiag.mu_diffusion(mom)
    # a kink at t = 2: the window must see only the second slope
    kink = np.where(t[:, None] < 2.0, 2.0 * rates * t[:, None],
                    2.0 * (rates * 2.0 + 3.0 * rates * (t[:, None] - 2.0)))
    late, _ = pdiag.heating_rate({"t": t, "vperp2": kink}, t0=2.0)
    early, _ = pdiag.heating_rate({"t": t, "vperp2": kink}, t1=2.0)
    rng = np.random.default_rng(0)
    noisy = {"t": t, "vperp2": mom["vperp2"] + 0.02 * rng.standard_normal((t.size, 2))}
    nrate, nerr = pdiag.heating_rate(noisy)
    print(f"heating_rate: {rate} +- {err}; noisy {nrate} +- {nerr}; D_mu {D} +- {Derr}")
    with checks() as c:
        c.check("OLS slope of 0.5*vperp2 recovers the planted rate",
                np.allclose(rate, rates, rtol=1e-12), str(rate))
        c.check("a noiseless line has a zero block-to-block error bar",
                np.allclose(err, 0.0, atol=1e-12), str(err))
        c.check("the local-B velocity-square columns are halved like vperp2/vz2 (they are "
                "what a heating rate is measured from)",
                all(np.allclose(r, rates, rtol=1e-12) for r in localb), str(localb))
        c.check("the w_* columns are NOT halved", np.allclose(wrate, rates, rtol=1e-12),
                str(wrate))
        c.check("t0 selects the late slope and t1 the early one",
                np.allclose(late, 3.0 * rates, rtol=1e-12)
                and np.allclose(early, rates, rtol=1e-12), f"{late} {early}")
        c.check("mu_diffusion returns 0.5 d Var(mu)/dt",
                np.allclose(D, [0.02, 0.005], rtol=1e-12), str(D))
        c.check("noise: the error bar is positive and covers the true rate within 3 sigma",
                np.all(nerr > 0.0) and np.all(np.abs(nrate - rates) < 3.0 * nerr),
                f"{nrate} {nerr}")
        c.check("too few samples for the blocks is a ValueError",
                _raises(pdiag.heating_rate, {"t": t[:4], "vperp2": mom["vperp2"][:4]}))


def _raises(fn, *a, **kw):
    return _message(fn, *a, **kw) != "no ValueError"


def _message(fn, *a, **kw):
    try:
        fn(*a, **kw)
        return "no ValueError"
    except ValueError as e:
        return str(e)


# -------------------------------------------------------- delta_u / xi / chandran_fit

def test_delta_u_and_chandran_fit():
    """delta_u = sqrt(2 k Ekin) log-log interpolated is exact on a power law, and the
    Chandran fit recovers (c1, c2) from data built with its own formula."""
    k = np.linspace(1.0, 64.0, 64)
    A, p = 0.4, 5.0 / 3.0
    Ekin = A * k ** (-p)
    rho = np.array([0.02, 0.05, 0.2, 0.5])
    exact = np.sqrt(2.0 * A) * (1.0 / rho) ** (0.5 * (1.0 - p))
    du = pdiag.delta_u(k, Ekin, rho)
    scalar = pdiag.delta_u(k, Ekin, 0.1)
    vperp = np.array([1.0, 0.5, 2.0, 1.5])
    x = pdiag.xi(du, vperp)
    c1_true, c2_true = 0.7, 0.34
    Q = c1_true * du ** 3 / rho * np.exp(-c2_true / x)
    c1, c2, info = pdiag.chandran_fit(x, Q, rho, du)
    with checks() as c:
        c.check(f"delta_u is exact on a power law (max rel {np.max(abs(du / exact - 1)):.1e})",
                np.allclose(du, exact, rtol=1e-12), str(du - exact))
        c.check("a scalar rho gives a float", isinstance(scalar, float))
        c.check("xi = delta_u/v_perp", np.allclose(x, du / vperp, rtol=1e-14))
        c.check(f"chandran_fit recovers c1 = {c1:.6f}, c2 = {c2:.6f}",
                abs(c1 - c1_true) < 1e-10 and abs(c2 - c2_true) < 1e-10, f"{c1} {c2}")
        c.check("... with R^2 = 1 and no dropped points",
                abs(info["r2"] - 1.0) < 1e-12 and info["n_used"] == rho.size,
                f"{info['r2']} {info['n_used']}")
        c.check("a spectrum with no usable bins is a ValueError",
                _raises(pdiag.delta_u, k, np.zeros_like(k), 0.1))


def test_kinetic_spectrum_normalization():
    """sum(Ekin)*dk = 0.5<u^2> for phi = a cos(k0 x) -- the convention behind
    <u^2> = 2 integral Ekin dk, i.e. delta_u^2 = 2 k Ekin(k)."""
    a, k0 = 0.7, 3.0
    params = _params()
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(lambda x, y: _fields(a * jnp.cos(k0 * x), 0.0, x, y), params)
    k, Ekin = pdiag.kinetic_spectrum(state, kgrid, params, bin_factor=1.0)
    dk = float(k[1] - k[0])
    total = float(np.sum(Ekin) * dk)
    exact = 0.25 * a ** 2 * k0 ** 2                        # 0.5*<|grad phi|^2>
    peak = float(k[np.argmax(Ekin)])
    print(f"kinetic_spectrum: sum(E)dk = {total:.8f} vs 0.5<u^2> = {exact:.8f}, "
          f"peak k = {peak}")
    with checks() as c:
        c.check(f"sum(Ekin)*dk = 0.5<u^2> ({total:.6f} vs {exact:.6f})",
                abs(total / exact - 1.0) < 1e-5, f"{total} {exact}")
        c.check(f"the single mode lands in the k = {k0} bin", abs(peak - k0) <= dk)


# ------------------------------------------------------------------- mu_of / jz_at

def _placed(params, seed=0):
    # a ParticleState at random positions/velocities (init_particles, then replace: the
    # NamedTuple's fields are built by the package, never positionally here)
    rng = np.random.default_rng(seed)
    n = params.particles["n"]
    n_ens = params.n_ens
    x = np.stack([np.stack([rng.uniform(0, params.Lx, n), rng.uniform(0, params.Ly, n),
                            np.zeros(n)], axis=1) for _ in range(n_ens)])
    v = rng.standard_normal((n_ens, n, 3))
    return init_particles(params)._replace(x=jnp.asarray(x), v=jnp.asarray(v))


_ENS = [{"qm": 5.0, "init": {"kind": "ring", "vperp": 1.0}},
        {"qm": 5.0, "init": {"kind": "ring", "vperp": 1.0}, "bperp": False}]


def test_mu_of():
    """mu = v_perpB^2/(2|B|) about the LOCAL B: exact for psi = 0, and to bilinear
    tolerance for psi = a cos(x) (b_perp = zhat x grad psi = (0, -a sin x)). The
    bperp = False ensemble sees B = B0 zhat, as its particles do."""
    a = 0.5
    params = _params(particles=_pconfig(_ENS))
    kgrid = jr.setup_kgrids(params)
    ps = _placed(params)
    v = np.asarray(ps.v)
    uniform = jr.initialize(lambda x, y: _fields(0.0, 0.0, x, y), params)
    sheared = jr.initialize(lambda x, y: _fields(0.0, a * jnp.cos(x), x, y), params)
    mu_u = pdiag.mu_of(ps, uniform, kgrid, params)
    mu_s = pdiag.mu_of(ps, sheared, kgrid, params)
    exact_u = (v[:, :, 0] ** 2 + v[:, :, 1] ** 2) / (2.0 * B0)
    xp = np.asarray(ps.x[0])[:, 0]
    B = np.stack([np.zeros_like(xp), -a * np.sin(xp), np.full_like(xp, B0)], axis=1)
    exact_s0 = pdiag._mu(v[0], B)
    err = float(np.max(np.abs(mu_s[0] / exact_s0 - 1.0)))
    print(f"mu_of: psi = {a} cos(x) max rel error vs analytic = {err:.3e} "
          f"(bilinear, dx = {params.dx:.4f})")
    with checks() as c:
        c.check("psi = 0: mu = (v_x^2+v_y^2)/(2 B0) exactly, with B0 from the config",
                np.allclose(mu_u, exact_u, rtol=1e-13, atol=0.0), str(mu_u[0][:3]))
        c.check(f"psi = a cos(x): matches the analytic local B to {err:.1e} (<= 5e-3)",
                err <= 5e-3, str(err))
        c.check("the local-B mu differs from v_perp^2/2B0 by more than that error",
                np.max(np.abs(mu_s[0] / exact_u[0] - 1.0)) > 50.0 * err,
                str(np.max(np.abs(mu_s[0] / exact_u[0] - 1.0))))
        c.check("a bperp = False ensemble keeps B = B0 zhat in the same field",
                np.allclose(mu_s[1], exact_u[1], rtol=1e-13, atol=0.0),
                str(np.max(np.abs(mu_s[1] / exact_u[1] - 1.0))))
        c.check("shape (n_ens, n) float64",
                mu_s.shape == (2, NPART) and mu_s.dtype == np.float64)


def test_mu_of_uses_the_per_ensemble_B0():
    """B0 is a per-ensemble knob (the amplitude parameter), so mu_of must read each
    ensemble's own value, not the top-level default -- with psi = 0 that is exactly
    v_perp^2/(2 B0_e)."""
    ens = [{"qm": 5.0, "init": {"kind": "ring", "vperp": 1.0}},
           {"qm": 1.0, "init": {"kind": "ring", "vperp": 1.0}, "B0": 5.0}]
    params = _params(particles=_pconfig(ens))
    kgrid = jr.setup_kgrids(params)
    ps = _placed(params, seed=4)
    v = np.asarray(ps.v)
    mu = pdiag.mu_of(ps, jr.initialize(lambda x, y: _fields(0.0, 0.0, x, y), params),
                     kgrid, params)
    b0s = [e["B0"] for e in params.particles["ensembles"]]
    exact = np.stack([(v[e, :, 0] ** 2 + v[e, :, 1] ** 2) / (2.0 * b0)
                      for e, b0 in enumerate(b0s)])
    with checks() as c:
        c.check(f"the ensembles carry B0 = {b0s} (top-level {B0} is only the default)",
                b0s == [B0, 5.0])
        c.check("mu = v_perp^2/(2 B0_e) per ensemble, each with its own B0",
                np.allclose(mu, exact, rtol=1e-13, atol=0.0), str(mu[:, :2] - exact[:, :2]))


def test_mu_of_matches_the_pushed_moment():
    """mu_of is the snapshot-cadence counterpart of the sidecar's "mu" column, so on the
    state the push itself used they must agree: the push's last half-kick samples B at
    the final position, which is what mu_of gathers. A different B0, a dropped mask or
    v_perp about zhat instead of about B would all show up here."""
    params = _params(particles=_pconfig(_ENS))
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(_live_ic, params)
    p1, mom = push_ensembles(_placed(params, seed=2), state, kgrid, params, 0.01)
    mu = pdiag.mu_of(p1, state, kgrid, params)
    mom = np.asarray(mom)
    ref, ref2 = mom[:, MOMENTS.index("mu")], mom[:, MOMENTS.index("mu2")]
    with checks() as c:
        c.check("mean(mu_of) equals the pushed moments' mu column",
                np.allclose(mu.mean(axis=1), ref, rtol=1e-12, atol=0.0),
                str(mu.mean(axis=1) - ref))
        c.check("mean(mu_of^2) equals its mu2 column",
                np.allclose((mu ** 2).mean(axis=1), ref2, rtol=1e-12, atol=0.0),
                str((mu ** 2).mean(axis=1) - ref2))


def test_jz_at():
    """j_z = laplacian(psi): psi = cos(x) gives -cos(x) at the particles."""
    params = _params(particles=_pconfig(_ENS[:1]))
    kgrid = jr.setup_kgrids(params)
    ps = _placed(params, seed=1)
    state = jr.initialize(lambda x, y: _fields(0.0, jnp.cos(x), x, y), params)
    jz = pdiag.jz_at(ps, state, kgrid, params)
    exact = -np.cos(np.asarray(ps.x[0])[:, 0])
    err = float(np.max(np.abs(jz[0] - exact)))
    print(f"jz_at: max |j_z - (-cos x)| = {err:.3e} (bilinear, dx = {params.dx:.4f})")
    with checks() as c:
        c.check(f"j_z = -cos(x) at the particles to {err:.1e} (<= 5e-3)", err <= 5e-3,
                str(err))
        c.check("shape (n_ens, n) float64",
                jz.shape == (1, NPART) and jz.dtype == np.float64)


# ------------------------------------------ conditional_stats / histogram / increments

def test_conditional_stats():
    """Equal-count quantile bins of |cond|, with the per-bin moments of q."""
    n = 100
    cond = np.stack([np.arange(n, dtype=float), -np.arange(n, dtype=float)])
    q = np.stack([np.arange(n, dtype=float) ** 2, np.ones(n)])
    st = pdiag.conditional_stats(q, cond, nq=4)
    quarters = np.arange(n, dtype=float).reshape(4, 25)
    with checks() as c:
        c.check("shapes: edges (n_ens, nq+1), the rest (n_ens, nq)",
                st["edges"].shape == (2, 5) and st["count"].shape == (2, 4)
                and st["mean"].shape == (2, 4) and st["mean_sq"].shape == (2, 4))
        c.check("equal counts in equal-quantile bins", np.all(st["count"] == 25),
                str(st["count"]))
        c.check("bins are on |cond|, so the sign of cond does not reorder them",
                np.allclose(st["mean"][0], np.mean(quarters ** 2, axis=1))
                and np.allclose(st["count"][1], st["count"][0]), str(st["mean"][0]))
        c.check("mean_sq is <q^2>, not <q>^2",
                np.allclose(st["mean_sq"][0], np.mean(quarters ** 4, axis=1)),
                str(st["mean_sq"][0]))
        c.check("edges are the |cond| quantiles",
                np.allclose(st["edges"][0], np.quantile(np.abs(cond[0]),
                                                        np.linspace(0, 1, 5))))
        c.check("mismatched shapes are a ValueError",
                _raises(pdiag.conditional_stats, q, cond[:1]))


def test_increment_histogram_and_diffusion():
    """Shared edges across ensembles, and the snapshot-pair diffusion estimator."""
    rng = np.random.default_rng(3)
    q = np.stack([rng.standard_normal(500), 3.0 * rng.standard_normal(500)])
    edges, counts = pdiag.increment_histogram(q, bins=21)
    edges_r, counts_r = pdiag.increment_histogram(q, bins=10, range=(-1.0, 1.0))
    q0 = np.zeros((2, 7))
    q1 = np.stack([np.full(7, 0.2), np.full(7, -0.5)])
    D = pdiag.increment_diffusion(q0, q1, 0.25)
    with checks() as c:
        c.check("shapes: edges (bins+1,), counts (n_ens, bins)",
                edges.shape == (22,) and counts.shape == (2, 21))
        c.check("every particle is binned (the default range spans all ensembles)",
                np.all(counts.sum(axis=1) == 500)
                and edges[0] <= q.min() and edges[-1] >= q.max())
        c.check("the wider ensemble has the fatter tails",
                counts[1, 0] + counts[1, -1] > counts[0, 0] + counts[0, -1],
                str(counts[:, [0, -1]]))
        c.check("an explicit range is honoured (and clips)",
                edges_r[0] == -1.0 and edges_r[-1] == 1.0
                and np.all(counts_r.sum(axis=1) < 500))
        c.check("increment_diffusion = 0.5<(q1-q0)^2>/dt per ensemble",
                np.allclose(D, [0.5 * 0.04 / 0.25, 0.5 * 0.25 / 0.25], rtol=1e-14),
                str(D))


# ------------------------------------------------- work_split / energy_budget (live run)

# full dpsi/dt, the default ideal-Ohm particle, and the E = 0 control -- the production
# paired-ensemble configuration of the plan's §1
_LIVE_ENS = [{"qm": 5.0, "init": {"kind": "ring", "vperp": 1.0},
              "ez_resistive": True, "ez_forcing": True},
             {"qm": 5.0, "init": {"kind": "ring", "vperp": 1.0}},
             {"qm": 5.0, "init": {"kind": "ring", "vperp": 1.0},
              "eperp": False, "ez_ideal": False}]
_LIVE_STEPS = 60


def _live_ic(x, y):
    return jnp.stack([jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y),
                      jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)], axis=0)


def test_energy_budget_closure():
    """The work accumulators account for the whole kinetic-energy change of a live
    co-stepped run: energy_budget's "closure" is round-off. The E = 0 control's work is
    EXACTLY zero in every piece, and the resistive piece separates the paired ensembles --
    which is the measurement the production setup exists to make."""
    params = _params(particles=_pconfig(_LIVE_ENS, n=128),
                     diss=1e-1, forcing=True, forcing_mode="elsasser",
                     forcing_power_elsasser=(0.5, 0.5), forcing_tau=1.0, fshell=(1, 3))
    kgrid = jr.setup_kgrids(params)
    state = run._refresh_forcing_scale(jr.initialize(_live_ic, params), kgrid, params)
    p0 = init_particles(params)
    stepper, scheme = get_scheme("lsrk33")
    step = jax.jit(run.block_of_steps, static_argnums=(2, 3, 4, 5))
    (_sT, pT), _ys = step((state, p0), kgrid, params, _LIVE_STEPS, scheme, stepper)
    bud = pdiag.energy_budget(pT, p0)
    split = pdiag.work_split(pT)
    ires = WORK_PIECES.index("ez_resistive")
    rel = np.abs(bud["closure"])/np.maximum(np.abs(bud["dke"]), 1e-30)
    print(f"energy_budget: dke {bud['dke']}, total_work {bud['total_work']}, "
          f"closure {bud['closure']} (rel {rel})")
    print(f"work_split by piece {WORK_PIECES}:\n{split}")
    with checks() as c:
        c.check("pieces are WORK_PIECES and the shapes are (n_ens,) / (n_ens, NWORK)",
                bud["pieces"] == WORK_PIECES and bud["dke"].shape == (3,)
                and bud["work"].shape == (3, len(WORK_PIECES)))
        c.check(f"closure = dke - total_work is round-off (max rel {rel[:2].max():.2e} "
                f"<= 1e-12)", rel[:2].max() <= 1e-12, str(bud["closure"]))
        c.check("the accelerated ensembles actually heated (the closure is not trivial)",
                np.all(np.abs(bud["dke"][:2]) > 0.01), str(bud["dke"]))
        c.check("the E = 0 control does exactly zero work in every piece, and its "
                "kinetic energy is conserved to round-off",
                np.all(bud["work"][2] == 0.0) and abs(bud["dke"][2]) < 1e-12,
                f"{bud['work'][2]} {bud['dke'][2]}")
        c.check("the paired ensembles are separated by w_ez_resistive: nonzero with the "
                "full mask, EXACTLY zero for the ideal-Ohm default",
                bud["work"][0, ires] != 0.0 and bud["work"][1, ires] == 0.0,
                str(bud["work"][:, ires]))
        c.check("work_split without a reference state is the work since init",
                np.array_equal(split, bud["work"]))


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
