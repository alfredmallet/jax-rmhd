# Test-particle diagnostics: read-only post-processing of a ParticleState, of the
# particle_moments.txt sidecar simulate_scan writes, and of the fields at the particles.
# plans/TESTPART_PLAN.md §5; conventions in docs/numerics.md "Test particles".
#
# Everything here is host-side numpy except the few functions that need a gather or an
# fft (mu_of, jz_at, kinetic_spectrum). Nothing here is called from the solver.
import os

import jax.numpy as jnp
import numpy as np

from .. import grids
from ..particles import interp
from ..particles.fields import WORK_PIECES, particle_fields
from .rmhd import perpspec

# run.py's sidecar file name (duplicated, not imported: diagnostics never depends on the
# run driver)
_MOMENTS_FILE = "particle_moments.txt"

# columns the sidecar stores as <v^2> means, which are halved to make an energy rate
_HALF_KEYS = ("vperp2", "vz2", "vperpB2", "vparB2")


############################
# the moments sidecar      #
############################

def read_moments(directory):
    """Parse <directory>/particle_moments.txt into
    {"t": (nsteps,), "<column>": (nsteps, n_ens)}.

    The header names the columns, so column ORDER and a narrower/wider set of moment
    columns are both tolerated; "t" and "ensemble" are consumed as the row keys and are
    not returned as data columns. Rows are sorted by t. Every ensemble must appear
    exactly once per t, and a repeated t is a ValueError (run.py truncates the sidecar on
    restart precisely so that cannot happen)."""
    path = os.path.join(str(directory), _MOMENTS_FILE)
    names, rows = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if names is None:
                    names = line.lstrip("#").split()
                continue
            rows.append([float(v) for v in line.split()])
    if names is None:
        raise ValueError(f"{path}: no '# t ensemble ...' header line")
    if not rows:
        raise ValueError(f"{path}: header only, no data rows")
    for key in ("t", "ensemble"):
        if key not in names:
            raise ValueError(f"{path}: header {names} has no {key!r} column")
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] != len(names):
        raise ValueError(f"{path}: {data.shape[1]} columns of data against "
                         f"{len(names)} header names {names}")
    it, ie = names.index("t"), names.index("ensemble")
    ens = data[:, ie].astype(np.int64)
    n_ens = int(ens.max()) + 1
    if set(ens.tolist()) != set(np.arange(n_ens).tolist()):
        raise ValueError(f"{path}: ensemble column {sorted(set(ens.tolist()))} is not "
                         f"0..{n_ens - 1}")
    data = data[np.lexsort((ens, data[:, it]))]
    ens = data[:, ie].astype(np.int64)
    same = (data[1:, it] == data[:-1, it]) & (ens[1:] == ens[:-1])
    if np.any(same):
        raise ValueError(f"{path}: repeated time(s) "
                         f"{np.unique(data[1:, it][same])} for the same ensemble -- a "
                         f"restart appended without truncating")
    t = np.unique(data[:, it])
    if data.shape[0] != t.size*n_ens:
        raise ValueError(f"{path}: {data.shape[0]} rows against {t.size} times x "
                         f"{n_ens} ensembles -- some t has a missing ensemble row")
    # fold the (step, ensemble) row-pair structure out
    block = data.reshape(t.size, n_ens, data.shape[1])
    if not np.array_equal(block[:, :, ie].astype(np.int64),
                          np.broadcast_to(np.arange(n_ens), block.shape[:2])):
        raise ValueError(f"{path}: some t has a missing or repeated ensemble row")
    out = {"t": t}
    out.update({name: block[:, :, i] for i, name in enumerate(names)
                if i not in (it, ie)})
    return out


def heating_rate(mom, key="vperp2", t0=None, t1=None, nblocks=5):
    """OLS slope of f(t) over [t0, t1] per ensemble -> (rate (n_ens,), err (n_ens,)).

    f = 0.5*mom[key] for the velocity-square columns ("vperp2", "vz2", "vperpB2",
    "vparB2"), so the rate is an energy-per-unit-mass rate; f = mom[key] for anything else
    (the w_* work columns are already energies, mu is not an energy at all). The local-B
    columns are the ones a heating measurement wants (docs/numerics.md, "Test particles").
    `err` is the standard deviation
    (ddof=1) of the slopes of `nblocks` contiguous sub-windows divided by sqrt(nblocks):
    a time series this correlated makes the OLS residual error meaningless."""
    t = np.asarray(mom["t"], dtype=np.float64)
    f = np.asarray(mom[key], dtype=np.float64)
    if key in _HALF_KEYS:
        f = 0.5*f
    sel = np.ones(t.shape, dtype=bool)
    if t0 is not None:
        sel &= t >= t0
    if t1 is not None:
        sel &= t <= t1
    t, f = t[sel], f[sel]
    if t.size < 2:
        raise ValueError(f"heating_rate: {t.size} sample(s) in the window [{t0}, {t1}]")
    if nblocks < 1 or t.size // nblocks < 2:
        raise ValueError(f"heating_rate: {t.size} samples cannot make {nblocks} blocks "
                         f"of >= 2 points")
    edges = np.linspace(0, t.size, nblocks + 1).astype(int)
    slopes = np.stack([_ols_slope(t[a:b], f[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    err = (np.std(slopes, axis=0, ddof=1)/np.sqrt(nblocks) if nblocks > 1
           else np.zeros(f.shape[1]))
    return _ols_slope(t, f), err


def _ols_slope(t, f):
    # least-squares d f/d t, one slope per column of f (nsteps, n_ens)
    dt = t - t.mean()
    return dt @ (f - f.mean(axis=0)) / (dt @ dt)


def mu_diffusion(mom, t0=None, t1=None, nblocks=5):
    """D_mu = 0.5 d Var(mu)/dt per ensemble -> (D (n_ens,), err), from the sidecar's
    mu/mu2 columns (Var = <mu^2> - <mu>^2) through heating_rate's slope machinery."""
    var = np.asarray(mom["mu2"], dtype=np.float64) - np.asarray(mom["mu"],
                                                                dtype=np.float64)**2
    rate, err = heating_rate({"t": mom["t"], "var": var}, key="var", t0=t0, t1=t1,
                             nblocks=nblocks)
    return 0.5*rate, 0.5*err


############################
# scales and the xi fit    #
############################

def gyroradius(vperp, qm, B0=1.0):
    """rho = v_perp/(|q/m| B0), with the ensemble's own B0 (docs/numerics.md "E_par
    projection and the amplitude parameter": B0 = 1/eps and qm*B0 is the gyrofrequency);
    the sign of qm sets the gyration sense, not a length, so it is taken out here."""
    return np.abs(np.asarray(vperp, dtype=np.float64)/(np.asarray(qm, dtype=np.float64)*B0))


def gyrofrequency(qm, B0=1.0):
    """Omega = (q/m) B0, signed (the sense of gyration)."""
    return np.asarray(qm, dtype=np.float64)*B0


def kinetic_spectrum(state, kgrid, params, bin_factor=1.0):
    """(k, Ekin) from diagnostics.rmhd.perpspec's phi column: Ekin(k) is 0.5|grad_perp phi|^2
    per unit k, i.e. the kinetic energy per unit k with 0.5<u^2> = integral Ekin dk.

    perpspec is z-averaged, so in 3D this is the perpendicular spectrum of the whole box --
    which is the one delta_u(rho) wants, rho being a perpendicular scale."""
    k, ekin, _emag = perpspec(state, kgrid, params, bin_factor=bin_factor)
    return np.asarray(k, dtype=np.float64), np.asarray(ekin, dtype=np.float64)


def delta_u(k, Ekin, rho):
    """The turbulent velocity at scale rho: delta_u = sqrt(2 k Ekin(k)) at k = 1/rho.

    Convention: <u^2> = 2 integral Ekin dk, so 2 k Ekin(k) is the velocity SQUARED per
    logarithmic band at k -- the standard delta_u(l) with l = 1/k. Interpolation is
    linear in (log k, log 2kEkin), which is exact for a power-law spectrum; empty and
    non-positive bins are dropped, and a k outside the binned range clamps to the nearest
    end (np.interp). Returns a float for a scalar rho, an array otherwise."""
    k = np.asarray(k, dtype=np.float64)
    g = 2.0*k*np.asarray(Ekin, dtype=np.float64)
    good = (k > 0.0) & (g > 0.0) & np.isfinite(g)
    if good.sum() < 2:
        raise ValueError(f"delta_u: {int(good.sum())} usable spectrum bin(s)")
    kq = 1.0/np.asarray(rho, dtype=np.float64)
    du = np.sqrt(np.exp(np.interp(np.log(kq), np.log(k[good]), np.log(g[good]))))
    return float(du) if np.ndim(rho) == 0 else du


def xi(delta_u, vperp):
    """The stochastic-heating parameter xi = delta_u(rho)/v_perp."""
    return np.asarray(delta_u, dtype=np.float64)/np.asarray(vperp, dtype=np.float64)


def chandran_fit(xi, Q, rho, delta_u):
    """Fit Q = c1 (delta_u^3/rho) exp(-c2/xi) -> (c1, c2, info).

    Linear least squares of log(Q rho/delta_u^3) = log c1 - c2/xi over the points with a
    finite, positive left-hand side (the rest are dropped and counted in `info`). This is
    the Chandran et al. 2010 stochastic-heating form; c2 is the headline number."""
    x = 1.0/np.asarray(xi, dtype=np.float64)
    y = np.log(np.asarray(Q, dtype=np.float64)*np.asarray(rho, dtype=np.float64)
               / np.asarray(delta_u, dtype=np.float64)**3)
    used = np.isfinite(x) & np.isfinite(y)
    if used.sum() < 2:
        raise ValueError(f"chandran_fit: {int(used.sum())} usable point(s)")
    coef = np.linalg.lstsq(np.stack([np.ones(used.sum()), -x[used]], axis=1), y[used],
                           rcond=None)[0]
    log_c1, c2 = float(coef[0]), float(coef[1])
    fit = log_c1 - c2*x
    res = y - fit
    ss_tot = float(np.sum((y[used] - y[used].mean())**2))
    info = {"log_c1": log_c1, "x": x, "y": y, "fit": fit, "residuals": res,
            "r2": 1.0 - float(np.sum(res[used]**2))/ss_tot if ss_tot > 0.0 else np.nan,
            "used": used, "n_used": int(used.sum())}
    return float(np.exp(log_c1)), c2, info


############################
# fields at the particles  #
############################

def mu_of(pstate, state, kgrid, params):
    """Per-particle magnetic moment per unit mass about the LOCAL B -> (n_ens, n) float64.

    mu = v_perpB^2/(2|B|) with v_perpB^2 = |v|^2 - (v.B)^2/|B|^2 and B = (b_perp, B0)
    gathered at pstate.x (bilinear in 2D, trilinear in 3D) -- the snapshot-cadence counterpart of the sidecar's
    "mu" column, and the same formula. B0 is the ensemble's own, as the push uses. An
    ensemble with `bperp` off sees B = B0 zhat, as its particles do."""
    cfg = params.particles
    pf = particle_fields(state, kgrid, params)
    bperp = jnp.stack([pf.bx, pf.by])
    out = []
    for e, ens in enumerate(cfg["ensembles"]):
        b = np.asarray(interp.gather(bperp, pstate.x[e], params), dtype=np.float64)
        if not ens["mask"]["bperp"]:
            b = np.zeros_like(b)
        B = np.concatenate([b, np.full((b.shape[0], 1), ens["B0"], dtype=np.float64)],
                           axis=1)
        out.append(_mu(np.asarray(pstate.v[e], dtype=np.float64), B))
    return np.stack(out)


def _mu(v, B):
    # v_perpB^2/(2|B|), both (N,3)
    bsq = np.sum(B*B, axis=1)
    return (np.sum(v*v, axis=1) - np.sum(v*B, axis=1)**2/bsq)/(2.0*np.sqrt(bsq))


def jz_at(pstate, state, kgrid, params):
    """j_z = laplacian(psi) = ifft(-ksq psik), gathered at the particle (bilinear in 2D, trilinear in 3D)
    positions -> (n_ens, n) float64. The conditioning variable for current-sheet
    acceleration statistics."""
    jz = grids.ifft(-kgrid.ksq*state.fields[1], params)[None]
    return np.stack([np.asarray(interp.gather(jz, pstate.x[e], params)[:, 0],
                                dtype=np.float64)
                     for e in range(pstate.x.shape[0])])


############################
# particle statistics      #
############################

def conditional_stats(q, cond, nq=4):
    """Per-ensemble statistics of q in quantile bins of |cond|; q and cond both (n_ens, n).

    Returns {"edges": (n_ens, nq+1), "count": (n_ens, nq), "mean": (n_ens, nq),
    "mean_sq": (n_ens, nq)}. Bin edges are per-ensemble quantiles of |cond|, so the bins
    hold equal particle counts up to ties; an empty bin's means are nan."""
    q = np.asarray(q, dtype=np.float64)
    c = np.abs(np.asarray(cond, dtype=np.float64))
    if q.shape != c.shape:
        raise ValueError(f"conditional_stats: q {q.shape} and cond {c.shape} must match")
    n_ens = q.shape[0]
    edges = np.empty((n_ens, nq + 1))
    count = np.zeros((n_ens, nq))
    mean = np.full((n_ens, nq), np.nan)
    mean_sq = np.full((n_ens, nq), np.nan)
    for e in range(n_ens):
        edges[e] = np.quantile(c[e], np.linspace(0.0, 1.0, nq + 1))
        # interior edges only: searchsorted then clip keeps the extremes in the end bins
        idx = np.clip(np.searchsorted(edges[e, 1:-1], c[e], side="right"), 0, nq - 1)
        count[e] = np.bincount(idx, minlength=nq)
        nz = count[e] > 0
        mean[e, nz] = (np.bincount(idx, weights=q[e], minlength=nq)/np.maximum(count[e], 1))[nz]
        mean_sq[e, nz] = (np.bincount(idx, weights=q[e]**2,
                                      minlength=nq)/np.maximum(count[e], 1))[nz]
    return {"edges": edges, "count": count, "mean": mean, "mean_sq": mean_sq}


def increment_histogram(q, bins=101, range=None):
    """Per-ensemble histogram of a per-particle increment q (n_ens, n) on SHARED edges
    -> (edges (bins+1,), counts (n_ens, bins)). `range` defaults to the min/max over
    every ensemble, which is what makes the ensembles' tails comparable."""
    q = np.asarray(q, dtype=np.float64)
    rng = (float(q.min()), float(q.max())) if range is None else (float(range[0]),
                                                                  float(range[1]))
    edges = np.histogram_bin_edges(np.empty(0), bins=bins, range=rng)
    counts = np.stack([np.histogram(row, bins=edges)[0] for row in q])
    return edges, counts


def increment_diffusion(q0, q1, dt):
    """Snapshot-pair diffusion estimator 0.5 <(q1 - q0)^2>/dt per ensemble -> (n_ens,),
    for per-particle arrays (n_ens, n) of mu, v_z, ... .

    Unlike mu_diffusion, which differentiates the ensemble VARIANCE and is therefore blind
    to a drifting mean, this estimator is not: a mean drift at rate D contributes
    0.5*(D*dt)^2/dt = 0.5*D^2*dt on top of the spread. Neither is it restricted to
    irreversible spreading -- unless `dt` is long enough to average the reversible part of
    q away, an oscillation of q is counted as diffusion exactly like a random walk, so the
    pair value is an UPPER bound on the transport
    coefficient (examples/test-particles-2D.ipynb measures both for mu and quantifies the
    gap against an E = 0 control)."""
    d = np.asarray(q1, dtype=np.float64) - np.asarray(q0, dtype=np.float64)
    return 0.5*np.mean(d*d, axis=1)/dt


############################
# work / energy budget     #
############################

def work_split(pstate, pstate0=None):
    """Per-ensemble mean work per unit mass by field piece -> (n_ens, NWORK), in
    WORK_PIECES order. With `pstate0` the work done since that state, otherwise since
    init (ParticleState.w is cumulative)."""
    w = np.asarray(pstate.w, dtype=np.float64)
    if pstate0 is not None:
        w = w - np.asarray(pstate0.w, dtype=np.float64)
    return np.mean(w, axis=1)


def energy_budget(pstate, pstate0):
    """Per-ensemble kinetic-energy budget between two particle states:

    {"dke": (n_ens,) mean(0.5|v|^2 - 0.5|v0|^2), "work": (n_ens, NWORK) by piece,
    "total_work": (n_ens,), "closure": dke - total_work, "pieces": WORK_PIECES}.

    The Boris rotation is norm-exact and each half-kick's work is accumulated with the
    exact same algebra, so "closure" is round-off -- a nonzero one means a field piece is
    doing work outside the accumulators."""
    v = np.asarray(pstate.v, dtype=np.float64)
    v0 = np.asarray(pstate0.v, dtype=np.float64)
    dke = 0.5*np.mean(np.sum(v*v, axis=2) - np.sum(v0*v0, axis=2), axis=1)
    work = work_split(pstate, pstate0)
    total = np.sum(work, axis=1)
    return {"dke": dke, "work": work, "total_work": total, "closure": dke - total,
            "pieces": WORK_PIECES}
