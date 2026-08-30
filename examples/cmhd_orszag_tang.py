# Compressible MHD (eqtype="CMHD") Orszag-Tang vortex, run as 2.5D.
#
# The standard compressible-MHD test problem, and the validation configuration of
# plans/CMHD_PLAN.md Phase C2. It is also the gate configuration of
# tests/test_cmhd_orszag_tang.py -- that file re-declares the IC rather than importing it,
# because tests do not depend on examples/; if you change the physics here, change it there
# too.
#
#   python examples/cmhd_orszag_tang.py                  # 256^2 x 4 to t = 0.5, with plots
#   python examples/cmhd_orszag_tang.py --n 128 --no-plot
#
# fp64 is strongly recommended (TARANIS_PRECISION=64); matplotlib is optional (--no-plot,
# or just let the import fail -- the run still prints its trace).
#
# ---------------------------------------------------------------------------- the problem
#
# The Athena normalization of the Orszag-Tang vortex (Stone, Gardiner, Teuben, Hawley &
# Simon 2008, ApJS 178, 137, section VIII.4 "Two-Dimensional MHD", figures 22-24, run to
# t_f = 1/2; the IC originates with Ryu, Jones & Frank 1995, ApJ 452, 785, figure 8) is:
# box [0,1]^2, gamma = 5/3, rho = 25/(36 pi), p = 5/(12 pi),
# u = (-sin 2pi y, sin 2pi x, 0), and B from Az = (B0/4pi) cos 4pi x + (B0/2pi) cos 2pi y
# with B0 = 1/sqrt(4 pi), i.e. B = (-B0 sin 2pi y, B0 sin 4pi x, 0).
#
# taranis works in Alfven units with the mean density scaled to 1
# (docs/numerics.md "Compressible MHD"), so that IC maps onto:
#
#     c_s = sqrt(gamma p/rho) = sqrt((5/3)(5/12pi)(36pi/25)) = 1        -> cs0 = 1
#     v_A = B0/sqrt(rho)      = (1/sqrt(4pi)) * 6 sqrt(pi)/5   = 0.6    -> B_code = 0.6 * pattern
#     rho -> 1,  u unchanged,  box [0,1]^2 unchanged
#     beta = 2 c_s^2/(gamma v_A^2) = 10/3,  M_s = M_A/... : u_rms = 1 so M_s = 1, M_A = 1/0.6
#
# giving the initial energies E_kin = <rho|u|^2>/2 = 1/2, E_mag = <|B|^2>/2 = 0.18 and
# E_int = <rho e> = 0.9 exactly.
#
# 2.5D: CMHD is dims=3 + z_spectral only (plans/CMHD_PLAN.md §1), so the 2D problem runs on
# a short z axis with a z-independent IC. Nothing in the equations then generates a kz != 0
# mode, and at nz = 4 the z-FFT twiddles are exactly +-1, +-i, so z-independence is
# preserved EXACTLY (measured: identically 0.0, not merely round-off). The test asserts it.
#
# ------------------------------------------------------------------- what this is NOT
#
# taranis is a dealiased pseudospectral code with no shock capturing and a POLYTROPIC
# closure (p = K rho^gamma, no energy equation), where Athena solves the adiabatic
# ideal-gas equations. For THIS IC the two coincide EXACTLY while the flow is smooth:
# rho and p are both uniform at t = 0, so the entropy is uniform, and smooth isentropic
# ideal-gas flow satisfies p = K rho^gamma with one global K -- which is the polytropic
# closure exactly. Nothing is approximated before the shocks. Once the OT vortex steepens
# into them (t ~ 0.13-0.15; Snow et al. 2021, Exp. Results 2, e35 §3, and Toth 2000, JCP
# 161, 605 §6.4 read through Stone's own t_2pi = pi <-> t = 1/2 mapping) they part ways
# for good, because shock heating raises entropy non-uniformly:
#
#   - an adiabatic code converts the shock's kinetic energy into heat and conserves total
#     energy; this one has no heat reservoir, so E = E_kin + E_mag + E_int DECLINES. That
#     decline is physics of the model, not a bug -- but it means the late-time energies are
#     NOT comparable to an adiabatic reference. Measured (256^2, inviscid, fixed dt): E is
#     conserved to 5.6e-9 relative through t = 0.12; in the run this script does (adaptive
#     dt, hyperdissipation) it has fallen 5.9% by t = 0.51.
#   - the shocks themselves are resolved as steep gradients absorbed by hyperdissipation,
#     not as jumps. Expect Gibbs ringing if you under-dissipate, and a NaN if you
#     under-dissipate badly enough to drive rho through zero (watch diagnostics.cmhd.rho_min,
#     which is exactly what it is for).
#
# So: quantitative comparison to a grid code belongs in the pre-shock smooth window
# (t <= 0.12 is the window tests/test_cmhd_orszag_tang.py gates in, and its header carries
# the literature the number comes from); after that this run is qualitative. To compare
# against raw Athena output, multiply its energies by 1/rho0 = 36 pi/25 = 4.5238934 -- and
# note that no PUBLISHED Orszag-Tang energy-vs-time trace exists in this normalization; the
# test file's test_energy_traces_against_a_reference documents the search and what would
# fill the gap.
import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

import taranis as jr
from taranis.diagnostics import cmhd as dcmhd
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

# the code-unit Alfven speed of the Athena OT IC (see the header derivation)
VA = 0.6
CS0 = 1.0
GAMMA = 5.0/3.0

# Hyperdissipation. k^4 (hyper=2) with diss sized so the damping rate at the 2/3 dealias
# cutoff k_c = (n/3) 2pi/L is a few thousand per unit time -- strong enough to absorb the
# post-shock cascade, negligible in the smooth phase. diss scales as k_c^-4, i.e. as
# (256/n)^4 off the reference below.
DISS_256 = 1.25e-8


def diss_for(n):
    return DISS_256*(256.0/n)**4


def orszag_tang_ic(x, y, z):
    """The Athena OT IC in taranis code units (header). z-independent by construction."""
    shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
    one = jnp.ones(shp)
    tp = 2*jnp.pi
    return jnp.stack([one,
                      -jnp.sin(tp*y)*one, jnp.sin(tp*x)*one, 0.0*one,
                      -VA*jnp.sin(tp*y)*one, VA*jnp.sin(2*tp*x)*one, 0.0*one])


def make_params(n=256, nz=4, hyper=2, cfl_safety=0.4):
    return jr.Parameters(nx=n, ny=n, nz=nz, Lx=1.0, Ly=1.0, Lz=1.0,
                         dims=3, z_spectral=True, eqtype="CMHD",
                         adaptive_timestep=True, cfl_safety=cfl_safety, dt=1e-3,
                         eqpars=dict(cs0=CS0, diss=diss_for(n), hyper=hyper, gamma=GAMMA))


def run(params, t_end=0.5, nblock=20, scheme="lsrk54", verbose=True):
    """Advance to t_end in blocks of nblock steps, recording diagnostics per block.
    Returns (final state, kgrid, trace dict of numpy arrays)."""
    kgrid = jr.setup_kgrids(params)
    state = jr.initialize(orszag_tang_ic, params)
    stepper, sch = get_scheme(scheme)
    advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))

    cols = ("t", "E_kin", "E_mag", "E_int", "E_tot", "M_s", "M_A", "rho_min", "divB")
    trace = {c: [] for c in cols}

    def record(s):
        ek, em, ei = dcmhd.energies(s, kgrid, params)
        ms, ma = dcmhd.mach_numbers(s, kgrid, params)
        row = (float(s.t), float(ek), float(em), float(ei), float(ek + em + ei),
               float(ms), float(ma), dcmhd.rho_min(s, kgrid, params),
               dcmhd.divB_max(s, kgrid, params))
        for c, v in zip(cols, row):
            trace[c].append(v)
        if verbose:
            print("  " + "  ".join(f"{c}={v:.6g}" for c, v in zip(cols, row)), flush=True)

    if verbose:
        print(f"Orszag-Tang: {params.nx}^2 x {params.nz}, gamma={GAMMA:.4f}, cs0={CS0}, "
              f"v_A={VA}, diss={params.eqpars['diss']:.3g} (hyper={params.eqpars['hyper']}), "
              f"to t={t_end}")
    record(state)
    t0 = time.perf_counter()
    nsteps = 0
    while float(state.t) < t_end:
        state = advance(state, kgrid, params, nblock, sch, stepper)
        nsteps += nblock
        record(state)
        if not np.isfinite(trace["E_tot"][-1]):
            raise RuntimeError("the run went non-finite: under-dissipated for this grid? "
                               "check the rho_min column above")
    wall = time.perf_counter() - t0
    if verbose:
        print(f"  {nsteps} steps in {wall:.1f}s ({1e3*wall/nsteps:.2f} ms/step)")
    return state, kgrid, {c: np.asarray(v) for c, v in trace.items()}


# --------------------------------------------------------------------------------- plots

def plot(state, kgrid, params, trace, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    t = trace["t"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    for k, lbl in (("E_kin", "$E_{kin}$"), ("E_mag", "$E_{mag}$"),
                   ("E_int", "$E_{int}$"), ("E_tot", "$E_{tot}$")):
        a.plot(t, trace[k], label=lbl)
    a.set_xlabel("t")
    a.set_ylabel("volume-averaged energy")
    a.set_title("energies (polytropic: $E_{tot}$ falls once shocks form)")
    a.legend()

    a = ax[0, 1]
    a.plot(t, trace["M_s"], label="$M_s$ (rms)")
    a.plot(t, trace["M_A"], label="$M_A$ (rms)")
    a.plot(t, trace["rho_min"], label=r"min $\rho$")
    a.set_xlabel("t")
    a.legend()
    a.set_title("Mach numbers and the rarefaction monitor")

    a = ax[1, 0]
    rho = np.asarray(jr.grids.ifft(state.fields[0], params))[0]
    im = a.imshow(rho.T, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    a.set_title(rf"$\rho$ at t = {float(state.t):.3f}")
    fig.colorbar(im, ax=a)

    a = ax[1, 1]
    kb, sk, sm, sd = dcmhd.spectra(state, kgrid, params, bin_factor=1.0)
    kb = np.asarray(kb)
    m = kb > 0
    for s, lbl in ((sk, "kinetic"), (sm, "magnetic"), (sd, "density")):
        a.loglog(kb[m], np.asarray(s)[m], label=lbl)
    a.set_xlabel(r"$k_\perp$")
    a.set_ylabel("E(k)")
    a.set_title(f"spectra at t = {float(state.t):.3f}")
    a.legend()

    fig.tight_layout()
    path = os.path.join(outdir, f"orszag_tang_{params.nx}.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")

    np.savez(os.path.join(outdir, f"orszag_tang_{params.nx}_trace.npz"), **trace)
    print(f"wrote {os.path.join(outdir, f'orszag_tang_{params.nx}_trace.npz')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=256, help="perpendicular resolution")
    ap.add_argument("--nz", type=int, default=4, help="z points (the 2.5D axis)")
    ap.add_argument("--t-end", type=float, default=0.5)
    ap.add_argument("--nblock", type=int, default=20)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "data", "cmhd-orszag-tang"))
    args = ap.parse_args()

    params = make_params(n=args.n, nz=args.nz)
    state, kgrid, trace = run(params, t_end=args.t_end, nblock=args.nblock)
    print(f"final: t={float(state.t):.4f}  E_tot={trace['E_tot'][-1]:.8f} "
          f"({100*(trace['E_tot'][-1]/trace['E_tot'][0] - 1):+.2f}% of initial)  "
          f"min rho={trace['rho_min'][-1]:.4f}  div B={trace['divB'][-1]:.2e}")
    if not args.no_plot:
        try:
            plot(state, kgrid, params, trace, args.outdir)
        except ImportError:
            print("matplotlib not installed; skipping plots "
                  "(pip install -e '.[examples]')")


if __name__ == "__main__":
    main()
