# diagnostics.perpspec / parspec / energy consistency (docs/TESTING_PLAN.md
# Phase 5). All energy-like reductions in the code share ONE normalization
# convention (rfft2 ky-doubling, / nz*(nx*ny)^2 -- CLAUDE.md): the integral of
# each spectrum must therefore equal diagnostics.energy EXACTLY (round-off only).
# This guards the convention itself: a diagnostic that silently changes
# normalization stops summing to the energy.
#
# Binning note: with nx==ny every dealias-surviving mode has |k_perp| < (nx/3)*
# kunit (the elliptical 2/3 mask is a circle), safely inside perpspec's histogram
# range [0, kmax+dk) -- nothing falls off the end, which is what makes the
# identity exact. Same for parspec: |kz| <= nz/2 lands in the (closed) last bin.
#
# perpspec is mpirun-safe (allreduce inside); parspec is size==1-only BY DESIGN
# and its tests soft-skip under a real multi-rank launch.
# Savio: `mpirun -n 4 python tests/test_diagnostics.py` (perpspec part).
from _rmhd_testing import (bootstrap, checks, ctx, fake_ranked_params, make_state,
                           mpi_size, multimode_ic)

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np

from jax_rmhd import diagnostics

_TOL = None  # set per-test: 1e-12 fp64 / 1e-5 fp32 (fft round-off floor)


def _tol():
    return 1e-12 if jax.config.jax_enable_x64 else 1e-5


def test_perpspec_sums_to_energy():
    params, kgrid = ctx()
    state = make_state(params, ic=multimode_ic)
    bins, spec_u, spec_b = diagnostics.perpspec(state, kgrid, params)
    dk = float(bins[1] - bins[0])
    E_kin, E_mag = diagnostics.energy(state, kgrid, params)
    int_u, int_b = float(jnp.sum(spec_u)) * dk, float(jnp.sum(spec_b)) * dk
    rel_u = abs(int_u - float(E_kin)) / float(E_kin)
    rel_b = abs(int_b - float(E_mag)) / float(E_mag)
    with checks() as c:
        c.check(f"sum(perpspec_u)*dk == E_kin (rel {rel_u:.2e} < {_tol():.0e})",
                rel_u < _tol())
        c.check(f"sum(perpspec_b)*dk == E_mag (rel {rel_b:.2e} < {_tol():.0e})",
                rel_b < _tol())
        c.check("energies are nonzero (test not vacuous)",
                float(E_kin) > 0.0 and float(E_mag) > 0.0)


def test_perpspec_single_mode_lands_in_one_bin():
    params, kgrid = ctx(dims=2)

    def ic(x, y):
        phi = jnp.cos(4.0 * x)  # |k_perp| = 4 exactly (kx=+-4, ky=0)
        return jnp.stack([phi, jnp.zeros_like(phi)], axis=0)

    state = make_state(params, ic=ic)
    bins, spec_u, spec_b = diagnostics.perpspec(state, kgrid, params)
    dk = float(bins[1] - bins[0])  # default bin_factor=2 -> dk=2, edges 0,2,4,...
    spec_u = np.asarray(spec_u)
    ibin = int(np.argmax(spec_u))
    # FFT round-off leaves ~1e-16 amplitudes in other modes -> compare bin
    # contents relative to the peak rather than asserting exact zeros.
    off = float(np.sum(np.delete(spec_u, ibin))) / float(spec_u[ibin])
    E_kin = float(diagnostics.energy(state, kgrid, params)[0])
    rel_E = abs(float(spec_u[ibin]) * dk - E_kin) / E_kin
    with checks() as c:
        left = float(bins[ibin]) - dk / 2  # bins holds CENTERS; edges are center -+ dk/2
        c.check(f"k=4 lands in the bin [{left:g},{left + dk:g}) "
                f"(center {float(bins[ibin]):g})", left <= 4.0 < left + dk)
        c.check(f"all other bins are empty to round-off (off/peak {off:.2e})",
                off < _tol())
        c.check(f"the single bin carries the whole kinetic energy "
                f"(rel {rel_E:.2e})", rel_E < _tol())
        c.check("magnetic spectrum is empty to round-off",
                float(np.max(np.asarray(spec_b))) < _tol() * float(spec_u[ibin]))


def test_parspec_requires_single_process():
    # parspec needs the WHOLE z-domain on this rank; params.size>1 must be refused
    # loudly (spoofed rank/size -- never handed to jitted physics).
    p_fake = fake_ranked_params(0, 2)
    try:
        diagnostics.parspec(None, None, p_fake)
        raised = False
    except AssertionError:
        raised = True
    with checks() as c:
        c.check("parspec(params.size==2) raises AssertionError", raised)


def test_parspec_parseval():
    if mpi_size() > 1:
        print("[SKIP] test_parspec_parseval -- parspec is size==1-only by design")
        return
    params, kgrid = ctx()
    state = make_state(params, ic=multimode_ic)  # genuine kz structure
    bins, spec_u, spec_b = diagnostics.parspec(state, kgrid, params)
    dk = float(bins[1] - bins[0])
    E_kin, E_mag = diagnostics.energy(state, kgrid, params)
    rel_u = abs(float(jnp.sum(spec_u)) * dk - float(E_kin)) / float(E_kin)
    rel_b = abs(float(jnp.sum(spec_b)) * dk - float(E_mag)) / float(E_mag)
    with checks() as c:
        c.check(f"sum(parspec_u)*dk == E_kin (rel {rel_u:.2e} < {_tol():.0e})",
                rel_u < _tol())
        c.check(f"sum(parspec_b)*dk == E_mag (rel {rel_b:.2e} < {_tol():.0e})",
                rel_b < _tol())


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
