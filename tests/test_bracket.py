# Poisson-bracket / NonlinearTerm correctness.
# pytest: single-process (stub). Savio driver: `python tests/test_bracket.py`
# (2D sections keep this single-process only; block_of_steps direct call keeps the
# ideal-run section to an exact step count regardless of adaptive dt).
from _rmhd_testing import bootstrap, checks, ctx, make_state

bootstrap()

import jax
import jax.numpy as jnp

from jax_rmhd import _precision, diagnostics, grids
from jax_rmhd.physics import rmhd, shared_physics
from jax_rmhd.physics.shared_physics import bracket, gradk
from jax_rmhd.run import block_of_steps
from jax_rmhd.timestepping import get_scheme

_IDEAL2D = dict(dims=2, diss=(0.0, 0.0))


def _low_modes_pair(x, y):
    # Two independent low-wavenumber modes (kx,ky up to 2), far below the 2/3
    # dealias cutoff at any of the resolutions used below -- no aliasing anywhere
    # in this module, so the only error source is floating-point round-off.
    f = jnp.cos(x) * jnp.cos(2 * y)
    g = jnp.sin(2 * x) * jnp.sin(y)
    return f, g


def _analytic_bracket_fg(x, y):
    # {f,g} = df/dx dg/dy - df/dy dg/dx, computed by hand for _low_modes_pair.
    dfdx = -jnp.sin(x) * jnp.cos(2 * y)
    dfdy = -2 * jnp.cos(x) * jnp.sin(2 * y)
    dgdx = 2 * jnp.cos(2 * x) * jnp.sin(y)
    dgdy = jnp.sin(2 * x) * jnp.cos(y)
    return dfdx * dgdy - dfdy * dgdx


def _perp_grad(fieldk, kgrid, params):
    # Real-space (d/dx, d/dy) of a k-space field, via the production gradk/ifft
    # path. fieldk carries whatever leading axes it already has (e.g. a z axis);
    # gradk's axis=1 insertion needs at least one axis ahead of (nx,nky), so add
    # (and then drop) a throwaway batch axis -- exactly what rmhd.grad relies on
    # for its 4-field batch, here used with a batch of 1.
    return grids.ifft(gradk(fieldk[None], kgrid), params)[0]


def _xy_grid(params):
    x = jnp.linspace(0, params.Lx, params.nx, endpoint=False).reshape(1, -1, 1)
    y = jnp.linspace(0, params.Ly, params.ny, endpoint=False).reshape(1, 1, -1)
    return x, y


def _ic2d(x, y):
    # Non-degenerate multi-mode IC (2D analogue of _rmhd_testing.multimode_ic).
    phi = jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y)
    psi = jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([phi, psi], axis=0)


def _ic2d_correlated(x, y):
    # Like _ic2d, but phi and psi share a cos(x)cos(y) component so their overlap
    # at that mode is between two *real* Fourier coefficients (nonzero product)
    # rather than a real and a purely-imaginary one. _ic2d's phi/psi (built from
    # cos*cos vs sin*cos factors at the same wavenumbers) have that mode's
    # coefficient purely real in phi and purely imaginary in psi -- their overlap
    # is Re(real * imaginary) = 0 to machine precision, so cross-helicity is ~1e-17
    # (round-off, not a physical zero) and unusable as a relative-error denominator.
    phi = jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y)
    psi = 0.4 * jnp.cos(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([phi, psi], axis=0)


def test_bracket_matches_analytic_solution():
    params, kgrid = ctx(dims=2, nx=32, ny=32)
    x, y = _xy_grid(params)
    f, g = _low_modes_pair(x, y)
    fk, gk = grids.fft(f, params), grids.fft(g, params)
    gradf = _perp_grad(fk, kgrid, params)
    gradg = _perp_grad(gk, kgrid, params)
    computed = bracket(gradf, gradg)
    analytic = _analytic_bracket_fg(x, y)
    scale = float(jnp.max(jnp.abs(analytic)))
    rel_err = float(jnp.max(jnp.abs(computed - analytic))) / scale
    tol = 1e-12 if _precision.precision == "64" else 2e-5
    with checks() as c:
        c.check("bracket matches hand-derived analytic solution (modes well below cutoff)",
                rel_err < tol, f"rel_err={rel_err:.3e}, tol={tol:.1e}")


def test_bracket_antisymmetry_and_self_zero():
    # Pure algebraic identities of bracket(a,b)=a0*b1-a1*b0: IEEE-754 multiplication
    # is bitwise commutative and x-x is exactly 0 for any finite x, and negation is
    # exact, so these hold bitwise at any precision -- no ctx/grid machinery needed.
    key = jax.random.PRNGKey(0)
    ka, kb = jax.random.split(key)
    shape = (2, 3, 5, 7)
    a = jax.random.normal(ka, shape)
    b = jax.random.normal(kb, shape)
    ab = bracket(a, b)
    ba = bracket(b, a)
    aa = bracket(a, a)
    with checks() as c:
        c.check("bracket(a,b) == -bracket(b,a) bitwise", bool(jnp.array_equal(ab, -ba)))
        c.check("bracket(a,a) == 0 bitwise", bool(jnp.array_equal(aa, jnp.zeros_like(aa))))


def test_bracket_conservation_identity():
    # sum_grid f*{f,g} == 0: the discrete analogue of integral f{f,g} dA = 0
    # (integration by parts / antisymmetry of the Jacobian), exact in the continuum
    # for periodic domains. Holds to near machine precision discretely as long as
    # f*{f,g} stays free of aliasing -- true here since f,g only involve mode
    # numbers <=2, far below the grid's Nyquist/3 dealias cutoff.
    params, kgrid = ctx(dims=2, nx=32, ny=32)
    x, y = _xy_grid(params)
    f, g = _low_modes_pair(x, y)
    fk, gk = grids.fft(f, params), grids.fft(g, params)
    gradf = _perp_grad(fk, kgrid, params)
    gradg = _perp_grad(gk, kgrid, params)
    fg_bracket = bracket(gradf, gradg)
    integrand = f * fg_bracket
    residual = float(jnp.abs(jnp.sum(integrand)))
    scale = float(jnp.sum(jnp.abs(integrand)))
    rel = residual / scale
    tol = 1e-10 if _precision.precision == "64" else 1e-4
    with checks() as c:
        c.check("sum(f * {f,g}) == 0 (conservation identity)", rel < tol,
                f"rel={rel:.3e}, tol={tol:.1e}")


def test_nonlinear_term_zero_outside_dealias_mask():
    params, kgrid = ctx(dims=2, diss=(0.0, 0.0))
    state = make_state(params, ic=_ic2d)
    grads = rmhd.grad(state, kgrid, params)
    nlterm = rmhd.NonlinearTerm(state, grads, kgrid, params)
    mask = kgrid.dealias  # (nx, nky), True = kept
    # every entry is either inside the mask, or (if outside) exactly zero
    inside_or_zero = mask[None, None, :, :] | (nlterm == 0)
    with checks() as c:
        c.check("NonlinearTerm output is exactly zero everywhere outside the dealias mask",
                bool(jnp.all(inside_or_zero)))
        c.check("NonlinearTerm has some nonzero output inside the mask (sanity)",
                bool(jnp.any(nlterm != 0)))


def test_ideal_run_conserves_energy_and_cross_helicity():
    # 20-step unforced, dissipation-free 2D run: total energy and cross-helicity
    # (<grad phi . grad psi>) are ideal invariants of RMHD. block_of_steps is called
    # directly (rather than simulate/simulate_scan) to get exactly 20 steps
    # regardless of the adaptive dt -- simulate*'s while-loop-to-a-target-time can't
    # guarantee an exact step count.
    params, kgrid = ctx(**_IDEAL2D)
    state0 = make_state(params, ic=_ic2d_correlated)
    E0 = sum(float(e) for e in diagnostics.energy(state0, kgrid, params))
    H0 = float(shared_physics.perp_inner_product(state0.fields[0], state0.fields[1], kgrid, params))
    stepper, scheme = get_scheme("lsrk33")
    final_state = block_of_steps(state0, kgrid, params, 20, scheme, stepper)
    with checks() as c:
        c.check("run advanced (t>0) and stayed finite",
                float(final_state.t) > 0.0 and bool(jnp.all(jnp.isfinite(final_state.fields))))
    E1 = sum(float(e) for e in diagnostics.energy(final_state, kgrid, params))
    H1 = float(shared_physics.perp_inner_product(final_state.fields[0], final_state.fields[1], kgrid, params))
    tol = 1e-6 if _precision.precision == "64" else 1e-3
    with checks() as c:
        c.check("ideal 20-step run conserves total energy",
                abs(E1 - E0) / abs(E0) < tol, f"E0={E0:.10e}, E1={E1:.10e}")
        c.check("ideal 20-step run conserves cross-helicity",
                abs(H1 - H0) / abs(H0) < tol, f"H0={H0:.10e}, H1={H1:.10e}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
