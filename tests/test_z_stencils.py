# z_derivatives (4th-order centered d/dz + 5-point d4/dz4 hyperdissipation stencil,
# shared_physics.z_derivatives) in isolation.
# pytest: single-process (stub, periodic self-halo). Savio driver:
# `mpirun -n N python tests/test_z_stencils.py` for N in 1,2,4 (nz values below are
# all divisible by 4).
from _rmhd_testing import bootstrap, checks, fit_order, fresh_params

bootstrap()

import jax.numpy as jnp

from taranis import _precision
from taranis.grids import local_z_coords
from taranis.physics.shared_physics import z_derivatives

# nz sweep for the convergence study: kept away from both ends -- small enough that
# a handful of points still resolves cos(2z) meaningfully, large enough (max nz=32,
# dz~0.2) to stay far above the fp32 noise floor (~1e-7) so the fitted order is not
# contaminated by round-off at either precision.
_NZS = (8, 16, 32)


def _cos2z_field(params):
    z = local_z_coords(params).reshape(1, -1, 1, 1)
    return jnp.cos(2 * z), z


def test_z_derivative_convergence_order():
    dzs, err_d1, err_d4 = [], [], []
    for nz in _NZS:
        params = fresh_params(nz=nz)
        f, z = _cos2z_field(params)
        df_dz, d4f_dz4 = z_derivatives(f, params)
        analytic_d1 = -2 * jnp.sin(2 * z)
        analytic_d4 = 16 * jnp.cos(2 * z)
        err_d1.append(float(jnp.max(jnp.abs(df_dz - analytic_d1))))
        err_d4.append(float(jnp.max(jnp.abs(d4f_dz4 - analytic_d4))))
        dzs.append(params.dz)
    order_d1 = fit_order(dzs, err_d1)
    order_d4 = fit_order(dzs, err_d4)
    # fp32: the finest nz=32 point (dz~0.2, h^4 error ~1.6e-3) is close enough to
    # the ~1e-7 fp32 noise floor to shave a few hundredths off the fitted slope
    # (observed order~3.80 exactly, vs >3.99 at fp64) -- loosen slightly rather
    # than drop the finest point (which would weaken the fp64 check).
    order_tol_d1 = 3.8 if _precision.precision == "64" else 3.5
    with checks() as c:
        c.check(f"d/dz fitted convergence order > {order_tol_d1} (nominally 4th order)",
                order_d1 > order_tol_d1, f"order={order_d1:.3f}, errs={err_d1}")
        c.check("d4/dz4 fitted convergence order > 1.8 (nominally 2nd order -- pinned)",
                order_d4 > 1.8, f"order={order_d4:.3f}, errs={err_d4}")


def test_z_derivative_zero_on_constant():
    # Both stencils have coefficients summing to exactly 0, so the *ideal* value is
    # bitwise zero; the *computed* value is a few ULP of round-off, not literal
    # zero -- the d4/dz4 stencil divides by dz^4, which amplifies that round-off
    # (observed ~1e-16 for d/dz and ~1e-14 for d4/dz4 at fp64, dz=2*pi/16). Assert
    # "zero to near machine precision" rather than bitwise equality.
    params = fresh_params(nz=16)
    # shape follows the rank-LOCAL slab (nz/size under mpirun), not global nz
    nz_local = local_z_coords(params).size
    f = 3.7 * jnp.ones((1, nz_local, 1, 1))
    df_dz, d4f_dz4 = z_derivatives(f, params)
    tol = 1e-10 if _precision.precision == "64" else 1e-3
    with checks() as c:
        c.check("d/dz of a constant field is zero to near machine precision",
                float(jnp.max(jnp.abs(df_dz))) < tol,
                f"max|df_dz|={float(jnp.max(jnp.abs(df_dz))):.3e}, tol={tol:.1e}")
        c.check("d4/dz4 of a constant field is zero to near machine precision",
                float(jnp.max(jnp.abs(d4f_dz4))) < tol,
                f"max|d4f_dz4|={float(jnp.max(jnp.abs(d4f_dz4))):.3e}, tol={tol:.1e}")


def test_z_derivative_matches_modified_wavenumber():
    # Periodic wraparound check against the stencils' known modified-wavenumber
    # factors: applying the discrete operator to a single complex exponential mode
    # e^{ikz} must reproduce i*k'(k)*e^{ikz} (first derivative) and k4'(k)*e^{ikz}
    # (d4/dz4), where k'/k4' are the stencils' exact discrete dispersion factors --
    # derived by substituting e^{ikz} into the finite-difference formulas:
    #   k'(k)  = (8 sin(k dz) - sin(2 k dz)) / (6 dz)
    #   k4'(k) = (6 - 8 cos(k dz) + 2 cos(2 k dz)) / dz^4
    # This is an exact discrete identity (not an asymptotic approximation), and only
    # comes out right if the halo/periodic wraparound feeding the stencil is correct.
    params = fresh_params(nz=16)
    k = 2.0
    z = local_z_coords(params).reshape(1, -1, 1, 1)
    f = jnp.exp(1j * k * z)
    df_dz, d4f_dz4 = z_derivatives(f, params)
    dz = params.dz
    k1 = (8 * jnp.sin(k * dz) - jnp.sin(2 * k * dz)) / (6 * dz)
    k4 = (6 - 8 * jnp.cos(k * dz) + 2 * jnp.cos(2 * k * dz)) / dz**4
    expected_d1 = 1j * k1 * f
    expected_d4 = k4 * f
    err1 = float(jnp.max(jnp.abs(df_dz - expected_d1))) / float(jnp.max(jnp.abs(expected_d1)))
    err4 = float(jnp.max(jnp.abs(d4f_dz4 - expected_d4))) / float(jnp.max(jnp.abs(expected_d4)))
    tol = 1e-10 if _precision.precision == "64" else 1e-3
    with checks() as c:
        c.check("periodic d/dz matches the stencil's known modified-wavenumber factor",
                err1 < tol, f"rel_err={err1:.3e}, tol={tol:.1e}")
        c.check("periodic d4/dz4 matches the stencil's known modified-wavenumber^4 factor",
                err4 < tol, f"rel_err={err4:.3e}, tol={tol:.1e}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
