# z_derivatives (4th-order centered d/dz + 5-point d4/dz4 hyperdissipation stencil,
# shared_physics.z_derivatives) in isolation, plus the bitwise gates pinning it and
# rmhd.FDLinearTerm to the independently written-out padded-slab reference below.
# pytest: single-process (stub, periodic self-halo). Savio driver:
# `mpirun -n N python tests/test_z_stencils.py` for N in 1,2,4 (nz values below are
# all divisible by 4).
from _rmhd_testing import (bootstrap, checks, fit_order, fresh_params, make_state,
                           mpi_size, multimode_ic)

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import taranis as jr
from taranis import _precision, comms
from taranis.grids import local_z_coords
from taranis.physics import equation_registry
from taranis.physics.shared_physics import z_derivatives

# nz sweep for the convergence study: kept away from both ends -- small enough that
# a handful of points still resolves cos(2z) meaningfully, large enough (max nz=32,
# dz~0.2) to stay far above the fp32 noise floor (~1e-7) so the fitted order is not
# contaminated by round-off at either precision.
_NZS = (8, 16, 32)


# nz_local values swept by the bitwise gates: 2 and 3 are the narrowest slabs the
# width-2 halo admits, 8 and 16 the production range.
_NZ_LOCALS = (2, 3, 4, 8, 16)


def _cos2z_field(params):
    z = local_z_coords(params).reshape(1, -1, 1, 1)
    return jnp.cos(2 * z), z


# --------------------------------------------------------------- padded reference
# Private copy of the padded-slab stencil and of the FDLinearTerm built on it: what
# the shipped functions must reproduce bitwise.

def _padded_z_derivatives(f, params, halo=None):
    dz = params.dz
    recv_left, recv_right = comms.halo_exchange(f, params, width=2) if halo is None else halo
    w = recv_left.shape[1]
    nz = f.shape[1]
    f_padded = jnp.concatenate([recv_left, f, recv_right], axis=1)
    p2 = f_padded[:, w + 2:w + 2 + nz, :, :]
    p1 = f_padded[:, w + 1:w + 1 + nz, :, :]
    c = f_padded[:, w:w + nz, :, :]
    m1 = f_padded[:, w - 1:w - 1 + nz, :, :]
    m2 = f_padded[:, w - 2:w - 2 + nz, :, :]
    df_dz = (- p2 + 8 * p1 - 8 * m1 + m2) / (12 * dz)
    d4f_dz4 = (p2 - 4 * p1 + 6 * c - 4 * m1 + m2) / (dz**4)
    return df_dz, d4f_dz4


def _padded_fd_linear_term(state, grads, kgrid, params, halo=None):
    if params.spatial_dimensions == 2 or params.z_spectral:
        return jnp.zeros_like(state.fields)
    dz = params.dz
    diss = params.z_diss * (dz / 2)**4
    df_dz, d4f_dz4 = _padded_z_derivatives(state.fields, params, halo=halo)
    df_dz_rmhd = jnp.stack([df_dz[1], df_dz[0]])
    return df_dz_rmhd - diss * d4f_dz4


def _random_fields(params, seed=0):
    # broadband complex field: every stencil coefficient contributes at full magnitude
    nz_local = local_z_coords(params).size
    shape = (params.nfields, nz_local, params.nx, params.ny // 2 + 1)
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
                       dtype=_precision.ctype)


def _bitwise_pair(f, params, halo, c, label):
    old = _padded_z_derivatives(f, params, halo=halo)
    new = z_derivatives(f, params, halo=halo)
    for name, a, b in (("df/dz", new[0], old[0]), ("d4f/dz4", new[1], old[1])):
        c.check(f"{label}: {name} bitwise equal to the padded reference",
                bool(jnp.array_equal(a, b)),
                f"max|diff|={float(jnp.max(jnp.abs(a - b))):.3e}")


def test_z_derivatives_bitwise_matches_padded_reference():
    # nz_local sweep, both halo sources (internal exchange and a pre-issued width-2 halo)
    size = mpi_size()
    with checks() as c:
        for nz_local in _NZ_LOCALS:
            params = fresh_params(nz=nz_local * size)
            f = _random_fields(params, seed=nz_local)
            _bitwise_pair(f, params, None, c, f"nz_local={nz_local} internal halo")
            halo = comms.halo_exchange(f, params, width=2)
            _bitwise_pair(f, params, halo, c, f"nz_local={nz_local} pre-issued width 2")


def test_z_derivatives_bitwise_wide_halo_and_serial_backend():
    # a wider-than-needed pre-issued halo, and the serial backend's slice self-exchange
    size = mpi_size()
    with checks() as c:
        for nz_local in (4, 8, 16):
            params = fresh_params(nz=nz_local * size)
            f = _random_fields(params, seed=nz_local + 100)
            halo = comms.halo_exchange(f, params, width=3)
            _bitwise_pair(f, params, halo, c, f"nz_local={nz_local} pre-issued width 3")
            if size == 1:
                sp = fresh_params(nz=nz_local, comm_backend="serial")
                _bitwise_pair(f, sp, None, c, f"nz_local={nz_local} serial backend")


@pytest.mark.multidev
def test_z_derivatives_bitwise_under_four_way_z_sharding():
    # 4-way z decomposition through the real collective path (comm_backend="jax",
    # ppermute halo inside shard_map): the stencil must still be bitwise.
    from jax.sharding import PartitionSpec as P
    with checks() as c:
        for nz_local in (4, 8):
            nz = 4 * nz_local
            params = fresh_params(nz=nz, comm_backend="jax")
            f = _random_fields(fresh_params(nz=nz), seed=nz_local + 200)
            fg = comms.to_global(f, params, z_axis=1)
            spec = P(None, comms.Z_AXIS)

            def run(fn, fg=fg, params=params, spec=spec):
                return jax.jit(comms._shard_map(lambda x: fn(x, params),
                                                (spec,), (spec, spec)))(fg)

            old_d1, old_d4 = run(_padded_z_derivatives)
            new_d1, new_d4 = run(z_derivatives)
            c.check(f"nz_local={nz_local}: input really split over 4 devices",
                    len(fg.addressable_shards) == 4
                    and fg.addressable_shards[0].data.shape[1] == nz_local,
                    f"{len(fg.addressable_shards)} shards, "
                    f"{fg.addressable_shards[0].data.shape}")
            for name, a, b in (("df/dz", new_d1, old_d1), ("d4f/dz4", new_d4, old_d4)):
                c.check(f"nz_local={nz_local} over 4 devices: {name} bitwise equal",
                        bool(jnp.array_equal(a, b)),
                        f"max|diff|={float(jnp.max(jnp.abs(a - b))):.3e}")


def _fdz_end_fields(term_funcs, nsteps, backend):
    # 20 fixed-dt FD-z steps with the RMHD recipe's term_funcs replaced. Each call builds
    # its OWN Parameters: params is a static jit argument hashed by identity, so the
    # variants cannot share a traced executable (an in-process A/B that reuses one
    # Parameters silently reuses the first variant's compile).
    from taranis.run import block_of_steps, _refresh_forcing_scale
    from taranis.timestepping import get_scheme
    params = fresh_params(nz=8 * mpi_size(), comm_backend=backend)
    kgrid = jr.setup_kgrids(params)
    recipe = equation_registry["RMHD"]
    equation_registry["RMHD"] = recipe._replace(term_funcs=term_funcs)
    try:
        state = _refresh_forcing_scale(make_state(params, ic=multimode_ic), kgrid, params)
        stepper, scheme = get_scheme("lsrk33")
        end = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))(
            state, kgrid, params, nsteps, scheme, stepper)
        return np.asarray(end.fields)
    finally:
        equation_registry["RMHD"] = recipe


def test_fdz_solver_bitwise_over_20_steps():
    # end-to-end: 3D FD-z RMHD output equals the private padded reference's.
    from taranis.physics import rmhd
    shipped = equation_registry["RMHD"].term_funcs
    reference = tuple(_padded_fd_linear_term if t is rmhd.FDLinearTerm else t
                      for t in shipped)
    with checks() as c:
        for backend in (None, "serial"):
            if backend == "serial" and mpi_size() > 1:
                continue
            tag = f"backend={backend or 'default'}"
            old = _fdz_end_fields(reference, 20, backend)
            c.check(f"{tag}: reference end fields finite and nonzero",
                    bool(np.all(np.isfinite(old)) and np.max(np.abs(old)) > 0))
            new = _fdz_end_fields(shipped, 20, backend)
            ndiff = int(np.count_nonzero(new != old))
            c.check(f"{tag}: fields bitwise equal to the padded reference after 20 steps",
                    ndiff == 0,
                    f"{ndiff}/{new.size} differ, "
                    f"max|diff|={float(np.max(np.abs(new - old))):.3e}")


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
