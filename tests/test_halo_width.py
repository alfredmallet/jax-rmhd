# Checks for the parameterized-width z-halo exchange (comms.halo_exchange).
# pytest: runs single-process (stub). Savio driver: `mpirun -n N python
# tests/test_halo_width.py` for N in 1, 2, 4. Keep it cheap: small grid, few steps.
from _rmhd_testing import bootstrap, checks, ctx, make_state, managed_manager, snap_dir

bootstrap()

import jax.numpy as jnp
import pytest

import jax_rmhd as jr
from jax_rmhd import comms
from jax_rmhd.physics import rmhd, shared_physics

# nz=16 keeps nz_local >= 3 (the width=3 case below) up to mpirun -n 4.
_CTX = dict(nz=16)


def _ic(x, y, z):
    # Non-degenerate in all three directions (real z-structure exercises the halo).
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
    psi = jnp.sin(x) * jnp.cos(y) * jnp.sin(z)
    return jnp.stack([phi, psi], axis=0)


def test_z_derivatives_halo_width_invariance():
    # z_derivatives is bitwise-identical whether fed a pre-issued width-2 halo, a
    # pre-issued (wider-than-needed) width-3 halo, or no halo at all (internal
    # fallback, which itself requests width=2): extra halo planes must be sliced
    # away exactly, not just "close".
    params, _ = ctx(**_CTX)
    f = make_state(params, ic=_ic).fields
    halo2 = comms.halo_exchange(f, params, width=2)
    halo3 = comms.halo_exchange(f, params, width=3)
    df_2, d4f_2 = shared_physics.z_derivatives(f, params, halo=halo2)
    df_3, d4f_3 = shared_physics.z_derivatives(f, params, halo=halo3)
    df_n, d4f_n = shared_physics.z_derivatives(f, params, halo=None)
    with checks() as c:
        c.check("df/dz bitwise-equal: width=2 vs width=3 halo",
                bool(jnp.array_equal(df_2, df_3)))
        c.check("d4f/dz4 bitwise-equal: width=2 vs width=3 halo",
                bool(jnp.array_equal(d4f_2, d4f_3)))
        c.check("df/dz bitwise-equal: width=2 vs no-halo fallback",
                bool(jnp.array_equal(df_2, df_n)))
        c.check("d4f/dz4 bitwise-equal: width=2 vs no-halo fallback",
                bool(jnp.array_equal(d4f_2, d4f_n)))


def test_halo_start_matches_bare_width2_exchange():
    # rmhd.halo_start (width=2) feeds z_derivatives identically to a bare width=2
    # halo_exchange call -- the coupling the halo-width plan calls out explicitly.
    params, kgrid = ctx(**_CTX)
    state = make_state(params, ic=_ic)
    f = state.fields
    halo2 = comms.halo_exchange(f, params, width=2)
    df_2, d4f_2 = shared_physics.z_derivatives(f, params, halo=halo2)
    df_hs, d4f_hs = shared_physics.z_derivatives(
        f, params, halo=rmhd.halo_start(state, kgrid, params))
    assert jnp.array_equal(df_hs, df_2) and jnp.array_equal(d4f_hs, d4f_2)


def test_halo_exchange_width_bounds_raise():
    # width > nz_local (or width < 1) raises ValueError, not a silently-wrong
    # short exchange.
    params, _ = ctx(**_CTX)
    f = make_state(params, ic=_ic).fields
    nz_local = f.shape[1]
    with pytest.raises(ValueError):
        comms.halo_exchange(f, params, width=nz_local + 1)
    with pytest.raises(ValueError):
        comms.halo_exchange(f, params, width=0)


def test_default_path_short_run_finite():
    # Default path (width=2 everywhere, unchanged): a few steps stay finite with the
    # right shape. simulate_scan donates its input state -- fresh state, shape read
    # before the call.
    params, kgrid = ctx(**_CTX)
    state0 = make_state(params, ic=_ic)
    fields_shape0 = state0.fields.shape
    with snap_dir() as d, managed_manager(params, d, nsnap=5) as mngr:
        end_state = jr.simulate_scan(state0, kgrid, params, 20, 0.05, 0.05, mngr,
                                     save=False)
        with checks() as c:
            c.check("output fields shape unchanged",
                    end_state.fields.shape == fields_shape0,
                    f"{end_state.fields.shape} vs {fields_shape0}")
            c.check("output fields finite after a few steps",
                    bool(jnp.all(jnp.isfinite(end_state.fields))))


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
