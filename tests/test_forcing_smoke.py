# Smoke tests for the O-U forcing implementation. forcing_norm_per_step=False
# throughout: these tests check the exact per-stage normalization (the per-step
# production default is covered by test_forcing_norm_per_step.py).
# pytest: single-process (stub). Savio driver: `python tests/test_forcing_smoke.py`
# (the dims=2 sections keep this single-process only).
from _rmhd_testing import bootstrap, checks, ctx, make_state, managed_manager, snap_dir, zero_ic, zero_ic_2d

bootstrap()

import jax
import jax.numpy as jnp

import jax_rmhd as jr
from jax_rmhd import _precision, snapshot_io
from jax_rmhd.physics import rmhd, shared_physics

_F = dict(diss=(0.0, 0.0), forcing=True, forcing_mode="momentum", forcing_power=1.0,
          forcing_tau=0.5, fshell=(1, 5), forcing_seed=1, forcing_norm_per_step=False)


def test_ou_update_preserves_hermitian_symmetry():
    params, kgrid = ctx(**_F)
    _, complex_t = snapshot_io.get_precision_types()
    forcing_state = jnp.zeros((params.n_ou, 2, params.nx, params.ny // 2 + 1),
                              dtype=complex_t)
    forcing_key = jax.random.key(params.forcing_seed)
    for _ in range(50):
        forcing_state, forcing_key = shared_physics.ou_update(
            forcing_state, forcing_key, 0.01, params, kgrid)
    mirror_idx = (-jnp.arange(forcing_state.shape[-2])) % forcing_state.shape[-2]
    tol = 1e-10 if _precision.precision == "64" else 1e-4
    with checks() as c:
        for ky_idx in (0, -1):
            col = forcing_state[..., ky_idx]
            err = float(jnp.max(jnp.abs(col - jnp.conj(col[..., mirror_idx]))))
            c.check(f"Hermitian symmetry holds at ky index {ky_idx} after 50 ou_update steps",
                    err < tol, f"max |A(kx)-conj(A(-kx))| = {err:.2e}")


def test_safe_scale_uncapped_and_capped():
    # Sign-correct where unclipped, capped at +-scale_max otherwise (the cap is on
    # the scale factor, never a floor on the denominator P).
    target = 2.0
    scale_max = 1.0  # default
    tol = 1e-8 if _precision.precision == "64" else 1e-6
    with checks() as c:
        for P_val in (3.0, -3.0):
            # |target/P| < scale_max: the cap must not engage -- exact match.
            scale = shared_physics.safe_scale(target, jnp.array(P_val), scale_max)
            result = float(scale) * P_val
            c.check(f"safe_scale: scale*P == target for P={P_val:+.0e} (uncapped regime)",
                    abs(result - target) < tol, f"got {result}")
        for P_val in (1e-40, -1e-40, 0.0):
            # |target/P| >> scale_max (or exactly target/0): scale lands exactly at
            # +-scale_max with the sign of target/P, not a blow-up.
            scale = float(shared_physics.safe_scale(target, jnp.array(P_val), scale_max))
            expected_sign = 1.0 if P_val >= 0.0 else -1.0
            c.check(f"safe_scale: near-zero P={P_val:+.0e} is capped at +-scale_max, not blown up",
                    jnp.isfinite(scale) and abs(abs(scale) - scale_max) < 1e-12
                    and jnp.sign(scale) == expected_sign, f"scale={scale}")


def test_forcing_term_exact_noop_when_off():
    # NB: kgrid must belong to params_off -- never reuse a kgrid built for a
    # different Parameters (the old script's bug).
    params_off, kgrid_off = ctx(diss=(0.0, 0.0), forcing=False)
    state_off = make_state(params_off, ic=zero_ic)
    grads_off = rmhd.grad(state_off, kgrid_off, params_off)
    f_off = rmhd.ForcingTerm(state_off, grads_off, kgrid_off, params_off)
    assert bool(jnp.all(f_off == 0))


def test_power_injection_3d():
    # Momentum mode, no dissipation, exact per-stage normalization: kinetic energy
    # injection rate should be near forcing_power (loose 3x bound).
    params, kgrid = ctx(**_F)
    state0 = make_state(params, ic=zero_ic)
    with snap_dir() as d, managed_manager(params, d, nsnap=10) as mngr:
        end_state = jr.simulate_scan(state0, kgrid, params, 50, 0.5, 0.5, mngr,
                                     save=False)
        phik = end_state.fields[0]
        E_kin = 0.5 * float(shared_physics.perp_inner_product(phik, phik, kgrid, params))
        rate = E_kin / float(end_state.t)
        target = params.forcing_power
        with checks() as c:
            c.check("kinetic energy injection rate within 3x of forcing_power",
                    target / 3.0 < rate < target * 3.0,
                    f"measured rate={rate:.4f}, target={target}, t_end={float(end_state.t)}")


def test_forcing_shapes_2d():
    params_2d, kgrid_2d = ctx(dims=2, **_F)
    state0_2d = make_state(params_2d, ic=zero_ic_2d)
    with checks() as c:
        c.check("2D fields have a singleton leading z axis (nfields,1,nx,nky)",
                state0_2d.fields.shape == (2, 1, params_2d.nx, params_2d.ny // 2 + 1),
                f"fields.shape={state0_2d.fields.shape}")
        grads_2d = rmhd.grad(state0_2d, kgrid_2d, params_2d)
        f_2d = rmhd.ForcingTerm(state0_2d, grads_2d, kgrid_2d, params_2d)
        c.check("2D ForcingTerm output shape matches state.fields",
                f_2d.shape == state0_2d.fields.shape,
                f"f_2d.shape={f_2d.shape}, fields.shape={state0_2d.fields.shape}")


def test_power_injection_2d():
    params_2d, kgrid_2d = ctx(dims=2, **_F)
    state0_2d = make_state(params_2d, ic=zero_ic_2d)
    with snap_dir() as d, managed_manager(params_2d, d, nsnap=10) as mngr:
        end_state = jr.simulate_scan(state0_2d, kgrid_2d, params_2d, 50, 0.5, 0.5,
                                     mngr, save=False)
        phik = end_state.fields[0]
        E_kin = 0.5 * float(shared_physics.perp_inner_product(phik, phik, kgrid_2d, params_2d))
        rate = E_kin / float(end_state.t)
        target = params_2d.forcing_power
        with checks() as c:
            c.check("2D kinetic energy injection rate within 3x of forcing_power",
                    target / 3.0 < rate < target * 3.0,
                    f"measured rate={rate:.4f}, target={target}, t_end={float(end_state.t)}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
