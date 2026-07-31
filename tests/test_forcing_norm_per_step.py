# Coverage for the production default forcing_norm_per_step=True (per-step, lagged
# normalization) and the opt-in forcing_shell_noise=True path -- neither is exercised
# by test_forcing_smoke, which pins norm_per_step=False to check exact per-stage
# normalization. 2D, so single-process only (pytest, or `python tests/...` on Savio).
from _rmhd_testing import bootstrap, checks, ctx, make_state, managed_manager, snap_dir, zero_ic_2d

bootstrap()

import jax
import jax.numpy as jnp

import jax_rmhd as jr
from jax_rmhd import snapshot_io
from jax_rmhd.physics import shared_physics

_FORCING = dict(nx=32, ny=32, dims=2, diss=(0.0, 0.0), forcing=True,
                forcing_power=1.0, forcing_power_elsasser=(0.5, 0.5),
                forcing_tau=0.5, fshell=(1, 5), forcing_seed=1,
                forcing_norm_per_step=True)


def _fctx(mode, shell_noise=False):
    return ctx(forcing_mode=mode, forcing_shell_noise=shell_noise, **_FORCING)


def _injection_rate_check(mode):
    # At fixed dt=0.01 << tau the one-step normalization lag is small: loose 2x
    # bound on the injection rate.
    p, kg = _fctx(mode)
    st = make_state(p, ic=zero_ic_2d)
    with snap_dir() as d, managed_manager(p, d, nsnap=2) as mngr:
        end = jr.simulate(st, kg, p, t_snap=1.0, t_end=0.5, mngr=mngr, save=False)
        E_kin = 0.5 * float(shared_physics.perp_inner_product(end.fields[0], end.fields[0], kg, p))
        E_mag = 0.5 * float(shared_physics.perp_inner_product(end.fields[1], end.fields[1], kg, p))
        # both modes target the same TOTAL injection rate: forcing_power for momentum,
        # eps_p+eps_m for elsasser.
        target = 1.0
        rate = (E_kin + E_mag) / float(end.t)
        with checks() as c:
            c.check(f"norm_per_step {mode}: injection rate {rate:.3f} within 2x of {target}",
                    0.5 * target < rate < 2.0 * target)
            c.check(f"norm_per_step {mode}: forcing_scale finite, shape ({p.n_ou},)",
                    end.forcing_scale.shape == (p.n_ou,)
                    and bool(jnp.all(jnp.isfinite(end.forcing_scale))))


def test_norm_per_step_injection_rate_momentum():
    _injection_rate_check("momentum")


def test_norm_per_step_injection_rate_elsasser():
    _injection_rate_check("elsasser")


def test_shell_noise_symmetry_and_support():
    # Shell-restricted noise: hermitian symmetry on the ky=0 / Nyquist rows + support
    # exactly on fmask, elsasser mode.
    p, kg = _fctx("elsasser", shell_noise=True)
    _, complex_t = snapshot_io.get_precision_types()
    fs = jnp.zeros((p.n_ou, 2, p.nx, p.ny // 2 + 1), dtype=complex_t)
    key = jax.random.key(0)
    for _ in range(50):
        fs, key = shared_physics.ou_update(fs, key, 0.01, p, kg)
    mirror = (-jnp.arange(p.nx)) % p.nx
    tol = 1e-12 if jax.config.jax_enable_x64 else 1e-4
    with checks() as c:
        for ky_idx in (0, -1):
            col = fs[..., ky_idx]
            herm_err = float(jnp.max(jnp.abs(col - jnp.conj(col[..., mirror]))))
            c.check(f"shell noise: hermitian symmetry at ky index {ky_idx} "
                    f"after 50 ou_updates (err {herm_err:.1e})", herm_err < tol)
        off_shell = float(jnp.max(jnp.abs(fs * (~kg.fmask)[None, None, :, :])))
        c.check(f"shell noise: support exactly on fmask (off-shell max {off_shell:.1e})",
                off_shell == 0.0)
        c.check("shell noise: nonzero forcing state produced",
                float(jnp.max(jnp.abs(fs))) > 0.0)


def test_shell_noise_norm_per_step_end_to_end():
    # Shell noise + norm_per_step together: the production-style combo stays finite.
    p, kg = _fctx("elsasser", shell_noise=True)
    st = make_state(p, ic=zero_ic_2d)
    with snap_dir() as d, managed_manager(p, d, nsnap=2) as mngr:
        end = jr.simulate(st, kg, p, t_snap=1.0, t_end=0.5, mngr=mngr, save=False)
        assert bool(jnp.all(jnp.isfinite(end.fields)))


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
