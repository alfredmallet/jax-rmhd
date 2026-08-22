# Coverage for the production default forcing_norm_per_step=True (per-step, lagged
# normalization) and the opt-in forcing_shell_noise=True path -- neither is exercised
# by test_forcing_smoke, which pins norm_per_step=False to check exact per-stage
# normalization. Also the restart behaviour the lagged scale implies: entry to
# simulate/simulate_scan computes a scale only for a state that carries none, so a
# checkpoint's stored scale survives and a forced restart continues the uninterrupted run
# bitwise. 2D, so single-process only (pytest, or `python tests/...` on Savio).
from _rmhd_testing import bootstrap, checks, ctx, make_state, managed_manager, snap_dir, zero_ic_2d

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np

import taranis as jr
from taranis import _precision, run, snapshot_io
from taranis.physics import equation_registry, shared_physics
from taranis.timestepping import get_scheme

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
    tol = 1e-12 if _precision.precision == "64" else 1e-4
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


# Restart coverage at the default forcing_norm_per_step=True: fixed dt, so a five-step
# block advances 0.05 -- more than T_SNAP, so every block snapshots and the restart lands
# on a step boundary of the uninterrupted run.
_RESTART_NBLOCK = 5
_RESTART_T_SNAP = 0.04
_RESTART_T_END = 0.2

_advance = jax.jit(run.block_of_steps, static_argnums=(2, 3, 4, 5))


def _restart_ic(x, y):
    return jnp.stack([jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y),
                      jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)], axis=0)


def _host_state(state):
    # host copies of everything a restart has to reproduce (the state itself is donated)
    return {"fields": np.asarray(state.fields),
            "forcing_state": np.asarray(state.forcing_state),
            "forcing_scale": np.asarray(state.forcing_scale),
            "forcing_key": np.asarray(jax.random.key_data(state.forcing_key)),
            "t": float(state.t)}


def _run_to_end(state, kg, p, path, t_end=_RESTART_T_END):
    with managed_manager(p, path, nsnap=10) as mngr:
        return _host_state(jr.simulate_scan(state, kg, p, _RESTART_NBLOCK,
                                            _RESTART_T_SNAP, t_end, mngr, save=True))


def test_restart_is_bitwise_at_default_norm():
    """A restart from a mid-run snapshot continues the uninterrupted forced run bitwise at
    the DEFAULT forcing_norm_per_step=True: the checkpoint's stored forcing_scale is the
    one that run would have used next, and entry keeps it."""
    p, kg = _fctx("elsasser")
    with snap_dir("fnorm_ref_") as d1:
        ref = _run_to_end(make_state(p, ic=_restart_ic), kg, p, d1)
        steps = snapshot_io.get_saved_steps(d1)
        mid = steps[len(steps) // 2]
        state_mid = snapshot_io.load_snapshot(mid, d1, p)
        t_mid = float(state_mid.t)
        mid_scale = np.asarray(state_mid.forcing_scale)
        with snap_dir("fnorm_restart_") as d2:
            got = _run_to_end(state_mid, kg, p, d2)

    print(f"restart from snapshot {mid} at t = {t_mid} of {ref['t']}, "
          f"forcing_norm_per_step={p.forcing_norm_per_step}")
    with checks() as c:
        c.check("the config really is at the default forcing_norm_per_step=True",
                p.forcing_norm_per_step is True)
        c.check("a mid-run snapshot really was mid-run (not the final state)",
                0.0 < t_mid < ref["t"], f"t_mid={t_mid}, t_end={ref['t']}")
        c.check("the snapshot carries a nonzero forcing_scale, so entry has something to "
                "preserve", bool(np.any(mid_scale != 0)), str(mid_scale))
        c.check(f"the restart lands on the same final time ({got['t']!r} vs {ref['t']!r})",
                got["t"] == ref["t"])
        for key in ("fields", "forcing_state", "forcing_scale", "forcing_key"):
            c.check(f"restarted {key} is bitwise identical",
                    np.array_equal(got[key], ref[key]))


def test_second_call_continues_the_returned_state_bitwise():
    """simulate_scan called again on a state it returned continues the run bitwise -- the
    same rule seen without a snapshot round trip."""
    p, kg = _fctx("elsasser")
    with snap_dir("fnorm_one_") as d1:
        ref = _run_to_end(make_state(p, ic=_restart_ic), kg, p, d1)
    with snap_dir("fnorm_two_a_") as d2, managed_manager(p, d2, nsnap=10) as m2:
        mid = jr.simulate_scan(make_state(p, ic=_restart_ic), kg, p, _RESTART_NBLOCK,
                               _RESTART_T_SNAP, 0.5 * _RESTART_T_END, m2, save=True)
        t_mid = float(mid.t)
        mid_scale = np.asarray(mid.forcing_scale)
        with snap_dir("fnorm_two_b_") as d3:
            got = _run_to_end(mid, kg, p, d3)

    with checks() as c:
        c.check("the first call stopped mid-run", 0.0 < t_mid < ref["t"],
                f"t_mid={t_mid}, t_end={ref['t']}")
        c.check("the returned state carries a nonzero forcing_scale",
                bool(np.any(mid_scale != 0)), str(mid_scale))
        c.check(f"two calls end at the same time ({got['t']!r} vs {ref['t']!r})",
                got["t"] == ref["t"])
        for key in ("fields", "forcing_state", "forcing_scale", "forcing_key"):
            c.check(f"two calls give the same {key} as one, bitwise",
                    np.array_equal(got[key], ref[key]))


def test_refresh_computes_a_scale_only_when_the_state_carries_none():
    """The all-zero test is the whole rule: a fresh initialize gets a live scale, a
    developed state keeps the one it carries -- and the dt = 0 value that would replace it
    is a different number, so the no-op is not vacuous."""
    p, kg = _fctx("elsasser")
    fresh = make_state(p, ic=_restart_ic)
    fresh_scale = np.asarray(fresh.forcing_scale)
    refreshed = np.asarray(run._refresh_forcing_scale(fresh, kg, p).forcing_scale)

    stepper, scheme = get_scheme("lsrk33")
    end = _advance(make_state(p, ic=_restart_ic), kg, p, 4, scheme, stepper)
    stored = np.asarray(end.forcing_scale)
    kept = np.asarray(run._refresh_forcing_scale(end, kg, p).forcing_scale)
    scale_func = equation_registry[p.eqtype].forcing_scale_func
    at_zero_dt = np.asarray(scale_func(end, kg, p, 0.0))

    with checks() as c:
        c.check("a fresh initialize carries an all-zero forcing_scale",
                bool(np.all(fresh_scale == 0)))
        c.check(f"... and entry gives it a live one ({refreshed})",
                bool(np.all(refreshed != 0)))
        c.check(f"a developed state carries a nonzero forcing_scale ({stored})",
                bool(np.all(stored != 0)))
        c.check("... which entry keeps bitwise", np.array_equal(kept, stored),
                f"{kept} vs {stored}")
        c.check("... and which recomputing at dt = 0 would not reproduce",
                not np.array_equal(at_zero_dt, stored), f"{at_zero_dt} vs {stored}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
