# The forcing spin-up kick (plans/FORCING_SPINUP_PLAN.md), fixed 2026-08-08.
#
# BEFORE: from a quiescent start the cross-power P = <grad z . grad f_raw> is 0, so
# safe_scale's s = target/P pinned at +-forcing_scale_max and the first forced step
# injected ~0.5*smax^2*F2*dt^2 of energy -- O(10) at 64^2, IDENTICAL for every
# forcing_power (eps enters the forcing ONLY through the scale, and the clamp erased it),
# and permanent at small dissipation. AFTER: rmhd._forcing_scale_from solves the
# self-energy-aware quadratic (shared_physics.selfnorm_scale), so the first injection is
# target*dt on the nose and scales with eps.
#
# Three checks, on the three configs recorded in _gen_forcing_spinup_reference.py (2D
# elsasser, 3D elsasser z-spectral, 3D elsasser finite-difference-z), 20 steps each:
#   1. the RNG-stream witness (see the long note on test_rng_stream_unchanged below),
#   2. E after step 2 == eps_tot*dt, the thing that was wrong,
#   3. E after step 2 now DEPENDS on eps (it did not before).
#
# fp64 only (the reference was recorded at fp64) and single-process (dims=2 and
# z_spectral are both size==1-only). Script: `python tests/test_forcing_spinup.py`.
from _rmhd_testing import bootstrap, checks

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import taranis as jr
from taranis import _precision
from taranis.physics.shared_physics import ou_update

EPS_TOT = 1e-2   # sum of the recorded forcing_power_elsasser entries (5e-3, 5e-3)


def _gen():
    """The reference generator module: the recorded configs ARE the test configs, imported
    rather than re-typed so the two can never drift apart (same contract as
    _gen_precision_reference.py in test_precision_dtypes).

    Imported LAZILY, inside the fp64-marked tests: the generator asserts at import that
    TARANIS_PRECISION == 64, so a module-level import would hard-fail pytest COLLECTION in the
    fp32 session instead of letting the marker skip these tests."""
    import _gen_forcing_spinup_reference as gen
    return gen


def _reference():
    return np.load(_gen().REFERENCE_PATH, allow_pickle=False)


def _with_eps(kwargs, eps_tot):
    return dict(kwargs, forcing_power_elsasser=(eps_tot / 2.0, eps_tot / 2.0))


@pytest.mark.fp64
def test_rng_stream_unchanged():
    """The fix adds NO RNG draws (F2 = <|grad f_raw|^2> is deterministic in the
    already-drawn f_raw), so the O-U machinery must reproduce the PRE-FIX forcing_state
    bitwise.

    Plan Phase 0 phrased this as "<name>_forcing_state_final comes out bitwise identical
    when the 20-step run is repeated post-fix". MEASURED: it does not, and cannot -- these
    configs use adaptive_timestep=True, so ou_update's decay/diffusion factors depend on
    the step dt, which depends on the field amplitude, which the fix changes by design
    (the pre-fix run's dt collapsed from 0.49 to 0.0037 at step 3 because of the kick).
    forcing_state is therefore trajectory-coupled through dt, not a pure RNG artefact, and
    a 20-step bitwise comparison would be a trajectory test wearing a stream test's hat.

    What IS a pure stream statement, and what is asserted here instead:
      (a) replaying the reference's OWN recorded dt sequence through the post-fix
          ou_update reproduces <name>_forcing_state_final BITWISE -- i.e. given the same
          dts, the noise draws, the key chain, the symmetrization and the shell mask are
          all untouched by the fix; and
      (b) the post-fix RUN's forcing_state after step 1 is bitwise equal to the first
          entry of that replay -- the end-to-end tie-in. Step 1 is exactly where the two
          agree on dt: it starts from rest in both trees (forcing_state is zero at t=0, so
          step 1 injects nothing whatever the scale is) and hence gets the same
          velocity-floor dt.
    """
    gen, ref = _gen(), _reference()
    with checks() as c:
        c.check("this session runs at fp64 (the reference precision)",
                _precision.precision == "64")
        for name, kwargs in gen.CONFIGS:
            params = jr.Parameters(**kwargs)
            kgrid = jr.setup_kgrids(params)
            # dts exactly as run.py forms them: t is fp64 and dt = t_new - t_prev.
            dts = np.diff(np.concatenate([[0.0], np.asarray(ref[f"{name}_t"])]))
            ou = jax.jit(lambda f, k, d: ou_update(f, k, d, params, kgrid))
            fs = jnp.zeros((params.n_ou, 2, params.nx, params.ny // 2 + 1),
                           dtype=_precision.ctype)
            key = jax.random.key(params.forcing_seed)
            fs_after_step1 = None
            for d in dts:
                fs, key = ou(fs, key, jnp.asarray(d, dtype=_precision.ftype))
                if fs_after_step1 is None:
                    fs_after_step1 = np.asarray(fs)
            c.check(f"{name}: replaying the reference dt sequence reproduces the pre-fix "
                    f"forcing_state after {len(dts)} steps BITWISE (no new RNG draws)",
                    np.array_equal(np.asarray(fs), ref[f"{name}_forcing_state_final"]),
                    f"max|diff| = {np.max(np.abs(np.asarray(fs) - ref[name + '_forcing_state_final'])):.3e}")

            run_state = _run(kwargs, n_steps=1)
            c.check(f"{name}: the post-fix run's own forcing_state after step 1 is "
                    f"bitwise identical to that replay's first entry",
                    np.array_equal(np.asarray(run_state.forcing_state), fs_after_step1))
            c.check(f"{name}: ... and step 1 took the same dt as the reference "
                    f"(quiescent velocity floor, unchanged by the fix)",
                    float(run_state.t) == float(ref[f"{name}_t"][0]),
                    f"{float(run_state.t)!r} vs {float(ref[f'{name}_t'][0])!r}")


def _run(kwargs, n_steps=None):
    """gen.run_config's stepping loop, optionally truncated. Returns the end state."""
    gen = _gen()
    if n_steps is None:
        return gen.run_config(kwargs)
    from taranis import run
    from taranis.timestepping import get_scheme
    params = jr.Parameters(**kwargs)
    kgrid = jr.setup_kgrids(params)
    ic = gen._zero_ic_3d if params.spatial_dimensions == 3 else gen._zero_ic_2d
    state = run._refresh_forcing_scale(jr.initialize(ic, params), kgrid, params)
    stepper, scheme = get_scheme("lsrk33")
    step_fn = jax.jit(run.block_of_steps, static_argnums=(2, 3, 4, 5))
    for _ in range(n_steps):
        state = step_fn(state, kgrid, params, 1, scheme, stepper)
    return state


@pytest.mark.fp64
def test_first_injection_equals_target_times_dt():
    # Step 1 injects NOTHING (forcing_state is initialized to zero, so f_raw = 0); the
    # whole first kick lands on step 2, whose scale was computed from the still-zero
    # fields -- exactly where safe_scale pinned at smax. Post-fix it must inject
    # eps_tot*dt2. Tolerance 1.5x: the quadratic is an exact solve for the FORWARD-EULER
    # injection over the step, while lsrk33 integrates the same force with RK weights, so
    # an O(dt/tau)-class discrepancy is expected and physical.
    gen, ref = _gen(), _reference()
    with checks() as c:
        for name, kwargs in gen.CONFIGS:
            result = gen.run_config(kwargs)
            # t[0] = t after step 1 = dt1, t[1] = t after step 2, so this is step 2's dt
            # (equal to dt1 here: step 1 leaves the box quiescent, so the velocity floor
            # sets the same dt again).
            dt2 = float(result["t"][1] - result["t"][0])
            E2 = float(result["E"][1])
            want = EPS_TOT * dt2
            c.check(f"{name}: E after step 2 = {E2:.6e} == eps_tot*dt = {want:.6e} "
                    f"within 1.5x (pre-fix: {float(ref[name + '_E'][1]):.4e})",
                    want / 1.5 < E2 < want * 1.5, f"ratio {E2 / want:.4f}")
            # the pre-fix value is the regression this test exists for -- assert we moved
            # away from it, so a revert cannot pass by coincidence.
            c.check(f"{name}: ... and is far below the pre-fix kick",
                    E2 < 0.1 * float(ref[f"{name}_E"][1]))


@pytest.mark.fp64
def test_first_injection_scales_with_eps():
    # The headline symptom: pre-fix E(step 2) was 1.0737e+01 for eps_tot = 1e-4, 1e-3,
    # 1e-2 and 1e-1 alike (identical to 4 digits) because eps enters the forcing ONLY
    # through the scale factor and the clamp erased it. Post-fix the ratio must track the
    # eps ratio. 2D config only -- this is a property of the normalization, not of the
    # z discretization, and the 3D runs are the expensive ones.
    gen = _gen()
    name, kwargs = gen.CONFIGS[0]
    E2 = {}
    for eps in (1e-2, 1e-4):
        result = gen.run_config(_with_eps(kwargs, eps))
        E2[eps] = float(result["E"][1])
    ratio = E2[1e-2] / E2[1e-4]
    with checks() as c:
        c.check(f"{name}: E(step 2) tracks eps -- {E2[1e-2]:.4e} at eps_tot=1e-2 vs "
                f"{E2[1e-4]:.4e} at 1e-4, ratio {ratio:.1f} (expect 100 within 2x; "
                f"pre-fix this ratio was 1.000)", 50.0 < ratio < 200.0)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
