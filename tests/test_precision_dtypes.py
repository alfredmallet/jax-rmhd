# Precision-model guards (plans/PRECISION_PLAN.md A5). Three independent questions:
#
#   1. DTYPE LEAKS (test_step_leaks_no_64bit_dtype / test_kgrid_carries_no_64bit_leaves).
#      jax_enable_x64 is now unconditionally on, so a bare jnp.array/linspace/arange or
#      an unpinned jax.random draw is a STRONG float64 that silently upcasts every field
#      it touches -- a ~2x slowdown and a changed bitstream that no tolerance-based test
#      would notice. These two run in BOTH precision sessions and assert the exact
#      dtypes of everything that leaves a step: fields/forcing_state at _precision.ctype,
#      forcing_scale at _precision.ftype, and t at float64 at EITHER precision. This is
#      the load-bearing regression test for the whole design; a future dtype source that
#      forgets its pin fails here, not in a benchmark three months later.
#
#   2. WHY t IS fp64 (test_t_accumulates_exactly_at_fp64). Pure scalar arithmetic, no
#      physics: 1e4 accumulations of t += dt reproduce the python-float (== fp64) sum
#      EXACTLY at float64 and drift at float32, and the hard failure the plan is really
#      about -- t + dt == t once t/dt > 1/eps32 ~ 1.7e7 -- is demonstrated directly.
#
#   3. THE fp32 RNG STREAM (test_fp32_matches_precision_reference, fp32-marked). The
#      pinned dtype= on the two jax.random.normal draws in shared_physics is what keeps
#      the fp32 forcing bitstream identical to the pre-x64 tree; an unpinned draw returns
#      float64 AND a different bitstream. Checked two ways, because neither alone is
#      enough: the recorded end-of-run reference (tests/data/precision_fp32_reference.npz,
#      generated on the pre-change tree by _gen_precision_reference.py) can only be
#      matched to ~4e-6 now -- fp64 t hands ou_update an exact dt = t_new - prev_t where
#      fp32 t handed it a quantized one -- so it catches a CHANGED stream (O(1) diffs) but
#      not a subtly different one; the first OU noise draw, on the other hand, involves no
#      t at all and is asserted BITWISE against hex constants recorded below. Since
#      2026-08-08 only forcing_state and t are compared at that tolerance: the forcing
#      normalization gained its self-energy term that day, which moves fields and
#      forcing_scale by a few percent BY DESIGN (see _REF_BAND_20260808 below).
#
# Single-process by design (the reference runs use snapshot tmp dirs, and the z_spectral
# kgrid variant is size==1 only); listed serially in the Savio manifest, both sessions.
# Script: `RMHD_PRECISION=32 python tests/test_precision_dtypes.py`.
from _rmhd_testing import bootstrap, checks, ctx, make_state, mpi_size, multimode_ic

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import _gen_precision_reference as gen
from taranis import _precision
from taranis.physics import shared_physics as sp
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

# jitted exactly like run.py's mpi4jax path (params/nblock/scheme/stepper static).
_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))

# one scheme from each family: lsrk33 is the integrating-factor production scheme
# (propagators.apply_exp), imexcb3e the recommended CB-IMEX one (solve_shifted +
# apply_L). They reach the propagator hook by different routes, and the IMEX steppers
# additionally build strong coefficient arrays inside the scan -- both worth covering.
_SCHEMES = ("lsrk33", "imexcb3e")

_FORCED = dict(forcing=True, forcing_mode="elsasser",
               forcing_power_elsasser=(0.5, 0.5), forcing_tau=1.0,
               fshell=(1, 3), forcing_seed=7)


def _dtypes(label, state, c):
    ftype, ctype = jnp.dtype(_precision.ftype), jnp.dtype(_precision.ctype)
    c.check(f"{label}: fields are {ctype.name}",
            state.fields.dtype == ctype, str(state.fields.dtype))
    c.check(f"{label}: forcing_state is {ctype.name}",
            state.forcing_state.dtype == ctype, str(state.forcing_state.dtype))
    c.check(f"{label}: forcing_scale is {ftype.name}",
            state.forcing_scale.dtype == ftype, str(state.forcing_scale.dtype))
    # t is float64 in BOTH sessions by design -- the one 64-bit quantity in an fp32 run.
    # It never multiplies fields (run._advance_forcing downcasts the single t-difference
    # that does), so this is not a leak, it is the point of the change.
    c.check(f"{label}: t is float64 (deliberate at every precision)",
            jnp.asarray(state.t).dtype == jnp.dtype(jnp.float64),
            str(jnp.asarray(state.t).dtype))


def test_step_leaks_no_64bit_dtype():
    with checks() as c:
        for forced in (False, True):
            params, kgrid = ctx(**(_FORCED if forced else {}))
            tag = "forced" if forced else "unforced"
            # initialize() first: a leaked dtype there would poison every step below
            _dtypes(f"{tag} initialize", make_state(params, ic=multimode_ic), c)
            for schemestr in _SCHEMES:
                stepper, scheme = get_scheme(schemestr)
                # fresh state per call -- block_of_steps is jitted without donation here,
                # but states are never shared across simulate-like calls regardless.
                end = _advance(make_state(params, ic=multimode_ic), kgrid, params,
                               1, scheme, stepper)
                _dtypes(f"{tag} {schemestr} 1 step", end, c)


def test_kgrid_carries_no_64bit_leaves():
    # setup_kgrids is the only sanctioned K_Grids constructor and therefore the one
    # choke point for the wavenumber grids; every float/complex leaf must come out at
    # FIELD precision. bool (dealias/fmask) and int (fidx_x/fidx_y) leaves are exempt.
    ftype, ctype = jnp.dtype(_precision.ftype), jnp.dtype(_precision.ctype)
    variants = [("plain", {}), ("forced", _FORCED),
                ("forced+shell_noise", dict(_FORCED, forcing_shell_noise=True))]
    if mpi_size() == 1:
        # z_spectral is the only mode that builds kz and a COMPLEX lin_L (+-i*kz), and
        # it is single-process by construction.
        variants.append(("z_spectral", dict(z_spectral=True)))
    with checks() as c:
        for label, kw in variants:
            _, kgrid = ctx(**kw)
            bad, seen = [], 0
            for path, leaf in jax.tree_util.tree_flatten_with_path(kgrid)[0]:
                dt = jnp.asarray(leaf).dtype
                name = jax.tree_util.keystr(path)
                if jnp.issubdtype(dt, jnp.complexfloating):
                    seen += 1
                    if dt != ctype:
                        bad.append(f"{name}:{dt}")
                elif jnp.issubdtype(dt, jnp.floating):
                    seen += 1
                    if dt != ftype:
                        bad.append(f"{name}:{dt}")
                elif not (jnp.issubdtype(dt, jnp.bool_) or jnp.issubdtype(dt, jnp.integer)):
                    bad.append(f"{name}: unexpected leaf dtype {dt}")
            c.check(f"kgrid ({label}): all {seen} float/complex leaves at field precision "
                    f"({ftype.name}/{ctype.name})", not bad, ", ".join(bad))


# ------------------------------------------------------------------ t at fp64

_N_ACC = 10_000
_T0, _DT = 1.0, 1e-4   # dt/t ~ 1e-4: the ratio a real run sits at


def _accumulate(dtype):
    # a jitted scalar loop, not a physics run: this test is about the ADD, nothing else
    t0 = jnp.asarray(_T0, dtype=dtype)
    dt = jnp.asarray(_DT, dtype=dtype)
    return jax.lax.fori_loop(0, _N_ACC, lambda i, t: t + dt, t0)


def test_t_accumulates_exactly_at_fp64():
    # python floats ARE IEEE doubles, so the reference below is the SAME arithmetic the
    # float64 loop performs, in the same order -- equality is exact, not a tolerance.
    ref = _T0
    for _ in range(_N_ACC):
        ref += _DT
    got64 = float(jax.jit(lambda: _accumulate(jnp.float64))())
    got32 = float(jax.jit(lambda: _accumulate(jnp.float32))())
    err32 = abs(got32 - ref)
    # the hard failure the plan is about: t stops advancing once t/dt > 1/eps32 ~ 1.7e7
    frozen32 = bool(jnp.float32(1e4) + jnp.float32(1e-4) == jnp.float32(1e4))
    frozen64 = bool(jnp.float64(1e4) + jnp.float64(1e-4) == jnp.float64(1e4))
    with checks() as c:
        c.check(f"float64: {_N_ACC} adds reproduce the python-float sum BITWISE "
                f"({got64!r})", got64 == ref, f"{got64!r} != {ref!r}")
        # counterexample: the same loop at float32 (measured 1.7e-4 absolute, i.e. ~1e-4
        # relative, under jax 0.6.2/CPU). Asserted loosely -- the point is that it is
        # orders of magnitude off, not the exact walk.
        c.check(f"float32: the same accumulation drifts (abs err {err32:.2e} > 1e-5) "
                f"-- this is why t is fp64", err32 > 1e-5, f"{got32!r} vs {ref!r}")
        c.check("float32 t FREEZES at t/dt ~ 1e8 (t + dt == t)", frozen32)
        c.check("float64 t still advances there", not frozen64)


# -------------------------------------------------- fp32 RNG stream / recorded reference

# First OU noise draw of each recorded config, entry by entry, as (n_ou, ab, kx, ky) index
# plus float.hex() of the real and imaginary parts. RECORDED FROM THE CURRENT TREE at
# RMHD_PRECISION=32 (2026-08-06) -- so they pin the stream going FORWARD; the pre-change
# agreement is what the reference npz below covers. float.hex() round-trips exactly and a
# float32 widens to double losslessly, so the comparison is bitwise.
#
# Entries were chosen to cover both symmetrized rows and the shell restriction:
#   - (n_ou, ab, 0, 0) is ky=0, kx=0: its own Hermitian mirror, hence exactly REAL
#   - (..., 3, 8) is the ky=Nyquist row (ny=16 -> nky=9), the other symmetrized row
#   - shellnoise's (0, 0, 0, 0) is exactly ZERO: kx=ky=0 is outside fshell=(1,5), which
#     is the whole point of forcing_shell_noise (a different, shell-only RNG stream)
_FIRST_NOISE = {
    "default": (
        (0, 0, 0, 0, "-0x1.29e4aa0000000p+0", "0x0.0p+0"),          # -1.16365+0j (real: kx=ky=0)
        (0, 0, 2, 0, "0x1.f79d8c0000000p+7", "0x1.c79d8a0000000p+7"),   # 251.808+227.808j
        (0, 0, 3, 8, "0x1.67b1360000000p+3", "-0x1.2f96da0000000p+7"),  # 11.2404-151.795j
        (0, 0, 1, 3, "-0x1.d5f6000000000p+6", "0x1.5c43360000000p+7"),  # -117.490+174.131j
        (0, 1, 5, 2, "-0x1.46a6c60000000p+7", "0x1.aa3e600000000p+7"),  # -163.326+213.122j
        (0, 1, 3, 0, "0x1.f46fb00000000p+7", "0x1.6db4140000000p+8"),   # 250.218+365.703j
    ),
    "shellnoise": (
        (0, 0, 2, 0, "0x1.93e11c0000000p+6", "0x1.e46b320000000p+7"),   # 100.970+242.209j
        (0, 0, 15, 4, "0x1.9e6c2e0000000p+6", "-0x1.2be7560000000p+7"), # 103.606-149.952j
        (0, 0, 1, 3, "0x1.6eace60000000p+6", "-0x1.81bcb00000000p+6"),  # 91.6688-96.4343j
        (0, 1, 3, 0, "-0x1.2258c80000000p+8", "-0x1.b950760000000p+6"), # -290.347-110.329j
        (0, 1, 2, 2, "-0x1.6025100000000p+6", "0x1.cb9bfe0000000p+6"),  # -88.0362+114.902j
        (0, 0, 0, 0, "0x0.0p+0", "0x0.0p+0"),                       # exactly 0: outside fshell
    ),
    "elsasser": (
        (0, 0, 0, 0, "-0x1.29e4aa0000000p+0", "0x0.0p+0"),          # -1.16365+0j (real: kx=ky=0)
        (0, 0, 2, 0, "0x1.f79d8c0000000p+7", "0x1.c79d8a0000000p+7"),   # 251.808+227.808j
        (1, 0, 2, 0, "0x1.51734e0000000p+8", "-0x1.2547c80000000p+4"),  # 337.450-18.3300j
        (1, 1, 4, 3, "0x1.a30de60000000p+7", "-0x1.a300960000000p+2"),  # 209.527-6.54691j
        (1, 0, 3, 8, "0x1.fe8f6e0000000p+4", "-0x1.b17e2e0000000p+4"),  # 31.9100-27.0933j
        (1, 1, 0, 0, "0x1.c821920000000p+7", "0x0.0p+0"),           # 228.066+0j (real: kx=ky=0)
    ),
}

# rel tolerance vs the pre-change reference, for the arrays the precision change was about.
# Measured max 4e-6 (fp64 t -> an exact rather than quantized dt into ou_update); 1e-5
# leaves margin while still failing loudly on a changed RNG stream, which moves the arrays
# by O(1). forcing_state and t are the only two that still qualify -- see below.
_REF_RTOL = 1e-5

# fields and forcing_scale MOVED ON PURPOSE on 2026-08-08: the forcing normalization gained
# the self-energy term (safe_scale -> selfnorm_scale, plans/FORCING_SPINUP_PLAN.md), which
# shifts the scale by O(F2*dt*target/P^2) per step -- measured 0.5-1.1% on forcing_scale and
# 2.6-3.4% on the 10-step fields, at the fixed dt=0.01 these configs use.
#
# The reference npz is NOT regenerated for this: it is the repo's only pre-x64-change
# artefact, and regenerating it on the current tree would make the whole comparison
# tautological (and invalidate the "reference t was recorded at fp32" check below). Instead
# the two arrays are bounded on BOTH sides -- they must have moved (a silent revert of the
# spin-up fix fails the lower bound) but only by a normalization-refinement amount (a
# changed RNG stream is O(1) and fails the upper bound). The exact post-fix normalization
# semantics are pinned by tests/test_forcing_spinup.py and
# test_forcing_smoke.test_selfnorm_scale_limits, not here; the RNG stream itself is pinned
# bitwise by test_fp32_forcing_noise_bitstream above.
_REF_BAND_20260808 = (1e-4, 1e-1)
_CHANGED_20260808 = ("fields", "forcing_scale")


def _first_noise(name, shell_noise, mode):
    """The noise ou_update draws on its very first call for this config -- no t involved,
    so it is bitwise-comparable regardless of t's dtype. Mirrors ou_update exactly:
    forcing_key starts at jax.random.key(forcing_seed) (run.initialize) and ou_update
    splits it before drawing."""
    params, kgrid = gen.config_ctx(shell_noise, mode)
    shape = (params.n_ou, 2, params.nx, params.ny // 2 + 1)
    key, _ = jax.random.split(jax.random.key(params.forcing_seed))
    grid_norm = float(params.nx * params.ny)
    if kgrid.fidx_x is not None:
        return sp._draw_shell_noise(key, shape, _precision.ctype, grid_norm,
                                    kgrid.fidx_x, kgrid.fidx_y)
    return sp._draw_symmetrized_noise(key, shape, _precision.ctype, grid_norm)


@pytest.mark.fp32
def test_fp32_forcing_noise_bitstream():
    with checks() as c:
        for name, shell_noise, mode in gen.CONFIGS:
            noise = np.asarray(_first_noise(name, shell_noise, mode))
            bad = []
            for i, j, ikx, iky, re_hex, im_hex in _FIRST_NOISE[name]:
                v = noise[i, j, ikx, iky]
                if (float(v.real) != float.fromhex(re_hex)
                        or float(v.imag) != float.fromhex(im_hex)):
                    bad.append(f"[{i},{j},{ikx},{iky}] {float(v.real).hex()}"
                               f"{float(v.imag).hex()}j")
            c.check(f"{name}: first OU noise draw matches the recorded fp32 bitstream "
                    f"in all {len(_FIRST_NOISE[name])} sampled entries", not bad,
                    "; ".join(bad))
            # the reality constraint the draw enforces on the two symmetrized rows;
            # exact because (a + conj(b))/sqrt2 and (b + conj(a))/sqrt2 are conjugates
            # entry for entry (fp addition is commutative)
            mirror = (-np.arange(noise.shape[-2])) % noise.shape[-2]
            c.check(f"{name}: noise is Hermitian on the ky=0 and ky=Nyquist rows",
                    np.array_equal(noise[..., 0], np.conj(noise[..., mirror, 0]))
                    and np.array_equal(noise[..., -1], np.conj(noise[..., mirror, -1])))


@pytest.mark.fp32
def test_fp32_matches_precision_reference():
    if mpi_size() > 1:
        print("[SKIP] test_fp32_matches_precision_reference -- single-process only "
              "(the recorded runs use snapshot tmp dirs)")
        return
    ref = np.load(gen.REFERENCE_PATH, allow_pickle=False)
    with checks() as c:
        want_keys = {f"{n}_{k}" for n, _, _ in gen.CONFIGS
                     for k in ("fields", "forcing_state", "forcing_scale", "t")}
        want_keys.add("git_commit")
        c.check(f"reference npz loads with all {len(want_keys)} keys "
                f"(recorded at commit {str(ref['git_commit'])[:12]})",
                set(ref.files) == want_keys, str(sorted(set(ref.files) ^ want_keys)))
        c.check("reference t was recorded at fp32 (pre-change tree)",
                ref["default_t"].dtype == np.float32, str(ref["default_t"].dtype))
        for name, shell_noise, mode in gen.CONFIGS:
            end = gen.run_config(shell_noise, mode)
            for key, got in (("fields", end.fields),
                             ("forcing_state", end.forcing_state),
                             ("forcing_scale", end.forcing_scale),
                             ("t", end.t)):
                a = np.asarray(got)
                b = ref[f"{name}_{key}"]
                rel = float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-30)
                if key in _CHANGED_20260808:
                    lo, hi = _REF_BAND_20260808
                    c.check(f"{name}.{key} differs from the pre-change reference by a "
                            f"self-energy-normalization amount (rel {rel:.2e} in "
                            f"[{lo:.0e}, {hi:.0e}]; changed by design 2026-08-08)",
                            lo <= rel <= hi)
                    continue
                c.check(f"{name}.{key} reproduces the pre-change reference "
                        f"(rel {rel:.2e} <= {_REF_RTOL:.0e})", rel <= _REF_RTOL)
            # t is the one thing that deliberately changed dtype
            c.check(f"{name}: the re-run carries float64 t",
                    jnp.asarray(end.t).dtype == jnp.dtype(jnp.float64))


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
