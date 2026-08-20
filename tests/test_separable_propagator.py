# The Elsasser-separable propagator backend (propagators.SeparableL /
# SeparablePropagator), which RMHD uses under z_spectral when nu == eta.
#
# There L = d*I + i*kz*sigma_x with a real, separable d = dperp(k_perp) + dz(kz), so
# exp(L*tau) and (I - a*L)^-1 have closed forms built from (nkx,nky) and (nz,1,1) arrays
# only -- no full-grid operator is stored or formed. What is checked here:
#
#   1. backend selection both ways: z_spectral + nu == eta -> separable (kgrid.lin_L None,
#      the three separable entries populated); nu != eta -> putzer2; FD-z / 2D -> diagonal.
#      The separable backend is hoistable and its whole per-stage ExpOp stack is a fraction
#      of one field array (test_hoist_propagator owns the hoisted == unhoisted cells).
#   2. the separable exp / solve_shifted / apply_L agree with the putzer2 backend running
#      on the SAME operator, materialised by propagators.dense_operator.
#   3. exp_op(tau).apply(arr) is apply_exp(arr, tau), bitwise (the propagators contract).
#   4. a real field stays real: the rfft2/rfftn conjugate rows survive 20 steps, which is
#      what the Nyquist-kz rule buys.
#   5. dense_operator of the separable entries reproduces rmhd.linear_matrix's dense form
#      bitwise, and setup rejects separable operators that would break reality.
#   6. FD-z and 2D are untouched: their solver output is bitwise identical with the
#      dense-path switch (rmhd.SEPARABLE_L) either way, and the z_spectral separable and
#      putzer2 paths agree to round-off.
#
# The switch is rmhd.SEPARABLE_L, a module constant: False makes linear_matrix return the
# dense 2x2 operator even at nu == eta, i.e. the pre-separable behaviour, in-process on
# this tree. Every kgrid here is built with fresh_params + setup_kgrids (never ctx) so a
# flipped switch is actually re-read.
#
# mpirun-safe: z_spectral requires size == 1, so every z_spectral check is skipped when
# the world size is larger.
from _rmhd_testing import bootstrap, checks, fresh_params, mpi_size

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np

import taranis as jr
from taranis import _precision, propagators
from taranis.physics import rmhd
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme, stage_exp_ops

_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))

# tau (and the IMEX shift a) over four decades, plus the exactly-zero case
_TAUS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)
_RTOL = 1e-13 if _precision.precision == "64" else 1e-5

_ZS = dict(z_spectral=True, nz=8, eqpars=dict(z_diss_k=1e-4))
_UNEQ = dict(_ZS, diss=(1e-4, 2e-4))


def _ctx(**overrides):
    p = fresh_params(**overrides)
    return p, jr.setup_kgrids(p)


def _ic(x, y, z=None):
    zc = 1.0 if z is None else jnp.cos(z)
    phi = (jnp.cos(x + 1.4) + jnp.cos(y + 2.0)) * zc + 0.3 * jnp.sin(2 * x + y)
    psi = (jnp.cos(2 * x + 2.3) + 0.5 * jnp.cos(y + 6.2)) * zc + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([phi, psi], axis=0)


def _random_fields(kgrid, params, seed=7):
    nz = params.nz if params.spatial_dimensions == 3 else 1
    shape = (2, nz, params.nx, params.ny // 2 + 1)
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape),
                       dtype=_precision.ctype)


def _dense_putzer(kgrid):
    # the putzer2 backend running on the operator the separable entries encode
    return propagators.Putzer2Propagator(
        *propagators.putzer2_precompute(propagators.dense_operator(kgrid)))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b))) / max(1e-300, float(np.max(np.abs(b))))


def _mirror(arr, axis):
    # index k -> -k along a two-sided FFT axis (0 and Nyquist map to themselves)
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)


def _reality_violation(fields):
    # F(-kx,-kz,ky) = conj(F(kx,kz,ky)) on the ky = 0 and Nyquist rows
    f = np.asarray(fields)
    m = _mirror(_mirror(f, -2), -3)
    worst = 0.0
    for row in (0, f.shape[-1] - 1):
        worst = max(worst, float(np.max(np.abs(m[..., row] - np.conj(f[..., row])))))
    return worst / max(1e-300, float(np.max(np.abs(f))))


def _run(schemestr, nsteps, params, kgrid, ic):
    stepper, scheme = get_scheme(schemestr)
    end = _advance(jr.initialize(ic, params), kgrid, params, nsteps, scheme, stepper)
    return np.asarray(end.fields)


def test_backend_selection():
    with checks() as c:
        for label, kw in (("2D", dict(dims=2)), ("FD-z", {})):
            params, kgrid = _ctx(**kw)
            c.check(f"{label}: diagonal backend",
                    isinstance(propagators.get_propagator(kgrid, params),
                               propagators.DiagonalPropagator))
            c.check(f"{label}: no separable entries",
                    kgrid.lin_dperp is None and kgrid.lin_dz is None
                    and kgrid.lin_kz is None and kgrid.lin_L is not None)
        if mpi_size() > 1:
            return
        params, kgrid = _ctx(**_ZS)
        prop = propagators.get_propagator(kgrid, params)
        nkx, nky = params.nx, params.ny // 2 + 1
        c.check("z_spectral nu == eta: separable backend",
                isinstance(prop, propagators.SeparablePropagator))
        c.check("z_spectral nu == eta: lin_L / lin_m / lin_s are all None",
                kgrid.lin_L is None and kgrid.lin_m is None and kgrid.lin_s is None)
        c.check("z_spectral nu == eta: the three separable entries are populated, real "
                "and plane-sized",
                kgrid.lin_dperp.shape == (nkx, nky)
                and kgrid.lin_dz.shape == (params.nz, 1, 1)
                and kgrid.lin_kz.shape == (params.nz, 1, 1)
                and not any(jnp.iscomplexobj(a) for a in
                            (kgrid.lin_dperp, kgrid.lin_dz, kgrid.lin_kz)),
                f"{kgrid.lin_dperp.shape} {kgrid.lin_dz.shape} {kgrid.lin_kz.shape}")
        # hoistable: the per-stage ExpOps are (nkx,nky) + 2*(nz,1,1), so hoisting the
        # coefficient evaluation off the stage costs a fraction of one field array
        c.check("the separable backend is hoistable", prop.hoistable is True)
        for schemestr, nstage in (("lsrk33", 3), ("lsrk54", 5), ("rk44", 2)):
            stepper, scheme = get_scheme(schemestr)
            ops = stage_exp_ops(kgrid, params, scheme, stepper, params.dt)
            c.check(f"z_spectral nu == eta {schemestr}: {nstage} SeparableExp stage ops",
                    ops is not None and len(ops) == nstage
                    and all(isinstance(o, propagators.SeparableExp) for o in ops))
            nleaf = sum(a.size for a in jax.tree.leaves(propagators.stack_exp_ops(ops)))
            # nstage*(P + c + s) and nothing full-grid: nstage/nz of a field array
            c.check(f"z_spectral nu == eta {schemestr}: the stage stack is "
                    f"{nstage}*(nkx*nky + 2*nz), never full-grid",
                    nleaf == nstage * (nkx * nky + 2 * params.nz),
                    f"{nleaf} elements, expected "
                    f"{nstage * (nkx * nky + 2 * params.nz)}, u = "
                    f"{params.nz * nkx * nky}")
        for schemestr in ("imexcb2", "imexcb3e"):
            stepper, scheme = get_scheme(schemestr)
            c.check(f"z_spectral nu == eta {schemestr}: stage_exp_ops is None for an IMEX "
                    f"scheme",
                    stage_exp_ops(kgrid, params, scheme, stepper, params.dt) is None)

        params, kgrid = _ctx(**_UNEQ)
        c.check("z_spectral nu != eta: putzer2 backend",
                isinstance(propagators.get_propagator(kgrid, params),
                           propagators.Putzer2Propagator))
        c.check("z_spectral nu != eta: dense lin_L, no separable entries",
                kgrid.lin_L is not None and kgrid.lin_L.ndim == 5
                and kgrid.lin_dperp is None and kgrid.lin_dz is None
                and kgrid.lin_kz is None)
        stepper, scheme = get_scheme("lsrk33")
        c.check("z_spectral nu != eta: stage_exp_ops still hoists putzer2",
                stage_exp_ops(kgrid, params, scheme, stepper, params.dt) is not None)


def test_separable_agrees_with_putzer2_on_the_same_operator():
    if mpi_size() > 1:
        return
    params, kgrid = _ctx(**_ZS)
    sep = propagators.get_propagator(kgrid, params)
    put = _dense_putzer(kgrid)
    arr = _random_fields(kgrid, params)
    with checks() as c:
        for tau in _TAUS:
            c.check(f"apply_exp(tau={tau:g}) matches putzer2 on dense_operator",
                    _rel(sep.apply_exp(arr, tau), put.apply_exp(arr, tau)) < _RTOL,
                    f"rel {_rel(sep.apply_exp(arr, tau), put.apply_exp(arr, tau)):.2e}")
            c.check(f"solve_shifted(a={tau:g}) matches putzer2 on dense_operator",
                    _rel(sep.solve_shifted(arr, tau), put.solve_shifted(arr, tau)) < _RTOL,
                    f"rel {_rel(sep.solve_shifted(arr, tau), put.solve_shifted(arr, tau)):.2e}")
        c.check("apply_L matches putzer2 on dense_operator",
                _rel(sep.apply_L(arr), put.apply_L(arr)) < _RTOL,
                f"rel {_rel(sep.apply_L(arr), put.apply_L(arr)):.2e}")
        # scaled(dt) is the form every LSRK stage uses
        dt = 0.021
        for tau in _TAUS:
            c.check(f"scaled(dt).apply_exp(tau={tau:g}) matches putzer2",
                    _rel(sep.scaled(dt).apply_exp(arr, tau),
                         put.scaled(dt).apply_exp(arr, tau)) < _RTOL)


def test_exp_op_is_apply_exp_bitwise():
    if mpi_size() > 1:
        return
    params, kgrid = _ctx(**_ZS)
    arr = _random_fields(kgrid, params)
    with checks() as c:
        for prop, label in ((propagators.get_propagator(kgrid, params), "separable"),
                            (propagators.get_propagator(kgrid, params).scaled(0.021),
                             "separable scaled(dt)")):
            for tau in _TAUS:
                a = jax.jit(lambda x, t=tau, p=prop: p.exp_op(t).apply(x))(arr)
                b = jax.jit(lambda x, t=tau, p=prop: p.apply_exp(x, t))(arr)
                c.check(f"{label}: exp_op({tau:g}).apply == apply_exp bitwise",
                        np.array_equal(np.asarray(a), np.asarray(b)))


def test_reality_is_preserved():
    if mpi_size() > 1:
        return
    params, kgrid = _ctx(**_ZS)
    tol = _RTOL
    with checks() as c:
        viol = _reality_violation(np.asarray(jr.initialize(_ic, params).fields))
        c.check("the initial condition satisfies the conjugate-row constraint",
                viol < tol, f"rel violation {viol:.2e}")
        for schemestr in ("lsrk33", "imexcb3e"):
            end = _run(schemestr, 20, params, kgrid, _ic)
            viol = _reality_violation(end)
            c.check(f"{schemestr}: 20 steps leave F(-kx,-kz,ky) = conj(F) at round-off",
                    viol < tol, f"rel violation {viol:.2e}")
            c.check(f"{schemestr}: the fields are finite and non-trivial",
                    np.all(np.isfinite(end)) and np.max(np.abs(end)) > 0.0)


def test_dense_operator_reproduces_the_dense_linear_matrix_bitwise():
    if mpi_size() > 1:
        return
    params, kgrid = _ctx(**_ZS)
    reference = _dense_linear_matrix(kgrid, params)
    with checks() as c:
        c.check("dense_operator(separable entries) == rmhd.linear_matrix's dense form, "
                "bitwise",
                np.array_equal(np.asarray(propagators.dense_operator(kgrid)),
                               np.asarray(reference)))


def test_particle_psi_diagonal_reads_both_separable_terms():
    # particles/fields._psi_linear_diagonal takes the psi diagonal of L, which on this
    # backend is the SUM dperp + dz. z_diss_k != 0 is what makes dz nonzero, and every
    # particle gate runs with it at 0, so this is the only check that the sum is a sum.
    if mpi_size() > 1:
        return
    from taranis.particles import fields as pfields
    params, kgrid = _ctx(**_ZS)
    psi_diag = np.asarray(pfields._psi_linear_diagonal(kgrid, params))
    dense_psi = np.asarray(propagators.dense_operator(kgrid)[1, 1])
    dz = np.asarray(kgrid.lin_dz)
    with checks() as c:
        c.check("the dz term is actually nonzero here (z_diss_k != 0)",
                float(np.max(np.abs(dz))) > 0.0, f"max|dz| = {np.max(np.abs(dz)):.3e}")
        c.check("_psi_linear_diagonal == dense_operator[1,1] exactly",
                np.array_equal(psi_diag, dense_psi),
                f"max diff {float(np.max(np.abs(psi_diag - dense_psi))):.3e}")
        # dperp alone (the term the sum could silently lose) is measurably different
        c.check("dropping dz would be visible",
                not np.array_equal(np.broadcast_to(np.asarray(kgrid.lin_dperp),
                                                   psi_diag.shape), psi_diag))


def _dense_linear_matrix(kgrid, params):
    # rmhd.linear_matrix's dense 2x2 form, cast exactly as grids._attach_linear_operator
    # casts it
    try:
        rmhd.SEPARABLE_L = False
        return rmhd.linear_matrix(kgrid, params).astype(_precision.ctype)
    finally:
        rmhd.SEPARABLE_L = True


def _with_switch(value, fn):
    try:
        rmhd.SEPARABLE_L = value
        return fn()
    finally:
        rmhd.SEPARABLE_L = True


def test_fd_and_2d_are_untouched_and_z_spectral_moves_at_round_off():
    with checks() as c:
        for label, kw in (("2D", dict(dims=2)), ("FD-z", {})):
            for schemestr in ("lsrk33", "imexcb3e"):
                out = []
                for value in (True, False):
                    params, kgrid = _with_switch(value, lambda: _ctx(**kw))
                    out.append(_with_switch(value,
                                            lambda: _run(schemestr, 20, params, kgrid, _ic)))
                c.check(f"{label} {schemestr}: 20 steps bitwise identical with the dense "
                        f"path forced",
                        np.array_equal(out[0], out[1]), f"rel {_rel(out[0], out[1]):.2e}")
        if mpi_size() > 1:
            return
        for schemestr in ("lsrk33", "lsrk54"):
            out = []
            for value in (True, False):
                params, kgrid = _with_switch(value, lambda: _ctx(**_ZS))
                out.append(_with_switch(value,
                                        lambda: _run(schemestr, 20, params, kgrid, _ic)))
            rel = _rel(out[0], out[1])
            c.check(f"z_spectral {schemestr}: separable and putzer2 agree to round-off "
                    f"over 20 steps", rel < _RTOL, f"rel {rel:.2e}")


def test_setup_rejects_separable_operators_that_break_reality_or_shape():
    if mpi_size() > 1:
        return
    params, kgrid = _ctx(**_ZS)
    good = propagators.SeparableL(kgrid.lin_dperp, kgrid.lin_dz, kgrid.lin_kz)
    nz = params.nz
    bad = {
        "a kz whose Nyquist plane is not zeroed":
            good._replace(kz=kgrid.kz),
        "a kz that is not antisymmetric":
            good._replace(kz=jnp.abs(kgrid.lin_kz)),
        "a complex dperp":
            good._replace(dperp=kgrid.lin_dperp.astype(_precision.ctype)),
        "a dz that is not symmetric under kz -> -kz":
            good._replace(dz=kgrid.lin_kz),
        "a dperp of the wrong shape":
            good._replace(dperp=jnp.zeros((params.nx, 3), dtype=_precision.ftype)),
        "a dz of the wrong z extent":
            good._replace(dz=jnp.zeros((nz + 2, 1, 1), dtype=_precision.ftype)),
    }
    with checks() as c:
        try:
            propagators.linear_fields(good, params)
            err = None
        except (ValueError, NotImplementedError) as e:
            err = str(e)
        c.check("the RMHD separable operator passes setup", err is None, f"raised {err!r}")
        for name, sep in bad.items():
            try:
                propagators.linear_fields(sep, params)
                err = None
            except (ValueError, NotImplementedError) as e:
                err = str(e)
            c.check(f"setup rejects {name}", err is not None)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
