# Hoisted per-stage propagators (params.hoist_propagator, default True).
#
# When dt is frozen over a block -- fixed dt, or one cfl_every block -- run.py forms every
# IF stage's exp(L*tau) ONCE per block (timestepping.stage_exp_ops) and the stepper applies
# the precomputed ExpOps, instead of re-evaluating the matrix exponential inside every
# stage of every step. The ExpOps are the very arrays apply_exp forms, in the same op
# order, so:
#
#   1. hoisted == unhoisted BITWISE at fp64, for every IF scheme (lsrk33, lsrk54, rk44),
#      in 2D, FD-z and z_spectral, fixed dt and cfl_every blocks, scan and unrolled stage
#      loops. The putzer2 (z_spectral nu != eta) and separable (z_spectral nu == eta)
#      backends are actually hoisted; the 2D / FD-z diagonal cells are never hoisted --
#      propagators.py header -- and guard that the knob leaves them alone.
#      At fp32 the assertion is round-off (rel < 1e-6 of max|fields|):
#      every cell but one is bitwise there too, and the one that is not (z_spectral,
#      lsrk54, cfl_every=2, scan) differs by 1 ulp from XLA fusing the precomputed-ExpOp
#      stage differently from the in-scan one -- the ExpOps themselves are bitwise
#      identical either way (checked in 4) -- the lsrk_scan class of caveat
#      (test_scheme_equivalence header). Measured on jax 0.10.0/CPU.
#   2. adaptive dt with cfl_every=1 is unchanged: stage_exp_ops is never consulted there
#      (nothing is frozen), so the knob has no effect on that path -- bitwise.
#   3. IMEX schemes never receive ExpOps (stage_exp_ops returns None for them) and are
#      bitwise independent of the knob.
#   4. stage_exp_ops returns one ExpOp per stage with the right leading shape, stacked
#      cleanly for the scan xs -- a Putzer2Exp per stage for a 2x2 L, a SeparableExp (the
#      small (nkx,nky) and (nz,1,1) coefficient arrays) for a separable L -- and None for a
#      diagonal L (the one backend that is not hoistable), for every IMEX scheme and with
#      the knob off; IdentityExp/DiagonalExp apply as apply_exp does, bitwise (the contract
#      the hoist rests on).
#   5. the knob round-trips through params.save / from_snapshot.
#   6. hoist_propagator=False really is the memory-light graph: the unhoisted putzer2
#      lsrk54 block compiles to strictly less than the hoisted one. The unhoisted stages
#      1..s-1 must keep forming exp_op from the SCANNED gamma -- forming them outside the
#      stage scan from static gammas passes every other gate here while growing the
#      unhoisted graph to the hoisted size, which is what this measurement catches.
#
# mpirun-safe: every configuration is single-process-sized and the z_spectral cells are
# skipped when size > 1 (z_spectral requires size == 1).
from _rmhd_testing import (bootstrap, checks, ctx, fresh_params, make_state, mpi_size,
                           snap_dir)

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np

import taranis as jr
from taranis import _precision, propagators
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme, stage_exp_ops

_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))


def _ic(x, y, z=None):
    # non-degenerate multi-mode IC in 2D and 3D (phi != psi, so both Elsasser fields and
    # every putzer2 entry are exercised)
    zc = 1.0 if z is None else jnp.cos(z)
    phi = (jnp.cos(x + 1.4) + jnp.cos(y + 2.0)) * zc + 0.3 * jnp.sin(2 * x + y)
    psi = (jnp.cos(2 * x + 2.3) + 0.5 * jnp.cos(y + 6.2)) * zc + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([phi, psi], axis=0)

_FORCED = dict(forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(1.0, 1.0),
               forcing_tau=1.0, fshell=(1, 3), forcing_seed=1)


def _end_fields(schemestr, nsteps=6, **kw):
    params, kgrid = ctx(**kw)
    stepper, scheme = get_scheme(schemestr)
    end = _advance(make_state(params, ic=_ic), kgrid, params, nsteps, scheme, stepper)
    return np.asarray(end.fields), float(end.t)


def _geoms():
    # label -> ctx overrides; z_spectral is size==1 only. Both z_spectral cells are hoisted
    # backends: nu == eta is separable, nu != eta is putzer2.
    g = {"2d": dict(dims=2, nx=16, ny=16),
         "3dfd": dict()}
    if mpi_size() == 1:
        g["3dspec"] = dict(z_spectral=True)
        g["3dspec_uneq"] = dict(z_spectral=True, diss=(1e-4, 2e-4))
    return g


def _dt_modes():
    return {"fixed dt": dict(adaptive_timestep=False),
            "cfl_every=2": dict(adaptive_timestep=True, cfl_every=2),
            "adaptive cfl_every=1": dict(adaptive_timestep=True, cfl_every=1)}


def test_hoisted_matches_unhoisted_bitwise():
    x64 = _precision.precision == "64"
    with checks() as c:
        for gl, geom in _geoms().items():
            for ml, mode in _dt_modes().items():
                for schemestr in ("lsrk33", "lsrk54", "rk44"):
                    for scan in (True, False):
                        if schemestr == "rk44" and not scan:
                            continue   # rk44 has no stage loop
                        kw = dict(geom, **mode, **_FORCED, lsrk_scan=scan)
                        a, ta = _end_fields(schemestr, hoist_propagator=True, **kw)
                        b, tb = _end_fields(schemestr, hoist_propagator=False, **kw)
                        rel = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
                        trel = abs(ta - tb) / tb
                        if x64:
                            c.check(f"{gl} {ml} {schemestr} lsrk_scan={scan}: hoisted == "
                                    f"unhoisted bitwise (t {ta:.4f})",
                                    np.array_equal(a, b) and ta == tb, f"rel {rel:.2e}")
                        else:
                            c.check(f"{gl} {ml} {schemestr} lsrk_scan={scan}: hoisted == "
                                    f"unhoisted to round-off (rel {rel:.1e}, t rel {trel:.1e})",
                                    rel < 1e-6 and trel < 1e-6)


def test_imex_independent_of_knob():
    with checks() as c:
        for gl, geom in _geoms().items():
            for ml, mode in _dt_modes().items():
                kw = dict(geom, **mode, **_FORCED)
                a, _ = _end_fields("imexcb3e", hoist_propagator=True, **kw)
                b, _ = _end_fields("imexcb3e", hoist_propagator=False, **kw)
                c.check(f"{gl} {ml} imexcb3e: bitwise independent of hoist_propagator",
                        np.array_equal(a, b))


def test_stage_exp_ops_structure():
    with checks() as c:
        # the diagonal backend (FD-z) is never hoisted
        params, kgrid = ctx()
        c.check("3dfd (diagonal L): propagator is not hoistable",
                kgrid.lin.hoistable is False)
        for schemestr in ("lsrk33", "lsrk54", "rk44"):
            stepper, scheme = get_scheme(schemestr)
            c.check(f"3dfd (diagonal L) {schemestr}: stage_exp_ops is None",
                    stage_exp_ops(kgrid, params, scheme, stepper, params.dt) is None)
        cases = {}
        if mpi_size() == 1:
            cases["3dspec (separable L)"] = (dict(z_spectral=True),
                                             propagators.SeparableExp)
            cases["3dspec_uneq (putzer2 L)"] = (dict(z_spectral=True, diss=(1e-4, 2e-4)),
                                                propagators.Putzer2Exp)
        for label, (geom, kind) in cases.items():
            params, kgrid = ctx(**geom)
            for schemestr, nstage in (("lsrk33", 3), ("lsrk54", 5), ("rk44", 2)):
                stepper, scheme = get_scheme(schemestr)
                ops = stage_exp_ops(kgrid, params, scheme, stepper, params.dt)
                c.check(f"{label} {schemestr}: {nstage} ops of kind {kind.__name__}",
                        ops is not None and len(ops) == nstage
                        and all(isinstance(o, kind) for o in ops))
                stacked = propagators.stack_exp_ops(ops)
                lead = {leaf.shape[0] for leaf in jax.tree.leaves(stacked)}
                c.check(f"{label} {schemestr}: stacked ExpOp has leading axis {nstage}",
                        lead == {nstage})
                # the stacked op slices back to the per-stage one, exactly
                sliced = jax.tree.map(lambda x: x[1], stacked)
                c.check(f"{label} {schemestr}: slicing the stack recovers stage 1 bitwise",
                        all(np.array_equal(np.asarray(x), np.asarray(y))
                            for x, y in zip(jax.tree.leaves(sliced), jax.tree.leaves(ops[1]))))
            for schemestr in ("imexcb2", "imexcb3e", "imexcb3c", "imexcb3f"):
                stepper, scheme = get_scheme(schemestr)
                c.check(f"{label} {schemestr}: stage_exp_ops is None for an IMEX scheme",
                        stage_exp_ops(kgrid, params, scheme, stepper, params.dt) is None)
            # knob off: None for every scheme
            p_off = fresh_params(hoist_propagator=False, **geom)
            kg_off = jr.setup_kgrids(p_off)
            stepper, scheme = get_scheme("lsrk33")
            c.check(f"{label}: hoist_propagator=False -> stage_exp_ops is None",
                    stage_exp_ops(kg_off, p_off, scheme, stepper, p_off.dt) is None)
        cases["3dfd (diagonal L)"] = (dict(), propagators.DiagonalExp)
        # the identity backend: an ExpOp with no arrays that applies as the identity
        ident = propagators.IdentityOperator().exp_op(0.3)
        x = jnp.ones((2, 3))
        c.check("IdentityOperator.exp_op applies as the identity and holds no arrays",
                isinstance(ident, propagators.IdentityExp) and ident.apply(x) is x
                and jax.tree.leaves(ident) == [])
        # a stepper's two exp_op forms agree with apply_exp bitwise (the contract the whole
        # hoist rests on): exp_op(tau).apply(arr) is apply_exp(arr, tau)
        for label, (geom, _) in cases.items():
            params, kgrid = ctx(**geom)
            prop = kgrid.lin.scaled(params.dt)
            arr = make_state(params, ic=_ic).fields
            c.check(f"{label}: exp_op(tau).apply(arr) == apply_exp(arr, tau) bitwise",
                    np.array_equal(np.asarray(jax.jit(lambda a: prop.exp_op(0.4).apply(a))(arr)),
                                   np.asarray(jax.jit(lambda a: prop.apply_exp(a, 0.4))(arr))))


def _block_total_u(schemestr, nsteps=5, **kw):
    # compiled temp+args+out of one block, in u = one field-sized complex array
    params, kgrid = ctx(**kw)
    stepper, scheme = get_scheme(schemestr)
    m = _advance.lower(make_state(params, ic=_ic), kgrid, params, nsteps,
                       scheme, stepper).compile().memory_analysis()
    total = (m.temp_size_in_bytes + m.argument_size_in_bytes + m.output_size_in_bytes)
    u = (params.nz * params.nx * (params.ny // 2 + 1)
         * np.dtype(_precision.ctype).itemsize)
    return total / u


# the measured lsrk54 gap is ~8.2-8.8 u across grids from 16^2x8 to 64^2x16 and both
# precisions; half of it is the margin. At lsrk33 the gap is under 1 u (only 3 stages of
# ExpOp to hoist) -- there is nothing to assert there, so the gate is lsrk54's.
_UNHOISTED_MARGIN_U = 4.0


def test_unhoisted_graph_stays_memory_light():
    if mpi_size() > 1:
        return   # z_spectral is size == 1 only
    geom = dict(z_spectral=True, diss=(1e-4, 2e-4))   # nu != eta: the putzer2 backend
    hoisted = _block_total_u("lsrk54", hoist_propagator=True, **geom)
    unhoisted = _block_total_u("lsrk54", hoist_propagator=False, **geom)
    with checks() as c:
        c.check("putzer2 lsrk54: hoist_propagator=False compiles at least "
                f"{_UNHOISTED_MARGIN_U:.0f} u below hoisted",
                unhoisted <= hoisted - _UNHOISTED_MARGIN_U,
                f"hoisted {hoisted:.2f} u, unhoisted {unhoisted:.2f} u, "
                f"gap {hoisted - unhoisted:.2f} u")


def test_knob_round_trips_through_save():
    with checks() as c, snap_dir("hoist_") as path:
        p = fresh_params(hoist_propagator=False)
        p.save(path)
        p2 = jr.Parameters.from_snapshot(path)
        c.check("hoist_propagator=False survives save/from_snapshot",
                p2.hoist_propagator is False)
        c.check("default is True", fresh_params().hoist_propagator is True)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
