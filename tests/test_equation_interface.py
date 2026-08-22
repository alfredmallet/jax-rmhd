# The typed equation interface (plans/REFACTOR_PLAN.md §3): a recipe's grad_func returns a
# NAMED tuple (rmhd.RMHDGrads / gdi.GDIGrads) that term funcs and set_timestep read by
# name, and term_funcs entries are physics.Term(func, active) whose static predicate
# decides -- at TRACE time, from the (static) Parameters -- whether the term enters the
# graph at all. An inactive term is not called and its zeros are not added.
#
# What is pinned here: the gradient names and their order against independently computed
# k-space gradients; that construct_rhs calls only the active terms; that filtering the
# inactive terms out is bitwise the same RHS as summing their zeros in; that a bare
# callable in term_funcs still works (tests/test_imex.py's toy recipes are built that way);
# that the predicates run once per TRACE, not once per step; and that a configuration with
# no active term is a clear error rather than a None RHS.
#
# Both precision sessions run this: the bitwise RHS comparison is an exact-equality check
# in the working dtype, and the rest is structural.
#
# mpirun-safe: every cell is dims=2 or z_spectral (single-process by construction) except
# the FD-z gradient/RHS cells, which are collective-free (grad_fields and the terms do no
# reductions; the halo exchange inside FDLinearTerm is called identically on every rank).
#
# pytest: `pytest tests/test_equation_interface.py`. Script:
# `python tests/test_equation_interface.py`.
from _rmhd_testing import bootstrap, checks, fresh_params, make_state, mpi_size

bootstrap()

import contextlib

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision, grids
from taranis.physics import (EquationRecipe, Term, _halo_start_enabled, construct_rhs,
                             equation_registry, gdi, rmhd)
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

# jitted exactly like run.py's non-"jax" path (params/nblock/scheme/stepper static)
_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))

_FORCED_2D = dict(dims=2, nx=32, ny=32, forcing=True, forcing_mode="elsasser",
                  forcing_power_elsasser=(1.0, 1.0), forcing_tau=1.0, fshell=(1, 3),
                  forcing_seed=1, adaptive_timestep=False, dt=0.01)
_GDI_2D = dict(dims=2, nx=32, ny=32, eqtype="GDI", adaptive_timestep=False, dt=0.01,
               diss=5e-6, hyper=2,
               eqpars=dict(Ln=392.0, nu_in=0.0106, v0=25.0, gpar_fac=1.0))


def _ic(x, y, z=None):
    # non-degenerate multi-mode IC, the two fields different (both Elsasser fields live)
    zc = 1.0 if z is None else jnp.cos(z)
    f0 = (jnp.cos(x + 1.4) + jnp.cos(y + 2.0)) * zc + 0.3 * jnp.sin(2 * x + y)
    f1 = (jnp.cos(2 * x + 2.3) + 0.5 * jnp.cos(y + 6.2)) * zc + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([f0, f1], axis=0)


def _ctx(**overrides):
    params = fresh_params(**overrides)
    return params, jr.setup_kgrids(params)


def _grad_of(fk, kgrid, params):
    """(d/dx, d/dy) of one k-space field, built here rather than by grad_fields."""
    return jnp.stack([grids.ifft(1j * kgrid.kx * fk, params),
                      grids.ifft(1j * kgrid.ky * fk, params)])


@contextlib.contextmanager
def _registered(name, recipe):
    """A test-only equation_registry entry for the duration of a test."""
    equation_registry[name] = recipe
    try:
        yield
    finally:
        equation_registry.pop(name, None)


# ------------------------------------------------------------------- named gradients

def test_grads_are_named_in_the_documented_order():
    """Each recipe's grad_func returns its NamedTuple, whose fields are the gradients of
    the documented k-space quantities in the documented order."""
    with checks() as c:
        params, kgrid = _ctx()
        state = make_state(params, ic=_ic)
        grads = equation_registry["RMHD"].grad_func(state, kgrid, params)
        phik, psik = state.fields[0], state.fields[1]
        want = {"gphi": phik, "gpsi": psik,
                "gvort": -kgrid.ksq * phik, "gjpar": -kgrid.ksq * psik}
        c.check("RMHD grads are an RMHDGrads and a tuple",
                isinstance(grads, rmhd.RMHDGrads) and isinstance(grads, tuple))
        c.check("RMHD grad field names and order are (gphi, gpsi, gvort, gjpar)",
                grads._fields == ("gphi", "gpsi", "gvort", "gjpar"), f"{grads._fields}")
        _check_grad_entries(c, "RMHD", grads, want, kgrid, params)

        params, kgrid = _ctx(**_GDI_2D)
        state = make_state(params, ic=_ic)
        grads = equation_registry["GDI"].grad_func(state, kgrid, params)
        Nk, phik = state.fields[0], state.fields[1]
        want = {"gphi": phik, "gN": Nk, "gvort": -kgrid.ksq * phik}
        c.check("GDI grads are a GDIGrads and a tuple",
                isinstance(grads, gdi.GDIGrads) and isinstance(grads, tuple))
        c.check("GDI grad field names and order are (gphi, gN, gvort)",
                grads._fields == ("gphi", "gN", "gvort"), f"{grads._fields}")
        _check_grad_entries(c, "GDI", grads, want, kgrid, params)


def _check_grad_entries(c, eq, grads, want, kgrid, params):
    shape = (2, params.nz, params.nx, params.ny)
    for i, name in enumerate(grads._fields):
        g = getattr(grads, name)
        c.check(f"{eq} {name}: real-space {shape} at field precision",
                g.shape == shape and g.dtype == _precision.ftype,
                f"shape {g.shape}, dtype {g.dtype}")
        c.check(f"{eq} {name}: positional entry {i} is the same array (a NamedTuple IS "
                f"a tuple)", grads[i] is g)
        ref = _grad_of(want[name], kgrid, params)
        err = float(jnp.max(jnp.abs(g - ref)))
        scale = float(jnp.max(jnp.abs(ref))) or 1.0
        tol = 1e-12 if _precision.precision == "64" else 1e-4
        c.check(f"{eq} {name}: matches the gradient of the documented k-space field "
                f"(rel {err / scale:.2e})", err / scale < tol)


def test_terms_and_set_timestep_read_grads_by_name():
    """The consumers work on a plain tuple in the recipe's order too: the names are a
    label on the existing positional contract, not a new one."""
    with checks() as c:
        params, kgrid = _ctx()
        state = make_state(params, ic=_ic)
        named = rmhd.grad(state, kgrid, params)
        plain = rmhd.RMHDGrads(*tuple(named))
        for label, term in (("NonlinearTerm", rmhd.NonlinearTerm),
                            ("FDLinearTerm", rmhd.FDLinearTerm)):
            a = term(state, named, kgrid, params)
            b = term(state, plain, kgrid, params)
            c.check(f"RMHD {label} is bitwise the same on a rebuilt grads tuple",
                    bool(jnp.array_equal(a, b)))
        c.check("RMHD set_timestep is bitwise the same on a rebuilt grads tuple",
                bool(rmhd.set_timestep(named, params) == rmhd.set_timestep(plain, params)))

        params, kgrid = _ctx(**_GDI_2D)
        state = make_state(params, ic=_ic)
        named = gdi.grad(state, kgrid, params)
        plain = gdi.GDIGrads(*tuple(named))
        c.check("GDI NonlinearTerm is bitwise the same on a rebuilt grads tuple",
                bool(jnp.array_equal(gdi.NonlinearTerm(state, named, kgrid, params),
                                     gdi.NonlinearTerm(state, plain, kgrid, params))))
        c.check("GDI set_timestep is bitwise the same on a rebuilt grads tuple",
                bool(gdi.set_timestep(named, params) == gdi.set_timestep(plain, params)))


# ---------------------------------------------------------------- static term filtering

class _Counter:
    """A term func plus its predicate, both counting their calls."""

    def __init__(self, active):
        self._active = active
        self.calls = 0
        self.active_calls = 0

    def func(self, state, grads, kgrid, params, halo=None):
        self.calls += 1
        return jnp.zeros_like(state.fields)

    def active(self, params):
        self.active_calls += 1
        return self._active


def _toy_grad(state, kgrid, params):
    return state.fields


def _toy_set_timestep(grads, params):
    return params.dt


def _one_field_ic(x, y):
    return jnp.stack([jnp.cos(x) * jnp.cos(y)])


def test_construct_rhs_calls_only_active_terms():
    """The inactive term's func is never called; a bare callable is wrapped as always
    active."""
    on, off = _Counter(True), _Counter(False)
    bare_calls = []

    def bare(state, grads, kgrid, params, halo=None):
        bare_calls.append(1)
        return jnp.zeros_like(state.fields)

    recipe = EquationRecipe(set_timestep_func=_toy_set_timestep,
                            term_funcs=(Term(on.func, active=on.active),
                                        Term(off.func, active=off.active), bare),
                            grad_func=_toy_grad, nfields=1)
    with _registered("EQIFACE", recipe), checks() as c:
        params = fresh_params(dims=2, nx=8, ny=8, eqtype="EQIFACE", eqpars={})
        kgrid = jr.setup_kgrids(params)
        state = jr.initialize(_one_field_ic, params)
        out, grads = construct_rhs(recipe)(state, kgrid, params)
        c.check("the active Term's func ran once", on.calls == 1, f"{on.calls}")
        c.check("the inactive Term's func never ran", off.calls == 0, f"{off.calls}")
        c.check("both predicates were consulted",
                on.active_calls == 1 and off.active_calls == 1,
                f"{on.active_calls}, {off.active_calls}")
        c.check("a bare callable in term_funcs is treated as always active",
                len(bare_calls) == 1, f"{len(bare_calls)}")
        c.check("the RHS has the field shape and grads come back alongside it",
                out.shape == state.fields.shape and grads is not None)


def test_all_terms_inactive_is_a_clear_error():
    """A configuration where nothing is active would give an identically zero RHS: that
    is a trace-time ValueError, not a silent None."""
    off1, off2 = _Counter(False), _Counter(False)
    recipe = EquationRecipe(set_timestep_func=_toy_set_timestep,
                            term_funcs=(Term(off1.func, active=off1.active),
                                        Term(off2.func, active=off2.active)),
                            grad_func=_toy_grad, nfields=1)
    with _registered("EQIFACE", recipe), checks() as c:
        params = fresh_params(dims=2, nx=8, ny=8, eqtype="EQIFACE", eqpars={})
        kgrid = jr.setup_kgrids(params)
        state = jr.initialize(_one_field_ic, params)
        try:
            construct_rhs(recipe)(state, kgrid, params)
            raised = ""
        except ValueError as exc:
            raised = str(exc)
        c.check("every-term-inactive raises ValueError naming construct_rhs and the terms",
                "construct_rhs" in raised and "inactive" in raised, f"raised: {raised!r}")


def _explicit_sum(state, grads, kgrid, params, halo):
    """The unfiltered RHS: every RMHD term summed in registry order, zeros included."""
    total = rmhd.NonlinearTerm(state, grads, kgrid, params, halo)
    total = total + rmhd.FDLinearTerm(state, grads, kgrid, params, halo)
    return total + rmhd.ForcingTerm(state, grads, kgrid, params, halo)


def test_filtered_rhs_equals_the_all_terms_sum_bitwise():
    """Dropping an inactive term drops an `x + 0`, which changes no value."""
    recipe = equation_registry["RMHD"]
    rhs = construct_rhs(recipe)
    cells = [("2D forced (FDLinearTerm inactive)", dict(_FORCED_2D)),
             ("3D FD-z forced (nothing inactive)",
              dict(forcing=True, forcing_mode="elsasser",
                   forcing_power_elsasser=(1.0, 1.0), forcing_tau=1.0, fshell=(1, 3),
                   forcing_seed=1))]
    if mpi_size() == 1:
        cells.append(("3D z_spectral unforced (FDLinearTerm and ForcingTerm inactive)",
                      dict(z_spectral=True)))
    with checks() as c:
        for label, kw in cells:
            params, kgrid = _ctx(**kw)
            state = make_state(params, ic=_ic)
            got, grads = rhs(state, kgrid, params)
            # the same pre-issued halo (or None) construct_rhs hands its terms
            halo = (recipe.halo_start_func(state, kgrid, params)
                    if _halo_start_enabled(params) else None)
            want = _explicit_sum(state, grads, kgrid, params, halo)
            same = bool(jnp.array_equal(got, want))
            detail = "" if same else f"max|diff| {float(jnp.max(jnp.abs(got - want))):.3e}"
            c.check(f"{label}: filtered RHS is bitwise the all-terms sum", same, detail)


def test_predicates_run_once_per_trace_not_once_per_step():
    """params is static, so the filtering is python that happens while the block is
    traced: the predicate count must not grow with the step count, and a second call at
    the same static arguments (a cache hit) must not consult them at all."""
    on = _Counter(True)
    recipe = EquationRecipe(set_timestep_func=_toy_set_timestep,
                            term_funcs=(Term(on.func, active=on.active),),
                            grad_func=_toy_grad, nfields=1)
    with _registered("EQIFACE", recipe), checks() as c:
        params = fresh_params(dims=2, nx=8, ny=8, eqtype="EQIFACE", eqpars={},
                              adaptive_timestep=False, dt=0.01)
        kgrid = jr.setup_kgrids(params)
        stepper, scheme = get_scheme("lsrk33")

        on.active_calls = 0
        _advance(make_state(params, ic=_one_field_ic), kgrid, params, 2, scheme, stepper)
        few = on.active_calls
        _advance(make_state(params, ic=_one_field_ic), kgrid, params, 2, scheme, stepper)
        cached = on.active_calls - few
        on.active_calls = 0
        _advance(make_state(params, ic=_one_field_ic), kgrid, params, 32, scheme, stepper)
        many = on.active_calls

        c.check(f"the predicate ran a bounded number of times per trace ({few})",
                0 < few < 16, f"{few}")
        c.check("16x the steps consults the predicate exactly as often "
                f"({many} vs {few})", many == few, f"{many} vs {few}")
        c.check("a repeat call at the same static arguments retraces nothing "
                f"({cached} further calls)", cached == 0, f"{cached}")


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
