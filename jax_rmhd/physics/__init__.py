from typing import NamedTuple,Tuple,Callable,Optional
from . import rmhd
from . import gdi
from .. import _precision

class EquationRecipe(NamedTuple):
    set_timestep_func: Callable
    term_funcs: Tuple[Callable,...]
    grad_func: Callable
    # number of evolved fields (sets the leading axis of state.fields)
    nfields: int
    # per-equation once-per-step forcing scale (params.forcing_norm_per_step)
    forcing_scale_func: Optional[Callable] = None
    # hook issuing the equation's halo exchange first
    halo_start_func: Optional[Callable] = None
    # (kgrid, params) -> the k-local LINEAR operator L, with dt f = L f + N(f). Built once
    # in grids.setup_kgrids; the timesteppers apply exp(L*tau) through jax_rmhd.propagators
    # instead of summing it into the RHS. None: no linear operator (identity propagator).
    linear_matrix_func: Optional[Callable] = None


# backends where pre-issuing the halo might overlap comms with compute. A wash on every
# backend measured so far, but that depends on the scheme and the GPU, so the hook stays.
# "serial" costs nothing extra either way (halo_exchange is a plain self-slice, no comm token
# to schedule), so it's on: test_z_derivatives_halo_width_invariance/
# test_halo_start_matches_bare_width2_exchange (tests/test_halo_width.py) show the pre-issued
# vs fallback path is bitwise-identical unconditionally, not just under mpi4jax/jax.
_HALO_START_BACKENDS = ("jax", "serial")

def _halo_start_enabled(params):
    # check if halo start is on or off, overriden by params
    override = getattr(params, "halo_start", None)
    return params.comm_backend in _HALO_START_BACKENDS if override is None else bool(override)

# Constructs the ideal RHS of the equations using the relevant EquationRecipe.
# NB: The dissipative terms are handled via integrating factor in timestepping.py
def construct_rhs(recipe):
    def rhs(state,kgrid,params):
        # Tripwire (plans/PRECISION_PLAN.md A3): dtype is static metadata on a traced array,
        # so these are plain python asserts checked once per unique trace -- zero runtime
        # overhead, not baked into the compiled graph. They catch a dtype leak (a strong
        # float64/complex128 array multiplying fields under x64) right here instead of it
        # surfacing later as a silent ~2x fp32 slowdown or a precision regression.
        assert state.fields.dtype == _precision.ctype, (
            f"state.fields dtype {state.fields.dtype} != expected field dtype "
            f"{_precision.ctype} (RMHD_PRECISION={_precision.precision}) -- a strong-typed "
            "array leaked into field math upstream; see plans/PRECISION_PLAN.md A2/A3."
        )
        assert state.forcing_state.dtype == _precision.ctype, (
            f"state.forcing_state dtype {state.forcing_state.dtype} != expected field dtype "
            f"{_precision.ctype} (RMHD_PRECISION={_precision.precision}) -- a strong-typed "
            "array leaked into forcing math upstream; see plans/PRECISION_PLAN.md A2/A3."
        )
        use_halo = recipe.halo_start_func is not None and _halo_start_enabled(params)
        halo = recipe.halo_start_func(state,kgrid,params) if use_halo else None
        grads=recipe.grad_func(state,kgrid,params)
        fields_rhs = None
        for term in recipe.term_funcs:
            if fields_rhs is None:
                fields_rhs = term(state,grads,kgrid,params,halo)
            else:
                fields_rhs = fields_rhs + term(state,grads,kgrid,params,halo)
        return fields_rhs, grads
    return rhs

equation_registry = {
    "RMHD": EquationRecipe(set_timestep_func = rmhd.set_timestep,
                           term_funcs = (rmhd.NonlinearTerm, rmhd.LinearTerm, rmhd.ForcingTerm),
                           grad_func = rmhd.grad,
                           nfields = 2,
                           forcing_scale_func = rmhd.forcing_scale,
                           halo_start_func = rmhd.halo_start,
                           linear_matrix_func = rmhd.linear_matrix
                           ),
    "GDI": EquationRecipe(set_timestep_func = gdi.set_timestep,
                          term_funcs = (gdi.NonlinearTerm,),
                          grad_func = gdi.grad,
                          nfields = 2,
                          linear_matrix_func = gdi.linear_matrix
                          ),
}