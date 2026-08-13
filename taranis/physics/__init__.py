from typing import NamedTuple,Tuple,Callable,Optional
from . import rmhd
from . import gdi
from .. import _precision

class EquationRecipe(NamedTuple):
    set_timestep_func: Callable
    term_funcs: Tuple[Callable,...]
    grad_func: Callable
    # number of fields; state.fields leading axis
    nfields: int
    # once-per-step forcing scale factor (params.forcing_norm_per_step)
    forcing_scale_func: Optional[Callable] = None
    # hook issuing the equation's halo exchange first
    halo_start_func: Optional[Callable] = None
    # k-local LINEAR operator L, with dt f = L f + N(f). 
    # built in grids.setup_kgrids; timesteppers apply exp(L*tau) through propagators
    # not in the RHS. no linear term: None.
    linear_matrix_func: Optional[Callable] = None


# backends where pre-issuing the halo might overlap comms with compute.
# so far doing this has not improved anything: but it could :)
_HALO_START_BACKENDS = ("jax", "serial")

def _halo_start_enabled(params):
    # check if halo start is on or off, overriden by params
    override = getattr(params, "halo_start", None)
    return params.comm_backend in _HALO_START_BACKENDS if override is None else bool(override)

# Constructs the ideal RHS of the equations using the relevant EquationRecipe.
# NB: *spectral* linear terms are handled in propagators.
def construct_rhs(recipe):
    def rhs(state,kgrid,params):
        # check that precision of the fields is correct.
        assert state.fields.dtype == _precision.ctype, (
            f"state.fields dtype {state.fields.dtype} != expected field dtype "
            f"{_precision.ctype} (RMHD_PRECISION={_precision.precision}) -- a strong-typed "
            "array leaked into field math upstream."
        )
        assert state.forcing_state.dtype == _precision.ctype, (
            f"state.forcing_state dtype {state.forcing_state.dtype} != expected field dtype "
            f"{_precision.ctype} (RMHD_PRECISION={_precision.precision}) -- a strong-typed "
            "array leaked into forcing math upstream."
        )
        use_halo = recipe.halo_start_func is not None and _halo_start_enabled(params)
        halo = recipe.halo_start_func(state,kgrid,params) if use_halo else None
        grads = recipe.grad_func(state,kgrid,params)
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