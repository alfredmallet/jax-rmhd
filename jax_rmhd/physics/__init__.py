from typing import NamedTuple,Tuple,Callable,Optional
from . import rmhd

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


# backends for which halo start *might* work better
# in current testing, it makes no difference for any backend
# keeping the infrastructure for future tweaking
_HALO_START_BACKENDS = ("jax",)

def _halo_start_enabled(params):
    # check if halo start is on or off, overriden by params
    override = getattr(params, "halo_start", None)
    return params.comm_backend in _HALO_START_BACKENDS if override is None else bool(override)

# Constructs the ideal RHS of the equations using the relevant EquationRecipe.
# NB: The dissipative terms are handled via integrating factor in timestepping.py
def construct_rhs(recipe):
    def rhs(state,kgrid,params):
        use_halo = recipe.halo_start_func is not None and _halo_start_enabled(params)
        halo = recipe.halo_start_func(state,kgrid,params) if use_halo else None
        grads=recipe.grad_func(state,kgrid,params)
        fields_rhs = None
        for term in recipe.term_funcs:
            if fields_rhs is None:
                fields_rhs = term(state,grads,kgrid,params,halo)  # T7: halo threaded to the terms
            else:
                fields_rhs = fields_rhs + term(state,grads,kgrid,params,halo)  # T7
        return fields_rhs, grads
    return rhs

equation_registry = {
    "RMHD": EquationRecipe(set_timestep_func = rmhd.set_timestep,
                           term_funcs = (rmhd.NonlinearTerm, rmhd.LinearTerm, rmhd.ForcingTerm),
                           grad_func = rmhd.grad,
                           nfields = 2,
                           forcing_scale_func = rmhd.forcing_scale,
                           halo_start_func = rmhd.halo_start
                           ),
}