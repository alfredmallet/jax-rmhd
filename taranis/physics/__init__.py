from typing import NamedTuple,Tuple,Callable,Optional,Union
from . import rmhd
from . import gdi
from .. import _precision

class Term(NamedTuple):
    # an RHS contribution plus the static predicate deciding whether this configuration
    # has it. active(params) is plain python run at trace time, so an inactive term never
    # reaches the graph.
    func: Callable                          # (state, grads, kgrid, params, halo) -> rhs
    active: Callable = lambda params: True

class EquationRecipe(NamedTuple):
    set_timestep_func: Callable
    # Term entries, or bare callables with the same signature (always active)
    term_funcs: Tuple[Union[Term,Callable],...]
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
    terms = tuple(t if isinstance(t,Term) else Term(t) for t in recipe.term_funcs)
    def rhs(state,kgrid,params):
        # check that precision of the fields is correct.
        assert state.fields.dtype == _precision.ctype, (
            f"state.fields dtype {state.fields.dtype} != expected field dtype "
            f"{_precision.ctype} (TARANIS_PRECISION={_precision.precision}) -- a strong-typed "
            "array leaked into field math upstream."
        )
        assert state.forcing_state.dtype == _precision.ctype, (
            f"state.forcing_state dtype {state.forcing_state.dtype} != expected field dtype "
            f"{_precision.ctype} (TARANIS_PRECISION={_precision.precision}) -- a strong-typed "
            "array leaked into forcing math upstream."
        )
        # params is static, so this selection is python and the inactive terms cost nothing
        active = [t.func for t in terms if t.active(params)]
        if not active:
            raise ValueError(
                "construct_rhs: every term in term_funcs is inactive for these parameters, "
                "so the RHS would be identically zero -- check the term predicates against "
                "the configuration. terms: "
                f"{[getattr(t.func,'__name__',t.func) for t in terms]}")
        use_halo = recipe.halo_start_func is not None and _halo_start_enabled(params)
        halo = recipe.halo_start_func(state,kgrid,params) if use_halo else None
        grads = recipe.grad_func(state,kgrid,params)
        fields_rhs = None
        for term in active:
            if fields_rhs is None:
                fields_rhs = term(state,grads,kgrid,params,halo)
            else:
                fields_rhs = fields_rhs + term(state,grads,kgrid,params,halo)
        return fields_rhs, grads
    return rhs

equation_registry = {
    "RMHD": EquationRecipe(set_timestep_func = rmhd.set_timestep,
                           term_funcs = (Term(rmhd.NonlinearTerm),
                                         Term(rmhd.FDLinearTerm, active=rmhd.fd_linear_active),
                                         Term(rmhd.ForcingTerm, active=rmhd.forcing_active)),
                           grad_func = rmhd.grad,
                           nfields = 2,
                           forcing_scale_func = rmhd.forcing_scale,
                           halo_start_func = rmhd.halo_start,
                           linear_matrix_func = rmhd.linear_matrix
                           ),
    "GDI": EquationRecipe(set_timestep_func = gdi.set_timestep,
                          term_funcs = (Term(gdi.NonlinearTerm),),
                          grad_func = gdi.grad,
                          nfields = 2,
                          linear_matrix_func = gdi.linear_matrix
                          ),
}