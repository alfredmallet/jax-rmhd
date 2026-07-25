from typing import NamedTuple,Tuple,Callable,Optional
from . import rmhd

class EquationRecipe(NamedTuple):
    set_timestep_func: Callable
    term_funcs: Tuple[Callable,...]
    grad_func: Callable
    # per-equation once-per-step forcing scale (params.forcing_norm_per_step); optional
    forcing_scale_func: Optional[Callable] = None
    # T7: optional hook issuing the equation's halo exchange first; its result is threaded
    # to every term func as a 5th argument (None when absent, e.g. dims=2)
    halo_start_func: Optional[Callable] = None


# Constructs the ideal RHS of the equations using the relevant EquationRecipe.
# NB: The dissipative terms are handled via integrating factor in timestepping.py
def construct_rhs(recipe):
    def rhs(state,kgrid,params):
        # T7: issue the halo exchange before the perpendicular FFT/bracket work
        halo = recipe.halo_start_func(state,kgrid,params) if recipe.halo_start_func is not None else None
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
                           forcing_scale_func = rmhd.forcing_scale,
                           # T7 (early halo issue) measured NO fp64 win on the mpi4jax CPU
                           # backend (-0.6% @32 ranks, +0.9% sub-noise @128) -> disabled per
                           # the Phase 2 revert rule. The hook + threading stay: re-enable
                           # with halo_start_func=rmhd.halo_start for backends with real
                           # comm/compute overlap (Phase 3 NCCL).
                           halo_start_func = None
                           ),
}