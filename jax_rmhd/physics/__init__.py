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


# Backends where a registered halo_start_func actually pays: measured no fp64 win on the
# mpi4jax CPU backend (T7), but overlap is the point of NCCL/shard_map (T9).
_HALO_START_BACKENDS = ("jax",)

def _halo_start_enabled(params):
    # per-backend default, overridable with params.halo_start=True/False so the benchmark
    # can measure the T7 on/off pair on ANY backend instead of silently no-op'ing
    override = getattr(params, "halo_start", None)
    return params.comm_backend in _HALO_START_BACKENDS if override is None else bool(override)

# Constructs the ideal RHS of the equations using the relevant EquationRecipe.
# NB: The dissipative terms are handled via integrating factor in timestepping.py
def construct_rhs(recipe):
    def rhs(state,kgrid,params):
        # T7/T9: pre-issue the halo before the perpendicular FFT/bracket work, but only on
        # backends with real comm/compute overlap (plain-python branch on the static param)
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
                           forcing_scale_func = rmhd.forcing_scale,
                           # T7 (early halo issue) measured NO fp64 win on the mpi4jax CPU
                           # backend (-0.6% @32 ranks, +0.9% sub-noise @128); construct_rhs
                           # therefore only USES this hook on _HALO_START_BACKENDS ("jax").
                           halo_start_func = rmhd.halo_start
                           ),
}