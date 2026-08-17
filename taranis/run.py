import jax
import jax.numpy as jnp
from functools import partial
from jax.sharding import PartitionSpec as P
from .timestepping import get_scheme
from .snapshot_io import save_snapshot, get_saved_steps
from time import perf_counter
from .physics import equation_registry, construct_rhs
from .physics.shared_physics import ou_update
from .types import SimulationState
from .grids import fft, local_z_coords, dealias_mask
from . import comms
from . import _precision

def initialize(func,params):
    # use this to initialize with some known function.
    # func should be a function that sets ALL fields in the problem, in real space.
    @partial(jax.jit,static_argnums=(0,))
    def _init(f):
        x = jnp.linspace(0, params.Lx, params.nx, endpoint=False,
                         dtype=_precision.ftype).reshape(1,-1,1)
        y = jnp.linspace(0, params.Ly, params.ny, endpoint=False,
                         dtype=_precision.ftype).reshape(1,1,-1)
        if params.spatial_dimensions==3:
            z_device = local_z_coords(params).reshape(-1,1,1)
            fields = fft(f(x,y,z_device),params) * dealias_mask(params)
        else:
            fields = fft(f(x,y),params) * dealias_mask(params)
        fields = fields.astype(_precision.ctype)
        nkx, nky = params.nx, params.ny//2 + 1
        forcing_state = jnp.zeros((params.n_ou, 2, nkx, nky), dtype=fields.dtype)
        forcing_key = jax.random.key(params.forcing_seed)
        forcing_scale = jnp.zeros((params.n_ou,), dtype=_precision.ftype)
        # t deliberately float64 at every precision
        return SimulationState(t=jnp.float64(0.0),fields=fields,forcing_state=forcing_state,
                               forcing_key=forcing_key,forcing_scale=forcing_scale)
    state = _init(func)
    if params.comm_backend == "jax":
        state = comms.state_to_global(state, params)  # process-local -> z-sharded global arrays
    return state

def _advance_forcing(new_state, prev_t, kgrid, params):
    # Per-full-step forcing update: OU advance plus (if forcing_norm_per_step) the
    # power-normalization scale reused across all sub-stages of the next step.
    dt = (new_state.t - prev_t).astype(_precision.ftype)
    new_forcing_state, new_forcing_key = ou_update(
        new_state.forcing_state, new_state.forcing_key, dt, params, kgrid
    )
    new_state = new_state._replace(forcing_state=new_forcing_state, forcing_key=new_forcing_key)
    if params.forcing_norm_per_step:
        scale_func = equation_registry[params.eqtype].forcing_scale_func
        new_state = new_state._replace(forcing_scale=scale_func(new_state, kgrid, params, dt))
    return new_state

def _refresh_forcing_scale(state, kgrid, params):
    # recompute the per-step scale for the initial state. Checkpoints do store
    # forcing_scale (recomputing is then a no-op), but repaired legacy snapshots carry
    # zeros and hand-built states may be stale — so recomputes.
    if params.forcing and params.forcing_norm_per_step:
        scale_func = equation_registry[params.eqtype].forcing_scale_func
        if params.comm_backend == "jax":
            # the psum inside needs a shard_map context
            f = comms.shard_call(lambda s,kg: scale_func(s,kg,params,0.0), params, kgrid, out_specs=P())
            state = state._replace(forcing_scale=jax.jit(f)(state, kgrid))
        else:
            state = state._replace(forcing_scale=scale_func(state, kgrid, params, 0.0))
    return state

#this can be used to estimate a good nblock. You can set the minimum higher.
def estimate_good_nblock(state,kgrid,params,t_snap,t_end,t_last_snap=0,nblock_min=10):
    # attribute access (not tuple unpack) so EquationRecipe can grow fields
    recipe = equation_registry[params.eqtype]
    set_timestep, grad = recipe.set_timestep_func, recipe.grad_func
    if params.comm_backend == "jax":
        # the pmax inside set_timestep needs a shard_map context
        f = comms.shard_call(lambda s,kg: set_timestep(grad(s,kg,params),params), params, kgrid, out_specs=P())
        dt = jax.jit(f)(state,kgrid)
    else:
        grads = grad(state,kgrid,params)
        dt = set_timestep(grads,params)
    t_next_snap = min(t_last_snap+t_snap,t_end)
    nblock_estimate = max((t_next_snap-state.t)/dt,nblock_min)
    return int(nblock_estimate)

def _use_cfl_blocks(params):
    # block the CFL reduction only when it can pay: with fixed dt there's no reduction anyway.
    return params.cfl_every > 1 and params.adaptive_timestep

def _block_dt(state,kgrid,params):
    # one global CFL allreduce for a whole block, from the block's starting state
    # (same grad_func + set_timestep_func path as estimate_good_nblock; never rank-local).
    recipe = equation_registry[params.eqtype]
    return recipe.set_timestep_func(recipe.grad_func(state,kgrid,params),params)

def _cfl_block(state,kgrid,params,rhs,set_timestep,scheme,stepper):
    # params.cfl_every full steps sharing one dt; forcing still advances every step.
    dt = _block_dt(state,kgrid,params)
    def stepping(state,_):
        new_state = stepper(state,kgrid,params,rhs,set_timestep,scheme,dt)
        if params.forcing:
            new_state = _advance_forcing(new_state, state.t, kgrid, params)
        return new_state, None
    final_state,_ = jax.lax.scan(stepping,state,None,params.cfl_every)
    return final_state

def block_of_steps(state,kgrid,params,nblock,scheme,stepper):
    recipe = equation_registry[params.eqtype]
    set_timestep = recipe.set_timestep_func
    rhs = construct_rhs(recipe)
    if _use_cfl_blocks(params):
        # nblock rounded up to a whole multiple of cfl_every: dt frozen per block.
        def block(state,_):
            return _cfl_block(state,kgrid,params,rhs,set_timestep,scheme,stepper), None
        final_state,_ = jax.lax.scan(block,state,None,-(-nblock//params.cfl_every))
        return final_state
    def stepping(state,_):
        new_state = stepper(state,kgrid,params,rhs,set_timestep,scheme)
        # advance the O-U forcing state (and per-step norm scale) once per full timestep
        if params.forcing:
            new_state = _advance_forcing(new_state, state.t, kgrid, params)
        return new_state, None
    final_state,_ = jax.lax.scan(stepping,state,None,nblock)
    return final_state

def _write_snapshot(snap,state,params,mngr,save,label="snapshot"):
    # one snapshot write, drained before returning; returns the next free index
    if not save:
        return snap
    if params.rank==0:
        print(f"Saving {label} as snapshot {snap} at t = {float(state.t)}")
    save_snapshot(snap,state,mngr,params)
    mngr.wait_until_finished()
    return snap+1

def _start_snapshots(state,params,mngr,save):
    # next free index, enumerated from disk and broadcast from rank 0, with the initial
    # state written as the first snapshot
    snap=max(get_saved_steps(mngr.directory), default=-1)+1
    if params.size>1:
        snap = params.comm.bcast(snap, root=0)
    return _write_snapshot(snap,state,params,mngr,save,"initial state")

def _finish(state,params,t_start):
    mngr_time = perf_counter()-t_start
    if params.rank==0:
        print("Ending simulation at t = "+str(state.t)+". It took "+str(mngr_time)+"s")
    return state

def simulate_scan(state,kgrid,params,nblock,t_snap,t_end,mngr,schemestr='lsrk33',save=True,print_every=1):
    # this simulates repeated fixed number of timesteps
    # for automatic differentiation sometime in the future
    # we should set nblock using the helper function estimate_good_nblock
    t_start = perf_counter()
    stepper,scheme = get_scheme(schemestr)
    state = _refresh_forcing_scale(state, kgrid, params)
    # donate_argnums=(0,): caller's input `state` buffer is consumed/reused for the output, since we always reassign `state` from the return value below.
    if params.comm_backend == "jax":
        # same stepper, wrapped in the z-mesh shard_map that the collectives need
        advance_jit = jax.jit(comms.shard_call(
            lambda s,kg: block_of_steps(s,kg,params,nblock,scheme,stepper), params, kgrid),
            donate_argnums=(0,))
        advance = lambda s: advance_jit(s,kgrid)
    else:
        block_of_steps_jit = jax.jit(block_of_steps,static_argnums=(2,3,4,5),donate_argnums=(0,))
        advance = lambda s: block_of_steps_jit(s,kgrid,params,nblock,scheme,stepper)
    # float(): pull to host so this doesn't alias state.t's buffer, which donate_argnums frees on the next jit call
    t_last_snapshot = float(state.t)
    snap = _start_snapshots(state,params,mngr,save)
    block_count=0
    saved_current=True
    while state.t<t_end:
        state = advance(state)
        block_count+=1
        saved_current=False
        if params.rank==0 and block_count%print_every==0:
            print(state.t) #this doesnt affect performance; state.t already on host from while
        if state.t - t_last_snapshot > t_snap:
            snap = _write_snapshot(snap,state,params,mngr,save)
            t_last_snapshot=float(state.t)
            saved_current=True
    if not saved_current:
        snap = _write_snapshot(snap,state,params,mngr,save,"final state")
    mngr.wait_until_finished()
    return _finish(state,params,t_start)

def simulate(initial_state,kgrid,params,t_snap,t_end,mngr,schemestr='lsrk33',save=True,print_every=1):
    t_start = perf_counter()
    stepper,scheme = get_scheme(schemestr)
    set_timestep = equation_registry[params.eqtype].set_timestep_func
    rhs = construct_rhs(equation_registry[params.eqtype])
    # kgrid is an explicit argument (not a closure) so the jax backend can hand it to
    # shard_map with its own in_specs; the mpi4jax branch below re-closes over it unchanged.
    def stepper_wrapped(state,kgrid):
        new_state = stepper(state,kgrid,params,rhs,set_timestep,scheme)
        if params.forcing:
            new_state = _advance_forcing(new_state, state.t, kgrid, params)
        return new_state
    def block_wrapped(state,kgrid):
        return _cfl_block(state,kgrid,params,rhs,set_timestep,scheme,stepper)
    # cfl_every>1: the while_loop iterates over whole blocks, so it can overshoot the
    # snapshot target by up to cfl_every-1 extra steps (on top of the usual one-step overshoot)
    loop_body = block_wrapped if _use_cfl_blocks(params) else stepper_wrapped
    def sim_to_next_snap(state,kgrid,target_t):
        def snap_cond(state):
            return state.t<target_t
        return jax.lax.while_loop(snap_cond,lambda s: loop_body(s,kgrid),state)
    # donate_argnums=(0,): caller's input `state` buffer is consumed/reused for the output, since we always reassign `state` from the return value below.
    if params.comm_backend == "jax":
        _jit = jax.jit(comms.shard_call(sim_to_next_snap,params,kgrid,nextra=1),donate_argnums=(0,))
        sim_to_next_snap_jit = lambda s,target_t: _jit(s,kgrid,target_t)
    else:
        sim_to_next_snap_jit = jax.jit(lambda state,target_t: sim_to_next_snap(state,kgrid,target_t),
                                       donate_argnums=(0,))
    state=_refresh_forcing_scale(initial_state, kgrid, params)
    # float(): pull to host so this doesn't alias state.t's buffer, which donate_argnums frees on the next jit call
    t_last_snapshot = float(state.t)
    snap = _start_snapshots(state,params,mngr,save)

    while state.t<t_end:
        state = sim_to_next_snap_jit(state,min(t_last_snapshot+t_snap,t_end))
        snap = _write_snapshot(snap,state,params,mngr,save)
        t_last_snapshot=float(state.t)
    mngr.wait_until_finished()
    return _finish(state,params,t_start)

