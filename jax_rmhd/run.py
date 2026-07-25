import jax
import jax.numpy as jnp
from functools import partial
from .timestepping import get_scheme
from .snapshot_io import save_snapshot
from time import perf_counter
from .physics import equation_registry, construct_rhs
from .physics.shared_physics import ou_update
from .types import SimulationState
from .grids import fft, local_z_coords

def initialize(func,params):
    # use this to initialize with some known function.
    # func should be a function that sets ALL fields in the problem, in real space.
    @partial(jax.jit,static_argnums=(0,))
    def _init(f):
        x = jnp.linspace(0, params.Lx, params.nx, endpoint=False).reshape(1,-1,1)
        y = jnp.linspace(0, params.Ly, params.ny, endpoint=False).reshape(1,1,-1)
        if params.spatial_dimensions==3:
            z_device = local_z_coords(params).reshape(-1,1,1)
            fields = fft(f(x,y,z_device))
        else:
            fields = fft(f(x,y))
        nkx, nky = params.nx, params.ny//2 + 1
        forcing_state = jnp.zeros((params.n_ou, 2, nkx, nky), dtype=fields.dtype)
        forcing_key = jax.random.key(params.forcing_seed)
        # forcing_scale is ALWAYS a concrete (n_ou,) array (zeros when unused) so every
        # SimulationState — and therefore every checkpoint — has one uniform pytree
        # structure regardless of forcing/forcing_norm_per_step settings.
        forcing_scale = jnp.zeros((params.n_ou,))
        return SimulationState(t=0.0,fields=fields,forcing_state=forcing_state,forcing_key=forcing_key,
                               forcing_scale=forcing_scale)
    return _init(func)

def _advance_forcing(new_state, prev_t, kgrid, params):
    # Per-full-step forcing update: OU advance plus, when forcing_norm_per_step, the
    # power-normalization scale reused across all sub-stages of the next step.
    dt = new_state.t - prev_t
    new_forcing_state, new_forcing_key = ou_update(
        new_state.forcing_state, new_state.forcing_key, dt, params, kgrid
    )
    new_state = new_state._replace(forcing_state=new_forcing_state, forcing_key=new_forcing_key)
    if params.forcing_norm_per_step:
        scale_func = equation_registry[params.eqtype].forcing_scale_func
        new_state = new_state._replace(forcing_scale=scale_func(new_state, kgrid, params))
    return new_state

def _refresh_forcing_scale(state, kgrid, params):
    # Recompute the per-step scale for the initial state. Checkpoints do store
    # forcing_scale (recomputing is then a no-op), but repaired legacy snapshots carry
    # zeros and hand-built states may be stale — one cheap recompute covers all cases.
    if params.forcing and params.forcing_norm_per_step:
        scale_func = equation_registry[params.eqtype].forcing_scale_func
        state = state._replace(forcing_scale=scale_func(state, kgrid, params))
    return state

#This can be used to estimate a good nblock. You can set the minimum higher.
def estimate_good_nblock(state,kgrid,params,t_snap,t_end,t_last_snap=0,nblock_min=10):
    # attribute access (not tuple unpack) so EquationRecipe can grow fields
    recipe = equation_registry[params.eqtype]
    set_timestep, grad = recipe.set_timestep_func, recipe.grad_func
    grads = grad(state,kgrid,params)
    dt = set_timestep(grads,params)
    t_next_snap = min(t_last_snap+t_snap,t_end)
    nblock_estimate = max((t_next_snap-state.t)/dt,nblock_min)
    return int(nblock_estimate)

def _use_cfl_blocks(params):
    # Block the CFL reduction only when it can pay: with fixed dt there's no reduction anyway.
    return params.cfl_every > 1 and params.adaptive_timestep

def _block_dt(state,kgrid,params):
    # One global CFL allreduce for a whole block, from the block's starting state
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
    if _use_cfl_blocks(params):
        # nblock still counts STEPS, but is rounded UP to a whole number of cfl_every-step
        # blocks (so nblock=10, cfl_every=4 runs 12 steps): dt is frozen per block.
        set_timestep = equation_registry[params.eqtype].set_timestep_func
        rhs = construct_rhs(equation_registry[params.eqtype])
        def block(state,_):
            return _cfl_block(state,kgrid,params,rhs,set_timestep,scheme,stepper), None
        final_state,_ = jax.lax.scan(block,state,None,-(-nblock//params.cfl_every))
        return final_state,None
    def stepping(state,_):
        set_timestep = equation_registry[params.eqtype].set_timestep_func
        rhs = construct_rhs(equation_registry[params.eqtype])
        new_state = stepper(state,kgrid,params,rhs,set_timestep,scheme)
        # Advance the O-U forcing state (and per-step norm scale) exactly once per full timestep
        if params.forcing:
            new_state = _advance_forcing(new_state, state.t, kgrid, params)
        return new_state, None
    final_state,_ = jax.lax.scan(stepping,state,None,nblock)
    return final_state,None

#currently an orbax checkpoint mngr must be set outside of the simulate function
#this makes it a little easier to set up snapshots etc but could be changed

def simulate_scan(state,kgrid,params,nblock,t_snap,t_end,mngr,schemestr='lsrk33',save=True,print_every=1):
    # this simulates repeated fixed number of timesteps
    # for automatic differentiation sometime in the future
    # we should set nblock using the helper function estimate_good_nblock
    t_start = perf_counter()
    stepper,scheme = get_scheme(schemestr)
    state = _refresh_forcing_scale(state, kgrid, params)
    # donate_argnums=(0,): caller's input `state` buffer is consumed/reused for the output, since we always reassign `state` from the return value below.
    block_of_steps_jit = jax.jit(block_of_steps,static_argnums=(2,3,4,5),donate_argnums=(0,))
    # float(): pull to host so this doesn't alias state.t's buffer, which donate_argnums frees on the next jit call
    t_last_snapshot = float(state.t)
    snap=max(mngr.all_steps(), default=-1)+1
    if params.size>1:
        snap = params.comm.bcast(snap, root=0)
    if save:
        if params.rank==0:
            print("Saving initial state as snapshot "+str(snap))
        save_snapshot(snap,state,mngr)
        mngr.wait_until_finished()
    block_count=0
    while state.t<t_end:
        state, _ = block_of_steps_jit(state,kgrid,params,nblock,scheme,stepper)
        block_count+=1
        # NB the while/snapshot conditions above already sync state.t to host every block;
        # print_every only saves the print itself, not a device sync
        if params.rank==0 and block_count%print_every==0:
            print(state.t)
        if state.t - t_last_snapshot > t_snap and save:
            snap=snap+1
            if params.rank==0:
                print("Saving snapshot "+str(snap))
            save_snapshot(snap,state,mngr)
            mngr.wait_until_finished()
            t_last_snapshot=float(state.t)
    snap=snap+1
    if save:
        if params.rank==0:
            print("Saving final state as snapshot "+str(snap))
        save_snapshot(snap,state,mngr)
    mngr.wait_until_finished()
    t_sim = perf_counter()-t_start
    if params.rank==0:
        print("Ending simulation at t = " + str(state.t)+". It took "+str(t_sim)+"s")
    return state

def simulate(initial_state,kgrid,params,t_snap,t_end,mngr,schemestr='lsrk33',save=True,print_every=1):
    t_start = perf_counter()
    stepper,scheme = get_scheme(schemestr)
    set_timestep = equation_registry[params.eqtype].set_timestep_func
    rhs = construct_rhs(equation_registry[params.eqtype])
    def stepper_wrapped(state):
        new_state = stepper(state,kgrid,params,rhs,set_timestep,scheme)
        if params.forcing:
            new_state = _advance_forcing(new_state, state.t, kgrid, params)
        return new_state
    def block_wrapped(state):
        return _cfl_block(state,kgrid,params,rhs,set_timestep,scheme,stepper)
    # cfl_every>1: the while_loop iterates over whole blocks, so it can overshoot the
    # snapshot target by up to cfl_every-1 extra steps (on top of the usual one-step overshoot)
    loop_body = block_wrapped if _use_cfl_blocks(params) else stepper_wrapped
    def sim_to_next_snap(state,target_t):
        def snap_cond(state):
            return state.t<target_t
        return jax.lax.while_loop(snap_cond,loop_body,state)
    # donate_argnums=(0,): caller's input `state` buffer is consumed/reused for the output, since we always reassign `state` from the return value below.
    sim_to_next_snap_jit = jax.jit(sim_to_next_snap,donate_argnums=(0,))
    # print_every: simulate has no non-snapshot prints currently, kept for API parity with simulate_scan; snapshot prints below stay unconditional.
    state=_refresh_forcing_scale(initial_state, kgrid, params)
    # float(): pull to host so this doesn't alias state.t's buffer, which donate_argnums frees on the next jit call
    t_last_snapshot = float(state.t)
    snap=max(mngr.all_steps(), default=-1)+1
    if params.size>1:
        snap = params.comm.bcast(snap, root=0)
    if save:
        if params.rank==0:
            print("Saving initial state as snapshot "+str(snap))
        save_snapshot(snap,state,mngr)
        mngr.wait_until_finished()
    while state.t<t_end:
        t_next_snapshot=min(t_last_snapshot+t_snap,t_end)
        state = sim_to_next_snap_jit(state,t_next_snapshot)
        snap=snap+1
        if save:
            if params.rank==0:
                print ("Saving snapshot "+str(snap)+ " at t = "+str(state.t))
            save_snapshot(snap,state,mngr)
            mngr.wait_until_finished()
        # advance the snapshot target unconditionally: leaving this inside `if save:`
        # froze t_next_snapshot with save=False, so once state.t passed the first target
        # the inner while_loop returned immediately and this loop spun forever
        t_last_snapshot=float(state.t)
    mngr.wait_until_finished()
    t_sim = perf_counter()-t_start
    if params.rank==0:
        print(f"Ending simulation at t = "+str(state.t)+". It took "+str(t_sim)+"s")
    return state

