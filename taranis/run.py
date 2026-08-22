import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import PartitionSpec as P
from .timestepping import get_scheme, stage_exp_ops
from .snapshot_io import save_snapshot, get_saved_steps
from time import perf_counter
from .physics import equation_registry, construct_rhs
from .physics.shared_physics import ou_update
from .particles.state import MOMENTS, push_ensembles
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

def _advance_particles(pstate, prev_state, new_state, kgrid, params):
    # One particle push per full step, mirroring _advance_forcing's placement: dt is the
    # step just taken, the fields are assembled from the PRE-step state (frozen at t_n).
    # Returns (pstate, per-ensemble moments).
    dt = (new_state.t - prev_state.t).astype(jnp.float64)
    return push_ensembles(pstate, prev_state, kgrid, params, dt)

def _refresh_forcing_scale(state, kgrid, params):
    # compute the per-step scale at dt = 0 only for a state that carries none -- an
    # all-zero forcing_scale, which is what initialize and a repaired legacy snapshot
    # produce. A nonzero stored scale is the lagged one the next step uses, so it is kept.
    if params.forcing and params.forcing_norm_per_step:
        if not bool(jnp.all(state.forcing_scale == 0)):
            return state
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

def _hoisted_exp_ops(kgrid,params,scheme,stepper,dt):
    # exp(L*tau) per stage, formed ONCE here for every step that shares dt (a fixed-dt block,
    # or one cfl_every block): evaluated outside the step scan, closed over by it. None means
    # the stepper forms them itself (timestepping.stage_exp_ops).
    return stage_exp_ops(kgrid,params,scheme,stepper,dt)

def _fixed_dt(params):
    # the frozen dt of the plain (non-cfl-block) step scan, or None when dt is adaptive
    return None if params.adaptive_timestep else params.dt

def _step(carry,kgrid,params,rhs,set_timestep,scheme,stepper,dt_override=None,exp_ops=None):
    # One full timestep on the run carry (state, aux), aux being the ParticleState when
    # particles are on and None off (a leafless pytree: scan/while_loop/donation ignore it).
    # Returns (carry, ys) with ys = (post-step t, per-ensemble moments) on, None off.
    state,aux = carry
    new_state = stepper(state,kgrid,params,rhs,set_timestep,scheme,dt_override,exp_ops)
    # advance the O-U forcing state (and per-step norm scale) once per full timestep
    if params.forcing:
        new_state = _advance_forcing(new_state, state.t, kgrid, params)
    if params.particles is not None:
        aux,mom = _advance_particles(aux, state, new_state, kgrid, params)
        return (new_state,aux), (new_state.t, mom)
    return (new_state,aux), None

def _cfl_block(carry,kgrid,params,rhs,set_timestep,scheme,stepper):
    # params.cfl_every full steps sharing one dt; forcing still advances every step.
    dt = _block_dt(carry[0],kgrid,params)
    exp_ops = _hoisted_exp_ops(kgrid,params,scheme,stepper,dt)
    def stepping(carry,_):
        return _step(carry,kgrid,params,rhs,set_timestep,scheme,stepper,dt,exp_ops)
    return jax.lax.scan(stepping,carry,None,params.cfl_every)

def _advance_block(carry,kgrid,params,nblock,scheme,stepper):
    # nblock full steps on the (state, aux) carry, ys stacked over every step.
    recipe = equation_registry[params.eqtype]
    set_timestep = recipe.set_timestep_func
    rhs = construct_rhs(recipe)
    if _use_cfl_blocks(params):
        # nblock rounded up to a whole multiple of cfl_every: dt frozen per block.
        def block(carry,_):
            return _cfl_block(carry,kgrid,params,rhs,set_timestep,scheme,stepper)
        carry,ys = jax.lax.scan(block,carry,None,-(-nblock//params.cfl_every))
        # (nblocks, cfl_every, ...) -> (nsteps, ...); a no-op on the particles-off None
        return carry, jax.tree.map(lambda y: y.reshape((-1,)+y.shape[2:]), ys)
    # fixed dt: the stage propagators are the same for every step of the block -> hoist
    dt = _fixed_dt(params)
    exp_ops = None if dt is None else _hoisted_exp_ops(kgrid,params,scheme,stepper,dt)
    def stepping(carry,_):
        return _step(carry,kgrid,params,rhs,set_timestep,scheme,stepper,None,exp_ops)
    return jax.lax.scan(stepping,carry,None,nblock)

def block_of_steps(x,kgrid,params,nblock,scheme,stepper):
    # public driver: particles off, state in -> state out; on, (state, pstate) in ->
    # ((state, pstate), ys) out.
    if params.particles is not None:
        return _advance_block(x,kgrid,params,nblock,scheme,stepper)
    return _advance_block((x,None),kgrid,params,nblock,scheme,stepper)[0][0]

def _write_snapshot(snap,state,params,mngr,save,label="snapshot",pstate=None):
    # one snapshot write, drained before returning; returns the next free index
    if not save:
        return snap
    if params.rank==0:
        print(f"Saving {label} as snapshot {snap} at t = {float(state.t)}")
    save_snapshot(snap,state,mngr,params,pstate)
    mngr.wait_until_finished()
    return snap+1

def _start_snapshots(state,params,mngr,save,pstate=None):
    # next free index, enumerated from disk and broadcast from rank 0, with the initial
    # state written as the first snapshot
    snap=max(get_saved_steps(mngr.directory), default=-1)+1
    if params.size>1:
        snap = params.comm.bcast(snap, root=0)
    return _write_snapshot(snap,state,params,mngr,save,"initial state",pstate)

_MOMENTS_FILE = "particle_moments.txt"

def _append_moments(directory,ys):
    # one row per (step, ensemble) appended to the snapshot dir's sidecar; the header is
    # written when the file is created, so a restart just appends.
    path = os.path.join(str(directory),_MOMENTS_FILE)
    t,mom = np.asarray(ys[0]),np.asarray(ys[1])
    header = not os.path.exists(path)
    with open(path,"a") as f:
        if header:
            f.write("# t ensemble "+" ".join(MOMENTS)+"\n")
        for i in range(t.shape[0]):
            for e in range(mom.shape[1]):
                f.write(" ".join(["%.17g" % t[i],str(e)]+["%.17g" % m for m in mom[i,e]])+"\n")

def _truncate_moments(directory,t):
    # restart housekeeping: drop the rows past the restart time (the run about to start
    # rewrites them), keeping the header. A clean restart's last row has t == state.t, so
    # nothing is dropped there.
    path = os.path.join(str(directory),_MOMENTS_FILE)
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = f.read().splitlines()
    kept = [ln for ln in lines if not ln.strip() or ln.startswith("#") or float(ln.split()[0]) <= t]
    if len(kept) != len(lines):
        with open(path,"w") as f:
            f.write("".join(ln+"\n" for ln in kept))

def _finish(state,params,t_start):
    mngr_time = perf_counter()-t_start
    if params.rank==0:
        print("Ending simulation at t = "+str(state.t)+". It took "+str(mngr_time)+"s")
    return state

def _check_pstate(params,pstate):
    # a particle carry is required exactly when params.particles is set
    if params.particles is not None and pstate is None:
        raise ValueError("params.particles is set, so this driver needs a particle carry: "
                         "pass pstate=particles.state.init_particles(params) or "
                         "snapshot_io.load_particles(isnap,snap_path,params).")
    if params.particles is None and pstate is not None:
        raise ValueError("pstate was passed but params.particles is None (particles are off).")

def simulate_scan(state,kgrid,params,nblock,t_snap,t_end,mngr,schemestr='lsrk33',save=True,print_every=1,pstate=None):
    # this simulates repeated fixed number of timesteps
    # for automatic differentiation sometime in the future
    # we should set nblock using the helper function estimate_good_nblock
    t_start = perf_counter()
    stepper,scheme = get_scheme(schemestr)
    _check_pstate(params,pstate)
    state = _refresh_forcing_scale(state, kgrid, params)
    # donate_argnums=(0,): caller's input `state` buffer is consumed/reused for the output, since we always reassign `state` from the return value below.
    if params.comm_backend == "jax":
        # same stepper, wrapped in the z-mesh shard_map that the collectives need.
        # shard_call specs the bare state, which is what the carry holds here: particles
        # are rejected on this backend, so aux is always None.
        advance_jit = jax.jit(comms.shard_call(
            lambda s,kg: _advance_block((s,None),kg,params,nblock,scheme,stepper)[0][0],
            params, kgrid), donate_argnums=(0,))
        advance = lambda c: ((advance_jit(c[0],kgrid),None),None)
    else:
        # the jitted block carries (state, aux) and emits per-step ys; the whole carry is donated
        block_jit = jax.jit(_advance_block,static_argnums=(2,3,4,5),donate_argnums=(0,))
        advance = lambda c: block_jit(c,kgrid,params,nblock,scheme,stepper)
    if params.particles is not None and save:
        _truncate_moments(mngr.directory,float(state.t))
    # float(): pull to host so this doesn't alias state.t's buffer, which donate_argnums frees on the next jit call
    t_last_snapshot = float(state.t)
    carry = (state,pstate)
    snap = _start_snapshots(state,params,mngr,save,pstate)
    block_count=0
    saved_current=True
    while state.t<t_end:
        carry,ys = advance(carry)
        if params.particles is not None and save:
            _append_moments(mngr.directory,ys)
        state,pstate = carry
        block_count+=1
        saved_current=False
        if params.rank==0 and block_count%print_every==0:
            print(state.t) #this doesnt affect performance; state.t already on host from while
        if state.t - t_last_snapshot > t_snap:
            snap = _write_snapshot(snap,state,params,mngr,save,"snapshot",pstate)
            t_last_snapshot=float(state.t)
            saved_current=True
    if not saved_current:
        snap = _write_snapshot(snap,state,params,mngr,save,"final state",pstate)
    mngr.wait_until_finished()
    state = _finish(state,params,t_start)
    return state if params.particles is None else (state,pstate)

def simulate(initial_state,kgrid,params,t_snap,t_end,mngr,schemestr='lsrk33',save=True,print_every=1,pstate=None):
    t_start = perf_counter()
    stepper,scheme = get_scheme(schemestr)
    _check_pstate(params,pstate)
    set_timestep = equation_registry[params.eqtype].set_timestep_func
    rhs = construct_rhs(equation_registry[params.eqtype])
    # kgrid is an explicit argument (not a closure) so the jax backend can hand it to
    # shard_map with its own in_specs; the mpi4jax branch below re-closes over it unchanged.
    # cfl_every>1: the while_loop iterates over whole blocks, so it can overshoot the
    # snapshot target by up to cfl_every-1 extra steps (on top of the usual one-step
    # overshoot). The while_loop cannot emit ys, so with particles on `simulate` has
    # snapshot-cadence diagnostics only -- _step's/_cfl_block's moments are dropped here.
    if _use_cfl_blocks(params):
        def loop_body(carry,kgrid):
            return _cfl_block(carry,kgrid,params,rhs,set_timestep,scheme,stepper)[0]
    else:
        def loop_body(carry,kgrid):
            return _step(carry,kgrid,params,rhs,set_timestep,scheme,stepper)[0]
    def sim_to_next_snap(carry,kgrid,target_t):
        def snap_cond(carry):
            return carry[0].t<target_t
        return jax.lax.while_loop(snap_cond,lambda c: loop_body(c,kgrid),carry)
    # donate_argnums=(0,): caller's input `state` buffer is consumed/reused for the output, since we always reassign `state` from the return value below.
    if params.comm_backend == "jax":
        # shard_call specs the bare state; particles are rejected on this backend, so the
        # carry's aux is always None
        _jit = jax.jit(comms.shard_call(
            lambda s,kg,target_t: sim_to_next_snap((s,None),kg,target_t)[0],
            params,kgrid,nextra=1),donate_argnums=(0,))
        sim_to_next_snap_jit = lambda c,target_t: (_jit(c[0],kgrid,target_t),None)
    else:
        sim_to_next_snap_jit = jax.jit(lambda carry,target_t: sim_to_next_snap(carry,kgrid,target_t),
                                       donate_argnums=(0,))
    state=_refresh_forcing_scale(initial_state, kgrid, params)
    # float(): pull to host so this doesn't alias state.t's buffer, which donate_argnums frees on the next jit call
    t_last_snapshot = float(state.t)
    carry = (state,pstate)
    snap = _start_snapshots(state,params,mngr,save,pstate)

    while state.t<t_end:
        carry = sim_to_next_snap_jit(carry,min(t_last_snapshot+t_snap,t_end))
        state,pstate = carry
        snap = _write_snapshot(snap,state,params,mngr,save,"snapshot",pstate)
        t_last_snapshot=float(state.t)
    mngr.wait_until_finished()
    state = _finish(state,params,t_start)
    return state if params.particles is None else (state,pstate)

