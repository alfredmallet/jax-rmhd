from mpi4py import MPI
import jax
import numpy as np
from jax.tree_util import register_pytree_node_class
from .types import SimulationState
from . import comms
import os
import json
import time
import inspect

def _json_scalar(v):
    # json.dump fallback for Parameters.save: numpy/jax 0-d scalars (e.g. a dt pulled out
    # of an array) aren't natively JSON-serializable; unwrap them to python scalars.
    item = getattr(v, "item", None)
    if item is not None and getattr(v, "ndim", 0) == 0:
        return item()
    raise TypeError(f"Parameter value {v!r} (type {type(v).__name__}) can't be recorded "
                    f"in params.json — pass plain python types to Parameters")

@register_pytree_node_class
class Parameters():
    #Stores all static parameters for the problem
    def __init__(self,nx,ny,Lx,Ly,diss,hyper,cfl_safety,dt=0.1,adaptive_timestep=True,dims=2,nz=1,Lz=0.0,z_diss=0.25,z_diss_hyper=2.0,z_diff_order=4,eqtype="RMHD",
                 forcing=False,forcing_mode="momentum",forcing_power=1.0,forcing_power_elsasser=(1.0,1.0),forcing_tau=1.0,fshell=(1,2),forcing_seed=0,forcing_scale_max=1.0,
                 forcing_norm_per_step=True,lsrk_scan=True,forcing_shell_noise=False,comm_backend="mpi4jax",
                 cfl_every=1):
        # capture the constructor arguments (before any normalization below) so
        # save()/from_snapshot() can reproduce this object exactly via __init__
        self._init_args = {k: v for k, v in locals().items() if k != "self"}
        self.eqtype=eqtype
        self.nfields=eqtype_registry[self.eqtype]
        #perpendicular grid
        self.nx=nx
        if ny%2==1:
            print("ny should be even: setting ny=ny-1")
            ny=ny-1
        self.ny=ny
        self.Lx=Lx
        self.Ly=Ly
        self.dx=Lx/nx
        self.dy=Ly/ny
        #perpendicular dissipation
        self.diss = diss # should be a tuple of length nfields
        self.hyper=hyper
        #timestepping
        self.cfl_safety=cfl_safety
        self.dt = dt # Only used if adaptive_timestep==False
        self.adaptive_timestep = adaptive_timestep #Usually we want this to be true
        # Recompute the CFL timestep (one global allreduce) only every cfl_every steps; dt is
        # then frozen for the whole block, so the CFL condition can be transiently violated
        # while the flow accelerates — compensate with extra cfl_safety margin. Only
        # meaningful with adaptive_timestep=True (fixed dt needs no reduction at all);
        # cfl_every=1 (default) is exactly the historical per-step behavior.
        # WARNING: from a quiescent start the CFL dt collapses ~10x within a few steps as
        # forcing spins up; a frozen quiescent dt at cfl_every>~5 runs far over CFL and
        # silently NaNs. Use cfl_every>1 only from developed states (e.g. after restart).
        if isinstance(cfl_every,bool) or not isinstance(cfl_every,(int,np.integer)) or cfl_every < 1:
            raise ValueError(f"cfl_every must be an int >= 1, got {cfl_every!r}")
        self.cfl_every = int(cfl_every)  # int(): accept numpy ints, store a plain int
        #dimensions
        self.spatial_dimensions=dims
        if dims==3:
            #z grid parameters
            self.nz = nz
            self.Lz = Lz
            self.dz = Lz/nz
            #z dissipation parameters
            self.z_diss = z_diss #This is in dimensionless units, dissipation coefficient is zdiss*(dz/2)^4
                                 #cf Pueschel et al. 2010
            self.z_diss_hyper = z_diss_hyper #currently unused, set =2
            self.z_diff_order = z_diff_order #currently unused, set =4
        else:
            self.nz=1
        #MPI
        # transport used by jax_rmhd.comms for halos/allreduces (static: dispatched in plain python)
        if comm_backend not in comms.COMM_BACKENDS:
            raise ValueError(f"comm_backend must be one of {comms.COMM_BACKENDS}, got {comm_backend!r}")
        self.comm_backend = comm_backend
        self.comm=MPI.COMM_WORLD
        self.rank=self.comm.Get_rank()
        self.size=self.comm.Get_size()
        if self.spatial_dimensions==3:
            self.cart_comm = self.comm.Create_cart(dims=[self.size],periods=[True],reorder=False)
            self.left_neighbor, self.right_neighbor = self.cart_comm.Shift(direction=0, disp=1)
        else:
            self.cart_comm = None
            self.left_neighbor = None
            self.right_neighbor = None
            if self.size > 1 and self.rank==0:
                print("You probably should only run a 2D run on one device, since this isn't parallelized.")
        #forcing
        self.forcing = forcing
        if forcing_mode not in ("momentum","elsasser"):
            raise ValueError(f"forcing_mode must be 'momentum' or 'elsasser', got {forcing_mode!r}")
        self.forcing_mode = forcing_mode
        self.forcing_power = forcing_power
        self.forcing_power_elsasser = forcing_power_elsasser
        self.forcing_tau = forcing_tau
        self.fshell = fshell
        self.forcing_seed = forcing_seed
        self.forcing_scale_max = forcing_scale_max
        self.forcing_norm_per_step = forcing_norm_per_step
        self.forcing_shell_noise = forcing_shell_noise
        self.n_ou = 1 if self.forcing_mode == "momentum" else 2
        #timestepping (structure): lax.scan LSRK stage loop (default; ~20% faster than the
        #unrolled loop on CPU) vs statically unrolled (lsrk_scan=False; to benchmark on GPU)
        self.lsrk_scan = lsrk_scan
    def save(self, snap_path, filename="params.json"):
        # Record the constructor arguments (not derived attrs) to snap_path/filename, so a
        # run directory documents how it was made and from_snapshot can reproduce it.
        # An existing file with DIFFERENT contents is a hard error — that's the guard
        # against continuing a directory with the wrong parameters.
        # Collective under MPI: rank 0 alone checks/writes and broadcasts the outcome, so
        # every rank raises or returns together — a per-rank exists check can desync ranks
        # on a lazy shared filesystem (some see the file, some don't, and whichever subset
        # skips/raises leaves the rest stuck in the collective).
        path = os.path.join(str(snap_path), filename)
        err = None
        if self.rank == 0:
            # round-trip through JSON up front (tuples->lists, numpy/jax scalars->python
            # via _json_scalar) so the comparison below sees exactly what a reload sees,
            # and so unserializable values fail here with a clear message, not mid-write
            rec = json.loads(json.dumps(self._init_args, default=_json_scalar))
            rec["_precision"] = "64" if jax.config.read("jax_enable_x64") else "32"
            if os.path.exists(path):
                with open(path) as f:
                    old = json.load(f)
                old.pop("_created", None)
                # a record written by older code lacks newer ctor args: backfill them with
                # the current signature defaults (JSON-normalized) so adding a Parameters
                # argument never invalidates existing run directories
                sig_defaults = {k: v.default for k, v in inspect.signature(type(self).__init__).parameters.items()
                                if v.default is not inspect.Parameter.empty}
                backfilled = sorted(k for k in rec if k not in old and k in sig_defaults)
                for k in backfilled:
                    old[k] = json.loads(json.dumps(sig_defaults[k], default=_json_scalar))
                diffs = {k: (old.get(k, "<absent>"), rec.get(k, "<absent>"))
                         for k in sorted(set(old) | set(rec))
                         if old.get(k, "<absent>") != rec.get(k, "<absent>")}
                if diffs:
                    err = (f"{path} already records different parameters "
                           f"(saved, current): {diffs}. If the change is intended, delete "
                           f"{filename} and re-save; to reuse the recorded values, "
                           f"Parameters.from_snapshot(...) and pass overrides explicitly.")
                elif backfilled:
                    # semantically identical: refresh the file so it records the new keys
                    old["_created"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(path, "w") as f:
                        json.dump(old, f, indent=1)
                # else: identical record already present, nothing to write
            else:
                os.makedirs(str(snap_path), exist_ok=True)
                rec["_created"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(path, "w") as f:
                    json.dump(rec, f, indent=1)
        if self.size > 1:
            err = self.comm.bcast(err, root=0)  # also orders every rank after the write
        if err is not None:
            raise ValueError(err)

    @classmethod
    def from_snapshot(cls, snap_path, filename="params.json", **overrides):
        # Reconstruct Parameters from a run directory's record (written by save());
        # explicitly passed overrides win, e.g. from_snapshot(path, dt=0.05).
        # Runs __init__, so all validation/derived attributes stay in one place.
        path = os.path.join(str(snap_path), filename)
        rec = json.load(open(path))
        rec.pop("_created", None)
        prec = rec.pop("_precision", None)
        current_prec = "64" if jax.config.read("jax_enable_x64") else "32"
        rank0 = MPI.COMM_WORLD.Get_rank() == 0
        if prec is not None and prec != current_prec and rank0:
            print(f"Warning: {path} was written at precision {prec}, but RMHD_PRECISION is "
                  f"currently {current_prec} (precision is set by env var at import time).")
        # tolerate records from newer/older code versions: ignore unknown keys with a warning
        known = set(inspect.signature(cls.__init__).parameters) - {"self"}
        unknown = sorted(set(rec) - known)
        if unknown and rank0:
            print(f"Warning: ignoring unknown parameters in {path}: {unknown}")
        args = {k: (tuple(v) if isinstance(v, list) else v) for k, v in rec.items() if k in known}
        args.update(overrides)
        return cls(**args)

    def tree_flatten(self):
        children = ()
        param_data = {k: v for k, v in self.__dict__.items()}
        return (children, param_data)
    @classmethod
    def tree_unflatten(cls,param_data,children):
        obj = cls.__new__(cls)
        obj.__dict__.update(param_data)
        return obj
        

# registry to set the # of fields we are solving for
eqtype_registry = {
    "RMHD": 2
}

def init_cluster():
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        try:
            jax.distributed.initialize()
            print("Distributed system initialized. Total devices: ",jax.device_count())
        except (ValueError, RuntimeError):
            jax.distributed.initialize(coordinator_address="localhost:8888", num_processes=1, process_id=0, local_device_ids=0)

