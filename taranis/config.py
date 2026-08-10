import numpy as np
from . import comms
from . import _precision
from ._mpi_compat import HAVE_MPI4JAX, HAVE_MPI4PY, MPI, _NullComm, launcher_world_size
from .physics import equation_registry
import os
import json
import time
import inspect
import warnings

def _json_scalar(v):
    # json.dump fallback for Parameters.save: numpy/jax 0-d scalars (e.g. a dt pulled out
    # of an array) aren't natively JSON-serializable; unwrap them to python scalars.
    item = getattr(v, "item", None)
    if item is not None and getattr(v, "ndim", 0) == 0:
        return item()
    raise TypeError(f"Parameter value {v!r} (type {type(v).__name__}) can't be recorded "
                    f"in params.json — pass plain python types to Parameters")

# ctor args excluded from params.json's "differing record" check (transport only, no
# effect on the trajectory or the on-disk layout)
_TRANSPORT_KEYS = ("comm_backend",)

# ctor args that MOVED into the per-equation `eqpars` dict (2026-08-01). A params.json
# written before that records them at top level; they carry the same information there.
_EQPARS_MOVED_KEYS = ("diss", "hyper")

def _fold_legacy_eqpars(rec):
    # top-level diss/hyper in an old record -> rec["eqpars"]; returns the moved key names
    moved = [k for k in _EQPARS_MOVED_KEYS if k in rec]
    if not moved:
        return moved
    eqpars = dict(rec.get("eqpars") or {})
    for k in moved:
        value = rec.pop(k)
        eqpars.setdefault(k, value)
    rec["eqpars"] = eqpars
    return moved

def _lists_to_tuples(v):
    # JSON restore: lists become tuples (Parameters compares/records tuples), recursing
    # into eqpars-style dicts so equation parameters round-trip like ctor args do
    if isinstance(v, list):
        return tuple(_lists_to_tuples(x) for x in v)   # recurse: nested tuples too
    if isinstance(v, dict):
        return {k: _lists_to_tuples(x) for k, x in v.items()}
    return v

_MPI_HINT = ('install the MPI extra: pip install "taranis[mpi]" (or use '
             "comm_backend='serial'/None for single-process runs)")

def _resolve_backend(requested):
    # Pick the transport for THIS process. requested=None (the default) auto-resolves:
    # mpi4py+mpi4jax -> "mpi4jax" (the unchanged production path), else "serial". Never
    # "serial" under a real multi-rank launcher — every rank would run the full domain and
    # overwrite the others' snapshots — so the launcher env is sniffed on every path.
    nlaunch, definitive = launcher_world_size()
    if HAVE_MPI4PY:
        size = MPI.COMM_WORLD.Get_size()
        if definitive and nlaunch > 1 and size == 1:
            raise RuntimeError(
                f"this process was launched as one rank of a {nlaunch}-rank job, but mpi4py "
                "reports MPI_COMM_WORLD size 1 — mpi4py is built against a different MPI than "
                "the launcher, so every rank would run the full domain and overwrite the "
                "others' snapshots. Rebuild mpi4py against the launcher's MPI, e.g. "
                "MPICC=$(which mpicc) pip install --no-binary mpi4py --force-reinstall mpi4py")
    else:
        # no mpi4py: the launcher environment is the only evidence of a multi-rank job
        size = nlaunch if definitive else 1
    if requested is not None:
        if requested in ("mpi4jax", "jax") and not HAVE_MPI4PY:
            raise ImportError(f"comm_backend={requested!r} needs mpi4py, which is not "
                              f"importable — {_MPI_HINT}")
        if requested == "mpi4jax" and not HAVE_MPI4JAX:
            raise ImportError(f"comm_backend='mpi4jax' needs mpi4jax, which is not "
                              f"importable — {_MPI_HINT}")
        if requested == "serial" and size > 1:
            raise ValueError(f"comm_backend='serial' is single-process only, but this process "
                             f"is one of {size} ranks; use comm_backend='mpi4jax'")
        return requested
    if HAVE_MPI4PY and HAVE_MPI4JAX:
        return "mpi4jax"
    if HAVE_MPI4PY:
        # half-installed (mpi4py builds easily, mpi4jax needs a matching jax): fine serially
        if size > 1:
            raise RuntimeError(f"this is a {size}-rank MPI run, but mpi4jax is not importable "
                               f"and multi-rank transport needs it — {_MPI_HINT}")
        return "serial"
    if size > 1:
        raise RuntimeError(
            f"this process was launched as one rank of a {nlaunch}-rank MPI job, but mpi4py is "
            "not importable — refusing to fall back to comm_backend='serial', where every rank "
            f"would run the full domain and overwrite the others' snapshots. {_MPI_HINT}")
    if nlaunch > 1:
        # allocation-wide task count only: a plain `python` inside a batch script looks like
        # this, so it is a warning, not an error
        warnings.warn(f"the batch environment advertises {nlaunch} tasks, but this process was "
                      "not started by an MPI launcher and mpi4py is not importable: running "
                      "single-process with comm_backend='serial'.", stacklevel=3)
    return "serial"

class Parameters():
    #Stores all static parameters for the problem
    def __init__(self,nx,ny,Lx,Ly,cfl_safety,eqpars=None,dt=0.1,adaptive_timestep=True,dims=2,nz=1,Lz=2*np.pi,z_diss=0.25,z_diss_hyper=2.0,z_diff_order=4,eqtype="RMHD",
                 forcing=False,forcing_mode="momentum",forcing_power=1.0,forcing_power_elsasser=(1.0,1.0),forcing_tau=1.0,fshell=(1,2),forcing_seed=0,forcing_scale_max=1.0e4,
                 forcing_norm_per_step=True,lsrk_scan=True,forcing_shell_noise=False,comm_backend=None,
                 cfl_every=1,z_spectral=False):
        # capture the constructor arguments (before any normalization below) so
        # save()/from_snapshot() can reproduce this object exactly via __init__
        self._init_args = {k: v for k, v in locals().items() if k != "self"}
        if eqtype not in equation_registry:
            raise ValueError(f"eqtype must be one of {list(equation_registry)}, got {eqtype!r}")
        self.eqtype=eqtype
        self.nfields=equation_registry[self.eqtype].nfields
        #perpendicular grid
        self.nx=nx
        if ny%2==1:
            warnings.warn(f"ny should be even: using ny={ny-1} instead of {ny}", stacklevel=2)
            ny=ny-1
        self.ny=ny
        self.Lx=Lx
        self.Ly=Ly
        self.dx=Lx/nx
        self.dy=Ly/ny
        #equation-set parameters (plain-JSON dict; RMHD: diss/hyper, which were ctor args
        #before 2026-08-01). Interpreted by the equation recipe, not here.
        if eqpars is not None and not isinstance(eqpars, dict):
            raise ValueError(f"eqpars must be a dict of equation parameters (e.g. "
                             f"{{'diss': (nu, eta), 'hyper': 1}}), got {eqpars!r}")
        self.eqpars = dict(eqpars) if eqpars is not None else {}
        self._init_args["eqpars"] = dict(self.eqpars)  # decoupled from the caller's dict
        #timestepping
        if np.shape(cfl_safety) != ():
            raise ValueError(f"cfl_safety must be a scalar, got {cfl_safety!r} — note that "
                             f"diss/hyper are no longer ctor args (pass eqpars=...)")
        self.cfl_safety=cfl_safety
        self.dt = dt # Only used if adaptive_timestep==False
        self.adaptive_timestep = adaptive_timestep #Usually we want this to be true
        # recompute the CFL timestep (one global allreduce) only every cfl_every steps
        # this is dangerous! use with caution.
        if isinstance(cfl_every,bool) or not isinstance(cfl_every,(int,np.integer)) or cfl_every < 1:
            raise ValueError(f"cfl_every must be an int >= 1, got {cfl_every!r}")
        self.cfl_every = int(cfl_every)  # int(): accept numpy ints, store a plain int
        #dimensions
        if dims not in (2,3):
            raise ValueError(f"dims must be 2 or 3, got {dims!r}")
        self.spatial_dimensions=dims
        if dims==3:
            #z grid parameters
            if Lz <= 0:
                raise ValueError(f"dims=3 requires Lz > 0, got {Lz!r}")
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
        # transport used by taranis.comms for halos/allreduces (static: dispatched in plain python)
        # comm_backend=None auto-resolves to mpi4jax (MPI present) or serial (no MPI toolchain)
        if comm_backend is not None and comm_backend not in comms.COMM_BACKENDS:
            raise ValueError(f"comm_backend must be one of {comms.COMM_BACKENDS} or None (auto), "
                             f"got {comm_backend!r}")
        self.comm_backend = _resolve_backend(comm_backend)
        # record what actually ran, not the request: params.json documents the resolved backend
        self._init_args["comm_backend"] = self.comm_backend
        self.comm=_NullComm() if self.comm_backend=="serial" else MPI.COMM_WORLD
        self.rank=self.comm.Get_rank()
        self.size=self.comm.Get_size()
        if self.size > 1:
            # All ranks must read the same RMHD_PRECISION, or halo exchanges and
            # allreduces mix float32/float64 buffers and fail far from the cause
            # (PRECISION_PLAN "Risks"). One host-side collective is fine here:
            # Parameters construction is already collective under MPI.
            precs = self.comm.allgather(_precision.precision)
            if len(set(precs)) != 1:
                raise RuntimeError(
                    f"RMHD_PRECISION differs across ranks: rank {self.rank} sees "
                    f"{_precision.precision!r}, gathered {precs!r} — export the same "
                    "RMHD_PRECISION in every rank's environment")
        if self.comm_backend=="jax" and self.spatial_dimensions!=3:
            raise ValueError("comm_backend='jax' requires dims=3 (there is no z decomposition to map in 2D)")
        if self.spatial_dimensions==3:
            if self.nz % self.size != 0:
                raise ValueError(f"nz={self.nz} must be divisible by the number of MPI ranks ({self.size})")
            if self.comm_backend=="serial":
                # single process: no cartesian topology to build, and cart_comm=None keeps the
                # allreduces at identity (comms.halo_exchange wraps z onto self)
                self.cart_comm = None
                self.left_neighbor = None
                self.right_neighbor = None
            else:
                self.cart_comm = self.comm.Create_cart(dims=[self.size],periods=[True],reorder=False)
                self.left_neighbor, self.right_neighbor = self.cart_comm.Shift(direction=0, disp=1)
        else:
            self.cart_comm = None
            self.left_neighbor = None
            self.right_neighbor = None
            if self.size > 1 and self.rank==0:
                warnings.warn("You probably should only run a 2D run on one device, since this "
                              "isn't parallelized.", stacklevel=2)
        # spectral z (plans/GDI_PLAN.md P2): fields' axis 1 is kz, not z. Single-process
        # only (a z-FFT needs the whole z domain), 3D only, and never under the sharded
        # "jax" backend. Recorded in params.json: snapshots are NOT cross-mode compatible,
        # and the save() comparison is what stops a mixed-mode restart.
        self.z_spectral = bool(z_spectral)
        if self.z_spectral:
            if self.spatial_dimensions != 3:
                raise ValueError("z_spectral=True requires dims=3 (there is no z axis to "
                                 "transform in 2D)")
            if self.size != 1:
                raise ValueError(f"z_spectral=True is single-process only (the z-FFT needs the "
                                 f"whole z domain on one rank), but this process is one of "
                                 f"{self.size} ranks")
            if self.comm_backend == "jax":
                raise ValueError("z_spectral=True is incompatible with comm_backend='jax' "
                                 "(the jax backend exists to decompose z across devices)")
        # bring up the chosen transport (jax.distributed + z mesh for "jax"; no-op otherwise).
        # must happen before any jax device work
        comms.init_backend(self)
        #forcing
        self.forcing = forcing
        if forcing_mode not in ("momentum","elsasser"):
            raise ValueError(f"forcing_mode must be 'momentum' or 'elsasser', got {forcing_mode!r}")
        if forcing:
            if np.shape(fshell) != (2,) or fshell[0] >= fshell[1]:
                raise ValueError(f"fshell must be (nmin, nmax) with nmin < nmax, got {fshell!r}")
            if forcing_mode == "elsasser" and np.shape(forcing_power_elsasser) != (2,):
                raise ValueError(f"forcing_mode='elsasser' requires forcing_power_elsasser to be "
                                 f"(eps_plus, eps_minus), got {forcing_power_elsasser!r}")
        self.forcing_mode = forcing_mode
        self.forcing_power = forcing_power
        #(eps_plus, eps_minus), total energy injection rate = eps_plus+eps_minus
        #nb. Etot = (E^+ + E^-)/2 with E^pm = (z^pm)^2/2
        self.forcing_power_elsasser = forcing_power_elsasser
        self.forcing_tau = forcing_tau
        self.fshell = fshell
        self.forcing_seed = forcing_seed
        self.forcing_scale_max = forcing_scale_max
        self.forcing_norm_per_step = forcing_norm_per_step
        self.forcing_shell_noise = forcing_shell_noise
        self.n_ou = 1 if self.forcing_mode == "momentum" else 2
        # timestepping (structure): lax.scan LSRK stage loop (default
        # vs statically unrolled (lsrk_scan=False; in principle could
        # be faster on some GPU systems)
        self.lsrk_scan = lsrk_scan
        if self.spatial_dimensions==3 and self.rank==0 and (z_diff_order != 4 or z_diss_hyper != 2.0):
            warnings.warn(f"z_diff_order={z_diff_order}, z_diss_hyper={z_diss_hyper}: both are "
                          "stored but IGNORED. rmhd.LinearTerm is fixed at 4th-order centered "
                          "differences with d_z^4 hyperdissipation.", stacklevel=2)
        if self.z_spectral and self.rank==0 and z_diss != 0.25:
            # every finite-difference-z knob is dead in spectral z: no stencil, no halo, no
            # dz-based CFL term. kz dissipation is eqpars['z_diss_k'] instead.
            warnings.warn(f"z_spectral=True: z_diss={z_diss} is a finite-difference-z knob and "
                          "is IGNORED; use eqpars['z_diss_k'] (-z_diss_k*kz^4) for kz "
                          "dissipation.", stacklevel=2)

    def save(self, snap_path, filename="params.json"):
        # record the constructor arguments (not derived attrs) to snap_path/filename, so a
        # run directory documents how it was made and from_snapshot can reproduce it.
        # saving over an existing file with DIFFERENT contents is a hard error
        # collective under MPI: rank 0 alone checks/writes and broadcasts the outcome
        path = os.path.join(str(snap_path), filename)
        err = None
        if self.rank == 0:
            # round-trip through JSON up front
            rec = json.loads(json.dumps(self._init_args, default=_json_scalar))
            rec["_precision"] = _precision.precision
            if os.path.exists(path):
                with open(path) as f:
                    old = json.load(f)
                old.pop("_created", None)
                # pre-eqpars records keep diss/hyper at top level: same information, so
                # fold them in (and refresh the file) rather than report a differing record
                moved = _fold_legacy_eqpars(old)
                # a record written by older code lacks newer ctor args: backfill them with
                # the current signature defaults (JSON-normalized) so adding a Parameters
                # argument never invalidates existing run directories
                sig_defaults = {k: v.default for k, v in inspect.signature(type(self).__init__).parameters.items()
                                if v.default is not inspect.Parameter.empty}
                backfilled = sorted(k for k in rec if k not in old and k in sig_defaults)
                for k in backfilled:
                    # transport keys (comm_backend) document what actually ran, not a generic
                    # ctor default — backfill from this save's resolved value so an old record
                    # gains e.g. "serial"/"mpi4jax" instead of a misleading null
                    old[k] = rec[k] if k in _TRANSPORT_KEYS else \
                        json.loads(json.dumps(sig_defaults[k], default=_json_scalar))
                # comm_backend is a transport choice, not a physics/grid parameter: a run
                # must be restartable across backends: recorded but not compared
                diffs = {k: (old.get(k, "<absent>"), rec.get(k, "<absent>"))
                         for k in sorted(set(old) | set(rec))
                         if k not in _TRANSPORT_KEYS and old.get(k, "<absent>") != rec.get(k, "<absent>")}
                if diffs:
                    err = (f"{path} already records different parameters "
                           f"(saved, current): {diffs}. If the change is intended, delete "
                           f"{filename} and re-save; to reuse the recorded values, "
                           f"Parameters.from_snapshot(...) and pass overrides explicitly.")
                elif backfilled or moved:
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
        # reconstruct Parameters from a run directory's record (written by save());
        # explicitly passed overrides win
        # runs __init__ to get derived params
        path = os.path.join(str(snap_path), filename)
        with open(path) as f:
            rec = json.load(f)
        rec.pop("_created", None)
        prec = rec.pop("_precision", None)
        current_prec = _precision.precision
        rank0 = MPI.COMM_WORLD.Get_rank() == 0 if HAVE_MPI4PY else True
        # legacy shim: diss/hyper were ctor args before 2026-08-01, they are equation
        # parameters now. Fold before the unknown-key check so they are not "unknown".
        moved = _fold_legacy_eqpars(rec)
        if moved and rank0:
            warnings.warn(f"{path} is a pre-eqpars record: top-level {moved} folded into "
                          f"eqpars={rec['eqpars']!r}. Re-save to update the file.",
                          stacklevel=2)
        if prec is not None and prec != current_prec and rank0:
            warnings.warn(f"{path} was written at precision {prec}, but RMHD_PRECISION is "
                          f"currently {current_prec} (precision is set by env var at import time).",
                          stacklevel=2)
        # tolerate records from newer/older code versions: ignore unknown keys with a warning
        known = set(inspect.signature(cls.__init__).parameters) - {"self"}
        unknown = sorted(set(rec) - known)
        if unknown and rank0:
            warnings.warn(f"ignoring unknown parameters in {path}: {unknown}", stacklevel=2)
        args = {k: _lists_to_tuples(v) for k, v in rec.items() if k in known}
        # transport, not physics: re-resolve on THIS machine unless the caller says otherwise
        # (a Savio params.json recording "mpi4jax" must load on a laptop with no MPI)
        args.pop("comm_backend", None)
        args.update(overrides)
        return cls(**args)


