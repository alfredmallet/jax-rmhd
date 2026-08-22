import numpy as np
import copy
from . import comms
from . import _precision
# imported BY VALUE: these three names are the substitution point comms._resolve_backend
# reads (HAVE_MPI4JAX only through it, hence the noqa)
from ._mpi_compat import HAVE_MPI4JAX, HAVE_MPI4PY, MPI  # noqa: F401
from .physics import equation_registry
import os
import json
import time
import inspect
import warnings

def _json_scalar(v):
    # fallback for Parameters.save: numpy/jax 0-d scalars unwrapped to python scalars.
    item = getattr(v, "item", None)
    if item is not None and getattr(v, "ndim", 0) == 0:
        return item()
    raise TypeError(f"Parameter value {v!r} (type {type(v).__name__}) can't be recorded "
                    f"in params.json — pass plain python types to Parameters")

# ctor args excluded from params.json's "differing record" check
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

# backend resolution lives in comms, next to the Runtime it builds; it reads this module's
# HAVE_MPI4PY/HAVE_MPI4JAX/MPI names (imported above BY VALUE from _mpi_compat, so this is
# where they can be substituted), and is re-exported here for callers of config._resolve_backend
_resolve_backend = comms._resolve_backend

class Parameters():
    # stores all static parameters for the problem
    def __init__(self,nx,ny,Lx,Ly,cfl_safety,eqpars=None,dt=0.1,adaptive_timestep=True,dims=2,nz=1,Lz=2*np.pi,z_diss=0.25,z_diss_hyper=2.0,z_diff_order=4,eqtype="RMHD",
                 forcing=False,forcing_mode="momentum",forcing_power=1.0,forcing_power_elsasser=(1.0,1.0),forcing_tau=1.0,fshell=(1,2),forcing_seed=0,forcing_scale_max=1.0e4,
                 forcing_norm_per_step=True,lsrk_scan=True,forcing_shell_noise=False,comm_backend=None,
                 cfl_every=1,z_spectral=False,particles=None,hoist_propagator=True,runtime=None):
        # capture the constructor arguments (before any normalization below) so
        # save()/from_snapshot() can reproduce this object exactly via __init__.
        # runtime is a live transport object, not a recordable value: it is popped below
        self._init_args = {k: v for k, v in locals().items() if k != "self"}
        self._init_args.pop("runtime")
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
        #equation-set parameters; interpreted by equation recipe in physics (plain-JSON dict).
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
        # transport used by taranis.comms for halos/allreduces (static: dispatched in plain
        # python). runtime=None resolves one for this process (comm_backend=None then
        # auto-resolves to mpi4jax with MPI present, serial without); an explicit Runtime is
        # reused as is, so a parameter scan shares one set of communicators
        self.z_spectral = bool(z_spectral)
        if runtime is None:
            self.runtime = comms.Runtime.resolve(comm_backend, dims=self.spatial_dimensions,
                                                 nz=self.nz, z_spectral=self.z_spectral)
        else:
            if comm_backend is not None and comm_backend != runtime.backend:
                raise ValueError(f"comm_backend={comm_backend!r} contradicts the given "
                                 f"runtime.backend={runtime.backend!r}: pass comm_backend=None "
                                 f"(or the runtime's own backend) alongside runtime=")
            self.runtime = runtime
        # record what actually ran, not the request: params.json documents the resolved backend
        self._init_args["comm_backend"] = self.comm_backend
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
        # timestepping (structure): form the IF schemes' per-stage exp(L*tau) once per
        # frozen-dt block (fixed dt, or a cfl_every block) instead of inside every stage of
        # every step (timestepping.stage_exp_ops). Same numbers; putzer2 backend only
        # (z_spectral RMHD, GDI under an IF scheme); costs one ExpOp per stage of memory,
        # 4 complex arrays of L's full shape each (the knob to turn when a z_spectral grid
        # is memory-bound). No effect on the diagonal backend (FD-z / 2D RMHD), on adaptive
        # dt with cfl_every=1 (nothing is frozen) or on the IMEX schemes.
        self.hoist_propagator = bool(hoist_propagator)
        if self.spatial_dimensions==3 and self.rank==0 and (z_diff_order != 4 or z_diss_hyper != 2.0):
            warnings.warn(f"z_diff_order={z_diff_order}, z_diss_hyper={z_diss_hyper}: both are "
                          "stored but IGNORED. rmhd.FDLinearTerm is fixed at 4th-order centered "
                          "differences with d_z^4 hyperdissipation.", stacklevel=2)
        if self.z_spectral and self.rank==0 and z_diss != 0.25:
            warnings.warn(f"z_spectral=True: z_diss={z_diss} is a finite-difference-z knob and "
                          "is ignored; use eqpars['z_diss_k'] (-z_diss_k*kz^4) for kz "
                          "dissipation.", stacklevel=2)
        #test particles (plans/TESTPART_PLAN.md); None = off, no particle code runs
        self.particles = None
        if particles is not None:
            if not isinstance(particles, dict):
                raise ValueError(f"particles must be a dict of particle configuration (e.g. "
                                 f"{{'n': 1024, 'ensembles': [{{'qm': 15.0, 'init': ...}}]}}), "
                                 f"got {particles!r}")
            # imported here, not at module scope: config itself does not depend on the
            # particle package (nothing under particles/ imports config — no cycle)
            from .particles.state import normalize_config
            self.particles = normalize_config(particles)
            self.n_ens = len(self.particles["ensembles"])
            self._init_args["particles"] = copy.deepcopy(particles)  # decoupled from the caller's dict
        self._validate_compat()

    # transport lives on the Runtime; these forward so every params.<attr> reader keeps
    # working, and none of them can be assigned on a Parameters (spoofing a rank means
    # dataclasses.replace on the runtime)
    @property
    def comm_backend(self):
        return self.runtime.backend

    @property
    def comm(self):
        return self.runtime.comm

    @property
    def rank(self):
        return self.runtime.rank

    @property
    def size(self):
        return self.runtime.size

    @property
    def cart_comm(self):
        return self.runtime.cart_comm

    @property
    def left_neighbor(self):
        return self.runtime.left_neighbor

    @property
    def right_neighbor(self):
        return self.runtime.right_neighbor

    def _validate_compat(self):
        # the compatibility matrix — backend x dims x size, z_spectral, particles — in one
        # place, called once from __init__ with every attribute it reads set. nz % size,
        # jax-needs-3D, the z_spectral trio and the device-count checks are also in
        # Runtime.resolve/init_backend, which guard the communicators and mesh THEY create;
        # here they cover the shared-Runtime path, which re-runs none of them.
        if self.comm_backend=="jax" and self.spatial_dimensions!=3:
            raise ValueError("comm_backend='jax' requires dims=3 (there is no z decomposition to map in 2D)")
        if self.spatial_dimensions==3:
            if self.nz % self.size != 0:
                raise ValueError(f"nz={self.nz} must be divisible by the number of MPI ranks ({self.size})")
            if self.comm_backend != "serial" and self.cart_comm is None:
                raise ValueError(f"dims=3 on comm_backend={self.comm_backend!r} needs the z "
                                 f"cartesian communicator, and the given runtime has none (it "
                                 f"was resolved for dims=2): the halo exchange would fail and "
                                 f"the CFL allreduce would silently go rank-local. Resolve a "
                                 f"Runtime with dims=3, or pass runtime=None")
        elif self.size > 1 and self.rank==0:
            warnings.warn("You probably should only run a 2D run on one device, since this "
                          "isn't parallelized.", stacklevel=3)
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
        if self.comm_backend == "jax":
            # comms.init_backend's device-count checks, against THIS nz: a shared Runtime
            # does not re-run them. The mesh exists already — the runtime built it.
            ndev = comms.get_mesh().size
            if ndev % self.size:
                raise ValueError(f"comm_backend='jax': global device count {ndev} must be a multiple "
                                 f"of the process count {self.size}")
            if self.nz % ndev:
                raise ValueError(f"comm_backend='jax': nz={self.nz} must be divisible by the "
                                 f"global device count {ndev}")
        if self.particles is not None:
            if self.eqtype != "RMHD" or self.size != 1 or self.comm_backend == "jax":
                raise ValueError(f"test particles require eqtype='RMHD', a single process and "
                                 f"a non-sharded backend (the gather needs the whole z domain "
                                 f"on this rank), got eqtype={self.eqtype!r}, "
                                 f"size={self.size}, comm_backend={self.comm_backend!r}; "
                                 f"z-decomposed particles are unimplemented "
                                 f"(plans/TESTPART_PLAN.md §4)")
            if self.spatial_dimensions == 3:
                bad = sorted({e["B0"] for e in self.particles["ensembles"] if e["B0"] != 1.0})
                if bad:
                    raise ValueError(
                        f"particles: B0 must be 1.0 in 3D, got {bad}. E_z = -B0*d_z(phi) + "
                        f"dpsi/dt collapses to the ideal -{{phi,psi}} only because the solver "
                        f"supplies +d_z(phi) — with coefficient exactly 1 "
                        f"(rmhd.linear_matrix's off-diagonal is 1j*kz, FDLinearTerm returns a "
                        f"bare d_z), i.e. the guide field IS 1 in code units. Any other B0 "
                        f"leaves (1-B0)*d_z(phi) in E_z, so E.B != 0 and the fields are no "
                        f"longer ideal-Ohm. In 3D the amplitude parameter eps = rms|grad psi| "
                        f"is set by the forcing amplitude and Lz instead (v_A = 1 on a box Lz "
                        f"is the same system as v_A = B0 on a box B0*Lz); B0 stays the free "
                        f"per-ensemble knob in 2D, which has no Alfven term")

    def save(self, snap_path, filename="params.json"):
        # record the constructor arguments (not derived attrs) to snap_path/filename, so a
        # run directory documents how it was made and from_snapshot can reproduce it.
        # saving over an existing file with different contents is a hard error
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
            warnings.warn(f"{path} was written at precision {prec}, but TARANIS_PRECISION is "
                          f"currently {current_prec} (precision is set by env var at import time).",
                          stacklevel=2)
        # tolerate records from newer/older code versions: ignore unknown keys with a warning
        known = set(inspect.signature(cls.__init__).parameters) - {"self"}
        unknown = sorted(set(rec) - known)
        if unknown and rank0:
            warnings.warn(f"ignoring unknown parameters in {path}: {unknown}", stacklevel=2)
        args = {k: _lists_to_tuples(v) for k, v in rec.items() if k in known}
        # transport, not physics: re-resolve on this machine unless the caller says otherwise
        args.pop("comm_backend", None)
        args.update(overrides)
        return cls(**args)


