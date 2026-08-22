# three comms backends:
# "mpi4jax" (CPU production): mpi4py communicators + mpi4jax device ops, arrays stay process-local.
# "jax": control plane is still mpi4py (rank/size, params.save, orbax, index broadcasts)
# only the three device ops become lax.ppermute/psum/pmax inside a shard_map over a 1D z mesh
# state/kgrid arrays are global jax.Arrays sharded along z; each device sees exactly the same local
# shapes mpi4jax ranks see.
# "serial": no MPI at all (single process, size 1). halos wrap onto self, allreduce is identity
import dataclasses
import os
import socket
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ._mpi_compat import MPI, mpi4jax, _NullComm, launcher_world_size  # MPI/mpi4jax are None when not installed

from . import _precision
from .types import SimulationState

# backends implemented here; Runtime.resolve rejects anything else.
COMM_BACKENDS = ("mpi4jax", "jax", "serial")
# mesh axis name for "jax" backend
Z_AXIS = "z"

_mesh = None
_dist_initialized = False

_MPI_HINT = ('install the MPI extra: pip install "taranis[mpi]" (or use '
             "comm_backend='serial'/None for single-process runs)")

def _mpi_names():
    # (HAVE_MPI4PY, HAVE_MPI4JAX, MPI) read from taranis.config, which imports them BY
    # VALUE from _mpi_compat: config is where a caller can substitute them, so backend
    # resolution and the world communicator both look there. Imported inside the function
    # because config imports this module at module scope.
    from . import config
    return config.HAVE_MPI4PY, config.HAVE_MPI4JAX, config.MPI

def _resolve_backend(requested):
    # pick the transport for THIS process. requested=None (the default) auto-resolves:
    # mpi4py+mpi4jax -> "mpi4jax" (the unchanged production path), else "serial". Never
    # "serial" under a real multi-rank launcher — every rank would run the full domain and
    # overwrite the others' snapshots — so the launcher env is sniffed on every path.
    HAVE_MPI4PY, HAVE_MPI4JAX, MPI = _mpi_names()
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

@dataclasses.dataclass(frozen=True, eq=False)
class Runtime:
    # process-level transport: the resolved backend plus the communicators the ops below
    # use. eq=False keeps it identity-hashed like Parameters; it is never a pytree and is
    # never traced. One Runtime can back many Parameters, so a parameter scan creates one
    # cartesian communicator instead of one per configuration.
    backend: str
    comm: object
    rank: int
    size: int
    cart_comm: object = None
    left_neighbor: object = None
    right_neighbor: object = None

    @classmethod
    def resolve(cls, comm_backend=None, *, dims, nz, z_spectral=False):
        # build this process's transport: resolve the backend, take the world communicator,
        # check every rank reads the same TARANIS_PRECISION, create the z cartesian
        # communicator (dims=3, non-serial backends) and bring the backend up.
        # init_backend runs jax.distributed.initialize for comm_backend="jax" and builds the
        # device mesh, so the first resolve() of such a process must precede any jax device
        # work in it — which is why every configuration this transport cannot serve (dims,
        # nz, z_spectral) is refused ahead of that irreversible bring-up.
        if comm_backend is not None and comm_backend not in COMM_BACKENDS:
            raise ValueError(f"comm_backend must be one of {COMM_BACKENDS} or None (auto), "
                             f"got {comm_backend!r}")
        backend = _resolve_backend(comm_backend)
        world = _mpi_names()[2]  # the mpi4py.MPI module, None without an MPI toolchain
        comm = _NullComm() if backend == "serial" else world.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if size > 1:
            # all ranks must read the same TARANIS_PRECISION
            precs = comm.allgather(_precision.precision)
            if len(set(precs)) != 1:
                raise RuntimeError(
                    f"TARANIS_PRECISION differs across ranks: rank {rank} sees "
                    f"{_precision.precision!r}, gathered {precs!r} — export the same "
                    "TARANIS_PRECISION in every rank's environment")
        if backend == "jax" and dims != 3:
            raise ValueError("comm_backend='jax' requires dims=3 (there is no z decomposition to map in 2D)")
        cart_comm = left_neighbor = right_neighbor = None
        if dims == 3:
            # also checked by Parameters._validate_compat, which sees the nz of a Parameters
            # built on a shared Runtime; here it guards the communicator this call creates
            if nz % size != 0:
                raise ValueError(f"nz={nz} must be divisible by the number of MPI ranks ({size})")
            if backend != "serial":
                cart_comm = comm.Create_cart(dims=[size], periods=[True], reorder=False)
                left_neighbor, right_neighbor = cart_comm.Shift(direction=0, disp=1)
        # z_spectral needs the whole z domain on one rank; also checked (with the same
        # messages) by Parameters._validate_compat for the shared-Runtime path
        if z_spectral:
            if dims != 3:
                raise ValueError("z_spectral=True requires dims=3 (there is no z axis to "
                                 "transform in 2D)")
            if size != 1:
                raise ValueError(f"z_spectral=True is single-process only (the z-FFT needs the "
                                 f"whole z domain on one rank), but this process is one of "
                                 f"{size} ranks")
            if backend == "jax":
                raise ValueError("z_spectral=True is incompatible with comm_backend='jax' "
                                 "(the jax backend exists to decompose z across devices)")
        runtime = cls(backend=backend, comm=comm, rank=rank, size=size, cart_comm=cart_comm,
                      left_neighbor=left_neighbor, right_neighbor=right_neighbor)
        init_backend(runtime, nz)
        return runtime

def _unknown_backend(backend):
    raise ValueError(f"unknown comm_backend {backend!r}, expected one of {COMM_BACKENDS}")

def get_mesh():
    # 1D device mesh over z, ordered by (process_index, device id) so mesh position i is
    # MPI rank i's device (one device per process in production)
    global _mesh
    if _mesh is None:
        devs = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
        _mesh = Mesh(np.array(devs).reshape(-1), (Z_AXIS,))
    return _mesh

def _local_device_ids(runtime):
    # one process per device
    # with many CUDA_VISIBLE_DEVICES it splits by node-local mpi rank
    # with only 1, returns [0]
    # on cpu runs or non-slurm cases returns None
    ids = os.environ.get("RMHD_LOCAL_DEVICE_IDS")
    if ids not in (None, "", "auto"):
        return [int(i) for i in ids.split(",")]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or visible == "":
        return None
    nvis = len(visible.split(","))
    if nvis == 1:
        # left as None, jax.distributed.initialize would parse CVD itself and feeds
        # the PHYSICAL id as the ordinal -> CUDA_ERROR_INVALID_DEVICE. so do this.
        return [0]
    node_comm = runtime.comm.Split_type(MPI.COMM_TYPE_SHARED)
    return [node_comm.Get_rank() % nvis]

def init_backend(runtime, nz):
    #called from Runtime.resolve
    global _dist_initialized
    if runtime.backend != "jax":
        return
    if runtime.size > 1 and not (_dist_initialized or
                                 getattr(jax.distributed, "is_initialized", lambda: False)()):
        addr = os.environ.get("RMHD_COORDINATOR_ADDRESS")
        if addr is None:
            # default port is job-specific
            port = os.environ.get("RMHD_COORDINATOR_PORT")
            _job = os.environ.get("SLURM_JOB_ID", "")
            if port is None and _job.isdigit():
                port = str(20000 + int(_job) % 20000)
            elif port is None and runtime.rank == 0:
                # plain mpirun (no Slurm)
                with socket.socket() as _s:
                    _s.bind(("", 0))
                    port = str(_s.getsockname()[1])
            addr = f"{socket.gethostbyname(socket.gethostname())}:{port}" if runtime.rank == 0 else None
            addr = runtime.comm.bcast(addr, root=0)
        kw = {}
        ids = _local_device_ids(runtime)
        if ids is not None:
            kw["local_device_ids"] = ids
        try:
            jax.distributed.initialize(coordinator_address=addr, num_processes=runtime.size,
                                       process_id=runtime.rank, **kw)
        except RuntimeError as e:  # jax refuses this once the local backend exists
            raise RuntimeError(
                "comm_backend='jax': jax.distributed.initialize() failed — construct the "
                "first Parameters(comm_backend='jax') BEFORE any jax device work "
                "(jax.devices(), any jit call) in the process.") from e
        _dist_initialized = True
    ndev = get_mesh().size
    if ndev % runtime.size:
        raise ValueError(f"comm_backend='jax': global device count {ndev} must be a multiple "
                         f"of the process count {runtime.size}")
    if nz % ndev:
        raise ValueError(f"comm_backend='jax': nz={nz} must be divisible by the "
                         f"global device count {ndev}")

def halo_exchange(f, params, width=2):
    # parameterized-width halo exchange along the z axis (axis 1) -> (recv_left, recv_right)
    # neighbor slabs, each `width` planes deep. width is a static Python int
    rt = params.runtime
    nz_local = f.shape[1]
    if width < 1 or width > nz_local:
        raise ValueError(f"halo_exchange: width={width} must be >= 1 and <= nz_local={nz_local} "
                         "(width > nz_local would need multi-hop neighbor comms, unsupported)")
    if rt.backend == "serial":
        # single process, periodic z: the only neighbor is self. Same identities the size-1
        # mpi4jax self-sendrecv produces; before any cart_comm/mesh use (serial has neither).
        return f[:,-width:,:,:], f[:,:width,:,:]
    if rt.backend == "mpi4jax":
        send_left = f[:,:width,:,:]
        send_right = f[:,-width:,:,:]
        recv_right = mpi4jax.sendrecv(send_left, send_left, dest=rt.left_neighbor, source=rt.right_neighbor,
                                         comm=rt.cart_comm, sendtag=101, recvtag=101)
        recv_left = mpi4jax.sendrecv(send_right, send_right, dest=rt.right_neighbor, source=rt.left_neighbor,
                                        comm=rt.cart_comm, sendtag=102, recvtag=102)
        return recv_left, recv_right
    if rt.backend == "jax":
        # ppermute perm entries are (source, destination): shifting each device's first `width`
        # planes DOWN the mesh (i -> i-1) makes device j receive device j+1's planes, i.e.
        # exactly the mpi4jax sendrecv(dest=left, source=right) that fills recv_right.
        n = get_mesh().size
        recv_right = jax.lax.ppermute(f[:,:width,:,:], Z_AXIS, [(i, (i-1) % n) for i in range(n)])
        recv_left = jax.lax.ppermute(f[:,-width:,:,:], Z_AXIS, [(i, (i+1) % n) for i in range(n)])
        return recv_left, recv_right
    _unknown_backend(rt.backend)

def allreduce_sum(x, params):
    # global SUM over the z-decomposition (scalar or array x); identity when not decomposed.
    rt = params.runtime
    if rt.backend == "serial" or rt.cart_comm is None:
        return x
    if rt.backend == "mpi4jax":
        return mpi4jax.allreduce(x, op=MPI.SUM, comm=rt.cart_comm)
    if rt.backend == "jax":
        return jax.lax.psum(x, Z_AXIS)
    _unknown_backend(rt.backend)

def allreduce_max(x, params):
    # global MAX over the z-decomposition (scalar or array x); identity when not decomposed.
    rt = params.runtime
    if rt.backend == "serial" or rt.cart_comm is None:
        return x
    if rt.backend == "mpi4jax":
        return mpi4jax.allreduce(x, op=MPI.MAX, comm=rt.cart_comm)
    if rt.backend == "jax":
        return jax.lax.pmax(x, Z_AXIS)
    _unknown_backend(rt.backend)

##########################################
# "jax" backend: shard_map + array layout #
##########################################

def state_specs():
    # shard_map specs for a SimulationState: z is fields axis 1, everything else is
    # replicated (forcing_state/key/scale and t are identical on every rank by construction).
    return SimulationState(t=P(), fields=P(None, Z_AXIS), forcing_state=P(),
                           forcing_key=P(), forcing_scale=P())

def _shard_map(f, in_specs, out_specs):
    # shard_map compat shim: public jax.shard_map (>=0.6) with the experimental one as fallback.
    if hasattr(jax, "shard_map"):
        return jax.shard_map(f, mesh=get_mesh(), in_specs=in_specs, out_specs=out_specs, check_vma=False)
    from jax.experimental.shard_map import shard_map
    return shard_map(f, mesh=get_mesh(), in_specs=in_specs, out_specs=out_specs, check_rep=False)

def shard_call(f, params, kgrid, nextra=0, out_specs=None):
    # Wraps f(state, kgrid, *nextra replicated scalars) in the z-mesh shard_map context the
    # ppermute/psum/pmax branches above require. Callers only use it for comm_backend=="jax".
    from .grids import kgrid_specs  # local import: grids imports comms at module level
    in_specs = (state_specs(), kgrid_specs(kgrid)) + (P(),)*nextra
    return _shard_map(f, in_specs, state_specs() if out_specs is None else out_specs)

def _z_spec(z_axis):
    # PartitionSpec sharding a single array axis over the z mesh.
    return P(*([None]*z_axis + [Z_AXIS]))

def to_global(x, params, z_axis=None):
    # process-local array -> global jax.Array on the z mesh (z_axis=None: replicated).
    rt = params.runtime
    mesh = get_mesh()
    per_proc = mesh.size // rt.size  # devices this process owns (1 in production)
    devs = mesh.devices.reshape(-1)[rt.rank*per_proc:(rt.rank+1)*per_proc]
    if z_axis is None:
        # replicated: every device of this process holds the whole (identical) value
        shards = [jax.device_put(x, d) for d in devs]
        return jax.make_array_from_single_device_arrays(jnp.shape(x),
                                                        NamedSharding(mesh, P()), shards)
    pieces = jnp.split(x, per_proc, axis=z_axis)  # jnp: stays on device, no host round-trip
    shards = [jax.device_put(pc, d) for pc, d in zip(pieces, devs)]
    gshape = list(jnp.shape(x))
    gshape[z_axis] *= rt.size
    return jax.make_array_from_single_device_arrays(tuple(gshape),
                                                    NamedSharding(mesh, _z_spec(z_axis)), shards)

def to_local(x, params, z_axis=None):
    # global jax.Array -> this process's addressable piece (host side: orbax, diagnostics).
    if z_axis is None:
        # replicated: take the local shard directly
        return x.addressable_shards[0].data
    shards = sorted(x.addressable_shards, key=lambda s: s.index[z_axis].start or 0)
    if len(shards) == 1:
        return shards[0].data  # production (one device per process): no copy, no gather
    # >1 device per process (e.g. the forced-host-device test): join on the host
    return jnp.asarray(np.concatenate([np.asarray(s.data) for s in shards], axis=z_axis))

def state_to_global(state, params):
    # SimulationState of process-local arrays -> z-sharded global arrays (jax backend only).
    return SimulationState(t=to_global(state.t, params), fields=to_global(state.fields, params, z_axis=1),
                           forcing_state=to_global(state.forcing_state, params),
                           forcing_key=to_global(state.forcing_key, params),
                           forcing_scale=to_global(state.forcing_scale, params))

def state_to_local(state, params):
    # inverse of state_to_global, for host-side consumers (diagnostics, tests). NOT on the
    # checkpoint path: orbax is handed the global arrays directly under comm_backend="jax".
    return SimulationState(t=to_local(state.t, params), fields=to_local(state.fields, params, z_axis=1),
                           forcing_state=to_local(state.forcing_state, params),
                           forcing_key=to_local(state.forcing_key, params),
                           forcing_scale=to_local(state.forcing_scale, params))
