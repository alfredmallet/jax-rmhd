# Communication abstraction layer: every distributed op used by the solver lives here, so
# a new transport (NCCL/shard_map, ...) only needs a new branch instead of edits to physics.
# Backend is chosen by params.comm_backend (static, validated in Parameters.__init__), so
# the dispatch below is plain-python branching, not lax.cond.
import mpi4jax
from mpi4py import MPI

# Backends implemented here; Parameters.__init__ rejects anything else at construction.
COMM_BACKENDS = ("mpi4jax",)

def _unknown_backend(params):
    raise ValueError(f"unknown comm_backend {params.comm_backend!r}, expected one of {COMM_BACKENDS}")

def halo_exchange(f, params):
    # Two-wide halo exchange along the z axis (axis 1) -> (recv_left, recv_right) neighbor slabs.
    if params.comm_backend == "mpi4jax":
        send_left = f[:,:2,:,:]
        send_right = f[:,-2:,:,:]
        recv_right = mpi4jax.sendrecv(send_left, send_left, dest=params.left_neighbor, source=params.right_neighbor,
                                         comm=params.cart_comm, sendtag=101, recvtag=101)
        recv_left = mpi4jax.sendrecv(send_right, send_right, dest=params.right_neighbor, source=params.left_neighbor,
                                        comm=params.cart_comm, sendtag=102, recvtag=102)
        return recv_left, recv_right
    _unknown_backend(params)

def allreduce_sum(x, params):
    # Global SUM over the z-decomposition (scalar or array x); identity when not decomposed.
    if params.cart_comm is None:
        return x
    if params.comm_backend == "mpi4jax":
        return mpi4jax.allreduce(x, op=MPI.SUM, comm=params.cart_comm)
    _unknown_backend(params)

def allreduce_max(x, params):
    # Global MAX over the z-decomposition (scalar or array x); identity when not decomposed.
    if params.cart_comm is None:
        return x
    if params.comm_backend == "mpi4jax":
        return mpi4jax.allreduce(x, op=MPI.MAX, comm=params.cart_comm)
    _unknown_backend(params)
