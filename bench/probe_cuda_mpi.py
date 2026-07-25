# CUDA-aware-MPI probe: runs the exact halo pattern of jax_rmhd.comms.halo_exchange (two
# mpi4jax.sendrecv in a ring) on device buffers, checks correctness and times it, so a
# CUDA-aware openmpi can be told apart from host-staged transport (MPI4JAX_USE_CUDA_MPI=0).
# usage: srun --ntasks=2 --gpus-per-task=1 python bench/probe_cuda_mpi.py [nx]  (see slurms/probe_cuda_mpi.sh)
import os, sys, time
import numpy as np, jax, jax.numpy as jnp
from mpi4py import MPI
import mpi4jax

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
nx = int(sys.argv[1]) if len(sys.argv) > 1 else 512
nky = nx // 2 + 1
left, right = (rank - 1) % size, (rank + 1) % size

# One line per rank: this is also the GPU-binding check (each rank must see exactly 1 device).
print(f"[rank {rank}/{size}] host={MPI.Get_processor_name()} "
      f"MPI4JAX_USE_CUDA_MPI={os.environ.get('MPI4JAX_USE_CUDA_MPI', '<unset>')} "
      f"platform={jax.default_backend()} local_devices={jax.local_devices()} "
      f"mpi4jax={mpi4jax.__version__} jax={jax.__version__}", flush=True)
comm.Barrier()

@jax.jit
def exchange(f):
    # same op pair, tags and directions as comms.halo_exchange
    sl, sr = f[:, :2], f[:, -2:]
    recv_right = mpi4jax.sendrecv(sl, sl, dest=left, source=right, comm=comm, sendtag=101, recvtag=101)
    recv_left = mpi4jax.sendrecv(sr, sr, dest=right, source=left, comm=comm, sendtag=102, recvtag=102)
    return recv_left, recv_right

for nz in (4, 16, 64):
    f = jnp.full((2, nz, nx, nky), rank + 1, dtype=jnp.complex64)
    rl, rr = exchange(f)
    jax.block_until_ready((rl, rr))
    ok = bool(np.all(np.asarray(rl) == left + 1) and np.all(np.asarray(rr) == right + 1))
    nbytes = 2 * 2 * nx * nky * 8  # one sendrecv payload: (nfields,2,nx,nky) complex64
    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(50):
        rl, rr = exchange(f)
    jax.block_until_ready((rl, rr))
    comm.Barrier()
    dt = (time.perf_counter() - t0) / 50
    if rank == 0:
        print(f"probe nz={nz:3d} payload={nbytes/1e6:6.3f} MB  correct={ok}  "
              f"{dt*1e6:9.1f} us/halo-pair  {2*nbytes/dt/1e9:6.2f} GB/s", flush=True)

if rank == 0:
    print("probe done: correct=True on a cuda platform means mpi4jax accepted device buffers; "
          "compare the us/halo-pair between MPI4JAX_USE_CUDA_MPI=1 and =0 runs.", flush=True)
