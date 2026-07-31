# Minimal multi-process NCCL reproducer for the jax backend hang (jobs 35861466/515/835:
# bootstrap + rings connect, first collective never completes, all ranks blocked in _value).
# No jax_rmhd imports — isolates infra from our code. Run 4-rank on 4 GPUs, interactively:
#
#   salloc -A fc_kawturb -p savio4_gpu -q a5k_gpu4_normal -N1 -n4 -c4 --gres=gpu:A5000:4 -t 0:30:00
#   # inside the allocation (module load anaconda3 gcc openmpi; source activate jax_gpu;
#   # unset PYTHONPATH; export PYTHONNOUSERSITE=1 and the NVLIBS block, as in the sbatch):
#   srun --mpi=pmix --ntasks=4 python bench/nccl_repro.py            # our config (MPI + jax.distributed)
#   RMHD_REPRO_NOMPI=1 srun --mpi=pmix --ntasks=4 python bench/nccl_repro.py   # no mpi4py: pure jax
#
# Knob ladder if the baseline hangs (set before srun, one at a time):
#   NCCL_CUMEM_ENABLE=0            # cuMem-handle P2P off        (already known: still hangs)
#   NCCL_P2P_DISABLE=1             # all GPU P2P off -> SHM
#   NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1   # force network/socket transport
#   NCCL_ALGO=Ring NCCL_PROTO=Simple        # simplest collective schedule
# Whichever first prints "psum OK" identifies the broken layer — put those exports in the
# GPU sbatch scripts and report the finding (with this file) to Savio support.
import os, socket, faulthandler
faulthandler.dump_traceback_later(60, repeat=True)  # stacks every 60 s if we hang

use_mpi = not os.environ.get("RMHD_REPRO_NOMPI")
if use_mpi:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    port = os.environ.get("RMHD_COORDINATOR_PORT", "29777")
    addr = f"{socket.gethostbyname(socket.gethostname())}:{port}" if rank == 0 else None
    addr = comm.bcast(addr, root=0)
else:
    rank = int(os.environ["SLURM_PROCID"]); size = int(os.environ["SLURM_NTASKS"])
    addr = f"{os.environ['SLURMD_NODENAME']}:29777"

import jax
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
nvis = len(visible.split(",")) if visible else 0
local = [0] if nvis == 1 else ([rank % nvis] if nvis > 1 else None)  # mirror comms._local_device_ids
jax.distributed.initialize(coordinator_address=addr, num_processes=size, process_id=rank,
                           **({"local_device_ids": local} if local is not None else {}))
print(f"[r{rank}] init ok mpi={use_mpi} CVD={visible!r} local={local} "
      f"devices={jax.local_devices()} global={jax.device_count()}", flush=True)

import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np

mesh = Mesh(np.array(sorted(jax.devices(), key=lambda d: (d.process_index, d.id))), ("z",))
x = jax.make_array_from_single_device_arrays(
    (size, 8), jax.sharding.NamedSharding(mesh, P("z", None)),
    [jax.device_put(jnp.full((1, 8), float(rank + 1)), jax.local_devices()[0])])
f = jax.jit(shard_map(lambda a: jax.lax.psum(a, "z"), mesh=mesh, in_specs=P("z", None), out_specs=P("z", None)))
out = f(x)
got = float(np.asarray(out.addressable_shards[0].data)[0, 0])
want = size * (size + 1) / 2.0
print(f"[r{rank}] psum OK got={got} want={want} {'PASS' if got == want else 'FAIL'}", flush=True)
