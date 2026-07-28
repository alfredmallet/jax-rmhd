import os
import jax

precision = os.environ.get("RMHD_PRECISION","32")

if precision == "64":
    jax.config.update("jax_enable_x64", True)
    print("jax is using 64bit precision.")
else:
    jax.config.update("jax_enable_x64", False)
    print("jax is using 32bit precision.")

# persistent JIT compilation cache, keyed off RMHD_COMPILATION_CACHE dir path.
cache_dir = os.environ.get("RMHD_COMPILATION_CACHE")
if cache_dir:
    try:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    except Exception as e:
        print(f"Warning: failed to set up JAX compilation cache at {cache_dir}: {e}")

from .types import SimulationState
from .grids import K_Grids, setup_kgrids
from .config import Parameters
from .snapshot_io import snapshot_manager_setup
from .run import simulate,simulate_scan,estimate_good_nblock,initialize
