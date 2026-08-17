# reads RMHD_PRECISION exactly ONCE, at taranis import time, and owns the
# "what is FIELD precision" question repo-wide
# import only after taranis/__init__.py has set jax_enable_x64
import os

import jax.numpy as jnp

precision = os.environ.get("RMHD_PRECISION", "32")
if precision not in ("32", "64"):
    raise ValueError(f"RMHD_PRECISION must be '32' or '64', got {precision!r}")

if precision == "64":
    ftype = jnp.float64
    ctype = jnp.complex128
else:
    ftype = jnp.float32
    ctype = jnp.complex64
