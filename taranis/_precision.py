# reads TARANIS_PRECISION exactly ONCE, at taranis import time, and owns the
# "what is FIELD precision" question repo-wide
# import only after taranis/__init__.py has set jax_enable_x64
import os

import jax.numpy as jnp

# the pre-rename name is never honored: a stale recipe setting it alone would
# otherwise run silently at the default 32.
if "RMHD_PRECISION" in os.environ and "TARANIS_PRECISION" not in os.environ:
    raise RuntimeError(
        "RMHD_PRECISION is no longer read; set TARANIS_PRECISION=<32|64> "
        "instead (renamed 2026-08-17)")

precision = os.environ.get("TARANIS_PRECISION", "32")
if precision not in ("32", "64"):
    raise ValueError(f"TARANIS_PRECISION must be '32' or '64', got {precision!r}")

if precision == "64":
    ftype = jnp.float64
    ctype = jnp.complex128
else:
    ftype = jnp.float32
    ctype = jnp.complex64
