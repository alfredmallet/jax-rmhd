# Reads RMHD_PRECISION exactly ONCE, at taranis import time, and owns the
# "what is FIELD precision" question repo-wide.
#
# taranis/__init__.py sets jax_enable_x64=True unconditionally, so
# jax.config.read("jax_enable_x64") / jax.config.jax_enable_x64 no longer means
# "fields are fp32/fp64" -- it is always True. This module is the ONLY place
# allowed to infer field precision from RMHD_PRECISION; everything else
# (including tests) re-points here. See docs/numerics.md "Precision model" and
# CLAUDE.md / plans/PRECISION_PLAN.md.
#
# Import this module only after taranis/__init__.py has set jax_enable_x64 --
# importing any taranis submodule runs the package __init__.py first, so that
# is automatic.
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
