import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional

class SimulationState(NamedTuple):
    t: float
    fields: jnp.ndarray         # shape (nfields, nz_local, nkx, nky), complex
    forcing_state: jnp.ndarray  # shape (n_ou, 2, nkx, nky), complex
                                # axis 0: n_ou = 1 ("momentum" mode) or 2 ("elsasser" mode: z+, z-)
                                # axis 1: [A, B] cosine/sine z-envelope coefficients
    forcing_key: jax.Array      # JAX PRNG key (typed key)
    forcing_scale: Optional[jnp.ndarray] = None
                                # shape (n_ou,), real: per-step power-normalization scale,
                                # updated once per full step in run.py when
                                # params.forcing_norm_per_step. ALWAYS a concrete array
                                # (zeros when unused) in states built by initialize /
                                # load_snapshot; None default so error is loud
