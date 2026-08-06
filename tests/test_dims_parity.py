# 2D/3D parity: a z-INVARIANT 3D IC must evolve exactly like the corresponding
# dims=2 run, plane by plane -- every z-derivative in LinearTerm acts on constants
# (round-off-level output, see test_z_stencils' constant-field case) and the
# perpendicular dynamics are identical code.
#
# This is a LOUD halo-bug detector: any error in the z halo exchange (wrong
# neighbor, wrong width, wrong slab offsets) breaks the z-invariance of the fields
# immediately, and the "planes stay bitwise identical to each other" check catches
# it long before the (looser) 2D comparison does. z_diss keeps its production
# default (0.25) ON PURPOSE so the d4/dz4 stencil goes through the same halo path.
#
# dt is FIXED (ctx default adaptive_timestep=False): the adaptive CFL differs
# between 2D and 3D by the 1/dz term, which would desynchronize the trajectories.
#
# mpirun-safe: each rank compares its own local z-planes against its own full 2D
# run (dims=2 runs identically on every rank); nz=8 divisible by 4.
# Savio: `mpirun -n 4 python tests/test_dims_parity.py`.
from _rmhd_testing import bootstrap, checks, ctx, make_state

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np

from jax_rmhd import _precision
from jax_rmhd.run import block_of_steps
from jax_rmhd.timestepping import get_scheme

_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))
_NSTEPS = 20


def _g(x, y):
    # multi-mode O(1) perpendicular structure (nonlinear terms genuinely active)
    phi = jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y)
    psi = jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)
    return phi, psi


def _ic3d(x, y, z):
    phi, psi = _g(x, y)
    zero_z = 0.0 * z  # broadcast the z-invariant fields onto the local z slab
    return jnp.stack([phi + zero_z, psi + zero_z], axis=0)


def _ic2d(x, y):
    phi, psi = _g(x, y)
    return jnp.stack([phi, psi], axis=0)


def test_z_invariant_3d_matches_2d():
    p3, kg3 = ctx()          # dims=3 defaults (fixed dt=0.01, z_diss=0.25)
    p2, kg2 = ctx(dims=2)
    stepper, scheme = get_scheme("lsrk33")
    end3 = _advance(make_state(p3, ic=_ic3d), kg3, p3, _NSTEPS, scheme, stepper)
    end2 = _advance(make_state(p2, ic=_ic2d), kg2, p2, _NSTEPS, scheme, stepper)
    f3 = np.asarray(end3.fields)          # (2, nz_local, nkx, nky)
    f2 = np.asarray(end2.fields)[:, 0]    # (2, nkx, nky)

    # planes vs each other: identical inputs feed identical per-plane arithmetic,
    # so all local planes must stay BITWISE equal -- z-invariance is exact.
    planes_bitwise = bool(np.all(f3 == f3[:, :1]))

    # planes vs the 2D run: the 3D RHS adds LinearTerm's round-off-level output
    # (z stencils on constants) that the 2D branch skips entirely, so this is
    # round-off, not bitwise. Observed rel ~1e-15 (fp64) / ~1e-7 (fp32) after 20
    # steps of O(1) nonlinear dynamics.
    tol = 1e-14 if _precision.precision == "64" else 1e-5
    rel = float(np.max(np.abs(f3 - f2[:, None])) / np.max(np.abs(f2)))

    with checks() as c:
        c.check("all local z-planes stay bitwise identical to each other "
                "(halo/z-stencil parity)", planes_bitwise,
                f"max plane spread {np.max(np.abs(f3 - f3[:, :1])):.3e}")
        c.check(f"every z-plane matches the dims=2 run (rel {rel:.2e} < {tol:.0e})",
                rel < tol)
        c.check("2D and 3D runs end at the same (fixed-dt) time",
                float(end3.t) == float(end2.t))
        c.check("fields are finite and nonzero",
                bool(np.all(np.isfinite(f3))) and float(np.max(np.abs(f3))) > 0.0)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
