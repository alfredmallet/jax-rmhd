# Independent real-space Parseval cross-check on the spectral energy path
# (shared_physics.perp_reduce / perp_inner_product, which diagnostics.energy wraps).
# A check is only worth having if it shares no code with the thing it checks: the
# real-space path below goes ifft -> pointwise square -> local sum -> allreduce,
# touching none of the perp_reduce normalization machinery.
# pytest: single-process (stub). Savio driver: `mpirun -n N python
# tests/test_energy_parseval.py` (N=1,2,4).
from _rmhd_testing import bootstrap, checks, ctx, make_state, multimode_ic

bootstrap()

import jax
import jax.numpy as jnp

from jax_rmhd import _precision, comms, diagnostics, grids
from jax_rmhd.physics.shared_physics import gradk, perp_inner_product


def _realspace_grad_meansq(fieldk, kgrid, params):
    # <|grad_perp f|^2>, volume-averaged over the GLOBAL domain: independent
    # real-space path (grad components summed explicitly; local z-slab sum, then
    # allreduce).
    g = grids.ifft(gradk(fieldk, kgrid), params)  # (nz_local, 2, nx, ny) real space
    local_sum = jnp.sum(g**2)
    total = comms.allreduce_sum(local_sum, params)  # no-op unless z-decomposed
    return total / (params.nz * params.nx * params.ny)


def test_parseval_spectral_vs_realspace():
    params, kgrid = ctx()
    state = make_state(params, ic=multimode_ic)
    spec = perp_inner_product(state.fields[:2], state.fields[:2], kgrid, params, batch=True)
    real_phi = _realspace_grad_meansq(state.fields[0], kgrid, params)
    real_psi = _realspace_grad_meansq(state.fields[1], kgrid, params)

    # Same quantity, different fp path.
    rtol = 1e-12 if _precision.precision == "64" else 2e-5
    with checks() as c:
        c.check("Parseval: spectral <|grad phi|^2> matches real-space path",
                bool(jnp.allclose(spec[0], real_phi, rtol=rtol)),
                f"spectral={spec[0]:.15e}, real={real_phi:.15e}")
        c.check("Parseval: spectral <|grad psi|^2> matches real-space path",
                bool(jnp.allclose(spec[1], real_psi, rtol=rtol)),
                f"spectral={spec[1]:.15e}, real={real_psi:.15e}")
        # sanity: nonzero (a bug that zeros both sides would "pass" the equality)
        c.check("Parseval: energies are nonzero",
                bool(spec[0] > 0) and bool(spec[1] > 0))


def test_energy_is_half_batch_reduction():
    # diagnostics.energy is exactly half the batch reduction (E = 0.5*<|grad|^2>).
    params, kgrid = ctx()
    state = make_state(params, ic=multimode_ic)
    spec = perp_inner_product(state.fields[:2], state.fields[:2], kgrid, params, batch=True)
    E_kin, E_mag = diagnostics.energy(state, kgrid, params)
    assert jnp.array_equal(E_kin, 0.5 * spec[0])
    assert jnp.array_equal(E_mag, 0.5 * spec[1])


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
