# Gradient transforms: shared_physics.grad_fields and its two callers.
# pytest: single-process (stub). Savio driver: `python tests/test_grad_memory.py`.
# The reference functions below are the pre-F1 implementation (one stacked field
# array, one batched ifft); the production tuple path must reproduce them bitwise.
from _rmhd_testing import bootstrap, checks, ctx, make_state

bootstrap()

import contextlib

import jax.numpy as jnp
import numpy as np

import taranis as jr
from taranis import grids
from taranis.physics import gdi, rmhd, shared_physics
from taranis.physics.shared_physics import gradk


def _ref_rmhd_grad(state, kgrid, params):
    phik = state.fields[0]
    psik = state.fields[1]
    vortk = -kgrid.ksq * phik
    jpark = -kgrid.ksq * psik
    fk = jnp.stack([phik, psik, vortk, jpark])
    return grids.ifft(gradk(fk, kgrid), params)


def _ref_gdi_grad(state, kgrid, params):
    Nk = state.fields[0]
    phik = state.fields[1]
    vortk = -kgrid.ksq * phik
    fk = jnp.stack([phik, Nk, vortk])
    return grids.ifft(gradk(fk, kgrid), params)


@contextlib.contextmanager
def _grad_chunk(n):
    old = shared_physics.GRAD_CHUNK
    shared_physics.GRAD_CHUNK = n
    try:
        yield
    finally:
        shared_physics.GRAD_CHUNK = old


def _ic3d(x, y, z):
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z) + 0.3 * jnp.sin(2 * x + y)
    psi = jnp.sin(x) * jnp.cos(y) * jnp.sin(z) + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([phi, psi], axis=0)


def _ic2d(x, y):
    phi = jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y)
    psi = jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([phi, psi], axis=0)


def _rmhd_cases():
    # (label, params, kgrid, ic) at the three geometries F1 spans
    yield ("2D 64^2", *ctx(dims=2, nx=64, ny=64), _ic2d)
    yield ("FD-z 64^2x16", *ctx(nx=64, ny=64, nz=16), _ic3d)
    yield ("z_spectral 64^2x16", *ctx(nx=64, ny=64, nz=16, z_spectral=True), _ic3d)


def _equal_per_field(new, ref):
    return all(np.array_equal(np.asarray(a), np.asarray(ref[i])) for i, a in enumerate(new))


def test_rmhd_grad_matches_stacked_reference():
    with checks() as c:
        for label, params, kgrid, ic in _rmhd_cases():
            state = make_state(params, ic=ic)
            ref = _ref_rmhd_grad(state, kgrid, params)
            new = rmhd.grad(state, kgrid, params)
            c.check(f"{label}: grad is a 4-tuple of (2,nz,nx,ny) real arrays",
                    isinstance(new, tuple) and len(new) == 4
                    and all(g.shape == ref[0].shape for g in new)
                    and all(g.dtype == ref.dtype for g in new))
            c.check(f"{label}: grad bitwise equals the stacked reference",
                    _equal_per_field(new, ref),
                    f"max|dev| = {max(float(np.max(np.abs(np.asarray(g) - np.asarray(ref[i])))) for i, g in enumerate(new))}")


def test_gdi_grad_matches_stacked_reference():
    common = dict(cfl_safety=0.5, adaptive_timestep=False, dt=0.01, eqtype="GDI",
                  eqpars=dict(Ln=2.0, nu_in=0.3, v0=1.0, gpar_fac=1.0, diss=0.0, hyper=1))
    params = jr.Parameters(nx=64, ny=64, Lx=2 * np.pi, Ly=2 * np.pi, dims=2, **common)
    kgrid = jr.setup_kgrids(params)
    state = make_state(params, ic=_ic2d)
    ref = _ref_gdi_grad(state, kgrid, params)
    new = gdi.grad(state, kgrid, params)
    with checks() as c:
        c.check("GDI 2D 64^2: grad is a 3-tuple", isinstance(new, tuple) and len(new) == 3)
        c.check("GDI 2D 64^2: grad bitwise equals the stacked reference",
                _equal_per_field(new, ref))


def test_grad_chunk_batching_matches_chunk_one():
    # GRAD_CHUNK is read at trace time, so setting it here changes the next call.
    # chunk=4 puts all four RMHD fields in one transform, i.e. the pre-F1 batched path.
    with checks() as c:
        for label, params, kgrid, ic in _rmhd_cases():
            state = make_state(params, ic=ic)
            with _grad_chunk(1):
                g1 = rmhd.grad(state, kgrid, params)
            for n in (2, 4):
                with _grad_chunk(n):
                    gn = rmhd.grad(state, kgrid, params)
                c.check(f"{label}: GRAD_CHUNK={n} bitwise equals GRAD_CHUNK=1",
                        _equal_per_field(gn, g1),
                        f"max|dev| = {max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(gn, g1))}")


def test_grad_fields_shapes_and_default_chunk():
    params, kgrid = ctx(nx=32, ny=32, nz=8)
    state = make_state(params, ic=_ic3d)
    out = shared_physics.grad_fields((state.fields[0],), kgrid, params)
    with checks() as c:
        c.check("default GRAD_CHUNK is 1", shared_physics.GRAD_CHUNK == 1)
        c.check("grad_fields returns one array per input field",
                isinstance(out, tuple) and len(out) == 1
                and out[0].shape == (2, params.nz, params.nx, params.ny))


def test_rhs_and_timestep_accept_the_tuple():
    # the consumers that index grads rather than unpacking it
    params, kgrid = ctx(nx=32, ny=32, nz=8, adaptive_timestep=True)
    state = make_state(params, ic=_ic3d)
    grads = rmhd.grad(state, kgrid, params)
    dt = float(rmhd.set_timestep(grads, params))
    nl = rmhd.NonlinearTerm(state, grads, kgrid, params)
    fd = rmhd.FDLinearTerm(state, grads, kgrid, params)
    with checks() as c:
        c.check("set_timestep on tuple grads gives a finite positive dt",
                np.isfinite(dt) and dt > 0.0, f"dt={dt}")
        c.check("NonlinearTerm/FDLinearTerm consume tuple grads",
                nl.shape == state.fields.shape and fd.shape == state.fields.shape)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
