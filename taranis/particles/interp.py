# Grid -> particle interpolation. gather() is the production path (periodic bilinear
# from the collocation grid); gather_spectral() is an exact but O(N*nkx*nky)
# validation-only evaluation from the rfft2 coefficients.
#
# Real-space field arrays are indexed [component, z, x, y]; the collocation grid is
# x_i = i*Lx/nx, y_j = j*Ly/ny (run.initialize). Particle positions are (N,3) float64 and
# may be unfolded (outside [0,L)); the fold happens here. The z axis is carried
# everywhere but Phase A only implements nz_local == 1.
import jax.numpy as jnp
import jax.numpy.fft as ft


def grid_coords(params):
    # collocation coordinates of the perpendicular grid
    x = jnp.linspace(0.0, params.Lx, params.nx, endpoint=False, dtype=jnp.float64)
    y = jnp.linspace(0.0, params.Ly, params.ny, endpoint=False, dtype=jnp.float64)
    return x, y


def _check(fields, name):
    assert fields.ndim == 4, f"{name}: expected (ncomp, nz, nkx-or-nx, nky-or-ny), got {fields.shape}"
    assert fields.shape[1] == 1, (
        f"{name}: nz_local={fields.shape[1]}, but z-interpolation is Phase B "
        f"(plans/TESTPART_PLAN.md) — Phase A requires nz_local == 1")


def _cell(coord, L, d, n):
    # periodic fold, then the lower cell index, upper cell index and linear weight
    s = jnp.mod(coord, L) / d
    lo = jnp.floor(s)
    w = s - lo
    i0 = lo.astype(jnp.int32) % n   # mod(coord,L) can round up to exactly L
    return i0, (i0 + 1) % n, w


def gather(fields, pos, params):
    # periodic bilinear sample of fields (ncomp, nz, nx, ny) at pos (N,3) -> (N, ncomp)
    # float64. The z coordinate is ignored (nz == 1).
    _check(fields, "gather")
    f = fields[:, 0].astype(jnp.float64)
    i0, i1, wx = _cell(pos[:, 0].astype(jnp.float64), params.Lx, params.dx, params.nx)
    j0, j1, wy = _cell(pos[:, 1].astype(jnp.float64), params.Ly, params.dy, params.ny)
    out = ((1.0 - wx) * ((1.0 - wy) * f[:, i0, j0] + wy * f[:, i0, j1])
           + wx * ((1.0 - wy) * f[:, i1, j0] + wy * f[:, i1, j1]))
    return out.T


def _kvecs(params):
    kx = 2.0 * jnp.pi * ft.fftfreq(params.nx, d=params.Lx / params.nx, dtype=jnp.float64)
    ky = 2.0 * jnp.pi * ft.rfftfreq(params.ny, d=params.Ly / params.ny, dtype=jnp.float64)
    return kx, ky


def gather_spectral(fields_k, pos, params):
    # EXACT evaluation of an rfft2 field (ncomp, nz, nkx, nky) at pos (N,3) -> (N, ncomp)
    # float64, in taranis's unnormalized convention f(x) = (1/(nx*ny)) sum_k F_k e^{ik.x}.
    # The stored half-plane is completed by hermiticity: columns 0 < ky < Nyquist count
    # twice, ky = 0 and (even ny) the ky-Nyquist column once; the sum is real.
    # VALIDATION ONLY: O(N*nkx*nky) work and memory, never a production gather.
    _check(fields_k, "gather_spectral")
    F = fields_k[:, 0].astype(jnp.complex128)
    kx, ky = _kvecs(params)
    x = jnp.mod(pos[:, 0].astype(jnp.float64), params.Lx)
    y = jnp.mod(pos[:, 1].astype(jnp.float64), params.Ly)
    phase = jnp.exp(1j * (x[:, None, None] * kx[None, :, None]
                          + y[:, None, None] * ky[None, None, :]))
    w = jnp.full((ky.size,), 2.0, dtype=jnp.float64).at[0].set(1.0)
    if params.ny % 2 == 0:
        w = w.at[-1].set(1.0)
    vals = jnp.sum(w * jnp.real(F[:, None, :, :] * phase[None, :, :, :]), axis=(2, 3))
    return (vals / (params.nx * params.ny)).T
