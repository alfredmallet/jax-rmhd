# Grid -> particle interpolation. gather() is the production path (periodic bilinear from
# the collocation grid, trilinear when the grid has a z axis); gather_spectral() is a
# validation-only evaluation from the stored Fourier coefficients.
#
# Real-space field arrays are indexed [component, z, x, y]; the collocation grid is
# x_i = i*Lx/nx, y_j = j*Ly/ny, z_k = k*Lz/nz (run.initialize). Particle positions are
# (N,3) float64 and may be unfolded (outside [0,L)); the fold happens here.
#
# gather_spectral is exact for what the field representation actually is: under
# z_spectral the arrays are rfftn over (z,x,y), so it is spectral in all three
# directions; without it they are one rfft2 per z plane, so it is spectral in (x,y) and
# linear in z (exact for a z-independent field).
import jax.numpy as jnp
import jax.numpy.fft as ft


def grid_coords(params):
    # collocation coordinates of the grid; z is the single plane z = 0 in 2D
    x = jnp.linspace(0.0, params.Lx, params.nx, endpoint=False, dtype=jnp.float64)
    y = jnp.linspace(0.0, params.Ly, params.ny, endpoint=False, dtype=jnp.float64)
    z = (jnp.linspace(0.0, params.Lz, params.nz, endpoint=False, dtype=jnp.float64)
         if params.spatial_dimensions == 3 else jnp.zeros((1,), dtype=jnp.float64))
    return x, y, z


def _check(fields, name, params):
    assert fields.ndim == 4, f"{name}: expected (ncomp, nz, nkx-or-nx, nky-or-ny), got {fields.shape}"
    assert fields.shape[1] == 1 or fields.shape[1] == params.nz, (
        f"{name}: nz_local={fields.shape[1]} against params.nz={params.nz}; particles need "
        f"the whole z domain on this rank (plans/TESTPART_PLAN.md §4)")


def _cell(coord, L, d, n):
    # periodic fold, then the lower cell index, upper cell index and linear weight
    s = jnp.mod(coord, L) / d
    lo = jnp.floor(s)
    w = s - lo
    i0 = lo.astype(jnp.int32) % n   # mod(coord,L) can round up to exactly L
    return i0, (i0 + 1) % n, w


def _bilinear(f, k, i0, i1, j0, j1, wx, wy):
    # bilinear sample of f (ncomp, nz, nx, ny) on z plane(s) k -> (ncomp, N); k is the
    # scalar 0 in 2D and a per-particle index array in 3D
    return ((1.0 - wx) * ((1.0 - wy) * f[:, k, i0, j0] + wy * f[:, k, i0, j1])
            + wx * ((1.0 - wy) * f[:, k, i1, j0] + wy * f[:, k, i1, j1]))


def gather(fields, pos, params):
    # periodic bilinear sample of fields (ncomp, nz, nx, ny) at pos (N,3) -> (N, ncomp)
    # float64, trilinear when nz > 1. The z coordinate is ignored when nz == 1.
    _check(fields, "gather", params)
    f = fields.astype(jnp.float64)
    i0, i1, wx = _cell(pos[:, 0].astype(jnp.float64), params.Lx, params.dx, params.nx)
    j0, j1, wy = _cell(pos[:, 1].astype(jnp.float64), params.Ly, params.dy, params.ny)
    if fields.shape[1] == 1:
        return _bilinear(f, 0, i0, i1, j0, j1, wx, wy).T
    k0, k1, wz = _cell(pos[:, 2].astype(jnp.float64), params.Lz, params.dz, params.nz)
    return ((1.0 - wz) * _bilinear(f, k0, i0, i1, j0, j1, wx, wy)
            + wz * _bilinear(f, k1, i0, i1, j0, j1, wx, wy)).T


def _kvecs(params):
    kx = 2.0 * jnp.pi * ft.fftfreq(params.nx, d=params.Lx / params.nx, dtype=jnp.float64)
    ky = 2.0 * jnp.pi * ft.rfftfreq(params.ny, d=params.Ly / params.ny, dtype=jnp.float64)
    return kx, ky


def _yweights(params):
    # the stored half-plane completed by hermiticity: columns 0 < ky < Nyquist count
    # twice, ky = 0 and (even ny) the ky-Nyquist column once; the weighted sum is real
    nky = params.ny // 2 + 1
    w = jnp.full((nky,), 2.0, dtype=jnp.float64).at[0].set(1.0)
    return w.at[-1].set(1.0) if params.ny % 2 == 0 else w


def _perp_eval(F, phase, w, params):
    # sum_k F_k e^{ik.x} over the perpendicular half-plane -> (ncomp, N). F is either
    # (ncomp, nkx, nky) (one shared plane) or (ncomp, N, nkx, nky) (a plane per particle);
    # phase is (N, nkx, nky).
    Fb = F[:, None, :, :] if F.ndim == 3 else F
    vals = jnp.sum(w * jnp.real(Fb * phase[None, :, :, :]), axis=(2, 3))
    return vals / (params.nx * params.ny)


def gather_spectral(fields_k, pos, params):
    # EXACT evaluation of an rfft2 field (ncomp, nz, nkx, nky) at pos (N,3) -> (N, ncomp)
    # float64, in taranis's unnormalized convention f(x) = (1/(nx*ny)) sum_k F_k e^{ik.x}.
    # With z_spectral the z axis is kz and the evaluation is spectral there too
    # (normalization 1/(nz*nx*ny)); without it the axis is real z and the two bracketing
    # planes are blended linearly, which is exact for a z-independent field.
    # VALIDATION ONLY: O(N*nkx*nky) work and memory per plane, never a production gather.
    _check(fields_k, "gather_spectral", params)
    F = fields_k.astype(jnp.complex128)
    kx, ky = _kvecs(params)
    w = _yweights(params)
    x = jnp.mod(pos[:, 0].astype(jnp.float64), params.Lx)
    y = jnp.mod(pos[:, 1].astype(jnp.float64), params.Ly)
    phase = jnp.exp(1j * (x[:, None, None] * kx[None, :, None]
                          + y[:, None, None] * ky[None, None, :]))
    if params.z_spectral:
        assert fields_k.shape[1] == params.nz, (
            f"gather_spectral: z_spectral coefficients need the full kz axis, got "
            f"{fields_k.shape[1]} of {params.nz} planes")
        kz = 2.0 * jnp.pi * ft.fftfreq(params.nz, d=params.Lz / params.nz, dtype=jnp.float64)
        z = jnp.mod(pos[:, 2].astype(jnp.float64), params.Lz)
        zphase = jnp.exp(1j * z[:, None] * kz[None, :])
        # the kz sum first, leaving a per-particle perp plane (ncomp, N, nkx, nky)
        Fz = jnp.moveaxis(jnp.tensordot(zphase, F, axes=([1], [1])), 0, 1)
        return (_perp_eval(Fz, phase, w, params) / params.nz).T
    if fields_k.shape[1] == 1:
        return _perp_eval(F[:, 0], phase, w, params).T
    k0, k1, wz = _cell(pos[:, 2].astype(jnp.float64), params.Lz, params.dz, params.nz)
    return ((1.0 - wz) * _perp_eval(F[:, k0], phase, w, params)
            + wz * _perp_eval(F[:, k1], phase, w, params)).T
