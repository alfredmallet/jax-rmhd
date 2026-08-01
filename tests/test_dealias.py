# 2/3-rule dealias mask (grids.dealias_mask) and its application in run.initialize.
# pytest: single-process (stub). Savio driver:
# `python tests/test_dealias.py` -- everything here is perpendicular-plane-only, so
# it's rank-independent (identical mask on every z-rank).
from _rmhd_testing import bootstrap, checks, ctx, make_state

bootstrap()

import jax.numpy as jnp

# nx=ny=12: both cleanly divisible by 3, so the elliptical cutoff (|n|/(n_grid/3))^2
# < 1 lands exactly on integer mode-number boundaries -- no ambiguity from rounding.
_NX = _NY = 12


def test_dealias_mask_cutoff_boundary_exact():
    params, kgrid = ctx(dims=2, nx=_NX, ny=_NY)
    mask = kgrid.dealias  # (nx, nky)
    nx, nky = mask.shape
    ny = 2 * (nky - 1)
    # Mode numbers exactly as dealias_mask derives them internally (fftfreq*n).
    kx_modes = jnp.fft.fftfreq(nx) * nx
    ky_modes = jnp.fft.rfftfreq(ny) * ny
    with checks() as c:
        # kx edge, at ky=0 (n_y=0): cutoff radius nx/3 = 4 exactly -- |n_x|<4 kept,
        # |n_x|>=4 (including the exact boundary n_x=4/-4) rejected.
        expected_kx = jnp.abs(kx_modes) < nx / 3.0
        c.check("kx cutoff boundary indices exact (ky=0 row)",
                bool(jnp.array_equal(mask[:, 0], expected_kx)),
                f"got {mask[:, 0]}, expected {expected_kx}")
        # ky edge, at kx=0 (n_x=0): cutoff radius ny/3 = 4 exactly.
        expected_ky = ky_modes < ny / 3.0
        c.check("ky cutoff boundary indices exact (kx=0 column)",
                bool(jnp.array_equal(mask[0, :], expected_ky)),
                f"got {mask[0, :]}, expected {expected_ky}")
        # sanity: boundary itself actually excluded (not a vacuously true check)
        c.check("kx boundary mode (n_x=4) is excluded, n_x=3 kept (sanity)",
                bool(mask[4, 0] == False) and bool(mask[3, 0] == True))  # noqa: E712


def test_dealias_mask_kx_symmetric():
    # kx is full two-sided; the cutoff only depends on kx^2, so the mask must be
    # symmetric under kx -> -kx (mode-number sign flip / mirrored index).
    params, kgrid = ctx(dims=2, nx=16, ny=16)
    mask = kgrid.dealias
    nx = mask.shape[0]
    mirrored = mask[(-jnp.arange(nx)) % nx, :]
    with checks() as c:
        c.check("dealias mask is exactly symmetric under kx -> -kx",
                bool(jnp.array_equal(mask, mirrored)))


def _high_mode_ic(x, y):
    # An above-cutoff kx=5 mode (with nx=ny=12 the cutoff is |n_x|<4) PLUS an
    # in-cutoff kx=1 mode, so the "did not zero everything" sanity check below is
    # satisfied by a real surviving mode rather than FFT round-off garbage.
    base = jnp.cos(5 * x) + 0.5 * jnp.cos(x) + 0.0 * y  # +0*y broadcasts to (1,nx,ny)
    psi = jnp.zeros_like(base)
    return jnp.stack([base, psi], axis=0)


def test_initialize_zeroes_above_cutoff_ic_mode():
    params, _ = ctx(dims=2, nx=_NX, ny=_NY)
    state = make_state(params, ic=_high_mode_ic)
    phik = state.fields[0, 0]  # (nkx, nky), z axis squeezed (nz=1 in 2D)
    # cos(5x) splits into conjugate peaks at n_x=+5 and n_x=-5 (index nx-5), ky=0.
    idx_pos = 5
    idx_neg = _NX - 5
    with checks() as c:
        c.check("initialize zeroes the above-cutoff +kx mode exactly",
                complex(phik[idx_pos, 0]) == 0j,
                f"got {complex(phik[idx_pos, 0])!r}")
        c.check("initialize zeroes the above-cutoff -kx mode exactly",
                complex(phik[idx_neg, 0]) == 0j,
                f"got {complex(phik[idx_neg, 0])!r}")
        c.check("initialize keeps the in-cutoff kx=1 mode (sanity: mask not over-zealous)",
                complex(phik[1, 0]) != 0j)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
