import jax
import jax.numpy as jnp
import jax.numpy.fft as ft
from jax.sharding import PartitionSpec as P
from typing import NamedTuple, Optional
from . import comms

# nb the code is only spectral in the perpendciular plane, so fourier stuff is all 2D

class K_Grids(NamedTuple):
    # Stores the wavenumber grids and precomputed concrete arrays derived from them.
    kx: jnp.ndarray
    ky: jnp.ndarray
    # Precomputed (in setup_kgrids) so these aren't recomputed as extra kernels every RHS
    # eval; None defaults keep K_Grids(kx=...,ky=...) constructible for standalone/tests.
    ksq_pc: Optional[jnp.ndarray] = None
    inv_ksq_pc: Optional[jnp.ndarray] = None
    dealias_pc: Optional[jnp.ndarray] = None
    hdiss_pc: Optional[jnp.ndarray] = None
    yfac: Optional[jnp.ndarray] = None
    fmask: Optional[jnp.ndarray] = None
    z_envcos: Optional[jnp.ndarray] = None
    z_envsin: Optional[jnp.ndarray] = None
    # (kx,ky) index arrays of the True entries of fmask (fixed shape), so forcing noise
    # is only drawn at shell modes rather than over the whole k-grid (ou_update).
    fidx_x: Optional[jnp.ndarray] = None
    fidx_y: Optional[jnp.ndarray] = None

    def ksq(self):
        # kx^2+ky^2; returns the precomputed array if available, else computes it.
        if self.ksq_pc is not None:
            return self.ksq_pc
        return self.kx**2 + self.ky**2

    def inv_ksq(self):
        # 1/ksq (0 at the zero mode); returns the precomputed array if available.
        if self.inv_ksq_pc is not None:
            return self.inv_ksq_pc
        return jnp.where(self.kx**2 + self.ky**2 > 0, 1.0/(self.kx**2 + self.ky**2), 0.0)

    def dealias_filter(self):
        # 2/3-rule elliptical dealiasing mask; returns the precomputed array if available.
        if self.dealias_pc is not None:
            return self.dealias_pc
        nx = jnp.shape(self.kx)[0]
        ny = 2 * (jnp.shape(self.ky)[1] - 1)
        kx_norm = self.kx / self.kx[1,0]
        ky_norm = self.ky / self.ky[0,1]
        return (kx_norm**2 / (nx / 3.0)**2 + ky_norm**2 / (ny / 3.0)**2) < 1.0

    def hdiss_exponents(self,params):
        # Hyperdissipation exponents per field; returns the precomputed array if available.
        if self.hdiss_pc is not None:
            return self.hdiss_pc
        diss=jnp.array(params.diss)
        diss_grid = diss.reshape(-1,1,1,1)
        return -diss_grid*self.ksq()**params.hyper

def _compute_dealias(kx_grid,ky_grid):
    # Standalone helper for the dealias filter formula (used by both the method fallback and setup_kgrids).
    nx = jnp.shape(kx_grid)[0]
    ny = 2 * (jnp.shape(ky_grid)[1] - 1)
    kx_norm = kx_grid / kx_grid[1,0]
    ky_norm = ky_grid / ky_grid[0,1]
    return (kx_norm**2 / (nx / 3.0)**2 + ky_norm**2 / (ny / 3.0)**2) < 1.0

def setup_kgrids(params):
    # Gets the wavenumber grid object from parameters, precomputing all the static
    # concrete arrays (ksq, inv_ksq, dealias, hdiss, y-doubling factor, forcing shell
    # mask/z-envelopes) once here rather than recomputing them every RHS eval.
    # The scaling here respects jax.numpy's fourier transform conventions
    # so that e.g. we can calculate derivatives correctly.
    kx = ft.fftfreq(params.nx) * params.nx * 2 * jnp.pi / params.Lx
    ky = ft.rfftfreq(params.ny) * params.ny * 2 * jnp.pi / params.Ly
    kx_grid = kx.reshape(-1, 1)
    ky_grid = ky.reshape(1, -1)

    ksq_pc = kx_grid**2 + ky_grid**2
    inv_ksq_pc = jnp.where(ksq_pc > 0, 1.0/ksq_pc, 0.0)
    dealias_pc = _compute_dealias(kx_grid,ky_grid)
    diss = jnp.array(params.diss).reshape(-1,1,1,1)
    hdiss_pc = -diss*ksq_pc**params.hyper

    nky = ky_grid.shape[-1]
    yfac = jnp.full((nky,), 2.0).at[0].set(1.0).at[-1].set(1.0)

    fmask = None
    fidx_x = None
    fidx_y = None
    z_envcos = None
    z_envsin = None
    if params.forcing:
        kunit = min(2*jnp.pi/params.Lx, 2*jnp.pi/params.Ly)
        nmin, nmax = params.fshell
        kmag_over_dk = jnp.sqrt(ksq_pc) / kunit
        fmask = (kmag_over_dk >= nmin) & (kmag_over_dk < nmax)
        # static shell index set (concrete here, outside jit) for shell-restricted noise;
        # only carried when opted in (params.forcing_shell_noise) — ou_update falls back
        # to the full-grid draw when these are None
        if params.forcing_shell_noise:
            fidx_x, fidx_y = jnp.nonzero(fmask)
        if params.spatial_dimensions == 3:
            z_local = local_z_coords(params)
            z_envcos = jnp.cos(2*jnp.pi*z_local/params.Lz)[:, None, None]
            z_envsin = jnp.sin(2*jnp.pi*z_local/params.Lz)[:, None, None]

    kgrid = K_Grids(kx=kx_grid, ky=ky_grid, ksq_pc=ksq_pc, inv_ksq_pc=inv_ksq_pc,
                    dealias_pc=dealias_pc, hdiss_pc=hdiss_pc, yfac=yfac, fmask=fmask,
                    z_envcos=z_envcos, z_envsin=z_envsin, fidx_x=fidx_x, fidx_y=fidx_y)
    if params.comm_backend == "jax":
        kgrid = _kgrid_to_global(kgrid, params)  # global (z-sharded) arrays for shard_map
    return kgrid

# K_Grids entries carrying a z axis (axis 0); everything else is perpendicular-only.
_Z_KGRID_FIELDS = ("z_envcos", "z_envsin")

def kgrid_specs(kgrid):
    # shard_map in_specs for a K_Grids: z-envelopes z-sharded, all other entries replicated
    # (None entries stay None — JAX treats them as empty subtrees).
    return K_Grids(**{name: (None if val is None else
                             (P(comms.Z_AXIS) if name in _Z_KGRID_FIELDS else P()))
                      for name, val in zip(K_Grids._fields, kgrid)})

def _kgrid_to_global(kgrid, params):
    # Process-local kgrid arrays -> global jax.Arrays matching kgrid_specs ("jax" backend).
    return K_Grids(**{name: (None if val is None else
                             comms.to_global(val, params, z_axis=0 if name in _Z_KGRID_FIELDS else None))
                      for name, val in zip(K_Grids._fields, kgrid)})

def fft(f):
    return ft.rfft2(f,axes=(-2,-1))

def ifft(f,params):
    return ft.irfft2(f,s=(params.nx,params.ny),axes=(-2,-1))

def local_z_coords(params):
    # z coords stored on local rank
    nz_device = params.nz // params.size
    idx_device = params.rank * nz_device + jnp.arange(nz_device)
    return idx_device * params.dz
