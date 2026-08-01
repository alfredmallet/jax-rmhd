import jax.numpy as jnp
import jax.numpy.fft as ft
from jax.sharding import PartitionSpec as P
from typing import NamedTuple, Optional
from . import comms

# nb the code is only spectral in the perpendicular plane, so fourier stuff is all 2D

class K_Grids(NamedTuple):
    # pytree container for the wavenumber grids and every derived concrete array.
    # setup_kgrids() is the ONLY sanctioned constructor (it precomputes everything);
    # computation cannot live here because jax rebuilds this NamedTuple constantly with
    # tracers / PartitionSpecs (kgrid_specs) / global arrays (_kgrid_to_global) as values.
    kx: jnp.ndarray
    ky: jnp.ndarray
    ksq: jnp.ndarray          # kx^2 + ky^2
    inv_ksq: jnp.ndarray      # 1/ksq, 0 at the zero mode
    dealias: jnp.ndarray      # 2/3-rule elliptical mask
    hdiss: jnp.ndarray        # per-field hyperdissipation exponents (-diss * ksq**hyper)
    yfac: jnp.ndarray         # rfft2 y-doubling factor (1 at ky=0 and Nyquist, else 2)
    # forcing-only entries: None when params.forcing is off
    fmask: Optional[jnp.ndarray] = None
    z_envcos: Optional[jnp.ndarray] = None
    z_envsin: Optional[jnp.ndarray] = None
    # (kx,ky) index arrays of the True entries of fmask (fixed shape), so forcing noise
    # is only drawn at shell modes rather than over the whole k-grid (ou_update).
    fidx_x: Optional[jnp.ndarray] = None
    fidx_y: Optional[jnp.ndarray] = None

def dealias_mask(params):
    # 2/3-rule elliptical dealiasing mask, in mode-index space (Lx/Ly cancel out)
    ix = ft.fftfreq(params.nx) * params.nx
    iy = ft.rfftfreq(params.ny) * params.ny
    return ((ix.reshape(-1,1)/(params.nx/3.0))**2 +
            (iy.reshape(1,-1)/(params.ny/3.0))**2) < 1.0

def setup_kgrids(params):
    # gets the wavenumber grid object from parameters, precomputing all the static
    # concrete arrays (ksq, inv_ksq, dealias, hdiss, y-doubling factor, forcing shell
    # mask/z-envelopes)
    # the scaling here respects jax.numpy's fourier transform conventions
    # so that e.g. we can calculate derivatives correctly.
    kx = ft.fftfreq(params.nx) * params.nx * 2 * jnp.pi / params.Lx
    ky = ft.rfftfreq(params.ny) * params.ny * 2 * jnp.pi / params.Ly
    kx_grid = kx.reshape(-1, 1)
    ky_grid = ky.reshape(1, -1)

    ksq = kx_grid**2 + ky_grid**2
    inv_ksq = jnp.where(ksq > 0, 1.0/ksq, 0.0)
    dealias = dealias_mask(params)
    diss = jnp.array(params.diss).reshape(-1,1,1,1)
    hdiss = -diss*ksq**params.hyper

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
        kmag_over_dk = jnp.sqrt(ksq) / kunit
        fmask = (kmag_over_dk >= nmin) & (kmag_over_dk < nmax)
        if not bool(jnp.any(fmask)):
            raise ValueError(f"forcing shell fshell={params.fshell} contains no modes on this grid "
                             f"(nx={params.nx}, ny={params.ny}, Lx={params.Lx}, Ly={params.Ly}); "
                             f"the run would proceed silently unforced")
        # static shell index set (concrete here, outside jit) for shell-restricted noise;
        # only carried when opted in (params.forcing_shell_noise)
        if params.forcing_shell_noise:
            fidx_x, fidx_y = jnp.nonzero(fmask)
        if params.spatial_dimensions == 3:
            z_local = local_z_coords(params)
            z_envcos = jnp.cos(2*jnp.pi*z_local/params.Lz)[:, None, None]
            z_envsin = jnp.sin(2*jnp.pi*z_local/params.Lz)[:, None, None]

    kgrid = K_Grids(kx=kx_grid, ky=ky_grid, ksq=ksq, inv_ksq=inv_ksq,
                    dealias=dealias, hdiss=hdiss, yfac=yfac, fmask=fmask,
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
