import jax
import jax.numpy as jnp
import jax.numpy.fft as ft
from jax.sharding import PartitionSpec as P
from typing import NamedTuple, Optional
from . import comms
from . import propagators
from . import _precision

class K_Grids(NamedTuple):
    # pytree container for the wavenumber grids and every derived concrete array.
    # setup_kgrids() is the only sanctioned constructor (it precomputes everything);
    # computation cannot live here because jax rebuilds this NamedTuple constantly with
    # tracers / PartitionSpecs (kgrid_specs) / global arrays (_kgrid_to_global) as values.
    kx: jnp.ndarray
    ky: jnp.ndarray
    ksq: jnp.ndarray          # kx^2 + ky^2
    inv_ksq: jnp.ndarray      # 1/ksq, 0 at the zero mode
    dealias: jnp.ndarray      # 2/3-rule elliptical mask ((nz,nkx,nky) when z_spectral)
    yfac: jnp.ndarray         # rfft2 y-doubling factor (1 at ky=0 and Nyquist, else 2)
    # parallel wavenumbers, shape (nz,1,1) so it broadcasts onto a (nz,nkx,nky) field.
    # built iff z_spectral=True
    kz: Optional[jnp.ndarray] = None
    # the equation set's k-local linear operator, as one of the taranis.propagators operator
    # pytrees (IdentityOperator / DiagonalOperator / Putzer2Operator / SeparableL). Always
    # populated by setup_kgrids; steppers use its methods, never its arrays.
    lin: tuple = propagators.IdentityOperator()
    # forcing-only entries
    fmask: Optional[jnp.ndarray] = None
    z_envcos: Optional[jnp.ndarray] = None
    z_envsin: Optional[jnp.ndarray] = None
    # (kx,ky) index arrays of the True entries of fmask (fixed shape), so forcing noise
    # is only drawn at shell modes rather than over the whole k-grid (ou_update).
    fidx_x: Optional[jnp.ndarray] = None
    fidx_y: Optional[jnp.ndarray] = None

def dealias_mask(params):
    # 2/3-rule elliptical dealiasing mask, in mode-index space (Lx/Ly[/Lz] cancel out).
    ix = ft.fftfreq(params.nx, dtype=_precision.ftype) * params.nx
    iy = ft.rfftfreq(params.ny, dtype=_precision.ftype) * params.ny
    perp = ((ix.reshape(-1,1)/(params.nx/3.0))**2 +
            (iy.reshape(1,-1)/(params.ny/3.0))**2) < 1.0
    if not params.z_spectral:
        return perp
    iz = ft.fftfreq(params.nz, dtype=_precision.ftype) * params.nz
    return (jnp.abs(iz).reshape(-1,1,1) < params.nz/3.0) & perp[None,:,:]

def setup_kgrids(params):
    # gets the wavenumber grid object from parameters, precomputing all the static
    # concrete arrays (ksq, inv_ksq, dealias, the equation set's linear operator,
    # y-doubling factor, forcing shell mask/z-envelopes)
    # the scaling here respects jax.numpy's fourier transform conventions
    # so that e.g. we calculate derivatives correctly.
    kx = ft.fftfreq(params.nx, dtype=_precision.ftype) * params.nx * 2 * jnp.pi / params.Lx
    ky = ft.rfftfreq(params.ny, dtype=_precision.ftype) * params.ny * 2 * jnp.pi / params.Ly
    kx_grid = kx.reshape(-1, 1)
    ky_grid = ky.reshape(1, -1)

    ksq = kx_grid**2 + ky_grid**2
    inv_ksq = jnp.where(ksq > 0, 1.0/ksq, 0.0)
    dealias = dealias_mask(params)

    nky = ky_grid.shape[-1]
    yfac = jnp.full((nky,), 2.0, dtype=_precision.ftype).at[0].set(1.0).at[-1].set(1.0)

    kz = None
    if params.z_spectral:
        kz = (ft.fftfreq(params.nz, dtype=_precision.ftype)
              * params.nz * 2 * jnp.pi / params.Lz).reshape(-1,1,1)

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
        if params.spatial_dimensions == 3 and not params.z_spectral:
            z_local = local_z_coords(params)
            z_envcos = jnp.cos(2*jnp.pi*z_local/params.Lz)[:, None, None]
            z_envsin = jnp.sin(2*jnp.pi*z_local/params.Lz)[:, None, None]

    kgrid = K_Grids(kx=kx_grid, ky=ky_grid, ksq=ksq, inv_ksq=inv_ksq,
                    dealias=dealias, yfac=yfac, kz=kz, fmask=fmask,
                    z_envcos=z_envcos, z_envsin=z_envsin, fidx_x=fidx_x, fidx_y=fidx_y)
    kgrid = _attach_linear_operator(kgrid, params)
    if params.comm_backend == "jax":
        kgrid = _kgrid_to_global(kgrid, params)  # global (z-sharded) arrays for shard_map
    return kgrid

def _attach_linear_operator(kgrid, params):
    # builds the linear operator L (dt f = L f + N(f)) and the propagator it selects
    from .physics import equation_registry   # local import: physics imports grids
    linear_matrix_func = equation_registry[params.eqtype].linear_matrix_func
    if linear_matrix_func is None:
        return kgrid._replace(lin=propagators.IdentityOperator())
    L = linear_matrix_func(kgrid, params)
    if isinstance(L, propagators.SeparableL):
        # all three entries are real
        L = propagators.SeparableL(*(a.astype(_precision.ftype) for a in L))
    else:
        L = L.astype(_precision.ctype if jnp.iscomplexobj(L) else _precision.ftype)
    return kgrid._replace(lin=propagators.build(L, params))

# K_Grids entries carrying a z axis (axis 0); everything else is perpendicular-only.
_Z_KGRID_FIELDS = ("z_envcos", "z_envsin")

def kgrid_specs(kgrid):
    # shard_map in_specs for a K_Grids: z-envelopes z-sharded, every other entry -- the
    # linear operator's leaves included -- replicated
    def spec(name, val):
        if name == "lin":
            return jax.tree.map(lambda _: P(), val)
        if val is None:
            return None
        return P(comms.Z_AXIS) if name in _Z_KGRID_FIELDS else P()
    return K_Grids(**{name: spec(name, val) for name, val in zip(K_Grids._fields, kgrid)})

def _kgrid_to_global(kgrid, params):
    # Process-local kgrid arrays -> global jax.Arrays matching kgrid_specs ("jax" backend).
    def to_global(name, val):
        if name == "lin":
            return jax.tree.map(lambda v: comms.to_global(v, params), val)
        if val is None:
            return None
        return comms.to_global(val, params,
                               z_axis=0 if name in _Z_KGRID_FIELDS else None)
    return K_Grids(**{name: to_global(name, val)
                      for name, val in zip(K_Grids._fields, kgrid)})

# forward/inverse transforms, unnormalized.
def fft(f,params):
    if params.z_spectral:
        return ft.rfftn(f,axes=(-3,-2,-1))
    return ft.rfft2(f,axes=(-2,-1))

def ifft(f,params):
    if params.z_spectral:
        return ft.irfftn(f,s=(params.nz,params.nx,params.ny),axes=(-3,-2,-1))
    return ft.irfft2(f,s=(params.nx,params.ny),axes=(-2,-1))

def local_z_coords(params):
    # z coords stored on local rank.
    nz_device = params.nz // params.size
    idx_device = params.rank * nz_device + jnp.arange(nz_device, dtype=_precision.ftype)
    return idx_device * params.dz
