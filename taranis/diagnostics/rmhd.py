# RMHD diagnostics: (phi, psi) energies and per-field grad-square spectra.
import jax.numpy as jnp
import jax.numpy.fft as ft
from .. import _precision
from ..physics.shared_physics import perp_gradsq, perp_reduce, perp_inner_product
from .core import _binned

def perpspec(state,kgrid,params,bin_factor=2.0):
    # perpendicular energy spectrum, z-averaged.
    spec = lambda fk: perp_reduce(0.5*perp_gradsq(fk,fk,kgrid),params,axis=0)
    kunit = min(2 * jnp.pi / params.Lx, 2 * jnp.pi / params.Ly)
    kmax = min(params.nx//2,params.ny//2)*kunit
    return _binned(jnp.sqrt(kgrid.ksq),(spec(state.fields[0]),spec(state.fields[1])),
                   kunit,kmax,bin_factor)

def parspec(state,kgrid,params,bin_factor=2.0):
    # parallel (z) energy spectrum. Requires the *whole* z-domain on this rank
    assert params.size == 1, "parspec requires the full z-domain on one rank (params.size==1)"
    half = params.nz // 2
    # the z-FFT is unnormalized, so nz^2 rides on top of the shared perpendicular norm
    norm = 1.0 / float(params.nx*params.ny*params.nz)**2
    def spec(fk):
        fkz = fk if params.z_spectral else ft.fft(fk,axis=0)
        full = jnp.sum(0.5*perp_gradsq(fkz,fkz,kgrid),axis=(1,2)) * norm
        return full[:half+1].at[1:half].add(full[half+1:][::-1])
    kz = ft.rfftfreq(params.nz, dtype=_precision.ftype) * params.nz * 2 * jnp.pi / params.Lz
    kunit = 2 * jnp.pi / params.Lz
    return _binned(kz,(spec(state.fields[0]),spec(state.fields[1])),kunit,half*kunit,bin_factor)

def energy(state,kgrid,params):
    # (E_kin, E_mag) = 0.5*(<|grad_perp phi|^2>, <|grad_perp psi|^2>), volume-averaged
    E2 = perp_inner_product(state.fields[:2],state.fields[:2],kgrid,params,batch=True)
    return (0.5*E2[0], 0.5*E2[1])
