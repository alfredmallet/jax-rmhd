import jax.numpy as jnp
import jax.numpy.fft as ft
from . import comms

def perpspec(state,kgrid,params,bin_factor=2.0):
    # perpendicular energy spectrum, z-averaged.
    phik=state.fields[0]
    psik=state.fields[1]
    rfft2_y_factor = jnp.full(phik.shape[-1],2.0)
    rfft2_y_factor = rfft2_y_factor.at[0].set(1.0)
    rfft2_y_factor = rfft2_y_factor.at[-1].set(1.0)
    energy_u = 0.5 * kgrid.ksq * jnp.abs(phik)**2.0 * rfft2_y_factor
    energy_b = 0.5 * kgrid.ksq * jnp.abs(psik)**2.0 * rfft2_y_factor
    # sum over the local z-slab first, then allreduce across z-ranks
    energy_u = jnp.sum(energy_u,axis=0)
    energy_b = jnp.sum(energy_b,axis=0)
    energy_u = comms.allreduce_sum(energy_u, params)  # no-ops unless z-decomposed
    energy_b = comms.allreduce_sum(energy_b, params)
    energy_u = energy_u/params.nz
    energy_b = energy_b/params.nz
    kunit = min(2 * jnp.pi / params.Lx, 2 * jnp.pi / params.Ly)
    kmax = min(params.nx//2,params.ny//2)*kunit
    dk=kunit*bin_factor
    bin_edges = jnp.arange(0,kmax+dk,dk)
    norm = 1 / float(params.nx*params.ny)**2
    spec_u, _ = jnp.histogram(jnp.sqrt(kgrid.ksq),bins=bin_edges,weights=energy_u*norm/dk)
    spec_b, _ = jnp.histogram(jnp.sqrt(kgrid.ksq),bins=bin_edges,weights=energy_b*norm/dk)

    bin_centers=(bin_edges[1:] + bin_edges[:-1]) / 2
    return bin_centers,spec_u,spec_b

def parspec(state,kgrid,params,bin_factor=2.0):
    # parallel (z) energy spectrum. Requires the *whole* z-domain on this rank
    assert params.size == 1, "parspec requires the full z-domain on one rank (params.size==1)"
    phik = state.fields[0]
    psik = state.fields[1]
    rfft2_y_factor = jnp.full(phik.shape[-1],2.0)
    rfft2_y_factor = rfft2_y_factor.at[0].set(1.0)
    rfft2_y_factor = rfft2_y_factor.at[-1].set(1.0)
    phikkz = ft.fft(phik,axis=0)
    psikkz = ft.fft(psik,axis=0)
    kz = ft.rfftfreq(params.nz) * params.nz * 2 * jnp.pi / params.Lz
    en_u_full = jnp.sum(0.5 * kgrid.ksq * jnp.abs(phikkz)**2.0 * rfft2_y_factor, axis=(1,2))
    en_b_full = jnp.sum(0.5 * kgrid.ksq * jnp.abs(psikkz)**2.0 * rfft2_y_factor, axis=(1,2))
    half = params.nz // 2
    energy_u = en_u_full[:half+1].at[1:half].add(en_u_full[half+1:][::-1])
    energy_b = en_b_full[:half+1].at[1:half].add(en_b_full[half+1:][::-1])
    kunit = 2 * jnp.pi / params.Lz
    kmax = params.nz//2 * kunit
    dk=kunit*bin_factor
    bin_edges = jnp.arange(0,kmax+dk,dk)
    norm= 1.0 /dk/float(params.nx*params.ny*params.nz)**2
    spec_u, _ = jnp.histogram(kz,bins=bin_edges,weights=energy_u*norm)
    spec_b, _ = jnp.histogram(kz,bins=bin_edges,weights=energy_b*norm)
    bin_centers=(bin_edges[1:] + bin_edges[:-1]) / 2
    return bin_centers,spec_u,spec_b

def energy(state,kgrid,params):
    # (E_kin, E_mag) = 0.5*(<|grad_perp phi|^2>, <|grad_perp psi|^2>), volume-averaged,
    # via perp_inner_product_batch: one stacked allreduce
    from .physics.shared_physics import perp_inner_product_batch
    fk = state.fields[:2]  # (phi, psi)
    E2 = perp_inner_product_batch(fk, fk, kgrid, params)  # (<|grad phi|^2>, <|grad psi|^2>)
    return (0.5*E2[0], 0.5*E2[1])

