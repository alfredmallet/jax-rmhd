# equation-agnostic diagnostics machinery.
import jax.numpy as jnp

from .. import _precision


def _binned(kmag, energies, kunit, kmax, bin_factor):
    # radial binning shared by perpspec/parspec: (bin_centers, one spectrum per energy)
    dk = kunit*bin_factor
    bin_edges = jnp.arange(0,kmax+dk,dk,dtype=_precision.ftype)
    specs = tuple(jnp.histogram(kmag,bins=bin_edges,weights=e/dk)[0] for e in energies)
    return ((bin_edges[1:] + bin_edges[:-1]) / 2,) + specs
