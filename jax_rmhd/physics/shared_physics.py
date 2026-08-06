import jax
import jax.numpy as jnp
from .. import comms
from .. import _precision

# takes gradient in fourier space
# expects the kx,ky axes to be the last two (-2,-1)
def gradk(fk,kgrid):
    return jnp.stack([1j*kgrid.kx*fk,1j*kgrid.ky*fk],axis=1)

# Poisson bracket of real-space fields A and B. Returns in real space
def bracket(a,b):
    return a[0]*b[1] - a[1]*b[0]

# gets the necessary z derivatives.
def z_derivatives(f,params,halo=None):
    # z axis is assumed to be axis 1. width must agree with rmhd.halo_start's
    # pre-issued halo width -- the one coupling in this design (see comment there).
    dz=params.dz
    recv_left, recv_right = comms.halo_exchange(f,params,width=2) if halo is None else halo
    # pad width w is derived from the received slab
    # the stencil itself stays fixed at half-width k=2.
    w = recv_left.shape[1]
    assert w >= 2, f"z_derivatives: pre-issued halo width={w} is narrower than the stencil " \
                    "half-width (2); need width >= 2"
    assert recv_right.shape[1] == w, \
        f"z_derivatives: recv_left width={w} != recv_right width={recv_right.shape[1]}"
    nz = f.shape[1]
    f_padded = jnp.concatenate([recv_left,f,recv_right],axis=1)
    p2 = f_padded[:,w+2:w+2+nz,:,:]
    p1 = f_padded[:,w+1:w+1+nz,:,:]
    c = f_padded[:,w:w+nz,:,:]
    m1 = f_padded[:,w-1:w-1+nz,:,:]
    m2 = f_padded[:,w-2:w-2+nz,:,:]
    df_dz = (- p2 + 8*p1 - 8*m1 + m2) / (12 * dz)
    d4f_dz4 = (p2 -4*p1 +6*c -4*m1 + m2) / (dz**4)
    return df_dz, d4f_dz4

############
# FORCING. #
############

def _symmetrize_real_line(col):
    # enforces hermitian symmetry of the forcing: needed at ky=0 and kymax
    nkx = col.shape[-1]
    mirror_idx = (-jnp.arange(nkx)) % nkx
    return (col + jnp.conj(col[..., mirror_idx])) / jnp.sqrt(2.0)

def _draw_symmetrized_noise(key, shape, dtype, grid_norm):
    # dtype=ftype on the draws is load-bearing, not cosmetic: with x64 unconditionally on, an
    # unpinned jax.random.normal returns float64 AND a DIFFERENT bitstream from the float32
    # draw, so pinning is what keeps the fp32 RNG stream identical to pre-x64 runs
    # (plans/PRECISION_PLAN.md rule 3). The old trailing .astype(dtype) is gone with it --
    # the arithmetic is now complex64/complex128 from the start. (jnp.sqrt(2.0) is a WEAK
    # float64 scalar and adopts the array's precision, so it needs no pin.)
    key_real, key_imag = jax.random.split(key)
    ftype = _precision.ftype
    noise = (jax.random.normal(key_real, shape, dtype=ftype)
             + 1j * jax.random.normal(key_imag, shape, dtype=ftype)) / jnp.sqrt(2.0)
    noise = (noise * grid_norm).astype(dtype)
    noise = noise.at[..., 0].set(_symmetrize_real_line(noise[..., 0]))
    noise = noise.at[..., -1].set(_symmetrize_real_line(noise[..., -1]))
    return noise

def _draw_shell_noise(key, shape, dtype, grid_norm, fidx_x, fidx_y):
    # draws RNG only at the shell modes and scatters into the k-grid: a different RNG
    # stream from the full-grid draw, so the two differ bitwise
    key_real, key_imag = jax.random.split(key)
    shell_shape = shape[:-2] + (fidx_x.shape[0],)
    ftype = _precision.ftype   # pinned draw: see _draw_symmetrized_noise
    raw = (jax.random.normal(key_real, shell_shape, dtype=ftype)
           + 1j * jax.random.normal(key_imag, shell_shape, dtype=ftype)) / jnp.sqrt(2.0)
    raw = (raw * grid_norm).astype(dtype)
    noise = jnp.zeros(shape, dtype).at[..., fidx_x, fidx_y].set(raw)
    noise = noise.at[..., 0].set(_symmetrize_real_line(noise[..., 0]))
    noise = noise.at[..., -1].set(_symmetrize_real_line(noise[..., -1]))
    return noise

def ou_update(forcing_state, forcing_key, dt, params, kgrid):
    # Ornstein-Uhlenbeck step on the forcing-shell modes.
    key, new_key = jax.random.split(forcing_key)
    grid_norm = float(params.nx * params.ny)
    # forcing_state has shape (n_ou, 2, nkx, nky)
    # n_ou is 1 for momentum forcing and 2 for elsasser
    # axis=1 has 2 components to set A and B in reconstruct_envelope below
    if kgrid.fidx_x is not None:
        noise = _draw_shell_noise(key, forcing_state.shape, forcing_state.dtype, grid_norm,
                                  kgrid.fidx_x, kgrid.fidx_y)
    else:
        noise = _draw_symmetrized_noise(key, forcing_state.shape, forcing_state.dtype, grid_norm)
    decay = jnp.exp(-dt / params.forcing_tau)
    diffusion = jnp.sqrt(1.0 - decay**2)
    new_forcing_state = forcing_state * decay + diffusion * noise
    mask = kgrid.fmask
    new_forcing_state = new_forcing_state * mask[None, None, :, :]
    return new_forcing_state, new_key

def reconstruct_envelope(forcing_state, kgrid, params):
    # Rebuilds the z-projected forcing from its (A,B) cos/sin envelope coefficients, using
    # the z-envelopes precomputed in setup_kgrids (dims==3).
    if params.spatial_dimensions == 3:
        A = forcing_state[:, 0]  # (n_ou, nkx, nky)
        B = forcing_state[:, 1]
        if params.z_spectral:
            # exact z-FFT of A*cos(2pi z/Lz) + B*sin(2pi z/Lz): the unnormalized transform
            # puts (A-iB)*nz/2 on kz index +1 and (A+iB)*nz/2 on kz index -1 (== nz-1) and
            # nothing anywhere else. `.add` rather than `.set` so the degenerate nz=2 case,
            # where those two planes coincide, still sums to the right thing.
            # This preserves reality for free: A/B are already kx-Hermitian on the ky=0 and
            # Nyquist rows, and mirroring BOTH kx and kz there maps +1 <-> -1 and conjugates.
            half_nz = 0.5*params.nz
            out = jnp.zeros((forcing_state.shape[0], params.nz) + forcing_state.shape[2:],
                            dtype=forcing_state.dtype)
            return (out.at[:, 1].add(half_nz*(A - 1j*B))
                       .at[:, params.nz - 1].add(half_nz*(A + 1j*B)))
        return A[:, None, :, :] * kgrid.z_envcos + B[:, None, :, :] * kgrid.z_envsin
    else:
        #in 2D just use the A coefficient
        return forcing_state[:, 0][:, None, :, :]  # (n_ou, 1, nkx, nky)

def perp_gradsq(field_a_k, field_b_k, kgrid):
    # per-mode Re( grad(a)^* . grad(b) ), with the rfft2 ky-doubling applied
    return kgrid.ksq * jnp.real(jnp.conj(field_a_k) * field_b_k) * kgrid.yfac

def perp_reduce(integrand, params, axis=None):
    # sum over `axis` (default: every axis), allreduce over z-ranks if applicable, and
    # apply the normalization every energy-like quantity in the code shares.
    # z_spectral: the z axis carries the UNNORMALIZED z-FFT, so Parseval turns the z-average
    # (1/nz)*sum_z into (1/nz^2)*sum_kz -- one extra factor of nz, shared by every
    # energy-like quantity (diagnostics.energy/perpspec, forcing power) so they stay
    # comparable across the two modes.
    P = comms.allreduce_sum(jnp.sum(integrand, axis=axis), params)  # no-op unless z-decomposed
    znorm = float(params.nz)**2 if params.z_spectral else float(params.nz)
    return P / (znorm * float(params.nx * params.ny)**2)

def perp_inner_product(field_a_k, field_b_k, kgrid, params, batch=False):
    # Re( sum_k grad(field_a_k)^* . grad(field_b_k) ), useful for e.g. energies and power
    # inputs. batch=True keeps the leading axis (e.g. stacked z+/z-), giving one value per
    # entry from a single stacked allreduce.
    integrand = perp_gradsq(field_a_k, field_b_k, kgrid)
    return perp_reduce(integrand, params, tuple(range(1, integrand.ndim)) if batch else None)

def perp_mean_square(field_a_k, field_b_k, kgrid, params, batch=False):
    # Re( sum_k field_a_k^* . field_b_k ), i.e. perp_inner_product without the k^2 weight
    # useful for anastrophy etc.
    integrand = jnp.real(jnp.conj(field_a_k) * field_b_k) * kgrid.yfac
    return perp_reduce(integrand, params, tuple(range(1, integrand.ndim)) if batch else None)

def safe_scale(target, P, scale_max=1.0):
    # used to rescale power inputs
    scale = jnp.where(target == 0.0, 0.0, target / P)
    return jnp.clip(scale, -scale_max, scale_max)
