import jax
import jax.numpy as jnp
import mpi4jax
from mpi4py import MPI

#Takes gradient in fourier space
#Expects the kx,ky axes to be the last two (-2,-1)
def gradk(fk,kgrid):
    return jnp.stack([1j*kgrid.kx*fk,1j*kgrid.ky*fk],axis=1)

# Poisson bracket of real-space fields A and B. Returns in real space
def bracket(a,b):
    return a[0]*b[1] - a[1]*b[0]

# Gets the necessary z derivatives.
def z_derivatives(f,params):
    #n.b. z axis is assumed to be axis 1
    dz=params.dz
    send_left = f[:,:2,:,:]
    send_right = f[:,-2:,:,:]
    recv_right = mpi4jax.sendrecv(send_left, send_left, dest=params.left_neighbor, source=params.right_neighbor,
                                     comm=params.cart_comm, sendtag=101, recvtag=101)
    recv_left = mpi4jax.sendrecv(send_right, send_right, dest=params.right_neighbor, source=params.left_neighbor,
                                    comm=params.cart_comm, sendtag=102, recvtag=102)
    f_padded = jnp.concatenate([recv_left,f,recv_right],axis=1)
    p2 = f_padded[:,4:,:,:]
    p1 = f_padded[:,3:-1,:,:]
    c = f_padded[:,2:-2,:,:]
    m1 = f_padded[:,1:-3,:,:]
    m2 = f_padded[:,:-4,:,:]
    df_dz = (- p2 + 8*p1 - 8*m1 + m2) / (12 * dz)
    d4f_dz4 = (p2 -4*p1 +6*c -4*m1 + m2) / (dz**4)
    return df_dz, d4f_dz4

############
# FORCING. #
############

def _symmetrize_real_line(col):
    # Enforces hermitian symmetry of the forcing: needed at ky=0 and kymax
    nkx = col.shape[-1]
    mirror_idx = (-jnp.arange(nkx)) % nkx
    return 0.5 * (col + jnp.conj(col[..., mirror_idx]))

def _draw_symmetrized_noise(key, shape, dtype, grid_norm):
    key_real, key_imag = jax.random.split(key)
    noise = (jax.random.normal(key_real, shape) + 1j * jax.random.normal(key_imag, shape)) / jnp.sqrt(2.0)
    noise = noise * grid_norm
    noise = noise.astype(dtype)
    noise = noise.at[..., 0].set(_symmetrize_real_line(noise[..., 0]))
    noise = noise.at[..., -1].set(_symmetrize_real_line(noise[..., -1]))
    return noise

def _draw_shell_noise(key, shape, dtype, grid_norm, fidx_x, fidx_y):
    # Same statistics as _draw_symmetrized_noise restricted to the shell, but only draws
    # RNG at the shell modes and scatters into the k-grid (different RNG stream, so
    # trajectories differ bitwise from the full-grid draw). Symmetrizing the scattered
    # ky=0/Nyquist rows is valid because the shell is |k|-symmetric under kx -> -kx.
    key_real, key_imag = jax.random.split(key)
    shell_shape = shape[:-2] + (fidx_x.shape[0],)
    raw = (jax.random.normal(key_real, shell_shape) + 1j * jax.random.normal(key_imag, shell_shape)) / jnp.sqrt(2.0)
    raw = (raw * grid_norm).astype(dtype)
    noise = jnp.zeros(shape, dtype).at[..., fidx_x, fidx_y].set(raw)
    noise = noise.at[..., 0].set(_symmetrize_real_line(noise[..., 0]))
    noise = noise.at[..., -1].set(_symmetrize_real_line(noise[..., -1]))
    return noise

def _shell_mask(kgrid, params):
    # Radially symmetric mask selecting nmin <= |k_perp|/dk < nmax.
    # Uses kgrid.fmask precomputed in setup_kgrids; falls back to computing it so
    # standalone/test construction of K_Grids without fmask still works.
    if kgrid.fmask is not None:
        return kgrid.fmask
    kunit = min(2*jnp.pi/params.Lx, 2*jnp.pi/params.Ly)
    nmin, nmax = params.fshell
    kmag_over_dk = jnp.sqrt(kgrid.ksq()) / kunit
    return (kmag_over_dk >= nmin) & (kmag_over_dk < nmax)

def ou_update(forcing_state, forcing_key, dt, params, kgrid):
    # Ornstein-Uhlenbeck step on the forcing-shell modes.
    # Noise is now drawn only at the precomputed shell indices (kgrid.fidx_*) when
    # available, instead of over the full (kx,ky) grid and masking down.
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
    mask = _shell_mask(kgrid, params)
    new_forcing_state = new_forcing_state * mask[None, None, :, :]
    return new_forcing_state, new_key

def reconstruct_envelope(forcing_state, kgrid, params, z_local=None):
    # Rebuilds the real-space-projected forcing from its (A,B) cos/sin envelope coefficients.
    # Now takes kgrid and uses its precomputed z_envcos/z_envsin when available (dims==3),
    # falling back to computing from z_local (still accepted, kept for standalone/test use).
    if params.spatial_dimensions == 3:
        if kgrid.z_envcos is not None:
            cos_env, sin_env = kgrid.z_envcos, kgrid.z_envsin
        else:
            cos_env = jnp.cos(2*jnp.pi*z_local/params.Lz)[:, None, None]
            sin_env = jnp.sin(2*jnp.pi*z_local/params.Lz)[:, None, None]
        A = forcing_state[:, 0]  # (n_ou, nkx, nky)
        B = forcing_state[:, 1]
        return A[:, None, :, :] * cos_env + B[:, None, :, :] * sin_env
    else:
        #in 2D just use the A coefficient
        return forcing_state[:, 0][:, None, :, :]  # (n_ou, 1, nkx, nky)

def _perp_yfac(kgrid):
    # rfft2 y-doubling factor; uses kgrid.yfac precomputed in setup_kgrids, falling back
    # to computing it so standalone/test construction of K_Grids without yfac still works.
    if kgrid.yfac is not None:
        return kgrid.yfac
    nky = kgrid.ky.shape[-1]
    return jnp.full((nky,), 2.0).at[0].set(1.0).at[-1].set(1.0)

def _perp_reduce(integrand, params):
    # sum over all axes (z-local, kx, ky, ...), allreduce over z-ranks if applicable
    # normalize: divide by nz*(nx*ny)^2.
    P = jnp.sum(integrand)
    if params.cart_comm is not None:
        P = mpi4jax.allreduce(P, op=MPI.SUM, comm=params.cart_comm)
    norm = float(params.nz) * float(params.nx * params.ny)**2
    return P / norm

def perp_inner_product(field_a_k, field_b_k, kgrid, params):
    # Re( sum_k grad(field_a_k)^* . grad(field_b_k) )
    # useful for e.g. energies and power inputs
    # computed via the identity sum_k |grad|-inner = ksq * Re(conj(a) b): identical result
    # (to roundoff), ~2x cheaper, and avoids materializing four gradient arrays
    integrand = kgrid.ksq() * jnp.real(jnp.conj(field_a_k) * field_b_k) * _perp_yfac(kgrid)
    return _perp_reduce(integrand, params)

def perp_mean_square(field_a_k, field_b_k, kgrid, params):
    # Re( sum_k field_a_k^* . field_b_k )
    # useful for anastrophy etc.
    integrand = jnp.real(jnp.conj(field_a_k) * field_b_k) * _perp_yfac(kgrid)
    return _perp_reduce(integrand, params)

def _perp_reduce_batch(integrand, params):
    # like _perp_reduce but keeps the leading batch axis: one stacked allreduce
    # for all batch entries instead of one allreduce each. Same normalization.
    P = jnp.sum(integrand, axis=tuple(range(1, integrand.ndim)))
    if params.cart_comm is not None:
        P = mpi4jax.allreduce(P, op=MPI.SUM, comm=params.cart_comm)
    norm = float(params.nz) * float(params.nx * params.ny)**2
    return P / norm

def perp_inner_product_batch(fields_a_k, fields_b_k, kgrid, params):
    # perp_inner_product over a leading batch axis (e.g. stacked z+/z-), returning a
    # (nbatch,) vector via a single stacked allreduce. Same ksq identity as above.
    integrand = kgrid.ksq() * jnp.real(jnp.conj(fields_a_k) * fields_b_k) * _perp_yfac(kgrid)
    return _perp_reduce_batch(integrand, params)

def safe_scale(target, P, scale_max=1.0):
    # used to rescale power inputs
    scale = jnp.where(target == 0.0, 0.0, target / P)
    return jnp.clip(scale, -scale_max, scale_max)
