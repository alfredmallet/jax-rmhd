# Boris pusher, kick-drift-kick form.
#
#   v_half = boris_kick(v_n,    E(x_n),     B(x_n),     qm, dt/2)
#   x_n+1  = drift(x_n, v_half, dt)
#   v_n+1  = boris_kick(v_half, E(x_n+1),   B(x_n+1),   qm, dt/2)
#
# x and v are synchronized at step boundaries, so dt may change from step to step. The
# second half-kick needs the fields at the new position, so a step costs one field gather
# per half-kick. The rotation is norm-exact: |v| is conserved to round-off when E = 0.
#
# boris_kick and drift are per-particle kernels — elementwise arithmetic on length-3
# vectors, no Parameters, no array idioms — and translate directly to WGSL. The caller
# vmaps them; push() is the jax driver that wires them to interp.gather.
import jax
import jax.numpy as jnp

from . import interp


def boris_kick(v, E, B, qm, dt):
    # half electric kick, rotation by B (t = qm*B*dt/2, s = 2t/(1+|t|^2)), half electric kick
    h = 0.5 * qm * dt
    vmx = v[0] + h * E[0]
    vmy = v[1] + h * E[1]
    vmz = v[2] + h * E[2]
    tx = h * B[0]
    ty = h * B[1]
    tz = h * B[2]
    s = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz)
    sx = s * tx
    sy = s * ty
    sz = s * tz
    # v' = v- + v- x t
    px = vmx + (vmy * tz - vmz * ty)
    py = vmy + (vmz * tx - vmx * tz)
    pz = vmz + (vmx * ty - vmy * tx)
    # v+ = v- + v' x s
    vpx = vmx + (py * sz - pz * sy)
    vpy = vmy + (pz * sx - px * sz)
    vpz = vmz + (px * sy - py * sx)
    return jnp.stack([vpx + h * E[0], vpy + h * E[1], vpz + h * E[2]])


def drift(x, v, dt):
    return jnp.stack([x[0] + v[0] * dt, x[1] + v[1] * dt, x[2] + v[2] * dt])


_kick = jax.vmap(boris_kick, in_axes=(0, 0, 0, None, None))
_drift = jax.vmap(drift, in_axes=(0, 0, None))


def push(x, v, E3, B3, qm, dt, params, substeps=1):
    # advance particles (x, v) both (N,3) float64 by dt, with the fields frozen at the
    # grid arrays E3, B3 (each (3, nz, nx, ny)); both half-kicks gather from the same
    # arrays. substeps (static) splits dt into equal pieces. Positions are left UNFOLDED —
    # interp.gather folds them mod L.
    h = dt / substeps
    for _ in range(substeps):
        v = _kick(v, interp.gather(E3, x, params), interp.gather(B3, x, params), qm, 0.5 * h)
        x = _drift(x, v, h)
        v = _kick(v, interp.gather(E3, x, params), interp.gather(B3, x, params), qm, 0.5 * h)
    return x, v
