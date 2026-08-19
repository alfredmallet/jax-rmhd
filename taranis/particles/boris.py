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
# vmaps them; push()/push_tracked() are the jax drivers that wire them to interp.gather.
#
# Work bookkeeping (push_tracked): because the rotation is norm-exact, one half-kick with
# h = qm*dt_k/2 changes the kinetic energy per unit mass by exactly h*E.(v_in + v_new), so
# summing that per E component gives a per-piece work split whose total is
# 1/2|v_out|^2 - 1/2|v_in|^2 to round-off. Derivation: docs/numerics.md, "Test particles".
#
# E_perp projection (push_tracked(..., project=True)): each gathered E sample is replaced
# by its component perpendicular to the gathered B before the kick, so the particle sees
# E.B = 0 exactly (Xia et al. 2013, ApJ 776, 90, eq. 21).
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


def project_perp(E, B):
    # the component of E perpendicular to B: E - (E.B)/(B.B) B, so E'.B = 0
    eb = E[0] * B[0] + E[1] * B[1] + E[2] * B[2]
    bb = B[0] * B[0] + B[1] * B[1] + B[2] * B[2]
    f = eb / bb
    return jnp.stack([E[0] - f * B[0], E[1] - f * B[1], E[2] - f * B[2]])


_kick = jax.vmap(boris_kick, in_axes=(0, 0, 0, None, None))
_drift = jax.vmap(drift, in_axes=(0, 0, None))
_project = jax.vmap(project_perp, in_axes=(0, 0))


def _kick_tracked(v, s, nez, qm, dt, dw, project=False):
    # one half-kick from a single field sample s (N, 2+nez+3): E = (s0, s1, sum of the nez
    # E_z pieces), B = the last three columns. With `project` (static) the E sample is
    # replaced by its component perpendicular to B before the kick, and the work columns
    # credit the projected components, so the closure identity still holds exactly.
    # Returns the new v and dw with this half-kick's per-piece work added.
    E = jnp.stack([s[:, 0], s[:, 1], jnp.sum(s[:, 2:2 + nez], axis=1)], axis=1)
    B = s[:, -3:]
    if project:
        E = _project(E, B)
    vn = _kick(v, E, B, qm, dt)
    h = 0.5 * qm * dt
    ez = [E[:, 2]] if project else [s[:, 2 + i] for i in range(nez)]
    cols = [h * (E[:, 0] * (v[:, 0] + vn[:, 0]) + E[:, 1] * (v[:, 1] + vn[:, 1]))]
    cols += [h * c * (v[:, 2] + vn[:, 2]) for c in ez]
    return vn, dw + jnp.stack(cols, axis=1)


def push_tracked(x, v, F, nez, qm, dt, params, substeps=1, gather=interp.gather,
                 project=False):
    # advance particles (x, v) both (N,3) float64 by dt in the stacked field array F
    # (fields.assemble_stacked's layout: E_perp, nez E_z pieces, B; frozen over the call),
    # with ONE gather per half-kick. substeps (static) splits dt into equal pieces.
    # Positions are left UNFOLDED — gather folds them mod L.
    # `gather(fields, pos, params) -> (N, ncomp)` is swappable: interp.gather_spectral
    # drives the same push from rfft2 arrays.
    # Returns (x, v, dw, bsample): dw (N, 1+nez) float64 is the work per unit mass done
    # over this call, column 0 by E_perp and column 1+i by E_z piece i; bsample (N,3) is
    # the B sample of the last half-kick, i.e. at the returned position.
    # `project` (static) removes the E component along B at every sample; the projected
    # E_z is then a mixture of the E_z pieces, so it needs a single one (nez == 1).
    assert nez == 1 or not project, (
        f"push_tracked(project=True) needs exactly one E_z piece, got nez={nez}: the "
        f"projection mixes the pieces, so their work could not be attributed separately")
    h = dt / substeps
    dw = jnp.zeros((x.shape[0], 1 + nez), dtype=jnp.float64)
    for _ in range(substeps):
        v, dw = _kick_tracked(v, gather(F, x, params), nez, qm, 0.5 * h, dw, project)
        x = _drift(x, v, h)
        s = gather(F, x, params)
        v, dw = _kick_tracked(v, s, nez, qm, 0.5 * h, dw, project)
    return x, v, dw, s[:, -3:]


def push(x, v, E3, B3, qm, dt, params, substeps=1, gather=interp.gather):
    # push_tracked from separate E3, B3 arrays (each (3, nz, nx, ny)), discarding the work
    # split: E_z is a single piece here.
    return push_tracked(x, v, jnp.concatenate([E3, B3]), 1, qm, dt, params,
                        substeps=substeps, gather=gather)[:2]
