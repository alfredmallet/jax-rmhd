# Real-space (E, B) a charged test particle sees in RMHD, decomposed into the pieces a
# per-ensemble mask can switch on and off.
#   u = zhat x grad(phi), b_perp = zhat x grad(psi), Phi = B0*phi, A_z = -psi, so
#   B = B0*zhat + b_perp,  E_perp = -B0*grad(phi),  E_z = +dpsi/dt
#     = -{phi,psi} + L_psi psi + f_psi   (the d_z(phi) terms cancel, which needs B0 == 1
#   in 3D: the solver's Alfven coefficient is 1 -- see docs/numerics.md).
# The resistive piece is the FULL linear non-ideal EMF on psi: the psi diagonal of
# rmhd.linear_matrix plus, for finite-difference z, FDLinearTerm's d4/dz4 filter.
# B0 enters in ONE place, assemble_stacked(); particle_fields stores E_perp per unit B0.
# Derivation and the dealiasing subtlety below: docs/numerics.md, "Test particles".
from typing import NamedTuple, Optional

import jax.numpy as jnp

from .. import grids
from .. import propagators
from ..physics import rmhd
from ..physics.shared_physics import bracket, gradk, z_derivatives

FIELD_PIECES = ("bperp", "eperp", "ez_ideal", "ez_resistive", "ez_forcing")
FIELD_MASK_DEFAULTS = dict(bperp=True, eperp=True, ez_ideal=True,
                           ez_resistive=False, ez_forcing=False)
# the only mask the E_parallel projection (boris.project_perp) is defined for: b_perp and
# E_perp live, E_z the ideal piece alone
EPAR_PROJECT_MASK = dict(FIELD_MASK_DEFAULTS)

# the E_z pieces, in assembly order; each maps to a PFields entry of the same name
_EZ_PIECES = ("ez_ideal", "ez_resistive", "ez_forcing")
# the ELECTRIC pieces, in the order the per-particle work accumulator stores them
# (ParticleState.w's last axis)
WORK_PIECES = ("eperp",) + _EZ_PIECES
NWORK = len(WORK_PIECES)
# particle_fields kwarg that produces each optional piece (used in the error message)
_EZ_KWARG = dict(ez_resistive="resistive", ez_forcing="forcing")


class PFields(NamedTuple):
    # real-space collocation arrays, each (nz_local, nx, ny) at field precision
    ux: jnp.ndarray            # u = zhat x grad(phi) = (-d_y phi, d_x phi); diagnostics only
    uy: jnp.ndarray
    bx: jnp.ndarray            # b_perp = zhat x grad(psi) = (-d_y psi, d_x psi)
    by: jnp.ndarray
    ex: jnp.ndarray            # E_perp/B0 = -grad(phi); assembly applies B0
    ey: jnp.ndarray
    ez_ideal: jnp.ndarray      # ifft(dealias * fft(-{phi,psi}))
    # ifft(L_psi*psik, + the FD-z filter in 3D); None unless requested
    ez_resistive: Optional[jnp.ndarray] = None
    ez_forcing: Optional[jnp.ndarray] = None     # ifft(f_psi); None unless requested


def particle_fields(state, kgrid, params, *, resistive=False, forcing=False):
    """The real-space field pieces at the collocation grid, with E_perp stored per unit
    B0 (the assembly applies B0). 4 iffts for the gradients, +2 for the dealiased ideal
    E_z, +1 per optional piece (the resistive piece's finite-difference-z filter is a
    z-stencil, not a transform)."""
    assert params.eqtype == "RMHD", (
        f"particle_fields is RMHD-only (it assumes fields=(phi,psi) with u=zhat x grad(phi), "
        f"b_perp=zhat x grad(psi)); got eqtype={params.eqtype!r}")
    # (2 fields, 2 components, nz, nx, ny): gradk stacks (d_x, d_y) on axis 1
    g = grids.ifft(gradk(state.fields[:2], kgrid), params)
    gphi, gpsi = g[0], g[1]
    # the discrete psi integrates dealias*NL_psi, so the ideal E_z the particle sees is the
    # dealiased bracket -- the raw product carries content out to 2*k_c that never enters psi
    ez_ideal = grids.ifft(kgrid.dealias * grids.fft(-bracket(gphi, gpsi), params), params)
    ez_resistive = (grids.ifft(_psi_non_ideal(state.fields, kgrid, params), params)
                    if resistive else None)
    ez_forcing = _forcing_ez(state, kgrid, params, ez_ideal) if forcing else None
    return PFields(ux=-gphi[1], uy=gphi[0],
                   bx=-gpsi[1], by=gpsi[0],
                   ex=-gphi[0], ey=-gphi[1],
                   ez_ideal=ez_ideal, ez_resistive=ez_resistive, ez_forcing=ez_forcing)


def _psi_non_ideal(fields, kgrid, params):
    # every linear non-ideal term the solver adds to dpsi/dt, in k space: the k-local
    # diagonal below, plus the finite-difference-z filter, which lives outside L.
    out = _psi_linear_diagonal(kgrid, params) * fields[1]
    if params.spatial_dimensions == 3 and not params.z_spectral:
        # FDLinearTerm's -z_diss*(dz/2)^4 d4psi/dz4; its Alfven stencil is the d_z(phi)
        # term already cancelled out of E_z, so only the filter belongs here
        _df_dz, d4f_dz4 = z_derivatives(fields, params)
        out = out - params.z_diss * (params.dz / 2) ** 4 * d4f_dz4[1]
    return out


def _psi_linear_diagonal(kgrid, params):
    # the psi DIAGONAL of the equation set's k-local L. Under z_spectral L is a 2x2
    # operator whose off-diagonal is the d_z(phi) Alfven term -- already cancelled out of
    # E_z -- so propagators.apply_L would be wrong here.
    L = rmhd.linear_matrix(kgrid, params)
    if isinstance(L, propagators.SeparableL):
        return L.dperp + L.dz
    if L.ndim == 5:
        return L[1, 1]
    # 4-d diagonal L: leading axis is nfields, or 1 when it broadcasts over fields
    return L[1] if L.shape[0] > 1 else L[0]


def _forcing_ez(state, kgrid, params, like):
    # f_psi, already scaled per params.forcing_norm_per_step (ForcingTerm ignores grads).
    # Zero array when forcing is off, and identically zero in momentum mode.
    if not params.forcing:
        return jnp.zeros_like(like)
    return grids.ifft(rmhd.ForcingTerm(state, None, kgrid, params)[1], params)


def full_mask():
    """All five pieces on: E_z is exactly dpsi/dt (the gate-4/7 configuration)."""
    return dict.fromkeys(FIELD_PIECES, True)


def resolve_mask(mask):
    """FIELD_MASK_DEFAULTS updated with `mask`; unknown keys are a ValueError."""
    m = dict(FIELD_MASK_DEFAULTS)
    if mask:
        unknown = [k for k in mask if k not in FIELD_MASK_DEFAULTS]
        if unknown:
            raise ValueError(f"unknown field-mask key(s) {unknown}; expected a subset of "
                             f"{list(FIELD_PIECES)}")
        m.update(mask)
    return m


def assemble_stacked(pf, mask, B0=1.0):
    """(F, ez_on): one stacked (ncomp, nz_local, nx, ny) array holding, in order,
    [B0*ex, B0*ey, the E_z pieces the mask selects (in _EZ_PIECES order), bx, by, B0],
    and the tuple of included E_z piece names (so ncomp = 2 + len(ez_on) + 3). Pieces the
    mask switches off are zero arrays, never dropped. One array means the pusher gathers
    E and B in a single call, and keeping the E_z pieces separate lets it attribute the
    work each one does. The ONLY place B0 enters. Static python throughout: the mask is
    compile-time config, never a traced condition."""
    m = resolve_mask(mask)
    zero = jnp.zeros_like(pf.ex)
    ez_on, ez = [], []
    for name in _EZ_PIECES:
        if not m[name]:
            continue
        piece = getattr(pf, name)
        if piece is None:
            raise ValueError(f"field mask requests {name}, but PFields.{name} is None -- "
                             f"call particle_fields(..., {_EZ_KWARG[name]}=True)")
        ez_on.append(name)
        ez.append(piece)
    ex, ey = (B0 * pf.ex, B0 * pf.ey) if m["eperp"] else (zero, zero)
    bx, by = (pf.bx, pf.by) if m["bperp"] else (jnp.zeros_like(pf.bx), jnp.zeros_like(pf.by))
    return (jnp.stack([ex, ey] + ez + [bx, by, jnp.full_like(pf.bx, B0)]), tuple(ez_on))


def assemble(pf, mask, B0=1.0):
    """(E, B), each (3, nz_local, nx, ny), from the pieces the mask selects: the summed
    form of assemble_stacked, for callers that do not track work per piece."""
    F, ez_on = assemble_stacked(pf, mask, B0)
    ez = jnp.zeros_like(F[0])
    for i in range(len(ez_on)):
        ez = F[2 + i] if i == 0 else ez + F[2 + i]
    return jnp.stack([F[0], F[1], ez]), F[-3:]
