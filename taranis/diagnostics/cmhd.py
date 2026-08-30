# CMHD diagnostics: read-only observers on a compressible-MHD state (energies, Mach
# numbers, div B, the rarefaction monitor, spectra, the hyperdissipation energy budget).
#
# Derivation of record: docs/numerics.md, section "Compressible MHD"; plan and scope:
# plans/CMHD_PLAN.md (Phase C2). The dependency runs diagnostics -> physics and never back:
# the eqpars parser, the enthalpy and the linear operator are imported from
# physics/cmhd.py (underscore names crossing a module boundary, the diagnostics/gdi.py
# precedent) so there is exactly one definition of each.
#
# NORMALIZATION. Every scalar here is a VOLUME AVERAGE <.> over the periodic box, which is
# the repo's one energy-like convention (CLAUDE.md): shared_physics.perp_reduce divides a
# k-space quadratic by nz^2*(nx*ny)^2, and under z_spectral's unnormalized rfftn that is
# exactly Parseval, i.e. <a*b>. So a real-space jnp.mean(a*b) and
# perp_reduce(real(conj(a^)b^)*yfac) are the SAME number to round-off, and the numbers here
# are directly comparable to diagnostics.rmhd.energy, to perpspec, and to a future CMHD
# forcing power. test_cmhd_diagnostics.py pins that identity (spectra sum to the energies).
#
# TRANSFORM COST. kinetic and internal energy are cubic/non-polynomial in the fields, so
# unlike RMHD's energy they cannot be read off k-space: rho and u must be inverse
# transformed. energies() and mach_numbers() cost 7 iffts each (the whole stacked field
# array), spectra() 4 + 3 forward (sqrt(rho)*u), rho_min() 1, divB_max() none (pure
# k-space), energy_budget() 7 + 7 (the three D_f f). None of this is on a
# solver path -- these are snapshot-cadence observers.
#
# NOT jit-friendly by design: rho_min and divB_max return host floats, and spectra's bin
# edges are built from a concrete max|k|. Call them between simulate() calls, on live
# states -- tests/test_cmhd_diagnostics.py pins that every one of them leaves the state and
# every kgrid leaf bitwise unchanged.

import jax.numpy as jnp
import numpy as np

from .. import grids
from ..physics import shared_physics
from ..physics.cmhd import _enthalpy, _eqpars, _kz_deriv, linear_matrix
from .core import _binned


# --------------------------------------------------------------------------- helpers

def _real_fields(state, params):
    # (rho, u, B) in real space: rho (nz,nx,ny), u and B (3,nz,nx,ny). One ifft over the
    # stacked array -- grids.ifft transforms the last three axes.
    f = grids.ifft(state.fields, params)
    return f[0], f[1:4], f[4:7]


def _cs2(rho, cs0, gamma):
    # c_s^2(rho) = cs0^2 rho^(gamma-1); the gamma == 1 branch is trace-time python, as in
    # physics/cmhd.py. NaN on rho < 0 at non-integer exponents is the intended loud failure.
    return cs0*cs0 if gamma == 1.0 else cs0*cs0 * rho**(gamma - 1.0)


def _rho_e(rho, cs0, gamma):
    # the internal energy DENSITY rho*e(rho) from e'(rho) = p/rho^2 (docs/numerics.md):
    #   rho e = p/(gamma-1) = cs0^2 rho^gamma / (gamma(gamma-1))   gamma > 1
    #   rho e = cs0^2 rho ln rho                                   gamma = 1
    # The gamma = 1 branch is not sign-definite; only its DRIFT is meaningful.
    if gamma == 1.0:
        return cs0*cs0 * rho*jnp.log(rho)
    return (cs0*cs0/(gamma*(gamma - 1.0))) * rho**gamma


def _Lrow(L, i):
    # one field's diagonal of L. linear_matrix returns leading axis 1 (uniform diss,
    # broadcast over the seven fields) or 7 (the (D_rho, nu, eta) schema).
    return L[0] if L.shape[0] == 1 else L[i]


# --------------------------------------------------------------------------- energies

def energies(state, kgrid, params):
    """(kinetic, magnetic, internal) volume-averaged energies:

        E_kin = <rho |u|^2 / 2>,  E_mag = <|B|^2 / 2>,  E_int = <rho e(rho)>

    summing to the ideal invariant E of docs/numerics.md (whose drift the C1 gate bounds).
    E_mag includes the k = 0 background |B0|^2/2 and E_int the uniform part of rho e; both
    are constants of the motion, so only differences carry information. At gamma = 1,
    rho e = cs0^2 rho ln rho can be negative -- that is the correct functional, and the
    +cs0^2 constant separating it from h drops out of the budget (docs/numerics.md).

    Costs 7 inverse transforms: E_kin is cubic in the fields and E_int non-polynomial, so
    neither can be evaluated in k-space the way diagnostics.rmhd.energy is. E_mag alone
    could be (it is quadratic), and is deliberately not -- computing it on the same
    real-space path keeps the spectra sum rule an independent Parseval check rather than a
    near-tautology."""
    cs0, _, _, gamma = _eqpars(params)
    rho, u, B = _real_fields(state, params)
    kinetic = 0.5*jnp.mean(rho*(u*u).sum(0))
    magnetic = 0.5*jnp.mean((B*B).sum(0))
    internal = jnp.mean(_rho_e(rho, cs0, gamma))
    return kinetic, magnetic, internal


def mach_numbers(state, kgrid, params):
    """(M_s, M_A), both rms/global ratios rather than pointwise averages:

        M_s = sqrt( <|u|^2> / <c_s^2> )        c_s^2(rho) = cs0^2 rho^(gamma-1)
        M_A = sqrt( <rho|u|^2> / <|B|^2> ) = sqrt(E_kin/E_mag)

    Pointwise |u|/c_s and |u|/v_A are deliberately NOT used: v_A = |B|/sqrt(rho) has zeros
    wherever B does (the OT vortex has them on whole lines), so their averages are dominated
    by null points rather than by the flow. M_A in the form above is the mass-weighted
    velocity over the volume-averaged field, i.e. exactly the energy ratio, which is what a
    turbulence run wants to monitor. At gamma = 1, M_s reduces to u_rms/cs0.

    Costs 7 inverse transforms."""
    cs0, _, _, gamma = _eqpars(params)
    rho, u, B = _real_fields(state, params)
    u2 = (u*u).sum(0)
    ms = jnp.sqrt(jnp.mean(u2)/jnp.mean(jnp.broadcast_to(_cs2(rho, cs0, gamma), u2.shape)))
    ma = jnp.sqrt(jnp.mean(rho*u2)/jnp.mean((B*B).sum(0)))
    return ms, ma


def rho_min(state, kgrid, params):
    """min over the grid of the real-space density, as a host float.

    The rarefaction monitor. CMHD evolves rho in flux form, which conserves mass exactly but
    does not guarantee positivity (plans/CMHD_PLAN.md §3.2): a strongly supersonic run can
    drive rho through zero, after which rho**(gamma-1) and log(rho) are NaN and c_s in
    set_timestep poisons dt. Watching this number is how a campaign sees that coming instead
    of finding a NaN. Costs 1 inverse transform."""
    rho, _, _ = _real_fields(state, params)
    return float(jnp.min(rho))


def divB_max(state, kgrid, params):
    """max_k |k.B^| / (|k| max_k |B^|), as a host float: the worst divergence-carrying
    component of B^ anywhere on the grid, measured against the FIELD scale.

    This is the metric tests/test_cmhd_conservation.py::test_div_b_stays_a_round_off_random_walk
    settled on, and the normalization is the load-bearing part. The round-off deposit
    curl-form induction leaves is O(eps*|k|^2|E^|), set by the field, not by the individual
    mode's own amplitude -- so dividing each mode by its own |B^| would inflate a mode a
    factor a below the peak by 1/a and report an amplitude ratio rather than div B. The
    k = 0 mode is excluded (|k| = 0 there and k.B^ is identically zero).

    kz is the Nyquist-zeroed physics/cmhd._kz_deriv kz, i.e. the same d/dz the induction
    term used; measuring with a bare kgrid.kz would report a nonzero divergence at the
    kz-Nyquist plane that the solver never created."""
    f = np.asarray(state.fields)
    kx, ky = np.asarray(kgrid.kx), np.asarray(kgrid.ky)
    kz = np.asarray(_kz_deriv(kgrid, params))
    d = np.abs(kx*f[4] + ky*f[5] + kz*f[6])
    # kx (nkx,1), ky (1,nky) and kz (nz,1,1) broadcast to d's (nz,nkx,nky) on their own; do
    # NOT reach for a `+ 0.0*d`-style broadcast, which turns a NaN field into an empty mask
    # and an unrelated "zero-size reduction" error instead of a NaN.
    kmag = np.sqrt(kx**2 + ky**2 + kz**2)
    bmag = np.sqrt(np.abs(f[4])**2 + np.abs(f[5])**2 + np.abs(f[6])**2)
    m = kmag > 0
    return float(np.max(d[m]/kmag[m])/bmag.max())


# ---------------------------------------------------------------------------- spectra

def spectra(state, kgrid, params, bin_factor=2.0, isotropic=False):
    """(kbins, E_kin(k), E_mag(k), E_rho(k)) -- radially binned spectra that SUM to

        sum E_kin(k)*dk = <rho|u|^2>/2 = energies()[0]
        sum E_mag(k)*dk = <|B|^2>/2    = energies()[1]
        sum E_rho(k)*dk = <(rho-<rho>)^2>/2

    to round-off (Parseval; the gate is in tests/test_cmhd_diagnostics.py). Note what the
    three weights are:

    - kinetic uses w = sqrt(rho) u, the Kritsuk et al. (2007) compressible variable, because
      |u^|^2 alone integrates to <|u|^2>/2 and NOT to the kinetic energy. w is a real-space
      product of a non-polynomial factor with u, so w^ has power at every k -- which is why
      the bins here run out to the GRID CORNER sqrt(kx_max^2 + ky_max^2 [+ kz_max^2]) rather
      than to diagnostics.rmhd.perpspec's min(nx,ny)//2 * kunit. perpspec can stop early
      because its fields are dealiased and have no corner power; a compressible kinetic
      spectrum does, and truncating it would break the sum rule silently.
    - magnetic is |B^|^2/2 including the k = 0 background.
    - density is the FLUCTUATION (rho - <rho>)^2/2, i.e. the k = 0 mode dropped; it is not
      an energy (E_int is not quadratic and has no spectrum) but it is the compressibility
      diagnostic that pairs with them.

    isotropic=False (default) bins in k_perp and sums over kz, the diagnostics.rmhd.perpspec
    convention. isotropic=True bins in |k| = sqrt(k_perp^2 + kz^2) over the full 3-d grid,
    which is the natural one for CMHD -- the equations have no preferred axis, only the mean
    field does. Both obey the same sum rules.

    Costs 4 inverse transforms (rho, u) and 3 forward (w); B and rho are read straight out
    of state.fields."""
    rho = grids.ifft(state.fields[0], params)
    u = grids.ifft(state.fields[1:4], params)
    rhok, Bk = state.fields[0], state.fields[4:7]
    wk = grids.fft(jnp.sqrt(rho)*u, params)
    yfac = kgrid.yfac

    def q(fk):
        return 0.5*jnp.real(jnp.conj(fk)*fk).sum(0)*yfac

    drhok = rhok.at[0, 0, 0].set(0.0)
    integ = (q(wk), q(Bk), q(drhok[None]))

    kunit_perp = min(2*jnp.pi/params.Lx, 2*jnp.pi/params.Ly)
    if isotropic:
        kunit = min(float(kunit_perp), 2*np.pi/params.Lz)
        # the RAW kgrid.kz, not physics.cmhd._kz_deriv's Nyquist-zeroed one: this is a bin
        # COORDINATE, not a d/dz. The module rule about _kz_deriv is about derivatives, and
        # a mode at the kz-Nyquist plane genuinely sits at |kz| = (nz/2)*2pi/Lz -- binning
        # it at k_perp instead would misplace exactly the non-polynomial residual of
        # sqrt(rho)*u that the corner bins exist to hold.
        kmag = jnp.sqrt(kgrid.ksq + kgrid.kz*kgrid.kz)
        # axis=() sums over NO axis: perp_reduce then applies the shared normalization
        # elementwise, leaving a per-mode (nz,nkx,nky) array to histogram.
        specs = tuple(shared_physics.perp_reduce(e, params, axis=()) for e in integ)
    else:
        kunit = float(kunit_perp)
        kmag = jnp.sqrt(kgrid.ksq)
        specs = tuple(shared_physics.perp_reduce(e, params, axis=0) for e in integ)
    return _binned(kmag, specs, kunit, float(jnp.max(kmag)), bin_factor)


# ----------------------------------------------------------------------- energy budget

def energy_budget(state, kgrid, params):
    """The hyperdissipation sink of docs/numerics.md, as a dict of volume-averaged rates:

        eps = -< rho u . (D_u u) + B . (D_B B) + (|u|^2/2 + h) . (D_rho rho) >

    with D_f = the field's diagonal of L = -diss_f k^(2 hyper), applied in k-space and
    transformed back. The three terms are dE/dt's pieces through delta E/delta u = rho u,
    delta E/delta B = B and delta E/delta rho = |u|^2/2 + h(rho); with the ideal part
    conserving E exactly in the continuum, the closure is

        dE/dt + eps ~ 0

    to the O(dt^p)-plus-non-polynomial-aliasing class the C1 energy gate lives in -- NEVER
    to round-off (plans/CMHD_PLAN.md §3.5). The test gates it with centered differences of
    E(t) against eps(t), as tests/test_gdi_linear.py does for GDI.

    Two traps this implementation is written around, both from docs/numerics.md:
    - the D_rho term is real work. Mass diffusion does pdV-like work through h; omitting it
      is the likely first bug in a budget implementation, and the test has a discriminator
      that fails if it is dropped.
    - at gamma = 1 the strict derivative is d(rho e)/drho = cs0^2(ln rho + 1) = h + cs0^2.
      The extra constant multiplies <D_rho rho>, whose only surviving mode would be k = 0,
      where L(0) = 0 exactly -- so it provably drops out. h is used, and the discrepancy is
      not chased.

    Returns dict(kinetic=, magnetic=, density=, total=, dEdt=). The three pieces already
    carry the minus sign of the eps definition, so total = eps >= 0 for diss >= 0, and
    dEdt = -total is the same number in diagnostics.gdi.energy_budget's sign convention
    (there "total" IS dE/dt). Both keys are returned because the two conventions are one
    sign apart and a diagnostic that can be read backwards is a bug waiting to happen.
    Costs 14 transforms."""
    cs0, _, _, gamma = _eqpars(params)
    rho, u, B = _real_fields(state, params)
    L = linear_matrix(kgrid, params)
    fk = state.fields

    def dissipate(i):
        return grids.ifft(_Lrow(L, i)*fk[i], params)

    d_rho = dissipate(0)
    d_u = jnp.stack([dissipate(1), dissipate(2), dissipate(3)])
    d_B = jnp.stack([dissipate(4), dissipate(5), dissipate(6)])

    h = _enthalpy(rho, cs0, gamma)
    kinetic = -jnp.mean(rho*(u*d_u).sum(0))
    magnetic = -jnp.mean((B*d_B).sum(0))
    density = -jnp.mean((0.5*(u*u).sum(0) + h)*d_rho)
    total = kinetic + magnetic + density
    return dict(kinetic=kinetic, magnetic=magnetic, density=density, total=total,
                dEdt=-total)
