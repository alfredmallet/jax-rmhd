# Exact per-mode propagators for the k-local linear part of an equation set.
# Sign convention: a recipe's linear_matrix_func returns L with
#       dt f = L f + N(f)          ->   propagator = exp(L*tau)
# Timesteppers never see L: they only call
#   apply_exp(arr, tau)     multiply by exp(L*tau)        (integrating-factor schemes)
#   solve_shifted(arr, a)   apply (I - a*L)^-1            (IMEX schemes)
#   apply_L(arr)            multiply by L                 (IMEX schemes)
# plus scaled(factor), which returns the propagator of factor*L (see LSRK note below), and
#   exp_op(tau)             the precomputed exp(L*tau) as an ExpOp pytree with .apply(arr)
# for the hoisted path: when dt is frozen over a block (fixed dt, or a cfl_every block)
# run.py builds every stage's ExpOp ONCE per block (timestepping.stage_exp_ops) instead of
# re-evaluating the matrix exponential inside every stage of every step. The ExpOps hold
# exactly the arrays apply_exp would form, in the same op order, so the hoisted and unhoisted
# paths compute the same numbers. The putzer2 and separable backends are hoisted
# (`hoistable`), because both evaluate transcendental coefficients per stage: putzer2 a
# complex sqrt/cosh/sinh/exp per mode, costing 4 complex full-grid arrays per stage of
# memory, and separable one real exp over (nkx,nky) plus exp/cos/sin over (nz), costing
# those same small arrays per stage (params.hoist_propagator turns both off). The diagonal
# backend is NOT: one real exp
# per mode per stage, z-broadcast for FD-z, so there is nothing to gain -- and leaving its
# exponent inside the stage keeps the FD-z/2D fixed-dt solver graph byte-identical to the
# pre-hoist one (gate 6's reference npz: with a literal gamma XLA folds (L*dt)*gamma
# differently, measured as 15 elements at 1e-23 absolute in the 64^2 gate-6 config).
#
# Backends are selected by what a recipe's linear_matrix_func returns (built once in
# grids.setup_kgrids):
#   diagonal   L.ndim == 4: (nfields-or-1, nz-or-1, nkx, nky)  elementwise
#   putzer2    L.ndim == 5: (2, 2, nz-or-1, nkx, nky)          closed-form 2x2 exponential
#   separable  a SeparableL: L = d*I + i*kz*sigma_x with real d = dperp(k_perp) + dz(kz)
import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple

from . import _precision

# Taylor branch for sinh(z)/z at small |z| (z = s*tau)
_TOL_Z2_FP64 = 1e-6   # |z| < 1e-3
_TOL_Z2_FP32 = 1e-4   # |z| < 1e-2

def _tol_z2():
    # precision-dependent Taylor cutoff
    return _TOL_Z2_FP64 if _precision.precision == "64" else _TOL_Z2_FP32

# ---------------------------------------------------------------- precomputed exp(L*tau)
# Pytrees (NamedTuples of arrays) so a per-stage stack of them can ride through a lax.scan
# as xs (timestepping._lsrk_scan_stages) and index back to one op per stage.

class IdentityExp(NamedTuple):
    def apply(self, arr):
        return arr

class DiagonalExp(NamedTuple):
    e: jnp.ndarray          # exp(L*tau), elementwise
    def apply(self, arr):
        return self.e*arr

class Putzer2Exp(NamedTuple):
    m00: jnp.ndarray        # the four entries of exp(L*tau), as Putzer2Propagator.apply_exp
    m01: jnp.ndarray        # forms them
    m10: jnp.ndarray
    m11: jnp.ndarray
    def apply(self, arr):
        return jnp.stack([self.m00*arr[0] + self.m01*arr[1], self.m10*arr[0] + self.m11*arr[1]])

def _mul_i(z):
    # i*z as the real swap it is
    return jax.lax.complex(-jnp.imag(z), jnp.real(z))

class SeparableExp(NamedTuple):
    P: jnp.ndarray          # exp(dperp*tau),               (nkx, nky)
    c: jnp.ndarray          # exp(dz*tau)*cos(kz*tau),      (nz, 1, 1)
    s: jnp.ndarray          # exp(dz*tau)*sin(kz*tau),      (nz, 1, 1)
    def apply(self, arr):
        return jnp.stack([self.P*(self.c*arr[0] + _mul_i(self.s*arr[1])),
                          self.P*(_mul_i(self.s*arr[0]) + self.c*arr[1])])

def stack_exp_ops(ops):
    # tuple of same-kind ExpOps -> one ExpOp with a leading stage axis (scan xs)
    return jax.tree.map(lambda *xs: jnp.stack(xs), *ops)

class IdentityPropagator:
    # L = 0
    hoistable = False

    def scaled(self, factor):
        return self

    def exp_op(self, tau):
        return IdentityExp()

    def apply_exp(self, arr, tau):
        return arr

    def solve_shifted(self, arr, a):
        return arr

    def apply_L(self, arr):
        return jnp.zeros_like(arr)

class DiagonalPropagator:
    # L diagonal in fields and k: everything is elementwise.
    hoistable = False     # see the header: nothing to gain, and it keeps the FD-z/2D graph

    def __init__(self, L):
        self.L = L

    def scaled(self, factor):
        # propagator of factor*L (LSRK pre-scales by dt: see lsrk_advance)
        return DiagonalPropagator(self.L*factor)

    def exp_op(self, tau):
        return DiagonalExp(jnp.exp(self.L*tau))

    def apply_exp(self, arr, tau):
        return self.exp_op(tau).apply(arr)

    def solve_shifted(self, arr, a):
        return arr/(1.0 - a*self.L)

    def apply_L(self, arr):
        return self.L*arr

class Putzer2Propagator:
    # nfields=2: exp(L*tau) = e^(m*tau)[cosh(s*tau) I + (sinh(s*tau)/s)(L - m I)] with
    # m = tr L/2 and s^2 = m^2 - det L (Putzer/Sylvester). 
    # m and s2 are precomputed at setup. 
    # all of the arithmetic is complex (waves)
    hoistable = True      # the per-stage coefficient evaluation is what hoisting removes

    def __init__(self, L, m, s2):
        self.L = L
        self.m = m
        self.s2 = s2

    def scaled(self, factor):
        # L -> factor*L scales the trace by factor and the discriminant by factor^2
        return Putzer2Propagator(self.L*factor, self.m*factor, self.s2*(factor*factor))

    def _coeffs(self, tau):
        # (cosh(s*tau), sinh(s*tau)/s) with a Taylor branch at small |s*tau|
        # overflow note: cosh/sinh overflow at |Re(s*tau)| ~ 710 (fp64) /
        # 88 (fp32). unreachable under adaptive dt but a large FIXED dt with
        # a strongly damped L can hit it, giving instant NaNs.
        z2 = self.s2*(tau*tau)
        z = jnp.sqrt(z2)
        small = jnp.abs(z2) < _tol_z2()
        z_safe = jnp.where(small, jnp.ones_like(z), z)   # keeps the 0/0 branch NaN-free
        sinhc = jnp.where(small,
                          1.0 + z2*(1.0/6.0 + z2*(1.0/120.0 + z2/5040.0)),
                          jnp.sinh(z_safe)/z_safe)
        return jnp.cosh(z), tau*sinhc

    def exp_op(self, tau):
        cosh_z, sinh_over_s = self._coeffs(tau)
        pref = jnp.exp(self.m*tau)
        # M = pref*(cosh I + (sinh/s)(L - m I)); its four entries, to apply to a (2, ...) stack
        d = cosh_z - sinh_over_s*self.m
        return Putzer2Exp(m00=pref*(d + sinh_over_s*self.L[0,0]),
                          m01=pref*(sinh_over_s*self.L[0,1]),
                          m10=pref*(sinh_over_s*self.L[1,0]),
                          m11=pref*(d + sinh_over_s*self.L[1,1]))

    def apply_exp(self, arr, tau):
        return self.exp_op(tau).apply(arr)

    def solve_shifted(self, arr, a):
        # closed-form inverse of the 2x2 M = I - a*L. Pole note (mirrors apply_exp's
        # overflow note): det(I - a*L) = 0 when a growing eigenvalue satisfies
        # lambda = 1/a (a = a_ii*dt in the IMEX steppers, so lambda*dt ~ 2-3). 
        # unreachable under an adaptive dt (gamma_max*dt ~ O(1) with safety < 1)
        # possible with large fixed dt: symptom is instant NaNs/huge values.
        m00 = 1.0 - a*self.L[0,0]
        m01 = -a*self.L[0,1]
        m10 = -a*self.L[1,0]
        m11 = 1.0 - a*self.L[1,1]
        det = m00*m11 - m01*m10
        return jnp.stack([(m11*arr[0] - m01*arr[1])/det,
                          (m00*arr[1] - m10*arr[0])/det])

    def apply_L(self, arr):
        return jnp.stack([self.L[0,0]*arr[0] + self.L[0,1]*arr[1],
                          self.L[1,0]*arr[0] + self.L[1,1]*arr[1]])

class SeparableL(NamedTuple):
    # nfields=2 with L = d*I + i*kz*sigma_x, d = dperp(k_perp) + dz(kz); all three entries
    # real. A recipe's linear_matrix_func may return this instead of a dense array.
    dperp: jnp.ndarray      # (nkx, nky)
    dz: jnp.ndarray         # (nz, 1, 1)
    kz: jnp.ndarray         # (nz, 1, 1)

class SeparablePropagator:
    # exp(L*tau) = exp(dperp*tau)*exp(dz*tau)*[cos(kz*tau) I + i sin(kz*tau) sigma_x] and
    # (I - a*L)^-1 = [(1-a*d) I + i*a*kz*sigma_x]/((1-a*d)^2 + (a*kz)^2): every per-stage
    # array is (nkx,nky) or (nz,1,1), never full-grid.
    hoistable = True      # see the header: the per-stage coefficient evaluation is the cost

    def __init__(self, sep):
        self.sep = sep

    def scaled(self, factor):
        # propagator of factor*L: d and kz both scale by factor
        return SeparablePropagator(SeparableL(self.sep.dperp*factor, self.sep.dz*factor,
                                              self.sep.kz*factor))

    def exp_op(self, tau):
        ez = jnp.exp(self.sep.dz*tau)
        theta = self.sep.kz*tau
        return SeparableExp(P=jnp.exp(self.sep.dperp*tau),
                            c=ez*jnp.cos(theta), s=ez*jnp.sin(theta))

    def apply_exp(self, arr, tau):
        return self.exp_op(tau).apply(arr)

    def solve_shifted(self, arr, a):
        # d <= 0, so for a > 0 the denominator is >= 1: this backend has no pole (unlike
        # putzer2's det(I - a*L); see its solve_shifted)
        w = 1.0 - a*(self.sep.dperp + self.sep.dz)
        akz = a*self.sep.kz
        den = w*w + akz*akz
        return jnp.stack([(w*arr[0] + _mul_i(akz*arr[1]))/den,
                          (_mul_i(akz*arr[0]) + w*arr[1])/den])

    def apply_L(self, arr):
        d = self.sep.dperp + self.sep.dz
        return jnp.stack([d*arr[0] + _mul_i(self.sep.kz*arr[1]),
                          _mul_i(self.sep.kz*arr[0]) + d*arr[1]])

def get_propagator(kgrid, params):
    if kgrid.lin_dperp is not None:
        return SeparablePropagator(SeparableL(kgrid.lin_dperp, kgrid.lin_dz, kgrid.lin_kz))
    L = kgrid.lin_L
    if L is None:
        return IdentityPropagator()
    if L.ndim == 4:
        return DiagonalPropagator(L)
    return Putzer2Propagator(L, kgrid.lin_m, kgrid.lin_s2)

def dense_operator(kgrid):
    # the (2, 2, nz-or-1, nkx, nky) L behind whatever backend the kgrid carries. Tests and
    # validation only: it is 4 u and must never be formed inside a step.
    if kgrid.lin_dperp is not None:
        d = kgrid.lin_dperp + kgrid.lin_dz
        off = jnp.broadcast_to(1j*kgrid.lin_kz, d.shape)
        zero = jnp.zeros_like(off)
        return jnp.stack([jnp.stack([d + zero, off]), jnp.stack([off, d + zero])])
    L = kgrid.lin_L
    if L is None:
        raise ValueError("this kgrid carries no linear operator (the identity backend)")
    if L.ndim == 5:
        return L
    if L.shape[0] not in (1, 2):
        raise ValueError(f"a diagonal operator with leading axis {L.shape[0]} is not a 2x2 L")
    d0, d1 = (L[0], L[0]) if L.shape[0] == 1 else (L[0], L[1])
    zero = jnp.zeros_like(d0)
    return jnp.stack([jnp.stack([d0, zero]), jnp.stack([zero, d1])])

def _mirror_k(arr, axis):
    # index k -> -k along a two-sided FFT axis (0 and Nyquist map to themselves)
    return np.roll(np.flip(arr, axis=axis), 1, axis=axis)

def _check_hermitian_compatible(L, params):
    # reality of the fields survives exp(L*tau) only if L(-kx,ky) = conj(L(kx,ky)) on the
    # rfft2 self-conjugate rows ky=0 and ky=Nyquist (kx is two-sided there). Under
    # params.z_spectral the z axis is kz (also two-sided), so the constraint there is
    # L(-kx,-kz,ky) = conj(L(kx,kz,ky)) and the mirror covers both axes.
    Ln = np.asarray(L)
    kx_axis = Ln.ndim - 2
    mirrored = _mirror_k(Ln, kx_axis)
    if getattr(params, "z_spectral", False) and Ln.shape[Ln.ndim - 3] > 1:
        mirrored = _mirror_k(mirrored, Ln.ndim - 3)
    tol = 1e3*np.finfo(np.asarray(Ln.real).dtype).eps*max(1.0, float(np.max(np.abs(Ln))))
    for row in (0, Ln.shape[-1] - 1):     # ky = 0 and the Nyquist row
        a = Ln[..., row]
        b = mirrored[..., row]
        if not np.allclose(b, np.conj(a), rtol=0.0, atol=tol):
            raise ValueError(
                f"{params.eqtype}: linear_matrix_func returned an L that breaks the rfft2 "
                f"reality constraint on the ky index {row} row (need L(-kx[,-kz],ky) = "
                f"conj(L(kx[,kz],ky)) there); max violation "
                f"{float(np.max(np.abs(b - np.conj(a)))):.3e}")

def _mirror_tol(arr):
    return 1e3*np.finfo(np.asarray(arr).dtype).eps*max(1.0, float(np.max(np.abs(arr))))

def _check_hermitian_separable(sep, params):
    # L = d*I + i*kz*sigma_x survives the rfft2/rfftn reality constraint iff d is real and
    # mirror-symmetric on the self-conjugate ky rows and kz is mirror-ANTIsymmetric (which
    # forces kz = 0 at the planes the mirror maps to themselves: kz index 0 and, for even
    # nz, the Nyquist plane).
    dperp, dz, kz = (np.asarray(a) for a in sep)
    for name, arr in zip(sep._fields, (dperp, dz, kz)):
        if np.iscomplexobj(arr):
            raise ValueError(f"{params.eqtype}: separable linear operator entry '{name}' "
                             f"must be real, got dtype {arr.dtype}")
    for row in (0, dperp.shape[-1] - 1):     # ky = 0 and the Nyquist row
        a = dperp[..., row]
        if not np.allclose(_mirror_k(a, 0), a, rtol=0.0, atol=_mirror_tol(dperp)):
            raise ValueError(
                f"{params.eqtype}: the separable operator's dperp is not symmetric under "
                f"kx -> -kx on the ky index {row} row (need dperp(-kx,ky) = dperp(kx,ky) "
                f"there for the fields to stay real)")
    if not np.allclose(_mirror_k(dz, 0), dz, rtol=0.0, atol=_mirror_tol(dz)):
        raise ValueError(f"{params.eqtype}: the separable operator's dz is not symmetric "
                         f"under kz -> -kz")
    if not np.allclose(_mirror_k(kz, 0), -kz, rtol=0.0, atol=_mirror_tol(kz)):
        raise ValueError(
            f"{params.eqtype}: the separable operator's kz is not antisymmetric under "
            f"kz -> -kz (i*kz must equal its own conjugate at the kz = 0 and Nyquist "
            f"planes, i.e. kz must vanish there); max violation "
            f"{float(np.max(np.abs(_mirror_k(kz, 0) + kz))):.3e}")

def _separable_fields(sep, params, nz_local, nkx, nky):
    if params.nfields != 2:
        raise ValueError(f"{params.eqtype}: the separable backend needs nfields=2, got "
                         f"{params.nfields}")
    shapes = (sep.dperp.shape, sep.dz.shape, sep.kz.shape)
    if shapes != ((nkx, nky), (nz_local, 1, 1), (nz_local, 1, 1)):
        raise ValueError(f"{params.eqtype}: the separable backend needs shapes dperp "
                         f"({nkx}, {nky}), dz and kz ({nz_local}, 1, 1), got {shapes}")
    if params.comm_backend == "jax":
        # kgrid_specs replicates these entries; a z-EXTENT operator would need its own
        # z-sharded spec
        raise NotImplementedError(f"{params.eqtype}: a z-dependent linear operator is not "
                                  "supported by comm_backend='jax' yet")
    _check_hermitian_separable(sep, params)
    return dict(lin_dperp=sep.dperp, lin_dz=sep.dz, lin_kz=sep.kz)

def linear_fields(L, params):
    # Validate a recipe's L and return the K_Grids entries it populates (grids.setup_kgrids
    # is the only caller: the propagator arrays are kgrid entries, not a separate object).
    nz_local = params.nz//params.size if params.spatial_dimensions == 3 else 1
    nkx, nky = params.nx, params.ny//2 + 1
    if isinstance(L, SeparableL):
        return _separable_fields(L, params, nz_local, nkx, nky)
    if L.ndim == 4:
        if L.shape[0] not in (1, params.nfields):
            raise ValueError(f"{params.eqtype}: diagonal linear operator has leading axis "
                             f"{L.shape[0]}, expected 1 or nfields={params.nfields}")
        zdim, perp = L.shape[1], L.shape[2:]
    elif L.ndim == 5:
        if params.nfields != 2 or L.shape[:2] != (2, 2):
            raise ValueError(f"{params.eqtype}: the putzer2 backend needs a (2, 2, nz-or-1, "
                             f"nkx, nky) linear operator with nfields=2, got shape "
                             f"{L.shape} with nfields={params.nfields}")
        zdim, perp = L.shape[2], L.shape[3:]
    else:
        raise ValueError(f"{params.eqtype}: linear_matrix_func must return a 4-d diagonal "
                         f"(nfields, nz-or-1, nkx, nky) or 5-d putzer2 (2, 2, nz-or-1, nkx, "
                         f"nky) operator, got shape {L.shape}")
    if perp != (nkx, nky):
        raise ValueError(f"{params.eqtype}: linear operator perpendicular shape {perp} "
                         f"does not match the grid ({nkx}, {nky})")
    if zdim not in (1, nz_local):
        raise ValueError(f"{params.eqtype}: linear operator z axis {zdim} must be 1 "
                         f"(broadcast) or the local z extent {nz_local}")
    if zdim > 1 and params.comm_backend == "jax":
        # kgrid_specs replicates these entries; a z-EXTENT operator would need its own
        # z-sharded spec (arrives with the spectral-z work).
        raise NotImplementedError(f"{params.eqtype}: a z-dependent linear operator is not "
                                  "supported by comm_backend='jax' yet")
    _check_hermitian_compatible(L, params)
    if L.ndim == 4:
        return dict(lin_L=L)
    Lc, m, s2 = putzer2_precompute(L)
    return dict(lin_L=Lc, lin_m=m, lin_s2=s2)

def putzer2_precompute(L):
    # local import: physics sits above this module (physics -> grids -> propagators)
    from .physics.shared_physics import eig2_ms
    Lc = jnp.asarray(L).astype(jnp.result_type(jnp.asarray(L).dtype, jnp.complex64))
    m, s2 = eig2_ms(Lc[0,0], Lc[0,1], Lc[1,0], Lc[1,1])
    return Lc, m, s2
