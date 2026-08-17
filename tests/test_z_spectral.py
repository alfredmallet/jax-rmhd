# Spectral-z mode (params.z_spectral, plans/GDI_PLAN.md P2): axis 1 of state.fields is kz
# instead of z, the parallel (Alfven) operator moves from an RHS finite-difference term into
# the exact linear propagator, and every energy-like reduction picks up the Parseval factor
# of nz for the unnormalized z-FFT.
# pytest: single-process (z_spectral IS single-process by construction).
# Savio driver: `python tests/test_z_spectral.py` -- never under mpirun (size==1 only).
from _rmhd_testing import (bootstrap, checks, ctx, fit_order, fresh_params, make_state,
                           snap_dir, zero_ic)

bootstrap()

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import taranis as jr
from taranis import _precision
from taranis import diagnostics as dg
from taranis.grids import fft, ifft
from taranis.physics import rmhd, shared_physics
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

_BOX = dict(nx=16, ny=16, Lx=2.0 * np.pi, Ly=2.0 * np.pi, Lz=2.0 * np.pi, dims=3)


def _fp64():
    # FIELD precision (RMHD_PRECISION) -- jax_enable_x64 is now unconditionally on.
    return _precision.precision == "64"


def _have_mpi4py():
    try:
        from mpi4py import MPI  # noqa: F401
        return True
    except Exception:
        return False


def _alfven_plus_ic(x, y, z):
    # phi == psi bitwise => z- == 0 => BOTH brackets vanish identically, so this is a pure
    # linear z+ Alfven packet: the exact solution is phi0(x, y, z + t) at every mode.
    phi = (jnp.cos(x) * jnp.cos(y) * jnp.cos(z) + 0.3 * jnp.sin(2 * x + y) * jnp.sin(2 * z)
           + 0.2 * jnp.cos(x - y) * jnp.cos(3 * z) + 0.1 * jnp.sin(x) * jnp.sin(4 * z))
    return jnp.stack([phi, phi])


def _nonlinear_ic(x, y, z):
    # z+ and z- both present: the brackets are live, so this exercises the full RHS
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z) + 0.3 * jnp.sin(2 * x + y) * jnp.sin(z)
    psi = phi + 0.3 * jnp.sin(x) * jnp.cos(y) * jnp.sin(z)
    return jnp.stack([phi, psi])


def _advance(state, kgrid, params, nsteps, schemestr="lsrk54"):
    # nsteps of the ordinary stepper, jitted exactly as run.py does it. NB donation is NOT
    # used here, so the caller's state stays valid (block_of_steps itself never donates).
    stepper, scheme = get_scheme(schemestr)
    f = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))
    return f(state, kgrid, params, nsteps, scheme, stepper)


# --------------------------------------------------------------------------- guards

def test_z_spectral_guards():
    # dims/size/backend guards, and the finite-difference-z knobs it silences
    with checks() as c:
        with pytest.raises(ValueError, match="dims=3"):
            jr.Parameters(nx=8, ny=8, Lx=1.0, Ly=1.0, cfl_safety=0.5, dims=2,
                          eqpars={"diss": 0.0, "hyper": 1}, z_spectral=True)
        c.check("z_spectral=True with dims=2 raises", True)
        if _have_mpi4py():   # comm_backend="jax" needs mpi4py (real or tests/local_mpi_stub)
            with pytest.raises(ValueError, match="comm_backend='jax'"):
                jr.Parameters(nz=8, cfl_safety=0.5, eqpars={"diss": 0.0, "hyper": 1},
                              comm_backend="jax", z_spectral=True, **_BOX)
            c.check("z_spectral=True with comm_backend='jax' raises", True)
        with pytest.warns(UserWarning, match="z_diss"):
            jr.Parameters(nz=8, cfl_safety=0.5, eqpars={"diss": 0.0, "hyper": 1},
                          z_diss=1.0, z_spectral=True, **_BOX)
        c.check("z_spectral=True warns that z_diss is ignored", True)
        with pytest.raises(ValueError, match="z_diss_k"):
            # the kz hyperdissipation knob is spectral-z only
            jr.setup_kgrids(fresh_params(nz=8, eqpars={"diss": (0.0, 0.0), "hyper": 1,
                                                       "z_diss_k": 1.0}))
        c.check("eqpars['z_diss_k'] without z_spectral raises", True)


def test_params_json_records_z_spectral():
    # snapshots are NOT cross-mode compatible; params.save's differing-record check is the guard
    params_sp, _ = ctx(nz=8, z_spectral=True)
    params_fd, _ = ctx(nz=8)
    with snap_dir() as d, checks() as c:
        params_sp.save(d)
        restored = jr.Parameters.from_snapshot(d)
        c.check("z_spectral round-trips through params.json",
                restored.z_spectral is True, f"got {restored.z_spectral!r}")
        params_sp.save(d)   # identical re-save: no-op
        c.check("identical re-save of a z_spectral record is a no-op", True)
        with pytest.raises(ValueError, match="z_spectral"):
            params_fd.save(d)
        c.check("saving a z_spectral=False record over a z_spectral=True one is a hard error",
                True)


# ------------------------------------------------------------------- grids / transforms

def test_transforms_and_dealias():
    params, kgrid = ctx(nz=16, z_spectral=True)
    nkx, nky = params.nx, params.ny // 2 + 1
    x = jnp.linspace(0, params.Lx, params.nx, endpoint=False).reshape(1, -1, 1)
    y = jnp.linspace(0, params.Ly, params.ny, endpoint=False).reshape(1, 1, -1)
    z = jnp.linspace(0, params.Lz, params.nz, endpoint=False).reshape(-1, 1, 1)
    f_real = _nonlinear_ic(x, y, z)
    fk = fft(f_real, params)
    tol = 1e-12 if _fp64() else 1e-5
    with checks() as c:
        c.check("fft keeps the (nfields, nz, nkx, nky) shape, axis 1 now kz",
                fk.shape == (2, params.nz, nkx, nky), f"{fk.shape}")
        rt = float(jnp.max(jnp.abs(ifft(fk, params) - f_real)))
        c.check("ifft(fft(f)) == f", rt < tol, f"max err {rt:.3e}")
        c.check("dealias mask gains a kz axis", kgrid.dealias.shape == (params.nz, nkx, nky),
                f"{kgrid.dealias.shape}")
        iz = np.fft.fftfreq(params.nz) * params.nz
        kz_keep = np.asarray(kgrid.dealias).any(axis=(1, 2))
        c.check("kz 2/3 cut: exactly the |n_z| < nz/3 planes survive",
                bool(np.all(kz_keep == (np.abs(iz) < params.nz / 3.0))),
                f"kept {iz[kz_keep]}")
        # the IC masking in run.initialize must pick the kz cut up with no physics change
        def high_kz(x, y, z):
            n = params.nz // 2   # far above the kz cutoff
            f = jnp.cos(x) * jnp.cos(n * z) * jnp.ones_like(y)
            return jnp.stack([f, jnp.zeros_like(f)])
        st = make_state(params, ic=high_kz)
        c.check("initialize() removes IC energy beyond the kz cutoff",
                float(jnp.max(jnp.abs(st.fields))) == 0.0,
                f"max|fields| = {float(jnp.max(jnp.abs(st.fields))):.3e}")


def test_reality_is_preserved_by_a_forced_run():
    # nothing in the mode may break F(-kx,-kz,ky) = conj(F(kx,kz,ky)): the propagator's
    # +-i*kz entries, the forcing scatter onto kz=+-2pi/Lz, and the brackets all have to
    # keep it. fft(ifft(f)) projects onto the Hermitian subspace, so it == f iff f is real.
    params, kgrid = ctx(nz=16, z_spectral=True, forcing=True, forcing_mode="elsasser",
                        forcing_power_elsasser=(0.4, 0.2), forcing_tau=0.5, fshell=(1, 3),
                        forcing_seed=3)
    end = _advance(make_state(params, ic=_nonlinear_ic), kgrid, params, 20)
    resid = float(jnp.max(jnp.abs(fft(ifft(end.fields, params), params) - end.fields)))
    scale = float(jnp.max(jnp.abs(end.fields)))
    tol = 1e-10 if _fp64() else 1e-3
    with checks() as c:
        c.check("evolved forced z_spectral state stays exactly Hermitian",
                resid / scale < tol, f"relative residual {resid / scale:.3e}")


# ---------------------------------------------------------------------------- physics

@pytest.mark.fp64
def test_alfven_dispersion_exact():
    # Single-(k_perp,kz)-per-coefficient measurement: with phi == psi the RHS is exactly
    # zero, so every Fourier coefficient is propagated by exp(L*tau) alone and must satisfy
    # F(T)/F(0) = exp(i*kz*T - diss*k_perp^(2*hyper)*T) to round-off -- no CFL, no order.
    with checks() as c:
        for diss in (0.0, 0.02):
            params, kgrid = ctx(nz=16, z_spectral=True, diss=(diss, diss), hyper=1,
                                dt=0.025, adaptive_timestep=False)
            state = make_state(params, ic=_alfven_plus_ic)
            f0 = np.asarray(state.fields[0])
            end = _advance(state, kgrid, params, 20)
            T = float(end.t)
            f1 = np.asarray(end.fields[0])
            sel = np.abs(f0) > 1e-3 * np.abs(f0).max()   # only well-resolved coefficients
            ratio = np.where(sel, f1 / np.where(sel, f0, 1.0), 1.0)
            kz = np.asarray(kgrid.kz)[:, 0, 0].reshape(-1, 1, 1)
            omega_err = np.max(np.abs(np.angle(ratio) / T - kz)[sel])   # |kz|*T <= 2 < pi
            gamma_err = np.max(np.abs(-np.log(np.abs(ratio)) / T
                                      - diss * np.asarray(kgrid.ksq)[None])[sel])
            c.check(f"measured omega == kz to machine precision (diss={diss}, "
                    f"{int(sel.sum())} modes)", omega_err < 1e-12, f"max err {omega_err:.3e}")
            c.check(f"measured damping == diss*k_perp^2 to machine precision (diss={diss})",
                    gamma_err < 1e-12, f"max err {gamma_err:.3e}")


@pytest.mark.fp64
def test_fd_z_converges_to_spectral_at_fourth_order():
    # A/B against the finite-difference-z code: same IC, same nz on both sides (identical kz
    # truncation, so the ONLY difference is d/dz vs i*kz), halving dz. z_diss=0 on the FD
    # side keeps the comparison to the 4th-order stencil error alone.
    nzs = (16, 32, 64)
    common = dict(diss=(0.0, 0.0), hyper=1, dt=0.02, adaptive_timestep=False)
    errs = []
    for nz in nzs:
        p_fd, kg_fd = ctx(nz=nz, z_diss=0.0, **common)
        p_sp, kg_sp = ctx(nz=nz, z_spectral=True, **common)
        f_fd = ifft(_advance(make_state(p_fd, ic=_nonlinear_ic), kg_fd, p_fd, 40).fields, p_fd)
        f_sp = ifft(_advance(make_state(p_sp, ic=_nonlinear_ic), kg_sp, p_sp, 40).fields, p_sp)
        errs.append(float(jnp.max(jnp.abs(f_fd - f_sp))))
    order = fit_order([2.0 * np.pi / nz for nz in nzs], errs)
    with checks() as c:
        c.check("FD-z result converges to the spectral-z one at 4th order",
                order > 3.5, f"order={order:.3f}, errs={errs}")
        c.check("FD-vs-spectral difference decreases monotonically in nz",
                all(a > b for a, b in zip(errs, errs[1:])), f"errs={errs}")


@pytest.mark.fp64
def test_forcing_power_parseval_matches_real_z():
    # The normalization sweep's cross-check: for the SAME physical field and the SAME
    # (A,B) forcing envelope, the injection power computed with a real-z state must equal
    # the one computed with a kz state (Parseval over the unnormalized z-FFT).
    forcing = dict(forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(0.4, 0.2),
                   forcing_tau=0.5, fshell=(1, 3), forcing_seed=5)
    out = {}
    for spectral in (False, True):
        params, kgrid = ctx(nz=16, z_spectral=spectral, **forcing)
        state = make_state(params, ic=_nonlinear_ic)
        fs, key = state.forcing_state, state.forcing_key
        for _ in range(5):   # identical RNG stream in both modes (ou_update is perp-only)
            fs, key = shared_physics.ou_update(fs, key, 0.05, params, kgrid)
        state = state._replace(forcing_state=fs)
        f_raw = shared_physics.reconstruct_envelope(fs, kgrid, params)
        za = jnp.stack([state.fields[0] + state.fields[1], state.fields[0] - state.fields[1]])
        power = shared_physics.perp_inner_product(za, f_raw, kgrid, params, batch=True)
        # dt=0.05 (the same step the OU state was advanced with): a NONZERO dt on purpose,
        # so this also cross-checks the self-energy reduction F2 = <|grad f_raw|^2> that
        # selfnorm_scale added in 2026-08-08 -- it carries the same nz/2 envelope factors and
        # perp_reduce 1/nz^2 as the power denominator, so it must Parseval-match too.
        scale = rmhd.forcing_scale(state, kgrid, params, 0.05)
        fterm = rmhd.ForcingTerm(state._replace(forcing_scale=scale),
                                 rmhd.grad(state, kgrid, params), kgrid, params)
        # realized dE/dt = <grad phi . grad f_phi> + <grad psi . grad f_psi>
        inject = float(shared_physics.perp_inner_product(state.fields[0], fterm[0], kgrid, params)
                       + shared_physics.perp_inner_product(state.fields[1], fterm[1], kgrid, params))
        out[spectral] = (np.asarray(power), np.asarray(scale), inject,
                         np.array([float(e) for e in dg.energy(state, kgrid, params)]))
    fd, sp = out[False], out[True]
    rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    with checks() as c:
        c.check("elsasser power denominators match real-z to round-off",
                rel(fd[0], sp[0]) < 1e-13, f"rel {rel(fd[0], sp[0]):.3e}")
        c.check("forcing normalization scales match real-z to round-off",
                rel(fd[1], sp[1]) < 1e-13, f"rel {rel(fd[1], sp[1]):.3e}")
        c.check("realized injection rate matches real-z to round-off",
                abs(fd[2] - sp[2]) / abs(fd[2]) < 1e-13,
                f"rel {abs(fd[2] - sp[2]) / abs(fd[2]):.3e}, fd={fd[2]:.6f}")
        c.check("diagnostics.energy matches real-z to round-off",
                rel(fd[3], sp[3]) < 1e-13, f"rel {rel(fd[3], sp[3]):.3e}")


def test_parspec_integrates_to_energy():
    # parspec is a trivial kz sum in this mode, so it works by construction -- and it must
    # still share the one normalization: integral of the parallel spectrum == diagnostics.energy
    params, kgrid = ctx(nz=16, z_spectral=True)
    state = make_state(params, ic=_nonlinear_ic)
    kz, spec_phi, spec_psi = dg.parspec(state, kgrid, params, bin_factor=1.0)
    dk = float(kz[1] - kz[0])
    E = dg.energy(state, kgrid, params)
    tol = 1e-10 if _fp64() else 1e-4
    with checks() as c:
        for name, spec, e in (("kinetic", spec_phi, E[0]), ("magnetic", spec_psi, E[1])):
            integral = float(jnp.sum(spec)) * dk
            c.check(f"integral of the {name} parallel spectrum == diagnostics.energy",
                    abs(integral - float(e)) <= tol * max(1.0, abs(float(e))),
                    f"integral={integral:.12g}, energy={float(e):.12g}")


def test_parallel_terms_are_skipped_and_dt_is_perp_only():
    # FDLinearTerm/halo_start are dead in this mode (the propagator owns the parallel physics),
    # and set_timestep drops the 1/dz and z_diss entries -- the payoff of the mode.
    # adaptive_timestep=True: set_timestep caps at rmhd._quiescent_dt, which is params.dt
    # on the fixed-dt path (tests/test_quiescent_dt.py).
    params, kgrid = ctx(nz=8, z_spectral=True, adaptive_timestep=True)
    params_fd, kgrid_fd = ctx(nz=8, adaptive_timestep=True)
    state = make_state(params, ic=zero_ic)
    grads = rmhd.grad(state, kgrid, params)
    dt_sp = float(rmhd.set_timestep(grads, params))
    dt_fd = float(rmhd.set_timestep(rmhd.grad(make_state(params_fd, ic=zero_ic),
                                              kgrid_fd, params_fd), params_fd))
    eps = shared_physics.QUIESCENT_EPS
    dt_perp_only = params.cfl_safety / max(eps / params.dx, eps / params.dy)
    with checks() as c:
        c.check("FDLinearTerm is exactly zero in z_spectral mode",
                bool(jnp.all(rmhd.FDLinearTerm(state, grads, kgrid, params) == 0)))
        c.check("halo_start issues no exchange in z_spectral mode",
                rmhd.halo_start(state, kgrid, params) is None)
        c.check("set_timestep drops the 1/dz and z_diss entries",
                abs(dt_sp - dt_perp_only) < (1e-12 if _fp64() else 1e-5) * dt_perp_only,
                f"dt_spectral={dt_sp:.6g}, perp-only CFL={dt_perp_only:.6g}")
        c.check("spectral dt is larger than the finite-difference-z dt on the same grid",
                dt_sp > dt_fd, f"dt_spectral={dt_sp:.6g}, dt_fd={dt_fd:.6g}")


def test_kz_hyperdissipation_knob():
    # optional -z_diss_k*kz^4 diagonal (off by default): a pure z+ packet then decays at
    # exactly z_diss_k*kz^4 per mode on top of the exact wave propagation.
    z_diss_k = 1e-3
    # fresh_params, not ctx: ctx's cache key must stay hashable, so it takes no eqpars dict
    params = fresh_params(nz=16, z_spectral=True, dt=0.025, adaptive_timestep=False,
                          eqpars={"diss": (0.0, 0.0), "hyper": 1, "z_diss_k": z_diss_k})
    kgrid = jr.setup_kgrids(params)
    state = make_state(params, ic=_alfven_plus_ic)
    f0 = np.asarray(state.fields[0])
    end = _advance(state, kgrid, params, 20)
    T = float(end.t)
    f1 = np.asarray(end.fields[0])
    # 1e-3 cut: at fp32 the coefficients below ~1e-6 of the peak are transform round-off,
    # and a rate measured from those is noise, not a damping rate
    sel = np.abs(f0) > 1e-3 * np.abs(f0).max()
    ratio = np.where(sel, f1 / np.where(sel, f0, 1.0), 1.0)
    kz = np.asarray(kgrid.kz)[:, 0, 0].reshape(-1, 1, 1)
    gamma_err = np.max(np.abs(-np.log(np.abs(ratio)) / T - z_diss_k * kz**4)[sel])
    tol = 1e-12 if _fp64() else 1e-4
    with checks() as c:
        c.check("kz hyperdissipation damps at exactly z_diss_k*kz^4",
                gamma_err < tol, f"max err {gamma_err:.3e}")
        c.check("z_diss_k is off by default", "z_diss_k" not in ctx(nz=8)[0].eqpars)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
