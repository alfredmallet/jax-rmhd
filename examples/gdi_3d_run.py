# 3D GDI production run (plans/GDI_PLAN.md P4b), examples/gdi_2d_run.py's pattern.
#
# Sized for a laptop/sandbox demonstration, NOT production (see the sizing note in
# taranis/physics/gdi.py's module docstring for the 512^3 single-GPU estimate): 64x64x16,
# fp32 fine here. z_spectral=True (required for GDI dims==3), eqpars D_par-only (gpar_fac=0,
# the P4b default -- see gdi.py's module docstring for the "floor not retired, default off"
# decision), production scheme imexcb3e (CB-IMEX, the P4b-recommended L-stable scheme).
#
# HONESTY NOTE (matches the P4a precedent in gdi_2d_run.py, but this file has NOT been
# through that notebook's multi-round saturation-amplitude retune): the run driven by
# make_data below reaches a clearly NONLINEAR state (energy grows by orders of magnitude
# past the linear e-folding budget below) but was not tuned/verified to reach a
# statistically STEADY saturated cascade the way the 2D driver eventually was -- there was
# no sandbox time budget in this task for that iteration. The diagnostics
# (taranis.diagnostics.gdi: kperp_break/measure_alpha/cross_phase_spectrum/
# theory_cross_phase) are demonstrated on whatever state this reaches via report() below
# (there is no companion notebook yet -- report() prints the run's actual energy-vs-time
# honestly rather than asserting saturation).
import time

import numpy as np
import jax.numpy as jnp

import taranis as jr
import taranis.snapshot_io as sn
from taranis.diagnostics import gdi as gdi_diag
from taranis.physics import gdi, shared_physics
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

NX = NY = 64
NZ = 16
LX = LY = 8.0 * np.pi
LZ = 2.0 * np.pi
# D_par chosen (see the module-level probe in the P4b report) so several kz planes are
# unstable (kperp_break shrinks with kz, vanishing by kz~3-4 on this box) while gpar_fac=0
# keeps the run an honest test of the REAL 3D closure, not the 2D floor.
EQPARS = dict(Ln=3.0, nu_in=0.2, v0=2.0, D_par=0.05, gpar_fac=0.0, diss=1e-3, hyper=2)


def make_params():
    # one place for the run configuration -- identical every call (params.save's
    # identical-record-is-a-no-op guard is what makes make_data's resume path safe).
    return jr.Parameters(nx=NX, ny=NY, nz=NZ, Lx=LX, Ly=LY, Lz=LZ, cfl_safety=0.4, dims=3,
                         z_spectral=True, eqtype="GDI", eqpars=EQPARS,
                         adaptive_timestep=True, cfl_every=1)


def ic(x, y, z):
    # small-amplitude (1e-3), multi-mode, random-phase perturbation in both N and phi,
    # spanning a range of kx, ky AND kz (unlike the 2D driver) so every kz plane the linear
    # operator favors has some seed to grow from.
    rng = np.random.default_rng(1)
    N = jnp.zeros_like(x * y * z)
    phi = jnp.zeros_like(x * y * z)
    for kxi in range(0, 8):
        for kyi in range(0, 8):
            for kzi in range(0, 4):
                if kxi == 0 and kyi == 0:
                    continue
                aN, aP, ph1, ph2 = rng.normal(size=4) * 1e-3
                kx = kxi * 2 * np.pi / LX
                ky = kyi * 2 * np.pi / LY
                kz = kzi * 2 * np.pi / LZ
                N = N + aN * jnp.cos(kx * x + ky * y + kz * z + ph1)
                phi = phi + aP * jnp.cos(kx * x + ky * y + kz * z + ph2)
    return jnp.stack([N, phi])


def make_data(snap_path, t_end=30.0, snap_every=1.0, wall_budget=33.0, nblock=100,
             schemestr="imexcb3e", nsnap=60):
    # Idempotent + resumable, same contract as examples/gdi_2d_run.py::make_data (see there
    # for the full rationale): call repeatedly (sandbox: chunked bash calls; laptop: a plain
    # loop) until it returns True.
    params = make_params()
    params.save(snap_path)
    kgrid = jr.setup_kgrids(params)
    stepper, scheme = get_scheme(schemestr)

    steps = sn.get_saved_steps(snap_path)
    if steps:
        state = sn.load_snapshot(steps[-1], snap_path, params)
        next_snap = steps[-1] + 1
        if float(state.t) >= t_end:
            print(f"make_data: already complete (t={float(state.t):.3f} >= t_end={t_end}) "
                 f"-- {len(steps)} snapshots on disk, nothing to do")
            return True
        print(f"make_data: resuming from snapshot {steps[-1]} at t={float(state.t):.3f}")
    else:
        state = jr.initialize(ic, params)
        next_snap = 0

    mngr = jr.snapshot_manager_setup(params=params, snap_path=snap_path, nsnap=nsnap)
    if next_snap == 0:
        sn.save_snapshot(0, state, mngr, params)
        mngr.wait_until_finished()
        next_snap = 1

    t_start_wall = time.time()
    t_last_snap = float(state.t)
    while float(state.t) < t_end and time.time() - t_start_wall < wall_budget:
        state = block_of_steps(state, kgrid, params, nblock, scheme, stepper)
        t_now = float(state.t)
        if t_now - t_last_snap >= snap_every or t_now >= t_end:
            sn.save_snapshot(next_snap, state, mngr, params)
            mngr.wait_until_finished()
            next_snap += 1
            t_last_snap = t_now
    if float(state.t) > t_last_snap:
        sn.save_snapshot(next_snap, state, mngr, params)
        next_snap += 1
        t_last_snap = float(state.t)
    mngr.wait_until_finished()

    done = float(state.t) >= t_end
    print(f"make_data: reached t={float(state.t):.3f} (target {t_end}) in "
         f"{time.time()-t_start_wall:.1f}s wall this call; done={done}")
    return done


def unstable_modes_report(params, kgrid):
    # quick static report: dt ceiling and the count/max of unstable dealiased lattice modes
    # (same style as gdi_2d_run.py's calibration probes, kept here as a reusable summary).
    L = gdi.linear_matrix(kgrid, params)
    m, s2 = shared_physics.eig2_ms(L[0, 0], L[0, 1], L[1, 0], L[1, 1])
    s = jnp.sqrt(s2.astype(jnp.complex64))
    max_re = jnp.maximum(jnp.real(m + s), jnp.real(m - s))
    dealias = kgrid.dealias
    n_unstable = int(jnp.sum((max_re > 1e-8) & dealias))
    n_dealias = int(jnp.sum(dealias))
    gmax = float(jnp.max(jnp.where(dealias, max_re, -jnp.inf)))
    # growth-only ceiling since review round 2 (M1): 0.0 = linearly stable = no ceiling
    lam = gdi._max_re_lambda(params)
    dt_ceiling = gdi._lin_dt_safety(params)/lam if lam > 0.0 else float("inf")
    return dict(n_unstable=n_unstable, n_dealias=n_dealias, gamma_max=gmax,
               dt_ceiling=dt_ceiling)


def report(snap_path="examples/data/gdi-3D", snapshot_index=None):
    # prints the full P4b science-diagnostics demonstration (k_perp break vs eq 4.5, alpha
    # vs eq 4.7, N-phi cross-phase vs eq 4.6/4.8) on whatever state make_data reached --
    # this file's own module docstring records the honest quality level (nonlinear but not
    # verified-saturated). Executed via `python -c "from gdi_3d_run import report; report()"`
    # after make_data has produced examples/data/gdi-3D.
    params = jr.Parameters.from_snapshot(snap_path)
    kgrid = jr.setup_kgrids(params)
    steps = sn.get_saved_steps(snap_path)
    print(f"=== run summary ({snap_path}) ===")
    print(f"eqpars: {params.eqpars}")
    print(unstable_modes_report(params, kgrid))
    print(f"\n=== energy vs t ({len(steps)} snapshots) ===")
    for i in steps:
        st = sn.load_snapshot(i, snap_path, params)
        E, Z = gdi_diag.energy_enstrophy(st, kgrid, params)
        print(f"  t={float(st.t):7.3f}  E={float(E):.4e}  Z={float(Z):.4e}")

    idx = steps[-1] if snapshot_index is None else snapshot_index
    state = sn.load_snapshot(idx, snap_path, params)
    print(f"\n=== diagnostics at snapshot {idx} (t={float(state.t):.3f}) ===")

    print("\n-- k_perp break (eq 4.5 analog), numeric root of max Re(lambda(L))=0 --")
    for kz in (0.0, 1.0, 2.0, 3.0, 4.0):
        kc = gdi_diag.kperp_break(params, kz=kz)
        print(f"  kz={kz:.1f}  k_perp_break={kc}")

    print("\n-- theoretical N/phi eigenvector cross-phase vs ky (kx=0), by kz --")
    ky_list = [0.25, 0.75, 1.25, 2.0, 2.75, 3.25, 4.0, 4.5, 5.0, 5.25, 6.0, 7.0]
    for kz in (0.0, 1.0, 2.0, 3.0):
        print(f"  kz={kz:.1f}:")
        for row in gdi_diag.theory_cross_phase(params, kz, ky_list):
            print(f"    ky={row['ky']:.2f}  growth={row['growth']:+.4f}  "
                 f"|N/phi|={row['amp_ratio']:.3f}  phase={row['phase_deg']:.1f}deg")

    print("\n-- alpha measurement (eq 4.7, note alpha=1+gpar_fac not gpar_fac) on the "
         "ACTUAL run state --")
    modes = [(kxi, kyi) for kxi in (0, 1, 2) for kyi in (1, 2, 3)]
    for ikz in (0, 1, 2):
        kz_val = float(kgrid.kz[ikz, 0, 0])
        print(f"  kz_index={ikz} (kz={kz_val:.1f}):")
        for r in gdi_diag.measure_alpha(state, kgrid, params, ikz, modes):
            print(f"    (ikx={r['ikx']},iky={r['iky']})  alpha_measured={r['alpha_measured']:.3f}"
                 f"  alpha_theory={r['alpha_theory']:.3f}  |ratio|={abs(r['ratio']):.3f}"
                 f"  arg={np.degrees(np.angle(r['ratio'])):.1f}deg")

    print("\n-- measured cross_phase_spectrum on the ACTUAL run state (cf. the theory table "
         "above) --")
    for ikz in (0, 1, 2):
        kz_val = float(kgrid.kz[ikz, 0, 0])
        kbins, amp, phase = gdi_diag.cross_phase_spectrum(state, kgrid, params, kz_index=ikz)
        print(f"  kz_index={ikz} (kz={kz_val:.1f}):")
        for k, a, p in zip(np.asarray(kbins), np.asarray(amp), np.degrees(np.asarray(phase))):
            if np.isfinite(a):
                print(f"    k={k:.2f}  amp_ratio={a:.3f}  phase={p:.1f}deg")
