# Retuned 2D GDI production run (plans/GDI_PLAN.md P4a review-round-1 MAJOR 2 fix; weak-
# drive retune superseding the original strong-drive numbers below -- see plans/GDI_PLAN.md
# Status).
#
# examples/gdi-2D.ipynb no longer bundles a pre-generated, gitignored snapshot directory
# that no cell produces (that broke on fresh clones). Instead this module holds the run
# configuration and a resumable, idempotent `make_data`: call it repeatedly (each call
# wall-clock-bounded) from the notebook -- or standalone via
# `python -c "from gdi_2d_run import make_data; make_data('data/gdi-2D', t_end=..., ...)"`
# -- and it picks up wherever the on-disk snapshots left off, via the same
# params.save/get_saved_steps/load_snapshot contract as every other example
# (restart-workflow.ipynb; CLAUDE.md's checkpointing invariants).
#
# Parameter choice (box + eqpars): the original 64^2, Lx=Ly=2*pi setup has exactly ONE
# unstable lattice mode ((kx,ky)=(0,1)) at its eqpars, so it is a single coherent "finger"
# forever, never a broadband cascade. Retuned for a DENSE k-lattice (larger box, kept at
# Lx=Ly=8*pi/256^2) so many discrete modes fall inside the unstable cone around kx=0
# (marginal at kc=sqrt(v0/(gpar_fac*nu_in*Ln))).
#
# WEAK-DRIVE RETUNE, then GAMMA-MAX CALIBRATION (two rounds; both superseded by the current
# EQPARS below -- kept here for the record, see plans/GDI_PLAN.md Status):
#
# Round 1 (weak-drive, Ln=84/v0=25/nu_in=0.05) superseded an earlier strong-drive choice
# (Ln=1, nu_in=0.3, v0=2, diss=2e-5) that saturated at delta n/n0 >> 1 (N_rms~10-17,
# max|N|~21-37), well outside the model's perturbative ordering, and never stopped
# drifting. It used a mixing-length estimate (max delta n/n ~ L_box/Ln) to pick Ln, and
# undershot the target: measured saturation was max|N|~1.3-1.8 (delta n/n restored to O(1)
# but still 2-6x the target 0.3).
#
# Round 2 (this file's current EQPARS) replaces the mixing-length-in-Ln estimate with a
# direct empirical calibration: at fixed kc=sqrt(v0/(gpar_fac*nu_in*Ln)), the SATURATION
# AMPLITUDE is controlled by gamma_max=sqrt(nu_in*v0/Ln), not by Ln alone -- the Ln-gradient
# drive term is negligible in this Pedersen-dominated family (see the energy-budget cell:
# drive_Ln << drive_pedersen), so changing Ln at fixed gamma_max/kc barely moves the
# saturated state, and the mixing-length-in-Ln picture that motivated round 1 doesn't apply
# here. Three probes at fixed kc~2.44 (256^2, Lx=Ly=8*pi, gpar_fac=1, diss=5e-6, hyper=2,
# seed 1e-3) measured late-time max|N| vs gamma_max:
#   gamma_max=0.134 (Ln=84,   nu_in=0.05,   v0=25): max|N|~1.6,  still drifting at t=170
#   gamma_max=0.060 (Ln=171,  nu_in=0.0246, v0=25): max|N|~0.83, still creeping at t~240
#   gamma_max=0.017 (Ln=605,  nu_in=0.007,  v0=25): max|N|~0.18, N_rms~0.06, statistically
#                                                     steady t~350-600
# A local power-law interpolation (exponent ~1.2, max|N| vs gamma_max) through these three
# points, holding kc fixed (nu_in=gamma_max/kc, v0/Ln=gamma_max^2/nu_in, v0=25 unchanged),
# gives the target point below: Ln=392.0, nu_in=0.0106, v0=25.0 -- gamma_max~0.026,
# kc~2.45 (~unchanged from round 1's ~2.44: nu_in*Ln=4.16 vs 4.2 before), dealias cutoff
# k~21.3 (unchanged -- box/grid unchanged), linear-operator dt ceiling
# 0.5/(nu_in*kmax_dealias^2)~0.10 (~4.7x looser than round 1's ~0.02, ~28x looser than the
# original strong-drive family's ~0.0037). diss/hyper are left unchanged (5e-6, hyper=2):
# still negligible at kc (diss*kc^4/gamma_max ~ 0.7%) but, because gamma_max is now much
# smaller, hyperdissipation now overtakes it earlier in k (diss*k^4 crosses gamma_max around
# k~8.5, vs ~13 in round 1) -- still well inside the dealiased range, so it arrests the
# cascade without touching the linear instability. Linear growth to saturation is
# correspondingly slower (e-folding time 1/gamma_max~38, vs ~7.5 in round 1), so t_end and
# the averaging window are both pushed out accordingly (see make_data's default below).
import time

import numpy as np
import jax.numpy as jnp

import jax_rmhd as jr
import jax_rmhd.snapshot_io as sn
from jax_rmhd.run import block_of_steps
from jax_rmhd.timestepping import get_scheme

NX = NY = 256
LX = LY = 8.0 * np.pi
EQPARS = dict(Ln=392.0, nu_in=0.0106, v0=25.0, gpar_fac=1.0, diss=5e-6, hyper=2)


def make_params():
    # one place for the run configuration -- identical every call (params.save's
    # identical-record-is-a-no-op guard is what makes make_data's resume path safe).
    return jr.Parameters(nx=NX, ny=NY, Lx=LX, Ly=LY, cfl_safety=0.4, dims=2,
                         eqtype="GDI", eqpars=EQPARS, adaptive_timestep=True, cfl_every=1)


def ic(x, y):
    # small-amplitude (1e-3, kept fixed across the retune -- a clean, slow linear phase is
    # now a diagnostic feature, see the notebook's eigenvector-vs-time cell), multi-mode,
    # random-phase perturbation in both N and phi, covering kx,ky out to ~6.75 (beyond the
    # unstable band's measured edge ~5.0 with margin) so every lattice mode the linear
    # operator favors has some seed to grow from -- no structure is hand-picked.
    rng = np.random.default_rng(1)
    N = jnp.zeros_like(x * y)
    phi = jnp.zeros_like(x * y)
    for kxi in range(0, 28):
        for kyi in range(0, 28):
            if kxi == 0 and kyi == 0:
                continue
            aN, aP, ph1, ph2 = rng.normal(size=4) * 1e-3
            kx = kxi * 2 * np.pi / LX
            ky = kyi * 2 * np.pi / LY
            N = N + aN * jnp.cos(kx * x + ky * y + ph1)
            phi = phi + aP * jnp.cos(kx * x + ky * y + ph2)
    return jnp.stack([N, phi])


def make_data(snap_path, t_end=550.0, snap_every=5.0, wall_budget=33.0, nblock=200,
             schemestr="lsrk54", nsnap=250):
    # Idempotent + resumable: if the last snapshot already reached t_end, return
    # immediately (True); otherwise resume from it (or start fresh) and advance in `nblock`
    # chunks, checking wall-clock time between chunks, saving a snapshot roughly every
    # `snap_every` time units, and returning (not raising) once `wall_budget` seconds have
    # elapsed -- so both the sandbox (chunked bash calls) and a laptop (a plain loop) can
    # call this repeatedly until it reports done. `simulate`/`simulate_scan`'s overshoot
    # caveat applies here too (up to `nblock` steps past t_end/snap_every) -- never assert
    # exact times/counts against this run's output.
    params = make_params()
    params.save(snap_path)  # first call: writes params.json; later calls: no-op (identical)
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
    # flush any progress since the last snapshot boundary too (not just on snap_every/t_end):
    # a wall-clock-bounded call would otherwise discard up to `nblock` steps of work every
    # time it is interrupted between snapshot boundaries, which adds up over many chunks.
    if float(state.t) > t_last_snap:
        sn.save_snapshot(next_snap, state, mngr, params)
        next_snap += 1
        t_last_snap = float(state.t)
    mngr.wait_until_finished()

    done = float(state.t) >= t_end
    print(f"make_data: reached t={float(state.t):.3f} (target {t_end}) in "
         f"{time.time()-t_start_wall:.1f}s wall this call; done={done}")
    return done


def make_gif(snap_path, out_path=None, duration_ms=180, hold_ms=1500, max_frames=60):
    # renders N and laplacian(phi) side-by-side frames (matplotlib Agg, headless -- no
    # fftshift needed since imshow gets an explicit extent) for every saved snapshot (evenly
    # subsampled down to `max_frames` if more snapshots than that exist on disk -- always
    # keeping the very last one, so the saturated end state is never dropped) and
    # assembles them into an animated GIF via PIL. Idempotent: if out_path already exists
    # and is newer than the last snapshot on disk, does nothing (so repeated notebook runs
    # / sandbox chunks don't re-render). Default out_path lives INSIDE snap_path, so it is
    # gitignored along with the rest of the snapshot directory.
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import jax.numpy.fft as ft

    steps = sn.get_saved_steps(snap_path)
    if not steps:
        raise RuntimeError(f"make_gif: no snapshots found in {snap_path}")
    if out_path is None:
        out_path = os.path.join(snap_path, "gdi-2D-evolution.gif")
    last_snap_mtime = os.path.getmtime(os.path.join(snap_path, str(steps[-1])))
    if os.path.exists(out_path) and os.path.getmtime(out_path) >= last_snap_mtime:
        print(f"make_gif: {out_path} already up to date ({len(steps)} snapshots), skipping")
        return out_path
    if max_frames is not None and len(steps) > max_frames:
        idx = np.unique(np.linspace(0, len(steps) - 1, max_frames).round().astype(int))
        steps = [steps[j] for j in idx]

    params = jr.Parameters.from_snapshot(snap_path)
    kgrid = jr.setup_kgrids(params)
    Lx, Ly = float(params.Lx), float(params.Ly)

    frames = []
    for i in steps:
        st = sn.load_snapshot(i, snap_path, params)
        N_real = np.asarray(ft.irfft2(st.fields[0, 0]))
        lap_phi = np.asarray(ft.irfft2(-kgrid.ksq * st.fields[1, 0]))
        vmaxN = float(np.abs(N_real).max()) or 1.0
        vmaxL = float(np.abs(lap_phi).max()) or 1.0

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), dpi=90)
        for ax, field, vmax, title in ((axes[0], N_real, vmaxN, "N"),
                                       (axes[1], lap_phi, vmaxL, r"$\nabla_\perp^2\phi$")):
            im = ax.imshow(field.T, origin="lower", extent=[0, Lx, 0, Ly],
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{title} (max {vmax:.2e})", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"2D GDI, t = {float(st.t):.2f}")
        plt.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf).convert("RGB"))
        plt.close(fig)

    durations = [duration_ms] * (len(frames) - 1) + [hold_ms]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)
    print(f"make_gif: wrote {out_path} ({len(frames)} frames, {frames[0].size})")
    return out_path
