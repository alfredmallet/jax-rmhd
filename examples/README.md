# Examples

All notebooks use the current API and run single-process on a laptop (3D ones at modest
resolutions) — a plain `pip install -e .` is enough, with no MPI toolchain: with
`mpi4py`/`mpi4jax` absent, `comm_backend` auto-resolves to `"serial"` (exact size-1
semantics), so no notebook sets it. Performance studies live in `../bench/`, cluster job scripts in
`../slurms/`, correctness tests in `../tests/`. Outputs are written under
`examples/data/` (gitignored).

Suggested order:

1. **orzag-tang-2D.ipynb** — pedagogical introduction (with exercises): decaying 2D
   Orszag–Tang vortex; parameters, initialization, running, snapshots, basic plots.
2. **orzag-tang-3d.ipynb** — the same problem in 3D (z-dependent initial condition).
   - **orzag-tang-3d-spectral-z.ipynb** — the same initial condition run twice at 32³, with
     finite-difference z and with `z_spectral=True` (plans/GDI_PLAN.md P2: axis 1 of
     `state.fields` becomes `kz`, the Alfvén operator moves into the exact linear propagator).
     Energy histories, real-space vorticity, perpendicular and parallel spectra side by side;
     a `dt` vs `nz` scan showing where dropping the `1/dz` CFL term actually pays (nothing at
     `nz ≈ nx` with O(1) amplitudes, 3.7× at `nz = 8·nx`); and the `params.save` guard that
     refuses a cross-mode restart. Single-process only, like the mode itself.
3. **AW_advection.ipynb** — verification: linear Alfvén-wave advection along z against
   the exact solution; error vs kz for the 4th-order z finite-difference scheme and the
   z dissipation filter.
4. **tearing-mode-2D.ipynb** — the classic RMHD validation problem: tearing instability
   of a cos(x) current sheet in inviscid (ν=0) runs, matching the assumptions of
   FKR/Coppi theory; island growth, measured growth rate vs resistivity against the FKR
   (η^3/5) and Coppi (η^1/3) scalings, plus a perturbation-current structure-vs-time GIF
   of the featured run.
5. **tearing-growth-vs-k.ipynb** — the tearing dispersion curve: growth rate vs
   perpendicular wavenumber at fixed η=1e-3 in an *inviscid* run (ν=0, matching the
   assumptions of the exact Coppi relation), ten modes measured from one long-box
   (Ly=16π) fp64 run; analytic Δ'(k)=2κ·tan(πκ/2) (numerically verified), the exact
   Coppi dispersion relation (arbitrary Δ′, both asymptotic limits verified
   numerically), an equilibrium-decay-corrected prediction, a mode-selection
   structure-vs-time GIF, and the peak of γ(k) vs the k ~ η^(1/4) estimate. Extends
   tearing-mode-2D and is cross-checked against it at k=1/2: both notebooks are now
   inviscid, and their independent measurements of γ(k=1/2, η=1e-3) agree to ~1%.
6. **alfven-wave-collision-3D.ipynb** — counter-propagating z⁺/z⁻ Alfvén wave packets
   (the minimal nonlinear RMHD interaction): purely magnetic secondary-mode generation
   and perpendicular cascade onset, with a single-wave control run. Nonlinear companion
   to AW_advection.
7. **forced-turbulence-2D.ipynb** — Ornstein–Uhlenbeck forced 2D MHD turbulence
   (elsasser forcing), including a snapshot save/reload round-trip check.
8. **forcing-modes-2D.ipynb** — momentum vs elsasser forcing side by side, including the
   documented gotcha (2D momentum forcing from rest is pure hydro: psi stays exactly 0)
   and injection-power bookkeeping (measured dE/dt vs `forcing_power` vs dissipation).
9. **forced-turbulence-3D.ipynb** — forced 3D RMHD turbulence.
10. **turbulence-spectra-analysis.ipynb** — analysis of a developed forced-turbulence-3D
   run (generates its own 64³ data if none exists): time-averaged perpendicular spectra,
   Elsasser spectra, residual energy, parallel spectrum (`parspec`, size==1 only), and
   spectral anisotropy.
11. **restart-workflow.ipynb** — the checkpoint lifecycle end to end: `params.save`,
    restart via `Parameters.from_snapshot`/`load_snapshot` (bitwise-identical
    continuation), and restarting with overridden parameters.
12. **postprocessing.ipynb** — loading snapshots from any run (local or copied from a
    cluster) without the solver running: slices, perpendicular spectra, and a 3D
    three-face volume rendering. Run orzag-tang-3d first to generate its input data.
13. **kaggle_forced_turbulence_256cubed.ipynb** — launcher for a 256³ forced-turbulence
    run on a Kaggle GPU instance (environment setup included; not runnable locally).
14. **gdi-2D.ipynb** — the 2D Gradient-Drift-Instability equation set (`eqtype="GDI"`,
    plans/GDI_PLAN.md P4a): a box/eqpars combination chosen by computing the discrete
    growth-rate landscape directly from `physics/gdi.py`'s linear operator (many unstable
    lattice modes, not the single-mode original demo), run through linear growth into a
    **statistically confirmed saturated, broadband turbulent state** — real-space structure,
    a permanent per-mode eigenvector-vs-time check, time-averaged $k_\perp$ spectra (measured
    slopes reported honestly, not forced to a target), the N-phi cross-phase/amplitude-ratio
    and adiabaticity diagnostics (eqs 4.6-4.8) compared against the exact linear-eigenmode
    curve, the energy budget (drive vs. dissipation) in saturation, and an animated GIF of the
    evolution. Retuned twice: a first weak-drive family (`Ln=84, v0=25, nu_in=0.05`, a
    mixing-length estimate) undershot the `max delta n/n ~ 0.3` target; a second, empirical
    **gamma_max-calibration** retune (`Ln=392, v0=25, nu_in=0.0106` — saturation amplitude
    tracks `gamma_max=sqrt(nu_in*v0/Ln)` at fixed `kc`, not `Ln` alone) lands on it: measured
    `max|N| ~ 0.35-0.52` (mean ~0.41), `N_rms ~ 0.11-0.13`, well within the model's own
    perturbative ordering, superseding an earlier strong-drive choice that saturated with
    `delta n/n >> 1`. Data is generated by `examples/gdi_2d_run.py` (resumable, idempotent)
    so the notebook runs from a fresh clone with no bundled/gitignored inputs.
15. **gdi_3d_run.py** — 3D GDI demonstration driver (plans/GDI_PLAN.md P4b; `z_spectral`,
    `D_par` closure, `imexcb3e`): resumable `make_data` plus a `report()` that prints the
    static linear summary and the P4b science diagnostics (`kperp_break`, `measure_alpha`,
    kz-resolved cross-phase). No companion notebook yet, and the shipped configuration is
    NOT saturation-calibrated (documented in the file header) — a demonstration of the
    diagnostics, not a turbulence result.

16. **test-particles-2D.ipynb** — Boris-pushed charged test particles co-stepped with
    forced 2D RMHD turbulence (plans/TESTPART_PLAN.md phases A3/A3b): 4096 particles per
    ensemble, sixteen ensembles aboard each of the two design-(a) 256² hyperviscous runs
    (ν=η=1e-10, hyper=3) and four aboard each of the four design-(b) runs, differing only in
    Ω = (q/m)B₀, initial distribution, the RMHD amplitude B₀ = 1/ε they are interpreted at, or
    *which pieces of the field they are allowed to see*. Measured Xia, Perez, Chandran &
    Quataert (2013) style: production at **B₀ = 10** (δB/B₀ = 0.18, β_i = 0.01) with the
    numerical E∥ left by interpolating **E** and **B** independently (measured at 1.6e-3 of
    |E||B|) projected out per particle, heating read in the **local-B frame**, and rate
    windows that start at t₀ = 10/Ω (after the E×B pickup) and end when ⟨v⊥B²⟩ has grown 20%
    or a 400-step floor is reached, whichever is later. Three ξ-scans — Ω sweep at two forcing
    powers, a ×30 forcing-amplitude sweep, and Xia's fixed-ρ rings — give 22 (Q⊥, ξ) points,
    of which **16 are measurements over ξ = 0.124–0.39** and six are upper limits (a rate not
    2σ above zero is a limit and is held out of every fit). They **do** show the Chandran
    exponential: c₁ = 0.98, **c₂ = 0.40±0.13** (lab v⊥, local-B Q⊥; 0.46±0.09 drift-corrected;
    the amplitude sweep alone gives 0.34±0.10 drift-corrected at R²(norm)=0.86), against
    Chandran et al. 2010's 0.75/0.34 and Xia et al.'s resolution-dependent 0.20–0.44. A
    printed sensitivity table carries the systematics: the frame is worth 0.001 in c₂, the
    drift correction +0.06, the limit rule −0.08 (limits let in as data) to +0.20 (ẑ frame
    with no limit handling), and the design mix 0.31–0.58 — wider than the fit error. It does
    not *discriminate*: a 3-parameter power law Q⊥ ∝ δu^4.5 ρ^-0.45 scores R²(logQ)=0.922
    against the 2-parameter exponential's 0.876 on the same points; the limits are what favour
    the exponential (the power law overshoots the suppressed rings by 6–49×, the exponential
    by 0.4–13.5×) and they want c₂ ≥ 0.34, 0.57 and 0.78, i.e. *more* suppression than fitted.
    The A3 pass's null result (c₂=0.02±0.02) is **not** explained by the amplitude alone: this
    run's B₀=1 cohort gives c₂ = 0.33±0.22 under the same protocol. What B₀ demonstrably
    controls is the channel split — the A3 "2D forces the parallel channel to dominate"
    argument used ψ_rms/ρ where the correct ratio is ψ_rms/(B₀ρ); at B₀=1 the E_z work is
    89–95% of the total and Q∥/Q⊥ = 0.47–0.62, at B₀=10 with identical Ω, ρ, ξ and fields it
    is 3–15% and 0.018–0.025. The first pass's null c₂ is left to its measurement protocol
    (E×B pickup inside the window, no upper-limit handling) and/or its hyper=1 base
    turbulence, which this notebook cannot separate and says so. The machinery is exact where
    it claims to be: per-piece work accumulators close against ½|v|² to 8e-14 (4e-14 of the
    initial kinetic energy) over the 14 ensembles that do work, both controls conserve |v|²
    per particle to ~6e-14 over 30 time units, and `p_z = v_z − (q/m)ψ` holds to 0.5% of its
    scale for the full-∂ψ/∂t ensemble against 120% for the ideal ones. Three honest negatives:
    the projected/unprojected pair differ by only 1.2σ of the 4096-particle sampling error
    (per-particle paired init is the open item), the |j_z|-conditioned ⟨Δv_z²⟩ rises 1.6–1.8×
    across quantiles — but so does the E=0 control's, so it is the v×b⊥ rotation finding
    small-scale structure, not sheet acceleration — and the two D_μ estimators disagree by
    9–41× because the snapshot pair interval is only ~2 gyroperiods. Data is generated by
    **particles_2d_run.py** (resumable, idempotent, wall-clock-bounded `make_data`: four
    spin-ups, two design-(a) runs, four design-(b) runs and a short high-cadence tail — seven
    particle runs in all; 44 min in the generating session and 687 MiB on an M1 laptop, fp64
    throughout, and an executed copy that finds the data on disk only re-does the analysis).

To run a 3D example under MPI instead: `pip install -e ".[mpi]"`, convert the notebook to a
script and `mpirun -n <N> python script.py` with `nz % N == 0` (`comm_backend` then
auto-resolves to `"mpi4jax"`). Launching under `mpirun` *without* the extra installed is a
hard error, never a silent single-domain fallback.
