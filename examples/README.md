# Examples

All notebooks use the current API and run single-process on a laptop (3D ones at modest
resolutions). Performance studies live in `../bench/`, cluster job scripts in
`../slurms/`, correctness tests in `../tests/`. Outputs are written under
`examples/data/` (gitignored).

Suggested order:

1. **orzag-tang-2D.ipynb** — pedagogical introduction (with exercises): decaying 2D
   Orszag–Tang vortex; parameters, initialization, running, snapshots, basic plots.
2. **orzag-tang-3d.ipynb** — the same problem in 3D (z-dependent initial condition).
3. **AW_advection.ipynb** — verification: linear Alfvén-wave advection along z against
   the exact solution; error vs kz for the 4th-order z finite-difference scheme and the
   z dissipation filter.
4. **tearing-mode-2D.ipynb** — the classic RMHD validation problem: tearing instability
   of a cos(x) current sheet; island growth, measured growth rate vs resistivity against
   the FKR (η^3/5) and Coppi (η^1/3) scalings.
5. **alfven-wave-collision-3D.ipynb** — counter-propagating z⁺/z⁻ Alfvén wave packets
   (the minimal nonlinear RMHD interaction): purely magnetic secondary-mode generation
   and perpendicular cascade onset, with a single-wave control run. Nonlinear companion
   to AW_advection.
6. **forced-turbulence-2D.ipynb** — Ornstein–Uhlenbeck forced 2D MHD turbulence
   (elsasser forcing), including a snapshot save/reload round-trip check.
7. **forcing-modes-2D.ipynb** — momentum vs elsasser forcing side by side, including the
   documented gotcha (2D momentum forcing from rest is pure hydro: psi stays exactly 0)
   and injection-power bookkeeping (measured dE/dt vs `forcing_power` vs dissipation).
8. **forced-turbulence-3D.ipynb** — forced 3D RMHD turbulence.
9. **turbulence-spectra-analysis.ipynb** — analysis of a developed forced-turbulence-3D
   run (generates its own 64³ data if none exists): time-averaged perpendicular spectra,
   Elsasser spectra, residual energy, parallel spectrum (`parspec`, size==1 only), and
   spectral anisotropy.
10. **restart-workflow.ipynb** — the checkpoint lifecycle end to end: `params.save`,
    restart via `Parameters.from_snapshot`/`load_snapshot` (bitwise-identical
    continuation), and restarting with overridden parameters.
11. **postprocessing.ipynb** — loading snapshots from any run (local or copied from a
    cluster) without the solver running: slices, perpendicular spectra, and a 3D
    three-face volume rendering. Run orzag-tang-3d first to generate its input data.
12. **kaggle_forced_turbulence_256cubed.ipynb** — launcher for a 256³ forced-turbulence
    run on a Kaggle GPU instance (environment setup included; not runnable locally).

To run a 3D example under MPI instead: convert to a script and
`mpirun -n <N> python script.py` with `nz % N == 0`.
