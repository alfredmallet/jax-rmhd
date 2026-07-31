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
4. **forced-turbulence-2D.ipynb** — Ornstein–Uhlenbeck forced 2D MHD turbulence
   (elsasser forcing), including a snapshot save/reload round-trip check.
5. **forced-turbulence-3D.ipynb** — forced 3D RMHD turbulence.
6. **postprocessing.ipynb** — loading snapshots from any run (local or copied from a
   cluster) without the solver running: slices, perpendicular spectra, and a 3D
   three-face volume rendering. Run orzag-tang-3d first to generate its input data.
7. **kaggle_forced_turbulence_256cubed.ipynb** — launcher for a 256³ forced-turbulence
   run on a Kaggle GPU instance (environment setup included; not runnable locally).

To run a 3D example under MPI instead: convert to a script and
`mpirun -n <N> python script.py` with `nz % N == 0`.
