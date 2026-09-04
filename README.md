# taranis
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21987739.svg)](https://doi.org/10.5281/zenodo.21987739)

Code to solve nonlinear plasma models in jax.

Requires jax (tested on 0.10.0), orbax-checkpoint (tested on 0.11.37), and python (tested on 3.11.5). You can get these with pip:

```
pip install -e .                # laptop / no MPI toolchain: runs single-process, comm_backend="serial"
pip install -e ".[mpi]"         # generic Linux box with a working MPI toolchain (mpi4py/mpi4jax); zsh needs the quotes
```

Currently solves three sets of equations

1) reduced MHD (RMHD)
2) drift-GDI equations (for ionospheric turbulence)
3) compressible MHD (CMHD): polytropic, fully spectral, with an optional expanding-box frame

It should now be relatively easy to add new equation types (haha).

The code has options to be fully spectral, or parallelized across devices using finite-differences in the z direction. There are also Boris-pushed test particles that ride the RMHD fields without back-reacting.

## Running it

- `examples/README.md` — every notebook and run script, in a suggested reading order. They all run single-process on a laptop.
- `docs/SAVIO_CPU_SETUP.md` — building the env and running under MPI on a CPU cluster.
- `docs/SAVIO_GPU_SETUP.md` — the same for GPUs, including multi-GPU: `sbatch slurms/forced_turbulence_multigpu.sh` is a production forced-turbulence run on 4 GPUs, and the driver it launches (`examples/multigpu_forced_turbulence.py`) is the template for your own.
- `docs/RUNNING_TESTS.md` — `make test` and everything around it.
- `docs/numerics.md` (derivations and conventions), `docs/performance.md` (measured scaling and tuning), `docs/checkpointing.md` (snapshots and restarts).

Parallelism, briefly: the decomposition is along z only, so the maximum useful device count is about `nz/2` and 2D runs are single-process. Three transports, picked with `comm_backend`: `"serial"` (no MPI installed — the laptop default), `"mpi4jax"` (CPU clusters), and `"jax"` (shard_map/NCCL — what you want for more than one GPU, where it scales 4.15x from 4 to 16 GPUs against mpi4jax's 1.72x).

Currently various basic explicit schemes are implemented: classic RK4, low-storage RK3 (LSRK33, Williamson 1980), low-storage 5-stage 4th-order RK (LSRK54, Carpenter & Kennedy 1994), and four related IMEX RK schemes (Cavaglieri & Bewley 2015).

To see how to use the code, there are some example notebooks.

To add a new equation type you need to:

1. Add a new entry to equation_registry in physics/__init__.py. This is an EquationRecipe: (set_timestep_func, term_funcs, grad_func, nfields, forcing_scale_func, halo_start_func, linear_matrix_func)
2. Add a new file under physics/. This should contain code to calculate everything you need in equation_registry
3. Import this file in physics/__init__.py

Tips:

set_timestep is supposed to encode any cfl conditions you want to satisfy.

The tuple term_funcs=(term1,term2,...) is functions which will be combined into the RHS $R$ of the equations $\partial_t f = R(f,t)$. You can put any terms you like, but all these terms will be solved explicitly.

linear_matrix defines the set of linear terms to be solved either with integrating factor or implicitly: e.g. in RMHD with z_spectral=True, this is dissipation and Alfvénic advection, while with z_spectral=False, this is just the perpendicular dissipation.

The grad function is supposed to calculate all (and only all!) the gradients you'll need for the other terms.

The terms and grad are then used to build the rhs of your equations.

If there is some function that should be useful for many equation sets, you can add it to physics/shared_physics.py.

Finally, webgpu contains a browser-based RMHD solver using the same algorithms as the main taranis code. This is for demonstration/educational/intuition-building purposes; resolution is limited so it can't really be used for production.

Some future plans: 

- Add different equation sets (in order of increasing complexity: compressible RMHD, KRMHD, FLR-MHD, KREHM, isothermal electron model, gyrokinetics...??) The last three models are kinetic, and will need a spectral treatment of the velocity space and more parallelization.
- Parallelize implicit solves: this is fiddly.

