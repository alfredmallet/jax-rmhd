# jax-rmhd
Code to solve nonlinear plasma models in jax.

Requires jax (tested on 0.10.0), orbax_checkpoint (tested on 0.11.37), and python (tested on 3.11.5). You can get these with pip.

Currently only the RMHD equations are implemented, but it should now be relatively easy to add new equation types (haha).

The current architecture is that the code is grid along the z-axis, and pseudospectral in x,y. This is to potentially handle non-periodic boundary conditions (future work), and for improved parallelization in z.
The fields are stored in k-space (nfields,nz,nkx,nky); the gradients are stored in real space (ngrads,nz,nx,ny). If you want to look at more details look at rmhd.py, for example.

Currently only some basic explicit solvers are implemented: classic RK4, low-storage RK3 (LSRK33, Williamson 1980), and low-storage 5-stage 4th-order RK (LSRK54, Carpenter & Kennedy 1994).

To see how to use the code, look at an example e.g. examples/orzag-tang-3D.ipynb

To add a new equation type you need to:

1. Add a new entry to equation_registry in physics/__init__.py. This is an EquationRecipe: (set_timestep, (term1,term2,...), grad, nfields, and optionally forcing_scale and halo_start functions)
2. Add a new file under physics/. This should contain everything you listed in equation_registry.
3. Import this file in physics/__init__.py

Tips:

set_timestep is supposed to encode any cfl conditions you want to satisfy.

The tuple (term1,term2,...) is functions which will be combined into the RHS $R$ of the equations $\partial_t f = R(f,t)$. You can put any terms you like, but be aware that currently the available solvers are explicit so it might be awkward if it is a stiff term (e.g. dispersive waves).

Perpendicular dissipation is handled via integrating factor, so you don't need to write a new term for that.

The grad function is supposed to calculate all (and only all!) the gradients you'll need for the other terms.

The terms and grad are then used to build the rhs of your equations.

If there is some function that should be useful for many equation sets, you can add it to physics/shared_physics.py.

Some future plans: 

- Do some more testing (tearing modes, AW collision,...)
- Add a spectral option for the z direction (the advection terms can then be shunted into an integrating factor)
- Add some implicit solver, at least for the z direction, and operator splitting capabilities
- Add different equation sets (in order of increasing complexity: gradient drift instability, compressible RMHD, KRMHD, FLR-MHD, KREHM, isothermal electron model, gyrokinetics...??) The last four models have fast waves, so likely need an implicit solver option first. The last three models are kinetic, and will need a spectral treatment of the velocity space and more parallelization.

