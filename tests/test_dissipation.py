import os
os.environ["RMHD_PRECISION"] = "64"
import jax
import jax_rmhd as jr
import jax.numpy as jnp
import jax.numpy.fft as ft
import matplotlib.pyplot as plt

#parameters
nx = 32
ny = 32
nz = 32
Lx = 2.0 * jnp.pi
Ly = 2.0 * jnp.pi
Lz = 2.0 * jnp.pi
t = 0.0
nsnap = 100
t_snap = 10.0
t_end = 0.1
cfl_safety = 0.5 
spatial_dimensions=3
nblock=100
snap_path="data/test_dissipation/"

hyper=1

mngr=None # created in the loop below, once params exists (API now needs params)

def init_fields(x,y,z):
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
    psi = phi # z+ wave propagates backwards at vA=1
    return jnp.stack([phi,psi],axis=0)


dt = 0.01
diss_list = jnp.arange(10)*0.1
fact_list = []

for diss in diss_list:
    params=jr.Parameters(nx=nx,ny=ny,nz=nz,Lx=Lx,Ly=Ly,Lz=Lz,diss=(diss,diss),
                         hyper=hyper,cfl_safety=cfl_safety,dt=dt,adaptive_timestep=False,dims=spatial_dimensions)
    kgrid = jr.setup_kgrids(params)
    if mngr is None:
        mngr=jr.snapshot_manager_setup(params,snap_path=snap_path,nsnap=nsnap)
    state=jr.initialize(init_fields,params)
    start_energy = jnp.sum(jnp.abs(state.fields)**2)/nx/ny/nz
    end_state=jr.simulate_scan(state,kgrid,params,nblock,t_snap,t_end,mngr,save=False)
    end_energy = jnp.sum(jnp.abs(end_state.fields)**2)/nx/ny/nz
    fact_list.append(end_energy/start_energy)

plt.figure(1)
plt.semilogy(diss_list,fact_list)
plt.semilogy(diss_list,jnp.exp(-4.0*jnp.array(diss_list)))
plt.xlabel(r'$\eta,\nu$')
plt.ylabel(r'$E_{end}/E_{start}$')
plt.savefig(snap_path+"test_diss.png")