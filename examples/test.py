import os
os.environ["RMHD_PRECISION"] = "64"
import jax
import jax_rmhd as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt

#parameters
nx = 128
ny = 128
nz = 128
Lx = 2.0 * jnp.pi
Ly = 2.0 * jnp.pi
Lz = 2.0 * jnp.pi
t = 0.0
nsnap = 100
t_snap = 100
t_end = 0.01
cfl_safety = 0.5 
spatial_dimensions=3
snap_path="/global/scratch/users/alfredmallet/data/test/"

#we will use hyperviscosity
visc=1e-9
res=1e-9
hyper=3

mngr=jr.snapshot_manager_setup(snap_path=snap_path,nsnap=nsnap)

#prepare necessary objects for simulation
params=jr.Parameters(nx=nx,ny=ny,nz=nz,Lx=Lx,Ly=Ly,Lz=Lz,diss=(visc,res),hyper=hyper,cfl_safety=cfl_safety,dims=spatial_dimensions)
kgrid = jr.setup_kgrids(params)

def test_init(x,y,z):
    phi = jnp.zeros_like(x) + jnp.zeros_like(y) + jnp.zeros_like(z)
    psi = jnp.zeros_like(x) + jnp.zeros_like(y) + jnp.zeros_like(z)
    return jnp.stack([phi,psi],axis=0)

state=jr.initialize(test_init,params)

nblock = jr.estimate_good_nblock(state,kgrid,params,t_snap,t_end,nblock_min=1)
print("nblock estimate: "+str(nblock)) #not actually using this, since we just want to simulate for a fixed 100 steps
nblock = 100

end_state=jr.simulate_scan(state,kgrid,params,nblock,t_snap=t_snap,t_end=t_end,mngr=mngr,save=False)
