import os
os.environ["RMHD_PRECISION"] = "64"
import jax
import jax_rmhd as jr
import jax.numpy as jnp
import jax.numpy.fft as ft
#import matplotlib.pyplot as plt
import time

#parameters
nx = 256
ny = 256
nz = 1024
Lx = 2.0 * jnp.pi
Ly = 2.0 * jnp.pi
Lz = 2.0 * jnp.pi
t = 0.0
nsnap = 100
t_snap = 10.0
t_end = 0.01
cfl_safety = 0.5 
spatial_dimensions=3
nblock=100
snap_path="data"

visc=0.0
res=0.0
hyper=3

def init_fields(x,y,z):
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
    psi = phi # z+ wave propagates backwards at vA=1
    return jnp.stack([phi,psi],axis=0)


params=jr.Parameters(nx=nx,ny=ny,nz=nz,Lx=Lx,Ly=Ly,Lz=Lz,diss=(visc,res),
                     hyper=hyper,cfl_safety=cfl_safety,dims=spatial_dimensions)
mngr=jr.snapshot_manager_setup(params,snap_path=snap_path,nsnap=nsnap)
kgrid = jr.setup_kgrids(params)
state=jr.initialize(init_fields,params)

is_control = (params.rank==0)

#warmup (donates state's buffers -- donate_argnums=(0,) -- so rebuild state after)
e=jr.simulate_scan(state,kgrid,params,nblock=1,t_snap=t_snap,t_end=0.0001,mngr=mngr,save=False)
jax.block_until_ready(e)
state=jr.initialize(init_fields,params)

start_time=time.perf_counter()

end_state=jr.simulate_scan(state,kgrid,params,nblock,t_snap,t_end,mngr,save=False)
jax.block_until_ready(end_state)

if is_control:
    end_time=time.perf_counter()
    print("NCPU ",params.size, "TIME", end_time-start_time)
