import os
os.environ["RMHD_PRECISION"] = "64"
import jax
import jax_rmhd as jr
import jax.numpy as jnp
import jax.numpy.fft as ft
import matplotlib.pyplot as plt
from mpi4py import MPI
import mpi4jax

#parameters
nx = 32
ny = 32
Lx = 2.0 * jnp.pi
Ly = 2.0 * jnp.pi
Lz = 2.0 * jnp.pi
t = 0.0
nsnap = 100
t_snap = 10.0
t_end = 0.1
cfl_safety = 0.5 
spatial_dimensions=3
nblock=6000
snap_path="data/test_advection"

visc=0.0
res=0.0
hyper=3

is_control = (jax.process_index() == 0)

mngr=None # created in the loop below, once params exists (API now needs params)

def init_fields(x,y,z):
    phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
    psi = phi # z+ wave propagates backwards at vA=1
    return jnp.stack([phi,psi],axis=0)


# vary nz, fix dt
dt=0.001
nz_list=[64,128,256,512,1024]
if is_control:
    l1err=[]
    l2err=[]
for nz in nz_list:
    params=jr.Parameters(nx=nx,ny=ny,nz=nz,Lx=Lx,Ly=Ly,Lz=Lz,diss=(visc,res),
                         hyper=hyper,cfl_safety=cfl_safety,dt=dt,adaptive_timestep=False,dims=spatial_dimensions)
    kgrid = jr.setup_kgrids(params)
    if mngr is None:
        mngr=jr.snapshot_manager_setup(params,snap_path=snap_path,nsnap=nsnap)
    state=jr.initialize(init_fields,params)
    end_state=jr.simulate_scan(state,kgrid,params,nblock,t_snap,t_end,mngr,save=False)
    def end_fields(x,y,z):
        phi = jnp.cos(x) * jnp.cos(y) * jnp.cos(z+end_state.t)
        psi = phi # z+ wave propagates backwards at vA=1
        return jnp.stack([phi,psi],axis=0)
    end_exact=jr.initialize(end_fields,params)
    local_err1=float(jnp.sum(jnp.abs(end_state.fields-end_exact.fields)))
    local_denom1=float(jnp.sum(jnp.abs(end_exact.fields)))
    local_err2=float(jnp.sum(jnp.abs((end_state.fields-end_exact.fields))**2))
    local_denom2=float(jnp.sum(jnp.abs(end_exact.fields)**2))
    err1 = mpi4jax.allreduce(local_err1,op=MPI.SUM)
    denom1 = mpi4jax.allreduce(local_denom1,op=MPI.SUM)
    err2_sq = mpi4jax.allreduce(local_err2,op=MPI.SUM)
    denom2_sq = mpi4jax.allreduce(local_denom2,op=MPI.SUM)
    err2 = jnp.sqrt(err2_sq)
    denom2 = jnp.sqrt(denom2_sq)
    rel1 = err1 / denom1
    rel2 = err2 / denom2
    if is_control:
        r1=float(rel1)
        r2=float(rel2)
        l1err.append(float(r1))
        l2err.append(float(r2))
        print("nz: ",nz, ", L1 Relative Error: ", r1, ", L2 Relative Error: ", r2)

if is_control:
    jnp.savez(snap_path+"nz_test.npz",nz=nz_list,l1=l1err,l2=l2err)

