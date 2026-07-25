from mpi4py import MPI
import jax
from jax.tree_util import register_pytree_node_class
from .types import SimulationState

@register_pytree_node_class
class Parameters():
    #Stores all static parameters for the problem
    def __init__(self,nx,ny,Lx,Ly,diss,hyper,cfl_safety,dt=0.1,adaptive_timestep=True,dims=2,nz=1,Lz=0.0,z_diss=0.25,z_diss_hyper=2.0,z_diff_order=4,eqtype="RMHD",
                 forcing=False,forcing_mode="momentum",forcing_power=1.0,forcing_power_elsasser=(1.0,1.0),forcing_tau=1.0,fshell=(1,2),forcing_seed=0,forcing_scale_max=1.0,
                 forcing_norm_per_step=True,lsrk_scan=True,forcing_shell_noise=False):
        self.eqtype=eqtype
        self.nfields=eqtype_registry[self.eqtype]
        #perpendicular grid
        self.nx=nx
        if ny%2==1:
            print("ny should be even: setting ny=ny-1")
            ny=ny-1
        self.ny=ny
        self.Lx=Lx
        self.Ly=Ly
        self.dx=Lx/nx
        self.dy=Ly/ny
        #perpendicular dissipation
        self.diss = diss # should be a tuple of length nfields
        self.hyper=hyper
        #timestepping
        self.cfl_safety=cfl_safety
        self.dt = dt # Only used if adaptive_timestep==False
        self.adaptive_timestep = adaptive_timestep #Usually we want this to be true
        #dimensions
        self.spatial_dimensions=dims
        if dims==3:
            #z grid parameters
            self.nz = nz
            self.Lz = Lz
            self.dz = Lz/nz
            #z dissipation parameters
            self.z_diss = z_diss #This is in dimensionless units, dissipation coefficient is zdiss*(dz/2)^4
                                 #cf Pueschel et al. 2010
            self.z_diss_hyper = z_diss_hyper #currently unused, set =2
            self.z_diff_order = z_diff_order #currently unused, set =4
        else:
            self.nz=1
        #MPI
        self.comm=MPI.COMM_WORLD
        self.rank=self.comm.Get_rank()
        self.size=self.comm.Get_size()
        if self.spatial_dimensions==3:
            self.cart_comm = self.comm.Create_cart(dims=[self.size],periods=[True],reorder=False)
            self.left_neighbor, self.right_neighbor = self.cart_comm.Shift(direction=0, disp=1)
        else:
            self.cart_comm = None
            self.left_neighbor = None
            self.right_neighbor = None
            if self.size > 1 and self.rank==0:
                print("You probably should only run a 2D run on one device, since this isn't parallelized.")
        #forcing
        self.forcing = forcing
        if forcing_mode not in ("momentum","elsasser"):
            raise ValueError(f"forcing_mode must be 'momentum' or 'elsasser', got {forcing_mode!r}")
        self.forcing_mode = forcing_mode
        self.forcing_power = forcing_power
        self.forcing_power_elsasser = forcing_power_elsasser
        self.forcing_tau = forcing_tau
        self.fshell = fshell
        self.forcing_seed = forcing_seed
        self.forcing_scale_max = forcing_scale_max
        # compute the forcing power-normalization scale once per full step (reused across
        # RK sub-stages, removing per-stage allreduces) instead of exactly per stage
        self.forcing_norm_per_step = forcing_norm_per_step
        # draw OU noise only at the shell modes (scatter) instead of over the full k-grid:
        # measured faster single-device but ~5% slower on Savio CPU at 32 ranks (jax
        # scatter lowering), so default off; both paths give identical statistics.
        self.forcing_shell_noise = forcing_shell_noise
        self.n_ou = 1 if self.forcing_mode == "momentum" else 2
        #timestepping (structure): lax.scan LSRK stage loop (default; ~20% faster than the
        #unrolled loop on CPU) vs statically unrolled (lsrk_scan=False; to benchmark on GPU)
        self.lsrk_scan = lsrk_scan
    def tree_flatten(self):
        children = ()
        param_data = {k: v for k, v in self.__dict__.items()}
        return (children, param_data)
    @classmethod
    def tree_unflatten(cls,param_data,children):
        obj = cls.__new__(cls)
        obj.__dict__.update(param_data)
        return obj
        

# registry to set the # of fields we are solving for
eqtype_registry = {
    "RMHD": 2
}

def init_cluster():
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        try:
            jax.distributed.initialize()
            print("Distributed system initialized. Total devices: ",jax.device_count())
        except (ValueError, RuntimeError):
            jax.distributed.initialize(coordinator_address="localhost:8888", num_processes=1, process_id=0, local_device_ids=0)

