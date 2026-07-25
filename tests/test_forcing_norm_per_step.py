# Coverage for the production default forcing_norm_per_step=True (per-step normalization)
# and the opt-in forcing_shell_noise=True path — neither is exercised by test_forcing_smoke,
# which pins norm_per_step=False to check exact per-stage normalization.
# Plain assert-and-print PASS/FAIL, runs single-process (2D) like the other test scripts.
import os
os.environ.setdefault("RMHD_PRECISION", "64")
import numpy as np
import jax
import jax.numpy as jnp
import jax_rmhd as jr
from jax_rmhd.physics import shared_physics

all_ok = True
def check(msg, ok):
    global all_ok
    print(("[PASS] " if ok else "[FAIL] ") + msg)
    all_ok &= ok
    return ok

def make_params(mode, shell_noise):
    return jr.Parameters(nx=32, ny=32, Lx=2*np.pi, Ly=2*np.pi, diss=(0.0, 0.0), hyper=1,
                         cfl_safety=0.5, dt=0.01, adaptive_timestep=False, dims=2,
                         forcing=True, forcing_mode=mode, forcing_power=1.0,
                         forcing_power_elsasser=(1.0, 1.0), forcing_tau=0.5, fshell=(1, 5),
                         forcing_seed=1, forcing_norm_per_step=True,
                         forcing_shell_noise=shell_noise)

def zero_ic(x, y):
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape))

# --- A. injection rate with the per-step (lagged) normalization, both modes ------------
# at fixed dt=0.01 << tau the one-step lag is small: loose 2x bound on the rate
import tempfile
for mode in ("momentum", "elsasser"):
    p = make_params(mode, shell_noise=False)
    kg = jr.setup_kgrids(p)
    st = jr.initialize(zero_ic, p)
    mngr = jr.snapshot_manager_setup(p, tempfile.mkdtemp(), nsnap=2)
    end = jr.simulate(st, kg, p, t_snap=1.0, t_end=0.5, mngr=mngr, save=False)
    E_kin = 0.5 * float(shared_physics.perp_inner_product(end.fields[0], end.fields[0], kg, p))
    E_mag = 0.5 * float(shared_physics.perp_inner_product(end.fields[1], end.fields[1], kg, p))
    # momentum: forcing_power into E_kin. elsasser: eps_p into E+=<|grad z+|^2>/2 and eps_m
    # into E-, and E_tot=(E+ + E-)/2, so the total-energy target is (eps_p+eps_m)/2 = 1.
    target = 1.0
    rate = (E_kin + E_mag) / float(end.t)
    check(f"norm_per_step {mode}: energy injection rate {rate:.3f} within 2x of {target}",
          0.5 * target < rate < 2.0 * target)
    check(f"norm_per_step {mode}: forcing_scale finite, shape ({p.n_ou},)",
          end.forcing_scale.shape == (p.n_ou,) and bool(jnp.all(jnp.isfinite(end.forcing_scale))))

# --- B. shell-restricted noise: hermitian symmetry + support, elsasser -----------------
p = make_params("elsasser", shell_noise=True)
kg = jr.setup_kgrids(p)
fs = jnp.zeros((p.n_ou, 2, p.nx, p.ny//2 + 1), dtype=jnp.complex128)
key = jax.random.key(0)
for _ in range(50):
    fs, key = shared_physics.ou_update(fs, key, 0.01, p, kg)
nkx = p.nx
mirror = (-jnp.arange(nkx)) % nkx
for ky_idx in (0, -1):
    col = fs[..., ky_idx]
    herm_err = float(jnp.max(jnp.abs(col - jnp.conj(col[..., mirror]))))
    check(f"shell noise: hermitian symmetry at ky index {ky_idx} after 50 ou_updates (err {herm_err:.1e})",
          herm_err < 1e-12)
off_shell = float(jnp.max(jnp.abs(fs * (~kg.fmask)[None, None, :, :])))
check(f"shell noise: support exactly on fmask (off-shell max {off_shell:.1e})", off_shell == 0.0)
check("shell noise: nonzero forcing state produced", float(jnp.max(jnp.abs(fs))) > 0.0)

# --- C. shell noise end-to-end with norm_per_step (production-style combo) -------------
st = jr.initialize(zero_ic, p)
mngr = jr.snapshot_manager_setup(p, tempfile.mkdtemp(), nsnap=2)
end = jr.simulate(st, kg, p, t_snap=1.0, t_end=0.5, mngr=mngr, save=False)
check("shell noise + norm_per_step: run finite",
      bool(jnp.all(jnp.isfinite(end.fields))))

print("\n" + ("ALL PASS" if all_ok else "SOME CHECKS FAILED"))
