# Phase 1 performance benchmark under real MPI: times the jitted block_of_steps loop and
# reports steps/sec (rank 0). Same script runs against the old (pre-Phase-1) or new
# package depending on PYTHONPATH -- see slurms/bench_phase1.sh.
# usage: bench_phase1.py <label> <case: 2d|2d_forced|3d|3d_forced> <donate|nodonate> [nx nz] [unroll] [nps]
#        Phase 2 extras: [cfl<N>] [halo_late] [nb<N> nr<N>]  -- see slurms/bench_phase2*.sh
#        Phase 3 extras: [nx<N> nz<N>] [backend=mpi4jax|jax] [halo_early] [--profile]
#                        -- see slurms/bench_phase3_*.sh
import sys, os, re, time, contextlib

# Select the package version via RMHD_PKG=<dir>, robustly: a PEP-660 editable install
# (pip install -e) registers a meta-path finder that silently beats PYTHONPATH, so we
# drop any such finder for jax_rmhd and put the requested dir first, then verify.
_pkgdir = os.environ.get("RMHD_PKG")
if _pkgdir:
    sys.meta_path = [f for f in sys.meta_path
                     if "jax_rmhd" not in (getattr(f, "__module__", "") or "")]
    sys.path.insert(0, _pkgdir)

import numpy as np, jax, jax.numpy as jnp
import jax_rmhd as jr
from jax_rmhd import run as jrun
from jax_rmhd.timestepping import get_scheme

if _pkgdir:
    assert jr.__file__.startswith(os.path.abspath(_pkgdir) + os.sep), \
        f"wrong jax_rmhd imported: {jr.__file__} (wanted {_pkgdir})"

getattr(jr, "init_cluster", lambda: None)()  # old packages (RMHD_PKG A/B) still define it
label, case, donate = sys.argv[1], sys.argv[2], sys.argv[3] == "donate"
nx = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 128
nz = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].isdigit() else 256
# Phase 3: nx<N>/nz<N> keyword forms of the positional grid args (GPU-scale grids)
for a in sys.argv:
    if (m := re.fullmatch(r"nx(\d+)", a)): nx = int(m.group(1))
    if (m := re.fullmatch(r"nz(\d+)", a)): nz = int(m.group(1))
forced = case.endswith("forced")
L = 2 * np.pi
# lsrk54 + elsasser + adaptive dt is the comm-heaviest configuration (plan F1)
common = dict(Lx=L, Ly=L, diss=(1e-4, 1e-4), hyper=2, cfl_safety=0.5,
              forcing=forced, forcing_mode="elsasser", forcing_power_elsasser=(1.0, 1.0),
              forcing_tau=1.0, fshell=(1, 3), forcing_seed=1)
# Phase 3: backend=<mpi4jax|jax> is a pass-through to Parameters(comm_backend=...); only added
# when given, so the old package (whose __init__ lacks the arg) still runs the default case.
_bk = [a.split("=", 1)[1] for a in sys.argv if a.startswith("backend=")]
if _bk:
    common["comm_backend"] = _bk[0]
if case.startswith("3d"):
    p = jr.Parameters(nx=nx, ny=nx, dims=3, nz=nz, Lz=L, z_diss=0.25, **common)
    ic = lambda x, y, z: jnp.stack([jnp.cos(x + 1.4) + jnp.cos(y + 2.0) * jnp.cos(z),
                                    jnp.cos(2 * x + 2.3) * jnp.cos(z) + 0.5 * jnp.cos(y)])
else:
    p = jr.Parameters(nx=nx, ny=nx, dims=2, **common)
    ic = lambda x, y: jnp.stack([jnp.cos(x + 1.4) + jnp.cos(y + 2.0),
                                 jnp.cos(2 * x + 2.3) + 0.5 * jnp.cos(y + 6.2)])
# set as plain attributes so the old package (whose __init__ lacks them) just ignores them
p.lsrk_scan = "unroll" not in sys.argv
p.forcing_norm_per_step = "nps" in sys.argv
# shell-restricted RNG is now opt-in (default full-grid after the Savio A/B result)
p.forcing_shell_noise = "shellrng" in sys.argv
# Phase 2: "cfl<N>" sets params.cfl_every (plain attribute, same old-package-tolerant style)
_cfl_arg = [int(m.group(1)) for a in sys.argv if (m := re.fullmatch(r"cfl(\d+)", a))]
if _cfl_arg:
    p.cfl_every = _cfl_arg[0]
# cfl blocking only engages with adaptive dt (run._use_cfl_blocks); the bench uses the default True
cfl_eff = (getattr(p, "cfl_every", 1) if p.adaptive_timestep else 1)
halo_late = "halo_late" in sys.argv
if halo_late:
    # revert T7 within the new package: rebuild the RMHD recipe without the early-halo hook
    # (in-place registry assignment, before any construct_rhs/jit trace)
    from jax_rmhd.physics import equation_registry
    _rec = equation_registry["RMHD"]
    assert hasattr(_rec, "halo_start_func"), "halo_late requires a package with T7"
    equation_registry["RMHD"] = _rec._replace(halo_start_func=None)
    p.halo_start = False   # new package also gates on this (construct_rhs._halo_start_enabled)
halo_early = "halo_early" in sys.argv
if halo_early:
    # force T7's early-halo hook on; the T7 on/off pair on GPU is halo_early vs halo_late.
    # p.halo_start overrides the new package's per-backend gate, which otherwise makes this
    # flag a silent no-op under comm_backend="mpi4jax" (measuring the baseline twice).
    from jax_rmhd.physics import equation_registry, rmhd as _rmhd
    _rec = equation_registry["RMHD"]
    assert hasattr(_rec, "halo_start_func"), "halo_early requires a package with T7"
    equation_registry["RMHD"] = _rec._replace(halo_start_func=_rmhd.halo_start)
    p.halo_start = True    # plain attribute: ignored by the old package, honored by the new
# Phase 3: per-rank device visibility -- each GPU rank must report exactly 1 distinct device
_devstr = ",".join(f"{d.platform}:{d.id}" for d in jax.local_devices())
if p.rank == 0 or jax.default_backend() != "cpu":  # per-rank on GPU, rank 0 only on 32-rank CPU jobs
    print(f"[rank {p.rank}] platform={jax.default_backend()} local_devices=[{_devstr}] "
          f"global_device_count={jax.device_count()}", flush=True)
# RMHD_REQUIRE_GPU=1: die loudly instead of timing a silently CPU-fallback rank (2080 job
# 35894622: transient cuInit NO_DEVICE put 2 of 28 ranks on CPU -> 4097 ms/step garbage).
if os.environ.get("RMHD_REQUIRE_GPU") and jax.default_backend() == "cpu":
    raise SystemExit(f"[rank {p.rank}] RMHD_REQUIRE_GPU=1 but jax fell back to CPU "
                     f"(transient cuInit failure? node issue?) -- aborting this case.")
kg = jr.setup_kgrids(p)
# A/B isolation switches (new package only), to attribute any forced-path regression:
if "sep" in sys.argv:
    # revert T4b: per-stage Pp/Pm as two separate scalar reductions/allreduces
    from jax_rmhd.physics import rmhd, shared_physics
    def _sep(fields, f_raw, kgrid, params):
        phik, psik = fields[0], fields[1]
        Pp = shared_physics.perp_inner_product(phik + psik, f_raw[0], kgrid, params)
        Pm = shared_physics.perp_inner_product(phik - psik, f_raw[1], kgrid, params)
        eps_p, eps_m = params.forcing_power_elsasser
        return jnp.stack([shared_physics.safe_scale(eps_p, Pp, params.forcing_scale_max),
                          shared_physics.safe_scale(eps_m, Pm, params.forcing_scale_max)])
    rmhd._forcing_scale_from = _sep
if "fullrng" in sys.argv and hasattr(kg, "fidx_x"):
    # revert T4a: drop the shell index set so ou_update falls back to the full-grid draw
    kg = kg._replace(fidx_x=None, fidx_y=None)
state = jrun.initialize(ic, p)
stepper, scheme = get_scheme("lsrk54")
kwargs = dict(static_argnums=(2, 3, 4, 5))
if donate:
    kwargs["donate_argnums"] = (0,)  # new run.py's production jit config; nodonate = old's
nblock, nrep = 25, 4
# optional overrides: "nb<N>"/"nr<N>" (keep nblock a multiple of every cfl_every compared)
for a in sys.argv:
    if (m := re.fullmatch(r"nb(\d+)", a)): nblock = int(m.group(1))
    if (m := re.fullmatch(r"nr(\d+)", a)): nrep = int(m.group(1))
# Phase 3: the "jax" backend's ppermute/psum/pmax are only valid inside a shard_map, so the
# stepper is wrapped in the z-mesh context here exactly as run.simulate_scan does; the
# mpi4jax path keeps the historical static-argnums jit verbatim.
if getattr(p, "comm_backend", "mpi4jax") == "jax":
    from jax_rmhd import comms
    _adv = jax.jit(comms.shard_call(
        lambda s, kgr: jrun.block_of_steps(s, kgr, p, nblock, scheme, stepper)[0], p, kg),
        **({"donate_argnums": (0,)} if donate else {}))
    step = lambda s: _adv(s, kg)
else:
    step_jit = jax.jit(jrun.block_of_steps, **kwargs)
    step = lambda s: step_jit(s, kg, p, nblock, scheme, stepper)[0]
def barrier():
    # real mpi4py only; the local test stub's fake comm has no Barrier (single rank anyway)
    if p.size > 1:
        p.comm.Barrier()

state = step(state)  # compile + warm block
jax.block_until_ready(state.fields)
barrier()
# --profile: trace the timed loop (rank 0 only unless RMHD_PROFILE_ALL=1) into RMHD_PROFILE_DIR
profile = "--profile" in sys.argv
_prof_me = profile and (p.rank == 0 or os.environ.get("RMHD_PROFILE_ALL"))
_profdir = os.path.join(os.environ.get("RMHD_PROFILE_DIR", f"prof_{label}"), f"rank{p.rank}")
with (jax.profiler.trace(_profdir) if _prof_me else contextlib.nullcontext()):
    t0 = time.perf_counter()
    for _ in range(nrep):
        state = step(state)
    jax.block_until_ready(state.fields)
    barrier()  # keep the straggler wait inside the timed region, as before
    dt = time.perf_counter() - t0
if p.rank == 0:
    # block_of_steps rounds nblock UP to whole cfl_every blocks -- count the steps actually run
    n = nrep * (-(-nblock // cfl_eff) * cfl_eff)
    tags = ("scan" if p.lsrk_scan else "unroll") + ("+nps" if p.forcing_norm_per_step else "") \
           + ("+sep" if "sep" in sys.argv else "") + ("+fullrng" if "fullrng" in sys.argv else "") \
           + ("+shellrng" if "shellrng" in sys.argv else "") \
           + f"+cfl{cfl_eff}" + ("+halo_late" if halo_late else "") \
           + ("+halo_early" if halo_early else "") + ("+prof" if profile else "")
    print(f"{label:6s} {case:10s} nx={nx} nz={nz} ranks={p.size} [{tags}] steps={n} "
          f"{n/dt:8.2f} steps/s  {dt/n*1e3:8.2f} ms/step  "
          f"backend={getattr(p, 'comm_backend', 'mpi4jax')} plat={jax.default_backend()} "
          f"dev=[{_devstr}] pkg={jr.__file__}", flush=True)
