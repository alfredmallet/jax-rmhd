# Job table for tests/run_savio_suite.py.
#
# Each job runs one test FILE in script mode (pytest is never run under mpirun);
# each phase is one subprocess launch. Fields:
#   name        unique id (--only matches substrings of this)
#   tier        "cpu" (savio3, mpirun) or "gpu" (savio4_gpu, srun launchers)
#   script      repo-relative path
#   phases      list of dicts: launch ("serial" | "mpi" | "gpu_mpi4jax" | "gpu_jax"),
#               n (rank count, substituted into the launcher template), and optional
#               args (list of strings, appended), env (per-phase overrides), label.
#               Phases run in order IN A SHARED per-(job,precision) scratch cwd --
#               that is what makes the 2-then-4-rank resharding sequence and the
#               cross-backend restart chain work, and it routes every cwd-relative
#               `data/...` write into a clean dir (stale dirs trip the snapshot
#               layout guard).
#   precisions  RMHD_PRECISION sessions to run the whole phase list under
#               (default ("64",)). Files with fp64-marked tests print [SKIP] in a
#               32 session; that still counts as ALL PASS.
#   banner      pass rule is ALWAYS "exit 0" plus, when banner=True (default),
#               "ALL PASS" printed and "SOME CHECKS FAILED" absent (mpirun can mask
#               a rank's exit code).
#
# NOT here: bench/savio_scaling is a benchmark, not a test.

_S = ("64", "32")  # cheap single-process files: run both precision sessions

JOBS = [
    # ---- single-process (converted dual-mode files; soft-skip / 2D-only content)
    dict(name="infra", tier="cpu", script="tests/test_infra.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    dict(name="bracket", tier="cpu", script="tests/test_bracket.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    dict(name="dealias", tier="cpu", script="tests/test_dealias.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    dict(name="snapshot_roundtrip", tier="cpu", script="tests/test_snapshot_roundtrip.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    dict(name="params", tier="cpu", script="tests/test_params.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    # propagator algebra only (no MPI content); both sessions because the Taylor
    # branch cutoff and the tolerances are precision-dependent
    dict(name="linear_propagator", tier="cpu", script="tests/test_linear_propagator.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    # 2D GDI physics (dims=2, single-process by construction)
    dict(name="gdi_linear", tier="cpu", script="tests/test_gdi_linear.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    # spectral-z mode: size==1 by construction (the z-FFT needs the whole z domain)
    dict(name="z_spectral", tier="cpu", script="tests/test_z_spectral.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    dict(name="forcing_smoke", tier="cpu", script="tests/test_forcing_smoke.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    dict(name="forcing_norm_per_step", tier="cpu", script="tests/test_forcing_norm_per_step.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    # single-process files (2D / spoofed-rank content)
    dict(name="forcing_modes", tier="cpu", script="tests/test_forcing_modes.py",
         phases=[dict(launch="serial", n=1)], precisions=_S),
    # fp32-marked throughout: the "32" session is the one that tests anything
    # (a "64" session would be all [SKIP]).
    dict(name="precision_fp32", tier="cpu", script="tests/test_precision_fp32.py",
         phases=[dict(launch="serial", n=1)], precisions=("32",)),
    # multidev tests need >=4 XLA devices; under real MPI there is 1 CPU device per
    # process, so this file prints [SKIP] everywhere on Savio -- kept for the banner
    # (its real coverage is local fake-device pytest + the gpu tier below).
    dict(name="backend_jax_serial", tier="cpu", script="tests/test_backend_jax.py",
         phases=[dict(launch="serial", n=1)]),

    # ---- real multi-rank coverage (the point of the cluster tier)
    dict(name="z_stencils", tier="cpu", script="tests/test_z_stencils.py",
         phases=[dict(launch="mpi", n=4)], precisions=_S),
    dict(name="halo_width", tier="cpu", script="tests/test_halo_width.py",
         phases=[dict(launch="mpi", n=2), dict(launch="mpi", n=4)]),
    dict(name="energy_parseval", tier="cpu", script="tests/test_energy_parseval.py",
         phases=[dict(launch="mpi", n=2), dict(launch="mpi", n=4)]),
    # asserted convergence/decay studies (mpirun-safe: all nz divisible by 4,
    # reductions via comms.allreduce_sum / diagnostics.energy).
    # RMHD_RUNSLOW=1 runs advection's full nz=64->1024 study (fp64-marked -- prints
    # [SKIP] in the 32 session, which still counts as ALL PASS).
    dict(name="advection", tier="cpu", script="tests/test_advection.py",
         phases=[dict(launch="mpi", n=4, env={"RMHD_RUNSLOW": "1"})], precisions=_S),
    dict(name="dissipation", tier="cpu", script="tests/test_dissipation.py",
         phases=[dict(launch="mpi", n=4)], precisions=_S),
    # mpirun-safe files (nz=8 divisible by 4; per-rank asserts, collectives
    # called in lockstep; parspec tests inside diagnostics soft-skip when size>1).
    dict(name="scheme_equivalence", tier="cpu", script="tests/test_scheme_equivalence.py",
         phases=[dict(launch="mpi", n=4)], precisions=_S),
    # fp64-only content: an order sweep's fine end floors on the fp32 round-off floor
    dict(name="time_order", tier="cpu", script="tests/test_time_order.py",
         phases=[dict(launch="mpi", n=4)]),
    dict(name="cfl_every", tier="cpu", script="tests/test_cfl_every.py",
         phases=[dict(launch="mpi", n=4)], precisions=_S),
    dict(name="dims_parity", tier="cpu", script="tests/test_dims_parity.py",
         phases=[dict(launch="mpi", n=4)], precisions=_S),
    dict(name="diagnostics", tier="cpu", script="tests/test_diagnostics.py",
         phases=[dict(launch="mpi", n=4)], precisions=_S),
    # two-phase by design: save on 2 ranks, restart+reshard on 4, same snapshot dir
    # (phase detection = whether the dir exists, hence the shared clean cwd).
    dict(name="restart_resharding", tier="cpu", script="tests/test_restart_resharding.py",
         phases=[dict(launch="mpi", n=2, label="save_2ranks"),
                 dict(launch="mpi", n=4, label="restart_4ranks")]),
    # 64^3 forced-turbulence physics smoke + snapshot reload check (~10 min).
    dict(name="forced_turbulence_64cubed", tier="cpu", script="tests/forced_turbulence_64cubed.py",
         phases=[dict(launch="mpi", n=8)]),

    # ---- gpu tier: the 5-phase mpi4jax-vs-jax/NCCL backend battery
    # (slurms/run_test_suite_gpu.sh supplies the two launcher templates; the compare
    # phases run serially on the batch shell, exactly as test_backend_jax_gpu.sh did).
    dict(name="backend_gpu", tier="gpu", script="tests/test_backend_jax_mpi.py",
         phases=[
             dict(launch="gpu_mpi4jax", n=4, args=["mpi4jax", "out/mpi4jax"],
                  label="fresh_mpi4jax"),
             dict(launch="gpu_jax", n=4, args=["jax", "out/jax"],
                  label="fresh_jax"),
             dict(launch="serial", n=1, args=["--compare", "out/mpi4jax", "out/jax"],
                  label="compare_backends"),
             dict(launch="gpu_jax", n=4, args=["jax", "out/xr_jax", "out/mpi4jax"],
                  label="restart_jax_from_mpi4jax"),
             dict(launch="gpu_mpi4jax", n=4, args=["mpi4jax", "out/xr_mpi4jax", "out/jax"],
                  label="restart_mpi4jax_from_jax"),
             # looser tolerance: the two restarts start from snapshots that already
             # differ at roundoff (copied from slurms/test_backend_jax_gpu.sh)
             dict(launch="serial", n=1, args=["--compare", "out/xr_jax", "out/xr_mpi4jax"],
                  env={"RMHD_CMP_TOL": "1e-10"}, label="compare_cross_restarts"),
         ]),
]
