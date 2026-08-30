#!/usr/bin/env python3
# CMHD vs z_spectral RMHD: the plans/CMHD_PLAN.md Phase C2 performance measurement.
#
# Same-session INTERLEAVED A/B (the standing measurement rule: A and B alternate within one
# process so that thermal state and machine load are shared, and the reported number is a
# median over repetitions with its spread, not a single timing). Both sides are the jitted
# `run.block_of_steps` at fixed dt with lsrk54, so the comparison is a solver step against a
# solver step with the same stepper, the same hoisting decision path and the same block
# structure. The only differences are the equation set, nfields (7 vs 2) and the propagator
# backend the recipe selects.
#
#   python bench/cmhd_perf.py                       # 128^2x16 and 256^2x16, fp64
#   python bench/cmhd_perf.py --grids 128 --nrep 9
#   python bench/cmhd_perf.py --out results.json
#
# TARANIS_PRECISION is read at taranis import, so --precision sets it before importing;
# nothing here imports taranis at module scope (bench/memory_probe.py's rule).
#
# The memory column is `jit(block_of_steps).lower(...).compile().memory_analysis()` in
# memory_probe.py's u convention: u = one field-sized complex array
# = nz_local * nkx * nky * itemsize, so CMHD's 7 fields cost 7 u of state where RMHD's cost
# 2 u, and "total_u" is (temp + args + out)/u.
#
# TAKE THE BENCH LOCK. Another agent timing on the same laptop invalidates both
# measurements; this script refuses to start if /private/tmp/taranis_bench.lock exists and
# removes its own on exit (--force overrides, --no-lock skips).
import argparse
import json
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

L = 6.283185307179586
LOCK = "/private/tmp/taranis_bench.lock"

# ny=eta, so rmhd.linear_matrix returns SeparableL: the z_spectral RMHD production
# backend, and the one the ratio below is quoted against. Print the resolved backend.
RMHD_EQPARS = {"diss": (1e-4, 1e-4), "hyper": 2, "z_diss_k": 1e-6}
CMHD_EQPARS = {"cs0": 1.0, "diss": 1e-4, "hyper": 2, "gamma": 1.0}


def _cases(n, nz, dt):
    common = dict(nx=n, ny=n, nz=nz, Lx=L, Ly=L, Lz=L, dims=3, z_spectral=True,
                  adaptive_timestep=False, dt=dt, cfl_safety=0.5, comm_backend="serial")
    return [("rmhd", dict(common, eqtype="RMHD", eqpars=RMHD_EQPARS)),
            ("cmhd", dict(common, eqtype="CMHD", eqpars=CMHD_EQPARS))]


def _ic(eqtype):
    import jax.numpy as jnp

    def rmhd_ic(x, y, z):
        return jnp.stack([jnp.cos(x)*jnp.sin(y) + 0.3*jnp.cos(2*x + z),
                          jnp.sin(x)*jnp.cos(y) + 0.3*jnp.sin(y + z)])

    def cmhd_ic(x, y, z):
        shp = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
        one = jnp.ones(shp)
        return jnp.stack([1.0 + 0.1*jnp.cos(x)*jnp.sin(y)*one,
                          0.2*jnp.sin(y)*jnp.cos(z)*one,
                          0.2*jnp.cos(x)*jnp.sin(z)*one,
                          0.1*jnp.sin(x + y)*one,
                          0.2*jnp.sin(y)*one,
                          0.2*jnp.cos(x)*one,
                          1.0 + 0.1*jnp.sin(x + z)*one])
    return cmhd_ic if eqtype == "CMHD" else rmhd_ic


def _build(kw, scheme, nblock):
    """(call, lowered, u_bytes, backend name) for one case, compiled and warm."""
    import jax
    import numpy as np
    import taranis as tr
    from taranis import _precision, run as jrun
    from taranis.grids import setup_kgrids
    from taranis.timestepping import get_scheme

    p = tr.Parameters(**kw)
    kgrid = setup_kgrids(p)
    state = jrun.initialize(_ic(p.eqtype), p)
    stepper, sch = get_scheme(scheme)
    fn = jax.jit(jrun.block_of_steps, static_argnums=(2, 3, 4, 5))
    lowered = fn.lower(state, kgrid, p, nblock, sch, stepper)
    u = p.nz * p.nx * (p.ny//2 + 1) * np.dtype(_precision.ctype).itemsize
    return (lambda: fn(state, kgrid, p, nblock, sch, stepper), lowered, int(u),
            type(kgrid.lin).__name__, p)


def main(**kwargs):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grids", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--nz", type=int, default=16)
    ap.add_argument("--nblock", type=int, default=10)
    ap.add_argument("--nrep", type=int, default=7)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--scheme", default="lsrk54")
    ap.add_argument("--precision", default="64")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="ignore an existing bench lock")
    ap.add_argument("--no-lock", action="store_true")
    args = ap.parse_args([]) if kwargs else ap.parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    if not args.no_lock:
        if os.path.exists(LOCK) and not args.force:
            with open(LOCK) as fh:
                raise SystemExit(f"bench lock held: {LOCK}\n  {fh.read().strip()}\n"
                                 f"another timing run is in progress; wait, or pass --force")
        with open(LOCK, "w") as fh:
            fh.write(f"pid {os.getpid()} bench/cmhd_perf.py started {time.ctime()}\n")

    os.environ.setdefault("TARANIS_PRECISION", str(args.precision))
    try:
        import jax
        results = []
        for n in args.grids:
            built = {}
            for label, kw in _cases(n, args.nz, args.dt):
                call, lowered, u, backend, p = _build(kw, args.scheme, args.nblock)
                m = lowered.compile().memory_analysis()
                mem = {"temp": int(m.temp_size_in_bytes),
                       "args": int(m.argument_size_in_bytes),
                       "out": int(m.output_size_in_bytes)}
                jax.block_until_ready(call())          # warm
                built[label] = dict(call=call, u=u, backend=backend, mem=mem,
                                    nfields=p.nfields,
                                    total_u=round(sum(mem.values())/u, 3), ts=[])
            # INTERLEAVED: one timed block of each per repetition, alternating, so drift in
            # machine state hits both sides equally.
            for _ in range(args.nrep):
                for label in ("rmhd", "cmhd"):
                    t0 = time.perf_counter()
                    jax.block_until_ready(built[label]["call"]())
                    built[label]["ts"].append(1e3*(time.perf_counter() - t0)/args.nblock)
            row = {"n": n, "nz": args.nz, "scheme": args.scheme, "nblock": args.nblock,
                   "nrep": args.nrep, "precision": os.environ["TARANIS_PRECISION"]}
            for label in ("rmhd", "cmhd"):
                b = built[label]
                ts = sorted(b["ts"])
                row[label] = {"ms_per_step": round(statistics.median(ts), 4),
                              "min": round(ts[0], 4), "max": round(ts[-1], 4),
                              "spread_pct": round(100*(ts[-1] - ts[0])/statistics.median(ts), 1),
                              "backend": b["backend"], "nfields": b["nfields"],
                              "u_bytes": b["u"], "mem": b["mem"], "total_u": b["total_u"]}
            row["ratio_cmhd_over_rmhd"] = round(row["cmhd"]["ms_per_step"]
                                                / row["rmhd"]["ms_per_step"], 3)
            results.append(row)
            print(f"{n}^2 x {args.nz}  {args.scheme}  fp{row['precision']}")
            for label in ("rmhd", "cmhd"):
                r = row[label]
                print(f"  {label:5s} {r['backend']:18s} nfields={r['nfields']}  "
                      f"{r['ms_per_step']:8.3f} ms/step  (min {r['min']:.3f}, max "
                      f"{r['max']:.3f}, spread {r['spread_pct']:.1f}%)  "
                      f"mem {r['total_u']:.2f} u  (u = {r['u_bytes']} B)")
            print(f"  ratio CMHD/RMHD = {row['ratio_cmhd_over_rmhd']:.3f}x", flush=True)
        if args.out:
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=2)
            print(f"wrote {args.out}")
        return results
    finally:
        if not args.no_lock and os.path.exists(LOCK):
            os.remove(LOCK)


if __name__ == "__main__":
    main()
