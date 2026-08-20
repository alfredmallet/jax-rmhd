#!/usr/bin/env python3
# Memory and step-time probe: the MEMORY_PERF_PLAN G0 driver.
#
# Reports, per case: XLA compiled memory_analysis (temp/args/out, bytes and u), device
# peak_bytes_in_use across a timed block (GPU only), and median ms/step. An OOM is a
# result row, not a crash.
#
# u = one field-sized complex array = nz_local*nkx*nky*itemsize.
#
#   python bench/memory_probe.py --profile laptop
#   python bench/memory_probe.py --profile p100 --precision 64 --out results.json
#   python bench/memory_probe.py --list
#
# main(**kwargs) takes the same options, for the lugus kernel entry point.
#
# TARANIS_PRECISION is read at taranis import, so --precision sets it before importing;
# nothing here may import taranis at module scope.
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

L = 6.283185307179586

RMHD_EQPARS = {"diss": (1e-4, 1e-4), "hyper": 2}
RMHD_EQPARS_ZS = {"diss": (1e-4, 1e-4), "hyper": 2, "z_diss_k": 1e-6}
RMHD_EQPARS_UNEQUAL = {"diss": (1e-4, 2e-4), "hyper": 2, "z_diss_k": 1e-6}
GDI_EQPARS_2D = {"Ln": 392.0, "nu_in": 0.0106, "v0": 25.0, "gpar_fac": 1.0,
                 "diss": 5e-6, "hyper": 2}
GDI_EQPARS_3D = {"Ln": 3.0, "nu_in": 0.2, "v0": 2.0, "D_par": 0.05, "gpar_fac": 0.0,
                 "diss": 1e-3, "hyper": 2}

# ------------------------------------------------------------------ case construction

def _case(label, scheme, nblock=10, nrep=5, **params):
    params.setdefault("Lx", L)
    params.setdefault("Ly", L)
    params.setdefault("cfl_safety", 0.5)
    params.setdefault("adaptive_timestep", False)
    params.setdefault("dt", 1e-3)
    params.setdefault("comm_backend", "serial")
    if params.get("dims", 2) == 3:
        params.setdefault("Lz", L)
    return {"label": label, "scheme": scheme, "nblock": nblock, "nrep": nrep,
            "params": params}


def _rmhd(label, nx, nz, scheme, z_spectral, scan=True, hoist=True, eqpars=None,
          adaptive=False, cfl_every=1, **extra):
    eqpars = eqpars or (RMHD_EQPARS_ZS if z_spectral else RMHD_EQPARS)
    return _case(label, scheme, nx=nx, ny=nx, nz=nz, dims=3 if nz > 1 else 2,
                 eqtype="RMHD", eqpars=eqpars, z_spectral=z_spectral, lsrk_scan=scan,
                 hoist_propagator=hoist, adaptive_timestep=adaptive,
                 cfl_every=cfl_every, **extra)


def _gdi(label, nx, nz, scheme, scan=True, **extra):
    if nz > 1:
        return _case(label, scheme, nx=nx, ny=nx, nz=nz, dims=3, eqtype="GDI",
                     eqpars=GDI_EQPARS_3D, z_spectral=True, lsrk_scan=scan, **extra)
    return _case(label, scheme, nx=nx, ny=nx, dims=2, eqtype="GDI",
                 eqpars=GDI_EQPARS_2D, lsrk_scan=scan, **extra)


def profile_laptop(precision):
    # the plan's section-1 regression matrix, small enough to run in seconds per case
    nx, nz = 64, 16
    cases = []
    for s in ("lsrk33", "lsrk54", "imexcb2", "imexcb3e", "imexcb3c", "imexcb3f"):
        cases.append(_rmhd(f"rmhd_fdz_{nx}x{nz}_{s}", nx, nz, s, False))
    for s in ("lsrk33", "lsrk54"):
        for h in (True, False):
            cases.append(_rmhd(f"rmhd_zspec_{nx}x{nz}_{s}_hoist{int(h)}", nx, nz, s,
                               True, hoist=h))
    cases.append(_rmhd(f"rmhd_zspec_{nx}x{nz}_imexcb3e", nx, nz, "imexcb3e", True))
    for s in ("lsrk33", "lsrk54"):
        for h in (True, False):
            cases.append(_rmhd(f"rmhd_zspec_{nx}x{nz}_{s}_unroll_hoist{int(h)}", nx, nz, s,
                               True, scan=False, hoist=h))
    for s in ("lsrk33", "lsrk54", "imexcb3e"):
        cases.append(_rmhd(f"rmhd_fdz_{nx}x{nz}_{s}_unroll", nx, nz, s, False, scan=False))
    # nu != eta keeps the putzer2 path under measurement after Z1 lands
    cases.append(_rmhd(f"rmhd_zspec_{nx}x{nz}_lsrk33_uneq", nx, nz, "lsrk33", True,
                       eqpars=RMHD_EQPARS_UNEQUAL))
    for s in ("lsrk33", "lsrk54", "imexcb2", "imexcb3e", "imexcb3c"):
        cases.append(_gdi(f"gdi2d_256_{s}", 256, 1, s))
    for s in ("lsrk33", "imexcb2", "imexcb3e", "imexcb3c"):
        cases.append(_gdi(f"gdi3d_{nx}x{nz}_{s}", nx, nz, s))
    return cases


def profile_laptop128(precision):
    # 128^2 x 32: the grid the F1/F2/F3 memory gates are quoted on
    nx, nz = 128, 32
    cases = []
    for s in ("lsrk33", "lsrk54", "imexcb3e"):
        cases.append(_rmhd(f"rmhd_fdz_{nx}x{nz}_{s}", nx, nz, s, False, nblock=5, nrep=3))
    for s in ("lsrk33", "lsrk54"):
        for h in (True, False):
            cases.append(_rmhd(f"rmhd_zspec_{nx}x{nz}_{s}_hoist{int(h)}", nx, nz, s, True,
                               hoist=h, nblock=5, nrep=3))
    cases.append(_rmhd(f"rmhd_zspec_{nx}x{nz}_imexcb3e", nx, nz, "imexcb3e", True,
                       nblock=5, nrep=3))
    return cases


def _gpu_cases(nx, nz, nx2d, nx_gdi3d, nz_gdi3d, precision):
    # shared shape of the p100 / gtx2080 / a100 profiles; sizes differ per card
    if precision == "64":
        nx, nz, nx2d = nx // 2, nz // 2, nx2d // 2
    cases = []
    for s in ("lsrk33", "lsrk54", "imexcb3e"):
        cases.append(_rmhd(f"rmhd_fdz_{nx}x{nz}_{s}", nx, nz, s, False))
    cases.append(_rmhd(f"rmhd_fdz_{nx}x{nz}_lsrk54_unroll", nx, nz, "lsrk54", False,
                       scan=False))
    for s in ("lsrk33", "lsrk54"):
        for h in (True, False):
            cases.append(_rmhd(f"rmhd_zspec_{nx}x{nz}_{s}_hoist{int(h)}", nx, nz, s,
                               True, hoist=h))
    cases.append(_rmhd(f"rmhd_zspec_{nx}x{nz}_imexcb3e", nx, nz, "imexcb3e", True))
    # adaptive cfl_every=1: nothing frozen, the path hoisting cannot help (Z2's lever)
    cases.append(_rmhd(f"rmhd_zspec_{nx}x{nz}_lsrk33_adaptive", nx, nz, "lsrk33", True,
                       adaptive=True, cfl_every=1))
    cases.append(_rmhd(f"rmhd_zspec_{nx}x{nz}_lsrk33_uneq", nx, nz, "lsrk33", True,
                       eqpars=RMHD_EQPARS_UNEQUAL))
    cases.append(_gdi(f"gdi2d_{nx2d}_lsrk33", nx2d, 1, "lsrk33"))
    cases.append(_gdi(f"gdi2d_{nx2d}_imexcb3e", nx2d, 1, "imexcb3e"))
    cases.append(_gdi(f"gdi3d_{nx_gdi3d}x{nz_gdi3d}_imexcb3e", nx_gdi3d, nz_gdi3d,
                      "imexcb3e"))
    return cases


def profile_p100(precision):        # 16 GB
    return _gpu_cases(512, 128, 1024, 256, 64, precision)


def profile_gtx2080(precision):     # 11 GB
    cases = _gpu_cases(512, 128, 1024, 256, 64, precision)
    if precision != "64":
        # the row expected to OOM today and to fit after Z1; keep it AND a smaller twin
        cases.append(_rmhd("rmhd_zspec_512x64_lsrk54_hoist1", 512, 64, "lsrk54", True))
    return cases


def profile_a100(precision):        # 40/80 GB
    return _gpu_cases(1024, 128, 2048, 512, 64, precision)


def profile_gtx2080_sharded(precision):
    # 4-GPU comm_backend="jax" FD-z row; per-GPU memory is what matters. nz is GLOBAL.
    return [_case("rmhd_fdz_512x128_lsrk54_jax4", "lsrk54", nx=512, ny=512, nz=128,
                  dims=3, eqtype="RMHD", eqpars=RMHD_EQPARS, z_spectral=False,
                  comm_backend="jax")]


PROFILES = {
    "laptop": profile_laptop,
    "laptop128": profile_laptop128,
    "p100": profile_p100,
    "gtx2080": profile_gtx2080,
    "gtx2080-sharded": profile_gtx2080_sharded,
    "a100": profile_a100,
}

# ------------------------------------------------------------------ measurement

def _is_oom(exc):
    s = f"{type(exc).__name__}: {exc}"
    return any(k in s for k in ("RESOURCE_EXHAUSTED", "out of memory", "Out of memory",
                                "OutOfMemory", "bad_alloc"))


def _ic_3d(jnp):
    def ic(x, y, z):
        return jnp.stack([jnp.cos(x + 1.4) + jnp.cos(y + 2.0) * jnp.cos(z),
                          jnp.cos(2 * x + 2.3) * jnp.cos(z) + 0.5 * jnp.cos(y)])
    return ic


def _ic_2d(jnp):
    def ic(x, y):
        return jnp.stack([jnp.cos(x + 1.4) + jnp.cos(y + 2.0),
                          jnp.cos(2 * x + 2.3) + 0.5 * jnp.cos(y + 6.2)])
    return ic


def _device_peak():
    import jax
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return {k: stats.get(k) for k in ("bytes_in_use", "peak_bytes_in_use",
                                      "bytes_limit") if k in stats}


def run_case(case, grad_chunk=None):
    """Measure one case in the CURRENT process. Returns the result dict."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import taranis as tr
    from taranis import comms, run as jrun, _precision
    from taranis.grids import setup_kgrids
    from taranis.timestepping import get_scheme

    out = {"label": case["label"], "scheme": case["scheme"], "params": case["params"],
           "nblock": case["nblock"], "nrep": case["nrep"], "oom": False, "error": None}

    if grad_chunk is not None:
        from taranis.physics import shared_physics
        if hasattr(shared_physics, "GRAD_CHUNK"):
            shared_physics.GRAD_CHUNK = int(grad_chunk)
            out["grad_chunk"] = int(grad_chunk)
        else:
            out["grad_chunk"] = None
            out["grad_chunk_note"] = "shared_physics.GRAD_CHUNK absent (pre-F1 tree)"

    try:
        p = tr.Parameters(**case["params"])
        # memory_analysis reports PER-DEVICE buffers, so u must be the per-device field
        # array. Under comm_backend="jax" the z mesh is every device jax can see (global
        # under jax.distributed), which is p.size only when it is one device per rank —
        # not so with a fake-device mesh. Everywhere else the process owns one device.
        ndev = jax.device_count() if p.comm_backend == "jax" else 1
        nz_dev = (p.nz // max(p.size, ndev)) if p.spatial_dimensions == 3 else 1
        u = nz_dev * p.nx * (p.ny // 2 + 1) * np.dtype(_precision.ctype).itemsize
        out["u_bytes"] = int(u)
        out["nz_per_device"] = int(nz_dev)
        out["devices_in_mesh"] = int(ndev)
        out["per_device"] = p.comm_backend == "jax"

        kgrid = setup_kgrids(p)
        state = jrun.initialize(_ic_3d(jnp) if p.spatial_dimensions == 3 else _ic_2d(jnp), p)
        stepper, scheme = get_scheme(case["scheme"])
        nblock = case["nblock"]

        if p.comm_backend == "jax":
            fn = jax.jit(comms.shard_call(
                lambda s, kg: jrun.block_of_steps(s, kg, p, nblock, scheme, stepper),
                p, kgrid))
            lowered = fn.lower(state, kgrid)
            call = lambda s: fn(s, kgrid)
        else:
            fn = jax.jit(jrun.block_of_steps, static_argnums=(2, 3, 4, 5))
            lowered = fn.lower(state, kgrid, p, nblock, scheme, stepper)
            call = lambda s: fn(s, kgrid, p, nblock, scheme, stepper)

        before = _device_peak()
        compiled = lowered.compile()
        m = compiled.memory_analysis()
        out["mem_analysis"] = {
            "temp": int(m.temp_size_in_bytes),
            "args": int(m.argument_size_in_bytes),
            "out": int(m.output_size_in_bytes),
            "code": int(getattr(m, "generated_code_size_in_bytes", 0) or 0),
        }
        total = sum(out["mem_analysis"][k] for k in ("temp", "args", "out"))
        out["total_bytes"] = int(total)
        out["total_u"] = round(total / u, 3)

        jax.block_until_ready(call(state))          # warm / compile
        ts = []
        for _ in range(case["nrep"]):
            t0 = time.perf_counter()
            jax.block_until_ready(call(state))
            ts.append(1e3 * (time.perf_counter() - t0))
        out["ms_per_block"] = round(float(statistics.median(ts)), 4)
        out["ms_per_step"] = round(float(statistics.median(ts)) / nblock, 5)
        out["ms_per_step_min"] = round(min(ts) / nblock, 5)

        after = _device_peak()
        out["device_mem_before"] = before
        out["device_mem_after"] = after
        if after and "peak_bytes_in_use" in after:
            out["device_peak_bytes"] = int(after["peak_bytes_in_use"])
            out["device_peak_u"] = round(after["peak_bytes_in_use"] / u, 3)
    except Exception as exc:                        # noqa: BLE001 - a row, not a crash
        out["oom"] = _is_oom(exc)
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-2000:]
    return out


# ------------------------------------------------------------------ isolation

def _run_case_isolated(case, grad_chunk, precision, timeout):
    """Run one case in a fresh process: per-case device peak, and OOM cannot poison
    the allocator for the cases that follow."""
    env = dict(os.environ)
    if precision:
        env["TARANIS_PRECISION"] = str(precision)
    payload = json.dumps({"case": case, "grad_chunk": grad_chunk})
    try:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--single-case",
                               payload], capture_output=True, text=True, env=env,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"label": case["label"], "scheme": case["scheme"], "oom": False,
                "error": f"TimeoutExpired after {timeout}s", "params": case["params"]}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__ "):
            return json.loads(line[len("__RESULT__ "):])
    tail = (proc.stderr or proc.stdout or "")[-2000:]
    return {"label": case["label"], "scheme": case["scheme"], "params": case["params"],
            "oom": _is_oom(Exception(tail)), "error": f"subprocess rc={proc.returncode}",
            "traceback": tail}


# ------------------------------------------------------------------ reporting

def _rank_path(path):
    # every rank of a multi-rank launch writes; rank 0 keeps the plain name
    for var in ("SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK"):
        r = os.environ.get(var)
        if r is not None and r.isdigit():
            if int(r) == 0:
                return path
            stem, ext = os.path.splitext(path)
            return f"{stem}_rank{int(r)}{ext}"
    return path


def _fmt_table(results):
    hdr = f"{'case':<44} {'u (MB)':>8} {'total u':>8} {'peak u':>8} {'ms/step':>9}  note"
    lines = [hdr, "-" * len(hdr)]
    for r in results:
        u_mb = (r.get("u_bytes") or 0) / 2**20
        tot = r.get("total_u")
        peak = r.get("device_peak_u")
        ms = r.get("ms_per_step")
        note = "OOM" if r.get("oom") else ("ERROR" if r.get("error") else "")
        if r.get("per_device"):
            note = (note + " per-dev x%d" % r.get("devices_in_mesh", 0)).strip()
        lines.append(f"{r['label']:<44} {u_mb:8.2f} "
                     f"{(f'{tot:8.2f}' if tot is not None else '       -')} "
                     f"{(f'{peak:8.2f}' if peak is not None else '       -')} "
                     f"{(f'{ms:9.4f}' if ms is not None else '        -')}  {note}")
    return "\n".join(lines)


def main(profile="laptop", out=None, cases=None, precision=None, precision_check=False,
         grad_chunk=None, isolate=None, tag=None, nblock=None, nrep=None, timeout=3600,
         list_only=False):
    if profile not in PROFILES:
        raise SystemExit(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")

    if precision:
        # an already-imported taranis is locked to its precision (unset env means 32)
        cur = os.environ.get("TARANIS_PRECISION", "32")
        if "taranis" in sys.modules and cur != str(precision):
            raise SystemExit("taranis is already imported at a different precision; "
                             "set TARANIS_PRECISION in the environment instead")
        os.environ["TARANIS_PRECISION"] = str(precision)

    import jax
    import taranis as tr  # noqa: F401 - import after the precision env is settled
    from taranis import _precision

    if precision_check and precision and str(precision) != _precision.precision:
        raise SystemExit(f"--precision-check: asked for {precision}, taranis reports "
                         f"{_precision.precision}")

    case_list = PROFILES[profile](_precision.precision)
    if cases:
        wanted = set(cases if isinstance(cases, (list, tuple)) else str(cases).split(","))
        case_list = [c for c in case_list if c["label"] in wanted
                     or any(w in c["label"] for w in wanted)]
        if not case_list:
            raise SystemExit(f"no case matched {sorted(wanted)}")
    for c in case_list:
        if nblock:
            c["nblock"] = int(nblock)
        if nrep:
            c["nrep"] = int(nrep)

    if list_only:
        for c in case_list:
            print(c["label"])
        return {"cases": [c["label"] for c in case_list]}

    sharded = any(c["params"].get("comm_backend") == "jax" for c in case_list)
    if sharded:
        # comm_backend="jax" brings up jax.distributed, which jax refuses once the XLA
        # backend exists — so it must precede every device query below.
        tr.Parameters(**next(c["params"] for c in case_list
                             if c["params"].get("comm_backend") == "jax"))

    if os.environ.get("RMHD_REQUIRE_GPU") and jax.default_backend() == "cpu":
        raise SystemExit("RMHD_REQUIRE_GPU=1 but jax fell back to CPU — refusing to "
                         "report CPU numbers as GPU numbers")
    on_gpu = jax.default_backend() != "cpu"
    if isolate is None:
        isolate = on_gpu          # per-case device peak needs a fresh process
    if isolate and sharded:
        # subprocesses would each be their own MPI world; never isolate a sharded run
        print("note: comm_backend='jax' case present — running in-process (no isolation); "
              "device peaks are process high-water marks, not per-case", flush=True)
        isolate = False
    dev = jax.local_devices()[0]

    header = {
        "tag": tag or os.environ.get("TAG", "unset"),
        "profile": profile,
        "cases_filter": cases,          # the subset that ran, None for the whole profile
        "device": f"{dev.platform}:{getattr(dev, 'device_kind', '?')}",
        "n_devices": jax.device_count(),
        "backend": jax.default_backend(),
        "jax": jax.__version__,
        "precision": _precision.precision,
        "isolated": bool(isolate),
        "grad_chunk_requested": grad_chunk,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(header, indent=2), flush=True)

    results = []
    for i, case in enumerate(case_list, 1):
        print(f"[{i}/{len(case_list)}] {case['label']} ...", flush=True)
        if isolate:
            r = _run_case_isolated(case, grad_chunk, precision or _precision.precision,
                                   timeout)
        else:
            r = run_case(case, grad_chunk=grad_chunk)
        if r.get("error"):
            print(f"    {'OOM' if r['oom'] else 'ERROR'}: {r['error']}", flush=True)
        else:
            print(f"    total {r['total_u']:.2f} u, {r['ms_per_step']:.4f} ms/step",
                  flush=True)
        results.append(r)

    payload = dict(header, cases=results)
    path = _rank_path(out or "results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print("\n" + _fmt_table(results))
    print(f"\nwrote {os.path.abspath(path)}")
    return payload


def _cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="laptop", choices=sorted(PROFILES))
    ap.add_argument("--out", default=None, help="JSON path (default results.json in cwd)")
    ap.add_argument("--cases", default=None,
                    help="comma-separated label substrings to restrict the run")
    ap.add_argument("--precision", default=None, choices=("32", "64"),
                    help="sets TARANIS_PRECISION before importing taranis")
    ap.add_argument("--precision-check", action="store_true",
                    help="assert the running precision matches --precision")
    ap.add_argument("--grad-chunk", default=None, type=int,
                    help="fields per ifft in grad (sets shared_physics.GRAD_CHUNK)")
    ap.add_argument("--isolate", dest="isolate", action="store_true", default=None,
                    help="one subprocess per case (default: on when a GPU is present)")
    ap.add_argument("--no-isolate", dest="isolate", action="store_false")
    ap.add_argument("--tag", default=None, help="baseline | postF | postZ")
    ap.add_argument("--nblock", default=None, type=int)
    ap.add_argument("--nrep", default=None, type=int)
    ap.add_argument("--timeout", default=3600, type=int)
    ap.add_argument("--list", dest="list_only", action="store_true")
    ap.add_argument("--single-case", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a.single_case:                      # internal: one case, result on stdout
        spec = json.loads(a.single_case)
        r = run_case(spec["case"], grad_chunk=spec.get("grad_chunk"))
        print("__RESULT__ " + json.dumps(r), flush=True)
        return
    main(profile=a.profile, out=a.out, cases=a.cases, precision=a.precision,
         precision_check=a.precision_check, grad_chunk=a.grad_chunk, isolate=a.isolate,
         tag=a.tag, nblock=a.nblock, nrep=a.nrep, timeout=a.timeout,
         list_only=a.list_only)


if __name__ == "__main__":
    _cli()
