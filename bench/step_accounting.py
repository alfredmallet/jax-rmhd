#!/usr/bin/env python3
# Step accounting: WHERE a step's memory and time go, per solver path.
#
# memory mode reports, per path, the XLA temp-arena lane table of the compiled
# run.block_of_steps -- each lane attributed to the taranis function that produced it --
# plus the temp/args/out split. timing mode runs an ablation ladder: one variant per
# process, each swapping one part of the step for a cheap stand-in of identical shapes,
# so the delta against the baseline is that part's cost inside the fused, scanned step.
#
# u = one field-sized complex array = nz_local*nkx*nky*itemsize. An OOM is a result row.
#
#   python bench/step_accounting.py --mode memory --profile laptop
#   python bench/step_accounting.py --mode both --profile p100 --precision 32 --out acct.json
#   python bench/step_accounting.py --mode timing --paths fdz,zspec --rounds 5
#   python bench/step_accounting.py --list
#
# main(**kwargs) takes the same options, for the lugus kernel entry point.
#
# TARANIS_PRECISION is read at taranis import, so --precision sets it before importing;
# nothing here may import taranis at module scope.
import argparse
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

L = 6.283185307179586

RMHD_EQPARS = {"diss": (1e-4, 1e-4), "hyper": 2}
RMHD_EQPARS_ZS = {"diss": (1e-4, 1e-4), "hyper": 2, "z_diss_k": 1e-6}
GDI_EQPARS_2D = {"Ln": 392.0, "nu_in": 0.0106, "v0": 25.0, "gpar_fac": 1.0,
                 "diss": 5e-6, "hyper": 2}

# elsasser forcing at the shell the production RMHD runs use; the forcing temporaries
# (O-U state, envelope, the power normalisation reductions) have never been on the
# memory probe's books, which run every case unforced
FORCING = dict(forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(0.5, 0.5),
               forcing_tau=1.0, fshell=(1, 3), forcing_seed=1)

# ------------------------------------------------------------------ paths and profiles


def _base(nx, ny, **kw):
    # the settings every accounting case shares: fixed dt (nothing frozen or refrozen
    # between steps), serial backend, no particles
    p = dict(nx=nx, ny=ny, Lx=L, Ly=L, cfl_safety=0.5, adaptive_timestep=False,
             dt=1e-3, comm_backend="serial")
    p.update(kw)
    return p


def _rmhd(nx, nz, z_spectral, **kw):
    return _base(nx, nx, nz=nz, dims=3, eqtype="RMHD", Lz=L, z_spectral=z_spectral,
                 eqpars=RMHD_EQPARS_ZS if z_spectral else RMHD_EQPARS, **kw)


def _paths(nx, nz, nx2d):
    # label -> (params, scheme, nblock). The three production paths plus the two extra
    # accounting cases (forced elsasser z_spectral; the snapshot peak sequence).
    return {
        "fdz":            (_rmhd(nx, nz, False), "lsrk54", 5),
        "zspec":          (_rmhd(nx, nz, True), "lsrk54", 5),
        "gdi2d":          (_base(nx2d, nx2d, dims=2, eqtype="GDI",
                                 eqpars=GDI_EQPARS_2D), "lsrk33", 10),
        "zspec_adaptive": (_rmhd(nx, nz, True, adaptive_timestep=True, cfl_every=1),
                           "lsrk33", 5),
        "zspec_forced":   (_rmhd(nx, nz, True, **FORCING), "lsrk54", 5),
        "snapshot_peak":  (_rmhd(nx, nz, True, **FORCING), "lsrk54", 5),
    }


PROFILES = {
    # the canonical CPU accounting grid, and the GPU grids the plan sizes production from
    "laptop": lambda prec: _paths(128, 32, 256),
    "p100":   lambda prec: _paths(512 if prec != "64" else 256,
                                  128 if prec != "64" else 64,
                                  1024 if prec != "64" else 512),
    "a100":   lambda prec: _paths(1024 if prec != "64" else 512,
                                  128 if prec != "64" else 64,
                                  2048 if prec != "64" else 1024),
}

DEFAULT_PATHS = ("fdz", "zspec", "gdi2d")

# ------------------------------------------------------------------ ablation variants
#
# Each variant replaces ONE part of the step with a cheap stand-in of identical shapes and
# dtypes. Every variant is numerically WRONG on purpose. The stand-ins keep the reads their
# original made wherever possible, so a variant does not dead-code the code that feeds it;
# the "_dead" twins deliberately do not, and bound the same part from above.

VARIANTS = ("base", "noifft", "nofft", "notrans", "nogradk", "nobracket",
            "nobracket_dead", "nozarith", "nohalo", "nofdlin", "nonlin", "noprop",
            "noexpform", "nocfl", "norhs", "norhs_noprop")

# the ladder each path is worth running; parts a path does not have are left out
PATH_VARIANTS = {
    "fdz": VARIANTS,
    "zspec": tuple(v for v in VARIANTS if v not in ("nozarith", "nohalo", "nofdlin",
                                                    "nocfl")),
    "gdi2d": tuple(v for v in VARIANTS if v not in ("nozarith", "nohalo", "nofdlin",
                                                    "nocfl")),
    "zspec_adaptive": ("base", "notrans", "noprop", "noexpform", "nocfl", "nonlin",
                       "norhs", "norhs_noprop"),
    "zspec_forced": ("base", "notrans", "noprop", "norhs", "norhs_noprop"),
}


def _apply_variant(variant):
    """Monkeypatch one part of the step. Must run before the first trace."""
    import jax
    import jax.numpy as jnp
    from taranis import run as jrun, _precision, grids, comms, propagators
    from taranis.physics import shared_physics, rmhd, gdi, equation_registry

    CT, FT = _precision.ctype, _precision.ftype

    def cheap_ifft(f, params):
        # grids.ifft's output shape and dtype with no transform arithmetic; the sum still
        # reads every input element, so the k-space gradient multiply feeding it stays live
        red = jnp.sum(jnp.real(f), axis=-1, keepdims=True)
        return red * jnp.ones((params.ny,), dtype=FT)

    def cheap_fft(f, params):
        red = jnp.sum(f, axis=-1, keepdims=True)
        z = jax.lax.complex(red, red)
        return z * jnp.ones((f.shape[-1] // 2 + 1,), dtype=CT)

    def cheap_bracket(a, b):
        # 3 adds instead of 2 mul + 1 sub, all four components still read
        return a[0] + a[1] + b[0] + b[1]

    def cheap_bracket_dead(a, b):
        # half the operands unread: their transforms die too (upper bound)
        return a[0] + b[0]

    def grad_fields_noscale(fks, kgrid, params):
        # the i*k_perp broadcast-array multiply replaced by a complex scalar multiply
        return tuple(jnp.stack([grids.ifft(1j * fk, params), grids.ifft(2j * fk, params)])
                     for fk in fks)

    def cheap_z_stencils(p2, p1, c, m1, m2, dz):
        # the halo, concatenate and slices stay; the 4th-order arithmetic goes
        return p1, c

    def cheap_halo(f, params, width=2):
        z = jnp.zeros((f.shape[0], width) + f.shape[2:], dtype=f.dtype)
        return z, z

    def zero_term(state, grads, kgrid, params, halo=None):
        return jnp.zeros_like(state.fields)

    def drop_term(name):
        for eq, rec in list(equation_registry.items()):
            equation_registry[eq] = rec._replace(
                # substitute inside the Term record: the predicate stays as shipped
                term_funcs=tuple(t._replace(func=zero_term) if t.func.__name__ == name
                                 else t for t in rec.term_funcs))
        jrun.equation_registry = equation_registry

    if variant in ("noifft", "notrans"):
        grids.ifft = cheap_ifft
    if variant in ("nofft", "notrans"):
        grids.fft = cheap_fft
    if variant in ("nobracket", "nobracket_dead"):
        b = cheap_bracket if variant == "nobracket" else cheap_bracket_dead
        shared_physics.bracket = rmhd.bracket = gdi.bracket = b
    if variant == "nogradk":
        shared_physics.grad_fields = rmhd.grad_fields = gdi.grad_fields = grad_fields_noscale
    if variant == "nozarith":
        shared_physics._z_stencils = cheap_z_stencils
    if variant == "nohalo":
        comms.halo_exchange = cheap_halo
        shared_physics.comms = comms
    if variant == "nofdlin":
        drop_term("FDLinearTerm")
    if variant == "nonlin":
        drop_term("NonlinearTerm")
    if variant in ("noprop", "norhs_noprop"):
        for cls in (propagators.SeparableExp, propagators.Putzer2Exp,
                    propagators.DiagonalExp, propagators.IdentityExp):
            cls.apply = lambda self, arr: arr
    if variant == "noexpform":
        # keep the apply, drop the transcendental coefficient evaluation
        for cls, mk in (
                (propagators.SeparableL,
                 lambda self, tau: propagators.SeparableExp(
                     P=jnp.ones_like(self.dperp), c=jnp.ones_like(self.dz),
                     s=jnp.zeros_like(self.kz))),
                (propagators.Putzer2Operator,
                 lambda self, tau: propagators.Putzer2Exp(
                     m00=jnp.ones_like(self.L[0, 0]), m01=jnp.zeros_like(self.L[0, 1]),
                     m10=jnp.zeros_like(self.L[1, 0]), m11=jnp.ones_like(self.L[1, 1]))),
                (propagators.DiagonalOperator,
                 lambda self, tau: propagators.DiagonalExp(e=jnp.ones_like(self.L)))):
            cls.exp_op = mk
    if variant in ("norhs", "norhs_noprop"):
        jrun.construct_rhs = lambda recipe: (
            lambda state, kgrid, params: (jnp.zeros_like(state.fields), None))
    if variant == "nocfl":
        for eq, rec in list(equation_registry.items()):
            equation_registry[eq] = rec._replace(
                set_timestep_func=lambda grads, params: jnp.asarray(params.dt, dtype=FT))
        jrun.equation_registry = equation_registry


# ------------------------------------------------------------------ scope attribution

SCOPES = ("FFTfwd", "IFFTinv", "GRADF", "BRACKET", "ZSTENCIL", "HALO", "EXPFORM",
          "EXPAPPLY", "SETDT", "NONLINEARTERM", "FDLINEARTERM", "FORCINGTERM", "GRAD")


def _apply_scopes():
    """Wrap the RHS seams in jax.named_scope so every HLO instruction's op_name carries
    the taranis function that produced it. Metadata only: the compiled memory is identical,
    which the memory mode asserts."""
    import functools
    import jax
    from taranis import run as jrun, grids, comms, propagators
    from taranis.physics import shared_physics, rmhd, gdi, equation_registry

    def scoped(name, fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with jax.named_scope(name):
                return fn(*a, **kw)
        return wrapper

    grids.fft = scoped("FFTfwd", grids.fft)
    grids.ifft = scoped("IFFTinv", grids.ifft)
    for mod in (rmhd, gdi):
        mod.grad_fields = scoped("GRADF", mod.grad_fields)
        mod.bracket = scoped("BRACKET", mod.bracket)
    shared_physics.grad_fields = scoped("GRADF", shared_physics.grad_fields)
    shared_physics.bracket = scoped("BRACKET", shared_physics.bracket)
    rmhd.z_derivatives = scoped("ZSTENCIL", rmhd.z_derivatives)
    shared_physics.z_derivatives = scoped("ZSTENCIL", shared_physics.z_derivatives)
    comms.halo_exchange = scoped("HALO", comms.halo_exchange)
    shared_physics.comms = comms
    for cls in (propagators.SeparableExp, propagators.Putzer2Exp, propagators.DiagonalExp,
                propagators.IdentityExp):
        cls.apply = scoped("EXPAPPLY", cls.apply)
    for cls in (propagators.SeparableL, propagators.Putzer2Operator,
                propagators.DiagonalOperator, propagators.IdentityOperator):
        cls.exp_op = scoped("EXPFORM", cls.exp_op)
    for eq, rec in list(equation_registry.items()):
        mod = rmhd if eq == "RMHD" else gdi
        equation_registry[eq] = rec._replace(
            grad_func=scoped("GRAD", mod.grad),
            term_funcs=tuple(t._replace(func=scoped(t.func.__name__.upper(), t.func))
                             for t in rec.term_funcs),
            set_timestep_func=scoped("SETDT", rec.set_timestep_func),
            halo_start_func=(scoped("HALOSTART", rec.halo_start_func)
                             if rec.halo_start_func is not None else None))
    jrun.equation_registry = equation_registry


def _lane_table(dumpdir, u, top=14):
    """Temp-arena lanes from the XLA buffer-assignment dump, attributed by named_scope.

    A lane is one arena OFFSET, sized by the largest buffer placed there. Buffers at
    overlapping addresses are lifetime-disjoint reuse, so the lane-max sum exceeds the
    arena high-water; both are reported."""
    ba = [f for f in glob.glob(os.path.join(dumpdir, "*buffer-assignment.txt"))
          if "block_of_steps" in f]
    if not ba:
        return None
    ba = sorted(ba, key=os.path.getsize)[-1]
    hlo = ba.replace("-buffer-assignment.txt", ".txt")

    meta, comp_ops, fusion_of, cur = {}, {}, {}, None
    if os.path.exists(hlo):
        for line in open(hlo, errors="replace"):
            mc = re.match(r"\s*(?:ENTRY )?%?([\w\.\-]+) \(", line)
            if mc and "= " not in line:
                cur = mc.group(1)
            mi = re.match(r"\s*%?([\w\.\-]+) = ", line)
            if not mi:
                continue
            name = mi.group(1)
            op = re.search(r'op_name="([^"]*)"', line)
            meta[name] = op.group(1) if op else ""
            if cur:
                comp_ops.setdefault(cur, []).append(meta[name])
            cf = re.search(r"calls=%?([\w\.\-]+)", line)
            if cf and "fusion" in line:
                fusion_of[name] = cf.group(1)

    def scopes_of(name):
        paths = [meta.get(name, "")]
        comp = fusion_of.get(name)      # only real fusions expand into their computation
        if comp:
            paths += comp_ops.get(comp, [])
        return [s for s in SCOPES if any(f"/{s}/" in p + "/" for p in paths)]

    lanes, args, outs, cur = {}, [], [], None
    for line in open(ba, errors="replace"):
        if line.startswith("allocation"):
            cur = line
            ma = re.match(r"allocation \d+: size (\d+), ([^\n]*)", line)
            if ma and "parameter" in ma.group(2):
                args.append((int(ma.group(1)), ma.group(2).strip()[:80]))
            elif ma and "maybe-live-out" in ma.group(2):
                outs.append((int(ma.group(1)), ma.group(2).strip()[:80]))
            continue
        mv = re.match(r"\s+value: <\d+ (\S+) [^>]*> \(size=(\d+),offset=(\d+)\): (\S+)",
                      line)
        if mv and cur and "maybe-live-out" not in cur and "parameter" not in cur:
            name, size, off, shape = (mv.group(1), int(mv.group(2)), int(mv.group(3)),
                                      mv.group(4))
            rec = lanes.setdefault(off, {"size": 0, "vals": []})
            rec["size"] = max(rec["size"], size)
            rec["vals"].append((size, name, shape))

    rows = []
    for off, rec in sorted(lanes.items(), key=lambda kv: -kv[1]["size"]):
        rec["vals"].sort(reverse=True)
        size, name, shape = rec["vals"][0]
        rows.append({"u": round(rec["size"] / u, 3), "offset": off, "shape": shape,
                     "n_values": len(rec["vals"]), "value": name,
                     "scopes": scopes_of(name) or ["carry/other"]})
    return {"lanes": rows[:top], "n_lanes": len(rows),
            "lane_max_sum_u": round(sum(r["size"] for r in lanes.values()) / u, 3),
            "args_allocs": [{"u": round(s / u, 3), "what": w} for s, w in
                            sorted(args, reverse=True)[:8]],
            "out_allocs": [{"u": round(s / u, 3), "what": w} for s, w in
                           sorted(outs, reverse=True)[:4]]}


# ------------------------------------------------------------------ one case, in-process


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


def _is_oom(exc):
    s = f"{type(exc).__name__}: {exc}"
    return any(k in s for k in ("RESOURCE_EXHAUSTED", "out of memory", "Out of memory",
                                "OutOfMemory", "bad_alloc"))


def _device_peak():
    import jax
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return {k: stats.get(k) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
            if k in stats}


def _setup(params, scheme_name):
    import jax.numpy as jnp
    import numpy as np
    import taranis as tr
    from taranis import run as jrun, _precision
    from taranis.grids import setup_kgrids
    from taranis.timestepping import get_scheme

    p = tr.Parameters(**params)
    u = ((p.nz if p.spatial_dimensions == 3 else 1) * p.nx * (p.ny // 2 + 1)
         * np.dtype(_precision.ctype).itemsize)
    kgrid = setup_kgrids(p)
    state = jrun.initialize(_ic_3d(jnp) if p.spatial_dimensions == 3 else _ic_2d(jnp), p)
    stepper, scheme = get_scheme(scheme_name)
    return p, u, kgrid, state, stepper, scheme


def run_memory(path, params, scheme_name, nblock, scopes=True, dumpdir=None, top=14):
    """Compile block_of_steps and report memory_analysis, plus the lane table when a
    dump directory was pre-armed via XLA_FLAGS (see _single)."""
    import jax
    out = {"path": path, "mode": "memory", "scheme": scheme_name, "nblock": nblock,
           "params": params, "oom": False, "error": None, "scoped": bool(scopes)}
    try:
        if scopes:
            _apply_scopes()
        from taranis import run as jrun
        p, u, kgrid, state, stepper, scheme = _setup(params, scheme_name)
        out["u_bytes"] = int(u)
        fn = jax.jit(jrun.block_of_steps, static_argnums=(2, 3, 4, 5))
        compiled = fn.lower(state, kgrid, p, nblock, scheme, stepper).compile()
        m = compiled.memory_analysis()
        mem = {"temp": int(m.temp_size_in_bytes), "args": int(m.argument_size_in_bytes),
               "out": int(m.output_size_in_bytes)}
        total = sum(mem.values())
        out["mem_analysis"] = mem
        out["temp_u"] = round(mem["temp"] / u, 3)
        out["args_u"] = round(mem["args"] / u, 3)
        out["out_u"] = round(mem["out"] / u, 3)
        out["total_bytes"] = int(total)
        out["total_u"] = round(total / u, 3)
        if dumpdir:
            out["lane_table"] = _lane_table(dumpdir, u, top=top)
    except Exception as exc:                        # noqa: BLE001 - a row, not a crash
        out["oom"] = _is_oom(exc)
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-2000:]
    return out


def run_timing(path, params, scheme_name, nblock, variant="base", nrep=21):
    """Time one ablation variant of one path. One variant per process: the jaxpr trace
    cache is keyed on function and avals, so a second variant in the same process would
    silently re-report the first."""
    import jax
    out = {"path": path, "mode": "timing", "scheme": scheme_name, "nblock": nblock,
           "variant": variant, "nrep": nrep, "params": params, "oom": False, "error": None}
    try:
        _apply_variant(variant)
        from taranis import run as jrun
        p, u, kgrid, state, stepper, scheme = _setup(params, scheme_name)
        out["u_bytes"] = int(u)
        step = jax.jit(jrun.block_of_steps, static_argnums=(2, 3, 4, 5))
        carry = step(state, kgrid, p, nblock, scheme, stepper)
        jax.block_until_ready(carry.fields)
        ts = []
        for _ in range(nrep):
            t0 = time.perf_counter()
            carry = step(carry, kgrid, p, nblock, scheme, stepper)
            jax.block_until_ready(carry.fields)
            ts.append(1e3 * (time.perf_counter() - t0) / nblock)
        compiled = step.lower(state, kgrid, p, nblock, scheme, stepper).compile()
        m = compiled.memory_analysis()
        cost = compiled.cost_analysis()
        if isinstance(cost, (list, tuple)):
            cost = cost[0]
        out["ms_per_step"] = round(float(statistics.median(ts)), 5)
        out["ms_min"] = round(float(min(ts)), 5)
        out["ms_spread"] = round(float(max(ts) - min(ts)), 5)
        out["total_u"] = round((m.temp_size_in_bytes + m.argument_size_in_bytes
                                + m.output_size_in_bytes) / u, 3)
        out["gflop_per_step"] = round(float(cost.get("flops", 0)) / nblock / 1e9, 6)
        out["gb_per_step"] = round(float(cost.get("bytes accessed", 0)) / nblock / 1e9, 6)
        out["device_peak"] = _device_peak()
    except Exception as exc:                        # noqa: BLE001
        out["oom"] = _is_oom(exc)
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-2000:]
    return out


def run_snapshot_peak(path, params, scheme_name, nblock, nrep=2):
    """Device peak across block -> save_snapshot -> block. The save materialises host
    copies of every state array and orbax holds its own buffers, so the run's high-water
    mark can sit at the checkpoint, not in the step -- which the step-only probe cannot
    see. On CPU memory_stats() is unavailable and the peaks come back null."""
    import jax
    out = {"path": path, "mode": "snapshot_peak", "scheme": scheme_name,
           "nblock": nblock, "params": params, "oom": False, "error": None}
    tmp = tempfile.mkdtemp(prefix="step_acct_snap_")
    try:
        from taranis import run as jrun
        from taranis.snapshot_io import snapshot_manager_setup, save_snapshot
        p, u, kgrid, state, stepper, scheme = _setup(params, scheme_name)
        out["u_bytes"] = int(u)
        step = jax.jit(jrun.block_of_steps, static_argnums=(2, 3, 4, 5))
        stages = {}
        state = step(state, kgrid, p, nblock, scheme, stepper)
        jax.block_until_ready(state.fields)
        stages["after_block_1"] = _device_peak()
        mngr = snapshot_manager_setup(p, snap_path=tmp, nsnap=4)
        save_snapshot(0, state, mngr, p)
        mngr.wait_until_finished()
        stages["after_save"] = _device_peak()
        state = step(state, kgrid, p, nblock, scheme, stepper)
        jax.block_until_ready(state.fields)
        stages["after_block_2"] = _device_peak()
        out["stages"] = stages
        peaks = [s["peak_bytes_in_use"] for s in stages.values()
                 if s and "peak_bytes_in_use" in s]
        out["peak_bytes"] = max(peaks) if peaks else None
        out["peak_u"] = round(max(peaks) / u, 3) if peaks else None
        out["peak_stage"] = (max(stages, key=lambda k: (stages[k] or {}).get(
            "peak_bytes_in_use", -1)) if peaks else None)
        if not peaks:
            out["note"] = "device memory_stats unavailable (CPU backend): peaks are null"
    except Exception as exc:                        # noqa: BLE001
        out["oom"] = _is_oom(exc)
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-2000:]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# ------------------------------------------------------------------ isolation


def _single(payload):
    """Run one unit of work in THIS process and print it as __RESULT__ json. Every unit
    is isolated because timing needs one variant per process and memory needs its own
    XLA dump directory."""
    job = json.loads(payload)
    mode = job["mode"]
    if mode == "memory":
        r = run_memory(job["path"], job["params"], job["scheme"], job["nblock"],
                       scopes=job.get("scopes", True), dumpdir=job.get("dumpdir"),
                       top=job.get("top", 14))
    elif mode == "timing":
        r = run_timing(job["path"], job["params"], job["scheme"], job["nblock"],
                       variant=job.get("variant", "base"), nrep=job.get("nrep", 21))
    elif mode == "snapshot_peak":
        r = run_snapshot_peak(job["path"], job["params"], job["scheme"], job["nblock"])
    else:
        raise SystemExit(f"unknown mode {mode!r}")
    print("__RESULT__ " + json.dumps(r))


def _isolated(job, precision, timeout, dump_root=None):
    """One unit of work in a fresh process. XLA_FLAGS must be set before the process
    starts for the buffer dump to exist, so it is set here, not inside the child."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PMI_", "PMIX_", "OMPI_", "SLURM_", "MPI4JAX_", "MV2_"))}
    if precision:
        env["TARANIS_PRECISION"] = str(precision)
    if job.get("dumpdir"):
        shutil.rmtree(job["dumpdir"], ignore_errors=True)
        os.makedirs(job["dumpdir"], exist_ok=True)
        env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                            + f" --xla_dump_to={job['dumpdir']}").strip()
    try:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--single",
                               json.dumps(job)], capture_output=True, text=True, env=env,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(job, oom=False, error=f"TimeoutExpired after {timeout}s")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__ "):
            return json.loads(line[len("__RESULT__ "):])
    tail = (proc.stderr or proc.stdout or "")[-2000:]
    return dict(job, oom=_is_oom(Exception(tail)),
                error=f"subprocess rc={proc.returncode}", traceback=tail)


# ------------------------------------------------------------------ reporting


def _fmt_memory(results):
    lines = [f"{'path':<18}{'u MiB':>9}{'temp':>9}{'args':>9}{'out':>9}{'total u':>10}"
             "  note", "-" * 78]
    for r in results:
        if r.get("error"):
            lines.append(f"{r['path']:<18}{'-':>9}{'-':>9}{'-':>9}{'-':>9}{'-':>10}  "
                         f"{'OOM' if r['oom'] else 'ERROR'}: {r['error'][:40]}")
            continue
        lines.append(f"{r['path']:<18}{r['u_bytes']/2**20:9.2f}{r['temp_u']:9.2f}"
                     f"{r['args_u']:9.2f}{r['out_u']:9.2f}{r['total_u']:10.3f}  "
                     f"{r.get('lane_table', {}).get('n_lanes', '-')} lanes")
    return "\n".join(lines)


def _fmt_lanes(r):
    lt = r.get("lane_table")
    if not lt:
        return ""
    out = [f"  {r['path']}: {lt['n_lanes']} lanes, lane-max sum {lt['lane_max_sum_u']} u,"
           f" arena {r['temp_u']} u"]
    for lane in lt["lanes"]:
        out.append(f"    {lane['u']:7.3f}u  {lane['shape']:24s} x{lane['n_values']:<3}"
                   f" {'+'.join(lane['scopes'])}")
    for a in lt["args_allocs"]:
        out.append(f"    arg {a['u']:7.3f}u  {a['what']}")
    return "\n".join(out)


def _fmt_timing(results):
    by = {}
    for r in results:
        by.setdefault(r["path"], []).append(r)
    lines = []
    for path, rs in by.items():
        base = [x for x in rs if x.get("variant") == "base" and not x.get("error")]
        bmin = min((x["ms_min"] for x in base), default=None)
        lines.append(f"\n{path}: base {bmin} ms/step (min over {len(base)} rounds)")
        lines.append(f"  {'variant':<16}{'min':>9}{'median':>9}{'delta':>9}{'% base':>9}"
                     f"{'u':>8}")
        agg = {}
        for r in rs:
            if r.get("error"):
                continue
            agg.setdefault(r["variant"], []).append(r)
        for v, xs in agg.items():
            mn = min(x["ms_min"] for x in xs)
            md = statistics.median([x["ms_per_step"] for x in xs])
            d = (bmin - mn) if bmin else float("nan")
            pc = 100 * d / bmin if bmin else float("nan")
            lines.append(f"  {v:<16}{mn:9.3f}{md:9.3f}{d:9.3f}{pc:9.1f}"
                         f"{xs[0]['total_u']:8.2f}")
    return "\n".join(lines)


# ------------------------------------------------------------------ entry point


def main(profile="laptop", mode="memory", paths=None, out=None, precision=None,
         rounds=3, nrep=21, nblock=None, top=14, timeout=3600, tag=None,
         list_only=False, dump_root=None):
    if profile not in PROFILES:
        raise SystemExit(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    if mode not in ("memory", "timing", "both"):
        raise SystemExit(f"unknown mode {mode!r}; choose memory, timing or both")

    if precision:
        cur = os.environ.get("TARANIS_PRECISION", "32")
        if "taranis" in sys.modules and cur != str(precision):
            raise SystemExit("taranis is already imported at a different precision; "
                             "set TARANIS_PRECISION in the environment instead")
        os.environ["TARANIS_PRECISION"] = str(precision)

    import jax
    import taranis as tr  # noqa: F401 - import after the precision env is settled
    from taranis import _precision

    path_specs = PROFILES[profile](_precision.precision)
    wanted = (list(paths) if isinstance(paths, (list, tuple))
              else (str(paths).split(",") if paths else list(DEFAULT_PATHS)))
    unknown = [w for w in wanted if w not in path_specs]
    if unknown:
        raise SystemExit(f"unknown paths {unknown}; choose from {sorted(path_specs)}")

    if list_only:
        for name in path_specs:
            print(name)
        return {"paths": list(path_specs)}

    dev = jax.local_devices()[0]
    header = {
        "tag": tag or os.environ.get("TAG", "unset"),
        "profile": profile, "mode": mode, "paths": wanted,
        "device": f"{dev.platform}:{getattr(dev, 'device_kind', '?')}",
        "backend": jax.default_backend(), "jax": jax.__version__,
        "precision": _precision.precision, "rounds": rounds, "nrep": nrep,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(header, indent=2), flush=True)

    dump_root = dump_root or tempfile.mkdtemp(prefix="step_acct_dump_")
    mem_results, time_results = [], []

    if mode in ("memory", "both"):
        for name in wanted:
            params, scheme, nb = path_specs[name]
            nb = int(nblock or nb)
            print(f"[memory] {name} ...", flush=True)
            if name == "snapshot_peak":
                r = _isolated({"mode": "snapshot_peak", "path": name, "params": params,
                               "scheme": scheme, "nblock": nb}, precision, timeout)
                mem_results.append(r)
                print(f"    peak {r.get('peak_u')} u at {r.get('peak_stage')}"
                      f" {r.get('note','')}", flush=True)
                continue
            job = {"mode": "memory", "path": name, "params": params, "scheme": scheme,
                   "nblock": nb, "top": top,
                   "dumpdir": os.path.join(dump_root, name)}
            # plain compile first, then the named_scope one: named_scope is metadata only,
            # so a difference in total means the attribution changed the graph
            plain = _isolated(dict(job, scopes=False, dumpdir=None), precision, timeout)
            r = _isolated(dict(job, scopes=True), precision, timeout)
            if not r.get("error") and not plain.get("error"):
                r["plain_total_u"] = plain["total_u"]
                r["scope_neutral"] = plain["total_bytes"] == r["total_bytes"]
            mem_results.append(r)
            if r.get("error"):
                print(f"    {'OOM' if r['oom'] else 'ERROR'}: {r['error']}", flush=True)
            else:
                print(f"    total {r['total_u']:.3f} u "
                      f"(temp {r['temp_u']:.3f} / args {r['args_u']:.3f} / "
                      f"out {r['out_u']:.3f}), scope-neutral="
                      f"{r.get('scope_neutral')}", flush=True)

    if mode in ("timing", "both"):
        # interleaved rounds: every variant re-measured in a fresh process each round, so
        # drift shows up as round-to-round scatter and not as a bias on one variant
        for rnd in range(1, int(rounds) + 1):
            for name in wanted:
                if name == "snapshot_peak":
                    continue
                params, scheme, nb = path_specs[name]
                nb = int(nblock or nb)
                for v in PATH_VARIANTS.get(name, VARIANTS):
                    print(f"[timing r{rnd}] {name} {v} ...", flush=True)
                    r = _isolated({"mode": "timing", "path": name, "params": params,
                                   "scheme": scheme, "nblock": nb, "variant": v,
                                   "nrep": int(nrep)}, precision, timeout)
                    r["round"] = rnd
                    time_results.append(r)
                    if r.get("error"):
                        print(f"    {'OOM' if r['oom'] else 'ERROR'}: {r['error']}",
                              flush=True)
                    else:
                        print(f"    {r['ms_per_step']:.3f} ms/step", flush=True)

    payload = dict(header, memory=mem_results, timing=time_results)
    path = out or "step_accounting.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    if mem_results:
        print("\n" + _fmt_memory([r for r in mem_results if r.get("mode") == "memory"]))
        for r in mem_results:
            txt = _fmt_lanes(r)
            if txt:
                print(txt)
    if time_results:
        print(_fmt_timing(time_results))
    print(f"\nwrote {os.path.abspath(path)}")
    return payload


def _cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="laptop", choices=sorted(PROFILES))
    ap.add_argument("--mode", default="memory", choices=("memory", "timing", "both"))
    ap.add_argument("--paths", default=None,
                    help="comma-separated path names (default fdz,zspec,gdi2d)")
    ap.add_argument("--out", default=None, help="JSON path")
    ap.add_argument("--precision", default=None, choices=("32", "64"),
                    help="sets TARANIS_PRECISION before importing taranis")
    ap.add_argument("--rounds", default=3, type=int, help="interleaved timing rounds")
    ap.add_argument("--nrep", default=21, type=int, help="timed calls per variant")
    ap.add_argument("--nblock", default=None, type=int, help="override steps per block")
    ap.add_argument("--top", default=14, type=int, help="lanes per path in the table")
    ap.add_argument("--timeout", default=3600, type=int)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--list", dest="list_only", action="store_true")
    ap.add_argument("--single", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.single:
        return _single(a.single)
    return main(profile=a.profile, mode=a.mode, paths=a.paths, out=a.out,
                precision=a.precision, rounds=a.rounds, nrep=a.nrep, nblock=a.nblock,
                top=a.top, timeout=a.timeout, tag=a.tag, list_only=a.list_only)


if __name__ == "__main__":
    _cli()
