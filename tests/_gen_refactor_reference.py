# Generator for tests/data/refactor_reference_fp{64,32}.npz and the HLO opcode sidecar
# tests/data/refactor_reference_hlo_fp{64,32}.json -- the Phase-0 bitwise and compiled-
# graph reference of plans/REFACTOR_PLAN.md §1. Twelve solver configurations covering the
# driver/backend/scheme paths the gate-6 reference (2D RMHD only) does not reach:
# 3D finite-difference z, z_spectral on the separable and putzer2 backends, GDI under
# both IF and IMEX, adaptive dt with and without cfl blocks, rk44, imexcb3f, and the
# hoisted / unhoisted stage propagators.
#
# The npz holds the final fields and time; the JSON holds, per config, the instruction
# count per opcode of the OPTIMIZED HLO of the same jitted block_of_steps, counted inside
# and outside fusions, plus the total instruction and fusion counts. Identical output and
# an identical opcode histogram together mean the same compute, which is the plan's speed
# gate (§0.4).
#
# Not a test module (leading underscore keeps pytest away). Run both precisions
# (separate processes -- TARANIS_PRECISION is read once at import):
#   TARANIS_PRECISION=64 python tests/_gen_refactor_reference.py
#   TARANIS_PRECISION=32 python tests/_gen_refactor_reference.py
#
# CONFIGS / _ic / config_ctx / run_config / reference_path are the SHARED definition of
# the recorded runs: tests/test_refactor_reference.py imports them, so the checked-in npz
# and the test that reads it cannot drift apart. Changing any of them invalidates the
# committed npz.
from _rmhd_testing import bootstrap, fresh_params, make_state

bootstrap()

import collections
import json
import os
import platform
import re
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

import taranis as jr
from taranis import _precision
from taranis.run import block_of_steps
from taranis.timestepping import get_scheme

NSTEPS = 6

# jitted public block_of_steps, as tests/test_hoist_propagator.py drives it: no snapshot
# dirs, one compile per (params, nblock, scheme, stepper).
_advance = jax.jit(block_of_steps, static_argnums=(2, 3, 4, 5))

_GRID_3D = dict(dims=3, nx=16, ny=16, nz=8, Lx=2.0 * np.pi, Ly=2.0 * np.pi,
                Lz=2.0 * np.pi)
_GRID_2D = dict(dims=2, nx=32, ny=32, Lx=2.0 * np.pi, Ly=2.0 * np.pi)

# elsasser forcing drives both Elsasser fields; the momentum cell is the phi-only path.
_FORCED = dict(forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(1.0, 1.0),
               forcing_tau=1.0, fshell=(1, 3), forcing_seed=1)
_FORCED_MOM = dict(forcing=True, forcing_mode="momentum", forcing_power=1.0,
                   forcing_tau=1.0, fshell=(1, 3), forcing_seed=1)

_RMHD_DISS = dict(diss=(1e-4, 1e-4), hyper=2)          # nu == eta: separable under z_spectral
_RMHD_DISS_UNEQ = dict(diss=(1e-4, 2e-4), hyper=2)     # nu != eta: putzer2

# GDI is instability-driven and rejects forcing; these eqpars are bench/memory_probe.py's
# (2D: the collisional regime, 3D: the D_par*kz^2 parallel closure, z_spectral only).
_GDI_2D = dict(_GRID_2D, eqtype="GDI", adaptive_timestep=False, dt=0.01,
               eqpars=dict(Ln=392.0, nu_in=0.0106, v0=25.0, gpar_fac=1.0),
               diss=5e-6, hyper=2)
_GDI_3D = dict(_GRID_3D, eqtype="GDI", z_spectral=True, adaptive_timestep=False, dt=0.01,
               eqpars=dict(Ln=3.0, nu_in=0.2, v0=2.0, D_par=0.05, gpar_fac=0.0),
               diss=1e-3, hyper=2)

_FIXED = dict(adaptive_timestep=False, dt=0.01)
_CFL2 = dict(adaptive_timestep=True, cfl_safety=0.5, cfl_every=2)
_ADAPT = dict(adaptive_timestep=True, cfl_safety=0.5, cfl_every=1)

# (key, Parameters kwargs, scheme). Each row pins one solver path; the table in
# plans/REFACTOR_PLAN.md §1 says which.
CONFIGS = (
    ("fd_fixed_lsrk54",
     dict(_GRID_3D, **_RMHD_DISS, **_FIXED, **_FORCED), "lsrk54"),
    ("fd_cfl2_lsrk33",
     dict(_GRID_3D, **_RMHD_DISS, **_CFL2, **_FORCED), "lsrk33"),
    ("fd_adapt_rk44",
     dict(_GRID_3D, **_RMHD_DISS, **_ADAPT, **_FORCED), "rk44"),
    ("sep_fixed_lsrk54",
     dict(_GRID_3D, z_spectral=True, **_RMHD_DISS, **_FIXED, **_FORCED), "lsrk54"),
    ("put_cfl2_lsrk54",
     dict(_GRID_3D, z_spectral=True, **_RMHD_DISS_UNEQ, **_CFL2, **_FORCED), "lsrk54"),
    ("put_cfl2_lsrk54_nohoist",
     dict(_GRID_3D, z_spectral=True, hoist_propagator=False, **_RMHD_DISS_UNEQ, **_CFL2,
          **_FORCED), "lsrk54"),
    ("put_adapt_lsrk33_unrolled",
     dict(_GRID_3D, z_spectral=True, lsrk_scan=False, **_RMHD_DISS_UNEQ, **_ADAPT,
          **_FORCED), "lsrk33"),
    ("rmhd2d_adapt_lsrk33_mom",
     dict(_GRID_2D, **_RMHD_DISS, **_ADAPT, **_FORCED_MOM), "lsrk33"),
    ("rmhd2d_fixed_imexcb3f",
     dict(_GRID_2D, **_RMHD_DISS, **_FIXED, **_FORCED), "imexcb3f"),
    ("gdi2d_fixed_imexcb3e", dict(_GDI_2D), "imexcb3e"),
    ("gdi2d_fixed_lsrk33", dict(_GDI_2D), "lsrk33"),
    ("gdi3d_fixed_imexcb3e", dict(_GDI_3D), "imexcb3e"),
)


def reference_path(precision=None):
    """The committed reference npz for a precision ("64"/"32"), resolved relative to this
    file (not the cwd: script mode runs from the repo root, the Savio suite from a per-job
    scratch dir). Defaults to the running session's precision."""
    prec = _precision.precision if precision is None else str(precision)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        f"refactor_reference_fp{prec}.npz")


def hlo_reference_path(precision=None):
    """The committed HLO opcode sidecar for a precision, beside the npz."""
    prec = _precision.precision if precision is None else str(precision)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        f"refactor_reference_hlo_fp{prec}.json")


# one HLO instruction line: "%name = shape opcode(operands...)". The shape may itself
# contain spaces (tuple types), so the opcode is the last token before the paren.
# bench/hlo_audit.py's scan; matching every line of the module counts fusion bodies
# (their own computations) along with the entry computation.
_HLO_INSTR = re.compile(r"^\s*%?(\S+) = (.+?) ([a-z0-9\-]+)\(")


def hlo_histogram(text):
    """{opcode: count} over the whole optimized module, plus the totals."""
    counts = collections.Counter(m.group(3) for m in
                                 (_HLO_INSTR.match(line) for line in text.splitlines())
                                 if m)
    return {"opcodes": dict(sorted(counts.items())),
            "total_instructions": sum(counts.values()),
            "fusions": counts.get("fusion", 0)}


def _ic(x, y, z=None):
    # non-degenerate multi-mode IC in 2D and 3D (the two fields differ, so both Elsasser
    # fields and every entry of a 2x2 L are exercised). Same function as
    # tests/test_hoist_propagator.py's _ic; duplicated here so this generator's recorded
    # runs do not depend on another test module.
    zc = 1.0 if z is None else jnp.cos(z)
    f0 = (jnp.cos(x + 1.4) + jnp.cos(y + 2.0)) * zc + 0.3 * jnp.sin(2 * x + y)
    f1 = (jnp.cos(2 * x + 2.3) + 0.5 * jnp.cos(y + 6.2)) * zc + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([f0, f1], axis=0)


def config_ctx(kwargs):
    """(params, kgrid) for one recorded config. fresh_params, not ctx(): the GDI cells
    carry an eqpars dict, which ctx()'s hashable cache key cannot hold."""
    params = fresh_params(**kwargs)
    return params, jr.setup_kgrids(params)


def run_config(kwargs, schemestr, nsteps=NSTEPS):
    """NSTEPS steps of the jitted public block_of_steps from the multi-mode IC, plus the
    opcode histogram of the block's optimized HLO. Returns host arrays and a plain dict."""
    params, kgrid = config_ctx(kwargs)
    stepper, scheme = get_scheme(schemestr)
    state = make_state(params, ic=_ic)
    lowered = _advance.lower(state, kgrid, params, nsteps, scheme, stepper)
    hlo = hlo_histogram(lowered.compile().as_text())
    end = _advance(state, kgrid, params, nsteps, scheme, stepper)
    return {"fields": np.asarray(end.fields), "t": np.asarray(float(end.t)), "hlo": hlo}


def main(out_path=None, hlo_out_path=None):
    prec = _precision.precision
    out_path = out_path or reference_path(prec)
    hlo_out_path = hlo_out_path or hlo_reference_path(prec)
    out = {}
    hlo = {}
    summary = []
    for name, kwargs, schemestr in CONFIGS:
        result = run_config(kwargs, schemestr)
        hlo[name] = result.pop("hlo")
        for key, val in result.items():
            out[f"{name}_{key}"] = val
        summary.append((name, result))

    try:
        out["git_commit"] = np.array(subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))).strip())
    except Exception:
        out["git_commit"] = np.array("unknown")
    out["jax_version"] = np.array(jax.__version__)
    out["precision"] = np.array(prec)
    out["platform"] = np.array(jax.default_backend())
    out["python_version"] = np.array(platform.python_version())
    # host identity: bitwise reproduction is only expected on the same machine/OS
    out["hostname"] = np.array(platform.node())
    out["host_platform"] = np.array(platform.platform())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    os.makedirs(os.path.dirname(hlo_out_path), exist_ok=True)
    with open(hlo_out_path, "w") as f:
        json.dump({"configs": hlo,
                   **{k: str(out[k]) for k in ("git_commit", "jax_version", "precision",
                                               "platform", "python_version", "hostname",
                                               "host_platform")}},
                  f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {out_path}")
    print(f"wrote {hlo_out_path}")
    print()
    print("summary (t_final, max|fields|, HLO instructions/fusions):")
    for name, result in summary:
        amp = float(np.max(np.abs(result["fields"])))
        print(f"  {name}: t={float(result['t']):.8g}  max|fields|={amp:.6e}  "
              f"finite={bool(np.all(np.isfinite(result['fields'])))}  "
              f"hlo={hlo[name]['total_instructions']} instrs / "
              f"{hlo[name]['fusions']} fusions")

    return out_path


if __name__ == "__main__":
    main(*sys.argv[1:])
