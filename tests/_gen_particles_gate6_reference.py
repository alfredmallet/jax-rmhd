# Generator for tests/data/particles_gate6_reference_fp{64,32}.npz -- gate 6
# ("Solver untouched", plans/TESTPART_PLAN.md §6) requires a reference run RECORDED
# BEFORE A2 WIRING BEGINS: once particle co-stepping is wired into run.py
# (block_of_steps / _cfl_block / simulate / simulate_scan carrying a (state, pstate)
# tuple), gate 6 asserts that with params.particles=None the solver output is bitwise
# identical to what current main produces. This script IS that pre-wiring capture, run
# on the current (unmodified) tree.
#
# GENERATED 2026-08-18. REGENERATED 2026-08-20 on the post-F3 tree: the F3 stage-0 peel
# (plans/MEMORY_PERF_PLAN.md §3) changed XLA's fusion of the LSRK stage scan at the 1-2 ulp
# level, accepted as a §9.1 decision.
# Not a test module (leading underscore keeps pytest away). Run
# both precisions (separate processes -- TARANIS_PRECISION is read once at import):
#   TARANIS_PRECISION=64 python tests/_gen_particles_gate6_reference.py
#   TARANIS_PRECISION=32 python tests/_gen_particles_gate6_reference.py
#
# CONFIGS / config_ctx / run_config / reference_path are the SHARED definition of the
# five recorded runs: the future tests/test_particles_coupled.py gate-6 test imports them
# so the checked-in reference and the test that reads it can never drift apart. Changing
# any of them invalidates the committed npz -- regenerate on the pre-A2 tree, not after.
from _rmhd_testing import bootstrap, ctx, make_state, managed_manager, snap_dir

bootstrap()

import os
import re
import subprocess
import sys
import platform

import jax
import numpy as np

import taranis as jr
from taranis import diagnostics as dg
from taranis import _precision
from taranis.snapshot_io import get_saved_steps, load_snapshot

NBLOCK_SCAN = 5      # fixed-dt configs: dt=0.01, t_end=0.2 -> ~20 full steps
NBLOCK_CFL = 4
T_SNAP = 100.0       # large: only the initial + final snapshot get written
T_END = 0.2

# shared grid/physics/forcing choices for all three configs (2D, single-process).
# diss/hyper passed FLAT (not as eqpars={}): ctx()'s cache key is
# tuple(sorted(kwargs.items())), which must stay hashable -- _rmhd_testing._ctor_kwargs
# folds these into eqpars only at Parameters-construction time.
_COMMON = dict(
    nx=64, ny=64, Lx=2.0 * np.pi, Ly=2.0 * np.pi, dims=2, eqtype="RMHD",
    diss=(1e-3, 1e-3), hyper=1,
    forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(5e-3, 5e-3),
    forcing_tau=1.0, fshell=(1, 3), forcing_seed=42,
)

# (name, extra Parameters kwargs, driver) -- the npz key prefixes and the five run.py
# code paths A2 will touch: simulate_scan/fixed-dt, simulate/fixed-dt,
# simulate_scan/adaptive-with-cfl_every (_cfl_block), simulate/adaptive-with-cfl_every
# (block_wrapped inside the while_loop) and simulate_scan with forcing off (no
# _advance_forcing, the path a particle push must mirror).
CONFIGS = (
    ("scan_fixed", dict(_COMMON, adaptive_timestep=False, dt=0.01), "scan"),
    ("while_fixed", dict(_COMMON, adaptive_timestep=False, dt=0.01), "while"),
    ("scan_cflblock", dict(_COMMON, adaptive_timestep=True, cfl_safety=0.5,
                            cfl_every=2), "scan"),
    ("while_cflblock", dict(_COMMON, adaptive_timestep=True, cfl_safety=0.5,
                            cfl_every=2), "while"),
    ("scan_unforced", dict(_COMMON, adaptive_timestep=False, dt=0.01,
                           forcing=False), "scan"),
)


def reference_path(precision=None):
    """The committed reference npz for a precision ("64"/"32"), resolved relative to this
    file (not the cwd: script mode runs from the repo root, the Savio suite from a per-job
    scratch dir). Defaults to the running session's precision."""
    prec = _precision.precision if precision is None else str(precision)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        f"particles_gate6_reference_fp{prec}.npz")


REFERENCE_PATH_FP64 = reference_path("64")
REFERENCE_PATH_FP32 = reference_path("32")


# orbax's OCDBT data-file leaves (default/d/<hash>, default/ocdbt.process_*/d/<hash>)
# carry a random hex name and a chunk count that both vary run-to-run -- collapsing
# hex leaves to a placeholder and deduping yields the deterministic structural skeleton
# (which subdirectories/keys exist), which is what "layout unchanged" actually means.
_HEX_LEAF = re.compile(r"^[0-9a-f]{16,}$")


def _snapshot_skeleton(step_dir):
    entries = set()
    for dp, _, fns in os.walk(step_dir):
        rel_dir = os.path.relpath(dp, step_dir)
        for fn in fns:
            leaf = "<hash>" if _HEX_LEAF.fullmatch(fn) else fn
            entries.add(leaf if rel_dir == "." else os.path.join(rel_dir, leaf))
    return np.array(sorted(entries))


def _state_leaves(snap_path, isnap, params):
    # "<name>:<shape>:<dtype>" for every leaf of the SimulationState restored from the
    # snapshot through the repo's own read path (load_snapshot -> bare
    # StandardCheckpointHandler, never a CheckpointManager: docs/checkpointing.md). A
    # witness that A2's particle checkpoint item is a SEPARATE item -- if a leaf were
    # added INSIDE the state item instead, this listing changes.
    # (forcing_key is a typed PRNG key array: read .shape/.dtype off it, never np.asarray)
    st = load_snapshot(isnap, snap_path, params)
    return np.array([f"{name}:{tuple(leaf.shape)}:{leaf.dtype}"
                     for name, leaf in zip(st._fields, st)])


def _ic(x, y):
    import jax.numpy as jnp
    phi = jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y)
    psi = jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)
    return jnp.stack([phi, psi], axis=0)


def config_ctx(kwargs):
    """(params, kgrid) for one recorded config. ctx() results are shared/frozen."""
    return ctx(**kwargs)


def run_config(kwargs, driver):
    """Runs the config with save=True into a fresh temp snapshot dir. Returns a dict
    of host arrays plus the sorted listing of the first written snapshot step dir.
    Never caches the returned state (donation)."""
    params, kgrid = config_ctx(kwargs)
    state0 = make_state(params, ic=_ic)
    with snap_dir() as d, managed_manager(params, d, nsnap=10) as mngr:
        if driver == "scan":
            nblock = NBLOCK_CFL if params.cfl_every > 1 else NBLOCK_SCAN
            end = jr.simulate_scan(state0, kgrid, params, nblock, T_SNAP, T_END, mngr,
                                   save=True)
        else:
            end = jr.simulate(state0, kgrid, params, T_SNAP, T_END, mngr, save=True)
        # first written step dir's structural skeleton plus the leaves of the state
        # restored from it -- witnesses that A2's separate particle checkpoint item
        # leaves the existing layout and the state item itself untouched when off.
        first_step = min(get_saved_steps(d))
        listing = _snapshot_skeleton(os.path.join(d, str(first_step)))
        leaves = _state_leaves(d, first_step, params)
        n_snap_files = len(get_saved_steps(d))
    Ek, Em = dg.energy(end, kgrid, params)
    return {
        "fields_final": np.asarray(end.fields),
        "forcing_state_final": np.asarray(end.forcing_state),
        "forcing_scale_final": np.asarray(end.forcing_scale),
        "forcing_key_final": np.asarray(jax.random.key_data(end.forcing_key)),
        "t_final": np.asarray(float(end.t)),
        "E_final": np.asarray(float(Ek) + float(Em)),
        "snapshot_listing": listing,  # plain unicode array, no pickle needed
        "state_leaves": leaves,       # ditto: "<name>:<shape>:<dtype>" per state leaf
        "n_snapshots": n_snap_files,
    }


def main(out_path=None):
    prec = _precision.precision
    out_path = out_path or reference_path(prec)
    out = {}
    summary = []
    for name, kwargs, driver in CONFIGS:
        result = run_config(kwargs, driver)
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
    params0, _ = config_ctx(dict(_COMMON, adaptive_timestep=False, dt=0.01))
    out["comm_backend"] = np.array(params0.comm_backend)
    out["platform"] = np.array(jax.default_backend())
    out["python_version"] = np.array(platform.python_version())
    # host identity: bitwise reproduction is only expected on the same machine/OS
    out["hostname"] = np.array(platform.node())
    out["host_platform"] = np.array(platform.platform())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    print(f"wrote {out_path}")
    for k, v in sorted(out.items()):
        if hasattr(v, "shape") and v.ndim:
            print(f"  {k}: {v.shape} {v.dtype}")

    print()
    print("summary (t_final, E_final, n_snapshots):")
    for name, result in summary:
        print(f"  {name}: t_final={result['t_final']:.6f}  "
              f"E_final={result['E_final']:.6e}  n_snapshots={result['n_snapshots']}")

    return out_path


if __name__ == "__main__":
    main(*sys.argv[1:])
