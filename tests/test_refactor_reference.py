# Phase-0 bitwise and compiled-graph reference of plans/REFACTOR_PLAN.md: the twelve
# solver paths in tests/_gen_refactor_reference.py must keep producing bit-identical
# fields and times through the four structural refactor phases (runtime split, uniform run
# carry, typed linear operator, typed equation interface), AND must keep compiling to the
# same HLO opcode histogram. Nothing in those phases changes a number or the amount of
# compute, so a difference here is a bug or a §9 decision -- never a reason to regenerate.
# The one allowed histogram change is Phase G's: fewer add/broadcast/constant ops in the
# configs with an inactive term, and nothing else (§0.4).
#
# The configs, the initial condition, the driver and the HLO parse live in the generator;
# this module imports them, so the checked-in reference and the comparison cannot drift
# apart.
#
# Bitwise reproduction and HLO layout are host-specific, so the comparison only runs when
# the recorded machine/jax/backend/python match -- print + return otherwise, per the house
# convention (the same logic as tests/test_particles_coupled.py's gate 6a). Both precision
# sessions read their own reference pair.
#
# pytest: `pytest tests/test_refactor_reference.py`. Script:
# `python tests/test_refactor_reference.py`.
from _rmhd_testing import bootstrap, checks, mpi_size

bootstrap()

import functools
import json
import os
import platform

import jax
import numpy as np

from _gen_refactor_reference import (CONFIGS, hlo_reference_path, reference_path,
                                     run_config)


@functools.lru_cache(maxsize=None)
def _reference():
    """The committed npz + HLO sidecar, plus whether this host can be expected to
    reproduce them (same machine, jax, backend and python)."""
    ref = np.load(reference_path(), allow_pickle=False)
    with open(hlo_reference_path()) as f:
        hlo = json.load(f)
    host_ok = (str(ref["hostname"]) == platform.node()
               and str(ref["jax_version"]) == jax.__version__
               and str(ref["platform"]) == jax.default_backend()
               and str(ref["python_version"]) == platform.python_version())
    return ref, hlo, host_ok


@functools.lru_cache(maxsize=None)
def _result(index):
    # keyed by position in CONFIGS: the GDI rows carry an unhashable eqpars dict
    _name, kwargs, schemestr = CONFIGS[index]
    return run_config(kwargs, schemestr)


def _results():
    # one run per config, shared by the two tests
    return [(CONFIGS[i][0], _result(i)) for i in range(len(CONFIGS))]


def _skip_reason():
    if mpi_size() > 1:
        return "the z_spectral and 2D configs are single-process only"
    _ref, _hlo, host_ok = _reference()
    if not host_ok:
        ref = _ref
        return (f"recorded on {ref['hostname']} / jax {ref['jax_version']} / "
                f"{ref['platform']} / py {ref['python_version']}, this host is "
                f"{platform.node()} / jax {jax.__version__} / {jax.default_backend()} / "
                f"py {platform.python_version()} (reproduction is host-specific)")
    return None


def test_solver_paths_match_the_phase0_reference():
    """Every recorded config's final fields and time, bitwise."""
    why = _skip_reason()
    if why:
        print(f"[SKIP] refactor reference fields -- {why}")
        return
    ref, _hlo, _ok = _reference()
    print(f"comparing against {os.path.basename(reference_path())} "
          f"(git {str(ref['git_commit'])[:9]}, precision {ref['precision']})")
    with checks() as c:
        for name, got in _results():
            want_f, want_t = ref[f"{name}_fields"], ref[f"{name}_t"]
            same = np.array_equal(got["fields"], want_f)
            if same:
                detail = ""
            else:
                d = np.abs(got["fields"] - want_f)
                detail = (f"{int(np.count_nonzero(d))} of {d.size} elements differ, "
                          f"max |diff| {float(np.max(d)):.3e}, rel "
                          f"{float(np.max(d) / np.max(np.abs(want_f))):.3e}")
            c.check(f"{name}: fields bitwise identical to the reference", same, detail)
            c.check(f"{name}: t bitwise identical to the reference",
                    got["t"] == want_t, f"{got['t']!r} vs {want_t!r}")


def _opcode_diff(got, want):
    """The opcodes whose counts differ, as 'opcode want->got' -- what moved, nothing else."""
    keys = sorted(set(got["opcodes"]) | set(want["opcodes"]))
    moved = [(k, want["opcodes"].get(k, 0), got["opcodes"].get(k, 0)) for k in keys
             if want["opcodes"].get(k, 0) != got["opcodes"].get(k, 0)]
    return ", ".join(f"{k} {w}->{g}" for k, w, g in moved)


def test_compiled_graphs_match_the_phase0_reference():
    """Every recorded config's optimized-HLO opcode histogram, exactly: same output and
    same instruction mix means the same compute, which is the plan's speed gate."""
    why = _skip_reason()
    if why:
        print(f"[SKIP] refactor reference HLO -- {why}")
        return
    _ref, hlo, _ok = _reference()
    print(f"comparing against {os.path.basename(hlo_reference_path())} "
          f"(git {hlo['git_commit'][:9]}, precision {hlo['precision']})")
    with checks() as c:
        for name, got in _results():
            want = hlo["configs"][name]
            g = got["hlo"]
            c.check(f"{name}: {want['total_instructions']} HLO instructions",
                    g["total_instructions"] == want["total_instructions"],
                    f"{g['total_instructions']} vs {want['total_instructions']}")
            c.check(f"{name}: {want['fusions']} fusions",
                    g["fusions"] == want["fusions"],
                    f"{g['fusions']} vs {want['fusions']}")
            c.check(f"{name}: opcode histogram unchanged",
                    g["opcodes"] == want["opcodes"], _opcode_diff(g, want))


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
