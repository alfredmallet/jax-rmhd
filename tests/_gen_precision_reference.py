# Generator for tests/data/precision_fp32_reference.npz — the pre-change RNG/step
# reference required by PRECISION_PLAN.md A5 (RNG-stream check + before/after
# comparison). It was RUN ON THE PRE-CHANGE TREE (commit recorded in the npz);
# re-running it after the precision change must reproduce the recorded arrays
# to within fp32 round-off (t's dtype, and hence the exact dt handed to ou_update,
# changed by design — see tests/test_precision_dtypes.py for the assertion and the
# mechanism). Run as:
#   RMHD_PRECISION=32 python tests/_gen_precision_reference.py [out.npz]
# Not a test module (leading underscore keeps pytest away).
#
# CONFIGS / config_ctx / run_config are the SHARED definition of the three recorded
# runs: tests/test_precision_dtypes.py imports them so the checked-in reference and
# the test that reads it can never drift apart. Changing any of them invalidates the
# committed npz — regenerate it on the pre-change tree, not on the current one.
from _rmhd_testing import bootstrap, ctx, make_state, managed_manager, snap_dir, zero_ic

bootstrap()

import os
import subprocess
import sys

import numpy as np

import taranis as jr

N_STEPS = 10

# (name, forcing_shell_noise, forcing_mode) — the npz key prefixes.
CONFIGS = (("default", False, "momentum"),
           ("shellnoise", True, "momentum"),
           ("elsasser", False, "elsasser"))

# committed reference, resolved relative to this file (not the cwd: script mode runs
# from the repo root, the Savio suite from a per-job scratch dir).
REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "precision_fp32_reference.npz")


def config_ctx(shell_noise, mode):
    """(params, kgrid) for one recorded config. ctx() results are shared/frozen.

    forcing_scale_max is PINNED at 1.0, the Parameters default when the reference npz
    was recorded. The default moved to 1e4 on 2026-08-09 (saturated-state smax pinning
    fix): from the zero IC these runs start at, the scale sits ON the cap for the first
    steps, so inheriting the new default injects ~1e4x more on step 1 and moves the
    10-step fields by 14-19% -- through the top of _REF_BAND_20260808, which was
    measured at cap 1.0. The comparison isolates the 2026-08-08 normalization change;
    every other knob must stay at its recorded value. (The 1e4 cap's own semantics are
    pinned by test_forcing_smoke and test_forcing_spinup, not here.)"""
    return ctx(diss=(0.0, 0.0), forcing=True, forcing_mode=mode,
               forcing_power=1.0, forcing_tau=0.5, fshell=(1, 5),
               forcing_seed=1, forcing_shell_noise=shell_noise,
               forcing_scale_max=1.0)


def run_config(shell_noise, mode):
    """The recorded run. Returns a FRESH end state (never cache one: donation)."""
    params, kgrid = config_ctx(shell_noise, mode)
    state0 = make_state(params, ic=zero_ic)
    with snap_dir() as d, managed_manager(params, d, nsnap=10) as mngr:
        end = jr.simulate_scan(state0, kgrid, params, N_STEPS, 1.0, 1.0, mngr,
                               save=False)
    return end


def main(out_path="tests/data/precision_fp32_reference.npz"):
    out = {}
    for name, shell_noise, mode in CONFIGS:
        end = run_config(shell_noise, mode)
        out[f"{name}_fields"] = np.asarray(end.fields)
        out[f"{name}_forcing_state"] = np.asarray(end.forcing_state)
        out[f"{name}_forcing_scale"] = np.asarray(end.forcing_scale)
        out[f"{name}_t"] = np.asarray(end.t)
    try:
        out["git_commit"] = np.array(subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip())
    except Exception:
        out["git_commit"] = np.array("unknown")
    np.savez(out_path, **out)
    print(f"wrote {out_path}")
    for k, v in out.items():
        if hasattr(v, "dtype") and v.ndim:
            print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    main(*sys.argv[1:])
