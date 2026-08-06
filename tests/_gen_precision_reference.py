# Generator for tests/data/precision_fp32_reference.npz — the pre-change RNG/step
# reference required by PRECISION_PLAN.md A5 (RNG-stream check + before/after
# comparison). It was RUN ON THE PRE-CHANGE TREE (commit recorded in the npz);
# re-running it after the precision change must reproduce the recorded arrays
# bitwise (except t's dtype). Run as:
#   RMHD_PRECISION=32 python tests/_gen_precision_reference.py [out.npz]
# Not a test module (leading underscore keeps pytest away).
from _rmhd_testing import bootstrap, ctx, make_state, managed_manager, snap_dir, zero_ic

bootstrap()

import subprocess
import sys

import numpy as np

import jax_rmhd as jr

N_STEPS = 10


def _run(shell_noise, mode):
    params, kgrid = ctx(diss=(0.0, 0.0), forcing=True, forcing_mode=mode,
                        forcing_power=1.0, forcing_tau=0.5, fshell=(1, 5),
                        forcing_seed=1, forcing_shell_noise=shell_noise)
    state0 = make_state(params, ic=zero_ic)
    with snap_dir() as d, managed_manager(params, d, nsnap=10) as mngr:
        end = jr.simulate_scan(state0, kgrid, params, N_STEPS, 1.0, 1.0, mngr,
                               save=False)
    return end


def main(out_path="tests/data/precision_fp32_reference.npz"):
    out = {}
    for name, shell_noise, mode in (("default", False, "momentum"),
                                    ("shellnoise", True, "momentum"),
                                    ("elsasser", False, "elsasser")):
        end = _run(shell_noise, mode)
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
