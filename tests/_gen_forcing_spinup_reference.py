# Generator for tests/data/forcing_spinup_reference.npz -- the pre-fix RNG/trajectory
# reference required by plans/FORCING_SPINUP_PLAN.md Phase 0 ("record an RNG/trajectory
# reference on main ... so the 'streams unchanged, trajectories changed only via scale'
# claim is checkable").
#
# GENERATED 2026-08-08 on PRE-FIX main, for plans/FORCING_SPINUP_PLAN.md Phase 0.
# fp64 ONLY (single process, lsrk33). Three quiescent-start forced configs (2D elsasser,
# 3D elsasser z-spectral, 3D elsasser finite-difference-z), each advanced 20 full steps
# from a zero IC.
#
# What each array means / what is expected to hold after the fix:
#   - "<name>_forcing_state_final" is the bitwise RNG-stream WITNESS. The fix (quadratic
#     self-energy-aware scale normalization) adds NO new RNG draws -- F2 = <|f_raw|^2> is
#     computed deterministically from the already-drawn f_raw -- so this array must come
#     out BITWISE IDENTICAL when regenerated on the post-fix tree. That equality is the
#     actual Phase-0 gate.
#   - "<name>_E" (total energy = E_kin+E_mag per step), "<name>_scale" (forcing_scale per
#     step), "<name>_t", and "<name>_fields_final" ARE EXPECTED TO CHANGE after the fix
#     (that is the whole point of the fix: the pinned-at-smax quiescent kick goes away).
#     They are stored here only as a before/after comparison baseline, not as invariants.
#
# Not a test module (leading underscore keeps pytest away). Run as:
#   python3 tests/_gen_forcing_spinup_reference.py [out.npz]
# (TARANIS_PRECISION defaults to 64 below if unset; must be 64 for this reference.)
import os
os.environ.setdefault("TARANIS_PRECISION", "64")
assert os.environ["TARANIS_PRECISION"] == "64", \
    "forcing_spinup_reference must be generated at fp64 (TARANIS_PRECISION=64)"

import subprocess
import sys

import jax
import numpy as np

import taranis as jr
from taranis import diagnostics as dg
from taranis import run
from taranis.timestepping import get_scheme

N_STEPS = 20

# committed reference, resolved relative to this file (not the cwd).
REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "forcing_spinup_reference.npz")

# Shared forcing/dissipation/grid choices across all three configs (diagnosis table in
# FORCING_SPINUP_PLAN.md: 2D elsasser, diss=(1e-5,1e-5) hyper=3, tau=1, fshell=(1,3),
# seed 42, fp64, lsrk33, eps_tot=1e-2 i.e. forcing_power_elsasser=(5e-3,5e-3)).
_COMMON = dict(
    eqpars={"diss": (1e-5, 1e-5), "hyper": 3},
    cfl_safety=0.5,
    forcing=True,
    forcing_mode="elsasser",
    forcing_power_elsasser=(5e-3, 5e-3),
    forcing_tau=1.0,
    fshell=(1, 3),
    forcing_seed=42,
    forcing_scale_max=1.0,
    adaptive_timestep=True,  # the quiescent velocity-floor dt is central to the bug
)

# (name, Parameters kwargs) -- the npz key prefixes.
CONFIGS = (
    ("e2d", dict(_COMMON, nx=64, ny=64, Lx=2.0 * np.pi, Ly=2.0 * np.pi, dims=2)),
    ("e3dz", dict(_COMMON, nx=32, ny=32, nz=32, Lx=2.0 * np.pi, Ly=2.0 * np.pi,
                  Lz=2.0 * np.pi, dims=3, z_spectral=True)),
    ("e3dfd", dict(_COMMON, nx=32, ny=32, nz=32, Lx=2.0 * np.pi, Ly=2.0 * np.pi,
                   Lz=2.0 * np.pi, dims=3, z_spectral=False)),
)


def _zero_ic_2d(x, y):
    import jax.numpy as jnp
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape))


def _zero_ic_3d(x, y, z):
    import jax.numpy as jnp
    return jnp.zeros((2,) + jnp.broadcast_shapes(x.shape, y.shape, z.shape))


def run_config(kwargs):
    """20 full lsrk33 steps from a zero IC, single-step calls to run.block_of_steps
    (mirrors run.simulate_scan's jit pattern: static_argnums=(2,3,4,5), no donation so
    the per-step host readback below stays valid). Returns per-step E/scale/t lists and
    the final state."""
    params = jr.Parameters(**kwargs)
    kgrid = jr.setup_kgrids(params)
    ic = _zero_ic_3d if params.spatial_dimensions == 3 else _zero_ic_2d
    state = jr.initialize(ic, params)
    # production parity with simulate_scan (harmless here: forcing_state is zero at t=0,
    # so step 1's forcing term is 0*scale regardless of what this recomputes it to).
    state = run._refresh_forcing_scale(state, kgrid, params)

    stepper, scheme = get_scheme("lsrk33")
    step_fn = jax.jit(run.block_of_steps, static_argnums=(2, 3, 4, 5))

    E_list, scale_list, t_list = [], [], []
    for _ in range(N_STEPS):
        state = step_fn(state, kgrid, params, 1, scheme, stepper)
        Ek, Em = dg.energy(state, kgrid, params)
        E_list.append(float(Ek) + float(Em))
        scale_list.append(np.asarray(state.forcing_scale))
        t_list.append(float(state.t))

    return {
        "E": np.asarray(E_list),
        "scale": np.asarray(scale_list),
        "t": np.asarray(t_list),
        "forcing_state_final": np.asarray(state.forcing_state),
        "fields_final": np.asarray(state.fields),
        "t_final": np.asarray(float(state.t)),
    }


def main(out_path=None):
    out_path = out_path or REFERENCE_PATH
    out = {}
    summary = []
    for name, kwargs in CONFIGS:
        result = run_config(kwargs)
        for key, val in result.items():
            out[f"{name}_{key}"] = val
        summary.append((name, result))

    try:
        out["git_commit"] = np.array(subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))).strip())
    except Exception:
        out["git_commit"] = np.array("unknown")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    print(f"wrote {out_path}")
    for k, v in sorted(out.items()):
        if hasattr(v, "dtype") and v.ndim:
            print(f"  {k}: {v.shape} {v.dtype}")

    print()
    print("summary (E after step 2, E after step 20, scale after step 2, "
          "forcing_state_final shape):")
    for name, result in summary:
        print(f"  {name}: E[1]={result['E'][1]:.6e}  E[19]={result['E'][19]:.6e}  "
              f"scale[1]={result['scale'][1]}  "
              f"forcing_state_final.shape={result['forcing_state_final'].shape}")


if __name__ == "__main__":
    main(*sys.argv[1:])
