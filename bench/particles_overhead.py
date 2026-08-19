# Per-step wall-time overhead of test-particle co-stepping (plans/TESTPART_PLAN.md sec 4
# budget: <=15-20%). Times jax.jit(block_of_steps) directly -- no simulate_scan, no
# snapshot I/O, no moments sidecar -- so this isolates exactly the particle_fields +
# per-ensemble push cost against the bare RMHD step.
#
# Also reports the fixed part on its own: jitted particle_fields against one solver step,
# which is the piece the plan's budget is about (the rest is the O(N) push gather).
#
# Not a test: not run by pytest/make test. usage:
#   TARANIS_PRECISION=64 python bench/particles_overhead.py [nblock] [nrep] [case]
# case is one of 2d (default, 256^2), 3dfd (128^2 x 16, finite-difference z) and 3dspec
# (the same grid with z_spectral=True).
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import taranis as jr
from taranis import run as jrun
from taranis.timestepping import get_scheme
from taranis.particles.fields import particle_fields
from taranis.particles.state import init_particles

L = 2 * np.pi
DT = 1e-3  # fixed (adaptive_timestep=False): determinism, same dt on/off
NBLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 12
CASE = sys.argv[3] if len(sys.argv) > 3 else "2d"

# plan sec 2 baseline: q/m ~ 10-20 puts rho ~ 2-4 dx at 256^2; vth=1
QM, VTH, N_PART = 15.0, 1.0, 32768

# 3D runs a smaller perp grid at nz = 16, so one case fits a laptop at fp64; what the
# report compares is the RATIO of particle_fields to the solver step, not the ms.
_GEOM = {"2d": dict(nx=256, ny=256, dims=2),
        "3dfd": dict(nx=128, ny=128, dims=3, nz=16, Lz=L),
        "3dspec": dict(nx=128, ny=128, dims=3, nz=16, Lz=L, z_spectral=True)}
if CASE not in _GEOM:
    raise SystemExit(f"unknown case {CASE!r}; expected one of {list(_GEOM)}")
N = _GEOM[CASE]["nx"]

COMMON = dict(Lx=L, Ly=L, eqtype="RMHD", cfl_safety=0.5,
             adaptive_timestep=False, dt=DT, eqpars={"diss": (1e-4, 1e-4), "hyper": 2},
             forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(1.0, 1.0),
             forcing_tau=1.0, fshell=(1, 3), forcing_seed=1, **_GEOM[CASE])

_default = {"qm": QM, "init": {"kind": "maxwellian", "vth": VTH}}
# plan sec 3's three standard ensembles: ideal-Ohm (default), full d(psi)/dt, E=0 control
_ENSEMBLES_3 = [dict(_default),
               dict(_default, ez_resistive=True, ez_forcing=True),
               dict(_default, eperp=False, ez_ideal=False)]
_ENSEMBLES_1 = [dict(_default)]  # default mask only: no optional iffts in particle_fields


def ic(x, y, z=None):
    zc = 1.0 if z is None else jnp.cos(z)
    return jnp.stack([(jnp.cos(x + 1.4) + jnp.cos(y + 2.0)) * zc,
                     (jnp.cos(2 * x + 2.3) + 0.5 * jnp.cos(y + 6.2)) * zc])


def _make_case(label, ensembles):
    # block_of_steps returns final_state when particles are off, (carry, ys) when on
    # (plans/TESTPART_PLAN.md sec 4) -- step()/leaf() below normalize that away.
    particles = None if ensembles is None else dict(seed=0, n=N_PART, ensembles=ensembles)
    p = jr.Parameters(particles=particles, **COMMON)
    kg = jr.setup_kgrids(p)
    state = jrun.initialize(ic, p)
    stepper, scheme = get_scheme("lsrk33")
    step_jit = jax.jit(jrun.block_of_steps, static_argnums=(2, 3, 4, 5))
    if particles is None:
        carry = state
        leaf = lambda c: c.fields
        step = lambda c: step_jit(c, kg, p, NBLOCK, scheme, stepper)
    else:
        carry = (state, init_particles(p))
        leaf = lambda c: c[0].fields
        step = lambda c: step_jit(c, kg, p, NBLOCK, scheme, stepper)[0]
    carry = step(carry)  # compile + warm block
    jax.block_until_ready(leaf(carry))
    return label, carry, step, leaf


def _time_block(carry, step, leaf):
    t0 = time.perf_counter()
    carry = step(carry)
    jax.block_until_ready(leaf(carry))
    return carry, time.perf_counter() - t0


def _time_pieces(p, kg, state):
    # the fixed (transform) part on its own, default mask and with the optional pieces
    out = {}
    for label, kw in (("fields", {}), ("fields+opt", dict(resistive=True, forcing=True))):
        f = jax.jit(lambda s, kw=kw: particle_fields(s, kg, p, **kw))
        jax.block_until_ready(f(state).ez_ideal)
        ts = []
        for _ in range(NREP):
            t0 = time.perf_counter()
            jax.block_until_ready(f(state).ez_ideal)
            ts.append(1e3 * (time.perf_counter() - t0))
        out[label] = float(np.median(ts))
    return out


def main():
    cases = [_make_case("off", None),
            _make_case("on3", _ENSEMBLES_3),
            _make_case("on1", _ENSEMBLES_1)]
    times = {label: [] for label, *_ in cases}
    carries = {label: carry for label, carry, _, _ in cases}
    steppers = {label: (step, leaf) for label, _, step, leaf in cases}
    # interleave: one block per case per rep, round-robin, so a load spike hits every
    # case about equally rather than skewing whichever case runs last
    for _ in range(NREP):
        for label, *_ in cases:
            step, leaf = steppers[label]
            carries[label], dt = _time_block(carries[label], step, leaf)
            times[label].append(dt)

    def per_step_ms(label):
        return [1e3 * t / NBLOCK for t in times[label]]

    med = {label: float(np.median(per_step_ms(label))) for label, *_ in cases}
    iqr = {label: float(np.percentile(per_step_ms(label), 75) -
                        np.percentile(per_step_ms(label), 25)) for label, *_ in cases}
    p_off = jr.Parameters(particles=None, **COMMON)
    pieces = _time_pieces(p_off, jr.setup_kgrids(p_off), jrun.initialize(ic, p_off))
    print(f"case={CASE}, grid={_GEOM[CASE]}, nblock={NBLOCK}, nrep={NREP}, dt={DT}, "
         f"n_particles/ensemble={N_PART}, jax={jax.__version__}, "
         f"backend={jax.default_backend()}")
    print(f"{'case':6s} {'ms/step':>10s} {'iqr':>8s} {'ratio vs off':>14s}")
    for label, *_ in cases:
        ratio = med[label] / med["off"]
        print(f"{label:6s} {med[label]:10.4f} {iqr[label]:8.4f} {ratio:14.3f}")
    over3 = 100.0 * (med["on3"] / med["off"] - 1.0)
    over1 = 100.0 * (med["on1"] / med["off"] - 1.0)
    print(f"overhead: on3 (3 ensembles: ideal, full, E=0) = {over3:+.1f}%, "
         f"on1 (1 default-mask ensemble) = {over1:+.1f}%")
    for label, ms in pieces.items():
        print(f"particle_fields [{label}]: {ms:.3f} ms = "
             f"{100.0 * ms / med['off']:.1f}% of the solver step")


if __name__ == "__main__":
    main()
