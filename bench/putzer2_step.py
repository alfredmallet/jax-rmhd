# What does the putzer2 backend cost inside an adaptive step?
#
# putzer2 serves GDI's integrating-factor path and z_spectral RMHD at nu != eta (nu == eta
# takes the separable backend). Under adaptive cfl_every=1 nothing is frozen over a block,
# so hoist_propagator cannot amortise the coefficient evaluation and the per-stage cost is
# paid in full -- that is the configuration timed here. Reported per case: ms/step and the
# transcendental inventory of the optimized HLO of one jitted block.
#
# Not a test: not run by pytest/make test. usage:
#   TARANIS_PRECISION=64 python bench/putzer2_step.py [--nblock N] [--nrep N] [--cases a,b]
import argparse
import re
import time

import numpy as np
import jax
import jax.numpy as jnp

import taranis as jr
from taranis import run as jrun
from taranis.timestepping import get_scheme

L = 2*np.pi
# op names to count in the optimized HLO; the putzer2 coefficients lower into these
_TRANSCENDENTAL = ("exponential", "sqrt", "rsqrt", "cosine", "sine", "log", "tanh",
                   "divide", "power")


def _gdi_2d(nx):
    p = jr.Parameters(nx=nx, ny=nx, Lx=L, Ly=L, dims=2, eqtype="GDI", cfl_safety=0.4,
                      adaptive_timestep=True, cfl_every=1, dt=1e-3,
                      eqpars=dict(Ln=2.0, nu_in=0.3, v0=1.0, gpar_fac=1.0, diss=1e-4,
                                  hyper=1))

    def ic(x, y):
        return jnp.stack([jnp.cos(x + 1.4) + jnp.cos(y + 2.0),
                          jnp.cos(2*x + 2.3) + 0.5*jnp.cos(y + 6.2)])
    return p, ic


def _rmhd_zspec(nx, nz):
    # nu != eta: the diagonal blocks differ, so this stays on putzer2
    p = jr.Parameters(nx=nx, ny=nx, nz=nz, Lx=L, Ly=L, Lz=L, dims=3, z_spectral=True,
                      eqtype="RMHD", cfl_safety=0.4, adaptive_timestep=True, cfl_every=1,
                      dt=1e-3, eqpars={"diss": (1e-4, 2e-4), "hyper": 2})

    def ic(x, y, z):
        zc = jnp.cos(z)
        return jnp.stack([(jnp.cos(x + 1.4) + jnp.cos(y + 2.0))*zc,
                          (jnp.cos(2*x + 2.3) + 0.5*jnp.cos(y + 6.2))*zc])
    return p, ic


CASES = {
    "gdi512_lsrk33": (lambda: _gdi_2d(512), "lsrk33"),
    "gdi256_lsrk33": (lambda: _gdi_2d(256), "lsrk33"),
    "rmhd_zspec_128x32_lsrk33": (lambda: _rmhd_zspec(128, 32), "lsrk33"),
    "rmhd_zspec_128x32_lsrk54": (lambda: _rmhd_zspec(128, 32), "lsrk54"),
}


def _hlo_counts(lowered):
    text = lowered.compile().as_text()
    counts = {}
    for op in _TRANSCENDENTAL:
        n = len(re.findall(rf"=\s*\S+\s+{op}\(", text))
        if n:
            counts[op] = n
    return counts


def _time_case(name, nblock, nrep):
    build, schemestr = CASES[name]
    p, ic = build()
    kg = jr.setup_kgrids(p)
    state = jrun.initialize(ic, p)
    stepper, scheme = get_scheme(schemestr)
    step = jax.jit(jrun.block_of_steps, static_argnums=(2, 3, 4, 5))
    carry = step(state, kg, p, nblock, scheme, stepper)
    jax.block_until_ready(carry.fields)
    ts = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        carry = step(carry, kg, p, nblock, scheme, stepper)
        jax.block_until_ready(carry.fields)
        ts.append(1e3*(time.perf_counter() - t0)/nblock)
    counts = _hlo_counts(step.lower(state, kg, p, nblock, scheme, stepper))
    return float(np.median(ts)), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nblock", type=int, default=20)
    ap.add_argument("--nrep", type=int, default=9)
    ap.add_argument("--cases", default=",".join(CASES))
    args = ap.parse_args()
    print(f"jax {jax.__version__}  {jax.default_backend()}  nblock={args.nblock} "
          f"nrep={args.nrep}")
    for name in args.cases.split(","):
        ms, counts = _time_case(name, args.nblock, args.nrep)
        inv = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"{name:28s} {ms:8.3f} ms/step   per block: {inv}")


if __name__ == "__main__":
    main()
