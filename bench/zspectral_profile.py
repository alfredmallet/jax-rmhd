# Where does the z_spectral=True step spend its extra time?
#
# docs/performance.md (B2) recorded that a z_spectral step costs ~1.8x a
# finite-difference-z step at the same grid, but not why. This times the bare RMHD step in
# both z modes at the same grid and then times the pieces of the step in isolation, so the
# gap can be attributed: transforms (rfftn/irfftn over (z,x,y) vs one rfft2 per plane),
# the linear propagator (putzer2 2x2 complex matrix exponential vs a real diagonal exp),
# and the finite-difference-z terms that only exist in the FD mode (z stencil + halo).
#
# Everything below the full-step table runs with hoist_propagator=False, i.e. dissects the
# legacy per-stage propagator evaluation; the full-step table also shows the hoisted
# (default) step so the two can be compared in one run.
#
# Not a test: not run by pytest/make test. usage:
#   TARANIS_PRECISION=64 python bench/zspectral_profile.py [nx] [nz] [nblock] [nrep]
#   TARANIS_PRECISION=64 python bench/zspectral_profile.py [nx] [nz] [nblock] [nrep] nzsweep
# nzsweep adds a step-only scan over nz = nz/4, nz/2, nz, 2*nz at fixed nx (the FD z
# stencil is O(nz), the z transform O(nz log nz), so the ratio should drift with nz).
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import taranis as jr
from taranis import grids
from taranis import _precision
from taranis import run as jrun
from taranis.physics import equation_registry, construct_rhs
from taranis.physics import rmhd
from taranis.physics.shared_physics import gradk
from taranis.propagators import get_propagator
from taranis.timestepping import get_scheme

L = 2 * np.pi
DT = 1e-3  # fixed (adaptive_timestep=False): same dt in both modes, no CFL allreduce noise
NX = int(sys.argv[1]) if len(sys.argv) > 1 else 128
NZ = int(sys.argv[2]) if len(sys.argv) > 2 else 16
NBLOCK = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NREP = int(sys.argv[4]) if len(sys.argv) > 4 else 12
NZSWEEP = "nzsweep" in sys.argv[1:]

COMMON = dict(Lx=L, Ly=L, Lz=L, dims=3, eqtype="RMHD", cfl_safety=0.5,
              adaptive_timestep=False, dt=DT, eqpars={"diss": (1e-4, 1e-4), "hyper": 2},
              forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(1.0, 1.0),
              forcing_tau=1.0, fshell=(1, 3), forcing_seed=1)


def ic(x, y, z):
    zc = jnp.cos(z)
    return jnp.stack([(jnp.cos(x + 1.4) + jnp.cos(y + 2.0)) * zc,
                      (jnp.cos(2 * x + 2.3) + 0.5 * jnp.cos(y + 6.2)) * zc])


def _median_ms(fn, *args, nrep=None):
    # median wall time of one jitted call, compiled and warmed first
    nrep = NREP if nrep is None else nrep
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(ts))


def _case(nx, nz, z_spectral, hoist=True):
    p = jr.Parameters(nx=nx, ny=nx, nz=nz, z_spectral=z_spectral, hoist_propagator=hoist,
                      **COMMON)
    kg = jr.setup_kgrids(p)
    state = jrun.initialize(ic, p)
    return p, kg, state


def _step_ms(p, kg, state):
    # ms per step of the production stepper (block of NBLOCK steps, jitted as one)
    stepper, scheme = get_scheme("lsrk33")
    step = jax.jit(jrun.block_of_steps, static_argnums=(2, 3, 4, 5))
    carry = step(state, kg, p, NBLOCK, scheme, stepper)
    jax.block_until_ready(carry.fields)
    ts = []
    for _ in range(NREP):
        t0 = time.perf_counter()
        carry = step(carry, kg, p, NBLOCK, scheme, stepper)
        jax.block_until_ready(carry.fields)
        ts.append(1e3 * (time.perf_counter() - t0) / NBLOCK)
    lowered = step.lower(state, kg, p, NBLOCK, scheme, stepper).compile()
    cost = lowered.cost_analysis()
    if isinstance(cost, (list, tuple)):
        cost = cost[0]
    flops = float(cost.get("flops", float("nan"))) / NBLOCK if cost else float("nan")
    return float(np.median(ts)), flops


def _pieces_ms(p, kg, state):
    # the RHS and its terms, each jitted on its own. kgrid and tau are passed as jit
    # ARGUMENTS, never closed over: as compile-time constants XLA folds exp(L*tau) away
    # entirely and the propagator times as free. These do not sum to the step (each pays
    # its own memory traffic where the fused step shares it) -- read them as the relative
    # weight of the same piece in the two modes.
    recipe = equation_registry[p.eqtype]
    rhs = construct_rhs(recipe)
    out = {}
    fields = state.fields

    out["rhs (all terms)"] = _median_ms(jax.jit(lambda s, k: rhs(s, k, p)[0]), state, kg)
    out["grad (gradk+ifft)"] = _median_ms(
        jax.jit(lambda s, k: recipe.grad_func(s, k, p)), state, kg)

    grads = jax.jit(lambda s, k: recipe.grad_func(s, k, p))(state, kg)
    halo = (jax.jit(lambda s, k: recipe.halo_start_func(s, k, p))(state, kg)
            if not p.z_spectral else None)
    for name, term in zip(("nonlinear", "fdlinear", "forcing"), recipe.term_funcs):
        out[f"term: {name}"] = _median_ms(
            jax.jit(lambda s, g, k, h: term(s, g, k, p, h)), state, grads, kg, halo)

    # transforms on their own, at the shapes the step actually uses
    real2 = jax.jit(lambda f: grids.ifft(f[:2], p))(fields)             # (2,nz,nx,ny) real
    gradk_arr = jax.jit(lambda f, k: gradk(jnp.stack([f[0], f[1], f[0], f[1]]), k))(fields, kg)
    out["fft  (2,nz,nx,ny)"] = _median_ms(jax.jit(lambda f: grids.fft(f, p)), real2)
    out["ifft (4,2,nz,nkx,nky)"] = _median_ms(jax.jit(lambda f: grids.ifft(f, p)), gradk_arr)

    # linear propagator: one apply_exp on the field stack, as an lsrk stage does (tau is
    # the stage's gamma*dt, a traced scalar there because the coefficients are scanned)
    def _exp(f, k, tau):
        return get_propagator(k, p).scaled(DT).apply_exp(f, tau)
    out["propagator apply_exp"] = _median_ms(jax.jit(_exp), fields, kg, jnp.float64(0.5))
    return out


def _swap_propagator(kg, p, kind):
    # Same grid and same physics terms, but the k-local linear operator L replaced by one
    # of the other propagator backend's shape, so the 2x2 table below separates "what the
    # transforms cost" from "what the propagator costs". kind="diag" drops the Alfven
    # waves (physically wrong, timing-valid); kind="putzer" gives the finite-difference-z
    # mode a wave-like off-diagonal it does not really have.
    from taranis import propagators
    diss, hyper = rmhd._diss_hyper(p)
    # z extent nz in BOTH modes (linear_fields allows zdim 1 or nz_local), so the
    # propagator does exactly the same arithmetic in the two rows and only the transforms
    # differ. Production finite-difference-z L is z-broadcast (zdim 1), so the "fd" rows
    # here sit slightly above the production fd step.
    nz = p.nz
    dphi = jnp.broadcast_to(-diss[0] * kg.ksq**hyper, (nz,) + kg.ksq.shape)
    dpsi = jnp.broadcast_to(-diss[1] * kg.ksq**hyper, (nz,) + kg.ksq.shape)
    if kind == "diag":
        L = jnp.stack([dphi, dpsi])
    else:
        # off-diagonal +-i*k, the shape of an Alfven coupling; i*kx keeps the rfft2
        # reality constraint (L(-kx,ky) = conj(L(kx,ky)) on the ky=0/Nyquist rows)
        kx = kg.kx.at[p.nx // 2].set(0.0) if p.nx % 2 == 0 else kg.kx   # Nyquist -> 0
        off = jnp.broadcast_to(1j * kx, dphi.shape)
        L = jnp.stack([jnp.stack([dphi + 0j, off]), jnp.stack([off, dpsi + 0j])])
    dtype = _precision.ctype if jnp.iscomplexobj(L) else _precision.ftype
    return kg._replace(lin_L=None, lin_m=None, lin_s=None, lin_dperp=None, lin_dz=None,
                       lin_kz=None)._replace(
        **propagators.linear_fields(L.astype(dtype), p))


def _ablation_ms(p, kg, state):
    # Ablation ladder inside the real step: strip one piece of the putzer2 propagator at a
    # time and re-time. Each variant is numerically WRONG on purpose -- it only answers
    # "what is that piece costing inside the fused, scanned step", which the isolated
    # timings above cannot (XLA fuses the propagator into the stage).
    # jax.clear_caches() is REQUIRED: the jaxpr trace cache is keyed on the function and
    # avals, so without it a monkeypatched propagator is silently ignored and every
    # variant re-reports the baseline.
    from taranis import propagators
    P = propagators.Putzer2Propagator
    orig_coeffs, orig_apply = P._coeffs, P.apply_exp

    def no_transcendental(self, tau):        # drop the complex sqrt/cosh/sinh
        one = jnp.ones_like(self.m)
        return one, tau*one

    def apply_mults_only(self, arr, tau):    # drop exp(m*tau) as well: 4 complex mults
        return jnp.stack([self.L[0,0]*arr[0] + self.L[0,1]*arr[1],
                          self.L[1,0]*arr[0] + self.L[1,1]*arr[1]])

    def apply_identity(self, arr, tau):      # no propagator work at all
        return arr

    out = {}
    try:
        jax.clear_caches()
        out["baseline (production)"] = _step_ms(p, kg, state)[0]
        P._coeffs = no_transcendental
        jax.clear_caches()
        out["- complex sqrt/cosh/sinh"] = _step_ms(p, kg, state)[0]
        P._coeffs = orig_coeffs
        P.apply_exp = apply_mults_only
        jax.clear_caches()
        out["- exp(m*tau) too"] = _step_ms(p, kg, state)[0]
        P.apply_exp = apply_identity
        jax.clear_caches()
        out["- the 2x2 apply too"] = _step_ms(p, kg, state)[0]
    finally:
        P._coeffs, P.apply_exp = orig_coeffs, orig_apply
        jax.clear_caches()
    return out


def main():
    print(f"nx=ny={NX}, nz={NZ}, nblock={NBLOCK}, nrep={NREP}, dt={DT}, scheme=lsrk33, "
          f"jax={jax.__version__}, backend={jax.default_backend()}, "
          f"TARANIS_PRECISION={_precision.precision}")
    # hoist_propagator=False is the legacy per-stage evaluation the attribution below
    # dissects; True (the default) is production
    res = {}
    for label, zs in (("fd", False), ("spec", True)):
        p, kg, state = _case(NX, NZ, zs, hoist=False)
        ms, flops = _step_ms(p, kg, state)
        res[label] = dict(step=ms, flops=flops, pieces=_pieces_ms(p, kg, state))
        ph, kgh, sth = _case(NX, NZ, zs, hoist=True)
        res[label]["hoisted"] = _step_ms(ph, kgh, sth)[0]

    print(f"\n== full step, {NX}^2 x {NZ} ==")
    print(f"{'mode':10s} {'ms/step':>10s} {'GFLOP/step':>12s} {'hoisted ms':>11s}")
    for label in ("fd", "spec"):
        print(f"{label:10s} {res[label]['step']:10.3f} {res[label]['flops']/1e9:12.3f} "
              f"{res[label]['hoisted']:11.3f}")
    print(f"z_spectral / finite-difference-z (unhoisted): "
          f"{res['spec']['step']/res['fd']['step']:.2f}x wall, "
          f"{res['spec']['flops']/res['fd']['flops']:.2f}x flops; "
          f"hoisted: {res['spec']['hoisted']/res['fd']['hoisted']:.2f}x wall")

    print(f"\n== isolated pieces (ms per call; each jitted alone) ==")
    print(f"{'piece':24s} {'fd':>9s} {'spec':>9s} {'spec/fd':>9s} {'delta':>9s}")
    for name in res["fd"]["pieces"]:
        a, b = res["fd"]["pieces"][name], res["spec"]["pieces"][name]
        ratio = b / a if a > 0 else float("inf")
        print(f"{name:24s} {a:9.3f} {b:9.3f} {ratio:9.2f} {b-a:+9.3f}")

    print(f"\n== propagator attribution: same grid, L swapped between backends ==")
    print(f"{'z mode':10s} {'L backend':12s} {'ms/step':>10s}")
    attrib = {}
    for label, zs in (("fd", False), ("spec", True)):
        p, kg, state = _case(NX, NZ, zs, hoist=False)
        for kind in ("diag", "putzer"):
            ms, _ = _step_ms(p, _swap_propagator(kg, p, kind), state)
            attrib[(label, kind)] = ms
            print(f"{label:10s} {kind:12s} {ms:10.3f}")
    print(f"putzer2 - diagonal: fd {attrib[('fd','putzer')]-attrib[('fd','diag')]:+.2f} ms, "
          f"spec {attrib[('spec','putzer')]-attrib[('spec','diag')]:+.2f} ms/step")
    print(f"spectral - finite-difference transforms at fixed backend: "
          f"diag {attrib[('spec','diag')]-attrib[('fd','diag')]:+.2f} ms, "
          f"putzer {attrib[('spec','putzer')]-attrib[('fd','putzer')]:+.2f} ms/step")

    print(f"\n== ablation ladder inside the z_spectral step (each row drops one piece) ==")
    ps, kgs, sts = _case(NX, NZ, True, hoist=False)
    lad = _ablation_ms(ps, kgs, sts)
    prev = None
    print(f"{'variant':28s} {'ms/step':>10s} {'piece cost':>12s}")
    for name, ms in lad.items():
        piece = "" if prev is None else f"{prev - ms:+.2f}"
        print(f"{name:28s} {ms:10.2f} {piece:>12s}")
        prev = ms
    print(f"stripped propagator vs finite-difference-z step: "
          f"{list(lad.values())[-1] - res['fd']['step']:+.2f} ms/step "
          f"(this is what the (z,x,y) transforms actually cost)")

    if NZSWEEP:
        print(f"\n== step vs nz at nx={NX} ==")
        print(f"{'nz':>5s} {'fd ms':>9s} {'spec ms':>9s} {'spec/fd':>9s}")
        for nz in (max(4, NZ // 4), max(4, NZ // 2), NZ, 2 * NZ):
            row = []
            for zs in (False, True):
                p, kg, state = _case(NX, nz, zs, hoist=False)
                row.append(_step_ms(p, kg, state)[0])
            print(f"{nz:5d} {row[0]:9.3f} {row[1]:9.3f} {row[1]/row[0]:9.2f}")


if __name__ == "__main__":
    main()
