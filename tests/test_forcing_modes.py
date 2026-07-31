# momentum <-> elsasser forcing relationship (docs/TESTING_PLAN.md Phase 5).
#
#   - elsasser with EQUAL envelopes and psi=0 fields: the two z+/z- normalization
#     denominators are computed from identical rows, so f_psi = (f_plus-f_minus)/2
#     vanishes EXACTLY, and f_phi reproduces the momentum-mode result. A same-RNG
#     full-run comparison of the two modes is IMPOSSIBLE by construction: n_ou
#     differs (1 vs 2), so ou_update draws differently-shaped noise and the RNG
#     streams diverge -- the envelopes are therefore injected by hand here.
#   - asymmetric forcing_power_elsasser=(0.6,0.2): with per-stage normalization
#     and diss=0, energy must grow at eps_p+eps_m = 0.8 and cross-helicity
#     <grad phi . grad psi> at eps_p-eps_m = 0.4. A swapped f_plus/f_minus flips
#     the cross-helicity SIGN -- caught loudly.
#   - 2D momentum forcing from a quiescent start is pure hydro: psi's only 2D
#     source vanishes and it stays EXACTLY zero (CLAUDE.md).
#
# All 2D, no collectives -- runs identically per rank under MPI, but only listed
# serially in the Savio manifest. Script: `python tests/test_forcing_modes.py`.
from _rmhd_testing import bootstrap, checks, ctx, make_state, managed_manager, snap_dir, zero_ic_2d

bootstrap()

import jax
import jax.numpy as jnp
import numpy as np

import jax_rmhd as jr
from jax_rmhd import snapshot_io
from jax_rmhd.physics import rmhd, shared_physics

_F = dict(nx=32, ny=32, dims=2, diss=(0.0, 0.0), forcing=True,
          forcing_power=0.7, forcing_power_elsasser=(0.35, 0.35),
          forcing_tau=0.5, fshell=(1, 5), forcing_seed=3,
          forcing_norm_per_step=False)  # per-stage normalization: exact injection


def _phi_only_ic(x, y):
    phi = jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y)
    return jnp.stack([phi, jnp.zeros_like(phi)], axis=0)


def test_elsasser_equal_envelopes_reduce_to_momentum():
    p_e, kg_e = ctx(forcing_mode="elsasser", **_F)
    p_m, kg_m = ctx(forcing_mode="momentum", **_F)
    _, ctype = snapshot_io.get_precision_types()

    # a properly symmetrized envelope from one OU kick, then copy z+ onto z-
    fs0 = jnp.zeros((p_e.n_ou, 2, p_e.nx, p_e.ny // 2 + 1), dtype=ctype)
    fs, _ = shared_physics.ou_update(fs0, jax.random.key(11), 0.01, p_e, kg_e)
    fs_eq = fs.at[1].set(fs[0])

    st_e = make_state(p_e, ic=_phi_only_ic)._replace(forcing_state=fs_eq)
    st_m = make_state(p_m, ic=_phi_only_ic)._replace(forcing_state=fs_eq[:1])
    F_e = np.asarray(rmhd.ForcingTerm(st_e, None, kg_e, p_e))
    F_m = np.asarray(rmhd.ForcingTerm(st_m, None, kg_m, p_m))

    # z+/z- denominators come from perp_inner_product_batch (row-wise sums) while
    # momentum uses perp_inner_product (one flat sum): equal values, different
    # summation order -- so f_phi agreement is round-off, not bitwise.
    tol = 1e-13 if jax.config.jax_enable_x64 else 1e-5
    rel = float(np.max(np.abs(F_e[0] - F_m[0])) / np.max(np.abs(F_m[0])))
    with checks() as c:
        c.check("equal elsasser envelopes: f_psi is EXACTLY zero (bitwise)",
                not np.any(F_e[1]), f"max|f_psi|={np.max(np.abs(F_e[1])):.3e}")
        c.check(f"equal elsasser envelopes: f_phi equals the momentum-mode result "
                f"(rel {rel:.2e} < {tol:.0e})", rel < tol)
        c.check("forcing term is nonzero (test not vacuous)",
                float(np.max(np.abs(F_m[0]))) > 0.0)


def test_asymmetric_elsasser_injection_rates():
    p, kg = ctx(forcing_mode="elsasser",
                **dict(_F, forcing_power_elsasser=(0.6, 0.2)))
    st = make_state(p, ic=zero_ic_2d)
    with snap_dir() as d, managed_manager(p, d, nsnap=2) as m:
        end = jr.simulate(st, kg, p, t_snap=1.0, t_end=0.5, mngr=m, save=False)
    phik, psik = end.fields[0], end.fields[1]
    E = 0.5 * float(shared_physics.perp_inner_product(phik, phik, kg, p)
                    + shared_physics.perp_inner_product(psik, psik, kg, p))
    Hc = float(shared_physics.perp_inner_product(phik, psik, kg, p))
    t = float(end.t)
    rate_E, rate_H = E / t, Hc / t
    with checks() as c:
        c.check(f"energy injection rate {rate_E:.3f} within 30% of "
                f"0.6+0.2 = 0.8", 0.7 * 0.8 < rate_E < 1.3 * 0.8)
        c.check(f"cross-helicity rate {rate_H:.3f} within 40% of "
                f"0.6-0.2 = 0.4 (sign catches swapped f_plus/f_minus)",
                0.6 * 0.4 < rate_H < 1.4 * 0.4)


def test_2d_momentum_quiescent_keeps_psi_zero():
    # production-style settings (norm_per_step True) -- psi must stay EXACTLY 0
    p, kg = ctx(forcing_mode="momentum", **dict(_F, forcing_norm_per_step=True))
    st = make_state(p, ic=zero_ic_2d)
    with snap_dir() as d, managed_manager(p, d, nsnap=2) as m:
        end = jr.simulate(st, kg, p, t_snap=1.0, t_end=0.3, mngr=m, save=False)
    f = np.asarray(end.fields)
    with checks() as c:
        c.check("psi stays EXACTLY zero (pure hydro -- its only 2D source vanishes)",
                not np.any(f[1]), f"max|psi_k|={np.max(np.abs(f[1])):.3e}")
        c.check("phi is forced away from zero", float(np.max(np.abs(f[0]))) > 0.0)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
