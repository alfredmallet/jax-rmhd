import jax
import jax.numpy as jnp
from typing import NamedTuple,Tuple
from .propagators import get_propagator
from . import _precision

# Standard RK4 with integrating factor.
# Problem is it uses a lot of memory on k1-k4.
# I think it is always better to use the LSRK schemes

def rk_advance(state,kgrid,params,rhs,set_timestep,scheme=None,dt_override=None):
    #RK4 substep 1
    k1, grads = rhs(state,kgrid,params)
    # dt_override: dt already computed for a whole cfl_every block by run.py
    if dt_override is not None:
        dt = dt_override
    elif params.adaptive_timestep==True:
        dt = set_timestep(grads,params)
    else:
        dt = params.dt
    # exact linear propagator (dissipation for RMHD); exp(L*tau) composes over sub-stages
    # since the same fixed L commutes with itself
    prop = get_propagator(kgrid,params)
    f1 = prop.apply_exp(state.fields + 0.5 * dt * k1, dt/2)
    #RK4 substep 2
    # NB: forcing_state/forcing_key are threaded through unchanged at every sub-stage via
    # _replace (they're only updated once per full step, in run.block_of_steps).
    k2,_ = rhs(state._replace(t=state.t+dt/2.0,fields=f1),kgrid,params)
    f2 = prop.apply_exp(state.fields,dt/2) + 0.5*dt*k2
    #RK4 substep 3
    k3,_ = rhs(state._replace(t=state.t+dt/2.0,fields=f2),kgrid,params)
    # coef=dt: the dt multiplies the propagator FACTOR (pre-P1 `dt*diss_half*k3` order)
    f3 = prop.apply_exp(state.fields,dt) + prop.apply_exp(k3,dt/2,dt)
    #RK4 final step
    k4,_ = rhs(state._replace(t=state.t+dt,fields=f3),kgrid,params)
    f_end = prop.apply_exp(state.fields,dt) + (dt/6.0) * (prop.apply_exp(k1,dt)
            + 2.0*prop.apply_exp(k2,dt/2) + 2.0*prop.apply_exp(k3,dt/2) + k4)
    return state._replace(t=state.t + dt, fields=f_end)

# object defining low-storage Runge-Kutta (lsrk) schemes
class LSRK_Scheme(NamedTuple):
    alphas: Tuple[float,...]
    betas: Tuple[float,...]
    gammas: Tuple[float,...]

# LSRK timestepper: includes an integrating factor for the dissipative terms.
# params.lsrk_scan=True: lax.scan (default); =False or (unrolled, could help on some GPU)
def lsrk_advance(state, kgrid, params, rhs, set_timestep, scheme, dt_override=None):
    init_rhs,grads = rhs(state,kgrid,params)
    if dt_override is not None:
        dt = dt_override
    elif params.adaptive_timestep==True:
        dt = set_timestep(grads,params)
    else:
        dt = params.dt

    # pre-scale the linear operator by dt once per step, so a stage's exponent is
    # (L*dt)*gamma -- the op order the pre-P1 `exp(kgrid.hdiss*dt*gamma)` used
    prop = get_propagator(kgrid,params).scaled(dt)

    if params.lsrk_scan:
        return _lsrk_scan_stages(state, kgrid, params, rhs, scheme, init_rhs, dt, prop)

    current_state = state
    delta = None
    for istage in range(len(scheme.alphas)):
        alpha, beta, gamma = scheme.alphas[istage], scheme.betas[istage], scheme.gammas[istage]
        # stage 0 reuses init_rhs (also used for dt above); alphas[0]=0 so delta starts at dt*rhs
        stage_rhs = init_rhs if istage == 0 else rhs(current_state,kgrid,params)[0]
        delta = prop.apply_exp(dt*stage_rhs if istage == 0 else alpha*delta + dt*stage_rhs, gamma)
        # forcing fields threaded through unchanged via _replace (see rk_advance comment above).
        current_state = current_state._replace(t=current_state.t + gamma*dt,
                                               fields=prop.apply_exp(current_state.fields,gamma) + beta*delta)
    return current_state

# used if params.lsrk_scan=True
def _lsrk_scan_stages(state, kgrid, params, rhs, scheme, init_rhs, dt, prop):
    # dtype=ftype: these are STRONG arrays (unlike the unrolled loop's python-float
    # coefficients, which are weak and adopt the field dtype) and they multiply delta/fields
    # inside the scan, so unpinned they would upcast the whole step under x64.
    alphas_arr = jnp.array(scheme.alphas, dtype=_precision.ftype)
    betas_arr = jnp.array(scheme.betas, dtype=_precision.ftype)
    gammas_arr = jnp.array(scheme.gammas, dtype=_precision.ftype)

    init_delta = jnp.zeros_like(state.fields)
    init_carry = (state,init_delta)

    stage_pars = (alphas_arr, betas_arr, gammas_arr, jnp.arange(len(scheme.alphas)))

    def scan_stage_func(carry,stage_vals):
        current_state, delta = carry
        alpha, beta, gamma, istage = stage_vals

        stage_rhs = jax.lax.cond(istage == 0,lambda: init_rhs,
                                 lambda: rhs(current_state,kgrid,params)[0])

        next_delta = prop.apply_exp(alpha * delta + dt * stage_rhs, gamma)
        next_fields = prop.apply_exp(current_state.fields, gamma) + beta*next_delta
        next_t = current_state.t + gamma*dt
        # forcing_state/forcing_key threaded through unchanged
        return (current_state._replace(t=next_t,fields=next_fields),next_delta), None

    (final_state, _), _ = jax.lax.scan(scan_stage_func,init_carry,stage_pars)

    return final_state

# ---------------------------------------------------------------- CB-IMEX (plans/GDI_PLAN.md P3)
#
# Low-storage IMEX-RK schemes of Cavaglieri & Bewley, JCP 286:172-193 (2015)
# (http://robotics.ucsd.edu/pubs/CB15.pdf; coefficients from their eqs 24, 28a, 30, 32c).
# ALL of the equation set's k-local linear operator L is treated IMPLICITLY here (dissipation
# included) -- no exponential anywhere in this path -- and the explicit part is exactly what
# construct_rhs returns (nonlinear + forcing + any non-k-local linear term, e.g. RMHD's
# finite-difference z derivatives). Each implicit stage is one propagator solve_shifted.
#
# Why these, and not the IF-LSRK production schemes: an integrating factor treats L exactly
# but misweights the nonlinear forcing of a STIFFLY DAMPED mode when |L|*dt >~ 1, giving
# u ~ dt*N instead of the correct quasi-static balance u ~ N/|L|. Every scheme below has an
# L-stable, stiffly accurate (a^IM_{s,i} = b_i) implicit part, which recovers that balance.
# The flip side: an L-stable solve DAMPS oscillatory linear terms at |omega|*dt >~ 1, so
# never point these at a wave-dominated L (spectral-z RMHD's +-i*kz) at large dt.
#
# All four have b^IM = b^EX = b and c^IM = c^EX = c (that is the paper's design constraint
# which makes the coupling conditions tractable), c_1 = 0 and c_s = 1 (FSAL).
class IMEX_Scheme(NamedTuple):
    a_im: Tuple[Tuple[float,...],...]   # dense DIRK tableau, lower triangular incl. diagonal
    a_ex: Tuple[Tuple[float,...],...]   # dense ERK tableau, strictly lower triangular
    b: Tuple[float,...]                 # shared assembly weights (b^IM = b^EX)
    c: Tuple[float,...]                 # shared abscissae (c^IM = c^EX)
    order: int
    structure: str                      # "2R" or "3R": the paper's low-storage tableau class
    registers: int                      # live field-sized arrays the stepper below keeps

# The dense tableaux above are the transcription of record; the steppers read ONLY the
# low-storage subset (the sub-diagonal, the diagonal, b and c), and tests/test_imex.py
# checks that the entries they ignore really are the b_j the [2R]/[3R] structure requires.
def _scheme_entries(scheme):
    # unpack + the one structural assumption both steppers bake in: an EXPLICIT first stage
    # (a^IM_{1,1} = 0), which is what makes stage 1's "solve" the identity
    if scheme.a_im[0][0] != 0.0:
        raise ValueError("the CB-IMEX steppers assume an explicit first implicit stage "
                         f"(a^IM_(1,1) = 0), got {scheme.a_im[0][0]!r}")
    return scheme.a_im, scheme.a_ex, scheme.b, scheme.c, len(scheme.b)

def _stepper_dt(state, kgrid, params, rhs, set_timestep, dt_override):
    # shared preamble: stage-1 explicit derivative g(x_n) (c_1 = 0 in every scheme here,
    # so this doubles as the CFL evaluation) and the step's dt
    init_rhs, grads = rhs(state,kgrid,params)
    if dt_override is not None:
        dt = dt_override
    elif params.adaptive_timestep==True:
        dt = set_timestep(grads,params)
    else:
        dt = params.dt
    return init_rhs, dt

# [2R] IMEX-RK, three-register implementation (Cavaglieri & Bewley eq. 19).
# REGISTERS: 3 field-sized arrays live at once -- x (the solution accumulator, which is the
# state's own fields buffer), y (the explicit stage derivative / partial stage sum) and z
# (the implicit stage derivative). The paper's TWO-register variant (their eq. 20) is not
# used: it trades the third register for two extra nonlinear evaluations and two extra L
# products per stage, which their own footnote 4 rules out for spectral methods (every
# nonlinear evaluation here is a batch of FFTs).
# Scan (params.lsrk_scan=True, default) and unrolled stage loops, like lsrk_advance: the
# structurally-different first stage (init_rhs, the skipped z_1) sits entirely BEFORE the
# k = 1..s-1 loop, whose body is uniform -- so unlike _lsrk_scan_stages no lax.cond is
# needed. The two loops agree to round-off, not bitwise (same XLA-fusion class as lsrk).
def imex2r_advance(state, kgrid, params, rhs, set_timestep, scheme, dt_override=None):
    init_rhs, dt = _stepper_dt(state, kgrid, params, rhs, set_timestep, dt_override)
    prop = get_propagator(kgrid,params)
    a_im, a_ex, b, c, s = _scheme_entries(scheme)
    x = state.fields
    y = init_rhs                    # g(u_1), u_1 = x_n since c_1 = 0
    first_col = b[0] != 0.0 or a_im[1][0] != 0.0   # static: does the DIRK first column act?
    if first_col:
        # generic first stage. a^IM_{1,1} = 0, so its "solve" is the identity: z_1 = L x.
        z = prop.apply_L(x)
        x = x + (b[0]*dt)*z + (b[0]*dt)*y
    else:
        # b_1 = a^IM_{2,1} = 0 (the whole first column of the DIRK tableau vanishes): z_1 is
        # multiplied by zero everywhere, so it is never formed -- s-1 implicit stages.
        z = None
    if params.lsrk_scan:
        # zeros stand in for the never-formed z_1: its k=1 coefficient a^IM_{2,1}-b_1 is
        # exactly 0.0, so the scan body's extra multiply-add contributes exact zeros
        return _imex2r_scan_stages(state, kgrid, params, rhs, prop, scheme, dt, x, y,
                                   z if first_col else jnp.zeros_like(x))
    for k in range(1,s):
        # partial stage sum, from the [2R] structure: x already carries b_j for j <= k-2
        acc = x + ((a_ex[k][k-1] - b[k-1])*dt)*y
        if z is not None:
            acc = acc + ((a_im[k][k-1] - b[k-1])*dt)*z
        z = prop.solve_shifted(prop.apply_L(acc), a_im[k][k]*dt)   # z_k = L u_k
        # forcing_state/forcing_key/forcing_scale thread through unchanged via _replace
        stage = state._replace(t=state.t + c[k]*dt, fields=acc + (a_im[k][k]*dt)*z)
        y = rhs(stage,kgrid,params)[0]
        x = x + (b[k]*dt)*z + (b[k]*dt)*y
    return state._replace(t=state.t + dt, fields=x)

# used if params.lsrk_scan=True: the k = 1..s-1 loop of imex2r_advance as a lax.scan over
# the per-stage coefficient 5-tuple. Carry is exactly the 3 registers (x, y, z).
def _imex2r_scan_stages(state, kgrid, params, rhs, prop, scheme, dt, x, y, z):
    a_im, a_ex, b, c, s = _scheme_entries(scheme)
    # dtype=ftype for the same reason as _lsrk_scan_stages: the SCANNED tableau entries are
    # strong arrays that multiply fields. (The unrolled loop above reads the same tableaus as
    # python floats -- weak, no pin needed, and the differences a_ex[k][k-1]-b[k-1] are still
    # formed in python/float64 exactly as before, then rounded once on the way into ftype.)
    ft_ = _precision.ftype
    stage_pars = (jnp.array([a_ex[k][k-1] - b[k-1] for k in range(1,s)], dtype=ft_),
                  jnp.array([a_im[k][k-1] - b[k-1] for k in range(1,s)], dtype=ft_),
                  jnp.array([a_im[k][k] for k in range(1,s)], dtype=ft_),
                  jnp.array([b[k] for k in range(1,s)], dtype=ft_),
                  jnp.array([c[k] for k in range(1,s)], dtype=ft_))

    def scan_stage_func(carry, stage_vals):
        xk, yk, zk = carry
        d_ex, d_im, aii, bk, ck = stage_vals
        acc = xk + (d_ex*dt)*yk + (d_im*dt)*zk
        z_new = prop.solve_shifted(prop.apply_L(acc), aii*dt)
        # forcing_state/forcing_key/forcing_scale thread through unchanged via _replace
        stage = state._replace(t=state.t + ck*dt, fields=acc + (aii*dt)*z_new)
        y_new = rhs(stage,kgrid,params)[0]
        return (xk + (bk*dt)*z_new + (bk*dt)*y_new, y_new, z_new), None

    (x, _, _), _ = jax.lax.scan(scan_stage_func, (x, y, z), stage_pars)
    return state._replace(t=state.t + dt, fields=x)

# [3R] IMEX-RK, four-register implementation (Cavaglieri & Bewley eq. 21). Same contract as
# imex2r_advance; the extra register buys stage order two for the implicit part.
# UNROLLED ONLY (params.lsrk_scan not read here): the y-update needs the LOOKAHEAD
# coefficients a_(k+1,k-1) and skips the last stage, so a scan would need a cond -- not
# worth it for the one [3R] scheme (imexcb3f).
# REGISTERS: 4 -- x (accumulator), y (the pending next-stage combination), z_im, z_ex.
# NB the (z_ex - y)/a^EX_{k,k-1} below is the paper's trick for recovering dt*g_{k-1} without
# a fifth register; it is a cancellation, so it costs ~eps*|y|/a^EX_{k,k-1} of accuracy in
# that one term (a few ulp for the coefficients used here).
def imex3r_advance(state, kgrid, params, rhs, set_timestep, scheme, dt_override=None):
    init_rhs, dt = _stepper_dt(state, kgrid, params, rhs, set_timestep, dt_override)
    prop = get_propagator(kgrid,params)
    a_im, a_ex, b, c, s = _scheme_entries(scheme)
    y = state.fields                # x_n, held for the pending stage combination
    z_im = prop.apply_L(y)          # a^IM_{1,1} = 0: the stage-1 solve is the identity
    z_ex = init_rhs
    x = y + (b[0]*dt)*z_im + (b[0]*dt)*z_ex
    for k in range(1,s):
        z_ex = y + (a_ex[k][k-1]*dt)*z_ex
        if k < s-1:
            y = (x + ((a_im[k+1][k-1] - b[k-1])*dt)*z_im
                   + ((a_ex[k+1][k-1] - b[k-1])/a_ex[k][k-1])*(z_ex - y))
        z_ex = z_ex + (a_im[k][k-1]*dt)*z_im
        z_im = prop.solve_shifted(prop.apply_L(z_ex), a_im[k][k]*dt)
        stage = state._replace(t=state.t + c[k]*dt, fields=z_ex + (a_im[k][k]*dt)*z_im)
        z_ex = rhs(stage,kgrid,params)[0]
        x = x + (b[k]*dt)*z_im + (b[k]*dt)*z_ex
    return state._replace(t=state.t + dt, fields=x)

def _cb3c():
    # IMEXRKCB3c (eq. 28a): (3,4)-stage, L-stable, SSP explicit part with the widest
    # negative-real-axis ERK stability of the family (delta = 1/54, |sigma| <= 1 to z = -6).
    c2  = 3375509829940/4525919076317
    c3  = 272778623835/1039454778728
    a22 = 3375509829940/4525919076317
    a32 = -11712383888607531889907/32694570495602105556248
    a33 = 566138307881/912153721139
    b = (0.0, 673488652607/2334033219546, 493801219040/853653026979,
         184814777513/1389668723319)
    ex43 = 1660544566939/2334033219546
    return IMEX_Scheme(
        a_im = ((0.0, 0.0, 0.0, 0.0),
                (0.0, a22, 0.0, 0.0),
                (b[0], a32, a33, 0.0),
                (b[0], b[1], b[2], b[3])),        # stiff accuracy: last row = b
        a_ex = ((0.0, 0.0, 0.0, 0.0),
                (c2, 0.0, 0.0, 0.0),
                (b[0], c3, 0.0, 0.0),
                (b[0], b[1], ex43, 0.0)),
        b = b, c = (0.0, c2, c3, 1.0), order = 3, structure = "2R", registers = 3)

def _cb3f():
    # IMEXRKCB3f (eq. 32c): (3,4)-stage [3R], L-stable, and the only stage-order-two member
    # implemented here -- no order reduction when the stiff limit turns the system into an
    # index-1 DAE. Not SSP; its ERK stability region is IMEXRKCB3c's.
    c2, c3 = 49/50, 1/25
    b = (-2179897048956/603118880443, 99189146040/891495457793,
         6064140186914/1415701440113, 146791865627/668377518349)
    return IMEX_Scheme(
        a_im = ((0.0, 0.0, 0.0, 0.0),
                (c2/2, c2/2, 0.0, 0.0),
                (-785157464198/1093480182337, -30736234873/978681420651,
                 983779726483/1246172347126, 0.0),
                b),                               # stiff accuracy: last row = b
        a_ex = ((0.0, 0.0, 0.0, 0.0),
                (c2, 0.0, 0.0, 0.0),
                (13244205847/647648310246, 13419997131/686433909488, 0.0, 0.0),
                (b[0], 231677526244/1085522130027, 3007879347537/683461566472, 0.0)),
        b = b, c = (0.0, c2, c3, 1.0), order = 3, structure = "3R", registers = 4)

#Code to set up a couple of common lsrk schemes

def _get_LSRK54_gammas():
    c2 = 1432997174477 / 9575080441755
    c3 = 2526269341429 / 6820363962896
    c4 = 2006345519317 / 3224310063776
    c5 = 2802321613138 / 2924317926251
    return (c2, c3 - c2, c4 - c3, c5 - c4, 1.0 - c5)

_scheme_registry = {
    "rk44": (rk_advance, None),
    "lsrk33": (lsrk_advance,
               LSRK_Scheme(
                   alphas = (0.0, -5.0 / 9.0, -153.0 / 128.0),
                   betas  = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0),
                   gammas = (1.0 / 3.0, 5.0 / 12.0, 1.0 / 4.0),
                   )), #This is the original Williamson 1980 RK3 scheme
    "lsrk54": (lsrk_advance,
               LSRK_Scheme(
                   alphas = (
                       0.0,
                       -567301805773 / 1357537059087,
                       -2404267990393 / 2016746695238,
                       -3550918686646 / 2091501179385,
                       -1275806237668 / 842570457699
                       ),
                   betas = (
                       1432997174477 / 9575080441755,
                       5161836677717 / 13612068292357,
                       1720146321549 / 2090206949498,
                       3134564353537 / 4481467310338,
                       2277821191437 / 14882151754819
                       ),
                   gammas = _get_LSRK54_gammas(),
                   )), #5-stage, 4th-order scheme from Carpenter & Kennedy 1994
    # CB-IMEX: L implicit (L-stable), term_funcs explicit. See the section above.
    "imexcb2": (imex2r_advance,
                IMEX_Scheme(          # IMEXRKCB2 (eq. 24): (2,3)-stage, L-stable, SSP explicit
                    a_im = ((0.0, 0.0, 0.0),
                            (0.0, 2/5, 0.0),
                            (0.0, 5/6, 1/6)),     # stiff accuracy: last row = b
                    a_ex = ((0.0, 0.0, 0.0),
                            (2/5, 0.0, 0.0),
                            (0.0, 1.0, 0.0)),
                    b = (0.0, 5/6, 1/6), c = (0.0, 2/5, 1.0),
                    order = 2, structure = "2R", registers = 3)),
    "imexcb3e": (imex2r_advance,
                 IMEX_Scheme(         # IMEXRKCB3e (eq. 30): (3,4)-stage, L-stable, ERK
                     a_im = ((0.0, 0.0, 0.0, 0.0),   # accuracy maximized (delta = 1/24, so
                             (0.0, 1/3, 0.0, 0.0),   # the ERK stability region is RK4's)
                             (0.0, 1/2, 1/2, 0.0),
                             (0.0, 3/4, -1/4, 1/2)), # stiff accuracy: last row = b
                     a_ex = ((0.0, 0.0, 0.0, 0.0),
                             (1/3, 0.0, 0.0, 0.0),
                             (0.0, 1.0, 0.0, 0.0),
                             (0.0, 3/4, 1/4, 0.0)),
                     b = (0.0, 3/4, -1/4, 1/2), c = (0.0, 1/3, 1.0, 1.0),
                     order = 3, structure = "2R", registers = 3)),
    "imexcb3c": (imex2r_advance, _cb3c()),
    "imexcb3f": (imex3r_advance, _cb3f()),
}

def get_scheme(name: str):
    if name not in _scheme_registry:
        raise ValueError(f"Unknown scheme '{name}'. Available: {list(_scheme_registry.keys())}")
    return _scheme_registry[name]