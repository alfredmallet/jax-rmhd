# Particle carry state, its configuration schema and the per-step ensemble push.
#
# ParticleState rides NEXT TO SimulationState as a run.py carry tuple; it is fp64 at every
# TARANIS_PRECISION and its positions are UNFOLDED (interp.gather folds them mod L). The z
# axis is carried everywhere; Phase A implements nz_local == 1 only.
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import boris
from .fields import FIELD_PIECES, assemble, particle_fields, resolve_mask


class ParticleState(NamedTuple):
    x: jnp.ndarray   # (n_ens, n, 3) float64, unfolded positions
    v: jnp.ndarray   # (n_ens, n, 3) float64


# per-ensemble scan-ys diagnostics, in the order `moments` stacks them
MOMENTS = ("vperp2", "vz2", "vz")
NMOM = len(MOMENTS)


def moments(pstate):
    """Per-ensemble means of v_x^2+v_y^2, v_z^2 and v_z -> (n_ens, NMOM) float64."""
    v = pstate.v.astype(jnp.float64)
    return jnp.stack([jnp.mean(v[:, :, 0] ** 2 + v[:, :, 1] ** 2, axis=1),
                      jnp.mean(v[:, :, 2] ** 2, axis=1),
                      jnp.mean(v[:, :, 2], axis=1)], axis=1)


########################
# configuration schema #
########################

_TOP_KEYS = ("seed", "n", "substeps", "B0", "init_on_restart", "ensembles")
# init kind -> {key: default}; None marks a required key
_INIT_KEYS = {"maxwellian": {"vth": None}, "ring": {"vperp": None, "vz": 0.0}}


def _int(value, name):
    try:
        ok = not isinstance(value, bool) and int(value) == value
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"particles: {name} must be an int, got {value!r}")
    return int(value)


def _bool(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"particles: {name} must be a bool, got {value!r}")
    return bool(value)


def _float(value, name):
    # every float cast in the schema goes through here, so a bad value names its key
    # instead of surfacing as a bare TypeError from float()
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"particles: {name} must be a float, got {value!r}") from None


def normalize_config(raw):
    """Validate params.particles and return its normalized form: plain python types, all
    defaults filled, each ensemble carrying the FULL resolved field mask under "mask".

    Idempotent: a normalized config is itself valid input (ensembles may carry "mask")."""
    if not isinstance(raw, dict):
        raise ValueError(f"particles must be a dict of particle configuration, got {raw!r}")
    unknown = sorted(k for k in raw if k not in _TOP_KEYS)
    if unknown:
        raise ValueError(f"unknown particles key(s) {unknown}; expected a subset of {list(_TOP_KEYS)}")
    if "n" not in raw:
        raise ValueError("particles['n'] (particles PER ENSEMBLE) is required")
    n = _int(raw["n"], "'n'")
    if n <= 0:
        raise ValueError(f"particles['n'] must be > 0, got {n}")
    substeps = _int(raw.get("substeps", 1), "'substeps'")
    if substeps < 1:
        raise ValueError(f"particles['substeps'] must be >= 1, got {substeps}")
    ensembles = raw.get("ensembles")
    if not isinstance(ensembles, (list, tuple)) or not ensembles:
        raise ValueError(f"particles['ensembles'] must be a non-empty list of ensemble dicts, "
                         f"got {ensembles!r}")
    return {"seed": _int(raw.get("seed", 0), "'seed'"),
            "n": n,
            "substeps": substeps,
            "B0": _float(raw.get("B0", 1.0), "'B0'"),
            "init_on_restart": _bool(raw.get("init_on_restart", False), "'init_on_restart'"),
            "ensembles": tuple(_normalize_ensemble(e, i) for i, e in enumerate(ensembles))}


def _normalize_ensemble(ens, i):
    where = f"particles['ensembles'][{i}]"
    if not isinstance(ens, dict):
        raise ValueError(f"{where} must be a dict, got {ens!r}")
    allowed = ("qm", "init", "mask") + FIELD_PIECES
    unknown = sorted(k for k in ens if k not in allowed)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {where}; expected a subset of {list(allowed)}")
    if "qm" not in ens:
        raise ValueError(f"{where} needs 'qm' (charge-to-mass ratio)")
    return {"qm": _float(ens["qm"], f"{where}['qm']"),
            "init": _normalize_init(ens.get("init"), where),
            "mask": _normalize_mask(ens, where)}


def _normalize_mask(ens, where):
    # a "mask" dict (as normalize_config itself emits) merged with the inline per-piece
    # keys; a piece given both ways must agree. The raw per-piece keys are consumed.
    inline = {k: _bool(ens[k], f"{where}[{k!r}]") for k in FIELD_PIECES if k in ens}
    given = ens.get("mask")
    if given is None:
        return resolve_mask(inline)
    if not isinstance(given, dict):
        raise ValueError(f"{where}['mask'] must be a dict of field-piece flags, got {given!r}")
    merged = {k: _bool(v, f"{where}['mask'][{k!r}]") for k, v in given.items()}
    for k, v in inline.items():
        if merged.get(k, v) != v:
            raise ValueError(f"{where} sets {k!r} to {v} inline but to {merged[k]} in "
                             f"'mask'; they must agree")
        merged[k] = v
    return resolve_mask(merged)


def _normalize_init(init, where):
    if not isinstance(init, dict) or "kind" not in init:
        raise ValueError(f"{where}['init'] must be a dict with a 'kind' "
                         f"({' or '.join(map(repr, _INIT_KEYS))}), got {init!r}")
    kind = init["kind"]
    if kind not in _INIT_KEYS:
        raise ValueError(f"unknown init kind {kind!r} in {where}; expected one of {list(_INIT_KEYS)}")
    spec = _INIT_KEYS[kind]
    unknown = sorted(k for k in init if k != "kind" and k not in spec)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {where}['init'] (kind {kind!r}); "
                         f"expected a subset of {['kind'] + list(spec)}")
    out = {"kind": kind}
    for key, default in spec.items():
        if key in init:
            out[key] = _float(init[key], f"{where}['init'][{key!r}]")
        elif default is None:
            raise ValueError(f"{where}['init'] (kind {kind!r}) needs {key!r}")
        else:
            out[key] = float(default)
    for key in ("vth", "vperp"):
        if key in out and out[key] <= 0:
            raise ValueError(f"{where}['init'][{key!r}] must be > 0, got {out[key]}")
    return out


##############################
# initialization and pushing #
##############################

def init_particles(params):
    """Fresh ParticleState from params.particles["seed"]. The ONLY use of the particle RNG
    stream: positions uniform over the box (z = 0 in 2D), velocities per ensemble init."""
    cfg = params.particles
    n = cfg["n"]
    key = jax.random.key(cfg["seed"])
    xs, vs = [], []
    for e, ens in enumerate(cfg["ensembles"]):
        kx, kv = jax.random.split(jax.random.fold_in(key, e))
        u = jax.random.uniform(kx, (n, 2), dtype=jnp.float64)
        xs.append(jnp.stack([u[:, 0] * params.Lx, u[:, 1] * params.Ly,
                             jnp.zeros((n,), dtype=jnp.float64)], axis=1))
        init = ens["init"]
        if init["kind"] == "maxwellian":
            # vth is the 1-D thermal speed: each component ~ N(0, vth^2)
            vs.append(init["vth"] * jax.random.normal(kv, (n, 3), dtype=jnp.float64))
        else:
            theta = jax.random.uniform(kv, (n,), dtype=jnp.float64, maxval=2.0 * jnp.pi)
            vs.append(jnp.stack([init["vperp"] * jnp.cos(theta),
                                 init["vperp"] * jnp.sin(theta),
                                 jnp.full((n,), init["vz"], dtype=jnp.float64)], axis=1))
    return ParticleState(x=jnp.stack(xs), v=jnp.stack(vs))


def template(params):
    """ParticleState of ShapeDtypeStructs, for checkpoint restores."""
    like = jax.ShapeDtypeStruct((params.n_ens, params.particles["n"], 3), jnp.float64)
    return ParticleState(x=like, v=like)


def push_ensembles(pstate, state, kgrid, params, dt):
    """Advance every ensemble by dt in the fields of `state` (frozen over the step). One
    field assembly per ensemble from one shared PFields; the mask loop is static python."""
    cfg = params.particles
    ensembles = cfg["ensembles"]
    pf = particle_fields(state, kgrid, params,
                         resistive=any(e["mask"]["ez_resistive"] for e in ensembles),
                         forcing=any(e["mask"]["ez_forcing"] for e in ensembles))
    xs, vs = [], []
    for e, ens in enumerate(ensembles):
        E, B = assemble(pf, ens["mask"], B0=cfg["B0"])
        x, v = boris.push(pstate.x[e], pstate.v[e], E, B, ens["qm"], dt, params,
                          substeps=cfg["substeps"])
        xs.append(x)
        vs.append(v)
    return ParticleState(x=jnp.stack(xs), v=jnp.stack(vs))
