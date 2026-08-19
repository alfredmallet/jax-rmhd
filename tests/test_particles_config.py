# Test-particle configuration and run-driver wiring, plans/TESTPART_PLAN.md §4.
#
# Everything here is about the PLUMBING around the physics gates in
# tests/test_particles_coupled.py:
#
#   - params.particles validation and normalization (the schema of
#     taranis.particles.state.normalize_config) and its params.json round trip;
#   - init_particles / template / moments: shapes, fp64-always, the two init kinds, and
#     the RNG stream's determinism and per-ensemble independence;
#   - the simulate / simulate_scan pstate contract (required exactly when particles are
#     on) and their (state, pstate) return;
#   - the particle_moments.txt sidecar: header, one row per (step, ensemble), append on
#     restart with the rows past the restart time dropped -- and that the while_loop
#     driver writes none;
#   - load_particles' refusal to run without params.particles.
#
# Cheap grids, no convergence: not precision-marked, so both sessions run it.
# Script: `python tests/test_particles_config.py`.
from _rmhd_testing import bootstrap, checks, fresh_params, managed_manager, snap_dir

bootstrap()

import json
import os

import jax.numpy as jnp
import numpy as np

import taranis as jr
from taranis.particles.fields import FIELD_MASK_DEFAULTS, FIELD_PIECES, NWORK, WORK_PIECES
from taranis.particles.state import (MOMENTS, NMOM, ParticleState, init_particles, moments,
                                     normalize_config, template)
from taranis.snapshot_io import get_saved_steps, load_particles, load_snapshot
from taranis.types import SimulationState

# one ideal-Ohm ensemble and one full-dpsi/dt ensemble: enough structure for the mask
# resolution, the two init kinds and a 2-row-per-step sidecar
PARTICLES = {"n": 32, "ensembles": [
    {"qm": 15.0, "init": {"kind": "maxwellian", "vth": 1.0}},
    {"qm": -15.0, "init": {"kind": "ring", "vperp": 1.0, "vz": 0.5},
     "ez_resistive": True, "ez_forcing": True},
]}


def _base(**over):
    kw = dict(dims=2, nx=32, ny=32, Lx=2 * np.pi, Ly=2 * np.pi, dt=0.01,
              adaptive_timestep=False, diss=1e-2, hyper=1, cfl_safety=0.5,
              forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(5e-3, 5e-3),
              forcing_tau=1.0, fshell=(1, 3), forcing_seed=42)
    kw.update(over)
    return fresh_params(**kw)


def _ic(x, y):
    return jnp.stack([jnp.cos(x) * jnp.cos(y) + 0.3 * jnp.sin(2 * x + y),
                      jnp.sin(x) * jnp.cos(y) + 0.2 * jnp.cos(x - 2 * y)], axis=0)


# --------------------------------------------------------------------- validation

def test_particles_config_validation():
    """Every way a particles config can be wrong is a ValueError that names the offending
    key -- a silently accepted typo would run a differently configured ensemble than the
    user asked for, which no later gate can detect."""
    good_ens = {"qm": 1.0, "init": {"kind": "maxwellian", "vth": 1.0}}

    def rejects(c, label, particles, needle, **over):
        try:
            _base(particles=particles, **over)
            c.check(f"{label} rejected", False, "no ValueError raised")
        except ValueError as e:
            c.check(f"{label} rejected, message names {needle!r}", needle in str(e), str(e))

    with checks() as c:
        rejects(c, "particles that is not a dict", [good_ens], "dict")
        rejects(c, "missing 'n'", {"ensembles": [good_ens]}, "'n'")
        rejects(c, "n = 0", {"n": 0, "ensembles": [good_ens]}, "> 0")
        rejects(c, "non-integer n", {"n": 3.5, "ensembles": [good_ens]}, "int")
        rejects(c, "unknown top-level key", {"n": 4, "nparticles": 8,
                                             "ensembles": [good_ens]}, "nparticles")
        rejects(c, "substeps = 0", {"n": 4, "substeps": 0, "ensembles": [good_ens]}, ">= 1")
        rejects(c, "missing ensembles", {"n": 4}, "ensembles")
        rejects(c, "empty ensembles", {"n": 4, "ensembles": []}, "ensembles")
        rejects(c, "ensemble missing 'qm'", {"n": 4, "ensembles": [{"init": good_ens["init"]}]},
                "qm")
        rejects(c, "unknown ensemble key",
                {"n": 4, "ensembles": [dict(good_ens, ez_ohmic=True)]}, "ez_ohmic")
        rejects(c, "unknown init kind",
                {"n": 4, "ensembles": [dict(good_ens, init={"kind": "beam", "vth": 1.0})]},
                "beam")
        rejects(c, "init missing vth",
                {"n": 4, "ensembles": [dict(good_ens, init={"kind": "maxwellian"})]}, "vth")
        rejects(c, "extra init key",
                {"n": 4, "ensembles": [dict(good_ens,
                                            init={"kind": "ring", "vperp": 1.0, "vth": 1.0})]},
                "vth")
        rejects(c, "vth <= 0",
                {"n": 4, "ensembles": [dict(good_ens, init={"kind": "maxwellian", "vth": 0.0})]},
                "> 0")
        # scalar coercions raise ValueError naming the key, never a bare TypeError
        rejects(c, "non-bool init_on_restart",
                {"n": 4, "init_on_restart": "yes", "ensembles": [good_ens]}, "init_on_restart")
        rejects(c, "non-bool mask piece",
                {"n": 4, "ensembles": [dict(good_ens, ez_forcing="on")]}, "ez_forcing")
        rejects(c, "non-float B0", {"n": 4, "B0": "one", "ensembles": [good_ens]}, "B0")
        rejects(c, "non-float qm",
                {"n": 4, "ensembles": [dict(good_ens, qm=None)]}, "qm")
        rejects(c, "non-float vth",
                {"n": 4, "ensembles": [dict(good_ens, init={"kind": "maxwellian",
                                                            "vth": [1.0]})]}, "vth")
        # a mask dict conflicting with the same piece inline
        rejects(c, "mask/inline conflict",
                {"n": 4, "ensembles": [dict(good_ens, ez_forcing=True,
                                            mask={"ez_forcing": False})]}, "ez_forcing")
        rejects(c, "unknown key inside 'mask'",
                {"n": 4, "ensembles": [dict(good_ens, mask={"ez_ohmic": True})]}, "ez_ohmic")
        # per-ensemble B0 (the amplitude parameter 1/epsilon) and the E_parallel projection
        rejects(c, "non-float per-ensemble B0",
                {"n": 4, "ensembles": [dict(good_ens, B0="ten")]}, "B0")
        rejects(c, "per-ensemble B0 = 0",
                {"n": 4, "ensembles": [dict(good_ens, B0=0.0)]}, "> 0")
        rejects(c, "negative per-ensemble B0",
                {"n": 4, "ensembles": [dict(good_ens, B0=-2.0)]}, "> 0")
        rejects(c, "top-level B0 <= 0 reaches the ensembles",
                {"n": 4, "B0": -1.0, "ensembles": [good_ens]}, "> 0")
        rejects(c, "non-bool epar_project",
                {"n": 4, "ensembles": [dict(good_ens, epar_project="yes")]}, "epar_project")
        # the projection is defined only for the exact ideal-Ohm mask
        for label, over in (("with ez_resistive on", {"ez_resistive": True}),
                            ("with ez_forcing on", {"ez_forcing": True}),
                            ("with ez_ideal off", {"ez_ideal": False}),
                            ("with eperp off", {"eperp": False}),
                            ("with bperp off", {"bperp": False})):
            rejects(c, f"epar_project {label}",
                    {"n": 4, "ensembles": [dict(good_ens, epar_project=True, **over)]},
                    "epar_project")
        rejects(c, "eqtype != RMHD", {"n": 4, "ensembles": [good_ens]}, "RMHD",
                eqtype="GDI", forcing=False)
        # 3D: B0 != 1 would leave a spurious (1-B0)*d_z(phi) in E_z, because the solver's
        # Alfven coefficient is 1. The message must say so, not just refuse.
        for label, particles in (
                ("top-level B0", {"n": 4, "B0": 10.0, "ensembles": [good_ens]}),
                ("per-ensemble B0", {"n": 4, "ensembles": [dict(good_ens, B0=10.0)]})):
            rejects(c, f"3D with {label} != 1", particles, "B0 must be 1.0 in 3D",
                    dims=3, nz=4, Lz=2 * np.pi, forcing=False)


def test_particles_3d_is_accepted_single_process():
    """3D particles are legal single-process in both z modes -- and B0 = 1 is the only
    amplitude 3D accepts, since the per-ensemble knob 2D enjoys would break the d_z(phi)
    cancellation in E_z (the rejection itself is in the validation test above)."""
    ens = {"qm": 5.0, "init": {"kind": "maxwellian", "vth": 1.0}}
    cfg = {"n": 8, "ensembles": [ens, dict(ens, ez_resistive=True, ez_forcing=True)]}
    kw = dict(dims=3, nz=8, Lz=2 * np.pi, forcing=False)
    p_fd = _base(particles=cfg, **kw)
    p_zs = _base(particles=cfg, z_spectral=True, **kw)
    p_b0 = _base(particles={"n": 4, "B0": 1.0, "ensembles": [dict(ens, B0=1.0)]}, **kw)
    with checks() as c:
        c.check("dims=3 finite-difference z accepts particles",
                p_fd.particles["ensembles"][1]["mask"]["ez_resistive"] is True
                and p_fd.n_ens == 2)
        c.check("dims=3 z_spectral accepts particles", p_zs.n_ens == 2 and p_zs.z_spectral)
        c.check("an explicit B0 = 1.0 is fine in 3D",
                [e["B0"] for e in p_b0.particles["ensembles"]] == [1.0])
        c.check("a 2D config still takes any B0 > 0 (no Alfven term to break)",
                _base(particles={"n": 4, "B0": 10.0,
                                 "ensembles": [ens]}).particles["ensembles"][0]["B0"] == 10.0)


def test_particles_config_normalization():
    """The stored config is fully normalized: defaults filled, every per-piece key folded
    into a complete 5-entry mask, plain JSON types only. Downstream code (push_ensembles,
    template, load_particles) indexes it without any get()/default of its own, so a
    half-normalized dict would be a KeyError at trace time."""
    params = _base(particles=PARTICLES)
    cfg = params.particles
    ens = cfg["ensembles"]
    off = _base()
    # a tuple of ensembles must validate identically: from_snapshot rebuilds JSON lists
    # as tuples, and re-running __init__ on that record must not be an error
    tup = _base(particles={"n": 4, "ensembles": ({"qm": 1.0,
                                                  "init": {"kind": "maxwellian", "vth": 2.0}},)})
    raw = {"n": 4, "ensembles": [{"qm": 1.0, "init": {"kind": "maxwellian", "vth": 1.0}}]}
    decoupled = _base(particles=raw)
    raw["n"] = 999
    raw["ensembles"][0]["qm"] = 999.0
    # per-ensemble B0 overriding the top-level default, and the E_parallel projection
    b0_cfg = _base(particles={"n": 4, "B0": 10.0, "ensembles": [
        {"qm": 1.5, "init": {"kind": "maxwellian", "vth": 1.0}, "epar_project": True},
        {"qm": 15.0, "init": {"kind": "maxwellian", "vth": 1.0}, "B0": 1.0},
    ]}).particles
    with checks() as c:
        c.check("defaults filled: seed/substeps/B0/init_on_restart",
                (cfg["seed"], cfg["substeps"], cfg["B0"], cfg["init_on_restart"])
                == (0, 1, 1.0, False), str(cfg))
        c.check("n and ensembles preserved",
                cfg["n"] == 32 and len(ens) == 2, str(cfg))
        c.check("ensembles is a tuple of dicts",
                isinstance(ens, tuple) and all(isinstance(e, dict) for e in ens))
        c.check("ensemble 0 carries the full default mask (ideal-Ohm particle)",
                ens[0]["mask"] == FIELD_MASK_DEFAULTS, str(ens[0]["mask"]))
        c.check("ensemble 1's per-piece keys are consumed into a full 5-key mask",
                set(ens[1]["mask"]) == set(FIELD_PIECES) and all(ens[1]["mask"].values())
                and not any(k in ens[1] for k in FIELD_PIECES), str(ens[1]))
        c.check("every ensemble carries its own B0 and epar_project after normalization",
                all("B0" in e and "epar_project" in e for e in ens)
                and [e["B0"] for e in ens] == [1.0, 1.0]
                and [e["epar_project"] for e in ens] == [False, False], str(ens))
        c.check("a per-ensemble B0 overrides the top-level one, which stays the default "
                "for the ensembles that do not set it",
                (b0_cfg["B0"], b0_cfg["ensembles"][0]["B0"], b0_cfg["ensembles"][1]["B0"])
                == (10.0, 10.0, 1.0), str(b0_cfg))
        c.check("epar_project is accepted with the ideal-Ohm mask",
                b0_cfg["ensembles"][0]["epar_project"] is True
                and b0_cfg["ensembles"][0]["mask"] == FIELD_MASK_DEFAULTS)
        c.check("a config with per-ensemble B0/epar_project normalizes idempotently",
                normalize_config(b0_cfg) == b0_cfg, str(normalize_config(b0_cfg)))
        c.check("ring init's vz defaults to 0.0 and vperp survives",
                ens[1]["init"] == {"kind": "ring", "vperp": 1.0, "vz": 0.5}, str(ens[1]["init"]))
        c.check("qm stored as float (sign preserved)",
                ens[1]["qm"] == -15.0 and isinstance(ens[1]["qm"], float))
        c.check("params.n_ens is set when particles are on", params.n_ens == 2)
        c.check("params.n_ens is NOT set when particles are off (guarded like the z attrs)",
                not hasattr(off, "n_ens") and off.particles is None)
        c.check("a tuple of ensembles validates like a list",
                tup.particles["ensembles"][0]["init"]["vth"] == 2.0)
        c.check("_init_args keeps a decoupled copy of the RAW dict",
                decoupled._init_args["particles"]
                == {"n": 4, "ensembles": [{"qm": 1.0,
                                           "init": {"kind": "maxwellian", "vth": 1.0}}]},
                str(decoupled._init_args["particles"]))
        # normalize_config is the single validation entry point (config.py calls it lazily)
        c.check("normalize_config alone reproduces params.particles",
                normalize_config(PARTICLES) == cfg)
        c.check("normalize_config is idempotent (its own output validates unchanged)",
                normalize_config(cfg) == cfg, str(normalize_config(cfg)))
        c.check("a normalized config can be fed back to Parameters",
                _base(particles=params.particles).particles == cfg)
        c.check("a 'mask' merges with non-conflicting inline per-piece keys",
                _base(particles={"n": 4, "ensembles": [
                    {"qm": 1.0, "init": {"kind": "maxwellian", "vth": 1.0},
                     "mask": {"ez_resistive": True}, "ez_forcing": True}]}
                      ).particles["ensembles"][0]["mask"]
                == dict(FIELD_MASK_DEFAULTS, ez_resistive=True, ez_forcing=True))


def test_params_json_roundtrip_with_particles():
    """params.json records the raw particles dict and from_snapshot reproduces the
    normalized one. The JSON round trip turns tuples into lists, so a re-save of a
    from_snapshot-built Parameters must still compare equal (no-op), not raise."""
    params = _base(particles=PARTICLES)
    off = _base()
    # the A3b keys ride the same raw-dict round trip
    projected = {"n": 8, "B0": 10.0, "ensembles": [
        {"qm": 1.5, "init": {"kind": "ring", "vperp": 1.0}, "epar_project": True},
        {"qm": 15.0, "init": {"kind": "ring", "vperp": 1.0}, "B0": 1.0},
    ]}
    with snap_dir() as d_p:
        pp = _base(particles=projected)
        pp.save(d_p)
        rec_p = json.load(open(os.path.join(d_p, "params.json")))
        back_p = jr.Parameters.from_snapshot(d_p)
        try:
            back_p.save(d_p)
            resave_p = None
        except ValueError as e:
            resave_p = str(e)
    with snap_dir() as d, snap_dir() as d_off:
        params.save(d)
        rec = json.load(open(os.path.join(d, "params.json")))
        back = jr.Parameters.from_snapshot(d)
        try:
            params.save(d)      # identical re-save
            back.save(d)        # from_snapshot's tuples vs the file's lists
            resave = None
        except ValueError as e:
            resave = str(e)
        off.save(d_off)
        rec_off = json.load(open(os.path.join(d_off, "params.json")))
        # an old record predating the ctor arg must backfill, not invalidate the directory
        rec_off.pop("particles")
        with open(os.path.join(d_off, "params.json"), "w") as f:
            json.dump(rec_off, f, indent=1)
        off.save(d_off)
        backfilled = json.load(open(os.path.join(d_off, "params.json")))
        try:
            params.save(d_off)
            differs = "no ValueError"
        except ValueError as e:
            differs = str(e)
        with checks() as c:
            c.check("params.json records the RAW particles dict",
                    rec["particles"] == PARTICLES, str(rec["particles"]))
            c.check("from_snapshot reproduces the normalized config",
                    back.particles == params.particles, str(back.particles))
            c.check("re-saving the original AND the from_snapshot copy are both no-ops "
                    "(JSON lists must compare equal to the normalized tuples)",
                    resave is None, str(resave))
            c.check("particles=None is recorded as null", rec_off.get("particles", 0) is None
                    or "particles" not in rec_off)
            c.check("a record with no 'particles' key backfills instead of failing",
                    backfilled["particles"] is None, str(backfilled.get("particles")))
            c.check("a differing particles record is a hard error naming particles",
                    "particles" in differs, differs)
            c.check("epar_project and per-ensemble B0 round-trip through params.json",
                    rec_p["particles"] == projected
                    and [e["B0"] for e in back_p.particles["ensembles"]] == [10.0, 1.0]
                    and [e["epar_project"] for e in back_p.particles["ensembles"]]
                    == [True, False], str(rec_p["particles"]))
            c.check("... and re-saving the from_snapshot copy is still a no-op",
                    resave_p is None, str(resave_p))


# ------------------------------------------------------------- init / template / moments

def test_init_particles_template_and_moments():
    """The particle RNG stream: fp64 state whatever TARANIS_PRECISION is, positions
    uniform over the box, ring/maxwellian drawn as documented, deterministic in the seed
    and independent between ensembles (fold_in). moments(pstate, bsample) is the sidecar's
    payload -- including mu about the LOCAL B, which is why it needs the B sample."""
    cfg = {"seed": 3, "n": 4096, "ensembles": [
        {"qm": 1.0, "init": {"kind": "maxwellian", "vth": 0.5}},
        {"qm": 1.0, "init": {"kind": "ring", "vperp": 2.0, "vz": -0.25}},
    ]}
    params = _base(particles=cfg)
    ps = init_particles(params)
    same = init_particles(params)
    other = init_particles(_base(particles=dict(cfg, seed=4)))
    x, v, w = np.asarray(ps.x), np.asarray(ps.v), np.asarray(ps.w)
    tm = template(params)
    vperp = np.hypot(v[1, :, 0], v[1, :, 1])
    # a tilted, non-uniform B so mu is not just v_perp^2/2B0, and a nonzero w to average
    rng = np.random.default_rng(7)
    b = 1.0 + 0.3 * rng.standard_normal(v.shape)
    b[..., 2] += 2.0
    ps_w = ps._replace(w=jnp.asarray(rng.standard_normal(w.shape)))
    mom = np.asarray(moments(ps_w, jnp.asarray(b)))
    bsq = np.sum(b * b, axis=2)
    vparb2 = np.sum(v * b, axis=2) ** 2 / bsq
    vperpb2 = np.sum(v * v, axis=2) - vparb2
    mu = vperpb2 / (2.0 * np.sqrt(bsq))
    ref = np.stack([np.mean(v[:, :, 0] ** 2 + v[:, :, 1] ** 2, axis=1),
                    np.mean(v[:, :, 2] ** 2, axis=1), np.mean(v[:, :, 2], axis=1),
                    np.mean(vperpb2, axis=1), np.mean(vparb2, axis=1),
                    np.mean(mu, axis=1), np.mean(mu ** 2, axis=1)]
                   + [np.mean(np.asarray(ps_w.w)[:, :, i], axis=1) for i in range(NWORK)],
                   axis=1)
    with checks() as c:
        c.check("ParticleState leaves are (n_ens, n, 3) float64 at every precision",
                ps.x.shape == (2, 4096, 3) and ps.x.dtype == jnp.float64
                and ps.v.dtype == jnp.float64, f"{ps.x.shape} {ps.x.dtype} {ps.v.dtype}")
        c.check(f"w is (n_ens, n, NWORK={NWORK}) float64 and starts at exactly zero "
                f"(pieces {WORK_PIECES})",
                ps.w.shape == (2, 4096, NWORK) and ps.w.dtype == jnp.float64
                and np.all(w == 0.0), f"{ps.w.shape} {ps.w.dtype}")
        c.check("template matches init_particles' shapes/dtypes (the restore tree)",
                tm.x.shape == ps.x.shape and tm.v.shape == ps.v.shape
                and tm.w.shape == ps.w.shape and tm.x.dtype == jnp.float64
                and tm.w.dtype == jnp.float64 and isinstance(tm, ParticleState))
        c.check("positions uniform over [0,Lx) x [0,Ly), z = 0 in 2D",
                x[..., 0].min() >= 0.0 and x[..., 0].max() < params.Lx
                and x[..., 1].min() >= 0.0 and x[..., 1].max() < params.Ly
                and np.all(x[..., 2] == 0.0))
        c.check(f"positions actually fill the box (x mean {x[..., 0].mean():.3f} ~ Lx/2)",
                abs(x[..., 0].mean() / params.Lx - 0.5) < 0.02)
        c.check(f"maxwellian: each component ~ N(0, vth^2), vth=0.5, measured "
                f"{v[0].std(axis=0)}",
                np.allclose(v[0].std(axis=0), 0.5, rtol=0.05, atol=0.0))
        c.check(f"ring: |v_perp| = vperp exactly (max dev "
                f"{np.max(np.abs(vperp - 2.0)):.2e}) and v_z = vz",
                np.max(np.abs(vperp - 2.0)) < 1e-14 and np.all(v[1, :, 2] == -0.25))
        c.check("same seed -> identical draw",
                np.array_equal(np.asarray(same.x), x) and np.array_equal(np.asarray(same.v), v))
        c.check("different seed -> different draw",
                not np.array_equal(np.asarray(other.x), x))
        c.check("ensembles are drawn independently (fold_in per ensemble)",
                not np.array_equal(x[0], x[1]))
        c.check(f"MOMENTS = {MOMENTS} and moments() is (n_ens, {NMOM}) matching numpy "
                f"(vperpB2/vparB2/mu about the local B, w means per piece)",
                mom.shape == (2, NMOM) and np.allclose(mom, ref, rtol=1e-13, atol=0.0),
                str(mom - ref))
        c.check("the local-B split is a partition of |v|^2 and differs from the "
                "zhat-referred one (the B sample is tilted)",
                np.allclose(mom[:, MOMENTS.index("vperpB2")] + mom[:, MOMENTS.index("vparB2")],
                            np.mean(np.sum(v * v, axis=2), axis=1), rtol=1e-13, atol=0.0)
                and not np.allclose(mom[:, MOMENTS.index("vparB2")],
                                    mom[:, MOMENTS.index("vz2")], rtol=1e-3, atol=0.0),
                str(mom[:, [MOMENTS.index(k) for k in ("vperpB2", "vparB2")]]))


def test_init_particles_3d_z_distribution():
    """3D draws z uniform over the box. 2D draws only (x, y): the 2D stream must be
    exactly what it was before z existed, or every existing 2D run's particles move."""
    import jax
    cfg = {"seed": 5, "n": 4096,
           "ensembles": [{"qm": 1.0, "init": {"kind": "maxwellian", "vth": 1.0}}]}
    p3 = _base(particles=cfg, dims=3, nz=8, Lz=3.0, forcing=False)
    p2 = _base(particles=cfg)
    x3 = np.asarray(init_particles(p3).x)
    x2 = np.asarray(init_particles(p2).x)
    kx = jax.random.split(jax.random.fold_in(jax.random.key(5), 0))[0]
    u2 = np.asarray(jax.random.uniform(kx, (4096, 2), dtype=jnp.float64))
    u3 = np.asarray(jax.random.uniform(kx, (4096, 3), dtype=jnp.float64))
    with checks() as c:
        c.check("2D positions are the (n,2) uniform draw exactly (unchanged stream)",
                np.array_equal(x2[0, :, 0], u2[:, 0] * p2.Lx)
                and np.array_equal(x2[0, :, 1], u2[:, 1] * p2.Ly)
                and np.all(x2[..., 2] == 0.0))
        c.check("3D positions are the (n,3) draw, z scaled by Lz",
                np.array_equal(x3[0, :, 2], u3[:, 2] * p3.Lz))
        c.check(f"3D z spans [0, Lz={p3.Lz}) and fills it "
                f"(mean {x3[..., 2].mean():.3f} ~ Lz/2)",
                x3[..., 2].min() >= 0.0 and x3[..., 2].max() < p3.Lz
                and abs(x3[..., 2].mean()/p3.Lz - 0.5) < 0.02)


# --------------------------------------------------------------- run.py pstate contract

def _run(params, kgrid, mngr, driver="scan", pstate=None, t_end=0.2, nblock=5,
         t_snap=100.0, save=True):
    state = jr.initialize(_ic, params)
    if driver == "scan":
        return jr.simulate_scan(state, kgrid, params, nblock, t_snap, t_end, mngr,
                                save=save, pstate=pstate)
    return jr.simulate(state, kgrid, params, t_snap, t_end, mngr, save=save, pstate=pstate)


def test_driver_pstate_contract():
    """A particle carry is required exactly when params.particles is set, and both drivers
    return (state, pstate) then -- a driver that silently ran without particles would make
    every downstream diagnostic quietly empty."""
    on = _base(particles=PARTICLES)
    off = _base()
    kg_on, kg_off = jr.setup_kgrids(on), jr.setup_kgrids(off)
    with checks() as c:
        with snap_dir() as d, managed_manager(on, d, nsnap=10) as mngr:
            try:
                _run(on, kg_on, mngr, t_end=0.02)
                c.check("simulate_scan without pstate raises", False, "no ValueError")
            except ValueError as e:
                c.check("simulate_scan with particles on but no pstate raises",
                        "pstate" in str(e), str(e))
            try:
                _run(on, kg_on, mngr, driver="while", t_end=0.02)
                c.check("simulate without pstate raises", False, "no ValueError")
            except ValueError as e:
                c.check("simulate with particles on but no pstate raises",
                        "pstate" in str(e), str(e))
        with snap_dir() as d, managed_manager(off, d, nsnap=10) as mngr:
            try:
                _run(off, kg_off, mngr, pstate=init_particles(on), t_end=0.02)
                c.check("simulate_scan with a stray pstate raises", False, "no ValueError")
            except ValueError as e:
                c.check("simulate_scan with particles off but a pstate raises",
                        "particles are off" in str(e), str(e))
            try:
                _run(off, kg_off, mngr, driver="while", pstate=init_particles(on), t_end=0.02)
                c.check("simulate with a stray pstate raises", False, "no ValueError")
            except ValueError as e:
                c.check("simulate with particles off but a pstate raises",
                        "particles are off" in str(e), str(e))
            ret_off = _run(off, kg_off, mngr, t_end=0.02)
            c.check("particles off: simulate_scan still returns a bare SimulationState",
                    isinstance(ret_off, SimulationState))
        for driver in ("scan", "while"):
            with snap_dir() as d, managed_manager(on, d, nsnap=10) as mngr:
                ret = _run(on, kg_on, mngr, driver=driver, pstate=init_particles(on),
                           t_end=0.02)
                c.check(f"{driver} driver returns (state, pstate)",
                        isinstance(ret, tuple) and len(ret) == 2
                        and isinstance(ret[0], SimulationState)
                        and isinstance(ret[1], ParticleState), str(type(ret)))


# ------------------------------------------------------------------------- sidecar

def _sidecar(d):
    with open(os.path.join(d, "particle_moments.txt")) as f:
        return f.read().splitlines()


def test_moments_sidecar():
    """simulate_scan appends one row per (step, ensemble) to particle_moments.txt: header
    once, times strictly increasing across the flattened scan ys, and the last block's
    rows equal moments(pstate) of the returned carry. A restart truncates the file to the
    rows with t <= the restart state's t first, so restarting from an older snapshot
    leaves no duplicated times. The while_loop driver emits nothing (it cannot carry scan
    ys), which is why production heating runs use simulate_scan."""
    dt = 0.01
    on = _base(particles=PARTICLES, dt=dt)
    kg = jr.setup_kgrids(on)
    n_ens = on.n_ens
    with snap_dir() as d, managed_manager(on, d, nsnap=10) as mngr:
        end, pend = _run(on, kg, mngr, pstate=init_particles(on), t_end=0.2)
        lines = _sidecar(d)
        # read everything off the carry BEFORE the restart call donates it. mu/mu2 need
        # the push's own B sample, which the carry does not keep, so the final-row check
        # below covers the columns computable from pstate alone.
        v_end, w_end = np.asarray(pend.v), np.asarray(pend.w)
        mom_end = np.stack([np.mean(v_end[:, :, 0] ** 2 + v_end[:, :, 1] ** 2, axis=1),
                            np.mean(v_end[:, :, 2] ** 2, axis=1),
                            np.mean(v_end[:, :, 2], axis=1)]
                           + [np.mean(w_end[:, :, i], axis=1) for i in range(NWORK)], axis=1)
        t_end1 = float(end.t)
        # restart into the SAME directory: the sidecar is appended to, header not repeated
        end2, _ = jr.simulate_scan(end, kg, on, 5, 100.0, 0.3, mngr, save=True, pstate=pend)
        lines2 = _sidecar(d)
        t_end2 = float(end2.t)
    rows = [ln.split() for ln in lines[1:]]
    t = np.array([float(r[0]) for r in rows])
    ens = np.array([int(r[1]) for r in rows])
    vals = np.array([[float(x) for x in r[2:]] for r in rows])
    nsteps = int(round(t_end1 / dt))

    # cfl_every > 1 flattens (nblocks, cfl_every, ...) ys -- times must still be monotone
    blocked = _base(particles=PARTICLES, adaptive_timestep=True, cfl_every=2, dt=0.01)
    with snap_dir() as d2, managed_manager(blocked, d2, nsnap=10) as mngr:
        endb, _ = _run(blocked, jr.setup_kgrids(blocked), mngr,
                       pstate=init_particles(blocked), t_end=0.2, nblock=4)
        rows_b = [ln.split() for ln in _sidecar(d2)[1:]]
    tb = np.array([float(r[0]) for r in rows_b])[::n_ens]

    # restart from an OLDER snapshot: the rows the restarted run is about to rewrite are
    # dropped on entry, so the file ends up exactly as an uninterrupted run's
    with snap_dir() as d3, managed_manager(on, d3, nsnap=10) as mngr:
        _run(on, kg, mngr, pstate=init_particles(on), t_end=0.2, t_snap=0.05)
        rows_1 = [ln.split() for ln in _sidecar(d3)[1:]]
        steps = get_saved_steps(d3)
        mid = steps[len(steps) // 2]
        state_mid, pstate_mid = load_snapshot(mid, d3, on), load_particles(mid, d3, on)
        t_mid = float(state_mid.t)
        jr.simulate_scan(state_mid, kg, on, 5, 100.0, 0.2, mngr, save=True, pstate=pstate_mid)
        lines_3 = _sidecar(d3)
    rows_3 = [ln.split() for ln in lines_3[1:]]
    t1 = np.array([float(r[0]) for r in rows_1])
    t3 = np.array([float(r[0]) for r in rows_3])

    with checks() as c:
        c.check("header written once, naming the moment columns",
                lines[0] == "# t ensemble " + " ".join(MOMENTS), lines[0])
        c.check(f"one row per (step, ensemble): {len(rows)} == {nsteps} steps * {n_ens}",
                len(rows) == nsteps * n_ens, f"{len(rows)} rows, nsteps={nsteps}")
        c.check(f"t, ensemble and {NMOM} moment columns per row",
                all(len(r) == 2 + NMOM for r in rows))
        c.check("the ensemble column cycles 0..n_ens-1 within each step",
                np.array_equal(ens, np.tile(np.arange(n_ens), len(rows) // n_ens)))
        c.check("t strictly increasing over the flattened scan ys",
                np.all(np.diff(t[::n_ens]) > 0), str(t[::n_ens][:6]))
        c.check(f"last row's t is the returned state's t ({t[-1]!r} vs {t_end1!r})",
                abs(t[-1] - t_end1) < 1e-14)
        cols = [MOMENTS.index(k) for k in ("vperp2", "vz2", "vz")
                + tuple("w_" + p for p in WORK_PIECES)]
        c.check("the final rows' velocity and work moments are those of the returned carry",
                np.allclose(vals[-n_ens:][:, cols], mom_end, rtol=1e-13, atol=1e-300),
                str(vals[-n_ens:][:, cols] - mom_end))
        c.check("the mu columns are finite and positive (mu = v_perpB^2/2|B| about the "
                "local B)",
                np.all(np.isfinite(vals[:, MOMENTS.index("mu")]))
                and np.all(vals[:, MOMENTS.index("mu")] > 0.0)
                and np.all(vals[:, MOMENTS.index("mu2")] > 0.0))
        c.check("a restart appends without a second header",
                sum(1 for ln in lines2 if ln.startswith("#")) == 1
                and len(lines2) == 1 + int(round(t_end2 / dt)) * n_ens,
                f"{len(lines2)} lines, t_end2={t_end2}")
        c.check("cfl_every>1: the flattened block ys stay time-ordered",
                len(rows_b) % n_ens == 0 and np.all(np.diff(tb) > 0)
                and abs(tb[-1] - float(endb.t)) < 1e-14, str(tb[:6]))
        c.check(f"a restart from an OLDER snapshot (t = {t_mid}) drops the rows past it: "
                f"same row count as the uninterrupted run",
                len(rows_3) == len(rows_1), f"{len(rows_3)} vs {len(rows_1)} rows")
        c.check("... the same times, in the same order (no duplicates, no gap)",
                np.array_equal(t3, t1), str(t3[:6]))
        c.check("... each ensemble's times are strictly increasing",
                all(np.all(np.diff(t3[e::n_ens]) > 0) for e in range(n_ens)))
        c.check("... and the header is still written exactly once",
                sum(1 for ln in lines_3 if ln.startswith("#")) == 1)


def test_simulate_writes_no_sidecar():
    """The while_loop driver has snapshot-cadence diagnostics only."""
    on = _base(particles=PARTICLES)
    with snap_dir() as d, managed_manager(on, d, nsnap=10) as mngr:
        _run(on, jr.setup_kgrids(on), mngr, driver="while",
             pstate=init_particles(on), t_end=0.05)
        exists = os.path.exists(os.path.join(d, "particle_moments.txt"))
    with checks() as c:
        c.check("simulate() writes no particle_moments.txt", not exists)


def test_load_particles_requires_particles_config():
    """load_particles refuses to run against a particle-less Parameters: there is no tree
    to restore into, and returning None would defer the failure into the pusher.
    (The missing-item paths -- FileNotFoundError and init_on_restart -- are gate 6c in
    tests/test_particles_coupled.py.)"""
    off = _base()
    with snap_dir() as d:
        try:
            load_particles(0, d, off)
            msg = "no ValueError"
        except ValueError as e:
            msg = str(e)
    with checks() as c:
        c.check("load_particles with params.particles=None raises ValueError",
                "particles" in msg and "no ValueError" != msg, msg)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
