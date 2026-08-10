# Parameters record/restore tests: params.save() and Parameters.from_snapshot()
#
# Invariants under test (CLAUDE.md "Parameters / physics registry"):
#   - save() records the CONSTRUCTOR ARGS + precision; from_snapshot() re-runs __init__
#     and reproduces the object (JSON turns tuples into lists -> restored as tuples);
#   - an identical re-save is a no-op (no rewrite at all, not just no error);
#   - a differing existing record is a HARD ERROR;
#   - a record written by older code (missing keys) is backfilled from the signature
#     defaults, so adding a ctor argument never invalidates existing run directories;
#   - unknown keys and a precision mismatch warn but do not fail;
#   - explicit overrides win;
#   - transport (comm/rank/size) is never persisted, and comm_backend is recorded but
#     excluded from the differing-record comparison (runs must restart across backends).
#
# NOTE: _capture() collects BOTH stdout and the warnings module, so the assertions hold
# whichever channel config.py uses (it warns via warnings.warn).
#
# pytest: `pytest tests/test_params.py`.  Script: `python tests/test_params.py`.
from _rmhd_testing import bootstrap, checks, fresh_params, mpi_size, snap_dir

bootstrap()

import contextlib
import io
import json
import os
import time
import warnings


import taranis as jr
from taranis import _precision

# A parameter set that exercises tuple-valued args (JSON lists on disk), the forcing
# block and a couple of non-default scalars.
_KW = dict(nx=16, ny=16, nz=8, diss=(0.01, 0.02), hyper=2, cfl_safety=0.4, dt=0.02,
           adaptive_timestep=False, dims=3, forcing=True, forcing_mode="elsasser",
           forcing_power_elsasser=(0.6, 0.2), fshell=(1, 3), forcing_seed=5,
           forcing_scale_max=2.0, cfl_every=3, lsrk_scan=False)


def _single_process_only(what):
    """Soft skip (print + return True). params.save() is collective, and every rank
    takes this branch together, so nothing can deadlock: these tests use rank-local
    tmp dirs, which only rank 0 would ever write."""
    if mpi_size() > 1:
        print(f"[SKIP] {what} -- single-process only (rank-local tmp dirs)")
        return True
    return False


def _params_path(d):
    return os.path.join(d, "params.json")


def _read_record(d):
    with open(_params_path(d)) as f:
        return json.load(f)


def _write_record(d, rec):
    with open(_params_path(d), "w") as f:
        json.dump(rec, f, indent=1)


def _stat(d):
    """(bytes, mtime_ns) of params.json -- the pair the no-op re-save must preserve."""
    path = _params_path(d)
    with open(path, "rb") as f:
        raw = f.read()
    return raw, os.stat(path).st_mtime_ns


def _capture(fn):
    """Run fn(), returning (result, captured_text). Collects stdout AND the warnings
    module, since pytest.warns is unusable here (script mode)."""
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(buf):
            result = fn()
    # ResourceWarnings are dropped: their messages quote tmp paths, which would
    # otherwise pollute the substring assertions.
    msgs = "".join(str(w.message) for w in caught
                   if not issubclass(w.category, ResourceWarning))
    return result, buf.getvalue() + msgs


def _current_precision():
    # FIELD precision (RMHD_PRECISION) -- jax_enable_x64 is now unconditionally on;
    # this must match config.py's own record (taranis._precision.precision).
    return _precision.precision


def test_from_snapshot_roundtrips_ctor_args():
    if _single_process_only("test_from_snapshot_roundtrips_ctor_args"):
        return
    p = fresh_params(**_KW)
    with snap_dir() as d:
        p.save(d)
        rec = _read_record(d)
        p2 = jr.Parameters.from_snapshot(d)
        with checks() as c:
            c.check("params.json records every constructor argument",
                    set(rec) - {"_created", "_precision"} == set(p._init_args))
            c.check(f"params.json records the session precision "
                    f"({rec.get('_precision')!r})",
                    rec.get("_precision") == _current_precision())
            c.check("tuple-valued args are stored as JSON lists",
                    rec["eqpars"]["diss"] == [0.01, 0.02] and rec["fshell"] == [1, 3]
                    and rec["forcing_power_elsasser"] == [0.6, 0.2])
            c.check("from_snapshot restores every ctor arg identically "
                    "(lists come back as tuples)",
                    p2._init_args == p._init_args,
                    detail=str({k: (v, p2._init_args.get(k)) for k, v in p._init_args.items()
                                if p2._init_args.get(k) != v}))
            c.check("restored tuple args really are tuples (incl. inside eqpars)",
                    isinstance(p2.eqpars["diss"], tuple) and isinstance(p2.fshell, tuple)
                    and isinstance(p2.forcing_power_elsasser, tuple))
            c.check("derived attributes are recomputed by the re-run __init__",
                    (p2.nfields, p2.n_ou, p2.spatial_dimensions) == (p.nfields, p.n_ou,
                                                                     p.spatial_dimensions)
                    and p2.dx == p.dx and p2.dy == p.dy and p2.dz == p.dz
                    and p2.cfl_every == p.cfl_every and p2.lsrk_scan == p.lsrk_scan)
            c.check("from_snapshot returns a NEW object (Parameters hashes by identity)",
                    p2 is not p)


def test_identical_resave_is_a_noop():
    if _single_process_only("test_identical_resave_is_a_noop"):
        return
    p = fresh_params(**_KW)
    with snap_dir() as d:
        p.save(d)
        before = _stat(d)
        time.sleep(0.05)  # so a rewrite would be visible in st_mtime_ns
        p.save(d)
        after = _stat(d)
        # a second, independently constructed but identical Parameters must also be a no-op
        time.sleep(0.05)
        fresh_params(**_KW).save(d)
        after2 = _stat(d)
        with checks() as c:
            c.check("identical re-save leaves params.json byte-identical",
                    after[0] == before[0])
            c.check(f"identical re-save does not rewrite the file "
                    f"(mtime_ns {before[1]} == {after[1]})", after[1] == before[1])
            c.check("an equal-but-distinct Parameters is also a no-op",
                    after2 == before)


def test_differing_record_raises():
    if _single_process_only("test_differing_record_raises"):
        return
    p = fresh_params(**_KW)
    with snap_dir() as d:
        p.save(d)
        before = _stat(d)
        errs = {}
        for label, other in (("nx", fresh_params(**dict(_KW, nx=32))),
                             ("eqpars", fresh_params(**dict(_KW, diss=(0.02, 0.02)))),
                             ("forcing_seed", fresh_params(**dict(_KW, forcing_seed=6)))):
            try:
                other.save(d)
                errs[label] = None
            except ValueError as e:
                errs[label] = str(e)
        with checks() as c:
            for label, msg in errs.items():
                c.check(f"a differing {label} is a hard error",
                        msg is not None and "already records different parameters" in msg
                        and label in msg)
            c.check("the rejected saves left params.json untouched", _stat(d) == before)


def test_backfill_of_deleted_keys():
    # A params.json written by older code lacks newer ctor args. save() backfills them
    # from the current signature defaults (and refreshes the file); from_snapshot()
    # simply lets __init__ supply them.
    if _single_process_only("test_backfill_of_deleted_keys"):
        return
    p = fresh_params(**_KW)
    dropped = ("forcing_tau", "z_diss")  # both left at their defaults by _KW
    with snap_dir() as d:
        p.save(d)
        orig = _read_record(d)
        defaults = {k: orig[k] for k in dropped}
        rec = {k: v for k, v in orig.items() if k not in dropped}
        _write_record(d, rec)

        p2, out = _capture(lambda: jr.Parameters.from_snapshot(d))
        p.save(d)  # must NOT raise: backfilled values equal the current ones
        after = _read_record(d)
        with checks() as c:
            c.check(f"from_snapshot tolerates a record missing {list(dropped)}",
                    p2.forcing_tau == p.forcing_tau and p2.z_diss == p.z_diss)
            c.check("... without warning about them (they are known keys)",
                    "unknown" not in out.lower())
            c.check("re-saving backfills the missing keys instead of erroring",
                    all(after[k] == defaults[k] for k in dropped))
            c.check("the backfilled record otherwise matches the original",
                    {k: v for k, v in after.items() if k != "_created"}
                    == {k: v for k, v in orig.items() if k != "_created"})
            c.check("from_snapshot of the refreshed record still round-trips",
                    jr.Parameters.from_snapshot(d)._init_args == p._init_args)


def test_unknown_key_warns_and_is_ignored():
    if _single_process_only("test_unknown_key_warns_and_is_ignored"):
        return
    p = fresh_params(**_KW)
    with snap_dir() as d:
        p.save(d)
        rec = _read_record(d)
        rec["bogus_param_from_the_future"] = 17
        rec["another_bogus_one"] = [1, 2]
        _write_record(d, rec)
        p2, out = _capture(lambda: jr.Parameters.from_snapshot(d))
        with checks() as c:
            c.check("from_snapshot ignores unknown keys instead of raising",
                    p2._init_args == p._init_args)
            c.check(f"... and says so, naming them (captured: {out.strip()!r})",
                    "bogus_param_from_the_future" in out and "another_bogus_one" in out
                    and "ignoring unknown parameters" in out)
            c.check("no bogus attribute leaked onto the object",
                    not hasattr(p2, "bogus_param_from_the_future"))


def test_precision_mismatch_warns():
    if _single_process_only("test_precision_mismatch_warns"):
        return
    p = fresh_params(**_KW)
    other = "32" if _current_precision() == "64" else "64"
    with snap_dir() as d:
        p.save(d)
        rec = _read_record(d)
        rec["_precision"] = other
        _write_record(d, rec)
        p2, out = _capture(lambda: jr.Parameters.from_snapshot(d))
        # ... and no warning when it matches
        rec["_precision"] = _current_precision()
        _write_record(d, rec)
        _, quiet = _capture(lambda: jr.Parameters.from_snapshot(d))
        with checks() as c:
            c.check(f"a record written at precision {other} warns in a "
                    f"{_current_precision()} session (captured: {out.strip()!r})",
                    "precision" in out.lower() and other in out)
            c.check("... but still returns usable Parameters",
                    p2._init_args == p._init_args)
            c.check("a matching precision does not warn",
                    "precision" not in quiet.lower())
            c.check("_precision is not passed to __init__",
                    "_precision" not in p2._init_args)


def test_overrides_win_over_recorded_values():
    if _single_process_only("test_overrides_win_over_recorded_values"):
        return
    p = fresh_params(**_KW)
    with snap_dir() as d:
        p.save(d)
        p2 = jr.Parameters.from_snapshot(d, nx=32, eqpars={"diss": (0.5, 0.5), "hyper": 1},
                                         forcing=False, cfl_every=1)
        with checks() as c:
            c.check("scalar overrides win", p2.nx == 32 and p2.eqpars["hyper"] == 1
                    and p2.cfl_every == 1)
            c.check("tuple overrides win (not the recorded list)",
                    p2.eqpars["diss"] == (0.5, 0.5))
            c.check("boolean overrides win", p2.forcing is False)
            c.check("un-overridden values still come from the record",
                    p2.ny == p.ny and p2.nz == p.nz and p2.forcing_seed == p.forcing_seed
                    and p2.fshell == p.fshell)
            c.check("overrides do not touch the file on disk",
                    _read_record(d)["nx"] == _KW["nx"])


def test_transport_is_not_persisted_or_compared():
    # comm/rank/size are runtime transport, never ctor args -> never recorded.
    # comm_backend IS recorded (a run directory should document how it ran) but is
    # excluded from the differing-record check: a run must be restartable on the
    # other backend without deleting params.json.
    if _single_process_only("test_transport_is_not_persisted_or_compared"):
        return
    p = fresh_params(**_KW)
    with snap_dir() as d:
        p.save(d)
        rec = _read_record(d)
        with checks() as c:
            c.check("no MPI transport objects in the record",
                    not ({"comm", "rank", "size", "cart_comm", "left_neighbor",
                          "right_neighbor"} & set(rec)))
            c.check("no transport objects among the captured ctor args",
                    not ({"comm", "rank", "size"} & set(p._init_args)))
            c.check(f"comm_backend IS recorded (got {rec.get('comm_backend')!r})",
                    rec.get("comm_backend") == "mpi4jax")
            # flip the recorded backend: re-saving must still be accepted
            rec["comm_backend"] = "jax"
            _write_record(d, rec)
            try:
                p.save(d)
                err = None
            except ValueError as e:
                err = str(e)
            c.check(f"a differing comm_backend is NOT a differing-record error "
                    f"(got {err!r})", err is None)
            c.check("the flipped backend is left on disk (recorded, not compared)",
                    _read_record(d)["comm_backend"] == "jax")


def test_constructor_rejects_malformed_arguments():
    # Arguments with no valid interpretation: each must be refused at construction
    # rather than failing late, or silently running a different problem.
    bad = {
        "dims=4 is not a dimensionality": dict(dims=4),
        "dims=2.5 is not a dimensionality": dict(dims=2.5),
        "Lz<=0 with dims=3": dict(dims=3, Lz=0.0),
        "a tuple cfl_safety (the old positional diss slot)": dict(dims=2,
                                                                  cfl_safety=(0.1, 0.1)),
        "fshell with nmin>=nmax": dict(dims=2, forcing=True, fshell=(5, 1)),
        "forcing_power_elsasser not a pair": dict(dims=2, forcing=True,
                                                  forcing_mode="elsasser",
                                                  forcing_power_elsasser=(1.0, 2.0, 3.0)),
    }
    ok = {
        "malformed fshell is ignored while forcing is off": dict(dims=2, forcing=False,
                                                                 fshell=(5, 1)),
    }
    with checks() as c:
        for name, kw in bad.items():
            try:
                fresh_params(**kw)
                raised = None
            except ValueError as e:
                raised = str(e)
            c.check(f"rejected: {name}", raised is not None, "constructed without error")
        for name, kw in ok.items():
            try:
                fresh_params(**kw)
                raised = None
            except ValueError as e:
                raised = str(e)
            c.check(f"accepted: {name}", raised is None, f"raised {raised!r}")


def test_offgrid_forcing_shell_is_rejected_by_setup_kgrids():
    # A shell that lands between grid modes leaves fmask empty; the run would otherwise
    # proceed silently unforced. Caught where fmask is built, not in Parameters.
    p = fresh_params(dims=2, forcing=True, fshell=(50, 60))
    with checks() as c:
        try:
            jr.setup_kgrids(p)
            raised = None
        except ValueError as e:
            raised = str(e)
        c.check("setup_kgrids rejects a forcing shell containing no modes",
                raised is not None and "no modes" in raised, f"raised {raised!r}")


def test_eqpars_reject_non_dict_and_are_read_by_the_recipe():
    # eqpars is the equation set's parameter dict: Parameters only type-checks it, the
    # recipe (rmhd.linear_matrix, via setup_kgrids) validates the contents.
    with checks() as c:
        try:
            jr.Parameters(nx=8, ny=8, Lx=1.0, Ly=1.0, cfl_safety=0.5, dims=2,
                          eqpars=[("diss", 0.0)])
            raised = None
        except ValueError as e:
            raised = str(e)
        c.check("a non-dict eqpars is rejected at construction",
                raised is not None and "eqpars" in raised, f"raised {raised!r}")
        # NB fresh_params folds its diss/hyper knobs into eqpars, so the "missing" case is
        # built straight from Parameters.
        empty = lambda: jr.Parameters(nx=8, ny=8, Lx=1.0, Ly=1.0, cfl_safety=0.5, dims=2)
        cases = {
            "diss longer than nfields": (lambda: fresh_params(dims=2, diss=(1.0, 2.0, 3.0)), True),
            "eqpars missing diss/hyper": (empty, True),
            "scalar diss (broadcast to every field)": (lambda: fresh_params(dims=2, diss=0.01), False),
            "diss of length nfields": (lambda: fresh_params(dims=2, diss=(0.01, 0.02)), False),
        }
        for name, (build, should_raise) in cases.items():
            p = build()
            try:
                jr.setup_kgrids(p)
                raised = None
            except ValueError as e:
                raised = str(e)
            c.check(f"setup_kgrids {'rejects' if should_raise else 'accepts'}: {name}",
                    (raised is not None) == should_raise, f"raised {raised!r}")


def test_legacy_toplevel_diss_hyper_folds_into_eqpars():
    # diss/hyper were Parameters ctor args before 2026-08-01 and are recorded at top level
    # in every params.json written until then: from_snapshot folds them into eqpars (with a
    # warning) and re-saving must refresh the file instead of reporting a differing record.
    if _single_process_only("test_legacy_toplevel_diss_hyper_folds_into_eqpars"):
        return
    p = fresh_params(**_KW)
    with snap_dir() as d:
        p.save(d)
        rec = _read_record(d)
        legacy = {k: v for k, v in rec.items() if k != "eqpars"}
        legacy["diss"], legacy["hyper"] = rec["eqpars"]["diss"], rec["eqpars"]["hyper"]
        _write_record(d, legacy)
        p2, out = _capture(lambda: jr.Parameters.from_snapshot(d))
        try:
            p.save(d)      # the same parameters, in the new format
            err = None
        except ValueError as e:
            err = str(e)
        after = _read_record(d)
        with checks() as c:
            c.check("the legacy record loads with diss/hyper in eqpars",
                    p2.eqpars == p.eqpars, f"{p2.eqpars!r} != {p.eqpars!r}")
            c.check(f"... and warns about the fold (captured: {out.strip()!r})",
                    "eqpars" in out and "diss" in out)
            c.check("... without calling them unknown keys", "unknown" not in out.lower())
            c.check(f"re-saving the same parameters is not a differing record (got {err!r})",
                    err is None)
            c.check("re-saving rewrites the record in eqpars form",
                    after.get("eqpars") == rec["eqpars"]
                    and "diss" not in after and "hyper" not in after)


def test_eqpars_nested_tuples_roundtrip():
    # JSON restore must recurse into eqpars VALUES: a nested tuple round-trips as nested
    # tuples, not a tuple of lists (review finding on _lists_to_tuples, 2026-08-01).
    if _single_process_only("test_eqpars_nested_tuples_roundtrip"):
        return
    p = fresh_params(**dict(_KW, eqpars={"diss": ((0.01,), (0.02,)), "hyper": 2}))
    with snap_dir() as d:
        p.save(d)
        p2 = jr.Parameters.from_snapshot(d)
        with checks() as c:
            c.check("nested tuples inside eqpars survive save/from_snapshot",
                    p2.eqpars == p.eqpars, f"{p2.eqpars!r} != {p.eqpars!r}")
            c.check("... as tuples all the way down",
                    isinstance(p2.eqpars["diss"][0], tuple), repr(p2.eqpars))


def test_unused_z_options_warn():
    # z_diff_order / z_diss_hyper are stored but never read back by rmhd.LinearTerm.
    with checks() as c:
        _, out = _capture(lambda: fresh_params(dims=3, z_diff_order=6))
        c.check("a non-default z_diff_order warns that it is ignored",
                "z_diff_order" in out and "IGNORED" in out, out)
        _, quiet = _capture(lambda: fresh_params(dims=3))
        c.check("the default z-stencil options warn about nothing", "IGNORED" not in quiet,
                quiet)


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
