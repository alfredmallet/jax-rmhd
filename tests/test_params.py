# Parameters record/restore tests: params.save() and Parameters.from_snapshot()
# (Phase 3 of docs/TESTING_PLAN.md).
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
# NOTE: config.py emits its "warnings" with print() on rank 0, not warnings.warn(), so
# _capture() collects BOTH stdout and the warnings module -- the assertions hold either
# way if that ever changes.
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

import jax

import jax_rmhd as jr

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
    module: config.py currently prints, and pytest.warns is unusable here (script mode)."""
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(buf):
            result = fn()
    # ResourceWarnings are dropped: config.from_snapshot does json.load(open(path))
    # without closing, and the leaked-handle message quotes the tmp path, which would
    # otherwise pollute substring assertions.
    msgs = "".join(str(w.message) for w in caught
                   if not issubclass(w.category, ResourceWarning))
    return result, buf.getvalue() + msgs


def _current_precision():
    return "64" if jax.config.read("jax_enable_x64") else "32"


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
                    rec["diss"] == [0.01, 0.02] and rec["fshell"] == [1, 3]
                    and rec["forcing_power_elsasser"] == [0.6, 0.2])
            c.check("from_snapshot restores every ctor arg identically "
                    "(lists come back as tuples)",
                    p2._init_args == p._init_args,
                    detail=str({k: (v, p2._init_args.get(k)) for k, v in p._init_args.items()
                                if p2._init_args.get(k) != v}))
            c.check("restored tuple args really are tuples",
                    isinstance(p2.diss, tuple) and isinstance(p2.fshell, tuple)
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
                             ("diss", fresh_params(**dict(_KW, diss=(0.02, 0.02)))),
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
        p2 = jr.Parameters.from_snapshot(d, nx=32, hyper=1, diss=(0.5, 0.5),
                                         forcing=False, cfl_every=1)
        with checks() as c:
            c.check("scalar overrides win", p2.nx == 32 and p2.hyper == 1
                    and p2.cfl_every == 1)
            c.check("tuple overrides win (not the recorded list)", p2.diss == (0.5, 0.5))
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


if __name__ == "__main__":
    import sys
    from _rmhd_testing import script_main
    sys.exit(script_main(globals()))
