#!/usr/bin/env python3
"""The script that actually runs ON Kaggle. Generated/pushed by launch.py.

launch.py rewrites the CONFIG sentinel line below with a JSON literal before pushing, so
this file is a valid, importable Python module both before and after the substitution --
which is what lets the sandbox unit-test the interesting bits (repo discovery, entry
resolution) without a GPU, without pip, and without Kaggle.

Everything above `pip_install_stack()` is stdlib only: on a fresh Kaggle image nothing
from this project is importable until the installs below have run.

Layout note: the run happens in /kaggle/tmp, NOT /kaggle/working. /kaggle/working is the
persisted, downloadable directory but it is capped at 20 GiB *and* ~500 files, and orbax
writes many small files per snapshot. So we work in the (large, unlimited-file, NOT
persisted) scratch space and drop exactly one zip into /kaggle/working at the end -- in a
finally block, so a crashed or budget-truncated run is still downloadable.
"""
import importlib     # invalidate_caches/import_module -- see enable_in_process_import
import importlib.util
import json  # noqa: F401 -- used by the substituted CONFIG line, keep this import
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

CONFIG = None  # __KAGGLE_LAUNCHER_CONFIG__

INPUT_ROOT = "/kaggle/input"
WORKING = "/kaggle/working"
SCRATCH = "/kaggle/tmp"
# where archive datasets get unpacked when Kaggle serves them unextracted. Under SCRATCH
# (writable and large; /kaggle/input is read-only), and named so it cannot collide with a
# driver's workdir -- see main(), which renames a workdir that would land here.
EXTRACT_ROOT = os.path.join(SCRATCH, "_repo_extract")
# where a repo found on the READ-ONLY dataset mount gets copied so pip can build in it (see
# ensure_writable_repo). Deliberately distinct from EXTRACT_ROOT and from any driver
# workdir -- main() renames a workdir that would land on either.
SRC_ROOT = os.path.join(SCRATCH, "_repo_src")
# the PACKAGE name in the repo's pyproject.toml -- dashed, not underscored (verified).
REPO_MARKER = 'name = "jax-rmhd"'
# archive extensions worth opening, and the size above which one is skipped rather than
# unpacked: the repo dataset is ~23 MB, so anything near a GiB is somebody else's data.
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz")
MAX_ARCHIVE_BYTES = 2 * 1024 ** 3


def log(msg):
    print(f"[kernel] {msg}", flush=True)


# ---------------------------------------------------------------- environment setup

def pip_install_stack():
    """The known-good Kaggle GPU recipe (from examples/kaggle_forced_turbulence_256cubed.ipynb).

    Order matters: orbax/tensorstore first, then a GPU jaxlib on top, so the CPU-only jax
    that orbax may drag in gets replaced rather than the other way round. If a future
    jaxlib drops Pascal (sm_60) support, pin the version here (see README).

    Always `sys.executable -m pip`, never a bare "pip": Kaggle images carry several
    interpreters, and whichever `pip` is first on PATH need not be the one belonging to the
    process running this kernel -- installing into a different interpreter's site-packages
    would leave every import here failing for no visible reason.
    """
    for cmd in ([sys.executable, "-m", "pip", "install", "-q", "orbax-checkpoint", "tensorstore"],
                [sys.executable, "-m", "pip", "install", "-q", "-U", "jax[cuda12]"]):
        log("$ " + " ".join(cmd))
        subprocess.run(cmd, check=True)


def list_tree(root, max_depth=4, max_entries=20):
    """Print a BOUNDED recursive listing of `root` (depth, and entries per directory).

    Printed before every repo search: the first real Kaggle run failed with "no jax_rmhd
    checkout found" and the log said nothing about what WAS mounted, which left the two
    plausible causes (dataset still processing -> empty mount; `--dir-mode zip` served
    unextracted -> a single .zip) indistinguishable. Never walks unbounded: a dataset mount
    can contain millions of files, and this runs before anything useful has happened.
    """
    log(f"--- listing {root} (depth<={max_depth}, <={max_entries} entries/dir) ---")
    if not os.path.isdir(root):
        log(f"{root} is not a directory (does it exist? is a dataset attached?)")
        return

    def walk(d, depth, indent):
        try:
            names = sorted(os.listdir(d))
        except OSError as exc:
            log(f"{indent}<cannot list: {exc}>")
            return
        for name in names[:max_entries]:
            full = os.path.join(d, name)
            if os.path.isdir(full):
                log(f"{indent}{name}/")
                if depth + 1 < max_depth:
                    walk(full, depth + 1, indent + "  ")
            else:
                try:
                    size = os.path.getsize(full)
                except OSError:
                    size = -1
                log(f"{indent}{name}  ({size} bytes)")
        if len(names) > max_entries:
            log(f"{indent}... {len(names) - max_entries} more entries (truncated)")

    walk(root, 0, "  ")
    log(f"--- end listing {root} ---")


def _marker_search(root, max_depth):
    """BFS for a directory holding a pyproject.toml that carries REPO_MARKER, else None."""
    if not os.path.isdir(root):
        return None
    frontier = [(root, 0)]
    while frontier:
        d, depth = frontier.pop(0)
        cand = os.path.join(d, "pyproject.toml")
        if os.path.isfile(cand):
            try:
                if REPO_MARKER in open(cand, "r", errors="replace").read():
                    return d
            except OSError:
                pass
        if depth < max_depth:
            try:
                for name in sorted(os.listdir(d)):
                    sub = os.path.join(d, name)
                    if os.path.isdir(sub):
                        frontier.append((sub, depth + 1))
            except OSError:
                pass
    return None


def _find_archives(root, max_depth):
    """Bounded BFS for archive files under `root` (same depth budget as the marker search)."""
    found = []
    if not os.path.isdir(root):
        return found
    frontier = [(root, 0)]
    while frontier:
        d, depth = frontier.pop(0)
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for name in names:
            full = os.path.join(d, name)
            if os.path.isdir(full):
                if depth < max_depth:
                    frontier.append((full, depth + 1))
            elif name.lower().endswith(ARCHIVE_SUFFIXES):
                found.append(full)
    return found


def _archive_stem(path):
    base = os.path.basename(path)
    low = base.lower()
    for suf in ARCHIVE_SUFFIXES:
        if low.endswith(suf):
            return base[: len(base) - len(suf)] or "archive"
    return base


def _safe_member(name, dest):
    """True iff `name` extracts strictly inside `dest` (no absolute paths, no `..`)."""
    if not name or name.startswith("/") or name.startswith("\\") or os.path.isabs(name):
        return False
    if ".." in name.replace("\\", "/").split("/"):
        return False
    dest_real = os.path.realpath(dest)
    target = os.path.realpath(os.path.join(dest_real, name))
    return target == dest_real or target.startswith(dest_real + os.sep)


def _extract_repo_archive(path, extract_root):
    """Unpack `path` into <extract_root>/<stem>/ iff it looks like the repo. Returns dir|None.

    "Looks like the repo" = it has a pyproject.toml member; we do not read it out of the
    archive, the marker check happens on the extracted tree. Refuses the whole archive if
    ANY member would escape the destination (path traversal) -- partial extraction of a
    hostile archive is worse than not trying.
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        log(f"  skip {path}: {exc}")
        return None
    if size > MAX_ARCHIVE_BYTES:
        log(f"  skip {path}: {size / 1e9:.1f} GB is over the "
            f"{MAX_ARCHIVE_BYTES / 1e9:.1f} GB archive limit")
        return None

    try:
        if path.lower().endswith(".zip"):
            opener = zipfile.ZipFile(path)
            names = opener.namelist()
        else:
            opener = tarfile.open(path, "r:gz")
            names = opener.getnames()
    except Exception as exc:                                   # noqa: BLE001
        log(f"  skip {path}: cannot read ({type(exc).__name__}: {exc})")
        return None

    with opener as arch:
        if not any(os.path.basename(n) == "pyproject.toml" for n in names):
            log(f"  skip {path}: {len(names)} members, no pyproject.toml")
            return None
        dest = os.path.join(extract_root, _archive_stem(path))
        bad = [n for n in names if not _safe_member(n, dest)]
        if bad:
            log(f"  REFUSING {path}: {len(bad)} member(s) escape the extraction directory "
                f"(e.g. {bad[0]!r})")
            return None
        if isinstance(arch, tarfile.TarFile):
            links = [m.name for m in arch.getmembers() if not (m.isfile() or m.isdir())]
            if links:
                log(f"  REFUSING {path}: contains non-regular members (e.g. {links[0]!r})")
                return None
        if os.path.isdir(dest):
            shutil.rmtree(dest)         # never merge with a previous attempt
        os.makedirs(dest, exist_ok=True)
        log(f"  extracting {path} ({size / 1e6:.1f} MB, {len(names)} members) -> {dest}")
        if isinstance(arch, tarfile.TarFile) and hasattr(tarfile, "data_filter"):
            arch.extractall(dest, filter="data")     # 3.12+: also the stdlib's own guard
        else:
            arch.extractall(dest)
    return dest


def find_repo_root(input_root=INPUT_ROOT, max_depth=5, extract_root=EXTRACT_ROOT):
    """Locate the pushed repo under the dataset mount, unpacking an archive if that is how
    Kaggle served it.

    `--dir-mode zip` datasets do not mount with a predictable directory depth (Kaggle may
    or may not interpose the archive name / the staging subfolder), and may not extract the
    archive at all, so rather than hardcoding a path we (1) print what is actually mounted,
    (2) walk down to `max_depth` looking for the repo's own pyproject.toml, (3) failing
    that, unpack any archive that contains a pyproject.toml into scratch and search again.
    """
    if not os.path.isdir(input_root):
        raise RuntimeError(f"{input_root} does not exist -- is a dataset attached?")
    list_tree(input_root)

    found = _marker_search(input_root, max_depth)
    if found:
        return found

    log(f"no pyproject.toml with {REPO_MARKER!r} under {input_root} (depth {max_depth}) -- "
        "looking for archives (Kaggle can serve a --dir-mode zip dataset unextracted)")
    archives = _find_archives(input_root, max_depth)
    log(f"archives found: {archives if archives else 'none'}")
    extracted = []
    for arch in archives:
        dest = _extract_repo_archive(arch, extract_root)
        if dest:
            extracted.append(dest)
            hit = _marker_search(dest, max_depth)
            if hit:
                log(f"found the repo inside {arch}")
                return hit

    raise RuntimeError(
        f"no jax_rmhd checkout found under {input_root} (searched every directory to depth "
        f"{max_depth} for a pyproject.toml containing {REPO_MARKER!r}; the bounded listing "
        f"of the mount is above in this log). Archives seen: {archives or 'none'}; "
        f"extracted and searched: {extracted or 'none'}. If the mount was EMPTY, the dataset "
        "was probably still processing when this kernel started -- re-push with "
        "--skip-dataset once its Kaggle page shows the new version. Otherwise check that "
        "the dataset pushed by launch.py is attached to this kernel.")


def _is_writable(path):
    """True iff a file can ACTUALLY be created inside `path`.

    os.access(path, os.W_OK) alone is not trustworthy here: it answers from the permission
    bits, which on a read-only bind/overlay mount (exactly what /kaggle/input is) can say
    "writable" while every write returns EROFS. So the access() check is only a cheap first
    gate; the verdict comes from creating and removing a real probe file.
    """
    if not os.access(path, os.W_OK):
        return False
    probe = os.path.join(path, ".kaggle_launcher_write_probe")
    try:
        with open(probe, "w") as fh:
            fh.write("probe")
    except OSError as exc:
        log(f"write probe in {path} failed ({type(exc).__name__}: {exc}) -- treating as read-only")
        return False
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass
    return True


def _grant_owner_write(root):
    """Add owner rwx/rw to every directory/file under `root` (inclusive).

    shutil.copytree preserves mode bits (copy2 on files, copystat on directories), so a
    dataset mount that serves its files 0444/0555 would copy to a tree that is *still*
    unwritable -- defeating the point of the copy, and making the next run's rmtree of it
    fail too. Cheap insurance; the copy is ours alone.
    """
    def grant(path, extra):
        try:
            os.chmod(path, os.stat(path).st_mode | extra)
        except OSError as exc:
            log(f"  could not chmod {path}: {exc}")

    grant(root, 0o700)
    for d, dirs, files in os.walk(root):
        for name in dirs:
            grant(os.path.join(d, name), 0o700)
        for name in files:
            grant(os.path.join(d, name), 0o600)


def ensure_writable_repo(repo_root, scratch_root=SRC_ROOT):
    """Return a repo root pip can build in, copying it off a read-only mount if need be.

    Field failure (2026-08-03): discovery worked, and then `pip install -e
    /kaggle/input/jax-rmhd-src` died in "Getting requirements to build editable ... exit code
    1". /kaggle/input is a READ-ONLY mount and an editable (PEP 660) install has to write
    build metadata (egg-info and friends) INTO the source tree. Dropping `-e` would not
    reliably fix it either -- modern pip also runs the PEP 517 build in-tree for a plain
    local-directory install. So the fix belongs upstream of pip: put the tree somewhere
    writable first. It is ~22 MB, so the copy costs nothing.

    A root that is already writable -- notably the archive-extraction fallback, which lands
    under /kaggle/tmp -- is returned unchanged and never re-copied; the probe is what tells
    the two cases apart.
    """
    if _is_writable(repo_root):
        return repo_root
    log(f"repo mount read-only -- copied to {scratch_root} (pip needs to write build "
        f"metadata into the source tree; source was {repo_root})")
    if os.path.isdir(scratch_root):
        _grant_owner_write(scratch_root)     # a previous copy may itself be mode-0555
        shutil.rmtree(scratch_root)          # never merge with a previous attempt
    elif os.path.lexists(scratch_root):
        os.remove(scratch_root)
    shutil.copytree(repo_root, scratch_root)
    _grant_owner_write(scratch_root)
    if not _is_writable(scratch_root):
        raise RuntimeError(f"copied the repo to {scratch_root} but it is still not writable "
                           "-- pip would fail there too; check free space on " + SCRATCH)
    return scratch_root


def install_repo(repo_root):
    # NOT -q: this is the step that failed in the field, and pip's `-q` swallowed the build
    # backend's actual error, leaving only "See above for output" with nothing above it.
    # The dependency installs above stay quiet; this one is the one worth reading.
    # `sys.executable -m pip` for the same reason as in pip_install_stack().
    cmd = [sys.executable, "-m", "pip", "install", "-e", repo_root]
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def enable_in_process_import(repo_root):
    """Make `import jax_rmhd` work in THIS interpreter, right after the editable install.

    Field failure (2026-08-03): `pip install -e` printed "Successfully installed
    jax-rmhd-0.1.0" and the very next `import jax_rmhd` -- the driver's own, via
    load_run_module -- still raised ModuleNotFoundError. Cause: a PEP 660 editable install
    exposes the package through an `__editable__*.pth` file dropped into site-packages, and
    .pth files are processed by the `site` module at interpreter STARTUP only. This kernel
    installs and imports within a single interpreter lifetime, so that .pth is never read
    here: the install is invisible to the very process that performed it. (`import jax`
    worked in the same process only because jax was already importable before/independently
    of our installs, not because the pip run took effect mid-process.)

    Fix: put the repo root itself on sys.path. The repo is a FLAT layout -- the `jax_rmhd/`
    package directory sits at the repo root -- so the root on sys.path makes the source
    imports work regardless of .pth timing, and points at exactly the tree the editable
    install would have pointed at. The alternative is `site.addsitedir(<site-packages>)`,
    which re-scans that directory and executes the pending .pth; we prefer the plain path
    insert because it needs no guess about WHICH site-packages pip chose (venv vs --user vs
    conda vs the image's system dirs) and brings no side effects from other pending .pth
    files. The editable install still earns its keep: it is what pulled the dependencies,
    and it is what makes jax_rmhd importable in any SUBPROCESS the driver spawns -- a fresh
    interpreter does process the .pth at startup.

    Must run AFTER RMHD_PRECISION is in os.environ (see set_precision): the verification
    import below is the first `import jax_rmhd` in the process, and precision is read at
    import time.
    """
    sys.path.insert(0, repo_root)
    importlib.invalidate_caches()   # a failed lookup earlier in the process may be cached
    log(f"sys.path[0] = {repo_root} (editable-install .pth files are only read at "
        "interpreter startup, so the install alone is invisible to this process)")
    try:
        importlib.import_module("jax_rmhd")
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(
            f"`import jax_rmhd` fails even after installing {repo_root} and putting it on "
            f"sys.path (sys.path[0]={sys.path[0]!r}): {type(exc).__name__}: {exc}. Expected "
            f"the package at {os.path.join(repo_root, 'jax_rmhd', '__init__.py')} -- if that "
            "file exists, this is a broken/partial dataset copy or a missing dependency of "
            "jax_rmhd, not a sys.path problem.") from exc
    log("import jax_rmhd: OK")


def set_precision(precision):
    """Put RMHD_PRECISION in the environment BEFORE anything imports jax_rmhd.

    Repo rule: jax_rmhd reads RMHD_PRECISION at IMPORT time (it flips jax_enable_x64 in
    jax_rmhd/__init__.py), so setting it later is a silent no-op. The first import in this
    process is now the verification import in enable_in_process_import(), which happens
    right after the repo install -- well before check_gpu -- so this belongs at the top of
    main(), not next to the jax import.
    """
    os.environ["RMHD_PRECISION"] = str(precision)
    log(f"RMHD_PRECISION={os.environ['RMHD_PRECISION']} (set before any jax_rmhd import)")


def check_gpu(precision):
    # RMHD_PRECISION was already set by set_precision() at the top of main(); re-assert
    # rather than re-set, because by now jax_rmhd has been imported and a late change would
    # take effect nowhere. (Standalone calls, e.g. from a test, still get it set.)
    want, have = str(precision), os.environ.get("RMHD_PRECISION")
    if have != want:
        if "jax_rmhd" in sys.modules:
            raise RuntimeError(
                f"RMHD_PRECISION={have!r} but the config asks for {want!r}, and jax_rmhd is "
                "already imported -- precision is read at import time, so this run would "
                "silently use the wrong precision. Set it before the first import.")
        os.environ["RMHD_PRECISION"] = want
    import jax
    devs = jax.devices()
    log(f"RMHD_PRECISION={precision}  jax.devices()={devs}")
    assert devs and devs[0].platform == "gpu", (
        f"no GPU visible ({devs}) -- check the kernel's accelerator/machine_shape setting")
    return devs


# ---------------------------------------------------------------- driver dispatch

def load_run_module(run_file):
    name = "_kaggle_run_" + os.path.splitext(os.path.basename(run_file))[0]
    spec = importlib.util.spec_from_file_location(name, run_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {run_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # the driver's own directory on sys.path, so sibling-module imports in examples/ work
    sys.path.insert(0, os.path.dirname(os.path.abspath(run_file)))
    spec.loader.exec_module(mod)
    return mod


def warn_wall_budget(entry_kwargs, session_budget_s):
    """The session budget is only ever checked BETWEEN make_data calls.

    Nothing here can interrupt a call in flight, so a single call longer than what is left
    of the session gets SIGKILLed by Kaggle -- and /kaggle/tmp is not persisted, so the
    finally-block zip never runs and the whole session's checkpoints are gone. The per-call
    bound is the driver's own `wall_budget` kwarg; warn (loudly, never silently rewriting
    the user's kwargs) when it is missing or not comfortably under the session budget.
    """
    wb = entry_kwargs.get("wall_budget")
    if wb is None:
        log(f"WARNING: entry_kwargs has no 'wall_budget'. The {session_budget_s:.0f}s session "
            "budget is only checked between make_data calls, so one long call can be killed "
            "mid-run and lose the output zip. Pass wall_budget well under the session budget.")
    elif float(wb) >= 0.5 * session_budget_s:
        log(f"WARNING: wall_budget={float(wb):.0f}s is not comfortably under the session "
            f"budget {session_budget_s:.0f}s. A call that overruns what is left of the session "
            "is killed mid-run, and /kaggle/tmp is not persisted, so no output zip is written. "
            "Prefer several short calls.")


def drive_make_data(make_data, snap_path, entry_kwargs, session_budget_s, now=time.time):
    """The make_data contract: idempotent + resumable, returns True at t_end.

    Call it until it says True or until the session budget runs out. The driver's own
    `wall_budget` kwarg bounds each individual call; this loop bounds the session, and
    reports honestly when it stops early rather than pretending the run finished.
    """
    warn_wall_budget(entry_kwargs, session_budget_s)
    t0 = now()
    calls = 0
    while True:
        done = bool(make_data(snap_path, **entry_kwargs))
        calls += 1
        elapsed = now() - t0
        log(f"make_data call {calls}: done={done}  elapsed={elapsed:.0f}s "
            f"of budget {session_budget_s:.0f}s")
        if done:
            return True
        if elapsed > session_budget_s:
            log(f"session budget exhausted after {calls} make_data calls "
                f"({elapsed:.0f}s) -- PARTIAL RUN, t_end was not reached. The checkpoints "
                "in the output zip are resumable, but this launcher does not carry them "
                "into a new session (see README).")
            return False


def resolve_entry(module, entry, entry_kwargs, snap_path, session_budget_s, run_file=None,
                  now=time.time):
    """Pick and call the module's entry point. Returns True iff the run reached t_end.

    'auto': make_data (the gdi_*_run.py contract) -> main() -> plain script execution.
    An explicit `--entry NAME` is called as fn(**entry_kwargs), except for `make_data`,
    which keeps its (snap_path, **kwargs) signature and its until-True loop.
    """
    if entry and entry != "auto":
        fn = getattr(module, entry, None)
        if not callable(fn):
            raise RuntimeError(f"--entry {entry!r} is not a callable in the run module")
        if entry == "make_data":
            return drive_make_data(fn, snap_path, entry_kwargs, session_budget_s, now=now)
        log(f"calling {entry}(**{entry_kwargs})")
        fn(**entry_kwargs)
        return True

    make_data = getattr(module, "make_data", None)
    if callable(make_data):
        return drive_make_data(make_data, snap_path, entry_kwargs, session_budget_s, now=now)

    main = getattr(module, "main", None)
    if callable(main):
        log(f"no make_data; calling main(**{entry_kwargs})")
        main(**entry_kwargs)
        return True

    log("no make_data/main; executing the file as __main__")
    runpy.run_path(run_file, run_name="__main__")
    return True


def maybe_report(module, snap_path):
    """Diagnostics go to the kernel log. Never let a broken report kill the zip."""
    report = getattr(module, "report", None)
    if not callable(report):
        return
    log("--- report() ---")
    try:
        report(snap_path)
    except Exception as exc:                                   # noqa: BLE001
        log(f"report() raised {type(exc).__name__}: {exc} (continuing to the zip)")


# ---------------------------------------------------------------- output

def zip_workdir(workdir, stem, working=WORKING):
    """One archive in /kaggle/working -- one file against the ~500-file cap."""
    os.makedirs(working, exist_ok=True)
    out = os.path.join(working, f"{stem}_output.zip")
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(workdir):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, workdir))
                n += 1
    size = os.path.getsize(out) / 1e6
    log(f"wrote {out}: {n} files, {size:.1f} MB")
    return out


# ---------------------------------------------------------------- main

def main(config=None):
    cfg = config if config is not None else CONFIG
    if cfg is None:
        raise RuntimeError("CONFIG was never substituted -- push this kernel via launch.py")
    log(f"config: {cfg}")

    set_precision(cfg["precision"])     # BEFORE any import of jax_rmhd (read at import time)
    pip_install_stack()
    repo_root = find_repo_root()
    log(f"repo root: {repo_root}")
    # from here on repo_root is the WRITABLE tree: pip installs it, and the run driver is
    # imported out of it, so the driver's module-level state belongs to the installed copy.
    repo_root = ensure_writable_repo(repo_root)
    install_repo(repo_root)
    enable_in_process_import(repo_root)   # the install alone is NOT importable in-process
    check_gpu(cfg["precision"])

    run_file = os.path.join(repo_root, cfg["run_relpath"])
    if not os.path.isfile(run_file):
        raise RuntimeError(f"run driver {cfg['run_relpath']} missing from the pushed dataset")
    stem = os.path.splitext(os.path.basename(run_file))[0]

    workdir = os.path.join(SCRATCH, stem)
    reserved = (EXTRACT_ROOT, SRC_ROOT)
    if os.path.realpath(workdir) in {os.path.realpath(r) for r in reserved}:
        # only reachable via a driver literally named _repo_extract.py / _repo_src.py, but
        # the collision would silently zip up (or delete) a repo copy, so rename instead.
        workdir = os.path.join(SCRATCH, stem + "_work")
        log(f"workdir would collide with one of {reserved}; using {workdir}")
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)          # relative paths the driver writes land in scratch, not /working
    snap_path = os.path.join(workdir, "checkpoints")
    log(f"workdir {workdir}  snap_path {snap_path}")

    t_start = time.time()
    try:
        module = load_run_module(run_file)
        done = resolve_entry(module, cfg.get("entry", "auto"), cfg.get("entry_kwargs") or {},
                             snap_path, float(cfg.get("session_budget_s", 39600.0)),
                             run_file=run_file)
        log(f"run finished: reached_t_end={done}  wall={time.time()-t_start:.0f}s")
        maybe_report(module, snap_path)
    finally:
        # unconditional: a crashed or truncated run still ships its partial checkpoints.
        try:
            zip_workdir(workdir, stem)
        except Exception as exc:                               # noqa: BLE001
            log(f"zipping failed: {type(exc).__name__}: {exc}")
            log(f"free space: {shutil.disk_usage(SCRATCH)}")
            raise


if __name__ == "__main__":
    main()
