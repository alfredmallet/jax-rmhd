#!/usr/bin/env python3
"""Regression tests for kaggle_launcher/launch.py and kaggle_launcher/_kernel_template.py.

Every test here guards a specific shipped fix -- most of them a failure that only showed
up on a real Kaggle GPU, minutes and a slice of the weekly quota after the mistake was
made. The test names carry the mapping.

Runnable two ways, like the rest of this repo's test files:

    python -m pytest kaggle_launcher/test_launcher.py -q
    python kaggle_launcher/test_launcher.py

Hermetic on purpose: no network, no pip, no `kaggle` binary (stubbed where one is
needed), no jax / no jax_rmhd import, and nothing outside temporary directories. The two
modules under test are loaded from their paths WITHOUT being registered in sys.modules,
and the one test that needs a `jax_rmhd` package on sys.path runs in a subprocess with
site-packages disabled -- so a laptop that has the real jax_rmhd installed is unaffected
and unaffecting.
"""
import ast
import atexit
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import types
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCH_PY = HERE / "launch.py"
TEMPLATE_PY = HERE / "_kernel_template.py"

# the marker _kernel_template.py's repo search looks for (kept in sync by test_find_*).
MARKER_PYPROJECT = '[build-system]\nrequires = ["setuptools"]\n\n[project]\nname = "jax-rmhd"\nversion = "0.0.0"\n'
DRIVER_SRC = ('"""A tiny stand-in for examples/gdi_3d_run.py."""\n\n\n'
              'def make_data(snap_path, **kwargs):\n    return True\n')


def _load(path, name):
    """Import a module from its path WITHOUT publishing it in sys.modules.

    Both files under test are stdlib-only and side-effect-free at import; keeping them out
    of sys.modules keeps this file safe to run inside a bigger pytest session.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


launch = _load(LAUNCH_PY, "_kl_launch_under_test")
tpl = _load(TEMPLATE_PY, "_kl_template_under_test")

# nothing in this file may import jax_rmhd; this records whatever a surrounding pytest
# session already did, so the no-pollution assertion below stays honest either way.
_JAX_RMHD_WAS_IMPORTED = "jax_rmhd" in sys.modules


# ---------------------------------------------------------------- tiny fixtures
# Deliberately not pytest fixtures: the same helpers have to work under the plain
# __main__ runner at the bottom of this file.

_TMPDIRS = []


def _tmpdir(prefix="kl_test_"):
    d = Path(tempfile.mkdtemp(prefix=prefix)).resolve()
    _TMPDIRS.append(d)
    return d


def _chmod_tree(root, dir_mode, file_mode):
    root = str(root)
    for d, dirs, files in os.walk(root, topdown=False):
        for name in files:
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(d, name), file_mode)
        for name in dirs:
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(d, name), dir_mode)
    with contextlib.suppress(OSError):
        os.chmod(root, dir_mode)


def _force_rmtree(path):
    if os.path.exists(str(path)):
        _chmod_tree(path, 0o700, 0o600)     # the read-only-tree tests leave 0o555 dirs
        shutil.rmtree(str(path), ignore_errors=True)


@atexit.register
def _cleanup_tmpdirs():
    for d in _TMPDIRS:
        _force_rmtree(d)


@contextlib.contextmanager
def _capture():
    """Collect everything the code under test prints (all of it goes through print())."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


@contextlib.contextmanager
def _env(**overrides):
    """Temporarily set (value) or unset (None) environment variables."""
    old = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _raises(exc_type, fn, *args, **kwargs):
    """assert-raises without pytest: returns the exception so the message can be checked."""
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    except Exception as exc:                                    # noqa: BLE001
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")


def _find_all(text, needle):
    out, i = [], text.find(needle)
    while i != -1:
        out.append(i)
        i = text.find(needle, i + 1)
    return out


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=str(root), capture_output=True, text=True,
                          check=True)


_MINI_REPO = None


def mini_repo():
    """A ~6-file git repo in tmp that looks enough like jax_rmhd for launch.py.

    Built once per session and reused. The real 22 MB tree is never staged, and the real
    repo's git index is never touched -- `git add` here happens in a throwaway clone-free
    repo of our own. No commit is needed: `git ls-files` reads the index.
    """
    global _MINI_REPO
    if _MINI_REPO is not None:
        return _MINI_REPO
    root = _tmpdir("kl_minirepo_")
    (root / "pyproject.toml").write_text(MARKER_PYPROJECT)
    (root / "jax_rmhd").mkdir()
    (root / "jax_rmhd" / "__init__.py").write_text("VERSION = 'mini'\n")
    (root / "examples").mkdir()
    (root / "examples" / "demo_run.py").write_text(DRIVER_SRC)
    (root / "examples" / "untracked_run.py").write_text(DRIVER_SRC)   # never `git add`ed
    kl = root / "kaggle_launcher"
    kl.mkdir()
    # the launcher itself has to live inside the mini repo: launch.py derives the repo root
    # from `git rev-parse` relative to its OWN location, so a copy is what makes a
    # subprocess run resolve to the mini repo instead of the real one.
    shutil.copy2(LAUNCH_PY, kl / "launch.py")
    shutil.copy2(TEMPLATE_PY, kl / "_kernel_template.py")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "kaggle launcher tests")
    _git(root, "add", "pyproject.toml", "jax_rmhd/__init__.py", "examples/demo_run.py",
         "kaggle_launcher/launch.py", "kaggle_launcher/_kernel_template.py")
    _MINI_REPO = root
    return root


def _run_launch(root, argv):
    """Run the mini repo's copy of launch.py as a real CLI. Returns (rc, stdout+stderr)."""
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(_tmpdir() / "no-kaggle-token")   # never created
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("KAGGLE_USERNAME", None)          # force the deterministic dry-run placeholder
    proc = subprocess.run([sys.executable, str(root / "kaggle_launcher" / "launch.py"), *argv],
                          cwd=str(root), capture_output=True, text=True, env=env)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _kaggle_stub(status_text="processing"):
    """A fake `kaggle` binary that logs its argv. Returns (dir_to_prepend_to_PATH, logfile)."""
    d = _tmpdir("kl_kagglestub_")
    log = d / "commands.log"
    stub = d / "kaggle"
    stub.write_text("#!/bin/sh\n"
                    'printf "%s\\n" "$*" >> "$KAGGLE_STUB_LOG"\n'
                    'if [ "$1" = "datasets" ] && [ "$2" = "status" ]; then\n'
                    f'  echo "{status_text}"\n'
                    "fi\n"
                    "exit 0\n")
    stub.chmod(0o755)
    log.write_text("")
    return d, log


class _Clock:
    """A fake clock whose only way to advance is sleep() -- so no test ever really sleeps."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def _fake_clock(step):
    """time.time() stand-in that advances by `step` on every call."""
    t = [0.0]

    def now():
        v = t[0]
        t[0] += step
        return v
    return now


def _fake_module(**attrs):
    return types.SimpleNamespace(**attrs)


# ================================================================ launch.py
#
# 1. CONFIG substitution -- JSON true/false/null must survive into the pushed kernel.

def test_config_line_preserves_bools_none_and_nesting():
    cfg = {"dataset_slug": "jax-rmhd-src",
           "run_relpath": "examples/gdi_3d_run.py",
           "entry": "auto",
           "entry_kwargs": {"t_end": 30, "flag": True, "none_val": None,
                            "nested": {"list": [1, False, None], "s": "it's \"quoted\""}},
           "precision": "64",
           "session_budget_s": 39600.0}
    line = launch.config_line(cfg)

    ast.parse(line)                       # syntactically valid Python, on its own
    # exec'd with ONLY json in scope: the naive `CONFIG = {"flag": true}` would NameError
    # here, which is exactly the field-failure precursor this line exists to avoid.
    ns = {"json": json}
    exec(compile(line, "<config_line>", "exec"), ns)          # noqa: S102
    assert ns["CONFIG"] == cfg
    assert ns["CONFIG"]["entry_kwargs"]["flag"] is True
    assert ns["CONFIG"]["entry_kwargs"]["none_val"] is None
    assert ns["CONFIG"]["entry_kwargs"]["nested"]["list"] == [1, False, None]
    assert isinstance(ns["CONFIG"]["entry_kwargs"]["t_end"], int)
    # the sentinel comment must survive so the substituted file stays greppable
    assert line.endswith("# __KAGGLE_LAUNCHER_CONFIG__")
    assert launch.CONFIG_SENTINEL in TEMPLATE_PY.read_text()


# 2. slug sanitize + the 5-char pad applied to the SLUG (so it reaches the ids), not the title.

def test_pad_slug_sanitizes_and_pads_both_ids():
    assert launch.slugify("GDI_2D test!") == "gdi-2d-test"
    assert launch.pad_slug("GDI_2D test!") == "gdi-2d-test"
    assert launch.slugify("gdi_3d_run.py") == "gdi-3d-run-py"
    assert launch.pad_slug("gdi") == "gdi-run"           # under Kaggle's 5-char floor
    assert launch.pad_slug("!!!") == "run-run"           # degenerate input still yields an id
    assert len(launch.pad_slug("a")) >= launch.MIN_SLUG_LEN

    staging = launch.prepare_staging(_tmpdir() / "stg")
    with _capture():
        kdir = launch.stage_kernel(staging, "gdi", "ds", "u", "NvidiaTeslaP100", {"a": 1})
    kmeta = json.loads((kdir / "kernel-metadata.json").read_text())
    assert kdir.name == "gdi-run"
    assert kmeta["id"] == "u/gdi-run"                    # the ID, not just the title
    assert kmeta["title"] == "gdi-run"
    assert kmeta["dataset_sources"] == ["u/ds-run"]      # the dataset id is padded too

    with _capture():
        dsdir = launch.stage_dataset(staging, "ds", "u", mini_repo())
    dmeta = json.loads((dsdir / "dataset-metadata.json").read_text())
    assert dmeta["id"] == "u/ds-run"
    assert dmeta["title"] == "ds-run"
    assert (dsdir / "ds-run" / "pyproject.toml").is_file()


# 3. untracked driver: rejected up front, before anything is uploaded or even printed.

def test_untracked_driver_rejected_before_any_kaggle_command():
    root = mini_repo()
    rc, out = _run_launch(root, ["push", "examples/untracked_run.py", "--dry-run",
                                 "--staging-dir", str(_tmpdir() / "stg")])
    assert rc != 0, out
    assert "not git-tracked" in out
    # nothing was staged and no kaggle command was even printed: the check is upstream of all
    # of it, which is the whole point (the alternative is finding out from a GPU kernel log).
    assert "[dry-run]" not in out
    assert "datasets version" not in out
    assert "kernels push" not in out
    assert "staged dataset" not in out


def test_tracked_driver_passes_the_git_check():
    root = mini_repo()
    rc, out = _run_launch(root, ["push", "examples/demo_run.py", "--dry-run",
                                 "--staging-dir", str(_tmpdir() / "stg")])
    assert rc == 0, out
    assert "not git-tracked" not in out
    assert "[dry-run] kaggle kernels push" in out


# 4. --entry-kwargs typos cost a second, not a 22 MB push -- and never a traceback.

def test_parse_kwargs_rejects_bad_json_with_a_clean_systemexit():
    assert launch.parse_kwargs("") == {}
    assert launch.parse_kwargs('{"t_end": 30, "flag": true, "none_val": null}') == {
        "t_end": 30, "flag": True, "none_val": None}

    exc = _raises(SystemExit, launch.parse_kwargs, "{t_end: 30}")
    assert "not valid JSON" in str(exc)
    assert "--entry-kwargs" in str(exc)
    exc = _raises(SystemExit, launch.parse_kwargs, "[1, 2]")
    assert "must be a JSON object" in str(exc)
    exc = _raises(SystemExit, launch.parse_kwargs, '"a string"')
    assert "must be a JSON object" in str(exc)


# 5. staging sentinel: never rmtree a directory this tool did not create.

def test_staging_sentinel_refuses_a_foreign_directory():
    foreign = _tmpdir() / "not-ours"
    foreign.mkdir()
    decoy = foreign / "precious.txt"
    decoy.write_text("keep me")

    exc = _raises(SystemExit, launch.prepare_staging, foreign)
    assert launch.STAGING_SENTINEL in str(exc)
    assert decoy.read_text() == "keep me"                     # untouched
    assert not (foreign / launch.STAGING_SENTINEL).exists()   # and not adopted either

    # wipe() applies the same rule, so a stale --staging-dir cannot be recursed into
    victim = foreign / "dataset"
    victim.mkdir()
    exc = _raises(SystemExit, launch.wipe, victim, foreign)
    assert "refusing to delete" in str(exc)
    assert victim.is_dir()

    # a root we created is marked, adopted idempotently, and wipe-able
    ours = _tmpdir() / "stg"
    launch.prepare_staging(ours)
    assert (ours / launch.STAGING_SENTINEL).is_file()
    launch.prepare_staging(ours)
    sub = ours / "dataset"
    sub.mkdir()
    (sub / "stale.txt").write_text("x")
    launch.wipe(sub, ours)
    assert not sub.exists()


# 6. wait_dataset_ready: the poll that stands between a dataset push and the kernel push.

def test_wait_dataset_ready_returns_on_the_poll_that_says_ready():
    calls = []

    def runner(ident):
        calls.append(ident)
        return "Dataset is still processing" if len(calls) == 1 else "status: ready"

    clk = _Clock()
    with _capture() as buf:
        got = launch.wait_dataset_ready("u/d", False, timeout=600.0, interval=10.0,
                                        runner=runner, now=clk.now, sleep=clk.sleep)
    assert got is True
    assert calls == ["u/d", "u/d"]
    assert clk.t == 10.0                 # exactly one poll interval waited
    assert "dataset ready" in buf.getvalue()


def test_wait_dataset_ready_unknown_status_needs_both_limit_and_grace():
    # unrecognized wording is not fatal -- but only after MORE than unknown_limit
    # consecutive unknowns AND unknown_grace seconds. elapsed = (poll-1) * interval, so
    # with limit 3 / grace 60 / interval 10 the first poll satisfying both is #7.
    calls = []

    def always_unknown(ident):
        calls.append(ident)
        return "Fetching dataset metadata"

    clk = _Clock()
    with _capture() as buf:
        got = launch.wait_dataset_ready("u/d", False, timeout=6000.0, interval=10.0,
                                        unknown_grace=60.0, unknown_limit=3,
                                        runner=always_unknown, now=clk.now, sleep=clk.sleep)
    assert got is True
    assert len(calls) == 7, f"warned after {len(calls)} polls; limit AND grace must both bind"
    assert "WARNING" in buf.getvalue()
    assert "--skip-dataset" in buf.getvalue()

    # grace long past, but only two consecutive unknowns: must NOT short-circuit
    calls2 = []

    def mostly_processing(ident):
        calls2.append(ident)
        n = len(calls2)
        if n <= 8:
            return "still processing"
        if n <= 10:
            return "???"
        return "ready"

    clk2 = _Clock()
    with _capture():
        got2 = launch.wait_dataset_ready("u/d", False, timeout=6000.0, interval=10.0,
                                         unknown_grace=60.0, unknown_limit=3,
                                         runner=mostly_processing, now=clk2.now,
                                         sleep=clk2.sleep)
    assert got2 is True
    assert len(calls2) == 11, "two unknowns after the grace must not be treated as ready"


def test_wait_dataset_ready_timeout_raises_and_points_at_skip_dataset():
    clk = _Clock()
    with _capture():
        exc = _raises(SystemExit, launch.wait_dataset_ready, "u/d", False, timeout=45.0,
                      interval=10.0, runner=lambda ident: "processing",
                      now=clk.now, sleep=clk.sleep)
    assert "still not ready" in str(exc)
    assert "NOT pushing the kernel" in str(exc)
    assert "--skip-dataset" in str(exc)


def test_dataset_timeout_never_reaches_the_kernel_push():
    """The field failure in full: a stub `kaggle` that never reports ready must not see
    `kernels push` -- a kernel pushed against a still-processing dataset burns GPU quota."""
    root = mini_repo()
    stubdir, log = _kaggle_stub(status_text="processing")
    staging = _tmpdir() / "stg"
    args = launch.build_parser().parse_args(
        ["push", str(root / "examples" / "demo_run.py"),
         "--staging-dir", str(staging), "--dataset-timeout", "1"])       # 1 s: one poll

    with _env(PATH=str(stubdir) + os.pathsep + os.environ.get("PATH", ""),
              KAGGLE_STUB_LOG=str(log),
              KAGGLE_USERNAME="tester",
              KAGGLE_CONFIG_DIR=str(_tmpdir() / "no-kaggle-token")):
        with _capture():
            exc = _raises(SystemExit, launch.cmd_push, args, root)

    assert "still not ready" in str(exc)
    cmds = log.read_text()
    assert "datasets version" in cmds
    assert "datasets status tester/jax-rmhd-src" in cmds
    assert "kernels push" not in cmds, f"the kernel was pushed anyway:\n{cmds}"


# 7. the whole push, dry, in the mini repo: staged layout, kernel metadata, generated kernel.

def test_dry_run_push_e2e_stages_dataset_and_kernel():
    root = mini_repo()
    staging = _tmpdir() / "stg"
    args = launch.build_parser().parse_args(
        ["push", str(root / "examples" / "demo_run.py"), "--dry-run",
         "--staging-dir", str(staging), "--machine", "NvidiaTeslaT4", "--precision", "32",
         "--session-budget", "1234",
         "--entry-kwargs", '{"t_end": 30, "flag": true, "none_val": null}'])
    with _env(KAGGLE_CONFIG_DIR=str(_tmpdir() / "no-kaggle-token"), KAGGLE_USERNAME=None):
        with _capture() as buf:
            ident = launch.cmd_push(args, root)
    out = buf.getvalue()
    assert ident == "USERNAME/demo-run"

    # -- dataset staging comes from `git ls-files`, so untracked bulk stays home
    ds = staging / "dataset"
    assert (ds / "jax-rmhd-src" / "examples" / "demo_run.py").is_file()
    assert (ds / "jax-rmhd-src" / "pyproject.toml").is_file()
    assert not (ds / "jax-rmhd-src" / "examples" / "untracked_run.py").exists()
    dmeta = json.loads((ds / "dataset-metadata.json").read_text())
    assert dmeta["id"] == "USERNAME/jax-rmhd-src"
    assert dmeta["licenses"] == [{"name": "CC0-1.0"}]

    # -- kernel metadata: the fields Kaggle actually reads
    kdir = staging / "kernel" / "demo-run"
    kmeta = json.loads((kdir / "kernel-metadata.json").read_text())
    assert kmeta["id"] == "USERNAME/demo-run"
    assert kmeta["code_file"] == "kernel.py"
    assert kmeta["language"] == "python"
    assert kmeta["kernel_type"] == "script"
    assert kmeta["is_private"] == "true"
    assert kmeta["enable_gpu"] == "true"
    assert kmeta["enable_internet"] == "true"          # the pip installs need it
    assert kmeta["machine_shape"] == "NvidiaTeslaT4"
    assert kmeta["dataset_sources"] == ["USERNAME/jax-rmhd-src"]
    assert kmeta["model_sources"] == []                # newer kaggle-api schema
    assert kmeta["kernel_sources"] == [] and kmeta["competition_sources"] == []

    # -- the generated kernel
    src = (kdir / "kernel.py").read_text()
    ast.parse(src)
    assert launch.CONFIG_SENTINEL not in src           # the sentinel was really replaced
    cfg_line = [ln for ln in src.splitlines() if ln.startswith("CONFIG = ")]
    assert len(cfg_line) == 1
    ns = {"json": json}
    exec(compile(cfg_line[0], "<kernel>", "exec"), ns)          # noqa: S102
    cfg = ns["CONFIG"]
    assert cfg["run_relpath"] == "examples/demo_run.py"
    assert cfg["dataset_slug"] == "jax-rmhd-src"
    assert cfg["entry"] == "auto"
    assert cfg["precision"] == "32"
    assert cfg["session_budget_s"] == 1234.0
    assert cfg["entry_kwargs"] == {"t_end": 30, "flag": True, "none_val": None}
    assert cfg["entry_kwargs"]["flag"] is True and cfg["entry_kwargs"]["none_val"] is None

    # every pip invocation goes through THIS interpreter (Kaggle images carry several)
    assert '[sys.executable, "-m", "pip"' in src
    for i in _find_all(src, '"pip",'):
        assert src[i - 6:i] == '"-m", ', f"bare pip argv at offset {i}: {src[i - 40:i + 40]!r}"
    # dependency installs stay quiet; the repo install does NOT (-q ate the build error)
    assert '"install", "-q", "orbax-checkpoint"' in src
    repo_install = [ln for ln in src.splitlines() if '"install", "-e"' in ln]
    assert len(repo_install) == 1 and "-q" not in repo_install[0]
    # field failure #3: an editable install is invisible to the installing process
    assert "sys.path.insert(0, repo_root)" in src
    assert "importlib.invalidate_caches()" in src

    # -- dry run printed the commands and ran none of them
    assert "[dry-run] kaggle datasets version" in out
    assert "[dry-run] kaggle datasets status USERNAME/jax-rmhd-src" in out
    assert "[dry-run] kaggle kernels push" in out
    assert "$ kaggle" not in out


# ================================================================ _kernel_template.py
#
# 8. repo discovery depth (the original bug searched to depth 3 and missed the mount).

def _mount_with_repo(depth):
    """A fake /kaggle/input whose repo sits `depth` directories below the mount root."""
    mount = _tmpdir() / "input"
    d = mount
    for i in range(depth):
        d = d / f"lvl{i}"
    d.mkdir(parents=True)
    (d / "pyproject.toml").write_text(MARKER_PYPROJECT)
    (d / "jax_rmhd").mkdir()
    (d / "jax_rmhd" / "__init__.py").write_text("VERSION = 'mini'\n")
    decoy = mount / "some-other-dataset"
    decoy.mkdir()
    (decoy / "pyproject.toml").write_text('[project]\nname = "not-jax-rmhd"\n')
    return mount, d


def test_find_repo_root_finds_the_marker_at_depth_4_and_5():
    for depth in (4, 5):
        mount, repo = _mount_with_repo(depth)
        with _capture() as buf:
            found = tpl.find_repo_root(input_root=str(mount),
                                       extract_root=str(_tmpdir() / "extract"))
        assert Path(found).resolve() == repo.resolve(), f"depth {depth}"
        assert "listing" in buf.getvalue()      # the bounded listing runs before the search


def test_find_repo_root_empty_mount_hints_at_a_still_processing_dataset():
    mount = _tmpdir() / "input"
    mount.mkdir()
    with _capture() as buf:
        exc = _raises(RuntimeError, tpl.find_repo_root, input_root=str(mount),
                      extract_root=str(_tmpdir() / "extract"))
    assert "no jax_rmhd checkout found" in str(exc)
    assert "still processing" in str(exc)
    assert "--skip-dataset" in str(exc)
    assert "Archives seen: none" in str(exc)
    assert "is not a directory" not in buf.getvalue()

    exc = _raises(RuntimeError, tpl.find_repo_root, input_root=str(_tmpdir() / "nope"),
                  extract_root=str(_tmpdir() / "extract"))
    assert "does not exist" in str(exc)


# 9. archive fallback -- Kaggle can serve a `--dir-mode zip` dataset unextracted.

def _make_repo_zip(path, top="jax_rmhd_src", marker=True, extra_member=None, pad=0):
    with zipfile.ZipFile(str(path), "w") as zf:
        if marker:
            zf.writestr(f"{top}/pyproject.toml", MARKER_PYPROJECT)
        zf.writestr(f"{top}/jax_rmhd/__init__.py", "VERSION = 'mini'\n" + "#" * pad)
        zf.writestr(f"{top}/README.md", "mini")
        if extra_member:
            zf.writestr(extra_member, "owned")


def test_zip_fallback_extracts_a_nested_repo():
    mount = _tmpdir() / "input"
    mount.mkdir()
    _make_repo_zip(mount / "src.zip")
    extract = _tmpdir() / "extract"
    with _capture() as buf:
        found = tpl.find_repo_root(input_root=str(mount), extract_root=str(extract))
    assert Path(found) == extract / "src" / "jax_rmhd_src"
    assert (Path(found) / "pyproject.toml").is_file()
    assert "extracting" in buf.getvalue()


def test_targz_fallback_extracts_a_nested_repo():
    base = _tmpdir()
    tree = base / "jax_rmhd_src"
    (tree / "jax_rmhd").mkdir(parents=True)
    (tree / "pyproject.toml").write_text(MARKER_PYPROJECT)
    (tree / "jax_rmhd" / "__init__.py").write_text("VERSION = 'mini'\n")
    mount = base / "input"
    mount.mkdir()
    with tarfile.open(str(mount / "src.tar.gz"), "w:gz") as tf:
        tf.add(str(tree), arcname="jax_rmhd_src")
    extract = base / "extract"
    with _capture():
        found = tpl.find_repo_root(input_root=str(mount), extract_root=str(extract))
    assert Path(found) == extract / "src" / "jax_rmhd_src"
    assert (Path(found) / "jax_rmhd" / "__init__.py").is_file()


def test_archive_without_a_pyproject_member_is_skipped():
    mount = _tmpdir() / "input"
    mount.mkdir()
    with zipfile.ZipFile(str(mount / "somebody_elses_data.zip"), "w") as zf:
        zf.writestr("data/train.csv", "a,b\n1,2\n")
    extract = _tmpdir() / "extract"
    with _capture() as buf:
        exc = _raises(RuntimeError, tpl.find_repo_root, input_root=str(mount),
                      extract_root=str(extract))
    assert "no pyproject.toml" in buf.getvalue()
    assert "extracted and searched: none" in str(exc)
    assert not (extract / "somebody_elses_data").exists()


def test_archive_with_a_traversal_member_is_refused_whole():
    base = _tmpdir()
    mount = base / "input"
    mount.mkdir()
    _make_repo_zip(mount / "src.zip", extra_member="../evil.txt")
    extract = base / "extract"
    extract.mkdir()
    with _capture() as buf:
        exc = _raises(RuntimeError, tpl.find_repo_root, input_root=str(mount),
                      extract_root=str(extract))
    assert "REFUSING" in buf.getvalue()
    assert "escape the extraction directory" in buf.getvalue()
    assert "no jax_rmhd checkout found" in str(exc)
    # nothing written -- not the good members, and certainly not the escaping one
    assert not (extract / "src").exists()
    assert not (extract / "evil.txt").exists()
    assert not (base / "evil.txt").exists()
    assert list(extract.iterdir()) == []


def test_oversized_archive_is_skipped():
    d = _tmpdir()
    z = d / "big.zip"
    _make_repo_zip(z, pad=200)
    extract = d / "extract"
    old = tpl.MAX_ARCHIVE_BYTES
    try:
        # patch the threshold rather than writing a >2 GiB file
        tpl.MAX_ARCHIVE_BYTES = 8
        with _capture() as buf:
            assert tpl._extract_repo_archive(str(z), str(extract)) is None
    finally:
        tpl.MAX_ARCHIVE_BYTES = old
    assert "archive limit" in buf.getvalue()
    assert not extract.exists()
    # control: with the shipped threshold the very same archive IS unpacked
    with _capture():
        dest = tpl._extract_repo_archive(str(z), str(extract))
    assert dest is not None and Path(dest) == extract / "big"


# 10. the pre-search listing must stay bounded: a dataset mount can hold millions of files.

def test_list_tree_is_bounded_in_entries_and_depth():
    root = _tmpdir() / "mount"
    root.mkdir()
    for i in range(25):
        (root / f"f{i:02d}.txt").write_text("x")
    with _capture() as buf:
        tpl.list_tree(str(root), max_depth=4, max_entries=20)
    out = buf.getvalue()
    assert "... 5 more entries (truncated)" in out
    assert out.count(".txt") == 20

    # the DEFAULT entry bound is what find_repo_root() actually calls this with
    with _capture() as buf_def:
        tpl.list_tree(str(root))
    assert "more entries (truncated)" in buf_def.getvalue()
    assert buf_def.getvalue().count(".txt") == 20

    deep_root = _tmpdir() / "deep"
    deep = deep_root / "lvl_a" / "lvl_b" / "lvl_c" / "lvl_d" / "lvl_e"
    deep.mkdir(parents=True)
    (deep / "deepfile.dat").write_text("x")
    with _capture() as buf2:
        tpl.list_tree(str(deep_root), max_depth=2, max_entries=50)
    out2 = buf2.getvalue()
    assert "lvl_a/" in out2 and "lvl_b/" in out2
    assert "lvl_c" not in out2 and "deepfile.dat" not in out2
    assert "truncated" not in out2

    with _capture() as buf3:
        tpl.list_tree(str(deep_root))              # default max_depth=4
    out3 = buf3.getvalue()
    assert "lvl_d/" in out3
    assert "lvl_e" not in out3 and "deepfile.dat" not in out3


# 11. field failure #2: pip cannot build in the read-only /kaggle/input mount.

def _readonly_repo():
    base = _tmpdir()
    repo = base / "mount" / "repo"
    (repo / "jax_rmhd").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(MARKER_PYPROJECT)
    (repo / "jax_rmhd" / "__init__.py").write_text("VERSION = 'mini'\n")
    _chmod_tree(repo, 0o555, 0o444)
    return base, repo


def test_ensure_writable_repo_copies_a_readonly_mount_and_the_copy_is_writable():
    base, repo = _readonly_repo()
    scratch = base / "_repo_src"
    try:
        with _capture() as buf:
            got = tpl.ensure_writable_repo(str(repo), scratch_root=str(scratch))
        assert Path(got) == scratch
        assert "read-only" in buf.getvalue()

        # THE regression: shutil.copytree preserves mode bits, so without the explicit
        # chmod the copy is every bit as unwritable as the mount it came from.
        copied = scratch / "jax_rmhd" / "__init__.py"
        assert copied.is_file()
        assert os.access(str(copied), os.W_OK)
        assert os.access(str(scratch / "jax_rmhd"), os.W_OK)
        with open(str(copied), "a") as fh:              # really writable, not just per bits
            fh.write("# touched by the test\n")
        (scratch / "jax_rmhd.egg-info").mkdir()         # what pip would need to do

        # the source mount is untouched
        assert not os.access(str(repo / "pyproject.toml"), os.W_OK)
        assert (repo / "jax_rmhd" / "__init__.py").read_text() == "VERSION = 'mini'\n"
        assert not (repo / "jax_rmhd.egg-info").exists()

        # a stale, itself-read-only copy from a previous run must not wedge the next one
        _chmod_tree(scratch, 0o555, 0o444)
        with _capture():
            got2 = tpl.ensure_writable_repo(str(repo), scratch_root=str(scratch))
        assert Path(got2) == scratch
        assert os.access(str(scratch / "jax_rmhd" / "__init__.py"), os.W_OK)
        # rebuilt from scratch, not merged with the previous attempt
        assert (scratch / "jax_rmhd" / "__init__.py").read_text() == "VERSION = 'mini'\n"
        assert not (scratch / "jax_rmhd.egg-info").exists()
    finally:
        _chmod_tree(repo, 0o755, 0o644)
        _chmod_tree(scratch, 0o755, 0o644)


def test_ensure_writable_repo_leaves_an_already_writable_tree_alone():
    base = _tmpdir()
    repo = base / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(MARKER_PYPROJECT)
    scratch = base / "_repo_src"
    with _capture() as buf:
        got = tpl.ensure_writable_repo(str(repo), scratch_root=str(scratch))
    assert Path(got) == repo                    # e.g. the archive-extraction fallback
    assert not scratch.exists()
    assert "read-only" not in buf.getvalue()
    # and the write probe left nothing behind
    assert sorted(p.name for p in repo.iterdir()) == ["pyproject.toml"]


# 12. RMHD_PRECISION is read at import time -- so the ORDER of main()'s calls is the rule.

def test_main_sets_precision_first_and_enables_import_between_install_and_check_gpu():
    tree = ast.parse(TEMPLATE_PY.read_text())
    own = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(mains) == 1
    seen = [(n.lineno, n.col_offset, n.func.id)
            for n in ast.walk(mains[0])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id in own and n.func.id != "log"]
    order = [name for _, _, name in sorted(seen)]

    assert order[0] == "set_precision", f"main() call order: {order}"
    assert order.count("enable_in_process_import") == 1
    assert order.count("ensure_writable_repo") == 1
    i_install = order.index("install_repo")
    # field failure #2 again: pip only ever sees a tree it can write build metadata into
    assert order.index("find_repo_root") < order.index("ensure_writable_repo") < i_install
    i_enable = order.index("enable_in_process_import")
    i_gpu = order.index("check_gpu")
    assert i_install < i_enable < i_gpu, f"main() call order: {order}"
    # check_gpu is the first thing that imports jax; the verification import inside
    # enable_in_process_import is the first `import jax_rmhd`, and both must follow
    # set_precision.
    assert order.index("set_precision") < i_enable


def test_set_precision_exports_the_env_var():
    with _env(RMHD_PRECISION=None):
        with _capture() as buf:
            tpl.set_precision(32)
        assert os.environ["RMHD_PRECISION"] == "32"
        assert "RMHD_PRECISION=32" in buf.getvalue()


# 13. field failure #3: `pip install -e` is invisible to the process that ran it.

_IN_PROCESS_IMPORT_PROBE = '''\
import importlib.util, os, sys

tpl_path, repo, expect = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("_kl_tpl", tpl_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if importlib.util.find_spec("jax_rmhd") is not None:
    print("PRE: importable -- this probe is not isolated")
    sys.exit(2)
print("PRE: not importable")

if expect == "ok":
    mod.enable_in_process_import(repo)
    import jax_rmhd
    assert jax_rmhd.STUB == "stub-ok", jax_rmhd.STUB
    pkg_root = os.path.dirname(os.path.dirname(os.path.realpath(jax_rmhd.__file__)))
    assert pkg_root == os.path.realpath(repo), pkg_root
    print("POST: OK", jax_rmhd.STUB)
else:
    try:
        mod.enable_in_process_import(repo)
    except RuntimeError as exc:
        assert "sys.path" in str(exc), str(exc)
        assert os.path.join(repo, "jax_rmhd", "__init__.py") in str(exc), str(exc)
        print("POST: RuntimeError names the expected package path")
    else:
        print("POST: no RuntimeError from an empty repo root")
        sys.exit(3)
'''


def _run_import_probe(repo, expect):
    """`enable_in_process_import` needs a real `import jax_rmhd`, so it runs in a SUBPROCESS
    with -S/-E (no site-packages, no PYTHONPATH): on a laptop where the real jax_rmhd IS
    installed it must not be able to satisfy the import, and this process's sys.path /
    sys.modules must not be touched at all."""
    probe = _tmpdir() / "probe.py"
    probe.write_text(_IN_PROCESS_IMPORT_PROBE)
    neutral = _tmpdir() / "cwd"
    neutral.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
    proc = subprocess.run([sys.executable, "-S", "-E", str(probe), str(TEMPLATE_PY),
                           str(repo), expect],
                          cwd=str(neutral), capture_output=True, text=True, env=env)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def test_enable_in_process_import_makes_a_flat_repo_importable():
    repo = _tmpdir() / "repo"
    (repo / "jax_rmhd").mkdir(parents=True)
    (repo / "jax_rmhd" / "__init__.py").write_text("STUB = 'stub-ok'\n")   # no dependencies
    (repo / "pyproject.toml").write_text(MARKER_PYPROJECT)

    rc, out = _run_import_probe(repo, "ok")
    assert rc == 0, out
    assert "PRE: not importable" in out       # find_spec fails before the fix runs
    assert "sys.path[0] = " in out            # the fix itself, logged
    assert "import jax_rmhd: OK" in out       # the template's own verification import
    assert "POST: OK stub-ok" in out
    # nothing leaked into the test process
    assert str(repo) not in sys.path
    assert ("jax_rmhd" in sys.modules) == _JAX_RMHD_WAS_IMPORTED


def test_enable_in_process_import_error_names_the_expected_package_path():
    empty = _tmpdir() / "empty-repo"
    empty.mkdir()
    rc, out = _run_import_probe(empty, "raises")
    assert rc == 0, out
    assert "POST: RuntimeError names the expected package path" in out


# 14. driver dispatch: the make_data contract, the budget, and the always-a-zip finally.

def test_resolve_entry_auto_loops_make_data_until_true():
    calls = []

    def make_data(snap_path, **kwargs):
        calls.append((snap_path, kwargs))
        return len(calls) >= 3

    with _capture() as buf:
        done = tpl.resolve_entry(_fake_module(make_data=make_data), "auto",
                                 {"wall_budget": 100.0}, "/snaps", 10000.0,
                                 now=_fake_clock(1.0))
    assert done is True
    assert len(calls) == 3
    assert all(c == ("/snaps", {"wall_budget": 100.0}) for c in calls)
    assert "make_data call 3: done=True" in buf.getvalue()
    assert "WARNING" not in buf.getvalue()          # wall_budget is well under the session


def test_drive_make_data_reports_a_partial_run_when_the_budget_runs_out():
    calls = []

    def make_data(snap_path, **kwargs):
        calls.append(snap_path)
        return False

    with _capture() as buf:
        done = tpl.drive_make_data(make_data, "/snaps", {"wall_budget": 50.0}, 250.0,
                                   now=_fake_clock(100.0))
    out = buf.getvalue()
    assert done is False                       # never claims the run reached t_end
    assert len(calls) == 3                     # elapsed 100, 200, 300 > 250
    assert "session budget exhausted" in out
    assert "PARTIAL RUN" in out


def test_warn_wall_budget_warns_on_missing_and_on_half_the_session():
    with _capture() as b1:
        tpl.warn_wall_budget({}, 1000.0)
    assert "WARNING" in b1.getvalue() and "no 'wall_budget'" in b1.getvalue()

    with _capture() as b2:
        tpl.warn_wall_budget({"wall_budget": 500}, 1000.0)
    assert "WARNING" in b2.getvalue() and "not comfortably under" in b2.getvalue()

    with _capture() as b3:
        tpl.warn_wall_budget({"wall_budget": 499}, 1000.0)
    assert "WARNING" not in b3.getvalue()


def test_resolve_entry_auto_falls_back_to_main_then_to_runpy():
    seen = {}
    with _capture() as buf:
        done = tpl.resolve_entry(_fake_module(main=lambda **kw: seen.update(kw)), "auto",
                                 {"t_end": 3}, "/snaps", 100.0)
    assert done is True and seen == {"t_end": 3}
    assert "no make_data" in buf.getvalue()

    d = _tmpdir()
    marker = d / "ran.txt"
    script = d / "plain_run.py"
    script.write_text("import pathlib\n"
                      "if __name__ == '__main__':\n"
                      f"    pathlib.Path({str(marker)!r}).write_text('ran')\n")
    with _capture() as buf2:
        done2 = tpl.resolve_entry(_fake_module(), "auto", {}, "/snaps", 100.0,
                                  run_file=str(script))
    assert done2 is True
    assert marker.read_text() == "ran"
    assert "executing the file as __main__" in buf2.getvalue()


def test_resolve_entry_explicit_entry_and_unknown_entry():
    seen = {}
    mod = _fake_module(custom=lambda **kw: seen.update(kw),
                       make_data=lambda sp, **kw: True)
    with _capture() as buf:
        assert tpl.resolve_entry(mod, "custom", {"a": 1}, "/snaps", 100.0) is True
    assert seen == {"a": 1}                    # explicit entry wins over make_data
    assert "calling custom" in buf.getvalue()

    got = []
    mod2 = _fake_module(make_data=lambda sp, **kw: got.append(sp) or True)
    with _capture():
        assert tpl.resolve_entry(mod2, "make_data", {}, "/snaps", 100.0) is True
    assert got == ["/snaps"]                   # explicit make_data keeps its own signature

    exc = _raises(RuntimeError, tpl.resolve_entry, _fake_module(), "nope", {}, "/snaps", 100.0)
    assert "not a callable" in str(exc)


def test_maybe_report_swallows_a_broken_report():
    def boom(snap_path):
        raise ValueError("no data yet")

    with _capture() as buf:
        tpl.maybe_report(_fake_module(report=boom), "/snaps")     # must not raise
    assert "report() raised ValueError" in buf.getvalue()
    assert "continuing to the zip" in buf.getvalue()

    with _capture() as buf2:
        tpl.maybe_report(_fake_module(), "/snaps")
    assert buf2.getvalue() == ""

    got = []
    with _capture():
        tpl.maybe_report(_fake_module(report=got.append), "/snaps")
    assert got == ["/snaps"]


def test_zip_workdir_writes_one_archive_with_relative_arcnames():
    base = _tmpdir()
    workdir = base / "work"
    (workdir / "checkpoints" / "step_1").mkdir(parents=True)
    (workdir / "log.txt").write_text("hi")
    (workdir / "checkpoints" / "step_1" / "a.bin").write_bytes(b"\0" * 32)
    working = base / "working"

    with _capture() as buf:
        out = tpl.zip_workdir(str(workdir), "demo_run", working=str(working))
    assert Path(out) == working / "demo_run_output.zip"
    # exactly ONE file lands in /kaggle/working (it is capped at ~500 files)
    assert sorted(p.name for p in working.iterdir()) == ["demo_run_output.zip"]
    with zipfile.ZipFile(str(out)) as zf:
        names = sorted(zf.namelist())
    assert names == ["checkpoints/step_1/a.bin", "log.txt"]
    assert "2 files" in buf.getvalue()


# ================================================================ standalone runner

def _collect():
    # globals() keeps definition order (3.7+), so failures read in source order.
    return [(name, fn) for name, fn in list(globals().items())
            if name.startswith("test_") and callable(fn)]


def _main():
    tests = _collect()
    t0 = time.time()
    failures = []
    for name, fn in tests:
        try:
            fn()
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit) as exc:                   # noqa: BLE001
            failures.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    dt = time.time() - t0
    print(f"\n{len(tests) - len(failures)} passed, {len(failures)} failed "
          f"in {dt:.1f}s")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
