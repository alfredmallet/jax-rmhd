#!/usr/bin/env python3
"""Launch any jax_rmhd run driver on a Kaggle GPU kernel from the command line.

The whole point is that nothing here needs a browser: `push` uploads the repo as a
Kaggle Dataset and a generated script Kernel that imports a run driver out of it,
`status` polls, `pull` downloads the kernel's output zip, and `run` chains all three.
See kaggle_launcher/README.md for the setup + quota story.

stdlib only, on purpose -- this file runs on the USER's laptop before any of the
project's own dependencies matter, and the sandbox that develops it has neither the
`kaggle` CLI nor a token. Everything is therefore exercisable with --dry-run, which
stages the exact directories that would be pushed and prints the exact commands that
would run, without invoking the kaggle binary.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "_kernel_template.py"
# the sentinel line in _kernel_template.py that gets rewritten to the real config.
# A line-for-line replacement (not str.format) because the template is full of braces.
CONFIG_SENTINEL = "CONFIG = None  # __KAGGLE_LAUNCHER_CONFIG__"

KAGGLE_HINT = ("`kaggle` not found on PATH. Install it with `pip install kaggle` and put an "
               "API token at ~/.kaggle/kaggle.json (Kaggle -> Account -> Create New API Token).")

# Kaggle rejects dataset/kernel titles (and the slugs derived from them) below 5 characters.
MIN_SLUG_LEN = 5
# marker file written into the staging ROOT when this tool creates it; nothing is ever
# rmtree'd out of a staging root that does not carry it (see wipe()).
STAGING_SENTINEL = ".kaggle_launcher_staging"


# ---------------------------------------------------------------- small helpers

def slugify(text):
    # Kaggle slugs: lowercase alphanumerics and dashes only. Everything else collapses to
    # a single dash; leading/trailing dashes are trimmed. `gdi_3d_run.py` -> `gdi-3d-run`.
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s or "run"


def pad_slug(text):
    """slugify() + Kaggle's 5-character floor, applied to the SLUG, not just the title.

    The slug is what goes into `user/slug` ids, so padding only the title would push a
    kernel/dataset whose id Kaggle refuses (or silently renames). Pad here, once, and let
    both the id and the title come out of the same string.
    """
    s = slugify(text)
    return s if len(s) >= MIN_SLUG_LEN else f"{s}-run"


def repo_root():
    # prefer git (correct even if kaggle_launcher/ is symlinked or vendored somewhere odd);
    # fall back to the parent of this folder, which is the layout we ship.
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(HERE),
                             capture_output=True, text=True, check=True)
        return Path(out.stdout.strip()).resolve()
    except Exception:
        return HERE.parent


def kaggle_username(dry_run=False):
    token = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if token.is_file():
        try:
            name = json.loads(token.read_text()).get("username")
        except Exception as exc:
            raise SystemExit(f"could not parse {token}: {exc}")
        if name:
            return name
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    if dry_run:
        # so the whole staging path is testable on a machine with no Kaggle credentials.
        print("[dry-run] no ~/.kaggle/kaggle.json found -- using placeholder username USERNAME")
        return "USERNAME"
    raise SystemExit(f"no Kaggle username: expected a 'username' key in {token} "
                     "(Kaggle -> Account -> Create New API Token), or $KAGGLE_USERNAME.")


def sh(cmd, dry_run, check=True, capture=False):
    """Run (or, under --dry-run, merely print) one kaggle command."""
    printable = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"[dry-run] {printable}")
        return 0, ""
    print(f"$ {printable}")
    proc = subprocess.run([str(c) for c in cmd], capture_output=capture, text=True)
    out = (proc.stdout or "") + (proc.stderr or "") if capture else ""
    if capture and out:
        print(out.rstrip())
    if check and proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {printable}")
    return proc.returncode, out


def require_kaggle(dry_run):
    if not dry_run and shutil.which("kaggle") is None:
        raise SystemExit(KAGGLE_HINT)


def _dataset_status(ident):
    """One `kaggle datasets status` call -> its combined output text (never raises).

    Deliberately not sh(): the poll below runs this dozens of times and wants one compact
    line per poll, not the full `$ cmd` + output echo.
    """
    try:
        proc = subprocess.run(["kaggle", "datasets", "status", ident],
                              capture_output=True, text=True)
    except OSError as exc:
        return f"could not run kaggle datasets status: {exc}"
    return ((proc.stdout or "") + (proc.stderr or "")).strip()


def wait_dataset_ready(ident, dry_run, timeout=600.0, interval=10.0, unknown_grace=60.0,
                       unknown_limit=3, runner=None, now=time.time, sleep=time.sleep):
    """Block until Kaggle reports the dataset ready -- BEFORE the kernel is pushed.

    Field failure (first real run, 2026-08-03): `datasets version` returns as soon as the
    upload lands, but Kaggle keeps *processing* the new version for a while afterwards. A
    kernel pushed into that window starts against a mount that is empty (or serves the raw
    archive), and dies in the kernel's find_repo_root() -- several minutes and a slice of
    the weekly GPU quota later, with a log that cannot distinguish this from a genuinely
    missing dataset. So we wait here instead, where failing is free.

    The CLI's status wording is not a stable API, so an unrecognized answer must not be
    fatal: after `unknown_limit` consecutive unrecognized replies AND `unknown_grace`
    seconds since the push, warn and continue. Only explicitly not-done states
    (pending/processing) can burn the whole timeout.

    timeout/interval/grace and the clock are parameters so this is testable without
    sleeping through a real poll cycle.
    """
    if dry_run:
        sh(["kaggle", "datasets", "status", ident], True)
        print(f"[dry-run] (polled every {interval:.0f}s until it reports ready, up to "
              f"--dataset-timeout {timeout:.0f}s; the kernel is pushed only after that)")
        return True
    runner = runner or _dataset_status
    t0 = now()
    unknown = 0
    print(f"waiting for dataset {ident} to finish processing "
          f"(every {interval:.0f}s, up to {timeout:.0f}s)")
    while True:
        out = runner(ident)
        low = out.lower()
        first = (out.splitlines() or [""])[0][:160]
        elapsed = now() - t0
        stamp = time.strftime("%H:%M:%S")
        if "ready" in low:
            print(f"[{stamp}] dataset ready ({elapsed:.0f}s)")
            return True
        if "pending" in low or "processing" in low:
            unknown = 0
            print(f"[{stamp}] dataset still processing ({elapsed:.0f}s): {first}")
        else:
            unknown += 1
            print(f"[{stamp}] unrecognized `datasets status` output ({elapsed:.0f}s): {first!r}")
            if unknown > unknown_limit and elapsed >= unknown_grace:
                print(f"WARNING: `kaggle datasets status {ident}` has not said 'ready', but its "
                      f"output is unrecognized after {unknown} polls / {elapsed:.0f}s. That text "
                      "is not a stable API, so this is treated as ready and the kernel push "
                      "continues. If the kernel then cannot find the repo under /kaggle/input, "
                      "the dataset was still processing: re-run with --skip-dataset once the "
                      "dataset page shows the new version.")
                return True
        if elapsed + interval > timeout:
            raise SystemExit(
                f"dataset {ident} was still not ready {elapsed:.0f}s after the push "
                f"(--dataset-timeout {timeout:.0f}s). NOT pushing the kernel: a dataset that "
                "is still processing mounts empty, and the kernel would only fail later on a "
                f"GPU. Check https://www.kaggle.com/datasets/{ident} -- once it shows the new "
                "version, re-run the same command with --skip-dataset (it reuses the dataset "
                "already on Kaggle and pushes the kernel only).")
        sleep(interval)


def require_tracked(rel, root):
    """The dataset ships `git ls-files` output, so an untracked driver never arrives.

    Checked here, before any upload: otherwise the failure surfaces only once the kernel is
    on a GPU, several minutes and a chunk of weekly quota later.
    """
    rc = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel], cwd=str(root),
                        capture_output=True, text=True).returncode
    if rc != 0:
        raise SystemExit(f"{rel} is not git-tracked; `git add {rel}` it first -- the dataset "
                         "ships tracked files only, so an untracked driver would upload fine "
                         "and then be missing at kernel runtime.")


# ---------------------------------------------------------------- staging

def prepare_staging(staging):
    """Create (or adopt) the staging root, and mark it as ours so wipe() may recurse into it."""
    sentinel = staging / STAGING_SENTINEL
    if staging.exists() and not sentinel.is_file():
        raise SystemExit(
            f"staging directory {staging} already exists and was not created by this tool "
            f"(no {STAGING_SENTINEL} marker). Refusing to touch it -- pass a different "
            "--staging-dir, or delete that directory yourself if it really is scratch.")
    staging.mkdir(parents=True, exist_ok=True)
    if not sentinel.is_file():
        sentinel.write_text("staging directory managed by kaggle_launcher/launch.py; "
                            "its contents are rebuilt on every push.\n")
    return staging


def wipe(path, staging):
    """rmtree a staging subdirectory, but only inside a staging root we own."""
    if not path.exists():
        return
    if not (staging / STAGING_SENTINEL).is_file():
        raise SystemExit(f"refusing to delete {path}: {staging} carries no "
                         f"{STAGING_SENTINEL} marker, so it is not a staging root of ours.")
    shutil.rmtree(path)


def stage_dataset(staging, dataset_slug, username, root):
    """Copy the git-tracked WORKING TREE into <staging>/dataset/<slug>/.

    `git ls-files` rather than a plain copytree for two reasons: it picks up uncommitted
    edits to tracked files (you nearly always want to test the thing you just changed),
    and it excludes examples/data, __pycache__, checkpoints and other untracked bulk --
    the tracked tree is ~23 MB, well inside Kaggle's dataset limits.
    """
    dataset_slug = pad_slug(dataset_slug)
    dsdir = staging / "dataset"
    wipe(dsdir, staging)              # rebuild from scratch every push: no stale files
    payload = dsdir / dataset_slug
    payload.mkdir(parents=True)

    listed = subprocess.run(["git", "ls-files", "-z"], cwd=str(root),
                            capture_output=True, check=True).stdout
    names = [n.decode() for n in listed.split(b"\0") if n]
    n_copied = 0
    for rel in names:
        src = root / rel
        if not src.is_file():
            continue              # tracked-but-deleted, or a submodule directory entry
        dst = payload / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n_copied += 1

    meta = {"title": dataset_slug,
            "id": f"{username}/{dataset_slug}",
            "licenses": [{"name": "CC0-1.0"}]}
    (dsdir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    size_mb = sum(f.stat().st_size for f in payload.rglob("*") if f.is_file()) / 1e6
    print(f"staged dataset: {n_copied} tracked files, {size_mb:.1f} MB -> {dsdir}")
    return dsdir


def config_line(config):
    """The replacement for CONFIG_SENTINEL: a `json.loads` of the config's JSON text.

    NOT a bare `CONFIG = <json>`: JSON's true/false/null are not Python, so a single
    `--entry-kwargs '{"adaptive_timestep": true}'` would push a kernel that NameErrors
    after the GPU spin-up. Going back through json.loads keeps the values' types exactly
    what the user typed, and repr() of the JSON text handles every quoting case.
    """
    return "CONFIG = json.loads(" + repr(json.dumps(config)) + ")  # __KAGGLE_LAUNCHER_CONFIG__"


def stage_kernel(staging, kernel_slug, dataset_slug, username, machine, config):
    kernel_slug, dataset_slug = pad_slug(kernel_slug), pad_slug(dataset_slug)
    kdir = staging / "kernel" / kernel_slug
    wipe(kdir, staging)
    kdir.mkdir(parents=True)

    template = TEMPLATE.read_text()
    if CONFIG_SENTINEL not in template:
        raise SystemExit(f"sentinel line missing from {TEMPLATE}: {CONFIG_SENTINEL!r}")
    body = template.replace(CONFIG_SENTINEL, config_line(config))
    (kdir / "kernel.py").write_text(body)

    meta = {"id": f"{username}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",          # pip installs need it
            "machine_shape": machine,           # e.g. NvidiaTeslaP100 / NvidiaTeslaT4
            "dataset_sources": [f"{username}/{dataset_slug}"],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": []}          # newer kaggle-api schema; empty is always safe
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"staged kernel: {kdir}")
    return kdir


# ---------------------------------------------------------------- subcommands

def resolve_run_file(arg, root):
    """Turn the CLI argument into (absolute path, repo-relative path)."""
    p = Path(arg)
    cand = p if p.is_absolute() else (Path.cwd() / p)
    cand = cand.resolve()
    if not cand.is_file():
        cand = (root / arg).resolve()
    if not cand.is_file():
        raise SystemExit(f"run driver not found: {arg}")
    try:
        rel = cand.relative_to(root)
    except ValueError:
        # it has to live inside the repo -- the dataset is the only thing that travels.
        raise SystemExit(f"{cand} is outside the repo ({root}); it would never reach Kaggle. "
                         "Move the driver into the repo (e.g. examples/) first.")
    return cand, rel.as_posix()


def parse_kwargs(text):
    # checked here, before the 22 MB dataset upload -- a JSON typo should cost a second,
    # not a push. Shell quoting bites here: use single quotes around the JSON.
    if not text:
        return {}
    try:
        kw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--entry-kwargs is not valid JSON ({exc}): {text!r}\n"
                         "  expected e.g. --entry-kwargs '{\"t_end\": 30.0, "
                         "\"wall_budget\": 3300}'")
    if not isinstance(kw, dict):
        raise SystemExit(f"--entry-kwargs must be a JSON object, got {type(kw).__name__}")
    return kw


def kernel_slug_for(args, run_rel=None):
    if args.kernel_slug:
        return pad_slug(args.kernel_slug)
    return pad_slug(Path(run_rel).stem)


def cmd_push(args, root):
    run_abs, run_rel = resolve_run_file(args.run, root)
    # before the username lookup and long before any upload: cheap, and the failure mode it
    # catches (untracked driver) is otherwise invisible until the kernel is already running.
    require_tracked(run_rel, root)
    username = kaggle_username(args.dry_run)
    require_kaggle(args.dry_run)
    dataset_slug = pad_slug(args.dataset_slug)
    kslug = kernel_slug_for(args, run_rel)
    staging = prepare_staging(Path(args.staging_dir) if args.staging_dir
                              else (HERE / ".staging"))

    config = {"dataset_slug": dataset_slug,
              "run_relpath": run_rel,
              "entry": args.entry,
              "entry_kwargs": parse_kwargs(args.entry_kwargs),
              "precision": str(args.precision),
              "session_budget_s": float(args.session_budget)}

    if args.skip_dataset:
        print("skipping dataset push (--skip-dataset): the kernel will use whatever version "
              f"of {username}/{dataset_slug} is already on Kaggle")
    else:
        dsdir = stage_dataset(staging, dataset_slug, username, root)
        if not (dsdir / dataset_slug / run_rel).is_file():
            raise SystemExit(f"{run_rel} did not make it into the staged dataset at "
                             f"{dsdir / dataset_slug} -- it must be a git-tracked regular "
                             "file (submodule contents and tracked-but-deleted paths are "
                             "skipped).")
        msg = args.dataset_message or f"jax_rmhd source {time.strftime('%Y-%m-%d %H:%M:%S')}"
        # `version` first, `create` only as the fallback: after the first push, `create`
        # errors out, and there is no cheap CLI probe for "does this dataset exist".
        rc, _ = sh(["kaggle", "datasets", "version", "-p", dsdir, "--dir-mode", "zip",
                    "-m", msg], args.dry_run, check=False, capture=True)
        if args.dry_run:
            print("[dry-run] (if the above fails because the dataset does not exist yet:)")
            sh(["kaggle", "datasets", "create", "-p", dsdir, "--dir-mode", "zip"], True)
        elif rc != 0:
            print("dataset version failed -- assuming it does not exist yet, creating it")
            sh(["kaggle", "datasets", "create", "-p", dsdir, "--dir-mode", "zip"], False,
               capture=True)
        # ALWAYS before the kernel push: a version/create returns on upload, not on
        # processing, and a kernel that starts against a still-processing dataset fails
        # on the GPU with a misleading "no jax_rmhd checkout found" (see wait_dataset_ready).
        wait_dataset_ready(f"{username}/{dataset_slug}", args.dry_run,
                           timeout=float(getattr(args, "dataset_timeout", 600.0)))

    kdir = stage_kernel(staging, kslug, dataset_slug, username, args.machine, config)
    sh(["kaggle", "kernels", "push", "-p", kdir], args.dry_run, capture=True)

    print(f"\nkernel id : {username}/{kslug}")
    print(f"kernel url: https://www.kaggle.com/code/{username}/{kslug}")
    print(f"machine   : {args.machine}   precision: RMHD_PRECISION={args.precision}")
    print(f"driver    : {run_rel}  entry={args.entry}  kwargs={config['entry_kwargs']}")
    return f"{username}/{kslug}"


def slug_from_arg(arg, root, args):
    """`status`/`pull` accept either a run driver path or a bare/qualified kernel slug."""
    if "/" in arg and not arg.endswith(".py"):
        return arg                                  # already user/slug
    if arg.endswith(".py"):
        _, rel = resolve_run_file(arg, root)
        base = kernel_slug_for(args, rel)
    else:
        base = pad_slug(args.kernel_slug or arg)
    return f"{kaggle_username(args.dry_run)}/{base}"


def cmd_status(args, root, quiet=False):
    require_kaggle(args.dry_run)
    ident = slug_from_arg(args.target, root, args)
    _, out = sh(["kaggle", "kernels", "status", ident], args.dry_run, check=False,
                capture=not quiet)
    return out


def cmd_pull(args, root):
    require_kaggle(args.dry_run)
    ident = slug_from_arg(args.target, root, args)
    outdir = Path(args.output) if args.output else (HERE / "output" / ident.split("/")[-1])
    outdir.mkdir(parents=True, exist_ok=True)
    sh(["kaggle", "kernels", "output", ident, "-p", outdir], args.dry_run, capture=True)
    print(f"output -> {outdir}")
    return outdir


def cmd_run(args, root):
    ident = cmd_push(args, root)
    if args.no_wait:
        print("\n--no-wait: not polling. Check with "
              f"`python {Path(__file__).name} status {ident}`.")
        return
    print(f"\npolling every {args.poll_interval}s (Ctrl-C is safe -- the kernel keeps running)")
    args.target = ident
    while True:
        out = cmd_status(args, root).lower()
        if args.dry_run:
            print("[dry-run] would poll until the status text says complete/error, then pull")
            break
        stamp = time.strftime("%H:%M:%S")
        if "complete" in out:
            print(f"[{stamp}] complete")
            break
        if "error" in out or "cancel" in out:
            print(f"[{stamp}] kernel failed/cancelled -- pulling the log anyway")
            break
        print(f"[{stamp}] still running")
        time.sleep(args.poll_interval)
    cmd_pull(args, root)


# ---------------------------------------------------------------- CLI

def build_parser():
    p = argparse.ArgumentParser(prog="launch.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        # genuinely shared: every subcommand needs the kernel identity and the dry switch.
        sp.add_argument("--kernel-slug", default="", help="default: slug of the run file stem")
        sp.add_argument("--dry-run", action="store_true",
                        help="stage everything and print the kaggle commands, run none of them")

    def push_only(sp):
        # everything that only means something while building/uploading a kernel. Kept off
        # status/pull, where they would parse fine and do nothing.
        sp.add_argument("--machine", default="NvidiaTeslaP100",
                        help="Kaggle machine_shape (NvidiaTeslaP100, NvidiaTeslaT4, Tpu1VmV38)")
        sp.add_argument("--precision", choices=["32", "64"], default="64",
                        help="RMHD_PRECISION set in the kernel before importing jax_rmhd")
        sp.add_argument("--entry", default="auto",
                        help="'auto' (make_data -> main -> __main__), or an explicit callable")
        sp.add_argument("--entry-kwargs", default="",
                        help="JSON dict forwarded to the entry, "
                             "e.g. '{\"t_end\": 30.0, \"wall_budget\": 3300}'")
        sp.add_argument("--session-budget", type=float, default=39600.0,
                        help="on-kernel wall-clock guard in seconds (default 39600 = 11 h)")
        sp.add_argument("--dataset-slug", default="jax-rmhd-src")
        sp.add_argument("--dataset-message", default="")
        sp.add_argument("--skip-dataset", action="store_true",
                        help="reuse the dataset already on Kaggle (fast kernel-only iteration)")
        sp.add_argument("--dataset-timeout", type=float, default=600.0,
                        help="seconds to wait for Kaggle to finish processing the pushed "
                             "dataset before the kernel is pushed (default 600)")
        sp.add_argument("--staging-dir", default="",
                        help="where to build the upload directories "
                             "(default kaggle_launcher/.staging)")

    sp = sub.add_parser("push", help="push dataset + kernel")
    sp.add_argument("run")
    common(sp)
    push_only(sp)

    sp = sub.add_parser("status", help="kernel status")
    sp.add_argument("target", help="run driver path, kernel slug, or user/slug")
    common(sp)

    sp = sub.add_parser("pull", help="download kernel output")
    sp.add_argument("target", help="run driver path, kernel slug, or user/slug")
    sp.add_argument("-o", "--output", default="")
    common(sp)

    sp = sub.add_parser("run", help="push, poll to completion, pull")
    sp.add_argument("run")
    sp.add_argument("-o", "--output", default="")
    sp.add_argument("--poll-interval", type=float, default=60.0)
    sp.add_argument("--no-wait", action="store_true")
    common(sp)
    push_only(sp)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = repo_root()
    if args.cmd == "push":
        cmd_push(args, root)
    elif args.cmd == "status":
        cmd_status(args, root)
    elif args.cmd == "pull":
        cmd_pull(args, root)
    elif args.cmd == "run":
        cmd_run(args, root)


if __name__ == "__main__":
    sys.exit(main())
