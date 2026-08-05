# Running a jax_rmhd driver on a Kaggle GPU, from the command line

`launch.py` pushes this repo to Kaggle as a Dataset, generates a script Kernel that
imports one of the repo's run drivers out of that dataset, runs it on a GPU (P100 by
default), and downloads the result — all through the `kaggle` CLI, with no browser and no
notebook editing. It is the command-line generalization of
`examples/kaggle_forced_turbulence_256cubed.ipynb`, which does the same thing by hand for
one specific run.

## One-time setup

```
pip install kaggle
```

Then get an API token: kaggle.com → your account → Settings → *Create New API Token*. That
downloads `kaggle.json`; put it at `~/.kaggle/kaggle.json` and `chmod 600` it. `launch.py`
reads the `username` key out of that file (or `$KAGGLE_USERNAME`), so nothing else needs
configuring. Your Kaggle account must be phone-verified before it will give kernels a GPU
or internet access.

## The happy path

From the repo root:

```
python kaggle_launcher/launch.py run examples/gdi_3d_run.py \
    --entry-kwargs '{"t_end": 30.0, "wall_budget": 3300}'
```

That pushes the source dataset, pushes the kernel, polls every 60 s until Kaggle reports
the kernel complete (or failed), and downloads the output into
`kaggle_launcher/output/gdi-3d-run/`. `--no-wait` returns immediately after the push; you
can then check on it with `launch.py status examples/gdi_3d_run.py` and collect it later
with `launch.py pull examples/gdi_3d_run.py`. The subcommands also accept a bare kernel
slug (`gdi-3d-run`) or a fully qualified one (`myuser/gdi-3d-run`).

Add `--dry-run` to any subcommand to stage everything under `kaggle_launcher/.staging/`
and print the exact `kaggle` commands without running them. That is the way to check what
you are about to upload, and it works on a machine with no Kaggle credentials at all.

`test_launcher.py` is this directory's own regression suite — one test per field failure
listed below. Run it either way, like the rest of the repo's test files:

```
python -m pytest kaggle_launcher/test_launcher.py -q
python kaggle_launcher/test_launcher.py
```

It is hermetic and fast (well under a second): no network, no pip, no `kaggle` binary (a
stub CLI stands in), no `jax_rmhd` import, and everything — including a throwaway git repo
and a stub read-only "dataset mount" — lives in temporary directories.

Useful knobs: `--machine` (`NvidiaTeslaP100` default, `NvidiaTeslaT4`, `Tpu1VmV38`),
`--precision {32,64}` (default 64 — it sets `RMHD_PRECISION` in the kernel before
`jax_rmhd` is imported, which is the only moment that variable is read), `--entry` if the
driver's entry point is not auto-detectable, `--kernel-slug` to run several configurations
of the same driver side by side, `--skip-dataset` to iterate on the kernel without
re-uploading ~23 MB of source each time, `--dataset-timeout` (default 600 s) to bound the
wait for Kaggle to finish processing that upload, and `--staging-dir` to build the upload
directories somewhere other than `kaggle_launcher/.staging` (that directory is wiped and
rebuilt on every push, so the launcher refuses to touch one it did not create — it marks
its own with a `.kaggle_launcher_staging` file).

## What gets pushed where

The **dataset** (`--dataset-slug`, default `jax-rmhd-src`) is the repo's git-tracked
*working tree* — `git ls-files`, not a commit — so uncommitted edits to tracked files come
along, and untracked bulk like `examples/data/` and stray checkpoints does not. It is
about 23 MB. Every `push` re-versions it; the first push falls back to `datasets create`.
Because only tracked files travel, a driver you have not `git add`ed yet cannot reach
Kaggle — `push` checks that up front and refuses, rather than letting you discover it from
a kernel log ten minutes later.

Kaggle takes a little while to *process* a new dataset version, and `datasets version`
returns as soon as the upload lands, not when the processing finishes. A kernel pushed
into that window starts against a mount that is empty or not yet extracted and dies with
`no jax_rmhd checkout found under /kaggle/input` — minutes and a slice of the weekly GPU
quota later. So after every dataset push `launch.py` polls `kaggle datasets status
<user>/<slug>` every 10 s and pushes the kernel only once it reports ready, up to
`--dataset-timeout` (default 600 s). On timeout it exits *before* pushing the kernel and
tells you to re-run with `--skip-dataset` once the dataset page shows the new version.
The CLI's status wording is not a stable API, so an unrecognized answer is not fatal: after
more than three consecutive unrecognized replies and at least 60 s, the poll prints a
warning and continues to the kernel push rather than blocking on wording drift. Under
`--dry-run` the poll command is printed, not run. `--skip-dataset` skips the dataset push
and its poll entirely — still the right flag for iterating on kernel-side changes.

The **kernel** is generated from `_kernel_template.py`: `launch.py` replaces a single
sentinel line with `CONFIG = json.loads('...')` carrying the config (dataset slug,
repo-relative path of the run driver, entry name, entry kwargs, precision, session budget)
and pushes that plus a `kernel-metadata.json` requesting a GPU, internet, and the source
dataset. It goes back through `json.loads` rather than pasting the JSON in as a literal so
that `true`/`false`/`null` in `--entry-kwargs` stay valid Python with their types intact.

On Kaggle the kernel installs `orbax-checkpoint`/`tensorstore`, then a CUDA jaxlib, then
`pip install -e` on the repo it finds under `/kaggle/input` — located by searching for a
`pyproject.toml` containing `name = "jax-rmhd"` rather than by hardcoding a mount path,
because `--dir-mode zip` datasets do not mount at a predictable depth. Before searching it
prints a bounded listing of `/kaggle/input` (depth ≤ 4, ≤ 20 entries per directory, files
with sizes), so any future mount-layout surprise is diagnosable from the kernel log alone.
The search walks to depth 5; if it finds nothing, it falls back to archives — Kaggle
sometimes serves a `--dir-mode zip` dataset unextracted — unpacking any `*.zip`/`*.tar.gz`
under the mount that contains a `pyproject.toml` member into
`/kaggle/tmp/_repo_extract/<archive-stem>/` and searching again. Archives over 2 GiB are
skipped, and any archive with a member that would escape the destination (absolute path or
`..`) is refused whole rather than partially extracted. `pip install -e` on an extracted
copy is fine: `/kaggle/tmp` is writable. The dataset *mount* is not, though, and an editable
install has to write build metadata into the source tree — so before installing, the kernel
probes the discovered repo root with a real file write and, if that fails, copies the whole
tree (~22 MB) to `/kaggle/tmp/_repo_src` and uses the copy for both the install and the
driver import (a root already under `/kaggle/tmp` passes the probe and is never re-copied).
Every pip call goes through `sys.executable -m pip`, never a bare `pip`, so the packages
land in the interpreter that is actually running the kernel. Straight after the install the
kernel does `sys.path.insert(0, repo_root)` + `importlib.invalidate_caches()` and imports
`jax_rmhd` once to verify: a PEP 660 editable install only exposes the package through an
`__editable__*.pth` in site-packages, and .pth files are processed at interpreter *startup*,
so an install and an import in the same process do not see each other — the repo's flat
layout (the `jax_rmhd/` package sits at the repo root) makes the path insert sufficient,
while the editable install remains what pulls the dependencies and what makes the package
importable in any subprocess. `RMHD_PRECISION` is exported before that first import, since
`jax_rmhd` reads it at import time. It then imports the
run driver and dispatches: `make_data(snap_path, **kwargs)` in a loop until it returns
`True` (the `examples/gdi_*_run.py` contract — idempotent and resumable, so repeated calls
are how a long run gets made out of short ones), else `main(**kwargs)`, else plain
execution of the file as `__main__`. Afterwards it calls the module's `report(snap_path)`
if it has one, so the diagnostics land in the kernel log.

## Why `/kaggle/tmp` and not `/kaggle/working`

`/kaggle/working` is the persisted, downloadable directory, but it is capped at 20 GiB
**and** roughly 500 files. Orbax writes many small files per snapshot, so a run of any
length blows the file cap long before the size cap. The kernel therefore `chdir`s into
`/kaggle/tmp/<driver-stem>/` (large scratch, no file-count limit, not persisted), keeps
checkpoints there, and writes exactly one `<driver-stem>_output.zip` into
`/kaggle/working` at the end. That zip is written in a `finally` block, so a crash or a
budget-truncated run still produces downloadable partial progress. The honest limit of
that: an *external* kill — Kaggle's hard session timeout, or the OOM killer — never runs
the `finally`, and `/kaggle/tmp` does not survive the session, so everything computed
since the last zip is lost. That is what the budgets below are for.

## Quotas, budgets, and the timeout you will actually hit

Free GPU sessions run up to about 12 h each, against a weekly quota of roughly 30 GPU-hours
(Kaggle adjusts these; check your account page). `--session-budget` (default 39600 s = 11 h)
is the launcher's own guard: the kernel stops calling `make_data` once that much wall time
has passed and reports honestly that the run is partial rather than letting Kaggle kill the
session mid-write. Set the driver's own `wall_budget` kwarg to something well under it
(3300 s in the example above) so each individual `make_data` call ends cleanly at a
checkpoint. The session budget is only *checked between calls* — nothing can interrupt a
`make_data` call in flight — so the kernel prints a warning at startup if `wall_budget` is
missing or is more than half the session budget.

## Resuming

There is no cross-session resume. `/kaggle/tmp` is not persisted, so re-running the kernel
starts from scratch even though `make_data` itself is perfectly resumable. Kaggle's
`kernel_sources` field can chain a kernel's output into the next kernel's input, which
would let a long run continue across sessions; that is deliberately not implemented here —
it adds a stateful, hard-to-debug dependency between kernel versions, and for now the
honest workflow is to size `t_end` so the run fits one session, or to pull the output zip
and restart from those checkpoints yourself. If you need it, the hook is
`kernel_sources` in `stage_kernel`.

## Caveats

The P100 is Pascal (sm_60). Current `jax[cuda12]` wheels still support it, but if a future
jaxlib drops sm_60 the kernel will install fine and then fail to see a GPU — pin the
version in `_kernel_template.py::pip_install_stack`, or switch to `--machine
NvidiaTeslaT4` (Turing, sm_75). The P100's fp64 throughput is good, which is why the
default here is `--precision 64` even though the solver defaults to 32.

Single GPU means single process: `mpi4py`/`mpi4jax` are absent, so `comm_backend`
auto-resolves to `"serial"` (exact size-1 semantics — see the comms bullet list in
`CLAUDE.md`). Nothing in this launcher sets up a multi-rank run.
