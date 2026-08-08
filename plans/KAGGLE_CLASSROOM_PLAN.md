# Kaggle classroom plan — CPU teaching notebook driving P100 runs via kaggle_launcher

Goal: let students explore RMHD/GDI turbulence on Kaggle using **their own accounts**,
with all analysis/visualization in a **CPU notebook session** (no GPU quota) and the
actual simulation in a **P100 script kernel** pushed/polled/pulled by `kaggle_launcher/`.
Written 2026-08-03, after the launcher itself was built and dry-run-verified (see
`kaggle_launcher/README.md`). **Plan only — no implementation yet.**

## Design background (why this shape)

- **No third-party auth.** Kaggle has no OAuth for external apps; the only programmatic
  credential is the all-or-nothing personal API token. Any architecture where students
  paste tokens into someone else's site is disqualified. Instead, everything runs inside
  Kaggle: students store their own token in the **Kaggle Secrets add-on** of their own
  notebook, and the notebook drives the `kaggle` CLI under their identity. No external
  service ever exists.
- **Quota split.** GPU quota (~30 h/week) is consumed only by the P100 script kernel;
  CPU notebook sessions are unmetered against it. Analysis, linear theory, and plotting
  are all cheap on CPU (taranis runs fine on CPU jax; snapshots load via orbax the
  same way). The notebook is "mission control": launch, then do productive theory work
  while polling, then pull and analyze.
- **Batch, not live.** A run is minutes-to-hours of wall clock; the notebook must be
  structured so the wait is pedagogically useful (linear-theory section between launch
  and pull), and analysis must also work with **no run at all** via a published
  precomputed-output dataset — students who hit quota limits or queue delays still get
  the full analysis experience.
- **Students have no git checkout.** The repo reaches them as a read-only public Kaggle
  dataset mount. The launcher currently assumes a local git checkout (stages
  `git ls-files`, validates the driver is tracked, names `dataset_sources` under the
  pusher's own username). Teaching mode inverts all three: reference the
  **instructor's** public dataset by full id, stage nothing, and validate the driver
  against the mount. This is P1 and is the only launcher surgery required.

## Binding constraints (carry into every phase)

- No student-credential handling outside Kaggle Secrets. The notebook writes
  `~/.kaggle/kaggle.json` from the secret at session start, chmod 600, and never prints it.
- Current launcher CLI behavior stays the default; teaching mode is opt-in flags. All
  existing dry-run behavior must keep working unchanged (the local-checkout workflow is
  still the instructor's production path).
- Kernel-side invariants from the existing template are untouched: `RMHD_PRECISION` set
  before any taranis import; run in `/kaggle/tmp/<stem>`; finally-zip to
  `/kaggle/working`; wall_budget-vs-session_budget warning.
- Sandbox has no kaggle token or network to kaggle.com: implementation phases are
  verified by `--dry-run` + mocked-CLI tests only; every real push is a user step and
  each phase below lists its user-verification explicitly.
- CLAUDE.md rules are binding in any notebook code (never cache a donated state, params
  identity-hash, `state._replace`, etc.).

## P1 — launcher remote-dataset ("classroom") mode

`launch.py` additions, default-off:

- `--dataset OWNER/SLUG`: use this exact id in `dataset_sources` instead of
  `<username>/<dataset-slug>`; implies `--skip-dataset` (you can't version someone
  else's dataset). Mutually exclusive with `--dataset-slug`.
- `--repo-root PATH`: where to resolve the run driver when there is no git checkout
  (e.g. `/kaggle/input/taranis-src/...` mount). With `--repo-root`: skip
  `git rev-parse`/`git ls-files` entirely, replace the git-tracked check with
  "driver exists under repo-root" (existence in the mount IS shipping — the mount is
  the dataset), and compute `run_relpath` relative to it. Without it: current behavior
  exactly, including the untracked-driver hard error.
- Kernel staging writes to a writable dir even when cwd is read-only (notebook cwd may
  be `/kaggle/working`): default `--staging-dir` must fall back to a tmp dir when
  `kaggle_launcher/.staging` isn't writable.
- Failure-mode guard: `--dataset` + a driver relpath that doesn't exist in the local
  `--repo-root` mount is a pre-push hard error (same spirit as the untracked check —
  never burn a GPU session on a path typo).

## P2 — Python API for notebook use (`kaggle_launcher/api.py`)

Thin wrapper over the CLI so notebook cells aren't `!`-shell soup; stdlib +
already-required deps only.

- `setup_credentials()`: read `KAGGLE_USERNAME`/`KAGGLE_KEY` via `kaggle_secrets`
  (guarded import — not present off-Kaggle), write `~/.kaggle/kaggle.json` (0600).
- `launch(run_py, dataset="OWNER/SLUG", repo_root=..., entry_kwargs={}, machine=...,
  precision=...) -> handle`: builds the same staging as `cmd_push` (shared code, not a
  subprocess of launch.py) and pushes; returns kernel id + urls.
- `status(handle)`, `wait(handle, poll_s=60, timeout_s=...)` (prints elapsed, returns
  final state), `fetch(handle, out_dir) -> path`: pull + unzip the `<stem>_output.zip`.
- `open_run(path)`: convenience for analysis — `Parameters.from_snapshot`,
  `setup_kgrids`, `get_saved_steps`, returning what the analysis cells need; must work
  identically on a fetched zip and on the precomputed dataset mount (that equivalence
  is what makes the no-GPU fallback seamless).
- Refactor note: whatever `cmd_push` logic `launch()` reuses moves into shared
  functions; `launch.py` stays the single source of truth for metadata generation
  (no copy-pasted kernel-metadata dicts in api.py).

## P3 — instructor-published assets (user steps, launcher-assisted)

- Make the source dataset **public** (currently the plan default is private):
  one-time `kaggle datasets` visibility change, plus a license decision for the repo
  dataset page (repo LICENSE governs; dataset metadata license field should match, not
  the placeholder CC0).
- Run and publish a **precomputed reference output dataset** (public): the teaching
  run at full length, produced by the existing launcher, output zip re-uploaded as its
  own dataset (`OWNER/taranis-classroom-reference`). Include in it a small
  `MANIFEST.json` (params, t_end, snapshot cadence, launcher/commit provenance) so the
  notebook can display where its fallback data came from.
- Record the chosen dataset ids in one place the notebook imports (a constants cell or
  tiny `classroom_config.py` in the repo), so a future re-publish is a one-line change.

## P4 — teaching notebook (`examples/kaggle_classroom.ipynb`)

Target run for v1: **2D forced turbulence (elsasser)** at ~256², minutes on the P100 —
short feedback loop; per CLAUDE.md, 2D momentum-mode from quiescent is pure hydro, so
elsasser is required for actual MHD. A 3D variant is an explicit exercise, not the
default path (cost jumps and the z-decomposition doesn't apply on one GPU anyway).
Structure:

1. **Setup** (CPU, internet on): pip installs (CPU jax — never `jax[cuda12]` here),
   repo from the public dataset mount, `api.setup_credentials()` from Secrets, with a
   markdown walkthrough of creating the token + secret (the one genuinely fiddly step).
2. **Physics primer**: RMHD equations, what forcing/dissipation/spectra mean, with the
   run's actual parameters printed and explained.
3. **Launch**: `api.launch("examples/<teaching_driver>.py", dataset=..., repo_root=...)`
   — the driver itself is a normal repo make_data-style module (P4 includes writing it
   if no existing driver fits the teaching parameters exactly).
4. **Theory interlude while it runs**: linear-theory exploration on CPU (dispersion,
   forcing-shell geometry, expected spectral slopes; for a GDI variant,
   `unstable_modes_report`/`theory_cross_phase` style probes) + `api.status()` checks.
5. **Fetch & analyze**: `api.fetch()` or fallback to the reference dataset via the same
   `open_run()`; energy vs t, perp spectra vs time, field-snapshot images/animation,
   with "what to look for" prose (saturation, inertial range, dissipation cutoff).
6. **Exercises**: parameter variations by editing `entry_kwargs` (forcing power, diss),
   each with expected-outcome discussion; the 3D stretch exercise; honest quota
   arithmetic (what a week of experiments costs).

## P5 — verification

- Sandbox: dry-run matrix over new flag combinations (`--dataset`+`--repo-root` against
  a fake mount tree, writable-staging fallback, typo'd driver rejection); api.py unit
  tests with a mocked `kaggle` binary (a stub script on PATH recording argv) covering
  launch/status/wait/fetch and the zip→`open_run` path on a synthetic output zip;
  notebook executed top-to-bottom on CPU with the mocked binary + a local mini reference
  run standing in for the datasets.
- User (real Kaggle, in order): publish datasets (P3), one end-to-end notebook session
  from a *fresh student-like account* if possible — the token/Secrets onboarding is the
  step most likely to lose students and only a real account exercises it; confirm the
  P100 run completes and `fetch` returns; confirm the no-GPU fallback path by running
  the analysis half only.

## Execution structure (when implementation starts)

A1: P1 (launch.py surgery + dry-run tests) → A2: P2 (api.py + mocked-CLI tests;
blocked by A1's shared-function refactor) → A3: teaching driver + notebook (P4;
blocked by A2) → A4: independent review of A1–A3 against this plan + CLAUDE.md →
user executes P3 and the real-Kaggle half of P5. Per project convention, implementation
goes to sonnet/opus agents with this plan as the spec.

## Open questions (decide before A3, none block A1/A2)

1. Teaching run family/resolution: 2D forced elsasser ~256² proposed above — confirm,
   and pick t_end/snapshot cadence sized for a ~10–20 min P100 run.
2. Audience level (grad plasma course vs general "explore turbulence" outreach) —
   changes primer depth and exercise design, not infrastructure.
3. Whether the notebook itself should also be published as a public Kaggle notebook
   (students "Copy & Edit" instead of uploading the .ipynb) — probably yes; costs
   nothing in code, one more P3 publish step.
4. Repo-dataset license choice (P3).
