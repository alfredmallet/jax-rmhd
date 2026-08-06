# Kaggle general-code plan — running arbitrary simulation codes via kaggle_launcher

Goal: generalize `kaggle_launcher/` from "jax_rmhd run drivers" to "any simulation code
on a single Kaggle GPU" — first targets a FORTRAN+OpenACC gyrokinetics code (architecture
not yet known) and GX (CUDA C++/CMake gyrokinetics), with all analysis in Python on the
kernel so only reduced data comes back. Written 2026-08-06. **Plan only — no
implementation yet.** Companion to `plans/KAGGLE_CLASSROOM_PLAN.md`; the two share the
P1 launcher surgery (see G1).

## Design background (why this shape)

- **The launcher is already ~80% code-agnostic.** Everything in `launch.py` (staging via
  `git ls-files`, dataset version + readiness poll, kernel metadata/push/poll, output
  pull, the staging-sentinel and slug machinery) never looks inside the repo. The
  jax_rmhd coupling is confined to `_kernel_template.py` in five places:
  `pip_install_stack()` (hardcoded jax/orbax recipe), `REPO_MARKER` (`name = "jax-rmhd"`
  in pyproject.toml), `install_repo`/`enable_in_process_import` (Python-package
  assumptions), `set_precision` (`RMHD_PRECISION`), and the make_data/report driver
  contract — plus one `launch.py` nit (`--precision` and the default dataset slug).
  Generalization = make those pluggable; everything else (writable-copy of read-only
  mounts, archive-extraction fallback, bounded mount listing, finally-zip) is exactly
  what a compiled code needs too and must not fork.
- **Compile-once, run-many via a binary-cache dataset.** A multi-GB toolchain install
  (NVIDIA HPC SDK for nvfortran/OpenACC) is tolerable once but not per session. So two
  kernel roles: a **build kernel** compiles and emits `<project>_build.zip` (binary +
  exact runtime .so set + build log + provenance manifest) into `/kaggle/working`; its
  output is then republished as a dataset (stable, shareable, classroom-compatible) or
  mounted directly into later kernels via `kernel_sources` (zero re-upload, but pinned
  to one kernel version). **Production kernels** mount the binary dataset, verify it
  runs (`ldd` + a --version/dry invocation), and skip the toolchain entirely — startup
  cost returns to ~1 min.
- **Analysis stays Python, on the kernel.** Kaggle's persisted output is capped (20 GiB,
  ~500 files) and network-pulled; gyrokinetics raw output (3D+2V fields, NetCDF/HDF5)
  can be tens of GB. The existing pattern — run in `/kaggle/tmp`, drop one zip in
  `/kaggle/working` — generalizes: the project's own analysis routines run in-kernel
  (numpy/xarray/netCDF4 are pip-cheap) and the zip carries reduced data + figures +
  whatever raw subset the spec asks for.
- **GPU target economics.** Kaggle offers P100 (sm_60, 16 GB, ~4.7 TF fp64) and T4×2
  (sm_75, 16 GB each, ~0.25 TF fp64 per card) on a shared ~30 h/week quota, 12 h/session.
  An fp64-dominated GK code wants the P100 (~18× fp64 throughput); but Pascal is removed
  from CUDA 13, so P100 builds must pin a toolchain bundling CUDA 12.x. Build for both
  where possible (`-gpu=cc60,cc75` fat binary under nvfortran; `CMAKE_CUDA_ARCHITECTURES
  "60;75"` for GX) so the binary dataset serves either machine_shape.
- **Foreign repos don't live in this repo.** The friend's code and GX arrive either as a
  local checkout (stage it: `--repo-root` pointing anywhere + git-optional staging) or as
  a dataset someone already published (`--dataset OWNER/SLUG`). Both flags are already
  specified in classroom-plan P1 — implement once, serve both plans.

## Binding constraints (carry into every phase)

- **jax_rmhd path unchanged.** The current CLI with no new flags must behave bitwise-
  identically (staging layout, metadata, template output), and the existing 32 tests in
  `kaggle_launcher/test_launcher.py` stay green unmodified. jax_rmhd becomes the built-in
  default project spec, not a special case in the code.
- Sandbox has no kaggle token/network: everything verified by `--dry-run` + the hermetic
  stub-kaggle/fake-tree test fixtures; every real push is a user step, listed per phase.
- Kernel-side invariants survive for all projects: work in `/kaggle/tmp/<stem>`,
  finally-zip to `/kaggle/working` (one file vs the ~500-file cap), bounded mount
  listing before any repo search, writable-copy of read-only mounts, wall-budget
  warning. `stdlib`-only above the setup phase.
- Never burn GPU quota on a preventable failure: every new failure mode gets a pre-push
  local check (spec-file validation, referenced-script existence) or a fast fail at
  kernel top (binary/loader check before any long step) — same spirit as
  `require_tracked` and `wait_dataset_ready`.
- Secrets/licensing: no third-party code is republished as a Kaggle dataset without an
  explicit license check (NVHPC runtime redistribution is EULA-permitted for bundled
  redistributable libs — verify the current list at implementation time; the friend's
  code and GX each need the owner's OK before any public dataset).

## G1 — project-spec refactor (the core surgery)

A project declares itself in a JSON spec file (`launch.py push --spec gk.json ...`;
stdlib-only parsing, validated before any upload). Absent `--spec`, the implied jax_rmhd
spec reproduces today's behavior exactly. Fields (all optional except `name`):

- `name`, `repo_marker`: marker file + content substring for `find_repo_root`
  (replaces the hardcoded pyproject/`jax-rmhd` pair; e.g. `{"file": "Makefile",
  "contains": "gx"}`).
- `setup`: list of shell commands run before anything else (replaces
  `pip_install_stack`; apt/pip both legal — kernels run as root with internet).
  jax_rmhd default: the current orbax→jax[cuda12] recipe.
- `python_install`: bool (default true only for the jax_rmhd spec) — whether to
  `pip install -e` the repo + do the in-process-import dance. Compiled projects: false.
- `build`: shell command list producing artifacts under a declared `build_dir`
  (build kernels only, see G2). Empty for jax_rmhd.
- `run`: either `{"driver": "path.py"}` (today's make_data/main/__main__ contract,
  unchanged) or `{"command": ["./gx", "input.in"], "inputs": [...]}` — subprocess with
  cwd=workdir, env passthrough + spec-declared additions, streamed to the log.
  v1 resumability contract for commands: the code checkpoints itself and exits within
  `wall_budget`; the finally-zip ships whatever exists. (Cross-session restart chaining
  via `kernel_sources` mount of the previous output = v2, not needed at first-target
  scale.)
- `analyze`: optional Python script path, called as `analyze(workdir)` after the run,
  wrapped exactly like `maybe_report` (never kills the zip). Its imports come from
  `setup`.
- `env`: dict of environment variables set before setup (generalizes `RMHD_PRECISION`;
  the `--precision` flag becomes sugar that writes into it for the jax_rmhd spec).
- `datasets`: extra `dataset_sources` ids to mount (binary cache, reference inputs).

Implementation notes: the template keeps ONE code path — the jax_rmhd flow is the spec
defaults, not an if/else fork. `CONFIG` grows a `spec` sub-dict via the existing
sentinel substitution (still `json.loads`-wrapped). Shared with classroom P1:
`--repo-root PATH` (stage a non-git tree or skip git checks) and `--dataset OWNER/SLug`
(reference a remote dataset, implies `--skip-dataset`) — one implementation, both
consumers. `stage_dataset` gains a non-git mode (copytree with an excludes list from the
spec) since foreign checkouts may not be git repos or may have huge untracked run dirs.

## G2 — build-cache workflow

- `launch.py build --spec gk.json`: pushes a kernel whose run phase is the spec's
  `build` commands; artifacts + `ldd`-derived runtime .so closure + `MANIFEST.json`
  (toolchain versions, GPU arch flags, source dataset version, git describe if
  available) are zipped to `/kaggle/working` by the existing finally-zip.
- `launch.py publish-build --spec gk.json`: pulls that output and republishes it as
  `<user>/<name>-build` dataset (reuses `stage_dataset`/version/create/readiness-poll
  machinery on a different payload). Alternative zero-upload path: `--use-kernel-output
  USER/KERNEL` writes the build kernel into `kernel_sources` instead.
- Production kernels with a `binary` dataset in `spec.datasets`: template locates the
  build (same marker-search machinery, marker = `MANIFEST.json` with matching `name`),
  copies to scratch if read-only (existing `ensure_writable_repo` path, renamed),
  prepends its `lib/` to `LD_LIBRARY_PATH`, and fast-fails on a loader/`--version`
  probe before touching inputs.
- Toolchain choices recorded in the spec, not the launcher: NVHPC via apt (build
  kernels only; pin the last CUDA-12.x-bundling release for P100/sm_60),
  `gcc-*-offload-nvptx` as the lightweight fallback (expect 2–5× slower OpenACC —
  acceptable for smoke tests, not production), plain CUDA toolkit + CMake for GX
  (already on Kaggle images; check `nvcc` version vs sm_60 at build time).

## G3 — pilot targets (in order)

1. **Self-test pilot: a trivial OpenACC saxpy/heat-equation mini-app** kept under
   `kaggle_launcher/pilots/` — proves spec + build-cache end-to-end without any
   third-party code, and becomes a permanent regression fixture (its spec exercises
   every G1 field).
2. **GX**: public repo, known CMake+CUDA build, NetCDF output, single-GPU native —
   the realistic pilot. Deliverables: `pilots/gx.json` spec + a small analyze script
   (growth rates/fluxes from the NetCDF, matching a published linear benchmark case,
   e.g. a cyclone-base-case linear ITG scan — cheap and checkable). Needs: apt netcdf
   + hdf5 in `setup`, `-DCMAKE_CUDA_ARCHITECTURES=60;75`.
3. **Friend's FORTRAN+OpenACC code**: blocked on its actual architecture (build system,
   input format, output format, checkpoint/restart story, MPI-required-or-optional,
   fp64 fraction → P100-vs-T4 choice). The spec format from G1 + lessons from GX should
   make this a spec-file exercise, not launcher surgery; if it isn't, that's a G1
   design bug to fix then.

## G4 — verification

- Sandbox (extend `test_launcher.py`, same hermetic style — stub kaggle, fake trees,
  no network/pip/GPU): spec parsing/validation + all pre-push checks; back-compat
  (spec-less run produces byte-identical staging vs a golden copy of today's output);
  template phases against a fake binary-dataset tree incl. read-only copy + loader
  fast-fail; command-runner subprocess capture; build/publish-build staging.
- User (real Kaggle, in order): (1) pilot mini-app `build` → `publish-build` → `run`
  on P100 AND T4 (the fat-binary check); (2) GX build + linear benchmark run, compare
  against published values; (3) friend's code once its architecture is known. Each
  step's quota cost is minutes except the NVHPC build session (~20–40 min).

## Execution structure

Per project convention (opus/sonnet implement, fresh-Fable adversarial review):
A1: G1 spec refactor + back-compat tests → A2: G2 build-cache (blocked by A1) →
A3: pilot mini-app + GX spec (blocked by A2; GX analyze script is opus-suitable) →
A4: independent review of A1–A3 against this plan + CLAUDE.md → user runs the
real-Kaggle ladder in G4. Classroom-plan A1 (its P1) should be merged into this A1 —
same flags, one review.

## Open questions (none block A1)

1. Friend's code architecture — build system, I/O formats, restart story, MPI
   dependency, precision mix. Blocks only G3.3.
2. Where the generalized launcher ultimately lives: it stops being jax_rmhd-specific
   after G1, so a standalone repo/pip package is natural — but splitting now costs the
   test harness conventions; propose deferring until after G3.2 proves the design.
3. `kernel_sources` vs republished dataset as the default binary-cache transport
   (dataset proposed above for stability + classroom sharing; revisit if the
   pull→republish round trip is annoying in practice).
4. NVHPC redistributable-runtime list at implementation time (licensing constraint
   above); affects whether the build zip can be public.
5. T4×2: expose the second GPU to codes that can use one node's two devices, or
   ignore it in v1 (proposed: ignore).
