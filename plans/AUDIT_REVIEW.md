# Audit review — plans/AUDIT_PLAN.md Items 1–8 (commits c5ffb0e..eaf0747)

Adversarial review, 2026-08-17, by a fresh session with no implementation context.
Reviewed: the 8 commits on top of 93d2173, each against its plan item; every gate re-run
here rather than taken from the commit messages. Machine: laptop, jax 0.10.0, Python
3.11.5, mpi4py + mpi4jax importable (so `comm_backend="mpi4jax"`, size 1).

## Verdict

**Accept.** All eight items are implemented as specified, every gate holds when re-run,
and the bit-identity claims in Items 1, 2 (and, on its recorded grids, 4) reproduce
exactly. Two things need follow-up outside the diff (F1, F2 below); the rest are minor
notes, none blocking. Commit order (1, 2, 8, 4, 3, 5, 6, 7) respects the plan's
ordering constraints (1→2→3 strictly, 7 last, one commit per item).

## Gates re-run

| Gate | Result |
|---|---|
| `make test` at HEAD (eaf0747) | green: fp64 144 passed / 22 skipped / 1 deselected; fp32 140 passed / 26 skipped / 1 deselected. Skips are the expected fp64-only / multidev ones; nothing new skips. |
| suite at 2f10716 (Item 4, mid-sequence spot check) | green: fp64 143 passed / 22 skipped; fp32 139 passed / 26 skipped (one fewer than HEAD in each session: Item 6's `test_stale_env_var_name_is_rejected` lands later) |
| Item 1: `kgrid.lin_L/lin_m/lin_s2` bit-identical | **confirmed** — `np.array_equal` + same dtype (complex128 / complex64) for an RMHD z_spectral 3D setup and the gdi_3d_run GDI 3D setup, base 93d2173 vs c5ffb0e, and also vs bfc996f and vs HEAD, in both precision sessions |
| Item 1: `_max_re_lambda`, `kperp_break`, `unstable_modes_report`, `theory_cross_phase` unchanged | **confirmed** identical floats at all four trees, both sessions (2D 0.028469431285782356, 3D 0.27858998347001823; kperp_break 4.7320880975127215 / 4.085478439271451 / 3.9632382685542114; theory_cross_phase rows and the unstable_modes_report dict identical) |
| Item 2: `_max_re_lambda` bit-identical after the extraction | **confirmed** (same probe, c5ffb0e vs bfc996f vs HEAD) |
| Item 2: new pin test present, exercised in both sessions | yes (`test_np_grid_rebuild_matches_grids_module`, 3 grids incl. 64x48x20 and two anisotropic boxes; regions exact, k values ≤ 4 ulp) |
| Item 3: old import paths | `diagnostics.energy/perpspec/parspec/_binned` importable and `is`-identical to `diagnostics.rmhd.*` / `diagnostics.core._binned` |
| Item 3: pure movement | verified function-by-function with an AST diff: `energy/perpspec/parspec/_binned` and `energy_enstrophy/energy_budget/kperp_break/measure_alpha` byte-identical; `perp_spectrum`/`cross_phase_spectrum` differ only by the deleted local `from .. import diagnostics` + `_binned` call; `theory_cross_phase` differs by `gdi._eqpars`→`_eqpars` and a trimmed comment (P4b-report references dropped — consistent with the plan's no-history-in-source rule) |
| Item 3: physics no longer imports diagnostics | confirmed (`physics/gdi.py` has only a pointer comment) |
| Item 3: gdi-2D notebook runs | ran end-to-end via `jupyter nbconvert --execute` against the on-disk `examples/data/gdi-2D` (must run at fp32 — the recorded params.json is fp32; at fp64 `params.save` correctly refuses) |
| Item 4: sanity test | `tests/test_quiescent_dt.py` present, both sessions, exact `==` in 2D / 2D-anisotropic / 3D z_spectral, `<=` in 3D FD-z (zero-gradient and developed) |
| Item 4: reference regen decision | see below — correct and correctly scoped |
| Item 5: `grep -rn LinearTerm` | only `plans/` and `FDLinearTerm` (14 files carry the new name) |
| Item 6: `RMHD_PRECISION` outside `plans/` | only `_precision.py` (the guard), `test_precision_dtypes.py` (the guard's test), CLAUDE.md (the rename note), and one `plans-webgpu/` mention (historical, same class as `plans/`) |
| Item 6: guard | old var alone → `RuntimeError` naming the new var; both set → TARANIS wins (32); new only → honored; bad value → `ValueError`. All exercised directly. |
| Item 7: name-set grep | only citation-class / ported-code-provenance / self-attribution hits (details below) |
| Item 8: `coef` gone | no `coef` in propagators/timestepping/tests/docs; the only 3-arg `apply_exp` call (rk44 stage 3) is now `dt*apply_exp(k3, dt/2)`; `IdentityPropagator.apply_exp` is `return arr` |

## Item 4 — reference regeneration: deliberate and correctly scoped

The plan expected Item 4 to regenerate "the recorded fp32 forcing-spinup reference". The
commit did **not** regenerate anything and says why. Checked, and the commit is right;
the plan's premise was wrong:

- `tests/data/forcing_spinup_reference.npz` is **fp64** (its generator asserts
  `TARANIS_PRECISION == "64"`; `test_forcing_spinup` is `@pytest.mark.fp64`). Its three
  configs are quiescent starts on 2π/64 and 2π/32 grids. Computed here:
  `cfl/max(eps/dx, eps/dy)` and `cfl·min(dx,dy)/eps` are bitwise equal at fp64 on both
  grids (0.4908738521234052, 0.9817477042468103), and `e3dfd` is 1/dz-bound (0.098) so
  the floor is inert. `test_forcing_spinup`'s exact step-1 dt check passes at HEAD.
- `tests/data/precision_fp32_reference.npz` runs at `ctx()` defaults —
  `adaptive_timestep=False, dt=0.01` — so `set_timestep` is never on its path.
- `webgpu/refvectors*.json` (fp64, adaptive, developed state): the 2D reference's
  `A_dt = 0.0242` vs floor 0.785 — velocity-bound, so identical (and the browser
  compares at fp32 tolerance anyway).
- Empirical before/after `set_timestep` on 7 grids × {zero, developed} × {fp64, fp32}
  (base 93d2173 vs HEAD): every developed-state dt bitwise identical; quiescent-floor dts
  identical except **one 1-ulp shift at fp64** (nx=100, L=1: 0.05 → 0.049999999999999996;
  fp32 identical on that grid). So the plan's "1-ulp differences at fp32" understates it —
  the shift is grid-dependent at both precisions. Nothing recorded is affected, but see
  N4.

## Item 7 — name set used

Grep set: Squire / Jono, "wife|spouse|husband|partner", the scholarly names Mallet,
Loureiro, Cowley, Dorland, Haines, Schekochihin, Howes, Verniero, Klein, plus the credit
patterns "suggested by|thanks to|feedback from|reported by|pointed out by|courtesy of|
per X's|via First Last", over every tracked file type (`.md .py .ipynb .js .html .sh
.toml .txt`, Makefile). Remaining hits, all KEEP-class:

- ported-code provenance (plan's explicit KEEP): `webgpu/README.md:966`,
  `plans-webgpu/FEEDBACK_2026-08-08.md:190-191`, **and `webgpu/common.js:5436`** — the
  last one is not in the plan's inventory but is the same auto_dissipation credit in
  the code that was ported; keeping it is exactly the plan's rule.
- citations: `webgpu/reading.html` (Squire et al. papers), Loureiro 2005 / Numata-Loureiro
  in `webgpu/README.md`, `common.js`, `rmhd2d.html`, `plans-webgpu/{README,REFINE,TEARNL,
  CHI,ISO}_PLAN.md`; Mallet et al. 2015 in `common.js`, `devtools/checkaniso.js`,
  CHI_PLAN; Howes & Nielson / Nielson, Howes & Dorland in `alfven-wave-collision-3D.ipynb`.
- self-attribution: the "Alfred Mallet" byline in `rmhd2d.html`/`rmhd3d.html`,
  "Alfred's phone" in LOOPLAT_PLAN's table.

The scrubbed lines (`ANISO_PLAN_2.md`, `LOOPLAT_PLAN.md`, `ONEPAGE_PLAN.md`,
`plans-webgpu/README.md`) read naturally after the edit.

## Findings

### F1 (follow-up required, outside the diff) — Item 6 breaks the lugus Kaggle launcher

`/Users/alfy/code/lugus/_kernel_template.py:420` sets `os.environ["RMHD_PRECISION"]`
before importing taranis (`set_precision`; `check_gpu` re-asserts it at :425;
`launch.py:401,480` and `README.md:55,121` document it). lugus uploads the working-tree
taranis source as the kernel's dataset, so the next launch against current main will die
at `import taranis` with the new `RuntimeError`. That is the guard doing its job (loud,
not silent), and the plan explicitly decided against a fallback — but lugus is Alfred's
own downstream and neither the plan nor the commit mentions it. Update lugus (env var name
in `set_precision`/`check_gpu`, the CLI help/print, README) before the next Kaggle run.

### F2 (follow-up, outside the repo) — the HOME-directory `~/CLAUDE.md` is stale

Not the repo's `taranis/CLAUDE.md` (that one is current — Items 3/5/6 updated it). The
separate `/Users/alfy/CLAUDE.md` (8.3 KB, last modified 2026-07-17, an old snapshot of the
taranis CLAUDE.md) is also loaded into every session and still says `RMHD_PRECISION=64
python your_script.py` (line 26), "Currently only RMHD is implemented" (line 9),
`SimulationState` as a 4-tuple (line 53), and `diagnostics.py` as a module (line 114).
After Item 6 its precision line is actively wrong — a session that reads it first could
set the old var and hit the import error. Delete it, or reduce it to a pointer at
`~/code/CLAUDE.md`.

### N1 (minor, Item 1) — `propagators -> physics` import inversion

`putzer2_precompute` does a function-local `from .physics.shared_physics import eig2_ms`
because a module-level import would cycle (`physics/__init__` → gdi → grids →
propagators). The plan put `eig2_ms` in `shared_physics`, and the implementer did as
told, but the local import is the symptom of the helper living one layer too high:
`eig2_ms` is pure algebra with no physics in it. A cleaner home is `propagators.py`
itself (or a tiny `_linalg.py`) with `shared_physics` re-exporting for the numpy/scalar
sites. Cosmetic; no behaviour at stake (verified bit-identical).

### N2 (minor, Item 2) — k values pinned to ≤4 ulp, not equality

The plan wrote "the kz/kx/ky values equal `np.asarray(kgrid.kz/kx/ky)`"; the test uses a
4-ulp bound because grids evaluates `((f·n)·2)·π/L` and gdi `(f·n)·(2π)/L`. The
deviation is explained in the test and is the right call (making them bit-equal would
mean changing one implementation's op order, which is out of scope for a pin test). The
part the plan actually cared about — the dealias REGIONS, including the fp32 knife-edge —
is compared exactly.

### N3 (minor, Item 4) — `estimate_good_nblock` behaviour change on the fixed-dt path

`run.estimate_good_nblock` calls `set_timestep` regardless of `adaptive_timestep`. With
`adaptive_timestep=False`, `_quiescent_dt` is `params.dt`, so it now returns
`min(cfl_dt, params.dt)` instead of `cfl_dt`. That is arguably more correct (the fixed
path steps at `params.dt`) and CLAUDE.md's new sentence covers it generically ("a direct
`set_timestep` call there also caps at `params.dt`"), but it is a small user-visible
change the commit message frames as test-only. Recording it here so it is not
rediscovered as a surprise.

### N4 (minor, Item 4) — the accepted 1-ulp cost is not recorded in the docs

The plan's "Cost (DECIDED)" section names it, but `docs/numerics.md` / CLAUDE.md gained
no dated note that quiescent-floor-bound dts can shift by 1 ulp vs pre-2026-08-17 (and,
per the measurement above, at fp64 as well as fp32). Anyone regenerating or bisecting a
quiescent-start reference across this commit should be able to find that in the docs,
not just in the plan file. One sentence next to the new "`_quiescent_dt` IS the floor"
paragraph would do.

### N5 (nit, Item 4) — no test that the velocity term can still bind

`test_quiescent_dt.py` proves `dt == dt_q` at zero gradient and `dt <= dt_q` otherwise;
nothing asserts `dt < dt_q` for a developed 2D state (i.e. that the `min` didn't collapse
to a constant). Other suites cover it implicitly (test_forcing_spinup's later-step dts,
the FD-z developed case here is 1/dz-bound not velocity-bound), and my probe shows
developed 2D dts of 8e-3 vs ceiling 0.49 — but a one-line check would close it.

### N6 (nit, Item 6) — `plans-webgpu/REFINE_PLAN.md:189` still says `RMHD_PRECISION`

The plan says leave `plans/`; the commit extended that to `plans-webgpu/` (stated in
its message). Consistent, just noting the one hit for the record.

## Process notes

- Commit messages match what the diffs do; the Item 1 and 2 messages' bit-identity
  claims and the Item 4 message's no-regen reasoning all reproduce here.
- No review/design history was added to source comments (checked the full 93d2173..HEAD
  diff of `taranis/`, `tests/`, `examples/*.py`); the only dated string is the Item 6
  error message, which is user-facing and appropriate.
- The before/after probes ran in throwaway `git worktree`s at 93d2173, c5ffb0e,
  bfc996f and 2f10716 (removed after the review); the main checkout was never modified.
