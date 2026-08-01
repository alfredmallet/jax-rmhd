# Examples cleanup and development plan

Audit date: 2026-07-30. "Old API" means any of: `visc=`/`res=` kwargs instead of
`diss=(visc,res)`, `jr.Fields`, `jr.setup_sharding`/`shardings` args, positional
`SimulationState(...)`, `snapshot_manager_setup(snap_path=...)` without `params` as first
arg, or `load_snapshot(isnap, mngr, params, shardings)` instead of
`load_snapshot(isnap, snap_path, params)`. Current API reference: forced-turbulence-2D/3D
and orzag-tang-2D/3d notebooks.

## Phase 1: fix / delete existing examples

### Keep as-is (current API, run correctly)

| File | Notes |
|---|---|
| `orzag-tang-2D.ipynb` | Canonical pedagogical intro (has Q1–Q5 exercises). Keep. |
| `orzag-tang-3d.ipynb` | Canonical 3D decaying example. Keep. |
| `forced-turbulence-2D.ipynb` | Canonical forcing example incl. snapshot round-trip check. Keep. |
| `forced-turbulence-3D.ipynb` | Same, 3D. Keep. |
| `kaggle_forced_turbulence_256cubed.ipynb` | Current API; Kaggle GPU launcher. Keep. Minor: has a syntax error in the assert cell (`...accelerator setting".` — stray dot after the closing quote); fix. |

### Fix (old API, worth keeping)

1. **`AW_advection.ipynb`** — the only quantitative verification example (Alfvén-wave
   advection error vs `kz`, tests the 4th-order z scheme + `z_diss`). Port to current
   API: `diss=(visc,res)`, `jr.initialize(init_func, params)` instead of
   `jr.Fields`/positional `SimulationState`, `snapshot_manager_setup(params, ...)`, drop
   `setup_sharding`/`shardings`. The last cell is a dangling `plt.` — finish the error-vs-kz
   plot. Mind buffer donation: never reuse a state after passing it to `simulate`
   (the kz loop already rebuilds `start_state` each iteration, which is correct).
   `tests/test_advection.py` covers similar ground under MPI; the notebook stays as the
   single-process, plotted version.

2. **`load_savio_on_laptop.ipynb`** — useful workflow (post-process cluster snapshots
   locally) but wrong on two counts: old load API, and it constructs a
   `CheckpointManager` for reading, which violates the checkpointing invariant
   (docs/checkpointing.md — reads use a bare handler, no manager). Rewrite around
   `Parameters.from_snapshot(snap_path)`, `get_saved_steps(snap_path)`, and
   `sn.load_snapshot(isnap, snap_path, params)`. Delete the exploratory
   `help(ckptr)`/manual-orbax cells.

3. **`3d_image.ipynb`** — the three-face 3D volume rendering of vorticity is worth
   keeping. Fix the loading cells (same rewrite as above: no manager, current
   `load_snapshot` signature, params via `from_snapshot`). Consider merging 2 and 3 into
   one `postprocessing.ipynb` (load → 2D slices → spectra → 3D render) so there is a
   single loading example to keep current.

4. **`savio_ot3d.py`** — one-line fix (`snapshot_manager_setup(params, ...)`), but it
   duplicates `tests/forced_turbulence_64cubed.py`'s role. Either fix and move to
   `slurms/` next to its job scripts, or delete. Recommendation: delete;
   the slurm workflow already has a maintained driver.

### Delete

| File | Reason |
|---|---|
| `random.ipynb` | RNG histogram scratch; no jax_rmhd content. |
| `tearing.ipynb` | Empty (no code cells). Delete the file; the topic becomes a Phase 2 notebook. |
| `test.py` | Zero-field smoke script with a hardcoded Savio scratch path; superseded by `tests/`. |
| `test_lsrk.ipynb` | Old-API perf scratch for the lsrk scan/unroll option; superseded by `bench/bench_phase1.py`. |
| `scalings-2D.ipynb` | Old-API wall-time-vs-resolution benchmark; belongs in `bench/` if wanted at all (port only if you still use the plot). |
| `physics_engine_tests.ipynb` | Dev scratch (HLO dump, sharding visualization) on a mixed old/new API; superseded by `tests/`. If the HLO-inspection recipe is worth keeping, move that one cell into a docs snippet. |
| `examples/.DS_Store` | Committed by accident — `git rm --cached`, add `.DS_Store` to .gitignore. |
| `.ipynb_checkpoints/` | Already gitignored; delete locally. |

Housekeeping while in there: `examples/data/` is correctly gitignored but is 15 GB on
disk — worth pruning locally. Committed notebooks carry 200–500 KB of embedded outputs;
consider `nbstripout` (or keep one representative figure per notebook) to stop the repo
growing every re-run. Add a short `examples/README.md` listing each notebook, what it
demonstrates, and the suggested reading order; point README.md's line 13 at it (it
currently references `orzag-tang-3D.ipynb`, which doesn't match the actual filename
`orzag-tang-3d.ipynb`).

End state (9 files): orzag-tang-2D, orzag-tang-3d, forced-turbulence-2D,
forced-turbulence-3D, AW_advection, postprocessing (merged load_savio + 3d_image),
kaggle_forced_turbulence_256cubed, README.md.

## Phase 2: new example notebooks (all possible with current code)

Status: items 1–5 done (2026-07-30); all five notebooks executed end-to-end and committed
with outputs. Item 6 (decaying-turbulence-2D) deliberately skipped for now.

Roughly in order of value:

1. **`tearing-mode-2D.ipynb`** — the intended-but-empty tearing example. Initialize a
   Harris-like or `psi ∝ cos(x)` current sheet in 2D, watch island growth, measure the
   linear growth rate vs resistivity scaling (γ ∝ η^3/5 FKR / η^1/3 large-Δ′) and the
   plasmoid transition at high S. Uses only existing machinery (2D, `diss`, hyper=1 for
   true Laplacian resistivity). The classic RMHD validation problem, currently missing.

2. **`alfven-wave-collision-3D.ipynb`** — counter-propagating z⁺/z⁻ wave packets
   (Howes-style AW collision): initialize distinct phi±psi waves, show secondary-mode
   generation and perpendicular cascade onset via `diagnostics.perpspec`. Natural
   companion to AW_advection (linear) — this is the minimal nonlinear RMHD interaction.

3. **`turbulence-spectra-analysis.ipynb`** — load a developed forced-turbulence-3D
   snapshot and do real science on it: time-averaged perp spectra with −5/3 and −3/2
   references, z⁺/z⁻ (Elsasser) spectra, residual energy, `parspec` for the parallel
   spectrum (note: size==1 only), spectral anisotropy. The forced-turbulence notebooks
   run the simulation; nothing currently demonstrates the analysis.

4. **`forcing-modes-2D.ipynb`** — momentum vs elsasser forcing side by side, explicitly
   demonstrating the documented gotcha that 2D momentum forcing from a quiescent start is
   pure hydro (psi stays exactly 0), plus injection-power bookkeeping: measured dE/dt vs
   `forcing_power` vs dissipation, all on the shared `perp_*` normalization convention.
   Doubles as a physical explanation of the `forcing_mode` parameter.

5. **`restart-workflow.ipynb`** — the checkpoint lifecycle end to end:
   `params.save`, run, `Parameters.from_snapshot` with an override (e.g. changed `visc`),
   continue from `load_snapshot`, verify continuity. Also demonstrate the resharding
   restore path (docs/checkpointing.md) at notebook scale. This is the workflow every
   production user needs and no example shows.

6. **`decaying-turbulence-2D.ipynb`** (optional) — random large-scale IC, decay of
   energy/cross-helicity, selective decay and dynamic alignment. Cheap, physically rich,
   good student exercise in the orzag-tang-2D style.

Not yet possible without new code (candidates once GDI_PLAN lands): multi-equation-set
comparison notebooks; structure-function/higher-order statistics would need new
diagnostics.

### Suggested conventions going forward

`examples/` holds only runnable, current-API, single-process-friendly notebooks
(3D ones at laptop resolutions, with a comment showing the mpirun variant). Performance
studies live in `bench/`, cluster drivers in `slurms/`/`tests/savio_scaling/`, and
correctness checks in `tests/`. When an API change lands, grep `examples/` for the old
signature as part of the change.
