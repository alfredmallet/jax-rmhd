# plans

Planning documents, progress reports and code reviews — the record of *how* the code got
here. Not documentation: for that see `docs/` (checkpointing, running the tests, Savio
setup) and `CLAUDE.md`.

`plans/` holds work that is still live. `plans/old/` holds work that is finished.

**The durable content of `plans/old/` has been extracted** (2026-07-31) into
`docs/numerics.md` (derivations and conventions), `docs/performance.md` (cluster
measurements, tuning knobs, negative results) and the CI section of
`docs/RUNNING_TESTS.md`. What remains in `plans/old/` is process — task breakdowns,
agent handoffs, status markers — kept for provenance but no longer load-bearing. Nothing
outside `plans/` links into it — with one exception, `docs/performance.md`'s pointer at
`old/TARANIS_MEMORY_HANDOFF.md` for the XLA buffer breakdown — so these files can be
deleted whenever they stop earning their disk space.

Where the extracted docs and an old plan disagree, **the docs are right**: several claims
were corrected on the way out (see below).

## Live

- **REFACTOR_PLAN.md** (written 2026-08-22) — the four behaviour-preserving structural
  moves accepted from an outside refactoring assessment: transport out of
  `Parameters.__init__` (R), one run-loop body over a uniform `(state, aux)` carry (C),
  the linear operator as one typed pytree slot `kgrid.lin` (L), named `grads` and
  statically filtered RHS terms (G) — plus the Phase-0 forced-restart fix and the 3D /
  GDI / IMEX reference npz the gates need. §7 records what was rejected and why (hoist
  unification, dropping `simulate`, ETDRK, config rewrite). Four opus agents in parallel
  worktrees, every phase bitwise.
- **GDI_PLAN.md** — roadmap from the current IF-dissipation-only solver to the 3D GDI
  equations (exact linear propagators, spectral-z, low-storage IMEX). Execution order
  P1 → P4a → P2 → P3 → P4b. Physics source: `docs/GDI_nonlinear_equations (10).pdf`.
- **CODE_REVIEW_2026-07-31.md** — full-codebase review. Sections 2–5 are done; section 6
  (repo hygiene) is partly open.

## Finished

- **MEMORY_PERF_PLAN.md** (closed 2026-08-20) — memory and step-time reduction for FD-z
  (F1–F4) and z_spectral (Z1–Z3), with the two-card GPU assessment (Kaggle P100 via lugus,
  Savio GTX 2080Ti). Outcome: z_spectral RMHD 62 → 17–18 u, FD-z 30 → 17–23 u (platform-dependent), and every production path 0.43–0.98× its old step time (one recorded exception: at the
  default GRAD_CHUNK=1 the P100 FD-z step is ~1.04× — the §9.2 device split; set
  GRAD_CHUNK=2 there to recover it); all §9 decisions closed (GRAD_CHUNK stays 1,
  `hoist_propagator` kept and documented putzer2-only, the block stencil deleted). The
  durable record is in `docs/performance.md` ("Memory: where it goes and what was removed",
  the accounting section, "Tuning knobs, measured") and `docs/numerics.md` (separable
  propagator, putzer2 coefficients, the `grads` tuple). **MEMORY_PERF_PLAN_REVIEW.md**
  stays beside it — the plan cites it. The two measurement hand-offs it superseded,
  **TARANIS_MEMORY_HANDOFF.md** (XLA buffer audit) and **ZSPECTRAL_PROPAGATOR_NOTES.md**
  (step-time profile, hoisted propagators), moved to `old/` in the closing docs sweep.

- **PERFORMANCE_PLAN.md** — the original perf findings (F1, F2) and phased task list.
- **PHASE2_PLAN.md**, **PHASE3_PLAN.md** — execution plans expanding that roadmap.
- **PHASE3_RESULTS.md** — consolidated GPU-backend results. Its benchmarks are now in
  `docs/performance.md` and its appendix in `docs/numerics.md`; CLAUDE.md points at those.
- **TESTING_PLAN.md** — test-suite systematization (phases 0–7).
- **EXAMPLES_PLAN.md** — examples audit and rewrite.
- **HALO_WIDTH_PLAN.md** — parameterizing the z-halo width.
- **SERIAL_BACKEND_PLAN.md** — making MPI optional via `comm_backend="serial"`
  (A1–A5 landed 2026-07-31). The design, the resolution table and the A2/A5 audits are
  process; the durable parts are already in CLAUDE.md (comms/backend-resolution bullets),
  `docs/RUNNING_TESTS.md` (serial tier + CI), the two Savio setup guides and
  `docs/performance.md`. **One item is still open and is the user's**: Savio verification
  that the real mpi4jax and `"jax"` paths are unchanged (`mpirun -n 4 python
  tests/test_snapshot_roundtrip.py`, one forced-turbulence restart, one GPU backend job).
- **REVIEW_FIXES.md** — an earlier round of code-review fixes. Almost entirely
  "was X, now Y"; the invariants those fixes established already live in CLAUDE.md and
  `docs/checkpointing.md`, so nothing was extracted.

## Claims corrected during extraction

Four statements in `plans/old/` are wrong about the code as it stands. They were fixed on
the way into `docs/`, and are listed here so nobody re-imports them from the originals:

- **"lsrk_scan gives bitwise-identical trajectories at fp64"** (PHASE3_RESULTS appendix)
  is false. `tests/test_scheme_equivalence.py` measures ~2e-15 (lsrk33) and ~7e-15
  (lsrk54) after 20 steps under jax 0.6.2/CPU; the divergence is machine-dependent.
- **"T7 halo_start stays unregistered"** (PHASE2_PLAN verdict) is stale — it is registered
  and enabled for `comm_backend="jax"`, on the reasoning that the measurement may change
  with a different scheme or on NVLink/IB hardware.
- **`perp_inner_product_batch`** (PHASE3_RESULTS appendix) no longer exists; it is
  `perp_inner_product(..., batch=True)`.
- **The `#TODO` in `rmhd.LinearTerm`** that HALO_WIDTH_PLAN says "stands" was removed —
  `Parameters` now warns when `z_diff_order`/`z_diss_hyper` are set away from default.

One finding was never acted on and is recorded in `docs/performance.md` under "Known, not
done": PERFORMANCE_PLAN's F7/T10 async checkpointing. `run.py` still drains orbax
synchronously after every save.
