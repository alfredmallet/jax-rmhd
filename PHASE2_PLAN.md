# Phase 2 execution plan — agent handoff

Expands PERFORMANCE_PLAN.md Phase 2 (T5 → T6, T7) into concrete tasks for Opus agents,
incorporating everything Phase 1 taught us. Read PERFORMANCE_PLAN.md for the original
findings (F1, F2) and CLAUDE.md for the invariants — both are binding.

## Measured baseline (what Phase 2 is up against)

Savio savio3, 32 ranks, 128×128×256, lsrk54 + elsasser + adaptive dt, production config
(`forcing_norm_per_step=True`, full-grid RNG, scan LSRK, donation on):

| | fp32 | fp64 (production truth) |
|---|---|---|
| unforced | 153 ms/step | 302 ms/step |
| forced, nps | 174 ms/step | 353 ms/step |
| one allreduce | ~4 ms | ~4 ms |

Remaining comm per step (worst case): 10 halo `sendrecv` (5 stages × 2), 1 CFL
`allreduce(MAX)`, 1 forcing-scale `allreduce(SUM)`. **Phase 1's lesson: fp32 flatters
comm savings ~3× (nps was +27% at fp32, +8% at fp64). All Phase 2 accept/revert
decisions are made on fp64 numbers.** At 32 ranks the CFL allreduce is ~1% of a step —
T6's honest payoff is at higher rank counts (nz_local shrinks, latency share grows) and
in enabling Phase 3; benchmark at 32 ranks AND at ≥128 ranks (needs nz ≥ 256; use
nz=512 for 128–256 ranks) before judging.

## Ground rules for every agent

- Obey CLAUDE.md invariants (it now covers donation, checkpoint structure, kgrid–params
  binding, forcing_scale lifecycle, energy normalization, `_replace` construction).
- Comments: ~1 line per new function, ~1 line per change. No walls of text. No git
  commits — leave the working tree for review.
- Local verification (no MPI in the sandbox): `tests/local_mpi_stub.py` fakes
  mpi4py/mpi4jax single-rank (incl. dims=3 via self-send). Run scripts as
  `MPLBACKEND=Agg PYTHONPATH=.:tests python3 -c "import local_mpi_stub, runpy;
  runpy.run_path('tests/<script>.py', run_name='__main__')"`. Keep each run <40 s
  (bash calls are hard-capped at 45 s); small grids (≤64², nz ≤ 16); fp64 via
  `RMHD_PRECISION=64`. NEVER copy the repo tree (examples/data is multi-GB); extract
  old file versions with `git show <ref>:<path>` into /tmp selectively.
- Version A/B on Savio: select code via the `RMHD_PKG` env var mechanism in
  `bench/bench_phase1.py` — never PYTHONPATH (the env's editable install silently
  overrides it; this invalidated a whole benchmark round in Phase 1).
- Savio jobs: agents PREPARE sbatch scripts (conventions: account fc_kawturb,
  partition savio3, ntasks-per-node=32, module purge; anaconda3 gcc openmpi, jax_cpu
  env, thread-pinning env block, OMPI_MCA_pml=ucx) — the user submits them and drops
  the .out back. Design every job so one submission answers the question.
- Local test battery that must pass after every task: `tests/test_forcing_smoke.py`,
  `tests/test_forcing_norm_per_step.py` (both end "ALL PASS"),
  `tests/test_dissipation.py` (must complete). Savio battery before merge:
  `tests/test_restart_resharding.py` (-n 2 then -n 4, same snap_path),
  `tests/test_advection.py` (-n 4).

## A1 — T5: comm abstraction layer (pure refactor, blocks everything else)

New `jax_rmhd/comms.py`:

- `halo_exchange(f, params)` → returns `(recv_left, recv_right)` for a 2-wide halo
  (extract the sendrecv pair from `shared_physics.z_derivatives`);
- `allreduce_sum(x, params)`, `allreduce_max(x, params)` (array-valued x allowed —
  `_perp_reduce_batch` reduces a stacked vector);
- backend selected by `params.comm_backend` (new ctor arg, default `"mpi4jax"`, the
  only backend in this phase; unknown values raise at construction). Plain-Python
  dispatch on the static param, per CLAUDE.md.

Port the ONLY three call sites: `shared_physics.z_derivatives`,
`shared_physics._perp_reduce`/`_perp_reduce_batch`, `rmhd.set_timestep`. Nothing else
imports mpi4jax afterwards (grep to prove it; `local_mpi_stub` compatibility depends on
comms.py importing mpi4jax at module level exactly as the current code does — verify the
stub still intercepts by running the local battery).

Acceptance: (a) local battery passes; (b) fp64 trajectory equivalence vs pre-T5 code —
reuse the Phase 1 recipe (`git show` extraction + same-seed forced 3D run, compare final
fields; must be BITWISE zero, it's a pure refactor of identical ops); (c) the no-op
guarantee means no Savio benchmark is needed for this task alone.

## A2 — T6: `params.cfl_every` (depends on A1)

Design constraints discovered in Phase 1 — read carefully:

- **Do not put a collective inside `lax.cond`.** The tempting design (recompute dt when
  `step % N == 0` inside the stepper) traces the allreduce under a cond; even with
  rank-identical predicates this couples mpi4jax token semantics to conditional
  execution. Forbidden.
- **Do not let dt be computed from rank-local data.** Reusing a locally-computed dt
  with periodic global sync desyncs trajectories across ranks. dt must come from one
  collective, used identically by all ranks.
- Preferred structure: hoist dt out of the stepper. `block_of_steps` (and
  `simulate`'s `sim_to_next_snap`) become loops over inner blocks of `cfl_every` steps:
  compute dt once per inner block (via `equation_registry` `grad_func` +
  `set_timestep_func`, one allreduce), then scan `cfl_every` steps passing dt as a
  traced argument; the stepper's own `set_timestep` call is skipped when dt is
  supplied. `cfl_every=1` must reproduce current behavior — but note the stage-0 rhs
  currently doubles as the dt source, so a naive hoist adds one standalone `grad_func`
  eval per block; either accept it for N>1 only (keep the legacy path at N=1, bitwise
  identity preserved) or thread stage-0 grads out of the stepper (more invasive; only
  if the cheap route measurably hurts).
- **No new SimulationState fields if avoidable.** dt recomputed at block start needs no
  persistence across checkpoints (same pattern as forcing_scale's recompute-on-start).
  If a state field turns out to be unavoidable, Phase 1's C1 lesson applies in full:
  always-concrete (never None), extend `old_snapshot_repair` with the new legacy
  template, update `load_snapshot`/resharding-test templates, and note it in CLAUDE.md.
- Physics caveat to document at the flag: with dt frozen for N steps, CFL can be
  transiently violated while the flow accelerates; users compensate with `cfl_safety`
  margin. Default `cfl_every=1` (exact current behavior).

Acceptance: local battery; fp64 equivalence at `cfl_every=1` vs A1 code (bitwise);
a short forced run at `cfl_every=5` stays finite and tracks the `cfl_every=1` run
statistically (energy within a few % at t~5); prepared Savio benchmark case (see A4).

## A3 — T7: halo issue-early reorder (depends on A1; exploratory, revert-biased)

Reorder the RHS so `halo_exchange` for `LinearTerm` is issued before the
perpendicular FFT/bracket work, giving the transport a chance to proceed concurrently.
Honest expectation: with mpi4jax's token chain serializing comm with compute, this may
win nothing on the CPU backend — the reorder's real value is that the T5 abstraction
makes it trivial for the Phase 3 NCCL backend where overlap is real. Implement it as a
small structural change (no flag if it's neutral-or-better; behind a flag only if it
changes numerics — it shouldn't, same ops, same values, possibly different schedule).
Rule: if the A4 benchmark shows no fp64 win at any rank count, revert to the simple
ordering and record the negative result in this file rather than keeping dead
complexity.

## A4 — benchmark round and gate (after A1–A3; user submits)

Extend `bench/bench_phase1.py` with flags `cfl_every=<N>` and `halo_early` (same
monkeypatch-or-param style as `nps`/`shellrng`), and write
`slurms/bench_phase2.sh` + `slurms/bench_phase2_scale.sh`:

- bench_phase2: savio3, 1 node / 32 ranks, nz=256, fp64, forced+nps: {A1 baseline,
  cfl_every 1/5/20, halo_early, best-combo} × 2 passes, plus one unforced pair.
- bench_phase2_scale: 4 nodes / 128 ranks, nz=512, fp64, same cases — this is where
  T6/T7 must show up if they show up at all. (256²-perp keeps memory sane; nz=512
  keeps nz_local=4 ≥ halo width 2... it is NOT ≥ 2×halo; use nz=1024 if
  nz_local≥2×halo is required — check z_derivatives' assumption before sizing.)
- Decision gate (fp64): keep cfl_every default 1 but document the measured win per N;
  keep or revert T7 per its numbers; update the baseline table above with Phase 2
  results.

## A5 — adversarial review + fix round (after A4 numbers are in)

Spawn a fresh Opus agent, read-only, against the full `git diff main` (or the Phase 1
merge point if Phase 1 was merged), with the same brief as Phase 1's review: rank
findings CRITICAL/MAJOR/MINOR, verify the specific traps above (cond+collective,
rank-local dt, checkpoint structure, stub compatibility, token-order changes from T7),
run the local battery itself, verdict on merge-safety. Then a fix round addressing
CRITICAL/MAJOR findings, rerunning the battery. Phase 1's review caught a
restart-breaking bug the tests couldn't see; treat this round as mandatory, not
optional.

## Sequencing

A1 → (A2, A3 in either order or parallel-by-file: A2 touches run.py/config.py, A3
touches physics/rmhd.py + shared_physics.py; both touch comms.py call sites — if
parallel, A2 first is safer) → A4 (user submits, results interpreted) → A5 → merge
gate. Each agent ends by updating this file's status line below.

## Status

A5 review verdict: merge-safe with fixes; no CRITICAL. Fixes applied 2026-07-25:
MAJOR-1 (cfl_every>~5 from quiescence silently NaNs — documented at the flag and in
CLAUDE.md, N>1 only from developed states; bench IC is non-quiescent so A4 numbers
unaffected), MAJOR-2 (params.save now backfills signature defaults for keys missing
from older records instead of hard-erroring — adding ctor args no longer invalidates
run dirs), MAJOR-3 (CLAUDE.md: 5-field EquationRecipe, 5-arg term-func contract,
halo_start_func hook, comms.py section), MINOR-1 (unforced pair de-confounded: baseU /
cfl20U+halo_late / t7U), MINOR-2 (-x all pinning env vars in both sbatch scripts),
MINOR-3 (numpy ints accepted for cfl_every), MINOR-4 (scale job 1h, `|| true` on the
grep pipe so one crashed case can't abort the matrix). Deferred: MINOR-5
(estimate_good_nblock rounding — documented behavior).

- A1 (T5): DONE — `jax_rmhd/comms.py` (`halo_exchange`/`allreduce_sum`/`allreduce_max`,
  backend dispatch on new `params.comm_backend`, default/only `"mpi4jax"`); all 4 mpi4jax
  call sites ported (the three planned + `diagnostics.perpspec`); local battery ALL PASS;
  fp64 20-step forced 3D A/B vs pre-T5 code bitwise identical (max|diff| = 0.0).
- A2 (T6): DONE — `params.cfl_every` (int>=1, default 1) hoists dt out of the stepper:
  `run._cfl_block` computes one dt per block (grad_func+set_timestep_func, one allreduce)
  and passes it to `rk_advance`/`lsrk_advance` via the new `dt_override` arg; used by both
  `block_of_steps` (nblock still counts steps, rounded up to whole blocks) and `simulate`'s
  `sim_to_next_snap` while_loop (which may now overshoot by up to `cfl_every` steps).
  `cfl_every=1` and `adaptive_timestep=False` keep the verbatim legacy path (both verified
  bitwise identical); forcing still updates every step; CFL allreduces per 10 steps 10 -> 2
  at `cfl_every=5`; local battery ALL PASS; energy at t~5 differs 1.2% between N=1 and N=5.
- A3 (T7): DONE — optional `EquationRecipe.halo_start_func` (`rmhd.halo_start`, None in 2D)
  issues the z halo exchange at the top of `construct_rhs`'s rhs, before grad/NonlinearTerm;
  the recv buffers thread to every term func as a 5th arg (`halo=None` default) and into
  `z_derivatives(f,params,halo=None)`, whose legacy 2-arg call still exchanges itself. Pure
  reordering: fp64 20-step forced 3D and 10-step forced 2D A/B vs pre-T7 code bitwise
  identical (max|diff| = 0.0); local battery ALL PASS. All touch points marked `# T7:`;
  awaiting A4's fp64 numbers for keep-vs-revert.
- A4 (benchmarks): PREPARED — awaiting user submission of slurms/bench_phase2.sh +
  bench_phase2_scale.sh. `bench/bench_phase1.py` gained `cfl<N>` (sets `p.cfl_every`),
  `halo_late` (registry `_replace(halo_start_func=None)` = T7 reverted inside the new
  code), `nb<N>`/`nr<N>` block/rep overrides, and a corrected step count (block_of_steps
  rounds nblock up to whole cfl blocks); both flags appear in the printed `[tags]`.
  Case matrix in both jobs: {halo_late, T7-on} x cfl_every {1,5,20}, 2 passes, nb20 nr4;
  32-rank job adds an unforced cfl1-vs-cfl20 pair. No old-package extraction needed
  (Phase 2 baseline = this repo at cfl1+halo_late).
- A5 (review): not started
