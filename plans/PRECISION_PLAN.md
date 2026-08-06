# Precision decoupling plan — fp64 time (and scalars) under fp32 fields

**Status: PLAN ONLY — not yet executed** (written 2026-08-06). Decide route before
starting; the main plan is Route A. CLAUDE.md invariants are binding throughout.

## Goal

Make `RMHD_PRECISION=32` mean "fields are complex64", not "the world is 32-bit", so
`SimulationState.t` (and, optionally, energy-like reductions) can be fp64 in fp32
production runs.

## Motivation (quantified)

- **Hard failure:** with naive fp32 accumulation, `t + dt == t` exactly once
  `t/dt > 1/eps32 ≈ 1.7e7` steps — time freezes while fields keep advancing, and
  `while state.t < t_end` never exits. A 1024^3 production campaign can plausibly reach
  this.
- **Soft failure:** roundoff random-walks t by ~sqrt(N)·eps32 (~6e-5 relative at 1e6
  steps) — snapshot-spacing jitter, frequency-measurement noise.
- **Payoff beyond t:** once x64 is available, energy/budget reductions can accumulate at
  fp64 over fp32 fields, recovering ~1e-15 budget closure (currently limited to ~1e-6 at
  fp32). Optional follow-on, Appendix B.

## Design decision

**Route A: always enable x64; pin field dtypes explicitly.** `jax_enable_x64=True`
unconditionally at import; a new `_precision.py` module owns `precision` ("32"/"64"),
`ftype`, `ctype` read from `RMHD_PRECISION` once at import. Everything that today infers
precision from `jax.config.read("jax_enable_x64")` re-points to this module.

Rejected / fallback alternatives:

- **Kahan-compensated t at fp32** (Appendix A): small blast radius, fixes both failure
  modes, but leaves every other scalar at fp32 and adds a `SimulationState` field
  (checkpoint-format churn). Fallback if Route A's audit turns up a blocker.
- **Per-call `jax.experimental.enable_x64` contexts:** interacts badly with jit caching
  and the persistent compilation cache; rejected.

Why this is safe to attempt: with x64 on, an op graph whose inputs are all
fp32/complex64 compiles to the same fp32 kernels as today — the flag only changes
*default* dtypes and promotion of 64-bit inputs. The entire risk is silent upcasts,
which the dtype-leak test (A4) converts from silent perf/precision bugs into loud test
failures.

## Promotion rules (the audit's decision procedure)

1. Python scalars are weak-typed: `0.5*fields`, IMEX tableau entries (python floats from
   the scheme tuples, e.g. `(b[k]*dt)*z`) do NOT upcast. Leave them alone.
2. `jnp.array(...)`/`np.array(...)` without dtype are STRONG float64 under x64 and
   poison every op they touch. Every such array that ever multiplies fields must be
   pinned to `ftype`/`ctype`.
3. `jax.random.*` draws without explicit dtype produce float64 under x64 — AND a
   different bitstream than float32 draws. Pin `dtype=ftype` explicitly; this keeps the
   fp32 RNG stream bitwise identical to today's.
4. fp64 is allowed only in quantities that never multiply fields: `t` (and later,
   reduction accumulators). Any place a t-derived quantity re-enters field math must
   downcast explicitly (there is exactly one today: see A3).

## Touch-point inventory (verified against source, 2026-08-06)

Dtype *sources* (create arrays with default dtype today):

- `run.py::initialize` — `linspace` x/y (19-20); fields from `fft(f(...))·mask` (23-25);
  `forcing_scale = jnp.zeros((n_ou,))` (32); `t=0.0` (33).
- `grids.py::setup_kgrids` — kx/ky/arange/dealias arrays and `lin_L`/`lin_m`/`lin_s2`
  (built from numpy via `linear_matrix_func` → float64 under x64). All multiply fields:
  pin to `ftype`/`ctype`. (`local_z_coords` arange at 155 feeds ICs only.)
- `timestepping.py:77-79` — `alphas_arr/betas_arr/gammas_arr = jnp.array(scheme.·)`:
  strong-typed, multiply `delta`/fields inside the scan. Pin to `ftype`. (The unrolled
  LSRK loop and all IMEX tableaus use python floats — weak, safe, rule 1.)
- `physics/shared_physics.py:50,62` — `jax.random.normal(key, shape)` with no dtype
  (then `.astype(dtype)`): pin `dtype=ftype` per rule 3 to preserve the fp32 stream.
- `diagnostics.py:8` — `bin_edges = jnp.arange(...)`: reduction-side only; fp64 here is
  harmless-to-beneficial. Decide in A2 and record.

Precision-flag readers to re-point at `_precision`:

- `propagators.py:33-34` — Taylor-branch tolerance selection.
- `config.py:262,318-328` — params.json `_precision` record + mismatch warning. The
  recorded value keeps meaning FIELD precision (`RMHD_PRECISION`), so old records stay
  comparable; no format change.
- `snapshot_io.py:27-30::get_dtypes` — becomes a re-export of `_precision.ftype/ctype`.
- `tests/conftest.py` / `tests/_rmhd_testing.py` — audit for fp32/fp64 marker logic that
  reads `jax_enable_x64` (`_script_skip_reason` and friends); re-point.

t plumbing (all verified):

- Steppers advance t per sub-stage: `state._replace(t=state.t + gamma*dt)` /
  `t=state.t + c[k]*dt` / `t=state.t + dt` (timestepping.py 27-37, 71, 95-97, 191-194,
  212-217, 243-246). With t fp64 and dt `ftype`, promotion gives fp64 t — correct,
  no change needed. Sub-stage t only enters `state`; no term func reads t today
  (grep before relying on this — and see the A3 guard for the future).
- `run.py::_advance_forcing:43` — `dt = new_state.t - prev_t` is **the one place a
  t-difference re-enters field math** (→ `ou_update` → OU decay factors × complex64
  `forcing_state`). Downcast: `dt = (new_state.t - prev_t).astype(ftype)`.
- `run.py` host side — `float(state.t)`, `state.t < t_end`, `min(t_last_snapshot+t_snap,
  t_end)` passed as traced `target_t`: python floats are fp64 natively; fine as is.
- `while_loop`/`scan` carries: t dtype must be fp64 *consistently* from `initialize` /
  `load_snapshot` onward or the carry structure check fails at trace time — which is the
  desired loud failure, not a risk.

## Tasks

Sequential; each ends with the full suite green at BOTH precisions
(`make test`-equivalent: fp64 then fp32 sessions).

**A1 — `_precision.py` + import rewiring.** New module reads `RMHD_PRECISION` once;
`__init__.py` sets `jax_enable_x64=True` unconditionally (keep the precision print,
now sourced from `_precision`), imports `_precision` before anything touches jax
numerics. Re-point the four flag-reader sites. No behavior change yet at fp64; fp32
will transiently upcast until A2 — so A1+A2 land as one commit, split only for review.

**A2 — pin the dtype sources.** Work through the inventory above. Single choke points
preferred over scattered `astype`s: cast fields once after the IC fft
(`.astype(ctype)`), cast kgrid arrays at construction in `setup_kgrids` (the only
sanctioned constructor — one site), pin the three LSRK tables, pin the two RNG draws,
`t=jnp.float64(0.0)`, `forcing_scale`/`zeros` pins. Decide + record the
`diagnostics.py` choice.

**A3 — t at fp64 + the forcing-dt downcast + a tripwire.** The `_advance_forcing`
downcast per the inventory. Add a comment at the downcast site: any future
time-dependent term func must do the same before mixing t into field math. Cheap
runtime tripwire in `construct_rhs` or the stepper (debug/assert-level, not traced
overhead): `assert state.fields.dtype == _precision.ctype`.

**A4 — snapshot round-trip.** Read docs/checkpointing.md first (binding). New saves
carry fp64 t (orbax handles per-leaf dtypes; the restore template's
`ShapeDtypeStruct` for t becomes float64). OLD snapshots carry fp32 t: restore with the
stored dtype, then cast — same repair pattern as `forcing_scale` (`snapshot_io.py:262`
already funnels t through `jnp.asarray`; make the cast explicit). Invariants intact:
no `CheckpointManager` on reads, index broadcast from rank 0, `forcing_scale` always
concrete.

**A5 — tests.**
- *Dtype-leak test* (the load-bearing one): under fp32, run one jitted step of each
  scheme family (one IF, one IMEX) + a forced step; assert `fields.dtype==complex64`,
  `forcing_state.dtype==complex64`, `forcing_scale.dtype==float32`, `t.dtype==float64`.
  Run it fp64 too (expect complex128/float64). This test is what makes the whole
  design safe against future regressions — new code that leaks a float64 array fails
  here, not in a benchmark three months later.
- *t-accumulation test*: 1e4 steps of `t += dt` with `dt ~ 1e-4·t`-scale ratio; assert
  fp64-exact against a python-float reference sum.
- *RNG-stream check*: forced fp32 step reproduces a recorded pre-change reference
  (guards rule 3). Generate the reference at the START of implementation, before A1.
- One-off (not a committed test): short forced fp32 run before vs after the change —
  expect bitwise identity except t's dtype; record the outcome in this file's status
  header, with tolerances if bitwise fails (same class of caveat as `lsrk_scan`
  fusion sensitivity).

**A6 — docs.** numerics.md gains a "Precision model" section (fp32 spectral noise shelf
eps²·E_peak; FD z-derivative error floor — Appendix C is the draft); CLAUDE.md setup
line changes meaning ("RMHD_PRECISION sets FIELD precision; t and x64 are always on");
docs/checkpointing.md notes the t-dtype repair rule.

## Acceptance

1. Full suite green at both precisions; fp64 session bitwise-unchanged (x64 was
   already on there — any diff means a pin was wrong).
2. Dtype-leak + RNG-stream + t-accumulation tests green.
3. fp32 before/after comparison recorded (bitwise or tolerances + reason).
4. No perf regression at fp32: one `bench/` datapoint, since a missed pin shows up as
   a silent ~2× slowdown even if tolerances pass.
5. Old-snapshot (fp32-t) restore verified.

## Risks / forbidden patterns

- **Never** infer precision from `jax.config.read("jax_enable_x64")` again — grep for
  it in review; only `_precision` may know.
- **Never** let a strong float64 array into field math: no bare `jnp.array`/`np.asarray`
  of numeric lists that touch fields; pin or use python-scalar weak types.
- `Parameters` hashes by identity and is closed over — nothing here changes that; do
  not move dtypes into `Parameters` (import-time constant, same everywhere).
- mpi4jax collectives: dt/CFL reductions stay `ftype` end-to-end (dt never becomes
  fp64 — only t is); all ranks read the same `RMHD_PRECISION` env or halo dtypes
  mismatch — worth one assertion at `Parameters` construction.
- Donation is untouched (t is a scalar leaf; `_replace` semantics unchanged).

## Appendix A — fallback: Kahan-compensated t at fp32

Carry `(t, t_err)`; per advance: `y = dt - t_err; s = t + y; t_err = (s - t) - y;
t = s`. Error stays O(eps·t) for arbitrary N and the freeze cannot occur (lost low bits
are carried). Costs: a new `SimulationState` field → old-snapshot default-fill repair
(forcing_scale pattern), touches every `_replace`-free construction site (none should
exist — positional construction is already banned), and sub-stage t updates inside
steppers need the compensated add too (or accept sub-stage t at fp32 accuracy — it's
transient; only the end-of-step add needs compensation). ~30 lines total. Choose only
if Route A's audit stalls.

## Appendix B — follow-on: fp64 reductions (optional, after A1-A6)

With x64 available, accumulate `perp_reduce`-family sums, `energy`, `perpspec` bins,
and the forcing normalization denominator at fp64 (`.astype(float64)` before the sum,
or `jnp.sum(..., dtype=...)`), downcasting the *forcing scale* back to `ftype` before
it multiplies fields. Restores ~1e-15 budget closure on fp32 fields. Keep ALL
energy-like diagnostics on the one shared normalization convention (CLAUDE.md); the
allreduce dtype changes to fp64 for these paths only. Do as its own reviewed change —
it moves numbers (at the 1e-7 level) in every diagnostic, so tests with recorded
references need regenerating.

## Appendix C — fp32 z-stencil error model (draft for numerics.md)

For a mode with parallel wavenumber k∥ stored at fp32, the 4th-order first-derivative
stencil has relative error ≈ (k∥dz)⁴/30 (truncation) + eps/(k∥dz) (roundoff, dominated
by fp32 STORAGE quantization of f — upcasting the stencil arithmetic recovers nothing).
Total is minimized at k∥dz ≈ (7.5·eps)^{1/5} ≈ 0.05 at fp32 (eps=6e-8), giving a floor
of ~1e-6 relative; refining z below this makes ∂∥ WORSE (roundoff term ∝ 1/(k∥dz)).
At fp64 the sweet spot is k∥dz ≈ 1e-3, floor ~1e-13 — invisible in practice. Notes:
(i) the stencil acts per (kx,ky) coefficient, so the error is relative to each perp
mode's own amplitude; critical balance (k∥ ∝ k⊥^{2/3}) puts the worst conditioning on
the outer-scale modes; (ii) the spurious content lands at grid-scale k_z with amplitude
~eps·|f| — exactly what z_diss damps; keep it on at fp32; (iii) `z_spectral` avoids the
cancellation entirely (∂∥ is the exact ±i·k_z in lin_L, applied as a unitary phase by
`apply_exp`) and is the fp32-robust parallel formulation, currently size==1 only.
Practical rule: at fp32, size nz so outer-scale k∥dz ≳ 0.05.
