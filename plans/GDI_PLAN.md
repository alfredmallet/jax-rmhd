# Linear-propagator → GDI execution plan

Roadmap from the current IF-dissipation-only solver to the 3D GDI equations, via general
exact linear propagators, a spectral-z mode, and low-storage IMEX. Physics source:
docs/"GDI_nonlinear_equations (10).pdf" (equation numbers below refer to it). CLAUDE.md
invariants are binding throughout. Execution order: **P1 → P4a → P2 → P3 → P4b**
(P4a is deliberately early: it exercises P1 on real physics before P2/P3 build on it).

## Design background (why the plan is shaped this way)

- The stiff terms in every target equation set are **linear and k-local**: an
  (nfields×nfields) block per perpendicular (or 3D) wavenumber. Treated exactly by a
  per-mode propagator, they impose no CFL limit and no accuracy floor.
- For nfields=2 the matrix exponential has a **closed form** (Putzer/Sylvester:
  eigenvalues m±s, m=trL/2, s=sqrt(m²−detL); exp(Lτ)=e^{mτ}[cosh(sτ)I+(sinh(sτ)/s)(L−mI)]).
  No eigendecomposition, no proj storage, robust at defective points via the small-|sτ|
  Taylor branch. **s² can be negative or complex (wave/growing modes) — all propagator
  arithmetic must be complex; a real sqrt silently NaNs.** Dense-eig backends are NOT
  built in this plan; the hook interface just leaves room for them.
- **IF vs stiff quasi-static balance**: IF-RK treats the linear physics exactly but
  misweights the nonlinear forcing of stiffly damped modes when γΔt ≳ 1 (fixed-node
  quadrature of a boundary-layer integral: gives u ~ Δt·N instead of u ≈ N/γ). GDI's
  observable physics (N–φ cross-phase, H-W-type drive ∝ 1/γ∥ deviation from
  adiabaticity) lives exactly in that balance, and the stiff rate γ∥/k⊥²ρs² is largest
  at the LARGEST perp scales — no resolution choice removes it. Hence P3 (L-stable
  IMEX, which recovers the quasi-static limit) before production 3D GDI. RMHD is
  unaffected (its stiff operator only acts at the grid cutoff); tearing benchmarks are
  safe on IF (stiffness parameter η/(v_A dx) ≪ 1 in any layer-resolved run).
- γ∥ = D∥kz² is diagonal in kz, so 3D GDI lives in the spectral-z single-GPU mode (P2).
  The 2D closure model (γ∥ → νin k⊥²ρs², eq 4.3) is perp-k-local and mild (the
  vorticity-equation relaxation rate is exactly νin), so 2D GDI (P4a) needs only P1 and
  runs on the existing IF-LSRK schemes.

## Future-proofing notes (not in scope, but the interfaces must not preclude them)

- **Scheme taxonomy**: exponential (IF/propagator) for oscillatory stiffness — waves,
  KAW dispersion, weak turbulence (unitary, exact phases; interaction-picture allows
  ωΔt > 1) — available only where z is spectral (or the term is perp-k-local);
  L-stable IMEX for dissipative stiffness — γ∥, collisions (recovers the quasi-static
  limit). L-stable solves artificially damp waves at ωΔt ≳ 1: never use the L-stable
  path for wave-dominated linear terms. Grid-in-z waves (multi-GPU): explicit by
  default (free in the CB-tailored regime, next bullet); if the wave CFL ever binds,
  the escape hatch is trapezoidal/CN implicit (|amplification| ≡ 1: stable, undamped,
  phase-errored only at corner modes) through the same parallel banded solve_shifted
  backend — CN/RKW3-style pairing, NOT the L-stable CB schemes.
- **KAW / strong CB turbulence**: explicit parallel+dispersive terms are fine provided
  the box is CB-tailored (k∥max matched to the critical-balance cone at the grid perp
  scale) — the perp advection CFL then binds. Untailored boxes reintroduce corner-mode
  stiffness ∝ k⊥maxρs (dispersion is superlinear in k). Record the tailoring assumption
  in any KAW run setup.
- **`solve_shifted` must not be assumed elementwise.** A future multi-GPU grid-in-z
  backend implements it as a distributed banded solve (Schur-complement/SPIKE: local
  Thomas elimination + a ~2-perp-plane interface allgather + redundant reduced solve,
  batched over perp modes, via a new comms.py primitive). Exact reduced solve required —
  the shifted inverse's decay length ~ sqrt(aΔtD∥) grid points is long-range precisely
  when stiff, so interface-truncating approximations (PDD) are forbidden. The CB-IMEX
  steppers see only the hook and need no changes.

## Ground rules for every task

- Obey CLAUDE.md. Comments ~1 line per new function/change; no git commits — leave the
  tree for review.
- Local verification: everything in this plan except the P1 regression of the 3D FD
  path is **single-rank testable** (2D GDI is dims=2; z_spectral requires size==1).
  For 3D-FD paths use `tests/local_mpi_stub.py` as in Phase 2 (run recipe there).
  Keep runs <40 s, small grids, fp64 via `RMHD_PRECISION=64` for all
  equivalence/physics checks. NEVER copy the repo tree in the sandbox.
- Local battery after every task: `tests/test_forcing_smoke.py`,
  `tests/test_forcing_norm_per_step.py` (both "ALL PASS"), `tests/test_dissipation.py`
  (completes), plus the new tests this plan adds as they land.
- Sign convention, fixed now: recipes provide L such that ∂t f = L f + N(f); the
  propagator is e^{Lτ}. (The gdi-branch `matrix_exp` was written for the ∂t f + M f = 0
  convention — do not trust it; verify any port against `scipy.linalg.expm` directly.)
- **Agent staffing**: a Fable session oversees; coding tasks are handed off to Sonnet
  or Opus subagents as appropriate — Opus for core-solver/convention-touching work
  (P1 stepper+hook machinery, P2's transform/normalization sweep, P3 scheme
  implementation from the paper), Sonnet for self-contained modules and tests
  (physics/gdi.py terms, diagnostics, notebooks, test scaffolding). Each handoff
  brief must include the relevant CLAUDE.md invariants and this file's task section.

## P1 — linear-propagator machinery + eqpars

The one architectural decision: **the stepper calls a propagator hook; it never sees a
matrix.** Recipes supply `linear_matrix_func(kgrid, params) → L`; setup builds a
propagator object with two methods used by the steppers:

- `apply_exp(arr, tau)` — multiply by e^{Lτ} (IF schemes; τ = γᵢ·dt etc.),
- `solve_shifted(arr, a)` — apply (I − a·L)⁻¹ (unused until P3; define the interface now).

Bundled backends, selected by L's shape (as on the gdi branch):

- **diagonal**: L shape `(nfields, 1, nkx, nky)` (today's hdiss convention).
  `apply_exp` = elementwise `jnp.exp(L*tau)*arr` with the SAME op order as the current
  `jnp.exp(diss_exponents*gamma)` — the RMHD bitwise-equivalence gate depends on it.
  `solve_shifted` = elementwise `arr/(1−a*L)`.
- **putzer2** (nfields=2 only): L shape `(2, 2, [nz-or-1,] nkx, nky)`. Precompute
  m=trL/2 and s²=m²−detL as static arrays at setup; per-step elementwise cosh/sinh
  (complex dtype throughout) with a Taylor branch for |sτ| < tol on sinh(sτ)/s — the
  branch threshold is precision-dependent; evaluate coefficients at fp64 and cast when
  RMHD_PRECISION=32. `solve_shifted` = closed-form 2×2 inverse via det(I−aL).
- Shape convention supports an optional z/kz axis (broadcast size-1 in P1/P4a; real kz
  extent arrives in P2/P4b). Storage lives on `K_Grids` as new optional fields (None
  default) built inside `setup_kgrids` from the recipe — this inherits the kgrid
  sharding/`to_global`/specs plumbing and the "setup_kgrids is the only constructor"
  invariant for free. `kgrid.hdiss` is REMOVED; timestepping reads only the hook.
- Setup-time assertions: hermitian-symmetry compatibility L(−kx,ky)=conj(L(kx,ky)) on
  the ky=0 and Nyquist rows (reality preservation); nfields=2 required for putzer2.

Both steppers convert: `rk_advance` (diss_full/diss_half become apply_exp at τ=dt,
dt/2 — composition is exact since one fixed L commutes with itself) and `lsrk_advance`
(scan AND unrolled paths; the factor applies to both the delta update and the fields,
as now). `_replace` discipline unchanged.

**eqpars** (same task): new `Parameters` ctor arg `eqpars=None` (plain-JSON dict),
recorded and compared in params.json. RMHD's `diss`/`hyper` MOVE into eqpars and the
RMHD `linear_matrix_func` builds the (diagonal) dissipation L from them —
`setup_kgrids` stops reading `params.diss`. Compatibility: `from_snapshot` gains a
legacy shim (top-level `diss`/`hyper` in old records fold into eqpars with a warning);
`params.save`'s backfill logic must treat the moved keys as equivalent, not differing;
the JSON list→tuple restore must recurse into eqpars values. `Parameters` hashes by
identity, so a dict attribute is safe.

Acceptance: (a) local battery; (b) **fp64 bitwise A/B vs pre-P1 code** for 2D and 3D
forced RMHD (diagonal backend is a pure refactor). Same `comm_backend` on both sides —
the no-MPI sandbox auto-resolves to `"serial"` (added 2026-07-31), which is fine but is
NOT bitwise vs size-1 mpi4jax, so never mix backends across the A/B; (c) new
`tests/test_linear_propagator.py`: putzer2 vs `scipy.linalg.expm` over random 2×2
batches (complex, defective, growing, oscillatory cases; both precisions), plus a
rotation test — conjugate a diagonal system by a constant 2×2 rotation, evolve with
putzer2, undo the rotation, match the diagonal-backend trajectory to fp64 roundoff;
(d) legacy params.json loads via the shim with a warning.

## P4a — 2D GDI (needs P1 only)

New `physics/gdi.py` + registry entry `"GDI"` (nfields=2, fields (N, φ); raise on
dims==3 until P4b). Equations (5.4)–(5.5), normalized ρs=1, cs=1, Ωi=1;
eqpars = {Ln, nu_in, v0, gpar_fac, diss, hyper} with the 2D current-closure floor
γ∥ = gpar_fac·νin·k⊥²ρs² (eq 4.3; gpar_fac ~ α of eq 4.7, default 1).

- `grad`: N, φ, ∇²φ (+ gradients for the brackets). `NonlinearTerm`: {φ,N} and
  {φ,∇²φ}/(−k⊥²) via `inv_ksq` (zero mode masked — the φ equation is divided by −k⊥²).
  No LinearTerm, no ForcingTerm (instability-driven; forcing=False).
- `linear_matrix_func`: the 2×2 of eq (5.3) (converted to the ∂t f = L f sign
  convention, normalized units) INCLUDING perp hyperdissipation on BOTH fields on the
  diagonal — the model has no intrinsic small-scale cutoff (γ → const at k⊥ρs ≫ 1,
  eq 3.14), so hyper-diss closes the range and convergence in (diss, hyper) must be
  checked in production runs.
- `set_timestep`: E×B CFL from |∇φ| (+ the eps floor). L is exact so it imposes no
  stability limit, but nonlinear-vs-linear accuracy near saturation wants
  dt·max|Reλ(L)| ≲ 1: compute max|Reλ| once at setup (static) and include it as a dt
  ceiling with its own safety knob.
- Diagnostics (respect the shared perp normalization convention or numbers won't be
  comparable — CLAUDE.md): energy E = ½⟨N² + |∇⊥φ|²⟩ and enstrophy
  Z = ½⟨(N − ∇²φ)²⟩ (eqs 3.16–3.17); dE/dt budget vs eq (3.18) as a verification
  diagnostic; N–φ cross-phase and amplitude-ratio vs k⊥ (eqs 4.6–4.8) as the science
  diagnostic.
- Validation, `tests/test_gdi_linear.py`: initialize single-k eigenmodes, measure
  complex ω, compare against the 2D dispersion relation (2.8) in both regimes
  (collisional 2.9, inertial 2.11) and against the closure-model quartic; nonlinear:
  fp64 energy-budget closure over a short run; qualitative k⊥⁻³-ish saturation
  spectrum in a notebook (`examples/gdi-2D.ipynb`).

## P2 — spectral-z mode (needs P1; single-GPU/single-rank)

`params.z_spectral` (default False; requires dims==3, size==1, `comm_backend`
irrelevant/no comms — raise otherwise), recorded in params.json (fields change meaning:
snapshots are NOT cross-mode compatible; the params.save comparison is the guard).

- `grids.fft/ifft` gain a params argument and dispatch: rfft2/irfft2 ↔
  rfftn/irfftn over axes (−3,−2,−1) (complex over z and x, real over y) — fields keep
  shape `(nfields, nz, nkx, nky)`, axis 1 now meaning kz. Update the call sites
  (rmhd/gdi NonlinearTerm, run.initialize). State is fully spectral everywhere; grad's
  ifft lands the brackets in real (x,y,z) automatically and the stepper needs no
  transforms at all.
- `kgrid.kz` array; `dealias` becomes 3D (kz 2/3 mask broadcast onto the perp
  elliptical mask) — brackets are pointwise in z so products alias in kz; the existing
  `* kgrid.dealias` in the nonlinear terms and the IC masking in `initialize` pick the
  kz cut up with zero physics-code changes.
- RMHD `linear_matrix_func` gains the wave coupling: off-diagonal ±i·kz entries
  (Elsasser eigenvalues ±ikz emerge from putzer2 automatically) alongside the
  dissipation diagonal; optional kz hyperdissipation knob (−z_diss_k·kz⁴ diagonal,
  default off — exactness makes it a truncation choice, not a stability need).
  `LinearTerm`/`halo_start` are skipped in z_spectral mode (plain `if` on the static
  param); `set_timestep` drops the 1/dz and z_diss entries (Alfvén propagation exact —
  this is the payoff: dt set by perp advection only).
- **Normalization sweep, one commit, all together**: `_perp_reduce`/`_perp_reduce_batch`
  (÷nz → ÷nz², Parseval over the unnormalized z-FFT), `perpspec`, `energy`, forcing
  power. `parspec` becomes a trivial kz-sum (and works in this mode by construction).
  `reconstruct_envelope` becomes an exact scatter: A cos + B sin → (A∓iB)·nz/2 at the
  kz = ±2π/Lz planes. Cross-check: forcing injection power in z_spectral mode equals
  the real-z computation to roundoff (Parseval), enforced in a test.
- Validation, `tests/test_z_spectral.py`: (a) linear Alfvén wave dispersion exact to
  machine precision (single (k⊥,kz) mode, measured ω vs kz); (b) A/B vs the FD-z code:
  same IC, moderate nz, decreasing dz — FD result converges to the spectral one at 4th
  order; (c) forcing-power Parseval check; (d) orzag-tang-3D notebook variant runs.

## P3 — low-storage IMEX (CB-IMEX; needs P1, wants P2 landed for testing)

Read Cavaglieri & Bewley, JCP 286:172 (2015). Implement their low-storage IMEX-RK
schemes (agent selects the recommended 2- and 3-register, 2nd/3rd-order, L-stable-
implicit members from the paper) as new entries in `_scheme_registry` — the existing
IF-LSRK schemes remain untouched as the RMHD production path.

- **All of L is implicit** (dissipation included; no exponential anywhere in the IMEX
  path): each implicit stage is one `solve_shifted(·, aᵢᵢΔt)` call — elementwise for
  the diagonal backend, closed-form 2×2 inverse for putzer2. Explicit part = the
  summed term_funcs (nonlinear + forcing).
- Adaptive dt: shift coefficients recomputed per step, elementwise — same cost class
  as today's `exp(hdiss·dt)`. `cfl_every` blocks (frozen dt) are compatible for free.
- Verify low-storage register count in the implementation (the point of the exercise);
  keep the scan-vs-unrolled option symmetric with lsrk if trivial.
- Acceptance, `tests/test_imex.py`: (a) measured convergence order on a nonlinear test
  problem with known solution; (b) **stiff quasi-static test** (the motivating
  physics): u′ = −γ(u − g(t)) + weak NL, γΔt ≫ 1 — IMEX tracks u ≈ g + O(1/γ);
  demonstrate and record the IF-LSRK error on the same problem; (c) RMHD forced-run
  regression: IMEX scheme tracks lsrk54 statistically (energy within a few % at t~5);
  (d) local battery under the new schemes.

## P4b — 3D GDI (needs P2 + P3)

Extend the `"GDI"` recipe to dims==3, requiring z_spectral=True (raise otherwise).

- `linear_matrix_func`: full eq (5.3) with γ∥ = D∥kz² on the kz axis (eqpars gains
  D_par; gpar_fac floor retired in 3D or kept as an optional large-scale closure —
  decide at implementation, document either way). kz=0 plane is exactly the 2D model
  with γ∥=0: consistency check against P4a runs.
- Production scheme: CB-IMEX (the near-adiabatic regime is the point); IF-LSRK
  permitted only with documented dt ≲ 1/max(γ∥/k⊥²ρs²) — record this at the recipe.
- Validation, extending `tests/test_gdi_linear.py`: measured complex ω vs the 3D
  quartic (3.11) across a γ∥ scan — 2D limit (3.6), H-W limit (3.12, νin=0), nearly-
  adiabatic drift waves (3.9), stabilization boundary (3.15). Nonlinear: energy budget
  (3.18); science diagnostics from P4a upgraded with k⊥-break location vs eq (4.5),
  α measurement (4.7), and the 180°→90° cross-phase transition across k⊥c.
- Sizing note: single GPU; at 512³ fp32 one field register ≈ 0.54 GB — IMEX register
  count × nfields=2 fits comfortably on 40–80 GB parts.

## Review

Two mandatory adversarial-review rounds (Phase-2 A5 style: read-only, full diff,
CRITICAL/MAJOR/MINOR, runs the battery itself): after P1+P4a (gate for merging the
machinery) and after P2+P3+P4b. **The reviewer is a FRESH Fable agent — not the Fable
session that oversaw the coding agents** (the overseer shares the design's blind
spots; the review's value is an unshared context). Coding-agent output is never
self-reviewed. Specific traps to hand the reviewer:
bitwise gate on the diagonal backend, complex-sqrt/Taylor-branch correctness at fp32,
sign convention vs the notes' eq (5.3), the normalization sweep's completeness
(forcing power vs diagnostics consistency), z_spectral snapshot guarding, eqpars
migration of old params.json, `_replace`/donation discipline in new steppers.

## Sequencing

P1 → P4a → P2 → P3 → P4b, reviews after P4a and P4b. P2 and P3 are independent of
each other in code (P3 touches timestepping + the hook; P2 touches grids/physics) but
P3's tests are best run with P2 available — if parallelized, land P2 first.

## Status

- P1 (propagator machinery + eqpars): code complete 2026-08-01 (fp64 A/B bitwise for 2D
  and 3D forced RMHD, all schemes)
- P4a (2D GDI): code complete 2026-08-01 (physics/gdi.py + "GDI" registry entry; L
  cross-checked against eqs 2.8/2.9/2.11/3.7 in tests/test_gdi_linear.py, measured
  propagator-evolved growth rate vs. dispersion relation agree to 2.6% in a live nonlinear
  run in examples/gdi-2D.ipynb; energy-budget closure ~4e-8 relative at fp64; full battery
  green both precisions, 0 regressions)
- Review round 1 (P1+P4a): DONE 2026-08-01, verdict merge-with-notes (0 CRITICAL,
  2 MAJOR, 9 MINOR; reviewer independently reproduced the bitwise A/B, the L derivation
  — adjudicating that (5.3) agrees entry-by-entry, see docs/gdi_linear_matrix_note.tex —
  and the energy budget term-by-term). All findings fixed same day EXCEPT
  examples/gdi-2D.ipynb (MAJOR 2: loads a gitignored snapshot no cell generates; the
  k⊥⁻³ saturation-spectrum acceptance item also still open — the notebook's diss settings
  diverge rather than saturate on longer runs). Deferred to a user pass.
  RESOLVED 2026-08-01: MAJOR 2 fixed — examples/gdi_2d_run.py now holds the run
  configuration and a resumable/idempotent `make_data` (params.save + snapshot_io, checked
  for existing/partial data, wall-clock-bounded per call); the notebook's data cell calls it
  in a loop and needs no bundled input on a fresh clone; the stale 64² snapshot dir was
  deleted. The saturation-spectrum acceptance item is also delivered: retuned to a denser
  k-lattice (256², Lx=Ly=8π, same Ln/v0/nu_in/gpar_fac family, diss=2e-5/hyper=2) giving 88
  unstable dealiased lattice modes (vs. 1 before) and confirmed statistical saturation
  (fluctuating energy, no secular trend, ~9% drive/dissipation budget residual over the
  averaging window) by t~36-52; measured saturated spectral slopes ≈ -3.3 (|∇⊥φ|², close to
  the anticipated k⊥⁻³-ish range) and ≈ -6 (|N|², steeper). The false claim that
  tests/test_gdi_linear.py covers GDI checkpoint-restore was also removed from the
  notebook (it does not; restart-workflow.ipynb's pattern is what the new data cell uses).
  RETUNED 2026-08-02 (weak-drive): the strong-drive family above saturated at
  delta n/n0 >> 1 (N_rms~10-17, max|N|~21-37), well outside the model's perturbative
  ordering, and kept drifting rather than settling. examples/gdi_2d_run.py's eqpars are now
  Ln=84, v0=25, nu_in=0.05, gpar_fac=1, diss=5e-6, hyper=2 (same 256², Lx=Ly=8π box; mixing-
  length estimate max delta n/n ~ L_box/Ln targeting ~0.3), giving kc~2.44, gamma_max~0.134,
  376 unstable dealiased lattice modes. Measured saturation (t_avg>=130 of a t_end=170 run):
  N_rms~0.52, max|N|~1.3-1.8 (delta n/n restored to O(1) but still ~2-6x the 0.3 mixing-
  length target — reported honestly in the notebook, not re-tuned further since the target
  was a rough estimate and the linear-band criteria were the ones specified for iteration);
  energy-budget residual ~1% (vs ~9% before); spectral slopes ≈ -3.3 (|∇⊥φ|²) and ≈ -4.8
  (|N|²). The notebook also gained a permanent per-mode eigenvector-vs-time check (theory
  ratio from gdi._L_entries vs measured, both amplitude and phase, across 4 representative
  modes) and an adiabaticity diagnostic (time-averaged |N|²/|φ|² and |N-φ|²/|φ|² vs k,
  locating the γ∥-relaxation crossover empirically) plus an animated GIF of N/∇²φ.
  RETUNED AGAIN 2026-08-02 (γmax-calibration, final): the weak-drive family above undershot
  its own mixing-length-in-Ln target (max|N|~1.3-1.8 vs ~0.3). Direct empirical calibration
  (three probes at fixed kc=sqrt(v0/(nu_in·Ln)): γmax=0.134→max|N|~1.6 still drifting at
  t=170; γmax=0.060→max|N|~0.83 still creeping at t~240; γmax=0.017→max|N|~0.18, N_rms~0.06,
  steady t=350-600) found the saturation amplitude is set by γmax=sqrt(nu_in·v0/Ln) at fixed
  kc, not by Ln alone — the Ln-gradient drive term is negligible in this Pedersen-dominated
  family (drive_Ln/drive_pedersen ~1%), so the mixing-length-in-Ln picture doesn't apply. A
  local power-law interpolation (exponent ~1.2) through the three probes gives
  examples/gdi_2d_run.py's current eqpars: Ln=392, v0=25, nu_in=0.0106, gpar_fac=1,
  diss=5e-6, hyper=2 (same 256², Lx=Ly=8π box), giving kc~2.45 (unchanged), gamma_max
  ~0.0285 (lattice-measured), 344 unstable dealiased lattice modes, dt ceiling ~0.10.
  t_end raised to 550 (e-folding ~35 t.u. vs ~7.5 before) and snap_every to 5. Measured
  saturation (t_avg>=350 of the t_end=550 run, 32 snapshots): max|N|~0.35-0.52 (mean
  ~0.41), N_rms~0.11-0.13 — inside the 0.3-mixing-length target's [0.2,0.45] acceptance
  band on the first attempt, no retune needed. E(t) itself oscillates by ~20-25% around a
  flat mean (verified flat across t_avg_min=300-500, not a secular drift — a genuine
  physical difference from the earlier families: the phi/vorticity-dominated energy swings
  more than the density field the amplitude target is defined on); energy-budget residual
  ~0.6%; spectral slopes ≈ -3.5 (|∇⊥φ|²) and ≈ -4.9 (|N|²); adiabaticity crossover
  k~6.75 (peak |N|²/|φ|²~4.75), |N-φ|²/|φ|² down to ~0.07 (a few percent) by the dealias
  cutoff — closer to fully adiabatic there than either earlier family. The eigenvector-
  vs-time check still confirms the linear solve (all 4 modes track theory to ~30%/15° through
  t~70) but is visibly noisier than the fast weak-drive family's few-percent match, because
  make_data's snapshot spacing is set by whole nblock-step blocks (~17-20 t.u. early on,
  comparable to the ~10 t.u. mode-eigenvector relaxation time) and because 344
  comparably-slow-growing modes cross-couple nonlinearly before any one dominates —
  reported honestly in the notebook rather than glossed over.
- P2 (z_spectral): code complete 2026-08-02. `params.z_spectral` (dims==3 + size==1, "jax"
  rejected, recorded/compared in params.json); `grids.fft` now takes `params` and both
  transforms dispatch rfft2 <-> rfftn over (z,x,y); `kgrid.kz` + 3D dealias mask; RMHD's
  `linear_matrix` gains the `+-i*kz` off-diagonals (putzer2 with a real kz extent — the P1
  z-extent path exercised for the first time) plus an optional `eqpars['z_diss_k']`
  (`-z_diss_k*kz^4`, default off); `LinearTerm`/`halo_start`/the 1/dz+z_diss CFL terms are
  skipped; normalization sweep done in one go (`perp_reduce` /nz -> /nz^2, `parspec` a plain
  kz sum, `reconstruct_envelope` an exact `(A -+ iB)*nz/2` scatter onto the kz=+-2pi/Lz planes).
  Verified: fp64 bitwise A/B for z_spectral=False (2D and 3D forced RMHD, lsrk54/lsrk33/rk44,
  scan and unrolled — ALL BITWISE IDENTICAL); measured Alfven dispersion |omega-kz| <= 2.2e-15
  and damping error <= 1.6e-14 with and without dissipation; FD-z -> spectral convergence order
  3.88 (nz=16/32/64, same-nz pairs); forcing power / normalization scale / injection rate /
  energy match the real-z computation to ~1e-16 relative (Parseval); `examples/orzag-tang-3d-
  spectral-z.ipynb` written and executed. Full battery green at both precisions (131/116
  passed, 0 failed).
- P3 (CB-IMEX): not started
- P4b (3D GDI): not started
