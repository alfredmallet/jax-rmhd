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
- P3 (CB-IMEX): code complete 2026-08-02. Four new `_scheme_registry` entries from Cavaglieri
  & Bewley, JCP 286:172-193 (2015) (PDF: http://robotics.ucsd.edu/pubs/CB15.pdf), all with an
  L-stable, stiffly accurate (a^IM_{s,i}=b_i) implicit part: `imexcb2` (IMEXRKCB2, eq. 24 —
  2nd order, [2R], 3 stages, SSP explicit part), `imexcb3e` (IMEXRKCB3e, eq. 30 — 3rd order,
  [2R], 4 stages, ERK accuracy maximized, delta=1/24 so its ERK stability region is RK4's;
  all coefficients exactly rational, hence the default recommendation), `imexcb3c`
  (IMEXRKCB3c, eq. 28a — same class, ERK negative-real-axis stability maximized, delta=1/54,
  SSP) and `imexcb3f` (IMEXRKCB3f, eq. 32c — 3rd order, [3R], the only stage-order-2 member).
  Two new steppers `imex2r_advance` / `imex3r_advance` implement the paper's three-register
  [2R] (eq. 19) and four-register [3R] (eq. 21) algorithms; the paper's two-register [2R]
  variant is deliberately NOT used (its own footnote 4 rules it out for spectral methods: it
  buys the register back with two extra nonlinear evaluations, i.e. FFT batches, per stage).
  UPDATE 2026-08-02: the [2R] stepper now honors `params.lsrk_scan` (scan default,
  user-requested — unrolled LSRK was slow on their CPU): stages 1..s-1 are uniform and the
  special first stage sits before the loop, so unlike `_lsrk_scan_stages` no `lax.cond` is
  needed; the never-formed z_1 of cb2/3c/3e enters the scan as zeros whose k=1 coefficient
  is exactly 0.0. Scan carry = exactly the 3 registers. `imexcb3f`'s [3R] stepper stays
  unrolled only (its y-update needs lookahead coefficients a_(k+1,k-1) and skips the last
  stage — a scan would need a cond for one scheme). test_imex.py gained
  test_imex2r_scan_vs_unrolled (tolerance assertion 1e-13 fp64 / 1e-5 fp32 — measured
  bitwise on the dev sandbox, NOT promised, same fusion caveat as lsrk_scan — plus a
  bitwise check that cb3f ignores the knob); with lsrk_scan defaulting True the whole
  IMEX test set (incl. the 1e-12 dense-tableau match) re-ran through the scan path at
  both precisions. Battery after: fp64 140 / fp32 123, 0 fail. The propagator hook gained one method,
  `apply_L(arr)` (diagonal/putzer2/identity), so the steppers still never touch
  `kgrid.lin_L`; each implicit stage is exactly one `solve_shifted(., a_ii*dt)`. b_1 = 0 and
  a^IM_{2,1} = 0 in cb2/3c/3e, so their first implicit derivative is skipped entirely
  (s-1 solves). ALL of L is implicit — no exponential in this path — so an L-stable solve
  damps oscillatory linear terms at |omega|*dt >~ 1 and these schemes must never be pointed
  at a wave-dominated L (z_spectral's +-i*kz) at large dt; that is recorded at the code.
  `rk_advance`/`lsrk_advance` and the LSRK coefficient tables are byte-for-byte untouched
  (the only diff line in the pre-existing code is a trailing comma on the lsrk54 entry).
  Verified in `tests/test_imex.py` (8 tests): every order condition of both tableaux plus the
  coupling conditions and the [2R]/[3R] structure re-derived from the stored coefficients
  (residuals <= 1e-13 at float64; the published 3c/3f decimals reproduce exactly to ~1e-24 in
  Fraction arithmetic); L-stability sampled over the left half plane to |z|=1e6 with the
  z -> -inf tail evaluated in exact rationals (a float evaluation bottoms out at ~1e-7 for
  3f); the low-storage steppers reproduce a dense-tableau IMEX integrator to 1e-12 relative
  after 6 steps (this is what validates the register recurrences, the [3R] one especially);
  measured convergence order on a manufactured NONLINEAR problem 2.00/2.99/2.97/2.98;
  forced-2D RMHD total energy at t=5 within 2% of lsrk54 (4.145/4.060/4.047/4.046 vs 4.061);
  z_spectral 3D runs finite and converges to the IF result at slope ~3.0.
  **Stiff quasi-static (the motivating item)**: at fixed gamma*dt = 10, taking gamma 200 ->
  2000, |u-g|/|g| falls exactly 10x for every IMEX scheme (cb2 1.33e-1 -> 1.31e-2, cb3e
  6.74e-3 -> 6.28e-4, cb3c 9.44e-2 -> 9.34e-3, cb3f 1.01e-1 -> 1.00e-2), i.e. u = g +
  O(1/gamma) as required, while the IF schemes' error is flat in gamma (lsrk33 5.71e-1 ->
  5.60e-1, lsrk54 1.08e-1 -> 1.16e-1, rk44 7.09e-1 -> 7.11e-1) — they miss the quasi-static
  balance outright. NB the IMEX error is O(dt) at fixed gamma*dt, not O(dt^order): the
  classic stage-order-one order reduction. IMEXRKCB3f's stage order 2 does NOT help there,
  because it applies to the implicit part only and the drive lives in the explicit part;
  cb3e simply has the smallest stiff error constant of the four (~15x below cb3c/cb3f).
  Register audit: the [2R] stepper holds 3 field-sized values live (x, y, z) and the [3R]
  stepper 4 (x, y, z_im, z_ex) — no per-stage k_1..k_s anywhere. Corroborated by XLA
  memory_analysis on a 256^2 step with a minimal non-fusable RHS: peak temporaries of 3.0
  (cb2), 4.0 (cb3e/3c/3f) field-register equivalents against 4.5 for lsrk33/lsrk54 and 5.0
  for rk44. (XLA does not always reach the algorithm's theoretical minimum — cb3e's dataflow
  is 3 but it schedules 4.) Full battery green at both precisions: fp64 139 passed / 5
  skipped, fp32 122 passed / 22 skipped, 0 failed, exactly +8 and +6 over the pre-P3
  131/116 (the 2 extra fp32 skips are this file's two fp64-marked runs).
- P4b (3D GDI): code complete 2026-08-02. `physics/gdi.py` extended to dims==3 (requires
  `z_spectral=True`, raised as `NotImplementedError` otherwise; dims==2 code paths are
  untouched — verified bit-for-bit, see below). `linear_matrix`/`_max_re_lambda`/
  `energy_budget` now share one helper (`_closure_terms`) computing gamma_par(k) =
  gpar_fac*nu_in*k_perp^2 [P4a 2D floor, eq 4.3] + D_par*kz^2 [P4b real closure, eqs
  3.5-3.7] ADDITIVELY. Decision: the 2D floor is NOT retired in 3D but its default flips to
  OFF (gpar_fac=0) — with a real kz axis resolved there is no sub-grid closure gap to patch,
  so the honest 3D default is the real D_par*kz^2 term alone; gpar_fac stays available as an
  optional supplement, unlocking the REQUIRED consistency property for free: at kz=0 with
  gpar_fac=0 (the 3D default) gamma_par_total==0 identically, so dims==3's L collapses to
  EXACTLY the dims==2 L at gpar_fac=0 (and, since the construction is additive, to the 2D L
  at ANY gpar_fac value if that value is matched on both sides — verified for gpar_fac in
  (0.0, 1.0)). eqpars gains `D_par` (required in 3D, rejected as an unknown key in 2D so a
  stray D_par silently doing nothing is impossible). [phi,N]/[phi,phi] divide gamma_par by
  k_perp^2 as before; for the 2D floor this ratio is exactly k-independent (bit-for-bit
  P4a), but D_par*kz^2/k_perp^2 is NOT k_perp-independent (diverges at k_perp->0, kz!=0) so
  it is masked with inv_ksq, the same zero-mode convention used everywhere else in the file.
  gamma_par*kz^2 is EVEN in kz so (unlike rmhd's +-i*kz) no kz-Nyquist fix was needed;
  `propagators._check_hermitian_compatible` (which mirrors both kx and kz under
  z_spectral) accepted every L built in testing with no special-casing. Production scheme:
  CB-IMEX (imexcb3e default, per P3) — recorded at the recipe (module docstring) alongside
  the documented IF-LSRK fixed-dt fallback bound; GDI's L has no +-i*kz wave term (pure
  growth/damping spectrum) so there is no wave-damping caveat against IMEX here, unlike
  z_spectral RMHD. Sizing note recorded verbatim in the recipe (512^3 fp32: ~0.54 GB/field
  register x up to 4 IMEX registers x nfields=2 ~= 4.3 GB, fits a single 40-80 GB GPU).
  Diagnostics: new `gdi.kperp_break(params, kz)` (numeric bisection on max Re(lambda(L)),
  not the paper's closed-form eq 4.5 — see below for why) and `gdi.measure_alpha(state,
  kgrid, params, kz_index, modes)` (mode-by-mode alpha = 1 + gamma_par/(nu_in*k_perp^2),
  review-round-1 NOTE preserved: alpha = 1 + gpar_fac, NOT gpar_fac); `cross_phase_spectrum`
  gained a `kz_index` argument (default 0, 2D behavior unchanged) to select a kz plane.
  L derivation route and cross-check: derived directly from eqs (3.5)-(3.6) (the *general*
  3D PDEs, of which P4a's (5.4)-(5.5) is the gamma_par->closure-floor special case) with
  gamma_par_total substituted for the bare D_par*d^2/dz^2 term — i.e. the SAME algebra as
  docs/gdi_linear_matrix_note.tex's P4a derivation, just not fixing gamma_par's k-dependence
  up front. Cross-checked three independent ways: (1) the exact quadratic (3.7) (same
  quadratic as P4a's, gamma_par now general) via `np.roots` vs L's putzer2 eigenvalues,
  scanned over gamma_par (kz values) at both a hand-built single mode AND the real
  `setup_kgrids`->`linear_matrix` pipeline (exercising the kz-broadcasting code, not just
  the algebra); (2) the literal quartic (3.11) and its (3.10) omega_R relation, independently
  transcribed from the PDF and evaluated (not solved) at the exact eigenvalues' Re/Im parts;
  (3) the named asymptotic limits eq (3.9) (nearly-adiabatic, gamma_par->infinity), eq (3.12)
  (Hasegawa-Wakatani, nu_in=0), and eq (3.15) (stabilization boundary sign flip) --
  all three limit formulas were FIRST derived independently by hand from (3.11)/(3.7) (not
  copied from the boxed PDF equations) and numerically verified (relative error -> 0 as the
  asymptotic parameter grows, e.g. eq (3.12) relerr 3.1e-3 -> 8e-6 -> 8e-8 as gamma_par
  20 -> 100 -> 1000) before being written into the test. The paper's OWN closed-form eq
  (4.5) ("k_perp^c break") was NOT trusted for a pytest assertion: substituting its boxed
  self-referential form back into itself gives a DIFFERENT power law ((k_perp^c)^3 ~
  ky^2*v0/(rho_s*Ln)) than solving the (4.4) balance it's allegedly derived from directly
  ((k_perp^c)^4 ~ ky^2*v0/(nu_in*Ln*rho_s^2)) -- a genuine unresolved transcription
  ambiguity (OCR/PDF subscript risk, the exact class of error CLAUDE.md's P4a history
  lesson warns about), so `kperp_break` bisects the ACTUAL L numerically instead (exact,
  no transcription risk) and is used as a diagnostic/demonstration tool, not asserted
  against eq (4.5)'s boxed algebra in any test.
  Validation (`tests/test_gdi_linear.py`, all PASS both precisions): `test_kz0_plane_matches_2D_model`
  (REQUIRED consistency check, both gpar_fac=0 and 1, tol 1e-12 fp64/1e-5 fp32);
  `test_3D_grid_dispersion_matches_quadratic_and_quartic_scan` (8 kz grid planes at a fixed
  mode, real setup_kgrids pipeline, quadratic rtol 1e-9 fp64 and quartic (3.11)/(3.10)
  residual both satisfied at every plane); `test_HW_limit_matches_eq312` (gamma_par
  20/100/1000, relerr 3.1e-3/2.0e-4/8e-6); `test_nearly_adiabatic_matches_eq39` (gamma_par
  1e4/1e5, relerr 3.2e-4/3.2e-5); `test_stabilization_boundary_matches_eq315` (6 gamma_par
  values straddling the predicted boundary near 0.2-0.5, sign of max Re(lambda) matches
  eq (3.15)'s prediction at every one); `test_3D_eqpars_validation` (D_par required/rejected
  correctly, gpar_fac defaults 0.0/1.0 in 3D/2D); `test_energy_budget_closure_nonlinear_3D`
  (fp64-only, imexcb3e, 8x8x4 grid, measured centered-difference dE/dt matches
  `gdi.energy_budget`'s total to rel=1e-4 tolerance, exercising the kz-dependent gamma_par
  sink term for the first time in a live nonlinear run). Every pre-existing 2D test in the
  file still passes UNCHANGED (verified the 2D `linear_matrix`/`_L_entries` arithmetic is
  bit-for-bit identical to pre-P4b by construction — kz=None short-circuits every new term).
  Science-diagnostic demonstration: `examples/gdi_3d_run.py` (gdi_2d_run.py's resumable
  `make_data` pattern; 64x64x16 grid, Lx=Ly=8*pi, Lz=2*pi, D_par-only eqpars, imexcb3e, fp32)
  plus a `report()` function printing the full diagnostic suite -- delivered as "cells
  appended to a small standalone driver" (the plan's explicit alternative to a notebook;
  no working Jupyter-kernel execution path was available in this sandbox session, unlike
  the prior P4a session's, so a live .ipynb was not attempted rather than risk an
  unverifiable one). Executed end to end (`python -c "from gdi_3d_run import report;
  report()"`, full output in the P4b agent report): `kperp_break` shrinks monotonically
  with kz (4.09 -> 3.96 -> 3.57 -> 2.74 -> unstable-nowhere by kz=4), confirming higher-kz
  planes are more strongly stabilized, as expected. A genuinely new 3D finding, reported
  honestly rather than glossed over: because gamma_par=D_par*kz^2 does NOT grow with
  k_perp (unlike the 2D floor's gamma_par=gpar_fac*nu_in*k_perp^2), the N-phi adiabaticity
  crossover at fixed nonzero kz runs OPPOSITE to the P4a 2D-floor picture -- adiabatic
  (small phase) at SMALL k_perp where gamma_par/k_perp^2 is large, GDI-like (phase->90deg)
  at LARGE k_perp where it vanishes -- and is a genuinely DIFFERENT scale from
  `kperp_break`'s marginal-stability crossing, not the same k_perp^c. The theoretical
  eigenvector cross-phase (computed directly from L, no run needed) shows this cleanly
  (e.g. kz=2: 26deg at k_perp=0.25 rising to 85.5deg at k_perp=7); the actual evolved run's
  `cross_phase_spectrum` at the matching kz plane reproduces this shape closely (24.6deg ->
  87.8deg across the same k range) -- a good quantitative match despite the run being well
  into the nonlinear regime by the demonstration snapshot (t~25.6). `measure_alpha` on the
  same run state matches `alpha_theory`'s ORDER OF MAGNITUDE and qualitative k-dependence at
  most probed modes but not tightly (occasional sign flips/large deviations on
  low-amplitude modes) -- attributed honestly to nonlinear mode-coupling contamination, not
  re-tuned further given the sandbox time budget. HONESTY on the underlying run itself
  (unlike P4a's, this one did NOT go through a multi-round saturation-amplitude retune):
  energy grows smoothly through >4 orders of magnitude from t=0 to t~45 with no clear
  plateau reached (E: 2.1e-4 -> 4.0 at t=30 -> 20.9 at t=45), and real-space max|N| reaches
  ~10-17 by t~36-45 -- well outside the model's own delta-n/n<<1 perturbative ordering
  (same failure mode P4a's first strong-drive family hit, see this Status section's P4a
  entry). The diagnostics are demonstrated on this genuinely-reached, genuinely-nonlinear
  but NOT verified-saturated state, exactly as plans/GDI_PLAN.md's escape hatch allows,
  rather than claiming saturation that was not confirmed. A calibrated, saturating 3D
  family (analogous to gdi_2d_run.py's several retune rounds) is left for a follow-up user
  pass.
  Battery: fp64 147 passed / 5 skipped (was 140/5), fp32 130 passed / 22 skipped (was
  123/22), 0 failed at either precision -- exactly +7 at both (the 7 new/renamed P4b test
  functions in tests/test_gdi_linear.py; one, the fp64-gated nonlinear energy test, prints
  [SKIP] internally and returns rather than using pytest.skip, so it still counts as a pass
  under pytest, matching the existing test_energy_budget_closure_nonlinear convention).
  `ruff check .` clean. Files changed: `jax_rmhd/physics/gdi.py` (extended, no other
  jax_rmhd/ files touched -- the propagator/kgrid/dealias/normalization machinery needed
  for 3D GDI was already generic from P1-P3, exactly as the plan anticipated);
  `tests/test_gdi_linear.py` (extended); new `examples/gdi_3d_run.py`.
- Review round 2 (P2+P3+P4b): DONE 2026-08-03, fresh Fable agent (not the overseer),
  verdict **merge-with-notes** (0 CRITICAL, 2 MAJOR, 4 MINOR). Independently reproduced:
  the full battery and ruff; every CB-IMEX tableau entry against the paper's eqs
  (24)/(28a)/(30)/(32c) -- adjudicating (28a)'s aIM_43/aEX_43 label as a paper typo via
  the exact identity b2+ex43=1 (shared denominator 2334033219546); order/coupling/
  stiff-accuracy/L-stability from the STORED tables (residuals <=2e-16, R(-1e12)~1e-12);
  both register recurrences line-by-line vs the paper's eqs (19)/(21) including the scan
  variant's z_1=zeros/exact-0.0-coefficient argument; the GDI (3.7)/(3.9)-(3.12)/(3.15)/
  (3.18)/(4.7) algebra from its own PDF transcription; the P2 normalization-sweep
  completeness by grep+tests; kperp_break's example sequence exactly. Pre-adjudicated
  non-findings (eq 4.5 non-use -- user accepted; unsaturated gdi_3d_run) respected.
  ALL FINDINGS FIXED same day (overseer, user-approved M1 direction):
  - M1: gdi.set_timestep's dt ceiling applied max|Re lambda| (stiff DAMPED branch
    included) for every scheme, throttling imexcb3e 67x on the shipped 3D config and
    ~1e5x at the 512^3 sizing -- negating P3's payoff. FIXED: growth-rate-only ceiling
    (max(Re lambda, 0) over the dealiased region; 0 = stable = no ceiling, CFL binds).
    Measured post-fix: shipped 3D config ceiling 1.79 vs CFL 1.57 (CFL binds). NOTE the
    2D behavior change: the weak-drive family's old 0.10 ceiling is gone (damped branch
    there is mild, gamma*dt<1 at CFL dt, so lsrk54 accuracy is unaffected); IF schemes on
    3D GDI now have NO automatic stiff-dt protection -- hand-set dt <~ 1/max(gamma_par),
    documented at _max_re_lambda and the module docstring.
  - M2: gdi.energy_budget's -gamma_par*|N-phi|^2 overcounted at k_perp=0, kz!=0 (L's
    phi-row gamma_par is inv_ksq-masked; the budget wasn't) -- 1.1e-2 closure error if
    such modes are seeded. FIXED with the exact per-row form (-gamma_par*Re(N*(N-phi)) +
    ksq*gpar_ratio*Re(phi*(N-phi))); reviewer's own repro now closes at 1.1e-8.
  - m1: stale gdi-3D.ipynb reference removed from gdi_3d_run.py; driver added to
    examples/README.md. m2: _max_re_lambda now computes one kz plane at a time -- 512^3
    evaluates in ~2 s (previously OOM-killed at the docstring's own sizing). m4:
    Putzer2Propagator.solve_shifted det(I-aL)=0 pole documented (mirrors the apply_exp
    overflow note). m3 (silent gpar_fac 1->0 default flip in 3D): left as-is -- D_par is
    mandatory in 3D so an eqpars edit is forced anyway, and the decision is documented;
    accepted residual risk.
  Post-fix battery: fp64 147 / 5 skipped, fp32 130 / 22 skipped, 0 failed; ruff clean.
