# Audit follow-ups (GDI / z_spectral), started 2026-08-17

Items arising from Alfred's audit of the GDI equations and z_spectral machinery. Live;
more items will be appended as the audit proceeds.

## Execution notes

- Order: Items 1 -> 2 -> 3 strictly (they edit the same gdi.py lines); Items 4, 5, 6, 8
  are independent of those and of each other; Item 7 last (it sweeps files the other
  items touch).
- One commit per item; `make test` green (both precision sessions) before the next item
  starts. Item 4 additionally regenerates the recorded fp32 forcing-spinup reference
  (`tests/_gen_forcing_spinup_reference.py`) — a deliberate step, named in that commit's
  message.
- Comments in code: succinct, present tense, what the code does and non-obvious
  constraints only — no review/design history in source (that history lives in this
  file).
- Implementation by opus/sonnet subagents with oversight; adversarial review of the full
  diff at the end by a FRESH session with no implementation context, against this file.
  Review verdict and findings go in plans/AUDIT_REVIEW.md.

## Item 1 — `eig2` helper in `shared_physics`

The analytic 2x2 eigenvalue pattern `m = (L00+L11)/2`, `s2 = m^2 - det L`,
`lambda = m +- s` is currently written out in five places:

- `physics/gdi.py::_max_re_lambda::_plane_gmax` (numpy, setup time)
- `physics/gdi.py::kperp_break::max_re` (python/numpy scalars)
- `propagators.py::putzer2_precompute` (jnp, m and s2 only)
- `examples/gdi_3d_run.py::unstable_modes_report` (jnp)
- `examples/gdi_3d_run.py::theory_cross_phase` (python/numpy scalars)

### Task

Add to `physics/shared_physics.py`:

```python
def eig2_ms(L00, L01, L10, L11):
    # half-trace and discriminant of a 2x2: m = tr/2, s2 = m^2 - det.
    m = 0.5*(L00 + L11)
    return m, m*m - (L00*L11 - L01*L10)
```

Pure arithmetic, no `np`/`jnp` calls — dispatch-neutral, so the same function serves the
numpy setup-time sites, the jnp sites, and the scalar sites. Callers keep their own
`sqrt` (numpy vs jnp, and the `astype(complex)` guard stays at the call site — the
complex-ness contract differs per site and belongs there). Rewire all five sites.

### Constraints / gates

- `putzer2_precompute` feeds `kgrid.lin_m`/`lin_s2`, which enter every propagator call:
  the helper's expressions must match its current op order exactly (they do, as written
  above). Gate: `lin_m`/`lin_s2` bit-identical before/after on an RMHD z_spectral and a
  GDI 3D setup, plus `make test` green (fp64 and fp32 sessions).
- No behavior change anywhere; this is deduplication only.

## Item 2 — pin `_max_re_lambda`'s hand-rolled grid against `grids`

`gdi._max_re_lambda` deliberately rebuilds the k-grid and dealias region in numpy from
`params` (cacheability, params-only signature, trace-time python float, plane-at-a-time
memory — see the function comment). The cost is drift risk: it re-encodes the 2/3
ellipse, the kz cut, and the fftfreq scaling by hand, and nothing currently asserts they
agree with `grids.dealias_mask`/`setup_kgrids`. Two ways to disagree: someone changes
the rule in `grids` only, or a knife-edge boundary mode flips between grids' ftype-pinned
mask and gdi's float64 numpy rebuild at fp32.

### Task

1. Mechanical refactor inside `gdi.py`, no behavior change: extract the region/grid
   construction from `_max_re_lambda` into module-level helpers so a test can import
   them — `_perp_grids_np(params)` (kx, ky, ksq, inv_ksq, ky_deriv, perp_dealias) and
   `_kz_values_np(params)` (the (kz, iz) pairs with the `|iz| < nz/3` cut applied).
2. New test in `tests/test_gdi_linear.py` (existing bootstrap/`script_main` conventions):
   - `perp_dealias` equals `np.asarray(grids.dealias_mask(params))` for 2D params;
   - in 3D, the kept kz planes equal the True planes of the mask's kz axis, and the
     kz/kx/ky values equal `np.asarray(kgrid.kz/kx/ky)` from `setup_kgrids`;
   - parametrized over grids including nx,ny,nz not divisible by 3 (e.g. 64, 48, 20) and
     an anisotropic box; runs in both precision sessions so the fp32 knife-edge case is
     exercised.

### Gates

- Test green under `make test` (both sessions).
- `_max_re_lambda` results bit-identical before/after the refactor (same floats for the
  gdi_2d and gdi_3d example params).

## Item 3 — split diagnostics out of `physics/` into a `diagnostics/` package

`physics/rmhd.py` already contains only recipe-consumed functions and their private
helpers; `physics/gdi.py` broke that pattern by carrying six read-only observer
functions, and needs two local `from .. import diagnostics` workarounds because the
dependency runs diagnostics -> physics, not the other way. Top-level `diagnostics.py` is
nominally generic but RMHD-flavored (`energy` = (E_kin, E_mag) of (phi, psi); `perpspec`
bins 0.5|grad f|^2 per field — which is why GDI grew its own `energy_enstrophy` and
`perp_spectrum` with |N|^2).

### Boundary rule

`physics/<eq>.py` holds what the solver consumes: the `EquationRecipe` functions plus
their helpers (`_eqpars`, `_closure_terms`, `_L_entries`, and `_max_re_lambda` /
`_lin_dt_safety`, which `set_timestep` calls on the hot path). Everything read-only and
user-facing lives in `diagnostics/<eq>.py`. The registry stays a solver contract — no
diagnostics hook on `EquationRecipe`; diagnostics are plain imports, with
`diagnostics.<eqtype>` as the naming convention.

### Task

Convert `diagnostics.py` into a package:

- `diagnostics/core.py` — the genuinely shared machinery (`_binned`).
- `diagnostics/rmhd.py` — `energy`, `perpspec`, `parspec` (parspec is per-field
  grad-square like perpspec, so it is RMHD-flavored too, despite running on any state).
- `diagnostics/gdi.py` — `energy_enstrophy`, `energy_budget`, `perp_spectrum`,
  `cross_phase_spectrum`, `kperp_break`, `measure_alpha`, moved from `physics/gdi.py`;
  also adopt `theory_cross_phase` from `examples/gdi_3d_run.py`. Imports `_eqpars` /
  `_closure_terms` / `_L_entries` from `physics.gdi` (underscore names crossing a module
  boundary — accepted with a one-line note; renaming is not worth touching Item 1's call
  sites twice).
- `diagnostics/__init__.py` — re-export the current top-level names (`energy`,
  `perpspec`, `parspec`, `_binned`) so every existing `diagnostics.foo(...)` call site
  works unchanged. RMHD-side churn: zero.
- Delete the two local-import workarounds in the moved GDI functions (the cycle no
  longer exists in the correct direction).
- Update the GDI call sites — `tests/test_gdi_linear.py`, `examples/gdi_3d_run.py`,
  `examples/gdi-2D.ipynb` — and CLAUDE.md's architecture section.

### Sequencing / gates

- After Items 1-2: Item 1 rewires functions this item moves, Item 2's refactor edits
  `_max_re_lambda`'s surroundings — do them first so the same lines are not edited twice.
- Pure code movement, no behavior change. Gates: `make test` green (both sessions);
  `diagnostics.energy/perpspec/parspec` importable at the old paths; the gdi-2D notebook
  runs.

## Item 4 — bind `_quiescent_dt` into `rmhd.set_timestep` via `min()`

`_quiescent_dt` currently MIRRORS `set_timestep`'s velocity floor (eps = 0.1) instead of
sharing it — the bound the forcing scale cap relies on holds only while two sites stay
in sync by hand ("CHANGE BOTH SITES TOGETHER"). The floor form
`cfl_safety/max(v-terms, eps/dx, eps/dy)` is algebraically
`min(cfl_safety/max(v-terms), dt_q)` with `dt_q = cfl_safety/max(eps/dx, eps/dy)`, so
the binding can be structural — and then `dt <= _quiescent_dt(params)` holds by
construction for EVERY state on the adaptive path, not just from rest by derivation.

### Task

- `rmhd.set_timestep`: drop the `max_eps` terms from the max chain; return
  `jnp.minimum(cfl_safety/max_all, _quiescent_dt(params))`. The 3D
  finite-difference-z terms (1/dz, z_diss) stay in `max_all` — `_quiescent_dt` remains a
  valid (slack) upper bound there.
- Delete both mirror warnings; reword `_quiescent_dt`'s comment: it is now the enforced
  ceiling `set_timestep` itself applies, not a mirror.
- Small sanity test: from a zero-gradient state, `set_timestep == _quiescent_dt` in 2D
  and `<= _quiescent_dt` in 3D FD-z (where 1/dz can bind below it), both precision
  sessions.
- `gdi.set_timestep` keeps its inline floor (no forcing, no `_quiescent_dt` to bind);
  optionally hoist the eps = 0.1 constant somewhere shared so the value has one source.

### Cost (DECIDED 2026-08-17: accepted)

NOT bit-identical where the floor binds: the in-graph floor rounds
`f32(cfl/f32(eps/dx))`, `_quiescent_dt` rounds `f32(cfl*dx/eps)` — measured 1-ulp
differences on 3/24 plausible (nx, Lx) grids at fp32. Quiescent starts sit exactly in
that regime, so expect to regenerate the recorded fp32 forcing-spinup reference
(`tests/_gen_forcing_spinup_reference.py`), under review per the usual
reference-regen discipline. Verify rather than assume that developed-state references
survive (they should: the floor is inert there and the value is identical, but the graph
change can refuse — same fusion-sensitivity class as lsrk_scan).
The zero-churn alternative (shared eps constant + binding test, no restructure) was
considered and passed over in favor of the by-construction bound.

## Item 5 — rename `rmhd.LinearTerm` -> `FDLinearTerm`

The name is misleading since the propagator work: the k-local linear physics lives in
`linear_matrix`/propagators, and `LinearTerm` is exactly the NON-k-local
finite-difference remainder (FD-z Alfven stencil + z filter), skipped entirely under
z_spectral. Mechanical rename, ~10 live files: `physics/rmhd.py`,
`physics/__init__.py` (registry), the `config.py` warning string, 5 test files,
`docs/numerics.md`, CLAUDE.md. `plans/` mentions stay as written (historical record).
Gate: `make test` green; `grep -rn LinearTerm` hits only `plans/` and the new name.

## Item 6 — rename env var `RMHD_PRECISION` -> `TARANIS_PRECISION`

Leftover from the jax_rmhd era (the 2026-08-08 repo rename covered in-repo names, not the
env var). ~100 mentions outside `plans/`: `_precision.py` (the single read),
`Makefile`, `pyproject.toml`, `tests/conftest.py` + the test modules' docstrings,
`bench/`, `webgpu/gen_refvectors*.py`, slurm scripts, CLAUDE.md, docs.

### Task

- Hard rename (DECIDED 2026-08-17: no compatibility fallback — updating a recipe is a
  small ask for external users): `_precision.py` reads `TARANIS_PRECISION` only.
- One guard against the silent failure mode: a stale recipe setting only the old var
  would otherwise quietly run at the default 32. If `RMHD_PRECISION` is set and
  `TARANIS_PRECISION` is not, raise with a message naming the new var. Not a fallback —
  the old var is never honored, the user is made to change it.
- Everything in-repo switches to the new name (mechanical sed; leave `plans/` as
  historical record). Gate: `make test` green both sessions; setting only the old var
  raises the rename error.

## Item 7 — scrub personal-name mentions from repo files

The repo has been public since 2026-08-12; plan and doc files should not carry
private-correspondence namedrops. Grep inventory (2026-08-17) over the likely name set,
classified:

- KEEP — scholarly citations: Mallet et al. 2015 (CHI_PLAN), the
  Loureiro/Cowley/Dorland/Haines/Schekochihin tearing paper (TEARNL_PLAN), the
  Howes/Verniero/Klein 2016 movie and "Howes-style AW collision" (ISO_PLAN, old
  EXAMPLES_PLAN), and the paper references in `examples/*.ipynb` (verify each notebook
  hit is citation-class during the pass).
- SCRUB — suggestion/feedback credits: `plans-webgpu/ANISO_PLAN_2.md:14` ("Suggested
  by ..."), `plans-webgpu/README.md:84`, and any equivalents the full sweep turns up.
  Replace with neutral phrasing ("suggested by a collaborator", "an external user
  report").
- KEEP (DECIDED 2026-08-17) — ported-code provenance: `webgpu/README.md:966` and
  `plans-webgpu/FEEDBACK_2026-08-08.md:191` credit the rmhd-gpu source that
  auto_dissipation was ported from. Ported code is always credited; these mentions are
  attribution, not namedropping, and the same rule applies to any future port.
- Out of scope: git commit history/authorship.

Gate: a repo-wide grep for the name set returns only citation-class hits.

## Item 8 — remove `apply_exp`'s `coef` parameter

`coef` exists solely to reproduce the pre-P1 floating-point op order
`dt*exp(hdiss*dt/2)*k3` — the P1 bitwise-equivalence gate. P1 is executed and merged;
the code it preserved equivalence with no longer exists (DECIDED 2026-08-17: not worth
keeping).

### Task

- Drop `coef` from all three propagator classes' `apply_exp`
  (`IdentityPropagator.apply_exp` collapses to `return arr`); shrink the propagators.py
  header comment accordingly.
- The one consumer, `rk_advance` stage 3 (`timestepping.py:33`), becomes
  `dt*prop.apply_exp(k3, dt/2)`; delete its coef comment (line 32).
- Delete the coef-contract sub-test (`tests/test_linear_propagator.py:202-205`).

### Blast radius / gates

Bitwise change confined to rk44, which is non-production (its own header prefers LSRK)
and covered only by tolerance-based tests (`test_scheme_equivalence`, `test_imex`'s IF
comparison, `test_time_order`) — no recorded reference uses it. lsrk/imex never pass
`coef` and the default-None branch resolves at trace time, so production graphs are
identical before/after. Gate: `make test` green both sessions.

## Deferred (decided 2026-08-17, recorded so it is not relitigated)

- **Do not hoist `linear_matrix` into shared_physics.** The equation-agnostic layer
  already exists one level down (recipe seam -> `grids._attach_linear_operator` ->
  `propagators.linear_fields`/putzer2). The rmhd and gdi builders share ~4 lines
  (the stack, the 2D `[:, :, None]` axis); the entries, coefficients, and which axis
  needs its Nyquist zeroed are all per-equation.
- **Do not hoist `_max_re_lambda`'s sweep scaffolding yet.** It is equation-agnostic
  given an entries callback, but has exactly one consumer (RMHD has no linear growth and
  its waves use the exact IF propagator). Revisit when a second unstable equation set
  lands; the shared version should then report max |Im lambda| as well as max Re, so a
  wave-bearing L can bound omega*dt under the L-stable IMEX schemes (which damp
  oscillatory modes at omega*dt >~ 1), not just growth.
