# Full code review — 2026-07-31

Scope: the whole `jax_rmhd/` package (1560 lines), the test suite, docs and repo hygiene.
Findings are ordered by severity within each section.

Baseline at review time: `ruff check .` clean; `RMHD_PRECISION=64 pytest tests`
→ 87 passed, 5 skipped.

**Status — B1–B9 fixed 2026-07-31.** Suite now 90 passed / 5 skipped (fp64), 80 passed /
15 skipped (fp32), ruff clean. Sections C (bloat/duplication), D (comment style), E (test
gaps) and F (repo hygiene) are NOT done and remain open. B5 changed the meaning of
`forcing_power_elsasser` — see the note under that finding.

---

## 0. Verdict

The numerics are correct. I re-derived the integrating-factor RK4, the IF-LSRK stage
recursion, the RMHD term algebra, the CFL bound and the Hermitian symmetrization from
scratch and all of them check out (details in §1). I found **no numerical bug**.

Everything below is hardening, deduplication, and comment hygiene. The two things I'd
actually fix are **B1** (`dims` unvalidated — a typo silently runs 2D physics) and **C1**
(diagnostics duplicating `kgrid.yfac` and the reduction, against a stated CLAUDE.md
invariant).

---

## 1. Correctness — what I checked and confirmed

Recording these so you know what's been verified, not just what's broken.

| Item | Verdict |
|---|---|
| `rk_advance` IF-RK4 | Correct. Derived `v = e^{-Lτ}u`, RK4 on `v`; all four stage arguments (`f1`,`f2`,`f3`) and the `E k1 + 2E₂k2 + 2E₂k3 + k4` combination match exactly. |
| `lsrk_advance` integrating factor | Correct. `Σγᵢ = 1` for both tables, so the `N=0` limit reproduces `exp(hdiss·dt)` exactly. Stage times land on the published `cᵢ`. Scan and unrolled paths are algebraically identical (the `istage==0` branch is redundant only because `α₀=0`). |
| LSRK54 coefficients | Match Carpenter & Kennedy (1994), including the A5 previously fixed. |
| `NonlinearTerm` | `{ψ,J} − {φ,ω}` for vorticity, `−{φ,ψ}` for ψ; `−inv_ksq` conversion is right given `ω = −k²φ`. `bracket(a,b) = ∂ₓa ∂_yb − ∂_ya ∂ₓb`. ✓ |
| `LinearTerm` field swap | Correct: `∂ₜω = ∂_z J` ÷ `−k²` → `∂ₜφ = ∂_zψ`, and `∂ₜψ = ∂_zφ`. The `[df_dz[1], df_dz[0]]` stack is exactly that. |
| `d4f_dz4` sign | `−ν∂⁴/∂z⁴` → eigenvalue `−νk_z⁴`. Damping. ✓ |
| `set_timestep` | `max_vx_eff/dx` and `max_vy_eff/dy` are paired correctly despite the names reading backwards at a glance (`gphi[0]=∂ₓφ=v_y`). The `z_diss` floor is exactly right: the 5-point d⁴ stencil has max eigenvalue `16/dz⁴`, and `z_diss·(dz/2)⁴·16/dz⁴ = z_diss`. Nothing says so — worth one comment. |
| `_symmetrize_real_line` | Hermitian and variance-preserving. `/√2` is right; the self-conjugate `kx=0`/Nyquist entries come out real with the correct variance. |
| `ou_update` | Exact discrete O-U (`decay=e^{-dt/τ}`, `diffusion=√(1−decay²)`) for a unit-variance stationary process. |
| `dealias_mask` | Elliptical index-space mask with semi-axes `n/3` is a strict subset of the 2/3 box rule, so it is a valid (slightly conservative) dealiasing set. |
| `parspec` kz folding | Index arithmetic correct for even `nz`; `1/(nx·ny·nz)²` is the right normalization given the unnormalized z-FFT. |
| Spectra vs `energy` normalization | Consistent (`/nz·(nx·ny)²` everywhere). |
| `halo_exchange` | mpi4jax `sendrecv(dest=left, source=right) → recv_right` and the `ppermute` perm `(i, i−1)` describe the same shift. ✓ |
| Forcing scale lag | Smaller than CLAUDE.md claims. `_advance_forcing` computes the scale from `(fields_{n+1}, forcing_{n+1})`, and `forcing_{n+1}` is what the next step actually uses — so the scale is exact at sub-stage 0 and only lags *within* a step, not by a whole step. |

---

## 2. Correctness gaps and risks

### B1. `dims` is never validated — **highest severity**

`Parameters(dims=4)` and `Parameters(dims=2.5)` both construct fine and run a **2D**
simulation (the `if dims==3:` / `else:` in `config.py:61` sends everything but 3 down the
2D branch). `params.spatial_dimensions` then reports 4, `nz` is silently forced to 1, and
`params.json` records `dims: 4`. Verified empirically. A typo produces wrong physics with
no error anywhere. One line in `__init__` fixes it.

### B2. An empty forcing shell is a silent no-op

`fshell=(5,1)` gives `sum(fmask) == 0` (verified). The envelope stays zero, `P == 0`,
`safe_scale` returns `inf` clipped to `forcing_scale_max`, and the run proceeds as
*unforced* — no warning, and a plausible-looking `forcing_scale` in every snapshot. The
same happens for any shell that falls between grid modes at low resolution. `fshell`
ordering isn't checked either.

### B3. `diss` length is not checked against `nfields`

A length-3 `diss` survives `Parameters`, `setup_kgrids`, and the entire RHS (dissipation
is an integrating factor, not an RHS term — verified). It dies inside the stepper at
`exp(kgrid.hdiss*dt) * fields` with a broadcast error that never mentions `diss`. Same
for `forcing_power_elsasser` vs `n_ou`, and `hyper` (any float is accepted).
A scalar `diss` *does* work by broadcast, which is convenient and worth keeping — but
then that should be explicit rather than accidental.

### B4. `Lz=0.0` is the default while `dims=3` is legal

`dz=0` → `LinearTerm`'s `d4f_dz4/dz**4` is `0/0 = NaN`, `setup_kgrids`' z-envelopes divide
by zero, and `set_timestep`'s `1.0/params.dz` raises a bare Python `ZeroDivisionError`.
Three different failure modes, none of which name `Lz`.

### B5. `forcing_power_elsasser` means something different from `forcing_power`

- momentum: `forcing_power` **is** the total `dE/dt`.
- elsasser: total `dE/dt = (ε₊ + ε₋)/2`, because `E_tot = (E₊ + E₋)/2` with
  `E_± = ⟨|∇z_±|²⟩/2`.

So switching modes at "the same" power halves the injection rate. I derived this and it
matches the code — but the only place it is written down is a comment inside
`tests/test_forcing_norm_per_step.py:35-37`. It belongs in `config.py` and CLAUDE.md.

> **Fixed 2026-07-31 by changing the convention, not the docs.** `_forcing_scale_from`
> now carries a factor 2, so total dE/dt = `eps_plus + eps_minus` and the two modes share
> one unit. **Any existing elsasser run reproduced at the same numbers now injects twice
> the energy** — halve `forcing_power_elsasser` to reproduce old results.
>
> Every caller was migrated `(0.3, 0.3)` → `(0.15, 0.15)`, preserving a total injection
> rate of 0.3 so all quoted numbers in the example prose stay correct:
> `examples/forced-turbulence-2D`, `-3D`, `forcing-modes-2D`, `turbulence-spectra-analysis`,
> `restart-workflow`, `kaggle_forced_turbulence_256cubed`,
> `tests/forced_turbulence_64cubed.py`, `tests/load_forced_turbulence_snapshot.ipynb`.
> `forcing-modes-2D`'s derivation cell was updated too (`pip(z±,f±) = 2ε±`, total
> `ε⁺+ε⁻`, cross-helicity `dH_c/dt = ε⁺−ε⁻` — it loses its old factor ½).
> `tests/test_restart_resharding.py` and `test_backend_jax.py` keep their values: both
> compare runs against each other, so the absolute rate is irrelevant.
>
> **Notebook outputs are stale until re-run** — only source cells were edited.

### B6. `z_diff_order` / `z_diss_hyper` are accepted and ignored

Known (`rmhd.LinearTerm` TODO), but currently a user can set `z_diff_order=6`, get no
warning, and get 4th-order results. Until they're wired, `__init__` should reject
non-default values rather than store them.

### B7. `simulate` and `simulate_scan` have different snapshot semantics

`simulate_scan` unconditionally writes an extra "final state" snapshot after the loop
(`run.py:167-172`), which duplicates the last in-loop save whenever the loop happened to
end on one. `simulate` has no such final save. Not wrong, but the two entry points aren't
drop-in equivalents and nothing says so.

### B8. `mngr.all_steps()` in `run.py` contradicts a documented invariant

CLAUDE.md's checkpointing section: *"Enumerate saved steps with `get_saved_steps(snap_path)`,
not `mngr.all_steps()`."* Both `simulate` (line 209) and `simulate_scan` (line 146) do
`max(mngr.all_steps(), default=-1)+1`. In context this is probably safe — a manager
already exists and the result is bcast from rank 0 — but a reader following CLAUDE.md
will read it as a bug. Reconcile one or the other.

### B9. Hardcoded `eps = 0.1` velocity floor in `set_timestep`

`rmhd.py:22`. It caps dt at `cfl_safety·min(dx,dy)/0.1` for a quiescent field. No comment,
no parameter. It silently sets the maximum timestep of every low-amplitude run.

---

## 3. Bloat and duplication

### C1. `diagnostics.py` — roughly half of it is duplicated logic

- `perpspec` and `parspec` each rebuild `kgrid.yfac` by hand (3 lines each) — the array is
  already in `K_Grids`, precomputed. This is the exact thing CLAUDE.md's "all `perp_*`
  reductions share one normalization" rule exists to prevent.
- `perpspec`'s `0.5·ksq·|f|²·yfac` + z-sum + `allreduce_sum` + `/nz` + `/(nx·ny)²` is
  `perp_inner_product` reimplemented inline.
- `perpspec` and `parspec` duplicate each other's entire binning block (`kunit`, `kmax`,
  `dk`, `bin_edges`, two `histogram` calls, `bin_centers`).
- `energy()` imports `perp_inner_product_batch` inside the function body; there's no
  circular-import reason for that — it can be top-level.

### C2. `grids.dealias_mask` recomputes the k-grid

It rebuilds `kx`/`ky` (the same four lines that open `setup_kgrids`) and then recovers
`nx`/`ny` from array shapes when `params.nx`/`params.ny` are in hand. It's a separate
function because `run.initialize` calls it — but it should take `params` alone and share
the k-grid construction with `setup_kgrids`.

### C3. `simulate` / `simulate_scan` share ~35 duplicated lines

Snap-index init + bcast, the initial save, the per-iteration save + `wait_until_finished`,
the rank-0 prints, the timing print — identical in both. The functions differ only in the
loop body (`while_loop` to a target `t` vs a fixed-`nblock` scan). One driver taking a
`step_to_next_snapshot` callable would remove the duplication and B7 with it.

### C4. `block_of_steps` builds `set_timestep`/`rhs` twice

Once at the top of the `_use_cfl_blocks` branch (line 107-108), once inside the non-block
`stepping` closure (line 114-115). Both are trace-time so there's no runtime cost, just
two copies. It also returns a vestigial `, None` that every caller strips with `[0]`.

### C5. Four functions for two reductions

`_perp_reduce` / `_perp_reduce_batch` and `perp_inner_product` /
`perp_inner_product_batch` differ only in which axes are summed. One implementation with
an `axis` argument collapses all four into two, and the `_batch` suffix disappears from
the call sites.

### C6. Smaller items

- `LSRK_Scheme.nstages` is redundant with `len(alphas)` — one more thing to keep in sync.
- `rk_advance:10` — commented-out `#print("---COMPILING rk_advance---")`.
- `Parameters._init_args = {k:v for k,v in locals().items()}` depends on being the literal
  first statement of `__init__`. It is, and there's a partial note, but nothing enforces it.
- `from_snapshot` does `json.load(open(path))` — leaked file handle.
- `config.py` uses `print()` for the odd-`ny` adjustment and the 2D-multirank note;
  `warnings.warn` is the conventional channel and is filterable.

---

## 4. Comment style — against your stated preference

The dominant issue is **change-history comments**: comments that describe the diff rather
than the code.

**Task-ID tags with no in-repo referent** (`T7` is from `docs/PHASE2_PLAN.md`):
`physics/__init__.py:36,38`; `physics/rmhd.py:33,56`. Four instances, all meaningless
without the plan doc open.

**"Now / instead of / used to" phrasing:**

- `timestepping.py:128-132` — a 5-line block recording the *wrong* LSRK54 A5 value, when
  it was corrected, and how it was caught. A reader of the current code needs the
  Carpenter & Kennedy citation, not the old number.
- `rmhd.py:68` — "(Pp,Pm) **now** computed with a single stacked allreduce **instead of two**".
- `shared_physics.py:71-72` — "noise is **now** drawn only at ... **instead of** over the
  full grid and masking down".
- `snapshot_io.py:84-88` — describes a rescue path that no longer occurs for anything current.
- Test headers carry the same pattern throughout: `FINDING (2026-07-30)`, "This used to
  live inside `diagnostics.energy` itself", `docs/TESTING_PLAN.md Phase 3/5` in almost
  every file header.

**Load-bearing comments I would keep as-is:** the orbax barrier/deadlock explanations in
`snapshot_io.py`, the `ppermute` direction derivation in `comms.py`, the
`_local_device_ids` CUDA_VISIBLE_DEVICES note, the `donate_argnums` warnings. These
encode non-obvious external behaviour and earn their length.

**Comment density.** `config.py`, `comms.py` and `snapshot_io.py` are roughly 30–40%
comment. By contrast `physics/rmhd.py` and `physics/shared_physics.py` are the closest to
"the code speaks for itself" and read well — they're the model for the rest.

**One thing to decide, not just trim:** `physics/__init__.py:16-18` says the halo-start
machinery "makes no difference for any backend" and is kept "for future tweaking". That
honesty is good, but it's paying for `_HALO_START_BACKENDS`, `_halo_start_enabled`,
`halo_start_func`, `rmhd.halo_start`, the `halo` threading through `construct_rhs`, and a
**5th positional argument on every term function every equation set will ever write** —
for no measured benefit. Worth an explicit keep/drop decision.

---

## 5. Test suite

Genuinely good. 87 passing at fp64, dual-mode (pytest + `mpirun python tests/…`), clean
marker/skip logic, `_rmhd_testing.py` encodes the real footguns (donation, identity
hashing, overshoot). No complaints about structure.

Gaps worth closing:

1. **No temporal order-of-accuracy test.** `test_scheme_equivalence` checks absolute error
   against fixed thresholds and that rk44/lsrk54 beat lsrk33. A `fit_order` sweep in `dt`
   — the helper already exists and is used for the z-stencil order in `test_advection` —
   would pin 3 vs 4 directly and catch any future coefficient typo the way the A5 bug was
   caught only by luck of threshold.
2. **No accuracy test with dissipation on.** Every accuracy test runs `diss=0`. The
   stage-wise integrating factor `exp(hdiss·dt·γ)` is the least standard piece of numerics
   in the code and it is the one piece with no order verification.
3. Nothing covers the `Parameters` validation gaps in B1–B4 — because there's no
   validation to test yet.

---

## 6. Repo hygiene

- **Notebooks are committed with outputs.** `examples/*.ipynb` is 6.5 MB of the repo's
  7.6 MB of tracked bytes; four notebooks are ~1 MB each, all base64 PNGs. Every re-run
  rewrites them into the diff. `nbstripout` (or a `.gitattributes` filter) cuts the repo
  by ~85% and makes example diffs readable.
- **`docs/` is 2513 lines against 1560 lines of package code**, and most of it is completed
  process artifacts: `PERFORMANCE_PLAN`, `PHASE2_PLAN`, `PHASE3_PLAN`, `HALO_WIDTH_PLAN`,
  `EXAMPLES_PLAN`, `TESTING_PLAN`, `REVIEW_FIXES` all describe work that is done.
  `REVIEW_FIXES.md` is a changelog of fixes already in the code.
  Still live and worth keeping: `checkpointing.md`, `RUNNING_TESTS.md`, `SAVIO_*_SETUP.md`,
  `GDI_PLAN.md`, and the appendix material in `PHASE3_RESULTS.md` that CLAUDE.md cites.
  Suggestion: fold the durable findings into CLAUDE.md, move the rest to `docs/archive/`
  or delete.
- **`rundir/` (5.3 MB) and `a5k/` are untracked but not in `.gitignore`** — they clutter
  every `git status`. `data/` and `examples/data` are correctly ignored.
- `pyproject.toml`'s ruff comment explaining the pinned `select=["F"]` scope is a good
  example of a comment that earns its length — keep it.

---

## 7. If you only do five things

1. Validate `dims`, `diss` length, `fshell` ordering, and `Lz>0` when `dims==3` (B1–B4).
2. Make `diagnostics` use `kgrid.yfac` and `perp_inner_product` instead of reimplementing
   them; merge the two binning blocks (C1).
3. Document the elsasser power convention in `config.py` (B5).
4. Strip the `T7:` tags and the "now / instead of / used to" comments (§4).
5. Add a `fit_order`-in-`dt` convergence test, with dissipation on (§5).
