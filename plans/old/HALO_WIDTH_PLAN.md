# Plan: parameterize the z-halo width

Goal: `comms.halo_exchange` currently hardcodes a two-wide halo (the requirement of the
4th-order centered first derivative and the 5-point d4/dz4 hyperdissipation stencil).
Allow an arbitrary width to be passed so future equation sets / higher-order z-stencils
(`z_diff_order != 4`) can request wider halos. This change is **plumbing only**: no new
stencils are implemented, `z_diff_order`/`z_diss_hyper` remain unread by `LinearTerm`
(the existing `#TODO` there stands), and the default behavior must be **bitwise identical**
to today.

Suggested agent: Sonnet. The change is small and mechanical; the subtleties are all listed
below.

## Constraints (from CLAUDE.md — read it first)

- `width` must be a **static Python int** everywhere (it selects slice extents at trace
  time). Never a traced value, never under `lax.cond`.
- Nothing outside `taranis/comms.py` imports mpi4jax. Keep it that way.
- `Parameters` is all-static aux_data; plain-Python branching on its attributes is the
  house style.
- Both comm backends (`"mpi4jax"` sendrecv, `"jax"` ppermute inside shard_map) must get
  the same treatment — physics code stays backend-agnostic.
- Do not build states positionally; irrelevant here but do not "fix" anything en route.

## Changes

### 1. `comms.halo_exchange(f, params, width=2)`

- Add keyword arg `width` (int, default 2 → today's behavior).
- mpi4jax branch: `f[:, :width, :, :]` / `f[:, -width:, :, :]` in place of the literal 2s.
- jax branch: same slices in the two `ppermute` calls.
- Guard: raise `ValueError` if `width < 1` or `width > nz_local` (`nz_local =
  f.shape[1]`). The second check is load-bearing: with `width > nz_local` the slice
  silently returns fewer planes than requested and the exchange is wrong, not an error.
  (A width-> nz_local exchange would need multi-hop neighbor comms — out of scope, error
  clearly instead.)
- Update the docstring/comment ("two-wide" → parameterized).

### 2. `rmhd.halo_start(state, kgrid, params)`

- Pass the width explicitly: `comms.halo_exchange(state.fields, params, width=2)`.
- Add a one-line comment: width must match what the consuming term's `z_derivatives`
  stencil expects (RMHD: 4th-order centered + 5-point d4 ⇒ 2). The pre-issued halo and
  the fallback exchange inside `z_derivatives` MUST use the same width — this is the one
  coupling in the design; say so at both sites.

### 3. `shared_physics.z_derivatives(f, params, halo=None)`

Two acceptable options; pick (a) unless it gets ugly:

(a) Derive the offsets from the received slab instead of hardcoding 2/4:
    `w = recv_left.shape[1]` (static under jit), assert `w >= 2`, then index the padded
    array relative to `w` (`p2 = f_padded[:, w+2:, ...]` etc. — careful: the current
    literals `4:`, `3:-1`, `2:-2`, `1:-3`, `:-4` are all expressed in "pad = 2" units;
    rewrite them as `w±k` with the stencil half-width k=2 kept literal, since the stencil
    itself is still fixed 4th order). This makes `z_derivatives` tolerant of a caller that
    pre-issued a wider halo than it needs.

(b) Keep the literals and just assert `recv_left.shape[1] == 2` with a clear message.

Either way the no-halo fallback becomes `comms.halo_exchange(f, params, width=2)`.

### 4. Optional (ask the user before doing it): `params.halo_width`

A derived attribute on `Parameters` (dims==3 only, like `dz`), e.g.
`self.halo_width = z_diff_order // 2` — would give future stencils a single source of
truth and let `halo_start`/`z_derivatives` read the width instead of hardcoding 2. BUT
`z_diff_order` is currently decorative, and deriving a width from a knob the stencils
ignore invites a silent mismatch (user sets `z_diff_order=6`, halos widen, stencil stays
4th order). Recommendation: **skip this for now**; keep explicit `width=2` at the two RMHD
call sites and leave the derivation to whoever actually implements variable-order
stencils. Flag it in the PR description instead.

## Explicitly out of scope

- Implementing `z_diff_order != 4` stencils or wiring `z_diss_hyper`.
- Multi-hop halos (`width > nz_local`).
- Any change to `EquationRecipe` / `construct_rhs` signatures — the halo object stays an
  opaque `(recv_left, recv_right)` pair threaded through unchanged.
- Halo-related changes to checkpointing, forcing, diagnostics (none needed: halos are
  transient, never serialized).

## Verification

The sandbox has no MPI — do not try to run MPI tests locally; correctness runs happen on
Savio (see repo SLURM scripts). What the agent CAN and MUST do:

1. Static review: confirm every call site of `halo_exchange` (grep: `rmhd.halo_start`,
   `shared_physics.z_derivatives`) passes/receives a consistent width; confirm no
   remaining literal-2 halo slicing anywhere.
2. Bitwise-identity argument: with `width=2` defaults, the traced computation is
   syntactically identical to the current code (same slices, same concatenate, same
   indices). State this explicitly in the summary, per-file.
3. Add a small width-consistency check to an existing standalone test script (e.g. extend
   `tests/test_advection.py` or add `tests/test_halo_width.py`): run a few steps of a 3D
   problem, once with the default path and once passing `width=2` explicitly through
   `halo_start`/`z_derivatives`, assert bitwise-equal fields. Structure it like the other
   tests (plain script, prints PASS/FAIL, runs under `mpirun -n N`); the user will run it
   on Savio with `-n 1`, `-n 2`, `-n 4`.
4. If option 3(a) was taken: also exercise a *wider-than-needed* halo (width=3) in the
   same script and check `z_derivatives` output is bitwise-equal to width=2 (the extra
   plane must be sliced away exactly). This is the only new behavior with a testable
   consequence today.
5. Verify the `width > nz_local` guard raises (single-rank: nz small, width large).

## Files touched

- `taranis/comms.py` (halo_exchange signature + both backends + guard)
- `taranis/physics/shared_physics.py` (z_derivatives offsets / assert + fallback call)
- `taranis/physics/rmhd.py` (halo_start width + comment)
- `tests/test_halo_width.py` (new, or extension of an existing script)
- CLAUDE.md architecture section: one sentence noting halo width is now a parameter and
  the halo_start/z_derivatives width-agreement requirement.
