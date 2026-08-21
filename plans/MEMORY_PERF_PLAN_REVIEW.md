# Review of MEMORY_PERF_PLAN (5ce902d)

Reviewed against the buffer dumps and probe runs behind `plans/old/TARANIS_MEMORY_HANDOFF.md`.
Verdict: the plan is sound and in several places better than the hand-off it supersedes.
Four issues below are substantive — two of them measured, not opinions. Numbered by where
they land in the plan.

---

## Substantive

### 1. Z2's `w − 1/w` form fails its own fp32 gate at the branch boundary — **measured**

Z2 proposes `w = exp(s·τ)`, `cosh = (w + 1/w)/2`, `sinh/s = τ·(w − 1/w)/(2·s·τ)`, keeping the
existing Taylor cutoffs (`_TOL_Z2_FP32 = 1e-4` on |z²|, i.e. |z| < 1e-2). The `w − 1/w` form
loses ~2 digits to cancellation, and the cutoff was tuned for `jnp.sinh(z)/z`, which does not:

| \|z\| | fp32 `cosh` rel err | fp32 `sinh(z)/z` rel err |
|---|---|---|
| 1e-1 | 5.0e-10 | 4.8e-07 |
| 3e-2 | 1.9e-08 | 8.8e-07 |
| **1e-2** (the cutoff — first point the w-form must carry) | 6.8e-08 | **1.2e-05** |

Z2's stated fp32 gate is 1e-5 relative. It is missed at exactly the worst point, and the
error there is *worse* than at |z| = 1e-1, because cancellation grows as z shrinks. fp64 is
marginal for the same reason: 6.2e-14 at |z| = 1e-3 against a 1e-13 gate.

**Fix:** raise both Taylor cutoffs by ~100× in |z²| (10× in |z|) — fp32 to |z²| < 1e-2
(|z| < 0.1, where the w-form is 8.8e-7), fp64 to |z²| < 1e-4. The existing 4-term series has
truncation |z|⁸/362880, which at |z| = 0.1 is 2.8e-14 — still far below fp32 eps, so the
series covers the widened branch with room to spare. Without this, Z2 either fails its gate
or (worse) the gate gets quietly relaxed.

Note this is a Z2-specific hazard, not a putzer2 one: the *current* code is fine at these
cutoffs. Z2 must move the cutoff and the coefficient form together, in one commit.

### 2. F2's model of the halo buffers is wrong in a way that changes the fix — **measured**

The plan describes "a `(2, nz+4, nkx, nky)` copy of the whole state, which the buffer table
shows twice … the two stencil consumers appear to each get their own materialised copy".
The optimized HLO says otherwise. XLA decomposes the single 3-way
`concatenate([recv_left, f, recv_right])` into **two 2-way concatenates of nz+2 planes each**:

```
%fused_computation.1 (c64[2,32,128,65]) -> c64[2,34,128,65]
  ROOT concatenate(%slice.210, %param_0.3), dimensions={1}
%fused_computation.2 (c64[2,32,128,65]) -> c64[2,34,128,65]
  ROOT concatenate(%param_0.4, %slice.211), dimensions={1}
```

Both are `kLoop` `slice_concatenate_fusion`s — the halo slice is already fused in; the cost is
purely the two materialised nz+2 buffers (2.13 u each, 4.26 u total). There is no nz+4 buffer
and no per-consumer duplication.

Consequences for F2:
- the target is still ~4.3 u and the interior-slice approach is still right;
- but the gate "if only one copy goes, report why" is premised on a wrong cause. Both halves
  are one concatenate; either the rewrite removes both or it removes neither;
- the metadata places these inside `cond/branch_0_fun` — i.e. the `lax.cond` in
  `_lsrk_scan_stages`. **F3 removes that cond.** Re-dump after F3 before designing F2's
  assembly, because the concatenate scheduling may change once it is no longer inside a
  cond branch. Cheap insurance: reorder to F3 → F2.

### 3. Z1's memory gates are 3–7 u looser than Z1's own arithmetic

§4 Z1 gates: "lsrk54 hoisted 62.0 → ≤ 41 u; lsrk33 43.2 → ≤ 37 u". But the stated content of
the phase is "the −22/−3.4 u hoist term and the −6 u `lin_*` both gone", which gives
62.0 − 28.2 = **33.8** and 43.2 − 9.4 = **33.8**. Both gates would pass with roughly a third
of the win missing.

There is a much sharper gate available, and it falls out of the measurements: after Z1
nothing scheme-dependent is stored, so **lsrk33 and lsrk54 must land on the same number**.
The unhoisted rows already demonstrate this — both measured 39.8 u today — and 39.8 − 6.0
(`lin_*`) = 33.8. So:

> **Gate: z_spectral lsrk33 and lsrk54 both at 33.8 ± 0.5 u, and equal to each other to
> within 0.1 u.**

That is a two-sided gate with teeth, it catches a partial `lin_*` removal, and it catches an
implementation that accidentally leaves a per-stage array live. It also makes §2's ~27 u
target check out exactly: 33.8 − 4.9 (F1, measured on z_spectral) − 2.0 (F3) = 26.9.

### 4. Z1's timing gate is set to the wrong side of the prediction

"z_spectral lsrk33 fixed-dt step ≤ 1.05× the hoisted putzer2 step" treats parity as success.
Z1 should be *faster* than hoisted putzer2 on GPU, and the gate should say so.

Per stage, the hoisted putzer2 apply streams `m00,m01,m10,m11` (4 u) plus the field (2 u) and
writes 2 u — **8 u of traffic**. The separable apply reads three arrays that are perp-plane
or length-nz (≈0 u), the field (2 u), and writes 2 u — **4 u**. On a bandwidth-bound GPU that
is a 2× reduction in the propagator's memory traffic, on top of removing the transcendentals.
The hoist bought its speedup by trading arithmetic for bandwidth; Z1 declines both.

**Suggested gate:** ≤ 1.0× the hoisted putzer2 step on CPU, and ≤ 0.85× on GPU. If GPU comes
in above 1.0×, that is a signal the apply is materialising an intermediate (e.g. `c·a0` before
the `P·` multiply) and should be caught, not accepted.

---

## Smaller points

**§6, B1.** Deferring the one-line default flip behind Z1 is the one place I would push back
on judgement rather than detail. Z1 is a new backend with seven gate items, particles
plumbing, and a doc sweep; B1 is one line for 22 u, reversible, and §9.4 already concedes a
production run might arrive first. The knob exists either way, so flipping the default is not
knob churn. The real cost is the 1.7× step-time regression for anyone on a fixed-dt
z_spectral run — but an OOM costs more than a slow step, and the flip is trivially undone.
I would land it in Phase 0 with the baseline and let Z1 delete it.

**Z1, the Hermitian check.** "building the dense operator in HOST numpy at setup and passing
it through `_check_hermitian_compatible`" materialises 4 u of numpy, and
`_check_hermitian_compatible` then does `np.roll(np.flip(...))`, which materialises again.
At 512²×128 fp64 that is ~3.2 GB of host RAM, transiently, in the setup path of a
memory-reduction plan. It will probably survive on a compute node and probably not on a
login node. Either check the constraint analytically for the separable form (`dperp` even in
kx by construction; the kz mirror handles the off-diagonal) or do it plane-by-plane. Worth a
line in the phase brief so an implementer does not discover it at 1024².

**Z1, a free win worth recording.** `(I − a·L)⁻¹ = [(1−a·d)I + i·a·kz·σₓ]/((1−a·d)² + a²kz²)`
— I checked the algebra, it is right. Note the denominator is `A² + B²` with `A = 1 − a·d`
and `d ≤ 0`, so `A ≥ 1` for `a > 0` and the denominator is bounded below by 1. The separable
backend therefore has **no pole**, unlike putzer2's `det(I − a·L) = 0` at `λ = 1/a`. That
retires one of the two documented NaN traps in `propagators.py` for this path, and should be
stated in `docs/numerics.md` rather than left implicit.

**F1.** The tuple return is the right call and catches the trap my per-component sketch fell
into (8 separate outputs then `stack` = a second 7.9 u copy). No notes. One consumer worth
adding to the grep list beyond those named: `diagnostics/` — check whether anything there
takes `grads` as an array.

**F1/Z1 parallelism.** Both touch `rmhd.py`. The conditional-parallel rule in §7 is more
coordination risk than it saves; F1 is small, so serialise F1 → Z1 and drop the caveat.

**§0.1.** "No new user-facing knobs", with Z3 proposing to delete one, is the right principle
for this codebase and worth keeping past this plan.

---

## Recommended edits, in order

1. Z2: move the Taylor cutoffs with the coefficient form; restate the fp32 gate (issue 1).
2. Z1: replace the two one-sided memory gates with the two-sided equal-schemes gate (3).
3. Z1: tighten the timing gate and state the bandwidth prediction (4).
4. F2: correct the buffer description; sequence F3 before F2 (2).
5. Phase 0: land B1.
6. Z1 brief: note the host-memory trap in the Hermitian check; record the no-pole property.
