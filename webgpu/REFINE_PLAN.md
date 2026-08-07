# Refinement plan (from Alfred's phone-testing notes, 2026-08-06)

Successor to DEMOS_PLAN.md (phases A–E, complete). Phases F → G(+I) → H → I2 → J → K, in
order, each gated by a review before the next. Numbers in [brackets] refer to Alfred's
notes; I2 and K added 2026-08-06 from Alfred's post-H requests.

## Standing rule: maintainability [21]

Applies to EVERY phase and is checked at EVERY gate, not just at the end:
- No copy-paste variants: any code needed twice becomes a parameterized function/template
  in common.js/physics.js (the FFT/physics templates are the model). A reviewer finding
  two near-identical blocks >10 lines is a MAJOR.
- Every gate includes a **duplication audit** (e.g. jscpd or a manual pass over the diff)
  and a line-count report; phases that add features should grow the codebase by roughly
  the size of the genuinely new logic, not more.
- The card/panel system (Phase F) replaces the chain-0/chain-1 special cases with ONE
  display-card class and ONE chart-card interface — dual-view-specific code should be
  DELETED, not kept alongside.
- Existing invariants stay sacred: generated-WGSL byte-diffs against a pre-phase baseline
  wherever physics kernels are not intentionally touched; both self-tests green; fp64
  node harnesses for anything numeric; node --check everything; giant inlined JSON lines
  spliced programmatically only; no repo copies in the sandbox; no background processes.

## Phase F — UI overhaul (mobile-first) [1, 2, 3, 5, 9, 13, 14, 18]

1. **Strip prose** [1]: remove the in-page hint paragraphs; one small "about / README"
   link per app (to the GitHub webgpu/README.md). Demo-specific hints reduce to one line.
2. **Merge demo pages into the apps** [2]: in-app "preset" dropdown (2D: default /
   decaying A-B; 3D: default / AW collision; later KH / tearing from Phase J). `?demo=`
   URLs keep working as deep links that preselect the dropdown. Landing page: exactly two
   cards (2D, 3D) + one line mentioning presets; delete the separate demo cards.
3. **Card system** [14]: displays and charts become addable/removable cards ("+ display",
   "+ chart" buttons; each card has a close button, compact header row with its own
   selectors). One `DisplayCard` class (canvas + field selector + per-card z-slice/track
   selector in 3D + per-card options: arrows on/off [fixes 5], colormap [18]) instantiated
   N times (N ≤ 3 is plenty; scratch buffers per card, lazily built as now). One
   `ChartCard` interface for energy trace / spectra / cut; the fixed chart stack and the
   dual-view checkbox/`ctx2` special-casing are REMOVED. Default layout = what a demo
   preset requests (e.g. collision opens two display cards tracking z± peaks).
4. **Collapsible control groups** [13]: `<details>` groups (sim: resolution/scheme knobs;
   dissipation; forcing; IC; displays), Run/Reset/steps-readout always visible in a slim
   sticky bar. Collapsed state persisted in localStorage… NOT persisted — keep it simple,
   default-collapsed on narrow viewports, default-open on wide.
5. **Mobile layout** [3]: control rows must wrap cleanly at 360px width; IC row splits
   (amplitude on its own line); DROP the letter-text input (fixed A/B) [3]; audit every
   row at 360/768/1200px with a DOM-stub layout check where possible and list the manual
   checks for Alfred otherwise.
6. **Colormaps** [18]: per-display-card dropdown: afmhot (default), viridis, RdBu,
   grayscale. Implement as a small uniform-selected branch or 4 tiny WGSL functions in
   the shared colorize template — one implementation, all cards.
   RdBu/viridis need polynomial fits (6th-order is fine, document coefficients' source).

GATE F: node --check; stub-GPU boot exercising card add/remove/close for every card type
at 2–3 cards; byte-diff: physics kernels untouched; display-kernel diffs are the
intended colormap/per-card ones only; duplication audit (the dual-view special-case code
must be gone); both self-tests; sonnet review.

## Phase G — IC / editor / collision physics [4, 6, 8, 10, 11, 12, 16, 17] (+ I [7])

1. **ζ± terminology** [17]: Alfred is right — what is painted/lettered is the Elsasser
   POTENTIAL ζ± (z± = ẑ×∇ζ±). Rename every UI label, README mention, and code comment
   (identifiers may stay zp/zm; labels must say ζ⁺/ζ⁻).
2. **Letter smoothing** [4]: blur specified as a physical length σ_letter = the blob
   default σ⊥ (Lx/32? — pick ONE constant shared by letters and blob default, defined
   once), not pixels. Letters and blobs then have the same k⊥ content and the same χ
   bookkeeping.
3. **Amplitude semantics** [8]: store drawn/letter ICs NORMALIZED; the amplitude
   slider(s) scale at apply/Reset time. Pause → change amplitude → Reset now genuinely
   rescales (this was a real defect: amplitude was baked in per blob at deposit).
   Add to the 2D decay hint (one line): ideal 2D RMHD is self-similar in amplitude
   ((a,t) → (a/λ, λt)); with adaptive dt the visual dynamics per frame are amplitude-
   independent — amplitude only matters against fixed dissipation. In 3D it is χ.
4. **Separate ζ± amplitudes** [11]: two sliders (linked by default via a lock toggle).
5. **Separate forcing ε±** [11] and **forcing band sliders** [12]: ε⁺ and ε⁻ sliders
   (lockable), fshell (n_min, n_max) two-handle or two sliders; forcing shell rebuild on
   change (shell list is CPU-side — cheap; kgrid fmask constant must move to a uniform or
   trigger the existing rebuild path — prefer rebuild, it's rare).
6. **Packet length sets χ** [6, 10]: σ_z slider for the collision preset (and the letter
   IC generally in 3D); χ = a·k̄⊥·σ_z/v_A with k̄⊥ from the shared σ_letter above (state
   the formula in the hint). Placement enforces ≥5σ_z separation: cap the σ_z slider at
   Lz/12 and show "increase Lz for longer packets" at the cap. Overlap [10] is thereby
   impossible.
7. **Editor as its own view** [16]: "edit IC" switches the main area to a dedicated
   editor canvas (sim hidden/paused), with **Save** (stores the normalized IC as the
   Reset target) and **Run** (save + apply + resume) buttons; Cancel discards. No
   painting over live displays.
8. **[I / 7] Collision display jitter**: fix peak tracking — replace per-readback argmax
   with (a) energy CENTROID along z (smooth, moves at v_A) as the default track source,
   (b) hysteresis for the argmax option (switch planes only when the new peak exceeds
   the current plane's energy by 10%). If stub-level reasoning identifies frame-pacing
   contributions from the readback stalls, note them for Alfred to verify on-device.

GATE G: node unit checks (normalized-IC rescale exactness; σ_letter physical-units
letters at two resolutions agree in k⊥ content; χ formula; separation cap; centroid
tracker on a moving synthetic packet: displacement linear in t, no plane-hopping);
byte-diff (physics kernels: only the forcing-band/ε± plumbing may change — justify each);
self-tests (forcing self-test uses recorded symmetric ε — must still pass); duplication
audit; sonnet review.

## Phase H — diagnostics [15]

0. **Carried-over from GATE G (MAJOR, ship-with-notes)**: dedupe the identical
   forcing-controls `<details>` markup block (and the smaller pre-existing per-app
   control-row twins: topbar, cfl row, diss/hyper row) via a `controlsBuild(spec)`
   helper in common.js, updating layout.js/bootstub.js to consume the spec. Checked
   at GATE H.
1. Spectrum chart card options: E_u/E_b (current), E⁺/E⁻ (same bin kernel on φk±ψk —
   one template, parameterized weight), both, and (3D) ∥/⊥ selection. Alfred
   2026-08-06: the 3D spectrum chart's y-limits must be set by the PERP spectrum
   alone — the dashed E(k∥) curves plot within those limits but don't stretch them.
2. Energy-trace card options: (E_kin, E_mag, E_tot) or (E⁺, E⁻, E_tot) — E± = 2(E_kin +
   E_mag ± H_c)… compute from the existing reductions plus the H_c inner product (one
   small kernel or an extra accumulator in energyPartial — prefer the extra accumulator,
   no new dispatch).
3. Cut card: component selection — (u_x,u_y) / (b_x,b_y) / (|z⁺|,|z⁻|) pairs as two
   curves with a small legend; reuses the display scratch of the card's source display…
   NO — cut becomes self-contained (own tiny prep) so it doesn't depend on which display
   cards exist. Keep it cheap: cut of the SELECTED quantity pair only.

GATE H: fp64 node checks of E± and the pairwise cuts against the inlined refvectors' A
states; node --check; duplication audit; brief sonnet review (can combine with Phase J's).

## Phase I2 — cube view for all fields + contour overlays (pre-J, display-only)

Added 2026-08-06 (Alfred). Physics WGSL stays byte-identical throughout — this is all
display chain.

1. **Cube faces become a VIEW, not a field**: fourth entry in the per-card z-slice/track
   dropdown (manual z / track ζ⁺ / track ζ⁻ / cube faces), flagged OUT of the cut card
   (meaningless there). The field selector stays orthogonal — any field renders as cube
   (3 boundary-plane preps through the shared colorize path). The special cube entries
   in the field dropdown are DELETED, not kept alongside.
2. **Top face follows the card's tracked plane** (cube + track ζ± puts the collision
   front on top); side faces remain boundary slices.
3. **Arrows on the visible top face**: CPU-side only — apply the cube's existing 2D
   affine projection to arrow anchors AND directions on the overlay canvas. Top face
   only (side faces are x/y-planes; u⊥/b⊥ are not tangent to them).
4. **Contour overlay (in-plane field lines)**: per-display-card toggle — ψ contours
   (= B⊥ field lines) and φ contours (= streamlines) — implemented in the shared
   colorize kernel via fract(ψ/Δψ) bands with a neighbor-texel crossing test (compute
   shader, no fwidth). Uniform Δψ, so line density ∝ |B⊥| (physically honest). Level
   spacing: small level-count select with a slowly-adapting range (no per-frame
   flicker). The potential plane must reach the display scratch whatever field is
   shown — at most one extra iFFT per card frame.

GATE I2: physics WGSL byte-identical (display diffs = the intended colorize/cube ones
only); stub boot exercising cube view on ≥3 fields × tracked/manual, contours on/off,
arrows-on-cube; duplication audit (cube field-dropdown special cases gone, ONE colorize
implementation); node --check; sonnet review folded into GATE J's (I2 is small and
display-only; the coordinator still runs the mechanical checks before J builds on it).

## Phase J — equilibrium demos: KH and tearing [19, 20]

Shared infrastructure first (this is most of the phase):
1. **Per-field dissipation in 2D** (ν for φ, η for ψ): the 2D propagator is diagonal —
   no equal-diagonal constraint (that was 3D-only). eqpars-style (diss_nu, diss_eta);
   UI: one diss slider + an "η/ν ratio" (Pm⁻¹) input, default 1 (existing behavior when
   ratio=1 — byte-diff gate on that path). Self-test vectors use scalar diss — unchanged.
2. **Rectangular boxes in 2D**: nx≠ny and Lx≠Ly. Audit the known square assumptions
   (arrowDims/vecGather stride, canvas aspect — display cards get aspect-correct
   canvases, NB=floor(min/3) already min-based, cut along y unchanged). Presets like
   512×128, Lx=4π×Ly=2π. 3D stays square-perp.
3. **Equilibrium IC presets** (2D): registered like letters/blobs, each with a small
   perturbation seed (k_y=2π/Ly mode, amplitude slider):
   - **KH** [20]: φ_eq with u_y(x) = U0·tanh((x−x1)/a) − U0·tanh((x−x2)/a) − U0 (two
     opposite shear layers for periodicity, x1=Lx/4, x2=3Lx/4 — standard double-layer
     trick; document that the two layers are independent while a ≪ |x2−x1|); ψ_eq for
     in-plane B∥ŷ: same double-tanh structure scaled by b0. Sliders U0, a, b0 [20].
     KH suppressed when b0 ≳ U0 (document the ideal threshold b0=U0 for this geometry
     and that resistivity softens it).
   - **Tearing** [19]: ψ_eq = ψ0/cosh²((x−Lx/2)/a) (Numata & Loureiro-style; net-flux
     free and exponentially periodic for a ≪ Lx), φ_eq = 0, perturbation in φ or ψ at
     k_y = 2π/Ly. Sliders ψ0, a, η (via the ratio control), Ly/a sets Δ'a. hyper LOCKED
     to 1 for these presets (hyper-dissipation falsifies tearing/Rutherford physics —
     enforce, don't trust the user to know). Default: 512×128 grid, Lx=4π (a ≈ 0.1·Lx
     stays ≫ box-edge distance — [19]'s "several cs widths"), Ly=2π.
4. **Island-width chart card** [19]: W(t) from ψ extrema difference on x = Lx/2 line
   (X/O points live on the resonant surface): tiny readback (one line of ψ, reuse the
   cut machinery), W = 4·sqrt(a²·(ψ_X−ψ_O)/ψ0'') — use the standard island-width formula
   with the measured equilibrium curvature, state it in the README. Log-linear W(t) chart
   makes the linear stage a straight line; Rutherford = algebraic; collapse/saturation
   visible. Also useful for KH (mixing-layer width from the u_y profile — OPTIONAL,
   skip if it bloats the phase).
5. Landing/preset wiring: KH and tearing join the 2D preset dropdown (+ `?demo=` deep
   links); one-line hints each.

Physics sanity targets (agent verifies in node with a small CPU spectral reference or
against jax_rmhd run in the sandbox — jax IS available, PYTHONPATH=/var/tmp/pylibs plus
repo root, RMHD_PRECISION=64, dims=2 serial): tearing linear growth rate at one
benchmark parameter set within ~10% of a jax_rmhd 2D run with the same IC (generate the
jax comparison the way webgpu/gen_refvectors.py does — a NEW small deterministic vector
file for the tearing IC is allowed IF the check needs GPU-path comparison; otherwise a
node-side estimate suffices); KH growth for b0=0 vs suppression at b0>U0.

GATE J: the above physics checks; byte-diff on all non-equilibrium paths (ratio=1 scalar
path bitwise-identical); self-tests; duplication audit (per-field diss must not fork the
stage kernel — parameterize); sonnet review.

## Phase J2 — post-J polish (Alfred 2026-08-06, items 1–6 of his queries)

1. **Contours without the field**: per-card background option (field / plain) — contour
   ink on a blank background.
2. **Both contour sets at once**: ψ AND φ (third contour-select entry "both"); second
   potential plane in scratch, second contInk pass in a fixed accent color (ψ stays
   ink-black/white) so the two are distinguishable on any background. Alignment
   inspection is the use case.
3. **Maintain equilibrium flux (tearing)**: static source S = −η∇²ψ_eq on the ψ
   equation (precompute η·k²·ψ_eq,k at Reset), emitted at WGSL-generation time ONLY
   when the preset enables it — every non-equilibrium path stays byte-identical (same
   pattern as the Pm substitution). Toggle "maintain equilibrium flux", default on for
   tearing. Source uses η (the ψ coefficient), not ν. fp64 gate: maintained ψ_eq
   exactly stationary; linear γ then matches the frozen-equilibrium eigenvalue
   (0.0287 at the η=ν=1e-3 benchmark — NOT the free-running 0.018); update
   hint/README (the free-running caveat text becomes the source-off description).
4. **Titles**: pages become "2D RMHD" / "3D RMHD"; drop forced-turbulence wording from
   titles and landing cards (forcing is one option among the ICs).
5. **Hyper lock split**: tearing stays locked to 1 (resistive-layer physics is the
   demo); KH UNLOCKED (ideal instability — hyper is desirable for secondary
   structure). KH hint's threshold footnote says "dissipation", not "resistivity".
6. **Pm replaces η/ν**: input is Pm = ν/η; the diss slider now means η and ν = Pm·diss
   (substitution moves to the φ diagonal + energyPartial's ek term). Pm=1 is the
   bitwise-identical legacy path (gate). Pm=0 allowed (pure tearing, ν=0) — one hint
   line warning that inviscid φ piles energy at the grid scale in long runs.

GATE J2: byte-diff (Pm=1, source-off paths bitwise identical; Pm≠1 and source-on diffs
justified line-by-line); fp64 checks per item 3 + Pm=0 tearing γ against the eigenvalue
solver (eqlinear.py takes independent ν, η already); both self-tests byte-identical;
stub boot incl. contour background/both modes and the KH hyper unlock; duplication
audit; adversarial Fable review (may combine with GATE K's).

## Phase K — 3D field lines + true k∥ spectrum (post-J)

Added 2026-08-06 (Alfred). This is the GPU-touching half of the field-line idea, split
out of I2 because it is a real project: new WGSL, volume pass, interpolation.

1. **GPU field-line integration**: b⊥ = ẑ×∇ψ volume pass (spectral gradient + iFFT —
   REUSE the existing gradient/FFT templates, do not fork) at the field-line update
   rate (~2 Hz, NOT per step), then a small integration kernel marching
   dx⊥/dz = b⊥/B0: RK2 (midpoint), bilinear in-plane interpolation, uniform dz steps,
   periodic ⊥ wrap. Seeds: N_lines grid on the bottom face. Readback = polylines ONLY
   (N_lines × nz × 2 floats, kilobytes) — never the b⊥ volume.
2. **Rendering**: polylines through the existing oblique cube projection on the overlay
   canvas. No WebGL, no occlusion handling.
3. **True k∥ spectrum**: sample z± along each line during the march (the interpolation
   is already in hand). Samples are uniform in z and arc length ≈ z to leading order in
   RMHD, but the signal is NOT periodic (lines exit ⊥-displaced) — Hann window before
   the FFT and/or second-order structure functions vs parallel lag (the more standard
   object); average over the line ensemble. Chart option beside the coordinate E(k∥):
   "k∥ (field line)". Same y-limit rule as H.1: the perp spectrum sets the limits.

GATE K: fp64 node check of the integrator against an analytic b⊥ (single-mode ψ →
known sinusoidal line displacement) and of the along-line sampler; window/SF
correctness on a synthetic signal; physics WGSL byte-identical outside the volume
gradient pass; stub boot; duplication audit; sonnet review.

## Phase K2 — field lines are a VIEW (Alfred on-device feedback, 2026-08-06)

Lines drawn over the cube faces looked wrong. Rework, display-side only (the fieldLine
kernel, march, readback and k∥(field-line) chart machinery are UNTOUCHED):

1. **"field lines" joins the per-card view dropdown** (manual z / track ζ⁺ / track ζ⁻ /
   cube faces / field lines), excluded from the cut card exactly like cube. The
   page-level lines select is DELETED.
2. **N_lines fixed at 8×8** — no UI; one constant.
3. **Lines view rendering**: the oblique box frame (edges) + the 64 polylines through
   the existing projection, on the card's overlay canvas, over a plain background —
   plus a TRANSPARENT top face (Alfred 2026-08-06): the top boundary plane rendered as
   contour INK ONLY (no fill/plate — lines behind it stay visible; thin-ink overprint
   is acceptable, solid-face overdraw was not) through the existing cube projection and
   J2 contour machinery. The per-card contour controls (ψ/φ/both, levels) stay LIVE in
   lines view and drive the face; default ψ. Physics tie-in: endpoints puncture the
   exit plane on its ⊥ field-line structure. No arrows; field selector is inert for
   the line geometry (lines are ψ-lines) — hide or ignore, whichever is less code.
   This is the ONE intended display-kernel WGSL diff of the phase; physics kernels and
   both self-tests stay byte-identical.
4. **Tracer scheduling**: march + polyline readback when any lines-view card exists OR a
   k∥(field-line) chart is open; the sample readback (flSmp) only when the chart needs
   it (reviewer note from GATE K folded in here).
5. Track/z-slider disabled in lines view (whole-box object, no plane).

GATE K2: physics WGSL byte-identical, display diffs = the intended top-face-ink ones
only; stub boot exercising the lines view per card incl. coexistence with cube/slice
cards, the chart-only path, and contour-select changes in lines view; dup audit;
node --check; line count should be ~flat (a select and its wiring are deleted, the
face path reuses existing machinery); compact adversarial Fable pass on the diff.

## Execution notes for the coordinator

- One opus agent per phase (continue the previous agent when its context is directly
  relevant); sonnet gate review after F, G(+I), H (done solo 2026-08-06), a combined
  I2+J review, and K; relay review fixes before proceeding.
- The coordinator personally enforces the maintainability rule at each gate: read the
  builder's line-count/duplication report critically; if a phase grew the code more than
  its new logic justifies, send it back before review.
- Nothing in F–J touches stepping physics except G's forcing plumbing and J's per-field
  diss — everything else must leave physics WGSL byte-identical. No new jax reference
  vectors EXCEPT the optional tearing comparison in J.
- After each phase lands, remind Alfred: reload, self-test both apps, click through the
  changed surfaces on desktop AND phone before the next phase builds on it.
