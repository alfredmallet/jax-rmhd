# Demos plan: Elsasser displays, dual view, decaying A/B, AW collision, drawn ICs

Five phases, executed by agents in order, each gated before the next. Base: the current
layout (landing `index.html`, `rmhd2d.html`, `rmhd3d.html`, `common.js`, `style.css`;
contracts SPEC.md/SPEC3D.md). Guiding principle: **the demos are configurations, not new
solvers** — no new physics enters the stepping loop in any phase. Both apps' self-tests
are the regression gate for every phase; any phase that changes generated WGSL for an
existing kernel must byte-diff it against the pre-phase baseline or justify the diff.

## Phase A — physics template refactor (`physics.js`)

Factor the per-app physics WGSL that differs only by indexing into parameterized
templates in a new shared `physics.js` (loaded between `common.js` and the app script):
`prepGrads`, `bracket`, `nlAssemble`, `energyPartial`, `ou`, `scale`, `icFinish`,
`prepDisp`, and the display-chain kernels, each parameterized by the same kind of
constants object the FFT template already takes (index prefix `m -> (iz, mp)`, dealias
expression, per-mode linear diagonal `d`, norm constant). NOT shared (stay per-app):
`stage` (diagonal vs 2×2 wave), `forcingAdd`/`envExpand` (dense vs two-plane),
`specReduceZ`/`specPar`, `sliceExtract`/`faceExtract`/cube path.

GATE A: generated WGSL byte-identical to pre-phase baseline for every kernel at every
resolution preset in both apps (the refactor moves code, it must not change one byte);
fp64 harnesses re-run unchanged; node --check; both self-test wirings intact.

## Phase B — dual displays + Elsasser display modes

1. **New display modes** (both apps, in the shared display templates):
   - `|z⁺|`, `|z⁻|`: z± = φ±ψ (Elsasser potentials); display magnitude of the vector
     field ẑ×∇z± = (−∂y z±, +∂x z±) — same machinery as the |u|/|b| modes with
     f = phik±psik formed in prepDisp (one extra complex add; no new FFTs beyond the
     vector modes' two). Arrow overlays work as for |u|/|b|.
   - `σ_c` (normalized cross-helicity, pointwise): σ_c = (|z⁺|²−|z⁻|²)/(|z⁺|²+|z⁻|²)
     with the vector z± above. Needs 4 inverse FFTs (both components of both z±) —
     acceptable display cost. Signed ∈[−1,1], afmhot symmetric with FIXED range ±1 (no
     autoscale). **Energy floor**: where (|z⁺|²+|z⁻|²) < 1e-4 × the slice max of that
     sum, render σ_c = 0 (prevents noise domination in quiet regions). The floor uses
     the existing max-reduction on the energy sum.
2. **Dual display**: a "dual view" toggle adds a second canvas (own `show` selector; own
   z-slice selector in 3D) beside/below the first (responsive: side-by-side ≥1400px,
   stacked below). Implementation: instantiate the display chain twice (scratch buffers,
   bind groups, textures ×2); one extra chain-run per rendered frame when enabled.
   Arrows + cut-trace remain display-1-only; cube mode allowed on display 1 only (when
   display 1 is in cube mode, display 2 still shows its slice mode). Default: dual OFF,
   ON in the demos.

GATE B: node --check; wgsl_reflect on new/changed shaders; fp64 spot-check of the σ_c
and |z±| formulas vs a numpy/node reference built from the existing refvectors' A state
(no new jax vectors needed — z± modes are linear combinations of recorded fields);
sonnet review.

## Phase C — IC-preset framework + Demo 1 (decaying A/B) + landing cards

1. **IC framework**: `setICFromReal(phiReal, psiReal)` — CPU Float32Array(s) (2D:
   (nx,ny); 3D: (nz,nx,ny), ≤8MB at 64²×256) → upload → forward FFT → dealias (existing
   icFinish). Presets registered per app: `quiescent` (current behavior), `letters`,
   later `custom` (Phase E). Reset re-applies the current preset.
2. **Letter ICs**: rasterize a glyph ("A" for z⁺, "B" for z⁻ — selectable text, default
   A/B) on an offscreen 2D canvas at ~60% of the box, gaussian-smooth (canvas
   `ctx.filter="blur(Npx)"` with a 3-pass box-blur JS fallback), normalize so the
   resulting max |ẑ×∇z±| equals the amplitude knob value, then φ=(z⁺+z⁻)/2,
   ψ=(z⁺−z⁻)/2. In 3D multiply each by a gaussian z-envelope exp(−(z−z₀±)²/2σ_z²)
   (per-packet z₀, σ_z from the preset).
3. **Demo 1 — decaying 2D turbulence** (`rmhd2d.html?demo=decay`): query-param
   preselect: letters IC (amplitude slider, default 1.0), forcing checkbox OFF, dual
   view ON with |z⁺| / |z⁻|, 512², hyper=4 + auto diss, spectrum panel visible.
   URL query parsing lives in common.js (`applyDemoParams(registry)`); each app
   registers its demos. Watch for: quasi-universal decaying spectrum, dynamic alignment
   (suggest σ_c in the hint text).
4. **Landing page**: a "Demos" card row linking `rmhd2d.html?demo=decay` (and Phase D's
   `rmhd3d.html?demo=collision` once it exists — add both cards in Phase D to avoid a
   dead link).

GATE C: node --check; IC unit checks in node (letter rasterization energy/normalization:
requested amplitude reproduced to ~1%; φ/ψ↔z± roundtrip exact); self-tests untouched;
`?demo=` and plain URLs both boot correctly.

## Phase D — Demo 2: Alfvén-wave collision (3D)

1. **Grid generality**: `Lz` becomes a per-preset parameter of the 3D app (kz, z_diss_k
   auto, parallel-spectrum axis, envelope math all already derive from Lz — audit every
   `2π/Lz`-adjacent constant). New resolution preset `64²×256` (+`64²×128` as the light
   option); z-FFT lines of 256 fit workgroup memory (4KB); check dispatch/buffer budgets.
2. **Demo 2** (`rmhd3d.html?demo=collision`): Lz = 8π, 64²×256 default. IC: letter z⁺
   packet ("A") with gaussian envelope at z₀⁺, letter z⁻ ("B") at z₀⁻, σ_z ≈ Lz/16,
   placed so the packets are well separated and counter-propagate toward each other —
   **determine the propagation direction of z± from the implemented propagator** (check:
   which sign of kz·t phase does each eigencombination get) and place packets so they
   collide near z = Lz/2. Amplitude slider; display a live **χ readout** χ ≈ a·k̄⊥/(k̄z·v_A)
   with k̄⊥ from the letter scale (≈ 2π/(0.3·Lx)) and k̄z = 1/σ_z, v_A = 1 — labeled as an
   estimate. Forcing OFF; dual view ON.
3. **Max-energy plane tracking**: per-z-plane perpendicular energy of z⁺ and z⁻ (small
   reduction kernel → nz values each), ~10Hz readback; each display in "track z±" mode
   sets its slice to the argmax plane (display selector gains "track z⁺ peak"/"track z⁻
   peak" options for the slice source; plain manual slider stays). Show the tracked
   plane index + its z-coordinate in the readout.
4. Landing card for the collision demo; hint text explains weak (χ≪1: packets pass
   through nearly unchanged, distortion accumulates over many transits — periodic box =
   repeated collisions) vs strong (χ≳1: single-collision distortion).

GATE D: node --check; envelope/placement unit checks (packet separation, direction);
energy-tracking indices verified against a node reference on a synthetic two-packet
field; self-tests still green at the standard presets (Lz=2π path unchanged — byte-diff
the standard-preset WGSL); sonnet review with Phase C.

## Phase E — Demo 3: drawn ICs (blob editor)

`custom` IC preset (both apps): an "edit IC" mode that pauses the sim and lets the user
paint into a chosen target field (z⁺, z⁻, φ, or ψ): click/drag on the display canvas
deposits gaussian blobs (controls: σ_perp, amplitude ±(sign toggle or right-drag =
negative), and in 3D σ_z + target z-plane taken from the current slice slider). Strokes
accumulate into the CPU-side IC arrays (Phase C framework) with live preview (cheap:
render the CPU array to the canvas via 2D context while editing — no GPU roundtrip);
"apply & run" uploads via setICFromReal; "clear" resets; the drawn IC persists as the
Reset target until changed. Keep the tool minimal (no undo stack; clear is the undo).

GATE E: node --check; blob math unit check (deposited gaussian integral/peak); UI
review; final sonnet review over the whole C+D+E diff.

## Cross-cutting rules

- No new jax reference vectors are needed anywhere (no physics changes); if any phase
  finds itself wanting one, that's a sign it's changing physics — stop and re-check.
- README.md gains a short section per phase; SPEC.md/SPEC3D.md get one-line notes only
  (display/IC features are outside the physics contract).
- All the usual constraints: file:// operation, giant inlined JSON lines never touched
  except by programmatic splice, WGSL hygiene checklist, no repo copies in the sandbox,
  node --check everything, self-tests are sacred.
