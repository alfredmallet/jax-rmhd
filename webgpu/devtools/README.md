# webgpu/devtools — sandbox verification tooling (REFINE_PLAN gates)

Node/python scripts built during REFINE_PLAN phases F–G to verify the apps without a
GPU. Saved here so later phases (and fresh sandboxes) don't rebuild them. Safe to
leave untracked or commit — they are dev-only, nothing in the apps loads them.

- `stubenv.js` — the shared stub: a DOM + WebGPU stub good enough to run a real app
  page (common.js + physics.js + its inline script) under node, plus `run()` to keep
  evaluating in the page's context. `require("./stubenv")(dir, page, demo)`. Since
  Phase H.0 the control panel is BUILT from a spec, so every tool that needs to see a
  control has to boot the page — hence one stub, three consumers. Its `Path2D` stub
  counts the points the overlays (arrows, field lines) push into a path and fails on any
  non-finite coordinate.
- `dumpwgsl2.js <dir> <page> "" <out.txt> ['{"pm":10}']` — emit every generated WGSL
  kernel to text for byte-diffing against a pre-phase baseline (capture the baseline from
  clean git HEAD first). `kdiff.py` diffs two dumps kernel-by-kernel. The optional JSON
  overrides every parameter set (`pm` = Pm = nu/eta, `eqsrc` = the maintained-flux
  source, `ny`/`Lx`/`Ly` = a rectangular 2D box),
  which is how a knob's kernel footprint is shown: dump twice, diff the two.
- `wgslparse.mjs` — parse all emitted kernels with wgsl_reflect (closest available
  compile check; npm i wgsl_reflect first).
- `bootstub.js <dir> <page> [demo]` — stub-GPU boot of a real app page; exercises
  cards (add/close/reuse/retype), every chart card's option selects with a frame drawn
  per value, the cut card's own z source, presets, ?demo= links, colormaps, the 3D cube
  VIEW (five fields × manual / track ζ±, top-face plane, arrow frame, cut card exclusion,
  add/remove with a cube card live — I2.1-I2.3), the contour overlay (ψ / φ / both / off ×
  two level counts × field and plain background × slice and cube — I2.4, J2.1, J2.2), Pm
  (4, 0, back to 1) and the maintain-flux toggle down to the emitted WGSL (J2.3, J2.6),
  the tearing/KH hyper-lock split (J2.5), the 3D field-lines VIEW (K2: per card, beside a
  cube and a slice card, the 64 polylines and 12 box edges projected point by point, the
  dead z controls, the contour selects driving the ink-only top face, its absence from the
  cut card, and the `k∥ (field line)` chart tracing on its own — with the sample readback
  asserted to follow that chart alone — K.1–K.3, K2.1–K2.5),
  the editor view
  (enter/paint/save/cancel/save&run), eps± lock, band rebuild, amplitude rescale, and
  the self-test path end to end. Validates dispatch sizes, bind groups, writeBuffer
  extents. Since FEEDBACK_2026-08-08 P2 it also drives the chart cards' NUMERIC options
  (the fit line's index / amplitude, incl. blank and NaN), the 3D sinusoidal z± packet IC
  (envelope planes, hidden paint row, live chi line) and the auto-diss controller end to
  end on the live solver: tick disables the slider, a loud synthetic shell drives it up
  and reaches `solver.p.diss`, an empty one holds, the no-spectrum-card path takes its own
  readback, and unticking leaves the value alone. The KH `k_y = 2pi/Ly mode` card is driven
  beside the island one: the preset must OPEN with it, the stub's all-zero readback must
  leave the log-y history EMPTY, a synthetic exponential line must trace and fit, a zero
  b_x line must drop its series rather than the axis, and only the KH IC may arm it.
- `contrepro.js <dir> <page> [demo]` — contour-overlay dataflow tracer
  (FEEDBACK_2026-08-08 P0.2). Patches the stub device to record every `writeBuffer` and
  every (pipeline, bind group) dispatch IN ORDER, names every buffer and pipeline, and
  replays the frame symbolically (plus a numeric model of `contLevel`'s adapting range,
  fed a tearing-like |psi| ~ 1 / |phi| ~ 1e-2). So it separates the three ways the overlay
  can lie: which potential is PREPARED into `cp`/`cp2`, which level table INKS it, and
  whether the level spacing still belongs to a PREVIOUS potential. Walks psi / phi / both /
  off transitions and a card close-and-re-add on the same chain slot, and asserts on the
  emitted `colorize` / `colorizeCube` that set 0's ink does not depend on the background
  colour. Run it against a reverse-patched copy of the two shared files to see the failures
  it was written for.
- `checks.js` — GATE G fp64 unit checks (normalized-IC rescale, sigma_letter k-perp
  content at two resolutions, chi formula, separation cap, centroid/argmax trackers)
  against the real common.js functions via vm. Sections 6-9 (FEEDBACK_2026-08-08 P2) add
  the auto-diss controller's PURE core on synthetic spectra (the analytic nu_target, the
  clamp / smoothing / cap, a quiescent start that holds instead of collapsing, a growing
  KH-like spectrum, and a closed-loop cascade model whose fixed point the controller finds
  from 3 decades either side), the diss slider's dynamic range (every preset's value
  survives the open-then-narrow sequence unchanged; a re-range widens outward instead of
  moving it), the fit line's index / label / anchor algebra, and the sinusoidal z± IC
  (2D DFT mode content, envelope placement, icZetaFields normalization, and the field it
  produces on its packet plane). Those sections need controls that REMEMBER writes, hence
  the `ELS` map behind `getEl`; sections 1-5 keep the old throwaway elements. Section 10
  covers the KH `k_y = 2pi/Ly mode` chart's arithmetic: `modeAmps` on a synthetic cut stack
  (known m = 1 amplitude and phase under a constant offset, an m = 2 contamination and
  6-decades-louder junk in the rows it must not read), `modeFitGamma` on exponentials
  (even/uneven sampling, trailing-window only, and NaN on flat / decaying / nonpositive
  data), and the `modeHist` record through HIST_MAX halving and a paused clock.
- `checkpin.js [dir]` — the PINCURVE gates (pinned ghost spectra): `specCurves` against the
  PRE-REFACTOR inline loop, ported verbatim into the file, over every (bin stack × `sq` ×
  `sd`) combination, plus — with `PINCURVE_REF=<a pre-refactor common.js>` — a byte-diff of
  the whole `drawSpectrum` CANVAS CALL LOG against that reference (the no-GPU stand-in for
  the plan's "draws pixel-identically by eye"). Then the pin arithmetic on the real
  `pinDraw` / `specFloor` (deep-copy independence, the `kunit_pin / kunit_live` rescale, the
  age-alpha ladder, the range union), the waiting-early-out truth table and the draw ORDER
  and alphas off a recording 2D context, and finally the four motivating workflows driven
  through booted `rmhd2d` / `rmhd3d` pages: pin then move `rDiss`, pin then move `rEpsP`,
  the 4-pin cap and its refusal, per-card independence, retype clearing, the
  `?demo=decay` → `forced` preset switch with the `cardsLayout` transplant, a pin at
  t = 0, and the 3D `sd = both` card's dashed pinned-parallel ghosts (which must not stretch
  the axis). No GPU, no pixels — it asserts on the pins arrays and the drawn call log.
- `checksh.js` — GATE H fp64 checks against the pages' inlined reference A states:
  the H_c accumulator lane, E± = E_kin+E_mag±H_c = ½⟨|z±|²⟩, the spectra lanes summing
  back to the energies, and cutPrep+rowsC2R reproducing (u_x,u_y,b_x,b_y) on x = Lx/2 —
  each compared against a direct fp64 inverse DFT, not against itself. Also drives the
  REAL chart series (ENERGY_MODES / SPEC_SETS / CUT_PAIRS) and checks that a 1e6×
  louder E(k∥) does not move the spectrum chart's y axis.
- `checkj.js [dir]` — GATE J physics checks: the equilibrium ICs built by the REAL app
  code (booted on the stub) against their analytic profiles, the island-width machinery
  on an analytic island, per-field nu/eta decay rates, and the tearing / KH linear
  GROWTH rates from a small fp64 pseudospectral 2D RMHD solver written in the file
  (plain complex FFTs + RK4, same 2/3 dealias) — compared against `eqlinear.py`, not
  against itself. Section 4b (GATE J2) adds the maintained equilibrium flux: the same
  solver with the app's source term, checking that psi_eq is then stationary to round-off
  and that the free-running growth rate is the frozen-equilibrium eigenvalue again.
  Section 5 also runs the app's OWN `modeAmps` (what the `k_y = 2pi/Ly mode` chart plots)
  over a cut line built from those fp64 fields on x = Lx/2, at the same tolerance as the
  KH rate beside it.
- `checkk.js [dir]` — GATE K fp64 checks: the field-line integrator against the analytic
  line of a single-mode ψ (both perpendicular components, RK2 order measured by halving
  dz), the along-line sampler against the analytic (u, b) at the position the polyline
  reports, and the REAL `flHann` / `flSpectrum` / `fftPow2` on synthetic signals (window
  shape and mean square, peak bin, Parseval through the window, the E±/H_c lane algebra,
  and the leakage a non-periodic line would otherwise put at high k∥). The marcher itself
  is WGSL, so sections 1–2 drive a documented fp64 MIRROR of that kernel — never the
  kernel against itself — and the kernel's own dz/dx, dz/dy constants are read out of the
  emitted WGSL. Section 0 also pins `cubeTopXform` to its pre-K pixels and checks the
  box frame against all twelve projected face corners.
- `sigrcheck.js [dir]` — the residual-energy display mode (2D `sigma_r`, mode 9): boots
  rmhd2d on the stub and records, off the stub device, the Mode uniforms each selected
  field writes (mode 9 + its pinned mate 5 = b, against sigma_c's 8 + 7) and the whole
  encoded frame as an ordered bind-group list (identical to sigma_c's, no autoscale
  reduction, no arrow gather). Then it PARSES the emitted WGSL — prepDisp's mode ->
  potential branch chain, its -i*ky / +i*kx components, sigmaCombine's ratio and floor,
  colorize's fixed +-1 predicate — and drives an fp64 mirror of that parsed chain on an
  analytic (phi, psi), comparing against sigma_r computed directly from the analytic
  gradients. It does not execute WGSL: what it pins is the wiring and the composition.
- `eqlinear.py [n]` — the linear reference for those rates: a 1D generalized eigenvalue
  solve of the linearized RMHD system on Fourier differentiation matrices at
  k_y = 2pi/Ly, plus a shooting solve for Delta'a. Prints the benchmark table checkj.js's
  REF block quotes (regenerate it there from this output), then eta- and b0-survey
  tables. `n` = Fourier modes, default 384; 768 reproduces every printed digit.
- `dup.py` — token-normalized >=10-line clone detector over common.js/physics.js +
  extracted app scripts (the standing-rule duplication audit). Run it over the extracted
  HTML *bodies* too: markup twins are what H.0 was about.
- `layout.js [dir]` — control-row wrap audit at 360/768/1200 px, off the BUILT element
  tree of a booted page (controls + every card header).
- `names.mjs [dir]` — cross-file identifier resolution check (no redeclares, no frees);
  needs acorn (`npm i acorn`, or `ACORN=<path-to-acorn.mjs>`).
- `cmapcheck.js` — colormap table vs emitted WGSL vs matplotlib reference.

Conventions: run from any cwd with absolute paths; each phase captures a FRESH WGSL
baseline from clean git state before editing; refvector JSON lines in the HTML are
spliced programmatically only.
