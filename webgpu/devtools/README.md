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
  non-finite coordinate. Since FEEDBACK_2026-08-10 item 13 it also stubs the CAPTURE path
  — `canvas.toBlob` / `canvas.captureStream`, `MediaRecorder` (with `isTypeSupported`),
  `Blob` and `URL.createObjectURL`, plus `click()` on an `<a download>` — and logs every
  blob and every download it produces on `env.caps`, so a consumer can assert on the file
  that came out rather than on the handler having run. The WebCodecs recording leg is
  stubbed the same way: a deterministic `VideoEncoder` (one chunk per `encode()`, an
  avcC-shaped `decoderConfig.description` on the first, and two knobs — `stall` to grow
  `encodeQueueSize` into the app's drop-frame guard, `noAvcC` to force the bail back to
  `MediaRecorder`), `VideoFrame`, `EncodedVideoChunk`, a `Blob` that KEEPS the bytes, and
  `setInterval` as a hand-driven pump — `env.tick(n)` fires exactly n frames, so a "30 s"
  recording costs no wall clock, and `env.fireTimeout(ms)` fires the armed 30 s hard stop.
  A fourth argument carries the boot knobs: `{noGpu: true}` removes `navigator.gpu`, so
  `initGPU` takes its first failure path and the no-WebGPU poster fallback runs for real.
- `dumpwgsl2.js <dir> <page> "" <out.txt> ['{"pm":10}']` — emit every generated WGSL
  kernel to text for byte-diffing against a pre-phase baseline (capture the baseline from
  clean git HEAD first). `kdiff.py` diffs two dumps kernel-by-kernel. The optional JSON
  overrides every parameter set (`pm` = Pm = nu/eta, `eqsrc` = the maintained-flux
  source, `ny`/`Lx`/`Ly` = a rectangular 2D box),
  which is how a knob's kernel footprint is shown: dump twice, diff the two.
- `wgslparse.mjs` — parse all emitted kernels with wgsl_reflect (closest available
  compile check; npm i wgsl_reflect first, or `WGSL_REFLECT=<path-to-wgsl_reflect.module.js>`,
  the same idiom names.mjs uses for acorn).
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
  b_x line must drop its series rather than the axis, and only the KH IC may arm it. The
  island card is driven the same way since FEEDBACK_2026-08-10 item 9: a flat W(t) must
  quote NO rate, and a synthetic linear-stage island (psitilde ~ e^{gamma t}, so
  W ~ e^{gamma t/2}) must come back at gamma, not gamma/2. The chart-count assertion is
  taken against the BOOT layout, not a constant -- the `rmhdvars` preset opens with none.
  FEEDBACK_2026-08-10 items 12/13 add the display card's colorbar and capture buttons: the
  three label conventions (signed +-max, magnitude 0..max, sigma fixed +-1) off a range
  handed in by hand, that a FIELD change drops the previous mode's range instead of
  relabelling with it, the readback gate (sigma modes take none), the strip following the
  card's colormap, its absence in the 3D lines view, and then `save` / `rec` end to end --
  filename pattern, a real PNG blob on a real `<a download>`, and the toggle's two states
  on BOTH recording legs. Leg 1 (WebCodecs): the async config probe gating the start, the
  encoder config, 45 pumped frames with forced keyframes at 0 and 30, timestamps an exact
  1/30 s apart, every `VideoFrame` closed, the drop-frame guard under a stalled encoder
  and the recovery after it, a downloaded `.mp4` whose bytes really are `ftyp+mdat+moov`
  with no `moof`, the 30 s hard stop fired by hand, destroy-mid-record, and the no-avcC
  bail to leg 2. Leg 2 (MediaRecorder, which the bail has just switched the page to): the
  vp9 pick and the 30 fps stream, the MP4 negotiation on an engine that offers it, that
  deleting `MediaRecorder` with WebCodecs off removes the rec button and leaves `save`
  alone, and that WebCodecs alone (no `MediaRecorder`: the iOS case this is all for) still
  offers it.
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
  it was written for. (3D: it puts the card on a PLANE first — since ISO_PLAN B a display
  card opens on the volume, which draws no plane and hence no contour overlay.)
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
  data), and the `modeHist` record through HIST_MAX halving and a paused clock. Sections
  11-12 (FEEDBACK_2026-08-10 items 9 and 15) add the island chart's `islandFitGamma` (the
  factor 2, that both wrappers ARE the one `fitLogSlope` at their own rise gate, that a
  MODE_FIT_DT window of the tearing stage clears ISLAND_FIT_RISE but not MODE_FIT_RISE,
  and the inherited saturated / decaying / trailing-window guards) and the (phi, psi)
  amplitude basis -- `icZetaFields(..., "pp")` normalizing the combinations exactly and
  independently on a MIXED drawing, bitwise agreement with the zeta basis on a phi-only
  one, and the `icAmpBasis` / label / `icAmpZeta` truth table on remembering controls.
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
  Section 4c runs the app's OWN `islandWidth` + `islandFitGamma` over that maintained run's
  cut line, at the cut card's cadence, against the same eigenvalue -- so the number the
  island chart's legend quotes is checked on physics, not on a synthetic exponential.
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
- `sigrcheck.js [dir]` — the residual-energy display mode (`sigma_r`, mode 9) in BOTH
  apps: boots rmhd2d and rmhd3d on the stub and records, off the stub device, the Mode
  uniforms each selected field writes (mode 9 + its pinned mate 5 = b, against sigma_c's
  8 + 7) and the whole encoded frame as an ordered bind-group list (identical to
  sigma_c's, no autoscale reduction, no arrow gather). Then it PARSES the emitted WGSL —
  prepDisp's mode -> potential branch chain, its -i*ky / +i*kx components, sigmaCombine's
  ratio and floor, colorize's fixed +-1 predicate — and drives an fp64 mirror of that
  parsed chain on an analytic (phi, psi), comparing against sigma_r computed directly
  from the analytic gradients. In 3D it runs the uniform + frame sections TWICE, once in
  the slice view and once in the cube-faces view (whose frame must be the same sequence
  through the face-sized instances: `magSqFA/FB`, `maxSumPartialFace`,
  `sigmaCombineFace`, `colorizeCube`), checks `colorizeCube` carries the same fixed ±1
  predicate as `colorize`, checks the face-sized `sigmaCombine` / `maxSumPartial` are the
  slice text with `NFACE` for `NRS` (so the quiet floor is the max over the three faces'
  own a+b), checks the mate uniform rides the card's z slice, and asserts `sliceExtract`
  / `faceExtract` are pure gathers — which is what makes the one-plane fp64 mirror the
  whole story in 3D too. It does not execute WGSL: what it pins is the wiring and the
  composition.
- `checkmp4.js [dir]` — the progressive-MP4 muxer (`mp4Mux`), driven with REAL H.264 and
  needing ffmpeg + ffprobe on PATH. ffmpeg encodes a synthetic Baseline stream at 30 fps
  with a keyframe every 30 frames; the script cuts the Annex-B into length-prefixed
  samples and builds the avcC from the stream's own SPS/PPS (the only Annex-B code in the
  project, deliberately in the test and not in the app), `mp4Mux` writes the file, and
  ffprobe/ffmpeg say what it is: top level exactly `ftyp+mdat+moov` with no slack, no
  `moof`/`mvex`/`trun`/`ctts`/`sdtp`/`co64` anywhere in the box tree, `stbl` complete,
  sync samples exactly on the forced indices, equal pts deltas of 1000 media ticks,
  `r_frame_rate 30/1`, size and codec preserved, and `ffmpeg -v error -f null -` decoding
  with zero stderr. Three cases — a square 512×512 take, the non-square 1024×256 wide box,
  and a one-frame file — plus the refusals (no samples / no avcC) and the codec-string
  level table. This is the gate that says the recording will play on a phone.
- `checkonepage.js [dir]` — the ONEPAGE_PLAN gates, on both booted pages. Phase A: the
  control panel hidden at boot, the params toggle's `localStorage` memory across two boots
  in one process (stubenv's store is per-PROCESS on purpose, so that IS a return visit),
  the guided-preset text still showing with the panel hidden, and `setRunning`'s
  flag/label/`.stop` coherence in both directions. Phase B: the shared tab strip, asserted
  as what a visitor sees (the current page a filled `<span>`, the other one an `<a href>`),
  the intro pane under the subtitle (open first visit, dismissal remembered, forced open
  on the no-GPU page), the what-is rail — the five panes, every one filled and COLLAPSED —
  plus a grep of all four html files proving the pane text lives only in `common.js`, then
  the no-WebGPU path driven for real through `stubenv(..., {noGpu: true})`: one `poster.png`
  node with alt text inside the display area, the browser advice moved under it with its
  `#status` id intact, every pane forced open, and NO poster node at all on a working boot.
  Run state: EVERY visit boots paused with the green "Run" label (the Phase C autoplay
  was reversed by Alfred 2026-08-10 — the hero button is the call to action), on the
  default preset at its ratified size; a later preset switch must leave the run state
  exactly as the user left it, paused or running, and the no-WebGPU boot starts no
  clock at all.
  Finally index.html's redirect, read as markup (there is no script to boot): meta refresh,
  canonical, `location.replace`, the no-JS link, and the two strings `pages.yml` seds.
- `checkiso.js [dir]` — the ISO_PLAN gates. Phases A (box-unit aspect), B (the volume
  raymarch) and C (the collision preset) plus the two
  disciplines the whole plan runs under: every emitted kernel of both pages parses
  (`dumpwgsl2` → `wgslparse.mjs`, with `WGSL_REFLECT` pointed at the local
  `node_modules`), `names.mjs` clean, `dup.py` showing no clone inside one file or
  reaching into the shared core, and every PHYSICS kernel BYTE-IDENTICAL to the plan's base
  commit — emitted from a `git show <base>:webgpu/...` checkout in a temp dir and diffed
  kernel by kernel, which is what "render-path only" has to mean. The DISPLAY kernels that
  moved, and the ones Phase B adds, are listed in the file and asserted to be exactly that
  list (a stale expectation fails too), and the three volume-length kernels are asserted to
  be the slice target's own template text at `NR`. The aspect legs boot
  the real rmhd3d page and drive the `L_z` select itself, so the whole path (select →
  rebuild → `cubeQuads` → `cubeFrame`) is measured: the z:x edge ratio tracks `Lz/Lx` at
  every option, y:x does not move, the 0.92 autoscale still fits the elongated box and
  stays tight against the canvas, and `cubeFrame` reproduces all 12 projected corners
  (so the field lines, the box wireframe and the arrow overlay ride the same projection).
  Both states of the panel's `true box shape` checkbox are driven: unticked, the drawn shape
  stops moving with `L_z` AND the twelve corners are the BASE COMMIT's corners float for
  float (the toggle really hands the old unit cube back), and ticking it puts the 4:1 column
  back under the same solver.
  A last leg makes the `ASPECT_CAP` on-device edit on a COPY of the page and boots that,
  so the display-cap switch is known to work before anyone flips it.
  The Phase B legs RUN the raymarch: `volRay`'s uniform is asserted to invert `cubeQuads`
  at all 12 drawn corners and to be orthogonal/unit-depth in BOX units (so the march's `t`
  is a true length), its slab test to reproduce the drawn box's silhouette and to enter
  through the three faces the cube draws; then the EMITTED fragment shader is executed by
  wgsl_reflect's WGSL interpreter (wrapped in a compute entry that calls `fs`) on a
  synthetic offset Gaussian blob and compared, pixel by pixel, with a CPU reference march
  written independently here (~1e-6 achieved, 1e-5 tolerance). Since Alfred's "rippling"
  report the same leg also marches a THIN-shell blob (a steep blob at a low level, where a
  coarse march would first alias) and asserts that the reference march at `VOL_STEPS` is the
  same picture as the same march at 4x that — the loop is CONVERGED, which is why the answer
  to that report was a measurement and not a dither. A last leg drives the same
  execution over every field-table entry — a negative blob must raise a shell exactly where
  the table says the field is signed — checks that the sigma modes fall back to the cube
  faces, and derives `prepDisp`'s two omega+- branches from its own vorticity branch.
  The Phase C leg boots the page at `?demo=collision` and asserts what the preset actually
  OPENS: ONE volume of j (resolved by the solver, not just by the select) carrying the
  preset's own level/opacity, on the SINUSOID packet IC at L_z = 8pi, with the chi line
  alive and the plane slider not merely dead but gone — and that the preset names no view at
  all, the volume being the default it inherits. Its other half is the promise that the
  preset only ever chooses among UNTOUCHED initial conditions: both packet ICs' own functions
  (`icPresetFields`, `icGaussZ`, `packetGeom`, `chiEstimate`, `icSinePlanes`, `icSineZeta`,
  `icZExtrude`, `icLetterZeta`, `applyIC`, `icInfoLine`) and the five one-line declarations
  around them are brace-matched out of both the working tree and the base checkout and
  compared byte for byte.
  The Phase D leg does for the k⊥ display filter what the Phase B legs do for the raymarch:
  it EXECUTES the emitted `prepDisp` (same interpreter) and mirrors it in fp64. The factor
  swept over k⊥ is the half-cosine band to ~1e-7, inside [0,1], monotone across each edge and
  EXACTLY 1 / EXACTLY 0 in the pass and stop bands; shell by shell on the page's own grid, the
  filtered field's energy is the unfiltered field's with the taper applied — shells strictly
  inside the band passing exactly, shells strictly outside exactly zero, on the spectrum
  kernel's own bins and in its own `INVKU` (band Parseval). Filter-OFF is proved bitwise three
  ways: the gated emission is base's `prepDisp` plus Phase B's four lines and nothing else,
  the executed kernel's output is bit-identical to the base kernel's on the same state, and
  the Mode uniforms the page writes over a sweep of modes / views / colormaps are the BASE
  page's bytes (booted from the base checkout, the same sweep) with two zero band words after
  them — that sweep now runs with the panel's `k⊥ filter` checkbox ticked and the handle at 0,
  since "the control is there and wide open" is the statement worth making bitwise. The UI
  side of the same leg follows the control as it is now: ONE handle per card writing
  `[k_min, 0]` (the high end is permanently the dealias cut, which is the kernel's "this end
  is off"), labelled and visible only while the checkbox is ticked, a caption reading
  `filter: k⊥ = 10:k_max` on the filtered card and nothing on the wide-open one, and — with
  the checkbox unticked — no handle, no caption, and zero band words on every uniform the
  card preps a field through. Later phases append to the `LEGS` list at the bottom.
- `check2dspec.js [dir]` — the ANISO_PLAN_2 gates: the generated E(k&perp;, k&#8741;) card, its
  band-gated `prepGrads` sweep (Phase A) and its coordinate binning kernel (Phase B). Seven
  legs. The emission first: every kernel parses, `names.mjs` clean, `dup.py` showing no
  clone inside a file, and both pages' WHOLE dump byte-identical to the plan's base commit
  with exactly two kernels ADDED (`prepGradsBand`, `specParBand`) and none removed — which
  is both halves of the plan's WGSL rule, the stepping `prepGrads` the RHS runs and the
  idle-path emission around it. Then the two new kernels as TEXT: each is the unbanded
  template's own lines plus an explicitly listed set (a line that changed and is not on the
  list fails, and so does a listed line that stopped changing), carrying prepDisp's band
  block and Mode struct VERBATIM — one half-cosine in the project, not three — measured in
  the spectrum chart's own k unit and binding `fields` read-only. That unit claim is made at
  TWO box sizes: every shipped preset fixes L<sub>x</sub> = L<sub>y</sub> = 2&pi;, so kunit
  is 1 and comparing conversions there is comparing the identity with itself — the same
  templates are therefore re-emitted (dumpwgsl2's own override argument) on a 4&pi; box,
  where a dropped or inverted conversion is a factor of two. State invariance is
  asserted five ways, because no sandbox makes them all at once: the two kernels EXECUTED
  over a band set leave the state array bit-identical word for word; the press on the booted
  page never writes, clears or copies INTO the state buffer (a full ordered buffer trace,
  named off `solver.buf`, with the row readbacks left REAL so it is the true encode path);
  the whole state buffer read back before and after the press is bitwise the same — the
  stub's buffers are given real backing stores for that, so `readBuf` really reads what was
  written; every bind group the sweep builds is checked BY NAME and in binding order, looked
  up by identity off `solver._gen` (`bgPrep` = fields, gridA, gradsK, genMode and `bgFL` =
  realGrads, genPos, genSmp, genCfg) — a group with `fields` in the read_write slot is legal
  WGSL, legal WebGPU and on a real device a press that OVERWRITES the state, and the other
  three legs are all blind to it; and the CONTENTS of the two uniforms it fills are read back
  out of those backing stores, the Mode words per band (`modeWords(0,0,0,[lo,hi])`, so
  swapped ends fail) and the coordinate pass's `(lo, hi, 0, 0)` band table.
  The fp64 mirror runs the emitted `prepGradsBand` on wgsl_reflect's interpreter:
  the factor swept over k&perp; against checkiso's own `bandFacJs` (~1e-7, in [0,1], monotone,
  exactly 1 / exactly 0 — and the same sweep again on the 4&pi; emission, so the band is
  measured in k/kunit as a matter of arithmetic and not just of a constant), all eight
  band-passed gradient lanes on the page's own grid, bit
  identity with `prepGrads` when both ends are off, and then the whole ROW — those gradients
  inverse-transformed by a direct fp64 DFT, sampled on the sweep's own 16&times;16 seed grid with
  the marcher's bilinear gather (&psi; = 0, so the lines are straight in z), fed to the app's
  real `flSpectrum` and compared with an fp64 Hann periodogram written from the definition.
  `specParBand` is executed too, one live perpendicular mode at a time — the interpreter runs
  a workgroup's invocations sequentially and treats `workgroupBarrier` as a no-op, so a tree
  reduction only survives it when one thread carries everything; on those states its rows
  match the fp64 mirror and its band-[0,0] row is the 1D `specPar`'s own bins BIT for BIT,
  which is Stage B's anchor. Ridge recovery drives the REAL press with a synthetic snapshot
  carrying a KNOWN k&#8741;(k&perp;): the real `gen2dBands` window off a real-shaped &perp; spectrum
  (log-spaced, octave-wide, overlapping, between `fitKA` and the shared `specKnee`), the real
  sweep, the real periodogram, and `gen2dRidge` landing within ONE k&#8741; bin of the law on both
  panels — with the coordinate one FLATTENING where the field-line one keeps climbing, which
  is the Cho&ndash;Vishniac contrast the card exists for; the same leg checks that the sweep
  still takes its extra UNBANDED pass (`parFL`) &mdash; which fed the measured overlay curves
  and, since Alfred's second feedback round dropped those, has no consumer left and is kept
  only pending a cleanup of the sweep. The choreography leg is Alfred's model as
  a sequence: press while running &rarr; paused, with the plot and its `generated @ t` legend;
  Run &rarr; unmoved; IC reset and full solver rebuild &rarr; still standing; second press &rarr;
  replaced; a press on a DEAD field &rarr; NOT replaced (an all-zero sweep is the null path,
  and the band set alone would not have caught it); no solver &rarr; button disabled and a
  forced press declines quietly. The last leg is the PLOT, both feedback rounds: the axis
  ends are the drawn columns' own edges and the k&#8741; floor is the forcing fundamental
  (tracked on a 4&times; box); the card's own GSW&times;GSH box makes the plot area square to
  the pixel; the three theory slopes are declared as 2/3, 1/2 and 1 under Alfred's GS95 /
  B06 / iso labels and drawn on ONE legend line with three swatches; the boundary per band is
  recomputed here from the raw rows and the floor, with planted sub-floor and 1.05&times;-floor
  cells that must be excluded by the noise margin; each slope is anchored on the boundary
  cell's TOP EDGE and is verified in pixels to clear the filled cells at every band and touch
  at one; the y axis is raised by the legend's own measured height, so no cell reaches the
  legend while every cell still sits exactly where the frame's map puts it; and the two
  measured anisotropy curves are gone, strokes, labels and call site alike. CI reports,
  never gates.
- `checkgc.js` — the ANALYTICS_PLAN gates: the GoatCounter beacon and the contact line.
  Three parts. TEXT over the five HTML files pins the **set of counted pages**, which is
  the load-bearing decision of the feature: the four content pages carry exactly one
  `data-goatcounter` each, all four endpoints identical, all four `async`, each with a
  self-referencing canonical (which is what stops `?fresh=<timestamp>` from the "load
  latest build" link and `?demo=` deep links scattering one page across unbounded distinct
  paths — `count.js` resolves the canonical and drops the query), and `index.html` carries
  NONE, because it is a pure redirect whose own canonical points at `rmhd2d.html`, so a
  beacon there would file every arrival at the bare `/webgpu/` URL under rmhd2d's path and
  inflate the one number the feature exists to produce. Then the ADDRESS: it must not
  appear contiguously in any file the deploy publishes — the walk covers all of `webgpu/`,
  `.md` plans and `devtools/` included, because `pages.yml` stages the directory
  wholesale, and a plan file spelling it out in full once defeated the entire
  runtime-assembly exercise while every narrower check stayed green. Then both apps
  BOOTED, each with and without WebGPU (`{noGpu: true}` — the fallback visitor is the one
  most likely to want the link): `#contact` holds exactly two anchors, the mailto carries
  the assembled address, the issues link has `target=_blank rel=noopener`, and the click
  handler — which builds the diagnostics on the way OUT, so a booted page names its GPU
  and a failed one honestly does not — yields a body with the build id, the userAgent, the
  viewport, and `webgpu: NO` with no `gpu:` line on the fallback. A pathological userAgent
  drives the 1500-char cap so the truncation branch really runs and the result must still
  decode: slicing an encoded query blind splits a percent-escape about one time in four.
  A text leg pins `contactBuild` being called before `chromeBuild`'s `if (!rail) return`,
  which no boot can reach. CI reports, never gates.
- `eqlinear.py [n]` — the linear reference for those rates: a 1D generalized eigenvalue
  solve of the linearized RMHD system on Fourier differentiation matrices at
  k_y = 2pi/Ly, plus a shooting solve for Delta'a. Prints the benchmark table checkj.js's
  REF block quotes (regenerate it there from this output), then eta- and b0-survey
  tables. `n` = Fourier modes, default 384; 768 reproduces every printed digit.
- `dup.py` — token-normalized >=10-line clone detector over common.js/physics.js +
  extracted app scripts (the standing-rule duplication audit). Run it over the extracted
  HTML *bodies* too: markup twins are what H.0 was about.
- `layout.js [dir]` — control-row wrap audit at 360/768/1200 px, off the BUILT element
  tree of a booted page (controls + every card header, and since items 12/13 the display
  card's `.viewfoot` too — the colorbar block is a fixed-width flex item and the save /
  rec buttons sit beside it).
- `names.mjs [dir]` — cross-file identifier resolution check (no redeclares, no frees);
  needs acorn (`npm i acorn`, or `ACORN=<path-to-acorn.mjs>`).
- `cmapcheck.js` — colormap table vs emitted WGSL vs matplotlib reference.

Conventions: run from any cwd with absolute paths; each phase captures a FRESH WGSL
baseline from clean git state before editing; refvector JSON lines in the HTML are
spliced programmatically only.
