# webgpu/devtools — sandbox verification tooling (REFINE_PLAN gates)

Node/python scripts built during REFINE_PLAN phases F–G to verify the apps without a
GPU. Saved here so later phases (and fresh sandboxes) don't rebuild them. Safe to
leave untracked or commit — they are dev-only, nothing in the apps loads them.

- `dispoffsets.js` — the two allowances the WGSL byte-identity pins share, so that
  `checkiso`, `checkeigf`, `check2dspec` and `checksolver2d` keep their older base commits
  and still fail on any OTHER change. (1) 268100f ("webgpu perp offsets") gave the 2D card
  its display offset, which moves the Mode struct line and inserts the translation phase:
  `applied(base)` is base's prepDisp text plus exactly that, so any other change to
  prepDisp — or the offset going missing — still fails. (2) FFTPERF_PLAN 2C turned one
  eight-lane `prepGrads` (and the sweep's banded twin) into four per-pair emissions:
  `gpairApplied(base, k)` is base's own text reduced to pair k — its source `let`s, its
  two writes, moved to lanes 0 and 1 — and `chunkAudit(base, cur)` spends that allowance
  exactly, demanding the eight-lane emission be GONE and all four pairs present and equal
  to the reduction. `checkiso` leg 2b is the allowance's own gate: BASE's text with a sign
  flip, a pair built from the wrong field, swapped lanes, the eight-lane emission left in
  place, a missing pair and a pair that is the whole text must each come back named.
- `stubenv.js` — the shared stub: a DOM + WebGPU stub good enough to run a real app
  page (the classic `<script src>` files the page's own markup names, in document order —
  common.js + physics.js, plus solver2d.js on the 2D page — then its inline script) under
  node, plus `run()` to keep
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
  The RESULT STRIP (2026-08-11) added the other half of that path: a byte-keeping `File`
  (a `Blob` with a name, and the bytes survive the rewrap, so what was SHARED can be
  compared with what was written), `navigator.canShare` / `navigator.share` logging every
  payload on `caps.shares` behind two knobs on `env.share` — `can` (an engine that cannot
  share files must simply grow no button) and `reject` (a named error, `AbortError` being
  the visitor closing the sheet) — and `Date`, because the MediaRecorder leg times itself
  by wall clock: `env.advance(ms)` moves the clock, so a "12 s" recording costs no wall
  clock either. Since the save path joined that strip (2026-08-12) `canvas.toBlob` hands
  back real BYTES as well — a PNG signature and a deterministic ramp, the size the strip
  quotes — because the picture is rewrapped as a File for the share sheet too, and "what was
  shared is what was written" is only a checkable statement when there is something to
  compare; its callback stays DEFERRED, as a browser's is, which is what lets a consumer
  close the card between the press and the picture and drive the dead-card branch honestly.
  Its 2D context logs both kinds of mark a composite is made of: `ctx.__draws` (every
  `drawImage` with its destination rectangle) and, since 2026-08-20, `ctx.__texts` (every
  `fillText` with the alignment, baseline and font in force) — which is what lets a gate
  assert a saved colorbar's tick GEOMETRY off the real file instead of calling the drawing
  code with a hand-built context.
  A fourth argument carries the boot knobs: `{noGpu: true}` removes `navigator.gpu`, so
  `initGPU` takes its first failure path and the no-WebGPU poster fallback runs for real;
  `{search: "?bench"}` sets the page's `location.search` verbatim (default: the third
  argument's `?demo=NAME`), which is how a gate reaches the URL-only developer flags the
  apps read off it — `?bench`, `?recdebug`.
  Since FFTPERF_PLAN phase 0 the device keeps an activity log on `env.gpu` (emptied by
  `env.gpuReset()`): `shaders` every `createShaderModule` as `{label, code}`, `dispatches`
  every compute dispatch as `{pipe, bg, d}`, and counters for render passes (`renders`) and
  queue drains (`drains`); a compute pipeline also carries its module's WGSL on `__code` and
  its label on `__name`, and a bind group its entries' buffers on `__buffers` — plus, since
  FFTPERF_PLAN 2C, each entry's whole `{buffer, offset, size}` on `__bindings`, with a
  misaligned (not 256-byte) or overrunning offset THROWN as a real implementation would
  reject it; a binding that is a WINDOW into a buffer is otherwise indistinguishable from
  one that binds the whole of it. So a gate can
  say that a caller's idea of a pipeline, an extent or a bind group is the one the app
  encoded, rather than that the call did not throw. `requestAnimationFrame` PARKS its
  callback (it always was inert); `env.frame()` fires the parked ones, which is one frame
  through the page's own `loop()`.
- `dumpwgsl2.js <dir> <page> "" <out.txt> ['{"pm":10}']` — emit every generated WGSL
  kernel to text for byte-diffing against a pre-phase baseline (capture the baseline from
  clean git HEAD first). `kdiff.py` diffs two dumps kernel-by-kernel. The optional JSON
  overrides every parameter set (`pm` = Pm = nu/eta, `eqsrc` = the maintained-flux
  source, `ny`/`Lx`/`Ly` = a rectangular 2D box),
  which is how a knob's kernel footprint is shown: dump twice, diff the two.
  Its preset list is every resolution the page's own `selRes` offers — 128/256/512/1024 in
  2D, 64²×32 through 256²×64 plus 64²×128 and 64²×256 in 3D — and it has to stay that way:
  a kernel that differs only at a line length the dump never visits (the 1024 row, the long
  z pass) is invisible to the byte-diff, whatever else is green. 155 kernels on rmhd2d,
  378 on rmhd3d.
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
  bail to leg 2. Since RECRAF_PLAN (2026-08-12) the frame loop is leg 1's real feeder, so
  three more legs drive `recCapture()` by hand the way `env.tick()` drives the timer (the
  stub `document.hidden` boots TRUE — an environment with no rAF loop is an honest
  "hidden" — which is why every OLDER leg above still exercises the timer path; round 2
  moved the watchdog's park condition from `lastRaf` freshness to visibility after the
  timing heuristic double-fed on a real phone). (a) rAF capture:
  35 direct calls encode 35 frames with ZERO renders of the card's own (a counting wrapper
  over `render()` proves it), forced keyframes at 0 and 30, timestamps an exact 1/30 s
  apart, every `VideoFrame` closed, the slots the clock jumped over COUNTED into `W.drop`
  rather than backfilled, a capture with `due` pushed to `Infinity` encoding nothing and
  dropping nothing while still moving `lastRaf` (the ?recdebug gap diagnostic), and the
  file's own `stss` read back as the 1-based `[1, 31]`; the `rafN`/`wdN` feeder tallies
  read 35/0. (b) watchdog handoff: 3 rAF frames, then `env.tick(5)` with nobody calling
  `recCapture` — the timer carries the same frame index on to 8, re-bases `due`, and the
  tallies split 3/5. (c) watchdog parked: with `document.hidden` flipped false (a visible
  page — restored true after) five timer ticks encode nothing at all, which is the double
  feed the iPhone stutter came back worse from. RECASYNC (2026-08-12) added the BUFFER
  CAPTURE legs on top: the stub gained `copyTextureToBuffer` (geometry-checked, logged on
  `caps.copies`), a `VideoFrameStub` that takes `(BufferSource, init)` and records
  format/dims/kind, `GPUTextureUsage` constants, and a hand-driven map pump —
  `env.holdMaps(true)` parks every `mapAsync`, `env.maps(rev, n)` releases them (in
  REVERSE arrival order if asked, which is how the ordered-chain claim is falsified),
  `env.mapsPending()` counts them, and `env.bufFrames(true)` filters the built
  VideoFrames to the from-bytes kind. The stub's bytes-capability boots OFF (the plan's
  fallback option: the OLD legs run the canvas path untouched, and each new leg arms the
  probe itself) — the knob and the reasoning live at the `bufArm` helper in bootstub. Its legs: a 35-frame take entirely FROM BYTES (0
  canvas frames, `vf`/`enc`/`lag` wired, stss `[1,31]`); maps resolved in reverse still
  yielding a monotonic fixed-step timestamp series; pool exhaustion (3 in flight → 4th
  slot dropped, pool reused after); padded rows (a 500 px canvas → bpr 2048, rows
  compacted) and a mid-take canvas resize dropping the slot; stop draining in-flight
  captures into the file; a hung map timing out at 500 ms with the pool destroyed and
  late maps inert; and the probe-fail fallback running the old canvas path with no
  copies, no pool, `lag` 0. Leg 2 (MediaRecorder, which the bail has just switched the page to): the
  vp9 pick and the 30 fps stream, the MP4 negotiation on an engine that offers it, that
  deleting `MediaRecorder` with WebCodecs off removes the rec button and leaves `save`
  alone, and that WebCodecs alone (no `MediaRecorder`: the iOS case this is all for) still
  offers it. Since the RESULT STRIP (2026-08-11) every stop on both legs is asserted to
  download NOTHING by itself and to leave one `.recres` on the card's footer instead: the
  size and length text computed here from the stubbed bytes and the pumped frame count
  (leg 1 quotes the frames it MUXED, so the drop-frame guard shortens it; leg 2 quotes
  `env.advance`'s wall clock), the download button yielding the same `ftyp+mdat+moov`
  bytes as before, a share button present exactly when `env.share.can` says the engine can
  share files, `navigator.share` handed a File with the right name and byte-identical
  contents, a `NotAllowedError` rejection falling back to a download while an `AbortError`
  does not, dismiss removing the strip with a second dismiss inert, and a new take
  replacing the old strip rather than stacking a second one. Destroy-mid-record is the one
  path that still downloads directly, and is asserted to.
  The SAVE path joined that strip in round 2 (2026-08-12) and is asserted the same way, with
  the differences that matter: the press downloads NOTHING, the line quotes the stubbed
  picture's SIZE and nothing else (a PNG has no length, so no seconds and no separator left
  hanging), the download button yields the same name / type / bytes the old direct download
  was checked on — the signature included, now that the stub writes one — the share button
  follows `env.share.can` and is handed a File of type `image/png` carrying the picture's own
  bytes, `NotAllowedError` falls back to a download and `AbortError` does not, and a second
  save replaces the first. The card is closed BETWEEN the press and `toBlob`'s deferred
  callback for the dead-card branch, which downloads on the spot. Then the two SLOTS, driven
  on leg 2: a picture survives a whole take including the take's own replace-on-start, a
  recording survives a save, each kind replaces only its own strip (node identity, not just
  the count), and each dismiss leaves the other standing. The capture GROUP is read off the
  footer too — `save` and `rec` children of one `.capgrp` and neither a loose child of the
  footer — and again on the card an engine with no recording leg builds, where the group must
  still be there around the one button that is left.
  TEARNL (2026-08-13) added the tearing FAMILY as one table — `kh`, `collapse`, `chain`,
  `tearing` in that order, because `chain` is the only preset carrying the broadband seed and
  that order runs the leak in the direction that can fail. Each is asserted on its grid, its
  shell count, its display card (`chain`'s is the 128 × 512 one; `collapse`'s square card
  leaves the wrapper's aspect-ratio EMPTY, by the setter's own rule), the hyper LOCK engaging
  for the three resistive presets and releasing for the ideal one, the maintained-flux source,
  the rows `icSyncRows` shows, and — the point of the sweep — the state of `#cbTearBroad`,
  which `presetWrite` only writes for the presets that NAME it, so a preset staying silent
  about it would inherit the last one's seed and silently stop being the run it is quoted as.
  Then the `tall` box itself: ISOTROPIC cells (dy > dx would leave the merger sheets
  unresolved, which is the whole reason the box exists), `kunit` = 1/4 following the box, and
  a rebuild in and out of it driven from the box select alone. Then Phase 1's IC-BUILDER
  change, the one thing here that is not a UI concept: `icPlaneFromX` with no y-factor is
  asserted BITWISE against the literal `cos(k_y y)` broadcast it always was, and the broadband
  factor against its own definition — the DERIVED mode count (24 at `chain`'s sliders, from
  the marginal k_y a), a maximum of EXACTLY 1 (which is what keeps `#rEqPert` the same
  physical quantity in both branches), a DFT flat over exactly modes 1..N and zero outside,
  the same seed reproducing the same factor, and `#nSeed` really changing it.
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
- `checkeigf.js [dir]` — the EIGF_PLAN gate: the 2D eigenfunction chart card
  (|ψ̂(x,k_y)|, |φ̂(x,k_y)| against x). Six sections. The discipline first — every emitted
  kernel parses, `names.mjs` clean, `dup.py` showing no clone reaching into the shared
  core, every kernel that existed at the base commit byte-identical — but for `prepDisp`
  on the 2D page, which the base predates the display offset by (`dispoffsets.js`, 268100f),
  and which must carry it at EVERY preset — with the additions
  being EXACTLY `[eigfGather]` on the 2D page and nothing at all on the 3D one, and the RNG
  reference (64 `Gauss(7)` draws, hashed) unmoved against a value recorded from the base
  tree before a line was written. Then the gather kernel EXECUTED (wgsl_reflect's WGSL
  interpreter) against the CPU-side strided column at five k_y bins — a BIT compare, the
  path being a pure copy — plus its out-of-range clamp. Then the fp64 mirror: a field whose
  k_y coefficient is known analytically (ψ = g(x)cos(k_y y+p), φ = h(x)sin(k_y y+p), the
  two 90° apart in y, which is why the card plots moduli), with the state built by a direct
  fp64 forward DFT written from the definition, so gather + `eigfProfile` is compared with
  analysis and never with itself: |ψ̂| = g/2 and |φ̂| = h/2 to ~1e-8, ψ̂ peaked ON
  x = L_x/2, φ̂ zero there with a lobe either side, every other column empty, and
  `eigfProfile` itself against a direct fp64 inverse DFT (the leg that still runs where
  wgsl_reflect is not installed). Then equilibrium exclusion — a y-independent ψ_eq is
  entirely in the k_y = 0 column, every other column of it zero to round-off, which is what
  makes "minus the equilibrium" free rather than a subtraction anyone performs. Then state
  invariance three ways: the executed kernel's input array bit-identical word for word, the
  emitted WGSL's access qualifiers (`fields` read-only, exactly one `read_write`), and the
  booted page's encode path traced buffer by buffer — `readEigf` writes only its k_y
  uniform, runs only the gather, copies only out of the column buffer, and its bind group
  is `(fields, eigfU, eigfK)` in that order, recorded off a second solver built with
  `createBindGroup` patched. Last the card: its `CHART_TYPES` entry sitting immediately
  after `mode` (the plan's own placement rule, so a concurrent edit inside `aniso` stays a
  disjoint hunk), 2D-only availability, its two options and their defaults, the hint's
  claims — including that it says OUTER solution and promises no resistive layer — the
  readback pool splitting when one card's k_y selector moves, `cardsThrottleReset`, 25
  (data × options) degenerates through a recording context, the `tearing` preset opening
  with it and no other preset doing so, and the plan's side task: the stale "30–40% below
  the reference" clause gone from the tearing hint with the diffusion statement it hung off
  still standing. CI reports, never gates.
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
  and a one-frame file, and (IO_PLAN item 5) the 1024×512 COMPOSITE frame of two 512²
  tiles — plus the refusals (no samples / no avcC) and the codec-string level table,
  which now includes the two composite sizes. This is the gate that says the recording will play on a phone.
- `checkrecall.js [dir]` — multi-display recording (IO_PLAN item 5), on both booted pages.
  Section A is the tiler and the compositor alone, with no page, GPU or encoder involved:
  the row/column choice (a tie goes to the row), the wide-box column, mismatched card
  sizes refusing to tile rather than scaling one, the halved fallback halving the TILES
  once, and then synthetic byte patterns composited with **every output row read back at
  its expected offset** — both layouts, padded and tight source rows, the point-sampled
  half size, the one-tile fast path returning the mapped range itself, and the label patch
  landing at its tile's inset and clipping at the frame edge. Section B drives the action:
  the offer rule (one card → not offered, two and three → offered with the composite size
  in the title, no WebCodecs → absent and the per-card `rec` untouched), a two-card take
  from the first `copyTextureToBuffer` to a downloaded `ftyp+mdat+moov`, the pool as
  REC_POOL SLOTS of one buffer per source, every source rendering on the loop's pass, and
  the four failure modes: an all-or-nothing slot driven twice (one source resized under
  the take, then one source's map REJECTED — each time the whole slot goes, `W.n` does not
  move and no half-captured slot enters the pool), the file's `stts` read back as a
  SINGLE uniform run afterwards, the `isConfigSupported` refusal halving the tiles with
  the card list intact and the strip saying `tiles at half size` (and a refusal of both
  sizes starting nothing and saying why), a source card closed mid-take ending the take
  with the file on the PAGE's strip, and the single-card `rec` path asserted unchanged —
  one tile, its own canvas size, its own name, no label, its file on the card's footer.
- `checkchartsave.js [dir]` — the chart card's `save` button (IO_PLAN item 2), on both
  booted pages. Every leg asserts on the FILE, never on a handler having run: the press
  produces a PNG whose bytes are the chart canvas (the stub sizes a PNG as 4·w·h of the
  canvas that wrote it), it lands on the card's own result strip with download / share /
  dismiss, and **nothing is downloaded by the press** — the house rule the chart card had
  no footer to honour. Then the four traps: the `bar` type (`gen2d`, 3D only) composites
  onto a canvas taller than its plot with BOTH the plot and the colour-scale canvas drawn
  into it (stubenv logs every `drawImage`) in a band of the shared `cbarDraw` geometry;
  the name is `taranis-<app>-<chart>-t<simT>.png` and the download hands back the same
  blob that was captured; a retype — including `gen2d → energy → gen2d` — keeps the
  waiting strip and puts the rebuilt scale back **above** it; and a card closed between
  the press and the deferred `toBlob` downloads its picture instead of appending it to a
  dead node. It also pins the capture/deliver split both card classes now expose
  (`captureShot()` resolves to the blob and delivers nothing — what save-all will use) and
  that the display card's own save still goes through the shared strip.
  **The label geometry (2026-08-20).** `cbarDraw` is the one piece of arithmetic item 2's
  refactor owns and nothing pinned it: the strip's rectangle, the tick baseline `3*sc`
  under it, the left / centre / right anchors on its own edges and the `9*sc` monospace
  were all free to move in every saved picture at once. stubenv now logs every `fillText`
  with the alignment and font in force (`ctx.__texts`), and both callers are asserted
  through the REAL file — the chart card's band under the plot and the display card's
  stamp over the field, which is the identity that makes `cbarDraw` one function rather
  than two. The gen2d legs also run BOTH ways round: with a panel on the card its three
  labels are real numbers in the composite, and with none (before `generate`) the save
  composites **no bar at all** rather than a ramp under three blank labels claiming a range
  the picture does not have. The panel is injected directly — a real generate sweep belongs
  to `check2dspec`.
- `checkzip.js [dir]` — the stored-ZIP writer, the save-all button and the field export
  (IO_PLAN: the writer, item 3 and item 4, sections A, B and D). Section A
  drives `zipStore` inside a booted page and reads every archive back with **python's
  `zipfile`** rather than with our own parser — `testzip()`, stored method, per-member CRC
  recomputed from the extracted bytes, names, order and header offsets. The adversarial
  cases are the ones that hide a format bug: an empty archive (22 bytes of EOCD, nothing to
  hide an error behind), a single member, a zero-length member, a non-ASCII name (UTF-8,
  general-purpose bit 11), and a member past 64 kB with a member after it — which is where a
  field written at the wrong width shows. `crc32("hello") == 0x3610a686` pins the reflected
  polynomial, and the offset walk pins byte offsets against member indices. Section B is the
  button on both booted pages, asserting on the FILE: one archive, one strip slot on the
  page's own `viewfoot`, and **zero `<a download>` clicks at save time** — the multi-download
  burst the archive exists to replace. It pins the member names and card order, the
  `params.json` manifest parsed out of the real archive and compared field by field with the
  page's own `liveParams()` (then again after forcing is switched off and the diss slider
  moved, so a manifest frozen at boot fails), and the **capture ordering**: every card's
  `captureShot` and every display card's `render` are instrumented, `saveAllZip()` is started
  and deliberately not awaited, and the log is read back in the same task — so an `await`
  between two cards fails here rather than silently capturing an expired texture.
  Section D is the field export (`.npz`) on both booted pages, read back by **`numpy.load`**
  — the reader the file exists for — as well as by `zipfile`. Its two load-bearing legs are
  the ones nothing inside the page could check for itself. **Axis order:** the export's two
  staging buffers are filled with the FLAT BUFFER INDEX through the real button (they are
  the only `MAP_READ` buffers the page creates once the leg is armed, so creation order is
  phi then psi), so the array numpy should see is exactly `arange(n).reshape(shape)` in C
  order and every element is compared against one numpy builds itself — a transpose, a
  `fortran_order` flag or a swapped shape tuple fails on nearly every element, with a
  readable probe at an index deliberately NOT symmetric in x and y printed beside it. **The
  pinned uniforms:** a chain is built ON PURPOSE with `queue.writeBuffer` traced, so the
  third and fourth Mode uniforms are caught being pinned to plain phi / plain psi with band,
  offset and colormap all zero; then two cards are loaded up with non-default modes, a
  k⊥ band and (2D) a display offset, every Mode-uniform write is traced across an export,
  and the export must produce **none** — which is the leg that catches an export that
  overwrote `B.mode`. The rest: the member list and order, `<f4` / C-contiguity / shape,
  the coordinate vectors exactly `i*L/n` on every axis, the `.npy` header (v1.0 magic, data
  on a 64-byte boundary, `fortran_order: False`, and both shape-tuple spellings — a 1-tuple
  carries the trailing comma), the manifest's sim time, axis order and dealiasing note, one
  strip slot of its own beside save-all's, and the memory rule: after a multi-MB read
  `_stagePool` holds no entry that big and the stub's live buffer count is exactly back
  where it started, so the staging buffer was destroyed rather than pooled.
  **Two readers, not one (2026-08-20).** `zipfile` reads the CENTRAL DIRECTORY and nothing
  else, and it ignores `compress_size` on a stored member entirely — so a writer declaring
  0 compressed bytes in a local header passed every leg above while Info-ZIP refused the
  archive outright (`ucsize 11 <> csize 0 for STORED entry`, then a bad CRC). Three things
  close that: `compress_size == file_size` is asserted per member; the python reader now
  also unpacks each 30-byte LOCAL header by hand and every field is compared against the
  central directory's copy (version, flag, method, CRC, both sizes, name, DOS stamp); and
  every archive that matters additionally goes through **`unzip -t`**, a second reader with
  different blind spots — skipped with a printed SKIP, never a silent pass, where the
  binary is absent. Section A also pins the writer's INPUT contract: a `Float32Array`
  member is written as its bytes (12, not 3 value-converted ones — a corruption no reader
  could ever flag, since the CRC would match), a string / plain array / number is refused
  rather than encoded on a guess, a bare `ArrayBuffer` is taken, and a member name past the
  16-bit name-length field throws on its UTF-8 BYTE length (40000 two-byte characters is an
  80000-byte name) while a 60000-byte one is still written. Plus version-needed 2.0 and the
  DOS date/time stamp against the PAGE's clock (stubenv stubs `Date`), the save-all
  re-entry guard (two presses in one task ⇒ exactly one `zipStore` call and one strip row),
  and the manifest's expression IC: with `preset: "expr"` both formulas are in
  `params.json`, read out of the real archive — the word `expr` alone names a run nobody
  could reproduce.
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
  Phase 8 (2026-08-13) sweeps every quoted local asset path out of `*.html` AND `*.js` —
  poster.png is named only by `common.js`, so markup alone is not the whole site — and
  asserts each one both EXISTS and is TRACKED BY GIT. A path counts only where something
  LOADS it (2026-08-20): a markup `src=`/`href=` attribute, or a JS literal that reaches
  `.src`/`.href`/`fetch()` directly or through the constant it is bound to, which is how
  poster.png is reached. A filename the page WRITES rather than reads is not an asset —
  `params.json` is a member inside the archive the exports build (IO_PLAN), and no deploy
  can drop a file that only ever exists inside a download. The second leg is the point:
  `pages.yml` deploys `cp -r webgpu/.` from a fresh clone, so an uncommitted file is a
  404 on the deployed site while `file://` still works locally. That is exactly how the
  favicons shipped broken — `.gitignore`'s blanket `*.png` swallowed them, poster.png
  being the only exception — and an existence-only check would have passed.
  **Known limit of that sweep (noted 2026-08-20).** "Something loads it" is decided by the
  TEXT immediately before the literal — an attribute name in markup, or `.src =` /
  `.href =` / `fetch(` / a `const` whose name reaches one of those in the same file. Any
  other way of naming a file is invisible to it: `setAttribute("href", …)`, a path
  assembled from pieces or built in a template literal, a name crossing files through
  anything but a bound constant, or a loader idiom not in that list. This is deliberately
  a whitelist — counting every quoted string that looks like a path would flag
  `params.json` and every other name the page WRITES rather than reads — so its failure
  mode is a new asset deploying unchecked, never a false alarm. Widen the two regexes when
  a new loader idiom appears; do not widen them to "any string with a dot in it".
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
  be the slice target's own template text at `NR`. The band's gated-off leg compares against
  base plus the 2D display offset (`dispoffsets.js`, 268100f) and nothing else. The aspect legs boot
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
  — but for the 2D `prepDisp`, which the base predates the display offset by
  (`dispoffsets.js`, 268100f), and which must carry it at every preset —
  with exactly two kernels ADDED (`prepGradsBand`, `specParBand`, plus EIGF_PLAN's
  `eigfGather` on the 2D page, 1b57875) and none removed — which
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
  makes exactly ONE `readGenBand` call per band and ships no unbanded row: the extra
  `parFL` pass, orphaned when Alfred's second feedback round dropped the measured overlay
  curves, was removed by the render audit (2026-08-12), and the leg pins both halves of
  that &mdash; the field is gone AND the pass is gone AND the button's progress `total` is
  the band count, since dropping the field alone would have saved nothing. The argmax
  ridge the recovery leg measures with (`ridgeOf`) is this file's own now; it had no
  runtime consumer in the app. The choreography leg is Alfred's model as
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
- `checksolver2d.js [dir]` — the GAME_PLAN Phase 0 gate: the 2D solver moved out of
  `rmhd2d.html`'s inline script into the shared classic script `solver2d.js`, which
  `game.html` will load beside it. Phase 0 is a PURE refactor, so most of the file asserts
  that nothing moved. Eight legs. The discipline first — every emitted kernel of both pages
  parses, `names.mjs` clean, and `dup.py` over the THREE shared files plus both inline
  scripts showing no clone inside a file, none reaching into common.js/physics.js, and none
  between `solver2d.js` and the page it came out of (a copy left behind). Then WGSL
  byte-identity against the base commit: every kernel that existed there is unmoved, the
  additions are EXACTLY nothing on BOTH pages, and each whole dump hashes the same. Then the
  RNG reference (64 `Gauss(7)` draws, hashed — checkeigf's value, one reference, two gates).
  Then the extraction's SHAPE: the three definitions gone from the inline script and the
  only top-level ones in `solver2d.js`, exactly one `<script src="solver2d.js">` tag sitting
  after physics.js's and before the page's own script, `rmhd3d.html` not loading it at all,
  and no `type="module"` / `import` / `export` anywhere in the three js files or the two
  inline scripts (Chrome blocks module scripts from `file://`). Then the move itself,
  VERBATIM: each definition's text — located by its header line and its closing brace at
  column 0 — byte for byte against `git show <base>:webgpu/rmhd2d.html`, `class Solver`
  and `buildShaders` after their RECORDED allowances are stripped (`IO4_INSERTS`: the exact
  lines IO_PLAN item 4's field export inserted — two pinned Mode uniforms, their two
  `prepDisp` bind groups and `encodeExport`, buffers and bind groups only, which is why the
  WGSL leg above does not move with it; `FFT2C_SOLVER` / `FFT2C_SHADERS`: FFTPERF_PLAN 2C's
  gradient chunking, recorded as REPLACEMENTS — the base lines and the lines that stand in
  for them — the two-lane `gradsK`/`specTmp`, the four `prepGrads` pipelines and their bind
  groups, the row kernel's per-pair `realGrads` window, `encodeGrads`, and the four
  emissions in `buildShaders`). The allowances are the `dispoffsets.js` idiom: they are put
  back and the base text is then still demanded byte for byte, so any OTHER change to
  either definition fails, and a companion leg fails it as STALE if a recorded block has
  stopped being there or moved off its recorded base line, or as VACUOUS if there was
  nothing to strip. Then reusability:
  `names.mjs` resolves `solver2d.js` against common.js + physics.js + builtins ALONE, never
  against `rmhd2d.html`'s inline script, which is what says a second page can load it. Then
  `pages.yml` — solver2d.js in the MISSING list and in the cache-bust sed, and the
  `game.html` prune pre-wired. Last, both pages booted through `bootstub.js` to the end of
  the self-test path. CI reports, never gates.
- `checkidle.js [dir]` — the RENDER GATE (audit of 2026-08-12), on both booted pages. It
  drives `renderCards(paused)`, the app's own per-frame display step (split out of `loop()`
  so it can be called at all — the stub's `requestAnimationFrame` is a no-op), and counts
  the chains actually encoded by wrapping `Solver.render`, so what is measured is work
  done and not the gate agreeing with itself. Every leg that asserts a SKIP is paired with
  one that asserts a DRAW, because a gate that only ever answers "no" would pass the first
  half and ship a frozen app: the first frame after boot draws every card and the next two
  draw nothing; a card's own control, a step, `chartsReset`, `cardsSync` and a newly ADDED
  card each re-open it; the IC editor's view draws none; a live take on either leg forces a
  draw over an unchanged state (the encoder reads that render's texture); a paused frame
  writes the zero contour range for every ACTIVE set and a running frame for none (settle
  versus relax); and `stateMark()` moves on a step, moves again on a state jump that takes
  no step (an IC upload resets `nsteps` to 0, so the count alone would repeat), and is
  forgotten by `cardsThrottleReset` — which is what feeds a chart card added while paused.
  Since the adversarial review it also drives the 3D hooks, which is where the one real
  defect of that round lived: a lines-view card, then a k∥ chart, then a SECOND lines card,
  each arriving over a state that never moves, must each be fed — and once all three are,
  three more hook calls must march nothing. Reverting `flStale()` to the shipped
  `&& flData` fails the middle two legs, which is the check this leg exists to be.
  Also driven: leaving the IC editor by save / cancel / run, exactly one arrow gather after
  the last frame drawn (and none after that however wide the throttle is), the spectrum
  marker suppressing and `cardsThrottleReset` re-permitting, and `recCapture` being called
  on every card every frame whether or not that card drew.
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
  appear contiguously in any file under `webgpu/` — the walk covers `devtools/` and the
  SPEC/README too, which is deliberately WIDER than what ships: `pages.yml` stages the
  directory wholesale and then prunes the dev-only files, and the pruning list is a thing
  that can be edited, so the walk does not depend on it. The reason it walks at all is
  that a plan file spelling the address out in full once defeated the entire
  runtime-assembly exercise while every narrower check stayed green. (Those plans now live
  in `plans-webgpu/` at the repo root, out of the copy's reach entirely.) Then both apps
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
- `checkoff.js [dir]` — the 2D DISPLAY OFFSET gate (the per-card x/y sliders that roll the
  picture inside its frame). Four legs. EMISSION: `shift` is a pure insertion into the band
  text — the struct's `bpad` -> `sh` rename plus one contiguous eight-line phase block, so a
  constants object without it emits the pre-offset kernel — and across the whole page exactly
  ONE emitted kernel mentions `md.sh`, `cutPrep` (the cut chart's line at the real x = Lx/2)
  emphatically not among them. LAYOUT: `modeWords` writes the offset into words 6-7 and leaves
  the mode / cmap / band words and `MODE_BYTES` alone. ARITHMETIC, by RUNNING the emitted
  prepDisp (wgsl_reflect's interpreter, as checkiso runs the raymarch) against an fp64 mirror:
  the factor is exactly e^{-i k.S}, it is a phase (every mode keeps its modulus), a whole-cell
  offset really is a ROLL of the real-space picture in the direction the tooltip claims (a CPU
  inverse transform in the app's own convention: +x -> +ix, rightward; +y -> +iy, downward),
  a whole box is the identity, the vector modes' second output takes the same phase, and OFF
  IS BITWISE OFF — at [0, 0] the output is bit-identical to the un-gated kernel's. Then the
  PAGES, booted: 2D has both sliders, visible with no panel gate, centred, labelled and
  travelling ±0.5 by 0.01; the words the page actually writes are the card's fractions times
  the BOX on every mode buffer of its chain (the sigma mate and both contour potentials, which
  is what rolls the contours WITH the field); a centred card writes exact zeros and captions
  nothing; a second card is untouched; a value forced past either stop is clamped. 3D has no
  sliders and no kernel that knows the word, its prepDisp still saying `bpad`.
- `eqlinear.py [n]` — the linear reference for those rates: a 1D generalized eigenvalue
  solve of the linearized RMHD system on Fourier differentiation matrices at
  k_y = 2pi/Ly, plus a shooting solve for Delta'a. Prints the benchmark table checkj.js's
  REF block quotes (regenerate it there from this output), then eta- and b0-survey
  tables. `n` = Fourier modes, default 384; 768 reproduces every printed digit.
- `checkexpr.js [dir]` — the IO_PLAN item 1 gate: the expression IC. Eight sections, both
  pages booted, because the name set differs between them. The parser first, in units —
  precedence and the two readings the plan pins by name (`^` right-associative and binding
  tighter than unary minus, so `-x^2` is `-(x^2)` and `2^3^2` is `2^9`), asserted on the
  emitted RPN and not only on its value; then every error path, each checked to RETURN
  rather than throw and to point at the right character (the message quotes a 1-based
  position, `at` carries the 0-based index it came from); then `z` / `Lz` rejected on the
  2D page and resolving on the 3D one. Evaluation is compared against the index formula
  `(iz*nx + ix)*ny + iy` written out longhand HERE, never read back off the thing under
  test. The non-finite guard is driven with `1/x`, `log(y)` and `sqrt(-1)`, counting the
  offenders and locating the first. The periodicity detector gets the plan's three cases
  (`sin(2*pi*x/Lx)` periodic, `x*y` not, `abs(x-Lx/2)` continuous but kinked at exactly
  the analytic `2h`) plus the one that decided the algorithm: a COSINE is even about the
  seam, so a slope test built from one-sided differences taken INSIDE the grid reads
  `h^2 f''` there and calls a perfectly smooth field kinked — the check pins it at
  round-off over an nz sweep, which is only true because the expression is evaluated one
  cell OUTSIDE each face. Section 7 drives the REAL app: the preset selectable, both boxes
  and the help line shown, the amp rows left hidden, a typed expression reaching
  `setICFromReal` with the right values, `1/x` arriving as `null` instead, the warning line
  naming the seam it failed on, and the line dying with its preset. Section 8 times one
  build at 1024² and at 256²×64 — and measures the HARNESS first: stubenv's `Math` belongs
  to the outer realm, so every `Math.sin` is a cross-context call ~10x a native one, which
  is most of what the raw number contains. The interpreter's own dispatch rate is timed
  separately on a transcendental-free program.
- `dup.py` — token-normalized >=10-line clone detector over the shared core
  (common.js/physics.js/solver2d.js) + extracted app scripts (the standing-rule
  duplication audit); callers pass the file list explicitly. Run it over the extracted
  HTML *bodies* too: markup twins are what H.0 was about.
- `layout.js [dir]` — control-row wrap audit at 360/768/1200 px, off the BUILT element
  tree of a booted page (controls + every card header, and since items 12/13 the display
  card's `.viewfoot` too — the colorbar block is a fixed-width flex item, and since
  2026-08-12 `save` and `rec` are ONE item, the `.capgrp` pair being measured as the sum of
  its visible buttons and its own gap: they must never wrap apart, so the pair is what has to
  fit, and the CSS is checked to say `nowrap` / `flex: 0 0 auto` for that measurement to be
  honest. The result strips are audited as rows of their own: they only exist after a save or
  a take, so the script hands a card a finished file of BOTH kinds through `recResult` (two
  files, two slots, two rows) and measures the widest version of each (download + share +
  dismiss). Since IO_PLAN item 2 a CHART card has the same footer and the same strip, so it
  is handed a picture too — on a `bar` type, where the strip shares a footer with a colorbar
  — making three strip rows per page; its `save` button lives on the card HEADER (a chart
  has no caption line), so it is measured among that row's items and its presence on every
  chart card is asserted.
- `checkbench.js [dir]` — the FFTPERF_PLAN phase 0 gate: the `?bench` harness and
  `fftKernel`'s probe seam. Five legs. The FLAG gates the panel — no `#benchpanel` element
  and no `window.bench` without it, both on both pages, with `FFT_PROBE` null on a page
  that has just booted — and with it every campaign is driven under the stub; one JSON
  record per run must land in the textarea carrying the page, the GPU, the resolution, the
  bytes/butterflies per step and, per FFT kernel, the three ladder times with their shares.
  Then the WIRING, off the stub's dispatch log: for every spec cell, the pipeline and bind
  group it names must be one the solver's own `step()` dispatched, at the spec's extent and
  lane count, with the FFT cells' `bufs` equal to that bind group's buffers in binding
  order — and every byte-table entry dispatched exactly `n` times (a spec name may stand for
  a GROUP of pipelines built from one template, as the four `prepGrads` do, and the group is
  counted together), with nothing dispatched
  the table has neither counted nor excused and each excused kernel dispatched exactly as
  often as the table assumes. Since FFTPERF_PLAN 2C the same leg pins the chunked chain's
  WIRING, which nothing else can see: its four row-kernel dispatches bind `realGrads` at
  `2·k·nr·4` with `size 2·nr·4` — four windows into one buffer, in pair order, the stub
  keeping each binding's `{buffer, offset, size}` and rejecting a misaligned or overrunning
  one as a device would — each behind its own `prepGrads<k>` pipeline, four distinct
  pipelines and not one dispatched four times. The `grads hash` cell, which encodes exactly
  one chain, is held to the same four windows and four pipelines.
  NOTE that the per-kernel `prepGrads` / `colsInv` / `rowsC2R` cells now time ONE two-lane
  chunk, dispatched 12 times a step, so they are not comparable with the Phase 1 rows (which
  timed the eight-lane form, 3 times a step) without a factor of four. Then the LOOP: with frames driven from inside
  the queue drains, a campaign must encode no display chain, read no stats, and leave the
  hero button paused — and the next frame after it must draw again. Its second half is the
  frame ALREADY in flight when the button is pressed: a frame is parked on its own drain
  (its render, stats readback and frame hooks still to come), the campaign is started, and
  the campaign must not step until that frame has finished (`loopBusy` at its first step)
  and must keep the frame's tail out of every timed rep. Each campaign that trampled the
  fields must also restore through the page's own `applyIC`, not a bare `setIC`. Then the REPS: one cell
  drains R+1 times and reports the median and min of what it kept, against a clock whose
  first rep is the cheap one, so a kept warm-up rep shows up as the min. Then the EMISSION:
  `fftKernel` / `fftRowPair` with no probe are byte-identical to
  `fixtures/fftkernel_f83386e.json` — every offered line length, both directions, with and
  without `lpb` — and so is the text the Solver COMPILES for each FFT pipeline, at the
  self-test grid, through the page's live solver at the default preset, and at the LONGEST
  line the page offers (1024² / 64²×256 — where a length-gated emission would hide from a
  dump that stopped short of it). That is the pair of legs that pins the kernel text from here on (2A and 2B
  regenerate the capture by booting that tree's pages under stubenv, replaying the fixture's
  `cases` and reading each pipeline's `__code`; anything else that moves the text fails).
  Then the PROBES: `consttw` differs from the default in exactly the two twiddle
  lines and keeps every barrier, `copy` has no stage loop at all and still carries the load
  and store bodies verbatim, all three parse, and a build that THROWS still hands the seam
  back. Then BYTES: the per-step sum reproduces a
  hand-computed number for 2D 256² (**77,549,568**) and 3D 128²×64 (**1,236,886,528**)
  exactly, with the appendix-A arithmetic written out in the check, and a 2D `eqsrc` solver
  counts its extra `eqk` binding on top. Both numbers grew with FFTPERF_PLAN 2C and only in
  the `prepGrads` row: the gradient chain runs four per-pair preps per stage, each reading
  ONE state field (phi or psi) plus the grid and writing two lanes, so it is 12 dispatches
  of `cx + gr + 2cx` a step where it was 3 of `2cx + gr + 8cx`; the transforms move the same
  bytes either way (12 × 2 lanes = 3 × 8) and the butterfly count does not move at all.
  Last, `fftAnalyticCase` — the self-test's analytic reference —
  returns three nonzero bins at the flat indices it reports, zeros everywhere else, and
  each bin holding `amp·nr/2·exp(i·phase)`; and `fftAnalyticRows` adds two rows on a live
  solver and a VISIBLE "skipped" row when the solver is rebuilt during either readback (the
  forward one, or the roundtrip, where the row it already had survives beside it). The
  process exit code starts at 1 and is cleared only by the summary line, so a leg that parks
  forever — a held `mapAsync` nothing releases — fails instead of exiting green in silence.
- `names.mjs [dir]` — cross-file identifier resolution check (no redeclares, no frees);
  needs acorn (`npm i acorn`, or `ACORN=<path-to-acorn.mjs>`). The shared set is PER PAGE
  (rmhd2d loads common + physics + solver2d, rmhd3d loads common + physics), and
  `solver2d.js` is checked as a unit of its own against common + physics ALONE — which is
  what says it can be loaded by a second page.
- `cmapcheck.js` — colormap table vs emitted WGSL vs matplotlib reference.

Conventions: run from any cwd with absolute paths; each phase captures a FRESH WGSL
baseline from clean git state before editing; refvector JSON lines in the HTML are
spliced programmatically only.
