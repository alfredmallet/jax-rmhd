# webgpu/devtools — sandbox verification tooling (REFINE_PLAN gates)

Node/python scripts built during REFINE_PLAN phases F–G to verify the apps without a
GPU. Saved here so later phases (and fresh sandboxes) don't rebuild them. Safe to
leave untracked or commit — they are dev-only, nothing in the apps loads them.

- `stubenv.js` — the shared stub: a DOM + WebGPU stub good enough to run a real app
  page (common.js + physics.js + its inline script) under node, plus `run()` to keep
  evaluating in the page's context. `require("./stubenv")(dir, page, demo)`. Since
  Phase H.0 the control panel is BUILT from a spec, so every tool that needs to see a
  control has to boot the page — hence one stub, three consumers.
- `dumpwgsl2.js <dir> <page> "" <out.txt>` — emit every generated WGSL kernel to text
  for byte-diffing against a pre-phase baseline (capture the baseline from clean git
  HEAD first). `kdiff.py` diffs two dumps kernel-by-kernel.
- `wgslparse.mjs` — parse all emitted kernels with wgsl_reflect (closest available
  compile check; npm i wgsl_reflect first).
- `bootstub.js <dir> <page> [demo]` — stub-GPU boot of a real app page; exercises
  cards (add/close/reuse/retype), every chart card's option selects with a frame drawn
  per value, the cut card's own z source, presets, ?demo= links, colormaps, editor view
  (enter/paint/save/cancel/save&run), eps± lock, band rebuild, amplitude rescale, and
  the self-test path end to end. Validates dispatch sizes, bind groups, writeBuffer
  extents.
- `checks.js` — GATE G fp64 unit checks (normalized-IC rescale, sigma_letter k-perp
  content at two resolutions, chi formula, separation cap, centroid/argmax trackers)
  against the real common.js functions via vm.
- `checksh.js` — GATE H fp64 checks against the pages' inlined reference A states:
  the H_c accumulator lane, E± = E_kin+E_mag±H_c = ½⟨|z±|²⟩, the spectra lanes summing
  back to the energies, and cutPrep+rowsC2R reproducing (u_x,u_y,b_x,b_y) on x = Lx/2 —
  each compared against a direct fp64 inverse DFT, not against itself. Also drives the
  REAL chart series (ENERGY_MODES / SPEC_SETS / CUT_PAIRS) and checks that a 1e6×
  louder E(k∥) does not move the spectrum chart's y axis.
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
