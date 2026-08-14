# plans-webgpu

Planning documents, feedback triage and audits for the WebGPU demos — the record of *how*
`webgpu/` got here. The counterpart of `plans/`, which does the same job for the solver.
Not documentation: the demos' documentation is `webgpu/docs.html` (user-facing) and
`webgpu/README.md` (developer-facing), and the physics contracts are `webgpu/SPEC.md` and
`webgpu/SPEC3D.md`.

**Why these live one level up.** `.github/workflows/pages.yml` stages the site with
`cp -r webgpu/. _site/webgpu/`, so anything sitting in `webgpu/` was *published* — plans,
feedback with Alfred's testing notes in it, the lot. Moving them out of that directory is
the fix that cannot regress: the staging step now also prunes the dev-only files it does
copy (`devtools/`, `__pycache__`, the generators and their refvector JSONs, README/SPEC),
but a file here is out of the copy's reach entirely. `devtools/checkgc.js` still walks all
of `webgpu/` for the contact address, deliberately wider than what ships.

**Reading convention.** These files were written while they lived in `webgpu/`, so a bare
backticked filename — `common.js`, `physics.js`, `rmhd2d.html`, `devtools/checkgc.js` —
means `webgpu/<that>`. Paths that start `plans/` or `taranis/` mean what they say.

**Status lives in each file's own header**, not here; where a plan and this index
disagree, the file is right.

## Live

- **VOLTEX_PLAN.md** — the 3D volume raymarch should sample a `texture_3d` with hardware
  trilinear filtering instead of doing eight scalar loads per sample out of a storage
  buffer. Written 2026-08-12, **not started**. Item 7 of the audit.
- **AUDIT_2026-08-12.md** — conciseness/duplication/efficiency audit of `webgpu/`. Items
  1–5 done the same day (render gate, dead code, staging pool, shared FFT row pair,
  `Recorder` split); item 6 (twiddle table) is dead on the LOOPLAT evidence; item 7 is
  VOLTEX_PLAN.md.

## Reverted — read before re-opening

- **LOOPLAT_PLAN.md** — taking the sync round trips out of the frame loop. Implemented,
  reviewed, measured on three devices, and **backed out** 2026-08-12. The post-mortem at
  the top is the durable part: the loop is idle ~96% of every pass and the step is
  *bandwidth*-bound (256²→512² took steps/s 500→125, ratio exactly 4.00, GB/s pinned at
  31), which closes the rest of the perf audit as well. Nothing below its banner describes
  the shipped app.

## Executed — kept for provenance

- **EIGF_PLAN.md** — ψ̂(x), φ̂(x) at fixed k_y as a 2D chart card: the tearing eigenmode's
  structure (the plot alone — a Δ′ legend readout was considered and dropped in review).
  Written, revised and executed 2026-08-14; `devtools/checkeigf.js` is its gate. One gather
  kernel and a CPU inverse along kx, so the whole card is a display-only transform of the
  state. Its header records why "minus the equilibrium" needed no subtraction and why three
  cards on `tearing` is the one thing left for on-device. Near-disjoint from CHI_PLAN.md;
  its Sequencing section owns the rules for both.
- **TEARNL_PLAN.md** — two new 2D presets off the tearing equilibrium: `tearing: X-point
  collapse` (large Δ′, secondary island, Loureiro+ 2005) and `tearing: island chain`
  (broadband seed, Δ′ selects the mode, then coalescence) in a new large 8π × 8π box.
  Shipped `tearing` kept its physics but not its defaults. Read the EXECUTION NOTES at the
  end: eight rounds of Alfred's on-device iteration moved nearly every number the plan had
  settled, and three of the findings generalise well beyond this plan — the 4.3 resolution
  benchmark is ~2× conservative (measured), a clean spectrum does NOT certify resolution,
  and rescaling a/η buys nothing because δ/dx is scale-invariant.
- **DEMOS_PLAN.md** — Elsasser displays, dual view, decaying A/B, AW collision, drawn ICs
  (phases A–E). The `physics.js` template refactor came out of Phase A.
- **REFINE_PLAN.md** — successor to DEMOS_PLAN, from Alfred's phone-testing notes
  (2026-08-06), phases F–K. Its standing no-copy-paste rule is still the house rule.
- **FEEDBACK_2026-08-08.md** — Alfred's testing notes triaged. P0 item 1 (3D forced energy
  jump) turned out to be a real solver bug and was fixed via `plans/FORCING_SPINUP_PLAN.md`.
- **FEEDBACK_2026-08-10.md** — sixteen items: colorbar, PNG/WebM save, tearing growth rate.
  All built and reviewed the same day.
- **ANISO_PLAN.md** — k∥/k⊥ vs k⊥ chart card (3D), global and field-line measures, entirely
  CPU-side arithmetic on existing readbacks.
- **CHI_PLAN.md** — χ = k⊥δb/k∥ as a second ordinate (`ay`) on that same card, with δb² the
  matched energy level itself and only δb crossing Elsasser lanes. χ is the invariant of the
  RMHD rescaling that k∥/k⊥ is not, so the card's L_z gauge caveat does not apply to it.
  Written, revised and **executed 2026-08-14**. Read the EXECUTION NOTES: the estimator bias
  came out at α = 0.988 (measured end to end, reported and never gated), the shipped ratio
  path is asserted bit-identical to base through `git show`, and the record of why the
  obvious `selLz` calibration sweep is null (forcing is pinned at |kz| bin 1) is worth
  keeping.
- **ISO_PLAN.md** — box-unit aspect, semi-transparent volume view, scale filter. Physics
  WGSL byte-identical throughout by construction.
- **ANISO_PLAN_2.md** — generated 2D spectrum E(k⊥, k∥) card (3D). Suggested by Jono
  Squire; reuses ISO's band factor. Sequencing note at the top explains why it is a
  separate file from ANISO_PLAN.
- **ONEPAGE_PLAN.md** — first-visit layout overhaul, from Alfred's wife's sketch: something
  pretty within one screen and one click.
- **PINCURVE_PLAN.md** — frozen ghost spectra on the spectrum chart cards, the comparison
  primitive the planned lessons need.
- **RECRAF_PLAN.md** — move recording capture into the rAF render and demote the 33 ms
  timer to a visibility-parked watchdog; `?recdebug` came from here.
- **RECASYNC_PLAN.md** — capture by async GPU readback (`copyTextureToBuffer` + `mapAsync`)
  instead of `new VideoFrame(canvas)`, which was costing 15–17 ms on the main thread every
  slot-due pass.
- **ANALYTICS_PLAN.md** — GoatCounter pageviews (no cookies, no custom events, hence no
  banner) and a contact line assembled in JS. `devtools/checkgc.js` is its gate, and the
  reason no file under `webgpu/` may spell the address out contiguously.
