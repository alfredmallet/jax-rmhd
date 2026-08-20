# Performance and scaling

Measured numbers, the tuning knobs they justify, and the negative results — so nobody
re-runs a cluster job to rediscover an answer. All figures from Savio (account
fc_kawturb), 2026-07-25 → 2026-07-27. Setup instructions live in
`docs/SAVIO_CPU_SETUP.md` / `docs/SAVIO_GPU_SETUP.md`; the reasoning behind the numerics
is in `docs/numerics.md`.

## Read this first: fp32 flatters communication savings

Every accept/revert decision in this code was made on **fp64** numbers, because fp32
halves message sizes and therefore roughly triples the apparent benefit of any
communication optimization. Measured example: per-step forcing normalization was +27% at
fp32 and +8% at fp64 on the same job. Benchmark at the precision you will actually run.

## Architecture: what sets the ceiling

The decomposition is **z-only**. The perpendicular plane is never distributed — the
`rfft2` is process-local on every rank — so:

- the maximum useful rank count is about `nz/2` (the halo is 2 planes wide);
- a 2D run has no parallelism to express at all, which is why `dims=2` is single-process
  and `comm_backend="jax"` rejects it at construction;
- going further would need a pencil decomposition and distributed FFTs, which is a
  different architecture, not a tuning change.

## Backends: why two

Three exist, but only two are production: `"serial"` is the single-process fallback
(auto-selected with no MPI installed — exact size-1 semantics, no collectives, no
tokens). Its performance has **not** been measured against size-1 `mpi4jax`; expect a
small win from the absent token-ordering constraints, and note it is not bitwise-identical
to size-1 `mpi4jax` (dropping the mpi4jax ops changes XLA fusion — the same class of
difference as `lsrk_scan`, so compare with tolerances, not equality).

`mpi4jax` is the CPU-cluster backend and `"jax"` (shard_map/NCCL) is the GPU one. This is
not arbitrary:

- XLA:CPU collectives are slow; shard_map was tried on CPU and performed badly.
- On GPU, mpi4jax has three specific costs: without CUDA-aware MPI every transfer stages
  through host memory; each mpi4jax op is an XLA custom call that forces a CUDA stream
  sync (negligible on CPU, a pipeline stall on GPU, ×10–20 per step); and the token chain
  is opaque to XLA, so there is no compute/comm overlap.

Savio's MPI is **not** CUDA-aware (built `--without-cuda`; forcing `MPI4JAX_USE_CUDA_MPI=1`
segfaults in UCX), so mpi4jax-on-GPU is a fallback or single-GPU option only.

## CPU (savio3, mpi4jax, fp64)

Baseline, 32 ranks, 128²×256, lsrk54 + elsasser forcing + adaptive dt, production config:

| | fp32 | fp64 |
|---|---|---|
| unforced | 153 ms/step | 302 ms/step |
| forced | 174 | 353 |

One allreduce costs ~4 ms at 32 ranks and ~22 ms at 128 — which is why the
communication knobs below only pay at high rank counts.

Benchmark caveat (found during the 2026-08 precision work): before fp64-t landed,
`initialize` returned a weak-typed `t`, so the SECOND jitted `block_of_steps` call
retraced (weak → strong scalar), and `bench_phase1.py`'s single warmup call left that
retrace inside the timed region — historical numbers taken with small `nrep` include
one recompile (measured up to ~2× at 64²×16 serial/CPU with `nrep=4`). Post-fp64-t the
carry is stable from the first call (steady-state per-step time is unchanged). When
comparing against pre-2026-08 numbers, warm up twice or use a large `nrep`.

Scaling, 256²×256 strong: 1331 / 1363 / 762 / 434 ms/step at 16 / 32 / 64 / 128 ranks.
Within a node the cores saturate memory bandwidth by about 16 (32 ranks is no faster than
16); across nodes it holds ~88% per doubling. Weak scaling at 256²×4 per rank: 370 / 750 /
762 / 771 — flat from 1 to 4 nodes (97%), which is the production regime.

## GPU

Single node, 4×A5000, fp32, 512²×128, forced:

| GPUs | mpi4jax | jax/NCCL | advantage |
|---|---|---|---|
| 1 | 294 ms/step | 310 | −5% |
| 2 | 176 | 158 | +11% |
| 4 | 126 | 77.0 | +63% |

Scaling 1→4: jax ~4.0× (ideal), mpi4jax ~2.3×. The −5% at one GPU is fixed
shard_map/global-array overhead and disappears as soon as there is communication to do.

Multi-node, 16×GTX2080Ti over 4 nodes, fp32, 512²×256:

| GPUs (nodes) | mpi4jax | jax/NCCL | advantage |
|---|---|---|---|
| 4 (1) | 284 ms/step | 200 | 1.4× |
| 8 (2) | 204 | 102 | 2.0× |
| 16 (4) | 165 | 48 | 3.4× |

Scaling 4→16: jax 4.15×, mpi4jax 1.72×. The jax curve holds **across node boundaries on
plain TCP** — no InfiniBand userspace, PCIe peer-to-peer disabled. On NVLink/IB hardware
the margin can only grow.

### Cost per timestep (fp32, 512²×128)

| Hardware | ms/step | SU/hr | SU per 1000 steps |
|---|---|---|---|
| 1 savio3 node (32c) | 1563 | 32 | 13.9 |
| 2 nodes | 866 | 64 | 15.4 |
| 1 × A5000 | 294 | 18.7 | 1.53 |
| 4 × A5000, mpi4jax | 126 | 74.7 | 2.61 |
| 4 × A5000, jax | 77 | 74.7 | 1.60 |

CPU cost per step *rises* under strong scaling — waiting cores still bill — while the jax
backend holds it flat while quadrupling throughput. Roughly **9× cheaper per timestep on
GPU at fp32**. Savio's workstation GPUs run fp64 at 1/32 rate, so fp64 production stays on
CPU there; on full-rate-fp64 hardware (A100/GH200/H100) the economics carry over.

## Tuning knobs, measured

| Knob | Effect | Guidance |
|---|---|---|
| `forcing_norm_per_step` | +8% at fp64/32 ranks | default on |
| `cfl_every` | +1.3% (N=20) at 32 ranks, +8.9% at 128 | 10–20 at ≥128 ranks, **developed states only** |
| `lsrk_scan=True` (scan) | ~20% faster than unrolled on CPU | default |
| `lsrk_scan=False` (unrolled) | +21% mpi4jax-GPU, +12% jax single-node, **−38% jax multi-node** | per-machine knob, benchmark it |
| `forcing_shell_noise` | faster single-device, ~5% *slower* on Savio CPU at 32 ranks | opt-in; revisit on GPU |
| `halo_start` | neutral (≤2%, sub-noise) everywhere measured | see below |

`cfl_every > 1` costs one extra standalone gradient evaluation per block, because the
stage-0 RHS no longer doubles as the dt source. At 32 ranks that cancels the saving at
N=5; it only pays once the allreduce is expensive.

**The `cfl_every` hazard is real:** from a quiescent forced start the CFL dt collapses by
~10× within a few steps of spin-up, so a frozen dt NaNs — measured, N=20 dies by t≈2 while
N=5 survives. Use N>1 only from developed states.

## Negative results

Recorded so they are not re-investigated.

- **Early halo issue (`halo_start`) buys nothing measurable.** On mpi4jax there is no fp64
  win at any rank count (−0.6% at 32, sub-noise at 128) because the token chain serializes
  communication with compute regardless. On the jax backend it is neutral on Savio at
  bench sizes, including multi-node NCCL. The hook is kept and enabled for `"jax"` because
  the answer plausibly changes with a different scheme or on NVLink/IB hardware, but it is
  not currently earning its keep.
- **Unrolled LSRK is not a universal win.** It helps every configuration except the one
  that matters most for scaling, jax multi-node, where it costs 38%.
- **shard_map on CPU** — measured slow, which is why `mpi4jax` remains the CPU backend.

## Test particles overhead (laptop)

### 2D (A2/A3, 2026-08-18)

`bench/particles_overhead.py`: `jax.jit(block_of_steps)` at 256², `Lx=Ly=2π`, fp64,
lsrk33, fixed `dt=1e-3` (`adaptive_timestep=False`, for determinism), elsasser forcing
(`forcing_power_elsasser=(1,1)`, `eqpars={"diss":(1e-4,1e-4),"hyper":2}`). Particles:
`n=32768` per ensemble, `qm=15`, `vth=1` (plan §2 baseline). Apple M1, macOS 14 (Darwin
23.6.0), jax 0.10.0, CPU backend, quiet machine.

| case | ms/step | overhead vs off |
|---|---|---|
| off | 13.2 | — |
| on, 1 ensemble (default mask), 32768 particles | 16.65 | +26% |
| on, 3 ensembles (ideal, full ∂ψ/∂t, E=0 control), 32768 each | 22.9 | +74% |

Breakdown, fixed (transforms) vs O(N) (gather/push):

| part | cost | share of solver step | scaling |
|---|---|---|---|
| `particle_fields` (4 gradient iffts + fft/ifft pair for the dealiased ideal E_z; optional resistive/forcing iffts add ~nothing measurable) | 2.24 ms | 17% | grid-like, same as the solver's own transforms |
| `boris.push` per ensemble of 32768 (4 bilinear gathers + 2 kicks) | 1–2.5 ms | — | O(N), resolution-independent per particle |

Verdict: the fixed part (`particle_fields`) lands INSIDE the plan's ≤15–20% budget
(plans/TESTPART_PLAN.md §4) — that budget was an FFT-count estimate of exactly this
piece. The O(N) push is what exceeds it: at 256² with 32768/ensemble the loading is 0.5
particles per grid point, dense, and XLA's CPU gather is a scalar loop. In 3D at
production sizes (e.g. 256²×128 with ~1e5 particles) the push is ~1% and only the fixed
transform part matters; on GPU the gather is cheap.

**Re-measured after A3 (2026-08-18, same machine and script, `nrep=25`).** A3 replaced the
two gathers per half-kick (E, then B) with a single gather of `assemble_stacked`'s combined
array, and added the per-piece work accumulation. Net effect on the total: nothing
measurable.

| case | ms/step (A2) | ms/step (A3) | overhead vs off |
|---|---|---|---|
| off | 13.2 | 12.9 | — |
| on, 1 ensemble (default mask), 32768 particles | 16.65 | 16.8 | +30% |
| on, 3 ensembles (ideal, full ∂ψ/∂t, E=0 control), 32768 each | 22.9 | 22.8 | +76% |

(The A2-vs-A3 overhead percentages differ mostly because `off` benchmarked ~2% faster in
the A3 run; the particle-on absolute times are unchanged within the run-to-run IQR, ~1–2
ms/step.) Isolated jitted pieces at the same size: `particle_fields` 2.16 ms with the
default mask, 2.50 ms with the resistive and forcing pieces on (17% / 19% of the solver
step, still inside the plan's budget); `boris.push_tracked` for one 32768-particle ensemble
2.30 ms at the default mask (6 gathered components) and 5.23 ms at the full mask (8
components — the three E_z pieces are gathered separately so their work can be attributed,
where A2 gathered one pre-summed E_z). So sharing the cell/weight work across E and B
bought back roughly what keeping the E_z pieces separate costs, and the O(N) push remains
the part that exceeds the budget at this 2D loading.

Deferred (Alfred, 2026-08-18): gather optimization would attack only the O(N) part
(est. 1.5–2×) — revisit if A3's 2D science runs at 3×32768 particles feel slow. Two
items flagged: (i) gather-side reorganization (`interp.gather`: share cell/weight
computation across E and B — done in A3, see above, no measurable win; `jnp.take` on flat
indices, cast samples not grids — still open); (ii)
reuse the stepper's stage-1 gradients in `particle_fields` (removes 4 of the 6 fixed
transforms) — the bigger lever, but changes the stepper contract, so not Phase A.

### Test particles overhead in 3D (B2, 2026-08-19)

Same script and machine (`bench/particles_overhead.py`, now taking a case argument:
`2d` | `3dfd` | `3dspec`; Apple M1, macOS 14 (Darwin 23.6.0), jax 0.10.0, CPU backend,
fp64, quiet machine, `nblock=20`, `nrep=12`). 3D runs **128²×16** — a smaller perpendicular
grid than the 2D case so one configuration fits a laptop — with everything else identical
(`dt=1e-3` fixed, lsrk33, elsasser forcing, `eqpars={"diss":(1e-4,1e-4),"hyper":2}`,
`n=32768` particles per ensemble, `qm=15`, `vth=1`). What transfers between grids is the
RATIO to the solver step, not the ms.

| case | finite-difference z, ms/step | z_spectral, ms/step |
|---|---|---|
| off | 42.2 | 75.8 |
| on, 1 ensemble (default mask) | 53.9 (**+27.7%**) | 97.9 (**+29.1%**) |
| on, 3 ensembles (ideal, full ∂ψ/∂t, E=0 control) | 74.3 (**+76.2%**) | 125.0 (**+65.0%**) |

Isolated jitted `particle_fields` at the same grid:

| mask | finite-difference z | z_spectral |
|---|---|---|
| default (4 gradient iffts + the dealiased-E_z fft/ifft pair) | 7.36 ms = **17.4%** of the solver step | 9.76 ms = **12.9%** |
| + resistive and forcing pieces | 9.66 ms = **22.9%** | 10.94 ms = **14.4%** |

**The finite-difference-z filter, timed for the first time** (B1 folded `−z_diss·(dz/2)⁴∂_z⁴ψ`
into `ez_resistive`; it costs a 4th-order z stencil and a `comms.halo_exchange`, which no
earlier measurement covered). Isolated at 128²×16 in a separate timing session, so compare
these two with each other and not with the table above: the resistive piece as the k-local ψ
diagonal alone is **1.50 ms**, and `fields._psi_non_ideal` with the filter is **3.36 ms** — the
stencil plus halo is **1.86 ms**, about **4.4%** of the 42.2 ms solver step. That is the whole
gap between the FD-z and z_spectral "+ resistive and forcing" rows, and it is why FD-z with the
optional pieces on is the one configuration that leaves the plan's ≤15–20% budget
(plans/TESTPART_PLAN.md §4) at 22.9%. It buys exactness: without it, full-mask E_z misses
∂ψ/∂t by 1.2e-3 of max|E_z| and gate 7 stops converging (`tests/test_particles_3d.py`).

Two more observations:

- **z_spectral looks cheaper only because its solver step is nearly twice as expensive**:
  `particle_fields` costs about the same absolute time in both modes, so its *share* halves.
  Nothing about the particle path is faster there. (The parenthetical this bullet first
  carried — "rfftn/irfftn over (z,x,y) against one rfft2 per plane" — was a guess, and the
  profile below shows it is wrong: the transforms account for ~3 ms of the ~38 ms gap.)
- **The trilinear gather roughly doubles the O(N) push cost per particle**, as its 8 corners
  against 4 predict: subtracting `particle_fields` from the one-ensemble overhead gives ≈1.9 ms
  in 2D at 32768 particles and ≈4.3 ms in 3D. It is still the part that exceeds the budget at
  this dense loading (0.5 particles per grid point in 2D, 0.125 in 3D); at production 3D sizes
  with ~1e5 particles it is a few percent, and only the fixed transform part matters.

(2D re-measured in the same session for comparability: off 12.65, on1 16.35 (+29.2%), on3 22.64
(+79.0%) ms/step; `particle_fields` 1.80 ms = 14.2% and 2.10 ms = 16.6% — the A3 numbers within
run-to-run scatter.)

## Where the z_spectral step's extra time goes (2026-08-19)

`bench/zspectral_profile.py`, same machine as the particle benchmarks (Apple M1, macOS 14,
jax 0.10.0, CPU, fp64, quiet machine), RMHD at **128²×16**, lsrk33, fixed `dt=1e-3`,
elsasser forcing, `nblock=20`, `nrep=10`. The whole gap is the **linear propagator**, not
the transforms.

| | finite-difference z | z_spectral | ratio |
|---|---|---|---|
| ms/step | 42.9 | 81.6 | 1.90 |
| GFLOP/step (XLA cost analysis) | 0.065 | 0.255 | 3.91 |

Ablation ladder inside the real (scanned, fused) z_spectral step — each row drops one piece
of `Putzer2Propagator` and re-times; every variant is numerically wrong on purpose:

| variant | ms/step | that piece costs |
|---|---|---|
| baseline (production) | 79.8 | — |
| − the complex `sqrt`/`cosh`/`sinh` in `_coeffs` | 51.9 | **27.9** |
| − `exp(m·tau)` as well | 48.1 | 3.8 |
| − the 2×2 apply as well (propagator = identity) | 45.9 | 2.2 |
| finite-difference-z step, for comparison | 42.9 | — |

So of the +37 ms, **33.9 ms is the putzer2 matrix exponential** (27.9 of that the complex
transcendentals alone, evaluated per stage on a complex (nz,nkx,nky) grid) and the remaining
3.0 ms is everything else — the (z,x,y) transforms net of the finite-difference-z stencil the
spectral mode does not run. The isolated pieces say the same: the whole RHS times 14.8 ms (FD)
against 15.8 ms (spectral), and the `ifft` of the gradient stack 4.17 vs 4.25 ms, while FD
alone pays 1.3 ms for `FDLinearTerm`. An independent check swapping only the operator on a fixed grid agrees:
with L forced to the diagonal backend, fd 42.4 vs spec 47.1 ms/step; with L forced to a 2×2
putzer2 operator of the same z extent in both modes, fd 82.8 vs spec 88.6.

**Isolated timings mislead here, in both directions.** `apply_exp` timed on its own with the
kgrid *closed over* reports 0.3 ms because XLA constant-folds `exp(L·tau)` away entirely; with
the kgrid passed as an argument it reports 19.6 ms, which then over-counts because the fused
step shares work across the two `apply_exp` calls of a stage. The ablation ladder is the number
to trust. Monkeypatching for an ablation needs `jax.clear_caches()` — the jaxpr trace cache is
keyed on the function and avals, so without it every variant silently re-reports the baseline
(this bit the first run of this profile).

Scaling with nz at nx=128 (ms/step, fd → spec): nz=4 11.4 → 18.1 (1.59×), nz=8 22.3 → 40.5
(1.82×), nz=16 42.9 → 88.5 (2.06×), nz=32 88.6 → 180.1 (2.03×). Both sides are linear in nz;
the ratio saturates near 2 once the propagator's per-mode cost dominates the fixed overheads.

**Why the complex transcendentals cost what they do** (optimized HLO, `jax.jit(...).compile()
.as_text()`, XLA CPU): `cosh(z)` on complex128 lowers to `(exp(z) + exp(−z))/2` with each
complex `exp` expanded as `exp(x)·(cos y + i sin y)` plus overflow guards — 4 real `exp`, 2
`cos`, 2 `sin`, 6 selects; `sinh` the same again with nothing shared (XLA does not recognise
`exp(−z)` as `1/exp(z)`, nor `cos(−y)` as `cos(y)`); complex `sqrt` is 7 real `sqrt`, 9
divides, 18 selects (branch-cut and overflow handling); complex `exp` is 2 `exp` + `cos` +
`sin`. So one putzer2 `_coeffs` + `exp(m·tau)` evaluates **10 exp + 5 cos + 5 sin + 7 sqrt + 9
div per mode**, where the mathematics needs 2 complex exps (4 exp + 2 cos + 2 sin) — or, for an
L whose `m` is real and `s²` real and one-signed, 1 real exp + 1 cos + 1 sin. And on XLA CPU
each real transcendental is 3–6 ns/element (`exp` is a vectorised polynomial, `sin`/`cos` are
slower library-class calls), so 20-odd of them per mode per stage over 133k modes is the
~10 ms per `apply_exp` the ablation found. The same operator shape and mode count occurs for
2D GDI at 512² on an IF scheme (putzer2 on (2,2,1,nkx,nky) = 131k modes), so the IF path
there pays it too; the IMEX path (GDI production) is rational — `solve_shifted` has no
transcendentals — and is unaffected.

**Done: hoisted stage propagators (2026-08-19, `params.hoist_propagator`, default True).**
The exponent depends on `tau = gamma_s·dt` only, so whenever dt is frozen over a block — fixed
dt, or one `cfl_every` block — `run.py` forms every stage's `exp(L·tau)` ONCE per block
(`timestepping.stage_exp_ops` → `propagators.ExpOp` pytrees, stacked as the stage scan's xs)
and each stage only applies it. Same arrays in the same op order, so **hoisted == unhoisted
bitwise** at fp64 in every cell of `tests/test_hoist_propagator.py` (2D/FD-z/z_spectral ×
lsrk33/lsrk54/rk44 × fixed/cfl_every=2/adaptive × scan/unrolled; at fp32 one cell is 1 ulp
off from a fusion difference). Re-measured
(same session, loaded machine, so compare within the row):

| z_spectral 128²×16 | unhoisted ms/step | hoisted | |
|---|---|---|---|
| fixed dt, lsrk33 | 78.7 | **48.7** | 0.62× |
| fixed dt, lsrk54 | 146.2 | **81.5** | 0.56× |
| fixed dt, rk44 | 64.9 | 62.1 | 0.96× (rk44's two taus were already shared) |
| adaptive, `cfl_every=4` | 90.0 | **54.5** | 0.61× |
| adaptive, `cfl_every=1` | 89.6 | 91.2 | nothing frozen, nothing to hoist |
| FD-z, fixed dt, lsrk33 | 44.5 | 44.0 | diagonal L is z-broadcast: nothing to gain — and not hoisted |

The remaining gap to the FD-z step (48.7 vs 44) is the (z,x,y) transforms plus the 2×2 apply
itself. Verified in the optimized HLO (`bench`-style count of `exponential`/`cosine`/`sine`/
`sqrt` per while body): with hoisting, zero transcendentals in the step loop for fixed dt and
all of them in the outer cfl-block loop for `cfl_every>1`; with `hoist_propagator=False` the
legacy graph (33 per stage, inside the stage scan, where `gamma` is a scanned value XLA cannot
hoist past). That knob exists for memory: one `ExpOp` per stage — for putzer2 4 complex
arrays of L's full shape, i.e. 4·nstage·nz·nkx·nky·16 B at fp64 (128²×16, lsrk33: 16 MB;
256²×64, lsrk54: 0.7 GB; 512²×128 fp32 lsrk54: 2.7 GB). Note XLA's own loop-invariant code
motion would hoist these too once they are formed outside the stage scan with static `gamma`
(observed in the HLO) — which is why the unhoisted path deliberately keeps the exponent
inside the stage scan: `False` must mean memory-light, not "hoisted by XLA instead".
Only the putzer2 backend is hoisted: the diagonal backend's exp is one real exp per mode per
stage and z-broadcast for FD-z, so there is nothing to gain, and a first version that hoisted
it too broke gate 6's bitwise reference on the 2D fixed-dt `simulate_scan` configs by 15
elements at 1e-23 absolute (with a literal `gamma`, XLA folds `(L·dt)·gamma` differently) —
the hoisted/unhoisted bitwise agreement is an op-order statement, not a guarantee against
constant folding, so expect round-off-level differences on other grids/versions and never
pin a hoisted putzer2 run bitwise against an unhoisted one across jax versions.

**Memory, measured** (`jit(block_of_steps).lower(...).compile().memory_analysis()`, fp64,
lsrk, elsasser forcing; U = one complex full-grid array = nz·nkx·nky·16 B; "temp" is XLA's
working set for the program, "total" adds the live arguments — state + kgrid):

| case | U | kgrid | state | temp | temp/U | total |
|---|---|---|---|---|---|---|
| spec 128²×16 lsrk33, unhoisted | 2.0 MB | 12.5 MB | 4.6 MB | 64.9 MB | 31.9 | 81.9 MB |
| spec 128²×16 lsrk33, hoisted | | | | 72.0 | 35.4 | 89.0 (+9%) |
| spec 128²×16 lsrk54, hoisted | | | | 105.0 | 51.7 | 122.0 (+49%) |
| FD-z 128²×16 lsrk33 | 2.0 | 0.3 | 4.6 | 46.0 | 22.6 | 50.8 |
| spec 256²×64 lsrk33, unhoisted | 32.2 | 196 | 66.5 | 1018 | 31.6 | 1281 |
| spec 256²×64 lsrk33, hoisted | | | | 1131 | 35.1 | 1394 (+9%) |
| spec 256²×64 lsrk54, hoisted | | | | 1661 | 51.5 | 1924 (+49%) |
| FD-z 256²×64 lsrk33 | 32.2 | 1.1 | 66.5 | 714 | 22.1 | 781 |

Reading it: the z_spectral step's working set is ~32 U before any hoisting (FD-z: ~22 U) —
the RHS's real-space gradient stack and the rfftn intermediates, not the propagator — and the
persistent putzer2 operator (`lin_L` 4 U + `lin_m` + `lin_s2`) is 6 U = 196 MB at 256²×64.
Hoisting adds 4·nstage U of live arrays less the per-stage coefficient temporaries it
removes: measured +3.5 U for lsrk33 (+9% of the program) and +20 U for lsrk54 (+49%);
`cfl_every` blocks give the same numbers as fixed dt. So lsrk33 hoisted is cheap; lsrk54
hoisted is the case to think about on a memory-bound grid.

**Splitting the operator into perp-only and z-only factors** would make the hoisted memory
vanish (perp arrays (nkx,nky) plus z arrays (nz,) per stage) and also drop the 6 U operator
itself — but `exp((A+B)τ) = exp(Aτ)exp(Bτ)` only when `[A,B] = 0`. RMHD spectral-z with
ν = η: `L = D(k⊥)·I + i·kz·σ_x` (+ `−z_diss_k·kz⁴·I`), everything commutes, the split is EXACT
(it is the Elsasser-separable form measured at 0.62× above, which needs no hoisting and no
change of state variables). ν ≠ η: `diag(d_φ,d_ψ)` does not commute with `σ_x`, the split is a
Lie splitting with O(τ²·(d_φ−d_ψ)·kz) error — not acceptable for a scheme whose point is the
exact linear propagator. KAW-type operators (entries `i·kz·f(k⊥)`): the exponential carries
`cos(kz·√(fg)(k⊥)·τ)`, a function of the *product*, which no product of a kz-only and a
k⊥-only array reproduces — there the memory-free choice is the per-stage real-trig
evaluation (`m` real, `s²` real ≤ 0: 1 real exp on (nkx,nky) + cos + sin on the full grid,
0.75×) and hoisting stays the speed-for-memory lever. Generic halving available either way:
a 2×2 with `L00 = L11` and `L01 = L10` has `m00 = m11`, `m01 = m10`, so `Putzer2Exp` could
store 2 arrays per stage instead of 4 (detectable at setup).

**Precomputed eigenvectors, reassessed against these numbers.** `V`, `V⁻¹`, `λ` are 10 U
persistent; hoisting on top stores `exp(λτ)` = 2 U per stage and the apply costs 10 complex
mults instead of putzer2's 4: totals 10 + 2·nstage (16/20 U for lsrk33/54) against putzer2's
6 + 4·nstage (18/26 U) — a saving only for lsrk54, bought with ~2 ms/step of extra multiplies.
Unhoisted it is the 0.69× per-step path. For RMHD specifically `V` is the constant Elsasser
transform (nothing stored) and `λ = d ± i·kz` is separable at ν = η (nothing stored): the
eigen route collapses into the separable form above. So: RMHD wants the separable propagator
(zero memory, 0.62× at every step, adaptive included); generic 2×2 operators want putzer2 +
hoisting for speed, eigen storage only if lsrk54's hoisted 26 U is the constraint.

**Schemes against each other under z_spectral** (same run, 128²×16, fixed dt, hoisting on
unless stated; memory at 256²×64 is the same picture). "stab." is the scheme's own
imaginary-axis stability limit `|ω·dt|_max` for the explicit (advective) part — L is exact
under IF so only the nonlinear term sees the RK stability polynomial (lsrk33 1.73, rk44 2.83,
lsrk54 3.34, computed from the stored tableaus); the last column is ms/step ÷ stab., i.e. cost
per unit simulated time IF `cfl_safety` were raised to each scheme's limit — at a common
`cfl_safety` the comparison is just ms/step:

| scheme | temp/U | ms/step | ms/stage | cost per unit t at stab.-limited dt (rel.) |
|---|---|---|---|---|
| lsrk33, unhoisted | 31.9 | 80.5 | 26.8 | 1.00 |
| lsrk33 | 35.4 | 49.2 | 16.4 | 0.61 |
| lsrk54, unhoisted | 31.9 | 130.5 | 26.1 | 0.84 |
| lsrk54 | 51.7 | 81.3 | 16.3 | 0.52 |
| rk44 (hoist on or off: identical) | 36.2 | 60.0 | 15.0 | **0.46** |

Two conclusions. (i) The 2N (two-register) property of the LSRK schemes buys nothing here:
the z_spectral working set is ~32 U of RHS temporaries either way, hoisted lsrk33 (35.4 U)
sits level with rk44 (36.2 U, four k-registers AND its two hoisted ops), and hoisted lsrk54
(51.7 U) is the most memory-hungry option of all. (ii) rk44 needs no hoisting machinery:
its two taus (`dt/2`, `dt`) are fixed per step, so XLA's loop-invariant code motion already
lifts them out of the step loop (hoist on/off identical), it is 4th order, its per-stage cost
is the lowest (no stage scan, no `cond`), and it tolerates the largest dt of the three per
stage but one. At a common `cfl_safety` hoisted lsrk33 remains the cheapest step (49 vs 60
ms); if `cfl_safety` is scaled to the scheme, rk44 is the cheapest per unit time and 4th
order — worth considering as the z_spectral default once the separable propagator lands
(which removes the hoisted-memory term from every row and leaves only the registers, where
lsrk33 wins again by 4 U). **FD-z is a different regime:** the diagonal z-broadcast exp costs
nothing per stage, so hoisting is neither needed nor enabled there (`hoistable=False`), and
the scheme choice is the classical one — lsrk33 for cost, lsrk54/rk44 for order.

**Available, not done — the adaptive `cfl_every=1` path.** Nothing is frozen there, so the
per-stage evaluation stays and only a cheaper evaluation helps. Generic, no memory, any L:
store `s = sqrt(s2)` at setup (kills the complex sqrt — its 7 sqrt/9 div/18 selects per mode —
from the step; `tau > 0` so the branch is immaterial and cosh/sinh·z are even anyway) and form
`w = exp(s·tau)` once, `cosh = (w + 1/w)/2`, `sinh/s = (w − 1/w)/(2s)` with the small-|z|
Taylor branch kept: 2 complex exps instead of 10 exp/5 cos/5 sin/7 sqrt. Structure-aware, when
`m` is real and `s2` real one-signed (RMHD: `m = −νk^{2h}`, `s2 = −kz²` — concrete at setup):
1 real exp + cos + sin — measured **0.75×** the step; and the Elsasser-separable form
(`e^{dτ}` on (nkx,nky) ⊗ `e^{±ikzτ}` on (nz,), no change of state variables) **0.62×** — the
same ratio as hoisting, but for every step including adaptive `cfl_every=1`, at no memory.
Both change round-off on the putzer2 paths (no bitwise gate pins them; the FD diagonal path is
untouched). A naive real-trig version that evaluates both `cosh`/`cos` branches under a `where`
and casts back to complex measured *slower* (114.8 ms/step). The full per-mode
eigendecomposition (V, V⁻¹, λ stored) was measured at 0.69× and rejected: 10 full-grid arrays
for less than the separable form gives.

## Memory: where it goes and what was removed

The measurement instrument is `bench/memory_probe.py` (Phase 0 of
`plans/MEMORY_PERF_PLAN.md`), which reports per case: XLA
`compiled.memory_analysis()` as temp/args/out in bytes and in **u** — one field-sized
complex array, `nz_local·nkx·nky·itemsize` (8 B fp32, 16 B fp64; the RMHD state is
2 u) — plus `total_u = temp+args+out`, the device `peak_bytes_in_use` on GPU, and the
median ms/step of a jitted `block_of_steps`. Three conventions to hold when reading any
number in this section: (i) `total_u` is the quotable one — `lin_*` lives in *args*
(the hoisted ExpOps are formed inside the jitted block and sit in *temp*), so temp-only
understates z_spectral by ~8–10 u; (ii) the probe
measures the **non-donated** graph (it reuses one state across reps), while production
jits with `donate_argnums=(0,)` and may alias input to output — u values describe the
probe's graph, consistently at every measurement point, and phase gates in the plan are
**deltas** between probe runs, never absolute targets; (iii) under `comm_backend="jax"`
both `memory_analysis()` and u are per-device (verified against a fake-device mesh).
(The "Memory, measured" block above uses U = the fp64 u at other grids — same idea,
different absolute numbers.)

**CPU baseline** (M1 laptop, jax 0.10.0, fp32, `bench/memory_probe_laptop_baseline.json`
— the regression reference. The fp64 twin `..._fp64.json` has the same memory to
≤0.38 u — every difference sits in the z_spectral/GDI args block (fp64-always scalars),
FD-z rows agree to ≤0.005 u — and costs 1.3–2.1× the time. The 128²×32 gate grid is in
`bench/memory_probe_laptop128_baseline.json`. u = 0.26 MB at 64²×16, 0.25 MB at 256²):

| case (64²×16 RMHD/GDI-3D, 256² GDI-2D) | temp u | args u | total u | ms/step |
|---|---|---|---|---|
| rmhd_fdz lsrk33 / lsrk54 | 26.58 | 2.26 | 30.96 | 5.07 / 8.28 |
| rmhd_fdz imexcb2 / cb3e / cb3c | 24.57 | 2.26 | 28.96 | 5.09 / 7.54 / 6.84 |
| rmhd_fdz imexcb3f | 56.45 | 2.26 | 60.83 | 10.96 |
| rmhd_fdz lsrk33 / lsrk54 / cb3e unrolled | 37.95 / 55.20 / 35.32 | 2.26 | 42.33 / 59.58 / 39.71 | 7.36 / 21.15 / 7.11 |
| rmhd_zspec lsrk33 hoisted / unhoisted | 32.89 / 29.51 | 8.31 | 43.33 / 39.95 | 6.04 / 12.02 |
| rmhd_zspec lsrk54 hoisted / unhoisted | 51.76 / 29.51 | 8.31 | 62.19 / 39.95 | 10.08 / 19.70 |
| rmhd_zspec lsrk33 / lsrk54 unrolled (hoist on = off) | 32.89 / 43.76 | 8.31 | 43.33 / 54.20 | 5.70 / 9.50 |
| rmhd_zspec imexcb3e | 18.89 | 6.31 | 27.33 | 9.52 |
| rmhd_zspec lsrk33 ν≠η (putzer2) | 32.89 | 8.31 | 43.33 | 5.95 |
| gdi2d_256 lsrk33 / lsrk54 | 33.47 / 53.64 | 11.13 | 48.60 / 68.77 | 4.07 / 6.77 |
| gdi2d_256 imexcb2 / cb3e / cb3c | 16.97 | 9.13 | 30.10 | 4.89 / 6.73 / 6.83 |
| gdi3d lsrk33 | 31.57 | 8.31 | 42.01 | 4.86 |
| gdi3d imexcb2 / cb3e / cb3c | 14.95 | 6.31 | 23.39 | 5.59 / 7.67 / 7.64 |

What the table says, structurally (buffer breakdown in `plans/TARANIS_MEMORY_HANDOFF.md`):
the FD-z IF working set is dominated by the batched gradient transforms (8 u k-space
stack + ~8 u real-space output — the plan's F1) and the halo-concatenate pair (~4.3 u,
F2/F3); z_spectral adds the resident putzer2 operator (6 u of args) and, hoisted,
4·nstage u of ExpOps (+3.4 u lsrk33, +22 u lsrk54 — the plan's Z1 removes both for
ν=η); the [2R] CB-IMEX steppers are the memory floor everywhere (no exponentials, one
shifted solve per stage); `imexcb3f` is unrolled-only and pays 2.1×. `lsrk_scan=False`
costs 1.37–1.92× the scan path's total for every stepper measured, and on the unrolled
path `hoist_propagator` is a no-op on both axes — XLA hoists the literal-gamma stage
exponents itself (the `_unroll_hoist{1,0}` rows: identical totals, identical speed at
both settings, both precisions). Memory in u reproduced identically (0.00 u per case) on a second
machine and across jax 0.10.0/0.10.2 during Phase 0 validation — timings are this
laptop's only.

**GPU baseline, G1 Kaggle P100 16 GB** (jax 0.11.1, isolated subprocess per case,
`bench/memory_probe_p100_baseline_fp{32,64}.json`; fp32 at 512²×128 / GDI-2D 1024², u =
135 MB; fp64 halves the grid. `peak` is the device allocator's `peak_bytes_in_use` — it
runs a near-constant ~16 u above `total_u` on every row, the state/kgrid/warm-up
residue):

| case (fp32) | total u | peak u | ms/step |
|---|---|---|---|
| rmhd_fdz lsrk33 / lsrk54 | 30.10 | 46.07 | 182.7 / 305.2 |
| rmhd_fdz imexcb3e | 24.06 | 40.03 | 290.3 |
| rmhd_fdz lsrk54 unrolled | 28.09 | 44.06 | 336.8 |
| rmhd_zspec lsrk33 hoisted / unhoisted | 54.17 / 36.17 | 70.14 / 52.14 | 178.0 / 209.8 |
| rmhd_zspec lsrk54 hoisted / unhoisted | 70.17 / 36.17 | 86.14 / 52.14 | 338.3 / 345.6 |
| rmhd_zspec imexcb3e | 30.30 | 48.27 | 222.7 |
| rmhd_zspec lsrk33 adaptive cfl_every=1 | 36.17 | 52.14 | 211.0 |
| rmhd_zspec lsrk33 ν≠η (putzer2) | 54.17 | 70.14 | 178.0 |
| gdi2d_1024 lsrk33 / imexcb3e | 53.63 / 30.25 | 66.12 / 44.74 | 4.4 / 5.3 |
| gdi3d_256x64 imexcb3e | 26.34 | 41.30 | 22.7 |

Zero OOMs: the hoisted z_spectral lsrk54 row (70.2 u ≈ 9.2 GB + peak 11.6 GB) fits the
16 GB card, as the plan predicted.

**GPU baseline, G2 Savio GTX 2080Ti 11 GB** (jax 0.10.2, job 37775868,
`bench/memory_probe_gtx2080_baseline_fp{32,64}.json` + `..._fp32_jax4.json`; same grids
as the P100 profile plus a 512²×64 twin of the OOM candidate):

| case (fp32, 512²×128) | total u | peak u | ms/step |
|---|---|---|---|
| rmhd_fdz lsrk33 / lsrk54 | 28.06 | 36.07 | 162 / 275 |
| rmhd_zspec lsrk33 hoisted / unhoisted | 54.17 / 36.17 | 54.2 / 44.2 | 149 / 199 |
| rmhd_zspec lsrk54 **hoisted: OOM** / unhoisted | 70.17 / 36.17 | — / 44.2 | OOM / 328 |
| rmhd_zspec lsrk54 hoisted, 512²×**64** | 70.21 | 79.2 | 135 |
| rmhd_zspec imexcb3e / adaptive lsrk33 | 30.30 / 36.17 | 40.3 / 44.2 | 204 / 201 |
| gdi2d_1024 lsrk33 / imexcb3e | 53.63 / 30.25 | 60.1 / 38.8 | 3.8 / 4.5 |
| FD-z lsrk54, 4-GPU sharded (`comm_backend="jax"`) | 24.76 /dev | 32.76 /dev | 75.3 |

The headline row is the recorded "before" of the Z1 flip: **hoisted z_spectral lsrk54 at
512²×128 fp32 OOMs the 11 GB card** (9.2 GB program + context; the allocator fails on a
7.5 GB request) while the unhoisted path fits — after Z1 removes the hoist memory this
row must fit, and that flip is the G2 post-Z deliverable. The 2080Ti is otherwise
~10–15% faster than the P100 per step, the 4-GPU sharded row matches the earlier run
exactly (near-perfect weak scaling at nz_local = 32, per-GPU per-point parity with the
P100), and fp64 mirrors the P100 structure. Absolute u differs a little from the P100
JSONs on some rows (e.g. FD-z 28.1 vs 30.1) — jax 0.10.2 vs 0.11.1 buffer accounting;
compare within a card/version, not across. What the GPU says that the CPU
could not: (i) hoisting buys lsrk33 0.85× but lsrk54 only ~0.98× — streaming 22 u of
ExpOps per step costs about what the transcendentals cost, exactly the bandwidth trade
Z1 sidesteps by storing nothing full-grid; (ii) z_spectral hoisted lsrk33 (178 ms) is
*at parity with FD-z lsrk33* (183 ms) on GPU — the z_spectral premium is a CPU
phenomenon at this size; (iii) adaptive `cfl_every=1` (211 ms) sits at the unhoisted
step, as expected (nothing frozen to hoist); (iv) FD-z lsrk54 unrolled costs LESS
memory than the scan on GPU (28.1 vs 30.1 u — the reverse of CPU, where it is 59.6 vs
31.0) and 1.10× the time, so `lsrk_scan=False` remains a no-win on this card. fp64 at
256²×64: same u structure to ≤2.1 u, 46.1/75.6 ms FD-z lsrk33/54. Post-Part-F and
post-Part-Z reruns of both G1 launches (`../lugus/launch.py run bench/memory_probe.py
--entry-kwargs '{"profile": "p100", "tag": "postF"}'` etc.) fill in the deltas here.

## Known, not done

`run.py` calls `mngr.wait_until_finished()` immediately after every `save_snapshot`, which
defeats orbax's asynchronous save: checkpoint I/O is serialized with compute rather than
overlapping the next block of steps. Waiting lazily instead — before the *next* save and
at the end of the run — was planned and never implemented. It needs care around
`max_to_keep` deletion.

## Production guidance

**CPU clusters:** `mpi4jax`, fp64, `forcing_norm_per_step=True`, `cfl_every` 10–20 at
≥128 ranks from developed states.

**Savio GPU:** `comm_backend="jax"`, fp32 workloads, sizes per the cost table above.

**z_spectral (single process):** run with `cfl_every > 1` or a fixed dt so the hoisted
propagators (`hoist_propagator`, default on) apply — at adaptive `cfl_every=1` the step pays
the full putzer2 coefficient cost every stage (~1.9× the FD-z step at 128²×16 on CPU); budget
4·nstage complex full-grid arrays for the hoisted ops, or turn the knob off on a memory-bound
grid.

**fp64 GPU production** needs full-rate-fp64 hardware. Verified candidates as of
2026-07-27: NASA HECC Cabeus (A100 NVLink, plus GH200 nodes), NSF ACCESS DeltaAI, TACC
Vista. Note a quoted state size is aggregate, not per-GPU — a 270 GB state fits on one
4-GPU A100/GH200 node.
