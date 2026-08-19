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
