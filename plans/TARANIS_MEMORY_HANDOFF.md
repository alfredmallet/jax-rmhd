# Taranis GPU-memory audit — handoff brief

**Repo:** `github.com/alfredmallet/taranis` @ `c71a3cb` ("hoist putzer computation outside
cfl_every blocks", 2026-08-19 — HEAD at time of audit)
**Question asked:** did the propagator work inflate memory enough to spoil the "pack as much
grid as possible onto each GPU" design goal that motivated the low-storage RK schemes?
**Answer:** yes. The default `z_spectral` + `lsrk54` configuration uses **31× the state
vector** and **2.03× the FD-z path** at the same grid. Roughly two thirds of the excess is
recoverable without changing any numerics.

---

## 1. Method, and its caveats

All numbers are XLA's own `compiled.memory_analysis()` (`temp + arguments + output`) on the
jitted `run.block_of_steps`, plus the `buffer-assignment` / `memory-usage-report` dumps
obtained with `XLA_FLAGS=--xla_dump_to=...`.

**Unit: `u` = one field-sized complex array** = `nz_local · nkx · nky · itemsize`
(8 B fp32 / 16 B fp64). The RMHD state is exactly **2 u**. At 128²×32 fp32, u = 2.031 MiB.

Ratios were checked at 128²×16, ×32 and ×64 and are identical to 2 significant figures at all
three, so they scale linearly and the `u`-normalised numbers can be read off at production
grid sizes.

**Caveats a follow-up should address:**

- Measured on the **CPU backend**, single process, `comm_backend="serial"`, no forcing, no
  particles, `lsrk_scan=True`. GPU fusion and buffer assignment differ; the *structure* of
  the allocation (how many field-sized buffers are simultaneously live) should carry over,
  but the totals want re-checking on an A100 before anyone sizes a run from them.
- fp64 spot-checked and consistent (all `u`-normalised numbers unchanged).
- Multi-rank / `comm_backend="jax"` sharded case not measured.

Reproduction scripts (`memprobe.py`, `memprobe2.py`, `redundancy.py`) accompany this document;
they need only `jax[cpu]` plus `pip install -e .` and go in the repo root.

---

## 2. Headline numbers (fp32, RMHD, 128²×32, `lsrk_scan=True`)

| config | total device memory |
|---|---|
| FD-z (the pre-propagator baseline), lsrk33 / lsrk54 | **30.5 u** |
| `z_spectral`, `hoist_propagator=False`, lsrk33 / lsrk54 | **39.8 u** |
| `z_spectral`, `hoist_propagator=True` (default), lsrk33 | 43.2 u |
| `z_spectral`, `hoist_propagator=True` (default), **lsrk54** | **62.0 u** |

At 512²×128 fp32 (u = 135 MB) that last row is **8.4 GB/GPU** against 4.1 GB for FD-z.

Hoisting is active only when dt is frozen — fixed dt, or `cfl_every > 1` — which is exactly
what `docs/performance.md` recommends for production `z_spectral` runs.

---

## 3. Where the memory actually goes

### 3a. FD-z, lsrk54 — the 30.5 u baseline

From the XLA buffer table:

| u | buffer | what |
|---|---|---|
| 8.00 | `c64[4,2,32,128,65]` | `gradk(fk)` — 4 fields × 2 perp components, still in **k-space** |
| 7.88 | `f32[4,2,32,128,128]` | the same 8 gradients after `ifft`, in **real space** (slot also hosts the 4.00 u `c64[4,32,128,65]` `fk` stack) |
| 2.13 | `c64[2,34,128,65]` | halo-padded field copy for the z stencil |
| 2.13 | `c64[2,34,128,65]` | the second one (`df_dz` / `d4f_dz4`) |
| 2.00 ×4 | `c64[2,32,128,65]` | LSRK `fields`, `delta`, `init_rhs`, NL/FD term temporaries, aliased state I/O |
| ≈28.1 | | + ~2.4 u parameters/output → **30.5 u** |

Two arithmetic traps worth recording, because both were got wrong on first pass:

1. **A real-space full-grid array costs ~1 u, not 0.5 u.** `f32[32,128,128]` = 2.097 MiB vs
   `c64[32,128,65]` = 2.130 MiB. `rfft2` halves the element count but complex64 doubles the
   bytes per element, and the `+1` in `ny/2+1` makes the spectral array marginally *bigger*.
   So 8 real gradient components = 7.9 u.
2. **The gradients are live in both spaces at once**, in two separate slots — 15.9 u, over
   half the FD-z total, for one quantity. See §4a for why.

**This ~28 u baseline is not the propagators' fault** — it predates them and is identical in
the FD-z path where `lin_*` is empty and nothing is hoisted. It is, however, the largest
single target in the code.

### 3b. `z_spectral` + hoisted, lsrk54 — the 62.0 u case

The same dump shows, on top of the above, **four separate `c64[5,32,128,65]` slots at
10.16 MiB each**. That is `propagators.stack_exp_ops` giving `m00/m01/m10/m11` a leading
5-stage axis: `4 × nstage` full-grid complex arrays, **20.3 u, all four simultaneously live**.
Confirmed at the buffer level, not inferred from totals.

Plus 6.0 u of permanently resident `kgrid.lin_*` (§4c), which arrives as a jit argument.

---

## 4. Findings

### 4a. The k-space gradient stack is materialised for no reason — 8 u

```python
# taranis/physics/shared_physics.py
def gradk(fk,kgrid):
    return jnp.stack([1j*kgrid.kx*fk, 1j*kgrid.ky*fk], axis=1)   # (4,2,nz,nkx,nky)

# taranis/physics/rmhd.py
gradients = grids.ifft(gradk(fk,kgrid), params)                  # ONE batched irfft2
```

Nobody chose to store it; two mechanisms force it. `jnp.stack` is a concatenate, so it
materialises rather than staying a fusable view. And XLA treats FFT as an opaque kernel:
elementwise producers cannot fuse into it, its input must be fully materialised, and its
output is a separate allocation. Routing all 8 components through **one** `irfft2` means all
8 k-space components are live while the 8 real outputs are written.

The 8 u buffer carries **2 u of information** — every entry is
`i·k_j × (1 or −k⊥²) × phik/psik`. Regenerating a component is one elementwise multiply,
free next to a transform.

**Verified fix.** Transforming component-by-component, regenerating each from `phik`/`psik`:

- output **bitwise identical** (`np.array_equal` → True, 64²×16 fp32)
- FD-z: **30.48 → 23.55 u** (−6.9 u, −23%)
- `z_spectral` unhoisted: **39.85 → 34.92 u** (−4.9 u)
- `z_spectral` hoisted: **61.97 → 61.97 u** (no change — the peak there is set by the ExpOp
  buffers, so this only pays once hoisting is dealt with)

Trade: 8 `irfft2` calls instead of 1 batched. Each is still a large batch (`nz` planes), so
GPU throughput should be close, but **this needs timing on an A100 before merging.**

### 4b. Hoisting stores `4 × nstage` full-grid complex arrays — 22 u

`c71a3cb` (HEAD, landed the same day as the audit) added `params.hoist_propagator`,
**default `True`**. `timestepping.stage_exp_ops` builds one `Putzer2Exp` per stage;
`stack_exp_ops` stacks them as the stage-scan `xs`.

| scheme | hoist off | hoist on | delta |
|---|---|---|---|
| lsrk33 (3 stages) | 39.8 u | 43.2 u | +3.4 u |
| lsrk54 (5 stages) | 39.8 u | **62.0 u** | **+22.2 u** |

XLA recovers most of the naive `4·nstage` for lsrk33 but not for lsrk54. The perverse
consequence: **hoisting inverts the memory ordering of the schemes.** lsrk54 — chosen
*because* it is a 2-register scheme — becomes the most memory-hungry option in the code,
20 registers' worth of static storage on top of its 2. That is precisely the design goal
being defeated.

`docs/performance.md` already quotes the cost (2.7 GB at 512²×128 fp32 lsrk54). The audit's
view is that that number should be a refusal, not a footnote.

The unrolled path (`lsrk_scan=False`) is a red herring — XLA hoists these itself there
(43.7 u hoisted *and* unhoisted for lsrk54), so `hoist_propagator=False` only genuinely buys
memory back on the scan path.

### 4c. `kgrid.lin_*` holds 6 u permanently, of which ~0.5% is information

```
lin_L      (2, 2, nz, nkx, nky)  complex   4.00 u
lin_m      (nz, nkx, nky)        complex   1.00 u
lin_s2     (nz, nkx, nky)        complex   1.00 u
                                          ------
                                           6.00 u   always resident (3× the state vector)
```

Verified element-wise at fp64, for both `nu = eta` and `nu ≠ eta`:

- `L01 == L10 == 1j·kz` exactly, with **zero perpendicular variation** — `nz` numbers stored
  as two full-grid complex arrays. `rmhd.linear_matrix` does this deliberately
  (`off = jnp.broadcast_to(1j*kz, dphi.shape)` then `jnp.stack`).
- `L00`, `L11` are **exactly real** and **exactly rank-1 separable** into
  `-diss·k⊥^{2h}` (perp only) and `-z_diss_k·kz⁴` (kz only); separable-reconstruction
  residual is 0.0 to the last bit. They are stored complex only because a *different* block
  of the stacked `L` is imaginary.
- `lin_m` is **exactly real** and equals `(L00+L11)/2` — derived, separable, stored complex.
- `lin_s2` is **exactly real**, and for `nu == eta` is **exactly `-kz²`**
  (`np.allclose(s2, -kz**2)` → True) — i.e. `nz` numbers in a full complex array.

Information content `2·nz + 2·nkx·nky`; storage `6·nz·nkx·nky` complex. At 512²×128 that is
**806 MB resident to hold ~260k independent numbers.**

GDI's `L` genuinely is full-grid in all four blocks, so this compression is RMHD-specific.

### 4d. `init_rhs` is a third LSRK register — 2 u

`_lsrk_scan_stages` keeps `init_rhs` alive across the whole stage scan so
`lax.cond(istage == 0, ...)` can reuse it. Measured at **exactly 2.0 u** (39.8 → 37.8 u with
the cond removed; 30.5 → 28.5 u on FD-z). One extra full field vector in a two-register
scheme.

Do *not* fix it by dropping the reuse — that costs a whole extra RHS evaluation per step
(+25% for lsrk54). Fix it by **peeling stage 0 out of the scan** and scanning stages
`1..s-1`; `init_rhs`'s liveness then ends before the scan, at zero arithmetic cost.
`_imex2r_scan_stages` already uses this pattern, so there is a model to copy in-file.

---

## 5. Options, ranked

| | change | Δ | status |
|---|---|---|---|
| **B2** | **Factored RMHD propagator (Elsässer).** For `nu == eta`, `L = a(k)·I + i·kz·σₓ`, so `exp(Lτ) = e^{aτ}[cos(kzτ)I + i sin(kzτ)σₓ]`, and `e^{aτ} = exp(−nu τ k⊥^{2h}) · exp(−z_diss_k τ kz⁴)` — a perp-only array times a kz-only array. Per stage: one `(nkx,nky)` array + three length-`nz` arrays instead of 4 full-grid complex. | **−22 u** | Exact, not an approximation. Keeps the hoisted speed. Rotation `(1,±1)/√2` is k-independent, so 2 adds + 2 subtracts per apply. Most work: new propagator backend. For `nu ≠ eta` the constant rotation no longer diagonalises, but `s²` is still separable. |
| **B1** | Default `hoist_propagator=False` | **−22.2 u** | One line. Costs the ~1.7× step speedup quoted in `docs/performance.md`. Stopgap until B2. |
| **A** | Chunk the gradient transform in `rmhd.grad` (§4a) | **−6.9 u** FD-z, **−4.9 u** z_spectral unhoisted, **0** hoisted | Verified bitwise identical. ~15 lines. **Needs GPU timing** for the 8-calls-vs-1-batched trade. |
| **C** | Broadcastable `L` blocks: let `linear_matrix_func` return the four blocks separately, each shaped `(nz-or-1, nkx-or-1, nky-or-1)`, instead of one stacked `(2,2,nz,nkx,nky)` array; let `lin_m`/`lin_s2` keep their natural shape and dtype | **−6.0 u** persistent | No numerics change — `Putzer2Exp.apply`, `apply_L`, `solve_shifted`, `scaled` are all elementwise multiplies that broadcast as written. Only `propagators.linear_fields`'s shape validator and `_check_hermitian_compatible` (call it on a `np.broadcast_to` view at setup) need work. `gdi.linear_matrix` already builds `L00,L01,L10,L11` separately and stacks at the end, so it adapts in one line. |
| **D** | Peel stage 0 out of `_lsrk_scan_stages` (§4d) | **−2.0 u** | Free. Pattern already in-file. |
| **B3** | Cache `s = sqrt(s2)` at setup — the "available, not done" item already in `docs/performance.md` | **0 u** | Most of the hoist speedup at zero memory, *and* it helps the adaptive `cfl_every=1` path, which hoisting cannot touch at all. For RMHD `s = i·kz` exactly, so the cache is `nz` numbers. |
| E | Reorder the brackets in `NonlinearTerm` so only 6 of the 8 real gradients are live at peak | ~−2 u (est.) | **Unmeasured.** `bracket(gphi,gpsi)` and `bracket(gphi,gvort)` first, free `gvort`, then `bracket(gpsi,gjpar)`. |

**Ordering matters:** A shows zero gain on the current default because the peak is set by the
ExpOp buffers. **Do B first, then A pays.**

### Cumulative projection, `z_spectral` + lsrk54

```
62.0 u  today (default)
39.9 u  after B (measured)
34.9 u  after B + A (measured)
~28.9 u after B + A + C (C is a 6.0 u argument buffer, subtraction is exact)
~26.9 u after B + A + C + D
```

**2.3× more grid per GPU, landing below today's FD-z number.** FD-z itself goes
30.5 → ~21.6 u with A + D.

---

## 6. Suggested next steps

1. **B1 now** — one-line default flip, recovers 22 u immediately on lsrk54 runs.
2. **D and A** — both small, both bitwise-safe (A verified; D is a pure liveness change).
   A needs a GPU timing run first.
3. **C** — contained to `linear_fields` + the two recipes.
4. **B2** — the design decision worth thinking about properly, since it changes the
   propagator API. It makes B1 moot and recovers the speed.

### Open questions for whoever picks this up

- Re-run the whole table on an A100 (`comm_backend="jax"`, sharded) — CPU-backend buffer
  assignment is indicative, not authoritative.
- Time the chunked `irfft2` (8 calls vs 1 batched) on GPU at production grid sizes.
- Does `tests/test_hoist_propagator.py`'s bitwise gate survive B2? It pins hoisted against
  unhoisted at fp64; a factored exponential changes op order, so that gate probably needs to
  become a tolerance rather than a bitwise comparison. `docs/performance.md` already warns
  against pinning hoisted putzer2 runs bitwise across jax versions.
- Does the FD-z / 2D diagonal path stay byte-identical under A? (It should — A is bitwise
  identical on `grad` — but gate 6's reference npz is the thing to check.)
- `nu ≠ eta`: confirm the separable form of `s²` before assuming B2 generalises.

## 7. Files

- `memprobe.py` — the config sweep table (§2) plus the `kgrid` entry breakdown
- `memprobe2.py` — scaling check across `nz`, scheme, `lsrk_scan`, hoist
- `redundancy.py` — the element-wise `lin_*` redundancy proofs (§4c)
- `PROPAGATOR_MEMORY_AUDIT.md` — the original longer-form audit this brief condenses

Buffer-level dumps are reproduced with
`XLA_FLAGS="--xla_dump_to=<dir> --xla_dump_hlo_pass_re=none"`, then read
`module_*.jit_block_of_steps.cpu_after_optimizations-memory-usage-report.txt`.
