# Compile-time XLA audit of one production step, without running a full benchmark.
#
# Answers "is XLA doing the right thing on multi-GPU?" from the OPTIMIZED, SCHEDULED HLO:
# how many collectives per step, whether they are async (-start/-done pairs) and how much
# compute XLA managed to slot between them, how many FFTs, and the peak temp memory.
# It compiles only -- no device execution, no MPI collectives, no timing -- so it runs on
# a login node or a laptop.
#
# usage:
#   python bench/hlo_audit.py [nx<N>] [nz<N>] [backend=jax|mpi4jax|serial] [scheme=lsrk54]
#                             [ndev<N>] [nblock<N>] [cfl<N>] [unroll] [nonps]
#                             [label=<tag>] [--dump=DIR]
#
# Prints a full report plus one greppable `AUDIT <label> ...` line per run, so a flag
# matrix (slurms/bench_xla_flags_2080.sh) collapses to `grep AUDIT job.out`.
#
# On a laptop, ndev<N> fakes N host devices (XLA_FLAGS=--xla_force_host_platform_device_count)
# so the shard_map/ppermute path compiles for a real N-way mesh. That gives the op COUNTS
# and the shard shapes, but CPU never emits async collectives -- for the async/overlap
# verdict run this inside a GPU allocation with the real device count.
#
# Companion to docs/performance.md ("Tuning knobs, measured") and the --profile case in
# bench/bench_phase1.py: this is the static half, the profile is the dynamic half.
import collections
import os
import re
import sys

_argv = sys.argv[1:]


def _kwarg(name, default):
    for a in _argv:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def _numarg(name, default):
    for a in _argv:
        if m := re.fullmatch(rf"{name}(\d+)", a):
            return int(m.group(1))
    return default


NX = _numarg("nx", 128)
NZ = _numarg("nz", 64)
NDEV = _numarg("ndev", 0)
NBLOCK = _numarg("nblock", 1)
CFL_EVERY = _numarg("cfl", 1)
BACKEND = _kwarg("backend", "jax")
SCHEME = _kwarg("scheme", "lsrk54")
DUMP = _kwarg("--dump", None)
LABEL = _kwarg("label", "run")   # tags the one-line AUDIT summary (flag-matrix jobs)

# must precede the first jax device call
if NDEV:
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (flags + f" --xla_force_host_platform_device_count={NDEV}").strip()

# Select the package version via RMHD_PKG=<dir>, exactly as bench/bench_phase1.py does
# (a PEP-660 editable install registers a meta-path finder that silently beats PYTHONPATH),
# falling back to this file's own repo root so the audit runs from a checkout with no
# install at all -- what bench/memory_probe.py does.
_pkgdir = os.environ.get("RMHD_PKG") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.meta_path = [f for f in sys.meta_path
                 if "taranis" not in (getattr(f, "__module__", "") or "")]
sys.path.insert(0, _pkgdir)

import jax                     # noqa: E402
import jax.numpy as jnp        # noqa: E402
import numpy as np             # noqa: E402
import taranis as jr           # noqa: E402
from taranis import comms, run as jrun          # noqa: E402
from taranis.timestepping import get_scheme     # noqa: E402

assert jr.__file__.startswith(os.path.abspath(_pkgdir) + os.sep), \
    f"wrong taranis imported: {jr.__file__} (wanted {_pkgdir})"

L = 2 * np.pi
p = jr.Parameters(nx=NX, ny=NX, nz=NZ, Lx=L, Ly=L, Lz=L, dims=3, z_diss=0.25,
                  eqpars={"diss": (1e-4, 1e-4), "hyper": 2}, cfl_safety=0.5,
                  forcing=True, forcing_mode="elsasser", forcing_power_elsasser=(0.5, 0.5),
                  forcing_tau=1.0, fshell=(1, 3), forcing_seed=1,
                  comm_backend=BACKEND, cfl_every=CFL_EVERY)
p.lsrk_scan = "unroll" not in _argv
p.forcing_norm_per_step = "nonps" not in _argv   # production default; `nonps` reverts it

# Every rank compiles (the shard_map lowering is collective), but only rank 0 reports --
# 16 ranks each dumping a full audit is unreadable. RMHD_AUDIT_ALL=1 overrides.
RANK0 = p.rank == 0 or os.environ.get("RMHD_AUDIT_ALL")
def say(*a, **kw):
    if RANK0:
        print(*a, **kw)

kg = jr.setup_kgrids(p)
ic = lambda x, y, z: jnp.stack([jnp.cos(x + 1.4) + jnp.cos(y + 2.0) * jnp.cos(z),
                                jnp.cos(2 * x + 2.3) * jnp.cos(z) + 0.5 * jnp.cos(y)])
state = jrun.initialize(ic, p)
stepper, scheme = get_scheme(SCHEME)

if BACKEND == "jax":
    fn = jax.jit(comms.shard_call(
        lambda s, k: jrun.block_of_steps(s, k, p, NBLOCK, scheme, stepper), p, kg),
        donate_argnums=(0,))
    compiled = fn.lower(state, kg).compile()
else:
    fn = jax.jit(jrun.block_of_steps, static_argnums=(2, 3, 4, 5), donate_argnums=(0,))
    compiled = fn.lower(state, kg, p, NBLOCK, scheme, stepper).compile()

text = compiled.as_text()
lines = text.splitlines()

# ---------------------------------------------------------------- op inventory
COLLECTIVES = ("all-reduce", "collective-permute", "all-gather", "reduce-scatter",
               "all-to-all", "collective-broadcast")
# shape may be a tuple type containing spaces, e.g. "(f32[2]{0}, f32[2]{0})"
op_re = re.compile(r"^\s*(?:ROOT\s+)?%?(\S+) = (.+?) ([a-z0-9\-]+)\(")
by_op = collections.Counter()
instrs = []          # (index, name, shape, opcode)
for i, line in enumerate(lines):
    if m := op_re.match(line):
        name, shape, opcode = m.groups()
        by_op[opcode] += 1
        instrs.append((i, name, shape, opcode))

def _kind(opcode):
    return opcode.rsplit("-", 1)[0] if opcode.endswith(("-start", "-done")) else opcode

coll = [x for x in instrs if _kind(x[3]) in COLLECTIVES]
ffts = [x for x in instrs if x[3] == "fft"]
n_async = sum(1 for x in coll if x[3].endswith("-start"))
n_sync = sum(1 for x in coll if not x[3].endswith(("-start", "-done")))

say("\n=== taranis HLO audit ===")
say(f"grid {NX}x{NX}x{NZ}  backend={BACKEND}  scheme={SCHEME}  nblock={NBLOCK} "
      f"cfl_every={CFL_EVERY}  lsrk_scan={p.lsrk_scan}")
say(f"platform={jax.default_backend()}  devices={jax.device_count()}  "
      f"mesh={comms.get_mesh().size if BACKEND == 'jax' else 1}  "
      f"nz_local={p.nz // max(jax.device_count() if BACKEND == 'jax' else p.size, 1)}")

say(f"\n-- FFTs ({len(ffts)} calls in the module) --")
for _, name, shape, _ in ffts:
    ftype = re.search(rf"{re.escape(name)} = .*?fft_type=(\w+)", text)
    say(f"   {shape:32s} {ftype.group(1) if ftype else '?'}")

say(f"\n-- collectives ({n_sync} synchronous, {n_async} async pairs) --")
for _, name, shape, opcode in coll:
    if opcode.endswith("-done"):
        continue
    ctx = re.search(rf"{re.escape(name)} = .*?op_name=\"([^\"]*)\"", text)
    where = (ctx.group(1).split("/", 1)[-1] if ctx else "?")
    say(f"   {opcode:26s} {shape:24s} {where}")

# ------------------------------------------------- async overlap window (GPU only)
def _overlap_gaps():
    # for each async collective, how many instructions XLA scheduled between its -start
    # and its -done. That IS the overlap: 0 means the collective blocks immediately.
    starts = {name: i for i, name, _, op in instrs if op.endswith("-start")}
    out = []
    for i, name, shape, opcode in instrs:
        if not opcode.endswith("-done"):
            continue
        operand = re.search(rf"{re.escape(name)} = .*?-done\(%?(\S+?)\)", text)
        src = operand.group(1) if operand else None
        if src in starts:
            out.append((sum(1 for j, *_ in instrs if starts[src] < j < i), _kind(opcode), shape))
    return out

if n_async:
    say("\n-- overlap windows (instructions scheduled between -start and -done) --")
    for gap, kind, shape in _overlap_gaps():
        verdict = "OVERLAPPED" if gap > 3 else "*** NOT OVERLAPPED ***"
        say(f"   {kind:26s} {shape:24s} {gap:4d} instrs   {verdict}")
else:
    say("\n   NOTE: no async collectives in this module. Expected on CPU; on GPU it means "
          "\n   XLA is serializing comms with compute -- see the flag checklist below.")

say("\n-- top ops --")
for k, v in by_op.most_common(12):
    say(f"   {k:26s} {v}")

mem = compiled.memory_analysis()
mb = 1024 ** 2
say("\n-- memory (per device) --")
say(f"   argument {mem.argument_size_in_bytes/mb:8.1f} MB   output {mem.output_size_in_bytes/mb:8.1f} MB"
      f"   temp {mem.temp_size_in_bytes/mb:8.1f} MB")
state_mb = 2 * (p.nz // max(jax.device_count() if BACKEND == "jax" else p.size, 1)) \
           * p.nx * (p.ny // 2 + 1) * (8 if os.environ.get("TARANIS_PRECISION") != "64" else 16) / mb
say(f"   one state array {state_mb:.1f} MB -> temp is {mem.temp_size_in_bytes/mb/state_mb:.1f}x a state")

# halo/volume ratio: the fundamental z-only-decomposition scaling limit
ndev = jax.device_count() if BACKEND == "jax" else p.size
nz_local = p.nz // max(ndev, 1)
say("\n-- decomposition --")
say(f"   nz_local={nz_local}, halo width 2 each side -> {4/nz_local:.0%} of the local "
      f"field is exchanged per RHS evaluation")
if nz_local < 8:
    say("   *** nz_local < 8: the halo is a large fraction of local data; expect poor scaling")

if DUMP:
    if RANK0:
        os.makedirs(DUMP, exist_ok=True)
        path = os.path.join(DUMP, f"optimized_{LABEL}_{BACKEND}_{NX}_{NZ}"
                                  f"_{'scan' if p.lsrk_scan else 'unroll'}.txt")
        with open(path, "w") as fh:
            fh.write(text)
        say(f"\nwrote {path} ({len(lines)} lines)")

# One greppable line per run, so a flag matrix collapses to `grep AUDIT job.out`.
# overlap_min/med are the -start..-done instruction gaps: 0 means no overlap at all.
gaps = sorted(g for g, _, _ in _overlap_gaps())
say(f"AUDIT {LABEL:14s} dev={ndev:<3d} {'scan' if p.lsrk_scan else 'unroll':6s} "
    f"coll_sync={n_sync:<3d} coll_async={n_async:<3d} "
    f"overlap_min={gaps[0] if gaps else 0:<5d} overlap_med={gaps[len(gaps)//2] if gaps else 0:<5d} "
    f"ffts={len(ffts):<3d} fusions={by_op.get('fusion', 0):<4d} "
    f"temp_MB={mem.temp_size_in_bytes/mb:.0f} temp_x_state={mem.temp_size_in_bytes/mb/state_mb:.1f} "
    f"hlo_lines={len(lines)}")

say("""
-- what to check next, on real GPUs --
1. Run this inside the GPU allocation, with the real rank/GPU count. Every collective
   should print as -start/-done with a non-trivial overlap window.
2. If they are synchronous, or the window is ~0, try (one at a time, then benchmark):
     XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
     XLA_FLAGS="$XLA_FLAGS --xla_gpu_collective_permute_decomposer_threshold=0"
     XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_pipelined_collectives=true"
   and re-run this audit to confirm the HLO actually changed before timing anything.
3. Cross-check dynamically: bench/bench_phase1.py ... --profile, then look at the GPU
   trace. Collectives on their own stream, concurrent with FFT kernels = overlap is real.
   See docs/SAVIO_GPU_SETUP.md section 5.
""")
