#!/bin/bash
#SBATCH --job-name=xlaflags_2080
#SBATCH --account=fc_kawturb
#SBATCH --partition=savio3_gpu
#SBATCH --qos=gtx2080_gpu3_normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:GTX2080TI:4
#SBATCH --time=01:30:00
#SBATCH --output=xlaflags_2080_%j.out
#SBATCH --error=xlaflags_2080_%j.err
#
# XLA compiler-flag matrix for the jax (shard_map/NCCL) backend on 16 GTX2080TI across 4
# nodes. 2:1 CPU:GPU in this pool -> 32 cores ~= 117 SU/hr, so ~175 SU for the full 1.5 hr.
#
# ---------------------------------------------------------------------------------------
# WHAT THIS ANSWERS
#
# The repo has never set a single XLA flag (grep XLA_FLAGS across slurms/ + bench/ finds
# only NCCL_P2P_DISABLE and a mention of XLA_PYTHON_CLIENT_MEM_FRACTION). Two open
# questions from docs/performance.md depend on that:
#
#   Q1  Are the halo ppermutes and the two allreduces ASYNC on GPU at all? If XLA emits
#       plain synchronous collective-permute, comms are serialized with compute and the
#       `halo_start` hook (T7, recorded as "buys nothing measurable") CANNOT help by
#       construction -- it would be a compiler-scheduling problem, not a physics one.
#
#   Q2  Why is lsrk_scan=False (unrolled) +12% on jax single-node but -38% multi-node?
#       Hypothesis: unrolling exposes 10 independent collective-permutes in one schedule,
#       so XLA can put several in flight at once. Single node has bandwidth to spare and
#       concurrency wins; multi-node here is plain TCP (no IB, PCIe P2P disabled) with a
#       4.2 MB halo payload, where N concurrent transfers each get 1/N of the pipe and ALL
#       land late. If that is the mechanism, latency hiding on/off should FLIP the sign of
#       the scan-vs-unroll delta at 16 GPUs while leaving 4 GPUs alone -- which is exactly
#       the 2x2x2 (flagset x scan/unroll x 4/16 GPUs) measured below.
#
# ---------------------------------------------------------------------------------------
# STRUCTURE  (phases are ordered cheapest-and-most-decisive first)
#
#   0  probe   -- which candidate XLA flags this build actually accepts (names drift
#                 between XLA versions; an unknown one ABORTS the process, so never
#                 assume). Rejected flags are dropped from the matrix, loudly.
#   1  audit   -- bench/hlo_audit.py per flagset: compile only, no timing. Reports whether
#                 collectives are async and how many instructions XLA scheduled inside
#                 each -start/-done window. If a flag does not change the HLO, its timing
#                 row is noise and can be ignored. `grep AUDIT` the .out for the table.
#   2  bench   -- timed ms/step over the same matrix, 2 passes so run-to-run spread is
#                 visible before any delta is believed.
#   3  profile -- one traced case per {base, best-flags} so the static verdict from phase 1
#                 can be checked against what NCCL actually did.
#
# NOTE the persistent compilation cache (RMHD_COMPILATION_CACHE) is deliberately NOT set:
# every case here differs only by XLA_FLAGS, and a cache that keyed on shapes alone would
# silently serve one flagset's binary to another and invalidate the whole experiment.
# Every case paying its own compile is the price of trusting the result.
#
# Companion reading: docs/SAVIO_GPU_SETUP.md (setup, binding, "Reading the profile"),
# docs/performance.md ("Tuning knobs, measured" + "Negative results").

set -uo pipefail   # not -e: one crashed case must not abort the matrix

module purge
module load anaconda3 gcc openmpi
source activate jax_gpu

# The anaconda3 module sets PYTHONPATH to the BASE anaconda site-packages, whose own
# nvidia-* packages shadow the env's (root cause of "cuSPARSE library was not found").
# GPU jobs never need PYTHONPATH (code selection uses RMHD_PKG) -- drop it.
unset PYTHONPATH
echo "python=$(which python)"

# Block ~/.local user-site packages from shadowing the env (stray mpi4py bit us once).
export PYTHONNOUSERSITE=1

# jax's CUDA plugin can fail to dlopen the pip-bundled nvidia libs by bare soname inside
# jobs even when the env is complete -- put every nvidia/*/lib dir on LD_LIBRARY_PATH.
NVLIBS=$("$HOME/.conda/envs/jax_gpu/bin/python" -c "import nvidia,os;print(':'.join(os.path.join(p,d,'lib') for p in nvidia.__path__ for d in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p,d,'lib'))))" 2>/dev/null || true)
[ -n "$NVLIBS" ] && export LD_LIBRARY_PATH="$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "NVLIBS=${NVLIBS:-EMPTY}"   # visible proof in the .out that this block ran

# PCIe GPU P2P is broken on Savio GPU nodes -- NCCL rings connect, then the first
# collective hangs (bench/nccl_repro.py). SHM transport works. Keep this until the cluster
# config changes; it means same-node NCCL is host-memory-limited, so these numbers
# UNDERSTATE what a P2P/NVLink cluster would do.
export NCCL_P2P_DISABLE=1
export RMHD_REQUIRE_GPU=1   # abort a case where a rank silently fell back to CPU

export TARANIS_PRECISION=32   # 2080Ti fp64 is ~1/32 of fp32; the fp64 anchor is the V100 job

PY=$HOME/.conda/envs/jax_gpu/bin/python
REPO=$HOME/taranis
BENCH=$REPO/bench/bench_phase1.py
AUDIT=$REPO/bench/hlo_audit.py
export RMHD_PKG=$REPO   # bench prints pkg= so the imported package is verifiable

MPI_MODE=${MPI_MODE:-pmix}   # this openmpi is --without-pmi + external PMIx; pmi2 aborts
GPUS_PER_NODE=${GPUS_PER_NODE:-$(( SLURM_NTASKS / SLURM_NNODES ))}
PASSES=${PASSES:-2}

NX=${NX:-512}; NZ=${NZ:-256}   # nz_local = 16 at 16 GPUs, 64 at 4; ~0.6 GB/state fits 11 GB
STEPS="nb20 nr6"               # 120 timed steps/case: compile dominates, so buy the samples

# ---------------------------------------------------------------------------- flag matrix
# Indexed (not associative) arrays so this runs under bash 3 as well.
# Each entry is a full XLA_FLAGS string. Keep `base` first: it is the current production
# configuration and every other row is read as a delta from it.
SET_NAME=(base lhs lhspipe lhsdec)
SET_FLAGS=(
  ""
  "--xla_gpu_enable_latency_hiding_scheduler=true"
  "--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_collectives=true"
  "--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_collective_permute_decomposer_threshold=0"
)
# flagsets carried into the (much more expensive) timing phase; the rest stay audit-only
BENCH_SETS_16="base lhs lhspipe"
BENCH_SETS_4="base lhs"

# ------------------------------------------------------------------------ phase 0: probe
# An unrecognized flag makes XLA abort at client init:
#   F0817 ... parse_flags_from_env.cc:234] Unknown flag in XLA_FLAGS: --xla_gpu_bogus
# so a typo is loud rather than silently ignored. Match on that exact message only --
# any OTHER failure (transient cuInit, node trouble) must not be misread as a bad flag.
echo "=== phase 0: which candidate flags does this XLA build accept? ==="
"$PY" -c "import jax; print('jax', jax.__version__)"
KEEP_NAME=(); KEEP_FLAGS=()
for i in "${!SET_NAME[@]}"; do
    name=${SET_NAME[$i]}; flags=${SET_FLAGS[$i]}
    if [ -z "$flags" ]; then
        KEEP_NAME+=("$name"); KEEP_FLAGS+=(""); echo "  $name: (no flags) OK"; continue
    fi
    out=$(XLA_FLAGS="$flags" "$PY" -c "import jax; jax.device_count()" 2>&1)
    case "$out" in
        *"Unknown flag"*)
            echo "  $name: REJECTED by this build -- dropped from the matrix"
            echo "      $flags"
            echo "      $(echo "$out" | grep -o 'Unknown flag in XLA_FLAGS.*' | head -1)" ;;
        *)  KEEP_NAME+=("$name"); KEEP_FLAGS+=("$flags"); echo "  $name: OK  [$flags]" ;;
    esac
done
echo "surviving flagsets: ${KEEP_NAME[*]}"

flags_for() {   # name -> flag string ("" if unknown/dropped)
    local n=$1 i
    for i in "${!KEEP_NAME[@]}"; do
        [ "${KEEP_NAME[$i]}" = "$n" ] && { printf '%s' "${KEEP_FLAGS[$i]}"; return 0; }
    done
    return 1
}

# --------------------------------------------------------------------------- run helpers
# NO --gpu-bind for the jax backend: all job GPUs must stay visible so
# comms._local_device_ids can pin per-process ordinals (docs/SAVIO_GPU_SETUP.md NCCL note).
# --gres is per node, so ask for ceil(n/gpn) nodes.
launch() {   # launch <ngpu> <script> <args...>   (XLA_FLAGS inherited from the caller)
    local n=$1; shift
    local gpn=$(( n < GPUS_PER_NODE ? n : GPUS_PER_NODE ))
    local nn=$(( (n + gpn - 1) / gpn ))
    srun --mpi=$MPI_MODE --nodes="$nn" --ntasks="$n" --ntasks-per-node="$gpn" \
         --cpus-per-task=2 --gres=gpu:GTX2080TI:"$gpn" \
         "$PY" -u "$@" 2>&1 | grep -v "bit precision" || true
}

audit() {    # audit <setname> <ngpu> <scan|unroll>
    local set=$1 n=$2 mode=$3 flags
    flags=$(flags_for "$set") || return 0
    echo "--- audit: flags=$set ngpu=$n $mode ---"
    XLA_FLAGS="$flags" launch "$n" "$AUDIT" nx$NX nz$NZ backend=jax \
        label="$set" $( [ "$mode" = unroll ] && echo unroll ) \
        --dump="$SLURM_SUBMIT_DIR/hlo_$SLURM_JOB_ID"
}

bench() {    # bench <setname> <ngpu> <scan|unroll> [extra bench args...]
    local set=$1 n=$2 mode=$3; shift 3
    local flags tag
    flags=$(flags_for "$set") || return 0
    tag="${set}_g${n}_${mode}"
    XLA_FLAGS="$flags" launch "$n" "$BENCH" "$tag" 3d_forced donate \
        nx$NX nz$NZ nps cfl1 backend=jax $( [ "$mode" = unroll ] && echo unroll ) \
        $STEPS "$@"
}

echo "=== config: precision=$TARANIS_PRECISION grid=${NX}^2x$NZ passes=$PASSES "
echo "===         gpus=$SLURM_NTASKS nodes=$SLURM_NNODES p2p_disable=$NCCL_P2P_DISABLE ==="

# ------------------------------------------------------------------------ phase 1: audit
# Compile-only, so this is cheap relative to phase 2 and it decides which phase-2 rows are
# worth reading. 16 GPUs only: the per-device HLO structure does not depend on the device
# count, just the shard shapes.
echo
echo "############ phase 1: HLO audit (compile only) ############"
for set in "${KEEP_NAME[@]}"; do
    audit "$set" 16 scan
    audit "$set" 16 unroll
done
echo "---- audit summary ----"
echo "(also: grep AUDIT on this file)"

# ------------------------------------------------------------------------ phase 2: bench
# The decisive comparison is scan-vs-unroll at 4 vs 16 GPUs under base-vs-lhs: Q2 predicts
# the sign of (unroll - scan) flips with GPU count, and that latency hiding moves it.
echo
echo "############ phase 2: timed matrix (${PASSES} passes) ############"
for pass in $(seq 1 "$PASSES"); do
    echo "=== pass $pass: 16 GPUs (multi-node, 4 nodes) ==="
    for set in $BENCH_SETS_16; do
        bench "$set" 16 scan
        bench "$set" 16 unroll
    done
    echo "=== pass $pass: 4 GPUs (single node -- the bandwidth-rich control) ==="
    for set in $BENCH_SETS_4; do
        bench "$set" 4 scan
        bench "$set" 4 unroll
    done
done

# ---------------------------------------------------------------------- phase 3: profile
# Traces the timed loop only (rank 0 unless RMHD_PROFILE_ALL=1). Reading it:
# docs/SAVIO_GPU_SETUP.md section 5. What settles Q1: collectives on their own stream
# running CONCURRENTLY with FFT kernels = real overlap; collectives sandwiched between
# kernels with idle gaps = the serialization the flags were supposed to remove.
echo
echo "############ phase 3: profiles (base vs lhs, 16 GPUs, scan) ############"
export RMHD_PROFILE_DIR=$SLURM_SUBMIT_DIR/prof_xlaflags_${SLURM_JOB_ID}_base
bench base 16 scan --profile
export RMHD_PROFILE_DIR=$SLURM_SUBMIT_DIR/prof_xlaflags_${SLURM_JOB_ID}_lhs
bench lhs 16 scan --profile

# ------------------------------------------------------------------------------ wrap-up
echo
echo "############ compact results ############"
echo "--- AUDIT lines (static: did the flags change the HLO?) ---"
grep -h "^AUDIT" "$SLURM_SUBMIT_DIR"/xlaflags_2080_${SLURM_JOB_ID}.out 2>/dev/null || \
    echo "(re-read this file: grep AUDIT)"
echo
echo "--- timing lines (dynamic: did it matter?) ---"
grep -h "ms/step" "$SLURM_SUBMIT_DIR"/xlaflags_2080_${SLURM_JOB_ID}.out 2>/dev/null || \
    echo "(re-read this file: grep ms/step)"
echo
echo "HLO dumps: $SLURM_SUBMIT_DIR/hlo_$SLURM_JOB_ID/"
echo "profiles:  $SLURM_SUBMIT_DIR/prof_xlaflags_${SLURM_JOB_ID}_{base,lhs}/"
echo "done."
