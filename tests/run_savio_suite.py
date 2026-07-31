#!/usr/bin/env python
# Savio test-suite driver (docs/TESTING_PLAN.md Phase 6). Launched INSIDE an sbatch
# allocation by slurms/run_test_suite_{cpu,gpu}.sh -- or locally for a smoke test of
# the runner itself (RMHD_LAUNCH="" runs the "mpi" phases single-process).
#
# pytest is never run under mpirun: every phase subprocess-runs one test file in
# script mode under a launcher prefix, tees its output to a per-phase log under
# $RMHD_TEST_OUTDIR, and applies the pass rule
#     exit 0  AND  (banner: "ALL PASS" printed, "SOME CHECKS FAILED" absent)
# -- the banner half matters because mpirun can mask a single rank's exit code.
#
# Launcher templates come from the environment so the cluster knowledge stays in the
# slurm wrappers ("{n}" is replaced by the phase's rank count):
#     RMHD_LAUNCH           "mpi" phases          default: mpirun -n {n}
#     RMHD_LAUNCH_MPI4JAX   "gpu_mpi4jax" phases  REQUIRED for the gpu tier
#     RMHD_LAUNCH_JAX       "gpu_jax" phases      REQUIRED for the gpu tier
# ("serial" phases run bare `python -u script`. Wrap `timeout N` into the template
# to turn hangs into failures -- the gpu wrapper does.)
#
# Each (job, precision) pair gets a CLEAN scratch cwd under $RMHD_TEST_OUTDIR shared
# by that job's phases: multi-phase sequences (resharding 2->4, cross-backend
# restarts) work, and cwd-relative data/ writes can never hit a stale dir.

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from savio_manifest import JOBS  # noqa: E402

_LAUNCH = {
    "mpi": ("RMHD_LAUNCH", "mpirun -n {n}"),
    "gpu_mpi4jax": ("RMHD_LAUNCH_MPI4JAX", None),
    "gpu_jax": ("RMHD_LAUNCH_JAX", None),
}


def _prefix(launch, n):
    if launch == "serial":
        return []
    var, default = _LAUNCH[launch]
    tmpl = os.environ.get(var, default)
    if tmpl is None:
        raise SystemExit(f"error: {var} must be exported (by the slurm wrapper) "
                         f"for launch={launch!r} phases")
    return shlex.split(tmpl.format(n=n))


def _run_phase(cmd, cwd, env, log_path):
    """Run cmd, teeing combined stdout/stderr to our stdout AND log_path.
    Returns (returncode, full_text)."""
    lines = []
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
            lines.append(line)
        rc = proc.wait()
    sys.stdout.flush()
    return rc, "".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=("cpu", "gpu", "all"), default="all")
    ap.add_argument("--only", action="append", default=[],
                    help="run only jobs whose name contains this substring (repeatable)")
    ap.add_argument("--precision", choices=("64", "32"),
                    help="run every selected job at this single precision only")
    ap.add_argument("--list", action="store_true", help="print the job table and exit")
    args = ap.parse_args(argv)

    jobs = [j for j in JOBS if args.tier in ("all", j["tier"])]
    if args.only:
        jobs = [j for j in jobs if any(s in j["name"] for s in args.only)]
    if args.list or not jobs:
        for j in jobs or JOBS:
            phases = ", ".join(f"{p['launch']}:{p['n']}" for p in j["phases"])
            print(f"{j['tier']:3s}  {j['name']:26s} {j['script']:38s} [{phases}] "
                  f"fp{'/'.join(j.get('precisions', ('64',)))}")
        return 0 if jobs else 2

    outdir = os.path.abspath(os.environ.get("RMHD_TEST_OUTDIR")
                             or os.path.join(os.getcwd(), "data",
                                             f"test_suite_{time.strftime('%Y%m%d_%H%M%S')}"))
    os.makedirs(outdir, exist_ok=True)
    print(f"== run_savio_suite: {len(jobs)} job(s), tier={args.tier}, logs in {outdir}\n")

    results = []  # (job, precision, label, ok, detail)
    for job in jobs:
        precisions = (args.precision,) if args.precision else job.get("precisions", ("64",))
        banner = job.get("banner", True)
        for prec in precisions:
            jobdir = os.path.join(outdir, f"{job['name']}_fp{prec}")
            shutil.rmtree(jobdir, ignore_errors=True)  # clean per-job dir: no stale
            os.makedirs(jobdir)                        # snapshot layouts, fresh phases
            job_ok = True
            for i, ph in enumerate(job["phases"], 1):
                label = ph.get("label", f"n{ph['n']}")
                if not job_ok:
                    results.append((job["name"], prec, label, None,
                                    "skipped: earlier phase failed"))
                    continue
                script = os.path.join(REPO, job["script"])
                cmd = (_prefix(ph["launch"], ph["n"])
                       + [sys.executable, "-u", script] + list(ph.get("args", ())))
                env = dict(os.environ, RMHD_PRECISION=prec, **ph.get("env", {}))
                log = os.path.join(jobdir, f"phase{i}_{label}.log")
                print(f"=== {job['name']} [fp{prec}] phase {i}/{len(job['phases'])} "
                      f"({label}): {' '.join(cmd)}")
                t0 = time.time()
                rc, text = _run_phase(cmd, jobdir, env, log)
                ok = (rc == 0 and (not banner
                                   or ("ALL PASS" in text
                                       and "SOME CHECKS FAILED" not in text)))
                detail = f"rc={rc}, {time.time() - t0:.0f}s, log={os.path.relpath(log, outdir)}"
                if rc == 0 and not ok:
                    detail += " (exit 0 but banner check failed)"
                print(f"=== {'ok' if ok else 'FAIL'}: {job['name']} [fp{prec}] {label} "
                      f"({detail})\n")
                results.append((job["name"], prec, label, ok, detail))
                job_ok = job_ok and ok

    print("\n== summary " + "=" * 60)
    failed = 0
    for name, prec, label, ok, detail in results:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        failed += status == "FAIL"
        print(f"[{status}] {name:26s} fp{prec}  {label:26s} {detail}")
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} PHASE(S) FAILED'} "
          f"({len(results)} phases; logs in {outdir})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
