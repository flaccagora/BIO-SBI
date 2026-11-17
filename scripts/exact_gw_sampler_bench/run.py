"""Benchmark the exact_gw sampler options (warm start, ESS / R-hat stopping) on XG2 data.

Per truth, seed 0, n = 128 -> 256 -> 512 -> 1024 (XG2 datasets are nested in n):

* ``baseline``: adjoint kernel, cold start, 4 chains x (500 warmup + 500 draws) (XG2);
* ``warm``: n = 128 is the baseline run; n > 128 warm-started from this arm's run at n / 2;
* ``stop``: n = 128 cold with the stopping rule; n > 128 warm start + stopping rule from
  this arm's run at n / 2.

Runs ``--parallel`` jobs at a time (4 threads each) as soon as their dependency is done;
skips jobs whose ``metrics.json`` exists. Usage::

    uv run python scripts/exact_gw_sampler_bench/run.py --out results/exact_gw/sampler_bench \
        --truths T1_ref T5_C7 --parallel 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

#: Truths not in the experiment file (as in scripts/run_xg2_adjoint.sh).
EXTRA = {
    "T5_C7": "+truths.T5_C7={source:reference,seed:null,values:{A:{death:0.05,persist:0.10,"
    "to_B:0.10,div_AA:0.50,div_AB:0.15,div_BB:0.10},B:{death:0.08,persist:0.05,to_A:0.10,"
    "div_AA:0.05,div_AB:0.05,div_BB:0.67}}}",
}
SIZES = [128, 256, 512, 1024]


def jobs_for(gt: str, out: Path, seed: int, stop_opts: list[str]) -> list[dict]:
    """Jobs (output directory, overrides, dependency) of one truth."""
    base = ["+experiment=exact_gw_data_size", "inference.method.kernel=adjoint", f"gt={gt}",
            f"seed={seed}"]  # fmt: skip
    if gt in EXTRA:
        base.insert(1, EXTRA[gt])
    stop = ["inference.method.stop.enabled=true", *stop_opts]
    jobs = []

    def d(arm, n):
        return out / gt / arm / f"n{n}_s{seed}"

    for n in SIZES:
        jobs.append({"dir": d("baseline", n), "args": [*base, f"design.clones={n}"], "dep": None})
    for i, n in enumerate(SIZES[1:]):
        prev = d("baseline", 128) if i == 0 else d("warm", SIZES[i])
        jobs.append({"dir": d("warm", n), "dep": prev,
                     "args": [*base, f"design.clones={n}",
                              f"inference.method.warm_start={prev}"]})  # fmt: skip
    for i, n in enumerate(SIZES):
        prev = None if i == 0 else d("stop", SIZES[i - 1])
        args = [*base, f"design.clones={n}", *stop]
        if prev is not None:
            args.append(f"inference.method.warm_start={prev}")
        jobs.append({"dir": d("stop", n), "dep": prev, "args": args})
    return jobs


def main() -> None:
    """Run the jobs as their dependencies finish, ``--parallel`` at a time."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--truths", nargs="+", default=["T1_ref", "T5_C7"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--stop", nargs="*", default=[], help="extra inference.method.stop.* overrides")
    a = p.parse_args()
    jobs = [j for gt in a.truths for j in jobs_for(gt, a.out.resolve(), a.seed, a.stop)]
    # critical paths first: warm / stop chains, then the independent baselines
    jobs.sort(key=lambda j: (j["dir"].parent.name == "baseline" and j["dir"].name[:4] != "n128",))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}  # fmt: skip
    running: dict[Path, subprocess.Popen] = {}
    done = {j["dir"] for j in jobs if (j["dir"] / "metrics.json").exists()}
    failed: set[Path] = set()
    pending = [j for j in jobs if j["dir"] not in done]
    while pending or running:
        for path, proc in list(running.items()):
            if proc.poll() is not None:
                del running[path]
                ok = proc.returncode == 0 and (path / "metrics.json").exists()
                (done if ok else failed).add(path)
                print(time.strftime("%H:%M:%S"), "finished" if ok else "FAILED", path, flush=True)
        for j in list(pending):
            if len(running) >= a.parallel:
                break
            if j["dep"] in failed:
                pending.remove(j)
                failed.add(j["dir"])
                print("skip (dependency failed)", j["dir"], flush=True)
                continue
            if j["dep"] is not None and j["dep"] not in done:
                continue
            pending.remove(j)
            j["dir"].mkdir(parents=True, exist_ok=True)
            cmd = ["uv", "run", "bio-sbi", *j["args"], f"hydra.run.dir={j['dir']}"]
            log = open(j["dir"].with_suffix(".log"), "w")  # noqa: SIM115
            log.write(" ".join(cmd) + "\n")
            log.flush()
            running[j["dir"]] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                 env=env)  # fmt: skip
            print(time.strftime("%H:%M:%S"), "started", j["dir"], flush=True)
        time.sleep(10)
    print("done:", len(done), "ok,", len(failed), "failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
