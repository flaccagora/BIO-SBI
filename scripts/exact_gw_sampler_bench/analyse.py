"""Tables for the exact_gw sampler benchmark (see run.py in this folder).

Per run: sampler wall time, gradient evaluations (warmup / sampling), min bulk / tail ESS
(rank-normalised), ESS per minute, max rank R-hat, divergences, idle-thread fraction,
stop reason; and posterior agreement with the baseline of the same cell and with the XG2
run of the same cell (|difference of means| in Monte Carlo standard errors, sd ratio).

    uv run python scripts/exact_gw_sampler_bench/analyse.py results/exact_gw/sampler_bench \
        --xg2 results/exact_gw/data_size
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ["baseline", "warm", "stop"]


def load(run: Path) -> dict | None:
    """Metrics, exact_gw diagnostics and free-parameter draws of a run (None if absent)."""
    m = run / "metrics.json"
    if not m.exists():
        return None
    metrics = json.loads(m.read_text())
    e = metrics["exact_gw"]
    post = pd.read_csv(run / "posterior_samples.csv.gz")
    free = list(e["chain_means"])
    return {"metrics": metrics, "e": e, "post": post[free], "free": free, "dir": run}


def ess_of(e: dict, name: str) -> float:
    """Bulk ESS of ``name`` (older runs: the plain ESS)."""
    for key in ("ess_bulk", "ess"):  # XG2 runs have the older ESS only
        if key in e and name in e[key]:
            return float(e[key][name])
    return float("nan")


def agreement(a: dict, b: dict) -> dict:
    """Max / median |mean_a - mean_b| / MCSE over the free parameters, max sd ratio."""
    z, ratio = [], []
    for name in a["free"]:
        xa, xb = a["post"][name].to_numpy(), b["post"][name].to_numpy()
        sa, sb = xa.std(ddof=1), xb.std(ddof=1)
        se = np.sqrt(sa**2 / ess_of(a["e"], name) + sb**2 / ess_of(b["e"], name))
        if se > 0:
            z.append(abs(xa.mean() - xb.mean()) / se)
        if sa > 0 and sb > 0:
            ratio.append(sa / sb)
    z = np.asarray(z)
    ratio = np.asarray(ratio)
    return {"max_z": float(z.max()), "median_z": float(np.median(z)),
            "frac_z_gt_3": float((z > 3).mean()),
            "sd_ratio_min": float(ratio.min()), "sd_ratio_max": float(ratio.max())}  # fmt: skip


def row(gt: str, arm: str, n: int, r: dict) -> dict:
    """One table row of a run."""
    e = r["e"]
    sec = float(e["seconds"])
    stop = e.get("stop", {})
    warm = e.get("warm_start", {})
    min_bulk = e.get("min_ess_bulk", float("nan"))
    return {
        "gt": gt, "arm": arm, "n": n,
        "wall_s": round(sec),
        "warmup": e["warmup"], "draws_per_chain": e["draws_per_chain"],
        "evals_warmup": e.get("warmup_gradient_evals"),
        "evals_sampling": e.get("sampling_gradient_evals"),
        "evals_total": e["gradient_evals"],
        "min_ess_bulk": round(min_bulk), "min_ess_tail": round(e.get("min_ess_tail", np.nan)),
        "ess_per_min": round(min_bulk / (sec / 60), 1),
        "max_rank_rhat": round(e.get("max_rank_rhat", np.nan), 4),
        "max_mean_spread": round(e.get("max_mean_spread", np.nan), 2),
        "divergences": e["divergences"], "warmup_divergences": e["warmup_divergences"],
        "accept": "/".join(f"{a:.2f}" for a in e["accept_stat"]),
        "step": "/".join(f"{s:.3f}" for s in e["step_size"]),
        "depth": "/".join(f"{d:.1f}" for d in e["mean_tree_depth"]),
        "idle_fraction": round(e.get("idle_fraction", np.nan), 3),
        "stop_reason": stop.get("reason", ""),
        "stop_failed": ",".join(stop.get("failed", [])),
        "warm_scale": warm.get("scale"),
        "warm_start_chains": warm.get("start_previous_chain"),
        "s_per_eval": round(e["seconds_per_eval"], 4),
    }  # fmt: skip


def main() -> None:
    """Print and save the tables."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("root", type=Path)
    p.add_argument("--xg2", type=Path, default=None, help="XG2 root (<gt>/n<n>_s<seed>/)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rows, agree = [], []
    for gt_dir in sorted(d for d in a.root.iterdir() if d.is_dir()):
        gt = gt_dir.name
        runs = {}
        for arm in ARMS:
            for n in (128, 256, 512, 1024):
                r = load(gt_dir / arm / f"n{n}_s{a.seed}")
                if r is None and arm == "warm" and n == 128:
                    r = load(gt_dir / "baseline" / f"n{n}_s{a.seed}")  # warm arm starts cold
                if r is not None:
                    runs[arm, n] = r
                    rows.append(row(gt, arm, n, r))
        for (arm, n), r in runs.items():
            ref = runs.get(("baseline", n))
            same_data = None
            if a.xg2 is not None:
                x = load(a.xg2 / gt / f"n{n}_s{a.seed}")
                if x is not None:
                    d0 = np.load(r["dir"] / "reference.npz")
                    d1 = np.load(a.xg2 / gt / f"n{n}_s{a.seed}" / "reference.npz")
                    same_data = all(np.array_equal(d0[k], d1[k]) for k in d1.files)
                    agree.append({"gt": gt, "arm": arm, "n": n, "vs": "xg2",
                                  "same_data": same_data, **agreement(r, x)})  # fmt: skip
            if ref is not None and arm != "baseline":
                agree.append({"gt": gt, "arm": arm, "n": n, "vs": "baseline",
                              "same_data": True, **agreement(r, ref)})  # fmt: skip
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 40)
    cols = ["gt", "arm", "n", "wall_s", "warmup", "draws_per_chain", "evals_warmup",
            "evals_sampling", "min_ess_bulk", "min_ess_tail", "ess_per_min", "max_rank_rhat",
            "max_mean_spread", "divergences", "warmup_divergences", "idle_fraction",
            "stop_reason", "stop_failed"]  # fmt: skip
    print(t[cols].to_string(index=False))
    print()
    print(t[["gt", "arm", "n", "accept", "step", "depth", "warm_start_chains",
             "s_per_eval"]].to_string(index=False))  # fmt: skip
    # speed-ups relative to the baseline of the same cell
    base = t[t.arm == "baseline"].set_index(["gt", "n"])
    sp = []
    for _, r in t[t.arm != "baseline"].iterrows():
        if (r.gt, r.n) in base.index:
            b = base.loc[(r.gt, r.n)]
            sp.append({"gt": r.gt, "arm": r.arm, "n": r.n,
                       "wall_speedup": round(b.wall_s / r.wall_s, 2),
                       "evals_ratio": round(r.evals_total / b.evals_total, 2),
                       "ess_per_min_ratio": round(r.ess_per_min / b.ess_per_min, 2)})  # fmt: skip
    if sp:
        print()
        print(pd.DataFrame(sp).to_string(index=False))
    if agree:
        print()
        print(pd.DataFrame(agree).round(2).to_string(index=False))
    t.to_csv(a.root / "bench_runs.csv", index=False)
    pd.DataFrame(sp).to_csv(a.root / "bench_speedups.csv", index=False)
    pd.DataFrame(agree).to_csv(a.root / "bench_agreement.csv", index=False)


if __name__ == "__main__":
    main()
