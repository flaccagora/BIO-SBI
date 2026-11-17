"""Tables of the exact_gw reparametrisation benchmark (results/exact_gw/reparam_bench).

    uv run python scripts/exact_gw_reparam/analyse.py [--out DIR]

Per run: cost (gradient / likelihood-only evaluations, wall time, CPU seconds inside the
likelihood), ESS (bulk / tail, min over sticks and free probabilities) per 1000 gradient
evaluations and per likelihood CPU minute, max rank R-hat, chain-mean spread,
divergences, step sizes, tree depths; T1: share of draws in the "B divides" branch
(``B.to_A < 0.45``) per chain, pooled, and in the SMC population; agreement of posterior
means with a reference (z in Monte-Carlo standard errors) and sd ratios.
Writes ``runs.csv``, ``agreement.csv``, ``branches.csv`` and prints them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/home/matteo/BIO-SBI/results/exact_gw/reparam_bench")
XG2 = Path("/home/matteo/BIO-SBI/results/exact_gw/data_size")
XG2_DIR = {"T1_n1024_s0": "T1_ref/n1024_s0", "P3_n1024_s0": "P3/n1024_s0",
           "T3_n512_s1": "T3_Bdiv/n512_s1"}  # fmt: skip
T5_BENCH = Path("/home/matteo/BIO-SBI/results/exact_gw/sampler_bench/T5_C7/baseline/n1024_s0")
BRANCH = 0.45  # B.to_A threshold between the "B divides" and "switches back" branches


def load_run(d: Path) -> dict | None:
    """Diagnostics and draws of one run directory."""
    if not (d / "diagnostics.json").exists():
        return None
    diag = json.loads((d / "diagnostics.json").read_text())
    z = np.load(d / "draws.npz")
    names = [str(n) for n in z["names"]]
    return {
        "diag": diag,
        "phys": z["physical"],
        "names": names,
        "pop": z["smc_population_physical"] if "smc_population_physical" in z.files else None,
        "dir": d,
    }


def reference(cell: str, runs: dict) -> tuple[pd.DataFrame | None, str]:
    """Return reference draws and ESS per name: XG2 run (T5: the sampler-bench baseline)."""
    src = XG2 / XG2_DIR[cell] if cell in XG2_DIR else (T5_BENCH if cell.startswith("T5") else None)
    if src is None or not (src / "posterior_samples.csv.gz").exists():
        return None, ""
    frame = pd.read_csv(src / "posterior_samples.csv.gz")
    m = json.loads((src / "metrics.json").read_text())["exact_gw"]
    ess = m.get("ess_bulk") or m.get("ess")
    return (frame, ess), str(src)


def summarise(cell: str, arm: str, r: dict) -> dict:
    """Return one row of the runs table."""
    d = r["diag"]
    names = r["names"]
    free = [n for n in names if n in d.get("chain_means", {})]
    grad = int(d.get("gradient_evals", 0))
    lik_only = int(d.get("likelihood_evals_smc", 0) or 0)
    cpu = float(d.get("likelihood_eval_seconds", float("nan")))
    ess_b, ess_t = d.get("min_ess_bulk"), d.get("min_ess_tail")
    phys = r["phys"]  # (chains, n, names)
    row = {
        "cell": cell,
        "arm": arm,
        "sampler": d.get("sampler"),
        "chains": d.get("chains"),
        "draws_per_chain": d.get("draws_per_chain"),
        "warmup": d.get("warmup"),
        "grad_evals": grad,
        "lik_only_evals": lik_only,
        "wall_s": round(d.get("wall_seconds", d.get("seconds", np.nan))),
        "lik_cpu_s": round(cpu),
        "ess_bulk": ess_b,
        "ess_tail": ess_t,
        "bulk_per_1k_grad": 1000 * ess_b / grad if grad else np.nan,
        "tail_per_1k_grad": 1000 * ess_t / grad if grad else np.nan,
        "bulk_per_cpu_min": 60 * ess_b / cpu if cpu else np.nan,
        "max_rank_rhat": d.get("max_rank_rhat"),
        "max_rhat_at": None,
        "mean_spread": d.get("max_mean_spread"),
        "divergences": d.get("divergences"),
        "warmup_div": d.get("warmup_divergences"),
        "step": "/".join(f"{s:.3f}" for s in d.get("step_size", [])),
        "depth": "/".join(f"{x:.1f}" for x in d.get("mean_tree_depth", [])),
        "stop": (d.get("stop") or {}).get("reason"),
        "failed": ",".join((d.get("stop") or {}).get("failed", [])),
    }
    rr = d.get("rank_rhat") or {}
    if rr:
        row["max_rhat_at"] = max(rr, key=lambda k: rr[k] if np.isfinite(rr[k]) else -1)
    if "B.to_A" in names:
        j = names.index("B.to_A")
        low = phys[..., j] < BRANCH
        row["low_branch_pooled"] = float(low.mean())
        row["low_branch_chains"] = "/".join(f"{x:.2f}" for x in low.mean(1))
        if r["pop"] is not None:
            row["low_branch_smc_pop"] = smc_low_share(r)
    _ = free
    return row


def main() -> None:
    """Print and write the tables."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    rows, agree = [], []
    for cell_dir in sorted(x for x in a.out.iterdir() if x.is_dir()):
        cell = cell_dir.name
        runs = {d.name: load_run(d) for d in sorted(cell_dir.iterdir()) if d.is_dir()}
        runs = {k: v for k, v in runs.items() if v is not None}
        ref, ref_src = reference(cell, runs)
        for arm, r in runs.items():
            row = summarise(cell, arm, r)
            rows.append(row)
            names = r["names"]
            free = list(r["diag"]["chain_means"])
            x = r["phys"].reshape(-1, len(names))
            ess_run = r["diag"].get("ess_bulk") or {}
            comps = []
            if ref is not None:
                comps.append(("xg2", ref[0][free].to_numpy(), ref[1]))
            if arm != "baseline" and "baseline" in runs:
                b = runs["baseline"]
                comps.append(("baseline", b["phys"].reshape(-1, len(names))[:, [names.index(n)
                              for n in free]], b["diag"].get("ess_bulk") or {}))  # fmt: skip
            for vs, y, ess_ref in comps:
                xi = x[:, [names.index(n) for n in free]]
                e1 = np.array([ess_run.get(n, len(xi)) for n in free], float)
                e0 = np.array([ess_ref.get(n, len(y)) for n in free], float)
                z = (xi.mean(0) - y.mean(0)) / np.sqrt(xi.var(0) / e1 + y.var(0) / e0)
                ratio = xi.std(0) / y.std(0)
                agree.append({"cell": cell, "arm": arm, "vs": vs, "max_abs_z": np.abs(z).max(),
                              "at": free[int(np.abs(z).argmax())],
                              "median_abs_z": np.median(np.abs(z)),
                              "sd_ratio_min": ratio.min(), "sd_ratio_max": ratio.max()})  # fmt: skip
    df = pd.DataFrame(rows)
    ag = pd.DataFrame(agree)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 40)
    cols = [
        "cell",
        "arm",
        "grad_evals",
        "lik_only_evals",
        "wall_s",
        "lik_cpu_s",
        "ess_bulk",
        "ess_tail",
        "bulk_per_1k_grad",
        "tail_per_1k_grad",
        "bulk_per_cpu_min",
        "max_rank_rhat",
        "max_rhat_at",
        "mean_spread",
        "divergences",
        "stop",
        "failed",
    ]
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))
    print()
    print(df[["cell", "arm", "step", "depth", "warmup", "draws_per_chain", "warmup_div"]]
          .to_string(index=False))  # fmt: skip
    if "low_branch_pooled" in df:
        print()
        br = df.dropna(subset=["low_branch_pooled"])
        bcols = ["cell", "arm", "low_branch_pooled", "low_branch_chains", "low_branch_smc_pop"]
        print(br[[c for c in bcols if c in br]].round(3).to_string(index=False))
        br[[c for c in bcols if c in br]].to_csv(a.out / "branches.csv", index=False)
    print()
    if len(ag):
        print(ag.round(3).to_string(index=False))
        ag.to_csv(a.out / "agreement.csv", index=False)
    df.to_csv(a.out / "runs.csv", index=False)


def smc_low_share(r: dict) -> float:
    """Share of the SMC population with B.to_A below :data:`BRANCH`."""
    return float((r["pop"][:, r["names"].index("B.to_A")] < BRANCH).mean())


if __name__ == "__main__":
    main()
