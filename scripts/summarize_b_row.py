"""Summarise the ``b_row_identifiability`` sweeps (see conf/experiment/).

For each law and simulation budget, averages over seeds: recovery of B.to_A and
B.div_AA, the posterior correlation between them, overall recovery, SBC calibration
and the posterior-predictive check.

Usage::

    uv run python scripts/summarize_b_row.py results/b_row
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_row(run: Path) -> dict:
    """Extract the quantities of interest from one finished run."""
    m = json.loads((run / "metrics.json").read_text())
    rec = {r["parameter"]: r for r in m["recovery"]["per_parameter"]}
    samples = pd.read_csv(run / "posterior_samples.csv.gz")
    cal = m.get("calibration") if isinstance(m.get("calibration"), dict) else None  # SNPE: none
    row = {
        "law": m["law"],
        "simulations": m["simulations"],
        "rounds": m["rounds"],
        "design": m["design"].get("name", "default"),
        "total_clones": sum(m["design"]["clones"]),
        "free": m["n_free_parameters"],
        "valid_sims": sum(h["valid_simulations"] for h in m["training"]),
        "epochs": m["training"][-1]["epochs"],
        "mean_abs_error": m["recovery"]["mean_abs_error"],
        "mean_contraction": m["recovery"]["mean_contraction"],
        "coverage90": m["recovery"]["coverage90"],
        "ppc_outside95": m["ppc"]["fraction_outside_95"],
        "sbc_cov_err": cal["mean_abs_coverage_error"] if cal else np.nan,
        "sbc_min_ks_p": min(cal["ks_pvalue"].values()) if cal else np.nan,
    }
    for name in ("B.to_A", "B.div_AA", "B.death", "B.persist"):
        r = rec[name]
        short = name.split(".")[1]
        row[f"{short}_mean"] = r["posterior_mean"]
        row[f"{short}_std"] = r["posterior_std"]
        row[f"{short}_err"] = r["abs_error"]
        row[f"{short}_contraction"] = r["contraction"]
    if samples["B.div_AA"].std() > 0:
        row["corr_toA_divAA"] = float(np.corrcoef(samples["B.to_A"], samples["B.div_AA"])[0, 1])
    else:
        row["corr_toA_divAA"] = np.nan
    return row


def main() -> None:
    """Print per-run and aggregated tables; write them next to the sweeps."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default="results/b_row")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        row = run_row(path.parent)
        row["sweep"] = path.parent.parent.name  # e.g. two_type, two_type_flat_to_A
        row["seed"] = int(path.parent.name.split("seed=")[1].split(",")[0])
        rows.append(row)
    runs = pd.DataFrame(rows)
    runs.to_csv(root / "b_row_runs.csv", index=False)
    cols = ["to_A_mean", "to_A_std", "to_A_err", "div_AA_mean", "div_AA_std",
            "corr_toA_divAA", "to_A_contraction", "mean_abs_error", "mean_contraction",
            "coverage90", "sbc_cov_err", "sbc_min_ks_p", "ppc_outside95"]  # fmt: skip
    agg = runs.groupby(["sweep", "rounds", "simulations"])[cols].agg(["mean", "std"]).round(3)
    agg.to_csv(root / "b_row_summary.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(runs.groupby(["sweep", "rounds", "simulations"])[cols].mean().round(3).to_string())
    print("\nper seed, B.to_A posterior mean (truth 0.80):")
    print(runs.pivot_table(index=["sweep", "rounds", "simulations"], columns="seed", values="to_A_mean")
          .round(2).to_string())  # fmt: skip


if __name__ == "__main__":
    main()
