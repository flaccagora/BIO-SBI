"""X5b (lifetime prior arms) and X5c (data vs prior) tables.

Usage: uv run python scripts/analyse_x5bc.py [results/x5]
Writes x5b_groups.csv (per arm x parameter) and c_data_vs_prior/x5c_{runs,groups}.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sbc_error(cal: dict) -> float:
    errs = [abs(v - float(level)) for level, per in (cal.get("coverage") or {}).items()
            for v in per.values() if v is not None and np.isfinite(v)]  # fmt: skip
    return float(np.mean(errs)) if errs else np.nan


def x5b(root: Path) -> pd.DataFrame:
    rows = []
    for arm in ("b_vague", "b_right", "b_wrong"):
        for path in sorted((root / arm).glob("*/metrics.json")):
            m = json.loads(path.read_text())
            for r in m["recovery"]["per_parameter"]:
                if r["inferred"]:
                    rows.append({"arm": arm, "seed": m["seed"], **r,
                                 "support": m["posterior_support_mass"],
                                 "ppc_out95": m["ppc"]["fraction_outside_95"],
                                 "sbc": sbc_error(m["calibration"]),
                                 "mae_prob": m["recovery"]["mean_abs_error"]})  # fmt: skip
    d = pd.DataFrame(rows)
    return d.groupby(["arm", "parameter"], sort=False).agg(
        truth=("truth", "first"), mean=("posterior_mean", "mean"), sd=("posterior_std", "mean"),
        covered=("covered", "sum"), contraction=("contraction", "mean"),
        support=("support", "mean"), ppc=("ppc_out95", "mean"), sbc=("sbc", "mean"),
        mae_prob=("mae_prob", "mean"))  # fmt: skip


def x5c(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((root / "c_data_vs_prior").glob("*/metrics.json")):
        m = json.loads(path.read_text())
        ov = (path.parent / ".hydra/overrides.yaml").read_text()
        per = {r["parameter"]: r for r in m["recovery"]["per_parameter"]}
        b = per["B.to_A"]
        clones = next(o.split("=")[1] for o in ov.split() if o.startswith("design.clones="))
        rows.append({"prior": "flat B.to_A" if "death:0.2" in ov else "uniform",
                     "clones": int(clones), "seed": m["seed"],
                     "to_A": b["posterior_mean"], "to_A_sd": b["posterior_std"],
                     "to_A_covered": b["covered"], "to_A_contraction": b["contraction"],
                     "B_div_AA": per["B.div_AA"]["posterior_mean"],
                     "mae": m["recovery"]["mean_abs_error"],
                     "support": m["posterior_support_mass"],
                     "ppc_out95": m["ppc"]["fraction_outside_95"],
                     "sbc": sbc_error(m["calibration"])})  # fmt: skip
    d = pd.DataFrame(rows)
    d.to_csv(root / "c_data_vs_prior" / "x5c_runs.csv", index=False)
    return d.groupby(["prior", "clones"]).mean(numeric_only=True).drop(columns="seed")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/x5")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    b = x5b(root)
    b.round(4).to_csv(root / "x5b_groups.csv")
    c = x5c(root)
    c.round(4).to_csv(root / "c_data_vs_prior" / "x5c_groups.csv")
    print(b.round(3), c.round(3), sep="\n\n")
