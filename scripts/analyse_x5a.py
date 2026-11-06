"""X5a: recovery of B.to_A as the truth moves from the prior bulk to its tail.

Groups runs by prior (law.concentration override) and true B.to_A; reports B.to_A
posterior mean / sd / coverage / contraction, run-level MAE, support, PPC and SBC.

Usage: uv run python scripts/analyse_x5a.py results/x5/a_reversible_ab
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


def prior_label(overrides: str) -> str:
    conc = next((o.split("=", 1)[1] for o in overrides.splitlines() if "law.concentration=" in o), "{}")
    if conc.strip() == "{}":
        return "uniform (1)"
    if "to_B:0.5" in conc or "div_AA:0.5,div_AB:0.5" in conc:
        return "all 0.5"
    return "flat B.to_A"


def main(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        per = {r["parameter"]: r for r in m["recovery"]["per_parameter"]}
        b = per["B.to_A"]
        rows.append({
            "prior": prior_label((path.parent / ".hydra/overrides.yaml").read_text()),
            "truth_to_A": round(b["truth"], 3), "seed": m["seed"],
            "to_A_mean": b["posterior_mean"], "to_A_sd": b["posterior_std"],
            "to_A_covered": b["covered"], "to_A_contraction": b["contraction"],
            "mae": m["recovery"]["mean_abs_error"], "coverage90": m["recovery"]["coverage90"],
            "support": m["posterior_support_mass"],
            "ppc_out95": (m.get("ppc") or {}).get("fraction_outside_95", np.nan),
            "sbc_cov_err": sbc_error(m.get("calibration") or {}),
        })  # fmt: skip
    d = pd.DataFrame(rows)
    g = d.groupby(["prior", "truth_to_A"]).agg(
        n=("seed", "size"), to_A_mean=("to_A_mean", "mean"), to_A_sd=("to_A_sd", "mean"),
        to_A_cov=("to_A_covered", "mean"), to_A_contr=("to_A_contraction", "mean"),
        mae=("mae", "mean"), coverage90=("coverage90", "mean"), support=("support", "median"),
        ppc=("ppc_out95", "mean"), sbc=("sbc_cov_err", "mean"))  # fmt: skip
    d.to_csv(root / "x5a_runs.csv", index=False)
    g.round(4).to_csv(root / "x5a_groups.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(g.round(3))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
