"""X1: inference methods (NPE, SNPE, NLE, NRE) on the decisive settings.

Per setting x method: probability MAE, 90% coverage, contraction, support mass (NPE
family), PPC, SBC coverage error, run time, and a few key parameters.
Usage: uv run python scripts/analyse_x1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/x1")
KEY = {"e0_free": "B.to_A", "e0_fixed": "B.to_A", "bh_ridge": "A.div_AA",
       "silent_short_lifetime": "A.div_AA", "loud_wrong_lifetime_prior": "A.div_AA"}  # fmt: skip


def sbc_error(cal: dict) -> float:
    errs = [abs(v - float(level)) for level, per in (cal.get("coverage") or {}).items()
            for v in per.values() if v is not None]  # fmt: skip
    return float(np.mean(errs)) if errs else np.nan


def main() -> None:
    rows = []
    for path in sorted(ROOT.glob("*/*/*/metrics.json")):
        setting, method = path.parts[-4], path.parts[-3]
        m = json.loads(path.read_text())
        pp = [r for r in m["recovery"]["per_parameter"] if r["inferred"]]
        prob = [r for r in pp if "lifetime" not in r["parameter"]]
        key = next((r for r in pp if r["parameter"] == KEY[setting]), None)
        rows.append({
            "setting": setting, "method": method, "seed": m["seed"],
            "mae": np.mean([r["abs_error"] for r in prob]),
            "coverage": np.mean([r["covered"] for r in pp if not r["at_boundary"]]),
            "contraction": np.mean([r["contraction"] for r in pp]),
            "support": m.get("posterior_support_mass"),
            "ppc_out95": (m.get("ppc") or {}).get("fraction_outside_95", np.nan),
            "sbc": sbc_error(m.get("calibration") or {}),
            "key_param": KEY[setting], "key_mean": key["posterior_mean"] if key else np.nan,
            "key_covered": key["covered"] if key else np.nan,
            "minutes": m.get("total_seconds", np.nan) / 60,
        })  # fmt: skip
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "x1_runs.csv", index=False)
    g = d.groupby(["setting", "method"]).agg(
        n=("seed", "size"), mae=("mae", "mean"), coverage=("coverage", "mean"),
        contraction=("contraction", "mean"), support=("support", "mean"),
        ppc=("ppc_out95", "mean"), sbc=("sbc", "mean"), key=("key_param", "first"),
        key_mean=("key_mean", "mean"), key_cov=("key_covered", "sum"),
        minutes=("minutes", "mean"))  # fmt: skip
    g.round(4).to_csv(ROOT / "x1_groups.csv")
    pd.set_option("display.width", 250)
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
