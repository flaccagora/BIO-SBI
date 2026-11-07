"""X4: combined realistic errors at the real design (BH, budget=full).

Per arm (control, combined, modelled, combined_c): probability MAE, coverage,
contraction, lifetime error, support, PPC, SBC, and key posterior means.
Usage: uv run python scripts/analyse_x4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/x4")
KEY = ["A.div_AA", "A.div_AB", "A.div_BB", "A.death", "B.to_A", "B.death",
       "A.mean_lifetime", "B.mean_lifetime", "founder.A"]  # fmt: skip


def sbc_error(cal: dict) -> float:
    errs = [abs(v - float(level)) for level, per in (cal.get("coverage") or {}).items()
            for v in per.values() if v is not None]  # fmt: skip
    return float(np.mean(errs)) if errs else np.nan


def main() -> None:
    rows = []
    for path in sorted(ROOT.glob("*/*/metrics.json")):
        arm = path.parts[-3]
        m = json.loads(path.read_text())
        pp = {r["parameter"]: r for r in m["recovery"]["per_parameter"] if r["inferred"]}
        prob = [r for k, r in pp.items() if "lifetime" not in k and np.isfinite(r["truth"])]
        row = {"arm": arm, "seed": m["seed"],
               "mae": np.mean([r["abs_error"] for r in prob]) if prob else np.nan,
               "coverage": np.mean([r["covered"] for r in prob if not r["at_boundary"]]) if prob else np.nan,
               "rel_err_lifetimes": m["recovery"].get("mean_rel_error_scalars"),
               "contraction": m["recovery"]["mean_contraction"],
               "support": m.get("posterior_support_mass"),
               "ppc_out95": (m.get("ppc") or {}).get("fraction_outside_95", np.nan),
               "sbc": sbc_error(m.get("calibration") or {})}  # fmt: skip
        for k in KEY:
            if k in pp:
                row[k] = pp[k]["posterior_mean"]
                row[k + "_cov"] = pp[k]["covered"]
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "x4_runs.csv", index=False)
    g = d.groupby("arm").mean(numeric_only=True).drop(columns="seed")
    g.round(4).to_csv(ROOT / "x4_groups.csv")
    pd.set_option("display.width", 250)
    print(g.round(3).T.to_string())


if __name__ == "__main__":
    main()
