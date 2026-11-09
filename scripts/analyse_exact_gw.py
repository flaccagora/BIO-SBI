"""Exact-likelihood GW (exact_gw) vs NPE on identical datasets.

Per experiment/law: B.to_A mean, posterior sd, 90% coverage, posterior mass on B
divisions, MAE over probabilities, max R-hat. Usage: uv run python scripts/analyse_exact_gw.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
ROWS = [  # (label, method, folder glob)
    ("E0 @4096 {4,6,10}, two_type", "exact", "exact_gw/e0_4096"),
    ("E0 @4096 {4,6,10}, two_type", "NPE 20k", "e4/gw_xxl_two_type_20k"),
    ("E0 @4096 {4,6,10}, two_type", "NPE 5k", "e4/gw_xxl_two_type_5k"),
    ("E0 {4,8,12}x32, two_type", "exact", "exact_gw/e0/two_type_*"),
    ("E0 {4,8,12}x32, two_type", "NPE 5k", "x1/e0_free/npe"),
    ("E0 {4,8,12}x32, two_type", "NPE 20k", "b_row/two_type/*simulations=20000*"),
    ("E0 {4,8,12}x32, two_type", "NPE 50k", "b_row/two_type/*simulations=50000*"),
    ("E0 {4,8,12}x32, reversible_ab", "exact", "exact_gw/e0/reversible_ab"),
    ("E0 {4,8,12}x32, reversible_ab", "NPE 5k", "x1/e0_fixed/npe"),
    ("E0 {4,8,12}x32, reversible_ab", "NPE 20k", "b_row/reversible_ab/*simulations=20000*"),
    ("real-like {6,10}x{165,101}, two_type", "exact", "exact_gw/real_like/two_type_*"),
    ("real-like {6,10}x{165,101}, reversible_ab", "exact", "exact_gw/real_like/reversible_ab_*"),
    ("real-like {6,10}x{165,101}, reversible_ab", "NPE 5k", "x2/gw_data"),
    ("early {3}x266, two_type", "exact", "exact_gw/early/*horizons:[[]3[]]*"),
    ("early {2,3}x133, two_type", "exact", "exact_gw/early/*horizons:[[]2,3[]]*"),
]


def rows_for(pattern: str):
    out = []
    paths = sorted(R.glob(pattern + "/*/metrics.json"))
    if not paths:  # pattern already names run folders
        paths = sorted(R.glob(pattern + "/metrics.json"))
    for path in paths:
        m = json.loads(path.read_text())
        pp = {r["parameter"]: r for r in m["recovery"]["per_parameter"] if r["inferred"]}
        if "B.to_A" not in pp:
            continue
        b = pp["B.to_A"]
        prob = [r for k, r in pp.items() if "lifetime" not in k]
        out.append({
            "seed": m.get("seed"), "to_A": b["posterior_mean"], "to_A_sd": b["posterior_std"],
            "covered": b["covered"],
            "b_div": sum(pp[k]["posterior_mean"] for k in ("B.div_AA", "B.div_AB", "B.div_BB") if k in pp),
            "mae": np.mean([r["abs_error"] for r in prob]),
            "rhat": (m.get("exact_gw") or {}).get("max_rhat", np.nan),
            "minutes": m.get("total_seconds", np.nan) / 60,
        })  # fmt: skip
    return out


def main() -> None:
    table = []
    for label, method, pattern in ROWS:
        rs = rows_for(pattern)
        if not rs:
            continue
        d = pd.DataFrame(rs)
        table.append({"setting": label, "method": method, "runs": len(d),
                      "B.to_A (0.8)": f"{d.to_A.mean():.3f}", "post sd": f"{d.to_A_sd.mean():.3f}",
                      "covered": f"{int(d.covered.sum())}/{len(d)}",
                      "B-div mass (0)": f"{d.b_div.mean():.3f}", "MAE": f"{d.mae.mean():.3f}",
                      "max R-hat": f"{d.rhat.max():.3f}" if d.rhat.notna().any() else "–",
                      "min/run": f"{d.minutes.mean():.0f}"})  # fmt: skip
    t = pd.DataFrame(table)
    (R / "exact_gw").mkdir(exist_ok=True)
    t.to_csv(R / "exact_gw" / "exact_vs_npe.csv", index=False)
    print(t.to_markdown(index=False))


if __name__ == "__main__":
    main()
