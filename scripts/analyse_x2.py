"""X2: dataset spread vs training spread of the posterior.

For each simulator, compares across 10 runs:
* training arm (same dataset, training seeds 0-9): sd of posterior means = training +
  sampling noise;
* data arm (datasets 0-9, training seed 0): sd of posterior means = dataset noise +
  training noise.
Dataset share of variance ~ 1 - var_training / var_data.

Usage: uv run python scripts/analyse_x2.py [results/x2]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load(folder: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        for r in m["recovery"]["per_parameter"]:
            if r["inferred"]:
                rows.append({"seed": m["seed"], "data_seed": m.get("data_seed"),
                             "parameter": r["parameter"], "truth": r["truth"],
                             "mean": r["posterior_mean"], "sd": r["posterior_std"],
                             "covered": r["covered"]})  # fmt: skip
    return pd.DataFrame(rows)


def main(root: Path) -> None:
    out = []
    for sim in ("gw", "bh"):
        tr, da = root / f"{sim}_training", root / f"{sim}_data"
        if not (tr.exists() and da.exists()):
            continue
        t, d = load(tr), load(da)
        if t.empty or d.empty:
            continue
        g = pd.DataFrame({
            "truth": t.groupby("parameter", sort=False).truth.first(),
            "post_sd": d.groupby("parameter", sort=False)["sd"].mean(),
            "sd_training": t.groupby("parameter", sort=False)["mean"].std(),
            "sd_data": d.groupby("parameter", sort=False)["mean"].std(),
            "mean_training": t.groupby("parameter", sort=False)["mean"].mean(),
            "mean_data": d.groupby("parameter", sort=False)["mean"].mean(),
            "cov_training": t.groupby("parameter", sort=False).covered.mean(),
            "cov_data": d.groupby("parameter", sort=False).covered.mean(),
        })  # fmt: skip
        g["training_share"] = (g.sd_training**2 / g.sd_data**2).clip(upper=1)
        g["training_sd_over_post_sd"] = g.sd_training / g.post_sd
        g.insert(0, "simulator", sim)
        out.append(g.reset_index())
    res = pd.concat(out, ignore_index=True)
    res.to_csv(root / "x2_spread.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/x2"))
