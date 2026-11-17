"""Merge summaries.csv with low_branch.csv: how often the stored MLE is not the global one,
B.to_A errors with the corrected (global) MLE, and the "B never divides" sub-model
(likelihood-ratio test, and B.to_A under the sub-model when it is not rejected).

  uv run python scripts/xg2_mle_vs_posterior/mle_global.py
Writes docs/xg2_mle_vs_posterior/mle_global.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import OUT
from scipy.stats import chi2

# chi-bar-squared(3) critical value at 5%, weights (1, 3, 3, 1) / 8 (orthogonal approx.)
def chibar_sf(x: float) -> float:
    return float(sum(w * chi2.sf(x, k) for k, w in ((1, 3 / 8), (2, 3 / 8), (3, 1 / 8)))) if x > 0 else 1.0


s = pd.read_csv(OUT / "summaries.csv")
lb = pd.read_csv(OUT / "low_branch.csv")
d = lb.merge(s[["gt", "seed", "n", "mean", "median", "mode_kde"]], on=["gt", "seed", "n"])
d["stored_better_found"] = d.low_minus_stored > 0.05
d["p_nodiv"] = d.LR_nodiv.map(chibar_sf)
d["nodiv_kept"] = d.p_nodiv > 0.05
# model-selected estimate: sub-model when "B never divides" is not rejected, else global MLE
d["selected"] = np.where(d.nodiv_kept, d.nodiv_BtoA, d.global_BtoA)
for k in ("stored_BtoA", "global_BtoA", "mean", "median", "nodiv_BtoA", "selected"):
    d[f"err_{k}"] = (d[k] - d.truth).abs()
d.to_csv(OUT / "mle_global.csv", index=False)
pd.set_option("display.width", 220)
print("runs where a better optimum exists on the low-B.to_A branch (> 0.05 nats):")
print(d.groupby("gt").agg(runs=("run", "size"), better=("stored_better_found", "sum"),
                          max_gain=("low_minus_stored", "max")))  # fmt: skip
print(d[d.stored_better_found][["run", "truth", "stored_BtoA", "global_BtoA", "low_minus_stored",
                                "mean"]].round(3).to_string())  # fmt: skip
print("\nmean |error| of B.to_A:")
cols = [c for c in d if c.startswith("err_")]
print(d.groupby("gt")[cols].mean().round(3).to_string())
print(d.groupby(["gt", "n"])[cols].mean().round(3).to_string())
print("\n'B never divides' kept (p > 0.05) by truth:")
print(d.groupby("gt").nodiv_kept.agg(["sum", "size"]).to_string())
print(d.groupby(["gt", "n"]).nodiv_kept.agg(["sum", "size"]).unstack("n").to_string())
