"""Difficulty ranking of the 12 transition probabilities (report_exact_gw, Section 5.4).

Usage (from the repository root)::

    uv run python report_exact_gw/make_param_ranking.py [--results /path/to/results]

Reads ``results/exact_gw/data_size/xg2_runs.csv`` and ``xg2_mle.csv`` (flat prior, XG2,
n = 128-1024, 9 truths x 3 datasets) and writes ``report_exact_gw/tables/param_rank.{tex,csv}``.

Seven measures, each ranked over the 12 parameters (12 = hardest):
RMSE of the posterior mean at n = 128 and 1024; posterior sd / prior sd at 1024 (geometric
mean); 90% coverage over truth > 0 (lower = harder); mean |z| at 1024; RMSE of the MLE at 1024;
prior sensitivity at 1024 (median over datasets of the largest reliable PSIS shift of the
posterior mean under the symmetric/sparse/central priors, in flat-posterior sd).
The order is by the mean rank. Every parameter has the same flat-prior marginal Beta(1, 5)
(sd 0.141), so absolute errors are comparable across parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "report_exact_gw" / "tables"
PRETTY = {
    "A.death": "A dies", "A.persist": "A persists", "A.to_B": r"A$\to$B",
    "A.div_AA": r"A$\to$AA", "A.div_AB": r"A$\to$AB", "A.div_BB": r"A$\to$BB",
    "B.death": "B dies", "B.persist": "B persists", "B.to_A": r"\BA",
    "B.div_AA": r"B$\to$AA", "B.div_AB": r"B$\to$AB", "B.div_BB": r"B$\to$BB",
}  # fmt: skip
TRUTH = {"T1_ref": "T1", "T2_toA05": "T2", "T3_Bdiv": "T3", "T4_slowA": "T4", "T5_C7": "T5"}
SYMMETRIC = ["sparse", "jeffreys", "smooth", "central"]


def rmse(e: pd.Series) -> float:
    return float(np.sqrt((e**2).mean()))


def main(results: Path) -> pd.DataFrame:
    xg2 = results / "exact_gw" / "data_size"
    d = pd.read_csv(xg2 / "xg2_runs.csv")
    d = d[d["n"] <= 1024]
    f = d[d["prior"] == "flat"].copy()
    f["err"] = f["mean"] - f["truth"]
    f["rel_sd"] = f["sd"] / np.sqrt(f["prior_var"])
    m = pd.read_csv(xg2 / "xg2_mle.csv")
    m = m[m["n"] == 1024]
    rows = []
    for p, g in f.groupby("parameter"):
        g1024 = g[g["n"] == 1024]
        pos = g[g["truth"] > 0]
        sh = d[(d["parameter"] == p) & (d["n"] == 1024) & d["prior"].isin(SYMMETRIC) & d["reliable"]]
        shift = sh.groupby(["gt", "seed"])["shift_sd"].apply(lambda s: s.abs().max()).median()
        per_truth = g1024.groupby("gt")["err"].apply(rmse)
        rows.append({
            "parameter": p,
            "rmse128": rmse(g[g["n"] == 128]["err"]),
            "rmse1024": rmse(g1024["err"]),
            "relsd1024": float(np.exp(np.log(g1024["rel_sd"]).mean())),
            "coverage": float(pos["covered"].mean()),
            "absz1024": float(g1024["z"].abs().mean()),
            "mle_rmse1024": rmse(m[m["parameter"] == p]["abs_err_mle"]),
            "prior_shift1024": float(shift),
            "hardest_truth": per_truth.idxmax(),
            "hardest_rmse1024": float(per_truth.max()),
        })  # fmt: skip
    t = pd.DataFrame(rows).set_index("parameter")
    ranks = pd.DataFrame({
        "rmse128": t["rmse128"].rank(), "rmse1024": t["rmse1024"].rank(),
        "relsd1024": t["relsd1024"].rank(), "coverage": (-t["coverage"]).rank(),
        "absz1024": t["absz1024"].rank(), "mle_rmse1024": t["mle_rmse1024"].rank(),
        "prior_shift1024": t["prior_shift1024"].rank(),
    })  # fmt: skip
    t["mean_rank"] = ranks.mean(axis=1)
    t["n_hardest"] = (ranks == 12).sum(axis=1)
    t = t.sort_values("mean_rank", ascending=False).reset_index()
    TAB.mkdir(parents=True, exist_ok=True)
    t.to_csv(TAB / "param_rank.csv", index=False)
    lines = []
    for i, r in t.iterrows():
        cells = [
            str(i + 1), PRETTY[r["parameter"]], f"{r['rmse128']:.3f}", f"{r['rmse1024']:.3f}",
            f"{r['relsd1024']:.2f}", f"{100 * r['coverage']:.0f}\\%", f"{r['absz1024']:.2f}",
            f"{r['prior_shift1024']:.2f}", f"{r['mle_rmse1024']:.3f}",
            f"{TRUTH.get(r['hardest_truth'], r['hardest_truth'])} ({r['hardest_rmse1024']:.2f})",
            f"{r['mean_rank']:.1f}",
        ]  # fmt: skip
        lines.append(" & ".join(cells) + r" \\")
    (TAB / "param_rank.tex").write_text("\n".join(lines) + "\n")
    print(t.round(3).to_string())
    print(ranks.loc[t["parameter"]].to_string())
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    main(ap.parse_args().results)
