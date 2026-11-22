"""All transition probabilities against n = 128-4096 (report_exact_gw, Section 7.1).

Usage (from the repository root)::

    uv run python report_exact_gw/make_row_large_n.py [--results /path/to/results]

Reads ``results/exact_gw/data_size/xg2_runs.csv`` (flat prior) and ``xg2_mle.csv`` and writes
``report_exact_gw/figures/row_B_large_n.pdf`` and ``row_A_large_n.pdf``: rows = the six
probabilities of the row, columns = truths; posterior mean +- posterior sd (mean over the
datasets) in blue, MLE (mean over datasets) in orange, truth dashed. Also
``report_exact_gw/tables/rows_large_n.csv``: RMSE of the posterior mean and of the MLE and
the mean z per (truth, parameter, n).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG, TAB = ROOT / "report_exact_gw" / "figures", ROOT / "report_exact_gw" / "tables"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
TRUTHS = ["T1_ref", "T2_toA05", "T3_Bdiv", "T4_slowA", "T5_C7", "P1", "P2", "P3", "P4"]
LABEL = {"T1_ref": "T1", "T2_toA05": "T2", "T3_Bdiv": "T3", "T4_slowA": "T4", "T5_C7": "T5"}
OUT = {"death": "dies", "persist": "persists", "to_A": "→A", "to_B": "→B",
       "div_AA": "→AA", "div_AB": "→AB", "div_BB": "→BB"}  # fmt: skip
ORDER = ["death", "persist", "to_A", "to_B", "div_AA", "div_AB", "div_BB"]


def main(results: Path) -> None:
    xg2 = results / "exact_gw" / "data_size"
    d = pd.read_csv(xg2 / "xg2_runs.csv")
    d = d[d["prior"] == "flat"]
    m = pd.read_csv(xg2 / "xg2_mle.csv")
    m = m.rename(columns={"mle": "mle_value"})[["gt", "seed", "n", "parameter", "mle_value"]]
    d = d.merge(m, on=["gt", "seed", "n", "parameter"], how="left")
    d["err"] = d["mean"] - d["truth"]
    d["err_mle"] = d["mle_value"] - d["truth"]
    tab = (d.groupby(["gt", "parameter", "n"])
           .agg(truth=("truth", "first"), datasets=("seed", "nunique"), post_mean=("mean", "mean"),
                post_sd=("sd", "mean"), mle=("mle_value", "mean"),
                rmse_post=("err", lambda e: float(np.sqrt((e**2).mean()))),
                rmse_mle=("err_mle", lambda e: float(np.sqrt((e**2).mean()))),
                mean_z=("z", "mean"))
           .reset_index())  # fmt: skip
    TAB.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TAB / "rows_large_n.csv", index=False)
    truths = [t for t in TRUTHS if t in set(tab["gt"])]
    ns = sorted(tab["n"].unique())
    for row in ("B", "A"):
        params = sorted({p for p in tab["parameter"] if p.startswith(row + ".")},
                        key=lambda p: ORDER.index(p.split(".")[1]))  # fmt: skip
        fig, axes = plt.subplots(len(params), len(truths), figsize=(1.55 * len(truths) + 0.8, 8.6),
                                 sharex=True, squeeze=False)  # fmt: skip
        for j, gt in enumerate(truths):
            for i, p in enumerate(params):
                ax = axes[i, j]
                g = tab[(tab["gt"] == gt) & (tab["parameter"] == p)].sort_values("n")
                x = np.log2(g["n"])
                ax.fill_between(x, g["post_mean"] - g["post_sd"], g["post_mean"] + g["post_sd"],
                                color=BLUE, alpha=0.18, lw=0)  # fmt: skip
                ax.plot(x, g["post_mean"], "-o", color=BLUE, ms=2.5, lw=1.2)
                ax.plot(x, g["mle"], "D", color=ORANGE, ms=2.5)
                ax.axhline(g["truth"].iloc[0], color="k", ls="--", lw=0.9)
                lo = min((g["post_mean"] - g["post_sd"]).min(), g["mle"].min(), g["truth"].iloc[0])
                hi = max((g["post_mean"] + g["post_sd"]).max(), g["mle"].max(), g["truth"].iloc[0])
                pad = max(0.03, 0.1 * (hi - lo))
                ax.set_ylim(max(-0.02, lo - pad), min(1.02, hi + pad))
                ax.set_xticks(np.log2(ns))
                ax.set_xticklabels([str(n) if n in (128, 1024, 4096) else "" for n in ns],
                                   fontsize=6.5)  # fmt: skip
                ax.tick_params(axis="y", labelsize=6.5)
                ax.grid(alpha=0.3, lw=0.4)
                if i == 0:
                    ax.set_title(LABEL.get(gt, gt), fontsize=9)
                if j == 0:
                    ax.set_ylabel(row + " " + OUT[p.split(".")[1]], fontsize=8.5)
                if i == len(params) - 1:
                    ax.set_xlabel("n", fontsize=7.5)
        fig.tight_layout(h_pad=0.3, w_pad=0.3)
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / f"row_{row}_large_n.pdf")
        fig.savefig(FIG / f"row_{row}_large_n.png", dpi=110)
        plt.close(fig)
        print("wrote", FIG / f"row_{row}_large_n.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    main(ap.parse_args().results)
