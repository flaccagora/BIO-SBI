"""Point summaries of B.to_A per run (no likelihood evaluations): posterior mean, median,
marginal mode (KDE), MLE (mle.json), chain spread, R-hat; errors vs truth, grouped by
truth class. Writes docs/xg2_mle_vs_posterior/summaries.csv and prints the group tables.

  uv run python scripts/xg2_mle_vs_posterior/summaries.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from common import OUT, ROOT, all_runs
from scipy.stats import gaussian_kde

P = "B.to_A"
BOUNDARY = ("T1_ref", "T2_toA05", "T4_slowA")  # B divisions exactly 0


def kde_mode(x: np.ndarray) -> float:
    """Mode of a reflected (at 0 and 1) Gaussian KDE of draws in [0, 1]."""
    kde = gaussian_kde(np.r_[x, -x, 2 - x], bw_method=0.5 * len(x) ** -0.2 / np.sqrt(3))
    g = np.linspace(0, 1, 1001)
    return float(g[np.argmax(kde(g))])


def main() -> None:
    rows = []
    for (gt, seed, n), run in sorted(all_runs().items()):
        post = pd.read_csv(run / "posterior_samples.csv.gz")
        met = json.loads((run / "metrics.json").read_text())["exact_gw"]
        mj = run / "mle.json"
        mle = json.loads(mj.read_text()) if mj.exists() else None
        x = post[P].to_numpy()
        div = post[["B.div_AA", "B.div_AB", "B.div_BB"]].sum(axis=1).to_numpy()
        truth = (mle or {}).get("truth", {}).get(P)
        if truth is None:
            truth = float(np.load(run / "reference.npz")["truth"][list(post.columns).index(P)])
        cm = np.array(met["chain_means"][P])
        rows.append({
            "gt": gt, "seed": seed, "n": n, "truth": truth,
            "mean": x.mean(), "median": np.median(x), "mode_kde": kde_mode(x),
            "sd": x.std(), "q05": np.quantile(x, 0.05), "q95": np.quantile(x, 0.95),
            "mle": mle["mle"][P] if mle else np.nan,
            "mle_prior_starts": (mle or {}).get("prior_starts", 0) if mle else np.nan,
            "mle_Bdiv": sum(mle["mle"][k] for k in ("B.div_AA", "B.div_AB", "B.div_BB"))
            if mle else np.nan,
            "post_Bdiv_mean": div.mean(),
            "ll_gap_truth": mle["ll_mle"] - mle["ll_truth"] if mle else np.nan,
            "ll_gap_pm": mle["ll_mle"] - mle["ll_posterior_mean"] if mle else np.nan,
            "second_optimum_gap": (mle["distinct_optima_ll"][0] - mle["distinct_optima_ll"][1])
            if mle and len(mle["distinct_optima_ll"]) > 1 else np.nan,
            "max_rhat": met["max_rhat"], "chain_mean_spread": cm.max() - cm.min(),
            "divergences": met.get("divergences", np.nan),
        })  # fmt: skip
    df = pd.DataFrame(rows)
    df["cls"] = np.where(df["gt"].isin(BOUNDARY), "boundary (B div = 0)", "interior")
    for k in ("mean", "median", "mode_kde", "mle"):
        df[f"err_{k}"] = (df[k] - df.truth).abs()
        df[f"bias_{k}"] = df[k] - df.truth
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "summaries.csv", index=False)
    pd.set_option("display.width", 220)
    errs = ["err_mean", "err_median", "err_mode_kde", "err_mle"]
    print("mean |error| of B.to_A by truth (all n, seeds):")
    print(df.groupby(["cls", "gt"])[errs + ["bias_mean", "bias_mle"]].mean().round(3))
    print("\nby class and n:")
    print(df.groupby(["cls", "n"])[errs].mean().round(3))
    print("\nRMSE by class:")
    print(df.groupby("cls")[errs].apply(lambda g: np.sqrt((g**2).mean())).round(3))
    print("\nruns where mean beats MLE, by truth:")
    print(df.assign(win=df.err_mean < df.err_mle).groupby("gt").win.agg(["sum", "count"]))
    print("\nR-hat > 1.05 or chain spread > 0.1:")
    print(df[(df.max_rhat > 1.05) | (df.chain_mean_spread > 0.1)][
        ["gt", "seed", "n", "max_rhat", "chain_mean_spread", "mean", "mle", "truth"]].round(3))
    print("\nMLE at B.to_A < 1e-3:", int((df.mle < 1e-3).sum()), "runs:",
          df[df.mle < 1e-3]["gt"].value_counts().to_dict())
    print("\nmle.json written with 24 prior starts:", int((df.mle_prior_starts == 24).sum()),
          "of", len(df), "; root", ROOT)


if __name__ == "__main__":
    main()
