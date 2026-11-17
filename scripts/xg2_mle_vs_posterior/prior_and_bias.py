"""Figures/tables from existing results (no likelihood evaluations):

* prior_alpha.png / prior_alpha.csv: posterior mean of B.to_A under symmetric Dirichlet(a),
  a in {0.5, 1, 2} (PSIS-reweighted flat-prior draws, reliable rows only; sparse 0.2 and
  central 5 are mostly unreliable), per truth, mean over seeds and n of the change vs flat.
  Also the analytic prediction of the "marginal prior" part: Beta(a, 5a) log density.
* bias_vs_truth.png: bias of posterior mean and MLE of B.to_A against the true value,
  n = 128 and 1024 (summaries.csv).

  uv run python scripts/xg2_mle_vs_posterior/prior_and_bias.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import OUT, ROOT  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ALPHA = {"sparse": 0.2, "jeffreys": 0.5, "flat": 1.0, "smooth": 2.0, "central": 5.0}


def main() -> None:
    d = pd.read_csv(ROOT / "xg2_runs.csv")
    d = d[(d.parameter == "B.to_A") & d.prior.isin(list(ALPHA)) & d.reliable]
    d = d.assign(alpha=d.prior.map(ALPHA), shift=d["mean"] - d.flat_mean)
    tab = d.groupby(["gt", "alpha"]).agg(shift=("shift", "mean"), runs=("shift", "size"))
    tab = tab.unstack("alpha")
    tab.to_csv(OUT / "prior_alpha.csv")
    pd.set_option("display.width", 200)
    print("mean change of the B.to_A posterior mean vs flat, by Dirichlet alpha (runs):")
    print(tab.round(3).to_string())
    # paired: same runs reliable under jeffreys, flat and smooth
    piv = d.pivot_table(index=["gt", "seed", "n"], columns="alpha", values="mean")
    both = piv[[0.5, 1.0, 2.0]].dropna()
    mono = ((both[0.5] >= both[1.0]) & (both[1.0] >= both[2.0])).groupby("gt").agg(["sum", "size"])
    print("\nruns with mean(a=0.5) >= mean(a=1) >= mean(a=2):")
    print(mono.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.8))
    grp = both.reset_index()
    grp["grp"] = grp["gt"].where(~grp["gt"].str.match(r"P[0-9]"), "P1-P4")
    order = ["T1_ref", "T4_slowA", "T3_Bdiv", "T2_toA05", "P1-P4", "T5_C7"]
    for k, gname in enumerate(order):
        g = grp[grp.grp == gname]
        m = g[[0.5, 1.0, 2.0]].mean()
        ax[0].plot([0.5, 1, 2], m.values, "o-", color=C[k], label=f"{gname} ({len(g)} runs)")
    ax[0].set_xscale("log")
    ax[0].minorticks_off()
    ax[0].set_xticks([0.5, 1, 2], ["0.5", "1 (flat)", "2"])
    ax[0].set_xlabel("Dirichlet concentration a (every entry)")
    ax[0].set_ylabel("posterior mean of B.to_A")
    ax[0].set_title("posterior mean vs prior concentration\n(PSIS, runs reliable at all 3 a)")
    ax[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    s = pd.read_csv(OUT / "summaries.csv")
    for k, n in enumerate((128, 1024)):
        g = s[s.n == n]
        ax[1].scatter(g.truth + 0.006 * (k - 0.5), g["mean"] - g.truth, color=C[0], marker="ox"[k],
                      label=f"posterior mean, n={n}")  # fmt: skip
        ax[1].scatter(g.truth + 0.006 * (k - 0.5), g.mle - g.truth, color=C[1], marker="ox"[k],
                      label=f"MLE, n={n}")  # fmt: skip
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].axvline(1 / 6, color="0.5", ls=":", lw=1)
    ax[1].text(1 / 6 + 0.01, 0.25, "prior mean 1/6", fontsize=7, color="0.3")
    ax[1].set_xlabel("true B.to_A")
    ax[1].set_ylabel("estimate - truth")
    ax[1].set_title("error of B.to_A by true value (3 datasets each)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "prior_alpha_and_bias.png", dpi=110)

    # analytic: marginal prior log density of B.to_A under Dir(a) (6 outcomes): Beta(a, 5a)
    c = np.array([0.2, 0.5, 0.8])
    for a in (0.5, 1, 2):
        lp = (a - 1) * np.log(c) + (5 * a - 1) * np.log1p(-c)
        print(f"a={a}: log Beta(a,5a) density at c=0.2/0.5/0.8 rel. to 0.2:",
              np.round(lp - lp[0], 2))


if __name__ == "__main__":
    main()
