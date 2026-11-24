"""Figures of the founder-distribution study (Fisher part; the NUTS figure is in analyse_sim.py).

  uv run python scripts/founder_distribution/figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fd_common as F  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]  # fmt: skip
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#a83232", "#e34948", "#f2a19b", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"])  # fmt: skip
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
              "#0d366b"])  # fmt: skip
SIZES = ["S1", "S2", "S3", "S4"]
SIZE_TXT = ["S1 single", "S2 mostly single", "S3 unif. 1-3", "S4 unif. 1-5"]
PAS = list(F.PAS)
DESIGN_TXT = {"XG2": "XG2 endpoints 4/6/8/10", "R1t": "R1 time points 1/2/10 (unlabelled)"}
DESIGN_SHORT = {"XG2": "XG2 design", "R1t": "R1 time points"}


def style():
    plt.rcParams.update({
        "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })  # fmt: skip


def heat(ax, M, cmap, norm, fmt="{:.2f}", log=False):
    vals = np.log2(M) if log else M
    im = ax.imshow(vals, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=7, color=INK2)
                continue
            r, g, b, _ = im.cmap(im.norm(vals[i, j]))
            dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.6,
                    color="white" if dark else INK)  # fmt: skip
    ax.set_xticks(range(len(PAS)), [f"{p:g}" for p in PAS])
    ax.set_yticks(range(4), SIZE_TXT)
    ax.set_xlabel("p_A (share of A founder cells)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def pivot(s, design, mode, col):
    t = s[(s.design == design) & (s["mode"] == mode)].pivot(index="size", columns="pA",
                                                            values=col)  # fmt: skip
    return t.reindex(index=SIZES, columns=PAS).to_numpy(float)


def fig_information(df: pd.DataFrame) -> None:
    """Median over truths of the gain vs single-A-founder known (same design)."""
    s = df.groupby(["design", "size", "pA", "mode"]).median(numeric_only=True).reset_index()
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-3.0, vmax=3.0)
    for metric, col, title in (
        ("lam", "gain_lam_vs_design", "weakest prior-whitened eigenvalue λ_min"),
        ("btoa", "gain_var_B.to_A_vs_design", "1 / Var(B→A)"),
        ("mba", "gain_var_m_BA_vs_design", "1 / Var(m_BA)"),
        ("rho", "gain_var_rho_vs_design", "1 / Var(growth rate ρ)"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), constrained_layout=True)
        for r, design in enumerate(("XG2", "R1t")):
            for c, (mode, mt) in enumerate((("K", "founders known"),
                                            ("E", "founders estimated"))):  # fmt: skip
                ax = axes[r, c]
                im = heat(ax, pivot(s, design, mode, col), DIVERGING, norm, log=True)
                ax.set_title(f"{DESIGN_SHORT[design]}, {mt}")
                if c:
                    ax.set_yticklabels([])
        cb = fig.colorbar(im, ax=axes, shrink=0.8, ticks=np.log2([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]))
        cb.ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
        cb.set_label("gain vs single A founder, known (same design)")
        fig.suptitle(f"Founder distribution and information: {title}, median over the 9 truths",
                     x=0.01, ha="left", fontsize=10, fontweight="bold")  # fmt: skip
        fig.savefig(F.FIG / f"info_{metric}.png")
        plt.close(fig)


def fig_cost(df: pd.DataFrame) -> None:
    """Cost of estimating (E/K) per truth, for lam_min and Var(B->A), both designs."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True, sharey="row")
    dists = ["S1_pA1.0", "S1_pA0.7", "S2_pA0.9", "S2_pA0.7", "S3_pA0.9", "S3_pA0.5",
             "S4_pA0.9", "S4_pA0.5"]  # fmt: skip
    truths = [F.SHORTNAME[t] for t in F.TRUTHS]
    for c, design in enumerate(("XG2", "R1t")):
        for r, (col, lab) in enumerate((("lam_min", "λ_min(E) / λ_min(K)"),
                                        ("sd_B.to_A", "Var_K(B→A) / Var_E(B→A)"))):  # fmt: skip
            ax = axes[r, c]
            for k, t in enumerate(truths):
                ys = []
                for d in dists:
                    sel = (df.truth == t) & (df.design == design) & (df.dist == d)
                    K = df[sel & (df["mode"] == "K")][col].iloc[0]
                    E = df[sel & (df["mode"] == "E")][col].iloc[0]
                    ys.append(E / K if col == "lam_min" else (K / E) ** 2)
                ax.plot(range(len(dists)), ys, "o", ms=4.5, color=SERIES[k % 8],
                        mfc="white" if k >= 8 else SERIES[k % 8], label=t, alpha=0.9)  # fmt: skip
            ax.set_yscale("log")
            ax.axhline(1, color=INK2, lw=0.8)
            ax.set_xticks(range(len(dists)), [d.replace("_pA", "\np_A ") for d in dists])
            ax.set_ylabel(lab)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_title(f"{DESIGN_SHORT[design]}: cost of estimating the founder distribution")
    axes[0, 1].legend(ncol=5, fontsize=7, loc="lower right")
    fig.savefig(F.FIG / "cost_per_truth.png")
    plt.close(fig)


def fig_founder_identifiability(df: pd.DataFrame) -> None:
    s = df[df["mode"] == "E"].groupby(["design", "size", "pA"]).median(numeric_only=True)
    s = s.reset_index()
    s["mode"] = "E"
    fig, axes = plt.subplots(2, 3, figsize=(15, 5.6), constrained_layout=True)
    specs = [("sd_pA_joint", "sd(p_A), transitions estimated", 0.6),
             ("sd_pA_th_known", "sd(p_A), transitions known", 0.6),
             ("sd_mean_size_joint", "sd(mean founder size), transitions estimated", 1.0)]  # fmt: skip
    for r, design in enumerate(("XG2", "R1t")):
        for c, (col, title, vmax) in enumerate(specs):
            ax = axes[r, c]
            M = pivot(s, design, "E", col)
            im = heat(ax, M, BLUES, plt.Normalize(0, vmax), fmt="{:.2f}")
            ax.set_title(f"{DESIGN_SHORT[design]}: {title}", fontsize=8.5)
            if c:
                ax.set_yticklabels([])
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("How well endpoint counts identify the founder distribution (asymptotic sd, "
                 "N = 2048 clones, median over truths)", x=0.01, ha="left", fontsize=10,
                 fontweight="bold")  # fmt: skip
    fig.savefig(F.FIG / "founder_identifiability.png")
    plt.close(fig)


def fig_multi(mt: pd.DataFrame) -> None:
    """Per-truth ratios: combined conditions vs single conditions (same total N)."""
    combos = list(dict.fromkeys(mt.combo))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, sharey=True)
    for ax, design in zip(axes, ("XG2", "R1t"), strict=True):
        for i, cb in enumerate(combos):
            sub = mt[(mt.design == design) & (mt.combo == cb)]
            E = sub[sub["mode"] == "E"].set_index("truth")
            singles = [m for m in sub["mode"].unique() if m.startswith("single")]
            roles = [0, 1, 2] if len(singles) == 3 else [0, 2]
            for j, m in zip(roles, singles, strict=True):
                S = sub[sub["mode"] == m].set_index("truth")
                r = (E.lam_min / S.lam_min).to_numpy()
                x = i + (j - 1) * 0.22
                ax.scatter(np.full(len(r), x), r, s=14, color=SERIES[j], alpha=0.75,
                           label=["vs first condition alone", "vs one condition at the mean p_A",
                                  "vs last condition alone"][j] if i == 0 and design == "XG2"
                           else None)  # fmt: skip
                ax.plot([x - 0.08, x + 0.08], [np.median(r)] * 2, color=INK, lw=1.5)
        ax.set_yscale("log")
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_xticks(range(len(combos)), [c.replace(" (", "\n(").replace(": ", ":\n")
                                           for c in combos], fontsize=7)  # fmt: skip
        ax.set_ylabel("λ_min(combined) / λ_min(single condition), both estimated")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(DESIGN_TXT[design])
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Several seeding conditions with shared transitions and unknown, separate "
                 "founder distributions (same total clones; bar = median over truths)", x=0.01,
                 ha="left", fontsize=10, fontweight="bold")  # fmt: skip
    fig.savefig(F.FIG / "multi_conditions.png")
    plt.close(fig)


def main() -> None:
    style()
    F.FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(F.TAB / "fisher_main.csv")
    fig_information(df)
    fig_cost(df)
    fig_founder_identifiability(df)
    fig_multi(pd.read_csv(F.TAB / "multi_conditions.csv"))


if __name__ == "__main__":
    main()
