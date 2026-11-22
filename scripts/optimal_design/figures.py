"""Figures of the optimal-design study (reads ``designs.npz`` and the tables).

  uv run python scripts/optimal_design/figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from od_common import FIG, OUT, POINTS, SHORTNAME, TAB, TRUTHS  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]  # fmt: skip
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
              "#0d366b"])  # fmt: skip
LAB = [f"{f}{h}" for f, h in POINTS]
TS = [SHORTNAME[t] for t in TRUTHS]


def style():
    plt.rcParams.update({
        "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight", "axes.titlesize": 9.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })  # fmt: skip


def payload() -> dict:
    return json.loads(str(np.load(OUT / "designs.npz")["payload"]))


def weights_figure(res: dict) -> None:
    rob = res["robust"]
    crits = [("E", "E-optimal (weakest direction, λ_min)"), ("c", "c-optimal (B→A variance)"),
             ("D", "D-optimal (log det)")]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for ax, (cr, title) in zip(axes, crits, strict=True):
        rows = [np.array(res["opt_full"][gt][cr]) for gt in TRUTHS]
        names = list(TS)
        rows += [np.array(rob[f"clone | full | {cr} maximin"]), np.array(rob[f"clone | full | {cr} mean"])]
        names += ["robust: maximin", "robust: mean"]
        rows.append(np.array(rob["clone | full | E+c maximin"]))
        names.append("robust: E+c maximin")
        Wm = np.array(rows)
        ax.imshow(Wm, cmap=BLUES, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for i in range(Wm.shape[0]):
            for j in range(Wm.shape[1]):
                if Wm[i, j] >= 0.015:
                    ax.text(j, i, f"{Wm[i, j]:.2f}"[1:] if Wm[i, j] < 1 else "1",
                            ha="center", va="center", fontsize=6.3,
                            color="white" if Wm[i, j] > 0.45 else INK)  # fmt: skip
        ax.axvline(11.5, color=INK2, lw=1.2)
        ax.axhline(8.5, color=INK2, lw=1.2)
        ax.set_xticks(range(len(LAB)))
        ax.set_xticklabels([str(h) for _, h in POINTS], fontsize=7)
        ax.set_xlabel("observation generation h   (left: A-founded | right: B-founded)")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_title(title, pad=18)
        ax.text(5.5, -0.75, "A founder", ha="center", fontsize=8, color=INK2)
        ax.text(17.5, -0.75, "B founder", ha="center", fontsize=8, color=INK2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Optimal shares of detected clones per (founder, generation): per truth and robust "
                 "over the 9 truths (efficiency-normalised); T1/T2/T4 at ε = 0.01",
                 x=0.01, ha="left", fontsize=10)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "optimal_weights.png")
    plt.close(fig)


def sd_figure(res: dict) -> None:
    N = np.array(res["curves_N"])
    protos = list(res["protocols"])
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.6), sharex=True, sharey=True)
    for ax, gt in zip(axes.ravel(), TRUTHS, strict=True):
        for k, name in enumerate(protos):
            cv = res["curves"][f"{name} | {gt}"]
            bold = k == 0 or name.startswith("R1")
            ax.plot(N, cv["sd"], color=SERIES[k % 8], lw=2.6 if bold else 1.3,
                    zorder=3 if bold else 2, label=name)  # fmt: skip
        ax.axhline(0.05, color=INK2, lw=0.9, ls=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.003, 1.0)
        ax.grid(True, which="major")
        ax.set_title(SHORTNAME[gt])
        for n in (512, 2048):
            ax.axvline(n, color=GRID, lw=0.8, zorder=0)
    for ax in axes[-1]:
        ax.set_xlabel("total detected clones N")
    for ax in axes[:, 0]:
        ax.set_ylabel("asymptotic sd of B→A")
    axes[0, 0].text(70, 0.056, "sd = 0.05", fontsize=7, color=INK2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("B→A asymptotic sd vs total number of detected clones, practical protocols "
                 "(Fisher prediction, no prior; T1/T2/T4 at ε = 0.01)", x=0.01, ha="left",
                 fontsize=10)  # fmt: skip
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(FIG / "btoa_sd_vs_N.png")
    plt.close(fig)


def efficiency_figure() -> None:
    prot = pd.read_csv(TAB / "protocols.csv")
    order = list(dict.fromkeys(prot.protocol))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (col, title) in zip(axes, [("eff_E", "E-efficiency (λ_min / optimal λ_min)"),
                                       ("eff_c", "B→A efficiency (optimal var / var)")]):  # fmt: skip
        pv = prot.pivot(index="protocol", columns="truth", values=col).loc[order, TS]
        M = pv.to_numpy()
        ax.imshow(np.log10(np.clip(M, 1e-3, 1)), cmap=BLUES, vmin=-2.3, vmax=0, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}" if v >= 0.095 else f"{v:.3f}", ha="center",
                        va="center", fontsize=7, color="white" if v > 0.07 else INK)  # fmt: skip
        ax.set_xticks(range(len(TS)))
        ax.set_xticklabels(TS)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Efficiency of the practical protocols across the truths, relative to each "
                 "truth's own optimum over all (founder, generation) designs at equal N (log colour scale)",
                 x=0.01, ha="left", fontsize=10)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "efficiency_across_truths.png")
    plt.close(fig)


def restriction_figure() -> None:
    eff = pd.read_csv(TAB / "robust_efficiencies.csv")
    eff = eff[(eff.cost == "clone") & (eff.design == "E+c maximin")]
    prot = pd.read_csv(TAB / "protocols.csv")
    cur = prot[prot.protocol == "current: A 4/6/8/10"]
    spaces = list(dict.fromkeys(eff.space))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, [("eff_E", "E-efficiency"), ("eff_c", "B→A efficiency")]):
        for i, sp in enumerate(spaces):
            v = eff[eff.space == sp][col].to_numpy()
            ax.plot([v.min(), v.max()], [i, i], color=SERIES[0], lw=2, solid_capstyle="round")
            ax.plot(v, np.full(len(v), i), "o", ms=4.5, color=SERIES[0], mec="white", mew=0.8)
        v = cur[col].to_numpy()
        i = len(spaces)
        ax.plot([v.min(), v.max()], [i, i], color=SERIES[1], lw=2)
        ax.plot(v, np.full(len(v), i), "o", ms=4.5, color=SERIES[1], mec="white", mew=0.8)
        ax.set_xscale("log")
        ax.set_xlim(0.005, 1.1)
        ax.grid(True, axis="x")
        ax.set_yticks(range(len(spaces) + 1))
        ax.set_yticklabels([f"robust optimum, {s}" for s in spaces] + ["current A 4/6/8/10"])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("efficiency vs the unrestricted optimum (dots: 9 truths)")
    fig.suptitle("What a restricted design space costs: robust (E+c maximin) design per "
                 "space", x=0.01, ha="left", fontsize=10)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "design_space_restrictions.png")
    plt.close(fig)


def main() -> None:
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    res = payload()
    weights_figure(res)
    sd_figure(res)
    efficiency_figure()
    restriction_figure()


if __name__ == "__main__":
    main()
