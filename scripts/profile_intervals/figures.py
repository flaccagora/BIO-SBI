"""Figures of the profile-interval study (called by analyse.py; house style of report_exact_gw)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from analyse import FIG, GTS, MLABEL, PARAMS12, THR  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7.5,
    "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 7.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "legend.frameon": False, "legend.fontsize": 7,
    "lines.linewidth": 1.4, "lines.markersize": 4, "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "pdf.fonttype": 42,
})  # fmt: skip
COLOR = {"profile": INK, "profile_bc": INK2, "flat": BLUE, "flat_btoa": ORANGE,
         "dir_half": AQUA, "jeffreys": YELLOW, "wald": MUTED}  # fmt: skip
TITLE = {"T1_ref": "T1 reference", "T2_toA05": "T2 B→A 0.5", "T4_slowA": "T4 slow A",
         "T3_Bdiv": "T3 B divides", "T5_C7": "T5 C7", "P1": "P1", "P2": "P2", "P3": "P3",
         "P4": "P4"}  # fmt: skip
NALL = [128, 256, 512, 1024, 2048, 4096]
SEQ = LinearSegmentedColormap.from_list("blues", ["#9ec3f0", "#6da7ec", "#2a78d6", "#1c5cab",
                                                  "#0d3a73"])  # fmt: skip
HEAT = LinearSegmentedColormap.from_list("heat", ["#f0f5fc", "#c6dbf7", "#6da7ec", "#2a78d6",
                                                 "#1c5cab", "#0d3a73"])  # fmt: skip
NCOL = {n: SEQ(i / (len(NALL) - 1)) for i, n in enumerate(NALL)}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


def pname(p: str) -> str:
    return {"B.div": "B div (total)", "B.m_BA": "m_BA"}.get(p, p.replace("div_", "→")
                                                           .replace("to_", "→")
                                                           .replace(".", " "))  # fmt: skip


# --------------------------------------------------------------------- (a) gallery
def curve_gallery(curves, prof, param="B.to_A", name="gallery_btoa", ymin=-6.0):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True, sharey=True)
    cv = curves[curves.parameter == param]
    for ax, gt in zip(axes.flat, GTS, strict=True):
        sub = cv[cv["gt"] == gt]
        if sub.empty:
            ax.set_visible(False)
            continue
        truth = prof[(prof["gt"] == gt) & (prof.parameter == param)].truth.iloc[0]
        for (n, s), g in sub.groupby(["n", "seed"]):
            g = g.sort_values("c")
            ax.plot(g.c, np.maximum(-g["drop"], ymin - 1), color=NCOL[n],
                    lw=1.3 if s == 0 else 0.6, alpha=1.0 if s == 0 else 0.45,
                    zorder=3 if s == 0 else 2)  # fmt: skip
        for thr, ls in ((THR[90], ":"), (THR[95], "--")):
            ax.axhline(-thr, color=INK2, lw=0.7, ls=ls, zorder=1)
        ax.axvline(truth, color=ORANGE, lw=1.0, zorder=4)
        ax.set_title(f"{TITLE[gt]}  ({param} = {truth:.2f})", fontsize=7.5)
        ax.set_ylim(ymin, 0.3)
        ax.set_xlim(0, curves[curves.parameter == param].c.max())
    for ax in axes[-1]:
        ax.set_xlabel(param)
    for ax in axes[:, 0]:
        ax.set_ylabel("ℓp − max (nats)")
    handles = [Line2D([], [], color=NCOL[n], lw=1.3, label=f"n = {n}") for n in NALL]
    handles += [Line2D([], [], color=INK2, ls=":", lw=0.7, label="90% (1.353)"),
                Line2D([], [], color=INK2, ls="--", lw=0.7, label="95% (1.921)"),
                Line2D([], [], color=ORANGE, lw=1.0, label="truth")]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=6.3,
               bbox_to_anchor=(0.5, -0.03), handlelength=1.6, columnspacing=1.0)  # fmt: skip
    fig.text(0.5, 1.0, "seed 0 bold, seeds 1–2 faint", ha="center", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    save(fig, name)


# --------------------------------------------------------------------- (b) forest
def forest(L, prof, param="B.to_A", level=95, name="forest_btoa", seeds=(0, 1, 2)):
    meths = ["profile", "flat", "flat_btoa", "dir_half", "jeffreys", "wald"]
    off = dict(zip(meths, np.linspace(0.3, -0.3, len(meths)), strict=True))
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 9.6 if len(seeds) > 1 else 7.0), sharex=True)
    sub = L[(L.parameter == param) & (L.level == level) & L.seed.isin(seeds)]
    for ax, gt in zip(axes.flat, GTS, strict=True):
        g = sub[sub["gt"] == gt]
        if g.empty:
            ax.set_visible(False)
            continue
        cells = sorted(g[["n", "seed"]].drop_duplicates().itertuples(index=False))
        ylab = []
        for y, (n, s) in enumerate(cells[::-1]):
            ylab.append(f"{n} s{s}" if len(seeds) > 1 else f"n={n}")
            pr = prof[(prof["gt"] == gt) & (prof.n == n) & (prof.seed == s)
                      & (prof.parameter == param)].iloc[0]  # fmt: skip
            for m in meths:
                yy = y + off[m]
                if m == "profile":
                    segs = [tuple(map(float, t.split(":"))) for t in pr[f"segs{level}"].split(";")
                            if t]  # fmt: skip
                    for lo, hi in segs:
                        ax.plot([lo, hi], [yy, yy], color=COLOR[m], lw=1.8, solid_capstyle="butt")
                    ax.plot(pr.mle_profile, yy, "|", color=COLOR[m], ms=4, mew=1.0)
                    continue
                r = g[(g.n == n) & (g.seed == s) & (g.method == m)]
                if r.empty or not r.exists.iloc[0]:
                    continue
                lo, hi = max(r.lo.iloc[0], 0), min(r.hi.iloc[0], 1)
                ax.plot([lo, hi], [yy, yy], color=COLOR[m], lw=1.1, solid_capstyle="butt")
            if y and (s == max(seeds)) and len(seeds) > 1:
                ax.axhline(y - 0.5, color=GRID, lw=0.8)
        ax.axvline(prof[(prof["gt"] == gt) & (prof.parameter == param)].truth.iloc[0], color=INK,
                   lw=0.8, ls="--", zorder=0)  # fmt: skip
        ax.set_yticks(range(len(cells)), ylab, fontsize=5.2)
        ax.set_title(TITLE[gt], fontsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(-0.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel(f"{param}, {level}% interval")
    handles = [Line2D([], [], color=COLOR[m], lw=1.8 if m == "profile" else 1.1,
                      label=MLABEL[m] + (" (union = gaps)" if m == "profile" else ""))
               for m in meths]  # fmt: skip
    handles.append(Line2D([], [], color=INK, lw=0.8, ls="--", label="truth"))
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=6.3,
               bbox_to_anchor=(0.5, -0.015), handlelength=1.6, columnspacing=1.0)  # fmt: skip
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    save(fig, name)


# --------------------------------------------------------------------- (c) coverage
def coverage_vs_n(L, name="coverage_vs_n"):
    meths = ["profile", "flat", "flat_btoa", "dir_half", "jeffreys", "wald"]
    pos = L[~L.truth_zero & L.exists]
    sets = [("B→A", pos[pos.parameter == "B.to_A"]),
            ("all 12 probabilities (truth > 0)", pos[pos.parameter.isin(PARAMS12)])]  # fmt: skip
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for i, (lab, df) in enumerate(sets):
        for j, lev in enumerate((90, 95)):
            ax = axes[i, j]
            d = df[df.level == lev]
            for m in meths:
                c = d[d.method == m].groupby("n")["covered"].mean()
                if c.empty:
                    continue
                ax.plot(np.log2(c.index / 128), c.values, "o-", color=COLOR[m], ms=3.5,
                        lw=1.6 if m == "profile" else 1.1, label=MLABEL[m])  # fmt: skip
            ax.axhline(lev / 100, color=INK2, lw=0.7, ls="--")
            ax.set_title(f"{lab}: {lev}% intervals", fontsize=7.5)
            ax.set_xticks(range(6), NALL)
            lo = 0.4 if i == 0 else 0.55
            ax.set_ylim(lo, 1.01)
    for ax in axes[-1]:
        ax.set_xlabel("clones per endpoint (n)")
    for ax in axes[:, 0]:
        ax.set_ylabel("coverage")
    axes[0, 0].legend(fontsize=6.3, loc="lower left", ncol=2)
    fig.tight_layout()
    save(fig, name)


# --------------------------------------------------------------------- (d) heatmap
def width_heatmap(prof, name="width_heatmap", level=95):
    params = PARAMS12 + ["B.div", "B.m_BA"]
    cols = [(gt, n) for gt in GTS for n in NALL if ((prof["gt"] == gt) & (prof.n == n)).any()]
    mat = np.full((len(params), len(cols)), np.nan)
    opn = np.zeros_like(mat, dtype=bool)
    uni = np.zeros_like(mat, dtype=bool)
    for j, (gt, n) in enumerate(cols):
        for i, p in enumerate(params):
            g = prof[(prof["gt"] == gt) & (prof.n == n) & (prof.parameter == p)]
            w = g[f"len{level}"] / g.cmax
            mat[i, j] = np.exp(np.log(np.maximum(w, 1e-4)).mean())
            opn[i, j] = g[f"open_hi{level}"].any()
            uni[i, j] = (g[f"nseg{level}"] > 1).any()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    im = ax.imshow(mat, cmap=HEAT, norm=LogNorm(vmin=0.02, vmax=1), aspect="auto",
                   interpolation="nearest")  # fmt: skip
    yy, xx = np.nonzero(opn)
    ax.plot(xx, yy, "o", ms=1.8, color=SURFACE, mec=INK, mew=0.4, lw=0)
    yy, xx = np.nonzero(uni)
    ax.plot(xx, yy, "x", ms=3.0, color=ORANGE, mew=0.9, lw=0)
    ax.set_yticks(range(len(params)), [pname(p) for p in params])
    ax.set_xticks(range(len(cols)), [str(n) for _, n in cols], fontsize=5.0, rotation=90)
    start = 0
    for k, gt in enumerate(GTS):
        w = sum(1 for g, _ in cols if g == gt)
        ax.text(start + (w - 1) / 2, -0.9, gt.split("_")[0], ha="center", va="bottom",
                fontsize=7, color=INK2, fontweight="bold")  # fmt: skip
        if k:
            ax.axvline(start - 0.5, color=SURFACE, lw=2)
        start += w
    ax.axhline(5.5, color=SURFACE, lw=2)
    ax.axhline(11.5, color=SURFACE, lw=2)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(f"profile {level}% interval length / range", fontsize=6.5)
    cb.outline.set_visible(False)
    ax.set_xlabel("clones per endpoint (n), grouped by truth   (geometric mean over seeds;"
                  " o reaches the upper bound, x union, in any seed)")  # fmt: skip
    fig.tight_layout()
    save(fig, name)


# --------------------------------------------------------------------- (e) others
def other_curves(curves, prof, name="curves_other"):
    picks = [("P3", "B.div_AA"), ("P3", "B.div_AB"), ("P3", "B.div"),
             ("T1_ref", "B.div"), ("T1_ref", "B.m_BA"), ("T3_Bdiv", "B.to_A"),
             ("T1_ref", "A.persist"), ("P1", "B.to_A"), ("T5_C7", "B.div_BB")]  # fmt: skip
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.2), sharey=True)
    for ax, (gt, p) in zip(axes.flat, picks, strict=True):
        sub = curves[(curves["gt"] == gt) & (curves.parameter == p) & (curves.seed == 0)]
        if sub.empty:
            ax.set_visible(False)
            continue
        for n, g in sub.groupby("n"):
            g = g.sort_values("c")
            ax.plot(g.c, np.maximum(-g["drop"], -7), color=NCOL[n], lw=1.2)
        truth = prof[(prof["gt"] == gt) & (prof.parameter == p)].truth.iloc[0]
        for thr, ls in ((THR[90], ":"), (THR[95], "--")):
            ax.axhline(-thr, color=INK2, lw=0.7, ls=ls)
        ax.axvline(truth, color=ORANGE, lw=1.0)
        ax.set_ylim(-6, 0.3)
        ax.set_xlim(0, 2 if p == "B.m_BA" else 1)
        ax.set_title(f"{TITLE[gt]}: {pname(p)}", fontsize=7.5)
        ax.set_xlabel(pname(p))
    for ax in axes[:, 0]:
        ax.set_ylabel("ℓp − max (nats)")
    handles = [Line2D([], [], color=NCOL[n], lw=1.2, label=f"n = {n}") for n in NALL]
    handles.append(Line2D([], [], color=ORANGE, lw=1.0, label="truth"))
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=6.3,
               bbox_to_anchor=(0.5, -0.03))  # fmt: skip
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, name)


def width_vs_n(L, name="width_vs_n"):
    meths = ["profile", "flat", "flat_btoa", "dir_half", "jeffreys", "wald"]
    d = L[L.exists & (L.level == 95) & ~L.truth_zero]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True)
    for ax, (lab, df) in zip(axes, [("B→A", d[d.parameter == "B.to_A"]),
                                    ("all 12 probabilities (truth > 0)",
                                     d[d.parameter.isin(PARAMS12)])], strict=True):  # fmt: skip
        for m in meths:
            w = df[df.method == m].groupby("n")["width"].mean()
            if w.empty:
                continue
            ax.plot(np.log2(w.index / 128), w.values, "o-", color=COLOR[m], ms=3.5,
                    lw=1.6 if m == "profile" else 1.1, label=MLABEL[m])  # fmt: skip
        ax.set_yscale("log")
        ax.set_xticks(range(6), NALL)
        ax.set_title(f"{lab}: mean 95% width", fontsize=7.5)
        ax.set_xlabel("clones per endpoint (n)")
    axes[0].set_ylabel("width (clipped to [0, 1])")
    axes[1].legend(fontsize=6.3, ncol=2)
    fig.tight_layout()
    save(fig, name)


def make_all(prof: pd.DataFrame, curves: pd.DataFrame, L: pd.DataFrame, t: dict) -> None:
    curve_gallery(curves, prof)
    forest(L, prof, seeds=(0,), name="forest_btoa")
    forest(L, prof, name="forest_btoa_allseeds")
    forest(L, prof, param="B.div", seeds=(0,), name="forest_bdiv")
    coverage_vs_n(L)
    width_heatmap(prof)
    width_vs_n(L)
    other_curves(curves, prof)
    curve_gallery(curves, prof, param="B.div", name="gallery_bdiv")
