"""Figures and tables of the exact-GW report (``report_exact_gw/report.tex``).

Usage (from the repository root)::

    uv run python report_exact_gw/make_figures.py [--results /path/to/results]

Reads ``results/exact_gw/data_size/`` (XG2: ``xg2_runs.csv``, ``xg2_mle.csv``,
``runs_index.csv`` and every run's ``metrics.json`` / ``posterior_samples.csv.gz``) and the
identifiability note ``docs/xg2_mle_vs_posterior/`` (figures copied). Writes vector PDFs
to ``report_exact_gw/figures/`` and LaTeX table bodies + CSVs to
``report_exact_gw/tables/``. Only reads ``results/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
XG2 = ROOT / "results" / "exact_gw" / "data_size"
NOTE = ROOT / "docs" / "xg2_mle_vs_posterior"
OUT = ROOT / "report_exact_gw"
FIG, TAB = OUT / "figures", OUT / "tables"

# house style of report/make_figures.py (validated reference palette)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#b8302f", "#ef8a7a", "#f0efec", "#6da7ec", "#1c5cab"]
)  # red = posterior wider than prior, blue = contracted
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

GTS = ["T1_ref", "T2_toA05", "T4_slowA", "T3_Bdiv", "T5_C7", "P1", "P2", "P3", "P4"]
LABEL = {"T1_ref": "T1 reference", "T2_toA05": "T2 B$\\to$A 0.5", "T4_slowA": "T4 slow A",
         "T3_Bdiv": "T3 B divides", "T5_C7": "T5 C7 (large clones)", "P1": "P1 (prior draw)",
         "P2": "P2 (prior draw)", "P3": "P3 (prior draw)", "P4": "P4 (prior draw)"}  # fmt: skip
TEXLABEL = {k: v.replace("$\\to$", "$\\to$") for k, v in LABEL.items()}
NS = [128, 256, 512, 1024]
PARAMS = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
          "B.death", "B.persist", "B.to_A", "B.div_AA", "B.div_AB", "B.div_BB"]  # fmt: skip
PNAME = {p: p.replace("div_", "$\\to$").replace(".to_", "$\\to$").replace(".", " ")
         for p in PARAMS}  # fmt: skip


def pname(p: str) -> str:
    row, out = p.split(".")
    nice = {"death": "death", "persist": "persist", "to_B": "$\\to$B", "to_A": "$\\to$A",
            "div_AA": "$\\to$AA", "div_AB": "$\\to$AB", "div_BB": "$\\to$BB"}[out]  # fmt: skip
    return f"{row} {nice}" if not nice.startswith("$") else f"{row}{nice}"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


def load():
    runs = pd.read_csv(XG2 / "xg2_runs.csv")
    flat = runs[runs.prior == "flat"].copy()
    mle = pd.read_csv(XG2 / "xg2_mle.csv")
    index = pd.read_csv(XG2 / "runs_index.csv")
    meta = []
    for r in index.itertuples():
        m = json.loads((XG2 / r.new / "metrics.json").read_text())
        e = m["exact_gw"]
        meta.append({"gt": r.gt, "n": r.n, "seed": r.seed, "kernel": r.kernel,
                     "hours": m["total_seconds"] / 3600, "grad_evals": e["gradient_evals"],
                     "rhat": e["max_rhat"], "min_ess": e["min_ess"],
                     "divergences": e["divergences"],
                     "depth": float(np.mean(e["mean_tree_depth"])),
                     "ms_per_eval": 1e3 * e["seconds_per_eval"]})  # fmt: skip
    return runs, flat, mle, index, pd.DataFrame(meta)


def tex_table(df: pd.DataFrame, name: str, fmt: dict | None = None) -> None:
    """Write a booktabs-free table body (rows only) for \\input inside a tabular."""
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    fmt = fmt or {}
    lines = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            f = fmt.get(c)
            cells.append(f(v) if f else (f"{v:.3g}" if isinstance(v, float) else str(v)))
        lines.append(" & ".join(cells) + r" \\")
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print("table", name)


# ------------------------------------------------------------------ figures
def fig_btoa(flat, mle):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.0), sharex=True)
    b = flat[flat.parameter == "B.to_A"]
    m = mle[mle.parameter == "B.to_A"]
    for ax, gt in zip(axes.flat, GTS, strict=True):
        g, gm = b[b["gt"] == gt], m[m["gt"] == gt]
        for s, off in zip((0, 1, 2), (-0.09, 0, 0.09), strict=True):
            gs = g[g.seed == s].sort_values("n")
            x = np.log2(gs.n) + off
            ax.errorbar(x, gs["mean"], yerr=gs.sd, fmt="o", color=BLUE, ms=3.2, lw=0.9,
                        capsize=0, label="posterior mean ± sd" if s == 0 else None)  # fmt: skip
            gms = gm[gm.seed == s].sort_values("n")
            ax.plot(np.log2(gms.n) + off + 0.04, gms.mle, "D", mfc="none", mec=ORANGE, ms=3.6,
                    mew=0.9, label="MLE" if s == 0 else None)  # fmt: skip
        ax.axhline(g.truth.iloc[0], color=INK, lw=0.9, ls="--", label="truth")
        ax.set_title(LABEL[gt])
        ax.set_ylim(-0.03, 1.0)
        ax.set_xticks(np.log2(NS), [str(n) for n in NS])
    for ax in axes[-1]:
        ax.set_xlabel("clones per endpoint (n)")
    for ax in axes[:, 0]:
        ax.set_ylabel("B$\\to$A")
    axes[0, 0].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "btoa_vs_n")


def _sd_heatmap(mat, name, label, dots=None, grey=None):
    """Heatmap of log10(sd / prior sd): rows = PARAMS, columns = truth x n."""
    cmap = DIV.reversed().copy()
    cmap.set_bad("#bdbcb6")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    shown = np.ma.masked_invalid(np.clip(mat, -1.5, 0.5))
    im = ax.imshow(shown, cmap=cmap, norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=0.5),
                   aspect="auto", interpolation="nearest")  # fmt: skip
    if dots is not None:
        yy, xx = np.nonzero(dots)
        ax.plot(xx, yy, "o", ms=1.6, color=INK, lw=0)
    ax.set_yticks(range(len(PARAMS)), [pname(p) for p in PARAMS])
    ax.set_xticks(range(len(GTS) * 4), [str(n) for _ in GTS for n in NS], fontsize=5.5,
                  rotation=90)  # fmt: skip
    for k, gt in enumerate(GTS):
        ax.text(4 * k + 1.5, -0.9, gt.split("_")[0], ha="center", va="bottom", fontsize=7,
                color=INK2, fontweight="bold")  # fmt: skip
        if k:
            ax.axvline(4 * k - 0.5, color=SURFACE, lw=2)
    ax.axhline(5.5, color=SURFACE, lw=2)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(label, fontsize=6.5)
    ticks = [-1.5, -1, -0.5, 0, 0.5]
    cb.set_ticks(ticks, labels=[f"{10 ** t:.2g}" for t in ticks])
    cb.outline.set_visible(False)
    ax.set_xlabel("clones per endpoint (n), grouped by truth")
    fig.tight_layout()
    save(fig, name)


def fig_contraction(flat):
    # posterior sd / prior sd on a log scale (geometric mean over datasets)
    f = flat.assign(lr=np.log10(flat.sd / np.sqrt(flat.prior_var)))
    c = f.groupby(["gt", "n", "parameter"])["lr"].mean().unstack("parameter")
    mat = np.vstack([c.loc[(gt, n), PARAMS].to_numpy() for gt in GTS for n in NS]).T
    _sd_heatmap(mat, "contraction", "posterior sd / prior sd")


def fig_contraction_mle(flat, index):
    """Same layout for the MLE: Wald standard error (observed Fisher information) / prior sd.

    Per cell: geometric mean over the datasets with a finite standard error. Grey = the MLE
    is 0 in every dataset (no standard error on the boundary); darkest red = no curvature
    (infinite standard error) wherever the MLE is nonzero; dot = standard error undefined
    (MLE 0 or no curvature) in some but not all datasets.
    """
    prior_sd = flat.groupby("parameter").prior_var.first().pow(0.5)
    rows = []
    for r in index.itertuples():
        m = json.loads((XG2 / r.new / "mle.json").read_text())
        se, zero = m.get("se"), set(m.get("se_info", {}).get("at_zero", []))
        if se is None:
            raise SystemExit("mle.json without se: run scripts/exact_gw_mle_se.py first")
        for p in PARAMS:
            v = se.get(p)
            state = "zero" if p in zero else ("inf" if v is None else "ok")
            rows.append({"gt": r.gt, "n": r.n, "seed": r.seed, "parameter": p, "state": state,
                         "lr": np.log10(v / prior_sd[p]) if state == "ok" else np.nan})  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "mle_se.csv", index=False)
    mat = np.full((len(PARAMS), len(GTS) * 4), np.nan)
    dots = np.zeros_like(mat, dtype=bool)
    for j, (gt, n) in enumerate((gt, n) for gt in GTS for n in NS):
        for i, p in enumerate(PARAMS):
            g = df[(df["gt"] == gt) & (df.n == n) & (df.parameter == p)]
            ok = g[g.state == "ok"]
            if len(ok):
                mat[i, j] = ok.lr.mean()
            elif (g.state == "inf").any():
                mat[i, j] = 0.5  # no curvature in every dataset with a nonzero MLE
            dots[i, j] = 0 < (g.state != "ok").sum() < len(g)
    _sd_heatmap(mat, "contraction_mle", "MLE standard error / prior sd", dots=dots)
    counts = df.groupby("state").size().to_dict()
    print("MLE se states:", counts)


def mse_table(flat, mle) -> pd.DataFrame:
    """MSE over the 3 datasets per (truth, n, parameter) for posterior mean and MLE,
    split into bias^2 and variance (population variance over datasets); plus the mean
    posterior variance (to compare actual with reported error)."""
    post = flat[["gt", "n", "seed", "parameter", "truth", "mean", "sd"]].merge(
        mle[["gt", "n", "seed", "parameter", "mle"]], on=["gt", "n", "seed", "parameter"])
    rows = []
    for (gt, n, p), g in post.groupby(["gt", "n", "parameter"]):
        t = g.truth.iloc[0]
        r = {"gt": gt, "n": n, "parameter": p, "truth": t, "post_var": float((g.sd ** 2).mean())}
        for est, col in (("post", "mean"), ("mle", "mle")):
            x = g[col].to_numpy()
            r[f"mse_{est}"] = float(np.mean((x - t) ** 2))
            r[f"bias2_{est}"] = float((x.mean() - t) ** 2)
            r[f"var_{est}"] = float(x.var())
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "mse.csv", index=False)
    return out


def fig_rmse(flat, mle):
    m = mse_table(flat, mle)
    prior_sd = flat.groupby("parameter").prior_var.first().pow(0.5)
    for est, label in (("post", "RMSE of the posterior mean / prior sd"),
                       ("mle", "RMSE of the MLE / prior sd")):  # fmt: skip
        mat = np.full((len(PARAMS), len(GTS) * 4), np.nan)
        for j, (gt, n) in enumerate((gt, n) for gt in GTS for n in NS):
            for i, p in enumerate(PARAMS):
                v = m[(m["gt"] == gt) & (m.n == n) & (m.parameter == p)][f"mse_{est}"]
                mat[i, j] = np.log10(np.sqrt(max(float(v.iloc[0]), 1e-12)) / prior_sd[p])
        _sd_heatmap(mat, f"rmse_{est}", label)


def mse_tables(flat, mle):
    m = pd.read_csv(TAB / "mse.csv")
    rows = []
    for gt in GTS:
        r = {"truth": LABEL[gt]}
        for n in (128, 1024):
            b = m[(m["gt"] == gt) & (m.n == n) & (m.parameter == "B.to_A")].iloc[0]
            rp, rm = np.sqrt(b.mse_post), np.sqrt(b.mse_mle)
            r[f"b{n}"] = (f"\\textbf{{{rm:.2f}}} / {rp:.2f}" if rm < rp
                          else f"{rm:.2f} / \\textbf{{{rp:.2f}}}")  # fmt: skip
            r[f"bias{n}"] = f"{100 * b.bias2_mle / max(b.mse_mle, 1e-12):.0f} / {100 * b.bias2_post / max(b.mse_post, 1e-12):.0f}"
        for n in (128, 1024):
            a = m[(m["gt"] == gt) & (m.n == n)]
            rp, rm = np.sqrt(a.mse_post.mean()), np.sqrt(a.mse_mle.mean())
            r[f"a{n}"] = (f"\\textbf{{{rm:.3f}}} / {rp:.3f}" if rm < rp
                          else f"{rm:.3f} / \\textbf{{{rp:.3f}}}")  # fmt: skip
            r[f"cal{n}"] = f"{np.sqrt(a.mse_post.mean() / a.post_var.mean()):.2f}"
        rows.append(r)
    tex_table(pd.DataFrame(rows)[["truth", "b128", "bias128", "b1024", "bias1024", "a128", "a1024",
                                  "cal128", "cal1024"]], "mse")  # fmt: skip


def fig_error(mle):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4), sharex=True)
    for ax, gt in zip(axes.flat, GTS, strict=True):
        g = mle[(mle["gt"] == gt) & (mle.parameter == "B.to_A")]
        a = mle[mle["gt"] == gt]
        e_b = g.groupby("n")[["abs_err_post", "abs_err_mle"]].mean()
        e_a = a.groupby("n")[["abs_err_post", "abs_err_mle"]].mean()
        x = np.log2(e_b.index)
        ax.plot(x, e_b.abs_err_post, "-o", color=BLUE, ms=3.5, label="posterior mean, B$\\to$A")
        ax.plot(x, e_b.abs_err_mle, "-D", color=ORANGE, ms=3.2, label="MLE, B$\\to$A")
        ax.plot(x, e_a.abs_err_post, ":", color=BLUE, lw=1.1, label="posterior mean, all 12")
        ax.plot(x, e_a.abs_err_mle, ":", color=ORANGE, lw=1.1, label="MLE, all 12")
        ax.set_title(LABEL[gt])
        ax.set_ylim(0, 0.46)
        ax.set_xticks(np.log2(NS), [str(n) for n in NS])
    for ax in axes[-1]:
        ax.set_xlabel("clones per endpoint (n)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean |estimate − truth|")
    axes[0, 0].legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    save(fig, "mle_vs_posterior")


def fig_bvm(flat):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4), sharex=True)
    f = flat.assign(sdn=flat.sd * np.sqrt(flat.n))
    for ax, gt in zip(axes.flat, GTS, strict=True):
        for p, col, mk in (("A.div_AA", BLUE, "o"), ("B.to_A", ORANGE, "D"), ("B.div_BB", AQUA, "s")):
            g = f[(f["gt"] == gt) & (f.parameter == p)].groupby("n").sdn.mean()
            ax.plot(np.log2(g.index), g.values, "-" + mk, color=col, ms=3.2, label=pname(p))
        ax.set_title(LABEL[gt])
        ax.set_ylim(0, 3.6)
        ax.set_xticks(np.log2(NS), [str(n) for n in NS])
    for ax in axes[-1]:
        ax.set_xlabel("clones per endpoint (n)")
    for ax in axes[:, 0]:
        ax.set_ylabel("posterior sd × $\\sqrt{n}$")
    axes[0, 0].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "sd_sqrt_n")


def fig_prior(runs):
    sym = runs[runs.prior.isin(["sparse", "jeffreys", "smooth", "central"]) & runs.reliable]
    wrong = runs[(runs.prior == "wrong") & runs.reliable]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.4), sharex=True)
    for ax, gt in zip(axes.flat, GTS, strict=True):
        for df, col, mk, lab in ((sym, BLUE, "o", "symmetric Dirichlet (0.2–5)"),
                                 (wrong, ORANGE, "D", "concentrated wrong prior")):  # fmt: skip
            g = df[(df["gt"] == gt) & (df.parameter == "B.to_A")]
            if g.empty:
                continue
            med = g.groupby(["n", "seed"]).shift_sd.max().groupby("n").median()
            ax.plot(np.log2(med.index), med.values, "-" + mk, color=col, ms=3.3, label=lab)
        ax.axhline(0.25, color=INK2, lw=0.8, ls="--")
        ax.set_title(LABEL[gt])
        ax.set_ylim(0, 2.0)
        ax.set_xticks(np.log2(NS), [str(n) for n in NS])
    for ax in axes[-1]:
        ax.set_xlabel("clones per endpoint (n)")
    for ax in axes[:, 0]:
        ax.set_ylabel("shift of B$\\to$A (flat sd)")
    axes[0, 0].legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    save(fig, "prior_shift")


def fig_p3_ridge():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    truth = {"A.div_AB": 0.212, "B.div_AB": 0.334, "A.div_BB": 0.396, "B.div_BB": 0.083}
    mles = {}
    for s, col in zip((0, 1, 2), (BLUE, ORANGE, AQUA), strict=True):
        run = XG2 / "P3" / f"n1024_s{s}"
        post = pd.read_csv(run / "posterior_samples.csv.gz")
        sub = post.sample(600, random_state=0)
        mles[s] = (json.loads((run / "mle.json").read_text())["mle"], col)
        for ax, (x, y) in zip(axes, (("A.div_AB", "B.div_AB"), ("A.div_BB", "B.div_BB")), strict=True):
            ax.scatter(sub[x], sub[y], s=3, color=col, alpha=0.35, lw=0,
                       label=f"dataset {s}")  # fmt: skip
    for ax, (x, y) in zip(axes, (("A.div_AB", "B.div_AB"), ("A.div_BB", "B.div_BB")), strict=True):
        for s, (m, col) in mles.items():
            ax.plot(m[x], m[y], "D", color=col, mec=INK, mew=0.8, ms=5.5,
                    label="MLE (per dataset)" if s == 0 else None)  # fmt: skip
        ax.plot(truth[x], truth[y], "*", color=INK, ms=7, label="truth")
        ax.set_xlabel(pname(x))
        ax.set_ylabel(pname(y))
    axes[0].set_title("mixed divisions: A$\\to$AB vs B$\\to$AB")
    axes[1].set_title("symmetric B divisions: A$\\to$BB vs B$\\to$BB")
    leg = axes[0].legend(loc="upper right", fontsize=6.5, markerscale=1.0)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    save(fig, "p3_ridge")


def fig_death_ridge():
    """The "which type dies" parent swap: A.death vs B.death draws for P4 and P2, n = 1024."""
    from make_swap_theory import OUT_A, OUT_B, mean_matrix, perron

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, gt in zip(axes, ("P4", "P2"), strict=True):
        z = np.load(XG2 / gt / "n1024_s0" / "reference.npz", allow_pickle=True)
        t = dict(zip([str(v) for v in z["truth_names"]], z["truth"].astype(float)))
        pA = np.array([t["A." + k] for k in OUT_A])
        pB = np.array([t["B." + k] for k in OUT_B])
        _, _, u = perron(mean_matrix(pA, pB))
        for s, col in zip((0, 1, 2), (BLUE, ORANGE, AQUA), strict=True):
            run = XG2 / gt / f"n1024_s{s}"
            post = pd.read_csv(run / "posterior_samples.csv.gz").sample(600, random_state=0)
            ax.scatter(post["A.death"], post["B.death"], s=3, color=col, alpha=0.35, lw=0,
                       label=f"dataset {s}")  # fmt: skip
            m = json.loads((run / "mle.json").read_text())["mle"]
            ax.plot(m["A.death"], m["B.death"], "D", color=col, mec=INK, mew=0.8, ms=5.5,
                    label="MLE (per dataset)" if s == 0 else None)  # fmt: skip
        # parent swap through the truth: u_A * A.death + u_B * B.death constant
        x = np.linspace(0, 0.45, 50)
        y = t["B.death"] - u[0] / u[1] * (x - t["A.death"])
        ok = (y >= 0) & (y <= 0.45)
        ax.plot(x[ok], y[ok], ls="--", color=INK2, lw=0.9, label="same average death rate")
        ax.plot(t["A.death"], t["B.death"], "*", color=INK, ms=8, label="truth")
        ax.set_xlabel(pname("A.death"))
        ax.set_ylabel(pname("B.death"))
        ax.set_xlim(0, 0.42)
        ax.set_ylim(0, 0.42)
        ax.set_title(f"{gt}, $n$ = 1024", fontsize=9)
    leg = axes[0].legend(loc="upper right", fontsize=6, markerscale=1.0)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    save(fig, "death_ridge")


def copy_note_figures():
    for name in ("profile_T1_ref_s0_n128", "profile_T2_toA05_s0_n1024", "decomposition",
                 "prior_alpha_and_bias"):  # fmt: skip
        src = NOTE / "figures" / f"{name}.png"
        if src.exists():
            FIG.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, FIG / f"{name}.png")
            print("copied", name)


# ------------------------------------------------------------------ tables
def tables(runs, flat, mle, meta):
    pct = lambda v: f"{100 * v:.0f}\\%"  # noqa: E731
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    f2 = lambda v: f"{v:.2f}"  # noqa: E731
    # B.to_A posterior mean (sd) and MLE by truth x n, mean over datasets
    b = flat[flat.parameter == "B.to_A"].groupby(["gt", "n"]).agg(
        truth=("truth", "first"), mean=("mean", "mean"), sd=("sd", "mean"))
    bm = mle[mle.parameter == "B.to_A"].groupby(["gt", "n"]).mle.mean()
    rows = []
    for gt in GTS:
        r = {"truth": LABEL[gt], "theta": f"{b.loc[(gt, 128), 'truth']:.2f}"}
        for n in NS:
            r[f"post{n}"] = f"{b.loc[(gt, n), 'mean']:.2f} ({b.loc[(gt, n), 'sd']:.2f})"
            r[f"mle{n}"] = f"{bm.loc[(gt, n)]:.2f}"
        rows.append(r)
    tex_table(pd.DataFrame(rows), "btoa")
    # errors: posterior mean vs MLE, B.to_A and all params
    e = mle.groupby(["gt", "n"])[["abs_err_post", "abs_err_mle"]].mean()
    eb = mle[mle.parameter == "B.to_A"].groupby(["gt", "n"])[["abs_err_post", "abs_err_mle"]].mean()
    rows = []
    for gt in GTS:
        r = {"truth": LABEL[gt]}
        for n in NS:
            pb, mb = eb.loc[(gt, n)]
            r[f"b{n}"] = (f"\\textbf{{{mb:.2f}}} / {pb:.2f}" if mb < pb
                          else f"{mb:.2f} / \\textbf{{{pb:.2f}}}")  # fmt: skip
        for n in (128, 1024):
            pa, ma = e.loc[(gt, n)]
            r[f"a{n}"] = (f"\\textbf{{{ma:.3f}}} / {pa:.3f}" if ma < pa
                          else f"{ma:.3f} / \\textbf{{{pa:.3f}}}")  # fmt: skip
        rows.append(r)
    tex_table(pd.DataFrame(rows), "errors")
    # calibration and contraction summary
    cov = flat[flat.truth > 0].groupby(["gt", "n"]).covered.mean()
    az = flat[flat.truth > 0].assign(az=lambda d: d.z.abs()).groupby(["gt", "n"]).az.mean()
    worst = flat.groupby(["gt", "seed", "n"]).contraction.min().groupby(["gt", "n"]).mean()
    rows = []
    for gt in GTS:
        r = {"truth": LABEL[gt]}
        for n in NS:
            r[f"c{n}"] = f"{worst.loc[(gt, n)]:.2f}"
        for n in (128, 1024):
            r[f"cov{n}"] = pct(cov.loc[(gt, n)])
            r[f"z{n}"] = f2(az.loc[(gt, n)])
        rows.append(r)
    tex_table(pd.DataFrame(rows), "calibration")
    # sampler and cost by truth and kernel
    rows = []
    for gt in GTS:
        for kernel in ("loop", "adjoint"):
            g = meta[(meta["gt"] == gt) & (meta.kernel == kernel)]
            if g.empty:
                continue
            rows.append({
                "truth": LABEL[gt], "kernel": kernel, "runs": str(len(g)),
                "hours": f"{g.hours.median():.2f} ({g.hours.min():.2f}--{g.hours.max():.2f})",
                "grad": f"{g.grad_evals.median() / 1e3:.0f}k", "ms": f"{g.ms_per_eval.median():.0f}",
                "depth": f"{g.depth.mean():.1f}", "rhat": f"{g.rhat.max():.2f}",
                "nbad": str(int((g.rhat > 1.05).sum())), "ess": f"{g.min_ess.min():.0f}",
                "div": f"{g.divergences.max():.0f}",
            })  # fmt: skip
    tex_table(pd.DataFrame(rows), "sampler")
    # prior reweighting reliability
    ps = runs[runs.source == "psis"].groupby(["gt", "seed", "n", "prior"]).pareto_k.first().reset_index()
    rel = (ps.assign(ok=ps.pareto_k <= 0.7).groupby(["prior", "n"]).ok.mean().unstack("n"))
    rel = rel.loc[["smooth", "jeffreys", "wrong", "central", "sparse"]]
    tex_table(rel.reset_index().assign(**{str(n): rel[n].map(pct).values for n in NS})
              [["prior"] + [str(n) for n in NS]], "psis")  # fmt: skip
    # bad runs list
    bad = meta[meta.rhat > 1.05].sort_values("rhat", ascending=False)
    tex_table(bad.assign(truth=bad["gt"].map(LABEL), rhat=bad.rhat.map(f2),
                         div=bad.divergences.astype(int).astype(str),
                         n=bad.n.astype(str), seed=bad.seed.astype(str))
              [["truth", "n", "seed", "rhat", "div", "kernel"]], "badruns")  # fmt: skip
    summary = {
        "runs": int(len(meta)), "kernel_adjoint": int((meta.kernel == "adjoint").sum()),
        "rhat_gt_105": int((meta.rhat > 1.05).sum()), "rhat_gt_110": int((meta.rhat > 1.10).sum()),
        "missed_optimum": int(mle.groupby(["gt", "seed", "n"]).missed_optimum.first().sum()),
        "ll_gap_truth_median_range": [float(v) for v in mle.groupby(["gt", "seed", "n"])
                                      .ll_gap_truth.first().quantile([0, 0.5, 1])],
        "psis_pairs_unreliable": int((ps.pareto_k > 0.7).sum()), "psis_pairs": int(len(ps)),
    }  # fmt: skip
    (TAB / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    import argparse

    global XG2
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results",
                    help="results root holding exact_gw/data_size (default: <repo>/results)")
    XG2 = ap.parse_args().results / "exact_gw" / "data_size"
    runs, flat, mle, index, meta = load()
    for f in (lambda: fig_btoa(flat, mle), lambda: fig_contraction(flat), lambda: fig_contraction_mle(flat, index),
              lambda: fig_error(mle), lambda: fig_rmse(flat, mle), lambda: mse_tables(flat, mle), lambda: fig_bvm(flat), lambda: fig_prior(runs),
              fig_p3_ridge, copy_note_figures, lambda: tables(runs, flat, mle, meta)):  # fmt: skip
        f()


if __name__ == "__main__":
    main()
