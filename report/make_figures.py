"""Build every figure of the BIO-SBI simulation report from ``results/``.

Usage (from the repository root)::

    uv run python report/make_figures.py

Writes vector PDFs to ``report/figures/`` and a few CSV tables of the plotted numbers
to ``report/tables/``. The script only *reads* ``results/``. Every figure is built
independently: if its runs are missing or the sweep is not complete yet, the figure is
skipped (a line is printed) and the LaTeX report shows a "result pending" box instead.
Re-run the script once a sweep has finished and rebuild the PDF to fill it in.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"

# --------------------------------------------------------------------------- style
# Reference palette of the dataviz method (validated categorical order, light mode).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)  # fmt: skip
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#1c5cab", "#6da7ec", "#f0efec", "#ef8a7a", "#b8302f"]
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 7.5,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,
    }
)
W_FULL, W_HALF = 6.3, 3.1  # inches (LaTeX text width ~ 16 cm)

# ------------------------------------------------------------------ data helpers
_spec = importlib.util.spec_from_file_location("summarize", ROOT / "scripts" / "summarize.py")
SUM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SUM)

_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}


def load(rel: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs and parameter rows below ``results/<rel>`` (with unit-aware errors)."""
    if rel not in _cache:
        root = RES / rel
        if not root.exists() or not any(root.rglob("metrics.json")):
            raise FileNotFoundError(f"no runs below {root}")
        runs, params = SUM.load(root)
        runs = SUM.add_unit_aware_errors(runs, params)
        meta = runs[["run", "ppc_fraction_outside_95", "posterior_support_mass"]].rename(
            columns={"ppc_fraction_outside_95": "ppc", "posterior_support_mass": "support"}
        ) if "posterior_support_mass" in runs else runs[["run", "ppc_fraction_outside_95"]].rename(
            columns={"ppc_fraction_outside_95": "ppc"})
        params = params.merge(meta, on="run", how="left")
        _cache[rel] = (runs, params)
    return _cache[rel]


def n_runs(rel: str) -> int:
    root = RES / rel
    return len(list(root.rglob("metrics.json"))) if root.exists() else 0


def require(rel: str, expected: int) -> None:
    """Raise unless ``results/<rel>`` holds at least ``expected`` finished runs."""
    n = n_runs(rel)
    if n < expected:
        raise FileNotFoundError(f"{rel}: {n}/{expected} runs finished")


def is_prob(par: str) -> bool:
    return not par.endswith("mean_lifetime")


def ov(frame: pd.DataFrame, key: str) -> pd.Series:
    col = f"override.{key}"
    return frame[col].astype(str) if col in frame else pd.Series("nan", index=frame.index)


def paired(params: pd.DataFrame, setting: pd.Series, control: str, pars) -> pd.DataFrame:
    """Per setting and parameter: mean posterior mean, shift from control (paired by
    seed), its standard error over seeds, mean posterior sd, coverage count, runs."""
    p = params.assign(setting=setting.values)
    p = p[p.parameter.isin(pars)]
    piv = p.pivot_table(index=["parameter", "seed"], columns="setting", values="posterior_mean")
    sd = p.groupby(["parameter", "setting"]).posterior_std.mean()
    cov = p.groupby(["parameter", "setting"]).covered.apply(
        lambda x: int(np.nansum(pd.to_numeric(x, errors="coerce"))))
    rows = []
    for par in pars:
        if par not in piv.index.get_level_values(0):
            continue
        t = piv.loc[par]
        for s in t.columns:
            d = t[s] - t[control] if control in t.columns else t[s] * np.nan
            rows.append(dict(parameter=par, setting=s, mean=t[s].mean(), shift=d.mean(),
                             se=d.std(ddof=1) / np.sqrt(d.count()) if d.count() > 1 else np.nan,
                             post_sd=sd.get((par, s), np.nan), covered=cov.get((par, s), np.nan),
                             n=int(t[s].count())))
    return pd.DataFrame(rows)


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def table(df: pd.DataFrame, name: str) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False, float_format="%.4g")


def truth_line(ax, y, label=None, horizontal=True):
    kw = dict(color=INK, lw=0.8, ls=(0, (4, 2)), zorder=1)
    (ax.axhline if horizontal else ax.axvline)(y, **kw)
    if label:
        ax.annotate(label, xy=(1, y), xycoords=("axes fraction", "data"), xytext=(2, 0),
                    textcoords="offset points", va="center", fontsize=6.5, color=INK2)


def zero_line(ax):
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)


def heat(ax, mat: pd.DataFrame, cmap, norm=None, fmt="{:.2f}", vmin=None, vmax=None, text=True):
    """Annotated heat map; near-white cells get dark text, dark cells white."""
    vals = mat.values.astype(float)
    im = ax.imshow(vals, cmap=cmap, norm=norm, vmin=None if norm else vmin,
                   vmax=None if norm else vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), [str(c) for c in mat.columns])
    ax.set_yticks(range(mat.shape[0]), [str(r) for r in mat.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if text:
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                rgba = im.cmap(im.norm(v))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.3,
                        color="white" if lum < 0.5 else INK)
    return im


PRETTY = {
    "A.death": "A death", "A.persist": "A persist", "A.to_B": "A→B", "A.div_AA": "A→AA",
    "A.div_AB": "A→AB", "A.div_BB": "A→BB", "B.death": "B death", "B.persist": "B persist",
    "B.to_A": "B→A", "B.div_AA": "B→AA", "B.div_AB": "B→AB", "B.div_BB": "B→BB",
    "A.mean_lifetime": "A lifetime (h)", "B.mean_lifetime": "B lifetime (h)",
    "founder.A": "founder A",
}
P_ROWS = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
          "B.death", "B.persist", "B.to_A"]


# ======================================================================= figures
def fig_e0_bto_a():
    runs, params = load("b_row")
    p = params[params.parameter == "B.to_A"].drop(columns=["simulations", "rounds"],
                                                   errors="ignore")
    p = p.merge(runs[["run", "simulations", "rounds"]], on="run")
    arm = []
    for _, r in p.iterrows():
        sims = f"{int(r.simulations) // 1000}k"
        if r.sweep == "two_type":
            arm.append(("all free", sims))
        elif r.sweep == "reversible_ab":
            arm.append(("B div. fixed", sims))
        elif r.sweep == "two_type_snpe":
            arm.append(("all free, SNPE", "4×5k"))
        else:
            arm.append(("all free, flat prior", sims))
    p["arm"], p["budget"] = zip(*arm)
    order = [("all free", "5k"), ("all free", "20k"), ("all free", "50k"),
             ("all free, SNPE", "4×5k"), ("all free, flat prior", "20k"),
             ("B div. fixed", "5k"), ("B div. fixed", "20k"), ("B div. fixed", "50k")]
    color = {"all free": ORANGE, "all free, SNPE": ORANGE, "all free, flat prior": AQUA,
             "B div. fixed": BLUE}
    marker = {"all free": "o", "all free, SNPE": "s", "all free, flat prior": "D",
              "B div. fixed": "o"}
    fig, ax = plt.subplots(figsize=(W_FULL, 2.3))
    for i, key in enumerate(order):
        sub = p[(p.arm == key[0]) & (p.budget == key[1])].sort_values("seed")
        xs = i + np.linspace(-0.22, 0.22, len(sub)) if len(sub) else []
        ax.errorbar(xs, sub.posterior_mean, yerr=sub.posterior_std, fmt=marker[key[0]],
                    color=color[key[0]], ms=4, elinewidth=0.8, capsize=0,
                    mec="white", mew=0.5)
    truth_line(ax, 0.8, "truth 0.80")
    ax.axhline(1 / 6, color=MUTED, lw=0.8, ls=":")
    ax.annotate("prior mean, uniform Dirichlet", xy=(-0.45, 1 / 6), xytext=(0, -8),
                textcoords="offset points", fontsize=6.3, color=MUTED, ha="left")
    short = {"all free": "", "all free, SNPE": "SNPE\n", "all free, flat prior": "flat prior\n",
             "B div. fixed": ""}
    ax.set_xticks(range(len(order)), [f"{short[a]}{b}" for a, b in order])
    ax.set_xlabel("training budget (simulations)")
    ax.set_ylabel("B→A posterior mean ± sd")
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(4.5, color=AXIS, lw=0.6)
    ax.text(2, 1.0, "B row free (10 parameters)", ha="center", fontsize=7, color=INK2)
    ax.text(6, 1.0, "B divisions fixed to 0 (7)", ha="center", fontsize=7, color=INK2)
    save(fig, "e0_bto_a")
    table(p.groupby(["arm", "budget"]).agg(mean=("posterior_mean", "mean"),
                                           sd=("posterior_std", "mean"),
                                           contraction=("contraction", "mean")).reset_index(),
          "e0_bto_a")


def fig_e0_c2st():
    d = pd.read_csv(RES / "b_row" / "distinguish.csv")
    d["label"] = [f"{a:.2f} ({t:.2f})" for a, t in zip(d.a, d["B.to_A"], strict=True)]
    mat = d.pivot_table(index="label", columns="clones", values="c2st_accuracy", sort=False)
    fig, ax = plt.subplots(figsize=(W_HALF, 1.9))
    im = heat(ax, mat, SEQ, vmin=0.5, vmax=1.0)
    ax.set_xlabel("clones per endpoint (3 endpoints)")
    ax.set_ylabel("shift a (B→A)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("C2ST accuracy", fontsize=6.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=0)
    save(fig, "e0_c2st")


def fig_design():
    runs = pd.read_csv(RES / "design_comparison" / "design_runs.csv")
    pars = pd.read_csv(RES / "design_comparison" / "design_parameters.csv")
    runs["design"] = runs.design.str.replace(" (b_row)", "", regex=False)
    pars["design"] = pars.design.str.replace(" (b_row)", "", regex=False)
    order = ["single_128", "three_32", "single_384", "three_128"]
    labels = {"single_128": "1×128\n(128)", "three_32": "3×32\n(96)",
              "single_384": "1×384\n(384)", "three_128": "3×128\n(384)"}
    laws = [("reversible_ab", "B div. fixed", BLUE), ("two_type", "all free", ORANGE)]
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.2),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.3], wspace=0.5))
    for k, (col, lab) in enumerate([("mae", "MAE (probabilities)"),
                                    ("con", "mean contraction")]):
        ax = axes[k]
        for j, (law, name, c) in enumerate(laws):
            g = runs[runs.law == law].groupby("design")[col]
            m, s = g.mean().reindex(order), g.std().reindex(order)
            x = np.arange(len(order)) + (j - 0.5) * 0.36
            ax.bar(x, m, width=0.34, color=c, label=name, edgecolor="white", lw=0.5)
            ax.errorbar(x, m, yerr=s, fmt="none", ecolor=INK2, elinewidth=0.7)
        ax.set_xticks(range(len(order)), [labels[o] for o in order], fontsize=6.2)
        ax.set_title(lab)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right", fontsize=6.3)
    rev = pars[pars.law == "reversible_ab"]
    mat = rev.pivot_table(index="design", columns="parameter", values="contraction")
    mat = mat.reindex(order)[[c for c in P_ROWS if c in mat.columns]].T
    mat.columns = [labels[o].split("\n")[0] for o in order]
    mat.index = [PRETTY[c] for c in mat.index]
    heat(axes[2], mat, SEQ, vmin=0.3, vmax=1.0, fmt="{:.2f}")
    axes[2].set_title("contraction (B div. fixed)")
    axes[2].tick_params(axis="x", labelsize=6.3)
    save(fig, "design_comparison")


def fig_gpu():
    runs, _ = load("gpu_bench")
    runs = runs.copy()
    runs["device"] = ov(runs, "device").replace("cuda", "GPU").replace("cpu", "CPU")
    runs["batch"] = ov(runs, "inference.method.training.training_batch_size")
    runs["mode"] = np.where(runs.rounds > 1, "sequential (4 rounds)", "amortised")
    runs["label"] = runs["mode"] + ", " + runs.device + ", batch " + runs.batch
    runs = runs.sort_values(["mode", "device", "batch"])
    fig, ax = plt.subplots(figsize=(W_FULL, 1.8))
    y = np.arange(len(runs))
    ax.barh(y, runs.training_seconds, color=BLUE, height=0.6, label="training")
    ax.barh(y, runs.total_seconds - runs.training_seconds, left=runs.training_seconds,
            color=AXIS, height=0.6, label="simulation + SBC + PPC")
    for yi, t in zip(y, runs.total_seconds, strict=True):
        ax.text(t + 8, yi, f"{t:.0f} s", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, runs.label)
    ax.invert_yaxis()
    ax.set_xlabel("wall time (s), one job at a time")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1080)
    ax.legend(loc="upper right")
    save(fig, "gpu_bench")
    table(runs[["label", "total_seconds", "training_seconds", "mae_probabilities",
                "mean_contraction", "sbc_mean_abs_coverage_error"]], "gpu_bench")


# ------------------------------------------------------------------- E4
HSET_BH = {"[240]": "d10", "[144,240]": "d6,10", "[144,240,336]": "d6,10,14",
           "[96,144,240]": "d4,6,10"}
HSET_GW = {"[10]": "g10", "[6,10]": "g6,10", "[6,10,14]": "g6,10,14", "[4,6,10]": "g4,6,10"}


def e4_frames(rel):
    runs, params = load(rel)
    p = params[params.inferred.astype(bool)].copy()
    p["clones"] = ov(p, "design.clones").astype(int)
    p["hset"] = ov(p, "design.horizons").str.replace(" ", "")
    return runs, p


def fig_e4(rel="e4/bh", name="e4_bh", hmap=HSET_BH, expected=120):
    require(rel, expected)
    _, p = e4_frames(rel)
    p["hset"] = p.hset.map(hmap)
    pr = p[p.parameter.map(is_prob)]
    per_run = pr.groupby(["run", "clones", "hset"]).agg(con=("contraction", "mean"),
                                                       mae=("abs_error", "mean")).reset_index()
    g = per_run.groupby(["hset", "clones"])[["con", "mae"]].mean().reset_index()
    order = list(hmap.values())
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 1.9), gridspec_kw=dict(wspace=0.35))
    for ax, col, lab, cmap, fmt in [(axes[0], "con", "mean contraction (probabilities)", SEQ,
                                     "{:.2f}"),
                                    (axes[1], "mae", "MAE (probabilities)", SEQ.reversed(),
                                     "{:.3f}")]:
        mat = g.pivot(index="hset", columns="clones", values=col).reindex(order)
        heat(ax, mat, cmap, fmt=fmt)
        ax.set_xlabel("clones per endpoint")
        ax.set_title(lab)
    axes[0].set_ylabel("endpoints")
    save(fig, f"{name}_heat")
    table(g, f"{name}_grid")
    # selected parameters vs clones, one line per endpoint set
    sel = [("A.div_AA", 0.55), ("B.to_A", 0.80)]
    if (p.parameter == "A.mean_lifetime").any():
        sel.append(("A.mean_lifetime", 20.0))
    else:
        sel.append(("A.div_AB", 0.05))
    fig, axes = plt.subplots(1, len(sel), figsize=(W_FULL, 2.0), gridspec_kw=dict(wspace=0.4))
    for ax, (par, tv) in zip(axes, sel, strict=True):
        s = p[p.parameter == par].groupby(["hset", "clones"]).posterior_mean.agg(
            ["mean", "std", "count"]).reset_index()
        for k, h in enumerate(order):
            t = s[s.hset == h].sort_values("clones")
            ax.errorbar(t.clones * (1 + 0.04 * (k - 1.5)), t["mean"],
                        yerr=t["std"] / np.sqrt(t["count"]), color=CAT[k], marker="o", ms=3,
                        lw=1.2, elinewidth=0.7, label=h)
        truth_line(ax, tv)
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 32, 64, 128, 256, 512], ["16", "32", "64", "128", "256", "512"])
        ax.minorticks_off()
        ax.set_xlabel("clones per endpoint")
        ax.set_title(f"{PRETTY[par]} (truth {tv:g})")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=6.5,
               title="endpoints:", title_fontsize=6.5, alignment="left")
    fig.legends[-1]._legend_box.align = "left"
    save(fig, f"{name}_params")


def fig_e4_gw():
    fig_e4("e4/gw", "e4_gw", HSET_GW, 120)


# ------------------------------------------------------------------- ridge
def _samples(run_dirs):
    out = []
    for d in run_dirs:
        f = Path(d) / "posterior_samples.csv.gz"
        if f.exists():
            s = pd.read_csv(f)
            s["run"] = str(d)
            out.append(s)
    return pd.concat(out, ignore_index=True)


def fig_ridge():
    ctrl = sorted((RES / "e3b").glob("*lifetime_shape=8,*"))
    fixed = sorted((RES / "bh_ridge" / "lifetimes_fixed").glob("*seed=*"))
    flat = sorted((RES / "bh_ridge" / "flat_div_AA").glob("*seed=*"))
    if not (ctrl and fixed and flat):
        raise FileNotFoundError("ridge runs missing")
    arms = [("control (uniform prior, lifetimes free)", ctrl, BLUE),
            ("lifetimes fixed to truth", fixed, ORANGE),
            ("flat prior on A→AA", flat, AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.1), gridspec_kw=dict(wspace=0.38))
    bins = np.linspace(0, 1, 41)
    xs = np.linspace(0, 1, 200)
    for name, dirs, c in arms:
        s = _samples(dirs)
        axes[0].hist(s["A.div_AA"], bins=bins, density=True, histtype="step", color=c, lw=1.3,
                     label=name.split(" (")[0])
        if name.startswith("control") or name.startswith("flat"):
            sub = s.groupby("run").head(300)
            axes[1].scatter(sub["A.div_AA"], sub["A.mean_lifetime"], s=1.5, color=c,
                            alpha=0.35, lw=0, rasterized=True)
        means = s.groupby("run")[["A.div_AA", "A.mean_lifetime"]].mean()
        sds = s.groupby("run")["A.div_AA"].std()
        k = [a[0] for a in arms].index(name)
        axes[2].errorbar(k + np.linspace(-0.2, 0.2, len(means)), means["A.div_AA"], yerr=sds,
                         fmt="o", color=c, ms=3.5, elinewidth=0.8, mec="white", mew=0.4)
    axes[0].plot(xs, 5 * (1 - xs) ** 4, color=MUTED, lw=0.8, ls=":", label="uniform prior")
    axes[0].axvline(0.55, color=INK, lw=0.8, ls=(0, (4, 2)))
    axes[0].set_xlabel("A→AA (self-renewal)")
    axes[0].set_ylabel("posterior density (5 seeds pooled)")
    axes[0].set_ylim(0, 9)
    axes[0].legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    axes[0].set_title("a  A→AA posterior")
    axes[1].scatter([0.55], [20], marker="*", s=60, color=INK, zorder=5)
    axes[1].set_xlabel("A→AA")
    axes[1].set_ylabel("A lifetime (h)")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("b  joint posterior")
    axes[1].annotate("truth", (0.55, 20), xytext=(6, 6), textcoords="offset points",
                     fontsize=6.3)
    truth_line(axes[2], 0.55)
    axes[2].set_xticks(range(3), ["control", "lifetimes\nfixed", "flat prior"])
    axes[2].set_ylabel("A→AA, mean ± sd per seed")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("c  per seed")
    save(fig, "bh_ridge")


# ------------------------------------------------------------------- E1b
def fig_e1b():
    rows = []
    for sim in ("bh", "gw"):
        for m in sorted((RES / "e1b" / sim).glob("*/metrics.json")):
            j = json.loads(m.read_text())
            truth = {r["parameter"]: r["truth"] for r in j["recovery"]["per_parameter"]}
            for r in j["reference_detection"]:
                v = r["detection_probability"] if sim == "gw" else r["detection_rate"]
                rows.append(dict(sim=sim, A=truth["A.death"], B=truth["B.death"],
                                 h=r["horizon"], v=v))
    d = pd.DataFrame(rows)
    if len(d) < 80:
        raise FileNotFoundError("E1b incomplete")
    panels = [("bh", 144, "BH, 144 h"), ("bh", 240, "BH, 240 h"),
              ("gw", 6, "GW, gen. 6 (exact)"), ("gw", 10, "GW, gen. 10 (exact)")]
    fig, axes = plt.subplots(1, 4, figsize=(W_FULL, 1.75), gridspec_kw=dict(wspace=0.15))
    for k, (ax, (sim, h, title)) in enumerate(zip(axes, panels, strict=True)):
        mat = d[(d.sim == sim) & (d.h == h)].pivot_table(index="A", columns="B", values="v")
        heat(ax, mat, SEQ.reversed(), vmin=0.25, vmax=1.0)
        i, j = list(mat.index).index(0.03), list(mat.columns).index(0.1)
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
        ax.set_title(title, fontsize=6.8)
        ax.set_xlabel("B death")
        if k == 0:
            ax.set_ylabel("A death")
        else:
            ax.set_yticklabels([])
    save(fig, "e1b_detection")
    table(d, "e1b_detection")


# ------------------------------------------------------------------- E1a
def _founder_f(series):
    return series.str.extract(r"A:([0-9.]+)")[0].astype(float)


def fig_e1a_type():
    require("e1a/founder_type", 80)
    runs, params = load("e1a/founder_type")
    params = params.copy()
    params["f"] = _founder_f(ov(params, "data.law.fixed.founder")).round(2).astype(str)
    sel = ["A.div_AA", "A.div_AB", "B.to_A", "B.mean_lifetime"]
    arms = [("fixed_bh", "BH, founder fixed to A", BLUE, "o", "-"),
            ("inferred_bh", "BH, founder inferred", BLUE, "s", "--"),
            ("fixed_gw", "GW, founder fixed to A", ORANGE, "o", "-"),
            ("inferred_gw", "GW, founder inferred", ORANGE, "s", "--")]
    fig, axes = plt.subplots(1, 5, figsize=(W_FULL, 1.95), gridspec_kw=dict(wspace=0.7))
    allrows = []
    for sweep, lab, c, mk, ls in arms:
        sub = params[params.sweep == sweep]
        if sub.empty:
            continue
        t = paired(sub, sub.f, "1.0", sel + ["founder.A"])
        t["arm"] = sweep
        allrows.append(t)
        for ax, par in zip(axes[:4], sel, strict=True):
            u = t[t.parameter == par].copy()
            if u.empty:
                continue
            u["x"] = 1 - u.setting.astype(float)
            u = u.sort_values("x")
            ax.errorbar(u.x, u["shift"], yerr=u.se, color=c, marker=mk, ls=ls, ms=3.2,
                        elinewidth=0.7, lw=1.1, label=lab, mec="white", mew=0.4)
        u = t[t.parameter == "founder.A"].copy()
        if not u.empty and sweep.startswith("inferred"):
            f = u.setting.astype(float)
            axes[4].errorbar(f, u["mean"], yerr=u.post_sd, color=c, marker=mk, ls=ls, ms=3.2,
                             elinewidth=0.7, lw=1.1)
    for ax, par in zip(axes[:4], sel, strict=True):
        zero_line(ax)
        ax.set_title(f"Δ {PRETTY[par]}", fontsize=7.2)
        ax.set_xticks([0, 0.1, 0.3, 0.5], ["0", ".1", ".3", ".5"])
    fig.text(0.42, -0.04, "share of B founders in the data, 1 − f", ha="center", color=INK2)
    axes[0].set_ylabel("shift vs f = 1 (± paired SE)")
    axes[4].plot([0.45, 1.05], [0.45, 1.05], color=INK, lw=0.8, ls=(0, (4, 2)))
    axes[4].set_xlabel("true f")
    axes[4].set_ylabel("posterior mean ± sd")
    axes[4].set_title("founder A (inferred)", fontsize=7.2)
    axes[4].set_xlim(0.4, 1.08)
    axes[4].set_ylim(0, 1.1)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1), fontsize=6.5)
    save(fig, "e1a_founder_type")
    table(pd.concat(allrows), "e1a_founder_type_shifts")


def _p1(series):
    return series.str.extract(r"\{1:([0-9.]+)")[0].astype(float)


def fig_e1a_number(rel="e1a/founder_number_bh", name="e1a_founder_number_bh", expected=30):
    require(rel, expected)
    _, params = load(rel)
    params = params.copy()
    params["p1"] = _p1(ov(params, "data.law.reference.founder_size")).astype(str)
    params["arm"] = ov(params, "law.founder_sizes").str.replace(" ", "")
    if (params.parameter == "A.mean_lifetime").any():
        sel = ["A.div_AB", "B.death", "B.to_A", "A.mean_lifetime"]
    else:  # GW: the largest shifts are in A's self-renewal and B's row
        sel = ["A.div_AA", "A.div_AB", "B.persist", "B.to_A"]
    fig, axes = plt.subplots(1, len(sel), figsize=(W_FULL, 1.9), gridspec_kw=dict(wspace=0.5))
    rows = []
    for arm, c, mk, lab in [("[1]", BLUE, "o", "inference: 1 founder cell"),
                            ("[1,2,3]", ORANGE, "s", "inference: founder size inferred")]:
        sub = params[params.arm == arm]
        t = paired(sub, sub.p1, "1.0", sel + ["founder_size.1"])
        t["arm"] = arm
        rows.append(t)
        for ax, par in zip(axes, sel, strict=True):
            u = t[t.parameter == par].copy()
            u["x"] = 1 - u.setting.astype(float)
            u = u.sort_values("x")
            ax.errorbar(u.x, u["shift"], yerr=u.se, color=c, marker=mk, ms=3.2, lw=1.1,
                        elinewidth=0.7, label=lab, mec="white", mew=0.4)
    for ax, par in zip(axes, sel, strict=True):
        zero_line(ax)
        ax.set_title(f"Δ {PRETTY[par]}", fontsize=7.2)
        ax.set_xlabel("multi-cell share 1−p1")
        ax.set_xticks([0, 0.15, 0.4])
    axes[0].set_ylabel("shift vs p1 = 1 (± paired SE)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1), fontsize=6.5)
    save(fig, name)
    table(pd.concat(rows), name)


def fig_e1a_number_gw():
    fig_e1a_number("e1a/founder_number_gw", "e1a_founder_number_gw", 30)


# ------------------------------------------------------------------- E1c
def fig_e1c(rel="e1c/bh", name="e1c_bh", expected=45):
    require(rel, expected)
    runs, params = load(rel)
    params = params.copy()
    setting = ov(params, "law.fixed.A.death") + " / " + ov(params, "law.fixed.B.death")
    pars = [c for c in P_ROWS if c not in ("A.death", "B.death")]
    if (params.parameter == "A.mean_lifetime").any():
        pars += ["A.mean_lifetime", "B.mean_lifetime"]
    t = paired(params, setting, "0.03 / 0.1", pars)
    runs = runs.copy()
    runs["setting"] = ov(runs, "law.fixed.A.death") + " / " + ov(runs, "law.fixed.B.death")
    rs = runs.groupby("setting").agg(ppc=("ppc_fraction_outside_95", "mean"),
                                     support=("posterior_support_mass", "mean"),
                                     support_min=("posterior_support_mass", "min")).reset_index()
    order = sorted(rs.setting, key=lambda s: tuple(float(x) for x in s.split(" / ")))
    probs = [p for p in pars if is_prob(p)]
    mat = t[t.parameter.isin(probs)].pivot(index="setting", columns="parameter", values="shift")
    mat = mat.reindex(order)[probs]
    # settings whose posterior lies (almost) entirely outside the prior support are not
    # estimates: no shift and no PPC shown (support mass < 0.02)
    no_est = rs.set_index("setting").support.reindex(order) < 0.02
    mat.loc[no_est.values] = np.nan
    mat.columns = [PRETTY[c] for c in mat.columns]
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 0.25 * len(order) + 1.0),
                             gridspec_kw=dict(width_ratios=[3.2, 1, 1], wspace=0.08))
    lim = max(0.2, np.nanmax(np.abs(mat.values)))
    heat(axes[0], mat, DIV, norm=TwoSlopeNorm(0, -lim, lim), fmt="{:+.2f}")
    axes[0].set_ylabel("fixed A death / B death")
    axes[0].set_title("shift of posterior mean vs control (0.03 / 0.1)", fontsize=7.2)
    axes[0].tick_params(axis="x", labelrotation=45)
    for i, bad in enumerate(no_est.values):
        if bad:
            axes[0].text((len(probs) - 1) / 2, i, "no estimate (support mass ≈ 0)",
                         ha="center", va="center", fontsize=6.2, color=INK2)
    y = np.arange(len(order))
    rs = rs.set_index("setting").reindex(order)
    rs.loc[no_est.values, "ppc"] = np.nan
    for ax, col, lab, c in [(axes[1], "ppc", "PPC outside 95%", ORANGE),
                            (axes[2], "support", "support mass", BLUE)]:
        ax.barh(y, rs[col].fillna(0), color=c, height=0.6)
        for yi, v in zip(y, rs[col], strict=True):
            txt = "n/e" if np.isnan(v) else (f"{v:.2f}" if col == "support" else f"{v:.3f}")
            ax.text((0 if np.isnan(v) else v) + 0.01, yi, txt, va="center", fontsize=6)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks([])
        ax.set_title(lab, fontsize=7.2)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
    axes[1].set_xlim(0, max(0.15, np.nanmax(rs.ppc.values) * 1.4))
    axes[2].set_xlim(0, 1.2)
    save(fig, name)
    table(t.merge(rs.reset_index(), on="setting", how="left"), name)


def fig_e1c_gw():
    fig_e1c("e1c/gw", "e1c_gw", 80)


# ------------------------------------------------------------------- E2a/b
def fig_e2(arms=(("known_bh", "known (modelled)"), ("ignored_bh", "ignored")), name="e2_bh",
           expected=60):
    total = sum(n_runs(f"e2/{a}") for a, _ in arms)
    if total < expected:
        raise FileNotFoundError(f"E2 {name}: {total}/{expected} runs")
    fig, axes = plt.subplots(1, 4, figsize=(W_FULL, 1.95), gridspec_kw=dict(wspace=0.5))
    rows = []
    for (sweep, lab), c, mk in zip(arms, [BLUE, ORANGE], ["s", "o"], strict=True):
        _, params = load(f"e2/{sweep}")
        e = ov(params, "data.observation.confusion")
        e = e.where(e != "nan", ov(params, "observation.confusion"))
        sel = ["A.div_AA", "A.div_AB", "B.to_A"]
        t = paired(params, e, "0", sel)
        con = params.assign(e=e.values).groupby(["parameter", "e"]).contraction.mean()
        t["contraction"] = [con.get((p, s), np.nan) for p, s in zip(t.parameter, t.setting,
                                                                    strict=True)]
        t["arm"] = sweep
        rows.append(t)
        for ax, par in zip(axes[:3], sel, strict=True):
            u = t[t.parameter == par].copy()
            u["x"] = u.setting.astype(float)
            u = u.sort_values("x")
            ax.errorbar(u.x, u["shift"], yerr=u.se, color=c, marker=mk, ms=3.2, lw=1.1,
                        elinewidth=0.7, label=f"error {lab}", mec="white", mew=0.4)
        u = t[t.parameter == "A.div_AB"].copy()
        u["x"] = u.setting.astype(float)
        u = u.sort_values("x")
        axes[3].plot(u.x, u.contraction, color=c, marker=mk, ms=3.2, lw=1.1)
    for ax, par in zip(axes[:3], ["A.div_AA", "A.div_AB", "B.to_A"], strict=True):
        zero_line(ax)
        ax.set_title(f"Δ {PRETTY[par]}", fontsize=7.2)
        ax.set_xlabel("mis-typing rate e")
    axes[0].set_ylabel("shift vs e = 0 (± paired SE)")
    zero_line(axes[3])
    axes[3].set_title("contraction A→AB", fontsize=7.2)
    axes[3].set_xlabel("mis-typing rate e")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1), fontsize=6.5)
    save(fig, name)
    table(pd.concat(rows), name)


def fig_e2_gw():
    known = sorted(p.name for p in (RES / "e2").glob("known*gw*"))
    ign = sorted(p.name for p in (RES / "e2").glob("ignored*gw*"))
    if not (known and ign):
        raise FileNotFoundError("E2 GW folders not present")
    fig_e2(((known[0], "known (modelled)"), (ign[0], "ignored")), "e2_gw", 60)


# ------------------------------------------------------------------- E2c
def _e2c_frame(rel):
    runs, params = load(rel)
    cfgq, frac = [], []
    import yaml  # noqa: PLC0415

    for r in params.run:
        cfg = yaml.safe_load((Path(r) / ".hydra" / "config.yaml").read_text())
        cfgq.append(float(cfg["data"]["observation"]["q"]))
        to_c = cfg["data"]["law"]["reference"]["A"]["to_C"]
        frac.append("≈5% C" if abs(to_c - 0.005) < 1e-6 else "≈15% C")
    return runs, params.assign(q=cfgq, frac=frac)


def fig_e2c_ignore():
    require("e2c/ignore_c", 30)
    runs, p = _e2c_frame("e2c/ignore_c")
    p = p[p.inferred.astype(bool)]
    sel = ["A.div_AA", "A.death", "A.persist", "A.to_B", "B.persist", "B.to_A"]
    fig, axes = plt.subplots(1, 4, figsize=(W_FULL, 2.2),
                             gridspec_kw=dict(width_ratios=[1.6, 1.6, 0.9, 0.9], wspace=0.55))
    groups = [("≈5% C", 0.0), ("≈5% C", 0.5), ("≈5% C", 1.0),
              ("≈15% C", 0.0), ("≈15% C", 0.5), ("≈15% C", 1.0)]
    colors = {0.0: BLUE, 0.5: AQUA, 1.0: ORANGE}
    for ax, frac in zip(axes[:2], ["≈5% C", "≈15% C"], strict=True):
        for k, par in enumerate(sel):
            sub = p[(p.frac == frac) & (p.parameter == par)]
            tv = sub.truth.iloc[0]
            ax.plot([k - 0.35, k + 0.35], [tv, tv], color=INK, lw=1.0, ls=(0, (3, 1.5)))
            for j, q in enumerate([0.0, 0.5, 1.0]):
                s = sub[sub.q == q]
                x = k + (j - 1) * 0.22
                ax.errorbar(x, s.posterior_mean.mean(), yerr=s.posterior_mean.std(),
                            fmt="o", color=colors[q], ms=3.5, elinewidth=0.8, mec="white",
                            mew=0.4, label=f"q = {q:g}" if k == 0 else None)
        ax.set_xticks(range(len(sel)), [PRETTY[s] for s in sel], rotation=45, ha="right")
        ax.set_title(f"{frac} of cells", fontsize=7.2)
        ax.set_ylim(-0.03, 1.0)
    axes[0].set_ylabel("posterior mean (± sd over seeds)")
    axes[0].legend(fontsize=6.2, loc="upper right", title="C read as A w.p.",
                   title_fontsize=6.2)
    r = runs.copy()
    q_of = p.groupby("run").q.first()
    f_of = p.groupby("run").frac.first()
    r["q"], r["frac"] = r.run.map(q_of), r.run.map(f_of)
    for ax, col, lab in [(axes[2], "ppc_fraction_outside_95", "PPC outside 95%"),
                         (axes[3], "posterior_support_mass", "support mass")]:
        for i, (fr, q) in enumerate(groups):
            v = r[(r.frac == fr) & (r.q == q)][col]
            ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v,
                       color=colors[q], s=8, lw=0)
        ax.set_xticks(range(6), ["0", ".5", "1"] * 2, fontsize=6)
        ax.set_xlabel("q  (5% | 15% C)")
        ax.axvline(2.5, color=AXIS, lw=0.6)
        ax.set_title(lab, fontsize=7.2)
    axes[2].axhspan(0, 0.03, color=GRID, lw=0)

    axes[3].axhspan(0.5, 0.9, color=GRID, lw=0)
    axes[3].set_ylim(-0.02, 1)
    save(fig, "e2c_ignore_c")
    agg = p[p.parameter.isin(sel)].groupby(["frac", "q", "parameter"]).agg(
        truth=("truth", "first"), mean=("posterior_mean", "mean"),
        sd_seeds=("posterior_mean", "std"), post_sd=("posterior_std", "mean")).reset_index()
    table(agg, "e2c_ignore_c")


def fig_e2c_model(arm="model_c_free", name="e2c_model_c_free"):
    require(f"e2c/{arm}", 30)
    runs, p = _e2c_frame(f"e2c/{arm}")
    p = p[p.inferred.astype(bool) & p.parameter.str.match(r"^[AB]\.")]
    pars = [x for x in ["A.death", "A.persist", "A.div_AA", "A.div_AC", "A.to_C", "B.death",
                        "B.persist"] if x in set(p.parameter)]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.1), gridspec_kw=dict(wspace=0.3))
    colors = {0.0: BLUE, 0.5: AQUA, 1.0: ORANGE}
    for ax, frac in zip(axes, ["≈5% C", "≈15% C"], strict=True):
        for k, par in enumerate(pars):
            sub = p[(p.frac == frac) & (p.parameter == par)]
            if sub.empty:
                continue
            tv = sub.truth.iloc[0]
            ax.plot([k - 0.35, k + 0.35], [tv, tv], color=INK, lw=1.0, ls=(0, (3, 1.5)))
            for j, q in enumerate([0.0, 0.5, 1.0]):
                s = sub[sub.q == q]
                ax.errorbar(k + (j - 1) * 0.22, s.posterior_mean.mean(),
                            yerr=s.posterior_mean.std(), fmt="o", color=colors[q], ms=3.5,
                            elinewidth=0.8, label=f"q = {q:g}" if k == 0 else None)
        ax.set_xticks(range(len(pars)), [p_.replace("_", " ") for p_ in pars], rotation=45,
                      ha="right")
        ax.set_title(frac)
    axes[0].legend(fontsize=6.2)
    save(fig, name)


def fig_e2c_traj():
    fig_e2c_model("model_c_traj", "e2c_model_c_traj")


def fig_e2c_arms():
    """Three E2c inference arms on the same data: support mass, PPC, A->AA per run and
    the prior mass above the true A->AA under each inference model."""
    arms = [("ignore_c", "ignore\nC"), ("model_c_free", "C\nfree"),
            ("model_c_traj", "C\ntraj.")]
    for a, _ in arms:
        require(f"e2c/{a}", 30)
    fig, axes = plt.subplots(1, 4, figsize=(W_FULL, 2.3),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9], wspace=0.55))
    colors = {"≈5% C": BLUE, "≈15% C": ORANGE}
    rows = []
    for k, (arm, lab) in enumerate(arms):
        runs, p = _e2c_frame(f"e2c/{arm}")
        frac = p.groupby("run").frac.first()
        r = runs.assign(frac=runs.run.map(frac))
        aa = p[p.parameter == "A.div_AA"].set_index("run")
        r["div_AA"] = r.run.map(aa.posterior_mean)
        r["div_AA_truth"] = r.run.map(aa.truth)
        r["arm"] = arm
        rows.append(r[["arm", "run", "frac", "posterior_support_mass", "posterior_samples_drawn",
                       "ppc_fraction_outside_95", "div_AA", "div_AA_truth"]])
        for j, fr in enumerate(["≈5% C", "≈15% C"]):
            u = r[r.frac == fr]
            x = k + (j - 0.5) * 0.3 + np.linspace(-0.08, 0.08, len(u))
            est = u.posterior_support_mass >= 0.02
            axes[0].scatter(x, u.posterior_support_mass, s=9, color=colors[fr], lw=0,
                            label=fr if k == 0 else None)
            axes[1].scatter(x, u.ppc_fraction_outside_95, s=9, color=colors[fr], lw=0)
            axes[2].scatter(x[est.values], u.div_AA[est], s=9, color=colors[fr], lw=0)
            axes[2].scatter(x[~est.values], u.div_AA[~est], s=9, facecolor="none",
                            edgecolor=colors[fr], lw=0.6)
    for ax, title in zip(axes[:3], ["support mass", "PPC outside 95%", "A→AA posterior mean"],
                         strict=True):
        ax.set_xticks(range(3), [lab for _, lab in arms], fontsize=6.2)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(title, fontsize=7.2)
    axes[0].axhspan(0.5, 0.9, color=GRID, lw=0, zorder=0)
    axes[0].set_ylim(-0.02, 1)
    axes[0].legend(fontsize=6, loc="upper right", handletextpad=0.2)
    axes[1].axhspan(0, 0.03, color=GRID, lw=0, zorder=0)
    axes[1].set_ylim(-0.02, 1)
    for tv, fr in [(0.905, "≈5% C"), (0.7925, "≈15% C")]:
        axes[2].axhline(tv, color=colors[fr], lw=0.8, ls=(0, (3, 1.5)))
    axes[2].set_ylim(0, 1)
    # prior mass above the truth: A row has 6 outcomes (two-type) or 10 (three-type)
    tail = pd.DataFrame([dict(model=m, frac=fr, p=(1 - tv) ** (k - 1))
                         for m, k in [("two-type", 6), ("three-type", 10)]
                         for fr, tv in [("≈5% C", 0.905), ("≈15% C", 0.7925)]])
    for j, fr in enumerate(["≈5% C", "≈15% C"]):
        u = tail[tail.frac == fr]
        axes[3].bar(np.arange(2) + (j - 0.5) * 0.35, u.p, width=0.33, color=colors[fr])
    axes[3].set_yscale("log")
    axes[3].set_ylim(1e-10, 1)
    axes[3].set_xticks(range(2), ["two-type\nprior", "three-type\nprior"], fontsize=5.8)
    axes[3].set_title("prior P(A→AA > truth)", fontsize=7.2)
    axes[3].axhline(1 / 5000, color=INK2, lw=0.7, ls=":")
    axes[3].text(1.6, 1 / 5000 * 2.5, "1 / 5k sims", fontsize=5.6, color=INK2, ha="right")
    save(fig, "e2c_arms")
    table(pd.concat(rows), "e2c_arms")


# ------------------------------------------------------- E2c rerun (prior truths)
E2CP_ARMS = [("ignore_c", "ignore\nC"), ("model_c_free", "C\nfree"),
             ("model_c_traj", "C\ntraj.")]
Q_COLORS = {0.0: BLUE, 0.5: AQUA, 1.0: ORANGE}


def fig_e2c_prior():
    """E2c rerun with 10 prior-drawn truths x q x 3 arms: per-run support mass and PPC
    (flag thresholds), A/B MAE vs the prior-mean baseline, and contraction."""
    for a, _ in E2CP_ARMS:
        require(f"e2c_prior/{a}", 30)
    r = pd.read_csv(RES / "e2c_prior" / "e2c_prior_runs.csv")
    s = pd.read_csv(RES / "e2c_prior" / "e2c_prior_shared_parameters.csv")
    tr = s.drop_duplicates(["seed", "parameter"])
    baseline = float((tr.truth - 0.1).abs().mean())  # three-type Dirichlet mean = 1/10
    r["usable"] = r.samples >= 100
    r["flagged"] = (r.support_mass < 0.1) | (r.ppc_out95 > 0.05)
    fig, axes = plt.subplots(1, 4, figsize=(W_FULL, 2.3), gridspec_kw=dict(wspace=0.5))
    for k, (arm, _) in enumerate(E2CP_ARMS):
        for j, q in enumerate([0.0, 0.5, 1.0]):
            u = r[(r.arm == arm) & (r.q == q)].sort_values("seed")
            x = k + (j - 1) * 0.25 + np.linspace(-0.07, 0.07, len(u))
            ok = u.usable.values
            kw = dict(s=9, lw=0)
            axes[0].scatter(x, u.support_mass, color=Q_COLORS[q], label=f"q = {q:g}"
                            if k == 0 else None, **kw)
            axes[2].scatter(x[ok], u.shared_mae[ok], color=Q_COLORS[q], **kw)
            for ax, col in [(axes[1], "ppc_out95"), (axes[3], "all_contraction")]:
                ax.scatter(x[ok], u[col][ok], color=Q_COLORS[q], **kw)
                # fewer than 100 posterior samples: shown open, not an estimate
                ax.scatter(x[~ok], u[col][~ok], facecolor="none", edgecolor=Q_COLORS[q],
                           s=9, lw=0.6)
    for ax, title in zip(axes, ["support mass", "PPC outside 95%", "A/B MAE (9 shared)",
                                "contraction (all free)"], strict=True):
        ax.set_xticks(range(3), [lab for _, lab in E2CP_ARMS], fontsize=6.2)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(title, fontsize=7.2)
        ax.grid(axis="x", visible=False)
    axes[0].axhline(0.1, color=RED, lw=0.7, ls=":")
    axes[0].set_ylim(-0.02, 1)
    h, lab = axes[0].get_legend_handles_labels()
    axes[2].legend(h, lab, fontsize=6, loc="upper right", handletextpad=0.2,
                   title="C read as A w.p.", title_fontsize=6)
    axes[1].axhline(0.05, color=RED, lw=0.7, ls=":")
    axes[1].set_ylim(-0.02, 1)
    axes[2].axhline(baseline, color=INK, lw=0.8, ls=(0, (4, 2)))
    axes[2].text(1.5, 0.105, f"dashed: prior-mean\nbaseline {baseline:.3f}", fontsize=5.8,
                 color=INK2, ha="center", va="bottom")
    axes[2].set_ylim(0, 0.3)
    axes[3].set_ylim(-0.05, 1.02)
    save(fig, "e2c_prior")
    g = r.groupby(["arm", "q"])
    # as scripts/analyse_e2c_prior.py: A/B recovery only from usable runs (>= 100
    # samples); PPC and contraction over every run that has them
    use = r[r.usable].groupby(["arm", "q"])
    agg = pd.DataFrame({
        "runs": g.size(), "usable": g.usable.sum(), "flagged": g.flagged.sum(),
        "support_median": g.support_mass.median(), "ppc_out95": g.ppc_out95.mean(),
        "sbc_cov_err": g.sbc_cov_err.mean(), "ab_mae": use.shared_mae.mean(),
        "ab_cov90": use.shared_cov90.mean(), "contraction": g.all_contraction.mean(),
    }).reset_index()
    pooled = r.groupby("arm")[["shared_mae", "shared_cov90", "all_contraction"]].mean()
    pooled = pooled.reset_index().assign(q="all")
    pooled = pooled.rename(columns={"shared_mae": "ab_mae", "shared_cov90": "ab_cov90",
                                    "all_contraction": "contraction"})
    out = pd.concat([agg, pooled], ignore_index=True)
    out["prior_mean_baseline"] = baseline
    table(out, "e2c_prior")


# ------------------------------------------------------------------- X5
X5_PRIORS = [("uniform (1)", "uniform Dirichlet", BLUE), ("all 0.5", "Jeffreys (all 0.5)", AQUA),
             ("flat B.to_A", "flat on B→A", ORANGE)]


def fig_x5a():
    """B->A posterior mean vs truth (0.1-0.9) per prior; B divisions free vs fixed."""
    require("x5/a_two_type", 75)
    require("x5/a_reversible_ab", 75)
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.5), sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    rows = []
    for ax, (sub, title, pm) in zip(axes, [
            ("a_two_type", "B divisions free (two_type, 10 free)", 1 / 6),
            ("a_reversible_ab", "B divisions fixed to 0 (reversible_ab, 7 free)", 1 / 3)],
            strict=True):
        r = pd.read_csv(RES / "x5" / sub / "x5a_runs.csv")
        ax.plot([0, 1], [0, 1], color=INK, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.axhline(pm, color=MUTED, lw=0.8, ls=":", zorder=1)  # uniform-prior mean
        for j, (key, lab, c) in enumerate(X5_PRIORS):
            g = r[r.prior == key].groupby("truth_to_A").agg(
                mean=("to_A_mean", "mean"), seed_sd=("to_A_mean", "std"),
                post_sd=("to_A_sd", "mean"), covered=("to_A_covered", "sum"),
                n=("seed", "size")).reset_index()
            x = g.truth_to_A + (j - 1) * 0.018
            ax.errorbar(x, g["mean"], yerr=g.post_sd, fmt="o-", color=c, ms=3.5, lw=1.0,
                        elinewidth=0.8, capsize=0, mec="white", mew=0.4, label=lab)
            rows.append(g.assign(law=sub.removeprefix("a_"), prior=key))
        ax.set_title(title, fontsize=7.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_xlabel("true B→A")
    axes[0].set_ylabel("B→A posterior mean ± posterior sd")
    axes[0].legend(fontsize=6.2, loc="upper left")
    save(fig, "x5a")
    table(pd.concat(rows, ignore_index=True), "x5a")


X5B_ARMS = [("b_vague", "vague\n10–80 h", MUTED), ("b_right", "right\n±25%", BLUE),
            ("b_wrong", "wrong\n+50%", RED)]
X5B_BOUNDS = {"b_vague": (10, 80), "b_right": (16, 25), "b_wrong": (24, 37.5)}


def fig_x5b():
    """BH, informative lifetime prior: A->AA and tau_A per seed for each prior arm."""
    for a, _, _ in X5B_ARMS:
        require(f"x5/{a}", 5)
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.0), gridspec_kw=dict(wspace=0.5))
    rows = []
    for k, (arm, lab, c) in enumerate(X5B_ARMS):
        runs, p = load(f"x5/{arm}")
        for ax, par in [(axes[0], "A.div_AA"), (axes[1], "A.mean_lifetime")]:
            u = p[p.parameter == par].sort_values("seed")
            x = k + np.linspace(-0.15, 0.15, len(u))
            ax.errorbar(x, u.posterior_mean, yerr=u.posterior_std, fmt="o", color=c, ms=3.5,
                        elinewidth=0.8, capsize=0, mec="white", mew=0.4)
            rows.append(dict(arm=arm, parameter=par, truth=u.truth.iloc[0],
                             mean=u.posterior_mean.mean(), post_sd=u.posterior_std.mean(),
                             covered=int(u.covered.astype(str).str.lower().isin(["true", "1", "1.0"]).sum()),
                             n=len(u)))
        lo, hi = X5B_BOUNDS[arm]
        axes[1].fill_between([k - 0.3, k + 0.3], lo, min(hi, 40), color=c, alpha=0.12, lw=0)
        axes[2].scatter(np.full(len(runs), k) + np.linspace(-0.15, 0.15, len(runs)),
                        runs.ppc_fraction_outside_95, s=10, color=c, lw=0)
        rows.append(dict(arm=arm, parameter="support_mass",
                         mean=runs.posterior_support_mass.mean()))
        rows.append(dict(arm=arm, parameter="ppc_out95", mean=runs.ppc_fraction_outside_95.mean()))
    truth_line(axes[0], 0.55)
    truth_line(axes[1], 20)
    axes[0].set_title("A→AA (truth 0.55)", fontsize=7.2)
    axes[1].set_title(r"$\tau_A$ (h; truth 20, band = prior)", fontsize=7.2)
    axes[1].set_ylim(10, 40)
    axes[2].set_title("PPC outside 95%", fontsize=7.2)
    axes[2].axhspan(0, 0.03, color=GRID, lw=0, zorder=0)
    axes[2].set_ylim(-0.02, 0.4)
    for ax in axes:
        ax.set_xticks(range(3), [lab for _, lab, _ in X5B_ARMS], fontsize=6.2)
        ax.set_xlim(-0.5, 2.5)
        ax.grid(axis="x", visible=False)
    save(fig, "x5b")
    table(pd.DataFrame(rows), "x5b")


def fig_x5c():
    """Tail truth B->A = 0.9 with all B outcomes free: B->A and B->AA vs clones."""
    require("x5/c_data_vs_prior", 30)
    r = pd.read_csv(RES / "x5" / "c_data_vs_prior" / "x5c_runs.csv")
    g = r.groupby(["prior", "clones"]).agg(
        to_A=("to_A", "mean"), to_A_post_sd=("to_A_sd", "mean"),
        to_A_covered=("to_A_covered", "sum"), to_A_contraction=("to_A_contraction", "mean"),
        B_div_AA=("B_div_AA", "mean"), mae=("mae", "mean"), support=("support", "mean"),
        ppc=("ppc_out95", "mean"), sbc=("sbc", "mean"), n=("seed", "size")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL * 0.72, 2.1), gridspec_kw=dict(wspace=0.35))
    for j, (key, lab, c) in enumerate([("uniform", "uniform Dirichlet", BLUE),
                                       ("flat B.to_A", "flat on B→A", ORANGE)]):
        u = g[g.prior == key].sort_values("clones")
        x = u.clones * (1 + (j - 0.5) * 0.12)
        axes[0].errorbar(x, u.to_A, yerr=u.to_A_post_sd, fmt="o-", color=c, ms=3.5, lw=1.0,
                         elinewidth=0.8, capsize=0, mec="white", mew=0.4, label=lab)
        axes[1].plot(x, u.B_div_AA, "o-", color=c, ms=3.5, lw=1.0, mec="white", mew=0.4)
    truth_line(axes[0], 0.9, "truth")
    truth_line(axes[1], 0.0, "truth")
    axes[0].set_title("B→A (mean ± posterior sd)", fontsize=7.2)
    axes[1].set_title("B→AA posterior mean", fontsize=7.2)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([32, 128, 512], ["32", "128", "512"])
        ax.minorticks_off()
        ax.set_xlabel("clones per endpoint (2 endpoints)")
        ax.set_ylim(-0.03, 1.0)
    h, lab = axes[0].get_legend_handles_labels()
    axes[1].legend(h, lab, fontsize=6.2, loc="upper right")
    save(fig, "x5c")
    table(g, "x5c")


def fig_x5d():
    """Prior-averaged recovery (30 truths from the prior): contraction and MAE per
    parameter, GW vs BH, with the prior-mean MAE baseline."""
    require("x5/d_gw", 30)
    require("x5/d_bh", 30)
    d = {s: pd.read_csv(RES / "x5" / f"d_{s}" / "prior_average_parameters.csv").set_index(
        "parameter") for s in ("gw", "bh")}
    pars = [p for p in P_ROWS if p in d["gw"].index]
    allp = pars + ["A.mean_lifetime", "B.mean_lifetime"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.3),
                             gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.25))
    xs = np.arange(len(allp))
    for j, (s, lab, c) in enumerate([("gw", "GW", BLUE), ("bh", "BH", ORANGE)]):
        v = d[s].contraction.reindex(allp)
        axes[0].bar(xs + (j - 0.5) * 0.38, v, width=0.36, color=c, label=lab, lw=0)
        e = d[s].abs_error.reindex(pars)
        b = d[s].prior_mean_error.reindex(pars)
        x = np.arange(len(pars)) + (j - 0.5) * 0.38
        axes[1].bar(x, e, width=0.36, color=c, label=f"{lab} posterior mean", lw=0)
        axes[1].scatter(x, b, marker="_", s=40, color=INK, lw=1.0,
                        label="prior-mean baseline" if j == 0 else None, zorder=3)
    axes[0].set_xticks(xs, [PRETTY[p].replace(" (h)", "") for p in allp], rotation=45,
                       ha="right", fontsize=6.2)
    axes[0].set_title("contraction (mean over 30 prior truths)", fontsize=7.2)
    axes[0].axhline(0.5, color=AXIS, lw=0.7, ls=":")
    axes[0].legend(fontsize=6.2, loc="upper right", ncol=2)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_xticks(np.arange(len(pars)), [PRETTY[p] for p in pars], rotation=45,
                       ha="right", fontsize=6.2)
    axes[1].set_title("absolute error (probabilities)", fontsize=7.2)
    axes[1].legend(fontsize=5.8, loc="upper left")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    save(fig, "x5d")
    t = pd.concat([d[s].reset_index().assign(simulator=s.upper()) for s in d], ignore_index=True)
    table(t, "x5d")


# ------------------------------------------------------------------- E3a
def fig_e3a():
    require("e3a", 60)
    runs, params = load("e3a")
    params = params.copy()
    b_free = ov(params, "data.simulator.reference.B.mean_lifetime")
    setting = ov(params, "lifetime_setting")
    params["B"] = np.where(params.sweep == "free", b_free,
                           setting.str.extract(r"B:([0-9.]+)")[0])
    params["T"] = setting.str.extract(r"T:([0-9.]+)")[0].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.0), gridspec_kw=dict(wspace=0.45))
    free = params[params.sweep == "free"]
    for par, c, lab in [("A.mean_lifetime", BLUE, "A (truth 20 h)"),
                        ("B.mean_lifetime", ORANGE, "B")]:
        s = free[free.parameter == par]
        x = s.B.astype(float) + (-1 if par.startswith("A") else 1)
        axes[0].errorbar(x, s.posterior_mean, yerr=s.posterior_std, fmt="o", color=c, ms=3,
                         elinewidth=0.6, label=lab, mec="white", mew=0.3)
    axes[0].plot([15, 50], [15, 50], color=INK, lw=0.8, ls=(0, (4, 2)))
    axes[0].axhline(20, color=BLUE, lw=0.6, ls=":")
    axes[0].set_xlabel("true B lifetime (h)")
    axes[0].set_ylabel("posterior mean ± sd (h)")
    axes[0].set_title("a  lifetimes inferred", fontsize=7.2)
    axes[0].legend(fontsize=6.2, loc="upper left")
    fx = runs[runs.sweep == "fixed"].copy()
    st = ov(fx, "lifetime_setting")
    fx["T"] = st.str.extract(r"T:([0-9.]+)")[0].astype(float)
    fx["B"] = st.str.extract(r"B:([0-9.]+)")[0].astype(float)
    colors = {20.0: BLUE, 30.0: ORANGE, 45.0: AQUA}
    for b, g in fx.groupby("B"):
        axes[1].scatter(g["T"] / 20 + np.random.default_rng(0).normal(0, 0.006, len(g)),
                        g.posterior_support_mass, color=colors[b], s=10, lw=0,
                        label=f"B = {b:g} h")
    axes[1].axvline(1.0, color=INK, lw=0.8, ls=(0, (4, 2)))
    axes[1].set_xlabel("fixed T / true A lifetime")
    axes[1].set_ylabel("posterior support mass")
    axes[1].set_title("b  one global lifetime T", fontsize=7.2)
    axes[1].legend(fontsize=6, loc="center right")
    axes[1].set_ylim(-0.03, 1.0)
    axes[1].text(1.05, 0.45, "T = 15, 20 h (B = 20 h):\nsupport not recorded", fontsize=5.5,
                 color=MUTED)
    # A.div_AA: free vs fixed (fits only)
    lab_order = [("free", "20", None), ("fixed", "20", 20.0), ("fixed", "20", 15.0),
                 ("free", "30", None), ("fixed", "30", 18.75)]
    for i, (sw, b, t) in enumerate(lab_order):
        s = params[(params.sweep == sw) & (params.B.astype(str) == b) &
                   (params.parameter == "A.div_AA")]
        if t is not None:
            s = s[np.isclose(s["T"], t)]
        c = BLUE if sw == "free" else ORANGE
        axes[2].errorbar(i + np.linspace(-0.2, 0.2, len(s)), s.posterior_mean,
                         yerr=s.posterior_std, fmt="o", color=c, ms=3, elinewidth=0.6,
                         mec="white", mew=0.3)
    truth_line(axes[2], 0.55)
    axes[2].set_xticks(range(len(lab_order)),
                       ["B20\nfree", "B20\nT=20", "B20\nT=15", "B30\nfree", "B30\nT=18.8"],
                       fontsize=6)
    axes[2].set_ylabel("A→AA")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("c  A→AA (runs that fit)", fontsize=7.2)
    save(fig, "e3a")


# ------------------------------------------------------------------- E3b
def fig_e3b():
    require("e3b", 35)
    _, params = load("e3b")
    shape = ov(params, "data.simulator.model.lifetime_shape")
    sel = ["A.div_AA", "A.div_AB", "B.to_A", "B.death", "A.mean_lifetime", "B.mean_lifetime"]
    t = paired(params, shape, "8", sel)
    order = ["1", "2", "4", "8", "32", "{A:4,B:16}", "{A:16,B:4}"]
    labels = ["1\n(1.0)", "2\n(.71)", "4\n(.5)", "8\n(.35)", "32\n(.18)", "A4\nB16", "A16\nB4"]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.0),
                             gridspec_kw=dict(width_ratios=[2, 1.2], wspace=0.3))
    probs = [s for s in sel if is_prob(s)]
    for k, par in enumerate(probs):
        u = t[t.parameter == par].set_index("setting").reindex(order)
        x = np.arange(len(order)) + (k - 1.5) * 0.17
        axes[0].errorbar(x, u["shift"], yerr=u.se, fmt="o", color=CAT[k], ms=3.2,
                         elinewidth=0.8, label=PRETTY[par], mec="white", mew=0.3)
    for k, par in enumerate([s for s in sel if not is_prob(s)]):
        u = t[t.parameter == par].set_index("setting").reindex(order)
        x = np.arange(len(order)) + (k - 0.5) * 0.25
        axes[1].errorbar(x, u["shift"], yerr=u.se, fmt="o", color=[BLUE, ORANGE][k], ms=3.2,
                         elinewidth=0.8, label=PRETTY[par], mec="white", mew=0.3)
    for ax in axes:
        zero_line(ax)
        ax.set_xticks(range(len(order)), labels, fontsize=6)
        ax.axvspan(2.5, 4.5, color=GRID, lw=0, alpha=0.6)
        ax.set_xlabel("data lifetime shape (CV); inference assumes 8")
        ax.legend(fontsize=6, ncol=2 if ax is axes[0] else 1, loc="lower right")
    axes[0].set_ylabel("shift vs shape 8 (± paired SE)")
    axes[0].set_title("outcome probabilities", fontsize=7.2)
    axes[1].set_title("lifetimes (h)", fontsize=7.2)
    save(fig, "e3b")
    table(t, "e3b_shifts")


# ------------------------------------------------------------------- E3c
def fig_e3c():
    require("e3c", 30)
    _, params = load("e3c")
    params = params.copy()
    params["ptrue"] = ov(params, "truth.values.A").str.extract(r"persist:([0-9.]+)")[0]
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.0), gridspec_kw=dict(wspace=0.45))
    free = params[(params.sweep == "persist_free") & (params.parameter == "A.persist")]
    x = free.ptrue.astype(float)
    axes[0].errorbar(x + np.random.default_rng(1).normal(0, 0.004, len(x)), free.posterior_mean,
                     yerr=free.posterior_std, fmt="o", color=BLUE, ms=3, elinewidth=0.6,
                     mec="white", mew=0.3)
    axes[0].plot([0, 0.32], [0, 0.32], color=INK, lw=0.8, ls=(0, (4, 2)))
    axes[0].axhline(1 / 6, color=MUTED, lw=0.8, ls=":")
    axes[0].text(0.31, 1 / 6 + 0.01, "prior mean", fontsize=5.8, color=MUTED, ha="right")
    axes[0].set_xlabel("true A persist")
    axes[0].set_ylabel("posterior mean ± sd")
    axes[0].set_title("a  A persist (free)", fontsize=7.2)
    # shifts fixed0 - free paired by seed and setting
    p = params[params.parameter.isin(["A.div_AA", "A.div_BB", "A.to_B", "A.mean_lifetime",
                                      "B.mean_lifetime"])]
    piv = p.pivot_table(index=["parameter", "ptrue", "seed"], columns="sweep",
                        values="posterior_mean")
    d = (piv["persist_fixed0"] - piv["persist_free"]).groupby(["parameter", "ptrue"]).agg(
        ["mean", "std", "count"]).reset_index()
    d["se"] = d["std"] / np.sqrt(d["count"])
    for k, par in enumerate(["A.div_AA", "A.div_BB", "A.to_B"]):
        u = d[d.parameter == par]
        axes[1].errorbar(u.ptrue.astype(float) + (k - 1) * 0.008, u["mean"], yerr=u.se,
                         color=CAT[k], marker="o", ms=3.2, lw=1.1, elinewidth=0.7,
                         label=PRETTY[par])
    for k, par in enumerate(["A.mean_lifetime", "B.mean_lifetime"]):
        u = d[d.parameter == par]
        axes[2].errorbar(u.ptrue.astype(float) + (k - 0.5) * 0.008, u["mean"], yerr=u.se,
                         color=[BLUE, ORANGE][k], marker="o", ms=3.2, lw=1.1, elinewidth=0.7,
                         label=PRETTY[par])
    for ax, ttl in [(axes[1], "b  Δ fixed 0 − free"), (axes[2], "c  Δ lifetimes (h)")]:
        zero_line(ax)
        ax.set_xlabel("true A persist")
        ax.set_title(ttl, fontsize=7.2)
        ax.legend(fontsize=6)
        ax.set_xticks([0.03, 0.15, 0.30])
    save(fig, "e3c")
    table(d, "e3c_shifts")


# ------------------------------------------------------------------- overview
def fig_detectability():
    """Largest control-relative shift (in posterior sd) vs PPC for every BH
    misspecified setting with a paired well-specified control."""
    pts = []

    def add(exp, params, runs_ppc, setting, control, drop=()):
        pars = [p for p in P_ROWS + ["A.mean_lifetime", "B.mean_lifetime"] if p not in drop]
        t = paired(params, setting, control, pars)
        ctrl_sd = t[t.setting == control].set_index("parameter").post_sd
        t = t[t.setting != control].copy()
        t["z"] = (t["shift"].abs() / t.parameter.map(ctrl_sd))
        best = t.loc[t.groupby("setting").z.idxmax()]
        for _, r in best.iterrows():
            pts.append(dict(exp=exp, setting=r.setting, z=r.z, par=r.parameter,
                            ppc=runs_ppc.get(r.setting, np.nan)))

    # E1a founder type (fixed arm) and founder number ([1] arm)
    _, p = load("e1a/founder_type")
    p = p[p.sweep == "fixed_bh"]
    s = _founder_f(ov(p, "data.law.fixed.founder")).round(2).astype(str)
    add("E1a type", p, p.assign(s=s.values).groupby("s").ppc.mean().to_dict(), s, "1.0")
    _, p = load("e1a/founder_number_bh")
    p = p[ov(p, "law.founder_sizes").str.replace(" ", "") == "[1]"]
    s = _p1(ov(p, "data.law.reference.founder_size")).astype(str)
    add("E1a number", p, p.assign(s=s.values).groupby("s").ppc.mean().to_dict(), s, "1.0")
    _, p = load("e1c/bh")
    s = ov(p, "law.fixed.A.death") + "/" + ov(p, "law.fixed.B.death")
    add("E1c", p, p.assign(s=s.values).groupby("s").ppc.mean().to_dict(), s, "0.03/0.1",
        drop=("A.death", "B.death"))
    _, p = load("e2/ignored_bh")
    s = ov(p, "data.observation.confusion")
    add("E2b", p, p.assign(s=s.values).groupby("s").ppc.mean().to_dict(), s, "0")
    _, p = load("e3b")
    s = ov(p, "data.simulator.model.lifetime_shape")
    add("E3b", p, p.assign(s=s.values).groupby("s").ppc.mean().to_dict(), s, "8")
    _, p = load("e3c")
    s = ov(p, "truth.values.A").str.extract(r"persist:([0-9.]+)")[0] + "|" + p.sweep
    for pt in ["0.03", "0.15", "0.3"]:
        q = p[s.str.startswith(pt + "|").values]
        ss = s[s.str.startswith(pt + "|")]
        add("E3c", q, q.assign(s=ss.values).groupby("s").ppc.mean().to_dict(), ss,
            f"{pt}|persist_free", drop=("A.persist",))
    # E3a fixed arm, settings that produced estimates (support mass > 0.05)
    runs, p = load("e3a")
    b = np.where(p.sweep == "free", ov(p, "data.simulator.reference.B.mean_lifetime"),
                 ov(p, "lifetime_setting").str.extract(r"B:([0-9.]+)")[0])
    s = pd.Series(np.where(p.sweep == "free", "free", ov(p, "lifetime_setting")),
                  index=p.index)
    for bv in ["20", "30"]:
        mask = (b == bv)
        q, ss = p[mask], s[mask]
        ok = q.groupby(ss).support.mean()
        # T equal to the true lifetimes ({B:20,T:20}) is not a misspecification
        keep = [k for k in ss.unique() if k == "free" or
                (not (ok.get(k, np.nan) < 0.05) and k != "{B:20,T:20}")]
        q, ss = q[ss.isin(keep)], ss[ss.isin(keep)]
        add("E3a", q, q.assign(s=ss.values).groupby("s").ppc.mean().to_dict(), ss, "free",
            drop=("A.mean_lifetime", "B.mean_lifetime"))
    d = pd.DataFrame(pts)
    table(d, "detectability")
    fig, ax = plt.subplots(figsize=(W_FULL * 0.62, 2.3))
    exps = list(dict.fromkeys(d.exp))
    for k, e in enumerate(exps):
        u = d[d.exp == e]
        ax.scatter(u.z, u.ppc, s=18, color=CAT[k % 8], label=e, lw=0.4, ec="white", zorder=3,
                   marker="osD^vP<"[k % 7])
    # E3a settings that cannot fit (support ~ 0): plotted separately
    fx = runs[runs.sweep == "fixed"]
    bad = fx.groupby(ov(fx, "lifetime_setting")).agg(ppc=("ppc_fraction_outside_95", "mean"),
                                                    sm=("posterior_support_mass", "mean"))
    bad = bad[bad.sm < 0.05].dropna(subset=["ppc"])
    ax.scatter(np.full(len(bad), 2.6), bad.ppc, marker="x", color=RED, s=18,
               label="E3a T > A lifetime\n(support ≈ 0)", zorder=3)
    ax.axhspan(0, 0.03, color=GRID, lw=0, zorder=0)
    ax.text(2.75, 0.015, "well-specified PPC range", fontsize=5.8, color=INK2, ha="right",
            va="center")
    ax.axvline(1, color=AXIS, lw=0.8, ls=":")
    ax.set_yscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)),
                                         lambda v: v ** 2))
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6])
    ax.set_xlim(0, 2.8)
    ax.set_xlabel("largest shift vs control (in control posterior sd)")
    ax.set_ylabel("PPC fraction outside 95%")
    ax.legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, "detectability")


def fig_detectability_gw():
    """GW counterpart of fig_detectability: largest control-relative shift (in control
    posterior sd) vs PPC and vs support mass, for every GW misspecified setting with a
    paired control; settings without an estimate (support < 0.02) are listed apart."""
    for rel, n in [("e1a/founder_type", 80), ("e1a/founder_number_gw", 30), ("e1c/gw", 80),
                   ("e2/ignored_gw", 30)]:
        require(rel, n)
    pts = []

    def add(exp, params, setting, control, drop=()):
        pars = [p for p in P_ROWS if p not in drop]
        t = paired(params, setting, control, pars)
        ctrl_sd = t[t.setting == control].set_index("parameter").post_sd
        t = t[t.setting != control].copy()
        t["z"] = t["shift"].abs() / t.parameter.map(ctrl_sd)
        q = params.assign(s=setting.values).groupby("s")
        ppc, sup = q.ppc.mean(), q.support.mean()
        for st, g in t.groupby("setting"):
            r = g.loc[g.z.idxmax()]
            pts.append(dict(exp=exp, setting=st, z=r.z, par=r.parameter, ppc=ppc.get(st),
                            support=sup.get(st)))

    _, p = load("e1a/founder_type")
    p = p[p.sweep == "fixed_gw"]
    add("E1a type", p, _founder_f(ov(p, "data.law.fixed.founder")).round(2).astype(str), "1.0")
    _, p = load("e1a/founder_number_gw")
    p = p[ov(p, "law.founder_sizes").str.replace(" ", "") == "[1]"]
    add("E1a number", p, _p1(ov(p, "data.law.reference.founder_size")).astype(str), "1.0")
    _, p = load("e1c/gw")
    add("E1c", p, ov(p, "law.fixed.A.death") + "/" + ov(p, "law.fixed.B.death"), "0.03/0.1",
        drop=("A.death", "B.death"))
    _, p = load("e2/ignored_gw")
    add("E2b", p, ov(p, "data.observation.confusion"), "0")
    d = pd.DataFrame(pts)
    table(d, "detectability_gw")
    ok = d[d.support >= 0.02]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.3), gridspec_kw=dict(wspace=0.3))
    exps = list(dict.fromkeys(d.exp))
    for k, e in enumerate(exps):
        u = ok[ok.exp == e]
        for ax, col in [(axes[0], "ppc"), (axes[1], "support")]:
            ax.scatter(u.z, u[col], s=18, color=CAT[k % 8], label=e if col == "ppc" else None,
                       lw=0.4, ec="white", zorder=3, marker="osD^vP<"[k % 7])
    axes[0].axhspan(0, 0.03, color=GRID, lw=0, zorder=0)
    axes[0].set_ylabel("PPC fraction outside 95%")
    axes[0].set_ylim(-0.01, 0.4)
    axes[1].axhspan(0.5, 0.9, color=GRID, lw=0, zorder=0)
    axes[1].set_ylabel("posterior support mass")
    axes[1].set_ylim(-0.02, 1)
    for ax in axes:
        ax.axvline(1, color=AXIS, lw=0.8, ls=":")
        ax.set_xlabel("largest shift vs control (in control posterior sd)")
    r1c, _ = load("e1c/gw")
    sm = r1c.groupby(ov(r1c, "law.fixed.A.death") + "/" + ov(r1c, "law.fixed.B.death"))
    n_bad = int((sm.posterior_support_mass.mean() < 0.02).sum())
    axes[1].text(0.98, 0.02, f"not shown: {n_bad} E1c settings\nwith support ≈ 0 (no estimate)",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=5.8, color=INK2)
    axes[0].legend(fontsize=6, loc="upper left")
    save(fig, "detectability_gw")


# ======================================================================= main
FIGURES = [
    fig_e0_bto_a, fig_e0_c2st, fig_design, fig_gpu, fig_e4, fig_ridge, fig_e1b,
    fig_e1a_type, fig_e1a_number, fig_e1c, fig_e2, fig_e2c_ignore, fig_e3a, fig_e3b,
    fig_e3c, fig_detectability,
    # pending sweeps: built only once complete
    fig_e4_gw, fig_e1c_gw, fig_e1a_number_gw, fig_e2_gw, fig_e2c_model, fig_e2c_traj,
    fig_e2c_arms, fig_detectability_gw,
    fig_e2c_prior, fig_x5a, fig_x5b, fig_x5c, fig_x5d,
]


def main(argv: list[str]) -> int:
    only = set(argv)
    for f in FIGURES:
        if only and f.__name__ not in only:
            continue
        print(f.__name__)
        try:
            f()
        except FileNotFoundError as err:
            print(f"  skipped: {err}")
        except Exception:  # noqa: BLE001 - one broken figure must not stop the others
            print("  FAILED:")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
