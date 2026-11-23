"""Analyse the free-founder NUTS runs (``run.py``): tables and figures for docs/free_founder.

Reads ``results/exact_gw/free_founder/<scen>/<model>/<gt>_s<seed>.{json,npz}`` and, for
comparison, the misspecification study's all-A NUTS posteriors (``nuts/F70``) and its MLE
fits (``fits_long.csv``: standard model ``std`` and extended model ``ext`` with the founder
mixture / doublet fraction estimated).

    uv run python scripts/free_founder/analyse.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = Path(os.environ.get("FF_RES", "/home/matteo/BIO-SBI/results/exact_gw/free_founder"))
MISSP = Path("/home/matteo/BIO-SBI/results/exact_gw/misspecification")
DOC = Path(os.environ.get("FF_DOC", ROOT / "docs" / "free_founder"))
TAB, FIG = DOC / "tables", DOC / "figures"
GTS = ["T1_ref", "T2_toA05", "T3_Bdiv", "T4_slowA", "T5_C7", "P1", "P2", "P3", "P4"]
SCEN = ["base", "F70", "D15A", "D15mix"]
KEYS = ["B.to_A", "m_BA", "rho", "stableA"]
COL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
       "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}  # fmt: skip
GENS = (4, 6, 8, 10)


def load_runs() -> list[dict]:
    runs = []
    for f in sorted(RES.glob("*/*/*.json")):
        r = json.loads(f.read_text())
        r["npz"] = f.with_suffix(".npz")
        runs.append(r)
    return runs


# ---------------------------------------------------------------------- posteriors
def fmt_iv(s, digits=2):
    return f"{s['mean']:.{digits}f} [{s['q05']:.{digits}f}, {s['q95']:.{digits}f}]"


def founder_table(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        st = r["stats"]
        row = {"scen": r["scen"], "model": r["model"], "gt": r["gt"], "seed": r["seed"]}
        for k, short in (("founder.A", "pA"), ("founder_size.2", "pi2")):
            if k in st and st[k]["sd"] > 0:
                s = st[k]
                row.update({f"{short}_mean": s["mean"], f"{short}_q05": s["q05"],
                            f"{short}_q95": s["q95"], f"{short}_truth": s["truth"],
                            f"{short}_cover90": s["cover90"]})  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


def key_table(runs) -> pd.DataFrame:
    """Bias, z, 90% coverage of B->A, m_BA, rho, stable A fraction (free-founder runs)."""
    rows = []
    for r in runs:
        for k in KEYS:
            s = r["stats"][k]
            rows.append({"scen": r["scen"], "model": r["model"], "gt": r["gt"],
                         "seed": r["seed"], "key": k, "mean": s["mean"], "sd": s["sd"],
                         "q05": s["q05"], "q95": s["q95"], "truth": s["truth"],
                         "bias": s["mean"] - s["truth"], "z": s["z"],
                         "cover90": s["cover90"]})  # fmt: skip
    return pd.DataFrame(rows)


def all_a_nuts() -> pd.DataFrame:
    rows = []
    for f in sorted((MISSP / "nuts").glob("*/*.json")):
        r = json.loads(f.read_text())
        for k in KEYS:
            s = r["stats"][k]
            rows.append({"scen": r["scen"], "model": "allA_nuts", "gt": r["gt"], "seed": 0,
                         "key": k, "mean": s["mean"], "sd": s["sd"], "q05": s["q05"],
                         "q95": s["q95"], "truth": s["truth"],
                         "bias": s["mean"] - s["truth"], "z": s["z"],
                         "cover90": s["cover90"]})  # fmt: skip
    return pd.DataFrame(rows)


def mle(seed_max: int = 9) -> pd.DataFrame:
    d = pd.read_csv(MISSP / "fits_long.csv")
    d = d[d.scen.isin(SCEN) & d.model.isin(["std", "ext"]) & (d.seed <= seed_max)]
    rows = []
    for _, x in d.iterrows():
        for k in KEYS:
            rows.append({"scen": x.scen, "model": f"mle_{x.model}", "gt": x["gt"], "seed": x.seed,
                         "key": k, "mean": x[k], "truth": x[f"t_{k}"],
                         "bias": x[k] - x[f"t_{k}"],
                         "x_founder_A": x.get("x_founder_A"),
                         "x_doublet": x.get("x_doublet")})  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- progenitors
def true_label(size, n_a):
    return np.where(size == 1, np.where(n_a == 1, "A", "B"),
                    np.where(n_a == 2, "AA", np.where(n_a == 1, "AB", "BB")))  # fmt: skip


def auc(score, y):
    """Mann-Whitney AUC (ties count 1/2); nan if one class is empty."""
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tasks(labels, prog, clones):
    """Binary tasks: (name, score, target) for the clones of one run."""
    lab = list(labels)
    tl = true_label(clones[:, 3], clones[:, 4])
    out = []
    a_cols = [lab.index(c) for c in ("A", "AA") if c in lab]
    out.append(("all_A", prog[:, a_cols].sum(1), np.isin(tl, ["A", "AA"])))
    d_cols = [lab.index(c) for c in ("AA", "AB", "BB") if c in lab]
    if d_cols:
        out.append(("doublet", prog[:, d_cols].sum(1), clones[:, 3] == 2))
    return out


def oracle_prog(r) -> tuple[list[str], np.ndarray]:
    """Per-clone progenitor posterior at the TRUE parameters and founder distribution
    (the information limit of the counts), aligned with the run's saved clones."""
    sys.path.insert(0, str(Path(__file__).parent))
    import run as R

    from bio_sbi.inference import dediff_lrt as dl
    from bio_sbi.inference.exact_gw import ExactLikelihood

    cohorts, _ = R.dataset_with_founders(r["scen"], r["gt"], r["seed"])
    sim = R.make_sim(R.MODELS[r["model"]])
    lik = ExactLikelihood(sim, dl.observation(None), cohorts, kernel="adjoint")
    o = R.C.SCENARIOS[r["scen"]][1]
    P = R.C.TRUTHS[r["gt"]]
    m = {"A": dict(zip(dl.ENTRIES["A"], P[0], strict=True)),
         "B": dict(zip(dl.ENTRIES["B"], P[1], strict=True)),
         "founder": {"A": o.get("founder_A", 1.0), "B": 1 - o.get("founder_A", 1.0)}}  # fmt: skip
    if "founder_size.2" in sim.space.names:
        m["founder_size"] = {"1": 1 - o.get("doublet", 0.0), "2": o.get("doublet", 0.0)}
    theta = sim.space.encode(sim.physical_from_mapping(m)).astype(float)
    labels, probs = lik.progenitor_posterior(theta)
    out = []
    for h, c, pr in zip(lik.data, cohorts, probs, strict=True):
        cell = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(h.a, h.b, strict=True))}
        out.append(pr[[cell[(int(a), int(b))] for a, b in c.counts]])
    return labels, np.concatenate(out)


def progenitor_table(runs, which="prog") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per run x task x generation: AUC, Brier, prior-only Brier, mean score / rate; and
    the pooled reliability data (score, target) per scenario x model x task.
    ``which``: ``prog`` (averaged over posterior draws), ``prog_at_mean`` or ``oracle``."""
    rows, pooled = [], []
    for r in runs:
        z = np.load(r["npz"])
        clones = z["clones"]
        if which == "oracle":
            labels, prog = oracle_prog(r)
        else:
            prog, labels = z[which].astype(float), z["labels"]
        for name, score, y in tasks(labels, prog, clones):
            for g in (*GENS, "all"):
                m = np.ones(len(y), bool) if g == "all" else clones[:, 0] == g
                s, t = score[m], y[m].astype(float)
                base = np.full_like(s, s.mean())  # constant prediction (no clone information)
                rows.append({"scen": r["scen"], "model": r["model"], "gt": r["gt"],
                             "seed": r["seed"], "task": name, "gen": g, "n": int(m.sum()),
                             "rate": float(t.mean()), "mean_score": float(s.mean()),
                             "auc": auc(s, t), "brier": float(np.mean((s - t) ** 2)),
                             "brier_const": float(np.mean((base - t) ** 2)),
                             "brier_rate": float(t.mean() * (1 - t.mean()))})  # fmt: skip
            pooled.append(pd.DataFrame({"scen": r["scen"], "model": r["model"], "gt": r["gt"],
                                        "task": name, "gen": clones[:, 0], "score": score,
                                        "y": y.astype(float)}))  # fmt: skip
    return pd.DataFrame(rows), pd.concat(pooled, ignore_index=True)


# ---------------------------------------------------------------------- mixing
def mixing_table(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        c = r["convergence"]
        cm = c.get("chain_means") or {}
        rows.append({"scen": r["scen"], "model": r["model"], "gt": r["gt"], "seed": r["seed"],
                     "max_rhat": c["max_rank_rhat"], "min_ess_bulk": c["min_ess_bulk"],
                     "min_ess_tail": c["min_ess_tail"], "max_mean_spread": c["max_mean_spread"],
                     "divergences": c["divergences"], "depth_hits": c["max_tree_depth_hits"],
                     "draws_per_chain": r["stop"]["draws_per_chain"],
                     "stop": r["stop"]["reason"], "failed": ",".join(r["stop"]["failed"]),
                     "pA_chain_means": cm.get("founder.A"),
                     "toA_chain_means": cm.get("B.to_A"),
                     "init_lp_spread": (max(c["init_log_posterior"])
                                        - min(c["init_log_posterior"])),
                     "init_lp_top4": [round(v, 1) for v in sorted(
                         c["init_log_posterior"], reverse=True)[:4]],
                     "corr_pA_toA": r["corr"]["founder.A~B.to_A"],
                     "minutes": r["seconds"] / 60})  # fmt: skip
    d = pd.DataFrame(rows)
    d["poor"] = (d.max_rhat > 1.05) | (d.min_ess_bulk < 100) | (d.max_mean_spread > 0.5)
    return d


# ---------------------------------------------------------------------- figures
def _style(ax):
    ax.grid(axis="x", color=COL["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=COL["ink2"])


def fig_founder(ft: pd.DataFrame, path: Path):
    combos = [(s, m) for s in SCEN for m in ("fA", "fA+d2")
              if ((ft.scen == s) & (ft.model == m)).any()]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.35 * len(combos) * len(GTS) / 3 + 2),
                             sharey=True)  # fmt: skip
    ylabels, y = [], 0
    for s, m in combos:
        for g in GTS:
            x = ft[(ft.scen == s) & (ft.model == m) & (ft["gt"] == g) & (ft.seed == 0)]
            if not len(x):
                continue
            x = x.iloc[0]
            for ax, k, c in ((axes[0], "pA", COL["blue"]), (axes[1], "pi2", COL["orange"])):
                if f"{k}_mean" in x and pd.notna(x.get(f"{k}_mean")):
                    ax.plot([x[f"{k}_q05"], x[f"{k}_q95"]], [y, y], color=c, lw=2,
                            solid_capstyle="round")  # fmt: skip
                    ax.plot(x[f"{k}_mean"], y, "o", color=c, ms=4)
                    ax.plot(x[f"{k}_truth"], y, "|", color=COL["ink"], ms=9, mew=1.5)
            ylabels.append(f"{s} {m} {g}")
            y += 1
        y += 0.6
    pos, k = [], 0
    for s, m in combos:
        for g in GTS:
            if ((ft.scen == s) & (ft.model == m) & (ft["gt"] == g) & (ft.seed == 0)).any():
                pos.append(k)
                k += 1
        k += 0.6
    for ax in axes:
        _style(ax)
        ax.set_xlim(-0.02, 1.02)
    axes[0].set_yticks(pos, ylabels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("P(founder = A): posterior mean, 90% interval; | = truth")
    axes[1].set_xlabel("doublet fraction π₂: posterior mean, 90% interval; | = truth")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_toa(kt: pd.DataFrame, nuts_a: pd.DataFrame, ml: pd.DataFrame, path: Path):
    """B->A per truth: free-founder posterior vs all-A NUTS (F70) and the MLEs (seed 0)."""
    fig, axes = plt.subplots(1, len(SCEN), figsize=(13, 4.2), sharey=True)
    for ax, s in zip(axes, SCEN, strict=True):
        for i, g in enumerate(GTS):
            for off, model, c in ((-0.2, "fA", COL["blue"]), (0.0, "fA+d2", COL["aqua"])):
                x = kt[(kt.scen == s) & (kt.model == model) & (kt["gt"] == g) & (kt.seed == 0)
                       & (kt.key == "B.to_A")]  # fmt: skip
                if len(x):
                    x = x.iloc[0]
                    ax.plot([i + off] * 2, [x.q05, x.q95], color=c, lw=2.5,
                            solid_capstyle="round")  # fmt: skip
                    ax.plot(i + off, x["mean"], "o", color=c, ms=4)
            x = nuts_a[(nuts_a.scen == s) & (nuts_a["gt"] == g) & (nuts_a.key == "B.to_A")]
            if len(x):
                x = x.iloc[0]
                ax.plot([i + 0.2] * 2, [x.q05, x.q95], color=COL["orange"], lw=2.5,
                        solid_capstyle="round")  # fmt: skip
                ax.plot(i + 0.2, x["mean"], "o", color=COL["orange"], ms=4)
            for model, mk in (("mle_std", "x"), ("mle_ext", "+")):
                x = ml[(ml.scen == s) & (ml.model == model) & (ml["gt"] == g) & (ml.seed == 0)
                       & (ml.key == "B.to_A")]  # fmt: skip
                if len(x):
                    ax.plot(i + 0.33, x["mean"].iloc[0], mk, color=COL["ink2"], ms=5)
            t = kt[(kt["gt"] == g) & (kt.key == "B.to_A")]["truth"]
            if len(t):
                ax.plot([i - 0.35, i + 0.4], [t.iloc[0]] * 2, color=COL["ink"], lw=1)
        ax.set_xticks(range(len(GTS)), GTS, rotation=60, fontsize=8)
        ax.set_title(s, fontsize=10)
        ax.grid(axis="y", color=COL["grid"], lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("B→A (posterior mean, 90% interval)")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=COL["blue"], lw=2.5, label="free p_A (sizes {1})"),
               Line2D([], [], color=COL["aqua"], lw=2.5, label="free p_A + π₂ (sizes {1,2})"),
               Line2D([], [], color=COL["orange"], lw=2.5, label="all-A NUTS (study)"),
               Line2D([], [], color=COL["ink2"], marker="x", ls="", label="MLE all-A"),
               Line2D([], [], color=COL["ink2"], marker="+", ls="", label="MLE extended"),
               Line2D([], [], color=COL["ink"], lw=1, label="truth")]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_reliability(pool: pd.DataFrame, path: Path):
    panels = [(s, m, t) for s, m, t in (("F70", "fA", "all_A"), ("D15mix", "fA+d2", "all_A"),
                                         ("D15A", "fA+d2", "doublet"),
                                         ("D15mix", "fA+d2", "doublet"))
              if ((pool.scen == s) & (pool.model == m) & (pool.task == t)).any()]  # fmt: skip
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6), squeeze=False)
    colors = dict(zip(GENS, (COL["blue"], COL["orange"], COL["aqua"], COL["yellow"]),
                      strict=True))  # fmt: skip
    edges = np.linspace(0, 1, 11)
    for ax, (s, m, t) in zip(axes[0], panels, strict=True):
        d = pool[(pool.scen == s) & (pool.model == m) & (pool.task == t)]
        ax.plot([0, 1], [0, 1], color=COL["grid"], lw=1)
        for g in GENS:
            x = d[d.gen == g]
            b = np.clip(np.digitize(x.score, edges) - 1, 0, 9)
            ms, ys, ns = [], [], []
            for k in range(10):
                sel = b == k
                if sel.sum() >= 20:
                    ms.append(x.score[sel].mean())
                    ys.append(x.y[sel].mean())
                    ns.append(sel.sum())
            ax.plot(ms, ys, "-o", color=colors[g], lw=1.5, ms=4, label=f"gen {g}")
        ax.set_title(f"{s} {m}: P({'all founders A' if t == 'all_A' else 'doublet'})",
                     fontsize=9)  # fmt: skip
        ax.set_xlabel("predicted probability (bins of 0.1, ≥ 20 clones)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("observed frequency")
    axes[0][-1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_auc(pt: pd.DataFrame, path: Path):
    panels = [(s, m, t) for s, m, t in (("F70", "fA", "all_A"), ("D15mix", "fA+d2", "all_A"),
                                         ("D15A", "fA+d2", "doublet"),
                                         ("D15mix", "fA+d2", "doublet"))
              if ((pt.scen == s) & (pt.model == m) & (pt.task == t)).any()]  # fmt: skip
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6), squeeze=False,
                             sharey=True)  # fmt: skip
    cmap = plt.get_cmap("Blues")
    for ax, (s, m, t) in zip(axes[0], panels, strict=True):
        d = pt[(pt.scen == s) & (pt.model == m) & (pt.task == t) & (pt.seed == 0)
               & (pt.gen != "all")]  # fmt: skip
        M = np.full((len(GTS), len(GENS)), np.nan)
        for i, g in enumerate(GTS):
            for j, h in enumerate(GENS):
                x = d[(d["gt"] == g) & (d.gen == h)]
                if len(x):
                    M[i, j] = x.auc.iloc[0]
        ax.imshow(M, vmin=0.5, vmax=1.0, cmap=cmap, aspect="auto")
        for i in range(len(GTS)):
            for j in range(len(GENS)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if M[i, j] > 0.8 else COL["ink"])  # fmt: skip
        ax.set_xticks(range(len(GENS)), [f"gen {g}" for g in GENS], fontsize=8)
        ax.set_yticks(range(len(GTS)), GTS, fontsize=8)
        ax.set_title(f"{s} {m}: AUC {'all-A' if t == 'all_A' else 'doublet'}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        sys.exit("no runs")
    print(f"{len(runs)} runs")
    ft = founder_table(runs)
    ft.to_csv(TAB / "founder_posteriors.csv", index=False)
    kt = key_table(runs)
    kt.to_csv(TAB / "key_posteriors.csv", index=False)
    nuts_a = all_a_nuts()
    nuts_a.to_csv(TAB / "allA_nuts_key.csv", index=False)
    ml = mle()
    ml.to_csv(TAB / "mle_key.csv", index=False)
    mt = mixing_table(runs)
    mt.to_csv(TAB / "mixing.csv", index=False)
    pt, pool = progenitor_table(runs)
    pt.to_csv(TAB / "progenitor.csv", index=False)
    pt_mean, _ = progenitor_table(runs, "prog_at_mean")
    pt_mean.to_csv(TAB / "progenitor_at_posterior_mean.csv", index=False)
    pt_or, pool_or = progenitor_table(runs, "oracle")
    pt_or.to_csv(TAB / "progenitor_oracle.csv", index=False)

    # ---- summaries printed for the write-up
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    print("\n== founder parameters (seed 0)")
    print(ft.round(3).to_string(index=False))
    print("\n== key quantities: free-founder posterior, per scenario x model")
    agg = kt[kt.seed == 0].groupby(["scen", "model", "key"]).agg(
        mean_bias=("bias", "mean"), max_abs_bias=("bias", lambda v: np.abs(v).max()),
        mean_abs_z=("z", lambda v: np.abs(v).mean()), max_abs_z=("z", lambda v: np.abs(v).max()),
        cover90=("cover90", "mean"), n=("bias", "size"))  # fmt: skip
    print(agg.round(3).to_string())
    agg.round(4).to_csv(TAB / "key_summary.csv")
    print("\n== all-A NUTS (study), same summaries")
    agg_a = nuts_a[nuts_a.scen.isin(SCEN)].groupby(["scen", "key"]).agg(
        mean_bias=("bias", "mean"), max_abs_bias=("bias", lambda v: np.abs(v).max()),
        mean_abs_z=("z", lambda v: np.abs(v).mean()), max_abs_z=("z", lambda v: np.abs(v).max()),
        cover90=("cover90", "mean"), n=("bias", "size"))  # fmt: skip
    print(agg_a.round(3).to_string())
    agg_a.round(4).to_csv(TAB / "allA_nuts_summary.csv")
    print("\n== MLE (seeds 0-9): mean / max |bias|")
    agg_m = ml.groupby(["scen", "model", "key"]).agg(
        mean_bias=("bias", "mean"), max_abs_bias=("bias", lambda v: np.abs(v).max()),
        n=("bias", "size"))  # fmt: skip
    print(agg_m.round(3).to_string())
    agg_m.round(4).to_csv(TAB / "mle_summary.csv")
    print("\n== per-truth B->A (seed 0): free-founder vs all-A NUTS vs MLE")
    wide = kt[(kt.key == "B.to_A") & (kt.seed == 0)].pivot_table(
        index=["scen", "gt"], columns="model", values=["mean", "z", "cover90"])  # fmt: skip
    print(wide.round(3).to_string())
    print("\n== progenitor posterior (all generations, seed 0)")
    pa = pt[(pt.gen == "all") & (pt.seed == 0)]
    print(pa.groupby(["scen", "model", "task"])[["rate", "mean_score", "auc", "brier",
                                                 "brier_const"]].mean().round(3).to_string())  # fmt: skip
    print("\n== progenitor posterior at the posterior mean / at the truth (oracle)")
    for name, d in (("at posterior mean", pt_mean), ("oracle", pt_or)):
        print(name)
        print(d[(d.gen == "all") & (d.seed == 0)].groupby(["scen", "model", "task"])[
            ["auc", "brier", "brier_const"]].mean().round(3).to_string())  # fmt: skip
    po = pt_or[(pt_or.gen != "all") & (pt_or.seed == 0)].groupby(["scen", "model", "task", "gen"])[
        ["auc", "brier", "brier_const"]].mean()  # fmt: skip
    po.round(4).to_csv(TAB / "progenitor_oracle_by_generation.csv")
    print(po.round(3).to_string())
    print("\n== progenitor AUC by generation (mean over truths)")
    pg = pt[(pt.gen != "all") & (pt.seed == 0)].groupby(["scen", "model", "task", "gen"])[
        ["auc", "brier", "brier_const", "rate"]].mean()  # fmt: skip
    print(pg.round(3).to_string())
    pg.round(4).to_csv(TAB / "progenitor_by_generation.csv")
    print("\n== mixing")
    print(mt.drop(columns=["init_lp_top4"]).round(3).to_string(index=False))

    fig_founder(ft, FIG / "founder_posteriors.png")
    fig_toa(kt, nuts_a, ml, FIG / "b_to_a.png")
    fig_reliability(pool, FIG / "progenitor_reliability.png")
    fig_auc(pt, FIG / "progenitor_auc.png")
    fig_reliability(pool_or, FIG / "progenitor_reliability_oracle.png")


if __name__ == "__main__":
    main()
