"""Simulation check of the founder-distribution study: NUTS (dataset 0) and MLE (datasets 0-5).

Tables (``docs/founder_distribution/tables``):
* ``mle_summary``: bias / RMSE of B->A, m_BA, rho, stable A fraction, p_A, mean founder size
  (best of the multistart fits) per truth, founder distribution and model (est / known),
  with the Fisher-predicted sd (XG2 design, N = 2048; eps variant for T1);
* ``nuts_runs``: posterior mean, sd, 90% coverage, z, mixing flags per NUTS run, and the
  Fisher-predicted sd;
* ``progenitor``: per-clone founder-count and composition posterior (AUC, Brier vs a
  constant, posterior mean count vs truth), fitted (NUTS average / MLE plug-in) and oracle.

  uv run python scripts/founder_distribution/analyse_sim.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fd_common as F  # noqa: E402
from figures import GRID, INK, INK2, SERIES, style  # noqa: E402

RES = F.RESULTS
KEYS = ["B.to_A", "m_BA", "rho", "stableA", "founder.A", "mean_size"]
SHORT = {"T1_ref": "T1", "P3": "P3", "T3_Bdiv": "T3"}


def fisher_sd() -> pd.DataFrame:
    df = pd.read_csv(F.TAB / "fisher_main.csv")
    df = df[(df.design == "XG2") & (df["mode"].isin(["K", "E"]))]
    df = df.rename(columns={"sd_B.to_A": "B.to_A", "sd_m_BA": "m_BA", "sd_rho": "rho",
                            "sd_stableA": "stableA", "sd_pA_joint": "founder.A",
                            "sd_mean_size_joint": "mean_size"})  # fmt: skip
    df["model"] = df["mode"].map({"K": "known", "E": "est"})
    for k in ("B.to_A", "m_BA", "rho"):
        df[f"prior_{k}"] = df[f"sdprior_{k}"]
    return df.set_index(["truth", "dist", "model"])[KEYS + ["prior_B.to_A", "prior_m_BA",
                                                            "prior_rho"]]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def progenitor_metrics(z) -> dict:
    labels = [str(s) for s in z["labels"]]
    m = np.array([len(s) for s in labels])
    nA = np.array([s.count("A") for s in labels])
    size, nA0 = z["clones"][:, 3], z["clones"][:, 4]
    out = {}
    for tag, P in (("fit", z["prog"].astype(float)), ("oracle", z["prog_true"].astype(float))):
        p_multi = P[:, m > 1].sum(1)
        p_allA = P[:, nA == m].sum(1)
        mean_m = P @ m
        y_multi, y_allA = size > 1, nA0 == size
        for name, p, y in (("multi", p_multi, y_multi), ("allA", p_allA, y_allA)):
            out[f"auc_{name}_{tag}"] = auc(p, y)
            out[f"brier_{name}_{tag}"] = float(np.mean((p - y) ** 2))
            out[f"brier_{name}_const"] = float(np.mean((y.mean() - y) ** 2))
            out[f"mean_p_{name}_{tag}"] = float(p.mean())
            out[f"rate_{name}"] = float(y.mean())
        out[f"corr_count_{tag}"] = (float(np.corrcoef(mean_m, size)[0, 1])
                                    if size.std() > 0 else np.nan)  # fmt: skip
        out[f"rmse_count_{tag}"] = float(np.sqrt(np.mean((mean_m - size) ** 2)))
    return out


def reliability(z, bins=np.linspace(0, 1, 6)) -> list:
    labels = [str(s) for s in z["labels"]]
    m = np.array([len(s) for s in labels])
    p = z["prog"].astype(float)[:, m > 1].sum(1)
    y = z["clones"][:, 3] > 1
    out = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        s = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if s.sum() >= 20:
            out.append((float(p[s].mean()), float(y[s].mean()), int(s.sum())))
    return out


def mle_table(fs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, prog = [], []
    for f in sorted((RES / "mle").glob("*/*/*.json")):
        r = json.loads(f.read_text())
        b = r["best"]
        row = {"truth": SHORT[r["gt"]], "dist": r["dist"], "model": r["model"], "seed": r["seed"],
               "ll_best_minus_truth": b["ll"] - r["ll_truth"],
               "n_branches": len({round(o["founder.A"], 1) for o in r["optima"]
                                  if b["ll"] - o["ll"] < 2})}  # fmt: skip
        for k in KEYS:
            row[f"err_{k}"] = r["err"].get(k, np.nan) if k in r["err"] else np.nan
            row[k] = b.get(k, np.nan)
        rows.append(row)
        z = np.load(f.with_suffix(".npz"))
        prog.append({"truth": row["truth"], "dist": r["dist"], "model": r["model"],
                     "seed": r["seed"], "method": "MLE plug-in", **progenitor_metrics(z)})  # fmt: skip
    d = pd.DataFrame(rows)
    g = d.groupby(["truth", "dist", "model"])
    s = g.agg(n=("seed", "size")).reset_index()
    s["frac_BtoA_off_0.3"] = g["err_B.to_A"].apply(lambda v: float(np.mean(np.abs(v) > 0.3))).values
    for k in KEYS:
        s[f"bias_{k}"] = g[f"err_{k}"].mean().values
        s[f"rmse_{k}"] = g[f"err_{k}"].apply(lambda v: float(np.sqrt(np.mean(v**2)))).values
        s[f"fisher_sd_{k}"] = [fs.loc[(t, dd, m), k] if (t, dd, m) in fs.index else np.nan
                               for t, dd, m in zip(s.truth, s.dist, s.model, strict=True)]  # fmt: skip
    return d, s, pd.DataFrame(prog)


def multi_table() -> pd.DataFrame:
    """Joint MLE of two conditions (0.9 + 0.3) vs one condition at p_A 0.6 (multi.py)."""
    rows = []
    for f in sorted((RES / "multi").glob("*/*.json")):
        r = json.loads(f.read_text())
        row = {"setup": r["setup"], "truth": SHORT[r["gt"]], "seed": r["seed"],
               **{f"err_{k}": v for k, v in r["err"].items()}}  # fmt: skip
        for i, (c, e) in enumerate(zip(r["conditions"], r["err_founder"], strict=True)):
            row[f"err_pA_{i + 1}"] = e
            row[f"mean_size_{i + 1}"] = c["mean_size"]
        rows.append(row)
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    out = []
    for (setup, t), x in d.groupby(["setup", "truth"]):
        o = {"setup": setup, "truth": t, "n": len(x)}
        for k in ("B.to_A", "m_BA", "rho"):
            o[f"bias_{k}"] = x[f"err_{k}"].mean()
            o[f"rmse_{k}"] = float(np.sqrt(np.mean(x[f"err_{k}"] ** 2)))
        o["frac_BtoA_off_0.3"] = float(np.mean(np.abs(x["err_B.to_A"]) > 0.3))
        for i in (1, 2):
            if f"err_pA_{i}" in x and x[f"err_pA_{i}"].notna().any():
                o[f"rmse_pA_{i}"] = float(np.sqrt(np.mean(x[f"err_pA_{i}"] ** 2)))
                o[f"mean_size_{i}"] = x[f"mean_size_{i}"].mean()
        out.append(o)
    return pd.DataFrame(out)


def mixing_flags(c: dict) -> list[str]:
    flags = []
    if (c.get("max_rank_rhat") or 0) > 1.05:
        flags.append(f"R-hat {c['max_rank_rhat']:.2f}")
    if (c.get("min_ess_bulk") or 1e9) < 100:
        flags.append(f"ESS {c['min_ess_bulk']:.0f}")
    if (c.get("max_mean_spread") or 0) > 0.5:
        flags.append(f"spread {c['max_mean_spread']:.2f}")
    return flags


def nuts_table(fs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, prog = [], []
    for f in sorted((RES / "nuts").glob("*/*/*.json")):
        r = json.loads(f.read_text())
        t = SHORT[r["gt"]]
        c = r["convergence"]
        fl = mixing_flags(c)
        for k in KEYS:
            st = r["stats"].get(k)
            if st is None or st.get("truth") is None:
                continue
            fisher = fs.loc[(t, r["dist"], r["model"]), k] if (t, r["dist"], r["model"]) in fs.index \
                else np.nan  # fmt: skip
            rows.append({"truth": t, "dist": r["dist"], "model": r["model"], "param": k,
                         "truth_value": st["truth"], "mean": st["mean"], "sd": st["sd"],
                         "q05": st["q05"], "q95": st["q95"], "z": st.get("z"),
                         "cover90": st.get("cover90"), "fisher_sd": fisher,
                         "fisher_sd_with_prior": fs.loc[(t, r["dist"], r["model"]), f"prior_{k}"]
                         if (t, r["dist"], r["model"]) in fs.index and f"prior_{k}" in fs.columns
                         else np.nan,
                         "mixing": "poor: " + ", ".join(fl) if fl else "ok",
                         "minutes": r["seconds"] / 60, "draws": r["draws"],
                         "divergences": c.get("divergences")})  # fmt: skip
        z = np.load(f.with_suffix(".npz"))
        prog.append({"truth": t, "dist": r["dist"], "model": r["model"], "seed": r["seed"],
                     "method": "NUTS average", **progenitor_metrics(z),
                     "reliability_multi": reliability(z)})  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(prog)


def fig_nuts(nt: pd.DataFrame, md: pd.DataFrame) -> None:
    if nt.empty:
        return
    params = [("B.to_A", "B→A"), ("m_BA", "m_BA"), ("rho", "growth rate ρ"),
              ("founder.A", "p_A"), ("mean_size", "mean founder size")]  # fmt: skip
    runs = nt[["truth", "dist"]].drop_duplicates().sort_values(["truth", "dist"])
    runs = list(runs.itertuples(index=False))
    fig, axes = plt.subplots(1, len(params), figsize=(15, 0.42 * len(runs) + 1.6),
                             sharey=True, constrained_layout=True)  # fmt: skip
    for ax, (p, lab) in zip(axes, params, strict=True):
        for i, (t, d) in enumerate(runs):
            for j, (model, col) in enumerate((("known", SERIES[0]), ("est", SERIES[1]))):
                s = nt[(nt.truth == t) & (nt.dist == d) & (nt.model == model) & (nt.param == p)]
                y = i + (j - 0.5) * 0.3
                if len(s):
                    s = s.iloc[0]
                    ax.plot([s.q05, s.q95], [y, y], color=col, lw=2, solid_capstyle="round")
                    ax.plot(s["mean"], y, "o", color=col, ms=4.5,
                            mfc="white" if s.mixing != "ok" else col)  # fmt: skip
                    ax.plot(s.truth_value, i, "|", color=INK, ms=11, mew=1.6)
                m = md[(md.truth == t) & (md.dist == d) & (md.model == model)]
                if len(m) and p in m:
                    ax.plot(m[p], np.full(len(m), y), ".", color=col, ms=3, alpha=0.5)
        ax.set_title(lab)
        ax.grid(axis="x", color=GRID, lw=0.6)
    axes[0].set_yticks(range(len(runs)), [f"{t} {d.replace('_pA', ' p_A ')}" for t, d in runs])
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=SERIES[0], lw=2, marker="o", label="founder distribution known"),
         Line2D([], [], color=SERIES[1], lw=2, marker="o", label="estimated"),
         Line2D([], [], color=INK2, lw=0, marker="o", mfc="white", label="poorly mixed run"),
         Line2D([], [], color=INK2, lw=0, marker=".", label=f"MLE, datasets 0–{int(md.seed.max())}"),
         Line2D([], [], color=INK, lw=0, marker="|", ms=10, label="truth")]  # fmt: skip
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle("Simulation check (n = 512 per endpoint 4/6/8/10): NUTS 90% intervals "
                 "(dataset 0) and MLEs", x=0.01, ha="left", fontsize=10, fontweight="bold")  # fmt: skip
    fig.savefig(F.FIG / "nuts_check.png")
    plt.close(fig)


def md(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False, floatfmt=".3g") + "\n")


def main() -> None:
    style()
    fs = fisher_sd()
    mle_d, mle_s, mle_p = mle_table(fs)
    mle_d.to_csv(F.TAB / "mle_datasets.csv", index=False)
    mle_s.to_csv(F.TAB / "mle_summary.csv", index=False)
    cols = ["truth", "dist", "model", "n", "frac_BtoA_off_0.3"] + [f"{a}_{k}" for k in ("B.to_A", "m_BA", "rho",
                                                                    "founder.A", "mean_size")
                                              for a in ("bias", "rmse", "fisher_sd")]  # fmt: skip
    md(mle_s[cols], F.TAB / "mle_summary.md")
    nt, nuts_p = nuts_table(fs)
    if not nt.empty:
        nt.to_csv(F.TAB / "nuts_runs.csv", index=False)
        md(nt.drop(columns=["draws"]), F.TAB / "nuts_runs.md")
    prog = pd.concat([nuts_p, mle_p], ignore_index=True)
    pc = ["auc_multi_fit", "auc_multi_oracle", "brier_multi_fit", "brier_multi_oracle",
          "brier_multi_const", "auc_allA_fit", "auc_allA_oracle", "brier_allA_fit",
          "brier_allA_oracle", "brier_allA_const", "mean_p_allA_fit", "rate_allA",
          "corr_count_fit", "corr_count_oracle"]  # fmt: skip
    pg = prog.groupby(["method", "truth", "dist", "model"])[pc].mean().reset_index()
    pg.to_csv(F.TAB / "progenitor.csv", index=False)
    md(pg, F.TAB / "progenitor.md")
    if "reliability_multi" in prog:
        prog[prog.method == "NUTS average"][["truth", "dist", "model", "reliability_multi"]].to_json(
            F.TAB / "progenitor_reliability.json", orient="records", indent=1)
    mm = multi_table()
    if not mm.empty:
        mm.to_csv(F.TAB / "multi_mle.csv", index=False)
        md(mm, F.TAB / "multi_mle.md")
        print(mm.round(3).to_string())
    fig_nuts(nt, mle_d)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(mle_s[cols].round(3).to_string())
    print(pg.round(3).to_string())
    if not nt.empty:
        print(nt.round(3).to_string())


if __name__ == "__main__":
    main()
