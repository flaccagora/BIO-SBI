"""Spike-and-slab / BMA analysis of the XG2 runs from the per-run evidences.

Usage: uv run python scripts/spike_slab/analysis.py
Reads results/exact_gw/spike_slab/runs/*/*/{meta.json,full.npz,M0.npz,...}; writes
docs/spike_slab/tables/*.csv|md and docs/spike_slab/figures/*.png.

Per run and prior weight pi0 = P(M0) in {0.1, 0.5, 0.9}:
  P(M0 | y) = pi0 Z0 / (pi0 Z0 + (1 - pi0) Zfull)
BMA posterior = (1 - P(M0|y)) x [M_full NUTS draws] + P(M0|y) x [M0 draws]; B.div_* are
exactly 0 under M0. Methods compared: flat (M_full posterior), BMA(pi0), M0 alone, MLE,
LRT-selected MLE (chi-bar-squared(3) test of B.div_* = 0 at 5%).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from core import DOCS, FREE12, OUT
from scipy.stats import chi2

LOGZ_KEY = "logZ_main"  # primary evidence estimate stored per model npz
PI0 = (0.1, 0.5, 0.9)
NO_DIV = ("T1_ref", "T2_toA05", "T4_slowA")
GT_ORDER = ["T1_ref", "T2_toA05", "T4_slowA", "T3_Bdiv", "T5_C7", "P1", "P2", "P3", "P4"]
TAB, FIG = DOCS / "tables", DOCS / "figures"


def wquantile(x: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    c = np.cumsum(w)
    c /= c[-1]
    return np.array([x[min(np.searchsorted(c, q, side="left"), len(x) - 1)] for q in qs])


def summarise(draws: np.ndarray, w: np.ndarray) -> dict[str, np.ndarray]:
    w = w / w.sum()
    mean = w @ draws
    sd = np.sqrt(np.maximum(w @ (draws - mean) ** 2, 0))
    q = np.array([wquantile(draws[:, j], w, (0.025, 0.05, 0.95, 0.975))
                  for j in range(draws.shape[1])])  # fmt: skip
    return {"mean": mean, "sd": sd, "q025": q[:, 0], "q05": q[:, 1], "q95": q[:, 2],
            "q975": q[:, 3]}  # fmt: skip


def chibar3_p(T: float) -> float:
    T = max(T, 0.0)
    if T <= 0:
        return 1.0
    return float(3 / 8 * chi2.sf(T, 1) + 3 / 8 * chi2.sf(T, 2) + 1 / 8 * chi2.sf(T, 3))


def load_run(d) -> dict | None:
    meta = json.loads((d / "meta.json").read_text())
    if not all((d / f"{m}.npz").exists() for m in ("full", "M0")):
        return None
    names = meta["names"]
    idx = [names.index(n) for n in FREE12]
    out = {"gt": meta["gt"], "n": meta["n"], "seed": meta["seed"],
           "truth": np.array(meta["truth"])[idx]}  # fmt: skip
    post = pd.read_csv(f"{meta['run']}/posterior_samples.csv.gz")[FREE12].to_numpy()
    for m in ("full", "M0", "nodediff"):
        f = d / f"{m}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        out[f"logZ_{m}"] = float(z[LOGZ_KEY])
        out[f"ll_{m}"] = float(z["ll_mle"])
        out[f"mle_{m}"] = np.asarray(z["mle_phys"])[idx]
        for k in z.files:
            if k.startswith("logZ_") and z[k].ndim == 0:
                out[f"{m}:{k}"] = float(z[k])
        if m == "M0":
            out["draws_M0"] = np.asarray(z["post_phys"], dtype=float)[:, idx]
            out["logw_M0"] = np.asarray(z["post_logw"], dtype=float)
    out["draws_full"] = post
    out["ll_full"] = max(out["ll_full"], meta["ll_mle_stored"])
    return out


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = [r for d in sorted((OUT / "runs").glob("*/*")) if (r := load_run(d)) is not None]
    print(f"{len(runs)} runs", file=sys.stderr)
    rows_run, rows_par = [], []
    for r in runs:
        lbf = r["logZ_M0"] - r["logZ_full"]  # log Bayes factor M0 : full
        T = 2 * (r["ll_full"] - r["ll_M0"])
        p_lrt = chibar3_p(T)
        keep = p_lrt > 0.05
        base = {"gt": r["gt"], "n": r["n"], "seed": r["seed"], "log_bf_M0_full": lbf,
                "lrt_T": T, "lrt_p": p_lrt, "lrt_keeps_M0": keep}  # fmt: skip
        if "logZ_nodediff" in r:
            lbd = r["logZ_nodediff"] - r["logZ_full"]
            base["log_bf_nodediff_full"] = lbd
            base["P_dediff_pi05"] = 1 / (1 + np.exp(np.clip(lbd, -700, 700)))
        for p0 in PI0:
            lo = lbf + np.log(p0 / (1 - p0))
            base[f"P_M0_pi{p0}"] = float(1 / (1 + np.exp(-np.clip(lo, -700, 700))))
        for k, v in r.items():
            if ":" in k:
                base[k] = v
        rows_run.append(base)
        Nf, w0 = len(r["draws_full"]), np.exp(r["logw_M0"] - r["logw_M0"].max())
        w0 /= w0.sum()
        methods = {"flat": (r["draws_full"], np.full(Nf, 1 / Nf)),
                   "M0": (r["draws_M0"], w0)}  # fmt: skip
        for p0 in PI0:
            pm = base[f"P_M0_pi{p0}"]
            methods[f"BMA_pi{p0}"] = (np.vstack([r["draws_full"], r["draws_M0"]]),
                                      np.concatenate([np.full(Nf, (1 - pm) / Nf), pm * w0]))  # fmt: skip
        for meth, (D, W) in methods.items():
            s = summarise(D, W)
            for j, nm in enumerate(FREE12):
                t = r["truth"][j]
                rows_par.append({
                    "gt": r["gt"], "n": r["n"], "seed": r["seed"], "method": meth, "parameter": nm,
                    "truth": t, "estimate": s["mean"][j], "sd": s["sd"][j],
                    "q025": s["q025"][j], "q05": s["q05"][j], "q95": s["q95"][j],
                    "q975": s["q975"][j],
                    "cov90": bool(s["q05"][j] <= t <= s["q95"][j]),
                    "cov95": bool(s["q025"][j] <= t <= s["q975"][j]),
                    "z": (s["mean"][j] - t) / s["sd"][j] if s["sd"][j] > 0 else 0.0,
                })  # fmt: skip
        for meth, est in (("MLE", r["mle_full"]), ("LRT", r["mle_M0"] if keep else r["mle_full"])):
            for j, nm in enumerate(FREE12):
                rows_par.append({"gt": r["gt"], "n": r["n"], "seed": r["seed"], "method": meth,
                                 "parameter": nm, "truth": r["truth"][j], "estimate": est[j]})  # fmt: skip
    run_df = pd.DataFrame(rows_run).sort_values(["gt", "n", "seed"])
    par_df = pd.DataFrame(rows_par)
    par_df["err"] = par_df["estimate"] - par_df["truth"]
    run_df.to_csv(TAB / "per_run.csv", index=False)
    par_df.to_csv(TAB / "per_run_parameter.csv", index=False)
    tables(run_df, par_df)
    figures(run_df, par_df)
    faces8(run_df)


def _md(df: pd.DataFrame, name: str, digits: int = 3) -> None:
    df.to_csv(TAB / f"{name}.csv")
    (TAB / f"{name}.md").write_text(df.round(digits).to_markdown())


def _gt_cat(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gt"] = pd.Categorical(df["gt"], [g for g in GT_ORDER if g in set(df["gt"])], ordered=True)
    return df


def agg(g: pd.DataFrame) -> pd.Series:
    out = {"runs": len(g), "rmse": np.sqrt((g["err"] ** 2).mean()), "bias": g["err"].mean()}
    if g["sd"].notna().any():
        out.update(cov90=g["cov90"].mean(), cov95=g["cov95"].mean(), mean_z=g["z"].mean(),
                   mean_sd=g["sd"].mean())  # fmt: skip
    return pd.Series(out)


MAIN = ["flat", "BMA_pi0.5", "M0", "MLE", "LRT"]


def tables(run_df: pd.DataFrame, par_df: pd.DataFrame) -> None:
    r = _gt_cat(run_df)
    # P(M0 | y) by truth and n (mean over datasets; also min / max)
    for p0 in PI0:
        t = r.groupby(["gt", "n"], observed=True)[f"P_M0_pi{p0}"].mean().unstack()
        _md(t, f"pm0_mean_pi{p0}")
    t = r.groupby(["gt", "n"], observed=True)["P_M0_pi0.5"].agg(["min", "mean", "max"])
    t["log_bf_mean"] = r.groupby(["gt", "n"], observed=True)["log_bf_M0_full"].mean()
    t["lrt_keeps_M0"] = r.groupby(["gt", "n"], observed=True)["lrt_keeps_M0"].mean()
    t["runs"] = r.groupby(["gt", "n"], observed=True).size()
    _md(t, "pm0_by_truth_n")
    if "P_dediff_pi05" in r:
        _md(r.groupby(["gt", "n"], observed=True)["P_dediff_pi05"].min().unstack(), "p_dediff_min")
        _md(r.groupby(["gt", "n"], observed=True)["log_bf_nodediff_full"].max().unstack(),
            "log_bf_nodediff_max", 1)  # fmt: skip
    p = _gt_cat(par_df)
    p["cls"] = np.where(p["gt"].isin(NO_DIV), "B never divides (T1,T2,T4)", "B divides (others)")
    bt = p[p["parameter"] == "B.to_A"]
    t = bt[bt["method"].isin(MAIN)].groupby(["gt", "n", "method"], observed=True).apply(
        agg, include_groups=False)
    _md(t, "btoa_by_truth_n_method")
    t = bt[bt["method"].isin(MAIN)].groupby(["cls", "n", "method"], observed=True).apply(
        agg, include_groups=False)
    _md(t, "btoa_by_class_n_method")
    t = bt.groupby(["cls", "method"], observed=True).apply(agg, include_groups=False)
    _md(t, "btoa_by_class_method")
    # mean z (sd units) of B.to_A by n: flat vs BMA, for the no-division truths
    z = bt[bt["method"].isin(["flat", "BMA_pi0.5", "M0"])].groupby(
        ["gt", "method", "n"], observed=True)["z"].mean().unstack()
    _md(z, "btoa_mean_z")
    # all 12 probabilities: averaged over parameters
    t = p[p["method"].isin(MAIN + ["BMA_pi0.1", "BMA_pi0.9"])].groupby(
        ["cls", "method"], observed=True).apply(agg, include_groups=False)
    _md(t, "all12_by_class_method")
    t = p[p["method"].isin(MAIN)].groupby(["cls", "parameter", "method"], observed=True).apply(
        agg, include_groups=False)
    _md(t, "all12_by_class_parameter_method")
    t = p[p["method"].isin(MAIN)].groupby(["gt", "method"], observed=True).apply(
        agg, include_groups=False)
    _md(t, "all12_by_truth_method")
    # prior-weight sensitivity (B.to_A and all 12)
    sens = p[p["method"].str.startswith("BMA") | (p["method"] == "flat")]
    t = sens[sens["parameter"] == "B.to_A"].groupby(["cls", "n", "method"], observed=True).apply(
        agg, include_groups=False)
    _md(t, "sensitivity_btoa")
    # wrong-model risk for dividing truths: P(M0 | y) > 0.5
    r["cls"] = np.where(r["gt"].isin(NO_DIV), "no-div", "div")
    t = r.groupby(["cls", "n"], observed=True).agg(
        **{f"frac_P_M0>0.5_pi{p0}": (f"P_M0_pi{p0}", lambda x: (x > 0.5).mean()) for p0 in PI0},
        lrt_keeps=("lrt_keeps_M0", "mean"), runs=("gt", "size"))
    _md(t, "decisions_by_class_n")


def figures(run_df: pd.DataFrame, par_df: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_FLAT, C_BMA, C_M0, C_LRT = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6})
    gts = [g for g in GT_ORDER if g in set(run_df["gt"])]
    # 1. P(M0 | y) vs n
    fig, axes = plt.subplots(3, 3, figsize=(9, 7.5), sharex=True, sharey=True)
    for ax, g in zip(axes.flat, gts, strict=False):
        d = run_df[run_df["gt"] == g]
        ns = sorted(d["n"].unique())
        for p0, ls in ((0.1, ":"), (0.9, "--")):
            m = d.groupby("n")[f"P_M0_pi{p0}"].mean()
            ax.plot(m.index, m.values, ls, color="#8a8984", lw=1.2, label=f"P(M0) = {p0}, mean")
        jit = {s: 1 + 0.06 * (s - 1) for s in range(3)}
        ax.scatter(d["n"] * d["seed"].map(jit), d["P_M0_pi0.5"], s=16, color=C_BMA, zorder=3,
                   label="P(M0) = 0.5, per dataset")  # fmt: skip
        m = d.groupby("n")["P_M0_pi0.5"].mean()
        ax.plot(m.index, m.values, "-", color=C_BMA, lw=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(x) for x in ns], rotation=45)
        ax.set_title(g + ("  (B never divides)" if g in NO_DIV else ""), fontsize=9)
        ax.set_ylim(-0.03, 1.03)
    for ax in axes.flat[len(gts):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.supxlabel("clones per endpoint n")
    fig.supylabel("posterior probability of M0 (B never divides)")
    fig.tight_layout()
    fig.savefig(FIG / "pm0_vs_n.png", dpi=180)
    plt.close(fig)
    # 2. B.to_A: flat vs BMA vs truth
    bt = par_df[par_df["parameter"] == "B.to_A"]
    fig, axes = plt.subplots(3, 3, figsize=(9, 7.5), sharex=True)
    for ax, g in zip(axes.flat, gts, strict=False):
        for meth, col, off in (("flat", C_FLAT, 0.92), ("BMA_pi0.5", C_BMA, 1.08)):
            d = bt[(bt["gt"] == g) & (bt["method"] == meth)]
            for s in range(3):
                e = d[d["seed"] == s]
                x = e["n"] * off * (1 + 0.025 * (s - 1))
                ax.errorbar(x, e["estimate"], yerr=[e["estimate"] - e["q05"], e["q95"] - e["estimate"]],
                            fmt="o", ms=3.5, color=col, elinewidth=0.9, capsize=0,
                            label=("flat posterior" if meth == "flat" else "BMA, P(M0) = 0.5")
                            if s == 0 else None)  # fmt: skip
        d = bt[(bt["gt"] == g) & (bt["method"] == "LRT")]
        ax.scatter(d["n"], d["estimate"], marker="x", s=14, color=C_LRT, zorder=4,
                   label="LRT-selected MLE")  # fmt: skip
        truth = bt[bt["gt"] == g]["truth"].iloc[0]
        ax.axhline(truth, color="#0b0b0b", ls="--", lw=1, label="truth")
        ns = sorted(bt[bt["gt"] == g]["n"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(x) for x in ns], rotation=45)
        ax.set_title(g, fontsize=9)
        ax.set_ylim(-0.02, 1.0)
    for ax in axes.flat[len(gts):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.supxlabel("clones per endpoint n")
    fig.supylabel("B.to_A: posterior mean and 90% interval")
    fig.tight_layout()
    fig.savefig(FIG / "btoa_bma_vs_flat.png", dpi=180)
    plt.close(fig)
    # 3. calibration of B.to_A: coverage 95% and mean z by method, class and n
    p = par_df[(par_df["parameter"] == "B.to_A") & par_df["method"].isin(["flat", "BMA_pi0.5", "M0"])].copy()
    p["cls"] = np.where(p["gt"].isin(NO_DIV), "B never divides (T1, T2, T4)", "B divides (T3, T5, P1-P4)")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for j, cls in enumerate(sorted(p["cls"].unique(), reverse=True)):
        d = p[p["cls"] == cls]
        for meth, col, lab in (("flat", C_FLAT, "flat posterior"), ("BMA_pi0.5", C_BMA, "BMA, P(M0) = 0.5"),
                               ("M0", C_M0, "M0 alone")):
            e = d[d["method"] == meth].groupby("n").agg(cov=("cov95", "mean"), z=("z", "mean"))
            axes[0, j].plot(e.index, e["cov"], "o-", color=col, lw=2, ms=4, label=lab)
            axes[1, j].plot(e.index, e["z"], "o-", color=col, lw=2, ms=4, label=lab)
        axes[0, j].axhline(0.95, color="#0b0b0b", ls="--", lw=1)
        axes[1, j].axhline(0, color="#0b0b0b", ls="--", lw=1)
        axes[0, j].set_title(cls, fontsize=9)
        axes[0, j].set_ylim(-0.03, 1.03)
        for ax in axes[:, j]:
            ax.set_xscale("log", base=2)
            ns = sorted(d["n"].unique())
            ax.set_xticks(ns)
            ax.set_xticklabels([str(x) for x in ns], rotation=45)
    axes[0, 0].set_ylabel("coverage of 95% interval, B.to_A")
    axes[1, 0].set_ylabel("mean z = (mean - truth) / sd, B.to_A")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.supxlabel("clones per endpoint n")
    fig.tight_layout()
    fig.savefig(FIG / "calibration_btoa.png", dpi=180)
    plt.close(fig)


def faces8(run_df: pd.DataFrame) -> None:
    """Uniform prior over the 8 division faces (each subset of B.div_* pinned at 0)."""
    from core import FACES8

    rows = []
    for d in sorted((OUT / "runs").glob("*/*")):
        if not all((d / f"{m}.npz").exists() for m in FACES8):
            continue
        meta = json.loads((d / "meta.json").read_text())
        names = meta["names"]
        idx = [names.index(n) for n in FREE12]
        truth = np.array(meta["truth"])[idx]
        lz = np.array([float(np.load(d / f"{m}.npz")[LOGZ_KEY]) for m in FACES8])
        pm = np.exp(lz - lz.max())
        pm /= pm.sum()
        Ds, Ws = [], []
        for m, p in zip(FACES8, pm, strict=True):
            if m == "full":
                D = pd.read_csv(f"{meta['run']}/posterior_samples.csv.gz")[FREE12].to_numpy()
            else:
                D = np.asarray(np.load(d / f"{m}.npz")["post_phys"], dtype=float)[:, idx]
            Ds.append(D)
            Ws.append(np.full(len(D), p / len(D)))
        s = summarise(np.vstack(Ds), np.concatenate(Ws))
        j = FREE12.index("B.to_A")
        r2 = run_df[(run_df["gt"] == meta["gt"]) & (run_df["n"] == meta["n"])
                    & (run_df["seed"] == meta["seed"])]  # fmt: skip
        row = {"gt": meta["gt"], "n": meta["n"], "seed": meta["seed"],
               **{f"P_{m}": p for m, p in zip(FACES8, pm, strict=True)},
               "P_M0_two_model": float(r2["P_M0_pi0.5"].iloc[0]) if len(r2) else np.nan,
               "btoa_truth": truth[j], "btoa_mean": s["mean"][j], "btoa_sd": s["sd"][j],
               "btoa_cov95": bool(s["q025"][j] <= truth[j] <= s["q975"][j]),
               "btoa_z": (s["mean"][j] - truth[j]) / s["sd"][j]}  # fmt: skip
        for k in ("B.div_AA", "B.div_AB", "B.div_BB"):
            jk = FREE12.index(k)
            row[f"{k}_mean"], row[f"{k}_truth"] = s["mean"][jk], truth[jk]
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows).sort_values(["gt", "n", "seed"]).set_index(["gt", "n", "seed"])
        _md(df, "faces8_per_run")


if __name__ == "__main__":
    main()
