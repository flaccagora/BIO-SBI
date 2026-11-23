"""Tables and figures of the misspecification study.

Reads ``results/exact_gw/misspecification/{fits,dediff,nuts}``; writes CSV + markdown tables
to ``docs/misspecification/tables`` and PNG figures to ``docs/misspecification/figures``.

    uv run python scripts/misspecification/analyse.py
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

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

DOC = C.ROOT / "docs" / "misspecification"
TAB, FIG = DOC / "tables", DOC / "figures"
SCEN = list(C.SCENARIOS)
PARS = C.NAMES + ["rho", "stableA", "m_BA", "B.div"]
KEY = ["B.to_A", "m_BA", "B.div", "rho", "stableA"]


def load_fits() -> pd.DataFrame:
    rows = []
    for f in sorted((C.OUT / "fits").glob("*/*/s*.json")):
        r = json.loads(f.read_text())
        # extended models from the grid-method job (the fit job's own were superseded)
        r.pop("ext", None)
        r.pop("ext_null", None)
        fe = C.OUT / "fits_ext" / r["scen"] / r["gt"] / f.name
        if fe.exists():
            e = json.loads(fe.read_text())
            r.update({k: e[k] for k in ("ext", "ext_null") if k in e})
        truth = {**r["truth"], **r["truth_summ"]}
        for model in ("std", "true", "ext"):
            if model not in r:
                continue
            x = r[model]
            est = {**x["P"], **x["summ"]}
            row = {"scen": r["scen"], "gt": r["gt"], "seed": r["seed"], "model": model}
            for p in PARS:
                row[p] = est[p]
                row[f"t_{p}"] = truth[p]
            if "gof" in x:
                row["gof_p"] = x["gof"]["p"]
                row["gof_p_upper"] = x["gof"]["p_upper"]
                row["gof_G2"] = x["gof"]["G2"]
                row["gof_df"] = x["gof"]["df"]
            for fn, key in (("B.to_A", "toA"), ("B.m_BA", "mBA")):
                pr = (x.get("profile") or {}).get(fn)
                if pr:
                    t = truth["B.to_A" if fn == "B.to_A" else "m_BA"]
                    lo = pr["lo"] if pr["lo"] is not None else np.nan
                    hi = pr["hi"] if pr["hi"] is not None else np.nan
                    row[f"{key}_lo"], row[f"{key}_hi"] = lo, hi
                    row[f"{key}_cover"] = float(lo - 1e-9 <= t <= hi + 1e-9) if np.isfinite(
                        lo + hi) else np.nan
                    row[f"{key}_width"] = hi - lo
            if model == "ext":
                row["ext_kind"] = x["kind"]
                row["ext_lr"] = x["lr"]
                row["ext_p"] = x["p"]
                for k, v in x["x"].items():
                    row[f"x_{k}"] = v
            rows.append(row)
        if "ext_null" in r:
            for kind, e in r["ext_null"].items():
                rows.append({"scen": "base", "gt": r["gt"], "seed": r["seed"],
                             "model": f"extnull_{kind}", "ext_lr": e["lr"], "ext_p": e["p"]})
    return pd.DataFrame(rows)


def load_dediff() -> pd.DataFrame:
    rows = []
    for f in sorted((C.OUT / "dediff").glob("*/*/s*.json")):
        r = json.loads(f.read_text())
        for model in ("std", "true"):
            if model in r:
                x = r[model]
                rows.append({"scen": r["scen"], "null": r["null"], "seed": r["seed"],
                             "model": model, "lambda": x["lambda"], "p_chibar": x["p_chibar"],
                             "p_chi2_3": x["p_chi2_3"], "m_ba_hat": x["m_ba_hat"]})
    return pd.DataFrame(rows)


def load_nuts() -> pd.DataFrame:
    rows = []
    for f in sorted((C.OUT / "nuts").glob("*/*.json")):
        r = json.loads(f.read_text())
        for k, s in r["stats"].items():
            rows.append({"scen": r["scen"], "gt": r["gt"], "par": k, **s,
                         "max_rhat": (r.get("convergence") or {}).get("max_rhat")})
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, name: str, floatfmt=".3g"):
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=floatfmt) + "\n")


def binom_ci(k, n):
    from scipy.stats import beta

    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    F = load_fits()
    D = load_dediff()
    S = load_nuts()
    F.to_csv(C.OUT / "fits_long.csv", index=False)
    D.to_csv(C.OUT / "dediff_long.csv", index=False)
    fits = F[F.model.isin(["std", "true", "ext"])].copy()
    for p in PARS:
        fits[f"e_{p}"] = fits[p] - fits[f"t_{p}"]
    fits["e_rho_rel"] = fits["rho"] / fits["t_rho"] - 1

    # ---- GOF calibration (size on the correctly specified data)
    # calibration uses every base dataset (incl. the GOF-only seeds >= 100); everything else
    # only the main seeds (< 100)
    base = fits[(fits.scen == "base") & (fits.model == "std")]
    size_p = float(np.mean(base.gof_p < 0.05))
    size_upper = float(np.mean(base.gof_p_upper < 0.05))
    size_by_truth = base.groupby("gt").gof_p.apply(lambda v: float(np.mean(v < np.quantile(base.gof_p, 0.05))))
    # empirical 5% critical value of the p-value under H0 (calibrated test)
    p_crit = float(np.quantile(base.gof_p, 0.05)) if len(base) >= 20 else 0.05
    pd.DataFrame({"gt": size_by_truth.index, "size_calibrated": size_by_truth.values,
                  "n": base.groupby("gt").size().values}).to_csv(TAB / "gof_size_by_truth.csv",
                                                                 index=False)
    (TAB / "gof_calibration.md").write_text(
        f"base datasets: {len(base)}; size at p<0.05 with chi2(K-10): {size_p:.3f}; "
        f"with chi2(K) (Chernoff-Lehmann upper): {size_upper:.3f}; calibrated threshold "
        f"(5% quantile of base p-values) p < {p_crit:.4f}\n\n"
        + size_by_truth.to_frame("size at calibrated threshold").to_markdown() + "\n")
    fits = fits[fits.seed < 100]

    # ---- per scenario x truth
    g = fits.groupby(["scen", "model", "gt"])
    per = g.agg(n=("seed", "size"),
                **{f"bias_{p}": (f"e_{p}", "mean") for p in KEY},
                **{f"rmse_{p}": (f"e_{p}", lambda v: float(np.sqrt(np.mean(v ** 2)))) for p in KEY},
                bias_rho_rel=("e_rho_rel", "mean"),
                cover_toA=("toA_cover", "mean"), cover_mBA=("mBA_cover", "mean"),
                width_toA=("toA_width", "median"),
                gof_power=("gof_p", lambda v: float(np.mean(v < p_crit))),
                gof_power_cons=("gof_p_upper", lambda v: float(np.mean(v < 0.05))),
                ).reset_index()  # fmt: skip
    per["scen"] = pd.Categorical(per.scen, SCEN, ordered=True)
    per["gt"] = pd.Categorical(per["gt"], C.GTS, ordered=True)
    per = per.sort_values(["scen", "model", "gt"])
    md(per, "per_scenario_truth")

    # ---- per scenario (over truths)
    rows = []
    for (scen, model), d in fits.groupby(["scen", "model"]):
        b = d.groupby("gt")
        r = {"scenario": scen, "model": model, "what": C.SCENARIOS[scen][0], "datasets": len(d),
             "bias_toA_median": float(b.e_B_to_A.mean().median()) if "e_B_to_A" in d else None}
        for p in KEY:
            bt = b[f"e_{p}"].mean()
            r[f"bias_{p}_mean"] = float(bt.mean())
            r[f"absbias_{p}_max"] = float(bt.abs().max())
            r[f"rmse_{p}"] = float(np.sqrt(np.mean(d[f"e_{p}"] ** 2)))
        r.pop("bias_toA_median")
        r["bias_rho_rel_pct"] = 100 * float(b.e_rho_rel.mean().mean())
        r["cover95_toA"] = float(d.toA_cover.mean())
        r["cover95_mBA"] = float(d.mBA_cover.mean())
        r["gof_power"] = float(np.mean(d.gof_p < p_crit))
        r["gof_power_cons"] = float(np.mean(d.gof_p_upper < 0.05))
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ["scenario"] = pd.Categorical(summ.scenario, SCEN, ordered=True)
    summ = summ.sort_values(["scenario", "model"])
    # RMSE relative to base (std)
    bstd = summ[(summ.scenario == "base") & (summ.model == "std")].iloc[0]
    for p in KEY:
        summ[f"rmse_ratio_{p}"] = summ[f"rmse_{p}"] / bstd[f"rmse_{p}"]
    # extended-model LRT power
    ext = fits[fits.model == "ext"].groupby("scen").agg(
        ext_kind=("ext_kind", "first"), ext_power=("ext_p", lambda v: float(np.mean(v < 0.05))),
        ext_lr_median=("ext_lr", "median")).reset_index()
    extnull = F[F.model.str.startswith("extnull_", na=False)].groupby("model").agg(
        size=("ext_p", lambda v: float(np.mean(v < 0.05))), n=("ext_p", "size")).reset_index()
    md(extnull, "extended_lrt_size_on_base")
    # dediff
    if len(D):
        dd = D.groupby(["scen", "null", "model"]).agg(
            n=("seed", "size"), fpr=("p_chibar", lambda v: float(np.mean(v < 0.05))),
            fpr_chi2_3=("p_chi2_3", lambda v: float(np.mean(v < 0.05))),
            lambda_median=("lambda", "median"), m_ba_hat_median=("m_ba_hat", "median"),
        ).reset_index()
        dd["ci_lo"], dd["ci_hi"] = zip(*[binom_ci(round(f * n), n) for f, n in zip(dd.fpr, dd.n)])
        dd["scen"] = pd.Categorical(dd.scen, SCEN, ordered=True)
        dd = dd.sort_values(["scen", "null", "model"])
        md(dd, "dediff_fpr")
    else:
        dd = pd.DataFrame()

    # ---- main scenario table (standard fit; fixes in separate columns)
    main_rows = []
    for scen in SCEN:
        s = summ[(summ.scenario == scen) & (summ.model == "std")]
        if not len(s):
            continue
        s = s.iloc[0]
        r = {"scenario": scen, "what": C.SCENARIOS[scen][0],
             "bias B.to_A (mean over truths)": s["bias_B.to_A_mean"],
             "max |bias| B.to_A": s["absbias_B.to_A_max"],
             "RMSE B.to_A / base": s["rmse_ratio_B.to_A"],
             "bias m_BA": s["bias_m_BA_mean"],
             "rho bias %": s["bias_rho_rel_pct"],
             "bias stable A frac": s["bias_stableA_mean"],
             "cover95 B.to_A": s["cover95_toA"], "cover95 m_BA": s["cover95_mBA"],
             "GOF power (calibrated)": s["gof_power"],
             "GOF power (conservative)": s["gof_power_cons"]}  # fmt: skip
        e = ext[ext.scen == scen]
        r["extended-LRT power"] = float(e.ext_power.iloc[0]) if len(e) else np.nan
        if len(dd):
            for nl in ("N1", "N2"):
                x = dd[(dd.scen == scen) & (dd.null == nl) & (dd.model == "std")]
                r[f"dediff FPR {nl}"] = float(x.fpr.iloc[0]) if len(x) else np.nan
        t = summ[(summ.scenario == scen) & (summ.model == "true")]
        if len(t):
            t = t.iloc[0]
            r["fixed: max |bias| B.to_A"] = t["absbias_B.to_A_max"]
            r["fixed: rho bias %"] = t["bias_rho_rel_pct"]
            r["fixed: cover95 B.to_A"] = t["cover95_toA"]
            if len(dd):
                for nl in ("N1", "N2"):
                    x = dd[(dd.scen == scen) & (dd.null == nl) & (dd.model == "true")]
                    r[f"fixed: dediff FPR {nl}"] = float(x.fpr.iloc[0]) if len(x) else np.nan
        x = summ[(summ.scenario == scen) & (summ.model == "ext")]
        if len(x):
            r["estimated: max |bias| B.to_A"] = x.iloc[0]["absbias_B.to_A_max"]
        main_rows.append(r)
    main = pd.DataFrame(main_rows)
    md(main, "scenario_table")
    md(summ, "per_scenario_all")
    # extended parameter estimates
    xcols = [c for c in fits.columns if c.startswith("x_")]
    if xcols:
        xe = fits[fits.model == "ext"].groupby(["scen", "gt"])[xcols].median().reset_index()
        md(xe.dropna(axis=1, how="all"), "extended_estimates")

    # ---- NUTS
    if len(S):
        sk = S[S.par.isin(KEY + C.NAMES)]
        nt = sk.groupby(["scen", "par"]).agg(
            runs=("gt", "size"), cover90=("cover90", "mean"),
            mean_abs_z=("z", lambda v: float(np.nanmean(np.abs(v)))),
            max_abs_z=("z", lambda v: float(np.nanmax(np.abs(v)))),
            bias_mean=("mean", "mean")).reset_index()
        nt["bias_mean"] = sk.assign(b=sk["mean"] - sk["truth"]).groupby(
            ["scen", "par"]).b.mean().values
        md(nt, "nuts_coverage")
        md(sk[sk.par.isin(KEY)][["scen", "gt", "par", "mean", "sd", "q05", "q95", "truth", "z",
                                 "cover90", "max_rhat"]], "nuts_key")  # fmt: skip
        allp = sk[sk.par.isin(C.NAMES)].groupby("scen").agg(
            cover90_all12=("cover90", "mean"), runs=("gt", "nunique")).reset_index()
        md(allp, "nuts_cover_all12")

    figures(per, dd, summ, ext, fits, p_crit)
    print(f"GOF size on base: {size_p:.3f} (n={len(base)}), calibrated p crit {p_crit:.3g}")
    print(main.to_string())


def heat(ax, M, rows, cols, fmt, cmap, vmax, title):
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    ax.set_title(title, fontsize=10)
    return im


def figures(per, dd, summ, ext, fits, p_crit):
    std = per[per.model == "std"]
    scen = [s for s in SCEN if s in set(std.scen)]
    gts = C.GTS
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), constrained_layout=True)
    for ax, (col, fmt, vmax, title) in zip(axes, [
        ("bias_B.to_A", "{:+.2f}", 0.3, "bias of B→A (MLE − truth)"),
        ("bias_m_BA", "{:+.2f}", 0.4, "bias of m_BA (MLE − truth)"),
        ("bias_rho_rel", "{:+.1%}", 0.1, "relative bias of growth rate ρ"),
    ]):  # fmt: skip
        M = np.full((len(scen), len(gts)), np.nan)
        for i, s in enumerate(scen):
            for j, g in enumerate(gts):
                x = std[(std.scen == s) & (std["gt"] == g)]
                if len(x):
                    M[i, j] = x[col].iloc[0]
        im = heat(ax, M, [f"{s}" for s in scen], gts, fmt, "RdBu_r", vmax, title)
        fig.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle("Standard model fitted to misspecified data: mean MLE error over datasets "
                 "(n = 512 clones per generation, generations 4/6/8/10)", fontsize=11)
    fig.savefig(FIG / "bias_heatmap.png", dpi=160)
    plt.close(fig)

    # coverage heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    M = np.full((len(scen), len(gts)), np.nan)
    for i, s in enumerate(scen):
        for j, g in enumerate(gts):
            x = std[(std.scen == s) & (std["gt"] == g)]
            if len(x):
                M[i, j] = x["cover_toA"].iloc[0]
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(gts)), gts, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(scen)), scen, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Coverage of the 95% profile-likelihood interval for B→A\n(standard model)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.savefig(FIG / "coverage_toA_heatmap.png", dpi=160)
    plt.close(fig)

    # dediff FPR
    if len(dd):
        fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
        sc = [s for s in SCEN if s in set(dd.scen)]
        x = np.arange(len(sc))
        w = 0.2
        for k, (nl, model, color, hatch) in enumerate([
            ("N1", "std", "#c0392b", ""), ("N2", "std", "#e67e22", ""),
            ("N1", "true", "#2c7fb8", "//"), ("N2", "true", "#7fcdbb", "//")]):  # fmt: skip
            vals, lo, hi = [], [], []
            for s in sc:
                r = dd[(dd.scen == s) & (dd.null == nl) & (dd.model == model)]
                if len(r):
                    vals.append(r.fpr.iloc[0])
                    lo.append(r.fpr.iloc[0] - r.ci_lo.iloc[0])
                    hi.append(r.ci_hi.iloc[0] - r.fpr.iloc[0])
                else:
                    vals.append(np.nan)
                    lo.append(0)
                    hi.append(0)
            lab = f"{nl}, {'standard model' if model == 'std' else 'true setting modelled'}"
            ax.bar(x + (k - 1.5) * w, vals, w, yerr=[lo, hi], color=color, hatch=hatch,
                   label=lab, error_kw={"lw": 0.7})  # fmt: skip
        ax.axhline(0.05, color="k", ls="--", lw=0.8)
        ax.set_xticks(x, sc, rotation=30, ha="right")
        ax.set_ylabel("false-positive rate (nominal 5%)")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, ncol=2)
        ax.set_title("Dedifferentiation test (χ̄² LRT) on data without dedifferentiation, "
                     "n = 512 per generation", fontsize=10)
        fig.savefig(FIG / "dediff_fpr.png", dpi=160)
        plt.close(fig)

    # does modelling it fix it? RMSE of B.to_A, m_BA and rho: standard / true / estimated
    fixable = [s for s in SCEN if s != "base" and s in set(summ[summ.model == "true"].scenario)]
    if fixable:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
        bstd = summ[(summ.scenario == "base") & (summ.model == "std")].iloc[0]
        for ax, p in zip(axes, ["B.to_A", "m_BA", "rho"]):
            x = np.arange(len(fixable))
            for k, (model, color, lab) in enumerate([
                ("std", "#c0392b", "standard model"), ("true", "#2c7fb8", "true setting known"),
                ("ext", "#27ae60", "setting estimated (extended model)")]):  # fmt: skip
                v = [summ[(summ.scenario == s) & (summ.model == model)][f"rmse_{p}"].iloc[0]
                     if len(summ[(summ.scenario == s) & (summ.model == model)]) else np.nan
                     for s in fixable]  # fmt: skip
                ax.bar(x + (k - 1) * 0.27, v, 0.27, color=color, label=lab)
            ax.axhline(bstd[f"rmse_{p}"], color="k", ls="--", lw=0.8,
                       label="correctly specified data")  # fmt: skip
            ax.set_xticks(x, fixable, rotation=30, ha="right")
            ax.set_title(f"RMSE of {p.replace('B.to_A', 'B→A')}", fontsize=10)
        axes[0].legend(fontsize=7)
        fig.suptitle("Does modelling the misspecification fix the estimates? "
                     "(MLE over truths × datasets)", fontsize=11)
        fig.savefig(FIG / "fix_rmse.png", dpi=160)
        plt.close(fig)

    # detection power
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    s2 = summ[summ.model == "std"]
    sc = [s for s in SCEN if s in set(s2.scenario)]
    x = np.arange(len(sc))
    gp = [float(s2[s2.scenario == s].gof_power.iloc[0]) for s in sc]
    ep = [float(ext[ext.scen == s].ext_power.iloc[0]) if s in set(ext.scen) else np.nan
          for s in sc]  # fmt: skip
    ax.bar(x - 0.2, gp, 0.4, color="#555",
           label=f"binned G² goodness of fit (calibrated: p < {p_crit:.3f})")
    ax.bar(x + 0.2, ep, 0.4, color="#2c7fb8",
           label="LRT extended vs standard model (where available)")  # fmt: skip
    ax.axhline(0.05, color="k", ls="--", lw=0.8)
    ax.set_xticks(x, sc, rotation=30, ha="right")
    ax.set_ylabel("detection rate over datasets × truths")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Can the misspecification be detected from the data? (n = 512 per generation; "
                 "'base' = false-alarm rate)", fontsize=10)
    fig.savefig(FIG / "detection_power.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
