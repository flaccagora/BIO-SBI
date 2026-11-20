"""XG2 under five arms: flat (XG2), flat-B->A, Dirichlet(1/2), exact Jeffreys, MLE.

Per (truth, n, seed) and free parameter: posterior mean, sd, 90% central interval and
coverage (MLE: Wald 90% interval from the observed information, ``mle.json`` ``se``; none
on the boundary), plus B's total division mass ``B.div`` (sum of B's three divisions).
Exact Jeffreys = PSIS-reweighted Dirichlet(1/2) draws (``jeffreys/<cell>/jeffreys_psis.npz``)
or, where a rerun exists (``jeffreys_rerun/<cell>/draws.npz``), its corrected draws.

Writes to results/exact_gw/data_size_jeffreys/analysis/: cells.csv (one row per arm x cell
x parameter), btoa.csv / allparams.csv / agreement.csv / diagnostics.csv / mixing.csv and
figures (figures/*.png).

  uv run python scripts/jeffreys/analyse_jeffreys.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FLAT_BTOA, JEFF, XG2  # noqa: E402

OUT = JEFF / "analysis"
ARMS = ["flat", "flat_btoa", "dir_half", "jeffreys", "mle"]
LABEL = {"flat": "flat (XG2)", "flat_btoa": "flat-B→A", "dir_half": "Dirichlet(1/2)",
         "jeffreys": "exact Jeffreys", "mle": "MLE"}  # fmt: skip
NS = (128, 256, 512, 1024)
Z90 = 1.6448536269514722
FREE = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
        "B.death", "B.to_A", "B.persist", "B.div_AA", "B.div_AB", "B.div_BB"]  # fmt: skip
EXTRA = "B.div"


def cells() -> list[tuple[str, int, int]]:
    out = []
    for gt_dir in sorted(XG2.iterdir()):
        if not gt_dir.is_dir() or gt_dir.name.startswith("_"):
            continue
        for n in NS:
            for s in range(3):
                if (gt_dir / f"n{n}_s{s}" / "metrics.json").exists():
                    out.append((gt_dir.name, n, s))
    return out


def summarise(x: np.ndarray, w: np.ndarray | None = None) -> dict:
    w = np.full(len(x), 1 / len(x)) if w is None else w / w.sum()
    m = float(w @ x)
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    q05, q95 = (float(x[order][min(np.searchsorted(cw, q), len(x) - 1)]) for q in (0.05, 0.95))
    return {"mean": m, "sd": float(np.sqrt(max(w @ (x - m) ** 2, 0))), "q05": q05, "q95": q95}


def draws_csv(run: Path) -> np.ndarray | None:
    f = run / "posterior_samples.csv.gz"
    if not f.exists():
        return None
    return pd.read_csv(f)[FREE].to_numpy()


def jeffreys_draws(gt: str, n: int, s: int, decode) -> tuple[np.ndarray, np.ndarray, dict] | None:
    cell = f"n{n}_s{s}"
    rerun = JEFF / "jeffreys_rerun" / gt / cell / "draws.npz"
    if rerun.exists():
        z = np.load(rerun)
        info = json.loads((rerun.parent / "summary.json").read_text())
        return z["phys"], z["w"], {"source": "rerun", **info}
    f = JEFF / "jeffreys" / gt / cell / "jeffreys_psis.npz"
    if not f.exists():
        return None
    z = np.load(f)
    info = json.loads((f.parent / "summary.json").read_text())
    phys = np.array([decode(t) for t in z["theta"]])[:, : len(FREE)]
    return phys, z["w"], {"source": "psis", "pareto_k": info["pareto_k"], "ess": info["ess"]}


def collect() -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from conftest import load_law

    from bio_sbi.simulators import GaltonWatson

    sim = GaltonWatson(load_law("two_type"))
    rows = []
    for gt, n, s in cells():
        cell = f"n{n}_s{s}"
        rec = pd.read_csv(XG2 / gt / cell / "recovery.csv").set_index("parameter")["truth"]
        truth = {p: float(rec[p]) for p in FREE}
        truth[EXTRA] = truth["B.div_AA"] + truth["B.div_AB"] + truth["B.div_BB"]
        base = {"gt": gt, "n": n, "seed": s}
        srcs = {"flat": XG2 / gt / cell, "flat_btoa": FLAT_BTOA / gt / cell,
                "dir_half": JEFF / "dir_half" / gt / cell}  # fmt: skip
        for arm, run in srcs.items():
            x = draws_csv(run)
            if x is None:
                continue
            x = np.column_stack([x, x[:, 9:12].sum(1)])
            for j, p in enumerate(FREE + [EXTRA]):
                rows.append({**base, "arm": arm, "parameter": p, "truth": truth[p],
                             **summarise(x[:, j])})  # fmt: skip
        jd = jeffreys_draws(gt, n, s, sim.decode)
        if jd is not None:
            x, w, info = jd
            x = np.column_stack([x, x[:, 9:12].sum(1)])
            for j, p in enumerate(FREE + [EXTRA]):
                rows.append({**base, "arm": "jeffreys", "parameter": p, "truth": truth[p],
                             **summarise(x[:, j], w), "pareto_k": info.get("pareto_k"),
                             "ess": info.get("ess"), "source": info["source"]})  # fmt: skip
        mf = XG2 / gt / cell / "mle.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            est = {p: float(m["mle"][p]) for p in FREE}
            est[EXTRA] = est["B.div_AA"] + est["B.div_AB"] + est["B.div_BB"]
            se = m.get("se") or {}
            for p in FREE + [EXTRA]:
                sd = se.get(p)
                sd = float(sd) if sd is not None else np.nan
                rows.append({**base, "arm": "mle", "parameter": p, "truth": truth[p],
                             "mean": est[p], "sd": sd, "q05": est[p] - Z90 * sd,
                             "q95": est[p] + Z90 * sd})  # fmt: skip
    df = pd.DataFrame(rows)
    df["err"] = df["mean"] - df["truth"]
    df["covered"] = np.where(df["sd"].notna() & np.isfinite(df["q05"]),
                             (df["q05"] <= df["truth"]) & (df["truth"] <= df["q95"]), np.nan)  # fmt: skip
    return df


def rmse(x: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    b = df[df["parameter"] == "B.to_A"]
    g = b.groupby(["gt", "n", "arm"])
    out["btoa"] = pd.DataFrame({
        "truth": g["truth"].first(), "mean": g["mean"].mean(), "sd": g["sd"].mean(),
        "bias": g["err"].mean(), "rmse": g["err"].apply(rmse), "coverage": g["covered"].mean(),
        "cells": g["mean"].size(),
    }).reset_index()  # fmt: skip
    g = b.groupby(["n", "arm"])
    out["btoa_by_n"] = pd.DataFrame({
        "rmse": g["err"].apply(rmse), "bias": g["err"].mean(), "abs_bias": g["err"].apply(
            lambda e: np.mean(np.abs(e))), "sd": g["sd"].mean(), "coverage": g["covered"].mean(),
        "cells": g["mean"].size(),
    }).reset_index()  # fmt: skip
    a = df[df["parameter"].isin(FREE)]
    g = a.groupby(["n", "arm"])
    out["allparams_by_n"] = pd.DataFrame({
        "rmse": g["err"].apply(rmse), "coverage": g["covered"].mean(),
        "coverage_defined": g["covered"].apply(lambda c: c.notna().mean()),
        "cells": g["mean"].size() / len(FREE),
    }).reset_index()  # fmt: skip
    g = a.groupby(["gt", "n", "arm"])
    out["allparams"] = pd.DataFrame({
        "rmse": g["err"].apply(rmse), "coverage": g["covered"].mean(),
    }).reset_index()  # fmt: skip
    bd = df[df["parameter"] == EXTRA].groupby(["n", "arm"])
    out["bdiv_by_n"] = pd.DataFrame({
        "rmse": bd["err"].apply(rmse), "bias": bd["err"].mean(), "coverage": bd["covered"].mean(),
    }).reset_index()  # fmt: skip
    # agreement of the Bayesian arms with each other: |mean_a - mean_b| / mean sd
    piv = df[df["arm"] != "mle"].pivot_table(index=["gt", "n", "seed", "parameter"],
                                             columns="arm", values=["mean", "sd"])  # fmt: skip
    rows = []
    pairs = [("flat", "flat_btoa"), ("flat", "dir_half"), ("flat", "jeffreys"),
             ("dir_half", "jeffreys"), ("flat_btoa", "jeffreys")]  # fmt: skip
    for (gt, n, s, p), r in piv.iterrows():
        for x, y in pairs:
            if pd.isna(r[("mean", x)]) or pd.isna(r[("mean", y)]):
                continue
            sd = 0.5 * (r[("sd", x)] + r[("sd", y)])
            rows.append({"gt": gt, "n": n, "seed": s, "parameter": p, "pair": f"{x}~{y}",
                         "shift_sd": abs(r[("mean", x)] - r[("mean", y)]) / sd})  # fmt: skip
    ag = pd.DataFrame(rows)
    out["agreement_cells"] = ag
    cellmax = ag[ag["parameter"].isin(FREE)].groupby(["gt", "n", "seed", "pair"])["shift_sd"].max()
    cellmax = cellmax.reset_index()
    btoa = ag[ag["parameter"] == "B.to_A"]
    out["agreement"] = pd.DataFrame({
        "btoa_median_shift": btoa.groupby(["n", "pair"])["shift_sd"].median(),
        "btoa_frac_lt_0.25": btoa.groupby(["n", "pair"])["shift_sd"].apply(
            lambda s: (s < 0.25).mean()),
        "all_max_median_shift": cellmax.groupby(["n", "pair"])["shift_sd"].median(),
        "all_frac_cells_lt_0.25": cellmax.groupby(["n", "pair"])["shift_sd"].apply(
            lambda s: (s < 0.25).mean()),
    }).reset_index()  # fmt: skip
    return out


def diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, mix = [], []
    roots = {"flat": XG2, "flat_btoa": FLAT_BTOA, "dir_half": JEFF / "dir_half"}
    for gt, n, s in cells():
        for arm, root in roots.items():
            f = root / gt / f"n{n}_s{s}" / "metrics.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            e = m.get("exact_gw", {})
            st = e.get("stop") or {}
            rr = e.get("rank_rhat") or e.get("rhat") or {}
            eb = e.get("ess_bulk") or e.get("ess") or {}
            et = e.get("ess_tail") or {}
            rows.append({
                "arm": arm, "gt": gt, "n": n, "seed": s,
                "stop_reason": st.get("reason", "fixed draws"), "met": st.get("met"),
                "failed": ",".join(st.get("failed") or []),
                "draws_per_chain": e.get("draws_per_chain"),
                "max_rank_rhat": e.get("max_rank_rhat", e.get("max_rhat")),
                "min_ess_bulk": e.get("min_ess_bulk", e.get("min_ess")),
                "min_ess_tail": e.get("min_ess_tail"), "divergences": e.get("divergences"),
                "tree_depth_hits": e.get("max_tree_depth_hits"),
                "minutes": m.get("total_seconds", np.nan) / 60,
                "warm_start": bool(e.get("warm_start")),
            })  # fmt: skip
            for p in FREE:
                mix.append({"arm": arm, "gt": gt, "n": n, "seed": s, "parameter": p,
                            "rank_rhat": rr.get(p), "ess_bulk": eb.get(p),
                            "ess_tail": et.get(p)})  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(mix)


def figures(df: pd.DataFrame, t: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"flat": "#4C72B0", "flat_btoa": "#DD8452", "dir_half": "#55A868",
              "jeffreys": "#C44E52", "mle": "#555555"}  # fmt: skip
    marks = {"flat": "o", "flat_btoa": "s", "dir_half": "^", "jeffreys": "D", "mle": "x"}
    off = dict(zip(ARMS, np.linspace(-0.16, 0.16, len(ARMS)), strict=True))
    b = df[df["parameter"] == "B.to_A"]
    gts = sorted(b["gt"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
    for ax, gt in zip(axes.flat, gts, strict=False):
        sub = b[b["gt"] == gt]
        for arm in ARMS:
            a = sub[sub["arm"] == arm]
            if a.empty:
                continue
            x = np.log2(a["n"].to_numpy() / 128) + off[arm] + (a["seed"].to_numpy() - 1) * 0.035
            ax.errorbar(x, a["mean"], yerr=[a["mean"] - a["q05"], a["q95"] - a["mean"]],
                        fmt=marks[arm], color=colors[arm], ms=4, lw=0.8, alpha=0.85,
                        label=LABEL[arm])  # fmt: skip
        ax.axhline(sub["truth"].iloc[0], color="k", lw=1, ls="--")
        ax.set_title(f"{gt} (B.to_A = {sub['truth'].iloc[0]:.2f})")
        ax.set_xticks(range(4), [str(n) for n in NS])
        ax.set_ylim(-0.02, 1.02)
    axes[0, 0].legend(fontsize=7, loc="lower left")
    for ax in axes[-1]:
        ax.set_xlabel("clones per endpoint n")
    for ax in axes[:, 0]:
        ax.set_ylabel("B.to_A: mean and 90% interval")
    fig.tight_layout()
    fig.savefig(fig_dir / "btoa_by_truth.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key, ax, ylab in (("btoa_by_n", axes[0], "B.to_A RMSE"),
                          ("allparams_by_n", axes[1], "all-parameter RMSE")):  # fmt: skip
        tt = t[key]
        for arm in ARMS:
            a = tt[tt["arm"] == arm].sort_values("n")
            ax.plot(a["n"], a["rmse"], marker=marks[arm], color=colors[arm], label=LABEL[arm])
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS, [str(n) for n in NS])
        ax.set_ylabel(ylab)
        ax.set_xlabel("n")
    tt = t["allparams_by_n"]
    for arm in ARMS:
        a = tt[tt["arm"] == arm].sort_values("n")
        axes[2].plot(a["n"], a["coverage"], marker=marks[arm], color=colors[arm])
    tb = t["btoa_by_n"]
    for arm in ARMS:
        a = tb[tb["arm"] == arm].sort_values("n")
        axes[2].plot(a["n"], a["coverage"], marker=marks[arm], color=colors[arm], ls=":")
    axes[2].axhline(0.9, color="k", lw=0.8, ls="--")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(NS, [str(n) for n in NS])
    axes[2].set_ylabel("90% coverage (solid: all, dotted: B.to_A)")
    axes[2].set_xlabel("n")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "rmse_coverage_by_n.png", dpi=130)
    plt.close(fig)


def _wide(t: pd.DataFrame, index, value: str, digits: int = 3) -> pd.DataFrame:
    w = t.pivot_table(index=index, columns="arm", values=value)
    return w[[a for a in ARMS if a in w.columns]].rename(columns=LABEL).round(digits)


def psis_table() -> pd.DataFrame:
    """Per cell: Pareto k of the PSIS step and, where rerun, of the rerun's correction."""
    rows = []
    for f in sorted((JEFF / "jeffreys").glob("*/n*_s*/summary.json")):
        d = json.loads(f.read_text())
        gt, cell = f.parent.parent.name, f.parent.name
        n, s = (int(x) for x in cell[1:].split("_s"))
        r = JEFF / "jeffreys_rerun" / gt / cell / "summary.json"
        rr = json.loads(r.read_text()) if r.exists() else {}
        rows.append({"gt": gt, "n": n, "seed": s, "k_psis": d["pareto_k"],
                     "ess_psis": d["ess"], "draws": d["draws"], "log_w_sd": d["log_w_sd"],
                     "rerun": bool(rr), "rerun_iteration": rr.get("iteration"),
                     "k_rerun": rr.get("pareto_k"), "ess_rerun": rr.get("ess"),
                     "rerun_stop": rr.get("stop"), "rerun_rhat": rr.get("max_rank_rhat"),
                     "rerun_ess_bulk": rr.get("min_ess_bulk"),
                     "rerun_minutes": rr.get("seconds", np.nan) / 60})  # fmt: skip
    return pd.DataFrame(rows)


def markdown(t: dict, diag: pd.DataFrame, mix: pd.DataFrame, ps: pd.DataFrame) -> str:
    """Markdown tables for docs/jeffreys/README.md."""
    out = []

    def add(title, frame):
        out.append(f"### {title}\n\n{frame.to_markdown()}\n")

    add(
        "B.to_A by n (9 truths x 3 datasets): RMSE of the posterior mean / MLE",
        _wide(t["btoa_by_n"], "n", "rmse"),
    )
    add("B.to_A by n: mean bias (estimate - truth)", _wide(t["btoa_by_n"], "n", "bias"))
    add("B.to_A by n: mean posterior sd (MLE: Wald se)", _wide(t["btoa_by_n"], "n", "sd"))
    add("B.to_A by n: 90% coverage", _wide(t["btoa_by_n"], "n", "coverage", 2))
    add("All 12 probabilities by n: RMSE", _wide(t["allparams_by_n"], "n", "rmse"))
    add(
        "All 12 probabilities by n: 90% coverage (MLE: where a Wald interval exists)",
        _wide(t["allparams_by_n"], "n", "coverage", 2),
    )
    add("B's division mass B.div by n: RMSE", _wide(t["bdiv_by_n"], "n", "rmse"))
    add(
        "B.to_A per truth and n: mean over the 3 datasets of the estimate",
        _wide(t["btoa"], ["gt", "n"], "mean", 2).join(
            t["btoa"].groupby(["gt", "n"])["truth"].first().round(2)
        ),
    )
    add("B.to_A per truth and n: RMSE over the 3 datasets", _wide(t["btoa"], ["gt", "n"], "rmse"))
    add("B.to_A per truth and n: coverage (of 3)", _wide(t["btoa"], ["gt", "n"], "coverage", 2))
    ag = t["agreement"].pivot(index="n", columns="pair",
                              values=["btoa_median_shift", "all_frac_cells_lt_0.25"])  # fmt: skip
    add(
        "Agreement between priors: median |shift| of the B.to_A mean in posterior sd, and "
        "fraction of cells where every parameter moves < 0.25 sd",
        ag.round(2),
    )
    g = diag.groupby(["arm", "n"])
    dsum = pd.DataFrame({
        "cells": g.size(), "stop_met": g["met"].apply(lambda m: (m == True).mean()),  # noqa: E712
        "median_max_rank_rhat": g["max_rank_rhat"].median(),
        "frac_rhat_gt_1.05": g["max_rank_rhat"].apply(lambda r: (r > 1.05).mean()),
        "median_min_ess_bulk": g["min_ess_bulk"].median(),
        "median_min_ess_tail": g["min_ess_tail"].median(),
        "median_divergences": g["divergences"].median(), "median_minutes": g["minutes"].median(),
    })  # fmt: skip
    add("Sampler diagnostics (flat: fixed 500 draws/chain, no stopping rule)", dsum.round(3))
    mg = mix[mix["arm"] != "flat"].groupby(["parameter", "arm"])
    msum = pd.DataFrame({
        "rhat_gt_1.01": mg["rank_rhat"].apply(lambda r: (r > 1.01).mean()),
        "ess_bulk_lt_400": mg["ess_bulk"].apply(lambda e: (e < 400).mean()),
        "ess_tail_lt_200": mg["ess_tail"].apply(lambda e: (e < 200).mean()),
    }).unstack("arm")  # fmt: skip
    add(
        "Which parameters fail to mix: fraction of the 108 cells failing each criterion",
        msum.round(2),
    )
    pg = ps.groupby("n")
    psum = pd.DataFrame({
        "cells": pg.size(), "median_k_psis": pg["k_psis"].median(),
        "k_psis_gt_0.7": pg["k_psis"].apply(lambda k: int((k > 0.7).sum())),
        "reruns": pg["rerun"].sum(), "max_k_rerun": pg["k_rerun"].max(),
    })  # fmt: skip
    add("Exact Jeffreys: PSIS from Dirichlet(1/2) and reruns", psum.round(2))
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "cells.csv", index=False)
    # tables and figures on the cells every arm has (same datasets in every column)
    have = df.groupby(["gt", "n", "seed"])["arm"].nunique()
    common = have[have == len(ARMS)].index
    df = df.set_index(["gt", "n", "seed"]).loc[common].reset_index()
    print(f"{len(common)} cells with all {len(ARMS)} arms")
    t = tables(df)
    for k, v in t.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    diag, mix = diagnostics()
    diag.to_csv(OUT / "diagnostics.csv", index=False)
    mix.to_csv(OUT / "mixing.csv", index=False)
    figures(df, t)
    ps = psis_table()
    ps.to_csv(OUT / "psis.csv", index=False)
    (OUT / "tables.md").write_text(markdown(t, diag, mix, ps))
    pd.set_option("display.width", 200)
    print(t["btoa_by_n"].pivot(index="n", columns="arm", values=["rmse", "coverage"]).round(3))
    print(t["allparams_by_n"].pivot(index="n", columns="arm", values=["rmse", "coverage"]).round(3))


if __name__ == "__main__":
    main()
