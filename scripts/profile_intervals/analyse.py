"""Profile-likelihood intervals of the XG2 exact-GW runs vs posterior CrIs and Wald.

Reads the profiles of ``run_profiles.py`` (results/exact_gw/profile_intervals/runs/) and
the posterior draws of the four arms (flat = XG2, flat-B->A, Dirichlet(1/2), exact
Jeffreys = PSIS-reweighted Dir(1/2) draws or their rerun, see scripts/jeffreys) and the
Wald intervals (``mle.json``: MLE +- z se). Writes

* results/exact_gw/profile_intervals/: profile_intervals.csv (one row per run x
  functional), profile_curves.csv (every profile point), intervals.csv (one row per run x
  functional x method x level), coverage / width / disagreement tables;
* docs/profile_intervals/tables/*.csv|md and docs/profile_intervals/figures/*.png|pdf.

  uv run python scripts/profile_intervals/analyse.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "jeffreys"))
from plib import BOUND_90, BOUND_95, CHI2_90, CHI2_95, FUNCTIONALS, intervals  # noqa: E402
from run_profiles import ARMS, FREE, OUT, RUNS, XG2, run_list  # noqa: E402

DOCS = REPO / "docs" / "profile_intervals"
FIG, TAB = DOCS / "figures", DOCS / "tables"
FUNCS = list(FUNCTIONALS)
PARAMS12 = FREE
Z = {90: 1.6448536269514722, 95: 1.959963984540054}
THR = {90: CHI2_90, 95: CHI2_95}
THR_BC = {90: BOUND_90, 95: BOUND_95}
METHODS = ["profile", "flat", "flat_btoa", "dir_half", "jeffreys", "wald"]
MLABEL = {"profile": "profile", "profile_bc": "profile (boundary-corr.)", "flat": "flat (XG2)",
          "flat_btoa": "flat-B→A", "dir_half": "Dir(1/2)", "jeffreys": "exact Jeffreys",
          "wald": "Wald"}  # fmt: skip
GTS = ["T1_ref", "T2_toA05", "T4_slowA", "T3_Bdiv", "T5_C7", "P1", "P2", "P3", "P4"]
ZERO = 1e-6  # truth / MLE "at the boundary"


def count_modes(drop: np.ndarray, within: float, dip: float) -> int:
    """Local minima of ``drop`` below ``within`` separated by a rise of more than ``dip``."""
    modes, m, peak, valley = 0, drop[0], 0.0, True
    for d in drop[1:]:
        if valley:
            if d < m:
                m = d
            elif d > m + dip:
                modes += int(m < within)
                valley, peak = False, d
        elif d > peak:
            peak = d
        elif d < peak - dip:
            valley, m = True, d
    if valley:
        modes += int(m < within)
    return modes


def truth_value(func, truth: np.ndarray, names: list[str]) -> float:
    return float(func.value(truth, names))


# ------------------------------------------------------------------------ profiles
def collect_profiles() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, curves, runs = [], [], []
    for gt, n, s in run_list():
        d = RUNS / gt / f"n{n}_s{s}"
        if not (d / "base.npz").exists() or not all((d / f"{f}.npz").exists() for f in FUNCS):
            continue
        b = np.load(d / "base.npz")
        names = [str(x) for x in b["names"]]
        prof = {f: np.load(d / f"{f}.npz", allow_pickle=True) for f in FUNCS}
        ll_glob = max(float(b["ll_refit"]), max(float(z["ll"].max()) for z in prof.values()))
        g = json.loads((d / "global.json").read_text()) if (d / "global.json").exists() else {}
        runs.append({"gt": gt, "n": n, "seed": s, "ll_stored": float(b["ll_stored"]),
                     "ll_refit": float(b["ll_refit"]), "ll_global": ll_glob,
                     "gain_refit": float(b["ll_refit"] - b["ll_stored"]),
                     "gain_global": ll_glob - float(b["ll_stored"]),
                     "new_max": ll_glob > float(b["ll_stored"]) + 0.01,
                     "redone": bool(g.get("new_max", False)),
                     "base_seconds": float(b["seconds"]), "base_evals": int(b["evals"]),
                     "draws_best_gap": ll_glob - float(b["draws_ll"].max())})  # fmt: skip
        mle = json.loads((XG2 / gt / f"n{n}_s{s}" / "mle.json").read_text())
        for fname, z in prof.items():
            func = FUNCTIONALS[fname]
            c, ll = z["c"], z["ll"]
            drop = ll_glob - ll
            jm = int(np.argmax(ll))
            tv = truth_value(func, b["truth"], names)
            chat = float(c[jm])
            at0 = chat <= ZERO * func.cmax or chat >= func.cmax * (1 - ZERO)
            row = {"gt": gt, "n": n, "seed": s, "parameter": fname, "truth": tv,
                   "mle_profile": chat, "mle_stored": float(func.value(b["mle_stored"], names)),
                   "ll_max_profile": float(ll[jm]), "max_gap": float(ll[jm] - ll_glob),
                   "max_minus_refit": float(ll[jm] - b["ll_refit"]),
                   "drop_truth": float(np.interp(tv, c, drop)),
                   "drop_at_0": float(drop[0]) if c[0] == 0 else np.nan,
                   "drop_at_cmax": float(drop[-1]) if c[-1] == func.cmax else np.nan,
                   "cmax": func.cmax, "points": len(c), "fits": int(z["fits"]),
                   "evals": int(z["evals"]), "seconds": float(z["seconds"]),
                   "redone": bool(z["redone"]), "mle_at_0": at0,
                   "se_wald": (mle.get("se") or {}).get(fname)}  # fmt: skip
            # local maxima of the curve within 4 nats, separated by a dip > 0.1 nats
            row["local_maxima"] = count_modes(drop, 4.0, 0.1)
            row["cross_added"] = int(z["cross"]) if "cross" in z.files else 0
            for lev in (90, 95):
                for kind, thr in (("", THR[lev]), ("_bc", THR_BC[lev] if at0 else THR[lev])):
                    segs = intervals(c, drop, thr, func.cmax)
                    row[f"lo{lev}{kind}"] = segs[0][0] if segs else np.nan
                    row[f"hi{lev}{kind}"] = segs[-1][1] if segs else np.nan
                    row[f"len{lev}{kind}"] = sum(h - lo for lo, h in segs)
                    row[f"nseg{lev}{kind}"] = len(segs)
                    row[f"segs{lev}{kind}"] = ";".join(f"{lo:.4f}:{h:.4f}" for lo, h in segs)
                    row[f"cov{lev}{kind}"] = any(lo - 1e-12 <= tv <= h + 1e-12 for lo, h in segs)
                    row[f"open_lo{lev}{kind}"] = bool(segs) and segs[0][0] <= 1e-9
                    row[f"open_hi{lev}{kind}"] = bool(segs) and segs[-1][1] >= func.cmax - 1e-9
            rows.append(row)
            for j in range(len(c)):
                curves.append({"gt": gt, "n": n, "seed": s, "parameter": fname, "c": c[j],
                               "ll": ll[j], "drop": drop[j],
                               **{nm: z["phys"][j][i] for i, nm in enumerate(PARAMS12)}})  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(runs)


# ------------------------------------------------------------------------ posteriors
def wquant(x: np.ndarray, w: np.ndarray | None, qs) -> list[float]:
    if w is None:
        return [float(v) for v in np.quantile(x, qs)]
    w = w / w.sum()
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return [float(x[o][min(np.searchsorted(cw, q), len(x) - 1)]) for q in qs]


def derived(x: np.ndarray) -> np.ndarray:
    """Append B.div and B.m_BA to (k, 12) draws."""
    return np.column_stack([x, x[:, 9:12].sum(1), x[:, 7] + 2 * x[:, 9] + x[:, 10]])


def collect_posteriors() -> pd.DataFrame:
    from analyse_jeffreys import jeffreys_draws

    sys.path.insert(0, str(REPO / "tests"))
    from conftest import load_law

    from bio_sbi.simulators import GaltonWatson

    sim = GaltonWatson(load_law("two_type"))
    qs = [0.025, 0.05, 0.95, 0.975]
    rows = []
    for gt, n, s in run_list():
        base = {"gt": gt, "n": n, "seed": s}
        draws = {}
        for arm, root in ARMS.items():
            f = root / gt / f"n{n}_s{s}" / "posterior_samples.csv.gz"
            if f.exists():
                draws[arm] = (derived(pd.read_csv(f)[FREE].to_numpy()), None)
        if n <= 1024:
            jd = jeffreys_draws(gt, n, s, sim.decode)
            if jd is not None:
                draws["jeffreys"] = (derived(jd[0]), jd[1])
        for arm, (x, w) in draws.items():
            for j, p in enumerate(FUNCS):
                q = wquant(x[:, j], w, qs)
                for lev, (lo, hi) in ((95, (q[0], q[3])), (90, (q[1], q[2]))):
                    rows.append({**base, "parameter": p, "method": arm, "level": lev,
                                 "lo": lo, "hi": hi})  # fmt: skip
        mle = json.loads((XG2 / gt / f"n{n}_s{s}" / "mle.json").read_text())
        se = mle.get("se") or {}
        for p in FUNCS:
            v = se.get(p)
            for lev in (90, 95):
                if v is None or p not in mle["mle"]:
                    rows.append({**base, "parameter": p, "method": "wald", "level": lev,
                                 "lo": np.nan, "hi": np.nan})  # fmt: skip
                else:
                    m = mle["mle"][p]
                    rows.append({**base, "parameter": p, "method": "wald", "level": lev,
                                 "lo": m - Z[lev] * v, "hi": m + Z[lev] * v})  # fmt: skip
    return pd.DataFrame(rows)


def long_table(prof: pd.DataFrame, post: pd.DataFrame) -> pd.DataFrame:
    keys = ["gt", "n", "seed", "parameter"]
    meta = prof[keys + ["truth", "mle_at_0", "cmax"]]
    rows = []
    for r in prof.itertuples():
        for lev in (90, 95):
            for kind, m in (("", "profile"), ("_bc", "profile_bc")):
                rows.append({"gt": r.gt, "n": r.n, "seed": r.seed, "parameter": r.parameter,
                             "method": m, "level": lev, "lo": getattr(r, f"lo{lev}{kind}"),
                             "hi": getattr(r, f"hi{lev}{kind}"),
                             "width": getattr(r, f"len{lev}{kind}"),
                             "covered": getattr(r, f"cov{lev}{kind}"),
                             "nseg": getattr(r, f"nseg{lev}{kind}"),
                             "open_lo": getattr(r, f"open_lo{lev}{kind}"),
                             "open_hi": getattr(r, f"open_hi{lev}{kind}")})  # fmt: skip
    pl = pd.DataFrame(rows)
    po = post.merge(meta, on=keys, how="inner")
    po["exists"] = po["lo"].notna()
    lo_c, hi_c = po["lo"].clip(lower=0), np.minimum(po["hi"], po["cmax"])
    po["width"] = np.where(po["exists"], hi_c - lo_c, np.nan)
    po["covered"] = np.where(po["exists"], (po["lo"] <= po["truth"]) & (po["truth"] <= po["hi"]),
                             np.nan)  # fmt: skip
    po["nseg"] = np.where(po["exists"], 1, 0)
    pl = pl.merge(meta, on=keys, how="left")
    pl["exists"] = True
    out = pd.concat([pl, po.drop(columns=[])], ignore_index=True)
    out["truth_zero"] = out["truth"] <= ZERO
    out["covered"] = out["covered"].astype(float)
    return out


# ------------------------------------------------------------------------ tables
def coverage_tables(L: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    pos = L[~L.truth_zero & L.exists]
    meths = METHODS + ["profile_bc"]

    def cov(df, by):
        g = df.groupby(by + ["method", "level"])["covered"]
        t = g.mean().unstack(["method", "level"])
        return t

    sets = {"btoa": pos[pos.parameter == "B.to_A"], "all12": pos[pos.parameter.isin(PARAMS12)],
            "bdiv": pos[pos.parameter == "B.div"], "mba": pos[pos.parameter == "B.m_BA"]}  # fmt: skip
    for name, df in sets.items():
        t = cov(df, ["n"])
        out[f"coverage_{name}_by_n"] = t
        tot = df.groupby(["method", "level"])["covered"].mean().to_frame("all n").T
        out[f"coverage_{name}_by_n"] = pd.concat([t, tot])
        out[f"coverage_{name}_by_gt"] = cov(df, ["gt"])
    out["coverage_by_param"] = cov(pos, ["parameter"])
    # counts behind each coverage cell
    out["coverage_counts_by_n"] = (pos[pos.parameter.isin(PARAMS12)]
                                   .groupby(["n", "method", "level"])["covered"].size()
                                   .unstack(["method", "level"]))  # fmt: skip
    # Wald existence (share of truth>0 cells with an interval)
    w = L[(L.method == "wald") & ~L.truth_zero & L.parameter.isin(PARAMS12) & (L.level == 95)]
    out["wald_exists_by_n"] = w.groupby("n")["exists"].mean().to_frame("wald_defined")
    # truth = 0 cells: profile coverage chi2 vs boundary-corrected thresholds (oracle rule:
    # corrected threshold applied to every truth-0 cell) and the data-driven rule
    z = L[L.truth_zero & L.method.isin(["profile", "profile_bc"])]
    out["coverage_truth0"] = (z.groupby(["n", "method", "level"])["covered"].mean()
                              .unstack(["method", "level"]))  # fmt: skip
    out["coverage_truth0_counts"] = (
        z.groupby(["n", "method", "level"]).size().unstack(["method", "level"])
    )
    out["width_by_n"] = (L[L.exists & L.parameter.isin(PARAMS12) & ~L.truth_zero
                           & L.method.isin(meths)]
                         .groupby(["n", "method", "level"])["width"].mean()
                         .unstack(["method", "level"]))  # fmt: skip
    out["width_btoa_by_n"] = (L[L.exists & (L.parameter == "B.to_A") & L.method.isin(meths)]
                              .groupby(["n", "method", "level"])["width"].mean()
                              .unstack(["method", "level"]))  # fmt: skip
    w95 = L[L.exists & (L.level == 95) & L.method.isin(meths)]
    out["width95_by_param_n"] = (w95.groupby(["parameter", "n", "method"])["width"].mean()
                                 .unstack("method"))  # fmt: skip
    return out


def disagreements(L: pd.DataFrame) -> pd.DataFrame:
    keys = ["gt", "n", "seed", "parameter", "level"]
    p = L[L.method == "profile"].set_index(keys)
    rows = []
    for arm in ["flat", "flat_btoa", "dir_half", "jeffreys", "wald"]:
        a = L[(L.method == arm) & L.exists].set_index(keys)
        j = p.join(a, rsuffix="_a", how="inner")
        j = j[~j.truth_zero]
        for k, r in j.iterrows():
            if r.covered != r.covered_a:
                rows.append({**dict(zip(keys, k, strict=True)), "arm": arm, "truth": r.truth,
                             "profile_covers": bool(r.covered),
                             "profile": r.lo if r.nseg <= 1 else f"{r.lo:.3f}..{r.hi:.3f}*",
                             "profile_lo": r.lo, "profile_hi": r.hi, "profile_nseg": r.nseg,
                             "arm_lo": r.lo_a, "arm_hi": r.hi_a})  # fmt: skip
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, fmt="{:.2f}") -> str:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{MLABEL.get(a, a)} {b}%" for a, b in df.columns]
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float | np.floating):
                cells.append("" if np.isnan(v) else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    import figures

    prof, curves, runs = collect_profiles()
    print(f"profiles: {prof[['gt', 'n', 'seed']].drop_duplicates().shape[0]} runs, "
          f"{len(prof)} rows, {len(curves)} points")  # fmt: skip
    post = collect_posteriors()
    L = long_table(prof, post)
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    prof.to_csv(OUT / "profile_intervals.csv", index=False)
    curves.to_csv(OUT / "profile_curves.csv.gz", index=False)
    runs.to_csv(OUT / "runs.csv", index=False)
    L.to_csv(OUT / "intervals.csv", index=False)
    prof.to_csv(TAB / "profile_intervals.csv", index=False)
    runs.to_csv(TAB / "runs.csv", index=False)
    L.to_csv(TAB / "intervals.csv.gz", index=False)
    t = coverage_tables(L)
    # truth = 0 cells, oracle boundary rule: the 50:50 chi2 mixture thresholds (0.821 /
    # 1.353 nats) applied to every truth-0 cell, i.e. covered iff drop(0) <= threshold
    z0 = prof[prof.truth <= ZERO]
    t["coverage_truth0"] = pd.DataFrame({
        "chi2 90%": z0.groupby("n")["cov90"].mean(),
        "chi2 95%": z0.groupby("n")["cov95"].mean(),
        "mixture 90% (0.821)": z0.groupby("n")["drop_truth"].apply(lambda d: (d <= THR_BC[90]).mean()),
        "mixture 95% (1.353)": z0.groupby("n")["drop_truth"].apply(lambda d: (d <= THR_BC[95]).mean()),
        "MLE at 0": z0.groupby("n")["mle_at_0"].mean(),
        "cells": z0.groupby("n").size(),
    })  # fmt: skip
    t.pop("coverage_truth0_counts", None)
    dis = disagreements(L)
    dis.to_csv(TAB / "disagreements.csv", index=False)
    t["disagreement_counts"] = (dis.assign(kind=np.where(dis.profile_covers, "profile only",
                                                          "arm only"))
                                .groupby(["arm", "level", "kind"]).size().unstack("kind")
                                .fillna(0).astype(int))  # fmt: skip
    # open-ended / union shares
    pp = L[L.method == "profile"]
    t["profile_shape"] = (
        pp.assign(
            open=pp.open_lo | pp.open_hi,
            open_interior=(pp.open_lo & ~pp.mle_at_0) | pp.open_hi,
            union=pp.nseg > 1,
        )
        .groupby(["n", "level"])[["open", "open_interior", "union"]]
        .mean()
    )
    t["profile_shape_param"] = (pp[pp.level == 95].assign(
        open=pp.open_lo | pp.open_hi, union=pp.nseg > 1)
        .groupby("parameter")[["open", "union"]].mean())  # fmt: skip
    with open(TAB / "tables.md", "w") as fh:
        for name, df in t.items():
            df.to_csv(TAB / f"{name}.csv")
            fh.write(f"### {name}\n\n{md(df, '{:.3f}' if 'width' in name else '{:.2f}')}\n\n")
    print((TAB / "tables.md").read_text()[:6000])
    figures.make_all(prof, curves, L, t)


if __name__ == "__main__":
    main()
