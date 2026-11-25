"""All-12-parameter error measures for report sections 10.1 (unknown progenitors) and 12 (founder
distribution), next to the B->A-specific ones.

Usage (from the repository root)::

    uv run python report_exact_gw/make_all12.py

* Fisher (founder distribution, ``results/exact_gw/founder_distribution/fisher``): total
  asymptotic variance of the 12 probabilities, tr(I_theta^-1)/n (the 12 x 10 contrast basis is
  orthonormal), founders known (K) or estimated (E, nuisances profiled out); reported as the
  gain ref/setting against single known A founders on the same design (median over truths).
* MLE and NUTS (founder distribution): mean |estimate - truth| over the 12 probabilities, and
  for NUTS the share of the 12 covered by the 90% interval.
* NUTS (free founder, section 10.1, and the all-A model of the misspecification study): the
  same per run.

Writes ``report_exact_gw/tables/all12_*.csv``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = Path("/home/matteo/BIO-SBI/results/exact_gw")
TAB = ROOT / "report_exact_gw" / "tables"
NAMES = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
         "B.death", "B.to_A", "B.persist", "B.div_AA", "B.div_AB", "B.div_BB"]  # fmt: skip


def fisher_table() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "scripts" / "founder_distribution"))
    import analyse_fisher as AF  # noqa: PLC0415
    import fd_common as F  # noqa: PLC0415

    rows = []
    for gt in F.TRUTHS:
        z = dict(np.load(F.CACHE / f"{gt}.npz"))
        for dname, design in F.DESIGNS.items():
            for dist in z["dists"]:
                pi, pa = F.all_dists()[str(dist)]
                Mf, W = AF.to_face(AF.design_info(z, str(dist), design), pi)
                k = W.shape[1]
                for mode in ("K", "E"):
                    I_th = F.schur(Mf, np.arange(10), np.array(AF.mode_nuisance(mode, k), int))
                    try:
                        tr = float(np.trace(np.linalg.inv(I_th))) / F.N_TOTAL
                    except np.linalg.LinAlgError:
                        tr = float(np.trace(np.linalg.pinv(I_th))) / F.N_TOTAL
                    rows.append({"truth": F.SHORTNAME[gt], "design": dname, "dist": str(dist),
                                 "mode": mode, "trvar12": tr})  # fmt: skip
    df = pd.DataFrame(rows)
    ref = df[(df.dist == "S1_pA1.0") & (df["mode"] == "K")].set_index(["truth", "design"])
    df["gain_all12"] = ref.loc[list(zip(df.truth, df.design, strict=True)), "trvar12"].values / df.trvar12
    return df


def mae12(d: dict) -> tuple[float, float]:
    """(mean |posterior mean - truth|, share covered by the 90% interval) over the 12."""
    st = d["stats"]
    err = [abs(st[p]["mean"] - st[p]["truth"]) for p in NAMES]
    cov = [bool(st[p]["cover90"]) for p in NAMES]
    return float(np.mean(err)), float(np.mean(cov))


def founder_sims() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = RES / "founder_distribution"
    rows = []
    for f in sorted((base / "mle").glob("*/*/*.json")):
        d = json.loads(f.read_text())
        rows.append({"truth": d["gt"], "dist": d["dist"], "model": d["model"], "seed": d["seed"],
                     "mae12": float(np.mean([abs(d["err"][p]) for p in NAMES]))})  # fmt: skip
    mle = pd.DataFrame(rows)
    rows = []
    for f in sorted((base / "nuts").glob("*/*/*.json")):
        d = json.loads(f.read_text())
        m, c = mae12(d)
        rows.append({"truth": d["gt"], "dist": d["dist"], "model": d["model"], "mae12": m,
                     "cover12": c})  # fmt: skip
    return mle, pd.DataFrame(rows)


def free_founder_sims() -> pd.DataFrame:
    rows = []
    for f in sorted((RES / "free_founder").glob("*/*/*_s0.json")):
        d = json.loads(f.read_text())
        m, c = mae12(d)
        rows.append({"scen": d["scen"], "model": d["model"], "truth": d["gt"], "mae12": m,
                     "cover12": c})  # fmt: skip
    for f in sorted((RES / "misspecification" / "nuts").glob("*/*.json")):
        d = json.loads(f.read_text())
        if d["scen"] != "F70":
            continue
        m, c = mae12(d)
        rows.append({"scen": "F70", "model": "all-A", "truth": d["gt"], "mae12": m, "cover12": c})
    return pd.DataFrame(rows)


def truth12(gt: str) -> np.ndarray:
    z = np.load(RES / "data_size" / gt / "n1024_s0" / "reference.npz", allow_pickle=True)
    t = dict(zip([str(v) for v in z["truth_names"]], z["truth"].astype(float), strict=True))
    return np.array([t[p] for p in NAMES])


def multi_fisher(fi: pd.DataFrame) -> pd.DataFrame:
    """All-12 total-variance ratio combined / single condition (both estimated), median."""
    sys.path.insert(0, str(ROOT / "scripts" / "founder_distribution"))
    import analyse_fisher as AF  # noqa: PLC0415
    import fd_common as F  # noqa: PLC0415

    rows = []
    for gt in F.TRUTHS:
        z = dict(np.load(F.CACHE / f"{gt}.npz"))
        for dname, design in F.DESIGNS.items():
            for name, parts in AF.COMBOS.items():
                J, _ = AF.joint_conditions(z, parts, design)
                I_th = F.schur(J, np.arange(10), np.arange(10, J.shape[0]))
                tr = float(np.trace(np.linalg.pinv(I_th))) / F.N_TOTAL
                for i, dist in enumerate(AF.SINGLES[name]):
                    s = fi[(fi.truth == F.SHORTNAME[gt]) & (fi.design == dname) & (fi.dist == dist)
                           & (fi["mode"] == "E")].trvar12.iloc[0]  # fmt: skip
                    rows.append({"truth": F.SHORTNAME[gt], "design": dname, "combo": name,
                                 "single": dist, "which": i, "ratio_all12": s / tr})  # fmt: skip
    return pd.DataFrame(rows)


def multi_mle() -> pd.DataFrame:
    rows = []
    for f in sorted((RES / "founder_distribution" / "multi").glob("*/*.json")):
        d = json.loads(f.read_text())
        est = np.array([d["conditions"][0][p] for p in NAMES])
        rows.append({"setup": d["setup"], "truth": d["gt"], "seed": d["seed"],
                     "mae12": float(np.mean(np.abs(est - truth12(d["gt"]))))})  # fmt: skip
    return pd.DataFrame(rows)


def heatmap_all12(fi: pd.DataFrame) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: PLC0415

    med = fi.groupby(["design", "dist", "mode"]).gain_all12.median().reset_index()
    med["size"] = med.dist.str.split("_").str[0]
    med["pA"] = med.dist.str.split("pA").str[1].astype(float)
    sizes = ["S1", "S2", "S3", "S4"]
    slab = {"S1": "S1 single", "S2": "S2 mostly single", "S3": "S3 unif. 1-3", "S4": "S4 unif. 1-5"}
    pas = sorted(med.pA.unique(), reverse=True)
    cmap = LinearSegmentedColormap.from_list("rb", ["#b8302f", "#e8584e", "#f0efec", "#6da7ec", "#1c5cab"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for (i, design), (j, mode) in [(a, b) for a in enumerate(("XG2", "R1t")) for b in enumerate(("K", "E"))]:
        ax = axes[i, j]
        M = np.array([[med[(med.design == design) & (med["mode"] == mode) & (med["size"] == sz)
                           & np.isclose(med.pA, pa)].gain_all12.iloc[0] for pa in pas] for sz in sizes])  # fmt: skip
        im = ax.imshow(M, cmap=cmap, norm=LogNorm(1 / 8, 8), aspect="auto")
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(pas)), [f"{p:g}" for p in pas])
        ax.set_yticks(range(4), [slab[s] for s in sizes])
        ax.set_xlabel("p_A (share of A founder cells)")
        ax.set_title(f"{'XG2 design' if design == 'XG2' else 'R1 time points'}, founders "
                     f"{'known' if mode == 'K' else 'estimated'}", fontsize=10)  # fmt: skip
    cb = fig.colorbar(im, ax=axes, shrink=0.7, label="gain vs single A founder, known (same design)")
    ticks = [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]
    cb.set_ticks(ticks, labels=["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    cb.minorticks_off()
    fig.suptitle("Founder distribution and information on all 12 probabilities: "
                 "total variance (reference / setting), median over the 9 truths", fontsize=11)
    out = ROOT / "docs" / "founder_distribution" / "figures"
    fig.savefig(out / "info_all12.png", dpi=150)
    plt.close(fig)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    fi = fisher_table()
    heatmap_all12(fi)
    mf = multi_fisher(fi)
    mf.to_csv(TAB / "all12_multi_fisher.csv", index=False)
    print(mf.groupby(["design", "combo", "which"]).ratio_all12.median().unstack("which").round(2).to_string())
    mm = multi_mle()
    mm.to_csv(TAB / "all12_multi_mle.csv", index=False)
    print(mm.groupby(["setup", "truth"]).mae12.mean().round(3).to_string())
    fi.to_csv(TAB / "all12_fisher.csv", index=False)
    fs = fi.groupby(["design", "dist", "mode"]).gain_all12.median().unstack("mode")
    print(fs.round(2).to_string())
    mle, nuts = founder_sims()
    mle.to_csv(TAB / "all12_founder_mle.csv", index=False)
    nuts.to_csv(TAB / "all12_founder_nuts.csv", index=False)
    print(mle.groupby(["truth", "dist", "model"]).mae12.mean().unstack("model").round(3).to_string())
    print(nuts.groupby(["model"])[["mae12", "cover12"]].mean().round(3).to_string())
    print(nuts.round(3).to_string())
    ff = free_founder_sims()
    ff.to_csv(TAB / "all12_free_founder.csv", index=False)
    print(ff.groupby(["scen", "model"])[["mae12", "cover12"]].agg(["mean", "min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
