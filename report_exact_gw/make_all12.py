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


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    fi = fisher_table()
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
