"""E2c rerun (truths drawn from the three-type prior): compare the three inference arms.

Per run: support mass, posterior samples drawn, PPC, SBC coverage error, and recovery of
the A/B parameters that exist in every arm (A.death, A.persist, A.to_B, A.div_AA,
A.div_AB, A.div_BB, B.death, B.to_A, B.persist). Per truth: the fraction of expected
cells in C, how much truth mass the "ignore C" arm cannot represent (B divisions), and
how much the trajectory arm forbids (C -> A outcomes).

Usage: uv run python scripts/analyse_e2c_prior.py [results/e2c_prior]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHARED = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
          "B.death", "B.to_A", "B.persist"]  # fmt: skip
ARMS = ["ignore_c", "model_c_free", "model_c_traj"]


def sbc_error(cal: dict) -> float:
    """Mean |empirical - nominal| SBC coverage over all levels and parameters."""
    errs = [abs(v - float(level)) for level, per in (cal.get("coverage") or {}).items()
            for v in per.values() if v is not None and np.isfinite(v)]  # fmt: skip
    return float(np.mean(errs)) if errs else np.nan


def main(root: Path) -> None:
    rows, params = [], []
    for arm in ARMS:
        for path in sorted((root / arm).glob("*/metrics.json")):
            m = json.loads(path.read_text())
            q = float(next(o.split("=")[1] for o in (path.parent / ".hydra/overrides.yaml")
                           .read_text().split() if "observation.q=" in o))  # fmt: skip
            truth = {r["parameter"]: r["truth"] for r in m["recovery"]["per_parameter"]}
            ref = np.load(path.parent / "reference.npz")
            full = dict(zip([str(n) for n in ref["truth_names"]], ref["truth"].tolist()))
            ec = m["reference_expected_counts"]
            last = ec[max(ec, key=float)]
            per = {r["parameter"]: r for r in m["recovery"]["per_parameter"]}
            shared = [per[p] for p in SHARED if p in per and per[p]["inferred"]]
            err = [r["abs_error"] for r in shared]
            cov = [r["covered"] for r in shared if not r["at_boundary"]]
            cal = m.get("calibration") or {}
            rows.append({
                "arm": arm, "q": q, "seed": m["seed"],
                "C_fraction_last": last["C"] / sum(last.values()),
                "B_div_mass": sum(full.get(f"B.div_{k}", 0) for k in ("AA", "AB", "BB", "AC", "BC", "CC")),
                "C_to_A_mass": sum(full.get(k, 0) for k in ("C.to_A", "C.div_AA", "C.div_AB", "C.div_AC")),
                "support_mass": m["posterior_support_mass"],
                "samples": m["posterior_samples_drawn"],
                "ppc_out95": (m.get("ppc") or {}).get("fraction_outside_95", np.nan),
                "sbc_cov_err": sbc_error(cal),
                "shared_mae": np.mean(err) if err and m["posterior_samples_drawn"] >= 100 else np.nan,
                "shared_cov90": np.mean(cov) if cov and m["posterior_samples_drawn"] >= 100 else np.nan,
                "all_contraction": m["recovery"]["mean_contraction"],
            })  # fmt: skip
            for r in shared:
                params.append({"arm": arm, "q": q, "seed": m["seed"], **r})
    df = pd.DataFrame(rows)
    df.to_csv(root / "e2c_prior_runs.csv", index=False)
    pd.DataFrame(params).to_csv(root / "e2c_prior_shared_parameters.csv", index=False)
    usable = df.samples >= 100
    agg = df.assign(usable=usable, flagged=(df.support_mass < 0.1) | (df.ppc_out95 > 0.05)).groupby(["arm", "q"]).agg(
        runs=("seed", "size"), usable=("usable", "sum"), flagged=("flagged", "sum"),
        support=("support_mass", "median"), ppc_out95=("ppc_out95", "mean"),
        sbc_cov_err=("sbc_cov_err", "mean"), shared_mae=("shared_mae", "mean"),
        shared_cov90=("shared_cov90", "mean"), contraction=("all_contraction", "mean"))  # fmt: skip
    pd.set_option("display.width", 200)
    print(agg.round(3))
    truths = df[df.arm == "ignore_c"].groupby("seed")[["C_fraction_last", "B_div_mass", "C_to_A_mass"]].first()
    print(truths.round(3))
    agg.round(4).to_csv(root / "e2c_prior_groups.csv")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "results/e2c_prior"))
