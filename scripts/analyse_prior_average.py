"""X5d: prior-averaged performance (truths drawn from the inference prior).

Per parameter: mean absolute error vs the prior-mean baseline, contraction, 90%
coverage (interior truths) and posterior sd; per run: support mass, PPC, SBC.

Usage: uv run python scripts/analyse_prior_average.py results/x5/d_gw
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sbc_error(cal: dict) -> float:
    errs = [abs(v - float(level)) for level, per in (cal.get("coverage") or {}).items()
            for v in per.values() if v is not None and np.isfinite(v)]  # fmt: skip
    return float(np.mean(errs)) if errs else np.nan


def main(root: Path) -> None:
    ms = [json.loads(p.read_text()) for p in sorted(root.glob("*/metrics.json"))]
    runs = pd.DataFrame([{
        "seed": m["seed"], "support": m["posterior_support_mass"],
        "samples": m["posterior_samples_drawn"],
        "ppc_out95": (m.get("ppc") or {}).get("fraction_outside_95", np.nan),
        "sbc_cov_err": sbc_error(m.get("calibration") or {}),
        "mae_prob": m["recovery"]["mean_abs_error"],
        "rel_err_scalars": m["recovery"].get("mean_rel_error_scalars"),
        "coverage90": m["recovery"]["coverage90"], "truth_draws": m.get("truth_draws"),
    } for m in ms])  # fmt: skip
    rows = []
    for m in ms:
        per = [r for r in m["recovery"]["per_parameter"] if r["inferred"]]
        size = {}
        for r in per:
            size[r["parameter"].rsplit(".", 1)[0]] = size.get(r["parameter"].rsplit(".", 1)[0], 0) + 1
        for r in per:
            row = r["parameter"].rsplit(".", 1)[0]
            scalar = r["parameter"].endswith("mean_lifetime")
            rows.append({"seed": m["seed"], **r, "scalar": scalar,
                         "prior_mean_simple": np.nan if scalar else 1 / size[row]})  # fmt: skip
    d = pd.DataFrame(rows)
    # Free-outcome count per row is the Dirichlet dimension when no fixed entries share
    # the row; fixed entries are not "inferred", so 1 / size is the uniform prior mean.
    d["prior_err"] = (d.truth - d.prior_mean_simple).abs()
    d["rel_err"] = np.where(d.scalar, d.abs_error / d.truth.abs(), np.nan)
    g = d.groupby("parameter", sort=False).agg(
        truth_mean=("truth", "mean"), abs_error=("abs_error", "mean"),
        prior_mean_error=("prior_err", "mean"), rel_error=("rel_err", "mean"),
        contraction=("contraction", "mean"), posterior_sd=("posterior_std", "mean"),
        coverage90=("covered", lambda s: s[~d.loc[s.index, "at_boundary"]].mean()),
    )  # fmt: skip
    runs.to_csv(root / "prior_average_runs.csv", index=False)
    g.round(4).to_csv(root / "prior_average_parameters.csv")
    pd.set_option("display.width", 200)
    print(g.round(3))
    prob = d[~d.scalar]
    print(f"runs {len(runs)}; support median {runs.support.median():.2f} (min {runs.support.min():.2f}); "
          f"PPC {runs.ppc_out95.mean():.3f}; SBC {runs.sbc_cov_err.mean():.3f}; "
          f"MAE prob {prob.abs_error.mean():.3f} vs prior-mean {prob.prior_err.mean():.3f}; "
          f"contraction {d.contraction.mean():.2f}; coverage {d[~d.at_boundary].covered.mean():.2f}; "
          f"redraws {int((runs.truth_draws > 1).sum())}")  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]))
