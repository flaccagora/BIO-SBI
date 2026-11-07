"""X3: Galton-Watson (hours per generation tau) fitted to Bellman-Harris data.

Per arm (GW tau, BH control): probability MAE / coverage / contraction, support, PPC,
SBC; per parameter posterior means. Usage: uv run python scripts/analyse_x3.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/x3")


def sbc_error(cal: dict) -> float:
    errs = [abs(v - float(level)) for level, per in (cal.get("coverage") or {}).items()
            for v in per.values() if v is not None]  # fmt: skip
    return float(np.mean(errs)) if errs else np.nan


def main() -> None:
    runs, pars = [], []
    for path in sorted(ROOT.glob("*/*/metrics.json")):
        m = json.loads(path.read_text())
        ov = (path.parent / ".hydra/overrides.yaml").read_text()
        arm = "BH" if "bellman_harris" in ov else "GW tau=" + ov.split("hours_per_generation=")[1].split()[0]
        pp = [r for r in m["recovery"]["per_parameter"] if r["inferred"] and "lifetime" not in r["parameter"]]
        runs.append({"arm": arm, "seed": m["seed"], "mae": np.mean([r["abs_error"] for r in pp]),
                     "coverage": np.mean([r["covered"] for r in pp if not r["at_boundary"]]),
                     "contraction": np.mean([r["contraction"] for r in pp]),
                     "support": m.get("posterior_support_mass"),
                     "ppc_out95": (m.get("ppc") or {}).get("fraction_outside_95", np.nan),
                     "sbc": sbc_error(m.get("calibration") or {}),
                     "samples": m.get("posterior_samples_drawn")})  # fmt: skip
        pars += [{"arm": arm, "seed": m["seed"], **r} for r in pp]
    r, p = pd.DataFrame(runs), pd.DataFrame(pars)
    g = r.groupby("arm").mean(numeric_only=True).drop(columns="seed")
    means = p.pivot_table(index="parameter", columns="arm", values="posterior_mean")
    means.insert(0, "truth", p.groupby("parameter").truth.first())
    cov = p.pivot_table(index="parameter", columns="arm", values="covered", aggfunc="sum")
    g.round(4).to_csv(ROOT / "x3_groups.csv")
    means.round(4).to_csv(ROOT / "x3_parameters.csv")
    r.to_csv(ROOT / "x3_runs.csv", index=False)
    pd.set_option("display.width", 250)
    print(g.round(3), means.round(2), cov, sep="\n\n")


if __name__ == "__main__":
    main()
