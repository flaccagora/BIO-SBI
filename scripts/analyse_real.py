"""R: compare all real-data model fits (results/real/<model>/fit_<variant>/).

Per model x data variant: posterior support mass, samples drawn, PPC fraction outside
95%, z of the A-B correlation and of sd(log total) per horizon, and key posterior means.
Usage: uv run python scripts/analyse_real.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/real")
KEY = ["A.div_AA", "B.div_BB", "B.to_A", "founder.A", "A.mean_lifetime",
       "B.mean_lifetime", "clone.kappa", "capacity.K"]  # fmt: skip


def main() -> None:
    rows = []
    for fit in sorted(ROOT.glob("*/fit_*/metrics.json")):
        model, variant = fit.parts[-3], fit.parts[-2].removeprefix("fit_")
        m = json.loads(fit.read_text())
        row = {"model": model, "variant": variant,
               "support": m.get("posterior_support_mass"),
               "drawn": m.get("posterior_samples_drawn"),
               "ppc_out95": (m.get("ppc") or {}).get("fraction_outside_95", np.nan)}  # fmt: skip
        ppc = fit.parent / "ppc.csv"
        if ppc.exists():
            t = pd.read_csv(ppc).set_index("feature")
            for f in t.index:
                if f.endswith(("corr_A_B", "total.sd_log", "total.mean_log")):
                    row[f"z_{f}"] = t.loc[f, "z"]
        for k in KEY:
            p = (m.get("parameters") or {}).get(k) or {}
            if "mean" in p:
                row[k] = p["mean"]
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "real_fits.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
