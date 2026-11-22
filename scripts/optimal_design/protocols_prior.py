"""The practical protocols evaluated on the 60 flat-prior draws (out-of-sample robustness).

Efficiencies relative to each draw's own optimum over generations 1..10 (the prior caches
stop at 10); B->A sd at N = 512 and clones for sd <= 0.05.

  uv run --with cvxpy python scripts/optimal_design/protocols_prior.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from design_lib import criteria, efficiencies, optimise  # noqa: E402
from od_common import OUT, POINTS, PRIOR_PREC  # noqa: E402
from optimise import PROTOCOLS, load, write  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    draws = np.load(OUT / "prior_draws.npz")["p12"]
    sub = np.array([h <= 10 for _, h in POINTS])
    rows = []
    for i in range(len(draws)):
        d = load(f"prior{i:03d}")
        K = np.zeros((len(POINTS),) + d["K"].shape[1:])
        K[sub] = d["K"]
        o = {}
        for crit in ("E", "c", "D"):
            w = optimise(K, d["c"], crit, mask=sub)
            o[crit] = {"w": w, **criteria(K, d["c"], w)}
        for name, w in PROTOCOLS.items():
            e = efficiencies(K, d["c"], w, o)
            rows.append({"protocol": name, "draw": i, "eff_E": e["eff_E"], "eff_c": e["eff_c"],
                         "sd_BtoA_512": float(np.sqrt(e["cvar"] / PRIOR_PREC / 512)),
                         "N_sd_0.05": e["cvar"] / PRIOR_PREC / 0.05**2})  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "protocols_prior_long.csv", index=False)
    cur = df[df.protocol == "current: A 4/6/8/10"].set_index("draw")
    df["gain_lam"] = df.apply(lambda r: r.eff_E / cur.loc[r.draw, "eff_E"], axis=1)
    df["gain_var"] = df.apply(lambda r: r.eff_c / cur.loc[r.draw, "eff_c"], axis=1)
    q = df.groupby("protocol", sort=False).agg(
        median_eff_E=("eff_E", "median"), p10_eff_E=("eff_E", lambda x: x.quantile(0.1)),
        median_eff_c=("eff_c", "median"), p10_eff_c=("eff_c", lambda x: x.quantile(0.1)),
        median_gain_lam=("gain_lam", "median"), median_gain_var=("gain_var", "median"),
        median_sd_512=("sd_BtoA_512", "median"), p90_sd_512=("sd_BtoA_512", lambda x: x.quantile(0.9)),
        median_N_sd005=("N_sd_0.05", "median"), p90_N_sd005=("N_sd_0.05", lambda x: x.quantile(0.9)),
    ).reset_index()  # fmt: skip
    write(q, "protocols_prior")
    print(q.round(3).to_string())


if __name__ == "__main__":
    main()
