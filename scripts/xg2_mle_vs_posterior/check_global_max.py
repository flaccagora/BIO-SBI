"""Is the best profile point a better optimum than the stored MLE? Re-evaluate it with the
run's own ExactLikelihood (theta coordinates, as scripts/exact_gw_mle.py does) and polish
it with an unconstrained L-BFGS-B in theta.

  uv run python scripts/xg2_mle_vs_posterior/check_global_max.py T1_ref:0:128 ...
Appends to docs/xg2_mle_vs_posterior/global_max.csv.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from common import HI, LO, OUT, all_runs, setup
from scipy.optimize import minimize

rows = []
for arg in sys.argv[1:]:
    g, s, n = arg.split(":")
    key = (g, int(s), int(n))
    run = all_runs()[key]
    z = np.load(OUT / "profiles" / f"{g}_s{s}_n{n}.npz")
    cfg, sim, lik, names, truth, post = setup(run)
    mle = json.loads((run / "mle.json").read_text())
    j = int(np.argmax(z["prof_ll"]))
    th = np.clip(sim.encode(z["prof_phys"][j]), LO, HI)
    v0 = float(lik.log_likelihood(th, grad=False)[0])

    def neg(t):
        v, gr = lik.log_likelihood(t)
        return -v, -gr

    r = minimize(neg, th, jac=True, method="L-BFGS-B", bounds=[(LO, HI)] * len(th))
    phys = sim.decode(r.x)
    ia = names.index("B.to_A")
    rows.append({"run": f"{g}_s{s}_n{n}", "stored_ll_mle": mle["ll_mle"],
                 "stored_mle_BtoA": mle["mle"]["B.to_A"], "prior_starts": mle.get("prior_starts", 0),
                 "profile_best_c": z["grid"][j], "profile_best_ll_recomputed": v0,
                 "polished_ll": -r.fun, "polished_BtoA": phys[ia],
                 "polished_Bdiv": phys[[names.index(k) for k in ("B.div_AA", "B.div_AB", "B.div_BB")]].sum(),
                 "gain_over_stored": -r.fun - mle["ll_mle"]})  # fmt: skip
    print(rows[-1], flush=True)
f = OUT / "global_max.csv"
df = pd.DataFrame(rows)
if f.exists():
    old = pd.read_csv(f)
    df = pd.concat([old[~old.run.isin(df.run)], df])
df.to_csv(f, index=False)
