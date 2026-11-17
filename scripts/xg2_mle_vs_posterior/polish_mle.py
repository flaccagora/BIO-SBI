"""Is the stored MLE (scripts/exact_gw_mle.py, default L-BFGS-B tolerances) the global
maximum? Per run: polish the stored MLE with tight tolerances, and optimise from the
best of 150 posterior draws on the *other* side of the posterior median of B.to_A than
the stored MLE (the other branch of the "switch back" vs "divide" ridge).

  uv run python scripts/xg2_mle_vs_posterior/polish_mle.py [gt ...]
Writes docs/xg2_mle_vs_posterior/polished_mle.csv (resumes: skips runs already there).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from common import HI, LO, OUT, all_runs, setup
from scipy.optimize import minimize

GTS = sys.argv[1:] or ["T1_ref", "T2_toA05", "T3_Bdiv", "T4_slowA"]
OPTS = {"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-7}
f = OUT / "polished_mle.csv"
done = set(pd.read_csv(f).run) if f.exists() else set()
for (gt, seed, n), run in sorted(all_runs().items()):
    tag = f"{gt}_s{seed}_n{n}"
    if gt not in GTS or tag in done or not (run / "mle.json").exists():
        continue
    t0 = time.perf_counter()
    cfg, sim, lik, names, truth, post = setup(run)
    mle = json.loads((run / "mle.json").read_text())
    ia = names.index("B.to_A")
    bdiv = [names.index(k) for k in ("B.div_AA", "B.div_AB", "B.div_BB")]

    def neg(t):
        v, g = lik.log_likelihood(t)
        return -v, -g

    def opt(phys):
        r = minimize(neg, np.clip(sim.encode(phys), LO, HI), jac=True, method="L-BFGS-B",
                     bounds=[(LO, HI)] * sim.space.theta_dim, options=OPTS)  # fmt: skip
        return -r.fun, sim.decode(r.x), r.nit

    phys = post.mean(0).copy()
    for nm, v in mle["mle"].items():
        phys[names.index(nm)] = v
    a_ll, a_x, a_it = opt(phys)
    med = np.median(post[:, ia])
    other = post[:, ia] < med if mle["mle"]["B.to_A"] >= med else post[:, ia] >= med
    idx = np.random.default_rng(0).choice(np.flatnonzero(other), 150, replace=False)
    dll = [lik.log_likelihood(np.clip(sim.encode(post[i]), LO, HI), grad=False)[0] for i in idx]
    b_ll, b_x, b_it = opt(post[idx[int(np.argmax(dll))]])
    best_x = a_x if a_ll >= b_ll else b_x
    row = {"run": tag, "gt": gt, "seed": seed, "n": n, "truth": truth[ia],
           "stored_ll": mle["ll_mle"], "stored_BtoA": mle["mle"]["B.to_A"],
           "prior_starts": mle.get("prior_starts", 0),
           "polished_ll": a_ll, "polished_BtoA": a_x[ia], "polish_iter": a_it,
           "other_ll": b_ll, "other_BtoA": b_x[ia], "other_iter": b_it,
           "best_ll": max(a_ll, b_ll), "best_BtoA": best_x[ia], "best_Bdiv": best_x[bdiv].sum(),
           "gain": max(a_ll, b_ll) - mle["ll_mle"], "post_mean": post[:, ia].mean(),
           "seconds": time.perf_counter() - t0}  # fmt: skip
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    df = pd.DataFrame([row])
    if f.exists():
        df = pd.concat([pd.read_csv(f), df])
    df.to_csv(f, index=False)
