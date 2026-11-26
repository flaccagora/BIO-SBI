"""Two checks per run, cheaper than a full profile:

1. Is the stored MLE (scripts/exact_gw_mle.py: multistart L-BFGS-B in stick coordinates)
   the global maximum, or is there a higher optimum on the low-B.to_A branch ("B divides /
   dies" instead of "B switches back")? Evaluate 150 posterior draws; (a) optimise freely
   (stick coordinates, tight tolerances) from the best draw with B.to_A < 0.35 - this is
   what a multistart does; (b) pin B.to_A at 0.1, 0.2, 0.3 and maximise the other 9 from
   the 2 best draws near each value (SubModel), then release the pin from the best of
   these (free optimisation, tight tolerances).
2. Sub-model "B never divides" (B.div_* = 0): best fit (5 starts) and the likelihood
   ratio vs the full model.

  uv run python scripts/xg2_mle_vs_posterior/low_branch.py GT[:N] ...
  e.g. T1_ref T2_toA05 T3_Bdiv T4_slowA P1:128 P2:128 P3:128 P4:128
Writes docs/xg2_mle_vs_posterior/low_branch.csv (resumes).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from common import HI, LO, OUT, SubModel, all_runs, setup
from scipy.optimize import minimize

OPTS = {"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-7}
DIV = ("B.div_AA", "B.div_AB", "B.div_BB")
sel = [(a.split(":")[0], int(a.split(":")[1]) if ":" in a else None) for a in sys.argv[1:]]
f = OUT / "low_branch.csv"
done = set(pd.read_csv(f).run) if f.exists() else set()
for (gt, seed, n), run in sorted(all_runs().items(), key=lambda kv: (kv[0][2], kv[0])):
    tag = f"{gt}_s{seed}_n{n}"
    if not any(gt == g and (nn is None or nn == n) for g, nn in sel) or tag in done:
        continue
    if not (run / "mle.json").exists():
        continue
    t0 = time.perf_counter()
    cfg, sim, lik, names, truth, post = setup(run)
    mle = json.loads((run / "mle.json").read_text())
    ia = names.index("B.to_A")
    bdiv = [names.index(k) for k in DIV]

    def neg(t):
        v, g = lik.log_likelihood(t)
        return -v, -g

    def free_opt(phys):
        r = minimize(neg, np.clip(sim.encode(phys), LO, HI), jac=True, method="L-BFGS-B",
                     bounds=[(LO, HI)] * sim.space.theta_dim, options=OPTS)  # fmt: skip
        return -r.fun, sim.decode(r.x)

    idx = np.random.default_rng(0).choice(len(post), 150, replace=False)
    dll = np.array([lik.log_likelihood(np.clip(sim.encode(post[i]), LO, HI), grad=False)[0]
                    for i in idx])  # fmt: skip
    low = idx[post[idx, ia] < 0.35]
    row = {"run": tag, "gt": gt, "seed": seed, "n": n, "truth": truth[ia],
           "stored_ll": mle["ll_mle"], "stored_BtoA": mle["mle"]["B.to_A"],
           "prior_starts": mle.get("prior_starts", 0), "post_mean": post[:, ia].mean(),
           "frac_draws_below_0.35": float((post[:, ia] < 0.35).mean())}  # fmt: skip
    best_low = (-np.inf, None)
    if len(low):
        i0 = low[np.argmax(dll[np.isin(idx, low)])]
        row["free_from_low_draw_ll"], x = free_opt(post[i0])
        row["free_from_low_draw_BtoA"] = x[ia]
        best_low = (row["free_from_low_draw_ll"], x)
        sub = SubModel(lik, {"B.to_A": 0.2})
        pinned = []
        for c in (0.1, 0.2, 0.3):
            sub.set_pinned({"B.to_A": c})
            near = idx[np.abs(post[idx, ia] - c) < 0.05]
            cand = near[np.argsort(dll[np.isin(idx, near)])[::-1][:2]] if len(near) else [i0]
            for i in cand:
                p = post[i].copy()
                b = [k for k, nm in enumerate(names) if nm.startswith("B.") and nm != "B.to_A"]
                p[b] *= (1 - c) / p[b].sum()
                p[ia] = c
                r = sub.maximise(sub.encode(p))
                pinned.append((r["ll"], sub.physical(r["s"])))
        pl, px = max(pinned, key=lambda t: t[0])
        row["pinned_best_ll"], row["pinned_best_c"] = pl, px[ia]
        rl, rx = free_opt(px)
        row["released_ll"], row["released_BtoA"] = rl, rx[ia]
        row["released_Bdiv"] = rx[bdiv].sum()
        if rl > best_low[0]:
            best_low = (rl, rx)
    row["low_branch_ll"] = best_low[0]
    row["low_minus_stored"] = best_low[0] - mle["ll_mle"]
    gl = max(best_low[0], mle["ll_mle"])
    row["global_BtoA"] = best_low[1][ia] if best_low[0] > mle["ll_mle"] else mle["mle"]["B.to_A"]

    # sub-model "B never divides"
    nod = SubModel(lik, dict.fromkeys(DIV, 0.0))
    phys = post.mean(0).copy()
    for nm, v in mle["mle"].items():
        phys[names.index(nm)] = v
    starts = [phys, post[idx[np.argmax(dll)]], post.mean(0)]
    starts += [post[i] for i in idx[np.argsort(post[idx][:, bdiv].sum(1))[:2]]]
    res = [nod.maximise(nod.encode(s)) for s in starts]
    rb = max(res, key=lambda r: r["ll"])
    row["nodiv_ll"], row["nodiv_BtoA"] = rb["ll"], nod.physical(rb["s"])[ia]
    row["LR_nodiv"] = 2 * (gl - rb["ll"])
    row["seconds"] = time.perf_counter() - t0
    print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
           for k, v in row.items()}, flush=True)  # fmt: skip
    df = pd.DataFrame([row])
    if f.exists():
        df = pd.concat([pd.read_csv(f), df])
    df.to_csv(f, index=False)
