"""Profile KL over B-division mass for the early-endpoint designs ({3}, {2,3}) and, for
comparison, the local Fisher (quadratic) prediction delta^2 / (2 sigma^2).

Checks whether the CR-based claim "early endpoints carry 10-15x the information" holds
globally or only locally. Writes profile_early.jsonl next to this script.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile as pr  # noqa: E402
from gwexact import THETA_STAR, law, law_and_grads, conditional_fisher, tangent_basis  # noqa: E402

DESIGNS = {"early_3": {3: 1.0}, "early_2_3": {2: 0.5, 3: 0.5}, "reallike_6_10": {6: 165 / 266, 10: 101 / 266}}
DELTAS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_early.jsonl")

T = tangent_basis(); gd = np.zeros(10); gd[7:] = 1
with open(OUT, "w") as fh:
    for name, des in DESIGNS.items():
        I = sum(w * conditional_fisher(*law_and_grads(THETA_STAR, n, T, L=pr.Ls[n]))[0] for n, w in des.items())
        var_delta = gd @ np.linalg.inv(I) @ gd          # per-clone CR variance of delta
        pr.STAR.clear()
        for n in des:
            q = law(THETA_STAR, n, pr.Ls[n]).copy(); q[0, 0] = 0; pr.STAR[n] = q / q.sum()
        for delta in DELTAS:
            best = None
            for z0 in pr.starts():
                r = minimize(pr.objective, z0, args=(delta, des), jac=True, method="L-BFGS-B",
                             options=dict(maxiter=500, gtol=1e-10))
                if best is None or r.fun < best.fun:
                    best = r
            P = pr.unpack(best.x, delta)
            rec = dict(design=name, delta=delta, kl_per_clone=float(best.fun),
                       quadratic_prediction=delta**2 / (2 * var_delta),
                       B_to_A=round(float(P[8]), 3), A_death=round(float(P[0]), 3))
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"{name:14s} delta {delta:.2f}  K {rec['kl_per_clone']:.2e}  quad {rec['quadratic_prediction']:.2e}"
                  f"  x266 {266 * rec['kl_per_clone']:6.2f} nats  B.to_A {rec['B_to_A']:.2f}  A.death {rec['A_death']:.2f}",
                  flush=True)
