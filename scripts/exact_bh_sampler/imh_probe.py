"""Independence-proposal acceptance of the final-stage t proposal in u vs reparam coordinates.

python imh_probe.py NAME DRAWS NX NY DT SPEC...   (DRAWS: .npz with u chains, or 'rev0_c')
Fits the sampler's t proposal (fit_proposal, scale 1.2, df 5) to posterior draws mapped to
each coordinate system, draws NY proposals, evaluates the exact log posterior there and at
NX posterior draws, and reports the expected IMH acceptance E min(1, w(y) / w(x)) over all
pairs (w = posterior / proposal) and the Pareto k of w(y).
"""

import json
import sys
import time

import numpy as np
import pandas as pd
from harness import load

from bio_sbi.inference.bh_reparam import Reparam
from bio_sbi.inference.exact_bh import ExactBH
from bio_sbi.inference.smc import fit_proposal, psis

name, src = sys.argv[1], sys.argv[2]
nx, ny, dt = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
specs = sys.argv[6:]
lik, *_ = load(name, dt=dt, threads=2)
sp = lik.space
rng = np.random.default_rng(0)
if src == "rev0_c":
    D = "/home/matteo/BIO-SBI/results/exact_bh/smc_check_rev0_c"
    phys = pd.read_csv(D + "/posterior_samples.csv.gz")[list(sp.names)].to_numpy()
    X = lik.u_from_theta(np.array([sp.encode(p) for p in phys], dtype=float))
    anchor = lik.u_from_theta(
        np.array(json.load(open(D + "/metrics.json"))["exact_bh"]["mode_theta"])
    )
else:
    U = np.load(src)["u"]
    X = U[:, U.shape[1] // 10 :: 5].reshape(-1, U.shape[2])
    opt = ExactBH.optimise(lik, np.median(X, 0), 40, 1e-3)
    anchor = opt["u"]
X = X[rng.permutation(len(X))]
t0 = time.perf_counter()
xs = X[:nx]
lp_x_u = lik.log_posterior_batch(xs)
print(f"{nx} draws evaluated ({time.perf_counter() - t0:.0f}s)", flush=True)
for spec in specs:
    rp = Reparam(lik, spec, anchor)
    V = rp.to_v(X)
    q, _ = fit_proposal(rng, V, np.ones(len(V)), 1, 1.2, 5.0, n_eff=len(V))
    Y = q.sample(rng, ny)
    lp_y = lik.log_posterior_batch(rp.to_u(Y)) + rp.log_jac(Y)
    w_y = np.where(np.isfinite(lp_y), lp_y - q.logpdf(Y), -np.inf)
    Vx = rp.to_v(xs)
    w_x = lp_x_u + rp.log_jac(Vx) - q.logpdf(Vx)
    acc = np.minimum(1.0, np.exp(np.clip(w_y[None, :] - w_x[:, None], -50, 50))).mean()
    k = psis(w_y)[1]
    print(
        f"{name} {spec:14s} IMH acceptance {acc:.3f}  Pareto k {k:.2f}  "
        f"({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )
