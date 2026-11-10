"""Long adaptive-Metropolis reference: python reference_mh.py NAME OUT.npz chains burn keep threads"""

import logging
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from harness import load

from bio_sbi.inference.exact_bh import ExactBH
from bio_sbi.inference.smc import convergence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
name, out = sys.argv[1], sys.argv[2]
C, burn, keep, threads = (int(v) for v in sys.argv[3:7])
lik, *_ = load(name, dt=2.0, threads=threads)
rng = np.random.default_rng(12345)
d = lik.dim
t0 = time.perf_counter()
opt = ExactBH.optimise(lik, np.zeros(d), 60, 1e-4)
print("mode", opt["log_posterior"], opt["status"], flush=True)
w, V = np.linalg.eigh(opt["curvature"])
cov = (V / np.maximum(w, 0.25)) @ V.T
cur = opt["u"] + rng.standard_normal((C, d)) @ np.linalg.cholesky(cov).T  # overdispersed-ish
cur_lp = lik.log_posterior_batch(cur)
scale = 2.38 / np.sqrt(d)
hist, acc = [], []
for it in range(burn + keep):
    if it < burn and it >= 100 and it % 50 == 0:  # Haario adaptation from pooled history
        H = np.concatenate(hist[it // 2 :], axis=0)
        cov = np.cov(H.T) + 1e-8 * np.eye(d)
    if it < burn and it % 50 == 0 and acc:
        a = np.mean(acc[-50:])
        scale *= np.exp(a - 0.25)
    L = np.linalg.cholesky(cov)
    prop = cur + scale * rng.standard_normal((C, d)) @ L.T
    plp = lik.log_posterior_batch(prop)
    ok = np.isfinite(plp) & (np.log(rng.uniform(size=C)) < plp - cur_lp)
    cur[ok], cur_lp[ok] = prop[ok], plp[ok]
    acc.append(ok.mean())
    hist.append(cur.copy())
    if it % 500 == 0:
        print(
            it,
            "acc",
            np.mean(acc[-500:]),
            "scale",
            scale,
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )
X = np.stack(hist[burn:], axis=1)  # (C, keep, d)
c = convergence(X, superchains=min(4, C))
theta = lik.theta_from_u(X.reshape(-1, d))
np.savez_compressed(out, u=X, theta=theta)
print("evals", lik.evals, "seconds", time.perf_counter() - t0)
print("max split rhat", c["max_rhat"], "min ess bulk", c["min_ess_bulk"], "tail", c["min_ess_tail"])
