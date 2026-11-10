"""Reference MCMC with a fixed RW + independence (t-mixture fitted to pilot draws) kernel.
python reference_mix.py NAME pilot.npz OUT.npz chains steps threads
"""

import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from harness import load

from bio_sbi.inference.smc import _chol, _MixT, convergence, fit_proposal

name, pilot, out = sys.argv[1:4]
C, steps, threads = (int(v) for v in sys.argv[4:7])
lik, *_ = load(name, dt=2.0, threads=threads)
d = lik.dim
rng = np.random.default_rng(777)
P = np.load(pilot)["u"].reshape(-1, d)
q, k = fit_proposal(rng, P, np.ones(len(P)), 8, 1.3, 4.0, n_eff=len(P) / 50, min_points=2)
cov = np.cov(P.T)
wide = _MixT(P.mean(0)[None], (_chol(cov) * 2.0)[None], np.zeros(1), 4.0)
q = _MixT(
    np.vstack([q.means, wide.means]),
    np.concatenate([q.chols, wide.chols]),
    np.r_[q.logpi + np.log(0.8), np.log(0.2)],
    4.0,
)
print("proposal components", len(q.logpi), flush=True)
L = _chol(cov) * 2.38 / np.sqrt(d) * 0.5
cur = P[rng.choice(len(P), C)]
cur_lp = lik.log_posterior_batch(cur)
cur_h = q.logpdf(cur)
hist, acc_r, acc_i = [], [], []
t0 = time.perf_counter()
for it in range(steps):
    imh = rng.uniform(size=C) < 0.5
    prop = cur + rng.standard_normal((C, d)) @ L.T
    prop[imh] = q.sample(rng, int(imh.sum()))
    plp = lik.log_posterior_batch(prop)
    ph = q.logpdf(prop)
    la = plp - cur_lp + np.where(imh, cur_h - ph, 0.0)
    ok = np.isfinite(plp) & (np.log(rng.uniform(size=C)) < la)
    cur[ok], cur_lp[ok], cur_h[ok] = prop[ok], plp[ok], ph[ok]
    acc_r.append((ok & ~imh).sum() / max((~imh).sum(), 1))
    acc_i.append((ok & imh).sum() / max(imh.sum(), 1))
    hist.append(cur.copy())
    if it % 500 == 0:
        print(
            it,
            "acc rw",
            np.mean(acc_r[-500:]),
            "imh",
            np.mean(acc_i[-500:]),
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )
X = np.stack(hist[steps // 10 :], axis=1)
c = convergence(X, 4)
np.savez_compressed(out, u=X, theta=lik.theta_from_u(X.reshape(-1, d)))
print("evals", lik.evals, "seconds", time.perf_counter() - t0)
print("max split rhat", c["max_rhat"], "min ess bulk", c["min_ess_bulk"], "tail", c["min_ess_tail"])
print("ess", np.round(c["ess_bulk"]).tolist())
