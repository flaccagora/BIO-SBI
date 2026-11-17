"""Sanity check of SubModel: matches ExactLikelihood, gradient vs finite differences, timing."""

import json
import time

import numpy as np
from common import LO, HI, SubModel, all_runs, setup

runs = all_runs()
print(len(runs), sorted({k[0] for k in runs}))
run = runs[("T1_ref", 0, 128)]
cfg, sim, lik, names, truth, post = setup(run)
print(names)
pm = post.mean(0)
sm = SubModel(lik, {})
v1 = sm.loglik(sm.encode(pm), grad=False)[0]
v2 = lik.log_likelihood(np.clip(sim.encode(pm), LO, HI), grad=False)[0]
print("ll submodel vs exact", v1, v2)
sm2 = SubModel(lik, {"B.to_A": 0.5})
s2 = sm2.encode(pm)
v, g = sm2.loglik(s2)
eps = 1e-6
gn = []
for i in range(sm2.dim):
    e = np.zeros(sm2.dim)
    e[i] = eps
    gn.append((sm2.loglik(s2 + e, False)[0] - sm2.loglik(s2 - e, False)[0]) / 2 / eps)
print(np.round(g, 4))
print(np.round(gn, 4))
print(dict(zip(names, sm2.physical(s2).round(3), strict=True)))
t = time.time()
for _ in range(10):
    sm2.loglik(s2)
print("grad eval s", (time.time() - t) / 10)
t = time.time()
for _ in range(10):
    sm2.loglik(s2, False)
print("obj eval s", (time.time() - t) / 10)
m = json.loads((run / "mle.json").read_text())
print(m["ll_mle"], m["mle"]["B.to_A"])
