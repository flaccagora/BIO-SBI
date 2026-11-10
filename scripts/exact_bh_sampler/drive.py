"""Run ExactBH.run on a stored reference dataset: python drive.py NAME OUT.json key=value..."""

import json
import logging
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from harness import load

from bio_sbi.inference.exact_bh import ExactBH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
name, out = sys.argv[1], sys.argv[2]
kw = {}
for a in sys.argv[3:]:
    k, v = a.split("=")
    try:
        v = json.loads(v)
    except Exception:
        pass
    kw[k] = v
threads = int(kw.pop("threads", 16))
dt = float(kw.pop("dt", 2.0))
seed = int(kw.pop("seed", 0))
lik, comp, cfg, d = load(name, dt=dt, threads=threads)
print("grid", lik.grid.Lx, lik.grid.Ly, "dim", lik.dim, flush=True)
m = ExactBH(threads=threads, dt=dt, seed=seed, **kw)
t = time.perf_counter()
theta, w, diag = m.run(lik)
diag["wall"] = time.perf_counter() - t
U = lik.u_from_theta(theta)
extra = {}
if getattr(m, "last_smc", None) is not None:
    ch = m.last_smc.chains
    if getattr(m, "last_to_u", None) is not None:
        ch = m.last_to_u(ch.reshape(-1, ch.shape[2])).reshape(ch.shape)
    extra = {"chains": ch, "chain_lp": m.last_smc.log_post}
np.savez_compressed(out.replace(".json", ".npz"), theta=theta, w=w, u=U, **extra)
lik.close()
json.dump(diag, open(out, "w"), indent=1, default=float)
print("mean", {k: round(v, 3) for k, v in diag["weighted_mean"].items()})
print("sd", {k: round(v, 3) for k, v in diag["weighted_sd"].items()})
print("evals", diag["likelihood_evals"], "grads", diag["gradient_evals"], "wall", diag["wall"])
