"""End-to-end NUTS (ExactGW.run) with kernel=loop vs kernel=adjoint on one dataset.

Same dataset, seed and settings; 2 chains in 2 threads. Reports wall time, gradient
evaluations, seconds per evaluation, posterior means / sds of the physical parameters and
their difference in units of the Monte Carlo standard error.

    OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 uv run python scripts/gw_speedup/nuts_compare.py \
        > scripts/gw_speedup/nuts_compare.txt
"""

import sys

import numpy as np

from bio_sbi.inference import ExactGW
from bio_sbi.inference.nuts import ess
from profile_parts import make

name = sys.argv[1] if len(sys.argv) > 1 else "two_type"
gens = tuple(int(g) for g in (sys.argv[2] if len(sys.argv) > 2 else "4,6,8,10").split(","))
clones = int(sys.argv[3]) if len(sys.argv) > 3 else 128
settings = dict(chains=2, warmup=300, draws=300, threads=2, init_starts=2, log_every=0, seed=0)
print(f"design {name} generations {gens} x {clones} clones; {settings}")
res = {}
for kernel in ("loop", "adjoint"):
    method = ExactGW(kernel=kernel, **settings)
    lik, _, sim = make(name, gens, clones)
    lik.kernel = kernel
    theta, diag = method.run(lik)
    phys = sim.decode(theta)
    res[kernel] = (phys, diag)
    print(f"{kernel:8s} wall {diag['seconds']:.1f}s, init {diag['init_seconds']:.1f}s, "
          f"chains {np.round(diag['chain_seconds'], 1).tolist()}s, gradient evals "
          f"{diag['gradient_evals']}, s/eval {diag['seconds_per_eval'] * 1e3:.2f} ms, max R-hat "
          f"{diag['max_rhat']:.3f}, min ESS {diag['min_ess']:.0f}, divergences "
          f"{diag['divergences']}, mean tree depth {np.round(diag['mean_tree_depth'], 2).tolist()}",
          flush=True)  # fmt: skip
names = sim.space.names
free = [i for i, n in enumerate(names) if n in set(sim.space.free_names)]
(p0, d0), (p1, d1) = res["loop"], res["adjoint"]
print("parameter | loop mean (sd) | adjoint mean (sd) | |diff| / MCSE")
for i in free:
    a, b = p0[:, i], p1[:, i]
    n0 = ess(a.reshape(settings["chains"], -1))
    n1 = ess(b.reshape(settings["chains"], -1))
    mcse = np.sqrt(a.var() / n0 + b.var() / n1)
    print(f"{names[i]} | {a.mean():.4f} ({a.std():.4f}) | {b.mean():.4f} ({b.std():.4f}) | "
          f"{abs(a.mean() - b.mean()) / mcse:.2f}")  # fmt: skip
print(f"wall-time ratio loop/adjoint: {d0['seconds'] / d1['seconds']:.2f}x; per-eval ratio "
      f"{d0['seconds_per_eval'] / d1['seconds_per_eval']:.2f}x")
