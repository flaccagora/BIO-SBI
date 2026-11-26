"""Locations of the L-BFGS posterior optima used to start the free-founder NUTS chains.

``ExactGW.run`` reports only the optimum values (``init_log_posterior``); this replays the
same 8 prior-draw starts (same seed / key, so the same optima) and records where each
optimum lies (p_A, doublet fraction, B->A, m_BA): separate optima = separate branches.

    uv run python scripts/free_founder/optima.py --workers 2
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import run as R  # noqa: E402


def one(path: str) -> dict:
    from scipy.optimize import minimize

    from bio_sbi.inference import dediff_lrt as dl
    from bio_sbi.inference.exact_gw import U_BOUND, ExactLikelihood
    from bio_sbi.rng import rng_for

    r = json.loads(Path(path).read_text())
    scen, model, gt, seed = r["scen"], r["model"], r["gt"], r["seed"]
    cohorts, _ = R.dataset_with_founders(scen, gt, seed)
    sim = R.make_sim(R.MODELS[model])
    lik = ExactLikelihood(sim, dl.observation(None), cohorts, kernel="adjoint")
    rng = rng_for(seed, "exact-init", "free_founder", scen, model, gt, seed)
    theta = np.clip(rng.beta(lik.prior_a, lik.prior_b, size=(8, lik.dim)), 1e-6, 1 - 1e-6)
    u0 = np.clip(np.log(theta) - np.log1p(-theta), -8, 8)

    def neg(u):
        lp, g = lik.log_posterior_u(u)
        return (1e300, np.zeros_like(u)) if not np.isfinite(lp) else (-lp, -g)

    out = []
    for u in u0:
        res = minimize(neg, u, jac=True, method="L-BFGS-B",
                       bounds=[(-U_BOUND, U_BOUND)] * len(u), options={"maxiter": 200})  # fmt: skip
        phys = sim.space.decode(1 / (1 + np.exp(-res.x)))
        P = phys[:12].reshape(2, 6)
        s = R.C.summaries(P)
        names = list(sim.space.names)
        ll = lik.log_likelihood(1 / (1 + np.exp(-res.x)), grad=False)[0]
        out.append({"lp": float(-res.fun), "ll": float(ll), "founder.A": float(phys[names.index("founder.A")]),
                    "founder_size.2": float(phys[names.index("founder_size.2")])
                    if "founder_size.2" in names else 0.0,
                    "B.to_A": float(P[1, 1]), "m_BA": s["m_BA"], "rho": s["rho"]})  # fmt: skip
    out.sort(key=lambda d: -d["lp"])
    # log posterior at the truth (founder rows at their true values where the model has them)
    o = R.C.SCENARIOS[scen][1]
    P = R.C.TRUTHS[gt]
    m = {"A": dict(zip(dl.ENTRIES["A"], P[0], strict=True)),
         "B": dict(zip(dl.ENTRIES["B"], P[1], strict=True)),
         "founder": {"A": o.get("founder_A", 1.0), "B": 1 - o.get("founder_A", 1.0)}}  # fmt: skip
    if "founder_size.2" in sim.space.names:
        m["founder_size"] = {"1": 1 - o.get("doublet", 0.0), "2": o.get("doublet", 0.0)}
    t = np.clip(sim.space.encode(sim.physical_from_mapping(m)).astype(float), 1e-9, 1 - 1e-9)
    lp_truth = float(lik.log_posterior_u(np.log(t) - np.log1p(-t))[0])
    ll_truth = float(lik.log_likelihood(t, grad=False)[0])
    return {"scen": scen, "model": model, "gt": gt, "seed": seed, "optima": out,
            "lp_truth": lp_truth, "ll_truth": ll_truth}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    dest = R.OUT / "optima.json"
    done = json.loads(dest.read_text()) if dest.exists() else []
    have = {(d["scen"], d["model"], d["gt"], d["seed"]) for d in done}
    todo = []
    for f in sorted(R.OUT.glob("*/*/*.json")):
        p = f.relative_to(R.OUT).parts
        key = (p[0], p[1], f.stem.rsplit("_s", 1)[0], int(f.stem.rsplit("_s", 1)[1]))
        if key not in have:
            todo.append(str(f))
    print(f"{len(todo)} datasets", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for res in pool.map(one, todo):
            done.append(res)
            dest.write_text(json.dumps(done, indent=1))
            o = res["optima"]
            print(res["scen"], res["model"], res["gt"], "ll_truth", round(res["ll_truth"], 1),
                  [(round(d["ll"], 1), round(d["founder.A"], 2), round(d["founder_size.2"], 2),
                    round(d["B.to_A"], 2)) for d in o], flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
