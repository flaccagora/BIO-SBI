"""Exact-Jeffreys posterior by differential-evolution MCMC on the exact target (no surrogate).

Arbiter for cells where PSIS (Dirichlet(1/2) draws reweighted) and the surrogate rerun
disagree: target ``L(theta) p_J(theta)`` in ``u = logit(theta)`` (Jacobian included), exact
likelihood (CPU threads) and exact Jeffreys density (GPU) at every proposal. ``K`` chains in
two halves, DE proposals from the other half (as scripts/jeffreys/jeffreys_prior_mcmc.py),
``gamma = 2.38 / sqrt(2 d)`` (1 every 10th iteration). Half the chains start at rerun draws,
half at PSIS-weighted Dirichlet(1/2) draws, so both candidate regions are seeded.
Checkpoints every ``--every`` iterations (restart-safe). Output:
results/exact_gw/data_size_jeffreys/jeffreys_exact/<gt>/n<n>_s<seed>/{demc.npz, summary.json}.

  uv run python scripts/jeffreys/jeffreys_exact_demc.py T1_ref 128 1 --iters 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit, log_expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JEFF, evaluate, jeffreys_prior  # noqa: E402
from jeffreys_rerun import _mle_module  # noqa: E402

OUT = JEFF / "jeffreys_exact"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gt")
    ap.add_argument("n", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("--chains", type=int, default=32)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--burn", type=float, default=0.33)
    args = ap.parse_args()
    cell = f"n{args.n}_s{args.seed}"
    run = JEFF / "dir_half" / args.gt / cell
    out = OUT / args.gt / cell
    out.mkdir(parents=True, exist_ok=True)
    cfg, sim, lik, names, truth, _ = _mle_module()._setup(run)
    prior = jeffreys_prior()
    pool = ThreadPoolExecutor(args.threads)

    def log_target(U):
        th = np.clip(expit(U), 1e-15, 1 - 1e-15)
        ll = np.array(list(pool.map(lambda t: lik.log_likelihood(t, grad=False)[0], th)))
        lj = evaluate(prior, th, log=None, chunk=len(th))["log_pj"]
        jac = (log_expit(U) + log_expit(-U)).sum(1)
        lp = ll + lj + jac
        return np.where(np.isfinite(lp), lp, -np.inf), ll, lj

    K, d = args.chains, prior.dim
    ck = out / "demc.npz"
    if ck.exists():
        z = dict(np.load(ck))
        hist = {k: list(z[k]) for k in ("U", "lp", "ll", "lj", "acc")}
        rng = np.random.default_rng([args.seed, len(hist["U"])])
        print(f"resumed at iteration {len(hist['U'])}", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        a = np.load(JEFF / "jeffreys" / args.gt / cell / "jeffreys_psis.npz")
        starts = [a["theta"][rng.choice(len(a["w"]), K // 2, replace=False, p=a["w"])]]
        r = JEFF / "jeffreys_rerun" / args.gt / cell / "draws.npz"
        if r.exists():
            b = np.load(r)
            starts.append(b["theta"][rng.choice(len(b["theta"]), K - K // 2, replace=False)])
        else:
            starts.append(a["theta"][rng.choice(len(a["theta"]), K - K // 2, replace=False)])
        th0 = np.clip(np.concatenate(starts), 1e-12, 1 - 1e-12)
        U = np.log(th0) - np.log1p(-th0)
        lp, ll, lj = log_target(U)
        hist = {"U": [U], "lp": [lp], "ll": [ll], "lj": [lj], "acc": [np.ones(K)]}
    U, lp = hist["U"][-1].copy(), hist["lp"][-1].copy()
    ll, lj = hist["ll"][-1].copy(), hist["lj"][-1].copy()
    gamma0 = 2.38 / np.sqrt(2 * d)
    halves = [np.arange(0, K, 2), np.arange(1, K, 2)]
    t0, it0 = time.time(), len(hist["U"])
    while len(hist["U"]) <= args.iters:
        it = len(hist["U"])
        gamma = 1.0 if it % 10 == 0 else gamma0
        acc = np.zeros(K)
        for h in (0, 1):
            cur, other = halves[h], halves[1 - h]
            prop = np.empty((len(cur), d))
            for j, i in enumerate(cur):
                a_, b_ = rng.choice(other, 2, replace=False)
                prop[j] = U[i] + gamma * (U[a_] - U[b_]) + rng.normal(0, 1e-3, d)
            lp_n, ll_n, lj_n = log_target(prop)
            ok = np.log(rng.uniform(size=len(cur))) < lp_n - lp[cur]
            idx = cur[ok]
            U[idx], lp[idx], ll[idx], lj[idx] = prop[ok], lp_n[ok], ll_n[ok], lj_n[ok]
            acc[cur] = ok
        for k, v in (("U", U), ("lp", lp), ("ll", ll), ("lj", lj), ("acc", acc)):
            hist[k].append(v.copy())
        if it % args.every == 0 or len(hist["U"]) > args.iters:
            np.savez_compressed(ck, **{k: np.asarray(v) for k, v in hist.items()})
            print(f"{time.strftime('%F %T')} iter {it}: acceptance "
                  f"{np.mean(hist['acc'][-args.every:]):.2f}, log target median {np.median(lp):.1f} "
                  f"({(time.time() - t0) / (it - it0 + 1):.1f} s/iter)", flush=True)  # fmt: skip

    from bio_sbi.inference.smc import ess_bulk, rhat

    Ua = np.asarray(hist["U"])
    start = int(args.burn * len(Ua))
    th = expit(Ua[start:])  # (draws, chains, d)
    chains = np.transpose(th, (1, 0, 2))
    phys = sim.space.decode(th.reshape(-1, d))[:, : len(names)]
    pch = phys.reshape(th.shape[0], K, -1).transpose(1, 0, 2)
    post, diag = {}, {}
    for n in sim.space.free_names:
        j = names.index(n)
        x = phys[:, j]
        post[n] = {"mean": float(x.mean()), "sd": float(x.std()),
                   "q05": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95))}  # fmt: skip
        diag[n] = {"rhat": float(rhat(pch[..., j])), "ess_bulk": float(ess_bulk(pch[..., j]))}
    info = {
        "run": str(run), "iterations": int(len(Ua) - 1), "burn": start, "chains": K,
        "acceptance": float(np.mean(hist["acc"][start:])),
        "max_rhat": max(v["rhat"] for v in diag.values()),
        "min_ess_bulk": min(v["ess_bulk"] for v in diag.values()),
        "max_rhat_theta": float(max(rhat(chains[..., i]) for i in range(d))),
        "posterior": post, "mixing": diag,
    }  # fmt: skip
    (out / "summary.json").write_text(json.dumps(info, indent=1))
    np.savez_compressed(out / "draws.npz", theta=th.reshape(-1, d), phys=phys,
                        w=np.full(len(phys), 1 / len(phys)))  # fmt: skip
    print(json.dumps({k: v for k, v in info.items() if k not in ("posterior", "mixing")}))


if __name__ == "__main__":
    main()
