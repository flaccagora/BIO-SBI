"""MCMC on the exact Jeffreys prior of the XG2 model (differential-evolution Metropolis).

Target: ``p_J(theta) ∝ sqrt(det I(theta))`` (theta = stick coordinates), sampled in
``u = logit(theta)`` (log Jacobian ``sum log theta (1 - theta)`` added). ``K`` chains in
two halves; each half is updated in one GPU batch with differential-evolution proposals
``u_i + gamma (u_a - u_b) + e`` from two distinct chains of the *other* half (exact
Metropolis for the half being updated given the other: ter Braak 2006 with complementary
sets), ``gamma = 2.38 / sqrt(2 d)`` (1 every 10th iteration for jumps between modes),
``e ~ N(0, 1e-3^2)``. Initial states: draws of the Dirichlet(1/2) importance pool resampled
with tempered weights ``w^0.5`` (over-dispersed). Checkpoint every ``--every`` iterations
(restart-safe). Output: results/exact_gw/data_size_jeffreys/prior/mcmc.npz.

  uv run python scripts/jeffreys/jeffreys_prior_mcmc.py --iters 3000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit, log_expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JEFF, evaluate, jeffreys_prior  # noqa: E402
from jeffreys_prior import load_is  # noqa: E402

OUT = JEFF / "prior"


def log_target(prior, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.clip(expit(U), 1e-15, 1 - 1e-15)
    res = evaluate(prior, theta, log=None, chunk=len(theta))
    jac = (log_expit(U) + log_expit(-U)).sum(1)
    lp = np.where(np.isfinite(res["log_pj"]), res["log_pj"] + jac, -np.inf)
    return lp, res["log_pj"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chains", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", default="is_dir0.5_s0")
    args = ap.parse_args()
    prior = jeffreys_prior()
    K, d = args.chains, prior.dim
    ck = OUT / "mcmc.npz"
    rng = np.random.default_rng(args.seed)
    if ck.exists():
        z = dict(np.load(ck, allow_pickle=False))
        U, lp = z["U"][-1].copy(), z["lp"][-1].copy()
        hist_U, hist_lp, hist_lj = list(z["U"]), list(z["lp"]), list(z["log_pj"])
        acc = list(z["acc"])
        rng = np.random.default_rng([args.seed, len(hist_U)])
        print(f"resumed at iteration {len(hist_U)}", flush=True)
    else:
        pool = load_is(args.pool)
        lw = pool["log_pj"] - pool["log_q"]
        lw = np.where(np.isfinite(lw), lw, -np.inf)
        p = np.exp(0.5 * (lw - lw.max()))
        idx = rng.choice(len(lw), size=K, replace=False, p=p / p.sum())
        U = np.log(pool["theta"][idx]) - np.log1p(-pool["theta"][idx])
        lp, lj = log_target(prior, U)
        hist_U, hist_lp, hist_lj, acc = [U.copy()], [lp.copy()], [lj.copy()], [np.ones(K)]
    gamma0 = 2.38 / np.sqrt(2 * d)
    halves = [np.arange(0, K, 2), np.arange(1, K, 2)]
    t0, it0 = time.time(), len(hist_U)
    while len(hist_U) <= args.iters:
        it = len(hist_U)
        gamma = 1.0 if it % 10 == 0 else gamma0
        a_it = np.zeros(K)
        lj_now = hist_lj[-1].copy()
        for h in (0, 1):
            cur, other = halves[h], halves[1 - h]
            prop = np.empty((len(cur), d))
            for j, i in enumerate(cur):
                a, b = rng.choice(other, 2, replace=False)
                prop[j] = U[i] + gamma * (U[a] - U[b]) + rng.normal(0, 1e-3, d)
            lp_new, lj_new = log_target(prior, prop)
            logu = np.log(rng.uniform(size=len(cur)))
            ok = logu < lp_new - lp[cur]
            U[cur[ok]] = prop[ok]
            lp[cur[ok]] = lp_new[ok]
            lj_now[cur[ok]] = lj_new[ok]
            a_it[cur] = ok
        hist_U.append(U.copy())
        hist_lp.append(lp.copy())
        hist_lj.append(lj_now)
        acc.append(a_it)
        if it % args.every == 0 or len(hist_U) > args.iters:
            np.savez_compressed(ck, U=np.asarray(hist_U), lp=np.asarray(hist_lp),
                                log_pj=np.asarray(hist_lj), acc=np.asarray(acc))  # fmt: skip
            recent = np.asarray(acc[-args.every :]).mean()
            print(
                f"{time.strftime('%F %T')} iter {it}: acceptance {recent:.2f}, log p_J "
                f"median {np.median(lj_now):.1f} max {lj_now.max():.1f} "
                f"({(time.time() - t0) / max(it - it0 + 1, 1):.1f} s/iter)",
                flush=True,
            )


if __name__ == "__main__":
    main()
