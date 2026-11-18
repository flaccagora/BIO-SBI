"""Posterior mass of the two B-row branches of an XG2 cell by per-branch tempered SMC.

The posterior is split at ``B.to_A = t`` ("B divides / is lost" below, "B switches back"
above). In ``brow`` coordinates ``B.to_A`` is the first stick of B's row, a
``Beta(alpha_toA, sum of the other alphas)`` independent of the other sticks, so each
branch is a box ``{s_toA < t}`` / ``{s_toA >= t}``. Per branch, tempered SMC runs from the
prior truncated to the branch (normalised: the stick's Beta truncated by inverse cdf) to
the posterior truncated to the branch; its evidence estimate is ``Z_b = int_branch
exp(log posterior)``, so the branch mass is ``Z_b / sum Z``. ``--reps`` independent runs
per branch give the Monte-Carlo spread. The final-stage populations (equal weights) are
saved too (within-branch posterior draws).

    uv run python scripts/exact_gw_reparam/branch_evidence.py --cell T1_n1024_s0 --reps 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).parent))
from harness import load  # noqa: E402

from bio_sbi.inference.gw_reparam import StickCoordinates  # noqa: E402
from bio_sbi.inference.smc import tempered_smc  # noqa: E402

OUT = Path("/home/matteo/BIO-SBI/results/exact_gw/reparam_bench")


def branch_target(coords: StickCoordinates, k: int, t: float, low: bool, pool):
    """``(logpost, ref_sample, ref_logpdf)`` of one branch (truncated at stick ``k = t``)."""
    a, b = coords.a[k], coords.b[k]
    F = beta_dist.cdf(t, a, b)
    lo, hi = (0.0, F) if low else (F, 1.0)
    log_mass = np.log(hi - lo)

    def inside(V):
        s = coords.sticks(V[:, k])[0]
        return s < t if low else s >= t

    def logpost(V):
        V = np.asarray(V)
        out = np.full(len(V), -np.inf)
        ok = np.flatnonzero(inside(V))
        if len(ok):
            out[ok] = np.fromiter(pool.map(coords.log_posterior_value, V[ok]), float, len(ok))
        return out

    def ref_sample(rng, n):
        V = coords.sample_prior(rng, n)
        u = rng.uniform(lo, hi, size=n)
        s = np.clip(beta_dist.ppf(u, a, b), 1e-300, 1 - 1e-16)
        V[:, k] = coords.from_sticks(s)
        return V

    def ref_logpdf(V):
        lp = coords.log_prior(V, normalised=True) - log_mass
        return np.where(inside(np.asarray(V)), lp, -np.inf)

    return logpost, ref_sample, ref_logpdf


def main() -> None:
    """Run the per-branch SMC replicates and print the branch masses."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cell", default="T1_n1024_s0")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--spec", default="brow+probit", help="brow or brow+probit")
    p.add_argument("--chains", type=int, default=128)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--ess-fraction", type=float, default=0.7)
    p.add_argument("--final-steps", type=int, default=24)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="")
    a = p.parse_args()
    out = OUT / a.cell / f"branch_evidence{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()],
    )
    _, lik, _, truth = load(a.cell)
    coords = StickCoordinates(lik, a.spec)
    k = next(b[4].start for b in coords.blocks if b[0] == lik.sim.law.types[1])
    assert coords.orders[lik.sim.law.types[1]][0] == f"to_{lik.sim.law.types[0]}"
    results, pops = [], {}
    with ThreadPoolExecutor(a.threads) as pool:
        for rep in range(a.reps):
            for low in (True, False):
                name = f"{'low' if low else 'high'}{rep}"
                logpost, ref_sample, ref_logpdf = branch_target(coords, k, a.threshold, low, pool)
                rng = np.random.default_rng([a.seed, rep, int(low)])
                t0 = time.perf_counter()
                res = tempered_smc(
                    logpost, ref_sample, ref_logpdf, rng, chains=a.chains, steps=a.steps,
                    ess_fraction=a.ess_fraction, imh_components=2, target_ess=0.0,
                    max_rhat=float("inf"), tail_fraction=0.0, min_final_steps=a.final_steps,
                    max_final_steps=a.final_steps, check_every=a.final_steps, final_burn=0.0,
                    max_evals=400_000, monitor=coords.theta, name=name,
                )  # fmt: skip
                info = res.info
                phys = coords.physical(res.draws)
                pops[name] = phys
                results.append({
                    "branch": "low" if low else "high", "rep": rep,
                    "log_evidence": info["log_evidence"], "stages": info["stages"],
                    "evals": info["evals"], "seconds": time.perf_counter() - t0,
                    "reached": info["reached_posterior"],
                    "max_log_post": float(np.max(res.log_post)),
                    "mean_log_post": float(np.mean(res.log_post)),
                    "B.to_A_mean": float(phys[:, lik.space.names.index("B.to_A")].mean()),
                })  # fmt: skip
                logging.info("%s: %s", name, results[-1])
    lz = {br: np.array([r["log_evidence"] for r in results if r["branch"] == br])
          for br in ("low", "high")}  # fmt: skip
    # mass of the low branch per replicate pair, and from the pooled evidence estimates
    pair = 1 / (1 + np.exp(lz["high"] - lz["low"]))
    pooled = {br: logsumexp(v) - np.log(len(v)) for br, v in lz.items()}
    mass_low = float(1 / (1 + np.exp(pooled["high"] - pooled["low"])))
    summary = {
        "cell": a.cell, "threshold": a.threshold, "spec": a.spec,
        "settings": vars(a) | {"out": None},
        "log_evidence_low": lz["low"].tolist(), "log_evidence_high": lz["high"].tolist(),
        "mass_low_per_rep": pair.tolist(), "mass_low_pooled": mass_low,
        "log_evidence_total": float(np.logaddexp(pooled["low"], pooled["high"])),
        "runs": results,
    }  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    np.savez_compressed(out / "populations.npz", names=np.array(lik.space.names), **pops)
    logging.info("log Z low %s high %s -> mass(B.to_A < %.2f) per rep %s, pooled %.3f",
                 np.round(lz["low"], 2), np.round(lz["high"], 2), a.threshold,
                 np.round(pair, 3), mass_low)  # fmt: skip


if __name__ == "__main__":
    main()
