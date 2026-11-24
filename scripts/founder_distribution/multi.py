"""Several seeding conditions, joint MLE: shared transitions, separate unknown founder rows.

Condition c has its own founder distribution (p_A and sizes, estimated) and n/C detected
clones per endpoint 4/6/8/10; the transitions (10 stick coordinates) are shared. The joint
log-likelihood is the sum of the per-condition exact likelihoods (``ExactLikelihood`` with
free founder rows), theta = (shared row sticks, founder sticks of condition 1, 2, ...).
Compared with one condition of the same total size (n = 512 per endpoint) at the mean p_A.

``ExactGW`` (NUTS) takes one ``ExactLikelihood`` (one founder distribution for all
cohorts), so the joint posterior is not sampled here; see the README for how to add it.

    uv run python scripts/founder_distribution/multi.py --truths T1_ref,P3 --seeds 0-5 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R  # noqa: E402

C = R.C
SETUPS = {
    "S2 0.9+0.3": [("S2_pA0.9", 256), ("S2_pA0.3", 256)],
    "S2 0.6 single": [("S2_pA0.6", 512)],
}
R.SIZE_DISTS.setdefault("S2", [0.7, 0.2, 0.07, 0.02, 0.01])


def dataset_n(dist: str, gt: str, seed: int, n: int, cond: int):
    from bio_sbi.summaries import Cohort

    pi, pa = R.parse_dist(dist)
    P = C.TRUTHS[gt]
    rng = np.random.default_rng([C.GTS.index(gt), seed, 2000 + cond, int(round(100 * pa))])
    cohorts = []
    for h in C.HORIZONS:
        kept, have, batch = [], 0, int(1.5 * n) + 64
        while have < n:
            tr, _ = R._simulate_true(P, h, batch, rng, pa, pi)
            det = tr.sum(1) > 0
            kept.append(tr[det])
            have += int(det.sum())
            batch = int(min(max((n - have) / (max(have, 1) / (batch * len(kept))) * 1.3, 64),
                            50 * n))  # fmt: skip
        cohorts.append(Cohort(float(h), np.concatenate(kept)[:n]))
    return cohorts


class Joint:
    def __init__(self, liks):
        self.liks = liks
        self.nr = liks[0].n_row_dims
        self.nf = [lk.dim - self.nr for lk in liks]
        self.dim = self.nr + sum(self.nf)

    def split(self, th):
        out, off = [], self.nr
        for k in self.nf:
            out.append(np.concatenate([th[: self.nr], th[off : off + k]]))
            off += k
        return out

    def log_likelihood(self, th, grad=True):
        tot, g = 0.0, np.zeros(self.dim)
        off = self.nr
        for lk, t, k in zip(self.liks, self.split(th), self.nf, strict=True):
            ll, gc = lk.log_likelihood(t, grad=grad)
            tot += ll
            if grad:
                g[: self.nr] += gc[: self.nr]
                g[off : off + k] = gc[self.nr :]
            off += k
        return tot, g


def job(gt: str, setup: str, seed: int, out_dir: Path):
    out = out_dir / "multi" / setup.replace(" ", "_") / f"{gt}_s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(out.with_suffix(".lock"), os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return gt, setup, seed, 0.0, "locked"
    t0 = time.perf_counter()
    try:
        from scipy.optimize import minimize

        conds = SETUPS[setup]
        liks, sims = [], []
        for c, (dist, n) in enumerate(conds):
            sim, lik = R.make_lik(dist, "est", dataset_n(dist, gt, seed, n, c))
            liks.append(lik)
            sims.append(sim)
        J = Joint(liks)
        rng = np.random.default_rng([seed, 9, C.GTS.index(gt)])
        tt = [R.truth_theta(s, d, gt) for s, (d, _) in zip(sims, conds, strict=True)]
        t_true = np.concatenate([tt[0][: J.nr]] + [t[J.nr :] for t in tt])
        starts = [t_true] + [np.clip(rng.uniform(size=J.dim), 1e-3, 1 - 1e-3) for _ in range(4)]

        def neg(u):
            th = 1 / (1 + np.exp(-u))
            ll, g = J.log_likelihood(th)
            if not np.isfinite(ll):
                return 1e300, np.zeros_like(u)
            return -ll, -g * th * (1 - th)

        fits = []
        for th0 in starts:
            u0 = np.clip(np.log(th0) - np.log1p(-th0), -12, 12)
            r = minimize(neg, u0, jac=True, method="L-BFGS-B", bounds=[(-14, 14)] * J.dim,
                         options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-7})  # fmt: skip
            fits.append((float(-r.fun), r.x))
        fits.sort(key=lambda t: -t[0])
        th = 1 / (1 + np.exp(-fits[0][1]))
        res = {"gt": gt, "setup": setup, "seed": seed, "ll": fits[0][0],
               "ll_truth": float(J.log_likelihood(t_true, grad=False)[0]),
               "lls": [f[0] for f in fits], "conditions": []}  # fmt: skip
        truth = R.truth_dict(conds[0][0], gt)
        for sim, t, (dist, _) in zip(sims, J.split(th), conds, strict=True):
            cols = R.columns(sim, sim.space.decode(t)[None])
            res["conditions"].append({"dist": dist, **{k: float(v[0]) for k, v in cols.items()}})
        c0 = res["conditions"][0]
        res["err"] = {k: c0[k] - truth[k] for k in ("B.to_A", "m_BA", "rho", "stableA")}
        res["err_founder"] = [c["founder.A"] - R.parse_dist(c["dist"])[1] for c in res["conditions"]]
        res["seconds"] = time.perf_counter() - t0
        out.write_text(json.dumps(res, indent=1))
        return gt, setup, seed, res["seconds"], None
    except Exception as exc:  # noqa: BLE001
        out.with_suffix(".err").write_text(traceback.format_exc())
        return gt, setup, seed, 0.0, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truths", default="T1_ref,P3")
    ap.add_argument("--seeds", default="0-5")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(R.RESULTS))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.seeds.split("-"))
    todo = [(gt, s, sd) for sd in range(lo, hi + 1) for gt in a.truths.split(",") for s in SETUPS]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(job, *t, Path(a.out)) for t in todo]
        for k, f in enumerate(as_completed(futs), 1):
            gt, s, sd, sec, err = f.result()
            print(f"[{k}/{len(todo)}] {time.time() - t0:.0f}s {s}/{gt}/s{sd} "
                  f"{'FAILED ' + err if err else f'{sec:.0f}s'}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
