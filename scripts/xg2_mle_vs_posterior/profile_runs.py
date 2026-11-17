"""Per run: exact log likelihood of every posterior draw, profile likelihood of B.to_A,
and the best fit of the sub-model "B never divides" (B.div_* = 0).

Usage (one thread, sequential):
  uv run python scripts/xg2_mle_vs_posterior/profile_runs.py T1_ref:0:128 P1:0:128 ...
Writes docs/xg2_mle_vs_posterior/profiles/<gt>_s<seed>_n<n>.npz (skips existing).

Profile: at each grid value c of B.to_A the other 9 free probabilities are maximised
(A row: 5 sticks; B row: the 5 other outcomes share 1 - c, 4 sticks), L-BFGS-B with the
exact gradient. Starts: warm start from the neighbouring grid point, sweeping outward
from the grid point nearest the MLE (started at the MLE), plus the 2 highest-likelihood
posterior draws with |B.to_A - c| < 0.05, projected onto B.to_A = c.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from common import OUT, SubModel, all_runs, setup

GRID = np.round(np.r_[0.01, 0.03, 0.05, np.arange(0.1, 0.951, 0.05)], 3)
DIV = ("B.div_AA", "B.div_AB", "B.div_BB")


def project(phys: np.ndarray, names: list[str], c: float) -> np.ndarray:
    p = phys.copy()
    b = [i for i, n in enumerate(names) if n.startswith("B.") and n != "B.to_A"]
    p[b] = np.clip(p[b], 1e-9, None)
    p[b] *= (1 - c) / p[b].sum()
    p[names.index("B.to_A")] = c
    return p


def do_run(key: tuple[str, int, int]) -> None:
    gt, seed, n = key
    out = OUT / "profiles" / f"{gt}_s{seed}_n{n}.npz"
    if out.exists():
        print("skip", out, flush=True)
        return
    t0 = time.perf_counter()
    run = all_runs()[key]
    cfg, sim, lik, names, truth, post = setup(run)
    mle = json.loads((run / "mle.json").read_text())
    full = SubModel(lik, {})

    # 1. log likelihood of every posterior draw (objective only)
    draws_ll = np.array([full.loglik(full.encode(p), grad=False)[0] for p in post])
    print(f"{key}: draws ll done {time.perf_counter() - t0:.0f}s, max {draws_ll.max():.2f} "
          f"vs mle {mle['ll_mle']:.2f}", flush=True)  # fmt: skip

    mle_phys = post.mean(0).copy()
    for nm, v in mle["mle"].items():
        mle_phys[names.index(nm)] = v
    # refit the MLE itself in this parametrisation (start: stored MLE, best draw)
    best = max((full.maximise(full.encode(s)) for s in (mle_phys, post[draws_ll.argmax()])),
               key=lambda r: r["ll"])  # fmt: skip
    mle_phys = full.physical(best["s"])
    ll_mle = best["ll"]

    # 2. profile of B.to_A
    ia = names.index("B.to_A")
    prof_ll = np.full(len(GRID), -np.inf)
    prof_phys = np.zeros((len(GRID), len(names)))
    sub = SubModel(lik, {"B.to_A": 0.5})

    def fit(j: int, starts: list[np.ndarray]) -> None:
        sub.set_pinned({"B.to_A": float(GRID[j])})
        for st in starts:
            r = sub.maximise(sub.encode(project(st, names, GRID[j])))
            if r["ll"] > prof_ll[j]:
                prof_ll[j], prof_phys[j] = r["ll"], sub.physical(r["s"])

    j0 = int(np.argmin(np.abs(GRID - mle_phys[ia])))
    order = [j0] + list(range(j0 + 1, len(GRID))) + list(range(j0 - 1, -1, -1))
    for j in order:
        starts = [mle_phys] if j == j0 else [prof_phys[j - 1 if j > j0 else j + 1]]
        near = np.flatnonzero(np.abs(post[:, ia] - GRID[j]) < 0.05)
        starts += [post[i] for i in near[np.argsort(draws_ll[near])[::-1][:2]]]
        fit(j, starts)
    # second outward pass from the best grid point (repairs branches lost in pass 1)
    jb = int(np.argmax(prof_ll))
    for j in list(range(jb + 1, len(GRID))) + list(range(jb - 1, -1, -1)):
        prev = prof_phys[j - 1 if j > jb else j + 1]
        before = prof_ll[j]
        fit(j, [prev])
        if prof_ll[j] > before + 0.01:
            print(f"  pass 2 improved c={GRID[j]}: +{prof_ll[j] - before:.2f}", flush=True)
    print(f"{key}: profile done {time.perf_counter() - t0:.0f}s", flush=True)

    # 3. sub-model "B never divides": best fit, starts = MLE / best draws / truth-free prior
    nodiv = SubModel(lik, dict.fromkeys(DIV, 0.0))
    starts = [mle_phys, post[draws_ll.argmax()], prof_phys[np.argmax(prof_ll)]]
    starts += [post[i] for i in np.argsort(post[:, names.index("B.div_AA")]
               + post[:, names.index("B.div_AB")] + post[:, names.index("B.div_BB")])[:2]]
    rng = np.random.default_rng(0)
    for _ in range(3):
        p = post.mean(0).copy()
        sl = sim.space.row_slice("A")
        p[sl] = rng.dirichlet(np.ones(6))
        p[[names.index("B.death"), ia, names.index("B.persist")]] = rng.dirichlet(np.ones(3))
        starts.append(p)
    res = [nodiv.maximise(nodiv.encode(s)) for s in starts]
    rb = max(res, key=lambda r: r["ll"])
    nodiv_phys = nodiv.physical(rb["s"])
    print(f"{key}: no-division fit ll {rb['ll']:.2f} (full {ll_mle:.2f}), "
          f"B.to_A {nodiv_phys[ia]:.3f}; {time.perf_counter() - t0:.0f}s", flush=True)  # fmt: skip

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(names), truth=truth, post=post, draws_ll=draws_ll,
             mle_phys=mle_phys, ll_mle=ll_mle, ll_mle_stored=mle["ll_mle"],
             ll_truth=mle["ll_truth"], grid=GRID, prof_ll=prof_ll, prof_phys=prof_phys,
             nodiv_ll=rb["ll"], nodiv_phys=nodiv_phys,
             nodiv_all=np.array([r["ll"] for r in res]), seconds=time.perf_counter() - t0,
             evals=lik.evals)  # fmt: skip


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        g, s, nn = arg.split(":")
        do_run((g, int(s), int(nn)))
