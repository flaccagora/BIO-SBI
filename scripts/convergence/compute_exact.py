"""Exact per-clone A/B composition of detected A-founded clones, generations 1..15.

For every parameter set (the 9 XG2 truths, the slow/fast sets of ``sets.py`` and a
subsample of the plausible flat-prior draws) and h = 1..HMAX: the exact law of one clone
conditioned on detection (pgf/FFT, :func:`common.clone_law`) and its composition summaries
(:func:`common.composition_stats`): pooled A fraction, mean / sd / 10-90% range of the
per-clone A fraction, P(all A), P(all B), expected cells. Stops at the first h whose grid
exceeds ``--max-grid``. One JSON per set in ``OUT/exact/<label>.json`` (restart-safe).

  python scripts/convergence/compute_exact.py [--prior N] [--workers W] [--max-grid L]
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

from common import (
    OUT,
    TRUTHS,
    clone_law,
    composition_stats,
    mean_matrix,
    n_mean,
    prior_draws,
    save_json,  # fmt: skip
    truth_xg2,
)
from sets import SETS

HMAX = 15
EXACT = OUT / "exact"


def run(task) -> str:
    label, p12, max_grid = task
    f = EXACT / f"{label}.json"
    if f.exists():
        return f"{label}: cached"
    t0 = time.time()
    rows = []
    for h in range(1, HMAX + 1):
        p, diag = clone_law(p12, h, max_grid=max_grid)
        if p is None:
            break
        s = composition_stats(p)
        s.update(diag, h=h)
        rows.append(s)
        del p
    save_json({"label": label, "p12": list(map(float, p12)), "rows": rows}, f)
    return f"{label}: h<= {rows[-1]['h'] if rows else 0} grid {rows[-1]['grid'] if rows else 0} " \
           f"{time.time() - t0:.0f} s"  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", type=int, default=200, help="prior draws (first N of 2000)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-grid", type=int, default=5000)
    ap.add_argument("--max-grid-prior", type=int, default=2500)
    ap.add_argument("--only", default="all", choices=["all", "truths", "prior"])
    a = ap.parse_args()
    EXACT.mkdir(parents=True, exist_ok=True)
    tasks = []
    if a.only in ("all", "truths"):
        tasks += [(gt, truth_xg2(gt), a.max_grid) for gt in TRUTHS]
        tasks += [(k, v, a.max_grid) for k, v in SETS.items()]
    if a.only in ("all", "prior"):
        draws, _ = prior_draws(2000)
        # every 10th draw: a subsample spread over the stream
        idx = list(range(0, 2000, max(2000 // a.prior, 1)))[: a.prior]
        # plus every slow-mixing draw (pooled composition within 0.01 only from h >= 9)
        idx += [i for i, p in enumerate(draws)
                if i not in idx and n_mean(mean_matrix(p), 0.01) >= 9]  # fmt: skip
        tasks += [(f"prior{i:04d}", draws[i], a.max_grid_prior) for i in idx]
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run, tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
