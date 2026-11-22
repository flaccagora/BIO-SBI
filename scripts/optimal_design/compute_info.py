"""Per-design-point Fisher information for the optimal-design study (cached).

* the 9 XG2 truths (eps = 0.01 variant for T1/T2/T4), generations 1..12, A and B founders;
* T1/T2/T4 at the exact boundary on the active face (B never divides known, 7 dims);
* the exact boundary truths without restriction, A founders only (check: eps vs exact);
* draws from the flat prior with 30-300 expected cells per A-founded clone at generation
  10 (the XG2 ``truth.mean_cells`` filter for P1-P4), generations 1..10 (memory: their
  generation-12 grids reach ~4000^2 x 10 directions).

  uv run python scripts/optimal_design/compute_info.py [n_prior_draws]  (default 40; ~4 numba threads)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from od_common import (  # noqa: E402
    BOUNDARY, OUT, TRUTHS, compute_info, design_truth, expected_cells, truth_xg2,
)  # fmt: skip

PRIOR_SEED = 20261122
MEAN_CELLS = (30.0, 300.0)  # at generation 10, A founder (pipeline truth.mean_cells)


def prior_draws(n: int) -> tuple[np.ndarray, int]:
    """``n`` flat-prior draws passing the XG2 plausibility filter, and the number drawn."""
    rng = np.random.default_rng(PRIOR_SEED)
    keep, tried = [], 0
    while len(keep) < n:
        p = rng.dirichlet(np.ones(6), size=2).ravel()
        tried += 1
        m = expected_cells(p, 10, "A")
        if MEAN_CELLS[0] <= m <= MEAN_CELLS[1]:
            keep.append(p)
    return np.array(keep), tried


def main() -> None:
    n_prior = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    for gt in TRUTHS:
        t0 = time.time()
        d = compute_info(gt, design_truth(gt))
        print(f"{gt}: {time.time() - t0:.1f} s  q_A1={d['q'][0, 0]:.3f} q_B1={d['q'][1, 0]:.3f}",
              flush=True)  # fmt: skip
    for gt in BOUNDARY:
        p = truth_xg2(gt)
        t0 = time.time()
        compute_info(f"{gt}_face", p, active=p > 0)
        compute_info(f"{gt}_exact", p)  # A-founded part is regular; B part floor-dependent
        print(f"{gt} face/exact: {time.time() - t0:.1f} s", flush=True)
    draws, tried = prior_draws(n_prior)
    np.savez(OUT / "prior_draws.npz", p12=draws, tried=tried, seed=PRIOR_SEED)
    print(f"prior: {n_prior} accepted of {tried} drawn", flush=True)
    for i, p in enumerate(draws):
        t0 = time.time()
        compute_info(f"prior{i:03d}", p, gens=tuple(range(1, 11)))
        print(f"prior{i:03d}: {time.time() - t0:.1f} s  mean10={expected_cells(p, 10):.0f}",
              flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
