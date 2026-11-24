"""Prior-whitened Fisher information of the slow/fast-mixing sets (A-founded, gens 1..12).

Same quantity and coordinates as the optimal-design study
(:func:`scripts/optimal_design/od_common.information_basis`): expected information of one
detected A-founded clone at generation h, in orthonormal row contrasts (10 dims), divided by
the flat-prior precision 42. Cached in ``OUT/info/<label>.npz`` (``K (H, 10, 10)``, ``q``,
``gens``, ``p12``, ``U``). The XG2 truths and the first 60 plausible prior draws are already
in ``results/exact_gw/optimal_design/info`` (same prior stream).

  python scripts/convergence/compute_info.py [--workers 2]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from multiprocessing import Pool

import numpy as np
from common import OUT, REPO
from sets import SETS

GENS = tuple(range(1, 13))
INFO = OUT / "info"


def od():
    spec = importlib.util.spec_from_file_location(
        "od_common", REPO / "scripts" / "optimal_design" / "od_common.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["od_common"] = mod
    spec.loader.exec_module(mod)
    return mod


MIX_T = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def family() -> dict[str, np.ndarray]:
    """S1 -> F1 path: A row fixed (shared), B row (1 - t) S1_B + t F1_B (matched growth)."""
    out = {}
    for t in MIX_T:
        p = SETS["S1"].copy()
        p[6:] = (1 - t) * SETS["S1"][6:] + t * SETS["F1"][6:]
        out[f"mix{t:.2f}"] = p
    return out


def run(task) -> str:
    label, p12 = task
    f = INFO / f"{label}.npz"
    if f.exists():
        return f"{label}: cached"
    m = od()
    U = m.tangent_basis()
    K = np.zeros((len(GENS), U.shape[1], U.shape[1]))
    q = np.zeros(len(GENS))
    for j, g in enumerate(GENS):
        info, q[j] = m.information_basis(p12, g, "A", U)
        K[j] = info / m.PRIOR_PREC
    np.savez_compressed(f, K=K, q=q, gens=np.array(GENS), p12=p12, U=U)
    return f"{label}: done"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    INFO.mkdir(parents=True, exist_ok=True)
    tasks = list(SETS.items()) + list(family().items())
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run, tasks):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
