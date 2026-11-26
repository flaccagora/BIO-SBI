"""Compute and cache the per-generation Fisher information of every truth (and variants).

For each XG2 truth (and, for the boundary truths T1/T2/T4, slightly interior variants with
each zero B division set to eps, the rest of the B row rescaled), the information of one
detected clone at generations 1..10, for an A-founded and a B-founded clone, is written to
``results/exact_gw/weak_directions/info_<label>.npz`` (12 x 12, tangent space).

  uv run python scripts/weak_directions/compute_info.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from common import BOUNDARY, TRUTHS, info_cache, truth  # noqa: E402

EPS = (0.01, 0.001)


def variants() -> dict[str, np.ndarray]:
    out = {gt: truth(gt) for gt in TRUTHS}
    for gt in BOUNDARY:
        p = truth(gt)
        zero = p == 0
        for e in EPS:
            q = p.copy()
            for r in (0, 1):
                sl = slice(6 * r, 6 * r + 6)
                z = zero[sl]
                if z.any():
                    q[sl][~z] *= 1 - e * z.sum()
                    q[sl][z] = e
            out[f"{gt}_eps{e:g}"] = q
    return out


def main() -> None:
    for lab, p in variants().items():
        t0 = time.time()
        info_cache(lab, p)
        print(f"{lab}: {time.time() - t0:.1f} s  p = {np.round(p, 4).tolist()}", flush=True)


if __name__ == "__main__":
    main()
