"""All-12-parameter precision of the practical protocols (report section 9).

Total asymptotic variance of the 12 transition probabilities, tr(K^-1) for the whitened
information K of a design (the contrast basis is orthonormal, so the trace is the summed
variance up to the common factor 1/(42 n)); reported as the gain current / protocol per truth
(T1/T2/T4 at eps = 0.01 as in the study). Founders recorded.

  uv run python scripts/optimal_design/all12.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from od_common import POINTS, SHORTNAME, TAB, TRUTHS, compute_info, design_truth  # noqa: E402
from optimise import PROTOCOLS  # noqa: E402


def main() -> None:
    rows = []
    for gt in TRUTHS:
        d = compute_info(gt, design_truth(gt))
        gens = list(d["gens"])
        Ks = np.array([d["K"][0 if f == "A" else 1, gens.index(h)] for f, h in POINTS])
        tr = {}
        for name, w in PROTOCOLS.items():
            M = np.tensordot(w, Ks, axes=1)
            tr[name] = float(np.trace(np.linalg.inv(M)))
        ref = tr["current: A 4/6/8/10"]
        for name, t in tr.items():
            rows.append({"truth": SHORTNAME.get(gt, gt), "protocol": name, "trace": t,
                         "gain_all12": ref / t})  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "protocols_all12.csv", index=False)
    s = df.groupby("protocol", sort=False).gain_all12.agg(["median", "min", "max"])
    print(s.round(2).to_string())


if __name__ == "__main__":
    main()
