"""Why is Lambda = 0 more often than the chi-bar weight w0 says? Score diagnostics.

For null datasets of the null study, recompute (data regenerated from the seed) the
observed information at theta_0 and the derivatives of the log likelihood along the three
dedifferentiation routes (scores). Asymptotically the efficient scores are N(0, J) and
Lambda = 0 iff all three are <= 0 (probability w0). Writes per-dataset standardised
scores to results/exact_gw/dediff_test/summary/scores.csv and prints their mean, sd,
skewness and P(all <= 0) per cell and n.

  uv run python scripts/dediff_test/scores.py [--seeds 0:100] [--workers 16]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cells as C  # noqa: E402

from bio_sbi.inference import dediff_lrt as D  # noqa: E402

ROOT = Path("/home/matteo/BIO-SBI/results/exact_gw/dediff_test")


def one(cell: str, seed: int) -> list[dict]:
    f = ROOT / "null" / cell / f"s{seed}.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    sim = D.make_simulator()
    c = C.cell(cell)
    data = D.simulate_counts(sim, c["P"], C.HORIZONS, max(C.NS), seed, ("reference",))
    rows = []
    for r in d["results"]["perfect"]:
        lik = D.likelihood(sim, D.subset(data, r["n"]))
        P0 = np.array(r["P0"])
        loc = D.local_information(lik, P0)
        T, F = loc.dediff, [k for k in range(len(loc.info)) if k not in loc.dediff]
        I = loc.info  # noqa: E741
        J = I[np.ix_(T, T)] - I[np.ix_(T, F)] @ np.linalg.pinv(I[np.ix_(F, F)]) @ I[np.ix_(F, T)]
        sc = D.dediff_scores(lik, P0)
        # local coordinates move mass from the reference entry; scores move it from all
        # H0 entries in proportion: same first-order direction up to the interior nuisance,
        # whose score is 0 at theta_0 (when interior) -> comparable to J
        s = np.array([sc[e] for e in D.DEDIFF])
        z = s / np.sqrt(np.diag(J))
        rows.append({"cell": cell, "seed": seed, "n": r["n"], "lam": r["lambda"],
                     "z_toA": z[0], "z_divAA": z[1], "z_divAB": z[2],
                     "all_nonpos": bool(np.all(s <= 0)), "w0": r["chibar"]["weights"][0],
                     "bnd": ",".join(r["chibar"]["boundary_nuisance"])})  # fmt: skip
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0:100")
    ap.add_argument("--cells", default="N1,N2,N3,N4")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split(":"))
    jobs = [(c, s) for c in a.cells.split(",") for s in range(lo, hi)]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = [r for rs in pool.map(one, *zip(*jobs, strict=True)) for r in rs]
    df = pd.DataFrame(rows)
    (ROOT / "summary").mkdir(exist_ok=True)
    df.to_csv(ROOT / "summary" / "scores.csv", index=False)
    out = []
    for (cell, n), g in df.groupby(["cell", "n"]):
        row = {"cell": cell, "n": n, "datasets": len(g), "P(Lambda=0)": (g.lam < 1e-6).mean(),
               "P(all scores<=0)": g.all_nonpos.mean(), "w0 (chi-bar)": g.w0.mean()}
        for e in ("toA", "divAA", "divAB"):
            z = g[f"z_{e}"]
            row[f"{e} mean"] = z.mean()
            row[f"{e} sd"] = z.std()
            row[f"{e} skew"] = stats.skew(z)
        out.append(row)
    t = pd.DataFrame(out)
    t.to_csv(ROOT / "summary" / "scores_summary.csv", index=False)
    print(t.round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
