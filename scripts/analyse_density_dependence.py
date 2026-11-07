"""Sanity check of density-dependent Bellman-Harris lifetimes against CRC0322.

Simulates the real design (day 6 / day 10 = 144 / 240 h, 165 / 101 detected clones,
266 in total) with Bellman-Harris (``law=two_type``, lifetime shape 8) at parameters
near the founder-mixture posterior of the real-data fits (founder B 0.7 / A 0.3; A:
div_AA 0.75, death 0.1, persist 0.1, to_B 0.02; B: div_BB 0.7, death 0.15, persist
0.05, div_AA 0.05; remaining entries small; tau_A 20 h, tau_B 18 h) and carrying
capacities ``K in {30, 100, 300, 1000, inf}`` (``density.capacity``; ``inf`` is the
constant-lifetime model), and reports per horizon

* ``total.mean_log`` - mean of log1p(A + B) across clones (CohortMoments),
* ``total.sd_log``   - sd of log1p(A + B) across clones (CohortMoments),
* ``corr_A_B``       - correlation of log1p(A), log1p(B) across clones (CohortMoments),
* ``fracA.sd``       - sd of the A fraction A / (A + B) across clones,
* ``ratio240/144``   - mean total cells at 240 h / mean total cells at 144 h,

as mean (sd) over replicate datasets, next to the same statistics of the real data
(data/real/crc0322_{3d_mean,2d}_bh.csv). Usage (single thread, ~1 min)::

    NUMBA_NUM_THREADS=1 uv run python scripts/analyse_density_dependence.py [--reps 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bio_sbi.design import Design
from bio_sbi.observation import ObservationModel
from bio_sbi.simulate import DataGenerator
from bio_sbi.simulators import BellmanHarris
from bio_sbi.summaries import Cohort, CohortMoments

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (144.0, 240.0)
CLONES = (165, 101)
SMALL = 0.01  # outcomes the description calls "small"
#: Near the founder-mixture posterior of the real-data fits (rows sum to one).
PARAMETERS = {
    "founder": {"A": 0.3, "B": 0.7},
    "A": {"div_AA": 0.75, "death": 0.1, "persist": 0.1, "to_B": 0.02,
          "div_AB": SMALL, "div_BB": 1 - 0.75 - 0.1 - 0.1 - 0.02 - SMALL},
    "B": {"div_BB": 0.7, "death": 0.15, "persist": 0.05, "div_AA": 0.05,
          "to_A": SMALL, "div_AB": 1 - 0.7 - 0.15 - 0.05 - 0.05 - SMALL},
    "A.mean_lifetime": 20.0, "B.mean_lifetime": 18.0,
}  # fmt: skip
CAPACITIES = (30.0, 100.0, 300.0, 1000.0, np.inf)
SUMMARY = CohortMoments()


def statistics(cohorts: list[Cohort]) -> dict[str, float]:
    """Per-horizon statistics plus the ratio of mean totals 240 h / 144 h."""
    names = SUMMARY.feature_names([c.horizon for c in cohorts], ["A", "B"])
    x = dict(zip(names, SUMMARY(cohorts), strict=True))
    out = {}
    for c in cohorts:
        h = f"h{c.horizon:g}"
        total = c.counts.sum(1)
        out[f"{h}.total.mean_log"] = x[f"{h}.total.mean_log"]
        out[f"{h}.total.sd_log"] = x[f"{h}.total.sd_log"]
        out[f"{h}.corr_A_B"] = x[f"{h}.corr_A_B"]
        out[f"{h}.fracA.sd"] = float(np.std(c.counts[:, 0] / total))
        out[f"{h}.total.mean"] = float(total.mean())
    out["ratio240/144"] = out["h240.total.mean"] / out["h144.total.mean"]
    return out


def real(variant: str) -> dict[str, float]:
    """Statistics of the measured CRC0322 counts (``_bh`` file of ``variant``)."""
    table = pd.read_csv(ROOT / "data" / "real" / f"crc0322_{variant}_bh.csv")
    cohorts = [
        Cohort(h, table.loc[np.isclose(table.horizon, h), ["A", "B"]].to_numpy(np.int64))
        for h in HORIZONS
    ]
    return statistics(cohorts)


def simulated(capacity: float, reps: int, seed: int) -> dict[str, str]:
    """Mean (sd) of the statistics over ``reps`` simulated real-design datasets."""
    dd = None if np.isinf(capacity) else {"capacity_bounds": [1.0, 1e9]}
    law = {"types": ["A", "B"]}  # two_type with the founder row free
    sim = BellmanHarris(law, lifetime_shape=8.0, lifetime_mean_bounds=[5.0, 100.0],
                        max_cells=50_000, density_dependence=dd)  # fmt: skip
    values = PARAMETERS if dd is None else {**PARAMETERS, "density.capacity": capacity}
    gen = DataGenerator(
        sim, ObservationModel(types=sim.types), Design(HORIZONS, CLONES), SUMMARY, seed=seed
    )
    p = sim.physical_from_mapping(values)
    rows = [statistics(d) for d in (gen.dataset(p, ("sanity", r)) for r in range(reps)) if d]
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    return {k: f"{frame[k].mean():.2f} ({frame[k].std():.2f})" for k in frame}


def main() -> None:
    """Print one table: real data (3D mean, 2D) and simulations per capacity."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    table = {f"real {v}": {k: f"{x:.2f}" for k, x in real(v).items()}
             for v in ("3d_mean", "2d")}  # fmt: skip
    for capacity in CAPACITIES:
        table[f"K={capacity:g}"] = simulated(capacity, args.reps, args.seed)
    frame = pd.DataFrame(table).astype(object).fillna("invalid")
    print(f"\n### Density dependence at the real design ({args.reps} datasets per K)\n")
    print(frame.to_markdown())


if __name__ == "__main__":
    main()
