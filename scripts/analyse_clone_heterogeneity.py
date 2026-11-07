"""Sanity check of the clone-heterogeneity (random-effects) law against CRC0322.

Simulates the real design (day 6 / day 10 = 144 / 240 h, 165 / 101 detected clones)
with Bellman-Harris at plausible parameters (posterior means of the homogeneous
``bh_rev`` and ``bh_two`` fits to the 2D and 3D-mean counts,
results/real/<model>/fit_*, copied below) and
``kappa in {3, 10, 100, inf}``, and reports per horizon

* ``corr_A_B``     - correlation of log1p(A), log1p(B) across clones (CohortMoments),
* ``total.sd_log`` - sd of log1p(A + B) across clones (CohortMoments),
* ``fracA.sd``     - sd of the A fraction A / (A + B) across clones,

as mean (sd) over replicate datasets ("invalid": every dataset exceeded max_cells),
for heterogeneity of whole outcome rows and of the division split only
(``clone_heterogeneity.outcomes: [div_AA, div_AB, div_BB]``),, next to the same statistics of the real data
(data/real/crc0322_{2d,3d_mean}_bh.csv). Usage (single thread, ~1 min)::

    NUMBA_NUM_THREADS=1 uv run python scripts/analyse_clone_heterogeneity.py [--reps 20]
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
REVERSIBLE = {"founder": {"A": 1.0}, "B": {"div_AA": 0.0, "div_AB": 0.0, "div_BB": 0.0}}
#: (model, data variant) -> (fixed law values, posterior means of the homogeneous fit
#: results/real/<model>/fit_<variant>, used as the population-level parameters).
PARAMETERS = {
    ("bh_rev", "2d"): (REVERSIBLE, {
        "A": {"death": 0.0186, "persist": 0.0063, "to_B": 0.0023, "div_AA": 0.7521,
              "div_AB": 0.0149, "div_BB": 0.2058},
        "B": {"death": 0.8893, "to_A": 0.0419, "persist": 0.0688},
        "A.mean_lifetime": 33.0, "B.mean_lifetime": 54.7,
    }),
    ("bh_rev", "3d_mean"): (REVERSIBLE, {
        "A": {"death": 0.0944, "persist": 0.0105, "to_B": 0.0041, "div_AA": 0.6205,
              "div_AB": 0.0261, "div_BB": 0.2444},
        "B": {"death": 0.9551, "to_A": 0.0144, "persist": 0.0305},
        "A.mean_lifetime": 14.2, "B.mean_lifetime": 17.9,
    }),
    ("bh_two", "2d"): ({"founder": {"A": 1.0}}, {
        "A": {"death": 0.2297, "persist": 0.0054, "to_B": 0.0237, "div_AA": 0.6683,
              "div_AB": 0.0035, "div_BB": 0.0694},
        "B": {"death": 0.1433, "to_A": 0.0187, "persist": 0.0623, "div_AA": 0.1598,
              "div_AB": 0.0140, "div_BB": 0.6019},
        "A.mean_lifetime": 23.6, "B.mean_lifetime": 41.5,
    }),
    ("bh_two", "3d_mean"): ({"founder": {"A": 1.0}}, {
        "A": {"death": 0.3691, "persist": 0.0017, "to_B": 0.0123, "div_AA": 0.6131,
              "div_AB": 0.0000, "div_BB": 0.0038},
        "B": {"death": 0.0652, "to_A": 0.0012, "persist": 0.0233, "div_AA": 0.1596,
              "div_AB": 0.0058, "div_BB": 0.7449},
        "A.mean_lifetime": 12.0, "B.mean_lifetime": 18.9,
    }),
}  # fmt: skip
KAPPAS = (3.0, 10.0, 100.0, np.inf)
#: Which outcomes vary between clones: the whole row, or only the division split.
MODES = {"whole rows": None, "divisions only": ["div_AA", "div_AB", "div_BB"]}
SUMMARY = CohortMoments()


def statistics(cohorts: list[Cohort]) -> dict[str, float]:
    """corr_A_B, total.sd_log, fracA.sd and mean total per cohort."""
    names = SUMMARY.feature_names([c.horizon for c in cohorts], ["A", "B"])
    x = dict(zip(names, SUMMARY(cohorts), strict=True))
    out = {}
    for c in cohorts:
        h = f"h{c.horizon:g}"
        total = c.counts.sum(1)
        out[f"{h}.corr_A_B"] = x[f"{h}.corr_A_B"]
        out[f"{h}.total.sd_log"] = x[f"{h}.total.sd_log"]
        out[f"{h}.fracA.sd"] = float(np.std(c.counts[:, 0] / total))
        out[f"{h}.total.mean"] = float(total.mean())
    return out


def real(variant: str) -> dict[str, float]:
    """Statistics of the measured CRC0322 counts (``_bh`` file of ``variant``)."""
    table = pd.read_csv(ROOT / "data" / "real" / f"crc0322_{variant}_bh.csv")
    cohorts = [
        Cohort(h, table.loc[np.isclose(table.horizon, h), ["A", "B"]].to_numpy(np.int64))
        for h in HORIZONS
    ]
    return statistics(cohorts)


def simulated(key, kappa: float, reps: int, seed: int, outcomes=None) -> dict[str, str]:
    """Mean (sd) of the statistics over ``reps`` simulated real-design datasets."""
    fixed, values = PARAMETERS[key]
    hetero = None if np.isinf(kappa) else {"kappa_bounds": [0.1, 1e9], "outcomes": outcomes}
    law = {"types": ["A", "B"], "fixed": fixed, "clone_heterogeneity": hetero}
    sim = BellmanHarris(law, lifetime_shape=8.0, lifetime_mean_bounds=[5.0, 100.0],
                        max_cells=50_000)  # fmt: skip
    if hetero is not None:
        values = {**values, "clone.kappa": kappa}
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
    """Print one table per (model, data variant, heterogeneity mode)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for model, variant in PARAMETERS:
        for label, outcomes in MODES.items():
            table = {"real": {k: f"{v:.2f}" for k, v in real(variant).items()}}
            for kappa in KAPPAS:
                table[f"kappa={kappa:g}"] = simulated(
                    (model, variant), kappa, args.reps, args.seed, outcomes
                )
            frame = pd.DataFrame(table).astype(object).fillna("invalid")
            print(f"\n### {model} @ {variant}, {label} (population parameters: "
                  f"{model} posterior mean, fit_{variant})\n")  # fmt: skip
            print(frame.to_markdown())


if __name__ == "__main__":
    main()
