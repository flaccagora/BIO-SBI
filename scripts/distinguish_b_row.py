"""Test whether endpoint data distinguish B -> A switching from B -> AA division.

Family of alternatives that keeps the mean offspring matrix equal to the truth:

    B.div_AA = a,  B.to_A = 0.8 - 2a,  B.death = 0.1 + a,  B.persist = 0.1

so expected counts are identical at every horizon for every ``a``; only higher
moments differ. For each ``a`` and clone number, simulate datasets at ``a`` and at the
truth ``a = 0`` and run a classifier two-sample test (C2ST, random forest, 5-fold) on
the summary vectors. Accuracy 0.5 = indistinguishable from one dataset; 1.0 = always
distinguishable.

Usage::

    uv run python scripts/distinguish_b_row.py --out results/b_row/distinguish.csv
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_module
from sbi.utils.metrics import c2st

from bio_sbi.pipeline import build_components, reference_parameters


def physical_at(sim, truth, a):
    """Truth with B's row moved along the mean-preserving family."""
    names = sim.parameter_names
    p = truth.copy()
    p[sim.space.row_slice("B")] = 0.0
    for name, value in {"B.div_AA": a, "B.to_A": 0.8 - 2 * a, "B.death": 0.1 + a,
                        "B.persist": 0.1}.items():  # fmt: skip
        p[names.index(name)] = value
    return sim.validate(p)


def main() -> None:
    """Run the distinguishability grid and print/save the table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--clones", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--datasets", type=int, default=500, help="per parameter setting")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results/b_row/distinguish.csv")
    args = parser.parse_args()

    rows = []
    for clones in args.clones:
        overrides = ["law=two_type", f"design.clones={clones}", f"seed={args.seed}",
                     f"workers={args.workers}"]  # fmt: skip
        with initialize_config_module("bio_sbi.conf", version_base="1.3"):
            cfg = compose("config", overrides=overrides)
        comp = build_components(cfg)
        sim, gen = comp.simulator, comp.generator
        truth = reference_parameters(cfg, sim)

        def summaries(physical, stage, sim=sim, gen=gen):
            theta = np.tile(sim.encode(physical), (args.datasets, 1))
            x, _ = gen.simulate_theta(theta, stage)
            return x[np.all(np.isfinite(x), axis=1)]

        start = time.perf_counter()
        x0 = summaries(truth, "truth")
        for a in args.a:
            alt = physical_at(sim, truth, a)
            m0 = sim.law.mean_matrix(sim.law.unpack(sim.space, truth)[0])
            m1 = sim.law.mean_matrix(sim.law.unpack(sim.space, alt)[0])
            assert np.allclose(m0, m1), "family must preserve the mean matrix"
            x1 = summaries(alt, f"alt-{a}")
            keep = np.concatenate([x0, x1]).std(0) > 1e-8
            n = min(len(x0), len(x1))
            acc = float(c2st(torch.tensor(x0[:n, keep]), torch.tensor(x1[:n, keep]),
                             seed=args.seed))  # fmt: skip
            # Largest standardized mean difference across features.
            pooled = np.sqrt((x0[:, keep].var(0) + x1[:, keep].var(0)) / 2) + 1e-12
            d = np.abs(x0[:, keep].mean(0) - x1[:, keep].mean(0)) / pooled
            rows.append({"clones": clones, "a": a, "B.to_A": 0.8 - 2 * a, "B.div_AA": a,
                         "c2st_accuracy": acc, "max_effect_size": float(d.max()),
                         "datasets": n})  # fmt: skip
            print(f"clones={clones:4d} a={a:.2f}  C2ST={acc:.3f}  max|d|={d.max():.2f}",
                  flush=True)  # fmt: skip
        print(f"  ({time.perf_counter() - start:.0f}s)", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(table.pivot(index="a", columns="clones", values="c2st_accuracy").round(3))


if __name__ == "__main__":
    main()
