"""Test whether data distinguish the true parameters from what the posterior reports.

For finished runs, simulate many datasets at the truth and at (a) the posterior mean
and (b) several individual posterior draws, under the run's own design and inference
model, and run a classifier two-sample test on the summaries. Accuracy near 0.5 means
the posterior's parameters explain data as well as the truth (non-identifiable at this
design); accuracy near 1 means the information is in the data and the posterior missed it.

Usage::

    uv run python scripts/truth_vs_posterior.py results/b_row/two_type/*simulations=20000*
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sbi.utils.metrics import c2st

from bio_sbi.pipeline import load_run


def main() -> None:
    """Run the test for every given run directory."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--datasets", type=int, default=400)
    parser.add_argument("--draws", type=int, default=3, help="individual posterior draws")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    rows = []
    for run in map(Path, args.runs):
        cfg, comp, _ = load_run(run)
        sim, gen = comp.simulator, comp.generator
        truth = np.load(run / "reference.npz")["truth"]
        samples = pd.read_csv(run / "posterior_samples.csv.gz")[list(sim.parameter_names)]
        candidates = {"posterior_mean": samples.mean().to_numpy()}
        rng = np.random.default_rng(0)
        for i in rng.choice(len(samples), args.draws, replace=False):
            candidates[f"draw_{i}"] = samples.iloc[i].to_numpy()

        def summaries(p, stage, sim=sim, gen=gen):
            x, _ = gen.simulate_theta(np.tile(sim.encode(p), (args.datasets, 1)), stage)
            return x[np.all(np.isfinite(x), axis=1)]

        x0 = summaries(truth, "truth")
        for label, p in candidates.items():
            p = sim.validate(p)
            x1 = summaries(p, label)
            keep = np.concatenate([x0, x1]).std(0) > 1e-8
            n = min(len(x0), len(x1))
            acc = float(c2st(torch.tensor(x0[:n, keep]), torch.tensor(x1[:n, keep])))
            names = sim.parameter_names
            rows.append({"run": run.name, "candidate": label, "c2st": acc,
                         "B.to_A": p[names.index("B.to_A")],
                         "B.div_AA": p[names.index("B.div_AA")]})  # fmt: skip
            print(
                f"{run.name[:40]:40s} {label:15s} B.to_A={rows[-1]['B.to_A']:.2f} "
                f"B.div_AA={rows[-1]['B.div_AA']:.2f}  C2ST={acc:.3f}",
                flush=True,
            )
    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
