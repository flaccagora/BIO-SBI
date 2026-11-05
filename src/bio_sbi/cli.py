"""Command-line entry points.

* ``bio-sbi`` - Hydra app running one experiment per (multi)run job, e.g.
  ``bio-sbi simulator=bellman_harris inference=nle budget=smoke``.
* ``bio-sbi-infer RUN_DIR DATA.csv`` - condition a trained run on measured counts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .pipeline import infer_measured, run_experiment


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Run one experiment in the Hydra output directory.

    Returns the mean absolute recovery error so Hydra sweepers can optimise it.
    """
    if cfg.torch_threads:
        torch.set_num_threads(int(cfg.torch_threads))
    out = Path(HydraConfig.get().runtime.output_dir)
    metrics = run_experiment(cfg, out)
    return metrics["recovery"]["mean_abs_error"]


def infer_main(argv: list[str] | None = None) -> None:
    """Parse arguments for ``bio-sbi-infer`` and run measured-data inference."""
    parser = argparse.ArgumentParser(description=infer_measured.__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="Finished amortised run (contains posterior.pt)")
    parser.add_argument("data", help="CSV with columns horizon,<type names...>")
    parser.add_argument("--out", default=None, help="Output directory (default: RUN/measured)")
    parser.add_argument("--samples", type=int, default=4096)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = infer_measured(
        args.run_dir, args.data, args.out or Path(args.run_dir) / "measured", args.samples
    )
    print(f"Posterior samples written to {out}")


if __name__ == "__main__":
    main()
