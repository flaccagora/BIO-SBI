"""Aggregate many runs into one table (``bio-sbi-collect``).

Walks a directory tree, finds every ``metrics.json`` written by
:func:`bio_sbi.pipeline.run_experiment`, and writes one row per run (and optionally one
row per run x parameter) combining the Hydra overrides with headline metrics. This is
the entry point for comparing simulators, methods and designs across a multirun sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def _overrides(run: Path) -> dict:
    path = run / ".hydra" / "overrides.yaml"
    if not path.exists():
        return {}
    items = yaml.safe_load(path.read_text()) or []
    return dict(item.split("=", 1) for item in items if "=" in item)


def collect(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(runs, parameters)`` tables for every run below ``root``."""
    runs, params = [], []
    for path in sorted(Path(root).rglob("metrics.json")):
        run = path.parent
        m = json.loads(path.read_text())
        row = {
            "run": str(run),
            **{f"override.{k}": v for k, v in _overrides(run).items()},
            "simulator": m["simulator"],
            "law": m["law"],
            "inference": m["inference"],
            "rounds": m["rounds"],
            "simulations": m["simulations"],
            "horizons": " ".join(f"{h:g}" for h in m["design"]["horizons"]),
            "clones": " ".join(str(c) for c in m["design"]["clones"]),
            "mean_abs_error": m["recovery"]["mean_abs_error"],
            "mean_contraction": m["recovery"]["mean_contraction"],
            "coverage90_reference": m["recovery"]["coverage90"],
            "training_seconds": sum(r["training_seconds"] for r in m["training"]),
            "simulation_seconds": sum(r["simulation_seconds"] for r in m["training"]),
            "total_seconds": m["total_seconds"],
        }
        cal = m.get("calibration")
        if isinstance(cal, dict):
            row["sbc_cases"] = cal["cases"]
            row["sbc_mean_abs_coverage_error"] = cal["mean_abs_coverage_error"]
            row["sbc_min_ks_pvalue"] = min(cal["ks_pvalue"].values())
        runs.append(row)
        for p in m["recovery"]["per_parameter"]:
            params.append({"run": str(run), **{k: row[k] for k in row if k != "run"}, **p})
    return pd.DataFrame(runs), pd.DataFrame(params)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``bio-sbi-collect ROOT [--out PREFIX]``."""
    parser = argparse.ArgumentParser(description="Collect metrics.json files into CSV tables")
    parser.add_argument("root", help="Directory containing runs (e.g. multirun/2026-11-05)")
    parser.add_argument("--out", default=None, help="Output prefix (default: ROOT/summary)")
    args = parser.parse_args(argv)
    runs, params = collect(args.root)
    if runs.empty:
        raise SystemExit(f"No metrics.json found below {args.root}")
    prefix = Path(args.out) if args.out else Path(args.root) / "summary"
    runs.to_csv(f"{prefix}_runs.csv", index=False)
    params.to_csv(f"{prefix}_parameters.csv", index=False)
    print(runs.drop(columns=["run"]).to_string(index=False))
    print(f"\nWrote {prefix}_runs.csv and {prefix}_parameters.csv")


if __name__ == "__main__":
    main()
