"""Summarise every run below an experiment directory, averaged over seeds.

Runs are found with :func:`bio_sbi.collect.collect` (every ``metrics.json``). They are
grouped by the sweep folder (the run's parent directory relative to ``root``, e.g.
``founder_type/fixed``) and by every Hydra override that takes more than one value
across the runs, except the seeds (``seed``, ``data_seed``), which are averaged over,
and settings that do not change results (``workers``, ``torch_threads``, ``device``).

Two tables are written next to the results and printed:

* ``<root>/summary_groups.csv`` - one row per group: number of runs, mean recovery
  error / contraction / coverage, SBC (mean absolute coverage error, minimum KS
  p-value), PPC (fraction of features outside the 95% predictive interval) and the
  observed-organoid fraction per horizon when recorded;
* ``<root>/summary_group_parameters.csv`` - one row per group x parameter: truth,
  posterior mean and sd (averaged over seeds), the seed-to-seed sd of the posterior
  mean, absolute error, contraction, 90% coverage across seeds and a ``recovered`` flag
  (contraction > 0.5, |error| < 0.05, coverage >= 0.8, as defined in TODO.md).

Usage::

    uv run python scripts/summarize.py results/e2/ignored
    uv run python scripts/summarize.py results/e1c --params 'A.*' B.to_A --group-by law.fixed.A.death
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd

from bio_sbi.collect import collect

SEED_KEYS = ("seed", "data_seed")
#: Overrides never used for automatic grouping (averaged over or not result-relevant).
IGNORED_KEYS = (*SEED_KEYS, "workers", "torch_threads", "device", "save_training_data")
RUN_METRICS = (
    "mean_abs_error",
    "mean_contraction",
    "coverage90_reference",
    "sbc_mean_abs_coverage_error",
    "sbc_min_ks_pvalue",
    "ppc_fraction_outside_95",
    "ppc_mean_abs_z",
    "total_seconds",
)


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect runs and parameter rows below ``root`` and add the ``sweep`` folder."""
    runs, params = collect(root)
    if runs.empty:
        raise SystemExit(f"No metrics.json found below {root}")
    for frame in (runs, params):
        frame["sweep"] = frame["run"].map(lambda r: Path(r).parent.relative_to(root).as_posix())
    return runs, params


def group_keys(runs: pd.DataFrame, requested: list[str] | None) -> list[str]:
    """Columns that define a group: sweep folder plus varying overrides.

    Seeds and the settings in :data:`IGNORED_KEYS` are not grouped by by default.

    Requested keys may be given without the ``override.`` prefix of the collected table.
    """
    if requested:
        keys = [k if k in runs.columns else f"override.{k}" for k in requested]
        missing = [k for k, c in zip(requested, keys, strict=True) if c not in runs.columns]
        if missing:
            known = [
                "sweep",
                *(c.removeprefix("override.") for c in runs if c.startswith("override.")),
            ]
            raise SystemExit(f"Unknown --group-by keys {missing}; available: {known}")
        return keys
    ignored = {f"override.{k}" for k in IGNORED_KEYS}
    overrides = [c for c in runs.columns if c.startswith("override.") and c not in ignored]
    varying = [c for c in overrides if runs[c].astype(str).nunique(dropna=False) > 1]
    return ["sweep", *varying]


def _bare(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``override.`` prefix from column names for output."""
    return frame.rename(columns=lambda c: c.removeprefix("override."))


def summarize_groups(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average run-level metrics over seeds within each group."""
    metrics = [c for c in RUN_METRICS if c in runs.columns]
    metrics += sorted(c for c in runs.columns if c.startswith(("detection_", "seeded_")))
    grouped = runs.fillna({k: "" for k in keys}).groupby(keys, sort=True)
    out = grouped[metrics].mean()
    out.insert(0, "runs", grouped.size())
    return out.reset_index()


def summarize_parameters(
    params: pd.DataFrame, keys: list[str], patterns: list[str] | None
) -> pd.DataFrame:
    """Per-parameter recovery averaged over seeds within each group."""
    frame = params[params["inferred"].astype(bool)] if patterns is None else params
    if patterns:
        frame = frame[frame["parameter"].map(lambda p: any(fnmatch(p, q) for q in patterns))]
    if frame.empty:
        raise SystemExit(f"No parameters match {patterns}")
    frame = frame.fillna({k: "" for k in keys}).copy()
    frame["covered"] = frame["covered"].astype(float)  # None (truth missing) -> NaN
    frame.loc[frame["at_boundary"].astype(bool), "covered"] = np.nan  # not scored
    grouped = frame.groupby([*keys, "parameter"], sort=False)
    out = grouped.agg(
        runs=("posterior_mean", "size"),
        truth=("truth", "first"),
        mean=("posterior_mean", "mean"),
        sd=("posterior_std", "mean"),
        mean_sd_across_seeds=("posterior_mean", "std"),
        abs_error=("abs_error", "mean"),
        contraction=("contraction", "mean"),
        coverage90=("covered", "mean"),
    ).reset_index()
    out["recovered"] = (
        (out["contraction"] > 0.5) & (out["abs_error"] < 0.05) & (out["coverage90"] >= 0.8)
    )
    return out.sort_values([*keys], kind="stable").reset_index(drop=True)


def main(argv: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse arguments, write the two CSV tables and print a compact view."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="Experiment directory, e.g. results/e2/ignored")
    parser.add_argument(
        "--params", nargs="+", default=None,
        help="Parameters to report (glob patterns, e.g. 'A.*'); default: all inferred",
    )  # fmt: skip
    parser.add_argument(
        "--group-by", nargs="+", default=None,
        help="Group keys (override keys or 'sweep'); default: sweep folder + varying overrides",
    )  # fmt: skip
    parser.add_argument("--out", default=None, help="Output prefix (default: ROOT/summary)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    runs, params = load(root)
    keys = group_keys(runs, args.group_by)
    groups = summarize_groups(runs, keys)
    table = summarize_parameters(params, keys, args.params)
    prefix = Path(args.out) if args.out else root / "summary"
    groups, table = _bare(groups), _bare(table)
    keys = [k.removeprefix("override.") for k in keys]
    groups.to_csv(f"{prefix}_groups.csv", index=False)
    table.to_csv(f"{prefix}_group_parameters.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 60)
    print(f"{len(runs)} runs in {len(groups)} groups (by {', '.join(keys)}); seeds averaged.\n")
    print(groups.round(3).to_string(index=False))
    print()
    shown = ["parameter", "truth", "mean", "sd", "abs_error", "contraction", "coverage90"]
    print(table[[*keys, *shown, "recovered"]].round(3).to_string(index=False))
    print(f"\nWrote {prefix}_groups.csv and {prefix}_group_parameters.csv")
    return groups, table


if __name__ == "__main__":
    main()
