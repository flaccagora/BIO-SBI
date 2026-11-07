"""Convert the CRC0322 organoid counts into ``bio-sbi-infer`` CSVs.

Reads the day-6 and day-10 tables of ``submodules/GW/data`` (the delimiter is
detected per file: the day-6 table is tab-separated, the day-10 one comma-separated
with two leading index columns) and writes, for each data variant and each simulator
time unit, a ``horizon,A,B`` CSV (A = LGR5 / stem, B = KRT20 / differentiated):

* ``crc0322_3d_mean_{bh,gw}.csv`` - round(mean of the three 3D whole-organoid
  estimates), per organoid;
* ``crc0322_3d_sample{1,2,3}_{bh,gw}.csv`` - each 3D estimate alone (sensitivity);
* ``crc0322_2d_{bh,gw}.csv`` - 2D section counts ``n_LGR5``, ``n_KRT20``.

Horizons: Bellman-Harris in hours (24 h/day), Galton-Watson in generations (one per
day, as ``simulator.real_horizons``). Organoids whose A + B is 0 in a variant are
dropped from that variant: the default observation model keeps detected clones only,
so the simulator never produces them (each drop is listed in the README).

A ``README.md`` with descriptive statistics and data-quality flags is written next to
the CSVs.

Usage: uv run python scripts/prepare_real_data.py [--data-dir DIR] [--out data/real]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {6: "df_6D_withmapping.csv", 10: "df_10D_withmapping_new.csv"}
SAMPLES = (1, 2, 3)
UNITS = {"bh": ("hours", 24.0), "gw": ("generations", 1.0)}
COUNT_COLUMNS = [
    "n_KRT20", "n_LGR5", "n_cells",
    *[f"n_cells_3D_sample{i}" for i in SAMPLES],
    *[f"LGR5_sample{i}" for i in SAMPLES],
    *[f"KRT20_sample{i}" for i in SAMPLES],
]  # fmt: skip


def read_table(path: str | Path) -> pd.DataFrame:
    """Read one organoid table, detecting a tab or comma delimiter.

    Unnamed index columns (written by ``DataFrame.to_csv``) are dropped.
    """
    path = Path(path)
    header = path.open().readline()
    table = pd.read_csv(path, sep="\t" if header.count("\t") > header.count(",") else ",")
    table = table.loc[:, ~table.columns.str.match(r"^Unnamed|^$")]
    missing = {"organoid", *COUNT_COLUMNS} - set(table.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {sorted(missing)}")
    return table


def quality_flags(table: pd.DataFrame, day: int) -> list[str]:
    """Human-readable data-quality issues of one day's table."""
    flags = []
    values = table[COUNT_COLUMNS]
    if values.isna().any().any():
        flags.append(f"day {day}: missing values {values.isna().sum()[lambda s: s > 0].to_dict()}")
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if (numeric.dropna() % 1 != 0).any().any():
        flags.append(f"day {day}: non-integer counts")
    if (numeric < 0).any().any():
        flags.append(f"day {day}: negative counts")
    if table["organoid"].duplicated().any():
        flags.append(f"day {day}: duplicated organoid ids")
    if (numeric["n_cells"] == 0).any():
        flags.append(f"day {day}: {int((numeric['n_cells'] == 0).sum())} organoids with 0 2D cells")
    for i in SAMPLES:
        zero = table.loc[numeric[f"n_cells_3D_sample{i}"] == 0, "organoid"].tolist()
        if zero:
            flags.append(
                f"day {day}: 3D sample {i} is 0 cells for {len(zero)} organoid(s) whose 2D "
                f"section is non-empty ({', '.join(zero)})"
            )
    return flags


def variants(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``{variant: DataFrame(organoid, A, B)}`` for one day (zero-total rows kept)."""
    out = {
        "2d": pd.DataFrame({"A": table["n_LGR5"], "B": table["n_KRT20"]}),
        "3d_mean": pd.DataFrame(
            {
                "A": np.rint(table[[f"LGR5_sample{i}" for i in SAMPLES]].mean(axis=1)),
                "B": np.rint(table[[f"KRT20_sample{i}" for i in SAMPLES]].mean(axis=1)),
            }
        ),
    }
    for i in SAMPLES:
        out[f"3d_sample{i}"] = pd.DataFrame(
            {"A": table[f"LGR5_sample{i}"], "B": table[f"KRT20_sample{i}"]}
        )
    for frame in out.values():
        frame.insert(0, "organoid", table["organoid"].to_numpy())
        frame[["A", "B"]] = frame[["A", "B"]].astype(np.int64)
    return out


def neither_type(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-organoid fraction of cells that are neither LGR5 nor KRT20."""
    out = {"2d": 1 - (table["n_LGR5"] + table["n_KRT20"]) / table["n_cells"]}
    for i in SAMPLES:
        total = table[f"n_cells_3D_sample{i}"]
        typed = table[f"LGR5_sample{i}"] + table[f"KRT20_sample{i}"]
        out[f"3d_sample{i}"] = (1 - typed / total.where(total > 0)).dropna()
    return out


def describe(frame: pd.DataFrame) -> dict:
    """Descriptive statistics of one cohort's ``A, B`` counts."""
    a, b = frame["A"], frame["B"]
    total = a + b
    return {
        "organoids": len(frame),
        "mean A": a.mean(), "median A": a.median(), "max A": a.max(),
        "mean B": b.mean(), "median B": b.median(), "max B": b.max(),
        "A fraction": a.sum() / total.sum() if total.sum() else np.nan,
        "frac A=0": (a == 0).mean(), "frac B=0": (b == 0).mean(),
        "mean cells": total.mean(), "median cells": total.median(), "max cells": total.max(),
    }  # fmt: skip


def convert(tables: dict[int, pd.DataFrame], out: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Write all variant CSVs to ``out``.

    Args:
        tables: ``{day: organoid table}``.
        out: Output directory.

    Returns:
        ``(stats, dropped)``: one stats row per (variant, day) and a list of the
        zero-total organoids dropped per variant.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per_variant: dict[str, list[pd.DataFrame]] = {}
    stats, dropped = [], []
    for day, table in sorted(tables.items()):
        for name, frame in variants(table).items():
            empty = frame["A"] + frame["B"] == 0
            if empty.any():
                dropped.append(f"{name}, day {day}: {', '.join(frame.loc[empty, 'organoid'])}")
            kept = frame[~empty].assign(day=day)
            per_variant.setdefault(name, []).append(kept)
            # Largest clone a single founder can reach in `day` binary generations.
            above = int((kept["A"] + kept["B"] > 2**day).sum())
            stats.append({"variant": name, "day": day, **describe(kept), "cells > 2^day": above})
    for name, frames in per_variant.items():
        frame = pd.concat(frames, ignore_index=True)
        for tag, (_, scale) in UNITS.items():
            horizon = (frame["day"] * scale).round().astype(np.int64)
            csv = pd.DataFrame({"horizon": horizon, "A": frame["A"], "B": frame["B"]})
            csv.to_csv(out / f"crc0322_{name}_{tag}.csv", index=False)
    return pd.DataFrame(stats), dropped


def _markdown(frame: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Minimal GitHub table (no ``tabulate`` dependency)."""

    def cell(v):
        if isinstance(v, float | np.floating):
            return format(v, floatfmt)
        return str(v)

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *rows])


def readme(tables: dict[int, pd.DataFrame], stats: pd.DataFrame, dropped: list[str]) -> str:
    """Text of ``data/real/README.md``."""
    flags = [f for day, t in sorted(tables.items()) for f in quality_flags(t, day)]
    neither_rows, ratio_rows = [], []
    for day, table in sorted(tables.items()):
        for name, frac in neither_type(table).items():
            neither_rows.append({
                "day": day, "count": name, "organoids": len(frac), "mean": frac.mean(),
                "max": frac.max(), "organoids with any": int((frac > 0).sum()),
            })  # fmt: skip
        total3d = table[[f"n_cells_3D_sample{i}" for i in SAMPLES]].mean(axis=1)
        ratio = total3d / table["n_cells"]
        a2 = table["n_LGR5"] / table["n_cells"]
        a3 = table[[f"LGR5_sample{i}" for i in SAMPLES]].sum(axis=1) / table[
            [f"n_cells_3D_sample{i}" for i in SAMPLES]
        ].sum(axis=1)
        cells3d = table[[f"n_cells_3D_sample{i}" for i in SAMPLES]]
        cv = (cells3d.std(axis=1) / cells3d.mean(axis=1)).median()
        ratio_rows.append({
            "day": day, "median 3D/2D cells": ratio.median(),
            "range 3D/2D": f"{ratio.min():.2f}-{ratio.max():.2f}",
            "corr(2D, 3D cells)": np.corrcoef(table["n_cells"], total3d)[0, 1],
            "corr(2D, 3D A fraction)": np.corrcoef(a2, a3)[0, 1],
            "median CV of 3 samples": cv,
        })  # fmt: skip
    files = "\n".join(
        f"* `crc0322_{v}_{{bh,gw}}.csv`"
        for v in ("3d_mean", "3d_sample1", "3d_sample2", "3d_sample3", "2d")  # fmt: skip
    )
    sources = ", ".join(f"`{f}` (day {d})" for d, f in sorted(FILES.items()))
    shown = stats.copy()
    for col in ("organoids", "max A", "max B", "max cells", "cells > 2^day"):
        shown[col] = shown[col].astype(int)
    return f"""# CRC0322 organoid counts (real data)

Generated by `scripts/prepare_real_data.py` from `submodules/GW/data`: {sources}.
Do not edit by hand; rerun the script.

## Files

One row per organoid, columns `horizon,A,B` (input of `bio-sbi-infer`).
A = LGR5+ (stem), B = KRT20+ (differentiated).

{files}

`_bh` files give horizons in hours (day 6 = 144, day 10 = 240; Bellman-Harris),
`_gw` files in generations (6, 10; Galton-Watson, one generation per day, as
`simulator.real_horizons`). `3d_mean` is round(mean of the three 3D whole-organoid
estimates, zeros included); `3d_sample<i>` uses estimate i alone (sensitivity);
`2d` uses the 2D section counts `n_LGR5`, `n_KRT20`.

Organoids with A + B = 0 are dropped from a file (the detected-clone selection of
the model never produces them):
{chr(10).join(f"* {d}" for d in dropped) or "* none"}

## Descriptive statistics (per file and day, after dropping)

`A fraction` = sum A / sum (A + B) over organoids; `cells` = A + B.
`cells > 2^day`: organoids larger than a single founder can grow in `day` binary
divisions, i.e. impossible for Galton-Watson at one generation per day (`_gw` files).

{_markdown(shown)}

## Cells that are neither LGR5 nor KRT20

The model only has types A and B, so cells of neither type would have to be ignored
or modelled as a third type. Fraction of each organoid that is neither type
(2D: `1 - (n_LGR5 + n_KRT20) / n_cells`; 3D sample i:
`1 - (LGR5_sample<i> + KRT20_sample<i>) / n_cells_3D_sample<i>`, organoids with a
0-cell estimate skipped):

{_markdown(pd.DataFrame(neither_rows), ".3f")}

## 2D section vs 3D estimates

{_markdown(pd.DataFrame(ratio_rows), ".3f")}

## Flags

{chr(10).join(f"* {f}" for f in flags) or "* none"}

## Notes for modelling

* Every cell is called LGR5 or KRT20 (the "neither" fractions above): the typing is
  a forced binary call, so double-negative / other cells are absorbed into A or B.
* The 3D estimates are not independent measurements: the per-organoid A fraction is
  almost identical to the 2D section's (correlation above) and total cells are a
  roughly constant multiple of the 2D count. They look like the 2D section scaled to
  the whole organoid (the "mapping"), with sample-to-sample noise in the totals.
* A 0-cell 3D sample for an organoid with a non-empty 2D section is a failed
  estimate, not an empty organoid. `3d_mean` averages it in (biasing those few
  organoids low); the per-sample files drop it.
* 3D organoids are ~4-5x larger than the 2D sections; check that the simulator's
  `max_cells`, lifetime prior and the prior predictive cover these sizes before
  training at the real design.
* Galton-Watson at one generation per day (`_gw` horizons 6, 10) cannot produce the
  day-6 3D counts: see `cells > 2^day`. Fitting 3D counts with Galton-Watson needs
  shorter generations (e.g. 20 h: horizons 7.2 and 12, rewrite the `_gw` horizons)
  or Bellman-Harris (hours).
"""


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, convert both days and write the README."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "submodules" / "GW" / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "real")
    args = parser.parse_args(argv)
    tables = {day: read_table(args.data_dir / name) for day, name in FILES.items()}
    stats, dropped = convert(tables, args.out)
    (args.out / "README.md").write_text(readme(tables, stats, dropped))
    print(stats.round(2).to_string(index=False))
    print(f"Wrote {len(list(args.out.glob('*.csv')))} CSVs and README.md to {args.out}")


if __name__ == "__main__":
    main()
