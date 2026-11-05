"""Insert or refresh the standard "Data" section of results folders' ``info.md``.

For each given folder, all runs below it (``*/reference.npz`` with ``.hydra/``) are read
and the section produced by :func:`bio_sbi.datacard.describe_data` replaces any existing
``## Data`` section (or is inserted before ``## Results``/``## Reading``). ``info.md`` is
created if missing.

Usage::

    uv run python scripts/add_data_section.py results/e3b results/e1a [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bio_sbi.datacard import describe_data, write_data_section


def main() -> None:
    """Parse arguments and update (or print) the Data sections."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="print instead of writing")
    args = parser.parse_args()
    for folder in args.folders:
        if args.dry_run:
            print(f"===== {folder}\n{describe_data(folder)}")
        else:
            print(f"updated {write_data_section(folder)}")


if __name__ == "__main__":
    main()
