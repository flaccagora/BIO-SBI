"""Reorganise finished XG2 runs into <root>/<gt>/n<n>_s<seed>/ (one folder per cell).

Every run folder with ``metrics.json`` and ``.hydra/overrides.yaml`` under ``root`` is
moved (rename, same filesystem) to ``<root>/<gt>/n<clones>_s<seed>``; everything else at
the top level (sweep folders with their multirun.yaml / info.md, logs, aborted attempts)
goes to ``<root>/_sweeps/`` unchanged. Writes ``<root>/runs_index.csv`` (gt, n, seed,
kernel, old and new path) and updates the ``run`` path stored in each ``mle.json``.
Top-level analysis outputs (xg2_*.csv, xg2_analysis.txt, xg2_reruns.txt) stay in place.
Refuses to run if two runs map to the same cell or a destination exists.

  uv run python scripts/reorganise_xg2.py results/exact_gw/data_size [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

KEEP_TOP = re.compile(r"^(xg2_.*|runs_index\.csv|_sweeps)$")


def cell(run: Path) -> dict:
    ov = list(OmegaConf.load(run / ".hydra" / "overrides.yaml"))
    get = lambda key: next((o.split("=", 1)[1] for o in ov if o.startswith(key + "=")), None)  # noqa: E731
    return {"gt": get("gt"), "n": int(get("design.clones")), "seed": int(get("seed")),
            "kernel": get("inference.method.kernel") or "loop"}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = args.root.resolve()
    rows = []
    for m in sorted(root.rglob("metrics.json")):
        run = m.parent
        if not (run / ".hydra" / "overrides.yaml").exists() or run.parent.name == "_sweeps":
            continue
        if run.parent.parent == root and run.parent.name not in ("_sweeps",) and \
                re.fullmatch(r"n\d+_s\d+", run.name):
            continue  # already reorganised
        c = cell(run)
        dest = root / c["gt"] / f"n{c['n']}_s{c['seed']}"
        rows.append({**c, "old": str(run), "new": str(dest), "sweep": run.parent.name})
    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing to move")
        return
    dup = df[df.duplicated(["gt", "n", "seed"], keep=False)]
    if len(dup):
        print("ERROR: several runs map to the same cell:\n", dup[["gt", "n", "seed", "old"]])
        sys.exit(1)
    exists = [d for d in df.new if Path(d).exists()]
    if exists:
        print("ERROR: destinations exist:", exists[:5])
        sys.exit(1)
    print(df.groupby(["gt", "kernel"]).size().unstack(fill_value=0))
    print(f"{len(df)} runs to move; cells per gt: {df.groupby('gt').size().to_dict()}")
    moved_from = {Path(o) for o in df.old}
    top_other = [p for p in root.iterdir()
                 if not KEEP_TOP.match(p.name) and p.name not in set(df["gt"])]  # fmt: skip
    print("to _sweeps/:", sorted(p.name for p in top_other))
    if args.dry_run:
        print(df[["gt", "n", "seed", "kernel", "sweep"]].sort_values(["gt", "n", "seed"])
              .to_string(index=False))  # fmt: skip
        return
    for r in df.itertuples():
        Path(r.new).parent.mkdir(exist_ok=True)
        shutil.move(r.old, r.new)
        mle = Path(r.new) / "mle.json"
        if mle.exists():
            d = json.loads(mle.read_text())
            d["run"] = r.new
            mle.write_text(json.dumps(d, indent=2))
    sweeps = root / "_sweeps"
    sweeps.mkdir(exist_ok=True)
    for p in top_other:
        if p in moved_from:
            continue
        shutil.move(str(p), str(sweeps / p.name))
    index = root / "runs_index.csv"
    df = df.assign(old=df.old.str.replace(str(root) + "/", "", regex=False),
                   new=df.new.str.replace(str(root) + "/", "", regex=False))  # fmt: skip
    if index.exists():
        df = pd.concat([pd.read_csv(index), df], ignore_index=True)
    df.sort_values(["gt", "n", "seed"]).to_csv(index, index=False)
    print(f"moved {len(rows)} runs; index {index}")


if __name__ == "__main__":
    main()
