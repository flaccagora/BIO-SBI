"""Which optimisation starts matter? (pilot run with --settings rich)

For every LRT of the pilot folders, recompute Lambda with subsets of the start families
and report how often (and by how much) Lambda changes against the full set. H0 maxima
found by the full fit (``from_full``) are counted as part of the full set.

  uv run python scripts/dediff_test/pilot_starts.py results/exact_gw/dediff_test/*_pilot
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H0_SUBSETS = {
    "all": lambda k: True,
    "generic+3random+warm": lambda k: k in ("generic", "warm_0") or k in ("random_0", "random_1", "random_2"),
    "generic+1random+warm": lambda k: k in ("generic", "warm_0", "random_0"),
    "random only (6)": lambda k: k.startswith("random"),
}
FULL_SUBSETS = {
    "all": lambda k: True,
    "default": lambda k: k in ("theta0", "theta0_opt", "random_0", "warm_0", "pin0.02", "pin0.3")
    or k.startswith("q0.01_"),
    "no pins": lambda k: not k.startswith("pin"),
    "no q starts": lambda k: not k.startswith("q"),
    "no random": lambda k: not k.startswith("random"),
    "theta0 + q0.01 only": lambda k: k.startswith("theta0") or k.startswith("q0.01_"),
    "random only (4)": lambda k: k.startswith("random") or k == "theta0",
}


def main() -> None:
    rows = []
    for root in map(Path, sys.argv[1:]):
        for f in sorted(root.rglob("s*.json")):
            d = json.loads(f.read_text())
            for an, recs in d["results"].items():
                for r in recs:
                    h0 = r["h0_ll"]
                    full = r["full_ll"]
                    ll0_all = r["ll0"]  # includes from_full improvements
                    ll1_all = r["ll1"]
                    for hs, hf in H0_SUBSETS.items():
                        ll0 = max(v for k, v in h0.items() if hf(k))
                        for fs, ff in FULL_SUBSETS.items():
                            # theta0 candidate of the subset = its own H0 optimum
                            cand = [v for k, v in full.items() if ff(k) and k != "theta0"]
                            ll1 = max([ll0, *cand])
                            lam = 2 * (ll1 - ll0)
                            rows.append({"study": d["study"], "cell": d["cell"], "seed": d["seed"],
                                         "analysis": an, "n": r["n"], "h0": hs, "full": fs,
                                         "lam_ref": r["lambda"], "lam": lam,
                                         "dlam": lam - r["lambda"],
                                         "h0_gap": ll0_all - ll0, "full_gap": ll1_all - ll1})
    df = pd.DataFrame(rows)
    df["big"] = df.dlam.abs() > 0.05
    print("LRTs:", len(df[(df.h0 == "all") & (df.full == "all")]))
    t = df.groupby(["h0", "full"]).agg(changed=("big", "mean"), max_abs=("dlam", lambda x: np.abs(x).max()),
                                        mean_abs=("dlam", lambda x: np.abs(x).mean()))
    print(t.round(4).to_string())
    sub = df[(df.h0 == "generic+3random+warm") & (df.full == "default") & df.big]
    print("\ndefault subset, |dLambda| > 0.05:")
    print(sub[["study", "cell", "seed", "analysis", "n", "lam_ref", "lam", "h0_gap", "full_gap"]].to_string())
    # which start won
    best = []
    for root in map(Path, sys.argv[1:]):
        for f in sorted(root.rglob("s*.json")):
            d = json.loads(f.read_text())
            for an, recs in d["results"].items():
                for r in recs:
                    best.append({"b0": re.sub(r"_\d+$", "", r["best_start0"]),
                                 "b1": re.sub(r"_\d+$", "", r["best_start1"]),
                                 "tie1": r["full_starts_tied"], "impr": r["h0_improved_by_full"],
                                 "miss0": r.get("missed0", 0), "miss1": r.get("missed1", 0),
                                 "null": d["m_ba"] == 0 and an != "ignored"})
    b = pd.DataFrame(best)
    print("\nH0 improved by full fit:", (b.impr > 1e-6).sum(), "max", b.impr.max())
    print("truth start better (null cells, >0.001 nats): H0", (b[b.null].miss0 > 1e-3).sum(),
          "full", (b[b.null].miss1 > 1e-3).sum(), "of", b.null.sum(),
          "; max", b[b.null].miss0.max(), b[b.null].miss1.max())
    print(b.b1.value_counts().to_string())


if __name__ == "__main__":
    main()
