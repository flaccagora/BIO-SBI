"""Practical readout: can the data at endpoints 4/6/8/10 tell whether the composition converged?

For the XG2 truths and the slow/fast sets, from the exact law of detected A-founded clones:

* pooled A fraction f_h = sum A / sum N over n clones per endpoint, its standard error
  (ratio estimator: sqrt(Var(A - f_h N) / n) / E[N]) and the z-score of the drift
  f_10 - f_8 and f_10 - f_4 (independent clones at each endpoint);
* the per-clone sd shrink per generation (sd_8 / sd_10)^(1/2) against the square root of
  the clone-size growth (E N_10 / E N_8)^(1/4) (= sqrt(rho) in the CLT regime): a slower
  sd shrink than sqrt(growth) flags lambda2^2 > rho (slow mixing);
* P(all B) among detected A-founded clones at 8 and 10 (a plateau = A lost, B not
  returning).

Writes ``docs/convergence/tables/readout.{csv,md}``.

  python scripts/convergence/readout.py [--n 512]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from common import SHORTNAME, TAB, TRUTHS, clone_law, mean_matrix, spectrum, truth_xg2
from sets import SETS


def moments(p12: np.ndarray, h: int) -> dict:
    p, _ = clone_law(p12, h)
    L = p.shape[0]
    A = np.arange(L)[:, None] * np.ones((1, p.shape[1]))
    N = A + np.arange(p.shape[1])[None, :]
    EA, EN = float((p * A).sum()), float((p * N).sum())
    f = EA / EN
    var_lin = float((p * (A - f * N) ** 2).sum())
    X = np.where(N > 0, A / np.maximum(N, 1), 0.0)
    m = float((p * X).sum())
    return {"f": f, "EN": EN, "var_lin": var_lin, "sd": float(np.sqrt((p * (X - m) ** 2).sum())),
            "p_allB": float(p[0, :].sum())}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    n = ap.parse_args().n
    rows = []
    for lab, p in [(gt, truth_xg2(gt)) for gt in TRUTHS] + list(SETS.items()):
        m = {h: moments(p, h) for h in (4, 6, 8, 10)}
        se = {h: np.sqrt(m[h]["var_lin"] / n) / m[h]["EN"] for h in m}
        s = spectrum(mean_matrix(p))
        rows.append({
            "set": SHORTNAME.get(lab, lab), "u_A": s["u_A"], "rho": s["rho"],
            "lambda2": s["lambda2"],
            **{f"f{h}": m[h]["f"] for h in m}, **{f"se{h}": se[h] for h in m},
            "z_10_vs_8": (m[10]["f"] - m[8]["f"]) / np.hypot(se[10], se[8]),
            "z_10_vs_4": (m[10]["f"] - m[4]["f"]) / np.hypot(se[10], se[4]),
            "sd_shrink_per_gen": (m[8]["sd"] / m[10]["sd"]) ** 0.5,
            "sqrt_growth_per_gen": (m[10]["EN"] / m[8]["EN"]) ** 0.25,
            "pallB8": m[8]["p_allB"], "pallB10": m[10]["p_allB"],
        })  # fmt: skip
        print(rows[-1]["set"], flush=True)
    df = pd.DataFrame(rows)
    df["shrink_over_sqrt_growth"] = df["sd_shrink_per_gen"] / df["sqrt_growth_per_gen"]
    df.to_csv(TAB / "readout.csv", index=False)
    cols = ["set", "u_A", "f4", "f6", "f8", "f10", "se10", "z_10_vs_8", "z_10_vs_4",
            "sd_shrink_per_gen", "sqrt_growth_per_gen", "shrink_over_sqrt_growth", "pallB8",
            "pallB10"]  # fmt: skip
    md = df[cols].round(3).to_markdown(index=False)
    (TAB / "readout.md").write_text(f"n = {n} clones per endpoint\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
