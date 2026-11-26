"""Theory check for Section 6.2: parent swaps and the Kesten-Stigum decay of their information.

Usage (from the repository root)::

    uv run python report_exact_gw/make_swap_theory.py [--results /path/to/results]

For each XG2 truth: the mean offspring matrix M (M[i, j] = expected type-j offspring of one
type-i cell per generation), its Perron root rho, second eigenvalue lambda2 and stable
composition u (left Perron vector). Checks numerically that a parent swap leaves u^T M (hence
rho and u) unchanged, and compares the decay with the generation h of the information per
clone along the 4 weakest directions (``docs/weak_directions/tables/generation_info.csv``,
A founder) with the Kesten-Stigum prediction rho^(-h) (regime lambda2^2 < rho) and with the
naive (|lambda2| / rho)^(2h). Writes ``report_exact_gw/tables/swap_theory.{csv,tex}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "report_exact_gw" / "tables"
TRUTHS = ["T1_ref", "T2_toA05", "T3_Bdiv", "T4_slowA", "T5_C7", "P1", "P2", "P3", "P4"]
LABEL = {"T1_ref": "T1", "T2_toA05": "T2", "T3_Bdiv": "T3", "T4_slowA": "T4", "T5_C7": "T5"}
# outcomes indexed by what they produce: dies, one A daughter, one B daughter, AA, AB, BB
# (for an A parent: persist = one A, to_B = one B; for a B parent: to_A = one A, persist = one B)
OUT_A = ("death", "persist", "to_B", "div_AA", "div_AB", "div_BB")
OUT_B = ("death", "to_A", "persist", "div_AA", "div_AB", "div_BB")
OFF = np.array([(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)], dtype=float)


def mean_matrix(pA: np.ndarray, pB: np.ndarray) -> np.ndarray:
    return np.vstack([pA @ OFF, pB @ OFF])


def perron(M: np.ndarray) -> tuple[float, float, np.ndarray]:
    ev = np.linalg.eigvals(M)
    ev = ev[np.argsort(-np.abs(ev))]
    w, V = np.linalg.eig(M.T)
    u = np.abs(V[:, np.argmax(w.real)].real)
    return float(ev[0].real), float(ev[1].real), u / u.sum()


def main(results: Path) -> pd.DataFrame:
    g = pd.read_csv(ROOT / "docs" / "weak_directions" / "tables" / "generation_info.csv")
    g = g[g["founder"] == "A"]
    rng = np.random.default_rng(0)
    rows = []
    for gt in TRUTHS:
        z = np.load(results / "exact_gw" / "data_size" / gt / "n1024_s0" / "reference.npz",
                    allow_pickle=True)  # fmt: skip
        t = dict(zip([str(x) for x in z["truth_names"]], z["truth"].astype(float)))
        pA = np.array([t["A." + k] for k in OUT_A])
        pB = np.array([t["B." + k] for k in OUT_B])
        M = mean_matrix(pA, pB)
        rho, l2, u = perron(M)
        # parent swap: outcome change d (sums to 0) taken from A parents and given to B parents,
        # weighted by the stable composition -> u^T M unchanged (exactly, for any step size)
        drift = 0.0
        for _ in range(200):
            d = rng.normal(size=6)
            d -= d.mean()
            d *= 0.01
            M2 = mean_matrix(pA - u[1] * d, pB + u[0] * d)
            drift = max(drift, float(np.abs(u @ M2 - u @ M).max()))
        # information decay along the 4 weakest directions, generations 4..10
        slopes, ratio = [], None
        for k, gk in g[g["truth"] == gt].groupby("k"):
            gk = gk.set_index("gen")["info_per_clone"].sort_index()
            h = np.arange(4, 11)
            slopes.append(np.polyfit(h, np.log(gk.loc[h].to_numpy()), 1)[0])
            if k == 1:
                ratio = float(gk.loc[4] / gk.loc[10])
        rows.append({
            "truth": gt, "rho": rho, "lambda2": l2, "lambda2_sq_lt_rho": l2**2 < rho, "u_A": u[0],
            "swap_max_change_uM": drift,
            "decay_per_gen_obs": float(np.exp(-np.median(slopes))),
            "decay_per_gen_obs_min": float(np.exp(-max(slopes))),
            "decay_per_gen_obs_max": float(np.exp(-min(slopes))),
            "decay_4_10_obs_k1": ratio, "rho6": rho**6,
            "naive_4_10": (rho / abs(l2)) ** 12,
        })  # fmt: skip
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "swap_theory.csv", index=False)
    lines = []
    for _, r in df.iterrows():
        naive = r["naive_4_10"]
        e = int(np.floor(np.log10(naive)))
        lines.append(" & ".join([
            LABEL.get(r["truth"], r["truth"]), f"{r['rho']:.2f}", f"${r['lambda2']:+.2f}$",
            f"{r['u_A']:.2f}", f"{r['decay_per_gen_obs']:.2f}",
            f"{r['decay_per_gen_obs_min']:.2f}--{r['decay_per_gen_obs_max']:.2f}",
            f"{r['decay_4_10_obs_k1']:.1f}", f"{r['rho6']:.1f}", f"$10^{{{e}}}$",
        ]) + r" \\")  # fmt: skip
    (TAB / "swap_theory.tex").write_text("\n".join(lines) + "\n")
    print(df.round(3).to_string())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    main(ap.parse_args().results)
