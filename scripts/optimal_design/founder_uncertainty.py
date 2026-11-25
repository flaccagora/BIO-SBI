"""Optimal-design gains when the founder (progenitor) type is not perfectly known.

Per clone the observed law is the mixture over its hidden founder,
``p(c) = w pA(c) + (1 - w) pB(c)`` with ``w`` = P(founder is A) for that cohort, conditioned on
detection with ``q = w qA + (1 - w) qB``. Information about the 10 transition directions
(whitened by the flat prior, as in the optimal-design study) and about ``w``:

* recorded founders: ``w`` = 1 (A cohort) or 0 (B cohort);
* sorted, known purity ``pi``: A cohort ``w = pi``, B cohort ``w = 1 - pi``, ``w`` known;
* sorted, unknown purity: as above, but each cohort's ``w`` is a nuisance estimated from that
  cohort's clones (Schur complement over the cohort's endpoints);
* unsorted: one cohort, ``w = p_A`` (0.7) unknown, estimated.

Reports the weakest whitened eigenvalue and the asymptotic sd of B->A at N = 512 clones for
the current design (A-founded 4/6/8/10) and the recommended design R1 (3/8 A2 + 1/4 B1 +
1/4 B2 + 1/8 A10), for the 9 XG2 truths (T1/T2/T4 at eps = 0.01, as in the study).

  uv run python scripts/optimal_design/founder_uncertainty.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from od_common import (  # noqa: E402
    I_BTOA, PRIOR_PREC, SHORTNAME, TAB, TRUTHS, c_vector, design_truth, gw_model, tangent_basis,
)  # fmt: skip

from bio_sbi.inference.gw_exact import GridSpec, detection_and_grad, law_and_jac  # noqa: E402
from bio_sbi.inference.gw_fisher import grid_size  # noqa: E402

N_TOTAL = 512
GENS = (1, 2, 4, 6, 8, 10)


def laws(p12: np.ndarray, h: int, U: np.ndarray) -> dict:
    """Law and Jacobian of one clone at generation h for A and B founders, on a common grid."""
    P = np.asarray(p12, float).reshape(2, 6)
    k = U.shape[1]
    dP = np.zeros((k, 6))
    rows = np.zeros(k, np.int64)
    for d in range(k):
        r = 0 if np.linalg.norm(U[:6, d]) > 0.5 else 1
        rows[d] = r
        dP[d] = U[6 * r : 6 * r + 6, d]
    base = gw_model()
    models = {f: replace(base, founder=np.array([1.0, 0.0] if f == "A" else [0.0, 1.0]))
              for f in "AB"}  # fmt: skip
    L = max(grid_size(m, P, h)[0] for m in models.values())
    out = {}
    for f, m in models.items():
        p, dp = law_and_jac(m, P, dP, rows, h, GridSpec(L, L, 1.0, 1.0))
        q, dq = detection_and_grad(m, P, dP, rows, h)
        p = p.copy()
        p[0, 0] = 0.0
        out[f] = (p, dp, float(q), np.asarray(dq))
    return out


def mix_info(L: dict, w: float, floor: float = 1e-13) -> np.ndarray:
    """(k+1) x (k+1) information of one detected clone with founder weight w; last = w."""
    pA, dpA, qA, dqA = L["A"]
    pB, dpB, qB, dqB = L["B"]
    p = w * pA + (1 - w) * pB
    q = w * qA + (1 - w) * qB
    dp = w * dpA + (1 - w) * dpB
    dq = w * dqA + (1 - w) * dqB
    keep = p > floor
    pc = p[keep] / q
    s_th = dp[:, keep] / p[keep][None, :] - (dq / q)[:, None]
    s_w = (pA[keep] - pB[keep]) / p[keep] - (qA - qB) / q
    S = np.vstack([s_th, s_w[None, :]])
    M = (S * pc[None, :]) @ S.T
    return 0.5 * (M + M.T)


def cohort_info(Ls: dict, shares: dict, w: float, nuisance: bool) -> np.ndarray:
    """Whitened k x k information per clone of one cohort (shares per generation)."""
    M = sum(s * mix_info(Ls[h], w) for h, s in shares.items())
    k = M.shape[0] - 1
    th = M[:k, :k]
    if nuisance:
        tw, ww = M[:k, k], M[k, k]
        th = th - np.outer(tw, tw) / ww
    return th / PRIOR_PREC


def summarise(K: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    lam = float(np.linalg.eigvalsh(K)[0])
    var = float(c @ np.linalg.solve(N_TOTAL * PRIOR_PREC * K, c))
    tr = float(np.trace(np.linalg.inv(K)))  # summed variance of the 12 (up to 1/(42 n))
    return lam, float(np.sqrt(max(var, 0.0))), tr


def scenarios(Ls: dict) -> dict:
    cur = {4: 0.25, 6: 0.25, 8: 0.25, 10: 0.25}
    r1A, r1B = {2: 3 / 8, 10: 1 / 8}, {1: 1 / 4, 2: 1 / 4}
    r1_all = {1: 1 / 4, 2: 5 / 8, 10: 1 / 8}  # same time points, founders not separated
    out = {
        "current, A founders recorded": cohort_info(Ls, cur, 1.0, False),
        "current, unsorted (p_A 0.7 unknown)": cohort_info(Ls, cur, 0.7, True),
        "R1, founders recorded": cohort_info(Ls, r1A, 1.0, False) + cohort_info(Ls, r1B, 0.0, False),
    }
    for pi in (0.95, 0.9, 0.8):
        out[f"R1, sorted, purity {pi:.2f} known"] = (cohort_info(Ls, r1A, pi, False)
                                                     + cohort_info(Ls, r1B, 1 - pi, False))  # fmt: skip
        out[f"R1, sorted, purity {pi:.2f} unknown"] = (cohort_info(Ls, r1A, pi, True)
                                                       + cohort_info(Ls, r1B, 1 - pi, True))  # fmt: skip
    out["R1 time points, unsorted (p_A 0.7 unknown)"] = cohort_info(Ls, r1_all, 0.7, True)
    out["R1 time points, unsorted (p_A 0.7 known)"] = cohort_info(Ls, r1_all, 0.7, False)
    return out


def main() -> None:
    import pandas as pd

    U = tangent_basis(None)
    c = c_vector(U, I_BTOA)
    rows = []
    for gt in TRUTHS:
        p12 = design_truth(gt)
        Ls = {h: laws(p12, h, U) for h in GENS}
        for name, K in scenarios(Ls).items():
            lam, sd, tr = summarise(K, c)
            rows.append({"truth": SHORTNAME.get(gt, gt), "scenario": name, "lam_min": lam,
                         "inv_lam_min": 1 / lam if lam > 0 else np.inf, "sd_BtoA_N512": sd,
                         "trace12": tr})  # fmt: skip
        print(gt, "done", flush=True)
    df = pd.DataFrame(rows)
    ref = df[df.scenario == "current, A founders recorded"].set_index("truth")
    df["gain_lam_vs_current"] = df.lam_min / df.truth.map(ref.lam_min)
    df["gain_var_BtoA_vs_current"] = (df.truth.map(ref.sd_BtoA_N512) / df.sd_BtoA_N512) ** 2
    df["gain_all12_vs_current"] = df.truth.map(ref.trace12) / df.trace12
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / "founder_uncertainty.csv", index=False)
    summ = df.groupby("scenario", sort=False).agg(
        gain_lam_median=("gain_lam_vs_current", "median"),
        gain_lam_min=("gain_lam_vs_current", "min"), gain_lam_max=("gain_lam_vs_current", "max"),
        gain_var_median=("gain_var_BtoA_vs_current", "median"),
        gain_var_min=("gain_var_BtoA_vs_current", "min"),
        gain_var_max=("gain_var_BtoA_vs_current", "max"),
        sd_min=("sd_BtoA_N512", "min"), sd_max=("sd_BtoA_N512", "max"),
        gain_all12_median=("gain_all12_vs_current", "median"),
        gain_all12_min=("gain_all12_vs_current", "min"),
        gain_all12_max=("gain_all12_vs_current", "max"),
    ).reset_index()  # fmt: skip
    summ.to_csv(TAB / "founder_uncertainty_summary.csv", index=False)
    (TAB / "founder_uncertainty_summary.md").write_text(summ.round(3).to_markdown(index=False))
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())
    print(df.pivot(index="scenario", columns="truth", values="sd_BtoA_N512").round(3).to_string())


if __name__ == "__main__":
    main()
