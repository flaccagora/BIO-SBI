"""Validation of the founder-distribution Fisher information (``fisher.py``).

1. Single A founder, known (S1, p_A = 1): the transition block equals the optimal-design
   information of one A-founded clone (``od_common.information_basis``).
2. Size-1 founder mixture (S1, p_A = 0.7): the (transitions, p_A) block equals
   ``scripts/optimal_design/founder_uncertainty.mix_info`` (weight w = p_A).
3. Founder sizes 1..5: the score of a set of cells drawn from the law equals the exact
   log-likelihood gradient (``gw_exact.log_likelihood_founder``, both kernels) and central
   finite differences of its value, in all 15 coordinates.
4. Sum of the conditional law over the grid = 1 (minus the kept-mass tolerance), and the
   expected score is 0.

  uv run python scripts/founder_distribution/validate_fisher.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fd_common as F  # noqa: E402
import fisher as FI  # noqa: E402
from fd_common import od  # noqa: E402

from bio_sbi.inference.gw_exact import GridSpec, HorizonData, log_likelihood_founder  # noqa: E402

sys.path.insert(0, str(F.REPO / "scripts" / "optimal_design"))
import founder_uncertainty as fu  # noqa: E402


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def check_single(gt: str, h: int) -> dict:
    p12 = od.design_truth(gt)
    U = od.tangent_basis(None)
    L = FI.grid_for(p12, h)
    parts = FI.single_parts(p12, h, U, L)
    M, q, mass = FI.information(parts, F.SIZE_DISTS["S1"], 1.0)
    ref, qref = od.information_basis(p12, h, "A", U)
    # 2. size-1 mixture vs founder_uncertainty.mix_info
    Ls = fu.laws(p12, h, U)
    M7, _, _ = FI.information(parts, F.SIZE_DISTS["S1"], 0.7)
    mref = fu.mix_info(Ls, 0.7)
    return {"truth": gt, "h": h, "L": L, "rel_err_singleA": rel(M[:10, :10], ref),
            "detection_err": abs(q - qref), "kept_mass": mass,
            "rel_err_mix_S1_pA0.7": rel(M7[:11, :11], mref)}  # fmt: skip


def check_scores(gt: str, h: int, dist: str, seed: int = 0) -> dict:
    """Score of a sampled histogram vs the exact log-likelihood gradient and FD."""
    pi, pa = F.all_dists()[dist]
    p12 = od.design_truth(gt)
    U = od.tangent_basis(None)
    L = FI.grid_for(p12, h)
    parts = FI.single_parts(p12, h, U, L)
    w, s, q = FI.mixture_law(parts, pi, pa)
    # recover the kept cells' coordinates
    G = sum(pi[m - 1] * (pa * parts["FA"] + (1 - pa) * parts["FB"]) ** m for m in F.SIZES)
    p = FI._irfft(G, L)
    p[0, 0] = 0
    keep = (p / q > FI.THRESH) & (p > FI.FLOOR)
    ai, bi = np.nonzero(keep)
    rng = np.random.default_rng(seed)
    n = 300
    idx = rng.choice(len(w), size=n, p=w / w.sum())
    cnt = np.bincount(idx, minlength=len(w))
    sel = np.flatnonzero(cnt)
    score = s[:, sel] @ cnt[sel]  # analytic gradient of the histogram's log-likelihood
    hd = HorizonData(h, ai[sel], bi[sel], cnt[sel].astype(float), GridSpec(L, L, 1.0, 1.0))
    model = replace(od.gw_model(), founder=np.array([pa, 1 - pa]), sizes=F.SIZES.copy(),
                    size_probs=np.asarray(pi, float))  # fmt: skip
    P = p12.reshape(2, 6)
    dP, rows = FI.directions(U)
    out = {"truth": gt, "h": h, "dist": dist, "cells": int(len(sel))}
    for kernel in ("loop", "adjoint"):
        ll, gd, gphi, gpi = log_likelihood_founder(model, P, dP, rows, [hd], floor=0.0,
                                                   kernel=kernel)  # fmt: skip
        g = np.concatenate([gd, [gphi[0] - gphi[1]], F.V_PI.T @ gpi])
        out[f"rel_err_vs_{kernel}"] = rel(score, g)

    # central finite differences of the value in all 15 coordinates
    def value(x):
        p12x = p12 + U @ x[:10]
        pax = pa + x[10]
        pix = pi + F.V_PI @ x[11:]
        mod = replace(model, founder=np.array([pax, 1 - pax]), size_probs=pix)
        return log_likelihood_founder(mod, p12x.reshape(2, 6), dP[:0], rows[:0], [hd],
                                      floor=0.0, kernel="loop", grad=False)[0]  # fmt: skip

    eps = 1e-5
    fd = np.zeros(15)
    for d in range(15):
        e = np.zeros(15)
        e[d] = eps
        fd[d] = (value(e) - value(-e)) / (2 * eps)
    out["rel_err_vs_fd"] = rel(score, fd)
    out["mean_score_max"] = float(np.max(np.abs(s @ w)))
    out["kept_mass"] = float(w.sum())
    return out


def main() -> None:
    res = {"single": [], "scores": []}
    for gt, h in (("P3", 4), ("P3", 8), ("T1_ref", 6), ("T3_Bdiv", 10)):
        r = check_single(gt, h)
        print(r, flush=True)
        res["single"].append(r)
    # S2 / S4 interior; S3 has zero pi_4, pi_5 (boundary, one-sided directions still smooth)
    for gt, h, dist in (("P3", 4, "S4_pA0.5"), ("T1_ref", 6, "S2_pA0.7"),
                        ("T3_Bdiv", 4, "S3_pA0.9"), ("T5_C7", 2, "S4_pA0.3")):  # fmt: skip
        r = check_scores(gt, h, dist)
        print(r, flush=True)
        res["scores"].append(r)
    F.TAB.mkdir(parents=True, exist_ok=True)
    (F.TAB / "fisher_validation.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
