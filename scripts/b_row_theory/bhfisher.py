"""Fisher information of the exact Bellman-Harris endpoint law at the BH reference truth.

Design: real-like {144 h: 165 clones, 240 h: 101 clones}, detected clones (law of
(A, B) conditioned on A + B > 0), one A founder, Gamma shape 8, A mean 20 h, B 30 h.

Coordinates: probabilities as "drop death" free coordinates (row k: every free entry
except death; death = 1 - sum), i.e. tangent directions e_j - e_death of the 12-vector
(the gwexact.tangent_basis convention); lifetimes as log mean_A, log mean_B.
  reversible_ab: A {persist, to_B, div_AA, div_AB, div_BB}, B {persist, to_A}  + 2 = 9
  two_type:      A {...5...}, B {persist, to_A, div_AA, div_AB, div_BB}       + 2 = 12
Variances of linear functions of the physical parameters (single entries, B-division
mass) do not depend on which entry is dropped; eigenvalues / condition numbers do, so
both the raw and the correlation-scaled (unit-diagonal) condition numbers are given.

Usage: nice -n 10 uv run python scripts/b_row_theory/bhfisher.py [LA LB]
"""
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import json  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bhexact as bx  # noqa: E402
from gwexact import NAMES, THETA_STAR, conditional_fisher  # noqa: E402

MEANS = np.array([20.0, 30.0])
SHAPE = 8
DESIGN = {144: 165, 240: 101}
NTOT = sum(DESIGN.values())


def fd_check(L=48, horizons=(48, 96), eps=1e-5):
    P = THETA_STAR
    p, dp = bx.law_bh_jac(P, MEANS, SHAPE, list(horizons), L)
    errs = []
    for d in range(14):
        if d < 12:
            e = np.zeros(12); e[d] = eps
            hi = bx.law_bh(P + e, MEANS, SHAPE, list(horizons), L)
            lo = bx.law_bh(P - e, MEANS, SHAPE, list(horizons), L)
        else:
            k = d - 12; up = MEANS.copy(); dn = MEANS.copy()
            up[k] *= np.exp(eps); dn[k] *= np.exp(-eps)
            hi = bx.law_bh(P, up, SHAPE, list(horizons), L)
            lo = bx.law_bh(P, dn, SHAPE, list(horizons), L)
        fd = (hi - lo) / (2 * eps)
        errs.append(dict(dir=(NAMES + ["log_mean_A", "log_mean_B"])[d],
                         max_abs_err=float(np.abs(fd - dp[:, d]).max()),
                         max_abs_deriv=float(np.abs(dp[:, d]).max())))
    return errs


def basis(law):
    """Rows: directions in the 14-dim (12 P entries + 2 log-means) space; names."""
    free = {"A": ["A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB"],
            "B": ["B.persist", "B.to_A"] + (["B.div_AA", "B.div_AB", "B.div_BB"]
                                           if law == "two_type" else [])}
    rows, names = [], []
    for k, row in (("A", 0), ("B", 1)):
        for nm in free[k]:
            v = np.zeros(14); v[NAMES.index(nm)] = 1; v[6 * row] = -1
            rows.append(v); names.append(nm)
    for k in range(2):
        v = np.zeros(14); v[12 + k] = 1; rows.append(v); names.append(("A", "B")[k] + ".log_mean")
    return np.array(rows), names


def summarise(I_tot, names, law):
    ev = np.linalg.eigvalsh(I_tot)
    w, V = np.linalg.eigh(I_tot)
    dsc = 1 / np.sqrt(np.diag(I_tot))
    R = I_tot * dsc[:, None] * dsc[None, :]
    evR = np.linalg.eigvalsh(R)
    C = np.linalg.inv(I_tot)
    sd = lambda c: float(np.sqrt(c @ C @ c))  # noqa: E731
    out = dict(law=law, coords=names, eig_min=float(ev[0]), eig_max=float(ev[-1]),
               cond=float(ev[-1] / ev[0]), eig_min_scaled=float(evR[0]),
               cond_scaled=float(evR[-1] / evR[0]),
               eigvec_min={n: round(float(x), 4) for n, x in zip(names, V[:, 0])},
               cr_sd_all={n: float(np.sqrt(C[i, i])) for i, n in enumerate(names)})
    targets = {}
    e = lambda n: np.eye(len(names))[names.index(n)]  # noqa: E731
    targets["A.div_AA"] = e("A.div_AA")
    targets["A.log_mean_lifetime"] = e("A.log_mean")
    targets["B.to_A"] = e("B.to_A")
    if law == "two_type":
        targets["B.division_mass"] = e("B.div_AA") + e("B.div_AB") + e("B.div_BB")
    out["cr"] = {}
    for k, c in targets.items():
        s = sd(c)
        out["cr"][k] = dict(sd=s, sd_per_clone=s * np.sqrt(NTOT),
                            n_for_sd_0_05=float(NTOT * (s / 0.05) ** 2))
    out["cr"]["A.mean_lifetime_hours_sd"] = out["cr"]["A.log_mean_lifetime"]["sd"] * MEANS[0]
    return out


def main(LA=768, LB=512):
    res = {}
    t = time.time(); res["fd_check"] = fd_check(); res["fd_seconds"] = time.time() - t
    print(json.dumps(res["fd_check"]), flush=True)
    t = time.time()
    p, dp = bx.law_bh_jac(THETA_STAR, MEANS, SHAPE, list(DESIGN), (LA, LB))
    res["jac_seconds"] = time.time() - t
    res["L"] = [LA, LB]
    for law in ("reversible_ab", "two_type"):
        Bm, names = basis(law)
        I_tot = 0
        per = {}
        for i, (h, n) in enumerate(DESIGN.items()):
            g = np.tensordot(Bm, dp[i], axes=(1, 0))
            I, missed = conditional_fisher(p[i], g)
            per[h] = dict(missed_conditional_mass=float(missed))
            I_tot = I_tot + n * I
        res[law] = summarise(I_tot, names, law)
        res[law]["per_horizon"] = per
        res[law]["I_total"] = I_tot.tolist()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(int(x) for x in a)) if a else main()
