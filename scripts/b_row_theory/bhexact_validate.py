"""Validate bhexact.law_bh against the repo's Bellman-Harris forward kernel.

Usage: nice -n 10 uv run python scripts/b_row_theory/bhexact_validate.py [n_mc]
Prints a JSON summary (all numbers quoted in bhexact_validation.md).
"""
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "2")

import json  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bhexact as bx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = [96, 144, 240]
MEANS = [20.0, 30.0]
SHAPE = 8
L = (1024, 512)  # (A, B) table size; tails checked below


def bh_sim(law="reversible_ab", shape=8.0):
    from bio_sbi.simulators import BellmanHarris

    spec = yaml.safe_load(open(os.path.join(ROOT, "src/bio_sbi/conf/law", f"{law}.yaml")))
    bh = BellmanHarris(spec, lifetime_shape=shape, lifetime_mean_bounds=[10, 80])
    phys = bh.physical_from_mapping({**spec["reference"], "A.mean_lifetime": MEANS[0],
                                     "B.mean_lifetime": MEANS[1]})
    Prepo = bh.law.unpack(bh.space, phys)[0]
    return bh, phys, bx.from_repo(Prepo)


def chi2_pooled(counts, p, n, min_exp=5.0):
    e = p * n
    keep = e >= min_exp
    obs = np.r_[counts[keep], counts[~keep].sum()]
    exp = np.r_[e[keep], e[~keep].sum()]
    if exp[-1] < min_exp:  # merge pooled remainder into the smallest kept bin
        obs, exp = obs[:-1].copy(), exp[:-1].copy()
        i = np.argmin(exp); obs[i] += counts[~keep].sum(); exp[i] += e[~keep].sum()
    stat = float(((obs - exp) ** 2 / exp).sum())
    dof = len(obs) - 1
    return stat, dof, float(stats.chi2.sf(stat, dof))


def main(n_mc=200_000):
    res = {}
    bh, phys, P = bh_sim()
    res["P_gwexact_order"] = dict(zip(bx.NAMES, P.tolist()))

    # warm-up compile
    bx.law_bh(P, MEANS, SHAPE, H, 16)

    # (a) table, normalisation, negative mass, tail mass
    t = time.time(); p = bx.law_bh(P, MEANS, SHAPE, H, L); t_law3 = time.time() - t
    pa, pb = bx.marginal_tails(P, MEANS, SHAPE, H, N=8192)
    a = {}
    for i, h in enumerate(H):
        tailA = float(1 - pa[i, : L[0]].sum()); tailB = float(1 - pb[i, : L[1]].sum())
        a[h] = dict(sum_minus_1=float(p[i].sum() - 1), min=float(p[i].min()),
                    neg_mass=float(-p[i][p[i] < 0].sum()), tail_A_ge_LA=tailA,
                    tail_B_ge_LB=tailB, p00=float(p[i, 0, 0]))
    res["a"] = a

    # (b), (c) Monte Carlo from the repo kernel
    b, c = {}, {}
    rng = np.random.default_rng(20261109)
    for i, h in enumerate(H):
        t = time.time(); cnt = bh.simulate(phys, h, n_mc, rng); t_mc = time.time() - t
        assert cnt.min() >= 0, "invalid clones"
        ex = p[i]
        aa = np.arange(L[0])[:, None]; bb = np.arange(L[1])[None, :]
        mA, mB = float((aa * ex).sum()), float((bb * ex).sum())
        mc = cnt.mean(0); se = cnt.std(0) / np.sqrt(n_mc)
        b[h] = dict(exact=[mA, mB], mc=mc.tolist(), se=se.tolist(),
                    z=((mc - [mA, mB]) / se).tolist(), mc_seconds=t_mc,
                    exact_p_extinct=float(ex[0, 0]), mc_p_extinct=float((cnt.sum(1) == 0).mean()))
        # empirical table
        assert cnt[:, 0].max() < L[0] and cnt[:, 1].max() < L[1]
        emp = np.zeros(L)
        np.add.at(emp, (cnt[:, 0], cnt[:, 1]), 1)
        exn = np.clip(ex, 0, None); exn /= exn.sum()
        tv = 0.5 * np.abs(emp / n_mc - exn).sum()
        # parametric bootstrap of TV under the exact law, same n
        flat = exn.ravel(); boot = []
        brng = np.random.default_rng(1)
        for _ in range(200):
            draw = brng.multinomial(n_mc, flat)
            boot.append(0.5 * np.abs(draw / n_mc - flat).sum())
        boot = np.array(boot)
        chi = chi2_pooled(emp.ravel(), flat, n_mc)
        c[h] = dict(tv=float(tv), tv_boot_mean=float(boot.mean()),
                    tv_boot_q95=float(np.quantile(boot, 0.95)),
                    tv_boot_pvalue=float((boot >= tv).mean()),
                    chi2=chi[0], chi2_dof=chi[1], chi2_p=chi[2])
        if h == 96:
            c["cells_96"] = {f"P({x},{y})": dict(exact=float(ex[x, y]), mc=float(emp[x, y] / n_mc),
                                                se=float(np.sqrt(ex[x, y] * (1 - ex[x, y]) / n_mc)))
                             for x, y in ((1, 0), (0, 1), (2, 0))}
    res["b"], res["c"] = b, c

    # (d) shape 1 means vs closed form (expected_counts)
    bh1, phys1, P1 = bh_sim(shape=1.0)
    qa, qb = bx.marginal_tails(P1, MEANS, 1, H, N=16384)
    d = {}
    for i, h in enumerate(H):
        n = np.arange(qa.shape[1])
        ex = [float((n * qa[i]).sum()), float((n * qb[i]).sum())]
        cf = bh1.expected_counts(phys1, h)
        d[h] = dict(stages=ex, closed_form=cf.tolist(),
                    rel_err=(np.abs(np.array(ex) / cf - 1)).tolist(),
                    tail_A_ge_N=float(1 - qa[i].sum() + qa[i, -1]),
                    tailA_last1000=float(qa[i, -1000:].sum()), tailB_last1000=float(qb[i, -1000:].sum()))
    res["d"] = d

    # (e) step-size convergence on the 2-D tables
    p_half = bx.law_bh(P, MEANS, SHAPE, H, L, dt=bx.DEFAULT_DT / 2)
    p_dbl = bx.law_bh(P, MEANS, SHAPE, H, L, dt=bx.DEFAULT_DT * 2)
    res["e"] = dict(dt=bx.DEFAULT_DT,
                    max_abs_change_halving=[float(np.abs(p_half[i] - p[i]).max()) for i in range(3)],
                    max_abs_change_doubling=[float(np.abs(p_dbl[i] - p[i]).max()) for i in range(3)])

    # (f) timing
    f = {"three_horizons_L1024x512": t_law3}
    for LL in ((1024, 512), (768, 512), (512, 512)):
        t = time.time(); bx.law_bh(P, MEANS, SHAPE, 240, LL); f[f"240h_L{LL[0]}x{LL[1]}"] = time.time() - t
    t = time.time(); bx.law_bh(P, MEANS, SHAPE, [144, 240], (1024, 512)); f["144+240_L1024x512"] = time.time() - t
    res["f"] = f
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200_000)
