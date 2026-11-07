"""Profile KL over the B-division mass delta, with exact gradients.

K(delta) = min_{theta: B-division mass = delta} sum_n w_n KL(p_n(theta*) || p_n(theta)),
p_n the law of Z_n conditioned on Z_n != 0. All 12 tangent directions are propagated
in one vectorised forward-mode pass; L-BFGS on a softmax parameterisation.

Usage: uv run python scripts/b_row_theory/profile.py [workers]
Writes profile.jsonl next to this script.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gwexact import OFF, THETA_STAR, grid, law  # noqa: E402

Ls = {2: 5, 3: 9, 4: 17, 6: 65, 8: 257, 10: 384}
DESIGNS = {
    "E0large_4_6_10": {4: 1 / 3, 6: 1 / 3, 10: 1 / 3},   # per-clone average, equal clones
    "reallike_6_10": {6: 165 / 266, 10: 101 / 266},
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile.jsonl")


def law_and_jac(P, n, L):
    """p (L, L) and dp/dP_i (12, L, L) for the unconstrained polynomial extension."""
    x, y = grid(L)
    FA, FB = x.copy(), y.copy()
    dA = np.zeros((12,) + x.shape, complex); dB = np.zeros_like(dA)
    for _ in range(n):
        mono = {k: [FA**a * FB**b for (a, b) in OFF[k]] for k in (0, 1)}
        fx = {k: sum(P[6*k+o] * a * FA**max(a-1, 0) * FB**b for o, (a, b) in enumerate(OFF[k]) if a) for k in (0, 1)}
        fy = {k: sum(P[6*k+o] * b * FA**a * FB**max(b-1, 0) for o, (a, b) in enumerate(OFF[k]) if b) for k in (0, 1)}
        nA = fx[0] * dA + fy[0] * dB
        nB = fx[1] * dA + fy[1] * dB
        for o in range(6):
            nA[o] += mono[0][o]
            nB[6 + o] += mono[1][o]
        FA, FB = sum(P[o] * mono[0][o] for o in range(6)), sum(P[6+o] * mono[1][o] for o in range(6))
        dA, dB = nA, nB
    p = np.real(np.fft.fft2(FA)) / L**2
    dp = np.real(np.fft.fft2(dA, axes=(1, 2))) / L**2
    return p, dp


STAR = {}


def kl_and_grad(P, des):
    kl, g = 0.0, np.zeros(12)
    for n, w in des.items():
        q_star = STAR[n]
        p, dp = law_and_jac(P, n, Ls[n])
        p0 = p[0, 0]
        m = q_star > 1e-13
        # FFT roundoff is ~1e-17; floor at 1e-15 so the score q*/p cannot explode on
        # cells where p(theta) is numerically zero (they then cost KL as if p = 1e-15).
        pz = np.clip(p[m], 1e-15, None)
        q = pz / (1 - p0)
        kl += w * np.sum(q_star[m] * (np.log(q_star[m]) - np.log(q)))
        # d log q_z = dp_z / p_z + dp0 / (1 - p0)
        g += w * -(dp[:, m] @ (q_star[m] / pz) + dp[:, 0, 0] / (1 - p0) * q_star[m].sum())
    return kl, g


def softmax(v):
    e = np.exp(v - v.max()); return e / e.sum()


def unpack(z, delta):
    return np.concatenate([softmax(z[:6]), softmax(z[6:9]) * (1 - delta), softmax(z[9:12]) * delta])


def objective(z, delta, des):
    P = unpack(z, delta)
    kl, gP = kl_and_grad(P, des)
    gz = np.zeros(12)
    for sl, scale in ((slice(0, 6), 1.0), (slice(6, 9), 1 - delta), (slice(9, 12), delta)):
        s = P[sl] / scale if scale > 0 else softmax(z[sl])
        gs = gP[sl] * scale
        gz[sl] = s * (gs - s @ gs)          # softmax Jacobian-vector product
    return kl, gz


def starts():
    lg = lambda v: np.log(np.clip(np.asarray(v, float), 1e-3, None))
    return [
        np.concatenate([lg(THETA_STAR[:6]), lg([.1, .1, .8]), lg([1, 1, 1])]),
        np.concatenate([lg([.09, .15, .02, .47, .06, .21]), lg([.15, .04, .32]), lg([.21, .11, .16])]),
        np.concatenate([lg(THETA_STAR[:6]), lg([.1, .1, .8]), lg([1, .2, .2])]),
    ]


def run(job):
    dname, delta = job
    des = DESIGNS[dname]
    for n in des:
        q = law(THETA_STAR, n, Ls[n]).copy(); q[0, 0] = 0; STAR[n] = q / q.sum()
    best = None
    for z0 in starts():
        r = minimize(objective, z0, args=(delta, des), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=500, gtol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    P = unpack(best.x, delta)
    return dict(design=dname, delta=delta, kl_per_clone=float(best.fun), nit=int(best.nit),
                theta=dict(zip(["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
                                "B.death", "B.persist", "B.to_A", "B.div_AA", "B.div_AB", "B.div_BB"],
                               P.round(4).tolist())))


if __name__ == "__main__":
    # gradient check
    des = DESIGNS["E0large_4_6_10"]
    for n in des:
        q = law(THETA_STAR, n, Ls[n]).copy(); q[0, 0] = 0; STAR[n] = q / q.sum()
    z = starts()[1]; f0, g0 = objective(z, 0.3, des); e = np.zeros(12); e[8] = 1e-6
    print("grad check:", g0[8], (objective(z + e, 0.3, des)[0] - objective(z - e, 0.3, des)[0]) / 2e-6, flush=True)
    jobs = [(d, dl) for d in DESIGNS for dl in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with Pool(workers) as pool, open(OUT, "w") as fh:
        for res in pool.imap_unordered(run, jobs):
            fh.write(json.dumps(res) + "\n"); fh.flush()
            print(json.dumps(res), flush=True)
