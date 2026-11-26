"""Expected Fisher information of one detected clone with a founder (seeding) distribution.

For every truth (9 XG2 truths, eps variant for the boundary truths as in the optimal-design
study), generation h in {1, 2, 4, 6, 8, 10} and founder distribution (size distribution
S1..S4 x p_A in 1.0..0.1): the 15 x 15 information in (10 transition directions, p_A,
4 founder-size contrasts), see ``fd_common``. The single-founder pgfs F_A, F_B and their
transition derivatives are computed once per (truth, h) in forward mode on a square grid
large enough for 5 founders (``P(N >= L | detected) <= 1e-12`` for 5 A or 5 B founders);
every founder distribution is then a pointwise polynomial in F_A, F_B on the grid plus
inverse FFTs.

  uv run python scripts/founder_distribution/fisher.py --workers 4

Output: ``CACHE/<truth>.npz`` with ``info (n_dist, n_gen, 15, 15)``, ``q (n_dist, n_gen)``,
``mass`` (kept conditional mass), ``dists``, ``gens``, ``p12``, ``U``, ``grid``.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.fft

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fd_common as F  # noqa: E402
from fd_common import od  # noqa: E402

from bio_sbi.inference.gw_exact import (  # noqa: E402
    GridSpec,
    _loop_single_founder,
    detection_and_grad,
)  # fmt: skip
from bio_sbi.inference.gw_fisher import grid_size  # noqa: E402

THRESH, FLOOR = 1e-14, 1e-13


def directions(U: np.ndarray):
    k = U.shape[1]
    dP = np.zeros((k, 6))
    rows = np.zeros(k, np.int64)
    for d in range(k):
        r = 0 if np.linalg.norm(U[:6, d]) > 0.5 else 1
        rows[d] = r
        dP[d] = U[6 * r : 6 * r + 6, d]
    return dP, rows


def grid_for(p12: np.ndarray, h: int, max_size: int = 5, tail_tol: float = 1e-12) -> int:
    base = od.gw_model()
    P = np.asarray(p12, float).reshape(2, 6)
    return max(grid_size(replace(base, founder=np.array(f), sizes=np.array([max_size]),
                                 size_probs=np.ones(1)), P, h, tail_tol)[0]
               for f in ([1.0, 0.0], [0.0, 1.0]))  # fmt: skip


def single_parts(p12: np.ndarray, h: int, U: np.ndarray, L: int) -> dict:
    """F_A, F_B, dF_A, dF_B (half grid) and g_A, g_B, dg_A, dg_B for one (truth, h)."""
    base = od.gw_model()
    P = np.asarray(p12, float).reshape(2, 6)
    dP, rows = directions(U)
    FA, dFA, FB, dFB = _loop_single_founder(base, P, dP, rows, h, GridSpec(L, L, 1.0, 1.0))
    one = {"sizes": np.array([1]), "size_probs": np.ones(1)}
    gA, dgA = detection_and_grad(replace(base, founder=np.array([1.0, 0.0]), **one), P, dP, rows, h)
    gB, dgB = detection_and_grad(replace(base, founder=np.array([0.0, 1.0]), **one), P, dP, rows, h)
    return {"FA": FA, "FB": FB, "dFA": dFA, "dFB": dFB, "gA": float(gA), "gB": float(gB),
            "dgA": np.asarray(dgA), "dgB": np.asarray(dgB), "L": L}  # fmt: skip


def _irfft(v: np.ndarray, L: int) -> np.ndarray:
    return scipy.fft.irfft2(v, s=(L, L), workers=1)


def mixture_law(parts: dict, pi: np.ndarray, pa: float):
    """Conditional law, detection, and the 15 score arrays on the kept cells."""
    FA, FB, L = parts["FA"], parts["FB"], parts["L"]
    S = pa * FA + (1 - pa) * FB
    Spow = [np.ones_like(S)]
    for _ in range(5):
        Spow.append(Spow[-1] * S)
    G = sum(pi[m - 1] * Spow[m] for m in F.SIZES if pi[m - 1])
    Gp = sum(pi[m - 1] * m * Spow[m - 1] for m in F.SIZES if pi[m - 1])
    # detection
    s = pa * parts["gA"] + (1 - pa) * parts["gB"]
    ds_th = pa * parts["dgA"] + (1 - pa) * parts["dgB"]
    q = float(sum(pi[m - 1] * -np.expm1(m * np.log1p(-s)) for m in F.SIZES if pi[m - 1]))
    qp = float(sum(pi[m - 1] * m * (1 - s) ** (m - 1) for m in F.SIZES if pi[m - 1]))
    dq = np.concatenate([qp * ds_th, [qp * (parts["gA"] - parts["gB"])],
                         F.V_PI.T @ np.array([-np.expm1(m * np.log1p(-s)) for m in F.SIZES])])  # fmt: skip
    p = _irfft(G, L)
    p[0, 0] = 0.0
    pc = p / q
    keep = (pc > THRESH) & (p > FLOOR)
    pk = p[keep]
    scores = np.empty((15, pk.size))
    for d in range(F.K_TH):
        dG = Gp * (pa * parts["dFA"][d] + (1 - pa) * parts["dFB"][d])
        scores[d] = _irfft(dG, L)[keep]
    scores[F.I_PA] = _irfft(Gp * (FA - FB), L)[keep]
    for j in range(4):
        scores[11 + j] = _irfft(sum(F.V_PI[m - 1, j] * Spow[m] for m in F.SIZES), L)[keep]
    scores = scores / pk[None, :] - (dq / q)[:, None]
    return pk / q, scores, q


def information(parts: dict, pi: np.ndarray, pa: float) -> tuple[np.ndarray, float, float]:
    w, s, q = mixture_law(parts, pi, pa)
    M = (s * w[None, :]) @ s.T
    return 0.5 * (M + M.T), q, float(w.sum())


def truth_job(gt: str) -> str:
    out = F.CACHE / f"{gt}.npz"
    if out.exists():
        return f"{gt} cached"
    t0 = time.time()
    p12 = od.design_truth(gt)
    U = od.tangent_basis(None)
    dists = F.all_dists()
    nd, ng = len(dists), len(F.GENS)
    info = np.zeros((nd, ng, 15, 15))
    q = np.zeros((nd, ng))
    mass = np.zeros((nd, ng))
    grid = np.zeros(ng, int)
    for j, h in enumerate(F.GENS):
        L = grid_for(p12, h)
        grid[j] = L
        parts = single_parts(p12, h, U, L)
        for i, (pi, pa) in enumerate(dists.values()):
            info[i, j], q[i, j], mass[i, j] = information(parts, pi, pa)
        del parts
        print(f"{gt} h={h} L={L} {time.time() - t0:.0f}s", flush=True)
    F.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, info=info, q=q, mass=mass, dists=np.array(list(dists)),
                        gens=np.array(F.GENS), p12=p12, U=U, grid=grid)  # fmt: skip
    return f"{gt} done {time.time() - t0:.0f}s"


def load(gt: str) -> dict:
    z = np.load(F.CACHE / f"{gt}.npz")
    return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--truths", default=",".join(F.TRUTHS))
    args = ap.parse_args()
    gts = args.truths.split(",")
    # big grids first
    order = ["T5_C7", "P3", "T3_Bdiv", "T1_ref", "P4", "T2_toA05", "P2", "T4_slowA", "P1"]
    gts = [g for g in order if g in gts]
    with ProcessPoolExecutor(args.workers) as pool:
        for msg in pool.map(truth_job, gts):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
