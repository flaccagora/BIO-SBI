"""Shared helpers of the composition-convergence study of the exact two-type GW model.

Mean matrix ``M[k, l]`` = expected type-``l`` daughters of one type-``k`` cell per generation,
``M = [[a, b], [c, d]]`` (a: A daughters of an A cell, b: B daughters of an A cell, c: A
daughters of a B cell, d: B daughters of a B cell). ``rho`` = Perron root, ``lambda2`` = the
other eigenvalue (real for a 2x2 non-negative matrix), ``u`` = stable composition (left
Perron vector, ``u M = rho u``, ``u_A + u_B = 1``), ``r = |lambda2| / rho``.

Exact per-clone law: the pgf/FFT law of :mod:`bio_sbi.inference.gw_exact` (no damping),
for one A (or B) founder, conditioned on detection (``Z_h != 0``; perfect typing, capture 1
in the XG2 preset), on the smallest grid with ``P(N >= L | detected) <= 1e-10``
(:func:`bio_sbi.inference.gw_fisher.grid_size`).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = Path("/home/matteo/BIO-SBI/results/exact_gw")
XG2 = RESULTS / "data_size"
OUT = RESULTS / "convergence"
DOCS = REPO / "docs" / "convergence"
FIG = DOCS / "figures"
TAB = DOCS / "tables"

NAMES = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
         "B.death", "B.to_A", "B.persist", "B.div_AA", "B.div_AB", "B.div_BB"]  # fmt: skip
TRUTHS = ["T1_ref", "T2_toA05", "T3_Bdiv", "T4_slowA", "T5_C7", "P1", "P2", "P3", "P4"]
SHORTNAME = {"T1_ref": "T1", "T2_toA05": "T2", "T3_Bdiv": "T3", "T4_slowA": "T4",
             "T5_C7": "T5", "P1": "P1", "P2": "P2", "P3": "P3", "P4": "P4"}  # fmt: skip
# daughters (A, B) per outcome, NAMES order within each row (B row: death, to_A, persist, ...)
OFF = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2],
                [0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]], float)  # fmt: skip
PRIOR_SEED = 20261122  # same stream as scripts/optimal_design/compute_info.py prior_draws
MEAN_CELLS = (30.0, 300.0)  # expected cells of an A-founded clone at generation 10


def truth_xg2(gt: str) -> np.ndarray:
    """Truth (12 probabilities, NAMES order) from the XG2 reference dataset."""
    z = np.load(XG2 / gt / "n1024_s0" / "reference.npz", allow_pickle=True)
    names = [str(s) for s in z["truth_names"]]
    return np.array([float(z["truth"][names.index(k)]) for k in NAMES])


def mean_matrix(p12: np.ndarray) -> np.ndarray:
    p = np.asarray(p12, float)
    return np.vstack([p[:6] @ OFF[:6], p[6:] @ OFF[6:]])


def spectrum(M: np.ndarray) -> dict:
    """rho, lambda2, r, u_A and the coupling / growth-difference descriptors of ``M``."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    tr, det = a + d, a * d - b * c
    disc = np.sqrt(max((a - d) ** 2 + 4 * b * c, 0.0))
    rho, l2 = 0.5 * (tr + disc), 0.5 * (tr - disc)
    # left Perron vector: u_A (a - rho) + u_B c = 0
    if b * c > 0:
        uA = c / (c + rho - a)
    elif c == 0 and b == 0:
        uA = 1.0 if a > d else (0.0 if d > a else 0.5)
    elif c == 0:  # B never makes A: rho = max(a, d)
        uA = 0.0 if d >= a else (a - d) / (a - d + b)
    else:  # b == 0: A never makes B
        uA = 1.0 if a >= d else c / (c + d - a)
    r = abs(l2) / rho if rho > 0 else np.nan
    return {"a": a, "b": b, "c": c, "d": d, "rho": rho, "lambda2": l2, "r": r, "u_A": uA,
            "coupling": b * c, "a_minus_d": a - d, "det": det,
            "regime": "CLT" if l2**2 < rho else ("critical" if l2**2 == rho else "large")}  # fmt: skip


def pooled_fraction(M: np.ndarray, hs, founder: str = "A") -> np.ndarray:
    """Pooled A fraction E[A_h] / E[N_h] (= ratio of means; detection cancels)."""
    e = np.array([1.0, 0.0]) if founder == "A" else np.array([0.0, 1.0])
    out, v, h0 = [], e.copy(), 0
    for h in hs:
        for _ in range(int(h) - h0):
            v = v @ M
            v = v / v.sum()  # renormalise: only the composition is needed
        h0 = int(h)
        out.append(v[0])
    return np.array(out)


def n_mean(M: np.ndarray, eps: float, founder: str = "A", hmax: int = 2000) -> float:
    """Smallest N with |pooled(h) - u_A| < eps for every h >= N (within ``hmax``; inf else).

    Beyond ``hmax`` the gap is non-increasing in the geometric regime (checked on the tail).
    """
    s = spectrum(M)
    u = s["u_A"]
    if s["rho"] - s["lambda2"] > 1e-6 and s["rho"] > 0:  # exact eigen formula (vectorised)
        C, D = gap_amplitude(M, founder)
        h = np.arange(0, hmax + 1)
        sg = np.sign(s["lambda2"]) ** h
        with np.errstate(under="ignore"):
            rh = s["r"] ** h
        f = u + C * sg * rh / (1 + D * sg * rh)
    else:
        f = pooled_fraction(M, range(0, hmax + 1), founder)
    bad = np.flatnonzero(np.abs(f - u) >= eps)
    if len(bad) == 0:
        return 0.0
    if bad[-1] >= hmax:
        return np.inf
    return float(bad[-1] + 1)


def gap_amplitude(M: np.ndarray, founder: str = "A") -> tuple[float, float]:
    """``(C, D)`` of the exact pooled gap  f_h - u_A = C s^h r^h / (1 + D s^h r^h)
    (``s`` = sign of lambda2): ``e M^h = alpha rho^h u + beta lambda2^h w`` with ``u``, ``w``
    the left eigenvectors (``u`` summing to 1), ``C = beta (w_A - u_A (w_A + w_B)) / alpha``,
    ``D = beta (w_A + w_B) / alpha``. Requires distinct eigenvalues.
    """
    w, V = np.linalg.eig(M.T)  # columns: left eigenvectors of M
    w, V = w.real, V.real
    o = np.argsort(-w)
    V = V[:, o]
    u = V[:, 0] / V[:, 0].sum()
    w2 = V[:, 1]
    e = np.array([1.0, 0.0]) if founder == "A" else np.array([0.0, 1.0])
    al, be = np.linalg.solve(np.column_stack([u, w2]), e)
    return float(be * (w2[0] - u[0] * w2.sum()) / al), float(be * w2.sum() / al)


def n_mean_theory(M: np.ndarray, eps: float, founder: str = "A") -> float:
    """N_eps ~ log(|C| / eps) / log(1 / r) (leading term of :func:`gap_amplitude`)."""
    s = spectrum(M)
    if not np.isfinite(s["r"]) or s["r"] >= 1:
        return np.nan
    if s["r"] == 0:
        return 1.0
    C, _ = gap_amplitude(M, founder)
    if abs(C) <= eps:
        return 0.0
    return float(np.log(abs(C) / eps) / np.log(1 / s["r"]))


def expected_cells(p12: np.ndarray, h: int, founder: str = "A") -> float:
    e = np.array([1.0, 0.0]) if founder == "A" else np.array([0.0, 1.0])
    return float((e @ np.linalg.matrix_power(mean_matrix(p12), int(h))).sum())


def prior_draws(n: int, seed: int = PRIOR_SEED) -> tuple[np.ndarray, int]:
    """``n`` flat-prior draws with 30-300 expected cells at generation 10 (XG2 filter)."""
    rng = np.random.default_rng(seed)
    keep, tried = [], 0
    while len(keep) < n:
        p = rng.dirichlet(np.ones(6), size=2).ravel()
        tried += 1
        m = expected_cells(p, 10, "A")
        if MEAN_CELLS[0] <= m <= MEAN_CELLS[1]:
            keep.append(p)
    return np.array(keep), tried


# ---------------------------------------------------------------------- exact law
@lru_cache(maxsize=1)
def gw_model():
    """GWModel of the XG2 preset (two_type, perfect typing, detected clones, A founder)."""
    from hydra import compose, initialize_config_module

    from bio_sbi.inference.exact_gw import ExactLikelihood
    from bio_sbi.pipeline import build_components
    from bio_sbi.summaries import Cohort

    with initialize_config_module("bio_sbi.conf", version_base=None):
        cfg = compose("config", overrides=["+experiment=exact_gw_data_size", "device=cpu"])
    comp = build_components(cfg)
    lik = ExactLikelihood(comp.simulator, comp.generator.observation,
                          [Cohort(4.0, np.array([[1, 0]]))])  # fmt: skip
    names = list(comp.simulator.parameter_names)[:12]
    assert names == NAMES, names
    return lik.model


def clone_law(p12: np.ndarray, h: int, founder: str = "A", tail_tol: float = 1e-10,
              max_grid: int = 6000) -> tuple[np.ndarray | None, dict]:  # fmt: skip
    """Exact law ``p[A, B]`` of one clone at generation ``h`` conditioned on detection.

    Returns ``(None, diag)`` if the grid would exceed ``max_grid``.
    """
    from bio_sbi.inference.gw_exact import GridSpec, law_and_jac
    from bio_sbi.inference.gw_fisher import grid_size

    model = gw_model()
    model = replace(model, founder=np.array([1.0, 0.0] if founder == "A" else [0.0, 1.0]))
    P = np.asarray(p12, float).reshape(2, 6)
    L, tail = grid_size(model, P, h, tail_tol)
    diag = {"grid": int(L), "tail": float(tail)}
    if L > max_grid:
        return None, diag
    p, _ = law_and_jac(model, P, np.zeros((0, 6)), np.zeros(0, np.int64), h,
                       GridSpec(L, L, 1.0, 1.0))  # fmt: skip
    q = 1.0 - p[0, 0]
    p[0, 0] = 0.0
    p = np.clip(p, 0.0, None) / q
    diag["detection"] = float(q)
    diag["mass"] = float(p.sum())
    return p, diag


def composition_stats(p: np.ndarray) -> dict:
    """Per-clone composition summaries of a conditional law ``p[A, B]`` (p[0, 0] = 0)."""
    L = p.shape[0]
    A = np.arange(L)[:, None]
    B = np.arange(p.shape[1])[None, :]
    N = A + B
    m = p.sum()
    p = p / m
    with np.errstate(invalid="ignore", divide="ignore"):
        X = np.where(N > 0, A / np.maximum(N, 1), 0.0)
    EA, EN = float((p * A).sum()), float((p * N).sum())
    EX = float((p * X).sum())
    sd = float(np.sqrt(max((p * X**2).sum() - EX**2, 0.0)))
    # quantiles of X: sort the support by X
    keep = p > 1e-15
    xs, ws = X[keep], p[keep]
    o = np.argsort(xs, kind="stable")
    xs, cw = xs[o], np.cumsum(ws[o])
    q10 = float(xs[np.searchsorted(cw, 0.1 * cw[-1])])
    q90 = float(xs[min(np.searchsorted(cw, 0.9 * cw[-1]), len(xs) - 1)])
    return {"pooled": EA / EN, "mean_frac": EX, "sd_frac": sd, "q10": q10, "q90": q90,
            "p_allA": float(p[:, 0].sum()), "p_allB": float(p[0, :].sum()),
            "mean_cells": EN, "mass": float(m)}  # fmt: skip


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float))


sys.path.insert(0, str(HERE))
