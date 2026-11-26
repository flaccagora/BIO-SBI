"""Shared helpers of the optimal-experimental-design study of the exact two-type GW model.

Design space. A design gives each design point (founder f in {A, B}, observation generation
h in 1..12) a share ``w_{f,h} >= 0`` of the clones, ``sum w = 1``. The information of one
clone under the design is ``K(w) = sum w_{f,h} K_{f,h}``, with ``K_{f,h}`` the expected
Fisher information of ONE DETECTED clone founded by one f cell and observed at generation
h (exact pgf/FFT law, :mod:`bio_sbi.inference.gw_fisher`, conditioned on detection as the
likelihood is), in the prior-whitened tangent coordinates of the weak-directions study:

* tangent space: the 10-dim space of the 12 outcome probabilities where each row's changes
  sum to 0 (orthonormal row contrasts ``U``, 12 x 10);
* prior whitening: the flat Dirichlet(1,...,1) row has covariance ``P_0 / 42``, so the
  whitened information is ``U^T I U / 42`` and its eigenvalue lam has 1/lam = the number of
  clones at which the data carry as much information as the prior along that direction.

Truths: the 9 XG2 truths (``reference.npz``). For the boundary truths T1/T2/T4 (B divisions
exactly 0) B-founded information is non-regular at the face (a B-founded clone larger than
2^(h-1) is impossible unless B divides), so the design numbers use the ``eps = 0.01``
variant (each zero set to 0.01, the rest of the row rescaled), as the weak-directions
study did. The exact boundary truths are also evaluated on the active face ("B never
divides" known, 7 dims), where B-founded clones are regular.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def _load_weak_directions_common():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "wd_common", REPO / "scripts" / "weak_directions" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wd_common"] = mod
    spec.loader.exec_module(mod)
    return mod


wd = _load_weak_directions_common()
NAMES, SHORT, gw_model, row_contrasts = wd.NAMES, wd.SHORT, wd.gw_model, wd.row_contrasts

RESULTS = Path("/home/matteo/BIO-SBI/results/exact_gw")
XG2 = RESULTS / "data_size"
OUT = RESULTS / "optimal_design"
CACHE = OUT / "info"
DOCS = REPO / "docs" / "optimal_design"
FIG = DOCS / "figures"
TAB = DOCS / "tables"

TRUTHS = ["T1_ref", "T2_toA05", "T3_Bdiv", "T4_slowA", "T5_C7", "P1", "P2", "P3", "P4"]
SHORTNAME = {"T1_ref": "T1", "T2_toA05": "T2", "T3_Bdiv": "T3", "T4_slowA": "T4",
             "T5_C7": "T5", "P1": "P1", "P2": "P2", "P3": "P3", "P4": "P4"}  # fmt: skip
BOUNDARY = ["T1_ref", "T2_toA05", "T4_slowA"]
EPS = 0.01
GENS = tuple(range(1, 13))
FOUNDERS = ("A", "B")
POINTS = [(f, h) for f in FOUNDERS for h in GENS]  # design points, A1..A12, B1..B12
PRIOR_PREC = 42.0
I_BTOA = NAMES.index("B.to_A")
OFF = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])  # daughters (A, B) per outcome


def truth_xg2(gt: str) -> np.ndarray:
    """Truth (12 probabilities, NAMES order) from the XG2 reference dataset."""
    z = np.load(XG2 / gt / "n1024_s0" / "reference.npz")
    names = [str(s) for s in z["truth_names"]]
    return np.array([float(z["truth"][names.index(k)]) for k in NAMES])


def eps_variant(p12: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Each zero outcome set to ``eps``, the rest of its row rescaled (row sums stay 1)."""
    q = np.asarray(p12, float).copy()
    for r in (0, 1):
        sl = slice(6 * r, 6 * r + 6)
        z = q[sl] == 0
        if z.any():
            row = q[sl]
            row[~z] *= 1 - eps * z.sum()
            row[z] = eps
            q[sl] = row
    return q


def design_truth(gt: str) -> np.ndarray:
    """Truth used for the design numbers (eps variant for the boundary truths)."""
    p = truth_xg2(gt)
    return eps_variant(p) if gt in BOUNDARY else p


def mean_matrix(p12: np.ndarray) -> np.ndarray:
    """Mean offspring matrix M[k, l] = E[# type-l daughters of a type-k cell]."""
    return np.asarray(p12, float).reshape(2, 6) @ OFF


def expected_cells(p12: np.ndarray, h: int, founder: str = "A") -> float:
    """Unconditional expected number of cells of a clone at generation ``h``."""
    e = np.array([1.0, 0.0]) if founder == "A" else np.array([0.0, 1.0])
    return float((e @ np.linalg.matrix_power(mean_matrix(p12), int(h))).sum())


def growth_rate(p12: np.ndarray) -> float:
    return float(np.max(np.linalg.eigvals(mean_matrix(p12)).real))


def tangent_basis(active: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal row contrasts (12, k); ``active`` restricts to a face."""
    return row_contrasts(active)


def information_basis(p12: np.ndarray, generations: int, founder: str, U: np.ndarray,
                      tail_tol: float = 1e-12) -> tuple[np.ndarray, float]:  # fmt: skip
    """Fisher information (k x k, in the coordinates of the basis ``U``) of one detected
    clone founded by one ``founder`` cell, observed at ``generations``; and P(detected)."""
    from bio_sbi.inference.gw_fisher import horizon_information

    model = gw_model()
    model = replace(model, founder=np.array([1.0, 0.0] if founder == "A" else [0.0, 1.0]))
    P = np.asarray(p12, float).reshape(2, 6)
    k = U.shape[1]
    dP = np.zeros((k, 6))
    rows = np.zeros(k, np.int64)
    for d in range(k):
        r = 0 if np.linalg.norm(U[:6, d]) > 0.5 else 1
        rows[d] = r
        dP[d] = U[6 * r : 6 * r + 6, d]
    h = horizon_information(model, P, dP, rows, generations, tail_tol=tail_tol)
    return 0.5 * (h.info + h.info.T), float(h.detection)


def compute_info(label: str, p12: np.ndarray, gens=GENS, active: np.ndarray | None = None,
                 force: bool = False) -> dict:  # fmt: skip
    """Per design point: whitened information K (k x k), detection q, mean detected size.

    Cached in ``CACHE / <label>.npz`` with arrays ``K (2, H, k, k)``, ``q (2, H)``,
    ``size (2, H)`` (expected cells of a detected clone), ``gens``, ``p12``, ``U``.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{label}.npz"
    if f.exists() and not force:
        z = np.load(f)
        if np.array_equal(z["p12"], p12) and tuple(z["gens"]) == tuple(gens):
            return dict(z)
    U = tangent_basis(active)
    k = U.shape[1]
    K = np.zeros((2, len(gens), k, k))
    q = np.zeros((2, len(gens)))
    size = np.zeros((2, len(gens)))
    for i, fo in enumerate(FOUNDERS):
        for j, g in enumerate(gens):
            info, det = information_basis(p12, g, fo, U)
            K[i, j] = info / PRIOR_PREC
            q[i, j] = det
            size[i, j] = expected_cells(p12, g, fo) / det
    out = {"K": K, "q": q, "size": size, "gens": np.array(gens), "p12": np.asarray(p12),
           "U": U}  # fmt: skip
    np.savez_compressed(f, **out)
    return out


def c_vector(U: np.ndarray, index: int = I_BTOA) -> np.ndarray:
    """``c = U^T e_index``: p_index = c . x for a tangent displacement p = p0 + U x.

    With K = U^T I U / 42 (whitened), Var(p_index) = c^T (n 42 K)^-1 c for n clones
    (:func:`var_functional`).
    """
    e = np.zeros(12)
    e[index] = 1.0
    return U.T @ e


def var_functional(K: np.ndarray, c: np.ndarray, n: float = 1.0,
                   prior: bool = False) -> float:  # fmt: skip
    """Variance of p_index = c . (tangent displacement) for n clones with whitened info K.

    Asymptotic (``prior=False``): c^T (n * 42 K)^-1 c. With the flat prior (Gaussian
    approximation, ``prior=True``): c^T (42 (I + n K))^-1 c.
    """
    k = K.shape[0]
    M = n * K + (np.eye(k) if prior else 0.0)
    try:
        return float(c @ np.linalg.solve(M, c)) / PRIOR_PREC
    except np.linalg.LinAlgError:
        return float("inf")
