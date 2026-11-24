"""Founder (seeding) distribution study of the exact two-type GW model: shared code.

A clone is seeded by ``m`` cells, ``P(m) = pi_m`` (m = 1..5), each founder cell independently
A with probability ``p_A`` (B otherwise); only endpoint (A, B) counts are observed. The clone
pgf is

    G(x, y) = sum_m pi_m S^m,   S = p_A F_A + (1 - p_A) F_B,

with ``F_k`` the pgf of a clone seeded by one type-k cell (the m-fold convolution of the
single-founder law is ``S^m`` in the FFT domain). Derivatives:

    dG/dtheta = G'(S) (p_A dF_A + (1 - p_A) dF_B),  G'(S) = sum_m pi_m m S^(m-1),
    dG/dp_A   = G'(S) (F_A - F_B),                  dG/dpi_m = S^m;

detection ``q = sum_m pi_m (1 - (1 - s)^m)``, ``s = p_A g_A + (1 - p_A) g_B`` (``g_k`` =
P(a type-k cell leaves a detected descendant)) is mixed the same way.

Coordinates: the 10 transition directions of the optimal-design study (orthonormal row
contrasts ``U``, prior whitening by 42), then ``p_A`` (1), then the founder-size simplex
(orthonormal contrasts ``V`` of the 5 sizes, 4 dims). The information of one detected clone
is the 15 x 15 matrix ``sum_c p~_c s_c s_c^T`` (``s_c = dp_c / p_c - dq / q``).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "optimal_design"))
import od_common as od  # noqa: E402,F401

RESULTS = Path("/home/matteo/BIO-SBI/results/exact_gw/founder_distribution")
CACHE = RESULTS / "fisher"
DOCS = REPO / "docs" / "founder_distribution"
FIG = DOCS / "figures"
TAB = DOCS / "tables"

TRUTHS = od.TRUTHS
SHORTNAME = od.SHORTNAME
SIZES = np.arange(1, 6)
SIZE_DISTS = {
    "S1": np.array([1.0, 0, 0, 0, 0]),  # all single
    "S2": np.array([0.7, 0.2, 0.07, 0.02, 0.01]),  # mostly single
    "S3": np.array([1 / 3, 1 / 3, 1 / 3, 0, 0]),  # uniform 1-3
    "S4": np.full(5, 0.2),  # uniform 1-5
}
SIZE_LABEL = {"S1": "S1 all single", "S2": "S2 mostly single", "S3": "S3 uniform 1-3",
              "S4": "S4 uniform 1-5"}  # fmt: skip
PAS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1)
GENS = (1, 2, 4, 6, 8, 10)
DESIGNS = {
    "XG2": {4: 0.25, 6: 0.25, 8: 0.25, 10: 0.25},  # A-founded XG2 endpoints
    "R1t": {1: 0.25, 2: 0.625, 10: 0.125},  # R1's time points, founders unlabelled
}
N_TOTAL = 2048  # 512 clones per XG2 endpoint; same total for the R1 time points
K_TH = 10  # transition directions
I_PA = 10  # index of p_A
I_PI = slice(11, 15)  # founder-size contrasts


def size_contrasts() -> np.ndarray:
    """Orthonormal contrasts (5, 4) of the founder-size simplex."""
    M = np.eye(5) - 1.0 / 5
    q, _ = np.linalg.qr(M)
    return q[:, :4]


V_PI = size_contrasts()


def dist_key(s: str, pa: float) -> str:
    return f"{s}_pA{pa:.1f}"


def all_dists() -> dict[str, tuple[np.ndarray, float]]:
    return {dist_key(s, pa): (pi, pa) for s, pi in SIZE_DISTS.items() for pa in PAS}


# ---------------------------------------------------------------------- functionals
def functionals(p12: np.ndarray) -> dict:
    """B->A, m_BA, growth rate rho, stable A fraction (as the misspecification study)."""
    M = np.asarray(p12, float).reshape(2, 6) @ od.OFF
    w, Vl = np.linalg.eig(M.T)
    i = int(np.argmax(w.real))
    v = np.abs(Vl[:, i].real)
    return {"B.to_A": float(p12[7]), "m_BA": float(M[1, 0]), "rho": float(w[i].real),
            "stableA": float(v[0] / v.sum())}  # fmt: skip


FUNCS = ["B.to_A", "m_BA", "rho", "stableA"]


def functional_gradients(p12: np.ndarray, U: np.ndarray, h: float = 1e-6) -> dict:
    """``c_f = U^T grad f`` (tangent coordinates) for every functional (central differences)."""
    out = {}
    for f in FUNCS:
        g = np.zeros(12)
        for i in range(12):
            e = np.zeros(12)
            e[i] = h
            g[i] = (functionals(p12 + e)[f] - functionals(p12 - e)[f]) / (2 * h)
        out[f] = U.T @ g
    return out


# ---------------------------------------------------------------------- information algebra
def schur(M: np.ndarray, keep: np.ndarray, nuis: np.ndarray) -> np.ndarray:
    """Information on ``keep`` with ``nuis`` profiled out (Schur complement, pseudo-inverse)."""
    A = M[np.ix_(keep, keep)]
    if len(nuis) == 0:
        return A
    B = M[np.ix_(keep, nuis)]
    C = M[np.ix_(nuis, nuis)]
    return A - B @ np.linalg.pinv(C, rcond=1e-12, hermitian=True) @ B.T


def lam_min(K_th: np.ndarray) -> float:
    """Weakest prior-whitened eigenvalue (K_th = transition information / 42)."""
    return float(np.linalg.eigvalsh(0.5 * (K_th + K_th.T))[0])


def sd_functional(I_th: np.ndarray, c: np.ndarray, n: float = N_TOTAL) -> float:
    """Asymptotic sd of ``c . x`` for n clones with (unwhitened) transition information."""
    try:
        v = float(c @ np.linalg.solve(n * I_th, c))
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.sqrt(v)) if v > 0 else float("inf")
