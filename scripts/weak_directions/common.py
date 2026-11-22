"""Shared helpers of the weak-direction (sloppiness) study of the exact two-type GW model.

Coordinates. The 12 outcome probabilities (A row: death, persist, to_B, div_AA, div_AB,
div_BB; B row: death, to_A, persist, div_AA, div_AB, div_BB - the order of the model's
``P[1]`` and of ``sim.parameter_names``) move in the 10-dim tangent space where each row's
changes sum to 0. Everything is stored as 12 x 12 matrices on that space (row-constant
vectors are in the null space), so any orthonormal basis of the tangent space gives the
same eigenvectors, read directly as loadings on the 12 named probabilities.

Information. ``I_h`` = expected Fisher information of ONE detected clone observed at
generation ``h`` (exact pgf/FFT law, :mod:`bio_sbi.inference.gw_fisher`), computed with
our own physical directions (orthonormal row contrasts) instead of stick coordinates, so
boundary truths (outcome probability 0) need no singular stick Jacobian: the directional
derivatives towards a zero outcome are the one-sided ones.

Prior scale. The flat Dirichlet(1,...,1) on a 6-outcome row has covariance
``(diag(m) - m m^T) / 7`` with ``m = 1/6``, i.e. ``P_0 / 42`` with ``P_0`` the projector on
the row's tangent space: isotropic. Prior whitening is therefore the Euclidean metric of
the probabilities times 42, and the generalised eigenproblem ``I v = lam Sigma^-1 v`` is
``eig(I / 42)`` on the tangent space.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")

import json  # noqa: E402
from dataclasses import replace  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path("/home/matteo/BIO-SBI/results/exact_gw")
XG2 = RESULTS / "data_size"
OUT = RESULTS / "weak_directions"
DOCS = REPO / "docs" / "weak_directions"
FIG = DOCS / "figures"
TAB = DOCS / "tables"

NAMES = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
         "B.death", "B.to_A", "B.persist", "B.div_AA", "B.div_AB", "B.div_BB"]  # fmt: skip
SHORT = ["A†", "A→A", "A→B", "A→AA", "A→AB", "A→BB",
         "B†", "B→A", "B→B", "B→AA", "B→AB", "B→BB"]  # fmt: skip
TRUTHS = ["T1_ref", "T2_toA05", "T3_Bdiv", "T4_slowA", "P1", "P2", "P3", "P4", "T5_C7"]
BOUNDARY = ["T1_ref", "T2_toA05", "T4_slowA"]  # B divisions exactly 0
HORIZONS = (4, 6, 8, 10)
ALL_GENS = tuple(range(1, 11))
PRIOR_PREC = 42.0  # flat Dirichlet on 6 outcomes: Sigma_prior = P_0 / 42


def truth(gt: str) -> np.ndarray:
    """Truth (12 probabilities, NAMES order) of an XG2 ground truth."""
    d = json.loads((XG2 / gt / "n512_s0" / "mle.json").read_text())
    return np.array([d["truth"][k] for k in NAMES])


def row_contrasts(active: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal basis ``(12, k)`` of the tangent space (each row's changes sum to 0).

    ``active`` (12 bools): restrict to vectors supported on the active outcomes (the face
    of the simplex where the inactive outcomes stay 0).
    """
    act = np.ones(12, bool) if active is None else np.asarray(active, bool)
    cols = []
    for r in (0, 1):
        idx = np.flatnonzero(act[6 * r : 6 * r + 6]) + 6 * r
        m = len(idx)
        if m < 2:
            continue
        M = np.zeros((12, m))
        M[idx, np.arange(m)] = 1.0
        M[idx] -= 1.0 / m  # centre within the active outcomes
        q, _ = np.linalg.qr(M)
        q = q[:, : m - 1]
        q[np.abs(q) < 1e-14] = 0.0
        cols.append(q)
    return np.hstack(cols)


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


def information_12(p12: np.ndarray, generations: int, founder: str = "A",
                   tail_tol: float = 1e-12) -> tuple[np.ndarray, dict]:  # fmt: skip
    """``I_h`` of one detected clone (12 x 12, tangent space) at probabilities ``p12``."""
    from bio_sbi.inference.gw_fisher import horizon_information

    model = gw_model()
    if founder == "B":
        model = replace(model, founder=np.array([0.0, 1.0]))
    U = row_contrasts()
    P = np.asarray(p12, float).reshape(2, 6)
    dP = np.zeros((10, 6))
    rows = np.zeros(10, np.int64)
    for d in range(10):
        r = 0 if np.linalg.norm(U[:6, d]) > 0.5 else 1
        rows[d] = r
        dP[d] = U[6 * r : 6 * r + 6, d]
    h = horizon_information(model, P, dP, rows, generations, tail_tol=tail_tol)
    I12 = U @ h.info @ U.T
    diag = {"detection": h.detection, "grid": h.grid, "tail": h.tail_mass,
            "omitted": h.omitted_mass}  # fmt: skip
    return 0.5 * (I12 + I12.T), diag


def info_cache(label: str, p12: np.ndarray, gens=ALL_GENS, founders=("A", "B"),
               force: bool = False) -> dict[tuple[str, int], np.ndarray]:  # fmt: skip
    """``{(founder, h): I_h}`` at ``p12``, cached in ``OUT / info_<label>.npz``."""
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"info_{label}.npz"
    have = {}
    if f.exists() and not force:
        z = np.load(f)
        if np.allclose(z["p12"], p12, atol=0, rtol=0):
            have = {(k.split("_")[0], int(k.split("_")[1])): z[k] for k in z.files
                    if k[0] in "AB" and k[1] == "_"}  # fmt: skip
    need = [(fo, g) for fo in founders for g in gens if (fo, g) not in have]
    diags = {}
    for fo, g in need:
        have[(fo, g)], diags[(fo, g)] = information_12(p12, g, fo)
    if need:
        np.savez_compressed(f, p12=p12, **{f"{fo}_{g}": m for (fo, g), m in have.items()})
        (OUT / f"info_{label}_diag.json").write_text(
            json.dumps({f"{fo}_{g}": d for (fo, g), d in diags.items()}, indent=1))
    return have


def eig_tangent(M12: np.ndarray, B: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ascending) and 12-dim unit eigenvectors of ``M12`` on span(``B``)."""
    B = row_contrasts() if B is None else B
    lam, c = np.linalg.eigh(B.T @ M12 @ B)
    V = B @ c
    # sign: largest |loading| positive
    s = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
    return lam, V * s


def label(v: np.ndarray, k: int = 4, tol: float = 0.15) -> str:
    """Compact signed list of the largest loadings, e.g. ``+B→A -B→BB -B†``."""
    o = np.argsort(-np.abs(v))[:k]
    return " ".join(f"{'+' if v[i] > 0 else '−'}{SHORT[i]}" for i in o if abs(v[i]) >= tol)
