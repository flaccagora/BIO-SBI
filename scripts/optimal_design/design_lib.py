"""Design criteria, exact convex solvers (cvxpy + Clarabel) and design evaluation.

A design is a weight vector ``w`` over the design points ``POINTS`` (A1..A12, B1..B12):
the share of the *budget* spent on each point. With a cost ``c_i`` per detected clone the
information per budget unit is ``sum_i w_i K_i / c_i``; with unit costs ``w`` is the share
of detected clones. Criteria (whitened K, so lam is in "clones per prior" units):

* E: maximise lam_min(K(w));
* c (B->A): minimise c^T K(w)^+ c (asymptotic variance of B->A per unit budget, x 1/42);
* D: maximise log det K(w).

All three are convex in ``w`` and solved as SDPs. Robust versions maximise the worst-case
(or average) efficiency over a set of truths, efficiencies relative to each truth's own
optimum: E-eff = lam_min(w) / lam_min*, c-eff = v* / v(w), D-eff = (det / det*)^(1/k).
"""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
except ImportError as e:  # pragma: no cover
    raise SystemExit("run with: uv run --with cvxpy python ...") from e

SOLVER = "CLARABEL"


# ------------------------------------------------------------------ evaluation (numpy)
def info(Ks: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.tensordot(w, Ks, axes=1)


def lam_min(M: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(0.5 * (M + M.T))[0])


def cvar(M: np.ndarray, c: np.ndarray) -> float:
    """c^T M^+ c (inf if c is not in the range of M)."""
    lam, V = np.linalg.eigh(0.5 * (M + M.T))
    a = V.T @ c
    tol = 1e-10 * max(lam[-1], 1e-300)
    if np.any((lam <= tol) & (np.abs(a) > 1e-8 * np.linalg.norm(c))):
        return float("inf")
    good = lam > tol
    return float(np.sum(a[good] ** 2 / lam[good]))


def logdet(M: np.ndarray) -> float:
    s, ld = np.linalg.slogdet(0.5 * (M + M.T))
    return float(ld) if s > 0 else -np.inf


def criteria(Ks: np.ndarray, c: np.ndarray, w: np.ndarray) -> dict:
    M = info(Ks, w)
    return {"lam_min": lam_min(M), "cvar": cvar(M, c), "logdet": logdet(M)}


# ------------------------------------------------------------------ cvxpy building blocks
def _sym(expr):
    return 0.5 * (expr + expr.T)


def _M(Ks: np.ndarray, w, scale: float = 1.0):
    return _sym(sum(w[i] * (Ks[i] * scale) for i in range(len(Ks))))


def _solve(prob) -> None:
    try:
        prob.solve(solver=SOLVER)
    except cp.error.SolverError:
        prob.solve(solver="SCS", eps=1e-9, max_iters=200000)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solver status {prob.status}")


def _clean(w: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    w = np.clip(np.asarray(w, float), 0.0, None)
    w[w < tol * w.max()] = 0.0
    return w / w.sum()


def _scale(Ks: np.ndarray) -> float:
    """Scalar making lam_min of the uniform design 1 (conditioning of the SDP)."""
    return 1.0 / max(lam_min(Ks.mean(0)), 1e-12)


def optimise(Ks: np.ndarray, c: np.ndarray, criterion: str, mask: np.ndarray | None = None,
             prior_n: float | None = None) -> np.ndarray:  # fmt: skip
    """Optimal budget shares for one truth. ``mask``: allowed points.

    ``prior_n``: Bayesian c-criterion c^T (I + n K(w))^-1 c (flat prior, n budget units).
    """
    n = len(Ks)
    w = cp.Variable(n, nonneg=True)
    cons = [cp.sum(w) == 1]
    if mask is not None:
        cons.append(w[~np.asarray(mask, bool)] == 0)
    s = _scale(Ks)
    M = _M(Ks, w, s)
    k = Ks.shape[1]
    if criterion == "E":
        t = cp.Variable()
        prob = cp.Problem(cp.Maximize(t), cons + [M - t * np.eye(k) >> 0])
    elif criterion == "c":
        if prior_n is not None:
            M = _sym(np.eye(k) + (prior_n / s) * M)
        prob = cp.Problem(cp.Minimize(cp.matrix_frac(c, M)), cons)
    elif criterion == "D":
        prob = cp.Problem(cp.Maximize(cp.log_det(M)), cons)
    else:
        raise ValueError(criterion)
    _solve(prob)
    return _clean(w.value)


def optimal_values(Ks: np.ndarray, c: np.ndarray) -> dict:
    """Per-criterion optimal designs and values for one truth."""
    out = {}
    for crit in ("E", "c", "D"):
        w = optimise(Ks, c, crit)
        out[crit] = {"w": w, **criteria(Ks, c, w)}
    return out


def efficiencies(Ks: np.ndarray, c: np.ndarray, w: np.ndarray, opt: dict) -> dict:
    v = criteria(Ks, c, w)
    k = Ks.shape[1]
    return {
        **v,
        "eff_E": v["lam_min"] / opt["E"]["lam_min"],
        "eff_c": opt["c"]["cvar"] / v["cvar"],
        "eff_D": float(np.exp((v["logdet"] - opt["D"]["logdet"]) / k)),
    }


def robust(Kset: list[np.ndarray], cset: list[np.ndarray], opts: list[dict], criterion: str,
           mode: str = "maximin", mask: np.ndarray | None = None,
           weights: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """Robust budget shares over several truths (efficiency-normalised).

    ``mode="maximin"``: maximise the worst-case efficiency. ``mode="mean"``: E - maximise
    the mean log efficiency; c - minimise the mean relative variance v/v*; D - maximise the
    mean log det (classical Bayesian D-optimality).
    """
    n = len(Kset[0])
    w = cp.Variable(n, nonneg=True)
    cons = [cp.sum(w) == 1]
    if mask is not None:
        cons.append(w[~np.asarray(mask, bool)] == 0)
    T = len(Kset)
    a = np.full(T, 1.0 / T) if weights is None else np.asarray(weights, float) / np.sum(weights)
    k = Kset[0].shape[1]
    terms = []
    for Ks, c, o in zip(Kset, cset, opts, strict=True):
        if criterion == "E":
            M = _M(Ks, w, 1.0 / o["E"]["lam_min"])  # efficiency scale: optimum = 1
            terms.append(M)
        elif criterion == "c":
            s = 1.0 / o["E"]["lam_min"]
            M = _M(Ks, w, s)
            terms.append(cp.matrix_frac(c, M) / (o["c"]["cvar"] / s))
        elif criterion == "D":
            ld = o["D"]["logdet"]
            terms.append((cp.log_det(_M(Ks, w, 1.0)) - ld) / k)
    if criterion == "E":
        if mode == "maximin":
            t = cp.Variable()
            prob = cp.Problem(cp.Maximize(t), cons + [M - t * np.eye(k) >> 0 for M in terms])
        else:
            ts = cp.Variable(T)
            cons += [M - ts[i] * np.eye(k) >> 0 for i, M in enumerate(terms)]
            prob = cp.Problem(cp.Maximize(a @ cp.log(ts)), cons)
    elif criterion == "c":
        if mode == "maximin":
            prob = cp.Problem(cp.Minimize(cp.maximum(*terms)), cons)
        else:
            prob = cp.Problem(cp.Minimize(sum(a[i] * terms[i] for i in range(T))), cons)
    else:
        if mode == "maximin":
            prob = cp.Problem(cp.Maximize(cp.minimum(*terms)), cons)
        else:
            prob = cp.Problem(cp.Maximize(sum(a[i] * terms[i] for i in range(T))), cons)
    _solve(prob)
    return _clean(w.value)


def equivalence_gap(Ks: np.ndarray, c: np.ndarray, w: np.ndarray, criterion: str) -> float:
    """General-equivalence-theorem check: max_i directional derivative ratio - 1 (>= 0;
    0 at the optimum). E: only exact for a simple lam_min (reported as is)."""
    M = info(Ks, w)
    k = M.shape[0]
    if criterion == "D":
        Mi = np.linalg.inv(M)
        d = np.array([np.trace(Mi @ K) for K in Ks])
        return float(d.max() / k - 1)
    if criterion == "c":
        Mi = np.linalg.pinv(M)
        u = Mi @ c
        d = np.array([u @ K @ u for K in Ks])
        return float(d.max() / (c @ u) - 1)
    lam, V = np.linalg.eigh(M)
    v = V[:, 0]
    d = np.array([v @ K @ v for K in Ks])
    return float(d.max() / lam[0] - 1)
