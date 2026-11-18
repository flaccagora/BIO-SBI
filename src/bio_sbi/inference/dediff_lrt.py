"""Likelihood-ratio test for global dedifferentiation (B -> A) in the two-type GW model.

Two cell types, A (stem-like) and B (differentiated); each generation every cell has one
of six outcomes (death, persist, switch, div_AA, div_AB, div_BB). The question is whether
B cells can give rise to A cells by any route:

* **H0** (no dedifferentiation): ``B.to_A = B.div_AA = B.div_AB = 0``;
* **H1**: at least one of them is positive. Effect size
  ``m_BA = B.to_A + 2 B.div_AA + B.div_AB`` (expected A cells made by one B cell per
  generation, an entry of the mean offspring matrix).

Statistic: ``Lambda = 2 [l(theta_full) - l(theta_0)]`` with both maxima of the *exact*
likelihood (:class:`~bio_sbi.inference.exact_gw.ExactLikelihood`) of the endpoint counts,
conditioned on detection. One likelihood object (one histogram and grid per horizon) serves
both models: each model is a :class:`Parametrisation` mapping box coordinates in [0, 1]
to the outcome matrix ``P`` and its derivatives, evaluated through
:meth:`ExactLikelihood.log_likelihood_P` (exact gradient).

* H0: A row free (6 outcomes, 5 sticks), B row free on {death, persist, div_BB}
  (2 sticks), the three dedifferentiation entries exactly 0 (the same model as
  ``law=two_type '++law.fixed.B={to_A:0,div_AA:0,div_AB:0}'``).
* Full: A row as above; B row in *grouped* sticks: first ``q`` = total dedifferentiation
  mass (H0 is exactly the face ``q = 0``), then the composition of the dedifferentiation
  group (to_A, div_AA, div_AB) and of the rest (death, persist, div_BB). The full fit is
  always started from theta_0 (q = 0 is a candidate, so ``Lambda >= 0``), from theta_0
  with a small q in several compositions, from fits with B.to_A pinned at low / high
  values (then released; plain multistart in stick coordinates misses low-B.to_A optima,
  docs/xg2_mle_vs_posterior), and from random (flat Dirichlet) starts.

Calibration helpers: :func:`local_information` (observed information at theta_0 in
local probability coordinates) and :func:`chibar_null` (Monte-Carlo asymptotic null of
Lambda: difference of the projections of N(0, I^-1) onto the H0 and H1 cones, with
nuisance entries estimated at 0 treated as boundary in both), :func:`chibar_pvalue` for
mixtures of chi-squares, and :func:`simulate_counts` for (bootstrap) datasets.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from scipy.optimize import minimize

from ..observation import ObservationModel
from ..rng import rng_for
from ..summaries import Cohort

TYPES = ("A", "B")
#: Outcome entries of each row, in the law's (= ``P`` column) order.
ENTRIES = {
    "A": ("death", "persist", "to_B", "div_AA", "div_AB", "div_BB"),
    "B": ("death", "to_A", "persist", "div_AA", "div_AB", "div_BB"),
}
#: B outcomes that produce A cells, and the ones allowed under H0.
DEDIFF = ("to_A", "div_AA", "div_AB")
KEEP = ("death", "persist", "div_BB")
#: A cells produced per B event of each dedifferentiation route (m_BA weights).
M_WEIGHT = {"to_A": 1.0, "div_AA": 2.0, "div_AB": 1.0}
B_DEDIFF = [ENTRIES["B"].index(e) for e in DEDIFF]
B_KEEP = [ENTRIES["B"].index(e) for e in KEEP]


# ---------------------------------------------------------------------- model / data
def make_simulator():
    """Two-type Galton-Watson simulator of ``law=two_type`` (single A founder)."""
    from ..simulators import GaltonWatson

    conf = Path(__file__).resolve().parents[1] / "conf" / "law" / "two_type.yaml"
    return GaltonWatson(yaml.safe_load(conf.read_text()))


def confusion_matrix(rate: float = 0.0, kind: str = "symmetric") -> np.ndarray:
    """Return the true type (rows A, B) -> label (columns A, B) probabilities.

    ``kind``: ``symmetric`` (each type mislabelled with probability ``rate``),
    ``b_as_a`` (only B cells are read as A, the direction that mimics
    dedifferentiation) or ``a_as_b``.
    """
    r = float(rate)
    if kind == "symmetric":
        return np.array([[1 - r, r], [r, 1 - r]])
    if kind == "b_as_a":
        return np.array([[1.0, 0.0], [r, 1 - r]])
    if kind == "a_as_b":
        return np.array([[1 - r, r], [0.0, 1.0]])
    raise ValueError(f"unknown mis-typing kind {kind!r} (symmetric | b_as_a | a_as_b)")


def observation(confusion: np.ndarray | None = None) -> ObservationModel:
    """Perfect-capture, detected-clone observation model with a 2x2 ``confusion``."""
    conf = None if confusion is None else np.asarray(confusion, dtype=float).tolist()
    return ObservationModel(types=TYPES, confusion=conf, selection="detected")


def P_from_rows(A: dict, B: dict) -> np.ndarray:
    """``(2, 6)`` outcome matrix from ``{entry: prob}`` per row (omitted entries 0)."""
    P = np.zeros((2, 6))
    for k, (t, row) in enumerate((("A", A), ("B", B))):
        unknown = set(row) - set(ENTRIES[t])
        if unknown:
            raise ValueError(f"unknown {t} entries {sorted(unknown)}")
        for e, v in row.items():
            P[k, ENTRIES[t].index(e)] = float(v)
    if np.any(P < 0) or not np.allclose(P.sum(1), 1.0, atol=1e-9):
        raise ValueError(f"rows must be probability vectors; sums {P.sum(1)}")
    return P


def P_to_dict(P: np.ndarray) -> dict[str, float]:
    """``{"A.death": ..., "B.to_A": ...}`` of an outcome matrix."""
    return {f"{t}.{e}": float(P[k, i]) for k, t in enumerate(TYPES)
            for i, e in enumerate(ENTRIES[t])}  # fmt: skip


def physical(sim, P: np.ndarray) -> np.ndarray:
    """Return the simulator physical vector of ``P`` (founder rows filled in)."""
    return sim.physical_from_mapping(P_to_dict(P))


def m_ba(P: np.ndarray) -> float:
    """Return the expected A cells produced by one B cell per generation."""
    return float(sum(M_WEIGHT[e] * P[1, ENTRIES["B"].index(e)] for e in DEDIFF))


def simulate_counts(sim, P: np.ndarray, horizons, n, seed: int, key: tuple,
                    confusion: np.ndarray | None = None) -> list[Cohort]:  # fmt: skip
    """Simulate one dataset of ``n`` detected clones per horizon (one A founder each).

    ``n`` is an int or one value per horizon.

    True counts come from the exact detection-conditioned sampler with stream
    ``rng_for(seed, *key, "cohort", i)`` (``key = ("reference",)`` reproduces the
    pipeline's reference dataset for ``data_seed = seed``); labels from a separate stream
    ``rng_for(seed, *key, "typing", i)``, so a mis-typed dataset has exactly the clones of
    the perfectly typed one. Clones are i.i.d. and generated in order, so the first ``m``
    rows of every cohort are the dataset with ``m`` clones (nested in ``n``).
    """
    phys = physical(sim, P)
    obs = observation(confusion)
    out = []
    ns = np.broadcast_to(np.asarray(n, dtype=int), (len(horizons),))
    for i, h in enumerate(horizons):
        true = sim.simulate_detected(phys, h, int(ns[i]), np.ones(2),
                                     rng_for(seed, *key, "cohort", i))  # fmt: skip
        if np.any(true < 0):
            raise RuntimeError(f"simulation limit hit at horizon {h}")
        lab = obs.relabel(true, rng_for(seed, *key, "typing", i))
        out.append(Cohort(float(h), lab.astype(np.int64)))
    return out


def subset(cohorts: list[Cohort], n: int) -> list[Cohort]:
    """Return the first ``n`` clones of every cohort."""
    return [Cohort(c.horizon, c.counts[:n]) for c in cohorts]


def likelihood(sim, cohorts: list[Cohort], confusion: np.ndarray | None = None,
               kernel: str = "adjoint"):  # fmt: skip
    """Exact likelihood of ``cohorts`` under the full two-type model (any confusion)."""
    from .exact_gw import ExactLikelihood

    return ExactLikelihood(sim, observation(confusion), cohorts, kernel=kernel)


# ---------------------------------------------------------------------- parametrisation
def _simplex(s: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Stick-breaking ``p (m,)`` from ``m - 1`` sticks and ``dp/ds (m - 1, m)`` (exact)."""
    m = len(s) + 1
    p = np.empty(m)
    tail = 1.0
    for i in range(m):
        p[i] = mass * tail * (s[i] if i < m - 1 else 1.0)
        if i < m - 1:
            tail *= 1.0 - s[i]
    d = np.zeros((m - 1, m))
    for j in range(m - 1):
        for i in range(j, m):
            others = np.prod([1.0 - s[k] for k in range(i) if k != j])
            si = s[i] if i < m - 1 else 1.0
            d[j, i] = mass * others * (1.0 if i == j else -si)
    return p, d


def _sticks_of(p: np.ndarray) -> np.ndarray:
    """Sticks of a non-negative vector (normalised); empty tails give 0.5."""
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    tot = p.sum()
    if tot <= 0:
        return np.full(len(p) - 1, 0.5)
    p = p / tot
    rem = np.cumsum(p[::-1])[::-1]
    out = np.full(len(p) - 1, 0.5)
    ok = rem[:-1] > 1e-300
    out[ok] = np.clip(p[:-1][ok] / rem[:-1][ok], 0.0, 1.0)
    return out


@dataclass
class RowSpec:
    """One outcome row in grouped stick coordinates.

    The row's free mass ``1 - sum(fixed)`` is first stick-broken across ``groups`` (lists
    of entry indices), then within each group. Entries in no group and not fixed are 0.
    """

    k: int
    groups: list[list[int]]
    fixed: dict[int, float] = field(default_factory=dict)

    @property
    def mass(self) -> float:
        """Free mass of the row (1 - fixed entries)."""
        return 1.0 - sum(self.fixed.values())

    @property
    def dim(self) -> int:
        """Number of box coordinates of the row."""
        return len(self.groups) - 1 + sum(len(g) - 1 for g in self.groups)


class Parametrisation:
    """Box coordinates ``s`` in [0, 1]^dim -> outcome matrix ``P (2, 6)`` and derivatives."""

    def __init__(self, rows: list[RowSpec], name: str = "") -> None:
        self.rows = rows
        self.name = name
        self.dim = sum(r.dim for r in rows)

    def P(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``P``, ``dP (dim, 6)`` and the row index of each coordinate."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        P = np.zeros((2, 6))
        dP = np.zeros((self.dim, 6))
        rows = np.zeros(self.dim, dtype=np.int64)
        o = 0
        for r in self.rows:
            for i, v in r.fixed.items():
                P[r.k, i] = v
            ng = len(r.groups)
            G, dG = _simplex(s[o : o + ng - 1], r.mass)
            go, o2 = o, o + ng - 1
            rows[o : o + r.dim] = r.k
            for j, grp in enumerate(r.groups):
                w, dw = _simplex(s[o2 : o2 + len(grp) - 1], 1.0)
                P[r.k, grp] = G[j] * w
                for a in range(ng - 1):
                    dP[go + a, grp] += dG[a, j] * w
                dP[o2 : o2 + len(grp) - 1][:, grp] = G[j] * dw
                o2 += len(grp) - 1
            o += r.dim
        return P, dP, rows

    def encode(self, P: np.ndarray, composition: dict[int, np.ndarray] | None = None,
               ) -> np.ndarray:  # fmt: skip
        """Return the coordinates of ``P``.

        Entries outside the model are dropped and the row renormalised.
        ``composition``: ``{group id: weights}`` used for groups whose mass is 0 (group
        id = position over all rows, in order).
        """
        out, gid = [], 0
        for r in self.rows:
            row = np.clip(np.asarray(P[r.k], dtype=float), 0, None)
            masses = np.array([row[g].sum() for g in r.groups])
            out.extend(_sticks_of(masses) if len(r.groups) > 1 else [])
            for g in r.groups:
                comp = row[g]
                if comp.sum() <= 1e-300 and composition and gid in composition:
                    comp = np.asarray(composition[gid], dtype=float)
                out.extend(_sticks_of(comp))
                gid += 1
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


def h0_model() -> Parametrisation:
    """H0: A free; B on {death, persist, div_BB}, dedifferentiation entries fixed at 0."""
    return Parametrisation([RowSpec(0, [list(range(6))]), RowSpec(1, [B_KEEP])], "H0")


def full_model() -> Parametrisation:
    """Full model; B row as (q = dedifferentiation mass, its composition, the rest)."""
    return Parametrisation([RowSpec(0, [list(range(6))]), RowSpec(1, [B_DEDIFF, B_KEEP])],
                           "full")  # fmt: skip


def pinned_model(to_a: float) -> Parametrisation:
    """Full model with ``B.to_A`` fixed at ``to_a`` (other entries free)."""
    i = ENTRIES["B"].index("to_A")
    other = [ENTRIES["B"].index(e) for e in ("div_AA", "div_AB")]
    return Parametrisation([RowSpec(0, [list(range(6))]),
                            RowSpec(1, [other, B_KEEP], {i: float(to_a)})],
                           f"pinned_to_A={to_a:g}")  # fmt: skip


# ---------------------------------------------------------------------- optimisation
OPTIONS = {"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 20}


def loglik(lik, par: Parametrisation, s: np.ndarray, grad: bool = True):
    """Exact log likelihood (and gradient in ``s``) of ``lik`` under ``par``."""
    P, dP, rows = par.P(s)
    if not grad:
        return lik.log_likelihood_P(P, dP[:0], rows[:0], grad=False)[0], None
    return lik.log_likelihood_P(P, dP, rows, grad=True)


def maximise(lik, par: Parametrisation, s0: np.ndarray, options: dict | None = None) -> dict:
    """L-BFGS-B on the box [0, 1]^dim from ``s0``."""

    def neg(s):
        v, g = loglik(lik, par, s)
        if not np.isfinite(v):
            return 1e300, np.zeros_like(s)
        return -v, -g

    e0 = lik.evals
    r = minimize(neg, np.clip(s0, 0, 1), jac=True, method="L-BFGS-B",
                 bounds=[(0.0, 1.0)] * par.dim, options={**OPTIONS, **(options or {})})  # fmt: skip
    s = np.clip(r.x, 0.0, 1.0)
    return {"ll": float(-r.fun), "s": s, "P": par.P(s)[0], "nit": int(r.nit),
            "evals": int(lik.evals - e0), "success": bool(r.success),
            "message": str(r.message)}  # fmt: skip


def random_P(rng: np.random.Generator, h0: bool) -> np.ndarray:
    """Flat-Dirichlet random outcome matrix (B row on KEEP only under ``h0``)."""
    P = np.zeros((2, 6))
    P[0] = rng.dirichlet(np.ones(6))
    if h0:
        P[1, B_KEEP] = rng.dirichlet(np.ones(3))
    else:
        P[1] = rng.dirichlet(np.ones(6))
    return P


#: Generic (not data-specific) H0 start: A self-renews and differentiates, B mostly persists.
GENERIC = P_from_rows(
    {"death": 0.05, "persist": 0.05, "to_B": 0.05, "div_AA": 0.45, "div_AB": 0.1, "div_BB": 0.3},
    {"death": 0.2, "persist": 0.6, "div_BB": 0.2},
)

COMPOSITIONS = {
    "to_A": np.array([1.0, 0.0, 0.0]),
    "div_AA": np.array([0.0, 1.0, 0.0]),
    "div_AB": np.array([0.0, 0.0, 1.0]),
    "mixed": np.array([4.0, 1.0, 2.0]) / 7.0,
}


def with_dediff(P0: np.ndarray, q: float, comp: np.ndarray) -> np.ndarray:
    """``P0``'s B row scaled by ``1 - q`` plus mass ``q`` on DEDIFF with ``comp``."""
    P = P0.copy()
    P[1, B_KEEP] = P0[1, B_KEEP] / P0[1, B_KEEP].sum() * (1 - q)
    P[1, B_DEDIFF] = q * np.asarray(comp) / np.sum(comp)
    return P


def with_to_a(P: np.ndarray, c: float) -> np.ndarray:
    """``P`` with ``B.to_A = c`` and the other B entries rescaled to ``1 - c``."""
    P = P.copy()
    i = ENTRIES["B"].index("to_A")
    rest = [j for j in range(6) if j != i]
    P[1, rest] = np.clip(P[1, rest], 1e-6, None)
    P[1, rest] *= (1 - c) / P[1, rest].sum()
    P[1, i] = c
    return P


@dataclass
class FitSettings:
    """Starts of the two fits (see module docstring)."""

    h0_random: int = 4  # flat-Dirichlet starts for H0 (+ the generic start)
    full_random: int = 2  # flat-Dirichlet starts for the full model
    q_starts: tuple = (0.01, 0.1)  # theta_0 + q in every composition
    compositions: tuple = ("to_A", "div_AA", "div_AB", "mixed")
    pins: tuple = (0.02, 0.3)  # B.to_A pinned, fitted, then released
    tie: float = 1e-3  # nats: optima within this of the best agree
    options: dict = field(default_factory=dict)


def _best(results: dict) -> tuple[str, dict]:
    name = max(results, key=lambda k: results[k]["ll"])
    return name, results[name]


def lrt(lik, rng: np.random.Generator, settings: FitSettings | None = None,
        warm: dict | None = None) -> dict:  # fmt: skip
    """Fit H0 and the full model to ``lik``'s data; return Lambda and diagnostics.

    ``warm``: optional extra starts ``{"h0": [P, ...], "full": [P, ...]}`` (e.g. the
    MLEs of the same clones at a smaller n, or the generating value of a bootstrap
    replicate). Never the truth of a simulation study.
    """
    st = settings or FitSettings()
    warm = warm or {}
    t0 = time.perf_counter()
    e0 = lik.evals
    H0, FULL = h0_model(), full_model()

    # ---- H0
    starts0 = {"generic": GENERIC}
    for i in range(st.h0_random):
        starts0[f"random_{i}"] = random_P(rng, h0=True)
    for i, P in enumerate(warm.get("h0", [])):
        starts0[f"warm_{i}"] = P
    res0 = {k: maximise(lik, H0, H0.encode(P), st.options) for k, P in starts0.items()}
    name0, best0 = _best(res0)

    # ---- full model
    P0 = best0["P"]
    resF = {"theta0": {"ll": best0["ll"], "s": FULL.encode(P0), "P": P0, "nit": 0,
                       "evals": 0, "success": True, "message": "H0 optimum (q = 0)"}}  # fmt: skip
    resF["theta0_opt"] = maximise(lik, FULL, FULL.encode(P0, {1: np.ones(3)}),
                                  st.options)  # fmt: skip
    for q in st.q_starts:
        for c in st.compositions:
            resF[f"q{q:g}_{c}"] = maximise(lik, FULL, FULL.encode(with_dediff(P0, q, COMPOSITIONS[c])),
                                           st.options)  # fmt: skip
    for c in st.pins:
        pin = pinned_model(c)
        rp = maximise(lik, pin, pin.encode(with_to_a(P0, c)), st.options)
        resF[f"pin{c:g}"] = maximise(lik, FULL, FULL.encode(rp["P"]), st.options)
        resF[f"pin{c:g}"]["pinned_ll"] = rp["ll"]
    for i in range(st.full_random):
        resF[f"random_{i}"] = maximise(lik, FULL, FULL.encode(random_P(rng, h0=False)), st.options)
    for i, P in enumerate(warm.get("full", [])):
        resF[f"warm_{i}"] = maximise(lik, FULL, FULL.encode(P, {1: np.ones(3)}),
                                     st.options)  # fmt: skip
    nameF, bestF = _best(resF)

    # A full-model optimum with no dedifferentiation mass is an H0 point: if it beats
    # the H0 fit, the H0 fit missed its optimum -> use it (keeps Lambda honest).
    h0_improved = 0.0
    for r in resF.values():
        if r["P"][1, B_DEDIFF].sum() < 1e-10 and r["ll"] > best0["ll"] + 1e-9:
            P = r["P"].copy()
            P[1, B_DEDIFF] = 0.0
            P[1] /= P[1].sum()
            s = H0.encode(P)
            ll = float(loglik(lik, H0, s, grad=False)[0])
            if ll > best0["ll"] + 1e-9:
                h0_improved = max(h0_improved, ll - best0["ll"])
                best0 = {**r, "ll": ll, "s": s, "P": H0.P(s)[0]}
                name0 = "from_full"
    lam = max(2.0 * (bestF["ll"] - best0["ll"]), 0.0)
    P0, P1 = best0["P"], bestF["P"]
    return {
        "lambda": lam,
        "ll0": best0["ll"], "ll1": bestF["ll"],
        "theta0": P_to_dict(P0), "theta1": P_to_dict(P1),
        "P0": P0, "P1": P1,
        "m_ba_hat": m_ba(P1), "q_hat": float(P1[1, B_DEDIFF].sum()),
        "best_start0": name0, "best_start1": nameF,
        "h0_improved_by_full": h0_improved,
        "h0_starts_tied": int(sum(r["ll"] > best0["ll"] - st.tie for r in res0.values())),
        "full_starts_tied": int(sum(r["ll"] > bestF["ll"] - st.tie for r in resF.values())),
        "h0_ll": {k: r["ll"] for k, r in res0.items()},
        "full_ll": {k: r["ll"] for k, r in resF.items()},
        "full_q": {k: float(r["P"][1, B_DEDIFF].sum()) for k, r in resF.items()},
        "converged0": bool(res0[name0]["success"]) if name0 in res0 else True,
        "converged1": bool(bestF["success"]),
        "evals": int(lik.evals - e0),
        "seconds": time.perf_counter() - t0,
    }  # fmt: skip


# ---------------------------------------------------------------------- asymptotics
@dataclass
class LocalInfo:
    """Observed information at ``P0`` in local coordinates.

    Coordinates: per row, every entry but a reference one (largest entry; for B among
    KEEP), moving mass to / from the reference. ``boundary``: coordinates with
    ``P0 < zero`` (dedifferentiation entries always).
    """

    info: np.ndarray
    names: list[str]
    dediff: list[int]
    nuisance_boundary: list[int]
    min_eig: float


def local_information(lik, P0: np.ndarray, h: float = 1e-5, zero: float = 1e-7) -> LocalInfo:
    """Observed information ``-d2 l`` at ``P0`` by differences of the exact gradient.

    One-sided (second-order forward) differences along coordinates whose entry is at 0,
    central differences otherwise.
    """
    dirs, names, rows = [], [], []
    for k, t in enumerate(TYPES):
        cand = B_KEEP if t == "B" else list(range(6))
        ref = max(cand, key=lambda i: P0[k, i])
        for i in range(6):
            if i == ref:
                continue
            d = np.zeros(6)
            d[i], d[ref] = 1.0, -1.0
            dirs.append(d)
            rows.append(k)
            names.append(f"{t}.{ENTRIES[t][i]}")
    D = np.array(dirs)
    R = np.array(rows, dtype=np.int64)

    def grad(P):
        return lik.log_likelihood_P(P, D, R, grad=True)[1]

    def shifted(c, x):
        P = P0.copy()
        P[R[c]] += x * D[c]
        return P

    n = len(D)
    H = np.zeros((n, n))
    g0 = None
    for c in range(n):
        i_up = int(np.argmax(D[c]))
        i_dn = int(np.argmin(D[c]))
        if P0[R[c], i_up] > 2 * h and P0[R[c], i_dn] > 2 * h:
            H[:, c] = (grad(shifted(c, h)) - grad(shifted(c, -h))) / (2 * h)
        else:
            g0 = grad(P0) if g0 is None else g0
            H[:, c] = (-3 * g0 + 4 * grad(shifted(c, h)) - grad(shifted(c, 2 * h))) / (2 * h)
    info = -0.5 * (H + H.T)
    dd = [names.index(f"B.{e}") for e in DEDIFF]
    nb = [c for c in range(n) if c not in dd and P0[R[c], int(np.argmax(D[c]))] < zero]
    return LocalInfo(info, names, dd, nb, float(np.linalg.eigvalsh(info).min()))


def _cone_distance(Z: np.ndarray, J: np.ndarray, eq: list[int], ineq: list[int]):
    """``min (Z - t)' J (Z - t)`` over ``t_eq = 0``, ``t_ineq >= 0``, other t free.

    ``Z (N, d)``. Exact, by enumerating the faces of the cone (2^len(ineq)). Returns the
    distances and, per draw, the boolean mask of ``ineq`` coordinates that are positive at
    the projection.
    """
    N, d = Z.shape
    best = np.full(N, np.inf)
    pos = np.zeros((N, len(ineq)), dtype=bool)
    for active in itertools.product((False, True), repeat=len(ineq)):
        zero = list(eq) + [c for c, a in zip(ineq, active, strict=True) if a]
        free = [c for c in range(d) if c not in zero]
        if not zero:
            val = np.zeros(N)
            t_free = Z
        else:
            z0 = Z[:, zero]
            if free:
                Juu = J[np.ix_(free, free)]
                Ju0 = J[np.ix_(free, zero)]
                K = np.linalg.solve(Juu, Ju0)  # (|free|, |zero|)
                S = J[np.ix_(zero, zero)] - Ju0.T @ K
                t_free = Z[:, free] + z0 @ K.T
            else:
                S = J[np.ix_(zero, zero)]
                t_free = np.zeros((N, 0))
            val = np.einsum("ni,ij,nj->n", z0, S, z0)
        tf = dict(zip(free, range(len(free)), strict=True))
        ok = np.ones(N, dtype=bool)
        for c, a in zip(ineq, active, strict=True):
            if not a:
                ok &= t_free[:, tf[c]] >= -1e-12
        upd = ok & (val < best)
        best[upd] = val[upd]
        mask = np.array([not a for a in active])
        pos[upd] = mask
    return best, pos


def chibar_null(local: LocalInfo, n_draws: int = 20000, seed: int = 0,
                nuisance_boundary: bool = True) -> dict:  # fmt: skip
    """Monte-Carlo asymptotic null distribution of Lambda at ``local``.

    Lambda ~ d(Z, C0) - d(Z, C1), ``Z ~ N(0, J^-1)`` with ``J`` the efficient
    information of the boundary coordinates (dedifferentiation entries, plus nuisance
    entries estimated at 0 when ``nuisance_boundary``); the interior nuisance is
    profiled out. C1: dedifferentiation >= 0; C0: dedifferentiation = 0; boundary
    nuisance >= 0 in both. Returns the draws, the 95% quantile, and the chi-bar weights
    w_k = P(k dedifferentiation coordinates positive at the C1 projection).
    """
    I = local.info  # noqa: E741
    T = local.dediff + (local.nuisance_boundary if nuisance_boundary else [])
    F = [c for c in range(len(I)) if c not in T]
    Itt = I[np.ix_(T, T)]
    if F:
        Iff = I[np.ix_(F, F)]
        Itf = I[np.ix_(T, F)]
        J = Itt - Itf @ np.linalg.pinv(Iff, rcond=1e-10, hermitian=True) @ Itf.T
    else:
        J = Itt
    J = 0.5 * (J + J.T)
    w, V = np.linalg.eigh(J)
    ridge = 0.0
    if w.min() <= 1e-10 * max(w.max(), 1e-300):
        ridge = 1e-8 * max(w.max(), 1.0)
        w = np.maximum(w, ridge)
        J = (V * w) @ V.T
    cov = (V / w) @ V.T
    rng = np.random.default_rng(seed)
    Z = rng.multivariate_normal(np.zeros(len(T)), cov, size=n_draws, method="cholesky")
    dd = list(range(len(local.dediff)))
    nb = list(range(len(local.dediff), len(T)))
    d1, pos = _cone_distance(Z, J, [], dd + nb)
    d0, _ = _cone_distance(Z, J, dd, nb)
    lam = np.maximum(d0 - d1, 0.0)
    k = pos[:, : len(dd)].sum(1)
    weights = np.bincount(k, minlength=len(dd) + 1) / n_draws
    return {"draws": lam, "crit95": float(np.quantile(lam, 0.95)), "weights": weights,
            "n_boundary_nuisance": len(nb), "ridge": ridge,
            "corr": (cov[np.ix_(dd, dd)] / np.sqrt(np.outer(np.diag(cov)[dd], np.diag(cov)[dd])))
            .tolist()}  # fmt: skip


def chibar_pvalue(lam: float, weights) -> float:
    """``P(sum_k w_k chi2_k > lam)`` (chi2_0 = point mass at 0)."""
    w = np.asarray(weights, dtype=float)
    p = sum(w[k] * stats.chi2.sf(lam, k) for k in range(1, len(w)))
    return float(p + (w[0] if lam <= 0 else 0.0))


def chibar_crit(weights, alpha: float = 0.05) -> float:
    """Critical value of a chi-bar-squared mixture at level ``alpha``."""
    from scipy.optimize import brentq

    return float(brentq(lambda c: chibar_pvalue(c, weights) - alpha, 1e-9, 100.0))


#: Chi-bar weights for three orthogonal boundary parameters.
BINOMIAL_WEIGHTS = np.array([1, 3, 3, 1]) / 8.0


def bootstrap_dataset(sim, P0: np.ndarray, horizons, ns, seed: int, key: tuple, b: int,
                      confusion: np.ndarray | None = None) -> list[Cohort]:  # fmt: skip
    """Simulate replicate ``b`` of the parametric bootstrap from ``P0`` (theta_0).

    Same design (horizons, clones per horizon) and typing as the analysis.
    """
    return simulate_counts(sim, P0, horizons, ns, seed, ("boot", *key, b), confusion)


def parametric_bootstrap(sim, P0: np.ndarray, horizons, ns, reps, seed: int, key: tuple = (),
                         confusion: np.ndarray | None = None,
                         settings: FitSettings | None = None) -> list[dict]:  # fmt: skip
    """Lambda on replicates ``reps`` (iterable of indices) simulated from ``P0``.

    Each replicate is refitted under both models like the original data; ``P0`` (the
    generating value, known in the bootstrap world) is an extra H0 start.
    """
    P0 = np.asarray(P0, dtype=float)
    if np.any(P0[1, B_DEDIFF] != 0):
        raise ValueError("the bootstrap must simulate from an H0 point (B dedifferentiation = 0)")
    out = []
    for b in reps:
        data = bootstrap_dataset(sim, P0, horizons, ns, seed, key, b, confusion)
        lik = likelihood(sim, data, confusion)
        r = lrt(lik, rng_for(seed, "boot-starts", *key, b), settings, warm={"h0": [P0]})
        out.append({"b": int(b), "lambda": r["lambda"], "ll0": r["ll0"], "ll1": r["ll1"],
                    "q_hat": r["q_hat"], "m_ba_hat": r["m_ba_hat"],
                    "best_start0": r["best_start0"], "best_start1": r["best_start1"],
                    "h0_improved_by_full": r["h0_improved_by_full"], "evals": r["evals"],
                    "seconds": r["seconds"]})  # fmt: skip
    return out


def dediff_scores(lik, P0: np.ndarray) -> dict[str, float]:
    """Return the log-likelihood derivative at ``P0`` along each dedifferentiation route.

    Mass moves from the H0 entries of B, in proportion, to that route. At the H0
    maximum, all three <= 0 means theta_0 satisfies the full model's KKT conditions
    (then Lambda = 0 unless a distant optimum exists).
    """
    D = np.zeros((3, 6))
    keep = P0[1, B_KEEP] / P0[1, B_KEEP].sum()
    for j, i in enumerate(B_DEDIFF):
        D[j, i] = 1.0
        D[j, B_KEEP] = -keep
    g = lik.log_likelihood_P(P0, D, np.ones(3, dtype=np.int64), grad=True)[1]
    return {e: float(v) for e, v in zip(DEDIFF, g, strict=True)}


def run_test(lik, rng: np.random.Generator, settings: FitSettings | None = None,
             warm: dict | None = None, chibar_draws: int = 20000, chibar_seed: int = 0,
             asymptotics: bool = True) -> dict:  # fmt: skip
    """LRT on one dataset plus its asymptotic p-values (JSON-ready record).

    p-values: ``p_chibar`` (Monte-Carlo chi-bar null from the observed information at
    theta_0, :func:`chibar_null`), ``p_binomial`` (weights 1/8, 3/8, 3/8, 1/8) and
    ``p_chi2_3`` (plain chi-square with 3 df, conservative). ``P0`` / ``P1`` are kept as
    lists for warm starts and bootstraps.
    """
    r = lrt(lik, rng, settings, warm)
    lam = r["lambda"]
    out = {k: v for k, v in r.items() if k not in ("P0", "P1")}
    out["P0"], out["P1"] = r["P0"].tolist(), r["P1"].tolist()
    out["score_dediff"] = dediff_scores(lik, r["P0"])
    out["p_binomial"] = chibar_pvalue(lam, BINOMIAL_WEIGHTS)
    out["p_chi2_3"] = float(stats.chi2.sf(lam, 3))
    if asymptotics:
        t0 = time.perf_counter()
        loc = local_information(lik, r["P0"])
        cb = chibar_null(loc, chibar_draws, chibar_seed)
        out["chibar"] = {
            "crit95": cb["crit95"], "weights": cb["weights"].tolist(),
            "n_boundary_nuisance": cb["n_boundary_nuisance"],
            "boundary_nuisance": [loc.names[c] for c in loc.nuisance_boundary],
            "ridge": cb["ridge"], "min_eig": loc.min_eig, "corr": cb["corr"],
            "seconds": time.perf_counter() - t0,
        }  # fmt: skip
        out["p_chibar"] = float((1 + np.sum(cb["draws"] >= lam)) / (1 + len(cb["draws"])))
        out["p_chibar_weights"] = chibar_pvalue(lam, cb["weights"])
    return out
