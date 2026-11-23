"""Misspecification study for the exact two-type GW likelihood: shared code.

* ``TRUTHS`` (the 9 XG2 ground truths) and ``NULLS`` (dediff-study null rows N1, N2);
* ``simulate``: numpy two-type branching simulator with optional misspecification
  (founder type mixture, multiple founders, per-clone Dirichlet rows, asynchronous
  generations, density dependence, time-varying rates, cell loss, mis-typing), conditioned
  on detection (clones with zero observed cells are discarded, as in the pipeline);
* ``make_lik``: exact likelihood (adjoint kernel) of a dataset under the standard model or
  an extended one (founder mixture / sizes, confusion matrix, capture);
* ``fit_mle`` / ``fit_extended``: multistart maximum likelihood (L-BFGS-B, tight
  tolerances), extended models with finite-difference gradients in the extra parameters;
* ``profile_interval``: profile-likelihood 95% interval of a B-row functional;
* ``gof``: binned likelihood-ratio goodness-of-fit (G^2) against the saturated multinomial;
* identifiable summaries (growth rate, stable A fraction, m_BA).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from bio_sbi.inference import dediff_lrt as dl  # noqa: E402
from bio_sbi.summaries import Cohort  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "profile_intervals"))
import plib  # noqa: E402

OUT = Path("/home/matteo/BIO-SBI/results/exact_gw/misspecification")
HORIZONS = (4, 6, 8, 10)
NAMES = [f"A.{e}" for e in dl.ENTRIES["A"]] + [f"B.{e}" for e in dl.ENTRIES["B"]]
# daughters (A, B) per outcome, rows A / B, columns in dl.ENTRIES order
OFF = np.array([
    [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]],  # A: death persist to_B AA AB BB
    [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]],  # B: death to_A persist AA AB BB
])
DIV = [3, 4, 5]  # division entries (both rows)
PERSIST = {0: 1, 1: 2}  # index of "persist" per row


def _P(a, b):
    return np.array([a, b], dtype=float)


#: XG2 truths (rows in dl.ENTRIES order: A death persist to_B AA AB BB; B death to_A persist AA AB BB)
TRUTHS = {
    "T1_ref": _P([.03, .03, .05, .55, .05, .29], [.10, .80, .10, 0, 0, 0]),
    "T2_toA05": _P([.03, .03, .05, .55, .05, .29], [.25, .50, .25, 0, 0, 0]),
    "T3_Bdiv": _P([.03, .03, .05, .55, .05, .29], [.10, .60, .08, .02, .05, .15]),
    "T4_slowA": _P([.06, .06, .06, .40, .10, .32], [.10, .80, .10, 0, 0, 0]),
    "T5_C7": _P([.05, .10, .10, .50, .15, .10], [.08, .10, .05, .05, .05, .67]),
}


def _load_prior_truths():
    for g in ("P1", "P2", "P3", "P4"):
        d = np.load(f"/home/matteo/BIO-SBI/results/exact_gw/data_size/{g}/n1024_s0/reference.npz")
        t = d["truth"][:12]
        TRUTHS[g] = np.array([t[:6], t[6:12]], dtype=float)


_load_prior_truths()
GTS = list(TRUTHS)
REF_A = [.03, .03, .05, .55, .05, .29]
NULLS = {
    "N1": _P(REF_A, [.2, 0, .8, 0, 0, 0]),
    "N2": _P(REF_A, [.1, 0, .5, 0, 0, .4]),
}


# ---------------------------------------------------------------------- summaries
def mean_matrix(P):
    """M[k, j] = expected type-j daughters of one type-k cell per generation."""
    return np.einsum("ko,koj->kj", np.asarray(P), OFF)


def summaries(P) -> dict:
    """Growth rate rho (Perron root), stable A fraction, m_BA, B division mass."""
    M = mean_matrix(P)
    w, V = np.linalg.eig(M.T)  # left eigenvectors of M
    i = int(np.argmax(w.real))
    v = np.abs(V[:, i].real)
    return {"rho": float(w[i].real), "stableA": float(v[0] / v.sum()),
            "m_BA": float(M[1, 0]), "B.div": float(np.sum(P[1, DIV])),
            "m_AB": float(M[0, 1])}  # fmt: skip


def pdict(P) -> dict:
    return {n: float(v) for n, v in zip(NAMES, np.asarray(P).ravel(), strict=True)}


# ---------------------------------------------------------------------- scenarios
SCENARIOS = {
    # name: (description, simulator options, extended fits)
    "base": ("correctly specified", {}, []),
    "F70": ("founder A 70% / B 30%", {"founder_A": 0.7}, ["fA"]),
    "D15A": ("15% doublets (2 A founders)", {"doublet": 0.15}, ["d2"]),
    "D15mix": ("15% doublets, founders A/B 70/30", {"doublet": 0.15, "founder_A": 0.7},
               ["fA+d2"]),  # fmt: skip
    "MTba0.5": ("B read as A 0.5%", {"confusion": ("b_as_a", 0.005)}, ["true", "eBA"]),
    "MTba2": ("B read as A 2%", {"confusion": ("b_as_a", 0.02)}, ["true", "eBA"]),
    "MTba5": ("B read as A 5%", {"confusion": ("b_as_a", 0.05)}, ["true", "eBA"]),
    "MTsym2": ("symmetric mis-typing 2%", {"confusion": ("symmetric", 0.02)}, ["true", "esym"]),
    "HET50": ("per-clone rows Dirichlet(50 theta)", {"kappa": 50.0}, []),
    "HET10": ("per-clone rows Dirichlet(10 theta)", {"kappa": 10.0}, []),
    "ASYNC": ("generation h-1/h/h+1 w.p. .25/.5/.25", {"async": True}, []),
    "DENS": ("division mass x 1/(1+N/K), gen-10 size halved", {"density": True}, []),
    "LOSS90": ("each cell observed w.p. 0.9", {"capture": 0.9}, ["true", "cap"]),
    "LOSS80": ("each cell observed w.p. 0.8", {"capture": 0.8}, ["true", "cap"]),
    "TVtoA": ("B->A doubles after generation 5 (capped)", {"tv": "toA"}, []),
    "TVslow": ("division mass x 0.75 from generation 6", {"tv": "slow"}, []),
}


def tv_matrix(P, kind: str, t: int) -> np.ndarray:
    """Outcome matrix in generation ``t`` (1-based) of a time-varying scenario."""
    if t <= 5:
        return P
    Q = P.copy()
    if kind == "toA":
        c = P[1, 1]
        c2 = min(2 * c, (1 + c) / 2)
        rest = [0, 2, 3, 4, 5]
        Q[1, rest] = P[1, rest] * (1 - c2) / max(1 - c, 1e-300)
        Q[1, 1] = c2
    elif kind == "slow":
        for k in (0, 1):
            lost = 0.25 * Q[k, DIV].sum()
            Q[k, DIV] *= 0.75
            Q[k, PERSIST[k]] += lost
    else:
        raise ValueError(kind)
    return Q


def _density_rows(P, N, K):
    """Per-clone matrices (n, 2, 6): division mass x 1/(1 + N/K), lost mass to persist."""
    f = 1.0 / (1.0 + N / K)
    Q = np.broadcast_to(P, (len(N), 2, 6)).copy()
    for k in (0, 1):
        lost = Q[:, k, DIV].sum(1) * (1 - f)
        Q[:, k, DIV] *= f[:, None]
        Q[:, k, PERSIST[k]] += lost
    return Q


def _dirichlet_rows(P, kappa, n, rng):
    """Per-clone rows ~ Dirichlet(kappa * p) over the positive entries (zeros stay 0)."""
    Q = np.zeros((n, 2, 6))
    for k in (0, 1):
        pos = P[k] > 0
        g = rng.gamma(kappa * P[k, pos], size=(n, pos.sum()))
        g /= g.sum(1, keepdims=True)
        Q[:, k, pos] = g
    return Q


def _step(a, b, Q, rng):
    """One synchronous generation. ``Q`` (2, 6) or (n, 2, 6)."""
    if Q.ndim == 2:
        oa = rng.multinomial(a, Q[0])
        ob = rng.multinomial(b, Q[1])
    else:
        oa = rng.multinomial(a, Q[:, 0])
        ob = rng.multinomial(b, Q[:, 1])
    na = oa @ OFF[0, :, 0] + ob @ OFF[1, :, 0]
    nb = oa @ OFF[0, :, 1] + ob @ OFF[1, :, 1]
    return na, nb


def simulate_true(P, gens: int, n: int, rng, founder_A=1.0, doublet=0.0, kappa=None,
                  async_=False, K=None, tv=None, max_cells=200_000):  # fmt: skip
    """True (A, B) counts of ``n`` clones (not conditioned on detection)."""
    P = np.asarray(P, float)
    size = 1 + (rng.random(n) < doublet)
    nA0 = rng.binomial(size, founder_A)
    a, b = nA0.astype(np.int64), (size - nA0).astype(np.int64)
    G = np.full(n, gens)
    if async_:
        G = gens + rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25])
    Gmax = int(G.max())
    Qc = _dirichlet_rows(P, kappa, n, rng) if kappa else None
    outA, outB = a.copy(), b.copy()
    for t in range(1, Gmax + 1):
        if tv is not None:
            Q = tv_matrix(P, tv, t)
        elif K is not None:
            Q = _density_rows(P, (a + b).astype(float), K)
        elif Qc is not None:
            Q = Qc
        else:
            Q = P
        a, b = _step(a, b, Q, rng)
        if a.max(initial=0) + b.max(initial=0) > max_cells:
            raise RuntimeError("max_cells exceeded")
        hit = G == t
        outA[hit], outB[hit] = a[hit], b[hit]
    return np.stack([outA, outB], 1)


def observe(true, rng, capture=1.0, confusion=None):
    """Thin each cell (capture) and relabel (confusion (2, 2): true -> label)."""
    a, b = true[:, 0], true[:, 1]
    if capture < 1:
        a, b = rng.binomial(a, capture), rng.binomial(b, capture)
    if confusion is not None:
        C = np.asarray(confusion)
        a_as_a = rng.binomial(a, C[0, 0])
        b_as_a = rng.binomial(b, C[1, 0])
        a, b = a_as_a + b_as_a, (a - a_as_a) + (b - b_as_a)
    return np.stack([a, b], 1).astype(np.int64)


def confusion_of(spec):
    if spec is None:
        return None
    kind, rate = spec
    return dl.confusion_matrix(rate, kind)


def simulate(P, n: int, rng, horizons=HORIZONS, founder_A=1.0, doublet=0.0, kappa=None,
             async_=False, K=None, tv=None, capture=1.0, confusion=None,
             return_true=False):  # fmt: skip
    """``n`` detected clones per horizon (independent cohorts), as a Cohort list.

    ``confusion``: (2, 2) matrix or None; clones with no observed cell are discarded.
    ``return_true``: also return the true counts of the kept clones (for paired relabels).
    """
    out, trues = [], []
    for h in horizons:
        kept_obs, kept_true, have, batch = [], [], 0, int(1.5 * n) + 64
        while have < n:
            tr = simulate_true(P, h, batch, rng, founder_A, doublet, kappa, async_, K, tv)
            ob = observe(tr, rng, capture, confusion)
            det = ob.sum(1) > 0
            kept_obs.append(ob[det])
            kept_true.append(tr[det])
            have += int(det.sum())
            rate = max(have, 1) / (batch * len(kept_obs))
            batch = int(min(max((n - have) / rate * 1.3, 64), 50 * n))
        out.append(Cohort(float(h), np.concatenate(kept_obs)[:n]))
        trues.append(np.concatenate(kept_true)[:n])
    return (out, trues) if return_true else out


def density_K(P, target=0.5, gens=10, n=4000, seed=0) -> float:
    """Carrying capacity K so that the mean true clone size at ``gens`` is ``target`` x
    the density-free mean (fixed seed; log-bisection)."""
    rng = np.random.default_rng(seed)
    base = simulate_true(P, gens, 4 * n, rng).sum(1).mean()
    lo, hi = np.log(0.1), np.log(1e5)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        m = simulate_true(P, gens, n, np.random.default_rng(seed + 1), K=np.exp(mid)).sum(1).mean()
        if m / base < target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def scenario_data(scen: str, P, n: int, rng, K_lookup=None, base_true=None):
    """Dataset of scenario ``scen`` (``K_lookup``: K for DENS)."""
    o = dict(SCENARIOS[scen][1])
    conf = confusion_of(o.pop("confusion", None))
    kw = {"founder_A": o.get("founder_A", 1.0), "doublet": o.get("doublet", 0.0),
          "kappa": o.get("kappa"), "async_": o.get("async", False),
          "K": K_lookup if o.get("density") else None, "tv": o.get("tv"),
          "capture": o.get("capture", 1.0), "confusion": conf}  # fmt: skip
    return simulate(P, n, rng, **kw)


# ---------------------------------------------------------------------- likelihood
_SIM = None


def sim():
    global _SIM
    if _SIM is None:
        _SIM = dl.make_simulator()
    return _SIM


def make_lik(cohorts, founder_A=1.0, size_probs=None, confusion=None, capture=1.0):
    """Exact likelihood; ``size_probs`` = (p1, p2) for founder sizes (1, 2)."""
    lik = dl.likelihood(sim(), cohorts, None)
    set_model(lik, founder_A, size_probs, confusion, capture)
    return lik


def set_model(lik, founder_A=1.0, size_probs=None, confusion=None, capture=1.0):
    m = lik.model
    kw = {"founder": np.array([founder_A, 1 - founder_A], float),
          "capture": np.array([capture, capture], float),
          "confusion": np.eye(2) if confusion is None else np.asarray(confusion, float)}  # fmt: skip
    if size_probs is not None:
        kw["sizes"] = np.array([1, 2], dtype=np.int64)
        kw["size_probs"] = np.asarray(size_probs, float)
    else:
        kw["sizes"] = np.array([1], dtype=np.int64)
        kw["size_probs"] = np.array([1.0])
    lik.model = replace(m, **kw)


FULL = dl.Parametrisation([dl.RowSpec(0, [list(range(6))]), dl.RowSpec(1, [list(range(6))])],
                          "full6")  # fmt: skip
TIGHT = {"ftol": 1e-15, "gtol": 1e-9, "maxiter": 3000}
LOOSE = {"ftol": 1e-11, "gtol": 1e-6, "maxiter": 1500}


def _starts(rng, n_random: int):
    st = {"generic": dl.with_dediff(dl.GENERIC, 0.3, dl.COMPOSITIONS["mixed"])}
    for i in range(n_random):
        st[f"random_{i}"] = dl.random_P(rng, h0=False)
    return st


def fit_mle(lik, rng, n_random: int = 6, pins=(0.02, 0.3, 0.7), extra_starts=()) -> dict:
    """Multistart MLE of the standard (12-probability) model.

    Starts: a generic point, ``n_random`` flat-Dirichlet draws, ``extra_starts`` (e.g. the
    standard fit when called for another dataset; never the truth), and B.to_A pinned at
    ``pins`` (fitted, then released): plain multistart misses low-B.to_A optima. Coarse
    tolerances first; the two best are polished with tight tolerances.
    """
    e0 = lik.evals
    res = []
    for name, P in list(_starts(rng, n_random).items()) + [
            (f"extra_{i}", P) for i, P in enumerate(extra_starts)]:  # fmt: skip
        r = dl.maximise(lik, FULL, FULL.encode(P), LOOSE)
        res.append((r["ll"], name, r["s"]))
    base = max(res)[2]
    Pb = FULL.P(base)[0]
    for c in pins:
        pin = dl.pinned_model(c)
        rp = dl.maximise(lik, pin, pin.encode(dl.with_to_a(Pb, c)), LOOSE)
        r = dl.maximise(lik, FULL, FULL.encode(rp["P"]), LOOSE)
        res.append((r["ll"], f"pin{c:g}", r["s"]))
    res.sort(key=lambda t: -t[0])
    best = None
    for ll, name, s in res[:2]:
        r = dl.maximise(lik, FULL, s, TIGHT)
        if best is None or r["ll"] > best["ll"]:
            best = {**r, "start": name}
    lls = sorted([t[0] for t in res], reverse=True)
    return {"ll": best["ll"], "P": best["P"], "s": best["s"], "start": best["start"],
            "n_tied": int(sum(v > best["ll"] - 0.01 for v in lls)),
            "second_ll": float(lls[1]) if len(lls) > 1 else None,
            "evals": int(lik.evals - e0)}  # fmt: skip


# ---------------------------------------------------------------------- extended models
EXTRA = {
    # name: (list of extra parameter names, model setter(x) -> kwargs for set_model)
    "fA": (["founder_A"], lambda x: {"founder_A": x[0]}),
    "d2": (["doublet"], lambda x: {"size_probs": (1 - x[0], x[0])}),
    "fA+d2": (["founder_A", "doublet"],
              lambda x: {"founder_A": x[0], "size_probs": (1 - x[1], x[1])}),
    "eBA": (["e_BA"], lambda x: {"confusion": dl.confusion_matrix(x[0], "b_as_a")}),
    "esym": (["e_sym"], lambda x: {"confusion": dl.confusion_matrix(x[0], "symmetric")}),
    "cap": (["capture"], lambda x: {"capture": x[0]}),
}
# value of the extra parameters at which the extended model is the standard one
NULL_X = {"fA": [1.0], "d2": [0.0], "fA+d2": [1.0, 0.0], "eBA": [0.0], "esym": [0.0],
          "cap": [1.0]}  # fmt: skip
X_BOUNDS = {"founder_A": (0.0, 1.0), "doublet": (0.0, 0.95), "e_BA": (0.0, 0.3),
            "e_sym": (0.0, 0.3), "capture": (0.3, 1.0)}  # fmt: skip


def fit_extended(lik, kind: str, std: dict, rng, x_starts=None, fd: float = 1e-6) -> dict:
    """MLE of the standard model extended by ``kind`` (``EXTRA``); the extra parameters'
    gradient by one-sided finite differences (towards the interior). Starts: the standard
    fit at the null value and at ``x_starts``; then one flat-Dirichlet start at each.
    """
    names, setter = EXTRA[kind]
    bounds = [X_BOUNDS[n] for n in names]
    d = FULL.dim
    e0 = lik.evals

    def ll_at(s, x, grad=True):
        set_model(lik, **setter(x))
        return dl.loglik(lik, FULL, s, grad)

    def neg(z):
        s, x = np.clip(z[:d], 0, 1), z[d:]
        v, g = ll_at(s, x)
        if not np.isfinite(v):
            return 1e300, np.zeros_like(z)
        gx = np.zeros(len(x))
        for j in range(len(x)):
            h = fd if x[j] + fd <= bounds[j][1] else -fd
            xp = x.copy()
            xp[j] += h
            gx[j] = (ll_at(s, xp, grad=False)[0] - v) / h
        return -v, -np.concatenate([g, gx])

    nullx = np.array(NULL_X[kind], float)
    xs = [np.clip(nullx + np.where(nullx > 0.5, -0.02, 0.02), *np.array(bounds).T)]
    for x in (x_starts or []):
        xs.append(np.asarray(x, float))
    res = []
    for x in xs:
        for s0 in (std["s"], FULL.encode(dl.random_P(rng, h0=False))):
            z0 = np.concatenate([s0, x])
            r = minimize(neg, z0, jac=True, method="L-BFGS-B",
                         bounds=[(0, 1)] * d + bounds, options=LOOSE)  # fmt: skip
            res.append((float(-r.fun), r.x))
    res.sort(key=lambda t: -t[0])
    r = minimize(neg, res[0][1], jac=True, method="L-BFGS-B", bounds=[(0, 1)] * d + bounds,
                 options=TIGHT)  # fmt: skip
    z = r.x
    ll = float(-r.fun)
    set_model(lik, **setter(nullx))  # restore the standard model
    if ll < std["ll"]:  # the standard fit is a point of the extended model
        z, ll = np.concatenate([std["s"], nullx]), std["ll"]
    return {"ll": ll, "P": FULL.P(np.clip(z[:d], 0, 1))[0], "x": dict(zip(names, z[d:].tolist())),
            "lr": max(2 * (ll - std["ll"]), 0.0), "evals": int(lik.evals - e0)}  # fmt: skip


X_GRID = {
    "fA": [[1.0], [0.8], [0.6], [0.4]],
    "d2": [[0.0], [0.12], [0.3]],
    "fA+d2": [[f, d] for f in (1.0, 0.75, 0.5) for d in (0.0, 0.15, 0.3)],
    "eBA": [[0.0], [0.015], [0.04], [0.1]],
    "esym": [[0.0], [0.015], [0.04], [0.1]],
    "cap": [[1.0], [0.88], [0.75], [0.6]],
}


def fit_extended_grid(lik, kind: str, std: dict, rng) -> dict:
    """Extended-model MLE by a grid over the extra parameters (standard multistart MLE with
    the extra parameters fixed at each grid point), then a joint polish (finite-difference
    gradient in the extra parameters) from the two best grid points. More robust than
    :func:`fit_extended`: the extended optimum can lie on another branch of the B row.
    """
    names, setter = EXTRA[kind]
    bounds = [X_BOUNDS[n] for n in names]
    d = FULL.dim
    e0 = lik.evals
    grid = []
    for x in X_GRID[kind]:
        set_model(lik, **setter(x))
        f = fit_mle(lik, rng, n_random=2, extra_starts=[std["P"]])
        grid.append((f["ll"], list(x), f["s"]))
    grid.sort(key=lambda t: -t[0])

    def ll_at(s, x, grad=True):
        set_model(lik, **setter(x))
        return dl.loglik(lik, FULL, s, grad)

    def neg(z, fd=1e-6):
        s, x = np.clip(z[:d], 0, 1), z[d:]
        v, g = ll_at(s, x)
        if not np.isfinite(v):
            return 1e300, np.zeros_like(z)
        gx = np.zeros(len(x))
        for j in range(len(x)):
            h = fd if x[j] + fd <= bounds[j][1] else -fd
            xp = x.copy()
            xp[j] += h
            gx[j] = (ll_at(s, xp, grad=False)[0] - v) / h
        return -v, -np.concatenate([g, gx])

    best = (grid[0][0], np.concatenate([grid[0][2], grid[0][1]]))
    for ll0, x0, s0 in grid[:2]:
        r = minimize(neg, np.concatenate([s0, x0]), jac=True, method="L-BFGS-B",
                     bounds=[(0, 1)] * d + bounds, options=TIGHT)  # fmt: skip
        if -r.fun > best[0]:
            best = (float(-r.fun), r.x)
    nullx = NULL_X[kind]
    set_model(lik, **setter(nullx))
    ll, z = best
    std_ll = max(std["ll"], next((g[0] for g in grid if g[1] == list(nullx)), -np.inf))
    if ll < std_ll:
        ll, z = std_ll, np.concatenate([std["s"], nullx])
    return {"ll": float(ll), "P": FULL.P(np.clip(z[:d], 0, 1))[0],
            "x": dict(zip(names, np.asarray(z[d:]).tolist())),
            "lr": max(2 * (ll - std_ll), 0.0), "std_ll_used": float(std_ll),
            "grid": [{"x": g[1], "ll": g[0]} for g in grid],
            "evals": int(lik.evals - e0)}  # fmt: skip


def lrt_pvalue(lr: float, kind: str) -> float:
    """p-value of the extended-vs-standard LR: 50:50 chi2_0/chi2_1 mixture per boundary
    parameter (fA+d2: chi-bar with weights 1/4, 1/2, 1/4)."""
    if kind == "fA+d2":
        return float(0.5 * stats.chi2.sf(lr, 1) + 0.25 * stats.chi2.sf(lr, 2)) if lr > 0 else 1.0
    return float(0.5 * stats.chi2.sf(lr, 1)) if lr > 0 else 1.0


# ---------------------------------------------------------------------- profile intervals
def phys_of(P) -> np.ndarray:
    return dl.physical(sim(), P)


def profile_interval(lik, P_hat, ll_hat: float, func: str = "B.to_A", thr: float = 1.920729,
                     max_steps: int = 40) -> dict:  # fmt: skip
    """Profile-likelihood interval ``{c: l_p(c) >= l_hat - thr}`` of a B-row functional.

    Outward steps from the MLE (warm starts + a projection of the MLE at every point),
    secant refinement of each crossing; an end of the range inside the set closes the
    interval there.
    """
    F = plib.FUNCTIONALS[func]
    m = plib.ConstrainedModel(lik, F)
    names = m.names
    ph_hat = phys_of(P_hat)
    c_hat = float(F.value(ph_hat, names))
    e0 = lik.evals
    cache = {}

    def fit(c, warm):
        m.set_c(c)
        best = (-np.inf, None)
        for st in (warm, ph_hat):
            r = m.maximise(m.encode(st), tight=False)
            if r["ll"] > best[0]:
                best = (r["ll"], m.physical(r["s"]))
        r = m.maximise(m.encode(best[1]), tight=True)
        best = max(best, (r["ll"], m.physical(r["s"])), key=lambda t: t[0])
        cache[c] = best
        return best

    ends = []
    for sgn in (-1, +1):
        c, warm, prev = c_hat, ph_hat, (c_hat, 0.0, ph_hat)
        h = max(0.02 * F.cmax, 0.25 * c_hat if sgn < 0 else 0.02)
        h = min(h, 0.1 * F.cmax)
        end = None
        for _ in range(max_steps):
            cn = c + sgn * h
            if cn <= 0 or cn >= F.cmax:
                cn = 0.0 if sgn < 0 else F.cmax
                ll, ph = fit(cn, warm)
                drop = ll_hat - ll
                if drop <= thr:
                    end = cn
                    break
            else:
                ll, ph = fit(cn, warm)
                drop = ll_hat - ll
            if drop > thr:
                # secant / bisection between prev (inside) and cn (outside)
                a, da, pa = prev[0], prev[1] - thr, prev[2]
                b, db, pb = cn, drop - thr, ph
                for it in range(8):
                    cm = a - da * (b - a) / (db - da) if it % 2 == 0 else 0.5 * (a + b)
                    llm, pm = fit(cm, pa)
                    dm = ll_hat - llm - thr
                    if abs(dm) < 0.01 or abs(b - a) < 1e-5 * F.cmax:
                        break
                    if dm < 0:
                        a, da, pa = cm, dm, pm
                    else:
                        b, db, pb = cm, dm, pm
                end = cm
                break
            prev = (cn, drop, ph)
            c, warm = cn, ph
            if drop < 0.5:  # still flat: larger steps
                h = min(2 * h, 0.1 * F.cmax)
        ends.append(end)
    best_seen = max(v[0] for v in cache.values()) if cache else ll_hat
    return {"func": func, "c_hat": c_hat, "lo": ends[0], "hi": ends[1],
            "better_point": float(best_seen - ll_hat), "fits": len(cache),
            "evals": int(lik.evals - e0)}  # fmt: skip


# ---------------------------------------------------------------------- goodness of fit
def law_tables(lik, P) -> list[np.ndarray]:
    """Detection-conditioned law of the observed counts per horizon on an enlarged grid
    (mass outside the grid is returned as the last element of each pair)."""
    from bio_sbi.inference.gw_exact import grid_spec, law_poly, monomials, poly_coefficients

    model = lik.model
    mono = monomials(model)
    C, _ = poly_coefficients(mono[2], len(mono[0]), P, np.zeros((0, 6)), np.zeros(0, np.int64))
    out = []
    for h in lik.data:
        g = grid_spec(int(2 * h.a.max() + 20), int(2 * h.b.max() + 20))
        p, _ = law_poly(model, C, mono, h.generations, g)
        p = np.clip(p, 0, None)
        p0 = p[0, 0]
        p[0, 0] = 0.0
        q = 1.0 - p0  # detection probability (law mass outside the grid is tiny)
        out.append((p / q, max(0.0, 1.0 - p.sum() / q)))
    return out


FRAC_EDGES = (1 / 3, 2 / 3)


def _classes(a, b, nbreaks):
    """Bin index: total-size class (by ``nbreaks``) x A-fraction class (5)."""
    tot = a + b
    sc = np.searchsorted(nbreaks, tot, side="right")
    f = np.where(tot > 0, a / np.maximum(tot, 1), 0.0)
    fc = np.where(a == 0, 0, np.where(b == 0, 4, 1 + np.searchsorted(FRAC_EDGES, f, side="left")))
    return sc * 5 + fc


def gof(lik, P, n_size_bins: int = 6, min_expected: float = 5.0) -> dict:
    """Binned G^2 (and Pearson X^2) of the observed (A, B) counts against the fitted law.

    Per horizon: bins = total size classes (quantiles of the fitted law) x A-fraction
    classes {0, (0, 1/3], (1/3, 2/3], (2/3, 1), 1}; bins with expected count below
    ``min_expected`` are pooled (and the pool merged into the largest bin if still small).
    Reference distributions: chi2 with sum(K_h - 1) - 10 df (``p``) and sum(K_h - 1) df
    (``p_upper``, Chernoff-Lehmann bound for raw-data MLEs).
    """
    G2, X2, K = 0.0, 0.0, 0
    per = []
    for h, (tab, tail) in zip(lik.data, law_tables(lik, P), strict=True):
        n = h.clones
        Lx, Ly = tab.shape
        aa, bb = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="ij")
        tot_p = np.bincount((aa + bb).ravel(), tab.ravel())
        cdf = np.cumsum(tot_p)
        qs = np.linspace(0, 1, n_size_bins + 1)[1:-1]
        nbreaks = np.unique(np.searchsorted(cdf, qs * cdf[-1]))  # size <= break -> lower class
        nbreaks = nbreaks + 0.5
        cls_grid = _classes(aa.ravel(), bb.ravel(), nbreaks)
        nb = int(cls_grid.max()) + 6
        E = np.bincount(cls_grid, tab.ravel(), minlength=nb) * n
        largest_size_class_bins = None  # noqa: F841
        E[-1] += tail * n
        cls_obs = _classes(h.a, h.b, nbreaks)
        O = np.bincount(cls_obs, h.weight, minlength=nb)
        O = np.concatenate([O[:nb - 1], [O[nb - 1:].sum()]]) if len(O) > nb else O
        small = E < min_expected
        Eb = list(E[~small])
        Ob = list(O[~small])
        Ep, Op = E[small].sum(), O[small].sum()
        if Ep >= min_expected:
            Eb.append(Ep)
            Ob.append(Op)
        elif Ep > 0 or Op > 0:
            j = int(np.argmax(Eb))
            Eb[j] += Ep
            Ob[j] += Op
        Eb, Ob = np.array(Eb), np.array(Ob)
        g2 = 2 * float(np.sum(np.where(Ob > 0, Ob * np.log(np.maximum(Ob, 1e-300) / Eb), 0.0)))
        x2 = float(np.sum((Ob - Eb) ** 2 / Eb))
        G2 += g2
        X2 += x2
        K += len(Eb) - 1
        per.append({"h": h.generations, "bins": len(Eb), "G2": g2})
    df = max(K - 10, 1)
    return {"G2": G2, "X2": X2, "df": df, "df_upper": K,
            "p": float(stats.chi2.sf(G2, df)), "p_upper": float(stats.chi2.sf(G2, K)),
            "pX": float(stats.chi2.sf(X2, df)), "per_horizon": per}  # fmt: skip
