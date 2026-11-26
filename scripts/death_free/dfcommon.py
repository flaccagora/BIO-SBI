"""Structured truths (D0, S0, M): shared code.

Datasets (``kind``), each built from the 9 XG2 truths:

* ``D0`` - no death: ``A.death = B.death = 0``, the other outcomes of each row rescaled
  proportionally. No clone goes extinct, so every clone is detected.
* ``S0`` - no switching: ``A.to_B = B.to_A = 0``, rows rescaled. B gives rise to A only by
  dividing (if at all); for T1/T2/T4 (B never divides) B is terminal.
* ``M``  - mirror roles: B behaves like A with the labels swapped (B.death = A.death,
  B.persist = A.persist, B.to_A = A.to_B, B.div_BB = A.div_AA, B.div_AB = A.div_AB,
  B.div_AA = A.div_BB); stable composition 50/50. T1/T2/T3 share A's row, so their mirrors
  coincide: M uses T1 (standing for T1-T3), T4, T5, P1-P4.

Models fitted (``model``): ``free`` (the standard 12-probability model, flat prior) and the
true structure built in: ``nodeath`` (D0; death fixed at 0), ``noswitch`` (S0; switching
fixed at 0), ``tied`` (M; B row = mirror of the A row, 5 free probabilities).

Datasets: the misspecification study's simulator and seeding (``n`` = 512 detected clones
per generation 4/6/8/10, A founders, rng ``[truth index, seed, 0]``), so dataset ``s`` of
every kind uses the same random stream as the base dataset ``s`` of the original truth.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]


def _load_missp():
    """misspecification/common.py under its own module name (several studies have a
    ``common`` module)."""
    name = "missp_common"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "misspecification" / "common.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load_missp()

from bio_sbi.inference import dediff_lrt as dl  # noqa: E402

OUT = Path("/home/matteo/BIO-SBI/results/exact_gw/death_free")
DOCS = ROOT / "docs" / "death_free"
N = 512
GTS = C.GTS  # T1_ref T2_toA05 T3_Bdiv T4_slowA T5_C7 P1 P2 P3 P4
SHORT = {"T1_ref": "T1", "T2_toA05": "T2", "T3_Bdiv": "T3", "T4_slowA": "T4", "T5_C7": "T5",
         "P1": "P1", "P2": "P2", "P3": "P3", "P4": "P4"}  # fmt: skip
NAMES = C.NAMES
I_DEATH = (0, 6)  # A.death, B.death in the 12-vector
I_SWITCH = (2, 7)  # A.to_B, B.to_A
#: B entry receiving each A entry under the mirror (A: death persist to_B AA AB BB;
#: B: death to_A persist AA AB BB)
MIRROR_B_OF_A = [0, 2, 1, 5, 4, 3]
CONSTRAINED = {"D0": "nodeath", "S0": "noswitch", "M": "tied"}
KINDS = ("D0", "S0", "M")


def gts(kind: str) -> list[str]:
    return [g for g in GTS if not (kind == "M" and g in ("T2_toA05", "T3_Bdiv"))]


def _rescale(P: np.ndarray, zero: int | dict) -> np.ndarray:
    Q = np.asarray(P, float).copy()
    for r, i in (zero.items() if isinstance(zero, dict) else ((0, zero), (1, zero))):
        Q[r, i] = 0.0
    return Q / Q.sum(1, keepdims=True)


def no_death(P: np.ndarray) -> np.ndarray:
    """Death 0, each row's other outcomes rescaled to sum 1."""
    return _rescale(P, 0)


def no_switch(P: np.ndarray) -> np.ndarray:
    """A.to_B = B.to_A = 0, rows rescaled."""
    return _rescale(P, {0: 2, 1: 1})


def mirror(P: np.ndarray) -> np.ndarray:
    """B row = A row with the labels swapped."""
    Q = np.asarray(P, float).copy()
    Q[1, MIRROR_B_OF_A] = Q[0]
    return Q


TRANSFORM = {"orig": lambda P: np.asarray(P, float).copy(), "D0": no_death, "S0": no_switch,
             "M": mirror}  # fmt: skip


def truth(gt: str, kind: str) -> np.ndarray:
    """Truth (2, 6) of ``gt`` under dataset ``kind`` (orig / D0 / S0 / M)."""
    return TRANSFORM[kind](C.TRUTHS[gt])


def dataset(gt: str, seed: int, kind: str):
    """512 detected clones per generation 4/6/8/10 (same rng stream as the base study)."""
    gi = C.GTS.index(gt)
    return C.simulate(truth(gt, kind), N, np.random.default_rng([gi, seed, 0]))


# ---------------------------------------------------------------------- parametrisations
FREE = C.FULL
NODEATH = dl.Parametrisation([dl.RowSpec(0, [[1, 2, 3, 4, 5]], {0: 0.0}),
                              dl.RowSpec(1, [[1, 2, 3, 4, 5]], {0: 0.0})], "nodeath")  # fmt: skip
NOSWITCH = dl.Parametrisation([dl.RowSpec(0, [[0, 1, 3, 4, 5]], {2: 0.0}),
                               dl.RowSpec(1, [[0, 2, 3, 4, 5]], {1: 0.0})], "noswitch")  # fmt: skip
I_TOA = dl.ENTRIES["B"].index("to_A")


class Tied:
    """B row = mirror of the A row: 5 stick coordinates of A's row, reused for B with the
    B entries stick-broken in mirrored order (death, persist, to_A, BB, AB, AA), so the same
    sticks give the mirrored row. ``P`` returns the 10-coordinate (untied) derivative; the
    gradient of the tied coordinates is the sum of the two halves (``tied_grad``)."""

    name = "tied"
    dim = 5
    _two = dl.Parametrisation([dl.RowSpec(0, [[0, 1, 2, 3, 4, 5]]),
                               dl.RowSpec(1, [[0, 2, 1, 5, 4, 3]])], "tied2")  # fmt: skip
    _a = dl.Parametrisation([dl.RowSpec(0, [[0, 1, 2, 3, 4, 5]])], "A")

    def P(self, s):
        s = np.asarray(s, float)
        return self._two.P(np.concatenate([s, s]))

    def encode(self, P):
        return self._a.encode(P)

    @staticmethod
    def tied_grad(g):
        return g[:5] + g[5:]


TIED = Tied()
PARS = {"free": FREE, "nodeath": NODEATH, "noswitch": NOSWITCH, "tied": TIED}
FIX = {"free": lambda P: P, "nodeath": no_death, "noswitch": no_switch, "tied": mirror}


def pinned(model: str, c: float) -> dl.Parametrisation | None:
    """``model`` with ``B.to_A`` fixed at ``c`` (the low-B.to_A starts of ``C.fit_mle``);
    None where B.to_A is not free on its own (noswitch, tied)."""
    if model == "free":
        return dl.pinned_model(c)
    if model == "nodeath":
        return dl.Parametrisation([dl.RowSpec(0, [[1, 2, 3, 4, 5]], {0: 0.0}),
                                   dl.RowSpec(1, [[3, 4], [2, 5]], {0: 0.0, I_TOA: float(c)})],
                                  f"nodeath_pinned={c:g}")  # fmt: skip
    return None


def maximise(lik, par, s0, options):
    """``dl.maximise`` for row parametrisations; the tied model with summed gradients."""
    if not isinstance(par, Tied):
        return dl.maximise(lik, par, s0, options)

    def neg(s):
        P, dP, rows = par.P(s)
        v, g = lik.log_likelihood_P(P, dP, rows, grad=True)
        if not np.isfinite(v):
            return 1e300, np.zeros_like(s)
        return -v, -par.tied_grad(np.asarray(g))

    r = minimize(neg, np.clip(s0, 0, 1), jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * 5,
                 options={**dl.OPTIONS, **(options or {})})  # fmt: skip
    s = np.clip(r.x, 0.0, 1.0)
    return {"ll": float(-r.fun), "s": s, "P": par.P(s)[0]}


def loglik_at(lik, P) -> float:
    """Log-likelihood at an outcome matrix (no gradient)."""
    return float(dl.loglik(lik, FREE, FREE.encode(P), False)[0])


def fit_mle(lik, rng, model: str, n_random: int = 6, pins=(0.02, 0.3, 0.7)) -> dict:
    """Multistart MLE as ``misspecification.common.fit_mle``, for every model: a generic
    start, flat-Dirichlet draws (projected on the model), B.to_A-pinned starts where B.to_A
    is a free entry of its own; coarse fits, the two best polished."""
    par, fix = PARS[model], FIX[model]
    if model == "tied":
        n_random += 4  # no pinned starts
    starts = [("generic", dl.with_dediff(dl.GENERIC, 0.3, dl.COMPOSITIONS["mixed"]))]
    starts += [(f"random_{i}", dl.random_P(rng, h0=False)) for i in range(n_random)]
    res = []
    for name, P in starts:
        r = maximise(lik, par, par.encode(fix(P)), C.LOOSE)
        res.append((r["ll"], name, r["s"]))
    Pb = par.P(max(res, key=lambda t: t[0])[2])[0]
    for c in pins:
        pin = pinned(model, c)
        if pin is None:
            continue
        rp = maximise(lik, pin, pin.encode(fix(dl.with_to_a(Pb, c))), C.LOOSE)
        r = maximise(lik, par, par.encode(rp["P"]), C.LOOSE)
        res.append((r["ll"], f"pin{c:g}", r["s"]))
    res.sort(key=lambda t: -t[0])
    best = None
    for _ll, name, s in res[:2]:
        r = maximise(lik, par, s, C.TIGHT)
        if best is None or r["ll"] > best["ll"]:
            best = {**r, "start": name}
    lls = sorted([t[0] for t in res], reverse=True)
    return {"ll": best["ll"], "P": best["P"], "start": best["start"],
            "n_tied": int(sum(v > best["ll"] - 0.01 for v in lls)),
            "second_ll": float(lls[1])}  # fmt: skip
