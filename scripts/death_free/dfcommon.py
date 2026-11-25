"""Death-free dataset D0: shared code.

D0 truths: each of the 9 XG2 truths with ``A.death = B.death = 0`` and the other outcomes of
each row rescaled proportionally (relative fates kept). Without death no clone goes
extinct, so every clone is detected and the detection conditioning drops out.

Models fitted to D0 data:

* ``free``    - the standard 12-probability model (death estimated, flat prior): does
  inference find death = 0, and what happens to the weak directions?
* ``nodeath`` - death fixed at 0 (lab knows cells do not die): 10 free probabilities.

Datasets: the misspecification study's simulator and seeding (``n`` = 512 detected clones
per generation 4/6/8/10, A founders, rng ``[truth index, seed, 0]``), so D0 dataset ``s`` uses
the same random stream as the base dataset ``s`` of the original truth (paired).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

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


def no_death(P: np.ndarray) -> np.ndarray:
    """``P`` (2, 6) with death 0 and each row's other outcomes rescaled to sum 1."""
    Q = np.asarray(P, float).copy()
    Q[:, 0] = 0.0
    return Q / Q.sum(1, keepdims=True)


def truth(gt: str, kind: str) -> np.ndarray:
    """``kind`` 'orig' (XG2 truth) or 'D0' (death-free version), (2, 6)."""
    P = C.TRUTHS[gt]
    return no_death(P) if kind == "D0" else P.copy()


def dataset(gt: str, seed: int, kind: str = "D0"):
    """512 detected clones per generation 4/6/8/10 (same rng stream as the base study)."""
    gi = C.GTS.index(gt)
    return C.simulate(truth(gt, kind), N, np.random.default_rng([gi, seed, 0]))


# ---------------------------------------------------------------------- parametrisations
FREE = C.FULL
NODEATH = dl.Parametrisation([dl.RowSpec(0, [[1, 2, 3, 4, 5]], {0: 0.0}),
                              dl.RowSpec(1, [[1, 2, 3, 4, 5]], {0: 0.0})], "nodeath")  # fmt: skip
I_TOA = dl.ENTRIES["B"].index("to_A")


def pinned(par_name: str, c: float) -> dl.Parametrisation:
    """``par`` with ``B.to_A`` fixed at ``c`` (the low-B.to_A starts of ``C.fit_mle``)."""
    if par_name == "free":
        return dl.pinned_model(c)
    return dl.Parametrisation([dl.RowSpec(0, [[1, 2, 3, 4, 5]], {0: 0.0}),
                               dl.RowSpec(1, [[3, 4], [2, 5]], {0: 0.0, I_TOA: float(c)})],
                              f"nodeath_pinned={c:g}")  # fmt: skip


def fit_mle(lik, rng, model: str, n_random: int = 6, pins=(0.02, 0.3, 0.7)) -> dict:
    """Multistart MLE as ``misspecification.common.fit_mle``, for model free / nodeath."""
    par = FREE if model == "free" else NODEATH
    fix = (lambda P: P) if model == "free" else no_death  # noqa: E731
    starts = [("generic", dl.with_dediff(dl.GENERIC, 0.3, dl.COMPOSITIONS["mixed"]))]
    starts += [(f"random_{i}", dl.random_P(rng, h0=False)) for i in range(n_random)]
    res = []
    for name, P in starts:
        r = dl.maximise(lik, par, par.encode(fix(P)), C.LOOSE)
        res.append((r["ll"], name, r["s"]))
    Pb = par.P(max(res, key=lambda t: t[0])[2])[0]
    for c in pins:
        pin = pinned(model, c)
        rp = dl.maximise(lik, pin, pin.encode(fix(dl.with_to_a(Pb, c))), C.LOOSE)
        r = dl.maximise(lik, par, par.encode(rp["P"]), C.LOOSE)
        res.append((r["ll"], f"pin{c:g}", r["s"]))
    res.sort(key=lambda t: -t[0])
    best = None
    for _ll, name, s in res[:2]:
        r = dl.maximise(lik, par, s, C.TIGHT)
        if best is None or r["ll"] > best["ll"]:
            best = {**r, "start": name}
    lls = sorted([t[0] for t in res], reverse=True)
    return {"ll": best["ll"], "P": best["P"], "start": best["start"],
            "n_tied": int(sum(v > best["ll"] - 0.01 for v in lls)),
            "second_ll": float(lls[1])}  # fmt: skip
