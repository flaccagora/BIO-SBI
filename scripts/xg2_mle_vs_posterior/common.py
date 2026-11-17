"""Shared helpers: run lookup, exact likelihood of a run, likelihood on sub-models.

``SubModel`` maximises the exact log likelihood with some free entries of a row pinned
(e.g. ``B.to_A = c`` for a profile, or ``B.div_* = 0`` for the "B never divides" model).
The remaining free entries of each row share the leftover mass through stick-breaking
coordinates in [0, 1]; the gradient comes from the exact likelihood's own forward-mode
Jacobian (``gw_exact.log_likelihood`` with our own ``dP``), so no finite differences.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
ROOT = Path("/home/matteo/BIO-SBI/results/exact_gw/data_size")
OUT = REPO / "docs" / "xg2_mle_vs_posterior"
LO, HI = 1e-9, 1 - 1e-9


def run_key(path: Path) -> tuple[str, int, int] | None:
    """(gt, seed, n) of a run folder (T5 runs: gt from ``+truths.T5_*``)."""
    s = path.name
    seed = re.search(r"seed=(\d+)", s)
    gt = re.search(r"gt=([A-Za-z0-9_]+)", s) or re.search(r"truths\.(T5_[A-Za-z0-9]+)", s)
    if gt is None:
        return None
    from omegaconf import OmegaConf

    ov = list(OmegaConf.load(path / ".hydra" / "overrides.yaml"))
    n = next((int(o.split("=")[1]) for o in ov if o.startswith("design.clones=")), 128)
    seed = next((int(o.split("=")[1]) for o in ov if o.startswith("seed=")), 0)
    return gt.group(1), seed, n


def all_runs() -> dict[tuple[str, int, int], Path]:
    out = {}
    for m in sorted(ROOT.glob("*/*/metrics.json")):
        if "aborted" in str(m):
            continue
        k = run_key(m.parent)
        if k is not None:
            out[k] = m.parent
    return out


def setup(run: Path):
    """``(cfg, sim, lik, names, truth, post)`` via scripts/exact_gw_mle.py::_setup."""
    from exact_gw_mle import _setup

    return _setup(run)


def _sticks(s: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Stick-breaking: ``p (m,)`` and ``dp/ds (m-1, m)`` for ``m-1`` sticks."""
    m = len(s) + 1
    tails = np.concatenate([[1.0], np.cumprod(1 - s)])
    p = np.empty(m)
    p[:-1] = mass * s * tails[:-1]
    p[-1] = mass * tails[-1]
    d = np.zeros((m - 1, m))
    for j in range(m - 1):
        d[j, j] = mass * tails[j]
        for i in range(j + 1, m):
            d[j, i] = -p[i] / max(1 - s[j], 1e-300)
    return p, d


class SubModel:
    """Exact likelihood with pinned entries ``{"B.to_A": 0.4, "B.div_AA": 0.0, ...}``."""

    def __init__(self, lik, pinned: dict[str, float]):
        from bio_sbi.inference.gw_exact import log_likelihood

        self._ll = log_likelihood
        self.lik = lik
        sim = lik.sim
        self.space, self.law = sim.space, sim.law
        self.names = list(sim.parameter_names)
        self.set_pinned(pinned)

    def set_pinned(self, pinned: dict[str, float]) -> None:
        self.pinned = dict(pinned)
        self.blocks = []  # (type index, free entry idx list, pinned {idx: v}, mass)
        for r in self.space.rows:
            if not r.n_sticks:
                continue
            k = self.law.types.index(r.prefix)
            pin = {r.entries.index(n.split(".")[1]): v for n, v in pinned.items()
                   if n.split(".")[0] == r.prefix}  # fmt: skip
            free = [i for i in r.free if i not in pin]
            self.blocks.append((k, free, pin, r.mass - sum(pin.values())))
        self.dim = sum(len(f) - 1 for _, f, _, _ in self.blocks)

    def physical(self, s: np.ndarray) -> np.ndarray:
        P = self._P(s)[0]
        phys = np.zeros(len(self.names))
        for k, t in enumerate(self.law.types):
            sl = self.space.row_slice(t)
            phys[sl] = P[k, : sl.stop - sl.start]
        for t in ("founder", "founder_size"):
            r = next(r for r in self.space.rows if r.prefix == t)
            sl = self.space.row_slice(t)
            for i, v in r.fixed.items():
                phys[sl.start + i] = v
            if len(r.free) == 1:
                phys[sl.start + r.free[0]] = r.mass
        return phys

    def _P(self, s: np.ndarray):
        P = np.zeros((len(self.law.types), self.lik.n_outcomes_max))
        dP = np.zeros((self.dim, self.lik.n_outcomes_max))
        rows = np.zeros(self.dim, dtype=np.int64)
        o = 0
        for k, free, pin, mass in self.blocks:
            m = len(free)
            p, d = _sticks(s[o : o + m - 1], mass)
            P[k, free] = p
            for i, v in pin.items():
                P[k, i] = v
            dP[o : o + m - 1][:, free] = d
            rows[o : o + m - 1] = k
            o += m - 1
        return P, dP, rows

    def loglik(self, s: np.ndarray, grad: bool = True):
        P, dP, rows = self._P(np.clip(s, LO, HI))
        if not grad:
            dP, rows = dP[:0], rows[:0]
        lk = self.lik
        v, g = self._ll(lk.model, P, dP, rows, lk.data, lk.floor, lk.min_detection)
        lk.evals += 1
        return v, g

    def encode(self, phys: np.ndarray) -> np.ndarray:
        """Sticks of a physical vector, its pinned entries' mass taken from the others."""
        out = []
        for k, free, pin, mass in self.blocks:
            sl = self.space.row_slice(self.law.types[k])
            row = np.clip(phys[sl][free], 1e-12, None)
            row = row / row.sum()
            rem = np.cumsum(row[::-1])[::-1]
            out.extend(row[:-1] / np.maximum(rem[:-1], 1e-300))
        return np.clip(np.asarray(out), 1e-6, 1 - 1e-6)

    def maximise(self, start: np.ndarray, maxiter: int = 400) -> dict:
        from scipy.optimize import minimize

        def neg(s):
            v, g = self.loglik(s)
            return -v, -g

        r = minimize(neg, start, jac=True, method="L-BFGS-B", bounds=[(LO, HI)] * self.dim,
                     options={"maxiter": maxiter})  # fmt: skip
        return {"ll": float(-r.fun), "s": r.x, "nfev": int(r.nfev), "ok": bool(r.success)}
