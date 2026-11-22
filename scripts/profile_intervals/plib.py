"""Profile likelihood of linear functionals of one row of the two-type GW law.

A functional is ``psi(theta) = sum_i w_i p_i`` over the entries ``p`` of one row (A or B),
with integer weights ``w_i`` in {0, 1, 2}:

* a single probability ``p_i``: ``w = e_i`` (range [0, 1]);
* B's total division mass ``B.div_AA + B.div_AB + B.div_BB`` (range [0, 1]);
* ``m_BA = B.to_A + 2 B.div_AA + B.div_AB``, the mean number of A cells a B cell makes
  in one generation (range [0, 2]).

``ConstrainedModel`` maximises the exact log likelihood on ``{psi = c}``. The entries of the
constrained row are grouped by weight; with two groups (weights 0 and 1) their masses are
fixed, ``q1 = c``, ``q0 = 1 - c``; with three groups (weights 0, 1, 2) one coordinate
``u in [0, 1]`` sets ``q2 = t = lo + u (hi - lo)``, ``q1 = c - 2 t``, ``q0 = 1 - c + t``,
``t in [max(0, c - 1), c / 2]``. Within each group the entries share its mass through
stick-breaking coordinates; the other row keeps its 5 free sticks. Every constrained
model has 9 coordinates in [0, 1]. The gradient comes from the exact likelihood's own
derivative along ``dP`` (adjoint kernel), so no finite differences.

``profile`` traces ``l_p(c) = max {l(theta): psi(theta) = c}`` on an adaptive grid and
locates the threshold crossings; see its docstring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

LO, HI = 1e-9, 1 - 1e-9
CHI2_90, CHI2_95 = 2.705543454095404 / 2, 3.841458820694124 / 2  # 1.353, 1.921 nats
BOUND_90 = 1.6423744151498225 / 2  # chi2_1 0.80 quantile / 2 = 0.821 (50:50 mixture, 90%)
BOUND_95 = CHI2_90  # chi2_1 0.90 quantile / 2 = 1.353 (50:50 mixture, 95%)
THRESHOLDS = (BOUND_90, CHI2_90, CHI2_95)
STOP_DROP = 3.5


def sticks(s: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Stick-breaking: ``p (m,)`` summing to ``mass`` and ``dp/ds (m-1, m)``."""
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


def stick_encode(p: np.ndarray) -> np.ndarray:
    """Sticks of a non-negative vector (normalised first); ``m-1`` values."""
    p = np.clip(np.asarray(p, float), 1e-12, None)
    p = p / p.sum()
    rem = np.cumsum(p[::-1])[::-1]
    return p[:-1] / np.maximum(rem[:-1], 1e-300)


@dataclass(frozen=True)
class Functional:
    name: str
    row: str  # "A" or "B"
    weights: tuple[int, ...]  # per row entry, in the row's entry order
    cmax: float

    def value(self, phys: np.ndarray, names: list[str]) -> np.ndarray:
        """Psi of physical vector(s) ``(..., 12+)``."""
        idx = [names.index(f"{self.row}.{e}") for e in ROW_ENTRIES[self.row]]
        return np.asarray(phys)[..., idx] @ np.asarray(self.weights, float)


ROW_ENTRIES = {
    "A": ("death", "persist", "to_B", "div_AA", "div_AB", "div_BB"),
    "B": ("death", "to_A", "persist", "div_AA", "div_AB", "div_BB"),
}


def functionals() -> list[Functional]:
    out = []
    for row, ents in ROW_ENTRIES.items():
        for i, e in enumerate(ents):
            w = [0] * 6
            w[i] = 1
            out.append(Functional(f"{row}.{e}", row, tuple(w), 1.0))
    out.append(Functional("B.div", "B", (0, 0, 0, 1, 1, 1), 1.0))
    out.append(Functional("B.m_BA", "B", (0, 1, 0, 2, 1, 0), 2.0))
    return out


FUNCTIONALS = {f.name: f for f in functionals()}


class ConstrainedModel:
    """Exact log likelihood on ``{psi = c}`` (``func=None``: the full model, 10 sticks)."""

    def __init__(self, lik, func: Functional | None):
        self.lik = lik
        sim = lik.sim
        self.space, self.law = sim.space, sim.law
        self.names = list(sim.parameter_names)
        self.func = func
        self.rows = {}
        for r in self.space.rows:
            if r.n_sticks:
                assert not r.fixed and len(r.free) == len(r.entries), "all entries free"
                assert tuple(r.entries) == ROW_ENTRIES[r.prefix]
                self.rows[r.prefix] = r
        self.c = 0.0
        if func is None:
            self.groups = None
            self.dim = sum(r.n_sticks for r in self.rows.values())
        else:
            w = np.asarray(func.weights)
            self.levels = sorted(set(w.tolist()))
            assert self.levels in ([0, 1], [0, 1, 2]), self.levels
            self.groups = [np.flatnonzero(w == v) for v in self.levels]
            self.has_u = len(self.levels) == 3
            other = sum(r.n_sticks for p, r in self.rows.items() if p != func.row)
            self.dim = other + sum(len(g) - 1 for g in self.groups) + int(self.has_u)

    def set_c(self, c: float) -> None:
        self.c = float(np.clip(c, 0.0, self.func.cmax))

    # ------------------------------------------------------------- group masses
    def _t_range(self) -> tuple[float, float]:
        c = self.c
        return max(0.0, c - 1.0), c / 2.0

    def _masses(self, u: float) -> tuple[np.ndarray, np.ndarray]:
        """Group masses and their derivative in ``u``."""
        c = self.c
        if not self.has_u:
            return np.array([1.0 - c, c]), np.zeros(2)
        lo, hi = self._t_range()
        t = lo + u * (hi - lo)
        return np.array([1.0 - c + t, c - 2 * t, t]), (hi - lo) * np.array([1.0, -2.0, 1.0])

    # ------------------------------------------------------------- P and dP
    def _P(self, s: np.ndarray):
        nO = self.lik.n_outcomes_max
        P = np.zeros((len(self.law.types), nO))
        dP = np.zeros((self.dim, nO))
        rows = np.zeros(self.dim, dtype=np.int64)
        o = 0
        for prefix, r in self.rows.items():
            k = self.law.types.index(prefix)
            if self.func is None or prefix != self.func.row:
                m = r.n_sticks
                p, d = sticks(s[o : o + m], r.mass)
                P[k, :6] = p
                dP[o : o + m, :6] = d
                rows[o : o + m] = k
                o += m
                continue
            u = s[o] if self.has_u else 0.0
            ou = o
            o += int(self.has_u)
            q, dq = self._masses(u)
            q = np.clip(q, 0.0, None)
            for g, idx in enumerate(self.groups):
                m = len(idx) - 1
                p, d = sticks(s[o : o + m], 1.0)  # within-group shares
                P[k, idx] = q[g] * p
                if m:
                    dP[o : o + m][:, idx] = q[g] * d
                    rows[o : o + m] = k
                if self.has_u:
                    dP[ou, idx] = dq[g] * p
                o += m
            if self.has_u:
                rows[ou] = k
        return P, dP, rows

    def loglik(self, s: np.ndarray, grad: bool = True) -> tuple[float, np.ndarray]:
        P, dP, rows = self._P(np.clip(s, LO, HI))
        if not grad:
            dP, rows = dP[:0], rows[:0]
        return self.lik.log_likelihood_P(P, dP, rows, grad)

    def physical(self, s: np.ndarray) -> np.ndarray:
        P = self._P(np.clip(s, LO, HI))[0]
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

    def encode(self, phys: np.ndarray) -> np.ndarray:
        """Coordinates of the point of ``{psi = c}`` nearest (in shares) to ``phys``."""
        out = []
        for prefix in self.rows:
            row = np.asarray(phys[self.space.row_slice(prefix)], float)
            if self.func is None or prefix != self.func.row:
                out.extend(stick_encode(row))
                continue
            if self.has_u:
                Q2 = float(np.clip(row[self.groups[2]].sum(), 0, None))
                lo, hi = self._t_range()
                out.append(0.5 if hi - lo < 1e-12 else (np.clip(Q2, lo, hi) - lo) / (hi - lo))
            for idx in self.groups:
                if len(idx) > 1:
                    out.extend(stick_encode(row[idx]))
        return np.clip(np.asarray(out, float), 1e-6, 1 - 1e-6)

    def maximise(self, start: np.ndarray, maxiter: int = 2000, tight: bool = True) -> dict:
        """L-BFGS-B; ``tight``: ftol 1e-15 / gtol 1e-9 (the scipy defaults stop up to
        ~0.2 nats short at n = 4096), else the scipy defaults (coarse scans only).
        """
        from scipy.optimize import minimize

        def neg(s):
            v, g = self.loglik(s)
            return -v, -g

        r = minimize(neg, start, jac=True, method="L-BFGS-B", bounds=[(LO, HI)] * self.dim,
                     options={"maxiter": maxiter, **({"ftol": 1e-15, "gtol": 1e-9} if tight
                                                     else {})})  # fmt: skip
        return {"ll": float(-r.fun), "s": r.x, "nfev": int(r.nfev), "ok": bool(r.success)}


# ---------------------------------------------------------------------------- profile
@dataclass
class Curve:
    """Profile points (unsorted while building)."""

    c: list = field(default_factory=list)
    ll: list = field(default_factory=list)
    phys: list = field(default_factory=list)
    fits: int = 0

    def add(self, c: float, ll: float, phys: np.ndarray) -> None:
        for i, ci in enumerate(self.c):
            if abs(ci - c) < 1e-12:
                if ll > self.ll[i]:
                    self.ll[i], self.phys[i] = ll, phys
                return
        self.c.append(float(c))
        self.ll.append(float(ll))
        self.phys.append(np.asarray(phys))

    def sorted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        o = np.argsort(self.c)
        return np.asarray(self.c)[o], np.asarray(self.ll)[o], np.asarray(self.phys)[o]

    def get(self, c: float) -> tuple[float, np.ndarray] | None:
        for i, ci in enumerate(self.c):
            if abs(ci - c) < 1e-12:
                return self.ll[i], self.phys[i]
        return None


def project(phys: np.ndarray, model: ConstrainedModel, c: float) -> np.ndarray:
    model.set_c(c)
    return model.encode(phys)


class Profiler:
    """Profile of one functional on one dataset.

    ``draws`` (k, 15) physical posterior draws with log likelihoods ``draws_ll`` (extra
    starts: the two best draws with ``|psi - c| < window`` projected onto ``psi = c``).
    """

    def __init__(self, lik, func: Functional, draws: np.ndarray, draws_ll: np.ndarray,
                 n_draw_starts: int = 2):  # fmt: skip
        self.lik, self.func = lik, func
        self.model = ConstrainedModel(lik, func)
        self.draws, self.draws_ll = draws, draws_ll
        self.draws_psi = func.value(draws, self.model.names)
        self.n_draw_starts = n_draw_starts
        self.curve = Curve()
        self.log: list[str] = []

    def fit(self, c: float, starts: list[np.ndarray], window: float = 0.0,
            n_draws: int | None = None, tight: bool = True) -> tuple[float, np.ndarray]:  # fmt: skip
        """Best fit on ``psi = c`` from ``starts`` (+ nearby draws); added to the curve."""
        m = self.model
        n_draws = self.n_draw_starts if n_draws is None else n_draws
        starts = list(starts)
        if n_draws and window > 0:
            near = np.flatnonzero(np.abs(self.draws_psi - c) < window)
            starts += [self.draws[i] for i in near[np.argsort(self.draws_ll[near])[::-1][:n_draws]]]
        best = (-np.inf, None)
        for st in starts:
            m.set_c(c)
            r = m.maximise(m.encode(st), tight=tight)
            self.curve.fits += 1
            if r["ll"] > best[0]:
                best = (r["ll"], m.physical(r["s"]))
        self.curve.add(c, *best)
        return self.curve.get(c)

    def _nearest_phys(self, c: float) -> list[np.ndarray]:
        cs, _, ph = self.curve.sorted()
        out = []
        below, above = cs[cs < c], cs[cs > c]
        if len(below):
            out.append(ph[np.flatnonzero(cs < c)[-1]])
        if len(above):
            out.append(ph[np.flatnonzero(cs > c)[0]])
        return out

    # ------------------------------------------------------------------ passes
    def outward(self, c0: float, phys0: np.ndarray, ll_ref: float, h0: float) -> None:
        """Main pass from ``c0`` both ways (adaptive step; stop past STOP_DROP nats),
        then a coarse continuation (step cmax/20, warm start + 1 draw) to the boundary.
        """
        cmax = self.func.cmax
        ll0, ph0 = self.fit(c0, [phys0], window=max(h0, 0.01 * cmax))
        hmin, hmax = 0.002 * cmax, 0.05 * cmax
        for sgn in (+1, -1):
            c, ll, ph, h = c0, ll0, ph0, float(np.clip(h0, hmin, hmax))
            done = False
            while not done:
                cn = c + sgn * h
                if cn >= cmax - 1e-9 or cn <= 1e-9:
                    cn, done = (cmax if sgn > 0 else 0.0), True
                lln, phn = self.fit(cn, [ph], window=max(h, 0.01 * cmax))
                d_new = ll_ref - lln
                delta = abs(lln - ll)
                c, ll, ph = cn, lln, phn
                if d_new > STOP_DROP:
                    break
                h *= float(np.clip(np.sqrt(0.5 / max(delta, 1e-3)), 0.5, 2.5))
                h = float(np.clip(h, hmin, hmax))
            if done:
                continue
            # coarse continuation to the boundary (safety net for second modes)
            step = 0.05 * cmax
            while True:
                cn = c + sgn * step
                last = cn >= cmax - 1e-9 or cn <= 1e-9
                if last:
                    cn = cmax if sgn > 0 else 0.0
                lln, phn = self.fit(cn, [ph], window=step / 2, n_draws=1, tight=False)
                c, ph = cn, phn
                if last:
                    break

    def sweep(self, ll_ref: float, tol: float = 1e-3, far: float = 8.0) -> int:
        """Warm-start sweep outward from the best point (points whose drop and whose
        neighbour's drop both exceed ``far`` nats are skipped); returns #points improved.
        """
        cs, lls, phs = self.curve.sorted()
        jb = int(np.argmax(lls))
        improved = 0
        for order, nb in ((range(jb + 1, len(cs)), -1), (range(jb - 1, -1, -1), +1)):
            for j in order:
                cs, lls, phs = self.curve.sorted()
                before = lls[j]
                if min(ll_ref - before, ll_ref - lls[j + nb]) > far:
                    continue
                ll, _ = self.fit(cs[j], [phs[j + nb]], n_draws=0)
                if ll > before + tol:
                    improved += 1
                    if ll > before + 0.01:
                        self.log.append(f"sweep c={cs[j]:.4f}: +{ll - before:.3f}")
        return improved

    def smooth(self, ll_ref: float, far: float = 6.0, gap: float = 0.02,
               passes: int = 4) -> int:  # fmt: skip
        """Refit every point (drop < ``far``) from each neighbour that is higher by more
        than ``gap`` nats, in both directions, until nothing improves by > 1e-3 nats.

        Repairs stale points next to a branch switch (a jump in the curve). Returns the
        number of improved points.
        """
        total = 0
        for _ in range(passes):
            improved = 0
            cs, lls, phs = self.curve.sorted()
            for j in range(len(cs)):
                if ll_ref - lls[j] > far:
                    continue
                for k in (j - 1, j + 1):
                    if 0 <= k < len(cs) and lls[k] > lls[j] + gap:
                        before = lls[j]
                        ll, ph = self.fit(cs[j], [phs[k]], n_draws=0)
                        if ll > before + 1e-3:
                            improved += 1
                            lls[j], phs[j] = ll, ph
                            if ll > before + 0.01:
                                self.log.append(f"smooth c={cs[j]:.4f}: +{ll - before:.3f}")
            total += improved
            if not improved:
                break
        return total

    def fill(self, ll_ref: float, max_new: int = 40) -> int:
        """Insert midpoints where the drop jumps (>0.75 nats) inside the relevant region
        (either end below STOP_DROP + 1); warm starts from both neighbours.
        """
        new = 0
        hmin = 0.002 * self.func.cmax
        while new < max_new:
            cs, lls, phs = self.curve.sorted()
            d = ll_ref - lls
            cand = [j for j in range(len(cs) - 1)
                    if min(d[j], d[j + 1]) < STOP_DROP + 1 and abs(d[j] - d[j + 1]) > 0.75
                    and cs[j + 1] - cs[j] > 2 * hmin]  # fmt: skip
            if not cand:
                break
            for j in cand[: max_new - new]:
                cm = 0.5 * (cs[j] + cs[j + 1])
                self.fit(cm, [phs[j], phs[j + 1]], window=(cs[j + 1] - cs[j]) / 2, n_draws=1)
                new += 1
        return new

    def crossings(self, ll_ref: float, thresholds=THRESHOLDS, tol: float = 0.01,
                  max_iter: int = 6) -> int:  # fmt: skip
        """Refine every crossing of ``drop = thr`` (secant / bisection with refits)."""
        n = 0
        xtol = 1e-5 * self.func.cmax
        for thr in thresholds:
            for _ in range(20):  # crossings may appear while refining
                cs, lls, phs = self.curve.sorted()
                d = ll_ref - lls - thr
                todo = [j for j in range(len(cs) - 1)
                        if d[j] * d[j + 1] < 0 and min(abs(d[j]), abs(d[j + 1])) > tol
                        and cs[j + 1] - cs[j] > xtol]  # fmt: skip
                if not todo:
                    break
                j = todo[0]
                a, b, da, db = cs[j], cs[j + 1], d[j], d[j + 1]
                pa, pb = phs[j], phs[j + 1]
                for it in range(max_iter):
                    cm = a - da * (b - a) / (db - da)
                    if it % 2 and it:  # alternate with bisection for safety
                        cm = 0.5 * (a + b)
                    ll, pm = self.fit(cm, [pa, pb], n_draws=0)
                    n += 1
                    dm = ll_ref - ll - thr
                    if abs(dm) < tol or b - a < xtol:
                        break
                    if dm * da < 0:
                        b, db, pb = cm, dm, pm
                    else:
                        a, da, pa = cm, dm, pm
        return n


def intervals(
    c: np.ndarray, drop: np.ndarray, thr: float, cmax: float
) -> list[tuple[float, float]]:
    """``{c : drop(c) <= thr}`` with linear interpolation between grid points (sorted c)."""
    inside = drop <= thr
    segs = []
    j = 0
    while j < len(c):
        if not inside[j]:
            j += 1
            continue
        k = j
        while k + 1 < len(c) and inside[k + 1]:
            k += 1
        if j == 0:
            lo = c[0]
        else:
            lo = c[j - 1] + (thr - drop[j - 1]) * (c[j] - c[j - 1]) / (drop[j] - drop[j - 1])
        if k == len(c) - 1:
            hi = c[-1]
        else:
            hi = c[k] + (thr - drop[k]) * (c[k + 1] - c[k]) / (drop[k + 1] - drop[k])
        segs.append((float(lo), float(hi)))
        j = k + 1
    return segs


def run_profile(lik, func: Functional, centres: list[np.ndarray], ll_ref: float,
                draws: np.ndarray, draws_ll: np.ndarray, h0: float,
                init: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                points: list[tuple[float, float, np.ndarray]] = ()) -> dict:  # fmt: skip
    """Full profile of ``func``: outward pass from each centre (``init``: points of an
    earlier curve to merge; ``points``: extra ``(c, l, theta)`` fits started at theta,
    see :func:`cross_violations`), warm-start sweeps, fill, crossing refinement.
    """
    t0 = time.perf_counter()
    e0 = lik.evals
    pr = Profiler(lik, func, draws, draws_ll)
    names = pr.model.names
    if init is not None:
        for c, ll, ph in zip(*init, strict=True):
            pr.curve.add(c, ll, ph)
    for ph in centres:
        pr.outward(float(func.value(ph, names)), ph, ll_ref, h0)
    for c, _l, ph in points:
        pr.fit(c, [ph], n_draws=0)
    passes = 0
    for _ in range(3):
        passes += 1
        if pr.sweep(ll_ref) == 0:
            break
    ll_max = max(ll_ref, max(pr.curve.ll))
    pr.fill(ll_max)
    pr.sweep(ll_max)
    ll_max = max(ll_ref, max(pr.curve.ll))
    # crossings add points (possibly on another branch): smooth and re-locate until stable
    for _ in range(4):
        pr.crossings(ll_max)
        if pr.smooth(ll_max) == 0:
            break
        ll_max = max(ll_ref, max(pr.curve.ll))
    pr.crossings(ll_max)
    c, ll, phys = pr.curve.sorted()
    return {"c": c, "ll": ll, "phys": phys, "fits": pr.curve.fits, "evals": lik.evals - e0,
            "seconds": time.perf_counter() - t0, "log": pr.log, "sweep_passes": passes}  # fmt: skip


def cross_violations(func: Functional, own: tuple[np.ndarray, np.ndarray],
                     others: list[tuple[np.ndarray, np.ndarray]], names: list[str],
                     ll_ref: float, max_drop: float = 5.0,
                     tol: float = 0.02) -> list[tuple[float, float, np.ndarray]]:  # fmt: skip
    """Points of other profiles that beat this profile at their own ``psi`` value.

    Every point ``theta`` of another profile of the same run is a feasible point of
    ``{psi = psi(theta)}``, so ``l_p(psi(theta)) >= l(theta)``. Returns ``(c, l, theta)``
    for the points (drop < ``max_drop``) where the curve ``own = (c, l)``, linearly
    interpolated, is lower by more than ``tol`` nats (one per 1e-4 * cmax bin, best kept).
    """
    c_own, ll_own = own
    out: dict[int, tuple[float, float, np.ndarray]] = {}
    for ll, phys in others:
        keep = ll > ll_ref - max_drop
        if not keep.any():
            continue
        cv = func.value(phys[keep], names)
        interp = np.interp(cv, c_own, ll_own)
        for c, l_pt, ph, li in zip(cv, ll[keep], phys[keep], interp, strict=True):
            if l_pt > li + tol:
                b = int(round(c / (1e-4 * func.cmax)))
                if b not in out or out[b][1] < l_pt:
                    out[b] = (float(np.clip(c, 0, func.cmax)), float(l_pt), ph)
    return sorted(out.values(), key=lambda t: t[0])
