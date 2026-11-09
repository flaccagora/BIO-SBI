"""A small No-U-Turn sampler (NUTS) with Stan-style warmup, plus R-hat / ESS.

Multinomial NUTS (Hoffman & Gelman 2014; Betancourt 2017) on an unconstrained space:
trajectory doubling with biased progressive sampling between subtrees, uniform
(multinomial) sampling within, the generalised no-U-turn criterion on the summed
momentum, divergence detection (energy error > 1000). Warmup adapts the step size by
dual averaging (target acceptance ``target_accept``) and a dense or diagonal inverse
mass matrix over Stan's expanding windows (initial buffer, doubling windows, terminal
buffer). Every chain owns a ``numpy.random.Generator``; chains share nothing, so they
can run in threads when the log-density releases the GIL (Numba ``nogil`` kernels).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

#: ``logp_and_grad(u) -> (log density, gradient)``; ``-inf`` outside the support.
LogDensity = Callable[[np.ndarray], tuple[float, np.ndarray]]

LOG = logging.getLogger(__name__)


@dataclass
class ChainResult:
    """Draws and diagnostics of one chain."""

    draws: np.ndarray  # (draws, d) unconstrained
    logp: np.ndarray  # (draws,)
    step_size: float
    inv_mass: np.ndarray
    divergences: int
    mean_tree_depth: float
    max_tree_depth_hits: int
    gradient_evals: int
    seconds: float
    accept_stat: float
    warmup_divergences: int = 0
    extra: dict = field(default_factory=dict)


class _Sampler:
    def __init__(self, logp_and_grad: LogDensity, inv_mass: np.ndarray, max_depth: int, rng):
        self.f = logp_and_grad
        self.max_depth = int(max_depth)
        self.rng = rng
        self.evals = 0
        self.set_mass(inv_mass)

    def set_mass(self, inv_mass: np.ndarray) -> None:
        inv_mass = np.asarray(inv_mass, dtype=float)
        if inv_mass.ndim == 1:
            inv_mass = np.diag(inv_mass)
        self.inv_mass = inv_mass
        self.chol_mass = np.linalg.cholesky(np.linalg.inv(inv_mass))  # p ~ N(0, M)

    def grad(self, q):
        self.evals += 1
        lp, g = self.f(q)
        if not np.isfinite(lp) or not np.all(np.isfinite(g)):
            return -np.inf, np.zeros_like(q)
        return lp, g

    def kinetic(self, p):
        return 0.5 * p @ self.inv_mass @ p

    def leapfrog(self, q, p, g, eps):
        p = p + 0.5 * eps * g
        q = q + eps * (self.inv_mass @ p)
        lp, g = self.grad(q)
        p = p + 0.5 * eps * g
        return q, p, g, lp

    def turning(self, rho, p_left, p_right):
        return (rho @ (self.inv_mass @ p_left) <= 0) or (rho @ (self.inv_mass @ p_right) <= 0)

    def build(self, q, p, g, v, depth, eps, H0):
        """Return a subtree dict built from ``(q, p, g)`` in direction ``v``."""
        if depth == 0:
            q1, p1, g1, lp1 = self.leapfrog(q, p, g, v * eps)
            H = -lp1 + self.kinetic(p1) if np.isfinite(lp1) else np.inf
            dH = H0 - H if np.isfinite(H) else -np.inf
            return {
                "left": (q1, p1, g1), "right": (q1, p1, g1), "prop": (q1, lp1, g1),
                "logw": dH, "rho": p1.copy(), "turn": False, "div": (-dH) > 1000,
                "acc": min(1.0, math.exp(min(dH, 0.0))), "n": 1,
            }  # fmt: skip
        t1 = self.build(q, p, g, v, depth - 1, eps, H0)
        if t1["turn"] or t1["div"]:
            return t1
        edge = t1["right"] if v > 0 else t1["left"]
        t2 = self.build(*edge, v, depth - 1, eps, H0)
        acc, n = t1["acc"] + t2["acc"], t1["n"] + t2["n"]
        if t2["turn"] or t2["div"]:
            return {**t1, "turn": t2["turn"], "div": t2["div"], "acc": acc, "n": n}
        logw = np.logaddexp(t1["logw"], t2["logw"])
        prop = t2["prop"] if math.log(self.rng.random()) < t2["logw"] - logw else t1["prop"]
        left, right = (t1["left"], t2["right"]) if v > 0 else (t2["left"], t1["right"])
        rho = t1["rho"] + t2["rho"]
        return {
            "left": left, "right": right, "prop": prop, "logw": logw, "rho": rho,
            "turn": self.turning(rho, left[1], right[1]), "div": False, "acc": acc, "n": n,
        }  # fmt: skip

    def transition(self, q, lp, g, eps):
        """One NUTS transition; returns ``(q, lp, g, accept_stat, depth, divergent)``."""
        p0 = self.chol_mass @ self.rng.standard_normal(len(q))
        H0 = -lp + self.kinetic(p0)
        left = right = (q, p0, g)
        prop, logw, rho = (q, lp, g), 0.0, p0.copy()
        acc, n, depth, div = 0.0, 0, 0, False
        for depth in range(self.max_depth):
            v = 1 if self.rng.random() < 0.5 else -1
            edge = right if v > 0 else left
            t = self.build(*edge, v, depth, eps, H0)
            acc, n = acc + t["acc"], n + t["n"]
            if t["div"]:
                div = True
                break
            if t["turn"]:
                break
            if math.log(self.rng.random()) < t["logw"] - logw:
                prop = t["prop"]
            logw = np.logaddexp(logw, t["logw"])
            if v > 0:
                right = t["right"]
            else:
                left = t["left"]
            rho = rho + t["rho"]
            if self.turning(rho, left[1], right[1]):
                break
        return (*prop, acc / max(n, 1), depth + 1, div)


def _find_step_size(s: _Sampler, q, lp, g, eps=0.1):
    """Heuristic initial step size (Hoffman & Gelman 2014, Algorithm 4)."""
    p = s.chol_mass @ s.rng.standard_normal(len(q))
    H0 = -lp + s.kinetic(p)

    def log_ratio(e):
        _, p1, _, lp1 = s.leapfrog(q, p, g, e)
        return (H0 - (-lp1 + s.kinetic(p1))) if np.isfinite(lp1) else -np.inf

    r = log_ratio(eps)
    direction = 1 if r > math.log(0.5) else -1
    for _ in range(50):
        r = log_ratio(eps)
        if (direction == 1 and not r > math.log(0.5)) or (direction == -1 and r > math.log(0.5)):
            break
        eps = eps * 2.0 if direction == 1 else eps / 2.0
    return eps


def _windows(warmup: int) -> list[int]:
    """End iterations of the mass-matrix adaptation windows (Stan's schedule)."""
    if warmup < 20:
        return []
    init, term, base = 75, 50, 25
    if init + term + base > warmup:
        init, term = int(0.15 * warmup), int(0.1 * warmup)
        base = warmup - init - term
    ends, start, size = [], init, base
    while start < warmup - term:
        end = start + size
        if end + 2 * size > warmup - term:  # last window absorbs the remainder
            end = warmup - term
        ends.append(end)
        start, size = end, size * 2
    return ends


def run_chain(
    logp_and_grad: LogDensity,
    q0: np.ndarray,
    warmup: int,
    draws: int,
    rng: np.random.Generator,
    max_depth: int = 8,
    target_accept: float = 0.8,
    dense: bool = True,
    inv_mass: np.ndarray | None = None,
    step_size: float | None = None,
    name: str = "chain",
    log_every: int = 0,
) -> ChainResult:
    """Warm up and sample one chain from ``q0`` (unconstrained coordinates).

    ``log_every > 0`` logs progress (iteration, step size, tree depth, log density) at
    INFO level every that many iterations.
    """
    t0 = time.perf_counter()
    d = len(q0)
    s = _Sampler(logp_and_grad, np.eye(d) if inv_mass is None else inv_mass, max_depth, rng)
    q = np.asarray(q0, dtype=float)
    lp, g = s.grad(q)
    if not np.isfinite(lp):
        raise ValueError("Initial point has zero posterior density")
    eps = step_size or _find_step_size(s, q, lp, g)
    # dual averaging state
    mu, log_eps_bar, h_bar, gamma, t0_da, kappa = math.log(10 * eps), 0.0, 0.0, 0.05, 10, 0.75
    ends, window = set(_windows(warmup)), []
    window_start = sorted(_windows(warmup))[0] if ends else warmup
    warm_div, da = 0, 0
    for it in range(warmup):
        q, lp, g, acc, depth, div = s.transition(q, lp, g, eps)
        warm_div += div
        if log_every and (it + 1) % log_every == 0:
            LOG.info(
                "%s warmup %d/%d: step %.3g depth %d logp %.1f evals %d (%.0fs)",
                name,
                it + 1,
                warmup,
                eps,
                depth,
                lp,
                s.evals,
                time.perf_counter() - t0,
            )
        da += 1
        m = da
        h_bar = (1 - 1 / (m + t0_da)) * h_bar + (target_accept - acc) / (m + t0_da)
        log_eps = mu - math.sqrt(m) / gamma * h_bar
        w = m ** (-kappa)
        log_eps_bar = w * log_eps + (1 - w) * log_eps_bar
        eps = math.exp(log_eps)
        if ends and window_start <= it:
            window.append(q.copy())
        if it + 1 in ends and len(window) >= 3:
            x = np.asarray(window)
            n = len(x)
            if dense:
                cov = np.cov(x, rowvar=False).reshape(d, d)
                cov = (n / (n + 5.0)) * cov + 1e-3 * (5.0 / (n + 5.0)) * np.eye(d)
            else:
                cov = (n / (n + 5.0)) * x.var(0) + 1e-3 * (5.0 / (n + 5.0))
            s.set_mass(cov)
            window = []
            eps = _find_step_size(s, q, lp, g, eps)
            mu, log_eps_bar, h_bar, da = math.log(10 * eps), 0.0, 0.0, 0  # restart
    if warmup > 0:
        eps = math.exp(log_eps_bar) if log_eps_bar != 0.0 else eps
    out = np.empty((draws, d))
    out_lp = np.empty(draws)
    depths, divs, hits, accs = [], 0, 0, []
    for i in range(draws):
        q, lp, g, acc, depth, div = s.transition(q, lp, g, eps)
        out[i], out_lp[i] = q, lp
        depths.append(depth)
        divs += div
        hits += depth >= s.max_depth
        accs.append(acc)
        if log_every and (i + 1) % log_every == 0:
            LOG.info(
                "%s draw %d/%d: step %.3g depth %d logp %.1f evals %d (%.0fs)",
                name,
                i + 1,
                draws,
                eps,
                depth,
                lp,
                s.evals,
                time.perf_counter() - t0,
            )
    return ChainResult(
        draws=out, logp=out_lp, step_size=eps, inv_mass=s.inv_mass, divergences=int(divs),
        mean_tree_depth=float(np.mean(depths)) if depths else float("nan"),
        max_tree_depth_hits=int(hits), gradient_evals=int(s.evals),
        seconds=time.perf_counter() - t0, accept_stat=float(np.mean(accs)) if accs else float("nan"),
        warmup_divergences=int(warm_div),
    )  # fmt: skip


# ---------------------------------------------------------------------- diagnostics
def _autocov(x: np.ndarray) -> np.ndarray:
    n = len(x)
    f = np.fft.rfft(x - x.mean(), n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / n


def ess(chains: np.ndarray) -> float:
    """Effective sample size of ``(chains, draws)`` (Geyer's initial monotone sequence)."""
    m, n = chains.shape
    if n < 4:
        return float("nan")
    acov = np.array([_autocov(c) for c in chains])
    chain_mean = chains.mean(1)
    w = acov[:, 0].mean() * n / (n - 1)
    b = n * chain_mean.var(ddof=1) if m > 1 else 0.0
    var_plus = w * (n - 1) / n + b / n
    if var_plus <= 0:
        return float(m * n)
    rho = 1 - (w - acov.mean(0)) / var_plus
    rho[0] = 1.0
    total, prev = 0.0, np.inf
    t = 0
    while t + 1 < n:
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        pair = min(pair, prev)
        total += pair
        prev = pair
        t += 2
    tau = -1 + 2 * total
    return float(m * n / max(tau, 1.0 / math.log10(max(m * n, 10))))


def split_rhat(chains: np.ndarray) -> float:
    """Split-R-hat of ``(chains, draws)``."""
    m, n = chains.shape
    half = n // 2
    if half < 2:
        return float("nan")
    x = np.concatenate([chains[:, :half], chains[:, half : 2 * half]])
    w = x.var(1, ddof=1).mean()
    b = half * x.mean(1).var(ddof=1)
    if w <= 0:
        return float("nan") if b > 0 else 1.0
    return float(math.sqrt(((half - 1) / half * w + b / half) / w))
