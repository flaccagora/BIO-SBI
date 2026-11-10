"""Adaptive tempered SMC sampler for expensive, low-dimensional posteriors.

Used by :mod:`bio_sbi.inference.exact_bh` (one likelihood evaluation costs ~0.01-5 s), but
generic: it needs a batched log target ``logpost(U) -> (n,)`` on ``R^d`` and a tractable
reference distribution ``q`` (sampler + normalised log density).

Algorithm (waste-free tempered SMC, Dau & Chopin 2022, with an adaptive schedule):

* path ``gamma_beta(u) ∝ q(u)^(1 - beta) post(u)^beta``, ``beta: 0 -> 1``; with ``q`` =
  the prior this is likelihood tempering, with ``q`` a Laplace / t approximation it is
  the geometric path from the approximation (few steps);
* ``N = chains x steps`` particles; the next ``beta`` is the largest one keeping the
  Kish ESS of the incremental weights at ``ess_fraction x N`` (bisection);
* move: ``chains`` starting points resampled (systematic) from the weighted population,
  each extended by ``steps - 1`` Metropolis-Hastings steps; all ``chains x steps`` states
  form the next (equally weighted) population. The kernel is a random mixture of a
  random walk ``N(u, s^2 Sigma)`` (``Sigma`` = weighted particle covariance, scale ``s``
  tuned between stages towards ``rw_acceptance``, optional log-normal scale jitter
  ``rw_scale_sd``), an independence proposal (t, or a BIC-selected t mixture, fitted to
  the population; share ``imh_prob`` adapted to twice its acceptance) and optional
  differential-evolution moves with differences of the stage population (``de_prob``);
  all leave ``gamma_beta`` invariant;
* at ``beta = 1`` the ``chains`` chains run on until, for every coordinate, the
  rank-normalised bulk ESS ``>= target_ess``, the tail ESS ``>= tail_fraction x
  target_ess`` and the nested R-hat ``<= max_rhat`` (Vehtari, Gelman, Simpson, Carpenter
  & Buerkner 2021; Margossian et al. 2024; ``superchains`` contiguous groups of chains,
  i.e. starts from different strata of the population), checked every ``check_every``
  steps after the first ``final_burn`` of the chains, or until ``max_evals`` /
  ``max_final_steps``. Until ``adapt_final_steps`` the random-walk covariance and scale
  and the independence proposal are refitted from the pooled chains at every check
  (adaptive Metropolis: the population can be narrower than the posterior). Output =
  the final-stage states after the burn-in (equal weights).

:func:`smc_islands` pools independent tempered runs instead (one superchain per island).

Every MH step is one batch of ``chains`` evaluations (evaluated in parallel by the
caller). Diagnostics: temperature schedule, per-stage ESS / Pareto ``k`` of the
incremental weights, acceptance rates, log evidence estimate, final R-hat / ESS.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp, ndtri
from scipy.stats import rankdata

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------- diagnostics
def _kish(lw: np.ndarray) -> float:
    lw = np.asarray(lw, float)
    lw = lw[np.isfinite(lw)]
    if not len(lw):
        return 0.0
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (w**2).sum())


def _split(x: np.ndarray) -> np.ndarray:
    """``(chains, n)`` -> ``(2 chains, n // 2)`` (first and second halves)."""
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, x.shape[1] - n :]], axis=0)


def _rank_normal(x: np.ndarray) -> np.ndarray:
    r = rankdata(x.ravel(), method="average").reshape(x.shape)
    return ndtri((r - 0.375) / (x.size + 0.25))


def _rhat_basic(x: np.ndarray) -> float:
    m, n = x.shape
    if n < 2:
        return float("inf")
    w = x.var(axis=1, ddof=1).mean()
    b = n * x.mean(axis=1).var(ddof=1)
    if w <= 0:
        return float("inf") if b > 0 else 1.0
    return float(math.sqrt(((n - 1) / n * w + b / n) / w))


def _ess_basic(x: np.ndarray) -> float:
    """ESS of ``(chains, n)`` draws.

    Multi-chain autocorrelation and Geyer's initial monotone sequence (as Stan / ArviZ).
    """
    m, n = x.shape
    if n < 4:
        return float("nan")
    xc = x - x.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, size, axis=1)
    acov = np.fft.irfft(f * np.conj(f), size, axis=1)[:, :n] / n  # biased autocovariance
    chain_var = acov[:, 0] * n / (n - 1)
    w = chain_var.mean()
    var_plus = w * (n - 1) / n + (x.mean(axis=1).var(ddof=1) if m > 1 else 0.0)
    if var_plus <= 0:
        return float("nan")
    rho = 1.0 - (w - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    # Geyer: sums of consecutive pairs, positive and monotone
    t = 0
    pairs = []
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        pairs.append(p)
        t += 2
    pairs = np.minimum.accumulate(np.array(pairs)) if pairs else np.array([1.0])
    tau = -1.0 + 2.0 * pairs.sum()
    tau = max(tau, 1.0 / math.log10(m * n)) if m * n > 1 else tau
    return float(m * n / tau)


def rhat(x: np.ndarray) -> float:
    """Rank-normalised split R-hat (max of bulk and folded) of ``(chains, n)`` draws."""
    s = _split(np.asarray(x, float))
    bulk = _rhat_basic(_rank_normal(s))
    fold = _rhat_basic(_rank_normal(np.abs(s - np.median(s))))
    return max(bulk, fold)


def nested_rhat(x: np.ndarray, superchains: int = 4) -> float:
    """Nested R-hat of ``(chains, n)`` draws in ``superchains`` contiguous groups of chains.

    Margossian, Hoffman, Sountsov, Riou-Durand, Vehtari & Gelman (2024).

    Designed for many short chains: ``sqrt(1 + B / W)`` with ``B`` the variance of the
    superchain means and ``W`` the mean (within-superchain between-chain + within-chain)
    variance; ``~1 + 1 / (2 ESS per superchain)`` once the chains have mixed.
    """
    x = np.asarray(x, float)
    m, n = x.shape
    k = int(superchains)
    if k < 2 or m < k or n < 2:
        return float("nan")
    per = m // k
    x = x[: k * per].reshape(k, per, n)
    chain_mean = x.mean(axis=2)
    super_mean = chain_mean.mean(axis=1)
    b = super_mean.var(ddof=1)
    within_chain = x.var(axis=2, ddof=1).mean(axis=1)
    between_chain = chain_mean.var(axis=1, ddof=1) if per > 1 else np.zeros(k)
    w = float((between_chain + within_chain).mean())
    return float(math.sqrt(1 + b / w)) if w > 0 else float("inf")


def ess_bulk(x: np.ndarray) -> float:
    """Rank-normalised split bulk ESS of ``(chains, n)`` draws."""
    return _ess_basic(_rank_normal(_split(np.asarray(x, float))))


def ess_tail(x: np.ndarray) -> float:
    """Tail ESS: min of the ESS of the 5 % and 95 % quantile indicators."""
    s = _split(np.asarray(x, float))
    out = []
    for q in (0.05, 0.95):
        ind = (s <= np.quantile(s, q)).astype(float)
        out.append(_ess_basic(ind))
    return float(min(out))


def convergence(x: np.ndarray, superchains: int = 4) -> dict:
    """Per-coordinate diagnostics of ``(chains, n, d)`` draws and their worst values.

    Nested R-hat (rank-normalised, max of bulk and folded), split R-hat, bulk and tail ESS.
    """
    x = np.asarray(x, float)
    d = x.shape[2]
    if x.shape[1] < 4:
        nan = [float("nan")] * d
        return {"nested_rhat": nan, "rhat": nan, "ess_bulk": nan, "ess_tail": nan,
                "max_nested_rhat": float("inf"), "max_rhat": float("inf"),
                "min_ess_bulk": 0.0, "min_ess_tail": 0.0}  # fmt: skip
    nr, rh, eb, et = [], [], [], []
    for j in range(d):
        v = x[:, :, j]
        z = _rank_normal(v)
        zf = _rank_normal(np.abs(v - np.median(v)))
        nr.append(max(nested_rhat(z, superchains), nested_rhat(zf, superchains)))
        rh.append(rhat(v))
        eb.append(ess_bulk(v))
        et.append(ess_tail(v))
    return {"nested_rhat": nr, "rhat": rh, "ess_bulk": eb, "ess_tail": et,
            "max_nested_rhat": float(np.nanmax(nr)), "max_rhat": float(np.nanmax(rh)),
            "min_ess_bulk": float(np.nanmin(eb)), "min_ess_tail": float(np.nanmin(et))}  # fmt: skip


def _gpd_fit(x: np.ndarray) -> tuple[float, float]:
    """Shape ``k`` and scale of a generalised Pareto fit (Zhang & Stephens 2009, weak prior as PSIS)."""
    x = np.sort(x)
    n = len(x)
    m = 30 + int(math.sqrt(n))
    b = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b /= 3 * x[int(n / 4 + 0.5) - 1]
    b += 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    ls = n * (np.log(-(b / k)) - k - 1)
    w = 1 / np.exp(ls - ls[:, None]).sum(axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, b = w[keep] / w[keep].sum(), b[keep]
    b_post = float((b * w).sum())
    k_post = float(np.log1p(-b_post * x).mean())
    sigma = -k_post / b_post
    return (n * k_post + 10 * 0.5) / (n + 10), sigma  # weakly informative prior towards 0.5


def psis(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smoothed log weights and the Pareto ``k`` of the weight tail.

    PSIS (Vehtari, Simpson, Gelman, Yao & Gabry 2024): the largest
    ``M = min(n / 5, 3 sqrt(n))`` weights are replaced by quantiles of a generalised
    Pareto fit to them (capped at the largest raw weight). ``k < 0.5`` good,
    ``0.5-0.7`` acceptable, ``> 0.7`` the weights are unreliable.
    """
    lw = np.asarray(log_w, float).copy()
    finite = np.isfinite(lw)
    n = int(finite.sum())
    tail = int(min(n / 5, 3 * math.sqrt(n)))
    if tail < 5:
        return lw, float("nan")
    idx = np.flatnonzero(finite)[np.argsort(lw[finite])]
    top = lw[idx[-1]]
    cutoff = math.exp(lw[idx[-tail - 1]] - top)
    x = np.exp(lw[idx[-tail:]] - top) - cutoff
    if np.all(x <= 0):
        return lw, float("-inf")
    k, sigma = _gpd_fit(np.maximum(x, 1e-300))
    if np.isfinite(k) and sigma > 0:
        p = (np.arange(1, tail + 1) - 0.5) / tail
        q = sigma / k * ((1 - p) ** -k - 1) if abs(k) > 1e-12 else -sigma * np.log1p(-p)
        lw[idx[-tail:]] = np.log(np.minimum(cutoff + q, 1.0)) + top
    return lw, float(k)


# ---------------------------------------------------------------------- proposals
def _t_logpdf(x: np.ndarray, mu: np.ndarray, chol: np.ndarray, df: float) -> np.ndarray:
    d = len(mu)
    z = solve_triangular(chol, (np.atleast_2d(x) - mu).T, lower=True)
    q = (z**2).sum(0)
    return (gammaln((df + d) / 2) - gammaln(df / 2) - 0.5 * d * math.log(df * math.pi)
            - np.log(np.diag(chol)).sum() - 0.5 * (df + d) * np.log1p(q / df))  # fmt: skip


def _t_sample(rng, n: int, mu: np.ndarray, chol: np.ndarray, df: float) -> np.ndarray:
    z = rng.standard_normal((n, len(mu)))
    w = rng.chisquare(df, n) / df
    return mu + (z @ chol.T) / np.sqrt(w)[:, None]


@dataclass
class _MixT:
    """Mixture of multivariate t (common ``df``): the independence proposal."""

    means: np.ndarray
    chols: np.ndarray
    logpi: np.ndarray
    df: float

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        comp = [lp + _t_logpdf(x, m, c, self.df)
                for m, c, lp in zip(self.means, self.chols, self.logpi, strict=True)]  # fmt: skip
        return logsumexp(np.stack(comp, axis=1), axis=1)

    def sample(self, rng, n: int) -> np.ndarray:
        k = rng.choice(len(self.logpi), size=n, p=np.exp(self.logpi - logsumexp(self.logpi)))
        out = np.empty((n, self.means.shape[1]))
        for j in range(len(self.logpi)):
            idx = np.flatnonzero(k == j)
            if len(idx):
                out[idx] = _t_sample(rng, len(idx), self.means[j], self.chols[j], self.df)
        return out


def _gauss_logpdf(x: np.ndarray, mu: np.ndarray, chol: np.ndarray) -> np.ndarray:
    z = solve_triangular(chol, (x - mu).T, lower=True)
    return (-0.5 * (z**2).sum(0) - np.log(np.diag(chol)).sum()
            - 0.5 * len(mu) * math.log(2 * math.pi))  # fmt: skip


def fit_gmm(rng, x: np.ndarray, w: np.ndarray, k: int, iters: int = 200, tol: float = 1e-6
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:  # fmt: skip
    """Weighted EM for a ``k``-component Gaussian mixture (full covariances).

    Initialised by weighted k-means++ seeding; covariances regularised by ``1e-6`` x the
    mean variance. Returns ``(means, chols, log weights, weighted mean log-likelihood)``.
    """
    n, d = x.shape
    w = w / w.sum()
    reg = 1e-6 * float(np.average(x.var(axis=0)) + 1e-300) * np.eye(d)
    centres = [x[rng.choice(n, p=w)]]
    for _ in range(1, k):
        d2 = np.min([((x - c) ** 2).sum(1) for c in centres], axis=0) * w
        centres.append(x[rng.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else x[rng.choice(n)])
    means = np.array(centres)
    _, cov = _weighted_cov(x, w)
    chols = np.array([_chol(cov + reg)] * k)
    logpi = np.full(k, -math.log(k))
    prev = -np.inf
    for _ in range(int(iters)):
        lc = np.stack(
            [lp + _gauss_logpdf(x, m, c) for m, c, lp in zip(means, chols, logpi, strict=True)], 1
        )
        ll = logsumexp(lc, axis=1)
        r = np.exp(lc - ll[:, None]) * w[:, None]  # weighted responsibilities
        cur = float(w @ ll)
        nk = r.sum(0)
        keep = nk > 1e-8
        r, nk = r[:, keep], nk[keep]
        means = (r.T @ x) / nk[:, None]
        chols = np.array([_chol(((x - m) * rj[:, None]).T @ (x - m) / nj + reg)
                          for m, rj, nj in zip(means, r.T, nk, strict=True)])  # fmt: skip
        logpi = np.log(nk / nk.sum())
        if cur - prev < tol * abs(cur):
            break
        prev = cur
    return means, chols, logpi, cur


def fit_proposal(rng, x: np.ndarray, w: np.ndarray, components: int, scale: float,
                 df: float, n_eff: float | None = None, min_points: float = 2.0
                 ) -> tuple[_MixT, int]:  # fmt: skip
    """Mixture-of-t independence proposal fitted to the weighted population ``(x, w)``.

    Gaussian mixtures with ``1 .. components`` components (weighted EM) are compared by
    BIC with the effective sample size ``n_eff`` (default: Kish ESS of ``w``), at most
    one component per ``min_points x`` its parameter count; the best one's scales are
    multiplied by ``scale`` and its components turned into t (``df``). Returns the
    proposal and the number of components.
    """
    d = x.shape[1]
    w = np.asarray(w, float) / np.sum(w)
    n = float(n_eff) if n_eff is not None else 1.0 / float((w**2).sum())
    n = n if np.isfinite(n) and n > 1 else 1.0 / float((w**2).sum())
    per = d + d * (d + 1) / 2 + 1  # parameters per component
    max_k = int(min(components, max(1, n // (min_points * per))))
    if max_k <= 1:
        mean, cov = _weighted_cov(x, w)
        return _MixT(mean[None], (_chol(cov) * scale)[None], np.zeros(1), df), 1
    best, best_bic = None, np.inf
    for k in range(1, max_k + 1):
        m, c, lp, ll = fit_gmm(rng, x, w, k)
        bic = -2 * n * ll + (k * per - 1) * math.log(n)
        if bic < best_bic:
            best, best_bic = (m, c, lp), bic
    m, c, lp = best
    return _MixT(m, c * scale, lp, df), len(lp)


def _logdiag(chols: np.ndarray) -> np.ndarray:
    return np.log(np.diagonal(chols, axis1=1, axis2=2)).sum(axis=1)


class _Archive:
    """Local covariances: ``(1 - mix) cov(k nearest archive states) + mix cov(archive)``.

    Nearest in the Mahalanobis metric of the global Cholesky factor ``chol``.
    """

    def __init__(self, pts: np.ndarray, chol: np.ndarray, k: int, mix: float) -> None:
        self.pts = np.asarray(pts, float)
        self.k = int(min(max(k, pts.shape[1] + 2), len(pts) - 1))
        self.mix = float(mix)
        self.chol = chol
        self.white = solve_triangular(chol, self.pts.T, lower=True).T
        self.glob = chol @ chol.T

    def chols(self, x: np.ndarray) -> np.ndarray:
        """Lower Cholesky factors ``(n, d, d)`` of the local covariances at ``x``."""
        xw = solve_triangular(self.chol, np.atleast_2d(x).T, lower=True).T
        d2 = (xw**2).sum(1)[:, None] - 2 * xw @ self.white.T + (self.white**2).sum(1)[None]
        idx = np.argpartition(d2, self.k, axis=1)[:, : self.k]
        nb = self.pts[idx]  # (n, k, d)
        c = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", c, c) / (self.k - 1)
        cov = (1 - self.mix) * cov + self.mix * self.glob
        return np.linalg.cholesky(cov + 1e-12 * np.eye(len(self.glob)))


def systematic(rng, w: np.ndarray, n: int) -> np.ndarray:
    """Systematic resampling: ``n`` indices for normalised weights ``w``."""
    c = np.cumsum(w)
    c[-1] = 1.0
    return np.searchsorted(c, (rng.uniform() + np.arange(n)) / n)


def _weighted_cov(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = w @ x
    c = ((x - m) * w[:, None]).T @ (x - m)
    return m, 0.5 * (c + c.T)


def _chol(cov: np.ndarray) -> np.ndarray:
    d = len(cov)
    jitter = 1e-10 * max(float(np.trace(cov)) / d, 1e-12)
    for _ in range(10):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(d))
        except np.linalg.LinAlgError:
            jitter *= 10
    w, V = np.linalg.eigh(cov)
    return np.linalg.cholesky((V * np.maximum(w, 1e-8)) @ V.T)


# ---------------------------------------------------------------------- sampler
@dataclass
class SMCResult:
    """Final-stage states ``(chains, length, d)`` and their log targets; diagnostics."""

    chains: np.ndarray
    log_post: np.ndarray
    info: dict

    @property
    def draws(self) -> np.ndarray:
        """All final-stage states ``(chains * length, d)`` (equal weights)."""
        return self.chains.reshape(-1, self.chains.shape[-1])


def tempered_smc(
    logpost: Callable[[np.ndarray], np.ndarray],
    ref_sample: Callable[[np.random.Generator, int], np.ndarray],
    ref_logpdf: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    *,
    chains: int = 64,
    steps: int = 8,
    ess_fraction: float = 0.5,
    rw_acceptance: float = 0.3,
    imh_prob: float = 0.5,
    imh_min: float = 0.05,
    imh_scale: float = 1.2,
    imh_components: int = 1,
    de_prob: float = 0.0,
    rw_scale_sd: float = 0.0,
    df: float = 5.0,
    target_ess: float = 400.0,
    max_rhat: float = 1.01,
    tail_fraction: float = 0.5,
    superchains: int = 4,
    min_final_steps: int = 16,
    max_final_steps: int = 2000,
    final_burn: float = 0.25,
    adapt_final: bool = True,
    adapt_final_steps: int = 1000,
    local_k: int = 0,
    local_mix: float = 0.1,
    check_every: int = 8,
    max_evals: int = 20000,
    max_stages: int = 100,
    name: str = "smc",
) -> SMCResult:
    """Waste-free adaptive tempered SMC (module docstring). Returns :class:`SMCResult`.

    ``logpost`` may return ``-inf`` (zero density); ``ref_logpdf`` must be normalised for
    the evidence estimate (only reported). ``max_evals`` counts ``logpost`` evaluations
    (including the ``chains x steps`` initial draws); stages stop early once it is hit
    (``info["status"] = "budget"``; then the output is the last population and the
    diagnostics say whether it reached ``beta = 1``).
    """
    t0 = time.perf_counter()
    chains, steps = int(chains), max(int(steps), 2)
    n_pop = chains * steps
    x = ref_sample(rng, n_pop)
    d = x.shape[1]
    lp = np.asarray(logpost(x), float)
    lq = np.asarray(ref_logpdf(x), float)
    evals = n_pop
    lam = np.where(np.isfinite(lp), lp - lq, -np.inf)  # log post / q
    beta, log_z = 0.0, 0.0
    logw = np.zeros(n_pop)
    rw_scale = 2.38 / math.sqrt(d)
    p_imh = float(imh_prob)
    stages: list[dict] = []
    status = "converged"
    final_info = None

    def target(lp_, lq_, b):
        return np.where(np.isfinite(lp_), b * lp_ + (1 - b) * lq_, -np.inf)

    for stage in range(int(max_stages)):
        # ---- next temperature: ESS of the incremental weights = ess_fraction x N
        def ess_at(b, _beta=beta, _logw=logw, _lam=lam, _n=n_pop):
            return _kish(_logw + (b - _beta) * _lam) / _n

        if ess_at(1.0) >= ess_fraction:
            new = 1.0
        else:
            lo, hi = beta, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if ess_at(mid) >= ess_fraction:
                    lo = mid
                else:
                    hi = mid
            new = max(lo, beta + 1e-12)
        inc = np.where(np.isfinite(lam), (new - beta) * lam, -np.inf)
        k_inc = psis(inc)[1] if new > beta else float("nan")
        lw_new = logw + inc
        log_z += float(logsumexp(lw_new) - logsumexp(logw))
        beta = new
        w = np.exp(lw_new - lw_new.max())
        w /= w.sum()
        stage_ess = _kish(lw_new)
        # ---- kernel from the weighted population
        mean, cov = _weighted_cov(x, w)
        chol = _chol(cov)
        prop_q, n_comp = fit_proposal(rng, x, w, imh_components, imh_scale, df)
        idx = systematic(rng, w, chains)
        cur = x[idx].copy()
        cur_lp, cur_lq = lp[idx].copy(), lq[idx].copy()
        cur_g = target(cur_lp, cur_lq, beta)
        cur_h = prop_q.logpdf(cur)
        is_final = beta >= 1.0
        n_steps = int(max_final_steps) if is_final else steps - 1
        # position-dependent random walk (final stage): covariance of the local_k nearest
        # archive states, so the steps follow a curved ridge; MH uses both proposal densities
        local = is_final and local_k > 0
        if local:
            arch = _Archive(x, chol, int(local_k), local_mix)
            cur_lc = arch.chols(cur)
        path_x, path_lp, path_lq = [cur.copy()], [cur_lp.copy()], [cur_lq.copy()]
        acc_rw, acc_imh, acc_de, n_rw, n_imh, n_de = 0, 0, 0, 0, 0, 0
        last_acc = (0, 0)
        checks: list[dict] = []
        t_stage = time.perf_counter()
        for k in range(n_steps):
            r_kind = rng.uniform(size=chains)
            use_imh = r_kind < p_imh
            use_de = ~use_imh & (r_kind < p_imh + de_prob)
            # random walk with a log-normal scale mixture (bigger and smaller jumps)
            scales = rw_scale * np.exp(rw_scale_sd * rng.standard_normal(chains))
            z = rng.standard_normal((chains, d))
            if local:
                prop = cur + scales[:, None] * np.einsum("nij,nj->ni", cur_lc, z)
            else:
                prop = cur + scales[:, None] * (z @ chol.T)
            if use_imh.any():
                prop[use_imh] = prop_q.sample(rng, int(use_imh.sum()))
            if use_de.any():
                # differential evolution with differences of the (fixed) stage population:
                # symmetric, follows its non-Gaussian shape; gamma = 1 sometimes (jumps)
                m = int(use_de.sum())
                a = rng.integers(len(x), size=m)
                b = (a + 1 + rng.integers(len(x) - 1, size=m)) % len(x)
                gam = np.where(rng.uniform(size=m) < 0.1, 1.0, 2.38 / math.sqrt(2 * d))
                prop[use_de] = (cur[use_de] + gam[:, None] * (x[a] - x[b])
                                + 1e-3 * rng.standard_normal((m, d)) @ chol.T)  # fmt: skip
            p_lp = np.asarray(logpost(prop), float)
            evals += chains
            p_lq = np.asarray(ref_logpdf(prop), float)
            p_g = target(p_lp, p_lq, beta)
            p_h = prop_q.logpdf(prop)
            log_a = p_g - cur_g + np.where(use_imh, cur_h - p_h, 0.0)
            if local:
                prop_lc = arch.chols(prop)
                rw = ~use_imh & ~use_de
                # log q(cur | prop) - log q(prop | cur), same scale both ways
                back = np.linalg.solve(prop_lc, ((cur - prop) / scales[:, None])[:, :, None])
                log_a += np.where(rw, -0.5 * (back[:, :, 0] ** 2).sum(1) + 0.5 * (z**2).sum(1)
                                  - _logdiag(prop_lc) + _logdiag(cur_lc), 0.0)  # fmt: skip
            ok = np.isfinite(p_g) & (np.log(rng.uniform(size=chains)) < log_a)
            cur[ok], cur_lp[ok], cur_lq[ok] = prop[ok], p_lp[ok], p_lq[ok]
            cur_g[ok], cur_h[ok] = p_g[ok], p_h[ok]
            if local:
                cur_lc[ok] = prop_lc[ok]
            use_rw = ~use_imh & ~use_de
            acc_rw += int((ok & use_rw).sum())
            acc_imh += int((ok & use_imh).sum())
            acc_de += int((ok & use_de).sum())
            n_rw += int(use_rw.sum())
            n_imh += int(use_imh.sum())
            n_de += int(use_de.sum())
            path_x.append(cur.copy())
            path_lp.append(cur_lp.copy())
            path_lq.append(cur_lq.copy())
            length = k + 2
            if is_final and ((length >= min_final_steps and (k + 1) % check_every == 0)
                             or evals + chains > max_evals or k == n_steps - 1):  # fmt: skip
                # (chains, length, d) without the first final_burn of the chains (transient
                # from the population if it is not yet exactly the posterior)
                arr = np.stack(path_x[int(final_burn * length) :], axis=1)
                c = convergence(arr, superchains)
                checks.append({"length": length, "evals": evals, **{
                    k: c[k] for k in ("max_nested_rhat", "max_rhat", "min_ess_bulk",
                                      "min_ess_tail")}})  # fmt: skip
                LOG.info("%s: final stage, chain length %d (%d evals, %.0fs): nested R-hat "
                         "%.4f, split R-hat %.3f, min ESS bulk %.0f tail %.0f, acceptance rw "
                         "%.2f imh %.2f", name, length, evals, time.perf_counter() - t0,
                         c["max_nested_rhat"], c["max_rhat"], c["min_ess_bulk"],
                         c["min_ess_tail"], acc_rw / max(n_rw, 1),
                         acc_imh / max(n_imh, 1))  # fmt: skip
                if (length >= min_final_steps and c["max_nested_rhat"] <= max_rhat
                        and c["min_ess_bulk"] >= target_ess
                        and c["min_ess_tail"] >= tail_fraction * target_ess):  # fmt: skip
                    break
                if evals + chains > max_evals:
                    status = "budget"
                    break
                if k == n_steps - 1:
                    status = "max_final_steps"
                if adapt_final and length < adapt_final_steps:
                    # adaptive Metropolis on the final chains: the stage population can be
                    # narrower than the posterior (tails), which would freeze too small a
                    # random walk; covariance, scale and independence proposal are refitted
                    # from the pooled chain states (the burn-in fraction is discarded)
                    pooled = arr.reshape(-1, d)
                    chol = _chol(np.cov(pooled.T))
                    x = pooled  # DE differences from the chains' past (DE-MCz)
                    if local:
                        sub = pooled[rng.choice(len(pooled), min(len(pooled), 4096),
                                                replace=False)]  # fmt: skip
                        arch = _Archive(sub, chol, int(local_k), local_mix)
                        cur_lc = arch.chols(cur)
                    win = (acc_rw - last_acc[0]) / max(n_rw - last_acc[1], 1)
                    rw_scale *= math.exp(1.5 * (win - rw_acceptance))
                    last_acc = (acc_rw, n_rw)
                    if p_imh > 0:
                        prop_q, n_comp = fit_proposal(rng, pooled, np.ones(len(pooled)),
                                                      imh_components, imh_scale, df,
                                                      n_eff=c["min_ess_bulk"])  # fmt: skip
                        cur_h = prop_q.logpdf(cur)
            elif not is_final and evals + chains > max_evals:
                status = "budget"
                break
        a_rw = acc_rw / n_rw if n_rw else float("nan")
        a_imh = acc_imh / n_imh if n_imh else float("nan")
        a_de = acc_de / n_de if n_de else float("nan")
        stages.append({
            "beta": float(beta), "ess": float(stage_ess), "pareto_k": float(k_inc),
            "unique_starts": int(len(np.unique(idx))), "steps": len(path_x) - 1,
            "acceptance_rw": a_rw, "acceptance_imh": a_imh, "rw_scale": float(rw_scale),
            "imh_prob": float(p_imh), "imh_components": n_comp, "acceptance_de": a_de,
            "evals": int(evals), "seconds": time.perf_counter() - t_stage,
        })  # fmt: skip
        LOG.info("%s: stage %d beta %.4g (ESS %.0f/%d, k %.2f), %d steps, acceptance rw %.2f "
                 "imh %.2f de %.2f, %d evals, %.0fs", name, stage, beta, stage_ess, n_pop, k_inc,
                 len(path_x) - 1, a_rw, a_imh, a_de, evals, time.perf_counter() - t0)  # fmt: skip
        # tune between stages: random-walk scale towards rw_acceptance (log scale); the
        # independence share follows its acceptance (a poor fit wastes evaluations)
        if np.isfinite(a_rw) and not is_final:
            rw_scale *= math.exp(1.5 * (a_rw - rw_acceptance))
        if np.isfinite(a_imh) and not is_final and imh_prob > 0:
            p_imh = float(np.clip(2.0 * a_imh, min(imh_min, imh_prob), imh_prob))
        b0 = int(final_burn * len(path_x)) if is_final else 0
        X = np.stack(path_x[b0:], axis=1)
        LP, LQ = np.stack(path_lp[b0:], axis=1), np.stack(path_lq[b0:], axis=1)
        if is_final:
            final_info = {"checks": checks}
            break
        x, lp, lq = X.reshape(-1, d), LP.reshape(-1), LQ.reshape(-1)
        lam = np.where(np.isfinite(lp), lp - lq, -np.inf)
        logw = np.zeros(len(x))
        n_pop = len(x)
        if status == "budget":
            break
    else:
        status = "max_stages"
    conv = convergence(X, superchains)
    info = {
        "status": status if beta >= 1.0 else ("budget" if status == "converged" else status),
        "reached_posterior": bool(beta >= 1.0),
        "beta": float(beta),
        "stages": len(stages),
        "temperatures": [s["beta"] for s in stages],
        "stage_diagnostics": stages,
        "chains": chains,
        "steps": steps,
        "final_length": int(X.shape[1]),
        "final_checks": (final_info or {}).get("checks", []),
        "superchains": int(superchains),
        **conv,
        "log_evidence": float(log_z),
        "evals": int(evals),
        "seconds": time.perf_counter() - t0,
    }  # fmt: skip
    return SMCResult(X, LP, info)


def smc_islands(
    logpost: Callable[[np.ndarray], np.ndarray],
    ref_sample: Callable[[np.random.Generator, int], np.ndarray],
    ref_logpdf: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    *,
    islands: int = 4,
    max_islands: int = 8,
    final_steps: int = 16,
    target_ess: float = 400.0,
    max_rhat: float = 1.01,
    tail_fraction: float = 0.5,
    max_evals: int = 40000,
    name: str = "smc",
    **smc_options,
) -> SMCResult:
    """Independent tempered-SMC runs ("islands") pooled, with between-island diagnostics.

    Each island is a :func:`tempered_smc` run whose final stage is ``final_steps`` MH
    steps at ``beta = 1`` (no stopping rule). Islands are added until at least
    ``islands`` have reached the posterior and the pooled diagnostics pass (nested R-hat
    with one superchain per island ``<= max_rhat``, min bulk ESS ``>= target_ess``, min
    tail ESS ``>= tail_fraction x target_ess``), or until ``max_islands`` / ``max_evals``.
    The output chains are the final-stage chains of all islands that reached
    ``beta = 1`` (island after island). ``info["ess_between"]`` = ``I x var / var(island
    means)`` per coordinate (noisy for few islands; information only).
    """
    t0 = time.perf_counter()
    runs: list[SMCResult] = []
    evals = 0
    status = "max_islands"
    n0 = int(smc_options.get("chains", 64)) * int(smc_options.get("steps", 8))
    for i in range(int(max_islands)):
        left = int(max_evals) - evals
        if left < 2 * n0:
            status = "budget"
            break
        r = tempered_smc(
            logpost, ref_sample, ref_logpdf, rng, target_ess=0.0, max_rhat=float("inf"),
            tail_fraction=0.0, min_final_steps=int(final_steps),
            max_final_steps=int(final_steps), check_every=int(final_steps), max_evals=left,
            final_burn=0.0,
            name=f"{name}/island{i}", **smc_options,
        )  # fmt: skip
        evals += r.info["evals"]
        runs.append(r)
        good = [x for x in runs if x.info["reached_posterior"]]
        LOG.info("%s: island %d: %s, %d stages, %d evals (total %d, %.0fs)", name, i,
                 r.info["status"], r.info["stages"], r.info["evals"], evals,
                 time.perf_counter() - t0)  # fmt: skip
        if len(good) >= max(int(islands), 2):
            conv = convergence(np.concatenate([x.chains for x in good], axis=0), len(good))
            LOG.info("%s: %d islands: nested R-hat %.4f, min ESS bulk %.0f tail %.0f", name,
                     len(good), conv["max_nested_rhat"], conv["min_ess_bulk"],
                     conv["min_ess_tail"])  # fmt: skip
            if (conv["max_nested_rhat"] <= max_rhat and conv["min_ess_bulk"] >= target_ess
                    and conv["min_ess_tail"] >= tail_fraction * target_ess):  # fmt: skip
                status = "converged"
                break
    good = [x for x in runs if x.info["reached_posterior"]]
    if not runs:
        raise RuntimeError("smc_islands: max_evals too small for one island")
    if not good:
        last = runs[-1]
        return SMCResult(last.chains, last.log_post, {
            **last.info, "status": "budget", "islands": 0, "islands_run": len(runs),
            "evals": evals, "seconds": time.perf_counter() - t0})  # fmt: skip
    X = np.concatenate([x.chains for x in good], axis=0)
    LP = np.concatenate([x.log_post for x in good], axis=0)
    d = X.shape[2]
    if len(good) >= 2:
        conv = convergence(X, len(good))
        means = np.array([x.chains.reshape(-1, d).mean(0) for x in good])
        between = np.maximum(means.var(axis=0, ddof=1), 1e-300)
        ess_between = (len(good) * X.reshape(-1, d).var(0) / between).tolist()
    else:
        conv = convergence(X, int(smc_options.get("superchains", 4)))
        ess_between = [float("nan")] * d
    info = {
        "status": status,
        "reached_posterior": True,
        "islands": len(good),
        "islands_run": len(runs),
        "final_steps": int(final_steps),
        "chains": int(X.shape[0]),
        "final_length": int(X.shape[1]),
        "stages": [x.info["stages"] for x in good],
        "temperatures": [x.info["temperatures"] for x in good],
        "stage_diagnostics": [x.info["stage_diagnostics"] for x in good],
        "island_log_evidence": [x.info["log_evidence"] for x in good],
        "island_evals": [x.info["evals"] for x in runs],
        "ess_between": [float(v) for v in ess_between],
        "superchains": len(good),
        **conv,
        "evals": int(evals),
        "seconds": time.perf_counter() - t0,
    }  # fmt: skip
    return SMCResult(X, LP, info)
