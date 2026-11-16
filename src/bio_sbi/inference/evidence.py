"""Model evidence (marginal likelihood) for the exact-likelihood methods.

``log Z = log int p(data | theta) p(theta) d theta``, with the prior *normalised* (the
log posteriors of :mod:`exact_gw` / :mod:`exact_bh` drop the Beta normalising constants of
the stick-breaking prior: :func:`prior_log_norm` gives the missing constant
``-sum log B(a_j, b_j)``; the uniform log-lifetime prior is Beta(1, 1) on the unit-scaled
coordinate and contributes 0). Evidences of different models fitted to the *same counts*
(same clones, same conditioning on detection) give Bayes factors.

Estimators:

* **Bridge sampling** (Meng & Wong 1996; iterative optimal bridge, Gronau et al. 2017),
  :func:`bridge_sampling`: posterior draws (MCMC chains) plus ``n_proposal`` draws of a
  multivariate-t proposal ``g`` fitted to them; only the proposal draws cost new
  likelihood evaluations. With ``l1 = log q~ - log g`` at the posterior draws and
  ``l2`` at the proposal draws (``q~`` = unnormalised posterior), iterate

      r <- [mean_j e^{l2_j} / (s1 e^{l2_j} + s2 r)] / [mean_i 1 / (s1 e^{l1_i} + s2 r)].

  MC error: relative MSE of Fruehwirth-Schnatter (2004),
  ``RE^2 = Var_g(f1) / (N2 E_g(f1)^2) + Var_p(f2) / (ESS(f2) E_p(f2)^2)`` with
  ``f1 = p / (s1 p + s2 g)`` at the proposal draws, ``f2 = g / (s1 p + s2 g)`` at the
  posterior draws (``p = q~ / r``) and ``ESS(f2)`` the multi-chain effective sample size
  of ``f2`` (autocorrelation of the chains); ``sd(log Z) ~= RE``. The proposal is
  fitted to the same draws (bias O(1/N), negligible here; tested against known
  evidences).
* **Tempered SMC** (:func:`bio_sbi.inference.smc.tempered_smc`): the product of the mean
  incremental weights along ``q^(1-beta) post^beta`` estimates ``int post / int q``;
  ``q`` is a normalised density (prior, Laplace t mixture, ...) and ``post`` the
  unnormalised posterior, so it estimates the same ``Z`` (plus the prior constant) for
  *any* start ``q`` -- a Laplace start needs no correction. Reported by ``exact_bh`` as a
  cross-check (no MC error).
* Importance sampling with the bridge proposal (``log_evidence_is``, PSIS ``k``):
  reported for information.

Bridge sampling runs in coordinates where the posterior is close to Gaussian: ``exact_bh``
uses its sampler coordinates (``reparam``: probit sticks, sheared lifetimes), ``exact_gw``
probit sticks (:func:`probit_view`) instead of ``logit``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.special import betaln, expit, log_ndtr, logsumexp, ndtri

from .smc import _ess_basic, fit_proposal, psis


def prior_log_norm(prior_a: np.ndarray, prior_b: np.ndarray) -> float:
    """``-sum log B(a, b)``: add to an evidence computed with the unnormalised Beta prior."""
    return -float(betaln(np.asarray(prior_a, float), np.asarray(prior_b, float)).sum())


def _logmeanexp(x: np.ndarray) -> float:
    return float(logsumexp(x) - math.log(len(x)))


def bridge_sampling(
    chains: np.ndarray,
    log_target_chains: np.ndarray,
    log_target: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    n_proposal: int = 1000,
    components: int = 1,
    df: float = 30.0,
    max_iter: int = 1000,
    tol: float = 1e-10,
    split: bool = False,
) -> dict:
    """Bridge-sampling estimate of ``log int exp(log_target)`` (module docstring).

    Args:
        chains: Posterior draws ``(C, L, d)`` (C chains of length L; ``(n, d)`` = one chain).
        log_target_chains: Unnormalised log target at the draws, ``(C, L)``.
        log_target: Batched unnormalised log target ``(m, d) -> (m,)`` (``-inf`` allowed),
            evaluated at the ``n_proposal`` proposal draws.
        rng: Random generator.
        n_proposal: Proposal draws (= new target evaluations).
        components: Max components of the (BIC-selected) t-mixture proposal.
        df: Degrees of freedom of the proposal's t components.
        split: Fit the proposal to the first half of every chain and use only the second
            halves as posterior draws of the estimator (no reuse of the fitted draws).

    Returns:
        ``log_evidence``, ``mc_se`` (sd of ``log_evidence``), ``re2``, ``n_post``,
        ``ess_post`` (ESS of ``f2``), ``n_proposal``, ``finite_proposal``, ``iterations``,
        ``components``, ``log_evidence_is`` and ``is_pareto_k``.
    """
    X = np.asarray(chains, float)
    if X.ndim == 2:
        X = X[None]
    C, L, d = X.shape
    lt = np.asarray(log_target_chains, float).reshape(C, L)
    if not np.all(np.isfinite(lt)):
        raise ValueError("bridge_sampling: non-finite log target at posterior draws")
    fit_x = X.reshape(-1, d)
    if split and L >= 8:
        h = L // 2
        fit_x = X[:, :h].reshape(-1, d)
        X, lt = X[:, h:], lt[:, h:]
        L = X.shape[1]
    flat = X.reshape(-1, d)
    n1 = C * L
    ess_lt = _ess_basic(lt) if L >= 4 else float(n1)
    ess_lt = float(np.clip(ess_lt if np.isfinite(ess_lt) else n1, 1.0, n1))
    q, k = fit_proposal(rng, fit_x, np.ones(len(fit_x)), components, 1.0, df, n_eff=ess_lt)
    Y = q.sample(rng, int(n_proposal))
    lt_prop = np.asarray(log_target(Y), float).reshape(-1)
    l1 = lt.reshape(-1) - q.logpdf(flat)
    l2 = np.where(np.isfinite(lt_prop), lt_prop - q.logpdf(Y), -np.inf)
    n2 = len(Y)
    s1 = ess_lt / (ess_lt + n2)
    s2 = 1.0 - s1
    ls1, ls2 = math.log(s1), math.log(s2)
    logr = _logmeanexp(l2) if np.isfinite(l2).any() else float(np.median(l1))
    it = 0
    for it in range(1, int(max_iter) + 1):
        num = _logmeanexp(l2 - np.logaddexp(ls1 + l2, ls2 + logr))
        den = _logmeanexp(-np.logaddexp(ls1 + l1, ls2 + logr))
        new = num - den
        done = abs(new - logr) < tol
        logr = new
        if done:
            break
    # relative MSE (Fruehwirth-Schnatter 2004); f1 at proposal, f2 at posterior draws
    with np.errstate(over="ignore"):
        f1 = 1.0 / (s1 + s2 * np.exp(np.minimum(logr - l2, 700)))
        f2 = 1.0 / (s1 * np.exp(np.minimum(l1 - logr, 700)) + s2)
    f2c = f2.reshape(C, L)
    ess_f2 = _ess_basic(f2c) if L >= 4 else float(n1)
    ess_f2 = float(np.clip(ess_f2 if np.isfinite(ess_f2) else n1, 1.0, n1))
    re2 = float(f1.var() / (n2 * f1.mean() ** 2) + f2.var() / (ess_f2 * f2.mean() ** 2))
    lis = l2[np.isfinite(l2)]
    return {
        "log_evidence": float(logr),
        "mc_se": float(math.sqrt(re2)),
        "re2": re2,
        "n_post": int(n1),
        "ess_post": ess_f2,
        "n_proposal": int(n2),
        "finite_proposal": float(np.isfinite(lt_prop).mean()),
        "iterations": int(it),
        "components": int(k),
        "df": float(df),
        "log_evidence_is": _logmeanexp(l2) if len(lis) else float("-inf"),
        "is_pareto_k": float(psis(l2)[1]) if len(lis) > 10 else float("nan"),
    }


# ---------------------------------------------------------------------- coordinates
def probit_from_logit(U: np.ndarray) -> np.ndarray:
    """``v = probit(sigmoid(u))`` without rounding ``sigmoid`` to 1."""
    U = np.asarray(U, float)
    return np.where(U <= 0, ndtri(expit(np.minimum(U, 0))), -ndtri(expit(-np.maximum(U, 0))))


def logit_from_probit(V: np.ndarray) -> np.ndarray:
    """``u = logit(Phi(v))``."""
    V = np.asarray(V, float)
    return log_ndtr(V) - log_ndtr(-V)


def log_jac_probit(V: np.ndarray) -> np.ndarray:
    """``log |du/dv|`` summed over the last axis (``u = logit(Phi(v))``)."""
    V = np.asarray(V, float)
    return (-0.5 * V**2 - 0.5 * math.log(2 * math.pi) - log_ndtr(V) - log_ndtr(-V)).sum(-1)


def probit_view(chains_u: np.ndarray, logp_u: np.ndarray,
                log_target_u: Callable[[np.ndarray], np.ndarray]
                ) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:  # fmt: skip
    """Draws, log targets and target function of a logit-coordinate posterior in probit
    coordinates (exact Jacobian; the integral is unchanged)."""
    V = probit_from_logit(chains_u)
    lv = np.asarray(logp_u, float) + log_jac_probit(V)

    def target_v(Vb):
        Vb = np.atleast_2d(Vb)
        return np.asarray(log_target_u(logit_from_probit(Vb)), float) + log_jac_probit(Vb)

    return V, lv, target_v
