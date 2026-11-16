"""Held-out predictive check for the exact-likelihood methods.

The organoids of every horizon are split (seeded, per horizon) into a training part and a
test part of ``fraction`` of the clones; the posterior is sampled on the training part and
the **test log predictive density**

    lpd = log p(test | train) = log mean_s p(test | theta_s),   theta_s ~ p(theta | train)

is the log-mean-exp over posterior draws of the exact joint log-likelihood of the test
organoids (the same clones under every model: lpd differences between models fitted to the
same split are log predictive Bayes factors). ``lpd_per_clone = lpd / n_test``. MC error by
the delta method on the importance-free average (``sd(w) / (sqrt(n_eff) mean(w))``,
``w = p(test | theta_s)``, ``n_eff`` = number of draws); PSIS ``k`` of the ``log w`` says
whether a few draws dominate (``k > 0.7``: more draws needed).

Configured by ``inference.method.holdout = {fraction, seed, draws}`` of ``exact_gw`` /
``exact_bh`` (``fraction = 0``: off). The split depends only on the data and ``seed``,
so different models with the same ``holdout`` settings on the same reference data use the
same test organoids.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

from ..rng import rng_for
from ..summaries import Cohort
from .smc import psis

DEFAULT = {"fraction": 0.0, "seed": 0, "draws": 400}


def settings(holdout: dict | None) -> dict:
    """Holdout options with defaults filled in."""
    s = {**DEFAULT, **dict(holdout or {})}
    s["fraction"] = float(s["fraction"])
    if not 0.0 <= s["fraction"] < 1.0:
        raise ValueError("holdout.fraction must be in [0, 1)")
    s["seed"], s["draws"] = int(s["seed"]), int(s["draws"])
    return s


def split_cohorts(cohorts: list[Cohort], fraction: float, seed: int
                  ) -> tuple[list[Cohort], list[Cohort]]:  # fmt: skip
    """Seeded per-horizon split into ``(train, test)``; ``round(fraction * n)`` test clones
    per horizon (at least one if ``fraction > 0``), clone order kept within each part."""
    train, test = [], []
    for c in cohorts:
        n = len(c.counts)
        k = max(1, int(round(fraction * n))) if fraction > 0 else 0
        if k >= n:
            raise ValueError(f"holdout: horizon {c.horizon:g} has only {n} clones")
        perm = rng_for(seed, "holdout", f"{float(c.horizon):g}").permutation(n)
        is_test = np.zeros(n, bool)
        is_test[perm[:k]] = True
        train.append(Cohort(c.horizon, np.asarray(c.counts)[~is_test]))
        test.append(Cohort(c.horizon, np.asarray(c.counts)[is_test]))
    return train, test


def lpd_summary(loglik_test: np.ndarray, n_test: int) -> dict:
    """Test lpd (log-mean-exp of per-draw test log-likelihoods) and its MC error."""
    ll = np.asarray(loglik_test, float)
    ll = ll[~np.isnan(ll)]
    n = len(ll)
    if n == 0 or not np.isfinite(ll).any():
        return {"lpd": float("-inf"), "lpd_se": float("nan"), "lpd_per_clone": float("-inf"),
                "draws": int(n), "n_test": int(n_test), "pareto_k": float("nan")}  # fmt: skip
    lpd = float(logsumexp(ll) - math.log(n))
    w = np.exp(ll - ll.max())
    se = float(w.std(ddof=1) / (math.sqrt(n) * w.mean())) if n > 1 else float("nan")
    return {
        "lpd": lpd,
        "lpd_se": se,
        "lpd_per_clone": lpd / max(n_test, 1),
        "mean_loglik": float(np.mean(ll[np.isfinite(ll)])),
        "draws": int(n),
        "n_test": int(n_test),
        "pareto_k": float(psis(ll)[1]) if n > 20 else float("nan"),
    }


def thin(draws: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` draws without replacement (all if fewer)."""
    if n <= 0 or n >= len(draws):
        return draws
    idx = np.sort(np.random.default_rng(seed).choice(len(draws), int(n), replace=False))
    return draws[idx]
