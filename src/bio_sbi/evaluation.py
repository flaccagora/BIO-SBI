"""Posterior quality diagnostics for synthetic experiments.

Two complementary checks are produced:

* **Recovery at the reference truth** (:func:`recovery_table`) - posterior mean,
  credible interval, z-score and *contraction* (``1 - posterior var / prior var``) for
  each physical parameter. Contraction near 0 means the data do not inform the
  parameter under this design: the key signal for experiment planning.
* **Posterior predictive check** (:func:`posterior_predictive_check`) - simulate the
  inference model at posterior draws and ask whether the observed summaries look
  typical. This is the main diagnostic of *model misspecification*: it works even when
  the data and inference models have different parameters.
* **Calibration across the prior** (:func:`calibration`) - simulation-based
  calibration (SBC) ranks via :func:`sbi.diagnostics.run_sbc`, KS uniformity p-values,
  and empirical coverage of central credible intervals. Only meaningful for amortised
  (single-round) posteriors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .inference import FittedPosterior
from .rng import seed_for, seed_torch

CREDIBLE_LEVELS = (0.5, 0.8, 0.9, 0.95)


def recovery_table(
    samples: np.ndarray,
    truth: np.ndarray,
    prior_samples: np.ndarray,
    names: Sequence[str],
    level: float = 0.9,
) -> list[dict]:
    """Per-parameter recovery statistics in physical space.

    Args:
        samples: ``(S, D)`` posterior samples (physical).
        truth: ``(D,)`` generating parameters; NaN where the data model has no such
            parameter (misspecified runs), which leaves error/coverage undefined.
        prior_samples: ``(S', D)`` prior samples (physical), for contraction.
        names: Parameter names.
        level: Credible level for the reported central interval.

    Returns:
        One dict per parameter.
    """
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=0)
    mean, std = samples.mean(0), samples.std(0)
    prior_var = prior_samples.var(0)
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "parameter": name,
                "truth": float(truth[i]),
                "posterior_mean": float(mean[i]),
                "posterior_median": float(np.median(samples[:, i])),
                "posterior_std": float(std[i]),
                f"ci{int(level * 100)}_low": float(lo[i]),
                f"ci{int(level * 100)}_high": float(hi[i]),
                "covered": bool(lo[i] <= truth[i] <= hi[i]) if np.isfinite(truth[i]) else None,
                # A truth of exactly 0 (e.g. a disabled transition) can never lie inside a
                # central interval of a [0, 1] posterior; such rows are not scored for coverage.
                "at_boundary": bool(np.isfinite(truth[i]) and truth[i] in (0.0, 1.0)),
                "abs_error": float(abs(mean[i] - truth[i])),
                "z_score": float((mean[i] - truth[i]) / std[i]) if std[i] > 0 else float("nan"),
                "contraction": float(1 - std[i] ** 2 / prior_var[i])
                if prior_var[i] > 0
                else float("nan"),
            }
        )
    return rows


def calibration(
    fitted: FittedPosterior,
    theta: np.ndarray,
    x: np.ndarray,
    names: Sequence[str],
    posterior_samples: int,
    seed: int = 0,
) -> tuple[dict, np.ndarray]:
    """Simulation-based calibration in inference coordinates.

    Args:
        fitted: Amortised posterior.
        theta: ``(N, d)`` prior draws.
        x: ``(N, x_dim)`` their simulated summaries (NaN rows are skipped).
        names: Names of the ``theta`` coordinates.
        posterior_samples: Samples per case used for ranks and intervals.
        seed: Seed for posterior sampling.

    Returns:
        ``(metrics, ranks)`` where ``ranks`` has shape ``(N_valid, d)``.
    """
    from sbi.diagnostics import run_sbc

    valid = np.all(np.isfinite(x), axis=1)
    theta_t = torch.as_tensor(theta[valid], dtype=torch.float32)
    x_t = fitted.condition(x[valid])
    seed_torch(seed_for(seed, "sbc"))
    ranks, _ = run_sbc(
        theta_t,
        x_t,
        fitted.posterior,
        num_posterior_samples=posterior_samples,
        show_progress_bar=False,
    )
    ranks = ranks.cpu().numpy()
    # Coverage of central intervals follows from ranks: u = rank / L is uniform if calibrated.
    u = (ranks + 0.5) / posterior_samples
    coverage = {
        f"{level:.2f}": {
            n: float(np.mean(np.abs(u[:, i] - 0.5) <= level / 2)) for i, n in enumerate(names)
        }
        for level in CREDIBLE_LEVELS
    }
    from scipy.stats import kstest

    ks = {n: float(kstest(u[:, i], "uniform").pvalue) for i, n in enumerate(names)}
    metrics = {
        "cases": int(valid.sum()),
        "invalid_cases": int((~valid).sum()),
        "posterior_samples": int(posterior_samples),
        "ks_pvalue": ks,
        "coverage": coverage,
        "mean_abs_coverage_error": float(
            np.mean([abs(v - float(lvl)) for lvl, d in coverage.items() for v in d.values()])
        ),
    }
    return metrics, ranks


def posterior_predictive_check(
    generator, samples_theta: np.ndarray, x_obs: np.ndarray, fitted: FittedPosterior
) -> tuple[dict, np.ndarray]:
    """Compare observed summaries with summaries simulated at posterior draws.

    For every summary feature used in training, the observed value's two-sided
    predictive tail probability ``p = 2 min(F, 1 - F)`` is computed from the predictive
    sample. Under a correct model these are roughly uniform; many small values flag
    a model that cannot reproduce the data, whatever the parameter values.

    Args:
        generator: Inference-model :class:`~bio_sbi.simulate.DataGenerator`.
        samples_theta: ``(S, d)`` posterior draws.
        x_obs: Observed summary vector.
        fitted: Posterior (provides the feature mask).

    Returns:
        ``(metrics, x_pred)`` with ``x_pred`` the ``(S, x_dim)`` predictive summaries.
    """
    x_pred, _ = generator.simulate_theta(samples_theta, "ppc")
    valid = np.all(np.isfinite(x_pred), axis=1)
    mask = fitted.feature_mask
    pred, obs = x_pred[valid][:, mask], np.asarray(x_obs)[mask]
    below = (pred < obs).mean(0) + 0.5 * (pred == obs).mean(0)
    tail = 2 * np.minimum(below, 1 - below)
    std = pred.std(0)
    z = np.where(std > 0, (obs - pred.mean(0)) / np.where(std > 0, std, 1), 0.0)
    metrics = {
        "samples": int(valid.sum()),
        "features": int(mask.sum()),
        "fraction_outside_95": float(np.mean(tail < 0.05)),
        "fraction_outside_99": float(np.mean(tail < 0.01)),
        "mean_abs_z": float(np.mean(np.abs(z))),
        "max_abs_z": float(np.max(np.abs(z))),
    }
    return metrics, x_pred
