"""Fixed-length summary statistics of a dataset.

sbi methods need a fixed-dimensional ``x``. A summary maps a dataset - a list of
:class:`Cohort` count matrices - to a float32 vector. Summaries only see measured
counts, never latent truth.

Add a summary by subclassing :class:`Summary` and pointing ``conf/summary/<name>.yaml``
at it via ``_target_``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cohort:
    """Measured endpoint counts of the clones in one cohort.

    Attributes:
        horizon: Measurement endpoint.
        counts: ``(n_clones, n_types)`` integer matrix of detected cells per type.
    """

    horizon: float
    counts: np.ndarray


class Summary(ABC):
    """Map a dataset to a fixed-length vector."""

    @abstractmethod
    def __call__(self, cohorts: Sequence[Cohort]) -> np.ndarray:
        """Return a 1-D float32 summary vector."""


@dataclass
class CohortMoments(Summary):
    """Distributional summaries of log counts, computed separately for each cohort.

    Per cohort and type: mean and standard deviation of ``log1p(count)``, the given
    quantiles, and the fraction of clones containing the type. Per cohort: the same
    statistics of ``log1p(total)``, the fraction of empty clones and the pairwise
    correlations of log counts between types.

    Args:
        quantiles: Quantile levels of the log-count distributions.
    """

    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)

    @staticmethod
    def _describe(values: np.ndarray, quantiles: Sequence[float]) -> list[float]:
        return [values.mean(), values.std(), *np.quantile(values, quantiles)]

    def __call__(self, cohorts: Sequence[Cohort]) -> np.ndarray:
        """See :meth:`Summary.__call__`."""
        out: list[float] = []
        for cohort in cohorts:
            counts = np.asarray(cohort.counts, dtype=float)
            logs = np.log1p(counts)
            total = counts.sum(1)
            for k in range(counts.shape[1]):
                out += self._describe(logs[:, k], self.quantiles)
                out.append(np.mean(counts[:, k] > 0))
            out += self._describe(np.log1p(total), self.quantiles)
            out.append(np.mean(total == 0))
            if counts.shape[1] > 1:
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr = np.corrcoef(logs.T)
                out += list(corr[np.triu_indices(counts.shape[1], 1)])
        return np.nan_to_num(np.asarray(out, dtype=np.float32))


@dataclass
class CohortMeans(Summary):
    """Low-dimensional baseline: mean log count per type and fraction of empty clones."""

    def __call__(self, cohorts: Sequence[Cohort]) -> np.ndarray:
        """See :meth:`Summary.__call__`."""
        out: list[float] = []
        for cohort in cohorts:
            counts = np.asarray(cohort.counts, dtype=float)
            out += list(np.log1p(counts).mean(0))
            out.append(np.mean(counts.sum(1) == 0))
        return np.asarray(out, dtype=np.float32)
