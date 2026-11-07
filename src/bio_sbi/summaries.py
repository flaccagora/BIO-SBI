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

    def feature_names(self, horizons: Sequence[float], labels: Sequence[str]) -> list[str]:
        """Names of the summary features for cohorts at ``horizons`` (for reports).

        Subclasses override this; the default numbers the features.
        """
        dummy = [Cohort(h, np.ones((2, len(labels)), dtype=np.int64)) for h in horizons]
        return [f"x{i}" for i in range(len(self(dummy)))]


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

    def _cohort(self, counts: np.ndarray, empty: float | None = None) -> list[float]:
        """Statistics of one ``(n_clones, n_types)`` float count matrix.

        ``empty`` overrides the empty-clone fraction (default: computed from ``counts``).
        """
        out: list[float] = []
        logs = np.log1p(counts)
        total = counts.sum(1)
        for k in range(counts.shape[1]):
            out += self._describe(logs[:, k], self.quantiles)
            out.append(np.mean(counts[:, k] > 0))
        out += self._describe(np.log1p(total), self.quantiles)
        out.append(np.mean(total == 0) if empty is None else empty)
        if counts.shape[1] > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(logs.T)
            out += list(corr[np.triu_indices(counts.shape[1], 1)])
        return out

    def __call__(self, cohorts: Sequence[Cohort]) -> np.ndarray:
        """See :meth:`Summary.__call__`."""
        out: list[float] = []
        for cohort in cohorts:
            out += self._cohort(np.asarray(cohort.counts, dtype=float))
        return np.nan_to_num(np.asarray(out, dtype=np.float32))

    def feature_names(self, horizons: Sequence[float], labels: Sequence[str]) -> list[str]:
        """See :meth:`Summary.feature_names` (``h<horizon>.<label|total>.<statistic>``)."""
        stats = ["mean_log", "sd_log", *[f"q{q:g}_log" for q in self.quantiles]]
        names: list[str] = []
        for h in horizons:
            for label in labels:
                names += [f"h{h:g}.{label}.{s}" for s in stats] + [f"h{h:g}.{label}.present"]
            names += [f"h{h:g}.total.{s}" for s in stats] + [f"h{h:g}.total.empty"]
            names += [f"h{h:g}.corr_{a}_{b}" for i, a in enumerate(labels) for b in labels[i + 1 :]]
        return names


@dataclass
class SurvivorMoments(CohortMoments):
    """:class:`CohortMoments` of the surviving (non-empty) clones plus the extinct fraction.

    With ``observation=all_clones`` extinct organoids are recorded as all-zero rows.
    :class:`CohortMoments` mixes those rows into every statistic, so the extinct
    fraction leaks into all log-count moments, presences and correlations. This
    variant computes, per cohort, every :class:`CohortMoments` statistic on the clones
    with a non-zero total only, and reports the fraction of empty (extinct) clones
    among *all* clones in the slot where :class:`CohortMoments` has ``total.empty``
    (which would be identically 0 for survivors), named ``total.extinct``.

    The vector therefore has the same length and layout as :class:`CohortMoments`, and
    on data without empty clones (e.g. ``selection: detected``) it is bit-identical to
    it; only the feature name of the empty-fraction slot differs.

    A cohort whose clones are all empty gets 0 for every survivor statistic (the value
    of ``log1p(0)``, zero presence and correlation) and an extinct fraction of 1. A
    single survivor gives zero standard deviations and correlations (as in
    :class:`CohortMoments`). No feature is NaN.
    """

    def __call__(self, cohorts: Sequence[Cohort]) -> np.ndarray:
        """See :meth:`Summary.__call__`."""
        out: list[float] = []
        for cohort in cohorts:
            counts = np.asarray(cohort.counts, dtype=float)
            alive = counts.sum(1) > 0
            extinct = float(np.mean(~alive)) if len(counts) else 0.0
            if alive.any():
                out += self._cohort(counts[alive], empty=extinct)
            else:  # no survivors: statistics of a single all-zero clone
                out += self._cohort(np.zeros((1, counts.shape[1])), empty=extinct)
        return np.nan_to_num(np.asarray(out, dtype=np.float32))

    def feature_names(self, horizons: Sequence[float], labels: Sequence[str]) -> list[str]:
        """See :meth:`Summary.feature_names` (as :class:`CohortMoments`, ``total.extinct``)."""
        return [
            n.replace(".total.empty", ".total.extinct")
            for n in super().feature_names(horizons, labels)
        ]


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

    def feature_names(self, horizons: Sequence[float], labels: Sequence[str]) -> list[str]:
        """See :meth:`Summary.feature_names`."""
        return [
            name
            for h in horizons
            for name in (*[f"h{h:g}.{label}.mean_log" for label in labels], f"h{h:g}.empty")
        ]
