"""Continuous-time, age-dependent multi-type Bellman-Harris process.

Each cell of type ``k`` lives for an independent ``Gamma(shape_k, mean_k / shape_k)``
lifetime (mean ``mean_k``, coefficient of variation ``1 / sqrt(shape_k)``). At the end
of its life it draws an outcome from its type's probability row and is replaced by that
outcome's daughters, each born at that instant with age zero. ``shape = 1`` recovers the
Markov (exponential-lifetime) branching process; large shapes approach synchronous
generations with period ``mean_k``.

Compared with :class:`~bio_sbi.simulators.galton_watson.GaltonWatson`, the extra
physical parameters are the mean lifetimes per type (log-uniform priors). Lifetime
shapes are fixed by configuration. Horizons are continuous times in the same unit as
the lifetimes (hours in the shipped config).

Outcomes such as ``persist`` keep their meaning as "replaced by one cell of the same
type", i.e. a lifetime renewal, so the same type graph can be used by both models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numba import njit

from .base import INVALID, BranchingSimulator, ScalarParameter, categorical


@njit(cache=True, nogil=True)
def bh_forward(
    P, n_out, children, n_children, founder, shape, mean, horizon, n, rng, max_cells, max_events
):
    """Simulate ``n`` clones depth-first; returns ``(n, K)`` true counts at ``horizon``."""
    k_types = P.shape[0]
    out = np.zeros((n, k_types), dtype=np.int64)
    capacity = max_cells + children.shape[2] + 1
    stack_type = np.empty(capacity, dtype=np.int64)
    stack_time = np.empty(capacity)
    for c in range(n):
        counts = np.zeros(k_types, dtype=np.int64)
        stack_type[0], stack_time[0], size, events, alive, ok = (
            categorical(founder, k_types, rng),
            0.0,
            1,
            0,
            0,
            True,
        )
        while size:
            size -= 1
            k, born = stack_type[size], stack_time[size]
            death = born + rng.gamma(shape[k], mean[k] / shape[k])
            if death >= horizon:
                counts[k] += 1
                alive += 1
                if alive > max_cells:
                    ok = False
                    break
                continue
            events += 1
            if events > max_events:
                ok = False
                break
            o = categorical(P[k], n_out[k], rng)
            for i in range(n_children[k, o]):
                if size >= capacity:
                    ok = False
                    break
                stack_type[size], stack_time[size] = children[k, o, i], death
                size += 1
            if not ok:
                break
        if ok:
            out[c] = counts
        else:
            out[c] = INVALID
    return out


class BellmanHarris(BranchingSimulator):
    """Age-dependent branching process with Gamma-distributed lifetimes.

    Args:
        types, outcomes, founder, concentration, max_cells, max_events: As in
            :class:`~bio_sbi.simulators.base.BranchingSimulator`.
        lifetime_shape: ``{type: shape}`` Gamma shape per type (fixed, not inferred).
        lifetime_mean_bounds: ``{type: [low, high]}`` log-uniform prior bounds on the
            mean lifetime of each type (inferred).
        time_unit: Label for the time axis (documentation and plots only).
    """

    def __init__(
        self,
        types: Sequence[str],
        outcomes: Mapping[str, Mapping[str, Sequence[int]]],
        founder: Sequence[float],
        lifetime_shape: Mapping[str, float],
        lifetime_mean_bounds: Mapping[str, Sequence[float]],
        concentration: Mapping[str, Sequence[float]] | None = None,
        max_cells: int = 100_000,
        max_events: int = 5_000_000,
        time_unit: str = "hours",
    ) -> None:
        super().__init__(types, outcomes, founder, concentration, max_cells, max_events)
        self.time_unit = time_unit
        if set(lifetime_shape) != set(self.types) or set(lifetime_mean_bounds) != set(self.types):
            raise ValueError("lifetime_shape and lifetime_mean_bounds need one entry per type")
        self.shape = np.array([float(lifetime_shape[t]) for t in self.types])
        if np.any(self.shape <= 0):
            raise ValueError("Lifetime shapes must be positive")
        self._scalars = tuple(
            ScalarParameter(f"{t}.mean_lifetime", *map(float, lifetime_mean_bounds[t]), log=True)
            for t in self.types
        )
        if any(not 0 < s.low < s.high for s in self._scalars):
            raise ValueError("Lifetime bounds must satisfy 0 < low < high")

    @property
    def scalar_parameters(self) -> tuple[ScalarParameter, ...]:
        """Mean lifetime of each type."""
        return self._scalars

    def simulate(self, physical, horizon, n, rng):
        """See :meth:`BranchingSimulator.simulate`."""
        h = self.validate_horizon(horizon)
        p = self.validate(physical)
        k = self.law.n_probabilities
        return bh_forward(
            self.law.probability_matrix(p[:k]),
            self.law.n_outcomes,
            self.law.children,
            self.law.n_children,
            self.founder,
            self.shape,
            p[k:],
            h,
            int(n),
            rng,
            self.max_cells,
            self.max_events,
        )

    def expected_counts(self, physical, horizon):
        """Exact means for exponential lifetimes (``shape == 1``) via a matrix exponential."""
        from scipy.linalg import expm

        if not np.allclose(self.shape, 1.0):
            raise NotImplementedError("Closed-form means are only available for shape == 1")
        p = self.validate(physical)
        k = self.law.n_probabilities
        rates = 1.0 / p[k:]
        generator = rates[:, None] * (self.law.mean_matrix(p[:k]) - np.eye(len(self.types)))
        return self.founder @ expm(generator * self.validate_horizon(horizon))
