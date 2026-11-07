"""Continuous-time, age-dependent multi-type Bellman-Harris process.

Each cell of type ``k`` lives for an independent ``Gamma(shape_k, mean_k / shape_k)``
lifetime (mean ``mean_k``, coefficient of variation ``1 / sqrt(shape_k)``). At the end
of its life it draws an outcome from its type's probability row and is replaced by that
outcome's daughters, each born at that instant with age zero. All founder cells are
born at ``t = 0``. ``shape = 1`` recovers the Markov (exponential-lifetime) process;
large shapes approach synchronous generations with period ``mean_k``.

Compared with :class:`~bio_sbi.simulators.galton_watson.GaltonWatson`, the extra
physical parameters are the mean lifetimes per type (log-uniform priors; can be fixed
like any other parameter). Lifetime shapes are fixed by configuration. ``persist``
means "replaced by one cell of the same type", i.e. a lifetime renewal.

With ``law.clone_heterogeneity`` every clone draws its own outcome rows (Dirichlet
around the law's rows, see :mod:`bio_sbi.simulators.law`) before it is simulated;
lifetimes are not affected. Detection is handled by forward simulation plus rejection
as for the homogeneous model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numba import njit

from .base import INVALID, BranchingSimulator, categorical, draw_founders
from .law import ScalarParameter


@njit(cache=True, nogil=True)
def bh_forward(
    P, n_out, children, n_children, founder, sizes, size_probs, shape, mean, horizon, n, rng,
    max_cells, max_events,
):  # fmt: skip
    """Simulate ``n`` clones depth-first; returns ``(n, K)`` true counts at ``horizon``.

    ``P`` is ``(1, K, O)`` (one outcome matrix shared by all clones) or ``(n, K, O)``
    (clone ``c`` uses ``P[c]``, heterogeneous law).
    """
    k_types = P.shape[1]
    shared = P.shape[0] == 1
    out = np.zeros((n, k_types), dtype=np.int64)
    capacity = max_cells + children.shape[2] + sizes.max() + 1
    stack_type = np.empty(capacity, dtype=np.int64)
    stack_time = np.empty(capacity)
    for c in range(n):
        Pc = P[0] if shared else P[c]
        counts = np.zeros(k_types, dtype=np.int64)
        size = draw_founders(founder, sizes, size_probs, rng, stack_type)
        stack_time[:size] = 0.0
        events, alive, ok = 0, 0, True
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
            o = categorical(Pc[k], n_out[k], rng)
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


def _per_type(value, types, name):
    """Broadcast a single value to every type, or check a ``{type: value}`` mapping."""
    if isinstance(value, Mapping):
        if set(value) != set(types):
            raise ValueError(f"{name} needs exactly one entry per type {types}")
        return dict(value)
    return dict.fromkeys(types, value)


class BellmanHarris(BranchingSimulator):
    """Age-dependent branching process with Gamma-distributed lifetimes.

    Args:
        law, fixed, max_cells, max_events: As in
            :class:`~bio_sbi.simulators.base.BranchingSimulator`. ``fixed`` may contain
            ``"<type>.mean_lifetime"``.
        lifetime_shape: Gamma shape (fixed, not inferred): one value for all types or
            ``{type: shape}``.
        lifetime_mean_bounds: Log-uniform prior bounds on each type's mean lifetime:
            one ``[low, high]`` for all types or ``{type: [low, high]}``.
        time_unit: Time-axis label; data and inference simulators must agree on it.
    """

    def __init__(
        self,
        law: Mapping,
        lifetime_shape: float | Mapping[str, float],
        lifetime_mean_bounds: Sequence[float] | Mapping[str, Sequence[float]],
        fixed: Mapping[str, float] | None = None,
        max_cells: int = 100_000,
        max_events: int = 5_000_000,
        time_unit: str = "hours",
    ) -> None:
        types = tuple(law["types"])
        shape = _per_type(lifetime_shape, types, "lifetime_shape")
        bounds = _per_type(lifetime_mean_bounds, types, "lifetime_mean_bounds")
        scalars = tuple(
            ScalarParameter(f"{t}.mean_lifetime", *map(float, bounds[t])) for t in types
        )
        if any(not 0 < s.low < s.high for s in scalars):
            raise ValueError("Lifetime bounds must satisfy 0 < low < high")
        super().__init__(law, fixed, max_cells, max_events, scalars)
        self.time_unit = time_unit
        self.shape = np.array([float(shape[t]) for t in self.types])
        if np.any(self.shape <= 0):
            raise ValueError("Lifetime shapes must be positive")

    def simulate(self, physical, horizon, n, rng):
        """See :meth:`BranchingSimulator.simulate`."""
        h = self.validate_horizon(horizon)
        p = self.validate(physical)
        P, founder, size_probs = self.law.unpack(self.space, p)
        P = self.clone_matrices(p, P, int(n), rng) if self.heterogeneous else P[None]
        return bh_forward(
            P, self.law.n_outcomes, self.law.children, self.law.n_children, founder,
            self.law.founder_sizes, size_probs, self.shape, self.scalar_values(p), h,
            int(n), rng, self.max_cells, self.max_events,
        )  # fmt: skip

    def expected_counts(self, physical, horizon):
        """Exact means for exponential lifetimes (``shape == 1``) via a matrix exponential.

        With ``law.clone_heterogeneity`` this is the *mean model* (clone rows replaced by
        their expectation ``p``), not the exact mean over clones (which is larger: the
        mean of ``expm`` over Dirichlet rows exceeds ``expm`` of the mean).
        """
        from scipy.linalg import expm

        if not np.allclose(self.shape, 1.0):
            raise NotImplementedError("Closed-form means are only available for shape == 1")
        p = self.validate(physical)
        P, founder, size_probs = self.law.unpack(self.space, p)
        rates = 1.0 / self.scalar_values(p)
        generator = rates[:, None] * (self.law.mean_matrix(P) - np.eye(len(self.types)))
        mean_founders = float(self.law.founder_sizes @ size_probs)
        return mean_founders * founder @ expm(generator * self.validate_horizon(horizon))
