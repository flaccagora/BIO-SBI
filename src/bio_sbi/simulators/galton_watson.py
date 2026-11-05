"""Discrete-generation multi-type Galton-Watson process.

Every living cell undergoes exactly one event per generation: it draws an outcome from
its type's probability row and is replaced by that outcome's daughters, which act in
the next generation. ``horizon = G`` means G synchronous generations after the founder.

Two samplers are provided:

* :func:`gw_forward` - forward simulation of true counts (multinomial per type, so the
  cost scales with the number of *types*, not cells, per generation);
* :func:`gw_detected` - exact sampling of *captured* counts conditioned on at least
  one captured endpoint cell. Subtrees without captured descendants are pruned
  analytically using the extinction-type recursion, so rare survivors under broad
  priors cost nothing extra (no rejection loop).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .base import INVALID, BranchingSimulator, categorical


@njit(cache=True, nogil=True)
def gw_forward(P, n_out, offspring, founder, generations, n, rng, max_cells):
    """Forward-simulate ``n`` clones; returns ``(n, K)`` true endpoint counts."""
    k_types = P.shape[0]
    out = np.zeros((n, k_types), dtype=np.int64)
    for c in range(n):
        x = np.zeros(k_types, dtype=np.int64)
        x[categorical(founder, k_types, rng)] = 1
        ok = True
        for _ in range(generations):
            new = np.zeros(k_types, dtype=np.int64)
            for k in range(k_types):
                remaining, mass = x[k], 1.0
                for o in range(n_out[k]):
                    if remaining == 0:
                        break
                    if o == n_out[k] - 1:
                        m = remaining
                    else:
                        q = P[k, o] / mass if mass > 0 else 1.0
                        m = rng.binomial(remaining, min(1.0, max(0.0, q)))
                    remaining -= m
                    mass -= P[k, o]
                    for j in range(k_types):
                        new[j] += m * offspring[k, o, j]
            x = new
            if x.sum() > max_cells:
                ok = False
                break
        if ok:
            out[c] = x
        else:
            out[c] = INVALID
    return out


@njit(cache=True, nogil=True)
def detection_success(P, n_out, offspring, generations, capture):
    """``s[g, k]``: P(a type-k cell with g generations left has >= 1 captured descendant)."""
    k_types = P.shape[0]
    s = np.empty((generations + 1, k_types))
    s[0] = capture
    for g in range(1, generations + 1):
        for k in range(k_types):
            total = 0.0
            for o in range(n_out[k]):
                fail = 1.0
                for j in range(k_types):
                    fail *= (1.0 - s[g - 1, j]) ** offspring[k, o, j]
                total += P[k, o] * (1.0 - fail)
            s[g, k] = total
    return s


@njit(cache=True, nogil=True)
def gw_detected(
    P,
    n_out,
    offspring,
    children,
    n_children,
    founder,
    generations,
    capture,
    n,
    rng,
    max_cells,
    max_events,
):
    """Sample captured counts of ``n`` clones conditioned on detection (exact)."""
    k_types = P.shape[0]
    s = detection_success(P, n_out, offspring, generations, capture)
    out = np.zeros((n, k_types), dtype=np.int64)
    max_kids = children.shape[2]
    capacity = (generations + 1) * max_kids + 2
    stack_type = np.empty(capacity, dtype=np.int64)
    stack_left = np.empty(capacity, dtype=np.int64)
    weights = np.empty(P.shape[1])
    tail = np.empty(max_kids + 1)
    root = np.empty(k_types)
    for k in range(k_types):
        root[k] = founder[k] * s[generations, k]
    if root.sum() <= 0:
        out[:] = INVALID
        return out
    for c in range(n):
        counts = np.zeros(k_types, dtype=np.int64)
        # Detection reweights the founder mixture: draw P(founder type | detected).
        stack_type[0], stack_left[0], size, events, ok = (
            categorical(root, k_types, rng),
            generations,
            1,
            0,
            True,
        )
        while size:
            size -= 1
            k, g = stack_type[size], stack_left[size]
            if g == 0:
                counts[k] += 1
                if counts.sum() > max_cells:
                    ok = False
                    break
                continue
            events += 1
            if events > max_events:
                ok = False
                break
            for o in range(n_out[k]):
                fail = 1.0
                for j in range(k_types):
                    fail *= (1.0 - s[g - 1, j]) ** offspring[k, o, j]
                weights[o] = P[k, o] * (1.0 - fail)
            o = categorical(weights, n_out[k], rng)
            m = n_children[k, o]
            # tail[i] = P(at least one of daughters i..m-1 succeeds).
            tail[m] = 0.0
            for i in range(m - 1, -1, -1):
                si = s[g - 1, children[k, o, i]]
                tail[i] = 1.0 - (1.0 - tail[i + 1]) * (1.0 - si)
            succeeded = False
            for i in range(m):
                child = children[k, o, i]
                si = s[g - 1, child]
                if succeeded:
                    keep = rng.random() < si
                else:
                    keep = rng.random() * tail[i] < si
                if keep:
                    succeeded = True
                    stack_type[size], stack_left[size] = child, g - 1
                    size += 1
        if ok:
            out[c] = counts
        else:
            out[c] = INVALID
    return out


class GaltonWatson(BranchingSimulator):
    """Synchronous discrete-generation branching process.

    Physical parameters are the outcome probabilities of every type (see
    :class:`~bio_sbi.simulators.law.OffspringLaw`); horizons are integer generations.
    All constructor arguments are documented in
    :class:`~bio_sbi.simulators.base.BranchingSimulator`.
    """

    time_unit = "generations"

    def validate_horizon(self, horizon: float) -> float:
        """Require a positive integer number of generations."""
        h = super().validate_horizon(horizon)
        if not h.is_integer():
            raise ValueError("Galton-Watson horizons must be integer generations")
        return h

    def _kernel_args(self, physical):
        p = self.validate(physical)
        return self.law.probability_matrix(p), self.law.n_outcomes, self.law.offspring

    def simulate(self, physical, horizon, n, rng):
        """See :meth:`BranchingSimulator.simulate`."""
        g = int(self.validate_horizon(horizon))
        P, n_out, off = self._kernel_args(physical)
        return gw_forward(P, n_out, off, self.founder, g, int(n), rng, self.max_cells)

    @property
    def supports_detected_sampling(self) -> bool:
        """Exact detection-conditioned sampling is available."""
        return True

    def simulate_detected(self, physical, horizon, n, capture, rng):
        """See :meth:`BranchingSimulator.simulate_detected`."""
        g = int(self.validate_horizon(horizon))
        P, n_out, off = self._kernel_args(physical)
        return gw_detected(
            P,
            n_out,
            off,
            self.law.children,
            self.law.n_children,
            self.founder,
            g,
            np.asarray(capture, dtype=float),
            int(n),
            rng,
            self.max_cells,
            self.max_events,
        )

    def expected_counts(self, physical, horizon):
        """``founder @ M**G`` with ``M`` the mean offspring matrix."""
        g = int(self.validate_horizon(horizon))
        m = self.law.mean_matrix(self.validate(physical)[: self.law.n_probabilities])
        return self.founder @ np.linalg.matrix_power(m, g)

    def detection_probability(self, physical, horizon, capture):
        """P(a clone has >= 1 captured cell at ``horizon``) under the founder mixture."""
        g = int(self.validate_horizon(horizon))
        P, n_out, off = self._kernel_args(physical)
        s = detection_success(P, n_out, off, g, np.asarray(capture, dtype=float))
        return float(self.founder @ s[g])
