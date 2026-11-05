"""Discrete-generation multi-type Galton-Watson process.

Every living cell undergoes exactly one event per generation: it draws an outcome from
its type's probability row and is replaced by that outcome's daughters, which act in
the next generation. ``horizon = G`` means G synchronous generations after seeding.

Two samplers are provided:

* :func:`gw_forward` - forward simulation of true counts (multinomial per type, so the
  cost per generation scales with the number of *types*, not cells);
* :func:`gw_detected` - exact sampling of *captured* counts conditioned on at least
  one captured endpoint cell. Founders and subtrees without captured descendants are
  pruned analytically using the extinction-type recursion, so rare survivors under
  broad priors cost nothing extra (no rejection loop).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .base import INVALID, BranchingSimulator, categorical, draw_founders


@njit(cache=True, nogil=True)
def gw_forward(P, n_out, offspring, founder, sizes, size_probs, generations, n, rng, max_cells):
    """Forward-simulate ``n`` clones; returns ``(n, K)`` true endpoint counts."""
    k_types = P.shape[0]
    out = np.zeros((n, k_types), dtype=np.int64)
    seeds = np.empty(sizes.max(), dtype=np.int64)
    for c in range(n):
        x = np.zeros(k_types, dtype=np.int64)
        for i in range(draw_founders(founder, sizes, size_probs, rng, seeds)):
            x[seeds[i]] += 1
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
    sizes,
    size_probs,
    generations,
    capture,
    n,
    rng,
    max_cells,
    max_events,
):
    """Sample captured counts of ``n`` clones conditioned on detection (exact).

    Each founder independently "succeeds" (has a captured descendant) with probability
    ``q = sum_k founder[k] s[G, k]``. The founder count is drawn with weights
    ``P(m) (1 - (1 - q)^m)``, founder success flags conditional on at least one
    success, and each successful founder's type with weights ``founder[k] s[G, k]``.
    Below that, outcomes and daughters are drawn conditional on success recursively.
    """
    k_types = P.shape[0]
    s = detection_success(P, n_out, offspring, generations, capture)
    out = np.zeros((n, k_types), dtype=np.int64)
    max_kids = max(children.shape[2], sizes.max())
    capacity = (generations + 1) * max_kids + sizes.max() + 2
    stack_type = np.empty(capacity, dtype=np.int64)
    stack_left = np.empty(capacity, dtype=np.int64)
    weights = np.empty(P.shape[1])
    tail = np.empty(max_kids + 1)
    root = founder * s[generations]
    q = root.sum()
    size_w = np.empty(len(sizes))
    for i in range(len(sizes)):
        size_w[i] = size_probs[i] * (1.0 - (1.0 - q) ** sizes[i])
    if size_w.sum() <= 0:
        out[:] = INVALID
        return out
    for c in range(n):
        counts = np.zeros(k_types, dtype=np.int64)
        size, events, ok = 0, 0, True
        m = sizes[categorical(size_w, len(sizes), rng)]
        succeeded = False
        for i in range(m):
            p_any = 1.0 - (1.0 - q) ** (m - i)  # P(>= 1 success among founders i..m-1)
            keep = rng.random() < q if succeeded else rng.random() * p_any < q
            if keep:
                succeeded = True
                stack_type[size] = categorical(root, k_types, rng)
                stack_left[size] = generations
                size += 1
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
            kids = n_children[k, o]
            # tail[i] = P(at least one of daughters i..kids-1 succeeds).
            tail[kids] = 0.0
            for i in range(kids - 1, -1, -1):
                tail[i] = 1.0 - (1.0 - tail[i + 1]) * (1.0 - s[g - 1, children[k, o, i]])
            got = False
            for i in range(kids):
                child = children[k, o, i]
                si = s[g - 1, child]
                keep = rng.random() < si if got else rng.random() * tail[i] < si
                if keep:
                    got = True
                    stack_type[size], stack_left[size] = child, g - 1
                    size += 1
        if ok:
            out[c] = counts
        else:
            out[c] = INVALID
    return out


class GaltonWatson(BranchingSimulator):
    """Synchronous discrete-generation branching process.

    Physical parameters are the outcome, founder-type and founder-size probabilities;
    horizons are integer generations. Constructor arguments are documented in
    :class:`~bio_sbi.simulators.base.BranchingSimulator`.
    """

    time_unit = "generations"

    def validate_horizon(self, horizon: float) -> float:
        """Require a positive integer number of generations."""
        h = super().validate_horizon(horizon)
        if not h.is_integer():
            raise ValueError("Galton-Watson horizons must be integer generations")
        return h

    def _arrays(self, physical):
        P, founder, size_probs = self.law.unpack(self.space, self.validate(physical))
        return P, founder, self.law.founder_sizes, size_probs

    def simulate(self, physical, horizon, n, rng):
        """See :meth:`BranchingSimulator.simulate`."""
        g = int(self.validate_horizon(horizon))
        P, founder, sizes, size_probs = self._arrays(physical)
        return gw_forward(
            P, self.law.n_outcomes, self.law.offspring, founder, sizes, size_probs, g,
            int(n), rng, self.max_cells,
        )  # fmt: skip

    @property
    def supports_detected_sampling(self) -> bool:
        """Exact detection-conditioned sampling is available."""
        return True

    def simulate_detected(self, physical, horizon, n, capture, rng):
        """See :meth:`BranchingSimulator.simulate_detected`."""
        g = int(self.validate_horizon(horizon))
        P, founder, sizes, size_probs = self._arrays(physical)
        return gw_detected(
            P, self.law.n_outcomes, self.law.offspring, self.law.children,
            self.law.n_children, founder, sizes, size_probs, g,
            np.asarray(capture, dtype=float), int(n), rng, self.max_cells, self.max_events,
        )  # fmt: skip

    def expected_counts(self, physical, horizon):
        """``E[N] * founder @ M**G`` with ``M`` the mean offspring matrix."""
        g = int(self.validate_horizon(horizon))
        P, founder, sizes, size_probs = self._arrays(physical)
        m = self.law.mean_matrix(P)
        return float(sizes @ size_probs) * founder @ np.linalg.matrix_power(m, g)

    def detection_probability(self, physical, horizon, capture):
        """P(a clone has >= 1 captured cell at ``horizon``)."""
        g = int(self.validate_horizon(horizon))
        P, founder, sizes, size_probs = self._arrays(physical)
        s = detection_success(P, self.law.n_outcomes, self.law.offspring, g, np.asarray(capture))
        q = float(founder @ s[g])
        return float(size_probs @ (1.0 - (1.0 - q) ** sizes))
