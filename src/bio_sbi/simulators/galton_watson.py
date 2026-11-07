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
  broad priors cost nothing extra (no rejection loop). With a heterogeneous law
  (``law.clone_heterogeneity``) each clone's outcome matrix is first drawn conditional
  on detection by rejection on its exact detection probability (cheap: no clone is
  simulated), then pruned sampling proceeds per clone (:func:`detected_clones`).

Time axis: by default horizons are integer generations (``time_unit = "generations"``).
With ``hours_per_generation = tau`` the process is placed on an hour axis instead
(``time_unit = "hours"``): an hour-valued horizon ``h`` is simulated as
``g = round_half_up(h / tau)`` synchronous generations. Designs, summaries and
reference counts stay keyed by ``h``, so a Galton-Watson inference model can be fitted
to Bellman-Harris data measured at the same hours.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Mapping

import numpy as np
from numba import njit

from .base import INVALID, BranchingSimulator, categorical, draw_founders


@njit(cache=True, nogil=True)
def gw_forward(P, n_out, offspring, founder, sizes, size_probs, generations, n, rng, max_cells):
    """Forward-simulate ``n`` clones; returns ``(n, K)`` true endpoint counts.

    ``P`` is ``(1, K, O)`` (one outcome matrix shared by all clones) or ``(n, K, O)``
    (clone ``c`` uses ``P[c]``, heterogeneous law).
    """
    k_types = P.shape[1]
    shared = P.shape[0] == 1
    out = np.zeros((n, k_types), dtype=np.int64)
    seeds = np.empty(sizes.max(), dtype=np.int64)
    for c in range(n):
        Pc = P[0] if shared else P[c]
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
                        q = Pc[k, o] / mass if mass > 0 else 1.0
                        m = rng.binomial(remaining, min(1.0, max(0.0, q)))
                    remaining -= m
                    mass -= Pc[k, o]
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
    root = founder * s[generations]
    q = root.sum()
    size_w = np.empty(len(sizes))
    for i in range(len(sizes)):
        size_w[i] = size_probs[i] * (1.0 - (1.0 - q) ** sizes[i])
    if size_w.sum() <= 0:
        out = np.zeros((n, k_types), dtype=np.int64)
        out[:] = INVALID
        return out
    return detected_clones(
        P.reshape((1, *P.shape)), s.reshape((1, *s.shape)), root.reshape((1, k_types)),
        np.array([q]), size_w.reshape((1, len(sizes))), n_out, offspring, children,
        n_children, sizes, generations, n, rng, max_cells, max_events,
    )  # fmt: skip


@njit(cache=True, nogil=True)
def detected_clones(
    P, s, root, q, size_w, n_out, offspring, children, n_children, sizes, generations, n, rng,
    max_cells, max_events,
):  # fmt: skip
    """Pruned detection-conditioned sampling of ``n`` clones (see :func:`gw_detected`).

    Per-clone arrays have a leading dimension of 1 (shared by all clones) or ``n``
    (clone ``c`` uses index ``c``; heterogeneous law): outcome matrices ``P``
    ``(., K, O)``, success tables ``s`` ``(., G + 1, K)``, founder-type weights ``root``
    ``(., K)``, founder success probabilities ``q`` ``(.,)`` and founder-count weights
    ``size_w`` ``(., n_sizes)``.
    """
    k_types = P.shape[1]
    shared = P.shape[0] == 1
    out = np.zeros((n, k_types), dtype=np.int64)
    max_kids = max(children.shape[2], sizes.max())
    capacity = (generations + 1) * max_kids + sizes.max() + 2
    stack_type = np.empty(capacity, dtype=np.int64)
    stack_left = np.empty(capacity, dtype=np.int64)
    weights = np.empty(P.shape[2])
    tail = np.empty(max_kids + 1)
    for c in range(n):
        j0 = 0 if shared else c
        Pc, sc, rootc, qc, size_wc = P[j0], s[j0], root[j0], q[j0], size_w[j0]
        counts = np.zeros(k_types, dtype=np.int64)
        size, events, ok = 0, 0, True
        m = sizes[categorical(size_wc, len(sizes), rng)]
        succeeded = False
        for i in range(m):
            p_any = 1.0 - (1.0 - qc) ** (m - i)  # P(>= 1 success among founders i..m-1)
            keep = rng.random() < qc if succeeded else rng.random() * p_any < qc
            if keep:
                succeeded = True
                stack_type[size] = categorical(rootc, k_types, rng)
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
                    fail *= (1.0 - sc[g - 1, j]) ** offspring[k, o, j]
                weights[o] = Pc[k, o] * (1.0 - fail)
            o = categorical(weights, n_out[k], rng)
            kids = n_children[k, o]
            # tail[i] = P(at least one of daughters i..kids-1 succeeds).
            tail[kids] = 0.0
            for i in range(kids - 1, -1, -1):
                tail[i] = 1.0 - (1.0 - tail[i + 1]) * (1.0 - sc[g - 1, children[k, o, i]])
            got = False
            for i in range(kids):
                child = children[k, o, i]
                si = sc[g - 1, child]
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


def detection_success_batch(P, offspring, generations, capture):
    """Vectorised :func:`detection_success` for a batch ``P`` ``(B, K, O)``.

    Padded outcomes (beyond ``n_out``) have probability 0 and contribute nothing.
    Returns ``s`` of shape ``(B, G + 1, K)``.
    """
    batch, k_types = P.shape[0], P.shape[1]
    s = np.empty((batch, generations + 1, k_types))
    s[:, 0] = capture
    for g in range(1, generations + 1):
        # fail[b, k, o] = prod_j (1 - s[b, g-1, j]) ** offspring[k, o, j]
        fail = np.prod((1.0 - s[:, g - 1])[:, None, None, :] ** offspring[None], axis=-1)
        s[:, g] = np.sum(P * (1.0 - fail), axis=-1)
    return s


LOG = logging.getLogger(__name__)

#: Largest tolerated |h / tau - g| before a horizon-rounding warning is emitted.
ROUNDING_TOLERANCE = 0.25

#: Heterogeneous law: Monte-Carlo draws and seed of :meth:`GaltonWatson.detection_probability`.
DETECTION_MC_DRAWS = 4096
DETECTION_MC_SEED = 20240611


class GaltonWatson(BranchingSimulator):
    """Synchronous discrete-generation branching process.

    Physical parameters are the outcome, founder-type and founder-size probabilities.

    Args:
        law, fixed, max_cells, max_events: As in
            :class:`~bio_sbi.simulators.base.BranchingSimulator`.
        hours_per_generation: ``None`` (default): horizons are integer generations and
            ``time_unit == "generations"``. A positive number ``tau``: horizons are
            hours (``time_unit == "hours"``) and ``h`` hours are simulated as
            ``g = floor(h / tau + 0.5)`` generations (round half up, so 2.5 -> 3, not
            Python's banker's rounding). A warning is emitted (once per horizon) when
            ``|h / tau - g| > 0.25``; horizons with ``g < 1`` are rejected. ``tau`` is
            fixed, not inferred.
        max_detection_draws: Heterogeneous law only: budget of clone-matrix draws per
            requested clone in the exact detection-conditioned sampler (see
            :meth:`simulate_detected`); exceeding it marks the cohort invalid.
    """

    def __init__(
        self,
        law: Mapping,
        fixed: Mapping[str, float] | None = None,
        max_cells: int = 100_000,
        max_events: int = 5_000_000,
        hours_per_generation: float | None = None,
        max_detection_draws: int = 1000,
    ) -> None:
        super().__init__(law, fixed, max_cells, max_events)
        self.max_detection_draws = int(max_detection_draws)
        if hours_per_generation is not None:
            hours_per_generation = float(hours_per_generation)
            if not np.isfinite(hours_per_generation) or hours_per_generation <= 0:
                raise ValueError("hours_per_generation must be a positive number or None")
        self.hours_per_generation = hours_per_generation
        self.time_unit = "generations" if hours_per_generation is None else "hours"
        self._warned: set[float] = set()

    def generations(self, horizon: float) -> int:
        """Return the number of synchronous generations simulated for ``horizon``.

        Without ``hours_per_generation`` the horizon must be a positive integer; with
        it, ``horizon`` is in hours and is rounded half up to whole generations.
        """
        h = super().validate_horizon(horizon)
        tau = self.hours_per_generation
        if tau is None:
            if not h.is_integer():
                raise ValueError("Galton-Watson horizons must be integer generations")
            return int(h)
        exact = h / tau
        g = math.floor(exact + 0.5)
        if g < 1:
            raise ValueError(
                f"Horizon {h:g} h is less than half a generation ({tau:g} h per generation)"
            )
        if abs(exact - g) > ROUNDING_TOLERANCE and h not in self._warned:
            self._warned.add(h)
            msg = (
                f"Galton-Watson horizon {h:g} h = {exact:.3g} generations of {tau:g} h, "
                f"simulated as {g} generations ({g * tau:g} h)"
            )
            LOG.warning(msg)
            warnings.warn(msg, stacklevel=2)
        return g

    def validate_horizon(self, horizon: float) -> float:
        """Check ``horizon`` (generations, or hours with ``hours_per_generation``)."""
        self.generations(horizon)
        return float(horizon)

    def _arrays(self, physical):
        P, founder, size_probs = self.law.unpack(self.space, self.validate(physical))
        return P, founder, self.law.founder_sizes, size_probs

    def simulate(self, physical, horizon, n, rng):
        """See :meth:`BranchingSimulator.simulate`."""
        g = self.generations(horizon)
        P, founder, sizes, size_probs = self._arrays(physical)
        if self.heterogeneous:
            P = self.clone_matrices(physical, P, int(n), rng)
        else:
            P = P[None]
        return gw_forward(
            P, self.law.n_outcomes, self.law.offspring, founder, sizes, size_probs, g,
            int(n), rng, self.max_cells,
        )  # fmt: skip

    @property
    def supports_detected_sampling(self) -> bool:
        """Exact detection-conditioned sampling is available."""
        return True

    def simulate_detected(self, physical, horizon, n, capture, rng):
        """See :meth:`BranchingSimulator.simulate_detected`.

        Heterogeneous law: a clone's outcome matrix ``P_c`` must itself be drawn
        conditional on detection, ``p(P_c | detected) ~ Dirichlet(P_c) D(P_c)``, with
        ``D(P_c)`` the clone's exact detection probability. This is done by rejection
        on the *parameters* (draw ``P_c``, accept with probability ``D(P_c)``, computed
        analytically - no clone is simulated), then the same pruned sampler runs with
        each accepted ``P_c``. Exact; the cost is one ``D(P_c)`` evaluation per draw.
        More than ``max_detection_draws`` draws per requested clone make the cohort
        :data:`~bio_sbi.simulators.base.INVALID` (like the rejection budget of the
        forward path).
        """
        g = self.generations(horizon)
        P, founder, sizes, size_probs = self._arrays(physical)
        capture = np.asarray(capture, dtype=float)
        if not self.heterogeneous:
            return gw_detected(
                P, self.law.n_outcomes, self.law.offspring, self.law.children,
                self.law.n_children, founder, sizes, size_probs, g, capture, int(n), rng,
                self.max_cells, self.max_events,
            )  # fmt: skip
        n = int(n)
        kept_P, kept_s, need, draws, batch = [], [], n, 0, n
        while need > 0:
            if draws + batch > n * self.max_detection_draws:
                return np.full((n, len(self.types)), INVALID, dtype=np.int64)
            Pc = self.clone_matrices(physical, P, batch, rng)
            sc = detection_success_batch(Pc, self.law.offspring, g, capture)
            q = sc[:, g] @ founder
            detect = (1.0 - (1.0 - q[:, None]) ** sizes[None]) @ size_probs
            accept = rng.random(batch) < detect
            draws += batch
            kept_P.append(Pc[accept][:need])
            kept_s.append(sc[accept][:need])
            need -= len(kept_P[-1])
            rate = max((n - need) / draws, 1.0 / self.max_detection_draws)
            batch = int(min(max(need / rate * 1.2, 1), n * self.max_detection_draws))
        Pc, sc = np.concatenate(kept_P), np.concatenate(kept_s)
        root = founder[None] * sc[:, g]
        q = root.sum(1)
        size_w = size_probs[None] * (1.0 - (1.0 - q[:, None]) ** sizes[None])
        return detected_clones(
            Pc, sc, root, q, size_w, self.law.n_outcomes, self.law.offspring,
            self.law.children, self.law.n_children, sizes, g, n, rng, self.max_cells,
            self.max_events,
        )  # fmt: skip

    def expected_counts(self, physical, horizon):
        """``E[N] * founder @ M**G`` with ``M`` the mean offspring matrix.

        With ``law.clone_heterogeneity`` this is the *mean model* (clone rows replaced
        by their expectation), not the exact mean over clones (``E[M_c^G] != M^G``).
        """
        g = self.generations(horizon)
        P, founder, sizes, size_probs = self._arrays(physical)
        m = self.law.mean_matrix(P)
        return float(sizes @ size_probs) * founder @ np.linalg.matrix_power(m, g)

    def detection_probability(self, physical, horizon, capture):
        """P(a clone has >= 1 captured cell at ``horizon``).

        Exact for a homogeneous law. With ``law.clone_heterogeneity`` it is a
        Monte-Carlo average of the exact per-clone probability over
        ``DETECTION_MC_DRAWS`` clone matrices from a fixed internal seed (it never
        touches the caller's random stream).
        """
        g = self.generations(horizon)
        P, founder, sizes, size_probs = self._arrays(physical)
        if self.heterogeneous:
            local = np.random.default_rng(DETECTION_MC_SEED)
            Pc = self.clone_matrices(physical, P, DETECTION_MC_DRAWS, local)
            sc = detection_success_batch(Pc, self.law.offspring, g, np.asarray(capture))
            q = sc[:, g] @ founder
            return float(np.mean((1.0 - (1.0 - q[:, None]) ** sizes[None]) @ size_probs))
        s = detection_success(P, self.law.n_outcomes, self.law.offspring, g, np.asarray(capture))
        q = float(founder @ s[g])
        return float(size_probs @ (1.0 - (1.0 - q) ** sizes))
