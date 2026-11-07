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

Density dependence (carrying capacity)
--------------------------------------
With ``density_dependence: {capacity_bounds: [low, high]}`` a cell of type ``k``
born when its clone has ``N`` living cells gets a Gamma lifetime of the same shape
with mean ``mean_k * (1 + N / K)``: larger clones divide more slowly, which bends the
growth curve and compresses the spread of clone sizes. ``N`` is the number of living
cells of the clone *right after* the birth event, i.e. including the newborn cell and
its sibling(s) and every other cell still alive (persisting/renewed and switched cells
included; dead cells and the parent whose life just ended excluded). Founders are born
together at ``t = 0`` with ``N`` = number of founders. A cell's lifetime is fixed at
birth (later growth of the clone does not slow it down further). ``K``
(``density.capacity``, theta ``density.log_capacity``) is one scalar parameter with a
log-uniform prior on ``capacity_bounds``, stored after the mean lifetimes and before
the law's ``clone.kappa``; ``K -> inf`` is the constant-lifetime model. The option is
off by default (``None``), in which case the original depth-first kernel
:func:`bh_forward` runs unchanged (bit-identical output). With it on,
:func:`bh_density_forward` processes the events of a clone in time order (binary heap
of end-of-life times), so ``N`` is the true population at each birth. It composes with
founder rows, multi-cell founders and ``clone_heterogeneity`` (per-clone outcome rows;
``K`` is shared by all clones). Detection: forward simulation plus rejection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numba import njit

from .base import INVALID, BranchingSimulator, categorical, draw_founders
from .law import ScalarParameter

#: Name of the density-dependence (carrying-capacity) scalar parameter.
CAPACITY = "density.capacity"
#: Clones and seed of the Monte-Carlo :meth:`BellmanHarris.expected_counts` (density
#: dependence only).
DENSITY_MC_CLONES = 4000
DENSITY_MC_SEED = 20261107


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


@njit(cache=True, nogil=True)
def _heap_push(times, kinds, size, t, k):
    """Push ``(t, k)`` onto the min-heap ``times[:size]``; return the new size."""
    i = size
    while i > 0:
        parent = (i - 1) >> 1
        if times[parent] <= t:
            break
        times[i], kinds[i] = times[parent], kinds[parent]
        i = parent
    times[i], kinds[i] = t, k
    return size + 1


@njit(cache=True, nogil=True)
def _heap_pop(times, kinds, size):
    """Remove the root of the min-heap ``times[:size]``; return the new size."""
    size -= 1
    t, k = times[size], kinds[size]
    i = 0
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and times[child + 1] < times[child]:
            child += 1
        if times[child] >= t:
            break
        times[i], kinds[i] = times[child], kinds[child]
        i = child
    if size > 0:
        times[i], kinds[i] = t, k
    return size


@njit(cache=True, nogil=True)
def bh_density_forward(
    P, n_out, children, n_children, founder, sizes, size_probs, shape, mean, capacity,
    horizon, n, rng, max_cells, max_events,
):  # fmt: skip
    """Density-dependent Bellman-Harris, event-driven; returns ``(n, K)`` counts.

    A cell born when its clone has ``N`` living cells (after the birth, newborns
    included) lives ``Gamma(shape_k, mean_k * (1 + N / capacity) / shape_k)``.
    Events are processed in time order with a binary min-heap of end-of-life times;
    the cells left in the heap once its minimum reaches ``horizon`` are the living
    cells at the endpoint. ``capacity = inf`` is the constant-lifetime model (in
    distribution, not draw for draw: :func:`bh_forward` uses another draw order).
    A clone is invalid when its living cells exceed ``max_cells`` at any time (not
    only at the endpoint) or its events exceed ``max_events``.
    """
    k_types = P.shape[1]
    shared = P.shape[0] == 1
    out = np.zeros((n, k_types), dtype=np.int64)
    room = max_cells + children.shape[2] + sizes.max() + 1
    heap_time = np.empty(room)
    heap_kind = np.empty(room, dtype=np.int64)
    founders = np.empty(sizes.max(), dtype=np.int64)
    for c in range(n):
        Pc = P[0] if shared else P[c]
        m = draw_founders(founder, sizes, size_probs, rng, founders)
        alive, size, events = m, 0, 0
        ok = m <= max_cells
        factor = 1.0 + alive / capacity
        for i in range(m):
            k = founders[i]
            life = rng.gamma(shape[k], mean[k] * factor / shape[k])
            size = _heap_push(heap_time, heap_kind, size, life, k)
        while ok and size > 0 and heap_time[0] < horizon:
            t, k = heap_time[0], heap_kind[0]
            size = _heap_pop(heap_time, heap_kind, size)
            events += 1
            if events > max_events:
                ok = False
                break
            o = categorical(Pc[k], n_out[k], rng)
            alive += n_children[k, o] - 1
            if alive > max_cells:
                ok = False
                break
            factor = 1.0 + alive / capacity
            for i in range(n_children[k, o]):
                kid = children[k, o, i]
                life = rng.gamma(shape[kid], mean[kid] * factor / shape[kid])
                size = _heap_push(heap_time, heap_kind, size, t + life, kid)
        if ok:
            for i in range(size):
                out[c, heap_kind[i]] += 1
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
            ``"<type>.mean_lifetime"`` and ``"density.capacity"``.
        lifetime_shape: Gamma shape (fixed, not inferred): one value for all types or
            ``{type: shape}``.
        lifetime_mean_bounds: Log-uniform prior bounds on each type's mean lifetime:
            one ``[low, high]`` for all types or ``{type: [low, high]}``.
        time_unit: Time-axis label; data and inference simulators must agree on it.
        density_dependence: ``None`` (default, constant mean lifetimes) or
            ``{capacity_bounds: [low, high]}`` (``True`` = ``[10, 1e5]``): mean
            lifetimes scale with ``1 + N / K`` (module docstring), ``K`` being the
            inferred scalar ``density.capacity`` with a log-uniform prior.
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
        density_dependence: Mapping | bool | None = None,
    ) -> None:
        types = tuple(law["types"])
        shape = _per_type(lifetime_shape, types, "lifetime_shape")
        bounds = _per_type(lifetime_mean_bounds, types, "lifetime_mean_bounds")
        scalars = tuple(
            ScalarParameter(f"{t}.mean_lifetime", *map(float, bounds[t])) for t in types
        )
        if any(not 0 < s.low < s.high for s in scalars):
            raise ValueError("Lifetime bounds must satisfy 0 < low < high")
        self.density_dependent = density_dependence not in (None, False)
        if self.density_dependent:
            spec = {} if density_dependence is True else dict(density_dependence)
            unknown = set(spec) - {"capacity_bounds"}
            if unknown:
                raise ValueError(f"Unknown density_dependence keys {sorted(unknown)}")
            low, high = map(float, spec.get("capacity_bounds", (10.0, 1e5)))
            if not 0 < low < high < np.inf:
                raise ValueError("capacity_bounds must satisfy 0 < low < high < inf")
            scalars += (ScalarParameter(CAPACITY, low, high, log=True),)
        super().__init__(law, fixed, max_cells, max_events, scalars)
        self.time_unit = time_unit
        self.shape = np.array([float(shape[t]) for t in self.types])
        if np.any(self.shape <= 0):
            raise ValueError("Lifetime shapes must be positive")

    def mean_lifetimes(self, physical) -> np.ndarray:
        """Mean lifetime per type (of a cell born into an empty clone, ``N / K -> 0``)."""
        return self.scalar_values(physical)[: len(self.types)]

    def capacity(self, physical) -> float:
        """Carrying-capacity scale ``K`` (``inf`` without density dependence)."""
        if not self.density_dependent:
            return np.inf
        return float(self.scalar_values(physical)[len(self.types)])

    def simulate(self, physical, horizon, n, rng):
        """See :meth:`BranchingSimulator.simulate`."""
        h = self.validate_horizon(horizon)
        p = self.validate(physical)
        P, founder, size_probs = self.law.unpack(self.space, p)
        P = self.clone_matrices(p, P, int(n), rng) if self.heterogeneous else P[None]
        if self.density_dependent:
            return bh_density_forward(
                P, self.law.n_outcomes, self.law.children, self.law.n_children, founder,
                self.law.founder_sizes, size_probs, self.shape, self.mean_lifetimes(p),
                self.capacity(p), h, int(n), rng, self.max_cells, self.max_events,
            )  # fmt: skip
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

        With density dependence there is no closed form (a cell's rate depends on its
        clone's size, so the means do not solve a linear ODE). The result is then a
        Monte-Carlo estimate for any lifetime shape: the mean over
        :data:`DENSITY_MC_CLONES` clones simulated with the fixed seed
        :data:`DENSITY_MC_SEED` (repeated calls agree), standard error about
        ``sd / sqrt(DENSITY_MC_CLONES)`` per type; clone heterogeneity is averaged over
        (the exact mean over clones, not the mean model). The constant-lifetime means
        (``K -> inf``) are an upper bound.
        """
        from scipy.linalg import expm

        if self.density_dependent:
            rng = np.random.default_rng(DENSITY_MC_SEED)
            counts = self.simulate(physical, horizon, DENSITY_MC_CLONES, rng)
            if np.any(counts == INVALID):
                raise ValueError("Monte-Carlo expected counts exceeded the simulation limits")
            return counts.mean(0).astype(float)
        if not np.allclose(self.shape, 1.0):
            raise NotImplementedError("Closed-form means are only available for shape == 1")
        p = self.validate(physical)
        P, founder, size_probs = self.law.unpack(self.space, p)
        rates = 1.0 / self.mean_lifetimes(p)
        generator = rates[:, None] * (self.law.mean_matrix(P) - np.eye(len(self.types)))
        mean_founders = float(self.law.founder_sizes @ size_probs)
        return mean_founders * founder @ expm(generator * self.validate_horizon(horizon))
