"""Abstract simulator contract and shared kernel helpers.

A simulator combines

1. an :class:`~bio_sbi.simulators.law.OffspringLaw` (types, outcomes, founders),
2. a :class:`~bio_sbi.simulators.law.ParameterSpace` (physical parameters, fixed
   values, inference coordinates ``theta`` and their prior), and
3. a **forward model** returning true endpoint counts of ``n`` independent clones
   (:meth:`BranchingSimulator.simulate`), optionally with an exact sampler conditioned
   on detection (:meth:`BranchingSimulator.simulate_detected`).

To add a model, subclass :class:`BranchingSimulator`, implement ``simulate`` (and
optionally ``simulate_detected``/``expected_counts``), pass any scalar parameters to
``super().__init__``, and add ``conf/simulator/<name>.yaml`` pointing at the class.

Organoid-level heterogeneity (``law.clone_heterogeneity``, see
:mod:`bio_sbi.simulators.law`) is handled here for every simulator: the law adds the
scalar ``clone.kappa`` *after* the simulator's own scalars, and
:meth:`BranchingSimulator.clone_matrices` draws one outcome matrix per clone, which
the kernels consume instead of the shared matrix.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from numba import njit

from .law import OffspringLaw, ParameterSpace, ScalarParameter, flatten

#: Marker written into a clone's count row when a resource limit was exceeded.
INVALID = -1


class BranchingSimulator(ABC):
    """Base class for multi-type branching simulators.

    Args:
        law: Law config: ``types``, ``outcomes``, ``max_daughters``, ``founder_sizes``,
            ``fixed``, ``concentration`` (see :class:`OffspringLaw`).
        fixed: Extra fixed values for simulator-specific (scalar) parameters; merged
            with ``law["fixed"]``.
        max_cells: Per-clone population cap. Exceeding it marks the clone invalid
            (the whole dataset is then dropped from training, never truncated).
        max_events: Per-clone cap on the number of simulated cell events.
        scalars: Simulator-specific scalar parameters (set by subclasses).
    """

    #: Unit of ``horizon``; data and inference simulators must agree on it.
    time_unit: str = "generations"

    def __init__(
        self,
        law: Mapping,
        fixed: Mapping[str, float] | None = None,
        max_cells: int = 100_000,
        max_events: int = 5_000_000,
        scalars: Sequence[ScalarParameter] = (),
    ) -> None:
        self.law = OffspringLaw(**dict(law))
        self.n_model_scalars = len(scalars)
        self.space = ParameterSpace(
            self.law.rows,
            tuple(scalars) + self.law.scalars,  # law scalars (clone.kappa) last
            fixed={**self.law.fixed, **flatten(fixed)},
            concentration=self.law.concentration,
        )
        self.max_cells = int(max_cells)
        self.max_events = int(max_events)

    # ------------------------------------------------------------------ parameters
    @property
    def types(self) -> tuple[str, ...]:
        """Cell-type names."""
        return self.law.types

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the physical parameter vector (free and fixed)."""
        return self.space.names

    @property
    def theta_names(self) -> tuple[str, ...]:
        """Names of the inference coordinates ``theta`` (free parameters only)."""
        return self.space.theta_names

    @property
    def theta_dim(self) -> int:
        """Dimension of ``theta``."""
        return self.space.theta_dim

    def prior(self, device: str = "cpu") -> torch.distributions.Distribution:
        """Prior over ``theta``."""
        return self.space.prior(device)

    def decode(self, theta: np.ndarray) -> np.ndarray:
        """Map ``theta`` to physical parameters (fixed values filled in)."""
        return self.space.decode(theta)

    def encode(self, physical: np.ndarray) -> np.ndarray:
        """Map a physical vector to ``theta``."""
        return self.space.encode(physical)

    def physical_from_mapping(self, values: Mapping[str, float]) -> np.ndarray:
        """Physical vector from ``{name: value}``; see :meth:`ParameterSpace.complete`."""
        return self.space.complete(values)

    def validate(self, physical: np.ndarray) -> np.ndarray:
        """Validate a physical parameter vector."""
        return self.space.validate(physical)

    def scalar_values(self, physical: np.ndarray) -> np.ndarray:
        """Values of the simulator's own scalars (stored after the probability rows).

        Law scalars (``clone.kappa``) are excluded; see :meth:`clone_kappa`.
        """
        n = len(self.space.names) - len(self.space.scalars)
        return np.asarray(physical, dtype=float)[n : n + self.n_model_scalars]

    @property
    def heterogeneous(self) -> bool:
        """Whether clones draw their own outcome rows (``law.clone_heterogeneity``)."""
        return self.law.heterogeneous

    def clone_kappa(self, physical: np.ndarray) -> float | None:
        """Clone-heterogeneity concentration ``kappa``; ``None`` for a homogeneous law."""
        if not self.heterogeneous:
            return None
        return float(np.asarray(physical, dtype=float)[self.space.names.index("clone.kappa")])

    def clone_matrices(
        self, physical: np.ndarray, P: np.ndarray, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Per-clone outcome matrices ``(n, K, O)`` drawn around ``P`` (heterogeneous law)."""
        return self.law.clone_matrices(self.space, P, self.clone_kappa(physical), n, rng)

    # ------------------------------------------------------------------ simulation
    def validate_horizon(self, horizon: float) -> float:
        """Check that an endpoint is valid for this model's time axis."""
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("Horizons must be finite and positive")
        return float(horizon)

    @abstractmethod
    def simulate(
        self, physical: np.ndarray, horizon: float, n: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Simulate ``n`` independent clones to ``horizon``.

        Returns:
            ``(n, n_types)`` int64 array of *true* living cell counts at the endpoint.
            Rows that hit a resource limit are filled with :data:`INVALID`.
        """

    @property
    def supports_detected_sampling(self) -> bool:
        """Whether :meth:`simulate_detected` is implemented."""
        return False

    def simulate_detected(
        self,
        physical: np.ndarray,
        horizon: float,
        n: int,
        capture: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample *captured* true-type counts of ``n`` clones given >= 1 captured cell.

        Optional exact alternative to forward simulation plus rejection. Rows are
        :data:`INVALID` when detection is impossible or a limit was exceeded.
        """
        raise NotImplementedError

    def expected_counts(self, physical: np.ndarray, horizon: float) -> np.ndarray:
        """Return expected living counts per type at ``horizon`` (unconditional)."""
        raise NotImplementedError


# ---------------------------------------------------------------------- numba helpers
@njit(cache=True, nogil=True)
def categorical(weights, size, rng):
    """Draw an index in ``[0, size)`` proportional to ``weights[:size]``."""
    total = 0.0
    for i in range(size):
        total += weights[i]
    u = rng.random() * total
    acc = 0.0
    for i in range(size - 1):
        acc += weights[i]
        if u < acc:
            return i
    return size - 1


@njit(cache=True, nogil=True)
def draw_founders(founder, sizes, size_probs, rng, out):
    """Draw a founder count ``m``, fill ``out[:m]`` with i.i.d. types, return ``m``."""
    m = sizes[categorical(size_probs, len(sizes), rng)]
    for i in range(m):
        out[i] = categorical(founder, len(founder), rng)
    return m
