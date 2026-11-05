"""Abstract simulator contract and shared kernel helpers.

A simulator owns three things:

1. a **parameterisation**: physical parameters (outcome probabilities, lifetimes, ...)
   and the unconstrained-ish *inference coordinates* ``theta`` used by sbi, with
   :meth:`BranchingSimulator.decode` / :meth:`BranchingSimulator.encode` between them;
2. a **prior** over ``theta`` as a torch distribution (:meth:`BranchingSimulator.prior`);
3. a **forward model** returning true endpoint cell counts for ``n`` independent clones
   (:meth:`BranchingSimulator.simulate`), optionally with an exact sampler conditioned
   on detection (:meth:`BranchingSimulator.simulate_detected`).

To add a new model, subclass :class:`BranchingSimulator`, implement ``simulate`` (and
optionally ``simulate_detected``/``expected_counts``), and add a YAML file to
``conf/simulator/`` whose ``_target_`` points at the class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numba import njit
from sbi.utils import MultipleIndependent

from .law import OffspringLaw

#: Marker written into a clone's count row when a resource limit was exceeded.
INVALID = -1


@dataclass(frozen=True)
class ScalarParameter:
    """A non-probability parameter with a (log-)uniform prior.

    Args:
        name: Physical name, e.g. ``"A.mean_lifetime"``.
        low: Lower prior bound (physical units).
        high: Upper prior bound (physical units).
        log: If true, ``theta = log(value)`` is uniform on ``[log low, log high]``.
    """

    name: str
    low: float
    high: float
    log: bool = True

    @property
    def theta_name(self) -> str:
        """Name of the inference coordinate."""
        if not self.log:
            return self.name
        head, dot, tail = self.name.rpartition(".")
        return f"{head}{dot}log_{tail}"

    @property
    def theta_bounds(self) -> tuple[float, float]:
        """Prior bounds in inference coordinates."""
        return (np.log(self.low), np.log(self.high)) if self.log else (self.low, self.high)


class BranchingSimulator(ABC):
    """Base class for multi-type branching simulators.

    Args:
        types: Cell-type names.
        outcomes: Type graph, see :class:`~bio_sbi.simulators.law.OffspringLaw`.
        founder: Probability that a clone's founder cell is of each type.
        concentration: Optional Dirichlet concentrations per type.
        max_cells: Per-clone population cap. Exceeding it marks the clone invalid
            (the whole dataset is then dropped from training, never truncated).
        max_events: Per-clone cap on the number of simulated cell events.
    """

    #: Human-readable time unit of ``horizon`` (overridden by subclasses).
    time_unit: str = "generations"

    def __init__(
        self,
        types: Sequence[str],
        outcomes: Mapping[str, Mapping[str, Sequence[int]]],
        founder: Sequence[float],
        concentration: Mapping[str, Sequence[float]] | None = None,
        max_cells: int = 100_000,
        max_events: int = 5_000_000,
    ) -> None:
        self.law = OffspringLaw(types, outcomes, concentration)
        f = np.asarray(founder, dtype=float)
        if f.shape != (len(self.law.types),) or np.any(f < 0) or not np.isclose(f.sum(), 1):
            raise ValueError("founder must be one probability per type, summing to one")
        self.founder = f
        self.max_cells = int(max_cells)
        self.max_events = int(max_events)

    # ------------------------------------------------------------------ parameters
    @property
    def scalar_parameters(self) -> tuple[ScalarParameter, ...]:
        """Non-probability parameters (none by default)."""
        return ()

    @property
    def types(self) -> tuple[str, ...]:
        """Cell-type names."""
        return self.law.types

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the physical parameter vector returned by :meth:`decode`."""
        return self.law.probability_names + tuple(s.name for s in self.scalar_parameters)

    @property
    def theta_names(self) -> tuple[str, ...]:
        """Names of the inference coordinates ``theta``."""
        return self.law.stick_names + tuple(s.theta_name for s in self.scalar_parameters)

    @property
    def theta_dim(self) -> int:
        """Dimension of ``theta``."""
        return len(self.theta_names)

    def prior(self, device: str = "cpu") -> torch.distributions.Distribution:
        """Prior over ``theta``: stick Betas times (log-)uniform scalars."""
        parts = []
        sticks = self.law.stick_prior(device)
        if sticks is not None:
            parts.append(torch.distributions.Independent(sticks, 1))
        if self.scalar_parameters:
            low, high = zip(*(s.theta_bounds for s in self.scalar_parameters), strict=True)
            uniform = torch.distributions.Uniform(
                torch.tensor(low, dtype=torch.float32, device=device),
                torch.tensor(high, dtype=torch.float32, device=device),
            )
            parts.append(torch.distributions.Independent(uniform, 1))
        return parts[0] if len(parts) == 1 else MultipleIndependent(parts, device=device)

    def decode(self, theta: np.ndarray) -> np.ndarray:
        """Map inference coordinates ``(..., theta_dim)`` to physical parameters."""
        z = np.asarray(theta, dtype=float)
        s = self.law.n_sticks
        probabilities = self.law.sticks_to_probabilities(z[..., :s])
        scalars = [
            np.exp(z[..., s + i]) if p.log else z[..., s + i]
            for i, p in enumerate(self.scalar_parameters)
        ]
        if not scalars:
            return probabilities
        return np.concatenate([probabilities, np.stack(scalars, axis=-1)], axis=-1)

    def encode(self, physical: np.ndarray) -> np.ndarray:
        """Map one physical parameter vector to inference coordinates."""
        p = np.asarray(physical, dtype=float)
        n = self.law.n_probabilities
        sticks = self.law.probabilities_to_sticks(p[:n])
        scalars = [
            np.log(v) if s.log else v for v, s in zip(p[n:], self.scalar_parameters, strict=True)
        ]
        return np.r_[sticks, scalars].astype(np.float32)

    def physical_from_mapping(self, values: Mapping[str, float]) -> np.ndarray:
        """Build a physical vector from ``{name: value}``; omitted outcomes default to 0.

        Missing *scalar* parameters raise, as there is no sensible default.
        """
        unknown = set(values) - set(self.parameter_names)
        if unknown:
            raise ValueError(f"Unknown parameters {sorted(unknown)}; known: {self.parameter_names}")
        n = self.law.n_probabilities
        vector = np.array([float(values.get(k, 0.0)) for k in self.parameter_names[:n]])
        for s in self.scalar_parameters:
            if s.name not in values:
                raise ValueError(f"Parameter {s.name} must be given explicitly")
        vector = np.r_[vector, [float(values[s.name]) for s in self.scalar_parameters]]
        self.validate(vector)
        return vector

    def validate(self, physical: np.ndarray) -> np.ndarray:
        """Validate a physical parameter vector and return it as float64."""
        p = np.asarray(physical, dtype=float)
        n = self.law.n_probabilities
        if p.shape != (len(self.parameter_names),):
            raise ValueError(f"Expected {len(self.parameter_names)} physical parameters")
        self.law.validate_probabilities(p[:n])
        if np.any(p[n:] <= 0) or not np.all(np.isfinite(p[n:])):
            raise ValueError("Scalar parameters must be finite and positive")
        return p

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
        """Sample *captured* counts of ``n`` clones conditioned on >= 1 captured cell.

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
