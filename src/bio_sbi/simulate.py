"""Seeded generation of datasets and their summaries.

:class:`DataGenerator` bundles a simulator, an observation model, a design and a
summary, and turns parameter vectors into training pairs ``(theta, x)``. Every dataset
uses its own keyed random stream (see :mod:`bio_sbi.rng`), so results do not depend on
batching or the number of worker threads.

Datasets whose simulation hit a resource limit, or whose clone recovery needed more
than the rejection budget, are returned as ``None`` (and as a NaN row of ``x``). They
are excluded from training: the fitted posterior is therefore conditioned on the
simulation limits, which should be set well above any plausible observed clone size.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .design import Design
from .observation import ObservationModel
from .rng import rng_for
from .simulators import INVALID, BranchingSimulator
from .summaries import Cohort, Summary


@dataclass
class DataGenerator:
    """Simulate observed datasets and summaries from parameters.

    Args:
        simulator: Biological model.
        observation: Measurement model.
        design: Endpoints and clones per endpoint.
        summary: Summary statistic applied to each dataset.
        seed: Root seed for all keyed streams.
        workers: Threads used for independent datasets (Numba kernels release the GIL).
    """

    simulator: BranchingSimulator
    observation: ObservationModel
    design: Design
    summary: Summary
    seed: int = 0
    workers: int = 1

    def __post_init__(self) -> None:
        if self.observation.n_types != len(self.simulator.types):
            raise ValueError("Observation model and simulator have different numbers of types")
        for h in self.design.horizons:
            self.simulator.validate_horizon(h)

    # ------------------------------------------------------------------ single dataset
    def cohort(
        self, physical: np.ndarray, horizon: float, n: int, rng: np.random.Generator
    ) -> np.ndarray | None:
        """Simulate one cohort of ``n`` *retained* clones; ``None`` if invalid."""
        sim, obs = self.simulator, self.observation
        if obs.selection == "detected" and obs.exact_detection and sim.supports_detected_sampling:
            captured = sim.simulate_detected(physical, horizon, n, obs.capture, rng)
            return None if np.any(captured == INVALID) else obs.relabel(captured, rng)
        kept, need, attempts, batch = [], n, 0, n
        while need > 0:
            if attempts + batch > n * obs.max_attempts_per_clone:
                return None
            true = sim.simulate(physical, horizon, batch, rng)
            attempts += batch
            if np.any(true == INVALID):
                return None
            measured = obs.measure(true, rng)
            accepted = measured[obs.keep(measured)][:need]
            kept.append(accepted)
            need -= len(accepted)
            # Grow the next batch according to the observed acceptance rate.
            rate = max(sum(len(k) for k in kept) / attempts, 1.0 / obs.max_attempts_per_clone)
            batch = int(min(max(need / rate * 1.2, 1), n * obs.max_attempts_per_clone))
        return np.concatenate(kept)

    def dataset(self, physical: np.ndarray, key: object) -> list[Cohort] | None:
        """Simulate all cohorts of the design for one parameter vector."""
        physical = self.simulator.validate(physical)
        cohorts = []
        for i, (horizon, n) in enumerate(self.design.cohorts):
            counts = self.cohort(physical, horizon, n, rng_for(self.seed, key, "cohort", i))
            if counts is None:
                return None
            cohorts.append(Cohort(horizon, counts))
        return cohorts

    def summarize(self, cohorts: list[Cohort] | None) -> np.ndarray:
        """Summary vector of a dataset; NaN-filled when the dataset is invalid."""
        if cohorts is None:
            return np.full(self.x_dim, np.nan, dtype=np.float32)
        return self.summary(cohorts)

    @property
    def x_dim(self) -> int:
        """Dimension of the summary vector (computed once from a dummy dataset)."""
        if not hasattr(self, "_x_dim"):
            k = len(self.simulator.types)
            dummy = [Cohort(h, np.ones((n, k), dtype=np.int64)) for h, n in self.design.cohorts]
            self._x_dim = len(self.summary(dummy))
        return self._x_dim

    # ------------------------------------------------------------------ batches
    def simulate_theta(
        self, theta: np.ndarray, stage: str, offset: int = 0
    ) -> tuple[np.ndarray, list[list[Cohort] | None]]:
        """Simulate one dataset per row of ``theta`` (inference coordinates).

        Args:
            theta: ``(N, theta_dim)`` parameters.
            stage: Label mixed into the random keys (e.g. ``"round-0"``, ``"sbc"``).
            offset: Index offset for the keys, to extend a batch reproducibly.

        Returns:
            ``x`` of shape ``(N, x_dim)`` (NaN rows for invalid datasets) and the raw
            cohorts per dataset.
        """
        physical = self.simulator.decode(np.asarray(theta, dtype=float))

        def run(i: int):
            return self.dataset(physical[i], (stage, offset + i))

        if self.workers > 1:
            with ThreadPoolExecutor(self.workers) as pool:
                data = list(pool.map(run, range(len(physical))))
        else:
            data = [run(i) for i in range(len(physical))]
        x = np.stack([self.summarize(d) for d in data]) if data else np.empty((0, self.x_dim))
        return x.astype(np.float32), data
