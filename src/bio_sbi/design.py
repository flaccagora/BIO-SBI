"""Experimental design: which endpoints are measured and on how many clones.

A *dataset* is a set of independent *cohorts*; each cohort contains clones measured at
one common endpoint (horizon) and every cohort shares the same generating parameters.
Different horizons use different clones (endpoint assays are destructive).

The design is the main lever for planning wet-lab experiments: sweep ``horizons`` and
``clones`` with Hydra multirun to see how posterior width depends on them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class Design:
    """Endpoints and clone budget.

    Args:
        horizons: Measurement endpoints in the simulator's time unit.
        clones: Clones per endpoint - one int for all endpoints or one per endpoint.
        name: Label used in run directories and result tables.
    """

    horizons: Sequence[float]
    clones: int | Sequence[int] = 32
    name: str = "custom"

    def __post_init__(self) -> None:
        self.horizons = tuple(float(h) for h in self.horizons)
        if not self.horizons or len(set(self.horizons)) != len(self.horizons):
            raise ValueError("Design needs at least one horizon and no duplicates")
        if any(not np.isfinite(h) or h <= 0 for h in self.horizons):
            raise ValueError("Horizons must be finite and positive")
        sizes = (
            (int(self.clones),) * len(self.horizons)
            if np.isscalar(self.clones)
            else tuple(int(n) for n in self.clones)
        )
        if len(sizes) != len(self.horizons) or any(n < 1 for n in sizes):
            raise ValueError("clones must be a positive int or one positive int per horizon")
        self.clones = sizes

    @property
    def cohorts(self) -> tuple[tuple[float, int], ...]:
        """``((horizon, n_clones), ...)`` in configuration order."""
        return tuple(zip(self.horizons, self.clones, strict=True))

    @property
    def total_clones(self) -> int:
        """Total number of measured clones in one dataset."""
        return int(sum(self.clones))
