"""Measurement model applied to true endpoint counts.

The biological process (simulator) and the measurement process are kept separate so
the same model can be studied under ideal and noisy assays:

* **capture** - each living endpoint cell of type ``k`` is detected independently with
  probability ``capture[k]``;
* **confusion** - a detected type-``k`` cell is labelled ``j`` with probability
  ``confusion[k][j]`` (rows sum to one);
* **selection** - ``"detected"`` keeps only clones with at least one detected cell
  (what an experiment that only sees surviving clones measures); ``"all"`` keeps every
  seeded clone, including empty ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class ObservationModel:
    """Known capture, type-confusion and clone-selection settings.

    Args:
        capture: Per-type detection probability.
        confusion: Row-stochastic ``(K, K)`` labelling matrix.
        selection: ``"detected"`` or ``"all"``.
        exact_detection: Use a simulator's exact detection-conditioned sampler when it
            has one (Galton-Watson). Otherwise forward simulation plus rejection is used.
        max_attempts_per_clone: Rejection budget; a dataset needing more seeded clones
            per recovered clone is marked invalid.
    """

    capture: Sequence[float] = (1.0, 1.0)
    confusion: Sequence[Sequence[float]] = ((1.0, 0.0), (0.0, 1.0))
    selection: str = "detected"
    exact_detection: bool = True
    max_attempts_per_clone: int = 1000

    def __post_init__(self) -> None:
        self.capture = np.asarray(self.capture, dtype=float)
        self.confusion = np.asarray(self.confusion, dtype=float)
        k = len(self.capture)
        if self.selection not in ("detected", "all"):
            raise ValueError("selection must be 'detected' or 'all'")
        if np.any((self.capture < 0) | (self.capture > 1)):
            raise ValueError("capture probabilities must lie in [0, 1]")
        if self.selection == "detected" and not np.any(self.capture > 0):
            raise ValueError("Detected-clone selection is impossible with zero capture")
        if (
            self.confusion.shape != (k, k)
            or np.any(self.confusion < 0)
            or not np.allclose(self.confusion.sum(1), 1.0)
        ):
            raise ValueError("confusion must be a row-stochastic K x K matrix")
        self._identity = np.array_equal(self.confusion, np.eye(k))

    @property
    def n_types(self) -> int:
        """Number of measured types."""
        return len(self.capture)

    def relabel(self, captured: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply type confusion to already-captured counts ``(n, K)``."""
        if self._identity:
            return captured.copy()
        out = np.zeros_like(captured)
        for k in range(self.n_types):
            out += rng.multinomial(captured[:, k], self.confusion[k])
        return out

    def measure(self, counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply capture then confusion to true counts ``(n, K)``."""
        captured = counts if np.all(self.capture == 1) else rng.binomial(counts, self.capture)
        return self.relabel(captured, rng)

    def keep(self, observed: np.ndarray) -> np.ndarray:
        """Boolean mask of clones retained by the selection rule."""
        if self.selection == "all":
            return np.ones(len(observed), dtype=bool)
        return observed.sum(axis=1) > 0
