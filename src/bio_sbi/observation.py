"""Measurement model applied to true endpoint counts.

The biological process (simulator) and the assay are separate so one model can be
studied under ideal and noisy measurements, and so data generated with *more or fewer
true types* than the inference model can be read through the same set of markers:

* **capture** - each living endpoint cell of true type ``k`` is detected
  independently with probability ``capture[k]``;
* **confusion** - a detected type-``k`` cell receives measured label ``j`` with
  probability ``confusion[k][j]``. Rows are *true types*, columns are *labels*, so the
  matrix may be rectangular (e.g. three true types read by two markers);
* **selection** - ``"detected"`` keeps only clones with at least one detected cell;
  ``"all"`` keeps every seeded clone, including empty ones.

Config forms (``types`` are the simulator's, ``labels`` default to the types)::

    capture: null | 0.9 | [0.9, 0.8] | {A: 0.9, B: 0.8}
    confusion: null                      # identity by name (each type -> same label)
             | 0.05                      # symmetric error, spread over other labels
             | [[0.95, 0.05], ...]       # matrix, types x labels
             | {A: {A: 1}, B: {B: 1}, C: {B: 1}}   # C is read as B
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def _per_type(value, types: Sequence[str], name: str) -> np.ndarray:
    if value is None:
        return np.ones(len(types))
    if isinstance(value, Mapping):
        missing = set(types) - set(value)
        if missing:
            raise ValueError(f"{name} is missing types {sorted(missing)}")
        return np.array([float(value[t]) for t in types])
    arr = np.asarray(value, dtype=float)
    return np.full(len(types), float(arr)) if arr.ndim == 0 else arr


def _confusion(value, types: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    k, n = len(types), len(labels)
    if value is None or np.isscalar(value):
        error = 0.0 if value is None else float(value)
        missing = set(types) - set(labels)
        if missing:
            raise ValueError(
                f"Types {sorted(missing)} have no same-named label; give an explicit confusion"
            )
        matrix = np.zeros((k, n))
        for i, t in enumerate(types):
            j = labels.index(t)
            matrix[i, :] = error / max(n - 1, 1)
            matrix[i, j] = 1.0 - error if n > 1 else 1.0
        return matrix
    if isinstance(value, Mapping):
        matrix = np.zeros((k, n))
        for i, t in enumerate(types):
            row = value.get(t)
            if row is None:
                raise ValueError(f"confusion has no row for type {t}")
            for label, p in row.items():
                matrix[i, labels.index(label)] = float(p)
        return matrix
    return np.asarray(value, dtype=float)


@dataclass
class ObservationModel:
    """Known capture, type-confusion and clone-selection settings.

    Args:
        types: True cell types of the simulator this assay observes (set by the
            pipeline from the simulator).
        labels: Measured label names (default: the types). Data and inference
            observation models must produce the same labels.
        capture: Per-type detection probability (see module docstring for forms).
        confusion: True-type -> label probabilities (see module docstring).
        selection: ``"detected"`` or ``"all"``.
        exact_detection: Use a simulator's exact detection-conditioned sampler when it
            has one (Galton-Watson). Otherwise forward simulation plus rejection is used.
        max_attempts_per_clone: Rejection budget; a dataset needing more seeded clones
            per recovered clone is marked invalid.
    """

    types: Sequence[str]
    labels: Sequence[str] | None = None
    capture: object = None
    confusion: object = None
    selection: str = "detected"
    exact_detection: bool = True
    max_attempts_per_clone: int = 1000

    def __post_init__(self) -> None:
        self.types = tuple(self.types)
        self.labels = tuple(self.labels) if self.labels is not None else self.types
        self.capture = _per_type(self.capture, self.types, "capture")
        self.confusion = _confusion(self.confusion, self.types, list(self.labels))
        if self.selection not in ("detected", "all"):
            raise ValueError("selection must be 'detected' or 'all'")
        if self.capture.shape != (len(self.types),):
            raise ValueError("capture needs one value per true type")
        if np.any((self.capture < 0) | (self.capture > 1)):
            raise ValueError("capture probabilities must lie in [0, 1]")
        if self.selection == "detected" and not np.any(self.capture > 0):
            raise ValueError("Detected-clone selection is impossible with zero capture")
        if (
            self.confusion.shape != (len(self.types), len(self.labels))
            or np.any(self.confusion < 0)
            or not np.allclose(self.confusion.sum(1), 1.0)
        ):
            raise ValueError("confusion must be row-stochastic with shape (types, labels)")
        self._identity = self.types == self.labels and np.array_equal(
            self.confusion, np.eye(len(self.types))
        )

    @property
    def n_labels(self) -> int:
        """Number of measured labels (columns of the count matrices)."""
        return len(self.labels)

    def relabel(self, captured: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Map captured true-type counts ``(n, K)`` to label counts ``(n, L)``."""
        if self._identity:
            return captured.copy()
        out = np.zeros((len(captured), self.n_labels), dtype=np.int64)
        for k in range(len(self.types)):
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
