"""Multi-type offspring law shared by all branching simulators.

A law is a *type graph*: for every cell type it lists the possible outcomes of one
cell event together with the offspring each outcome leaves behind. For example, the
default two-type law used throughout the project is::

    A: death -> {}      persist -> {A}    switch -> {B}
       asymmetric -> {A, B}   self_renew -> {A, A}   sym_diff -> {B, B}
    B: death -> {}      persist -> {B}    switch -> {A}

Each type's outcome probabilities form one simplex row. Removing an outcome from the
config fixes its probability to zero, which is how sub-models are specified.

Inference coordinates
---------------------
Each row ``p ~ Dirichlet(alpha)`` is parameterised by ``len(alpha) - 1`` stick-breaking
fractions ``v_i ~ Beta(alpha_i, sum_{j>i} alpha_j)``, independent across ``i``. This is
an exact reparameterisation of the Dirichlet, gives a product prior that every sbi
method can use (``support`` is a box), and maps back via
``p_i = v_i * prod_{j<i} (1 - v_j)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class OffspringLaw:
    """Type graph plus Dirichlet prior over each type's outcome probabilities.

    Args:
        types: Ordered cell-type names, e.g. ``("A", "B")``.
        outcomes: ``{type: {outcome: offspring}}`` where ``offspring`` is a list of
            daughter counts per type (same order as ``types``). ``[0, 0]`` is death,
            ``[1, 0]`` an A daughter (persistence or switch to A), ``[2, 0]`` AA, etc.
        concentration: Optional ``{type: [alpha_1, ...]}`` Dirichlet concentrations,
            one per outcome in config order. Defaults to all ones (uniform simplex).
            Larger values on one outcome make the prior informative about it.
    """

    types: Sequence[str]
    outcomes: Mapping[str, Mapping[str, Sequence[int]]]
    concentration: Mapping[str, Sequence[float]] | None = None
    # Derived arrays (filled in __post_init__).
    offspring: np.ndarray = field(init=False, repr=False)
    n_outcomes: np.ndarray = field(init=False, repr=False)
    children: np.ndarray = field(init=False, repr=False)
    n_children: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.types = tuple(self.types)
        k = len(self.types)
        if k < 1 or set(self.outcomes) != set(self.types):
            raise ValueError(f"outcomes must define every type exactly once: {self.types}")
        self.outcomes = {t: dict(self.outcomes[t]) for t in self.types}
        if any(len(self.outcomes[t]) < 1 for t in self.types):
            raise ValueError("Every type needs at least one outcome")
        width = max(len(v) for v in self.outcomes.values())
        self.offspring = np.zeros((k, width, k), dtype=np.int64)
        self.n_outcomes = np.array([len(self.outcomes[t]) for t in self.types], dtype=np.int64)
        for i, t in enumerate(self.types):
            for o, (name, daughters) in enumerate(self.outcomes[t].items()):
                d = np.asarray(daughters, dtype=np.int64)
                if d.shape != (k,) or np.any(d < 0):
                    raise ValueError(f"Offspring of {t}.{name} must be {k} nonnegative ints")
                self.offspring[i, o] = d
        # Expanded daughter-type lists, used by event-driven kernels.
        max_children = max(1, int(self.offspring.sum(-1).max()))
        self.children = np.full((k, width, max_children), -1, dtype=np.int64)
        self.n_children = self.offspring.sum(-1).astype(np.int64)
        for i in range(k):
            for o in range(width):
                kids = np.repeat(np.arange(k), self.offspring[i, o])
                self.children[i, o, : len(kids)] = kids
        alpha = self.concentration or {}
        self.concentration = {
            t: tuple(float(a) for a in alpha.get(t, [1.0] * len(self.outcomes[t])))
            for t in self.types
        }
        for t in self.types:
            a = np.asarray(self.concentration[t])
            if a.shape != (len(self.outcomes[t]),) or np.any(a <= 0):
                raise ValueError(f"concentration[{t}] needs one positive value per outcome")

    # ------------------------------------------------------------------ names / sizes
    @property
    def probability_names(self) -> tuple[str, ...]:
        """Physical parameter names, ``"<type>.<outcome>"`` in config order."""
        return tuple(f"{t}.{o}" for t in self.types for o in self.outcomes[t])

    @property
    def stick_names(self) -> tuple[str, ...]:
        """Inference-coordinate names (one stick per free probability)."""
        return tuple(f"{t}.stick_{o}" for t in self.types for o in list(self.outcomes[t])[:-1])

    @property
    def n_probabilities(self) -> int:
        """Total number of physical outcome probabilities."""
        return int(self.n_outcomes.sum())

    @property
    def n_sticks(self) -> int:
        """Number of free probability coordinates (``n_probabilities - n_types``)."""
        return int((self.n_outcomes - 1).sum())

    # ------------------------------------------------------------------ transforms
    def sticks_to_probabilities(self, sticks: np.ndarray) -> np.ndarray:
        """Map stick fractions ``(..., n_sticks)`` to probabilities ``(..., n_probabilities)``."""
        v = np.asarray(sticks, dtype=float)
        rows, start = [], 0
        for n in self.n_outcomes:
            s = v[..., start : start + n - 1]
            start += n - 1
            tails = np.concatenate(
                [np.ones((*s.shape[:-1], 1)), np.cumprod(1.0 - s, axis=-1)], axis=-1
            )
            rows.append(np.concatenate([s * tails[..., :-1], tails[..., -1:]], axis=-1))
        return np.concatenate(rows, axis=-1)

    def probabilities_to_sticks(self, probabilities: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`sticks_to_probabilities` for a single parameter vector."""
        p = self.validate_probabilities(probabilities)
        sticks, start = [], 0
        for n in self.n_outcomes:
            row = p[start : start + n]
            start += n
            remaining = np.cumsum(row[::-1])[::-1]
            sticks.extend(row[:-1] / np.maximum(remaining[:-1], 1e-300))
        return np.asarray(sticks, dtype=float)

    def probability_matrix(self, probabilities: np.ndarray) -> np.ndarray:
        """Return a ``(n_types, max_outcomes)`` matrix, zero-padded, for the kernels."""
        p = np.asarray(probabilities, dtype=float)
        matrix = np.zeros(self.offspring.shape[:2])
        start = 0
        for i, n in enumerate(self.n_outcomes):
            matrix[i, :n] = p[start : start + n]
            start += n
        return matrix

    def validate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Check shape, nonnegativity and that every type's row sums to one."""
        p = np.asarray(probabilities, dtype=float)
        if p.shape != (self.n_probabilities,) or not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError(f"Expected {self.n_probabilities} finite nonnegative probabilities")
        start = 0
        for t, n in zip(self.types, self.n_outcomes, strict=True):
            if abs(p[start : start + n].sum() - 1.0) > 1e-6:
                raise ValueError(f"Outcome probabilities of type {t} must sum to one")
            start += n
        return p

    def mean_matrix(self, probabilities: np.ndarray) -> np.ndarray:
        """Mean offspring matrix ``M[k, j] = E[# type-j daughters of one type-k event]``."""
        matrix = self.probability_matrix(self.validate_probabilities(probabilities))
        return np.einsum("ko,koj->kj", matrix, self.offspring)

    # ------------------------------------------------------------------ prior
    def stick_prior_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Beta ``(a, b)`` parameters of the stick prior equivalent to each Dirichlet row."""
        a, b = [], []
        for t in self.types:
            alpha = np.asarray(self.concentration[t])
            tail = np.cumsum(alpha[::-1])[::-1]
            a.extend(alpha[:-1])
            b.extend(tail[1:])
        return np.asarray(a, dtype=float), np.asarray(b, dtype=float)

    def stick_prior(self, device: str = "cpu") -> torch.distributions.Distribution | None:
        """Torch Beta distribution (batch shape ``n_sticks``) or ``None`` if no sticks."""
        if self.n_sticks == 0:
            return None
        a, b = self.stick_prior_parameters()
        return torch.distributions.Beta(
            torch.as_tensor(a, dtype=torch.float32, device=device),
            torch.as_tensor(b, dtype=torch.float32, device=device),
        )
