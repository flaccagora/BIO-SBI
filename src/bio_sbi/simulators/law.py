"""Type graph, founder distribution and the parameter space built from them.

Offspring law
-------------
For every cell type the law lists the possible outcomes of one cell event and the
daughters each outcome leaves. With ``outcomes: all`` (the default) every outcome with
at most ``max_daughters`` daughters is enabled. For two types ``A, B`` this gives,
per type::

    death -> {}      persist -> {self}      to_<other> -> {other}
    div_AA -> {A, A}   div_AB -> {A, B}   div_BB -> {B, B}

Outcomes can also be listed explicitly as ``{type: {name: [n_A, n_B, ...]}}``.

Founders
--------
A clone is seeded by ``N`` founder cells, ``P(N = n) = founder_size.<n>`` over
``founder_sizes`` (default ``[1]``), and each founder independently has type ``k`` with
probability ``founder.<k>``.

Parameter space
---------------
All probabilities live in *simplex rows*: one row per type (its outcomes), the
``founder`` row (types) and the ``founder_size`` row (sizes). Scalar parameters (e.g.
Bellman-Harris lifetimes) are appended by simulators. Any parameter can be **fixed**
to a value, in which case it is not inferred and data are generated with that value.
The free entries of a row share the leftover mass ``1 - sum(fixed)`` with a
``Dirichlet(concentration)`` prior, parameterised by independent stick-breaking Betas
(an exact reparameterisation with box support, usable by every sbi method).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
import torch
from sbi.utils import MultipleIndependent

_TOL = 1e-6


def flatten(mapping: Mapping | None, prefix: str = "") -> dict[str, float]:
    """Flatten ``{A: {persist: 0.1}}`` (or already-flat ``{"A.persist": 0.1}``) to names.

    Nested form is what Hydra command-line overrides produce
    (``+law.fixed.A.persist=0``); both forms may be mixed.
    """
    out: dict[str, float] = {}
    for key, value in (mapping or {}).items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, name + "."))
        else:
            out[name] = float(value)
    return out


# ---------------------------------------------------------------------- outcome tables
def all_outcomes(types: Sequence[str], max_daughters: int = 2) -> dict[str, dict[str, list]]:
    """Enumerate every outcome with ``<= max_daughters`` daughters for every type.

    Names: ``death``, ``persist`` (one daughter of the same type), ``to_<T>`` (one
    daughter of another type) and ``div_<T1><T2>...`` (two or more daughters).
    """
    k = len(types)
    table: dict[str, dict[str, list]] = {}
    for parent in types:
        row: dict[str, list] = {"death": [0] * k}
        for child in types:
            vec = [int(t == child) for t in types]
            row["persist" if child == parent else f"to_{child}"] = vec
        for size in range(2, max_daughters + 1):
            for combo in combinations_with_replacement(range(k), size):
                vec = [combo.count(i) for i in range(k)]
                row["div_" + "".join(types[i] for i in combo)] = vec
        table[parent] = row
    return table


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


@dataclass
class _Row:
    """One simplex row after resolving fixed entries."""

    prefix: str
    entries: tuple[str, ...]
    fixed: dict[int, float]  # entry index -> value (includes forced zeros)
    free: tuple[int, ...]  # entry indices sharing the leftover mass
    alpha: np.ndarray  # concentration of the free entries
    mass: float  # 1 - sum(fixed)

    @property
    def n_sticks(self) -> int:
        return max(len(self.free) - 1, 0)


class ParameterSpace:
    """Physical parameters, their fixed values, and inference coordinates ``theta``.

    Args:
        rows: ``[(prefix, entries), ...]`` simplex rows; parameter ``prefix.entry``.
        scalars: Scalar parameters appended after the rows.
        fixed: ``{name: value}`` (flat or nested, see :func:`flatten`) for parameters
            that are not inferred.
        concentration: ``{name: alpha}`` (flat or nested) Dirichlet concentration of free entries
            (default 1, i.e. uniform over the free sub-simplex).
    """

    def __init__(
        self,
        rows: Sequence[tuple[str, Sequence[str]]],
        scalars: Sequence[ScalarParameter] = (),
        fixed: Mapping[str, float] | None = None,
        concentration: Mapping[str, float] | None = None,
    ) -> None:
        fixed = flatten(fixed)
        alpha = flatten(concentration)
        self.scalars = tuple(scalars)
        self.names = tuple(f"{p}.{e}" for p, entries in rows for e in entries) + tuple(
            s.name for s in self.scalars
        )
        unknown = (set(fixed) | set(alpha)) - set(self.names)
        if unknown:
            raise ValueError(f"Unknown parameters {sorted(unknown)}; known: {self.names}")
        self.rows: list[_Row] = []
        for prefix, entries in rows:
            entries = tuple(str(e) for e in entries)
            names = [f"{prefix}.{e}" for e in entries]
            fx = {i: fixed[n] for i, n in enumerate(names) if n in fixed}
            if any(v < 0 or v > 1 for v in fx.values()):
                raise ValueError(f"Fixed probabilities of row {prefix} must lie in [0, 1]")
            mass = 1.0 - sum(fx.values())
            free = [i for i in range(len(entries)) if i not in fx]
            if mass < -_TOL or (not free and abs(mass) > _TOL):
                raise ValueError(
                    f"Fixed probabilities of row {prefix} must sum to <= 1 (= 1 if all fixed)"
                )
            if mass <= _TOL:  # nothing left: remaining entries are zero
                fx.update({i: 0.0 for i in free})
                free, mass = [], 0.0
            a = np.array([alpha.get(names[i], 1.0) for i in free])
            if np.any(a <= 0):
                raise ValueError("Concentrations must be positive")
            self.rows.append(_Row(prefix, entries, fx, tuple(free), a, mass))
        for s in self.scalars:
            if s.name in alpha:
                raise ValueError(f"{s.name} is a scalar; concentration applies to probabilities")
        self.fixed_scalars = {s.name: fixed[s.name] for s in self.scalars if s.name in fixed}
        self.free_scalars = tuple(s for s in self.scalars if s.name not in fixed)

    # ------------------------------------------------------------------ names / sizes
    @property
    def theta_names(self) -> tuple[str, ...]:
        """Names of the free inference coordinates."""
        names = [f"{r.prefix}.stick_{r.entries[i]}" for r in self.rows for i in r.free[:-1]]
        return tuple(names) + tuple(s.theta_name for s in self.free_scalars)

    @property
    def theta_dim(self) -> int:
        """Dimension of ``theta``."""
        return len(self.theta_names)

    @property
    def free_names(self) -> tuple[str, ...]:
        """Physical parameters that are inferred (not fixed or determined by fixed ones)."""
        names = [f"{r.prefix}.{r.entries[i]}" for r in self.rows if r.n_sticks for i in r.free]
        return tuple(names) + tuple(s.name for s in self.free_scalars)

    def row_slice(self, prefix: str) -> slice:
        """Slice of the physical vector holding row ``prefix``."""
        start = 0
        for r in self.rows:
            if r.prefix == prefix:
                return slice(start, start + len(r.entries))
            start += len(r.entries)
        raise KeyError(prefix)

    # ------------------------------------------------------------------ prior
    def prior(self, device: str = "cpu") -> torch.distributions.Distribution:
        """Product prior over ``theta``: stick Betas times (log-)uniform scalars."""
        a, b = [], []
        for r in self.rows:
            tail = np.cumsum(r.alpha[::-1])[::-1]
            a.extend(r.alpha[:-1])
            b.extend(tail[1:])
        parts = []
        if a:
            beta = torch.distributions.Beta(
                torch.tensor(a, dtype=torch.float32, device=device),
                torch.tensor(b, dtype=torch.float32, device=device),
            )
            parts.append(torch.distributions.Independent(beta, 1))
        if self.free_scalars:
            low, high = zip(*(s.theta_bounds for s in self.free_scalars), strict=True)
            uniform = torch.distributions.Uniform(
                torch.tensor(low, dtype=torch.float32, device=device),
                torch.tensor(high, dtype=torch.float32, device=device),
            )
            parts.append(torch.distributions.Independent(uniform, 1))
        if not parts:
            raise ValueError("Every parameter is fixed: nothing to infer")
        return parts[0] if len(parts) == 1 else MultipleIndependent(parts, device=device)

    # ------------------------------------------------------------------ transforms
    def decode(self, theta: np.ndarray) -> np.ndarray:
        """Map ``theta`` ``(..., theta_dim)`` to physical parameters ``(..., n_names)``."""
        z = np.asarray(theta, dtype=float)
        batch = z.shape[:-1]
        out, k = [], 0
        for r in self.rows:
            row = np.zeros((*batch, len(r.entries)))
            for i, v in r.fixed.items():
                row[..., i] = v
            if r.free:
                s = z[..., k : k + r.n_sticks]
                k += r.n_sticks
                tails = np.concatenate([np.ones((*batch, 1)), np.cumprod(1 - s, axis=-1)], -1)
                p = np.concatenate([s * tails[..., :-1], tails[..., -1:]], -1) * r.mass
                row[..., list(r.free)] = p
            out.append(row)
        for s in self.scalars:
            if s.name in self.fixed_scalars:
                out.append(np.full((*batch, 1), self.fixed_scalars[s.name]))
            else:
                v = z[..., k : k + 1]
                k += 1
                out.append(np.exp(v) if s.log else v)
        return np.concatenate(out, axis=-1)

    def encode(self, physical: np.ndarray) -> np.ndarray:
        """Map one physical vector to ``theta`` (must agree with the fixed values)."""
        p = self.validate(physical)
        theta, start = [], 0
        for r in self.rows:
            row = p[start : start + len(r.entries)]
            start += len(r.entries)
            if r.free:
                free = row[list(r.free)] / r.mass
                remaining = np.cumsum(free[::-1])[::-1]
                theta.extend(free[:-1] / np.maximum(remaining[:-1], 1e-300))
        for s, v in zip(self.scalars, p[start:], strict=True):
            if s.name not in self.fixed_scalars:
                theta.append(np.log(v) if s.log else v)
        return np.asarray(theta, dtype=np.float32)

    def complete(self, values: Mapping[str, float]) -> np.ndarray:
        """Physical vector from ``{name: value}``; fixed values fill in, omitted entries are 0.

        A row's only free entry, when omitted, takes the leftover mass.

        Raises if a given value contradicts a fixed one, a free scalar is missing, or
        a row does not sum to one.
        """
        values = flatten(values)
        unknown = set(values) - set(self.names)
        if unknown:
            raise ValueError(f"Unknown parameters {sorted(unknown)}; known: {self.names}")
        vector = []
        for r in self.rows:
            for i, e in enumerate(r.entries):
                name = f"{r.prefix}.{e}"
                if i in r.fixed:
                    if name in values and abs(values[name] - r.fixed[i]) > _TOL:
                        raise ValueError(f"{name} is fixed to {r.fixed[i]}, got {values[name]}")
                    vector.append(r.fixed[i])
                elif len(r.free) == 1:  # determined by the fixed entries
                    vector.append(values.get(name, r.mass))
                else:
                    vector.append(values.get(name, 0.0))
        for s in self.scalars:
            if s.name in self.fixed_scalars:
                vector.append(self.fixed_scalars[s.name])
            elif s.name in values:
                vector.append(values[s.name])
            else:
                raise ValueError(f"Parameter {s.name} must be given explicitly")
        return self.validate(np.asarray(vector))

    def validate(self, physical: np.ndarray) -> np.ndarray:
        """Check shape, row sums, fixed values and scalar positivity."""
        p = np.asarray(physical, dtype=float)
        if p.shape != (len(self.names),) or not np.all(np.isfinite(p)):
            raise ValueError(f"Expected {len(self.names)} finite parameters")
        start = 0
        for r in self.rows:
            row = p[start : start + len(r.entries)]
            start += len(r.entries)
            if np.any(row < -_TOL) or abs(row.sum() - 1.0) > _TOL:
                raise ValueError(f"Row {r.prefix} must be nonnegative and sum to one")
            for i, v in r.fixed.items():
                if abs(row[i] - v) > _TOL:
                    raise ValueError(f"{r.prefix}.{r.entries[i]} is fixed to {v}")
        if np.any(p[start:] <= 0):
            raise ValueError("Scalar parameters must be positive")
        return p


# ---------------------------------------------------------------------- the law
class OffspringLaw:
    """Cell types, outcome table and founder distribution.

    Args:
        types: Ordered cell-type names.
        outcomes: ``"all"`` or ``{type: {outcome: daughters}}`` with ``daughters`` a
            list of counts per type.
        max_daughters: Largest division arity enumerated by ``outcomes: all``.
        founder_sizes: Possible numbers of founder cells per clone.
        fixed: ``{parameter: value}`` not inferred (probabilities of this law).
        concentration: ``{parameter: alpha}`` Dirichlet concentrations.
        **_: Other law-config keys (``name``, ``reference``) are accepted and ignored.
    """

    def __init__(
        self,
        types: Sequence[str],
        outcomes: str | Mapping[str, Mapping[str, Sequence[int]]] = "all",
        max_daughters: int = 2,
        founder_sizes: Sequence[int] = (1,),
        fixed: Mapping[str, float] | None = None,
        concentration: Mapping[str, float] | None = None,
        **_: object,
    ) -> None:
        self.types = tuple(str(t) for t in types)
        k = len(self.types)
        if k < 1 or len(set(self.types)) != k:
            raise ValueError("types must be unique and non-empty")
        if outcomes == "all":
            outcomes = all_outcomes(self.types, max_daughters)
        if not isinstance(outcomes, Mapping) or set(outcomes) != set(self.types):
            raise ValueError(f"outcomes must be 'all' or define every type: {self.types}")
        self.outcomes = {t: dict(outcomes[t]) for t in self.types}
        if any(len(v) < 1 for v in self.outcomes.values()):
            raise ValueError("Every type needs at least one outcome")
        self.founder_sizes = np.array(sorted({int(n) for n in founder_sizes}), dtype=np.int64)
        if len(self.founder_sizes) == 0 or self.founder_sizes[0] < 1:
            raise ValueError("founder_sizes must be positive integers")
        width = max(len(v) for v in self.outcomes.values())
        self.offspring = np.zeros((k, width, k), dtype=np.int64)
        self.n_outcomes = np.array([len(self.outcomes[t]) for t in self.types], dtype=np.int64)
        for i, t in enumerate(self.types):
            for o, (name, daughters) in enumerate(self.outcomes[t].items()):
                d = np.asarray(daughters, dtype=np.int64)
                if d.shape != (k,) or np.any(d < 0):
                    raise ValueError(f"Offspring of {t}.{name} must be {k} nonnegative ints")
                self.offspring[i, o] = d
        max_children = max(1, int(self.offspring.sum(-1).max()))
        self.children = np.full((k, width, max_children), -1, dtype=np.int64)
        self.n_children = self.offspring.sum(-1).astype(np.int64)
        for i in range(k):
            for o in range(width):
                kids = np.repeat(np.arange(k), self.offspring[i, o])
                self.children[i, o, : len(kids)] = kids
        self.fixed = flatten(fixed)
        self.concentration = flatten(concentration)

    @property
    def rows(self) -> list[tuple[str, tuple[str, ...]]]:
        """Simplex rows: one per type, then ``founder`` and ``founder_size``."""
        rows = [(t, tuple(self.outcomes[t])) for t in self.types]
        rows.append(("founder", self.types))
        rows.append(("founder_size", tuple(str(n) for n in self.founder_sizes)))
        return rows

    def unpack(self, space: ParameterSpace, physical: np.ndarray):
        """Split a physical vector into kernel arrays.

        Returns:
            ``(P, founder_types, founder_size_probs)`` where ``P`` is the zero-padded
            ``(n_types, max_outcomes)`` outcome-probability matrix.
        """
        P = np.zeros(self.offspring.shape[:2])
        for i, t in enumerate(self.types):
            P[i, : self.n_outcomes[i]] = np.clip(physical[space.row_slice(t)], 0, None)
        founder = np.clip(physical[space.row_slice("founder")], 0, None)
        sizes = np.clip(physical[space.row_slice("founder_size")], 0, None)
        return P, founder / founder.sum(), sizes / sizes.sum()

    def mean_matrix(self, P: np.ndarray) -> np.ndarray:
        """Mean offspring matrix ``M[k, j]`` from an outcome-probability matrix."""
        return np.einsum("ko,koj->kj", P, self.offspring)
