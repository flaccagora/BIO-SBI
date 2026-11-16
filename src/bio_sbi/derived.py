"""Derived population quantities per posterior draw (two types), comparable across models.

For a draw of the physical parameters of a two-type model (rows A, B; ``M[i, j]`` = mean
number of type-``j`` children of a type-``i`` cell at the end of its life, *persist* = one
child of its own type):

* ``A_self_renewal`` = ``A.div_AA``;
* ``growth_rate_per_h`` = asymptotic (Malthusian) exponential growth rate of the clone per
  hour, ``doubling_time_h = log 2 / growth_rate``;
* ``stable_B_to_A`` = ratio of living B to A cells in the stable regime
  (``stable_B_fraction = r / (1 + r)``).

Bellman-Harris (Gamma lifetimes, shape ``m_j``, mean ``tau_j``): root ``alpha`` of
``rho(diag(L(alpha)) M) = 1`` with ``L_j`` the lifetime Laplace transforms, and
``r = L_A M_AB (1 - L_B) / ((1 - L_B M_BB) (1 - L_A))``
(:mod:`bio_sbi.inference.bh_reparam`). Galton-Watson with ``tau`` hours per generation:
``alpha = log rho(M) / tau``, ``r = M_AB / (rho - M_BB)`` (left Perron vector: the
composition of ``Z_n = Z_0 M^n``); without ``hours_per_generation`` the rate is per
generation. The BH formulas tend to the GW ones when both lifetimes are ``tau`` and the
shapes grow (synchronous divisions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .inference.bh_reparam import _rho2, malthusian, stable_ratio

COLUMNS = ["A_self_renewal", "growth_rate_per_h", "doubling_time_h", "stable_B_to_A",
           "stable_B_fraction"]  # fmt: skip


def mean_matrices(sim, physical: np.ndarray) -> np.ndarray:
    """Mean offspring matrices ``(n, 2, 2)`` of the rows of ``physical``."""
    phys = np.atleast_2d(np.asarray(physical, float))
    law = sim.law
    off = np.asarray(law.offspring, float)
    P = np.zeros((len(phys),) + off.shape[:2])
    for i, t in enumerate(law.types):
        sl = sim.space.row_slice(t)
        P[:, i, : int(law.n_outcomes[i])] = np.clip(phys[:, sl], 0, None)
    return np.einsum("nko,koj->nkj", P, off)


def derived_quantities(sim, physical: np.ndarray) -> pd.DataFrame:
    """Derived quantities (module docstring) of every row of ``physical`` (``sim`` order)."""
    from .simulators import BellmanHarris, GaltonWatson

    phys = np.atleast_2d(np.asarray(physical, float))
    if len(sim.types) != 2:
        raise ValueError("derived quantities need two cell types")
    names = list(sim.parameter_names)
    M = mean_matrices(sim, phys)
    if isinstance(sim, BellmanHarris):
        cols = [names.index(f"{t}.mean_lifetime") for t in sim.types]
        tau = phys[:, cols]
        alpha = malthusian(M, tau, np.asarray(sim.shape, float))
        ratio = stable_ratio(M, tau, np.asarray(sim.shape, float), alpha)
    elif isinstance(sim, GaltonWatson):
        rho = _rho2(M)
        step = sim.hours_per_generation or 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.log(rho) / step
            ratio = M[:, 0, 1] / (rho - M[:, 1, 1])
    else:
        raise TypeError(f"derived quantities: unsupported simulator {type(sim).__name__}")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.DataFrame({
            "A_self_renewal": phys[:, names.index("A.div_AA")],
            "growth_rate_per_h": alpha,
            "doubling_time_h": np.log(2) / alpha,
            "stable_B_to_A": ratio,
            "stable_B_fraction": ratio / (1 + ratio),
        })  # fmt: skip
    return out
