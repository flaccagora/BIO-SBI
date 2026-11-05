"""Generative branching-process models.

All simulators subclass :class:`BranchingSimulator` and share the configurable
:class:`OffspringLaw` (cell types, outcomes and Dirichlet priors). Select one with the
Hydra group ``simulator`` (``simulator=galton_watson`` or ``simulator=bellman_harris``).
"""

from .base import INVALID, BranchingSimulator, ScalarParameter
from .bellman_harris import BellmanHarris
from .galton_watson import GaltonWatson
from .law import OffspringLaw

__all__ = [
    "INVALID",
    "BellmanHarris",
    "BranchingSimulator",
    "GaltonWatson",
    "OffspringLaw",
    "ScalarParameter",
]
