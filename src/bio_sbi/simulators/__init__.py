"""Generative branching-process models.

All simulators subclass :class:`BranchingSimulator` and share the configurable
:class:`OffspringLaw` (cell types, outcomes, founders) and :class:`ParameterSpace`
(fixed values, priors, inference coordinates). Select one with the Hydra group
``simulator`` (``galton_watson`` or ``bellman_harris``); the data-generating copy is
``simulator@data.simulator``.
"""

from .base import INVALID, BranchingSimulator
from .bellman_harris import BellmanHarris
from .galton_watson import GaltonWatson
from .law import OffspringLaw, ParameterSpace, ScalarParameter, all_outcomes

__all__ = [
    "INVALID",
    "BellmanHarris",
    "BranchingSimulator",
    "GaltonWatson",
    "OffspringLaw",
    "ParameterSpace",
    "ScalarParameter",
    "all_outcomes",
]
