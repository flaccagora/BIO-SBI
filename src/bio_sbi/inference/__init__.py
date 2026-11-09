"""sbi inference methods behind a common interface.

Select with the Hydra group ``inference`` (``inference=npe|nle|nre|exact_gw``). Each method is
an :class:`InferenceMethod`; :meth:`InferenceMethod.fit` returns a
:class:`FittedPosterior` that can be saved, reloaded and conditioned on new data.
"""

from .base import FittedPosterior, InferenceMethod
from .exact_gw import ExactGW, ExactPosterior
from .methods import NLE, NPE, NRE

__all__ = ["NLE", "NPE", "NRE", "ExactGW", "ExactPosterior", "FittedPosterior", "InferenceMethod"]
