"""The three sbi method families: posterior (NPE), likelihood (NLE) and ratio (NRE).

* **NPE** learns ``q(theta | x)`` directly with a conditional normalising flow.
  Sampling is instant and amortised, which makes calibration checks cheap. Sequential
  rounds use sbi's NPE-C (APT) atomic loss.
* **NLE** learns the summary likelihood ``q(x | theta)`` and samples the posterior with
  MCMC. Robust to proposal choice in sequential rounds and reusable under a different
  prior, but slower to sample.
* **NRE** learns the likelihood-to-evidence ratio with a classifier (contrastive
  NRE-C by default) and samples with MCMC. Sample-efficient and simple to train.

Each class only adapts sbi's trainer; the training loop lives in
:class:`~bio_sbi.inference.base.InferenceMethod`.
"""

from __future__ import annotations

from typing import Any

from sbi.inference import NLE as _NLE
from sbi.inference import NPE as _NPE
from sbi.inference import NRE_B, NRE_C
from sbi.neural_nets import classifier_nn, likelihood_nn, posterior_nn

from .base import InferenceMethod


class _MCMCSampled(InferenceMethod):
    """Mixin for methods whose posterior is sampled by MCMC on the learned potential.

    ``sampling`` holds the fields of ``sbi``'s ``MCMCPosteriorParameters`` (``method``,
    ``num_chains``, ``warmup_steps``, ``thin``, ``init_strategy``, ...).
    """

    def build_posterior(self, trainer, estimator):
        """Build an ``MCMCPosterior`` with the configured sampler settings."""
        from sbi.inference.posteriors.posterior_parameters import MCMCPosteriorParameters

        return trainer.build_posterior(
            estimator, posterior_parameters=MCMCPosteriorParameters(**self.sampling)
        )


class NPE(InferenceMethod):
    """Neural posterior estimation (amortised NPE or sequential NPE-C)."""

    name = "npe"

    def make_trainer(self, prior, tracker):
        """Build an ``sbi.inference.NPE`` trainer with a conditional flow."""
        builder = posterior_nn(model=self.estimator, **self.network)
        return _NPE(
            prior=prior,
            density_estimator=builder,
            device=self.device,
            tracker=tracker,
            show_progress_bars=False,
        )

    def append(self, trainer, theta, x, proposal):
        """Pass the proposal so later rounds use the atomic (APT) correction."""
        trainer.append_simulations(theta, x, proposal=proposal)


class NLE(_MCMCSampled):
    """Neural likelihood estimation with MCMC posterior sampling."""

    name = "nle"

    def make_trainer(self, prior, tracker):
        """Build an ``sbi.inference.NLE`` trainer with a flow over summaries."""
        builder = likelihood_nn(model=self.estimator, **self.network)
        return _NLE(
            prior=prior,
            density_estimator=builder,
            device=self.device,
            tracker=tracker,
            show_progress_bars=False,
        )


class NRE(_MCMCSampled):
    """Neural ratio estimation (NRE-C by default) with MCMC posterior sampling.

    Args:
        variant: ``"C"`` (contrastive, Miller et al. 2022) or ``"B"`` (SRE, Durkan
            et al. 2020).
        **kwargs: See :class:`~bio_sbi.inference.base.InferenceMethod`.
    """

    name = "nre"

    def __init__(self, variant: str = "C", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if variant not in ("B", "C"):
            raise ValueError("NRE variant must be 'B' or 'C'")
        self.variant = variant

    def make_trainer(self, prior, tracker):
        """Build an ``sbi`` NRE-B/C trainer with a classifier network."""
        builder = classifier_nn(model=self.estimator, **self.network)
        cls = NRE_C if self.variant == "C" else NRE_B
        return cls(
            prior=prior,
            classifier=builder,
            device=self.device,
            tracker=tracker,
            show_progress_bars=False,
        )
