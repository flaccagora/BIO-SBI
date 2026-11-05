"""Common training loop shared by all sbi inference methods.

:class:`InferenceMethod` implements single-round (amortised) and multi-round
(sequential) training against a simulation callback. Subclasses only decide how the
sbi trainer is constructed, how simulations are appended and how a posterior is built.

The result is a :class:`FittedPosterior`, a picklable wrapper that applies the same
feature mask used in training and returns samples in inference coordinates ``theta``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..rng import seed_for, seed_torch

#: ``simulate(theta, round_index) -> x`` with NaN rows for invalid datasets.
SimulateFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class FittedPosterior:
    """Trained sbi posterior plus the preprocessing needed to condition it.

    Attributes:
        posterior: The sbi ``NeuralPosterior``.
        feature_mask: Boolean mask of summary features used in training (features
            constant over the training set are dropped, which matters for NLE).
        method: Name of the inference method.
        amortized: ``True`` if trained on prior simulations only (valid for any ``x``).
        history: Per-round training diagnostics.
        device: Torch device the posterior lives on (``"cpu"`` or ``"cuda"``).
    """

    posterior: Any
    feature_mask: np.ndarray
    method: str
    amortized: bool
    history: list[dict] = field(default_factory=list)
    device: str = "cpu"

    def condition(self, x: np.ndarray) -> torch.Tensor:
        """Apply the feature mask and convert a summary vector (or batch) to a tensor."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float32))[:, self.feature_mask]
        return torch.as_tensor(x, device=getattr(self, "device", "cpu"))

    #: Wall-clock cap (s) on NPE rejection sampling; see :meth:`sample`.
    max_sampling_time: float = 60.0

    def _is_direct(self) -> bool:
        from sbi.inference.posteriors import DirectPosterior

        return isinstance(self.posterior, DirectPosterior)

    def sample(self, x: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
        """Draw up to ``n`` posterior samples of ``theta`` given one summary vector ``x``.

        For NPE, sbi draws from the flow and rejects samples outside the prior's support.
        If the flow puts almost all its mass outside the support (typically when ``x``
        is far from anything the model can produce, i.e. under misspecification), this
        never finishes. Sampling is therefore capped at :attr:`max_sampling_time`
        seconds and returns the samples accepted so far (possibly fewer than ``n``, or
        none), with a warning. Use :meth:`support_mass` to quantify the problem.
        """
        seed_torch(seed)
        options = {"show_progress_bars": False}
        if self._is_direct():
            options.update(
                max_sampling_time=getattr(self, "max_sampling_time", 60.0),
                return_partial_on_timeout=True,
            )
        try:
            with torch.no_grad():
                samples = self.posterior.sample((n,), x=self.condition(x), **options)
        except RuntimeError as error:  # sbi raises when the cap is hit with 0 accepted
            if "max_sampling_time" not in str(error):
                raise
            samples = torch.empty((0, self.posterior.prior.event_shape[0]))
        if len(samples) < n:
            import warnings

            warnings.warn(
                f"Only {len(samples)}/{n} posterior samples inside the prior support were "
                "drawn before the time limit: the posterior mass lies mostly outside the "
                "prior (see support_mass); results for this x are unreliable.",
                stacklevel=2,
            )
        return samples.cpu().numpy()

    def support_mass(self, x: np.ndarray, n: int = 10_000, seed: int = 0) -> float | None:
        """Fraction of the NPE flow's mass at ``x`` that lies inside the prior support.

        Close to 1 normally; close to 0 means the trained network maps ``x`` to
        parameters the prior (hence the simulator) does not allow, a strong sign that the
        model cannot explain ``x``. ``None`` for NLE/NRE (their MCMC respects the
        support by construction).
        """
        if not self._is_direct():
            return None
        seed_torch(seed)
        with torch.no_grad():
            draws = self.posterior.sample(
                (n,), x=self.condition(x), reject_outside_prior=False, show_progress_bars=False
            )
            inside = self.posterior.prior.support.check(draws)
        return float(inside.float().mean())

    def log_prob(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Posterior log-density (normalised for NPE; unnormalised for NLE/NRE)."""
        with torch.no_grad():
            values = self.posterior.log_prob(
                torch.as_tensor(theta, dtype=torch.float32, device=getattr(self, "device", "cpu")),
                x=self.condition(x),
            )
        return values.cpu().numpy()

    def save(self, path: str | Path) -> None:
        """Pickle the posterior with ``torch.save`` (load only trusted files)."""
        torch.save(self, path)

    @staticmethod
    def load(path: str | Path) -> FittedPosterior:
        """Load a posterior written by :meth:`save` onto the CPU (pickle: trusted files only).

        Posteriors trained on a GPU are moved to the CPU, so they can be used on machines
        without one; call ``fitted.to("cuda")`` to move them back.
        """
        fitted = torch.load(path, weights_only=False, map_location="cpu")
        if getattr(fitted, "device", "cpu") != "cpu":
            fitted.to("cpu")
        return fitted

    def to(self, device: str) -> FittedPosterior:
        """Move the posterior (estimator, prior, default ``x``) to ``device``."""
        self.posterior.to(device)
        self.device = str(device)
        return self


class InferenceMethod(ABC):
    """Base class for sbi methods.

    Args:
        rounds: ``1`` = amortised training on prior simulations. ``> 1`` = sequential
            training where rounds ``2..R`` simulate from the current posterior at the
            reference observation (more simulation-efficient, but valid only for that
            observation, so SBC is skipped).
        estimator: Network family passed to the sbi builder (e.g. ``"nsf"``, ``"maf"``,
            ``"resnet"``).
        network: Extra keyword arguments for the sbi network builder.
        training: Keyword arguments forwarded to ``trainer.train`` (batch size,
            learning rate, epochs, early stopping, ...).
        sampling: Keyword arguments forwarded to ``trainer.build_posterior``.
        device: Torch device for training.
        seed: Root seed for torch stages.
    """

    name: str = "base"

    def __init__(
        self,
        rounds: int = 1,
        estimator: str = "nsf",
        network: Mapping[str, Any] | None = None,
        training: Mapping[str, Any] | None = None,
        sampling: Mapping[str, Any] | None = None,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        self.rounds = int(rounds)
        self.estimator = estimator
        self.network = dict(network or {})
        self.training = dict(training or {})
        self.sampling = dict(sampling or {})
        self.device = device
        self.seed = seed

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def make_trainer(self, prior: torch.distributions.Distribution, tracker: Any) -> Any:
        """Construct the sbi trainer object (``tracker`` logs training curves or is None)."""

    def append(self, trainer: Any, theta: torch.Tensor, x: torch.Tensor, proposal: Any) -> None:
        """Add one round of simulations (NPE overrides this to pass the proposal)."""
        trainer.append_simulations(theta, x)

    def build_posterior(self, trainer: Any, estimator: Any) -> Any:
        """Build the sbi posterior object from a trained estimator."""
        return trainer.build_posterior(estimator, **self.sampling)

    # ------------------------------------------------------------------ loop
    def split_budget(self, simulations: int) -> list[int]:
        """Split a total simulation budget into equal rounds (remainder to round 1)."""
        per = simulations // self.rounds
        if per < 16:
            raise ValueError("Need at least 16 simulations per round")
        return [simulations - per * (self.rounds - 1)] + [per] * (self.rounds - 1)

    def fit(
        self,
        prior: torch.distributions.Distribution,
        simulate: SimulateFn,
        simulations: int,
        x_observed: np.ndarray | None = None,
        log_dir: str | Path | None = None,
    ) -> FittedPosterior:
        """Train the estimator, simulating each round's parameters on the fly.

        Args:
            prior: Prior over ``theta``, on the training device (``self.device``).
            simulate: Callback returning summaries for a ``(N, d)`` theta batch.
            simulations: Total simulation budget across all rounds.
            x_observed: Reference summary; required for sequential rounds.
            log_dir: Optional TensorBoard directory for sbi's training curves.

        Returns:
            The fitted posterior wrapper.
        """
        if self.rounds > 1 and x_observed is None:
            raise ValueError("Sequential inference needs a reference observation")
        writer = tracker = None
        if log_dir is not None:
            from sbi.utils.tracking import TensorBoardTracker
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(log_dir))
            tracker = TensorBoardTracker(writer)
        trainer = self.make_trainer(prior, tracker)
        proposal, mask, posterior, history = prior, None, None, []
        for r, n in enumerate(self.split_budget(simulations)):
            seed_torch(seed_for(self.seed, "proposal", r))
            start = time.perf_counter()
            options = {} if r == 0 else {"show_progress_bars": False}
            theta = proposal.sample((n,), **options).cpu().numpy()
            x = simulate(theta, r)
            sim_seconds = time.perf_counter() - start
            valid = np.all(np.isfinite(x), axis=1)
            if valid.sum() < 16:
                raise RuntimeError(f"Only {valid.sum()} valid simulations in round {r}")
            if mask is None:
                mask = x[valid].std(0) > 1e-8
            self.append(
                trainer,
                torch.as_tensor(theta[valid], dtype=torch.float32, device=self.device),
                torch.as_tensor(x[valid][:, mask], device=self.device),
                proposal,
            )
            seed_torch(seed_for(self.seed, "train", r))
            start = time.perf_counter()
            estimator = trainer.train(**self.training)
            posterior = self.build_posterior(trainer, estimator)
            summary = trainer._summary
            history.append(
                {
                    "round": r,
                    "simulations": int(n),
                    "valid_simulations": int(valid.sum()),
                    "simulation_seconds": sim_seconds,
                    "training_seconds": time.perf_counter() - start,
                    "epochs": int(trainer.epoch),
                    "best_validation_loss": float(summary["best_validation_loss"][-1]),
                }
            )
            if r + 1 < self.rounds:
                x_o = torch.as_tensor(
                    np.asarray(x_observed, np.float32)[None, mask], device=self.device
                )
                proposal = posterior.set_default_x(x_o)
        if writer is not None:
            writer.close()
        return FittedPosterior(
            posterior, mask, self.name, self.rounds == 1, history, str(self.device)
        )
