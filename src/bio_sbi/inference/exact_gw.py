"""Exact-likelihood Bayesian inference for the two-type Galton-Watson model (``exact_gw``).

Instead of learning from simulations, the posterior is sampled with NUTS
(:mod:`bio_sbi.inference.nuts`) on the *exact* likelihood of the raw endpoint counts,

    log p(data | theta) = sum_h sum_clones log p~_h(A, B | theta),

with ``p~_h`` the law of a clone's counts at horizon ``h`` (computed by
:mod:`bio_sbi.inference.gw_exact`, conditioned on detection for ``selection=detected``).
Clones are independent and each horizon's cohort is independent. Counts are histogrammed
per horizon, so one evaluation is a dot product of multiplicities with ``log p~``.

The posterior lives on the same coordinates and prior as NPE: ``theta`` = the
stick-breaking coordinates of :class:`~bio_sbi.simulators.law.ParameterSpace` with their
Beta priors (Dirichlet rows). NUTS runs on ``u = logit(theta)`` (Jacobian included).
Gradients are exact (forward mode through the pgf recursion).

Supported: ``simulator=galton_watson`` / ``galton_watson_hours``, two types (any outcome
table and fixed entries, e.g. ``two_type`` or ``reversible_ab``), fixed founder rows
(e.g. single A founder) or free ones (opt-in: e.g. ``'~law.fixed.founder'`` infers the
founder-type probability ``founder.A``; ``law.founder_sizes=[1,2]`` also infers the
founder-size probabilities; Dirichlet priors from ``law.concentration`` as for every
row; exact gradients, see :func:`~bio_sbi.inference.gw_exact.log_likelihood_founder`;
default coordinates only, ``reparam: none``), homogeneous law, two measured labels with any capture and 2x2
confusion (pgf substitution ``x -> 1 - c + c (C x + ...)``), ``selection`` detected or
all. Anything else raises ``ValueError``.

The fitted object is an :class:`ExactPosterior`: a :class:`FittedPosterior` holding the
MCMC draws for the reference dataset (not amortised). The pipeline runs SBC by
re-running MCMC on each calibration dataset (:meth:`ExactGW.calibrate`).
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from ..rng import rng_for
from .base import FittedPosterior, InferenceMethod
from .gw_exact import (
    GridSpec,
    GWModel,
    HorizonData,
    founder_composition_laws,
    grid_spec,
    histogram,
    law_and_jac,
    log_likelihood_founder,
)
from .gw_exact import log_likelihood as _log_likelihood
from .gw_reparam import StickCoordinates, parse_spec
from .nuts import NutsChain, ess, split_rhat
from .smc import _rank_normal, tempered_smc
from .smc import ess_bulk as rank_ess_bulk
from .smc import ess_tail as rank_ess_tail
from .smc import rhat as rank_rhat

LOG = logging.getLogger(__name__)

#: Box for the unconstrained coordinates u = logit(theta) during initial optimisation.
U_BOUND = 15.0


class GridTooLarge(ValueError):
    """The counts need an FFT grid above ``grid.max_points``."""


# ---------------------------------------------------------------------- support check
def check_supported(simulator, observation) -> None:
    """Raise ``ValueError`` unless the exact likelihood applies to this model."""
    from ..simulators import GaltonWatson

    if not isinstance(simulator, GaltonWatson):
        raise ValueError(
            "inference=exact_gw needs simulator=galton_watson (or galton_watson_hours); "
            f"got {type(simulator).__name__}"
        )
    if len(simulator.types) != 2:
        raise ValueError(f"inference=exact_gw supports two cell types; got {simulator.types}")
    if simulator.heterogeneous:
        raise ValueError("inference=exact_gw does not support law.clone_heterogeneity")
    space = simulator.space
    if space.free_scalars:
        raise ValueError(
            f"inference=exact_gw: free scalar parameters {[s.name for s in space.free_scalars]}"
            " are not supported"
        )
    if observation.n_labels != 2 or tuple(observation.types) != tuple(simulator.types):
        raise ValueError("inference=exact_gw needs two measured labels for the two types")


# ---------------------------------------------------------------------- likelihood
def _stick_jacobian(s: np.ndarray, r) -> np.ndarray:
    """``d p_free / d s`` ``(n_sticks, len(free))`` of a stick-broken row (mass ``r.mass``)."""
    m = len(r.free)
    d = np.zeros((m - 1, m))
    for j in range(m - 1):
        for i in range(j, m):
            others = np.prod([1.0 - s[k] for k in range(i) if k != j])
            si = s[i] if i < m - 1 else 1.0
            d[j, i] = r.mass * others * (1.0 if i == j else -si)
    return d


class ExactLikelihood:
    """Exact log-likelihood and log-posterior of one dataset (list of cohorts).

    Args:
        simulator: Inference-model Galton-Watson simulator.
        observation: Inference-model observation model.
        cohorts: The dataset (``Cohort`` list); counts per measured label.
        grid: ``factor``, ``margin``, ``min_size``, ``damping``, ``max_points`` for
            :func:`~bio_sbi.inference.gw_exact.grid_spec` (grid size per horizon from the
            largest observed counts of *this* dataset).
        floor, min_detection: Probability and detection-probability floors (see
            :func:`~bio_sbi.inference.gw_exact.log_likelihood`).
        kernel: ``"loop"`` (forward-mode derivatives, default) or ``"adjoint"``
            (polynomial-form pgf, reverse-mode gradient: same values and gradients up to
            round-off, ~3x faster gradients); see
            :func:`~bio_sbi.inference.gw_exact.log_likelihood`.
    """

    def __init__(self, simulator, observation, cohorts, grid: dict | None = None,
                 floor: float = 1e-15, min_detection: float = 1e-6,
                 kernel: str = "loop") -> None:  # fmt: skip
        check_supported(simulator, observation)
        if kernel not in ("loop", "adjoint"):
            raise ValueError(f"inference.method.kernel must be loop or adjoint; got {kernel!r}")
        self.kernel = kernel
        self.sim = simulator
        self.space = simulator.space
        self.floor = float(floor)
        self.min_detection = float(min_detection)
        law = simulator.law
        theta0 = np.full(self.space.theta_dim, 0.5)  # fixed founder rows: any theta
        P, founder, size_probs = law.unpack(self.space, self.space.decode(theta0))
        self.model = GWModel(
            offspring=np.ascontiguousarray(law.offspring, dtype=np.int64),
            n_out=np.ascontiguousarray(law.n_outcomes, dtype=np.int64),
            capture=np.asarray(observation.capture, dtype=float),
            confusion=np.asarray(observation.confusion, dtype=float),
            founder=np.asarray(founder, dtype=float),
            sizes=np.asarray(law.founder_sizes, dtype=np.int64),
            size_probs=np.asarray(size_probs, dtype=float),
            detected=observation.selection == "detected",
        )
        grid = dict(grid or {})
        max_points = float(grid.pop("max_points", 4e6))
        self.grid_settings = dict(grid)  # grid_spec options (re-gridding: with_settings)
        self.data: list[HorizonData] = []
        for c in cohorts:
            a, b, w = histogram(c.counts)
            if self.model.detected and np.any((a == 0) & (b == 0)):
                raise ValueError("selection=detected but the data contain empty clones")
            spec = grid_spec(int(a.max()), int(b.max()), **grid)
            if spec.Lx * spec.Ly > max_points:
                raise GridTooLarge(
                    f"Horizon {c.horizon:g}: counts up to ({a.max()}, {b.max()}) need a "
                    f"{spec.Lx}x{spec.Ly} grid > grid.max_points={max_points:g}"
                )
            self.data.append(HorizonData(simulator.generations(c.horizon), a, b, w, spec))
        # theta coordinate -> (type row, free entries, stick index); free founder rows
        # (founder types / founder sizes) come after the type rows in theta
        self.directions = []
        self.founder_rows: list = []  # (row, first theta index) of free founder rows
        self.prior_a, self.prior_b = [], []
        t = 0
        for r in self.space.rows:
            if not r.n_sticks:
                continue
            if r.prefix in ("founder", "founder_size"):
                self.founder_rows.append((r, t))
            else:
                if self.founder_rows:
                    raise ValueError("exact_gw: type rows must precede the founder rows")
                k = law.types.index(r.prefix)
                for j in range(r.n_sticks):
                    self.directions.append((k, r, j))
            t += r.n_sticks
            tail = np.cumsum(r.alpha[::-1])[::-1]
            self.prior_a.extend(r.alpha[:-1])
            self.prior_b.extend(tail[1:])
        self.prior_a = np.asarray(self.prior_a, dtype=float)
        self.prior_b = np.asarray(self.prior_b, dtype=float)
        self.dim = t
        #: Theta coordinates of the outcome rows (``dim`` unless founder rows are free).
        self.n_row_dims = len(self.directions)
        self.n_outcomes_max = law.offspring.shape[1]
        self._lock = threading.Lock()
        self.evals, self.eval_seconds = 0, 0.0
        self.grad_evals = 0  # evaluations with gradient (``evals`` counts all)

    def with_settings(self, grid: dict | None = None, floor: float | None = None,
                      min_detection: float | None = None) -> ExactLikelihood:  # fmt: skip
        """Copy of this likelihood with other grid options / floors (fresh counters).

        ``grid`` updates the :func:`~bio_sbi.inference.gw_exact.grid_spec` options this
        likelihood was built with (no ``max_points`` check); used by numerical checks.
        """
        new = copy.copy(self)
        if grid is not None:
            new.grid_settings = {**self.grid_settings, **grid}
            new.grid_settings.pop("max_points", None)
            new.data = [
                replace(h, grid=grid_spec(int(h.a.max()), int(h.b.max()), **new.grid_settings))
                for h in self.data
            ]
        if floor is not None:
            new.floor = float(floor)
        if min_detection is not None:
            new.min_detection = float(min_detection)
        new._lock = threading.Lock()
        new.evals, new.eval_seconds, new.grad_evals = 0, 0.0, 0
        return new

    # ---------------------------------------------------------------- parameters
    @property
    def founder_free(self) -> bool:
        """Whether the founder type and/or founder size probabilities are inferred."""
        return bool(self.founder_rows)

    def model_at(self, theta: np.ndarray) -> GWModel:
        """:attr:`model` with the founder distribution at ``theta`` (free founder rows)."""
        if not self.founder_free:
            return self.model
        _, founder, size_probs = self.sim.law.unpack(self.space, self.space.decode(theta))
        return replace(self.model, founder=np.asarray(founder, dtype=float),
                       size_probs=np.asarray(size_probs, dtype=float))  # fmt: skip

    @property
    def grids(self) -> list[GridSpec]:
        """FFT grid of every horizon."""
        return [h.grid for h in self.data]

    def probabilities(self, theta: np.ndarray) -> np.ndarray:
        """Outcome-probability matrix ``P (2, O)`` at ``theta`` (no derivatives)."""
        theta = np.asarray(theta, dtype=float)
        P, _, _ = self.sim.law.unpack(self.space, self.space.decode(theta))
        return P

    def jacobian(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``P (2, O)``, ``dP (d, O)`` (row derivative per coordinate) and its row index."""
        theta = np.asarray(theta, dtype=float)
        P = self.probabilities(theta)
        dP = np.zeros((self.n_row_dims, self.n_outcomes_max))
        rows = np.zeros(self.n_row_dims, dtype=np.int64)
        start = 0
        for t, (k, r, j) in enumerate(self.directions):
            if j == 0:
                start = t
            s = theta[start : start + r.n_sticks]
            free = list(r.free)
            m = len(free)
            rows[t] = k
            for i in range(m):
                # p_i = mass * s_i * prod_{l<i} (1 - s_l), with s_{m-1} := 1 for the last
                if i < j:
                    continue
                si = s[i] if i < m - 1 else 1.0
                others = np.prod([1 - s[k2] for k2 in range(i) if k2 != j])
                dP[t, free[i]] = r.mass * (others if i == j else -si * others)
        return P, dP, rows

    def log_likelihood(self, theta: np.ndarray, grad: bool = True,
                       weights: list[np.ndarray] | None = None) -> tuple[float, np.ndarray]:  # fmt: skip
        """Log-likelihood and its gradient with respect to ``theta``.

        ``grad=False`` skips the Jacobian and the derivative recursion (objective only;
        the returned gradient is empty). The value is the same computation either way.
        ``weights`` (one array per horizon, aligned with that horizon's histogram cells)
        replaces the cell multiplicities, e.g. rescaled minibatch counts (each must sum
        to the horizon's clone count so the detection term is unchanged); the pgf / FFT
        cost is the same as for the full data.
        """
        t0 = time.perf_counter()
        if grad:
            P, dP, rows = self.jacobian(theta)
        else:
            P = self.probabilities(theta)
            dP, rows = np.zeros((0, self.n_outcomes_max)), np.zeros(0, dtype=np.int64)
        if self.founder_free:
            return self._log_likelihood_founder(theta, P, dP, rows, grad, weights, t0)
        return self.log_likelihood_P(P, dP, rows, grad, weights, t0)

    def _log_likelihood_founder(self, theta, P, dP, rows, grad, weights, t0):
        """Free founder rows: :func:`~bio_sbi.inference.gw_exact.log_likelihood_founder`."""
        theta = np.asarray(theta, dtype=float)
        data = self.data
        if weights is not None:
            data = [replace(h, weight=np.asarray(w, dtype=float))
                    for h, w in zip(self.data, weights, strict=True)]  # fmt: skip
        model = self.model_at(theta)
        ll, g_dir, g_phi, g_pi = log_likelihood_founder(
            model, P, dP, rows, data, self.floor, self.min_detection, self.kernel, grad
        )
        g = np.zeros(self.dim if grad else 0)
        if grad:
            g[: self.n_row_dims] = g_dir
            for r, t in self.founder_rows:
                ge = g_phi if r.prefix == "founder" else g_pi  # per entry of the row
                s = theta[t : t + r.n_sticks]
                g[t : t + r.n_sticks] = _stick_jacobian(s, r) @ ge[list(r.free)]
        with self._lock:
            self.evals += 1
            self.grad_evals += bool(grad)
            self.eval_seconds += time.perf_counter() - t0
        return ll, g

    def progenitor_posterior(self, theta: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
        """Posterior founder composition of every histogram cell at ``theta``.

        Returns ``(labels, probs)``: ``labels`` the compositions (``"A"``, ``"B"``,
        ``"AA"``, ``"AB"``, ``"BB"``, ... with nonzero prior probability), ``probs`` one
        ``(cells, K)`` array per horizon aligned with that horizon's histogram cells
        ``(h.a, h.b)``: ``P(founders = c | counts)`` from :func:`~bio_sbi.inference.
        gw_exact.founder_composition_laws` (cells of negligible probability get the
        prior composition). Every horizon has the same labels.
        """
        theta = np.asarray(theta, dtype=float)
        model = self.model_at(theta)
        P = self.probabilities(theta)
        labels, out = None, []
        for h in self.data:
            lab, p = founder_composition_laws(model, P, h.generations, h.grid)
            labels = lab
            w = p[:, h.a, h.b].T  # (cells, K)
            tot = w.sum(1, keepdims=True)
            prior = np.array([p_c.sum() for p_c in p])
            prior = prior / prior.sum()
            out.append(np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), prior[None]))
        return labels, out

    def log_likelihood_P(self, P: np.ndarray, dP: np.ndarray, rows: np.ndarray,
                         grad: bool = True, weights: list[np.ndarray] | None = None,
                         t0: float | None = None) -> tuple[float, np.ndarray]:  # fmt: skip
        """Log-likelihood at the outcome matrix ``P (2, O)`` and its gradient along ``dP``.

        ``dP (D, O)``: derivative of ``P`` along direction ``t``, which changes row
        ``rows[t]`` only - any parametrisation of the rows (e.g.
        :mod:`~bio_sbi.inference.gw_reparam`). ``grad=False``: pass empty ``dP`` /
        ``rows``. Counts towards ``evals`` like :meth:`log_likelihood`.
        """
        t0 = time.perf_counter() if t0 is None else t0
        data = self.data
        if weights is not None:
            data = [replace(h, weight=np.asarray(w, dtype=float))
                    for h, w in zip(self.data, weights, strict=True)]  # fmt: skip
        ll, g = _log_likelihood(
            self.model, P, dP, rows, data, self.floor, self.min_detection, self.kernel
        )
        with self._lock:
            self.evals += 1
            self.grad_evals += bool(grad)
            self.eval_seconds += time.perf_counter() - t0
        return ll, g

    def log_posterior_u(self, u: np.ndarray) -> tuple[float, np.ndarray]:
        """Log-posterior density of ``u = logit(theta)`` (Jacobian included) and gradient."""
        u = np.asarray(u, dtype=float)
        theta = expit(u)
        if np.any(theta <= 0) or np.any(theta >= 1):
            return -np.inf, np.zeros_like(u)
        ll, g = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf, np.zeros_like(u)
        # Beta(a, b) prior on theta times |d theta / d u| = theta (1 - theta)
        lp = ll + float(self.prior_a @ log_expit(u) + self.prior_b @ log_expit(-u))
        grad = g * theta * (1 - theta) + self.prior_a * (1 - theta) - self.prior_b * theta
        return lp, grad

    @property
    def seconds_per_eval(self) -> float:
        """Mean wall time of one likelihood (+ gradient) evaluation so far."""
        return self.eval_seconds / self.evals if self.evals else float("nan")

    # ---------------------------------------------------------------- checks
    def grid_check(self, theta: np.ndarray) -> list[dict]:
        """Accuracy of the grids at ``theta`` against an undamped grid twice as large.

        Per horizon: ``tail_mass`` = mass of the (conditional) law at counts outside the
        ``Lx x Ly`` grid, and ``max_rel_error`` of the used (damped) law on the observed
        cells.
        """
        P, dP, rows = self.jacobian(theta)
        model = self.model_at(theta)
        out = []
        for h in self.data:
            g = h.grid
            p, _ = law_and_jac(model, P, dP[:0], rows[:0], h.generations, g)
            big = GridSpec(2 * g.Lx, 2 * g.Ly, 1.0, 1.0)
            pb, _ = law_and_jac(model, P, dP[:0], rows[:0], h.generations, big)
            norm = 1 - pb[0, 0] if self.model.detected else 1.0
            inside = pb[: g.Lx, : g.Ly].sum() - (pb[0, 0] if self.model.detected else 0.0)
            ref = pb[h.a, h.b]
            keep = ref > 100 * self.floor
            rel = np.abs(p[h.a, h.b][keep] - ref[keep]) / ref[keep]
            out.append({
                "generations": h.generations, "Lx": g.Lx, "Ly": g.Ly, "rx": g.rx, "ry": g.ry,
                "max_count_A": int(h.a.max()), "max_count_B": int(h.b.max()),
                "tail_mass": float(max(1 - inside / norm, 0.0)),
                "max_rel_error": float(rel.max()) if len(rel) else 0.0,
            })  # fmt: skip
        return out


# ---------------------------------------------------------------------- warm start / stop
#: Defaults of ``inference.method.warm_start_options`` (see :class:`ExactGW`).
WARM_START_DEFAULTS: dict = {
    "scale": None,  # covariance multiplier; None = n_prev / n (total clones)
    "n_prev": None,  # total clones of the previous dataset (None: from its diagnostics)
    "warmup": 150,  # warmup iterations per chain (replaces ``warmup``)
    "init_buffer": 25,  # step-size-only iterations before the mass-matrix window
    "term_buffer": 50,  # step-size-only iterations after it
    "mass_prior": 50.0,  # pseudo-draws of the scaled previous covariance in the window
    "select": "kmeans",  # kmeans (spread over the previous posterior) | chains
    "step_size": "previous",  # previous (median adapted step) | heuristic
    "check_starts": 0,  # >0: also run this many L-BFGS starts and flag unvisited optima
    "check_tolerance": 2.0,  # nats above the best sampled log posterior
}

#: Defaults of ``inference.method.stop`` (see :class:`ExactGW`).
STOP_DEFAULTS: dict = {
    "enabled": False,
    "ess": 400.0,  # min rank-normalised bulk ESS (all chains, every monitored quantity)
    "tail_ess": 200.0,  # min tail ESS (5 % / 95 % quantiles)
    "rhat": 1.01,  # max rank-normalised split R-hat (bulk and folded)
    "max_mean_spread": 0.5,  # max spread of the chain means of rank-normalised draws
    "block": 50,  # draws per chain between checks (chains advance in lockstep)
    "min_draws": 100,  # per chain, before the first check
    "max_draws": 1000,  # per chain: stop here even if the targets are not met
    "confirm": 1,  # consecutive passing checks required
    "monitor_logp": True,  # include the log density in the R-hat / spread checks
}


#: Defaults of ``inference.method.smc`` (see :class:`ExactGW`).
SMC_DEFAULTS: dict = {
    "enabled": False,
    "mode": "start",  # start (SMC population -> NUTS chains) | final (SMC chains = draws)
    "chains": 64,  # MH chains; population = chains x steps (smc.tempered_smc)
    "steps": 8,
    "ess_fraction": 0.5,  # next temperature keeps this Kish ESS fraction
    "rw_acceptance": 0.3,
    "imh_prob": 0.5,
    "imh_components": 2,  # BIC-selected t mixture (separate branches)
    "df": 5.0,
    "de_prob": 0.0,
    "rw_scale_sd": 0.0,
    "max_evals": 60000,  # likelihood evaluations (no gradient) incl. the final stage
    # mode=start: MH steps at beta = 1 before the hand-over, NUTS schedule
    "final_steps": 16,
    "select": "resample",  # resample (chain starts in proportion to mass) | kmeans (spread)
    "warmup": 200,  # NUTS warmup per chain (replaces ``warmup``)
    "init_buffer": 25,
    "term_buffer": 50,
    "mass_prior": 50.0,  # pseudo-draws of the population covariance in the mass window
    # mode=final: stopping rule of the final chains (as exact_bh)
    "target_ess": 400.0,
    "max_rhat": 1.01,  # nested R-hat
    "tail_fraction": 0.5,
    "superchains": 4,
    "min_final_steps": 16,
    "max_final_steps": 3000,
    "final_burn": 0.25,
    "check_every": 8,
    "local_k": 0,
    "keep_population": True,  # store the beta = 1 population in the diagnostics
}


def _options(given: dict | None, defaults: dict, what: str) -> dict:
    given = dict(given or {})
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f"inference.method.{what}: unknown options {sorted(unknown)}")
    return {**defaults, **given}


@dataclass
class PreviousPosterior:
    """Draws of an earlier ``exact_gw`` run, used to warm-start a new one."""

    theta: np.ndarray  # (draws, d) stick coordinates
    chain: np.ndarray | None  # (draws,) chain index, or None if unknown
    clones: int | None  # total clones of the dataset (all horizons)
    step_size: list | None  # adapted step sizes of its chains
    source: str


def load_previous_posterior(path, space) -> PreviousPosterior:
    """Read a previous run's draws: a run directory, ``posterior.pt`` or ``posterior_samples.csv.gz``.

    A run directory is read from ``posterior.pt`` (all draws, chain-major, with the
    sampler diagnostics) or else ``posterior_samples.csv.gz`` (physical parameters,
    encoded to ``theta``; chains known if the file holds all ``chains x draws_per_chain``
    draws, from ``metrics.json``). ``posterior.pt`` is a pickle: trusted files only.
    """
    path = Path(path).expanduser()
    if path.is_dir():
        run_dir = path
        path = path / "posterior.pt" if (path / "posterior.pt").exists() else (
            path / "posterior_samples.csv.gz")  # fmt: skip
    else:
        run_dir = path.parent
    if not path.exists():
        raise FileNotFoundError(f"inference.method.warm_start: {path} not found")
    diag: dict = {}
    if path.suffix == ".pt":
        from .base import FittedPosterior

        fitted = FittedPosterior.load(path)
        theta = np.asarray(getattr(fitted, "draws", None), dtype=float)
        diag = dict(getattr(fitted, "diagnostics", {}) or {})
        if theta.ndim != 2:
            raise ValueError(f"{path}: not an exact_gw posterior (no draws)")
    else:
        import pandas as pd

        frame = pd.read_csv(path)
        missing = [n for n in space.names if n not in frame.columns]
        if missing:
            raise ValueError(f"{path}: columns {missing} missing")
        phys = frame[list(space.names)].to_numpy(dtype=float)
        theta = np.asarray([space.encode(p) for p in phys], dtype=float)
        metrics = run_dir / "metrics.json"
        if metrics.exists():
            diag = json.loads(metrics.read_text()).get("exact_gw", {}) or {}
    if theta.shape[1] != space.theta_dim:
        raise ValueError(
            f"{path}: draws have {theta.shape[1]} coordinates, the model has {space.theta_dim}"
        )
    chain = None
    m, n = diag.get("chains"), diag.get("draws_per_chain")
    if m and n and int(m) * int(n) == len(theta):
        chain = np.repeat(np.arange(int(m)), int(n))
    clones = None
    if diag.get("grids"):
        clones = int(sum(g["clones"] for g in diag["grids"]))
    steps = diag.get("step_size")
    return PreviousPosterior(theta, chain, clones, list(steps) if steps else None, str(path))


def _to_u(theta: np.ndarray) -> np.ndarray:
    t = np.clip(theta, 1e-12, 1 - 1e-12)
    return np.clip(np.log(t) - np.log1p(-t), -U_BOUND, U_BOUND)


def _within_covariance(u: np.ndarray, chain: np.ndarray | None) -> np.ndarray:
    """Mean within-chain covariance (pooled covariance if the chains are unknown).

    Chains stuck in different modes would inflate the pooled covariance by the distance
    between the modes; the within-chain covariance is the local scale.
    """
    d = u.shape[1]
    if chain is not None:
        covs = [np.cov(u[chain == c], rowvar=False).reshape(d, d)
                for c in np.unique(chain) if (chain == c).sum() > d + 1]  # fmt: skip
        if covs:
            return np.mean(covs, axis=0)
    return np.cov(u, rowvar=False).reshape(d, d)


def spread_starts(u: np.ndarray, k: int, rng: np.random.Generator,
                  iters: int = 50) -> list[np.ndarray]:  # fmt: skip
    """k-means (k-means++ seeding) of ``u`` whitened by its covariance.

    Returns, per cluster, the indices of its members sorted by distance to the centroid
    (the first is the chain's start, the others fallbacks). Clusters are spread over the
    whole previous posterior, so separate modes get their own chains.
    """
    n, d = u.shape
    k = int(min(k, n))
    cov = np.cov(u, rowvar=False).reshape(d, d) + 1e-12 * np.eye(d)
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (u - u.mean(0)).T).T
    centers = [z[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([((z - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(z[rng.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else z[rng.integers(n)])
    C = np.asarray(centers)
    for _ in range(iters):
        dist = ((z[:, None, :] - C[None]) ** 2).sum(2)
        lab = dist.argmin(1)
        new = np.array([z[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    dist = ((z[:, None, :] - C[None]) ** 2).sum(2)
    lab = dist.argmin(1)
    out = []
    for j in range(k):
        idx = np.flatnonzero(lab == j)
        if not len(idx):  # empty cluster: nearest draws to its centre
            idx = np.arange(n)
        out.append(idx[np.argsort(dist[idx, j])])
    return out


def convergence_check(x: np.ndarray, names: list[str], ess_names: set | None = None) -> dict:
    """Rank-normalised split R-hat, bulk / tail ESS and chain-mean spread of ``(chains, n, q)``.

    ``mean_spread`` = (max - min) over chains of the chain means of the rank-normalised
    (pooled) draws, in posterior sd units: chains sitting in different modes give spreads
    of order 1 or more. ESS is computed for ``ess_names`` only (default: all).
    """
    rh, eb, et, sp = {}, {}, {}, {}
    for j, name in enumerate(names):
        v = x[:, :, j]
        rh[name] = rank_rhat(v)
        if ess_names is None or name in ess_names:
            eb[name] = rank_ess_bulk(v)
            et[name] = rank_ess_tail(v)
        z = _rank_normal(v)
        sp[name] = float(np.ptp(z.mean(1))) if x.shape[0] > 1 else 0.0

    def worst(d, fn):
        vals = {k: v for k, v in d.items() if np.isfinite(v)}
        if not vals:
            return float("nan"), None
        key = fn(vals, key=vals.get)
        return float(vals[key]), key

    out = {"rhat": rh, "ess_bulk": eb, "ess_tail": et, "mean_spread": sp}
    out["max_rhat"], out["max_rhat_at"] = worst(rh, max)
    out["min_ess_bulk"], out["min_ess_bulk_at"] = worst(eb, min)
    out["min_ess_tail"], out["min_ess_tail_at"] = worst(et, min)
    out["max_mean_spread"], out["max_mean_spread_at"] = worst(sp, max)
    return out


# ---------------------------------------------------------------------- posterior object
@dataclass
class ExactPosterior(FittedPosterior):
    """MCMC draws of the exact posterior for one dataset.

    ``sample(x, n)`` returns ``n`` draws (without replacement, all draws if fewer are
    available) regardless of ``x``: the draws belong to the dataset they were computed
    for (``amortized`` is ``False``). ``theta`` coordinates as for NPE.
    """

    draws: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)
    #: Point estimate (``inference.method.optimise.enabled``; ``optimisation.json``).
    #: Kept apart from ``draws``: swarm particles / optima are not posterior samples.
    optimisation: dict | None = None
    #: ``beta = 1`` SMC population (``theta``) when ``inference.method.smc`` is enabled.
    smc_population: np.ndarray | None = None

    def sample(self, x: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
        """Return ``n`` stored draws (random subset; all if fewer), ignoring ``x``."""
        draws = self.draws
        if n >= len(draws):
            return draws.copy()
        idx = np.random.default_rng(seed).choice(len(draws), size=int(n), replace=False)
        return draws[np.sort(idx)]

    def support_mass(self, x: np.ndarray, n: int = 10_000, seed: int = 0) -> float:
        """1 by construction: MCMC only visits the prior support."""
        return 1.0

    def _is_direct(self) -> bool:
        return False

    def log_prob(self, theta, x):
        """Not available: the posterior is represented by draws."""
        raise NotImplementedError("exact_gw posteriors are represented by MCMC draws")

    def to(self, device: str) -> ExactPosterior:
        """No-op (NumPy draws)."""
        return self


# ---------------------------------------------------------------------- method
class ExactGW(InferenceMethod):
    """Exact-likelihood NUTS for the two-type Galton-Watson model.

    Args:
        chains: Independent NUTS chains (run in ``threads`` threads).
        warmup: Warmup (adaptation) iterations per chain.
        draws: Kept iterations per chain.
        max_tree_depth: NUTS tree depth cap (``2^depth`` leapfrog steps).
        target_accept: Dual-averaging target acceptance statistic.
        dense_mass: Adapt a dense (else diagonal) inverse mass matrix.
        threads: Threads for chains / initial optimisations (the Numba kernel releases
            the GIL). Each thread uses one core.
        init_starts: Prior draws optimised (L-BFGS on the log posterior) to initialise
            the chains (Laplace draws around the best optima, see ``init_tolerance``).
        init_maxiter: L-BFGS iterations per start.
        init_tolerance: Optima more than this many nats below the best are not used as
            chain starts (negligible posterior mass); all optima are reported.
        grid: FFT grid options (``factor``, ``margin``, ``min_size``, ``damping``,
            ``max_points``); see :func:`~bio_sbi.inference.gw_exact.grid_spec`.
        floor: Probability floor of the likelihood.
        min_detection: Floor of the detection probability in the conditioning.
        kernel: Likelihood kernel, ``loop`` (forward-mode derivatives; default) or
            ``adjoint`` (reverse-mode gradient, same values up to round-off, faster).
        sbc: Overrides of ``chains``/``warmup``/``draws``/``init_starts`` for each SBC case.
        optimise: Point-estimation settings (:mod:`~bio_sbi.inference.gw_optimise`: MLE or
            physical-probability MAP by PSO + L-BFGS-B). With ``enabled: true``,
            :meth:`fit_data` also computes the point estimate after (and independently
            of) NUTS; the pipeline writes it to ``optimisation.json``. Off by default.
        log_every: Log chain progress every this many iterations (0 = silent).
        warm_start: Previous ``exact_gw`` run (directory, ``posterior.pt`` or
            ``posterior_samples.csv.gz``) of the same model on a dataset this one extends
            (e.g. nested datasets of increasing ``n``); ``None`` (default) = cold start.
            The reference fit then skips the L-BFGS initialisation and starts the chains
            at well-separated previous draws (k-means clusters in whitened ``u``), with
            the previous within-chain covariance of ``u`` times ``n_prev / n`` as initial
            (and prior) inverse mass matrix, the previous median step size, and a short
            warmup (``warm_start_options``, :data:`WARM_START_DEFAULTS`). SBC cases are
            not warm-started.
        warm_start_options: Overrides of :data:`WARM_START_DEFAULTS`.
        stop: Adaptive run length (:data:`STOP_DEFAULTS`; ``enabled: false`` by
            default = ``draws`` per chain). When enabled the reference fit draws
            ``block`` iterations per chain at a time (chains in lockstep) and stops once
            the rank-normalised split R-hat, bulk / tail ESS (stick coordinates and free
            physical probabilities; R-hat also on the log density) and the spread of the
            chain means meet the targets, or at ``max_draws`` per chain (reported in
            ``stop.reason``). ``draws`` is then unused.
        reparam: NUTS coordinates (:mod:`~bio_sbi.inference.gw_reparam`): ``none``
            (default, ``u = logit(theta)``), ``probit`` (``stick = Phi(v)``), ``brow`` (B's
            row broken as to_A, persist, death, divisions) or ``brow+probit``. Exact
            reparametrisations: same prior and posterior; draws are returned in ``theta``.
        stick_order: ``{row: [entries]}`` stick-order overrides (with any ``reparam``).
        smc: Tempered SMC from the prior (:data:`SMC_DEFAULTS`, ``enabled: false`` by
            default; reference fit only, not with ``warm_start``). ``mode: start``: the
            ``beta = 1`` population (likelihood-only evaluations) replaces the L-BFGS
            initialisation: NUTS chains start at population members (``select:
            resample`` = in proportion to posterior mass, ``kmeans`` = spread), with the
            population covariance as initial / prior inverse mass matrix and a short
            warmup. ``mode: final``: the SMC final-stage chains (adaptive RW + independence
            MH run to nested R-hat / ESS targets, as ``exact_bh``) are the draws, no NUTS.
            The population and the estimated evidence are in ``diagnostics["smc"]``.
        device, seed, rounds, **_: Accepted for interface compatibility (``rounds`` must
            be 1; neural-network settings are ignored).
    """

    name = "exact_gw"
    exact = True

    def __init__(
        self,
        chains: int = 4,
        warmup: int = 300,
        draws: int = 500,
        max_tree_depth: int = 8,
        target_accept: float = 0.8,
        dense_mass: bool = True,
        threads: int = 4,
        init_starts: int = 8,
        init_maxiter: int = 200,
        init_tolerance: float = 20.0,
        grid: dict | None = None,
        floor: float = 1e-15,
        min_detection: float = 1e-6,
        kernel: str = "loop",
        sbc: dict | None = None,
        optimise: dict | None = None,
        log_every: int = 50,
        warm_start: str | None = None,
        warm_start_options: dict | None = None,
        stop: dict | None = None,
        reparam: str = "none",
        stick_order: dict | None = None,
        smc: dict | None = None,
        device: str = "cpu",
        seed: int = 0,
        rounds: int = 1,
        **_: Any,
    ) -> None:
        if rounds != 1:
            raise ValueError("exact_gw has no sequential rounds (rounds must be 1)")
        super().__init__(rounds=1, device=device, seed=seed)
        self.chains, self.warmup, self.draws = int(chains), int(warmup), int(draws)
        self.max_tree_depth = int(max_tree_depth)
        self.target_accept = float(target_accept)
        self.dense_mass = bool(dense_mass)
        self.threads = max(int(threads), 1)
        self.init_starts = max(int(init_starts), int(chains))
        self.init_maxiter = int(init_maxiter)
        self.init_tolerance = float(init_tolerance)
        self.grid = dict(grid or {})
        self.floor = float(floor)
        self.min_detection = float(min_detection)
        self.kernel = str(kernel)
        self.sbc = dict(sbc or {})
        self.optimise = dict(optimise or {})
        self.log_every = int(log_every)
        self.warm_start = None if warm_start in (None, "", "null") else str(warm_start)
        self.warm_start_options = _options(warm_start_options, WARM_START_DEFAULTS,
                                           "warm_start_options")  # fmt: skip
        self.stop = _options(stop, STOP_DEFAULTS, "stop")
        self.last_smc = None
        if self.warm_start_options["select"] not in ("kmeans", "chains"):
            raise ValueError("warm_start_options.select must be kmeans or chains")
        parse_spec(reparam)  # raises on unknown specs
        self.reparam = str(reparam or "none")
        self.stick_order = {str(k): list(v) for k, v in (stick_order or {}).items()}
        self.smc = _options(smc, SMC_DEFAULTS, "smc")
        if self.smc["mode"] not in ("start", "final"):
            raise ValueError("inference.method.smc.mode must be start or final")
        if self.smc["select"] not in ("resample", "kmeans"):
            raise ValueError("inference.method.smc.select must be resample or kmeans")

    # The sbi training hooks do not apply.
    def make_trainer(self, prior, tracker):
        """Not applicable."""
        raise NotImplementedError("exact_gw does not train a network")

    def fit(self, *args, **kwargs):
        """Not applicable: see :meth:`fit_data`."""
        raise TypeError("exact_gw conditions on raw counts: use fit_data(...)")

    # ---------------------------------------------------------------- core
    def likelihood(self, simulator, observation, cohorts, max_points=None) -> ExactLikelihood:
        """Exact likelihood of one dataset (validates the model)."""
        grid = dict(self.grid)
        if max_points is not None:
            grid["max_points"] = max_points
        return ExactLikelihood(
            simulator, observation, cohorts, grid, self.floor, self.min_detection, self.kernel
        )

    def coordinates(self, lik: ExactLikelihood) -> StickCoordinates:
        """NUTS coordinates of ``lik`` (``reparam`` / ``stick_order``)."""
        return StickCoordinates(lik, self.reparam, self.stick_order)

    def _initial_points(self, lik: ExactLikelihood, n_starts: int, n_keep: int, key,
                        coords: StickCoordinates | None = None):  # fmt: skip
        # prior draws of the sticks (the same Betas as ParameterSpace.prior), drawn with
        # NumPy so concurrent SBC cases do not share torch's global RNG
        coords = coords or self.coordinates(lik)
        rng = rng_for(self.seed, "exact-init", *key)
        theta = rng.beta(lik.prior_a, lik.prior_b, size=(n_starts, lik.dim))
        theta = np.clip(theta, 1e-6, 1 - 1e-6)
        if coords.identity:
            u0 = np.clip(np.log(theta) - np.log1p(-theta), -8, 8)
        else:  # the same prior draws, sticks (new order) within [expit(-8), expit(8)]
            u0 = coords.from_theta(theta, stick_clip=float(expit(-8)))
        bound = U_BOUND if coords.identity else coords.bound

        def neg(u):
            lp, g = coords.log_posterior(u)
            if not np.isfinite(lp):
                return 1e300, np.zeros_like(u)
            return -lp, -g

        def optimise(u):
            res = minimize(neg, u, jac=True, method="L-BFGS-B",
                           bounds=[(-bound, bound)] * len(u),
                           options={"maxiter": self.init_maxiter})  # fmt: skip
            return res.x, -float(res.fun)

        with ThreadPoolExecutor(min(self.threads, n_starts)) as pool:
            results = list(pool.map(optimise, u0))
        order = np.argsort([-lp for _, lp in results])
        best = results[order[0]][1]
        # Optima far below the best carry negligible posterior mass (a mode 20 nats down
        # has relative weight ~e^-20): chains start around the good ones only, cycling.
        good = [results[i] for i in order if results[i][1] >= best - self.init_tolerance]
        kept = [good[c % len(good)] for c in range(n_keep)]
        return [u for u, _ in kept], [lp for _, lp in results]

    @staticmethod
    def laplace_covariance(lik: ExactLikelihood, u: np.ndarray, h: float = 1e-4,
                           max_var: float = 4.0, logp=None) -> np.ndarray:  # fmt: skip
        """Inverse Hessian of ``-log posterior(u)`` (finite differences of the gradient).

        Eigenvalues are floored at ``1 / max_var`` so flat or non-convex directions get
        a prior-scale variance instead of an infinite one. ``logp``: log density and
        gradient to use (default ``lik.log_posterior_u``).
        """
        logp = logp or lik.log_posterior_u
        d = len(u)
        H = np.zeros((d, d))
        for i in range(d):
            e = np.zeros(d)
            e[i] = h
            H[i] = -(logp(u + e)[1] - logp(u - e)[1]) / (2 * h)
        H = 0.5 * (H + H.T)
        if not np.all(np.isfinite(H)):
            return np.eye(d)
        w, V = np.linalg.eigh(H)
        w = np.maximum(w, 1.0 / max_var)
        return (V / w) @ V.T

    def _warm_plan(self, lik: ExactLikelihood, prev: PreviousPosterior, chains: int, key,
                   coords: StickCoordinates | None = None, options: dict | None = None):  # fmt: skip
        """Chain starts, inverse mass matrix, step size and warmup schedule from ``prev``.

        ``options`` replaces :attr:`warm_start_options` (SMC start: ``select: resample``
        takes the starts at systematic strata of a random order of ``prev``, i.e. in
        proportion to mass).
        """
        o = options or self.warm_start_options
        coords = coords or self.coordinates(lik)
        u = coords.from_theta(prev.theta)
        d = lik.dim
        if u.shape[1] != d:
            raise ValueError(f"warm start: previous draws have {u.shape[1]} coordinates, not {d}")
        rng = rng_for(self.seed, "exact-warm", *key)
        n_new = int(sum(h.clones for h in lik.data))
        n_prev = o["n_prev"] if o["n_prev"] is not None else prev.clones
        if o["scale"] is not None:
            scale = float(o["scale"])
        elif n_prev:
            scale = float(n_prev) / n_new
        else:
            raise ValueError("warm start: the previous run's clone count is unknown; set "
                             "inference.method.warm_start_options.n_prev or .scale")  # fmt: skip
        cov = _within_covariance(u, prev.chain) * scale
        cov = 0.5 * (cov + cov.T) + 1e-10 * np.eye(d)
        use_chains = o["select"] == "chains" and prev.chain is not None
        if use_chains:
            ids = np.unique(prev.chain)
            cands = [rng.permutation(np.flatnonzero(prev.chain == ids[c % len(ids)]))
                     for c in range(chains)]  # fmt: skip
        elif o["select"] == "resample":  # systematic: start c in the c-th stratum
            order = rng.permutation(len(u))
            pos = ((np.arange(chains) + rng.uniform()) / chains * len(u)).astype(int)
            cands = [np.roll(order, -p) for p in pos]
        else:
            cands = spread_starts(u, chains, rng)
        starts, picked = [], []
        for c in range(chains):
            for i in cands[c % len(cands)][:200]:
                if np.isfinite(coords.log_posterior(u[i])[0]):
                    starts.append(u[i].copy())
                    picked.append(int(i))
                    break
            else:
                raise ValueError("warm start: no previous draw has finite density on this data")
        step = None
        if o["step_size"] == "previous" and prev.step_size:
            step = float(np.median(prev.step_size))
        warmup = int(o["warmup"])
        init, term = int(o["init_buffer"]), int(o["term_buffer"])
        end = warmup - term
        adapt = {
            "window_ends": [end] if end - max(init, 0) >= 3 else [],
            "collect_from": init,
            "mass_prior": cov if self.dense_mass else np.diag(cov),
            "mass_prior_weight": float(o["mass_prior"]),
        }
        info = {
            "source": prev.source, "previous_draws": len(u),
            "previous_chains_known": prev.chain is not None, "n_prev": n_prev, "n": n_new,
            "scale": scale,
            "select": "chains" if use_chains else ("resample" if o["select"] == "resample"
                                                   else "kmeans"),
            "start_indices": picked,
            "start_previous_chain": ([int(prev.chain[i]) for i in picked]
                                     if prev.chain is not None else None),
            "cluster_sizes": None if use_chains else [int(len(c)) for c in cands],
            "initial_step_size": step, "warmup": warmup,
            "window": [init, end] if adapt["window_ends"] else None,
            "mass_prior": float(o["mass_prior"]),
        }  # fmt: skip
        return starts, cov, step, warmup, adapt, info

    def run(self, lik: ExactLikelihood, key=("reference",), chains=None, warmup=None,
            draws=None, init_starts=None, max_tree_depth=None,
            warm: PreviousPosterior | None = None,
            stop: dict | None = None, smc: dict | None = None,
            coords: StickCoordinates | None = None) -> tuple[np.ndarray, dict]:  # fmt: skip
        """Sample the posterior of ``lik``; returns ``(theta draws, diagnostics)``.

        ``warm`` (a previous posterior) replaces the L-BFGS initialisation and the warmup
        schedule (see ``warm_start``); ``stop`` (options as :data:`STOP_DEFAULTS`, used if
        ``enabled``) replaces the fixed ``draws``. Both off reproduce the plain run.
        ``smc`` (options as :data:`SMC_DEFAULTS`, used if ``enabled`` and not ``warm``):
        tempered SMC start or sampler. ``coords``: NUTS coordinates (default
        :meth:`coordinates`, i.e. ``reparam``).
        """
        depth = int(max_tree_depth or self.max_tree_depth)
        chains = int(chains or self.chains)
        warmup = int(self.warmup if warmup is None else warmup)
        draws = int(draws or self.draws)
        starts = max(int(init_starts or self.init_starts), chains)
        stop = stop if stop and stop.get("enabled") else None
        smc = smc if smc and smc.get("enabled") and warm is None else None
        coords = coords or self.coordinates(lik)
        name = "/".join(map(str, key))
        t0 = time.perf_counter()
        warm_info, check, smc_info = None, None, None
        self.last_smc = None
        if smc is not None:
            smc_res, smc_info = self._run_smc(lik, coords, smc, key, name)
            if smc["mode"] == "final":
                return self._smc_draws(lik, coords, smc_res, smc_info, t0)
            pop = PreviousPosterior(coords.theta(smc_res.draws), None, None, None, "smc")
            plan = {**WARM_START_DEFAULTS, "scale": 1.0, "step_size": "heuristic",
                    **{k: smc[k] for k in ("warmup", "init_buffer", "term_buffer",
                                           "mass_prior", "select")}}  # fmt: skip
            w_starts, w_cov, w_step, warmup, adapt, warm_info = self._warm_plan(
                lik, pop, chains, key, coords, plan
            )
            optima = []
            t_init = time.perf_counter() - t0
            LOG.info("%s: SMC start (%d likelihood evaluations, %.0fs), NUTS chains from "
                     "population members %s, warmup %d", name, smc_info["evals"], t_init,
                     warm_info["start_indices"], warmup)  # fmt: skip
        elif warm is None:
            inits, optima = self._initial_points(lik, starts, chains, key, coords)
            t_init = time.perf_counter() - t0
            LOG.info("%s: initial optimisation %.1fs, best log posteriors %s", name, t_init,
                     np.round(sorted(optima, reverse=True)[:chains], 1))  # fmt: skip
            adapt: dict = {}
        else:
            w_starts, w_cov, w_step, warmup, adapt, warm_info = self._warm_plan(
                lik, warm, chains, key, coords
            )
            optima = []
            n_check = int(self.warm_start_options["check_starts"])
            if n_check > 0:
                _, check = self._initial_points(lik, n_check, 1, key, coords)
            t_init = time.perf_counter() - t0
            LOG.info("%s: warm start from %s (scale %.3g, step %s, warmup %d), starts from "
                     "previous draws %s", name, warm_info["source"], warm_info["scale"],
                     w_step, warmup, warm_info["start_indices"])  # fmt: skip

        planned = warm is not None or smc is not None
        logp = coords.log_posterior

        def start(c):
            rng = rng_for(self.seed, "exact-nuts", *key, c)
            if not planned:
                # Start from a draw of the Laplace approximation at this chain's optimum,
                # with its covariance as initial inverse mass matrix (the posterior can be
                # very narrow and correlated: an identity mass matrix needs tiny steps).
                cov = self.laplace_covariance(lik, inits[c], logp=logp)
                q0 = inits[c] + np.linalg.cholesky(cov) @ rng.standard_normal(len(cov))
                if not np.isfinite(logp(q0)[0]):
                    q0 = inits[c]
                step = None
            else:
                q0, cov, step = w_starts[c], w_cov, w_step
            chain = NutsChain(
                logp, q0, rng, max_depth=depth,
                target_accept=self.target_accept, dense=self.dense_mass,
                inv_mass=cov if self.dense_mass else np.diag(cov), step_size=step,
                name=f"{name} chain {c}", log_every=self.log_every,
            )  # fmt: skip
            chain.adapt(warmup, **adapt)
            return chain

        def full(c):
            chain = start(c)
            chain.draw(draws, total=draws)
            return chain

        workers = min(self.threads, chains)
        t_chains = time.perf_counter()
        stop_info = None
        with ThreadPoolExecutor(workers) as pool:
            if stop is None:
                objs = list(pool.map(full, range(chains)))
            else:
                objs = list(pool.map(start, range(chains)))
                stop_info = self._draw_until(objs, pool, workers, lik, stop, name, coords)
        chain_wall = time.perf_counter() - t_chains
        res = [o.result() for o in objs]
        draws = len(res[0].draws)
        u = np.stack([r.draws for r in res])  # (chains, draws, d)
        theta = coords.theta(u)
        physical = lik.space.decode(theta)
        free = [i for i, n in enumerate(lik.space.names) if n in set(lik.space.free_names)]
        rhat = {n: split_rhat(theta[..., i]) for i, n in enumerate(lik.space.theta_names)}
        rhat.update({lik.space.names[i]: split_rhat(physical[..., i]) for i in free})
        n_ess = {n: ess(theta[..., i]) for i, n in enumerate(lik.space.theta_names)}
        n_ess.update({lik.space.names[i]: ess(physical[..., i]) for i in free})
        final = self._convergence(lik, res, free, monitor_logp=True, coords=coords)
        busy = float(sum(r.seconds for r in res))
        warm_evals = int(sum(r.warmup_gradient_evals for r in res))
        total_evals = int(sum(r.gradient_evals for r in res))
        diag = {
            "sampler": "nuts",
            "chains": chains,
            "warmup": warmup,
            "draws_per_chain": draws,
            "max_rhat": float(np.nanmax(list(rhat.values()))),
            "min_ess": float(np.nanmin(list(n_ess.values()))),
            "rhat": rhat,
            # per-chain posterior means: chains stuck in different modes (R-hat >> 1)
            # show which parameters separate the modes
            "chain_means": {
                lik.space.names[i]: physical[..., i].mean(1).round(4).tolist() for i in free
            },
            "ess": n_ess,
            # rank-normalised split R-hat (bulk / folded; also on the log density), bulk and
            # tail ESS, spread of the chain means (rank-normalised, posterior sd units)
            "max_rank_rhat": final["max_rhat"],
            "min_ess_bulk": final["min_ess_bulk"],
            "min_ess_tail": final["min_ess_tail"],
            "max_mean_spread": final["max_mean_spread"],
            "rank_rhat": final["rhat"],
            "ess_bulk": final["ess_bulk"],
            "ess_tail": final["ess_tail"],
            "divergences": int(sum(r.divergences for r in res)),
            "warmup_divergences": int(sum(r.warmup_divergences for r in res)),
            "max_tree_depth_hits": int(sum(r.max_tree_depth_hits for r in res)),
            "mean_tree_depth": [r.mean_tree_depth for r in res],
            "step_size": [r.step_size for r in res],
            "accept_stat": [r.accept_stat for r in res],
            "chain_seconds": [r.seconds for r in res],
            "chain_warmup_seconds": [r.warmup_seconds for r in res],
            "gradient_evals": total_evals,
            "warmup_gradient_evals": warm_evals,
            "sampling_gradient_evals": total_evals - warm_evals,
            "chain_gradient_evals": [r.gradient_evals for r in res],
            # 1 - (chain busy time) / (workers x wall time of the chain phase)
            "chain_phase_seconds": chain_wall,
            "idle_fraction": float(max(0.0, 1 - busy / (workers * chain_wall)))
            if chain_wall > 0
            else 0.0,
            "init_seconds": t_init,
            "init_log_posterior": sorted(optima, reverse=True),
            "chain_mean_log_posterior": [float(r.logp.mean()) for r in res],
            "seconds": time.perf_counter() - t0,
            "likelihood_evals": lik.evals,
            "seconds_per_eval": lik.seconds_per_eval,
            "grids": [
                {
                    "generations": h.generations,
                    "Lx": h.grid.Lx,
                    "Ly": h.grid.Ly,
                    "clones": h.clones,
                    "cells": len(h.a),
                }
                for h in lik.data
            ],  # fmt: skip
        }
        if warm_info is not None:
            if check is not None:
                best_sampled = float(max(r.logp.max() for r in res))
                tol = float(self.warm_start_options["check_tolerance"])
                warm_info["check_optima"] = sorted(check, reverse=True)
                warm_info["max_sampled_log_posterior"] = best_sampled
                warm_info["unvisited_optimum"] = bool(max(check) > best_sampled + tol)
                if warm_info["unvisited_optimum"]:
                    LOG.warning("%s: an L-BFGS optimum (log posterior %.1f) lies above every "
                                "warm-started draw (max %.1f): a mode the previous posterior "
                                "did not have? Re-run cold.", name, max(check), best_sampled)  # fmt: skip
            diag["warm_start"] = warm_info
        if stop_info is not None:
            diag["stop"] = stop_info
        if not coords.identity:
            diag["reparam"] = coords.info()
        if smc_info is not None:
            diag["smc"] = smc_info
            diag["smc_start"] = warm_info
            diag["likelihood_evals_smc"] = smc_info["evals"]
            diag.pop("warm_start", None)
        return theta.reshape(-1, theta.shape[-1]), diag

    # ---------------------------------------------------------------- SMC
    def _run_smc(self, lik: ExactLikelihood, coords: StickCoordinates, o: dict, key,
                 name: str):  # fmt: skip
        """Tempered SMC from the prior in the NUTS coordinates (likelihood-only evaluations).

        Returns ``(SMCResult, info)``; :attr:`last_smc` keeps the result.
        """
        rng = rng_for(self.seed, "exact-smc", *key)
        final = o["mode"] == "final"
        common = {k: o[k] for k in ("chains", "steps", "ess_fraction", "rw_acceptance",
                                    "imh_prob", "imh_components", "df", "de_prob",
                                    "rw_scale_sd", "local_k")}  # fmt: skip
        common["max_evals"] = int(o["max_evals"])
        if final:
            common.update({k: o[k] for k in ("target_ess", "max_rhat", "tail_fraction",
                                             "superchains", "min_final_steps",
                                             "max_final_steps", "final_burn",
                                             "check_every")})  # fmt: skip
        else:  # population at beta = 1 after final_steps MH steps; no stopping rule
            n = max(int(o["final_steps"]), 1)
            common.update({"target_ess": 0.0, "max_rhat": float("inf"), "tail_fraction": 0.0,
                           "min_final_steps": n, "max_final_steps": n, "check_every": n,
                           "final_burn": 0.0, "superchains": int(o["superchains"])})  # fmt: skip
        with ThreadPoolExecutor(self.threads) as pool:

            def logpost(V):
                return np.fromiter(pool.map(coords.log_posterior_value, np.asarray(V)),
                                   float, len(V))  # fmt: skip

            res = tempered_smc(
                logpost, coords.sample_prior, lambda V: coords.log_prior(V, normalised=True),
                rng, monitor=coords.theta, name=f"{name} smc", **common,
            )  # fmt: skip
        self.last_smc = res
        info = {k: v for k, v in res.info.items() if k not in ("rhat", "ess_bulk", "ess_tail",
                                                                 "nested_rhat")}  # fmt: skip
        info["mode"] = o["mode"]
        LOG.info("%s: SMC %s after %d stages, %d likelihood evaluations, %.0fs, log evidence "
                 "%.2f", name, info["status"], info["stages"], info["evals"], info["seconds"],
                 info["log_evidence"])  # fmt: skip
        if not res.info["reached_posterior"]:
            raise RuntimeError(
                f"{name}: SMC did not reach the posterior within "
                f"smc.max_evals={o['max_evals']} (beta {res.info['beta']:.3g})"
            )
        return res, info

    def _smc_draws(self, lik: ExactLikelihood, coords: StickCoordinates, res, info: dict,
                   t0: float) -> tuple[np.ndarray, dict]:  # fmt: skip
        """Draws and diagnostics of ``smc.mode=final`` (final-stage chains, equal weights)."""
        u = res.chains  # (chains, length, d)
        theta = coords.theta(u)
        physical = lik.space.decode(theta)
        free = [i for i, n in enumerate(lik.space.names) if n in set(lik.space.free_names)]
        x, names, ess_names = self._monitored(lik, u, res.log_post, free, coords)
        final = convergence_check(x, names, ess_names)
        diag = {
            "sampler": "smc", "chains": int(u.shape[0]), "warmup": 0,
            "draws_per_chain": int(u.shape[1]),
            "max_rhat": final["max_rhat"], "min_ess": final["min_ess_bulk"],
            "max_rank_rhat": final["max_rhat"], "min_ess_bulk": final["min_ess_bulk"],
            "min_ess_tail": final["min_ess_tail"], "max_mean_spread": final["max_mean_spread"],
            "rank_rhat": final["rhat"], "ess_bulk": final["ess_bulk"],
            "ess_tail": final["ess_tail"],
            "max_nested_rhat": res.info.get("max_nested_rhat"),
            "chain_means": {lik.space.names[i]: physical[..., i].mean(1).round(4).tolist()
                            for i in free},
            "divergences": 0, "gradient_evals": 0,
            "likelihood_evals": lik.evals, "likelihood_evals_smc": info["evals"],
            "seconds": time.perf_counter() - t0, "seconds_per_eval": lik.seconds_per_eval,
            "smc": info,
            "grids": [{"generations": h.generations, "Lx": h.grid.Lx, "Ly": h.grid.Ly,
                       "clones": h.clones, "cells": len(h.a)} for h in lik.data],
        }  # fmt: skip
        if not coords.identity:
            diag["reparam"] = coords.info()
        return theta.reshape(-1, theta.shape[-1]), diag

    @staticmethod
    def _monitored(lik: ExactLikelihood, u: np.ndarray, logp: np.ndarray | None,
                   free: list[int], coords: StickCoordinates | None = None
                   ) -> tuple[np.ndarray, list[str], set]:  # fmt: skip
        """``(chains, n, q)`` sticks, free physical probabilities (and log density)."""
        theta = expit(u) if coords is None else coords.theta(u)
        physical = lik.space.decode(theta)[..., free]
        names = list(lik.space.theta_names) + [lik.space.names[i] for i in free]
        ess_names = set(names)
        parts = [theta, physical]
        if logp is not None:
            parts.append(logp[..., None])
            names.append("lp__")
        return np.concatenate(parts, axis=-1), names, ess_names

    def _convergence(self, lik, res, free, monitor_logp=True, coords=None) -> dict:
        u = np.stack([r.draws for r in res])
        if u.shape[1] < 4:
            nan = float("nan")
            return {"rhat": {}, "ess_bulk": {}, "ess_tail": {}, "mean_spread": {},
                    "max_rhat": nan, "min_ess_bulk": nan, "min_ess_tail": nan,
                    "max_mean_spread": nan}  # fmt: skip
        logp = np.stack([r.logp for r in res]) if monitor_logp else None
        x, names, ess_names = self._monitored(lik, u, logp, free, coords)
        return convergence_check(x, names, ess_names)

    def _draw_until(self, objs: list[NutsChain], pool, workers: int, lik: ExactLikelihood,
                    stop: dict, name: str, coords=None) -> dict:  # fmt: skip
        """Draw ``block`` iterations per chain at a time until the targets are met."""
        block = max(int(stop["block"]), 1)
        min_d, max_d = int(stop["min_draws"]), max(int(stop["max_draws"]), 1)
        confirm = max(int(stop["confirm"]), 1)
        free = [i for i, n in enumerate(lik.space.names) if n in set(lik.space.free_names)]
        t0 = time.perf_counter()
        trajectory, passes, reason, idle = [], 0, None, 0.0
        failed: list[str] = []
        n_now = objs[0].n_draws
        while n_now < max_d:
            b = min(block, max_d - n_now)
            busy0 = [o.seconds for o in objs]
            tb = time.perf_counter()
            list(pool.map(lambda o, b=b: o.draw(b), objs))
            wall = time.perf_counter() - tb
            per = [o.seconds - s0 for o, s0 in zip(objs, busy0, strict=True)]
            idle += workers * wall - sum(per)
            n_now = objs[0].n_draws
            entry = {"draws_per_chain": n_now, "seconds": time.perf_counter() - t0,
                     "gradient_evals": int(sum(o.s.evals for o in objs)),
                     "block_seconds": per}  # fmt: skip
            if n_now >= min_d:
                c = self._convergence(lik, [o.result() for o in objs], free,
                                      bool(stop["monitor_logp"]), coords)  # fmt: skip
                failed = []
                if not c["max_rhat"] <= float(stop["rhat"]):
                    failed.append("rhat")
                if not c["min_ess_bulk"] >= float(stop["ess"]):
                    failed.append("ess")
                if not c["min_ess_tail"] >= float(stop["tail_ess"]):
                    failed.append("tail_ess")
                if not c["max_mean_spread"] <= float(stop["max_mean_spread"]):
                    failed.append("mean_spread")
                entry.update({k: c[k] for k in (
                    "max_rhat", "max_rhat_at", "min_ess_bulk", "min_ess_bulk_at",
                    "min_ess_tail", "min_ess_tail_at", "max_mean_spread",
                    "max_mean_spread_at")})  # fmt: skip
                entry["failed"] = list(failed)
                passes = passes + 1 if not failed else 0
                LOG.info("%s: %d draws/chain: rank R-hat %.4f (%s), ESS bulk %.0f, tail %.0f, "
                         "chain-mean spread %.2f%s", name, n_now, c["max_rhat"],
                         c["max_rhat_at"], c["min_ess_bulk"], c["min_ess_tail"],
                         c["max_mean_spread"],
                         "" if not failed else f" (not met: {', '.join(failed)})")  # fmt: skip
            trajectory.append(entry)
            if n_now >= min_d and passes >= confirm:
                reason = "targets_met"
                break
        if reason is None:
            reason = "max_draws"
            LOG.warning("%s: stopped at max_draws=%d per chain without meeting the targets "
                        "(not met: %s)", name, max_d, ", ".join(failed) or "not checked")  # fmt: skip
        return {
            "settings": dict(stop), "reason": reason, "met": reason == "targets_met",
            "failed": [] if reason == "targets_met" else failed,
            "draws_per_chain": n_now, "seconds": time.perf_counter() - t0,
            # sum over blocks of (workers x block wall time - chain busy time): time lost
            # waiting for the slowest chain of each block
            "lockstep_idle_seconds": idle, "trajectory": trajectory,
        }  # fmt: skip

    def point_estimate(self, lik: ExactLikelihood, key=("reference",), **overrides) -> dict:
        """MLE / physical MAP of ``lik`` (``optimise`` settings updated by ``overrides``).

        Workers default to ``threads``. See :func:`~bio_sbi.inference.gw_optimise.optimise`.
        """
        from .gw_optimise import optimise

        opts = {**self.optimise, **overrides}
        opts.pop("enabled", None)
        if opts.get("workers") is None:
            opts["workers"] = self.threads
        return optimise(lik, seed=self.seed, key=key, **opts)

    # ---------------------------------------------------------------- pipeline API
    def fit_data(self, simulator, observation, cohorts, x_dim: int) -> ExactPosterior:
        """Sample the exact posterior of the reference dataset ``cohorts``."""
        lik = self.likelihood(simulator, observation, cohorts)
        LOG.info("exact_gw grids: %s", [(g.Lx, g.Ly) for g in lik.grids])
        warm = None
        if self.warm_start is not None:
            warm = load_previous_posterior(self.warm_start, lik.space)
        draws, diag = self.run(lik, warm=warm, stop=self.stop, smc=self.smc)
        diag["grid_check"] = lik.grid_check(np.median(draws, axis=0))
        for g in diag["grid_check"]:
            if g["max_rel_error"] > 1e-5:  # tail mass beyond L is damped, not an error
                LOG.warning("exact_gw grid check (generation %d): tail mass %.2g, max relative "
                            "error %.2g on observed cells; raise inference.method.grid.factor",
                            g["generations"], g["tail_mass"], g["max_rel_error"])  # fmt: skip
        LOG.info("exact_gw: %d draws in %.1fs (%.3fs/eval), max R-hat %.3f, min ESS %.0f, "
                 "%d divergences", len(draws), diag["seconds"], diag["seconds_per_eval"],
                 diag["max_rhat"], diag["min_ess"], diag["divergences"])  # fmt: skip
        if diag["max_rhat"] > 1.05 or diag["divergences"] > 0.01 * len(draws):
            LOG.warning("exact_gw: chains may not have converged (max R-hat %.3f, %d "
                        "divergences); increase inference.method.warmup/draws",
                        diag["max_rhat"], diag["divergences"])  # fmt: skip
        history = [{
            "round": 0, "simulations": 0, "valid_simulations": 0, "simulation_seconds": 0.0,
            "training_seconds": diag["seconds"], "epochs": 0,
            "best_validation_loss": float("nan"),
        }]  # fmt: skip
        optimisation = None
        if self.optimise.get("enabled"):
            optimisation = self.point_estimate(lik)
            LOG.info("exact_gw point estimate (%s): %.4f", optimisation["objective"],
                     optimisation["value"])  # fmt: skip
        smc_population = None
        if diag.get("smc") is not None and self.smc["keep_population"] and self.last_smc:
            smc_population = self.coordinates(lik).theta(self.last_smc.draws)
        return ExactPosterior(
            posterior=None, feature_mask=np.ones(int(x_dim), dtype=bool), method=self.name,
            amortized=False, history=history, draws=draws, diagnostics=diag,
            optimisation=optimisation, smc_population=smc_population,
        )  # fmt: skip

    def calibrate(self, simulator, observation, theta: np.ndarray, datasets: list,
                  posterior_samples: int) -> tuple[np.ndarray, dict]:  # fmt: skip
        """SBC ranks: rerun MCMC (``sbc`` settings) on every valid calibration dataset.

        Cases run ``threads // sbc.chains`` at a time. Cases whose counts need a grid
        above ``sbc.max_points`` are skipped: selecting cases on the data keeps SBC ranks
        uniform, and prior draws can produce huge clones whose grids would cost hours.
        Returns ``(ranks, info)``; ranks count draws below the truth among
        ``min(posterior_samples, chains * draws)`` draws per case.
        """
        opts = {"chains": 2, "warmup": 100, "draws": 100, "init_starts": 2, "max_tree_depth": 6,
                **self.sbc}  # fmt: skip
        max_points = float(opts.pop("max_points", 2.5e5))
        info = {"invalid_cases": 0, "excluded_grid": 0, "seconds": [], "max_rhat": [],
                "settings": {**opts, "max_points": max_points}}  # fmt: skip
        jobs = []
        for i, (t, data) in enumerate(zip(theta, datasets, strict=True)):
            if data is None:
                info["invalid_cases"] += 1
                continue
            try:
                jobs.append((i, t, self.likelihood(simulator, observation, data, max_points)))
            except GridTooLarge:
                info["excluded_grid"] += 1
        n_draws = int(min(posterior_samples, opts["chains"] * opts["draws"]))

        def case(job):
            i, t, lik = job
            draws, diag = self.run(lik, key=("sbc", i), **opts)
            rng = rng_for(self.seed, "exact-sbc-thin", i)
            if len(draws) > n_draws:
                draws = draws[np.sort(rng.choice(len(draws), n_draws, replace=False))]
            LOG.info("SBC case %d: %.1fs, max R-hat %.3f", i, diag["seconds"], diag["max_rhat"])
            return (draws < t[None, :]).sum(axis=0), diag

        parallel = max(1, self.threads // max(int(opts["chains"]), 1))
        with ThreadPoolExecutor(parallel) as pool:
            results = list(pool.map(case, jobs))
        info["seconds"] = [d["seconds"] for _, d in results]
        info["max_rhat"] = [d["max_rhat"] for _, d in results]
        info["draws_per_case"] = n_draws
        ranks = np.asarray([r for r, _ in results], dtype=float)
        return ranks.reshape(-1, theta.shape[1]), info
