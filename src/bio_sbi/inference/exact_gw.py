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
(e.g. single A founder), homogeneous law, two measured labels with any capture and 2x2
confusion (pgf substitution ``x -> 1 - c + c (C x + ...)``), ``selection`` detected or
all. Anything else raises ``ValueError``.

The fitted object is an :class:`ExactPosterior`: a :class:`FittedPosterior` holding the
MCMC draws for the reference dataset (not amortised). The pipeline runs SBC by
re-running MCMC on each calibration dataset (:meth:`ExactGW.calibrate`).
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from ..rng import rng_for
from .base import FittedPosterior, InferenceMethod
from .gw_exact import GridSpec, GWModel, HorizonData, grid_spec, histogram, law_and_jac
from .gw_exact import log_likelihood as _log_likelihood
from .nuts import ess, run_chain, split_rhat

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
    for row in space.rows:
        if row.prefix in ("founder", "founder_size") and row.n_sticks:
            raise ValueError(
                f"inference=exact_gw needs a fixed {row.prefix} row (e.g. law.fixed.founder.A=1);"
                " free founder parameters are not supported"
            )
    if observation.n_labels != 2 or tuple(observation.types) != tuple(simulator.types):
        raise ValueError("inference=exact_gw needs two measured labels for the two types")


# ---------------------------------------------------------------------- likelihood
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
    """

    def __init__(self, simulator, observation, cohorts, grid: dict | None = None,
                 floor: float = 1e-15, min_detection: float = 1e-6) -> None:  # fmt: skip
        check_supported(simulator, observation)
        self.sim = simulator
        self.space = simulator.space
        self.floor = float(floor)
        self.min_detection = float(min_detection)
        law = simulator.law
        theta0 = np.full(self.space.theta_dim, 0.5)  # founder rows are fixed: any theta
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
        # theta coordinate -> (type row, free entries, stick index)
        self.directions = []
        self.prior_a, self.prior_b = [], []
        for r in self.space.rows:
            if not r.n_sticks:
                continue
            k = law.types.index(r.prefix)  # founder rows have no sticks (checked)
            for j in range(r.n_sticks):
                self.directions.append((k, r, j))
            tail = np.cumsum(r.alpha[::-1])[::-1]
            self.prior_a.extend(r.alpha[:-1])
            self.prior_b.extend(tail[1:])
        self.prior_a = np.asarray(self.prior_a, dtype=float)
        self.prior_b = np.asarray(self.prior_b, dtype=float)
        self.dim = len(self.directions)
        self.n_outcomes_max = law.offspring.shape[1]
        self._lock = threading.Lock()
        self.evals, self.eval_seconds = 0, 0.0

    # ---------------------------------------------------------------- parameters
    @property
    def grids(self) -> list[GridSpec]:
        """FFT grid of every horizon."""
        return [h.grid for h in self.data]

    def jacobian(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``P (2, O)``, ``dP (d, O)`` (row derivative per coordinate) and its row index."""
        theta = np.asarray(theta, dtype=float)
        P, _, _ = self.sim.law.unpack(self.space, self.space.decode(theta))
        dP = np.zeros((self.dim, self.n_outcomes_max))
        rows = np.zeros(self.dim, dtype=np.int64)
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

    def log_likelihood(self, theta: np.ndarray, grad: bool = True) -> tuple[float, np.ndarray]:
        """Log-likelihood and its gradient with respect to ``theta``."""
        t0 = time.perf_counter()
        P, dP, rows = self.jacobian(theta)
        if not grad:
            dP, rows = dP[:0], rows[:0]
        ll, g = _log_likelihood(self.model, P, dP, rows, self.data, self.floor, self.min_detection)
        with self._lock:
            self.evals += 1
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
        out = []
        for h in self.data:
            g = h.grid
            p, _ = law_and_jac(self.model, P, dP[:0], rows[:0], h.generations, g)
            big = GridSpec(2 * g.Lx, 2 * g.Ly, 1.0, 1.0)
            pb, _ = law_and_jac(self.model, P, dP[:0], rows[:0], h.generations, big)
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
        sbc: Overrides of ``chains``/``warmup``/``draws``/``init_starts`` for each SBC case.
        log_every: Log chain progress every this many iterations (0 = silent).
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
        sbc: dict | None = None,
        log_every: int = 50,
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
        self.sbc = dict(sbc or {})
        self.log_every = int(log_every)

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
            simulator, observation, cohorts, grid, self.floor, self.min_detection
        )

    def _initial_points(self, lik: ExactLikelihood, n_starts: int, n_keep: int, key):
        # prior draws of the sticks (the same Betas as ParameterSpace.prior), drawn with
        # NumPy so concurrent SBC cases do not share torch's global RNG
        rng = rng_for(self.seed, "exact-init", *key)
        theta = rng.beta(lik.prior_a, lik.prior_b, size=(n_starts, lik.dim))
        theta = np.clip(theta, 1e-6, 1 - 1e-6)
        u0 = np.clip(np.log(theta) - np.log1p(-theta), -8, 8)

        def neg(u):
            lp, g = lik.log_posterior_u(u)
            if not np.isfinite(lp):
                return 1e300, np.zeros_like(u)
            return -lp, -g

        def optimise(u):
            res = minimize(neg, u, jac=True, method="L-BFGS-B",
                           bounds=[(-U_BOUND, U_BOUND)] * len(u),
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
                           max_var: float = 4.0) -> np.ndarray:  # fmt: skip
        """Inverse Hessian of ``-log posterior(u)`` (finite differences of the gradient).

        Eigenvalues are floored at ``1 / max_var`` so flat or non-convex directions get
        a prior-scale variance instead of an infinite one.
        """
        d = len(u)
        H = np.zeros((d, d))
        for i in range(d):
            e = np.zeros(d)
            e[i] = h
            H[i] = -(lik.log_posterior_u(u + e)[1] - lik.log_posterior_u(u - e)[1]) / (2 * h)
        H = 0.5 * (H + H.T)
        if not np.all(np.isfinite(H)):
            return np.eye(d)
        w, V = np.linalg.eigh(H)
        w = np.maximum(w, 1.0 / max_var)
        return (V / w) @ V.T

    def run(self, lik: ExactLikelihood, key=("reference",), chains=None, warmup=None,
            draws=None, init_starts=None) -> tuple[np.ndarray, dict]:  # fmt: skip
        """Sample the posterior of ``lik``; returns ``(theta draws, diagnostics)``."""
        chains = int(chains or self.chains)
        warmup = int(self.warmup if warmup is None else warmup)
        draws = int(draws or self.draws)
        starts = max(int(init_starts or self.init_starts), chains)
        t0 = time.perf_counter()
        inits, optima = self._initial_points(lik, starts, chains, key)
        t_init = time.perf_counter() - t0
        LOG.info(
            "%s: initial optimisation %.1fs, best log posteriors %s",
            "/".join(map(str, key)),
            t_init,
            np.round(sorted(optima, reverse=True)[:chains], 1),
        )

        def chain(c):
            # Start from a draw of the Laplace approximation at this chain's optimum, with
            # its covariance as initial inverse mass matrix (the posterior can be very
            # narrow and correlated: an identity mass matrix would need tiny steps).
            rng = rng_for(self.seed, "exact-nuts", *key, c)
            cov = self.laplace_covariance(lik, inits[c])
            q0 = inits[c] + np.linalg.cholesky(cov) @ rng.standard_normal(len(cov))
            if not np.isfinite(lik.log_posterior_u(q0)[0]):
                q0 = inits[c]
            return run_chain(
                lik.log_posterior_u, q0, warmup, draws, rng, max_depth=self.max_tree_depth,
                target_accept=self.target_accept, dense=self.dense_mass,
                inv_mass=cov if self.dense_mass else np.diag(cov),
                name=f"{'/'.join(map(str, key))} chain {c}", log_every=self.log_every,
            )  # fmt: skip

        with ThreadPoolExecutor(min(self.threads, chains)) as pool:
            res = list(pool.map(chain, range(chains)))
        u = np.stack([r.draws for r in res])  # (chains, draws, d)
        theta = expit(u)
        physical = lik.space.decode(theta)
        free = [i for i, n in enumerate(lik.space.names) if n in set(lik.space.free_names)]
        rhat = {n: split_rhat(theta[..., i]) for i, n in enumerate(lik.space.theta_names)}
        rhat.update({lik.space.names[i]: split_rhat(physical[..., i]) for i in free})
        n_ess = {n: ess(theta[..., i]) for i, n in enumerate(lik.space.theta_names)}
        n_ess.update({lik.space.names[i]: ess(physical[..., i]) for i in free})
        diag = {
            "sampler": "nuts",
            "chains": chains,
            "warmup": warmup,
            "draws_per_chain": draws,
            "max_rhat": float(np.nanmax(list(rhat.values()))),
            "min_ess": float(np.nanmin(list(n_ess.values()))),
            "rhat": rhat,
            "ess": n_ess,
            "divergences": int(sum(r.divergences for r in res)),
            "warmup_divergences": int(sum(r.warmup_divergences for r in res)),
            "max_tree_depth_hits": int(sum(r.max_tree_depth_hits for r in res)),
            "mean_tree_depth": [r.mean_tree_depth for r in res],
            "step_size": [r.step_size for r in res],
            "accept_stat": [r.accept_stat for r in res],
            "chain_seconds": [r.seconds for r in res],
            "gradient_evals": int(sum(r.gradient_evals for r in res)),
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
        return theta.reshape(-1, theta.shape[-1]), diag

    # ---------------------------------------------------------------- pipeline API
    def fit_data(self, simulator, observation, cohorts, x_dim: int) -> ExactPosterior:
        """Sample the exact posterior of the reference dataset ``cohorts``."""
        lik = self.likelihood(simulator, observation, cohorts)
        LOG.info("exact_gw grids: %s", [(g.Lx, g.Ly) for g in lik.grids])
        draws, diag = self.run(lik)
        diag["grid_check"] = lik.grid_check(np.median(draws, axis=0))
        for g in diag["grid_check"]:
            if g["tail_mass"] > 1e-6 or g["max_rel_error"] > 1e-6:
                LOG.warning("exact_gw grid check (generation %d): tail mass %.2g, max relative "
                            "error %.2g on observed cells; raise inference.method.grid.factor",
                            g["generations"], g["tail_mass"], g["max_rel_error"])  # fmt: skip
        LOG.info("exact_gw: %d draws in %.1fs (%.3fs/eval), max R-hat %.3f, min ESS %.0f, "
                 "%d divergences", len(draws), diag["seconds"], diag["seconds_per_eval"],
                 diag["max_rhat"], diag["min_ess"], diag["divergences"])  # fmt: skip
        if diag["max_rhat"] > 1.05 or diag["divergences"]:
            LOG.warning("exact_gw: chains may not have converged (max R-hat %.3f, %d "
                        "divergences); increase inference.method.warmup/draws",
                        diag["max_rhat"], diag["divergences"])  # fmt: skip
        history = [{
            "round": 0, "simulations": 0, "valid_simulations": 0, "simulation_seconds": 0.0,
            "training_seconds": diag["seconds"], "epochs": 0,
            "best_validation_loss": float("nan"),
        }]  # fmt: skip
        return ExactPosterior(
            posterior=None, feature_mask=np.ones(int(x_dim), dtype=bool), method=self.name,
            amortized=False, history=history, draws=draws, diagnostics=diag,
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
        opts = {"chains": 2, "warmup": 100, "draws": 100, "init_starts": 2, **self.sbc}
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
