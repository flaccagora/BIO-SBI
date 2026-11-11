"""Exact-likelihood Bayesian inference for the two-type Bellman-Harris model (``exact_bh``).

The likelihood of the raw endpoint counts is exact (method of stages for integer Gamma
lifetime shapes, :mod:`bio_sbi.inference.bh_exact`):

    log p(data | theta) = sum_h sum_clones log p~_h(A, B | theta),

with ``p~_h`` the law of a clone at horizon ``h`` (conditioned on detection for
``selection=detected``). One ODE trajectory per grid point serves every horizon.

Parameters and prior are those of NPE: ``theta`` = the stick-breaking coordinates of
:class:`~bio_sbi.simulators.law.ParameterSpace` (Beta priors, Dirichlet rows) plus the
free log mean lifetimes (uniform on ``log lifetime_mean_bounds``). The sampler works on
``u = logit((theta - low) / (high - low))`` (prior Jacobian included).

Posterior sampling. One likelihood evaluation costs ~0.2 s at 16 threads on the
real-like design (dt = 2 h, grid ~340 x 260), a gradient (forward-mode tangents) ~8-10
evaluations. The default sampler (``sampler=smc``) is **adaptive tempered SMC followed by
adaptive Metropolis chains run to a convergence target**
(:func:`bio_sbi.inference.smc.tempered_smc`):

1. *Start*: the prior centre and ``init_candidates`` prior draws are evaluated (in
   parallel); the best ``opt_starts`` start optimisations.
2. *Modes*: Levenberg-Marquardt ascent of the log posterior in ``u`` with exact gradients
   and the BHHH (outer product of per-clone scores) curvature, which needs no second
   derivatives; typically 10-30 gradient evaluations per start.
3. *Laplace*: covariance = inverse curvature at each distinct mode within
   ``mode_tolerance`` nats of the best (eigenvalues floored at ``1 / laplace_max_var``);
   mixture of t (``df``) components weighted by Laplace evidence.
4. *Tempering* (``start=laplace``): ``chains x steps`` draws from the Laplace mixture
   widened by ``start_inflate``, then the geometric path ``q^(1-beta) posterior^beta``
   with adaptive ``beta`` (ESS of the incremental weights = ``ess_fraction``), waste-free
   resample-move stages (``chains`` resampled starts x ``steps - 1`` MH steps; random
   walk with the population covariance + independence t proposals). ``start=prior``
   tempers the likelihood from the prior instead (no optimisation; ``coords=probit``).
5. *Final chains* at ``beta = 1``: the ``chains`` chains run on with adaptive Metropolis
   (covariance, scale and independence proposal refitted from the pooled chains) until
   min bulk ESS ``>= target_ess``, min tail ESS ``>= tail_fraction x target_ess`` and
   max nested R-hat (superchains = contiguous groups of chains) ``<= max_rhat``, or the
   ``max_evals`` budget; the first ``final_burn`` of the chains is discarded. The draws
   (equal weights) are the remaining chain states. ``islands > 0`` replaces 5. by
   independent tempered runs pooled until the same targets hold across islands.

Optional sampler coordinates (``reparam``, :mod:`bio_sbi.inference.bh_reparam`, needs
``start=laplace``): after the optimisation, stages 4-5 run on ``v``, a bijection of ``u``
anchored at the mode, with the exact log Jacobian (prior and posterior unchanged; the
Laplace mixture is mapped to ``v``; R-hat / ESS are computed on ``u``). ``stable`` shears
both log mean lifetimes onto the surface where the Malthusian parameter and the stable
B:A ratio keep their values at the mode, which straightens the lifetime / self-renewal
ridge; ``+probit`` adds probit stick coordinates (Gaussian tails at probabilities ~0).

Why not importance sampling: the posterior in ``u`` has long tails (probabilities the
data allow to be ~0) and the curved lifetime / self-renewal ridge, so Laplace or
mixture proposals give Pareto ``k > 0.7`` (the first-look AMIS runs: ESS 12-38 of
~1300); a Laplace-started SMC population alone is under-dispersed (tails); only long
chains matched a 128k-evaluation reference on the validation design.

Legacy ``sampler=amis``: Laplace-initialised adaptive multiple importance sampling,
``max_iterations`` x ``particles`` draws with deterministic-mixture PSIS weights and a
weighted EM refit per iteration; optional resample-move (``mh_steps``).

Diagnostics (``metrics.json`` under ``exact_bh``): ``sampler``, ``ess`` (smc: min bulk
ESS; amis: PSIS ESS), ``reliable`` (smc: all targets met), ``smc`` (temperatures, stage
ESS / Pareto ``k`` / acceptance rates, final checks, per-coordinate nested and split
R-hat, bulk and tail ESS), optima, evaluation counts and timings, grid and a grid accuracy
check at the posterior median.

Supported: ``simulator=bellman_harris`` with integer ``lifetime_shape`` (per type
allowed), two types (``two_type``, ``reversible_ab``, any fixed entries), fixed founder
rows, homogeneous law, constant lifetimes (no density dependence), two labels with any
capture and 2x2 confusion (pgf substitution), ``selection`` detected or all. Anything
else raises ``ValueError`` (non-integer shapes ``NotImplementedError``).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import betaln, expit, gammaln, log_expit, log_ndtr, logsumexp, ndtri

from ..rng import rng_for
from .base import InferenceMethod
from .bh_exact import BHModel, check_shapes, detection_and_grad, law_and_jac
from .bh_reparam import SPECS as REPARAM_SPECS
from .bh_reparam import Reparam
from .exact_gw import ExactPosterior, GridTooLarge
from .gw_exact import GridSpec, grid_spec, histogram
from .smc import psis, smc_islands, tempered_smc

LOG = logging.getLogger(__name__)

#: Box for ``u`` during optimisation.
U_BOUND = 12.0


# ---------------------------------------------------------------------- support check
def check_supported(simulator, observation) -> None:
    """Raise ``ValueError`` unless the exact BH likelihood applies to this model."""
    from ..simulators import BellmanHarris

    if not isinstance(simulator, BellmanHarris):
        raise ValueError(
            f"inference=exact_bh needs simulator=bellman_harris; got {type(simulator).__name__}"
        )
    if simulator.density_dependent:
        raise ValueError("inference=exact_bh does not support density_dependence")
    if len(simulator.types) != 2:
        raise ValueError(f"inference=exact_bh supports two cell types; got {simulator.types}")
    if simulator.heterogeneous:
        raise ValueError("inference=exact_bh does not support law.clone_heterogeneity")
    lifetimes = {f"{t}.mean_lifetime" for t in simulator.types}
    other = [s.name for s in simulator.space.free_scalars if s.name not in lifetimes]
    if other:
        raise ValueError(f"inference=exact_bh: free scalar parameters {other} are not supported")
    for row in simulator.space.rows:
        if row.prefix in ("founder", "founder_size") and row.n_sticks:
            raise ValueError(
                f"inference=exact_bh needs a fixed {row.prefix} row (e.g. law.fixed.founder.A=1);"
                " free founder parameters are not supported"
            )
    if observation.n_labels != 2 or tuple(observation.types) != tuple(simulator.types):
        raise ValueError("inference=exact_bh needs two measured labels for the two types")
    check_shapes(simulator.shape)  # NotImplementedError for non-integer shapes


# ---------------------------------------------------------------------- likelihood
@dataclass
class CohortData:
    """Histogram of one cohort and the index of its horizon."""

    horizon: float
    index: int
    a: np.ndarray
    b: np.ndarray
    weight: np.ndarray

    @property
    def clones(self) -> int:
        """Number of clones."""
        return int(self.weight.sum())


class BHLikelihood:
    """Exact log-likelihood / log-posterior of one dataset under the BH model.

    Args:
        simulator: Inference-model :class:`~bio_sbi.simulators.BellmanHarris`.
        observation: Inference-model observation model.
        cohorts: The dataset (``Cohort`` list).
        grid: ``factor``, ``margin``, ``min_size``, ``damping`` (see
            :func:`~bio_sbi.inference.gw_exact.grid_spec`) and ``max_points``. One
            rectangular grid serves all horizons, sized from the largest A and B counts.
        floor: Probability floor.
        min_detection: Floor of the detection probability in the conditioning.
        dt, integrator: Runge-Kutta step and tableau.
        threads: Threads per likelihood evaluation (grid split) and for batches.
    """

    def __init__(self, simulator, observation, cohorts, grid: dict | None = None,
                 floor: float = 1e-15, min_detection: float = 1e-6, dt: float = 1.0,
                 integrator: str = "dop853", threads: int = 1) -> None:  # fmt: skip
        check_supported(simulator, observation)
        self.sim = simulator
        self.space = space = simulator.space
        law = simulator.law
        self.floor = float(floor)
        self.min_detection = float(min_detection)
        self.threads = max(int(threads), 1)
        theta0 = np.array([0.5 if i < self._n_sticks() else np.mean(b)
                           for i, b in enumerate(self._bounds())])  # fmt: skip
        _, founder, size_probs = law.unpack(space, space.decode(theta0))
        self.model = BHModel(
            offspring=np.ascontiguousarray(law.offspring, dtype=np.int64),
            n_out=np.ascontiguousarray(law.n_outcomes, dtype=np.int64),
            phases=check_shapes(simulator.shape),
            capture=np.asarray(observation.capture, dtype=float),
            confusion=np.asarray(observation.confusion, dtype=float),
            founder=np.asarray(founder, dtype=float),
            sizes=np.asarray(law.founder_sizes, dtype=np.int64),
            size_probs=np.asarray(size_probs, dtype=float),
            detected=observation.selection == "detected",
            dt=float(dt),
            integrator=str(integrator),
        )
        # data: one histogram per cohort; one grid for all horizons
        self.horizons = sorted({float(c.horizon) for c in cohorts})
        self.data: list[CohortData] = []
        for c in cohorts:
            a, b, w = histogram(c.counts)
            if self.model.detected and np.any((a == 0) & (b == 0)):
                raise ValueError("selection=detected but the data contain empty clones")
            self.data.append(CohortData(float(c.horizon), self.horizons.index(float(c.horizon)),
                                        a, b, w))  # fmt: skip
        grid = dict(grid or {})
        max_points = float(grid.pop("max_points", 2e6))
        max_a = max(int(d.a.max()) for d in self.data)
        max_b = max(int(d.b.max()) for d in self.data)
        self.grid = grid_spec(max_a, max_b, **grid)
        if self.grid.Lx * self.grid.Ly > max_points:
            raise GridTooLarge(
                f"Counts up to ({max_a}, {max_b}) need a {self.grid.Lx}x{self.grid.Ly} grid "
                f"> grid.max_points={max_points:g}"
            )
        # theta coordinate -> derivative direction
        self.directions: list[tuple] = []  # (kind, type index, row, stick j)
        prior_a, prior_b = [], []
        for r in space.rows:
            if not r.n_sticks:
                continue
            k = law.types.index(r.prefix)  # founder rows have no sticks (checked)
            for j in range(r.n_sticks):
                self.directions.append((0, k, r, j))
            tail = np.cumsum(r.alpha[::-1])[::-1]
            prior_a.extend(r.alpha[:-1])
            prior_b.extend(tail[1:])
        for s in space.free_scalars:
            self.directions.append((1, law.types.index(s.name.split(".")[0]), None, 0))
            prior_a.append(1.0)  # uniform on [log low, log high]
            prior_b.append(1.0)
        self.prior_a = np.asarray(prior_a, dtype=float)
        self.prior_b = np.asarray(prior_b, dtype=float)
        lo_hi = np.asarray(self._bounds(), dtype=float)
        self.low, self.width = lo_hi[:, 0], lo_hi[:, 1] - lo_hi[:, 0]
        self.dim = len(self.directions)
        self.kind = np.array([d[0] for d in self.directions], dtype=np.int64)
        self.rows = np.array([d[1] for d in self.directions], dtype=np.int64)
        self.n_outcomes_max = law.offspring.shape[1]
        self._pool = ThreadPoolExecutor(self.threads) if self.threads > 1 else None
        self._lock = threading.Lock()
        self.evals = self.grad_evals = 0
        self.eval_seconds = self.grad_seconds = 0.0

    def _n_sticks(self) -> int:
        return sum(r.n_sticks for r in self.space.rows)

    def _bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)] * self._n_sticks() + [s.theta_bounds for s in self.space.free_scalars]

    def close(self) -> None:
        """Shut the evaluation thread pool down."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    # ---------------------------------------------------------------- coordinates
    def theta_from_u(self, u: np.ndarray) -> np.ndarray:
        """``theta = low + width * sigmoid(u)``."""
        return self.low + self.width * expit(u)

    def u_from_theta(self, theta: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`theta_from_u` (clipped to the open box)."""
        s = np.clip((np.asarray(theta, float) - self.low) / self.width, 1e-9, 1 - 1e-9)
        return np.log(s) - np.log1p(-s)

    def parameters(self, theta: np.ndarray):
        """``P (2, O)``, mean lifetimes ``(2,)``, ``dP (d, O)`` per coordinate."""
        theta = np.asarray(theta, dtype=float)
        physical = self.space.decode(theta)
        P, _, _ = self.sim.law.unpack(self.space, physical)
        means = self.sim.mean_lifetimes(physical)
        dP = np.zeros((self.dim, self.n_outcomes_max))
        start = 0
        for t, (kind, _k, r, j) in enumerate(self.directions):
            if kind:
                continue  # d / d log mean_k: handled by the kernel
            if j == 0:
                start = t
            s = theta[start : start + r.n_sticks]
            free = list(r.free)
            m = len(free)
            for i in range(j, m):
                # p_i = mass * s_i * prod_{l<i} (1 - s_l), with s_{m-1} := 1 for the last
                si = s[i] if i < m - 1 else 1.0
                others = np.prod([1 - s[k2] for k2 in range(i) if k2 != j])
                dP[t, free[i]] = r.mass * (others if i == j else -si * others)
        return P, means, dP

    # ---------------------------------------------------------------- likelihood
    def evaluate(self, theta: np.ndarray, grad: bool = False, workers: int | None = None
                 ) -> tuple[float, np.ndarray | None, np.ndarray | None]:  # fmt: skip
        """Log-likelihood, its gradient in ``theta`` and the BHHH matrix (``grad=True``).

        BHHH = per cohort, the sum over clones of the outer products of the centred
        per-clone scores (detection conditioning included): an estimate of the
        information that stays a good Newton metric far from the mode.
        """
        t0 = time.perf_counter()
        workers = self.threads if workers is None else int(workers)
        pool = self._pool if workers > 1 else None
        P, means, dP = self.parameters(theta)
        if not (np.all(np.isfinite(P)) and np.all(np.isfinite(means))):
            return -np.inf, None, None
        args = (dP, self.kind, self.rows) if grad else (None, None, None)
        p, dp = law_and_jac(self.model, P, means, self.horizons, self.grid, *args,
                            workers=workers, pool=pool)  # fmt: skip
        if self.model.detected:
            q, dq = detection_and_grad(self.model, P, means, self.horizons, *args)
        total = 0.0
        g = np.zeros(self.dim) if grad else None
        bhhh = np.zeros((self.dim, self.dim)) if grad else None
        for d in self.data:
            pc = p[d.index, d.a, d.b]
            ok = pc > self.floor
            pz = np.where(ok, pc, self.floor)
            total += float(d.weight @ np.log(pz))
            if grad:
                score = (dp[d.index][:, d.a, d.b] * np.where(ok, 1.0 / pz, 0.0)).T  # (cells, D)
            if self.model.detected:
                qh = float(q[d.index])
                if qh > self.min_detection:
                    total -= d.clones * math.log(qh)
                    if grad:
                        score = score - dq[d.index] / qh
                else:
                    total -= d.clones * math.log(self.min_detection)
            if grad:
                gh = d.weight @ score
                g += gh
                # centred: far from the mode the mean score would add g g' / n and
                # cap every step's gain at O(n) nats (slow BHHH ascent)
                bhhh += (score * d.weight[:, None]).T @ score - np.outer(gh, gh) / d.clones
        with self._lock:
            if grad:
                self.grad_evals += 1
                self.grad_seconds += time.perf_counter() - t0
            else:
                self.evals += 1
                self.eval_seconds += time.perf_counter() - t0
        return total, g, bhhh

    def log_posterior_u(self, u: np.ndarray, grad: bool = False, workers: int | None = None):
        """Log posterior density of ``u`` (prior and Jacobian included).

        With ``grad=True`` returns ``(lp, gradient, curvature)``; ``curvature``
        approximates ``-Hessian`` (BHHH for the likelihood, exact for the rest).
        """
        u = np.asarray(u, dtype=float)
        sig = expit(u)
        theta = self.low + self.width * sig
        ll, g, bhhh = self.evaluate(theta, grad, workers)
        prior = float(self.prior_a @ log_expit(u) + self.prior_b @ log_expit(-u))
        if not np.isfinite(ll):
            return (-np.inf, None, None) if grad else -np.inf
        if not grad:
            return ll + prior
        J = self.width * sig * (1 - sig)  # d theta / d u
        J2 = J * (1 - 2 * sig)  # d^2 theta / d u^2
        gu = g * J + self.prior_a * (1 - sig) - self.prior_b * sig
        curv = bhhh * np.outer(J, J) + np.diag((self.prior_a + self.prior_b) * sig * (1 - sig))
        curv -= np.diag(g * J2)
        return ll + prior, gu, 0.5 * (curv + curv.T)

    def log_posterior_batch(self, U: np.ndarray) -> np.ndarray:
        """Log posterior of every row of ``U``: one particle per thread.

        Batches smaller than ``threads`` give each particle ``threads // len(U)`` threads
        (grid split; one particle alone uses all of them).
        """
        U = np.atleast_2d(U)
        if self._pool is None or len(U) == 1:
            return np.array([self.log_posterior_u(u) for u in U])
        per = self.threads // len(U)
        if per <= 1:
            return np.array(list(self._pool.map(lambda u: self.log_posterior_u(u, workers=1),
                                                U)))  # fmt: skip
        return np.array(list(self._pool.map(lambda u: self._lane_eval(u, per), U)))

    def _lane_eval(self, u: np.ndarray, workers: int) -> float:
        """One particle on its own ``workers``-thread pool (no nesting in ``_pool``)."""
        with ThreadPoolExecutor(workers) as inner:
            return self._log_posterior_with_pool(np.asarray(u, dtype=float), workers, inner)

    def _log_posterior_with_pool(self, u, workers, pool) -> float:
        t0 = time.perf_counter()
        P, means, _ = self.parameters(self.theta_from_u(u))
        if not (np.all(np.isfinite(P)) and np.all(np.isfinite(means))):
            return -np.inf
        p, _ = law_and_jac(self.model, P, means, self.horizons, self.grid, workers=workers,
                           pool=pool)  # fmt: skip
        ll = self._loglik_from_law(p, P, means)
        with self._lock:
            self.evals += 1
            self.eval_seconds += time.perf_counter() - t0
        prior = float(self.prior_a @ log_expit(u) + self.prior_b @ log_expit(-u))
        return ll + prior if np.isfinite(ll) else -np.inf

    def _loglik_from_law(self, p, P, means) -> float:
        if self.model.detected:
            q, _ = detection_and_grad(self.model, P, means, self.horizons)
        total = 0.0
        for d in self.data:
            pc = p[d.index, d.a, d.b]
            total += float(d.weight @ np.log(np.where(pc > self.floor, pc, self.floor)))
            if self.model.detected:
                qh = float(q[d.index])
                total -= d.clones * math.log(max(qh, self.min_detection))
        return total

    @property
    def seconds_per_eval(self) -> float:
        """Mean wall time of one likelihood evaluation (all threads on one particle)."""
        return self.eval_seconds / self.evals if self.evals else float("nan")

    # ---------------------------------------------------------------- checks
    def grid_check(self, theta: np.ndarray) -> list[dict]:
        """Accuracy of the (damped) grid at ``theta`` against an undamped grid twice as large.

        Per horizon: ``tail_mass`` = mass of the law at counts outside the grid, and
        ``max_rel_error`` of the used law on the observed cells.
        """
        P, means, _ = self.parameters(theta)
        g = self.grid
        pool = self._pool
        p, _ = law_and_jac(self.model, P, means, self.horizons, g, workers=self.threads,
                           pool=pool)  # fmt: skip
        big = GridSpec(2 * g.Lx, 2 * g.Ly, 1.0, 1.0)
        pb, _ = law_and_jac(self.model, P, means, self.horizons, big, workers=self.threads,
                            pool=pool)  # fmt: skip
        out = []
        for d in self.data:
            ref = pb[d.index]
            det = self.model.detected
            norm = 1 - ref[0, 0] if det else 1.0
            inside = ref[: g.Lx, : g.Ly].sum() - (ref[0, 0] if det else 0.0)
            r = ref[d.a, d.b]
            keep = r > 100 * self.floor
            rel = np.abs(p[d.index, d.a, d.b][keep] - r[keep]) / r[keep]
            out.append({
                "horizon": d.horizon, "Lx": g.Lx, "Ly": g.Ly, "rx": g.rx, "ry": g.ry,
                "max_count_A": int(d.a.max()), "max_count_B": int(d.b.max()),
                "tail_mass": float(max(1 - inside / norm, 0.0)),
                "max_rel_error": float(rel.max()) if len(rel) else 0.0,
            })  # fmt: skip
        return out


# ---------------------------------------------------------------------- IS helpers
def t_logpdf(x: np.ndarray, mu: np.ndarray, chol: np.ndarray, df: float) -> np.ndarray:
    """Log density of a multivariate t (location ``mu``, scale ``chol chol^T``)."""
    d = len(mu)
    z = solve_triangular(chol, (np.atleast_2d(x) - mu).T, lower=True)
    q = (z**2).sum(0)
    return (gammaln((df + d) / 2) - gammaln(df / 2) - 0.5 * d * math.log(df * math.pi)
            - np.log(np.diag(chol)).sum() - 0.5 * (df + d) * np.log1p(q / df))  # fmt: skip


def t_sample(rng, n: int, mu: np.ndarray, chol: np.ndarray, df: float) -> np.ndarray:
    """``n`` draws of a multivariate t."""
    z = rng.standard_normal((n, len(mu)))
    w = rng.chisquare(df, n) / df
    return mu + (z @ chol.T) / np.sqrt(w)[:, None]


def kish_ess(log_w: np.ndarray) -> float:
    """Effective sample size ``(sum w)^2 / sum w^2`` of log weights."""
    lw = np.asarray(log_w, float)
    lw = lw[np.isfinite(lw)]
    if not len(lw):
        return 0.0
    w = np.exp(lw - lw.max())
    return float(w.sum() ** 2 / (w**2).sum())


def pareto_k(log_w: np.ndarray) -> float:
    """PSIS Pareto ``k`` of the importance-weight tail (see :func:`psis`)."""
    return psis(log_w)[1]


def _mahalanobis(x: np.ndarray, mu: np.ndarray, chol: np.ndarray) -> float:
    return float(np.linalg.norm(solve_triangular(chol, x - mu, lower=True)))


@dataclass
class MixtureT:
    """Mixture of multivariate t distributions (common ``df``) for importance sampling."""

    means: np.ndarray  # (K, d)
    chols: np.ndarray  # (K, d, d) lower Cholesky factors of the scale matrices
    logpi: np.ndarray  # (K,) log component weights
    df: float

    def component_logpdf(self, x: np.ndarray) -> np.ndarray:
        """``(n, K)`` log densities of every component (weights included)."""
        return np.stack([lp + t_logpdf(x, m, c, self.df)
                         for m, c, lp in zip(self.means, self.chols, self.logpi, strict=True)],
                        axis=1)  # fmt: skip

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Mixture log density."""
        return logsumexp(self.component_logpdf(x), axis=1)

    def sample(self, rng, n: int) -> np.ndarray:
        """``n`` draws."""
        k = rng.choice(len(self.logpi), size=n, p=np.exp(self.logpi - logsumexp(self.logpi)))
        out = np.empty((n, self.means.shape[1]))
        for j in range(len(self.logpi)):
            idx = np.flatnonzero(k == j)
            if len(idx):
                out[idx] = t_sample(rng, len(idx), self.means[j], self.chols[j], self.df)
        return out

    def refit(self, x: np.ndarray, w: np.ndarray, min_weight: float = 1e-3) -> MixtureT:
        """One weighted EM step (PMC, Cappe et al. 2008) towards the weighted particles.

        Each component's new covariance is shrunk towards its old scale when its share
        of the weights has a small effective sample size (``ess / (ess + d + 1)``).
        Components with weight below ``min_weight`` are dropped.
        """
        d = x.shape[1]
        lr = self.component_logpdf(x)
        r = np.exp(lr - logsumexp(lr, axis=1, keepdims=True))  # responsibilities
        means, chols, logpi = [], [], []
        for j in range(len(self.logpi)):
            wr = w * r[:, j]
            pi = wr.sum()
            if pi < min_weight:
                continue
            v = wr / pi
            m = v @ x
            c = ((x - m) * v[:, None]).T @ (x - m)
            ess = 1.0 / float((v**2).sum())
            alpha = ess / (ess + d + 1)
            old = self.chols[j] @ self.chols[j].T
            cov = alpha * c + (1 - alpha) * old
            means.append(m)
            chols.append(np.linalg.cholesky(cov + 1e-10 * np.eye(d)))
            logpi.append(math.log(pi))
        if not means:  # keep the old mixture
            return self
        lp = np.array(logpi)
        return MixtureT(np.array(means), np.array(chols), lp - logsumexp(lp), self.df)


def _floored_inverse(curv: np.ndarray, max_var: float) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (curv + curv.T))
    w = np.maximum(w, 1.0 / max_var)
    return (V / w) @ V.T


# ---------------------------------------------------------------------- method
class ExactBH(InferenceMethod):
    """Exact-likelihood inference for the two-type Bellman-Harris model (module docstring).

    Args:
        threads: Threads (grid split within an evaluation, particles across threads).
        dt: Runge-Kutta step in hours.
        integrator: ``dop853`` (default) or ``rk4``.
        init_candidates: Prior draws evaluated to pick the optimisation start.
        opt_starts: Optimisations, from the best candidates (each ~5-20 gradients).
        opt_maxiter: Levenberg-Marquardt iterations (each accepted one = 1 gradient).
        opt_tol: Stop when the Newton decrement (predicted gain) is below this.
        mode_tolerance: Optima more than this many nats below the best are dropped;
            distinct remaining optima become components of the first proposal.
        defensive_weight, defensive_scale: Weight and scale factor of a wider copy of
            every Laplace component in the first proposal (0 disables).
        laplace_max_var: Variance cap of the Laplace covariance in ``u`` (flat directions).
        inflate: Scale factor of the Laplace covariance for the first proposal.
        df: Degrees of freedom of the t proposals.
        particles: Draws per AMIS iteration.
        max_iterations: AMIS iterations (budget = particles x max_iterations evaluations).
        target_ess: Stop once the Kish ESS reaches this.
        psis: Pareto-smooth the importance weights (PSIS) for adaptation and output.
        mh_steps: Resample-move rejuvenation: ``particles`` draws resampled from the
            weights, each moved by this many random-walk Metropolis steps (proposal
            covariance ``2.38^2 / d`` x weighted covariance; ``particles`` evaluations per
            step). The moved draws (equal weights) are the output. 0 disables.
        reparam: SMC sampler coordinates (:mod:`~bio_sbi.inference.bh_reparam`):
            ``none`` (``u``), ``stable``, ``growth``, ``probit`` or ``stable+probit`` /
            ``growth+probit``; needs ``sampler=smc`` and ``start=laplace``.
        grid: Grid options (see :class:`BHLikelihood`).
        floor, min_detection: Likelihood floors.
        sbc: Overrides of the sampler settings (and ``max_points``) for SBC cases.
        device, seed, rounds, **_: Interface compatibility (``rounds`` must be 1).
    """

    name = "exact_bh"
    exact = True

    def __init__(
        self,
        threads: int = 2,
        dt: float = 2.0,
        integrator: str = "dop853",
        init_candidates: int = 32,
        opt_starts: int = 2,
        opt_maxiter: int = 40,
        opt_tol: float = 1e-3,
        mode_tolerance: float = 20.0,
        defensive_weight: float = 0.0,
        defensive_scale: float = 2.0,
        laplace_max_var: float = 4.0,
        inflate: float = 1.0,
        df: float = 5.0,
        particles: int = 200,
        max_iterations: int = 6,
        target_ess: float = 400.0,
        psis: bool = True,
        mh_steps: int = 0,
        sampler: str = "smc",
        start: str = "laplace",
        coords: str = "probit",
        reparam: str = "none",
        start_inflate: float = 2.0,
        chains: int = 64,
        steps: int = 8,
        ess_fraction: float = 0.5,
        rw_acceptance: float = 0.3,
        imh_prob: float = 0.5,
        imh_min: float = 0.05,
        imh_scale: float = 1.2,
        imh_components: int = 1,
        de_prob: float = 0.0,
        rw_scale_sd: float = 1.0,
        max_rhat: float = 1.01,
        tail_fraction: float = 0.5,
        superchains: int = 4,
        islands: int = 0,
        max_islands: int = 8,
        final_steps: int = 16,
        min_final_steps: int = 16,
        max_final_steps: int = 5000,
        final_burn: float = 0.25,
        adapt_final: bool = True,
        adapt_final_steps: int = 1000,
        local_k: int = 0,
        check_every: int = 8,
        max_evals: int = 40000,
        grid: dict | None = None,
        floor: float = 1e-15,
        min_detection: float = 1e-6,
        sbc: dict | None = None,
        device: str = "cpu",
        seed: int = 0,
        rounds: int = 1,
        **_: Any,
    ) -> None:
        if rounds != 1:
            raise ValueError("exact_bh has no sequential rounds (rounds must be 1)")
        super().__init__(rounds=1, device=device, seed=seed)
        self.threads = max(int(threads), 1)
        self.dt, self.integrator = float(dt), str(integrator)
        self.settings = {
            "init_candidates": int(init_candidates), "opt_starts": int(opt_starts),
            "opt_maxiter": int(opt_maxiter), "mode_tolerance": float(mode_tolerance),
            "defensive_weight": float(defensive_weight),
            "defensive_scale": float(defensive_scale),
            "opt_tol": float(opt_tol), "laplace_max_var": float(laplace_max_var),
            "inflate": float(inflate), "df": float(df), "particles": int(particles),
            "max_iterations": int(max_iterations), "target_ess": float(target_ess),
            "psis": bool(psis), "mh_steps": int(mh_steps),
            "sampler": str(sampler), "start": str(start), "coords": str(coords),
            "reparam": str(reparam or "none"), "start_inflate": float(start_inflate),
            "chains": int(chains), "steps": int(steps), "ess_fraction": float(ess_fraction),
            "rw_acceptance": float(rw_acceptance), "imh_prob": float(imh_prob),
            "imh_min": float(imh_min),
            "imh_scale": float(imh_scale), "imh_components": int(imh_components),
            "de_prob": float(de_prob), "rw_scale_sd": float(rw_scale_sd),
            "max_rhat": float(max_rhat),
            "tail_fraction": float(tail_fraction), "superchains": int(superchains),
            "islands": int(islands), "max_islands": int(max_islands),
            "final_steps": int(final_steps),
            "min_final_steps": int(min_final_steps), "max_final_steps": int(max_final_steps),
            "final_burn": float(final_burn), "adapt_final": bool(adapt_final),
            "adapt_final_steps": int(adapt_final_steps), "local_k": int(local_k),
            "check_every": int(check_every), "max_evals": int(max_evals),
        }  # fmt: skip
        if start not in ("laplace", "prior"):
            raise ValueError(f"exact_bh: unknown start {start!r} (laplace | prior)")
        if self.settings["reparam"] not in REPARAM_SPECS:
            raise ValueError(f"exact_bh: unknown reparam {reparam!r} ({' | '.join(REPARAM_SPECS)})")
        if self.settings["reparam"] != "none" and (start != "laplace" or sampler != "smc"):
            raise ValueError(
                "exact_bh: reparam needs sampler=smc and start=laplace (the "
                "growth shear is anchored at the posterior mode)"
            )
        self.grid = dict(grid or {})
        self.floor = float(floor)
        self.min_detection = float(min_detection)
        self.sbc = dict(sbc or {})

    def make_trainer(self, prior, tracker):
        """Not applicable."""
        raise NotImplementedError("exact_bh does not train a network")

    def fit(self, *args, **kwargs):
        """Not applicable: see :meth:`fit_data`."""
        raise TypeError("exact_bh conditions on raw counts: use fit_data(...)")

    def likelihood(self, simulator, observation, cohorts, max_points=None) -> BHLikelihood:
        """Exact likelihood of one dataset (validates the model)."""
        grid = dict(self.grid)
        if max_points is not None:
            grid["max_points"] = max_points
        return BHLikelihood(simulator, observation, cohorts, grid, self.floor,
                            self.min_detection, self.dt, self.integrator, self.threads)  # fmt: skip

    # ---------------------------------------------------------------- sampler stages
    @staticmethod
    def optimise(lik: BHLikelihood, u0: np.ndarray, maxiter: int, tol: float,
                 min_curvature: float = 0.25) -> dict:  # fmt: skip
        """Levenberg-Marquardt ascent of the log posterior in ``u`` with BHHH curvature.

        The curvature's eigenvalues are floored at ``min_curvature`` (steps of at most
        ``|g| / min_curvature`` in flat directions). Coordinates at the box ``|u| =
        U_BOUND`` whose gradient points outwards are held fixed. Converged when the
        Newton decrement ``g' H^-1 g / 2`` of the free coordinates is below ``tol``.
        """
        u = np.clip(np.asarray(u0, float), -U_BOUND, U_BOUND)
        lp, g, curv = lik.log_posterior_u(u, grad=True)
        if not np.isfinite(lp):
            raise RuntimeError("exact_bh: log posterior is not finite at the start point")
        damp, trace, status = 1e-3, [float(lp)], "maxiter"
        for _ in range(int(maxiter)):
            held = (np.abs(u) >= U_BOUND - 1e-9) & (np.sign(g) == np.sign(u))
            gf = np.where(held, 0.0, g)
            w, V = np.linalg.eigh(curv)
            w = np.maximum(w, min_curvature)
            H = (V * w) @ V.T
            if 0.5 * float(gf @ np.linalg.solve(H, gf)) < tol:
                status = "converged"
                break
            while True:
                step = np.linalg.solve(H + damp * np.diag(np.diag(H)), gf)
                u_new = np.clip(u + np.where(held, 0.0, step), -U_BOUND, U_BOUND)
                lp_new = lik.log_posterior_u(u_new)
                if np.isfinite(lp_new) and lp_new > lp:
                    damp = max(damp / 10, 1e-6)
                    break
                damp *= 10
                if damp > 1e6:
                    break
            if damp > 1e6:
                status = "stalled"  # no ascent along the damped Newton direction
                break
            u = u_new
            lp, g, curv = lik.log_posterior_u(u, grad=True)
            trace.append(float(lp))
        return {"u": u, "log_posterior": float(lp), "gradient": g, "curvature": curv,
                "trace": trace, "status": status}  # fmt: skip

    @staticmethod
    def rejuvenate(lik: BHLikelihood, x: np.ndarray, lp: np.ndarray, w: np.ndarray,
                   steps: int, n: int, rng, name: str = "") -> tuple:  # fmt: skip
        """Resample ``n`` particles by weight and apply ``steps`` random-walk MH moves.

        MH leaves the posterior invariant, so the moves only remove the duplicates of
        resampling (and part of the remaining weight error). Returns
        ``(particles, uniform weights, info)``.
        """
        d = x.shape[1]
        idx = rng.choice(len(x), n, p=w)
        cur, cur_lp = x[idx].copy(), lp[idx].copy()
        m = w @ x
        cov = ((x - m) * w[:, None]).T @ (x - m)
        chol = np.linalg.cholesky(cov * 2.38**2 / d + 1e-10 * np.eye(d))
        acc = []
        for k in range(int(steps)):
            prop = cur + rng.standard_normal((n, d)) @ chol.T
            plp = lik.log_posterior_batch(prop)
            ok = np.isfinite(plp) & (np.log(rng.uniform(size=n)) < plp - cur_lp)
            cur[ok], cur_lp[ok] = prop[ok], plp[ok]
            acc.append(float(ok.mean()))
            LOG.info("%s: MH rejuvenation step %d: acceptance %.2f", name, k, acc[-1])
        info = {"steps": int(steps), "particles": int(n), "acceptance": acc,
                "unique_resampled": int(len(np.unique(idx))),
                "unique_final": int(len(np.unique(cur, axis=0)))}  # fmt: skip
        return cur, np.full(n, 1.0 / n), info

    def _run_smc(self, lik: BHLikelihood, s: dict, rng, name: str, ref: MixtureT | None,
                 optima: list, t0: float, t_init: float, t_opt: float, init_best: float
                 ) -> tuple[np.ndarray, np.ndarray, dict]:  # fmt: skip
        """Tempered SMC from ``ref`` (Laplace mixture) or, for ``ref=None``, the prior."""
        logpost, to_u, reparam_info = lik.log_posterior_batch, None, {"spec": "none"}
        if ref is None and s["coords"] == "probit":
            # v = probit(unit-scaled theta): posteriors piling up at a boundary of a stick
            # (e.g. a probability ~0 the data allow) have Gaussian, not exponential, tails
            a, b = lik.prior_a, lik.prior_b
            log_norm = float(betaln(a, b).sum())

            def to_u(V):
                V = np.atleast_2d(V)
                return log_ndtr(V) - log_ndtr(-V)

            def log_jac(V):  # log |du/dv|
                return (-0.5 * V**2 - 0.5 * math.log(2 * math.pi) - log_ndtr(V)
                        - log_ndtr(-V)).sum(axis=1)  # fmt: skip

            def logpost(V):
                V = np.atleast_2d(V)
                return lik.log_posterior_batch(to_u(V)) + log_jac(V)

            def ref_sample(r, n):
                return ndtri(np.clip(r.beta(a, b, size=(n, lik.dim)), 1e-15, 1 - 1e-15))

            def ref_logpdf(V):
                V = np.atleast_2d(V)
                return (log_ndtr(V) @ (a - 1) + log_ndtr(-V) @ (b - 1) - log_norm
                        + (-0.5 * V**2 - 0.5 * math.log(2 * math.pi)).sum(axis=1))  # fmt: skip
        elif ref is None:
            a, b = lik.prior_a, lik.prior_b
            log_norm = float(betaln(a, b).sum())

            def ref_sample(r, n):
                v = np.clip(r.beta(a, b, size=(n, lik.dim)), 1e-12, 1 - 1e-12)
                return np.log(v) - np.log1p(-v)

            def ref_logpdf(U):
                U = np.atleast_2d(U)
                return log_expit(U) @ a + log_expit(-U) @ b - log_norm
        elif s["reparam"] == "none":

            def ref_sample(r, n):
                return np.clip(ref.sample(r, n), -30, 30)

            ref_logpdf = ref.logpdf
        else:
            # sampler coordinates v (bh_reparam): the target and the Laplace mixture are
            # transformed with the exact Jacobian; convergence is checked on u
            best_u = max(optima, key=lambda o: o["log_posterior"])["u"]
            rp = Reparam(lik, s["reparam"], anchor_u=best_u)
            to_u = rp.to_u
            reparam_info = rp.info()
            LOG.info("%s: reparam %s", name, reparam_info)

            def logpost(V):
                V = np.atleast_2d(V)
                return lik.log_posterior_batch(rp.to_u(V)) + rp.log_jac(V)

            def ref_sample(r, n):
                return rp.to_v(np.clip(ref.sample(r, n), -30, 30))

            def ref_logpdf(V):
                V = np.atleast_2d(V)
                return ref.logpdf(rp.to_u(V)) + rp.log_jac(V)

        t2 = time.perf_counter()
        common = {
            "chains": s["chains"], "steps": s["steps"], "ess_fraction": s["ess_fraction"],
            "rw_acceptance": s["rw_acceptance"], "imh_prob": s["imh_prob"],
            "imh_min": s["imh_min"], "imh_scale": s["imh_scale"],
            "imh_components": s["imh_components"], "df": s["df"],
            "de_prob": s["de_prob"], "rw_scale_sd": s["rw_scale_sd"],
        }  # fmt: skip
        monitor = to_u if s["reparam"] != "none" else None  # diagnostics on u
        if s["islands"] > 0:
            res = smc_islands(
                logpost, ref_sample, ref_logpdf, rng, islands=s["islands"],
                max_islands=s["max_islands"], final_steps=s["final_steps"],
                target_ess=s["target_ess"], max_rhat=s["max_rhat"],
                tail_fraction=s["tail_fraction"], max_evals=s["max_evals"], name=name,
                monitor=monitor, **common,
            )  # fmt: skip
        else:
            res = tempered_smc(
                logpost, ref_sample, ref_logpdf, rng, **common,
                target_ess=s["target_ess"], max_rhat=s["max_rhat"],
                tail_fraction=s["tail_fraction"], superchains=s["superchains"],
                min_final_steps=s["min_final_steps"], max_final_steps=s["max_final_steps"],
                final_burn=s["final_burn"], adapt_final=s["adapt_final"],
                adapt_final_steps=s["adapt_final_steps"], local_k=s["local_k"],
                check_every=s["check_every"], max_evals=s["max_evals"], name=name,
                monitor=monitor,
            )  # fmt: skip
        smc_seconds = time.perf_counter() - t2
        self.last_smc, self.last_to_u = res, to_u  # final-stage chains (inspection)
        info = res.info
        u = res.draws if to_u is None else to_u(res.draws)
        w = np.full(len(u), 1.0 / len(u))
        theta = lik.theta_from_u(u)
        physical = lik.space.decode(theta)
        free = [i for i, n in enumerate(lik.space.names) if n in set(lik.space.free_names)]
        mean = w @ physical
        sd = np.sqrt(np.maximum(w @ (physical - mean) ** 2, 0))
        coords = list(lik.space.theta_names)
        per_coord = {k: dict(zip(coords, map(float, info[k]), strict=True))
                     for k in ("nested_rhat", "rhat", "ess_bulk", "ess_tail")}  # fmt: skip
        best_opt = max(optima, key=lambda o: o["log_posterior"]) if optima else None
        reliable = bool(info["reached_posterior"] and info["min_ess_bulk"] >= s["target_ess"]
                        and info["max_nested_rhat"] <= s["max_rhat"]
                        and info["min_ess_tail"] >= s["tail_fraction"] * s["target_ess"])  # fmt: skip
        diag = {
            "sampler": "smc",
            "settings": s,
            "dt": lik.model.dt,
            "integrator": lik.model.integrator,
            "threads": lik.threads,
            "particles": int(len(u)),
            "iterations": int(np.sum(info["stages"])),
            # ``ess`` = smallest bulk ESS over the theta coordinates (MCMC estimate over the
            # final-stage chains); ``pareto_k`` = largest PSIS k of the incremental
            # tempering weights over the stages (information only: moves correct them)
            "ess": float(info["min_ess_bulk"]),
            "ess_tail": float(info["min_ess_tail"]),
            "pareto_k": float(np.nanmax([g["pareto_k"] for g in _flat(info["stage_diagnostics"])]
                                        + [-np.inf])),
            "reliable": reliable,
            "smc": {**{k: v for k, v in info.items()
                       if k not in ("nested_rhat", "rhat", "ess_bulk", "ess_tail")},
                    **per_coord},
            "weighted_mean": {lik.space.names[i]: float(mean[i]) for i in free},
            "weighted_sd": {lik.space.names[i]: float(sd[i]) for i in free},
            "quantiles": {lik.space.names[i]: np.quantile(physical[:, i], [0.05, 0.25, 0.5,
                                                                           0.75, 0.95]).tolist()
                          for i in free},
            "finite_fraction": float(np.isfinite(res.log_post).mean()),
            "modes": len(ref.logpi) if ref is not None else 0,
            "reparam": reparam_info,
            "optima_log_posterior": [o["log_posterior"] for o in optima],
            "optima_status": [o["status"] for o in optima],
            "optimisation_traces": [o["trace"] for o in optima],
            "mode_theta": lik.theta_from_u(best_opt["u"]).tolist() if best_opt else None,
            "mode_log_posterior": best_opt["log_posterior"] if best_opt else None,
            "laplace_sd_u": np.sqrt(np.diag(_floored_inverse(
                best_opt["curvature"], s["laplace_max_var"]))).tolist() if best_opt else None,
            "init_best_log_posterior": init_best,
            "init_seconds": t_init,
            "optimisation_seconds": t_opt,
            "smc_seconds": smc_seconds,
            "seconds": time.perf_counter() - t0,
            "likelihood_evals": lik.evals,
            "gradient_evals": lik.grad_evals,
            "seconds_per_eval": lik.seconds_per_eval,
            "seconds_per_gradient": lik.grad_seconds / lik.grad_evals if lik.grad_evals else None,
            "grid": {"Lx": lik.grid.Lx, "Ly": lik.grid.Ly, "rx": lik.grid.rx, "ry": lik.grid.ry},
            "horizons": lik.horizons,
            "clones": [c.clones for c in lik.data],
        }  # fmt: skip
        return theta, w, diag

    def run(self, lik: BHLikelihood, key=("reference",), **overrides
            ) -> tuple[np.ndarray, np.ndarray, dict]:  # fmt: skip
        """Sample the posterior of ``lik``; returns ``(theta particles, weights, diagnostics)``."""
        s = {**self.settings, **{k: v for k, v in overrides.items() if k in self.settings}}
        name = "/".join(map(str, key))
        rng = rng_for(self.seed, "exact-bh", *key)
        t0 = time.perf_counter()
        d = lik.dim
        if s["sampler"] not in ("smc", "amis"):
            raise ValueError(f"exact_bh: unknown sampler {s['sampler']!r} (smc | amis)")
        if s["sampler"] == "smc" and s["start"] == "prior":
            return self._run_smc(lik, s, rng, name, None, [], t0, 0.0, 0.0, float("nan"))
        # 1. starts: prior centre + prior draws, best few
        cand = rng.beta(lik.prior_a, lik.prior_b, size=(s["init_candidates"], d))
        cand = np.clip(cand, 1e-3, 1 - 1e-3)
        U0 = np.vstack([np.zeros(d), np.log(cand) - np.log1p(-cand)])
        lp0 = lik.log_posterior_batch(U0)
        lp0 = np.where(np.isfinite(lp0), lp0, -np.inf)
        starts = U0[np.argsort(-lp0)[: max(1, s["opt_starts"])]]
        t_init = time.perf_counter() - t0
        # 2. modes (LM + BHHH) and 3. Laplace components
        t1 = time.perf_counter()
        optima = [self.optimise(lik, u, s["opt_maxiter"], s["opt_tol"]) for u in starts]
        t_opt = time.perf_counter() - t1
        best = max(o["log_posterior"] for o in optima)
        comps: list[tuple[np.ndarray, np.ndarray, float]] = []
        for o in sorted(optima, key=lambda o: -o["log_posterior"]):
            if o["log_posterior"] < best - s["mode_tolerance"]:
                continue
            cov = _floored_inverse(o["curvature"], s["laplace_max_var"]) * s["inflate"] ** 2
            chol = np.linalg.cholesky(cov)
            if any(_mahalanobis(o["u"], m, c) < 2.0 for m, c, _ in comps):
                continue  # same mode as an earlier optimum
            # Laplace evidence: log p(mode) + log det(cov) / 2
            comps.append((o["u"], chol, o["log_posterior"] + np.log(np.diag(chol)).sum()))
        logpi = np.array([c[2] for c in comps])
        logpi -= logsumexp(logpi)
        means, chols = np.array([c[0] for c in comps]), np.array([c[1] for c in comps])
        if s["defensive_weight"] > 0:
            # defensive copies (Hesterberg 1995): the same modes, wider, bound the weights
            # where the Laplace approximation is too narrow (skewed / boundary-heavy tails)
            dw = s["defensive_weight"]
            logpi = np.r_[logpi + math.log1p(-dw), logpi + math.log(dw)]
            means = np.r_[means, means]
            chols = np.r_[chols, chols * s["defensive_scale"]]
        mix = MixtureT(means, chols, logpi, s["df"])
        LOG.info("%s: %d optimisations (%.0fs): log posteriors %s (%s); %d mode(s)", name,
                 len(optima), t_opt, [round(o["log_posterior"], 2) for o in optima],
                 [o["status"] for o in optima], len(comps))  # fmt: skip
        if s["sampler"] == "smc":
            ref = MixtureT(means, chols * s["start_inflate"], logpi, s["df"])
            return self._run_smc(lik, s, rng, name, ref, optima, t0, t_init, t_opt,
                                 float(lp0.max()))  # fmt: skip
        # 4. AMIS with mixture-of-t proposals (deterministic-mixture weights)
        proposals, X, LP, ess_trace, k_trace = [], [], [], [], []
        t2 = time.perf_counter()
        for it in range(s["max_iterations"]):
            x = np.clip(mix.sample(rng, s["particles"]), -30, 30)
            lp = lik.log_posterior_batch(x)
            proposals.append((mix, s["particles"]))
            X.append(x)
            LP.append(lp)
            allx, alllp = np.vstack(X), np.concatenate(LP)
            n_tot = len(allx)
            logq = logsumexp([math.log(n / n_tot) + q.logpdf(allx) for q, n in proposals], axis=0)
            logw = np.where(np.isfinite(alllp), alllp - logq, -np.inf)
            slogw, k_hat = psis(logw) if s["psis"] else (logw, pareto_k(logw))
            ess = kish_ess(slogw)
            ess_trace.append(ess)
            k_trace.append(k_hat)
            LOG.info("%s: AMIS iteration %d: %d particles, ESS %.1f (raw %.1f), Pareto k %.2f "
                     "(%.0fs)", name, it, n_tot, ess, kish_ess(logw), k_hat,
                     time.perf_counter() - t2)  # fmt: skip
            if (ess >= s["target_ess"] and k_hat <= 0.7) or it == s["max_iterations"] - 1:
                break
            w = np.exp(slogw - slogw.max())
            mix = mix.refit(allx, w / w.sum())
        w = np.exp(slogw - slogw.max())
        w /= w.sum()
        amis_seconds = time.perf_counter() - t2
        # 5. optional resample-move: MH rejuvenation of a weighted resample
        mh = {}
        if s["mh_steps"] > 0:
            allx, w, mh = self.rejuvenate(lik, allx, alllp, w, s["mh_steps"],
                                          s["particles"], rng, name)  # fmt: skip
        theta = lik.theta_from_u(allx)
        physical = lik.space.decode(theta)
        free = [i for i, n in enumerate(lik.space.names) if n in set(lik.space.free_names)]
        mean = w @ physical
        sd = np.sqrt(np.maximum(w @ (physical - mean) ** 2, 0))
        best_opt = max(optima, key=lambda o: o["log_posterior"])
        diag = {
            "sampler": "amis",
            "settings": s,
            "dt": lik.model.dt,
            "integrator": lik.model.integrator,
            "threads": lik.threads,
            "particles": int(n_tot),
            "iterations": len(ess_trace),
            "ess": float(ess),
            "ess_raw": kish_ess(logw),
            "pareto_k": float(k_hat),
            "ess_per_iteration": [float(e) for e in ess_trace],
            "pareto_k_per_iteration": [float(k) for k in k_trace],
            "finite_fraction": float(np.isfinite(alllp).mean()),
            "max_weight": float(w.max()),
            "rejuvenation": mh,
            "weighted_mean": {lik.space.names[i]: float(mean[i]) for i in free},
            "weighted_sd": {lik.space.names[i]: float(sd[i]) for i in free},
            "modes": len(comps),
            "final_components": len(mix.logpi),
            "optima_log_posterior": [o["log_posterior"] for o in optima],
            "optima_status": [o["status"] for o in optima],
            "optimisation_traces": [o["trace"] for o in optima],
            "mode_theta": lik.theta_from_u(best_opt["u"]).tolist(),
            "mode_log_posterior": best_opt["log_posterior"],
            "laplace_sd_u": np.sqrt(np.diag(
                _floored_inverse(best_opt["curvature"], s["laplace_max_var"]))).tolist(),
            "init_best_log_posterior": float(lp0.max()),
            "init_seconds": t_init,
            "optimisation_seconds": t_opt,
            "amis_seconds": amis_seconds,
            "seconds": time.perf_counter() - t0,
            "likelihood_evals": lik.evals,
            "gradient_evals": lik.grad_evals,
            "seconds_per_eval": lik.seconds_per_eval,
            "seconds_per_gradient": lik.grad_seconds / lik.grad_evals if lik.grad_evals else None,
            "grid": {"Lx": lik.grid.Lx, "Ly": lik.grid.Ly, "rx": lik.grid.rx, "ry": lik.grid.ry},
            "horizons": lik.horizons,
            "clones": [c.clones for c in lik.data],
        }  # fmt: skip
        return theta, w, diag

    # ---------------------------------------------------------------- pipeline API
    def fit_data(self, simulator, observation, cohorts, x_dim: int) -> WeightedPosterior:
        """Sample the exact posterior of the reference dataset ``cohorts``."""
        lik = self.likelihood(simulator, observation, cohorts)
        LOG.info("exact_bh grid %dx%d for horizons %s", lik.grid.Lx, lik.grid.Ly, lik.horizons)
        try:
            theta, w, diag = self.run(lik)
            median = _weighted_median(theta, w)
            diag["grid_check"] = lik.grid_check(median)
        finally:
            lik.close()
        for g in diag["grid_check"]:
            if g["max_rel_error"] > 1e-5:
                LOG.warning("exact_bh grid check (horizon %g): tail mass %.2g, max relative "
                            "error %.2g on observed cells; raise inference.method.grid.factor",
                            g["horizon"], g["tail_mass"], g["max_rel_error"])  # fmt: skip
        LOG.info("exact_bh (%s): %d particles in %.1fs (%d evaluations, %.2fs/eval, %s "
                 "s/gradient), ESS %.1f, Pareto k %.2f", diag["sampler"], diag["particles"],
                 diag["seconds"], diag["likelihood_evals"], diag["seconds_per_eval"],
                 diag["seconds_per_gradient"], diag["ess"], diag["pareto_k"])  # fmt: skip
        if diag["sampler"] == "smc":
            if not diag["reliable"]:
                m = diag["smc"]
                LOG.warning("exact_bh: SMC diagnostics fail (reached beta=1: %s, min ESS bulk "
                            "%.0f / tail %.0f, nested R-hat %.3f; status %s); raise "
                            "inference.method.max_evals", m["reached_posterior"],
                            m["min_ess_bulk"], m["min_ess_tail"], m["max_nested_rhat"],
                            m["status"])  # fmt: skip
        elif diag["pareto_k"] > 0.7 or diag["ess"] < 0.25 * self.settings["target_ess"]:
            LOG.warning("exact_bh: importance weights unreliable (ESS %.1f, Pareto k %.2f); "
                        "increase inference.method.particles / max_iterations",
                        diag["ess"], diag["pareto_k"])  # fmt: skip
        history = [{
            "round": 0, "simulations": 0, "valid_simulations": 0, "simulation_seconds": 0.0,
            "training_seconds": diag["seconds"], "epochs": 0,
            "best_validation_loss": float("nan"),
        }]  # fmt: skip
        return WeightedPosterior(
            posterior=None, feature_mask=np.ones(int(x_dim), dtype=bool), method=self.name,
            amortized=False, history=history, draws=theta, weights=w, diagnostics=diag,
        )  # fmt: skip

    def calibrate(self, simulator, observation, theta: np.ndarray, datasets: list,
                  posterior_samples: int) -> tuple[np.ndarray, dict]:  # fmt: skip
        """SBC ranks: rerun the sampler (``sbc`` settings) on every valid calibration dataset.

        Cases needing a grid above ``sbc.max_points`` are skipped (selection on the data
        keeps ranks uniform). Ranks count resampled draws below the truth.
        """
        opts = {"init_candidates": 8, "opt_starts": 1, "opt_maxiter": 15, "particles": 100,
                "max_iterations": 3, "chains": 32, "steps": 6, "max_evals": 4000,
                "target_ess": 100, **self.sbc}  # fmt: skip
        max_points = float(opts.pop("max_points", 1e5))
        info = {"invalid_cases": 0, "excluded_grid": 0, "seconds": [], "ess": [],
                "pareto_k": [], "settings": {**opts, "max_points": max_points}}  # fmt: skip
        n_draws = int(posterior_samples)
        ranks = []
        for i, (t, data) in enumerate(zip(theta, datasets, strict=True)):
            if data is None:
                info["invalid_cases"] += 1
                continue
            try:
                lik = self.likelihood(simulator, observation, data, max_points)
            except GridTooLarge:
                info["excluded_grid"] += 1
                continue
            try:
                th, w, diag = self.run(lik, key=("sbc", i), **opts)
            finally:
                lik.close()
            rng = rng_for(self.seed, "exact-bh-sbc", i)
            draws = th[rng.choice(len(th), n_draws, p=w)]
            ranks.append((draws < t[None, :]).sum(axis=0))
            info["seconds"].append(diag["seconds"])
            info["ess"].append(diag["ess"])
            info["pareto_k"].append(diag["pareto_k"])
            LOG.info("SBC case %d: %.1fs, ESS %.1f", i, diag["seconds"], diag["ess"])
        info["draws_per_case"] = n_draws
        return np.asarray(ranks, dtype=float).reshape(-1, theta.shape[1]), info


def _flat(items: list) -> list:
    """Flatten one level of nesting (per-island lists of stage records)."""
    out = []
    for x in items:
        out.extend(x if isinstance(x, list) else [x])
    return out


def _weighted_median(theta: np.ndarray, w: np.ndarray) -> np.ndarray:
    out = np.empty(theta.shape[1])
    for j in range(theta.shape[1]):
        o = np.argsort(theta[:, j])
        c = np.cumsum(w[o])
        out[j] = theta[o[np.searchsorted(c, 0.5 * c[-1])], j]
    return out


# ---------------------------------------------------------------------- posterior object
@dataclass
class WeightedPosterior(ExactPosterior):
    """Importance-weighted particles of the exact posterior for one dataset.

    ``sample(x, n)`` resamples ``n`` draws (multinomial, with replacement, random order)
    regardless of ``x`` (not amortised).
    """

    weights: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    def sample(self, x: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
        """``n`` weighted resamples of the particles (``theta`` coordinates)."""
        if self.weights is None:
            return super().sample(x, n, seed)
        rng = np.random.default_rng(seed)
        return self.draws[rng.choice(len(self.draws), int(n), p=self.weights)]

    def log_prob(self, theta, x):
        """Not available: the posterior is represented by weighted particles."""
        raise NotImplementedError("exact_bh posteriors are represented by weighted particles")
