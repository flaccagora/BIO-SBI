"""Point estimates for ``exact_gw``: particle swarm (PSO) + bounded L-BFGS-B refinement.

Objectives (``objective``), both on the exact observed-data log likelihood
``l(theta)`` of :class:`~bio_sbi.inference.exact_gw.ExactLikelihood` (same observation
model, detection conditioning, probability / detection floors and FFT grids as NUTS):

* ``mle``: maximise ``l(theta)``.
* ``map``: physical-probability MAP, maximise ``l(theta) + log pi(p)`` with the
  Dirichlet prior on the *normalised free probabilities* of every type row ``k``,

      log pi(p) = sum_k sum_{j in F_k} (alpha_kj - 1) log q_kj,   q_kj = p_kj / m_k,

  up to parameter-independent constants (``m_k = 1 - sum of the fixed entries``).
  Rows without free sticks (all entries fixed, or one free entry) contribute nothing.

Neither is the NUTS target :meth:`ExactLikelihood.log_posterior_u`: that is the density
of ``u = logit(theta)`` and includes the Jacobians of ``p -> theta -> u``, so its mode
moves with the coordinates. With the default ``alpha = 1`` the physical MAP equals the MLE.

Coordinates. The search runs in the existing stick-breaking coordinates
``theta in [lo, hi]^dim`` (``lo = boundary_eps``, ``hi = 1 - boundary_eps``), decoded by
:class:`~bio_sbi.simulators.law.ParameterSpace`, which keeps every row nonnegative with
its fixed entries and sum one by construction: no sum constraint or penalty on the sticks.
In sticks, ``q_i = s_i prod_{l<i} (1 - s_l)`` (last ``q = prod (1 - s_l)``), so

    log pi = sum_l (alpha_l - 1) log s_l + c_l log(1 - s_l),  c_l = sum_{i>l} (alpha_i - 1),

whose gradient is used by the refinement (no transformation Jacobian added).

Boundaries. Decoding and the likelihood accept exact stick endpoints (``0`` / ``1``:
outcome probabilities become exactly zero, which the pgf kernel skips; impossible
observed cells cost ``log floor``). So the MLE, and the MAP when every free
``alpha >= 1``, are searched on the closed box ``[0, 1]`` (``boundary_eps = 0``; for the
MAP, bounds where an ``alpha > 1`` term makes the prior density zero are kept
``ZERO_DENSITY_GAP = 1e-10`` inside, a numerical safeguard that cannot exclude the
maximum). With any
``alpha < 1`` the prior density is unbounded where that ``q -> 0``: the posterior density
may have no maximum. Then ``objective=map`` requires ``boundary_eps > 0``; the result is
labelled ``map_eps_restricted`` (maximum over the stick box ``[eps, 1 - eps]``, *not* an
unrestricted MAP) and validation re-optimises at ``eps / 10`` to show how much the value
depends on the restriction.

Algorithm (``method=pso``): ``restarts`` independent global-best swarms seeded from
:func:`~bio_sbi.rng.rng_for` and initialised from the stick prior (the Dirichlet rows);
objective-only evaluations (``log_likelihood(grad=False)``) in ``workers`` threads (the
Numba kernel releases the GIL, FFTs use one worker; results do not depend on the number
of workers; the likelihood keeps no mutable state besides its locked counters, so
concurrent calls are safe; memory per worker ~ ``16 Lx (Ly/2+1)`` bytes per horizon
for an objective-only call, ``(1 + dim)`` times that with a gradient; keep ``workers``
times the BLAS / Numba threads within the CPU budget). Then the ``refine_top`` best distinct personal bests (deduplicated on decoded
probabilities) are refined by bounded L-BFGS-B with analytic gradients; a refinement
never replaces its start by a worse or failed point. ``method=multistart`` is the
baseline: L-BFGS-B from ``starts`` prior draws, same objective and bounds.
PSO convergence is no proof of global optimality.

Algorithm (``method=sgd``, MLE or MAP, see :func:`sgd`): first-order ascent (Adam or
heavy-ball momentum) from the *same* ``starts`` prior draws as ``multistart`` (same RNG
stream), on the objective divided by the total clone count (per-clone mean, so learning
rates do not scale with the data size), with a learning-rate schedule, stagnation
stopping and an optional bounded L-BFGS-B polish of the ``refine_top`` best distinct
endpoints. Coordinates ``sgd_coords``: ``box`` = projected steps in the stick box (same
box as PSO / L-BFGS-B; boundary faces reachable exactly) or ``logit`` = unconstrained
``z = logit(theta)`` clipped to ``|z| <= sgd_logit_bound`` (the MLE is invariant to the
reparametrisation, so no Jacobian is added; boundary faces are only approached,
``theta >= expit(-bound)``). ``sgd_batch`` (fraction of clones per horizon) makes the
gradient a minibatch estimate: clones sampled without replacement, cell counts rescaled
by ``N_h / B_h`` (unbiased; the detection term is exact). Each minibatch step still
computes the full pgf / FFT law, so it costs as much as a full-batch step (see the
benchmark in ``scripts/pso_benchmark``); the best point is then tracked by full
objective-only evaluations every ``sgd_eval_every`` steps.

Output (:func:`optimise`): a JSON-serialisable dict (written to ``optimisation.json``);
swarm particles are *not* posterior draws and never go to ``ExactPosterior.draws``.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from ..rng import rng_for
from .gw_exact import law_and_jac

LOG = logging.getLogger(__name__)

OBJECTIVES = ("mle", "map")
METHODS = ("pso", "multistart", "sgd")
SGD_OPTIMIZERS = ("adam", "momentum")
SGD_COORDS = ("box", "logit")
SGD_SCHEDULES = ("constant", "cosine", "exp")
BOUNDARY_MODES = ("reflect", "clamp")

#: Distance kept from stick bounds where the MAP prior density vanishes (see PointObjective).
ZERO_DENSITY_GAP = 1e-10

#: Defaults of every setting (``inference.method.optimise`` in ``exact_gw.yaml``).
DEFAULTS = {
    "objective": "mle",
    "method": "pso",
    "boundary_eps": 0.0,
    "init": "prior",  # prior | uniform: initial particles / multistart starts
    # PSO
    "restarts": 2,
    "swarm": 24,
    "max_iter": 100,
    "max_evals": None,  # per restart (objective-only evaluations); None = swarm * (max_iter+1)
    "inertia": 0.7298,  # Clerc-Kennedy constriction coefficients
    "inertia_end": None,  # linear inertia schedule to this value (None = constant)
    "cognitive": 1.49618,
    "social": 1.49618,
    "velocity_max": 0.25,  # |v| <= velocity_max * (hi - lo) per coordinate
    "boundary": "reflect",  # reflect | clamp (see _repair)
    "patience": 20,  # stop when the best improved by < tol over this many iterations
    "tol": 1e-4,
    "xtol": 1e-6,  # ... or when the swarm has collapsed (max coordinate spread < xtol)
    # refinement / baseline
    "refine_top": 4,  # distinct PSO candidates refined by L-BFGS-B (0 = PSO alone)
    "refine_maxiter": 500,
    "starts": 8,  # method=multistart / sgd: L-BFGS-B / SGD starts (same prior draws)
    "dedup_tol": 1e-2,  # candidates closer than this (max |dp|) are duplicates
    # SGD family (method=sgd; refine_top / refine_maxiter apply to the polish)
    "sgd_optimizer": "adam",  # adam | momentum (heavy ball)
    "sgd_coords": "box",  # box (projected, sticks) | logit (unconstrained, clipped)
    "sgd_lr": 0.02,  # step size in sgd_coords (Adam: ~max move per step)
    "sgd_schedule": "cosine",  # constant | cosine | exp (to sgd_lr * sgd_lr_final)
    "sgd_lr_final": 0.01,
    "sgd_momentum": 0.9,  # momentum (momentum) / beta1 (adam)
    "sgd_beta2": 0.999,
    "sgd_iter": 500,  # steps per start
    "sgd_max_evals": None,  # gradient evaluations per start (None = sgd_iter)
    "sgd_batch": None,  # None / 1 = full batch; fraction of clones per horizon (minibatch)
    "sgd_eval_every": 10,  # minibatch: full objective-only evaluation every N steps
    "sgd_patience": 100,  # stop if the best improved < sgd_tol over this many steps
    "sgd_tol": 1e-3,
    "sgd_logit_bound": 15.0,  # sgd_coords=logit: |z| <= bound
    "sgd_polish": True,  # bounded L-BFGS-B on the refine_top best distinct endpoints
    # execution / reporting
    "workers": 1,
    "log_every": 10,
    "validate": True,
    "near_tol": 1e-6,  # boundary proximity threshold (sticks and probabilities)
    "validation_grids": [2, 4],  # re-evaluate on grids this many times larger
    "validation_max_points": 4e6,  # skip larger validation grids (memory)
}


# ---------------------------------------------------------------------- objective
class PointObjective:
    """MLE or physical-probability MAP objective in stick coordinates.

    Args:
        lik: The exact likelihood.
        objective: ``"mle"`` or ``"map"``.
        boundary_eps: Stick box ``[eps, 1 - eps]``; must be ``> 0`` for ``map`` when a
            free row has ``alpha < 1`` (unbounded prior density at the boundary).
    """

    def __init__(self, lik, objective: str = "mle", boundary_eps: float = 0.0) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}; got {objective!r}")
        eps = float(boundary_eps)
        if not 0.0 <= eps < 0.5:
            raise ValueError("boundary_eps must lie in [0, 0.5)")
        self.lik, self.objective, self.eps = lik, objective, eps
        self.dim = lik.dim
        self.n_clones = sum(h.clones for h in lik.data)  # gradient scale of method=sgd
        coef_s, coef_t, alpha_min = [], [], math.inf
        for r in lik.space.rows:
            if not r.n_sticks:  # deterministic row: q = 1 (or no free entry)
                continue
            am1 = np.asarray(r.alpha, dtype=float) - 1.0
            alpha_min = min(alpha_min, float(r.alpha.min()))
            tail = np.cumsum(am1[::-1])[::-1]  # tail[i] = sum_{j >= i} (alpha_j - 1)
            coef_s.extend(am1[:-1])
            coef_t.extend(tail[1:])
        self.coef_s = np.asarray(coef_s, dtype=float)
        self.coef_t = np.asarray(coef_t, dtype=float)
        self.alpha_min = alpha_min
        self.improper = objective == "map" and alpha_min < 1.0
        if self.improper and eps <= 0.0:
            raise ValueError(
                "objective=map with Dirichlet concentration < 1: the posterior density is "
                "unbounded at the boundary (no MAP); set boundary_eps > 0 to maximise over "
                "the stick box [eps, 1 - eps] (reported as map_eps_restricted)"
            )
        self.name = "map_eps_restricted" if self.improper else objective
        # Per-coordinate box. For map, a stick bound where the prior density is zero
        # (alpha > 1 terms: log prior = -inf) is moved inside by ZERO_DENSITY_GAP: the
        # maximum cannot lie there, and L-BFGS-B's projected steps onto such a bound would
        # otherwise abort the line search on a nonfinite value.
        self.lower = np.full(self.dim, eps)
        self.upper = np.full(self.dim, 1.0 - eps)
        if objective == "map":
            self.lower[self.coef_s > 0] = max(eps, ZERO_DENSITY_GAP)
            self.upper[self.coef_t > 0] = min(1.0 - eps, 1.0 - ZERO_DENSITY_GAP)

    def log_prior(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Dirichlet log density of the normalised free probabilities (+ const) and gradient.

        ``0 * log 0`` terms (``alpha = 1``) are zero; ``-inf`` where the density is zero.
        """
        s = np.asarray(theta, dtype=float)
        lp, g = 0.0, np.zeros_like(s)
        with np.errstate(divide="ignore", invalid="ignore"):
            for coef, x, sign in ((self.coef_s, s, 1.0), (self.coef_t, 1.0 - s, -1.0)):
                nz = coef != 0
                if np.any(nz):
                    lp += float(coef[nz] @ np.log(x[nz]))
                    g[nz] += sign * coef[nz] / x[nz]
        return lp, g

    def evaluate(self, theta: np.ndarray, grad: bool = False, weights=None) -> dict:
        """``value`` (``-inf`` if not finite), ``log_likelihood``, ``log_prior`` (+ ``grad``).

        ``weights``: minibatch cell weights per horizon (see
        :meth:`ExactLikelihood.log_likelihood`); the prior term is always exact.
        """
        theta = np.asarray(theta, dtype=float)
        if weights is None:
            ll, gl = self.lik.log_likelihood(theta, grad=grad)
        else:
            ll, gl = self.lik.log_likelihood(theta, grad=grad, weights=weights)
        lp, gp = self.log_prior(theta) if self.objective == "map" else (0.0, 0.0)
        value = ll + lp
        out = {"value": value if np.isfinite(value) else -math.inf, "log_likelihood": ll,
               "log_prior": lp if self.objective == "map" else None}  # fmt: skip
        if grad:
            out["grad"] = gl + gp
        return out

    def value(self, theta: np.ndarray) -> float:
        """Objective value (objective-only likelihood path; ``-inf`` if not finite)."""
        return self.evaluate(theta)["value"]

    def value_and_grad(self, theta: np.ndarray, weights=None) -> tuple[float, np.ndarray]:
        """Objective value and analytic gradient in stick coordinates (``weights``: minibatch)."""
        e = self.evaluate(theta, grad=True, weights=weights)
        return e["value"], e["grad"]


# ---------------------------------------------------------------------- helpers
def _map(f, X: np.ndarray, workers: int) -> np.ndarray:
    """``f`` on every row of ``X`` (threads; order preserved), nonfinite -> ``-inf``."""
    if workers > 1 and len(X) > 1:
        with ThreadPoolExecutor(min(workers, len(X))) as pool:
            vals = list(pool.map(f, X))
    else:
        vals = [f(x) for x in X]
    v = np.asarray(vals, dtype=float)
    return np.where(np.isfinite(v), v, -np.inf)


def _initial(obj: PointObjective, rng: np.random.Generator, n: int, init: str) -> np.ndarray:
    """``n`` starting points in the box: stick prior (Dirichlet rows) or uniform draws."""
    lik = obj.lik
    if init == "prior":
        X = rng.beta(lik.prior_a, lik.prior_b, size=(n, obj.dim))
    elif init == "uniform":
        X = rng.uniform(size=(n, obj.dim))
    else:
        raise ValueError(f"init must be prior or uniform; got {init!r}")
    return np.clip(X, obj.lower, obj.upper)


def _repair(x: np.ndarray, v: np.ndarray, lo: float, hi: float, mode: str):
    """Bring positions back into ``[lo, hi]`` and make velocities consistent.

    ``reflect``: mirror at the violated bound and reverse that velocity component (a
    step longer than the box width is clamped instead). ``clamp``: project onto the
    bound and zero that velocity component (absorbing walls).
    """
    out = (x < lo) | (x > hi)
    if not np.any(out):
        return x, v
    x, v = x.copy(), v.copy()
    if mode == "reflect":
        x = np.where(x < lo, 2 * lo - x, x)
        x = np.where(x > hi, 2 * hi - x, x)
        v = np.where(out, -v, v)
        x = np.clip(x, lo, hi)
    elif mode == "clamp":
        x = np.clip(x, lo, hi)
        v = np.where(out, 0.0, v)
    else:
        raise ValueError(f"boundary must be one of {BOUNDARY_MODES}; got {mode!r}")
    return x, v


# ---------------------------------------------------------------------- PSO
def pso(obj: PointObjective, rng: np.random.Generator, s: dict, name: str = "pso") -> dict:
    """One global-best particle swarm maximising ``obj`` over the stick box.

    Returns positions / values of the global and personal bests, the best value per
    iteration, objective evaluations (all attempted, and nonfinite ones), the stopping
    reason and wall time. Raises ``RuntimeError`` if no initial particle is finite.
    """
    t0 = time.perf_counter()
    lo, hi, d = obj.lower, obj.upper, obj.dim
    n, workers = int(s["swarm"]), max(int(s["workers"]), 1)
    max_iter = int(s["max_iter"])
    max_evals = int(s["max_evals"]) if s.get("max_evals") else n * (max_iter + 1)
    vmax = float(s["velocity_max"]) * (hi - lo)
    w0 = float(s["inertia"])
    w1 = w0 if s.get("inertia_end") is None else float(s["inertia_end"])
    c1, c2 = float(s["cognitive"]), float(s["social"])
    x = _initial(obj, rng, n, s["init"])
    v = np.clip(0.5 * (rng.uniform(lo, hi, size=(n, d)) - x), -vmax, vmax)
    f = _map(obj.value, x, workers)
    evals, invalid = n, int(np.sum(~np.isfinite(f)))
    if not np.any(np.isfinite(f)):
        raise RuntimeError(
            f"{name}: no initial particle has a finite objective ({n} tried); check the "
            "data / floors or widen the swarm"
        )
    pbest_x, pbest_f = x.copy(), f.copy()
    g = int(np.argmax(pbest_f))
    gbest_x, gbest_f = pbest_x[g].copy(), float(pbest_f[g])
    history, reason, it = [gbest_f], "max_iter", 0
    for it in range(1, max_iter + 1):
        if evals + n > max_evals:
            reason, it = "max_evals", it - 1
            break
        w = w0 + (w1 - w0) * (it - 1) / max(max_iter - 1, 1)
        r1, r2 = rng.uniform(size=(n, d)), rng.uniform(size=(n, d))
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x[None, :] - x)
        v = np.clip(v, -vmax, vmax)
        x, v = _repair(x + v, v, lo, hi, s["boundary"])
        f = _map(obj.value, x, workers)
        evals += n
        invalid += int(np.sum(~np.isfinite(f)))
        better = f > pbest_f  # -inf never improves
        pbest_x[better] = x[better]  # row copies (fancy indexing), no aliasing
        pbest_f[better] = f[better]
        g = int(np.argmax(pbest_f))
        if pbest_f[g] > gbest_f:
            gbest_x, gbest_f = pbest_x[g].copy(), float(pbest_f[g])
        history.append(gbest_f)
        if s["log_every"] and it % int(s["log_every"]) == 0:
            LOG.info("%s iteration %d: best %.4f (%d evals)", name, it, gbest_f, evals)
        p = int(s["patience"])
        if p and it >= p and history[-1] - history[-1 - p] < float(s["tol"]):
            reason = "stagnation"
            break
        if float(np.max(x.max(0) - x.min(0))) < float(s["xtol"]):
            reason = "collapsed"
            break
    return {
        "best_x": gbest_x, "best_value": gbest_f, "pbest_x": pbest_x, "pbest_f": pbest_f,
        "history": history, "iterations": it, "evals": evals, "invalid_evals": invalid,
        "stop_reason": reason, "seconds": time.perf_counter() - t0,
    }  # fmt: skip


# ---------------------------------------------------------------------- SGD family
def minibatch_weights(lik, rng: np.random.Generator, fraction: float, cells=None) -> list:
    """Rescaled cell counts of a minibatch of clones (per horizon, without replacement).

    ``B_h = max(1, round(fraction * N_h))`` clones of horizon ``h``; each sampled cell
    count is multiplied by ``N_h / B_h``, so the weights sum to ``N_h`` and the
    minibatch log likelihood and gradient are unbiased for the full ones.
    ``cells`` (per horizon: the cell index of every clone) can be precomputed.
    """
    if cells is None:
        cells = [np.repeat(np.arange(len(h.weight)), h.weight.astype(np.int64)) for h in lik.data]
    out = []
    for h, c in zip(lik.data, cells, strict=True):
        n = len(c)
        b = min(max(1, int(round(fraction * n))), n)
        pick = rng.choice(n, size=b, replace=False)
        out.append(np.bincount(c[pick], minlength=len(h.weight)) * (n / b))
    return out


def _lr(s: dict, it: int, n_iter: int) -> float:
    """Learning rate of step ``it`` (0-based) out of ``n_iter``."""
    lr, f = float(s["sgd_lr"]), float(s["sgd_lr_final"])
    t = it / max(n_iter - 1, 1)
    if s["sgd_schedule"] == "constant":
        return lr
    if s["sgd_schedule"] == "cosine":
        return lr * (f + (1 - f) * 0.5 * (1 + math.cos(math.pi * t)))
    if s["sgd_schedule"] == "exp":
        return lr * f**t
    raise ValueError(f"sgd_schedule must be one of {SGD_SCHEDULES}; got {s['sgd_schedule']!r}")


def sgd(obj: PointObjective, x0: np.ndarray, s: dict, rng: np.random.Generator | None = None,
        name: str = "sgd") -> dict:  # fmt: skip
    """First-order ascent of ``obj`` from ``x0`` (Adam or momentum; full or minibatch).

    The gradient is divided by ``obj.n_clones`` (per-clone mean objective). ``box``
    coordinates: ``x <- clip(x + step, lower, upper)`` (projected ascent; moments are
    kept). ``logit``: steps in ``z = logit(x)`` with ``dz = g * x (1 - x)``, ``z``
    clipped to the box mapped to logits and to ``|z| <= sgd_logit_bound``.

    Full batch: every step evaluates value and gradient at the iterate, so the best
    visited point is tracked exactly. Minibatch (``0 < sgd_batch < 1``, needs ``rng``):
    the gradient is a minibatch estimate and the best point is tracked by objective-only
    full evaluations every ``sgd_eval_every`` steps and at the end. A nonfinite value or
    gradient sends the iterate back to the best point, resets the moments and halves the
    learning rate. Returns the best (tracked) point, never a worse one, with its value,
    the start / final values, evaluations, the best value per step (``history``), the
    stopping reason and wall time.
    """
    t0 = time.perf_counter()
    lo, hi = np.asarray(obj.lower, float), np.asarray(obj.upper, float)
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    coords, opt = s["sgd_coords"], s["sgd_optimizer"]
    if coords not in SGD_COORDS:
        raise ValueError(f"sgd_coords must be one of {SGD_COORDS}; got {coords!r}")
    if opt not in SGD_OPTIMIZERS:
        raise ValueError(f"sgd_optimizer must be one of {SGD_OPTIMIZERS}; got {opt!r}")
    if coords == "logit":
        bound = float(s["sgd_logit_bound"])
        with np.errstate(divide="ignore"):
            zlo, zhi = np.maximum(logit(lo), -bound), np.minimum(logit(hi), bound)
        z = np.clip(logit(np.clip(x0, expit(-bound), expit(bound))), zlo, zhi)

        def to_x(z):
            return np.clip(expit(z), lo, hi)
    else:
        zlo, zhi, z = lo, hi, x0.copy()

        def to_x(z):
            return z
    batch = s.get("sgd_batch")
    if batch is not None and not 0.0 < float(batch) <= 1.0:
        raise ValueError(f"sgd_batch must be None or a fraction in (0, 1]; got {batch!r}")
    full = batch is None or float(batch) == 1.0
    if not full:
        if rng is None:
            raise ValueError("minibatch SGD needs an rng")
        lik = obj.lik
        cells = [np.repeat(np.arange(len(h.weight)), h.weight.astype(np.int64)) for h in lik.data]
    scale = 1.0 / max(float(getattr(obj, "n_clones", 1.0)), 1.0)
    n_iter = int(s["sgd_iter"])
    max_evals = int(s["sgd_max_evals"]) if s.get("sgd_max_evals") else n_iter
    every, patience, tol = int(s["sgd_eval_every"]), int(s["sgd_patience"]), float(s["sgd_tol"])
    b1, b2, eps = float(s["sgd_momentum"]), float(s["sgd_beta2"]), 1e-8
    m, v = np.zeros_like(z), np.zeros_like(z)
    best_x, best_f, best_z = x0.copy(), -math.inf, z.copy()
    start_value = None
    grad_evals = obj_evals = invalid = 0
    lr_mult, t_adam = 1.0, 0
    history: list[float] = []
    reason, it = "sgd_iter", 0

    def track(x, zz, f):
        nonlocal best_x, best_f, best_z, start_value
        if start_value is None:
            start_value = float(f)
        if np.isfinite(f) and f > best_f:
            best_x, best_f, best_z = x.copy(), float(f), zz.copy()

    for it in range(n_iter):
        if grad_evals >= max_evals:
            reason = "max_evals"
            break
        x = to_x(z)
        if full:
            f, g = obj.value_and_grad(x)
            grad_evals += 1
            track(x, z, f)
            ok = np.isfinite(f)
        else:
            if it % every == 0:
                f = obj.value(x)
                obj_evals += 1
                track(x, z, f)
            _, g = obj.value_and_grad(x, weights=minibatch_weights(lik, rng, float(batch), cells))
            grad_evals += 1
            ok = True
        g = np.asarray(g, dtype=float)
        if not ok or not np.all(np.isfinite(g)):
            invalid += 1
            z, m, v, t_adam = best_z.copy(), np.zeros_like(z), np.zeros_like(z), 0
            lr_mult *= 0.5
            history.append(best_f)
            continue
        g = g * scale
        if coords == "logit":
            g = g * x * (1.0 - x)
        lr = _lr(s, it, n_iter) * lr_mult
        if opt == "adam":
            t_adam += 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            step = lr * (m / (1 - b1**t_adam)) / (np.sqrt(v / (1 - b2**t_adam)) + eps)
        else:
            m = b1 * m + g
            step = lr * m
        z = np.clip(z + step, zlo, zhi)
        history.append(best_f)
        if patience and len(history) > patience and history[-1] - history[-1 - patience] < tol:
            reason = "stagnation"
            break
        if s.get("log_every") and (it + 1) % (10 * int(s["log_every"])) == 0:
            LOG.info("%s step %d: best %.4f", name, it + 1, best_f)
    x = to_x(z)  # the last update has not been evaluated yet
    final = obj.value(x)
    obj_evals += 1
    track(x, z, final)
    history.append(best_f)
    return {
        "best_x": best_x, "best_value": best_f, "start_value": start_value,
        "final_value": float(final), "history": history, "iterations": it + 1,
        "grad_evals": grad_evals, "obj_evals": obj_evals, "invalid_steps": invalid,
        "stop_reason": reason, "seconds": time.perf_counter() - t0,
    }  # fmt: skip


# ---------------------------------------------------------------------- L-BFGS-B
def lbfgsb(obj: PointObjective, x0: np.ndarray, maxiter: int, lower=None, upper=None) -> dict:
    """Bounded L-BFGS-B from ``x0`` with analytic gradients; keeps ``x0`` if not improved.

    Returns ``x``, ``value`` (the better of start and result), ``start_value``,
    ``refined_value`` (L-BFGS-B's own, may be worse / ``-inf``), ``improved``, status.
    """
    x0 = np.asarray(x0, dtype=float)
    lo = np.broadcast_to(obj.lower if lower is None else lower, x0.shape)
    hi = np.broadcast_to(obj.upper if upper is None else upper, x0.shape)
    x0 = np.clip(x0, lo, hi)
    start = obj.value(x0)
    t0 = time.perf_counter()

    def neg(x):
        val, g = obj.value_and_grad(x)
        if not np.isfinite(val) or not np.all(np.isfinite(g)):
            return 1e300, np.zeros_like(x)
        return -val, -g

    try:
        res = minimize(neg, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True)),
                       options={"maxiter": int(maxiter)})  # fmt: skip
        xr = np.clip(res.x, lo, hi)
        refined = obj.value(xr)  # objective-only re-evaluation at the returned point
        status = {"success": bool(res.success), "message": str(res.message),
                  "nit": int(res.nit), "nfev": int(res.nfev)}  # fmt: skip
    except Exception as err:  # noqa: BLE001 - a failed refinement must not lose the start
        xr, refined = x0, -math.inf
        status = {"success": False, "message": f"error: {err}", "nit": 0, "nfev": 0}
    improved = bool(np.isfinite(refined) and refined >= start)
    return {
        "x": xr if improved else x0, "value": refined if improved else start,
        "start_value": start, "refined_value": refined, "improved": improved,
        "seconds": time.perf_counter() - t0, **status,
    }  # fmt: skip


# ---------------------------------------------------------------------- candidates
def _dedup(obj: PointObjective, X: np.ndarray, f: np.ndarray, tol: float, k: int | None,
           first: list[int] | tuple = ()):  # fmt: skip
    """Return indices of the best distinct points (max |decoded p diff| > tol), best first.

    Indices in ``first`` (e.g. the best of every swarm) are considered before the rest,
    so candidates come from different swarms when they found different regions.
    """
    order = [i for i in np.argsort(-f, kind="stable") if np.isfinite(f[i])]
    order = [i for i in first if np.isfinite(f[i])] + [i for i in order if i not in set(first)]
    phys = obj.lik.space.decode(X)
    keep: list[int] = []
    for i in order:
        if all(np.max(np.abs(phys[i] - phys[j])) > tol for j in keep):
            keep.append(i)
            if k is not None and len(keep) >= k:
                break
    return keep


def _describe(obj: PointObjective, x: np.ndarray, near_tol: float) -> dict:
    """Describe one point: decoded probabilities, objective parts, boundary proximity."""
    space = obj.lik.space
    phys = space.decode(x)
    e = obj.evaluate(x)
    free = set(space.free_names)
    return {
        "theta": dict(zip(space.theta_names, map(float, x), strict=True)),
        "physical": dict(zip(space.names, map(float, phys), strict=True)),
        "value": float(e["value"]),
        "log_likelihood": float(e["log_likelihood"]),
        "log_prior": None if e["log_prior"] is None else float(e["log_prior"]),
        "boundary": {
            "sticks_at_lower": [
                n
                for n, t, b in zip(space.theta_names, x, obj.lower, strict=True)
                if t - b <= near_tol
            ],
            "sticks_at_upper": [
                n
                for n, t, b in zip(space.theta_names, x, obj.upper, strict=True)
                if b - t <= near_tol
            ],
            "probabilities_near_zero": [
                n for n, p in zip(space.names, phys, strict=True) if n in free and p <= near_tol
            ],
        },  # fmt: skip
    }


# ---------------------------------------------------------------------- validation
def cell_report(lik, theta: np.ndarray) -> dict:
    """Observed cells whose probability is floored and detection-floor activity at ``theta``."""
    from .gw_exact import detection_and_grad

    P = lik.probabilities(theta)
    empty_dP, empty_rows = np.zeros((0, P.shape[1])), np.zeros(0, dtype=np.int64)
    floored_cells = floored_clones = 0
    min_p, detection = math.inf, []
    for h in lik.data:
        p, _ = law_and_jac(lik.model, P, empty_dP, empty_rows, h.generations, h.grid)
        pc = p[h.a, h.b]
        low = pc <= lik.floor
        floored_cells += int(low.sum())
        floored_clones += int(h.weight[low].sum())
        min_p = min(min_p, float(pc.min()))
        if lik.model.detected:
            q, _ = detection_and_grad(lik.model, P, empty_dP, empty_rows, h.generations)
            detection.append(float(q))
    return {"floored_cells": floored_cells, "floored_clones": floored_clones,
            "min_cell_probability": min_p, "detection": detection,
            "detection_floored": any(q <= lik.min_detection for q in detection)}  # fmt: skip


def validate(obj: PointObjective, candidates: list[dict], s: dict) -> dict:
    """Numerical checks of the best and competitive candidates (see module docstring).

    * constraints: nonnegativity, row sums, fixed entries of the decoded probabilities;
    * grids ``validation_grids`` times larger (factor and margin scaled; same damping):
      objective of every candidate, change of ranking;
    * floors x1e-3 / x1e3 (probability and detection): objective changes;
    * floored observed cells and detection-floor activity at the best;
    * ``map_eps_restricted``: best re-refined on the box with ``eps / 10``;
    * the existing ``grid_check`` (diagnostic, its reference grid is itself finite).

    ``flags`` lists settings-dependent conclusions (empty when nothing was flagged).
    """
    lik, space = obj.lik, obj.lik.space
    X = np.array([[c["theta"][n] for n in space.theta_names] for c in candidates])
    X = X.reshape(len(candidates), obj.dim)
    base = np.array([c["value"] for c in candidates])
    out: dict = {"constraints": [], "grids": [], "floors": [], "flags": []}
    for c in candidates:
        phys = np.array([c["physical"][n] for n in space.names])
        rows, start, ok = {}, 0, True
        for r in space.rows:
            row = phys[start : start + len(r.entries)]
            start += len(r.entries)
            fixed_err = max((abs(row[i] - v) for i, v in r.fixed.items()), default=0.0)
            rows[r.prefix] = {"sum_error": float(abs(row.sum() - 1.0)),
                              "min": float(row.min()), "fixed_error": float(fixed_err)}  # fmt: skip
            ok &= rows[r.prefix]["sum_error"] < 1e-12 and row.min() >= 0 and fixed_err < 1e-12
        out["constraints"].append({"ok": bool(ok), "rows": rows})
    if not all(c["ok"] for c in out["constraints"]):
        out["flags"].append("constraint_violation")

    def scored(variant: PointObjective):
        return np.array([variant.value(x) for x in X])

    ranking = list(np.argsort(-base, kind="stable"))
    gap = float(base[ranking[0]] - base[ranking[1]]) if len(base) > 1 else math.inf
    max_change = 0.0
    for scale in s["validation_grids"]:
        g = {k: v for k, v in lik.grid_settings.items()}
        g["factor"] = g.get("factor", 1.25) * scale
        g["margin"] = int(g.get("margin", 16) * scale)
        big = lik.with_settings(grid=g)
        points = max(h.grid.Lx * h.grid.Ly for h in big.data)
        if points > float(s["validation_max_points"]):
            out["grids"].append({"scale": scale, "skipped": f"{points} grid points"})
            continue
        vals = scored(PointObjective(big, obj.objective, obj.eps))
        change = vals - base
        max_change = max(max_change, float(np.max(np.abs(change))))
        new_rank = list(np.argsort(-vals, kind="stable"))
        out["grids"].append({
            "scale": scale, "grids": [(h.grid.Lx, h.grid.Ly) for h in big.data],
            "values": vals.tolist(), "max_abs_change": float(np.max(np.abs(change))),
            "best_unchanged": bool(new_rank[0] == ranking[0]),
            "ranking_unchanged": bool(new_rank == ranking),
        })  # fmt: skip
        if new_rank[0] != ranking[0]:
            out["flags"].append(f"best_changes_with_grid_x{scale}")
    for mult in (1e-3, 1e3):
        variant = lik.with_settings(floor=lik.floor * mult, min_detection=lik.min_detection * mult)
        vals = scored(PointObjective(variant, obj.objective, obj.eps))
        change = vals - base
        max_change = max(max_change, float(np.max(np.abs(change))))
        new_rank = list(np.argsort(-vals, kind="stable"))
        out["floors"].append({"multiplier": mult, "values": vals.tolist(),
                              "max_abs_change": float(np.max(np.abs(change))),
                              "best_unchanged": bool(new_rank[0] == ranking[0])})  # fmt: skip
        if new_rank[0] != ranking[0]:
            out["flags"].append(f"best_changes_with_floor_x{mult:g}")
    out["best_runner_up_gap"] = gap
    out["max_numerical_change"] = max_change
    if gap < 10 * max_change:
        out["flags"].append("best_vs_runner_up_within_10x_numerical_change")
    best = X[ranking[0]]
    out["cells"] = cell_report(lik, best)
    if out["cells"]["floored_cells"]:
        out["flags"].append("floored_observed_cells_at_best")
    if out["cells"]["detection_floored"]:
        out["flags"].append("detection_floor_active_at_best")
    out["grid_check"] = lik.grid_check(best)
    if obj.eps > 0:
        tighter = PointObjective(lik, obj.objective, obj.eps / 10)
        r = lbfgsb(tighter, best, int(s["refine_maxiter"]))
        moved = float(np.max(np.abs(space.decode(r["x"]) - space.decode(best))))
        out["epsilon"] = {"eps": obj.eps, "eps_tighter": obj.eps / 10,
                          "value_tighter": float(r["value"]),
                          "value_change": float(r["value"] - base[ranking[0]]),
                          "max_probability_change": moved}  # fmt: skip
        if r["value"] - base[ranking[0]] > 1e-3 or moved > 1e-3:
            out["flags"].append("solution_depends_on_boundary_eps")
    return out


# ---------------------------------------------------------------------- driver
def settings(overrides: dict | None = None) -> dict:
    """Defaults updated with ``overrides`` (unknown keys raise ``ValueError``)."""
    overrides = dict(overrides or {})
    overrides.pop("enabled", None)
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown optimisation settings {sorted(unknown)}")
    s = {**DEFAULTS, **overrides}
    if s["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got {s['method']!r}")
    if s["boundary"] not in BOUNDARY_MODES:
        raise ValueError(f"boundary must be one of {BOUNDARY_MODES}; got {s['boundary']!r}")
    for key, allowed in (("sgd_optimizer", SGD_OPTIMIZERS), ("sgd_coords", SGD_COORDS),
                         ("sgd_schedule", SGD_SCHEDULES)):  # fmt: skip
        if s[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}; got {s[key]!r}")
    s["validation_grids"] = list(s["validation_grids"] or [])
    return s


def _sgd_candidates(obj, s, seed, key, workers, out) -> list[dict]:
    """``method=sgd``: SGD from the multistart prior draws, then optional polish."""
    # same RNG stream as multistart: start i is the same point for both methods
    X0 = _initial(obj, rng_for(seed, "multistart", *key), int(s["starts"]), s["init"])

    def run(i):
        return sgd(obj, X0[i], s, rng_for(seed, "sgd-batch", *key, i), name=f"sgd start {i}")

    lik = obj.lik
    ev0, gr0 = lik.evals, lik.grad_evals
    t_sgd = time.perf_counter()
    with ThreadPoolExecutor(min(workers, len(X0))) as pool:
        runs = list(pool.map(run, range(len(X0))))
    out["sgd_seconds"] = time.perf_counter() - t_sgd
    out["evaluations_before_polish"] = {
        "objective_only": lik.evals - ev0 - (lik.grad_evals - gr0),
        "gradient": lik.grad_evals - gr0,
    }
    out["sgd"] = [{k: v for k, v in r.items() if k != "best_x"} | {"best_theta": r["best_x"].tolist()}
                  for r in runs]  # fmt: skip
    f = np.array([r["best_value"] for r in runs])
    if not np.any(np.isfinite(f)):
        raise RuntimeError("sgd: no start reached a finite objective")
    out["sgd_best_value"] = float(np.max(f))
    X = np.array([r["best_x"] for r in runs]).reshape(len(runs), obj.dim)
    refs: dict[int, dict] = {}
    t_ref = time.perf_counter()
    if s["sgd_polish"] and int(s["refine_top"]) > 0:
        polish = _dedup(obj, X, f, float(s["dedup_tol"]), int(s["refine_top"]))
        with ThreadPoolExecutor(min(workers, len(polish))) as pool:
            refs = dict(zip(polish, pool.map(lambda i: lbfgsb(obj, X[i], s["refine_maxiter"]),
                                             polish), strict=True))  # fmt: skip
    out["refine_seconds"] = time.perf_counter() - t_ref
    candidates = []
    for i, r in enumerate(runs):
        ref = refs.get(i)
        c = _describe(obj, X[i] if ref is None else ref["x"], s["near_tol"])
        gap = np.minimum(X[i] - obj.lower, obj.upper - X[i])
        c.update(source=f"sgd{i}", start_value=float(r["best_value"]),
                 sgd_min_stick_gap=float(gap.min()) if len(gap) else None,
                 refinement=None if ref is None else {k: v for k, v in ref.items() if k != "x"})
        candidates.append(c)
    return candidates


def optimise(lik, seed: int = 0, key=("reference",), **overrides) -> dict:
    """Point estimate of ``lik`` (MLE or physical MAP): PSO + L-BFGS-B, multistart or SGD.

    Returns a JSON-serialisable dict: ``objective`` (name), ``value``, ``best`` (stick
    coordinates, decoded probabilities, log likelihood and log prior separately,
    boundary proximity), ``candidates`` (every refined / baseline solution with pre- and
    post-refinement values and L-BFGS-B status), ``pso`` (per restart: best per
    iteration, evaluations, stopping reason, seconds), ``sgd`` (method=sgd, per start:
    best value per step, start / final value, evaluations, stopping reason; plus
    ``sgd_best_value`` and ``evaluations_before_polish``), ``evaluations`` (objective-only
    and gradient likelihood calls), ``seconds``, ``settings``, ``seed`` and, with
    ``validate``, ``validation`` (see :func:`validate`).
    """
    s = settings(overrides)
    obj = PointObjective(lik, s["objective"], s["boundary_eps"])
    t0 = time.perf_counter()
    ev0, gr0 = lik.evals, lik.grad_evals
    out: dict = {"objective": obj.name, "method": s["method"], "seed": int(seed),
                 "key": [str(k) for k in key], "settings": s,
                 "box": [obj.eps, 1.0 - obj.eps], "dim": obj.dim,
                 "bounds": {"lower": obj.lower.tolist(), "upper": obj.upper.tolist()},
                 "note": "point estimate; not posterior draws, no global-optimality guarantee"}  # fmt: skip
    workers = max(int(s["workers"]), 1)
    if obj.dim == 0:  # every probability fixed: one admissible model
        best = _describe(obj, np.zeros(0), s["near_tol"])
        if not np.isfinite(best["value"]):
            raise RuntimeError("The fully fixed model has a nonfinite objective")
        candidates = [{**best, "source": "fixed", "start_value": best["value"],
                       "refinement": None}]  # fmt: skip
        out["pso"] = []
    elif s["method"] == "pso":
        runs = []
        for r in range(int(s["restarts"])):
            rng = rng_for(seed, "pso", *key, r)
            runs.append(pso(obj, rng, {**s, "workers": workers}, name=f"pso restart {r}"))
            LOG.info("PSO restart %d: best %.4f after %d iterations (%s, %d evals, %.1fs)", r,
                     runs[-1]["best_value"], runs[-1]["iterations"], runs[-1]["stop_reason"],
                     runs[-1]["evals"], runs[-1]["seconds"])  # fmt: skip
        out["pso"] = [{k: v for k, v in run.items() if k not in ("pbest_x", "pbest_f", "best_x")}
                      | {"best_theta": run["best_x"].tolist()} for run in runs]  # fmt: skip
        X = np.concatenate([run["pbest_x"] for run in runs])
        f = np.concatenate([run["pbest_f"] for run in runs])
        src = [f"pso{r}" for r, run in enumerate(runs) for _ in run["pbest_f"]]
        k = int(s["refine_top"]) or 4  # PSO alone: report the 4 best distinct
        offsets = np.cumsum([0] + [len(run["pbest_f"]) for run in runs])
        bests = sorted((int(o + np.argmax(run["pbest_f"])) for o, run in zip(offsets[:-1], runs, strict=True)),
                       key=lambda i: -f[i])  # fmt: skip
        idx = _dedup(obj, X, f, float(s["dedup_tol"]), k, first=bests)
        if not idx:
            raise RuntimeError("PSO found no finite candidate")
        t_ref = time.perf_counter()
        if int(s["refine_top"]) > 0:
            with ThreadPoolExecutor(min(workers, len(idx))) as pool:
                refs = list(pool.map(lambda i: lbfgsb(obj, X[i], s["refine_maxiter"]), idx))
        else:
            refs = [None] * len(idx)
        out["refine_seconds"] = time.perf_counter() - t_ref
        candidates = []
        for i, ref in zip(idx, refs, strict=True):
            x = X[i] if ref is None else ref["x"]
            c = _describe(obj, x, s["near_tol"])
            c.update(source=src[i], start_value=float(f[i]),
                     refinement=None if ref is None else
                     {k2: v for k2, v in ref.items() if k2 not in ("x",)})  # fmt: skip
            candidates.append(c)
    elif s["method"] == "sgd":
        candidates = _sgd_candidates(obj, s, seed, key, workers, out)
    else:
        rng = rng_for(seed, "multistart", *key)
        X0 = _initial(obj, rng, int(s["starts"]), s["init"])
        with ThreadPoolExecutor(min(workers, len(X0))) as pool:
            refs = list(pool.map(lambda x: lbfgsb(obj, x, s["refine_maxiter"]), X0))
        if not any(np.isfinite(r["value"]) for r in refs):
            raise RuntimeError("multistart: no start or optimum has a finite objective")
        candidates = []
        for i, ref in enumerate(refs):
            c = _describe(obj, ref["x"], s["near_tol"])
            c.update(source=f"start{i}", start_value=float(ref["start_value"]),
                     refinement={k2: v for k2, v in ref.items() if k2 != "x"})  # fmt: skip
            candidates.append(c)
    candidates.sort(key=lambda c: -c["value"] if np.isfinite(c["value"]) else math.inf)
    # distinct solutions (refinements often converge to the same optimum)
    Xc = np.array([[c["theta"][n] for n in lik.space.theta_names] for c in candidates])
    fc = np.array([c["value"] for c in candidates])
    distinct = _dedup(obj, Xc.reshape(len(candidates), obj.dim), fc, float(s["dedup_tol"]), None)
    for i, c in enumerate(candidates):
        c["distinct"] = i in distinct
    if not np.isfinite(candidates[0]["value"]):
        raise RuntimeError("No feasible candidate with a finite objective")
    out["evaluations"] = {"objective_only": lik.evals - ev0 - (lik.grad_evals - gr0),
                          "gradient": lik.grad_evals - gr0}  # fmt: skip
    out["seconds"] = time.perf_counter() - t0
    out["best"] = candidates[0]
    out["value"] = candidates[0]["value"]
    out["candidates"] = candidates
    if s["validate"]:
        t_val = time.perf_counter()
        competitive = [c for c in candidates if c["distinct"]][:5]
        out["validation"] = validate(obj, competitive, s)
        out["validation"]["seconds"] = time.perf_counter() - t_val
        for flag in out["validation"]["flags"]:
            LOG.warning("exact_gw optimisation: %s", flag)
    return out
